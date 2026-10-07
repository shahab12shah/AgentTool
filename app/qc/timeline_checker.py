"""TimelineChecker (spec 7): the structural integrity of the whole timeline, read from the detached QC snapshot.

It REUSES ``TimelineValidator`` and ``PresentationValidator``: their ``ValidationIssue`` codes are mapped to QC codes and severities, never re-implemented. What neither
knows is added here: gaps in the picture (intentional vs unintended), clips outside / beyond the narration, exact duplicates, orphaned references, invalid transforms /
opacity / levels, and cross-track overlaps that are not a deliberate layer.

Severity is CRITICAL exactly where ``RenderDiagnostics.validate_timeline`` would refuse to start a render, so the QC gate and the export preflight never disagree.
Findings from the validators and from the checks below are merged per (code, clip) before issues are built, so one broken clip is one issue, not one per rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.core.constants import MIN_CLIP_DURATION
from app.editing.assembly import subtract
from app.editing.validator import TimelineValidator, ValidationIssue
from app.media.asset import AssetType
from app.presentation.assembly import PresState
from app.presentation.validator import MAX_VOLUME, PresentationValidator
from app.qc import fix_catalog
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import VISUAL_TRACK_KINDS, ProgressFn, QCContext
from app.qc.issue_model import QCCategory, QCFixSpec, QCIssue
from app.qc.severity import Severity, worse
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.timeline_commands import SCALE_RANGE
from app.timeline.track import Track, TrackKind
from app.transcription.sentences import PAUSE_SECONDS

C, E, W, N = Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.NOTICE
EPS = 0.05  # the validators' tolerance for source ranges
GAP_ERROR_SECONDS = 1.0  # an unintended picture gap longer than this is an ERROR (optional setting coverage.gap_error_seconds)
BEYOND_ERROR_SECONDS = 1.0  # a picture clip running this far past the narration is an ERROR
KEYFRAME_RANGES = {"opacity": (0.0, 1.0), "reveal": (0.0, 1.0), "volume": (0.0, MAX_VOLUME), "scale": (SCALE_RANGE[0], SCALE_RANGE[1])}

# validator code -> (QC code, severity when the validator reports an *error*). A validator "warning" (an element the user owns) stays a WARNING, except where the render
# itself would fail. scene.coverage and visual.overlap are deliberately absent: gaps and layer-aware overlaps are judged below, for every scene and every visual track.
TV_CODES: dict[str, tuple[str, Severity]] = {
    "track.duplicate": ("timeline.duplicate.track", E), "clip.duplicate": ("timeline.duplicate.id", E), "clip.track": ("timeline.orphan.track", E),
    "clip.nonfinite": ("timeline.clip.nonfinite", C), "clip.start": ("timeline.clip.out_of_bounds", C), "clip.duration": ("timeline.clip.zero_duration", C),
    "clip.overlap": ("timeline.overlap.same_track", E), "clip.speed": ("timeline.clip.speed", C), "clip.asset": ("timeline.orphan.asset", C),
    "clip.track_kind": ("timeline.clip.track_kind", E), "clip.source": ("timeline.clip.source_range", E), "clip.source_len": ("timeline.clip.source_range", E),
    "clip.text": ("timeline.clip.empty_content", E), "keyframe": ("timeline.keyframe.invalid", C), "transition": ("timeline.transition.invalid", E),
    "voice.missing": ("timeline.empty.voice", C), "voice.alignment": ("timeline.voice.misaligned", E), "decision.target": ("timeline.orphan.decision", W),
}
# The structural half of PresentationValidator. caption.range|words|line|lines|size|safe_area|position, graphic.position|size|counter and audio.* belong to the caption,
# text and audio checkers, which map them themselves (mapping them here too would count one problem twice).
PV_CODES: dict[str, tuple[str, Severity]] = {
    "track.duplicate": TV_CODES["track.duplicate"], "clip.duplicate": TV_CODES["clip.duplicate"], "clip.track": TV_CODES["clip.track"], "clip.nonfinite": TV_CODES["clip.nonfinite"],
    "clip.start": TV_CODES["clip.start"], "clip.duration": TV_CODES["clip.duration"], "clip.overlap": TV_CODES["clip.overlap"], "clip.asset": TV_CODES["clip.asset"],
    "clip.scene": ("timeline.orphan.scene", W), "keyframe": TV_CODES["keyframe"], "animation": ("timeline.animation.invalid", C), "graphic.text": TV_CODES["clip.text"],
    "graphic.region": ("timeline.effect.invalid", E), "caption.text": ("timeline.caption.invalid", E), "decision.target": TV_CODES["decision.target"],
    "decision.scene": ("timeline.orphan.decision", W),
}

# QC code -> (title, why it matters, suggested fix, viewer impact 0..1)
INFO: dict[str, tuple[str, str, str, float]] = {
    "timeline.empty.visual": ("No picture on the timeline", "Without a visual clip there is nothing to show over the narration.", "Add visuals to the timeline (run the AI edit or place media on a video track).", 1.0),
    "timeline.empty.voice": ("No voice-over on the timeline", "The voice-over is the master clock: scenes, captions and audio have nothing to follow without it.", "Put the voice-over clip back on the audio track at 0:00.", 1.0),
    "timeline.duration.invalid": ("The timeline has no duration", "A timeline of zero length cannot be rendered.", "Add media to the timeline.", 1.0),
    "timeline.clip.nonfinite": ("A clip has a non-numeric time value", "The render cannot place a clip whose start, duration or speed is not a number.", "Delete the clip and add it again.", 0.9),
    "timeline.clip.zero_duration": ("Clip with no usable length", f"The render rejects clips shorter than {MIN_CLIP_DURATION:g} s, and such a clip shows nothing.", "Remove the empty item.", 0.2),
    "timeline.clip.empty_content": ("Empty timeline item", "An item with no media or text draws nothing but still occupies the timeline.", "Remove the empty item.", 0.3),
    "timeline.clip.speed": ("Invalid clip speed", "A speed of zero or below cannot be played.", "Set a speed above zero.", 0.8),
    "timeline.clip.out_of_bounds": ("Clip outside the video", "Anything before 0:00 or after the narration ends is never seen, and a negative start stops the render.", "Move the clip inside the video or delete it.", 0.5),
    "timeline.clip.beyond_narration": ("Clip runs past the narration", "The video would be longer than the voice-over, leaving a held picture or silence at the end.", "Trim the clip so it ends with the narration.", 0.5),
    "timeline.clip.track_kind": ("Media on the wrong kind of track", "Audio on a picture track (or the reverse) is skipped or breaks the render.", "Move the clip to a track of the right kind.", 0.7),
    "timeline.clip.source_range": ("Impossible source range", "The clip asks for footage the media does not contain, so the render would freeze or fail.", "Re-trim the clip within its media.", 0.7),
    "timeline.overlap.same_track": ("Clips overlap on one track", "Two clips on the same track compete for the same moment; one of them is hidden or cut off.", "Move or trim one of the clips.", 0.6),
    "timeline.overlap.cross_track": ("Unintended overlap between picture tracks", "A picture clip that covers another without being a layer hides footage you placed.", "Trim one of the clips, or make the upper one a deliberate overlay.", 0.5),
    "timeline.gap.unintended": ("Gap in the picture", "The screen is empty while the narration is playing.", "Extend the previous clip or add a visual here.", 0.7),
    "timeline.duplicate.track": ("Duplicate track id", "Two tracks share one id, so edits and the render cannot tell them apart.", "Remove or recreate one of the tracks.", 0.4),
    "timeline.duplicate.id": ("Duplicate clip id", "Commands and QC fixes address clips by id; a repeated id makes them act on the wrong clip.", "Delete and re-add one of the clips.", 0.4),
    "timeline.duplicate.element": ("Duplicate timeline element", "The same element placed twice at the same time adds nothing (audio duplicates play twice as loud).", "Delete the duplicate.", 0.4),
    "timeline.orphan.asset": ("Clip refers to a missing asset", "The render has no media for this clip and stops.", "Relink the asset or remove the clip.", 0.9),
    "timeline.orphan.scene": ("Reference to a scene that does not exist", "Regeneration and QC cannot place this element in the story.", "Open the element and assign it to an existing scene.", 0.2),
    "timeline.orphan.decision": ("Reference to a missing editing decision", "Regeneration cannot tell why this element exists, so it may be duplicated or dropped.", "Review the element; it is safe to keep as a manual edit.", 0.1),
    "timeline.orphan.track": ("Clip refers to a track that does not match", "A clip whose track reference is wrong is edited and rendered on the wrong layer, or not at all.", "Move the clip to an existing track.", 0.6),
    "timeline.transform.invalid": ("Invalid position, scale or rotation", "A scale of zero or less, or a value that is not a number, cannot be drawn.", "Reset the transform.", 0.7),
    "timeline.opacity.invalid": ("Opacity outside 0..1", "The render refuses opacity outside 0..1.", "Clamp the opacity to a valid value.", 0.7),
    "timeline.audio_level.invalid": ("Invalid audio level", f"Volume must be a number between 0 and {MAX_VOLUME:g}; anything else is clipped, silent or breaks the mix.", "Clamp the volume to a valid value.", 0.6),
    "timeline.keyframe.invalid": ("Invalid keyframe", "A keyframe outside its clip, or with an unknown property or non-numeric value, stops the render.", "Move the keyframe inside the clip or delete it.", 0.8),
    "timeline.keyframe.value": ("Keyframe value out of range", "The animation would push opacity, scale or volume outside what can be drawn or played.", "Clamp the keyframe value.", 0.5),
    "timeline.animation.invalid": ("Invalid animation", "An animation that does not fit its clip, or has an unknown preset, stops the render.", "Shorten or remove the animation.", 0.7),
    "timeline.transition.invalid": ("Broken transition", "A transition that is negative, unknown or longer than its clip cannot be rendered as designed.", "Shorten the transition to fit the clip.", 0.5),
    "timeline.effect.invalid": ("Invalid effect parameter", "A crop, focus or highlight region outside the frame (or an unknown fit mode) is ignored or breaks the effect.", "Correct the effect settings.", 0.4),
    "timeline.caption.invalid": ("Caption without usable content", "A caption with no text or word timing cannot be shown or synchronised.", "Delete the caption or regenerate captions.", 0.5),
    "timeline.voice.misaligned": ("Voice-over no longer aligned with the scenes", "Scenes, captions and music follow the voice-over; if it moves everything drifts.", "Put the voice-over back at 0:00 and play it in full.", 0.9),
}


@dataclass
class _F:
    """One finding before it becomes an issue (validator results and own checks merge here)."""

    code: str
    sev: Severity
    clip: Clip | None = None
    track: Track | None = None
    messages: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    scene_id: str | None = None
    fix: QCFixSpec | None = None
    fix_blocked: str = ""
    confidence: float = 100.0
    impact: float | None = None
    current: str = ""
    recommended: str = ""
    signature: str = ""
    affected: list[str] = field(default_factory=list)
    changes: dict[str, Any] = field(default_factory=dict)  # param.normalize payload, built up by the transform / opacity / level / keyframe checks


class _Findings:
    def __init__(self) -> None:
        self.items: dict[tuple, _F] = {}

    def add(self, code: str, sev: Severity, *, clip: Clip | None = None, track: Track | None = None, msg: str = "", key: str = "", **kw: Any) -> _F:
        k = (code, clip.id if clip is not None else "", track.id if (track is not None and clip is None) else "", key)
        f = self.items.get(k)
        new = f is None
        if f is None:
            f = self.items[k] = _F(code, sev, clip, track)
        f.sev = worse(f.sev, sev)
        if msg and msg not in f.messages:
            f.messages.append(msg)
        for name, val in kw.items():
            if name == "changes":
                f.changes.update(val)
            elif name == "affected":
                f.affected.extend(v for v in val if v not in f.affected)
            elif name == "confidence":
                f.confidence = float(val) if new else min(f.confidence, float(val))
            elif val not in (None, "") and (new or not getattr(f, name)):  # the first finding's details win; later ones only fill what is still empty
                setattr(f, name, val)
        return f


# ---------------------------------------------------------------------------------------------- small helpers
def _t(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def _finite(*vals: Any) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def _below(v: Any, limit: float) -> bool:
    return _finite(v) and float(v) < limit


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _is_voice(c: Clip) -> bool:
    return str(c.audio.get("role", "")).upper() == "VOICE" or c.slot == "voice"


def _on_screen(t: Track, c: Clip) -> bool:
    """A media clip the viewer can actually see: on a visible picture track, with a real length and not fully transparent."""
    if t.kind not in VISUAL_TRACK_KINDS or t.hidden or c.kind != KIND_MEDIA or not _finite(c.timeline_start, c.duration) or c.duration <= 0:
        return False
    return not (_finite(c.opacity) and c.opacity <= 0.01 and not any(k.property == "opacity" for k in c.keyframes))


def _speech_spans(ctx: QCContext) -> list[tuple[float, float]]:
    """When the narration is active: word spans, bridged across pauses shorter than the sentence-split pause. Longer pauses are deliberate and are not "narration".
    Without a transcript the whole voice-over counts as active."""
    def calc() -> list[tuple[float, float]]:
        words = sorted((w for w in ctx.words if _finite(w.start, w.end)), key=lambda w: w.start)
        if not words:
            return [(0.0, ctx.duration)]
        spans: list[tuple[float, float]] = []
        for w in words:
            b, e = w.start, max(w.end, w.start)
            if spans and b - spans[-1][1] <= PAUSE_SECONDS:
                spans[-1] = (spans[-1][0], max(spans[-1][1], e))
            else:
                spans.append((b, e))
        return spans

    return ctx.memo("timeline.speech", calc)


def _overlap_seconds(span: tuple[float, float], others: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(span[1], b) - max(span[0], a)) for a, b in others)


def _region_problem(v: Any) -> str:
    """Same validity rule as the renderer's crop: [x, y, w, h] normalised, inside the frame."""
    try:
        x, y, w, h = (float(n) for n in v)
    except (TypeError, ValueError):
        return "is not four numbers"
    if not all(math.isfinite(n) for n in (x, y, w, h)) or not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1) or x + w > 1.0001 or y + h > 1.0001:
        return "lies outside the frame"
    return ""


def _clamped(v: Any, lo: float, hi: float, default: float) -> float:
    return default if not _finite(v) else min(hi, max(lo, float(v)))


def _is_layer(ctx: QCContext, up: Clip, lo: Clip, o0: float, o1: float) -> bool:
    """True when the upper picture clip is a deliberate layer over the lower one: a fade / dissolve layer, a picture-in-picture inset, a flagged overlay,
    a cross-dissolve, or a B-roll cutaway that sits entirely inside the clip it covers."""
    fr = ctx.frame
    w, h = ctx.canvas
    if _below(up.opacity, 0.95) or any(k.property == "opacity" for k in up.keyframes):
        return True
    if _below(up.scale, 0.9) or (_finite(*up.position) and (abs(up.position[0]) > 0.02 * w or abs(up.position[1]) > 0.02 * h)):
        return True
    layer = str(up.metadata.get("layer", "")).lower()
    if any(k in up.effects for k in ("overlay", "pip")) or layer in ("overlay", "pip", "broll", "b-roll") or up.slot.lower().startswith(("overlay", "pip", "broll")):
        return True
    tr = up.transition or {}
    if _finite(tr.get("duration", 0.0)) and float(tr.get("duration", 0.0)) > 0 and (o1 - o0) <= float(tr["duration"]) + 2 * fr:
        return True
    return up.timeline_start >= lo.timeline_start - fr and up.timeline_end <= lo.timeline_end + fr


def _normalize_summary(changes: dict[str, Any]) -> str:
    parts = []
    for k, v in changes.items():
        if k == "keyframes":
            parts.append(f"clamp {len(v)} keyframe value(s)")
        elif k == "audio.volume":
            parts.append(f"set the volume to {v:g}")
        else:
            parts.append(f"set {k} to {v:g}")
    return ("; ".join(parts) or "Bring the value inside its valid range").capitalize()


# ---------------------------------------------------------------------------------------------- the checker
class TimelineChecker(BaseChecker):
    id = "timeline"
    label = "Timeline Integrity"
    categories = (QCCategory.TIMELINE,)
    # transcript: the narration-activity test (pauses are not gaps) reads the words; audio: the tolerance for "beyond the narration"
    domains = ("timeline", "scenes", "assets", "transcript")
    settings_sections = ("intentional_gaps", "audio", "coverage", "fix_permissions")
    scene_local = False
    expensive = False
    version = "1"

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        fs = _Findings()
        tl = ctx.timeline
        index: dict[str, tuple[Track, Clip]] = {}
        for t in tl.tracks:
            for c in t.clips:
                index.setdefault(c.id, (t, c))
        n_clips = len(index)

        report(0.03, "Checking the timeline structure")
        self._structure(ctx, fs)
        report(0.12, "Checking every clip")
        narr_end = ctx.voice_duration or (max((s.end for s in ctx.project.scenes), default=0.0) or None)
        tol = max(ctx.frame, float(getattr(ctx.settings.audio, "voice_duration_tolerance", 0.5)))
        track_ids = {t.id for t in tl.tracks}
        for t in tl.tracks:
            ctx.check_cancel()
            if t.kind is TrackKind.AUDIO and not (_finite(t.volume) and 0.0 <= t.volume <= MAX_VOLUME):
                fs.add("timeline.audio_level.invalid", E, track=t, msg=f"Track {t.name} has the volume {t.volume!r}; it must be between 0 and {MAX_VOLUME:g}.",
                       current=f"{t.volume!r}", recommended=f"0 to {MAX_VOLUME:g}", signature="track")
            for c in t.clips:
                self._clip(ctx, fs, t, c, narr_end, tol, track_ids)
        report(0.40, "Looking for duplicate elements")
        dup_of = self._duplicates(ctx, fs)
        report(0.50, "Running the timeline and presentation validators")
        self._validators(ctx, fs, index, out)
        ctx.check_cancel()
        report(0.70, "Checking overlaps between picture tracks")
        self._cross_track(ctx, fs)
        report(0.82, "Looking for gaps in the picture")
        gap_metrics = self._gaps(ctx, fs)
        report(0.92, "Building the report")

        issues = []
        for f in sorted(fs.items.values(), key=lambda f: (f.start if f.start is not None else (f.clip.timeline_start if f.clip else 0.0), f.code)):
            iss = self._build(ctx, f, index, dup_of)
            if iss is not None:
                issues.append(iss)
        out.issues = issues
        out.metrics = self._metrics(ctx, n_clips, gap_metrics)
        out.notes.append(f"checked {n_clips} clips on {len(tl.tracks)} tracks")
        report(1.0, f"{len(issues)} timeline issue(s)")
        return out

    # ------------------------------------------------------------------ whole-timeline conditions
    def _structure(self, ctx: QCContext, fs: _Findings) -> None:
        if not ctx.visual_clips():
            fs.add("timeline.empty.visual", C, msg="There is no media clip on any video or image track.")
        voice = [c for _t, c in ctx.clips(kind=KIND_MEDIA, track_kinds=(TrackKind.AUDIO,)) if _is_voice(c) or (ctx.project.voice_over.asset_id and c.asset_id == ctx.project.voice_over.asset_id)]
        if not voice:
            fs.add("timeline.empty.voice", C, msg="There is no voice-over clip on any audio track.")
        ends = [c.timeline_end for c in ctx.timeline.all_clips() if _finite(c.timeline_start, c.duration)]
        total = max(ends, default=0.0)
        if total <= 1e-6:
            fs.add("timeline.duration.invalid", C, msg=f"The timeline lasts {total:.2f} s.", current=f"{total:.2f} s", recommended="more than 0 s")

    # ------------------------------------------------------------------ one clip
    def _clip(self, ctx: QCContext, fs: _Findings, t: Track, c: Clip, narr_end: float | None, tol: float, track_ids: set[str]) -> None:
        p = ctx.project
        kw: dict[str, Any] = {"clip": c, "track": t}
        timed = _finite(c.timeline_start, c.duration)
        if not timed:
            fs.add("timeline.clip.nonfinite", C, msg="The start or duration is not a number.", **kw)
        else:
            if c.duration < MIN_CLIP_DURATION - 1e-9:
                fs.add("timeline.clip.zero_duration", C, msg=f"The clip lasts {c.duration:.3f} s; the render needs at least {MIN_CLIP_DURATION:g} s.",
                       current=f"{c.duration:.3f} s", recommended=f"at least {MIN_CLIP_DURATION:g} s or remove it", **kw)
            if c.timeline_start < -1e-6:
                fs.add("timeline.clip.out_of_bounds", C, msg=f"The clip starts {abs(c.timeline_start):.2f} s before 0:00.", current=f"starts at {c.timeline_start:.2f} s", recommended="start at 0:00 or later", **kw)
            elif narr_end is not None and c.kind != KIND_CAPTION and c.duration > 0 and not _is_voice(c):  # captions: the caption checker owns "past the voice-over"
                audio = t.kind is TrackKind.AUDIO
                if c.timeline_start >= narr_end - 1e-6:
                    fs.add("timeline.clip.out_of_bounds", W if audio else E, msg=f"The clip starts at {_t(c.timeline_start)}, after the narration ends at {_t(narr_end)}; it is never seen or heard.",
                           current=f"starts at {_t(c.timeline_start)}", recommended=f"start before {_t(narr_end)}", **kw)
                elif c.timeline_end > narr_end + tol:
                    excess = c.timeline_end - narr_end
                    sev = N if audio else (E if t.kind in VISUAL_TRACK_KINDS and excess > BEYOND_ERROR_SECONDS else W)
                    fs.add("timeline.clip.beyond_narration", sev, msg=f"The clip runs {excess:.2f} s past the end of the narration ({_t(narr_end)}).", current=f"ends at {_t(c.timeline_end)}",
                           recommended=f"end by {_t(narr_end)}", confidence=75.0 if audio else 100.0, **kw)  # a music tail past the voice can be a deliberate outro
        # contentless items
        if c.kind == KIND_MEDIA and not c.asset_id:
            fs.add("timeline.clip.empty_content", E, msg="No media is attached to this clip.", **kw)
        elif c.kind == KIND_TEXT and not (isinstance(c.text, dict) and str(c.text.get("content", "")).strip()):
            fs.add("timeline.clip.empty_content", E, msg="The text element is empty.", **kw)
        # transforms
        problems, changes = [], {}
        if not _finite(c.scale) or c.scale <= 0:
            problems.append(f"the scale is {c.scale!r}")
            changes["scale"] = 1.0
        elif c.scale < SCALE_RANGE[0] or c.scale > SCALE_RANGE[1]:
            problems.append(f"the scale {c.scale:g} is outside {SCALE_RANGE[0]:g}-{SCALE_RANGE[1]:g}")
            changes["scale"] = _clamped(c.scale, *SCALE_RANGE, 1.0)
        if not _finite(*c.position):
            problems.append(f"the position {tuple(c.position)!r} is not a pair of numbers")
        if not _finite(c.rotation):
            problems.append(f"the rotation {c.rotation!r} is not a number")
        if problems:
            hard = not _finite(c.scale) or c.scale <= 0 or not _finite(*c.position) or not _finite(c.rotation)
            fs.add("timeline.transform.invalid", C if hard else E, msg="Invalid transform: " + "; ".join(problems) + ".", changes=changes, signature="|".join(sorted(changes) or ["position"]),
                   current="; ".join(problems), recommended=f"scale {SCALE_RANGE[0]:g}-{SCALE_RANGE[1]:g}, finite position and rotation", **kw)
        if not _finite(c.opacity) or not 0.0 <= c.opacity <= 1.0:
            fs.add("timeline.opacity.invalid", C, msg=f"The opacity is {c.opacity!r}; it must be between 0 and 1.", changes={"opacity": _clamped(c.opacity, 0.0, 1.0, 1.0)},
                   current=f"{c.opacity!r}", recommended="0 to 1", **kw)
        vol = c.audio.get("volume") if isinstance(c.audio, dict) else None
        if vol is not None and (not _finite(vol) or not 0.0 <= float(vol) <= MAX_VOLUME):
            fs.add("timeline.audio_level.invalid", E, msg=f"The volume is {vol!r}; it must be between 0 and {MAX_VOLUME:g}.", changes={"audio.volume": _clamped(vol, 0.0, MAX_VOLUME, 1.0)},
                   current=f"{vol!r}", recommended=f"0 to {MAX_VOLUME:g}", **kw)
        # keyframes: only the VALUE range here; time / property / non-finite problems come from the validators (timeline.keyframe.invalid)
        fixes = []
        for k in c.keyframes:
            lo_hi = KEYFRAME_RANGES.get(k.property)
            if lo_hi is None or not _finite(k.value, k.time):
                continue
            bad = (k.value <= 0 or k.value > lo_hi[1]) if k.property == "scale" else not lo_hi[0] <= k.value <= lo_hi[1]
            if bad:
                fixes.append({"property": k.property, "time": round(k.time, 3), "value": round(_clamped(k.value, *lo_hi, 1.0), 4)})
        if fixes:
            fs.add("timeline.keyframe.value", E, msg=f"{len(fixes)} keyframe value(s) are outside the valid range for {', '.join(sorted({f['property'] for f in fixes}))}.", changes={"keyframes": fixes},
                   current=f"{len(fixes)} out-of-range keyframe(s)", recommended="values inside the valid range", signature="|".join(sorted({f["property"] for f in fixes})), **kw)
        self._effects(fs, c, kw)
        # references
        if c.kind == KIND_CAPTION:
            d = c.text
            if not isinstance(d, dict):
                fs.add("timeline.caption.invalid", E, msg="The caption has no content record.", **kw)
            elif d.get("scene_id") and d["scene_id"] not in {s.id for s in p.scenes}:
                fs.add("timeline.orphan.scene", W, msg=f"The caption refers to the unknown scene {d['scene_id']}.", **kw)
        if c.ai_decision_id and c.ai_decision_id not in p.editing_decisions and c.ai_decision_id not in p.presentation_decisions:
            fs.add("timeline.orphan.decision", W, msg=f"The clip refers to the decision {c.ai_decision_id}, which no longer exists.", **kw)
        if c.track_id not in track_ids:
            fs.add("timeline.orphan.track", E, msg=f"The clip refers to the track {c.track_id}, which does not exist.", **kw)

    def _effects(self, fs: _Findings, c: Clip, kw: dict[str, Any]) -> None:
        if not c.effects:
            return
        bad = []
        if not isinstance(c.effects, dict):
            fs.add("timeline.effect.invalid", E, msg="The effects record is not a dictionary.", **kw)
            return
        if "fit" in c.effects and c.effects["fit"] not in ("cover", "contain"):
            bad.append(f"the fit mode {c.effects['fit']!r} is unknown")
        for key in ("focus_region", "crop"):
            if key in c.effects and (why := _region_problem(c.effects[key])):
                bad.append(f"the {key.replace('_', ' ')} {why}")
        for key in ("highlight", "evidence"):
            sub = c.effects.get(key)
            if isinstance(sub, dict) and sub.get("region") is not None and (why := _region_problem(sub["region"])):
                bad.append(f"the {key} region {why}")
        if bad:
            fs.add("timeline.effect.invalid", E, msg="Invalid effect: " + "; ".join(bad) + ".", signature="|".join(bad), **kw)

    # ------------------------------------------------------------------ duplicates
    def _duplicates(self, ctx: QCContext, fs: _Findings) -> dict[str, set[str]]:
        """Exact duplicates: the same element (asset / text) at the same time on the same track. Returns victim id -> ids of its twins, so the overlap that duplicates
        necessarily cause is not reported a second time."""
        tol = ctx.frame / 2
        dup_of: dict[str, set[str]] = {}
        for t in ctx.timeline.tracks:
            groups: dict[tuple, list[Clip]] = {}
            for c in t.clips:
                if not _finite(c.timeline_start, c.duration, c.source_in) or (c.kind == KIND_MEDIA and not c.asset_id):
                    continue
                body = str((c.text or {}).get("content" if c.kind == KIND_TEXT else "text", "")) if c.kind in (KIND_TEXT, KIND_CAPTION) else repr(c.effects.get("highlight")) if c.kind == KIND_GRAPHIC else ""
                groups.setdefault((c.kind, c.asset_id, body, round(c.source_in, 2)), []).append(c)
            for g in groups.values():
                g.sort(key=lambda c: (c.timeline_start, c.id))
                kept: list[Clip] = []
                for c in g:
                    twin = next((k for k in kept if k.id != c.id and abs(k.timeline_start - c.timeline_start) <= tol and abs(k.duration - c.duration) <= tol), None)
                    if twin is None:
                        kept.append(c)
                        continue
                    victim, keeper = (twin, c) if (ctx.is_protected(t, c)[0] and not ctx.is_protected(t, twin)[0]) else (c, twin)  # delete the copy the user does not own
                    dup_of.setdefault(victim.id, set()).add(keeper.id)
                    audio = t.kind is TrackKind.AUDIO
                    fs.add("timeline.duplicate.element", E if audio else W, clip=victim, track=t, msg=f"The same {victim.kind} element is placed twice at {_t(victim.timeline_start)} on {t.name}.",
                           current=f"2 identical clips at {_t(victim.timeline_start)}", recommended="1 clip", signature=f"{keeper.id}", affected=[self._name(ctx, victim), self._name(ctx, keeper)],
                           fix=fix_catalog.clip_delete(victim.id, "Delete the duplicate copy", ctx.settings))
        return dup_of

    # ------------------------------------------------------------------ the reused validators
    def _validators(self, ctx: QCContext, fs: _Findings, index: dict[str, tuple[Track, Clip]], out: CheckerOutput) -> None:
        p = ctx.project
        for label, table, run in (
            ("TimelineValidator", TV_CODES, lambda: TimelineValidator(p.assets, p.scenes, p.voice_over.asset_id, p.editing_decisions, p.editing_strategy, p.timeline_generation).validate(p.timeline)),
            ("PresentationValidator", PV_CODES, lambda: PresentationValidator(p, PresState.capture(p)).validate()),
        ):
            try:
                found = run()
            except Exception as exc:  # noqa: BLE001 - a corrupt project must not hide the checks that still work
                out.notes.append(f"{label} could not run ({type(exc).__name__}: {exc})")
                out.complete = False
                continue
            for vi in found:
                self._map(fs, index, vi, table)

    @staticmethod
    def _map(fs: _Findings, index: dict[str, tuple[Track, Clip]], vi: ValidationIssue, table: dict[str, tuple[str, Severity]]) -> None:
        spec = table.get(vi.code)
        if spec is None:
            return
        code, err_sev = spec
        track, clip = index.get(vi.clip_id, (None, None)) if vi.clip_id else (None, None)
        if code == "timeline.orphan.asset" and clip is not None and not clip.asset_id:
            code, err_sev = "timeline.clip.empty_content", E  # no asset id at all is an empty item, not a dangling reference
        sev = err_sev if (vi.severity == "error" or err_sev is C) else W
        fs.add(code, sev, clip=clip, track=track, msg=vi.message, scene_id=vi.scene_id or None, key=vi.message if (clip is None and code != "timeline.empty.voice") else "")

    # ------------------------------------------------------------------ cross-track overlaps
    def _cross_track(self, ctx: QCContext, fs: _Findings) -> None:
        tracks = ctx.timeline.tracks
        vis = sorted(((ti, t, c) for ti, t in enumerate(tracks) for c in t.clips if _on_screen(t, c)), key=lambda x: x[2].timeline_start)
        fr = ctx.frame
        for i, (ti, ta, a) in enumerate(vis):
            for tj, tb, b in vis[i + 1:]:
                if b.timeline_start >= a.timeline_end - fr:
                    break  # sorted by start: nothing later overlaps `a` by a full frame
                if ta is tb:
                    continue  # same-track overlap is the validators' (timeline.overlap.same_track)
                o0, o1 = max(a.timeline_start, b.timeline_start), min(a.timeline_end, b.timeline_end)
                if o1 - o0 < fr - 1e-6:
                    continue
                (lo, up, tup) = (a, b, tb) if ti < tj else (b, a, ta)  # later tracks are drawn on top
                if _is_layer(ctx, up, lo, o0, o1):
                    continue
                hidden = lo.timeline_start >= up.timeline_start - fr and lo.timeline_end <= up.timeline_end + fr
                user = str(a.created_by).upper() == "USER" or str(b.created_by).upper() == "USER"
                fs.add("timeline.overlap.cross_track", W if user else E, clip=up, track=tup, key=lo.id, start=o0, end=o1, confidence=85.0,
                       msg=f"{self._name(ctx, up)} covers {self._name(ctx, lo)} for {o1 - o0:.2f} s ({_t(o0)}-{_t(o1)})" + (": the lower clip is hidden completely." if hidden else "."),
                       current=f"{o1 - o0:.2f} s of overlap", recommended="no overlap, or a deliberate overlay (opacity, inset)", signature=f"{lo.id}:{round(o0)}",
                       affected=[self._name(ctx, lo), self._name(ctx, up)], impact=0.6 if hidden else 0.45)

    # ------------------------------------------------------------------ gaps
    def _gaps(self, ctx: QCContext, fs: _Findings) -> dict[str, float]:
        horizon = ctx.duration
        stats = {"gap_count": 0.0, "gap_seconds": 0.0, "intentional_gap_seconds": 0.0, "pause_gap_seconds": 0.0, "covered_seconds": 0.0}
        if horizon <= 0 or not ctx.visual_clips():
            return stats  # an empty timeline is the CRITICAL timeline.empty.visual, not one giant gap
        covered = _merge([(max(0.0, c.timeline_start), min(horizon, c.timeline_end)) for t, c in ctx.visual_clips() if _on_screen(t, c) and c.timeline_end > 0 and c.timeline_start < horizon])
        stats["covered_seconds"] = sum(b - a for a, b in covered)
        declared = [(float(g[0]), float(g[1])) for g in ctx.settings.intentional_gaps if len(g) >= 2]
        blank = subtract((0.0, horizon), covered)
        pieces = subtract((0.0, horizon), covered + declared)
        stats["intentional_gap_seconds"] = max(0.0, sum(b - a for a, b in blank) - sum(b - a for a, b in pieces))
        speech = _speech_spans(ctx)
        error_after = float(getattr(ctx.settings.coverage, "gap_error_seconds", GAP_ERROR_SECONDS))
        for b, e in pieces:
            length = e - b
            if length < ctx.frame - 1e-6:
                continue  # under one frame: the render snaps it away
            if _overlap_seconds((b, e), speech) < ctx.frame - 1e-6:
                stats["pause_gap_seconds"] += length  # blank only while nobody is speaking: a deliberate pause, not an error
                continue
            stats["gap_count"] += 1
            stats["gap_seconds"] += length
            fix, blocked, prev, prev_track = self._gap_fix(ctx, b, e)
            scene = ctx.scene_at((b + e) / 2)
            fs.add("timeline.gap.unintended", E if length > error_after else W, clip=prev if fix else None, track=prev_track if fix else None, key=f"{b:.3f}", start=b, end=e,
                   scene_id=scene.id if scene else None, fix=fix, fix_blocked=blocked, signature=f"{round(b * 2) / 2}-{round(e * 2) / 2}", impact=min(1.0, 0.5 + 0.2 * length),
                   msg=f"No picture for {length:.2f} s ({_t(b)}-{_t(e)}) while the narration is playing.", current=f"{length:.2f} s without a visual",
                   recommended="a visual on screen for the whole narration (or declare the gap intentional)", affected=[self._name(ctx, prev)] if prev else [])
        return stats

    def _gap_fix(self, ctx: QCContext, b: float, e: float) -> tuple[QCFixSpec | None, str, Clip | None, Track | None]:
        """Extend the clip that ends where the gap starts, but only as far as its source media and the free space allow (the engine re-checks everything)."""
        cands = [(t, c) for t, c in ctx.visual_clips() if _on_screen(t, c) and abs(c.timeline_end - b) <= 1.5 * ctx.frame]
        if not cands:
            return None, "", None, None
        tracks = ctx.timeline.tracks
        t, c = min(cands, key=lambda tc: (tracks.index(tc[0]), -tc[1].timeline_end))
        need = e - c.timeline_end
        asset = ctx.asset(c.asset_id)
        if asset is None:
            return None, "The previous clip's media is missing, so it cannot be extended", c, t
        if asset.type is AssetType.VIDEO and asset.duration and c.source_out + need * max(c.speed, 1e-6) > asset.duration + EPS:
            return None, f"The previous clip's media is too short to extend by {need:.2f} s", c, t
        if any(o is not c and c.timeline_end - 1e-6 <= o.timeline_start < e - 1e-6 for o in t.clips):
            return None, "Another clip on the track is in the way", c, t
        return fix_catalog.gap_close(c.id, e, ctx.settings), "", c, t

    # ------------------------------------------------------------------ output
    @staticmethod
    def _name(ctx: QCContext, c: Clip | None) -> str:
        if c is None:
            return ""
        a = ctx.asset(c.asset_id)
        label = a.name if a else (str((c.text or {}).get("content") or (c.text or {}).get("text") or "")[:30] if c.text else c.kind)
        return f"{label or c.kind} ({c.id[-6:]})"

    def _build(self, ctx: QCContext, f: _F, index: dict[str, tuple[Track, Clip]], dup_of: dict[str, set[str]]) -> QCIssue | None:
        title, why, suggested, impact = INFO[f.code]
        start, end, fix = f.start, f.end, f.fix
        if f.code == "timeline.overlap.same_track" and f.clip is not None and f.track is not None:
            partners = [o for o in f.track.clips if o is not f.clip and _finite(o.timeline_start, o.duration) and (o.timeline_start, o.id) <= (f.clip.timeline_start, f.clip.id)
                        and o.timeline_end > f.clip.timeline_start + 1e-4]  # the clips before it that it runs into (the validators flag the later clip)
            if partners and f.clip.id in dup_of and {o.id for o in partners} <= dup_of[f.clip.id]:
                return None  # the overlap of an exact duplicate: reported once, as timeline.duplicate.element
            if partners:
                start, end = f.clip.timeline_start, min(f.clip.timeline_end, max(o.timeline_end for o in partners))
                f.messages.append(f"It overlaps {', '.join(self._name(ctx, o) for o in partners[:3])} by {end - start:.2f} s.")
                f.affected = [self._name(ctx, o) for o in partners[:3]] + [self._name(ctx, f.clip)]
        if f.code in ("timeline.clip.zero_duration", "timeline.clip.empty_content") and f.clip is not None and fix is None:
            fix = fix_catalog.clip_remove_empty(f.clip.id, ctx.settings)
        if f.changes and f.clip is not None and fix is None:
            fix = fix_catalog.param_normalize(f.clip.id, f.changes, _normalize_summary(f.changes), ctx.settings)
        scene_id = f.scene_id or (ctx.clip_scene_id(f.clip) if f.clip is not None else None)
        description = " ".join(f.messages) or title
        sig = f.signature or ("|".join(sorted({m for m in f.messages})) if f.clip is None else f.code)
        iss = self.issue(f.code, QCCategory.TIMELINE, f.sev, title, description=description, scene_id=scene_id, clip=f.clip, track=f.track, start=start, end=end, confidence=f.confidence,
                         affected=f.affected, why=why, current=f.current, recommended=f.recommended, suggested_fix=suggested, fix=fix, fix_blocked=f.fix_blocked,
                         viewer_impact=f.impact if f.impact is not None else impact, metrics={"signature": sig}, signature=sig, ctx=ctx)
        if iss.scene_id and ctx.scene(iss.scene_id) is None:
            iss.scene_id = None  # an orphaned scene reference must not become a scene-scoped issue the UI cannot open
        return iss

    def _metrics(self, ctx: QCContext, n_clips: int, gaps: dict[str, float]) -> dict[str, Any]:
        by_kind = {k.value: 0 for k in TrackKind}
        for t in ctx.timeline.tracks:
            by_kind[t.kind.value] += len(t.clips)
        ends = [c.timeline_end for c in ctx.timeline.all_clips() if _finite(c.timeline_start, c.duration)]
        horizon = ctx.duration
        speech = _speech_spans(ctx)
        speech_total = sum(b - a for a, b in speech)
        covered = gaps.get("covered_seconds", 0.0)
        visual = _merge([(max(0.0, c.timeline_start), min(horizon, c.timeline_end)) for t, c in ctx.visual_clips() if _on_screen(t, c) and c.timeline_end > 0 and c.timeline_start < horizon])
        spoken_covered = sum(_overlap_seconds(s, visual) for s in speech)
        return {
            "clip_count": n_clips, "clips_by_track_kind": by_kind, "total_duration": round(max(ends, default=0.0), 3), "narration_duration": round(float(ctx.voice_duration or 0.0), 3),
            "visual_covered_ratio": round(covered / horizon, 4) if horizon > 0 else 0.0,
            "visual_covered_ratio_while_speaking": round(spoken_covered / speech_total, 4) if speech_total > 0 else 1.0,
            "gap_count": int(gaps.get("gap_count", 0)), "gap_seconds": round(gaps.get("gap_seconds", 0.0), 3), "intentional_gap_seconds": round(gaps.get("intentional_gap_seconds", 0.0), 3),
            "pause_gap_seconds": round(gaps.get("pause_gap_seconds", 0.0), 3),
        }
