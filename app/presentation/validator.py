"""PresentationValidator: nothing replaces the current valid presentation unless it passes.

Captions, audio, graphics and the timeline references are checked on a *candidate* state. Problems in AI-owned objects are errors
(the candidate is rejected); the same problems in objects the user owns are warnings (the user may have meant it).
"""

from __future__ import annotations

import math

from app.captions.engine import layout_for
from app.captions.styles import effective_style, style_for
from app.editing.validator import ValidationIssue
from app.media.asset import AssetType
from app.presentation import animation
from app.presentation.assembly import PresState, owned
from app.presentation.models import SfxCategory
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT
from app.timeline.track import TrackKind

EPS = 0.05
MIN_FONT_REL = 0.03
MAX_VOLUME = 4.0
VALID_SFX = {c.value for c in SfxCategory}


class PresentationValidator:
    def __init__(self, project, state: PresState) -> None:
        self.p, self.st = project, state
        self.scene_ids = {s.id for s in project.scenes}

    def validate(self) -> list[ValidationIssue]:
        out: list[ValidationIssue] = []
        tl = self.st.timeline

        def add(c, code, msg, strict=True):
            sev = "error" if (strict and not (c is not None and owned(c))) else "warning"
            out.append(ValidationIssue(sev, code, msg, c.scene_id if c else "", c.id if c else ""))

        track_ids: set[str] = set()
        clip_ids: set[str] = set()
        voice_dur = self._voice_duration()
        for t in tl.tracks:
            if t.id in track_ids:
                out.append(ValidationIssue("error", "track.duplicate", f"Duplicate track id {t.id}."))
            track_ids.add(t.id)
            prev = None
            for c in sorted(t.clips, key=lambda x: x.timeline_start):
                if c.id in clip_ids:
                    add(c, "clip.duplicate", f"Duplicate clip id {c.id}.")
                clip_ids.add(c.id)
                if c.track_id != t.id:
                    add(c, "clip.track", f"Clip {c.id} is stored on {t.id} but claims {c.track_id}.")
                if not all(math.isfinite(v) for v in (c.timeline_start, c.duration, c.source_in, c.source_out, c.speed)):
                    add(c, "clip.nonfinite", "A clip has a non-numeric time value.")
                    continue
                if c.timeline_start < -1e-6:
                    add(c, "clip.start", f"A clip starts before 0:00 ({c.id}).")
                if c.duration <= 0:
                    add(c, "clip.duration", f"A clip has a zero or negative duration ({c.id}).")
                if prev is not None and c.timeline_start < prev.timeline_end - 1e-4:
                    add(c, "clip.overlap", f"Clips overlap on track “{t.name}”.")
                prev = c if prev is None or c.timeline_end > prev.timeline_end else prev
                if c.scene_id and c.scene_id not in self.scene_ids:
                    add(c, "clip.scene", f"Clip {c.id} refers to the unknown scene {c.scene_id}.", strict=False)
                for k in c.keyframes:
                    for pr in k.problems(c.duration):
                        add(c, "keyframe", pr)
                for pr in animation.problems(c.animation, c.duration) if c.animation else []:
                    add(c, "animation", pr)
                if c.kind == KIND_CAPTION:
                    self._caption(c, t, voice_dur, add)
                elif c.kind in (KIND_TEXT, KIND_GRAPHIC):
                    self._graphic(c, add)
                elif c.kind == KIND_MEDIA and t.kind is TrackKind.AUDIO:
                    self._audio(c, t, add)
                elif c.kind == KIND_MEDIA and self.p.assets.get(c.asset_id) is None:
                    add(c, "clip.asset", f"Clip {c.id} references a missing asset {c.asset_id}.")
        for d in self.st.decisions.values():
            if d.target_id and d.target_id not in clip_ids:
                out.append(ValidationIssue("warning", "decision.target", f"Decision {d.decision_id} points at a clip that no longer exists.", d.scene_id))
            if d.scene_id and d.scene_id not in self.scene_ids:
                out.append(ValidationIssue("warning", "decision.scene", f"Decision {d.decision_id} refers to the unknown scene {d.scene_id}.", d.scene_id))
        return out

    def _voice_duration(self) -> float | None:
        a = self.p.assets.get(self.p.voice_over.asset_id) if self.p.voice_over.asset_id else None
        return a.duration if a else None

    # ------------------------------------------------------------------ captions
    def _caption(self, c, track, voice_dur, add) -> None:
        d = c.text or {}
        text = str(d.get("text", "")).strip()
        if not text or not d.get("words"):
            add(c, "caption.text", "A caption has no text or no word timing.")
            return
        if voice_dur and c.timeline_end > voice_dur + 1.0:
            add(c, "caption.range", f"A caption ends at {c.timeline_end:.1f}s, after the voice-over ({voice_dur:.1f}s).")
        last = -1.0
        for w in d["words"]:
            if not (math.isfinite(w.get("start", math.nan)) and math.isfinite(w.get("end", math.nan))) or w["end"] < w["start"] or w["start"] < last - 1e-6:
                add(c, "caption.words", "Caption word timing is invalid.")
                break
            last = w["start"]
        cs = self.st.caption_settings
        style = effective_style(style_for(self.p.caption_styles, d.get("style_id", cs.style_id)), cs, d.get("style_overrides"))
        lay = layout_for(cs, style, (self.p.settings.width, self.p.settings.height))
        for line in d.get("lines") or [text]:
            if len(line) > lay.chars_per_line * 1.2:
                add(c, "caption.line", f"A caption line is too long to read comfortably ({len(line)} characters).")
                break
        if len(d.get("lines") or [text]) > max(1, cs.max_lines):
            add(c, "caption.lines", f"A caption has more than {cs.max_lines} lines.")
        if style.size_rel < MIN_FONT_REL:
            add(c, "caption.size", "Caption text is too small to read.")
        pos = d.get("position", cs.position)
        xy = d.get("position_xy") or []
        if pos == "custom" and len(xy) == 2:
            if not (cs.safe_margin_left <= xy[0] <= 1 - cs.safe_margin_right and cs.safe_margin_top <= xy[1] <= 1 - cs.safe_margin_bottom):
                add(c, "caption.safe_area", "The caption position is outside the safe area.", strict=False)
        elif pos not in ("bottom", "center", "top", "custom"):
            add(c, "caption.position", f"Unknown caption position “{pos}”.")
        est_w = max((len(l) for l in (d.get("lines") or [text])), default=0) * style.size_rel * self.p.settings.height * 0.52
        if est_w > lay.box_width * 1.15:
            add(c, "caption.safe_area", "The caption is wider than the safe area.", strict=False)

    # ------------------------------------------------------------------ graphics
    def _graphic(self, c, add) -> None:
        if c.kind == KIND_TEXT:
            t = c.text or {}
            if not str(t.get("content", "")).strip():
                add(c, "graphic.text", "A text graphic is empty.")
            pos = t.get("position", (0.5, 0.5))
            if not (len(pos) == 2 and 0.0 <= pos[0] <= 1.0 and 0.0 <= pos[1] <= 1.0):
                add(c, "graphic.position", "A text graphic is positioned outside the frame.")
            if float(t.get("size", 1)) <= 0:
                add(c, "graphic.size", "A text graphic has an invalid size.")
            if t.get("counter") and not str(t.get("content", "")).strip():
                add(c, "graphic.counter", "A counter has no target number.")
        else:
            hl = (c.effects or {}).get("highlight") or (c.effects or {}).get("evidence")
            region = (hl or {}).get("region")
            if region and not (len(region) == 4 and all(0.0 <= v <= 1.0 for v in region) and region[2] > 0 and region[3] > 0 and region[0] + region[2] <= 1.0001
                               and region[1] + region[3] <= 1.0001):
                add(c, "graphic.region", "A highlight region is outside the frame.")

    # ------------------------------------------------------------------ audio
    def _audio(self, c, track, add) -> None:
        a = self.p.assets.get(c.asset_id)
        if a is None:
            add(c, "audio.asset", f"Audio clip {c.id} references a missing asset.")
            return
        if a.type is not AssetType.AUDIO:
            add(c, "audio.type", f"{a.name} is not an audio file.")
            return
        if a.duration and (c.source_in < -1e-6 or c.source_out > a.duration + EPS or c.source_out <= c.source_in):
            add(c, "audio.source", f"{a.name}: the source range {c.source_in:.2f}–{c.source_out:.2f}s is impossible (the file is {a.duration:.2f}s).")
        vol = float(c.audio.get("volume", 1.0))
        if not (0.0 <= vol <= MAX_VOLUME):
            add(c, "audio.volume", f"{a.name}: the volume must be between 0 and {MAX_VOLUME:g}.")
        if float(c.audio.get("fade_in", 0)) + float(c.audio.get("fade_out", 0)) > c.duration + EPS or float(c.audio.get("fade_in", 0)) < 0 or float(c.audio.get("fade_out", 0)) < 0:
            add(c, "audio.fade", f"{a.name}: the fades do not fit the clip.")
        for k in c.keyframes:
            if k.property == "volume" and not (0.0 <= k.value <= MAX_VOLUME):
                add(c, "audio.keyframe", f"{a.name}: a volume keyframe is out of range.")
        cat = c.audio.get("category")
        if c.audio.get("role") == "SFX" and cat and cat not in VALID_SFX:
            add(c, "audio.category", f"Unknown SFX category “{cat}”.")
