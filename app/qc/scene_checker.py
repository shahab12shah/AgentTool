"""SceneChecker (spec 8): does every scene have the right picture on screen while it is narrated?

For each scene it compares what was ASSIGNED (project.visual_assignments, via the Phase 4 SceneContext) with what is ACTUALLY on the timeline, and looks for: narration with no
picture, an approved visual that never reached the timeline, one picture held too long, a sentence cut to pieces, and important statements (claims, numbers, dates, evidence
scenes) that only have a decorative picture and no text / number / evidence treatment.

It is scene-local: the answer for a scene depends on that scene, its words, its assignment and the clips around it (``ctx.scene_signature``) - plus the track flags and the
protection state, which ``scene_input_hash`` adds below. Every fix it recommends is a NAVIGATE route: this checker never offers to change the timeline by itself.
The reused pieces: ``VisualStatus`` / ``visual_status`` (via ``ctx.scene_ctx``), ``planners.is_evidence_visual`` / ``EVIDENCE_SOURCES``, ``editing.assembly.subtract`` and the
sentence-split pause of the transcript (``PAUSE_SECONDS``) for "narration is active".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.analysis.models import NumberKind, Scene, VisualType
from app.core.constants import MIN_CLIP_DURATION
from app.editing.assembly import subtract
from app.editing.context import SceneContext
from app.editing.models import VisualStatus
from app.editing.planners import EVIDENCE_SOURCES, is_evidence_visual
from app.media.asset import AssetType
from app.qc import fix_catalog
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.media_facts import extras
from app.qc.severity import Severity
from app.qc.timeline_checker import _below, _finite, _merge, _on_screen, _t
from app.timeline.clip import KIND_GRAPHIC, KIND_TEXT, Clip
from app.timeline.track import Track, TrackKind
from app.transcription.sentences import PAUSE_SECONDS

E, W, N, I = Severity.ERROR, Severity.WARNING, Severity.NOTICE, Severity.INFO
CAT = QCCategory.SCENE_COVERAGE
ERROR_UNCOVERED_SECONDS = 1.0  # uncovered narration longer than this is an ERROR in a scene of average importance (optional setting coverage.error_uncovered_seconds)
WHOLE_SCENE_UNCOVERED = 0.95  # share of the narration without a picture that counts as "the whole scene"
HOLD_WARNING_FACTOR = 1.5  # a hold this many times the limit is a WARNING instead of a NOTICE (optional setting coverage.hold_warning_factor)
SIMPLE_SENTENCE_WORDS = 14  # a sentence up to this long is "simple"; longer ones may carry proportionally more cuts (optional setting coverage.simple_sentence_words)
SHORT_CLUSTER = 3  # this many consecutive too-short shots are a cluster (optional setting coverage.short_cluster)
MIN_TREATMENT_SECONDS = 0.4  # a text / graphic element must be on screen this long inside the scene to count as a treatment of the statement
SUPPORT_NUMBERS = frozenset({NumberKind.PRICE, NumberKind.DOLLAR_AMOUNT, NumberKind.PERCENTAGE, NumberKind.DATE, NumberKind.DEADLINE, NumberKind.YEAR})  # not QUANTITY / AGE: too weak a claim

# QC code -> (title, why it matters, suggested fix, viewer impact 0..1)
INFO: dict[str, tuple[str, str, str, float]] = {
    "scene.coverage.missing": ("Narration without a visual", "The viewer hears the scene while the screen shows nothing for it.", "Place or choose a visual that covers the whole narration of the scene.", 0.8),
    "scene.coverage.skipped": ("Scene without a visual (skipped)", "You chose to show no visual here, so QC only mentions it.", "Nothing to do unless you want a visual after all.", 0.0),
    "scene.visual.not_on_timeline": ("Approved visual is not on the timeline", "The visual you approved never reached the video, so the scene is shown with something else or with nothing.", "Open the scene and place the approved visual (or re-run the AI edit for it).", 0.8),
    "scene.hold.excessive": ("One visual held for a long time", "A picture that stays unchanged this long loses the viewer's attention.", "Add a second visual or a cut-in inside the scene, or trim the hold.", 0.3),
    "scene.fragmentation.cuts": ("Many cuts inside one sentence", "More visual changes than the sentence can carry make the picture restless and hard to follow.", "Merge some of the shots so the sentence is carried by fewer visuals.", 0.4),
    "scene.fragmentation.short_shots": ("Cluster of very short shots", "Several shots shorter than the viewer needs to read the picture feel like flicker.", "Merge or lengthen the shots, or remove the filler ones.", 0.45),
    "scene.support.under_supported": ("Important statement may be under-supported", "A statement the viewer should be able to check is shown with a generic picture and no text, number or evidence treatment.",
                                      "Review the scene: add an evidence or data visual, or a text / number treatment.", 0.55),
}


def _opt(ctx: QCContext, name: str, default: float) -> float:
    return float(getattr(ctx.settings.coverage, name, default))


def _name(ctx: QCContext, c: Clip) -> str:
    a = ctx.asset(c.asset_id)
    return f"{a.name if a else c.kind} ({c.id[-6:]})"


def _span(spans: list[tuple[float, float]]) -> str:
    return ", ".join(f"{_t(a)}-{_t(b)}" for a, b in spans[:3]) + (f" and {len(spans) - 3} more" if len(spans) > 3 else "")


def _layered(ctx: QCContext, c: Clip) -> bool:
    """A picture-in-picture inset, a flagged overlay or a translucent clip sits OVER the picture instead of replacing it. (A B-roll cutaway is a real shot and is not a layer here.)"""
    if _below(c.opacity, 0.95) or _below(c.scale, 0.9):
        return True
    w, h = ctx.canvas
    if _finite(*c.position) and (abs(c.position[0]) > 0.02 * w or abs(c.position[1]) > 0.02 * h):
        return True
    return any(k in c.effects for k in ("overlay", "pip")) or str(c.metadata.get("layer", "")).lower() in ("overlay", "pip") or c.slot.lower().startswith(("overlay", "pip"))


def _moves(c: Clip) -> bool:
    """A still with real keyframed motion (Ken Burns) is "alive": it is held to the video limit, not the stricter still limit."""
    return sum(1 for k in c.keyframes if k.property in ("scale", "position_x", "position_y")) >= 2


@dataclass
class _Shot:
    """A stretch of one continuous picture as the viewer sees it (the topmost opaque clip); ``ext_*`` are its real limits, which may reach past the scene."""

    start: float
    end: float
    clips: list[Clip] = field(default_factory=list)
    ext_start: float = 0.0
    ext_end: float = 0.0

    @property
    def length(self) -> float:
        return self.ext_end - self.ext_start


@dataclass
class _Picture:
    shots: list[_Shot]
    overlap_seconds: float  # time with two or more opaque full pictures on screen at once
    track_of: dict[str, Track] = field(default_factory=dict)  # clip id -> the track it is on

    def owner(self, ctx: QCContext, clips: list[Clip]) -> tuple[Clip, Track | None]:
        """The clip a finding about a shot is anchored to: one the user owns / locked if there is one (so its protection applies), else the first."""
        pick = next((c for c in clips if ctx.is_protected(self.track_of.get(c.id), c)[0]), clips[0])
        return pick, self.track_of.get(pick.id)


def _same_picture(a: Clip, b: Clip, still: bool) -> bool:
    """Two neighbouring clips that are one continuous picture: the same clip, or the same asset continuing where the other stopped (a split, not a cut)."""
    if a.id == b.id:
        return True
    return bool(a.asset_id) and a.asset_id == b.asset_id and (still or abs(b.source_in - a.source_out) <= 0.15)


class SceneChecker(BaseChecker):
    id = "scene"
    label = "Scene Coverage"
    categories = (CAT,)
    # transcript: when the narration is active; assets: still vs video, evidence sources, the media files (visual_status looks for them)
    domains = ("timeline", "scenes", "visual", "transcript", "assets")
    settings_sections = ("coverage", "intentional_gaps", "fix_permissions")
    scene_local = True
    expensive = False
    uses_shared = True  # it stands down on a hole in the picture that the timeline checker already reports (same stretch, same cause)
    version = "1"

    # ------------------------------------------------------------------ cache keys
    def input_hash(self, ctx: QCContext) -> str:
        """The scene's claims, numbers and intent (``scene.support.under_supported``) are not part of the scene hash."""
        return sha(super().input_hash(ctx), extras(ctx, "facts"))

    def scene_input_hash(self, ctx: QCContext, scene_id: str) -> str:
        """``ctx.scene_signature`` plus what this checker reads beyond it: track flags (a hidden track shows nothing), who owns / locked the clips (the fix is disabled for them),
        and the evidence kind of the assignment."""
        base = sha(super().scene_input_hash(ctx, scene_id), extras(ctx, "facts", scene_id=scene_id))
        s = ctx.scene(scene_id)
        if s is None:
            return base
        base = sha(base, [g for g in self._timeline_gaps(ctx) if g[1] > s.start and g[0] < s.end])
        a = ctx.project.visual_assignments.get(scene_id)
        tracks = [[t.id, t.kind.value, t.hidden, t.locked] for t in ctx.timeline.tracks]
        prot = [[c.id, ctx.is_protected(t, c)[0]] for t, c in ctx.clips_in(s.start - 0.5, s.end + 0.5)]
        assets = [[x.id, x.type.value, x.source_type.value] for x in (ctx.asset(i) for i in sorted({c.asset_id for _t, c in ctx.clips_in(s.start, s.end) if c.asset_id}) + ([a.asset_id] if a and a.asset_id else [])) if x]
        return sha(base, tracks, prot, assets, [a.evidence_kind.value, a.source_type.value if a.source_type else ""] if a else None, scene_id in ctx.locked_scene_ids())

    @staticmethod
    def _timeline_gaps(ctx: QCContext) -> list[tuple[float, float]]:
        """The holes in the picture the timeline checker has already reported (empty when it did not run before this checker)."""
        out = ctx.shared.get("timeline")
        return sorted((round(i.start_time, 3), round(i.end_time, 3)) for i in getattr(out, "issues", []) if i.code == "timeline.gap.unintended" and i.start_time is not None and i.end_time is not None)

    def _owned_by_timeline(self, ctx: QCContext, status: str, uncovered: list[tuple[float, float]], unc_s: float) -> bool:
        """True when the only thing wrong is a hole between pictures that the timeline checker reports (with its own fix): one problem, one finding. A scene without a usable visual keeps
        its own finding, because the cause (nothing assigned / approved / on disk) and the way out (choose a visual) are only known here."""
        if status in (VisualStatus.UNAPPROVED.value, VisualStatus.MISSING_MEDIA.value, VisualStatus.MISSING.value) or unc_s <= 0:
            return False
        gaps = self._timeline_gaps(ctx)
        return sum(max(0.0, min(b, y) - max(a, x)) for a, b in uncovered for x, y in gaps) >= 0.9 * unc_s

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        scenes = ctx.target_scenes()
        every = ctx.visual_clips()  # every media clip on a picture track (the approved visual may sit on a hidden track: that is a coverage problem, not "missing")
        vis = [(t, c) for t, c in every if _on_screen(t, c)]
        placed = [c for _t, c in every]
        rows: dict[str, dict[str, Any]] = {}
        for i, s in enumerate(scenes):
            ctx.check_cancel()
            report(i / max(1, len(scenes)), f"Scene {s.label}")
            out.issues.extend(self._scene(ctx, s, vis, placed, rows))
        covered = [r["coverage_ratio"] for r in rows.values() if r["narration_seconds"] > 0]
        out.metrics = {
            "scenes_checked": len(rows), "scene_rows": rows, "coverage_ratio_mean": round(sum(covered) / len(covered), 4) if covered else 1.0,
            "scenes_with_uncovered_narration": sum(1 for r in rows.values() if r["uncovered_seconds"] > 0), "uncovered_seconds": round(sum(r["uncovered_seconds"] for r in rows.values()), 3),
            "holds_over_limit": sum(r["holds_over_limit"] for r in rows.values()), "fragmented_sentences": sum(r["fragmented_sentences"] for r in rows.values()),
        }
        out.notes.append(f"checked {len(rows)} scene(s)" + (f" of {len(ctx.scenes)}" if len(rows) != len(ctx.scenes) else ""))
        report(1.0, f"{len(out.issues)} scene coverage issue(s)")
        return out

    # ------------------------------------------------------------------ one scene
    def _scene(self, ctx: QCContext, s: Scene, vis: list[tuple[Track, Clip]], placed: list[Clip], rows: dict[str, dict[str, Any]]) -> list[QCIssue]:
        p = ctx.project
        fr, lo, hi = ctx.frame, float(s.start), float(s.end)
        sc = ctx.scene_ctx(s.id)
        a = p.visual_assignments.get(s.id)
        status = sc.visual_status if sc else VisualStatus.MISSING.value
        skipped = a is not None and a.skipped
        inside = [(t, c) for t, c in vis if c.timeline_end > lo + 1e-6 and c.timeline_start < hi - 1e-6]
        issues: list[QCIssue] = []

        # --- what the narration needs, and what covers it
        active = self._active(ctx, s)
        active_s = sum(b - a_ for a_, b in active)
        covered = _merge([(max(lo, c.timeline_start), min(hi, c.timeline_end)) for _t, c in inside])
        uncovered = [piece for span in active for piece in subtract(span, covered) if piece[1] - piece[0] >= fr - 1e-6]
        unc_s = sum(b - a_ for a_, b in uncovered)
        ratio = 1.0 - unc_s / active_s if active_s > 1e-6 else 1.0
        pic = self._picture(ctx, lo, hi, inside)
        main = max(inside, key=lambda tc: (max(0.0, min(hi, tc[1].timeline_end) - max(lo, tc[1].timeline_start)), -tc[1].timeline_start), default=None)  # the clip that carries most of the scene
        row = {"start": round(lo, 3), "end": round(hi, 3), "importance": round(float(s.importance), 3), "visual_status": status, "assigned_asset": a.asset_id if a else None,
               "actual_assets": sorted({c.asset_id for _t, c in inside if c.asset_id}), "visual_seconds": round(sum(b - a_ for a_, b in covered), 3), "narration_seconds": round(active_s, 3),
               "uncovered_seconds": round(unc_s, 3), "coverage_ratio": round(ratio, 4), "gaps": [[round(a_, 3), round(b, 3)] for a_, b in uncovered], "overlap_seconds": round(pic.overlap_seconds, 3),
               "shots": len(pic.shots), "longest_hold": 0.0, "holds_over_limit": 0, "fragmented_sentences": 0}
        rows[s.id] = row

        short_of_ratio = active_s > fr and ratio < ctx.settings.coverage.min_covered_ratio - 1e-9 and unc_s >= fr
        on_timeline = self._placed(ctx, s, a, placed)
        not_placed = a is not None and a.approved and not skipped and bool(a.asset_id) and ctx.asset(a.asset_id) is not None and status == VisualStatus.APPROVED.value and not on_timeline
        if not_placed:
            issues.append(self._not_on_timeline(ctx, s, a, inside, main, unc_s, active_s, uncovered, ratio, short_of_ratio))
        elif short_of_ratio and not skipped and self._owned_by_timeline(ctx, status, uncovered, unc_s):
            row["reported_by"] = "timeline"  # measured (the metrics keep it), but reported once, as timeline.gap.unintended
        elif short_of_ratio:
            issues.append(self._missing(ctx, s, status, skipped, unc_s, active_s, uncovered, ratio, inside))

        # --- the picture itself
        issues.extend(self._holds(ctx, s, sc, a, pic, row))
        issues.extend(self._fragmentation(ctx, s, sc, pic, row))
        if not skipped and inside and not short_of_ratio:  # a scene without a picture is a coverage problem first; judging its (absent) support would only repeat it
            sup = self._support(ctx, s, sc, a, main, inside)
            if sup is not None:
                issues.append(sup)
        return issues

    # ------------------------------------------------------------------ narration activity
    def _active(self, ctx: QCContext, s: Scene) -> list[tuple[float, float]]:
        """The stretches of the scene in which the narration is playing: word spans bridged across pauses shorter than the sentence-split pause (longer pauses are deliberate and
        need no picture), minus the gaps the user declared intentional. Without any transcript the whole scene counts."""
        lo, hi = float(s.start), float(s.end)
        words = [w for w in ctx.words_between(lo, hi) if _finite(w.start, w.end)]
        if not ctx.words:
            spans = [(lo, hi)]
        else:
            spans = []
            for w in sorted(words, key=lambda w: w.start):
                b, e = max(lo, w.start), min(hi, max(w.end, w.start))
                if e < b:
                    continue
                if spans and b - spans[-1][1] <= PAUSE_SECONDS:
                    spans[-1] = (spans[-1][0], max(spans[-1][1], e))
                else:
                    spans.append((b, e))
        declared = [(float(g[0]), float(g[1])) for g in ctx.settings.intentional_gaps if len(g) >= 2]
        return [piece for span in spans for piece in subtract(span, declared)]

    # ------------------------------------------------------------------ the picture the viewer sees
    def _picture(self, ctx: QCContext, lo: float, hi: float, inside: list[tuple[Track, Clip]]) -> _Picture:
        """Sweep the scene: at every moment the topmost opaque on-screen clip is the picture; stretches of one continuous picture are merged into shots."""
        if not inside:
            return _Picture([], 0.0)
        order = {t.id: i for i, t in enumerate(ctx.timeline.tracks)}
        still_of = {c.id: (t.kind is TrackKind.IMAGE or (ctx.asset(c.asset_id) is not None and ctx.asset(c.asset_id).type is AssetType.IMAGE)) for t, c in inside}
        cuts = sorted({lo, hi} | {min(hi, max(lo, x)) for _t, c in inside for x in (c.timeline_start, c.timeline_end)})
        shots: list[_Shot] = []
        prev: _Shot | None = None
        overlap = 0.0
        for a, b in zip(cuts, cuts[1:]):
            if b - a < 1e-6:
                continue
            mid = (a + b) / 2
            live = [(t, c) for t, c in inside if c.timeline_start <= mid < c.timeline_end]
            if not live:
                prev = None
                continue
            base = [tc for tc in live if not _layered(ctx, tc[1])] or live
            if len(base) > 1:
                overlap += b - a
            top = max(base, key=lambda tc: (order.get(tc[0].id, 0), tc[1].timeline_start, tc[1].id))[1]
            if prev is not None and abs(prev.end - a) < 1e-6 and _same_picture(prev.clips[-1], top, still_of.get(top.id, False)):
                prev.end = b
                if top.id != prev.clips[-1].id:
                    prev.clips.append(top)
                continue
            prev = _Shot(a, b, [top])
            shots.append(prev)
        for sh in shots:  # a shot that touches the scene edge goes on in the neighbouring scene: its true length is that of its clips
            sh.ext_start = min(c.timeline_start for c in sh.clips) if sh.start <= lo + 1e-6 else sh.start
            sh.ext_end = max(c.timeline_end for c in sh.clips) if sh.end >= hi - 1e-6 else sh.end
        return _Picture(shots, overlap, {c.id: t for t, c in inside})

    # ------------------------------------------------------------------ coverage
    def _placed(self, ctx: QCContext, s: Scene, a, placed: list[Clip]) -> bool:
        """The approved visual is on the timeline somewhere in this scene (any visual track, visible or not: visibility is judged by the coverage)."""
        if a is None or not a.asset_id:
            return True
        return any(c.asset_id == a.asset_id and c.timeline_end > s.start + ctx.frame and c.timeline_start < s.end - ctx.frame for c in placed)

    @staticmethod
    def _cause(status: str) -> str:
        return {VisualStatus.UNAPPROVED.value: "A visual was chosen for this scene but not approved yet.", VisualStatus.MISSING_MEDIA.value: "The approved visual's media file cannot be found.",
                VisualStatus.MISSING.value: "No visual is assigned to this scene."}.get(status, "The visual does not run for the whole narration.")

    def _missing(self, ctx: QCContext, s: Scene, status: str, skipped: bool, unc_s: float, active_s: float, uncovered, ratio: float, inside) -> QCIssue:
        pct = 100.0 * (1.0 - ratio)
        if skipped:
            title, why, suggested, impact = INFO["scene.coverage.skipped"]
            return self.issue("scene.coverage.skipped", CAT, I, title, scene_id=s.id, start=uncovered[0][0], end=uncovered[-1][1],
                              description=f"You marked scene {s.label} as skipped (no visual); {unc_s:.1f} s of its narration has no picture ({_span(uncovered)}).", why=why, current=f"{unc_s:.1f} s without a visual",
                              recommended="a visual, only if you want one", suggested_fix=suggested, viewer_impact=impact, signature="skipped", ctx=ctx)
        cause = self._cause(status)
        whole = ratio <= 1.0 - WHOLE_SCENE_UNCOVERED
        imp = float(s.importance)
        err_after = min(1.5, max(0.5, _opt(ctx, "error_uncovered_seconds", ERROR_UNCOVERED_SECONDS) * (1.5 - imp)))  # the more important the scene, the shorter the hole that still is an ERROR
        sev = E if (whole or unc_s > err_after) else (N if (imp < 0.35 and unc_s < 0.5) else W)
        title, why, suggested, impact = INFO["scene.coverage.missing"]
        if status == VisualStatus.MISSING.value:
            fix = fix_catalog.navigate("visual.search_again", "Search for a visual for this scene", scene_id=s.id)
        elif status in (VisualStatus.UNAPPROVED.value, VisualStatus.MISSING_MEDIA.value):
            fix = fix_catalog.navigate("visual.replace", "Choose or approve a visual for this scene", scene_id=s.id)
        else:
            fix = fix_catalog.navigate("open.scene", "Open the scene and extend or add a visual", scene_id=s.id)
        return self.issue("scene.coverage.missing", CAT, sev, title, scene_id=s.id, start=uncovered[0][0], end=uncovered[-1][1], importance=imp,
                          description=(f"{'The whole narration' if whole else f'{unc_s:.1f} s of {active_s:.1f} s of the narration'} of scene {s.label} has no visual on screen ({pct:.0f}% uncovered: {_span(uncovered)}). {cause}"),
                          why=why, current=f"{100 - pct:.0f}% of the narration covered", recommended=f"at least {100 * ctx.settings.coverage.min_covered_ratio:.0f}% covered", suggested_fix=suggested, fix=fix,
                          viewer_impact=min(1.0, impact + 0.2 * min(1.0, unc_s / max(active_s, 1e-6))), signature=f"{'whole' if whole else round(unc_s)}",
                          metrics={"uncovered_seconds": round(unc_s, 3), "coverage_ratio": round(ratio, 4), "visual_status": status}, ctx=ctx)

    def _not_on_timeline(self, ctx: QCContext, s: Scene, a, inside, main, unc_s: float, active_s: float, uncovered, ratio: float, short_of_ratio: bool) -> QCIssue:
        title, why, suggested, impact = INFO["scene.visual.not_on_timeline"]
        asset = ctx.asset(a.asset_id)
        shown = sorted({_name(ctx, c) for _t, c in inside})
        replaced_by_user = bool(inside) and not short_of_ratio and all(str(c.created_by).upper() == "USER" or c.locked for _t, c in inside)
        sev = N if replaced_by_user else E  # a picture you placed yourself instead of the approved one is your decision, not a defect
        what = f"The approved visual {asset.name if asset else a.asset_id} is not on the timeline in scene {s.label}; "
        if inside:
            what += f"the scene shows {', '.join(shown[:3])} instead."
        else:
            what += "the scene shows nothing."
        if short_of_ratio:
            what += f" {unc_s:.1f} s of {active_s:.1f} s of the narration has no picture ({_span(uncovered)})."
        return self.issue("scene.visual.not_on_timeline", CAT, sev, title, scene_id=s.id, clip=main[1] if (replaced_by_user and main) else None, track=main[0] if (replaced_by_user and main) else None,
                          start=s.start, end=s.end, description=what, why=why, current=f"{', '.join(shown[:2]) or 'no visual'} on the timeline", recommended=f"{asset.name if asset else a.asset_id} on the timeline",
                          suggested_fix=suggested, fix=fix_catalog.navigate("open.scene", "Open the scene to place the approved visual", scene_id=s.id), viewer_impact=impact if inside else 1.0,
                          signature=f"{a.asset_id}", metrics={"assigned_asset": a.asset_id, "uncovered_seconds": round(unc_s, 3), "coverage_ratio": round(ratio, 4)}, ctx=ctx)

    # ------------------------------------------------------------------ holds
    def _evidence(self, ctx: QCContext, sc: SceneContext | None, a, clip: Clip) -> bool:
        """A picture that needs reading time: an evidence / data visual, a document or screenshot, or a clip that carries an evidence highlight."""
        if clip.effects.get("evidence") or clip.effects.get("highlight"):
            return True
        asset = ctx.asset(clip.asset_id)
        if asset is None:
            return False
        if sc is not None and sc.asset is not None and sc.asset.asset_id == asset.id and is_evidence_visual(sc, sc.asset):
            return True
        if asset.source_type.value in EVIDENCE_SOURCES or (a is not None and a.asset_id == asset.id and a.evidence_kind.value == "EVIDENCE"):
            return True
        intent = ctx.project.visual_intents.get(sc.scene.id) if sc else None
        return bool(intent is not None and intent.type in (VisualType.EVIDENCE, VisualType.DATA) and asset.type is AssetType.IMAGE)

    def _holds(self, ctx: QCContext, s: Scene, sc: SceneContext | None, a, pic: _Picture, row: dict[str, Any]) -> list[QCIssue]:
        fr = ctx.frame
        cov = ctx.settings.coverage
        warn_factor = _opt(ctx, "hold_warning_factor", HOLD_WARNING_FACTOR)
        out: list[QCIssue] = []
        for sh in pic.shots:
            if sh.ext_start < s.start - fr:
                continue  # a hold that began in an earlier scene is that scene's finding
            clip, track = pic.owner(ctx, sh.clips)
            asset = ctx.asset(clip.asset_id)
            still = (track is not None and track.kind is TrackKind.IMAGE) or (asset is not None and asset.type is AssetType.IMAGE)
            limit = cov.max_hold_still_seconds if (still and not _moves(clip)) else cov.max_hold_seconds
            row["longest_hold"] = max(row["longest_hold"], round(sh.length, 3))
            if sh.length <= limit + fr:
                continue
            row["holds_over_limit"] += 1
            if self._evidence(ctx, sc, a, clip):
                continue  # a deliberate hold on something the viewer has to read
            title, why, suggested, impact = INFO["scene.hold.excessive"]
            sev = W if sh.length > warn_factor * limit else N
            out.append(self.issue("scene.hold.excessive", CAT, sev, title, scene_id=s.id, clip=clip, track=track, start=sh.ext_start, end=sh.ext_end,
                                  description=f"{_name(ctx, clip)} stays on screen for {sh.length:.1f} s ({_t(sh.ext_start)}-{_t(sh.ext_end)}); the limit for {'a still' if still and not _moves(clip) else 'a video'} is {limit:g} s.",
                                  why=why, current=f"{sh.length:.1f} s on screen", recommended=f"at most {limit:g} s", suggested_fix=suggested,
                                  fix=fix_catalog.navigate("open.scene", "Review the shots of this scene", scene_id=s.id), viewer_impact=min(1.0, impact + 0.2 * (sh.length / limit - 1.0)),
                                  signature=f"{clip.asset_id}:{round(sh.length)}", metrics={"hold_seconds": round(sh.length, 3), "limit_seconds": limit}, ctx=ctx))
        return out

    # ------------------------------------------------------------------ fragmentation
    def _fragmentation(self, ctx: QCContext, s: Scene, sc: SceneContext | None, pic: _Picture, row: dict[str, Any]) -> list[QCIssue]:
        fr = ctx.frame
        cov = ctx.settings.coverage
        out: list[QCIssue] = []
        shots = pic.shots
        # (1) visual changes inside one sentence
        instants: list[float] = []
        for sh in shots:
            for x in (sh.start, sh.end):
                if not instants or abs(x - instants[-1]) > 1.5 * fr:
                    instants.append(x)
        instants = [x for x in instants if s.start + 1.5 * fr < x < s.end - 1.5 * fr]
        simple = _opt(ctx, "simple_sentence_words", SIMPLE_SENTENCE_WORDS)
        title, why, suggested, impact = INFO["scene.fragmentation.cuts"]
        for sent in (sc.sentences if sc else []):
            lo, hi = max(float(sent.start), float(s.start)), min(float(sent.end), float(s.end))
            if hi - lo < 4 * fr:
                continue
            n = sum(1 for x in instants if lo + fr < x < hi - fr)
            allowed = cov.max_cuts_per_sentence * max(1, math.ceil(len(sent.word_ids) / simple))
            if n <= allowed:
                continue
            row["fragmented_sentences"] += 1
            first = next((sh for sh in shots if sh.end > lo + fr), None)
            clip, track = pic.owner(ctx, first.clips) if first is not None else (None, None)
            out.append(self.issue("scene.fragmentation.cuts", CAT, W if n >= allowed + 2 else N, title, scene_id=s.id, clip=clip, track=track, start=lo, end=hi,
                                  description=f"The sentence \"{sent.text.strip()[:60]}\" ({_t(lo)}-{_t(hi)}, {len(sent.word_ids)} words) is cut {n} times; up to {allowed:g} is the limit.",
                                  why=why, current=f"{n} visual changes in {hi - lo:.1f} s", recommended=f"at most {allowed:g} visual changes", suggested_fix=suggested,
                                  fix=fix_catalog.navigate("open.scene", "Review the shots of this sentence", scene_id=s.id), viewer_impact=min(1.0, impact + 0.1 * (n - allowed)),
                                  signature=f"{sent.sentence_id}:{n}", metrics={"cuts": n, "allowed": allowed}, ctx=ctx))
        # (2) clusters of very short shots (the shot's real length: a shot cut by the scene edge is not short)
        min_shot, need = cov.min_shot_seconds, int(_opt(ctx, "short_cluster", SHORT_CLUSTER))
        title, why, suggested, impact = INFO["scene.fragmentation.short_shots"]
        run: list[_Shot] = []

        def flush() -> None:
            if len(run) >= need and run[0].ext_start >= s.start - fr:  # reported by the scene the cluster starts in
                clip, track = pic.owner(ctx, [c for x in run for c in x.clips])
                a0, b0 = run[0].ext_start, run[-1].ext_end
                out.append(self.issue("scene.fragmentation.short_shots", CAT, W, title, scene_id=s.id, clip=clip, track=track, start=a0, end=b0,
                                      description=f"{len(run)} shots in a row are shorter than {min_shot:g} s ({_t(a0)}-{_t(b0)}; shortest {min(x.length for x in run):.2f} s).", why=why,
                                      current=f"{len(run)} shots under {min_shot:g} s", recommended=f"shots of at least {min_shot:g} s", suggested_fix=suggested,
                                      fix=fix_catalog.navigate("open.scene", "Review the shots of this scene", scene_id=s.id), viewer_impact=impact, signature=f"{clip.id}:{len(run)}",
                                      metrics={"shots": len(run), "shortest_seconds": round(min(x.length for x in run), 3)}, ctx=ctx))

        for sh in shots:
            short = MIN_CLIP_DURATION <= sh.length < min_shot - 1e-9
            if short and (not run or sh.ext_start - run[-1].ext_end <= 1.5 * fr):
                run.append(sh)
                continue
            flush()
            run = [sh] if short else []
        flush()
        return out

    # ------------------------------------------------------------------ important statements
    def _support(self, ctx: QCContext, s: Scene, sc: SceneContext | None, a, main, inside: list[tuple[Track, Clip]]) -> QCIssue | None:
        cov = ctx.settings.coverage
        if float(s.importance) < cov.important_scene - 1e-9:
            return None
        reasons: list[str] = []
        sig: list[str] = []
        conf = 0.0
        claims = {c.claim_id or c.text: c for c in s.claims if c.requires_evidence}
        if claims:
            first = next(iter(claims.values()))
            reasons.append(f"a claim that needs evidence (\"{first.text.strip()[:50]}\")")
            sig.append("claim:" + ",".join(sorted(claims)))
            conf = max(conf, 72.0)
        nums = [n for n in s.numbers if n.kind in SUPPORT_NUMBERS]
        if nums:
            reasons.append("spoken " + ", ".join(sorted({n.text for n in nums})[:3]))
            sig.append("num:" + ",".join(sorted({n.text for n in nums})))
            conf = max(conf, 78.0)
        intent = ctx.project.visual_intents.get(s.id)
        if intent is not None and intent.type in (VisualType.EVIDENCE, VisualType.DATA):
            reasons.append(f"a {intent.type.value.lower()} visual intent")
            sig.append("intent:" + intent.type.value)
            conf = max(conf, 62.0)
        if not reasons:
            return None
        if any(self._evidence(ctx, sc, a, c) for _t, c in inside):
            return None  # an evidence / data visual (or a highlighted clip) is on screen
        shown = ctx.memo("scene.treatments", lambda: [c for t, c in ctx.clips(track_kinds=(TrackKind.GRAPHICS, TrackKind.TEXT)) if not t.hidden and c.kind in (KIND_TEXT, KIND_GRAPHIC)
                                                     and (c.kind == KIND_GRAPHIC or str((c.text or {}).get("content", "")).strip())])
        if any(min(c.timeline_end, s.end) - max(c.timeline_start, s.start) >= MIN_TREATMENT_SECONDS for c in shown):
            return None  # a text / number / highlight element is on screen during the scene
        title, why, suggested, impact = INFO["scene.support.under_supported"]
        conf = min(85.0, conf + 4.0 * (len(reasons) - 1))
        current = _name(ctx, main[1]) if main else "the current visual"
        return self.issue("scene.support.under_supported", CAT, W, title, scene_id=s.id, clip=main[1] if main else None, track=main[0] if main else None, start=s.start, end=s.end, confidence=conf,
                          description=(f"Potential under-support detected in scene {s.label} (importance {s.importance:.2f}): it contains {'; '.join(reasons)}, but {current} looks decorative and "
                                       f"no text, number or evidence treatment is on screen. Review recommended."),
                          why=why, current="a generic visual and no text or evidence treatment", recommended="an evidence / data visual, or a text / number treatment on V4 / V5", suggested_fix=suggested,
                          fix=fix_catalog.navigate("visual.replace", "Choose an evidence or data visual for this scene", scene_id=s.id), viewer_impact=min(1.0, impact + 0.3 * (float(s.importance) - cov.important_scene)),
                          signature="|".join(sig), metrics={"triggers": sig}, ctx=ctx)
