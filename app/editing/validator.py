"""TimelineValidator: nothing is committed to the project timeline unless this passes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.editing.models import TransitionType
from app.media.asset import AssetType
from app.timeline.timeline import Timeline
from app.timeline.track import TrackKind

EPS = 0.05
VISUAL_TRACKS = ("track_v1", "track_v2", "track_v3")


@dataclass
class ValidationIssue:
    severity: str  # error | warning
    code: str
    message: str
    scene_id: str = ""
    clip_id: str = ""

    def __str__(self) -> str:
        where = f" [scene {self.scene_id}]" if self.scene_id else ""
        return f"{self.message}{where}"


class TimelineValidator:
    def __init__(self, assets, scenes, voice_asset_id: str | None, decisions: dict, strategy, generation) -> None:
        self.assets, self.scenes, self.voice, self.decisions = assets, scenes, voice_asset_id, decisions
        self.strategy, self.generation = strategy, generation

    def validate(self, timeline: Timeline) -> list[ValidationIssue]:
        out: list[ValidationIssue] = []
        add = lambda sev, code, msg, c=None, sid="": out.append(ValidationIssue(sev, code, msg, sid or (c.scene_id if c else ""), c.id if c else ""))  # noqa: E731
        track_ids: set[str] = set()
        clip_ids: set[str] = set()
        for t in timeline.tracks:
            if t.id in track_ids:
                add("error", "track.duplicate", f"Duplicate track id {t.id}.")
            track_ids.add(t.id)
            prev = None
            for c in sorted(t.clips, key=lambda x: x.timeline_start):
                if c.id in clip_ids:
                    add("error", "clip.duplicate", f"Duplicate clip id {c.id}.", c)
                clip_ids.add(c.id)
                if c.track_id != t.id:
                    add("error", "clip.track", f"Clip {c.id} is stored on {t.id} but claims {c.track_id}.", c)
                if not all(math.isfinite(v) for v in (c.timeline_start, c.duration, c.source_in, c.source_out, c.speed)):
                    add("error", "clip.nonfinite", "A clip has a non-numeric time value.", c)
                    continue
                if c.timeline_start < -1e-6:
                    add("error", "clip.start", f"Clip {c.id} starts before 0:00.", c)
                if c.duration <= 0:
                    add("error", "clip.duration", f"Clip {c.id} has a zero or negative duration.", c)
                if prev is not None and c.timeline_start < prev.timeline_end - 1e-4:
                    add("error", "clip.overlap", f"Clips overlap on track “{t.name}”.", c)
                prev = c if prev is None or c.timeline_end > prev.timeline_end else prev
                self._clip(c, t, add)
        self._cross_track(timeline, add)
        self._voice(timeline, add)
        self._coverage(timeline, add)
        for d in self.decisions.values():
            if d.target_id and d.target_id not in clip_ids:
                add("warning", "decision.target", f"Decision {d.decision_id} points at a clip that no longer exists.", None, d.scene_id)
        return out

    def _clip(self, c, track, add) -> None:
        if c.speed <= 0:
            add("error", "clip.speed", "A clip has an invalid speed.", c)
        if c.kind == "media":
            asset = self.assets.get(c.asset_id)
            if asset is None:
                add("error", "clip.asset", f"Clip {c.id} references a missing asset {c.asset_id}.", c)
            else:
                if asset.type is AssetType.AUDIO and track.kind is not TrackKind.AUDIO or asset.type is not AssetType.AUDIO and track.kind is TrackKind.AUDIO:
                    add("error", "clip.track_kind", f"{asset.name} cannot be placed on track “{track.name}”.", c)
                if asset.type is not AssetType.IMAGE and asset.duration:
                    if c.source_in < -1e-6 or c.source_out <= c.source_in or c.source_out > asset.duration + EPS:
                        add("error", "clip.source", f"{asset.name}: the source range {c.source_in:.2f}–{c.source_out:.2f}s is impossible "
                                                    f"(media is {asset.duration:.2f}s).", c)
                    elif abs((c.source_out - c.source_in) - c.duration * c.speed) > EPS:
                        add("error", "clip.source_len", f"{asset.name}: source range and duration disagree.", c)
        elif c.kind == "text":
            if not (c.text and str(c.text.get("content", "")).strip()):
                add("error", "clip.text", "A text element is empty.", c)
        for k in c.keyframes:
            for problem in k.problems(c.duration):
                add("error", "keyframe", problem, c)
        if c.transition is not None:
            kind, dur = c.transition.get("type"), c.transition.get("duration", 0.0)
            if kind not in {t.value for t in TransitionType} or not isinstance(dur, (int, float)) or dur < 0 or dur > c.duration + 1e-6:
                add("error", "transition", f"Invalid transition on clip {c.id}.", c)

    def _cross_track(self, timeline: Timeline, add) -> None:
        vis = [(c, t) for t in timeline.tracks if t.id in VISUAL_TRACKS for c in t.clips if c.kind == "media"]
        vis.sort(key=lambda p: p[0].timeline_start)
        for i, (a, ta) in enumerate(vis):
            for b, tb in vis[i + 1:]:
                if b.timeline_start >= a.timeline_end - 1e-4:
                    break
                if ta.id != tb.id:
                    user = a.created_by == "USER" or b.created_by == "USER"
                    add("warning" if user else "error", "visual.overlap",
                        f"Two visuals overlap between {ta.name} and {tb.name}.", b)

    def _voice(self, timeline: Timeline, add) -> None:
        if not self.voice:
            return
        asset = self.assets.get(self.voice)
        clips = [c for t in timeline.tracks if t.kind is TrackKind.AUDIO for c in t.clips if c.asset_id == self.voice]
        if asset is None or not clips:
            add("error", "voice.missing", "The voice-over is not on the timeline.")
            return
        c = min(clips, key=lambda x: x.timeline_start)
        sev = "warning" if c.created_by == "USER" else "error"
        if abs(c.timeline_start) > EPS or (asset.duration and abs(c.duration - asset.duration) > 0.1):
            add(sev, "voice.alignment", "The voice-over is no longer aligned with the scenes (it must start at 0:00 and play in full).", c)

    def _coverage(self, timeline: Timeline, add) -> None:
        for sc in self.scenes:
            st = self.generation.scenes.get(sc.id)
            if st is None or st.status.value != "COMPLETE" or not self.strategy.segments.get(sc.id):
                continue
            clips = [c for t in timeline.tracks if t.id in VISUAL_TRACKS for c in t.clips if c.scene_id == sc.id and c.kind == "media"]
            owned_by_user = any(c.created_by == "USER" or c.locked for c in clips)
            cursor, gap = sc.start, 0.0
            for c in sorted(clips, key=lambda x: x.timeline_start):
                if c.timeline_start - cursor > EPS:
                    gap = max(gap, c.timeline_start - cursor)
                cursor = max(cursor, c.timeline_end)
            if sc.end - cursor > EPS:
                gap = max(gap, sc.end - cursor)
            if not clips or gap > EPS:
                add("warning" if owned_by_user else "error", "scene.coverage",
                    f"Scene {sc.label}: the visuals leave a {gap:.2f}s gap in the narration." if clips else f"Scene {sc.label} has no visual on the timeline.",
                    None, sc.id)
