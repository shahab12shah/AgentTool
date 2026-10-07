"""Undoable timeline operations."""

from __future__ import annotations

from app.core.commands import Command
from app.core.constants import MIN_CLIP_DURATION
from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.timeline import Timeline
from app.timeline.track import Track

SCALE_RANGE = (0.01, 20.0)
SPEED_RANGE = (0.1, 8.0)


class _TimelineCommand(Command):
    scope = "timeline"

    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline


# ---------------------------------------------------------------- tracks
class AddTrackCommand(_TimelineCommand):
    description = "Add track"

    def __init__(self, timeline: Timeline, track: Track, index: int | None = None) -> None:
        super().__init__(timeline)
        self.track = track
        self.index = index

    def do(self) -> None:
        if not self.track.name.strip():
            raise TimelineError("A track needs a name.")
        self.timeline.insert_track(self.track, self.index)

    def undo(self) -> None:
        self.timeline.remove_track(self.track.id)


class RemoveTrackCommand(_TimelineCommand):
    description = "Delete track"

    def __init__(self, timeline: Timeline, track_id: str) -> None:
        super().__init__(timeline)
        self.track_id = track_id
        self._removed: tuple[Track, int] | None = None

    def do(self) -> None:
        self.timeline.require_unlocked(self.track_id)
        self._removed = self.timeline.remove_track(self.track_id)

    def undo(self) -> None:
        assert self._removed is not None
        track, index = self._removed
        self.timeline.insert_track(track, index)


class RenameTrackCommand(_TimelineCommand):
    description = "Rename track"

    def __init__(self, timeline: Timeline, track_id: str, name: str) -> None:
        super().__init__(timeline)
        self.track_id = track_id
        self.name = name.strip()
        self._old: str | None = None

    def do(self) -> None:
        if not self.name:
            raise TimelineError("A track needs a name.")
        track = self.timeline.get_track(self.track_id)
        self._old = track.name
        track.name = self.name

    def undo(self) -> None:
        assert self._old is not None
        self.timeline.get_track(self.track_id).name = self._old


_FLAG_TEXT = {
    "hidden": ("Hide track", "Show track"),
    "muted": ("Mute track", "Unmute track"),
    "locked": ("Lock track", "Unlock track"),
    "solo": ("Solo track", "Unsolo track"),
}


class SetTrackFlagCommand(_TimelineCommand):
    """Toggle ``hidden`` / ``muted`` / ``locked``."""

    major = False

    def __init__(self, timeline: Timeline, track_id: str, flag: str, value: bool) -> None:
        if flag not in ("hidden", "muted", "locked", "solo"):
            raise ValueError(flag)
        super().__init__(timeline)
        self.track_id, self.flag, self.value = track_id, flag, value
        self.description = _FLAG_TEXT[flag][0 if value else 1]
        self._old = False

    def do(self) -> None:
        track = self.timeline.get_track(self.track_id)
        self._old = getattr(track, self.flag)
        setattr(track, self.flag, self.value)

    def undo(self) -> None:
        setattr(self.timeline.get_track(self.track_id), self.flag, self._old)


class SetTrackVolumeCommand(_TimelineCommand):
    """Track gain (audio tracks), 0..2 linear."""

    description = "Change track volume"
    major = False
    merge_key = None

    def __init__(self, timeline: Timeline, track_id: str, volume: float) -> None:
        super().__init__(timeline)
        if not (0.0 <= volume <= 2.0):
            raise TimelineError("Track volume must be between 0 and 200%.")
        self.track_id, self.volume = track_id, volume
        self._old = 1.0

    def do(self) -> None:
        track = self.timeline.get_track(self.track_id)
        self._old = track.volume
        track.volume = self.volume

    def undo(self) -> None:
        self.timeline.get_track(self.track_id).volume = self._old


# ---------------------------------------------------------------- clips
class AddClipCommand(_TimelineCommand):
    description = "Add clip"

    def __init__(self, timeline: Timeline, clip: Clip) -> None:
        super().__init__(timeline)
        self.clip = clip

    def do(self) -> None:
        self.timeline.require_unlocked(self.clip.track_id)
        self.timeline.check_free(self.clip.track_id, self.clip.timeline_start, self.clip.duration)
        self.timeline.insert_clip(self.clip.snapshot())

    def undo(self) -> None:
        self.timeline.detach_clip(self.clip.id)


class DeleteClipCommand(_TimelineCommand):
    description = "Delete clip"

    def __init__(self, timeline: Timeline, clip_id: str) -> None:
        super().__init__(timeline)
        self.clip_id = clip_id
        self._removed: Clip | None = None

    def do(self) -> None:
        track, clip = self.timeline.find_clip(self.clip_id)
        self.timeline.require_unlocked(track.id)
        self._removed = self.timeline.detach_clip(self.clip_id)

    def undo(self) -> None:
        assert self._removed is not None
        self.timeline.insert_clip(self._removed)


class _ClipEditCommand(_TimelineCommand):
    """Base for edits expressed as before/after clip states."""

    major = True

    def __init__(self, timeline: Timeline, clip_id: str) -> None:
        super().__init__(timeline)
        self.clip_id = clip_id
        self._before: Clip | None = None
        self._after: Clip | None = None

    def _compute(self, before: Clip) -> Clip:  # pragma: no cover - abstract
        raise NotImplementedError

    def do(self) -> None:
        if self._after is None:
            track, clip = self.timeline.find_clip(self.clip_id)
            self.timeline.require_unlocked(track.id)
            before = clip.snapshot()
            after = self._compute(before)
            if after.track_id != before.track_id:
                self.timeline.require_unlocked(after.track_id)
            self.timeline.check_free(after.track_id, after.timeline_start, after.duration, ignore_clip_id=after.id)
            self._before, self._after = before, after
        self.timeline.restore_clip(self._after)

    def undo(self) -> None:
        assert self._before is not None
        self.timeline.restore_clip(self._before)


class MoveClipCommand(_ClipEditCommand):
    description = "Move clip"

    def __init__(self, timeline: Timeline, clip_id: str, new_start: float, new_track_id: str | None = None) -> None:
        super().__init__(timeline, clip_id)
        self.new_start = new_start
        self.new_track_id = new_track_id

    def _compute(self, before: Clip) -> Clip:
        after = before.snapshot()
        after.timeline_start = max(0.0, self.new_start)
        if self.new_track_id:
            after.track_id = self.new_track_id
        return after


class TrimClipCommand(_ClipEditCommand):
    description = "Trim clip"

    def __init__(
        self,
        timeline: Timeline,
        clip_id: str,
        *,
        new_start: float | None = None,
        new_end: float | None = None,
        max_source: float | None = None,
    ) -> None:
        super().__init__(timeline, clip_id)
        self.new_start, self.new_end, self.max_source = new_start, new_end, max_source
        self.description = "Trim clip start" if new_start is not None and new_end is None else "Trim clip end"

    def _compute(self, before: Clip) -> Clip:
        r = self.timeline.compute_trim(
            before, new_start=self.new_start, new_end=self.new_end, max_source=self.max_source
        )
        after = before.snapshot()
        after.timeline_start, after.duration = r.start, r.duration
        after.source_in, after.source_out = r.source_in, r.source_out
        return after


class SetClipPropertiesCommand(_ClipEditCommand):
    """Change transform properties (position/scale/rotation/opacity/speed)."""

    description = "Edit clip properties"
    major = False
    merge_key = None

    def __init__(self, timeline: Timeline, clip_id: str, **changes: object) -> None:
        super().__init__(timeline, clip_id)
        allowed = {"position", "scale", "rotation", "opacity", "speed"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported clip properties: {sorted(unknown)}")
        self.changes = changes

    def _compute(self, before: Clip) -> Clip:
        after = before.snapshot()
        c = self.changes
        if "position" in c:
            x, y = c["position"]  # type: ignore[misc]
            after.position = (float(x), float(y))
        if "scale" in c:
            after.scale = _in_range("Scale", float(c["scale"]), *SCALE_RANGE)  # type: ignore[arg-type]
        if "rotation" in c:
            after.rotation = float(c["rotation"])  # type: ignore[arg-type]
        if "opacity" in c:
            after.opacity = _in_range("Opacity", float(c["opacity"]), 0.0, 1.0)  # type: ignore[arg-type]
        if "speed" in c:
            after.speed = _in_range("Speed", float(c["speed"]), *SPEED_RANGE)  # type: ignore[arg-type]
            after.duration = (before.source_out - before.source_in) / after.speed
        return after


def _in_range(label: str, value: float, low: float, high: float) -> float:
    if not (low <= value <= high):
        raise TimelineError(f"{label} must be between {low:g} and {high:g}.")
    return value


class SplitClipCommand(_TimelineCommand):
    """Split one clip into two at ``at`` (timeline seconds). Non-destructive: both halves keep the same source media."""

    description = "Split clip"

    def __init__(self, timeline: Timeline, clip_id: str, at: float, new_clip_id: str) -> None:
        super().__init__(timeline)
        self.clip_id, self.at, self.new_clip_id = clip_id, at, new_clip_id
        self._before: Clip | None = None
        self._left: Clip | None = None
        self._right: Clip | None = None

    def do(self) -> None:
        if self._left is None:
            track, clip = self.timeline.find_clip(self.clip_id)
            self.timeline.require_unlocked(track.id)
            if not (clip.timeline_start + MIN_CLIP_DURATION < self.at < clip.timeline_end - MIN_CLIP_DURATION):
                raise TimelineError("Place the playhead inside the clip (not at its very edge) to split it.")
            before = clip.snapshot()
            cut = self.at - clip.timeline_start
            left, right = before.snapshot(), before.snapshot()
            left.duration = cut
            left.source_out = before.source_in + cut * before.speed
            right.id, right.timeline_start, right.duration = self.new_clip_id, self.at, before.duration - cut
            right.source_in = left.source_out
            left.keyframes, right.keyframes = split_keyframes(before.keyframes, cut, before.duration)
            right.transition, right.slot, right.ai_decision_id = None, (before.slot + ".r") if before.slot else "", ""
            right.locked, right.created_by = False, "USER" if before.scene_id else before.created_by
            self._before, self._left, self._right = before, left, right
        self.timeline.restore_clip(self._left)
        self.timeline.insert_clip(self._right.snapshot())

    def undo(self) -> None:
        assert self._before is not None
        self.timeline.detach_clip(self.new_clip_id)
        self.timeline.restore_clip(self._before)


def split_keyframes(kfs, cut: float, duration: float):
    """Divide keyframes at ``cut`` (clip-local seconds): the right half is shifted to start at 0 and both halves get boundary keyframes."""
    from app.timeline.keyframes import Keyframe, value_at

    left, right = [], []
    for prop in {k.property for k in kfs}:
        mine = sorted((k for k in kfs if k.property == prop), key=lambda k: k.time)
        mid = value_at(mine, prop, cut)
        interp = next((k.interpolation for k in reversed(mine) if k.time <= cut), mine[0].interpolation)
        left += [Keyframe(k.property, k.time, k.value, k.interpolation, k.decision_id) for k in mine if k.time < cut - 1e-6]
        left.append(Keyframe(prop, cut, mid, interp, ""))
        right.append(Keyframe(prop, 0.0, mid, interp, ""))
        right += [Keyframe(k.property, k.time - cut, k.value, k.interpolation, k.decision_id) for k in mine if k.time > cut + 1e-6]
    return left, right
