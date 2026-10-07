"""Timeline model: tracks, clips and the pure rules that govern them.

No UI, no commands, no persistence here. Methods named ``require_*``/``check_*`` raise
``TimelineError``; the raw mutators (``insert_clip``, ``restore_clip``...) do not validate
so undo/redo can always restore state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.track import Track, TrackKind

DEFAULT_TRACKS: tuple[tuple[str, str, TrackKind], ...] = (
    ("track_v1", "V1 Main Video", TrackKind.VIDEO),
    ("track_v2", "V2 B-Roll", TrackKind.VIDEO),
    ("track_v3", "V3 Images", TrackKind.IMAGE),
    ("track_v4", "V4 Graphics", TrackKind.GRAPHICS),
    ("track_v5", "V5 Text", TrackKind.TEXT),
    ("track_a1", "A1 Voice-over", TrackKind.AUDIO),
    ("track_a2", "A2 Music", TrackKind.AUDIO),
    ("track_a3", "A3 SFX", TrackKind.AUDIO),
)


def new_track_id() -> str:
    return f"track_{uuid.uuid4().hex[:8]}"


def new_clip_id() -> str:
    return f"clip_{uuid.uuid4().hex[:12]}"


@dataclass
class TrimResult:
    start: float
    duration: float
    source_in: float
    source_out: float


class Timeline:
    def __init__(self, tracks: list[Track] | None = None) -> None:
        self.tracks: list[Track] = tracks if tracks is not None else []

    @classmethod
    def default(cls) -> "Timeline":
        return cls([Track(id=i, name=n, kind=k) for i, n, k in DEFAULT_TRACKS])

    # ----- lookup -----
    def get_track(self, track_id: str) -> Track:
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise TimelineError("That track no longer exists.", details=track_id)

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for t in self.tracks:
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        raise TimelineError("That clip no longer exists.", details=clip_id)

    def get_clip(self, clip_id: str) -> Clip | None:
        try:
            return self.find_clip(clip_id)[1]
        except TimelineError:
            return None

    def all_clips(self) -> list[Clip]:
        return [c for t in self.tracks for c in t.clips]

    def clips_for_asset(self, asset_id: str) -> list[Clip]:
        return [c for c in self.all_clips() if c.asset_id == asset_id]

    @property
    def duration(self) -> float:
        return max((c.timeline_end for c in self.all_clips()), default=0.0)

    # ----- rules -----
    def require_unlocked(self, track_id: str) -> Track:
        track = self.get_track(track_id)
        if track.locked:
            raise TimelineError(f"Track “{track.name}” is locked. Unlock it to edit.")
        return track

    def check_free(self, track_id: str, start: float, duration: float, ignore_clip_id: str | None = None) -> None:
        if start < -TIME_EPSILON:
            raise TimelineError("A clip cannot start before 0:00.")
        if duration < MIN_CLIP_DURATION - TIME_EPSILON:
            raise TimelineError("The clip would be too short.")
        end = start + duration
        for c in self.get_track(track_id).clips:
            if c.id == ignore_clip_id:
                continue
            if start < c.timeline_end - TIME_EPSILON and end > c.timeline_start + TIME_EPSILON:
                raise TimelineError("That would overlap another clip on the same track.")

    def first_free_start(self, track_id: str, start: float, duration: float) -> float:
        """Earliest start >= ``start`` where a clip of ``duration`` fits on the track."""
        t = max(0.0, start)
        for c in self.get_track(track_id).clips:  # sorted by start
            if t + duration <= c.timeline_start + TIME_EPSILON:
                break
            if t < c.timeline_end - TIME_EPSILON:
                t = c.timeline_end
        return t

    def neighbours(self, clip: Clip) -> tuple[float, float | None]:
        """(end of previous clip or 0, start of next clip or None) on the clip's track."""
        prev_end, next_start = 0.0, None
        for c in self.get_track(clip.track_id).clips:
            if c.id == clip.id:
                continue
            if c.timeline_end <= clip.timeline_start + TIME_EPSILON:
                prev_end = max(prev_end, c.timeline_end)
            elif c.timeline_start >= clip.timeline_end - TIME_EPSILON:
                next_start = c.timeline_start if next_start is None else min(next_start, c.timeline_start)
        return prev_end, next_start

    def compute_trim(
        self,
        clip: Clip,
        *,
        new_start: float | None = None,
        new_end: float | None = None,
        max_source: float | None = None,
    ) -> TrimResult:
        """Trim one edge, clamping to what is physically possible (neighbours, source bounds, min length)."""
        prev_end, next_start = self.neighbours(clip)
        start, end = clip.timeline_start, clip.timeline_end
        if new_start is not None:
            lowest = max(prev_end, start - clip.source_in / clip.speed, 0.0)
            start = min(max(new_start, lowest), end - MIN_CLIP_DURATION)
        if new_end is not None:
            highest = float("inf") if next_start is None else next_start
            if max_source is not None:
                highest = min(highest, clip.timeline_start + (max_source - clip.source_in) / clip.speed)
            end = max(min(new_end, highest), start + MIN_CLIP_DURATION)
        duration = end - start
        source_in = clip.source_in + (start - clip.timeline_start) * clip.speed
        return TrimResult(start, duration, max(source_in, 0.0), max(source_in, 0.0) + duration * clip.speed)

    # ----- raw mutators (no validation; used by commands incl. undo) -----
    def insert_track(self, track: Track, index: int | None = None) -> None:
        if any(t.id == track.id for t in self.tracks):
            raise TimelineError("A track with that id already exists.")
        self.tracks.insert(len(self.tracks) if index is None else index, track)

    def remove_track(self, track_id: str) -> tuple[Track, int]:
        track = self.get_track(track_id)
        index = self.tracks.index(track)
        self.tracks.remove(track)
        return track, index

    def insert_clip(self, clip: Clip) -> None:
        track = self.get_track(clip.track_id)
        track.clips.append(clip)
        track.sort()

    def detach_clip(self, clip_id: str) -> Clip:
        track, clip = self.find_clip(clip_id)
        track.clips.remove(clip)
        return clip

    def restore_clip(self, state: Clip) -> None:
        """Replace the clip with id ``state.id`` (possibly on another track) by a copy of ``state``."""
        if self.get_clip(state.id) is not None:
            self.detach_clip(state.id)
        self.insert_clip(state.snapshot())

    # ----- persistence -----
    def to_dict(self) -> dict[str, Any]:
        return {"tracks": [t.to_dict() for t in self.tracks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Timeline":
        return cls([Track.from_dict(t) for t in d.get("tracks", [])])
