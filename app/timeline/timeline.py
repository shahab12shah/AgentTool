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
from app.timeline.index import TimelineIndex, bump, epoch, watched
from app.timeline.track import Track, TrackKind

DEFAULT_TRACKS: tuple[tuple[str, str, TrackKind], ...] = (
    ("track_v1", "V1 Main Video", TrackKind.VIDEO),
    ("track_v2", "V2 B-Roll", TrackKind.VIDEO),
    ("track_v3", "V3 Images", TrackKind.IMAGE),
    ("track_v4", "V4 Graphics", TrackKind.GRAPHICS),
    ("track_v5", "V5 Text", TrackKind.TEXT),
    ("track_v6", "V6 Captions", TrackKind.CAPTIONS),
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


_BUILD_AFTER = 6  # a stale index is only rebuilt once this many lookups hit the same state; mutate/lookup loops keep the old linear cost instead of paying a rebuild per step


class Timeline:
    def __init__(self, tracks: list[Track] | None = None) -> None:
        self._rev = 0
        self._ix: TimelineIndex | None = None
        self._seen = (-1, 0)
        self.tracks: list[Track] = tracks if tracks is not None else []

    def __setattr__(self, name: str, value: object) -> None:
        if name == "tracks":
            value = watched(value)  # type: ignore[arg-type]
        object.__setattr__(self, name, value)
        if name == "tracks":
            bump()

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_ix"], d["_seen"] = None, (-1, 0)  # the index is derived data: copies/pickles rebuild their own
        return d

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        bump()

    # ----- revision / index -----
    @property
    def revision(self) -> int:
        """Changes whenever anything in the process wrote a clip's geometry or a clip/track list, or ``touch()`` was called. Safe cache key for derived data (never goes backwards)."""
        return self._rev + epoch()

    def touch(self) -> None:
        """Declare that timeline content changed in a way the automatic tracking cannot see."""
        self._rev += 1
        bump()

    def _signature(self) -> tuple:
        tracks = self.tracks
        return (id(tracks), len(tracks), tuple((id(t.clips), len(t.clips)) for t in tracks))

    def index(self, *, build: bool = True) -> TimelineIndex | None:
        """The current index, or ``None`` when it is stale and ``build`` is False and the state has not been looked up often enough to justify a rebuild."""
        ix, e = self._ix, epoch()
        sig = self._signature()
        if ix is not None and ix.epoch == e and ix.sig == sig:
            return ix
        if not build:
            seen_e, n = self._seen
            n = n + 1 if seen_e == e else 1
            self._seen = (e, n)
            if n < _BUILD_AFTER:
                return None
        ix = TimelineIndex(self.tracks, e, sig)
        self._ix = ix
        return ix

    def _fast(self) -> TimelineIndex | None:
        ix = self.index(build=False)
        return ix if ix is not None and ix.sane else None

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
        ix = self._fast()
        if ix is not None:
            hit = ix.by_id.get(clip_id)
            if hit is not None and hit[1].id == clip_id:
                return hit
            if hit is None:
                raise TimelineError("That clip no longer exists.", details=clip_id)
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
        ix = self._fast()
        if ix is not None:
            return [c for c in ix.by_asset.get(asset_id, ()) if c.asset_id == asset_id]
        return [c for c in self.all_clips() if c.asset_id == asset_id]

    @property
    def duration(self) -> float:
        ix = self._fast()
        if ix is not None:
            return ix.duration
        return max((c.timeline_end for c in self.all_clips()), default=0.0)

    # ----- range queries for the canvas (always indexed) -----
    def clips_in_range(self, track_id: str | None, t0: float, t1: float) -> list[Clip]:
        """Clips overlapping the open interval (t0, t1), per track in start order; ``track_id=None`` means every track in track order."""
        ix = self.index()
        assert ix is not None
        if not ix.sane:
            return [c for t in ([self.get_track(track_id)] if track_id is not None else self.tracks) for c in t.clips if c.timeline_end > t0 and c.timeline_start < t1]
        if track_id is not None:
            self.get_track(track_id)  # same error as the other track lookups when the track is gone
            ti = ix.tracks.get(track_id)
            return ti.in_range(t0, t1) if ti is not None else []
        out: list[Clip] = []
        for t in self.tracks:
            ti = ix.tracks.get(t.id)
            if ti is not None and ti.track is t:
                out += ti.in_range(t0, t1)
        return out

    def snap_points(self, exclude_clip_id: str | None = None) -> list[float]:
        """Sorted start and end times of every clip except ``exclude_clip_id``. A fresh list the caller may keep (it is only valid for the ``revision`` it was taken at)."""
        ix = self.index()
        assert ix is not None
        if not ix.sane:
            return sorted(v for c in self.all_clips() if c.id != exclude_clip_id for v in (c.timeline_start, c.timeline_end))
        return ix.snap_points(exclude_clip_id)

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
        track = self.get_track(track_id)
        ix = self._fast()
        if ix is not None:
            ti = ix.tracks.get(track_id)
            if ti is not None and ti.track is track:
                if ti.overlaps(start, end, ignore_clip_id):
                    raise TimelineError("That would overlap another clip on the same track.")
                return
        for c in track.clips:
            if c.id == ignore_clip_id:
                continue
            if start < c.timeline_end - TIME_EPSILON and end > c.timeline_start + TIME_EPSILON:
                raise TimelineError("That would overlap another clip on the same track.")

    def first_free_start(self, track_id: str, start: float, duration: float) -> float:
        """Earliest start >= ``start`` where a clip of ``duration`` fits on the track."""
        t = max(0.0, start)
        track = self.get_track(track_id)
        ix = self._fast()
        if ix is not None and duration > 0:
            ti = ix.tracks.get(track_id)
            if ti is not None and ti.track is track and ti.ordered:
                return ti.first_free(t, duration)
        for c in track.clips:  # sorted by start
            if t + duration <= c.timeline_start + TIME_EPSILON:
                break
            if t < c.timeline_end - TIME_EPSILON:
                t = c.timeline_end
        return t

    def neighbours(self, clip: Clip) -> tuple[float, float | None]:
        """(end of previous clip or 0, start of next clip or None) on the clip's track."""
        prev_end, next_start = 0.0, None
        track = self.get_track(clip.track_id)
        ix = self._fast()
        if ix is not None:
            ti = ix.tracks.get(track.id)
            if ti is not None and ti.track is track:
                return ti.neighbours(clip)
        for c in track.clips:
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
        self.touch()

    def remove_track(self, track_id: str) -> tuple[Track, int]:
        track = self.get_track(track_id)
        index = self.tracks.index(track)
        self.tracks.remove(track)
        self.touch()
        return track, index

    def insert_clip(self, clip: Clip) -> None:
        track = self.get_track(clip.track_id)
        track.clips.append(clip)
        track.sort()
        self.touch()

    def detach_clip(self, clip_id: str) -> Clip:
        track, clip = self.find_clip(clip_id)
        for i, c in enumerate(track.clips):  # by identity: list.remove would run the dataclass __eq__ against every clip on the way
            if c is clip:
                del track.clips[i]
                break
        self.touch()
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
