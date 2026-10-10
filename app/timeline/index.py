"""Lazily built lookup structures behind ``Timeline`` (Phase 9).

Clips are mutated in place by many commands, so nothing here trusts a hand-maintained revision alone: a process-wide mutation epoch is bumped by every
write to a clip's geometry/identity, every mutation of a track's clip list and of the track list (``WatchedList``), and every ``Timeline`` mutator.
``Timeline`` rebuilds an index whose epoch (or cheap shape signature) no longer matches, and every lookup re-checks its answer against the live clip, so the
index can only ever be slower than a scan, never wrong.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import TYPE_CHECKING, Any, Iterable

from app.core.constants import TIME_EPSILON

if TYPE_CHECKING:  # pragma: no cover
    from app.timeline.clip import Clip
    from app.timeline.track import Track

_EPOCH = [0]
CLIP_WATCHED = frozenset({"id", "track_id", "asset_id", "timeline_start", "duration"})


def bump() -> None:
    _EPOCH[0] += 1


def epoch() -> int:
    return _EPOCH[0]


class WatchedList(list):
    """A list that bumps the mutation epoch on every structural change (used for ``Track.clips`` and ``Timeline.tracks``)."""

    __slots__ = ()

    def append(self, x):
        list.append(self, x)
        bump()

    def extend(self, xs):
        list.extend(self, xs)
        bump()

    def insert(self, i, x):
        list.insert(self, i, x)
        bump()

    def remove(self, x):
        list.remove(self, x)
        bump()

    def pop(self, *a):
        r = list.pop(self, *a)
        bump()
        return r

    def clear(self):
        list.clear(self)
        bump()

    def sort(self, *a, **k):
        list.sort(self, *a, **k)
        bump()

    def reverse(self):
        list.reverse(self)
        bump()

    def __setitem__(self, i, v):
        list.__setitem__(self, i, v)
        bump()

    def __delitem__(self, i):
        list.__delitem__(self, i)
        bump()

    def __iadd__(self, other):
        r = list.__iadd__(self, other)
        bump()
        return r

    def __imul__(self, n):
        r = list.__imul__(self, n)
        bump()
        return r


def watched(items: Iterable[Any]) -> WatchedList:
    return items if isinstance(items, WatchedList) else WatchedList(items)


class TrackIndex:
    """One track's clips sorted by start, with prefix-max ends so overlap / visible-range queries are bisects. ``sane`` is False for NaN/negative durations (the caller then scans)."""

    __slots__ = ("track", "clips", "starts", "ends", "maxend", "sorted_ends", "sorted_end_clips", "sane", "ordered")

    def __init__(self, track: "Track") -> None:
        self.track = track
        raw = list(track.clips)
        self.sane = all(math.isfinite(c.timeline_start) and math.isfinite(c.duration) and c.duration >= 0 for c in raw)
        clips = sorted(raw, key=_start) if self.sane else raw
        self.clips = clips
        self.ordered = all(a is b for a, b in zip(raw, clips))  # first_free_start walks the track's own order, so it only trusts the sorted copy when they agree
        self.starts = [c.timeline_start for c in clips]
        self.ends = [c.timeline_start + c.duration for c in clips]
        m = float("-inf")
        maxend: list[float] = []
        for e in self.ends:
            if e > m:
                m = e
            maxend.append(m)
        self.maxend = maxend
        order = sorted(range(len(clips)), key=self.ends.__getitem__)
        self.sorted_ends = [self.ends[i] for i in order]
        self.sorted_end_clips = [clips[i] for i in order]

    def overlaps(self, start: float, end: float, ignore_id: str | None) -> bool:
        """Same predicate as the linear ``check_free`` scan, evaluated only on the bisect-narrowed candidates."""
        hi = bisect_right(self.starts, end)
        lo = bisect_right(self.maxend, start)
        for i in range(lo, hi):
            c = self.clips[i]
            if c.id == ignore_id:
                continue
            if start < c.timeline_end - TIME_EPSILON and end > c.timeline_start + TIME_EPSILON:
                return True
        return False

    def in_range(self, t0: float, t1: float) -> list["Clip"]:
        """Clips with ``end > t0 and start < t1`` in start order."""
        hi = bisect_left(self.starts, t1)
        lo = bisect_right(self.maxend, t0)
        ends = self.ends
        return [self.clips[i] for i in range(lo, hi) if ends[i] > t0]

    def first_free(self, t: float, duration: float) -> float:
        """The sequential walk of ``Timeline.first_free_start``, started at the first clip that can still matter."""
        for i in range(bisect_right(self.maxend, t - TIME_EPSILON), len(self.clips)):
            c = self.clips[i]
            if t + duration <= c.timeline_start + TIME_EPSILON:
                break
            if t < c.timeline_end - TIME_EPSILON:
                t = c.timeline_end
        return t

    def neighbours(self, clip: "Clip") -> tuple[float, float | None]:
        prev_end: float = 0.0
        limit = clip.timeline_start + TIME_EPSILON
        j = bisect_right(self.sorted_ends, limit) - 1
        while j >= 0:
            if self.sorted_end_clips[j].id != clip.id:
                prev_end = max(prev_end, self.sorted_ends[j])
                break
            j -= 1
        next_start: float | None = None
        floor = clip.timeline_end - TIME_EPSILON
        for i in range(bisect_left(self.starts, floor), len(self.clips)):
            c = self.clips[i]
            if c.id == clip.id or c.timeline_end <= limit:
                continue  # the second case is the "previous clip" branch of the linear scan (only reachable for degenerate zero-length clips)
            next_start = c.timeline_start
            break
        return prev_end, next_start


def _start(c: "Clip") -> float:
    return c.timeline_start


class TimelineIndex:
    """Everything the canvas and the editing rules ask of a timeline, built in one pass."""

    __slots__ = ("epoch", "sig", "by_id", "by_asset", "tracks", "track_by_id", "duration", "points", "sane", "count")

    def __init__(self, tracks: list["Track"], epoch_: int, sig: tuple) -> None:
        self.epoch, self.sig = epoch_, sig
        self.by_id: dict[str, tuple["Track", "Clip"]] = {}
        self.by_asset: dict[str, list["Clip"]] = {}
        self.track_by_id: dict[str, "Track"] = {}
        self.tracks: dict[str, TrackIndex] = {}
        duration = 0.0
        points: list[float] = []
        sane = True
        count = 0
        for t in tracks:
            self.track_by_id.setdefault(t.id, t)
            ti = TrackIndex(t)
            sane = sane and ti.sane
            self.tracks.setdefault(t.id, ti)
            for c in t.clips:
                count += 1
                self.by_id.setdefault(c.id, (t, c))
                self.by_asset.setdefault(c.asset_id, []).append(c)
            if ti.ends:
                duration = max(duration, ti.maxend[-1])
            points += ti.starts
            points += ti.ends
        points.sort()
        self.points, self.duration, self.sane, self.count = points, duration, sane, count

    def snap_points(self, exclude_clip_id: str | None) -> list[float]:
        """Sorted start/end times of every clip except the (possibly several) with id ``exclude_clip_id``; always a fresh list."""
        pts = list(self.points)
        if exclude_clip_id:
            for t, c in self._all_with_id(exclude_clip_id):
                for v in (c.timeline_start, c.timeline_start + c.duration):
                    i = bisect_left(pts, v)
                    if i < len(pts) and pts[i] == v:
                        del pts[i]
        return pts

    def _all_with_id(self, clip_id: str):
        first = self.by_id.get(clip_id)
        if first is None:
            return
        # duplicates are not expected; a scan of the owning tracks keeps the answer identical to the old rebuild-from-all_clips
        for t in self.track_by_id.values():
            for c in t.clips:
                if c.id == clip_id:
                    yield t, c
