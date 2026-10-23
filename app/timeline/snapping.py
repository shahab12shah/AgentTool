"""Qt-free snapping engine for timeline drags (clips, trims, markers, the playhead).

What it snaps to (``SnapSettings.targets``), highest priority first::

    playhead > markers > clip_edges > scenes > captions > range > sentences > words

  * ``clip_edges``  start / end of every non-caption clip, plus the timeline start (0:00). Locked clips' edges are valid targets.
  * ``captions``    start / end of ``kind == "caption"`` clips (kept apart from ``clip_edges`` because caption tracks hold hundreds of tiny clips).
  * ``markers``     marker times;  ``range``  the marked in / out;  ``scenes``  scene boundaries (``project.scenes``);
  * ``sentences`` / ``words``  spoken boundaries of the transcript (default OFF).

Design rules:
  * Tolerance is *frames* of the project timebase, converted to seconds and capped to ``MAX_SNAP_PX`` pixels so a zoomed-out timeline never grabs everything.
  * When several targets are within tolerance the highest priority kind wins; inside a kind the nearest target wins, then the earlier one.
  * Per-kind target arrays are sorted and built lazily, cached by what they depend on (timeline revision, marker / range / scene fingerprints, transcript list identity), so a
    mouse-move query is a handful of ``bisect`` calls and never scans the clips. Excluded ids (the clips being dragged) never snap to themselves.
  * A ``SnapSession`` adds hysteresis for the length of one drag: it keeps the last target until the pointer is ``STICKY_FACTOR`` times the tolerance away, so two close
    competing targets never make the snap point oscillate.
  * The engine only reports times; whether an off-grid target (a scene boundary at 3.217 s) is acceptable for picture work is the caller's decision (``frame_aligned``).
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, replace as _dc_replace
from operator import attrgetter
from typing import Any, Callable, Iterable

from app.timeline.timebase import Timebase

SNAP_KINDS: tuple[str, ...] = ("playhead", "markers", "clip_edges", "scenes", "captions", "range", "sentences", "words")  # priority order, best first
PRIORITY: dict[str, int] = {k: i for i, k in enumerate(SNAP_KINDS)}
TARGET_LABELS: dict[str, str] = {
    "playhead": "Playhead", "markers": "Markers", "clip_edges": "Clip edges", "scenes": "Scene boundaries",
    "captions": "Caption edges", "range": "In / out range", "sentences": "Sentence boundaries", "words": "Word boundaries",
}
DEFAULT_TARGETS: frozenset[str] = frozenset({"clip_edges", "playhead", "markers", "range", "scenes", "captions"})  # sentences / words are opt-in
TOLERANCE_CHOICES: tuple[int, ...] = (2, 4, 8)  # the menu offers these and "custom" (any value in MIN_TOLERANCE..MAX_TOLERANCE)
DEFAULT_TOLERANCE = 4
MIN_TOLERANCE, MAX_TOLERANCE = 1, 60
MAX_SNAP_PX = 30.0  # the effective tolerance never exceeds this many pixels
STICKY_FACTOR = 1.5  # hysteresis: a session keeps its target until the pointer is this many tolerances away
_EPS = 1e-9
_CANDIDATE_CAP = 64  # per kind, for ``candidates``


# ------------------------------------------------------------------------------------------------------------------------ settings
def _clamp_tolerance(v: Any) -> int:
    try:
        if isinstance(v, bool):
            raise ValueError
        n = int(round(float(v)))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_TOLERANCE
    return max(MIN_TOLERANCE, min(MAX_TOLERANCE, n))


@dataclass(frozen=True)
class SnapSettings:
    """Immutable snapping preferences (replace the whole object to change them, so listeners see the change)."""

    enabled: bool = True
    tolerance_frames: int = DEFAULT_TOLERANCE
    targets: frozenset[str] = DEFAULT_TARGETS

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "tolerance_frames", _clamp_tolerance(self.tolerance_frames))
        t = (self.targets,) if isinstance(self.targets, str) else (self.targets or ())
        object.__setattr__(self, "targets", frozenset(k for k in t if k in PRIORITY))  # unknown kinds are dropped

    def wants(self, kind: str) -> bool:
        return kind in self.targets

    def replace(self, **changes: Any) -> "SnapSettings":
        return _dc_replace(self, **changes)

    def with_target(self, kind: str, on: bool) -> "SnapSettings":
        return self.replace(targets=(self.targets | {kind}) if on else (self.targets - {kind}))

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "tolerance_frames": self.tolerance_frames, "targets": [k for k in SNAP_KINDS if k in self.targets]}

    @classmethod
    def from_dict(cls, d: Any) -> "SnapSettings":
        """Tolerant loader: anything missing or invalid falls back to the default."""
        if not isinstance(d, dict):
            return cls()
        targets = d.get("targets", DEFAULT_TARGETS)
        if not isinstance(targets, (list, tuple, set, frozenset)):
            targets = DEFAULT_TARGETS
        return cls(bool(d.get("enabled", True)), _clamp_tolerance(d.get("tolerance_frames", DEFAULT_TOLERANCE)), frozenset(k for k in targets if isinstance(k, str)))


# ------------------------------------------------------------------------------------------------------------------------ results
@dataclass(frozen=True)
class SnapResult:
    time: float  # where the pointer should be (the target's time when ``snapped``, otherwise the query time)
    target_kind: str = ""
    target_time: float | None = None
    distance_frames: float = 0.0  # how far the query was from the target (0 when not snapped)
    snapped: bool = False
    frame_aligned: bool = True  # ``time`` lies on the project's frame grid (an off-grid scene / word boundary is reported, the caller decides)
    target_id: str = ""  # clip / marker / scene / sentence / word id ("" for the timeline start, the playhead has "playhead")


@dataclass(frozen=True)
class SpanSnap:
    """A rigid span (one clip or the union span of a multi-clip drag) moved so that one of its edges lands on a target."""

    start: float
    end: float
    edge: str | None  # "start" | "end" | None (nothing snapped)
    result: SnapResult
    delta: float = 0.0  # how far the span moved (new start - requested start)

    @property
    def snapped(self) -> bool:
        return self.result.snapped

    @property
    def frame_aligned(self) -> bool:
        return self.result.frame_aligned if self.result.snapped else True


@dataclass(frozen=True)
class SnapCandidate:
    kind: str
    time: float
    distance_frames: float
    target_id: str = ""


# ------------------------------------------------------------------------------------------------------------------------ sorted target arrays
class _Targets:
    """Sorted target times with the id of the thing each belongs to (a clip may appear twice: start and end)."""

    __slots__ = ("times", "ids")

    def __init__(self, pairs: list[tuple[float, str]]) -> None:
        pairs = [p for p in pairs if math.isfinite(p[0])]
        pairs.sort()
        self.times: list[float] = [p[0] for p in pairs]
        self.ids: list[str] = [p[1] for p in pairs]

    def __len__(self) -> int:
        return len(self.times)

    def nearest(self, t: float, tol: float, exclude: Any) -> tuple[float, str] | None:
        """The nearest non-excluded target within ``tol`` of ``t`` (ties -> the earlier one), or None. O(log n) plus the excluded run next to ``t``."""
        times, ids = self.times, self.ids
        i = bisect_left(times, t)
        left = None
        j = i - 1
        while j >= 0 and t - times[j] <= tol + _EPS:
            if ids[j] not in exclude:
                left = j
                break
            j -= 1
        right = None
        k, n = i, len(times)
        while k < n and times[k] - t <= tol + _EPS:
            if ids[k] not in exclude:
                right = k
                break
            k += 1
        if left is None and right is None:
            return None
        if right is None or (left is not None and t - times[left] <= times[right] - t):
            return times[left], ids[left]  # type: ignore[index]
        return times[right], ids[right]

    def window(self, t: float, tol: float, exclude: Any, cap: int) -> list[tuple[float, str]]:
        lo, hi = bisect_left(self.times, t - tol - _EPS), bisect_right(self.times, t + tol + _EPS)
        out: list[tuple[float, str]] = []
        for x in range(lo, hi):
            if self.ids[x] not in exclude:
                out.append((self.times[x], self.ids[x]))
                if len(out) >= cap:
                    break
        return out

    def contains(self, time: float, target_id: str) -> bool:
        i = bisect_left(self.times, time)
        while i < len(self.times) and self.times[i] == time:
            if self.ids[i] == target_id:
                return True
            i += 1
        return False


_MARKER_KEY = attrgetter("time", "id")
_SCENE_KEY = attrgetter("id", "start", "end")


# ------------------------------------------------------------------------------------------------------------------------ engine
class SnapEngine:
    """``SnapEngine(get_timeline, get_project, get_playhead, get_settings, timebase_getter)`` - every dependency is a callable so the engine always sees the live state."""

    def __init__(
        self,
        get_timeline: Callable[[], Any],
        get_project: Callable[[], Any] | None = None,
        get_playhead: Callable[[], float] | None = None,
        get_settings: Callable[[], SnapSettings] | None = None,
        timebase_getter: Callable[[], Timebase] | None = None,
    ) -> None:
        self._get_timeline = get_timeline
        self._get_project = get_project or (lambda: None)
        self._get_playhead = get_playhead or (lambda: 0.0)
        default = SnapSettings()
        self._get_settings = get_settings or (lambda: default)
        self._get_timebase = timebase_getter or (lambda: Timebase(30))
        self._cache: dict[str, tuple[Any, Any, Any]] = {}  # name -> (owner, key, built value)
        self.builds: Counter[str] = Counter()  # how many times each target array was (re)built - tests and the performance benchmark read this

    # ----- plumbing -----
    @property
    def settings(self) -> SnapSettings:
        return self._get_settings()

    @property
    def timebase(self) -> Timebase:
        return self._get_timebase()

    def invalidate(self, kind: str | None = None) -> None:
        """Drop cached arrays (``None`` = all). Only needed for in-place edits the cache keys cannot see (e.g. transcript word times rewritten in place)."""
        if kind is None:
            self._cache.clear()
        else:
            self._cache.pop("clips" if kind in ("clip_edges", "captions") else kind, None)

    def session(self) -> "SnapSession":
        return SnapSession(self)

    def tolerance_seconds(self, pps: float) -> float:
        """Snap tolerance in seconds: ``tolerance_frames`` of the timebase, never more than ``MAX_SNAP_PX`` pixels at ``pps`` pixels per second."""
        s = self.timebase.frame_s * self.settings.tolerance_frames
        if pps and pps > 0 and math.isfinite(pps):
            s = min(s, MAX_SNAP_PX / pps)
        return s

    def tolerance_px(self, pps: float) -> float:
        return self.tolerance_seconds(pps) * pps

    # ----- target arrays (lazy, cached) -----
    def _cached(self, name: str, owner: Any, key: Any, build: Callable[[], Any]) -> Any:
        ent = self._cache.get(name)
        if ent is not None and ent[0] is owner and ent[1] == key:
            return ent[2]
        self.builds[name] += 1
        value = build()
        self._cache[name] = (owner, key, value)
        return value

    def _build_clip_pass(self, tl: Any) -> tuple[_Targets, _Targets]:
        edges: list[tuple[float, str]] = [(0.0, "")]
        caps: list[tuple[float, str]] = []
        for track in tl.tracks:
            for c in track.clips:
                dst = caps if c.kind == "caption" else edges
                dst.append((c.timeline_start, c.id))
                dst.append((c.timeline_start + c.duration, c.id))
        return _Targets(edges), _Targets(caps)

    def _targets(self, kind: str) -> _Targets | None:
        tl = self._get_timeline()
        if kind in ("clip_edges", "captions"):
            if tl is None:
                return None
            edges, caps = self._cached("clips", tl, tl.revision, lambda: self._build_clip_pass(tl))
            return edges if kind == "clip_edges" else caps
        if kind == "markers":
            if tl is None:
                return None
            ms = tl.markers
            return self._cached("markers", tl, tuple(map(_MARKER_KEY, ms)), lambda: _Targets([(m.time, m.id) for m in ms]))
        if kind == "range":
            if tl is None:
                return None
            r = tl.range
            return self._cached("range", tl, r, lambda: _Targets([(r[0], "range_in"), (r[1], "range_out")] if r else []))
        project = self._get_project()
        if project is None:
            return None
        if kind == "scenes":
            scenes = getattr(project, "scenes", None) or []
            return self._cached("scenes", project, tuple(map(_SCENE_KEY, scenes)), lambda: _Targets([p for s in scenes for p in ((s.start, s.id), (s.end, s.id))]))
        if kind in ("sentences", "words"):
            tr = getattr(getattr(project, "transcription", None), "transcript", None)
            items = getattr(tr, kind, None) if tr is not None else None
            if items is None:
                return None
            if kind == "sentences":
                return self._cached(kind, items, len(items), lambda: _Targets([p for s in items for p in ((s.start, s.sentence_id), (s.end, s.sentence_id))]))
            return self._cached(kind, items, len(items), lambda: _Targets([p for w in items for p in ((w.start, w.word_id), (w.end, w.word_id))]))
        return None

    def target_times(self, kind: str) -> list[float]:
        """A copy of the sorted target times of one kind (for drawing guides and for tests). The playhead is a single point."""
        if kind == "playhead":
            return [float(self._get_playhead())]
        tg = self._targets(kind)
        return list(tg.times) if tg is not None else []

    # ----- queries -----
    def _active_kinds(self, skip_kinds: Iterable[str], force: bool) -> tuple[str, ...]:
        s = self.settings
        if not (s.enabled or force):
            return ()
        skip = skip_kinds if isinstance(skip_kinds, (set, frozenset)) else frozenset(skip_kinds)
        return tuple(k for k in SNAP_KINDS if k in s.targets and k not in skip)

    def _best(self, t: float, tol: float, exclude: Any, kinds: tuple[str, ...]) -> tuple[str, float, str] | None:
        """The winning (kind, time, id): highest priority kind that has any target within tolerance, its nearest target."""
        for kind in kinds:
            if kind == "playhead":
                ph = float(self._get_playhead())
                if abs(ph - t) <= tol + _EPS:
                    return kind, ph, "playhead"
                continue
            tg = self._targets(kind)
            if tg is None or not tg.times:
                continue
            hit = tg.nearest(t, tol, exclude)
            if hit is not None:
                return kind, hit[0], hit[1]
        return None

    def _still_valid(self, kind: str, time: float, target_id: str, exclude: Any, kinds: tuple[str, ...]) -> bool:
        if kind not in kinds or (target_id in exclude and kind != "playhead"):
            return False
        if kind == "playhead":
            return abs(float(self._get_playhead()) - time) <= _EPS
        tg = self._targets(kind)
        return tg is not None and tg.contains(time, target_id)

    def _result(self, t: float, hit: tuple[str, float, str] | None) -> SnapResult:
        tb = self.timebase
        if hit is None:
            return SnapResult(t, frame_aligned=tb.is_on_frame(t) if math.isfinite(t) else True)
        kind, ttime, tid = hit
        return SnapResult(ttime, kind, ttime, abs(ttime - t) / tb.frame_s, True, tb.is_on_frame(ttime), tid)

    def _resolve_point(self, t: float, exclude: Any, tol: float, kinds: tuple[str, ...], sticky: tuple[str, float, str] | None) -> tuple[str, float, str] | None:
        if not kinds or not math.isfinite(t):
            return None
        if sticky is not None:
            kind, ttime, tid = sticky
            if abs(ttime - t) <= tol * STICKY_FACTOR + _EPS and self._still_valid(kind, ttime, tid, exclude, kinds):
                return sticky
        return self._best(t, tol, exclude, kinds)

    def snap(
        self, t: float, *, exclude_ids: Iterable[str] = frozenset(), pps: float, skip_kinds: Iterable[str] = frozenset(), force: bool = False,
    ) -> SnapResult:
        """Snap one time. ``exclude_ids``: clips / markers being dragged (never targets). ``skip_kinds``: e.g. ``{"playhead"}`` while the playhead itself is dragged.
        ``force`` ignores the on/off switch (the caller's temporary-override key)."""
        return self._snap(t, exclude_ids, pps, skip_kinds, force, None)[0]

    def _snap(self, t, exclude_ids, pps, skip_kinds, force, sticky):
        ex = exclude_ids if isinstance(exclude_ids, (set, frozenset, dict)) else frozenset(exclude_ids)
        kinds = self._active_kinds(skip_kinds, force)
        hit = self._resolve_point(t, ex, self.tolerance_seconds(pps), kinds, sticky)
        return self._result(t, hit), hit

    def snap_span(
        self, start: float, end: float, *, exclude_ids: Iterable[str] = frozenset(), pps: float, prefer: str = "nearest",
        skip_kinds: Iterable[str] = frozenset(), force: bool = False,
    ) -> SpanSnap:
        """Move the rigid span [start, end] so that one edge lands on a target. ``prefer``: ``"start"`` / ``"end"`` try that edge first and fall back to the other; ``"nearest"``
        (default) snaps by whichever edge is closer to a target (equal distance -> higher priority kind -> the start edge). A multi-clip drag passes the union span."""
        return self._snap_span(start, end, exclude_ids, pps, prefer, skip_kinds, force, None)[0]

    def _snap_span(self, start, end, exclude_ids, pps, prefer, skip_kinds, force, sticky):
        if prefer not in ("start", "end", "nearest"):
            raise ValueError("prefer must be 'start', 'end' or 'nearest'")
        ex = exclude_ids if isinstance(exclude_ids, (set, frozenset, dict)) else frozenset(exclude_ids)
        kinds = self._active_kinds(skip_kinds, force)
        tol = self.tolerance_seconds(pps)
        chosen: tuple[str, tuple[str, float, str]] | None = None
        if sticky is not None and kinds:
            edge, kind, ttime, tid = sticky
            pos = start if edge == "start" else end
            if abs(ttime - pos) <= tol * STICKY_FACTOR + _EPS and self._still_valid(kind, ttime, tid, ex, kinds):
                chosen = (edge, (kind, ttime, tid))
        if chosen is None and kinds:
            hs = self._best(start, tol, ex, kinds) if math.isfinite(start) else None
            he = self._best(end, tol, ex, kinds) if math.isfinite(end) else None
            if hs is not None and he is not None:
                ds, de = abs(hs[1] - start), abs(he[1] - end)
                if prefer == "start":
                    chosen = ("start", hs)
                elif prefer == "end":
                    chosen = ("end", he)
                elif ds < de - _EPS:
                    chosen = ("start", hs)
                elif de < ds - _EPS:
                    chosen = ("end", he)
                else:
                    chosen = ("end", he) if PRIORITY[he[0]] < PRIORITY[hs[0]] else ("start", hs)
            elif hs is not None:
                chosen = ("start", hs)
            elif he is not None:
                chosen = ("end", he)
        if chosen is None:
            return SpanSnap(start, end, None, self._result(start, None)), None
        edge, hit = chosen
        pos = start if edge == "start" else end
        delta = hit[1] - pos
        res = self._result(pos, hit)
        return SpanSnap(start + delta, end + delta, edge, res, delta), (edge, *hit)

    def candidates(
        self, t: float, *, exclude_ids: Iterable[str] = frozenset(), pps: float, skip_kinds: Iterable[str] = frozenset(), force: bool = False,
    ) -> list[SnapCandidate]:
        """Every target within tolerance of ``t`` (at most ``_CANDIDATE_CAP`` per kind), best first: priority, then distance, then time. ``snap`` returns the first one."""
        ex = exclude_ids if isinstance(exclude_ids, (set, frozenset, dict)) else frozenset(exclude_ids)
        kinds = self._active_kinds(skip_kinds, force)
        if not math.isfinite(t):
            return []
        tol = self.tolerance_seconds(pps)
        fs = self.timebase.frame_s
        out: list[SnapCandidate] = []
        for kind in kinds:
            if kind == "playhead":
                ph = float(self._get_playhead())
                if abs(ph - t) <= tol + _EPS:
                    out.append(SnapCandidate(kind, ph, abs(ph - t) / fs, "playhead"))
                continue
            tg = self._targets(kind)
            if tg is None:
                continue
            rows = sorted(((abs(x - t), x, i) for x, i in tg.window(t, tol, ex, _CANDIDATE_CAP)))
            out.extend(SnapCandidate(kind, x, d / fs, i) for d, x, i in rows)
        return out


class SnapSession:
    """Hysteresis for one drag: remembers the last target and keeps it until the pointer is ``STICKY_FACTOR`` x tolerance away from it. Create one per gesture (``engine.session()``)."""

    def __init__(self, engine: SnapEngine) -> None:
        self.engine = engine
        self._last: tuple[str, float, str] | None = None
        self._last_span: tuple[str, str, float, str] | None = None

    def reset(self) -> None:
        self._last = None
        self._last_span = None

    @property
    def last_target(self) -> tuple[str, float, str] | None:
        """(kind, time, id) the session is currently stuck to, or None."""
        return self._last

    def snap(self, t: float, *, exclude_ids: Iterable[str] = frozenset(), pps: float, skip_kinds: Iterable[str] = frozenset(), force: bool = False) -> SnapResult:
        res, hit = self.engine._snap(t, exclude_ids, pps, skip_kinds, force, self._last)
        self._last = hit
        return res

    def snap_span(
        self, start: float, end: float, *, exclude_ids: Iterable[str] = frozenset(), pps: float, prefer: str = "nearest",
        skip_kinds: Iterable[str] = frozenset(), force: bool = False,
    ) -> SpanSnap:
        res, hit = self.engine._snap_span(start, end, exclude_ids, pps, prefer, skip_kinds, force, self._last_span)
        self._last_span = hit
        return res
