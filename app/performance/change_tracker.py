"""ProjectChangeTracker: turns every ``PROJECT_CHANGED`` event into "what exactly may be out of date now".

It subscribes to the bus, maps the command that was executed / undone / redone to the scenes, project domains and dependency ids it can affect, keeps a
bounded change log with a monotonically increasing revision, and invalidates only the cache entries that depend on those ids (``MediaCacheManager.invalidate_by_dependency``).
Consumers (incremental QC, preview, analysis skip) ask ``peek`` / ``consume`` for the changes since *their* last look; nothing is ever consumed for another consumer.

Principles:
* A command the tracker does not understand is never guessed narrow: it marks the whole timeline (every scene, the broad domains) and says ``unknown``.
* A scene change also dirties its previous / next scene (transitions, continuity, repetition and pacing read across the boundary) in ``ChangeSet.scene_ids``;
  ``direct_scene_ids`` holds the scenes the command itself touched.
* Per event the work is a few set operations and, for clip edits, a bisect over the scene boundaries: no hashing, no scan of the timeline (asset / track level
  edits scan only the clips of that asset / track). Repeated identical events (a drag) are merged in the log.
* The change log is only an optimisation. Anything that must be *right* (the export gate, which QC findings are reusable) still compares content hashes; a change
  that bypassed the bus is therefore at worst reported late by ``qc_staleness``, never certified as current.
"""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from app.core.events import EventBus, Topics
from app.logging.logger import get_logger, log_event
from app.performance.dependencies import TRANSCRIPT_DEP, DependencyGraph, asset_dep, scene_dep, settings_dep, timeline_dep

_log = get_logger(__name__)

# project domains (the first nine are the names QCContext.domain_hash knows; the rest are cross-cutting)
D_TIMELINE, D_SCENES, D_TRANSCRIPT, D_ASSETS, D_AUDIO, D_CAPTIONS, D_VISUAL, D_REFERENCE, D_RENDER = ("timeline", "scenes", "transcript", "assets", "audio", "captions", "visual", "reference", "render")
D_EDITING = "editing"  # the editing brief / style / locked decisions: read by every checker
D_SCRIPT = "script"
D_PACING = "pacing"  # project-wide pacing statistics (average shot length ...): any timing change moves them
D_QC_SETTINGS = "qc_settings"
QC_DOMAINS = (D_TIMELINE, D_SCENES, D_TRANSCRIPT, D_ASSETS, D_AUDIO, D_CAPTIONS, D_VISUAL, D_REFERENCE, D_RENDER, D_EDITING)
BROAD_TIMELINE = (D_TIMELINE, D_CAPTIONS, D_AUDIO, D_PACING)

QC_NEAR_SECONDS = 0.5  # QC looks at clips this close to a scene when it judges the scene (QCContext.scene_signature)
_EPS = 1e-3
_TRACK_CLIP_CAP = 4000  # a track edit scans its clips once; above this it is cheaper (and safer) to call every scene affected
LOG_LIMIT = 512


def scene_layout_dep(scene_id: str) -> str:
    """Dependency id for "what is placed in / around this scene" (clips, captions, transitions). ``scene_dep`` is the scene's own content (text, timing, claims)."""
    return f"layout:{scene_id}"


@dataclass(frozen=True)
class ChangeSet:
    scene_ids: frozenset[str] = frozenset()  # direct + neighbours (what a reader that looks across scene boundaries must re-examine)
    direct_scene_ids: frozenset[str] = frozenset()
    content_scene_ids: frozenset[str] = frozenset()  # scenes whose own content changed (text, timing, assignment), not just their surroundings
    domains: frozenset[str] = frozenset()
    deps: frozenset[str] = frozenset()
    revision: int = 0  # the tracker's revision when this was read
    since: int = 0
    epoch: int = 0
    all_scenes: bool = False  # every scene is affected (global change or something not understood)
    unknown: bool = False  # at least one command was not understood and was handled broadly

    @property
    def empty(self) -> bool:
        return not (self.scene_ids or self.domains or self.deps or self.all_scenes)

    def affects(self, scene_id: str) -> bool:
        return self.all_scenes or scene_id in self.scene_ids


@dataclass
class _Effect:
    direct: set[str] = field(default_factory=set)
    content: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    deps: set[str] = field(default_factory=set)
    all_scenes: bool = False
    unknown: bool = False
    neigh: set[str] = field(default_factory=set)

    def merge(self, o: "_Effect") -> None:
        self.direct |= o.direct
        self.content |= o.content
        self.domains |= o.domains
        self.deps |= o.deps
        self.neigh |= o.neigh
        self.all_scenes = self.all_scenes or o.all_scenes
        self.unknown = self.unknown or o.unknown

    @property
    def empty(self) -> bool:
        return not (self.direct or self.domains or self.deps or self.all_scenes or self.neigh)


@dataclass
class _Entry:
    first: int
    last: int
    direct: set[str]
    content: set[str]
    neigh: set[str]
    domains: set[str]
    deps: set[str]
    all_scenes: bool
    unknown: bool

    def covers(self, e: _Effect) -> bool:
        return (e.direct <= self.direct and e.content <= self.content and e.neigh <= self.neigh and e.domains <= self.domains and e.deps <= self.deps
                and (self.all_scenes or not e.all_scenes) and (self.unknown or not e.unknown))

    def absorb(self, e: _Effect, rev: int) -> None:
        self.last = rev

    def union(self, o: "_Entry") -> None:
        self.first, self.last = min(self.first, o.first), max(self.last, o.last)
        self.direct |= o.direct
        self.content |= o.content
        self.neigh |= o.neigh
        self.domains |= o.domains
        self.deps |= o.deps
        self.all_scenes = self.all_scenes or o.all_scenes
        self.unknown = self.unknown or o.unknown


class _SceneIndex:
    """Scenes by start time, for "which scenes does this time range touch" in O(log n). Rebuilt when the scene list, any scene object or any scene's timing differs."""

    def __init__(self, scenes: list) -> None:
        self.src = list(scenes)
        self.bounds = [(s.start, s.end) for s in scenes]
        self.scenes = sorted(scenes, key=lambda s: (s.start, s.end))
        self.starts = [s.start for s in self.scenes]
        self.ids = [s.id for s in self.scenes]
        self.pos = {s.id: i for i, s in enumerate(self.scenes)}
        self.max_len = max([s.end - s.start for s in self.scenes] + [0.0])

    def fresh_for(self, scenes: list) -> bool:
        return len(scenes) == len(self.src) and all(a is b for a, b in zip(self.src, scenes)) and [(s.start, s.end) for s in scenes] == self.bounds

    def overlapping(self, t0: float, t1: float) -> list[str]:
        if not self.scenes:
            return []
        lo = bisect.bisect_left(self.starts, t0 - self.max_len - _EPS)
        hi = bisect.bisect_right(self.starts, t1 + _EPS)
        return [s.id for s in self.scenes[lo:hi] if s.end > t0 - _EPS and s.start < t1 + _EPS]

    def neighbours(self, sid: str, hops: int = 1) -> list[str]:
        i = self.pos.get(sid)
        if i is None:
            return []
        return [self.ids[j] for j in range(max(0, i - hops), min(len(self.ids), i + hops + 1)) if j != i]


class ProjectChangeTracker:
    def __init__(self, bus: EventBus | None, project_getter: Callable[[], Any], cache_manager_getter: Callable[[], Any] | None = None, graph: DependencyGraph | None = None,
                 *, log_limit: int = LOG_LIMIT) -> None:
        self._bus, self._project, self._cache = bus, project_getter, cache_manager_getter or (lambda: None)
        self.graph = graph if graph is not None else DependencyGraph()
        self.graph.set_neighbours(self._dep_neighbours)
        self._lock = threading.RLock()
        self._log: list[_Entry] = []
        self._limit = max(8, int(log_limit))
        self._revision = 0
        self._epoch = 0
        self._cursors: dict[str, int] = {}
        self._index: _SceneIndex | None = None
        self._warned: set[str] = set()
        self._unsubs: list[Callable[[], None]] = []
        self.events_seen = 0
        self._validated_event = -1  # the event during which the scene index was last checked against the live scenes (checked once per event, not per lookup)
        self.invalidated = 0  # cache entries dropped through dependency invalidation
        if bus is not None:
            self._unsubs = [bus.subscribe(Topics.PROJECT_CHANGED, self._on_changed), bus.subscribe(Topics.PROJECT_OPENED, self._on_reset), bus.subscribe(Topics.PROJECT_CLOSED, self._on_reset)]

    def close(self) -> None:
        for u in self._unsubs:
            u()
        self._unsubs = []

    # ------------------------------------------------------------------ reading
    @property
    def revision(self) -> int:
        return self._revision

    @property
    def epoch(self) -> int:
        """Changes on every project open / close: a revision from another epoch says nothing about this project."""
        return self._epoch

    def cursor(self, consumer: str = "default") -> int:
        return self._cursors.get(consumer, 0)

    @property
    def dirty_scenes(self) -> frozenset[str]:
        return self.peek().scene_ids

    @property
    def dirty_domains(self) -> frozenset[str]:
        return self.peek().domains

    def peek(self, since: int | None = None, consumer: str = "default") -> ChangeSet:
        """Everything that changed after revision ``since`` (default: after this consumer's last ``consume`` / ``mark_clean``). Never changes the tracker."""
        with self._lock:
            s = self._cursors.get(consumer, 0) if since is None else since
            return self._collect(s)

    def consume(self, consumer: str = "default") -> ChangeSet:
        """``peek`` and move this consumer's cursor to now."""
        with self._lock:
            cs = self._collect(self._cursors.get(consumer, 0))
            self._cursors[consumer] = self._revision
            return cs

    def mark_clean(self, consumer: str = "default", revision: int | None = None) -> None:
        """This consumer has dealt with everything up to ``revision`` (default: now). Changes that arrived later stay pending."""
        with self._lock:
            r = self._revision if revision is None else min(int(revision), self._revision)
            self._cursors[consumer] = max(self._cursors.get(consumer, 0), r)

    def _collect(self, since: int) -> ChangeSet:
        direct: set[str] = set()
        content: set[str] = set()
        neigh: set[str] = set()
        domains: set[str] = set()
        deps: set[str] = set()
        all_scenes = unknown = False
        for e in self._log:
            if e.last <= since:
                continue
            direct |= e.direct
            content |= e.content
            neigh |= e.neigh
            domains |= e.domains
            deps |= e.deps
            all_scenes, unknown = all_scenes or e.all_scenes, unknown or e.unknown
        return ChangeSet(frozenset(direct | neigh), frozenset(direct), frozenset(content), frozenset(domains), frozenset(deps), self._revision, since, self._epoch, all_scenes, unknown)

    # ------------------------------------------------------------------ neighbours
    def neighbours(self, scene_id: str, hops: int = 1) -> list[str]:
        with self._lock:
            self._validated_event = -1
            idx = self._scene_index()
            return idx.neighbours(scene_id, hops) if idx else []

    def _dep_neighbours(self, dep_id: str) -> list[str]:
        if not dep_id.startswith("scene:"):
            return []
        return [scene_dep(n) for n in self.neighbours(dep_id[6:])]

    def _scene_index(self) -> _SceneIndex | None:
        p = self._project()
        if p is None:
            self._index = None
            return None
        if self._index is not None and self._validated_event == self.events_seen:
            return self._index
        scenes = p.scenes
        if self._index is None or not self._index.fresh_for(scenes):
            self._index = _SceneIndex(scenes)
        self._validated_event = self.events_seen
        return self._index

    # ------------------------------------------------------------------ events
    def _on_reset(self, _topic: str, _payload: dict) -> None:
        with self._lock:
            self._epoch += 1
            self._revision += 1
            self._log.clear()
            self._cursors.clear()
            self._index = None

    def _on_changed(self, _topic: str, payload: dict) -> None:
        self.events_seen += 1
        cmd = payload.get("command")
        scope = str(payload.get("scope") or getattr(cmd, "scope", "") or "")
        try:
            with self._lock:
                if scope in ("scenes",) or self._index is None:
                    self._index = None
                self._validated_event = -1
                eff = self._effect_of(cmd, scope)
                self._validated_event = -1
        except Exception:  # noqa: BLE001  (a mapping bug must widen the change, never lose it)
            _log.warning("change mapping failed for %s", type(cmd).__name__, exc_info=True)
            eff = self._broad(scope, unknown=True)
        self.record(eff)

    def mark(self, *, scene_ids: Iterable[str] = (), domains: Iterable[str] = (), deps: Iterable[str] = (), all_scenes: bool = False, content: bool = False) -> None:
        """For code that changes the project without a command (or learns of an external change, e.g. a file changed on disk)."""
        eff = _Effect(direct=set(scene_ids), domains=set(domains), deps=set(deps), all_scenes=all_scenes)
        if content:
            eff.content = set(eff.direct)
        with self._lock:
            self._validated_event = -1
            self._add_neighbours(eff)
        self.record(eff)

    def record(self, eff: _Effect) -> None:
        if eff.empty:
            return
        with self._lock:
            self._revision += 1
            rev = self._revision
            if self._log and self._log[-1].covers(eff):
                self._log[-1].absorb(eff, rev)  # a drag repeats the same change: one log entry
            else:
                self._log.append(_Entry(rev, rev, set(eff.direct), set(eff.content), set(eff.neigh), set(eff.domains), set(eff.deps), eff.all_scenes, eff.unknown))
                if len(self._log) > self._limit:
                    old = self._log[: self._limit // 2]
                    merged = old[0]
                    for o in old[1:]:
                        merged.union(o)  # a superset: a reader behind the merge sees more than it needs, never less
                    self._log[: self._limit // 2] = [merged]
            dep_ids = set(eff.deps) | {scene_dep(s) for s in eff.content} | {scene_layout_dep(s) for s in eff.direct}
            if eff.unknown:
                dep_ids.add("scene:*")
            if eff.all_scenes and (eff.unknown or D_TIMELINE in eff.domains):
                dep_ids.add("layout:*")  # the clips / captions around every scene may differ
        self._invalidate(dep_ids)

    def _invalidate(self, dep_ids: set[str]) -> None:
        cache = self._cache()
        if cache is None or not dep_ids:
            return
        n = 0
        try:
            for d in self.graph.affected_many(d for d in dep_ids if not d.endswith(":*")):  # what was registered as depending on these ids, transitively
                n += int(cache.invalidate_by_dependency(d) or 0)
            for d in dep_ids:
                if d.endswith(":*"):
                    n += int(cache.invalidate_by_dependency_prefix(d[:-1]) or 0)
        except Exception:  # noqa: BLE001
            _log.warning("dependency invalidation failed", exc_info=True)
        if n:
            self.invalidated += n
            log_event(_log, "perf.cache_invalidated", entries=n, deps=len(dep_ids))

    # ------------------------------------------------------------------ mapping: command -> effect
    def _broad(self, scope: str, unknown: bool) -> _Effect:
        e = _Effect(all_scenes=True, unknown=unknown)
        if scope in ("timeline", "editing", "meta", "") or unknown:
            e.domains |= set(BROAD_TIMELINE)
        if unknown and scope not in ("timeline",):
            e.domains |= set(QC_DOMAINS)  # nothing is known about it: everything QC reads may have moved
        e.deps.add(timeline_dep("*"))
        return e

    def _add_neighbours(self, eff: _Effect) -> None:
        if eff.all_scenes or not eff.direct:
            return
        idx = self._scene_index()
        if idx is None:
            return
        for sid in list(eff.direct):
            eff.neigh.update(idx.neighbours(sid))
        eff.neigh -= eff.direct

    def _effect_of(self, cmd: Any, scope: str) -> _Effect:
        eff = self._effect_for_command(cmd, scope)
        self._add_neighbours(eff)
        return eff

    def _effect_for_command(self, cmd: Any, scope: str) -> _Effect:
        if cmd is None:
            return self._unknown(None, scope)
        subs = getattr(cmd, "_commands", None)
        if isinstance(subs, (list, tuple)):  # CompositeCommand: the union of its parts (the composite's own scope is only a label)
            out = _Effect()
            for c in subs:
                out.merge(self._effect_for_command(c, str(getattr(c, "scope", "") or scope)))
            return out
        for klass in type(cmd).__mro__:
            h = _HANDLERS.get(klass.__name__)
            if h is not None:
                return h(self, cmd)
        return self._unknown(cmd, scope)

    def _unknown(self, cmd: Any, scope: str) -> _Effect:
        name = type(cmd).__name__ if cmd is not None else "None"
        if name not in self._warned and len(self._warned) < 64:
            self._warned.add(name)
            log_event(_log, "perf.change_unknown_command", command=name, scope=scope)
        if scope in _SCOPE_NO_EFFECT:
            return _Effect()
        return self._broad(scope, unknown=True)

    # ---- helpers used by the handlers
    def _scenes_in_range(self, t0: float, t1: float, margin: float = 0.0) -> set[str]:
        idx = self._scene_index()
        if idx is None:
            return set()
        a, b = t0 - margin + 2 * _EPS, t1 + margin - 2 * _EPS  # touching a boundary is not overlapping it
        if b <= a:
            a = b = (t0 + t1) / 2
        return set(idx.overlapping(a, b))

    def _range_into(self, e: _Effect, scene_id: str, t0: float, t1: float) -> None:
        """A clip's footprint: its own scene(s) are affected directly; the scenes within QC's look-around distance are affected through their surroundings."""
        if scene_id:
            e.direct.add(scene_id)
        e.direct |= self._scenes_in_range(t0, t1)
        e.neigh |= self._scenes_in_range(t0, t1, QC_NEAR_SECONDS)

    def _clips_effect(self, clips: Iterable[Any]) -> _Effect:
        e = _Effect(domains={D_TIMELINE, D_PACING})
        seen = 0
        for c in clips:
            if c is None:
                continue
            seen += 1
            self._range_into(e, c.scene_id, c.timeline_start, c.timeline_start + c.duration)
            e.deps.add(timeline_dep(c.track_id))
            if c.kind == "caption":
                e.domains.add(D_CAPTIONS)
        if not seen:
            return self._broad("timeline", unknown=True)
        idx = self._scene_index()
        if idx is not None:
            e.direct &= set(idx.ids)
            e.neigh &= set(idx.ids)
        return e

    def _clips_of(self, cmd: Any) -> list[Any]:
        """Every clip state a clip command carries (before / after / the added or removed one), plus the live clip by id."""
        out = []
        for name in ("clip", "_removed", "_before", "_after", "_left", "_right"):
            v = getattr(cmd, name, None)
            if v is not None and hasattr(v, "timeline_start"):
                out.append(v)
        for name in ("_clips",):
            v = getattr(cmd, name, None)
            if isinstance(v, list):
                out.extend(c for c in v if hasattr(c, "timeline_start"))
        p = self._project()
        for name in ("clip_id", "new_clip_id"):
            cid = getattr(cmd, name, None)
            if cid and p is not None:
                live = p.timeline.get_clip(cid)
                if live is not None:
                    out.append(live)
        return out

    def _track_effect(self, track_id: str, track: Any = None) -> _Effect:
        p = self._project()
        t = track
        if t is None and p is not None:
            try:
                t = p.timeline.get_track(track_id)
            except Exception:  # noqa: BLE001
                t = None
        e = _Effect(domains={D_TIMELINE, D_AUDIO, D_CAPTIONS, D_PACING}, deps={timeline_dep(track_id)})
        if t is None:
            e.all_scenes = True
            return e
        if len(t.clips) > _TRACK_CLIP_CAP:
            e.all_scenes = True
            return e
        for c in t.clips:
            self._range_into(e, c.scene_id, c.timeline_start, c.timeline_start + c.duration)
        return e

    def _asset_effect(self, asset_id: str, cmd: Any = None, domains: Iterable[str] = (D_ASSETS,)) -> _Effect:
        e = _Effect(domains=set(domains), deps={asset_dep(asset_id)})
        p = self._project()
        clips = list(self._clips_of(cmd)) if cmd is not None else []
        if p is not None:
            clips += list(p.timeline.clips_for_asset(asset_id))
            e.direct |= {sid for sid, a in p.visual_assignments.items() if getattr(a, "asset_id", None) == asset_id}
            if p.voice_over.asset_id == asset_id:
                e.all_scenes = True
                e.domains |= {D_AUDIO, D_TRANSCRIPT}
                e.deps.add(TRANSCRIPT_DEP)
        for c in clips:
            self._range_into(e, c.scene_id, c.timeline_start, c.timeline_start + c.duration)
        if clips:
            e.domains |= {D_TIMELINE, D_PACING}
        return e

    def _scene_ids_effect(self, ids: Iterable[str], domains: Iterable[str], content: bool = True) -> _Effect:
        e = _Effect(domains=set(domains), direct={i for i in ids if i})
        if content:
            e.content = set(e.direct)
        return e


def _h_add_clip(t: ProjectChangeTracker, c: Any) -> _Effect:
    return t._clips_effect(t._clips_of(c))


def _h_track(t: ProjectChangeTracker, c: Any) -> _Effect:
    track = getattr(c, "track", None)
    removed = getattr(c, "_removed", None)
    if removed is not None and isinstance(removed, tuple):
        track = removed[0]
    tid = getattr(c, "track_id", None) or (track.id if track is not None else "")
    return t._track_effect(tid, track)


def _h_track_rename(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(domains={D_TIMELINE}, deps={timeline_dep(c.track_id)})


_STATE_CLIP_CAP = 20000


def _briefs_sig(strategy: Any) -> dict:
    return {sid: (bool(getattr(b, "keep_static", False)), bool(getattr(b, "evidence_treatment_needed", False))) for sid, b in (getattr(strategy, "briefs", None) or {}).items()} if strategy else {}


def _h_state_install(t: ProjectChangeTracker, c: Any) -> _Effect:
    """ApplyEditCommand / ApplyPresentationCommand: a whole candidate state replaces the timeline. The scenes it touches are found by comparing the clips of the state before and after
    (equality only, no hashing); the parts of the state that QC reads globally (caption settings, ducking, locks, the editing brief) widen the change to every scene."""
    before, after = getattr(c, "_before", None), getattr(c, "after", None)
    if before is None or after is None:
        return t._broad("timeline", unknown=True)
    tb, ta = {k.id: k for k in before.timeline.tracks}, {k.id: k for k in after.timeline.tracks}
    cb = {cl.id: cl for tr in before.timeline.tracks for cl in tr.clips}
    ca = {cl.id: cl for tr in after.timeline.tracks for cl in tr.clips}
    if len(cb) + len(ca) > 2 * _STATE_CLIP_CAP:
        return t._broad("timeline", unknown=False)
    e = _Effect(domains={D_TIMELINE, D_PACING})
    changed = [cl for i, cl in ca.items() if cb.get(i) != cl] + [cl for i, cl in cb.items() if i not in ca or ca[i] != cl]
    if changed:
        e.merge(t._clips_effect(changed))
    for tid in set(tb) | set(ta):
        x, y = tb.get(tid), ta.get(tid)
        if x is None or y is None or (x.kind, x.hidden, x.muted, x.locked, x.solo, round(x.volume, 6)) != (y.kind, y.hidden, y.muted, y.locked, y.solo, round(y.volume, 6)):
            e.merge(t._track_effect(tid, y or x))
    wide = False
    if getattr(before, "caption_settings", None) != getattr(after, "caption_settings", None):
        e.domains.add(D_CAPTIONS)
        wide = True
    if getattr(before, "ducking_events", None) != getattr(after, "ducking_events", None):
        e.domains.add(D_AUDIO)
        wide = True
    locked = lambda st: ({k for k, d in (getattr(st, "decisions", None) or {}).items() if getattr(d, "locked", False)}, sorted(getattr(getattr(st, "generation", None), "locked_scenes", []) or []))  # noqa: E731
    if locked(before) != locked(after) or _briefs_sig(getattr(before, "strategy", None)) != _briefs_sig(getattr(after, "strategy", None)):
        e.domains.add(D_EDITING)
        wide = True
    e.all_scenes = e.all_scenes or wide
    return e


def _h_clip_flag(t: ProjectChangeTracker, c: Any) -> _Effect:  # ownership marks: the clip's scene
    clip = t._project().timeline.get_clip(c.clip_id) if t._project() is not None else None
    if clip is not None:
        return t._clips_effect([clip])
    sid = getattr(c, "scene_id", "")
    if sid:
        return t._scene_ids_effect([sid], (D_TIMELINE, D_EDITING), content=False)
    return t._broad("timeline", unknown=True)


def _h_set_setting(t: ProjectChangeTracker, c: Any) -> _Effect:
    attr = str(getattr(c, "attr", ""))
    dom = {"audio_settings": {D_AUDIO}, "audio_processing": {D_AUDIO}, "caption_settings": {D_CAPTIONS}, "caption_styles": {D_CAPTIONS}}.get(attr)
    if dom is None:
        return t._broad("editing", unknown=True)
    return _Effect(all_scenes=True, domains=dom | {D_TIMELINE}, deps={settings_dep(attr)})


def _h_editing_settings(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(all_scenes=True, domains={D_EDITING}, deps={settings_dep("editing")})


def _h_set_script(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(domains={D_SCRIPT}, deps={TRANSCRIPT_DEP})


def _h_add_asset(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(domains={D_ASSETS}, deps={asset_dep(c.asset.id)})


def _h_asset_by_id(t: ProjectChangeTracker, c: Any) -> _Effect:
    return t._asset_effect(c.asset_id, c)


def _h_voice_over(t: ProjectChangeTracker, c: Any) -> _Effect:  # the narration is the master clock: every scene is judged against it
    return _Effect(all_scenes=True, domains={D_AUDIO, D_TRANSCRIPT, D_ASSETS, D_TIMELINE, D_PACING}, deps={TRANSCRIPT_DEP})


def _h_transcript(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(all_scenes=True, domains={D_TRANSCRIPT, D_PACING}, deps={TRANSCRIPT_DEP})


def _h_set_scenes(t: ProjectChangeTracker, c: Any) -> _Effect:
    p = t._project()
    return _Effect(all_scenes=True, domains={D_SCENES, D_VISUAL, D_PACING}, deps={scene_dep(s.id) for s in (p.scenes if p else [])} | {scene_dep(s.id) for s in getattr(c, "scenes", [])})


def _h_replace_scenes(t: ProjectChangeTracker, c: Any) -> _Effect:
    ids = list(getattr(c, "old_ids", [])) + [s.id for s in getattr(c, "new_scenes", [])]
    e = t._scene_ids_effect(ids, (D_SCENES, D_VISUAL, D_PACING))
    e.deps |= {scene_dep(i) for i in ids}
    return e


def _h_visual_preferences(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(all_scenes=True, domains={D_VISUAL}, deps={settings_dep("visual_preferences")})


def _h_research(t: ProjectChangeTracker, c: Any) -> _Effect:
    return t._scene_ids_effect([c.scene_id], (D_VISUAL,))


def _h_reference(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(all_scenes=True, domains={D_REFERENCE, D_PACING}, deps={settings_dep("reference")})


def _h_render_settings(t: ProjectChangeTracker, c: Any) -> _Effect:  # resolution / fps / codec change every preview section and QC's render checks
    return _Effect(all_scenes=True, domains={D_RENDER, D_EDITING}, deps={settings_dep("render")})


def _h_proxy(t: ProjectChangeTracker, c: Any) -> _Effect:
    # a proxy record changing never changes the ORIGINAL media: the asset's own dependants (thumbnails, probe results, waveforms) stay valid. Only render-side state follows.
    return _Effect(domains={D_RENDER}, deps={settings_dep("proxies"), f"proxy:{c.asset_id}"})


def _h_qc_settings(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect(domains={D_QC_SETTINGS}, deps={settings_dep("qc")})


def _h_nothing(t: ProjectChangeTracker, c: Any) -> _Effect:
    return _Effect()


_SCOPE_NO_EFFECT = frozenset({"qc", "performance", "waveform", "proxies_ui"})
# class name -> handler; subclasses are matched through their MRO. A class that is missing here falls back to a broad, "unknown" effect.
_HANDLERS: dict[str, Callable[[ProjectChangeTracker, Any], _Effect]] = {
    "AddTrackCommand": _h_track, "RemoveTrackCommand": _h_track, "RenameTrackCommand": _h_track_rename, "SetTrackFlagCommand": _h_track, "SetTrackVolumeCommand": _h_track,
    "AddClipCommand": _h_add_clip, "DeleteClipCommand": _h_add_clip, "MoveClipCommand": _h_add_clip, "TrimClipCommand": _h_add_clip, "SetClipPropertiesCommand": _h_add_clip,
    "SplitClipCommand": _h_add_clip,
    "ApplyEditCommand": _h_state_install, "ApplyPresentationCommand": _h_state_install,
    "SetEditingSettingsCommand": _h_editing_settings, "SetSettingCommand": _h_set_setting,
    "MarkUserEditCommand": _h_clip_flag, "RecordUserDeleteCommand": _h_clip_flag, "MarkPresentationEditCommand": _h_clip_flag, "RecordPresentationDeleteCommand": _h_clip_flag,
    "SetScriptCommand": _h_set_script, "AddAssetCommand": _h_add_asset, "RemoveAssetCommand": _h_asset_by_id, "SetAssetExtraCommand": _h_asset_by_id, "RelinkAssetCommand": _h_asset_by_id,
    "SetVoiceOverCommand": _h_voice_over,
    "ApplyTranscriptionCommand": _h_transcript, "SetTranscriptionFailureCommand": _h_nothing, "SetAlignmentCommand": _h_transcript,
    "SetScenesCommand": _h_set_scenes, "ReplaceScenesCommand": _h_replace_scenes, "SetVisualPreferencesCommand": _h_visual_preferences,
    "ApplyResearchCommand": _h_research, "SceneDecisionCommand": _h_research,
    "RegisterReferenceCommand": _h_reference, "RemoveReferenceCommand": _h_reference, "StoreAnalysisCommand": _h_reference, "SetReferenceSettingsCommand": _h_reference, "ApplyStyleCommand": _h_reference,
    "SetRenderSettingsCommand": _h_render_settings, "RenderRecordCommand": _h_nothing, "DeleteRenderRecordCommand": _h_nothing, "SetProxyRecordCommand": _h_proxy,
    "SetQCSettingsCommand": _h_qc_settings, "StoreQCRunCommand": _h_nothing, "IgnoreIssuesCommand": _h_nothing, "UnignoreCommand": _h_nothing, "MarkIssueFixedCommand": _h_nothing,
    "StoreRenderQCCommand": _h_nothing, "SetPerformanceOverridesCommand": _h_nothing,
}
