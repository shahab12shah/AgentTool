"""Dependency vocabulary for cache invalidation: file fingerprints, stable cache keys, dependency ids and a tiny dependency graph.

A cache entry records ``{dep_id: fingerprint}``; when the current fingerprint of any dependency differs the entry is stale. ``DependencyGraph`` lets
later phases ask "what else is affected when this scene / asset / setting changes" (incremental QC and preview invalidation).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import threading
from collections import deque
from pathlib import PurePath
from typing import Any, Callable, Iterable

TRANSCRIPT_DEP = "transcript"
FINGERPRINT_MODES = ("stat", "quick", "full")
_QUICK_BLOCK = 64 * 1024
_CASEFOLD = os.name == "nt"  # Windows paths compare case-insensitively


def asset_dep(asset_id: str) -> str:
    return f"asset:{asset_id}"


def scene_dep(scene_id: str) -> str:
    return f"scene:{scene_id}"


def settings_dep(name: str) -> str:
    return f"settings:{name}"


def timeline_dep(track_id: str) -> str:
    return f"timeline:{track_id}"


def file_fingerprint(path: str | os.PathLike, mode: str = "stat") -> str:
    """Cheap identity of a file. ``stat``: size + mtime; ``quick``: stat + hash of the first and last 64 KiB; ``full``: streaming sha1 of the content (opt-in, slow)."""
    if mode not in FINGERPRINT_MODES:
        raise ValueError(f"unknown fingerprint mode: {mode!r}")
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    stat_part = f"{st.st_size}:{st.st_mtime_ns}"
    if mode == "stat":
        return f"s:{stat_part}"
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            if mode == "quick":
                h.update(fh.read(_QUICK_BLOCK))
                if st.st_size > 2 * _QUICK_BLOCK:
                    fh.seek(-_QUICK_BLOCK, os.SEEK_END)
                    h.update(b"|")
                    h.update(fh.read(_QUICK_BLOCK))
                elif st.st_size > _QUICK_BLOCK:
                    h.update(fh.read(_QUICK_BLOCK))
                return f"q:{stat_part}:{h.hexdigest()}"
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
            return f"f:{st.st_size}:{h.hexdigest()}"
    except OSError:
        return "unreadable"


def normalize_path(p: str | os.PathLike, *, casefold: bool | None = None) -> str:
    """Forward slashes, no trailing slash; lower-cased on Windows (or when ``casefold`` is forced)."""
    s = (p.as_posix() if isinstance(p, PurePath) else str(p).replace("\\", "/")).rstrip("/") or "/"
    return s.casefold() if (_CASEFOLD if casefold is None else casefold) else s


def _canon(x: Any) -> Any:
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        r = round(x, 6) + 0.0  # + 0.0 turns -0.0 into 0.0
        return int(r) if r == int(r) and abs(r) < 1e15 else r
    if isinstance(x, PurePath):
        return normalize_path(x)
    if isinstance(x, enum.Enum):
        return _canon(x.value)
    if isinstance(x, (bytes, bytearray)):
        return "b:" + bytes(x).hex()
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _canon(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (set, frozenset)):
        return sorted((_canon(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return str(x)


def stable_key(category: str, *parts: Any) -> str:
    """sha1 of a canonical JSON of (category, parts): independent of dict ordering, float noise below 1e-6 and path separators."""
    blob = json.dumps([category, _canon(list(parts))], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(blob.encode("ascii")).hexdigest()


class DependencyGraph:
    """``add_edge(A, B)`` means "B depends on A". ``affected(A)`` is everything that must be recomputed when A changes.

    ``neighbours(dep_id)`` is an optional callback (e.g. the previous/next scene ids) for results that read across a boundary:
    ``affected(x, include_neighbours=1)`` also treats x's neighbours (and what depends on them) as changed.
    """

    def __init__(self, neighbours: Callable[[str], Iterable[str]] | None = None) -> None:
        self._out: dict[str, set[str]] = {}
        self._in: dict[str, set[str]] = {}
        self._neighbours = neighbours
        self._lock = threading.RLock()

    def set_neighbours(self, fn: Callable[[str], Iterable[str]] | None) -> None:
        self._neighbours = fn

    def add_edge(self, source: str, dependant: str) -> None:
        with self._lock:
            self._out.setdefault(source, set()).add(dependant)
            self._in.setdefault(dependant, set()).add(source)

    def add_dependency(self, dependant: str, *sources: str) -> None:
        for s in sources:
            self.add_edge(s, dependant)

    def remove_edge(self, source: str, dependant: str) -> None:
        with self._lock:
            self._out.get(source, set()).discard(dependant)
            self._in.get(dependant, set()).discard(source)

    def remove_node(self, node: str) -> None:
        with self._lock:
            for d in self._out.pop(node, set()):
                self._in.get(d, set()).discard(node)
            for s in self._in.pop(node, set()):
                self._out.get(s, set()).discard(node)

    def clear(self) -> None:
        with self._lock:
            self._out.clear()
            self._in.clear()

    def dependants(self, dep_id: str) -> set[str]:
        with self._lock:
            return set(self._out.get(dep_id, ()))

    def sources(self, dep_id: str) -> set[str]:
        with self._lock:
            return set(self._in.get(dep_id, ()))

    def nodes(self) -> set[str]:
        with self._lock:
            return set(self._out) | set(self._in)

    def affected(self, dep_id: str, include_neighbours: int = 0, include_self: bool = False) -> set[str]:
        seeds = {dep_id}
        frontier = {dep_id}
        if self._neighbours is not None:
            for _ in range(max(0, include_neighbours)):
                nxt: set[str] = set()
                for n in frontier:
                    try:
                        nxt.update(self._neighbours(n))
                    except Exception:  # noqa: BLE001 - a broken neighbour callback must not break invalidation
                        pass
                frontier = nxt - seeds
                seeds |= nxt
        out = set(seeds)
        with self._lock:
            queue = deque(seeds)
            while queue:
                for d in self._out.get(queue.popleft(), ()):
                    if d not in out:
                        out.add(d)
                        queue.append(d)
        if not include_self:
            out.discard(dep_id)
        return out

    def affected_many(self, dep_ids: Iterable[str], include_neighbours: int = 0) -> set[str]:
        ids = set(dep_ids)
        out: set[str] = set()
        for d in ids:
            out |= self.affected(d, include_neighbours)
        return out | ids
