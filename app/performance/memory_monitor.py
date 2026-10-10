"""Memory growth tracking for long sessions: RSS history, growth since a baseline, and a scoped ``watch``.

This observes; it never frees memory by itself. Bounded caches (``BoundedLRU`` below) are what keep memory in check — the monitor tells you whether
they work, and the diagnostic report flags sustained growth.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Generic, Hashable, Iterator, TypeVar

from app.performance.resource_monitor import process_rss_bytes

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class MemoryMonitor:
    def __init__(self, keep: int = 600) -> None:
        self._lock = threading.Lock()
        self._samples: list[tuple[float, int]] = []
        self._keep = keep
        self.baseline: int | None = None

    def sample(self, label: str = "") -> int | None:  # noqa: ARG002 - label reserved for diagnostics
        rss = process_rss_bytes()
        if rss is None:
            return None
        with self._lock:
            self._samples.append((time.time(), rss))
            del self._samples[: max(0, len(self._samples) - self._keep)]
            if self.baseline is None:
                self.baseline = rss
        return rss

    def mark_baseline(self) -> int | None:
        rss = process_rss_bytes()
        with self._lock:
            self.baseline = rss
        return rss

    def growth_bytes(self) -> int | None:
        """Current RSS minus the baseline (None when RSS is unavailable)."""
        rss = process_rss_bytes()
        with self._lock:
            return None if rss is None or self.baseline is None else rss - self.baseline

    def trend_bytes_per_minute(self) -> float | None:
        """Least-squares slope over the retained samples; None with fewer than 3 samples spanning >= 10 s."""
        with self._lock:
            s = list(self._samples)
        if len(s) < 3 or s[-1][0] - s[0][0] < 10:
            return None
        n = len(s)
        mx = sum(t for t, _ in s) / n
        my = sum(v for _, v in s) / n
        den = sum((t - mx) ** 2 for t, _ in s)
        return None if den == 0 else 60.0 * sum((t - mx) * (v - my) for t, v in s) / den

    @contextmanager
    def watch(self, sink: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Measure RSS before/after a block; the (mutable) result dict is filled when the block exits."""
        out: dict[str, Any] = sink if sink is not None else {}
        before = process_rss_bytes()
        try:
            yield out
        finally:
            after = process_rss_bytes()
            out["before"], out["after"] = before, after
            out["delta"] = None if before is None or after is None else after - before


class BoundedLRU(Generic[K, V]):
    """A thread-safe LRU bounded by item count and/or total weight (bytes). The in-memory tier of the rebuildable caches (decoded thumbnails, frames).

    ``on_evict(key, value)`` lets the owner release a resource (close a handle, drop a pixmap) — eviction is explicit, not left to the garbage collector.
    """

    def __init__(self, max_items: int = 0, max_weight: int = 0, weigher: Callable[[V], int] | None = None, on_evict: Callable[[K, V], None] | None = None) -> None:
        self.max_items, self.max_weight = max_items, max_weight
        self._weigher = weigher or (lambda _v: 1)
        self._on_evict = on_evict
        self._d: "OrderedDict[K, tuple[V, int]]" = OrderedDict()
        self._weight = 0
        self._lock = threading.RLock()
        self.hits = self.misses = self.evictions = 0

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            item = self._d.get(key)
            if item is None:
                self.misses += 1
                return default
            self._d.move_to_end(key)
            self.hits += 1
            return item[0]

    def peek(self, key: K) -> V | None:
        with self._lock:
            item = self._d.get(key)
            return item[0] if item else None

    def put(self, key: K, value: V) -> None:
        evicted: list[tuple[K, V]] = []
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._weight -= old[1]
            w = max(0, int(self._weigher(value)))
            self._d[key] = (value, w)
            self._weight += w
            while self._d and ((self.max_items and len(self._d) > self.max_items) or (self.max_weight and self._weight > self.max_weight and len(self._d) > 1)):
                k, (v, vw) = self._d.popitem(last=False)
                self._weight -= vw
                self.evictions += 1
                evicted.append((k, v))
        self._release(evicted)

    def pop(self, key: K) -> V | None:
        with self._lock:
            item = self._d.pop(key, None)
            if item is None:
                return None
            self._weight -= item[1]
        return item[0]

    def discard_where(self, pred: Callable[[K], bool]) -> int:
        with self._lock:
            keys = [k for k in self._d if pred(k)]
            gone = [(k, self._d.pop(k)) for k in keys]
            for _k, (_v, w) in gone:
                self._weight -= w
        self._release([(k, v) for k, (v, _w) in gone])
        return len(gone)

    def clear(self) -> None:
        with self._lock:
            items = [(k, v) for k, (v, _w) in self._d.items()]
            self._d.clear()
            self._weight = 0
        self._release(items)

    def resize(self, max_items: int | None = None, max_weight: int | None = None) -> None:
        with self._lock:
            if max_items is not None:
                self.max_items = max_items
            if max_weight is not None:
                self.max_weight = max_weight
            last = next(reversed(self._d), None) if self._d else None
        if last is not None:  # re-apply the limits by touching the newest entry
            v = self.peek(last)
            if v is not None:
                self.put(last, v)

    def _release(self, items: list[tuple[K, V]]) -> None:
        if self._on_evict is None:
            return
        for k, v in items:
            try:
                self._on_evict(k, v)
            except Exception:  # noqa: BLE001 - a failing release hook must not break the cache
                pass

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: object) -> bool:
        return key in self._d

    @property
    def weight(self) -> int:
        return self._weight

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"items": len(self._d), "weight": self._weight, "max_items": self.max_items, "max_weight": self.max_weight, "hits": self.hits, "misses": self.misses,
                    "evictions": self.evictions}
