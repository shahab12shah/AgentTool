"""Thread-safe, bounded, low-overhead metrics: operation timings, counters, gauges and cache hit/miss counts.

Nothing here ever stores file contents, paths or credentials — only operation names (dotted, e.g. ``project.open``), numbers and short tags
supplied by the caller. Each operation keeps aggregates plus a small ring of recent durations so p50/p95 can be reported without unbounded growth.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

RECENT = 256  # durations kept per operation for percentiles


@dataclass
class OpStats:
    count: int = 0
    errors: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    last: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT))

    def add(self, seconds: float, ok: bool) -> None:
        self.count += 1
        self.errors += 0 if ok else 1
        self.total += seconds
        self.last = seconds
        self.min = min(self.min, seconds)
        self.max = max(self.max, seconds)
        self.recent.append(seconds)

    def percentile(self, q: float) -> float:
        if not self.recent:
            return 0.0
        data = sorted(self.recent)
        return data[min(len(data) - 1, int(q * len(data)))]

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "errors": self.errors, "total_s": round(self.total, 6), "mean_s": round(self.total / self.count, 6) if self.count else 0.0,
                "min_s": round(self.min, 6) if self.count else 0.0, "max_s": round(self.max, 6), "last_s": round(self.last, 6),
                "p50_s": round(self.percentile(0.5), 6), "p95_s": round(self.percentile(0.95), 6)}


class MetricsRegistry:
    """Aggregates only; recording is one lock acquisition. ``enabled = False`` makes every call return immediately."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._ops: dict[str, OpStats] = {}
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._started = time.time()

    # ---- recording
    def record(self, name: str, seconds: float, ok: bool = True) -> None:
        if not self.enabled:
            return
        with self._lock:
            st = self._ops.get(name)
            if st is None:
                st = self._ops[name] = OpStats()
            st.add(seconds, ok)

    def incr(self, name: str, n: float = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def gauge(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._gauges[name] = float(value)

    def cache_hit(self, category: str) -> None:
        self.incr(f"cache.{category}.hit")

    def cache_miss(self, category: str) -> None:
        self.incr(f"cache.{category}.miss")

    # ---- queries
    def op(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            st = self._ops.get(name)
            return st.to_dict() if st else None

    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0)

    def hit_rates(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            cats = sorted({k.split(".")[1] for k in self._counters if k.startswith("cache.") and k.count(".") == 2})
            out = {}
            for c in cats:
                h, m = self._counters.get(f"cache.{c}.hit", 0), self._counters.get(f"cache.{c}.miss", 0)
                out[c] = {"hits": int(h), "misses": int(m), "rate": round(h / (h + m), 4) if h + m else None}
            return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"uptime_s": round(time.time() - self._started, 1), "enabled": self.enabled, "operations": {k: v.to_dict() for k, v in sorted(self._ops.items())},
                    "counters": {k: (int(v) if float(v).is_integer() else round(v, 4)) for k, v in sorted(self._counters.items())}, "gauges": {k: round(v, 4) for k, v in sorted(self._gauges.items())},
                    "cache_hit_rates": {}}  # filled below without re-locking

    def full_snapshot(self) -> dict[str, Any]:
        snap = self.snapshot()
        snap["cache_hit_rates"] = self.hit_rates()
        return snap

    def reset(self) -> None:
        with self._lock:
            self._ops.clear()
            self._counters.clear()
            self._gauges.clear()
            self._started = time.time()
