"""``Profiler``: the process-wide entry point for timing operations. Optional, aggregate-only and cheap.

Slow operations (over ``slow_threshold_s``) are logged once as a structured ``perf.slow_operation`` event with the operation name and duration
only. Use the module-level ``profiler`` singleton; services accept an injected one for tests.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from app.logging.logger import get_logger, log_event
from app.performance.metrics import MetricsRegistry
from app.performance.operation_timer import OperationTimer

_log = get_logger(__name__)


class Profiler:
    def __init__(self, enabled: bool = True, slow_threshold_s: float = 1.5) -> None:
        self.metrics = MetricsRegistry(enabled)
        self.slow_threshold_s = slow_threshold_s
        self._slow_logged: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.metrics.enabled

    def configure(self, *, enabled: bool | None = None, slow_threshold_s: float | None = None) -> None:
        if enabled is not None:
            self.metrics.enabled = bool(enabled)
        if slow_threshold_s is not None:
            self.slow_threshold_s = max(0.0, float(slow_threshold_s))

    def timer(self, name: str) -> OperationTimer:
        return OperationTimer(self.metrics, name, self._after)

    def timed(self, name: str) -> Callable:
        def deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*a: Any, **k: Any):
                with self.timer(name):
                    return fn(*a, **k)

            return wrapper

        return deco

    def _after(self, name: str, seconds: float, ok: bool) -> None:
        if self.slow_threshold_s and seconds >= self.slow_threshold_s:
            n = self._slow_logged.get(name, 0)
            if n < 20:  # a repeatedly slow operation must not flood the log
                self._slow_logged[name] = n + 1
                log_event(_log, "perf.slow_operation", operation=name, seconds=round(seconds, 3), ok=ok)

    # convenience pass-throughs
    def incr(self, name: str, n: float = 1) -> None:
        self.metrics.incr(name, n)

    def gauge(self, name: str, value: float) -> None:
        self.metrics.gauge(name, value)

    def cache_hit(self, category: str) -> None:
        self.metrics.cache_hit(category)

    def cache_miss(self, category: str) -> None:
        self.metrics.cache_miss(category)

    def snapshot(self) -> dict[str, Any]:
        return self.metrics.full_snapshot()


profiler = Profiler()
