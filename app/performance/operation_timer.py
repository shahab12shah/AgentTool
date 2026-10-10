"""``OperationTimer``: a context manager / decorator that records an operation's duration (and failure) in a ``MetricsRegistry``.

    with profiler.timer("project.open"):        # or @profiler.timed("project.open")
        ...

When the registry is disabled the timer does not even read the clock.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from app.performance.metrics import MetricsRegistry


class OperationTimer:
    __slots__ = ("_registry", "name", "_t0", "seconds", "_on_done", "ok")

    def __init__(self, registry: MetricsRegistry, name: str, on_done: Callable[[str, float, bool], None] | None = None) -> None:
        self._registry, self.name, self._on_done = registry, name, on_done
        self._t0 = 0.0
        self.seconds = 0.0
        self.ok = True

    def __enter__(self) -> "OperationTimer":
        if self._registry.enabled:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._registry.enabled and self._t0:
            self.seconds = time.perf_counter() - self._t0
            self.ok = exc_type is None
            self._registry.record(self.name, self.seconds, self.ok)
            if self._on_done is not None:
                self._on_done(self.name, self.seconds, self.ok)
        return False

    def stop(self, ok: bool = True) -> float:
        """For code that cannot use ``with``: stop explicitly (idempotent)."""
        if self._registry.enabled and self._t0:
            self.seconds = time.perf_counter() - self._t0
            self.ok = ok
            self._registry.record(self.name, self.seconds, ok)
            if self._on_done is not None:
                self._on_done(self.name, self.seconds, ok)
            self._t0 = 0.0
        return self.seconds


def timed(profiler_getter: Callable[[], Any], name: str) -> Callable:
    """Decorator factory: ``@timed(lambda: profiler, "x.y")``. Prefer ``Profiler.timed``."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with profiler_getter().timer(name):
                return fn(*a, **k)

        return wrapper

    return deco
