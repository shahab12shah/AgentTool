"""Tiny thread-safe publish/subscribe bus.

The core is Qt-free; the UI bridges this bus into Qt signals (see ``ui/qt_bridge.py``).
Handlers may be invoked on any thread: publishers decide, not the bus.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

from app.logging.logger import get_logger

Handler = Callable[[str, dict[str, Any]], None]

_log = get_logger(__name__)


class Topics:
    """Well-known event topic names."""

    PROJECT_OPENED = "project.opened"
    PROJECT_CLOSED = "project.closed"
    PROJECT_SAVED = "project.saved"
    PROJECT_CHANGED = "project.changed"  # payload: scope = script|assets|timeline|voice_over|meta
    DIRTY_CHANGED = "project.dirty_changed"
    COMMAND_STACK_CHANGED = "commands.changed"
    SELECTION_CHANGED = "selection.changed"
    JOB_ADDED = "job.added"
    JOB_UPDATED = "job.updated"
    THUMBNAIL_READY = "media.thumbnail_ready"
    RENDER_UPDATED = "render.updated"  # payload: job (RenderJob)
    RENDER_HISTORY_CHANGED = "render.history_changed"
    PROXY_CHANGED = "proxy.changed"  # payload: asset_id
    ERROR = "app.error"  # payload: message, details
    STATUS = "app.status"  # payload: message


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Subscribe to ``topic`` (``"*"`` receives everything). Returns an unsubscribe callable."""
        with self._lock:
            self._handlers[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers.get(topic, []):
                    self._handlers[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: str, **payload: Any) -> None:
        with self._lock:
            handlers = list(self._handlers.get(topic, ())) + list(self._handlers.get("*", ()))
        for handler in handlers:
            try:
                handler(topic, payload)
            except Exception:  # a broken subscriber must never break the publisher
                _log.exception("Event handler failed", extra={"topic": topic})
