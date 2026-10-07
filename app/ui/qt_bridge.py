"""Bridges the Qt-free core (event bus, job callbacks) onto the Qt main thread."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, Signal, Slot

from app.core.events import EventBus


class UiBridge(QObject):
    """Re-emits bus events and runs dispatched callables on the thread this object lives in.

    Signals emitted from worker threads are delivered as queued connections, so every
    receiver slot runs on the UI thread.
    """

    event = Signal(str, object)
    _call = Signal(object)

    def __init__(self, bus: EventBus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._call.connect(self._run, Qt.ConnectionType.QueuedConnection)
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self.event.connect(self._route)
        bus.subscribe("*", self._forward)

    def on(self, topic: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Call ``handler(payload)`` on the UI thread whenever ``topic`` is published."""
        self._handlers.setdefault(topic, []).append(handler)

    @Slot(str, object)
    def _route(self, topic: str, payload: dict[str, Any]) -> None:
        for handler in list(self._handlers.get(topic, ())):
            try:
                handler(payload)
            except RuntimeError:  # widget already destroyed during shutdown
                pass
            except Exception:
                from app.logging.logger import get_logger

                get_logger(__name__).exception("UI event handler failed", extra={"topic": topic})

    def _forward(self, topic: str, payload: dict[str, Any]) -> None:
        self.event.emit(topic, payload)

    def dispatch(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the UI thread (always queued, never re-entrant)."""
        self._call.emit(fn)

    @Slot(object)
    def _run(self, fn: Callable[[], None]) -> None:
        fn()
