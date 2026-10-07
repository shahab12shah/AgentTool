"""Shared handle given to every UI panel: the workspace plus the Qt event bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QWidget

from app.core.exceptions import AppError
from app.logging.logger import get_logger
from app.services.workspace import Workspace
from app.ui.qt_bridge import UiBridge

_log = get_logger(__name__)


@dataclass
class UiContext:
    ws: Workspace
    bridge: UiBridge

    def status(self, message: str) -> None:
        from app.core.events import Topics

        self.ws.bus.publish(Topics.STATUS, message=message)

    def guard(self, parent: QWidget | None, action: Callable[[], object], *, modal: bool = False, title: str = "") -> bool:
        """Run ``action``; turn expected errors into user-friendly feedback. Returns True on success."""
        from app.ui.dialogs.message import show_error

        try:
            action()
            return True
        except AppError as exc:
            _log.warning("Action failed: %s", exc, extra={"details": exc.details})
            if modal:
                show_error(parent, exc.user_message, title=title or "Cannot do that")
            else:
                self.status(exc.user_message)
        except Exception as exc:  # unexpected: never crash the UI
            _log.exception("Unexpected error in UI action")
            show_error(parent, "Something unexpected went wrong. The details were written to the log.", str(exc))
        return False
