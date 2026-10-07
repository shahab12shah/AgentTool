"""Command-based undo/redo.

Commands mutate domain models only; they never touch the UI. The stack publishes
events so any view can refresh itself.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from app.core.constants import UNDO_LIMIT
from app.core.events import EventBus, Topics
from app.logging.logger import get_logger

_log = get_logger(__name__)
MERGE_WINDOW_SECONDS = 1.5


class Command(ABC):
    """A reversible change to the project."""

    description: str = "Change"
    scope: str = "meta"  # which part of the project changes (used by views to refresh)
    major: bool = True  # major commands trigger an immediate autosave
    merge_key: str | None = None  # consecutive commands with the same key may be merged

    @abstractmethod
    def do(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...

    def absorb(self, newer: "Command") -> None:  # pragma: no cover - only for mergeable commands
        raise NotImplementedError


class CompositeCommand(Command):
    """Runs several commands as one undo step; rolls back if one fails."""

    def __init__(self, description: str, commands: list[Command], scope: str = "meta") -> None:
        self.description = description
        self.scope = scope
        self._commands = commands

    def do(self) -> None:
        done: list[Command] = []
        try:
            for cmd in self._commands:
                cmd.do()
                done.append(cmd)
        except Exception:
            for cmd in reversed(done):
                cmd.undo()
            raise

    def undo(self) -> None:
        for cmd in reversed(self._commands):
            cmd.undo()


class CommandStack:
    def __init__(self, bus: EventBus | None = None, limit: int = UNDO_LIMIT) -> None:
        self._bus = bus
        self._limit = limit
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._last_time = 0.0

    # ----- state -----
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_text(self) -> str:
        return self._undo[-1].description if self._undo else ""

    @property
    def redo_text(self) -> str:
        return self._redo[-1].description if self._redo else ""

    # ----- operations -----
    def execute(self, command: Command) -> Command:
        """Run ``command`` and record it. If ``do`` raises, nothing is recorded."""
        command.do()
        now = time.monotonic()
        top = self._undo[-1] if self._undo else None
        if (
            top is not None
            and command.merge_key is not None
            and type(top) is type(command)
            and top.merge_key == command.merge_key
            and not self._redo
            and now - self._last_time < MERGE_WINDOW_SECONDS
        ):
            top.absorb(command)
        else:
            self._undo.append(command)
            if len(self._undo) > self._limit:
                del self._undo[0]
        self._redo.clear()
        self._last_time = now
        self._notify(command, "do")
        return command

    def undo(self) -> Command | None:
        if not self._undo:
            return None
        command = self._undo.pop()
        try:
            command.undo()
        except Exception:
            self._undo.append(command)
            raise
        self._redo.append(command)
        self._last_time = 0.0
        self._notify(command, "undo")
        return command

    def redo(self) -> Command | None:
        if not self._redo:
            return None
        command = self._redo.pop()
        try:
            command.do()
        except Exception:
            self._redo.append(command)
            raise
        self._undo.append(command)
        self._last_time = 0.0
        self._notify(command, "redo")
        return command

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._publish_stack()

    # ----- events -----
    def _notify(self, command: Command, action: str) -> None:
        _log.debug("command %s: %s", action, command.description)
        if self._bus:
            self._bus.publish(Topics.PROJECT_CHANGED, scope=command.scope, command=command, action=action)
        self._publish_stack()

    def _publish_stack(self) -> None:
        if self._bus:
            self._bus.publish(
                Topics.COMMAND_STACK_CHANGED,
                can_undo=self.can_undo,
                can_redo=self.can_redo,
                undo_text=self.undo_text,
                redo_text=self.redo_text,
            )
