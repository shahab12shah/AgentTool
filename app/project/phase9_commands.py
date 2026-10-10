"""Project-level performance overrides (Phase 9). Pure settings: they never touch the timeline, assets or any editing decision."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.commands import Command
from app.project.project import Project


class SetPerformanceOverridesCommand(Command):
    """Replace the project's performance overrides (``{}`` = follow the global settings). Undoable, not a major (autosave-triggering) edit."""

    scope = "performance"
    major = False
    description = "Change performance settings"

    def __init__(self, project: Project, overrides: dict[str, Any]) -> None:
        self.project = project
        self.new = deepcopy(overrides or {})
        self.old: dict[str, Any] = {}

    def do(self) -> None:
        self.old = deepcopy(self.project.performance_overrides)
        self.project.performance_overrides = deepcopy(self.new)

    def undo(self) -> None:
        self.project.performance_overrides = deepcopy(self.old)
