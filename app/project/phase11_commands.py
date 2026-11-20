"""Phase 11 commands: changes to the editorial section of the project (settings, results, recommendation status, overrides, preferences).

One small command replaces named fields of ``Project.editorial`` (shallow: the caller hands over NEW lists / dicts and never mutates the old ones, so the old values are kept by reference for
undo). Results of an analysis are installed with ``apply_command`` (not an edit of the video, not in the undo history); changing a recommendation's status as part of applying it is executed
inside the apply transaction, so one undo takes the edit and the status back together.
"""

from __future__ import annotations

from typing import Any

from app.core.commands import Command
from app.project.project import Project

_FIELDS = frozenset({
    "editorial_settings", "narrative_structure", "scene_roles", "story_sections", "scene_relationships", "visual_strategy", "visual_sequence_analysis", "visual_consistency_results",
    "visual_diversity_results", "pacing_recommendations", "emphasis_results", "evidence_alignment_results", "style_consistency_results", "findings", "editorial_recommendations",
    "editorial_overrides", "user_editorial_preferences", "editorial_analysis_history", "scorecard", "sections", "stage_cache", "project_hash",
})


class SetEditorialFieldsCommand(Command):
    scope = "editorial"
    major = False

    def __init__(self, project: Project, description: str = "Update editorial analysis", **fields: Any) -> None:
        bad = set(fields) - _FIELDS
        if bad:
            raise ValueError(f"unknown editorial field(s): {sorted(bad)}")
        self.project = project
        self.description = description
        self.fields = fields
        self._old: dict[str, Any] = {}

    def do(self) -> None:
        ed = self.project.editorial
        self._old = {k: getattr(ed, k) for k in self.fields}
        for k, v in self.fields.items():
            setattr(ed, k, v)

    def undo(self) -> None:
        ed = self.project.editorial
        for k, v in self._old.items():
            setattr(ed, k, v)
