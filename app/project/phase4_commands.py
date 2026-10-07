"""Undoable commands for the AI edit.

``ApplyEditCommand`` installs a whole candidate edit state (timeline tracks + decisions + strategy + generation) as ONE undo step.
Existing timeline commands hold a reference to the project's ``Timeline`` object, so the object itself is never replaced — only its
``tracks`` list — which keeps older undo entries valid.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.commands import Command
from app.editing.assembly import EditState
from app.editing.models import Creator, EditingDecision, EditingSettings, OverrideRecord
from app.project.project import Project


def install_state(project: Project, st: EditState) -> None:
    project.timeline.tracks = deepcopy(st.timeline.tracks)
    project.editing_decisions.clear()
    project.editing_decisions.update(deepcopy(st.decisions))
    project.ai_overrides[:] = deepcopy(st.overrides)
    project.editing_strategy = deepcopy(st.strategy)
    project.timeline_generation = deepcopy(st.generation)
    project.timeline_version = st.version


class ApplyEditCommand(Command):
    scope = "timeline"
    major = True

    def __init__(self, project: Project, after: EditState, description: str = "Generate AI edit") -> None:
        self.project, self.after, self.description = project, after, description
        self._before: EditState | None = None

    def do(self) -> None:
        if self._before is None:
            self._before = EditState.capture(self.project)
        try:
            install_state(self.project, self.after)
        except Exception:  # never leave the project half-edited
            install_state(self.project, self._before)
            raise

    def undo(self) -> None:
        assert self._before is not None
        install_state(self.project, self._before)


class SetEditingSettingsCommand(Command):
    scope = "editing"
    description = "Change editing settings"
    major = False

    def __init__(self, project: Project, settings: EditingSettings) -> None:
        self.project, self.new = project, deepcopy(settings)
        self._old: EditingSettings | None = None

    def do(self) -> None:
        self._old = deepcopy(self.project.editing_settings)
        self.project.editing_settings = deepcopy(self.new)

    def undo(self) -> None:
        assert self._old is not None
        self.project.editing_settings = self._old


class MarkUserEditCommand(Command):
    """After the user changes an AI-created clip: it becomes USER-owned and its AI decision is replaced by a USER decision that
    records which AI decision it overrides. Executed together with the edit itself (one undo step)."""

    scope = "timeline"
    major = False
    description = "Mark user edit"

    def __init__(self, project: Project, clip_id: str) -> None:
        self.project, self.clip_id = project, clip_id
        self._undo: dict[str, Any] | None = None

    def do(self) -> None:
        p = self.project
        clip = p.timeline.get_clip(self.clip_id)
        self._undo = {}
        if clip is None:
            return
        d = p.editing_decisions.get(clip.ai_decision_id) if clip.ai_decision_id else None
        if clip.created_by == Creator.USER.value and d is None:
            return
        self._undo.update(created_by=clip.created_by, decision_id=clip.ai_decision_id)
        clip.created_by = Creator.USER.value
        if d is None:
            return
        self._undo["old_params"], self._undo["old_timing"] = deepcopy(d.parameters), (d.start, d.duration)
        if d.created_by is Creator.USER:
            self._sync(d, clip)
            return
        new = deepcopy(d)
        n = max((int(k.rsplit("_", 1)[-1]) for k in p.editing_decisions if k.rsplit("_", 1)[-1].isdigit()), default=0)
        new.decision_id = f"dec_{n + 1:05d}"
        new.created_by, new.overrides_decision_id = Creator.USER, d.decision_id
        self._sync(new, clip)
        del p.editing_decisions[d.decision_id]
        p.editing_decisions[new.decision_id] = new
        p.ai_overrides.append(OverrideRecord(new.decision_id, deepcopy(d)))
        clip.ai_decision_id = new.decision_id
        self._undo.update(swapped=(d, new.decision_id))

    @staticmethod
    def _sync(d: EditingDecision, clip) -> None:
        d.start, d.duration = round(clip.timeline_start, 4), round(clip.duration, 4)
        if d.type.value == "VISUAL_TIMING":
            d.parameters.update(source_in=clip.source_in, source_out=clip.source_out, speed=clip.speed, track_id=clip.track_id, operation="TRIM")
        elif d.type.value in ("TEXT", "NUMBER_EMPHASIS"):
            d.parameters.update(start=round(clip.timeline_start, 4), duration=round(clip.duration, 4))

    def undo(self) -> None:
        u = self._undo or {}
        p = self.project
        clip = p.timeline.get_clip(self.clip_id)
        if clip is None or "created_by" not in u:
            return
        clip.created_by, clip.ai_decision_id = u["created_by"], u["decision_id"]
        if "swapped" in u:
            old, new_id = u["swapped"]
            p.editing_decisions.pop(new_id, None)
            p.editing_decisions[old.decision_id] = old
            p.ai_overrides[:] = [o for o in p.ai_overrides if o.override_id != new_id]
        elif "old_params" in u and u["decision_id"] in p.editing_decisions:
            d = p.editing_decisions[u["decision_id"]]
            d.parameters, (d.start, d.duration) = u["old_params"], u["old_timing"]


class RecordUserDeleteCommand(Command):
    """The user deleted an AI element: remember the slot so regeneration does not bring it back."""

    scope = "timeline"
    major = False
    description = "Record deletion"

    def __init__(self, project: Project, clip_id: str) -> None:
        self.project = project
        clip = project.timeline.get_clip(clip_id)
        self.scene_id = clip.scene_id if clip else ""
        self.slot = clip.slot if clip else ""
        self.clip_id = clip_id
        self.removed: list[EditingDecision] = [deepcopy(d) for d in project.editing_decisions.values() if d.target_id == clip_id]
        self._added = False

    def do(self) -> None:
        p = self.project
        for d in self.removed:
            p.editing_decisions.pop(d.decision_id, None)
        key = f"{self.scene_id}|{self.slot}"
        if self.scene_id and self.slot and key not in p.timeline_generation.suppressed_slots:
            p.timeline_generation.suppressed_slots.append(key)
            self._added = True

    def undo(self) -> None:
        p = self.project
        for d in self.removed:
            p.editing_decisions[d.decision_id] = deepcopy(d)
        if self._added:
            p.timeline_generation.suppressed_slots.remove(f"{self.scene_id}|{self.slot}")
            self._added = False
