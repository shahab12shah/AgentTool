"""Undoable commands for the presentation layer (captions, graphics, music, SFX, ducking, settings).

Like the AI-edit commands, they never replace the project's ``Timeline`` object (older undo entries hold references to it) — only its
``tracks`` list is swapped.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.commands import Command
from app.editing.models import Creator
from app.presentation.assembly import PresState
from app.presentation.models import PresentationDecision, PresentationOverride, PresentationType
from app.project.project import Project

OWNING = (PresentationType.CAPTION, PresentationType.NUMBER_GRAPHIC, PresentationType.DATE_GRAPHIC, PresentationType.LOWER_THIRD, PresentationType.HEADLINE,
          PresentationType.TEXT_GRAPHIC, PresentationType.MOTION_GRAPHIC, PresentationType.EVIDENCE_GRAPHIC, PresentationType.MUSIC, PresentationType.SFX)


class ApplyPresentationCommand(Command):
    scope = "timeline"
    major = True

    def __init__(self, project: Project, after: PresState, description: str = "Presentation") -> None:
        self.project, self.after, self.description = project, after, description
        self._before: PresState | None = None

    def do(self) -> None:
        if self._before is None:
            self._before = PresState.capture(self.project)
        try:
            self.after.install(self.project)
        except Exception:
            self._before.install(self.project)
            raise

    def undo(self) -> None:
        assert self._before is not None
        self._before.install(self.project)


class SetSettingCommand(Command):
    """Change one of the project-level presentation settings objects (audio_settings, caption_settings, audio_processing, caption_styles)."""

    scope = "editing"
    major = False

    def __init__(self, project: Project, attr: str, new: Any, description: str) -> None:
        self.project, self.attr, self.new, self.description = project, attr, deepcopy(new), description
        self._old: Any = None

    def do(self) -> None:
        self._old = deepcopy(getattr(self.project, self.attr))
        setattr(self.project, self.attr, deepcopy(self.new))

    def undo(self) -> None:
        setattr(self.project, self.attr, self._old)


class MarkPresentationEditCommand(Command):
    """After the user changes a presentation object on the timeline: it becomes USER-owned, its AI decision is replaced by a USER decision
    that records which AI decision it overrides (same undo step as the edit)."""

    scope = "timeline"
    major = False
    description = "Mark user edit"

    def __init__(self, project: Project, clip_id: str) -> None:
        self.project, self.clip_id = project, clip_id
        self._undo: dict[str, Any] = {}

    def do(self) -> None:
        p = self.project
        clip = p.timeline.get_clip(self.clip_id)
        self._undo = {}
        if clip is None:
            return
        self._undo["created_by"] = clip.created_by
        clip.created_by = Creator.USER.value
        swapped = []
        for d in [d for d in p.presentation_decisions.values() if d.target_id == self.clip_id and d.type in OWNING and d.created_by is Creator.AI]:
            new = deepcopy(d)
            n = max((int(k.rsplit("_", 1)[-1]) for k in p.presentation_decisions if k.rsplit("_", 1)[-1].isdigit()), default=0)
            new.decision_id, new.created_by, new.overrides_decision_id = f"pdec_{n + 1:05d}", Creator.USER, d.decision_id
            new.start, new.duration = round(clip.timeline_start, 4), round(clip.duration, 4)
            del p.presentation_decisions[d.decision_id]
            p.presentation_decisions[new.decision_id] = new
            p.presentation_overrides.append(PresentationOverride(new.decision_id, deepcopy(d)))
            if clip.ai_decision_id == d.decision_id:
                clip.ai_decision_id = new.decision_id
            swapped.append((d, new.decision_id))
        self._undo["swapped"] = swapped

    def undo(self) -> None:
        p = self.project
        clip = p.timeline.get_clip(self.clip_id)
        if clip is None or "created_by" not in self._undo:
            return
        clip.created_by = self._undo["created_by"]
        for old, new_id in self._undo.get("swapped", []):
            p.presentation_decisions.pop(new_id, None)
            p.presentation_decisions[old.decision_id] = old
            p.presentation_overrides[:] = [o for o in p.presentation_overrides if o.override_id != new_id]
            if clip.ai_decision_id == new_id:
                clip.ai_decision_id = old.decision_id


class RecordPresentationDeleteCommand(Command):
    """The user deleted an AI presentation object: remember the slot so regeneration does not bring it back."""

    scope = "timeline"
    major = False
    description = "Record deletion"

    def __init__(self, project: Project, clip_id: str) -> None:
        self.project = project
        clip = project.timeline.get_clip(clip_id)
        self.scene_id, self.slot = (clip.scene_id, clip.slot) if clip else ("", "")
        self.clip_id = clip_id
        self.removed: list[PresentationDecision] = [deepcopy(d) for d in project.presentation_decisions.values() if d.target_id == clip_id and d.type in OWNING]
        self._added = False

    def do(self) -> None:
        p = self.project
        for d in self.removed:
            p.presentation_decisions.pop(d.decision_id, None)
        key = f"{self.scene_id}|{self.slot}"
        if self.scene_id and self.slot and key not in p.presentation_generation.suppressed_slots:
            p.presentation_generation.suppressed_slots.append(key)
            self._added = True

    def undo(self) -> None:
        p = self.project
        for d in self.removed:
            p.presentation_decisions[d.decision_id] = deepcopy(d)
        if self._added:
            p.presentation_generation.suppressed_slots.remove(f"{self.scene_id}|{self.slot}")
            self._added = False


class SetAssetExtraCommand(Command):
    """Tag an asset (music / SFX role, SFX category). Stored in ``Asset.extra``."""

    scope = "assets"
    major = False
    description = "Tag audio asset"

    def __init__(self, project: Project, asset_id: str, **extra: Any) -> None:
        self.project, self.asset_id, self.extra = project, asset_id, extra
        self._old: dict | None = None

    def do(self) -> None:
        a = self.project.assets.require(self.asset_id)
        self._old = dict(a.extra)
        a.extra.update(self.extra)

    def undo(self) -> None:
        assert self._old is not None
        self.project.assets.require(self.asset_id).extra = self._old
