"""Undoable (or recorded) project changes for the reference style layer.

None of these commands touches the timeline, the media library or any locked object: they change only ``reference_*`` sections of the
project (which the editing engines read on their *next* run) and the style application history.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.commands import Command
from app.editing.overrides import EditingStrategyOverrides
from app.project.project import Project
from app.reference.application import ReferenceAsset, ReferenceSettings, StyleApplication
from app.reference.style_model import ReferenceStyleProfile


class RegisterReferenceCommand(Command):
    """Record an imported reference video (the file is already in ``references/<id>/``). Never adds anything to ``project.assets``."""

    scope = "reference"
    major = True
    description = "Import reference video"

    def __init__(self, project: Project, asset: ReferenceAsset, make_active: bool = True) -> None:
        self.project, self.asset, self.make_active = project, asset, make_active
        self._old_active = ""

    def do(self) -> None:
        self._old_active = self.project.reference_settings.active_reference_id
        self.project.reference_assets[self.asset.reference_id] = deepcopy(self.asset)
        if self.make_active:
            self.project.reference_settings.active_reference_id = self.asset.reference_id

    def undo(self) -> None:
        self.project.reference_assets.pop(self.asset.reference_id, None)
        self.project.reference_analysis.pop(self.asset.reference_id, None)
        self.project.reference_settings.active_reference_id = self._old_active


class RemoveReferenceCommand(Command):
    """Forget a reference video (its folder is deleted by the service after this succeeds). If a style from it is applied, the applied overrides stay: they are abstract numbers."""

    scope = "reference"
    major = True
    description = "Remove reference video"

    def __init__(self, project: Project, reference_id: str) -> None:
        self.project, self.reference_id = project, reference_id
        self._saved: dict[str, Any] = {}

    def do(self) -> None:
        p, rid = self.project, self.reference_id
        self._saved = {"asset": p.reference_assets.pop(rid, None), "analysis": p.reference_analysis.pop(rid, None), "active": p.reference_settings.active_reference_id,
                       "profile": p.reference_style_profile}
        if p.reference_settings.active_reference_id == rid:
            others = [k for k in p.reference_assets if k != rid]
            p.reference_settings.active_reference_id = others[-1] if others else ""
        if p.reference_style_profile is not None and p.reference_style_profile.reference_id == rid:
            p.reference_style_profile = None

    def undo(self) -> None:
        p, rid = self.project, self.reference_id
        if self._saved.get("asset") is not None:
            p.reference_assets[rid] = self._saved["asset"]
        if self._saved.get("analysis") is not None:
            p.reference_analysis[rid] = self._saved["analysis"]
        p.reference_settings.active_reference_id = self._saved.get("active", "")
        p.reference_style_profile = self._saved.get("profile")


class StoreAnalysisCommand(Command):
    """Store the outcome of an analysis: the asset's status fields, the compact record and (for the active reference) the style profile."""

    scope = "reference"
    major = False
    description = "Reference analysis"

    def __init__(self, project: Project, reference_id: str, asset_fields: dict[str, Any], record: dict[str, Any] | None, profile: ReferenceStyleProfile | None) -> None:
        self.project, self.reference_id = project, reference_id
        self.asset_fields, self.record, self.profile = dict(asset_fields), deepcopy(record), deepcopy(profile)
        self._old: dict[str, Any] = {}

    def do(self) -> None:
        p, rid = self.project, self.reference_id
        asset = p.reference_assets.get(rid)
        if asset is None:
            return
        self._old = {"fields": {k: deepcopy(getattr(asset, k)) for k in self.asset_fields}, "record": deepcopy(p.reference_analysis.get(rid)), "profile": p.reference_style_profile,
                     "opv": p.reference_settings.analysis_version}
        for k, v in self.asset_fields.items():
            setattr(asset, k, deepcopy(v))
        if self.record is not None:
            p.reference_analysis[rid] = deepcopy(self.record)
        if self.profile is not None and (p.reference_settings.active_reference_id in ("", rid) or p.reference_style_profile is None):
            p.reference_style_profile = deepcopy(self.profile)
            p.reference_settings.analysis_version = self.profile.analysis_version

    def undo(self) -> None:
        p, rid = self.project, self.reference_id
        asset = p.reference_assets.get(rid)
        if asset is None or not self._old:
            return
        for k, v in self._old["fields"].items():
            setattr(asset, k, v)
        if self._old["record"] is None:
            p.reference_analysis.pop(rid, None)
        else:
            p.reference_analysis[rid] = self._old["record"]
        p.reference_style_profile = self._old["profile"]
        p.reference_settings.analysis_version = self._old["opv"]


class SetReferenceSettingsCommand(Command):
    """Customize state (mode, strength, slider targets, active reference, preserve-user-edits) — saved without applying anything."""

    scope = "reference"
    major = False
    description = "Change reference style settings"

    def __init__(self, project: Project, new: ReferenceSettings, profile: ReferenceStyleProfile | None = None, switch_profile: bool = False) -> None:
        self.project, self.new = project, deepcopy(new)
        self.profile, self.switch_profile = deepcopy(profile), switch_profile
        self._old: ReferenceSettings | None = None
        self._old_profile: ReferenceStyleProfile | None = None

    def do(self) -> None:
        self._old, self._old_profile = deepcopy(self.project.reference_settings), self.project.reference_style_profile
        self.project.reference_settings = deepcopy(self.new)
        if self.switch_profile:
            self.project.reference_style_profile = deepcopy(self.profile)

    def undo(self) -> None:
        assert self._old is not None
        self.project.reference_settings = self._old
        if self.switch_profile:
            self.project.reference_style_profile = self._old_profile


class ApplyStyleCommand(Command):
    """Apply a reference style: set the abstract overrides + settings and record the application (with what it replaced). One undo step.

    The timeline is *not* changed: the next Phase 4 / Phase 5 run reads the overrides, and every locked / user-owned object stays as it is.
    """

    scope = "reference"
    major = True

    def __init__(self, project: Project, settings: ReferenceSettings, overrides: EditingStrategyOverrides, application: StyleApplication, description: str = "Apply reference style") -> None:
        self.project, self.settings, self.overrides, self.application = project, deepcopy(settings), deepcopy(overrides), deepcopy(application)
        self.description = description
        self._old: dict[str, Any] = {}

    def do(self) -> None:
        p = self.project
        self._old = {"settings": deepcopy(p.reference_settings), "overrides": deepcopy(p.reference_style_overrides), "history": len(p.style_application_history)}
        self.application.before = deepcopy(p.reference_style_overrides)
        self.application.before_settings = deepcopy(p.reference_settings)
        self.application.after = deepcopy(self.overrides)
        p.reference_settings = deepcopy(self.settings)
        p.reference_style_overrides = deepcopy(self.overrides)
        p.style_application_history.append(deepcopy(self.application))

    def undo(self) -> None:
        p = self.project
        p.reference_settings = self._old["settings"]
        p.reference_style_overrides = self._old["overrides"]
        del p.style_application_history[self._old["history"]:]


class ClearStyleCommand(ApplyStyleCommand):
    """Turn the applied reference style off (the overrides are emptied; the history keeps a ``CLEARED`` record). Undoable."""

    def __init__(self, project: Project, application: StyleApplication) -> None:
        settings = deepcopy(project.reference_settings)
        settings.enabled = False
        super().__init__(project, settings, EditingStrategyOverrides(), application, "Remove reference style")
