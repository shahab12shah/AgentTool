"""Project edits made by the rendering layer. They only touch render bookkeeping (settings, history, proxy records, an asset's file location) —
never the timeline."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.core.commands import Command
from app.project.project import Project
from app.project.project_schema import RenderSettings


class SetRenderSettingsCommand(Command):
    scope = "render"
    major = False

    def __init__(self, project: Project, new: RenderSettings, description: str = "Change export settings") -> None:
        self.project, self.new, self.description = project, new, description
        self._old: RenderSettings | None = None

    def do(self) -> None:
        self._old = replace(self.project.render_settings)
        self.project.render_settings = replace(self.new)

    def undo(self) -> None:
        assert self._old is not None
        self.project.render_settings = self._old


class RenderRecordCommand(Command):
    """Add or update a render record in ``project.render_history`` (matched by ``render_id``)."""

    scope = "render"
    major = False
    description = "Render record"

    def __init__(self, project: Project, record: dict[str, Any]) -> None:
        self.project, self.record = project, copy.deepcopy(record)
        self._old: list[dict[str, Any]] | None = None

    def do(self) -> None:
        self._old = copy.deepcopy(self.project.render_history)
        hist = self.project.render_history
        for i, r in enumerate(hist):
            if r.get("render_id") == self.record["render_id"]:
                hist[i] = copy.deepcopy(self.record)
                return
        hist.append(copy.deepcopy(self.record))

    def undo(self) -> None:
        assert self._old is not None
        self.project.render_history[:] = self._old


class DeleteRenderRecordCommand(Command):
    scope = "render"
    major = False
    description = "Remove render record"

    def __init__(self, project: Project, render_id: str) -> None:
        self.project, self.render_id = project, render_id
        self._old: list[dict[str, Any]] | None = None

    def do(self) -> None:
        self._old = copy.deepcopy(self.project.render_history)
        self.project.render_history[:] = [r for r in self.project.render_history if r.get("render_id") != self.render_id]

    def undo(self) -> None:
        assert self._old is not None
        self.project.render_history[:] = self._old


class SetProxyRecordCommand(Command):
    """Set (or remove, with ``None``) the proxy record of one asset."""

    scope = "proxies"
    major = False
    description = "Proxy record"

    def __init__(self, project: Project, asset_id: str, record: dict[str, Any] | None) -> None:
        self.project, self.asset_id, self.record = project, asset_id, copy.deepcopy(record)
        self._old: dict[str, Any] | None = None

    def do(self) -> None:
        self._old = copy.deepcopy(self.project.proxies.get(self.asset_id))
        if self.record is None:
            self.project.proxies.pop(self.asset_id, None)
        else:
            self.project.proxies[self.asset_id] = copy.deepcopy(self.record)

    def undo(self) -> None:
        if self._old is None:
            self.project.proxies.pop(self.asset_id, None)
        else:
            self.project.proxies[self.asset_id] = self._old


class RelinkAssetCommand(Command):
    """Point an asset at another file on disk (the asset id — and so every timeline reference — stays the same). Undoable."""

    scope = "assets"
    major = True

    def __init__(self, project: Project, asset_id: str, new_path: Path, facts: dict[str, Any]) -> None:
        self.project, self.asset_id, self.new_path, self.facts = project, asset_id, Path(new_path), dict(facts)
        self.description = "Relink media"
        self._old: dict[str, Any] | None = None

    _FIELDS = ("path", "link_mode", "size_bytes", "content_hash", "duration", "width", "height", "fps", "codec", "has_audio", "audio_codec", "sample_rate", "channels")

    def do(self) -> None:
        a = self.project.assets.require(self.asset_id)
        self._old = {f: getattr(a, f) for f in self._FIELDS} | {"extra": copy.deepcopy(a.extra)}
        a.path, a.link_mode = str(self.new_path), "reference"
        for k, v in self.facts.items():
            if k in self._FIELDS and v is not None:
                setattr(a, k, v)
        a.extra["relinked_from"] = self._old["path"]

    def undo(self) -> None:
        assert self._old is not None
        a = self.project.assets.require(self.asset_id)
        for f in self._FIELDS:
            setattr(a, f, self._old[f])
        a.extra = self._old["extra"]
