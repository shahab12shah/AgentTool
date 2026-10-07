"""Undoable commands that act on the project as a whole (script, voice-over, assets)."""

from __future__ import annotations

from dataclasses import replace

from app.core.commands import Command
from app.media.asset import Asset
from app.project.project import Project
from app.project.project_schema import VoiceOverData
from app.timeline.clip import Clip


class SetScriptCommand(Command):
    description = "Edit script"
    scope = "script"
    major = False
    merge_key = "script"

    def __init__(self, project: Project, text: str) -> None:
        self.project = project
        self.new_text = text
        self.old_text = project.script.text

    def do(self) -> None:
        self.project.script.text = self.new_text

    def undo(self) -> None:
        self.project.script.text = self.old_text

    def absorb(self, newer: Command) -> None:
        assert isinstance(newer, SetScriptCommand)
        self.new_text = newer.new_text


class AddAssetCommand(Command):
    """Registers an already-imported asset. Files on disk are not undone (see RemoveAssetCommand)."""

    description = "Import media"
    scope = "assets"

    def __init__(self, project: Project, asset: Asset) -> None:
        self.project, self.asset = project, asset

    def do(self) -> None:
        self.project.assets.add(self.asset)

    def undo(self) -> None:
        self.project.assets.remove(self.asset.id)


class SetVoiceOverCommand(Command):
    description = "Set voice-over"
    scope = "voice_over"

    def __init__(self, project: Project, asset: Asset | None) -> None:
        self.project = project
        self.new = (
            VoiceOverData(asset.id, asset.name, asset.duration) if asset else VoiceOverData()
        )
        self.old = replace(project.voice_over)
        self.description = "Set voice-over" if asset else "Remove voice-over"

    def do(self) -> None:
        self.project.voice_over = replace(self.new)

    def undo(self) -> None:
        self.project.voice_over = replace(self.old)


class RemoveAssetCommand(Command):
    """Removes an asset from the project, along with any clips using it.

    The media file itself is never deleted from disk, so undo is always possible and the
    user's original files are never touched.
    """

    description = "Remove media"
    scope = "assets"

    def __init__(self, project: Project, asset_id: str) -> None:
        self.project = project
        self.asset_id = asset_id
        self._asset: Asset | None = None
        self._clips: list[Clip] = []
        self._voice: VoiceOverData | None = None

    def do(self) -> None:
        self._asset = self.project.assets.require(self.asset_id)
        self._clips = [c.snapshot() for c in self.project.timeline.clips_for_asset(self.asset_id)]
        for clip in self._clips:
            self.project.timeline.detach_clip(clip.id)
        if self.project.voice_over.asset_id == self.asset_id:
            self._voice = replace(self.project.voice_over)
            self.project.voice_over = VoiceOverData()
        self.project.assets.remove(self.asset_id)

    def undo(self) -> None:
        assert self._asset is not None
        self.project.assets.add(self._asset)
        for clip in self._clips:
            self.project.timeline.insert_clip(clip)
        if self._voice is not None:
            self.project.voice_over = self._voice
            self._voice = None
