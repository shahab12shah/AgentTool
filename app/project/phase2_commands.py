"""Undoable commands for transcripts, scenes and visual preferences.

AI results arrive through these commands like any other edit, so they can always be undone,
and user edits are distinguishable from AI output (see ``Scene.origin`` / ``user_edited_fields``).
"""

from __future__ import annotations

from copy import deepcopy

from app.analysis.models import Scene, SceneAnalysisState, VisualIntent
from app.core.commands import Command
from app.core.exceptions import SceneEditError
from app.project.project import Project
from app.transcription.alignment import ScriptAlignment
from app.transcription.models import Transcript, TranscriptionState
from app.visual.preferences import VisualPreferences


class ApplyTranscriptionCommand(Command):
    """Installs a finished transcript (+ alignment). Applied outside undo history: it is the result of a job."""

    description = "Transcribe voice-over"
    scope = "transcript"

    def __init__(self, project: Project, transcript: Transcript, alignment: ScriptAlignment | None) -> None:
        self.project, self.transcript, self.alignment = project, transcript, alignment
        self._old = (project.transcription, project.script_alignment)

    def do(self) -> None:
        self.project.transcription = TranscriptionState(self.transcript)
        self.project.script_alignment = self.alignment

    def undo(self) -> None:
        self.project.transcription, self.project.script_alignment = self._old


class SetTranscriptionFailureCommand(Command):
    description = "Transcription failed"
    scope = "transcript"
    major = False

    def __init__(self, project: Project, error: str, audio_hash: str | None) -> None:
        self.project, self.error, self.audio_hash = project, error, audio_hash
        self._old = project.transcription

    def do(self) -> None:
        old = self._old
        self.project.transcription = TranscriptionState(old.transcript, self.error, self.audio_hash)

    def undo(self) -> None:
        self.project.transcription = self._old


class SetAlignmentCommand(Command):
    description = "Align script to transcript"
    scope = "transcript"
    major = False

    def __init__(self, project: Project, alignment: ScriptAlignment | None) -> None:
        self.project, self.alignment = project, alignment
        self._old = project.script_alignment

    def do(self) -> None:
        self.project.script_alignment = self.alignment

    def undo(self) -> None:
        self.project.script_alignment = self._old


class SetScenesCommand(Command):
    """Replaces every scene, intent and the analysis state (result of a pipeline run)."""

    scope = "scenes"

    def __init__(self, project: Project, scenes: list[Scene], intents: dict[str, VisualIntent],
                 state: SceneAnalysisState, description: str = "Generate scenes") -> None:
        self.project, self.scenes, self.intents, self.state = project, scenes, intents, state
        self.description = description
        self._old: tuple | None = None

    def do(self) -> None:
        p = self.project
        self._old = (p.scenes, p.visual_intents, p.scene_analysis)
        p.scenes, p.visual_intents, p.scene_analysis = deepcopy(self.scenes), deepcopy(self.intents), self.state

    def undo(self) -> None:
        assert self._old is not None
        self.project.scenes, self.project.visual_intents, self.project.scene_analysis = self._old


class ReplaceScenesCommand(Command):
    """Replaces a contiguous run of scenes by other scenes (split: 1->2, merge: 2->1, edit/approve: 1->1)."""

    scope = "scenes"

    def __init__(self, project: Project, old_ids: list[str], new_scenes: list[Scene], new_intents: dict[str, VisualIntent],
                 description: str) -> None:
        self.project, self.old_ids = project, old_ids
        self.new_scenes, self.new_intents = deepcopy(new_scenes), deepcopy(new_intents)
        self.description = description
        self._index = -1
        self._old_scenes: list[Scene] = []
        self._old_intents: dict[str, VisualIntent | None] = {}

    def do(self) -> None:
        p = self.project
        ids = [s.id for s in p.scenes]
        try:
            idx = ids.index(self.old_ids[0])
        except ValueError:
            raise SceneEditError("That scene no longer exists.") from None
        if ids[idx : idx + len(self.old_ids)] != self.old_ids:
            raise SceneEditError("The scenes to change are no longer next to each other.")
        self._index = idx
        self._old_scenes = p.scenes[idx : idx + len(self.old_ids)]
        self._old_intents = {sid: p.visual_intents.get(sid) for sid in self.old_ids}
        p.scenes[idx : idx + len(self.old_ids)] = deepcopy(self.new_scenes)
        for sid in self.old_ids:
            p.visual_intents.pop(sid, None)
        p.visual_intents.update(deepcopy(self.new_intents))

    def undo(self) -> None:
        p = self.project
        new_ids = [s.id for s in self.new_scenes]
        p.scenes[self._index : self._index + len(new_ids)] = self._old_scenes
        for sid in new_ids:
            p.visual_intents.pop(sid, None)
        for sid, intent in self._old_intents.items():
            if intent is not None:
                p.visual_intents[sid] = intent


class SetVisualPreferencesCommand(Command):
    description = "Change visual preferences"
    scope = "preferences"
    merge_key = "visual_prefs"

    def __init__(self, project: Project, prefs: VisualPreferences) -> None:
        self.project, self.new = project, prefs.sanitized()
        self.old = deepcopy(project.visual_preferences)

    def do(self) -> None:
        self.project.visual_preferences = deepcopy(self.new)

    def undo(self) -> None:
        self.project.visual_preferences = deepcopy(self.old)

    def absorb(self, newer: Command) -> None:
        assert isinstance(newer, SetVisualPreferencesCommand)
        self.new = newer.new
