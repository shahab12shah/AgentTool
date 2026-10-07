"""Application exception hierarchy.

Every ``AppError`` carries a ``user_message`` that is safe to show in the UI and an
optional ``details`` string meant for developer logs only.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, recoverable errors."""

    def __init__(self, user_message: str, *, details: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.details = details

    def __str__(self) -> str:
        return self.user_message if not self.details else f"{self.user_message} ({self.details})"


class ProjectError(AppError):
    """Project could not be created, opened or saved."""


class InvalidProjectError(ProjectError):
    """Project file is corrupt, malformed or fails schema validation."""

    def __init__(self, user_message: str, *, problems: list[str] | None = None, details: str | None = None) -> None:
        super().__init__(user_message, details=details)
        self.problems = problems or []


class MediaError(AppError):
    """Generic media import/processing failure."""


class UnsupportedMediaError(MediaError):
    """File type is not supported."""


class MediaProbeError(MediaError):
    """ffprobe could not read the file."""


class FFmpegUnavailableError(MediaError):
    """FFmpeg/ffprobe binaries could not be located."""


class TimelineError(AppError):
    """Invalid timeline operation (overlap, locked track, bad range...)."""


class CommandError(AppError):
    """Undo/redo stack failure."""


class RecoveryError(AppError):
    """Crash-recovery data could not be read or applied."""


class JobError(AppError):
    """Invalid job operation."""


class JobCancelled(Exception):
    """Raised inside a job function when cancellation was requested."""


class NotAvailableInPhase(AppError):
    """A feature that is intentionally not implemented yet."""


class TranscriptionError(AppError):
    """Transcription provider failed or is unavailable."""


class InvalidTranscriptError(AppError):
    """Transcript data violates timing/structure rules."""


class AnalysisError(AppError):
    """Scene analysis failed."""


class SceneEditError(AppError):
    """A manual scene edit (split/merge/edit) was rejected."""


class UserEditsPresentError(AppError):
    """Regeneration would overwrite scenes the user edited or approved."""

    def __init__(self, user_message: str, scene_labels: list[str]) -> None:
        super().__init__(user_message)
        self.scene_labels = scene_labels
