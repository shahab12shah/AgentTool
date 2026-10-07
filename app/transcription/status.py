"""Derived (never stored) status of the transcript and of the script alignment."""

from __future__ import annotations

from enum import Enum

from app.project.project import Project
from app.transcription.alignment import script_hash


class TranscriptStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    COMPLETE = "COMPLETE"
    OUTDATED = "OUTDATED"  # the voice-over changed or was removed after transcription
    FAILED = "FAILED"


class AlignmentStatus(str, Enum):
    NONE = "NONE"
    CURRENT = "CURRENT"
    OUTDATED = "OUTDATED"  # the script changed since it was aligned (re-aligning is cheap; no re-transcription needed)


def current_audio_hash(project: Project) -> str | None:
    asset_id = project.voice_over.asset_id
    asset = project.assets.get(asset_id) if asset_id else None
    return asset.content_hash if asset else None


def transcript_status(project: Project) -> TranscriptStatus:
    tr = project.transcription.transcript
    audio_hash = current_audio_hash(project)
    if tr is None:
        if project.transcription.last_error and project.transcription.failed_audio_hash == audio_hash:
            return TranscriptStatus.FAILED
        return TranscriptStatus.NOT_STARTED
    if audio_hash is None or tr.audio.content_hash != audio_hash:
        return TranscriptStatus.OUTDATED
    return TranscriptStatus.COMPLETE


def alignment_status(project: Project) -> AlignmentStatus:
    al, tr = project.script_alignment, project.transcription.transcript
    if al is None or tr is None:
        return AlignmentStatus.NONE
    if al.transcript_id != tr.transcript_id or al.script_hash != script_hash(project.script.text):
        return AlignmentStatus.OUTDATED
    return AlignmentStatus.CURRENT
