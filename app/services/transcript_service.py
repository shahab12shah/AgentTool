"""Application service for transcription: provider choice, caching, change detection, jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from app.core.commands import Command
from app.core.config import Settings
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, TranscriptionError
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.phase2_commands import ApplyTranscriptionCommand, SetAlignmentCommand, SetTranscriptionFailureCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.storage.atomic import atomic_write_text
from app.transcription.alignment import align_script
from app.transcription.models import Transcript
from app.transcription.provider import ProviderRegistry, TranscriptionProvider
from app.transcription.providers.api_provider import OpenAICompatibleProvider
from app.transcription.providers.pocketsphinx_provider import PocketSphinxProvider
from app.transcription.providers.whisper_provider import FasterWhisperProvider
from app.transcription.service import TranscriptionEngine, TranscriptionResult
from app.transcription.status import TranscriptStatus, alignment_status, current_audio_hash, transcript_status

_log = get_logger(__name__)
AUTO_ORDER = ("faster-whisper", "api", "pocketsphinx")


class TranscriptService:
    def __init__(self, projects: ProjectManager, jobs: JobManager, bus: EventBus, apply: Callable[[Command], None],
                 settings: Callable[[], Settings]) -> None:
        self._projects, self._jobs, self._bus, self._apply, self._settings = projects, jobs, bus, apply, settings
        self.engine = TranscriptionEngine()
        self.registry = ProviderRegistry()
        self.registry.register(FasterWhisperProvider(model=lambda: settings().whisper_model))
        self.registry.register(OpenAICompatibleProvider(
            base_url=lambda: settings().api_base_url, model=lambda: settings().api_model,
            key_env=lambda: settings().api_key_env, ffmpeg_path=lambda: settings().ffmpeg_path))
        self.registry.register(PocketSphinxProvider(ffmpeg_path=lambda: settings().ffmpeg_path))

    # ------------------------------------------------------------ providers
    def provider_report(self) -> list[tuple[str, bool, str, str | None]]:
        """[(name, available, reason-if-not, accuracy-note)] in auto-selection order."""
        out = []
        for name in AUTO_ORDER:
            p = self.registry.get(name)
            if p:
                ok, why = p.is_available()
                out.append((name, ok, why, p.accuracy_note))
        return out

    def resolve_provider(self, name: str | None = None) -> TranscriptionProvider:
        name = (name or self._settings().transcription_provider or "auto").strip()
        if name == "auto":
            for candidate in AUTO_ORDER:
                p = self.registry.get(candidate)
                if p and p.is_available()[0]:
                    return p
            raise TranscriptionError("No transcription provider is available. See Settings → Transcription.")
        p = self.registry.get(name)
        if p is None:
            raise TranscriptionError(f"Unknown transcription provider “{name}”.")
        ok, why = p.is_available()
        if not ok:
            raise TranscriptionError(why)
        return p

    # ------------------------------------------------------------ state
    def _project(self) -> Project:
        if self._projects.current is None:
            raise TranscriptionError("Open or create a project first.")
        return self._projects.current

    def status(self) -> TranscriptStatus:
        return transcript_status(self._project())

    # ------------------------------------------------------------ transcription
    def _cache_file(self, project: Project, provider: TranscriptionProvider, language: str | None) -> Path | None:
        audio_hash = current_audio_hash(project)
        if not audio_hash or project.root is None:
            return None
        cfg = hashlib.sha1(f"{provider.config_key()}|{language or ''}".encode()).hexdigest()[:10]
        return project.root / "cache" / "transcripts" / f"{audio_hash[:20]}_{provider.name}_{cfg}.json"

    def transcribe(self, force: bool = False, provider_name: str | None = None) -> Job | None:
        """Start transcription in the background. Returns ``None`` when an up-to-date transcript already exists."""
        project = self._project()
        asset = project.assets.get(project.voice_over.asset_id) if project.voice_over.asset_id else None
        if asset is None:
            raise TranscriptionError("Import a voice-over first.")
        audio_path = project.asset_path(asset)
        if not audio_path.is_file():
            raise TranscriptionError("The voice-over file is missing from the project.")
        if not force and self.status() is TranscriptStatus.COMPLETE:
            self._bus.publish(Topics.STATUS, message="The transcript is already up to date.")
            return None
        provider = self.resolve_provider(provider_name)
        language = (self._settings().transcription_language or "").strip() or None
        cache = self._cache_file(project, provider, language)
        script = project.script.text
        audio_hash = asset.content_hash

        def work(ctx) -> TranscriptionResult:
            if cache is not None and not force and cache.is_file():
                try:
                    transcript = Transcript.from_dict(json.loads(cache.read_text(encoding="utf-8")))
                    transcript.validate()
                    transcript.audio.file_id, transcript.audio.filename = asset.id, asset.name
                    ctx.report(90, "Using cached transcript")
                    alignment = align_script(script, transcript) if script.strip() else None
                    log_event(_log, "transcription.cache_hit", cache=str(cache))
                    return TranscriptionResult(transcript, alignment)
                except (OSError, ValueError, KeyError, TypeError, AppError):
                    _log.warning("Ignoring unreadable transcript cache", extra={"cache": str(cache)})
            result = self.engine.run(provider, audio_path, asset, language, script,
                                     progress=lambda f, m: ctx.report(f * 100, m), should_cancel=ctx.is_cancelled)
            if cache is not None:
                try:
                    atomic_write_text(cache, json.dumps(result.transcript.to_dict(), ensure_ascii=False))
                except AppError:
                    _log.warning("Could not write transcript cache", exc_info=True)
            return result

        def done(job: Job) -> None:
            if self._projects.current is not project or current_audio_hash(project) != audio_hash:
                return  # project closed or voice-over replaced while transcribing: result is stale
            res: TranscriptionResult = job.result
            self._apply(ApplyTranscriptionCommand(project, res.transcript, res.alignment))
            self._bus.publish("transcription.done", words=len(res.transcript.words))

        def failed(job: Job) -> None:
            if self._projects.current is project:
                self._apply(SetTranscriptionFailureCommand(project, job.error or "Transcription failed.", audio_hash))
                self._bus.publish("transcription.failed", error=job.error)

        self._bus.publish("transcription.started")
        return self._jobs.submit("transcription", work, title=f"Transcribing {asset.name}", on_complete=done, on_error=failed)

    # ------------------------------------------------------------ alignment
    def realign(self) -> Job | None:
        """Recompute script↔transcript alignment (cheap; never re-transcribes)."""
        project = self._project()
        tr = project.transcription.transcript
        if tr is None:
            raise TranscriptionError("There is no transcript to align the script to.")
        script = project.script.text

        def work(ctx):
            ctx.report(10, "Aligning script to spoken words")
            return align_script(script, tr) if script.strip() else None

        def done(job: Job) -> None:
            if self._projects.current is project and project.transcription.transcript is tr:
                self._apply(SetAlignmentCommand(project, job.result))

        return self._jobs.submit("alignment", work, title="Aligning script to transcript", on_complete=done)
