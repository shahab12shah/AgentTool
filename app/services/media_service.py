"""Application service for importing, listing, removing media and generating thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.core.commands import Command, CommandStack
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, MediaError
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.media.asset import Asset, AssetType
from app.media.importer import LINK_COPY, MediaImporter, PreparedMedia, build_asset
from app.media.thumbnails import ThumbnailService
from app.project.project import Project
from app.project.project_commands import AddAssetCommand, RemoveAssetCommand, SetVoiceOverCommand
from app.project.project_manager import ProjectManager

_log = get_logger(__name__)
Apply = Callable[[Command], None]


class MediaService:
    def __init__(
        self,
        projects: ProjectManager,
        commands: CommandStack,
        jobs: JobManager,
        importer: MediaImporter,
        thumbnails: ThumbnailService,
        bus: EventBus,
        apply: Apply,
    ) -> None:
        self._projects = projects
        self._commands = commands
        self._jobs = jobs
        self._importer = importer
        self._thumbnails = thumbnails
        self._bus = bus
        self._apply = apply
        self._thumb_in_flight: set[str] = set()
        self._thumb_failed: set[str] = set()

    # ----- helpers -----
    def _project(self) -> Project:
        project = self._projects.current
        if project is None or project.root is None:
            raise MediaError("Open or create a project first.")
        return project

    def _fail(self, job: Job) -> None:
        self._bus.publish(Topics.ERROR, message=job.error or "Import failed.", title=job.title)

    # ----- import -----
    def import_files(
        self,
        paths: list[Path],
        link_mode: str = LINK_COPY,
        on_asset: Callable[[Asset], None] | None = None,
    ) -> list[Job]:
        """Import files in background jobs (one per file). Returns the jobs."""
        project = self._project()
        jobs = []
        for path in paths:
            jobs.append(self._submit_import(project, Path(path), link_mode, on_asset))
        return jobs

    def _submit_import(
        self, project: Project, path: Path, link_mode: str, on_asset: Callable[[Asset], None] | None
    ) -> Job:
        root = project.root
        assert root is not None

        def work(ctx) -> PreparedMedia:
            return self._importer.prepare(
                root, path, known_hashes=known, link_mode=link_mode,
                progress=lambda frac, msg: ctx.report(frac * 100, f"{msg}: {path.name}"),
                should_cancel=ctx.is_cancelled,
            )

        known = project.assets.known_hashes()

        def done(job: Job) -> None:
            asset = self._register(project, job.result)
            if asset is not None and on_asset is not None:
                on_asset(asset)

        def cancelled(job: Job) -> None:
            prepared = job.result
            if isinstance(prepared, PreparedMedia) and prepared.copied and prepared.stored_path:
                (root / prepared.stored_path).unlink(missing_ok=True)

        return self._jobs.submit(
            "media.import", work, title=f"Importing {path.name}", on_complete=done, on_error=self._fail, on_cancel=cancelled
        )

    def _register(self, project: Project, prepared: PreparedMedia) -> Asset | None:
        """Main-thread step: add the prepared media to the project."""
        if self._projects.current is not project or project.root is None:
            if prepared.copied and prepared.stored_path and project.root:
                (project.root / prepared.stored_path).unlink(missing_ok=True)
            return None
        existing = project.assets.get(prepared.duplicate_of) if prepared.duplicate_of else None
        existing = existing or project.assets.find_by_hash(prepared.content_hash)
        if existing is not None:
            if prepared.copied and prepared.stored_path:  # same file imported twice in one batch
                (project.root / prepared.stored_path).unlink(missing_ok=True)
            self._bus.publish(Topics.STATUS, message=f"“{prepared.source.name}” is already in the project.")
            return existing
        asset = build_asset(project.assets.new_id(), prepared)
        self._apply(AddAssetCommand(project, asset))
        log_event(_log, "media.imported", asset_id=asset.id, name=asset.name, type=asset.type.value)
        self.ensure_thumbnail(asset)
        return asset

    def import_voice_over(self, path: Path) -> Job:
        project = self._project()

        def got_asset(asset: Asset) -> None:
            if asset.type is not AssetType.AUDIO:
                self._bus.publish(Topics.ERROR, message="A voice-over must be an audio file.", title="Voice-over")
                return
            self._commands.execute(SetVoiceOverCommand(project, asset))

        return self._submit_import(project, Path(path), LINK_COPY, got_asset)

    def remove_voice_over(self) -> None:
        self._commands.execute(SetVoiceOverCommand(self._project(), None))

    # ----- removal -----
    def remove_asset(self, asset_id: str) -> None:
        """Remove from the project (and any timeline clips). The media file stays on disk."""
        self._commands.execute(RemoveAssetCommand(self._project(), asset_id))

    # ----- thumbnails -----
    def thumbnail_file(self, asset: Asset) -> Path | None:
        project = self._projects.current
        if project is None or project.root is None:
            return None
        return self._thumbnails.thumbnail_path(project.root, asset) if self._thumbnails.is_cached(project.root, asset) else None

    def ensure_thumbnail(self, asset: Asset) -> Job | None:
        project = self._project()
        root = project.root
        assert root is not None
        key = f"{project.project_id}:{asset.id}"
        if key in self._thumb_in_flight or key in self._thumb_failed or self._thumbnails.is_cached(root, asset):
            return None
        self._thumb_in_flight.add(key)

        def work(ctx) -> str:
            ctx.report(10, f"Generating thumbnail: {asset.name}")
            return str(self._thumbnails.ensure(root, asset, ctx.is_cancelled))

        def done(job: Job) -> None:
            self._thumb_in_flight.discard(key)
            self._bus.publish(Topics.THUMBNAIL_READY, asset_id=asset.id)

        def failed(job: Job) -> None:
            self._thumb_in_flight.discard(key)
            self._thumb_failed.add(key)  # do not retry automatically; avoids error loops

        def cancelled(job: Job) -> None:
            self._thumb_in_flight.discard(key)

        return self._jobs.submit(
            "media.thumbnail", work, title=f"Generating thumbnail for {asset.name}",
            on_complete=done, on_error=failed, on_cancel=cancelled,
        )

    def ensure_all_thumbnails(self) -> None:
        """Generate any thumbnails that are missing (cached ones are skipped)."""
        project = self._projects.current
        if project is None:
            return
        for asset in project.assets:
            try:
                self.ensure_thumbnail(asset)
            except AppError:
                _log.warning("Could not schedule thumbnail", exc_info=True)
