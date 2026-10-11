"""Application service for importing, listing, removing media and generating thumbnails."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from app.core.commands import Command, CommandStack
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, MediaError
from app.jobs.job import Job, Priority
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.media.asset import Asset, AssetType
from app.media.importer import LINK_COPY, MediaImporter, PreparedMedia, build_asset
from app.media.thumbnails import ThumbnailService
from app.performance.thumbnail_queue import PriorityWorkQueue
from app.project.project import Project
from app.project.project_commands import AddAssetCommand, RemoveAssetCommand, SetVoiceOverCommand
from app.project.project_manager import ProjectManager

_log = get_logger(__name__)
Apply = Callable[[Command], None]
THUMBNAIL_FAILED = "media.thumbnail_failed"  # payload: asset_id, reason, missing_source
DEFAULT_THUMBNAIL_CONCURRENCY = 2


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
        self._perf_getter: Callable[[], object | None] | None = None
        self._queue: PriorityWorkQueue | None = None
        self._queue_project: Project | None = None
        self._thumb_missing: set[str] = set()
        self._limits_cache, self._limits_at = None, 0.0
        for topic in (Topics.PROJECT_OPENED, Topics.PROJECT_CLOSED, Topics.PROJECT_CHANGED):
            bus.subscribe(topic, self._on_project_event)

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

    def register_prepared(self, project: Project, prepared: PreparedMedia, **provenance) -> Asset | None:
        """Register media prepared elsewhere (e.g. an acquired research candidate) with its provenance."""
        return self._register(project, prepared, **provenance)

    def _register(self, project: Project, prepared: PreparedMedia, **provenance) -> Asset | None:
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
        asset = build_asset(project.assets.new_id(), prepared, **provenance)
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
    def attach_performance(self, getter: Callable[[], object | None]) -> None:
        """Let thumbnails use the performance service (limits, cache index, pressure). Optional: without it a fixed small limit and no index are used."""
        self._perf_getter = getter

    def _perf(self):
        try:
            return self._perf_getter() if self._perf_getter else None
        except Exception:  # noqa: BLE001
            return None

    def _cache(self):
        perf = self._perf()
        return getattr(perf, "cache", None) if perf is not None else None

    def _limits(self):
        now = time.monotonic()
        if self._limits_at and now - self._limits_at < 2.0:
            return self._limits_cache
        perf, lim = self._perf(), None
        if perf is not None:
            try:
                lim = perf.limits()
            except Exception:  # noqa: BLE001
                lim = None
        self._limits_cache, self._limits_at = lim, now
        return lim

    def _pressure_ok(self) -> bool:
        perf = self._perf()
        mon = getattr(perf, "monitor", None) if perf is not None else None
        try:
            s = mon.latest() if mon is not None else None
            return True if s is None else mon.pressure(s).get("overall", "normal") == "normal"
        except Exception:  # noqa: BLE001
            return True

    def _queue_for(self, project: Project) -> PriorityWorkQueue:
        q = self._queue
        if q is not None and (self._queue_project is not project or q.closed):
            q.close()
            q = None
        if q is None:
            lim_c = lambda: max(1, getattr(self._limits(), "thumbnail_concurrency", DEFAULT_THUMBNAIL_CONCURRENCY))  # noqa: E731
            idle = lambda: bool(getattr(self._limits(), "idle_work", True))  # noqa: E731
            q = PriorityWorkQueue(self._jobs, owner=project.project_id, run=lambda payload, cancel: self._generate(project, payload, cancel), concurrency=lim_c, idle_allowed=idle,
                                  pressure_ok=self._pressure_ok, on_done=self._thumb_done, on_failed=self._thumb_failed, on_idle=self._thumb_batch_finished)
            self._queue, self._queue_project = q, project
            self._thumb_missing.clear()
        return q

    def _generate(self, project: Project, asset_id: str, cancel: Callable[[], bool]) -> bool:
        asset = project.assets.get(asset_id)
        if asset is None or project.root is None or self._projects.current is not project:
            return False
        _path, generated = self._thumbnails.ensure_ex(project.root, asset, cancel, self._cache())
        return generated

    def _thumb_done(self, asset_id: str, _payload, generated) -> None:
        self._thumb_missing.discard(asset_id)
        if generated:
            self._bus.publish(Topics.THUMBNAIL_READY, asset_id=asset_id)

    def _thumb_failed(self, asset_id: str, _payload, reason: str) -> None:
        if reason.startswith("Media file is missing"):
            self._thumb_missing.add(asset_id)
        self._bus.publish(THUMBNAIL_FAILED, asset_id=asset_id, reason=reason, missing_source=asset_id in self._thumb_missing)

    def _thumb_batch_finished(self, stats) -> None:
        if not stats.batch_failed:
            return
        first = next(iter(stats.batch_reasons), "")
        log_event(_log, "media.thumbnails_failed", count=stats.batch_failed, created=stats.batch_done)
        self._bus.publish(Topics.STATUS, message=f"{stats.batch_failed} thumbnail(s) could not be created ({first.split(chr(10))[0][:80]}). Use “Retry” on the item to try again.")

    def _on_project_event(self, topic: str, payload: dict) -> None:
        q = self._queue
        if q is None:
            return
        cur = self._projects.current
        if topic == Topics.PROJECT_CLOSED or cur is None or cur is not self._queue_project:
            q.close()
            self._queue = None
        elif topic == Topics.PROJECT_CHANGED and payload.get("scope") == "assets" and q.pending():
            q.cancel([k for k in q.keys() if cur.assets.get(k) is None])  # an asset that was removed no longer needs its thumbnail

    def thumbnail_file(self, asset: Asset) -> Path | None:
        project = self._projects.current
        if project is None or project.root is None:
            return None
        return self._thumbnails.thumbnail_path(project.root, asset) if self._thumbnails.is_cached(project.root, asset) else None

    def request_thumbnails(self, asset_ids, priority: Priority = Priority.MEDIUM) -> int:
        """Queue thumbnails for these assets (deduplicated; cheap for ones that already have a current thumbnail). Returns how many tasks were queued.

        ``Priority.LOW`` is the bulk / idle path and only checks that a thumbnail file exists; MEDIUM and HIGH also verify it against the source file."""
        project = self._projects.current
        if project is None or project.root is None:
            return 0
        root, cache, q = project.root, self._cache(), self._queue_for(project)
        full = priority is not Priority.LOW
        batch: list[tuple[str, str]] = []
        for aid in asset_ids:
            asset = project.assets.get(aid)
            if asset is None:
                continue
            try:
                if (self._thumbnails.is_current(root, asset, cache) if full else self._thumbnails.is_cached(root, asset)):
                    continue
            except OSError:
                pass
            if q.failure(aid) is not None and priority is Priority.LOW:
                continue  # a failed one is not retried by bulk requests (retry_thumbnail does that)
            batch.append((aid, aid))
        return q.submit_many(batch, priority) if batch else 0

    def set_visible_assets(self, asset_ids) -> None:
        """The library calls this when the visible rows change: those thumbnails run first; ones that scrolled away (and were only queued for being visible) are dropped."""
        project = self._projects.current
        if project is None or project.root is None:
            return
        root, cache, q = project.root, self._cache(), self._queue_for(project)

        def payload(aid: str):
            asset = project.assets.get(aid)
            if asset is None or self._thumbnails.is_current(root, asset, cache):
                return None
            return aid

        q.set_visible(list(asset_ids), payload)

    def thumbnail_state(self, asset_id: str) -> str:
        """``ready`` | ``pending`` (queued, running or not yet requested) | ``failed`` | ``missing_source``."""
        project = self._projects.current
        if project is None or project.root is None:
            return "pending"
        q = self._queue
        st = q.state(asset_id) if q is not None else None
        if st in ("pending", "running"):
            return "pending"
        if st == "failed":
            return "missing_source" if asset_id in self._thumb_missing else "failed"
        asset = project.assets.get(asset_id)
        if asset is not None and self._thumbnails.is_cached(project.root, asset):
            return "ready"
        return "pending"

    def thumbnail_failure(self, asset_id: str) -> str:
        q = self._queue
        return (q.failure(asset_id) or "") if q is not None else ""

    def retry_thumbnail(self, asset_id: str) -> bool:
        project = self._projects.current
        if project is None or project.assets.get(asset_id) is None:
            return False
        q = self._queue_for(project)
        self._thumb_missing.discard(asset_id)
        return bool(q.retry(asset_id, asset_id, Priority.MEDIUM))

    def cancel_thumbnails(self) -> int:
        q = self._queue
        return q.cancel_all() if q is not None else 0

    def thumbnail_stats(self) -> dict:
        q = self._queue
        return dict(q.stats.__dict__, pending=q.pending()) if q is not None else {}

    def ensure_thumbnail(self, asset: Asset) -> Job | None:
        """Queue this asset's thumbnail (REQUESTED priority). Returns the queue's worker job when one is running or queued."""
        project = self._project()
        if project.assets.get(asset.id) is None:
            return None
        self.request_thumbnails([asset.id], Priority.MEDIUM)
        q = self._queue
        return q.current_job() if q is not None else None

    def ensure_all_thumbnails(self) -> None:
        """Queue any thumbnails that are missing at idle priority (existing ones are skipped; nothing is generated here, on the calling thread)."""
        project = self._projects.current
        if project is None:
            return
        try:
            self.request_thumbnails([a.id for a in project.assets], Priority.LOW)
        except AppError:
            _log.warning("Could not schedule thumbnails", exc_info=True)
