"""The application facade the UI talks to.

``Workspace`` wires every service together and owns cross-cutting policy: dirty tracking,
autosave triggers, recovery, settings and the undo stack lifecycle. The UI never reaches
into project files, FFmpeg or the filesystem on its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from app.ai.provider import AIProviderRegistry
from app.core.commands import Command, CommandStack, CompositeCommand
from app.core.config import Settings, SettingsStore
from app.core.constants import DEFAULT_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_RESOLUTION, resolve_dimensions
from app.core.events import EventBus, Topics
from app.core.exceptions import FFmpegUnavailableError, ProjectError, RecoveryError
from app.jobs.job import Job
from app.jobs.job_manager import Dispatcher, JobManager
from app.logging.logger import get_logger, log_event
from app.media.importer import MediaImporter
from app.media.media_probe import MediaProber, locate_binary, run_process
from app.media.thumbnails import ThumbnailService
from app.project.autosave import AutosaveService
from app.project.project import Project
from app.project.project_commands import SetScriptCommand
from app.project.project_manager import ProjectManager
from app.project.project_schema import ProjectSettings
from app.project.recovery import RecoveryEntry, RecoveryManager
from app.services.media_service import MediaService
from app.services.scene_service import SceneService
from app.services.timeline_service import TimelineService
from app.services.transcript_service import TranscriptService
from app.visual.preferences import VisualPreferences
from app.visual.research import VisualResearchService
from app.services.research_service import ResearchService
from app.services.editing_service import EditingService
from app.services.presentation_service import PresentationService
from app.qc.qc_service import QCService
from app.performance.change_tracker import ProjectChangeTracker
from app.services.reference_service import ReferenceService
from app.services.render_service import RenderService
from app.services.performance_service import PerformanceService
from app.project.phase2_commands import SetVisualPreferencesCommand
from app.storage.paths import AppPaths

_log = get_logger(__name__)


class Workspace:
    def __init__(self, paths: AppPaths | None = None, dispatcher: Dispatcher | None = None) -> None:
        self.paths = paths or AppPaths.default()
        self.paths.ensure()
        self.bus = EventBus()
        self.settings_store = SettingsStore(self.paths.settings_file)
        self.settings: Settings = self.settings_store.load()

        self.commands = CommandStack(self.bus)
        self.jobs = JobManager(self.bus, dispatcher=dispatcher)
        self.projects = ProjectManager(self.bus, self.paths.recent_projects_file)
        self.changes = ProjectChangeTracker(self.bus, lambda: self.projects.current, lambda: getattr(getattr(self, "performance", None), "cache", None))  # what each edit can have made stale (incremental QC / preview / cache invalidation)
        self.prober = MediaProber(self.settings.ffprobe_path)
        self.thumbnails = ThumbnailService(self.settings.ffmpeg_path)
        self.importer = MediaImporter(self.prober)
        self.recovery = RecoveryManager(self.paths.recovery_dir)
        self.autosave = AutosaveService(
            self.recovery, on_error=lambda msg: self.bus.publish(Topics.ERROR, message=msg, title="Autosave")
        )
        self.ai = AIProviderRegistry()

        self.media = MediaService(
            self.projects, self.commands, self.jobs, self.importer, self.thumbnails, self.bus, self.apply_command
        )
        self.timeline = TimelineService(self.projects, self.commands)
        self.transcripts = TranscriptService(self.projects, self.jobs, self.bus, self.apply_command, lambda: self.settings)
        self.scenes = SceneService(self.projects, self.commands, self.jobs, self.bus)
        self.research: ResearchService = ResearchService(self.projects, self.commands, self.jobs, self.bus, self.apply_command,
                                                         self.media, self.importer, lambda: self.settings)
        self.editing = EditingService(self.projects, self.commands, self.jobs, self.bus, self.apply_command,
                                      lambda pid: self.paths.data_dir / "checkpoints" / pid)
        self.presentation = PresentationService(self.projects, self.commands, self.jobs, self.bus, self.apply_command, lambda: self.settings.ffmpeg_path,
                                                self.media, self.editing._checkpoint)
        self.render = RenderService(self.projects, self.jobs, self.bus, self.apply_command, self.commands.execute, lambda: self.settings, self.media, self.editing._checkpoint,
                                    lambda project: self.autosave.request(project, force=True))
        self.reference = ReferenceService(self.projects, self.jobs, self.bus, self.apply_command, self.commands.execute, lambda: self.settings, self.editing._checkpoint)
        self.qc = QCService(self.projects, self.jobs, self.bus, self.apply_command, self.commands.execute, lambda: self.settings, self.editing._checkpoint)
        self.qc.tracker = self.changes
        self.performance = PerformanceService(self.projects, self.jobs, self.bus, self.commands.execute, lambda: self.settings, self.settings_store.save,
                                              lambda: self.render.engine.hardware, proxy_in_use=lambda p: any(r.proxy_path == str(p) for r in self.render.proxies.records().values() if r.proxy_status in ("READY", "QUEUED")))
        self.render.qc_gate = self.qc.export_gate  # the export is blocked only by a QC run that still matches the project
        self._install_fix_engine()
        self.bus.subscribe(Topics.RENDER_HISTORY_CHANGED, self._on_render_history)
        self.timeline.edit_hook = self._edit_hook
        self.selected_clip_id: str | None = None

        self.bus.subscribe(Topics.PROJECT_CHANGED, self._on_project_changed)

    def _install_fix_engine(self) -> None:
        """QC fixes are undoable commands built from the editing / presentation / timeline / render services (kept optional: QC analysis works without the fix engine)."""
        try:
            from types import SimpleNamespace  # noqa: PLC0415

            from app.qc.fix_engine import QCFixEngine  # noqa: PLC0415

            self.qc.fixes = QCFixEngine(self.projects, self.commands.execute, self.editing._checkpoint,
                                        SimpleNamespace(editing=self.editing, presentation=self.presentation, timeline=self.timeline, render=self.render, research=self.research, media=self.media))
        except ImportError:
            self.qc.fixes = None

    def _on_render_history(self, _topic: str, payload: dict) -> None:
        """A render was recorded: if it is a finished export, check the rendered file (post-render QC). Runs on the render worker thread: hand off to the callback thread."""
        render_id = payload.get("render_id")
        if render_id:
            self.jobs.dispatch(lambda: self._post_render_qc(render_id))

    def _post_render_qc(self, render_id: str) -> None:
        from app.rendering.models import RenderStatus  # noqa: PLC0415

        project = self.projects.current
        if project is None or project.root is None or not project.qc_settings.post_render_qc or render_id in project.render_qc_results:
            return
        job = self.render.job(render_id)
        if job is None or job.status is not RenderStatus.COMPLETED or job.record.kind != "export" or job.result is None:
            return
        out = Path(job.result.output_path)
        if not out.is_file() or project.project_id != job.record.project_id:
            return
        try:
            self.qc.run_post_render_qc(render_id, out, snapshot=getattr(getattr(job, "spec", None), "snapshot", None))  # measured against what was rendered, not the project as it is now
        except Exception:  # noqa: BLE001  (the second QC pass is advisory: it must never disturb the finished export)
            _log.warning("post-render QC could not start", exc_info=True)

    def _edit_hook(self, clip_id: str, action: str) -> Command | None:
        """Manual timeline edits take ownership of AI-created objects (AI edit decisions and presentation decisions)."""
        cmds = [c for c in (self.editing.override_command(clip_id, action), self.presentation.override_command(clip_id, action)) if c is not None]
        if not cmds:
            return None
        return cmds[0] if len(cmds) == 1 else CompositeCommand("Take ownership", cmds, scope="timeline")

    # ------------------------------------------------------------ state
    @property
    def project(self) -> Project | None:
        return self.projects.current

    def require_project(self) -> Project:
        if self.projects.current is None:
            raise ProjectError("Open or create a project first.")
        return self.projects.current

    # ------------------------------------------------------------ commands
    def apply_command(self, command: Command) -> None:
        """Apply a change that is a *project edit* but should not appear in undo history (e.g. import)."""
        command.do()
        self.bus.publish(Topics.PROJECT_CHANGED, scope=command.scope, command=command, action="do")

    def _on_project_changed(self, topic: str, payload: dict) -> None:
        project = self.projects.current
        if project is None:
            return
        self.projects.set_dirty(True)
        command = payload.get("command")
        if command is None or getattr(command, "major", True):
            self.autosave.request(project)  # after major operations
        if self.selected_clip_id and project.timeline.get_clip(self.selected_clip_id) is None:
            self.select_clip(None)

    def undo(self) -> None:
        self.commands.undo()

    def redo(self) -> None:
        self.commands.redo()

    def select_clip(self, clip_id: str | None) -> None:
        if clip_id != self.selected_clip_id:
            self.selected_clip_id = clip_id
            self.bus.publish(Topics.SELECTION_CHANGED, clip_id=clip_id)

    # ------------------------------------------------------------ project lifecycle
    def new_project(
        self,
        name: str,
        location: Path | None = None,
        resolution: str = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    ) -> Project:
        old = self.projects.current
        w, h = resolve_dimensions(resolution, aspect_ratio)
        project = self.projects.create(
            name, location or Path(self.settings.default_project_location), ProjectSettings(w, h, fps, aspect_ratio)
        )
        self._reset_session(old)
        self._after_open()
        return project

    def open_project(self, path: Path, from_backup: bool = False) -> Project:
        old = self.projects.current
        project = self.projects.open(path, from_backup=from_backup)
        self._reset_session(old)
        self._after_open()
        return project

    def _after_open(self) -> None:
        project = self.require_project()
        self.media.ensure_all_thumbnails()
        for asset in project.missing_assets():
            self.bus.publish(Topics.STATUS, message=f"Media file missing: {asset.name}")

    def _reset_session(self, old: Project | None) -> None:
        """Drop session state belonging to ``old`` (the project that was just replaced or closed)."""
        current = self.projects.current
        if old is not None:
            if current is None or current.project_id != old.project_id:
                self.autosave.clear(old.project_id)
            for job in self.jobs.active_jobs():
                self.jobs.cancel(job.id)
        self.commands.clear()
        self.select_clip(None)

    def save(self) -> None:
        project = self.require_project()
        self.projects.save(project)
        self.autosave.clear(project.project_id)
        self.bus.publish(Topics.STATUS, message=f"Saved “{project.project_name}”")

    def save_as(self, new_parent: Path, new_name: str, on_done: Callable[[Project], None] | None = None) -> Job:
        """Copy the project to a new location in a background job, then switch to the copy."""
        project = self.require_project()
        project.validate()

        def work(ctx) -> Project:
            return self.projects.save_as(
                new_parent, new_name, project, progress=lambda f, m: ctx.report(f * 100, m)
            )

        def done(job: Job) -> None:
            old = self.projects.current
            self.projects.switch_to(job.result)
            self._reset_session(old)
            self._after_open()
            self.bus.publish(Topics.STATUS, message=f"Saved as “{job.result.project_name}”")
            if on_done:
                on_done(job.result)

        def failed(job: Job) -> None:
            self.bus.publish(Topics.ERROR, message=job.error or "Save As failed.", title="Save As")

        return self.jobs.submit("project.save_as", work, title=f"Saving project as {new_name}", on_complete=done, on_error=failed)

    def close_project(self) -> None:
        """Close the current project. The caller is responsible for asking about unsaved changes."""
        old = self.projects.current
        self.projects.close()
        self._reset_session(old)

    # ------------------------------------------------------------ script
    def set_script(self, text: str) -> None:
        project = self.require_project()
        if text != project.script.text:
            self.commands.execute(SetScriptCommand(project, text))

    # ------------------------------------------------------------ visual preferences
    def set_visual_preferences(self, prefs: VisualPreferences) -> None:
        self.commands.execute(SetVisualPreferencesCommand(self.require_project(), prefs))

    # ------------------------------------------------------------ autosave & recovery
    def autosave_tick(self) -> bool:
        """Periodic autosave (driven by a timer in the UI)."""
        return self.autosave.request(self.projects.current)

    def pending_recovery(self) -> list[RecoveryEntry]:
        return self.recovery.list_entries()

    def recover(self, project_id: str) -> Project:
        old = self.projects.current
        project = self.recovery.load_project(project_id)
        self.projects.switch_to(project)
        self._reset_session(old)
        self.projects.set_dirty(True)  # recovered state is NOT saved until the user chooses to
        self.media.ensure_all_thumbnails()
        log_event(_log, "recovery.applied", project_id=project_id)
        return project

    def discard_recovery(self, project_id: str) -> None:
        self.recovery.discard(project_id)

    # ------------------------------------------------------------ settings
    def update_settings(self, settings: Settings) -> None:
        settings = settings.sanitized()
        self.settings_store.save(settings)
        self.settings = settings
        self.prober.configure(settings.ffprobe_path)  # providers read settings lazily through getters
        self.thumbnails.configure(settings.ffmpeg_path)
        self.render.reconfigure()
        self.bus.publish(Topics.STATUS, message="Settings saved")

    def describe_ffmpeg(self, configured_path: str = "") -> str:
        """Locate FFmpeg (optionally at ``configured_path``) and return its version line. Raises ``AppError``."""
        binary = locate_binary("ffmpeg", configured_path)
        try:
            first = run_process([binary, "-version"], timeout=10).stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            raise FFmpegUnavailableError("FFmpeg was found but could not be run.", details=str(exc)) from exc
        return first[0] if first else binary

    # ------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        project = self.projects.current
        if project is not None and not project.dirty:
            self.autosave.clear(project.project_id)  # clean exit leaves no recovery data
        self.render.shutdown()
        self.performance.shutdown()
        self.jobs.shutdown()
        self.autosave.shutdown()
