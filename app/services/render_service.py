"""Application service for rendering, proxies, previews and export (Phase 6).

    UI -> RenderService -> RenderEngine (planner, compiler, FFmpeg service, executor, validator, output manager) -> project storage

The UI never builds FFmpeg commands, starts FFmpeg, touches render files or edits project JSON: it calls this service. The timeline stays the
source of truth; a render works from a frozen ``RenderSnapshot`` and writes a *copy* (the MP4) plus its log and metadata.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

from app.performance.profiler import profiler
from app.core.commands import Command
from app.core.events import EventBus, Topics
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.project.project_schema import RenderSettings
from app.rendering import presets as P
from app.rendering.commands import DeleteRenderRecordCommand, RenderRecordCommand, SetRenderSettingsCommand
from app.rendering.engine import RenderEngine
from app.rendering.errors import RenderError
from app.rendering.executor import JobSpec
from app.rendering.models import PreflightReport, RenderRecord, RenderSnapshot, RenderStatus, new_id, now_iso
from app.rendering.output import OutputManager
from app.rendering.preview import PREVIEW_MODES, PreviewEngine, PreviewPlan, PreviewResult
from app.rendering.proxy import ProxyManager
from app.rendering.queue import RenderJob, RenderQueue
from app.rendering.relink import MediaRelinkService

_log = get_logger(__name__)


class PreflightFailed(RenderError):
    def __init__(self, report: PreflightReport) -> None:
        first = report.errors[0]
        super().__init__(f"The project is not ready to export: {first.message}", stage="Validating Project", kind="preflight_failed", possible_issue=first.fix, asset_ids=first.asset_ids)
        self.report = report


class QCGateBlocked(RenderError):
    """The latest, still-current quality-control run blocks the export (Critical issues, or Errors / Warnings according to the project's QC setting)."""

    def __init__(self, message: str, *, overridable: bool = False, issue_ids: list[str] | None = None) -> None:
        super().__init__(message, stage="Quality Control", kind="qc_blocked", possible_issue="Fix the blocking issues in AI Quality Control, then export again.")
        self.overridable = overridable
        self.issue_ids = issue_ids or []


class RenderService:
    def __init__(self, projects: ProjectManager, jobs: JobManager, bus: EventBus, apply_command: Callable[[Command], None], execute_command: Callable[[Command], object],
                 settings_getter: Callable, media_service, checkpoint: Callable[[Project, str], str], autosave: Callable[[Project], None]) -> None:
        self._projects, self._jobs, self._bus, self._apply = projects, jobs, bus, apply_command
        self._settings = settings_getter
        self._checkpoint, self._autosave = checkpoint, autosave
        self.engine = RenderEngine(lambda: self._settings().ffmpeg_path, lambda: self._settings().ffprobe_path)
        self.proxies = ProxyManager(self._project, jobs, bus, apply_command, self.engine.ffmpeg, self.engine.probe)
        self.proxies.in_use = self._proxy_in_use
        self.relink = MediaRelinkService(self._project, execute_command, self.engine.probe)  # relinking is a normal, undoable edit
        self.preview = PreviewEngine(self.engine, lambda: self._projects.current.root if self._projects.current else None)
        self.queue = RenderQueue(self.engine, self._on_update, self._on_finished, max_parallel=1)
        self._outputs: dict[str, Path] = {}
        self.qc_gate: Callable[[], object] | None = None  # installed by the Workspace: returns app.qc.qc_service.GateResult (blocks only when a *current* QC run says so)
        self.chunk_seconds = 30.0  # target length of one cached video section (scene aligned)
        self.chunk_max_seconds: float | None = None

    def _proxy_in_use(self, path: Path) -> bool:
        """A render or preview that has not finished may be reading this proxy (its snapshot lists it): it must not be removed under it."""
        import os  # noqa: PLC0415

        p = os.path.normcase(os.path.abspath(path))
        for j in self.queue.jobs():
            if not j.status.is_terminal and any(r.path and os.path.normcase(os.path.abspath(r.path)) == p for r in j.spec.snapshot.proxies.values()):
                return True
        return False

    # ------------------------------------------------------------------ project / settings
    def _project(self) -> Project:
        from app.core.exceptions import ProjectError

        if self._projects.current is None:
            raise ProjectError("Open or create a project first.")
        return self._projects.current

    @property
    def settings(self) -> RenderSettings:
        return self._project().render_settings

    def update_settings(self, **changes) -> RenderSettings:
        """Change the saved export settings (any field of ``RenderSettings``). Choosing a value by hand makes the preset ``custom``."""
        cur = self._project().render_settings
        new = replace(cur, **changes)
        if "preset_id" not in changes and new != cur:
            new = replace(new, preset_id="custom")
        problems = P.compatibility_problems(new)
        if problems:
            raise RenderError(problems[0], kind="invalid_settings", details="; ".join(problems))
        self._apply(SetRenderSettingsCommand(self._project(), new))
        return new

    def apply_preset(self, preset_id: str) -> RenderSettings:
        new = P.apply_preset(self._project().render_settings, preset_id)
        self._apply(SetRenderSettingsCommand(self._project(), new, f"Export preset: {preset_id}"))
        return new

    def reconfigure(self) -> None:
        self.engine.reconfigure()

    def capabilities(self) -> dict:
        """What this computer can encode (shown in the Export screen so unavailable choices are explained, not hidden)."""
        ok, msg = self.engine.ffmpeg.detect()
        out: dict = {"ffmpeg_ok": ok, "message": msg, "video_codecs": {}, "audio_codecs": {}, "hardware": {}}
        if not ok:
            return out
        caps = self.engine.ffmpeg.capabilities()
        for c in P.CODECS:
            enc = next((e for e in P.SOFTWARE_ENCODERS[c] if e in caps.encoders), "")
            out["video_codecs"][c] = {"available": bool(enc), "encoder": enc, "label": P.CODEC_LABELS[c]}
        for a, e in (("aac", "aac"), ("opus", "libopus"), ("flac", "flac")):
            out["audio_codecs"][a] = {"available": e in caps.encoders, "encoder": e, "label": P.AUDIO_LABELS[a]}
        out["hardware"] = self.engine.ffmpeg.hardware_encoders()
        return out

    # ------------------------------------------------------------------ snapshot / preflight
    def snapshot(self, settings: RenderSettings | None = None) -> RenderSnapshot:
        p = self._project()
        return RenderSnapshot.from_project(p, settings or p.render_settings, self.proxies.refs())

    def output_dir(self) -> Path:
        p = self._project()
        d = Path(p.render_settings.output_dir) if p.render_settings.output_dir else p.root / "renders"  # type: ignore[operator]
        return d

    def preflight(self, output: Path | None = None, output_dir: Path | None = None, allow_proxy_assets: set[str] | None = None, settings: RenderSettings | None = None) -> PreflightReport:
        p = self._project()
        snap = self.snapshot(settings)
        diag = self.engine.diagnostics(self.output_dir())
        return diag.run(snap, output_dir or None, output, p.root / "cache" / "render", allow_proxy_assets)  # type: ignore[operator]

    def preflight_async(self, on_done: Callable[[PreflightReport], None], output: Path | None = None, allow_proxy_assets: set[str] | None = None,
                        settings: RenderSettings | None = None) -> Job:
        """The preflight check in the background (it probes every media file). The snapshot is taken now, on the caller's thread."""
        p = self._project()
        snap = self.snapshot(settings)
        diag = self.engine.diagnostics(self.output_dir())
        cache = p.root / "cache" / "render"  # type: ignore[operator]
        allow = set(allow_proxy_assets or ())

        def work(ctx) -> PreflightReport:
            return diag.run(snap, None, output, cache, allow)

        return self._jobs.submit("render.preflight", work, title="Checking the project before export", on_complete=lambda j: on_done(j.result))

    # ------------------------------------------------------------------ export
    @profiler.timed("render.start")
    def start_export(self, output: Path | None = None, *, overwrite: bool = False, allow_proxy_assets: set[str] | None = None, settings: RenderSettings | None = None,
                     kind: str = "export", force_cpu: bool = False, skip_preflight: bool = False, qc_override: bool = False) -> RenderJob:
        """Autosave, checkpoint, preflight, freeze a snapshot and queue the render. Raises ``PreflightFailed`` when something blocks it and ``QCGateBlocked`` when the current
        quality-control result blocks the export (``qc_override`` is the user's explicit "export anyway", honoured only where the QC settings allow it)."""
        project = self._project()
        if kind == "export" and not skip_preflight:
            self._check_qc_gate(qc_override)
        allow = set(allow_proxy_assets or ())
        settings = replace(settings or project.render_settings)
        if not skip_preflight:
            report = self.preflight(output, None, allow, settings)
            if not report.can_start:
                raise PreflightFailed(report)
        # ---- autosave + timeline checkpoint + immutable snapshot (the render never reads the live project again)
        self._autosave(project)
        try:
            self._checkpoint(project, "before_render")
        except OSError:
            _log.warning("checkpoint before render failed", exc_info=True)
        snap = self.snapshot(settings)
        resolved = self.engine.resolve(settings, snap, force_cpu=force_cpu)
        outputs = OutputManager(self.output_dir())
        target = outputs.reserve(project.project_name, resolved, self.output_dir() / "draft" if kind == "draft" else None, kind=kind, overwrite=overwrite, explicit=output)
        render_id = new_id("render")
        root: Path = project.root  # type: ignore[assignment]
        spec = JobSpec(render_id, snap, resolved, target, root / "cache" / "render" / render_id, root / "cache" / "render", root / "renders" / render_id, kind,
                       "final", allow, self.chunk_seconds, self.chunk_max_seconds, True, settings.duration_tolerance)
        rec = RenderRecord(render_id, project.project_id, snap.timeline_version, snap.content_hash(), now_iso(), RenderStatus.QUEUED.value, settings.to_dict(), resolved.summary(),
                           str(target), kind=kind, log_path=str(spec.run_dir / "render.log"), snapshot_path=str(spec.run_dir / "snapshot.json"), proxy_used=bool(allow) or settings.use_proxies)
        job = RenderJob(render_id, f"{'Draft' if kind == 'draft' else 'Export'}: {target.name}", spec, rec, settings, force_cpu=force_cpu)
        self._outputs[render_id] = target
        self._persist(rec, snap.project_id)
        log_event(_log, "render.queued", render_id=render_id, output=str(target), kind=kind)
        self.queue.submit(job)
        return job

    def _check_qc_gate(self, override: bool) -> None:
        if self.qc_gate is None:
            return
        try:
            gate = self.qc_gate()
        except Exception:  # noqa: BLE001  (QC is advisory infrastructure: if it cannot answer, the render path must not break)
            _log.warning("QC gate unavailable", exc_info=True)
            return
        if gate is None or not getattr(gate, "blocked", False):
            return
        d = gate.decision
        if override and d.overridable:
            log_event(_log, "render.qc_override", blocking=len(d.blocking_ids))
            return
        raise QCGateBlocked(d.message, overridable=d.overridable, issue_ids=list(d.blocking_ids))

    def start_draft(self, settings: RenderSettings | None = None) -> RenderJob:
        s = P.apply_preset(settings or self._project().render_settings, "draft")
        return self.start_export(settings=s, kind="draft")

    # ------------------------------------------------------------------ queue control
    def jobs(self) -> list[RenderJob]:
        return self.queue.jobs()

    def job(self, job_id: str) -> RenderJob | None:
        return self.queue.get(job_id)

    def cancel(self, job_id: str) -> bool:
        return self.queue.cancel(job_id)

    def pause(self, job_id: str) -> bool:
        return self.queue.pause(job_id)

    def resume(self, job_id: str) -> bool:
        return self.queue.resume(job_id)

    def remove(self, job_id: str) -> bool:
        return self.queue.remove(job_id)

    def retry(self, job_id: str, *, cpu_fallback: bool = False, allow_proxy_assets: set[str] | None = None) -> RenderJob:
        """Render again with the same settings from the *current* project (so a relinked asset or a fix is picked up). ``cpu_fallback`` re-runs a failed hardware render on the CPU."""
        old = self.queue.get(job_id)
        if old is None or not old.status.is_terminal or old.status is RenderStatus.COMPLETED:
            raise RenderError("Only a failed or cancelled render can be retried.", kind="invalid_state")
        return self.start_export(old.output_path, settings=old.settings, kind=old.record.kind, force_cpu=cpu_fallback or old.force_cpu, allow_proxy_assets=allow_proxy_assets)

    # ------------------------------------------------------------------ events and records
    def _on_update(self, job: RenderJob) -> None:
        job.record.status = job.status.value
        self._bus.publish(Topics.RENDER_UPDATED, job=job)

    def _on_finished(self, job: RenderJob) -> None:
        rec = job.record
        rec.status = job.status.value
        rec.finished_at = job.finished_at or now_iso()
        if job.result is not None:
            r = job.result
            rec.output_path, rec.duration, rec.size_bytes = str(r.output_path), r.duration, r.size_bytes
            rec.width, rec.height, rec.fps = r.plan.output_resolution[0], r.plan.output_resolution[1], float(r.plan.fps)
            rec.warnings, rec.validation, rec.proxy_used = list(r.warnings), r.report.to_dict(), bool(r.used_proxy_assets)
        if job.error is not None:
            rec.error, rec.failed_stage, rec.possible_issue = job.error.user_message, job.error.stage, job.error.possible_issue
        OutputManager.release(self._outputs.pop(job.id, job.spec.output_path))
        try:
            OutputManager.save_metadata(job.spec.run_dir, rec.to_dict())
        except OSError:
            _log.warning("could not write render metadata", exc_info=True)
        self._persist(rec, job.spec.snapshot.project_id)
        log_event(_log, "render.finished", render_id=job.id, status=job.status.value)
        self._bus.publish(Topics.RENDER_HISTORY_CHANGED, render_id=job.id)

    def _persist(self, rec: RenderRecord, project_id: str) -> None:
        """Save the record in the project's render history (on the thread that owns the project). If another project is open now, the metadata file still has it."""
        data = rec.to_dict()

        def apply() -> None:
            p = self._projects.current
            if p is not None and p.project_id == project_id:
                self._apply(RenderRecordCommand(p, data))

        self._jobs.dispatch(apply)

    def history(self) -> list[RenderRecord]:
        return [RenderRecord.from_dict(d) for d in self._project().render_history]

    def delete_record(self, render_id: str) -> None:
        self._apply(DeleteRenderRecordCommand(self._project(), render_id))
        self._bus.publish(Topics.RENDER_HISTORY_CHANGED, render_id=render_id)

    def read_log(self, render_id: str, tail: int = 400) -> str:
        p = self._project()
        f = p.root / "renders" / render_id / "render.log"  # type: ignore[operator]
        if not f.is_file():
            return ""
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    def log_path(self, render_id: str) -> Path:
        return self._project().root / "renders" / render_id / "render.log"  # type: ignore[operator]

    def open_location(self, path: Path) -> bool:
        return OutputManager.open_location(Path(path))

    # ------------------------------------------------------------------ preview
    @property
    def preview_modes(self):
        return PREVIEW_MODES

    def preview_plan(self, mode: str = "draft") -> PreviewPlan:
        return self.preview.plan(self.snapshot(), mode)

    def build_preview(self, mode: str = "draft", scene_ids: list[str] | None = None, on_done: Callable[[PreviewResult], None] | None = None,
                      on_error: Callable[[Job], None] | None = None) -> Job:
        """Render the preview in the background. With ``scene_ids`` only those scenes (cached sections of other scenes are not touched)."""
        snap = self.snapshot()
        ranges = None
        if scene_ids:
            ranges = [(s.start, s.end) for s in snap.scenes if s.id in set(scene_ids)] or None

        def work(ctx) -> PreviewResult:
            return self.preview.build(snap, mode, ranges, ctx.job.cancel_event, lambda p: ctx.report(p.overall * 100.0, f"{p.stage.value}"))

        def done(job: Job) -> None:
            if on_done:
                on_done(job.result)

        return self._jobs.submit("render.preview", work, title=f"Preview ({PREVIEW_MODES[mode].label})", on_complete=done, on_error=on_error)

    # ------------------------------------------------------------------ housekeeping
    def clear_render_cache(self) -> int:
        """Delete cached sections and audio mixes (never renders, proxies or originals)."""
        root = self._project().root
        n = 0
        for sub in ("chunks", "audio"):
            d = root / "cache" / "render" / sub  # type: ignore[operator]
            if d.is_dir():
                n += sum(1 for _ in d.iterdir())
                shutil.rmtree(d, ignore_errors=True)
        return n + self.preview.clear()

    def shutdown(self) -> None:
        self.queue.shutdown()


_ = uuid
