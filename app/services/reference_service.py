"""Application service for reference-video style analysis (Phase 7).

    UI -> ReferenceService -> ReferenceVideoAnalyzer -> ReferenceFeatureExtractor -> ReferenceStyleModel -> ReferenceStyleAdapter -> EditingStrategyOverrides

The reference is *analysis input only*: it lives in ``<project>/references/<id>/`` (never in the media library, never on the timeline), the analysis never
changes the timeline, and applying a style changes only abstract AI-editing preferences that the next Phase 4 / Phase 5 run reads.

    REFERENCE VIDEO -> ABSTRACT STYLE -> EDITING PARAMETERS -> USER'S OWN CONTENT      (never: reference -> copy -> user video)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from app.core.commands import Command
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, JobCancelled, ProjectError
from app.editing.overrides import EditingStrategyOverrides
from app.jobs.job import Job, JobStatus
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.phase7_commands import (
    ApplyStyleCommand, ClearStyleCommand, RegisterReferenceCommand, RemoveReferenceCommand, SetReferenceSettingsCommand, StoreAnalysisCommand,
)
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.reference.analyzer import REFERENCE_EXTENSIONS, ReferenceAnalysis, ReferenceVideoAnalyzer, file_hash
from app.reference.application import (
    APPLY_NOTICE, MODES, STRENGTHS, ReferenceAsset, ReferenceSettings, StyleAdjustments, StyleApplication,
)
from app.reference.feature_extractor import AnalysisSettings
from app.reference.signals import AnalysisCancelled, ReferenceAnalysisError
from app.reference.style_model import (
    ANALYSIS_VERSION, DIMENSIONS, ComparisonRow, ReferenceStyleProfile, StyleSimilarityScore, compare, now_iso,
)
from app.rendering.ffmpeg_service import FFmpegService
from app.storage.atomic import atomic_write_text

_log = get_logger(__name__)
THUMBNAILS = 6
ANALYSIS_FILE = "analysis.json"
LOG_FILE = "analysis.log"
MAX_LOG_LINES = 400


class ReferenceError(AppError):
    """A reference video could not be imported, analysed or applied."""


@dataclass
class StylePlan:
    """What applying the style *would* do (nothing has been changed yet): the abstract overrides, why, and the project's expected new reading."""

    reference_id: str
    overrides: EditingStrategyOverrides
    effective_targets: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    settings: ReferenceSettings = field(default_factory=ReferenceSettings)
    projected: object | None = None  # ProjectedStyle (reference/style_adapter.py)
    rows: list[ComparisonRow] = field(default_factory=list)
    similarity: StyleSimilarityScore | None = None
    notice: str = APPLY_NOTICE


class ReferenceService:
    def __init__(self, projects: ProjectManager, jobs: JobManager, bus: EventBus, apply_command: Callable[[Command], None], execute_command: Callable[[Command], object],
                 settings_getter: Callable, checkpoint: Callable[[Project, str], str]) -> None:
        self._projects, self._jobs, self._bus, self._apply, self._execute = projects, jobs, bus, apply_command, execute_command
        self._settings, self._checkpoint = settings_getter, checkpoint
        self._ffmpeg = FFmpegService(lambda: self._settings().ffmpeg_path, lambda: self._settings().ffprobe_path)
        self.analysis_settings = AnalysisSettings()
        self._jobs_by_ref: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._cache: dict[str, ReferenceAnalysis] = {}  # in-memory copy of loaded analyses (reference id -> analysis)
        bus.subscribe(Topics.PROJECT_CLOSED, lambda _t, _p: self._drop_memory_cache())  # the in-memory copies belong to the project that was open

    def _drop_memory_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._jobs_by_ref.clear()

    # ------------------------------------------------------------------ basics
    def _project(self) -> Project:
        p = self._projects.current
        if p is None or p.root is None:
            raise ProjectError("Open or create a project first.")
        return p

    def _analyzer(self) -> ReferenceVideoAnalyzer:
        return ReferenceVideoAnalyzer(self._ffmpeg, settings=self.analysis_settings)

    def folder(self, reference_id: str) -> Path:
        return self._project().paths.reference_dir(reference_id)

    def references(self) -> list[ReferenceAsset]:
        return sorted(self._project().reference_assets.values(), key=lambda a: a.imported_at)

    def reference(self, reference_id: str | None = None) -> ReferenceAsset | None:
        p = self._project()
        rid = reference_id or p.reference_settings.active_reference_id
        return p.reference_assets.get(rid) if rid else None

    def video_path(self, asset: ReferenceAsset) -> Path:
        return self._project().paths.resolve(asset.path)

    @property
    def settings(self) -> ReferenceSettings:
        return self._project().reference_settings

    def _publish(self, kind: str, reference_id: str = "") -> None:
        self._bus.publish(Topics.REFERENCE_UPDATED, reference_id=reference_id, kind=kind)

    # ------------------------------------------------------------------ import (copy into references/<id>/, hash, probe; never into project.assets)
    def import_reference(self, source: Path, *, link: bool = False, progress: Callable[[float, str], None] | None = None, cancel: threading.Event | None = None) -> ReferenceAsset:
        """Copy (or link) a local video into the project's isolated ``references/`` folder and register it. Raises ``ReferenceError`` with a readable message."""
        p = self._project()
        asset = self._prepare_import(p.root, Path(source), link, progress, cancel)  # type: ignore[arg-type]
        self._apply(RegisterReferenceCommand(p, asset))
        self._publish("imported", asset.reference_id)
        log_event(_log, "reference.imported", reference_id=asset.reference_id, name=asset.name, link=link)
        return asset

    def import_reference_async(self, source: Path, *, link: bool = False, on_done: Callable[[ReferenceAsset], None] | None = None, on_error: Callable[[Job], None] | None = None) -> Job:
        p = self._project()
        root, project_id = p.root, p.project_id

        def work(ctx) -> ReferenceAsset:
            return self._prepare_import(root, Path(source), link, lambda f, m: ctx.report(f * 100.0, m), ctx.job.cancel_event)  # type: ignore[arg-type]

        def done(job: Job) -> None:
            cur = self._projects.current
            if cur is None or cur.project_id != project_id:
                return
            self._apply(RegisterReferenceCommand(cur, job.result))
            self._publish("imported", job.result.reference_id)
            if on_done:
                on_done(job.result)

        return self._jobs.submit("reference.import", work, title=f"Importing reference “{Path(source).name}”", on_complete=done, on_error=on_error)

    def _prepare_import(self, root: Path, source: Path, link: bool, progress: Callable[[float, str], None] | None, cancel: threading.Event | None) -> ReferenceAsset:
        if not source.is_file():
            raise ReferenceError(f"The file “{source.name}” was not found.")
        if source.suffix.lower() not in REFERENCE_EXTENSIONS:
            raise ReferenceError(f"“{source.name}” is not a supported video file. Choose an MP4, MOV, MKV, WebM or AVI video.")
        rid = f"ref_{uuid.uuid4().hex[:10]}"
        folder = root / "references" / rid
        folder.mkdir(parents=True, exist_ok=True)
        try:
            analyzer = self._analyzer()
            try:
                meta = analyzer.read_metadata(source)  # reject corrupt / audio-only / unsupported files before copying gigabytes
            except ReferenceAnalysisError as exc:
                raise ReferenceError(exc.user_message, details=exc.details) from exc
            size = source.stat().st_size
            h = hashlib.sha1()
            if link:
                dest_rel = str(source.resolve())
                with open(source, "rb") as fh:
                    done = 0
                    while chunk := fh.read(1 << 20):
                        if cancel is not None and cancel.is_set():
                            raise JobCancelled()
                        h.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done / max(size, 1), "Reading the reference")
            else:
                dest = folder / f"reference_video{source.suffix.lower()}"
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(source, "rb") as src, open(tmp, "wb") as out:
                    done = 0
                    while chunk := src.read(1 << 20):
                        if cancel is not None and cancel.is_set():
                            raise JobCancelled()
                        out.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done / max(size, 1), "Copying the reference into the project")
                tmp.replace(dest)
                dest_rel = dest.relative_to(root).as_posix()
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return ReferenceAsset(rid, source.stem, dest_rel, "reference" if link else "copy", h.hexdigest(), size, now_iso(), _meta_dict(meta), "NONE")

    def remove_reference(self, reference_id: str) -> None:
        p = self._project()
        if reference_id not in p.reference_assets:
            raise ReferenceError("That reference video is not part of this project.")
        job = self._jobs_by_ref.get(reference_id)
        if job is not None and not job.status.is_terminal:
            self._jobs.cancel(job.id)
        self._apply(RemoveReferenceCommand(p, reference_id))
        self._cache.pop(reference_id, None)
        shutil.rmtree(p.paths.reference_dir(reference_id), ignore_errors=True)  # the copied video, the analysis and the thumbnails
        self._publish("removed", reference_id)

    def set_active(self, reference_id: str) -> None:
        p = self._project()
        if reference_id not in p.reference_assets:
            raise ReferenceError("That reference video is not part of this project.")
        new = replace(p.reference_settings, active_reference_id=reference_id)
        analysis = self.cached_analysis(reference_id)
        self._execute(SetReferenceSettingsCommand(p, new, analysis.profile if analysis else None, switch_profile=True))
        self._publish("active", reference_id)

    # ------------------------------------------------------------------ the analysis cache (references/<id>/analysis.json)
    def cache_file(self, reference_id: str) -> Path:
        return self.folder(reference_id) / ANALYSIS_FILE

    def cached_analysis(self, reference_id: str, *, check_valid: bool = True) -> ReferenceAnalysis | None:
        """The stored analysis when it is still valid: same video content, same analysis version and the same detection settings."""
        p = self._project()
        asset = p.reference_assets.get(reference_id)
        if asset is None:
            return None
        with self._lock:
            hit = self._cache.get(reference_id)
        if hit is None:
            f = self.cache_file(reference_id)
            if not f.is_file():
                return None
            try:
                hit = ReferenceAnalysis.from_dict(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, KeyError):
                _log.warning("reference analysis cache unreadable: %s", f, exc_info=True)
                return None
            with self._lock:
                self._cache[reference_id] = hit
        if check_valid and not self._is_valid(asset, hit):
            return None
        return hit

    def _is_valid(self, asset: ReferenceAsset, a: ReferenceAnalysis) -> bool:
        return a.reference_hash == asset.content_hash and a.analysis_version == ANALYSIS_VERSION and a.settings_hash == self.analysis_settings.settings_hash()

    def is_stale(self, reference_id: str) -> bool:
        """True when an analysis exists but no longer matches the video / version / settings (it will be re-run on ``analyze``)."""
        p = self._project()
        asset = p.reference_assets.get(reference_id)
        return bool(asset and asset.analysis_status in ("COMPLETED", "PARTIAL") and self.cached_analysis(reference_id) is None)

    def analysis(self, reference_id: str | None = None) -> ReferenceAnalysis | None:
        asset = self.reference(reference_id)
        return self.cached_analysis(asset.reference_id) if asset else None

    def profile(self, reference_id: str | None = None) -> ReferenceStyleProfile | None:
        p = self._project()
        if reference_id is None or reference_id == p.reference_settings.active_reference_id:
            if p.reference_style_profile is not None:
                return p.reference_style_profile
        a = self.analysis(reference_id)
        return a.profile if a else None

    # ------------------------------------------------------------------ analysis job
    def analyze(self, reference_id: str | None = None, *, force: bool = False, on_done: Callable[[ReferenceAnalysis], None] | None = None,
                on_error: Callable[[Job], None] | None = None) -> Job | None:
        """Analyse a reference in the background (never touches the timeline). Returns ``None`` when a valid cached analysis was reused (``on_done`` is still called)."""
        p = self._project()
        asset = self.reference(reference_id)
        if asset is None:
            raise ReferenceError("Import a reference video first.")
        rid = asset.reference_id
        running = self._jobs_by_ref.get(rid)
        if running is not None and not running.status.is_terminal:
            return running
        path = self.video_path(asset)
        if not path.is_file():
            raise ReferenceError(f"The reference video “{asset.name}” is missing. Re-import it to analyse again.")
        if asset.link_mode == "reference":  # a linked file may have changed on disk: the hash decides
            current = file_hash(path)
            if current != asset.content_hash:
                asset = replace(asset, content_hash=current)
                self._apply(StoreAnalysisCommand(p, rid, {"content_hash": current, "size_bytes": path.stat().st_size}, None, None))
                self._cache.pop(rid, None)
        if not force:
            hit = self.cached_analysis(rid)
            if hit is not None:
                self._adopt(p, asset, hit, reused=True)
                if on_done:
                    on_done(hit)
                return None
        project_id = p.project_id
        content_hash = asset.content_hash
        folder = self.folder(rid)
        analyzer = self._analyzer()
        logs: list[str] = []
        self._apply(StoreAnalysisCommand(p, rid, {"analysis_status": "RUNNING", "error": ""}, None, None))
        self._publish("status", rid)

        def work(ctx) -> ReferenceAnalysis:
            current = {"stage": ""}

            def report(stage: str, fraction: float, message: str) -> None:
                changed = stage != current["stage"]  # a new stage is always published (the 20 Hz throttle must not swallow a short one)
                current["stage"] = stage
                ctx.report(fraction * 100.0, f"{stage}: {message}" if message and message != stage else stage, force=changed)

            def say(line: str) -> None:
                logs.append(line)

            try:
                analysis = analyzer.analyze(path, reference_id=rid, content_hash=content_hash, progress=report, log_fn=say, cancel=ctx.job.cancel_event)
            except AnalysisCancelled as exc:
                raise JobCancelled() from exc
            except ReferenceAnalysisError as exc:
                logs.append(f"FAILED: {exc.user_message}")
                self._write_log(folder, logs)
                raise
            self._write_analysis(folder, analysis, logs)
            self._make_thumbnails(path, folder, analysis.features.metadata.duration)
            return analysis

        def done(job: Job) -> None:
            cur = self._projects.current
            a: ReferenceAnalysis = job.result
            if cur is None or cur.project_id != project_id:
                return
            with self._lock:
                self._cache[rid] = a
            self._adopt(cur, cur.reference_assets.get(rid), a, reused=False)
            if on_done:
                on_done(a)

        def failed(job: Job) -> None:
            cur = self._projects.current
            if cur is not None and cur.project_id == project_id and rid in cur.reference_assets:
                self._apply(StoreAnalysisCommand(cur, rid, {"analysis_status": "FAILED", "error": job.error or "The analysis failed."}, None, None))
                self._publish("status", rid)
            if on_error:
                on_error(job)

        def cancelled(job: Job) -> None:
            cur = self._projects.current
            if cur is not None and cur.project_id == project_id and rid in cur.reference_assets:
                self._apply(StoreAnalysisCommand(cur, rid, {"analysis_status": "CANCELED", "error": ""}, None, None))
                self._publish("status", rid)

        job = self._jobs.submit("reference.analyze", work, title=f"Analyzing reference “{asset.name}”", on_complete=done, on_error=failed, on_cancel=cancelled)
        self._jobs_by_ref[rid] = job
        return job

    def reanalyze(self, reference_id: str | None = None, **kw) -> Job | None:
        return self.analyze(reference_id, force=True, **kw)

    def cancel_analysis(self, reference_id: str | None = None) -> bool:
        asset = self.reference(reference_id)
        job = self._jobs_by_ref.get(asset.reference_id) if asset else None
        return bool(job and not job.status.is_terminal and self._jobs.cancel(job.id))

    def retry_analysis(self, reference_id: str | None = None) -> Job:
        asset = self.reference(reference_id)
        if asset is None:
            raise ReferenceError("Import a reference video first.")
        job = self._jobs_by_ref.get(asset.reference_id)
        if job is not None and job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
            new = self._jobs.retry(job.id)
            self._apply(StoreAnalysisCommand(self._project(), asset.reference_id, {"analysis_status": "RUNNING", "error": ""}, None, None))
            return new
        return self.analyze(asset.reference_id, force=True)  # type: ignore[return-value]

    def job_for(self, reference_id: str) -> Job | None:
        return self._jobs_by_ref.get(reference_id)

    def read_log(self, reference_id: str | None = None, tail: int = MAX_LOG_LINES) -> str:
        asset = self.reference(reference_id)
        if asset is None:
            return ""
        f = self.folder(asset.reference_id) / LOG_FILE
        if not f.is_file():
            return ""
        return "\n".join(f.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:])

    def _adopt(self, p: Project, asset: ReferenceAsset | None, a: ReferenceAnalysis, *, reused: bool) -> None:
        if asset is None:
            return
        fields = {"analysis_status": a.status, "analyzed_hash": a.reference_hash, "analyzed_version": a.analysis_version, "analyzed_settings_hash": a.settings_hash,
                  "analyzed_at": a.created_at, "error": "", "metadata": _meta_dict(a.features.metadata)}
        thumbs = sorted(f"references/{asset.reference_id}/thumbnails/{t.name}" for t in (self.folder(asset.reference_id) / "thumbnails").glob("*.jpg")) if (self.folder(asset.reference_id) / "thumbnails").is_dir() else []
        fields["thumbnails"] = thumbs
        self._apply(StoreAnalysisCommand(p, asset.reference_id, fields, a.record(), a.profile))
        self._publish("analysis", asset.reference_id)
        log_event(_log, "reference.analyzed", reference_id=asset.reference_id, status=a.status, reused=reused)

    @staticmethod
    def _write_log(folder: Path, lines: list[str]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_write_text(folder / LOG_FILE, "\n".join(lines[-MAX_LOG_LINES:]) + "\n")

    def _write_analysis(self, folder: Path, a: ReferenceAnalysis, logs: list[str]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_write_text(folder / ANALYSIS_FILE, json.dumps(a.to_dict(), ensure_ascii=False))
        self._write_log(folder, a.log or logs)

    def _make_thumbnails(self, video: Path, folder: Path, duration: float) -> None:
        """A few small frames for the user's own review of their own reference (kept only in references/<id>/thumbnails)."""
        import subprocess  # noqa: PLC0415

        out = folder / "thumbnails"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        if duration <= 0:
            return
        for i in range(THUMBNAILS):
            t = duration * (i + 0.5) / THUMBNAILS
            try:
                subprocess.run([self._ffmpeg.ffmpeg(), "-v", "error", "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=320:-2", str(out / f"thumb_{i + 1:02d}.jpg")],
                               capture_output=True, timeout=30, check=False)
            except (OSError, subprocess.SubprocessError):
                break

    # ------------------------------------------------------------------ settings the user can change without applying anything
    def update_settings(self, **changes) -> ReferenceSettings:
        p = self._project()
        new = replace(p.reference_settings, **changes)
        self._validate_settings(new)
        self._execute(SetReferenceSettingsCommand(p, new))
        self._bus.publish(Topics.REFERENCE_STYLE_CHANGED, action="settings")
        return new

    @staticmethod
    def _validate_settings(s: ReferenceSettings) -> None:
        if s.application_mode not in MODES:
            raise ReferenceError(f"Unknown style mode “{s.application_mode}”.")
        if round(s.style_strength, 2) not in STRENGTHS:
            raise ReferenceError("Style strength must be 25%, 50%, 75% or 100%.")
        bad = [d for d in s.custom_dimensions if d not in DIMENSIONS]
        if bad:
            raise ReferenceError(f"Unknown style characteristics: {', '.join(bad)}.")
        for d, v in s.adjustments.items():
            if d not in DIMENSIONS or not (0.0 <= float(v) <= 100.0):
                raise ReferenceError(f"The target for “{d}” must be between 0 and 100.")

    # ------------------------------------------------------------------ the user's own project, on the same scale
    def project_profile(self) -> ReferenceStyleProfile:
        from app.reference.project_metrics import project_style_profile  # noqa: PLC0415

        return project_style_profile(self._project())

    def comparison(self, reference_id: str | None = None) -> tuple[list[ComparisonRow], StyleSimilarityScore | None]:
        """REFERENCE vs CURRENT PROJECT rows plus the editing-feature similarity (not a copyright / content measure)."""
        from app.reference.style_adapter import similarity_between  # noqa: PLC0415

        prof = self.profile(reference_id)
        if prof is None:
            return [], None
        mine = self.project_profile()
        return compare(prof, mine), similarity_between(prof, mine)

    # ------------------------------------------------------------------ planning and applying (never modifies the timeline)
    def plan(self, reference_id: str | None = None, *, adjustments: dict[str, float] | None = None, mode: str | None = None, strength: float | None = None,
             custom_dimensions: list[str] | None = None, preserve_user_edits: bool | None = None) -> StylePlan:
        """Compute the abstract overrides and the simulated result for review. Changes nothing."""
        from app.reference.project_metrics import adaptation_baseline, project_content, project_style_profile  # noqa: PLC0415
        from app.reference.style_adapter import ReferenceStyleAdapter, simulate  # noqa: PLC0415

        p = self._project()
        asset = self.reference(reference_id)
        prof = self.profile(asset.reference_id if asset else reference_id)
        if asset is None or prof is None:
            raise ReferenceError("Analyze a reference video first.")
        s = replace(p.reference_settings, active_reference_id=asset.reference_id)
        if mode is not None:
            s.application_mode = mode
        if strength is not None:
            s.style_strength = float(strength)
        if custom_dimensions is not None:
            s.custom_dimensions = list(custom_dimensions)
        if preserve_user_edits is not None:
            s.preserve_user_edits = bool(preserve_user_edits)
        if adjustments is not None:
            s.adjustments = {k: float(v) for k, v in adjustments.items()}
        self._validate_settings(s)
        adj = StyleAdjustments(dict(s.adjustments))
        content, baseline = project_content(p), adaptation_baseline(p)
        result = ReferenceStyleAdapter().adapt(prof, adj, s, content, baseline)
        mine = project_style_profile(p)
        proj = simulate(prof, adj, s, content, baseline, mine)
        rows, sim = self.comparison(asset.reference_id)
        return StylePlan(asset.reference_id, result.overrides, result.effective_targets, result.skipped, result.notes, result.warnings, s, proj, rows, sim)

    def apply_style(self, plan: StylePlan | None = None, **plan_kwargs) -> StyleApplication:
        """Apply the (reviewed) plan: checkpoint, then one undoable command that records the application. The timeline is not touched; nothing is re-rendered."""
        p = self._project()
        plan = plan or self.plan(**plan_kwargs)
        if plan.overrides.is_empty:
            reasons = "; ".join(f"{k}: {v}" for k, v in plan.skipped.items()) or "no characteristic could be applied"
            raise ReferenceError(f"There is nothing to apply ({reasons}).")
        try:
            checkpoint = self._checkpoint(p, "before_reference_style")
        except Exception:  # noqa: BLE001  (a failed safety copy must not block a reversible, abstract change; the undo history still holds the old values)
            _log.warning("checkpoint before applying a style failed", exc_info=True)
            checkpoint = ""
        settings = deepcopy(plan.settings)
        settings.enabled = True
        prof = self.profile(plan.reference_id)
        settings.analysis_version = prof.analysis_version if prof else ANALYSIS_VERSION
        app = StyleApplication(f"app_{uuid.uuid4().hex[:10]}", plan.reference_id, now_iso(), settings.application_mode, settings.style_strength, settings.dimensions(),
                               dict(settings.adjustments), prof.signature() if prof else "", None, None, None, checkpoint, list(plan.notes))
        self._execute(ApplyStyleCommand(p, settings, plan.overrides, app))
        self._bus.publish(Topics.REFERENCE_STYLE_CHANGED, action="applied")
        log_event(_log, "reference.style_applied", reference_id=plan.reference_id, mode=settings.application_mode, strength=settings.style_strength)
        return p.style_application_history[-1]

    def clear_style(self) -> None:
        """Stop using the reference style (undoable). Existing timeline content is untouched; the next generation uses the user's own settings again."""
        p = self._project()
        if not p.reference_settings.enabled and p.reference_style_overrides.is_empty:
            return
        app = StyleApplication(f"app_{uuid.uuid4().hex[:10]}", p.reference_settings.active_reference_id, now_iso(), "CLEARED", 0.0, [], {}, "", None, None, None, "", ["Reference style removed"])
        self._execute(ClearStyleCommand(p, app))
        self._bus.publish(Topics.REFERENCE_STYLE_CHANGED, action="cleared")

    def revert_last_application(self) -> bool:
        """Restore the values that were in force before the last application (works even after the undo stack was cleared)."""
        p = self._project()
        if not p.style_application_history:
            return False
        last = p.style_application_history[-1]
        if last.before is None:
            return False
        settings = deepcopy(last.before_settings) if last.before_settings is not None else deepcopy(p.reference_settings)
        app = StyleApplication(f"app_{uuid.uuid4().hex[:10]}", last.reference_id, now_iso(), "REVERTED", 0.0, [], {}, "", None, None, None, "", [f"Reverted application {last.application_id}"])
        self._execute(ApplyStyleCommand(p, settings, last.before, app, "Revert reference style"))
        self._bus.publish(Topics.REFERENCE_STYLE_CHANGED, action="reverted")
        return True

    def history(self) -> list[StyleApplication]:
        return list(self._project().style_application_history)

    # ------------------------------------------------------------------ the optional free-text style request (OriginalityGuard)
    def guard_request(self, text: str):
        """Turn a free-text style wish into an abstract instruction. Requests to copy exact text, order, graphics, logos, composition or branding are flagged and converted."""
        from app.reference.originality import OriginalityGuard  # noqa: PLC0415

        res = OriginalityGuard().convert_request(text)
        p = self._project()
        new = replace(p.reference_settings, style_request=res.abstract_instruction if res.flagged else text.strip())
        self._execute(SetReferenceSettingsCommand(p, new))
        return res


def _meta_dict(meta) -> dict:
    from app.core.serialization import to_plain  # noqa: PLC0415

    return to_plain(meta)

