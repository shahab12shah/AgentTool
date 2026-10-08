"""QCService: the only thing the UI (and the export path) talks to for quality control.

    UI -> QCService -> QCEngine -> checkers -> QCRun (issues + scores) -> installed in the project as *findings* (never as edits)
                    -> QCFixEngine -> undoable commands (fixes are explicit, previewable, reversible)

Runs are background jobs (progress per stage, cancel, retry, partial results). The job works on a detached snapshot taken on the caller's thread; results are
installed back on the dispatcher thread by a small, non-undoable command that only touches the ``qc_*`` project sections.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from app.core.commands import Command
from app.core.events import EventBus, Topics
from app.core.exceptions import ProjectError
from app.qc.errors import QCError
from app.core.constants import APP_VERSION
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.phase8_commands import IgnoreIssuesCommand, SetQCSettingsCommand, StoreQCRunCommand, StoreRenderQCCommand, UnignoreCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.qc.context import QCContext, sha
from app.qc.issue_model import (
    CATEGORY_GROUP, CheckerStatus, IgnoreRecord, IssueStatus, QCIssue, QCRun, QCScores, now_iso, sorted_issues,
)
from app.qc.qc_engine import CHECKER_CATEGORIES, EngineResult, PreviousState, QCEngine, refresh_fix_flags
from app.qc.qc_history import RunComparison, compare_entries, entry_for
from app.qc.report_generator import build_report
from app.qc.scoring import ExportDecision, decide_export
from app.qc.settings import QCSettings
from app.qc.severity import Severity
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.probe import MediaProbeService

_log = get_logger(__name__)


@dataclass
class QCProgress:
    state: str = "IDLE"  # IDLE | RUNNING | CANCELING
    fraction: float = 0.0
    message: str = ""
    job_id: str = ""
    checkers: dict[str, CheckerStatus] = field(default_factory=dict)


@dataclass
class GateResult:
    """The export gate as the export page needs it."""

    decision: ExportDecision
    run_current: bool  # the latest QC run still matches the project as it is now
    needs_run: bool  # QC has never run, or the project changed since: run it before exporting (the UI does this automatically when ``run_before_export`` is on)
    message: str = ""

    @property
    def blocked(self) -> bool:
        return self.run_current and self.decision.blocked


class QCService:
    def __init__(self, projects: ProjectManager, jobs: JobManager, bus: EventBus, apply_command: Callable[[Command], None], execute_command: Callable[[Command], object],
                 settings_getter: Callable, checkpoint: Callable[[Project, str], str], engine: QCEngine | None = None) -> None:
        self._projects, self._jobs, self._bus, self._apply, self._execute = projects, jobs, bus, apply_command, execute_command
        self._settings, self._checkpoint = settings_getter, checkpoint
        self.ffmpeg = FFmpegService(lambda: self._settings().ffmpeg_path, lambda: self._settings().ffprobe_path)
        self.probe = MediaProbeService(self.ffmpeg)
        self._engine = engine
        self.fixes = None  # QCFixEngine, installed by the Workspace (it needs the editing / presentation / render services)
        self.progress = QCProgress()
        self._job: Job | None = None
        self._lock = threading.RLock()
        self._ai_provider = None  # AIEditorialQCProvider override (tests / future providers)
        self._last_pub = (0.0, "")
        bus.subscribe(Topics.PROJECT_CHANGED, self._on_project_changed)
        bus.subscribe(Topics.PROJECT_OPENED, lambda *_a: self._reset())
        bus.subscribe(Topics.PROJECT_CLOSED, lambda *_a: self._reset())

    # ------------------------------------------------------------------ basics
    def _project(self) -> Project:
        p = self._projects.current
        if p is None or p.root is None:
            raise ProjectError("Open or create a project first.")
        return p

    @property
    def engine(self) -> QCEngine:
        """Built on first use (importing sixteen checkers is not free and nothing needs them until QC runs)."""
        if self._engine is None:
            self._engine = QCEngine()
        return self._engine

    @engine.setter
    def engine(self, value: QCEngine) -> None:
        self._engine = value

    @property
    def settings(self) -> QCSettings:
        return self._project().qc_settings

    def set_ai_provider(self, provider) -> None:
        """Use a specific AIEditorialQCProvider (None = the one named in the QC settings)."""
        self._ai_provider = provider

    def ai_provider(self):
        return self._ai_provider

    def _publish(self, kind: str, **extra: Any) -> None:
        self._bus.publish(Topics.QC_UPDATED, kind=kind, **extra)

    def _reset(self) -> None:
        self.progress, self._job = QCProgress(), None

    def _on_project_changed(self, _topic: str, payload: dict) -> None:
        """Ctrl+Z / Ctrl+Shift+Z of an ignore, a fix or a settings change: scores follow at once (the undo stack publishes no QC event of its own)."""
        cmd = payload.get("command")
        if payload.get("action") not in ("undo", "redo") or cmd is None or getattr(cmd, "scope", "") not in ("qc", "timeline", "editing"):
            return
        p = self._projects.current
        if p is None or p.qc_scores is None or not p.qc_runs:
            return
        try:
            self._refresh_derived(p)
        except Exception:  # noqa: BLE001  (a bookkeeping refresh must never break undo)
            _log.warning("QC recount after undo failed", exc_info=True)
            return
        self._publish("issue_changed")

    @property
    def running(self) -> bool:
        return self._job is not None and not self._job.status.is_terminal

    # ------------------------------------------------------------------ settings (undoable)
    def update_settings(self, **changes: Any) -> QCSettings:
        p = self._project()
        new = deepcopy(p.qc_settings)
        for k, v in changes.items():
            if "." in k:
                new.set_path(k, v)
            elif hasattr(new, k):
                setattr(new, k, v)
            else:
                raise QCError(f"Unknown QC setting “{k}”.")
        self._validate(new)
        self._execute(SetQCSettingsCommand(p, new))
        self._refresh_derived(p)
        self._publish("settings")
        return new

    @staticmethod
    def _validate(s: QCSettings) -> None:
        from app.qc.severity import BlockLevel  # noqa: PLC0415

        if s.block_level not in {b.value for b in BlockLevel}:
            raise QCError("Block export on: choose Critical, Critical + Errors or Critical + Errors + Warnings.")
        if s.marker_mode not in ("all", "critical_only", "hidden"):
            raise QCError("Timeline markers: choose all, critical only or hidden.")
        if not (s.sync.minor_ms <= s.sync.moderate_ms <= s.sync.major_ms):
            raise QCError("Sync tolerances must satisfy minor ≤ moderate ≤ major.")
        for k, v in s.fix_permissions.items():
            if v not in ("auto", "confirm", "never"):
                raise QCError(f"Auto-fix permission for “{k}” must be auto, confirm or never.")
        if s.visual.error_below > s.visual.warning_below:
            raise QCError("Visual accuracy: the error threshold must not be above the warning threshold.")

    # ------------------------------------------------------------------ running
    def _previous_state(self, p: Project) -> PreviousState:
        return PreviousState(deepcopy(p.qc_issues), deepcopy(p.qc_cache))

    def _next_number(self, p: Project) -> int:
        return max([int(r.get("number", 0)) for r in p.qc_runs] + [0]) + 1

    def current_content_hash(self, p: Project | None = None) -> str:
        """Fingerprint of everything QC analyses, for the *live* project (no copy: only hashes are computed)."""
        p = p or self._project()
        ctx = QCContext.build(p, detach=False)
        return sha([ctx.domain_hash(d) for d in ("timeline", "scenes", "transcript", "assets", "audio", "captions", "visual", "reference", "render")], p.qc_settings.analysis_version())

    def run_full_qc(self, project_id: str | None = None, *, on_done: Callable[[QCRun], None] | None = None, on_error: Callable[[Job], None] | None = None, trigger: str = "manual",
                    force: bool = False, rendered_file: Path | None = None) -> Job:
        """The whole pipeline in the background. Unchanged analysis is reused (``force`` re-runs everything)."""
        p = self._project()
        if project_id and project_id != p.project_id:
            raise QCError("That project is not open.")
        selected = self.engine.select(p.qc_settings)
        return self._start(p, selected, trigger=trigger, scope={"categories": [], "scene_ids": []}, force=force, on_done=on_done, on_error=on_error, rendered_file=rendered_file)

    def run_scene_qc(self, scene_id: str, *, on_done: Callable[[QCRun], None] | None = None, on_error: Callable[[Job], None] | None = None) -> Job:
        """Re-analyse one scene (the scene-local checkers only); everything else keeps its previous findings."""
        p = self._project()
        if p.qc_settings and not any(s.id == scene_id for s in p.scenes):
            raise QCError("That scene does not exist.")
        selected = self.engine.select(p.qc_settings, scenes_only=True)
        return self._start(p, selected, trigger="scene", scope={"categories": [], "scene_ids": [scene_id]}, scene_ids=[scene_id], force=False, on_done=on_done, on_error=on_error)

    def run_category_qc(self, category: str, *, on_done: Callable[[QCRun], None] | None = None, on_error: Callable[[Job], None] | None = None, force: bool = True) -> Job:
        """Run the checkers of one category (or one checker id) again."""
        p = self._project()
        selected = self.engine.select(p.qc_settings, categories=[category])
        if selected <= {"preflight"}:
            raise QCError(f"There is no QC check for “{category}”.")
        return self._start(p, selected, trigger="category", scope={"categories": [category], "scene_ids": []}, force=force, on_done=on_done, on_error=on_error)

    def retry_failed_check(self, qc_run_id: str | None = None, category: str | None = None, *, on_done: Callable[[QCRun], None] | None = None,
                           on_error: Callable[[Job], None] | None = None) -> Job:
        """Run only the checks that failed (or the named one) and merge them into the same run, without restarting the others."""
        p = self._project()
        rec = next((r for r in reversed(p.qc_runs) if qc_run_id in (None, r.get("run_id"))), None)
        if rec is None:
            raise QCError("That QC run does not exist.")
        if rec["run_id"] != p.qc_runs[-1]["run_id"]:
            raise QCError("Only the latest QC run can be retried; run QC again instead.")
        failed = [k for k, v in (rec.get("checkers") or {}).items() if v.get("state") in ("FAILED", "CANCELED")]
        if category:
            failed = [c for c in self.engine.select(p.qc_settings, categories=[category])] if category not in failed else [category]
        failed = [c for c in failed if c != "preflight"] or failed
        if not failed:
            raise QCError("Nothing failed in that run.")
        return self._start(p, set(failed), trigger="retry", scope={"categories": failed, "scene_ids": []}, force=True, replace_run=rec["run_id"], on_done=on_done, on_error=on_error)

    def _start(self, p: Project, selected: set[str], *, trigger: str, scope: dict[str, Any], force: bool, scene_ids: list[str] | None = None, replace_run: str = "",
               on_done: Callable[[QCRun], None] | None, on_error: Callable[[Job], None] | None, rendered_file: Path | None = None) -> Job:
        if self.running:
            raise QCError("Quality control is already running. Wait for it to finish or cancel it.")
        previous = self._previous_state(p)
        ignores = deepcopy(p.qc_ignored_issues)
        number = self._next_number(p) if not replace_run else next((int(r["number"]) for r in p.qc_runs if r["run_id"] == replace_run), self._next_number(p))
        run_id = replace_run or None
        ctx = QCContext.build(p, ffmpeg=self.ffmpeg, probe=self.probe, scene_filter=scene_ids, rendered_file=rendered_file)  # detached snapshot (UI thread)
        ctx.ai_provider = self._ai_provider
        content_hash = self.current_content_hash(p)
        project_id = p.project_id
        holder: dict[str, EngineResult] = {}
        version = f"{APP_VERSION}/schema {p.schema_version}"
        self.progress = QCProgress("RUNNING", 0.0, "Starting…", "", {})
        self._publish("run_started", trigger=trigger)

        def work(jc) -> EngineResult:
            ctx.cancel = jc.job.cancel_event

            def on_progress(frac: float, msg: str, statuses: dict[str, CheckerStatus]) -> None:
                self.progress = QCProgress("CANCELING" if jc.job.cancel_requested else "RUNNING", frac, msg, jc.job.id, statuses)
                jc.report(frac * 100.0, msg)
                last_t, last_msg = self._last_pub
                now = time.monotonic()
                if msg != last_msg or frac >= 1.0 or now - last_t >= 0.15:  # coalesced: the UI redraws on every event
                    self._last_pub = (now, msg)
                    self._publish("progress", fraction=frac, message=msg)

            res = self.engine.run(ctx, previous=previous, ignores=ignores, selected=selected, trigger=trigger, number=number, scope=scope, progress=on_progress, use_cache=not force,
                                  run_id=run_id, project_version=version)
            res.run.content_hash = content_hash
            holder["r"] = res
            return res

        def install(res: EngineResult, canceled: bool = False) -> None:
            cur = self._projects.current
            if cur is None or cur.project_id != project_id:
                return
            self._install(cur, res, replace_run)
            self.progress = QCProgress("IDLE", 1.0, "Canceled" if canceled else "Done", "", res.run.checkers)
            self._publish("run_canceled" if canceled else "run_finished", run_id=res.run.run_id)
            log_event(_log, "qc.run", run_id=res.run.run_id, state=res.run.state, overall=res.run.scores.overall, trigger=trigger)

        def done(job: Job) -> None:
            res: EngineResult = job.result
            install(res)
            if on_done:
                on_done(res.run)

        def cancelled(job: Job) -> None:
            res = holder.get("r")
            if res is not None:  # partial results: finished checkers are kept, the rest keeps its previous findings
                install(res, canceled=True)
            else:
                self.progress = QCProgress()
                self._publish("run_canceled")

        def failed(job: Job) -> None:
            self.progress = QCProgress("IDLE", 0.0, f"QC failed: {job.error}", "", {})
            self._publish("run_failed", error=job.error)
            if on_error:
                on_error(job)

        title = {"scene": "Quality control — scene", "category": "Quality control — category", "retry": "Quality control — retry"}.get(trigger, "AI quality control")
        job = self._jobs.submit("qc.run", work, title=title, on_complete=done, on_error=failed, on_cancel=cancelled)
        self._job = job
        self.progress.job_id = job.id
        return job

    def cancel(self) -> bool:
        job = self._job
        if job is None or job.status.is_terminal:
            return False
        self.progress = replace(self.progress, state="CANCELING")
        return self._jobs.cancel(job.id)

    # ------------------------------------------------------------------ installing results
    def _install(self, p: Project, res: EngineResult, replace_run: str = "") -> None:
        run = res.run
        ignored = [r.fingerprint or r.code for r in p.qc_ignored_issues]
        run.fixes = [f.fix_id for f in p.qc_fixes if not f.reverted][-50:]
        hist = run.history_entry(ignored, run.fixes)
        cmd = StoreQCRunCommand(p, run.issues, run.scores, run.record(), hist, res.cache, replace_run_id=replace_run)
        self._apply(cmd)

    # ------------------------------------------------------------------ results
    def latest_run(self) -> dict[str, Any] | None:
        p = self._project()
        return deepcopy(p.qc_runs[-1]) if p.qc_runs else None

    def get_results(self, qc_run_id: str | None = None) -> dict[str, Any]:
        """The run record + issues + scores. The latest run has full issue details; older runs have the compact archive (fingerprints, severities, titles)."""
        p = self._project()
        if qc_run_id is None or (p.qc_runs and qc_run_id == p.qc_runs[-1]["run_id"]):
            rec = p.qc_runs[-1] if p.qc_runs else None
            return {"run": deepcopy(rec), "issues": sorted_issues(deepcopy(p.qc_issues)), "scores": deepcopy(p.qc_scores), "archived": False}
        entry = entry_for(p.qc_history, qc_run_id)
        rec = next((r for r in p.qc_runs if r.get("run_id") == qc_run_id), None)
        if entry is None and rec is None:
            raise QCError("That QC run does not exist.")
        return {"run": deepcopy(rec), "issues": deepcopy((entry or {}).get("issues", [])), "scores": None, "archived": True, "history": deepcopy(entry)}

    def scores(self) -> QCScores | None:
        return deepcopy(self._project().qc_scores)

    def issues(self, *, severity: str | None = None, category: str | None = None, scene_id: str | None = None, include_ignored: bool = False, include_fixed: bool = False) -> list[QCIssue]:
        out = []
        for i in self._project().qc_issues:
            if not include_ignored and i.ignored_by_user:
                continue
            if not include_fixed and i.status is IssueStatus.FIXED:
                continue
            if severity and i.severity.value != severity.upper():
                continue
            if category and i.category.value != category.upper():
                continue
            if scene_id and i.scene_id != scene_id:
                continue
            out.append(i)
        return sorted_issues(deepcopy(out))

    def issue(self, issue_id: str) -> QCIssue:
        for i in self._project().qc_issues:
            if i.issue_id == issue_id:
                return deepcopy(i)
        raise QCError("That issue no longer exists (run QC again).")

    def issues_at(self, t: float, window: float = 0.5) -> list[QCIssue]:
        return [i for i in self.issues() if i.start_time is not None and i.start_time - window <= t <= (i.end_time if i.end_time is not None else i.start_time) + window]

    def markers(self, mode: str | None = None) -> list[dict[str, Any]]:
        """Timeline markers derived from the current issues. ``mode``: all | critical_only | hidden (default: the project's marker setting)."""
        p = self._project()
        mode = mode or p.qc_settings.marker_mode
        if mode == "hidden":
            return []
        out = []
        for i in sorted_issues(p.qc_issues):
            if not i.active or i.start_time is None or i.severity is Severity.INFO:
                continue
            if mode == "critical_only" and i.severity is not Severity.CRITICAL:
                continue
            out.append({"issue_id": i.issue_id, "time": i.start_time, "end": i.end_time, "severity": i.severity.value, "title": i.title, "scene_id": i.scene_id,
                        "clip_id": i.timeline_item_id, "track_id": i.track_id, "confidence": i.confidence})
        return sorted(out, key=lambda m: m["time"])

    def set_marker_mode(self, mode: str) -> None:
        """A view preference (stored with the project, not an undoable edit)."""
        p = self._project()
        new = deepcopy(p.qc_settings)
        new.marker_mode = mode
        self._validate(new)
        self._apply(SetQCSettingsCommand(p, new))
        self._publish("settings")

    # ------------------------------------------------------------------ the user's decisions
    def ignore_issue(self, issue_id: str, reason: str = "", scope: str = "issue") -> IgnoreRecord:
        """Keep something QC flagged. ``scope='type'`` ignores every issue of that kind (optionally only in its scene). Undoable."""
        p = self._project()
        i = self.issue(issue_id)
        if scope not in ("issue", "type", "type_scene"):
            raise QCError("Ignore scope must be issue, type or type_scene.")
        rec = IgnoreRecord(f"ign_{sha(issue_id, now_iso())}", "type" if scope.startswith("type") else "issue", i.fingerprint, i.code, i.scene_id if scope == "type_scene" else None, reason, now_iso(), i.title)
        self._execute(IgnoreIssuesCommand(p, rec))
        self._recount(p)
        self._publish("issue_changed", issue_id=issue_id)
        return rec

    def ignore_type(self, issue_id: str, reason: str = "") -> IgnoreRecord:
        return self.ignore_issue(issue_id, reason, "type")

    def unignore(self, ignore_id: str) -> None:
        p = self._project()
        self._execute(UnignoreCommand(p, ignore_id))
        self._recount(p)
        self._publish("issue_changed", ignore_id=ignore_id)

    def ignored(self) -> list[IgnoreRecord]:
        return deepcopy(self._project().qc_ignored_issues)

    def _refresh_derived(self, p: Project) -> None:
        """Everything that follows from the settings without a new analysis: fix flags (permissions) and the scores (weights, block level)."""
        for i in p.qc_issues:
            refresh_fix_flags(i, p.qc_settings)
        self._recount(p)

    def _recount(self, p: Project) -> None:
        """Scores follow the user's decisions at once (ignored / fixed issues stop counting) without re-analysing anything."""
        from app.qc.scoring import compute_scores  # noqa: PLC0415

        if p.qc_scores is None:
            return
        last = p.qc_runs[-1] if p.qc_runs else {}
        failed_ids = list(last.get("failed", []))
        failed_groups = sorted({CATEGORY_GROUP[c] for cid in failed_ids for c in CHECKER_CATEGORIES.get(cid, ())} | set(p.qc_scores.unavailable))
        p.qc_scores = compute_scores(p.qc_issues, p.qc_settings, self._duration(p), len(p.scenes), failed_groups, failed_ids)

    @staticmethod
    def _duration(p: Project) -> float:
        if p.voice_over.duration:
            return float(p.voice_over.duration)
        return float(max([s.end for s in p.scenes] + [p.timeline.duration, 0.0]))

    # ------------------------------------------------------------------ fixes (delegated to the QCFixEngine)
    def _fix_engine(self):
        if self.fixes is None:
            raise QCError("Fixes are not available in this session.")
        return self.fixes

    def preview_fix(self, issue_id: str):
        return self._fix_engine().preview_fix(issue_id)

    def apply_fix(self, issue_id: str, *, confirmed: bool = False):
        rec = self._fix_engine().apply_fix(issue_id, confirmed=confirmed)
        self._recount(self._project())
        self._publish("fix_applied", issue_id=issue_id)
        return rec

    def fix_similar(self, issue_id: str, *, confirmed: bool = False):
        """Apply the same kind of fix to every open issue of the same type (the engine re-checks each one and respects locks)."""
        out = self._fix_engine().fix_similar(issue_id, confirmed=confirmed)
        self._recount(self._project())
        self._publish("fix_applied", issue_id=issue_id)
        return out

    def apply_safe_fixes(self, issue_ids: list[str] | None = None, *, code_prefix: str | None = None):
        out = self._fix_engine().apply_safe_fixes(issue_ids, code_prefix=code_prefix)
        self._recount(self._project())
        self._publish("fix_applied")
        return out

    # ------------------------------------------------------------------ history / comparison / report
    def history(self) -> list[dict[str, Any]]:
        return deepcopy(self._project().qc_runs)

    def compare_runs(self, a: "str | int", b: "str | int") -> RunComparison:
        """QC run ``a`` (older) vs ``b`` (newer): by run id or run number."""
        h = self._project().qc_history
        ea, eb = entry_for(h, a), entry_for(h, b)
        if ea is None or eb is None:
            raise QCError("One of those QC runs is not in the history.")
        if int(ea.get("number", 0)) > int(eb.get("number", 0)):
            ea, eb = eb, ea
        return compare_entries(ea, eb)

    def report(self, qc_run_id: str | None = None) -> str:
        p = self._project()
        if not p.qc_runs:
            raise QCError("Run QC first.")
        scores = p.qc_scores or QCScores()
        rec = p.qc_runs[-1]
        scene_names = {s.id: f"Scene {s.label}" for s in p.scenes}
        return build_report(p.project_name, rec, scores, p.qc_issues, p.qc_fixes, p.qc_ignored_issues, scene_names, p.qc_settings.block_level, p.render_qc_results, p.qc_settings.allow_export_override)

    def save_report(self, path: Path | None = None) -> Path:
        p = self._project()
        text = self.report()
        out = Path(path) if path else p.root / "qc" / f"qc_report_run{p.qc_runs[-1]['number']}.md"  # type: ignore[operator]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out

    # ------------------------------------------------------------------ export gate
    def export_gate(self) -> GateResult:
        p = self._project()
        s = p.qc_settings
        if not p.qc_runs or p.qc_scores is None:
            return GateResult(ExportDecision("READY", s.block_level, message="QC has not run yet"), False, True, "QC has not run yet")
        current = p.qc_runs[-1].get("content_hash") == self.current_content_hash(p) and bool(p.qc_runs[-1].get("content_hash"))
        failed = list(p.qc_runs[-1].get("failed", []))
        decision = decide_export(p.qc_issues, s.block_level, s.allow_export_override, failed)
        msg = decision.message if current else "The project changed since the last QC run. " + decision.message
        return GateResult(decision, current, not current, msg)

    # ------------------------------------------------------------------ post-render QC (a second pass on the actual file)
    def run_post_render_qc(self, render_id: str, output_path: Path, *, on_done: Callable[[dict[str, Any]], None] | None = None, on_error: Callable[[Job], None] | None = None) -> Job:
        from app.qc.render_checker import RenderedFileChecker  # noqa: PLC0415

        p = self._project()
        ctx = QCContext.build(p, ffmpeg=self.ffmpeg, probe=self.probe, rendered_file=Path(output_path))
        project_id = p.project_id
        from app.rendering.presets import output_size  # noqa: PLC0415

        out_w, out_h = output_size(*ctx.canvas, p.render_settings.resolution)  # the export is scaled to the chosen resolution
        expected = {"duration": ctx.duration, "width": out_w, "height": out_h, "fps": p.render_settings.fps or ctx.fps, "has_audio": bool(p.voice_over.asset_id or ctx.audio_clips())}
        checker = RenderedFileChecker()

        def work(jc) -> dict[str, Any]:
            ctx.cancel = jc.job.cancel_event
            return checker.inspect(ctx, Path(output_path), render_id=render_id, expected=expected, report=lambda f, m: jc.report(f * 100.0, m))

        def done(job: Job) -> None:
            cur = self._projects.current
            if cur is not None and cur.project_id == project_id:
                self._apply(StoreRenderQCCommand(cur, render_id, job.result))
                self._publish("render_qc", render_id=render_id)
            if on_done:
                on_done(job.result)

        def failed(job: Job) -> None:
            if on_error:
                on_error(job)

        return self._jobs.submit("qc.render", work, title="Checking the rendered file", on_complete=done, on_error=failed)

    def render_results(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._project().render_qc_results)
