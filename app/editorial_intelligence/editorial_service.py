"""EditorialIntelligenceService: the one entry point of the editorial layer (``ws.editorial``).

    UI  ->  service  ->  [analysis job: snapshot -> engine (stages) -> provider overlay -> recommendations]  ->  stored in ``Project.editorial``
                     ->  [user action: preview / apply / reject / reset override]  ->  structured Phase 10 operations in ONE command transaction

Analysis never touches the timeline. Applying is a separate, explicit user action; it is validated again against the live project, runs inside ``CommandStack.transaction`` (one undo step,
rolled back completely on any failure) and respects every lock and manual override.
"""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.commands import Command, CommandStack
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, ProjectError
from app.editorial_intelligence import override_resolver
from app.editorial_intelligence.context import EditorialContext, build_editorial_context
from app.editorial_intelligence.editorial_models import (
    AnalysisMode,
    AnalysisRun,
    Basis,
    ContentType,
    EditorialOverride,
    EditorialSettings,
    EditorialState,
    Finding,
    OverrideType,
    Recommendation,
    RecStatus,
    StageRecord,
    now_iso,
)
from app.editorial_intelligence.engine import EditorialEngine, EngineResult
from app.editorial_intelligence.provider import (
    EditorialIntelligenceProvider,
    EditorialProviderError,
    LocalEditorialProvider,
    compact_context,
    validate_observations_payload,
    validate_relations_payload,
    validate_story_payload,
)
from app.editorial_intelligence.scorecard import build_scorecard
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.phase11_commands import SetEditorialFieldsCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager

_log = get_logger(__name__)
MAX_HISTORY = 30
MAX_KEPT_DECIDED = 300  # applied / rejected recommendations kept so the decisions (and "do not offer again") survive


class EditorialError(AppError):
    pass


@dataclass
class EditorialProgress:
    state: str = "IDLE"  # IDLE | RUNNING | CANCELING
    fraction: float = 0.0
    message: str = ""
    job_id: str = ""
    stages: list[StageRecord] = field(default_factory=list)


@dataclass
class IncrementalPlan:
    mode: str = "none"  # none | incremental | full
    scene_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    source: str = "tracker"


class EditorialIntelligenceService:
    def __init__(self, projects: ProjectManager, jobs: JobManager, bus: EventBus, apply_command: Callable[[Command], None], commands: CommandStack, settings_getter: Callable[[], Any],
                 snapshot: Callable[[Project], Project] | None = None) -> None:
        self._projects, self._jobs, self._bus, self._apply, self._commands = projects, jobs, bus, apply_command, commands
        self._settings = settings_getter
        if snapshot is None:
            from app.qc.context import snapshot_project  # noqa: PLC0415

            snapshot = snapshot_project
        self._snapshot = snapshot
        self.tracker = None  # ProjectChangeTracker (installed by the Workspace)
        self.services: Any = None  # the live services the apply step may use: edit, presentation, editing, research, timeline (installed by the Workspace)
        self.global_preferences = None  # UserEditorialPreferences | None (the machine-wide copy; the Workspace keeps it in the settings file)
        self._engine: EditorialEngine | None = None
        self._provider: EditorialIntelligenceProvider = LocalEditorialProvider()
        self.progress = EditorialProgress()
        self._job: Job | None = None
        self._lock = threading.RLock()
        self._last_pub = (0.0, "")
        bus.subscribe(Topics.PROJECT_OPENED, lambda *_a: self._reset())
        bus.subscribe(Topics.PROJECT_CLOSED, lambda *_a: self._reset())

    # ------------------------------------------------------------------ basics
    def _project(self) -> Project:
        p = self._projects.current
        if p is None or p.root is None:
            raise ProjectError("Open or create a project first.")
        return p

    def _publish(self, kind: str, **extra: Any) -> None:
        self._bus.publish(Topics.EDITORIAL_UPDATED, kind=kind, **extra)

    def _reset(self) -> None:
        self.progress, self._job = EditorialProgress(), None

    @property
    def running(self) -> bool:
        return self._job is not None and not self._job.status.is_terminal

    @property
    def engine(self) -> EditorialEngine:
        if self._engine is None:
            self._engine = EditorialEngine(recommend=self._recommend, scorecard=build_scorecard)
        return self._engine

    @engine.setter
    def engine(self, value: EditorialEngine) -> None:
        self._engine = value

    @property
    def provider(self) -> EditorialIntelligenceProvider:
        return self._provider

    def set_provider(self, provider: EditorialIntelligenceProvider | None) -> None:
        self._provider = provider or LocalEditorialProvider()

    @property
    def state(self) -> EditorialState:
        return self._project().editorial

    @property
    def settings(self) -> EditorialSettings:
        return self._project().editorial.editorial_settings

    # ------------------------------------------------------------------ settings (undoable)
    def update_settings(self, **changes: Any) -> EditorialSettings:
        p = self._project()
        cur = p.editorial.editorial_settings
        new = deepcopy(cur)
        for k, v in changes.items():
            if not hasattr(new, k):
                raise EditorialError(f"Unknown editorial setting “{k}”.")
            if k == "mode":
                v = AnalysisMode(v)
            elif k == "content_type":
                v = ContentType(v)
            elif k == "window":
                v = int(v)
                if not 0 <= v <= 10:
                    raise EditorialError("The analysis window must be between 0 and 10 scenes.")
            elif k == "max_recommendations":
                v = int(v)
                if not 5 <= v <= 1000:
                    raise EditorialError("The number of recommendations must be between 5 and 1,000.")
            elif k == "min_confidence":
                v = float(v)
                if not 0.0 <= v <= 100.0:
                    raise EditorialError("The minimum confidence must be between 0 and 100.")
            elif k == "stages":
                from app.editorial_intelligence.editorial_models import STAGE_IDS  # noqa: PLC0415

                v = [s for s in STAGE_IDS if s in set(v)]
                if "context" not in v:
                    v.insert(0, "context")
            elif k == "provider" and v not in ("local", "api", "unavailable"):
                raise EditorialError("The provider must be local, api or unavailable.")
            setattr(new, k, v)
        if new == cur:
            return cur
        self._commands.execute(SetEditorialFieldsCommand(p, "Change editorial settings", editorial_settings=new))
        self._publish("settings")
        return new

    # ------------------------------------------------------------------ analysis
    def _global_prefs(self):
        return self.global_preferences

    def _recommend(self, ctx: EditorialContext, findings: list[Finding], previous: list[Recommendation], sections: dict[str, Any]) -> list[Recommendation]:
        from app.editorial_intelligence import recommendation_engine  # noqa: PLC0415  (written after the analyzers; imported lazily so the foundation runs without it)

        return recommendation_engine.generate_recommendations(ctx, findings, previous, sections)

    def run_analysis(self, *, scenes: list[str] | None = None, stages: list[str] | None = None, force: bool = False, trigger: str = "manual",
                     on_done: Callable[[AnalysisRun], None] | None = None, on_error: Callable[[Job], None] | None = None) -> Job:
        """Analyse the whole video (``scenes=None``) or only the given scenes. Unchanged work is reused unless ``force``."""
        p = self._project()
        return self._start(p, selected=set(stages) if stages else None, scene_filter=set(scenes) if scenes else None, force=force, trigger=trigger, on_done=on_done, on_error=on_error)

    def run_incremental(self, on_done: Callable[[AnalysisRun], None] | None = None, on_error: Callable[[Job], None] | None = None) -> Job | None:
        plan = self.plan_incremental()
        if plan.mode == "none":
            return None
        return self._start(self._project(), selected=None, scene_filter=set(plan.scene_ids) if plan.mode == "incremental" else None, force=False, trigger="incremental",
                           on_done=on_done, on_error=on_error)

    def retry_failed(self, on_done: Callable[[AnalysisRun], None] | None = None, on_error: Callable[[Job], None] | None = None) -> Job:
        """Run again only the stages that failed (or were skipped because of a failure) in the latest run."""
        p = self._project()
        run = p.editorial.editorial_analysis_history[-1] if p.editorial.editorial_analysis_history else None
        if run is None:
            raise EditorialError("There is no earlier analysis to retry.")
        todo = {s.stage for s in run.stages if s.status in ("FAILED", "SKIPPED", "CANCELED", "PENDING")}
        if not todo:
            raise EditorialError("Nothing failed in the latest analysis.")
        return self._start(p, selected=todo, scene_filter=None, force=True, trigger="retry", on_done=on_done, on_error=on_error)

    def cancel(self) -> bool:
        job = self._job
        if job is None or job.status.is_terminal:
            return False
        self._jobs.cancel(job.id)
        self.progress.state = "CANCELING"
        self._publish("progress", fraction=self.progress.fraction, message="Canceling…")
        return True

    def _start(self, p: Project, *, selected: set[str] | None, scene_filter: set[str] | None, force: bool, trigger: str, on_done, on_error) -> Job:
        if self.running:
            raise EditorialError("Editorial analysis is already running. Wait for it to finish or cancel it.")
        if not p.scenes:
            raise EditorialError("There are no scenes yet. Run scene analysis first.")
        t0 = self.tracker
        rev0, epoch0 = (t0.revision, t0.epoch) if t0 is not None else (0, -1)
        snap = self._snapshot(p)  # detached copy (UI thread); the worker never touches the live project
        settings = deepcopy(p.editorial.editorial_settings)
        ctx = build_editorial_context(snap, settings, self._global_prefs())
        previous = deepcopy(snap.editorial)
        run = AnalysisRun(run_id=f"edrun_{uuid.uuid4().hex[:10]}", mode=settings.mode.value, trigger=trigger, provider=self._provider.name if settings.ai_enabled else "local",
                          scope={"scenes": sorted(scene_filter) if scene_filter else [], "stages": sorted(selected) if selected else [], "all": scene_filter is None and selected is None, "force": force})
        project_id = p.project_id
        holder: dict[str, EngineResult] = {}
        provider = self._provider if settings.ai_enabled else None
        self.progress = EditorialProgress("RUNNING", 0.0, "Starting…", "", [])
        self._publish("run_started", trigger=trigger)

        def work(jc) -> EngineResult:
            cancel = jc.job.cancel_event

            def on_progress(frac: float, msg: str, records: list[StageRecord]) -> None:
                self.progress = EditorialProgress("CANCELING" if jc.job.cancel_requested else "RUNNING", frac, msg, jc.job.id, [deepcopy(r) for r in records])
                jc.report(frac * 100.0, msg)
                last_t, last_msg = self._last_pub
                now = time.monotonic()
                if msg != last_msg or frac >= 1.0 or now - last_t >= 0.15:
                    self._last_pub = (now, msg)
                    self._publish("progress", fraction=frac, message=msg)

            res = self.engine.run(ctx, previous, selected=selected, scene_filter=scene_filter, force=force, cancel=cancel, on_progress=on_progress)
            if provider is not None and res.status != "CANCELED":
                self._ai_overlay(provider, ctx, res)
            holder["r"] = res
            return res

        def install(res: EngineResult, canceled: bool = False) -> None:
            cur = self._projects.current
            if cur is None or cur.project_id != project_id:
                return
            self._install(cur, ctx, res, run, rev0, epoch0, canceled)
            self.progress = EditorialProgress("IDLE", 1.0, "Canceled" if canceled else "Done", "", res.records)
            self._publish("run_canceled" if canceled else "run_finished", run_id=run.run_id)
            log_event(_log, "editorial.run", run_id=run.run_id, status=run.status, trigger=trigger, recommendations=run.recommendations)

        def done(job: Job) -> None:
            install(job.result)
            if on_done:
                on_done(run)

        def cancelled(job: Job) -> None:
            res = holder.get("r")
            if res is not None:
                install(res, canceled=True)
            else:
                self.progress = EditorialProgress()
                self._publish("run_canceled")

        def failed(job: Job) -> None:
            self.progress = EditorialProgress("IDLE", 0.0, f"Editorial analysis failed: {job.error}", "", [])
            self._publish("run_failed", error=job.error)
            if on_error:
                on_error(job)

        title = {"incremental": "Editorial analysis — changes only", "retry": "Editorial analysis — retry"}.get(trigger, "AI editorial analysis")
        job = self._jobs.submit("editorial.run", work, title=title, on_complete=done, on_error=failed, on_cancel=cancelled)
        self._job = job
        self.progress.job_id = job.id
        return job

    # ------------------------------------------------------------------ AI overlay (optional; validated; never overrules the rule-based result)
    def _ai_overlay(self, provider: EditorialIntelligenceProvider, ctx: EditorialContext, res: EngineResult) -> None:
        scene_ids = set(ctx.scene_ids)
        payload = compact_context(ctx)
        dropped = 0
        try:
            roles, problems = validate_story_payload(provider.analyze_story(payload) or {}, scene_ids) if provider.available() else ({}, [])
            dropped += len(problems)
            narrative = res.sections.setdefault("narrative", {"scenes": {}, "global": {}})
            for sid, items in roles.items():
                entry = narrative.setdefault("scenes", {}).setdefault(sid, {"scene_id": sid, "roles": [], "evidence_terms": [], "uncertain": True, "basis": "AI", "note": ""})
                have = {r["role"] for r in entry.get("roles", [])}
                for role, conf in items:
                    if role.value not in have:
                        entry.setdefault("roles", []).append({"role": role.value, "confidence": conf})
                entry["basis"] = "RULES+AI" if entry.get("evidence_terms") else "AI"
            rel, problems = validate_relations_payload(provider.analyze_scene_relationships(payload) or {}, scene_ids) if provider.available() else ([], [])
            dropped += len(problems)
            relations = res.sections.setdefault("relationships", {"scenes": {}, "global": {}}).setdefault("global", {}).setdefault("relations", [])
            have_rel = {(r.get("from_scene"), r.get("to_scene"), r.get("type")) for r in relations}
            order = {sid: i for i, sid in enumerate(ctx.scene_ids)}
            for a, b, typ, conf, ev in rel:
                if (a, b, typ.value) not in have_rel:
                    relations.append({"from_scene": a, "to_scene": b, "type": typ.value, "confidence": conf, "evidence": [ev] if ev else [], "distance": abs(order[a] - order[b]), "basis": "AI"})
            obs, problems = validate_observations_payload(provider.generate_recommendations({**payload, "findings": [f.title for f in res.findings][:50]}) or {}, scene_ids) if provider.available() else ([], [])
            dropped += len(problems)
            from app.editorial_intelligence.editorial_models import ActionKind, ActionSpec, finding_key  # noqa: PLC0415

            for o in obs:
                res.findings.append(Finding(f"ai_{uuid.uuid4().hex[:8]}", "recommendations", o["category"], o["scene_ids"], [], o["title"], o["detail"], o["confidence"], 40.0, Basis.JUDGMENT, [],
                                            ActionSpec(ActionKind.REVIEW_ONLY, {}, [], "AI observation: review it yourself."), [], "", [], [], False, False,
                                            finding_key("ai", o["category"].value, ",".join(o["scene_ids"]), o["title"][:40])))
            res.log.append(f"AI analysis ({provider.name}): {len(roles)} scene role(s), {len(rel)} relation(s), {len(obs)} observation(s); {dropped} invalid item(s) dropped.")
        except EditorialProviderError as exc:
            res.log.append(f"AI analysis FAILED: {exc.user_message}. The rule-based analysis is unaffected; the run is reported as partial.")
            res.status = "PARTIAL" if res.status == "COMPLETE" else res.status
            res.records.append(StageRecord("ai_analysis", "FAILED", error=exc.user_message, message="AI analysis failed"))
        except Exception as exc:  # noqa: BLE001
            _log.exception("editorial AI overlay failed")
            res.log.append(f"AI analysis FAILED ({type(exc).__name__}).")
            res.status = "PARTIAL" if res.status == "COMPLETE" else res.status
            res.records.append(StageRecord("ai_analysis", "FAILED", error=type(exc).__name__, message="AI analysis failed"))

    # ------------------------------------------------------------------ install results
    def _install(self, p: Project, ctx: EditorialContext, res: EngineResult, run: AnalysisRun, rev0: int, epoch0: int, canceled: bool) -> None:
        st = p.editorial
        changed_scenes, unknown = self._changed_since(rev0, epoch0, set(ctx.scene_ids))
        stale_ids = set(ctx.scene_ids) if unknown else changed_scenes  # the project moved while the job ran: what rests on those scenes is out of date
        recs = self._merge_recommendations(st.editorial_recommendations, res.recommendations, res, set(ctx.scene_ids), stale_ids)
        run.stages = res.records
        run.finished_at = now_iso()
        run.status = "CANCELED" if canceled else res.status
        run.recommendations = sum(1 for r in recs if r.status is RecStatus.PENDING)
        run.scorecard = res.scorecard
        run.content_hash = ctx.signature if res.status == "COMPLETE" and not stale_ids else ""
        run.log = res.log[-30:]
        run.message = "; ".join(r.error for r in res.records if r.error)[:300]
        sections = deepcopy(res.sections)
        fields = dict(sections_to_views(sections, ctx))
        history = [*st.editorial_analysis_history, run][-MAX_HISTORY:]
        stage_cache = deepcopy(res.stage_cache)
        if self.tracker is not None:
            stage_cache["_baseline"] = {"epoch": epoch0, "revision": rev0, "project_id": p.project_id}  # what the change log is measured from (a later session cannot use it: see staleness)
        self._apply(SetEditorialFieldsCommand(p, "Store editorial analysis", sections=sections, stage_cache=stage_cache, findings=res.findings,
                                              editorial_recommendations=recs, editorial_analysis_history=history, scorecard=res.scorecard or st.scorecard,
                                              project_hash=run.content_hash or st.project_hash, **fields))
        if self.tracker is not None and not unknown and not canceled and res.status != "FAILED":
            self.tracker.mark_clean("editorial", rev0)

    def _changed_since(self, rev0: int, epoch0: int, live: set[str]) -> tuple[set[str], bool]:
        t = self.tracker
        if t is None:
            return set(), False
        try:
            if epoch0 != t.epoch:
                return set(), True
            cs = t.peek(since=rev0)
        except Exception:  # noqa: BLE001
            return set(), True
        if cs.empty:
            return set(), False
        if cs.all_scenes or cs.unknown:
            return set(), True
        return set(cs.scene_ids) & live, False

    @staticmethod
    def _merge_recommendations(previous: list[Recommendation], new: list[Recommendation] | None, res: EngineResult, live: set[str], stale_ids: set[str]) -> list[Recommendation]:
        """New recommendations replace the pending ones; what the user decided (applied / rejected) is kept, and a rejected one is not offered again while its inputs are unchanged."""
        if new is None:  # the recommendation step failed or did not run: keep what there was, honestly
            return deepcopy(previous)
        by_key = {r.key: r for r in previous if r.key}
        out: list[Recommendation] = []
        seen: set[str] = set()
        for r in new:
            prev = by_key.get(r.key) if r.key else None
            if prev is not None and prev.status in (RecStatus.REJECTED, RecStatus.APPLIED) and prev.status is RecStatus.REJECTED and prev.inputs_hash == r.inputs_hash:
                kept = deepcopy(prev)
                out.append(kept)
            else:
                if prev is not None and prev.status is RecStatus.PENDING:
                    r.created_at = prev.created_at
                out.append(r)
            seen.add(r.key)
        for prev in previous:
            if prev.key in seen:
                continue
            if prev.status in (RecStatus.APPLIED, RecStatus.REJECTED, RecStatus.FAILED):
                out.append(deepcopy(prev))
            elif prev.status in (RecStatus.PENDING, RecStatus.BLOCKED, RecStatus.INVALID) and not (set(prev.scene_ids) & res.analysed_scenes) and set(prev.scene_ids) <= live:
                out.append(deepcopy(prev))  # its scenes were not analysed again, so the finding still stands
        decided = [r for r in out if r.status in (RecStatus.APPLIED, RecStatus.REJECTED, RecStatus.FAILED)]
        if len(decided) > MAX_KEPT_DECIDED:
            drop = {id(r) for r in decided[: len(decided) - MAX_KEPT_DECIDED]}
            out = [r for r in out if id(r) not in drop]
        if stale_ids:
            for r in out:
                if r.status in (RecStatus.PENDING, RecStatus.BLOCKED) and set(r.scene_ids) & stale_ids:
                    r.status, r.status_note, r.status_changed_at = RecStatus.OUTDATED, "The project changed while the analysis was running. Run the analysis again.", now_iso()
        return out

    # ------------------------------------------------------------------ incremental planning (from the Phase 9 change tracker: no hashing)
    def plan_incremental(self) -> IncrementalPlan:
        p = self._project()
        st = p.editorial
        if not st.editorial_analysis_history or not st.sections:
            return IncrementalPlan("full", [], ["No earlier editorial analysis."], "none")
        t = self.tracker
        base = st.stage_cache.get("_baseline") or {}
        if t is None or base.get("epoch") != t.epoch or base.get("project_id") != p.project_id:
            # a new session (or no change log): the log cannot say what changed, so the hashes decide - a full run still reuses every stage whose inputs are unchanged
            return IncrementalPlan("full", [], ["The change log does not cover this session: unchanged work is reused by comparing content hashes."], "hash")
        cs = t.peek(since=int(base.get("revision", 0)))
        if cs.empty:
            return IncrementalPlan("none", [], [], "tracker")
        if cs.all_scenes or cs.unknown:
            return IncrementalPlan("full", [], ["A change affects the whole project or could not be located."], "tracker")
        live = {s.id for s in p.scenes}
        ids = sorted(set(cs.scene_ids) & live)
        reasons = [f"{len(ids)} scene(s) changed: {', '.join(ids[:5])}{'…' if len(ids) > 5 else ''}"] if ids else ["Project-wide settings changed."]
        return IncrementalPlan("incremental", ids, reasons, "tracker")

    def staleness(self, exact: bool = False) -> dict[str, Any]:
        """Which part of the stored analysis no longer matches the project. From the change log (cheap, safe on every edit); ``exact=True`` compares content hashes (builds the context)."""
        p = self._project()
        st = p.editorial
        if not st.editorial_analysis_history:
            return {"stale": True, "scene_ids": [], "reason": "Editorial analysis has not run yet", "source": "none"}
        if exact:
            ctx = build_editorial_context(p, st.editorial_settings, self._global_prefs())
            same = bool(st.project_hash) and st.project_hash == ctx.signature
            return {"stale": not same, "scene_ids": [], "reason": "" if same else "The project differs from what was analysed.", "source": "hash"}
        plan = self.plan_incremental()
        if plan.mode == "none":
            return {"stale": False, "scene_ids": [], "reason": "", "source": plan.source}
        return {"stale": True, "scene_ids": plan.scene_ids, "reason": "; ".join(plan.reasons), "source": plan.source, "all": plan.mode == "full"}

    # ------------------------------------------------------------------ reading results
    def latest_run(self) -> AnalysisRun | None:
        h = self._project().editorial.editorial_analysis_history
        return h[-1] if h else None

    def recommendations(self, *, status: RecStatus | str | None = None, category: str | None = None, scene_id: str | None = None, include_decided: bool = True) -> list[Recommendation]:
        from app.editorial_intelligence.editorial_models import PRIORITY_ORDER  # noqa: PLC0415

        out = []
        for r in self._project().editorial.editorial_recommendations:
            if status is not None and r.status.value != (status.value if isinstance(status, RecStatus) else status):
                continue
            if category is not None and r.category.value != category:
                continue
            if scene_id is not None and scene_id not in r.scene_ids:
                continue
            if not include_decided and r.status in (RecStatus.APPLIED, RecStatus.REJECTED):
                continue
            out.append(r)
        return sorted(out, key=lambda r: (PRIORITY_ORDER[r.priority], -r.confidence, r.time_range[0] if r.time_range else 0.0, r.recommendation_id))

    def recommendation(self, rec_id: str) -> Recommendation:
        for r in self._project().editorial.editorial_recommendations:
            if r.recommendation_id == rec_id:
                return r
        raise EditorialError("That recommendation no longer exists. Run the analysis again.")

    def overrides_of(self, scene_id: str) -> dict[str, list[str]]:
        return override_resolver.scene_overrides(self._project(), scene_id)

    # ------------------------------------------------------------------ user decisions on recommendations
    def _replace_rec(self, rec: Recommendation, **changes: Any) -> Recommendation:
        p = self._project()
        new = deepcopy(rec)
        for k, v in changes.items():
            setattr(new, k, v)
        new.status_changed_at = now_iso()
        recs = [new if r.recommendation_id == rec.recommendation_id else r for r in p.editorial.editorial_recommendations]
        self._commands.execute(SetEditorialFieldsCommand(p, f"Recommendation {new.status.value.lower()}", editorial_recommendations=recs))
        return new

    def reject_recommendation(self, rec_id: str, reason: str = "", *, keep: bool = False) -> Recommendation:
        """Ignore a recommendation (undoable). ``keep=True`` also records that the user wants this aspect left as it is."""
        p = self._project()
        rec = self.recommendation(rec_id)
        if rec.status is RecStatus.APPLIED:
            raise EditorialError("That recommendation was already applied. Undo it instead.")
        with self._commands.transaction("Ignore recommendation", scope="editorial"):
            new = self._replace_rec(rec, status=RecStatus.REJECTED, status_note=reason or "Ignored by you")
            if keep:
                rec_kind = {"PACING": OverrideType.TIMING_OVERRIDE, "VISUAL_RELEVANCE": OverrideType.VISUAL_OVERRIDE, "VISUAL_DIVERSITY": OverrideType.VISUAL_OVERRIDE,
                            "VISUAL_CONTINUITY": OverrideType.VISUAL_OVERRIDE, "STYLE_CONSISTENCY": OverrideType.STYLE_OVERRIDE, "CAPTION_HIERARCHY": OverrideType.CAPTION_OVERRIDE,
                            "EMPHASIS": OverrideType.STYLE_OVERRIDE, "AUDIO_VISUAL_ALIGNMENT": OverrideType.TIMING_OVERRIDE}.get(rec.category.value, OverrideType.STYLE_OVERRIDE)
                ov = EditorialOverride(f"edov_{uuid.uuid4().hex[:8]}", rec_kind, rec.scene_ids[0] if rec.scene_ids else "", "", reason, "KEEP", rec.key)
                self._commands.execute(SetEditorialFieldsCommand(p, "Keep as is", editorial_overrides=[*p.editorial.editorial_overrides, ov]))
            self._note_decision(rec, "REJECTED")
        self._publish("recommendation_changed", recommendation_id=rec_id, status="REJECTED")
        return new

    def reset_override(self, scene_id: str, override_type: OverrideType | str, *, note: str = "") -> EditorialOverride:
        """Give a manual override back to the AI for one scene and aspect: from then on recommendations may touch it again (they still need your approval)."""
        p = self._project()
        typ = override_type if isinstance(override_type, OverrideType) else OverrideType(override_type)
        ov = EditorialOverride(f"edov_{uuid.uuid4().hex[:8]}", typ, scene_id, "", "reset by you", "RESET", "", note=note)
        self._commands.execute(SetEditorialFieldsCommand(p, "Reset override", editorial_overrides=[*p.editorial.editorial_overrides, ov]))
        self._publish("overrides", scene_id=scene_id)
        return ov

    def _note_decision(self, rec: Recommendation, decision: str) -> None:
        """Hook for the preference adapter: a decision is evidence, never an instruction (see preference_adapter)."""
        try:
            from app.editorial_intelligence import preference_adapter  # noqa: PLC0415

            note = getattr(preference_adapter, "note_decision", None)
            if note is not None:
                note(self._project(), self._commands, rec, decision, self.global_preferences)
        except Exception:  # noqa: BLE001  (learning must never break an edit)
            _log.warning("preference signal could not be recorded", exc_info=True)


# ------------------------------------------------------------------------------------------------------------------------ section -> typed view
def sections_to_views(sections: dict[str, dict[str, Any]], ctx: EditorialContext) -> dict[str, Any]:
    """Materialise the typed / named fields of ``EditorialState`` from the canonical stage sections."""
    from app.editorial_intelligence.editorial_models import Relation, SceneRole, StorySection, StoryStructure, VisualStrategy  # noqa: PLC0415
    from app.core.serialization import from_plain  # noqa: PLC0415

    def safe(cls, d):
        try:
            return from_plain(cls, d)
        except (KeyError, TypeError, ValueError):
            return None

    out: dict[str, Any] = {}
    nar = sections.get("narrative") or {}
    roles = {sid: safe(SceneRole, v) for sid, v in (nar.get("scenes") or {}).items()}
    out["scene_roles"] = {k: v for k, v in roles.items() if v is not None}
    arc = (sections.get("story_arc") or {}).get("global") or {}
    structure = safe(StoryStructure, arc.get("structure")) if arc.get("structure") else None
    out["narrative_structure"] = structure
    out["story_sections"] = list(structure.sections) if structure else [s for s in (safe(StorySection, d) for d in arc.get("sections") or []) if s is not None]
    rel = (sections.get("relationships") or {}).get("global") or {}
    out["scene_relationships"] = [r for r in (safe(Relation, d) for d in rel.get("relations") or []) if r is not None]
    seq = sections.get("sequence") or {}
    out["visual_strategy"] = {sid: v for sid, v in ((sid, safe(VisualStrategy, d)) for sid, d in (seq.get("scenes") or {}).items()) if v is not None}
    out["visual_sequence_analysis"] = seq
    out["visual_consistency_results"] = sections.get("continuity") or {}
    out["visual_diversity_results"] = sections.get("diversity") or {}
    out["pacing_recommendations"] = sections.get("pacing") or {}
    out["emphasis_results"] = sections.get("emphasis") or {}
    out["evidence_alignment_results"] = sections.get("evidence") or {}
    out["style_consistency_results"] = sections.get("style") or {}
    return out
