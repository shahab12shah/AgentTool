"""Visual research application service: Structured Scene -> Brief -> Queries -> Providers -> Candidates -> Scores -> Ranking -> Assignment.

* every scene is researched by its own job (independently recoverable; one failure never touches other scenes)
* nothing is added to the timeline here, and nothing becomes a project asset until the user selects it
* user decisions (choose, approve, reject, skip) are undoable commands and are always recorded as such
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Callable

from app.analysis.models import SceneStatus
from app.core.commands import Command, CommandStack
from app.core.config import Settings
from app.core.events import EventBus, Topics
from app.core.exceptions import AcquisitionError, AppError, JobCancelled, ProviderError, ResearchError
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.media.asset import SourceType as S
from app.media.importer import LINK_COPY, MediaImporter, PreparedMedia
from app.project.phase3_commands import ApplyResearchCommand, SceneDecisionCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.research.brief import BriefBuilder
from app.research.dedupe import fingerprint_file
from app.research.engine import ResearchEngine, ResearchOutcome, RunContext, decide_status
from app.research.evaluation import VisualEvaluationService
from app.research.http import HttpClient
from app.research.models import (
    Acquisition,
    Candidate,
    CandidateStatus,
    EvidenceKind,
    ResearchBrief,
    ResearchQuery,
    ResearchSession,
    ResearchStatus,
    SceneResearchState,
    VisualAssignment,
    now_iso,
)
from app.research.providers.ai_image import AIImageProvider
from app.research.providers.base import ProviderRegistry, SearchContext
from app.research.providers.local_stock import LocalStockProvider
from app.research.providers.pexels import PexelsProvider
from app.research.providers.screenshot import ScreenshotProvider
from app.research.providers.wikimedia import WikimediaProvider
from app.research.providers.youtube import YouTubeProvider
from app.research.queries import STRATEGIES, QueryGenerator
from app.research.ranking import RankingHistory, UsedVisual, title_terms
from app.research.dedupe import identity_keys
from app.services.media_service import MediaService
from app.visual.research import SceneResearchResult, VisualResearchService
from app.visual.preferences import SourceKind

_log = get_logger(__name__)
Apply = Callable[[Command], None]


class ResearchService(VisualResearchService):
    def __init__(self, projects: ProjectManager, commands: CommandStack, jobs: JobManager, bus: EventBus, apply: Apply,
                 media: MediaService, importer: MediaImporter, settings: Callable[[], Settings]) -> None:
        self._projects, self._commands, self._jobs, self._bus, self._apply = projects, commands, jobs, bus, apply
        self._media, self._importer, self._settings = media, importer, settings
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self.brief_builder = BriefBuilder()
        self.registry = ProviderRegistry()
        s = settings
        http = HttpClient(min_interval=0.2)
        self.registry.register(LocalStockProvider(lambda: s().local_stock_dir, lambda: s().ffmpeg_path, http=http))
        self.registry.register(WikimediaProvider(lambda: s().wikimedia_api_url, http=http))
        self.registry.register(YouTubeProvider(lambda: s().youtube_api_url, lambda: s().youtube_key_env, http=http))
        self.registry.register(PexelsProvider(lambda: s().pexels_api_url, lambda: s().pexels_key_env, http=http))
        self.registry.register(ScreenshotProvider(lambda: s().chromium_path, http=http))
        self.registry.register(AIImageProvider(lambda: s().ai_image_base_url, lambda: s().ai_image_model, lambda: s().ai_image_key_env, http=http))
        self.engine = ResearchEngine(self.registry, http=http)

    # ------------------------------------------------------------------ basics
    def _project(self) -> Project:
        if self._projects.current is None or self._projects.current.root is None:
            raise ResearchError("Open or create a project first.")
        return self._projects.current

    def provider_report(self) -> list[dict]:
        out = []
        for p in self.registry.all():
            ok, why = p.is_available()
            out.append({"name": p.name, "label": p.label, "available": ok, "reason": why, "verified_live": p.verified_live,
                        "sources": [t.value for t in p.source_types]})
        return out

    def brief(self, scene_id: str) -> ResearchBrief:
        return self.brief_builder.build(self._project(), scene_id)

    @staticmethod
    def brief_hash(b: ResearchBrief) -> str:
        return hashlib.sha1("|".join([b.narration, b.topic, b.primary_subject, b.secondary_subject, b.visual_type]).encode()).hexdigest()[:12]

    def state(self, scene_id: str) -> SceneResearchState:
        return self._project().research_status.get(scene_id) or SceneResearchState(scene_id)

    def is_stale(self, scene_id: str) -> bool:
        """True when the scene's content changed after it was researched (the research may no longer fit)."""
        project = self._project()
        st = project.research_status.get(scene_id)
        if st is None or not st.brief_hash:
            return False
        try:
            return self.brief_hash(self.brief(scene_id)) != st.brief_hash
        except AppError:
            return False

    def _next_id(self, kind: str) -> str:
        rs = self._project().research_settings
        with self._lock:
            if kind == "candidate":
                rs.candidate_counter += 1
                return f"candidate_{rs.candidate_counter:05d}"
            if kind == "query":
                rs.query_counter += 1
                return f"query_{rs.query_counter:05d}"
            rs.session_counter += 1
            return f"session_{rs.session_counter:04d}"

    # ------------------------------------------------------------------ history (repetition / continuity / balance)
    def history_for(self, project: Project, scene_id: str) -> RankingHistory:
        order = [s.id for s in project.scenes]
        idx = order.index(scene_id) if scene_id in order else 0
        used: list[UsedVisual] = []
        counts: Counter = Counter()
        for sid, a in project.visual_assignments.items():
            if sid == scene_id or a.skipped or a.candidate_id is None:
                continue
            c = project.visual_candidates.get(a.candidate_id)
            if c is None:
                continue
            si = order.index(sid) if sid in order else 0
            keys = identity_keys(c) | ({f"asset:{a.asset_id}"} if a.asset_id else set())
            used.append(UsedVisual(si, keys, c.fingerprint, title_terms(c), c.source_type))
            if c.source_type:
                counts[c.source_type] += 1
        prev_terms: frozenset[str] = frozenset()
        if idx > 0:
            pid = order[idx - 1]
            pa = project.visual_assignments.get(pid)
            pst = project.research_status.get(pid)
            cid = pa.candidate_id if pa and pa.candidate_id else (pst.best_id if pst else None)
            if cid and cid in project.visual_candidates:
                prev_terms = title_terms(project.visual_candidates[cid])
        return RankingHistory(used, counts, idx, prev_terms)

    # ------------------------------------------------------------------ research
    def research_scene(self, scene_id: str, strategy: str = "initial", fresh: bool = False, expand: bool = False) -> Job:
        project = self._project()
        brief = self.brief(scene_id)
        with self._lock:
            if scene_id in self._running:
                raise ResearchError("Research for this scene is already running.")
            self._running.add(scene_id)
        try:
            prev_state = project.research_status.get(scene_id)
            prior_q = [q for q in project.research_queries.values() if q.scene_id == scene_id]
            generation = 0 if strategy == "initial" else (prev_state.generation if prev_state else 0) + 1
            qgen = QueryGenerator(project.research_settings)
            queries = qgen.generate(brief, strategy, generation, previous=prior_q if strategy != "initial" else [],
                                    id_factory=lambda: self._next_id("query"))
            if not queries:
                raise ResearchError("No new research queries could be generated for this scene.")
            session_id = self._next_id("session")
            existing = [deepcopy(c) for c in project.visual_candidates.values() if c.scene_id == scene_id]
            prefs, rs = deepcopy(project.visual_preferences), deepcopy(project.research_settings)
            history = self.history_for(project, scene_id)
            root = project.root
            assert root is not None
            running_state = SceneResearchState(scene_id, ResearchStatus.RESEARCHING, "Researching…", prev_state.best_id if prev_state else None,
                                               list(prev_state.alternatives) if prev_state else [], list(prev_state.ranked) if prev_state else [],
                                               prev_state.session_id if prev_state else None, generation, [], self.brief_hash(brief),
                                               prev_state.status.value if prev_state else "NOT_STARTED")
            self._apply(SceneDecisionCommand(project, scene_id, "Researching", state=running_state))
        except Exception:
            with self._lock:
                self._running.discard(scene_id)
            raise
        ffmpeg = self._settings().ffmpeg_path

        def work(ctx) -> ResearchOutcome:
            rctx = RunContext(root / "cache" / "research", root, ctx.is_cancelled, lambda f, m: ctx.report(f * 100, m), ffmpeg, fresh, expand)
            return self.engine.run(brief, queries, prefs, rs, history, rctx, lambda: self._next_id("candidate"), existing)

        def finish() -> None:
            with self._lock:
                self._running.discard(scene_id)

        def completed(job: Job) -> None:
            finish()
            if self._projects.current is not project:
                return
            out: ResearchOutcome = job.result
            self._install(project, scene_id, brief, queries, out, session_id, generation, strategy, fresh, expand, prev_state)

        def failed(job: Job) -> None:
            finish()
            if self._projects.current is project:
                msg = job.error or "Visual research failed."
                self._apply(SceneDecisionCommand(project, scene_id, "Research failed", state=SceneResearchState(
                    scene_id, ResearchStatus.ERROR, f"Visual research unavailable. {msg}", generation=generation,
                    brief_hash=self.brief_hash(brief), updated_at=now_iso())))

        def cancelled(job: Job) -> None:
            finish()
            if self._projects.current is project:
                self._apply(SceneDecisionCommand(project, scene_id, "Research cancelled", state=deepcopy(prev_state) or SceneResearchState(scene_id)))

        log_event(_log, "research.started", scene_id=scene_id, strategy=strategy, fresh=fresh, expand=expand, queries=len(queries))
        return self._jobs.submit("visual_research", work, title=f"Researching visuals for scene {self._label(project, scene_id)}",
                                 on_complete=completed, on_error=failed, on_cancel=cancelled)

    def _install(self, project: Project, scene_id: str, brief: ResearchBrief, queries: list[ResearchQuery], out: ResearchOutcome,
                 session_id: str, generation: int, strategy: str, fresh: bool, expand: bool, prev_state: SceneResearchState | None) -> None:
        old_assign = project.visual_assignments.get(scene_id)
        prefs = project.visual_preferences
        used_disabled = [TYPE_TO_KIND(t) for t in out.searched_sources if not prefs.setting(SourceKind(TYPE_TO_KIND(t))).enabled] if expand else []
        state = self._state_from(scene_id, out, brief, session_id, generation, used_disabled)
        # a user's approval survives new research: it is never silently replaced
        if old_assign and old_assign.approved:
            state.status = ResearchStatus.APPROVED
            state.message = "Approved visual kept. New candidates are available for comparison."
        session = ResearchSession(session_id, scene_id, strategy, fresh, expand, brief, [q.query_id for q in queries],
                                  [c.candidate_id for c in out.candidates], out.reports,
                                  "FAILED" if state.status is ResearchStatus.ERROR else ("PARTIAL" if any(r.status == "FAILED" for r in out.reports) else "COMPLETE"),
                                  finished_at=now_iso())
        stats = {r.provider: {"queries": r.queries, "status": r.status, "error": r.error, "when": now_iso()} for r in out.reports}
        self._apply(ApplyResearchCommand(project, scene_id, queries, session, out.candidates, out.scores, state, stats))
        log_event(_log, "research.finished", scene_id=scene_id, status=state.status.value, candidates=len(out.candidates))

    def _state_from(self, scene_id: str, out: ResearchOutcome, brief: ResearchBrief, session_id: str | None, generation: int,
                    expanded: list[str]) -> SceneResearchState:
        best = out.ranked[0].candidate_id if out.ranked else None
        alts = [e.candidate_id for e in out.ranked if e.role == "ALTERNATIVE"]
        return SceneResearchState(scene_id, out.status, out.message, best, alts, out.ranked, session_id, generation, expanded,
                                  self.brief_hash(brief))

    def research_many(self, scene_ids: list[str], **kw) -> list[Job]:
        """One independent job per scene; scenes that cannot be researched are skipped with a message (never a hard stop)."""
        jobs = []
        for sid in scene_ids:
            try:
                jobs.append(self.research_scene(sid, **kw))
            except AppError as exc:
                self._bus.publish(Topics.STATUS, message=f"Scene {self._label(self._project(), sid)}: {exc.user_message}")
        return jobs

    def search_again(self, scene_id: str, strategy: str | None = None, fresh: bool = False, expand: bool = False) -> Job:
        """New query variations (broader / specific / process / evidence / alternative / sources) — never the same search twice."""
        st = self._project().research_status.get(scene_id)
        if strategy is None:
            strategy = STRATEGIES[((st.generation if st else 0)) % len(STRATEGIES)]
        return self.research_scene(scene_id, strategy, fresh, expand)

    def generate_more(self, scene_id: str, expand: bool = False) -> Job:
        """More candidates through a different interpretation (alternative terms and visual types)."""
        return self.research_scene(scene_id, "alternative", False, expand)

    def expandable_sources(self) -> list[dict]:
        """Sources the user disabled that have a usable provider (for the 'Expand Sources?' question)."""
        project = self._project()
        out = []
        for k in SourceKind:
            if project.visual_preferences.setting(k).enabled:
                continue
            st = _TYPE[k]
            usable = [p.label for p in self.registry.for_source(st) if p.is_available()[0]]
            if usable:
                out.append({"kind": k.value, "source_type": st.value, "providers": usable})
        return out

    def rescore(self, scene_id: str) -> Job:
        """Re-evaluate and re-rank the stored candidates (after changing the minimum score, weights or preferences)."""
        project = self._project()
        brief = self.brief(scene_id)
        cands = [deepcopy(c) for c in project.visual_candidates.values() if c.scene_id == scene_id]
        if not cands:
            raise ResearchError("There are no candidates to evaluate yet.")
        prefs, rs, history = deepcopy(project.visual_preferences), deepcopy(project.research_settings), self.history_for(project, scene_id)
        prev = project.research_status.get(scene_id)

        def work(ctx):
            ctx.report(30, "Evaluating candidates")
            return self.engine.evaluate_and_rank(brief, cands, prefs, rs, history)

        def done(job: Job) -> None:
            if self._projects.current is not project:
                return
            out: ResearchOutcome = job.result
            out.reports = []
            out.status, out.message = decide_status(out, brief, prefs, True, False, final=False)
            state = self._state_from(scene_id, out, brief, prev.session_id if prev else None, prev.generation if prev else 0, [])
            a = project.visual_assignments.get(scene_id)
            if a and a.approved:
                state.status = ResearchStatus.APPROVED
            self._apply(ApplyResearchCommand(project, scene_id, [], None, out.candidates, out.scores, state))

        return self._jobs.submit("candidate_evaluation", work, title=f"Evaluating candidates for scene {self._label(project, scene_id)}", on_complete=done)

    # ------------------------------------------------------------------ decisions
    def _candidate(self, project: Project, scene_id: str, candidate_id: str) -> Candidate:
        c = project.visual_candidates.get(candidate_id)
        if c is None or c.scene_id != scene_id:
            raise ResearchError("That candidate is not part of this scene's research.")
        return c

    def _assignment_for(self, project: Project, c: Candidate, by: str, approved: bool = False) -> VisualAssignment:
        score = project.candidate_scores.get(c.candidate_id)
        return VisualAssignment(
            scene_id=c.scene_id, candidate_id=c.candidate_id, asset_id=c.asset_id, selected_by=by,
            accuracy_score=score.overall if score else None, approved=approved, evidence_kind=c.evidence_kind,
            acquisition=c.acquisition, segment=deepcopy(c.segment) if c.segment.basis != "UNKNOWN" or c.segment.start is not None else None,
            recommended_duration=score.recommended_duration if score else None, source_type=c.source_type,
            note="Reference only: this application does not download this source." if c.acquisition is Acquisition.REFERENCE_ONLY else "")

    def choose_candidate(self, scene_id: str, candidate_id: str, by: str = "USER") -> Job | None:
        """Select a candidate (the user may pick any score). Acquires the media into the project unless it is reference-only."""
        project = self._project()
        c = deepcopy(self._candidate(project, scene_id, candidate_id))
        if c.status is CandidateStatus.REJECTED:
            raise ResearchError("This candidate was rejected. Un-reject it first.")
        if c.status is CandidateStatus.PROPOSED and c.acquisition is Acquisition.GENERATE:
            raise ResearchError("This AI visual has not been generated yet. Use “Generate AI Visual” first.")
        old = project.visual_assignments.get(scene_id)
        if old and old.candidate_id and old.candidate_id != candidate_id and old.candidate_id in project.visual_candidates:
            prior = deepcopy(project.visual_candidates[old.candidate_id])
            if prior.status is CandidateStatus.SELECTED:
                prior.status = CandidateStatus.READY
            changed = [prior]
        else:
            changed = []
        if c.status is not CandidateStatus.ACQUIRED:
            c.status = CandidateStatus.SELECTED
        assignment = self._assignment_for(project, c, by)
        if old and old.approved and old.candidate_id == candidate_id:
            assignment.approved = True
        st = deepcopy(self.state(scene_id))
        if st.status is ResearchStatus.APPROVED and not assignment.approved:
            st.status = ResearchStatus.CANDIDATES_READY
        self._commands.execute(SceneDecisionCommand(project, scene_id, f"Use visual for scene {self._label(project, scene_id)}",
                                                    state=st, assignment=assignment, candidates=changed + [c]))
        if c.acquisition is Acquisition.REFERENCE_ONLY or c.asset_id:
            return None
        return self._acquire(project, scene_id, c)

    def _acquire(self, project: Project, scene_id: str, c: Candidate) -> Job:
        """CandidateDownloadJob: acquire media -> validate (probe) -> copy into the project -> asset registry."""
        provider = self.registry.get(c.provider)
        root = project.root
        assert root is not None
        known = project.assets.known_hashes()
        label = self._label(project, scene_id)

        def work(ctx) -> PreparedMedia:
            if provider is None:
                raise AcquisitionError(f"The provider “{c.provider}” is not available.")
            ctx.report(5, f"Acquiring {c.title[:50]}")
            sctx = SearchContext(None, root / "cache" / "research", ctx.is_cancelled)
            try:
                file = provider.acquire(c, root / "cache" / "research" / "acquire", sctx)
            except ProviderError as exc:
                raise AcquisitionError(exc.user_message, details=exc.details) from exc
            ctx.report(50, "Validating and copying into the project")
            prepared = self._importer.prepare(root, file, known_hashes=known, link_mode=LINK_COPY,
                                              progress=lambda f_, m: ctx.report(50 + 50 * f_, m), should_cancel=ctx.is_cancelled)
            if file.parent.name == "acquire":
                file.unlink(missing_ok=True)
            return prepared

        def done(job: Job) -> None:
            if self._projects.current is not project:
                return
            extra = {"candidate_id": c.candidate_id, "scene_id": scene_id, "provider": c.provider, "title": c.title,
                     "license": {"name": c.license.name, "url": c.license.url, "attribution": c.license.attribution,
                                 "status": c.license.status, "verified": c.license.verified},
                     "evidence_kind": c.evidence_kind.value, "prompt": c.prompt or None}
            asset = self._media.register_prepared(project, job.result, name=(c.title or c.provider_id)[:80], source_type=c.source_type,
                                                  source_url=c.source_reference or None, extra=extra)
            if asset is None:
                return
            cur = project.visual_candidates.get(c.candidate_id)
            if cur is not None:
                upd = deepcopy(cur)
                upd.asset_id, upd.status = asset.id, CandidateStatus.ACQUIRED
                a = deepcopy(project.visual_assignments.get(scene_id))
                if a and a.candidate_id == c.candidate_id:
                    a.asset_id = asset.id
                kw = {"assignment": a} if a is not None else {}
                self._apply(SceneDecisionCommand(project, scene_id, "Acquire visual", candidates=[upd], **kw))
            log_event(_log, "research.acquired", candidate_id=c.candidate_id, asset_id=asset.id)

        def failed(job: Job) -> None:
            if self._projects.current is project:
                cur = project.visual_candidates.get(c.candidate_id)
                if cur is not None:
                    upd = deepcopy(cur)
                    upd.status = CandidateStatus.FAILED
                    upd.metadata["acquire_error"] = job.error or "Acquisition failed."
                    self._apply(SceneDecisionCommand(project, scene_id, "Acquisition failed", candidates=[upd]))
            self._bus.publish(Topics.ERROR, message=f"Scene {label}: the visual could not be added to the project. {job.error or ''}", title="Visual research")

        return self._jobs.submit("candidate_download", work, title=f"Adding visual to project: {c.title[:40]}", on_complete=done, on_error=failed)

    def approve(self, scene_id: str) -> Job | None:
        """Approve the scene's visual. With no explicit choice, the AI's best candidate is the one approved (selected_by AI)."""
        project = self._project()
        a = project.visual_assignments.get(scene_id)
        job = None
        if a is None:
            st = self.state(scene_id)
            if not st.best_id:
                raise ResearchError("There is no candidate to approve for this scene.")
            if st.status in (ResearchStatus.LOW_CONFIDENCE, ResearchStatus.ERROR):
                raise ResearchError("The best candidate is below your minimum score. Choose a visual yourself (Use) to approve it.")
            job = self.choose_candidate(scene_id, st.best_id, by="AI")
            a = project.visual_assignments[scene_id]
        a = deepcopy(a)
        if a.skipped is False and a.candidate_id is None:
            raise ResearchError("There is no candidate to approve for this scene.")
        a.approved = True
        st = deepcopy(self.state(scene_id))
        st.status, st.message = ResearchStatus.APPROVED, "Approved."
        self._commands.execute(SceneDecisionCommand(project, scene_id, f"Approve visual for scene {self._label(project, scene_id)}", state=st, assignment=a))
        return job

    def unapprove(self, scene_id: str) -> None:
        project = self._project()
        a = deepcopy(project.visual_assignments.get(scene_id))
        if a is None:
            return
        a.approved = False
        st = deepcopy(self.state(scene_id))
        st.status = ResearchStatus.CANDIDATES_READY if st.best_id else ResearchStatus.NOT_STARTED
        self._commands.execute(SceneDecisionCommand(project, scene_id, "Unapprove visual", state=st, assignment=a))

    def skip_visual(self, scene_id: str) -> None:
        project = self._project()
        a = VisualAssignment(scene_id, None, None, "USER", None, True, True, note="Visual skipped by the user.")
        st = deepcopy(self.state(scene_id))
        st.status, st.message = ResearchStatus.APPROVED, "Visual skipped."
        self._commands.execute(SceneDecisionCommand(project, scene_id, f"Skip visual for scene {self._label(project, scene_id)}", state=st, assignment=a))

    def reject_candidate(self, scene_id: str, candidate_id: str) -> None:
        """Reject one candidate; the rest are re-ranked from the stored pool (no new search)."""
        project = self._project()
        c = deepcopy(self._candidate(project, scene_id, candidate_id))
        c.status = CandidateStatus.REJECTED
        pool = [deepcopy(x) for x in project.visual_candidates.values() if x.scene_id == scene_id and x.candidate_id != candidate_id] + [c]
        brief = self.brief(scene_id)
        out = self.engine.evaluate_and_rank(brief, pool, deepcopy(project.visual_preferences), deepcopy(project.research_settings),
                                            self.history_for(project, scene_id))
        out.status, out.message = decide_status(out, brief, project.visual_preferences, True, False, final=False)
        prev = self.state(scene_id)
        state = self._state_from(scene_id, out, brief, prev.session_id, prev.generation, prev.expanded_sources)
        a = project.visual_assignments.get(scene_id)
        assignment: object = ...
        if a and a.candidate_id == candidate_id:
            assignment = None  # the chosen visual was rejected: the scene needs a new decision
        elif a and a.approved:
            state.status = ResearchStatus.APPROVED
        kw = {"assignment": assignment} if assignment is not ... else {}
        self._commands.execute(SceneDecisionCommand(project, scene_id, "Reject candidate", state=state, candidates=[c], **kw))

    def unreject_candidate(self, scene_id: str, candidate_id: str) -> None:
        project = self._project()
        c = deepcopy(self._candidate(project, scene_id, candidate_id))
        if c.status is CandidateStatus.REJECTED:
            c.status = CandidateStatus.ACQUIRED if c.asset_id else CandidateStatus.READY
        self._commands.execute(SceneDecisionCommand(project, scene_id, "Restore candidate", candidates=[c]))
        self.rescore(scene_id)

    def reject_best(self, scene_ids: list[str]) -> list[str]:
        """Bulk: reject each scene's current best candidate. Returns the scene ids that were changed."""
        done = []
        for sid in scene_ids:
            st = self.state(sid)
            if st.best_id and st.status is not ResearchStatus.APPROVED:
                self.reject_candidate(sid, st.best_id)
                done.append(sid)
        return done

    def approve_scenes(self, scene_ids: list[str], include_low_confidence: bool = False) -> tuple[list[str], list[str]]:
        """Bulk approve. Weak (LOW_CONFIDENCE) and empty scenes are skipped unless explicitly included."""
        approved, skipped = [], []
        for sid in scene_ids:
            st = self.state(sid)
            a = self._project().visual_assignments.get(sid)
            ok = (a is not None) or (st.best_id and (include_low_confidence or st.status in (ResearchStatus.CANDIDATES_READY, ResearchStatus.NEEDS_REVIEW)))
            if not ok or (a is None and st.status is ResearchStatus.APPROVED):
                skipped.append(sid)
                continue
            try:
                self.approve(sid)
                approved.append(sid)
            except AppError:
                skipped.append(sid)
        return approved, skipped

    # ------------------------------------------------------------------ manual / extra candidates
    def add_manual_visual(self, scene_id: str, path: Path) -> list[Job]:
        """Manual Select: import a file as the user's own visual for the scene (selected_by USER, no score)."""
        project = self._project()
        self._candidate_scene_check(project, scene_id)

        def got(asset) -> None:
            c = Candidate(self._next_id("candidate"), scene_id, S.USER_MEDIA, "VIDEO" if asset.type.value == "video" else "IMAGE",
                          title=asset.name, description="Chosen manually by the user.", duration=asset.duration, width=asset.width,
                          height=asset.height, provider="manual", provider_id=asset.id, source_reference=asset.path,
                          acquisition=Acquisition.LOCAL, status=CandidateStatus.ACQUIRED, asset_id=asset.id)
            a = VisualAssignment(scene_id, c.candidate_id, asset.id, "USER", None, False, c.evidence_kind, Acquisition.LOCAL,
                                 source_type=S.USER_MEDIA, note="Manual selection (not scored).")
            self._commands.execute(SceneDecisionCommand(project, scene_id, f"Manual visual for scene {self._label(project, scene_id)}",
                                                        assignment=a, candidates=[c]))

        return self._media.import_files([Path(path)], on_asset=got)

    def add_page_screenshot(self, scene_id: str, url: str) -> Job:
        """Capture a page the user points at and add it to the pool as an evidence candidate (needs verification)."""
        project = self._project()
        brief = self.brief(scene_id)
        if not url.lower().startswith(("http://", "https://")):
            raise ResearchError("Only http(s) pages can be captured.")
        prov: ScreenshotProvider = self.registry.get("screenshot")  # type: ignore[assignment]
        ok, why = prov.is_available()
        if not ok:
            raise ResearchError(why)
        root = project.root
        assert root is not None
        q = ResearchQuery(self._next_id("query"), scene_id, url, _qtype_document(), "Page chosen by the user", 1, [S.SCREENSHOT.value])

        def work(ctx):
            return prov.capture_url(url, q, SearchContext(brief, root / "cache" / "research", ctx.is_cancelled))

        def done(job: Job) -> None:
            if self._projects.current is project:
                c: Candidate = job.result
                c.candidate_id = self._next_id("candidate")
                c.fingerprint = fingerprint_file(Path(c.thumbnail_path), self._settings().ffmpeg_path)
                self._merge_candidates(project, scene_id, [c], f"Added screenshot of {url}")

        def failed(job: Job) -> None:
            self._bus.publish(Topics.ERROR, message=job.error or "The page could not be captured.", title="Screenshot")

        return self._jobs.submit("screenshot", work, title="Capturing page", on_complete=done, on_error=failed)

    def _merge_candidates(self, project: Project, scene_id: str, new: list[Candidate], note: str) -> None:
        brief = self.brief(scene_id)
        pool = [deepcopy(x) for x in project.visual_candidates.values() if x.scene_id == scene_id] + new
        out = self.engine.evaluate_and_rank(brief, pool, deepcopy(project.visual_preferences), deepcopy(project.research_settings),
                                            self.history_for(project, scene_id))
        prev = self.state(scene_id)
        out.status, out.message = decide_status(out, brief, project.visual_preferences, True, False, final=False)
        state = self._state_from(scene_id, out, brief, prev.session_id, prev.generation, prev.expanded_sources)
        a = project.visual_assignments.get(scene_id)
        if a and a.approved:
            state.status = ResearchStatus.APPROVED
        state.message = f"{note}. {state.message}"
        self._apply(ApplyResearchCommand(project, scene_id, [], None, out.candidates, out.scores, state))

    @staticmethod
    def _candidate_scene_check(project: Project, scene_id: str) -> None:
        if not any(s.id == scene_id for s in project.scenes):
            raise ResearchError("That scene no longer exists.")

    # ------------------------------------------------------------------ AI visual (only on explicit request)
    def generate_ai_visual(self, scene_id: str, candidate_id: str | None = None) -> Job:
        """AIImageGenerationJob: turn the AI *proposal* into a real image (costs money on the service side)."""
        project = self._project()
        brief = self.brief(scene_id)
        prov: AIImageProvider = self.registry.get("ai_image")  # type: ignore[assignment]
        ok, why = prov.is_available()
        if not ok:
            raise ResearchError(f"AI image generation is not available. {why}")
        cand = None
        if candidate_id:
            cand = deepcopy(self._candidate(project, scene_id, candidate_id))
        else:
            cand = next((deepcopy(c) for c in project.visual_candidates.values()
                         if c.scene_id == scene_id and c.source_type is S.AI_GENERATED and c.status is not CandidateStatus.REJECTED), None)
        root = project.root
        assert root is not None
        if cand is None:  # no proposal yet (e.g. AI images were disabled): build one from the brief
            q = ResearchQuery(self._next_id("query"), scene_id, "ai concept", _qtype_alt(), "AI concept from the research brief", 5, [S.AI_GENERATED.value])
            cand = prov.search(q, S.AI_GENERATED, 1, SearchContext(brief))[0]
            cand.candidate_id = self._next_id("candidate")
        ffmpeg = self._settings().ffmpeg_path

        def work(ctx) -> Candidate:
            ctx.report(10, "Generating image")
            file = prov.generate(cand, root / "generated")
            ctx.report(80, "Preparing preview")
            thumb = root / "cache" / "research" / "thumbs" / f"{cand.candidate_id}.jpg"
            local = self.registry.get("local_stock")
            if local is not None:
                tmp = cand.__class__.from_dict(cand.to_dict())
                tmp.local_path, tmp.kind = str(file), "IMAGE"
                local.fetch_thumbnail(tmp, thumb, None)  # reuse the ffmpeg thumbnailer
            cand.local_path = str(file)
            cand.thumbnail_path = str(thumb.relative_to(root)) if thumb.is_file() else ""
            cand.fingerprint = fingerprint_file(thumb, ffmpeg) if thumb.is_file() else ""
            cand.status = CandidateStatus.READY
            cand.width, cand.height = cand.width or 1536, cand.height or 1024
            return cand

        def done(job: Job) -> None:
            if self._projects.current is project:
                self._merge_candidates(project, scene_id, [job.result], "AI visual generated")

        def failed(job: Job) -> None:
            self._bus.publish(Topics.ERROR, message=f"The AI visual could not be generated. {job.error or ''}", title="AI visual")

        return self._jobs.submit("ai_image_generation", work, title="Generating AI visual", on_complete=done, on_error=failed)

    # ------------------------------------------------------------------ Phase 2 interface (synchronous, read-only: does not touch the project)
    def search_scene(self, scene, intent, prefs) -> SceneResearchResult:
        from app.ai.interfaces import VisualCandidate

        project = self._project()
        brief = self.brief(scene.id)
        queries = QueryGenerator(project.research_settings).generate(brief, id_factory=lambda: self._next_id("query"))
        root = project.root
        assert root is not None
        out = self.engine.run(brief, queries, prefs, project.research_settings, self.history_for(project, scene.id),
                              RunContext(root / "cache" / "research", root, ffmpeg_path=self._settings().ffmpeg_path), lambda: self._next_id("candidate"))
        res = SceneResearchResult(scene.id, notes=[out.message])
        for c in out.candidates:
            res.candidates.append(VisualCandidate(c.candidate_id, scene.id, c.source_type.value, c.source_reference or None, c.asset_id,
                                                  {"title": c.title, "score": out.scores[c.candidate_id].overall}))
        return res

    def search_candidates(self, scene, intent, source, limit: int = 10):
        project = self._project()
        prefs = deepcopy(project.visual_preferences)
        for k in SourceKind:
            prefs.setting(k).enabled = k is source
        return self.search_scene(scene, intent, prefs).candidates[:limit]

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _label(project: Project, scene_id: str) -> str:
        return next((s.label for s in project.scenes if s.id == scene_id), scene_id)

    def scenes_ready_for_research(self) -> list[str]:
        """Scenes that have been analysed (and are not yet researched)."""
        project = self._project()
        return [s.id for s in project.scenes
                if s.status not in (SceneStatus.PENDING, SceneStatus.FAILED) and s.id in project.visual_intents
                and project.research_status.get(s.id, SceneResearchState(s.id)).status is ResearchStatus.NOT_STARTED]


_TYPE = {SourceKind.YOUTUBE: S.YOUTUBE, SourceKind.STOCK_IMAGES: S.STOCK_IMAGE, SourceKind.STOCK_VIDEOS: S.STOCK_VIDEO,
         SourceKind.AI_IMAGES: S.AI_GENERATED, SourceKind.WEB_IMAGES: S.WEB_IMAGE, SourceKind.WEB_VIDEOS: S.WEB_VIDEO,
         SourceKind.SCREENSHOTS: S.SCREENSHOT}


def TYPE_TO_KIND(source_type_value: str) -> str:
    st = S(source_type_value)
    return next(k.value for k, v in _TYPE.items() if v is st)


def _qtype_document():
    from app.research.models import QueryType

    return QueryType.DOCUMENT


def _qtype_alt():
    from app.research.models import QueryType

    return QueryType.ALTERNATIVE
