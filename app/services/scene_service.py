"""Scene pipeline (sentence analysis -> segmentation -> enrichment) and manual scene editing.

The pipeline runs in a background job. It never mutates the project: results come back as a
single undoable command on the UI thread. If enrichment fails on scene N, scenes 1..N-1 are kept
and a retry resumes at N without redoing earlier work.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from app.analysis.analyzer import EnrichContext, EnrichedScene, RuleBasedAnalyzer, SemanticAnalyzer, new_context
from app.analysis.intent import suggest_visuals
from app.analysis.models import (
    Origin,
    Scene,
    SceneAnalysisState,
    SceneStatus,
    SentenceAnalysis,
    VisualIntent,
    VisualType,
)
from app.analysis.segmenter import SceneDraft, SegmentationParams
from app.core.commands import Command, CommandStack
from app.core.events import EventBus, Topics
from app.core.exceptions import AnalysisError, JobCancelled, SceneEditError, UserEditsPresentError
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.phase2_commands import ReplaceScenesCommand, SetScenesCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.transcription.alignment import script_hash
from app.transcription.models import Transcript
from app.transcription.status import TranscriptStatus, transcript_status

_log = get_logger(__name__)
MIN_SPLIT_MARGIN = 0.2  # seconds a split point must keep from either scene edge
USER_FIELDS = ("topic", "summary", "notes")


class SceneState(str, Enum):
    NONE = "NONE"
    UP_TO_DATE = "UP_TO_DATE"
    PARTIAL = "PARTIAL"  # stopped on a failure; retry resumes
    OUTDATED = "OUTDATED"  # transcript, script or analyzer changed since the scenes were generated


@dataclass
class _Partial:
    enriched: list[EnrichedScene]
    drafts: list[SceneDraft]
    ids: list[str]
    labels: list[str]
    failed_index: int
    error: str
    analyses: dict[str, SentenceAnalysis]
    overall_topic: str
    params: SegmentationParams


@dataclass
class _Output:
    enriched: list[EnrichedScene]
    analyses: dict[str, SentenceAnalysis]
    overall_topic: str
    params: SegmentationParams
    ids: list[str] = field(default_factory=list)


def sentence_ids_for(tr: Transcript, start: float, end: float) -> list[str]:
    wm = tr.word_map()
    return [s.sentence_id for s in tr.sentences if any(start <= (wm[w].start + wm[w].end) / 2 < end for w in s.word_ids)]


def _pending_scene(tr: Transcript, d: SceneDraft, scene_id: str, label: str, status: SceneStatus) -> Scene:
    words = tr.words_between(d.start, d.end)
    return Scene(id=scene_id, label=label, start=d.start, end=d.end, narration=" ".join(w.text for w in words),
                 sentence_ids=list(d.sentence_ids), segmentation_confidence=d.boundary_conf, status=status,
                 rationale=list(d.rationale))


def _draft_of(scene: Scene) -> SceneDraft:
    return SceneDraft(scene.start, scene.end, list(scene.sentence_ids), list(scene.rationale), scene.segmentation_confidence)


class SceneService:
    def __init__(self, projects: ProjectManager, commands: CommandStack, jobs: JobManager, bus: EventBus,
                 analyzer: SemanticAnalyzer | None = None) -> None:
        self._projects, self._commands, self._jobs, self._bus = projects, commands, jobs, bus
        self.analyzer: SemanticAnalyzer = analyzer or RuleBasedAnalyzer()

    # ------------------------------------------------------------ helpers
    def _project(self) -> Project:
        if self._projects.current is None:
            raise AnalysisError("Open or create a project first.")
        return self._projects.current

    @staticmethod
    def _surfaces(project: Project) -> dict[str, str]:
        return project.script_alignment.surface_by_word() if project.script_alignment else {}

    def sentence_key(self, project: Project) -> str:
        tr = project.transcription.transcript
        raw = f"{tr.transcript_id if tr else ''}|{script_hash(project.script.text)}|{self.analyzer.name}|{self.analyzer.version}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def input_hash(self, project: Project, params: SegmentationParams) -> str:
        raw = self.sentence_key(project) + json.dumps(asdict(params), sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def state(self) -> SceneState:
        project = self._project()
        st = project.scene_analysis
        if not project.scenes:
            return SceneState.NONE
        if st.status == "PARTIAL":
            return SceneState.PARTIAL
        params = SegmentationParams(**st.params) if st.params else SegmentationParams()
        if transcript_status(project) is not TranscriptStatus.COMPLETE or st.input_hash != self.input_hash(project, params):
            return SceneState.OUTDATED
        return SceneState.UP_TO_DATE

    def user_touched_labels(self) -> list[str]:
        project = self._project()
        return [s.label for s in project.scenes
                if s.is_user_touched or s.id in project.visual_assignments]  # an approved/selected visual is a user decision too

    # ------------------------------------------------------------ pipeline
    def analyze(self, force: bool = False, overwrite_user_edits: bool = False,
                params: SegmentationParams | None = None) -> Job | None:
        project = self._project()
        tr = project.transcription.transcript
        status = transcript_status(project)
        if tr is None:
            raise AnalysisError("Transcribe the voice-over first.")
        if status is not TranscriptStatus.COMPLETE:
            raise AnalysisError("The transcript is outdated because the voice-over changed. Re-transcribe before analysing scenes.")
        touched = self.user_touched_labels()
        if touched and not overwrite_user_edits:
            raise UserEditsPresentError(
                "Some scenes were edited, approved or split by you. Regenerating would replace them.", touched)
        params = params or (SegmentationParams(**project.scene_analysis.params) if project.scene_analysis.params else SegmentationParams())
        key = self.input_hash(project, params)
        st = project.scene_analysis
        if not force and project.scenes and st.status == "COMPLETE" and st.input_hash == key:
            self._bus.publish(Topics.STATUS, message="Scenes are already up to date for this transcript and script.")
            return None
        reuse = st.sentence_analysis if st.sentence_key == self.sentence_key(project) and st.sentence_analysis else None
        prior = dict(reuse) if reuse else None
        surfaces = dict(self._surfaces(project))
        base = st.scene_counter
        analyzer = self.analyzer
        box: list[_Partial] = []

        def work(ctx) -> _Output:
            analyses = prior or analyzer.analyze_sentences(
                tr, surfaces, lambda f, m: ctx.report(35 * f, m), ctx.is_cancelled)
            ctx.report(36, "Finding scene boundaries")
            overall = analyzer.overall_topic(tr, analyses)
            drafts = analyzer.segment(tr, analyses, params)
            if not drafts:
                raise AnalysisError("No scenes could be created from this transcript.")
            ids = [f"scene_{base + k + 1:03d}" for k in range(len(drafts))]
            labels = [str(k + 1) for k in range(len(drafts))]
            ctx.report(45, f"Analysing {len(drafts)} scenes")
            ectx = new_context(analyzer, tr, analyses, surfaces, params, overall)
            done = self._enrich(drafts, ids, labels, 0, None, ectx, ctx, box, analyses, overall, params)
            return _Output(done, analyses, overall, params, ids)

        def completed(job: Job) -> None:
            if self._projects.current is not project or project.transcription.transcript is not tr:
                return
            out: _Output = job.result
            self._install(project, [e.scene for e in out.enriched], {e.scene.id: e.intent for e in out.enriched},
                          out, "COMPLETE", key, None, None, "Generate scenes")

        def failed(job: Job) -> None:
            if self._projects.current is not project or not box:
                self._bus.publish(Topics.ERROR, message=job.error or "Scene analysis failed.", title="Scene analysis")
                return
            p = box[0]
            self._install_partial(project, tr, p, key)

        self._bus.publish("scenes.started")
        return self._jobs.submit("scene_analysis", work, title="Analysing scenes", on_complete=completed, on_error=failed)

    def _enrich(self, drafts, ids, labels, start_index, prev, ectx: EnrichContext, jctx, box, analyses, overall, params,
                existing: list[EnrichedScene] | None = None) -> list[EnrichedScene]:
        """Enrich ``drafts[start_index:]`` one by one. On failure stash a ``_Partial`` and raise."""
        done = list(existing or [])
        total = len(drafts)
        for k in range(start_index, total):
            jctx.check_cancelled()
            ectx.index, ectx.total, ectx.prev = k, total, prev
            try:
                prev = self.analyzer.enrich_scene(drafts[k], ids[k], labels[k], ectx)
            except JobCancelled:
                raise
            except Exception as exc:
                msg = f"Scenes 1–{k} complete. Scene {k + 1} failed." if k else "Scene 1 failed."
                _log.exception("Scene enrichment failed", extra={"scene_index": k})
                box.append(_Partial(done, drafts, ids, labels, k, f"{type(exc).__name__}: {exc}", analyses, overall, params))
                raise AnalysisError(msg, details=f"{type(exc).__name__}: {exc}") from exc
            done.append(prev)
            jctx.report(45 + 55 * (k + 1) / total, f"Analysed scene {k + 1} of {total}")
        return done

    def _install(self, project: Project, scenes: list[Scene], intents: dict[str, VisualIntent], out: _Output, status: str,
                 key: str, failed_id: str | None, failed_error: str | None, description: str) -> None:
        old = project.scene_analysis
        state = SceneAnalysisState(
            status=status, input_hash=key, transcript_id=project.transcription.transcript.transcript_id if project.transcription.transcript else "",
            analyzer=self.analyzer.name, analyzer_version=self.analyzer.version, overall_topic=out.overall_topic,
            sentence_analysis=out.analyses, scene_counter=max(old.scene_counter, *(int(s.id.rsplit("_", 1)[-1]) for s in scenes)) if scenes else old.scene_counter,
            params=asdict(out.params), failed_scene_id=failed_id, failed_error=failed_error,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), sentence_key=self.sentence_key(project),
        )
        self._commands.execute(SetScenesCommand(project, scenes, intents, state, description))
        log_event(_log, "scenes.installed", count=len(scenes), status=status)

    def _install_partial(self, project: Project, tr: Transcript, p: _Partial, key: str) -> None:
        scenes = [e.scene for e in p.enriched]
        intents = {e.scene.id: e.intent for e in p.enriched}
        for k in range(p.failed_index, len(p.drafts)):
            scenes.append(_pending_scene(tr, p.drafts[k], p.ids[k], p.labels[k],
                                         SceneStatus.FAILED if k == p.failed_index else SceneStatus.PENDING))
        out = _Output(p.enriched, p.analyses, p.overall_topic, p.params)
        self._install(project, scenes, intents, out, "PARTIAL", key, p.ids[p.failed_index], p.error, "Generate scenes (partial)")

    def retry_failed(self) -> Job:
        """Resume at the first unfinished scene; completed scenes are kept as they are."""
        project = self._project()
        tr = project.transcription.transcript
        st = project.scene_analysis
        pending = [i for i, s in enumerate(project.scenes) if s.status in (SceneStatus.PENDING, SceneStatus.FAILED)]
        if st.status != "PARTIAL" or not pending or tr is None:
            raise AnalysisError("There is no interrupted scene analysis to resume.")
        first = pending[0]
        params = SegmentationParams(**st.params) if st.params else SegmentationParams()
        scenes = deepcopy(project.scenes)
        drafts = [_draft_of(s) for s in scenes]
        ids, labels = [s.id for s in scenes], [s.label for s in scenes]
        existing = [EnrichedScene(deepcopy(s), deepcopy(project.visual_intents[s.id])) for s in scenes[:first]]
        analyses, overall, surfaces, key = dict(st.sentence_analysis), st.overall_topic, dict(self._surfaces(project)), st.input_hash
        box: list[_Partial] = []

        def work(ctx) -> _Output:
            ectx = new_context(self.analyzer, tr, analyses, surfaces, params, overall)
            ctx.report(5, f"Resuming at scene {first + 1}")
            done = self._enrich(drafts, ids, labels, first, existing[-1] if existing else None, ectx, ctx, box, analyses, overall, params, existing)
            return _Output(done, analyses, overall, params)

        def completed(job: Job) -> None:
            if self._projects.current is project:
                out: _Output = job.result
                self._install(project, [e.scene for e in out.enriched], {e.scene.id: e.intent for e in out.enriched},
                              out, "COMPLETE", key, None, None, "Resume scene analysis")

        def failed(job: Job) -> None:
            if self._projects.current is project and box:
                self._install_partial(project, tr, box[0], key)
            else:
                self._bus.publish(Topics.ERROR, message=job.error or "Scene analysis failed.", title="Scene analysis")

        return self._jobs.submit("scene_analysis", work, title=f"Resuming scene analysis at scene {first + 1}",
                                 on_complete=completed, on_error=failed)

    # ------------------------------------------------------------ manual editing
    def _scene_index(self, project: Project, scene_id: str) -> int:
        for i, s in enumerate(project.scenes):
            if s.id == scene_id:
                return i
        raise SceneEditError("That scene no longer exists.")

    def _edit_context(self, project: Project) -> tuple[Transcript, EnrichContext]:
        tr = project.transcription.transcript
        st = project.scene_analysis
        if tr is None or not st.sentence_analysis:
            raise SceneEditError("Run scene analysis before editing scenes.")
        params = SegmentationParams(**st.params) if st.params else SegmentationParams()
        return tr, new_context(self.analyzer, tr, st.sentence_analysis, self._surfaces(project), params, st.overall_topic)

    def _prev_enriched(self, project: Project, index: int) -> EnrichedScene | None:
        if index <= 0:
            return None
        prev = project.scenes[index - 1]
        intent = project.visual_intents.get(prev.id)
        return EnrichedScene(prev, intent) if intent else None

    def _next_id(self, project: Project) -> str:
        project.scene_analysis.scene_counter += 1
        return f"scene_{project.scene_analysis.scene_counter:03d}"

    @staticmethod
    def _child_labels(label: str) -> tuple[str, str]:
        return (f"{label}A", f"{label}B") if label[-1:].isdigit() else (f"{label}1", f"{label}2")

    @staticmethod
    def _carry_user_fields(parents: list[Scene], child: Scene) -> None:
        for f in USER_FIELDS:
            for p in parents:
                if f in p.user_edited_fields and getattr(p, f):
                    setattr(child, f, getattr(p, f))
                    if f not in child.user_edited_fields:
                        child.user_edited_fields.append(f)
                    break

    @staticmethod
    def _user_intent(project: Project, parents: list[Scene]) -> VisualIntent | None:
        cands = [(p, project.visual_intents[p.id]) for p in parents
                 if p.id in project.visual_intents and project.visual_intents[p.id].author is Origin.USER]
        return deepcopy(max(cands, key=lambda c: c[0].duration)[1]) if cands else None

    def split_scene(self, scene_id: str, at: float) -> tuple[Scene, Scene]:
        """Split a scene at ``at`` seconds. Both halves keep the user's edits and are re-analysed from their own words."""
        project = self._project()
        idx = self._scene_index(project, scene_id)
        scene = project.scenes[idx]
        if not (scene.start + MIN_SPLIT_MARGIN <= at <= scene.end - MIN_SPLIT_MARGIN):
            raise SceneEditError(
                f"Choose a split time inside the scene, at least {MIN_SPLIT_MARGIN:g}s from either edge "
                f"({scene.start:.3f}–{scene.end:.3f}).")
        tr, ectx = self._edit_context(project)
        la, lb = self._child_labels(scene.label)
        why = [f"Split by you at {at:.3f}s"]
        drafts = [SceneDraft(scene.start, at, sentence_ids_for(tr, scene.start, at), list(why), 1.0, "manual"),
                  SceneDraft(at, scene.end, sentence_ids_for(tr, at, scene.end), list(why), 1.0, "manual")]
        ids = [scene.id, self._next_id(project)]
        ectx.total = len(project.scenes) + 1
        parent_intent = self._user_intent(project, [scene])
        out: list[EnrichedScene] = []
        prev = self._prev_enriched(project, idx)
        for k, (d, sid, lab) in enumerate(zip(drafts, ids, (la, lb))):
            ectx.index, ectx.prev = idx + k, prev
            e = self.analyzer.enrich_scene(d, sid, lab, ectx)
            e.scene.origin, e.scene.boundaries_locked = Origin.USER, True
            self._carry_user_fields([scene], e.scene)
            if parent_intent:
                e.intent = deepcopy(parent_intent)
                e.intent.scene_id = sid
            out.append(e)
            prev = e
        self._commands.execute(ReplaceScenesCommand(
            project, [scene.id], [e.scene for e in out], {e.scene.id: e.intent for e in out}, f"Split scene {scene.label}"))
        return out[0].scene, out[1].scene

    def merge_scenes(self, first_id: str, second_id: str) -> Scene:
        """Merge two adjacent scenes. Narration and timing are rebuilt from the untouched transcript words."""
        project = self._project()
        i, j = self._scene_index(project, first_id), self._scene_index(project, second_id)
        if j != i + 1:
            raise SceneEditError("Only neighbouring scenes can be merged.")
        a, b = project.scenes[i], project.scenes[j]
        tr, ectx = self._edit_context(project)
        draft = SceneDraft(a.start, b.end, sentence_ids_for(tr, a.start, b.end),
                           [f"Merged by you: scenes {a.label} + {b.label}"], 1.0, "manual")
        ectx.total, ectx.index, ectx.prev = len(project.scenes) - 1, i, self._prev_enriched(project, i)
        e = self.analyzer.enrich_scene(draft, a.id, a.label, ectx)
        e.scene.origin, e.scene.boundaries_locked = Origin.USER, True
        self._carry_user_fields([a, b], e.scene)
        user_intent = self._user_intent(project, [a, b])
        if user_intent:
            e.intent = user_intent
            e.intent.scene_id = a.id
        self._commands.execute(ReplaceScenesCommand(
            project, [a.id, b.id], [e.scene], {e.scene.id: e.intent}, f"Merge scenes {a.label} + {b.label}"))
        return e.scene

    def merge_with_previous(self, scene_id: str) -> Scene:
        project = self._project()
        i = self._scene_index(project, scene_id)
        if i == 0:
            raise SceneEditError("This is the first scene; there is nothing before it to merge with.")
        return self.merge_scenes(project.scenes[i - 1].id, scene_id)

    def merge_with_next(self, scene_id: str) -> Scene:
        project = self._project()
        i = self._scene_index(project, scene_id)
        if i >= len(project.scenes) - 1:
            raise SceneEditError("This is the last scene; there is nothing after it to merge with.")
        return self.merge_scenes(scene_id, project.scenes[i + 1].id)

    def edit_scene(self, scene_id: str, *, topic: str | None = None, summary: str | None = None, notes: str | None = None,
                   intent: dict | None = None) -> Scene:
        """Record a user edit. Edited fields are remembered so the AI never overwrites them silently."""
        project = self._project()
        scene = deepcopy(project.scenes[self._scene_index(project, scene_id)])
        for name, value in (("topic", topic), ("summary", summary), ("notes", notes)):
            if value is not None and value != getattr(scene, name):
                setattr(scene, name, value)
                if name not in scene.user_edited_fields:
                    scene.user_edited_fields.append(name)
        vi = deepcopy(project.visual_intents.get(scene_id))
        if intent and vi is not None:
            for k, v in intent.items():
                if k == "type":
                    v = VisualType(v)
                if not hasattr(vi, k):
                    raise SceneEditError(f"Unknown visual-intent field “{k}”.")
                setattr(vi, k, v)
            if "type" in intent and "preferred_visuals" not in intent:
                vi.preferred_visuals = suggest_visuals(vi.type, vi.primary_subject, vi.secondary_subject, vi.action, scene.topic)
            vi.author = Origin.USER
            if "visual_intent" not in scene.user_edited_fields:
                scene.user_edited_fields.append("visual_intent")
        self._commands.execute(ReplaceScenesCommand(
            project, [scene_id], [scene], {scene_id: vi} if vi else {}, f"Edit scene {scene.label}"))
        return scene

    def set_approved(self, scene_id: str, approved: bool = True) -> Scene:
        project = self._project()
        scene = deepcopy(project.scenes[self._scene_index(project, scene_id)])
        if scene.status in (SceneStatus.PENDING, SceneStatus.FAILED):
            raise SceneEditError("This scene has not been analysed yet.")
        scene.status = SceneStatus.APPROVED if approved else (
            SceneStatus.READY if scene.segmentation_confidence >= 0.6 else SceneStatus.NEEDS_REVIEW)
        intent = project.visual_intents.get(scene_id)
        self._commands.execute(ReplaceScenesCommand(
            project, [scene_id], [scene], {scene_id: deepcopy(intent)} if intent else {},
            f"{'Approve' if approved else 'Unapprove'} scene {scene.label}"))
        return scene
