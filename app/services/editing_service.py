"""Application service for the AI Advanced Editing Engine (Phase 4).

    UI -> EditingService -> EditingStrategyService / TimelineAssemblyService -> Timeline -> Project

The AI creates the edit; the user owns it. Everything the AI produces becomes the real project timeline plus structured
``EditingDecision`` records. Nothing is rendered or flattened here.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.core.commands import Command, CommandStack
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, EditingError
from app.editing.assembly import EditState, TimelineAssemblyService, update_decision
from app.editing.compose import PreviewComposer
from app.editing.context import EditingContext, build_context, visual_status
from app.editing.models import (
    Creator,
    DecisionType,
    EditingDecision,
    EditingSession,
    EditingSettings,
    SceneEditStatus,
    SceneGeneration,
    ScenePlan,
    VideoProfile,
    VisualStatus,
    now_iso,
)
from app.editing.presets import PRESETS
from app.editing.strategy import EditingStrategyService
from app.editing.validator import TimelineValidator, ValidationIssue
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.project.phase3_commands import SceneDecisionCommand
from app.project.phase4_commands import ApplyEditCommand, MarkUserEditCommand, RecordUserDeleteCommand, SetEditingSettingsCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.research.models import Acquisition, VisualAssignment
from app.storage.atomic import atomic_write_text

_log = get_logger(__name__)
Apply = Callable[[Command], None]
LOCK_ASPECTS = ("VISUAL", "TIMING", "TEXT", "MOTION", "SCENE")
OPERATIONS = ("Analyzing narration...", "Determining shot timing...", "Creating motion...", "Building timeline...")


@dataclass
class GenerationOutcome:
    plans: list[ScenePlan] = field(default_factory=list)
    profile: VideoProfile = field(default_factory=VideoProfile)
    failure: tuple[str, str] | None = None  # (scene id, message)
    note: str = ""


class EditingService:
    def __init__(self, projects: ProjectManager, commands: CommandStack, jobs: JobManager, bus: EventBus, apply: Apply,
                 checkpoint_dir: Callable[[str], Path]) -> None:
        self._projects, self._commands, self._jobs, self._bus, self._apply = projects, commands, jobs, bus, apply
        self._checkpoint_dir = checkpoint_dir
        self.strategy = EditingStrategyService()
        self._running = False
        self.progress: dict = {"scene": 0, "total": 0, "operation": "", "scene_id": "", "state": "IDLE"}

    # ------------------------------------------------------------------ basics
    def _project(self) -> Project:
        if self._projects.current is None:
            raise EditingError("Open or create a project first.")
        return self._projects.current

    @property
    def running(self) -> bool:
        return self._running

    def presets(self) -> dict:
        return PRESETS

    def settings(self) -> EditingSettings:
        return deepcopy(self._project().editing_settings)

    def update_settings(self, **changes) -> EditingSettings:
        p = self._project()
        new = replace(p.editing_settings, **changes)
        if new.style not in PRESETS:
            raise EditingError(f"Unknown editing style “{new.style}”.")
        for k in ("pacing", "motion_intensity", "transition_frequency"):
            setattr(new, k, min(1.0, max(0.0, float(getattr(new, k)))))
        # what the user changes on purpose is remembered: an applied reference style never overrides it (see editing/effective.py)
        changed = [k for k in changes if k not in ("user_set", "reference") and getattr(p.editing_settings, k) != getattr(new, k)]
        if changed:
            new.user_set = list(dict.fromkeys([*p.editing_settings.user_set, *changed]))
        self._commands.execute(SetEditingSettingsCommand(p, new))
        return new

    def context(self) -> EditingContext:
        return build_context(self._project())

    def visual_status(self, scene_id: str) -> dict:
        p = self._project()
        status, a = visual_status(p, scene_id, p.editing_strategy.scene_visuals.get(scene_id))
        return {"status": status, "asset_id": a.asset_id if a else None, "accuracy": a.accuracy_score if a else None, "selected_by": a.selected_by if a else ""}

    def scene_rows(self) -> list[dict]:
        p = self._project()
        ctx = build_context(p)
        rows = []
        for sc in ctx.scenes:
            st = p.timeline_generation.scenes.get(sc.scene.id)
            status = st.status if st else SceneEditStatus.PENDING
            if sc.scene.id in p.timeline_generation.locked_scenes:
                status = SceneEditStatus.LOCKED
            elif st and st.status in (SceneEditStatus.COMPLETE, SceneEditStatus.NEEDS_VISUAL) and st.input_hash != sc.input_hash:
                status = SceneEditStatus.OUTDATED
            decs = [d for d in p.editing_decisions.values() if d.scene_id == sc.scene.id]
            rows.append({"scene_id": sc.scene.id, "label": sc.scene.label, "start": sc.scene.start, "end": sc.scene.end,
                         "visual_status": sc.visual_status, "status": status, "error": st.error if st else "", "decisions": len(decs),
                         "min_confidence": min((d.confidence for d in decs), default=None),
                         "locked": sc.scene.id in p.timeline_generation.locked_scenes,
                         "user_owned": any(d.created_by is Creator.USER for d in decs)})
        return rows

    def outdated_scenes(self) -> list[str]:
        return [r["scene_id"] for r in self.scene_rows() if r["status"] is SceneEditStatus.OUTDATED]

    # ------------------------------------------------------------------ generation (background, per-scene, incremental)
    def generate(self, scene_ids: list[str] | None = None, force: bool = False, scope: str | None = None) -> Job | None:
        project = self._project()
        if self._running:
            raise EditingError("An AI edit is already being created. Wait for it to finish or cancel it.")
        if not project.scenes:
            raise EditingError("There are no scenes yet. Analyse the voice-over into scenes first (Scenes page).")
        ctx = build_context(project)
        order = [s.id for s in project.scenes]
        wanted = [i for i in order if scene_ids is None or i in scene_ids]
        if not wanted:
            raise EditingError("None of the requested scenes exist.")
        gen = project.timeline_generation
        todo: list[str] = []
        skipped: list[str] = []
        for sid in wanted:
            st = gen.scenes.get(sid)
            if sid in gen.locked_scenes:
                skipped.append(sid)
            elif not force and st and st.status in (SceneEditStatus.COMPLETE, SceneEditStatus.NEEDS_VISUAL) and st.input_hash == ctx.by_id(sid).input_hash:
                skipped.append(sid)  # unchanged: never re-analysed
            else:
                todo.append(sid)
        scope = scope or ("ALL" if scene_ids is None else "SCENE" if len(wanted) == 1 else "SELECTED")
        session = EditingSession(f"edit_{len(project.editing_sessions) + 1:04d}", scope, list(todo), "QUEUED", skipped=list(skipped))
        project.editing_sessions.append(session)
        if not todo:
            session.status, session.finished_at = "COMPLETED", now_iso()
            session.log.append("Nothing to do: every requested scene is up to date or locked.")
            self._bus.publish(Topics.STATUS, message="The AI edit is already up to date.")
            return None
        provider, note = self.strategy.resolve(ctx.settings.provider)
        ctx.neighbour_transitions = {s.scene.id for s in ctx.scenes if s.scene.id not in todo and any(
            d.scene_id == s.scene.id and d.type is DecisionType.TRANSITION and d.parameters.get("type") != "CUT" for d in project.editing_decisions.values())}
        self._running = True
        self.progress = {"scene": 0, "total": len(todo), "operation": "Starting", "scene_id": "", "state": "RUNNING"}
        session.status = "RUNNING"
        if note:
            session.log.append(note)
        session.log.append(f"Editing {len(todo)} scene(s) in style “{ctx.settings.style}”; {len(skipped)} unchanged/locked scene(s) skipped.")

        def work(jc) -> GenerationOutcome:
            out = GenerationOutcome(note=note)
            profile = provider.analyze_video(ctx)
            out.profile = profile
            for n, sid in enumerate(todo, start=1):
                if jc.is_cancelled():
                    break
                label = ctx.by_id(sid).scene.label
                try:
                    for k, op in enumerate(OPERATIONS[:3]):
                        self.progress.update(scene=n, operation=op, scene_id=sid, state="RUNNING")
                        jc.report(100.0 * ((n - 1) + k / 4) / len(todo), f"Scene {n} / {len(todo)} — {op}")
                    plan = self.strategy.plan_scene(provider, ctx.by_id(sid), ctx, profile)
                    out.plans.append(plan)
                    session.log.append(f"Scene {label}: planned ({len(plan.segments)} visual(s), {len(plan.texts)} text).")
                except Exception as exc:  # stop at the first failure: earlier scenes are kept, later ones stay pending
                    _log.exception("Editing scene %s failed", sid)
                    out.failure = (sid, getattr(exc, "user_message", None) or str(exc) or type(exc).__name__)
                    session.log.append(f"Scene {label}: FAILED — {out.failure[1]}")
                    break
            return out

        def done(job: Job) -> None:
            self._running = False
            if self._projects.current is not project:
                return
            self._finish(project, ctx, session, todo, job.result)

        def failed(job: Job) -> None:
            self._running = False
            session.status, session.error, session.finished_at = "FAILED", job.error or "The AI edit failed.", now_iso()
            self.progress["state"] = "FAILED"
            self._bus.publish(Topics.ERROR, message=session.error, title="AI edit")

        def cancelled(job: Job) -> None:
            self._running = False
            session.status, session.finished_at = "CANCELED", now_iso()
            session.log.append("Canceled: nothing was changed.")
            self.progress["state"] = "CANCELED"
            self._bus.publish(Topics.STATUS, message="AI edit canceled — the timeline was not changed.")

        log_event(_log, "editing.started", scenes=len(todo), style=ctx.settings.style)
        return self._jobs.submit("ai_edit", work, title=f"AI edit ({len(todo)} scene{'s' if len(todo) != 1 else ''})", on_complete=done,
                                 on_error=failed, on_cancel=cancelled)

    def regenerate_scene(self, scene_id: str) -> Job | None:
        return self.generate([scene_id], force=True, scope="SCENE")

    def regenerate_scenes(self, scene_ids: list[str]) -> Job | None:
        return self.generate(scene_ids, force=True, scope="SELECTED")

    def regenerate_all(self) -> Job | None:
        return self.generate(None, force=True, scope="ALL")

    def retry_failed(self) -> Job | None:
        """Resume from the first failed scene: completed scenes are not touched."""
        p = self._project()
        ids = [s.id for s in p.scenes if (p.timeline_generation.scenes.get(s.id) or SceneGeneration(s.id)).status in (SceneEditStatus.FAILED, SceneEditStatus.PENDING)]
        if not ids:
            raise EditingError("There are no failed or pending scenes to retry.")
        return self.generate(ids, force=True, scope="RETRY")

    def cancel(self) -> None:
        for j in self._jobs.active_jobs():
            if j.type == "ai_edit":
                self._jobs.cancel(j.id)

    # ------------------------------------------------------------------ commit (UI thread)
    def _finish(self, project: Project, ctx: EditingContext, session: EditingSession, todo: list[str], out: GenerationOutcome) -> None:
        try:
            plans = out.plans
            asm = TimelineAssemblyService(project, ctx)
            if plans:
                self.strategy.limit_transitions(plans, ctx)
                session.checkpoint = self._checkpoint(project, "before_ai_edit")
                state = asm.assemble(plans, out.profile, session.session_id)
            else:
                state = asm.state
            gen = state.generation
            done_ids = {p.scene_id for p in plans}
            failed_at = todo.index(out.failure[0]) if out.failure and out.failure[0] in todo else None
            for i, sid in enumerate(todo):
                if sid in done_ids:
                    continue
                prev = gen.scenes.get(sid) or SceneGeneration(sid)
                if failed_at is not None and i == failed_at:
                    gen.scenes[sid] = SceneGeneration(sid, SceneEditStatus.FAILED, out.failure[1], prev.input_hash, prev.visual_status, prev.attempts + 1)  # type: ignore[index]
                elif failed_at is not None and i > failed_at:
                    gen.scenes[sid] = SceneGeneration(sid, SceneEditStatus.PENDING, "", prev.input_hash, prev.visual_status, prev.attempts)
            gen.status = "FAILED" if out.failure and not plans else "PARTIAL" if out.failure else "COMPLETE"
            if session.scope == "ALL":
                self._drop_orphans(state, {s.id for s in project.scenes})
            issues = self.validate_state(project, state)
            errors = [i for i in issues if i.severity == "error"]
            session.completed = [p.scene_id for p in plans]
            if errors:
                session.status, session.validation_errors, session.finished_at = "FAILED", [str(e) for e in errors[:20]], now_iso()
                session.error = "The generated edit did not pass validation, so the current timeline was kept."
                self.progress["state"] = "FAILED"
                self._bus.publish(Topics.ERROR, message=session.error, title="AI edit", details="; ".join(session.validation_errors[:5]))
                return
            if plans:
                self._commands.execute(ApplyEditCommand(project, state, f"AI edit ({len(plans)} scene{'s' if len(plans) != 1 else ''})"))
            else:  # only failure bookkeeping
                self._apply(ApplyEditCommand(project, state, "AI edit status"))
            session.failed_scene = out.failure[0] if out.failure else ""
            session.error = out.failure[1] if out.failure else ""
            session.status = "FAILED" if out.failure else "COMPLETED"
            session.finished_at = now_iso()
            session.log.append(f"Committed {len(plans)} scene(s) as one undo step." if plans else "Nothing was committed.")
            self.progress["state"] = session.status
            self._bus.publish(Topics.STATUS, message=(f"AI edit stopped at a failed scene; {len(plans)} scene(s) were edited." if out.failure
                                                      else f"AI edit complete: {len(plans)} scene(s)."))
        except Exception as exc:  # nothing was applied: the assembler works on copies
            _log.exception("AI edit commit failed")
            session.status, session.error, session.finished_at = "FAILED", getattr(exc, "user_message", None) or str(exc), now_iso()
            self.progress["state"] = "FAILED"
            self._bus.publish(Topics.ERROR, message=f"The AI edit could not be applied; the timeline was not changed. {session.error}", title="AI edit")

    @staticmethod
    def _drop_orphans(state: EditState, valid: set[str]) -> None:
        for t in state.timeline.tracks:
            t.clips = [c for c in t.clips if not (c.scene_id and c.scene_id not in valid and c.created_by == "AI" and not c.locked and c.metadata.get("phase") != 5)]
        for did in [k for k, d in state.decisions.items() if d.scene_id and d.scene_id not in valid and d.created_by is Creator.AI]:
            del state.decisions[did]

    # ------------------------------------------------------------------ validation / checkpoints
    def validate_state(self, project: Project, state: EditState) -> list[ValidationIssue]:
        v = TimelineValidator(project.assets, project.scenes, project.voice_over.asset_id, state.decisions, state.strategy, state.generation)
        return v.validate(state.timeline)

    def validate_timeline(self) -> list[ValidationIssue]:
        p = self._project()
        return self.validate_state(p, EditState.capture(p))

    def _checkpoint(self, project: Project, label: str) -> str:
        """Safety copy of the whole project before the AI changes the timeline (kept outside the project, newest 10)."""
        folder = self._checkpoint_dir(project.project_id)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = folder / f"{label}_{stamp}.json"
        n = 1
        while path.exists():
            path = folder / f"{label}_{stamp}_{n}.json"
            n += 1
        atomic_write_text(path, json.dumps(project.to_document(), ensure_ascii=False))
        for old in sorted(folder.glob(f"{label}_*.json"))[:-10]:
            old.unlink(missing_ok=True)
        return path.name

    def checkpoints(self) -> list[Path]:
        folder = self._checkpoint_dir(self._project().project_id)
        return sorted(folder.glob("before_ai_edit_*.json")) if folder.is_dir() else []

    # ------------------------------------------------------------------ inspection
    def decisions_for_scene(self, scene_id: str) -> list[EditingDecision]:
        p = self._project()
        order = {t: i for i, t in enumerate(DecisionType)}
        return sorted((d for d in p.editing_decisions.values() if d.scene_id == scene_id), key=lambda d: (d.start, order[d.type], d.slot))

    def decision_for_clip(self, clip_id: str) -> EditingDecision | None:
        p = self._project()
        clip = p.timeline.get_clip(clip_id)
        if clip is None:
            return None
        d = p.editing_decisions.get(clip.ai_decision_id) if clip.ai_decision_id else None
        return d or next((x for x in p.editing_decisions.values() if x.target_id == clip_id), None)

    def decisions_for_clip(self, clip_id: str) -> list[EditingDecision]:
        return [d for d in self._project().editing_decisions.values() if d.target_id == clip_id]

    def override_of(self, decision_id: str) -> EditingDecision | None:
        for o in self._project().ai_overrides:
            if o.override_id == decision_id:
                return o.original
        return None

    # ------------------------------------------------------------------ user changes
    def update_decision(self, decision_id: str, parameters: dict | None = None, *, start: float | None = None, duration: float | None = None) -> EditingDecision:
        """Edit the real parameters of a decision. The decision becomes USER-owned and records which AI decision it overrides."""
        p = self._project()
        asm = TimelineAssemblyService(p, None)
        d = update_decision(asm, decision_id, parameters, start, duration)
        issues = [i for i in self.validate_state(p, asm.state) if i.severity == "error" and i.scene_id == d.scene_id]
        if issues:
            raise EditingError(issues[0].message, details="; ".join(str(i) for i in issues[:5]))
        self._commands.execute(ApplyEditCommand(p, asm.state, f"Edit {d.type.value.replace('_', ' ').lower()}"))
        return p.editing_decisions[d.decision_id]

    def set_lock(self, scene_id: str, aspect: str, locked: bool = True) -> None:
        aspect = aspect.upper()
        if aspect not in LOCK_ASPECTS:
            raise EditingError(f"Unknown lock “{aspect}”.")
        p = self._project()
        st = EditState.capture(p)
        tl = st.timeline
        clips = [c for t in tl.tracks for c in t.clips if c.scene_id == scene_id]
        if aspect == "SCENE":
            ids = st.generation.locked_scenes
            if locked and scene_id not in ids:
                ids.append(scene_id)
            if not locked and scene_id in ids:
                ids.remove(scene_id)
        elif aspect in ("VISUAL", "TIMING"):
            for c in clips:
                if c.kind == "media":
                    c.locked = locked
            self._lock_decisions(st, scene_id, (DecisionType.VISUAL_TIMING, DecisionType.TRIM, DecisionType.CUT), locked)
        elif aspect == "TEXT":
            for c in clips:
                if c.kind == "text":
                    c.locked = locked
            self._lock_decisions(st, scene_id, (DecisionType.TEXT, DecisionType.NUMBER_EMPHASIS), locked)
        else:  # MOTION
            self._lock_decisions(st, scene_id, (DecisionType.ZOOM, DecisionType.PAN, DecisionType.KEYFRAME, DecisionType.EVIDENCE_FOCUS), locked)
        self._commands.execute(ApplyEditCommand(p, st, f"{'Lock' if locked else 'Unlock'} {aspect.lower()}"))

    @staticmethod
    def _lock_decisions(st: EditState, scene_id: str, types: tuple[DecisionType, ...], locked: bool) -> None:
        for d in st.decisions.values():
            if d.scene_id == scene_id and d.type in types:
                d.locked = locked

    def lock_clip(self, clip_id: str, locked: bool = True) -> None:
        p = self._project()
        st = EditState.capture(p)
        clip = st.timeline.get_clip(clip_id)
        if clip is None:
            raise EditingError("That clip no longer exists.")
        clip.locked = locked
        for d in st.decisions.values():
            if d.target_id == clip_id and d.type in (DecisionType.VISUAL_TIMING, DecisionType.TEXT, DecisionType.NUMBER_EMPHASIS, DecisionType.TRIM):
                d.locked = locked
        self._commands.execute(ApplyEditCommand(p, st, f"{'Lock' if locked else 'Unlock'} clip"))

    def replace_clip_asset(self, clip_id: str, asset_id: str) -> None:
        """REPLACE: swap the media of a visual clip, keeping its place and timing. The clip becomes USER-owned (AI regeneration keeps it)."""
        from app.editing.assembly import _own
        from app.media.asset import AssetType

        p = self._project()
        asm = TimelineAssemblyService(p, None)
        clip = asm.state.timeline.get_clip(clip_id)
        asset = p.assets.get(asset_id)
        if clip is None or clip.kind != "media" or clip.track_id in ("track_a1", "track_a2", "track_a3"):
            raise EditingError("Choose a visual clip to replace.")
        if asset is None or asset.type is AssetType.AUDIO:
            raise EditingError("Choose a video or image asset from the project.")
        need = clip.duration * clip.speed
        if asset.type is AssetType.IMAGE:
            clip.source_in, clip.source_out = 0.0, need
        else:
            total = asset.duration or 0.0
            if total + 1e-6 < need:
                raise EditingError(f"“{asset.name}” is only {total:.1f}s long but the clip needs {need:.1f}s. Shorten the clip first or choose longer media.")
            s_in = min(clip.source_in, max(0.0, total - need))
            clip.source_in, clip.source_out = s_in, s_in + need
        clip.asset_id, clip.created_by = asset.id, Creator.USER.value
        d = asm.state.decisions.get(clip.ai_decision_id) if clip.ai_decision_id else None
        if d is not None:
            d = _own(asm, d)
            d.parameters.update(asset_id=asset.id, operation="REPLACE", source_in=clip.source_in, source_out=clip.source_out)
            d.reason = f"Replaced by the user with {asset.name}."
            d.confidence = 100.0
        issues = [i for i in self.validate_state(p, asm.state) if i.severity == "error" and i.clip_id == clip_id]
        if issues:
            raise EditingError(issues[0].message)
        self._commands.execute(ApplyEditCommand(p, asm.state, "Replace visual"))

    def assign_scene_visual(self, scene_id: str, asset_id: str) -> None:
        """Manual Add: use an asset that is already in the project as the scene's approved visual."""
        p = self._project()
        asset = p.assets.get(asset_id)
        if asset is None or asset.type.value == "audio":
            raise EditingError("Choose a video or image asset from the project.")
        if not any(s.id == scene_id for s in p.scenes):
            raise EditingError("That scene no longer exists.")
        a = VisualAssignment(scene_id, None, asset_id, "USER", None, True, False, acquisition=Acquisition.LOCAL, source_type=asset.source_type,
                             note="Added manually in the AI Edit page.")
        self._commands.execute(SceneDecisionCommand(p, scene_id, "Add visual to scene", assignment=a))

    def add_scene_visual(self, scene_id: str, asset_id: str) -> None:
        """Add one more visual to a scene (multi-visual scenes). The assignment stays the primary visual."""
        p = self._project()
        if p.assets.get(asset_id) is None:
            raise EditingError("That asset is not in the project.")
        st = EditState.capture(p)
        lst = st.strategy.scene_visuals.setdefault(scene_id, [])
        if asset_id not in lst:
            lst.append(asset_id)
        self._commands.execute(ApplyEditCommand(p, st, "Add visual to scene"))

    # ------------------------------------------------------------------ hooks / preview
    def override_command(self, clip_id: str, action: str) -> Command | None:
        p = self._projects.current
        clip = p.timeline.get_clip(clip_id) if p else None
        if p is None or clip is None or not (clip.scene_id or clip.ai_decision_id):
            return None
        return RecordUserDeleteCommand(p, clip_id) if action == "delete" else MarkUserEditCommand(p, clip_id)

    def composer(self) -> PreviewComposer:
        return PreviewComposer(self._project())


_ = (AppError, VisualStatus, re)
