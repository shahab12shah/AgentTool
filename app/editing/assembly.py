"""TimelineAssemblyService: turns scene plans into the real, editable project timeline.

The assembler works on *copies* of the project's timeline and editing state and respects ownership:
  * clips/decisions created by the USER (or locked) are preserved,
  * AI-owned elements of the scene are replaced by the new plan,
  * slots the user deleted are not re-created.
It returns a candidate state; the service validates it and installs it as one undoable command.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from app.editing.context import EditingContext
from app.editing.models import (
    Creator,
    DecisionType,
    EditingDecision,
    EditingStrategy,
    EvidencePlan,
    MotionPlan,
    OverrideRecord,
    PlannedSegment,
    SceneEditStatus,
    SceneGeneration,
    ScenePlan,
    TextGraphic,
    TimelineGeneration,
    TransitionPlan,
    VisualSegment,
    VisualStatus,
    now_iso,
)
from app.editing.planners import evidence_keyframes, motion_keyframes
from app.media.asset import AssetType
from app.timeline.clip import KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.timeline import DEFAULT_TRACKS, Timeline, new_clip_id
from app.timeline.track import Track

MIN_PIECE = 0.5
VIDEO_TRACKS = ("track_v1", "track_v2")
STILL_TRACKS = ("track_v3", "track_v2", "track_v1")
OWN_MOTION = (DecisionType.ZOOM.value, DecisionType.PAN.value, DecisionType.KEYFRAME.value)


@dataclass
class EditState:
    """Everything an AI edit can change; snapshotted for undo."""

    timeline: Timeline
    decisions: dict[str, EditingDecision]
    overrides: list[OverrideRecord]
    strategy: EditingStrategy
    generation: TimelineGeneration
    version: int
    notes: list[str] = field(default_factory=list)
    placed: list[str] = field(default_factory=list)  # scene ids whose plan was installed

    @classmethod
    def capture(cls, project) -> "EditState":
        return cls(deepcopy(project.timeline), deepcopy(project.editing_decisions), deepcopy(project.ai_overrides),
                   deepcopy(project.editing_strategy), deepcopy(project.timeline_generation), project.timeline_version)

    def install(self, project) -> None:
        project.timeline = deepcopy(self.timeline)
        project.editing_decisions = deepcopy(self.decisions)
        project.ai_overrides = deepcopy(self.overrides)
        project.editing_strategy = deepcopy(self.strategy)
        project.timeline_generation = deepcopy(self.generation)
        project.timeline_version = self.version


def subtract(span: tuple[float, float], fixed: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [span]
    for a, b in sorted(fixed):
        nxt = []
        for s, e in pieces:
            if b <= s or a >= e:
                nxt.append((s, e))
                continue
            if a > s:
                nxt.append((s, a))
            if b < e:
                nxt.append((b, e))
        pieces = nxt
    return [(s, e) for s, e in pieces if e - s >= 1e-6]


class TimelineAssemblyService:
    def __init__(self, project, ctx: EditingContext | None = None, assets=None) -> None:
        self.project, self.ctx = project, ctx
        self.assets = assets or project.assets
        self.canvas = ctx.canvas if ctx else (project.settings.width, project.settings.height)
        self.state = EditState.capture(project)
        self._counter = max((int(d.rsplit("_", 1)[-1]) for d in self.state.decisions if d.rsplit("_", 1)[-1].isdigit()), default=0)

    # ------------------------------------------------------------------ ids / small helpers
    def _id(self) -> str:
        self._counter += 1
        return f"dec_{self._counter:05d}"

    def _decision(self, scene_id: str, type_: DecisionType, slot: str, target: str, start: float, duration: float, params: dict, reason: str,
                  conf: float, by: Creator = Creator.AI) -> EditingDecision:
        d = EditingDecision(self._id(), scene_id, type_, slot, target, round(start, 4), round(duration, 4), params, reason, round(conf, 1), by)
        self.state.decisions[d.decision_id] = d
        return d

    def _ensure_tracks(self) -> None:
        tl = self.state.timeline
        have = {t.id for t in tl.tracks}
        for i, (tid, name, kind) in enumerate(DEFAULT_TRACKS):
            if tid not in have:
                video_like = kind.value != "audio"
                at = sum(1 for t in tl.tracks if t.kind.value != "audio") if video_like else len(tl.tracks)
                tl.insert_track(Track(tid, name, kind), at)

    def _track_free(self, track_id: str, start: float, end: float, ignore: set[str] = frozenset()) -> bool:  # type: ignore[assignment]
        try:
            t = self.state.timeline.get_track(track_id)
        except Exception:
            return False
        return not any(c.id not in ignore and start < c.timeline_end - 1e-6 and end > c.timeline_start + 1e-6 for c in t.clips)

    # ------------------------------------------------------------------ voice-over + global audio
    def _ensure_voice(self) -> None:
        vo = self.ctx.voice_asset_id
        asset = self.assets.get(vo) if vo else None
        if asset is None or not asset.duration:
            return
        a1 = self.state.timeline.get_track("track_a1")
        if any(c.asset_id == vo for t in self.state.timeline.tracks if t.is_audio for c in t.clips):
            return
        if not self._track_free("track_a1", 0.0, asset.duration):
            self.state.notes.append("Voice-over could not be placed: track A1 is occupied.")
            return
        a1.clips.append(Clip(new_clip_id(), "track_a1", vo, 0.0, asset.duration, 0.0, asset.duration, created_by=Creator.SYSTEM.value, slot="voice",
                             kind=KIND_MEDIA, audio={"volume": 1.0, "role": "VOICE"}))
        a1.sort()

    def _global_audio(self, scene_ids: list[str]) -> None:
        if not self.ctx.scenes:
            return
        audio = self.state.strategy.audio
        first, last = self.ctx.scenes[0].scene, self.ctx.scenes[-1].scene
        for key, slot, scene, start, dur in (("FADE_IN", "audio:fade_in", first, first.start, audio.fade_in),
                                              ("FADE_OUT", "audio:fade_out", last, max(last.start, last.end - audio.fade_out), audio.fade_out)):
            if scene.id not in scene_ids:
                continue
            for d in [d for d in self.state.decisions.values() if d.scene_id == scene.id and d.slot == slot and d.created_by is not Creator.USER]:
                del self.state.decisions[d.decision_id]
            if not self.ctx.settings.smart_audio_ducking:
                continue
            self._decision(scene.id, DecisionType.AUDIO_DUCK, slot, "", start, min(dur, scene.duration),
                           {"kind": key, "music_level": audio.music_level, "start": round(start, 3), "end": round(start + min(dur, scene.duration), 3)},
                           "Music fades in under the opening." if key == "FADE_IN" else "Music fades out with the closing narration.", 95, Creator.SYSTEM)

    # ------------------------------------------------------------------ the main entry point
    def assemble(self, plans: list[ScenePlan], profile, session_id: str = "") -> EditState:
        st = self.state
        self._ensure_tracks()
        self._ensure_voice()
        st.strategy.profile = deepcopy(profile)
        st.strategy.style = self.ctx.settings.style
        st.strategy.provider = self.ctx.settings.provider
        st.strategy.audio = type(st.strategy.audio)() if not st.generation.version else st.strategy.audio
        if not st.generation.version:
            from app.editing.planners import AudioPlanner
            from app.editing.presets import preset_for

            st.strategy.audio = AudioPlanner(preset_for(self.ctx.settings), self.ctx.settings).global_plan()
        st.strategy.captions.mode = self.ctx.settings.caption_mode
        for plan in plans:
            self._scene(plan)
        self._global_audio([p.scene_id for p in plans if p.scene_id in st.placed])
        for t in st.timeline.tracks:
            t.sort()
        st.generation.version = st.version = st.version + 1
        st.generation.style = self.ctx.settings.style
        st.generation.generated_at = now_iso()
        st.generation.last_session_id = session_id
        return st

    # ------------------------------------------------------------------ per scene
    def _scene(self, plan: ScenePlan) -> None:
        st, sid = self.state, plan.scene_id
        gen = st.generation
        sc = self.ctx.by_id(sid)
        if sid in gen.locked_scenes:
            gen.scenes[sid] = SceneGeneration(sid, SceneEditStatus.LOCKED, "", gen.scenes.get(sid, SceneGeneration(sid)).input_hash,
                                              sc.visual_status, generated_at=now_iso())
            return
        tl = st.timeline
        scene_clips = [(t, c) for t in tl.tracks for c in t.clips if c.scene_id == sid and c.metadata.get("phase") != 5]  # presentation layer clips are Phase 5's
        preserved = [c for _t, c in scene_clips if c.locked or c.created_by == Creator.USER.value]
        for t, c in scene_clips:
            if c not in preserved and c.created_by == Creator.AI.value:
                t.clips.remove(c)
        kept_ids = {c.id for c in preserved}
        user_dec = {d.slot: d for d in st.decisions.values() if d.scene_id == sid and (d.created_by is Creator.USER or d.locked)}
        protected = {c.ai_decision_id for c in preserved if c.ai_decision_id}
        for d in [d for d in st.decisions.values() if d.scene_id == sid and d.created_by is Creator.AI and not d.locked and d.decision_id not in protected
                  and not (d.type.value in (DecisionType.CUT.value, DecisionType.TRIM.value) and d.target_id in kept_ids)]:
            del st.decisions[d.decision_id]
        # a user decision that pointed at a removed clip waits to be re-applied to the regenerated clip
        for d in user_dec.values():
            if d.target_id and d.target_id not in kept_ids and d.type.value in (*OWN_MOTION, DecisionType.EVIDENCE_FOCUS.value, DecisionType.TRANSITION.value):
                d.target_id = ""
        suppressed = {x.split("|", 1)[1] for x in gen.suppressed_slots if x.split("|", 1)[0] == sid}
        placed: list[VisualSegment] = []
        host_by_index: dict[int, Clip] = {}
        fixed_clips = [c for c in preserved if c.kind == KIND_MEDIA and c.track_id in ("track_v1", "track_v2", "track_v3")]
        preserved_media = {c.slot: c for c in preserved if c.kind == KIND_MEDIA and c.slot}
        for i, ps in enumerate(plan.segments):
            seg = ps.segment
            if seg.slot in suppressed:
                continue
            existing = preserved_media.get(seg.slot)
            if existing is not None:
                host_by_index[i] = existing
                placed.append(self._segment_of(existing, seg))
                continue
            others = [(c.timeline_start, c.timeline_end) for c in fixed_clips if c.slot != seg.slot]
            for piece_no, (a, b) in enumerate(subtract((seg.start, seg.start + seg.duration), others)):
                if b - a < MIN_PIECE and (b - a) < seg.duration - 1e-6:
                    continue
                clip = self._media_clip(plan, ps, a, b, piece_no)
                if clip is None:
                    continue
                host_by_index.setdefault(i, clip)
                placed.append(self._segment_of(clip, seg))
        for i, ps in enumerate(plan.segments):
            host = host_by_index.get(i)
            if host is not None:
                self._effects(plan, ps, host, i, user_dec, first=(i == min(host_by_index)))
        self._texts(plan, user_dec, suppressed, preserved)
        self._ducks(plan, user_dec)
        self._captions(plan, user_dec)
        st.strategy.briefs[sid] = plan.brief
        st.strategy.segments[sid] = placed
        st.strategy.captions.emphasis[sid] = list(plan.caption_emphasis)
        st.strategy.captions.region_by_scene[sid] = plan.caption_region
        no_visual = sc.visual_status != VisualStatus.APPROVED.value
        gen.scenes[sid] = SceneGeneration(sid, SceneEditStatus.NEEDS_VISUAL if no_visual else SceneEditStatus.COMPLETE, "", plan.input_hash,
                                          sc.visual_status, gen.scenes.get(sid, SceneGeneration(sid)).attempts + 1, now_iso())
        st.placed.append(sid)

    @staticmethod
    def _segment_of(clip: Clip, seg: VisualSegment) -> VisualSegment:
        out = deepcopy(seg)
        out.start, out.duration, out.asset_id, out.source_in, out.source_out, out.speed = (
            clip.timeline_start, clip.duration, clip.asset_id, clip.source_in, clip.source_out, clip.speed)
        return out

    # ------------------------------------------------------------------ visual clip
    def _media_clip(self, plan: ScenePlan, ps: PlannedSegment, a: float, b: float, piece_no: int) -> Clip | None:
        seg = ps.segment
        asset = self.assets.get(seg.asset_id)
        if asset is None:
            self.state.notes.append(f"Scene {seg.scene_id}: asset {seg.asset_id} is missing; the segment was skipped.")
            return None
        dur = b - a
        trimmed = (a - seg.start) > 1e-6 or dur < seg.duration - 1e-6
        s_in = seg.source_in + (a - seg.start) * seg.speed
        s_out = s_in + dur * seg.speed
        if asset.type is not AssetType.IMAGE and asset.duration and s_out > asset.duration + 1e-6:
            s_out = asset.duration
            s_in = max(0.0, s_out - dur * seg.speed)
        primary = plan.segments[0].segment.asset_id if plan.segments else asset.id
        if asset.type is AssetType.IMAGE:
            order = STILL_TRACKS
        elif asset.id != primary:  # cutaway footage from an additional visual -> B-roll track
            order = ("track_v2", "track_v1", "track_v3")
        else:
            order = ("track_v1", "track_v2", "track_v3")
        track_id = next((t for t in order if self._track_free(t, a, b)), None)
        if track_id is None:
            self.state.notes.append(f"Scene {seg.scene_id}: no free visual track for {a:.2f}-{b:.2f}s; segment skipped.")
            return None
        slot = seg.slot + (f".{piece_no}" if piece_no else "")
        clip = Clip(new_clip_id(), track_id, asset.id, round(a, 4), round(dur, 4), round(s_in, 4), round(s_out, 4), speed=seg.speed, kind=KIND_MEDIA,
                    scene_id=seg.scene_id, slot=slot, created_by=Creator.AI.value, effects={"fit": seg.fit},
                    metadata={"segment_id": seg.visual_segment_id, "reuse_count": seg.reuse_count, "previous_scene_id": seg.previous_scene_id,
                              "reuse_reason": seg.reuse_reason, "continues_previous": seg.continues_previous})
        d = self._decision(seg.scene_id, DecisionType.VISUAL_TIMING, slot, clip.id, a, dur,
                           {"asset_id": asset.id, "track_id": track_id, "source_in": clip.source_in, "source_out": clip.source_out, "speed": seg.speed,
                            "operation": seg.operation if not trimmed else "TRIM", "fit": seg.fit, "segment_id": seg.visual_segment_id,
                            "reuse_count": seg.reuse_count, "previous_scene_id": seg.previous_scene_id, "reuse_reason": seg.reuse_reason},
                           seg.reason or "Visual timed to the narration.", seg.confidence)
        clip.ai_decision_id = d.decision_id
        self.state.timeline.get_track(track_id).clips.append(clip)
        self.state.timeline.get_track(track_id).sort()
        if a > self.ctx.by_id(seg.scene_id).scene.start + 1e-6:
            self._decision(seg.scene_id, DecisionType.CUT, slot, clip.id, a, 0.0, {"at": round(a, 3), "operation": "CUT"},
                           f"Visual change at {a:.2f}s. {seg.reason.split(';')[0]}", seg.confidence)
        if seg.operation in ("TRIM", "EXTEND", "SHORTEN") or s_in > 1e-6 or trimmed:
            self._decision(seg.scene_id, DecisionType.TRIM, slot, clip.id, a, dur,
                           {"source_in": clip.source_in, "source_out": clip.source_out, "speed": seg.speed, "asset_duration": asset.duration},
                           "Source range chosen non-destructively; the original media is untouched.", seg.confidence)
        return clip

    # ------------------------------------------------------------------ motion / evidence / transition on a host clip
    def _effects(self, plan: ScenePlan, ps: PlannedSegment, clip: Clip, idx: int, user_dec: dict[str, EditingDecision], first: bool) -> None:
        sid = plan.scene_id
        ev_slot, mo_slot, tr_slot = f"evidence:{idx}", f"motion:{idx}", "transition:in"
        # evidence
        ud = user_dec.get(ev_slot)
        clip.keyframes = [k for k in clip.keyframes if not self._owned_by_ai(k.decision_id)]
        if ud is not None:
            if not ud.target_id:  # re-apply the user's evidence parameters to the regenerated clip
                ud.target_id = clip.id
                ud.start, ud.duration = clip.timeline_start, clip.duration
                self.apply_evidence(clip, ud)
        elif ps.evidence is not None:
            self._new_evidence(sid, clip, ps.evidence, ev_slot)
        # motion (never together with evidence: the evidence zoom owns the scale)
        ud = user_dec.get(mo_slot)
        has_ev = ev_slot in user_dec or ps.evidence is not None
        if ud is not None:
            if not ud.target_id:
                ud.target_id = clip.id
                ud.start, ud.duration = clip.timeline_start, clip.duration
                clip.keyframes += self.keyframes_for(ud, clip)
        elif ps.motion is not None and not has_ev:
            m = ps.motion
            d = self._decision(sid, DecisionType(m.family), mo_slot, clip.id, clip.timeline_start, clip.duration,
                               {"kind": m.kind, "start_scale": m.start_scale, "end_scale": m.end_scale, "start_pos": list(m.start_pos),
                                "end_pos": list(m.end_pos), "interpolation": m.interpolation}, m.reason, m.confidence)
            clip.keyframes += motion_keyframes(m, clip.duration, d.decision_id)
        # transition into the scene
        if first:
            ud = user_dec.get(tr_slot)
            if ud is not None:
                ud.target_id = clip.id
                clip.transition = {"type": ud.parameters.get("type", "CUT"), "duration": ud.parameters.get("duration", 0.0), "decision_id": ud.decision_id}
            elif ps.transition is not None:
                t = ps.transition
                d = self._decision(sid, DecisionType.TRANSITION, tr_slot, clip.id, clip.timeline_start, t.duration,
                                   {"type": t.type, "duration": t.duration}, t.reason, t.confidence)
                clip.transition = {"type": t.type, "duration": min(t.duration, clip.duration * 0.5), "decision_id": d.decision_id}

    def _owned_by_ai(self, decision_id: str) -> bool:
        """A keyframe is stale AI output when its decision is gone (deleted before regeneration); user/locked ones stay."""
        return decision_id != "" and decision_id not in self.state.decisions

    def _new_evidence(self, sid: str, clip: Clip, ev: EvidencePlan, slot: str) -> None:
        d = self._decision(sid, DecisionType.EVIDENCE_FOCUS, slot, clip.id, clip.timeline_start, clip.duration,
                           {"region": list(ev.region), "region_detected": ev.region_detected, "zoom_scale": ev.zoom_scale, "highlight": ev.highlight,
                            "darken": ev.darken, "hold_wide": ev.hold_wide}, ev.reason, ev.confidence)
        self.apply_evidence(clip, d)

    def apply_evidence(self, clip: Clip, d: EditingDecision) -> None:
        """Keyframes on the host clip + a non-destructive highlight overlay. The document itself is never altered."""
        ev = self._evidence_plan(d)
        clip.keyframes = [k for k in clip.keyframes if k.decision_id != d.decision_id]
        clip.keyframes += evidence_keyframes(ev, clip.duration, self.canvas, d.decision_id)
        clip.effects["focus_region"] = list(ev.region)
        tl = self.state.timeline
        for t in tl.tracks:
            t.clips = [c for c in t.clips if not (c.kind == KIND_GRAPHIC and c.metadata.get("evidence_decision") == d.decision_id)]
        if ev.highlight:
            hold = min(ev.hold_wide, clip.duration * 0.25)
            start = clip.timeline_start + hold + min(clip.duration * 0.35, 0.8)
            end = clip.timeline_end - (max(0.6, clip.duration * 0.25) if clip.duration >= 4.0 else 0.0)
            if end - start >= 0.4 and self._track_free("track_v4", start, end):
                g = Clip(new_clip_id(), "track_v4", "", round(start, 4), round(end - start, 4), kind=KIND_GRAPHIC, scene_id=clip.scene_id,
                         slot=f"{d.slot}:highlight", created_by=d.created_by.value, ai_decision_id=d.decision_id,
                         effects={"highlight": {"region": list(ev.region), "style": "box", "darken_surround": ev.darken}},
                         metadata={"evidence_decision": d.decision_id, "host_clip": clip.id})
                tl.get_track("track_v4").clips.append(g)
                tl.get_track("track_v4").sort()

    @staticmethod
    def _evidence_plan(d: EditingDecision) -> EvidencePlan:
        p = d.parameters
        return EvidencePlan(tuple(p.get("region", (0.12, 0.28, 0.76, 0.22))), bool(p.get("region_detected", False)), float(p.get("zoom_scale", 1.8)),  # type: ignore[arg-type]
                            bool(p.get("highlight", True)), bool(p.get("darken", True)), float(p.get("hold_wide", 0.8)), d.reason, d.confidence)

    @staticmethod
    def keyframes_for(d: EditingDecision, clip: Clip):
        """Keyframes implied by a (possibly user-edited) motion decision."""
        p = d.parameters
        if p.get("keyframes"):
            from app.timeline.keyframes import Keyframe

            return [Keyframe(k["property"], float(k["time"]), float(k["value"]), k.get("interpolation", "linear"), d.decision_id) for k in p["keyframes"]]
        m = MotionPlan(str(p.get("kind", "SUBTLE_ZOOM")), d.type.value if d.type.value in ("ZOOM", "PAN") else "ZOOM", float(p.get("start_scale", 1.0)),
                       float(p.get("end_scale", 1.0)), tuple(p.get("start_pos", (0, 0))), tuple(p.get("end_pos", (0, 0))),  # type: ignore[arg-type]
                       str(p.get("interpolation", "ease_in_out")))
        return motion_keyframes(m, clip.duration, d.decision_id)

    # ------------------------------------------------------------------ text, audio, captions
    def _texts(self, plan: ScenePlan, user_dec: dict[str, EditingDecision], suppressed: set[str], preserved: list[Clip]) -> None:
        tl = self.state.timeline
        kept = {c.slot: c for c in preserved if c.kind == KIND_TEXT}
        for pt in plan.texts:
            if pt.slot in suppressed or pt.slot in kept or pt.slot in user_dec:
                continue
            g = pt.graphic
            if not self._track_free("track_v5", g.start, g.start + g.duration):
                continue
            clip = Clip(new_clip_id(), "track_v5", "", g.start, g.duration, kind=KIND_TEXT, scene_id=plan.scene_id, slot=pt.slot, created_by=Creator.AI.value,
                        text=g.to_dict(), animation={"in": g.animation, "out": "fade", "duration": 0.25}, metadata={"source_ref": g.source_ref})
            d = self._decision(plan.scene_id, DecisionType(pt.decision_type), pt.slot, clip.id, g.start, g.duration, g.to_dict(), pt.reason, pt.confidence)
            clip.ai_decision_id = d.decision_id
            tl.get_track("track_v5").clips.append(clip)
            tl.get_track("track_v5").sort()

    def _ducks(self, plan: ScenePlan, user_dec: dict[str, EditingDecision]) -> None:
        user_spans = [(d.parameters.get("start", d.start), d.parameters.get("end", d.start + d.duration)) for d in user_dec.values()
                      if d.type is DecisionType.AUDIO_DUCK]
        for i, du in enumerate(plan.ducks):
            if any(du.start < b and du.end > a for a, b in user_spans):
                continue
            self._decision(plan.scene_id, DecisionType.AUDIO_DUCK, f"duck:{i}", "", du.start, du.end - du.start,
                           {"kind": du.kind, "music_level": du.music_level, "ramp": du.ramp, "start": round(du.start, 3), "end": round(du.end, 3)},
                           du.reason, du.confidence)

    def _captions(self, plan: ScenePlan, user_dec: dict[str, EditingDecision]) -> None:
        if "caption" in user_dec or not self.ctx.settings.caption_mode == "ENABLED":
            return
        self._decision(plan.scene_id, DecisionType.CAPTION_EMPHASIS, "caption", "", self.ctx.by_id(plan.scene_id).scene.start, 0.0,
                       {"words": list(plan.caption_emphasis), "region": plan.caption_region, "mode": self.ctx.settings.caption_mode},
                       "Captions stay timeline data; these words get emphasis and the region avoids on-screen text.", 88)


_ = (TextGraphic, TransitionPlan)


# ====================================================================== user edits of AI decisions (inspector)
CLIP_OWNING = (DecisionType.VISUAL_TIMING.value, DecisionType.TEXT.value, DecisionType.NUMBER_EMPHASIS.value, DecisionType.TRIM.value)


def _own(asm: "TimelineAssemblyService", d: EditingDecision) -> EditingDecision:
    """The USER decision that now owns ``d`` (``d`` itself if it already is one). The replaced AI decision is kept in ``ai_overrides``."""
    if d.created_by is Creator.USER:
        return d
    new = deepcopy(d)
    new.decision_id, new.created_by, new.overrides_decision_id = asm._id(), Creator.USER, d.decision_id
    st = asm.state
    del st.decisions[d.decision_id]
    st.decisions[new.decision_id] = new
    st.overrides.append(OverrideRecord(new.decision_id, deepcopy(d)))
    for t in st.timeline.tracks:  # clips and keyframes that referenced the AI decision follow its replacement
        for c in t.clips:
            if c.ai_decision_id == d.decision_id:
                c.ai_decision_id = new.decision_id
            for k in c.keyframes:
                if k.decision_id == d.decision_id:
                    k.decision_id = new.decision_id
            if c.transition and c.transition.get("decision_id") == d.decision_id:
                c.transition["decision_id"] = new.decision_id
            if c.metadata.get("evidence_decision") == d.decision_id:
                c.metadata["evidence_decision"] = new.decision_id
    return new


def update_decision(asm: "TimelineAssemblyService", decision_id: str, parameters: dict | None = None, start: float | None = None,
                    duration: float | None = None) -> EditingDecision:
    from app.core.exceptions import EditingError

    st = asm.state
    d0 = st.decisions.get(decision_id)
    if d0 is None:
        raise EditingError("That decision no longer exists.")
    d = _own(asm, d0)
    params = dict(parameters or {})
    d.parameters.update(params)
    tl = st.timeline
    clip = tl.get_clip(d.target_id) if d.target_id else None
    t = d.type.value
    if t == DecisionType.VISUAL_TIMING.value and clip is not None:
        new_start = float(params.get("start", start if start is not None else clip.timeline_start))
        new_dur = float(params.get("duration", duration if duration is not None else clip.duration))
        if new_dur <= 0.05:
            raise EditingError("A visual needs a positive duration.")
        track = tl.get_track(clip.track_id)
        if any(o.id != clip.id and new_start < o.timeline_end - 1e-6 and new_start + new_dur > o.timeline_start + 1e-6 for o in track.clips):
            raise EditingError("That timing would overlap another clip on the same track.")
        clip.timeline_start, clip.duration = round(new_start, 4), round(new_dur, 4)
        s_in = float(params.get("source_in", clip.source_in))
        asset = asm.assets.get(clip.asset_id)
        max_out = asset.duration if asset and asset.type is not AssetType.IMAGE and asset.duration else None
        s_out = s_in + new_dur * clip.speed
        if max_out is not None and s_out > max_out + 1e-6:
            raise EditingError(f"The source media is only {max_out:.2f}s long; shorten the visual or move the source start earlier.")
        clip.source_in, clip.source_out = s_in, s_out
        d.parameters.update(source_in=s_in, source_out=s_out, start=clip.timeline_start, duration=clip.duration)
        clip.created_by = Creator.USER.value
        for k in clip.keyframes:  # keep keyframes inside the clip
            k.time = min(k.time, clip.duration)
    elif t in (DecisionType.ZOOM.value, DecisionType.PAN.value, DecisionType.KEYFRAME.value) and clip is not None:
        clip.keyframes = [k for k in clip.keyframes if k.decision_id != d.decision_id]
        clip.keyframes += TimelineAssemblyService.keyframes_for(d, clip)
    elif t == DecisionType.EVIDENCE_FOCUS.value and clip is not None:
        asm.apply_evidence(clip, d)
    elif t == DecisionType.TRANSITION.value and clip is not None:
        kind, dur = str(d.parameters.get("type", "CUT")), float(d.parameters.get("duration", 0.0))
        clip.transition = {"type": kind, "duration": min(dur, clip.duration), "decision_id": d.decision_id} if kind != "CUT" or dur else \
            {"type": "CUT", "duration": 0.0, "decision_id": d.decision_id}
    elif t in (DecisionType.TEXT.value, DecisionType.NUMBER_EMPHASIS.value) and clip is not None:
        g = {**(clip.text or {}), **{k: v for k, v in d.parameters.items() if k in TextGraphic.__dataclass_fields__}}
        if not str(g.get("content", "")).strip():
            raise EditingError("The text cannot be empty.")
        if start is not None or "start" in params:
            clip.timeline_start = float(params.get("start", start))
        if duration is not None or "duration" in params:
            clip.duration = float(params.get("duration", duration))
        if clip.duration <= 0.05:
            raise EditingError("The text needs a positive duration.")
        g["start"], g["duration"] = clip.timeline_start, clip.duration
        clip.text = g
        d.parameters = dict(g)
        clip.created_by = Creator.USER.value
    elif t == DecisionType.TRIM.value and clip is not None:
        s_in = float(d.parameters.get("source_in", clip.source_in))
        clip.source_in, clip.source_out = s_in, s_in + clip.duration * clip.speed
        clip.created_by = Creator.USER.value
    elif t == DecisionType.AUDIO_DUCK.value:
        a = float(d.parameters.get("start", d.start))
        b = float(d.parameters.get("end", a + d.duration))
        if b <= a:
            raise EditingError("The end must come after the start.")
        d.start, d.duration = a, b - a
    if start is not None and t not in (DecisionType.VISUAL_TIMING.value, DecisionType.TEXT.value, DecisionType.NUMBER_EMPHASIS.value, DecisionType.AUDIO_DUCK.value):
        d.start = start
    if clip is not None and t in (DecisionType.VISUAL_TIMING.value, DecisionType.TEXT.value, DecisionType.NUMBER_EMPHASIS.value):
        d.start, d.duration = clip.timeline_start, clip.duration
    d.confidence = 100.0  # the user decided
    for tr in tl.tracks:
        tr.sort()
    return d
