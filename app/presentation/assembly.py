"""PresentationAssembler: turns captions, graphics, music, SFX and ducking into real timeline clips and decisions.

Same ownership rules as the AI edit: USER-owned and locked elements are preserved, AI-owned unlocked elements of the regenerated part
are replaced, slots the user deleted are not re-created. Everything is built on a copy of the project state and installed as one
undoable command only after validation.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field

from app.audio.ducking import DuckPlan
from app.audio.sfx import SfxEvent
from app.captions.keywords import Keyword
from app.editing.models import Creator
from app.presentation.animation import normalize
from app.presentation.graphics import GraphicsPlan, PlannedGraphic, _norm
from app.presentation.models import (
    CaptionSegment,
    CaptionSettings,
    DuckingEvent,
    PartState,
    PresentationDecision,
    PresentationGeneration,
    PresentationOverride,
    PresentationType,
    ScenePresentationPlan,
)
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.keyframes import Keyframe, value_at
from app.timeline.timeline import DEFAULT_TRACKS, Timeline, new_clip_id
from app.timeline.track import Track, TrackKind

PHASE = 5
GRAPHIC_TYPES = {PresentationType.NUMBER_GRAPHIC, PresentationType.DATE_GRAPHIC, PresentationType.LOWER_THIRD, PresentationType.HEADLINE, PresentationType.TEXT_GRAPHIC,
                 PresentationType.MOTION_GRAPHIC, PresentationType.EVIDENCE_GRAPHIC}


def is_p5(c: Clip) -> bool:
    return c.metadata.get("phase") == PHASE


def owned(c: Clip) -> bool:
    """User-edited or locked: regeneration leaves it alone."""
    return c.locked or c.created_by == Creator.USER.value


@dataclass
class PresState:
    """Everything the presentation layer can change; snapshotted for undo."""

    timeline: Timeline
    decisions: dict[str, PresentationDecision]
    overrides: list[PresentationOverride]
    plans: dict[str, ScenePresentationPlan]
    ducking_events: list[DuckingEvent]
    keyword_emphasis: dict[str, list]
    generation: PresentationGeneration
    caption_settings: CaptionSettings
    notes: list[str] = field(default_factory=list)

    @classmethod
    def capture(cls, project) -> "PresState":
        return cls(deepcopy(project.timeline), deepcopy(project.presentation_decisions), deepcopy(project.presentation_overrides), deepcopy(project.presentation_plans),
                   deepcopy(project.ducking_events), deepcopy(project.keyword_emphasis), deepcopy(project.presentation_generation), deepcopy(project.caption_settings))

    def install(self, project) -> None:
        project.timeline.tracks = deepcopy(self.timeline.tracks)
        project.presentation_decisions.clear()
        project.presentation_decisions.update(deepcopy(self.decisions))
        project.presentation_overrides[:] = deepcopy(self.overrides)
        project.presentation_plans.clear()
        project.presentation_plans.update(deepcopy(self.plans))
        project.ducking_events[:] = deepcopy(self.ducking_events)
        project.keyword_emphasis.clear()
        project.keyword_emphasis.update(deepcopy(self.keyword_emphasis))
        project.presentation_generation = deepcopy(self.generation)
        project.caption_settings = deepcopy(self.caption_settings)


def settings_hash(cs: CaptionSettings, canvas) -> str:
    keys = ("style_id", "position", "custom_x", "custom_y", "max_lines", "max_words", "uppercase", "safe_margin_left", "safe_margin_right", "safe_margin_top",
            "safe_margin_bottom", "large_text", "high_contrast", "reading_speed", "keyword_highlight", "number_emphasis", "highlight_mode")
    return hashlib.sha1(json.dumps([getattr(cs, k) for k in keys] + [list(canvas)]).encode()).hexdigest()[:12]


class PresentationAssembler:
    def __init__(self, project) -> None:
        self.project = project
        self.state = PresState.capture(project)
        self.canvas = (project.settings.width, project.settings.height)
        self._n = max((int(k.rsplit("_", 1)[-1]) for k in self.state.decisions if k.rsplit("_", 1)[-1].isdigit()), default=0)

    # ------------------------------------------------------------------ helpers
    def new_id(self) -> str:
        self._n += 1
        return f"pdec_{self._n:05d}"

    def decision(self, scene_id: str, type_: PresentationType, slot: str, target: str, start: float, duration: float, params: dict, reason: str, conf: float,
                 by: Creator = Creator.AI) -> PresentationDecision:
        d = PresentationDecision(self.new_id(), scene_id, type_, slot, target, round(start, 4), round(duration, 4), params, reason, round(conf, 1), by)
        self.state.decisions[d.decision_id] = d
        return d

    def ensure_tracks(self) -> None:
        tl = self.state.timeline
        have = {t.id for t in tl.tracks}
        for tid, name, kind in DEFAULT_TRACKS:
            if tid not in have:
                at = sum(1 for t in tl.tracks if t.kind is not TrackKind.AUDIO) if kind is not TrackKind.AUDIO else len(tl.tracks)
                tl.insert_track(Track(tid, name, kind), at)

    def track_free(self, track_id: str, start: float, end: float, ignore: set[str] = frozenset()) -> bool:  # type: ignore[assignment]
        try:
            t = self.state.timeline.get_track(track_id)
        except Exception:
            return False
        return not any(c.id not in ignore and start < c.timeline_end - 1e-6 and end > c.timeline_start + 1e-6 for c in t.clips)

    def suppressed(self, scene_id: str) -> set[str]:
        return {x.split("|", 1)[1] for x in self.state.generation.suppressed_slots if x.split("|", 1)[0] == scene_id}

    def locked_scene(self, scene_id: str) -> bool:
        return scene_id in self.state.generation.locked_scenes

    def _scene_clips(self, scene_id: str, pred) -> list[tuple[Track, Clip]]:
        return [(t, c) for t in self.state.timeline.tracks for c in t.clips if c.scene_id == scene_id and pred(t, c)]

    def _drop_ai_decisions(self, scene_id: str, types: set[PresentationType], keep_targets: set[str]) -> None:
        for d in [d for d in self.state.decisions.values() if d.scene_id == scene_id and d.type in types and d.created_by is Creator.AI and not d.locked
                  and d.target_id not in keep_targets]:
            del self.state.decisions[d.decision_id]

    def finish_part(self, part: str, input_hash: str) -> None:
        g = self.state.generation
        st: PartState = getattr(g, part)
        setattr(g, part, PartState(st.version + 1, "COMPLETE", _now(), input_hash))
        g.version += 1

    def plan_for(self, scene_id: str) -> ScenePresentationPlan:
        return self.state.plans.setdefault(scene_id, ScenePresentationPlan(scene_id))

    # ================================================================== captions
    def captions(self, scene_id: str, segments: list[CaptionSegment], keywords: list[Keyword]) -> list[str]:
        """Replace the AI captions of one scene. Returns the new caption clip ids."""
        st = self.state
        if self.locked_scene(scene_id):
            return []
        old = self._scene_clips(scene_id, lambda t, c: c.kind == KIND_CAPTION)
        preserved = [c for _t, c in old if owned(c)]
        keep = {c.id for c in preserved}
        for t, c in old:
            if c.id not in keep and c.created_by == Creator.AI.value:
                t.clips.remove(c)
        self._drop_ai_decisions(scene_id, {PresentationType.CAPTION}, keep)
        sup = self.suppressed(scene_id)
        occupied = [(c.timeline_start, c.timeline_end) for c in preserved]
        made: list[str] = []
        track = st.timeline.get_track("track_v6")
        for i, seg in enumerate(segments):
            slot = f"caption:{i}"
            if slot in sup or any(seg.start < b - 0.02 and seg.end > a + 0.02 for a, b in occupied):
                continue
            if not self.track_free("track_v6", seg.start, seg.end):
                continue
            seg.caption_id = f"{scene_id}_c{i}"
            clip = Clip(new_clip_id(), "track_v6", "", seg.start, round(seg.end - seg.start, 4), kind=KIND_CAPTION, scene_id=scene_id, slot=slot, created_by=Creator.AI.value,
                        text=seg.to_dict(), animation=deepcopy(seg.animation), metadata={"phase": PHASE, "caption_id": seg.caption_id})
            d = self.decision(scene_id, PresentationType.CAPTION, slot, clip.id, seg.start, seg.end - seg.start,
                              {"caption_id": seg.caption_id, "style_id": seg.style_id, "lines": len(seg.lines), "reading_cps": seg.reading_cps,
                               "emphasis_words": seg.emphasis_words, "highlight_mode": seg.highlight_mode},
                              f"Caption from word timestamps ({len(seg.words)} words, {seg.reading_cps:.0f} characters/s).", 92.0 if seg.reading_cps <= 17 else 78.0)
            clip.ai_decision_id = d.decision_id
            track.clips.append(clip)
            made.append(clip.id)
        track.sort()
        # keyword emphasis record for the scene (the marks themselves live in the captions)
        for k in [k for k, d in st.decisions.items() if d.scene_id == scene_id and d.type is PresentationType.KEYWORD_EMPHASIS and d.created_by is Creator.AI and not d.locked]:
            del st.decisions[k]
        st.keyword_emphasis[scene_id] = [{"category": k.category, "text": k.text, "word_ids": k.word_ids, "importance": k.importance, "reason": k.reason, "source": k.source}
                                         for k in keywords]
        if keywords:
            self.decision(scene_id, PresentationType.KEYWORD_EMPHASIS, "keywords", "", segments[0].start if segments else 0.0, 0.0,
                          {"keywords": st.keyword_emphasis[scene_id]}, f"{len(keywords)} important word(s) emphasised in the captions.", 85.0)
        p = self.plan_for(scene_id)
        p.caption_plan = {"caption_ids": made, "count": len(made), "style_id": segments[0].style_id if segments else ""}
        p.keyword_plan = list(st.keyword_emphasis[scene_id])
        return made

    # ================================================================== graphics
    def graphics(self, gp: GraphicsPlan) -> list[str]:
        st, sid = self.state, gp.scene_id
        if self.locked_scene(sid):
            return []
        tl = st.timeline
        sup = self.suppressed(sid)
        all_gfx = self._scene_clips(sid, lambda t, c: c.kind in (KIND_TEXT, KIND_GRAPHIC))
        for t, c in all_gfx:  # AI-owned Phase 5 graphics are regenerated; Phase 4 ones are adopted when they match
            if is_p5(c) and c.kind == KIND_TEXT and not owned(c) and c.created_by == Creator.AI.value and not c.metadata.get("adopted"):
                t.clips.remove(c)
        keep = {c.id for _t, c in self._scene_clips(sid, lambda t, c: c.kind in (KIND_TEXT, KIND_GRAPHIC) and owned(c))}
        self._drop_ai_decisions(sid, GRAPHIC_TYPES, keep)
        existing = [c for _t, c in self._scene_clips(sid, lambda t, c: c.kind == KIND_TEXT and c.text)]
        made: list[str] = []
        plan = self.plan_for(sid)
        plan.number_graphics, plan.lower_thirds, plan.text_graphics, plan.motion_graphics = [], [], [], []
        for pg in gp.graphics:
            if pg.slot in sup:
                continue
            g = pg.graphic
            match = next((c for c in existing if _norm(str(c.text.get("content", ""))) == _norm(g.content) and not owned(c)), None)
            if match is None and any(c.text and _norm(str(c.text.get("content", ""))) == _norm(g.content) and owned(c) for c in existing):
                continue  # the user already has this graphic
            clip = match
            if clip is None:
                if not self.track_free("track_v5", g.start, g.start + g.duration):
                    gp.notes.append(f"{g.content}: no free text slot at {g.start:.1f}s; skipped.")
                    continue
                clip = Clip(new_clip_id(), "track_v5", "", g.start, g.duration, kind=KIND_TEXT, scene_id=sid, slot=pg.slot, created_by=Creator.AI.value,
                            metadata={"phase": PHASE})
                tl.get_track("track_v5").clips.append(clip)
                tl.get_track("track_v5").sort()
            else:  # adopt the Phase 4 overlay: same text, now with Phase 5 timing, variant and animation
                if abs(clip.timeline_start - g.start) > 1e-6 or abs(clip.duration - g.duration) > 1e-6:
                    if self.track_free(clip.track_id, g.start, g.start + g.duration, {clip.id}):
                        clip.timeline_start, clip.duration = g.start, g.duration
                clip.metadata["adopted"], clip.metadata["phase5"] = True, True
            clip.text = {**g.to_dict(), "content": g.content, "variant": pg.variant, "title": pg.title or g.content, "subtitle": pg.subtitle, "counter": pg.counter,
                         "derived": pg.derived}
            clip.text["start"], clip.text["duration"] = clip.timeline_start, clip.duration
            clip.animation = deepcopy(pg.animation)
            d = self.decision(sid, pg.type, pg.slot, clip.id, clip.timeline_start, clip.duration,
                              {"content": g.content, "variant": pg.variant, "style": g.style, "position": list(g.position), "size": g.size, "animation": deepcopy(pg.animation),
                               "counter": pg.counter, "source_ref": g.source_ref, "derived": pg.derived}, pg.reason, pg.confidence)
            if not clip.ai_decision_id or clip.metadata.get("phase") == PHASE:
                clip.ai_decision_id = d.decision_id
            made.append(clip.id)
            entry = {"clip_id": clip.id, "content": g.content, "start": clip.timeline_start, "duration": clip.duration, "decision_id": d.decision_id}
            {"NUMBER": plan.number_graphics, "DATE": plan.number_graphics, "LOWER_THIRD": plan.lower_thirds}.get(pg.variant, plan.text_graphics).append(entry)
        # evidence graphics (document highlights made by the AI edit): give them a tool and an animation
        if gp.evidence is not None:
            for _t, c in self._scene_clips(sid, lambda t, c: c.kind == KIND_GRAPHIC and "highlight" in c.effects and not owned(c)):
                ev = gp.evidence
                c.effects["evidence"] = {"tool": ev.tool, "dim": ev.dim, "region": c.effects["highlight"].get("region")}
                c.effects["highlight"]["darken_surround"] = ev.dim
                c.animation = deepcopy(ev.animation)
                c.metadata["phase5"] = True
                d = self.decision(sid, PresentationType.EVIDENCE_GRAPHIC, f"p5:evidence:{c.slot}", c.id, c.timeline_start, c.duration,
                                  {"tool": ev.tool, "dim": ev.dim, "region": c.effects["evidence"]["region"], "animation": deepcopy(ev.animation)}, ev.reason, 80.0)
                plan.motion_graphics.append({"clip_id": c.id, "tool": ev.tool, "decision_id": d.decision_id})
                made.append(c.id)
        plan.notes = list(gp.notes)
        return made

    # ================================================================== audio: music, SFX, ducking
    def place_music(self, aid: str, asset, start: float, end: float, volume: float, fade_in: float, fade_out: float, loop: bool, by: Creator, reason: str,
                    conf: float, ducking: bool = True) -> list[Clip]:
        """Music on A2 from ``start`` to ``end`` (looped back-to-back when the asset is shorter). Never overrides the voice timing."""
        tl = self.state.timeline
        self.ensure_tracks()
        track = tl.get_track("track_a2")
        total = float(asset.duration or 0.0)
        if total <= 0:
            raise ValueError("The music file has no duration.")
        pieces: list[Clip] = []
        t = start
        k = 0
        while t < end - 0.02 and (loop or k == 0):
            dur = min(total, end - t)
            c = Clip(new_clip_id(), "track_a2", asset.id, round(t, 4), round(dur, 4), 0.0, round(dur, 4), kind=KIND_MEDIA, slot=f"music:{aid}:{k}", created_by=by.value,
                     audio={"role": "MUSIC", "volume": volume, "fade_in": 0.0, "fade_out": 0.0, "loop": loop, "ducking": ducking},
                     metadata={"phase": PHASE, "assignment_id": aid, "piece": k})
            pieces.append(c)
            t += dur
            k += 1
        if not pieces:
            raise ValueError("Nothing to place.")
        pieces[0].audio["fade_in"] = min(fade_in, pieces[0].duration)
        pieces[-1].audio["fade_out"] = min(fade_out, pieces[-1].duration)
        for c in pieces:
            if not self.track_free("track_a2", c.timeline_start, c.timeline_end):
                raise ValueError("The music track already has audio in that range.")
        d = self.decision("", PresentationType.MUSIC, f"music:{aid}", pieces[0].id, start, end - start,
                          {"assignment_id": aid, "asset_id": asset.id, "volume": volume, "fade_in": fade_in, "fade_out": fade_out, "loop": loop, "ducking": ducking,
                           "clip_ids": [c.id for c in pieces]}, reason, conf, by)
        for c in pieces:
            c.ai_decision_id = d.decision_id
            track.clips.append(c)
        track.sort()
        return pieces

    def apply_ducking(self, plan: DuckPlan, keyframes: list[tuple[float, float]], audio_settings) -> int:
        """Attach the ducking envelope to every music clip as editable volume keyframes (skipping clips whose mix the user owns/locked)."""
        st = self.state
        music = [c for c in st.timeline.get_track("track_a2").clips if c.kind == KIND_MEDIA and c.audio.get("ducking", True)] if any(
            t.id == "track_a2" for t in st.timeline.tracks) else []
        by_assignment: dict[str, list[Clip]] = {}
        for c in music:
            by_assignment.setdefault(c.metadata.get("assignment_id", c.id), []).append(c)
        # events: replace AI ones, keep USER ones
        st.ducking_events[:] = [e for e in st.ducking_events if e.created_by == "USER"] + [deepcopy(e) for e in plan.events]
        done = 0
        for aid, pieces in by_assignment.items():
            slot = f"duck:{aid}"
            user = next((d for d in st.decisions.values() if d.slot == slot and (d.created_by is Creator.USER or d.locked)), None)
            if user is not None:
                continue
            for d in [d for d in st.decisions.values() if d.slot == slot and d.created_by is Creator.AI]:
                del st.decisions[d.decision_id]
            d = self.decision("", PresentationType.DUCKING, slot, pieces[0].id, plan.levels[0][0] if plan.levels else 0.0, (plan.levels[-1][1] - plan.levels[0][0]) if plan.levels else 0.0,
                              {"assignment_id": aid, "events": [e.event_id for e in plan.events], "keyframe_count": len(keyframes),
                               "levels": {"normal": audio_settings.music_level, "important": audio_settings.important_level, "pause": audio_settings.pause_level}},
                              "Music follows the narration: ducks under important speech, rises modestly in pauses.", 90.0)
            for c in sorted(pieces, key=lambda x: x.timeline_start):
                c.keyframes = [k for k in c.keyframes if not (k.property == "volume" and (k.decision_id == "" or k.decision_id not in st.decisions or
                                                                                             st.decisions[k.decision_id].created_by is Creator.AI))]
                pts = [(t, v) for t, v in keyframes if c.timeline_start - 1e-6 <= t <= c.timeline_end + 1e-6]
                src = [Keyframe("volume", t, v) for t, v in keyframes]
                edge = [(c.timeline_start, value_at(src, "volume", c.timeline_start)), (c.timeline_end, value_at(src, "volume", c.timeline_end))]
                allp = sorted({round(t, 3): v for t, v in edge + pts}.items())
                c.keyframes += [Keyframe("volume", round(t - c.timeline_start, 3), round(v, 4), "linear", d.decision_id) for t, v in allp]
                c.ai_decision_id = c.ai_decision_id or d.decision_id
            done += 1
        return done

    def place_sfx(self, scene_ids: set[str], events: list[SfxEvent], assets, by: Creator = Creator.AI) -> list[str]:
        """Replace the AI sound effects of ``scene_ids`` with ``events`` (events without a library asset are recorded but not placed)."""
        st = self.state
        self.ensure_tracks()
        tl = st.timeline
        for t in tl.tracks:
            if t.kind is TrackKind.AUDIO:
                for c in [c for c in t.clips if c.scene_id in scene_ids and c.audio.get("role") == "SFX" and is_p5(c) and c.created_by == Creator.AI.value and not c.locked]:
                    t.clips.remove(c)
        for sid in scene_ids:
            self._drop_ai_decisions_any(sid, {PresentationType.SFX})
            self.plan_for(sid).sfx_events = []
        made: list[str] = []
        n = 0
        for ev in events:
            if ev.scene_id not in scene_ids or self.locked_scene(ev.scene_id):
                continue
            plan = self.plan_for(ev.scene_id)
            entry = {"time": ev.time, "duration": ev.duration, "category": ev.category, "volume": ev.volume, "reason": ev.reason, "confidence": ev.confidence,
                     "trigger": ev.trigger, "asset_id": ev.asset_id, "placed": False}
            if f"sfx:{ev.trigger}:{ev.time:.2f}" in self.suppressed(ev.scene_id):
                plan.sfx_events.append(entry)
                continue
            asset = assets.get(ev.asset_id) if ev.asset_id else None
            if asset is None:
                entry["note"] = f"No SFX asset of category {ev.category} in the library."
                plan.sfx_events.append(entry)
                continue
            dur = min(ev.duration, float(asset.duration or ev.duration))
            track_id = self._free_sfx_track(ev.time, ev.time + dur)
            slot = f"sfx:{ev.trigger}:{ev.time:.2f}"
            clip = Clip(new_clip_id(), track_id, asset.id, round(ev.time, 4), round(dur, 4), 0.0, round(dur, 4), kind=KIND_MEDIA, scene_id=ev.scene_id, slot=slot,
                        created_by=by.value, audio={"role": "SFX", "volume": ev.volume, "fade_in": 0.0, "fade_out": min(0.08, dur / 3), "category": ev.category},
                        metadata={"phase": PHASE, "sfx_id": f"sfx_{n:03d}_{ev.scene_id}", "trigger": ev.trigger})
            n += 1
            d = self.decision(ev.scene_id, PresentationType.SFX, slot, clip.id, ev.time, dur,
                              {"asset_id": asset.id, "category": ev.category, "volume": ev.volume, "timestamp": ev.time, "trigger": ev.trigger}, ev.reason, ev.confidence, by)
            clip.ai_decision_id = d.decision_id
            tl.get_track(track_id).clips.append(clip)
            tl.get_track(track_id).sort()
            entry.update(placed=True, clip_id=clip.id, decision_id=d.decision_id)
            plan.sfx_events.append(entry)
            made.append(clip.id)
        return made

    def _free_sfx_track(self, a: float, b: float) -> str:
        """A3 first; sounds that overlap are layered on extra SFX tracks."""
        tl = self.state.timeline
        k = 1
        while True:
            tid = "track_a3" if k == 1 else f"track_a3_{k}"
            if not any(t.id == tid for t in tl.tracks):
                tl.insert_track(Track(tid, f"A3 SFX layer {k}", TrackKind.AUDIO), len(tl.tracks))
            if self.track_free(tid, a, b):
                return tid
            k += 1

    def _drop_ai_decisions_any(self, scene_id: str, types: set[PresentationType]) -> None:
        keep = {c.id for t in self.state.timeline.tracks for c in t.clips if c.scene_id == scene_id and owned(c)}
        self._drop_ai_decisions(scene_id, types, keep)


def _now() -> str:
    from app.editing.models import now_iso

    return now_iso()


_ = (normalize,)


# ====================================================================== user edits (inspector / services)
OWNING_TYPES = (PresentationType.CAPTION, PresentationType.NUMBER_GRAPHIC, PresentationType.DATE_GRAPHIC, PresentationType.LOWER_THIRD, PresentationType.HEADLINE,
                PresentationType.TEXT_GRAPHIC, PresentationType.MOTION_GRAPHIC, PresentationType.EVIDENCE_GRAPHIC, PresentationType.MUSIC, PresentationType.SFX)


def own_clip(asm: PresentationAssembler, clip: Clip) -> list[PresentationDecision]:
    """The clip and the presentation decisions that target it become USER-owned; each replaced AI decision is kept in ``presentation_overrides``."""
    st = asm.state
    clip.created_by = Creator.USER.value
    out: list[PresentationDecision] = []
    for d in [d for d in st.decisions.values() if d.target_id == clip.id and d.type in OWNING_TYPES]:
        if d.created_by is Creator.USER:
            out.append(d)
            continue
        new = deepcopy(d)
        new.decision_id, new.created_by, new.overrides_decision_id = asm.new_id(), Creator.USER, d.decision_id
        del st.decisions[d.decision_id]
        st.decisions[new.decision_id] = new
        st.overrides.append(PresentationOverride(new.decision_id, deepcopy(d)))
        if clip.ai_decision_id == d.decision_id:
            clip.ai_decision_id = new.decision_id
        for k in clip.keyframes:
            if k.decision_id == d.decision_id:
                k.decision_id = new.decision_id
        out.append(new)
    return out


def own_decision(asm: PresentationAssembler, d: PresentationDecision) -> PresentationDecision:
    """Same for a decision that is not tied to one clip (ducking, keywords)."""
    if d.created_by is Creator.USER:
        return d
    st = asm.state
    new = deepcopy(d)
    new.decision_id, new.created_by, new.overrides_decision_id = asm.new_id(), Creator.USER, d.decision_id
    del st.decisions[d.decision_id]
    st.decisions[new.decision_id] = new
    st.overrides.append(PresentationOverride(new.decision_id, deepcopy(d)))
    for t in st.timeline.tracks:
        for c in t.clips:
            for k in c.keyframes:
                if k.decision_id == d.decision_id:
                    k.decision_id = new.decision_id
            if c.ai_decision_id == d.decision_id:
                c.ai_decision_id = new.decision_id
    return new
