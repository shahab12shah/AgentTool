"""Phase 4 engine tests: shot timing, visual segmentation, motion, keyframes, numbers, transitions, audio, composer."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

from app.analysis.models import NumberKind, NumericMention
from app.editing.context import build_context
from app.editing.models import DecisionType, EditingSettings, SceneEditingBrief, TransitionType
from app.editing.planners import MotionPlanner, TextPlanner, evidence_keyframes, motion_keyframes
from app.editing.presets import PRESETS, preset_for
from app.editing.strategy import EditingStrategyService, RuleBasedProvider
from app.editing.timing import ShotTimingService, fit_source, narration_stats, speed_class
from app.editing.models import MotionPlan
from app.timeline.keyframes import Keyframe, value_at
from app.tests.conftest import needs_ffmpeg
from app.transcription.models import Word

pytestmark = needs_ffmpeg


def words(spec):
    return [Word(f"w{i}", "x", a, b) for i, (a, b) in enumerate(spec)]


# ---------------------------------------------------------------- keyframes
def test_keyframe_interpolation_modes():
    kfs = [Keyframe("scale", 0.0, 1.0, "linear"), Keyframe("scale", 4.0, 1.12, "linear")]
    assert value_at(kfs, "scale", 0.0) == 1.0 and value_at(kfs, "scale", 4.0) == 1.12
    assert value_at(kfs, "scale", 2.0) == pytest.approx(1.06)
    assert value_at(kfs, "scale", -1) == 1.0 and value_at(kfs, "scale", 9) == 1.12  # holds outside the range
    ease_in = [Keyframe("opacity", 0, 0, "ease_in"), Keyframe("opacity", 2, 1)]
    ease_out = [Keyframe("opacity", 0, 0, "ease_out"), Keyframe("opacity", 2, 1)]
    inout = [Keyframe("opacity", 0, 0, "ease_in_out"), Keyframe("opacity", 2, 1)]
    assert value_at(ease_in, "opacity", 1) < 0.5 < value_at(ease_out, "opacity", 1)
    assert value_at(inout, "opacity", 1) == pytest.approx(0.5)
    assert value_at([], "volume", 1.0) == 1.0  # property defaults


def test_keyframe_validation_and_roundtrip():
    k = Keyframe("scale", 1.0, 1.1, "ease_in", "dec_00001")
    assert Keyframe.from_dict(k.to_dict()) == k
    assert Keyframe("bogus", 0, 1).problems(4) and Keyframe("scale", 9, 1).problems(4) and Keyframe("scale", 0, float("nan")).problems(4)
    assert not Keyframe("blur", 0, 2).problems(4)


# ---------------------------------------------------------------- narration analysis & shot timing
def test_narration_speed_and_pauses():
    fast = narration_stats(words([(i * 0.25, i * 0.25 + 0.2) for i in range(20)]), 0, 5)
    slow = narration_stats(words([(i * 0.6, i * 0.6 + 0.4) for i in range(20)]), 0, 12)
    assert fast.wps > 3.1 and speed_class(fast.wps) == "FAST" and slow.wps < 2.1 and speed_class(slow.wps) == "SLOW"
    paused = narration_stats(words([(0, 0.3), (0.35, 0.6), (2.6, 2.9), (2.95, 3.2)]), 0, 4)
    assert paused.longest_pause == pytest.approx(2.0) and paused.pause_total >= 2.0
    assert paused.wps > 2.0  # the silence does not make the speaker "slow"
    assert narration_stats([], 0, 3).words == 0


def test_shot_duration_follows_narration_not_a_constant(edit_ws):
    ctx = build_context(edit_ws.project)
    sc = ctx.scenes[0]
    svc = ShotTimingService(preset_for(ctx.settings), ctx.settings)
    base = SceneEditingBrief(sc.scene.id, importance=0.5, narration_speed=2.5, visual_complexity=0.3, information_density=0.3)
    normal = svc.target_shot(base, sc)
    fast = svc.target_shot(SceneEditingBrief(sc.scene.id, importance=0.5, narration_speed=3.6, visual_complexity=0.3, information_density=0.3), sc)
    slow = svc.target_shot(SceneEditingBrief(sc.scene.id, importance=0.5, narration_speed=1.7, visual_complexity=0.3, information_density=0.3), sc)
    chart = svc.target_shot(SceneEditingBrief(sc.scene.id, importance=0.5, narration_speed=2.5, visual_complexity=0.9, information_density=0.8,
                                              evidence_treatment_needed=True), sc)
    assert fast < normal < slow and chart > normal
    assert len({round(x, 1) for x in (fast, normal, slow, chart)}) == 4  # never "every visual = N seconds"


def test_presets_and_pacing_slider_change_shot_length(edit_ws):
    ctx = build_context(edit_ws.project)
    sc = ctx.scenes[1]
    b = SceneEditingBrief(sc.scene.id, narration_speed=2.5, visual_complexity=0.3, information_density=0.3)
    lens = {}
    for name in PRESETS:
        s = EditingSettings(style=name)
        lens[name] = ShotTimingService(PRESETS[name], s).target_shot(b, sc)
    assert lens["dynamic"] < lens["professional"] < lens["documentary"]
    slow = ShotTimingService(PRESETS["professional"], EditingSettings(pacing=0.0)).target_shot(b, sc)
    fast = ShotTimingService(PRESETS["professional"], EditingSettings(pacing=1.0)).target_shot(b, sc)
    assert fast < slow


def test_long_sentence_scene_is_cut_at_word_boundaries_and_tiles_exactly(edit_ws):
    ctx = build_context(edit_ws.project)
    sc = max(ctx.scenes, key=lambda c: c.scene.duration)  # the longest scene
    svc = ShotTimingService(preset_for(ctx.settings), ctx.settings)
    plan = RuleBasedProvider().plan_scene(sc, ctx, RuleBasedProvider().analyze_video(ctx))
    segs = [p.segment for p in plan.segments]
    assert len(segs) >= 2
    assert segs[0].start == pytest.approx(sc.scene.start) and segs[-1].start + segs[-1].duration == pytest.approx(sc.scene.end, abs=1e-3)
    for a, b in zip(segs, segs[1:]):
        assert a.start + a.duration == pytest.approx(b.start, abs=1e-3)  # no gap, no overlap
    starts = {round(w.start, 2) for w in sc.words} | {round(s.start, 2) for s in sc.sentences}
    for seg in segs[1:]:
        near = min(abs(seg.start - w.start) for w in sc.words)
        assert near <= 0.16  # cut on (or just before) a word, never mid-word
    assert starts


def test_short_scene_gets_a_single_hold(edit_ws):
    ctx = build_context(edit_ws.project)
    sc = min(ctx.scenes, key=lambda c: c.scene.duration)
    plan = RuleBasedProvider().plan_scene(sc, ctx, RuleBasedProvider().analyze_video(ctx))
    assert len(plan.segments) == 1 and plan.segments[0].segment.reason


def test_fit_source_is_non_destructive():
    from app.editing.context import AssetInfo
    from app.media.asset import AssetType

    vid = AssetInfo("a", AssetType.VIDEO, 12.0, 640, 360, "STOCK_VIDEO")
    f = fit_source(vid, 5.0, 2.0)
    assert (f.source_in, f.source_out, f.operation) == (2.0, 7.0, "TRIM")  # original 0-12 untouched, timeline uses 2-7
    f = fit_source(vid, 5.0, 10.0)
    assert f.source_out <= 12.0 and f.source_out - f.source_in == pytest.approx(5.0)
    short = AssetInfo("b", AssetType.VIDEO, 3.0, 640, 360, "STOCK_VIDEO")
    assert fit_source(short, 3.6, 0.0).operation == "EXTEND" and fit_source(short, 3.6, 0.0).speed < 1.0
    assert fit_source(short, 9.0, 0.0).operation == "SHORTEN" and fit_source(short, 9.0, 0.0).duration == 3.0
    still = AssetInfo("c", AssetType.IMAGE, None, 100, 100, "STOCK_IMAGE")
    assert fit_source(still, 4.0, 0.0).operation == "HOLD"


# ---------------------------------------------------------------- planning a whole scene
def provider_plan(ctx, sc):
    p = RuleBasedProvider()
    return p.plan_scene(sc, ctx, p.analyze_video(ctx))


def test_scene_brief_is_structured_and_concise(edit_ws):
    ctx = build_context(edit_ws.project)
    plan = provider_plan(ctx, ctx.scenes[2])
    b = plan.brief
    assert 0 <= b.importance <= 1 and b.speed_class in ("SLOW", "NORMAL", "FAST") and b.factors and all(len(f) < 80 for f in b.factors)
    assert b.recommended_transition in {t.value for t in TransitionType}
    assert isinstance(b.should_zoom, bool) and isinstance(b.text_needed, bool) and isinstance(b.needs_visual_change, bool)


def test_motion_choice_depends_on_composition_and_is_not_applied_to_every_clip(edit_ws):
    ctx = build_context(edit_ws.project)
    plans = [provider_plan(ctx, sc) for sc in ctx.scenes]
    kinds = [s.motion.kind if s.motion else "NONE" for p in plans for s in p.segments]
    assert "NONE" in kinds and any(k.startswith("PAN_") for k in kinds)  # some clips are static, wide photos pan
    assert sum(k != "NONE" for k in kinds) < len(kinds)
    wide = next(p for p in plans for s in p.segments if s.motion and s.motion.kind in ("PAN_LEFT", "PAN_RIGHT"))
    m = next(s.motion for s in wide.segments if s.motion and s.motion.kind.startswith("PAN_"))
    assert m.start_pos[0] == -m.end_pos[0] != 0 and m.start_scale == m.end_scale > 1.0  # travels across, never leaves the frame
    W = ctx.canvas[0]
    assert abs(m.start_pos[0]) <= (m.start_scale - 1) * W / 2 + 1e-6  # the pan stays inside the oversize margin: nothing is cropped away


def test_motion_intensity_slider_controls_amount_of_movement(edit_ws):
    p = edit_ws.project

    def count(intensity):
        ctx = build_context(p, EditingSettings(motion_intensity=intensity))
        pl = [provider_plan(ctx, sc) for sc in ctx.scenes]
        return sum(1 for x in pl for s in x.segments if s.motion)

    assert count(0.0) < count(1.0)


def test_important_statements_get_a_stronger_punch_in_than_ordinary_ones(edit_ws):
    ctx = build_context(edit_ws.project)
    sc = next(c for c in ctx.scenes if c.asset and c.asset.is_still and c.asset.asset_id != edit_ws.assets_by["doc.png"].id)
    planner = MotionPlanner(preset_for(ctx.settings), ctx.settings)
    plan = provider_plan(ctx, sc)
    seg = plan.segments[0].segment
    hi = SceneEditingBrief(sc.scene.id, importance=0.9)
    lo = SceneEditingBrief(sc.scene.id, importance=0.3)
    seg.duration = 4.0
    m_hi = planner.plan(seg, sc.asset, sc, hi, 0, ctx.canvas, False)
    assert m_hi and m_hi.kind == "PUNCH_IN"
    punch = m_hi.end_scale - 1
    ordinary = [planner.plan(seg, sc.asset, sc, lo, i, ctx.canvas, False) for i in range(12)]
    assert all(o is None or o.end_scale - 1 < punch or o.start_scale - 1 < punch for o in ordinary)


def test_motion_and_evidence_keyframes_are_editable_and_interpolate():
    m = MotionPlan("SUBTLE_ZOOM", "ZOOM", 1.0, 1.12)
    kfs = motion_keyframes(m, 4.0, "dec_1")
    assert [(k.property, k.time, k.value) for k in kfs] == [("scale", 0.0, 1.0), ("scale", 4.0, 1.12)] and all(k.decision_id == "dec_1" for k in kfs)
    assert motion_keyframes(MotionPlan("NO_ZOOM"), 4.0) == []
    from app.editing.models import EvidencePlan

    ev = evidence_keyframes(EvidencePlan(region=(0.1, 0.2, 0.5, 0.2), zoom_scale=2.0), 6.0, (1920, 1080))
    scales = sorted((k.time, k.value) for k in ev if k.property == "scale")
    assert scales[0][1] == 1.0 and max(v for _, v in scales) == 2.0 and scales[-1][1] == 1.0  # wide -> zoom into the region -> back to wide
    assert value_at(ev, "scale", 3.0) == 2.0 and value_at(ev, "scale", 6.0) == 1.0
    short = evidence_keyframes(EvidencePlan(), 2.5, (1920, 1080))
    assert value_at(short, "scale", 2.5) > 1.0  # too short to return: ends on the detail


# ---------------------------------------------------------------- numbers and text
def test_important_numbers_come_only_from_the_narration(edit_ws):
    ctx = build_context(edit_ws.project)
    plans = [provider_plan(ctx, sc) for sc in ctx.scenes]
    numbers = [(sc, t) for sc, p in zip(ctx.scenes, plans) for t in p.texts if t.decision_type == DecisionType.NUMBER_EMPHASIS.value]
    assert numbers
    for sc, t in numbers:
        text = re.sub(r"[^a-z0-9%$]+", " ", (sc.scene.narration + " " + sc.scene.script_text).lower())
        assert all(tok in text for tok in re.sub(r"[^a-z0-9%$]+", " ", t.graphic.content.lower()).split())
        assert t.graphic.source_scene == sc.scene.id and t.graphic.source_ref in ("script", "transcript")
        assert sc.scene.start <= t.graphic.start and t.graphic.start + t.graphic.duration <= sc.scene.end + 1e-6


def test_fabricated_numbers_are_rejected(edit_ws):
    ctx = build_context(edit_ws.project)
    sc = deepcopy(ctx.scenes[0])
    sc.scene.numbers = [NumericMention("$999,999", NumberKind.DOLLAR_AMOUNT, 999999.0, "", False, [])]
    texts = TextPlanner(preset_for(ctx.settings), ctx.settings).plan(sc, SceneEditingBrief(sc.scene.id, importance=0.9), [], 1.0)
    assert all("999" not in t.graphic.content for t in texts)


def test_text_emphasis_respects_toggles_and_is_not_on_every_scene(edit_ws):
    p = edit_ws.project
    on = build_context(p)
    off = build_context(p, EditingSettings(text_emphasis=False, number_emphasis=False))
    n_on = sum(len(provider_plan(on, sc).texts) for sc in on.scenes)
    n_off = sum(len(provider_plan(off, sc).texts) for sc in off.scenes)
    assert n_off == 0 and 0 < n_on < len(on.scenes) * 2
    assert sum(1 for sc in on.scenes if provider_plan(on, sc).texts) < len(on.scenes)  # not every scene


def test_text_graphics_are_fully_described_and_serialisable(edit_ws):
    ctx = build_context(edit_ws.project)
    t = next(t for sc in ctx.scenes for t in provider_plan(ctx, sc).texts)
    g = t.graphic
    for field in ("text_id", "content", "start", "duration", "font", "size", "position", "alignment", "opacity", "animation", "background", "style",
                  "importance", "source_scene"):
        assert getattr(g, field) is not None
    from app.editing.models import TextGraphic

    assert TextGraphic.from_dict(g.to_dict()) == g


# ---------------------------------------------------------------- evidence, transitions, audio, captions
def test_evidence_treatment_is_flagged_for_review_and_never_alters_the_document(edit_ws):
    ctx = build_context(edit_ws.project)
    plans = [provider_plan(ctx, sc) for sc in ctx.scenes]
    ev = [s.evidence for p in plans for s in p.segments if s.evidence]
    assert ev and all(not e.region_detected and e.confidence < 70 and "not detected" in e.reason for e in ev)  # honest: no region was located
    off = build_context(edit_ws.project, EditingSettings(evidence_treatment=False))
    assert not [s for sc in off.scenes for s in provider_plan(off, sc).segments if s.evidence]
    for p in plans:  # document visuals are never cropped: fit stays "contain"
        for s in p.segments:
            if s.evidence:
                assert s.segment.fit == "contain"


def test_transitions_default_to_cut_and_are_never_between_every_scene(edit_ws):
    ctx = build_context(edit_ws.project)
    svc = EditingStrategyService()
    prov = RuleBasedProvider()
    profile = prov.analyze_video(ctx)
    plans = [svc.plan_scene(prov, sc, ctx, profile) for sc in ctx.scenes]
    svc.limit_transitions(plans, ctx)
    types = [p.segments[0].transition.type for p in plans if p.segments]
    assert types[0] == "CUT" and types.count("CUT") >= len(types) * 0.6
    assert set(types) - {"CUT"} <= set(preset_for(ctx.settings).transitions)
    for a, b in zip(types, types[1:]):
        assert not (a != "CUT" and b != "CUT")  # no back-to-back transitions
    for p in plans:
        t = p.segments[0].transition
        assert t.reason and 0 <= t.confidence <= 100 and (t.type == "CUT") == (t.duration == 0)


def test_smart_transitions_off_means_cuts_only(edit_ws):
    ctx = build_context(edit_ws.project, EditingSettings(smart_transitions=False, style="dynamic", transition_frequency=1.0))
    assert all(provider_plan(ctx, sc).segments[0].transition.type == "CUT" for sc in ctx.scenes)


def test_audio_instructions_put_the_voice_first(edit_ws):
    ctx = build_context(edit_ws.project)
    plans = [provider_plan(ctx, sc) for sc in ctx.scenes]
    ducks = [d for p in plans for d in p.ducks]
    assert ducks and all(d.music_level <= 0.3 and d.reason for d in ducks)
    from app.editing.planners import AudioPlanner

    ap = AudioPlanner(preset_for(ctx.settings), ctx.settings).global_plan()
    assert ap.priority == ["VOICE", "SFX", "MUSIC"] and ap.duck_level < ap.music_level < ap.rise_level and ap.voice_level == 1.0
    assert any(d.kind == "DUCK" for d in ducks)
    off = build_context(edit_ws.project, EditingSettings(smart_audio_ducking=False))
    assert not [d for sc in off.scenes for d in provider_plan(off, sc).ducks]


def test_caption_instructions_are_data_not_burned_in(edit_ws):
    ctx = build_context(edit_ws.project)
    plans = [provider_plan(ctx, sc) for sc in ctx.scenes]
    assert any(p.caption_emphasis for p in plans)
    assert all(p.caption_region in ("bottom_safe_area", "bottom_safe_area_raised") for p in plans)


def test_reuse_of_a_visual_is_tracked_and_not_forced(edit_ws):
    from app.project.phase3_commands import SceneDecisionCommand
    import copy

    ws = edit_ws
    p = ws.project
    a0 = p.visual_assignments[p.scenes[0].id]
    for sc in p.scenes[1:3]:
        ws.apply_command(SceneDecisionCommand(p, sc.id, "same", assignment=copy.deepcopy(a0).__class__(**{**a0.__dict__, "scene_id": sc.id})))
    ctx = build_context(p)
    assert ctx.scenes[1].reuse_count == 1 and ctx.scenes[2].reuse_count == 2 and ctx.scenes[1].previous_scene_with_asset == p.scenes[0].id
    plan = provider_plan(ctx, ctx.scenes[1])
    seg = plan.segments[0].segment
    assert seg.reuse_count == 1 and seg.previous_scene_id == p.scenes[0].id and seg.reuse_reason and seg.confidence <= 80


def test_unchanged_scenes_have_stable_hashes_and_changes_are_detected(edit_ws):
    p = edit_ws.project
    a, b = build_context(p), build_context(p)
    assert [s.input_hash for s in a.scenes] == [s.input_hash for s in b.scenes]
    c = build_context(p, EditingSettings(style="dynamic"))
    assert all(x.input_hash != y.input_hash for x, y in zip(a.scenes, c.scenes))


def test_missing_visual_is_never_invented(edit_ws):
    ws = edit_ws
    p = ws.project
    sid = p.scenes[3].id
    del p.visual_assignments[sid]
    ctx = build_context(p)
    sc = ctx.by_id(sid)
    assert sc.visual_status == "MISSING" and sc.asset is None
    plan = provider_plan(ctx, sc)
    assert plan.segments == [] and any("nothing was invented" in n for n in plan.notes)
