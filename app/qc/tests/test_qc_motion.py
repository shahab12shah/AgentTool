"""Motion checker: zoom, pan, rotation and keyframes."""

from __future__ import annotations

import pytest

from app.analysis.models import VisualIntent, VisualType
from app.qc.motion_checker import MotionChecker
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, find, narrate, new_project, qc_ctx, run_checker
from app.timeline.keyframes import Keyframe


def kf(prop, *pts, interp="linear"):
    return [Keyframe(prop, float(t), float(v), interp) for t, v in pts]


def project(tmp_path, keyframes=(), *, duration=10.0, **clip_kw):
    p = new_project(tmp_path, seconds=20)
    s1 = add_scene(p, 0, 10, "Silver prices rose sharply last week.")
    s2 = add_scene(p, 10, 20, "Industrial demand is the main reason.")
    narrate(p)
    a = add_asset(p, "city.mp4", "video", duration=60)
    c = add_clip(p, "track_v1", a, 0, duration, scene=s1, created_by=clip_kw.pop("created_by", "AI"), **clip_kw)
    c.keyframes = list(keyframes)
    return p, c, s1, s2, a


def run(p, **kw):
    return run_checker(MotionChecker(), qc_ctx(p, **kw))


def test_the_spec_example_is_flagged_with_a_safe_alternative(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (0.4, 1.35)))
    out = run(p)
    i = find(out, "motion.abrupt_zoom")
    assert len(i) == 1 and i[0].title == "Potentially excessive zoom speed" and i[0].severity is Severity.WARNING
    assert "0.88 per second" in i[0].description and "1.00 -> 1.35" in i[0].description
    assert "over" in i[0].recommended_value and i[0].fix.kind == "motion.soften"
    # the offer: same endpoints, stretched in time until it is under the limit; confirmation required, never automatic
    new = i[0].fix.params["keyframes"]
    assert [k["property"] for k in new] == ["scale", "scale"]
    rate = abs(new[1]["value"] - new[0]["value"]) / (new[1]["time"] - new[0]["time"])
    assert rate <= qc_ctx(p).settings.motion.max_scale_per_second
    assert i[0].fix.needs_confirmation and not i[0].fix.safe and not i[0].auto_fix_safe
    assert out.metrics["max_scale_rate"] == pytest.approx(0.875) and out.metrics["clips_with_motion"] == 1


def test_gentle_zoom_and_unanimated_clips_are_clean(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (5, 1.10)))
    assert run(p).issues == []
    p2, *_ = project(tmp_path / "static")
    out = run(p2)
    assert out.issues == [] and out.metrics["clips_with_motion"] == 0


def test_total_zoom_beyond_the_limit(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (10, 1.9)))
    i = find(run(p), "motion.excessive_zoom")
    assert len(i) == 1 and "1.90x" in i[0].description and i[0].fix.kind == "motion.soften"
    assert max(k["value"] for k in i[0].fix.params["keyframes"]) < 1.9


def test_zoom_that_pumps_in_and_out(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (0.6, 1.1), (1.2, 1.0), (1.8, 1.1)))
    i = find(run(p), "motion.abrupt_zoom")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and "pumps" in i[0].title


def test_fast_rotation(tmp_path):
    p, c, *_ = project(tmp_path, kf("rotation", (0, 0), (0.5, 45)))
    i = find(run(p), "motion.rotation")
    assert len(i) == 1 and "90 degrees per second" in i[0].description
    p2, *_ = project(tmp_path / "slow", kf("rotation", (0, 0), (5, 10)))
    assert find(run(p2), "motion.rotation") == []


def test_keyframe_that_makes_the_picture_jump(tmp_path):
    p, c, *_ = project(tmp_path, kf("position_x", (1.0, 0), (1.02, 700)))
    i = find(run(p), "motion.keyframe_jump")
    assert len(i) == 1 and "less than one frame" in i[0].description and i[0].start_time == pytest.approx(1.0)
    p2, *_ = project(tmp_path / "smooth", kf("position_x", (1.0, 0), (3.0, 700)))
    assert find(run(p2), "motion.keyframe_jump") == []


def test_movement_that_runs_past_the_clip(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (14, 1.2)))
    assert find(run(p), "motion.animation_overrun")
    p2, c2, *_ = project(tmp_path / "anim", duration=2.0)
    c2.animation = {"in": {"preset": "fade", "duration": 3.0, "delay": 0.0, "easing": "linear"}}
    assert find(run(p2), "motion.animation_overrun")
    c2.animation = {"in": {"preset": "fade", "duration": 0.5, "delay": 0.0, "easing": "linear"}}
    assert find(run(p2), "motion.animation_overrun") == []


def test_movement_that_shows_empty_edges(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (5, 0.8)))
    i = find(run(p), "motion.off_image")
    assert len(i) == 1 and "0.80x" in i[0].description
    p2, c2, *_ = project(tmp_path / "pan", kf("position_x", (0, 0), (5, 300)))  # 300 px pan at 1.0x on a picture that exactly fills the frame
    assert len(find(run(p2), "motion.off_image")) == 1
    c2.effects["fit"] = "contain"
    assert find(run(p2), "motion.off_image") == []
    p3, *_ = project(tmp_path / "ok", kf("scale", (0, 1.2), (5, 1.3)) + kf("position_x", (0, 0), (5, 100)))
    assert find(run(p3), "motion.off_image") == []


def test_motion_under_text_hurts_reading(tmp_path):
    p, c, s1, *_ = project(tmp_path, kf("scale", (0, 1.0), (4, 1.8)))  # 0.2 per second
    assert find(run(p), "motion.readability") == []
    add_clip(p, "track_v6", None, 1.0, 3.0, scene=s1, kind="caption", text={"words": [], "text": "x"})
    i = find(run(p), "motion.readability")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and "a caption" in i[0].description
    p2, c2, s12, *_ = project(tmp_path / "t", kf("scale", (0, 1.0), (4, 1.8)))
    add_clip(p2, "track_v5", None, 1.0, 3.0, scene=s12, kind="text", text={"text": "5%"})
    assert find(run(p2), "motion.readability")[0].severity is Severity.WARNING
    p3, c3, s13, *_ = project(tmp_path / "slow", kf("scale", (0, 1.0), (10, 1.2)))
    add_clip(p3, "track_v6", None, 1.0, 3.0, scene=s13, kind="caption", text={"words": [], "text": "x"})
    assert find(run(p3), "motion.readability") == []


def test_movement_on_a_very_short_clip_or_on_evidence(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (0.5, 1.15)), duration=0.6)
    i = find(run(p), "motion.unnecessary")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and i[0].fix is None
    p2, c2, s1, *_ = project(tmp_path / "ev", kf("scale", (0, 1.0), (8, 1.3)))
    p2.visual_intents[s1.id] = VisualIntent(s1.id, VisualType.EVIDENCE)
    assert any("has to be read" in i.title for i in find(run(p2), "motion.unnecessary"))
    c2.effects["highlight"] = True  # a deliberate evidence focus
    assert find(run(p2), "motion.unnecessary") == []


def test_user_owned_clip_is_reported_but_never_changed(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (0.4, 1.35)), created_by="USER")
    i = find(run(p), "motion.abrupt_zoom")[0]
    assert i.locked and not i.auto_fix_available and not i.auto_fix_safe and "you" in i.fix_blocked_reason.lower()


def test_locked_track_blocks_the_fix(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (0.4, 1.35)))
    p.timeline.get_track("track_v1").locked = True
    assert not find(run(p), "motion.abrupt_zoom")[0].auto_fix_available


def test_scene_local_contract(tmp_path):
    p, c, s1, s2, a = project(tmp_path, kf("scale", (0, 1.0), (0.4, 1.35)))
    c2 = add_clip(p, "track_v1", a, 10, 10, scene=s2, created_by="AI")
    c2.keyframes = kf("scale", (0, 1.0), (0.4, 1.4))
    ck = MotionChecker()
    assert ck.scene_local and ck.settings_sections == ("motion",)
    both = run(p)
    assert sorted({i.scene_id for i in both.issues}) == sorted([s1.id, s2.id])
    only2 = run_checker(ck, qc_ctx(p, scene_filter=[s2.id]))
    assert {i.scene_id for i in only2.issues} == {s2.id}
    ctx = qc_ctx(p)
    h2 = ck.scene_input_hash(ctx, s2.id)
    c2.keyframes = kf("scale", (0, 1.0), (4.0, 1.1))
    assert ck.scene_input_hash(qc_ctx(p), s2.id) != h2  # editing a scene's keyframes invalidates that scene's cached result


def test_invalid_values_are_left_to_the_timeline_checker(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, -1.0), (1, -2.0)))
    assert find(run(p), "motion.excessive_zoom") == []  # a negative scale keyframe is an invalid value: reported by the timeline checker, not measured here
    c.scale = 0.0
    run(p)  # must not raise


def test_does_not_modify_the_project(tmp_path):
    p, c, *_ = project(tmp_path, kf("scale", (0, 1.0), (0.4, 1.35)))
    before = p.to_document()
    run(p)
    assert p.to_document() == before


def test_a_planned_evidence_focus_is_not_judged_as_decoration(tmp_path):
    """The Phase 4 evidence focus (zoom to the region, hold, return) is the engine's own signature move: its depth and speed are the point of it. Its keyframes are owned by an
    EVIDENCE_FOCUS decision that targets the clip; the same keyframes without that decision, or a number card shown while it moves, are still findings."""
    from app.editing.models import Creator, DecisionType, EditingDecision
    from app.timeline.clip import KIND_GRAPHIC, KIND_TEXT, Clip

    focus = [Keyframe("scale", 0.0, 1.0, "ease_in_out", "dec_f"), Keyframe("scale", 0.7, 1.8, "ease_in_out", "dec_f")]
    p, c, s1, s2, a = project(tmp_path, focus, duration=6.0)
    out = run(p)
    assert find(out, "motion.excessive_zoom") and find(out, "motion.abrupt_zoom")  # no decision behind it: ordinary decoration rules apply
    p.editing_decisions["dec_f"] = EditingDecision("dec_f", s1.id, DecisionType.EVIDENCE_FOCUS, "focus", c.id, 0.0, 6.0, {}, "Evidence document: zoom to the relevant area.", 64.0, Creator.AI)
    hl = Clip("hl1", "track_v4", "", 0.2, 3.0, kind=KIND_GRAPHIC, effects={"highlight": {"region": [0.1, 0.3, 0.6, 0.2]}}, metadata={"evidence_decision": "dec_f", "host_clip": c.id})
    p.timeline.get_track("track_v4").clips.append(hl)
    out = run(p)
    assert [i for i in out.issues if i.code.startswith("motion.")] == []  # zoom depth, zoom speed and "moves under its own highlight" are all the plan
    card = Clip("tx1", "track_v5", "", 0.2, 1.0, kind=KIND_TEXT, text={"content": "42%", "style": "NUMBER_CARD"})
    p.timeline.get_track("track_v5").clips.append(card)
    assert [i.code for i in run(p).issues] == ["motion.readability"]  # a figure the viewer has to read while the picture moves is still reported
    c.keyframes = [Keyframe(k.property, k.time, k.value, k.interpolation, "") for k in focus]  # the user re-keyed it: their own keyframes, judged as usual
    assert find(run(p), "motion.abrupt_zoom")
