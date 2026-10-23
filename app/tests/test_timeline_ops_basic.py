"""Phase 10 / W1-A: basic edit planners (move, numeric edits, trim, split, delete, duplicate, overwrite, replace, insert) and the clipboard.

Every planner is exercised through ``PlanCommand`` (validate -> apply -> undo -> redo identity) on timelines built directly, without a project or ffmpeg."""

from __future__ import annotations

import copy
import random

import pytest

from app.core.constants import MIN_CLIP_DURATION
from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.clipboard import (
    ClipboardData,
    copy_attributes,
    plan_paste,
    plan_paste_attributes,
)
from app.timeline.keyframes import Keyframe, value_at
from app.timeline.ops_basic import (
    plan_delete,
    plan_duplicate,
    plan_insert_free,
    plan_move,
    plan_overwrite,
    plan_replace,
    plan_set_duration,
    plan_set_source_range,
    plan_set_start,
    plan_split,
    plan_trim,
)
from app.timeline.plan import PlanCommand, PlanEnv
from app.timeline.timebase import Timebase
from app.timeline.timeline import Timeline

ASSETS = {
    "v10": ("video", 10.0, "ten.mp4"),
    "v4": ("video", 4.0, "four.mp4"),
    "v30": ("video", 30.0, "thirty.mp4"),
    "img": ("image", None, "pic.png"),
    "a20": ("audio", 20.0, "voice.wav"),
    "a5": ("audio", 5.0, "sfx.wav"),
}
FPS = [30, 24, 29.97]


# ------------------------------------------------------------------------------------------------------------------------ helpers
def make_env(fps=30) -> PlanEnv:
    return PlanEnv(Timeline.default(), ASSETS.get, Timebase(fps))


def add(env, cid, start, dur, *, asset="v10", track="track_v1", si=0.0, speed=1.0, **kw) -> Clip:
    c = Clip(cid, track, asset, start, dur, source_in=si, source_out=si + dur * speed, speed=speed, **kw)
    env.timeline.insert_clip(c)
    return c


def state(tl: Timeline):
    return [(t.id, [c.to_dict() for c in t.clips]) for t in tl.tracks]


def clip_of(env, cid) -> Clip:
    return env.timeline.find_clip(cid)[1]


def apply(env, plan) -> PlanCommand:
    """Run a plan and prove the command contract: undo restores the exact previous state, redo reproduces the exact result."""
    before = state(env.timeline)
    cmd = PlanCommand(env, plan)
    cmd.do()
    after = state(env.timeline)
    cmd.undo()
    assert state(env.timeline) == before, "undo must restore the exact previous timeline"
    cmd.do()
    assert state(env.timeline) == after, "redo must reproduce the exact result"
    return cmd


def rejects(env, plan, match=None):
    """The plan must be refused by PlanCommand and leave the timeline untouched."""
    before = state(env.timeline)
    with pytest.raises(TimelineError, match=match):
        PlanCommand(env, plan).do()
    assert state(env.timeline) == before


def span(c: Clip):
    return (round(c.timeline_start, 6), round(c.timeline_end, 6))


def check_invariants(env: PlanEnv) -> None:
    tl = env.timeline
    seen: set[str] = set()
    for t in tl.tracks:
        prev_end = 0.0
        for c in sorted(t.clips, key=lambda c: c.timeline_start):
            assert c.id not in seen
            seen.add(c.id)
            assert c.timeline_start >= -1e-9
            assert c.duration >= MIN_CLIP_DURATION - 1e-9
            assert c.timeline_start >= prev_end - 1e-6, f"overlap on {t.id}"
            prev_end = c.timeline_end
            if c.kind == "media":
                info = ASSETS[c.asset_id]
                if info[0] != "image":
                    assert c.source_in >= -1e-3
                    assert abs((c.source_out - c.source_in) - c.duration * c.speed) <= 1e-3
                    assert c.source_out <= info[1] + 1e-3
            for k in c.keyframes:
                assert -1e-6 <= k.time <= c.duration + 1e-6


# ------------------------------------------------------------------------------------------------------------------------ move
@pytest.mark.parametrize("fps", FPS)
def test_move_delta_is_whole_frames(fps):
    env = make_env(fps)
    tb = env.timebase
    add(env, "a", tb.to_seconds(30), tb.to_seconds(60))
    add(env, "b", tb.to_seconds(100), tb.to_seconds(40))
    apply(env, plan_move(env, ["a", "b"], 0.1234))
    n = round(0.1234 * fps)
    plan = plan_move(env, ["a", "b"], 0.1234)
    apply(env, plan)  # a second time, to prove the grid is kept
    a, b = clip_of(env, "a"), clip_of(env, "b")
    assert tb.to_frame(a.timeline_start) == 30 + 2 * n and tb.is_on_frame(a.timeline_start)
    assert tb.to_frame(b.timeline_start) == 100 + 2 * n and tb.is_on_frame(b.timeline_start)


def test_move_multi_clip_keeps_relative_spacing_spec_example():
    env = make_env(30)
    add(env, "c1", 10, 2)
    add(env, "c2", 13, 2)
    add(env, "c3", 16, 2)
    plan = plan_move(env, ["c1", "c2", "c3"], 5)
    assert plan.description == "Move clips" and not plan.notes
    apply(env, plan)
    assert [span(clip_of(env, i)) for i in ("c1", "c2", "c3")] == [(15, 17), (18, 20), (21, 23)]


def test_move_selection_onto_its_own_old_positions():
    env = make_env(30)
    for i, s in enumerate((0, 2, 4)):
        add(env, f"c{i}", s, 2)
    apply(env, plan_move(env, ["c0", "c1", "c2"], 2))  # c0 lands on c1's old slot, c1 on c2's, c2 beyond
    assert [span(clip_of(env, f"c{i}")) for i in range(3)] == [(2, 4), (4, 6), (6, 8)]
    apply(env, plan_move(env, ["c0", "c1", "c2"], -2))
    assert [span(clip_of(env, f"c{i}")) for i in range(3)] == [(0, 2), (2, 4), (4, 6)]


def test_move_onto_unselected_clip_is_rejected_atomically():
    env = make_env(30)
    add(env, "a", 0, 2)
    add(env, "b", 3, 2)
    add(env, "other", 6, 2)
    rejects(env, plan_move(env, ["a", "b"], 2.5), "overlap")  # b would land on "other"; a's part must not be applied either


def test_move_before_zero_rejected_or_clamped_for_whole_selection():
    env = make_env(30)
    add(env, "a", 1, 2)
    add(env, "b", 4, 2)
    with pytest.raises(TimelineError, match="before 0:00"):
        plan_move(env, ["a", "b"], -3)
    plan = plan_move(env, ["a", "b"], -3, clamp_at_zero=True)
    assert plan.notes and "start of the timeline" in plan.notes[0]
    apply(env, plan)
    assert span(clip_of(env, "a")) == (0, 2) and span(clip_of(env, "b")) == (3, 5)  # spacing kept


def test_move_noop_is_an_empty_plan():
    env = make_env(30)
    add(env, "a", 1, 2)
    assert plan_move(env, ["a"], 0).empty
    assert plan_move(env, [], 3).empty


def test_audio_move_keeps_sample_precision_but_video_snaps_to_frames():
    env = make_env(30)
    add(env, "au", 1, 2, asset="a20", track="track_a1")
    apply(env, plan_move(env, ["au"], 0.123456))
    assert clip_of(env, "au").timeline_start == pytest.approx(1.123456, abs=1 / 96000)
    assert not env.timebase.is_on_frame(clip_of(env, "au").timeline_start)


def test_move_to_other_track_and_incompatible_tracks():
    env = make_env(30)
    add(env, "v", 0, 2)
    add(env, "au", 0, 2, asset="a20", track="track_a1")
    apply(env, plan_move(env, ["v"], 1, track_moves={"v": "track_v2"}))
    assert clip_of(env, "v").track_id == "track_v2" and clip_of(env, "v").timeline_start == 1
    with pytest.raises(TimelineError, match="does not hold"):
        plan_move(env, ["v"], 0, track_moves={"v": "track_a2"})
    with pytest.raises(TimelineError, match="does not hold"):
        plan_move(env, ["au"], 0, track_moves={"au": "track_v1"})
    env.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_move(env, ["v"], 0, track_moves={"v": "track_v1"})


def test_multi_selection_vertical_move_by_track_offset():
    env = make_env(30)
    add(env, "a", 0, 2, track="track_v1")
    add(env, "b", 5, 2, track="track_v2")
    apply(env, plan_move(env, ["a", "b"], 1, track_moves={"a": "track_v2", "b": "track_v3"}))
    assert (clip_of(env, "a").track_id, clip_of(env, "b").track_id) == ("track_v2", "track_v3")
    assert clip_of(env, "b").timeline_start == 6


def test_linked_and_grouped_clips_follow_a_move():
    env = make_env(30)
    add(env, "v", 2, 3, link_id="L")
    add(env, "au", 2, 3, asset="a20", track="track_a1", link_id="L")
    add(env, "g1", 8, 1, group_id="G")
    add(env, "g2", 9, 1, track="track_v2", group_id="G")
    plan = plan_move(env, ["v"], 1)
    assert set(plan.updates) == {"v", "au"}
    apply(env, plan)
    assert clip_of(env, "au").timeline_start == 3
    assert set(plan_move(env, ["g1"], 1).updates) == {"g1", "g2"}
    assert set(plan_move(env, ["g1"], 1, grouped=False).updates) == {"g1"}
    assert set(plan_move(env, ["v"], 1, linked=False).updates) == {"v"}


def test_locks_reject_move_and_leave_timeline_untouched():
    env = make_env(30)
    add(env, "a", 0, 2)
    add(env, "b", 3, 2, locks=frozenset({"position"}))
    add(env, "c", 6, 2, locks=frozenset({"transform"}))
    add(env, "d", 9, 2, locks=frozenset({"item"}))
    with pytest.raises(TimelineError, match="locked"):
        plan_move(env, ["b"], 1)
    with pytest.raises(TimelineError, match="locked"):
        plan_move(env, ["d"], 1)
    apply(env, plan_move(env, ["c"], 1))  # a transform lock does not stop a move
    env.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_move(env, ["a"], 1)
    # a hand-made plan that bypasses the pre-check is still refused by PlanCommand
    env.timeline.get_track("track_v1").locked = False
    a = clip_of(env, "a").snapshot()
    a.timeline_start = 20
    env.timeline.get_track("track_v1").locked = True
    from app.timeline.plan import EditPlan

    rejects(env, EditPlan("x", updates={"a": a}), "locked")


def test_moving_a_caption_carries_its_word_timing():
    env = make_env(30)
    words = [{"word_id": "w0", "text": "hi", "start": 10.0, "end": 10.4}, {"word_id": "w1", "text": "there", "start": 10.5, "end": 11.0}]
    env.timeline.insert_clip(Clip("cap", "track_v6", "", 10.0, 2.0, kind="caption", text={"text": "hi there", "words": words, "start": 10.0, "end": 12.0}))
    apply(env, plan_move(env, ["cap"], 2))
    t = clip_of(env, "cap").text
    assert t["start"] == 12.0 and t["end"] == 14.0
    assert [(w["start"], w["end"]) for w in t["words"]] == [(12.0, 12.4), (12.5, 13.0)]


# ------------------------------------------------------------------------------------------------------------------------ numeric edits
def test_set_start_snaps_and_linked_follow():
    env = make_env(30)
    add(env, "v", 2, 3, link_id="L")
    add(env, "au", 2, 3, asset="a20", track="track_a1", link_id="L")
    plan = plan_set_start(env, "v", 4.0123)
    apply(env, plan)
    assert clip_of(env, "v").timeline_start == pytest.approx(4.0, abs=1 / 60)
    assert env.timebase.is_on_frame(clip_of(env, "v").timeline_start)
    assert clip_of(env, "au").timeline_start - 2 == pytest.approx(clip_of(env, "v").timeline_start - 2, abs=1e-9)
    with pytest.raises(TimelineError, match="before 0:00"):
        plan_set_start(env, "v", -1)
    assert plan_set_start(env, "v", clip_of(env, "v").timeline_start).empty


def test_set_duration_moves_the_end_edge_and_source_follows():
    env = make_env(30)
    add(env, "a", 1, 4, si=2.0, speed=2.0)  # source 2..10
    apply(env, plan_set_duration(env, "a", 3))
    a = clip_of(env, "a")
    assert (a.timeline_start, a.duration, a.source_in, a.source_out) == (1, 3, 2.0, 8.0)


def test_set_duration_is_limited_by_source_and_next_clip_with_notes():
    env = make_env(30)
    add(env, "a", 0, 4, asset="v10", si=3.0)  # 7 s of source left after source_in
    add(env, "b", 20, 2, track="track_v2")
    p = plan_set_duration(env, "a", 12)
    assert p.updates["a"].duration == pytest.approx(7.0) and "source" in p.notes[0]
    apply(env, p)
    env2 = make_env(30)
    add(env2, "a", 0, 4, asset="v30")
    add(env2, "n", 6, 2)
    p = plan_set_duration(env2, "a", 10)
    assert p.updates["a"].duration == 6 and "next clip" in p.notes[0]
    p = plan_set_duration(env2, "a", 0.01)  # shorter than the minimum -> clamped
    assert p.updates["a"].duration == pytest.approx(env2.timebase.to_seconds(2)) and "minimum" in p.notes[0]
    with pytest.raises(TimelineError):
        plan_set_duration(env2, "a", 0)


def test_set_source_range():
    env = make_env(30)
    add(env, "a", 2, 4)  # source 0..4
    apply(env, plan_set_source_range(env, "a", source_in=1.0))
    a = clip_of(env, "a")
    assert (a.timeline_start, a.source_in, a.source_out, a.duration) == (2, 1.0, 4.0, 3.0)  # end edge moved, start did not
    p = plan_set_source_range(env, "a", source_out=50.0)
    assert p.updates["a"].source_out == 10.0 and "end of the media" in p.notes[0]
    apply(env, p)
    with pytest.raises(TimelineError, match="after the source in"):
        plan_set_source_range(env, "a", source_in=5.0, source_out=4.0)
    with pytest.raises(TimelineError, match="before the start"):
        plan_set_source_range(env, "a", source_in=-1.0)
    add(env, "img", 30, 3, asset="img")
    with pytest.raises(TimelineError, match="no source range"):
        plan_set_source_range(env, "img", source_in=1.0)
    add(env, "nx", 20, 3)
    p = plan_set_source_range(env, "a", source_in=0.0, source_out=10.0)  # would run 2 + 10 = 12 s, but the next clip starts at 20: fits
    assert p.updates["a"].duration == 10.0
    add(env, "nx2", 11, 2, track="track_v1")  # directly after "a" (2..11)
    p = plan_set_source_range(env, "a", source_in=0.0, source_out=10.0)
    assert p.updates["a"].timeline_end <= 11 + 1e-9 and "next clip" in " ".join(p.notes)
    apply(env, p)


# ------------------------------------------------------------------------------------------------------------------------ trim
def test_trim_end_and_start_distinguish_timeline_from_source():
    env = make_env(30)
    add(env, "a", 2, 4, si=1.0, speed=2.0)  # timeline 2..6, source 1..9
    apply(env, plan_trim(env, "a", "end", 5.0))
    a = clip_of(env, "a")
    assert (a.timeline_start, a.duration, a.source_in, a.source_out) == (2, 3, 1.0, 7.0)  # source_in untouched
    apply(env, plan_trim(env, "a", "start", 3.0))
    a = clip_of(env, "a")
    assert (a.timeline_start, a.duration, a.source_in, a.source_out) == (3, 2, 3.0, 7.0)  # source_in += 1 s * speed 2; source_out untouched
    check_invariants(env)


def test_trim_start_extension_is_limited_by_source_and_previous_clip():
    env = make_env(30)
    add(env, "prev", 0, 3)
    add(env, "a", 5, 3, si=1.0)
    p = plan_trim(env, "a", "start", 3.5)  # wants 1.5 s earlier; only 1.0 s of source before source_in
    assert p.updates["a"].timeline_start == 4 and p.updates["a"].source_in == 0 and "source" in p.notes[0]
    apply(env, p)
    with pytest.raises(TimelineError, match="any further"):
        plan_trim(env, "a", "start", 3.9)
    add(env, "b", 20, 3, si=5.0)
    p = plan_trim(env, "b", "start", 1.0)  # 5 s of source available, but "a" ends at 7
    assert p.updates["b"].timeline_start == 15 and "source" in p.notes[0]
    add(env, "c", 40, 3, si=8.0, track="track_v2")
    add(env, "d", 36, 3, track="track_v2")
    p = plan_trim(env, "c", "start", 30)
    assert p.updates["c"].timeline_start == 39 and "previous clip" in p.notes[0]


def test_trim_end_limits_and_minimum_length():
    env = make_env(30)
    add(env, "a", 0, 3, asset="v4")
    add(env, "n", 6, 2)
    p = plan_trim(env, "a", "end", 5.0)
    assert p.updates["a"].duration == 4 and "source" in p.notes[0]  # asset has only 4 s
    add(env, "full", 20, 4, asset="v4", track="track_v2")
    with pytest.raises(TimelineError, match="any further"):
        plan_trim(env, "full", "end", 25.0)
    add(env, "b", 10, 4, asset="v10")
    p = plan_trim(env, "b", "end", 11.0)
    assert p.updates["b"].duration == 1
    p = plan_trim(env, "b", "end", 10.001)
    assert p.updates["b"].duration == pytest.approx(env.timebase.to_seconds(2)) and "minimum" in p.notes[0]
    add(env, "c", 20, 2, asset="img", track="track_v3")
    p = plan_trim(env, "c", "end", 400.0)  # a picture has no source limit
    assert p.updates["c"].duration == 380 and not p.notes
    p = plan_trim(env, "c", "start", 3.0)
    assert p.updates["c"].timeline_start == 3 and not p.notes and (p.updates["c"].source_in, p.updates["c"].source_out) == (0.0, 19.0)
    assert plan_trim(env, "c", "end", 22.0).empty
    with pytest.raises(TimelineError, match="start or the end"):
        plan_trim(env, "c", "middle", 3)


def test_trim_start_past_the_end_stops_at_the_minimum_length():
    env = make_env(30)
    add(env, "a", 0, 4, si=1.0)
    p = plan_trim(env, "a", "start", 9.0)
    assert p.updates["a"].timeline_end == pytest.approx(4.0) and p.updates["a"].duration == pytest.approx(env.timebase.to_seconds(2)) and "minimum" in p.notes[0]
    apply(env, p)
    check_invariants(env)


def test_trim_keeps_text_timing_in_step():
    env = make_env(30)
    _caption(env)
    env.timeline.insert_clip(Clip("g", "track_v5", "", 0.0, 4.0, kind="text", text={"content": "Title", "start": 0.0, "duration": 4.0}))
    apply(env, plan_trim(env, "cap", "end", 12.0))
    t = clip_of(env, "cap").text
    assert (t["start"], t["end"]) == (10.0, 12.0) and t["words"][-1]["end"] == 12.9  # words stay where they were spoken
    apply(env, plan_trim(env, "g", "start", 1.0))
    assert clip_of(env, "g").text["start"] == 1.0 and clip_of(env, "g").text["duration"] == 3.0
    apply(env, plan_set_start(env, "g", 5.0))
    assert clip_of(env, "g").text["start"] == 5.0


def test_trim_at_timeline_start_cannot_go_negative():
    env = make_env(30)
    add(env, "a", 0, 3, si=2.0)
    with pytest.raises(TimelineError, match="start of the timeline"):
        plan_trim(env, "a", "start", -1.0)


@pytest.mark.parametrize("fps", FPS)
def test_trim_snaps_to_frames_and_never_drifts(fps):
    env = make_env(fps)
    tb = env.timebase
    add(env, "a", tb.to_seconds(60), tb.to_seconds(120), si=1.5)
    orig = clip_of(env, "a").snapshot()
    f0, f1 = 60, 71
    for i in range(100):  # alternate start trims, then end trims
        apply_only(env, plan_trim(env, "a", "start", tb.to_seconds(f1 if i % 2 == 0 else f0) + 0.0001))
    a = clip_of(env, "a")
    assert (a.timeline_start, a.source_in, a.source_out) == (orig.timeline_start, orig.source_in, orig.source_out)
    assert a.duration == pytest.approx(orig.duration, abs=1e-12)
    for i in range(100):
        apply_only(env, plan_trim(env, "a", "end", tb.to_seconds(180 - 7 if i % 2 == 0 else 180)))
    a = clip_of(env, "a")
    assert (a.timeline_start, a.source_in) == (orig.timeline_start, orig.source_in)
    assert a.duration == pytest.approx(orig.duration, abs=1e-12) and a.source_out == pytest.approx(orig.source_out, abs=1e-9)
    # repeated moves back and forth are exact too
    for i in range(100):
        apply_only(env, plan_move(env, ["a"], 0.5 if i % 2 == 0 else -0.5))
    assert clip_of(env, "a").timeline_start == orig.timeline_start


def apply_only(env, plan) -> None:
    PlanCommand(env, plan).do()


def test_trim_keyframe_policy_and_preserved_properties():
    env = make_env(30)
    kfs = [Keyframe("opacity", 0.0, 1.0), Keyframe("opacity", 4.0, 0.0), Keyframe("position_x", 1.0, 50.0)]
    add(env, "a", 10, 4, keyframes=kfs, effects={"fit": "contain"}, scale=1.3, rotation=5.0, opacity=0.9, audio={"volume": 0.7},
        animation={"in": {"preset": "fade"}}, transition={"type": "FADE", "duration": 0.5}, metadata={"k": 1})
    before = clip_of(env, "a").snapshot()
    p = plan_trim(env, "a", "start", 12.0)
    apply(env, p)
    a = clip_of(env, "a")
    op = sorted((k.time, k.value) for k in a.keyframes if k.property == "opacity")
    assert op == [(0.0, pytest.approx(0.5)), (2.0, 0.0)]  # boundary keyframe holds the interpolated value; the end keyframe moved with the content
    assert [(k.time, k.value) for k in a.keyframes if k.property == "position_x"] == [(0.0, 50.0)]  # all-cut property keeps its value
    for f in ("effects", "scale", "rotation", "opacity", "audio", "animation", "transition", "metadata"):
        assert getattr(a, f) == getattr(before, f)
    apply(env, plan_trim(env, "a", "end", 13.0))  # shorten to 1 s: boundary keyframe at the new end
    a = clip_of(env, "a")
    assert sorted((k.time, round(k.value, 6)) for k in a.keyframes if k.property == "opacity") == [(0.0, 0.5), (1.0, 0.25)]
    # extending the start moves the keyframes later but keeps them at the same timeline time
    apply(env, plan_trim(env, "a", "start", 11.0))
    assert max(k.time for k in clip_of(env, "a").keyframes if k.property == "opacity") == pytest.approx(2.0)
    check_invariants(env)


def test_linked_trim_uses_the_strictest_member_and_names_it():
    env = make_env(30)
    add(env, "v", 5, 4, si=5.0, link_id="L")
    add(env, "au", 5, 4, asset="a20", track="track_a1", si=0.5, link_id="L")
    p = plan_trim(env, "v", "start", 3.0)  # picture could extend 2 s, the audio only 0.5 s
    assert p.updates["v"].timeline_start == p.updates["au"].timeline_start == 4.5
    assert "source" in p.notes[0] and "voice" in p.notes[0]
    apply(env, p)
    assert clip_of(env, "v").source_in == pytest.approx(4.5) and clip_of(env, "au").source_in == 0
    with pytest.raises(TimelineError, match="voice.wav"):
        plan_trim(env, "v", "start", 3.0)
    p = plan_trim(env, "v", "end", 8.0, linked=False)
    assert set(p.updates) == {"v"}
    p = plan_trim(env, "v", "end", 8.0)
    assert set(p.updates) == {"v", "au"} and p.updates["au"].timeline_end == p.updates["v"].timeline_end


def test_trim_respects_aspect_locks_and_track_locks():
    env = make_env(30)
    add(env, "a", 0, 4, locks=frozenset({"timing"}))
    add(env, "b", 6, 4, locks=frozenset({"position"}))
    with pytest.raises(TimelineError, match="locked"):
        plan_trim(env, "a", "end", 3.0)
    with pytest.raises(TimelineError, match="locked"):
        plan_trim(env, "b", "start", 7.0)  # moves the start -> position
    apply(env, plan_trim(env, "b", "end", 9.0))  # the end edge does not touch position
    env.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_trim(env, "b", "end", 8.0)


# ------------------------------------------------------------------------------------------------------------------------ split
def test_split_source_mapping_and_fresh_ids():
    env = make_env(30)
    add(env, "a", 2, 6, asset="v30", si=1.0, speed=2.0, transition={"type": "FADE", "duration": 0.5}, effects={"fit": "contain"}, scale=1.2, audio={"volume": 0.5})
    plan = plan_split(env, ["a"], 4.0)
    assert plan.description == "Split clip" and set(plan.updates) == {"a"} and len(plan.adds) == 1
    apply(env, plan)
    left, right = clip_of(env, "a"), plan.adds[0]
    right = clip_of(env, right.id)
    assert right.id != "a" and right.asset_id == left.asset_id
    assert left.source_out == right.source_in == pytest.approx(5.0)  # 1 + 2 s * speed 2
    assert right.source_out == pytest.approx(13.0) and left.source_in == 1.0
    assert left.duration + right.duration == pytest.approx(6.0)
    assert (left.timeline_start, left.timeline_end, right.timeline_start, right.timeline_end) == (2, 4, 4, 8)
    assert left.transition == {"type": "FADE", "duration": 0.5} and right.transition is None
    for c in (left, right):
        assert c.effects == {"fit": "contain"} and c.scale == 1.2 and c.audio == {"volume": 0.5}
    check_invariants(env)


@pytest.mark.parametrize("fps", FPS)
def test_split_snaps_to_the_frame_grid(fps):
    env = make_env(fps)
    add(env, "a", 0, 8)
    apply(env, plan_split(env, ["a"], 3.1234))
    r = [c for c in env.timeline.all_clips() if c.id != "a"][0]
    assert env.timebase.is_on_frame(r.timeline_start) and abs(r.timeline_start - 3.1234) <= 0.5 / fps + 1e-9
    assert clip_of(env, "a").timeline_end == r.timeline_start


def test_split_near_edge_or_outside_is_rejected():
    env = make_env(30)
    add(env, "a", 2, 6)
    with pytest.raises(TimelineError, match="too close"):
        plan_split(env, ["a"], 2.04)
    with pytest.raises(TimelineError, match="too close"):
        plan_split(env, ["a"], 7.97)
    with pytest.raises(TimelineError, match="not inside"):
        plan_split(env, ["a"], 9.0)
    with pytest.raises(TimelineError, match="not inside"):
        plan_split(env, ["a"], 2.0)


def test_split_ai_fields_like_the_split_command():
    env = make_env(30)
    add(env, "a", 0, 6, scene_id="s1", slot="visual:0", created_by="AI", ai_decision_id="d1", locked=True)
    apply(env, plan_split(env, ["a"], 3.0))
    left = clip_of(env, "a")
    right = next(c for c in env.timeline.all_clips() if c.id != "a")
    assert (left.created_by, left.ai_decision_id, left.locked, left.slot) == ("AI", "d1", True, "visual:0")
    assert (right.created_by, right.ai_decision_id, right.locked, right.slot, right.scene_id) == ("USER", "", False, "visual:0.r", "s1")


def test_split_keeps_audio_fades_only_at_the_outer_edges():
    env = make_env(30)
    add(env, "au", 0, 10, asset="a20", track="track_a1", audio={"volume": 0.8, "fade_in": 1.0, "fade_out": 2.0})
    plan = plan_split(env, ["au"], 4.0)
    apply(env, plan)
    left, right = clip_of(env, "au"), clip_of(env, plan.adds[0].id)
    assert left.audio == {"volume": 0.8, "fade_in": 1.0, "fade_out": 0.0}
    assert right.audio == {"volume": 0.8, "fade_in": 0.0, "fade_out": 2.0}


def test_split_keyframes_are_continuous():
    env = make_env(30)
    add(env, "a", 0, 4, keyframes=[Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 4.0, 1.0), Keyframe("scale", 1.0, 2.0)])
    plan = plan_split(env, ["a"], 2.0)
    apply(env, plan)
    left, right = clip_of(env, "a"), clip_of(env, plan.adds[0].id)
    for prop in ("opacity", "scale"):
        assert value_at(left.keyframes, prop, left.duration) == pytest.approx(value_at(right.keyframes, prop, 0.0))
    assert value_at(left.keyframes, "opacity", 2.0) == pytest.approx(0.5)
    check_invariants(env)


def test_linked_split_new_halves_are_linked_to_each_other():
    env = make_env(30)
    add(env, "v", 0, 6, link_id="L")
    add(env, "au", 0, 6, asset="a20", track="track_a1", link_id="L")
    plan = plan_split(env, ["v"], 3.0)
    assert set(plan.updates) == {"v", "au"} and len(plan.adds) == 2
    apply(env, plan)
    rights = plan.adds
    assert clip_of(env, "v").link_id == clip_of(env, "au").link_id == "L"
    assert rights[0].link_id == rights[1].link_id and rights[0].link_id not in ("", "L")
    assert set(plan_split(env, ["v"], 1.0, linked=False).updates) == {"v"}
    lone = plan_split(env, ["v"], 1.0, linked=False)
    assert lone.adds[0].link_id == ""  # a single right half must not keep a dangling link id


def test_split_all_clips_on_chosen_tracks_skips_locked_ones():
    env = make_env(30)
    add(env, "v1", 0, 6)
    add(env, "v2", 1, 6, track="track_v2")
    add(env, "au", 0, 6, asset="a20", track="track_a1")
    add(env, "later", 10, 2, track="track_v3")
    env.timeline.get_track("track_a1").locked = True
    plan = plan_split(env, ["v1"], 3.0, include_intersecting_on=["track_v2", "track_a1", "track_v3"])
    assert set(plan.updates) == {"v1", "v2"} and any("locked" in n for n in plan.notes)
    apply(env, plan)
    with pytest.raises(TimelineError, match="locked"):
        plan_split(env, ["au"], 3.0)  # explicitly asking for a locked clip is an error, not a silent skip


def test_split_several_clips_and_skip_those_not_under_the_cut():
    env = make_env(30)
    add(env, "a", 0, 6)
    add(env, "b", 1, 6, track="track_v2")
    add(env, "c", 20, 2, track="track_v3")
    plan = plan_split(env, ["a", "b", "c"], 3.0)
    assert set(plan.updates) == {"a", "b"} and plan.description == "Split 2 clips" and "not inside" in " ".join(plan.notes)
    apply(env, plan)
    with pytest.raises(TimelineError, match="not inside"):
        plan_split(env, ["c"], 3.0)


def test_split_audio_only_keeps_sample_precision():
    env = make_env(30)
    add(env, "au", 0, 6, asset="a20", track="track_a1")
    apply(env, plan_split(env, ["au"], 2.123456))
    r = [c for c in env.timeline.all_clips() if c.id != "au"][0]
    assert r.timeline_start == pytest.approx(2.123456, abs=1 / 96000) and not env.timebase.is_on_frame(r.timeline_start)


def _caption(env, start=10.0, **over):
    words = [
        {"word_id": "w0", "text": "hello", "start": 10.0, "end": 10.6},
        {"word_id": "w1", "text": "brave", "start": 10.7, "end": 11.3},
        {"word_id": "w2", "text": "new", "start": 11.5, "end": 12.0},
        {"word_id": "w3", "text": "world", "start": 12.2, "end": 12.9},
    ]
    text = {"text": "hello brave new world", "lines": ["hello brave", "new world"], "words": words, "start": 10.0, "end": 13.0, "reading_cps": 7.0,
            "emphasis": [{"word_index": 0, "category": "x", "style": "y"}, {"word_index": 2, "category": "x", "style": "y"}]}
    text.update(over)
    c = Clip("cap", "track_v6", "", start, 3.0, kind="caption", text=text, created_by="AI", scene_id="s1")
    env.timeline.insert_clip(c)
    return c


def test_caption_split_between_words():
    env = make_env(30)
    _caption(env)
    plan = plan_split(env, ["cap"], 11.4)
    assert not plan.notes
    apply(env, plan)
    left, right = clip_of(env, "cap"), clip_of(env, plan.adds[0].id)
    assert left.text["text"] == "hello brave" and right.text["text"] == "new world"
    assert left.text["lines"] == ["hello brave"] and right.text["lines"] == ["new world"]
    assert [w["text"] for w in left.text["words"]] == ["hello", "brave"] and [w["text"] for w in right.text["words"]] == ["new", "world"]
    assert right.text["emphasis"] == [{"word_index": 0, "category": "x", "style": "y"}] and left.text["emphasis"] == [{"word_index": 0, "category": "x", "style": "y"}]
    assert (left.text["start"], left.text["end"], right.text["start"], right.text["end"]) == (10.0, 11.4, 11.4, 13.0)
    assert left.text["words"][0]["start"] == 10.0 and right.text["words"][0]["start"] == 11.5  # word times are untouched


def test_caption_split_inside_a_word_uses_the_nearest_boundary_and_says_so():
    env = make_env(30)
    _caption(env)
    plan = plan_split(env, ["cap"], 10.9)  # inside "brave" (10.7-11.3), nearer its start
    assert "inside a word" in plan.notes[0]
    assert [w["text"] for w in plan.updates["cap"].text["words"]] == ["hello"]
    assert [w["text"] for w in plan.adds[0].text["words"]] == ["brave", "new", "world"]


def test_text_without_word_timing_cannot_be_split_and_empty_halves_are_refused():
    env = make_env(30)
    env.timeline.insert_clip(Clip("g", "track_v5", "", 0.0, 4.0, kind="text", text={"content": "Big title", "start": 0.0, "duration": 4.0}))
    env.timeline.insert_clip(Clip("e", "track_v4", "", 0.0, 4.0, kind="graphic"))
    for cid in ("g", "e"):
        with pytest.raises(TimelineError, match="cannot be split"):
            plan_split(env, [cid], 2.0)
    _caption(env)
    with pytest.raises(TimelineError, match="no words"):
        plan_split(env, ["cap"], 10.2)  # inside the first word, nearer its start: the left part would hold no words


# ------------------------------------------------------------------------------------------------------------------------ delete
def test_delete_follows_links_groups_and_leaves_a_gap():
    env = make_env(30)
    add(env, "v", 0, 3, link_id="L")
    add(env, "au", 0, 3, asset="a20", track="track_a1", link_id="L")
    add(env, "g1", 5, 1, group_id="G")
    add(env, "g2", 6, 1, group_id="G", track="track_v2")
    add(env, "keep", 8, 1)
    plan = plan_delete(env, ["v"])
    assert set(plan.removes) == {"v", "au"}
    apply(env, plan)
    assert set(plan_delete(env, ["g1"]).removes) == {"g1", "g2"} and plan_delete(env, ["g1"], grouped=False).removes == ["g1"]
    apply(env, plan_delete(env, ["g1"]))
    assert [c.id for c in env.timeline.all_clips()] == ["keep"] and clip_of(env, "keep").timeline_start == 8  # no ripple
    assert plan_delete(env, []).empty


def test_delete_locked_is_rejected():
    env = make_env(30)
    add(env, "a", 0, 3, locks=frozenset({"style"}))
    with pytest.raises(TimelineError, match="locked"):
        plan_delete(env, ["a"])
    add(env, "b", 5, 3)
    env.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_delete(env, ["b"])


# ------------------------------------------------------------------------------------------------------------------------ duplicate
def test_duplicate_default_placement_and_independence():
    env = make_env(30)
    add(env, "v", 2, 3, link_id="L", group_id="G", scene_id="s1", slot="visual:0", created_by="AI", ai_decision_id="d1", locked=True,
        locks=frozenset({"audio"}), keyframes=[Keyframe("opacity", 0.0, 1.0)], effects={"fit": "contain"})
    add(env, "au", 2, 3, asset="a20", track="track_a1", link_id="L", group_id="G")
    plan = plan_duplicate(env, ["v"])
    assert plan.description == "Duplicate 2 clips" and not plan.updates and len(plan.adds) == 2
    before = state(env.timeline)
    apply(env, plan)
    v2 = next(c for c in plan.adds if c.track_id == "track_v1")
    a2 = next(c for c in plan.adds if c.track_id == "track_a1")
    assert v2.timeline_start == a2.timeline_start == 5  # right after the selection's end
    assert v2.id not in ("v", "au") and v2.created_by == "USER" and v2.ai_decision_id == "" and not v2.locked and v2.locks == frozenset()
    assert v2.scene_id == "s1" and v2.slot == "visual:0.copy" and v2.asset_id == "v10" and v2.source_out == 3.0
    assert v2.link_id == a2.link_id and v2.link_id not in ("", "L") and v2.group_id == a2.group_id and v2.group_id not in ("", "G")
    v2c = clip_of(env, v2.id)
    v2c.effects["fit"] = "cover"
    v2c.keyframes[0].value = 0.1
    assert clip_of(env, "v").effects == {"fit": "contain"} and clip_of(env, "v").keyframes[0].value == 1.0  # deep copy
    assert [d for d in before if d[0] == "track_v1"][0][1][0]["clip_id"] == "v"  # originals untouched
    original = clip_of(env, "v")
    assert (original.timeline_start, original.link_id, original.group_id, original.locks, original.locked) == (2, "L", "G", frozenset({"audio"}), True)


def test_duplicate_options_and_collisions():
    env = make_env(30)
    add(env, "a", 0, 2)
    add(env, "b", 3, 2)
    with pytest.raises(TimelineError, match="overlap"):
        plan_duplicate(env, ["a"], offset=2.5)
    p = plan_duplicate(env, ["a"], to_start=10.0)
    assert p.adds[0].timeline_start == 10
    p = plan_duplicate(env, ["a"], offset=7.0, target_tracks={"a": "track_v2"})
    assert p.adds[0].timeline_start == 7 and p.adds[0].track_id == "track_v2"
    with pytest.raises(TimelineError, match="not both"):
        plan_duplicate(env, ["a"], offset=1, to_start=2)
    with pytest.raises(TimelineError, match="does not hold"):
        plan_duplicate(env, ["a"], target_tracks={"a": "track_a1"})
    env.timeline.get_track("track_v2").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_duplicate(env, ["a"], offset=7.0, target_tracks={"a": "track_v2"})
    # default placement for 'a' (right after it, at 2..4) collides with "b" at 3..5
    with pytest.raises(TimelineError, match="overlap"):
        plan_duplicate(env, ["a"])


def test_duplicate_one_member_without_linking_drops_the_link():
    env = make_env(30)
    add(env, "v", 0, 2, link_id="L")
    add(env, "au", 0, 2, asset="a20", track="track_a1", link_id="L")
    p = plan_duplicate(env, ["v"], linked=False)
    assert len(p.adds) == 1 and p.adds[0].link_id == ""
    apply(env, p)


def test_duplicate_caption_shifts_its_words():
    env = make_env(30)
    _caption(env)
    plan = plan_duplicate(env, ["cap"])
    c = plan.adds[0]
    assert c.timeline_start == 13.0 and c.text["words"][0]["start"] == pytest.approx(13.0) and c.text["start"] == 13.0
    assert clip_of(env, "cap").text["words"][0]["start"] == 10.0


# ------------------------------------------------------------------------------------------------------------------------ overwrite
def mapping(env, t):
    """(asset, source time) shown at timeline time ``t`` on V1, or None."""
    for c in env.timeline.get_track("track_v1").clips:
        if c.timeline_start <= t < c.timeline_end:
            return c.asset_id, c.source_in + (t - c.timeline_start) * c.speed
    return None


def new_clip(cid, start, dur, asset="v30", track="track_v1", si=0.0):
    return Clip(cid, track, asset, start, dur, source_in=si, source_out=si + dur)


def test_overwrite_trims_head_and_tail():
    env = make_env(30)
    add(env, "e", 2, 8, si=1.0)
    apply(env, plan_overwrite(env, [new_clip("n", 0, 4)]))  # head of e covered (2..4)
    e = clip_of(env, "e")
    assert (e.timeline_start, e.timeline_end, e.source_in, e.source_out) == (4, 10, 3.0, 9.0)
    apply(env, plan_overwrite(env, [new_clip("n2", 8, 4)]))  # tail covered
    e = clip_of(env, "e")
    assert (e.timeline_start, e.timeline_end, e.source_in, e.source_out) == (4, 8, 3.0, 7.0)
    check_invariants(env)


def test_overwrite_splits_when_the_new_clip_lies_inside():
    env = make_env(30)
    add(env, "e", 0, 10, transition={"type": "FADE", "duration": 0.5}, scene_id="s1", slot="visual:0", created_by="AI", ai_decision_id="d1")
    plan = overwrite = plan_overwrite(env, [new_clip("n", 4, 2)])
    assert set(plan.updates) == {"e"} and len(plan.adds) == 2 and "split" in plan.notes[-1]
    apply(env, overwrite)
    left = clip_of(env, "e")
    right = next(c for c in env.timeline.all_clips() if c.id not in ("e", "n"))
    assert span(left) == (0, 4) and span(right) == (6, 10) and (right.source_in, right.source_out) == (6.0, 10.0)
    assert left.transition is not None and right.transition is None and right.ai_decision_id == "" and right.slot == "visual:0.r"
    check_invariants(env)


def test_overwrite_removes_covered_clips_and_handles_several_new_clips():
    env = make_env(30)
    add(env, "a", 1, 2)
    add(env, "b", 4, 2)
    add(env, "c", 7, 6)
    plan = plan_overwrite(env, [new_clip("n1", 0, 7), new_clip("n2", 9, 1)])
    assert set(plan.removes) == {"a", "b"} and set(plan.updates) == {"c"} and len(plan.adds) == 3  # n1, n2 and the right piece of c
    apply(env, plan)
    assert sorted(span(c) for c in env.timeline.get_track("track_v1").clips) == [(0, 7), (7, 9), (9, 10), (10, 13)]
    check_invariants(env)


def test_overwrite_ignores_other_tracks_and_drops_slivers_with_a_note():
    env = make_env(30)
    add(env, "e", 0.0, 10)
    add(env, "other", 0.0, 10, track="track_v2")
    plan = plan_overwrite(env, [new_clip("n", 0.03, 5)])
    assert "other" not in plan.updates and any("shorter than the minimum" in n for n in plan.notes)
    apply(env, plan)
    assert span(clip_of(env, "e")) == (5.03, 10) and span(clip_of(env, "other")) == (0, 10)


def test_overwrite_locked_blockers_reject_naming_them():
    env = make_env(30)
    add(env, "e", 0, 10, locks=frozenset({"item"}))
    with pytest.raises(TimelineError, match="ten.mp4"):
        plan_overwrite(env, [new_clip("n", 4, 2)])
    env2 = make_env(30)
    add(env2, "e", 0, 10)
    env2.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_overwrite(env2, [new_clip("n", 4, 2)])
    env3 = make_env(30)
    add(env3, "e", 0, 10, locks=frozenset({"position"}))  # trimming the head moves the start -> position lock
    with pytest.raises(TimelineError, match="locked"):
        plan_overwrite(env3, [new_clip("n", 0, 3)])
    plan = plan_overwrite(env3, [new_clip("n", 7, 3)])  # the tail trim does not
    apply(env3, plan)
    assert plan_overwrite(env, []).empty


def test_overwrite_text_clip_is_trimmed_but_not_split_without_words():
    env = make_env(30)
    env.timeline.insert_clip(Clip("g", "track_v5", "", 0.0, 6.0, kind="text", text={"content": "Title", "start": 0.0, "duration": 6.0}))
    plan = plan_overwrite(env, [Clip("g2", "track_v5", "", 4.0, 4.0, kind="text", text={"content": "Other", "start": 4.0, "duration": 4.0})])
    apply(env, plan)
    assert clip_of(env, "g").duration == 4.0 and clip_of(env, "g").text["duration"] == 4.0
    env2 = make_env(30)
    env2.timeline.insert_clip(Clip("g", "track_v5", "", 0.0, 9.0, kind="text", text={"content": "Title"}))
    with pytest.raises(TimelineError, match="single block of text"):
        plan_overwrite(env2, [Clip("g2", "track_v5", "", 3.0, 2.0, kind="text", text={"content": "X"})])


def test_overwrite_source_mapping_matches_a_brute_force_model():
    rng = random.Random(7)
    for _ in range(40):
        env = make_env(30)
        t = 0.0
        for i in range(rng.randint(1, 5)):
            t += rng.choice([0, 0, 0.5, 1.5])
            d = rng.choice([1, 2, 3.5, 4])
            add(env, f"e{i}", t, d, asset="v30", si=rng.choice([0, 2, 5.5]), speed=rng.choice([1.0, 1.0, 2.0]))
            t += d
        news = []
        cur = rng.choice([0.0, 1.0, 3.0])
        for i in range(rng.randint(1, 3)):
            cur += rng.choice([0, 0.5, 2])
            d = rng.choice([0.5, 1, 2.5, 6])
            news.append(new_clip(f"n{i}", cur, d, asset="v30", si=rng.choice([0, 1])))
            cur += d + rng.choice([0, 1, 2])
        ticks = [(k + 0.5) / 30 for k in range(30 * 30)]
        before = {t: mapping(env, t) for t in ticks}
        try:
            plan = plan_overwrite(env, news)
            PlanCommand(env, plan).do()
        except TimelineError:
            continue
        check_invariants(env)
        for t in ticks:
            covering = next((n for n in news if n.timeline_start <= t < n.timeline_end), None)
            got = mapping(env, t)
            if covering is not None:
                want = (covering.asset_id, covering.source_in + (t - covering.timeline_start))
            else:
                want = before[t]
            if want is None:
                assert got is None
            else:
                assert got is not None and got[0] == want[0] and got[1] == pytest.approx(want[1], abs=1e-6), (t, got, want)


# ------------------------------------------------------------------------------------------------------------------------ replace
def test_replace_keeps_place_look_and_resets_source():
    env = make_env(30)
    add(env, "a", 4, 5, si=2.0, scale=1.2, rotation=3.0, opacity=0.8, effects={"fit": "contain", "crop": [0, 0, 1, 1], "focus_region": [0, 0, 1, 1]},
        keyframes=[Keyframe("opacity", 0, 1.0), Keyframe("opacity", 4, 0.0)], audio={"volume": 0.5}, transition={"type": "FADE", "duration": 0.5},
        animation={"in": {"preset": "pop"}}, metadata={"phase": 5})
    before = clip_of(env, "a").snapshot()
    plan = plan_replace(env, "a", "v30")
    apply(env, plan)
    a = clip_of(env, "a")
    assert (a.asset_id, a.timeline_start, a.duration, a.track_id) == ("v30", 4, 5, "track_v1")
    assert (a.source_in, a.source_out) == (2.0, 7.0)  # old source_in kept: it fits
    assert a.effects == {"fit": "contain"} and "crop" in plan.notes[0]
    for f in ("scale", "rotation", "opacity", "keyframes", "audio", "transition", "animation", "metadata"):
        assert getattr(a, f) == getattr(before, f)


def test_replace_with_shorter_media_shortens_and_notes_it():
    env = make_env(30)
    add(env, "a", 4, 5, si=2.0)
    plan = plan_replace(env, "a", "v4")
    assert plan.updates["a"].duration == 4 and plan.updates["a"].source_in == 0 and "only 4.00 s" in plan.notes[0]
    apply(env, plan)
    plan = plan_replace(env, "a", "v30", keep_duration=False)
    assert plan.updates["a"].duration == 30 and plan.updates["a"].source_out == 30
    add(env, "n", 20, 2)
    plan = plan_replace(env, "a", "v30", keep_duration=False)
    assert plan.updates["a"].timeline_end <= 20 and any("next clip" in n for n in plan.notes)


def test_replace_between_video_and_image_resets_speed():
    env = make_env(30)
    add(env, "a", 0, 5, si=0.0, speed=2.0)
    plan = plan_replace(env, "a", "img")
    n = plan.updates["a"]
    assert (n.speed, n.source_in, n.source_out, n.duration) == (1.0, 0.0, 5.0, 5) and "speed" in plan.notes[0]
    apply(env, plan)
    plan = plan_replace(env, "a", "v10")
    assert plan.updates["a"].speed == 1.0 and plan.updates["a"].source_out == 5.0
    apply(env, plan)
    check_invariants(env)


def test_replace_rejections():
    env = make_env(30)
    add(env, "a", 0, 3)
    add(env, "au", 0, 3, asset="a20", track="track_a1")
    add(env, "locked", 5, 3, locks=frozenset({"style"}))
    env.timeline.insert_clip(Clip("cap", "track_v6", "", 0.0, 2.0, kind="caption", text={"text": "x"}))
    with pytest.raises(TimelineError, match="cannot hold audio"):
        plan_replace(env, "a", "a20")
    with pytest.raises(TimelineError, match="cannot hold video"):
        plan_replace(env, "au", "v10")
    with pytest.raises(TimelineError, match="not in the project"):
        plan_replace(env, "a", "nope")
    with pytest.raises(TimelineError, match="locked"):
        plan_replace(env, "locked", "v30")
    with pytest.raises(TimelineError, match="Only clips that play media"):
        plan_replace(env, "cap", "v30")
    assert plan_replace(env, "a", "v10").empty


# ------------------------------------------------------------------------------------------------------------------------ insert free
def test_insert_free_places_clips_one_after_another_around_obstacles():
    env = make_env(30)
    add(env, "x", 2, 3)
    clips = [new_clip("n1", 0, 2, asset="v4"), new_clip("n2", 0, 1.5, asset="v4"), new_clip("n3", 0, 1, asset="v4")]
    plan = plan_insert_free(env, clips, 0.0, "track_v1")
    starts = [c.timeline_start for c in plan.adds]
    assert starts == [0.0, 5.0, 6.5]  # n1 fits before x (0..2), n2 has to wait until x ends
    apply(env, plan)
    check_invariants(env)
    plan = plan_insert_free(env, [new_clip("m", 0, 1)], 3.0, "track_v1")
    assert plan.adds[0].timeline_start == 7.5 and plan.notes
    env.timeline.get_track("track_v2").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_insert_free(env, [new_clip("m2", 0, 1)], 0, "track_v2")
    with pytest.raises(TimelineError, match="cannot hold"):
        PlanCommand(env, plan_insert_free(env, [new_clip("m3", 0, 1)], 0, "track_a1")).do()


def test_insert_free_snaps_the_start_to_the_frame_grid():
    env = make_env(24)
    plan = plan_insert_free(env, [new_clip("m", 0, 1)], 1.0123, "track_v1")
    assert env.timebase.is_on_frame(plan.adds[0].timeline_start)


# ------------------------------------------------------------------------------------------------------------------------ clipboard
def _scene(env):
    add(env, "v", 2, 3, link_id="L", group_id="G", scene_id="s1", slot="visual:0", created_by="AI", ai_decision_id="d1", locked=True, locks=frozenset({"audio"}))
    add(env, "au", 2, 3, asset="a20", track="track_a1", link_id="L", group_id="G")
    add(env, "b", 6, 2, track="track_v2")


def test_clipboard_roundtrip_and_paste_to_the_same_tracks():
    env = make_env(30)
    _scene(env)
    data = ClipboardData.from_clips(env.timeline, ["v"])
    assert [d["clip_id"] for d in data.clips] and set(data.asset_ids) == {"v10", "a20"} and data.link_ids == ["L"] and data.group_ids == ["G"]
    again = ClipboardData.from_json(data.to_json())
    assert again.to_dict() == data.to_dict()
    plan = plan_paste(env, again, 20.0)
    assert plan.description == "Paste 2 clips"
    apply(env, plan)
    v2 = next(c for c in plan.adds if c.track_id == "track_v1")
    a2 = next(c for c in plan.adds if c.track_id == "track_a1")
    assert v2.timeline_start == a2.timeline_start == 20.0 and v2.asset_id == "v10" and a2.asset_id == "a20"
    # fresh ids, USER ownership, locks cleared, relations remapped consistently and independent from the originals
    assert {v2.id, a2.id}.isdisjoint({"v", "au", "b"})
    assert v2.created_by == "USER" and v2.ai_decision_id == "" and not v2.locked and v2.locks == frozenset() and v2.scene_id == "s1"
    assert v2.link_id == a2.link_id and v2.link_id not in ("", "L") and v2.group_id == a2.group_id and v2.group_id not in ("", "G")
    links = {}
    for c in env.timeline.all_clips():
        if c.link_id:
            links.setdefault(c.link_id, []).append(c.id)
    assert all(len(v) >= 2 for v in links.values())  # no dangling link ids
    assert clip_of(env, "v").locks == frozenset({"audio"}) and clip_of(env, "v").link_id == "L"


def test_paste_with_target_track_uses_relative_track_offsets():
    env = make_env(30)
    add(env, "a", 0, 2, track="track_v1")
    add(env, "b", 1, 2, track="track_v2")
    data = ClipboardData.from_clips(env.timeline, ["a", "b"], linked=False)
    assert data.track_offsets == [0, 1] and data.offsets == [0.0, 1.0]
    plan = plan_paste(env, data, 10.0, target_track_id="track_v2")
    by_start = sorted(plan.adds, key=lambda c: c.timeline_start)
    assert [(c.track_id, c.timeline_start) for c in by_start] == [("track_v2", 10.0), ("track_v3", 11.0)]
    apply(env, plan)
    with pytest.raises(TimelineError, match="not enough tracks|cannot hold"):
        plan_paste(env, data, 10.0, target_track_id="track_a3")
    with pytest.raises(TimelineError, match="cannot hold"):
        plan_paste(env, data, 10.0, target_track_id="track_v6")
    with pytest.raises(TimelineError, match="no longer exists"):
        plan_paste(env, data, 10.0, target_track_id="gone")


def test_paste_rejects_collisions_locked_tracks_missing_media_and_can_overwrite():
    env = make_env(30)
    add(env, "a", 0, 4)
    data = ClipboardData.from_clips(env.timeline, ["a"])
    with pytest.raises(TimelineError, match="overlap"):
        plan_paste(env, data, 2.0)
    plan = plan_paste(env, data, 2.0, policy="overwrite")
    apply(env, plan)
    assert span(clip_of(env, "a")) == (0, 2)
    env.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        plan_paste(env, data, 50.0)
    env.timeline.get_track("track_v1").locked = False
    ghost = ClipboardData.from_dict({**data.to_dict(), "clips": [{**data.clips[0], "asset_id": "gone"}]})
    with pytest.raises(TimelineError, match="no longer in this project"):
        plan_paste(env, ghost, 50.0)
    with pytest.raises(TimelineError, match="does not contain"):
        ClipboardData.from_json("{not json")
    with pytest.raises(TimelineError, match="does not contain"):
        ClipboardData.from_dict({"clips": []})
    with pytest.raises(TimelineError, match="Select at least one"):
        ClipboardData.from_clips(env.timeline, [])
    with pytest.raises(TimelineError):
        plan_paste(env, data, 5.0, policy="merge")


@pytest.mark.parametrize("fps", FPS)
def test_paste_positions_are_frame_aligned(fps):
    env = make_env(fps)
    tb = env.timebase
    add(env, "a", tb.to_seconds(10), tb.to_seconds(30))
    add(env, "b", tb.to_seconds(50), tb.to_seconds(30), track="track_v2")
    data = ClipboardData.from_clips(env.timeline, ["a", "b"], linked=False)
    plan = plan_paste(env, data, 20.0123)
    for c in plan.adds:
        assert tb.is_on_frame(c.timeline_start)
    assert tb.to_frame(plan.adds[1].timeline_start) - tb.to_frame(plan.adds[0].timeline_start) == 40


def test_copy_and_paste_attributes():
    env = make_env(30)
    add(env, "src", 0, 3, scale=1.4, rotation=7.0, opacity=0.6, position=(10.0, -20.0), effects={"fit": "contain", "crop": [0, 0, 1, 1]},
        audio={"volume": 0.4, "fade_in": 0.5, "fade_out": 0.5, "role": "X"})
    add(env, "dst", 4, 3, effects={"focus_region": [0.1, 0.1, 0.5, 0.5]}, audio={"volume": 1.0})
    add(env, "short", 8, 0.8)
    add(env, "au", 10, 3, asset="a20", track="track_a1")
    add(env, "pic", 20, 2, asset="img")
    attrs = copy_attributes(clip_of(env, "src"))
    assert "crop" not in attrs["effects"] and attrs["audio"] == {"volume": 0.4, "fade_in": 0.5, "fade_out": 0.5} and "speed" not in attrs
    plan = plan_paste_attributes(env, ["dst", "short", "au", "pic"], attrs)
    assert plan.description == "Paste attributes"
    apply(env, plan)
    d = clip_of(env, "dst")
    assert (d.scale, d.rotation, d.opacity, d.position) == (1.4, 7.0, 0.6, (10.0, -20.0))
    assert d.effects == {"fit": "contain", "focus_region": [0.1, 0.1, 0.5, 0.5]} and d.audio == {"volume": 0.4, "fade_in": 0.5, "fade_out": 0.5}
    assert d.timeline_start == 4 and d.duration == 3 and d.speed == 1.0
    s = clip_of(env, "short")
    assert s.audio["fade_in"] + s.audio["fade_out"] <= 0.8 + 1e-9 and any("fades" in n for n in plan.notes)
    au = clip_of(env, "au")
    assert au.scale == 1.0 and au.audio["volume"] == 0.4  # audio clips take only the audio part
    assert "volume" not in clip_of(env, "pic").audio  # stills have no audio
    assert plan_paste_attributes(env, ["dst"], attrs).empty
    env.timeline.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, plan_paste_attributes(env, ["dst"], {"scale": 2.0})).do()


# ------------------------------------------------------------------------------------------------------------------------ randomised sequences
def _build_random_timeline(env: PlanEnv, rng: random.Random) -> None:
    t = 0.0
    for i in range(6):
        t += rng.choice([0, 0.5, 1])
        d = rng.choice([1.5, 2, 3])
        link = f"L{i}" if i % 2 == 0 else ""
        add(env, f"v{i}", t, d, asset="v30", si=rng.choice([0, 1, 4]), link_id=link, speed=rng.choice([1.0, 1.0, 1.5]),
            keyframes=[Keyframe("opacity", 0, 1.0), Keyframe("opacity", d, 0.2)] if i % 3 == 0 else [])
        if link:
            add(env, f"a{i}", t, d, asset="a20", track="track_a1", si=rng.choice([0, 1]), link_id=link)
        t += d
    add(env, "b0", 1, 2, track="track_v2", asset="img")
    add(env, "b1", 5, 2, track="track_v2", asset="v10")
    _caption(env)


def _random_plan(env: PlanEnv, rng: random.Random, clip_data: list):
    tl = env.timeline
    clips = tl.all_clips()
    pick = lambda n=1: [c.id for c in rng.sample(clips, min(n, len(clips)))]
    t = rng.uniform(-1, 25)
    op = rng.choice(["move", "move", "trim", "trim", "split", "delete", "dup", "overwrite", "replace", "dur", "paste", "start", "src"])
    if op == "move":
        ids = pick(rng.randint(1, 3))
        moves = {}
        if rng.random() < 0.3:
            moves = {ids[0]: rng.choice(["track_v1", "track_v2", "track_v3", "track_a1", "track_a2"])}
        return plan_move(env, ids, rng.uniform(-3, 3), track_moves=moves, clamp_at_zero=rng.random() < 0.5)
    if op == "trim":
        return plan_trim(env, pick()[0], rng.choice(["start", "end"]), t, linked=rng.random() < 0.7)
    if op == "split":
        ids = pick(rng.randint(1, 2))
        if rng.random() < 0.8:  # mostly cut somewhere inside the first clip
            c = tl.find_clip(ids[0])[1]
            t = rng.uniform(c.timeline_start, c.timeline_end)
        return plan_split(env, ids, t, include_intersecting_on=rng.choice([None, ["track_v1", "track_a1"]]))
    if op == "delete":
        return plan_delete(env, pick())
    if op == "dup":
        return plan_duplicate(env, pick(rng.randint(1, 2)), offset=rng.choice([None, rng.uniform(-2, 8), rng.uniform(15, 40)]))
    if op == "overwrite":
        return plan_overwrite(env, [new_clip(f"ow{rng.randint(0, 10**6)}", max(0.0, t), rng.choice([0.5, 1, 3]), asset=rng.choice(["v4", "v10"]), track=rng.choice(["track_v1", "track_v2"]))])
    if op == "replace":
        return plan_replace(env, pick()[0], rng.choice(["v4", "v10", "v30", "img", "a5"]), keep_duration=rng.random() < 0.5)
    if op == "dur":
        return plan_set_duration(env, pick()[0], rng.uniform(0.1, 6))
    if op == "start":
        return plan_set_start(env, pick()[0], max(0.0, t))
    if op == "src":
        return plan_set_source_range(env, pick()[0], source_in=rng.choice([None, rng.uniform(0, 3)]), source_out=rng.choice([None, rng.uniform(3, 12)]))
    if rng.random() < 0.5 or not clip_data:
        clip_data[:] = [ClipboardData.from_clips(tl, pick(rng.randint(1, 3)))]
    return plan_paste(env, clip_data[0], max(0.0, t), policy=rng.choice(["reject", "overwrite"]))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("fps", [30, 29.97])
def test_random_edit_sequences_keep_invariants_and_undo_redo_exactly(seed, fps):
    rng = random.Random(seed * 101 + int(fps))
    env = make_env(fps)
    _build_random_timeline(env, rng)
    initial = state(env.timeline)
    done: list[tuple[PlanCommand, list]] = []
    states = [initial]
    clip_data: list = []
    applied = 0
    for _ in range(60):
        try:
            plan = _random_plan(env, rng, clip_data)
        except TimelineError:
            continue
        before = state(env.timeline)
        cmd = PlanCommand(env, plan)
        try:
            cmd.do()
        except TimelineError:
            assert state(env.timeline) == before, "a refused plan must leave the timeline untouched"
            continue
        if plan.empty:
            continue
        applied += 1
        check_invariants(env)
        done.append((cmd, copy.deepcopy(plan.notes)))
        states.append(state(env.timeline))
    assert applied >= 8
    for i in range(len(done) - 1, -1, -1):
        done[i][0].undo()
        assert state(env.timeline) == states[i], f"undo of step {i} did not restore the previous state"
    assert state(env.timeline) == initial
    for i, (cmd, _n) in enumerate(done):
        cmd.do()
        assert state(env.timeline) == states[i + 1], f"redo of step {i} differs"


def test_trim_and_split_preserve_source_mapping_of_surviving_material():
    rng = random.Random(11)
    for _ in range(250):
        env = make_env(30)
        t = 0.0
        for i in range(4):
            t += rng.choice([0, 1])
            d = rng.choice([2, 3, 4])
            add(env, f"e{i}", t, d, asset="v30", si=rng.choice([0, 3, 7]), speed=rng.choice([1.0, 2.0]))
            t += d
        ticks = [(k + 0.5) / 30 for k in range(30 * 20)]
        before = {x: mapping(env, x) for x in ticks}
        cid = rng.choice([c.id for c in env.timeline.all_clips()])
        try:
            if rng.random() < 0.5:
                plan = plan_trim(env, cid, rng.choice(["start", "end"]), rng.uniform(0, 18))
            else:
                plan = plan_split(env, [cid], rng.uniform(0, 18))
            PlanCommand(env, plan).do()
        except TimelineError:
            continue
        check_invariants(env)
        for x in ticks:
            after = mapping(env, x)
            if before[x] is not None and after is not None:
                assert after[0] == before[x][0] and after[1] == pytest.approx(before[x][1], abs=1e-6)
            if plan.adds:  # a split never loses material
                assert (before[x] is None) == (after is None)


# ------------------------------------------------------------------------------------------------------------------------ locks are enforced by PlanCommand for every planner
def _lock_cases():
    def mk():
        env = make_env(30)
        add(env, "a", 0, 4, link_id="L")
        add(env, "au", 0, 4, asset="a20", track="track_a1", link_id="L")
        add(env, "b", 6, 4, track="track_v2")
        return env

    yield "move", mk, lambda e: plan_move(e, ["a"], 1)
    yield "trim", mk, lambda e: plan_trim(e, "a", "end", 3.0)
    yield "split", mk, lambda e: plan_split(e, ["a"], 2.0)
    yield "delete", mk, lambda e: plan_delete(e, ["a"])
    yield "replace", mk, lambda e: plan_replace(e, "b", "v30")
    yield "overwrite", mk, lambda e: plan_overwrite(e, [new_clip("n", 7, 1, track="track_v2")])
    yield "duplicate", mk, lambda e: plan_duplicate(e, ["b"], to_start=20.0)
    yield "paste", mk, lambda e: plan_paste(e, ClipboardData.from_clips(e.timeline, ["b"]), 30.0)
    yield "insert", mk, lambda e: plan_insert_free(e, [new_clip("n", 0, 1, track="track_v2")], 20.0, "track_v2")


@pytest.mark.parametrize("name,mk,planner", list(_lock_cases()), ids=[c[0] for c in _lock_cases()])
def test_every_planner_result_is_refused_on_locked_tracks_and_leaves_the_timeline_untouched(name, mk, planner):
    env = mk()
    plan = planner(env)
    assert not plan.empty
    apply(env, plan)  # without locks the plan works, undoes and redoes exactly
    env = mk()
    plan = planner(env)
    for t in env.timeline.tracks:
        t.locked = True
    rejects(env, plan, "locked")


@pytest.mark.parametrize("name,mk,planner", list(_lock_cases())[:5], ids=[c[0] for c in _lock_cases()][:5])
def test_item_locked_clips_are_refused_by_the_planners(name, mk, planner):
    env = mk()
    for c in env.timeline.all_clips():
        c.locks = frozenset({"item"})
    with pytest.raises(TimelineError, match="locked"):
        planner(env)
