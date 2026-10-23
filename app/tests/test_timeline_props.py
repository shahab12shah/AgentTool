"""Phase 10 / W1-C: clip-update planners (properties, speed, transitions, keyframes, links / groups / locks), each applied through ``PlanCommand`` with undo / redo."""

from __future__ import annotations

import json
import random

import pytest

from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.keyframes import INTERPOLATIONS, Keyframe
from app.timeline.ops_props import (
    LINK_POLICIES,
    SpeedChoiceRequired,
    clip_category,
    dissolve_source,
    keyframe_properties,
    plan_add_keyframe,
    plan_copy_keyframes,
    plan_delete_keyframes,
    plan_group,
    plan_link,
    plan_move_keyframes,
    plan_paste_keyframes,
    plan_remove_transition,
    plan_set_keyframe_interpolation,
    plan_set_locks,
    plan_set_speed,
    plan_set_transition,
    plan_ungroup,
    plan_unlink,
    plan_update_clip,
)
from app.timeline.plan import EditPlan, PlanCommand, PlanEnv
from app.timeline.timeline import Timeline

ASSETS = {"vid": ("video", 30.0, "Interview"), "vid2": ("video", 30.0, "B-roll"), "img": ("image", None, "Photo"), "aud": ("audio", 120.0, "Voice"), "mus": ("audio", 200.0, "Music")}


def state(tl: Timeline) -> str:
    return json.dumps(tl.to_dict(), sort_keys=True)


def media(cid: str, track: str, start: float, dur: float, asset: str = "vid", *, source_in: float = 0.0, speed: float = 1.0, **kw) -> Clip:
    return Clip(cid, track, asset, start, dur, source_in=source_in, source_out=source_in + dur * speed, speed=speed, **kw)


@pytest.fixture
def tl() -> Timeline:
    t = Timeline.default()
    t.insert_clip(media("v1", "track_v1", 0.0, 3.0, source_in=1.0))
    t.insert_clip(media("v2", "track_v1", 3.0, 3.0, "vid2"))
    t.insert_clip(media("v3", "track_v1", 8.0, 2.0))
    t.insert_clip(media("img", "track_v3", 0.0, 4.0, "img"))
    t.insert_clip(media("a1", "track_a1", 0.0, 6.0, "aud", audio={"volume": 1.0, "role": "VOICE"}))
    t.insert_clip(media("m1", "track_a2", 0.0, 10.0, "mus", audio={"volume": 0.5, "fade_in": 1.0, "fade_out": 2.0}))
    t.insert_clip(Clip("tx", "track_v5", "", 1.0, 2.0, kind="text", text={"content": "Hello", "position": [0.5, 0.5], "size": 56, "style": "HEADLINE"}))
    t.insert_clip(Clip("cap", "track_v6", "", 0.0, 2.0, kind="caption", text={"text": "Hi there", "lines": ["Hi there"], "position": "bottom", "position_xy": []}))
    t.insert_clip(Clip("gr", "track_v4", "", 2.0, 1.0, kind="graphic", effects={"highlight": {"region": [0.1, 0.1, 0.3, 0.3]}}))
    return t


@pytest.fixture
def env(tl: Timeline) -> PlanEnv:
    return PlanEnv(tl, ASSETS.get)


def run(env: PlanEnv, plan: EditPlan, *, check_undo: bool = True) -> PlanCommand:
    """Apply a plan through PlanCommand; verify undo restores the exact state and redo reproduces the result."""
    tl = env.timeline
    before = state(tl)
    cmd = PlanCommand(env, plan)
    cmd.do()
    after = state(tl)
    if check_undo:
        cmd.undo()
        assert state(tl) == before, "undo must restore the timeline exactly"
        cmd.do()
        assert state(tl) == after, "redo must reproduce the edit exactly"
    return cmd


def relocate(tl: Timeline, cid: str, start: float) -> None:
    """Move a clip in the fixture (keeps the track list sorted like every real edit does)."""
    track, c = tl.find_clip(cid)
    c.timeline_start = start
    track.sort()


def rejects(env: PlanEnv, fn, *args, match: str = "", **kw) -> TimelineError:
    before = state(env.timeline)
    with pytest.raises(TimelineError, match=match) as e:
        fn(env, *args, **kw)
    assert state(env.timeline) == before, "a planner must never mutate the timeline"
    return e.value


# ------------------------------------------------------------------------------------------------------------------------ properties
def test_planners_are_pure(env):
    before = state(env.timeline)
    plan = plan_update_clip(env, "v1", scale=1.5, opacity=0.5)
    assert state(env.timeline) == before and set(plan.updates) == {"v1"}
    assert env.timeline.get_clip("v1").scale == 1.0  # not applied until PlanCommand runs it


def test_update_picture_properties(env):
    plan = plan_update_clip(env, "v1", position=(120, -40), scale=1.25, rotation=-15.5, opacity=0.75, fit="contain", crop=(0.1, 0.2, 0.5, 0.6))
    assert plan.description.startswith("Change position")
    run(env, plan)
    c = env.timeline.get_clip("v1")
    assert c.position == (120.0, -40.0) and (c.scale, c.rotation, c.opacity) == (1.25, -15.5, 0.75)
    assert c.effects["fit"] == "contain" and c.effects["crop"] == [0.1, 0.2, 0.5, 0.6]
    run(env, plan_update_clip(env, "v1", crop=None))
    assert "crop" not in env.timeline.get_clip("v1").effects
    run(env, plan_update_clip(env, "v1", crop=(0, 0, 1, 1)))  # whole picture: nothing to store
    assert "crop" not in env.timeline.get_clip("v1").effects


def test_update_images_use_the_same_picture_properties(env):
    run(env, plan_update_clip(env, "img", scale=2.0))
    assert env.timeline.get_clip("img").scale == 2.0


@pytest.mark.parametrize("changes, match", [
    ({"scale": 0.0}, "Scale must be between 0.01 and 20"),
    ({"scale": 21}, "Scale"),
    ({"opacity": 1.01}, "Opacity must be between 0 and 1"),
    ({"opacity": -0.1}, "Opacity"),
    ({"rotation": 361}, "Rotation must be between -360 and 360"),
    ({"scale": float("nan")}, "finite"),
    ({"scale": "big"}, "must be a number"),
    ({"scale": True}, "must be a number"),
    ({"position": (1,)}, "two numbers"),
    ({"position": (1e9, 0)}, "Position"),
    ({"fit": "stretch"}, "Fit must be"),
    ({"crop": (0.5, 0.5, 0.6, 0.6)}, "inside the picture"),
    ({"crop": (0, 0, 0.001, 1)}, "Crop width"),
    ({"crop": (0.1, 0.1, 0.5)}, "four numbers"),
])
def test_picture_property_validation(env, changes, match):
    rejects(env, plan_update_clip, "v1", match=match, **changes)


def test_boundary_values_are_accepted(env):
    run(env, plan_update_clip(env, "v1", scale=0.01, opacity=0.0, rotation=360.0, position=(0, 0)))
    run(env, plan_update_clip(env, "v1", scale=20, opacity=1, rotation=-360))


def test_whitelist_rejects_unknown_and_wrong_kind(env):
    rejects(env, plan_update_clip, "v1", match="not a clip property", frobnicate=1)
    rejects(env, plan_update_clip, "v1", match="not a clip property", timeline_start=3.0)  # moves are other planners
    rejects(env, plan_update_clip, "v1", match="own sound is not played", volume=1.0)
    rejects(env, plan_update_clip, "a1", match="audio clip", scale=2.0)
    rejects(env, plan_update_clip, "tx", match="text_position", position=(1, 1))
    rejects(env, plan_update_clip, "cap", match="caption", text_content="x")
    rejects(env, plan_update_clip, "gr", match="Graphics panel", opacity=0.5)
    rejects(env, plan_update_clip, "v1", match="Nothing to change")
    assert (clip_category(env.timeline.get_track("track_v1"), env.timeline.get_clip("v1")), clip_category(env.timeline.get_track("track_a1"), env.timeline.get_clip("a1"))) == ("visual", "audio")


def test_noop_update_returns_an_empty_plan(env):
    plan = plan_update_clip(env, "v1", scale=1.0, opacity=1.0)
    assert plan.empty and plan.notes
    before = state(env.timeline)
    PlanCommand(env, plan).do()
    assert state(env.timeline) == before


def test_audio_properties(env):
    run(env, plan_update_clip(env, "m1", volume=2.5, fade_in=0.5, fade_out=3.0, pan=-0.25))
    a = env.timeline.get_clip("m1").audio
    assert (a["volume"], a["fade_in"], a["fade_out"], a["pan"]) == (2.5, 0.5, 3.0, -0.25)
    run(env, plan_update_clip(env, "m1", volume=4.0))
    run(env, plan_update_clip(env, "m1", volume=0.0))


@pytest.mark.parametrize("changes, match", [
    ({"volume": 4.01}, "Volume must be between 0 and 4"),
    ({"volume": -0.1}, "Volume"),
    ({"pan": 1.5}, "Pan must be between -1 and 1"),
    ({"fade_in": -1}, "Fade in"),
    ({"fade_in": 11}, "Fade in must be between 0 and 10"),
    ({"fade_in": 6.0, "fade_out": 5.0}, "longer than the clip"),
    ({"fade_out": 9.5}, "longer than the clip"),  # the existing 1 s fade in + 9.5 s
    ({"muted": "yes"}, "on or off"),
    ({"muted": True, "volume": 1.0}, "not both"),
])
def test_audio_validation(env, changes, match):
    rejects(env, plan_update_clip, "m1", match=match, **changes)


def test_mute_is_volume_zero_with_a_note_and_unmute_restores(env):
    plan = plan_update_clip(env, "m1", muted=True)
    assert plan.description == "Mute clip" and any("volume to 0" in n for n in plan.notes)
    run(env, plan)
    c = env.timeline.get_clip("m1")
    assert c.audio["volume"] == 0.0 and "muted" not in c.audio  # the mixer has no mute key, so none is invented
    assert plan_update_clip(env, "m1", muted=True).empty  # already muted
    run(env, plan_update_clip(env, "m1", muted=False))
    assert env.timeline.get_clip("m1").audio["volume"] == 0.5  # the volume before muting
    assert "premute_volume" not in env.timeline.get_clip("m1").metadata
    assert plan_update_clip(env, "m1", muted=False).empty
    # an explicit volume change forgets the remembered one
    run(env, plan_update_clip(env, "m1", muted=True), check_undo=False)
    run(env, plan_update_clip(env, "m1", volume=1.5), check_undo=False)
    assert env.timeline.get_clip("m1").audio["volume"] == 1.5 and "premute_volume" not in env.timeline.get_clip("m1").metadata


def test_text_properties(env):
    run(env, plan_update_clip(env, "tx", text_content="  New title ", text_style="WARNING", text_position=(0.25, 0.75), text_size=72, text_alignment="left",
                              text_opacity=0.5, text_background="box", text_font="Serif", text_emphasis="BOLD_TEXT", text_subtitle="sub"))
    t = env.timeline.get_clip("tx").text
    assert (t["content"], t["title"], t["style"], t["position"], t["size"], t["alignment"], t["opacity"], t["background"], t["font"], t["emphasis"], t["subtitle"]) == \
        ("New title", "New title", "WARNING", [0.25, 0.75], 72, "left", 0.5, "box", "Serif", "BOLD_TEXT", "sub")
    assert env.timeline.get_clip("tx").position == (0.0, 0.0)  # the clip-level transform is not touched for text


@pytest.mark.parametrize("changes, match", [
    ({"text_content": "   "}, "cannot be empty"),
    ({"text_content": 5}, "cannot be empty"),
    ({"text_style": "FANCY"}, "Unknown text style"),
    ({"text_position": (1.5, 0.5)}, "Text position"),
    ({"text_size": 3}, "Text size"),
    ({"text_size": 56.5}, "whole number"),
    ({"text_alignment": "right"}, "Alignment"),
    ({"text_background": "gradient"}, "Background"),
    ({"text_opacity": 2}, "Text opacity"),
    ({"text_font": ""}, "font"),
    ({"text_emphasis": "LOUD"}, "Unknown emphasis"),
])
def test_text_validation(env, changes, match):
    rejects(env, plan_update_clip, "tx", match=match, **changes)


def test_text_counter_follows_the_new_number(env):
    env.timeline.get_clip("tx").text = {"content": "12", "counter": {"from": 0.0, "to": 12.0, "decimals": 0, "prefix": "", "suffix": "", "thousands": False}}
    run(env, plan_update_clip(env, "tx", text_content="1,250"))
    assert env.timeline.get_clip("tx").text["counter"]["to"] == 1250.0
    plan = plan_update_clip(env, "tx", text_content="a lot")
    assert any("counting" in n for n in plan.notes)
    run(env, plan)
    assert "counter" not in env.timeline.get_clip("tx").text


def test_caption_properties(env):
    run(env, plan_update_clip(env, "cap", caption_position="top"))
    assert env.timeline.get_clip("cap").text["position"] == "top"
    run(env, plan_update_clip(env, "cap", caption_position_xy=(0.3, 0.9), caption_highlight_mode="PROGRESSIVE"))
    t = env.timeline.get_clip("cap").text
    assert (t["position"], t["position_xy"], t["highlight_mode"]) == ("custom", [0.3, 0.9], "PROGRESSIVE")
    run(env, plan_update_clip(env, "cap", caption_position="center"))
    assert env.timeline.get_clip("cap").text["position_xy"] == []
    rejects(env, plan_update_clip, "cap", match="custom caption position needs", caption_position="custom")
    rejects(env, plan_update_clip, "cap", match="only be given together", caption_position="top", caption_position_xy=(0.5, 0.5))
    rejects(env, plan_update_clip, "cap", match="Caption position must be", caption_position="left")
    rejects(env, plan_update_clip, "cap", match="Highlight mode", caption_highlight_mode="SPARKLE")
    rejects(env, plan_update_clip, "cap", match="Caption position", caption_position_xy=(2, 0.5))


# ------------------------------------------------------------------------------------------------------------------------ locks vs edits
def test_lock_edit_works_on_a_locked_clip_but_a_move_is_rejected(env):
    tl = env.timeline
    run(env, plan_set_locks(env, ["v1"], {"position", "transform"}))
    assert tl.get_clip("v1").locks == frozenset({"position", "transform"})
    # editing a locked aspect is refused by the one validator
    before = state(tl)
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, plan_update_clip(env, "v1", scale=2.0)).do()
    moved = tl.get_clip("v1").snapshot()
    moved.timeline_start = 0.5
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, EditPlan("move", {"v1": moved})).do()
    assert state(tl) == before
    # ...but another aspect is free, and the lock itself can be edited while locked
    run(env, plan_update_clip(env, "v1", crop=(0, 0, 0.5, 0.5)))
    run(env, plan_set_locks(env, ["v1"], {"audio"}, "add"))
    assert tl.get_clip("v1").locks == frozenset({"position", "transform", "audio"})
    run(env, plan_set_locks(env, ["v1"], {"position"}, "remove"))
    run(env, plan_set_locks(env, ["v1"], set(), "set"))
    assert tl.get_clip("v1").locks == frozenset() and not tl.get_clip("v1").protected
    run(env, plan_update_clip(env, "v1", scale=2.0))


def test_item_lock_blocks_everything_but_lock_edits(env):
    tl = env.timeline
    run(env, plan_set_locks(env, ["m1"], {"item"}), check_undo=False)
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, plan_update_clip(env, "m1", volume=2.0)).do()
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, plan_add_keyframe(env, "m1", "volume", 1.0, 0.2)).do()
    run(env, plan_set_locks(env, ["m1"], {"item"}, "remove"), check_undo=False)
    assert tl.get_clip("m1").locks == frozenset()


def test_lock_planner_validation(env):
    rejects(env, plan_set_locks, ["v1"], {"bogus"}, match="Unknown lock")
    rejects(env, plan_set_locks, ["v1"], {"item"}, "toggle", match="set, add or remove")
    rejects(env, plan_set_locks, [], {"item"}, match="at least one clip")
    rejects(env, plan_set_locks, ["ghost"], {"item"}, match="no longer exists")
    assert plan_set_locks(env, ["v1"], {"item"}, "remove").empty
    plan = plan_set_locks(env, ["v1", "v2", "img"], "timing")  # a bare string is one lock
    assert plan.description == "Lock 3 clips" and set(plan.updates) == {"v1", "v2", "img"}
    env.timeline.get_track("track_v1").locked = True
    rejects(env, plan_set_locks, ["v1"], {"item"}, match="Unlock the track")


# ------------------------------------------------------------------------------------------------------------------------ speed
@pytest.mark.parametrize("speed, duration", [(0.5, 6.0), (1.5, 2.0), (2.0, 1.5), (1.0, 3.0)])
def test_speed_duration_math_keeps_the_source_range(env, speed, duration):
    c0 = env.timeline.get_clip("v1")
    src = (c0.source_in, c0.source_out)
    assert (src[1] - src[0]) == pytest.approx(3.0)
    if speed == 1.0:
        assert plan_set_speed(env, "v1", 1.0).empty
        return
    relocate(env.timeline, "v2", 20.0)  # make room for slowing down
    plan = plan_set_speed(env, "v1", speed)
    assert set(plan.updates) == {"v1"}
    run(env, plan)
    c = env.timeline.get_clip("v1")
    assert c.speed == speed and c.duration == pytest.approx(duration) and (c.source_in, c.source_out) == src
    assert (c.source_out - c.source_in) == pytest.approx(c.duration * c.speed)
    assert c.timeline_start == 0.0


def test_speed_roundtrip_is_exact_and_never_drifts(env):
    relocate(env.timeline, "v2", 30.0)
    for s in (0.7, 1.3, 2.4, 0.9, 1.0, 1.7):
        run(env, plan_set_speed(env, "v1", s), check_undo=False)
    c = env.timeline.get_clip("v1")
    assert (c.source_in, c.source_out) == (1.0, 4.0) and c.duration == pytest.approx(3.0 / c.speed, abs=1e-12)


def test_speed_up_needs_no_room_but_slow_down_is_limited_by_the_next_clip(env):
    run(env, plan_set_speed(env, "v1", 2.0))  # shorter: fine even though v2 follows
    assert env.timeline.get_clip("v1").duration == pytest.approx(1.5)
    err = rejects(env, plan_set_speed, "v2", 0.5, match="free before the next clip")  # v2 wants 6 s but only 5 s are free before v3 at 8
    assert "slowest possible speed" in str(err) and "0.60×" in str(err)
    run(env, plan_set_speed(env, "v2", 0.6))
    assert env.timeline.get_clip("v2").duration == pytest.approx(5.0)


def test_speed_range_reverse_and_kinds(env):
    rejects(env, plan_set_speed, "v1", 0, match="Reverse playback")
    rejects(env, plan_set_speed, "v1", -1, match="Reverse playback")
    rejects(env, plan_set_speed, "v1", 0.05, match="between 0.1× and 8×")
    rejects(env, plan_set_speed, "v1", 9, match="between 0.1× and 8×")
    rejects(env, plan_set_speed, "v1", float("nan"), match="must be a number")
    rejects(env, plan_set_speed, "v1", "fast", match="must be a number")
    rejects(env, plan_set_speed, "img", 2, match="picture")
    rejects(env, plan_set_speed, "tx", 2, match="not a video or audio")
    tiny = media("tiny", "track_v2", 0.0, 0.1)
    env.timeline.insert_clip(tiny)
    rejects(env, plan_set_speed, "tiny", 8, match="shortest clip")


def test_audio_speed_is_allowed_and_says_so(env):
    env.timeline.get_clip("m1").duration = 8.0  # keep the source range consistent: 8 s of source at speed 1
    env.timeline.get_clip("m1").source_out = 8.0
    plan = plan_set_speed(env, "m1", 2.0)
    assert any("playback tempo" in n for n in plan.notes)
    run(env, plan)
    c = env.timeline.get_clip("m1")
    assert c.duration == pytest.approx(4.0) and c.audio["fade_out"] == 2.0 and c.audio["fade_in"] == 1.0
    # fades that no longer fit are shortened with a note
    plan = plan_set_speed(env, "m1", 8.0)  # 1 s long now: fades 1 + 2 s > 1 s
    assert any("fades" in n for n in plan.notes)
    run(env, plan)
    a = env.timeline.get_clip("m1").audio
    assert a["fade_in"] + a["fade_out"] == pytest.approx(env.timeline.get_clip("m1").duration)


def test_voice_over_speed_never_changes_as_a_side_effect(env):
    before = {c.id: (c.speed, c.duration, c.source_in, c.source_out) for c in env.timeline.all_clips() if c.id != "v1"}
    run(env, plan_set_speed(env, "v1", 2.0))
    for cid, snap in before.items():
        c = env.timeline.get_clip(cid)
        assert (c.speed, c.duration, c.source_in, c.source_out) == snap
    assert any("voice-over" in n for n in plan_set_speed(env, "a1", 2.0).notes)


def test_speed_stretches_picture_keyframes_and_clamps_transition(env):
    c = env.timeline.get_clip("v1")
    c.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 3.0, 1.4, "ease_out", "dec9")]
    c.transition = {"type": "FADE", "duration": 2.5}
    plan = plan_set_speed(env, "v1", 2.0)  # 3 s -> 1.5 s
    assert any("keyframes" in n for n in plan.notes) and any("transition" in n for n in plan.notes)
    run(env, plan)
    c = env.timeline.get_clip("v1")
    assert [(k.time, k.value, k.decision_id) for k in c.keyframes] == [(0.0, 1.0, ""), (1.5, 1.4, "dec9")]
    assert c.transition["duration"] == pytest.approx(1.5)


def test_audio_speed_keeps_ducking_times_and_windows_the_rest(env):
    c = env.timeline.get_clip("m1")
    c.duration, c.source_out = 8.0, 8.0
    c.keyframes = [Keyframe("volume", 1.0, 1.0), Keyframe("volume", 2.0, 0.3), Keyframe("volume", 6.0, 0.3), Keyframe("volume", 7.0, 1.0)]
    run(env, plan_set_speed(env, "m1", 2.0))  # 4 s: keyframes after 4 s are cut, the value at the end is kept
    ks = env.timeline.get_clip("m1").keyframes
    assert [round(k.time, 6) for k in ks] == [1.0, 2.0, 4.0] and ks[-1].value == pytest.approx(0.3)


def test_linked_speed_requires_an_explicit_policy(env):
    tl = env.timeline
    tl.get_clip("v1").link_id = "L"
    tl.get_clip("a1").link_id = "L"
    tl.get_clip("a1").source_in, tl.get_clip("a1").source_out, tl.get_clip("a1").duration = 1.0, 4.0, 3.0  # same span as v1
    err = rejects(env, plan_set_speed, "v1", 2.0, match="Choose how to change the speed")
    assert isinstance(err, SpeedChoiceRequired) and err.options == LINK_POLICIES and err.member_ids == ("a1",)
    rejects(env, plan_set_speed, "v1", 2.0, link_policy="sideways", match="Unknown link choice")
    rejects(env, plan_set_speed, "v1", 2.0, linked=False, link_policy="keep_linked_audio_in_sync", match="cannot both")
    both = plan_set_speed(env, "v1", 2.0, link_policy="keep_linked_audio_in_sync")
    assert set(both.updates) == {"v1", "a1"}
    run(env, both)
    assert tl.get_clip("v1").duration == pytest.approx(1.5) and tl.get_clip("a1").duration == pytest.approx(1.5) and tl.get_clip("a1").speed == 2.0
    # independent / linked=False: only this clip, with a warning that sync may break
    for kw in ({"link_policy": "independent"}, {"linked": False}):
        plan = plan_set_speed(env, "v1", 4.0, **kw)
        assert set(plan.updates) == {"v1"} and any("out of sync" in n for n in plan.notes)
    run(env, plan_set_speed(env, "v1", 4.0, link_policy="independent"))
    assert tl.get_clip("a1").speed == 2.0
    # when the mates already have the requested speed nothing is ambiguous
    tl.get_clip("v1").speed = 1.0
    tl.get_clip("v1").duration = 3.0
    tl.get_clip("v1").source_in, tl.get_clip("v1").source_out = 1.0, 4.0
    tl.get_clip("a1").speed, tl.get_clip("a1").duration = 2.0, 1.5
    assert set(plan_set_speed(env, "v1", 2.0).updates) == {"v1"}


def test_linked_speed_skips_members_that_cannot_change_speed(env):
    tl = env.timeline
    tl.get_clip("v1").link_id = tl.get_clip("img").link_id = tl.get_clip("tx").link_id = "L"
    plan = plan_set_speed(env, "v1", 2.0)  # picture / text mates have no speed: nothing to choose
    assert set(plan.updates) == {"v1"}


# ------------------------------------------------------------------------------------------------------------------------ transitions
def test_transitions_set_replace_remove(env):
    for kind in ("FADE", "WIPE", "SLIDE"):
        plan = plan_set_transition(env, "v1", kind.lower(), 1.0)
        run(env, plan)
        assert env.timeline.get_clip("v1").transition == {"type": kind, "duration": 1.0}
    assert any("left to right" in n for n in plan_set_transition(env, "v1", "WIPE", 0.5).notes)
    run(env, plan_set_transition(env, "v1", "FADE", 0.5), check_undo=False)
    assert env.timeline.get_clip("v1").transition["duration"] == 0.5
    run(env, plan_remove_transition(env, "v1"))
    assert env.timeline.get_clip("v1").transition is None
    assert plan_remove_transition(env, "v1").empty
    run(env, plan_set_transition(env, "v1", "FADE", 0.5), check_undo=False)
    run(env, plan_set_transition(env, "v1", None), check_undo=False)
    assert env.timeline.get_clip("v1").transition is None
    run(env, plan_set_transition(env, "v1", "FADE", 0.5), check_undo=False)
    run(env, plan_set_transition(env, "v1", "CUT"), check_undo=False)  # a manual cut is "no transition"
    assert env.timeline.get_clip("v1").transition is None
    assert plan_set_transition(env, "v1", "CUT").empty


def test_manual_transition_drops_the_ai_decision(env):
    env.timeline.get_clip("v1").transition = {"type": "FADE", "duration": 1.0, "decision_id": "dec5"}
    run(env, plan_set_transition(env, "v1", "FADE", 1.0))
    assert "decision_id" not in env.timeline.get_clip("v1").transition


@pytest.mark.parametrize("args, match", [
    (("PIXELATE", 1.0), "Unknown transition"),
    (("FADE", None), "needs a duration"),
    (("FADE", 0), "at least one frame"),
    (("FADE", -1), "at least one frame"),
    (("FADE", 3.5), "longer than the clip"),
    (("FADE", float("inf")), "number of seconds"),
    (("FADE", "long"), "number of seconds"),
])
def test_transition_validation(env, args, match):
    rejects(env, plan_set_transition, "v1", *args, match=match)


def test_transition_params_and_clip_kinds(env):
    rejects(env, plan_set_transition, "v1", "WIPE", 1.0, {"direction": "left"}, match="no options")
    run(env, plan_set_transition(env, "v1", "WIPE", 1.0, {}))
    for cid in ("a1", "tx", "cap", "gr"):
        rejects(env, plan_set_transition, cid, "FADE", 0.5, match="not a picture clip")
        rejects(env, plan_remove_transition, cid, match="not a picture clip")
    run(env, plan_set_transition(env, "img", "FADE", 1.0))  # images can fade in


def test_dissolve_needs_a_picture_ending_where_the_clip_starts(env):
    tl = env.timeline
    assert dissolve_source(env, tl.get_clip("v1")) is None
    rejects(env, plan_set_transition, "v1", "DISSOLVE", 1.0, match="needs a picture that ends exactly where")
    rejects(env, plan_set_transition, "v3", "DISSOLVE", 1.0, match="needs a picture")  # v3 starts at 8; v2 ends at 6: a gap
    plan = plan_set_transition(env, "v2", "DISSOLVE", 1.0)  # v1 ends at 3.0 where v2 starts
    assert any("Interview" in n for n in plan.notes) and dissolve_source(env, tl.get_clip("v2")).id == "v1"
    run(env, plan)
    assert tl.get_clip("v2").transition == {"type": "DISSOLVE", "duration": 1.0}
    # the previous picture may sit on another visible picture track, as in the compiler; not on a hidden one
    tl.insert_clip(media("over", "track_v2", 8.0, 1.0, "vid2"))
    relocate(tl, "v3", 9.0)
    tl.get_clip("v3").duration = 1.0
    tl.get_clip("v3").source_out = 1.0
    assert dissolve_source(env, tl.get_clip("v3")).id == "over"
    tl.get_track("track_v2").hidden = True
    assert dissolve_source(env, tl.get_clip("v3")) is None
    # an audio file / missing media is never a dissolve source
    tl.get_track("track_v2").hidden = False
    tl.get_clip("over").asset_id = "nowhere"
    assert dissolve_source(env, tl.get_clip("v3")) is None


# ------------------------------------------------------------------------------------------------------------------------ keyframes
def kfs(env, cid: str):
    return [(k.property, round(k.time, 6), k.value, k.interpolation, k.decision_id) for k in env.timeline.get_clip(cid).keyframes]


def test_add_keyframes_sorted_and_rendered_properties_only(env):
    run(env, plan_add_keyframe(env, "v1", "scale", 3.0, 1.5, "ease_out"))
    run(env, plan_add_keyframe(env, "v1", "scale", 0.0, 1.0))
    run(env, plan_add_keyframe(env, "v1", "position_x", 1.5, 40))
    run(env, plan_add_keyframe(env, "v1", "scale", 1.5, 1.2, "ease_in_out"))
    assert kfs(env, "v1") == [("position_x", 1.5, 40.0, "linear", ""), ("scale", 0.0, 1.0, "linear", ""), ("scale", 1.5, 1.2, "ease_in_out", ""), ("scale", 3.0, 1.5, "ease_out", "")]
    for prop in ("rotation", "opacity", "reveal", "position_y"):
        run(env, plan_add_keyframe(env, "v2", prop, 1.0, 0.5))
    run(env, plan_add_keyframe(env, "v2", "blur", 1.0, 5.0))
    assert keyframe_properties(env.timeline.get_track("track_v1"), env.timeline.get_clip("v1")) == ("position_x", "position_y", "scale", "rotation", "opacity", "blur", "reveal")
    run(env, plan_add_keyframe(env, "a1", "volume", 2.0, 0.3, "linear"))
    run(env, plan_add_keyframe(env, "img", "scale", 4.0, 1.3))  # a keyframe exactly at the clip end is allowed


@pytest.mark.parametrize("cid, args, match", [
    ("v1", ("volume", 1.0, 0.5), "picture clips have no volume keyframes"),
    ("a1", ("scale", 1.0, 1.0), "audio clips only have volume"),
    ("a1", ("opacity", 1.0, 1.0), "audio clips only have volume"),
    ("v1", ("sparkle", 1.0, 1.0), "not a property that can be animated"),
    ("tx", ("scale", 0.5, 1.0), "animation presets"),
    ("cap", ("opacity", 0.5, 1.0), "animation presets"),
    ("gr", ("opacity", 0.5, 1.0), "animation presets"),
    ("v1", ("scale", -0.1, 1.0), "outside the clip"),
    ("v1", ("scale", 3.5, 1.0), "outside the clip"),
    ("v1", ("scale", float("nan"), 1.0), "valid time"),
    ("v1", ("scale", 1.0, float("inf")), "finite"),
    ("v1", ("scale", 1.0, 0.0), "scale value must be between 0.01 and 20"),
    ("v1", ("opacity", 1.0, 1.5), "opacity value must be between 0 and 1"),
    ("v1", ("reveal", 1.0, -0.2), "reveal value"),
    ("v1", ("blur", 1.0, 500), "blur value"),
    ("a1", ("volume", 1.0, 5), "volume value must be between 0 and 4"),
    ("v1", ("position_x", 1.0, "left"), "must be a number"),
])
def test_keyframe_add_validation(env, cid, args, match):
    rejects(env, plan_add_keyframe, cid, *args, match=match)


def test_keyframe_interpolation_is_limited_to_what_the_renderer_does(env):
    for interp in INTERPOLATIONS:
        run(env, plan_add_keyframe(env, "v1", "scale", 1.0, 1.1, interp), check_undo=False)
        run(env, plan_delete_keyframes(env, "v1", [("scale", 1.0)]), check_undo=False)
    err = rejects(env, plan_add_keyframe, "v1", "scale", 1.0, 1.1, "hold", match="Unknown interpolation")
    assert "hold / step" in str(err)
    run(env, plan_add_keyframe(env, "v1", "scale", 1.0, 1.1))
    rejects(env, plan_set_keyframe_interpolation, "v1", [("scale", 1.0)], "step", match="Unknown interpolation")


def test_no_two_keyframes_of_a_property_at_the_same_time(env):
    run(env, plan_add_keyframe(env, "v1", "scale", 1.0, 1.1), check_undo=False)
    rejects(env, plan_add_keyframe, "v1", "scale", 1.0, 1.3, match="already a scale keyframe at 1.00 s")
    rejects(env, plan_add_keyframe, "v1", "scale", 1.0 + 1e-8, 1.3, match="already a scale keyframe")
    run(env, plan_add_keyframe(env, "v1", "opacity", 1.0, 0.5), check_undo=False)  # another property at the same time is fine
    run(env, plan_add_keyframe(env, "v1", "scale", 2.0, 1.3), check_undo=False)
    rejects(env, plan_move_keyframes, "v1", [("scale", 2.0, 1.0, None)], match="two scale keyframes at 1.00 s")
    # a swap in one edit is valid because only the RESULT matters
    run(env, plan_move_keyframes(env, "v1", [("scale", 1.0, 2.0, None), ("scale", 2.0, 1.0, None)]))
    assert [(t, v) for p, t, v, *_ in kfs(env, "v1") if p == "scale"] == [(1.0, 1.3), (2.0, 1.1)]
    rejects(env, plan_move_keyframes, "v1", [("scale", 1.0, 3.0, None), ("scale", 1.0, 2.5, None)], match="moved twice")
    rejects(env, plan_move_keyframes, "v1", [("scale", 2.0, 2.0 + 1e-8, None), ("scale", 1.0, 2.0, None)], match="two scale keyframes")


def test_delete_move_and_value_edit(env):
    for t, v in ((0.0, 1.0), (1.0, 1.2), (2.0, 1.4)):
        run(env, plan_add_keyframe(env, "v1", "scale", t, v), check_undo=False)
    run(env, plan_add_keyframe(env, "v1", "opacity", 1.0, 0.5), check_undo=False)
    run(env, plan_move_keyframes(env, "v1", [("scale", 1.0, 1.5, 1.25), ("opacity", 1.0, 2.5, None)]))
    assert kfs(env, "v1") == [("scale", 0.0, 1.0, "linear", ""), ("scale", 1.5, 1.25, "linear", ""), ("scale", 2.0, 1.4, "linear", ""), ("opacity", 2.5, 0.5, "linear", "")]  # property order, then time
    run(env, plan_move_keyframes(env, "v1", [("scale", 0.0, 0.0, 2.0)]))  # value only
    assert kfs(env, "v1")[0][2] == 2.0
    assert plan_move_keyframes(env, "v1", [("scale", 0.0, 0.0, 2.0)]).empty  # nothing moves
    rejects(env, plan_move_keyframes, "v1", [("scale", 0.5, 1.0, None)], match="no scale keyframe at 0.50 s")
    rejects(env, plan_move_keyframes, "v1", [("scale", 0.0, 3.5, None)], match="outside the clip")
    rejects(env, plan_move_keyframes, "v1", [("scale", 0.0, 1.0, 99)], match="scale value")
    rejects(env, plan_move_keyframes, "v1", [], match="at least one keyframe")
    rejects(env, plan_move_keyframes, "v1", [("scale", 0.0)], match="old time, new time")
    rejects(env, plan_move_keyframes, "v1", [("volume", 0.0, 1.0, None)], match="no volume keyframes")
    run(env, plan_delete_keyframes(env, "v1", [("scale", 0.0), ("opacity", 2.5), ("scale", 0.0)]))  # duplicates in the selection are fine
    assert [k[:3] for k in kfs(env, "v1")] == [("scale", 1.5, 1.25), ("scale", 2.0, 1.4)]
    rejects(env, plan_delete_keyframes, "v1", [("scale", 0.7)], match="no scale keyframe")
    rejects(env, plan_delete_keyframes, "v1", [], match="at least one keyframe")


def test_editing_an_ai_keyframe_makes_it_a_manual_override(env):
    c = env.timeline.get_clip("v1")
    c.keyframes = [Keyframe("scale", 0.0, 1.0, "linear", "dec1"), Keyframe("scale", 1.0, 1.2, "linear", "dec1"), Keyframe("scale", 2.0, 1.4, "linear", "dec1")]
    run(env, plan_move_keyframes(env, "v1", [("scale", 1.0, 1.5, None)]))
    assert [k.decision_id for k in env.timeline.get_clip("v1").keyframes] == ["dec1", "", "dec1"]  # only the edited one
    run(env, plan_set_keyframe_interpolation(env, "v1", [("scale", 0.0)], "ease_in"))
    assert [(k.interpolation, k.decision_id) for k in env.timeline.get_clip("v1").keyframes] == [("ease_in", ""), ("linear", ""), ("linear", "dec1")]
    run(env, plan_add_keyframe(env, "v1", "opacity", 1.0, 0.8))
    assert [k.decision_id for k in env.timeline.get_clip("v1").keyframes if k.property == "scale"][-1] == "dec1"  # unrelated keyframes keep their owner
    assert plan_set_keyframe_interpolation(env, "v1", [("scale", 0.0)], "ease_in").empty
    rejects(env, plan_set_keyframe_interpolation, "v1", [], "linear", match="at least one keyframe")


def test_copy_and_paste_keyframes(env):
    for t, v in ((0.5, 1.0), (1.0, 1.4), (2.0, 1.8)):
        run(env, plan_add_keyframe(env, "v1", "scale", t, v), check_undo=False)
    run(env, plan_add_keyframe(env, "v1", "opacity", 1.0, 0.3), check_undo=False)
    env.timeline.get_clip("v1").keyframes[0].decision_id = "dec3"
    src = env.timeline.get_clip("v1")
    copied = plan_copy_keyframes(src, [("scale", 1.0), ("scale", 2.0), ("opacity", 1.0)])
    assert [(k.property, k.time) for k in copied] == [("scale", 1.0), ("scale", 2.0), ("opacity", 1.0)] and all(k.decision_id == "" for k in copied)
    assert len(plan_copy_keyframes(src)) == 4
    rejects(env, lambda e, *a: plan_copy_keyframes(src, [("scale", 9.0)]), match="no scale keyframe")
    assert copied[0] is not src.keyframes[0]  # copies, not references
    plan = plan_paste_keyframes(env, "v2", copied, 0.5)  # earliest (1.0) lands at 0.5, spacing kept
    run(env, plan)
    assert kfs(env, "v2") == [("scale", 0.5, 1.4, "linear", ""), ("scale", 1.5, 1.8, "linear", ""), ("opacity", 0.5, 0.3, "linear", "")]
    # dicts (clipboard from JSON) are accepted
    run(env, plan_paste_keyframes(env, "img", [k.to_dict() for k in copied[1:]], 1.0), check_undo=False)
    assert [(k.property, k.time) for k in env.timeline.get_clip("img").keyframes] == [("scale", 2.0), ("opacity", 1.0)]


def test_paste_keyframes_rejections_and_skips(env):
    for t, v in ((0.0, 1.0), (2.0, 1.8)):
        run(env, plan_add_keyframe(env, "v1", "scale", t, v), check_undo=False)
    copied = plan_copy_keyframes(env.timeline.get_clip("v1"))
    rejects(env, plan_paste_keyframes, "v2", copied, 1.5, match="past the end of the clip")  # 1.5 + 2 > 3
    rejects(env, plan_paste_keyframes, "v2", copied, 4.0, match="outside the clip")
    rejects(env, plan_paste_keyframes, "v2", [], 0.0, match="no copied keyframes")
    run(env, plan_paste_keyframes(env, "v2", copied, 0.0), check_undo=False)
    rejects(env, plan_paste_keyframes, "v2", copied, 0.0, match="already a scale keyframe")  # overlaps the first paste
    rejects(env, plan_paste_keyframes, "a1", copied, 0.0, match="None of the copied keyframes")  # audio cannot animate scale
    mixed = [*copied, Keyframe("volume", 1.0, 0.5)]
    plan = plan_paste_keyframes(env, "v3", mixed, 0.0)
    assert any("Skipped" in n and "volume" in n for n in plan.notes)
    run(env, plan)
    assert {k.property for k in env.timeline.get_clip("v3").keyframes} == {"scale"}
    bad = [Keyframe("scale", 0.0, 99.0)]
    rejects(env, plan_paste_keyframes, "v2", bad, 2.5, match="scale value")
    rejects(env, plan_paste_keyframes, "v2", [Keyframe("scale", 0.0, 1.0, "hold")], 2.5, match="Unknown interpolation")


def test_keyframe_paste_between_audio_clips(env):
    run(env, plan_add_keyframe(env, "m1", "volume", 1.0, 0.2), check_undo=False)
    run(env, plan_add_keyframe(env, "m1", "volume", 3.0, 0.9), check_undo=False)
    copied = plan_copy_keyframes(env.timeline.get_clip("m1"))
    run(env, plan_paste_keyframes(env, "a1", copied, 2.0))
    assert [(k.time, k.value) for k in env.timeline.get_clip("a1").keyframes] == [(2.0, 0.2), (4.0, 0.9)]


# ------------------------------------------------------------------------------------------------------------------------ links and groups
def test_link_unlink_group_ungroup(env):
    tl = env.timeline
    plan = plan_link(env, ["v1", "a1"], link_id="LNK")
    run(env, plan)
    assert tl.get_clip("v1").link_id == tl.get_clip("a1").link_id == "LNK"
    assert plan_link(env, ["v1", "a1"], link_id="LNK").empty and plan_link(env, ["v1", "a1"]).empty
    # linking a clip that is linked elsewhere merges the old partner into the new link
    plan = plan_link(env, ["v1", "m1"])
    assert set(plan.updates) == {"v1", "a1", "m1"} and any("joined" in n for n in plan.notes)
    run(env, plan)
    ids = {tl.get_clip(c).link_id for c in ("v1", "a1", "m1")}
    assert len(ids) == 1 and ids != {"LNK"} and ids != {""}
    run(env, plan_unlink(env, ["v1"]))
    assert tl.get_clip("v1").link_id == "" and tl.get_clip("a1").link_id and tl.get_clip("m1").link_id
    run(env, plan_unlink(env, ["a1"], include_mates=True))
    assert tl.get_clip("a1").link_id == tl.get_clip("m1").link_id == ""
    assert plan_unlink(env, ["v1"]).empty
    rejects(env, plan_link, ["v1"], match="at least two clips")
    rejects(env, plan_link, ["v1", "v1"], match="at least two clips")
    rejects(env, plan_link, ["v1", "ghost"], match="no longer exists")

    run(env, plan_group(env, ["v1", "v2", "img"], group_id="G"))
    assert {tl.get_clip(c).group_id for c in ("v1", "v2", "img")} == {"G"}
    run(env, plan_group(env, ["v3", "v1"]))  # v1's old group joins
    assert len({tl.get_clip(c).group_id for c in ("v1", "v2", "img", "v3")}) == 1
    run(env, plan_ungroup(env, ["v3"], include_mates=True))
    assert all(tl.get_clip(c).group_id == "" for c in ("v1", "v2", "img", "v3"))
    rejects(env, plan_group, ["v1"], match="at least two clips")


def test_link_and_group_reject_locked_tracks_and_leave_clip_locks_alone(env):
    tl = env.timeline
    tl.get_track("track_a1").locked = True
    rejects(env, plan_link, ["v1", "a1"], match="Track “A1 Voice-over” is locked")
    rejects(env, plan_group, ["v1", "a1"], match="locked")
    tl.get_track("track_a1").locked = False
    tl.get_clip("v1").locks = frozenset({"item"})
    run(env, plan_link(env, ["v1", "a1"]))  # a link is not an edit of the locked clip's content
    assert tl.get_clip("v1").locks == frozenset({"item"})
    # a partner dragged in by an old link may sit on a locked track too
    tl.get_track("track_a2").locked = True
    tl.get_clip("m1").link_id = tl.get_clip("v2").link_id = "OLD"
    rejects(env, plan_link, ["v2", "img"], match="Track “A2 Music” is locked")


# ------------------------------------------------------------------------------------------------------------------------ persistence / integrity
def test_everything_survives_a_save_round_trip(env):
    tl = env.timeline
    run(env, plan_update_clip(env, "v1", scale=1.2, crop=(0.1, 0.1, 0.8, 0.8)), check_undo=False)
    run(env, plan_set_transition(env, "v2", "DISSOLVE", 1.0), check_undo=False)
    run(env, plan_add_keyframe(env, "v1", "scale", 1.0, 1.4, "ease_in"), check_undo=False)
    run(env, plan_update_clip(env, "m1", muted=True), check_undo=False)
    run(env, plan_set_locks(env, ["m1"], {"audio"}), check_undo=False)
    run(env, plan_link(env, ["v1", "a1"]), check_undo=False)
    run(env, plan_set_speed(env, "v1", 2.0, link_policy="independent"), check_undo=False)
    back = Timeline.from_dict(json.loads(json.dumps(tl.to_dict())))
    assert state(back) == state(tl)
    assert back.get_clip("v1").effects["crop"] == [0.1, 0.1, 0.8, 0.8] and back.get_clip("m1").locks == frozenset({"audio"})


def test_plan_command_reports_the_changed_clip(env):
    cmd = PlanCommand(env, plan_update_clip(env, "v1", scale=1.5))
    cmd.do()
    before, after = cmd.states()
    assert [c.scale for c in before] == [1.0] and [c.scale for c in after] == [1.5] and cmd.affected_clip_ids() == {"v1"} and cmd.major is False


# ------------------------------------------------------------------------------------------------------------------------ randomised
def _model_apply_add(model, prop, t, v, interp):
    if any(abs(t - e[0]) < 1e-6 for e in model.get(prop, [])):
        return False
    model.setdefault(prop, []).append([t, v, interp])
    return True


def test_random_keyframe_edits_match_a_brute_force_model_and_undo_redo_exactly():
    rng = random.Random(20260615)
    tl = Timeline.default()
    tl.insert_clip(media("c", "track_v1", 0.0, 5.0))
    tl.insert_clip(media("d", "track_v2", 0.0, 5.0))
    env = PlanEnv(tl, ASSETS.get)
    props = ["scale", "position_x", "opacity", "rotation"]
    grid = [round(i * 0.25, 6) for i in range(0, 21)]  # 0..5 s: many collisions on purpose
    model: dict[str, list[list]] = {}
    history: list[tuple[PlanCommand, str, str]] = []
    clipboard: list[Keyframe] = []
    ok = bad = 0

    def current() -> dict[str, list[list]]:
        out: dict[str, list[list]] = {}
        for k in tl.get_clip("c").keyframes:
            out.setdefault(k.property, []).append([k.time, k.value, k.interpolation])
        return {p: sorted(v) for p, v in out.items()}

    for step in range(400):
        before = state(tl)
        op = rng.choice(["add", "add", "add", "delete", "move", "interp", "paste", "copy"])
        trial = {p: [list(e) for e in v] for p, v in model.items()}
        expect_ok, plan_fn = True, None
        if op == "add":
            prop, t, v, ip = rng.choice(props), rng.choice(grid), round(rng.uniform(0.1, 2.0), 3), rng.choice(INTERPOLATIONS)
            if prop == "opacity":
                v = round(rng.uniform(0, 1), 3)
            expect_ok = _model_apply_add(trial, prop, t, v, ip)
            plan_fn = lambda: plan_add_keyframe(env, "c", prop, t, v, ip)  # noqa: E731
        elif op == "delete":
            flat = [(p, e[0]) for p, v in trial.items() for e in v]
            if not flat:
                continue
            sel = rng.sample(flat, k=min(len(flat), rng.randint(1, 3)))
            for p, t in sel:
                trial[p] = [e for e in trial[p] if abs(e[0] - t) > 1e-9]
            plan_fn = lambda: plan_delete_keyframes(env, "c", sel)  # noqa: E731
        elif op == "move":
            flat = [(p, e[0]) for p, v in trial.items() for e in v]
            if not flat:
                continue
            sel = rng.sample(flat, k=min(len(flat), rng.randint(1, 3)))
            moves = [(p, t, rng.choice(grid), None) for p, t in sel]
            moved = {(p, t): nt for p, t, nt, _ in moves}
            new_trial: dict[str, list[list]] = {}
            for p, v in trial.items():
                new_trial[p] = [[moved.get((p, e[0]), e[0]), e[1], e[2]] for e in v]
            expect_ok = all(len({round(e[0], 6) for e in v}) == len(v) for v in new_trial.values())
            if expect_ok:
                trial = new_trial
            plan_fn = lambda: plan_move_keyframes(env, "c", moves)  # noqa: E731
        elif op == "interp":
            flat = [(p, e[0]) for p, v in trial.items() for e in v]
            if not flat:
                continue
            sel = rng.sample(flat, k=min(len(flat), rng.randint(1, 3)))
            ip = rng.choice(INTERPOLATIONS)
            for p, t in sel:
                for e in trial[p]:
                    if abs(e[0] - t) < 1e-9:
                        e[2] = ip
            plan_fn = lambda: plan_set_keyframe_interpolation(env, "c", sel, ip)  # noqa: E731
        elif op == "copy":
            sel = [(p, e[0]) for p, v in model.items() for e in v]
            clipboard = plan_copy_keyframes(tl.get_clip("c"), rng.sample(sel, k=min(len(sel), 3))) if sel else []
            continue
        else:  # paste onto the second clip c <- clipboard at a time: model only c, paste on c
            if not clipboard:
                continue
            at = rng.choice(grid)
            base = min(k.time for k in clipboard)
            new_trial = {p: [list(e) for e in v] for p, v in trial.items()}
            expect_ok = True
            for k in clipboard:
                t = round(at + (k.time - base), 6)
                if t > 5.0 + 1e-6 or any(abs(t - e[0]) < 1e-6 for e in new_trial.get(k.property, [])):
                    expect_ok = False
                    break
                new_trial.setdefault(k.property, []).append([t, k.value, k.interpolation])
            if expect_ok:
                trial = new_trial
            plan_fn = lambda: plan_paste_keyframes(env, "c", clipboard, at)  # noqa: E731
        try:
            plan = plan_fn()
        except TimelineError:
            assert not expect_ok, f"step {step} {op}: planner rejected a valid edit"
            assert state(tl) == before
            bad += 1
            continue
        assert expect_ok, f"step {step} {op}: planner accepted an edit the model rejects"
        if plan.empty:
            continue
        cmd = PlanCommand(env, plan)
        cmd.do()
        history.append((cmd, before, state(tl)))
        model = {p: v for p, v in trial.items() if v}
        got = current()
        assert {p: [[round(e[0], 6), e[1], e[2]] for e in v] for p, v in got.items()} == {p: [[round(e[0], 6), e[1], e[2]] for e in sorted(v)] for p, v in model.items()}, f"step {step} {op}"
        kf = tl.get_clip("c").keyframes
        assert all(a.time <= b.time for a, b in zip(kf, kf[1:]) if a.property == b.property)  # sorted per property
        ok += 1
    assert ok > 80 and bad > 10  # the run exercised both outcomes
    final = state(tl)
    for cmd, before, after in reversed(history):
        cmd.undo()
        assert state(tl) == before
    assert json.dumps(Timeline.from_dict(tl.to_dict()).to_dict(), sort_keys=True) == state(tl)
    for cmd, before, after in history:
        cmd.do()
        assert state(tl) == after
    assert state(tl) == final


def test_random_speed_changes_keep_the_source_range_and_undo_exactly():
    rng = random.Random(77)
    tl = Timeline.default()
    tl.insert_clip(media("c", "track_v1", 2.0, 4.0, source_in=3.0))
    tl.insert_clip(media("n", "track_v1", 30.0, 2.0))
    env = PlanEnv(tl, ASSETS.get)
    cmds = []
    for _ in range(60):
        s = rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0])
        before = state(tl)
        plan = plan_set_speed(env, "c", s)
        if plan.empty:
            continue
        cmd = PlanCommand(env, plan)
        cmd.do()
        cmds.append((cmd, before))
        c = tl.get_clip("c")
        assert (c.source_in, c.source_out) == (3.0, 7.0) and c.speed == s and c.timeline_start == 2.0
        assert c.duration == pytest.approx(4.0 / s, abs=1e-12)
    for cmd, before in reversed(cmds):
        cmd.undo()
        assert state(tl) == before
