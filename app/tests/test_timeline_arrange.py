"""Phase 10: align and distribute planners (frame-exact, relative timing kept, linked clips follow, atomic, undoable)."""

from __future__ import annotations

import json
import random

import pytest

from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.ops_arrange import plan_align, plan_distribute
from app.timeline.plan import PlanCommand, PlanEnv
from app.timeline.timeline import Timeline
from app.timeline.timebase import Timebase

ASSETS = {"vid": ("video", 60.0, "Interview"), "aud": ("audio", 120.0, "Voice")}


def state(tl):
    return json.dumps(tl.to_dict(), sort_keys=True)


def clip(cid, track, start, dur, asset="vid", **kw):
    return Clip(cid, track, asset, start, dur, source_in=0.0, source_out=dur, **kw)


@pytest.fixture
def tl():
    t = Timeline.default()
    t.insert_clip(clip("a", "track_v1", 1.0, 2.0))
    t.insert_clip(clip("b", "track_v2", 4.5, 1.0))
    t.insert_clip(clip("c", "track_v3", 8.0, 3.0))
    t.insert_clip(clip("d", "track_v1", 20.0, 1.0))
    t.insert_clip(clip("m", "track_a1", 0.0, 6.0, "aud"))
    return t


@pytest.fixture
def env(tl):
    return PlanEnv(tl, lambda i: ASSETS.get(i), Timebase(30))


def run(env, plan):
    cmd = PlanCommand(env, plan)
    before = state(env.timeline)
    cmd.do()
    after = state(env.timeline)
    cmd.undo()
    assert state(env.timeline) == before
    cmd.do()
    assert state(env.timeline) == after
    return cmd


def start(tl, cid):
    return tl.get_clip(cid).timeline_start


def test_align_starts_and_ends_to_a_time(tl, env):
    run(env, plan_align(env, ["a", "b", "c"], "start", 12.0))
    assert [start(tl, x) for x in "abc"] == [12.0, 12.0, 12.0]
    run(env, plan_align(env, ["a", "b", "c"], "end", 30.0))
    assert [tl.get_clip(x).timeline_end for x in "abc"] == [30.0, 30.0, 30.0]
    run(env, plan_align(env, ["a", "b"], "center", 10.0))
    assert tl.get_clip("a").timeline_start + 1.0 == pytest.approx(10.0) and tl.get_clip("b").timeline_start + 0.5 == pytest.approx(10.0)


def test_align_to_the_selection_itself(tl, env):
    run(env, plan_align(env, ["a", "b", "c"], "first_start"))
    assert [start(tl, x) for x in "abc"] == [1.0, 1.0, 1.0]
    run(env, plan_align(env, ["a", "b", "c"], "last_end"))
    assert len({round(tl.get_clip(x).timeline_end, 6) for x in "abc"}) == 1


def test_align_to_a_time_per_clip(tl, env):
    run(env, plan_align(env, ["a", "c"], "start", {"a": 5.0, "c": 15.0}))
    assert (start(tl, "a"), start(tl, "c")) == (5.0, 15.0)
    with pytest.raises(TimelineError, match="no time"):
        plan_align(env, ["a", "c"], "start", {"a": 5.0})


def test_align_snaps_to_whole_frames_and_never_changes_durations(tl, env):
    before = {x: (tl.get_clip(x).duration, tl.get_clip(x).source_in, tl.get_clip(x).source_out) for x in "abc"}
    run(env, plan_align(env, ["a", "b", "c"], "start", 12.0123))
    frames = [start(tl, x) * 30 for x in "abc"]
    assert all(f == pytest.approx(round(f), abs=1e-6) for f in frames)
    assert {x: (tl.get_clip(x).duration, tl.get_clip(x).source_in, tl.get_clip(x).source_out) for x in "abc"} == before


def test_align_on_one_track_is_refused_with_the_clips_named_and_nothing_changes(tl, env):
    snapshot = state(tl)
    plan = plan_align(env, ["a", "d"], "start", 3.0)  # both on V1 -> on top of each other
    with pytest.raises(TimelineError, match="overlap"):
        PlanCommand(env, plan).do()
    assert state(tl) == snapshot


def test_align_errors_are_explained(tl, env):
    with pytest.raises(TimelineError, match="Unknown alignment"):
        plan_align(env, ["a"], "left", 1.0)
    with pytest.raises(TimelineError, match="needs that time"):
        plan_align(env, ["a"], "start")
    with pytest.raises(TimelineError, match="Select at least one"):
        plan_align(env, [], "start", 1.0)


def test_nothing_to_do_gives_an_empty_plan_with_a_note(tl, env):
    plan = plan_align(env, ["a"], "start", 1.0)
    assert plan.empty and plan.notes


def test_linked_clips_follow_and_conflicting_targets_are_refused(tl, env):
    tl.get_clip("a").link_id = tl.get_clip("m").link_id = "L1"
    plan = plan_align(env, ["a"], "start", 7.0)  # the linked audio follows by the same amount
    assert set(plan.updates) == {"a", "m"}
    run(env, plan)
    assert start(tl, "m") - 0.0 == pytest.approx(6.0) and start(tl, "a") == 7.0
    with pytest.raises(TimelineError, match="cannot be moved to different places"):
        plan_align(env, ["a", "m"], "start", 20.0)  # both selected, mates of each other: a's start -> 20 means delta 13, m's start -> 20 means delta 14
    run(env, plan_align(env, ["a", "m"], "first_start", None, linked=False))
    assert start(tl, "a") == start(tl, "m")


def test_locks_are_respected(tl, env):
    tl.get_clip("b").locks = frozenset({"position"})
    with pytest.raises(TimelineError, match="(?i)lock"):
        plan_align(env, ["a", "b"], "start", 12.0)


def test_distribute_gaps_starts_and_butt(tl, env):
    for cid, track, s, d in (("x1", "track_v4", 0.0, 1.0), ("x2", "track_v5", 2.0, 1.0), ("x3", "track_v6", 7.0, 1.0), ("x4", "track_v4", 10.0, 2.0)):
        tl.insert_clip(clip(cid, track, s, d))
    ids = ["x1", "x2", "x3", "x4"]
    run(env, plan_distribute(env, ids, "gaps"))
    gaps = [start(tl, "x2") - tl.get_clip("x1").timeline_end, start(tl, "x3") - tl.get_clip("x2").timeline_end, start(tl, "x4") - tl.get_clip("x3").timeline_end]
    assert max(gaps) - min(gaps) <= 1 / 30 + 1e-6 and start(tl, "x1") == 0.0 and tl.get_clip("x4").timeline_end == 12.0
    run(env, plan_distribute(env, ids, "starts"))
    steps = [start(tl, b) - start(tl, a) for a, b in zip(ids, ids[1:])]
    assert max(steps) - min(steps) <= 1 / 30 + 1e-6
    run(env, plan_distribute(env, ids, "butt"))
    assert [start(tl, "x2"), start(tl, "x3"), start(tl, "x4")] == [1.0, 2.0, 3.0]


def test_distribute_refuses_what_cannot_work(tl, env):
    with pytest.raises(TimelineError, match="at least 3"):
        plan_distribute(env, ["a", "b"], "gaps")
    with pytest.raises(TimelineError, match="Unknown distribution"):
        plan_distribute(env, ["a", "b", "c"], "wild")
    tl.insert_clip(clip("big1", "track_v4", 0.0, 5.0))
    tl.insert_clip(clip("big2", "track_v5", 1.0, 5.0))
    tl.insert_clip(clip("big3", "track_v6", 2.0, 5.0))  # together longer than the 7 s they span
    with pytest.raises(TimelineError, match="longer together"):
        plan_distribute(env, ["big1", "big2", "big3"], "gaps")


def test_randomised_align_and_distribute_keep_durations_and_undo_exactly(env, tl):
    rng = random.Random(11)
    for i in range(30):
        tl.insert_clip(clip(f"r{i}", f"track_v{4 + i % 3}" if i % 3 else "track_v4", 100.0 + i * 3.0 + rng.uniform(0, 1), rng.uniform(0.5, 2.0)))
    for _ in range(40):
        ids = rng.sample([f"r{i}" for i in range(30)], rng.randint(2, 6))
        before = {x: tl.get_clip(x).duration for x in ids}
        snapshot = state(tl)
        try:
            if rng.random() < 0.5:
                plan = plan_align(env, ids, rng.choice(["first_start", "last_end", "start", "end"]), rng.uniform(90, 200))
            else:
                plan = plan_distribute(env, ids, rng.choice(["gaps", "starts", "butt"]))
            cmd = PlanCommand(env, plan)
            cmd.do()
        except TimelineError:
            assert state(tl) == snapshot
            continue
        assert {x: tl.get_clip(x).duration for x in ids} == before
        cmd.undo()
        assert state(tl) == snapshot
