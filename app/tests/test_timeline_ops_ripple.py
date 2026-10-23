"""Ripple / roll / slip / slide planners (Phase 10, W1-B). Timelines are built directly (no ffmpeg); every plan goes through ``PlanCommand`` so apply / undo / redo are exercised."""

from __future__ import annotations

import random

import pytest

from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.keyframes import Keyframe
from app.timeline.ops_ripple import (
    RIPPLE_SCOPES, plan_ripple_delete, plan_ripple_insert, plan_ripple_move, plan_ripple_trim, plan_roll, plan_slide, plan_slip, resolve_scope, roll_limits, slide_limits,
    slip_limits,
)
from app.timeline.plan import PlanCommand, PlanEnv
from app.timeline.timebase import Timebase
from app.timeline.timeline import Timeline
from app.timeline.track import Track

FPS = 30
TB = Timebase(FPS)
ASSETS = {
    "vid": ("video", 100.0, "Video"),
    "vid2": ("video", 100.0, "Other video"),
    "short": ("video", 12.0, "Short video"),
    "img": ("image", None, "Picture"),
    "aud": ("audio", 200.0, "Voice"),
}


def F(n: int) -> float:
    """Frames -> seconds."""
    return n / FPS


def env_of(tl: Timeline) -> PlanEnv:
    return PlanEnv(tl, lambda aid: ASSETS.get(aid), TB)


def vclip(cid, track, start, dur, *, asset="vid", src_in=0.0, speed=1.0, **kw) -> Clip:
    return Clip(id=cid, track_id=track, asset_id=asset, timeline_start=start, duration=dur, source_in=src_in, source_out=src_in + dur * speed, speed=speed, **kw)


def aclip(cid, track, start, dur, *, src_in=0.0, **kw) -> Clip:
    return vclip(cid, track, start, dur, asset="aud", src_in=src_in, **kw)


def iclip(cid, track, start, dur, **kw) -> Clip:
    return Clip(id=cid, track_id=track, asset_id="img", timeline_start=start, duration=dur, source_in=0.0, source_out=dur, **kw)


def words_for(start: float, n: int, each: float) -> list[dict]:
    return [{"text": f"w{i}", "start": start + i * each, "end": start + (i + 1) * each} for i in range(n)]


def caption(cid, start, dur, *, nwords=4, track="track_v6", **kw) -> Clip:
    each = dur / nwords
    return Clip(id=cid, track_id=track, asset_id="", timeline_start=start, duration=dur, kind="caption",
                text={"text": " ".join(f"w{i}" for i in range(nwords)), "words": words_for(start, nwords, each)}, **kw)


def build(*clips: Clip) -> tuple[Timeline, PlanEnv]:
    tl = Timeline.default()
    for c in clips:
        tl.insert_clip(c)
    return tl, env_of(tl)


def rows(tl: Timeline, tid: str, nd: int = 4) -> list[tuple[str, float, float]]:
    return [(c.id, round(c.timeline_start, nd), round(c.duration, nd)) for c in tl.get_track(tid).clips]


def spans(tl: Timeline, tid: str) -> list[tuple[float, float]]:
    return [(round(c.timeline_start, 4), round(c.timeline_end, 4)) for c in tl.get_track(tid).clips]


def check_invariants(tl: Timeline, env: PlanEnv) -> None:
    for t in tl.tracks:
        cs = t.clips
        assert cs == sorted(cs, key=lambda c: (c.timeline_start, c.id)), f"track {t.id} not sorted"
        assert len({c.id for c in cs}) == len(cs)
        for a, b in zip(cs, cs[1:]):
            assert b.timeline_start >= a.timeline_end - 1e-6, f"overlap on {t.id}: {a.id} / {b.id}"
        for c in cs:
            assert c.duration > 0 and c.timeline_start >= -1e-9
            if c.kind == "media":
                info = env.asset_info(c.asset_id)
                if info and info[0] != "image":
                    assert abs((c.source_out - c.source_in) - c.duration * c.speed) < 2e-3, f"source/duration mismatch on {c.id}"
                    assert c.source_in >= -1e-6
                    assert info[1] is None or c.source_out <= info[1] + 1e-3


def roundtrip(env: PlanEnv, plan) -> PlanCommand:
    """Apply the plan, undo, redo; assert the COMPLETE timeline dict is restored each way. Leaves the plan applied."""
    tl = env.timeline
    before = tl.to_dict()
    cmd = PlanCommand(env, plan)
    cmd.do()
    after = tl.to_dict()
    check_invariants(tl, env)
    cmd.undo()
    assert tl.to_dict() == before
    cmd.do()
    assert tl.to_dict() == after
    return cmd


def must_fail_cleanly(env: PlanEnv, call, *fragments: str) -> str:
    """The planner (or, failing that, applying its plan) must raise a plain-language TimelineError and leave the timeline untouched."""
    before = env.timeline.to_dict()
    with pytest.raises(TimelineError) as ei:
        plan = call()
        PlanCommand(env, plan).do()
    msg = str(ei.value)
    for f in fragments:
        assert f.lower() in msg.lower(), f"{f!r} not in {msg!r}"
    assert env.timeline.to_dict() == before
    return msg


# ====================================================================================================================== scopes
def test_resolve_scope_vocabulary():
    tl = Timeline.default()
    tl.get_track("track_a1").locked = True
    assert RIPPLE_SCOPES == ("track", "tracks", "all")
    tr, sk = resolve_scope(tl, "track", ["track_v1"])
    assert [t.id for t in tr] == ["track_v1"] and sk == []
    tr, sk = resolve_scope(tl, "tracks", ["track_v1"], ["track_v6", "track_a1"])
    assert [t.id for t in tr] == ["track_v1", "track_v6", "track_a1"] and sk == []  # explicit scopes keep the locked track (the planner raises only if it must change)
    tr, sk = resolve_scope(tl, "all", ["track_v1"])
    assert "track_a1" not in [t.id for t in tr] and [t.id for t in sk] == ["track_a1"] and len(tr) == len(tl.tracks) - 1
    with pytest.raises(TimelineError):
        resolve_scope(tl, "everything", ["track_v1"])
    with pytest.raises(TimelineError):
        resolve_scope(tl, "track", ["nope"])


# ====================================================================================================================== ripple delete
def test_ripple_delete_spec_example_and_full_undo_redo():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10))
    before = tl.to_dict()
    plan = plan_ripple_delete(env, ["B"])
    assert tl.to_dict() == before  # planners are pure
    assert plan.removes == ["B"] and set(plan.updates) == {"C"} and plan.shifted == 1
    assert any("Shifted 1 clip" in n for n in plan.notes)
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("C", 10, 10)]
    assert tl.duration == pytest.approx(20)


def test_ripple_delete_preserves_gaps_and_handles_several_clips():
    tl, env = build(vclip("A", "track_v1", 0, 5), vclip("B", "track_v1", 5, 5), vclip("C", "track_v1", 12, 4), vclip("D", "track_v1", 16, 4), vclip("E", "track_v1", 25, 5))
    roundtrip(env, plan_ripple_delete(env, ["B", "D"]))
    # B frees 5 s, D frees 4 s: C moves by 5, E by 9; the 2 s gap before C and 5 s gap before E survive
    assert rows(tl, "track_v1") == [("A", 0, 5), ("C", 7, 4), ("E", 16, 5)]


def test_ripple_delete_without_close_gap_is_a_plain_delete():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10))
    plan = plan_ripple_delete(env, ["B"], close_gap=False)
    assert plan.removes == ["B"] and not plan.updates
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("C", 20, 10)]


def _multitrack():
    """V1: A B C (10 s each). Captions c1 c2 c3 over A B C. Voice-over: one long clip + three short ones. Music spanning 5-25."""
    return build(
        vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10),
        caption("c1", 0, 10), caption("c2", 10, 10), caption("c3", 20, 10),
        aclip("vo1", "track_a1", 0, 9), aclip("vo2", "track_a1", 11, 8), aclip("vo3", "track_a1", 21, 8),
        aclip("mus", "track_a2", 5, 20, src_in=3.0),
    )


def test_scope_track_leaves_other_tracks_alone():
    tl, env = _multitrack()
    roundtrip(env, plan_ripple_delete(env, ["B"], scope="track"))
    assert rows(tl, "track_v1") == [("A", 0, 10), ("C", 10, 10)]
    assert rows(tl, "track_v6") == [("c1", 0, 10), ("c2", 10, 10), ("c3", 20, 10)]
    assert rows(tl, "track_a1") == [("vo1", 0, 9), ("vo2", 11, 8), ("vo3", 21, 8)]
    assert rows(tl, "track_a2") == [("mus", 5, 20)]


def test_scope_all_keeps_captions_and_voice_over_in_sync():
    tl, env = _multitrack()
    plan = plan_ripple_delete(env, ["B"], scope="all")
    assert any("had nothing left" in n for n in plan.notes)  # c2 and vo2 lie entirely inside 10-20
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("C", 10, 10)]
    assert rows(tl, "track_v6") == [("c1", 0, 10), ("c3", 10, 10)]
    assert [(r[1], r[2]) for r in rows(tl, "track_a1")] == [(0, 9), (11, 8)]  # vo1 stays, vo3 follows C, vo2 (inside the cut) is gone
    assert {r[0] for r in rows(tl, "track_a1")} >= {"vo1", "vo3"}
    # the music crossed the whole cut: split and closed, source continuous across the two halves
    mus = sorted(tl.get_track("track_a2").clips, key=lambda c: c.timeline_start)
    assert [(c.timeline_start, c.duration) for c in mus] == [(5, 5), (10, 5)]
    assert mus[0].id == "mus" and mus[1].id != "mus"
    assert mus[0].source_in == pytest.approx(3.0) and mus[0].source_out == pytest.approx(8.0)
    assert mus[1].source_in == pytest.approx(18.0) and mus[1].source_out == pytest.approx(23.0)
    assert tl.duration == pytest.approx(20)


def test_scope_tracks_uses_the_selected_tracks_only():
    tl, env = _multitrack()
    roundtrip(env, plan_ripple_delete(env, ["B"], scope="tracks", selected_track_ids=["track_v6"]))
    assert rows(tl, "track_v6") == [("c1", 0, 10), ("c3", 10, 10)]
    assert rows(tl, "track_a1") == [("vo1", 0, 9), ("vo2", 11, 8), ("vo3", 21, 8)]  # not selected: untouched
    assert rows(tl, "track_a2") == [("mus", 5, 20)]


def test_cut_crossing_one_edge_trims_the_clip():
    tl, env = build(vclip("B", "track_v1", 10, 10), aclip("tail", "track_a2", 5, 10, src_in=2.0), aclip("head", "track_a3", 15, 10, src_in=4.0))
    roundtrip(env, plan_ripple_delete(env, ["B"], scope="all"))
    tail = tl.get_clip("tail")
    assert (tail.timeline_start, tail.duration) == (5, 5) and tail.source_in == pytest.approx(2.0) and tail.source_out == pytest.approx(7.0)  # tail removed
    head = tl.get_clip("head")
    assert head.timeline_start == pytest.approx(10) and head.duration == pytest.approx(5)
    assert head.source_in == pytest.approx(9.0) and head.source_out == pytest.approx(14.0)  # head removed: source advanced by 5 s


def test_locked_track_in_explicit_scope_rejects_naming_it():
    tl, env = _multitrack()
    tl.get_track("track_a1").locked = True
    must_fail_cleanly(env, lambda: plan_ripple_delete(env, ["B"], scope="tracks", selected_track_ids=["track_a1"]), "A1 Voice-over", "locked")
    tl.get_track("track_v1").locked = True  # the edited track itself
    must_fail_cleanly(env, lambda: plan_ripple_delete(env, ["B"]), "V1 Main Video", "locked")


def test_locked_explicit_track_with_nothing_to_move_is_fine():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), aclip("early", "track_a1", 0, 5))
    tl.get_track("track_a1").locked = True
    roundtrip(env, plan_ripple_delete(env, ["B"], scope="tracks", selected_track_ids=["track_a1"]))
    assert rows(tl, "track_a1") == [("early", 0, 5)]


def test_scope_all_skips_locked_track_with_a_note():
    tl, env = _multitrack()
    tl.get_track("track_a1").locked = True
    plan = plan_ripple_delete(env, ["B"], scope="all")
    assert any("A1 Voice-over" in n and "locked" in n and "not shifted" in n for n in plan.notes)
    assert not any(c in plan.affected_ids() for c in ("vo1", "vo2", "vo3"))
    roundtrip(env, plan)
    assert rows(tl, "track_a1") == [("vo1", 0, 9), ("vo2", 11, 8), ("vo3", 21, 8)]
    assert rows(tl, "track_v6") == [("c1", 0, 10), ("c3", 10, 10)]


def test_scope_all_does_not_mention_a_locked_track_it_would_not_have_touched():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), aclip("early", "track_a1", 0, 5))
    tl.get_track("track_a1").locked = True
    assert not any("locked" in n for n in plan_ripple_delete(env, ["B"], scope="all").notes)


def test_locked_clip_that_would_have_to_move_rejects_naming_it():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10, locks=frozenset({"position"})))
    must_fail_cleanly(env, lambda: plan_ripple_delete(env, ["B"]), "locked", "position")
    tl2, env2 = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), aclip("vo", "track_a1", 12, 5, locks=frozenset({"item"})))
    must_fail_cleanly(env2, lambda: plan_ripple_delete(env2, ["A"], scope="all"), "locked", "Voice")  # also under "all": a locked CLIP on an unlocked track blocks the edit


def test_locked_clip_before_the_cut_is_not_a_problem_and_locked_target_is():
    tl, env = build(vclip("A", "track_v1", 0, 10, locks=frozenset({"item"})), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10))
    roundtrip(env, plan_ripple_delete(env, ["B"]))
    assert rows(tl, "track_v1") == [("A", 0, 10), ("C", 10, 10)]
    tl2, env2 = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10, locks=frozenset({"timing"})))
    must_fail_cleanly(env2, lambda: plan_ripple_delete(env2, ["B"]), "locked")


def test_linked_clips_are_deleted_and_followers_shift_even_outside_the_scope():
    tl, env = build(
        vclip("A", "track_v1", 0, 10, link_id="LA"), vclip("B", "track_v1", 10, 10, link_id="LB"), vclip("C", "track_v1", 20, 10, link_id="LC"),
        aclip("a", "track_a1", 0, 10, link_id="LA"), aclip("b", "track_a1", 10, 10, link_id="LB"), aclip("c", "track_a1", 20, 10, link_id="LC"), aclip("x", "track_a1", 40, 5),
    )
    plan = plan_ripple_delete(env, ["B"])
    assert set(plan.removes) == {"B", "b"}
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("C", 10, 10)]
    assert rows(tl, "track_a1") == [("a", 0, 10), ("c", 10, 10), ("x", 40, 5)]  # c follows its link partner; unlinked x stays (scope "track")
    tl2, env2 = build(vclip("A", "track_v1", 0, 10, link_id="LA"), vclip("B", "track_v1", 10, 10, link_id="LB"), aclip("b", "track_a1", 10, 10, link_id="LB"))
    assert plan_ripple_delete(env2, ["B"], linked=False).removes == ["B"]


def test_linked_partner_on_a_locked_track_blocks_the_delete():
    tl, env = build(vclip("B", "track_v1", 10, 10, link_id="LB"), aclip("b", "track_a1", 10, 10, link_id="LB"))
    tl.get_track("track_a1").locked = True
    must_fail_cleanly(env, lambda: plan_ripple_delete(env, ["B"], scope="all"), "A1 Voice-over", "locked")


def test_grouped_clips_are_deleted_and_follow_shifts():
    tl, env = build(
        vclip("B", "track_v1", 10, 10, group_id="g1"), vclip("G", "track_v2", 12, 6, group_id="g1"),
        vclip("D", "track_v1", 30, 10, group_id="g2"), vclip("H", "track_v2", 31, 8, group_id="g2"), vclip("Z", "track_v2", 45, 3),
    )
    plan = plan_ripple_delete(env, ["B"])
    assert set(plan.removes) == {"B", "G"}
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("D", 20, 10)]
    assert rows(tl, "track_v2") == [("H", 21, 8), ("Z", 45, 3)]  # the group mate follows although V2 is out of scope; unrelated Z stays
    tl2, env2 = build(vclip("B", "track_v1", 10, 10, group_id="g1"), vclip("G", "track_v2", 12, 6, group_id="g1"))
    assert plan_ripple_delete(env2, ["B"], grouped=False).removes == ["B"]


def test_text_straddling_the_cut_is_split_between_words_and_text_without_words_is_refused():
    tl, env = build(vclip("B", "track_v1", 10, 10), caption("cap", 5, 20, nwords=4))  # words at 5-10, 10-15, 15-20, 20-25
    roundtrip(env, plan_ripple_delete(env, ["B"], scope="all"))
    caps = sorted(tl.get_track("track_v6").clips, key=lambda c: c.timeline_start)
    assert [(c.timeline_start, c.duration) for c in caps] == [(5, 5), (10, 5)]
    assert [w["text"] for w in caps[0].text["words"]] == ["w0"]
    assert [w["text"] for w in caps[1].text["words"]] == ["w3"]
    assert caps[1].text["words"][0]["start"] == pytest.approx(10.0)  # the words moved with their piece
    title = Clip(id="t1", track_id="track_v5", asset_id="", timeline_start=5, duration=20, kind="text", text={"content": "Title"})
    tl2, env2 = build(vclip("B", "track_v1", 10, 10), title)
    must_fail_cleanly(env2, lambda: plan_ripple_delete(env2, ["B"], scope="all"), "single block of text")


def test_short_leftovers_are_dropped_with_a_note_and_split_keeps_ids_and_keyframes_sane():
    kf = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 16.0, 1.0)]
    tl, env = build(vclip("B", "track_v1", 10, 10), aclip("m", "track_a2", F(299), 20.5 - F(299)), vclip("top", "track_v2", 8, 16, keyframes=kf, transition={"type": "FADE", "duration": 0.5}))
    plan = plan_ripple_delete(env, ["B"], scope="all")
    assert any("fragment" in n for n in plan.notes)  # m: one frame before the cut is below the minimum length
    roundtrip(env, plan)
    assert [c.id for c in tl.get_track("track_a2").clips] == ["m"]
    m = tl.get_clip("m")  # the one-frame piece before the cut was dropped; the piece after it kept the clip's id and closed up
    assert m.timeline_start == pytest.approx(10.0) and m.duration == pytest.approx(0.5) and m.source_in == pytest.approx(20.0 - F(299))
    left, right = sorted(tl.get_track("track_v2").clips, key=lambda c: c.timeline_start)
    assert (left.id, left.timeline_start, left.duration) == ("top", 8, 2) and left.transition == {"type": "FADE", "duration": 0.5}
    assert right.id != "top" and right.transition is None and right.timeline_start == pytest.approx(10) and right.duration == pytest.approx(4)
    assert right.source_in == pytest.approx(12.0) and left.source_out == pytest.approx(2.0)
    assert [(k.property, round(k.time, 3), round(k.value, 3)) for k in left.keyframes] == [("opacity", 0.0, 0.0), ("opacity", 2.0, 0.125)]
    assert right.keyframes[0].time == 0.0 and right.keyframes[0].value == pytest.approx(0.75) and right.keyframes[-1].value == pytest.approx(1.0)


def test_frame_arithmetic_never_drifts_over_many_ripple_deletes():
    n = 60
    tl, env = build(*(vclip(f"c{i}", "track_v1", F(7 * i), F(7)) for i in range(n)))
    rng = random.Random(3)
    alive = list(range(n))
    for _ in range(40):
        i = rng.choice(alive)
        alive.remove(i)
        PlanCommand(env, plan_ripple_delete(env, [f"c{i}"])).do()
    for k, i in enumerate(alive):
        c = tl.get_clip(f"c{i}")
        assert c.timeline_start == TB.to_seconds(7 * k)  # exactly the grid value, not "close to"
        assert c.duration == TB.to_seconds(7)


def test_audio_precision_is_kept_and_picture_tracks_stay_on_the_frame_grid():
    odd = 1.234567
    tl, env = build(aclip("a1", "track_a1", 1.0, odd), aclip("a2", "track_a1", 1.0 + odd, 3.0), vclip("v", "track_v1", 4.0, 5.0))
    assert not TB.is_on_frame(odd)
    roundtrip(env, plan_ripple_delete(env, ["a1"], scope="all"))
    assert tl.get_clip("a2").timeline_start == pytest.approx(1.0, abs=1e-9)  # exact: closed by the sample-precise length, not rounded to a frame
    v = tl.get_clip("v")
    assert TB.is_on_frame(v.timeline_start) and abs(v.timeline_start - (4.0 - odd)) <= 0.5 / FPS + 1e-9  # picture clips moved by a whole number of frames


def test_ripple_delete_input_errors():
    tl, env = build(vclip("A", "track_v1", 0, 10))
    with pytest.raises(TimelineError):
        plan_ripple_delete(env, [])
    with pytest.raises(TimelineError):
        plan_ripple_delete(env, ["ghost"])
    with pytest.raises(TimelineError):
        plan_ripple_delete(env, ["A"], scope="galaxy")


def test_locked_linked_follower_that_would_have_to_shift_blocks_the_delete():
    tl, env = build(vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10, link_id="LC"), aclip("c", "track_a1", 20, 10, link_id="LC"))
    tl.get_track("track_a1").locked = True
    must_fail_cleanly(env, lambda: plan_ripple_delete(env, ["B"]), "A1 Voice-over", "locked")  # the follower c would have to move although A1 is outside the scope


def test_every_planner_is_pure_and_never_aliases_timeline_clips():
    kf = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 10.0, 1.0)]
    tl, env = build(
        vclip("A", "track_v1", 0, 10, keyframes=kf, link_id="LA"), vclip("B", "track_v1", 10, 10, src_in=5.0, link_id="LB"), vclip("C", "track_v1", 20, 10),
        aclip("a", "track_a1", 0, 10, link_id="LA"), aclip("b", "track_a1", 10, 10, src_in=4.0, link_id="LB"), caption("cap", 5, 20), aclip("m", "track_a2", 5, 20, src_in=3.0),
    )
    before = tl.to_dict()
    plans = [
        plan_ripple_delete(env, ["B"], scope="all"), plan_ripple_trim(env, "A", "end", 6, scope="all"), plan_ripple_trim(env, "B", "start", 12, scope="all"),
        plan_ripple_insert(env, [vclip("N", "track_v1", 0, 4, asset="vid2")], 8, "track_v1", scope="all"), plan_ripple_move(env, ["A"], 20), plan_roll(env, "A", "B", 9),
        plan_slip(env, "B", 2), plan_slide(env, "B", 1),
    ]
    assert tl.to_dict() == before
    for plan in plans:
        for cid, c in plan.updates.items():
            assert c is not tl.get_clip(cid)
    plans[0].updates["C"].timeline_start = 999  # mutating a plan never reaches the timeline
    assert tl.to_dict() == before


def test_planners_reject_nonsense_numbers():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10, src_in=5.0))
    nan = float("nan")
    for call in (lambda: plan_ripple_trim(env, "A", "end", nan), lambda: plan_ripple_insert(env, [vclip("N", "track_v1", 0, 1)], nan, "track_v1"),
                 lambda: plan_ripple_move(env, ["A"], float("inf")), lambda: plan_roll(env, "A", "B", nan), lambda: plan_slip(env, "B", nan), lambda: plan_slide(env, "A", nan)):
        with pytest.raises(TimelineError):
            call()


# ====================================================================================================================== ripple trim
def _chain():
    return build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10, src_in=5.0), vclip("C", "track_v1", 20, 10))


def test_ripple_trim_end_shorter_and_longer():
    tl, env = _chain()
    roundtrip(env, plan_ripple_trim(env, "A", "end", 8))
    assert rows(tl, "track_v1") == [("A", 0, 8), ("B", 8, 10), ("C", 18, 10)]
    assert tl.get_clip("A").source_out == pytest.approx(8.0)
    tl, env = _chain()
    plan = plan_ripple_trim(env, "A", "end", 12)
    assert plan.shifted == 2
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("A", 0, 12), ("B", 12, 10), ("C", 22, 10)]
    assert tl.get_clip("A").source_out == pytest.approx(12.0) and tl.get_clip("B").source_in == pytest.approx(5.0)
    assert tl.duration == pytest.approx(32)  # +2 s exactly


def test_ripple_trim_start_keeps_the_clip_in_place_and_moves_what_follows():
    tl, env = _chain()
    roundtrip(env, plan_ripple_trim(env, "B", "start", 12))  # remove the first 2 s of B
    b = tl.get_clip("B")
    assert (b.timeline_start, b.duration) == (10, 8) and b.source_in == pytest.approx(7.0) and b.source_out == pytest.approx(15.0)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("B", 10, 8), ("C", 18, 10)]
    tl, env = _chain()
    roundtrip(env, plan_ripple_trim(env, "B", "start", 8))  # reveal 2 s of earlier source
    b = tl.get_clip("B")
    assert (b.timeline_start, b.duration) == (10, 12) and b.source_in == pytest.approx(3.0) and b.source_out == pytest.approx(15.0)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("B", 10, 12), ("C", 22, 10)]


def test_ripple_trim_is_clamped_by_source_and_minimum_length_with_notes():
    tl, env = build(vclip("S", "track_v1", 0, 10, asset="short"), vclip("N", "track_v1", 10, 5))
    plan = plan_ripple_trim(env, "S", "end", 20)  # the media is 12 s long
    assert any("end of the source media" in n for n in plan.notes)
    roundtrip(env, plan)
    assert tl.get_clip("S").duration == pytest.approx(12) and tl.get_clip("N").timeline_start == pytest.approx(12)
    tl, env = build(vclip("S", "track_v1", 0, 10, src_in=1.0), vclip("N", "track_v1", 10, 5))
    plan = plan_ripple_trim(env, "S", "start", -4)  # only 1 s of media before the in point
    assert any("start of the source media" in n for n in plan.notes)
    roundtrip(env, plan)
    s = tl.get_clip("S")
    assert (s.timeline_start, s.duration, s.source_in) == (0, 11, pytest.approx(0.0)) and tl.get_clip("N").timeline_start == pytest.approx(11)
    tl, env = _chain()
    plan = plan_ripple_trim(env, "A", "end", 0.001)  # cannot go below the minimum length
    assert any("minimum clip length" in n for n in plan.notes)
    roundtrip(env, plan)
    assert tl.get_clip("A").duration == pytest.approx(F(2))
    must_fail_cleanly(env, lambda: plan_ripple_trim(env, "A", "end", 0.001), "any further")


def test_ripple_trim_nothing_to_do_and_bad_edge():
    tl, env = _chain()
    p = plan_ripple_trim(env, "A", "end", 10)
    assert p.empty and p.notes
    with pytest.raises(TimelineError):
        plan_ripple_trim(env, "A", "middle", 5)


def test_ripple_trim_other_tracks_follow_the_scope_and_are_cut_or_opened():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), caption("c", 0, 20), aclip("vo", "track_a1", 12, 6))
    roundtrip(env, plan_ripple_trim(env, "A", "end", 6, scope="all"))  # cut 6-10 everywhere
    assert rows(tl, "track_v1") == [("A", 0, 6), ("B", 6, 10)]
    assert [(round(c.timeline_start, 3), round(c.duration, 3)) for c in sorted(tl.get_track("track_v6").clips, key=lambda c: c.timeline_start)] == [(0, 6), (6, 10)]
    assert rows(tl, "track_a1") == [("vo", 8, 6)]
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), aclip("vo", "track_a1", 12, 6), aclip("keep", "track_a2", 0, 8))
    roundtrip(env, plan_ripple_trim(env, "A", "end", 13, scope="tracks", selected_track_ids=["track_a1"]))
    assert rows(tl, "track_v1") == [("A", 0, 13), ("B", 13, 10)]
    assert rows(tl, "track_a1") == [("vo", 15, 6)] and rows(tl, "track_a2") == [("keep", 0, 8)]


def test_ripple_trim_linked_clips_with_a_coinciding_edge_trim_together():
    tl, env = build(vclip("V", "track_v1", 0, 10, link_id="L"), aclip("a", "track_a1", 0, 10, link_id="L"), vclip("W", "track_v1", 10, 5, link_id="M"), aclip("b", "track_a1", 10, 5, link_id="M"))
    roundtrip(env, plan_ripple_trim(env, "V", "end", 7))
    assert rows(tl, "track_v1") == [("V", 0, 7), ("W", 7, 5)] and rows(tl, "track_a1") == [("a", 0, 7), ("b", 7, 5)]
    # the audio's media limits the video trim as well
    tl, env = build(vclip("V", "track_v1", 0, 10, link_id="L"), vclip("a", "track_a1", 0, 10, link_id="L", asset="short"))
    plan = plan_ripple_trim(env, "V", "end", 15)
    roundtrip(env, plan)
    assert tl.get_clip("V").duration == pytest.approx(12) and tl.get_clip("a").duration == pytest.approx(12) and any("Short video" in n for n in plan.notes)


def test_ripple_trim_locked_downstream_clip_rejects():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10, locks=frozenset({"position"})))
    must_fail_cleanly(env, lambda: plan_ripple_trim(env, "A", "end", 8), "locked", "position")
    tl2, env2 = build(vclip("A", "track_v1", 0, 10, locks=frozenset({"timing"})), vclip("B", "track_v1", 10, 10))
    must_fail_cleanly(env2, lambda: plan_ripple_trim(env2, "A", "end", 8), "locked")


def test_ripple_trim_keyframes_follow_the_trim_window():
    kf = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 10.0, 1.0)]
    tl, env = build(vclip("A", "track_v1", 0, 10, keyframes=kf), vclip("B", "track_v1", 10, 5))
    roundtrip(env, plan_ripple_trim(env, "A", "end", 5))
    ks = tl.get_clip("A").keyframes
    assert [(round(k.time, 3), round(k.value, 3)) for k in ks] == [(0.0, 0.0), (5.0, 0.5)]
    for k in tl.get_clip("A").keyframes:
        assert 0 <= k.time <= tl.get_clip("A").duration + 1e-9


# ====================================================================================================================== ripple insert
def test_ripple_insert_shifts_everything_after_and_splits_the_straddler():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10))
    new = vclip("N", "track_v1", 0, 5, asset="vid2", src_in=40.0)
    plan = plan_ripple_insert(env, [new], 10, "track_v1")
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("A", 0, 10), ("N", 10, 5), ("B", 15, 10), ("C", 25, 10)]
    tl, env = build(vclip("A", "track_v1", 0, 20, src_in=2.0), vclip("B", "track_v1", 20, 10))
    plan = plan_ripple_insert(env, [vclip("N", "track_v1", 0, 5, asset="vid2")], 8, "track_v1")
    roundtrip(env, plan)
    cs = tl.get_track("track_v1").clips
    assert [(c.timeline_start, c.duration) for c in cs] == [(0, 8), (8, 5), (13, 12), (25, 10)]
    left, mid, right, last = cs
    assert left.id == "A" and mid.id == "N" and right.id not in ("A", "N", "B")
    assert left.source_out == pytest.approx(10.0) and right.source_in == pytest.approx(10.0) and right.source_out == pytest.approx(22.0)
    assert last.id == "B" and any("Split 1 clip" in n for n in plan.notes)


def test_ripple_insert_scope_all_and_multi_clip_layout():
    tl, env = build(vclip("A", "track_v1", 0, 20), caption("c", 0, 20), aclip("vo", "track_a1", 4, 4), aclip("mus", "track_a2", 0, 30))
    clips = [vclip("N1", "track_v1", 100, 5, asset="vid2"), aclip("N2", "track_a1", 102, 3)]  # layout: audio starts 2 s after the picture, span 5 s
    plan = plan_ripple_insert(env, clips, 10, "track_v1", scope="all")
    roundtrip(env, plan)
    assert spans(tl, "track_v1") == [(0, 10), (10, 15), (15, 25)]
    assert tl.duration == pytest.approx(35)  # 30 s of music + 5 s inserted
    assert spans(tl, "track_a1") == [(4, 8), (12, 15)]  # vo (4-8, before the insert point) stays; N2 sits 2 s after the picture
    n2 = tl.get_clip("N2")
    assert n2.timeline_start == pytest.approx(12) and n2.duration == pytest.approx(3)
    assert len(tl.get_track("track_v6").clips) == 2 and len(tl.get_track("track_a2").clips) == 2
    caps = sorted(tl.get_track("track_v6").clips, key=lambda c: c.timeline_start)
    assert [(c.timeline_start, c.duration) for c in caps] == [(0, 10), (15, 10)]
    assert [w["text"] for w in caps[0].text["words"]] == ["w0", "w1"] and [w["text"] for w in caps[1].text["words"]] == ["w2", "w3"]
    assert caps[1].text["words"][0]["start"] == pytest.approx(15.0)


def test_ripple_insert_moves_the_timing_of_an_inserted_caption_and_refuses_to_split_plain_text():
    cap = caption("capN", 0, 4, nwords=2)
    tl, env = build(vclip("A", "track_v1", 0, 10))
    cap.text["start"], cap.text["end"] = 0.0, 4.0
    roundtrip(env, plan_ripple_insert(env, [cap], 6, "track_v6"))
    n = tl.get_clip("capN")
    assert n.timeline_start == pytest.approx(6) and n.text["start"] == pytest.approx(6) and n.text["end"] == pytest.approx(10)
    assert n.text["words"][0]["start"] == pytest.approx(6)  # the words travelled with the clip
    title = Clip(id="t1", track_id="track_v5", asset_id="", timeline_start=0, duration=10, kind="text", text={"content": "Title"})
    tl2, env2 = build(vclip("A", "track_v1", 0, 10), title)
    must_fail_cleanly(env2, lambda: plan_ripple_insert(env2, [vclip("N", "track_v1", 0, 2, asset="vid2")], 5, "track_v1", scope="all"), "single block of text")


def test_ripple_insert_scope_all_skips_locked_tracks_with_a_note_and_grouped_clips_follow():
    tl, env = build(vclip("A", "track_v1", 0, 10), aclip("vo", "track_a1", 5, 10), vclip("D", "track_v1", 20, 5, group_id="g"), vclip("H", "track_v2", 21, 3, group_id="g"))
    tl.get_track("track_a1").locked = True
    plan = plan_ripple_insert(env, [vclip("N", "track_v1", 0, 4, asset="vid2")], 5, "track_v1", scope="all")
    assert any("A1 Voice-over" in n and "not shifted" in n for n in plan.notes)
    roundtrip(env, plan)
    assert rows(tl, "track_a1") == [("vo", 5, 10)]
    assert tl.get_clip("H").timeline_start == pytest.approx(25)  # V2 is in scope anyway, the group would follow regardless
    tl2, env2 = build(vclip("A", "track_v1", 0, 10), vclip("D", "track_v1", 20, 5, group_id="g"), vclip("H", "track_v2", 21, 3, group_id="g"))
    roundtrip(env2, plan_ripple_insert(env2, [vclip("N", "track_v1", 0, 4, asset="vid2")], 5, "track_v1"))
    assert tl2.get_clip("H").timeline_start == pytest.approx(25) and tl2.get_clip("D").timeline_start == pytest.approx(24)


def test_ripple_insert_ids_locks_and_bounds():
    tl, env = build(vclip("A", "track_v1", 0, 10))
    dup = vclip("A", "track_v1", 0, 3, asset="vid2")  # an existing id: it gets a fresh one
    plan = plan_ripple_insert(env, [dup], 10, "track_v1")
    assert plan.adds[0].id != "A" and plan.adds[0].timeline_start == pytest.approx(10)
    roundtrip(env, plan)
    assert rows(tl, "track_v1")[0] == ("A", 0, 10) and len(tl.get_track("track_v1").clips) == 2
    must_fail_cleanly(env, lambda: plan_ripple_insert(env, [vclip("N", "track_v1", 0, 3)], -2, "track_v1"), "before 0")
    tl.get_track("track_v1").locked = True
    must_fail_cleanly(env, lambda: plan_ripple_insert(env, [vclip("N", "track_v1", 0, 3)], 4, "track_v1"), "V1 Main Video", "locked")
    with pytest.raises(TimelineError):
        plan_ripple_insert(env, [], 4, "track_v1")


def test_ripple_insert_at_the_frame_grid_and_locked_clip_downstream():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10))
    plan = plan_ripple_insert(env, [vclip("N", "track_v1", 0, 1.0, asset="vid2")], 10.01, "track_v1")  # 10.01 s -> frame 300.3 -> 300
    assert plan.adds[0].timeline_start == TB.to_seconds(300)
    tl2, env2 = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10, locks=frozenset({"item"})))
    must_fail_cleanly(env2, lambda: plan_ripple_insert(env2, [vclip("N", "track_v1", 0, 1, asset="vid2")], 10, "track_v1"), "locked")


# ====================================================================================================================== ripple move
def test_ripple_move_to_the_right_and_left():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10))
    roundtrip(env, plan_ripple_move(env, ["A"], 20))  # drop A between B and C
    assert rows(tl, "track_v1") == [("B", 0, 10), ("A", 10, 10), ("C", 20, 10)]
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10), vclip("C", "track_v1", 20, 10))
    roundtrip(env, plan_ripple_move(env, ["C"], -20))
    assert rows(tl, "track_v1") == [("C", 0, 10), ("A", 10, 10), ("B", 20, 10)]
    assert tl.duration == pytest.approx(30)


def test_ripple_move_into_the_middle_of_a_clip_splits_it_and_inside_own_span_is_a_noop():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 20, src_in=5.0))
    roundtrip(env, plan_ripple_move(env, ["A"], 20))  # A would start at 20 of the current layout -> inside B
    cs = tl.get_track("track_v1").clips
    assert [(c.timeline_start, c.duration) for c in cs] == [(0, 10), (10, 10), (20, 10)]
    assert [c.id for c in cs][1] == "A" and cs[0].id == "B" and cs[2].id not in ("A", "B")
    assert cs[0].source_out == pytest.approx(15.0) and cs[2].source_in == pytest.approx(15.0)
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10))
    assert plan_ripple_move(env, ["A"], 4).empty
    assert plan_ripple_move(env, ["A"], 0).empty


def test_ripple_move_block_linked_followers_and_rejections():
    tl, env = build(
        vclip("A", "track_v1", 0, 10, link_id="LA"), vclip("B", "track_v1", 10, 10, link_id="LB"), vclip("C", "track_v1", 20, 10, link_id="LC"),
        aclip("a", "track_a1", 0, 10, link_id="LA"), aclip("b", "track_a1", 10, 10, link_id="LB"), aclip("c", "track_a1", 20, 10, link_id="LC"),
    )
    plan = plan_ripple_move(env, ["A"], 20)
    assert set(plan.updates) == {"A", "a", "B", "b"}  # C / c net zero
    roundtrip(env, plan)
    assert rows(tl, "track_v1") == [("B", 0, 10), ("A", 10, 10), ("C", 20, 10)]
    assert rows(tl, "track_a1") == [("b", 0, 10), ("a", 10, 10), ("c", 20, 10)]
    tl, env = build(vclip("A", "track_v1", 0, 5), vclip("B", "track_v1", 5, 5), vclip("C", "track_v1", 10, 5), vclip("D", "track_v1", 15, 5), vclip("E", "track_v1", 20, 5))
    roundtrip(env, plan_ripple_move(env, ["B", "C"], 20))  # a block of two (start 5 -> 25 of the current layout) moves behind E
    assert rows(tl, "track_v1") == [("A", 0, 5), ("D", 5, 5), ("E", 10, 5), ("B", 15, 5), ("C", 20, 5)]
    tl, env = build(vclip("A", "track_v1", 0, 5), vclip("B", "track_v1", 5, 5), vclip("C", "track_v1", 10, 5), vclip("X", "track_v2", 0, 5))
    must_fail_cleanly(env, lambda: plan_ripple_move(env, ["A", "C"], 20), "next to each other")
    must_fail_cleanly(env, lambda: plan_ripple_move(env, ["A", "X"], 20), "one track")
    p = plan_ripple_move(env, ["B"], -50)  # clamped at the start of the timeline
    assert any("start of the timeline" in n for n in p.notes)
    roundtrip(env, p)
    assert rows(tl, "track_v1")[0] == ("B", 0, 5)
    assert plan_ripple_move(env, ["B"], 7, scope="all").notes[0].startswith("A ripple move only shifts")


def test_ripple_move_locked_track_rejects():
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10))
    tl.get_track("track_v1").locked = True
    must_fail_cleanly(env, lambda: plan_ripple_move(env, ["A"], 20), "V1 Main Video", "locked")


# ====================================================================================================================== roll
def test_roll_moves_the_shared_cut_and_keeps_the_combined_duration():
    tl, env = build(vclip("L", "track_v1", 0, 10), vclip("R", "track_v1", 10, 10, src_in=50.0))
    total_before = tl.duration
    roundtrip(env, plan_roll(env, "L", "R", 9.5))
    left, right = tl.get_clip("L"), tl.get_clip("R")
    assert (left.timeline_start, left.duration, left.source_in, left.source_out) == (0, 9.5, 0.0, pytest.approx(9.5))
    assert right.timeline_start == pytest.approx(9.5) and right.duration == pytest.approx(10.5)
    assert right.source_in == pytest.approx(49.5) and right.source_out == pytest.approx(60.0)
    assert tl.duration == pytest.approx(total_before) and left.timeline_end == pytest.approx(right.timeline_start)
    roundtrip(env, plan_roll(env, "L", "R", 11))  # and back later
    assert tl.get_clip("L").duration == pytest.approx(11) and tl.get_clip("R").source_in == pytest.approx(51.0)
    assert plan_roll(env, "L", "R", 11).empty


def test_roll_rejections_leave_everything_unchanged():
    tl, env = build(vclip("L", "track_v1", 0, 10, asset="short"), vclip("R", "track_v1", 10, 10, src_in=0.0), vclip("F", "track_v1", 30, 5), vclip("O", "track_v2", 0, 5))
    must_fail_cleanly(env, lambda: plan_roll(env, "L", "R", 13), "source media", "between")  # left would pass the end of its 12 s media
    must_fail_cleanly(env, lambda: plan_roll(env, "L", "R", 9.5), "before its current start")  # right already starts at source 0
    must_fail_cleanly(env, lambda: plan_roll(env, "L", "R", 0.01), "minimum clip length")
    must_fail_cleanly(env, lambda: plan_roll(env, "R", "F", 25), "not next to each other")
    must_fail_cleanly(env, lambda: plan_roll(env, "L", "O", 5), "different tracks")
    must_fail_cleanly(env, lambda: plan_roll(env, "R", "L", 5), "not next to each other")
    with pytest.raises(TimelineError):
        plan_roll(env, "L", "L", 5)
    lo, hi = roll_limits(env, "L", "R")
    assert lo == pytest.approx(10.0) and hi == pytest.approx(12.0)  # R already starts at the beginning of its media; L's media ends at 12 s


def test_roll_tolerates_a_sub_frame_gap_and_closes_it_and_works_on_stills():
    tl, env = build(vclip("L", "track_v1", 0, 10), vclip("R", "track_v1", 10 + 0.02, 10, src_in=20.0), iclip("P", "track_v3", 0, 5), iclip("Q", "track_v3", 5, 5))
    roundtrip(env, plan_roll(env, "L", "R", 9))
    assert tl.get_clip("L").timeline_end == pytest.approx(9) and tl.get_clip("R").timeline_start == pytest.approx(9)
    roundtrip(env, plan_roll(env, "P", "Q", 7))  # pictures have no source limit
    assert spans(tl, "track_v3") == [(0, 7), (7, 10)]
    assert tl.get_clip("Q").source_in >= 0


def test_roll_audio_keeps_sample_precision():
    tl, env = build(aclip("L", "track_a1", 0, 10), aclip("R", "track_a1", 10, 10, src_in=30.0))
    roundtrip(env, plan_roll(env, "L", "R", 9.123456))
    assert tl.get_clip("L").duration == pytest.approx(round(9.123456 * 48000) / 48000, abs=1e-9)
    assert not TB.is_on_frame(tl.get_clip("L").duration)


def test_roll_linked_members_follow():
    tl, env = build(
        vclip("L", "track_v1", 0, 10, link_id="X"), vclip("R", "track_v1", 10, 10, src_in=10.0, link_id="Y"),
        aclip("la", "track_a1", 0, 10, link_id="X"), aclip("ra", "track_a1", 10, 10, src_in=10.0, link_id="Y"),
    )
    plan = plan_roll(env, "L", "R", 8)
    assert set(plan.updates) == {"L", "R", "la", "ra"}
    roundtrip(env, plan)
    assert spans(tl, "track_a1") == [(0, 8), (8, 20)] and tl.get_clip("ra").source_in == pytest.approx(8.0)
    assert set(plan_roll(env, "L", "R", 7, linked=False).updates) == {"L", "R"}
    tl2, env2 = build(vclip("L", "track_v1", 0, 10, link_id="X"), vclip("R", "track_v1", 10, 10, src_in=10.0, link_id="Y"),
                      vclip("la", "track_a1", 0, 10, link_id="X", asset="short"), aclip("ra", "track_a1", 10, 10, src_in=10.0, link_id="Y"))
    must_fail_cleanly(env2, lambda: plan_roll(env2, "L", "R", 13), "Linked clip")  # the linked audio runs out of media first


def test_roll_locked_clip_or_track_rejects():
    tl, env = build(vclip("L", "track_v1", 0, 10, locks=frozenset({"timing"})), vclip("R", "track_v1", 10, 10, src_in=10.0))
    must_fail_cleanly(env, lambda: plan_roll(env, "L", "R", 9), "locked")
    tl2, env2 = build(vclip("L", "track_v1", 0, 10), vclip("R", "track_v1", 10, 10, src_in=10.0))
    tl2.get_track("track_v1").locked = True
    must_fail_cleanly(env2, lambda: plan_roll(env2, "L", "R", 9), "locked", "V1 Main Video")


# ====================================================================================================================== slip
def test_slip_moves_only_the_source_range():
    kf = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 5.0, 1.0)]
    tl, env = build(vclip("C", "track_v1", 10, 10, src_in=20.0, keyframes=kf), vclip("N", "track_v1", 20, 5))
    roundtrip(env, plan_slip(env, "C", 5))
    c = tl.get_clip("C")
    assert (c.timeline_start, c.duration) == (10, 10) and c.source_in == pytest.approx(25.0) and c.source_out == pytest.approx(35.0)
    assert [(k.time, k.value) for k in c.keyframes] == [(0.0, 0.0), (5.0, 1.0)]  # clip-local: untouched
    assert tl.get_clip("N").timeline_start == 20
    assert slip_limits(env, "C") == (pytest.approx(-25.0), pytest.approx(65.0))
    roundtrip(env, plan_slip(env, "C", -25))  # right up to the start of the media
    assert tl.get_clip("C").source_in == pytest.approx(0.0)


def test_slip_beyond_the_media_is_rejected_with_the_range():
    tl, env = build(vclip("C", "track_v1", 10, 10, src_in=20.0))
    must_fail_cleanly(env, lambda: plan_slip(env, "C", -21), "can only slip between", "-20.00")
    must_fail_cleanly(env, lambda: plan_slip(env, "C", 71), "can only slip between", "+70.00")
    roundtrip(env, plan_slip(env, "C", 70))
    assert tl.get_clip("C").source_out == pytest.approx(100.0)
    assert plan_slip(env, "C", 0).empty


def test_slip_nothing_to_slip_for_stills_and_captions_and_speed_is_respected():
    tl, env = build(iclip("P", "track_v3", 0, 5), caption("cap", 0, 5), vclip("S", "track_v1", 0, 5, src_in=10.0, speed=2.0))
    must_fail_cleanly(env, lambda: plan_slip(env, "P", 1), "nothing to slip", "Picture")
    must_fail_cleanly(env, lambda: plan_slip(env, "cap", 1), "no source media")
    roundtrip(env, plan_slip(env, "S", 3))
    s = tl.get_clip("S")
    assert (s.source_in, s.source_out, s.duration, s.speed) == (pytest.approx(13.0), pytest.approx(23.0), 5, 2.0)


def test_slip_linked_follow_and_quantisation():
    tl, env = build(vclip("V", "track_v1", 0, 10, src_in=10.0, link_id="L"), aclip("a", "track_a1", 0, 10, src_in=10.0, link_id="L"), iclip("P", "track_v3", 0, 5, link_id="L"))
    plan = plan_slip(env, "V", 2.51)  # picture clip: a whole number of frames (75)
    assert set(plan.updates) == {"V", "a"}  # the linked still has nothing to slip and is left alone
    roundtrip(env, plan)
    assert tl.get_clip("V").source_in == pytest.approx(10 + F(75)) and tl.get_clip("a").source_in == pytest.approx(10 + F(75))
    tl2, env2 = build(vclip("V", "track_v1", 0, 10, src_in=10.0, link_id="L"), vclip("a", "track_a1", 0, 10, src_in=1.0, link_id="L", asset="short"))
    must_fail_cleanly(env2, lambda: plan_slip(env2, "V", -5), "Linked clip")
    assert set(plan_slip(env2, "V", -5, linked=False).updates) == {"V"}


def test_slip_locked_clip_rejects():
    tl, env = build(vclip("C", "track_v1", 10, 10, src_in=20.0, locks=frozenset({"timing"})))
    must_fail_cleanly(env, lambda: plan_slip(env, "C", 2), "locked")


# ====================================================================================================================== slide
def _slide_setup():
    return build(vclip("A", "track_v1", 0, 10, src_in=0.0), vclip("X", "track_v1", 10, 10, src_in=30.0), vclip("B", "track_v1", 20, 10, src_in=40.0))


def test_slide_moves_the_clip_and_its_neighbours_absorb_it():
    tl, env = _slide_setup()
    roundtrip(env, plan_slide(env, "X", 2))
    assert rows(tl, "track_v1") == [("A", 0, 12), ("X", 12, 10), ("B", 22, 8)]
    a, x, b = tl.get_clip("A"), tl.get_clip("X"), tl.get_clip("B")
    assert a.source_out == pytest.approx(12.0) and (x.source_in, x.source_out) == (30.0, 40.0)  # the slid clip's own source range does not change
    assert b.source_in == pytest.approx(42.0) and b.source_out == pytest.approx(50.0)
    assert tl.duration == pytest.approx(30)
    tl, env = _slide_setup()
    roundtrip(env, plan_slide(env, "X", -3))
    assert rows(tl, "track_v1") == [("A", 0, 7), ("X", 7, 10), ("B", 17, 13)]
    assert tl.get_clip("B").source_in == pytest.approx(37.0)
    lo, hi = slide_limits(env, "X")  # now A is 7 s and B is 13 s long: each may shrink to 2 frames
    assert lo == pytest.approx(-(7 - F(2))) and hi == pytest.approx(13 - F(2))
    assert plan_slide(env, "X", 0).empty


def test_slide_with_a_missing_neighbour_on_one_side():
    tl, env = build(vclip("X", "track_v1", 0, 10), vclip("B", "track_v1", 10, 10, src_in=20.0))
    must_fail_cleanly(env, lambda: plan_slide(env, "X", -2), "before 0:00")
    roundtrip(env, plan_slide(env, "X", 2))  # leaves a 2 s gap before X
    assert rows(tl, "track_v1") == [("X", 2, 10), ("B", 12, 8)]
    tl, env = build(vclip("A", "track_v1", 0, 10), vclip("X", "track_v1", 10, 10))
    roundtrip(env, plan_slide(env, "X", 5))
    assert rows(tl, "track_v1") == [("A", 0, 15), ("X", 15, 10)]
    roundtrip(env, plan_slide(env, "X", -8))
    assert rows(tl, "track_v1") == [("A", 0, 7), ("X", 7, 10)]


def test_slide_rejections():
    tl, env = _slide_setup()
    must_fail_cleanly(env, lambda: plan_slide(env, "X", 10), "minimum clip length", "can slide between")  # B would vanish
    must_fail_cleanly(env, lambda: plan_slide(env, "X", -10), "minimum clip length")
    tl2, env2 = build(vclip("A", "track_v1", 0, 10, asset="short"), vclip("X", "track_v1", 10, 10), vclip("B", "track_v1", 20, 10))
    must_fail_cleanly(env2, lambda: plan_slide(env2, "X", 3), "source media")  # A would need 13 s of a 12 s file
    tl3, env3 = build(vclip("A", "track_v1", 0, 10), vclip("X", "track_v1", 12, 10), vclip("B", "track_v1", 22, 10))
    must_fail_cleanly(env3, lambda: plan_slide(env3, "X", 1), "not directly before")
    tl4, env4 = build(vclip("X", "track_v1", 0, 10), vclip("B", "track_v1", 13, 10))
    must_fail_cleanly(env4, lambda: plan_slide(env4, "X", 1), "not directly after")
    tl5, env5 = build(vclip("A", "track_v1", 0, 10), vclip("X", "track_v1", 10, 10, src_in=5.0), vclip("B", "track_v1", 20, 10, src_in=0.0))
    must_fail_cleanly(env5, lambda: plan_slide(env5, "X", -2), "before its current start")  # B has no earlier media to reveal
    tl6, env6 = _slide_setup()
    tl6.get_track("track_v1").locked = True
    must_fail_cleanly(env6, lambda: plan_slide(env6, "X", 1), "locked")


def test_slide_works_for_pictures_and_linked_audio_stays_in_sync():
    tl, env = build(iclip("P", "track_v3", 0, 5), iclip("Q", "track_v3", 5, 5), iclip("R", "track_v3", 10, 5))
    roundtrip(env, plan_slide(env, "Q", 2))
    assert spans(tl, "track_v3") == [(0, 7), (7, 12), (12, 15)]
    tl, env = build(
        vclip("A", "track_v1", 0, 10, link_id="LA"), vclip("X", "track_v1", 10, 10, src_in=30.0, link_id="LX"), vclip("B", "track_v1", 20, 10, src_in=40.0, link_id="LB"),
        aclip("a", "track_a1", 0, 10, link_id="LA"), aclip("x", "track_a1", 10, 10, src_in=30.0, link_id="LX"), aclip("b", "track_a1", 20, 10, src_in=40.0, link_id="LB"),
    )
    plan = plan_slide(env, "X", 2)
    assert set(plan.updates) == {"A", "X", "B", "a", "x", "b"}
    roundtrip(env, plan)
    assert spans(tl, "track_a1") == spans(tl, "track_v1") == [(0, 12), (12, 22), (22, 30)]
    assert tl.get_clip("b").source_in == pytest.approx(42.0)
    assert set(plan_slide(env, "x", 1, linked=False).updates) == {"a", "x", "b"}  # sliding the audio clip alone still adjusts the audio neighbours


# ====================================================================================================================== randomised: ripple vs. a brute-force frame model
MIN_FRAMES = 2  # 0.05 s at 30 fps rounds up to 2 frames


def _model_pieces(s: int, e: int, kind: str, a: int, b: int):
    """Brute force on integer frames: the (start, end) pieces of [s, e) after cutting [a, b) or opening room b - a at a; slivers dropped unless the clip is untouched."""
    out = []
    if kind == "cut":
        parts = [(x, 0) for x in [(s, min(e, a))] if x[1] > x[0]] + [((max(s, b), e), -(b - a)) for _ in [0] if e > max(s, b)]
        for (ps, pe), sh in parts:
            out.append((ps + sh, pe + sh, ps, pe))
    else:
        span = b - a
        if e <= a:
            out.append((s, e, s, e))
        elif s >= a:
            out.append((s + span, e + span, s, e))
        else:
            out.append((s, a, s, a))
            out.append((a + span, e + span, a, e))
    if len(out) == 1 and out[0][2] == s and out[0][3] == e:
        return [(out[0][0], out[0][1])]
    return [(n0, n1) for n0, n1, o0, o1 in out if o1 - o0 >= MIN_FRAMES]


def _model_apply(track: list[tuple[int, int]], kind: str, a: int, b: int, skip: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    out = []
    for iv in track:
        if iv == skip:
            continue
        out += _model_pieces(iv[0], iv[1], kind, a, b)
    return sorted(out)


def _frames(tl: Timeline, tid: str) -> list[tuple[int, int]]:
    out = []
    for c in tl.get_track(tid).clips:
        s, e = c.timeline_start * FPS, c.timeline_end * FPS
        assert abs(s - round(s)) < 1e-6 and abs(e - round(e)) < 1e-6, f"{c.id} left the frame grid"
        out.append((round(s), round(e)))
    return sorted(out)


def _random_timeline(rng: random.Random) -> tuple[Timeline, PlanEnv]:
    tl = Timeline.default()
    n = 0
    for tid, asset in (("track_v1", "vid"), ("track_v2", "vid2"), ("track_a1", "aud")):
        t = rng.randint(0, 8)
        for _ in range(rng.randint(5, 10)):
            ln = rng.randint(3, 25)
            n += 1
            tl.insert_clip(vclip(f"{tid}_{n}", tid, F(t), F(ln), asset=asset, src_in=40.0 + F(t)))
            t += ln + rng.choice([0, 0, 0, 1, 4, 9])
    return tl, env_of(tl)


@pytest.mark.parametrize("seed", range(12))
def test_random_ripple_sequences_match_the_brute_force_model_and_undo_redo(seed):
    rng = random.Random(seed)
    tl, env = _random_timeline(rng)
    tracks = ["track_v1", "track_v2", "track_a1"]
    snaps = [tl.to_dict()]
    cmds: list[PlanCommand] = []
    for step in range(14):
        scope = rng.choice(["track", "all"])
        in_scope = ["track_v1"] if scope == "track" else tracks
        model = {tid: _frames(tl, tid) for tid in tracks}
        dur_before = tl.duration
        v1 = tl.get_track("track_v1").clips
        op = rng.choice(["delete", "insert", "trim_end", "trim_start"])
        if not v1 and op != "insert":
            op = "insert"
        if op == "delete":
            c = rng.choice(v1)
            a, b = round(c.timeline_start * FPS), round(c.timeline_end * FPS)
            plan = plan_ripple_delete(env, [c.id], scope=scope)
            expect = {tid: (_model_apply(model[tid], "cut", a, b, skip=(a, b) if tid == "track_v1" else None) if tid in in_scope else
                            [iv for iv in model[tid] if iv != (a, b)] if tid == "track_v1" else model[tid]) for tid in tracks}
        elif op == "insert":
            at = rng.randint(0, (max((iv[1] for tid in tracks for iv in model[tid]), default=0)) + 5)
            ln = rng.randint(3, 15)
            plan = plan_ripple_insert(env, [vclip(f"new{seed}_{step}", "track_v1", F(at), F(ln), asset="vid2", src_in=10.0)], F(at), "track_v1", scope=scope)
            expect = {tid: (_model_apply(model[tid], "gap", at, at + ln) if tid in in_scope else model[tid]) for tid in tracks}
            expect["track_v1"] = sorted([*expect["track_v1"], (at, at + ln)])
        elif op == "trim_end":
            c = rng.choice(v1)
            s, e = round(c.timeline_start * FPS), round(c.timeline_end * FPS)
            ne = rng.randint(s + 3, e + 12)
            plan = plan_ripple_trim(env, c.id, "end", F(ne), scope=scope)
            kind, (a, b) = ("cut", (ne, e)) if ne < e else ("gap", (e, ne))
            if ne == e:
                continue
            expect = {tid: (_model_apply(model[tid], kind, a, b, skip=(s, e) if tid == "track_v1" else None) if tid in in_scope else model[tid]) for tid in tracks}
            expect["track_v1"] = sorted([*expect["track_v1"], (s, ne)])
        else:
            c = rng.choice(v1)
            s, e = round(c.timeline_start * FPS), round(c.timeline_end * FPS)
            ns = rng.randint(max(0, s - 10), e - 3)
            if ns == s:
                continue
            plan = plan_ripple_trim(env, c.id, "start", F(ns), scope=scope)
            kind, (a, b) = ("cut", (s, ns)) if ns > s else ("gap", (s, s + (s - ns)))
            expect = {tid: (_model_apply(model[tid], kind, a, b, skip=(s, e) if tid == "track_v1" else None) if tid in in_scope else model[tid]) for tid in tracks}
            expect["track_v1"] = sorted([*expect["track_v1"], (s, e + (s - ns))])
        cmd = PlanCommand(env, plan)
        cmd.do()
        cmds.append(cmd)
        snaps.append(tl.to_dict())
        check_invariants(tl, env)
        for tid in tracks:
            assert _frames(tl, tid) == expect[tid], f"seed {seed} step {step} {op} {scope} on {tid}"
        if op in ("delete",) and scope == "all":
            assert tl.duration == pytest.approx(max((iv[1] for tid in tracks for iv in expect[tid]), default=0) / FPS)
        assert plan.shifted >= 0 and dur_before >= 0
    for i in range(len(cmds), 0, -1):  # undo everything, comparing the complete timeline each time
        cmds[i - 1].undo()
        assert tl.to_dict() == snaps[i - 1]
    for i, cmd in enumerate(cmds, 1):  # and redo it all
        cmd.do()
        assert tl.to_dict() == snaps[i]


# ====================================================================================================================== performance shape
def test_thousand_clip_ripple_is_one_rebuild_per_track(monkeypatch):
    n = 1000
    clips = [vclip(f"v{i}", "track_v1", float(i), 1.0) for i in range(n)] + [aclip(f"a{i}", "track_a1", float(i), 1.0, src_in=float(i % 100)) for i in range(n)]
    tl, env = build(*clips)
    plan = plan_ripple_delete(env, ["v10"], scope="all")
    assert set(plan.removes) == {"v10", "a10"} and plan.shifted == 2 * (n - 11)  # a10 lay entirely inside the removed second
    assigned: list[str] = []
    orig = Track.__setattr__

    def spy(self, name, value):
        if name == "clips":
            assigned.append(self.id)
        orig(self, name, value)

    monkeypatch.setattr(Track, "__setattr__", spy)
    cmd = PlanCommand(env, plan)
    cmd.do()
    assert sorted(assigned) == ["track_a1", "track_v1"]  # exactly one rebuild per touched track
    assert len(tl.get_track("track_v1").clips) == n - 1 and tl.get_clip("v11").timeline_start == pytest.approx(10.0)
    assert tl.get_clip("a11").timeline_start == pytest.approx(10.0) and tl.get_clip("a9").timeline_start == pytest.approx(9.0)
    check_invariants(tl, env)
    assigned.clear()
    cmd.undo()
    assert sorted(assigned) == ["track_a1", "track_v1"]
    assert tl.get_clip("v11").timeline_start == pytest.approx(11.0)
    assigned.clear()
    roundtrip_plan = plan_ripple_insert(env, [vclip("ins", "track_v1", 0, 5, asset="vid2")], 100.0, "track_v1", scope="all")
    PlanCommand(env, roundtrip_plan).do()
    assert sorted(assigned) == ["track_a1", "track_v1"]


# ====================================================================================================================== randomised: linked clips stay in sync, roll / slide / slip bounds
@pytest.mark.parametrize("seed", range(8))
def test_random_ripples_keep_linked_audio_in_sync(seed):
    rng = random.Random(100 + seed)
    clips, t = [], rng.randint(0, 5)
    for i in range(10):
        ln = rng.randint(4, 24)
        clips += [vclip(f"v{i}", "track_v1", F(t), F(ln), src_in=20.0, link_id=f"L{i}"), aclip(f"a{i}", "track_a1", F(t), F(ln), src_in=20.0, link_id=f"L{i}")]
        t += ln + rng.choice([0, 0, 0, 3])
    tl, env = build(*clips)
    cmds, snaps = [], [tl.to_dict()]
    for step in range(12):
        v1 = tl.get_track("track_v1").clips
        op = rng.choice(["delete", "insert", "trim_end", "trim_start", "move"])
        c = rng.choice(v1)
        s, e = round(c.timeline_start * FPS), round(c.timeline_end * FPS)
        if op == "delete":
            plan = plan_ripple_delete(env, [c.id])
        elif op == "insert":
            at = rng.randint(0, e + 5)
            plan = plan_ripple_insert(env, [vclip(f"n{seed}_{step}", "track_v1", F(at), F(rng.randint(3, 12)), asset="vid2", src_in=30.0)], F(at), "track_v1")
        elif op == "trim_end":
            ne = rng.randint(s + 3, e + 8)
            if ne == e:
                continue
            plan = plan_ripple_trim(env, c.id, "end", F(ne))
        elif op == "trim_start":
            ns = rng.randint(s - 6, e - 3)
            if ns == s:
                continue
            plan = plan_ripple_trim(env, c.id, "start", F(ns))
        else:
            if len(v1) < 2:
                continue
            plan = plan_ripple_move(env, [c.id], F(rng.randint(-40, 60) or 1))
        cmd = PlanCommand(env, plan)
        cmd.do()
        cmds.append(cmd)
        snaps.append(tl.to_dict())
        check_invariants(tl, env)
        by_link: dict[str, list[Clip]] = {}
        for cl in tl.all_clips():
            if cl.link_id:
                by_link.setdefault(cl.link_id, []).append(cl)
        for lid, members in by_link.items():
            assert len(members) <= 2, f"{op}: link {lid} grew"
            if len(members) == 2:
                a, b = members
                assert a.timeline_start == pytest.approx(b.timeline_start, abs=1e-6) and a.duration == pytest.approx(b.duration, abs=1e-6), f"seed {seed} step {step} {op}: pair {lid} out of sync"
    for i in range(len(cmds), 0, -1):
        cmds[i - 1].undo()
        assert tl.to_dict() == snaps[i - 1]


@pytest.mark.parametrize("seed", range(8))
def test_random_roll_slide_slip_accept_exactly_what_the_media_allows(seed):
    rng = random.Random(500 + seed)
    clips, t = [], 0
    for i in range(8):
        ln = rng.randint(4, 30)
        if rng.random() < 0.4:
            clips.append(vclip(f"c{i}", "track_v1", F(t), F(ln), asset="short", src_in=F(rng.randint(0, 360 - ln))))
        else:
            clips.append(vclip(f"c{i}", "track_v1", F(t), F(ln), src_in=F(rng.randint(0, 1800))))
        t += ln
    tl, env = build(*clips)
    md = 2  # minimum length in frames

    def length_f(c):  # frames of source media
        return round(ASSETS[c.asset_id][1] * FPS)

    def fr(x):
        return round(x * FPS)

    for _step in range(40):
        cs = tl.get_track("track_v1").clips
        op = rng.choice(["roll", "slide", "slip"])
        i = rng.randrange(len(cs) - 1) if op == "roll" else rng.randrange(1, len(cs) - 1) if op == "slide" else rng.randrange(len(cs))
        before = tl.to_dict()
        if op == "roll":
            L, R = cs[i], cs[i + 1]
            T = rng.randint(fr(L.timeline_start) + 1, fr(R.timeline_end) - 1)
            lo = max(fr(L.timeline_start) + md, fr(R.timeline_start) - fr(R.source_in))
            hi = min(fr(R.timeline_end) - md, fr(L.timeline_start) + length_f(L) - fr(L.source_in))
            ok = lo <= T <= hi
            if T == fr(L.timeline_end):
                continue
            if not ok:
                with pytest.raises(TimelineError):
                    plan_roll(env, L.id, R.id, F(T))
                assert tl.to_dict() == before
                continue
            ls, re_ = fr(L.timeline_start), fr(R.timeline_end)
            cmd = PlanCommand(env, plan_roll(env, L.id, R.id, F(T)))
            cmd.do()
            L2, R2 = tl.get_clip(L.id), tl.get_clip(R.id)
            assert (fr(L2.timeline_start), fr(L2.timeline_end), fr(R2.timeline_start), fr(R2.timeline_end)) == (ls, T, T, re_)
            assert L2.source_in == L.source_in and R2.source_out == pytest.approx(R.source_out)
        elif op == "slide":
            P, X, N = cs[i - 1], cs[i], cs[i + 1]
            d = rng.randint(-14, 14)
            if d == 0:
                continue
            ok = (fr(P.duration) + d >= md and fr(N.duration) - d >= md and fr(P.source_out) + d <= length_f(P) and fr(N.source_in) + d >= 0)
            if not ok:
                with pytest.raises(TimelineError):
                    plan_slide(env, X.id, F(d))
                assert tl.to_dict() == before
                continue
            cmd = PlanCommand(env, plan_slide(env, X.id, F(d)))
            cmd.do()
            P2, X2, N2 = tl.get_clip(P.id), tl.get_clip(X.id), tl.get_clip(N.id)
            assert fr(X2.timeline_start) == fr(X.timeline_start) + d and X2.duration == X.duration
            assert (X2.source_in, X2.source_out) == (X.source_in, X.source_out)
            assert fr(P2.timeline_end) == fr(X2.timeline_start) and fr(N2.timeline_start) == fr(X2.timeline_end) and fr(N2.timeline_end) == fr(N.timeline_end)
            assert fr(P2.timeline_start) == fr(P.timeline_start)
        else:
            X = cs[i]
            d = rng.randint(-40, 40)
            if d == 0:
                continue
            ok = fr(X.source_in) + d >= 0 and fr(X.source_out) + d <= length_f(X)
            if not ok:
                with pytest.raises(TimelineError):
                    plan_slip(env, X.id, F(d))
                assert tl.to_dict() == before
                continue
            cmd = PlanCommand(env, plan_slip(env, X.id, F(d)))
            cmd.do()
            X2 = tl.get_clip(X.id)
            assert (X2.timeline_start, X2.duration) == (X.timeline_start, X.duration) and fr(X2.source_in) == fr(X.source_in) + d
        check_invariants(tl, env)
        cmd.undo()
        assert tl.to_dict() == before
        cmd.do()  # keep the edit: the next step works on the new arrangement
