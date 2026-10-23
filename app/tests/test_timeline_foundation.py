"""Phase 10 foundation: frame-exact timebase, markers / range / link / group / lock persistence, EditPlan validation + atomic PlanCommand."""

from __future__ import annotations

import json
import random

import pytest

from app.core.constants import SCHEMA_VERSION
from app.core.exceptions import TimelineError
from app.media.asset import AssetType
from app.project.project import Project
from app.project.project_schema import migrate_document, validate_document
from app.timeline.clip import Clip
from app.timeline.markers import Marker
from app.timeline.plan import EditPlan, PlanCommand, PlanEnv, changed_aspects, clip_fits_track
from app.timeline.timebase import Timebase, timebase_of
from app.timeline.timeline import Timeline, new_clip_id


# ------------------------------------------------------------------------------------------------ timebase
@pytest.mark.parametrize("fps", [24, 25, 30, 60, 23.976, 29.97, 59.94])
def test_frame_round_trip_and_no_drift(fps):
    tb = Timebase(fps)
    for n in range(0, 300_000, 977):
        assert tb.to_frame(tb.to_seconds(n)) == n
    t = 0.0
    for _ in range(50_000):  # 50k single-frame steps never drift off the integer grid
        t = tb.add_frames(t, 1)
    assert tb.to_frame(t) == 50_000 and tb.is_on_frame(t)
    for _ in range(1000):  # alternating +3 / -3 frames returns exactly to the start
        t = tb.add_frames(tb.add_frames(t, 3), -3)
    assert tb.to_frame(t) == 50_000


def test_snap_modes_and_audio_precision():
    tb = Timebase(30)
    assert tb.to_frame(1.0 / 30 * 10.4, "round") == 10 and tb.to_frame(1.0 / 30 * 10.4, "ceil") == 11 and tb.to_frame(1.0 / 30 * 10.9, "floor") == 10
    assert tb.to_frame(10 / 30, "floor") == 10 and tb.to_frame(10 / 30, "ceil") == 10  # already on a boundary
    assert tb.snap_samples(1.0000123, 48000) != tb.snap(1.0000123)  # audio keeps sample precision, pictures snap to frames
    assert abs(tb.snap_samples(1.0000123, 48000) - 1.0000123) < 1 / 96000 + 1e-12


def test_timecode_format_and_parse():
    tb = Timebase(30)
    assert tb.format(61.5) == "00:01:01:15" and tb.format(None) == "--:--:--:--" and tb.format(1.2349, with_frames=False) == "00:00:01.235"
    assert tb.parse("00:01:01:15") == 61.5 and tb.parse("1:01:15") == 61.5 and tb.parse("45f") == 1.5 and tb.parse("1.25s") == 1.25 and tb.parse("2") == 2.0
    for bad in ("", "abc", "00:00:00:30", "1:2:3:4:5:x"):
        with pytest.raises(ValueError):
            tb.parse(bad)
    assert Timebase(29.97).format(Timebase(29.97).to_seconds(1800)) == "00:01:00:00"  # NDF label at the nominal rate
    assert Timebase(None).fps == 30 and Timebase("30000/1001").rate.denominator == 1001 and Timebase(-5).fps == 30


# ------------------------------------------------------------------------------------------------ model persistence
def test_markers_range_links_locks_persist_and_old_documents_open():
    tl = Timeline.default()
    tl.insert_marker(Marker("m2", 5.0, "later", "review"))
    tl.insert_marker(Marker("m1", 2.0, "early", "scene", color="#fff", note="n"))
    tl.set_range((1.0, 4.0))
    c = Clip("c1", "track_v1", "a", 0, 2, link_id="L1", group_id="G1", locks=frozenset({"timing", "item"}))
    tl.insert_clip(c)
    back = Timeline.from_dict(json.loads(json.dumps(tl.to_dict())))
    assert [m.id for m in back.markers] == ["m1", "m2"] and back.range == (1.0, 4.0)
    assert back.markers[0].display_color == "#fff" and back.markers[1].display_color != ""
    got = back.get_clip("c1")
    assert (got.link_id, got.group_id, got.locks, got.protected) == ("L1", "G1", frozenset({"timing", "item"}), True)
    assert tl.next_marker(2.0).id == "m2" and tl.prev_marker(5.0).id == "m1" and tl.prev_marker(2.0) is None
    plain = Timeline.default().to_dict()
    assert "markers" not in plain and "range" not in plain  # nothing new is written for a timeline that does not use it
    assert Timeline.from_dict({"tracks": plain["tracks"], "markers": [{"nope": 1}], "range": [3, 1]}).range is None  # damaged optional data is ignored, not fatal
    assert Clip.from_dict(Clip("x", "t", "a", 0, 1, locks=frozenset({"bogus"})).to_dict() | {"locks": ["bogus", "audio"]}).locks == frozenset({"audio"})


def test_schema_9_migration_is_additive():
    p = Project.new("S")
    doc = p.to_document()
    doc8 = {**doc, "schema_version": 8}
    up = migrate_document(doc8)
    assert up["schema_version"] == SCHEMA_VERSION == 9 and up["timeline"] == doc8["timeline"]
    validate_document(up)
    again = Project.from_document(up)
    assert again.timeline.markers == [] and again.timeline.range is None
    newer = {**doc, "schema_version": SCHEMA_VERSION + 1}
    with pytest.raises(Exception):
        migrate_document(newer)


# ------------------------------------------------------------------------------------------------ plan / command
def make_env(tl: Timeline, durations: dict[str, float | None] | None = None) -> PlanEnv:
    d = durations or {}

    def info(aid: str):
        if not aid:
            return None
        if aid.startswith("img"):
            return ("image", None, aid)
        return ("audio" if aid.startswith("aud") else "video", d.get(aid, 10.0), aid)

    return PlanEnv(tl, info, Timebase(30))


def vclip(cid, start, dur, track="track_v1", asset="vid", src_in=0.0, **kw) -> Clip:
    return Clip(cid, track, asset, start, dur, src_in, src_in + dur * kw.get("speed", 1.0), **kw)


def timeline_of(*clips: Clip) -> Timeline:
    tl = Timeline.default()
    for c in clips:
        tl.insert_clip(c)
    return tl


def state(tl: Timeline) -> str:
    return json.dumps(tl.to_dict(), sort_keys=True)


def moved(c: Clip, **kw) -> Clip:
    n = c.snapshot()
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def test_plan_command_applies_undoes_and_redoes_atomically():
    a, b, c = vclip("a", 0, 2), vclip("b", 2, 2), vclip("c", 4, 2)
    tl = timeline_of(a, b, c)
    env = make_env(tl)
    before = state(tl)
    plan = EditPlan("shift two", {"b": moved(b, timeline_start=7.0), "c": moved(c, timeline_start=9.0)}, removes=["a"])
    cmd = PlanCommand(env, plan)
    cmd.do()
    after = state(tl)
    assert [(x.id, x.timeline_start) for x in tl.get_track("track_v1").clips] == [("b", 7.0), ("c", 9.0)]
    cmd.undo()
    assert state(tl) == before
    cmd.do()
    assert state(tl) == after
    cmd.undo()
    assert state(tl) == before
    old, new = cmd.states()
    assert {x.id for x in old} == {"a", "b", "c"} and {x.id for x in new} == {"b", "c"}


def test_plan_validation_rejects_and_leaves_the_timeline_untouched():
    a, b = vclip("a", 0, 2), vclip("b", 2, 2)
    tl = timeline_of(a, b)
    env = make_env(tl)
    snap = state(tl)
    cases = [
        (EditPlan("overlap", {"b": moved(b, timeline_start=1.0)}), "overlap"),
        (EditPlan("neg", {"a": moved(a, timeline_start=-1.0)}), "before 0:00"),
        (EditPlan("short", {"a": moved(a, duration=0.01, source_out=0.01)}), "minimum clip length"),
        (EditPlan("src", {"a": moved(a, duration=20.0, source_out=20.0)}), "past the end of its source"),
        (EditPlan("srcmap", {"a": moved(a, duration=1.0)}), "does not match"),
        (EditPlan("speed", {"a": moved(a, speed=20.0)}), "Speed"),
        (EditPlan("gone", {"nope": a.snapshot()}), "no longer exists"),
        (EditPlan("both", {"a": a.snapshot()}, removes=["a"]), "both changes and deletes"),
        (EditPlan("kf", {"a": moved(a, keyframes=[__import__("app.timeline.keyframes", fromlist=["Keyframe"]).Keyframe("scale", 5.0, 1.0)])}), "outside the clip"),
        (EditPlan("track", {"a": moved(a, track_id="track_a1")}), "cannot hold"),
        (EditPlan("add-dup", adds=[a.snapshot()]), "reuse an existing clip id"),
        (EditPlan("nomedia", adds=[Clip(new_clip_id(), "track_v2", "", 0, 1)]), "media for a new clip"),
    ]
    for plan, text in cases:
        with pytest.raises(TimelineError) as e:
            PlanCommand(env, plan).do()
        assert text in e.value.user_message if hasattr(e.value, "user_message") else text in str(e.value), (plan.description, str(e.value))
        assert state(tl) == snap, plan.description


def test_lock_rules_track_clip_and_aspect():
    a = vclip("a", 0, 2, locks=frozenset({"position"}))
    b = vclip("b", 2, 2, locks=frozenset({"item"}))
    c = vclip("c", 4, 2, locks=frozenset({"transform"}))
    tl = timeline_of(a, b, c)
    env = make_env(tl)
    snap = state(tl)
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, EditPlan("m", {"a": moved(a, timeline_start=10.0)})).do()
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, EditPlan("t", {"b": moved(b, opacity=0.5)})).do()  # an item lock blocks every aspect
    with pytest.raises(TimelineError, match="locked"):
        PlanCommand(env, EditPlan("d", removes=["b"])).do()
    PlanCommand(env, EditPlan("ok", {"c": moved(c, timeline_start=10.0)})).do()  # a transform lock does not block moving
    PlanCommand(env, EditPlan("timing", {"a": moved(a, duration=1.0, source_out=1.0)})).do()  # nor does a position lock block a duration change... 
    assert state(tl) != snap
    tl2 = timeline_of(vclip("x", 0, 2), vclip("y", 3, 2))
    tl2.get_track("track_v1").locked = True
    with pytest.raises(TimelineError, match="Track .* is locked"):
        PlanCommand(make_env(tl2), EditPlan("m", {"x": moved(tl2.get_clip("x"), timeline_start=1.0)})).do()
    with pytest.raises(TimelineError, match="Track .* is locked"):
        PlanCommand(make_env(tl2), EditPlan("a", adds=[vclip(new_clip_id(), 6, 1)])).do()


def test_lock_changes_are_allowed_on_a_locked_clip_and_ai_protection_follows():
    a = vclip("a", 0, 2, locks=frozenset({"item"}))
    tl = timeline_of(a)
    env = make_env(tl)
    assert changed_aspects(a, moved(a, locks=frozenset())) == set()  # editing the lock itself is not a locked aspect
    PlanCommand(env, EditPlan("unlock", {"a": moved(a, locks=frozenset())})).do()
    assert not tl.get_clip("a").protected
    legacy = vclip("b", 3, 1, locked=True)
    assert legacy.protected and not vclip("c", 5, 1).protected


def test_changed_aspects_classification():
    a = vclip("a", 0, 2)
    assert changed_aspects(a, moved(a, timeline_start=1.0)) == {"position"}
    assert changed_aspects(a, moved(a, scale=2.0)) == {"transform"}
    assert changed_aspects(a, moved(a, audio={"volume": 0.5})) == {"audio"}
    assert changed_aspects(a, moved(a, text={"text": "x"})) == {"style"}
    assert changed_aspects(a, moved(a, duration=1.0, source_out=1.0)) == {"timing"}


def test_track_compatibility():
    tl = timeline_of()
    v, aud = vclip("v", 0, 1), Clip("au", "track_a1", "aud1", 0, 1, 0, 1)
    cap = Clip("cp", "track_v6", "", 0, 1, kind="caption")
    assert clip_fits_track(v, tl.get_track("track_v1"), "video") and not clip_fits_track(v, tl.get_track("track_a1"), "video")
    assert clip_fits_track(aud, tl.get_track("track_a2"), "audio") and clip_fits_track(cap, tl.get_track("track_v6"), None)
    assert not clip_fits_track(cap, tl.get_track("track_v1"), None)


def test_big_ripple_like_plan_rebuilds_each_track_once():
    clips = [vclip(f"c{i}", i * 2.0, 2.0) for i in range(2000)]
    tl = timeline_of(*clips)
    env = make_env(tl)
    ups = {c.id: moved(c, timeline_start=c.timeline_start + 1000.0) for c in clips}
    writes = []
    orig = type(tl.get_track("track_v1")).__setattr__

    def spy(self, name, value):
        if name == "clips":
            writes.append(self.id)
        orig(self, name, value)

    type(tl.get_track("track_v1")).__setattr__ = spy
    try:
        cmd = PlanCommand(env, EditPlan("shift all", ups))
        cmd.do()
    finally:
        type(tl.get_track("track_v1")).__setattr__ = orig
    assert writes == ["track_v1"] and tl.get_clip("c0").timeline_start == 1000.0 and len(tl.get_track("track_v1").clips) == 2000
    cmd.undo()
    assert tl.get_clip("c1999").timeline_start == 3998.0


def test_random_plans_keep_invariants_and_undo_to_the_start():
    rng = random.Random(11)
    tl = timeline_of(*[vclip(f"c{i}", i * 3.0, 2.0) for i in range(30)])
    env = make_env(tl)
    start = state(tl)
    done: list[PlanCommand] = []
    for _ in range(200):
        c = rng.choice(tl.all_clips())
        kind = rng.random()
        if kind < 0.5:
            plan = EditPlan("mv", {c.id: moved(c, timeline_start=max(0.0, c.timeline_start + rng.choice([-1.0, 0.5, 1.0, 2.5])))})
        elif kind < 0.8:
            plan = EditPlan("rm", removes=[c.id])
        else:
            plan = EditPlan("add", adds=[vclip(new_clip_id(), rng.uniform(0, 100), rng.uniform(0.1, 2.0))])
        cmd = PlanCommand(env, plan)
        try:
            cmd.do()
            done.append(cmd)
        except TimelineError:
            pass
        for t in tl.tracks:  # invariants after every step: sorted and non-overlapping
            for x, y in zip(t.clips, t.clips[1:]):
                assert (x.timeline_start, x.id) <= (y.timeline_start, y.id) and y.timeline_start >= x.timeline_end - 1e-6
    mid = state(tl)
    for cmd in reversed(done):
        cmd.undo()
    assert state(tl) == start
    for cmd in done:
        cmd.do()
    assert state(tl) == mid


def test_plan_env_for_project_resolves_assets_and_timebase():
    from app.media.asset import Asset, SourceType

    p = Project.new("E")
    aid = p.assets.new_id()
    p.assets.add(Asset(aid, AssetType.VIDEO, SourceType.USER_MEDIA, "media/video/x.mp4", "x.mp4", 12.0, 1920, 1080, 30.0, None, False, None, None, None, 1, "h"))
    env = PlanEnv.for_project(p)
    assert env.asset_info(aid) == ("video", 12.0, "x.mp4") and env.asset_info("missing") is None and env.timebase == timebase_of(p)
    assert env.label(Clip("c", "track_v1", aid, 0, 1)) == "x.mp4" and env.label(Clip("c2", "track_v6", "", 0, 1, kind="caption", text={"text": "Hello there"})) == "“Hello there”"


def test_moving_a_caption_is_not_a_style_change():
    text = {"text": "Hello world", "start": 1.0, "end": 2.0, "lines": ["Hello world"], "reading_cps": 11.0, "words": [{"word": "Hello", "start": 1.0, "end": 1.4}, {"word": "world", "start": 1.5, "end": 2.0}]}
    cap = Clip("cp", "track_v6", "", 1.0, 1.0, kind="caption", text=text, locks=frozenset({"style"}))
    moved_cap = cap.snapshot()
    moved_cap.timeline_start = 3.0
    moved_cap.text = {**text, "start": 3.0, "end": 4.0, "words": [{"word": "Hello", "start": 3.0, "end": 3.4}, {"word": "world", "start": 3.5, "end": 4.0}]}
    assert changed_aspects(cap, moved_cap) == {"position"}  # the style lock does not block a move
    edited = moved_cap.snapshot()
    edited.text = {**moved_cap.text, "text": "Goodbye world"}
    assert "style" in changed_aspects(moved_cap, edited)
