"""Phase 10 / W1-C: markers, range and track commands (do / undo / redo, validation, persistence)."""

from __future__ import annotations

import json

import pytest

from app.core.commands import CommandStack
from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.keyframes import Keyframe
from app.timeline.markers import MARKER_KINDS, Marker
from app.timeline.meta_commands import (
    MAX_MARKERS,
    AddMarkerCommand,
    ClearRangeCommand,
    DeleteMarkerCommand,
    DuplicateTrackCommand,
    EditMarkerCommand,
    MoveMarkerCommand,
    MoveTrackCommand,
    RemoveTrackSafeCommand,
    SetRangeCommand,
    SetTrackDisplayCommand,
    SetTrackKindCommand,
    TrackHasContent,
    make_duplicate_ids,
)
from app.timeline.timeline import Timeline
from app.timeline.track import Track, TrackKind

ASSETS = {"a_video": ("video", 30.0, "Interview"), "a_img": ("image", None, "Photo"), "a_aud": ("audio", 60.0, "Voice")}


def state(tl: Timeline) -> str:
    return json.dumps(tl.to_dict(), sort_keys=True)


def clip(cid: str, track: str, start: float, dur: float, asset: str = "a_video", **kw) -> Clip:
    return Clip(cid, track, asset, start, dur, source_in=kw.pop("source_in", 0.0), source_out=kw.pop("source_out", dur), **kw)


def cycle(cmd, tl: Timeline) -> None:
    """do -> undo -> redo with exact state checks."""
    before = state(tl)
    cmd.do()
    after = state(tl)
    cmd.undo()
    assert state(tl) == before
    cmd.do()
    assert state(tl) == after


# ------------------------------------------------------------------------------------------------------------------------ markers
def test_marker_add_move_edit_delete_cycle():
    tl = Timeline.default()
    add = AddMarkerCommand(tl, 5.0, "Intro", "section", "#112233", "note")
    cycle(add, tl)
    m = tl.get_marker(add.marker_id)
    assert (m.time, m.name, m.kind, m.color, m.note) == (5.0, "Intro", "section", "#112233", "note")
    mid = add.marker_id
    cycle(MoveMarkerCommand(tl, mid, 2.5), tl)
    cycle(EditMarkerCommand(tl, mid, name="Opening", kind="scene", color="", note="n2"), tl)
    dele = DeleteMarkerCommand(tl, mid)
    cycle(dele, tl)
    assert tl.markers == []  # the cycle ends with the delete redone
    dele.undo()
    restored = tl.get_marker(mid)
    assert (restored.time, restored.name, restored.kind) == (2.5, "Opening", "scene")
    add.undo()
    assert tl.get_marker(mid) is None


def test_markers_stay_sorted_and_may_share_a_time():
    tl = Timeline.default()
    cmds = [AddMarkerCommand(tl, t) for t in (9.0, 1.0, 5.0, 5.0)]
    for c in cmds:
        c.do()
    assert [m.time for m in tl.markers] == [1.0, 5.0, 5.0, 9.0]
    assert len({m.id for m in tl.markers}) == 4
    mv = MoveMarkerCommand(tl, cmds[0].marker_id, 0.0)  # 9.0 -> 0.0 jumps to the front
    mv.do()
    assert [m.time for m in tl.markers] == [0.0, 1.0, 5.0, 5.0]
    mv.undo()
    assert [m.time for m in tl.markers] == [1.0, 5.0, 5.0, 9.0]
    assert tl.next_marker(1.0).time == 5.0 and tl.prev_marker(5.0).time == 1.0


def test_marker_ids_survive_redo():
    tl = Timeline.default()
    stack = CommandStack()
    cmd = stack.execute(AddMarkerCommand(tl, 3.0, "x"))
    mid = cmd.marker_id
    stack.undo()
    assert tl.markers == []
    stack.redo()
    assert [m.id for m in tl.markers] == [mid]


@pytest.mark.parametrize("kwargs, why", [
    (dict(time=-1.0), "before 0:00"),
    (dict(time=float("nan")), "valid time"),
    (dict(time="soon"), "valid time"),
    (dict(time=1.0, kind="bogus"), "Unknown marker type"),
    (dict(time=1.0, color="red"), "#RRGGBB"),
    (dict(time=1.0, name="x" * 201), "name"),
])
def test_marker_validation(kwargs, why):
    tl = Timeline.default()
    before = state(tl)
    stack = CommandStack()
    with pytest.raises(TimelineError, match=why):
        stack.execute(AddMarkerCommand(tl, **kwargs))
    assert state(tl) == before and not stack.can_undo  # a refused command leaves no undo entry


def test_marker_limit_and_unknown_marker():
    tl = Timeline.default()
    tl.markers = [Marker(f"m{i}", float(i)) for i in range(MAX_MARKERS)]
    with pytest.raises(TimelineError, match="at most"):
        AddMarkerCommand(tl, 1.0).do()
    for cmd in (MoveMarkerCommand(tl, "nope", 1.0), EditMarkerCommand(tl, "nope", name="x"), DeleteMarkerCommand(tl, "nope")):
        with pytest.raises(TimelineError, match="no longer exists"):
            cmd.do()
    with pytest.raises(TimelineError):
        EditMarkerCommand(tl, "m1", kind="bogus").do()
    assert set(MARKER_KINDS) >= {"generic", "scene", "section"}


def test_markers_and_range_persist():
    tl = Timeline.default()
    AddMarkerCommand(tl, 2.0, "A", "evidence", "#ff00ff", "check").do()
    AddMarkerCommand(tl, 1.0).do()
    SetRangeCommand(tl, 1.5, 4.0).do()
    SetTrackDisplayCommand(tl, "track_v1", height=90, collapsed=True).do()
    back = Timeline.from_dict(json.loads(json.dumps(tl.to_dict())))
    assert state(back) == state(tl)
    assert [(m.time, m.name, m.kind, m.color, m.note) for m in back.markers] == [(1.0, "", "generic", "", ""), (2.0, "A", "evidence", "#ff00ff", "check")]
    assert back.range == (1.5, 4.0)
    t = back.get_track("track_v1")
    assert (t.height, t.collapsed) == (90, True)


# ------------------------------------------------------------------------------------------------------------------------ range
def test_range_set_clear_undo():
    tl = Timeline.default()
    cycle(SetRangeCommand(tl, 1.0, 3.0), tl)
    assert tl.range == (1.0, 3.0)
    again = SetRangeCommand(tl, 2.0, 8.0)
    again.do()
    assert tl.range == (2.0, 8.0)
    again.undo()
    assert tl.range == (1.0, 3.0)  # undo restores the PREVIOUS range, not None
    cycle(ClearRangeCommand(tl), tl)
    assert tl.range is None
    ClearRangeCommand(tl).do()  # clearing nothing is harmless
    assert tl.range is None


@pytest.mark.parametrize("a, b", [(-1.0, 2.0), (3.0, 3.0), (4.0, 2.0), (float("nan"), 2.0), (0.0, float("inf")), ("a", 2.0)])
def test_range_validation(a, b):
    tl = Timeline.default()
    with pytest.raises(TimelineError):
        SetRangeCommand(tl, a, b).do()
    assert tl.range is None


# ------------------------------------------------------------------------------------------------------------------------ track order / display
def test_move_track_reorder_and_undo():
    tl = Timeline.default()
    tl.insert_clip(clip("c1", "track_v1", 0, 2))
    ids = [t.id for t in tl.tracks]
    cmd = MoveTrackCommand(tl, "track_a1", 0)
    cycle(cmd, tl)
    assert [t.id for t in tl.tracks][0] == "track_a1" and sorted(t.id for t in tl.tracks) == sorted(ids)
    assert tl.find_clip("c1")[0].id == "track_v1"
    cmd.undo()
    assert [t.id for t in tl.tracks] == ids
    last = MoveTrackCommand(tl, "track_v1", len(ids) - 1)
    cycle(last, tl)
    assert tl.tracks[-1].id == "track_v1"  # no "audio stays below" restriction


def test_move_track_validation_and_noop():
    tl = Timeline.default()
    ids = [t.id for t in tl.tracks]
    for bad in (-1, len(ids), 1.5, True):
        with pytest.raises(TimelineError):
            MoveTrackCommand(tl, "track_v1", bad).do()
    with pytest.raises(TimelineError, match="no longer exists"):
        MoveTrackCommand(tl, "ghost", 0).do()
    same = MoveTrackCommand(tl, "track_v1", 0)
    same.do()
    same.undo()
    assert [t.id for t in tl.tracks] == ids


def test_track_display_validation():
    tl = Timeline.default()
    cmd = SetTrackDisplayCommand(tl, "track_v2", height=120, collapsed=True)
    cycle(cmd, tl)
    tr = tl.get_track("track_v2")
    assert (tr.height, tr.collapsed) == (120, True)
    cmd.undo()
    assert (tr.height, tr.collapsed) == (0, False)
    for h in (0, 18, 240):
        SetTrackDisplayCommand(tl, "track_v2", height=h).do()
        assert tl.get_track("track_v2").height == h
    for h in (1, 17, 241, -5, 30.5, float("nan"), "big", True):
        with pytest.raises(TimelineError):
            SetTrackDisplayCommand(tl, "track_v2", height=h).do()
    SetTrackDisplayCommand(tl, "track_v2", collapsed=False).do()
    assert tl.get_track("track_v2").height == 240  # untouched by a collapsed-only change
    assert SetTrackDisplayCommand.major is False


# ------------------------------------------------------------------------------------------------------------------------ duplicate track
def _dup_timeline() -> Timeline:
    tl = Timeline.default()
    tl.insert_clip(clip("p1", "track_v1", 0, 2, link_id="L1", group_id="G1", locks=frozenset({"timing"}), locked=True, created_by="AI", ai_decision_id="dec1",
                        keyframes=[Keyframe("scale", 0.0, 1.0, "linear", "dec1"), Keyframe("scale", 1.0, 1.2, "ease_in", "dec1")], transition={"type": "FADE", "duration": 0.5, "decision_id": "dec1"}))
    tl.insert_clip(clip("p2", "track_v1", 2, 2, link_id="L2", group_id="G1", scene_id="s1", slot="visual:1"))
    tl.insert_clip(clip("p3", "track_v1", 4, 2))
    tl.insert_clip(clip("a1", "track_a1", 0, 2, asset="a_aud", link_id="L1"))  # linked to p1 on ANOTHER track
    tl.insert_clip(clip("a2", "track_a1", 2, 2, asset="a_aud", link_id="L2"))
    return tl


def test_duplicate_track_copies_clips_with_fresh_ids_and_remapped_relations():
    tl = _dup_timeline()
    tid, idmap = make_duplicate_ids(tl, "track_v1")
    cmd = DuplicateTrackCommand(tl, "track_v1", tid, idmap)
    src_before = json.dumps(tl.get_track("track_v1").to_dict(), sort_keys=True)
    cycle(cmd, tl)
    names = [t.id for t in tl.tracks]
    assert names.index(tid) == names.index("track_v1") + 1  # right below the original
    new = tl.get_track(tid)
    assert new.name == "V1 Main Video copy" and new.kind is TrackKind.VIDEO and not new.locked
    assert [c.id for c in new.clips] == [idmap["p1"], idmap["p2"], idmap["p3"]]
    p1, p2, p3 = new.clips
    assert all(c.track_id == tid and c.created_by == "USER" and not c.locks and not c.locked and c.ai_decision_id == "" for c in new.clips)
    assert p1.timeline_start == 0 and p1.duration == 2 and p3.timeline_start == 4
    assert [k.decision_id for k in p1.keyframes] == ["", ""] and "decision_id" not in p1.transition
    assert p1.group_id and p1.group_id == p2.group_id and p1.group_id != "G1"  # remapped, shared inside the copy
    assert p1.link_id == "" and p2.link_id == ""  # the other link member lives on another track and was not copied: a link of one means nothing
    assert p2.scene_id == "s1" and p2.slot == "visual:1"
    assert json.dumps(tl.get_track("track_v1").to_dict(), sort_keys=True) == src_before  # the original is untouched
    assert tl.get_clip("p1").link_id == "L1" and tl.get_clip("a1").link_id == "L1"
    cmd.undo()
    assert tid not in [t.id for t in tl.tracks] and idmap["p1"] not in {c.id for c in tl.all_clips()}


def test_duplicate_track_keeps_links_between_copied_clips():
    tl = Timeline.default()
    tl.insert_clip(clip("x1", "track_v2", 0, 2, link_id="L"))
    tl.insert_clip(clip("x2", "track_v2", 2, 2, link_id="L"))
    tid, idmap = make_duplicate_ids(tl, "track_v2")
    DuplicateTrackCommand(tl, "track_v2", tid, idmap).do()
    a, b = tl.get_track(tid).clips
    assert a.link_id and a.link_id == b.link_id and a.link_id != "L"


def test_duplicate_track_without_clips_and_validation():
    tl = _dup_timeline()
    tl.get_track("track_v1").locked = True  # duplicating reads the track: allowed; the copy is unlocked
    cmd = DuplicateTrackCommand(tl, "track_v1", "dup1", include_clips=False)
    cycle(cmd, tl)
    assert tl.get_track("dup1").clips == [] and tl.get_track("dup1").locked is False
    with pytest.raises(TimelineError, match="already exists"):
        DuplicateTrackCommand(tl, "track_v1", "dup1", include_clips=False).do()
    with pytest.raises(TimelineError, match="no new id"):
        DuplicateTrackCommand(tl, "track_v1", "dup2", {"p1": "n1"}).do()
    with pytest.raises(TimelineError, match="reuse"):
        DuplicateTrackCommand(tl, "track_v1", "dup3", {"p1": "n1", "p2": "n1", "p3": "n3"}).do()
    with pytest.raises(TimelineError, match="reuse"):
        DuplicateTrackCommand(tl, "track_v1", "dup4", {"p1": "p1", "p2": "n2", "p3": "n3"}).do()
    assert "dup2" not in [t.id for t in tl.tracks]


# ------------------------------------------------------------------------------------------------------------------------ track kind
def test_set_track_kind_rules():
    tl = Timeline.default()
    tl.insert_clip(clip("g1", "track_v4", 0, 2, kind="graphic", asset=""))
    tl.insert_clip(clip("v1", "track_v1", 0, 2))
    tl.insert_clip(clip("t1", "track_v5", 0, 2, kind="text", asset=""))
    info = ASSETS.get
    cmd = SetTrackKindCommand(tl, "track_v4", TrackKind.TEXT)  # graphics -> text: both hold graphic / text clips
    cycle(cmd, tl)
    assert tl.get_track("track_v4").kind is TrackKind.TEXT
    cmd.undo()
    assert tl.get_track("track_v4").kind is TrackKind.GRAPHICS
    with pytest.raises(TimelineError, match="cannot be on a captions track") as e:
        SetTrackKindCommand(tl, "track_v4", "captions").do()
    assert "g1" in str(e.value)  # names the first offending clip
    with pytest.raises(TimelineError, match="cannot be on a audio track"):
        SetTrackKindCommand(tl, "track_v1", TrackKind.AUDIO, info).do()
    assert tl.get_track("track_v1").kind is TrackKind.VIDEO
    ok = SetTrackKindCommand(tl, "track_v1", TrackKind.IMAGE, info)  # picture media may sit on a video or an image track
    cycle(ok, tl)
    assert tl.get_track("track_v1").kind is TrackKind.IMAGE
    with pytest.raises(TimelineError, match="unknown"):  # no asset info: media cannot be verified
        SetTrackKindCommand(tl, "track_v1", TrackKind.VIDEO).do()
    SetTrackKindCommand(tl, "track_v2", TrackKind.AUDIO).do()  # an empty track can become anything
    assert tl.get_track("track_v2").kind is TrackKind.AUDIO
    with pytest.raises(TimelineError, match="Unknown track type"):
        SetTrackKindCommand(tl, "track_v2", "hologram").do()
    tl.get_track("track_v3").locked = True
    with pytest.raises(TimelineError, match="locked"):
        SetTrackKindCommand(tl, "track_v3", TrackKind.VIDEO).do()


# ------------------------------------------------------------------------------------------------------------------------ safe delete
def test_remove_track_safe_refuses_content_and_recovers():
    tl = _dup_timeline()
    tl.get_clip("p1").locks = frozenset()  # (a locked clip blocks the delete: covered below)
    idx = [t.id for t in tl.tracks].index("track_v1")
    before = state(tl)
    stack = CommandStack()
    with pytest.raises(TrackHasContent) as e:
        stack.execute(RemoveTrackSafeCommand(tl, "track_v1"))
    assert e.value.clip_count == 3 and e.value.track_name == "V1 Main Video" and "3 clips" in str(e.value)
    assert state(tl) == before and not stack.can_undo
    stack.execute(RemoveTrackSafeCommand(tl, "track_v1", confirm_content=True))
    assert "track_v1" not in [t.id for t in tl.tracks] and tl.get_clip("p1") is None
    stack.undo()
    assert state(tl) == before and [t.id for t in tl.tracks].index("track_v1") == idx
    assert [c.id for c in tl.get_track("track_v1").clips] == ["p1", "p2", "p3"]
    stack.redo()
    assert "track_v1" not in [t.id for t in tl.tracks]
    stack.undo()
    assert state(tl) == before


def test_remove_track_safe_empty_locked_and_locked_clip():
    tl = _dup_timeline()
    RemoveTrackSafeCommand(tl, "track_v2").do()  # empty: no confirmation needed
    assert "track_v2" not in [t.id for t in tl.tracks]
    tl.get_track("track_a1").locked = True
    with pytest.raises(TimelineError, match="locked"):
        RemoveTrackSafeCommand(tl, "track_a1", confirm_content=True).do()
    tl.get_track("track_a1").locked = False
    tl.get_clip("a1").locks = frozenset({"item"})
    with pytest.raises(TimelineError, match="unlock"):
        RemoveTrackSafeCommand(tl, "track_a1", confirm_content=True).do()
    assert "track_a1" in [t.id for t in tl.tracks]
    with pytest.raises(TimelineError, match="no longer exists"):
        RemoveTrackSafeCommand(tl, "ghost").do()


def test_track_commands_through_the_stack_are_timeline_scope():
    tl = Timeline.default()
    for cmd in (AddMarkerCommand(tl, 1.0), SetRangeCommand(tl, 0.0, 1.0), MoveTrackCommand(tl, "track_v1", 2), RemoveTrackSafeCommand(tl, "track_v2")):
        assert cmd.scope == "timeline"
    t = Track("t_extra", "Extra", TrackKind.VIDEO)
    assert t.height == 0 and t.collapsed is False
