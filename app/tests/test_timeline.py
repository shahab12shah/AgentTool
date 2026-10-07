from __future__ import annotations

import pytest

from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.timeline import Timeline
from app.timeline.track import TrackKind
from app.tests.conftest import import_and_wait, needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture
def tl_ws(project_ws, media_dir):
    ws = project_ws
    ws.v = import_and_wait(ws, media_dir / "clip.mp4")   # 3 s
    ws.v2 = import_and_wait(ws, media_dir / "other.mp4")  # 2 s
    ws.img = import_and_wait(ws, media_dir / "pic.png")
    ws.aud = import_and_wait(ws, media_dir / "voice.wav")  # 4 s
    return ws


def test_default_tracks():
    tl = Timeline.default()
    assert [t.name for t in tl.tracks] == [
        "V1 Main Video", "V2 B-Roll", "V3 Images", "V4 Graphics", "V5 Text", "A1 Voice-over", "A2 Music", "A3 SFX",
    ]


def test_add_rename_flag_delete_track_with_undo(tl_ws):
    ws = tl_ws
    n = len(ws.timeline.timeline.tracks)
    t = ws.timeline.add_track(TrackKind.VIDEO)
    assert t.name == "V6 Video" and len(ws.timeline.timeline.tracks) == n + 1
    assert ws.timeline.timeline.tracks.index(t) == 5  # video tracks stay above audio
    ws.timeline.rename_track(t.id, "  Overlay ")
    assert ws.timeline.timeline.get_track(t.id).name == "Overlay"
    with pytest.raises(TimelineError):
        ws.timeline.rename_track(t.id, "  ")
    ws.timeline.set_track_flag(t.id, "hidden", True)
    ws.timeline.set_track_flag(t.id, "muted", True)
    ws.timeline.set_track_flag(t.id, "locked", True)
    tr = ws.timeline.timeline.get_track(t.id)
    assert (tr.hidden, tr.muted, tr.locked) == (True, True, True)
    with pytest.raises(TimelineError):  # locked tracks cannot be deleted
        ws.timeline.remove_track(t.id)
    ws.timeline.set_track_flag(t.id, "locked", False)
    ws.timeline.remove_track(t.id)
    assert len(ws.timeline.timeline.tracks) == n
    ws.undo()
    assert len(ws.timeline.timeline.tracks) == n + 1
    ws.undo()  # un-lock
    ws.undo()
    ws.undo()
    ws.undo()  # hidden
    ws.undo()  # rename
    assert ws.timeline.timeline.get_track(t.id).name == "V6 Video"
    ws.undo()  # add track
    assert len(ws.timeline.timeline.tracks) == n


def test_add_clip_defaults_and_placement(tl_ws):
    ws = tl_ws
    c1 = ws.timeline.add_asset(ws.v.id)
    assert c1.track_id == "track_v1" and c1.timeline_start == 0 and c1.duration == pytest.approx(ws.v.duration)
    assert (c1.source_in, c1.source_out) == (0, c1.duration)
    assert (c1.scale, c1.rotation, c1.opacity, c1.speed, c1.position) == (1, 0, 1, 1, (0, 0))
    c2 = ws.timeline.add_asset(ws.v2.id)  # appended after c1
    assert c2.timeline_start == pytest.approx(c1.timeline_end)
    img = ws.timeline.add_asset(ws.img.id)
    assert img.track_id == "track_v3" and img.duration == 5.0
    vo = ws.timeline.add_asset(ws.aud.id)
    assert vo.track_id == "track_a2"  # not the voice-over -> music track
    with pytest.raises(TimelineError):
        ws.timeline.add_asset(ws.aud.id, track_id="track_v1")  # audio on video track
    with pytest.raises(TimelineError):
        ws.timeline.add_asset(ws.v.id, track_id="track_a1")


def test_add_clip_at_occupied_position_finds_free_slot(tl_ws):
    ws = tl_ws
    c1 = ws.timeline.add_asset(ws.v.id, track_id="track_v1", start=0)
    c2 = ws.timeline.add_asset(ws.v2.id, track_id="track_v1", start=1.0)
    assert c2.timeline_start == pytest.approx(c1.timeline_end)


def test_move_clip_rules_and_undo_redo(tl_ws):
    ws = tl_ws
    a = ws.timeline.add_asset(ws.v.id)
    b = ws.timeline.add_asset(ws.v2.id)
    ws.timeline.move_clip(b.id, 10.0)
    clip = ws.timeline.timeline.get_clip(b.id)
    assert clip.timeline_start == 10.0
    with pytest.raises(TimelineError):  # overlap
        ws.timeline.move_clip(b.id, 1.0)
    assert ws.timeline.timeline.get_clip(b.id).timeline_start == 10.0  # unchanged after failure
    with pytest.raises(TimelineError):  # negative start clamps to 0, which overlaps clip a
        ws.timeline.move_clip(b.id, -5)
    ws.project.validate()
    ws.undo()
    assert ws.timeline.timeline.get_clip(b.id).timeline_start == pytest.approx(a.timeline_end)
    ws.redo()
    assert ws.timeline.timeline.get_clip(b.id).timeline_start == 10.0


def test_move_across_tracks(tl_ws):
    ws = tl_ws
    a = ws.timeline.add_asset(ws.v.id)
    ws.timeline.move_clip(a.id, 2.0, "track_v2")
    t, c = ws.timeline.timeline.find_clip(a.id)
    assert t.id == "track_v2" and c.track_id == "track_v2" and c.timeline_start == 2.0
    ws.undo()
    t, c = ws.timeline.timeline.find_clip(a.id)
    assert t.id == "track_v1" and c.timeline_start == 0.0
    ws.redo()
    assert ws.timeline.timeline.find_clip(a.id)[0].id == "track_v2"
    with pytest.raises(TimelineError):
        ws.timeline.move_clip(a.id, 0, "track_a1")  # wrong media kind


def test_trim_start_and_end(tl_ws):
    ws = tl_ws
    c = ws.timeline.add_asset(ws.v.id)  # 0..3
    dur = c.duration
    ws.timeline.trim_clip(c.id, new_end=2.0)
    cur = ws.timeline.timeline.get_clip(c.id)
    assert cur.duration == pytest.approx(2.0) and cur.source_out == pytest.approx(2.0) and cur.source_in == 0
    ws.timeline.trim_clip(c.id, new_start=0.5)
    cur = ws.timeline.timeline.get_clip(c.id)
    assert cur.timeline_start == pytest.approx(0.5) and cur.source_in == pytest.approx(0.5)
    assert cur.duration == pytest.approx(1.5) and cur.source_out == pytest.approx(2.0)
    # Cannot extend beyond the source media or before its start.
    ws.timeline.trim_clip(c.id, new_end=100.0)
    assert ws.timeline.timeline.get_clip(c.id).source_out == pytest.approx(dur)
    ws.timeline.trim_clip(c.id, new_start=-3.0)
    cur = ws.timeline.timeline.get_clip(c.id)
    assert cur.timeline_start == pytest.approx(0.0) and cur.source_in == pytest.approx(0.0)
    # Minimum duration respected.
    ws.timeline.trim_clip(c.id, new_end=0.0)
    assert ws.timeline.timeline.get_clip(c.id).duration > 0
    ws.project.validate()


def test_trim_cannot_overlap_neighbour(tl_ws):
    ws = tl_ws
    a = ws.timeline.add_asset(ws.v.id)           # 0..3
    b = ws.timeline.add_asset(ws.v2.id)          # 3..5
    ws.timeline.trim_clip(a.id, new_end=1.0)     # a: 0..1
    ws.timeline.trim_clip(b.id, new_start=3.8)   # b keeps source from 0.8
    ws.timeline.move_clip(b.id, 1.5)             # source start would map to 0.7, i.e. inside a
    ws.timeline.trim_clip(b.id, new_start=0.0)   # must clamp to a's end, not overlap it
    assert ws.timeline.timeline.get_clip(b.id).timeline_start == pytest.approx(1.0)
    ws.project.validate()


def test_image_clips_can_extend_freely(tl_ws):
    ws = tl_ws
    img = ws.timeline.add_asset(ws.img.id)
    ws.timeline.trim_clip(img.id, new_end=20.0)
    assert ws.timeline.timeline.get_clip(img.id).duration == pytest.approx(20.0)


def test_delete_clip_undo_redo(tl_ws):
    ws = tl_ws
    c = ws.timeline.add_asset(ws.v.id)
    ws.timeline.delete_clip(c.id)
    assert ws.timeline.timeline.get_clip(c.id) is None
    ws.undo()
    assert ws.timeline.timeline.get_clip(c.id) is not None
    ws.redo()
    assert ws.timeline.timeline.get_clip(c.id) is None
    ws.undo()
    ws.undo()  # removes the add
    assert ws.timeline.timeline.all_clips() == []
    ws.redo()
    assert len(ws.timeline.timeline.all_clips()) == 1


def test_locked_track_blocks_edits_but_not_undo(tl_ws):
    ws = tl_ws
    c = ws.timeline.add_asset(ws.v.id)
    ws.timeline.set_track_flag("track_v1", "locked", True)
    for op in (
        lambda: ws.timeline.move_clip(c.id, 5),
        lambda: ws.timeline.trim_clip(c.id, new_end=1),
        lambda: ws.timeline.delete_clip(c.id),
        lambda: ws.timeline.add_asset(ws.v2.id, track_id="track_v1"),
        lambda: ws.timeline.set_clip_properties(c.id, scale=2),
    ):
        with pytest.raises(TimelineError):
            op()
    ws.undo()  # undoing the lock itself works
    ws.timeline.move_clip(c.id, 5)


def test_clip_properties_validation_and_speed(tl_ws):
    ws = tl_ws
    c = ws.timeline.add_asset(ws.v.id)  # 3 s
    ws.timeline.set_clip_properties(c.id, position=(10, -20), scale=1.5, rotation=90, opacity=0.5)
    cur = ws.timeline.timeline.get_clip(c.id)
    assert (cur.position, cur.scale, cur.rotation, cur.opacity) == ((10, -20), 1.5, 90, 0.5)
    for bad in ({"opacity": 2}, {"scale": 0}, {"speed": 100}):
        with pytest.raises(TimelineError):
            ws.timeline.set_clip_properties(c.id, **bad)
    ws.timeline.set_clip_properties(c.id, speed=2.0)
    cur = ws.timeline.timeline.get_clip(c.id)
    assert cur.duration == pytest.approx(ws.v.duration / 2) and cur.source_out - cur.source_in == pytest.approx(ws.v.duration)
    ws.undo()
    assert ws.timeline.timeline.get_clip(c.id).speed == 1.0


def test_undo_redo_stack_semantics(tl_ws):
    ws = tl_ws
    assert not ws.commands.can_redo
    c = ws.timeline.add_asset(ws.v.id)
    ws.timeline.move_clip(c.id, 4)
    ws.undo()
    assert ws.commands.can_redo
    ws.timeline.move_clip(c.id, 7)  # new action clears redo history
    assert not ws.commands.can_redo
    assert ws.commands.undo_text == "Move clip"


def test_script_edits_are_undoable_and_merge(tl_ws):
    ws = tl_ws
    ws.set_script("H")
    ws.set_script("He")
    ws.set_script("Hello")
    assert ws.project.script.text == "Hello"
    ws.undo()  # merged typing burst is a single undo step
    assert ws.project.script.text == ""
    ws.redo()
    assert ws.project.script.text == "Hello"


def test_timeline_serialisation_roundtrip(tl_ws):
    ws = tl_ws
    ws.timeline.add_asset(ws.v.id)
    ws.timeline.add_asset(ws.img.id)
    doc = ws.project.timeline.to_dict()
    again = Timeline.from_dict(doc)
    assert again.to_dict() == doc
    assert Clip.from_dict(again.all_clips()[0].to_dict()) == again.all_clips()[0]
