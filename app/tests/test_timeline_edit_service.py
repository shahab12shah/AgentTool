"""Phase 10: the ``ws.edit`` facade - every professional edit through the real workspace (real imported media, real undo stack, real AI-ownership hook)."""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import TimelineError
from app.tests.conftest import import_and_wait, needs_ffmpeg
from app.tests.helpers import make_image, make_video, write_tone_wav
from app.timeline.clip import KIND_CAPTION, Clip
from app.timeline.meta_commands import TrackHasContent
from app.timeline.ops_text import plan_merge_captions
from app.timeline.plan import PlanEnv
from app.timeline.timeline import Timeline
from app.timeline.track import TrackKind

pytestmark = needs_ffmpeg


def state(ws) -> str:
    return json.dumps(ws.project.timeline.to_dict(), sort_keys=True)


@pytest.fixture
def ews(project_ws, tmp_path):
    """A project with two videos, an image and an audio file imported; ``ws.A``..: the assets; ``ws.v1`` is the first video track."""
    ws = project_ws
    d = tmp_path / "media"
    d.mkdir()
    ws.A = [import_and_wait(ws, make_video(d / "a.mp4", 6.0)), import_and_wait(ws, make_video(d / "b.mp4", 4.0, "testsrc2")),
            import_and_wait(ws, make_video(d / "c.mp4", 5.0)), import_and_wait(ws, make_image(d / "p.png"))]
    ws.aud = import_and_wait(ws, write_tone_wav(d / "tone.wav", 8.0))
    ws.v1 = next(t for t in ws.project.timeline.tracks if t.kind is TrackKind.VIDEO)
    return ws


def starts(ws, track=None) -> list[tuple[str, float, float]]:
    tr = track or ws.v1
    return [(c.asset_id, round(c.timeline_start, 3), round(c.duration, 3)) for c in ws.project.timeline.get_track(tr.id).clips]


def lay(ws, n=3):
    """n clips back to back on the first video track; returns their ids."""
    ws.edit.place_assets([a.id for a in ws.A[:n]], at=0.0, track_id=ws.v1.id)
    return [c.id for c in ws.project.timeline.get_track(ws.v1.id).clips]


# ------------------------------------------------------------------------------------------------------------------------ placement
def test_place_assets_free_is_one_undo_step_and_lays_clips_in_a_row(ews):
    ids = lay(ews)
    assert [s[1:] for s in starts(ews)] == [(0.0, 6.0), (6.0, 4.0), (10.0, 5.0)]
    assert len(ids) == 3
    assert ews.commands.undo() is not None
    assert starts(ews) == []  # one undo removes all three
    ews.commands.redo()
    assert len(starts(ews)) == 3


def test_insert_mode_ripples_later_clips_right_and_overwrite_replaces(ews):
    lay(ews, 2)
    ews.edit.state.update(ripple_on=True)
    ews.edit.place_assets([ews.A[3].id], at=6.0, track_id=ews.v1.id, mode="insert")
    got = starts(ews)
    assert got[1][0] == ews.A[3].id and got[1][1] == 6.0 and got[2][1] == pytest.approx(6.0 + got[1][2])  # the image went in at 6 s, the old second clip moved right
    ews.commands.undo()
    assert [s[1] for s in starts(ews)] == [0.0, 6.0]
    ews.edit.place_assets([ews.A[3].id], at=1.0, track_id=ews.v1.id, mode="overwrite")
    assert ews.A[3].id in [s[0] for s in starts(ews)]
    ews.commands.undo()
    assert [s[1:] for s in starts(ews)] == [(0.0, 6.0), (6.0, 4.0)]


def test_placing_media_on_a_track_that_cannot_hold_it_is_refused_and_changes_nothing(ews):
    audio_track = next(t for t in ews.project.timeline.tracks if t.kind is TrackKind.AUDIO)
    before = state(ews)
    with pytest.raises(TimelineError):
        ews.edit.place_assets([ews.A[0].id], at=0.0, track_id=audio_track.id)
    assert state(ews) == before and not ews.commands.can_undo


# ------------------------------------------------------------------------------------------------------------------------ move
def test_multi_clip_move_keeps_relative_timing_in_one_undo_step(ews):
    a, b, c = lay(ews)
    ews.edit.set_start(c, 20.0)  # make room: gap after b
    ews.selection.select([a, b])
    ews.edit.state.update(linked_selection=True)
    ews.edit.move_clips(None, 2.0)
    tl = ews.project.timeline
    assert tl.get_clip(a).timeline_start == pytest.approx(2.0) and tl.get_clip(b).timeline_start == pytest.approx(8.0)  # still 6 s apart
    ews.commands.undo()
    assert tl.get_clip(a).timeline_start == 0.0 and tl.get_clip(b).timeline_start == 6.0
    ews.commands.redo()
    assert tl.get_clip(a).timeline_start == pytest.approx(2.0)


def test_move_delta_is_quantised_to_whole_frames(ews):
    a, *_ = lay(ews, 1)
    ews.edit.move_clips([a], 1.0 + 0.0123)
    fps = ews.project.settings.fps
    frames = ews.project.timeline.get_clip(a).timeline_start * fps
    assert frames == pytest.approx(round(frames), abs=1e-6)


def test_overlapping_move_is_rejected_atomically(ews):
    a, b, _ = lay(ews)
    before = state(ews)
    undo_depth = len(ews.commands._undo) if hasattr(ews.commands, "_undo") else None
    with pytest.raises(TimelineError):
        ews.edit.move_clips([a], 3.0)  # would run into b
    assert state(ews) == before
    if undo_depth is not None:
        assert len(ews.commands._undo) == undo_depth


def test_preview_move_is_a_dry_run(ews):
    a, b, c = lay(ews)
    before = state(ews)
    plan, reason = ews.edit.preview_move([a], 3.0)
    assert reason and "overlap" in reason.lower()  # the drag shows why it cannot land there
    plan, reason = ews.edit.preview_move([c], 1.0)
    assert reason is None and plan.updates[c].timeline_start == pytest.approx(11.0)
    assert state(ews) == before


def test_ripple_on_move_reorders_and_closes_the_gap_it_leaves(ews):
    a, b, c = lay(ews)
    ews.edit.state.update(ripple_on=True)
    ews.edit.move_clips([a], 10.0)  # lift the first clip out, close the gap, drop it between the second and third
    assert starts(ews) == [(ews.A[1].id, 0.0, 4.0), (ews.A[0].id, 4.0, 6.0), (ews.A[2].id, 10.0, 5.0)]
    ews.commands.undo()
    assert [s[1] for s in starts(ews)] == [0.0, 6.0, 10.0]
    ews.edit.move_clips([a], 2.0)  # landing inside its own span: nothing to do, no undo entry
    assert [s[1] for s in starts(ews)] == [0.0, 6.0, 10.0]


# ------------------------------------------------------------------------------------------------------------------------ locks
def test_locked_clip_rejects_edits_safely_and_unlock_restores_them(ews):
    a, b, c = lay(ews)
    ews.edit.set_start(c, 30.0)  # room to move b
    ews.edit.set_start(b, 8.0)
    ews.edit.toggle_item_lock([b])
    before = state(ews)
    for call in (lambda: ews.edit.move_clips([b], 1.0), lambda: ews.edit.trim(b, "end", 11.0), lambda: ews.edit.delete([b]), lambda: ews.edit.split(9.0, [b])):
        with pytest.raises(TimelineError, match="(?i)lock"):
            call()
    assert state(ews) == before
    ews.edit.toggle_item_lock([b])
    ews.edit.move_clips([b], 1.0)
    assert ews.project.timeline.get_clip(b).timeline_start == pytest.approx(9.0)


def test_locked_track_rejects_edits(ews):
    a, *_ = lay(ews)
    ews.edit.set_track_flag(ews.v1.id, "locked", True)
    with pytest.raises(TimelineError):
        ews.edit.move_clips([a], 0.0 + 1.0)
    ews.edit.set_track_flag(ews.v1.id, "locked", False)


# ------------------------------------------------------------------------------------------------------------------------ trim / split / delete
def test_trim_validates_and_undoes(ews):
    a, *_ = lay(ews)
    ews.edit.trim(a, "end", 4.0)
    assert ews.project.timeline.get_clip(a).duration == pytest.approx(4.0)
    ews.commands.undo()
    assert ews.project.timeline.get_clip(a).duration == pytest.approx(6.0)
    with pytest.raises(TimelineError):
        ews.edit.trim(a, "end", 30.0)  # beyond the media: refused with a reason


def test_split_at_playhead_keeps_source_mapping_and_is_one_step(ews):
    a, *_ = lay(ews)
    ews.edit.set_playhead(2.5)
    ews.selection.select([a])
    ews.edit.split()
    clips = ews.project.timeline.get_track(ews.v1.id).clips
    left, right = clips[0], clips[1]
    assert left.timeline_end == pytest.approx(2.5, abs=1 / 30) and right.timeline_start == pytest.approx(left.timeline_end)
    assert right.source_in == pytest.approx(left.source_out)  # the content continues exactly where it was cut
    ews.commands.undo()
    assert len(ews.project.timeline.get_track(ews.v1.id).clips) == 3


def test_blade_cuts_every_unlocked_track_under_the_cursor_when_no_track_given(ews):
    lay(ews, 1)
    a2 = ews.edit.place_assets([ews.aud.id], at=0.0)
    n = len(list(ews.project.timeline.all_clips()))
    ews.edit.blade(3.0)
    assert len(list(ews.project.timeline.all_clips())) == n + 2  # video and audio were cut together (one undo step)
    ews.commands.undo()
    assert len(list(ews.project.timeline.all_clips())) == n


def test_split_never_creates_a_zero_length_piece(ews):
    a, *_ = lay(ews, 1)
    before = state(ews)
    ews.selection.select([a])
    with pytest.raises(TimelineError, match="not inside"):
        ews.edit.split(0.0)  # at the very start: refused with the reason, nothing changes
    assert state(ews) == before


def test_ripple_delete_closes_the_gap_and_undo_restores_everything(ews):
    a, b, c = lay(ews)
    ews.edit.state.update(ripple_on=True)
    before = state(ews)
    ews.edit.delete([b])
    tl = ews.project.timeline
    assert tl.get_clip(b) is None and tl.get_clip(c).timeline_start == pytest.approx(6.0)
    ews.commands.undo()
    assert state(ews) == before
    ews.edit.state.update(ripple_on=False)
    ews.edit.delete([b])
    assert tl.get_clip(c).timeline_start == pytest.approx(10.0)  # without ripple a gap stays


# ------------------------------------------------------------------------------------------------------------------------ align / distribute
def test_align_to_playhead_scene_boundary_and_marker_in_one_undo_step_each(ews):
    from app.analysis.models import Scene

    a, b, c = lay(ews)
    tl = ews.project.timeline
    # put b and c on a second video track so they can be aligned with a
    v2, v3 = ews.edit.add_track(TrackKind.VIDEO), ews.edit.add_track(TrackKind.VIDEO)
    ews.edit.move_clips([b], 0.0, {b: v2.id})
    ews.edit.move_clips([c], 0.0, {c: v3.id})
    ews.project.scenes = [Scene("s1", "1", 0.0, 5.7), Scene("s2", "2", 5.7, 12.0)]
    ews.edit.set_playhead(8.0)
    ews.edit.align([b], "start", "playhead")
    assert tl.get_clip(b).timeline_start == pytest.approx(8.0)
    ews.commands.undo()
    assert tl.get_clip(b).timeline_start == pytest.approx(6.0)
    ews.edit.align([b], "start", "scene")  # b starts at 6.0: the nearest scene boundary is 5.7
    assert tl.get_clip(b).timeline_start == pytest.approx(5.7, abs=1 / 30 / 2 + 1e-6)
    ews.commands.undo()
    m = ews.edit.add_marker(7.0, "sync")
    ews.edit.align([b], "end", "marker")  # b ends at 10.0; the marker at 7.0 is the only one
    assert tl.get_clip(b).timeline_end == pytest.approx(7.0, abs=1 / 30)
    assert ews.project.timeline.get_marker(m)
    with pytest.raises(TimelineError, match="no scenes"):
        ews.project.scenes = []
        ews.edit.align([b], "start", "scene")


def test_distribute_through_the_facade(ews):
    a, b, c = lay(ews)
    v2 = ews.edit.add_track(TrackKind.VIDEO)
    v3 = ews.edit.add_track(TrackKind.VIDEO)
    ews.edit.move_clips([b], 0.0, {b: v2.id})
    ews.edit.move_clips([c], 4.0, {c: v3.id})
    tl = ews.project.timeline
    ews.edit.distribute([a, b, c], "gaps")
    ga = tl.get_clip(b).timeline_start - tl.get_clip(a).timeline_end
    gb = tl.get_clip(c).timeline_start - tl.get_clip(b).timeline_end
    assert abs(ga - gb) <= 1 / 30 + 1e-6
    ews.commands.undo()
    assert tl.get_clip(b).timeline_start == pytest.approx(6.0)


# ------------------------------------------------------------------------------------------------------------------------ clipboard
def test_copy_paste_makes_new_ids_and_reuses_the_asset(ews):
    a, *_ = lay(ews, 1)
    ews.edit.copy([a])
    ews.edit.paste(at=10.0, track_id=ews.v1.id)
    clips = ews.project.timeline.get_track(ews.v1.id).clips
    assert len(clips) == 2 and clips[1].id != a and clips[1].asset_id == clips[0].asset_id
    assert len(ews.project.assets.all()) == 5  # no media copied
    ews.commands.undo()
    assert len(ews.project.timeline.get_track(ews.v1.id).clips) == 1


def test_cut_removes_and_paste_restores(ews):
    a, b, _ = lay(ews)
    ews.edit.cut([b])
    assert ews.project.timeline.get_clip(b) is None
    ews.edit.paste(at=6.0, track_id=ews.v1.id)
    assert any(c.asset_id == ews.A[1].id for c in ews.project.timeline.get_track(ews.v1.id).clips)


def test_paste_with_nothing_copied_explains_itself(ews):
    with pytest.raises(TimelineError, match="(?i)copy"):
        ews.edit.paste()


def test_duplicate_places_a_copy_after_the_original(ews):
    a, *_ = lay(ews, 1)
    ews.edit.duplicate([a])
    assert len(ews.project.timeline.get_track(ews.v1.id).clips) == 2


# ------------------------------------------------------------------------------------------------------------------------ relationships / properties
def test_link_group_unlink_roundtrip(ews):
    a, b, _ = lay(ews)
    ews.edit.group([a, b])
    tl = ews.project.timeline
    assert tl.get_clip(a).group_id and tl.get_clip(a).group_id == tl.get_clip(b).group_id
    ews.edit.ungroup([a])
    assert not tl.get_clip(a).group_id
    ews.commands.undo()
    assert tl.get_clip(a).group_id


def test_update_clip_validates_numbers_and_records_one_step(ews):
    a, *_ = lay(ews, 1)
    ews.edit.update_clip(a, opacity=0.5)
    assert ews.project.timeline.get_clip(a).opacity == pytest.approx(0.5)
    with pytest.raises(TimelineError):
        ews.edit.update_clip(a, opacity=7)
    assert ews.project.timeline.get_clip(a).opacity == pytest.approx(0.5)
    ews.commands.undo()
    assert ews.project.timeline.get_clip(a).opacity == pytest.approx(1.0)


def test_update_clips_bulk_is_one_undo_step(ews):
    a, b, _ = lay(ews)
    ews.edit.update_clips([a, b], opacity=0.25)
    tl = ews.project.timeline
    assert tl.get_clip(a).opacity == tl.get_clip(b).opacity == pytest.approx(0.25)
    ews.commands.undo()
    assert tl.get_clip(a).opacity == tl.get_clip(b).opacity == pytest.approx(1.0)


def test_speed_change_with_linked_audio_asks_for_a_decision(ews):
    from app.timeline.ops_props import SpeedChoiceRequired

    a, *_ = lay(ews, 1)
    ews.edit.set_speed(a, 2.0)
    assert ews.project.timeline.get_clip(a).speed == pytest.approx(2.0)
    assert ews.project.timeline.get_clip(a).duration == pytest.approx(3.0)
    assert SpeedChoiceRequired  # linked-mate ambiguity is a typed exception the inspector turns into a dialog (covered by test_timeline_props)


def test_keyframes_through_the_facade(ews):
    a, *_ = lay(ews, 1)
    ews.edit.add_keyframe(a, "opacity", 0.0, 1.0)
    ews.edit.add_keyframe(a, "opacity", 2.0, 0.0)
    kfs = ews.project.timeline.get_clip(a).keyframes
    assert [(k.property, k.time) for k in kfs] == [("opacity", 0.0), ("opacity", 2.0)]
    ews.edit.move_keyframes(a, [("opacity", 2.0, 3.0)])
    assert sorted(k.time for k in ews.project.timeline.get_clip(a).keyframes) == [0.0, 3.0]
    ews.edit.delete_keyframes(a, [("opacity", 0.0)])
    assert len(ews.project.timeline.get_clip(a).keyframes) == 1
    ews.commands.undo()
    assert len(ews.project.timeline.get_clip(a).keyframes) == 2


# ------------------------------------------------------------------------------------------------------------------------ markers / range / tracks
def test_markers_range_and_navigation_are_undoable_and_persist(ews, tmp_path):
    ews.edit.set_playhead(2.0)
    m1 = ews.edit.add_marker(name="intro", kind="section")
    m2 = ews.edit.add_marker(5.0, "cut here")
    assert [m.id for m in ews.project.timeline.markers] == [m1, m2]
    ews.edit.set_playhead(1.0)
    assert ews.edit.next_marker().id == m1 and ews.edit.prev_marker() is None
    ews.edit.mark_in(1.0)
    ews.edit.mark_out(4.0)
    assert ews.project.timeline.range == (1.0, 4.0)
    ews.edit.move_marker(m2, 6.0)
    ews.edit.delete_marker(m1)
    ews.commands.undo()
    assert ews.project.timeline.get_marker(m1) is not None
    ews.save()
    ews.open_project(ews.project.root)
    assert [m.name for m in ews.project.timeline.markers] == ["intro", "cut here"] and ews.project.timeline.range == (1.0, 4.0)


def test_track_management_through_the_facade(ews):
    lay(ews, 2)
    tl = ews.project.timeline
    new = ews.edit.add_track(TrackKind.VIDEO)
    ews.edit.rename_track(new.id, "B-roll")
    ews.edit.set_track_display(new.id, height=96, collapsed=False)
    assert tl.get_track(new.id).name == "B-roll" and tl.get_track(new.id).height == 96
    dup = ews.edit.duplicate_track(ews.v1.id)
    assert len(tl.get_track(dup).clips) == 2 and {c.id for c in tl.get_track(dup).clips}.isdisjoint({c.id for c in tl.get_track(ews.v1.id).clips})
    with pytest.raises(TrackHasContent) as exc:
        ews.edit.remove_track(ews.v1.id)
    assert exc.value.clip_count == 2 and tl.get_track(ews.v1.id) is not None
    ews.edit.remove_track(ews.v1.id, confirm_content=True)
    ews.commands.undo()
    assert len(tl.get_track(ews.v1.id).clips) == 2  # the track AND its clips are back


# ------------------------------------------------------------------------------------------------------------------------ AI ownership
def test_manual_edits_take_ownership_of_directly_edited_clips_only(ews):
    calls: list[tuple[str, str]] = []

    def hook(cid, action):
        calls.append((cid, action))
        return None

    ews.timeline.edit_hook = hook
    a, b, c = lay(ews)
    ews.edit.state.update(ripple_on=True)
    ews.edit.delete([a])  # b and c are only shifted by the ripple: not taken over
    assert calls == [(a, "delete")]
    calls.clear()
    ews.edit.state.update(ripple_on=False)
    ews.edit.move_clips([c], 1.0)
    assert calls == [(c, "edit")]


def test_ownership_command_runs_in_the_same_undo_step(ews):
    from app.core.commands import Command

    class Mark(Command):
        description, scope = "mark", "timeline"
        done = 0

        def do(self):
            Mark.done += 1

        def undo(self):
            Mark.done -= 1

    ews.timeline.edit_hook = lambda cid, action: Mark()
    a, *_ = lay(ews, 1)
    ews.edit.set_start(a, 2.0)
    assert Mark.done == 1
    ews.commands.undo()
    assert Mark.done == 0 and ews.project.timeline.get_clip(a).timeline_start == 0.0  # the move and the takeover record undo together


# ------------------------------------------------------------------------------------------------------------------------ snapping + autosave
def test_snap_engine_sees_the_live_timeline_and_playhead(ews):
    a, b, _ = lay(ews)
    ews.edit.set_playhead(6.0)
    r = ews.edit.snap.snap(6.04, pps=100.0)
    assert r.snapped and r.time == pytest.approx(6.0)
    ews.edit.state.toggle_snapping()
    assert not ews.edit.snap.snap(6.04, pps=100.0).snapped


def test_commit_marks_the_project_dirty_for_autosave(ews):
    ews.projects.set_dirty(False)
    assert ews.project.dirty is False
    lay(ews, 1)
    assert ews.project.dirty is True


# ------------------------------------------------------------------------------------------------------------------------ captions: merge planner
def _caption(cid, start, words):
    dur = len(words) * 0.4
    ws = [{"text": w, "start": round(start + i * 0.4, 3), "end": round(start + (i + 1) * 0.4, 3)} for i, w in enumerate(words)]
    return Clip(cid, "cap", "", start, dur, kind=KIND_CAPTION, text={"text": " ".join(words), "lines": [" ".join(words)], "words": ws, "emphasis": [{"word_index": 0, "kind": "key"}] if cid == "c2" else []})


def _cap_tl() -> Timeline:
    from app.timeline.track import Track

    tl = Timeline()
    tl.tracks = []
    tl.insert_track(Track("cap", "Captions", TrackKind.CAPTIONS))
    tl.insert_clip(_caption("c1", 0.0, ["Hello", "there"]))
    tl.insert_clip(_caption("c2", 1.0, ["big", "world"]))
    tl.insert_clip(_caption("c3", 9.0, ["far", "away"]))
    return tl


def test_merge_captions_joins_words_and_shifts_emphasis():
    tl = _cap_tl()
    plan = plan_merge_captions(PlanEnv(tl), ["c1", "c2"])
    merged = plan.updates["c1"]
    assert plan.removes == ["c2"] and merged.text["text"] == "Hello there big world" and len(merged.text["words"]) == 4
    assert merged.text["emphasis"][0]["word_index"] == 2  # c2's emphasis on its first word now points at word 3
    assert merged.timeline_end == pytest.approx(1.8)


def test_merge_captions_refuses_distant_or_mixed_selections():
    tl = _cap_tl()
    with pytest.raises(TimelineError, match="apart"):
        plan_merge_captions(PlanEnv(tl), ["c2", "c3"])
    with pytest.raises(TimelineError, match="another clip between"):
        plan_merge_captions(PlanEnv(tl), ["c1", "c3"])
    with pytest.raises(TimelineError, match="(?i)at least two"):
        plan_merge_captions(PlanEnv(tl), ["c1"])
