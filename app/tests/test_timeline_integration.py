"""Phase 10 integration: manual edits agree with the render, survive save / reopen, undo exactly, invalidate only what they touch and keep autosave honest."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from app.analysis.models import Scene
from app.core.exceptions import TimelineError
from app.rendering.models import RenderStatus
from app.tests.conftest import import_and_wait, needs_ffmpeg, settle
from app.tests.helpers import make_image, make_video, write_tone_wav
from app.tests.render_helpers import BLUE, GREEN, RED, color_video, frame_at, near, px, wait_job
from app.timeline.track import TrackKind

pytestmark = needs_ffmpeg


def state(ws) -> str:
    return json.dumps(ws.project.timeline.to_dict(), sort_keys=True)


def track(ws, kind=TrackKind.VIDEO):
    return next(t for t in ws.project.timeline.tracks if t.kind is kind)


@pytest.fixture
def cut_ws(project_ws, tmp_path):
    """A 3 s red/green/blue video (1 s each) on V1 and a 3 s tone on A1, exportable at 480p draft."""
    ws = project_ws
    d = tmp_path / "m"
    d.mkdir()
    rgb = import_and_wait(ws, color_video(d / "rgb.mp4", ("red", "green", "blue"), 1.0, "320x180", 30))
    tone = import_and_wait(ws, write_tone_wav(d / "tone.wav", 3.0, 0.3, 220.0, 48000))
    ws.rgb, ws.tone = rgb, tone
    ws.edit.place_assets([rgb.id], at=0.0, track_id=track(ws).id)
    ws.edit.place_assets([tone.id], at=0.0, track_id=track(ws, TrackKind.AUDIO).id)
    ws.project.scenes = [Scene("s1", "1", 0.0, 3.0)]
    ws.project.timeline_version = 1
    ws.render.update_settings(resolution="480p", quality="draft", preset_id="draft")
    ws.commands.clear()
    return ws


def export(ws, tmp_path, name):
    job = wait_job(ws.render.start_export(tmp_path / "out" / f"{name}.mp4"))
    assert job.status is RenderStatus.COMPLETED, job.error
    return Path(job.record.output_path), ws.render.engine.probe.probe(Path(job.record.output_path))


def centre_colour(out, t):
    return px(frame_at(out, t), 0.5, 0.5)


# ------------------------------------------------------------------------------------------------------------------------ the render shows exactly what the timeline says
def test_cut_and_ripple_delete_are_what_the_export_contains(cut_ws, tmp_path):
    ws = cut_ws
    v = track(ws)
    ws.edit.state.update(linked_selection=False, ripple_on=True)
    ws.edit.blade(1.0, v.id)
    ws.edit.blade(2.0, v.id)
    clips = ws.project.timeline.get_track(v.id).clips
    assert [round(c.duration, 3) for c in clips] == [1.0, 1.0, 1.0] and clips[1].source_in == pytest.approx(1.0)
    ws.edit.delete([clips[1].id])  # ripple delete the green second
    assert ws.project.timeline.duration == pytest.approx(3.0)  # the tone on A1 still runs 3 s (the audio track is not in the ripple scope)
    out, info = export(ws, tmp_path, "cut")
    assert info.duration == pytest.approx(3.0, abs=0.15)
    assert near(centre_colour(out, 0.5), RED) and near(centre_colour(out, 1.5), BLUE)  # green is gone, blue follows red
    ws.commands.undo()  # undo the ripple delete: green is back
    out2, _ = export(ws, tmp_path, "uncut")
    assert near(centre_colour(out2, 1.5), GREEN) and near(centre_colour(out2, 2.5), BLUE)


def test_trim_slip_and_speed_change_the_export_frames(cut_ws, tmp_path):
    ws = cut_ws
    v = track(ws)
    cid = ws.project.timeline.get_track(v.id).clips[0].id
    ws.edit.state.update(linked_selection=False)
    ws.edit.trim(cid, "start", 1.0)  # head trim: the clip now starts at 1 s showing the green part first
    c = ws.project.timeline.get_clip(cid)
    assert c.timeline_start == pytest.approx(1.0) and c.source_in == pytest.approx(1.0)
    out, _ = export(ws, tmp_path, "trim")
    assert near(centre_colour(out, 0.5), (0, 0, 0)) and near(centre_colour(out, 1.5), GREEN) and near(centre_colour(out, 2.5), BLUE)
    ws.edit.set_start(cid, 0.0)
    lo, hi = ws.edit.slip_limits(cid)
    assert (round(lo, 3), round(hi, 3)) == (-1.0, 0.0)  # the clip shows source 1..3 of a 3 s file: it can only slip earlier
    with pytest.raises(TimelineError, match="can only slip"):
        ws.edit.slip(cid, 1.0)
    ws.edit.slip(cid, -1.0)  # the same 2 s window of the timeline, content now starting at the beginning of the media
    c = ws.project.timeline.get_clip(cid)
    assert c.timeline_start == 0.0 and c.source_in == pytest.approx(0.0) and c.duration == pytest.approx(2.0)
    out, _ = export(ws, tmp_path, "slip")
    assert near(centre_colour(out, 0.5), RED) and near(centre_colour(out, 1.5), GREEN)
    ws.edit.set_speed(cid, 2.0)  # twice as fast: the 2 s of source now take 1 s on the timeline
    c = ws.project.timeline.get_clip(cid)
    assert c.speed == pytest.approx(2.0) and c.duration == pytest.approx(1.0)
    out2, info = export(ws, tmp_path, "fast")
    assert near(centre_colour(out2, 0.25), RED) and near(centre_colour(out2, 0.75), GREEN)  # red then green, each half as long


# ------------------------------------------------------------------------------------------------------------------------ undo / redo is exact (randomised)
def random_ops(ws, rng):
    tl = ws.project.timeline
    clips = [c for c in tl.all_clips() if c.kind == "media"]
    c = rng.choice(clips) if clips else None
    ids = [x.id for x in rng.sample(clips, min(len(clips), rng.randint(1, 3)))] if clips else []
    t = round(rng.uniform(0, max(1.0, tl.duration)), 2)
    if c is None or rng.random() < 0.12:  # keep the timeline populated: an empty one would make the rest of the sequence a no-op
        return lambda: ws.edit.place_assets([ws.rgb.id], at=t, track_id=track(ws).id, mode=rng.choice(["free", "free", "insert", "overwrite"]))
    ops = [
        lambda: ws.edit.move_clips(ids, rng.choice([-1.0, -0.5, 0.33, 1.0, 2.5])),
        lambda: ws.edit.trim(c.id, rng.choice(["start", "end"]), c.timeline_start + rng.uniform(0.1, c.duration)),
        lambda: ws.edit.split(t, [c.id]),
        lambda: ws.edit.delete(ids[:1], ripple=rng.random() < 0.5),
        lambda: ws.edit.blade(t, c.track_id),
        lambda: ws.edit.cut(ids[:1]),
        lambda: ws.edit.duplicate(ids),
        lambda: (ws.edit.copy(ids), ws.edit.paste(at=t, track_id=c.track_id)),
        lambda: ws.edit.update_clip(c.id, opacity=round(rng.uniform(0, 1), 2)) if c.track_id.startswith("track_v") else ws.edit.update_clip(c.id, volume=round(rng.uniform(0, 1.5), 2)),
        lambda: ws.edit.set_speed(c.id, rng.choice([0.5, 1.0, 1.5, 2.0])),
        lambda: ws.edit.group(ids),
        lambda: ws.edit.set_locks(ids, "position", rng.choice(["add", "remove"])),
        lambda: ws.edit.add_marker(t, "m"),
        lambda: ws.edit.add_keyframe(c.id, "opacity" if c.track_id.startswith("track_v") else "volume", round(rng.uniform(0, c.duration), 2), round(rng.uniform(0, 1), 2)),
        lambda: ws.edit.set_range(t, t + 1.0),
        lambda: ws.edit.move_clips(ids, 1.0, ripple=True),
    ]
    return rng.choice(ops)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_random_edit_sequences_undo_and_redo_exactly_and_failures_change_nothing(cut_ws, tmp_path, seed):
    ws = cut_ws
    rng = random.Random(seed)
    extra = import_and_wait(ws, make_video(tmp_path / "x.mp4", 5.0, "testsrc2"))
    ws.edit.place_assets([extra.id], at=4.0, track_id=track(ws).id)
    ws.commands.clear()
    ws.edit.state.update(linked_selection=rng.random() < 0.5)
    history = [state(ws)]
    for _ in range(80):
        before = history[-1]
        try:
            random_ops(ws, rng)()
        except TimelineError:
            assert state(ws) == before  # a refused edit leaves no trace
            continue
        after = state(ws)
        if after != before:
            history.append(after)
    assert len(history) > 8  # the sequence really edited things
    for expected in reversed(history[:-1]):  # undo step by step: every intermediate state is restored exactly
        assert ws.commands.can_undo
        ws.commands.undo()
        assert state(ws) == expected
    assert not ws.commands.can_undo
    for expected in history[1:]:
        assert ws.commands.can_redo
        ws.commands.redo()
        assert state(ws) == expected


# ------------------------------------------------------------------------------------------------------------------------ persistence
def test_every_phase10_field_survives_save_close_reopen(cut_ws):
    ws = cut_ws
    tl = ws.project.timeline
    v = track(ws)
    ws.edit.state.update(linked_selection=False)
    ws.edit.blade(1.0, v.id)
    a, b = tl.get_track(v.id).clips
    ws.edit.group([a.id, b.id])
    ws.edit.link([a.id, b.id])
    ws.edit.set_locks([b.id], "timing")
    ws.edit.add_keyframe(a.id, "opacity", 0.2, 0.4)
    ws.edit.add_keyframe(a.id, "opacity", 0.8, 1.0, "ease_in_out")
    ws.edit.set_transition(b.id, "FADE", 0.3)
    m = ws.edit.add_marker(1.5, "act two", "section", "#ff8800", "turn")
    ws.edit.set_range(0.5, 2.5)
    ws.edit.set_track_display(v.id, height=80, collapsed=False)
    ws.edit.set_track_display(track(ws, TrackKind.AUDIO).id, collapsed=True)
    before = state(ws)
    root = ws.project.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert state(ws) == before
    tl = ws.project.timeline
    assert tl.get_marker(m).color == "#ff8800" and tl.range == (0.5, 2.5) and tl.get_track(v.id).height == 80 and tl.get_track(track(ws, TrackKind.AUDIO).id).collapsed
    ca, cb = tl.get_track(v.id).clips
    assert ca.group_id and ca.group_id == cb.group_id and ca.link_id and ca.link_id == cb.link_id and "timing" in cb.locks and cb.protected
    assert cb.transition and cb.transition["type"] == "FADE" and cb.transition["duration"] == pytest.approx(0.3)
    assert [(k.time, k.value, k.interpolation) for k in ca.keyframes] == [(0.2, 0.4, "linear"), (0.8, 1.0, "ease_in_out")]
    # the loaded project is fully editable and undo history starts clean
    assert not ws.commands.can_undo
    with pytest.raises(TimelineError, match="(?i)lock"):
        ws.edit.trim(cb.id, "end", 1.5)


def test_a_version_8_project_opens_unchanged_and_new_fields_are_optional(cut_ws):
    ws = cut_ws
    doc = ws.project.to_document()
    assert doc["schema_version"] == 9
    for t in doc["timeline"]["tracks"]:
        assert "height" not in t and "collapsed" not in t  # defaults are not written: a v8 file and a v9 file without Phase 10 data are the same shape
    assert "markers" not in doc["timeline"] and "range" not in doc["timeline"]
    for t in doc["timeline"]["tracks"]:
        for c in t["clips"]:
            assert not {"link_id", "group_id", "locks"} & set(c)
    doc["schema_version"] = 8
    root = ws.project.root
    (root / "project.json").write_text(json.dumps(doc))
    before = state(ws)
    ws.close_project()
    ws.open_project(root)
    assert state(ws) == before and ws.project.to_document()["schema_version"] == 9


# ------------------------------------------------------------------------------------------------------------------------ autosave + change tracking + AI protection
def test_a_committed_edit_autosaves_but_a_drag_preview_does_not_touch_the_project(cut_ws):
    ws = cut_ws
    ws.save()  # clean slate: saved, no recovery data
    settle(ws)
    assert ws.pending_recovery() == [] and not ws.project.dirty
    v = track(ws)
    cid = ws.project.timeline.get_track(v.id).clips[0].id
    before = state(ws)
    revision = ws.project.timeline.revision
    plan, reason = ws.edit.preview_move([cid], 1.0)  # what a drag asks for on every mouse move
    plan2, reason2 = ws.edit.preview_trim(cid, "end", 2.0)
    assert plan is not None and plan2 is not None and reason is None and reason2 is None
    settle(ws)
    assert state(ws) == before and not ws.commands.can_undo and not ws.project.dirty and ws.pending_recovery() == []  # previews changed and saved nothing
    assert ws.project.timeline.revision == revision  # ... and did not even invalidate the timeline's indexes / the snap engine's caches
    ws.edit.state.update(linked_selection=False)
    ws.edit.move_clips([cid], 1.0)  # the commit: one undo step, project dirty, autosave written
    settle(ws)
    assert ws.project.dirty and ws.commands.can_undo and len(ws.pending_recovery()) == 1
    assert ws.commands.undo_text.lower().startswith("move")


def test_a_burst_of_edits_does_not_reserialise_the_project_each_time_but_the_first_edit_always_autosaves(cut_ws):
    ws = cut_ws
    ws.save()
    settle(ws)
    ws.edit.state.update(linked_selection=False)
    cid = ws.project.timeline.get_track(track(ws).id).clips[0].id
    ws.autosave._cost = 10.0  # pretend serialising this project is slow (0.2 s would already be): the next snapshot has to wait
    ws.edit.move_clips([cid], 0.5)
    settle(ws)
    assert len(ws.pending_recovery()) == 1  # the first edit after a save is snapshotted
    skipped = ws.autosave.throttled
    for _ in range(5):
        ws.edit.move_clips([cid], 0.1)
    settle(ws)
    assert ws.autosave.throttled == skipped + 5  # the burst was coalesced ...
    assert ws.autosave_tick()  # ... and the periodic tick (never throttled) writes the latest state
    settle(ws)


def test_change_tracker_marks_only_what_an_edit_touched(cut_ws):
    ws = cut_ws
    ws.project.scenes = [Scene("s1", "1", 0.0, 1.0), Scene("s2", "2", 1.0, 2.0), Scene("s3", "3", 2.0, 3.0)]
    ws.edit.state.update(linked_selection=False)
    v = track(ws)
    ws.edit.blade(1.0, v.id)
    ws.edit.blade(2.0, v.id)
    ws.changes.consume("t")
    middle = ws.project.timeline.get_track(v.id).clips[1]
    ws.edit.trim(middle.id, "end", 1.8)
    cs = ws.changes.peek(consumer="t")
    assert "s2" in cs.direct_scene_ids and "s1" in cs.scene_ids and "s3" in cs.scene_ids  # the edited scene and its neighbours (transitions / continuity read across)
    assert not cs.all_scenes and not cs.unknown and "timeline" in cs.domains
    ws.changes.consume("t")
    ws.commands.undo()
    assert "s2" in ws.changes.peek(consumer="t").direct_scene_ids  # undo is tracked the same way


def test_an_item_lock_protects_the_clip_from_the_user_and_from_ai_checks(cut_ws):
    ws = cut_ws
    cid = ws.project.timeline.get_track(track(ws).id).clips[0].id
    ws.edit.set_locks([cid], "item")
    clip = ws.project.timeline.get_clip(cid)
    assert clip.protected and "item" in clip.locks  # `protected` is what every AI regeneration check reads
    with pytest.raises(TimelineError, match="(?i)lock"):
        ws.edit.move_clips([cid], 1.0)  # the user's own lock is respected by the user too
    ws.edit.set_locks([cid], "item", "remove")
    assert not ws.project.timeline.get_clip(cid).protected
    ws.edit.set_start(cid, 1.0)
    assert ws.project.timeline.get_clip(cid).timeline_start == pytest.approx(1.0)


def test_image_and_audio_media_place_on_matching_tracks_and_unsupported_pairs_are_refused(cut_ws, tmp_path):
    ws = cut_ws
    img = import_and_wait(ws, make_image(tmp_path / "p.png"))
    ws.edit.place_assets([img.id], at=5.0)
    placed = [c for c in ws.project.timeline.all_clips() if c.asset_id == img.id]
    assert len(placed) == 1 and not ws.project.timeline.get_track(placed[0].track_id).is_audio
    with pytest.raises(TimelineError):
        ws.edit.place_assets([ws.tone.id], at=0.0, track_id=track(ws).id)  # audio onto a video track
    assert len([c for c in ws.project.timeline.all_clips() if c.asset_id == ws.tone.id]) == 1
