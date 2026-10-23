"""Phase 10: manual edits made through ``ws.edit`` take ownership of AI-created clips exactly like the older timeline service did - recorded in the same undo step, never overwritten by a
later AI regeneration - and the user's own locks (item and aspect level) keep the AI's hands off too."""

from __future__ import annotations

import pytest

from app.core.exceptions import TimelineError
from app.editing.models import Creator
from app.tests.conftest import needs_ffmpeg
from app.tests.test_editing_service import clips, gen, scene_with  # noqa: F401

pytestmark = needs_ffmpeg


def single_visual(ws):
    sc = scene_with(ws, lambda s: len(clips(ws, s.id, "media")) == 1)
    return sc, clips(ws, sc.id, "media")[0]


def geometry(c):
    return (c.id, round(c.timeline_start, 4), round(c.duration, 4), c.asset_id, round(c.source_in, 4), round(c.source_out, 4))


def test_a_manual_trim_through_ws_edit_takes_ownership_in_the_same_undo_step(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    _sc, clip = single_visual(ws)
    dec = p.editing_decisions[clip.ai_decision_id]
    assert clip.created_by == "AI" and dec.created_by is Creator.AI
    ws.edit.state.update(linked_selection=False)
    ws.edit.trim(clip.id, "end", clip.timeline_end - 0.4)
    clip = p.timeline.get_clip(clip.id)
    now = p.editing_decisions[clip.ai_decision_id]
    assert clip.created_by == "USER" and now.created_by is Creator.USER and now.overrides_decision_id == dec.decision_id
    assert now.duration == pytest.approx(clip.duration) and ws.editing.override_of(now.decision_id).decision_id == dec.decision_id
    ws.undo()  # ONE undo reverts the trim and the ownership record
    clip = p.timeline.get_clip(clip.id)
    assert clip.created_by == "AI" and p.editing_decisions[clip.ai_decision_id].decision_id == dec.decision_id and not p.ai_overrides
    ws.redo()
    assert p.timeline.get_clip(clip.id).created_by == "USER"


@pytest.mark.parametrize("how", ["move", "trim_start", "split", "speed", "opacity"])
def test_regeneration_never_overwrites_a_clip_the_user_edited_with_the_new_tools(edit_ws, how):
    ws = edit_ws
    gen(ws)
    sc, clip = single_visual(ws)
    ws.edit.state.update(linked_selection=False)
    if how == "move":
        ws.edit.move_clips([clip.id], 0.2)
    elif how == "trim_start":
        ws.edit.trim(clip.id, "start", clip.timeline_start + 0.3)
    elif how == "split":
        ws.edit.split(clip.timeline_start + clip.duration / 2, [clip.id])
    elif how == "speed":
        ws.edit.set_speed(clip.id, 1.5)
    else:
        ws.edit.update_clip(clip.id, opacity=0.6)
    edited = sorted((geometry(c) for c in clips(ws, sc.id, "media")), key=lambda g: g[1])
    owners = {c.id: c.created_by for c in clips(ws, sc.id, "media")}
    assert "USER" in owners.values()
    ws.editing.regenerate_all()
    assert ws.jobs.wait_idle(60)
    after = sorted((geometry(c) for c in clips(ws, sc.id, "media")), key=lambda g: g[1])
    for g in edited:  # every piece the user made is still there, unchanged (regeneration may add what it owns elsewhere, never touch these)
        assert g in after, (how, g)
    assert not [i for i in ws.editing.validate_timeline() if i.severity == "error"]


def test_ripple_shifted_followers_stay_ai_owned_but_the_deleted_clip_is_recorded(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    ws.edit.state.update(ripple_on=True, linked_selection=False)
    first = next(c for c in sorted(p.timeline.all_clips(), key=lambda c: c.timeline_start) if c.kind == "media" and c.created_by == "AI")
    track_id = first.track_id
    followers = [c for c in p.timeline.get_track(track_id).clips if c.timeline_start >= first.timeline_end - 1e-6 and c.created_by == "AI"]
    assert followers
    slot_key = f"{first.scene_id}|{first.slot}"
    ws.edit.delete([first.id])
    for f in followers:
        now = p.timeline.get_clip(f.id)
        assert now.timeline_start == pytest.approx(f.timeline_start - first.duration, abs=1 / 30) and now.created_by == "AI"  # shifted by the ripple, not taken over
    assert slot_key in p.timeline_generation.suppressed_slots  # the user's deletion is remembered, so regeneration does not bring the clip back
    ws.undo()
    assert p.timeline.get_clip(first.id) is not None and slot_key not in p.timeline_generation.suppressed_slots


def test_an_item_lock_set_by_the_user_keeps_ai_regeneration_away(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    sc, clip = single_visual(ws)
    ws.edit.set_locks([clip.id], "item")
    assert p.timeline.get_clip(clip.id).protected
    before = sorted(geometry(c) for c in clips(ws, sc.id))
    ws.editing.update_settings(style="dynamic")
    ws.editing.regenerate_scene(sc.id)
    assert ws.jobs.wait_idle(60)
    assert sorted(geometry(c) for c in clips(ws, sc.id) if c.id == clip.id) == [geometry(clip)]  # the locked clip is untouched
    assert p.timeline.get_clip(clip.id) is not None
    assert before  # (the scene's other, unlocked pieces may be regenerated)
    with pytest.raises(TimelineError, match="(?i)lock"):
        ws.edit.trim(clip.id, "end", clip.timeline_end - 0.2)  # and the user's own lock holds against the user until removed
    ws.edit.set_locks([clip.id], "item", "remove")
    ws.edit.trim(clip.id, "end", clip.timeline_end - 0.2)


@pytest.mark.parametrize("aspect,allowed,blocked", [("timing", "move", "trim"), ("position", "trim", "move")])
def test_aspect_locks_block_only_their_aspect_for_the_user(edit_ws, aspect, allowed, blocked):
    ws = edit_ws
    gen(ws)
    _sc, clip = single_visual(ws)
    ws.edit.state.update(linked_selection=False)
    ws.edit.set_locks([clip.id], aspect)

    def do(what):
        c = ws.project.timeline.get_clip(clip.id)
        if what == "move":
            ws.edit.move_clips([clip.id], 0.1)
        else:
            ws.edit.trim(clip.id, "end", c.timeline_end - 0.1)

    with pytest.raises(TimelineError, match="(?i)lock"):
        do(blocked)
    do(allowed)
