"""Phase 10: QC and the change log stay in step with manual edits made through ``ws.edit`` - the same invariants Phase 9 proved for the older commands: staleness is known
immediately and bounded, and an incremental QC run equals a full run from scratch."""

from __future__ import annotations

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.test_incremental_qc import (  # noqa: F401  (fixtures: the module-scoped synthetic workspace with a completed full QC run)
    baseline,
    media,
    sid,
    test_incremental_equals_a_full_run_after_each_kind_of_edit as incremental_equals_full,
    visual_clip,
)


def edit_p10_move(ws):
    c = next(c for t in ws.project.timeline.tracks for c in t.clips if c.scene_id == sid(5) and c.kind == "text")
    ws.edit.state.update(linked_selection=False)
    ws.edit.move_clips([c.id], 0.3)
    return {"direct": {sid(5)}, "all": False}


def edit_p10_trim(ws):
    c = visual_clip(ws.project, 3)
    ws.edit.state.update(linked_selection=False)
    ws.edit.trim(c.id, "end", c.timeline_end - 0.4)
    return {"direct": {sid(3)}, "all": False}


def edit_p10_blade(ws):
    c = visual_clip(ws.project, 8)
    ws.edit.state.update(linked_selection=False)
    ws.edit.blade(c.timeline_start + c.duration / 2, c.track_id)
    return {"direct": {sid(8)}, "all": False}


def edit_p10_lock(ws):
    c = visual_clip(ws.project, 2)
    ws.edit.set_locks([c.id], "timing")
    return {"direct": {sid(2)}, "all": False}


def edit_p10_opacity(ws):
    c = visual_clip(ws.project, 11)
    ws.edit.update_clip(c.id, opacity=0.5)
    return {"direct": {sid(11)}, "all": False}


EDITS = [edit_p10_move, edit_p10_trim, edit_p10_blade, edit_p10_lock, edit_p10_opacity]


@needs_ffmpeg
@pytest.mark.parametrize("edit", EDITS, ids=[e.__name__[5:] for e in EDITS])
def test_incremental_qc_equals_a_full_run_after_a_phase10_edit(baseline, edit, monkeypatch):  # noqa: F811
    incremental_equals_full(baseline, edit, monkeypatch)
