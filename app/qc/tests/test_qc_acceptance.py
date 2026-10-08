"""Phase 8 acceptance: the whole workflow on a real project with real media, through the same services the UI uses.

run QC -> inspect findings -> find them on the timeline -> preview a fix -> apply it -> undo -> apply again -> re-run QC -> the score improves -> export -> the rendered
file is checked -> save, close and reopen -> results, history and the post-render report are intact.
"""

from __future__ import annotations

import json

import pytest

from app.qc.issue_model import IssueStatus
from app.qc.severity import Severity
from app.qc.tests.conftest import needs_ffmpeg
from app.rendering.models import RenderStatus
from app.tests.render_helpers import wait_job

pytestmark = needs_ffmpeg


def run_qc(ws, **kw):
    job = ws.qc.run_full_qc(**kw)
    assert ws.jobs.wait_idle(120)
    assert job.status.value == "COMPLETED"
    return ws.project.qc_runs[-1]


def timeline_json(ws) -> str:
    return json.dumps(ws.project.timeline.to_dict(), sort_keys=True)


def music_clip(ws):
    return next(c for c in ws.project.timeline.get_track("track_a2").clips)


@pytest.fixture
def ws(render_ws):
    """The Phase 6 demo, prepared like an AI edit left it: consistent source ranges, an AI-made music bed that plays at full level over the voice."""
    w = render_ws
    for t in w.project.timeline.tracks:
        for c in t.clips:
            if c.kind == "media" and t.kind.value in ("video", "image"):
                c.source_out = c.source_in + c.duration * c.speed
    m = music_clip(w)
    m.keyframes = []
    m.audio["volume"] = 1.0
    m.created_by = "AI"
    m.metadata["assignment_id"] = "music_bed"
    return w


def test_the_whole_quality_control_workflow(ws):
    p = ws.project
    # ---- 1. run QC (background job, all 16 checkers) and inspect it
    before_edit = timeline_json(ws)
    run1 = run_qc(ws)
    assert timeline_json(ws) == before_edit  # analysis never edits
    assert run1["state"] == "COMPLETED" and len(run1["checkers"]) == 16 and all(c["state"] in ("DONE", "CACHED") for c in run1["checkers"].values()), run1["checkers"]
    score1 = p.qc_scores.overall
    issues = ws.qc.issues()
    assert issues and p.qc_scores.counts["CRITICAL"] == 0
    loud = [i for i in issues if i.code in ("audio.music_masks_speech", "audio.insufficient_ducking", "audio.music_loud")]
    assert loud, [i.code for i in issues]
    target = next((i for i in loud if i.fix and i.fix.kind == "audio.duck"), loud[0])
    assert target.fix is not None and target.fix.kind == "audio.duck" and target.auto_fix_available and target.severity in (Severity.ERROR, Severity.WARNING)
    # ---- 2. every finding can be found: scene / time / clip
    assert target.start_time is not None and target.timeline_item_id and p.timeline.get_clip(target.timeline_item_id) is not None
    assert ws.qc.markers() and all("time" in m for m in ws.qc.markers())
    # ---- 3. preview the fix: nothing changes
    prev = ws.qc.preview_fix(target.issue_id)
    assert not prev.blocked_reason and prev.before and prev.after and prev.before != prev.after
    assert timeline_json(ws) == before_edit
    # ---- 4. apply (one undoable step), see the keyframes, the issue is marked fixed
    ws.qc.apply_fix(target.issue_id, confirmed=True)
    assert music_clip(ws).keyframes and timeline_json(ws) != before_edit
    assert next(i for i in p.qc_issues if i.issue_id == target.issue_id).status is IssueStatus.FIXED and p.qc_fixes and not p.qc_fixes[-1].reverted
    # ---- 5. undo restores the timeline exactly, and the finding is open again
    ws.undo()
    assert timeline_json(ws) == before_edit
    assert next(i for i in p.qc_issues if i.issue_id == target.issue_id).status is IssueStatus.OPEN
    # ---- 6. apply again, re-run: the finding is gone and the score went up
    ws.qc.apply_fix(target.issue_id, confirmed=True)
    run2 = run_qc(ws)
    assert run2["number"] == 2 and p.qc_scores.overall > score1, (score1, p.qc_scores.overall)
    assert not [i for i in ws.qc.issues() if i.code == target.code and i.scene_id == target.scene_id and i.start_time == target.start_time]
    cmp = ws.qc.compare_runs(1, 2)
    assert cmp.resolved_issues and cmp.improved and cmp.overall[1] > cmp.overall[0]
    # ---- 7. export: allowed by the gate, rendered, and the file is checked afterwards
    gate = ws.qc.export_gate()
    assert gate.run_current and not gate.decision.blocked, gate.decision
    job = wait_job(ws.render.start_export(settings=None, kind="export"))
    assert job.status is RenderStatus.COMPLETED
    assert ws.jobs.wait_idle(120)
    rid = job.id
    res = p.render_qc_results[rid]
    assert res["status"] in ("PASSED", "WARNINGS") and {c["id"] for c in res["checks"]} >= {"duration", "resolution", "black_frames", "corruption"}, res["summary"]
    # the post-render pass is a second, separate result: the timeline findings were not replaced
    assert p.qc_runs[-1]["number"] == 2 and len(p.qc_issues) >= 1
    # ---- 8. a report that says what happened
    text = ws.qc.report()
    assert "Run 2" in text or "run 2" in text.lower() or "#2" in text
    # ---- 9. save, close, reopen: results, history, fixes and the post-render report are intact
    snap = (p.qc_scores.overall, len(p.qc_runs), len(p.qc_history), len(p.qc_fixes), p.qc_runs[-1]["run_id"], res["status"])
    ws.save()
    root = p.root
    ws.close_project()
    ws.open_project(root)
    q = ws.project
    assert (q.qc_scores.overall, len(q.qc_runs), len(q.qc_history), len(q.qc_fixes), q.qc_runs[-1]["run_id"], q.render_qc_results[rid]["status"]) == snap
    assert ws.qc.export_gate().run_current  # the stored run still matches the project
    assert ws.qc.compare_runs(1, 2).resolved_issues
    run3 = run_qc(ws)
    assert run3["number"] == 3 and run3["cache_hits"] >= 10  # unchanged content: most checkers come from the cache


def test_a_broken_project_blocks_the_export_until_it_is_fixed(ws):
    """Critical findings stop the export, the reason is listed, and only a real fix lifts the block."""
    from app.services.render_service import QCGateBlocked

    p = ws.project
    music_clip(ws).audio["volume"] = 0.05  # a sensible music level: the missing file is the only problem
    asset = ws.demo.assets["main.mp4"]
    path = p.asset_path(asset)
    saved = path.read_bytes()
    path.unlink()
    run_qc(ws)
    assert p.qc_scores.counts["CRITICAL"] >= 1 and p.qc_scores.status == "BLOCKED" and p.qc_scores.export == "BLOCKED"
    missing = [i for i in ws.qc.issues() if i.code == "asset.missing"]
    assert missing and missing[0].scene_id is None or missing[0].affected_elements
    with pytest.raises(QCGateBlocked):
        ws.render.start_export()
    path.write_bytes(saved)  # the file is back
    run_qc(ws)
    assert p.qc_scores.counts["CRITICAL"] == 0 and not ws.qc.export_gate().decision.blocked
