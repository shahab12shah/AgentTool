"""Incremental QC through a real Workspace on a synthetic project with tiny real media: the result after an incremental run is identical (findings, scores, export gate) to a full
run from scratch after each kind of edit; staleness is reported per scene / domain without hashing; nothing outdated is ever shown as current."""

from __future__ import annotations

import copy
import shutil
import subprocess
from dataclasses import replace

import pytest

from app.core.commands import CompositeCommand
from app.media.asset import AssetType
from app.performance import change_tracker as ct
from app.performance.synthetic import SyntheticSpec, build_project
from app.presentation.assembly import PresState
from app.project.phase3_commands import SceneDecisionCommand
from app.project.phase5_commands import ApplyPresentationCommand
from app.project.project_commands import SetVoiceOverCommand
from app.qc.qc_engine import CHECKER_CATEGORIES
from app.rendering.commands import SetRenderSettingsCommand
from app.research.models import VisualAssignment
from app.services.workspace import Workspace
from app.storage.paths import AppPaths
from app.tests.conftest import HAS_FFMPEG, needs_ffmpeg
from app.timeline.clip import Clip
from app.timeline.timeline import new_clip_id
from app.timeline.timeline_commands import AddClipCommand, DeleteClipCommand

SCENES = 14
SCENE_LOCAL = ("scene", "sync", "visual", "caption", "text", "motion")


def sid(i: int) -> str:
    return f"scene_{i:04d}"


def _ff(*a: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *a], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("incr_media")
    if HAS_FFMPEG:
        _ff("-f", "lavfi", "-i", "testsrc=duration=3:size=320x180:rate=30", "-pix_fmt", "yuv420p", str(d / "clip.mp4"))
        _ff("-f", "lavfi", "-i", "color=c=red:size=64x64", "-frames:v", "1", str(d / "pic.png"))
        _ff("-f", "lavfi", "-i", "sine=frequency=220:duration=4", "-ar", "48000", "-ac", "2", str(d / "voice.wav"))
    return d


def make_workspace(tmp_path, media, scenes: int = SCENES) -> Workspace:
    ws = Workspace(AppPaths(tmp_path / "cfg", tmp_path / "data"))
    ws.new_project("Incremental", tmp_path / "projects")
    sp = build_project(tmp_path / "syn", SyntheticSpec(scenes=scenes, keyframes_per_visual=2), real_files=True)
    for a in sp.project.assets.all():  # tiny real files so the media-reading checkers run too
        shutil.copyfile(media / {AssetType.VIDEO: "clip.mp4", AssetType.IMAGE: "pic.png", AssetType.AUDIO: "voice.wav"}[a.type], sp.project.root / a.path)
    ws.projects.switch_to(sp.project)
    return ws


def run(ws: Workspace, incremental: bool = False, force: bool = False):
    job = ws.qc.run_incremental_qc() if incremental else ws.qc.run_full_qc(force=force)
    if job is not None:
        assert ws.jobs.wait_idle(240)
        assert job.error is None, job.error
    return job


def outcome(ws: Workspace) -> dict:
    p = ws.project
    gate = ws.qc.export_gate()
    return {
        "issues": sorted((i.checker, i.code, i.fingerprint, i.severity.value, i.scene_id or "", round(i.confidence, 1), i.title, i.description, i.start_time, i.end_time, i.status.value) for i in p.qc_issues),
        "scores": p.qc_scores.to_dict(),
        "gate": (gate.decision.status, gate.decision.blocked, gate.decision.critical, sorted(gate.decision.blocking_ids) and len(gate.decision.blocking_ids)),
        "state": p.qc_runs[-1]["state"],
        "failed": p.qc_runs[-1].get("failed"),
    }


def visual_clip(p, i: int) -> Clip:
    return next(c for t in p.timeline.tracks for c in t.clips if c.scene_id == sid(i) and c.kind == "media")


# ---------------------------------------------------------------------------------------------------------- the edits
def edit_caption_text(ws):
    p = ws.project
    after = PresState.capture(p)
    cap = next(c for t in after.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(6))
    cap.text = {**cap.text, "text": "A caption the user retyped completely differently"}
    ws.commands.execute(ApplyPresentationCommand(p, after, "Edit caption"))
    return {"direct": {sid(6)}, "all": False}


def edit_audio_volume(ws):
    ws.timeline.set_track_volume("track_a2", 0.45)
    return {"all_scene_ids": True, "all": False, "domains": {"audio"}}


def edit_visual_replacement(ws):
    p = ws.project
    old = visual_clip(p, 9)
    other = next(a for a in p.assets.all() if a.type is AssetType.VIDEO and a.id != old.asset_id)
    new = Clip(new_clip_id(), old.track_id, other.id, old.timeline_start, old.duration, source_in=0.0, source_out=old.duration, scene_id=old.scene_id, created_by="USER")
    ws.commands.execute(CompositeCommand("Replace visual", [DeleteClipCommand(p.timeline, old.id), AddClipCommand(p.timeline, new),
                                                             SceneDecisionCommand(p, sid(9), "Replace", assignment=VisualAssignment(sid(9), asset_id=other.id, selected_by="USER"))], scope="timeline"))
    return {"direct": {sid(9)}, "all": False}


def edit_clip_move(ws):
    p = ws.project
    c = next(c for t in p.timeline.tracks for c in t.clips if c.scene_id == sid(5) and c.kind == "text")
    ws.timeline.move_clip(c.id, c.timeline_start + 0.3)
    return {"direct": {sid(5)}, "all": False}


def edit_export_setting(ws):
    ws.commands.execute(SetRenderSettingsCommand(ws.project, replace(ws.project.render_settings, resolution="720p", fps=24)))
    return {"all": True, "domains": {"render"}}


def edit_voice_over(ws):
    p = ws.project
    music = next(a for a in p.assets.all() if a.type is AssetType.AUDIO and a.id != p.voice_over.asset_id)
    ws.commands.execute(SetVoiceOverCommand(p, music))
    return {"all": True, "domains": {"audio", "transcript"}}


EDITS = [edit_caption_text, edit_audio_volume, edit_visual_replacement, edit_clip_move, edit_export_setting, edit_voice_over]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory, media):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    ws = make_workspace(tmp_path_factory.mktemp("incr"), media)
    run(ws)
    yield ws
    ws.shutdown()


@needs_ffmpeg
def test_a_complete_run_records_a_compact_baseline_and_nothing_is_stale(baseline):
    ws = baseline
    p = ws.project
    b = p.qc_cache["_incremental"]
    assert set(b["domains"]) >= {"timeline", "scenes", "transcript", "assets", "audio", "captions", "visual", "reference", "render"} and len(b["scenes"]) == SCENES
    assert all(isinstance(v, str) and len(v) <= 14 for v in b["scenes"].values()) and b["epoch"] == ws.changes.epoch
    assert ws.qc.qc_staleness() == {"stale": False, "scene_ids": [], "domains": [], "reason": "", "source": "tracker", "unknown_change": False}
    assert ws.qc.plan_incremental().mode == "none" and ws.qc.run_incremental_qc() is None  # nothing to do: no new run
    assert ws.qc.export_gate().run_current


@needs_ffmpeg
@pytest.mark.parametrize("edit", EDITS, ids=[e.__name__[5:] for e in EDITS])
def test_incremental_equals_a_full_run_after_each_kind_of_edit(baseline, edit, monkeypatch):
    ws = baseline
    p = ws.project
    assert ws.qc.qc_staleness()["stale"] is False
    want = edit(ws)
    order = [s.id for s in sorted(p.scenes, key=lambda s: s.start)]

    # 1. staleness is known immediately, from the change log (no hashing)
    st = ws.qc.qc_staleness()
    assert st["stale"] and st["source"] == "tracker" and not st["unknown_change"]
    if "direct" in want:
        assert set(want["direct"]) <= set(st["scene_ids"]) and len(st["scene_ids"]) < len(order)  # bounded: neighbours, not the whole project
    if want.get("all_scene_ids"):
        assert st["scene_ids"] == order
    assert set(want.get("domains", ())) <= set(st["domains"])
    assert ws.qc.export_gate().needs_run and not ws.qc.export_gate().run_current  # the content-hash gate agrees

    # 2. the plan
    plan = ws.qc.plan_incremental()
    assert plan.mode == "incremental" and plan.source == "tracker"
    assert plan.all_scenes == want["all"] or want["all"] is False
    if "direct" in want:
        assert set(want["direct"]) <= set(plan.direct_scene_ids) and set(plan.scene_ids) == set(st["scene_ids"])

    events: list[dict] = []
    import app.qc.qc_service as svc

    real = svc.log_event
    monkeypatch.setattr(svc, "log_event", lambda lg, name, **kw: (events.append({"event": name, **kw}), real(lg, name, **kw))[1])

    # 3. incremental run
    assert run(ws, incremental=True) is not None
    inc = outcome(ws)
    rec = ws.project.qc_runs[-1]
    assert rec["trigger"] == "incremental" and rec["content_hash"], "an incremental run must be certified current"
    gate = ws.qc.export_gate()
    assert gate.run_current and not gate.needs_run
    assert ws.qc.qc_staleness()["stale"] is False
    selected = next(e for e in events if e["event"] == "qc.incremental_selected")
    assert selected["mode"] == "incremental" and selected["scenes"] == len(plan.scene_ids) and selected["total_scenes"] == SCENES

    # work is bounded for the scene-local checkers when only some scenes changed
    counts = rec["scope"]["incremental"]["scenes"]
    assert set(counts) <= set(SCENE_LOCAL) | {"caption", "text", "motion", "sync", "scene", "visual"}
    if not want["all"] and "direct" in want:
        for cid, (analysed, _reused) in counts.items():
            assert analysed <= len(plan.scene_ids), (cid, analysed, len(plan.scene_ids))
    if not want["all"] and not want.get("all_scene_ids"):  # a clip edit changes the timeline domain, so the whole-project checkers run again; the scene-local ones re-use the scenes it did not touch
        assert counts and sum(r for _a, r in counts.values()) > 0, "unchanged scenes must have been re-used"

    # 4. a full run from scratch says exactly the same
    run(ws, force=True)
    full = outcome(ws)
    assert inc["issues"] == full["issues"]
    assert inc["scores"] == full["scores"]
    assert inc["gate"] == full["gate"] and inc["state"] == full["state"] and inc["failed"] == full["failed"]


@needs_ffmpeg
def test_staleness_never_vouches_for_an_unseen_change(tmp_path, media):
    ws = make_workspace(tmp_path, media, scenes=6)
    try:
        run(ws)
        assert ws.qc.qc_staleness()["stale"] is False
        # a change that bypassed the command stack (and so the tracker): the content-hash gate and the exact comparison still catch it
        cap = next(c for t in ws.project.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(3))
        cap.text = {**cap.text, "text": "sneaky"}
        gate = ws.qc.export_gate()
        assert gate.needs_run and not gate.run_current
        exact = ws.qc.qc_staleness(exact=True)
        assert exact["stale"] and exact["source"] == "hash" and sid(3) in exact["scene_ids"] and "timeline" in exact["domains"]
        plan = ws.qc.plan_incremental()
        assert plan.mode == "incremental" and plan.source == "fingerprints" and sid(3) in plan.direct_scene_ids  # the tracker saw nothing: fingerprints found it
        assert run(ws, incremental=True) is not None
        inc = outcome(ws)
        assert ws.qc.export_gate().run_current
        run(ws, force=True)
        assert outcome(ws)["issues"] == inc["issues"]
    finally:
        ws.shutdown()


@needs_ffmpeg
def test_without_a_baseline_the_run_is_a_full_one_and_still_correct(tmp_path, media):
    ws = make_workspace(tmp_path, media, scenes=6)
    try:
        assert ws.qc.plan_incremental().mode == "full"  # QC never ran
        job = run(ws, incremental=True)
        assert job is not None and ws.project.qc_runs[-1]["trigger"] == "incremental" and ws.project.qc_runs[-1]["content_hash"]
        del ws.project.qc_cache["_incremental"]  # a project saved by an older version / a baseline that is gone
        cap = next(c for t in ws.project.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(2))
        ws.timeline.trim_clip(cap.id, new_end=cap.timeline_end - 0.2)
        plan = ws.qc.plan_incremental()
        assert plan.mode == "full" and plan.source == "none"  # nothing to compare with: every check runs (unchanged analysis is still re-used inside the engine)
        assert run(ws, incremental=True) is not None
        inc = outcome(ws)
        run(ws, force=True)
        assert inc["issues"] == outcome(ws)["issues"] and inc["scores"] == outcome(ws)["scores"]
    finally:
        ws.shutdown()


@needs_ffmpeg
def test_a_restarted_app_has_no_tracker_history_but_the_stored_fingerprints_still_decide(tmp_path, media):
    ws = make_workspace(tmp_path, media, scenes=6)
    try:
        run(ws)
        cap = next(c for t in ws.project.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(5))
        ws.timeline.trim_clip(cap.id, new_end=cap.timeline_end - 0.2)
        fresh = ct.ProjectChangeTracker(ws.bus, lambda: ws.projects.current)  # a new session: epoch and revision mean nothing for the stored baseline
        fresh._epoch += 7
        old, ws.qc.tracker = ws.qc.tracker, fresh
        try:
            st = ws.qc.qc_staleness()
            assert st["stale"] and st["source"] == "hash" and sid(5) in st["scene_ids"] and len(st["scene_ids"]) < 6
            plan = ws.qc.plan_incremental()
            assert plan.source == "fingerprints" and sid(5) in plan.direct_scene_ids and len(plan.scene_ids) < 6 and plan.mode == "incremental"  # scene 4 looks at the neighbouring caption too
        finally:
            ws.qc.tracker = old
        assert run(ws, incremental=True) is not None
        inc = outcome(ws)
        run(ws, force=True)
        assert outcome(ws)["issues"] == inc["issues"]
    finally:
        ws.shutdown()


@needs_ffmpeg
def test_an_edit_made_while_the_run_is_running_stays_pending(tmp_path, media):
    ws = make_workspace(tmp_path, media, scenes=6)
    try:
        run(ws)
        cap = next(c for t in ws.project.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(2))
        ws.timeline.trim_clip(cap.id, new_end=cap.timeline_end - 0.2)
        job = ws.qc.run_incremental_qc()
        other = next(c for t in ws.project.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(5))
        ws.timeline.trim_clip(other.id, new_end=other.timeline_end - 0.2)  # lands after the snapshot was taken
        assert ws.jobs.wait_idle(240) and job.error is None
        st = ws.qc.qc_staleness()
        assert st["stale"] and sid(5) in st["scene_ids"] and sid(2) not in st["scene_ids"]  # scene 2 was covered by the run, scene 5 was not
        assert ws.qc.export_gate().needs_run  # the hash gate agrees: the run does not vouch for the later edit
        assert run(ws, incremental=True) is not None and ws.qc.export_gate().run_current
    finally:
        ws.shutdown()


@needs_ffmpeg
def test_qc_settings_that_change_the_analysis_make_results_stale_but_presentation_ones_do_not(tmp_path, media):
    ws = make_workspace(tmp_path, media, scenes=6)
    try:
        run(ws)
        ws.qc.update_settings(block_level="CRITICAL_ERROR_WARNING")  # how findings are acted on, not what is found
        assert ws.qc.qc_staleness()["stale"] is False and ws.qc.plan_incremental().mode == "none"
        ws.qc.update_settings(**{"sync.major_ms": ws.project.qc_settings.sync.major_ms + 50})
        st = ws.qc.qc_staleness()
        assert st["stale"] and "qc_settings" in st["domains"]
        assert run(ws, incremental=True) is not None
        inc = outcome(ws)
        run(ws, force=True)
        assert outcome(ws)["issues"] == inc["issues"] and outcome(ws)["scores"] == inc["scores"]
    finally:
        ws.shutdown()


def test_category_table_is_untouched_by_the_baseline_key():
    assert "_incremental" not in CHECKER_CATEGORIES and copy.deepcopy(CHECKER_CATEGORIES)
