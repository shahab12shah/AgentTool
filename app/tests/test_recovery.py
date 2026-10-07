from __future__ import annotations

import json

import pytest

from app.core.exceptions import RecoveryError
from app.services.workspace import Workspace
from app.tests.conftest import import_and_wait, needs_ffmpeg, settle


def test_autosave_writes_snapshot_outside_project_and_not_main_file(project_ws):
    ws = project_ws
    main = ws.project.paths.project_file
    before = main.read_text()
    ws.set_script("unsaved words")  # minor command: no immediate autosave
    assert ws.project.dirty
    assert ws.autosave_tick()  # periodic autosave
    settle(ws)
    entries = ws.pending_recovery()
    assert len(entries) == 1 and entries[0].project_name == "Demo"
    assert ws.paths.recovery_dir in (ws.recovery.dir,) and not str(ws.recovery.dir).startswith(str(ws.project.root))
    assert main.read_text() == before  # main project is never touched by autosave


def test_autosave_skipped_when_clean(project_ws):
    ws = project_ws
    assert not ws.autosave_tick()
    assert ws.pending_recovery() == []


def test_major_operation_triggers_immediate_autosave(project_ws):
    ws = project_ws
    ws.timeline.add_track()
    settle(ws)
    assert len(ws.pending_recovery()) == 1


def test_save_and_clean_close_remove_recovery_data(project_ws):
    ws = project_ws
    ws.timeline.add_track()
    settle(ws)
    assert ws.pending_recovery()
    ws.save()
    assert ws.pending_recovery() == [] and not ws.project.dirty
    ws.timeline.add_track()
    settle(ws)
    assert ws.pending_recovery()
    ws.close_project()
    assert ws.pending_recovery() == []


def test_autosave_does_not_resurrect_snapshot_after_save(project_ws):
    ws = project_ws
    for _ in range(20):
        ws.timeline.add_track()
    ws.save()
    settle(ws)
    assert ws.pending_recovery() == []


def test_crash_then_recover_restores_unsaved_state(app_paths, tmp_path):
    ws1 = Workspace(app_paths)
    ws1.new_project("Crashy", tmp_path / "p")
    ws1.set_script("saved text")
    ws1.save()
    ws1.set_script("text written after the last save")
    t = ws1.timeline.add_track()
    settle(ws1)
    # --- simulated crash: no shutdown(), no close_project(); process just disappears ---
    ws2 = Workspace(app_paths)
    entries = ws2.pending_recovery()
    assert [e.project_name for e in entries] == ["Crashy"]
    assert entries[0].saved_at_local  # HH:MM:SS shown in the recovery prompt
    project = ws2.recover(entries[0].project_id)
    assert project.script.text == "text written after the last save"
    assert t.id in [tr.id for tr in project.timeline.tracks]
    assert project.dirty  # not silently saved
    on_disk = json.loads((tmp_path / "p" / "Crashy" / "project.json").read_text())
    assert on_disk["script"]["text"] == "saved text"  # main file untouched until the user saves
    ws2.save()
    assert ws2.pending_recovery() == []
    on_disk = json.loads((tmp_path / "p" / "Crashy" / "project.json").read_text())
    assert on_disk["script"]["text"] == "text written after the last save"
    ws2.shutdown()
    ws1.jobs.shutdown()
    ws1.autosave.shutdown()


def test_discard_recovery_keeps_main_project(app_paths, tmp_path):
    ws1 = Workspace(app_paths)
    ws1.new_project("Keep", tmp_path / "p")
    ws1.set_script("draft")
    ws1.autosave_tick()
    settle(ws1)
    ws2 = Workspace(app_paths)
    pid = ws2.pending_recovery()[0].project_id
    ws2.discard_recovery(pid)
    assert ws2.pending_recovery() == []
    assert ws2.open_project(tmp_path / "p" / "Keep").script.text == ""
    ws1.jobs.shutdown(), ws1.autosave.shutdown(), ws2.shutdown()


def test_corrupt_recovery_data_is_discarded_not_fatal(ws):
    bad = ws.recovery.dir / "proj_bad"
    bad.mkdir(parents=True)
    (bad / "recovery.json").write_text("{ nope")
    assert ws.pending_recovery() == []
    assert not bad.exists()


def test_recover_when_project_folder_is_gone(app_paths, tmp_path):
    import shutil

    ws1 = Workspace(app_paths)
    ws1.new_project("Gone", tmp_path / "p")
    ws1.timeline.add_track()
    settle(ws1)
    shutil.rmtree(tmp_path / "p")
    ws2 = Workspace(app_paths)
    with pytest.raises(RecoveryError):
        ws2.recover(ws2.pending_recovery()[0].project_id)
    ws1.jobs.shutdown(), ws1.autosave.shutdown(), ws2.shutdown()


@needs_ffmpeg
def test_crash_with_background_job_running(app_paths, tmp_path, media_dir):
    """Acceptance: start a background job, 'crash', restart, recover."""
    import threading

    ws1 = Workspace(app_paths)
    ws1.new_project("JobCrash", tmp_path / "p")
    asset = import_and_wait(ws1, media_dir / "clip.mp4")
    ws1.timeline.add_asset(asset.id)
    ws1.set_script("work in progress")
    gate = threading.Event()
    job = ws1.jobs.submit("long", lambda ctx: gate.wait(30), title="Long running job")
    ws1.autosave_tick()  # the periodic timer fires before the crash
    settle_autosave = ws1.autosave.wait_idle(10)
    assert settle_autosave and job.status.value in ("QUEUED", "RUNNING")
    # crash: abandon ws1 without any clean-up while the job is still running
    ws2 = Workspace(app_paths)
    entry = ws2.pending_recovery()[0]
    project = ws2.recover(entry.project_id)
    assert len(project.timeline.all_clips()) == 1 and project.script.text == "work in progress"
    assert project.assets.get(asset.id) is not None and project.asset_path(asset).is_file()
    gate.set()
    ws1.jobs.shutdown(), ws1.autosave.shutdown(), ws2.shutdown()
