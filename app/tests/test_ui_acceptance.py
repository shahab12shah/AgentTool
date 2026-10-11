"""Drives the real main window (offscreen Qt) through the Phase 1 acceptance workflow."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.tests.conftest import needs_ffmpeg

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main import create_window  # noqa: E402
from app.ui.timeline_canvas import RULER_H, ROW_H  # noqa: E402

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(cond, timeout: float = 20.0) -> None:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if cond():
            return
        time.sleep(0.01)  # not QTest.qWait: PySide6 6.11 keeps the GIL there, which starves Python worker threads (the Phase 7 numpy analysis then never finishes)
    raise AssertionError("condition not met in time")


def wait_ms(ms: int) -> None:
    """Process events for ``ms`` milliseconds without ``QTest.qWait`` (it can hold the GIL and starve worker threads, see ``pump``)."""
    import time

    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def settle(ws) -> None:
    pump(lambda: ws.jobs.wait_idle(0.0) and ws.autosave.wait_idle(0.0))


@pytest.fixture
def win(qapp, app_paths, monkeypatch):
    import app.ui.dialogs.message as msg
    import app.ui.context as context

    errors: list[str] = []
    monkeypatch.setattr(msg, "show_error", lambda parent, message, details=None, title="": errors.append(message))
    monkeypatch.setattr("app.ui.main_window.show_error", lambda parent, message, details=None, title="": errors.append(message))
    window, ws = create_window(app_paths)
    window.show()
    window.test_errors = errors
    yield window
    window.ws.close_project()
    window.ws.shutdown()
    window.close()
    window.deleteLater()  # a window left alive is re-polished by every later setStyleSheet, so a long module gets slower with every test
    QApplication.processEvents()


def create_project(win, tmp_path, name="Acceptance"):
    pv = win.project_view
    pv.name.setText(name)
    pv.location.setText(str(tmp_path / "projects"))
    QTest.mouseClick(pv.create_btn, Qt.MouseButton.LeftButton)
    assert win.ws.project is not None and win.ws.project.project_name == name
    return win.ws.project


def import_all(win, media_dir):
    win.library.import_paths([media_dir / "clip.mp4", media_dir / "pic.png"])
    win.voice_panel.import_path(media_dir / "voice.wav")
    ws = win.ws
    pump(lambda: len(ws.project.assets) == 3 and ws.project.voice_over.asset_id is not None)
    settle(ws)
    pump(lambda: win.library.list.count() == 3)  # the asset browser coalesces bursts of changes into one refresh (a short timer)


def clip_center(win, clip) -> QPoint:
    canvas = win.timeline_panel.canvas
    tracks = canvas.tracks()
    idx = next(i for i, t in enumerate(tracks) if t.id == clip.track_id)
    r = canvas.clip_rect(clip, idx)
    return QPoint(int(r.center().x()), int(r.center().y()))


def test_full_acceptance_workflow(win, tmp_path, media_dir, app_paths, qapp):
    ws = win.ws
    # Launch: home screen, other pages disabled until a project exists.
    assert win.project_view.currentIndex() == 0
    assert not win.nav.item(win.page_index["Timeline"]).flags() & Qt.ItemFlag.ItemIsEnabled

    # Create project (defaults 1920x1080 / 30 / 16:9).
    project = create_project(win, tmp_path)
    root = project.root
    assert (project.settings.width, project.settings.height, project.settings.fps) == (1920, 1080, 30)
    assert (root / "project.json").is_file() and (root / "media" / "audio").is_dir()
    assert win.project_view.currentIndex() == 1 and "Acceptance" in win.name_label.text()
    assert win.nav.item(win.page_index["Timeline"]).flags() & Qt.ItemFlag.ItemIsEnabled

    # Import video, image, voice-over; they appear in the library with thumbnails.
    import_all(win, media_dir)
    names = sorted(win.library.list.item(i).text().split("\n")[0] for i in range(win.library.list.count()))
    assert names == ["clip.mp4", "pic.png", "voice.wav"]
    assert win.library.list.count() == 3
    assert win.voice_panel.filename.text() == "voice.wav" and win.voice_panel.duration.text() == "0:04"
    for a in project.assets:
        assert ws.media.thumbnail_file(a) is not None, a.name

    # Script.
    win.script_panel.editor.setPlainText("This is my script.\nSecond line.")
    win.script_panel.flush()
    assert project.script.text == "This is my script.\nSecond line."
    assert "6 words" in win.script_panel.counts.text()
    assert win.script_panel.analyze.isEnabled() and "Scenes" in win.script_panel.analyze.text()  # real action, not a stub

    # Preview image and video.
    video = next(a for a in project.assets if a.type.value == "video")
    image = next(a for a in project.assets if a.type.value == "image")
    win.preview_asset(image.id)
    assert win.preview.stack.currentWidget() is win.preview.image and win.preview.image.pixmap() is not None
    win.preview_asset(video.id)
    assert win.preview.stack.currentWidget() is win.preview.video
    pump(lambda: win.preview.player.duration > 2.5)
    win.preview.player.seek(1.0)
    win.preview.player.set_volume(0.3)
    assert win.preview.transport.play_btn.isEnabled()

    # Timeline page; add media.
    win.go_to("Timeline")
    qapp.processEvents()
    assert win.library.parentWidget() is win.tl_library_slot  # shared widget moved to this page
    win._add_to_timeline(video.id)
    clip = project.timeline.all_clips()[0]
    assert clip.track_id == "track_v1" and clip.timeline_start == 0
    win._add_to_timeline(image.id)
    assert len(project.timeline.all_clips()) == 2

    # Select via mouse -> inspector shows it.
    canvas = win.timeline_panel.canvas
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=clip_center(win, clip))
    assert ws.selected_clip_id == clip.id
    qapp.processEvents()
    assert win.inspector.body.isVisible()
    assert win.inspector.duration.value() == pytest.approx(clip.duration, abs=0.01)

    # Move with the mouse (drag body 200 px to the right).
    pps = canvas.pps
    c0 = clip_center(win, clip)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=c0)
    QTest.mouseMove(canvas, c0 + QPoint(100, 0))
    QTest.mouseMove(canvas, c0 + QPoint(200, 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=c0 + QPoint(200, 0))
    moved = project.timeline.get_clip(clip.id)
    assert moved.timeline_start == pytest.approx(200 / pps, abs=SNAP_TOL)
    start_after_move = moved.timeline_start

    # Trim the end with the mouse (drag right edge 80 px left).
    idx = 0
    r = canvas.clip_rect(moved, idx)
    edge = QPoint(int(r.right()) - 2, int(r.center().y()))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=edge)
    QTest.mouseMove(canvas, edge + QPoint(-40, 0))
    QTest.mouseMove(canvas, edge + QPoint(-80, 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=edge + QPoint(-80, 0))
    trimmed = project.timeline.get_clip(clip.id)
    assert trimmed.duration == pytest.approx(clip.duration - 80 / pps, abs=SNAP_TOL)
    assert trimmed.source_out == pytest.approx(trimmed.duration)

    # Trim the start by typing in the inspector (Start is editable) -> also exercises the inspector.
    win.inspector.start.setValue(start_after_move + 0.0)

    # Undo / Redo through toolbar actions and shortcuts' actions.
    assert win.undo_action.isEnabled()
    win.undo_action.trigger()
    assert project.timeline.get_clip(clip.id).duration == pytest.approx(clip.duration)  # trim undone
    win.undo_action.trigger()
    assert project.timeline.get_clip(clip.id).timeline_start == pytest.approx(0.0)       # move undone
    win.redo_action.trigger()
    win.redo_action.trigger()
    final = project.timeline.get_clip(clip.id)
    assert (final.timeline_start, final.duration) == (trimmed.timeline_start, trimmed.duration)

    # Track controls.
    svc = ws.timeline
    t = svc.add_track()
    svc.rename_track(t.id, "My Track")
    svc.set_track_flag(t.id, "locked", True)
    assert project.timeline.get_track(t.id).locked

    # Save, close the application, reopen, open the project, everything is restored.
    snapshot = json.dumps(project.to_document()["timeline"], sort_keys=True)
    assert project.dirty
    win.save()
    assert not project.dirty and not ws.pending_recovery()
    win.close()
    ws.shutdown()

    win2, ws2 = create_window(app_paths)
    win2.show()
    assert ws2.pending_recovery() == []  # clean exit leaves no recovery data
    win2.open_project(root)
    p2 = ws2.project
    assert p2 is not None and p2.project_name == "Acceptance"
    assert json.dumps(p2.to_document()["timeline"], sort_keys=True) == snapshot
    assert p2.script.text == "This is my script.\nSecond line."
    assert p2.voice_over.filename == "voice.wav" and p2.voice_over.asset_id
    assert len(p2.assets) == 3 and win2.library.list.count() == 3
    assert win2.script_panel.editor.toPlainText() == "This is my script.\nSecond line."
    assert win2.voice_panel.filename.text() == "voice.wav"
    assert [r["name"] for r in ws2.projects.recent_projects()] == ["Acceptance"]
    ws2.close_project()
    ws2.shutdown()
    win2.close()
    assert win.test_errors == []


SNAP_TOL = 0.12  # mouse drags may snap to clip edges/playhead (8 px at 80 px/s = 0.1 s)


def test_crash_simulation_then_recovery_prompt_in_ui(win, tmp_path, media_dir, app_paths, qapp, monkeypatch):
    ws = win.ws
    project = create_project(win, tmp_path, "CrashProject")
    import_all(win, media_dir)
    video = next(a for a in project.assets if a.type.value == "video")
    win._add_to_timeline(video.id)
    win.script_panel.editor.setPlainText("unsaved script after last save")
    win.script_panel.flush()
    # A background job is running when the 'crash' happens.
    gate = threading.Event()
    job = ws.jobs.submit("long", lambda ctx: (ctx.report(34, "Generating thumbnail..."), gate.wait(30)), title="Long job")
    pump(lambda: win.job_bar.percent.text() == "34%")  # UI is fed through queued events
    assert win.job_bar.progress.isVisible() and "Generating thumbnail" in win.job_bar.message.text()
    assert win.job_bar.cancel.isVisible()
    assert ws.autosave_tick()
    pump(lambda: ws.autosave.wait_idle(0.0))
    assert project.dirty

    # --- crash: abandon the first process state completely (no close, no shutdown) ---
    from app.ui.dialogs.recovery_dialog import RecoveryDialog

    win2, ws2 = create_window(app_paths)
    win2.show()
    shown = {}

    def fake_exec(self):
        from PySide6.QtWidgets import QLabel

        shown["labels"] = [w.text() for w in self.findChildren(QLabel)]
        return RecoveryDialog.RECOVER

    monkeypatch.setattr(RecoveryDialog, "exec", fake_exec)
    win2.check_recovery()
    joined = " ".join(shown["labels"])
    assert "Project recovery available." in joined and "CrashProject" in joined and "Recover?" in joined
    p2 = ws2.project
    assert p2 is not None and p2.script.text == "unsaved script after last save" and len(p2.timeline.all_clips()) == 1
    assert p2.dirty and "•" in win2.name_label.text()
    on_disk = json.loads((p2.root / "project.json").read_text())
    assert on_disk["script"]["text"] == ""  # never silently overwritten
    gate.set()
    ws2.close_project()
    ws2.shutdown()
    win2.close()
    ws.jobs.shutdown()
    ws.autosave.shutdown()


def test_discard_recovery_via_ui(win, tmp_path, app_paths, qapp, monkeypatch):
    ws = win.ws
    create_project(win, tmp_path, "DiscardMe")
    ws.timeline.add_track()
    pump(lambda: ws.autosave.wait_idle(0.0))
    from app.ui.dialogs.recovery_dialog import RecoveryDialog

    win2, ws2 = create_window(app_paths)
    monkeypatch.setattr(RecoveryDialog, "exec", lambda self: RecoveryDialog.DISCARD)
    win2.check_recovery()
    assert ws2.project is None and ws2.pending_recovery() == []
    ws2.shutdown()
    win2.close()
    ws.jobs.shutdown()
    ws.autosave.shutdown()


def test_errors_never_crash_ui(win, tmp_path, media_dir):
    ws = win.ws
    # Creating a project with an invalid name shows a message instead of raising.
    pv = win.project_view
    pv.name.setText("   ")
    pv.location.setText(str(tmp_path / "p"))
    QTest.mouseClick(pv.create_btn, Qt.MouseButton.LeftButton)
    assert ws.project is None and win.test_errors and "name" in win.test_errors[-1].lower()
    # Opening a non-project folder is reported, not raised.
    (tmp_path / "empty").mkdir()
    win.open_project(tmp_path / "empty")
    assert ws.project is None and "not a project" in win.test_errors[-1]
    # Corrupt project file.
    create_project(win, tmp_path, "Corrupt")
    root = ws.project.root
    ws.close_project()
    (root / "project.json").write_text("{ broken")
    win.open_project(root)
    assert "corrupt" in win.test_errors[-1].lower()
    # Bad media: error is queued and shown once; project untouched.
    create_project(win, tmp_path, "BadMedia")
    win.library.import_paths([media_dir / "broken.mp4", media_dir / "notes.txt"])
    pump(lambda: len(win.test_errors) >= 4)  # one coalesced dialog for both failed imports
    assert len(ws.project.assets) == 0


def test_drag_from_library_onto_timeline(win, tmp_path, media_dir, qapp):
    from PySide6.QtCore import QMimeData, QPointF
    from PySide6.QtGui import QDropEvent
    from app.ui.media_library import ASSET_MIME

    ws = win.ws
    project = create_project(win, tmp_path, "DnD")
    import_all(win, media_dir)
    win.go_to("Timeline")
    audio = next(a for a in project.assets if a.type.value == "audio")
    canvas = win.timeline_panel.canvas
    md = QMimeData()
    md.setData(ASSET_MIME, audio.id.encode())
    idx = next(i for i, t in enumerate(canvas.tracks()) if t.id == "track_a3")
    pos = QPointF(canvas.time_to_x(2.0), RULER_H + idx * ROW_H + ROW_H / 2)
    ev = QDropEvent(pos, Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.dropEvent(ev)
    clip = project.timeline.all_clips()[0]
    assert clip.track_id == "track_a3" and clip.timeline_start == pytest.approx(2.0, abs=0.12)
    # wrong track kind (audio onto video track) is refused
    md2 = QMimeData()
    md2.setData(ASSET_MIME, audio.id.encode())
    ev2 = QDropEvent(QPointF(10, RULER_H + ROW_H / 2), Qt.DropAction.CopyAction, md2, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.dropEvent(ev2)
    assert len(project.timeline.all_clips()) == 1


def test_media_library_search_sort_remove(win, tmp_path, media_dir, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    ws = win.ws
    project = create_project(win, tmp_path, "Lib")
    import_all(win, media_dir)
    lib = win.library
    lib.search.setText("pic")
    assert lib.list.count() == 1
    lib.search.setText("")
    lib.sort.setCurrentText("Name")
    assert [lib.list.item(i).text().split("\n")[0] for i in range(3)] == ["clip.mp4", "pic.png", "voice.wav"]
    lib.sort.setCurrentText("Duration")
    assert lib.list.item(0).text().startswith("voice.wav")  # 4 s is the longest
    win.go_to("Project")
    lib.list.item(1).setSelected(True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    media_files_before = sorted(p.name for p in (project.root / "media").rglob("*") if p.is_file())
    lib.remove_selected()
    assert len(project.assets) == 2
    assert sorted(p.name for p in (project.root / "media").rglob("*") if p.is_file()) == media_files_before  # nothing deleted on disk
