"""Phase 6 through the real main window: Export screen -> preflight -> export -> progress -> completion -> failure, cancel, proxies, preview, relink."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import make_video
from app.tests.render_helpers import build_demo, install_fake_ffmpeg

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from app.rendering.models import RenderStatus  # noqa: E402
from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)

pytestmark = needs_ffmpeg


@pytest.fixture
def export_win(win, tmp_path, monkeypatch):
    opened = []
    from PySide6.QtGui import QDesktopServices

    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toLocalFile()) or True))
    win.opened = opened
    project = create_project(win, tmp_path, "UiExport")
    win.demo = build_demo(win.ws, tmp_path, big_video=(3840, 2160), wait=lambda: pump(lambda: len(win.ws.project.assets.all()) >= 7 and win.ws.jobs.wait_idle(0), 120))
    win.ws.render.update_settings(resolution="480p", quality="draft", preset_id="draft")
    win.go_to("Export")
    panel = win.export_panel
    pump(lambda: panel.start_btn.isEnabled() and "✓" in panel.preflight_view.text())
    win.project = project
    return win


def finish(win, job, timeout=120):
    pump(lambda: job.finished_event.is_set(), timeout)


def start(win):
    panel = win.export_panel
    n = len(win.ws.render.jobs())
    QTest.mouseClick(panel.start_btn, Qt.MouseButton.LeftButton)
    pump(lambda: len(win.ws.render.jobs()) > n)
    job = win.ws.render.jobs()[-1]
    return job


def test_export_screen_shows_settings_preflight_and_completes_an_export(export_win):
    win = export_win
    panel, ws = win.export_panel, win.ws
    assert panel.resolution.currentData() == "480p" and panel.codec.currentData() == "h264" and panel.audio.currentData() == "aac" and panel.hardware.currentData() == "auto"
    assert [panel.quality.itemData(i) for i in range(panel.quality.count())] == ["draft", "standard", "high", "maximum", "custom"]
    assert [panel.fps.itemData(i) for i in range(panel.fps.count())] == [0, 24, 30, 60] and "Project FPS" in panel.fps.itemText(0)
    assert [panel.preset.itemText(i) for i in range(panel.preset.count())] == ["YouTube 1080p", "YouTube 4K", "High Quality", "Draft", "Custom"]
    text = panel.preflight_view.text()
    for line in ("FFmpeg", "Timeline valid", "Voice-over", "visuals found", "Captions valid", "Fonts available", "Audio valid", "Output settings valid", "Output location valid", "Disk space"):
        assert line in text, line
    assert not panel.fix_btn.isVisible() and not panel.result_card.isVisible()
    # choose the high-level preset: widgets and the saved project settings follow
    panel.preset.setCurrentIndex(panel.preset.findData("youtube_1080p"))
    panel.preset.activated.emit(panel.preset.currentIndex())
    assert panel.resolution.currentData() == "1080p" and panel.quality.currentData() == "high" and ws.render.settings.preset_id == "youtube_1080p"
    # a hand edit makes it Custom and is saved with the project
    panel.resolution.setCurrentIndex(panel.resolution.findData("480p"))
    panel.quality.setCurrentIndex(panel.quality.findData("draft"))
    assert panel.preset.currentData() == "custom" and ws.render.settings.resolution == "480p" and ws.project.render_settings.quality == "draft"
    pump(lambda: panel.start_btn.isEnabled())
    # START EXPORT -> queue row, real progress, completion card
    job = start(win)
    assert panel.queue_table.rowCount() == 1 and panel.tabs.currentIndex() == 0
    finish(win, job)
    pump(lambda: panel.result_card.isVisible())
    assert job.status is RenderStatus.COMPLETED
    info = panel.result_info.text()
    assert "UiExport_480p_30fps.mp4" in info and "00:08" in info and "854×480" in info and "FPS:\n30" in info
    assert panel.queue_table.item(0, 1).text() == "Completed" and panel.queue_table.cellWidget(0, 3).value() == 1000
    assert "Completed" in panel.detail.toPlainText()
    panel.open_video_btn.click()
    panel.open_folder_btn.click()
    assert win.opened and win.opened[0].endswith("UiExport_480p_30fps.mp4")
    assert panel.history_table.rowCount() == 1 and panel.history_table.item(0, 1).text() == "Completed" and "UiExport_480p_30fps.mp4" in panel.history_table.item(0, 2).text()
    panel.again_btn.click()
    assert not panel.result_card.isVisible()
    # the project is intact and editable
    assert ws.project.timeline.all_clips() and ws.commands.can_undo is not None


def test_progress_detail_follows_the_real_ffmpeg_data(export_win):
    win = export_win
    panel = win.export_panel
    win.ws.render.update_settings(resolution="1080p", quality="high", preset_id="custom")
    panel.refresh()
    pump(lambda: panel.start_btn.isEnabled())
    job = start(win)
    pump(lambda: job.progress.overall > 0.25 or job.status.is_terminal)
    panel._show_detail()
    d = panel.detail.toPlainText()
    if job.status is RenderStatus.RUNNING:
        assert "Video composition:" in d and "Audio mix:" in d and "Encoding:" in d and "Speed:" in d and "ETA:" in d and "Scene" not in d.split("\n")[0]
        assert panel.queue_table.item(0, 2).text() in ("Rendering", "Encoding") and 0 < panel.queue_table.cellWidget(0, 3).value() < 1000
    finish(win, job)
    assert job.status is RenderStatus.COMPLETED


def test_missing_media_blocks_start_and_the_relink_dialog_fixes_it(export_win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from app.ui.dialogs.relink_dialog import RelinkDialog

    win = export_win
    panel, ws = win.export_panel, win.ws
    asset = win.demo.assets["broll.mp4"]
    path = ws.project.asset_path(asset)
    moved = tmp_path / "moved" / "broll.mp4"
    moved.parent.mkdir()
    path.rename(moved)
    panel.run_preflight()
    pump(lambda: "✗" in panel.preflight_view.text())
    assert not panel.start_btn.isEnabled() and panel.fix_btn.isVisible() and "missing" in panel.preflight_view.text() and "2/3 visuals found" in panel.preflight_view.text()
    # Start is blocked even if clicked programmatically
    QTest.mouseClick(panel.start_btn, Qt.MouseButton.LeftButton)
    assert not ws.render.jobs()
    # Fix Issues -> relink dialog: locate the file
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(moved), "")))
    monkeypatch.setattr(RelinkDialog, "exec", lambda self: (self.locate(), 1)[1])
    panel.fix_btn.click()
    pump(lambda: panel.start_btn.isEnabled())
    assert ws.project.asset_path(asset) == moved and not panel.fix_btn.isVisible() and "3/3 visuals found" in panel.preflight_view.text()
    job = start(win)
    finish(win, job)
    assert job.status is RenderStatus.COMPLETED


def test_original_missing_with_a_proxy_offers_locate_use_proxy_or_cancel(export_win, tmp_path, monkeypatch):
    win = export_win
    panel, ws = win.export_panel, win.ws
    big = win.demo.assets["big.mp4"]
    from app.tests.render_helpers import media_clip, put

    put(ws, "track_v2", media_clip(big, 5.0, 2.0, scale=0.3))
    ws.render.proxies.generate([big.id])
    pump(lambda: ws.jobs.wait_idle(0) and ws.render.proxies.record(big.id).proxy_status == "READY", 120)
    path = ws.project.asset_path(big)
    path.rename(tmp_path / "gone.mp4")
    panel.run_preflight()
    pump(lambda: panel.fix_btn.isVisible())
    assert "proxy available" in panel.preflight_view.text()
    seen = {}

    def fake_exec(self):
        seen["text"] = self.text()
        seen["buttons"] = [b.text() for b in self.buttons()]

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(b for b in self.buttons() if b.text() == "Cancel Render"))
    panel.fix_btn.click()
    assert "Original media missing." in seen["text"] and "big.mp4" in seen["text"] and "Proxy available." in seen["text"]
    assert sorted(seen["buttons"]) == ["Cancel Render", "Locate Original", "Use Proxy Anyway"]
    assert not panel._allow_proxy  # cancelling changes nothing: the proxy is not used behind the user's back
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(b for b in self.buttons() if b.text() == "Use Proxy Anyway"))
    panel.fix_btn.click()
    pump(lambda: panel.start_btn.isEnabled())
    assert panel._allow_proxy == {big.id}
    job = start(win)
    finish(win, job)
    assert job.status is RenderStatus.COMPLETED and job.result.used_proxy_assets == [big.id]


def test_cancelling_from_the_queue_shows_canceled_and_keeps_the_project(export_win):
    win = export_win
    panel, ws = win.export_panel, win.ws
    ws.render.update_settings(resolution="2160p", quality="draft", preset_id="custom")
    panel.refresh()
    pump(lambda: panel.start_btn.isEnabled())
    doc = ws.project.to_document()["timeline"]
    job = start(win)
    pump(lambda: job.progress.overall >= 0.2, 120)
    panel.queue_table.selectRow(0)
    assert panel.cancel_btn.isEnabled()
    QTest.mouseClick(panel.cancel_btn, Qt.MouseButton.LeftButton)
    finish(win, job, 60)
    pump(lambda: panel.queue_table.item(0, 1).text() == "Canceled")
    assert job.status is RenderStatus.CANCELED and "Canceled" in panel.detail.toPlainText() and not job.output_path.exists() and ws.project.to_document()["timeline"] == doc
    assert not panel.result_card.isVisible() and not panel.fail_card.isVisible()
    panel.queue_table.selectRow(0)
    assert panel.retry_sel_btn.isEnabled() and panel.remove_btn.isEnabled() and not panel.cancel_btn.isEnabled()


def test_a_failed_render_shows_stage_issue_and_retry_options(export_win, tmp_path):
    win = export_win
    panel, ws = win.export_panel, win.ws
    ctl = install_fake_ffmpeg(ws, tmp_path, "fail_chunk:0.72")
    panel.refresh()
    pump(lambda: panel.start_btn.isEnabled())
    job = start(win)
    finish(win, job)
    pump(lambda: panel.fail_card.isVisible())
    t = panel.fail_info.text()
    assert job.status is RenderStatus.FAILED and "Stage:\nRendering" in t and "Possible issue:" in t and panel.queue_table.item(0, 1).text() == "FAILED"
    assert panel.retry_btn.isVisible() and panel.logs_btn.isVisible() and panel.settings_btn.isVisible() and panel.dismiss_btn.isVisible() and not panel.cpu_btn.isVisible()
    panel.logs_btn.click()
    assert win.opened and win.opened[-1].endswith("render.log") and "Conversion failed" in Path(win.opened[-1]).read_text()
    ctl.write_text("ok")
    panel.retry_btn.click()
    job2 = ws.render.jobs()[-1]
    assert job2.id != job.id and not panel.fail_card.isVisible()
    finish(win, job2)
    pump(lambda: panel.result_card.isVisible())
    assert job2.status is RenderStatus.COMPLETED


def test_proxies_tab_generates_and_deletes_proxies_and_preview_tab_renders_a_cached_preview(export_win):
    win = export_win
    panel, ws = win.export_panel, win.ws
    from app.tests.render_helpers import media_clip, put

    put(ws, "track_v2", media_clip(win.demo.assets["big.mp4"], 5.0, 2.0, scale=0.3))  # the 4K clip is part of the edit
    panel.tabs.setCurrentIndex(2)
    pump(lambda: panel.proxy_table.rowCount() >= 3)
    names = [panel.proxy_table.item(r, 0).text() for r in range(panel.proxy_table.rowCount())]
    assert "big.mp4" in names and panel.proxy_res.currentText() == "720p"
    QTest.mouseClick(panel.proxy_gen, Qt.MouseButton.LeftButton)
    pump(lambda: ws.jobs.wait_idle(0) and any(panel.proxy_table.item(r, 2).text() == "Ready" for r in range(panel.proxy_table.rowCount())), 120)
    row = names.index("big.mp4")
    assert panel.proxy_table.item(row, 2).text() == "Ready" and panel.proxy_table.item(row, 3).text() == "720p" and "1 proxy file(s) ready" in panel.proxy_note.text()
    # preview tab: render, then the cache is reported
    panel.tabs.setCurrentIndex(3)
    panel.preview_mode.setCurrentIndex(panel.preview_mode.findData("realtime"))
    QTest.mouseClick(panel.preview_btn, Qt.MouseButton.LeftButton)
    pump(lambda: panel.preview_open.isEnabled(), 120)
    assert "Preview ready" in panel.preview_status.text() and "proxy media used" in panel.preview_status.text()
    panel.check_preview_cache()
    assert "1 of 1 preview section(s) are cached; 0 would be rendered" in panel.preview_status.text()
    panel.preview_open.click()
    assert win.opened[-1].endswith(".mp4")
    # delete proxies: files and records go, originals stay
    panel.tabs.setCurrentIndex(2)
    QTest.mouseClick(panel.proxy_delete, Qt.MouseButton.LeftButton)
    assert ws.render.proxies.record(win.demo.assets["big.mp4"].id) is None and ws.project.asset_path(win.demo.assets["big.mp4"]).is_file()
    assert panel.proxy_table.item(row, 2).text() == "None"


def test_export_settings_survive_closing_and_reopening_the_project(export_win):
    win = export_win
    panel, ws = win.export_panel, win.ws
    panel.fps.setCurrentIndex(panel.fps.findData(24))
    panel.codec.setCurrentIndex(panel.codec.findData("h264"))
    panel.hardware.setCurrentIndex(panel.hardware.findData("cpu"))
    panel.advanced.setChecked(True)
    panel.crf.setValue(20)
    panel.quality.setCurrentIndex(panel.quality.findData("custom"))
    root = ws.project.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    win.go_to("Export")
    panel.refresh()
    assert panel.fps.currentData() == 24 and panel.hardware.currentData() == "cpu" and panel.quality.currentData() == "custom" and panel.crf.value() == 20
    pump(lambda: panel.start_btn.isEnabled())


def test_unavailable_codec_is_explained_not_hidden(export_win, monkeypatch):
    win = export_win
    panel, ws = win.export_panel, win.ws
    caps = ws.render.engine.ffmpeg.capabilities()
    monkeypatch.setattr(caps, "encoders", {e for e in caps.encoders if e != "libx265"})
    panel.refresh()
    i = panel.codec.findData("h265")
    assert "not available here" in panel.codec.itemText(i)
    panel.codec.setCurrentIndex(i)
    assert "H.265" in panel.notice.text() and "not available" in panel.notice.text() and "H.264" in panel.notice.text()
    pump(lambda: not panel.start_btn.isEnabled() and "✗" in panel.preflight_view.text())
    assert ws.render.settings.video_codec == "h265"  # nothing was changed silently
    _ = time
