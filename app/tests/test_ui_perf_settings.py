"""Performance settings dialog through the real main window (offscreen Qt): global vs project scope, disabled hardware option with its reason, cache table and cleanup, diagnostics export."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)
from app.ui.dialogs.performance_dialog import PerformanceDialog  # noqa: E402


@pytest.fixture
def pwin(win, tmp_path):  # noqa: F811
    create_project(win, tmp_path, "PerfUi")
    return win


def test_dialog_shows_global_and_project_scope_and_persists(pwin, tmp_path):
    ws = pwin.ws
    dlg = PerformanceDialog(ws, pwin)
    assert dlg.profile.currentData() == "balanced" and dlg.scope.model().item(1).isEnabled()
    dlg.profile.setCurrentIndex(dlg.profile.findData("power_saver"))
    dlg.preview.setCurrentIndex(dlg.preview.findData("draft"))
    dlg.workers.setValue(2)
    dlg.save()
    assert ws.performance.global_settings().profile == "power_saver" and ws.settings.performance["preview_quality"] == "draft"
    assert json.loads(ws.paths.settings_file.read_text())["performance"]["profile"] == "power_saver"  # persisted globally
    assert ws.performance.limits().background_workers <= 2

    dlg.scope.setCurrentIndex(1)  # this project only
    dlg.profile.setCurrentIndex(dlg.profile.findData("performance"))
    dlg.save()
    assert ws.project.performance_overrides == {"profile": "performance"}  # only the changed key
    assert ws.performance.settings().profile == "performance" and ws.performance.global_settings().profile == "power_saver"
    ws.undo()
    assert ws.project.performance_overrides == {} and ws.performance.settings().profile == "power_saver"


def test_hardware_option_is_disabled_with_a_reason_when_nothing_was_tested(pwin):
    dlg = PerformanceDialog(pwin.ws, pwin)
    item = dlg.backend.model().item(dlg.backend.findData("hardware"))
    if item.isEnabled():
        pytest.skip("this machine has a hardware encoder that passed its test")
    assert dlg.backend_note.text() and ("not been checked" in dlg.backend_note.text() or "No hardware video encoder" in dlg.backend_note.text())
    assert dlg.backend.currentData() in ("auto", "cpu")
    dlg.backend.setCurrentIndex(dlg.backend.findData("cpu"))
    dlg.save()
    assert pwin.ws.performance.settings().render_backend == "cpu"


def test_proxy_controls_follow_the_policy_and_note_states_export_quality(pwin):
    dlg = PerformanceDialog(pwin.ws, pwin)
    dlg.proxy_policy.setCurrentIndex(dlg.proxy_policy.findData("off"))
    assert not dlg.proxy_profile.isEnabled() and not dlg.bg_proxy.isEnabled()
    dlg.proxy_policy.setCurrentIndex(dlg.proxy_policy.findData("automatic"))
    assert dlg.proxy_profile.isEnabled()
    assert "never silently lowers final export quality" in dlg.note.text()


def test_cache_tab_lists_categories_and_cleans_only_rebuildable_files(pwin, monkeypatch):
    ws = pwin.ws
    cache = ws.performance.cache
    assert cache is not None
    f = cache.root / "temporary" / "x.tmp.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x" * 2048)
    cache.put("tmp1", category="temporary", path=f, data_type="scratch")
    media = ws.project.root / "media" / "keep.bin"
    media.write_bytes(b"keep")
    dlg = PerformanceDialog(ws, pwin)
    assert dlg.table.rowCount() == 7 and dlg.table.item(6, 1).text() == "1"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dlg._clean_cache()
    assert not f.exists() and media.read_bytes() == b"keep"
    assert dlg.table.item(6, 1).text() == "0" and dlg.cleanup_label.text() != "never"


def test_diagnostics_tab_and_export(pwin, tmp_path, monkeypatch):
    ws = pwin.ws
    dlg = PerformanceDialog(ws, pwin)
    assert "Performance report" in dlg.diag.toPlainText() and "Hardware" in dlg.diag.toPlainText()
    out = tmp_path / "diag.json"
    monkeypatch.setattr("app.ui.dialogs.performance_dialog.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), "JSON"))
    dlg._export()
    data = json.loads(out.read_text())
    assert "bottlenecks" in data and "limits" in data and "hardware" in data
    assert "api_key" not in out.read_text().lower() or "***" in out.read_text()
    job = ws.performance.detect_hardware_async(force=True)
    assert job is not None
    pump(lambda: ws.jobs.wait_idle(0.0), 120)
    assert ws.performance.hardware_summary().get("text")


def test_main_window_exposes_the_performance_action(pwin):
    assert pwin.performance_action.text() == "Performance"
