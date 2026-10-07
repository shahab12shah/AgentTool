"""The AI Quality Control page through the real main window: run, progress, scores, issues, detail, ignore, markers on the timeline, export gate.

QC runs through a small fake engine here (the analysers have their own tests); what is under test is the UI: it only talks to ``ws.qc``, stays responsive, and shows
severity / confidence / the export decision faithfully.
"""

from __future__ import annotations

import pytest

from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.issue_model import QCCategory, QCFixSpec, QCIssue  # noqa: F401
from app.qc.qc_engine import QCEngine
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, narrate
from app.tests.conftest import needs_ffmpeg
from app.tests.render_helpers import build_demo

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QInputDialog, QMessageBox  # noqa: E402

from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)
from app.ui.timeline_canvas import RULER_H  # noqa: E402

pytestmark = needs_ffmpeg


class Fake(BaseChecker):
    domains = ("timeline", "scenes")

    def __init__(self, cid, issues=(), cats=(QCCategory.TIMELINE,)):
        self.id, self.label, self.categories, self._issues = cid, cid.title(), cats, list(issues)

    def run(self, ctx, report):
        return CheckerOutput([QCIssue.from_dict(i.to_dict()) for i in self._issues], {})


def mk(sev, code, cat, scene, start, end, *, fix=None, item=None, track=None, conf=100.0, **kw) -> QCIssue:
    i = QCIssue(f"qci_{code}", code, cat, sev, kw.pop("title", "Title " + code), kw.pop("description", "what was detected"), scene, item, track, start, end, conf, "deterministic:fake",
                why_it_matters=kw.pop("why", "viewers notice"), suggested_fix=kw.pop("suggested", ""), fix=fix, auto_fix_available=fix is not None, auto_fix_safe=bool(fix and fix.safe), **kw)
    i.fingerprint = i.make_fingerprint()
    return i


@pytest.fixture
def qwin(win, tmp_path):
    project = create_project(win, tmp_path, "UiQc")
    ws = win.ws
    ws.qc.update_settings(run_before_export=False)
    project.voice_over.duration = 30.0
    a = add_asset(project, "a.mp4", "video", duration=30)
    s1 = add_scene(project, 0, 15, "Silver rose sharply this week.", importance=0.8)
    s2 = add_scene(project, 15, 30, "Demand keeps growing steadily.")
    narrate(project)
    c1 = add_clip(project, "track_v1", a, 0, 15, scene=s1)
    c2 = add_clip(project, "track_v1", a, 15, 15, scene=s2)
    win.s1, win.s2, win.c1, win.c2 = s1, s2, c1, c2
    ws.qc.engine = QCEngine([
        Fake("timeline", [mk(Severity.ERROR, "visual.mismatch", QCCategory.VISUAL_ACCURACY, s1.id, 3.0, 6.0, item=c1.id, track="track_v1", conf=82.0, title="Visual mismatch",
                             suggested="Choose a visual about silver supply."),
                          mk(Severity.WARNING, "caption.too_fast", QCCategory.CAPTION, s2.id, 20.0, 22.0, item=c2.id, track="track_v1", title="Caption too dense"),
                          mk(Severity.NOTICE, "pacing.slow", QCCategory.PACING, None, 8.0, 9.0, title="Slow stretch")]),
        Fake("caption", cats=(QCCategory.CAPTION,)),
    ])
    return win


def run_and_wait(win, force=False):
    panel = win.qc_panel
    panel.run_qc(force)
    pump(lambda: not win.ws.qc.running and win.ws.jobs.wait_idle(0.0))
    pump(lambda: panel.issue_table.rowCount() > 0 or win.ws.project.qc_runs)
    panel.refresh()


def test_the_quality_page_exists_runs_in_the_background_and_shows_scores_and_issues(qwin):
    win = qwin
    assert "Quality" in win.page_index and win.page_index["Quality"] < win.page_index["Export"]
    win.go_to("Quality")
    panel = win.qc_panel
    assert panel.overall_label.text() == "Not analysed yet" and panel.run_btn.isEnabled() and not panel.cancel_btn.isEnabled()
    QTest.mouseClick(panel.run_btn, Qt.MouseButton.LeftButton)
    assert win.ws.qc.running or win.ws.project.qc_runs  # started as a job: the click returned at once
    pump(lambda: win.ws.project.qc_runs and not win.ws.qc.running)
    panel.refresh()
    s = win.ws.project.qc_scores
    assert panel.overall_label.text() == f"Overall Score: {s.overall:.0f}/100" and panel.status_label.text() in ("REVIEW RECOMMENDED", "FIX REQUIRED", "READY", "EXPORT BLOCKED")
    assert "Errors  1" in panel.count_labels["ERROR"].text() and "Warnings  1" in panel.count_labels["WARNING"].text() and "Notices  1" in panel.count_labels["NOTICE"].text()
    assert "EXPORT BLOCKED" in panel.export_label.text()  # default level: Critical + Errors
    assert panel.issue_table.rowCount() == 3 and panel.issue_table.item(0, 0).text() == "ERROR" and panel.issue_table.item(0, 4).text() == "Visual mismatch"
    assert panel.issue_table.item(0, 5).text().startswith("82%") and panel.issue_table.item(0, 1).text() == "Scene 1"
    assert panel.group_table.rowCount() == 8 and panel.group_table.item(0, 0).text() == "Visual Accuracy"
    assert panel.stages.rowCount() == 2 and panel.progress.value() == 100


def test_selecting_an_issue_shows_the_detail_and_the_right_buttons(qwin):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    panel.issue_table.selectRow(0)
    pump(lambda: panel._current() is not None)
    text = panel.detail.toPlainText()
    assert "Visual mismatch" in text and "Severity: ERROR" in text and "Confidence: 82%" in text and "Scene: Scene 1" in text and "Why it matters" in text and "viewers notice" in text
    assert panel.replace_btn.isEnabled() and panel.search_btn.isEnabled() and panel.timeline_btn.isEnabled() and panel.scene_open_btn.isEnabled() and panel.ignore_btn.isEnabled()
    assert not panel.fix_btn.isEnabled()  # no automatic fix for this one (it needs the user to choose a visual)
    panel.issue_table.selectRow(2)  # the notice has no scene
    pump(lambda: panel._current() is not None and panel._current().code == "pacing.slow")
    assert not panel.replace_btn.isEnabled() and not panel.scene_open_btn.isEnabled()


def test_ignore_removes_the_issue_from_the_open_list_and_it_can_be_shown_and_undone(qwin, monkeypatch):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Intentional", True)))
    panel.issue_table.selectRow(1)
    pump(lambda: panel._current() is not None and panel._current().code == "caption.too_fast")
    before = win.ws.project.qc_scores.overall
    QTest.mouseClick(panel.ignore_btn, Qt.MouseButton.LeftButton)
    pump(lambda: panel.issue_table.rowCount() == 2)
    assert win.ws.project.qc_ignored_issues[0].reason == "Intentional" and win.ws.project.qc_scores.overall >= before and "Warnings  0" in panel.count_labels["WARNING"].text()
    panel.show_closed.setChecked(True)
    pump(lambda: panel.issue_table.rowCount() == 3)
    assert any("(ignored)" in panel.issue_table.item(r, 0).text() for r in range(3))
    panel.show_closed.setChecked(False)
    win.ws.undo()  # the ignore is an ordinary undoable edit
    pump(lambda: panel.issue_table.rowCount() == 3)
    assert not win.ws.project.qc_ignored_issues and "Warnings  1" in panel.count_labels["WARNING"].text()


def test_timeline_markers_follow_the_issues_open_them_and_respect_the_marker_mode(qwin):
    win = qwin
    run_and_wait(win)
    win.go_to("Timeline")
    canvas = win.timeline_panel.canvas
    canvas.reload_qc_markers()
    assert [m["severity"] for m in canvas.qc_markers] == ["ERROR", "NOTICE", "WARNING"] or len(canvas.qc_markers) == 3
    err = next(m for m in canvas.qc_markers if m["severity"] == "ERROR")
    x = int(canvas.time_to_x(err["time"]))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(x, RULER_H - 6))  # click the flag on the ruler
    pump(lambda: win.pages.currentIndex() == win.page_index["Quality"])
    assert win.qc_panel._selected == err["issue_id"] and "Visual mismatch" in win.qc_panel.detail.toPlainText()
    assert abs(canvas.playhead - err["time"]) < 1e-6
    win.go_to("Timeline")
    mode = win.timeline_panel.marker_mode
    mode.setCurrentIndex(mode.findData("critical_only"))
    mode.activated.emit(mode.currentIndex())
    pump(lambda: win.ws.project.qc_settings.marker_mode == "critical_only")
    assert canvas.qc_markers == []  # no critical issue in this run
    mode.setCurrentIndex(mode.findData("hidden"))
    mode.activated.emit(mode.currentIndex())
    pump(lambda: win.ws.project.qc_settings.marker_mode == "hidden")
    assert canvas.qc_markers == []
    mode.setCurrentIndex(mode.findData("all"))
    mode.activated.emit(mode.currentIndex())
    pump(lambda: len(canvas.qc_markers) == 3)
    canvas.grab()  # painting with markers must not raise


def test_open_timeline_button_jumps_to_the_clip(qwin):
    win = qwin
    run_and_wait(win)
    win.go_to("Quality")
    panel = win.qc_panel
    panel.issue_table.selectRow(0)
    pump(lambda: panel._current() is not None)
    QTest.mouseClick(panel.timeline_btn, Qt.MouseButton.LeftButton)
    pump(lambda: win.pages.currentIndex() == win.page_index["Timeline"])
    assert abs(win.timeline_panel.canvas.playhead - 3.0) < 1e-6 and win.ws.selected_clip_id == win.c1.id


def test_export_is_blocked_with_the_issues_listed_and_the_blocking_level_is_configurable(qwin, monkeypatch):
    win = qwin
    run_and_wait(win)
    win.go_to("Export")
    shown: list[tuple[str, list[str]]] = []
    choice = {"text": "Cancel"}
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append((self.text(), [b.text() for b in self.buttons()])) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(b for b in self.buttons() if b.text() == choice["text"]))
    panel = win.export_panel
    pump(lambda: "Quality control" in panel.qc_line.text())
    assert "EXPORT BLOCKED" in panel.qc_line.text()
    n = len(win.ws.render.jobs())
    panel.start_export()
    assert shown and "EXPORT BLOCKED" in shown[0][0] and "Visual mismatch" in shown[0][0] and "Open Quality Control" in shown[0][1] and "Export anyway" not in shown[0][1]  # errors block, no override by default
    assert len(win.ws.render.jobs()) == n  # nothing was queued
    choice["text"] = "Open Quality Control"
    panel.start_export()
    pump(lambda: win.pages.currentIndex() == win.page_index["Quality"])


@pytest.fixture
def export_demo(win, tmp_path):
    project = create_project(win, tmp_path, "UiQcExport")
    win.demo = build_demo(win.ws, tmp_path, wait=lambda: pump(lambda: len(win.ws.project.assets.all()) >= 6 and win.ws.jobs.wait_idle(0), 120))
    win.ws.render.update_settings(resolution="480p", quality="draft", preset_id="draft")
    win.project = project
    return win


def test_run_before_export_runs_qc_first_and_then_exports_when_nothing_blocks(export_demo):
    win = export_demo
    ws = win.ws
    ws.qc.engine = QCEngine([Fake("timeline")])  # a clean QC result
    assert ws.project.qc_settings.run_before_export and not ws.project.qc_runs
    win.go_to("Export")
    panel = win.export_panel
    pump(lambda: panel.start_btn.isEnabled() and "✓" in panel.preflight_view.text())
    assert "not run yet" in panel.qc_line.text() and "automatically" in panel.qc_line.text()
    n = len(ws.render.jobs())
    QTest.mouseClick(panel.start_btn, Qt.MouseButton.LeftButton)
    pump(lambda: len(ws.project.qc_runs) == 1 and len(ws.render.jobs()) > n, 120)  # QC ran, then the export was queued by itself
    assert ws.project.qc_runs[0]["trigger"] == "export" and ws.project.qc_scores.export == "READY"
    job = ws.render.jobs()[-1]
    pump(lambda: job.finished_event.is_set(), 180)
    assert job.status.value == "COMPLETED"
