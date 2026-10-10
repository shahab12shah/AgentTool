"""The AI Quality Control page through the real main window: run, progress, scores, issues, detail, ignore, markers on the timeline, export gate.

QC runs through a small fake engine here (the analysers have their own tests); what is under test is the UI: it only talks to ``ws.qc``, stays responsive, and shows
severity / confidence / the export decision faithfully.
"""

from __future__ import annotations

import pytest

from app.qc import fix_catalog as fc
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.issue_model import IssueStatus, QCCategory, QCFixSpec, QCIssue  # noqa: F401
from app.qc.qc_engine import QCEngine
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, narrate
from app.qc.tests.test_qc_fix_engine import add_visual, issue as fix_issue
from app.tests.conftest import needs_ffmpeg
from app.tests.render_helpers import build_demo

pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QInputDialog, QMessageBox  # noqa: E402

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


# ================================================================================================ review fixes: filters, buttons, confirmations, refresh, export flow
def select_rows(panel, rows):
    sm = panel.issue_table.selectionModel()
    sm.clearSelection()
    for r in rows:
        sm.select(panel.issue_table.model().index(r, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    pump(lambda: len(panel._selected_ids()) == len(rows))


def test_the_issue_list_filters_by_category_scene_status_and_confidence_and_an_empty_result_says_why(qwin):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    assert panel.issue_table.rowCount() == 3 and panel.issues_hint.text() == ""
    panel.cat_filter.setCurrentIndex(panel.cat_filter.findData("CAPTION"))
    assert [panel.issue_table.item(r, 4).text() for r in range(panel.issue_table.rowCount())] == ["Caption too dense"]
    panel.cat_filter.setCurrentIndex(0)
    panel.scene_filter.setCurrentIndex(panel.scene_filter.findData(win.s1.id))
    assert [panel.issue_table.item(r, 4).text() for r in range(panel.issue_table.rowCount())] == ["Visual mismatch"]
    panel.scene_filter.setCurrentIndex(0)
    panel.conf_filter.setCurrentIndex(panel.conf_filter.findData("low"))  # nothing is a low-confidence judgement here
    assert panel.issue_table.rowCount() == 0 and "No issue matches the filters" in panel.issues_hint.text()
    panel.conf_filter.setCurrentIndex(panel.conf_filter.findData("high"))
    assert panel.issue_table.rowCount() == 3
    panel.conf_filter.setCurrentIndex(0)
    panel.status_filter.setCurrentIndex(panel.status_filter.findData("fixed"))
    assert panel.issue_table.rowCount() == 0
    panel.status_filter.setCurrentIndex(panel.status_filter.findData("all"))
    assert panel.issue_table.rowCount() == 3 and panel.show_closed.isChecked()
    panel.show_closed.setChecked(False)  # the old checkbox drives the status filter
    assert panel.status_filter.currentData() == "open"
    # severities stay sorted most severe first whatever the filter
    assert [panel.issue_table.item(r, 0).text() for r in range(3)] == ["ERROR", "WARNING", "NOTICE"]


def test_a_clean_run_and_a_never_run_project_have_an_empty_state_message(qwin):
    win = qwin
    win.go_to("Quality")
    panel = win.qc_panel
    panel.refresh()
    assert "has not run yet" in panel.issues_hint.text()
    win.ws.qc.engine = QCEngine([Fake("timeline")])
    run_and_wait(win)
    assert panel.issue_table.rowCount() == 0 and "No open issues" in panel.issues_hint.text()


def test_opening_an_issue_from_a_marker_clears_filters_that_hide_it(qwin):
    win = qwin
    run_and_wait(win)
    panel = win.qc_panel
    panel.sev_filter.setCurrentIndex(panel.sev_filter.findData("NOTICE"))
    assert panel.issue_table.rowCount() == 1
    err = next(m for m in win.timeline_panel.canvas.qc_markers if m["severity"] == "ERROR")
    panel.select_issue(err["issue_id"])
    assert panel._selected == err["issue_id"] and panel.sev_filter.currentData() in ("", None) and panel.issue_table.rowCount() == 3


def test_a_refresh_keeps_the_selection_and_the_unsaved_settings_edits(qwin, monkeypatch):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    select_rows(panel, [0, 1])
    first = panel._selected
    ids = panel._selected_ids()
    panel._refresh_now()
    assert panel._selected_ids() == ids and panel._selected == first  # a background refresh (a finished run, an edit) must not drop what the user selected
    # settings edits that were not saved survive a refresh; Revert brings the saved values back
    path, w = next((k, v) for k, v in panel.set_widgets.items() if not isinstance(v, QCheckBox))
    saved = w.value()
    w.setValue(w.maximum() if w.value() != w.maximum() else w.minimum())
    edited = w.value()
    panel._refresh_now()
    assert w.value() == edited
    calls = []
    real = win.ws.qc.update_settings
    monkeypatch.setattr(win.ws.qc, "update_settings", lambda **kw: calls.append(kw) or real(**kw))
    panel.save_settings()
    assert list(calls[0]) == [path]  # only what was changed is sent (the form must not round or rewrite anything else)
    assert win.ws.project.qc_settings.get_path(path) == edited
    calls.clear()
    panel.save_settings()
    assert not calls  # nothing changed: no new undo step
    w.setValue(saved)
    panel.revert_settings()
    assert w.value() == edited


def test_progress_ticks_do_not_rebuild_the_page_and_a_hidden_page_waits_until_it_opens(qwin, monkeypatch):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    QTest.qWait(300)  # let the refreshes queued by the run itself finish first
    built = []
    real = panel._fill_issues
    monkeypatch.setattr(panel, "_fill_issues", lambda *a: built.append(1) or real(*a))
    win.ws.bus.publish("qc.updated", kind="progress", fraction=0.5, message="x")
    QTest.qWait(250)
    assert not built  # only the progress widgets follow a tick
    win.go_to("Timeline")  # the page is no longer shown
    win.ws.bus.publish("qc.updated", kind="issue_changed")
    QTest.qWait(250)
    assert not built and panel._dirty
    win.go_to("Quality")
    assert built and not panel._dirty


def test_the_stale_banner_follows_an_edit_and_a_partial_run_says_so(qwin):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    assert panel.stale_label.text() == ""
    win.ws.timeline.move_clip(win.c2.id, 16.0)  # an edit after the run
    pump(lambda: "changed since this QC run" in panel.stale_label.text())
    run_and_wait(win)
    assert panel.stale_label.text() == ""
    win.ws.project.qc_runs[-1]["content_hash"] = ""  # what a canceled run / a scene run stores: it does not vouch for the whole project
    panel.refresh()
    assert "did not cover the whole project" in panel.stale_label.text() and "run QC again" in panel.stale_label.text()


def make_issue_set(win):
    """A critical issue, one with a protected (disabled) fix, one that only navigates, one with a real command fix."""
    ws = win.ws
    ext = fc.clip_extend(win.c1.id, 15.0, 1.0, ws.project.qc_settings)
    items = [
        mk(Severity.CRITICAL, "timeline.broken", QCCategory.TIMELINE, win.s1.id, 1.0, 2.0, item=win.c1.id, track="track_v1", title="Broken"),
        mk(Severity.WARNING, "visual.search", QCCategory.ASSET, win.s2.id, 16.0, 18.0, fix=fc.navigate("visual.search_again", "Search again", scene_id=win.s2.id), title="Low resolution"),
        mk(Severity.WARNING, "x.protected", QCCategory.TIMELINE, win.s1.id, 5.0, 6.0, fix=ext, title="Protected"),
        mk(Severity.NOTICE, "x.command", QCCategory.TIMELINE, win.s2.id, 20.0, 21.0, fix=ext, title="Has a command"),
    ]
    items[2].auto_fix_available = False  # a user-owned / locked element: reported, never fixable
    items[2].fix_blocked_reason = "This element was edited or created by you: auto-fix is disabled (the issue is still reported)"
    return items


def test_the_nine_buttons_follow_the_selected_issue(qwin):
    win = qwin
    win.go_to("Quality")
    panel = win.qc_panel
    items = make_issue_set(win)
    win.ws.qc.engine = QCEngine([Fake("timeline", items)])
    run_and_wait(win)

    def pick(title):
        row = next(r for r in range(panel.issue_table.rowCount()) if panel.issue_table.item(r, 4).text() == title)
        panel.issue_table.selectRow(row)
        pump(lambda: panel._current() is not None and panel._current().title == title)
        return {k: b.isEnabled() for k, b in dict(fix=panel.fix_btn, preview=panel.preview_btn, ignore=panel.ignore_btn, ignore_type=panel.ignore_type_btn, similar=panel.similar_btn,
                                                  timeline=panel.timeline_btn, scene=panel.scene_open_btn, replace=panel.replace_btn, search=panel.search_btn).items()}

    crit = pick("Broken")
    assert not crit["ignore"] and not crit["ignore_type"] and not crit["fix"] and crit["timeline"] and crit["scene"]  # a CRITICAL can never be ignored
    nav = pick("Low resolution")
    assert not nav["fix"] and not nav["preview"] and not nav["similar"] and nav["replace"] and nav["search"] and nav["ignore"]  # navigate-only: its own buttons, no Fix
    prot = pick("Protected")
    assert not prot["fix"] and not prot["preview"] and not prot["similar"] and prot["ignore"] and "edited or created by you" in panel.fix_note.text()
    cmd = pick("Has a command")
    assert cmd["fix"] and cmd["preview"] and cmd["similar"] and cmd["ignore"] and cmd["ignore_type"] and not cmd["replace"]


def test_a_fixed_or_ignored_issue_offers_neither_fix_nor_ignore(qwin, monkeypatch):
    win = qwin
    win.go_to("Quality")
    panel = win.qc_panel
    win.ws.qc.engine = QCEngine([Fake("timeline", make_issue_set(win)[3:])])
    run_and_wait(win)
    iss = next(i for i in win.ws.project.qc_issues if i.title == "Has a command")
    iss.status = IssueStatus.FIXED
    panel.status_filter.setCurrentIndex(panel.status_filter.findData("all"))
    panel.select_issue(iss.issue_id)
    assert panel.issue_table.item(0, 0).text().endswith("(fixed)")
    assert not panel.fix_btn.isEnabled() and not panel.similar_btn.isEnabled() and not panel.ignore_btn.isEnabled() and not panel.fix_selected_btn.isEnabled()
    assert not any(b.isEnabled() for b in panel.batch_btns.values())


@pytest.fixture
def fixwin(win, tmp_path):
    """A project with AI-made clips and real, fixable issues (two short clips): the real fix engine runs, no QC analysis."""
    project = create_project(win, tmp_path, "UiQcFix")
    ws = win.ws
    ws.qc.update_settings(run_before_export=False)
    project.voice_over.duration = 40.0
    ws.vid = add_asset(project, "long.mp4", "video", duration=40)
    ws.s1 = add_scene(project, 0, 20, "Silver rose sharply this week.", importance=0.8)
    ws.s2 = add_scene(project, 20, 40, "Demand keeps growing steadily.")
    narrate(project)
    win.clips = [add_visual(ws, 0.0, 8.0, scene=ws.s1), add_visual(ws, 10.0, 8.0, scene=ws.s1), add_visual(ws, 20.0, 8.0, scene=ws.s2), add_visual(ws, 30.0, 8.0, scene=ws.s2)]
    # two safe fixes (extend by <= 1 s) and two that need confirmation (extend by 2 s)
    win.safe = [fix_issue(ws, "timeline.clip.short", fc.clip_extend(c.id, c.timeline_end + 1.0, 1.0, project.qc_settings), clip=c, category=QCCategory.TIMELINE, title=f"Short {n}")
                for n, c in enumerate(win.clips[:2])]
    win.careful = [fix_issue(ws, "timeline.clip.long", fc.clip_extend(c.id, c.timeline_end + 2.0, 2.0, project.qc_settings), clip=c, category=QCCategory.TIMELINE, title=f"Long {n}")
                   for n, c in enumerate(win.clips[2:])]
    assert all(i.auto_fix_safe for i in win.safe) and not any(i.auto_fix_safe for i in win.careful)
    win.go_to("Quality")
    win.qc_panel.refresh()
    return win


def decline(monkeypatch, answer):
    asked = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: asked.append(self.objectName()) or answer)
    return asked


def clip_ends(win):
    return [round(win.ws.project.timeline.get_clip(c.id).timeline_end, 3) for c in win.clips]


def test_fix_all_similar_asks_before_a_fix_that_needs_confirmation(fixwin, monkeypatch):
    win = fixwin
    panel = win.qc_panel
    before = clip_ends(win)
    panel.select_issue(win.careful[0].issue_id)
    asked = decline(monkeypatch, QMessageBox.StandardButton.Cancel)
    panel.fix_similar()
    assert asked == ["qcConfirmFixSimilar"] and clip_ends(win) == before  # declined: nothing changed
    asked = decline(monkeypatch, QMessageBox.StandardButton.Ok)
    panel.fix_similar()
    assert asked == ["qcConfirmFixSimilar"] and clip_ends(win)[2:] == [before[2] + 2.0, before[3] + 2.0]
    win.ws.undo()  # one undo step for the whole batch
    assert clip_ends(win) == before


def test_fix_selected_applies_the_safe_ones_as_one_undo_step_and_asks_for_the_rest(fixwin, monkeypatch):
    win = fixwin
    panel = win.qc_panel
    before = clip_ends(win)
    rows = {panel.issue_table.item(r, 4).text(): r for r in range(panel.issue_table.rowCount())}
    select_rows(panel, [rows["Short 0"], rows["Short 1"], rows["Long 0"]])
    asked = decline(monkeypatch, QMessageBox.StandardButton.Cancel)
    panel.fix_selected()
    after = clip_ends(win)
    assert asked == ["qcConfirmFixSelected"]
    assert after[:2] == [before[0] + 1.0, before[1] + 1.0] and after[2:] == before[2:]  # declined: only the safe fixes went ahead
    win.ws.undo()  # the two safe fixes were ONE undo step
    assert clip_ends(win) == before
    pump(lambda: panel.issue_table.rowCount() == 4)
    rows = {panel.issue_table.item(r, 4).text(): r for r in range(panel.issue_table.rowCount())}
    select_rows(panel, [rows["Long 0"]])
    asked = decline(monkeypatch, QMessageBox.StandardButton.Ok)
    panel.fix_selected()
    assert asked == ["qcConfirmFixSelected"] and clip_ends(win)[2] == before[2] + 2.0


def test_a_fix_that_turns_out_to_need_confirmation_asks_instead_of_failing(fixwin, monkeypatch):
    win = fixwin
    panel = win.qc_panel
    before = clip_ends(win)
    iss = win.safe[0]
    iss.fix.params["new_end"] = win.clips[0].timeline_end + 1.0
    iss.auto_fix_safe = True  # the issue says "safe" ...
    win.ws.project.qc_settings.fix_permissions["clip.extend"] = "confirm"  # ... but the user's permission table says "ask me first"
    panel.refresh()
    panel.select_issue(iss.issue_id)
    asked = decline(monkeypatch, QMessageBox.StandardButton.Ok)
    shown = []
    monkeypatch.setattr("app.ui.dialogs.message.show_error", lambda *a, **k: shown.append(a))
    panel.apply_fix(False)
    assert asked == ["qcConfirmFix"] and not shown and clip_ends(win)[0] == before[0] + 1.0


def test_markers_follow_a_fix_and_its_undo_and_the_batch_buttons_need_something_to_fix(fixwin):
    win = fixwin
    panel = win.qc_panel
    canvas = win.timeline_panel.canvas
    canvas.reload_qc_markers()
    n = len(canvas.qc_markers)
    assert n == 4 and panel.batch_btns["fixSafeAll"].isEnabled() and not panel.batch_btns["fixCaptionTiming"].isEnabled()
    panel.select_issue(win.safe[0].issue_id)
    panel.apply_fix(False)
    pump(lambda: len(canvas.qc_markers) == n - 1)
    assert win.safe[0].issue_id not in {m["issue_id"] for m in canvas.qc_markers}
    win.ws.undo()
    pump(lambda: len(canvas.qc_markers) == n)
    assert any(m["issue_id"] == win.safe[0].issue_id for m in canvas.qc_markers)


# ---- export page
def test_the_export_waits_for_quality_control_and_does_not_export_work_qc_never_saw(qwin, monkeypatch):
    win = qwin
    ws = win.ws
    ws.qc.update_settings(run_before_export=True)
    win.go_to("Export")
    panel = win.export_panel
    pump(lambda: panel.start_btn.isEnabled() or panel._report is not None)
    captured = {}
    said: list[str] = []
    ws.bus.subscribe("app.status", lambda _t, p: said.append(p.get("message", "")))
    monkeypatch.setattr(ws.qc, "run_full_qc", lambda **kw: captured.update(kw))
    started = []
    monkeypatch.setattr(ws.render, "start_export", lambda *a, **k: started.append(1))
    panel._submit(False)
    assert panel._qc_pending and not panel.start_btn.isEnabled() and "running before the export" in panel.qc_line.text()
    panel._submit(False)  # a second click while QC runs does nothing
    assert not started and len(captured) >= 1
    # an edit made while QC was running: the finished run no longer matches, so nothing is exported unchecked
    ws.timeline.move_clip(win.c2.id, 16.0)
    captured["on_done"](None)
    assert not started and not panel._qc_pending and any("press Start export again" in m for m in said)
    # the user switched to another project while QC ran: the old project's callback must not start an export of the new one
    panel._submit(False)
    assert panel._qc_pending
    ws.new_project("Other", win.ws.project.root.parent / "other")
    pump(lambda: not panel._qc_pending)
    captured["on_done"](None)
    assert not started


def test_the_export_page_shows_the_file_check_and_follows_the_gate(export_demo):
    win = export_demo
    ws = win.ws
    ws.qc.engine = QCEngine([Fake("timeline")])
    win.go_to("Export")
    panel = win.export_panel
    pump(lambda: panel.start_btn.isEnabled() and "✓" in panel.preflight_view.text())
    QTest.mouseClick(panel.start_btn, Qt.MouseButton.LeftButton)
    pump(lambda: len(ws.project.qc_runs) == 1 and len(ws.render.jobs()) > 0, 120)
    assert "READY" in panel.qc_line.text()  # the gate line followed the run without leaving the page
    job = ws.render.jobs()[-1]
    pump(lambda: job.finished_event.is_set(), 180)
    pump(lambda: panel.render_check.text().startswith("File check:") and "checking" not in panel.render_check.text(), 120)
    assert ws.project.render_qc_results and panel.render_check.text().split(":")[1].strip().split(" ")[0] in ("Passed", "Warnings", "Failed", "Skipped", "Pass", "Ok")


class Boom(BaseChecker):
    domains = ("timeline",)

    def __init__(self, cid, cats):
        self.id, self.label, self.categories = cid, cid.title(), cats

    def run(self, ctx, report):
        raise RuntimeError("boom")


class Slow(BaseChecker):
    domains = ("timeline",)

    def __init__(self, cid, secs):
        self.id, self.label, self.categories, self.secs = cid, cid.title(), (QCCategory.TIMELINE,), secs

    def run(self, ctx, report):
        import time

        end = time.monotonic() + self.secs
        while time.monotonic() < end:
            ctx.check_cancel()
            time.sleep(0.02)
        return CheckerOutput([], {})


def test_a_failed_check_is_shown_as_not_analysed_and_can_be_retried(qwin):
    win = qwin
    win.go_to("Quality")
    panel = win.qc_panel
    boom = Boom("caption", (QCCategory.CAPTION,))
    win.ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.NOTICE, "pacing.slow", QCCategory.PACING, None, 8.0, 9.0, title="Slow stretch")], cats=(QCCategory.PACING,)), boom])
    run_and_wait(win)
    states = {panel.stages.item(r, 0).text(): panel.stages.item(r, 1).text() for r in range(panel.stages.rowCount())}
    assert states["Captions"] == "FAILED" and "These checks did not complete" in panel.stale_label.text()
    assert panel.retry_btn.isEnabled() and "—" in {panel.group_table.item(r, 1).text() for r in range(panel.group_table.rowCount())}  # a group that could not be checked is "—", never 100
    win.ws.qc.engine.checkers[1] = Fake("caption", cats=(QCCategory.CAPTION,))  # the cause was fixed
    QTest.mouseClick(panel.retry_btn, Qt.MouseButton.LeftButton)
    pump(lambda: not win.ws.qc.running and win.ws.jobs.wait_idle(0.0))
    panel.refresh()
    assert not panel.retry_btn.isEnabled() and "—" not in {panel.group_table.item(r, 1).text() for r in range(panel.group_table.rowCount())}
    assert len(win.ws.project.qc_runs) == 1  # the retry completed the same run


def test_a_run_can_be_canceled_and_keeps_what_was_found(qwin):
    win = qwin
    win.go_to("Quality")
    panel = win.qc_panel
    win.ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.NOTICE, "pacing.slow", QCCategory.PACING, None, 8.0, 9.0, title="Slow stretch")], cats=(QCCategory.PACING,)), Slow("pacing", 30.0)])
    QTest.mouseClick(panel.run_btn, Qt.MouseButton.LeftButton)
    pump(lambda: win.ws.qc.running and panel.cancel_btn.isEnabled())
    assert not panel.run_btn.isEnabled() and not panel.fix_selected_btn.isEnabled()
    QTest.mouseClick(panel.cancel_btn, Qt.MouseButton.LeftButton)
    pump(lambda: not win.ws.qc.running and win.ws.jobs.wait_idle(0.0), 30)
    pump(lambda: win.ws.project.qc_runs)
    panel.refresh()
    assert panel.run_btn.isEnabled() and not panel.cancel_btn.isEnabled() and panel.stage_label.text() == "Canceled"
    assert win.ws.project.qc_runs[-1]["state"] == "CANCELED" and panel.issue_table.rowCount() == 1  # the finished check's findings are kept


def test_runs_can_be_compared_and_the_report_saved(qwin):
    win = qwin
    run_and_wait(win)
    run_and_wait(win, force=True)
    panel = win.qc_panel
    assert panel.history_table.rowCount() == 2 and panel.compare_btn.isEnabled()
    panel.compare_runs()
    assert "Select two runs" in panel.compare_view.toPlainText()
    panel.history_table.selectAll()
    pump(lambda: len(panel._history_pick) == 2)
    panel.compare_runs()
    assert "Select two runs" not in panel.compare_view.toPlainText() and panel.compare_view.toPlainText().strip()
    assert panel.report_btn.isEnabled()
    panel.export_report()
    reports = list((win.ws.project.root / "qc").glob("qc_report_run*.md"))
    assert reports and "Visual mismatch" in reports[0].read_text(encoding="utf-8")


def blocked_dialog_probe(monkeypatch, choice):
    shown: list[list[str]] = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append([b.text() for b in self.buttons()]) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(b for b in self.buttons() if b.text() == choice["text"]))
    return shown


def test_export_anyway_is_offered_only_where_the_settings_allow_it_and_never_with_a_critical_issue(qwin, monkeypatch):
    win = qwin
    ws = win.ws
    run_and_wait(win)  # one ERROR, one WARNING, one NOTICE: blocks at the default level
    ws.qc.update_settings(allow_export_override=True)
    win.go_to("Export")
    panel = win.export_panel
    pump(lambda: "Quality control" in panel.qc_line.text())
    choice = {"text": "Export anyway"}
    shown = blocked_dialog_probe(monkeypatch, choice)
    calls: list[bool] = []
    real = ws.render.start_export
    monkeypatch.setattr(ws.render, "start_export", lambda *a, **k: calls.append(bool(k.get("qc_override"))) or real(*a, **k))
    panel.start_export()
    assert len(shown) == 1 and "Export anyway" in shown[0] and "Open Quality Control" in shown[0]
    assert calls == [False, True]  # the explicit choice is passed on, and the second attempt is not blocked again
    # a CRITICAL issue can never be overridden, whatever the setting says
    ws.qc.engine = QCEngine([Fake("timeline", make_issue_set(win)[:1])])
    run_and_wait(win, force=True)
    panel.refresh()
    shown.clear()
    choice["text"] = "Cancel"
    panel.start_export()
    assert len(shown) == 1 and "Export anyway" not in shown[0] and "Open Quality Control" in shown[0]


def test_checks_can_be_switched_off_in_the_settings_editor_but_preflight_always_runs(qwin):
    win = qwin
    win.go_to("Quality")
    panel = win.qc_panel
    panel.refresh()
    assert all(b.isChecked() for b in panel.check_boxes.values()) and not panel.check_boxes["preflight"].isEnabled()
    panel.check_boxes["motion"].setChecked(False)
    panel.save_settings()
    assert "motion" not in win.ws.project.qc_settings.enabled_checkers and "preflight" in win.ws.project.qc_settings.enabled_checkers
    panel.refresh()
    assert not panel.check_boxes["motion"].isChecked() and panel.check_boxes["audio"].isChecked()
