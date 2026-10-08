"""Phase 7 through the real main window (offscreen Qt): Reference page -> import -> analyse -> review -> customize -> apply -> undo / remove.

The panel only calls ``ws.reference``; these tests check what the user sees and that nothing but abstract preferences changes (the timeline and the media library never do).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import make_audio, mux, overlay_video

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from app.reference.style_model import DIMENSION_LABELS, DIMENSIONS  # noqa: E402
from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)

pytestmark = needs_ffmpeg

SECONDS = 20.0


@pytest.fixture(scope="module")
def ref_video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("uiref")
    caps = [{"start": 0.4 + 2.0 * i, "end": 2.0 + 2.0 * i, "text": f"Line {i}", "pos": "bottom", "size": 0.07} for i in range(9)]
    silent = overlay_video(d / "silent.mp4", SECONDS, caps, shots=[2.0] * 10)
    audio = make_audio(d / "mix.wav", SECONDS, voice=[(0.4 + 3.0 * i, 2.6 + 3.0 * i) for i in range(6)], music=[(0.0, SECONDS, 0.12)], sfx=[3.0, 9.0], duck_to=0.5)
    return mux(silent, audio, d / "reference.mp4")


@pytest.fixture
def ref_win(win, tmp_path, ref_video, monkeypatch):  # noqa: F811
    create_project(win, tmp_path, "UiReference")
    win.go_to("Reference")
    win.confirm_answer = {"ok": True}  # what the "Apply reference style" confirmation answers
    monkeypatch.setattr("app.ui.reference_panel.confirm_apply_style", lambda parent: win.confirm_answer["ok"])
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes)
    return win


def snapshot(p) -> str:
    return json.dumps([p.timeline.to_dict(), p.assets.to_list()], sort_keys=True, default=str)


def import_and_analyze(window, ref_video):
    panel, ws = window.reference_panel, window.ws
    panel.import_path(ref_video)
    pump(lambda: len(ws.project.reference_assets) == 1)
    pump(lambda: panel.ref_combo.count() == 1 and panel.analyze_btn.isEnabled())
    panel.analyze_btn.click()
    rid = next(iter(ws.project.reference_assets))
    pump(lambda: ws.project.reference_assets[rid].analysis_status in ("COMPLETED", "PARTIAL"), 180)
    pump(lambda: panel.profile_table.rowCount() == len(DIMENSIONS))
    return rid


def test_the_page_starts_empty_and_import_is_the_only_action(ref_win):
    window = ref_win
    panel = window.reference_panel
    assert panel.import_btn.isEnabled() and not panel.analyze_btn.isEnabled() and not panel.apply_btn.isEnabled() and not panel.customize_btn.isEnabled()
    assert "Import a reference video" in panel.status_label.text() and panel.profile_table.rowCount() == 0
    assert "No reference style is applied" in panel.applied_label.text()


def test_import_analyze_review_and_the_profile_table(ref_win, ref_video):
    window = ref_win
    panel, ws = window.reference_panel, window.ws
    before = snapshot(ws.project)
    rid = import_and_analyze(window, ref_video)
    asset = ws.project.reference_assets[rid]
    assert asset.path.startswith(f"references/{rid}/") and len(ws.project.assets.all()) == 0  # isolated from the media library
    assert "Analyzed" in panel.status_label.text()
    rows = {panel.profile_table.item(r, 0).text(): [panel.profile_table.item(r, c).text() for c in range(4)] for r in range(panel.profile_table.rowCount())}
    assert set(rows) == {DIMENSION_LABELS[d] for d in DIMENSIONS}
    assert rows["Pacing"][1] in ("HIGH", "VERY HIGH", "MEDIUM-HIGH") and float(rows["Pacing"][2]) > 40  # a cut every 2 s is fast
    assert panel.summary_label.text().lower().startswith(("fast", "very fast", "moderate")) and "paced" in panel.summary_label.text().lower()
    assert "SHOTS" in panel.details_view.toPlainText() and "Line" not in panel.details_view.toPlainText()  # aggregates only; no reference text
    assert panel.log_view.toPlainText().strip() and panel.reco_label.text().startswith("•")
    assert panel.compare_table.rowCount() == 10 and "Editing-style similarity" in panel.similarity_label.text() and "not a copyright check" in panel.similarity_label.text()
    assert panel.progress.value() == 100
    # the apply controls are ready, nothing has been applied, the project itself is unchanged
    pump(lambda: panel.apply_btn.isEnabled())
    assert "No reference style is applied" in panel.applied_label.text() and snapshot(ws.project) == before and ws.project.reference_style_overrides.is_empty


def test_plan_preview_customize_and_apply(ref_win, ref_video):
    window = ref_win
    panel, ws = window.reference_panel, window.ws
    rid = import_and_analyze(window, ref_video)
    pump(lambda: panel.apply_btn.isEnabled())
    before = snapshot(ws.project)
    # the plan view shows what would be set; the Result column follows
    pump(lambda: panel.plan_view.text().startswith("Will set:"))
    assert "Target shot length" in panel.plan_view.text()
    assert all(panel.result_values[d].text() != "—" for d in DIMENSIONS if d in ("pacing", "motion_intensity"))
    # style strength and mode feed the plan and are saved with the project
    panel.strength_combo.setCurrentIndex(panel.strength_combo.findData(0.5))
    pump(lambda: ws.project.reference_settings.style_strength == 0.5)
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("BALANCED"))
    pump(lambda: ws.project.reference_settings.application_mode == "BALANCED")
    pump(lambda: "Words per caption" not in panel.plan_view.text() and "Music level" not in panel.plan_view.text() and panel.plan_view.text().startswith("Will set:"))
    # Customize: reference vs your target
    panel.customize_btn.click()
    assert not panel.custom_group.isHidden()
    panel.follow_checks["pacing"].setChecked(False)
    assert panel.target_sliders["pacing"].isEnabled() and panel.target_sliders["pacing"].value() == round(ws.reference.profile(rid).score("pacing"))
    panel.target_sliders["pacing"].setValue(15)
    pump(lambda: ws.project.reference_settings.adjustments.get("pacing") == 15.0)
    pump(lambda: panel._plan is not None and panel._plan.settings.adjustments.get("pacing") == 15.0)
    slow = panel._plan.overrides.target_shot_duration
    panel.follow_checks["pacing"].setChecked(True)  # back to following the reference
    pump(lambda: "pacing" not in ws.project.reference_settings.adjustments)
    pump(lambda: panel._plan is not None and "pacing" not in panel._plan.settings.adjustments)
    assert panel._plan.overrides.target_shot_duration < slow
    # simulation: Current Edit vs Reference Style
    panel.sim_check.setChecked(True)
    pump(lambda: panel.sim_table.rowCount() == len(DIMENSIONS))
    assert "now" in panel.sim_label.text() and "timeline is not changed" in panel.sim_label.text()
    # a cancelled confirmation applies nothing
    window.confirm_answer["ok"] = False
    panel.apply_btn.click()
    assert ws.project.reference_style_overrides.is_empty and not ws.project.reference_settings.enabled and not ws.project.style_application_history
    window.confirm_answer["ok"] = True
    panel.apply_btn.click()
    p = ws.project
    assert p.reference_settings.enabled and not p.reference_style_overrides.is_empty and len(p.style_application_history) == 1
    assert p.reference_style_overrides.caption_max_words is None and p.reference_style_overrides.target_shot_duration is not None  # Balanced: pacing, density, motion only
    assert "APPLIED" in panel.applied_label.text() and "Balanced" in panel.applied_label.text() and "50%" in panel.applied_label.text()
    assert snapshot(p) == before  # applying never touches the timeline or the library, and does not start a render
    assert panel.remove_style_btn.isEnabled() and panel.undo_style_btn.isEnabled() and not p.render_history
    # undo restores the previous (empty) strategy
    panel.undo_style_btn.click()
    pump(lambda: p.reference_style_overrides.is_empty)
    assert "No reference style is applied" in panel.applied_label.text()
    panel.apply_btn.click()
    assert not p.reference_style_overrides.is_empty
    panel.remove_style_btn.click()
    assert p.reference_style_overrides.is_empty and not p.reference_settings.enabled


def test_a_style_wish_is_turned_into_an_abstract_instruction(ref_win, ref_video):
    window = ref_win
    panel, ws = window.reference_panel, window.ws
    import_and_analyze(window, ref_video)
    panel.request_edit.setText("Copy the competitor's exact opening.")
    panel.request_btn.click()
    assert "abstract editing instruction" in panel.request_feedback.text() and "fast, high-information opening" in panel.request_feedback.text()
    assert ws.project.reference_settings.style_request == "Use a fast, high-information opening with strong text emphasis."
    panel.request_edit.setText("Faster pacing with subtle zooms")
    panel.request_btn.click()
    assert panel.request_feedback.text().startswith("Noted:") and ws.project.reference_settings.style_request == "Faster pacing with subtle zooms"


def test_cancel_retry_remove_and_a_failing_analysis(ref_win, ref_video, monkeypatch):
    window = ref_win
    panel, ws = window.reference_panel, window.ws
    from app.reference.analyzer import ReferenceVideoAnalyzer
    from app.reference.signals import ReferenceAnalysisError

    panel.import_path(ref_video)
    pump(lambda: len(ws.project.reference_assets) == 1 and panel.analyze_btn.isEnabled())
    rid = next(iter(ws.project.reference_assets))
    orig = ReferenceVideoAnalyzer.analyze

    def broken(self, path, **kw):
        raise ReferenceAnalysisError("The reference could not be decoded.")

    monkeypatch.setattr(ReferenceVideoAnalyzer, "analyze", broken)
    panel.analyze_btn.click()
    pump(lambda: ws.project.reference_assets[rid].analysis_status == "FAILED")
    pump(lambda: panel.retry_btn.isEnabled())
    assert "could not be decoded" in panel.status_label.text() and not panel.apply_btn.isEnabled() and panel.profile_table.rowCount() == 0
    monkeypatch.setattr(ReferenceVideoAnalyzer, "analyze", orig)
    panel.retry_btn.click()
    pump(lambda: ws.project.reference_assets[rid].analysis_status in ("COMPLETED", "PARTIAL"), 180)
    pump(lambda: panel.profile_table.rowCount() == len(DIMENSIONS))
    folder = ws.project.root / "references" / rid
    assert folder.is_dir()
    panel.remove_btn.click()
    pump(lambda: not ws.project.reference_assets)
    assert not folder.exists() and panel.ref_combo.count() == 0 and panel.profile_table.rowCount() == 0 and not panel.apply_btn.isEnabled()


def test_the_page_follows_a_reopened_project(ref_win, ref_video, tmp_path):
    window = ref_win
    panel, ws = window.reference_panel, window.ws
    rid = import_and_analyze(window, ref_video)
    pump(lambda: panel.apply_btn.isEnabled())
    panel.apply_btn.click()
    sig = ws.reference.profile().signature()
    root = ws.project.root
    ws.save()
    ws.close_project()
    assert panel.ref_combo.count() == 0 and panel.profile_table.rowCount() == 0
    ws.open_project(root)
    pump(lambda: panel.ref_combo.count() == 1 and panel.profile_table.rowCount() == len(DIMENSIONS))
    assert panel.ref_combo.currentData() == rid and ws.reference.profile().signature() == sig and "APPLIED" in panel.applied_label.text()
