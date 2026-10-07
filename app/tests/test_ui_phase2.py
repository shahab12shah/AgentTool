"""Phase 2 acceptance workflow, driven through the real main window (offscreen Qt)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import NARRATION, ScriptedProvider, make_audio

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.analysis.models import SceneStatus  # noqa: E402
from app.main import create_window  # noqa: E402
from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)

pytestmark = needs_ffmpeg


class GatedProvider(ScriptedProvider):
    """Blocks inside transcribe() so the UI's 'Transcribing…' state can be observed."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.gate = threading.Event()

    def transcribe(self, audio_path, language=None, progress=None, should_cancel=None):
        if progress:
            progress(0.72, "Processing audio...")
        assert self.gate.wait(20)
        return super().transcribe(audio_path, language, progress, should_cancel)


def use_provider(win, provider):
    win.ws.transcripts.registry.register(provider)
    win.ws.transcripts.resolve_provider = lambda name=None: provider
    win.voice_panel.refresh()


def setup_project(win, tmp_path, provider):
    ws = win.ws
    project = create_project(win, tmp_path, "Phase2")
    use_provider(win, provider)
    win.script_panel.editor.setPlainText(NARRATION)
    win.script_panel.flush()
    audio = make_audio(tmp_path / "narration.wav", provider.words[-1].end + 1.0)
    win.voice_panel.import_path(audio)
    pump(lambda: project.voice_over.asset_id is not None and ws.jobs.wait_idle(0))
    win.go_to("Voice")  # visibility assertions only make sense on the page the user is looking at
    return project, audio


def scene_rows(win):
    return [win.scene_panel.table.item(r, 0).text() for r in range(win.scene_panel.table.rowCount())]


def test_phase2_acceptance_workflow(win, tmp_path, app_paths, qapp, monkeypatch):
    ws = win.ws
    provider = GatedProvider(NARRATION)
    project, audio = setup_project(win, tmp_path, provider)
    vp = win.voice_panel

    # ---- before transcription ----
    assert "Not Started" in vp.state_label.text() and vp.transcribe_btn.isEnabled()
    from app.core.timecode import format_duration_short

    asset = project.assets.get(project.voice_over.asset_id)
    expected_dur = format_duration_short(asset.duration)  # e.g. "2:04"
    assert vp.filename.text() == "narration.wav" and vp.duration.text() == expected_dur
    assert "scripted-test" in vp.engine_label.text()
    assert not win.scene_panel.analyze_btn.isEnabled()  # scenes need a transcript first

    # ---- transcribing: progress + message visible, UI responsive ----
    QTest.mouseClick(vp.transcribe_btn, Qt.MouseButton.LeftButton)
    pump(lambda: vp.progress.isVisible() and vp.progress.value() >= 60)
    assert vp.state_label.text() == "Transcribing…" and "Processing audio" in vp.progress_msg.text()
    assert vp.cancel_btn.isVisible() and not vp.import_btn.isEnabled()
    provider.gate.set()
    pump(lambda: "Transcription Complete" in vp.state_label.text())

    # ---- completion summary + transcript appears ----
    n_words = len(provider.words)
    assert f"Words: {n_words}" in vp.summary_label.text() and "Sentences: 32" in vp.summary_label.text()
    assert f"Duration: {expected_dur}" in vp.summary_label.text()
    text = vp.viewer.view.toPlainText()
    assert "00:00:00.000" in text and "Silver demand has changed dramatically." in text
    assert "matches the script exactly" in vp.align_label.text()
    assert project.transcription.transcript.sentence_strategy == "script-guided"

    # ---- search ----
    vp.viewer.search.setText("Tesla")
    assert vp.viewer.match_label.text() == "1 / 1"
    vp.viewer.search.setText("zzz-not-there")
    assert vp.viewer.match_label.text() == "no matches"
    vp.viewer.search.clear()

    # ---- word timing display ----
    vp.viewer.timings.setChecked(True)
    assert "‹0.00›" in vp.viewer.view.toPlainText()
    vp.viewer.timings.setChecked(False)

    # ---- playback: the highlight follows the audio position ----
    tr = project.transcription.transcript
    vp.player.play()
    pump(lambda: vp.player.position > 0.4 or vp.viewer.current_sentence >= 0, timeout=10)
    vp.player.pause()
    target = tr.sentences[7]
    mid_word = tr.words[tr.words.index(next(w for w in tr.words if w.word_id == target.word_ids[2]))]
    vp.player.position_changed.emit((mid_word.start + mid_word.end) / 2)  # drive position deterministically (no audio device here)
    assert vp.viewer.current_sentence == 7
    assert tr.words[vp.viewer.current_word].word_id == mid_word.word_id
    sels = vp.viewer.view.extraSelections()
    assert any(s.cursor.selectedText() == mid_word.text for s in sels)  # the spoken word is highlighted
    vp.player.position_changed.emit(target.end + 0.4)  # in the pause after the sentence: no word is highlighted
    assert vp.viewer.current_word == -1

    # ---- click sentence -> jumps (and the player seeks) ----
    got = []
    vp.viewer.seek_requested.connect(got.append)
    sent = tr.sentences[12]
    pos = vp.viewer._sent_ranges[12][0] + 3
    vp.viewer._on_click(pos)
    assert got == [sent.start]
    pump(lambda: abs(vp.player.position - sent.start) < 0.5, timeout=10)
    vp.viewer._on_double_click(vp.viewer._word_ranges[40][0] + 1)
    assert got[-1] == tr.words[vp.viewer._word_ranges[40][2]].start

    # ---- scene analysis ----
    win.go_to("Scenes")
    sp = win.scene_panel
    assert sp.analyze_btn.isEnabled() and "No scenes yet" in sp.state_label.text()
    QTest.mouseClick(sp.analyze_btn, Qt.MouseButton.LeftButton)
    pump(lambda: len(project.scenes) >= 14 and not sp._running())
    assert sp.table.rowCount() == len(project.scenes) >= 14
    assert "scenes" in sp.state_label.text() and "overall topic" in sp.state_label.text()

    # ---- review Scene 14 ----
    sp.select_scene(project.scenes[13].id)
    scene = project.scenes[13]
    intent = project.visual_intents[scene.id]
    assert sp.d_title.text() == "Scene 14"
    assert sp.narration.toPlainText() == scene.narration and scene.narration.startswith("Next, consider inflation")
    assert "→" in sp.d_time.text() and sp.topic.text() == scene.topic
    assert sp.vtype.currentText() == intent.type.value == "EVIDENCE"
    assert f"{scene.importance:.0%}" in sp.d_metrics.text() and f"{scene.segmentation_confidence:.0%}" in sp.d_metrics.text()
    assert "Federal Reserve" in sp.extracted.toPlainText() and "3.5%" in sp.extracted.toPlainText() and "needs evidence" in sp.extracted.toPlainText()
    assert "Why:" in sp.rationale.text()

    # ---- manual split via the UI ----
    n = len(project.scenes)
    mid = scene.start + scene.duration * 0.5
    sp.split_at.setValue(mid)
    mid = sp.split_at.value()  # what the user entered (the spin box has millisecond precision)
    QTest.mouseClick(sp.split_btn, Qt.MouseButton.LeftButton)
    labels = scene_rows(win)
    assert len(labels) == n + 1 and "14A" in labels and "14B" in labels
    a = next(s for s in project.scenes if s.label == "14A")
    b = next(s for s in project.scenes if s.label == "14B")
    assert a.end == b.start == mid and a.start == scene.start and b.end == scene.end
    assert (a.narration + " " + b.narration).split() == scene.narration.split()
    win.undo_action.trigger()
    assert len(project.scenes) == n and "14A" not in scene_rows(win)

    # ---- manual merge via the UI ----
    sp.select_scene(project.scenes[4].id)
    QTest.mouseClick(sp.merge_next, Qt.MouseButton.LeftButton)
    assert len(project.scenes) == n - 1
    win.undo_action.trigger()
    assert len(project.scenes) == n

    # ---- edit + approve ----
    sp.select_scene(project.scenes[2].id)
    sp.topic.setText("My topic")
    sp.vtype.setCurrentText("PERSON")
    sp.primary.setText("A taxpayer")
    QTest.mouseClick(sp.apply_btn, Qt.MouseButton.LeftButton)
    edited = project.scenes[2]
    assert edited.topic == "My topic" and "topic" in edited.user_edited_fields
    assert project.visual_intents[edited.id].type.value == "PERSON" and project.visual_intents[edited.id].author.value == "USER"
    QTest.mouseClick(sp.approve_btn, Qt.MouseButton.LeftButton)
    assert project.scenes[2].status is SceneStatus.APPROVED and sp.approve_btn.text() == "Unapprove"

    # ---- visual preferences (soft targets) ----
    win.go_to("Visuals")
    vis = win.visuals_panel
    assert "100%" in vis.total_label.text() and vis.warning_label.text() == "" and vis.accuracy.value() == 85
    vis.target[next(k for k in vis.target if k.value == "YOUTUBE")].setValue(35)
    assert "115%" in vis.total_label.text() and "exceed 100%" in vis.warning_label.text()
    assert project.visual_preferences.setting(next(k for k in vis.target if k.value == "YOUTUBE")).target_percent == 35  # saved anyway
    vis.accuracy.setValue(90)
    vis.rules["prefer_real_visuals"].setChecked(True)

    # ---- save, close, reopen: everything is restored ----
    before = project.to_document()
    root = project.root
    assert win.save()
    win.close()
    ws.shutdown()
    win2, ws2 = create_window(app_paths)
    win2.show()
    win2.open_project(root)
    p2 = ws2.project
    for key in ("transcription", "script_alignment", "scenes", "visual_intents", "scene_analysis", "visual_preferences"):
        assert p2.to_document()[key] == before[key], key
    assert "Silver demand has changed dramatically." in win2.voice_panel.viewer.view.toPlainText()
    assert "Transcription Complete" in win2.voice_panel.state_label.text()
    assert win2.scene_panel.table.rowCount() == len(before["scenes"]) >= 14
    win2.scene_panel.select_scene(p2.scenes[13].id)
    assert win2.scene_panel.d_title.text() == "Scene 14" and win2.scene_panel.vtype.currentText() == "EVIDENCE"
    assert win2.visuals_panel.accuracy.value() == 90 and "115%" in win2.visuals_panel.total_label.text()
    assert win2.visuals_panel.rules["prefer_real_visuals"].isChecked()

    # ---- replace the voice-over: transcript becomes OUTDATED, scenes can't be regenerated until re-transcribed ----
    other = Path(tmp_path / "other.wav")
    new_provider = ScriptedProvider("Copper mining happens underground. Engineers drill deep shafts. Now let's talk about smelting. "
                                    "Furnaces melt the ore into pure metal.")
    use_provider(win2, new_provider)
    make_audio(other, new_provider.words[-1].end + 1.0)
    win2.voice_panel.import_path(other)
    pump(lambda: p2.voice_over.filename == "other.wav" and ws2.jobs.wait_idle(0))
    assert "OUTDATED" in win2.voice_panel.state_label.text() and win2.voice_panel.transcribe_btn.text() == "Re-transcribe"
    assert "Silver demand" in win2.voice_panel.viewer.view.toPlainText()  # old transcript kept, never silently deleted
    win2.go_to("Scenes")
    assert "outdated" in win2.scene_panel.state_label.text() and not win2.scene_panel.analyze_btn.isEnabled()
    old_scene_count = len(p2.scenes)

    # ---- re-transcribe, then scenes can be regenerated (with confirmation because the user edited scenes) ----
    win2.go_to("Voice")
    QTest.mouseClick(win2.voice_panel.transcribe_btn, Qt.MouseButton.LeftButton)
    pump(lambda: "Transcription Complete" in win2.voice_panel.state_label.text())
    assert "Copper mining" in win2.voice_panel.viewer.view.toPlainText() and "script" in win2.voice_panel.align_label.text().lower()
    assert "significant differences" in win2.voice_panel.align_label.text()  # the old script no longer matches the new narration
    win2.go_to("Scenes")
    assert win2.scene_panel.analyze_btn.isEnabled()
    asked = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.Yes)[1])
    QTest.mouseClick(win2.scene_panel.analyze_btn, Qt.MouseButton.LeftButton)
    pump(lambda: 0 < len(p2.scenes) < old_scene_count and not win2.scene_panel._running())
    assert asked and "edited, approved or split" in asked[0]
    assert all(s.narration for s in p2.scenes) and "Copper" in p2.scenes[0].narration
    ws2.close_project()
    ws2.shutdown()
    win2.close()


def test_transcription_failure_ui_offers_retry_and_choose_another_file(win, tmp_path):
    provider = ScriptedProvider(NARRATION, fail="The speech engine crashed.")
    project, _ = setup_project(win, tmp_path, provider)
    vp = win.voice_panel
    QTest.mouseClick(vp.transcribe_btn, Qt.MouseButton.LeftButton)
    pump(lambda: "Transcription failed." in vp.state_label.text())
    assert "The speech engine crashed." in vp.state_label.text()
    assert vp.retry_btn.isVisible() and vp.choose_btn.isVisible() and not vp.transcribe_btn.isVisible()
    provider.fail = None  # the problem is fixed
    QTest.mouseClick(vp.retry_btn, Qt.MouseButton.LeftButton)
    pump(lambda: "Transcription Complete" in vp.state_label.text())
    assert not vp.retry_btn.isVisible()


def test_scene_failure_ui_shows_partial_progress_and_retry_from_scene(win, tmp_path):
    from app.tests.test_scenes import FailingAnalyzer

    provider = ScriptedProvider(NARRATION)
    project, _ = setup_project(win, tmp_path, provider)
    win.voice_panel.transcribe_btn.click()
    pump(lambda: "Transcription Complete" in win.voice_panel.state_label.text())
    fa = FailingAnalyzer(fail_at=5)
    win.ws.scenes.analyzer = fa
    win.go_to("Scenes")
    sp = win.scene_panel
    sp.analyze_btn.click()
    pump(lambda: project.scene_analysis.status == "PARTIAL" and not sp._running())
    assert sp.failure_box.isVisible()
    assert "Scenes 1–5 complete." in sp.failure_label.text() and "Scene 6 failed" in sp.failure_label.text()
    assert sp.retry_btn.text() == "Retry From Scene 6"
    statuses = [sp.table.item(r, 6).text() for r in range(sp.table.rowCount())]
    assert statuses[5] == "FAILED" and statuses[6] == "Pending" and "Failed" not in statuses[:5]
    fa.armed = False
    fa.enrich_calls = []
    sp.retry_btn.click()
    pump(lambda: project.scene_analysis.status == "COMPLETE" and not sp._running())
    assert not sp.failure_box.isVisible() and fa.enrich_calls[0] == 5  # resumed at scene 6, not scene 1


def test_settings_dialog_exposes_transcription_without_storing_keys(win, tmp_path, monkeypatch):
    from app.ui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(win.ws.settings, win.ws.describe_ffmpeg, win.ws.transcripts.provider_report)
    assert "pocketsphinx" in dlg.provider_status.text() and "faster-whisper" in dlg.provider_status.text()
    dlg.api_key_env.setText("MY_STT_KEY")
    dlg.provider.setCurrentText("api")
    s = dlg.result_settings()
    assert s.transcription_provider == "api" and s.api_key_env == "MY_STT_KEY"
    win.ws.update_settings(s)
    text = win.ws.paths.settings_file.read_text()
    assert "MY_STT_KEY" in text and "sk-" not in text and "api_key\"" not in text  # only the variable NAME is stored
