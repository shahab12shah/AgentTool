"""Phase 5 acceptance workflow through the real main window: captions -> graphics -> music/SFX/ducking -> preview -> edit -> lock -> save -> reopen -> regenerate."""

from __future__ import annotations

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import write_tone_wav
from app.tests.test_ui_phase4 import build_project

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from app.editing.models import Creator  # noqa: E402
from app.main import create_window  # noqa: E402
from app.presentation.models import PresentationType  # noqa: E402
from app.tests.test_ui_acceptance import pump, qapp, win  # noqa: E402,F401  (fixtures)

pytestmark = needs_ffmpeg


def finished(project):
    return bool(project.presentation_sessions) and project.presentation_sessions[-1].status in ("COMPLETED", "FAILED") and True


def run(panel, button, win):
    n = len(win.ws.project.presentation_sessions)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    pump(lambda: len(win.ws.project.presentation_sessions) > n and win.ws.project.presentation_sessions[-1].status in ("COMPLETED", "FAILED") and not win.ws.presentation.running)
    assert win.ws.project.presentation_sessions[-1].status == "COMPLETED", (win.ws.project.presentation_sessions[-1].error, win.ws.project.presentation_sessions[-1].validation_errors)


def test_phase5_acceptance_workflow(win, tmp_path, app_paths, qapp, monkeypatch):
    ws = win.ws
    project = build_project(win, tmp_path)
    # the AI timeline already exists (Phase 4)
    ws.editing.generate()
    pump(lambda: project.editing_sessions and project.editing_sessions[-1].status == "COMPLETED")
    lib = tmp_path / "audio"
    lib.mkdir()
    ws.presentation.import_audio(write_tone_wav(lib / "bed.wav", 40.0, 0.4, 110.0), "music")
    for n, f in (("impact", 70.0), ("tick", 1800.0), ("paper", 900.0), ("warning", 300.0)):
        ws.presentation.import_audio(write_tone_wav(lib / f"{n}.wav", 0.8, 0.5, f), "sfx", n.upper())
    pump(lambda: len(ws.presentation.library("sfx")) == 4 and len(ws.presentation.library("music")) == 1 and ws.jobs.wait_idle(0))

    panel = win.presentation_panel
    win.go_to("Audio & Captions")
    assert panel.table.rowCount() == len(project.scenes) >= 14
    assert panel.cap_enabled.isChecked() and panel.cap_keywords.isChecked() and panel.cap_numbers.isChecked() and panel.gen_all_btn.isEnabled()
    assert panel.music_lib.count() == 1 and panel.sfx_lib.count() == 4
    assert not panel.stale_box.isVisible()

    # ---- enable captions, choose Professional style, generate captions
    panel.cap_style.setCurrentIndex(panel.cap_style.findData("professional"))
    assert project.caption_settings.style_id == "professional" and project.caption_settings.enabled
    run(panel, panel.gen_captions_btn, win)
    caps = [c for c in project.timeline.all_clips() if c.kind == "caption"]
    assert caps and all(c.track_id == "track_v6" for c in caps)
    panel.refresh()
    assert panel.cap_table.rowCount() == len(caps)
    assert any(project.keyword_emphasis.values()) and any(c.text["emphasis"] for c in caps)  # keywords + numbers identified

    # ---- number graphics, lower thirds, headlines
    run(panel, panel.gen_graphics_btn, win)
    variants = {c.text.get("variant") for c in project.timeline.all_clips() if c.kind == "text"}
    assert {"NUMBER", "DATE", "LOWER_THIRD"} <= variants
    panel.refresh()
    assert panel.gfx_table.rowCount() > 0

    # ---- add music, SFX where appropriate and automatic ducking
    panel.music_lib.setCurrentRow(0)
    QTest.mouseClick(panel.add_music_btn, Qt.MouseButton.LeftButton)
    pump(lambda: project.timeline.get_track("track_a2").clips)
    run(panel, panel.gen_audio_btn, win)
    sfx = [c for c in project.timeline.all_clips() if c.audio.get("role") == "SFX"]
    assert sfx and all(c.audio["volume"] <= 0.5 for c in sfx)
    mid = project.timeline.get_track("track_a2").clips[0].metadata["assignment_id"]
    assert len(ws.presentation.ducking_keyframes(mid)) > 20 and project.ducking_events
    assert any(d.type is PresentationType.DUCKING for d in project.presentation_decisions.values())
    assert ws.presentation.masking() == []

    # ---- preview the full mix (rendered in a background job)
    panel.tabs.setCurrentIndex(3)
    panel.preview_mode.setCurrentIndex(panel.preview_mode.findData("FULL"))
    QTest.mouseClick(panel.preview_btn, Qt.MouseButton.LeftButton)
    pump(lambda: "Preview ready" in panel.preview_status.text(), timeout=60)
    assert "Full" in panel.preview_status.text()
    panel.slider.setValue(int((caps[3].timeline_start + 0.3) * 10))
    fs = panel.preview.frame
    assert fs is not None and any(l.kind == "caption" for l in fs.layers) and fs.audio["VOICE"] > 0
    assert "caption" in panel.preview_note.text() and "music" in panel.preview_note.text()
    panel.preview.grab()  # captions paint with their style

    # ---- open the timeline: edit a caption, move a graphic, change the music volume, lock a caption
    win.go_to("Timeline")
    cap = caps[3]
    ws.select_clip(cap.id)
    pump(lambda: win.inspector.pres_box.isVisible())
    assert "Caption" in win.inspector.pres_info.text() and "AI" in win.inspector.pres_info.text() and win.inspector.pres_text.isVisible()
    win.inspector.pres_text.setText("Silver is running out")
    QTest.mouseClick(win.inspector.pres_apply, Qt.MouseButton.LeftButton)
    c = project.timeline.get_clip(cap.id)
    assert c.text["text"] == "Silver is running out" and c.created_by == "USER" and project.presentation_decisions[c.ai_decision_id].created_by is Creator.USER
    QTest.mouseClick(win.inspector.pres_lock, Qt.MouseButton.LeftButton)
    assert project.timeline.get_clip(cap.id).locked
    gfx = next(c for c in project.timeline.all_clips() if c.kind == "text" and c.text.get("variant") == "NUMBER")
    win.go_to("Audio & Captions")
    panel.tabs.setCurrentIndex(1)
    panel.clip_id = gfx.id
    panel._fill_lists()
    panel.gfx_x.setValue(0.3)
    panel.gfx_y.setValue(0.6)
    QTest.mouseClick(panel.gfx_apply_btn, Qt.MouseButton.LeftButton)
    g2 = project.timeline.get_clip(gfx.id)
    assert g2.text["position"] == [0.3, 0.6] and g2.created_by == "USER"
    music_clip = project.timeline.get_track("track_a2").clips[0]
    win.go_to("Timeline")
    ws.select_clip(music_clip.id)
    pump(lambda: win.inspector.pres_volume.isVisible())
    win.inspector.pres_volume.setValue(60)
    win.inspector.pres_volume.editingFinished.emit()
    QTest.mouseClick(win.inspector.pres_apply, Qt.MouseButton.LeftButton)
    assert all(c.audio["volume"] == pytest.approx(0.6) for c in project.timeline.get_track("track_a2").clips)

    # ---- track controls
    ws.timeline.set_track_flag("track_a2", "solo", True)
    ws.timeline.set_track_volume("track_a3", 0.5)
    win.go_to("Timeline")
    assert win.timeline_panel.headers.height() > 0

    # ---- save, close, reopen: everything restored
    before = project.to_document()
    root = project.root
    assert win.save()
    win.close()
    ws.shutdown()
    win2, ws2 = create_window(app_paths)
    win2.show()
    win2.open_project(root)
    p2 = ws2.project
    for key in ("timeline", "presentation_decisions", "presentation_overrides", "presentation_plans", "ducking_events", "caption_settings", "audio_settings", "keyword_emphasis",
                "music_assignments", "sfx_assignments", "caption_segments", "text_graphics", "motion_graphics"):
        assert p2.to_document()[key] == before[key], key
    panel2 = win2.presentation_panel
    win2.go_to("Audio & Captions")
    pump(lambda: panel2.table.rowCount() == len(p2.scenes))
    assert panel2.cap_style.currentData() == "professional"
    restored = p2.timeline.get_clip(cap.id)
    assert restored.text["text"] == "Silver is running out" and restored.locked and p2.timeline.get_track("track_a2").solo
    assert panel2.cap_table.rowCount() == len([c for c in p2.timeline.all_clips() if c.kind == "caption"])

    # ---- regenerate the presentation: user edits and locked elements remain, AI-owned elements may regenerate
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.Yes)[1])
    n = len(p2.presentation_sessions)
    QTest.mouseClick(panel2.regen_all_btn, Qt.MouseButton.LeftButton)
    assert asked and "preserved" in asked[0]
    pump(lambda: len(p2.presentation_sessions) > n and p2.presentation_sessions[-1].status in ("COMPLETED", "FAILED") and not ws2.presentation.running)
    assert p2.presentation_sessions[-1].status == "COMPLETED"
    kept = p2.timeline.get_clip(cap.id)
    assert kept is not None and kept.text["text"] == "Silver is running out" and kept.locked  # USER + LOCKED caption survives
    assert p2.timeline.get_clip(gfx.id).text["position"] == [0.3, 0.6]  # moved graphic survives
    assert all(c.audio["volume"] == pytest.approx(0.6) for c in p2.timeline.get_track("track_a2").clips)  # music volume survives
    assert not [i for i in ws2.presentation.validate() if i.severity == "error"]
    ws2.close_project()
    ws2.shutdown()
    win2.close()


def test_voice_replacement_shows_the_stale_banner_and_actions(win, tmp_path, monkeypatch):
    from app.tests.helpers import make_audio

    ws = win.ws
    project = build_project(win, tmp_path)
    panel = win.presentation_panel
    win.go_to("Audio & Captions")
    run(panel, panel.gen_captions_btn, win)
    assert not panel.stale_box.isVisible()
    ws.media.import_voice_over(make_audio(tmp_path / "other.wav", 30.0))
    pump(lambda: ws.presentation.staleness()["captions_outdated"])
    panel.refresh()
    assert panel.stale_box.isVisible() and panel.stale_label.text().startswith("Voice-over changed. Captions need regeneration.")
    assert {panel.stale_regen_btn.text(), panel.stale_keep_btn.text(), panel.stale_review_btn.text()} == {"Regenerate Captions", "Keep Existing", "Review Changes"}
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    QTest.mouseClick(panel.stale_review_btn, Qt.MouseButton.LeftButton)
    assert shown and "Nothing has been changed" in shown[0]
    caps_before = len([c for c in project.timeline.all_clips() if c.kind == "caption"])
    QTest.mouseClick(panel.stale_keep_btn, Qt.MouseButton.LeftButton)
    assert "keep" in panel.stale_label.text().lower() and len([c for c in project.timeline.all_clips() if c.kind == "caption"]) == caps_before


def test_settings_controls_map_to_project_settings_and_errors_are_shown(win, tmp_path):
    ws = win.ws
    project = build_project(win, tmp_path)
    panel = win.presentation_panel
    win.go_to("Audio & Captions")
    panel.cap_position.setCurrentIndex(panel.cap_position.findData("top"))
    panel.cap_keywords.setChecked(False)
    panel.cap_numbers.setChecked(False)
    panel.cap_large.setChecked(True)
    panel.cap_reduced.setChecked(True)
    cs = project.caption_settings
    assert cs.position == "top" and not cs.keyword_highlight and not cs.number_emphasis and cs.large_text and cs.reduced_motion
    panel.music_on.setChecked(False)
    panel.ducking_on.setChecked(False)
    panel.voice_enh.setChecked(True)
    a = project.audio_settings
    assert not a.music_enabled and not a.auto_ducking and a.voice_enhancement and project.audio_processing.enabled
    ws.undo()
    ws.undo()
    panel.lvl["important_level"].setValue(90)  # louder than "normal narration": refused
    assert win.test_errors and "important" in win.test_errors[-1].lower()
    assert project.audio_settings.important_level < 0.5


def test_waveforms_are_generated_in_the_background_and_painted_on_audio_tracks(win, tmp_path):
    ws = win.ws
    project = build_project(win, tmp_path)
    ws.editing.generate()
    pump(lambda: project.editing_sessions and project.editing_sessions[-1].status == "COMPLETED")
    vo = project.voice_over.asset_id
    assert ws.presentation.waveform(vo, request=False) is None
    win.go_to("Timeline")
    canvas = win.timeline_panel.canvas
    canvas.grab()  # painting asks for the waveform; the job is started outside the paint event
    pump(lambda: ws.presentation.waveform(vo, request=False) is not None, timeout=60)
    wf = ws.presentation.waveform(vo, request=False)
    assert wf.duration == pytest.approx(project.assets.get(vo).duration, abs=0.1)
    img = canvas.grab().toImage()
    voice_row_y = canvas.row_rect_y([t.id for t in project.timeline.tracks].index("track_a1")) + 18
    voice = next(c for c in project.timeline.get_track("track_a1").clips)
    x = int(canvas.time_to_x(voice.timeline_start + 5.0))
    colours = {img.pixelColor(x + dx, voice_row_y + dy).name() for dx in range(0, 40) for dy in range(-10, 11)}
    assert len(colours) > 2  # the clip body is no longer a flat colour: the waveform is drawn
    # a broken audio file must not break painting or loop forever
    for a in project.assets.all():
        if a.type.value == "audio" and a.id != vo:
            project.asset_path(a).write_bytes(b"junk")
    canvas.grab()
    QTest.qWait(50)
    canvas.grab()
