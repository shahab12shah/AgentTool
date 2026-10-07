"""Phase 4 acceptance workflow through the real main window: AI edit -> inspect -> change -> lock -> save -> reopen -> regenerate."""

from __future__ import annotations

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import NARRATION, ScriptedProvider, make_audio, make_image, make_video

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QMessageBox, QPushButton  # noqa: E402

from app.editing.models import Creator, DecisionType, SceneEditStatus  # noqa: E402
from app.main import create_window  # noqa: E402
from app.project.phase3_commands import SceneDecisionCommand  # noqa: E402
from app.research.models import Acquisition, VisualAssignment  # noqa: E402
from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)

pytestmark = needs_ffmpeg


def build_project(win, tmp_path):
    from app.analysis.segmenter import SegmentationParams

    ws = win.ws
    project = create_project(win, tmp_path, "Phase4")
    voice = ScriptedProvider(NARRATION)
    ws.transcripts.registry.register(voice)
    ws.transcripts.resolve_provider = lambda name=None: voice
    ws.set_script(NARRATION)
    win.voice_panel.import_path(make_audio(tmp_path / "vo.wav", voice.words[-1].end + 1.0))
    pump(lambda: project.voice_over.asset_id is not None and ws.jobs.wait_idle(0))
    ws.transcripts.transcribe(provider_name=voice.name)
    pump(lambda: project.transcription.transcript is not None and ws.jobs.wait_idle(0))
    ws.scenes.analyze(params=SegmentationParams(threshold=0.6, min_scene_seconds=2.0))
    pump(lambda: len(project.scenes) > 0 and ws.jobs.wait_idle(0))
    d = tmp_path / "vis"
    d.mkdir()
    files = [make_video(d / "v.mp4", 12.0), make_image(d / "wide.png", "testsrc", "1600x600"), make_image(d / "portrait.png", "testsrc2", "600x900"),
             make_image(d / "plain.png", "gradient", "1280x720"), make_image(d / "doc.png", "mandel", "800x1000")]
    ws.media.import_files(files)
    pump(lambda: len([a for a in project.assets.all() if a.type.value != "audio"]) == 5 and ws.jobs.wait_idle(0))
    by = {a.name: a for a in project.assets.all()}
    pool = [by["v.mp4"], by["wide.png"], by["portrait.png"], by["plain.png"]]
    for i, sc in enumerate(project.scenes):
        evid = project.visual_intents[sc.id].type.value == "EVIDENCE"
        asset = by["doc.png"] if evid else pool[i % len(pool)]
        ws.apply_command(SceneDecisionCommand(project, sc.id, "assign", assignment=VisualAssignment(
            sc.id, None, asset.id, "USER", 90.0, True, False, acquisition=Acquisition.LOCAL, source_type=asset.source_type)))
    return project


def spin(panel, path) -> QDoubleSpinBox:
    return panel.inspector.findChild(QDoubleSpinBox, f"param_{path}")


def select_decision(panel, decision_id):
    panel.select_decision(decision_id)
    assert panel.decision_id == decision_id


def test_phase4_acceptance_workflow(win, tmp_path, app_paths, qapp, monkeypatch):
    ws = win.ws
    project = build_project(win, tmp_path)
    panel = win.ai_edit_panel
    win.go_to("AI Edit")
    assert panel.table.rowCount() == len(project.scenes) >= 14
    assert panel.generate_btn.isEnabled() and "No AI edit yet" in panel.state_label.text()
    assert all(panel.table.item(r, 2).text() == "Approved" and panel.table.item(r, 3).text() == "Pending" for r in range(panel.table.rowCount()))

    # ---- choose Professional, set sliders/toggles, generate
    panel.style.setCurrentIndex(panel.style.findData("professional"))
    panel.motion.setValue(55)
    panel.pacing.setValue(50)
    assert project.editing_settings.style == "professional" and project.editing_settings.motion_intensity == 0.55
    for cb in (panel.text_emphasis, panel.number_emphasis, panel.evidence, panel.smart_transitions):
        assert cb.isChecked()
    QTest.mouseClick(panel.generate_btn, Qt.MouseButton.LeftButton)
    pump(lambda: project.editing_sessions and project.editing_sessions[-1].status in ("COMPLETED", "FAILED"))
    session = project.editing_sessions[-1]
    assert session.status == "COMPLETED", (session.error, session.validation_errors)
    panel.refresh()
    assert "Edit complete" in panel.state_label.text() and any("Scene" in l for l in panel.log.toPlainText().splitlines())
    assert all(panel.table.item(r, 3).text() == "Edited ✓" for r in range(panel.table.rowCount()))
    assert project.timeline_version == 1 and len(project.timeline.all_clips()) > len(project.scenes)

    # ---- the AI timeline is the normal timeline: undo reverts the whole edit in one step
    ws.undo()
    assert not project.timeline.all_clips() and not project.editing_decisions
    ws.redo()
    assert project.timeline.all_clips() and project.editing_decisions

    # ---- preview: scrubbing shows visuals, text and music instructions without rendering
    sc_text = next(c for c in project.timeline.all_clips() if c.kind == "text")
    panel.slider.setValue(int((sc_text.timeline_start + 0.4) * 10))
    fs = panel.preview.frame
    assert fs is not None and any(l.kind == "media" for l in fs.layers) and any(l.kind == "text" for l in fs.layers)
    assert "visual layer" in panel.preview_note.text() and "music" in panel.preview_note.text()
    panel.preview.grab()  # painting works (frames are extracted on demand)

    # ---- select a scene with motion: the AI Decision Inspector shows the real parameters
    d = next(x for x in project.editing_decisions.values() if x.type in (DecisionType.ZOOM, DecisionType.PAN)
             and len([c for c in project.timeline.all_clips() if c.scene_id == x.scene_id and c.kind == "media"]) == 1)
    sid = d.scene_id
    row = [r for r in range(panel.table.rowCount()) if panel.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == sid][0]
    panel.table.selectRow(row)
    assert panel.scene_id == sid and panel.decisions.rowCount() >= 3
    select_decision(panel, d.decision_id)
    info = panel.i_info.text()
    assert d.type.value.replace("_", " ").title() in info and "AI" in info and f"{d.confidence:.0f}%" in info and d.reason in info
    end_scale = spin(panel, "end_scale")
    assert end_scale is not None and end_scale.value() == pytest.approx(d.parameters["end_scale"])

    # ---- user changes the zoom
    end_scale.setValue(1.25)
    QTest.mouseClick(panel.apply_btn, Qt.MouseButton.LeftButton)
    user_zoom = next(x for x in project.editing_decisions.values() if x.scene_id == sid and x.type in (DecisionType.ZOOM, DecisionType.PAN))
    assert user_zoom.created_by is Creator.USER and user_zoom.overrides_decision_id == d.decision_id and user_zoom.parameters["end_scale"] == 1.25
    assert "USER" in panel.i_info.text()

    # ---- user changes the visual duration
    vt = next(x for x in panel.ctx.ws.editing.decisions_for_scene(sid) if x.type is DecisionType.VISUAL_TIMING)
    select_decision(panel, vt.decision_id)
    dur = spin(panel, "duration")
    old = dur.value()
    dur.setValue(old - 0.5)
    QTest.mouseClick(panel.apply_btn, Qt.MouseButton.LeftButton)
    clip = next(c for c in project.timeline.all_clips() if c.scene_id == sid and c.kind == "media")
    assert clip.duration == pytest.approx(old - 0.5) and clip.created_by == "USER"
    # ---- an impossible change is refused with a message, nothing changes
    errors_before = list(win.test_errors)
    select_decision(panel, next(x for x in ws.editing.decisions_for_scene(sid) if x.type is DecisionType.VISUAL_TIMING).decision_id)
    spin(panel, "duration").setValue(0.05)
    spin(panel, "duration").setValue(0.0)
    QTest.mouseClick(panel.apply_btn, Qt.MouseButton.LeftButton)
    assert next(c for c in project.timeline.all_clips() if c.id == clip.id).duration == pytest.approx(old - 0.5)
    assert len(win.test_errors) == len(errors_before) + 1 and "positive duration" in win.test_errors[-1]

    # ---- lock the visual
    QTest.mouseClick(panel.lock_visual, Qt.MouseButton.LeftButton)
    clip = next(c for c in project.timeline.all_clips() if c.scene_id == sid and c.kind == "media")
    assert clip.locked and panel.lock_visual.isChecked()
    locked_id, locked_dur = clip.id, clip.duration

    # ---- timeline selection opens the same decision; the timeline inspector shows the AI box and lock state
    win.go_to("Timeline")
    ws.select_clip(locked_id)
    pump(lambda: win.inspector.ai_box.isVisible())
    assert win.inspector.ai_lock.isChecked() and "created by USER" in win.inspector.ai_info.text()
    win.go_to("AI Edit")

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
    for key in ("timeline", "editing_decisions", "editing_strategy", "timeline_generation", "ai_overrides", "timeline_version", "editing_settings"):
        assert p2.to_document()[key] == before[key], key
    panel2 = win2.ai_edit_panel
    win2.go_to("AI Edit")
    pump(lambda: panel2.table.rowCount() == len(p2.scenes))
    assert panel2.style.currentData() == "professional" and panel2.motion.value() == 55
    row = [r for r in range(panel2.table.rowCount()) if panel2.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == sid][0]
    panel2.table.selectRow(row)
    assert panel2.lock_visual.isChecked() and any(panel2.decisions.item(r, 4).text() == "USER" for r in range(panel2.decisions.rowCount()))

    # ---- regenerate the scene: the locked visual and the user's zoom survive
    QTest.mouseClick(panel2.regen_scene_btn, Qt.MouseButton.LeftButton)
    pump(lambda: p2.editing_sessions[-1].status in ("COMPLETED", "FAILED") and not ws2.editing.running)
    assert p2.editing_sessions[-1].status == "COMPLETED"
    after = next(c for c in p2.timeline.all_clips() if c.id == locked_id)
    assert after.locked and after.duration == pytest.approx(locked_dur)
    zoom = [x for x in ws2.editing.decisions_for_scene(sid) if x.type in (DecisionType.ZOOM, DecisionType.PAN)]
    assert zoom and zoom[0].created_by is Creator.USER and zoom[0].parameters["end_scale"] == 1.25
    assert max(k.value for k in after.keyframes if k.property == "scale") == pytest.approx(1.25)

    # ---- a scene without an approved visual shows the recovery options (nothing is substituted)
    other = p2.scenes[2].id
    del p2.visual_assignments[other]
    panel2.refresh()
    panel2.table.selectRow(2)
    assert panel2.missing_box.isVisible() and "Nothing is substituted" in panel2.missing_label.text()
    assert {panel2.return_btn.text(), panel2.manual_btn.text(), panel2.skip_btn.text()} == {"Return to Research", "Replace Manually…", "Skip Visual"}
    QTest.mouseClick(panel2.return_btn, Qt.MouseButton.LeftButton)
    assert win2.pages.currentIndex() == win2.page_index["Review"] and win2.review_panel.scene_id == other

    # ---- regenerate everything: confirmation, user work preserved
    win2.go_to("AI Edit")
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.Yes)[1])
    QTest.mouseClick(panel2.regen_all_btn, Qt.MouseButton.LeftButton)
    assert asked and "preserved" in asked[0]
    pump(lambda: p2.editing_sessions[-1].status in ("COMPLETED", "FAILED") and not ws2.editing.running)
    assert any(c.id == locked_id and c.locked for c in p2.timeline.all_clips())
    ws2.close_project()
    ws2.shutdown()
    win2.close()


def test_failed_scene_is_shown_and_retry_resumes(win, tmp_path):
    from app.editing.strategy import RuleBasedProvider

    ws = win.ws
    project = build_project(win, tmp_path)
    fail_id = project.scenes[5].id

    class Flaky(RuleBasedProvider):
        name, label = "flaky", "Flaky"
        broken = True

        def plan_scene(self, sc, ctx, profile):
            if sc.scene.id == fail_id and Flaky.broken:
                raise RuntimeError("model unavailable")
            return super().plan_scene(sc, ctx, profile)

    ws.editing.strategy.register(Flaky())
    ws.editing.update_settings(provider="flaky")
    panel = win.ai_edit_panel
    win.go_to("AI Edit")
    QTest.mouseClick(panel.generate_btn, Qt.MouseButton.LeftButton)
    pump(lambda: project.editing_sessions and project.editing_sessions[-1].status in ("COMPLETED", "FAILED"))
    panel.refresh()
    texts = [panel.table.item(r, 3).text() for r in range(panel.table.rowCount())]
    assert texts[:5] == ["Edited ✓"] * 5 and texts[5] == "FAILED" and set(texts[6:]) == {"Pending"}
    assert "model unavailable" in panel.state_label.text() and panel.retry_btn.isEnabled()
    Flaky.broken = False
    QTest.mouseClick(panel.retry_btn, Qt.MouseButton.LeftButton)
    pump(lambda: project.editing_sessions[-1].status in ("COMPLETED", "FAILED") and not ws.editing.running)
    panel.refresh()
    assert project.editing_sessions[-1].status == "COMPLETED" and all(panel.table.item(r, 3).text() == "Edited ✓" for r in range(panel.table.rowCount()))
    assert not panel.retry_btn.isEnabled()
    _ = (QCheckBox, QComboBox, QPushButton, SceneEditStatus)
