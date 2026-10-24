"""Phase 10 end-to-end acceptance through the real main window (offscreen Qt): import media, drag it onto the timeline, move, multi-select, snap, align, trim, split, ripple delete,
undo / redo, keyframes, caption timing, audio volume, locks, save / close / reopen, preview, export and QC - with real mouse events, key presses and widget input, asserting on the model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tests.conftest import needs_ffmpeg

pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from app.analysis.models import Scene  # noqa: E402
from app.core.exceptions import TimelineError  # noqa: E402
from app.rendering.models import RenderStatus  # noqa: E402
from app.tests.test_ui_acceptance import create_project, pump, qapp, settle, wait_ms, win  # noqa: E402,F401
from app.timeline.clip import KIND_CAPTION, Clip  # noqa: E402
from app.timeline.timeline import new_clip_id  # noqa: E402
from app.ui.media_library import ASSET_MIME  # noqa: E402

pytestmark = needs_ffmpeg
NO = Qt.KeyboardModifier.NoModifier
LEFT = Qt.MouseButton.LeftButton


class Ui:
    """Small helpers over the real canvas: pixel positions of a time on a track, real drags, real key presses."""

    def __init__(self, win):  # noqa: F811
        self.win, self.ws = win, win.ws
        self.panel = win.timeline_panel
        self.canvas = self.panel.canvas

    @property
    def tl(self):
        return self.ws.project.timeline

    def row(self, track_id: str) -> int:
        return next(i for i, t in enumerate(self.canvas.tracks()) if t.id == track_id)

    def pos(self, t: float, track_id: str, frac: float = 0.5) -> QPoint:
        i = self.row(track_id)
        return QPoint(int(self.canvas.time_to_x(t)), int(self.canvas.row_rect_y(i) + self.canvas.row_height(i) * frac))

    def centre(self, clip) -> QPoint:
        return self.pos(clip.timeline_start + clip.duration / 2, clip.track_id)

    def drop(self, asset_id: str, t: float, track_id: str):
        md = QMimeData()
        md.setData(ASSET_MIME, asset_id.encode())
        ev = QDropEvent(QPointF(self.pos(t, track_id)), Qt.DropAction.CopyAction, md, LEFT, NO)
        self.canvas.dropEvent(ev)
        return ev

    def drag(self, start: QPoint, end: QPoint, mods=NO, steps: int = 6):
        QTest.mousePress(self.canvas, LEFT, mods, start)
        for k in range(1, steps + 1):
            QTest.mouseMove(self.canvas, QPoint(start.x() + (end.x() - start.x()) * k // steps, start.y() + (end.y() - start.y()) * k // steps))
        QTest.mouseRelease(self.canvas, LEFT, mods, end)
        wait_ms(30)

    def click(self, p: QPoint, mods=NO):
        QTest.mouseClick(self.canvas, LEFT, mods, p)

    def key(self, key, mods=NO):
        self.canvas.setFocus()
        QTest.keyClick(self.canvas, key, mods)
        wait_ms(30)

    def clip(self, asset_id: str, nth: int = 0):
        return sorted((c for c in self.tl.all_clips() if c.asset_id == asset_id), key=lambda c: c.timeline_start)[nth]

    def doc(self) -> str:
        return json.dumps(self.ws.project.to_document()["timeline"], sort_keys=True)


def test_professional_timeline_workflow_end_to_end(win, tmp_path, media_dir, qapp):  # noqa: F811
    ws = win.ws
    # 1. open project, import media ----------------------------------------------------------------------------------------------------------------------------------------------
    project = create_project(win, tmp_path, "Phase10")
    root = project.root
    win.library.import_paths([media_dir / "clip.mp4", media_dir / "other.mp4", media_dir / "pic.png", media_dir / "voice.wav"])
    pump(lambda: len(ws.project.assets) == 4)
    settle(ws)
    pump(lambda: win.library.list.count() == 4)
    by = {a.name: a for a in project.assets}
    win.go_to("Timeline")
    ui = Ui(win)
    fps = project.settings.fps

    # 2. drag media onto the timeline (real drop events) ---------------------------------------------------------------------------------------------------------------------------
    assert ui.drop(by["clip.mp4"].id, 0.0, "track_v1").isAccepted()
    assert ui.drop(by["other.mp4"].id, 5.0, "track_v1").isAccepted()
    assert ui.drop(by["pic.png"].id, 1.0, "track_v2").isAccepted()
    assert ui.drop(by["voice.wav"].id, 0.0, "track_a1").isAccepted()
    assert not ui.drop(by["voice.wav"].id, 0.0, "track_v1").isAccepted()  # audio onto a video track is refused, nothing is added
    assert len(list(ui.tl.all_clips())) == 4
    main, other, pic, voice = (ui.clip(by[n].id) for n in ("clip.mp4", "other.mp4", "pic.png", "voice.wav"))
    ws.commands.clear()  # the history starts here: the rest of the workflow is judged on its own undo steps
    ws.edit.state.update(linked_selection=False)

    # 3. move a clip with the mouse ---------------------------------------------------------------------------------------------------------------------------------------------
    dx = int(ui.canvas.time_to_x(1.0) - ui.canvas.time_to_x(0.0))
    start = ui.centre(main)
    ui.drag(start, QPoint(start.x() + dx, start.y()))
    main = ui.tl.get_clip(main.id)
    assert main.timeline_start == pytest.approx(1.0, abs=2 / fps) and main.track_id == "track_v1"
    assert ws.commands.undo_text.lower().startswith("move")
    n_undo = 0
    while ws.commands.can_undo:
        ws.undo()
        n_undo += 1
    assert n_undo == 1 and ui.tl.get_clip(main.id).timeline_start == 0.0  # the whole drag was ONE undo step
    for _ in range(n_undo):
        ws.redo()

    # 4. multi-select (Ctrl+click) and move the selection: relative timing preserved, one undo step ------------------------------------------------------------------------------
    main, pic = ui.tl.get_clip(main.id), ui.tl.get_clip(pic.id)
    gap_before = pic.timeline_start - main.timeline_start
    ui.click(ui.centre(main))
    ui.click(ui.centre(pic), Qt.KeyboardModifier.ControlModifier)
    assert set(ws.selection.ids) == {main.id, pic.id}
    before_move = ui.doc()
    dx2 = int(ui.canvas.time_to_x(1.0) - ui.canvas.time_to_x(0.0))
    start = ui.centre(pic)
    ui.drag(start, QPoint(start.x() + dx2, start.y()))
    main2, pic2 = ui.tl.get_clip(main.id), ui.tl.get_clip(pic.id)
    assert main2.timeline_start == pytest.approx(main.timeline_start + 1.0, abs=2 / fps) and pic2.timeline_start - main2.timeline_start == pytest.approx(gap_before, abs=1e-6)
    ws.undo()
    assert ui.doc() == before_move  # both clips came back with one undo
    ws.redo()
    main, pic = ui.tl.get_clip(main.id), ui.tl.get_clip(pic.id)

    # 5. snapping: a marker at 10 s pulls the end of the second video onto it exactly -----------------------------------------------------------------------------------------------
    ui.canvas.set_playhead(10.0)
    ui.key(Qt.Key.Key_M)  # the Add-marker shortcut, from the keyboard
    assert [round(m.time, 3) for m in ui.tl.markers] == [10.0]
    other = ui.tl.get_clip(other.id)
    start = ui.centre(other)
    near = 10.0 - other.timeline_end - 0.03  # stop a few pixels short of the marker
    ui.drag(start, QPoint(start.x() + int(near * ui.canvas.pps), start.y()))
    other = ui.tl.get_clip(other.id)
    assert other.timeline_end == pytest.approx(10.0, abs=1e-6)  # snapped exactly (not "close to")

    # 6. align to a scene boundary --------------------------------------------------------------------------------------------------------------------------------------------------
    ws.project.scenes = [Scene("s1", "1", 0.0, 3.5), Scene("s2", "2", 3.5, 12.0)]
    ws.selection.select([pic.id])
    ws.edit.align(None, "start", "scene")
    assert ui.tl.get_clip(pic.id).timeline_start == pytest.approx(3.5, abs=1 / fps)

    # 7. trim with the mouse (right edge of the first video) ---------------------------------------------------------------------------------------------------------------------------
    main = ui.tl.get_clip(main.id)
    edge = QPoint(int(ui.canvas.time_to_x(main.timeline_end)) - 3, ui.centre(main).y())
    dur_before, src_before = main.duration, main.source_out
    ui.drag(edge, QPoint(edge.x() - int(0.5 * ui.canvas.pps), edge.y()))
    main = ui.tl.get_clip(main.id)
    assert main.duration == pytest.approx(dur_before - 0.5, abs=3 / fps) and main.source_out < src_before and main.source_out - main.source_in == pytest.approx(main.duration * main.speed, abs=1e-3)

    # 8. split with the blade tool (B key, click) ----------------------------------------------------------------------------------------------------------------------------------
    n_clips = len(ui.tl.get_track("track_v1").clips)
    ui.key(Qt.Key.Key_B)
    assert ws.edit_state.tool == "blade"
    cut_at = main.timeline_start + main.duration / 2
    ui.click(ui.pos(cut_at, "track_v1"))
    clips = ui.tl.get_track("track_v1").clips
    assert len(clips) == n_clips + 1
    left, right = clips[0], clips[1]
    assert left.timeline_end == pytest.approx(cut_at, abs=1 / fps) and right.source_in == pytest.approx(left.source_out, abs=1e-6) and right.id != left.id
    ui.key(Qt.Key.Key_V)
    assert ws.edit_state.tool == "select"

    # 9. ripple delete ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    other = ui.tl.get_clip(other.id)
    right = ui.tl.get_clip(right.id)
    followers = {c.id: c.timeline_start for c in ui.tl.get_track("track_v1").clips if c.timeline_start >= right.timeline_end - 1e-6}
    assert other.id in followers
    ui.click(ui.centre(right))
    QTest.mouseClick(ui.panel.ripple_btn, LEFT)  # the toolbar toggle: Ripple ON is visible
    assert ws.edit_state.ripple_on and "ON" in ui.panel.ripple_btn.text()
    ui.key(Qt.Key.Key_Delete)
    assert ui.tl.get_clip(right.id) is None
    for cid, was in followers.items():
        assert ui.tl.get_clip(cid).timeline_start == pytest.approx(was - right.duration, abs=1 / fps)
    QTest.mouseClick(ui.panel.ripple_btn, LEFT)
    assert not ws.edit_state.ripple_on

    # 10. undo / redo through the real actions -------------------------------------------------------------------------------------------------------------------------------------
    after_ripple = ui.doc()
    win.undo_action.trigger()
    assert ui.tl.get_clip(right.id) is not None and ui.doc() != after_ripple
    win.redo_action.trigger()
    assert ui.doc() == after_ripple

    # 11. keyframes in the inspector's lane (click to add, drag to move) -----------------------------------------------------------------------------------------------------------
    pic = ui.tl.get_clip(pic.id)
    ws.selection.select([pic.id])
    wait_ms(60)
    kf = win.inspector.kf
    assert kf.available
    kf.prop_combo.setCurrentIndex(kf.prop_combo.findData("opacity"))
    assert kf.prop() == "opacity"
    lane = kf.lane
    QTest.mouseClick(lane, LEFT, NO, lane.pos_for(pic.duration * 0.25, 0.5))
    QTest.mouseClick(lane, LEFT, NO, lane.pos_for(pic.duration * 0.75, 1.0))
    pic = ui.tl.get_clip(pic.id)
    assert len(pic.keyframes) == 2 and {k.property for k in pic.keyframes} == {"opacity"}
    ws.undo()
    assert len(ui.tl.get_clip(pic.id).keyframes) == 1
    ws.redo()

    # 12. caption timing: drag a caption, its word timing follows ------------------------------------------------------------------------------------------------------------------
    words = [{"word_id": f"w{i}", "text": w, "start": 6.0 + i * 0.4, "end": 6.0 + i * 0.4 + 0.36} for i, w in enumerate("Silver is running low".split())]
    cap = Clip(new_clip_id(), "track_v6", "", 6.0, 1.6, kind=KIND_CAPTION, text={"caption_id": "cap_x", "scene_id": "", "start": 6.0, "end": 7.6, "text": "Silver is running low", "lines": ["Silver is running low"],
                                                                               "words": words, "emphasis": [], "style_id": "professional", "style_overrides": {}, "position": "bottom",
                                                                               "position_xy": [], "highlight_mode": "HIGHLIGHT", "reading_cps": 12.0})
    ui.tl.get_track("track_v6").clips.append(cap)
    ui.tl.get_track("track_v6").sort()
    ui.panel.reload()
    wait_ms(60)
    start = ui.centre(cap)
    ui.drag(start, QPoint(start.x() + int(0.5 * ui.canvas.pps), start.y()))
    cap2 = ui.tl.get_clip(cap.id)
    shift = cap2.timeline_start - 6.0
    assert shift == pytest.approx(0.5, abs=2 / fps)
    assert [round(w["start"] - o["start"], 3) for w, o in zip(cap2.text["words"], words)] == [round(shift, 3)] * 4 and cap2.text["start"] == pytest.approx(cap2.timeline_start)

    # 13. audio volume in the inspector --------------------------------------------------------------------------------------------------------------------------------------------
    voice = ui.tl.get_clip(voice.id)
    ws.selection.select([voice.id])
    wait_ms(60)
    ins = win.inspector
    ins.volume.selectAll()
    QTest.keyClicks(ins.volume, "50")
    QTest.keyClick(ins.volume, Qt.Key.Key_Return)
    wait_ms(60)
    assert ui.tl.get_clip(voice.id).audio.get("volume") == pytest.approx(0.5)
    ws.undo()
    assert ui.tl.get_clip(voice.id).audio.get("volume", 1.0) == pytest.approx(1.0)
    ws.redo()

    # 14. lock a clip, attempt a locked operation -> safe rejection ----------------------------------------------------------------------------------------------------------------
    main = ui.tl.get_clip(main.id)
    ui.click(ui.centre(main))
    ws.edit.toggle_item_lock([main.id])
    assert ui.tl.get_clip(main.id).protected
    before_locked = ui.doc()
    undo_text_before = ws.commands.undo_text
    messages: list[str] = []
    ws.bus.subscribe("app.status", lambda _t, p: messages.append(p.get("message", "")))
    start = ui.centre(main)
    ui.drag(start, QPoint(start.x() + int(0.3 * ui.canvas.pps), start.y()))
    assert ui.doc() == before_locked  # nothing moved
    assert any("lock" in m.lower() for m in messages), messages  # and the user was told why
    with pytest.raises(TimelineError, match="(?i)lock"):
        ws.edit.trim(main.id, "end", main.timeline_end - 0.1)
    assert ws.commands.undo_text == undo_text_before  # no undo entry was created by the refused attempts
    ws.edit.toggle_item_lock([main.id])
    assert not ui.tl.get_clip(main.id).protected

    # 15. save, close, reopen, verify -----------------------------------------------------------------------------------------------------------------------------------------------
    final = ui.doc()
    win.save_action.trigger()
    settle(ws)
    ws.close_project()
    win.open_project(root)
    pump(lambda: ws.project is not None)
    assert json.dumps(ws.project.to_document()["timeline"], sort_keys=True) == final
    assert [round(m.time, 3) for m in ws.project.timeline.markers] == [10.0]
    assert not ws.commands.can_undo  # a fresh session

    # 16. preview: the AI Edit page's timeline preview shows the frame the model says ---------------------------------------------------------------------------------------
    win.go_to("AI Edit")
    first = ui.clip(by["clip.mp4"].id, 0)  # reopened: look the clip up again
    t_probe = first.timeline_start + 0.2
    frame = win.ai_edit_panel.preview.show_time(t_probe)
    assert frame is not None
    layers = [layer for layer in frame.layers if layer.kind == "media" and layer.asset_id == by["clip.mp4"].id]
    assert layers and layers[0].src_time == pytest.approx(first.source_in + 0.2 * first.speed, abs=2 / fps)  # the preview shows the source moment the model maps this time to
    wait_ms(30)

    # 17. render export, compare with the timeline, run QC -------------------------------------------------------------------------------------------------------------------------
    ws.project.scenes = [Scene("s1", "1", 0.0, ws.project.timeline.duration)]
    ws.render.update_settings(resolution="480p", quality="draft", preset_id="draft")
    job = ws.render.start_export(Path(tmp_path) / "out" / "phase10.mp4")
    pump(lambda: job.finished_event.is_set(), timeout=180)
    assert job.status is RenderStatus.COMPLETED, job.error
    info = ws.render.engine.probe.probe(Path(job.record.output_path))
    assert info.duration == pytest.approx(ws.project.timeline.duration, abs=0.25)
    qc_job = ws.qc.run_full_qc(force=True)
    pump(lambda: ws.jobs.wait_idle(0.0), timeout=180)
    assert qc_job is None or qc_job.error is None
    assert ws.project.qc_runs and ws.project.qc_runs[-1].get("content_hash")
