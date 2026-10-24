"""Phase 10 (agent U2): the timeline panel - toolbar, track headers, configurable shortcuts, Timeline menu, play transport.

Everything is driven through real widgets (offscreen Qt): mouse clicks / drags on the header column and toolbar buttons, key presses on the focused canvas, and asserted on the MODEL
(``ws.project.timeline``), the editing state and the undo stack - never on pixels (except that the header rows are painted where the canvas' ``TrackLayout`` says they are).

The shortcuts call the canvas' keyboard API (``nudge``, ``step_playhead``, ``jump_to_edge``, ``jump_to_marker``, ``delete_selection``, ``copy_selection`` ... - see ``ui_canvas.md``): these tests run
against the REAL canvas, so they also pin that contract.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.tests.conftest import import_and_wait, needs_ffmpeg  # noqa: F401  (needs_ffmpeg is the module mark below)

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit, QMessageBox, QWidget  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.main import create_window  # noqa: E402
from app.storage.paths import AppPaths  # noqa: E402
from app.tests.helpers import make_image, make_video, write_tone_wav  # noqa: E402
from app.tests.test_ui_acceptance import pump  # noqa: E402
from app.timeline.snapping import SnapSettings  # noqa: E402
from app.timeline.timebase import timebase_of  # noqa: E402
from app.timeline.timeline import Timeline  # noqa: E402
from app.timeline.track import TrackKind  # noqa: E402
from app.ui.dialogs.shortcuts_dialog import ShortcutsDialog  # noqa: E402
from app.ui.timeline_panel import HEADER_W  # noqa: E402
from app.ui.timeline_shortcuts import SPECS, ShortcutConflict  # noqa: E402

pytestmark = needs_ffmpeg


# ------------------------------------------------------------------------------------------------------------------------ fixtures
class Env:
    pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def env(qapp, tmp_path_factory):
    root = tmp_path_factory.mktemp("u2")
    mp = pytest.MonkeyPatch()
    errors: list[str] = []
    mp.setattr("app.ui.dialogs.message.show_error", lambda parent, message, details=None, title="": errors.append(message))
    mp.setattr("app.ui.main_window.show_error", lambda parent, message, details=None, title="": errors.append(message))
    e = Env()
    e.paths = AppPaths(root / "cfg", root / "data")
    e.win, e.ws = create_window(e.paths)
    e.errors, e.status = errors, []
    e.ws.bus.subscribe("app.status", lambda topic, p: e.status.append(p.get("message", "")))
    e.win.show()
    e.ws.new_project("U2", root / "projects")
    media = root / "media"
    media.mkdir()
    got = []
    files = [make_video(media / "a.mp4", 6.0), make_video(media / "b.mp4", 4.0, "testsrc2"), make_video(media / "c.mp4", 5.0), make_image(media / "p.png"), write_tone_wav(media / "tone.wav", 8.0)]
    e.ws.media.import_files(files, on_asset=got.append)
    pump(lambda: len(got) == len(files), timeout=60)
    by_name = {a.name: a for a in got}
    e.A, e.B, e.C, e.IMG, e.AUD = (by_name[n] for n in ("a.mp4", "b.mp4", "c.mp4", "p.png", "tone.wav"))
    e.win.go_to("Timeline")
    e.panel = e.win.timeline_panel
    for _ in range(5):
        QApplication.processEvents()
    yield e
    e.ws.close_project()
    e.ws.shutdown()
    e.win.close()
    e.win.deleteLater()
    QApplication.processEvents()
    mp.undo()


def _reset(e: Env) -> None:
    ws, panel = e.ws, e.panel
    panel.transport.stop()
    panel.transport.clock = __import__("time").monotonic
    ws.commands.clear()
    ws.selection.select_none()
    ws.project.timeline = Timeline.default()
    ws.edit.clipboard = None
    ws.edit._attrs = None
    ws.edit.playhead = 0.0
    ws.edit_state.update(tool="select", ripple_on=False, ripple_scope="track", snap=SnapSettings(), linked_selection=True, follow_playhead="page", auto_scroll=True)
    ws.settings = replace(ws.settings, timeline_shortcuts={})
    panel.shortcuts.apply()
    panel.selected_provider = lambda: []
    e.win.library.search.clear()  # a test that typed into the library's search box must not filter the next test's library
    panel.canvas.set_zoom(80.0)
    panel.reload()
    panel.canvas.set_playhead(0.0)
    e.errors.clear()
    e.status.clear()
    QApplication.processEvents()


@pytest.fixture
def e(env):
    _reset(env)
    env.win.go_to("Timeline")
    env.panel.canvas.setFocus()
    yield env
    _reset(env)


def flush() -> None:
    """Let the panel's coalesced reload (a burst of change events inside one event-loop turn folds into one trailing reload) run."""
    for _ in range(3):
        QApplication.processEvents()


def focus(w: QWidget) -> None:
    w.setFocus()
    pump(lambda: QApplication.focusWidget() is w, timeout=5)


def press(widget: QWidget, seq: str) -> None:
    kc = QKeySequence(seq, QKeySequence.SequenceFormat.PortableText)[0]
    QTest.keyClick(widget, kc.key(), kc.keyboardModifiers())
    QApplication.processEvents()


def tl(e):
    return e.ws.project.timeline


def lay(e, n=3, track="track_v1", at=0.0):
    """n media clips back to back on a video track; returns their clip ids."""
    assets = [e.A, e.B, e.C, e.IMG][:n]
    e.ws.edit.place_assets([a.id for a in assets], at=at, track_id=track)
    return [c.id for c in tl(e).get_track(track).clips]


def row_of(e, track_id: str) -> int:
    return next(i for i, t in enumerate(tl(e).tracks) if t.id == track_id)


def track_ids(e) -> list[str]:
    return [t.id for t in tl(e).tracks]


def undo_text(e) -> str:
    return e.ws.commands.undo_text


def snapshot(e) -> str:
    import json

    return json.dumps(tl(e).to_dict(), sort_keys=True)


# ------------------------------------------------------------------------------------------------------------------------ settings field
def test_settings_field_is_sanitised_and_round_trips(tmp_path):
    s = Settings()
    assert s.timeline_shortcuts == {}
    s.timeline_shortcuts = "oops"  # type: ignore[assignment]
    assert s.sanitized().timeline_shortcuts == {}
    s.timeline_shortcuts = {"split": ["Ctrl+K"]}
    assert s.sanitized().timeline_shortcuts == {"split": ["Ctrl+K"]}


# ------------------------------------------------------------------------------------------------------------------------ toolbar: modes round-trip, persistence
def test_tool_buttons_and_edit_state_stay_in_sync_both_ways(e):
    p, state = e.panel, e.ws.edit_state
    assert p.select_btn.isChecked() and not p.blade_btn.isChecked()
    QTest.mouseClick(p.blade_btn, Qt.MouseButton.LeftButton)
    assert state.tool == "blade" and p.blade_btn.isChecked() and not p.select_btn.isChecked()
    QTest.mouseClick(p.blade_btn, Qt.MouseButton.LeftButton)  # clicking the active tool again must not leave "no tool"
    assert state.tool == "blade" and p.blade_btn.isChecked()
    state.update(tool="select")  # changed from somewhere else (a shortcut, another panel)
    assert p.select_btn.isChecked() and not p.blade_btn.isChecked()


def test_snap_link_ripple_toggles_round_trip_with_the_state(e):
    p, state = e.panel, e.ws.edit_state
    assert state.snap.enabled and p.snap_btn.isChecked()
    QTest.mouseClick(p.snap_btn, Qt.MouseButton.LeftButton, pos=QPoint(8, p.snap_btn.height() // 2))  # the left part toggles, the arrow opens the menu
    assert not state.snap.enabled and not p.snap_btn.isChecked()
    state.toggle_snapping()
    assert p.snap_btn.isChecked()
    QTest.mouseClick(p.link_btn, Qt.MouseButton.LeftButton)
    assert not state.linked_selection and not p.link_btn.isChecked()
    state.toggle_linked_selection()
    assert p.link_btn.isChecked()
    QTest.mouseClick(p.ripple_btn, Qt.MouseButton.LeftButton)
    assert state.ripple_on and p.ripple_btn.isChecked()
    state.toggle_ripple()
    assert not p.ripple_btn.isChecked()


def test_ripple_state_is_unmistakable(e):
    p, state = e.panel, e.ws.edit_state
    assert p.ripple_btn.text() == "Ripple OFF" and "Ripple off" in p.status_line.text()
    state.update(ripple_on=True)
    assert p.ripple_btn.text() == "Ripple ON" and p.ripple_btn.isChecked()
    assert "Ripple ON" in p.status_line.text() and "this track" in p.status_line.text()
    assert "ripple edit" in p.scope_combo.toolTip()  # the scope combo explains what it controls
    state.update(ripple_scope="all")
    assert "all tracks" in p.status_line.text()
    assert "Ctrl" not in p.ripple_btn.toolTip() and "Shift+R" in p.ripple_btn.toolTip()  # the tooltip names the current key


def test_ripple_scope_combo_writes_the_state_and_follows_it(e):
    p, state = e.panel, e.ws.edit_state
    assert [p.scope_combo.itemData(i) for i in range(p.scope_combo.count())] == ["track", "tracks", "all"]
    p.scope_combo.setCurrentIndex(2)
    assert state.ripple_scope == "all"
    p.scope_combo.setCurrentIndex(1)
    assert state.ripple_scope == "tracks"
    state.update(ripple_scope="track")
    assert p.scope_combo.currentData() == "track"


def test_snap_menu_sets_tolerance_and_targets(e, monkeypatch):
    p, state = e.panel, e.ws.edit_state
    assert p.tolerance_actions[4].isChecked()
    p.tolerance_actions[8].trigger()
    assert state.snap.tolerance_frames == 8 and p.tolerance_actions[8].isChecked() and not p.tolerance_actions[4].isChecked()
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: (12, True)))
    p.custom_tolerance_action.trigger()
    assert state.snap.tolerance_frames == 12 and p.custom_tolerance_action.isChecked() and "12" in p.custom_tolerance_action.text()
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: (99, False)))  # cancelled: unchanged
    p.custom_tolerance_action.trigger()
    assert state.snap.tolerance_frames == 12
    assert p.target_actions["scenes"].isChecked() and not p.target_actions["words"].isChecked()
    p.target_actions["scenes"].trigger()
    p.target_actions["words"].trigger()
    assert "scenes" not in state.snap.targets and "words" in state.snap.targets
    state.update(snap=SnapSettings())
    assert p.target_actions["scenes"].isChecked() and not p.target_actions["words"].isChecked() and p.tolerance_actions[4].isChecked()


def test_follow_combo_and_auto_scroll_checkbox(e):
    p, state = e.panel, e.ws.edit_state
    assert [p.follow_combo.itemData(i) for i in range(p.follow_combo.count())] == ["off", "page", "smooth", "center"]
    for i, mode in enumerate(("off", "page", "smooth", "center")):
        p.follow_combo.setCurrentIndex(i)
        assert state.follow_playhead == mode
    state.update(follow_playhead="page")
    assert p.follow_combo.currentData() == "page"
    assert p.autoscroll_box.isChecked()
    p.autoscroll_box.click()
    assert state.auto_scroll is False
    state.update(auto_scroll=True)
    assert p.autoscroll_box.isChecked()


def test_edit_state_preferences_persist_across_a_workspace_restart(e):
    from app.services.workspace import Workspace

    e.ws.edit_state.update(ripple_on=True, ripple_scope="all", follow_playhead="center", auto_scroll=False, linked_selection=False)
    QTest.mouseClick(e.panel.snap_btn, Qt.MouseButton.LeftButton, pos=QPoint(8, e.panel.snap_btn.height() // 2))  # snapping off
    e.panel.tolerance_actions[8].trigger()
    again = Workspace(e.paths)
    try:
        s = again.edit_state
        assert (s.ripple_on, s.ripple_scope, s.follow_playhead, s.auto_scroll, s.linked_selection) == (True, "all", "center", False, False)
        assert s.snap.enabled is False and s.snap.tolerance_frames == 8
        assert s.tool == "select"  # the tool is per session
    finally:
        again.shutdown()


# ------------------------------------------------------------------------------------------------------------------------ timecode field
def test_timecode_field_shows_the_playhead_and_jumps_on_enter(e):
    p, c = e.panel, e.panel.canvas
    tb = timebase_of(e.ws.project)
    lay(e, 2)
    c.set_playhead(2.5)
    assert p.timecode.text() == tb.format(2.5) == "00:00:02:15"
    assert p.length_label.text() == "Length " + tb.format(10.0)
    for text, expect in (("00:00:04:10", 4 + 10 / 30), ("3", 3.0), ("1.5", 1.5), ("45f", 1.5), ("0:07:00", 7.0), (" 00:00:01:00 ", 1.0)):
        p.timecode.setText(text)
        QTest.keyClick(p.timecode, Qt.Key.Key_Return)
        assert c.playhead == pytest.approx(expect, abs=1e-6), text
        assert not p.timecode.property("invalid")
        assert p.timecode.text() == tb.format(expect)  # shows the canonical form


def test_timecode_field_rejects_garbage_and_leaves_the_playhead_alone(e):
    p, c = e.panel, e.panel.canvas
    c.set_playhead(2.0)
    for bad in ("banana", "00:00:01:99", "", "1:2:3:4:5", "-3", "9999999"):
        e.status.clear()
        p.timecode.setText(bad)
        QTest.keyClick(p.timecode, Qt.Key.Key_Return)
        assert c.playhead == pytest.approx(2.0), bad
        assert p.timecode.property("invalid") is True, bad
        assert e.status and e.status[-1], bad  # the parser's own message
    assert "Frames must be 0" in _message_for(p, "00:00:01:99", e)
    p.timecode.setText("00:00:01:00")  # typing again clears the red outline
    p.timecode.textEdited.emit("00:00:01:00")
    assert not p.timecode.property("invalid")


def _message_for(panel, text, e) -> str:
    e.status.clear()
    panel.timecode.setText(text)
    QTest.keyClick(panel.timecode, Qt.Key.Key_Return)
    return e.status[-1]


def test_timecode_field_is_not_overwritten_while_typing_and_follows_the_playhead_otherwise(e):
    p, c = e.panel, e.panel.canvas
    focus(p.timecode)
    p.timecode.setText("00:00:0")
    c.set_playhead(3.0)
    assert p.timecode.text() == "00:00:0"  # the user is typing
    focus(c)
    c.set_playhead(1.0)
    assert p.timecode.text() == "00:00:01:00"


# ------------------------------------------------------------------------------------------------------------------------ toolbar: enable states, selection label, edit buttons
def test_buttons_follow_real_state_and_say_why_they_are_disabled(e):
    p = e.panel
    for btn in (p.split_btn, p.delete_btn, p.ripple_delete_btn, p.clear_range_btn):
        assert not btn.isEnabled()
    assert "Select a clip first." in p.delete_btn.toolTip() and "Not available" in p.delete_btn.toolTip()
    assert "timeline is empty" in p.split_btn.toolTip()
    assert "no in / out range" in p.clear_range_btn.toolTip()
    assert not p.add_selected_btn.isEnabled() and "Select media in the library first." in p.add_selected_btn.toolTip()
    assert p.selection_label.text() == "No selection"
    ids = lay(e, 3)
    e.ws.set_selection(ids[:2])
    assert p.split_btn.isEnabled() and p.delete_btn.isEnabled() and p.ripple_delete_btn.isEnabled()
    assert p.selection_label.text() == "2 clips selected, 10.0 s"
    e.ws.set_selection(ids[:1])
    assert p.selection_label.text() == "1 clip selected, 6.0 s"
    e.ws.selection.select_none()
    assert not p.delete_btn.isEnabled()
    e.ws.edit.set_range(1.0, 2.0)
    assert p.clear_range_btn.isEnabled()
    # the selected clip's track is locked: delete says so instead of failing later
    e.ws.set_selection(ids[:1])
    e.ws.edit.set_track_flag("track_v1", "locked", True)
    assert not p.delete_btn.isEnabled() and "locked" in p.delete_btn.toolTip()
    e.ws.undo()
    assert p.delete_btn.isEnabled()


def test_tooltips_show_the_current_shortcut_and_follow_rebinding(e):
    p = e.panel
    assert "Ctrl+B" in p.split_btn.toolTip() and "Del" in p.delete_btn.toolTip()
    assert "S / V" in p.select_btn.toolTip() and "B" in p.blade_btn.toolTip()
    p.shortcuts.set_binding("split", "Ctrl+K")
    assert "Ctrl+K" in p.split_btn.toolTip() and "Ctrl+B" not in p.split_btn.toolTip()
    assert "no audio output" in p.play_btn.toolTip()  # honest about what Space does


def test_toolbar_edit_buttons_run_real_edits(e):
    p, c = e.panel, e.panel.canvas
    ids = lay(e, 2)
    c.set_playhead(2.0)
    e.ws.set_selection([ids[0]])
    QTest.mouseClick(p.split_btn, Qt.MouseButton.LeftButton)
    assert [(round(x.timeline_start, 3), round(x.duration, 3)) for x in tl(e).get_track("track_v1").clips][:2] == [(0.0, 2.0), (2.0, 4.0)]
    ws = e.ws
    ws.undo()
    assert len(tl(e).get_track("track_v1").clips) == 2
    e.ws.set_selection([ids[0]])
    QTest.mouseClick(p.delete_btn, Qt.MouseButton.LeftButton)
    assert [x.id for x in tl(e).get_track("track_v1").clips] == [ids[1]]
    assert tl(e).get_clip(ids[1]).timeline_start == pytest.approx(6.0)  # plain delete leaves the gap
    ws.undo()
    e.ws.set_selection([ids[0]])
    QTest.mouseClick(p.ripple_delete_btn, Qt.MouseButton.LeftButton)
    assert tl(e).get_clip(ids[1]).timeline_start == pytest.approx(0.0)  # ripple delete closes it
    ws.undo()
    c.set_playhead(3.0)
    QTest.mouseClick(p.marker_btn, Qt.MouseButton.LeftButton)
    assert [m.time for m in tl(e).markers] == [3.0]
    c.set_playhead(1.0)
    QTest.mouseClick(p.in_btn, Qt.MouseButton.LeftButton)
    c.set_playhead(4.0)
    QTest.mouseClick(p.out_btn, Qt.MouseButton.LeftButton)
    assert tl(e).range == (1.0, 4.0)
    assert p.clear_range_btn.isEnabled()
    QTest.mouseClick(p.clear_range_btn, Qt.MouseButton.LeftButton)
    assert tl(e).range is None and not p.clear_range_btn.isEnabled()


def test_a_refused_edit_reports_the_engines_reason_in_the_status_line(e):
    p, c = e.panel, e.panel.canvas
    ids = lay(e, 1)
    e.ws.set_selection(ids)
    c.set_playhead(0.0)  # a cut at a clip's very first frame would leave an empty piece
    QTest.mouseClick(p.split_btn, Qt.MouseButton.LeftButton)
    assert len(tl(e).get_track("track_v1").clips) == 1
    assert not e.errors  # reported in the status line, not as a crash dialog


def test_undo_redo_buttons_mirror_the_command_stack(e):
    p = e.panel
    assert not p.undo_btn.isEnabled() and not p.redo_btn.isEnabled()
    lay(e, 1)
    assert p.undo_btn.isEnabled() and "Add media" in p.undo_btn.toolTip()
    QTest.mouseClick(p.undo_btn, Qt.MouseButton.LeftButton)
    assert not tl(e).all_clips() and p.redo_btn.isEnabled() and "Add media" in p.redo_btn.toolTip()
    QTest.mouseClick(p.redo_btn, Qt.MouseButton.LeftButton)
    assert len(tl(e).all_clips()) == 1


def test_zoom_buttons_and_slider(e):
    p, c = e.panel, e.panel.canvas
    lay(e, 2)
    pps = c.pps
    QTest.mouseClick(p.zoom_in_btn, Qt.MouseButton.LeftButton)
    assert c.pps > pps and p.zoom.value() == int(c.pps)
    QTest.mouseClick(p.zoom_out_btn, Qt.MouseButton.LeftButton)
    assert c.pps == pytest.approx(pps, rel=0.01)
    QTest.mouseClick(p.fit_btn, Qt.MouseButton.LeftButton)
    assert c.pps * 10.0 <= c.viewport_width()  # the whole 10 s timeline fits
    assert not p.zoom_sel_btn.isEnabled()
    e.ws.set_selection([tl(e).get_track("track_v1").clips[0].id])
    assert p.zoom_sel_btn.isEnabled()
    QTest.mouseClick(p.zoom_sel_btn, Qt.MouseButton.LeftButton)
    p.zoom.setValue(200)
    assert c.pps == pytest.approx(200.0, abs=1)


def test_add_track_menu_adds_each_supported_kind(e):
    p = e.panel
    labels = [a.text() for a in p.add_track_btn.menu().actions()]
    assert labels == ["Video track", "Image track", "Graphics track", "Audio track", "Text track", "Captions track"]
    n = len(tl(e).tracks)
    for act in p.add_track_btn.menu().actions():
        act.trigger()
    assert len(tl(e).tracks) == n + 6
    new = [t for t in tl(e).tracks if t.id not in {d.id for d in Timeline.default().tracks}]
    assert sorted(t.kind.value for t in new) == sorted(["video", "image", "graphics", "audio", "text", "captions"])
    assert len({t.name for t in tl(e).tracks}) == len(tl(e).tracks)  # distinct names
    e.ws.undo()
    assert len(tl(e).tracks) == n + 5


# ------------------------------------------------------------------------------------------------------------------------ Add selected media
def test_add_selected_media_multi_selection_is_one_undo_step(e):
    p = e.panel
    p.selected_provider = lambda: [e.A.id, e.B.id, e.C.id]
    p.update_controls()
    assert p.add_selected_btn.isEnabled() and "3 selected" in p.add_selected_btn.toolTip()
    e.panel.canvas.set_playhead(1.0)
    QTest.mouseClick(p.add_selected_btn, Qt.MouseButton.LeftButton)
    clips = tl(e).get_track("track_v1").clips
    assert [c.asset_id for c in clips] == [e.A.id, e.B.id, e.C.id]
    assert [round(c.timeline_start, 3) for c in clips] == [1.0, 7.0, 11.0]
    assert undo_text(e) == "Add media"
    e.ws.undo()
    assert tl(e).all_clips() == []  # all three removed by ONE undo


def _row(e, track="track_v1") -> list[tuple]:
    return [(c.asset_id, round(c.timeline_start, 3), round(c.duration, 3)) for c in tl(e).get_track(track).clips]


def test_add_selected_media_shift_inserts_and_alt_overwrites(e):
    p = e.panel
    first = lay(e, 2)  # a (0-6), b (6-10)
    p.selected_provider = lambda: [e.C.id]
    p.update_controls()
    e.panel.canvas.set_playhead(2.0)
    QTest.mouseClick(p.add_selected_btn, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)  # insert: ripples the rest right
    assert undo_text(e).lower().startswith("insert")
    assert _row(e) == [(e.A.id, 0.0, 2.0), (e.C.id, 2.0, 5.0), (e.A.id, 7.0, 4.0), (e.B.id, 11.0, 4.0)]  # a was cut at 2 s, c went in, everything after moved right by c's length
    e.ws.undo()
    assert [c.id for c in tl(e).get_track("track_v1").clips] == first and _row(e) == [(e.A.id, 0.0, 6.0), (e.B.id, 6.0, 4.0)]
    QTest.mouseClick(p.add_selected_btn, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier)  # overwrite: nothing moves, what is underneath is replaced
    assert undo_text(e).lower().startswith("overwrite")
    assert _row(e) == [(e.A.id, 0.0, 2.0), (e.C.id, 2.0, 5.0), (e.B.id, 7.0, 3.0)]  # c replaced what was under it (2-7 s); b lost its first second; nothing moved
    e.ws.undo()
    assert [c.id for c in tl(e).get_track("track_v1").clips] == first and _row(e) == [(e.A.id, 0.0, 6.0), (e.B.id, 6.0, 4.0)]


def test_add_selected_media_refuses_mixed_kinds_and_adds_nothing(e):
    p = e.panel
    p.selected_provider = lambda: [e.A.id, e.AUD.id]
    p.update_controls()
    QTest.mouseClick(p.add_selected_btn, Qt.MouseButton.LeftButton)
    assert tl(e).all_clips() == [] and "one kind at a time" in e.status[-1]
    p.selected_provider = lambda: [e.AUD.id]
    QTest.mouseClick(p.add_selected_btn, Qt.MouseButton.LeftButton)
    assert [tl(e).get_track(c.track_id).is_audio for c in tl(e).all_clips()] == [True]  # audio lands on an audio track


# ------------------------------------------------------------------------------------------------------------------------ track headers: geometry
def _canvas_extra() -> int:
    import app.ui.timeline_canvas as canvas_module

    return getattr(canvas_module, "NEW_TRACK_STRIP_H", 0)


def _assert_headers_match_canvas(e):
    flush()
    h, c = e.panel.headers, e.panel.canvas
    lay_ = c.track_layout
    assert h.layout_ is lay_  # one layout object, never a copy
    assert h.height() == lay_.total_height and c.height() == lay_.total_height + _canvas_extra()  # the canvas may keep a band under the last row (new-track drop strip)
    assert lay_.count == len(tl(e).tracks)
    for i, t in enumerate(tl(e).tracks):
        assert lay_.top(i) == c.row_rect_y(i) and lay_.height(i) == c.row_height(i)
        assert h.hit(100, lay_.top(i) + 2)[1] == i and c.track_index_at(lay_.top(i) + 2) == i
        assert h.hit(100, lay_.top(i) + lay_.height(i) - 1)[1] == i
        name = h.name_rect(i, t)
        assert lay_.top(i) <= name.top() and name.bottom() <= lay_.bottom(i) and name.left() >= 40


def test_header_geometry_equals_the_canvas_layout_for_default_custom_and_collapsed_heights(e):
    _assert_headers_match_canvas(e)
    assert e.panel.headers.height() == 28 + 9 * 36 + 1
    ws = e.ws
    ws.edit.set_track_display("track_v2", height=90)
    ws.edit.set_track_display("track_v3", collapsed=True)
    ws.edit.set_track_display("track_a2", height=24)
    _assert_headers_match_canvas(e)
    lay_ = e.panel.canvas.track_layout
    assert [lay_.height(i) for i in range(9)] == [36, 90, 18, 36, 36, 36, 36, 24, 36]
    assert e.panel.headers.height() == 28 + sum(lay_.height(i) for i in range(9)) + 1
    ws.undo()
    ws.undo()
    ws.undo()
    _assert_headers_match_canvas(e)
    assert e.panel.headers.height() == 28 + 9 * 36 + 1


def test_header_paints_its_row_separators_where_the_layout_puts_them(e):
    from app.ui.theme import palette

    e.ws.edit.set_track_display("track_v2", height=70)
    e.ws.edit.set_track_display("track_v4", collapsed=True)
    flush()
    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    img = h.grab().toImage()
    border = palette(e.ws.settings.theme)["border"].lower()
    for i in range(lay_.count):
        y = lay_.bottom(i) - 1  # the row's last pixel row
        assert img.pixelColor(HEADER_W - 1, y).name().lower() == border, i
        assert img.pixelColor(HEADER_W - 1, y - 3).name().lower() != border, i  # ... and only there


def test_header_rows_show_name_kind_and_flag_buttons(e):
    h = e.panel.headers
    ts = tl(e).tracks
    audio_row, video_row = row_of(e, "track_a1"), row_of(e, "track_v1")
    assert [f for f, _r in h.flag_rects(video_row, ts[video_row])] == ["hidden", "muted", "locked"]  # solo is audio only
    assert [f for f, _r in h.flag_rects(audio_row, ts[audio_row])] == ["hidden", "muted", "solo", "locked"]
    assert "preview AND from the exported" in h.tooltip_at(*_flag_pos(h, video_row, "hidden"))
    assert "silences its audio" in h.tooltip_at(*_flag_pos(h, video_row, "muted"))
    assert "cannot be edited" in h.tooltip_at(*_flag_pos(h, video_row, "locked"))
    assert "hear only the soloed" in h.tooltip_at(*_flag_pos(h, audio_row, "solo"))
    assert "V1 Main Video: video track, 0 clips" in h.tooltip_at(100, e.panel.canvas.track_layout.top(video_row) + 10)
    assert "resize" in h.tooltip_at(100, e.panel.canvas.track_layout.bottom(video_row) - 2)
    assert "Collapse" in h.tooltip_at(*_chevron_pos(h, video_row))


def _flag_pos(h, row, flag):
    r = dict(h.flag_rects(row, h._tracks()[row]))[flag]
    return r.center().x(), r.center().y()


def _chevron_pos(h, row):
    r = h.chevron_rect(row)
    return r.center().x(), r.center().y()


# ------------------------------------------------------------------------------------------------------------------------ track headers: gestures
def test_flag_buttons_toggle_hide_mute_solo_lock_and_undo(e):
    h = e.panel.headers
    for track_id, flag in (("track_v1", "hidden"), ("track_v1", "muted"), ("track_v1", "locked"), ("track_a1", "solo"), ("track_a2", "muted")):
        row = row_of(e, track_id)
        x, y = _flag_pos(h, row, flag)
        assert getattr(tl(e).get_track(track_id), flag) is False
        QTest.mouseClick(h, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
        assert getattr(tl(e).get_track(track_id), flag) is True, (track_id, flag)
        e.ws.undo()
        assert getattr(tl(e).get_track(track_id), flag) is False
        e.ws.redo()
        assert getattr(tl(e).get_track(track_id), flag) is True
        QTest.mouseClick(h, Qt.MouseButton.LeftButton, pos=QPoint(x, y))  # and off again
        assert getattr(tl(e).get_track(track_id), flag) is False


def test_header_click_moves_keyboard_focus_to_the_canvas(e):
    h, c = e.panel.headers, e.panel.canvas
    e.panel.timecode.setFocus()
    x, y = _flag_pos(h, row_of(e, "track_v1"), "muted")
    QTest.mouseClick(h, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    pump(lambda: QApplication.focusWidget() is c, timeout=5)


def test_collapse_chevron_and_menu_collapse_and_expand(e):
    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    row = row_of(e, "track_v2")
    x, y = _chevron_pos(h, row)
    QTest.mouseClick(h, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    assert tl(e).get_track("track_v2").collapsed and lay_.height(row) == 18
    _assert_headers_match_canvas(e)
    assert "Expand" in h.tooltip_at(*_chevron_pos(h, row))
    assert h.hit(100, lay_.bottom(row) - 1)[0] == "body"  # a collapsed strip has no resize handle
    QTest.mouseClick(h, Qt.MouseButton.LeftButton, pos=QPoint(x, lay_.top(row) + 9))
    assert not tl(e).get_track("track_v2").collapsed and lay_.height(row) == 36
    e.ws.undo()
    assert tl(e).get_track("track_v2").collapsed
    e.ws.undo()
    assert not tl(e).get_track("track_v2").collapsed
    _assert_headers_match_canvas(e)


def test_resize_drag_previews_live_and_commits_one_undo_step(e):
    h, c, lay_ = e.panel.headers, e.panel.canvas, e.panel.canvas.track_layout
    row = row_of(e, "track_v2")
    before = undo_text(e)
    start = QPoint(100, lay_.bottom(row) - 2)
    assert h.hit(start.x(), start.y())[0] == "resize"
    QTest.mousePress(h, Qt.MouseButton.LeftButton, pos=start)
    for dy in (6, 14, 22, 30):
        QTest.mouseMove(h, start + QPoint(0, dy))
        assert lay_.height(row) == 36 + dy  # live: the layout (hence the canvas rows and header rows) follows the pointer
        assert h.height() == lay_.total_height and c.height() == lay_.total_height + _canvas_extra()
        assert tl(e).get_track("track_v2").height == 0  # ... but nothing is committed yet
        assert undo_text(e) == before
    QTest.mouseRelease(h, Qt.MouseButton.LeftButton, pos=start + QPoint(0, 30))
    assert tl(e).get_track("track_v2").height == 66 and lay_.height(row) == 66
    assert undo_text(e) == "Change track display"
    _assert_headers_match_canvas(e)
    e.ws.undo()
    assert tl(e).get_track("track_v2").height == 0 and undo_text(e) == before  # exactly ONE step
    assert lay_.height(row) == 36
    _assert_headers_match_canvas(e)


def test_resize_is_clamped_to_the_layout_limits_and_a_no_op_drag_commits_nothing(e):
    from app.ui.timeline_layout import MAX_ROW_H, MIN_ROW_H

    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    row = row_of(e, "track_v3")
    start = QPoint(100, lay_.bottom(row) - 2)
    QTest.mousePress(h, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(h, start + QPoint(0, 1000))
    assert lay_.height(row) == MAX_ROW_H
    QTest.mouseMove(h, start + QPoint(0, -1000))
    assert lay_.height(row) == MIN_ROW_H
    QTest.mouseMove(h, start)
    QTest.mouseRelease(h, Qt.MouseButton.LeftButton, pos=start)
    assert tl(e).get_track("track_v3").height == 0 and not e.ws.commands.can_undo  # back to the original: no command
    QTest.mouseDClick(h, Qt.MouseButton.LeftButton, pos=QPoint(100, lay_.bottom(row) - 2))  # double-click on the handle = default height
    assert tl(e).get_track("track_v3").height == 0
    e.ws.edit.set_track_display("track_v3", height=80)
    QTest.mouseDClick(h, Qt.MouseButton.LeftButton, pos=QPoint(100, lay_.bottom(row) - 2))
    assert tl(e).get_track("track_v3").height == 0


def test_reorder_by_dragging_a_header_with_a_drop_line_and_one_undo_step(e):
    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    assert track_ids(e)[:3] == ["track_v1", "track_v2", "track_v3"]
    src = row_of(e, "track_v1")
    press_pos = QPoint(90, lay_.top(src) + lay_.height(src) // 2)
    QTest.mousePress(h, Qt.MouseButton.LeftButton, pos=press_pos)
    QTest.mouseMove(h, press_pos + QPoint(0, 3))
    assert h._drop_slot is None  # below the drag threshold: still a click
    QTest.mouseMove(h, QPoint(90, lay_.top(2) + 5))  # a little into row 2 (the top half: before it)
    assert h._drop_slot == 2
    QTest.mouseMove(h, QPoint(90, lay_.top(2) + 30))  # the lower half of row 2: after it
    assert h._drop_slot == 3
    before = undo_text(e)
    QTest.mouseRelease(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.top(2) + 30))
    assert track_ids(e)[:3] == ["track_v2", "track_v3", "track_v1"]
    assert h._drop_slot is None and undo_text(e) == "Move track"
    _assert_headers_match_canvas(e)
    e.ws.undo()
    assert track_ids(e)[:3] == ["track_v1", "track_v2", "track_v3"] and undo_text(e) == before


def test_reorder_to_the_very_top_and_bottom_and_back_to_the_same_place(e):
    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    last = len(tl(e).tracks) - 1
    QTest.mousePress(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.top(last) + 10))
    QTest.mouseMove(h, QPoint(90, 5 + 28))
    QTest.mouseMove(h, QPoint(90, lay_.top(0) + 2))
    assert h._drop_slot == 0
    QTest.mouseRelease(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.top(0) + 2))
    assert track_ids(e)[0] == "track_a3"
    e.ws.undo()
    n = len(tl(e).tracks)
    QTest.mousePress(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.top(0) + 10))
    QTest.mouseMove(h, QPoint(90, lay_.bottom(n - 1) - 2))
    QTest.mouseRelease(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.bottom(n - 1) - 2))
    assert track_ids(e)[-1] == "track_v1"
    e.ws.undo()
    # dropping where it already is changes nothing and records nothing
    assert not e.ws.commands.can_undo
    QTest.mousePress(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.top(2) + 10))
    QTest.mouseMove(h, QPoint(90, lay_.top(2) + 17))
    assert h._drop_slot in (2, 3)  # either side of its own row: the same place
    QTest.mouseRelease(h, Qt.MouseButton.LeftButton, pos=QPoint(90, lay_.top(2) + 17))
    assert track_ids(e)[:3] == ["track_v1", "track_v2", "track_v3"] and not e.ws.commands.can_undo


def test_a_plain_click_on_a_header_does_not_reorder_or_edit(e):
    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    ids = track_ids(e)
    QTest.mouseClick(h, Qt.MouseButton.LeftButton, pos=QPoint(100, lay_.top(3) + 12))
    assert track_ids(e) == ids and not e.ws.commands.can_undo


def test_double_click_renames_in_place_and_escape_cancels(e):
    h, lay_ = e.panel.headers, e.panel.canvas.track_layout
    row = row_of(e, "track_v2")
    QTest.mouseDClick(h, Qt.MouseButton.LeftButton, pos=QPoint(100, lay_.top(row) + 10))
    ed = h._editor
    assert ed is not None and ed.text() == "V2 B-Roll"
    assert lay_.top(row) <= ed.geometry().top() and ed.geometry().bottom() <= lay_.bottom(row)
    ed.setText("Cutaways")
    QTest.keyClick(ed, Qt.Key.Key_Return)
    assert tl(e).get_track("track_v2").name == "Cutaways" and h._editor is None
    assert undo_text(e) == "Rename track"
    e.ws.undo()
    assert tl(e).get_track("track_v2").name == "V2 B-Roll"
    h.begin_rename("track_v2")
    h._editor.setText("Never")
    QTest.keyClick(h._editor, Qt.Key.Key_Escape)
    assert tl(e).get_track("track_v2").name == "V2 B-Roll" and h._editor is None and not e.ws.commands.can_undo
    h.begin_rename("track_v2")
    h._editor.setText("   ")  # an empty name: the engine refuses and says so
    QTest.keyClick(h._editor, Qt.Key.Key_Return)
    assert tl(e).get_track("track_v2").name == "V2 B-Roll" and any("needs a name" in m for m in e.errors)


def test_typing_in_the_rename_editor_never_triggers_shortcuts(e):
    h = e.panel.headers
    ids = lay(e, 2)
    e.ws.set_selection(ids)
    h.begin_rename("track_v2")
    ed = h._editor
    focus(ed)
    before = snapshot(e)
    ed.setText("")
    QTest.keyClicks(ed, "svbnl")  # tool / blade / snap / link keys
    for key in (Qt.Key.Key_Delete, Qt.Key.Key_Space, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home):
        QTest.keyClick(ed, key)
    assert ed.text().startswith("svbnl") and snapshot(e) == before
    assert e.ws.edit_state.tool == "select" and e.ws.edit_state.snap.enabled and e.ws.edit_state.linked_selection
    assert not e.panel.transport.playing and list(e.ws.selection.ids) == ids
    QTest.keyClick(ed, Qt.Key.Key_Escape)


# ------------------------------------------------------------------------------------------------------------------------ track context menu
def _menu_actions(menu) -> dict:
    """{label: action} of a menu; submenu entries are keyed "Submenu > label"."""
    out = {}
    for a in menu.actions():
        if a.menu() is not None:
            for s in a.menu().actions():
                out[f"{a.text()} > {s.text()}"] = s
        elif not a.isSeparator():
            out[a.text()] = a
    return out


def test_track_menu_lists_every_operation_with_state_dependent_wording(e):
    h = e.panel.headers
    v1, a1 = tl(e).get_track("track_v1"), tl(e).get_track("track_a1")
    m = _menu_actions(h.build_menu(v1))
    for key in ("Rename…", "Duplicate track (with clips)", "Duplicate track (empty)", "Expand" if v1.collapsed else "Collapse", "Hide track", "Mute track", "Solo track", "Lock track",
                "Select all clips on track", "Delete track", "Change type > Video", "Change type > Audio", "Height > Small (24 px)", "Height > Normal (default)", "Height > Large (72 px)", "Height > Huge (120 px)"):
        assert key in m, key
    assert "Set volume…" not in m and not m["Solo track"].isEnabled() and "audio tracks only" in m["Solo track"].toolTip()
    assert not m["Select all clips on track"].isEnabled() and "no clips" in m["Select all clips on track"].toolTip()
    assert m["Height > Normal (default)"].isChecked() and not m["Change type > Video"].isEnabled() and m["Change type > Video"].isChecked()
    assert "Set volume…" in _menu_actions(h.build_menu(a1)) and _menu_actions(h.build_menu(a1))["Solo track"].isEnabled()
    e.ws.edit.set_track_flag("track_v1", "hidden", True)
    e.ws.edit.set_track_flag("track_v1", "locked", True)
    m2 = _menu_actions(h.build_menu(tl(e).get_track("track_v1")))
    assert "Show track" in m2 and "Unlock track" in m2 and "Hide track" not in m2
    assert "preview AND from the exported" in _menu_actions(h.build_menu(a1))["Hide track"].toolTip()


def test_menu_height_presets_set_heights_and_normal_resets_to_default(e):
    h = e.panel.headers
    for label, px in (("Small (24 px)", 24), ("Large (72 px)", 72), ("Huge (120 px)", 120)):
        _menu_actions(h.build_menu(tl(e).get_track("track_v3")))[f"Height > {label}"].trigger()
        assert tl(e).get_track("track_v3").height == px and e.panel.canvas.track_layout.height(row_of(e, "track_v3")) == px
        _assert_headers_match_canvas(e)
    _menu_actions(h.build_menu(tl(e).get_track("track_v3")))["Height > Normal (default)"].trigger()
    assert tl(e).get_track("track_v3").height == 0
    _assert_headers_match_canvas(e)


def test_menu_collapse_hide_mute_solo_lock_volume(e, monkeypatch):
    h = e.panel.headers
    for track_id, label, flag in (("track_v1", "Collapse", "collapsed"), ("track_v1", "Hide track", "hidden"), ("track_v1", "Mute track", "muted"),
                                  ("track_a1", "Solo track", "solo"), ("track_a1", "Lock track", "locked")):
        _menu_actions(h.build_menu(tl(e).get_track(track_id)))[label].trigger()
        assert getattr(tl(e).get_track(track_id), flag) is True, label
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: (50, True)))
    e.ws.undo()  # unlock A1 so the volume change is allowed
    _menu_actions(h.build_menu(tl(e).get_track("track_a2")))["Set volume…"].trigger()
    assert tl(e).get_track("track_a2").volume == pytest.approx(0.5)


def test_menu_duplicate_track_with_clips_and_empty(e):
    h = e.panel.headers
    ids = lay(e, 2)
    n = len(tl(e).tracks)
    _menu_actions(h.build_menu(tl(e).get_track("track_v1")))["Duplicate track (with clips)"].trigger()
    assert len(tl(e).tracks) == n + 1
    copy = tl(e).tracks[row_of(e, "track_v1") + 1]
    assert copy.name == "V1 Main Video copy" and [c.asset_id for c in copy.clips] == [e.A.id, e.B.id]
    assert {c.id for c in copy.clips}.isdisjoint(ids) and [round(c.timeline_start, 3) for c in copy.clips] == [0.0, 6.0]
    assert undo_text(e) == "Duplicate track"
    e.ws.undo()
    assert len(tl(e).tracks) == n
    _menu_actions(h.build_menu(tl(e).get_track("track_v1")))["Duplicate track (empty)"].trigger()
    copy = tl(e).tracks[row_of(e, "track_v1") + 1]
    assert copy.clips == [] and copy.kind is TrackKind.VIDEO
    _assert_headers_match_canvas(e)
    e.ws.undo()
    assert len(tl(e).tracks) == n and len(tl(e).get_track("track_v1").clips) == 2


def test_menu_change_type_allowed_and_refused_with_the_engines_message(e):
    h = e.panel.headers
    _menu_actions(h.build_menu(tl(e).get_track("track_v4")))["Change type > Image"].trigger()  # an empty graphics track may become an image track
    assert tl(e).get_track("track_v4").kind is TrackKind.IMAGE
    e.ws.undo()
    assert tl(e).get_track("track_v4").kind is TrackKind.GRAPHICS
    lay(e, 2)
    e.errors.clear()
    _menu_actions(h.build_menu(tl(e).get_track("track_v1")))["Change type > Audio"].trigger()  # video clips cannot live on an audio track
    assert tl(e).get_track("track_v1").kind is TrackKind.VIDEO
    assert e.errors and "a.mp4" in e.errors[-1] and "audio" in e.errors[-1]  # the engine names the clip and the type
    assert undo_text(e) == "Add media"  # nothing was recorded for the refused change
    e.ws.edit.set_track_flag("track_v4", "locked", True)
    e.errors.clear()
    _menu_actions(h.build_menu(tl(e).get_track("track_v4")))["Change type > Video"].trigger()
    assert tl(e).get_track("track_v4").kind is TrackKind.GRAPHICS and e.errors


def test_menu_delete_track_confirms_when_it_holds_clips_and_undo_restores_them(e, monkeypatch):
    h = e.panel.headers
    ids = lay(e, 3)
    asked: list[str] = []

    def answer(value):
        def question(parent, title, text, buttons=None, default=None):
            asked.append(text)
            return value
        return question

    track = tl(e).get_track("track_v1")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer(QMessageBox.StandardButton.No)))
    _menu_actions(h.build_menu(track))["Delete track"].trigger()  # asked, said no: nothing happens
    assert len(asked) == 1 and "Delete track “V1 Main Video” and its 3 clips? You can undo this." == asked[0]
    assert [c.id for c in tl(e).get_track("track_v1").clips] == ids and not e.ws.commands.undo_text.startswith("Delete")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer(QMessageBox.StandardButton.Yes)))
    n = len(tl(e).tracks)
    _menu_actions(h.build_menu(track))["Delete track"].trigger()
    assert len(asked) == 2 and len(tl(e).tracks) == n - 1 and all(tl(e).get_clip(i) is None for i in ids)
    _assert_headers_match_canvas(e)
    assert undo_text(e) == "Delete track"
    e.ws.undo()  # the track AND its clips come back, in place
    assert len(tl(e).tracks) == n and track_ids(e)[0] == "track_v1"
    assert [c.id for c in tl(e).get_track("track_v1").clips] == ids
    _assert_headers_match_canvas(e)


def test_delete_track_without_clips_does_not_ask_and_a_locked_track_is_refused(e, monkeypatch):
    h = e.panel.headers

    def boom(*a, **k):
        raise AssertionError("an empty track must be deleted without a confirmation")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(boom))
    n = len(tl(e).tracks)
    _menu_actions(h.build_menu(tl(e).get_track("track_v2")))["Delete track"].trigger()
    assert len(tl(e).tracks) == n - 1
    e.ws.undo()
    assert len(tl(e).tracks) == n
    e.ws.edit.set_track_flag("track_v3", "locked", True)
    e.errors.clear()
    _menu_actions(h.build_menu(tl(e).get_track("track_v3")))["Delete track"].trigger()
    assert len(tl(e).tracks) == n and e.errors  # the engine's reason is shown


def test_menu_select_all_clips_on_track(e):
    h = e.panel.headers
    ids = lay(e, 3)
    e.ws.selection.select_none()
    _menu_actions(h.build_menu(tl(e).get_track("track_v1")))["Select all clips on track"].trigger()
    assert set(e.ws.selection.ids) == set(ids)


# ------------------------------------------------------------------------------------------------------------------------ shortcuts: table, dispatch, text-entry safety
def test_default_shortcuts_have_no_conflicts_and_match_the_spec(e):
    reg = e.panel.shortcuts
    keys = {}
    for aid, ks in reg.bindings().items():
        for k in ks:
            assert k not in keys, f"{k} is used by both {keys[k]} and {aid}"
            keys[k] = aid
    assert set(reg.bindings()) == {s.id for s in SPECS} and all(reg.bindings().values()) is True
    wanted = {"play_pause": ["Space"], "tool_select": ["S", "V"], "tool_blade": ["B"], "delete": ["Del", "Backspace"], "ripple_delete": ["Shift+Del", "Shift+Backspace"], "copy": ["Ctrl+C"],
              "cut": ["Ctrl+X"], "paste": ["Ctrl+V"], "paste_attributes": ["Ctrl+Shift+V"], "duplicate": ["Ctrl+D"], "go_start": ["Home"], "go_end": ["End"], "frame_prev": ["Left"],
              "frame_next": ["Right"], "second_prev": ["Shift+Left"], "second_next": ["Shift+Right"], "edit_prev": ["Up"], "edit_next": ["Down"], "mark_in": ["I"], "mark_out": ["O"],
              "clear_range": ["Alt+X"], "add_marker": ["M"], "marker_next": ["Shift+M"], "marker_prev": ["Alt+M"], "split": ["Ctrl+B"], "toggle_snap": ["N"], "toggle_ripple": ["Shift+R"],
              "toggle_link": ["L"], "select_all": ["A", "Ctrl+A"], "invert_selection": ["Ctrl+Alt+I"], "zoom_in": ["Ctrl+=", "Ctrl++"], "zoom_out": ["Ctrl+-"], "zoom_fit": ["Shift+Z"],
              "zoom_selection": ["Alt+Z"], "group": ["Ctrl+G"], "ungroup": ["Ctrl+Shift+G"], "link_clips": ["Ctrl+L"]}
    for aid, ks in wanted.items():
        assert reg.keys(aid) == ks, aid
    window_keys = {k for k, _l in reg.window_shortcuts()}
    assert {"Ctrl+S", "Ctrl+I", "Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z"} <= window_keys  # undo / redo / save / import stay the window's
    assert window_keys.isdisjoint(keys)
    assert "undo" not in reg.specs and "redo" not in reg.specs  # not registered twice


def test_every_default_key_reaches_exactly_its_action(e, monkeypatch):
    reg, canvas = e.panel.shortcuts, e.panel.canvas
    called: list[str] = []
    for aid in reg.specs:
        label, defaults, _handler, need = reg.table[aid]
        monkeypatch.setitem(reg.table, aid, (label, defaults, (lambda a=aid: called.append(a)), need))
    monkeypatch.setattr(reg, "why_not", lambda aid, state=None: None)  # availability is tested separately; this checks the key -> action wiring
    focus(canvas)
    for aid in reg.specs:
        for key in reg.keys(aid):
            called.clear()
            press(canvas, key)
            assert called == [aid], (aid, key, called)


def test_shortcuts_only_act_while_the_timeline_page_is_shown_and_has_focus(e):
    canvas, library = e.panel.canvas, e.win.library
    ids = lay(e, 2)
    e.ws.set_selection(ids[:1])
    before = snapshot(e)
    for widget in (library.search, library.list):  # the library (text box and list) is on the Timeline page too, but it is not the timeline
        focus(widget)
        for key in ("Ctrl+D", "Delete", "Ctrl+B", "Left", "Space", "N", "Ctrl+A"):
            press(widget, key)
        assert snapshot(e) == before and not e.panel.transport.playing and e.ws.edit_state.snap.enabled, widget
    e.win.go_to("Project")  # another page: the timeline is not shown at all
    QApplication.processEvents()
    assert not canvas.isVisible()
    typing_into = QApplication.focusWidget()  # real key presses go to the widget that has focus (QTest would deliver straight to a hidden canvas)
    assert typing_into is not None and not e.panel.isAncestorOf(typing_into)
    for key in ("Ctrl+D", "Delete", "N", "Ctrl+B"):
        press(typing_into, key)
    assert snapshot(e) == before and e.ws.edit_state.snap.enabled
    e.win.go_to("Timeline")
    focus(canvas)
    e.ws.set_selection(ids[1:])  # the last clip: its copy fits behind it
    press(canvas, "Ctrl+D")
    assert len(tl(e).all_clips()) == 3 and snapshot(e) != before


class _KeyEater(QWidget):
    """A custom text-input widget (input methods enabled) that is not a QLineEdit: what the guard must protect without Qt's own help."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.keys: list[int] = []

    def keyPressEvent(self, ev):  # noqa: N802
        self.keys.append(ev.key())
        ev.accept()


def test_keys_are_never_stolen_from_text_entry(e):
    p, c = e.panel, e.panel.canvas
    ids = lay(e, 3)
    e.ws.set_selection(ids[:2])
    c.set_playhead(1.0)
    before = snapshot(e)
    edit = QLineEdit(p)
    edit.setGeometry(0, 0, 100, 20)
    edit.show()
    eater = _KeyEater(p)
    eater.setGeometry(0, 30, 100, 20)
    eater.show()
    for widget in (edit, eater, p.timecode):
        focus(widget)
        for key in ("s", "v", "b", "n", "l", "m", "i", "o", "a", "Del", "Backspace", "Space", "Left", "Right", "Up", "Down", "Home", "End", "Shift+Left", ",", ".", "Esc", "Shift+Z"):
            press(widget, key)
        assert snapshot(e) == before, widget  # not a single edit
        assert e.ws.edit_state.tool == "select" and e.ws.edit_state.snap.enabled and e.ws.edit_state.linked_selection and not e.ws.edit_state.ripple_on
        assert c.playhead == pytest.approx(1.0) and not p.transport.playing and tl(e).markers == [] and tl(e).range is None
        assert list(e.ws.selection.ids) == ids[:2]
    focus(edit)
    edit.clear()
    QTest.keyClicks(edit, "svb 1")
    assert edit.text() == "svb 1"  # typing still types
    focus(eater)
    press(eater, "Del")
    press(eater, "Ctrl+D")  # bound here, and a text widget has focus: delivered to the widget, not run
    assert Qt.Key.Key_Delete in eater.keys and Qt.Key.Key_D in eater.keys
    assert len(tl(e).all_clips()) == 3
    focus(c)
    e.ws.set_selection([ids[2]])  # the last clip: its copy fits behind it
    press(c, "Ctrl+D")  # back on the canvas the same key works
    assert len(tl(e).all_clips()) == 4
    edit.deleteLater()
    eater.deleteLater()


def test_an_action_triggered_while_a_text_field_has_focus_does_nothing(e):
    reg, p = e.panel.shortcuts, e.panel
    ids = lay(e, 2)
    e.ws.set_selection(ids)
    focus(p.timecode)
    reg.action("delete").trigger()  # as if Qt had delivered the shortcut anyway: the handler's own focus check refuses
    assert len(tl(e).all_clips()) == 2
    assert reg.trigger("delete", from_menu=True) is True  # from the menu / a button the text focus is irrelevant
    assert tl(e).all_clips() == []


def test_unavailable_actions_report_why_instead_of_doing_something(e):
    reg, c = e.panel.shortcuts, e.panel.canvas
    focus(c)
    press(c, "Del")
    assert e.status[-1] == "Select a clip first."
    press(c, "Ctrl+V")
    assert "nothing to paste" in e.status[-1].lower()
    press(c, "Alt+X")
    assert "no in / out range" in e.status[-1]
    press(c, "Shift+M")
    assert "no markers" in e.status[-1]
    press(c, "Space")
    assert "empty" in e.status[-1] and not e.panel.transport.playing
    assert reg.why_not("delete") == "Select a clip first." and reg.is_enabled("toggle_snap")
    assert not e.errors


# ------------------------------------------------------------------------------------------------------------------------ shortcuts: real effects through the keyboard
def test_tool_and_mode_keys(e):
    c, s = e.panel.canvas, e.ws.edit_state
    focus(c)
    press(c, "B")
    assert s.tool == "blade" and e.panel.blade_btn.isChecked()
    press(c, "V")
    assert s.tool == "select"
    press(c, "B")
    press(c, "S")
    assert s.tool == "select"
    press(c, "N")
    assert not s.snap.enabled and not e.panel.snap_btn.isChecked()
    press(c, "N")
    assert s.snap.enabled
    press(c, "Shift+R")
    assert s.ripple_on and e.panel.ripple_btn.text() == "Ripple ON"
    press(c, "Shift+R")
    assert not s.ripple_on
    press(c, "L")
    assert not s.linked_selection and not e.panel.link_btn.isChecked()
    press(c, "L")
    assert s.linked_selection


def test_editing_keys_delete_copy_cut_paste_duplicate_split(e):
    c = e.panel.canvas
    ids = lay(e, 3)  # a 0-6, b 6-10, c 10-15
    focus(c)
    e.ws.set_selection([ids[1]])
    press(c, "Delete")
    assert tl(e).get_clip(ids[1]) is None and tl(e).get_clip(ids[2]).timeline_start == pytest.approx(10.0)
    e.ws.undo()
    e.ws.set_selection([ids[1]])
    press(c, "Shift+Del")  # ripple delete closes the gap
    assert tl(e).get_clip(ids[2]).timeline_start == pytest.approx(6.0)
    e.ws.undo()
    e.ws.set_selection([ids[0]])
    press(c, "Ctrl+C")
    assert e.ws.edit.clipboard is not None
    c.set_playhead(20.0)
    press(c, "Ctrl+V")
    pasted = [x for x in tl(e).all_clips() if x.id not in ids]
    assert len(pasted) == 1 and pasted[0].timeline_start == pytest.approx(20.0) and pasted[0].asset_id == e.A.id
    e.ws.undo()
    e.ws.set_selection([ids[2]])
    press(c, "Ctrl+X")
    assert tl(e).get_clip(ids[2]) is None and e.ws.edit.clipboard is not None
    e.ws.undo()
    e.ws.set_selection([ids[2]])  # the last clip: its copy fits behind it
    n = len(tl(e).all_clips())
    press(c, "Ctrl+D")
    assert len(tl(e).all_clips()) == n + 1
    e.ws.undo()
    c.set_playhead(3.0)
    e.ws.set_selection([ids[0]])
    press(c, "Ctrl+B")
    assert sorted((round(x.timeline_start, 3), round(x.duration, 3)) for x in tl(e).get_track("track_v1").clips if x.id != ids[1] and x.id != ids[2]) == [(0.0, 3.0), (3.0, 3.0)]
    e.ws.undo()
    e.ws.selection.select_none()
    c.set_playhead(8.0)  # nothing selected: split every clip under the playhead
    press(c, "Ctrl+B")
    assert len(tl(e).get_track("track_v1").clips) == 4
    e.ws.undo()
    press(c, "Ctrl+Shift+B")  # cut all tracks at the playhead
    assert len(tl(e).get_track("track_v1").clips) == 4


def test_delete_key_removes_the_selected_marker_before_any_clip(e):
    c = e.panel.canvas
    ids = lay(e, 2)
    mid = e.ws.edit.add_marker(3.0, "intro")
    e.ws.selection.select_none()
    focus(c)
    press(c, "Del")  # no clip selected and no marker picked: refused with the reason
    assert e.status[-1] == "Select a clip first." and len(tl(e).markers) == 1
    r = c._pennant_rect(tl(e).get_marker(mid))
    QTest.mouseClick(c, Qt.MouseButton.LeftButton, pos=QPoint(int(r.center().x()), int(r.center().y())))  # pick the marker on the ruler
    assert e.panel.shortcuts.is_enabled("delete") and e.panel.shortcuts.is_enabled("ripple_delete")
    e.ws.set_selection(ids[:1])  # a clip is selected too: the marker the user picked wins (the canvas' own rule)
    press(c, "Del")
    assert tl(e).markers == [] and len(tl(e).all_clips()) == 2
    e.ws.undo()
    assert len(tl(e).markers) == 1
    c._sel_marker = None
    press(c, "Del")  # now the clip
    assert tl(e).get_clip(ids[0]) is None and len(tl(e).markers) == 1


def test_selection_keys(e):
    c = e.panel.canvas
    ids = lay(e, 3)
    focus(c)
    press(c, "Ctrl+A")
    assert set(e.ws.selection.ids) == set(ids)
    press(c, "Ctrl+Shift+A")
    assert not e.ws.selection
    press(c, "A")
    assert set(e.ws.selection.ids) == set(ids)
    e.ws.set_selection(ids[:1])
    press(c, "Ctrl+Alt+I")
    assert set(e.ws.selection.ids) == set(ids[1:])
    press(c, "Esc")
    assert not e.ws.selection  # nothing playing: Escape reaches the canvas (clear / cancel)


def test_playhead_navigation_keys(e):
    c = e.panel.canvas
    lay(e, 2)  # 0-6, 6-10
    tb = timebase_of(e.ws.project)
    focus(c)
    press(c, "End")
    assert c.playhead == pytest.approx(10.0)
    press(c, "Home")
    assert c.playhead == pytest.approx(0.0)
    press(c, "Right")
    assert c.playhead == pytest.approx(tb.to_seconds(1))
    press(c, "Right")
    press(c, "Left")
    assert c.playhead == pytest.approx(tb.to_seconds(1))
    press(c, "Shift+Right")
    assert c.playhead == pytest.approx(tb.to_seconds(1) + 1.0)
    press(c, "Shift+Left")
    assert c.playhead == pytest.approx(tb.to_seconds(1))
    c.set_playhead(1.0)
    press(c, "Down")
    assert c.playhead == pytest.approx(6.0)
    press(c, "Down")
    assert c.playhead == pytest.approx(10.0)
    press(c, "Up")
    assert c.playhead == pytest.approx(6.0)
    press(c, "Up")
    assert c.playhead == pytest.approx(0.0)


def test_marker_range_and_nudge_keys(e):
    c = e.panel.canvas
    ids = lay(e, 1)
    focus(c)
    c.set_playhead(2.0)
    press(c, "M")
    c.set_playhead(5.0)
    press(c, "M")
    assert [m.time for m in tl(e).markers] == [2.0, 5.0] and undo_text(e) == "Add marker"
    c.set_playhead(0.0)
    press(c, "Shift+M")
    assert c.playhead == pytest.approx(2.0)
    press(c, "Shift+M")
    assert c.playhead == pytest.approx(5.0)
    press(c, "Alt+M")
    assert c.playhead == pytest.approx(2.0)
    c.set_playhead(1.0)
    press(c, "I")
    c.set_playhead(4.0)
    press(c, "O")
    assert tl(e).range == (1.0, 4.0)
    press(c, "Alt+X")
    assert tl(e).range is None
    e.ws.set_selection(ids)
    tb = timebase_of(e.ws.project)
    start = tl(e).get_clip(ids[0]).timeline_start
    press(c, ".")
    assert tl(e).get_clip(ids[0]).timeline_start == pytest.approx(start + tb.frame_s, abs=1e-6)
    press(c, ",")
    assert tl(e).get_clip(ids[0]).timeline_start == pytest.approx(start, abs=1e-6)
    press(c, "Alt+Right")
    assert tl(e).get_clip(ids[0]).timeline_start == pytest.approx(start + 10 * tb.frame_s, abs=1e-6)
    press(c, "Alt+Left")
    assert tl(e).get_clip(ids[0]).timeline_start == pytest.approx(start, abs=1e-6)


def test_link_group_lock_keys(e):
    c = e.panel.canvas
    ids = lay(e, 2)
    focus(c)
    e.ws.set_selection(ids)
    press(c, "Ctrl+G")
    assert tl(e).get_clip(ids[0]).group_id and tl(e).get_clip(ids[0]).group_id == tl(e).get_clip(ids[1]).group_id
    press(c, "Ctrl+Shift+G")
    assert not tl(e).get_clip(ids[0]).group_id
    press(c, "Ctrl+L")
    assert tl(e).get_clip(ids[0]).link_id and tl(e).get_clip(ids[0]).link_id == tl(e).get_clip(ids[1]).link_id
    press(c, "Ctrl+Shift+L")
    assert not tl(e).get_clip(ids[0]).link_id
    press(c, "Ctrl+Alt+L")
    assert "item" in tl(e).get_clip(ids[0]).locks and "item" in tl(e).get_clip(ids[1]).locks
    press(c, "Ctrl+Alt+L")
    assert not tl(e).get_clip(ids[0]).locks


def test_zoom_keys(e):
    c = e.panel.canvas
    lay(e, 2)
    focus(c)
    pps = c.pps
    press(c, "Ctrl+=")
    assert c.pps > pps
    press(c, "Ctrl+-")
    assert c.pps == pytest.approx(pps, rel=0.02)
    press(c, "Shift+Z")
    assert c.pps * 10.0 <= c.viewport_width() + 1
    assert not e.panel.shortcuts.is_enabled("zoom_selection")
    e.ws.set_selection([tl(e).get_track("track_v1").clips[1].id])
    press(c, "Alt+Z")
    assert c.pps * 4.0 <= c.viewport_width() + 1 and c.pps > 40


# ------------------------------------------------------------------------------------------------------------------------ shortcuts: customisation, conflicts, persistence, dialog
def test_custom_binding_rebinds_live_and_old_key_stops_working(e):
    reg, c = e.panel.shortcuts, e.panel.canvas
    ids = lay(e, 2)
    focus(c)
    c.set_playhead(3.0)
    e.ws.set_selection(ids[:1])
    reg.set_binding("split", "Ctrl+K")
    assert reg.keys("split") == ["Ctrl+K"] and reg.is_overridden("split")
    press(c, "Ctrl+B")  # the old key is free now
    assert len(tl(e).get_track("track_v1").clips) == 2
    press(c, "Ctrl+K")
    assert len(tl(e).get_track("track_v1").clips) == 3
    assert _item(_tmenu(e), "Split at playhead").shortcut().toString() == "Ctrl+K"  # the Timeline menu shows the new key
    reg.reset("split")
    assert reg.keys("split") == ["Ctrl+B"] and not reg.is_overridden("split")
    assert e.ws.settings.timeline_shortcuts == {}


def test_conflicting_assignment_is_refused_naming_the_other_action(e):
    reg = e.panel.shortcuts
    with pytest.raises(ShortcutConflict) as err:
        reg.set_binding("split", "Ctrl+D")
    assert err.value.owner == "Duplicate" and "Duplicate" in str(err.value) and "Ctrl+D" in str(err.value)
    assert reg.keys("split") == ["Ctrl+B"] and reg.keys("duplicate") == ["Ctrl+D"]  # nothing changed
    with pytest.raises(ShortcutConflict) as err2:
        reg.set_binding("split", ["Ctrl+B", "Ctrl+S"])  # a main-window key
    assert err2.value.owner == "Save"
    with pytest.raises(ShortcutConflict) as err3:
        reg.set_binding("copy", "Ctrl+Y")
    assert err3.value.owner == "Redo"
    with pytest.raises(ValueError):
        reg.set_binding("split", "Ctrl+K, Ctrl+L")  # chords are not supported
    with pytest.raises(ValueError):
        reg.set_binding("split", ["F1", "F2", "F3"])
    assert e.ws.settings.timeline_shortcuts == {}
    assert reg.find_conflict("Ctrl+B", "split") is None and reg.find_conflict("Ctrl+B", "copy") == "Split at playhead"


def test_bindings_persist_and_load_tolerantly(e):
    reg, p = e.panel.shortcuts, e.panel
    reg.set_bindings({"split": ["Ctrl+K"], "toggle_snap": [], "zoom_fit": ["F", "Ctrl+F"]})
    import json

    saved = json.loads(e.paths.settings_file.read_text(encoding="utf-8"))["timeline_shortcuts"]
    assert saved == {"split": ["Ctrl+K"], "toggle_snap": [], "zoom_fit": ["F", "Ctrl+F"]}
    assert reg.keys("toggle_snap") == [] and "Snapping" in p.snap_btn.toolTip()
    # a hand-edited / damaged settings file: unknown ids, chords, junk types and a key taken by a main-window action are dropped; overrides beat defaults
    e.ws.settings = replace(e.ws.settings, timeline_shortcuts={"split": ["Ctrl+K, Ctrl+L"], "nope": ["X"], "delete": 5, "copy": ["Ctrl+X", "Ctrl+S", 7], "paste": "Ctrl+Shift+P", "mark_in": ["", "  "]})
    reg.apply()
    assert reg.keys("split") == ["Ctrl+B"] and reg.keys("mark_in") == ["I"]  # unusable entries (a chord, blanks) are ignored: the default stays
    assert reg.keys("delete") == ["Del", "Backspace"] and "nope" not in reg.bindings()
    assert reg.keys("copy") == ["Ctrl+X"] and reg.keys("cut") == []  # the override took the key that cut had by default
    assert reg.keys("paste") == ["Ctrl+Shift+P"]
    e.ws.settings = replace(e.ws.settings, timeline_shortcuts="not a dict")  # type: ignore[arg-type]
    reg.apply()
    assert reg.keys("split") == ["Ctrl+B"] and reg.keys("cut") == ["Ctrl+X"]


def test_custom_bindings_survive_a_restart(e):
    e.panel.shortcuts.set_bindings({"split": ["Ctrl+K"], "toggle_ripple": ["R"]})
    win2, ws2 = create_window(e.paths)
    try:
        reg2 = win2.timeline_panel.shortcuts
        assert reg2.keys("split") == ["Ctrl+K"] and reg2.keys("toggle_ripple") == ["R"] and reg2.keys("delete") == ["Del", "Backspace"]
        assert next(a for a in win2.timeline_menu.actions() if a.text() == "Split at playhead").shortcut().toString() == "Ctrl+K"
        assert reg2.action("split").shortcut().toString() == "Ctrl+K"
    finally:
        ws2.shutdown()
        win2.close()
        win2.deleteLater()
        QApplication.processEvents()


def test_shortcuts_dialog_edits_conflicts_reset_and_apply(e):
    reg = e.panel.shortcuts
    dlg = ShortcutsDialog(reg, e.win)
    try:
        assert dlg.table.rowCount() == len(SPECS)
        assert dlg.editors[("split", 0)].keySequence().toString() == "Ctrl+B" and dlg.editors[("tool_select", 1)].keySequence().toString() == "V"
        assert not dlg.apply_btn.isEnabled() and not dlg.reset_buttons["split"].isEnabled()
        assert "Save" in dlg.fixed_note.text() and "Ctrl+S" in dlg.fixed_note.text()  # the main window's fixed keys are listed
        dlg.editors[("split", 0)].setKeySequence(QKeySequence("Ctrl+D"))  # taken by Duplicate: refused on the spot
        assert "Duplicate" in dlg.message.text() and dlg.editors[("split", 0)].keySequence().toString() == "Ctrl+B"
        dlg.editors[("split", 0)].setKeySequence(QKeySequence("Ctrl+S"))  # taken by the main window
        assert "Save" in dlg.message.text() and dlg.editors[("split", 0)].keySequence().toString() == "Ctrl+B"
        assert not dlg.apply_btn.isEnabled()
        dlg.editors[("split", 0)].setKeySequence(QKeySequence("Ctrl+K"))
        assert dlg.message.text() == "" and dlg.pending["split"] == ["Ctrl+K"] and dlg.apply_btn.isEnabled() and dlg.reset_buttons["split"].isEnabled()
        assert reg.keys("split") == ["Ctrl+B"]  # not applied yet
        dlg.editors[("split", 1)].setKeySequence(QKeySequence("Ctrl+K"))  # the same key twice on one action
        assert dlg.pending["split"] == ["Ctrl+K"]
        dlg.editors[("tool_select", 1)].setKeySequence(QKeySequence())  # clear the alternative
        assert dlg.pending["tool_select"] == ["S"]
        assert dlg.apply() is True and reg.keys("split") == ["Ctrl+K"] and reg.keys("tool_select") == ["S"] and not dlg.apply_btn.isEnabled()
        assert e.ws.settings.timeline_shortcuts == {"split": ["Ctrl+K"], "tool_select": ["S"]}
        dlg.reset_one("split")
        assert dlg.pending["split"] == ["Ctrl+B"] and dlg.apply_btn.isEnabled()
        dlg.reset_all()
        assert dlg.pending == reg.defaults() and dlg.apply()
        assert reg.keys("split") == ["Ctrl+B"] and reg.keys("tool_select") == ["S", "V"] and e.ws.settings.timeline_shortcuts == {}
        dlg.filter.setText("blade")
        assert not dlg.table.isRowHidden(dlg._rows["tool_blade"]) and dlg.table.isRowHidden(dlg._rows["delete"])
        dlg.filter.setText("")
        # a reset that would collide with another action's pending key is refused
        dlg.editors[("duplicate", 0)].setKeySequence(QKeySequence("Ctrl+H"))
        dlg.editors[("split", 0)].setKeySequence(QKeySequence("Ctrl+D"))
        assert dlg.pending["split"] == ["Ctrl+D"] and dlg.pending["duplicate"] == ["Ctrl+H"]
        assert dlg.reset_one("duplicate") is False and "Split at playhead" in dlg.message.text()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_dialog_is_reachable_from_the_edit_menu_and_the_toolbar(e, monkeypatch):
    opened = []
    monkeypatch.setattr("app.ui.dialogs.shortcuts_dialog.ShortcutsDialog.exec", lambda self: opened.append(self) or 0)
    e.win.shortcuts_action.trigger()
    QTest.mouseClick(e.panel.keys_btn, Qt.MouseButton.LeftButton)
    assert len(opened) == 2
    labels = [a.text() for a in e.win.menuBar().actions()]
    assert labels == ["&File", "&Edit", "&Timeline"]
    edit_menu = e.win.menuBar().actions()[1].menu()
    assert "Keyboard Shortcuts…" in [a.text() for a in edit_menu.actions()]
    timeline_menu = e.win.timeline_menu
    assert any(a.text() == "Keyboard Shortcuts…" for a in timeline_menu.actions())
    next(a for a in timeline_menu.actions() if a.text() == "Keyboard Shortcuts…").trigger()
    assert len(opened) == 3


# ------------------------------------------------------------------------------------------------------------------------ Timeline menu
def _tmenu(e) -> dict:
    e.win.timeline_menu.aboutToShow.emit()
    return _menu_actions(e.win.timeline_menu)


def _item(menu: dict, label: str):
    return next(a for k, a in menu.items() if k.split(" > ")[-1] == label)


def test_timeline_menu_shows_current_keys_and_enables_only_what_is_possible(e):
    m = _tmenu(e)
    for label in ("Cut", "Copy", "Paste at playhead", "Paste attributes", "Duplicate", "Split at playhead", "Cut all tracks at playhead", "Delete", "Ripple delete", "Select all", "Select none",
                  "Invert selection", "Link clips", "Unlink clips", "Group clips", "Ungroup clips", "Lock / unlock clips", "Add marker", "Mark in", "Mark out", "Clear in / out range",
                  "Select tool", "Blade tool", "Go to start", "Go to end", "Next marker", "Previous marker", "Next edit point", "Zoom in", "Zoom out", "Zoom to fit", "Zoom to selection",
                  "Snapping", "Ripple editing", "Linked selection", "Play / pause", "Keyboard Shortcuts…"):
        assert any(k.split(" > ")[-1] == label for k in m), label
    assert _item(m, "Split at playhead").shortcut().toString() == "Ctrl+B" and _item(m, "Cut").shortcut().toString() == "Ctrl+X"
    # empty project: only the always-possible items are enabled
    for label in ("Cut", "Copy", "Paste at playhead", "Delete", "Ripple delete", "Split at playhead", "Select all", "Link clips", "Clear in / out range", "Play / pause", "Next marker"):
        assert not _item(m, label).isEnabled(), label
        assert _item(m, label).toolTip(), label  # says why
    for label in ("Add marker", "Mark in", "Select tool", "Blade tool", "Snapping", "Zoom in", "Keyboard Shortcuts…"):
        assert _item(m, label).isEnabled(), label
    ids = lay(e, 3)
    m = _tmenu(e)
    assert _item(m, "Split at playhead").isEnabled() and _item(m, "Select all").isEnabled() and _item(m, "Play / pause").isEnabled()
    assert not _item(m, "Delete").isEnabled() and "Select a clip first." in _item(m, "Delete").toolTip()
    e.ws.set_selection(ids[:1])
    m = _tmenu(e)
    assert _item(m, "Delete").isEnabled() and _item(m, "Copy").isEnabled() and _item(m, "Duplicate").isEnabled() and _item(m, "Unlink clips").isEnabled()
    assert not _item(m, "Link clips").isEnabled() and "two or more" in _item(m, "Link clips").toolTip() and not _item(m, "Group clips").isEnabled()
    assert not _item(m, "Paste at playhead").isEnabled() and not _item(m, "Paste attributes").isEnabled()
    e.ws.set_selection(ids[:2])
    m = _tmenu(e)
    assert _item(m, "Link clips").isEnabled() and _item(m, "Group clips").isEnabled()
    e.ws.edit.copy()
    e.ws.edit.add_marker(1.0)
    e.ws.edit.set_range(1.0, 2.0)
    m = _tmenu(e)
    assert _item(m, "Paste at playhead").isEnabled() and _item(m, "Next marker").isEnabled() and _item(m, "Clear in / out range").isEnabled()
    e.ws.edit.set_track_flag("track_v1", "locked", True)
    m = _tmenu(e)
    assert not _item(m, "Delete").isEnabled() and "locked" in _item(m, "Delete").toolTip()


def test_timeline_menu_checkable_items_reflect_and_change_the_state(e):
    s = e.ws.edit_state
    m = _tmenu(e)
    assert _item(m, "Select tool").isChecked() and not _item(m, "Blade tool").isChecked()
    assert _item(m, "Snapping").isChecked() and _item(m, "Linked selection").isChecked() and not _item(m, "Ripple editing").isChecked()
    _item(m, "Blade tool").trigger()
    _item(m, "Ripple editing").trigger()
    _item(m, "Snapping").trigger()
    assert s.tool == "blade" and s.ripple_on and not s.snap.enabled
    m = _tmenu(e)
    assert _item(m, "Blade tool").isChecked() and not _item(m, "Select tool").isChecked() and _item(m, "Ripple editing").isChecked() and not _item(m, "Snapping").isChecked()


def test_timeline_menu_items_run_the_same_actions_and_never_double_fire_with_the_keys(e):
    c = e.panel.canvas
    ids = lay(e, 2)
    m = _tmenu(e)
    c.set_playhead(2.0)
    _item(m, "Add marker").trigger()
    assert len(tl(e).markers) == 1  # by menu
    focus(c)
    press(c, "M")
    assert len(tl(e).markers) == 2  # by key: once, not twice and not zero times (the menu's copy of the key never competes with the panel's action)
    press(c, "N")
    assert not e.ws.edit_state.snap.enabled
    _item(_tmenu(e), "Snapping").trigger()
    assert e.ws.edit_state.snap.enabled
    e.ws.set_selection([ids[1]])
    n = len(tl(e).all_clips())
    _item(_tmenu(e), "Duplicate").trigger()
    assert len(tl(e).all_clips()) == n + 1
    _item(_tmenu(e), "Go to end").trigger()
    assert c.playhead == pytest.approx(tl(e).duration)


def test_menu_item_runs_even_while_a_text_field_has_focus(e):
    ids = lay(e, 2)
    e.ws.set_selection(ids[:1])
    focus(e.panel.timecode)
    _item(_tmenu(e), "Delete").trigger()
    assert tl(e).get_clip(ids[0]) is None


# ------------------------------------------------------------------------------------------------------------------------ play transport
def test_space_plays_with_a_fake_clock_frame_accurately_and_stops_at_the_end(e):
    p, c = e.panel, e.panel.canvas
    lay(e, 1)  # 6 s
    tb = timebase_of(e.ws.project)
    now = [100.0]
    p.transport.clock = lambda: now[0]
    focus(c)
    press(c, "Space")
    assert p.transport.playing and p.play_btn.isChecked() and p.play_btn.text().endswith("Pause") and "Playing" in p.status_line.text()
    for dt in (0.5, 1.01, 2.0):
        now[0] = 100.0 + dt
        p.transport.tick()
        assert c.playhead == pytest.approx(tb.snap(dt, "floor"), abs=1e-9) and p.transport.playing, dt
    now[0] = 100.0 + 2.0 + 1 / 60  # less than a frame later: no movement
    p.transport.tick()
    assert c.playhead == pytest.approx(2.0)
    now[0] = 100.0 + 2.5
    p.transport.tick()
    assert c.playhead == pytest.approx(tb.snap(2.5, "floor")) and p.timecode.text() == tb.format(2.5)  # the timecode field follows
    now[0] = 100.0 + 9.0  # past the end
    p.transport.tick()
    assert c.playhead == pytest.approx(6.0) and not p.transport.playing and not p.play_btn.isChecked() and p.play_btn.text().endswith("Play")
    assert "end of the timeline" in e.status[-1]
    press(c, "Space")  # at the end, Space plays again from the start
    assert p.transport.playing and c.playhead == pytest.approx(0.0)
    press(c, "Space")
    assert not p.transport.playing


def test_shift_space_plays_only_the_in_out_range(e):
    p, c = e.panel, e.panel.canvas
    lay(e, 1)  # 6 s
    now = [50.0]
    p.transport.clock = lambda: now[0]
    focus(c)
    press(c, "Shift+Space")
    assert not p.transport.playing and any("no in / out range" in m for m in e.status)  # without a range there is nothing to play
    e.ws.edit.set_range(2.0, 4.0)
    c.set_playhead(5.0)  # anywhere: range playback starts at the range start
    press(c, "Shift+Space")
    assert p.transport.playing and c.playhead == pytest.approx(2.0)
    now[0] = 50.0 + 1.0
    p.transport.tick()
    assert c.playhead == pytest.approx(3.0) and p.transport.playing
    now[0] = 50.0 + 2.5  # past the range end (but inside the timeline)
    p.transport.tick()
    assert c.playhead == pytest.approx(4.0) and not p.transport.playing and "end of the range" in e.status[-1]
    press(c, "Space")  # plain Space still plays to the end of the timeline
    assert p.transport.playing and p.transport._end is None
    press(c, "Space")


def test_playback_stops_on_manual_playhead_changes_edits_escape_and_leaving_the_page(e):
    p, c = e.panel, e.panel.canvas
    ids = lay(e, 2)
    focus(c)

    def play():
        p.transport.clock = lambda: 0.0
        c.set_playhead(1.0)
        press(c, "Space")
        assert p.transport.playing

    play()
    c.set_playhead(3.0)  # the user scrubbed
    assert not p.transport.playing
    play()
    p.timecode.setText("00:00:04:00")
    QTest.keyClick(p.timecode, Qt.Key.Key_Return)  # typed a jump
    assert not p.transport.playing and c.playhead == pytest.approx(4.0)
    focus(c)
    play()
    e.ws.edit.set_track_flag("track_v2", "muted", True)  # any edit
    assert not p.transport.playing
    play()
    e.ws.set_selection(ids[:1])
    press(c, "Esc")  # Escape stops playback first and leaves the selection alone
    assert not p.transport.playing and list(e.ws.selection.ids) == ids[:1]
    play()
    e.ws.undo()
    assert not p.transport.playing
    play()
    e.win.go_to("Project")
    QApplication.processEvents()
    assert not p.transport.playing
    e.win.go_to("Timeline")


def test_space_plays_in_real_time_and_stops_by_itself_at_the_end(e):
    p, c = e.panel, e.panel.canvas
    ids = lay(e, 1)
    e.ws.edit.set_duration(ids[0], 0.5)
    assert tl(e).duration == pytest.approx(0.5)
    c.set_playhead(0.0)
    seen: list[float] = []
    c.playhead_changed.connect(seen.append)
    focus(c)
    press(c, "Space")
    assert p.transport.playing
    pump(lambda: not p.transport.playing, timeout=10)
    assert c.playhead == pytest.approx(0.5) and len(seen) >= 3  # it advanced in steps, not in one jump
    assert seen == sorted(seen)  # forward only
    tb = timebase_of(e.ws.project)
    assert all(abs(t / tb.frame_s - round(t / tb.frame_s)) < 1e-6 for t in seen)  # every position is on a frame boundary


def test_play_button_and_empty_timeline(e):
    p = e.panel
    assert not p.play_btn.isEnabled() and "empty" in p.play_btn.toolTip()
    lay(e, 1)
    assert p.play_btn.isEnabled()
    QTest.mouseClick(p.play_btn, Qt.MouseButton.LeftButton)
    assert p.transport.playing
    QTest.mouseClick(p.play_btn, Qt.MouseButton.LeftButton)
    assert not p.transport.playing and not p.play_btn.isChecked()


# ------------------------------------------------------------------------------------------------------------------------ main-window wiring
def test_request_signals_reach_the_main_window(e):
    ids = lay(e, 1)
    e.ws.selection.select_none()
    e.panel.request_inspector.emit(ids[0])
    assert e.ws.selected_clip_id == ids[0]
    e.panel.request_inspector.emit("no-such-clip")  # a vanished clip is ignored
    assert e.ws.selected_clip_id == ids[0]
    asked: list[tuple] = []
    insp = e.win.inspector
    for name in ("open_speed_dialog", "open_replace_dialog"):
        setattr(insp, name, lambda cid, n=name: asked.append((n, cid)))
    try:
        e.panel.request_speed_dialog.emit(ids[0])
        e.panel.request_replace_dialog.emit(ids[0])
        assert asked == [("open_speed_dialog", ids[0]), ("open_replace_dialog", ids[0])]
    finally:
        for name in ("open_speed_dialog", "open_replace_dialog"):
            if name in insp.__dict__:
                delattr(insp, name)
    if not hasattr(type(insp), "open_speed_dialog"):
        e.status.clear()
        e.panel.request_speed_dialog.emit(ids[0])
        assert "not available" in e.status[-1]


def test_canvas_request_signals_are_forwarded_when_the_canvas_defines_them(e):
    c = e.panel.canvas
    if not hasattr(c, "request_inspector"):
        pytest.skip("the canvas rewrite (agent U1) does not define request_inspector yet")
    got = []
    e.panel.request_inspector.connect(got.append)
    c.request_inspector.emit("x")
    assert got == ["x"]


def test_library_selection_enables_add_selected_media(e):
    pump(lambda: e.win.library.list.count() >= 5, timeout=20)
    e.panel.selected_provider = e.win.library.selected_ids
    e.win.library.list.clearSelection()
    assert not e.panel.add_selected_btn.isEnabled()
    e.win.library.list.item(0).setSelected(True)  # the window listens to the list's selection signal
    assert e.panel.add_selected_btn.isEnabled()
    e.win.library.list.clearSelection()
    assert not e.panel.add_selected_btn.isEnabled()


def test_library_context_add_goes_through_the_edit_service(e):
    e.win._add_to_timeline(e.A.id)
    assert undo_text(e) == "Add media" and [c.asset_id for c in tl(e).all_clips()] == [e.A.id]


def test_theme_change_restyles_the_toolbar(e):
    before = e.panel.styleSheet()
    ws = e.ws
    ws.settings = replace(ws.settings, theme="light")
    try:
        e.win._apply_settings()
        assert e.panel.styleSheet() != before
    finally:
        ws.settings = replace(ws.settings, theme="dark")
        e.win._apply_settings()
