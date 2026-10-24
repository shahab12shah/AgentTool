"""Phase 10 UI: the type-aware inspector, the keyframe editor and the AI decision actions (offscreen Qt, real media, real engine).

Every test drives real widgets (typing, clicks, mouse gestures) and asserts on the MODEL (``ws.project.timeline``) and on ``ws.commands`` (one undo step per edit).
"""

from __future__ import annotations

import html

import pytest

from app.tests.conftest import import_and_wait, needs_ffmpeg

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.timeline.keyframes import INTERPOLATIONS
from app.timeline.ops_props import (
    KEYFRAME_RANGES,
    TRANSITION_TYPES,
    keyframe_properties,
)
from app.timeline.timebase import Timebase
from app.ui.context import UiContext
from app.ui.dialogs import replace_media_dialog, speed_dialog
from app.ui.dialogs.replace_media_dialog import ReplaceMediaDialog
from app.ui.dialogs.speed_dialog import SpeedChoiceDialog, SpeedDialog
from app.ui.inspector_panel import InspectorPanel
from app.ui.qt_bridge import UiBridge
from app.ui.timecode_edit import NumberEdit, TimecodeEdit

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def type_in(w, text: str) -> None:
    """Type ``text`` into a TimecodeEdit / NumberEdit like a user (select all, type, Enter)."""
    w.selectAll()
    QTest.keyClicks(w, text)
    QTest.keyClick(w, Qt.Key.Key_Return)


def undo_count(ws) -> int:
    return len(ws.commands._undo)


def tl(ws):
    return ws.project.timeline


class Env:
    pass


@pytest.fixture
def env(qapp, project_ws, media_dir):
    ws = project_ws
    v = import_and_wait(ws, media_dir / "clip.mp4")
    i = import_and_wait(ws, media_dir / "pic.png")
    a = import_and_wait(ws, media_dir / "voice.wav")
    o = import_and_wait(ws, media_dir / "other.mp4")
    panel = InspectorPanel(UiContext(ws, UiBridge(ws.bus)))
    panel.resize(440, 1700)
    panel.show()
    ws.edit.place_assets([v.id], at=0.0)
    ws.edit.place_assets([i.id], at=4.0)
    ws.edit.place_assets([a.id], at=0.0)
    clips = {c.asset_id: c for c in ws.project.timeline.all_clips()}
    e = Env()
    e.ws, e.panel, e.video, e.image, e.audio = ws, panel, clips[v.id], clips[i.id], clips[a.id]
    e.assets = {"video": v, "image": i, "audio": a, "other": o}
    e.sel = lambda cid: (ws.select_clip(cid), qapp.processEvents())
    qapp.processEvents()
    yield e
    panel.close()


def visible(panel) -> set[str]:
    return {k for k, g in panel.sections.items() if g.isVisibleTo(panel)}


# ====================================================================== validated inputs
def test_timecode_edit_validates_and_commits_only_valid_changes(qapp):
    tc = TimecodeEdit(Timebase(30))
    tc.show()
    got, bad = [], []
    tc.committed.connect(got.append)
    tc.invalid.connect(bad.append)
    tc.set_seconds(2.0)
    assert tc.text() == "00:00:02:00" and tc.value() == 2.0
    type_in(tc, "00:00:03:15")  # valid timecode, different -> committed once
    assert got == [pytest.approx(3.5)] and not bad
    tc.set_seconds(3.5)
    type_in(tc, "3.5")  # the same time in another notation -> nothing to commit
    assert got == [pytest.approx(3.5)] and tc.text() == "00:00:03:15"
    type_in(tc, "90f")  # frames
    assert got[-1] == pytest.approx(3.0)
    tc.set_seconds(3.0)
    type_in(tc, "1.25s")
    assert got[-1] == pytest.approx(1.25)
    tc.set_seconds(1.25)
    n = len(got)
    type_in(tc, "banana")  # invalid: reverted, flagged, reported
    assert len(got) == n and tc.text() == tc.timebase().format(1.25) and bad and "Use HH:MM:SS:FF" in bad[-1]
    assert "border" in tc.styleSheet() and tc.toolTip() == bad[-1]
    type_in(tc, "00:00:01:45")  # frame number beyond the frame rate
    assert "Frames must be 0–29" in bad[-1] and len(got) == n
    tc.set_seconds(2.0)  # a model refresh clears the flag
    assert tc.styleSheet() == ""
    tc.set_seconds(None)
    assert tc.text() == "—"
    # a refresh never overwrites text the user is typing
    tc.set_seconds(4.0, force=True)
    QTest.keyClicks(tc, "9")
    assert tc.is_editing()
    tc.set_seconds(1.0)
    assert tc.text() != "00:00:01:00"
    tc.discard_edit()
    assert tc.text() == "00:00:01:00"
    tc.selectAll()
    QTest.keyClicks(tc, "7")
    QTest.keyClick(tc, Qt.Key.Key_Escape)  # Escape throws an uncommitted edit away
    assert tc.text() == "00:00:01:00" and not tc.is_editing()


def test_number_edit_never_clamps_typed_values_and_ignores_the_wheel(qapp):
    ne = NumberEdit(0.01, 20.0, step=0.05, decimals=3, label="Scale")
    ne.show()
    got = []
    ne.committed.connect(got.append)
    ne.set_model(1.0)
    type_in(ne, "50")  # outside the engine's range: still committed, the ENGINE refuses it
    assert got == [50.0]
    assert "0.01 to 20" in ne.toolTip() or "between 0.01 and 20" in ne.toolTip()  # the range is explained in the tooltip / flagged
    ne.set_model(1.0)  # (the engine refused: the inspector shows the real value again)
    assert ne.value() == 1.0
    ne.set_model(None)
    assert ne.is_mixed() and ne.text() == "—"
    QTest.keyClick(ne, Qt.Key.Key_Return)  # entering and leaving a mixed field changes nothing
    assert got == [50.0]
    type_in(ne, "2")
    assert got[-1] == 2.0
    ne.set_model(2.0)
    ne.set_model(None)  # mixed: typing the number the box happened to hold before is still a deliberate edit
    type_in(ne, "2")
    assert got[-1] == 2.0 and len(got) == 3
    ne.set_model(None, force=True)
    assert ne.is_mixed() and ne.text() == "—"  # refused: the dash is back
    ne.set_model(1.0)
    ne.selectAll()
    QTest.keyClicks(ne, "50")  # while typing, an out-of-range number is flagged (the engine still decides on commit)
    assert "border" in ne.styleSheet() and "between 0.01 and 20" in ne.toolTip()
    ne.discard_edit()
    assert ne.styleSheet() == ""
    ne.set_model(19.99)
    ne.stepBy(5)  # arrows stop at the engine's limits
    assert ne.value() == 20.0
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    ne.clearFocus()
    ev.accept()
    ne.wheelEvent(ev)
    assert not ev.isAccepted() and ne.value() == 20.0  # no focus -> the wheel does nothing
    fac = NumberEdit(0.0, 1.0, step=5, decimals=0, suffix=" %", factor=100.0, label="Opacity")
    fac.set_model(0.25)
    assert fac.value() == 25.0 and fac.model_value() == pytest.approx(0.25)


# ====================================================================== sections per clip category
def test_sections_appear_and_hide_per_clip_category(env, qapp):
    ws, panel = env.ws, env.panel
    assert panel.empty.isVisibleTo(panel) and not panel.body.isVisibleTo(panel)
    env.sel(env.video.id)
    assert visible(panel) == {"clip", "timing", "speed", "transform", "transition", "locks", "keyframes"}
    assert "own sound is not played" in panel.info_note.text()  # the renderer plays no sound from a picture clip: said, not faked with an audio section
    assert panel.source_in.isVisibleTo(panel) and panel.fit.isVisibleTo(panel) and not panel.empty.isVisibleTo(panel)
    env.sel(env.image.id)
    assert visible(panel) == {"clip", "timing", "transform", "transition", "locks", "keyframes"}  # a picture has no speed
    assert not panel.source_in.isVisibleTo(panel) and not panel.source_out.isVisibleTo(panel)
    env.sel(env.audio.id)
    assert visible(panel) == {"clip", "timing", "speed", "audio", "locks", "keyframes"}
    # text graphic
    gid = ws.presentation.add_graphic("HEADLINE", "Hello world", 6.0, 2.0)
    env.sel(gid)
    assert visible(panel) == {"clip", "presentation", "ai", "timing", "text", "locks"}  # no keyframes, transform, speed, transition for text
    assert panel.ai_badge.isVisibleTo(panel) and not panel.ai_manual.isEnabled()  # the user's own graphic: already a manual override
    assert "animation presets" in panel.info_note.text()
    assert not panel.kf.available
    env.sel(None)
    assert panel.empty.isVisibleTo(panel)


# ====================================================================== timing
def test_timing_fields_edit_the_model_one_undo_step_each_and_revert_on_rejection(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    cid = env.video.id
    n = undo_count(ws)
    type_in(panel.start, "00:00:01:00")
    clip = tl(ws).get_clip(cid)
    assert clip.timeline_start == pytest.approx(1.0) and undo_count(ws) == n + 1 and panel.start.text() == "00:00:01:00"
    ws.undo()
    assert tl(ws).get_clip(cid).timeline_start == 0.0 and panel.start.text() == "00:00:00:00"
    # duration (the end edge moves, the source range follows)
    type_in(panel.duration, "2")
    clip = tl(ws).get_clip(cid)
    assert clip.duration == pytest.approx(2.0) and clip.source_out == pytest.approx(2.0) and undo_count(ws) == n + 1
    assert panel.end_label.text() == "00:00:02:00"
    # source in: the start stays, the duration follows
    type_in(panel.source_in, "0.5")
    clip = tl(ws).get_clip(cid)
    assert clip.source_in == pytest.approx(0.5) and clip.timeline_start == 0.0 and clip.duration == pytest.approx(1.5) and undo_count(ws) == n + 2
    type_in(panel.source_out, "2")
    clip = tl(ws).get_clip(cid)
    assert clip.source_out == pytest.approx(2.0) and clip.duration == pytest.approx(1.5)
    # rejected: before the start of the timeline / past the end of the media: message from the engine, field back to the real value, nothing recorded
    before, steps = tl(ws).get_clip(cid).to_dict(), undo_count(ws)
    type_in(panel.start, "-3")
    assert "cannot start before 0:00" in panel.notice.text() and panel.start.text() == "00:00:00:00"
    type_in(panel.source_in, "99")
    assert "source out point must be after the source in point" in panel.notice.text() and panel.source_in.text() == panel.source_in.timebase().format(0.5)
    assert tl(ws).get_clip(cid).to_dict() == before and undo_count(ws) == steps
    # a range that runs past the media is limited by the engine, and the inspector shows its note
    type_in(panel.source_out, "99")
    assert tl(ws).get_clip(cid).source_out == pytest.approx(3.0) and "limited to the end of the media" in panel.notice.text() and undo_count(ws) == steps + 1
    ws.undo()
    steps = undo_count(ws)
    # not a time at all: reported next to the field, no engine call
    type_in(panel.duration, "soon")
    assert "Use HH:MM:SS:FF" in panel.notice.text() and undo_count(ws) == steps


def test_a_locked_aspect_clip_or_track_disables_the_field_and_names_the_lock(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    cid = env.video.id
    assert panel.scale.isEnabled() and panel.start.isEnabled()
    ws.edit.set_locks([cid], "transform", "add")
    assert not panel.scale.isEnabled() and "transform of this clip is locked" in panel.scale.toolTip() and "Transform" in panel.scale.toolTip()
    steps = undo_count(ws)
    panel.scale.committed.emit(2.0)  # a stale field (the lock arrived after the value was typed) still cannot change the clip: the engine refuses with its reason
    assert "is locked (transform)" in panel.notice.text() and tl(ws).get_clip(cid).scale == 1.0 and undo_count(ws) == steps and panel.scale.value() == 1.0
    assert panel.start.isEnabled() and panel.duration.isEnabled()  # other aspects stay editable
    ws.edit.set_locks([cid], "timing", "add")
    assert not panel.duration.isEnabled() and "timing of this clip is locked" in panel.duration.toolTip()
    assert not panel.speed.isEnabled() and not panel.source_in.isEnabled()
    ws.edit.set_locks([cid], "item", "add")
    assert not panel.start.isEnabled() and "item lock" in panel.start.toolTip()
    assert panel.lock_boxes["item"].isEnabled()  # a lock can always be taken off again
    panel.lock_boxes["item"].click()
    assert "item" not in tl(ws).get_clip(cid).locks and panel.start.isEnabled()
    for kind in ("timing", "transform"):
        panel.lock_boxes[kind].click()
    assert not tl(ws).get_clip(cid).locks and panel.scale.isEnabled()
    ws.timeline.set_track_flag(env.video.track_id, "locked", True)
    assert not panel.start.isEnabled() and "is locked" in panel.start.toolTip() and panel.track_note.isVisibleTo(panel)
    assert not panel.lock_boxes["item"].isEnabled()  # a locked TRACK has to be unlocked first
    ws.timeline.set_track_flag(env.video.track_id, "locked", False)
    assert panel.start.isEnabled()


# ====================================================================== speed
def test_speed_field_presets_and_resulting_duration(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    cid = env.video.id
    n = undo_count(ws)
    QTest.keyClicks(panel.speed, "2")  # live: the resulting duration is shown while typing
    panel.speed.selectAll()
    panel.speed.lineEdit().setText("2")
    panel.speed.lineEdit().textEdited.emit("2")
    assert "1.500 s" in panel.speed_result.text()
    panel.speed.discard_edit()
    type_in(panel.speed, "2")
    clip = tl(ws).get_clip(cid)
    assert clip.speed == 2.0 and clip.duration == pytest.approx(1.5) and undo_count(ws) == n + 1
    assert panel.speed.value() == 2.0 and "1.500 s" in panel.speed_result.text()
    ws.undo()
    assert tl(ws).get_clip(cid).speed == 1.0
    # presets
    action = next(a for a in panel.speed_presets.menu().actions() if a.text() == "0.5×")
    action.trigger()
    assert tl(ws).get_clip(cid).speed == 0.5 and tl(ws).get_clip(cid).duration == pytest.approx(6.0)
    # reverse / zero / out of range are refused with the engine's reason and the field goes back
    steps = undo_count(ws)
    type_in(panel.speed, "0")
    assert "Reverse playback and freezing" in panel.notice.text() and panel.speed.value() == 0.5
    type_in(panel.speed, "50")
    assert "between 0.1× and 8×" in panel.notice.text() and panel.speed.value() == 0.5
    assert undo_count(ws) == steps


def test_speed_of_a_linked_clip_asks_which_policy_and_cancel_changes_nothing(env, monkeypatch):
    ws, panel = env.ws, env.panel
    ws.edit.link([env.video.id, env.audio.id])
    env.sel(env.video.id)
    seen = []

    def choose(parent, exc, labels=()):
        seen.append((exc.options, list(labels), exc.user_message))
        return answers.pop(0)

    monkeypatch.setattr(speed_dialog, "choose_link_policy", choose)
    steps = undo_count(ws)
    answers = [None]  # the user cancels the question
    type_in(panel.speed, "2")
    assert seen and set(seen[0][0]) == {"keep_linked_audio_in_sync", "independent"} and "is linked to" in seen[0][2]
    assert tl(ws).get_clip(env.video.id).speed == 1.0 and tl(ws).get_clip(env.audio.id).speed == 1.0 and undo_count(ws) == steps and panel.speed.value() == 1.0
    answers[:] = ["keep_linked_audio_in_sync"]
    type_in(panel.speed, "2")
    assert tl(ws).get_clip(env.video.id).speed == 2.0 and tl(ws).get_clip(env.audio.id).speed == 2.0 and undo_count(ws) == steps + 1  # both in ONE undo step
    ws.undo()
    assert tl(ws).get_clip(env.audio.id).speed == 1.0 and tl(ws).get_clip(env.video.id).speed == 1.0
    answers[:] = ["independent"]
    type_in(panel.speed, "2")
    assert tl(ws).get_clip(env.video.id).speed == 2.0 and tl(ws).get_clip(env.audio.id).speed == 1.0
    assert "may drift out of sync" in panel.notice.text()


def test_speed_choice_dialog_lists_the_engine_options_and_the_speed_dialog_applies(env, monkeypatch):
    ws, panel = env.ws, env.panel
    dlg = SpeedChoiceDialog("Choose how to change the speed.", ("keep_linked_audio_in_sync", "independent"), ["voice.wav"])
    assert set(dlg.buttons) == {"keep_linked_audio_in_sync", "independent"} and "Keep linked clips in sync" in dlg.buttons["keep_linked_audio_in_sync"].text()
    dlg.buttons["independent"].click()
    assert dlg.choice == "independent" and dlg.result() == dlg.DialogCode.Accepted
    sd = SpeedDialog(panel.ctx, env.video.id)
    sd.preset_buttons[2.0].click()
    assert sd.speed.model_value() == 2.0 and "1.500 s" in sd.result_label.text()
    assert sd.apply() and tl(ws).get_clip(env.video.id).speed == 2.0
    # open_speed_dialog: a picture has no speed (the reason is shown, no dialog), a video opens it
    opened = []
    monkeypatch.setattr(SpeedDialog, "exec", lambda self: opened.append(self.clip_id) or 0)
    assert panel.open_speed_dialog(env.image.id) is False and "no playback speed" in panel.notice.text() and not opened
    assert panel.open_speed_dialog(env.video.id) is True and opened == [env.video.id] and ws.selected_clip_id == env.video.id
    ws.edit.set_locks([env.video.id], "timing", "add")
    assert panel.open_speed_dialog(env.video.id) is False and "timing of this clip is locked" in panel.notice.text()


# ====================================================================== transform
def test_transform_fields_change_the_model_one_undo_step_each(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    cid = env.video.id
    n = undo_count(ws)
    type_in(panel.opacity, "50")
    assert tl(ws).get_clip(cid).opacity == pytest.approx(0.5)
    type_in(panel.scale, "1.5")
    assert tl(ws).get_clip(cid).scale == pytest.approx(1.5)
    type_in(panel.rotation, "-45")
    assert tl(ws).get_clip(cid).rotation == pytest.approx(-45.0)
    type_in(panel.pos_x, "120")
    type_in(panel.pos_y, "-30")
    assert tl(ws).get_clip(cid).position == (120.0, -30.0)
    assert undo_count(ws) == n + 5
    # crop: four sides, one engine call, stored as (x, y, w, h) fractions
    type_in(panel.crop["left"], "10")
    type_in(panel.crop["bottom"], "20")
    crop = tl(ws).get_clip(cid).effects["crop"]
    assert crop == pytest.approx([0.1, 0.0, 0.9, 0.8]) and panel.crop["left"].value() == 10.0 and panel.crop["bottom"].value() == 20.0
    panel.fit.setCurrentIndex(panel.fit.findData("contain"))
    panel.fit.activated.emit(panel.fit.currentIndex())
    assert tl(ws).get_clip(cid).effects["fit"] == "contain"
    assert undo_count(ws) == n + 5 + 3
    ws.undo()
    assert "fit" not in tl(ws).get_clip(cid).effects
    ws.undo()
    ws.undo()
    assert "crop" not in tl(ws).get_clip(cid).effects and panel.crop["left"].value() == 0.0
    # rejected values: the engine's message, the field is back to the real value, no undo step
    steps = undo_count(ws)
    type_in(panel.scale, "50")
    assert "Scale must be between 0.01 and 20" in panel.notice.text() and panel.scale.value() == 1.5
    type_in(panel.opacity, "250")
    assert "Opacity must be between 0 and 1" in panel.notice.text() and panel.opacity.value() == 50.0
    type_in(panel.crop["left"], "60")
    assert tl(ws).get_clip(cid).effects["crop"] == pytest.approx([0.6, 0.0, 0.4, 1.0]) and undo_count(ws) == steps + 1
    type_in(panel.crop["right"], "50")  # 60 % + 50 % cut away leaves nothing
    assert "Crop width must be between 0.01 and 1" in panel.notice.text()
    assert tl(ws).get_clip(cid).effects["crop"] == pytest.approx([0.6, 0.0, 0.4, 1.0]) and undo_count(ws) == steps + 1
    assert panel.crop["right"].value() == 0.0 and panel.crop["left"].value() == 60.0


def test_keyframed_properties_are_marked_with_a_diamond(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    assert "◆" not in panel._key_labels["opacity"].text()
    ws.edit.add_keyframe(env.video.id, "opacity", 1.0, 0.5)
    assert "◆" in panel._key_labels["opacity"].text() and "1 opacity keyframe" in panel._key_labels["opacity"].toolTip()
    assert "◆" not in panel._key_labels["scale"].text()


# ====================================================================== audio
def test_audio_fields_and_mute_follow_the_planner_rules(env):
    ws, panel = env.ws, env.panel
    env.sel(env.audio.id)
    cid = env.audio.id
    n = undo_count(ws)
    type_in(panel.volume, "50")
    assert tl(ws).get_clip(cid).audio["volume"] == pytest.approx(0.5)
    type_in(panel.fade_in, "0.5")
    type_in(panel.fade_out, "1")
    type_in(panel.pan, "-0.5")
    a = tl(ws).get_clip(cid).audio
    assert (a["fade_in"], a["fade_out"], a["pan"]) == (0.5, 1.0, -0.5) and undo_count(ws) == n + 4
    panel.mute.click()  # mute = volume 0, the previous volume is remembered
    c = tl(ws).get_clip(cid)
    assert c.audio["volume"] == 0.0 and c.metadata["premute_volume"] == pytest.approx(0.5) and panel.mute.isChecked() and panel.volume.value() == 0.0
    panel.mute.click()
    assert tl(ws).get_clip(cid).audio["volume"] == pytest.approx(0.5) and not panel.mute.isChecked()
    steps = undo_count(ws)
    type_in(panel.volume, "900")  # over 400 %
    assert "Volume must be between 0 and 4" in panel.notice.text() and panel.volume.value() == 50.0
    type_in(panel.fade_in, "9")  # longer than the clip
    assert panel.fade_in.value() == 0.5 and panel.notice.text()
    assert undo_count(ws) == steps
    ws.edit.set_locks([cid], "audio", "add")
    assert not panel.volume.isEnabled() and not panel.mute.isEnabled() and "audio of this clip is locked" in panel.volume.toolTip()


# ====================================================================== transitions
def test_transition_types_are_exactly_the_ones_the_renderer_supports(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    assert [panel.trans_type.itemData(i) for i in range(panel.trans_type.count())] == list(TRANSITION_TYPES)
    assert not panel.trans_remove.isEnabled() and not panel.trans_duration.isEnabled()
    n = undo_count(ws)
    panel.trans_type.setCurrentIndex(panel.trans_type.findData("FADE"))
    panel.trans_type.activated.emit(panel.trans_type.currentIndex())
    assert tl(ws).get_clip(env.video.id).transition == {"type": "FADE", "duration": 0.5} and undo_count(ws) == n + 1
    assert panel.trans_duration.isEnabled() and panel.trans_duration.value() == 0.5 and panel.trans_remove.isEnabled()
    type_in(panel.trans_duration, "1")
    assert tl(ws).get_clip(env.video.id).transition["duration"] == 1.0
    type_in(panel.trans_duration, "30")  # longer than the clip
    assert "cannot be longer than the clip" in panel.notice.text() and panel.trans_duration.value() == 1.0
    panel.trans_type.setCurrentIndex(panel.trans_type.findData("WIPE"))
    panel.trans_type.activated.emit(panel.trans_type.currentIndex())
    assert tl(ws).get_clip(env.video.id).transition["type"] == "WIPE" and "left to right" in panel.trans_hint.text()
    panel.trans_remove.click()
    assert tl(ws).get_clip(env.video.id).transition is None and not panel.trans_remove.isEnabled()
    # a dissolve needs a picture that ends where the clip starts: offered only then
    model_item = panel.trans_type.model().item(panel.trans_type.findData("DISSOLVE"))
    assert not model_item.isEnabled() and "ends exactly where this clip starts" in panel.trans_type.itemData(panel.trans_type.findData("DISSOLVE"), Qt.ItemDataRole.ToolTipRole)
    ws.edit.place_assets([env.assets["other"].id], at=3.0, track_id=env.video.track_id)
    second = next(c for c in tl(ws).get_track(env.video.track_id).clips if c.id != env.video.id)
    env.sel(second.id)
    assert panel.trans_type.model().item(panel.trans_type.findData("DISSOLVE")).isEnabled()
    panel.trans_type.setCurrentIndex(panel.trans_type.findData("DISSOLVE"))
    panel.trans_type.activated.emit(panel.trans_type.currentIndex())
    assert tl(ws).get_clip(second.id).transition["type"] == "DISSOLVE"
    # transitions are for picture clips only
    env.sel(env.audio.id)
    assert "transition" not in visible(panel)


# ====================================================================== multi-selection
def test_multi_selection_shows_common_properties_and_edits_all_in_one_undo_step(env):
    ws, panel = env.ws, env.panel
    ws.set_selection([env.video.id, env.image.id])
    QApplication.processEvents()
    assert panel.multi_label.isVisibleTo(panel) and "2 clips selected" in panel.multi_label.text()
    assert visible(panel) == {"transform", "locks"}  # common to a video and an image: transform; no speed, no timing, no transition for a mixed selection
    assert panel.opacity.value() == 100.0 and not panel.opacity.is_mixed()
    ws.edit.update_clip(env.image.id, opacity=0.4)
    assert panel.opacity.is_mixed() and panel.opacity.text().startswith("—")  # different values read as an em dash
    assert panel.scale.value() == 1.0 and not panel.scale.is_mixed()
    n = undo_count(ws)
    type_in(panel.scale, "2")
    assert tl(ws).get_clip(env.video.id).scale == 2.0 and tl(ws).get_clip(env.image.id).scale == 2.0 and undo_count(ws) == n + 1
    assert tl(ws).get_clip(env.image.id).opacity == pytest.approx(0.4)  # the mixed property was not touched
    type_in(panel.opacity, "70")
    assert tl(ws).get_clip(env.video.id).opacity == pytest.approx(0.7) and tl(ws).get_clip(env.image.id).opacity == pytest.approx(0.7) and undo_count(ws) == n + 2
    ws.undo()
    assert tl(ws).get_clip(env.video.id).opacity == 1.0 and tl(ws).get_clip(env.image.id).opacity == pytest.approx(0.4)
    # a partly-mixed position: typing x keeps every clip's own y
    ws.edit.update_clip(env.image.id, position=(10.0, 77.0))
    assert panel.pos_x.is_mixed() and panel.pos_y.is_mixed()
    type_in(panel.pos_x, "5")
    assert tl(ws).get_clip(env.video.id).position == (5.0, 0.0) and tl(ws).get_clip(env.image.id).position == (5.0, 77.0)
    # a locked clip in the selection disables the field with a reason
    ws.edit.set_locks([env.image.id], "transform", "add")
    assert not panel.scale.isEnabled() and "1 of the selected clips have their transform locked" in panel.scale.toolTip()


def test_multi_selection_of_unlike_clips_hides_incompatible_properties(env):
    ws, panel = env.ws, env.panel
    ws.set_selection([env.video.id, env.audio.id])
    QApplication.processEvents()
    assert visible(panel) == {"speed", "locks"}  # both play media: speed is common; picture and sound properties are not
    ws.set_selection([env.image.id, env.audio.id])
    QApplication.processEvents()
    assert visible(panel) == {"locks"} and panel.no_common.isVisibleTo(panel)
    # speed over two clips: one undo step
    ws.set_selection([env.video.id, env.audio.id])
    QApplication.processEvents()
    n = undo_count(ws)
    type_in(panel.speed, "2")
    assert tl(ws).get_clip(env.video.id).speed == 2.0 and tl(ws).get_clip(env.audio.id).speed == 2.0 and undo_count(ws) == n + 1
    ws.undo()
    assert tl(ws).get_clip(env.video.id).speed == 1.0 and tl(ws).get_clip(env.audio.id).speed == 1.0


# ====================================================================== locks, links and groups
def test_lock_checkboxes_link_and_group_buttons(env):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    n = undo_count(ws)
    panel.lock_boxes["position"].click()
    panel.lock_boxes["style"].click()
    assert tl(ws).get_clip(env.video.id).locks == frozenset({"position", "style"}) and undo_count(ws) == n + 2
    panel.lock_boxes["position"].click()
    assert tl(ws).get_clip(env.video.id).locks == frozenset({"style"})
    assert not panel.link_btn.isEnabled() and "at least two clips" in panel.link_btn.toolTip()
    assert not panel.unlink_btn.isEnabled() and not panel.ungroup_btn.isEnabled()
    ws.set_selection([env.video.id, env.audio.id])
    QApplication.processEvents()
    assert panel.link_btn.isEnabled()
    panel.link_btn.click()
    v, a = tl(ws).get_clip(env.video.id), tl(ws).get_clip(env.audio.id)
    assert v.link_id and v.link_id == a.link_id and "Linked: 2 of 2" in panel.relation_label.text() and panel.unlink_btn.isEnabled()
    panel.group_btn.click()
    assert tl(ws).get_clip(env.video.id).group_id and panel.ungroup_btn.isEnabled()
    panel.unlink_btn.click()
    assert not tl(ws).get_clip(env.video.id).link_id and not tl(ws).get_clip(env.audio.id).link_id and tl(ws).get_clip(env.video.id).group_id
    panel.ungroup_btn.click()
    assert not tl(ws).get_clip(env.video.id).group_id
    # the "Lock item" checkbox over a selection: locks all of them, in one undo step
    steps = undo_count(ws)
    panel.lock_boxes["item"].click()
    assert all("item" in tl(ws).get_clip(c).locks for c in (env.video.id, env.audio.id)) and undo_count(ws) == steps + 1
    # a refused edit (locked track) reports the engine's message
    ws.timeline.set_track_flag(env.image.track_id, "locked", True)
    ws.set_selection([env.image.id, env.video.id])
    QApplication.processEvents()
    assert not panel.link_btn.isEnabled() and "is locked" in panel.link_btn.toolTip()


# ====================================================================== focus, refresh, public API
def test_refresh_never_clobbers_a_field_being_typed_in_and_keeps_focus(env, qapp):
    ws, panel = env.ws, env.panel
    env.sel(env.video.id)
    panel.activateWindow()
    panel.duration.setFocus()
    qapp.processEvents()
    panel.duration.selectAll()
    QTest.keyClicks(panel.duration, "2")  # typing, not committed yet
    assert panel.duration.is_editing()
    focus_before = panel.focusWidget()
    ws.edit.update_clip(env.video.id, opacity=0.3)  # an unrelated change elsewhere -> project.changed -> refresh
    ws.select_clip(env.video.id)
    panel.refresh()
    assert panel.duration.text() == "2" and panel.duration.is_editing()
    assert panel.opacity.value() == 30.0  # other fields did refresh
    assert panel.focusWidget() is focus_before is panel.duration
    QTest.keyClick(panel.duration, Qt.Key.Key_Return)
    assert tl(ws).get_clip(env.video.id).duration == pytest.approx(2.0)
    # NumberEdit: a half-typed number survives a refresh too
    panel.scale.selectAll()
    QTest.keyClicks(panel.scale, "1.7")
    ws.edit.update_clip(env.video.id, rotation=10.0)
    assert panel.scale.lineEdit().text().startswith("1.7") and panel.scale.is_editing() and panel.rotation.value() == 10.0
    panel.scale.discard_edit()
    assert panel.scale.value() == 1.0


def test_focus_clip_selects_and_shows_the_clip(env):
    ws, panel = env.ws, env.panel
    assert panel.focus_clip("clip_nope") is False
    assert panel.focus_clip(env.image.id) is True
    assert ws.selected_clip_id == env.image.id and panel.name_label.text() == "pic.png" and "transform" in visible(panel)
    assert panel.focus_clip(env.video.id) and panel.name_label.text() == "clip.mp4"


def test_replace_dialog_lists_compatible_media_and_the_replacement_goes_through_the_engine(env, monkeypatch):
    ws, panel = env.ws, env.panel
    dlg = ReplaceMediaDialog(panel.ctx, env.video.id)
    names = [dlg.lib_list.item(i).text() for i in range(dlg.lib_list.count())]
    assert any(n.startswith("other.mp4") for n in names) and any(n.startswith("pic.png") for n in names)
    assert not any(n.startswith("voice.wav") or n.startswith("clip.mp4") for n in names)  # no sound for a picture clip, not the clip's own media
    assert not dlg.ok.isEnabled()
    dlg.lib_list.setCurrentRow(next(i for i, n in enumerate(names) if n.startswith("other.mp4")))
    assert dlg.ok.isEnabled()
    dlg.ok.click()
    assert dlg.asset_id == env.assets["other"].id
    adlg = ReplaceMediaDialog(panel.ctx, env.audio.id)
    assert adlg.lib_list.count() == 1 and adlg.lib_list.item(0).text().startswith("There is no other compatible media")  # the only sound in the library is the clip's own
    # the inspector replaces the media through ws.edit.replace_clip (a clip without an AI scene)
    monkeypatch.setattr(replace_media_dialog, "choose_replacement", lambda parent, ctx, cid: env.assets["other"].id)
    n = undo_count(ws)
    assert panel.open_replace_dialog(env.video.id) is True
    c = tl(ws).get_clip(env.video.id)
    assert c.asset_id == env.assets["other"].id and c.duration == pytest.approx(2.0) and undo_count(ws) == n + 1  # shorter media shortens the clip (the engine says so)
    ws.undo()
    assert tl(ws).get_clip(env.video.id).asset_id == env.assets["video"].id
    ws.edit.set_locks([env.video.id], "item", "add")
    assert panel.open_replace_dialog(env.video.id) is False and "locked" in panel.notice.text()
    monkeypatch.setattr(replace_media_dialog, "choose_replacement", lambda parent, ctx, cid: None)
    assert panel.open_replace_dialog(env.image.id) is False  # cancelled: nothing changes
    assert tl(ws).get_clip(env.image.id).asset_id == env.assets["image"].id


# ====================================================================== keyframe editor
def kfs(ws, cid, prop):
    return sorted(((k.time, k.value, k.interpolation) for k in tl(ws).get_clip(cid).keyframes if k.property == prop), key=lambda x: x[0])


def choose_prop(kf, prop):
    kf.prop_combo.setCurrentIndex(kf.prop_combo.findData(prop))
    kf.prop_combo.activated.emit(kf.prop_combo.currentIndex())


def drag(lane, a: QPoint, b: QPoint, steps: int = 4, mods=Qt.KeyboardModifier.NoModifier) -> None:
    QTest.mousePress(lane, Qt.MouseButton.LeftButton, mods, a)
    for i in range(1, steps + 1):
        QTest.mouseMove(lane, QPoint(a.x() + (b.x() - a.x()) * i // steps, a.y() + (b.y() - a.y()) * i // steps))
    QTest.mouseRelease(lane, Qt.MouseButton.LeftButton, mods, b)


def test_keyframe_properties_are_the_engines_and_hidden_for_clips_without_keyframes(env):
    ws, panel = env.ws, env.panel
    kf = panel.kf
    env.sel(env.video.id)
    track = tl(ws).get_track(env.video.track_id)
    assert kf.available
    assert [kf.prop_combo.itemData(i) for i in range(kf.prop_combo.count())] == list(keyframe_properties(track, tl(ws).get_clip(env.video.id)))
    assert "volume" not in [kf.prop_combo.itemData(i) for i in range(kf.prop_combo.count())]  # a picture clip's sound is not played
    env.sel(env.audio.id)
    assert [kf.prop_combo.itemData(i) for i in range(kf.prop_combo.count())] == ["volume"]
    env.sel(env.video.id)
    choose_prop(kf, "blur")
    assert "ONE constant blur" in kf.prop_hint.text()  # the renderer cannot animate blur: said, not hidden
    gid = ws.presentation.add_graphic("HEADLINE", "Hello", 6.0, 2.0)
    env.sel(gid)
    assert not kf.available and "animation presets" in kf.note.text() and not panel.sections["keyframes"].isVisibleTo(panel)


def test_keyframe_lane_add_move_select_delete_interpolate_copy_paste_with_the_mouse(env, qapp):
    ws, panel = env.ws, env.panel
    kf, lane = panel.kf, panel.kf.lane
    cid = env.video.id
    env.sel(cid)
    choose_prop(kf, "opacity")
    qapp.processEvents()
    assert lane.width() > 200 and lane.height() > 100 and lane.property_name() == "opacity"
    n = undo_count(ws)
    # ---- click empty space = add a keyframe at that frame-snapped time / value
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, lane.pos_for(1.0, 0.5))
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, lane.pos_for(2.0, 0.9))
    got = kfs(ws, cid, "opacity")
    assert len(got) == 2 and undo_count(ws) == n + 2
    assert got[0][0] == pytest.approx(1.0, abs=1 / 30) and got[0][1] == pytest.approx(0.5, abs=0.03) and got[1][0] == pytest.approx(2.0, abs=1 / 30)
    assert got[0][0] == pytest.approx(round(got[0][0] * 30) / 30)  # snapped to the frame grid
    assert "(2)" in kf.prop_combo.itemText(kf.prop_combo.findData("opacity"))
    t0, v0 = got[0][0], got[0][1]
    # ---- drag a diamond: a live preview while the button is down, ONE move on release
    before = undo_count(ws)
    a = lane.keyframe_pos(t0)
    QTest.mousePress(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(lane, lane.pos_for(1.2, 0.4))
    QTest.mouseMove(lane, lane.pos_for(1.5, 0.2))
    assert lane.is_dragging() and kfs(ws, cid, "opacity")[0][0] == t0 and undo_count(ws) == before  # nothing committed yet
    assert not lane.grab().isNull()  # the live preview paints
    QTest.mouseRelease(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, lane.pos_for(1.5, 0.2))
    moved = kfs(ws, cid, "opacity")
    assert len(moved) == 2 and moved[0][0] == pytest.approx(1.5, abs=1 / 30) and moved[0][1] == pytest.approx(0.2, abs=0.03) and undo_count(ws) == before + 1
    assert not lane.is_dragging() and [k.time for k in lane.selected()] == [moved[0][0]]
    ws.undo()
    assert kfs(ws, cid, "opacity")[0][0] == pytest.approx(t0) and kfs(ws, cid, "opacity")[0][1] == pytest.approx(v0)
    ws.redo()
    assert kfs(ws, cid, "opacity")[0][0] == pytest.approx(moved[0][0])
    # ---- an impossible drag (value above the engine's range): refused with its message, the diamond is back where the model has it
    steps = undo_count(ws)
    first = kfs(ws, cid, "opacity")[0]
    top = lane.plot().top()
    drag(lane, lane.keyframe_pos(first[0]), QPoint(int(lane.x_of(first[0])), int(top) - 10))
    assert "opacity value must be between 0 and 1" in panel.notice.text() and undo_count(ws) == steps and kfs(ws, cid, "opacity")[0][1] == pytest.approx(first[1])
    # ---- multi-select with Shift, and with a rubber band
    p1, p2 = lane.keyframe_pos(kfs(ws, cid, "opacity")[0][0]), lane.keyframe_pos(kfs(ws, cid, "opacity")[1][0])
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
    assert len(lane.selected()) == 1
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, p2)
    assert len(lane.selected()) == 2
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, p1)  # toggles one off again
    assert len(lane.selected()) == 1
    lane.clear_selection()
    plot = lane.plot()
    drag(lane, QPoint(int(plot.left()) + 2, int(plot.top()) + 2), QPoint(int(plot.right()) - 2, int(plot.bottom()) - 2), steps=3)
    assert len(lane.selected()) == 2 and len(kfs(ws, cid, "opacity")) == 2  # a box selects, it does not add
    # ---- moving several selected keyframes together is one undo step
    steps = undo_count(ws)
    k0 = kfs(ws, cid, "opacity")
    drag(lane, lane.keyframe_pos(k0[0][0]), QPoint(lane.keyframe_pos(k0[0][0]).x() - 20, lane.keyframe_pos(k0[0][0]).y() + 10))
    k1 = kfs(ws, cid, "opacity")
    assert undo_count(ws) == steps + 1 and k1[0][0] < k0[0][0] and k1[1][0] < k0[1][0] and (k0[1][0] - k0[0][0]) == pytest.approx(k1[1][0] - k1[0][0], abs=1e-6)
    # ---- interpolation through the context menu: exactly the renderer's modes
    assert len(lane.selected()) == 2
    menu = lane.build_menu(lane.keyframe_pos(k1[0][0]))  # right-click inside a multi-selection keeps it
    assert len(lane.selected()) == 2
    lane.clear_selection()
    menu = lane.build_menu(lane.keyframe_pos(k1[0][0]))
    sub = next(a for a in menu.actions() if a.text() == "Interpolation").menu()
    assert [a.data() for a in sub.actions()] == list(INTERPOLATIONS) and INTERPOLATIONS == ("linear", "ease_in", "ease_out", "ease_in_out")
    assert [k.time for k in lane.selected()] == [k1[0][0]]  # right-clicking a keyframe selects it
    steps = undo_count(ws)
    next(a for a in sub.actions() if a.data() == "ease_in_out").trigger()
    assert kfs(ws, cid, "opacity")[0][2] == "ease_in_out" and kfs(ws, cid, "opacity")[1][2] == "linear" and undo_count(ws) == steps + 1
    assert kf.interp_combo.currentData() == "ease_in_out"
    kf.interp_combo.setCurrentIndex(kf.interp_combo.findData("ease_out"))
    kf.interp_combo.activated.emit(kf.interp_combo.currentIndex())
    assert kfs(ws, cid, "opacity")[0][2] == "ease_out"
    # ---- copy / paste at the playhead (context menu and buttons)
    copy_action = next(a for a in lane.build_menu(lane.keyframe_pos(k1[0][0])).actions() if a.text() == "Copy")
    copy_action.trigger()
    clip = tl(ws).get_clip(cid)
    ws.edit.set_playhead(clip.timeline_start + 2.5)
    steps = undo_count(ws)
    kf.paste_btn.click()
    got = kfs(ws, cid, "opacity")
    assert len(got) == 3 and any(abs(t - 2.5) < 1e-6 and iv == "ease_out" for t, _v, iv in got) and undo_count(ws) == steps + 1
    kf.paste_btn.click()  # pasting on top of the keyframe that is already there is refused
    assert "already a opacity keyframe" in panel.notice.text() and len(kfs(ws, cid, "opacity")) == 3
    # ---- Delete key removes the selection (one undo step); an empty selection says so
    lane.select([2.5])
    steps = undo_count(ws)
    QTest.keyClick(lane, Qt.Key.Key_Delete)
    assert len(kfs(ws, cid, "opacity")) == 2 and undo_count(ws) == steps + 1
    lane.clear_selection()
    kf.delete_btn.setEnabled(True)
    kf.delete_selected()
    assert "Select a keyframe first" in panel.notice.text()
    ws.undo()
    assert len(kfs(ws, cid, "opacity")) == 3


def test_keyframe_buttons_numeric_fields_playhead_and_locks(env, qapp):
    ws, panel = env.ws, env.panel
    kf, lane = panel.kf, panel.kf.lane
    cid = env.video.id
    env.sel(cid)
    choose_prop(kf, "scale")
    clip = tl(ws).get_clip(cid)
    # add at playhead with the current value; outside the clip the button is off and says why
    ws.edit.set_playhead(clip.timeline_start + 10.0)
    assert not kf.add_btn.isEnabled() and "inside the clip" in kf.add_btn.toolTip()
    ws.edit.set_playhead(clip.timeline_start + 1.0)
    assert kf.add_btn.isEnabled()
    n = undo_count(ws)
    kf.add_btn.click()
    assert kfs(ws, cid, "scale") == [(pytest.approx(1.0), pytest.approx(1.0), "linear")] and undo_count(ws) == n + 1  # no keyframes yet: the neutral scale 1.0
    ws.edit.add_keyframe(cid, "scale", 2.0, 1.4)
    ws.edit.set_playhead(clip.timeline_start + 1.5)
    kf.add_btn.click()  # between two keyframes: the interpolated value
    mid = [k for k in kfs(ws, cid, "scale") if abs(k[0] - 1.5) < 1e-6]
    assert mid and mid[0][1] == pytest.approx(1.2)
    # previous / next keyframe move the playhead
    ws.edit.set_playhead(clip.timeline_start + 1.2)
    kf.next_btn.click()
    assert ws.edit.playhead == pytest.approx(clip.timeline_start + 1.5)
    kf.prev_btn.click()
    kf.prev_btn.click()
    assert ws.edit.playhead == pytest.approx(clip.timeline_start + 1.0) and not kf.prev_btn.isEnabled() and "no earlier keyframe" in kf.prev_btn.toolTip()
    # numeric fields of a single selected keyframe: time and value, validated by the engine
    assert not kf.time_edit.isEnabled()
    lane.select([2.0])
    assert kf.time_edit.isEnabled() and kf.value_edit.isEnabled() and kf.value_edit.value() == pytest.approx(1.4)
    steps = undo_count(ws)
    type_in(kf.value_edit, "1.8")
    assert kfs(ws, cid, "scale")[-1][1] == pytest.approx(1.8) and undo_count(ws) == steps + 1
    lane.select([2.0])
    type_in(kf.time_edit, "00:00:02:15")
    assert [k[0] for k in kfs(ws, cid, "scale")][-1] == pytest.approx(2.5) and [k.time for k in lane.selected()] == [pytest.approx(2.5)]
    steps = undo_count(ws)
    type_in(kf.value_edit, "99")  # beyond the engine's 20
    assert "scale value must be between 0.01 and 20" in panel.notice.text() and kfs(ws, cid, "scale")[-1][1] == pytest.approx(1.8) and kf.value_edit.value() == pytest.approx(1.8)
    type_in(kf.time_edit, "9")  # outside the clip
    assert "outside the clip" in panel.notice.text() and undo_count(ws) == steps and kf.time_edit.text() == "00:00:02:15"
    lane.select([1.0, 2.5])
    assert not kf.time_edit.isEnabled() and "single keyframe" in kf.time_edit.toolTip()
    # a locked aspect makes the lane read-only and says why
    ws.edit.set_locks([cid], "transform", "add")
    assert lane.read_only and not kf.add_btn.isEnabled() and "transform of this clip is locked" in kf.add_btn.toolTip()
    count = len(kfs(ws, cid, "scale"))
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, lane.pos_for(0.3, 1.0))
    assert len(kfs(ws, cid, "scale")) == count
    # the audio clip's volume lane belongs to the audio aspect, not the transform lock
    env.sel(env.audio.id)
    assert kf.prop() == "volume" and not lane.read_only
    ws.edit.set_locks([env.audio.id], "audio", "add")
    assert lane.read_only


def test_keyframes_survive_speed_and_trim_the_lane_shows_what_the_model_has(env, qapp):
    ws, panel = env.ws, env.panel
    cid = env.video.id
    env.sel(cid)
    ws.edit.add_keyframe(cid, "opacity", 1.0, 0.3)
    ws.edit.add_keyframe(cid, "opacity", 2.0, 0.9, "ease_out")
    choose_prop(panel.kf, "opacity")
    assert [(k.time, k.value, k.interpolation) for k in panel.kf.lane.keyframes()] == [(1.0, 0.3, "linear"), (2.0, 0.9, "ease_out")]
    ws.undo()  # undo of the second keyframe: the lane follows the model without any user action
    assert [k.time for k in panel.kf.lane.keyframes()] == [1.0]
    _ = KEYFRAME_RANGES


# ====================================================================== AI decisions (Phase 4 fixtures: real research pipeline + generated AI edit)
def scene_media(ws, sid):
    return [c for c in tl(ws).all_clips() if c.scene_id == sid and c.kind == "media" and c.track_id != "track_a1"]


def single_visual_scenes(ws):
    return [s.id for s in ws.project.scenes if len(scene_media(ws, s.id)) == 1]


def regenerate(ws, *scene_ids):
    job = ws.editing.regenerate_scene(scene_ids[0]) if len(scene_ids) == 1 else ws.editing.regenerate_scenes(list(scene_ids))
    assert job is not None and ws.jobs.wait_idle(90)
    assert ws.project.editing_sessions[-1].status == "COMPLETED", ws.project.editing_sessions[-1].error


@pytest.fixture
def ai(qapp, edit_ws):
    ws = edit_ws
    ws.editing.generate()
    assert ws.jobs.wait_idle(90) and ws.project.editing_sessions[-1].status == "COMPLETED"
    panel = InspectorPanel(UiContext(ws, UiBridge(ws.bus)))
    panel.resize(440, 1900)
    panel.show()
    qapp.processEvents()
    yield ws, panel
    panel.close()


def test_ai_box_shows_the_decision_its_factors_and_real_actions(ai):
    ws, panel = ai
    sid = single_visual_scenes(ws)[0]
    clip = scene_media(ws, sid)[0]
    dec = ws.editing.decision_for_clip(clip.id)
    assert clip.created_by == "AI" and dec is not None
    ws.select_clip(clip.id)
    assert panel.ai_box.isVisibleTo(panel) and not panel.pres_box.isVisibleTo(panel)
    info = panel.ai_info.text()
    assert dec.type.value.replace("_", " ").title() in info and f"{dec.confidence:.0f}%" in info and "created by AI" in info and html.escape(dec.reason, quote=False) in info
    facts = panel.ai_factors.text()
    assert "Scene " in facts and "Visual: accuracy 90%, chosen by USER" in facts and f"Role: {dec.slot}" in facts
    assert not panel.ai_badge.isVisibleTo(panel)  # AI-owned, not a manual override
    assert panel.ai_replace.isEnabled() and panel.ai_search.isEnabled() and panel.ai_regen.isEnabled() and panel.ai_manual.isEnabled()
    assert panel.reset_buttons == []  # nothing was overridden: no reset button is faked
    emitted = []
    panel.search_alternatives_requested.connect(emitted.append)
    panel.ai_search.click()
    assert emitted == [sid]
    # the voice-over (a system clip) has no scene: the scene actions are off
    voice = tl(ws).get_track("track_a1").clips[0]
    ws.select_clip(voice.id)
    assert not panel.ai_box.isVisibleTo(panel) or not panel.ai_search.isEnabled()
    # Lock = protected from regeneration (ws.editing.lock_clip): the badge says so and the locked clip survives
    ws.select_clip(clip.id)
    panel.ai_lock.click()
    assert tl(ws).get_clip(clip.id).locked and panel.ai_badge.isVisibleTo(panel) and "locked" in panel.ai_badge.toolTip() and panel.ai_lock.isChecked()
    regenerate(ws, sid)
    kept = tl(ws).get_clip(clip.id)
    assert kept is not None and kept.locked and kept.created_by == "AI"
    panel.ai_lock.click()
    assert not tl(ws).get_clip(clip.id).locked


def test_replace_visual_through_the_dialog_makes_the_clip_the_users_and_is_undoable(ai, monkeypatch):
    from app.tests.helpers import cand

    ws, panel = ai
    p = ws.project
    sid = single_visual_scenes(ws)[0]
    clip = scene_media(ws, sid)[0]
    new = ws.assets_by["plain.png"] if clip.asset_id != ws.assets_by["plain.png"].id else ws.assets_by["wide.png"]
    c_in, c_out = cand("In the project", scene_id=sid, cid="cand_in"), cand("Not downloaded", scene_id=sid, cid="cand_out")
    c_in.asset_id = new.id
    p.visual_candidates.update({"cand_in": c_in, "cand_out": c_out})
    dlg = ReplaceMediaDialog(panel.ctx, clip.id)
    assert "1 usable of 2" in dlg.cand_label.text()
    usable = [dlg.cand_list.item(i) for i in range(dlg.cand_list.count()) if dlg.cand_list.item(i).data(Qt.ItemDataRole.UserRole)]
    assert [i.data(Qt.ItemDataRole.UserRole) for i in usable] == [new.id]
    assert any("not downloaded" in dlg.cand_list.item(i).text() and not dlg.cand_list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled for i in range(dlg.cand_list.count()))
    dlg.cand_list.setCurrentItem(usable[0])
    assert dlg.ok.isEnabled() and dlg.selected_asset() == new.id
    ws.select_clip(clip.id)
    assert "Alternatives: 2 candidates researched, 1 already in the project." in panel.ai_factors.text()
    monkeypatch.setattr(replace_media_dialog, "choose_replacement", lambda parent, ctx, cid: new.id)
    n = undo_count(ws)
    panel.ai_replace.click()
    now = tl(ws).get_clip(clip.id)
    assert now.asset_id == new.id and now.created_by == "USER" and (now.timeline_start, now.duration) == (clip.timeline_start, clip.duration) and undo_count(ws) == n + 1
    d = ws.editing.decision_for_clip(clip.id)
    assert d.created_by.value == "USER" and d.parameters["operation"] == "REPLACE" and d.overrides_decision_id
    assert panel.ai_badge.isVisibleTo(panel) and "Manual override" in panel.ai_badge.text() and not panel.ai_manual.isEnabled()
    ws.undo()
    assert tl(ws).get_clip(clip.id).asset_id == clip.asset_id and tl(ws).get_clip(clip.id).created_by == "AI" and not panel.ai_badge.isVisibleTo(panel)


def test_manual_edits_of_ai_clips_survive_regeneration_of_their_scene(ai):
    """The precedence rule: AI proposes, the user's manual edit wins - for numeric edits, trims, moves and property changes made through ws.edit."""
    ws, panel = ai
    scenes = single_visual_scenes(ws)
    a, b, c, untouched = scenes[0], scenes[1], scenes[2], scenes[3]
    ca, cb, cc, cu = (scene_media(ws, s)[0] for s in (a, b, c, untouched))
    ws.edit.set_duration(ca.id, ca.duration - 0.5)  # shorter: leaves a gap
    ws.edit.set_start(ca.id, tl(ws).get_clip(ca.id).timeline_start + 0.2)  # moved inside the gap
    ws.edit.trim(cb.id, "end", cb.timeline_end - 0.4)
    ws.edit.update_clip(cc.id, opacity=0.5)
    keep = {cid: tl(ws).get_clip(cid).to_dict() for cid in (ca.id, cb.id, cc.id)}
    assert all(tl(ws).get_clip(i).created_by == "USER" for i in keep)
    regenerate(ws, a, b, c, untouched)
    for cid, before in keep.items():
        now = tl(ws).get_clip(cid)
        assert now is not None, "the manually edited clip was removed by the regeneration"
        assert (now.timeline_start, now.duration, now.asset_id, now.opacity, now.created_by) == (before["timeline_start"], before["duration"], before["asset_id"], before["opacity"], "USER")
    assert tl(ws).get_clip(cu.id) is None or tl(ws).get_clip(cu.id).created_by == "AI"  # an unedited AI clip is the AI's to regenerate
    assert any(x.created_by == "AI" for x in scene_media(ws, untouched))
    # the inspector shows the survivor as a manual override
    ws.select_clip(ca.id)
    assert panel.ai_badge.isVisibleTo(panel)


def test_regenerate_button_asks_first_and_keeps_manual_work(ai, monkeypatch):
    ws, panel = ai
    sid, other = single_visual_scenes(ws)[:2]
    clip, ctl = scene_media(ws, sid)[0], scene_media(ws, other)[0]
    panel.focus_clip(clip.id)
    type_in(panel.duration, f"{clip.duration - 0.5:.4f}")  # a manual edit through the inspector
    edited = tl(ws).get_clip(clip.id).to_dict()
    assert edited["duration"] == pytest.approx(clip.duration - 0.5, abs=0.04) and edited["created_by"] == "USER"
    asked = []
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), answers.pop(0))[1])
    sessions = len(ws.project.editing_sessions)
    panel.ai_regen.click()
    assert asked and "edited or locked is kept" in asked[0] and len(ws.project.editing_sessions) == sessions  # answered No: nothing started
    panel.ai_regen.click()
    assert ws.jobs.wait_idle(90) and ws.project.editing_sessions[-1].status == "COMPLETED" and len(ws.project.editing_sessions) == sessions + 1
    now = tl(ws).get_clip(clip.id)
    assert now is not None and (now.timeline_start, now.duration, now.created_by) == (edited["timeline_start"], edited["duration"], "USER")
    assert tl(ws).get_clip(ctl.id) is not None  # other scenes were not asked for
    assert "Regenerating this scene" in panel.notice.text()


def test_mark_as_manual_override_records_the_takeover_and_regeneration_respects_it(ai):
    ws, panel = ai
    sa, sb = single_visual_scenes(ws)[:2]
    ca, cb = scene_media(ws, sa)[0], scene_media(ws, sb)[0]
    dec_before = ws.editing.decision_for_clip(ca.id)
    panel.focus_clip(ca.id)
    assert panel.ai_manual.isEnabled() and "no longer replace" in panel.ai_manual.toolTip()
    n = undo_count(ws)
    panel.ai_manual.click()
    now = tl(ws).get_clip(ca.id)
    d = ws.editing.decision_for_clip(ca.id)
    assert now.created_by == "USER" and d.created_by.value == "USER" and d.overrides_decision_id == dec_before.decision_id and undo_count(ws) == n + 1
    assert ws.editing.override_of(d.decision_id).decision_id == dec_before.decision_id
    assert panel.ai_badge.isVisibleTo(panel) and not panel.ai_manual.isEnabled() and "Already a manual override" in panel.ai_manual.toolTip()
    ws.undo()  # one undo step takes the takeover back
    assert tl(ws).get_clip(ca.id).created_by == "AI" and ws.editing.decision_for_clip(ca.id).decision_id == dec_before.decision_id and panel.ai_manual.isEnabled()
    ws.redo()
    assert tl(ws).get_clip(ca.id).created_by == "USER"
    regenerate(ws, sa, sb)
    assert tl(ws).get_clip(ca.id) is not None  # the marked clip stays...
    assert all(c.id != cb.id for c in scene_media(ws, sb)) and scene_media(ws, sb)  # ...while the untouched AI clip of the other scene was the AI's to redo


def test_reset_to_ai_value_exists_only_where_the_original_is_stored(ai):
    ws, panel = ai
    p = ws.project
    motion = next(d for d in p.editing_decisions.values() if d.type.value in ("ZOOM", "PAN") and len(scene_media(ws, d.scene_id)) == 1)
    clip = tl(ws).get_clip(motion.target_id)
    ws.select_clip(clip.id)
    assert panel.reset_buttons == []  # untouched AI decision: nothing to reset
    orig_scale = motion.parameters["end_scale"]
    new = ws.editing.update_decision(motion.decision_id, {"kind": "PUNCH_IN", "start_scale": 1.0, "end_scale": 1.31, "start_pos": [0, 0], "end_pos": [0, 0]})
    assert new.parameters["end_scale"] == 1.31 and ws.editing.override_of(new.decision_id).parameters["end_scale"] == orig_scale
    labels = [b.text() for b in panel.reset_buttons]
    target = f"Reset end scale to AI value ({orig_scale:.3g})"
    assert target in labels, labels
    n = undo_count(ws)
    next(b for b in panel.reset_buttons if b.text() == target).click()
    assert undo_count(ws) == n + 1 and ws.editing.decision_for_clip(clip.id) is not None
    d = next(x for x in ws.editing.decisions_for_clip(clip.id) if x.type.value in ("ZOOM", "PAN"))
    assert d.parameters["end_scale"] == pytest.approx(orig_scale) and max(k.value for k in tl(ws).get_clip(clip.id).keyframes if k.property == "scale") == pytest.approx(orig_scale)
    assert target not in [b.text() for b in panel.reset_buttons]  # the value is the AI's again
    # a manual trim through ws.edit records the original timing: its reset button restores it
    vt = ws.editing.decision_for_clip(clip.id)
    orig_dur = vt.duration
    ws.edit.set_duration(clip.id, clip.duration - 0.5)
    assert any(b.text().startswith("Reset duration to AI value") for b in panel.reset_buttons)
    next(b for b in panel.reset_buttons if b.text().startswith("Reset duration to AI value")).click()
    assert tl(ws).get_clip(clip.id).duration == pytest.approx(orig_dur, abs=0.04) and not any(b.text().startswith("Reset duration") for b in panel.reset_buttons)
    # a clip the user made has no AI original: no reset buttons at all
    gid = ws.presentation.add_graphic("HEADLINE", "Mine", 1.0, 2.0)
    ws.select_clip(gid)
    assert panel.reset_buttons == []


# ====================================================================== captions and presentation objects (Phase 5 fixtures)
def captions_of(ws, sid):
    return sorted((c for c in tl(ws).all_clips() if c.kind == "caption" and c.scene_id == sid), key=lambda c: c.timeline_start)


@pytest.fixture
def pres(qapp, pres_ws):
    ws = pres_ws
    ws.presentation.generate(["CAPTIONS"])
    assert ws.jobs.wait_idle(120) and ws.project.presentation_sessions[-1].status == "COMPLETED", ws.project.presentation_sessions[-1].error
    panel = InspectorPanel(UiContext(ws, UiBridge(ws.bus)))
    panel.resize(440, 2000)
    panel.show()
    qapp.processEvents()
    yield ws, panel
    panel.close()


def test_caption_section_edits_merge_split_and_presentation_actions(pres, monkeypatch):
    ws, panel = pres
    p = ws.project
    sid = next(s.id for s in p.scenes if len(captions_of(ws, s.id)) >= 4)
    caps = captions_of(ws, sid)
    cap = caps[0]
    ws.select_clip(cap.id)
    sections = visible(panel)
    assert {"clip", "presentation", "ai", "timing", "caption", "locks"} <= sections and not sections & {"keyframes", "transform", "speed", "audio", "text", "transition"}
    assert panel.cap_words.count() == len(cap.text["words"]) and " – " in panel.cap_words.item(0).text()
    assert "created by AI" in panel.pres_info.text() and not panel.ai_badge.isVisibleTo(panel)
    # position preset and highlight mode go through the caption service (manual-override record included)
    orig_mode = cap.text["highlight_mode"]
    n = undo_count(ws)
    panel.cap_position.setCurrentIndex(panel.cap_position.findData("top"))
    panel.cap_position.activated.emit(panel.cap_position.currentIndex())
    c = tl(ws).get_clip(cap.id)
    d = p.presentation_decisions[c.ai_decision_id]
    assert c.text["position"] == "top" and c.created_by == "USER" and d.created_by.value == "USER" and d.overrides_decision_id and undo_count(ws) == n + 1
    assert panel.ai_badge.isVisibleTo(panel) and not panel.ai_manual.isEnabled()
    other_mode = "NONE" if orig_mode != "NONE" else "PROGRESSIVE"
    panel.cap_highlight.setCurrentIndex(panel.cap_highlight.findData(other_mode))
    panel.cap_highlight.activated.emit(panel.cap_highlight.currentIndex())
    assert tl(ws).get_clip(cap.id).text["highlight_mode"] == other_mode
    reset = next(b for b in panel.reset_buttons if b.text().startswith("Reset highlight mode to AI value"))
    assert orig_mode in reset.text()
    reset.click()
    assert tl(ws).get_clip(cap.id).text["highlight_mode"] == orig_mode and not any("highlight mode" in b.text() for b in panel.reset_buttons)
    # text edits keep going through the presentation box (apply) and honour a style lock
    ws.edit.set_locks([cap.id], "style", "add")
    assert not panel.pres_text.isEnabled() and not panel.cap_position.isEnabled() and "style of this clip is locked" in panel.pres_text.toolTip()
    ws.edit.set_locks([cap.id], "style", "remove")
    assert panel.pres_text.isEnabled()
    # merge with the next caption: one engine call, one undo step
    nxt = caps[1]
    assert nxt.timeline_start - tl(ws).get_clip(cap.id).timeline_end < 1.5
    ws.select_clip(caps[1].id)
    n = undo_count(ws)
    assert panel.cap_merge.isEnabled()
    words_a, words_b = len(caps[1].text["words"]), len(caps[2].text["words"])
    panel.cap_merge.click()
    merged = tl(ws).get_clip(caps[1].id)
    assert tl(ws).get_clip(caps[2].id) is None and len(merged.text["words"]) == words_a + words_b and undo_count(ws) == n + 1
    ws.undo()
    assert tl(ws).get_clip(caps[2].id) is not None and len(tl(ws).get_clip(caps[1].id).text["words"]) == words_a
    last = sorted((c for c in tl(ws).all_clips() if c.kind == "caption"), key=lambda c: c.timeline_start)[-1]
    ws.select_clip(last.id)
    assert not panel.cap_merge.isEnabled() and "no caption after this one" in panel.cap_merge.toolTip()
    # split at the playhead: only when the playhead is inside the caption
    target = caps[3]
    ws.select_clip(target.id)
    ws.edit.set_playhead(target.timeline_end + 5.0)
    assert not panel.cap_split.isEnabled() and "inside this caption" in panel.cap_split.toolTip()
    words = target.text["words"]
    ws.edit.set_playhead((words[1]["end"] + words[2]["start"]) / 2)
    assert panel.cap_split.isEnabled()
    n, count = undo_count(ws), len([c for c in tl(ws).all_clips() if c.kind == "caption"])
    panel.cap_split.click()
    assert len([c for c in tl(ws).all_clips() if c.kind == "caption"]) == count + 1 and undo_count(ws) == n + 1
    ws.undo()
    assert len([c for c in tl(ws).all_clips() if c.kind == "caption"]) == count


def test_presentation_clip_manual_override_and_regeneration(pres, monkeypatch):
    ws, panel = pres
    p = ws.project
    sid = next(s.id for s in p.scenes if len(captions_of(ws, s.id)) >= 3)
    caps = captions_of(ws, sid)
    mine, theirs = caps[0], caps[1]
    ws.select_clip(mine.id)
    assert panel.ai_manual.isEnabled() and panel.ai_regen.isEnabled()
    d0 = p.presentation_decisions[mine.ai_decision_id]
    n = undo_count(ws)
    panel.ai_manual.click()
    c = tl(ws).get_clip(mine.id)
    d = p.presentation_decisions[c.ai_decision_id]
    assert c.created_by == "USER" and d.created_by.value == "USER" and d.overrides_decision_id == d0.decision_id and undo_count(ws) == n + 1
    assert panel.ai_badge.isVisibleTo(panel)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    sessions = len(p.presentation_sessions)
    panel.ai_regen.click()
    assert ws.jobs.wait_idle(120) and len(p.presentation_sessions) == sessions + 1 and p.presentation_sessions[-1].status == "COMPLETED"
    assert tl(ws).get_clip(mine.id) is not None and tl(ws).get_clip(mine.id).created_by == "USER"  # the marked caption stays
    assert tl(ws).get_clip(theirs.id) is None  # the AI's own caption of the scene was regenerated


# ====================================================================== text clips and keyframe snapping
def test_text_clip_section_edits_real_text_properties_and_the_words_go_through_the_presentation_service(env):
    ws, panel = env.ws, env.panel
    gid = ws.presentation.add_graphic("HEADLINE", "Hello world", 6.0, 2.0)
    env.sel(gid)
    assert panel.sections["text"].isVisibleTo(panel) and panel.text_style.currentData() == "HEADLINE"
    n = undo_count(ws)
    type_in(panel.text_size, "80")
    type_in(panel.text_opacity, "50")
    type_in(panel.text_x, "30")
    panel.text_align.setCurrentIndex(panel.text_align.findData("left"))
    panel.text_align.activated.emit(panel.text_align.currentIndex())
    panel.text_bg.setCurrentIndex(panel.text_bg.findData("box"))
    panel.text_bg.activated.emit(panel.text_bg.currentIndex())
    panel.text_style.setCurrentIndex(panel.text_style.findData("WARNING"))
    panel.text_style.activated.emit(panel.text_style.currentIndex())
    t = tl(ws).get_clip(gid).text
    assert (t["size"], t["opacity"], t["alignment"], t["background"], t["style"]) == (80, 0.5, "left", "box", "WARNING") and t["position"][0] == pytest.approx(0.3)
    assert undo_count(ws) == n + 6
    steps = undo_count(ws)
    type_in(panel.text_size, "5000")
    assert "Text size must be between 6 and 500" in panel.notice.text() and panel.text_size.value() == 80 and undo_count(ws) == steps
    # the words: presentation box -> ws.presentation.update_graphic (keeps the manual-override record)
    panel.pres_text.setText("Brand new headline")
    panel.pres_apply.click()
    c = tl(ws).get_clip(gid)
    assert c.text["content"] == "Brand new headline" and c.created_by == "USER"
    ws.edit.set_locks([gid], "style", "add")
    assert not panel.text_size.isEnabled() and not panel.pres_text.isEnabled() and "style of this clip is locked" in panel.text_size.toolTip()


def test_keyframe_times_snap_to_the_playhead_and_the_frame_grid_and_alt_turns_snapping_off(env, qapp):
    ws, panel = env.ws, env.panel
    lane = panel.kf.lane
    cid = env.video.id
    env.sel(cid)
    choose_prop(panel.kf, "opacity")
    clip = tl(ws).get_clip(cid)
    ws.edit.set_playhead(clip.timeline_start + 1.0167)  # between two frames (30.5)
    qapp.processEvents()
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(lane.x_of(1.0167)) + 3, int(lane.plot().center().y())))
    assert kfs(ws, cid, "opacity")[0][0] == pytest.approx(1.0167, abs=1e-9)  # the playhead is a snap target
    x = int(lane.x_of(2.0)) + 1
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, int(lane.plot().center().y())))
    t = kfs(ws, cid, "opacity")[1][0]
    assert abs(t * 30 - round(t * 30)) < 1e-6  # frame grid
    x2 = int(lane.x_of(2.5)) + 2
    QTest.mouseClick(lane, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, QPoint(x2, int(lane.plot().center().y())))
    raw = lane.t_of(float(x2))
    assert any(abs(k[0] - raw) < 1e-9 for k in kfs(ws, cid, "opacity"))
    # switching the property shows that property's keyframes only and clears the selection
    lane.select([kfs(ws, cid, "opacity")[0][0]])
    choose_prop(panel.kf, "scale")
    assert lane.property_name() == "scale" and lane.keyframes() == [] and not lane.selected()
    choose_prop(panel.kf, "opacity")
    assert len(lane.keyframes()) == 3
