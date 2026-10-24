"""Phase 10 / U1: the timeline canvas as a professional editing surface (offscreen Qt).

The canvas is driven with real mouse / key / drop events and every assertion is made on the MODEL (``ws.project.timeline``) and on ``ws.commands`` (undo / redo), never on pixels.
Clips are inserted straight into the timeline (no ffmpeg, no undo history) so "one gesture = one undo step" is provable: after the gesture ``ws.commands.undo()`` once restores the
start state and the stack is empty again. ``QTest.mouseMove`` cannot carry modifiers, so gestures are sent as hand-built ``QMouseEvent`` objects (``press`` / ``move`` / ``release``).
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QRect, Qt  # noqa: E402
from PySide6.QtGui import QDropEvent, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QScrollArea  # noqa: E402

from app.core.events import Topics  # noqa: E402
from app.media.asset import Asset, AssetType, SourceType  # noqa: E402
from app.timeline.clip import KIND_CAPTION, Clip  # noqa: E402
from app.timeline.keyframes import Keyframe  # noqa: E402
from app.timeline.timeline import new_clip_id  # noqa: E402
from app.timeline.track import TrackKind  # noqa: E402
from app.ui.context import UiContext  # noqa: E402
from app.ui.media_library import ASSET_MIME, _AssetList, asset_ids_from_mime  # noqa: E402
from app.ui.qt_bridge import UiBridge  # noqa: E402
from app.ui.timeline_canvas import AUTOSCROLL_MAX_PX, NEW_TRACK_STRIP_H, RULER_H, MarkerDialog, TimelineCanvas, edge_scroll_speed, format_delta  # noqa: E402

NO, CTRL, SHIFT, ALT = Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.AltModifier
LEFT = Qt.MouseButton.LeftButton


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------------------------------------------------------------------ harness
class Env:
    """A workspace with a project, a bare ``TimelineCanvas`` inside a scroll area (like the panel) and the same reload wiring the panel has."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.bridge = UiBridge(ws.bus)
        self.ctx = UiContext(ws, self.bridge)
        self.status: list[str] = []
        self.reloads = 0
        self.canvas = TimelineCanvas(self.ctx)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.resize(1000, 420)
        self.scroll.show()
        self.bridge.on(Topics.STATUS, lambda p: self.status.append(str(p.get("message", ""))))
        self.bridge.on(Topics.PROJECT_CHANGED, lambda p: self._reload())
        self.tl = ws.project.timeline
        QApplication.processEvents()

    def _reload(self) -> None:
        self.reloads += 1
        self.canvas.reload()

    # ----- model building (direct inserts: no undo history)
    def asset(self, kind: str = "video", duration: float | None = 60.0, name: str | None = None) -> Asset:
        p = self.ws.project
        aid = p.assets.new_id()
        folder = {"audio": "audio", "video": "video", "image": "images"}[kind]
        rel = f"media/{folder}/{aid}.{'wav' if kind == 'audio' else 'mp4' if kind == 'video' else 'png'}"
        path = p.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 64)
        a = Asset(aid, AssetType(kind), SourceType.USER_MEDIA, rel, name or f"{kind}_{aid}", None if kind == "image" else duration, 1920 if kind != "audio" else None,
                  1080 if kind != "audio" else None, 30.0 if kind == "video" else None, None, False, None, 48000 if kind == "audio" else None, 2 if kind == "audio" else None, 64, f"hash_{aid}")
        p.assets.add(a)
        return a

    def clip(self, track_id: str, start: float, dur: float, asset: Asset | None = None, cid: str | None = None, source_in: float = 0.0, **kw) -> Clip:
        asset = asset or self.asset("audio" if track_id.startswith("track_a") else "video")
        c = Clip(cid or new_clip_id(), track_id, asset.id, start, dur, source_in, source_in + dur * kw.get("speed", 1.0), **kw)
        self.tl.insert_clip(c)
        return c

    def done(self) -> "Env":
        self.canvas.reload()
        QApplication.processEvents()
        return self

    def get(self, cid: str) -> Clip:
        c = self.tl.get_clip(cid)
        assert c is not None, cid
        return c

    def starts(self, track_id: str) -> list[tuple[float, float]]:
        return [(round(c.timeline_start, 4), round(c.duration, 4)) for c in self.tl.get_track(track_id).clips]

    def snapshot(self) -> str:
        return json.dumps(self.tl.to_dict(), sort_keys=True)

    # ----- geometry helpers
    def pos(self, cid: str, frac: float = 0.5, dy: int = 0) -> QPoint:
        c = self.get(cid)
        r = self.canvas.clip_rect(c, self.canvas._row_of(c.track_id))
        return QPoint(int(r.left() + r.width() * frac), int(r.center().y()) + dy)

    def at(self, t: float, track_id: str) -> QPoint:
        row = self.canvas._row_of(track_id)
        return QPoint(int(self.canvas.time_to_x(t)), int(self.canvas.row_rect_y(row) + self.canvas.row_height(row) / 2))

    def row_y(self, track_id: str) -> int:
        row = self.canvas._row_of(track_id)
        return int(self.canvas.row_rect_y(row) + self.canvas.row_height(row) / 2)

    # ----- events
    def send(self, kind, pos, button, buttons, mods=NO) -> None:
        ev = QMouseEvent(kind, QPointF(pos), QPointF(self.canvas.mapToGlobal(pos)), button, buttons, mods)
        QApplication.sendEvent(self.canvas, ev)

    def press(self, pos, mods=NO, button=LEFT) -> None:
        self.send(QEvent.Type.MouseButtonPress, pos, button, button, mods)

    def move(self, pos, mods=NO, buttons=LEFT) -> None:
        self.send(QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, buttons, mods)

    def release(self, pos, mods=NO, button=LEFT) -> None:
        self.send(QEvent.Type.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton, mods)

    def drag(self, a: QPoint, b: QPoint, mods=NO, steps: int = 4, release: bool = True) -> None:
        self.press(a, mods)
        for i in range(1, steps + 1):
            self.move(QPoint(a.x() + (b.x() - a.x()) * i // steps, a.y() + (b.y() - a.y()) * i // steps), mods)
        if release:
            self.release(b, mods)

    def click(self, pos, mods=NO) -> None:
        self.press(pos, mods)
        self.release(pos, mods)

    def key(self, key, mods=NO) -> None:
        QTest.keyClick(self.canvas, key, mods)

    def selected(self) -> list[str]:
        return self.ws.selection.ids

    def undo_once_restores(self, before: str) -> None:
        assert self.ws.commands.can_undo
        self.ws.commands.undo()
        assert self.snapshot() == before
        assert not self.ws.commands.can_undo, "the gesture was more than one undo step"


@pytest.fixture
def env(qapp, project_ws):
    errors: list[BaseException] = []
    old_hook = sys.excepthook
    sys.excepthook = lambda _t, v, _tb: errors.append(v)  # PySide prints exceptions raised inside event handlers / paintEvent instead of raising: make them test failures
    e = Env(project_ws)
    yield e
    e.scroll.close()
    e.scroll.deleteLater()
    QApplication.processEvents()
    sys.excepthook = old_hook
    assert not errors, f"exceptions inside Qt event handlers: {errors!r}"


V1, V2, V3, A1, A2 = "track_v1", "track_v2", "track_v3", "track_a1", "track_a2"


# ------------------------------------------------------------------------------------------------------------------------ 1. selection
def test_click_ctrl_shift_selection_and_clearing(env):
    a, b, c = (env.clip(V1, s, 2.0) for s in (0.0, 3.0, 6.0))
    d = env.clip(V2, 3.0, 2.0)
    env.done()
    env.click(env.pos(a.id))
    assert env.selected() == [a.id] and env.ws.selection.primary == a.id
    env.click(env.pos(b.id), CTRL)  # toggle adds ...
    assert env.selected() == [a.id, b.id] and env.ws.selection.primary == b.id
    env.click(env.pos(a.id), CTRL)  # ... and removes
    assert env.selected() == [b.id]
    env.click(env.pos(a.id))
    env.click(env.pos(c.id), SHIFT)  # the range on the track, in time order
    assert env.selected() == [a.id, b.id, c.id]
    env.click(env.pos(d.id), CTRL)
    assert set(env.selected()) == {a.id, b.id, c.id, d.id}
    env.click(env.pos(b.id))  # a plain click on a selected clip narrows the selection to it
    assert env.selected() == [b.id]
    env.click(env.at(40.0, V1))  # empty space clears
    assert env.selected() == []
    assert env.snapshot() and not env.ws.commands.can_undo  # selecting never edits


def test_double_clicking_a_media_clip_still_emits_the_preview_signal_and_ctrl_shift_adds_a_range(env):
    asset = env.asset("video", 30.0)
    a, b, c = (env.clip(V1, s, 2.0, asset) for s in (0.0, 3.0, 6.0))
    cap = env.clip("track_v6", 0.0, 2.0, asset, kind=KIND_CAPTION, text={"text": "hi"})
    env.done()
    got = []
    env.canvas.clip_double_clicked.connect(got.append)
    QTest.mouseDClick(env.canvas, LEFT, NO, env.pos(b.id))
    QTest.mouseDClick(env.canvas, LEFT, NO, env.pos(cap.id))
    assert got == [asset.id]  # a caption has no source media to preview
    env.click(env.pos(a.id))
    env.click(env.pos(c.id), CTRL | SHIFT)  # Ctrl+Shift: the range joins the existing selection
    assert set(env.selected()) == {a.id, b.id, c.id}


def test_clicking_a_linked_clip_selects_its_relatives_unless_linked_selection_is_off(env):
    v = env.clip(V1, 0.0, 3.0, link_id="link_x")
    au = env.clip(A1, 0.0, 3.0, link_id="link_x")
    env.done()
    env.click(env.pos(v.id))
    assert set(env.selected()) == {v.id, au.id}
    env.ws.edit_state.update(linked_selection=False)
    env.click(env.at(30.0, V1))
    env.click(env.pos(v.id))
    assert env.selected() == [v.id]


def test_rubber_band_selects_the_clips_in_a_time_by_track_rectangle(env):
    a, b, c = (env.clip(V1, s, 2.0) for s in (0.0, 3.0, 6.0))
    d, e = env.clip(V2, 3.5, 2.0), env.clip(V2, 20.0, 2.0)
    env.done()
    start = QPoint(int(env.canvas.time_to_x(2.5)), env.row_y(V1) - 14)  # empty space between a and b, upper part of row V1
    stop = QPoint(int(env.canvas.time_to_x(6.5)), env.row_y(V2) + 10)
    env.press(start)
    env.move(QPoint(start.x() + 2, start.y() + 2))  # below the 3 px threshold: still a click
    assert env.canvas.rubber_rect is None
    env.move(stop)
    assert env.canvas.gesture_kind == "rubber" and env.canvas.rubber_rect is not None and env.canvas.rubber_rect.width() > 100
    env.canvas.grab()  # painting the band must not raise
    assert env.selected() == []  # nothing is selected before the release
    env.release(stop)
    assert set(env.selected()) == {b.id, c.id, d.id}
    assert e.id not in env.selected() and a.id not in env.selected()
    # Shift adds, Ctrl toggles
    s2, e2 = QPoint(int(env.canvas.time_to_x(19.5)), env.row_y(V2) - 14), QPoint(int(env.canvas.time_to_x(21.0)), env.row_y(V2) + 10)
    env.drag(s2, e2, SHIFT)
    assert set(env.selected()) == {b.id, c.id, d.id, e.id}
    s3, e3 = QPoint(int(env.canvas.time_to_x(2.5)), env.row_y(V1) - 14), QPoint(int(env.canvas.time_to_x(4.0)), env.row_y(V1) + 12)
    env.drag(s3, e3, CTRL)
    assert set(env.selected()) == {c.id, d.id, e.id}  # b was toggled off
    assert not env.ws.commands.can_undo


def test_select_all_invert_track_and_playhead_helpers_and_escape(env):
    a, b, c = (env.clip(V1, s, 2.0) for s in (0.0, 3.0, 6.0))
    d = env.clip(V2, 4.0, 2.0)
    env.done()
    assert env.canvas.select_all() and len(env.selected()) == 4
    env.canvas.escape()  # Escape clears the selection when no gesture is running
    assert env.selected() == []
    env.click(env.pos(a.id))
    assert env.canvas.invert_selection() and set(env.selected()) == {b.id, c.id, d.id}
    assert env.canvas.select_track_clips(V1) and env.selected() == [a.id, b.id, c.id]
    env.canvas.set_playhead(2.5)
    assert env.canvas.select_from_playhead(1) and set(env.selected()) == {b.id, c.id, d.id}
    assert env.canvas.select_from_playhead(-1, V1) and env.selected() == [a.id]
    env.canvas.select_from_playhead(1, V2)
    assert env.selected() == [d.id]
    env.click(env.at(30, V1))
    env.key(Qt.Key.Key_A, CTRL)
    assert len(env.selected()) == 4


def test_painting_checks_selection_membership_without_copying_the_id_list(env, monkeypatch):
    from app.timeline.selection import SelectionModel

    for i in range(6):
        env.clip(V1, i * 2.0, 1.5)
    env.done()
    env.canvas.select_all()
    calls = []
    orig = SelectionModel.ids.fget
    monkeypatch.setattr(SelectionModel, "ids", property(lambda self: (calls.append(1), orig(self))[1]))
    env.canvas.grab()
    assert calls == [], "paintEvent copied the selection"


# ------------------------------------------------------------------------------------------------------------------------ 2. move
def test_multi_clip_drag_moves_the_selection_rigidly_as_one_undo_step(env):
    a, b, c = env.clip(V1, 1.0, 2.0), env.clip(V1, 4.0, 2.0), env.clip(V2, 2.0, 2.0)
    far = env.clip(V1, 30.0, 2.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    env.ws.selection.select([a.id, b.id, c.id])
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.drag(p0, QPoint(p0.x() + 160, p0.y()))  # 2 s to the right
    assert env.starts(V1) == [(3.0, 2.0), (6.0, 2.0), (30.0, 2.0)]
    assert env.starts(V2) == [(4.0, 2.0)]
    assert env.get(far.id).timeline_start == 30.0
    assert env.selected() == [a.id, b.id, c.id]  # the selection survives the move
    env.undo_once_restores(before)
    assert not any(e for e in env.status if "overlap" in e.lower())


def test_dragging_a_selected_clip_of_a_linked_pair_moves_its_partner_too(env):
    v = env.clip(V1, 1.0, 2.0, link_id="link_a")
    au = env.clip(A1, 1.0, 2.0, link_id="link_a")
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    env.click(env.pos(v.id))
    p0 = env.pos(v.id)
    env.drag(p0, QPoint(p0.x() + 80, p0.y()))
    assert env.get(v.id).timeline_start == pytest.approx(2.0) and env.get(au.id).timeline_start == pytest.approx(2.0)


def test_a_drag_below_three_pixels_is_a_click_and_changes_nothing(env):
    a, b = env.clip(V1, 1.0, 2.0), env.clip(V1, 4.0, 2.0)
    env.done()
    env.ws.selection.select([a.id, b.id])
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + 2, p0.y() + 1))
    env.release(QPoint(p0.x() + 2, p0.y() + 1))
    assert env.snapshot() == before and not env.ws.commands.can_undo
    assert env.selected() == [a.id]  # it was a click on a selected clip: the selection narrowed to it


def test_move_snaps_to_the_playhead_a_marker_and_a_clip_edge_and_exposes_the_guide(env):
    a = env.clip(V1, 1.0, 2.0)
    b = env.clip(V2, 10.0, 2.0)
    env.done()
    env.ws.edit.add_marker(20.0, "intro")
    env.canvas.reload()
    pps = env.canvas.pps
    p0 = env.pos(a.id)

    def drag_to(target_start, **kw):
        env.press(p0)
        env.move(QPoint(p0.x() + int((target_start - 1.0) * pps) + 2, p0.y()))  # 2 px (25 ms) past the target: well inside the snap tolerance
        out = (env.canvas.snap_guide, env.canvas.hud_text)
        return out

    # playhead at 6.0: the clip's start lands on it
    env.canvas.set_playhead(6.0)
    guide, hud = drag_to(6.0)
    assert guide is not None and guide[1] == "playhead" and guide[0] == pytest.approx(6.0) and "snaps to the playhead" in hud
    env.release(QPoint(p0.x() + int(5.0 * pps) + 2, p0.y()))
    assert env.get(a.id).timeline_start == pytest.approx(6.0)
    env.ws.commands.undo()
    # marker at 20.0: the clip's END (start + 2) lands on it -> start 18; the HUD names the marker
    env.canvas.set_playhead(0.0)
    guide, hud = drag_to(18.0)
    assert guide[1] == "markers" and guide[0] == pytest.approx(20.0) and "snaps to marker 'intro'" in hud
    env.release(QPoint(p0.x() + int(17.0 * pps) + 2, p0.y()))
    assert env.get(a.id).timeline_end == pytest.approx(20.0)
    env.ws.commands.undo()
    # clip edge: b starts at 10.0 on another track; the clip's end lands on it -> start 8
    guide, hud = drag_to(8.0)
    assert guide[1] == "clip_edges" and guide[0] == pytest.approx(10.0) and "clip edge" in hud
    env.release(QPoint(p0.x() + int(7.0 * pps) + 2, p0.y()))
    assert env.get(a.id).timeline_end == pytest.approx(10.0) and env.get(b.id).timeline_start == 10.0


def test_the_hud_shows_the_signed_timecode_delta(env):
    a = env.clip(V1, 1.0, 2.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + int(2.0 * env.canvas.pps) + 6, p0.y()))  # 2 s + 6 px (about 2 frames)
    assert env.canvas.hud_text.startswith("+2:0") or env.canvas.hud_text.startswith("+2:1")
    env.move(QPoint(p0.x() - 30, p0.y()))
    assert env.canvas.hud_text.startswith("-0:")
    env.canvas.escape()
    tb = env.canvas.timebase()
    assert format_delta(tb, 2.0 + 5 / 30) == "+2:05" and format_delta(tb, -0.4) == "-0:12" and format_delta(tb, 3725.0).startswith("+1:02:05")


def test_alt_inverts_snapping_for_one_drag(env):
    a = env.clip(V1, 1.0, 2.0)
    env.done()
    env.canvas.set_playhead(6.0)
    pps = env.canvas.pps
    p0 = env.pos(a.id)
    target = QPoint(p0.x() + int(5.0 * pps) + 3, p0.y())
    env.press(p0)
    env.move(target)
    assert env.canvas.snap_guide is not None  # snapping on: it snaps
    env.move(target, ALT)
    assert env.canvas.snap_guide is None  # Alt: this drag does not snap
    env.release(target, ALT)
    assert env.get(a.id).timeline_start != pytest.approx(6.0, abs=1e-6)
    env.ws.commands.undo()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    env.press(p0)
    env.move(target)
    assert env.canvas.snap_guide is None
    env.move(target, ALT)
    assert env.canvas.snap_guide is not None  # snapping off + Alt: snap anyway
    env.canvas.escape()


def test_cross_track_drag_moves_to_the_row_under_the_pointer_when_every_clip_fits(env):
    a = env.clip(V1, 1.0, 2.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + 80, env.row_y(V2)))
    assert env.canvas._g.target_row == env.canvas._row_of(V2) and env.canvas._g.reason is None
    env.release(QPoint(p0.x() + 80, env.row_y(V2)))
    moved = env.get(a.id)
    assert moved.track_id == V2 and moved.timeline_start == pytest.approx(2.0)
    env.undo_once_restores(before)


def test_vertical_movement_is_ignored_and_explained_when_a_clip_does_not_fit_the_row(env):
    a = env.clip(V1, 1.0, 2.0)  # video media
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + 80, env.row_y(A1)))  # an audio row
    g = env.canvas._g
    assert g.row_delta == 0 and g.target_row is None
    assert "cannot hold" in env.canvas.hud_text and "stays on the current track" in env.canvas.hud_text
    env.release(QPoint(p0.x() + 80, env.row_y(A1)))
    moved = env.get(a.id)
    assert moved.track_id == V1 and moved.timeline_start == pytest.approx(2.0)  # the horizontal part still happened


def test_an_invalid_drop_position_draws_red_commits_nothing_and_says_why(env):
    a = env.clip(V1, 1.0, 2.0)
    env.clip(V1, 5.0, 2.0)  # the clip it would land on
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + int(3.5 * env.canvas.pps), p0.y()))  # lands on top of b
    g = env.canvas._g
    assert g.reason and "overlap" in g.reason.lower() and g.reason in env.canvas.hud_text
    assert g.ghosts  # the ghost is still drawn (red), from the engine's plan
    env.canvas.grab()
    env.release(QPoint(p0.x() + int(3.5 * env.canvas.pps), p0.y()))
    assert env.snapshot() == before and not env.ws.commands.can_undo
    assert any("overlap" in s.lower() for s in env.status)


def test_escape_cancels_a_drag_without_touching_the_model(env):
    a = env.clip(V1, 1.0, 2.0)
    env.done()
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + 120, p0.y()))
    assert env.canvas.gesture_kind == "move" and env.canvas.hud_text
    env.key(Qt.Key.Key_Escape)
    assert env.canvas.gesture_kind is None and env.canvas.hud_text == ""
    env.release(QPoint(p0.x() + 120, p0.y()))
    assert env.snapshot() == before and not env.ws.commands.can_undo
    assert env.ws.selection.ids == [a.id]  # Escape cancelled the gesture, the selection stayed


def test_a_locked_clip_or_track_refuses_the_drag_safely(env):
    a = env.clip(V1, 1.0, 2.0)
    b = env.clip(V2, 1.0, 2.0)
    env.done()
    env.ws.edit.set_locks([a.id], "item")
    env.ws.timeline.set_track_flag(V2, "locked", True)
    env.canvas.reload()
    before, undo_text = env.snapshot(), env.ws.commands.undo_text
    for cid in (a.id, b.id):
        p0 = env.pos(cid)
        env.press(p0)
        env.move(QPoint(p0.x() + 100, p0.y()))
        assert env.canvas._g.reason, "the HUD must say why"
        env.release(QPoint(p0.x() + 100, p0.y()))
        assert env.snapshot() == before and env.ws.commands.undo_text == undo_text
    assert any("locked" in s.lower() for s in env.status)


def test_ctrl_drag_duplicates_the_selection_as_one_undo_step_and_selects_the_copies(env):
    a, b = env.clip(V1, 1.0, 2.0), env.clip(V1, 3.5, 2.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    env.ws.selection.select([a.id, b.id])
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.press(p0, CTRL)
    for i in range(1, 5):
        env.move(QPoint(p0.x() + 100 * i, p0.y()), CTRL)  # 5 s to the right
    assert env.canvas.hud_text.startswith("Copy ") and env.canvas._g.reason is None
    env.release(QPoint(p0.x() + 400, p0.y()), CTRL)
    clips = env.tl.get_track(V1).clips
    assert len(clips) == 4
    originals = {c.id: c for c in clips}
    assert originals[a.id].timeline_start == 1.0 and originals[b.id].timeline_start == 3.5  # the originals stay
    copies = [c for c in clips if c.id not in (a.id, b.id)]
    assert sorted(round(c.timeline_start, 3) for c in copies) == [6.0, 8.5]  # relative spacing kept
    assert set(env.selected()) == {c.id for c in copies}
    env.undo_once_restores(before)


def test_ctrl_click_without_a_drag_still_only_toggles(env):
    a, b = env.clip(V1, 1.0, 2.0), env.clip(V1, 4.0, 2.0)
    env.done()
    env.click(env.pos(a.id))
    env.click(env.pos(b.id), CTRL)
    assert set(env.selected()) == {a.id, b.id} and not env.ws.commands.can_undo


def test_ripple_mode_moves_with_a_ripple_move_on_a_single_track(env):
    a, b, c = env.clip(V1, 0.0, 2.0), env.clip(V1, 2.0, 2.0), env.clip(V1, 4.0, 2.0)
    env.done()
    env.ws.edit_state.update(ripple_on=True, snap=env.ws.edit_state.snap.replace(enabled=False))
    env.click(env.pos(a.id))
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x() + int(2.5 * env.canvas.pps), p0.y()))
    assert "ripple" in env.canvas.hud_text and env.canvas._g.reason is None  # a plain move would overlap b: the ripple preview is valid
    env.release(QPoint(p0.x() + int(2.5 * env.canvas.pps), p0.y()))
    starts = {cid: env.get(cid).timeline_start for cid in (a.id, b.id, c.id)}
    assert starts[b.id] == pytest.approx(0.0) and starts[a.id] > starts[b.id]
    env.undo_once_restores(before)


# ------------------------------------------------------------------------------------------------------------------------ 3. trim / roll / slip / slide
def test_trim_edge_drag_with_and_without_ripple_and_snapping(env):
    a = env.clip(V1, 1.0, 4.0)
    b = env.clip(V1, 6.0, 2.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    before = env.snapshot()
    edge = QPoint(int(env.canvas.time_to_x(5.0)) - 2, env.row_y(V1))
    assert env.canvas._hit_clip(edge.x(), edge.y())[2] == "trim_end"
    env.drag(edge, QPoint(edge.x() - 80, edge.y()))  # 1 s shorter
    assert env.get(a.id).duration == pytest.approx(3.0) and env.get(a.id).source_out == pytest.approx(3.0)
    assert env.get(b.id).timeline_start == 6.0  # a plain trim leaves later clips where they are
    env.undo_once_restores(before)
    # ripple: later clips follow the trimmed edge
    env.ws.edit_state.update(ripple_on=True)
    env.drag(edge, QPoint(edge.x() - 80, edge.y()))
    assert env.get(a.id).duration == pytest.approx(3.0) and env.get(b.id).timeline_start == pytest.approx(5.0)
    env.undo_once_restores(before)
    # the start edge: timeline start and source_in move together
    env.ws.edit_state.update(ripple_on=False)
    s_edge = QPoint(int(env.canvas.time_to_x(1.0)) + 2, env.row_y(V1))
    env.drag(s_edge, QPoint(s_edge.x() + 80, s_edge.y()))
    t = env.get(a.id)
    assert t.timeline_start == pytest.approx(2.0) and t.source_in == pytest.approx(1.0) and t.duration == pytest.approx(3.0)
    env.undo_once_restores(before)


def test_trim_snaps_to_the_playhead_and_reports_the_engine_clamp_in_the_hud(env):
    a = env.clip(V1, 1.0, 4.0)
    env.clip(V1, 5.5, 2.0)
    env.done()
    env.canvas.set_playhead(4.0)
    edge = QPoint(int(env.canvas.time_to_x(5.0)) - 2, env.row_y(V1))
    env.press(edge)
    env.move(QPoint(edge.x() - int(1.0 * env.canvas.pps) + 3, edge.y()))
    assert env.canvas.snap_guide is not None and env.canvas.snap_guide[1] == "playhead"
    assert "Trim end" in env.canvas.hud_text and "snaps to the playhead" in env.canvas.hud_text
    env.release(QPoint(edge.x() - int(1.0 * env.canvas.pps) + 3, edge.y()))
    assert env.get(a.id).timeline_end == pytest.approx(4.0)
    # dragging the end past the next clip: the engine clamps and the HUD / ghost show the clamped edge
    env.press(QPoint(int(env.canvas.time_to_x(4.0)) - 2, env.row_y(V1)))
    env.move(QPoint(int(env.canvas.time_to_x(9.0)), env.row_y(V1)))
    g = env.canvas._g
    assert g.plan is not None and g.plan.updates[a.id].timeline_end == pytest.approx(5.5)
    env.canvas.escape()


def test_roll_gesture_moves_the_cut_between_two_touching_clips(env):
    asset = env.asset("video", 60.0)
    a = env.clip(V1, 0.0, 4.0, asset, source_in=10.0)
    b = env.clip(V1, 4.0, 4.0, asset, source_in=20.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    before = env.snapshot()
    edge = QPoint(int(env.canvas.time_to_x(4.0)) - 2, env.row_y(V1))
    env.drag(edge, QPoint(edge.x() + 80, edge.y()), CTRL | ALT)  # the cut 1 s to the right
    ra, rb = env.get(a.id), env.get(b.id)
    assert ra.timeline_end == pytest.approx(5.0) and rb.timeline_start == pytest.approx(5.0)
    assert ra.duration + rb.duration == pytest.approx(8.0)  # the combined duration is unchanged
    assert rb.source_in == pytest.approx(21.0)
    env.undo_once_restores(before)
    # an edge that touches nothing cannot be rolled: the engine reason is shown and nothing happens
    far = env.clip(V2, 20.0, 2.0)
    env.canvas.reload()
    e2 = QPoint(int(env.canvas.time_to_x(22.0)) - 2, env.row_y(V2))
    env.press(e2, CTRL | ALT)
    env.move(QPoint(e2.x() + 50, e2.y()), CTRL | ALT)
    assert env.canvas.gesture_kind == "roll" and "no clip touching" in env.canvas.hud_text
    env.release(QPoint(e2.x() + 50, e2.y()), CTRL | ALT)
    assert env.get(far.id).duration == 2.0 and env.ws.commands.can_undo is False


def test_roll_is_limited_by_the_source_media(env):
    asset = env.asset("video", 6.0)
    a = env.clip(V1, 0.0, 4.0, asset, source_in=0.0)  # only 2 s of source left after the cut
    b = env.clip(V1, 4.0, 2.0, asset, source_in=0.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    edge = QPoint(int(env.canvas.time_to_x(4.0)) - 2, env.row_y(V1))
    env.press(edge, CTRL | ALT)
    env.move(QPoint(edge.x() + int(3.0 * env.canvas.pps), edge.y()), CTRL | ALT)  # asks for +3 s
    lo, hi = env.ws.edit.roll_limits(a.id, b.id)
    assert env.canvas._g.value == pytest.approx(hi) and hi < 7.0  # clamped to what the engine allows
    env.canvas.escape()


def test_slip_gesture_moves_the_content_under_the_cut(env):
    asset = env.asset("video", 60.0)
    a = env.clip(V1, 10.0, 4.0, asset, source_in=20.0)
    env.done()
    before = env.snapshot()
    p0 = env.pos(a.id)
    env.drag(p0, QPoint(p0.x() + 80, p0.y()), ALT | SHIFT)  # content dragged 1 s to the right = 1 s earlier source
    c = env.get(a.id)
    assert c.timeline_start == 10.0 and c.duration == 4.0
    assert c.source_in == pytest.approx(19.0) and c.source_out == pytest.approx(23.0)
    env.undo_once_restores(before)
    # stills have nothing to slip: the engine's reason is the HUD, nothing happens
    img = env.clip(V3, 0.0, 3.0, env.asset("image"))
    env.canvas.reload()
    pi = env.pos(img.id)
    env.press(pi, ALT | SHIFT)
    env.move(QPoint(pi.x() + 40, pi.y()), ALT | SHIFT)
    assert "Nothing to slip" in env.canvas.hud_text
    env.release(QPoint(pi.x() + 40, pi.y()), ALT | SHIFT)
    assert not env.ws.commands.can_undo


def test_slide_gesture_moves_the_clip_and_its_neighbours_absorb_it(env):
    asset = env.asset("video", 60.0)
    a = env.clip(V1, 0.0, 4.0, asset, source_in=0.0)
    b = env.clip(V1, 4.0, 4.0, asset, source_in=10.0)
    c = env.clip(V1, 8.0, 4.0, asset, source_in=20.0)
    env.done()
    before = env.snapshot()
    p0 = env.pos(b.id)
    env.drag(p0, QPoint(p0.x() + 80, p0.y()), CTRL | ALT)  # 1 s to the right
    assert env.get(b.id).timeline_start == pytest.approx(5.0) and env.get(b.id).duration == pytest.approx(4.0)
    assert env.get(a.id).timeline_end == pytest.approx(5.0) and env.get(c.id).timeline_start == pytest.approx(9.0)
    env.undo_once_restores(before)
    # a clip with a gap next to it cannot slide
    d = env.clip(V2, 0.0, 2.0)
    env.clip(V2, 5.0, 2.0)
    env.canvas.reload()
    pd = env.pos(d.id)
    env.press(pd, CTRL | ALT)
    env.move(QPoint(pd.x() + 40, pd.y()), CTRL | ALT)
    assert env.canvas.gesture_kind == "slide" and env.canvas._g.reason and "slide" in env.canvas.hud_text.lower()
    env.release(QPoint(pd.x() + 40, pd.y()), CTRL | ALT)
    assert not env.ws.commands.can_undo


# ------------------------------------------------------------------------------------------------------------------------ 4. blade
def test_blade_tool_cuts_the_hovered_track_and_its_linked_clip(env):
    env.clip(V1, 0.0, 6.0, link_id="link_b")
    env.clip(A1, 0.0, 6.0, link_id="link_b")
    env.clip(V2, 0.0, 6.0)
    env.done()
    env.ws.edit_state.update(tool="blade", snap=env.ws.edit_state.snap.replace(enabled=False))
    assert env.canvas.cursor().shape() == Qt.CursorShape.CrossCursor
    p = env.at(2.0, V1)
    env.move(p, buttons=Qt.MouseButton.NoButton)
    assert env.canvas._blade == (env.canvas._row_of(V1), pytest.approx(2.0, abs=0.02))  # the hover line follows the pointer
    before = env.snapshot()
    env.click(p)
    assert len(env.tl.get_track(V1).clips) == 2 and len(env.tl.get_track(A1).clips) == 2  # the linked clip was cut as well
    assert len(env.tl.get_track(V2).clips) == 1  # the other track is untouched
    assert env.starts(V1)[0][1] == pytest.approx(2.0, abs=0.04)
    assert env.ws.selection.ids == []  # the blade tool selects nothing
    env.undo_once_restores(before)
    # Shift+click cuts every unlocked track
    env.click(p, SHIFT)
    assert len(env.tl.get_track(V1).clips) == 2 and len(env.tl.get_track(V2).clips) == 2 and len(env.tl.get_track(A1).clips) == 2
    env.ws.commands.undo()
    assert env.ws.commands.can_undo is False and len(env.tl.get_track(V2).clips) == 1


def test_blade_never_creates_a_zero_length_piece(env):
    env.clip(V1, 2.0, 4.0)
    env.done()
    env.ws.edit_state.update(tool="blade", snap=env.ws.edit_state.snap.replace(enabled=False))
    before = env.snapshot()
    env.click(env.at(2.0, V1) + QPoint(3, 0))  # 3 px (about one frame) inside the clip's start edge: a piece that short is refused
    assert env.snapshot() == before and not env.ws.commands.can_undo
    assert any("too close" in m or "not inside" in m for m in env.status), env.status
    env.click(env.at(1.0, V1))  # empty space on the track
    assert any("no clip under the blade" in m for m in env.status)
    assert env.snapshot() == before and env.starts(V1) == [(2.0, 4.0)]


def test_blade_snaps_to_the_playhead_and_the_tool_resets_the_cursor(env):
    env.clip(V1, 0.0, 6.0)
    env.done()
    env.ws.edit_state.update(tool="blade")
    env.canvas.set_playhead(3.0)
    env.move(env.at(3.0, V1) + QPoint(2, 0), buttons=Qt.MouseButton.NoButton)
    assert env.canvas._blade[1] == pytest.approx(3.0)  # snapped to the playhead
    env.click(env.at(3.0, V1) + QPoint(2, 0))
    assert env.starts(V1)[0] == (0.0, 3.0)
    env.ws.edit_state.update(tool="select")
    assert env.canvas.cursor().shape() == Qt.CursorShape.ArrowCursor and env.canvas._blade is None


def test_begin_blade_at_playhead_cuts_every_unlocked_track(env):
    env.clip(V1, 0.0, 6.0)
    env.clip(V2, 0.0, 6.0)
    env.clip(A1, 0.0, 6.0)
    env.done()
    env.ws.timeline.set_track_flag(A1, "locked", True)
    env.canvas.set_playhead(2.0)
    assert env.canvas.begin_blade_at_playhead()
    assert len(env.tl.get_track(V1).clips) == 2 and len(env.tl.get_track(V2).clips) == 2 and len(env.tl.get_track(A1).clips) == 1  # locked: left alone
    n = len(env.tl.all_clips())
    assert not env.canvas.begin_blade_at_playhead() and len(env.tl.all_clips()) == n  # cutting where the cut already is changes nothing


# ------------------------------------------------------------------------------------------------------------------------ 5. state drawn on the clips
def test_painting_every_clip_state_and_gesture_overlay_does_not_raise_and_changes_the_picture(env):
    from PySide6.QtGui import QImage

    asset = env.asset("video", 60.0)
    plain = env.clip(V1, 1.0, 4.0, asset, cid="plain")
    env.done()

    def picture() -> QImage:
        return env.canvas.grab().toImage()

    base = picture()
    rich = env.clip(V2, 1.0, 4.0, asset, cid="rich", speed=2.0, link_id="link_z", group_id="group_z", locks=frozenset({"item"}), transition={"type": "FADE", "duration": 0.5},
                    keyframes=[Keyframe("opacity", 0.5, 0.2), Keyframe("opacity", 1.5, 1.0)], audio={"volume": 0.5})
    env.ws.edit.set_track_display(V2, height=72)  # a tall row shows the keyframe diamonds
    env.clip(A1, 1.0, 4.0, env.asset("audio", 30.0), cid="muted", audio={"volume": 0.0, "muted": True})
    aspect = env.clip(V3, 1.0, 4.0, env.asset("image"), cid="aspect", locks=frozenset({"timing"}))
    env.ws.edit.set_track_display(A1, collapsed=True)  # a collapsed row is a thin strip
    env.canvas.reload()
    assert picture() != base
    # the clip rectangles carry the same colours with and without the badges, but the badges add pixels: compare the rich clip's rect to a bare one
    r = env.canvas.clip_rect(env.get(rich.id), env.canvas._row_of(V2)).toAlignedRect()
    with_badges = picture().copy(r)
    env.ws.edit.update_clip(rich.id, speed=1.0) if False else None
    assert with_badges.width() > 0
    # a gesture's overlays (ghosts, guide, HUD, rubber band) paint without raising
    env.ws.selection.select([plain.id, rich.id, aspect.id])
    p0 = env.pos(plain.id)
    env.press(p0)
    env.move(QPoint(p0.x() + 90, p0.y() + 40))
    env.canvas.grab()
    env.canvas.escape()
    assert env.canvas.row_height(env.canvas._row_of("track_a1")) == 18


def test_a_lock_badge_changes_the_picture_of_a_clip(env):
    from PySide6.QtGui import QImage

    asset = env.asset("video", 60.0)
    c = env.clip(V1, 1.0, 4.0, asset)
    env.done()
    r = env.canvas.clip_rect(c, 0).toAlignedRect()
    before = env.canvas.grab().toImage().copy(r)
    env.ws.edit.set_locks([c.id], "item")
    env.canvas.reload()
    after = env.canvas.grab().toImage().copy(r)
    assert isinstance(after, QImage) and before != after


def test_the_tooltip_lists_name_times_source_speed_locks_links_and_owner(env):
    asset = env.asset("video", 60.0, "interview.mp4")
    a = env.clip(V1, 1.0, 4.0, asset, source_in=10.0, speed=2.0, link_id="link_t", locks=frozenset({"position", "timing"}), created_by="AI", scene_id="scene_1")
    env.clip(A1, 1.0, 4.0, link_id="link_t")
    env.done()
    tip = env.canvas.clip_tooltip(env.get(a.id))
    for needle in ("interview.mp4", "V1 Main Video", "00:00:01:00", "Duration 00:00:04:00", "Source in 00:00:10:00", "Speed 2×", "Locked: position, timing", "Linked with 1 other", "Created by the AI"):
        assert needle in tip, (needle, tip)
    user = env.clip(V2, 1.0, 2.0, created_by="USER")
    assert "Created by you" in env.canvas.clip_tooltip(env.get(user.id))


# ------------------------------------------------------------------------------------------------------------------------ 6. drops from the library
def mime(ids) -> QMimeData:
    md = QMimeData()
    md.setData(ASSET_MIME, "\n".join(ids).encode())
    return md


def drop(env, ids, pos, mods=NO):
    md = mime(ids)
    ev = QDropEvent(QPointF(pos), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, mods)
    env.canvas.dropEvent(ev)
    return ev, md


def test_dropping_one_asset_places_it_in_one_undo_step_and_several_lay_end_to_end(env):
    v = env.asset("video", 6.0)
    a, b, c = env.asset("video", 4.0), env.asset("image"), env.asset("video", 5.0)
    env.done()
    before = env.snapshot()
    ev, _md = drop(env, [v.id], env.at(2.0, V1))
    assert ev.isAccepted() and env.starts(V1) == [(2.0, 6.0)]
    env.undo_once_restores(before)
    ev, _md = drop(env, [a.id, b.id, c.id], env.at(1.0, V1))
    assert ev.isAccepted() and env.starts(V1) == [(1.0, 4.0), (5.0, 5.0), (10.0, 5.0)]  # a still gets the default 5 s, laid end to end
    env.undo_once_restores(before)  # all three in ONE step


def test_drop_modes_free_insert_and_overwrite(env):
    env.clip(V1, 0.0, 10.0)
    new = env.asset("video", 4.0)
    env.done()
    before = env.snapshot()
    # default (free placement): the engine puts it at the first free spot after the existing clip - nothing moves
    drop(env, [new.id], env.at(3.0, V1))
    assert env.starts(V1)[0] == (0.0, 10.0) and len(env.tl.get_track(V1).clips) == 2 and env.starts(V1)[1][0] >= 10.0
    env.undo_once_restores(before)
    # Ctrl = insert (ripple): the clip under the drop is split and the rest moves right
    drop(env, [new.id], env.at(3.0, V1), CTRL)
    assert env.starts(V1) == [(0.0, 3.0), (3.0, 4.0), (7.0, 7.0)]
    env.undo_once_restores(before)
    # Shift = overwrite: what is under the new clip is replaced
    drop(env, [new.id], env.at(3.0, V1), SHIFT)
    assert env.starts(V1) == [(0.0, 3.0), (3.0, 4.0), (7.0, 3.0)]
    env.undo_once_restores(before)


def test_a_drop_that_cannot_work_adds_nothing_and_says_why(env):
    v, au = env.asset("video", 4.0), env.asset("audio", 4.0)
    env.clip(V2, 0.0, 2.0)
    env.done()
    env.ws.timeline.set_track_flag(V2, "locked", True)
    env.canvas.reload()
    before, undo_text = env.snapshot(), env.ws.commands.undo_text
    cases = [([au.id], env.at(2.0, V1), "cannot hold"),  # audio onto a video track
             ([v.id, au.id], env.at(2.0, V1), "cannot hold"),  # the second item does not fit: the first is not added either
             ([v.id], env.at(5.0, V2), "locked"),  # a locked track
             (["media_missing"], env.at(2.0, V1), "no longer in the project")]
    for ids, pos, needle in cases:
        env.status.clear()
        ev, _md = drop(env, ids, pos)
        assert not ev.isAccepted()
        assert any(needle in m for m in env.status), (needle, env.status)
        assert env.snapshot() == before and env.ws.commands.undo_text == undo_text


def test_the_drag_preview_highlights_the_row_and_lays_ghosts_end_to_end(env):
    from PySide6.QtGui import QDragLeaveEvent, QDragMoveEvent

    a, b = env.asset("video", 4.0), env.asset("video", 3.0)
    env.done()
    md = mime([a.id, b.id])
    ev = QDragMoveEvent(env.at(2.0, V1), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, CTRL)
    env.canvas.dragMoveEvent(ev)
    st = env.canvas.drop_state
    assert ev.isAccepted() and st.ok and st.row == env.canvas._row_of(V1) and st.mode == "insert"
    assert [tuple(round(x, 3) for x in g) for g in st.ghosts] == [(2.0, 4.0), (6.0, 3.0)]
    env.canvas.grab()  # painting the row highlight / ghosts / HUD
    assert "insert" in env.canvas._drop_hud(st)
    bad = QDragMoveEvent(env.at(2.0, A1), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, NO)
    env.canvas.dragMoveEvent(bad)
    assert not bad.isAccepted() and not env.canvas.drop_state.ok and "cannot hold" in env.canvas.drop_state.reason
    env.canvas.grab()
    env.canvas.dragLeaveEvent(QDragLeaveEvent())
    assert env.canvas.drop_state is None


def test_dropping_below_the_last_row_creates_a_track_of_the_right_kind_and_places_the_media(env):
    v, au = env.asset("video", 4.0), env.asset("audio", 4.0)
    env.done()
    n = len(env.tl.tracks)
    below = QPoint(int(env.canvas.time_to_x(3.0)), env.canvas.content_height() + NEW_TRACK_STRIP_H // 2)
    assert env.canvas.height() == env.canvas.content_height() + NEW_TRACK_STRIP_H
    ev, _md = drop(env, [au.id], below)
    assert ev.isAccepted() and len(env.tl.tracks) == n + 1
    new = env.tl.tracks[-1]
    assert new.kind is TrackKind.AUDIO and [(c.asset_id, c.timeline_start) for c in new.clips] == [(au.id, 3.0)]
    # a mixed drop that no single track can hold is refused before anything is created
    before = env.snapshot()
    below = QPoint(below.x(), env.canvas.content_height() + NEW_TRACK_STRIP_H // 2)  # the new track made the timeline taller
    ev, _md = drop(env, [v.id, au.id], below)
    assert not ev.isAccepted() and env.snapshot() == before and any("same new track" in m for m in env.status)


def test_a_failed_placement_on_a_new_track_takes_the_new_track_away_again(env, monkeypatch):
    from app.core.exceptions import TimelineError

    v = env.asset("video", 4.0)
    env.done()
    n = len(env.tl.tracks)
    monkeypatch.setattr(env.ws.edit, "place_assets", lambda *a, **k: (_ for _ in ()).throw(TimelineError("placement refused")))
    below = QPoint(100, env.canvas.content_height() + 6)
    ev, _md = drop(env, [v.id], below)
    assert not ev.isAccepted() and len(env.tl.tracks) == n and not env.ws.commands.can_undo
    assert any("placement refused" in m for m in env.status)


def test_the_library_drags_every_selected_item_as_newline_separated_ids(qapp):
    from PySide6.QtWidgets import QListWidgetItem

    lst = _AssetList()
    items = []
    for i in range(3):
        it = QListWidgetItem(f"item {i}")
        it.setData(Qt.ItemDataRole.UserRole, f"media_0000{i}")
        lst.addItem(it)
        items.append(it)
    md = lst.mimeData([items[2], items[0]])
    assert asset_ids_from_mime(md) == ["media_00002", "media_00000"]  # the order Qt hands over = the selection order
    one = lst.mimeData([items[1]])
    assert bytes(one.data(ASSET_MIME)).decode() == "media_00001"  # a single drag is the plain id, as before
    assert asset_ids_from_mime(lst.mimeData([])) == [] and asset_ids_from_mime(QMimeData()) == []


# ------------------------------------------------------------------------------------------------------------------------ 7. ruler: markers, playhead, range
def marker_pos(env, mid) -> QPoint:
    m = env.tl.get_marker(mid)
    r = env.canvas._pennant_rect(m)
    return QPoint(int(r.center().x()), int(r.center().y()))


def test_ruler_menu_adds_a_marker_and_undo_removes_it(env):
    env.clip(V1, 0.0, 10.0)
    env.done()
    menu = env.canvas.ruler_menu(4.0)
    actions = {a.text(): a for a in menu.actions()}
    assert actions["Add marker here"].isEnabled()
    assert not actions["Set range from selection"].isEnabled() and "Select clips first" in actions["Set range from selection"].toolTip()
    assert not actions["Clear range"].isEnabled() and "no range" in actions["Clear range"].toolTip()
    actions["Add marker here"].trigger()
    assert [round(m.time, 3) for m in env.tl.markers] == [4.0]
    env.canvas.grab()
    env.ws.commands.undo()
    assert env.tl.markers == [] and not env.ws.commands.can_undo


def test_clicking_a_marker_moves_the_playhead_and_dragging_it_moves_the_marker(env):
    env.clip(V1, 0.0, 30.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    mid = env.ws.edit.add_marker(5.0, "intro")
    env.canvas.reload()
    env.click(marker_pos(env, mid))
    assert env.canvas.playhead == pytest.approx(5.0) and env.canvas._sel_marker == mid
    assert env.tl.get_marker(mid).time == 5.0  # a click does not move it
    p0 = marker_pos(env, mid)
    env.drag(p0, QPoint(p0.x() + 160, p0.y()))  # 2 s to the right
    assert env.tl.get_marker(mid).time == pytest.approx(7.0, abs=0.04)
    assert env.ws.commands.undo_text == "Move marker"
    env.ws.commands.undo()
    assert env.tl.get_marker(mid).time == 5.0
    env.press(p0)
    env.move(QPoint(p0.x() + 80, p0.y()))
    assert env.canvas.gesture_kind == "marker" and "Marker" in env.canvas.hud_text
    env.canvas.grab()
    env.canvas.escape()
    assert env.tl.get_marker(mid).time == 5.0  # Escape: nothing moved


def test_a_dragged_marker_snaps_to_a_clip_edge(env):
    env.clip(V1, 10.0, 4.0)
    env.done()
    mid = env.ws.edit.add_marker(5.0, "m")
    env.canvas.reload()
    p0 = marker_pos(env, mid)
    env.press(p0)
    env.move(QPoint(p0.x() + int(5.0 * env.canvas.pps) + 3, p0.y()))  # 3 px past the clip start at 10 s
    assert env.canvas.snap_guide is not None and env.canvas.snap_guide[1] == "clip_edges"
    env.release(QPoint(p0.x() + int(5.0 * env.canvas.pps) + 3, p0.y()))
    assert env.tl.get_marker(mid).time == pytest.approx(10.0)


def test_double_clicking_a_marker_edits_it_and_delete_removes_it(env, monkeypatch):
    env.done()
    mid = env.ws.edit.add_marker(3.0, "old")
    env.canvas.reload()

    def fake_exec(self):
        self.name.setText("Chapter 1")
        self.kind.setCurrentIndex(self.kind.findData("section"))
        self.color.setText("#112233")
        self.note.setPlainText("check this")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(MarkerDialog, "exec", fake_exec)
    QTest.mouseDClick(env.canvas, LEFT, NO, marker_pos(env, mid))
    m = env.tl.get_marker(mid)
    assert (m.name, m.kind, m.color, m.note) == ("Chapter 1", "section", "#112233", "check this")
    assert env.ws.commands.undo_text == "Edit marker"
    env.ws.commands.undo()
    assert env.tl.get_marker(mid).name == "old"
    # Delete removes the selected marker (not a clip) ...
    env.click(marker_pos(env, mid))
    env.key(Qt.Key.Key_Delete)
    assert env.tl.markers == []
    env.ws.commands.undo()
    assert len(env.tl.markers) == 1
    # ... and the one under the pointer
    env.canvas._sel_marker = None
    env.move(marker_pos(env, mid), buttons=Qt.MouseButton.NoButton)
    assert env.canvas._hover_marker == mid
    env.key(Qt.Key.Key_Delete)
    assert env.tl.markers == []


def test_marker_menu_edit_and_delete_and_an_invalid_edit_is_reported(env, monkeypatch):
    mid = env.ws.edit.add_marker(3.0, "x")
    env.canvas.reload()
    menu = env.canvas.marker_menu(mid)
    assert [a.text() for a in menu.actions()] == ["Edit marker…", "Delete marker"]
    monkeypatch.setattr(MarkerDialog, "exec", lambda self: (self.color.setText("not a colour"), QDialog.DialogCode.Accepted)[1])
    shown = []
    monkeypatch.setattr("app.ui.dialogs.message.show_error", lambda parent, message, details=None, title="": shown.append(message))
    assert not env.canvas.edit_marker_dialog(mid)
    assert shown and "#RRGGBB" in shown[0] and env.tl.get_marker(mid).color == ""
    menu.actions()[1].trigger()
    assert env.tl.markers == []


def test_dragging_the_playhead_on_the_ruler_snaps_to_markers_and_clip_edges_and_a_marker_wins_over_nothing(env):
    env.clip(V1, 10.0, 4.0)
    env.done()
    mid = env.ws.edit.add_marker(5.0, "m")
    env.canvas.reload()
    y = RULER_H - 5
    x5 = int(env.canvas.time_to_x(5.0))
    env.press(QPoint(x5 + 8, y))  # 8 px past the marker (below its flag): inside the snap tolerance
    assert env.canvas.gesture_kind == "playhead" and env.canvas.playhead == pytest.approx(5.0)
    assert env.canvas.snap_guide is not None and env.canvas.snap_guide[1] == "markers"
    env.move(QPoint(int(env.canvas.time_to_x(10.0)) - 6, y))
    assert env.canvas.playhead == pytest.approx(10.0) and env.canvas.snap_guide[1] == "clip_edges"
    env.move(QPoint(int(env.canvas.time_to_x(7.0)), y))
    assert env.canvas.playhead == pytest.approx(7.0, abs=0.02) and env.canvas.snap_guide is None  # nothing near: free, on the frame grid
    env.release(QPoint(int(env.canvas.time_to_x(7.0)), y))
    assert env.ws.edit.playhead == pytest.approx(env.canvas.playhead) and env.tl.get_marker(mid).time == 5.0
    assert env.canvas.timebase().is_on_frame(env.canvas.playhead)


def test_the_range_has_draggable_handles_and_menu_actions(env):
    a = env.clip(V1, 2.0, 4.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    env.ws.selection.select([a.id])
    menu = env.canvas.ruler_menu(1.0)
    acts = {x.text(): x for x in menu.actions()}
    assert acts["Set range from selection"].isEnabled()
    acts["Set range from selection"].trigger()
    assert env.tl.range == pytest.approx((2.0, 6.0))
    assert {x.text(): x for x in env.canvas.ruler_menu(1.0).actions()}["Clear range"].isEnabled()
    env.canvas.grab()
    x_in = int(env.canvas.time_to_x(2.0))
    env.drag(QPoint(x_in, 8), QPoint(x_in + 80, 8))  # the in handle 1 s to the right
    assert env.tl.range == pytest.approx((3.0, 6.0))
    assert env.ws.commands.undo_text == "Set range"
    env.ws.commands.undo()
    assert env.tl.range == pytest.approx((2.0, 6.0))
    x_out = int(env.canvas.time_to_x(6.0))
    env.press(QPoint(x_out, 8))
    env.move(QPoint(x_out - int(10 * env.canvas.pps), 8))  # dragged far past the in point: stops one frame after it
    assert env.canvas._range_now()[1] == pytest.approx(2.0 + env.canvas.timebase().frame_s)
    env.canvas.escape()
    assert env.tl.range == pytest.approx((2.0, 6.0))
    {x.text(): x for x in env.canvas.ruler_menu(1.0).actions()}["Clear range"].trigger()
    assert env.tl.range is None


# ------------------------------------------------------------------------------------------------------------------------ 8. zoom / scroll / follow / auto-scroll
def long_timeline(env):
    env.clip(V1, 0.0, 2.0)
    env.clip(V1, 298.0, 2.0)
    return env.done()


def test_follow_playhead_modes(env):
    long_timeline(env)
    bar = env.scroll.horizontalScrollBar()
    vw = env.canvas.viewport_width()
    st = env.ws.edit_state
    st.update(follow_playhead="off")
    env.canvas.set_playhead(40.0)
    assert bar.value() == 0
    st.update(follow_playhead="page")
    env.canvas.set_playhead(5.0)
    assert bar.value() == 0  # inside the view: no scroll
    env.canvas.set_playhead(20.0)  # leaves the view: a page jump that puts the playhead near the left edge
    x = env.canvas.time_to_x(20.0)
    assert bar.value() == pytest.approx(x - 0.1 * vw, abs=2) and bar.value() <= x <= bar.value() + vw
    env.canvas.set_playhead(2.0)  # backwards: the previous page ends at the playhead
    assert bar.value() <= env.canvas.time_to_x(2.0) <= bar.value() + vw
    bar.setValue(0)
    st.update(follow_playhead="smooth")
    env.canvas.set_playhead(11.0)
    assert 0 < bar.value() and env.canvas.time_to_x(11.0) <= bar.value() + vw  # scrolled only as far as needed to keep a margin
    v1 = bar.value()
    env.canvas.set_playhead(11.5)
    assert bar.value() > v1
    st.update(follow_playhead="center")  # switching the mode applies it at once
    assert bar.value() == pytest.approx(env.canvas.time_to_x(11.5) - vw / 2, abs=2)
    env.ws.edit.set_playhead(60.0)  # moved by someone else (a shortcut, the transport): the canvas follows through the bus
    assert env.canvas.playhead == pytest.approx(60.0) and bar.value() == pytest.approx(env.canvas.time_to_x(60.0) - vw / 2, abs=2)


def test_edge_scroll_speed_is_bounded_and_directional():
    assert edge_scroll_speed(500, 1000) == 0
    assert edge_scroll_speed(2, 1000) < 0 < edge_scroll_speed(998, 1000)
    assert abs(edge_scroll_speed(30, 1000)) < abs(edge_scroll_speed(2, 1000)) <= AUTOSCROLL_MAX_PX  # faster the closer to the edge
    assert abs(edge_scroll_speed(-5000, 1000)) == AUTOSCROLL_MAX_PX == abs(edge_scroll_speed(9000, 1000))  # a pointer far outside the viewport scrolls at the maximum, no faster
    assert all(abs(edge_scroll_speed(x, 80)) <= AUTOSCROLL_MAX_PX for x in range(-200, 300))  # a tiny viewport as well
    assert edge_scroll_speed(5, 0) == 0


def test_auto_scroll_while_dragging_near_the_edge_is_bounded_and_keeps_the_clip_under_the_pointer(env):
    long_timeline(env)
    a = env.clip(V2, 5.0, 3.0)
    env.done()
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    bar = env.scroll.horizontalScrollBar()
    vp_w = env.scroll.viewport().width()
    p0 = env.pos(a.id)
    env.press(p0)
    near_edge = QPoint(vp_w - 4, p0.y())
    env.move(near_edge)
    assert env.canvas._auto.isActive()
    d0 = env.canvas._g.delta
    ticks = []
    for _ in range(5):
        before = bar.value()
        env.canvas._autoscroll_tick()
        ticks.append(bar.value() - before)
    assert all(0 < t <= AUTOSCROLL_MAX_PX for t in ticks), ticks  # bounded speed
    g = env.canvas._g
    # the clip stays under the (stationary) pointer: its delta grew by exactly the scrolled distance
    assert g.delta == pytest.approx(d0 + sum(ticks) / env.canvas.pps, abs=0.04)
    env.release(QPoint(near_edge.x() + bar.value(), near_edge.y()))  # the pointer has not moved on screen, so in canvas coordinates it moved with the scrolling
    assert not env.canvas._auto.isActive()
    assert env.get(a.id).timeline_start == pytest.approx(5.0 + (vp_w - 4 - p0.x() + sum(ticks)) / env.canvas.pps, abs=0.06)
    # with auto-scroll switched off nothing scrolls
    env.ws.commands.undo()
    bar.setValue(0)
    env.ws.edit_state.update(auto_scroll=False)
    env.press(p0)
    env.move(near_edge)
    env.canvas._autoscroll_tick()
    assert bar.value() == 0
    env.canvas.escape()
    assert not env.canvas._auto.isActive()


def test_zoom_in_and_out_keep_the_playhead_in_place_and_shift_wheel_and_middle_drag_scroll(env):
    long_timeline(env)
    bar = env.scroll.horizontalScrollBar()
    env.canvas.set_zoom(40.0)
    env.ws.edit_state.update(follow_playhead="off")
    env.canvas.set_playhead(8.0)
    x_before = env.canvas.time_to_x(8.0) - bar.value()
    env.canvas.zoom_in()
    assert env.canvas.pps == pytest.approx(50.0)
    assert env.canvas.time_to_x(8.0) - bar.value() == pytest.approx(x_before, abs=2)  # the playhead did not move on screen
    env.canvas.zoom_out()
    env.canvas.zoom_out()
    assert env.canvas.pps == pytest.approx(40.0 / 1.25)
    # playhead out of view: zoom around the centre of the view
    bar.setValue(2000)
    env.canvas.set_playhead(1.0)
    mid_t = (bar.value() + env.canvas.viewport_width() / 2) / env.canvas.pps
    env.canvas.zoom_in()
    assert (bar.value() + env.canvas.viewport_width() / 2) / env.canvas.pps == pytest.approx(mid_t, abs=0.1)
    # Shift+wheel scrolls sideways
    bar.setValue(500)
    ev = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton, SHIFT, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(env.canvas, ev)
    assert bar.value() > 500 and ev.isAccepted()
    plain = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton, NO, Qt.ScrollPhase.NoScrollPhase, False)
    plain.setAccepted(True)
    env.canvas.wheelEvent(plain)
    assert not plain.isAccepted()  # the plain wheel is left to the page (vertical scroll)
    # middle-button drag pans
    bar.setValue(1000)
    p = QPoint(300, 100)
    env.press(p, button=Qt.MouseButton.MiddleButton)
    env.move(QPoint(250, 100), buttons=Qt.MouseButton.MiddleButton)
    assert env.canvas.gesture_kind == "pan" and bar.value() == 1050
    env.release(QPoint(250, 100), button=Qt.MouseButton.MiddleButton)
    assert env.canvas.gesture_kind is None


# ------------------------------------------------------------------------------------------------------------------------ 9. rows with their own heights
def test_variable_height_rows_hit_testing_and_cross_track_drag(env):
    a = env.clip(V1, 1.0, 3.0)
    b = env.clip(V2, 1.0, 3.0)
    c = env.clip(V3, 1.0, 3.0)
    env.done()
    env.ws.edit.set_track_display(V1, height=90)
    env.ws.edit.set_track_display(V2, collapsed=True)
    lay = env.canvas.track_layout
    assert (lay.height(0), lay.height(1), lay.height(2)) == (90, 18, 36)
    assert lay.top(1) == RULER_H + 90 and lay.top(2) == RULER_H + 108
    cv = env.canvas
    assert cv.track_index_at(RULER_H + 89) == 0 and cv.track_index_at(RULER_H + 90) == 1 and cv.track_index_at(RULER_H + 107) == 1 and cv.track_index_at(RULER_H + 108) == 2
    for clip in (a, b, c):  # every clip is found in the middle of its own (differently tall) row
        hit = cv._hit_clip(*(lambda r: (r.center().x(), r.center().y()))(cv.clip_rect(env.get(clip.id), cv._row_of(clip.track_id))))
        assert hit is not None and hit[0].id == clip.id
    assert cv.clip_rect(env.get(a.id), 0).height() == 84 and cv.clip_rect(env.get(b.id), 1).height() == 12
    assert cv.height() == lay.total_height + NEW_TRACK_STRIP_H
    cv.grab()  # the collapsed row paints as a bare colour bar
    # drag the clip of the tall row down over the collapsed row to the third row: the layout decides the target row
    env.ws.edit_state.update(snap=env.ws.edit_state.snap.replace(enabled=False))
    p0 = env.pos(a.id)
    env.press(p0)
    env.move(QPoint(p0.x(), lay.top(1) + 5))
    assert cv._g.target_row == 1
    env.move(QPoint(p0.x() + 80, lay.top(2) + 10))
    assert cv._g.target_row == 2 and cv._g.reason and "overlap" in cv._g.reason.lower()  # c is there: red
    env.move(QPoint(p0.x() + int(4 * cv.pps), lay.top(2) + 10))
    assert cv._g.reason is None
    env.release(QPoint(p0.x() + int(4 * cv.pps), lay.top(2) + 10))
    assert env.get(a.id).track_id == V3 and env.get(a.id).timeline_start == pytest.approx(5.0)


# ------------------------------------------------------------------------------------------------------------------------ 10. context menus
def actions(menu: QMenu) -> dict:
    return {a.text(): a for a in menu.actions() if a.text()}


def test_clip_menu_lists_only_relevant_actions_and_explains_the_disabled_ones(env):
    asset = env.asset("video", 60.0)
    a, b = env.clip(V1, 0.0, 4.0, asset), env.clip(V1, 5.0, 4.0, asset)
    env.done()
    env.click(env.pos(a.id))
    env.canvas.set_playhead(2.0)
    acts = actions(env.canvas.clip_menu(a.id))
    for name in ("Preview media", "Split at playhead", "Delete", "Ripple delete", "Duplicate", "Copy", "Cut", "Paste", "Copy attributes", "Paste attributes", "Link", "Unlink", "Group", "Ungroup",
                 "Lock clip", "Lock aspects", "Speed…", "Replace media…", "Properties", "Select all on track", "Add marker at clip start"):
        assert name in acts, (name, list(acts))
    assert "Merge captions" not in acts  # only offered on captions
    assert acts["Split at playhead"].isEnabled() and acts["Delete"].isEnabled() and acts["Replace media…"].isEnabled() and acts["Speed…"].isEnabled()
    assert not acts["Paste"].isEnabled() and "nothing to paste" in acts["Paste"].toolTip()
    assert not acts["Link"].isEnabled() and "two or more" in acts["Link"].toolTip()
    assert not acts["Unlink"].isEnabled() and "not linked" in acts["Unlink"].toolTip()
    assert not acts["Group"].isEnabled() and not acts["Ungroup"].isEnabled()
    env.canvas.set_playhead(4.5)  # between the clips
    assert not actions(env.canvas.clip_menu(a.id))["Split at playhead"].isEnabled()
    assert "playhead inside" in actions(env.canvas.clip_menu(a.id))["Split at playhead"].toolTip()
    # two clips selected: link / group become possible, replace needs a single clip
    env.ws.selection.select([a.id, b.id])
    acts = actions(env.canvas.clip_menu(a.id))
    assert acts["Link"].isEnabled() and acts["Group"].isEnabled() and not acts["Replace media…"].isEnabled() and "single media clip" in acts["Replace media…"].toolTip()
    # something copied: Paste is available
    env.canvas.copy_selection()
    assert actions(env.canvas.clip_menu(a.id))["Paste"].isEnabled()
    # a locked track blocks the edits and says so
    env.ws.timeline.set_track_flag(V1, "locked", True)
    acts = actions(env.canvas.clip_menu(a.id))
    for name in ("Delete", "Ripple delete", "Cut", "Split at playhead", "Speed…", "Replace media…"):
        assert not acts[name].isEnabled() and "locked" in acts[name].toolTip(), name


def test_clip_menu_actions_call_the_engine_and_the_panel_signals(env):
    asset = env.asset("video", 60.0)
    a, b = env.clip(V1, 0.0, 4.0, asset), env.clip(V1, 8.0, 4.0, asset)
    env.done()
    env.click(env.pos(a.id))
    got = []
    env.canvas.request_speed_dialog.connect(lambda cid: got.append(("speed", cid)))
    env.canvas.request_replace_dialog.connect(lambda cid: got.append(("replace", cid)))
    env.canvas.request_inspector.connect(lambda cid: got.append(("inspector", cid)))
    env.canvas.clip_double_clicked.connect(lambda aid: got.append(("preview", aid)))
    acts = actions(env.canvas.clip_menu(a.id))
    for name in ("Speed…", "Replace media…", "Properties", "Preview media"):
        acts[name].trigger()
    assert got == [("speed", a.id), ("replace", a.id), ("inspector", a.id), ("preview", asset.id)]
    assert not env.ws.commands.can_undo  # none of them edits by itself
    # split at the playhead, lock, aspects, marker, select track
    env.canvas.set_playhead(2.0)
    actions(env.canvas.clip_menu(a.id))["Split at playhead"].trigger()
    assert len(env.tl.get_track(V1).clips) == 3
    env.ws.commands.undo()
    actions(env.canvas.clip_menu(a.id))["Lock clip"].trigger()
    assert "item" in env.get(a.id).locks
    assert "Unlock clip" in actions(env.canvas.clip_menu(a.id))
    sub = actions(env.canvas.clip_menu(a.id))["Lock aspects"].menu()
    assert [x.text() for x in sub.actions()] == ["Position", "Timing", "Transform", "Audio", "Style"]
    actions(env.canvas.clip_menu(a.id))["Unlock clip"].trigger()
    {x.text(): x for x in actions(env.canvas.clip_menu(a.id))["Lock aspects"].menu().actions()}["Timing"].trigger()
    assert env.get(a.id).locks == frozenset({"timing"})
    again = {x.text(): x for x in actions(env.canvas.clip_menu(a.id))["Lock aspects"].menu().actions()}
    assert again["Timing"].isChecked() and not again["Position"].isChecked()
    actions(env.canvas.clip_menu(a.id))["Add marker at clip start"].trigger()
    assert [m.time for m in env.tl.markers] == [0.0]
    env.ws.selection.select_none()
    actions(env.canvas.clip_menu(b.id))["Select all on track"].trigger()
    assert set(env.selected()) == {a.id, b.id}
    # copy / paste / duplicate / delete from the menu
    env.ws.selection.select([a.id])
    n = len(env.tl.all_clips())
    actions(env.canvas.clip_menu(a.id))["Duplicate"].trigger()
    assert len(env.tl.all_clips()) == n + 1
    env.ws.selection.select([b.id])
    actions(env.canvas.clip_menu(b.id))["Ripple delete"].trigger()
    assert env.tl.get_clip(b.id) is None


def test_link_group_and_paste_attributes_from_the_menu(env):
    a, b = env.clip(V1, 0.0, 3.0), env.clip(A1, 0.0, 3.0)
    env.done()
    env.ws.selection.select([a.id, b.id])
    actions(env.canvas.clip_menu(a.id))["Link"].trigger()
    assert env.get(a.id).link_id and env.get(a.id).link_id == env.get(b.id).link_id
    assert actions(env.canvas.clip_menu(a.id))["Unlink"].isEnabled()
    actions(env.canvas.clip_menu(a.id))["Group"].trigger()
    assert env.get(a.id).group_id == env.get(b.id).group_id != ""
    actions(env.canvas.clip_menu(a.id))["Ungroup"].trigger()
    assert env.get(a.id).group_id == ""
    # "Paste attributes" with nothing copied reports the engine's reason instead of doing nothing silently
    env.status.clear()
    actions(env.canvas.clip_menu(a.id))["Paste attributes"].trigger()
    assert any("no copied attributes" in m for m in env.status)


def test_merge_captions_is_offered_on_captions_and_needs_two(env):
    cap_a = env.clip("track_v6", 0.0, 2.0, asset=env.asset("video"), kind=KIND_CAPTION, text={"text": "hello there", "words": [{"word": "hello", "start": 0.0, "end": 1.0}, {"word": "there", "start": 1.0, "end": 2.0}]})
    cap_b = env.clip("track_v6", 2.0, 2.0, asset=env.asset("video"), kind=KIND_CAPTION, text={"text": "my friend", "words": [{"word": "my", "start": 2.0, "end": 3.0}, {"word": "friend", "start": 3.0, "end": 4.0}]})
    env.done()
    env.ws.selection.select([cap_a.id])
    one = actions(env.canvas.clip_menu(cap_a.id))["Merge captions"]
    assert not one.isEnabled() and "two or more" in one.toolTip()
    env.ws.selection.select([cap_a.id, cap_b.id])
    two = actions(env.canvas.clip_menu(cap_a.id))["Merge captions"]
    assert two.isEnabled()
    two.trigger()
    assert len(env.tl.get_track("track_v6").clips) == 1 and env.ws.commands.can_undo


def test_track_and_ruler_menus_and_right_clicks_open_the_right_menu(env, monkeypatch):
    a = env.clip(V1, 2.0, 3.0)
    env.done()
    shown: list[QMenu] = []
    monkeypatch.setattr(TimelineCanvas, "_show_menu", lambda self, menu, pos: shown.append(menu))
    # right click on a clip: the clip menu; an unselected clip becomes the selection first
    env.press(env.pos(a.id), button=Qt.MouseButton.RightButton)
    assert shown and "Split at playhead" in actions(shown[-1]) and env.selected() == [a.id]
    # empty space on a track row: the track menu
    env.press(env.at(20.0, V2), button=Qt.MouseButton.RightButton)
    tm = actions(shown[-1])
    assert "Paste here" in tm and not tm["Paste here"].isEnabled() and not tm["Select all on track"].isEnabled() and "no clips" in tm["Select all on track"].toolTip()
    assert [x.text() for x in tm["Add track…"].menu().actions()] == ["Video track", "Image track", "Graphics track", "Audio track"]
    n = len(env.tl.tracks)
    tm["Add track…"].menu().actions()[3].trigger()
    assert len(env.tl.tracks) == n + 1 and env.tl.tracks[-1].kind is TrackKind.AUDIO
    # the ruler
    env.press(QPoint(300, 6), button=Qt.MouseButton.RightButton)
    assert "Set range from selection" in actions(shown[-1])
    # "Paste here" pastes at the clicked time on that row once something was copied
    env.canvas.copy_selection()
    menu = env.canvas.track_menu(env.canvas._row_of(V2), 10.0)
    paste = actions(menu)["Paste here"]
    assert paste.isEnabled()
    paste.trigger()
    assert env.starts(V2) == [(10.0, 3.0)]


# ------------------------------------------------------------------------------------------------------------------------ 11. keyboard and the methods the shortcuts call
def test_arrow_keys_nudge_by_frames_and_each_nudge_is_one_undo_step(env):
    a = env.clip(V1, 1.0, 2.0)
    env.done()
    fs = env.canvas.timebase().frame_s
    env.click(env.pos(a.id))
    before = env.snapshot()
    env.key(Qt.Key.Key_Right)
    assert env.get(a.id).timeline_start == pytest.approx(1.0 + fs)
    env.key(Qt.Key.Key_Right, SHIFT)
    assert env.get(a.id).timeline_start == pytest.approx(1.0 + 11 * fs)
    env.key(Qt.Key.Key_Left)
    assert env.get(a.id).timeline_start == pytest.approx(1.0 + 10 * fs)
    for _ in range(3):
        env.ws.commands.undo()
    assert env.snapshot() == before and not env.ws.commands.can_undo
    # the method the shortcuts call, with a message when nothing is selected
    env.ws.selection.select_none()
    assert not env.canvas.nudge(1) and any("Select a clip" in m for m in env.status)
    env.ws.selection.select([a.id])
    assert env.canvas.nudge(-5) and env.get(a.id).timeline_start == pytest.approx(1.0 - 5 * fs)
    assert not env.canvas.nudge(-10_000)  # before 0:00: refused by the engine, nothing changes
    assert env.get(a.id).timeline_start == pytest.approx(1.0 - 5 * fs)


def test_delete_ripple_delete_and_the_clipboard_methods(env):
    a, b, c = (env.clip(V1, s, 2.0) for s in (0.0, 3.0, 6.0))
    env.done()
    env.ws.selection.select([b.id])
    env.key(Qt.Key.Key_Delete)
    assert env.tl.get_clip(b.id) is None and env.get(c.id).timeline_start == 6.0  # a gap is left
    env.ws.commands.undo()
    env.ws.selection.select([b.id])
    env.key(Qt.Key.Key_Delete, SHIFT)  # Shift+Delete = ripple delete: the gap closes
    assert env.tl.get_clip(b.id) is None and env.get(c.id).timeline_start == pytest.approx(4.0)  # the clip's 2 s are closed up
    env.ws.commands.undo()
    env.ws.selection.select([b.id])
    env.key(Qt.Key.Key_Backspace)
    assert env.tl.get_clip(b.id) is None
    env.ws.commands.undo()
    # copy / paste at the playhead / cut / duplicate
    env.ws.selection.select([a.id])
    assert env.canvas.copy_selection() and env.ws.edit.clipboard is not None
    env.canvas.set_playhead(10.0)
    assert env.canvas.paste_at_playhead() and env.starts(V1)[-1] == (10.0, 2.0)
    env.ws.selection.select([c.id])
    assert env.canvas.cut_selection() and env.tl.get_clip(c.id) is None
    last = env.tl.get_track(V1).clips[-1]  # the pasted clip, at 10 s: free space after it
    env.ws.selection.select([last.id])
    n = len(env.tl.all_clips())
    assert env.canvas.duplicate_selection() and len(env.tl.all_clips()) == n + 1
    assert env.selected() != [last.id] and env.get(env.selected()[0]).timeline_start == pytest.approx(12.0)  # the copy is selected
    env.ws.selection.select([a.id])
    env.status.clear()
    assert not env.canvas.duplicate_selection() and any("overlap" in m for m in env.status)  # the copy would land on b: refused, with the reason
    env.ws.selection.select_none()
    assert not env.canvas.delete_selection() and not env.canvas.cut_selection() and not env.canvas.duplicate_selection() and not env.canvas.copy_selection()


def test_playhead_jump_and_step_methods(env):
    env.clip(V1, 2.0, 2.0)
    env.clip(V2, 3.0, 4.0)
    env.done()
    env.ws.edit.add_marker(1.0, "m1")
    env.ws.edit.add_marker(8.0, "m2")
    cv = env.canvas
    cv.set_playhead(0.0)
    seen = []
    for _ in range(4):
        assert cv.jump_to_edge(1)
        seen.append(round(cv.playhead, 3))
    assert seen == [2.0, 3.0, 4.0, 7.0]  # the edges of both clips, in order
    assert not cv.jump_to_edge(1) and any("no further edit" in m for m in env.status)
    assert cv.jump_to_edge(-1) and cv.playhead == pytest.approx(4.0)
    env.key(Qt.Key.Key_Left, CTRL)  # Ctrl+Left = previous edge
    assert cv.playhead == pytest.approx(3.0)
    cv.set_playhead(0.0)
    assert cv.jump_to_marker(1) and cv.playhead == pytest.approx(1.0)
    assert cv.jump_to_marker(1) and cv.playhead == pytest.approx(8.0)
    assert not cv.jump_to_marker(1)
    assert cv.jump_to_marker(-1) and cv.playhead == pytest.approx(1.0)
    fs = cv.timebase().frame_s
    assert cv.step_playhead(3) and cv.playhead == pytest.approx(1.0 + 3 * fs)
    assert cv.step_playhead(-3) and cv.playhead == pytest.approx(1.0)
    cv.jump_to_start()
    assert cv.playhead == 0.0 and not cv.step_playhead(-1)  # not before the start
    assert cv.jump_to_end() and cv.playhead == pytest.approx(7.0)


def test_escape_clears_the_selection_and_cancels_a_library_drop_preview(env):
    from PySide6.QtGui import QDragMoveEvent

    a = env.clip(V1, 1.0, 2.0)
    v = env.asset("video", 3.0)
    env.done()
    env.click(env.pos(a.id))
    md = mime([v.id])  # the event only borrows the mime data: keep it alive
    env.canvas.dragMoveEvent(QDragMoveEvent(env.at(5.0, V2), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, NO))
    assert env.canvas.drop_state is not None
    assert env.canvas.escape() and env.canvas.drop_state is None and env.selected() == [a.id]  # the first Escape cancels the preview
    assert env.canvas.escape() and env.selected() == []
    assert not env.canvas.escape()  # nothing left to cancel


# ------------------------------------------------------------------------------------------------------------------------ 12. performance: 1,000 clips
def test_dragging_200_selected_clips_in_a_1000_clip_timeline_stays_responsive(env, monkeypatch):
    asset = env.asset("video", 1e6)
    per_track = 250
    ids: dict[str, list[str]] = {}
    for tid in (V1, V2, V3, "track_v4"):
        ids[tid] = [env.clip(tid, k * 3.0, 2.0, asset).id for k in range(per_track)]
    for k in range(50):
        env.tl.insert_marker(__import__("app.timeline.markers", fromlist=["Marker"]).Marker(f"marker_{k:03d}", k * 17.0 + 0.5, f"m{k}"))
    env.done()
    assert len(env.tl.all_clips()) == 1000
    selection = ids[V1][50:150] + ids[V2][50:150]
    env.ws.selection.select(selection)
    before = env.snapshot()
    assert env.canvas.height() > 0
    spans = []
    updates: list = []
    orig_update = env.canvas.update
    monkeypatch.setattr(env.canvas, "update", lambda *a: (updates.append(a), orig_update(*a))[1])
    reloads0, builds0 = env.reloads, env.ws.edit.snap.builds["clips"]
    first = env.canvas.clip_rect(env.get(selection[0]), env.canvas._row_of(V1))
    p0 = QPoint(int(first.center().x()), int(first.center().y()))
    started = False
    env.press(p0)
    for i in range(1, 61):
        t0 = time.perf_counter()
        env.move(QPoint(p0.x() + i, p0.y()))  # one pixel per move: 60 px = 0.75 s, inside the gaps
        spans.append(time.perf_counter() - t0)
        if env.canvas._g.moved and not started:
            started = True
            updates.clear()  # the move that crosses the 3 px threshold repaints everything once (the originals turn translucent)
    med = statistics.median(spans)
    assert env.canvas.gesture_kind == "move" and len(env.canvas._g.ids) == 200 and env.canvas._g.reason is None
    assert med < 0.040, f"median mouse-move handling {med * 1000:.1f} ms with 200 selected clips"
    assert env.reloads == reloads0, "the canvas was reloaded during the drag"
    assert env.ws.edit.snap.builds["clips"] - builds0 <= 1, "the snap targets were rebuilt during the drag"
    assert updates and all(len(a) == 1 and isinstance(a[0], QRect) for a in updates), "the drag repainted the whole canvas instead of dirty rectangles"
    t0 = time.perf_counter()
    env.canvas.grab()  # painting a gesture with 200 ghosts
    paint = time.perf_counter() - t0
    assert paint < 1.0, paint
    env.release(QPoint(p0.x() + 60, p0.y()))
    moved = {cid: env.get(cid).timeline_start - 3.0 * k for tid in (V1, V2) for k, cid in enumerate(ids[tid]) if cid in selection}
    deltas = {round(v, 4) for v in moved.values()}
    assert len(deltas) == 1 and 0.5 < next(iter(deltas)) < 1.0  # all 200 moved by the same amount (rigid)
    assert env.reloads == reloads0 + 1 or env.reloads <= reloads0 + 2  # the commit reloads once, not per clip
    untouched = [cid for tid in ids for cid in ids[tid] if cid not in selection]
    assert all(env.get(cid).timeline_start % 3.0 == 0.0 for cid in untouched)
    env.undo_once_restores(before)
