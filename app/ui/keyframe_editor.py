"""Keyframe editor for the inspector: a painted lane for one animatable property of the selected clip plus the controls around it.

The lane only DISPLAYS and gathers gestures; every change is one call on ``ws.edit`` (add / move / delete / interpolation / copy / paste), committed once when the gesture ends (a
drag previews in memory and moves the keyframes on release = one undo step). Which properties exist comes from the engine (``keyframe_properties``); the value limits are the engine's
``KEYFRAME_RANGES``; the interpolation names are exactly the ones the renderer implements (``keyframes.INTERPOLATIONS``). Times are seconds from the start of the clip.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.exceptions import AppError
from app.timeline.clip import Clip
from app.timeline.keyframes import DEFAULTS, INTERPOLATIONS, Keyframe, value_at
from app.timeline.ops_props import KEYFRAME_RANGES, KF_MATCH_EPS, keyframe_properties
from app.timeline.timebase import Timebase, timebase_of
from app.timeline.track import Track
from app.ui.context import UiContext
from app.ui.theme import palette
from app.ui.timecode_edit import NumberEdit, TimecodeEdit

PROP_LABELS = {"position_x": "Position X", "position_y": "Position Y", "scale": "Scale", "rotation": "Rotation", "opacity": "Opacity", "volume": "Volume", "blur": "Blur", "reveal": "Reveal"}
PROP_HINT = {"position_x": "added to the clip position", "position_y": "added to the clip position", "scale": "multiplies the clip scale", "rotation": "added to the clip rotation",
             "opacity": "multiplies the clip opacity", "volume": "the clip gain over time", "blur": "rendered as ONE constant blur (the largest value) - it is not animated",
             "reveal": "wipe-style reveal, 1 = fully shown"}
# display unit of the value: (factor model -> display, suffix, decimals, step)
PROP_UNITS = {"position_x": (1.0, " px", 1, 5.0), "position_y": (1.0, " px", 1, 5.0), "scale": (1.0, "×", 3, 0.05), "rotation": (1.0, "°", 1, 1.0), "opacity": (100.0, " %", 1, 5.0),
              "volume": (100.0, " %", 0, 5.0), "blur": (1.0, "", 1, 1.0), "reveal": (100.0, " %", 0, 5.0)}
MIN_SPAN = {"position_x": 400.0, "position_y": 400.0, "scale": 1.0, "rotation": 90.0, "opacity": 1.0, "volume": 2.0, "blur": 20.0, "reveal": 1.0}
PAD_L, PAD_R, PAD_T, PAD_B = 46, 12, 12, 20
HIT = 8  # px
DRAG_START = 3  # px before a press on empty space turns into a rubber band / a press on a keyframe into a move
CURVE_STEP = 3  # px between the samples of the drawn curve


def has_keyframes(clip: Clip, prop: str) -> int:
    return sum(1 for k in clip.keyframes if k.property == prop)


class KeyframeLane(QWidget):
    """The painted lane. Signals carry the user's intent; the owner decides (and asks the engine)."""

    selection_changed = Signal()
    add_requested = Signal(float, float)  # time, value
    move_requested = Signal(list)  # [(property, old time, new time, new value)]
    delete_requested = Signal()
    copy_requested = Signal()
    paste_requested = Signal()
    interpolation_requested = Signal(str)
    add_at_playhead_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("keyframe_lane")
        self.setMinimumHeight(132)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)
        self.setToolTip("Click empty space to add a keyframe, drag a diamond to move it, Shift/Ctrl+click or drag a box to select several, Delete removes, right-click for more. "
                        "Hold Alt to turn snapping off.")
        self.theme = "dark"
        self._prop = "opacity"
        self._duration = 1.0
        self._tb: Timebase = Timebase(30)
        self._kfs: list[Keyframe] = []
        self._sel: set[float] = set()
        self._playhead: float | None = None
        self._view: tuple[float, float] = (0.0, 1.0)
        self._drag: dict | None = None
        self._live: dict[float, tuple[float, float]] | None = None  # original time -> (time, value) while a drag previews
        self._band: QRectF | None = None
        self.read_only = False  # locked clip / aspect: the lane still shows and selects, but gestures do not edit
        self.can_paste: Callable[[], bool] = lambda: True
        self.playhead_inside: Callable[[], bool] = lambda: self._playhead is not None

    # ------------------------------------------------------------------ data
    def set_data(self, prop: str, duration: float, keyframes: list[Keyframe], timebase: Timebase, playhead: float | None, theme: str = "dark") -> None:
        if self._drag is not None:  # never swap the data under a gesture
            return
        changed_prop = prop != self._prop
        self._prop, self._duration, self._tb, self.theme = prop, max(1e-6, float(duration)), timebase, theme
        self._kfs = sorted((Keyframe(k.property, k.time, k.value, k.interpolation, k.decision_id) for k in keyframes if k.property == prop), key=lambda k: k.time)
        self._sel = set() if changed_prop else {k.time for k in self._kfs if any(abs(k.time - s) <= KF_MATCH_EPS for s in self._sel)}
        self._playhead = playhead
        self._view = self._fit_view()
        self.update()

    def set_playhead(self, t: float | None) -> None:
        if t != self._playhead:
            self._playhead = t
            self.update()

    def property_name(self) -> str:
        return self._prop

    def keyframes(self) -> list[Keyframe]:
        return list(self._kfs)

    def selected(self) -> list[Keyframe]:
        return [k for k in self._kfs if any(abs(k.time - s) <= KF_MATCH_EPS for s in self._sel)]

    def selection(self) -> list[tuple[str, float]]:
        return [(self._prop, k.time) for k in self.selected()]

    def select(self, times: list[float]) -> None:
        self._sel = set(times)
        self.update()
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        if self._sel:
            self.select([])

    def is_dragging(self) -> bool:
        return self._drag is not None

    # ------------------------------------------------------------------ geometry
    def _fit_view(self) -> tuple[float, float]:
        """The value range the lane shows: the data (plus the neutral value) with some room, never smaller than a sensible span, inside the engine's limits."""
        lo_e, hi_e = KEYFRAME_RANGES[self._prop]
        vals = [k.value for k in self._kfs] + [DEFAULTS[self._prop]]
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, MIN_SPAN[self._prop])
        mid = (hi + lo) / 2
        lo, hi = mid - span * 0.6, mid + span * 0.6
        if hi_e - lo_e <= span * 1.2:  # small ranges (opacity 0..1): always the whole range
            return lo_e, hi_e
        if lo < lo_e:
            lo, hi = lo_e, lo_e + (hi - lo)
        if hi > hi_e:
            lo, hi = hi_e - (hi - lo), hi_e
        return lo, hi

    def plot(self) -> QRectF:
        return QRectF(PAD_L, PAD_T, max(10.0, self.width() - PAD_L - PAD_R), max(10.0, self.height() - PAD_T - PAD_B))

    def x_of(self, t: float) -> float:
        r = self.plot()
        return r.left() + (t / self._duration) * r.width()

    def t_of(self, x: float) -> float:
        r = self.plot()
        return (x - r.left()) / r.width() * self._duration

    def y_of(self, v: float) -> float:
        r = self.plot()
        lo, hi = self._view
        return r.bottom() - (v - lo) / max(1e-9, hi - lo) * r.height()

    def v_of(self, y: float) -> float:
        r = self.plot()
        lo, hi = self._view
        return lo + (r.bottom() - y) / r.height() * (hi - lo)

    def pos_for(self, t: float, v: float) -> QPoint:
        """Widget position of a (time, value) pair - what a mouse gesture on the lane uses."""
        return QPoint(int(round(self.x_of(t))), int(round(self.y_of(v))))

    def keyframe_pos(self, t: float) -> QPoint:
        k = min(self._kfs, key=lambda k: abs(k.time - t))
        return self.pos_for(k.time, k.value)

    def _current(self, k: Keyframe) -> tuple[float, float]:
        if self._live is not None and k.time in self._live:
            return self._live[k.time]
        return k.time, k.value

    def _hit(self, pos: QPointF) -> Keyframe | None:
        best, best_d = None, HIT + 1.0
        for k in self._kfs:
            t, v = self._current(k)
            d = max(abs(self.x_of(t) - pos.x()), abs(self.y_of(v) - pos.y()))
            if d <= HIT and d < best_d:
                best, best_d = k, d
        return best

    # ------------------------------------------------------------------ snapping
    def snap_time(self, t: float, alt: bool = False) -> float:
        t = min(max(t, 0.0), self._duration)
        if alt:
            return t
        if self._playhead is not None and abs(self.x_of(self._playhead) - self.x_of(t)) <= 6.0 and 0.0 <= self._playhead <= self._duration:
            return self._playhead
        return min(max(self._tb.snap(t), 0.0), self._duration)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:
        c = palette(self.theme)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(c["panel2"]))
        r = self.plot()
        p.setPen(QPen(QColor(c["border"]), 1))
        p.drawRect(r)
        lo, hi = self._view
        font = p.font()
        font.setPixelSize(10)
        p.setFont(font)
        for i in range(5):  # value grid
            v = lo + (hi - lo) * i / 4
            y = self.y_of(v)
            p.setPen(QPen(QColor(c["border"]), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(c["muted"]))
            p.drawText(QRectF(0, y - 8, PAD_L - 4, 16), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), self._fmt(v))
        neutral = DEFAULTS[self._prop]
        if lo <= neutral <= hi:
            p.setPen(QPen(QColor(c["muted"]), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(r.left(), self.y_of(neutral)), QPointF(r.right(), self.y_of(neutral)))
        p.setPen(QColor(c["muted"]))
        p.drawText(QRectF(r.left(), r.bottom() + 2, 120, 16), int(Qt.AlignmentFlag.AlignLeft), self._tb.format(0.0))
        p.drawText(QRectF(r.right() - 120, r.bottom() + 2, 120, 16), int(Qt.AlignmentFlag.AlignRight), self._tb.format(self._duration))
        pts = sorted(((k.time, k.value, k.interpolation) for k in self._kfs), key=lambda x: x[0]) if self._live is None else \
            sorted((*self._current(k), k.interpolation) for k in self._kfs)
        shown = [Keyframe(self._prop, t, v, i) for t, v, i in pts]
        if shown:  # the curve exactly as the renderer computes it (holds the first / last value outside the keyframes)
            p.setPen(QPen(QColor(c["accent"]), 2))
            path = QPolygonF()
            x = r.left()
            while x <= r.right() + 0.5:
                path.append(QPointF(x, self.y_of(value_at(shown, self._prop, self.t_of(x)))))
                x += CURVE_STEP
            p.drawPolyline(path)
        else:
            p.setPen(QColor(c["muted"]))
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), "No keyframes: the value is constant. Click here to add one.")
        if self._playhead is not None and 0.0 <= self._playhead <= self._duration:
            p.setPen(QPen(QColor(c["danger"]), 1))
            p.drawLine(QPointF(self.x_of(self._playhead), r.top()), QPointF(self.x_of(self._playhead), r.bottom()))
        sel = {s for s in self._sel}
        lo_e, hi_e = KEYFRAME_RANGES[self._prop]
        for k in self._kfs:
            t, v = self._current(k)
            cx, cy = self.x_of(t), self.y_of(v)
            is_sel = any(abs(k.time - s) <= KF_MATCH_EPS for s in sel)
            bad = not (lo_e - 1e-9 <= v <= hi_e + 1e-9)
            d = QPolygonF([QPointF(cx, cy - 6), QPointF(cx + 6, cy), QPointF(cx, cy + 6), QPointF(cx - 6, cy)])
            p.setPen(QPen(QColor(c["danger"] if bad else c["selection"] if is_sel else c["text"]), 2 if is_sel else 1))
            p.setBrush(QColor(c["selection"] if is_sel else c["accent"]))
            p.drawPolygon(d)
            if is_sel:
                p.setPen(QColor(c["text"]))
                p.drawText(QPointF(cx + 8, cy - 8), self._fmt(v))
        if self._band is not None:
            p.setPen(QPen(QColor(c["accent"]), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(74, 163, 255, 40))
            p.drawRect(self._band)
        p.end()

    def _fmt(self, v: float) -> str:
        f, suffix, dec, _step = PROP_UNITS[self._prop]
        return f"{v * f:.{min(dec, 2)}f}{suffix.strip()}"

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus()
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self.read_only:
            hit = self._hit(e.position())
            self.select([hit.time] if hit is not None else [])
            return
        mods = e.modifiers()
        extend = bool(mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
        hit = self._hit(e.position())
        if hit is not None:
            was_sel = any(abs(hit.time - s) <= KF_MATCH_EPS for s in self._sel)
            if extend:
                self._sel = {s for s in self._sel if abs(s - hit.time) > KF_MATCH_EPS} if was_sel else {*self._sel, hit.time}
                self.selection_changed.emit()
                self.update()
                self._drag = None
                return
            if not was_sel:
                self._sel = {hit.time}
                self.selection_changed.emit()
            movers = self.selected()
            self._drag = {"mode": "move", "press": e.position(), "anchor": hit, "orig": {k.time: (k.time, k.value) for k in movers}, "moved": False}
        else:
            self._drag = {"mode": "band", "press": e.position(), "extend": extend, "alt": bool(mods & Qt.KeyboardModifier.AltModifier), "moved": False,
                          "base": set(self._sel) if extend else set()}
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        d = self._drag
        if d is None:
            return
        press: QPointF = d["press"]
        pos = e.position()
        if not d["moved"] and max(abs(pos.x() - press.x()), abs(pos.y() - press.y())) < DRAG_START:
            return
        d["moved"] = True
        alt = bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
        if d["mode"] == "move":
            anchor: Keyframe = d["anchor"]
            at, av = d["orig"][anchor.time]
            new_t = self.snap_time(self.t_of(pos.x()), alt)
            new_v = self.v_of(pos.y())
            dt, dv = new_t - at, new_v - av
            live = {}
            for k in self.selected():
                if k.time in d["orig"]:
                    ot, ov = d["orig"][k.time]
                    live[k.time] = (min(max(ot + dt, 0.0), self._duration), ov + dv)
            self._live = live
        else:
            self._band = QRectF(press, pos).normalized()
            inside = [k.time for k in self._kfs if self._band.contains(QPointF(self.x_of(k.time), self.y_of(k.value)))]
            self._sel = set(d["base"]) | set(inside)
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        d, self._drag = self._drag, None
        live, self._live = self._live, None
        band, self._band = self._band, None
        self.update()
        if d is None or e.button() != Qt.MouseButton.LeftButton:
            return
        if d["mode"] == "move":
            if d["moved"] and live:
                moves = [(self._prop, k.time, live[k.time][0], live[k.time][1]) for k in self.selected() if k.time in live
                         and (abs(live[k.time][0] - k.time) > 1e-9 or abs(live[k.time][1] - k.value) > 1e-12)]
                if moves:
                    self.move_requested.emit(moves)
            return
        if band is not None and d["moved"]:
            self.selection_changed.emit()
            return
        if d["extend"]:
            return  # a modifier click on empty space only keeps the selection
        # a plain click on empty space: add a keyframe at that time / value
        r = self.plot()
        pos = e.position()
        if not r.contains(pos):
            self.clear_selection()
            return
        self.add_requested.emit(self.snap_time(self.t_of(pos.x()), d["alt"]), self.v_of(pos.y()))

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if not self.read_only:
                self.delete_requested.emit()
        elif e.key() == Qt.Key.Key_Escape:
            if self._drag is not None:
                self._drag, self._live, self._band = None, None, None
                self.update()
            else:
                self.clear_selection()
        elif e.key() == Qt.Key.Key_C and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.copy_requested.emit()
        elif e.key() == Qt.Key.Key_V and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if not self.read_only:
                self.paste_requested.emit()
        elif e.key() == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.select([k.time for k in self._kfs])
        else:
            super().keyPressEvent(e)

    # ------------------------------------------------------------------ context menu
    def build_menu(self, pos: QPoint) -> QMenu:
        """The right-click menu at ``pos`` (a keyframe under the cursor joins the selection first)."""
        hit = self._hit(QPointF(pos))
        if hit is not None and not any(abs(hit.time - s) <= KF_MATCH_EPS for s in self._sel):
            self.select([hit.time])
        sel = self.selected()
        menu = QMenu(self)
        sub = menu.addMenu("Interpolation")
        sub.setEnabled(bool(sel))
        if not sel:
            sub.setToolTip("Select keyframes first.")
        shared = {k.interpolation for k in sel}
        for name in INTERPOLATIONS:
            a = sub.addAction(name.replace("_", " "))
            a.setCheckable(True)
            a.setChecked(shared == {name})
            a.setData(name)
            a.triggered.connect(lambda _=False, n=name: self.interpolation_requested.emit(n))
        sub.setEnabled(bool(sel) and not self.read_only)
        copy = menu.addAction("Copy")
        copy.setEnabled(bool(sel))
        copy.triggered.connect(lambda _=False: self.copy_requested.emit())
        paste = menu.addAction("Paste at playhead")
        paste.setEnabled(self.can_paste() and not self.read_only)
        paste.triggered.connect(lambda _=False: self.paste_requested.emit())
        add = menu.addAction("Add keyframe at playhead")
        add.setEnabled(self.playhead_inside() and not self.read_only)
        add.triggered.connect(lambda _=False: self.add_at_playhead_requested.emit())
        delete = menu.addAction("Delete")
        delete.setEnabled(bool(sel) and not self.read_only)
        delete.triggered.connect(lambda _=False: self.delete_requested.emit())
        return menu

    def contextMenuEvent(self, e) -> None:
        self.build_menu(e.pos()).exec(e.globalPos())


class KeyframeEditor(QWidget):
    """Lane + controls for the keyframes of one clip. ``set_clip`` (re)loads it; ``message(text, is_error)`` reports results and the engine's refusals."""

    message = Signal(str, bool)

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._clip_id: str | None = None
        self._props: tuple[str, ...] = ()
        self.available = False
        self.lock_reason: Callable[[str], str] = lambda _prop: ""  # the owner says why a property cannot be edited ("" = it can)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        lay.addWidget(self.note)
        self.body = QWidget()
        b = QVBoxLayout(self.body)
        b.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.body)

        row = QHBoxLayout()
        row.addWidget(QLabel("Property"))
        self.prop_combo = QComboBox()
        self.prop_combo.setObjectName("kf_property")
        row.addWidget(self.prop_combo, 1)
        b.addLayout(row)
        self.prop_hint = QLabel()
        self.prop_hint.setWordWrap(True)
        self.prop_hint.setObjectName("muted")
        b.addWidget(self.prop_hint)

        self.lane = KeyframeLane()
        b.addWidget(self.lane)

        nav = QHBoxLayout()
        self.prev_btn, self.next_btn = QPushButton("◀ Previous"), QPushButton("Next ▶")
        self.add_btn = QPushButton("Add at playhead")
        for w in (self.prev_btn, self.next_btn, self.add_btn):
            nav.addWidget(w)
        b.addLayout(nav)
        ops = QHBoxLayout()
        self.copy_btn, self.paste_btn, self.delete_btn = QPushButton("Copy"), QPushButton("Paste at playhead"), QPushButton("Delete")
        for w in (self.copy_btn, self.paste_btn, self.delete_btn):
            ops.addWidget(w)
        b.addLayout(ops)

        form = QFormLayout()
        self.interp_combo = QComboBox()
        self.interp_combo.setObjectName("kf_interpolation")
        for name in INTERPOLATIONS:
            self.interp_combo.addItem(name.replace("_", " "), name)
        self.time_edit = TimecodeEdit()
        self.time_edit.setObjectName("kf_time")
        self.value_edit = NumberEdit(0.0, 1.0, label="Value")
        self.value_edit.setObjectName("kf_value")
        form.addRow("Interpolation", self.interp_combo)
        form.addRow("Time", self.time_edit)
        form.addRow("Value", self.value_edit)
        b.addLayout(form)

        self.prop_combo.activated.connect(lambda _i: self._property_changed())
        self.lane.selection_changed.connect(self._sync_fields)
        self.lane.add_requested.connect(self._add)
        self.lane.move_requested.connect(self._move)
        self.lane.delete_requested.connect(self.delete_selected)
        self.lane.copy_requested.connect(self.copy_selected)
        self.lane.paste_requested.connect(self.paste_at_playhead)
        self.lane.interpolation_requested.connect(self.set_interpolation)
        self.lane.add_at_playhead_requested.connect(self.add_at_playhead)
        self.lane.can_paste = self._can_paste
        self.lane.playhead_inside = lambda: self._local_playhead() is not None
        self.interp_combo.activated.connect(lambda _i: self.set_interpolation(self.interp_combo.currentData()))
        self.prev_btn.clicked.connect(lambda: self._jump(-1))
        self.next_btn.clicked.connect(lambda: self._jump(+1))
        self.add_btn.clicked.connect(self.add_at_playhead)
        self.copy_btn.clicked.connect(self.copy_selected)
        self.paste_btn.clicked.connect(self.paste_at_playhead)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.time_edit.committed.connect(self._time_committed)
        self.value_edit.committed.connect(self._value_committed)
        self.time_edit.invalid.connect(lambda m: self.message.emit(m, True))
        self.set_clip(None)

    # ------------------------------------------------------------------ model access
    def _project(self):
        return self.ctx.ws.project

    def _clip(self) -> tuple[Track, Clip] | None:
        p = self._project()
        if p is None or not self._clip_id or p.timeline.get_clip(self._clip_id) is None:
            return None
        return p.timeline.find_clip(self._clip_id)

    def prop(self) -> str:
        return self.prop_combo.currentData() or (self._props[0] if self._props else "")

    def _tb(self) -> Timebase:
        p = self._project()
        return timebase_of(p) if p else Timebase(30)

    def _local_playhead(self) -> float | None:
        hit = self._clip()
        if hit is None:
            return None
        t = self.ctx.ws.edit.playhead - hit[1].timeline_start
        return t if -1e-9 <= t <= hit[1].duration + 1e-9 else None

    @staticmethod
    def _can_paste() -> bool:
        return True  # the engine answers "There are no copied keyframes." when the clipboard is empty

    # ------------------------------------------------------------------ loading
    def set_clip(self, clip_id: str | None) -> None:
        """Show the keyframes of ``clip_id`` (``None``: nothing). Reloading while a drag is in progress is skipped."""
        self._clip_id = clip_id
        hit = self._clip()
        if hit is None:
            self._props, self.available = (), False
            self.body.hide()
            self.note.setText("")
            self.note.hide()
            return
        track, clip = hit
        props = keyframe_properties(track, clip)
        self._props = props
        self.available = bool(props)
        self.body.setVisible(self.available)
        self.note.setVisible(not self.available)
        if not self.available:
            self.note.setText("Keyframes are not available for this clip: text, captions and graphics are animated with animation presets (the export does not read their keyframes).")
            return
        self.note.setText("")
        if [self.prop_combo.itemData(i) for i in range(self.prop_combo.count())] != list(props):
            keep = self.prop_combo.currentData()
            self.prop_combo.blockSignals(True)
            self.prop_combo.clear()
            for p in props:
                self.prop_combo.addItem(PROP_LABELS[p] + (f"  ({has_keyframes(clip, p)})" if has_keyframes(clip, p) else ""), p)
            self.prop_combo.setCurrentIndex(max(0, self.prop_combo.findData(keep)))
            self.prop_combo.blockSignals(False)
        else:
            for i, p in enumerate(props):
                n = has_keyframes(clip, p)
                self.prop_combo.setItemText(i, PROP_LABELS[p] + (f"  ({n})" if n else ""))
        self._load_lane(clip)

    def _load_lane(self, clip: Clip) -> None:
        prop = self.prop()
        self.prop_hint.setText(f"{PROP_LABELS[prop]}: {PROP_HINT[prop]}.")
        f, suffix, dec, step = PROP_UNITS[prop]
        lo, hi = KEYFRAME_RANGES[prop]
        self.value_edit.set_unit(f, suffix, dec, step)
        self.value_edit.set_limits(lo, hi)
        self.lane.set_data(prop, clip.duration, clip.keyframes, self._tb(), self._local_playhead(), self.ctx.ws.settings.theme)
        self.lane.read_only = bool(self.lock_reason(prop))
        self._sync_fields()

    def update_playhead(self) -> None:
        if self.available:
            self.lane.set_playhead(self._local_playhead())
            self._sync_nav()

    def _property_changed(self) -> None:
        hit = self._clip()
        if hit is not None:
            self._load_lane(hit[1])

    # ------------------------------------------------------------------ controls state
    def _sync_fields(self) -> None:
        sel = self.lane.selected()
        one = len(sel) == 1
        self.time_edit.setEnabled(one)
        self.value_edit.setEnabled(one)
        tip = "" if one else "Select a single keyframe to type its time and value."
        if one:
            self.time_edit.set_timebase(self._tb())
            self.time_edit.set_seconds(sel[0].time)
            self.value_edit.set_model(sel[0].value)
            self.time_edit.set_base_tooltip("Keyframe time from the start of the clip: HH:MM:SS:FF, seconds (1.25) or frames (12f).")
            self.value_edit.set_base_tooltip("")
        else:
            self.time_edit.set_seconds(None, force=True)
            self.value_edit.set_model(None, force=True)
            self.time_edit.set_base_tooltip(tip)
            self.value_edit.set_base_tooltip(tip)
        shared = {k.interpolation for k in sel}
        self.interp_combo.setEnabled(bool(sel))
        self.interp_combo.setToolTip("" if sel else "Select keyframes to change how the value travels to the next keyframe.")
        if len(shared) == 1:
            self.interp_combo.setCurrentIndex(max(0, self.interp_combo.findData(next(iter(shared)))))
        self.copy_btn.setEnabled(bool(self.lane.keyframes()))
        self.copy_btn.setToolTip("Copy the selected keyframes (all keyframes of this property when none is selected)." if self.lane.keyframes() else "There are no keyframes to copy.")
        self.delete_btn.setEnabled(bool(sel))
        self.delete_btn.setToolTip("" if sel else "Select keyframes first.")
        self.paste_btn.setEnabled(True)
        self.paste_btn.setToolTip("Paste the copied keyframes so the earliest one lands at the playhead.")
        self._sync_nav()
        reason = self.lock_reason(self.prop()) if self.available else ""
        if reason:  # a locked track / clip / aspect: the lane only shows, every editing control says why it is off
            for w in (self.time_edit, self.value_edit, self.interp_combo, self.delete_btn, self.paste_btn, self.add_btn):
                w.setEnabled(False)
                (w.set_base_tooltip if hasattr(w, "set_base_tooltip") else w.setToolTip)(reason)

    def _sync_nav(self) -> None:
        local = self._local_playhead()
        times = [k.time for k in self.lane.keyframes()]
        eps = 1e-6
        prev_t = max((t for t in times if local is not None and t < local - eps), default=None)
        next_t = min((t for t in times if local is not None and t > local + eps), default=None)
        if local is None:  # playhead outside the clip: jump to the nearest keyframe from the clip's edge
            hit = self._clip()
            if hit is not None:
                head = self.ctx.ws.edit.playhead
                prev_t = max((t for t in times if hit[1].timeline_start + t < head - eps), default=None)
                next_t = min((t for t in times if hit[1].timeline_start + t > head + eps), default=None)
        self.prev_btn.setEnabled(prev_t is not None)
        self.next_btn.setEnabled(next_t is not None)
        self.prev_btn.setToolTip("" if prev_t is not None else "There is no earlier keyframe on this property.")
        self.next_btn.setToolTip("" if next_t is not None else "There is no later keyframe on this property.")
        self.add_btn.setEnabled(local is not None)
        self.add_btn.setToolTip("Add a keyframe at the playhead with the value the property has there." if local is not None else "Move the playhead inside the clip first.")
        reason = self.lock_reason(self.prop()) if self.available else ""
        if reason:
            self.add_btn.setEnabled(False)
            self.add_btn.setToolTip(reason)

    # ------------------------------------------------------------------ actions (each = one ws.edit call)
    def _run(self, action: Callable[[], object]) -> bool:
        err: list[str] = []

        def wrapped() -> None:
            try:
                action()
            except AppError as exc:
                err.append(exc.user_message)
                raise

        before = self.ctx.ws.edit.last_notes
        ok = self.ctx.guard(self, wrapped)
        if err:
            self.message.emit(err[0], True)
        elif ok and self.ctx.ws.edit.last_notes is not before and self.ctx.ws.edit.last_notes:
            self.message.emit(self.ctx.ws.edit.last_notes[0], False)
        hit = self._clip()
        if hit is not None:  # always re-sync: a refused edit puts the diamonds back where the model has them
            self._load_lane(hit[1])
        return ok

    def _add(self, t: float, v: float) -> None:
        cid, prop = self._clip_id, self.prop()
        if cid:
            self._run(lambda: self.ctx.ws.edit.add_keyframe(cid, prop, t, v))

    def add_at_playhead(self) -> None:
        cid, prop = self._clip_id, self.prop()
        hit = self._clip()
        local = self._local_playhead()
        if not cid or hit is None:
            return
        if local is None:
            self.message.emit(f"Move the playhead inside the clip (0 – {hit[1].duration:.2f} s from its start) to add a keyframe there.", True)
            return
        t = min(max(self._tb().snap(local), 0.0), hit[1].duration)
        v = value_at(hit[1].keyframes, prop, t)
        self._run(lambda: self.ctx.ws.edit.add_keyframe(cid, prop, t, v))

    def _move(self, moves: list) -> None:
        cid = self._clip_id
        if not cid:
            return
        new_times = [m[2] for m in moves]
        if self._run(lambda: self.ctx.ws.edit.move_keyframes(cid, moves)):
            self.lane.select(new_times)

    def delete_selected(self) -> None:
        cid = self._clip_id
        sel = self.lane.selection()
        if not cid:
            return
        if not sel:
            self.message.emit("Select a keyframe first.", True)
            return
        self._run(lambda: self.ctx.ws.edit.delete_keyframes(cid, sel))
        self.lane.select([])

    def set_interpolation(self, name: str) -> None:
        cid = self._clip_id
        sel = self.lane.selection()
        if not cid:
            return
        if not sel:
            self.message.emit("Select keyframes first.", True)
            return
        self._run(lambda: self.ctx.ws.edit.set_keyframe_interpolation(cid, sel, name))

    def copy_selected(self) -> None:
        cid = self._clip_id
        if not cid:
            return
        sel = self.lane.selection() or [(self.prop(), k.time) for k in self.lane.keyframes()]
        if not sel:
            self.message.emit("There are no keyframes to copy.", True)
            return
        if self._run(lambda: self.ctx.ws.edit.copy_keyframes(cid, sel)):
            self.message.emit(f"Copied {len(sel)} keyframe{'s' if len(sel) != 1 else ''}.", False)
            self._sync_fields()

    def paste_at_playhead(self) -> None:
        cid = self._clip_id
        hit = self._clip()
        if not cid or hit is None:
            return
        local = self._local_playhead()
        if local is None:
            self.message.emit(f"Move the playhead inside the clip (0 – {hit[1].duration:.2f} s from its start) to paste keyframes there.", True)
            return
        t = min(max(self._tb().snap(local), 0.0), hit[1].duration)
        self._run(lambda: self.ctx.ws.edit.paste_keyframes(cid, t))

    def _jump(self, direction: int) -> None:
        hit = self._clip()
        if hit is None:
            return
        head = self.ctx.ws.edit.playhead
        times = sorted(hit[1].timeline_start + k.time for k in self.lane.keyframes())
        eps = 1e-6
        cand = [t for t in times if t < head - eps] if direction < 0 else [t for t in times if t > head + eps]
        if not cand:
            return
        self.ctx.ws.edit.set_playhead(cand[-1] if direction < 0 else cand[0])

    def _time_committed(self, t: float) -> None:
        sel = self.lane.selected()
        if len(sel) == 1 and self._clip_id:
            self._move_one(sel[0], t, None)

    def _value_committed(self, v: float) -> None:
        sel = self.lane.selected()
        if len(sel) == 1 and self._clip_id:
            self._move_one(sel[0], sel[0].time, v)

    def _move_one(self, k: Keyframe, new_t: float, new_v: float | None) -> None:
        cid, prop = self._clip_id, self.prop()
        if cid and self._run(lambda: self.ctx.ws.edit.move_keyframes(cid, [(prop, k.time, new_t, new_v if new_v is not None else k.value)])):
            self.lane.select([new_t])
