"""Timeline panel: toolbar, track headers, scrollable canvas and a real-time playhead transport.

The panel never edits a clip and never keeps timing of its own: every button / menu entry / key ends in ``ws.edit`` (``TimelineEditService``), ``ws.edit_state`` (tool, ripple, snapping,
linked selection, follow) or a canvas method, wrapped in ``ctx.guard`` so the engine's plain-language refusal reaches the user. The track-header column paints from the canvas'
``TrackLayout`` (the same object the canvas paints and hit-tests with), so the two can never disagree about where a row is.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QCursor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from app.core.exceptions import TimelineError
from app.timeline.meta_commands import TrackHasContent
from app.timeline.snapping import MAX_TOLERANCE, MIN_TOLERANCE, SNAP_KINDS, TARGET_LABELS, TOLERANCE_CHOICES
from app.timeline.timebase import timebase_of
from app.timeline.track import TrackKind
from app.ui.context import UiContext
from app.ui.theme import palette
from app.ui.timeline_canvas import MAX_PPS, MIN_PPS, TimelineCanvas
from app.ui.timeline_layout import DEFAULT_ROW_H, MAX_ROW_H, MIN_ROW_H, RULER_H
from app.ui.timeline_shortcuts import TimelineShortcuts

HEADER_W = 236
RESIZE_PX = 4  # the bottom edge of a header row that resizes the track
DRAG_PX = 4  # a header press must move this far before it becomes a reorder drag
MAX_TIME = 86400.0  # the timecode field refuses jumps beyond 24 hours
HEIGHT_PRESETS = (("Small", 24), ("Normal", 0), ("Large", 72), ("Huge", 120))  # 0 = the default row height
# flag, letter, tooltip. The tooltips are written by ``flag_tip`` (they depend on the track's state); these are the fall-backs.
BUTTONS = (("hidden", "H", "Hide / show track"), ("muted", "M", "Mute track"), ("solo", "S", "Solo track (audio only)"), ("locked", "L", "Lock / unlock track"))
AUDIO_ONLY = {"solo"}
KIND_STYLE = {  # kind -> (palette key or colour, short name)
    TrackKind.VIDEO: ("clip_video", "video"), TrackKind.IMAGE: ("clip_image", "image"), TrackKind.AUDIO: ("clip_audio", "audio"),
    TrackKind.GRAPHICS: ("#b9772f", "graphics"), TrackKind.TEXT: ("#7a5bbf", "text"), TrackKind.CAPTIONS: ("#2c8c8c", "captions"),
}
ADD_TRACK_KINDS = (("Video track", TrackKind.VIDEO), ("Image track", TrackKind.IMAGE), ("Graphics track", TrackKind.GRAPHICS), ("Audio track", TrackKind.AUDIO),
                   ("Text track", TrackKind.TEXT), ("Captions track", TrackKind.CAPTIONS))
FOLLOW_CHOICES = (("Follow: off", "off"), ("Follow: page", "page"), ("Follow: smooth", "smooth"), ("Follow: center", "center"))
SCOPE_CHOICES = (("Ripple: this track", "track"), ("Ripple: selected tracks", "tracks"), ("Ripple: all tracks", "all"))
RIPPLE_COLOR = "#e8912d"


def flag_tip(track, flag: str) -> str:
    on = bool(getattr(track, flag))
    if flag == "hidden":
        return ("Show track again." if on else "Hide track. A hidden track is removed from the preview AND from the exported video.")
    if flag == "muted":
        return "Unmute track." if on else "Mute track: silences its audio in the preview and in the export."
    if flag == "solo":
        return "Solo is on: only soloed audio tracks are heard. Click to turn it off." if on else "Solo: hear only the soloed audio tracks (audio tracks only)."
    return "Unlock track: its clips can be edited again." if on else "Lock track: its clips cannot be edited, moved or deleted until it is unlocked."


# ------------------------------------------------------------------------------------------------------------------------ small widgets
class FlowLayout(QLayout):
    """Wraps its widgets onto as many lines as the width needs (a toolbar that never hides a control and never forces a wide window)."""

    def __init__(self, parent: QWidget | None = None, hspacing: int = 10, vspacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h, self._v = hspacing, vspacing

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._flow(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._flow(rect, False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _flow(self, rect: QRect, test: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:  # does not fit on this line
                x, y, line_h = area.x(), y + line_h + self._v, 0
            if not test:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


class _ModButton(QPushButton):
    """A button that remembers the keyboard modifiers held when it was pressed (Shift / Alt change what "Add selected media" does)."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._mods = Qt.KeyboardModifier.NoModifier

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._mods = e.modifiers()
        super().mousePressEvent(e)

    def take_modifiers(self):
        mods, self._mods = self._mods, Qt.KeyboardModifier.NoModifier
        return mods or QApplication.keyboardModifiers()  # a click made from the keyboard has no mouse event


class PlayheadTransport(QObject):
    """Space = play / pause: a timer moves the playhead in real time (frame-accurate to the project's frame rate) and stops at the end of the timeline or when anything else moves
    the playhead, edits the project or presses Escape. There is NO audio output: the timeline has no composited playback engine, the preview follows the playhead frame by frame."""

    changed = Signal(bool)  # playing / stopped

    def __init__(self, panel: "TimelinePanel", clock=time.monotonic) -> None:
        super().__init__(panel)
        self.panel, self.clock = panel, clock
        self.playing = False
        self._t0 = 0.0  # wall clock when playing started
        self._p0 = 0.0  # playhead when playing started
        self._expected = 0.0  # where this transport last put the playhead (anything else moving it is a manual change)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.tick)

    def start(self) -> None:
        ws = self.panel.ctx.ws
        project = ws.project
        if project is None:
            raise TimelineError("Open or create a project first.")
        duration = project.timeline.duration
        if duration <= 0:
            raise TimelineError("There is nothing to play: the timeline is empty.")
        tb = timebase_of(project)
        canvas = self.panel.canvas
        if canvas.playhead >= duration - tb.frame_s * 0.5:  # at the end: play from the start
            self._move(0.0)
        self._p0, self._t0, self._expected = canvas.playhead, self.clock(), canvas.playhead
        self._timer.start(max(8, int(500.0 / max(1.0, tb.fps))))
        self.playing = True
        self.changed.emit(True)

    def stop(self, why: str = "") -> None:
        if not self.playing:
            return
        self.playing = False
        self._timer.stop()
        self.changed.emit(False)
        if why:
            self.panel.ctx.status(why)

    def toggle(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.start()

    def _move(self, t: float) -> None:
        self._expected = t
        self.panel.canvas.set_playhead(t)

    def tick(self) -> None:
        project = self.panel.ctx.ws.project
        if not self.playing:
            return
        if project is None:
            self.stop()
            return
        target = self._p0 + (self.clock() - self._t0)
        duration = project.timeline.duration
        if target >= duration - 1e-9:
            self._move(duration)
            self.stop("Reached the end of the timeline.")
            return
        t = timebase_of(project).snap(target, "floor")
        if abs(t - self._expected) > 1e-9:
            self._move(t)

    def on_playhead(self, t: float) -> None:
        """Someone other than this transport moved the playhead: playback stops (the user took over)."""
        if self.playing and abs(t - self._expected) > 1e-6:
            self.stop()


# ------------------------------------------------------------------------------------------------------------------------ track headers
class TrackHeaders(QWidget):
    """Left column: one header per track, aligned with the canvas rows through the canvas' ``TrackLayout``.

    Per row: collapse chevron, kind icon / colour, elided name (double-click renames in place), H / M / S / L toggles, and a drag handle on the bottom edge that resizes the track (live,
    committed once on release). Dragging the header body up or down reorders the tracks (a drop line shows where it lands). Right-click opens the track menu (``build_menu``).
    """

    def __init__(self, ctx: UiContext, canvas: TimelineCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.canvas = canvas
        self.guard_widget = lambda w: None  # the panel installs the shortcut registry's text-entry guard here
        self.setFixedWidth(HEADER_W)
        self.setMouseTracking(True)
        self.setObjectName("trackHeaders")
        self._resize: dict | None = None
        self._press: dict | None = None
        self._drop_slot: int | None = None
        self._editor: QLineEdit | None = None
        self._rename_id = ""
        self._cursor_resize = False

    # ------------------------------------------------------------ geometry (always the canvas' layout)
    @property
    def layout_(self):
        return self.canvas.track_layout

    def _tracks(self):
        project = self.ctx.ws.project
        return project.timeline.tracks if project else []

    def _aligned(self) -> bool:
        return self.layout_.count == len(self._tracks()) and bool(self._tracks())

    def reload(self) -> None:
        self.setFixedHeight(self.layout_.total_height)
        ids = {t.id for t in self._tracks()}
        if self._editor is not None and self._rename_id not in ids:
            self._end_rename(False)
        self.update()

    @staticmethod
    def button_size(h: int) -> int:
        return max(12, min(20, h - 8))

    def flag_rects(self, i: int, track) -> list[tuple[str, QRect]]:
        top, h = self.layout_.top(i), self.layout_.height(i)
        size = self.button_size(h)
        flags = [b[0] for b in BUTTONS if not (b[0] in AUDIO_ONLY and not track.is_audio)]
        out, x = [], HEADER_W - 6
        for flag in reversed(flags):
            x -= size
            out.append((flag, QRect(x, top + (h - size) // 2, size, size)))
            x -= 3
        return list(reversed(out))

    def chevron_rect(self, i: int) -> QRect:
        top, h = self.layout_.top(i), self.layout_.height(i)
        return QRect(6, top + (h - 14) // 2, 16, 14)

    def name_rect(self, i: int, track) -> QRect:
        top, h = self.layout_.top(i), self.layout_.height(i)
        left = min((r.left() for _f, r in self.flag_rects(i, track)), default=HEADER_W - 6)
        return QRect(44, top, max(20, left - 6 - 44), h)

    def hit(self, x: float, y: float) -> tuple[str, int, str | None] | None:
        """What is under a point: ("resize" | "chevron" | "flag" | "body", row, flag name or None), or None outside the rows."""
        if not self._aligned():
            return None
        i = self.layout_.index_at(y)
        if i is None:
            return None
        track = self._tracks()[i]
        top, h = self.layout_.top(i), self.layout_.height(i)
        if not track.collapsed and y >= top + h - RESIZE_PX:
            return "resize", i, None
        if self.chevron_rect(i).contains(int(x), int(y)):
            return "chevron", i, None
        for flag, rect in self.flag_rects(i, track):
            if rect.contains(int(x), int(y)):
                return "flag", i, flag
        return "body", i, None

    def tooltip_at(self, x: float, y: float) -> str:
        hit = self.hit(x, y)
        if hit is None:
            return ""
        kind, i, flag = hit
        track = self._tracks()[i]
        if kind == "resize":
            return "Drag to resize this track. Double-click to reset its height."
        if kind == "chevron":
            return "Expand the track." if track.collapsed else "Collapse the track to a thin strip."
        if kind == "flag":
            return flag_tip(track, flag or "")
        n = len(track.clips)
        return (f"{track.name}: {KIND_STYLE[track.kind][1]} track, {n} clip{'s' if n != 1 else ''}"
                + ("\nLocked" if track.locked else "") + ("\nHidden (not previewed or exported)" if track.hidden else "")
                + "\nDrag to reorder · double-click to rename · right-click for more")

    # ------------------------------------------------------------ painting
    def _kind_color(self, c: dict, kind: TrackKind) -> QColor:
        key = KIND_STYLE[kind][0]
        return QColor(c.get(key, key))

    def _paint_kind_icon(self, p: QPainter, r: QRect, kind: TrackKind, color: QColor, c: dict) -> None:
        p.save()
        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        if kind is TrackKind.VIDEO:
            p.drawRoundedRect(x, y + 1, w, h - 2, 2, 2)
            tri = QPainterPath()
            tri.moveTo(x + w * 0.38, y + h * 0.3)
            tri.lineTo(x + w * 0.38, y + h * 0.7)
            tri.lineTo(x + w * 0.7, y + h * 0.5)
            tri.closeSubpath()
            p.fillPath(tri, color)
        elif kind is TrackKind.IMAGE:
            p.drawRoundedRect(x, y + 1, w, h - 2, 2, 2)
            p.drawLine(x + 2, y + h - 3, x + w // 2, y + h // 2)
            p.drawLine(x + w // 2, y + h // 2, x + w - 2, y + h - 3)
        elif kind is TrackKind.AUDIO:
            for k, f in enumerate((0.4, 0.9, 0.6, 1.0, 0.5)):
                bx = x + 1 + k * (w - 2) // 5
                bh = max(2, int((h - 2) * f))
                p.drawLine(bx, y + (h - bh) // 2, bx, y + (h + bh) // 2)
        elif kind is TrackKind.GRAPHICS:
            p.drawRect(x + 1, y + 2, w - 5, h - 5)
            p.drawEllipse(x + 4, y, w - 5, h - 5)
        else:
            f = QFont(self.font())
            f.setPixelSize(max(8, h - 3))
            f.setBold(True)
            p.setFont(f)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "T" if kind is TrackKind.TEXT else "CC")
        p.restore()

    def paintEvent(self, event) -> None:  # noqa: N802
        c = palette(self.ctx.ws.settings.theme)
        p = QPainter(self)
        p.fillRect(event.rect(), QColor(c["panel"]))
        p.fillRect(0, 0, HEADER_W, RULER_H, QColor(c["panel2"]))
        p.setPen(QColor(c["muted"]))
        p.drawText(10, 0, HEADER_W - 10, RULER_H, Qt.AlignmentFlag.AlignVCenter, "Tracks")
        if not self._aligned():
            return
        tracks, layout = self._tracks(), self.layout_
        fm = QFontMetrics(self.font())
        rows = layout.visible(event.rect().top(), event.rect().bottom() + 1)
        dragged = self._press["index"] if self._press and self._press["active"] else None
        for i in rows:
            track, y, h = tracks[i], layout.top(i), layout.height(i)
            p.fillRect(0, y, HEADER_W, h, QColor(c["panel"] if i % 2 == 0 else c["panel2"]))
            p.setPen(QColor(c["border"]))
            p.drawLine(0, y + h, HEADER_W, y + h)
            col = self._kind_color(c, track.kind)
            if track.hidden:
                col.setAlpha(90)
            p.fillRect(0, y, 4, h, col)
            # collapse chevron
            ch = self.chevron_rect(i)
            path = QPainterPath()
            if track.collapsed:
                path.moveTo(ch.x() + 4, ch.y() + 2)
                path.lineTo(ch.x() + 4, ch.y() + ch.height() - 2)
                path.lineTo(ch.x() + 11, ch.y() + ch.height() // 2)
            else:
                path.moveTo(ch.x() + 2, ch.y() + 4)
                path.lineTo(ch.x() + 12, ch.y() + 4)
                path.lineTo(ch.x() + 7, ch.y() + ch.height() - 3)
            path.closeSubpath()
            p.fillPath(path, QColor(c["muted"]))
            # kind icon
            isize = min(14, max(10, h - 6))
            self._paint_kind_icon(p, QRect(24, y + (h - isize) // 2, 16, isize), track.kind, col, c)
            # name (+ a second line on tall rows)
            nr = self.name_rect(i, track)
            p.setPen(QColor(c["muted"] if track.hidden else c["text"]))
            tall = h >= 48 and not track.collapsed
            label = track.name
            if tall:
                p.drawText(nr.x(), y + 4, nr.width(), h // 2 - 2, Qt.AlignmentFlag.AlignBottom, fm.elidedText(label, Qt.TextElideMode.ElideRight, nr.width()))
                n = len(track.clips)
                sub = f"{n} clip{'s' if n != 1 else ''}" + (f" · {track.volume:.0%}" if track.is_audio and abs(track.volume - 1.0) > 1e-9 else "")
                p.setPen(QColor(c["muted"]))
                p.drawText(nr.x(), y + h // 2 + 2, nr.width(), h // 2 - 4, Qt.AlignmentFlag.AlignTop, fm.elidedText(sub, Qt.TextElideMode.ElideRight, nr.width()))
            else:
                if track.is_audio and abs(track.volume - 1.0) > 1e-9:
                    label += f"  {track.volume:.0%}"
                p.drawText(nr, Qt.AlignmentFlag.AlignVCenter, fm.elidedText(label, Qt.TextElideMode.ElideRight, nr.width()))
            # flags
            for flag, rect in self.flag_rects(i, track):
                letter = next(b[1] for b in BUTTONS if b[0] == flag)
                on = bool(getattr(track, flag))
                p.setBrush(QColor(c["accent"]) if on else QColor(c["bg"]))
                p.setPen(QPen(QColor(c["accent"] if on else c["border"])))
                p.drawRoundedRect(rect, 4, 4)
                p.setPen(QColor(c["accent_text"] if on else c["muted"]))
                f = QFont(self.font())
                f.setPixelSize(max(8, rect.height() - 8))
                p.setFont(f)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, letter)
                p.setFont(self.font())
            if dragged == i:
                shade = QColor(c["accent"])
                shade.setAlpha(45)
                p.fillRect(0, y, HEADER_W, h, shade)
        if self._drop_slot is not None:
            n = len(tracks)
            ly = layout.top(self._drop_slot) if self._drop_slot < n else layout.bottom(n - 1)
            p.setPen(QPen(QColor(c["accent"]), 3))
            p.drawLine(0, ly, HEADER_W, ly)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip:
            text = self.tooltip_at(e.pos().x(), e.pos().y())
            if text:
                QToolTip.showText(e.globalPos(), text, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._end_rename(True)
        self.canvas.setFocus()  # the timeline shortcuts work while the timeline has focus
        pos = e.position()
        hit = self.hit(pos.x(), pos.y())
        if hit is None:
            return
        kind, i, flag = hit
        track = self._tracks()[i]
        if e.button() == Qt.MouseButton.RightButton:
            self.build_menu(track).exec(e.globalPosition().toPoint())
        elif e.button() == Qt.MouseButton.LeftButton:
            if kind == "resize":
                self._resize = {"id": track.id, "y0": pos.y(), "h0": self.layout_.height(i), "h": self.layout_.height(i)}
            elif kind == "chevron":
                self.toggle_collapsed(track)
            elif kind == "flag":
                self.toggle_flag(track, flag or "")
            else:
                self._press = {"id": track.id, "index": i, "y0": pos.y(), "active": False}

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        pos = e.position()
        if self._resize is not None:
            r = self._resize
            r["h"] = max(MIN_ROW_H, min(MAX_ROW_H, int(round(r["h0"] + pos.y() - r["y0"]))))
            self.preview_height(r["id"], r["h"])
            return
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton:
            pr = self._press
            if not pr["active"] and abs(pos.y() - pr["y0"]) >= DRAG_PX:
                pr["active"] = True
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            if pr["active"]:
                self._drop_slot = self.slot_at(pos.y())
                self.update()
            return
        if self._press is not None:  # the button was released outside the widget
            self._press, self._drop_slot = None, None
            self.update()
        hit = self.hit(pos.x(), pos.y())
        resize = hit is not None and hit[0] == "resize"
        if resize != self._cursor_resize:
            self._cursor_resize = resize
            if resize:
                self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
            else:
                self.unsetCursor()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._resize is not None:
            r, self._resize = self._resize, None
            self.commit_height(r["id"], r["h"])
        elif self._press is not None:
            pr, self._press = self._press, None
            slot, self._drop_slot = self._drop_slot, None
            self.unsetCursor()
            if pr["active"] and slot is not None:
                self.commit_reorder(pr["id"], slot)
            self.update()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        pos = e.position()
        hit = self.hit(pos.x(), pos.y())
        if hit is None or e.button() != Qt.MouseButton.LeftButton:
            return
        kind, i, _flag = hit
        track = self._tracks()[i]
        if kind == "resize":
            self.set_height(track.id, 0)
        elif kind == "body":
            self.begin_rename(track.id)
        else:
            self.mousePressEvent(e)  # a quick second click on a toggle is another toggle

    # ------------------------------------------------------------ resize (live preview, one command on release)
    def preview_height(self, track_id: str, h: int) -> None:
        """Show ``h`` pixels for the track in both the headers and the canvas without touching the model (the layout is rebuilt from the model on release)."""
        rows = [SimpleNamespace(id=t.id, height=h if t.id == track_id else t.height, collapsed=False if t.id == track_id else t.collapsed) for t in self._tracks()]
        if self.layout_.update(rows):
            self._sync_sizes()

    def _sync_sizes(self) -> None:
        self.setFixedHeight(self.layout_.total_height)
        self.canvas.setFixedHeight(self.layout_.total_height)
        self.canvas.update()
        self.update()

    def commit_height(self, track_id: str, h: int) -> None:
        track = next((t for t in self._tracks() if t.id == track_id), None)
        self.layout_.update(self._tracks())  # drop the preview: the geometry is the model's again
        self._sync_sizes()
        if track is None:
            return
        stored = 0 if h == DEFAULT_ROW_H else h
        if stored != track.height:
            self.set_height(track_id, stored)

    def set_height(self, track_id: str, height: int) -> bool:
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.set_track_display(track_id, height=height), modal=True, title="Track height")

    def toggle_collapsed(self, track) -> bool:
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.set_track_display(track.id, collapsed=not track.collapsed), title="Collapse track")

    def toggle_flag(self, track, flag: str) -> bool:
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.set_track_flag(track.id, flag, not getattr(track, flag)))

    # ------------------------------------------------------------ reorder
    def slot_at(self, y: float) -> int:
        """The insertion slot (0 = above the first row, n = below the last) the pointer is nearest to."""
        layout, n = self.layout_, len(self._tracks())
        for i in range(n):
            if y < layout.top(i) + layout.height(i) / 2:
                return i
        return n

    def commit_reorder(self, track_id: str, slot: int) -> bool:
        tracks = self._tracks()
        src = next((k for k, t in enumerate(tracks) if t.id == track_id), None)
        if src is None:
            return False
        new_index = slot - 1 if slot > src else slot
        if new_index == src:
            return False
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.move_track(track_id, new_index), title="Move track")

    # ------------------------------------------------------------ rename in place
    def rename(self, track_id: str) -> None:
        self.begin_rename(track_id)

    def begin_rename(self, track_id: str) -> None:
        tracks = self._tracks()
        i = next((k for k, t in enumerate(tracks) if t.id == track_id), None)
        if i is None or not self._aligned():
            return
        self._end_rename(False)
        track = tracks[i]
        nr = self.name_rect(i, track)
        h = min(nr.height() - 2, 26)
        ed = QLineEdit(track.name, self)
        ed.setObjectName("trackRenameEdit")
        ed.setGeometry(QRect(nr.x() - 2, nr.y() + (nr.height() - h) // 2, nr.width() + 4, h))
        ed.setMaxLength(80)
        ed.editingFinished.connect(lambda: self._end_rename(True))
        ed.installEventFilter(self)
        self.guard_widget(ed)
        self._editor, self._rename_id = ed, track_id
        ed.show()
        ed.setFocus()
        ed.selectAll()

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        if obj is self._editor and ev.type() == QEvent.Type.KeyPress and ev.key() == Qt.Key.Key_Escape:
            self._end_rename(False)
            self.canvas.setFocus()
            return True
        return False

    def _end_rename(self, commit: bool) -> None:
        ed, self._editor = self._editor, None
        if ed is None:
            return
        track_id, text = self._rename_id, ed.text().strip()
        ed.hide()
        ed.deleteLater()
        track = next((t for t in self._tracks() if t.id == track_id), None)
        if commit and track is not None and text != track.name:
            self.ctx.guard(self, lambda: self.ctx.ws.edit.rename_track(track_id, text), modal=True, title="Rename track")

    # ------------------------------------------------------------ track operations
    def delete_track(self, track) -> bool:
        """Delete a track. A track with clips asks first (and says how many clips go with it); undo brings the track and its clips back."""
        def run() -> None:
            ws = self.ctx.ws
            try:
                ws.edit.remove_track(track.id)
            except TrackHasContent as exc:
                answer = QMessageBox.question(self, "Delete track", f"Delete track “{exc.track_name}” and its {exc.clip_count} clip{'s' if exc.clip_count != 1 else ''}? You can undo this.",
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if answer == QMessageBox.StandardButton.Yes:
                    ws.edit.remove_track(track.id, confirm_content=True)

        return self.ctx.guard(self, run, modal=True, title="Delete track")

    def duplicate_track(self, track, include_clips: bool) -> bool:
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.duplicate_track(track.id, include_clips=include_clips), modal=True, title="Duplicate track")

    def change_kind(self, track, kind: TrackKind) -> bool:
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.set_track_kind(track.id, kind), modal=True, title="Change track type")

    def select_track_clips(self, track) -> None:
        fn = getattr(self.canvas, "select_track_clips", None)
        if fn is not None:
            fn(track.id)
        else:
            self.ctx.ws.selection.select_track(self.ctx.ws.project.timeline, track.id)

    def ask_volume(self, track) -> None:
        pct, ok = QInputDialog.getInt(self, "Track volume", "Volume (%):", int(round(track.volume * 100)), 0, 200)
        if ok:
            self.ctx.guard(self, lambda: self.ctx.ws.timeline.set_track_volume(track.id, pct / 100.0), modal=True, title="Track volume")

    @staticmethod
    def _act(menu: QMenu, text: str, fn, tip: str = "") -> QAction:
        act = QAction(text, menu)
        act.triggered.connect(lambda _checked=False: fn())  # the explicit parameter absorbs the "checked" argument
        if tip:
            act.setToolTip(tip)
        menu.addAction(act)
        return act

    def build_menu(self, track) -> QMenu:
        """The track's context menu (built fresh for the track's current state; the caller shows it)."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        act = self._act
        act(menu, "Rename…", lambda: self.begin_rename(track.id))
        act(menu, "Duplicate track (with clips)", lambda: self.duplicate_track(track, True))
        act(menu, "Duplicate track (empty)", lambda: self.duplicate_track(track, False))
        kinds = menu.addMenu("Change type")
        kinds.setToolTipsVisible(True)
        for kind, (_key, name) in KIND_STYLE.items():
            a = act(kinds, name.capitalize(), lambda k=kind: self.change_kind(track, k),
                    "The track's current type" if kind is track.kind else "Only possible while every clip on the track fits the new type; otherwise the track is left as it is.")
            a.setCheckable(True)
            a.setChecked(kind is track.kind)
            a.setEnabled(kind is not track.kind)
        heights = menu.addMenu("Height")
        for label, px in HEIGHT_PRESETS:
            a = act(heights, f"{label} ({px} px)" if px else f"{label} (default)", lambda v=px: self.set_height(track.id, v))
            a.setCheckable(True)
            a.setChecked(not track.collapsed and track.height == px)
        act(menu, "Expand" if track.collapsed else "Collapse", lambda: self.toggle_collapsed(track))
        menu.addSeparator()
        act(menu, "Show track" if track.hidden else "Hide track", lambda: self.toggle_flag(track, "hidden"), flag_tip(track, "hidden"))
        act(menu, "Unmute track" if track.muted else "Mute track", lambda: self.toggle_flag(track, "muted"), flag_tip(track, "muted"))
        solo = act(menu, "Turn solo off" if track.solo else "Solo track", lambda: self.toggle_flag(track, "solo"), flag_tip(track, "solo") if track.is_audio else "Solo applies to audio tracks only.")
        solo.setEnabled(track.is_audio)
        act(menu, "Unlock track" if track.locked else "Lock track", lambda: self.toggle_flag(track, "locked"), flag_tip(track, "locked"))
        if track.is_audio:
            act(menu, "Set volume…", lambda: self.ask_volume(track))
        menu.addSeparator()
        sel = act(menu, "Select all clips on track", lambda: self.select_track_clips(track), "" if track.clips else "This track has no clips.")
        sel.setEnabled(bool(track.clips))
        act(menu, "Delete track", lambda: self.delete_track(track))
        return menu


# ------------------------------------------------------------------------------------------------------------------------ the panel
class TimelinePanel(QWidget):
    preview_requested = Signal(str)  # asset id
    qc_issue_requested = Signal(str)  # QC issue id (a marker was clicked)
    request_inspector = Signal(str)  # clip id: show the clip in the inspector (forwarded from the canvas)
    request_speed_dialog = Signal(str)  # clip id
    request_replace_dialog = Signal(str)  # clip id

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.canvas = TimelineCanvas(ctx)
        self.headers = TrackHeaders(ctx, self.canvas)
        self.transport = PlayheadTransport(self)
        self.shortcuts = TimelineShortcuts(None, ctx, self)  # key bindings; every button below that has a shortcut runs the same action
        self.headers.guard_widget = self.shortcuts.guard_widget
        self.canvas.clip_double_clicked.connect(self.preview_requested)
        self.canvas.marker_clicked.connect(self.qc_issue_requested)
        for name in ("request_inspector", "request_speed_dialog", "request_replace_dialog"):
            sig = getattr(self.canvas, name, None)  # the canvas defines them (agent U1); a canvas without them simply never emits
            if sig is not None:
                sig.connect(getattr(self, name))
        self.selected_provider = lambda: []  # set by the main window (library selection)
        self._undo_action: QAction | None = None
        self._redo_action: QAction | None = None
        self._tips: list[tuple[QWidget, str, str]] = []  # (widget, base tooltip, action id) - the tooltip shows the current key
        self._gated: dict[QWidget, str] = {}  # button -> action id: enabled only when the action is possible

        self._build_toolbar()

        self.inner = QScrollArea()
        self.inner.setWidget(self.canvas)
        self.inner.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner.setFrameShape(QScrollArea.Shape.NoFrame)
        self.inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.headers, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.inner, 1)
        self.outer = QScrollArea()
        self.outer.setWidgetResizable(True)
        self.outer.setWidget(body)
        self.outer.setFrameShape(QScrollArea.Shape.NoFrame)
        self.outer.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.status_line = QLabel()
        self.status_line.setObjectName("timelineModes")
        self.status_line.setTextFormat(Qt.TextFormat.RichText)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.outer, 1)
        layout.addWidget(self.status_line)

        self.zoom.valueChanged.connect(lambda v: self.canvas.zoom_at(float(v)) if abs(v - self.canvas.pps) > 1 else None)
        self.canvas.zoom_changed.connect(lambda v: self.zoom.setValue(int(v)) if int(v) != self.zoom.value() else None)
        self.canvas.playhead_changed.connect(lambda _t: self._update_time())
        self.transport.changed.connect(self._playing_changed)
        self.shortcuts.changed.connect(self._refresh_tips)

        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: (self.transport.stop(), self.reload()))
        b.on("project.changed", lambda p: self.request_reload() if p.get("scope") in ("timeline", "assets", "waveform", "editing") else None)
        b.on("project.changed", lambda p: (self.canvas.reload_qc_markers(), self._sync_marker_mode()) if p.get("scope") == "qc" else None)
        b.on("qc.updated", lambda p: None if p.get("kind") == "progress" else (self.canvas.reload_qc_markers(), self._sync_marker_mode()))  # progress ticks change no marker
        b.on("selection.changed", lambda p: (self.canvas.update(), self.update_controls()))
        b.on("timeline.edit_state_changed", lambda p: self._sync_edit_state())
        b.on("timeline.playhead", lambda p: self.transport.on_playhead(float(p.get("time", 0.0))))
        b.on("commands.changed", lambda p: (self.transport.stop(), self._sync_history()))
        self._reload_gate = False
        self._reload_pending = False
        self.apply_theme()
        self.reload()
        self._sync_edit_state()
        self._refresh_tips()

    # ------------------------------------------------------------ construction
    def _group(self, *widgets: QWidget) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        for w in widgets:
            lay.addWidget(w)
        return box

    def _button(self, text: str, tip: str, *, action: str = "", checkable: bool = False, name: str = "") -> QPushButton:
        btn = _ModButton(text)
        btn.setProperty("tl", True)
        btn.setCheckable(checkable)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the timeline keeps keyboard focus, so its shortcuts keep working after a click
        if name:
            btn.setObjectName(name)
        if action:
            self._tips.append((btn, tip, action))
            self._gated[btn] = action
            btn.clicked.connect(lambda _c=False, a=action: self.shortcuts.trigger(a, from_menu=True))
        else:
            btn.setToolTip(tip)
        return btn

    def _build_toolbar(self) -> None:
        self.toolbar = QWidget()
        self.toolbar.setObjectName("timelineToolbar")
        flow = FlowLayout(self.toolbar, 12, 4)
        flow.setContentsMargins(8, 4, 8, 4)

        # tools
        self.select_btn = self._button("Select", "Select tool: click, drag and trim clips.", checkable=True, name="toolSelect")
        self.blade_btn = self._button("Blade", "Blade tool: click a clip to cut it (Shift+click cuts every unlocked track).", checkable=True, name="toolBlade")
        self._tips += [(self.select_btn, "Select tool: click, drag and trim clips.", "tool_select"), (self.blade_btn, "Blade tool: click a clip to cut it (Shift+click cuts every unlocked track).", "tool_blade")]
        self.select_btn.clicked.connect(lambda _c=False: self._set_state(tool="select"))
        self.blade_btn.clicked.connect(lambda _c=False: self._set_state(tool="blade"))
        flow.addWidget(self._group(self.select_btn, self.blade_btn))

        # modes: snapping (with its options), linked selection, ripple (+ scope)
        self.snap_btn = QToolButton()
        self.snap_btn.setText("Snap")
        self.snap_btn.setProperty("tl", True)
        self.snap_btn.setObjectName("snapToggle")
        self.snap_btn.setCheckable(True)
        self.snap_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.snap_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.snap_menu = QMenu(self.snap_btn)
        self.snap_btn.setMenu(self.snap_menu)
        self._build_snap_menu()
        self.snap_btn.clicked.connect(lambda checked=False: self._set_snap_enabled(checked))
        self.link_btn = self._button("Link", "Linked selection.", checkable=True, name="linkToggle")
        self.link_btn.clicked.connect(lambda checked=False: self._set_state(linked_selection=bool(checked)))
        self.ripple_btn = self._button("Ripple OFF", "Ripple editing.", checkable=True, name="rippleToggle")
        self.ripple_btn.clicked.connect(lambda checked=False: self._set_state(ripple_on=bool(checked)))
        self.scope_combo = QComboBox()
        self.scope_combo.setObjectName("rippleScope")
        self.scope_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for label, value in SCOPE_CHOICES:
            self.scope_combo.addItem(label, value)
        self.scope_combo.setToolTip("Which tracks move when a ripple edit closes or opens a gap (used while Ripple is ON). Linked and grouped clips always follow.")
        self.scope_combo.currentIndexChanged.connect(lambda _i: self._set_state(ripple_scope=self.scope_combo.currentData()))
        self._tips += [(self.snap_btn, "Snapping on / off. The arrow sets the snap distance and what clips snap to.", "toggle_snap"),
                       (self.link_btn, "Linked selection: selecting a clip also selects its linked and grouped clips, and edits keep them in sync.", "toggle_link"),
                       (self.ripple_btn, "Ripple editing: trims, deletes and inserts shift the clips after them, so no gap is left.", "toggle_ripple")]
        flow.addWidget(self._group(self.snap_btn, self.link_btn, self.ripple_btn, self.scope_combo))

        # history
        self.undo_btn = self._button("Undo", "Undo the last change.", name="tlUndo")
        self.redo_btn = self._button("Redo", "Redo the change that was undone.", name="tlRedo")
        self.undo_btn.clicked.connect(lambda _c=False: self._history(self._undo_action, self.ctx.ws.undo))
        self.redo_btn.clicked.connect(lambda _c=False: self._history(self._redo_action, self.ctx.ws.redo))
        flow.addWidget(self._group(self.undo_btn, self.redo_btn))

        # clip operations
        self.split_btn = self._button("Split", "Split at the playhead: the selected clips, or every clip under it when nothing is selected.", action="split", name="tlSplit")
        self.delete_btn = self._button("Delete", "Delete the selected clips (the gap stays).", action="delete", name="tlDelete")
        self.ripple_delete_btn = self._button("Ripple delete", "Delete the selected clips and close the gap.", action="ripple_delete", name="tlRippleDelete")
        flow.addWidget(self._group(self.split_btn, self.delete_btn, self.ripple_delete_btn))

        # markers and range
        self.marker_btn = self._button("Marker", "Add a marker at the playhead.", action="add_marker", name="tlMarker")
        self.in_btn = self._button("In", "Mark in: start of the in / out range at the playhead.", action="mark_in", name="tlIn")
        self.out_btn = self._button("Out", "Mark out: end of the in / out range at the playhead.", action="mark_out", name="tlOut")
        self.clear_range_btn = self._button("Clear range", "Remove the in / out range.", action="clear_range", name="tlClearRange")
        flow.addWidget(self._group(self.marker_btn, self.in_btn, self.out_btn, self.clear_range_btn))

        # transport + timecode
        self.play_btn = self._button("▶ Play", "Play.", action="play_pause", checkable=True, name="tlPlay")
        self.timecode = QLineEdit()
        self.timecode.setObjectName("timecodeEdit")
        self.timecode.setFixedWidth(118)
        self.timecode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timecode.setToolTip("Playhead position (HH:MM:SS:FF). Type a timecode, seconds (1.5) or frames (45f) and press Enter to jump there.")
        self.timecode.returnPressed.connect(self._commit_timecode)
        self.timecode.textEdited.connect(lambda _t: self._set_timecode_invalid(False))
        self.timecode.installEventFilter(self)
        self.shortcuts.guard_widget(self.timecode)
        self.length_label = QLabel()
        self.length_label.setObjectName("muted")
        flow.addWidget(self._group(self.play_btn, self.timecode, self.length_label))

        self.selection_label = QLabel()
        self.selection_label.setObjectName("muted")
        flow.addWidget(self.selection_label)

        # zoom
        self.zoom_out_btn = self._button("−", "Zoom out.", action="zoom_out", name="tlZoomOut")
        self.zoom_in_btn = self._button("+", "Zoom in.", action="zoom_in", name="tlZoomIn")
        self.fit_btn = self._button("Fit", "Show the whole timeline.", action="zoom_fit", name="tlFit")
        self.zoom_sel_btn = self._button("Selection", "Zoom to the selected clips.", action="zoom_selection", name="tlZoomSelection")
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(int(MIN_PPS), int(MAX_PPS))
        self.zoom.setValue(int(self.canvas.pps))
        self.zoom.setFixedWidth(120)
        self.zoom.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.zoom.setToolTip("Zoom (Ctrl+wheel zooms around the pointer)")
        flow.addWidget(self._group(self.zoom_out_btn, self.zoom, self.zoom_in_btn, self.fit_btn, self.zoom_sel_btn))

        # following the playhead
        self.follow_combo = QComboBox()
        self.follow_combo.setObjectName("followMode")
        self.follow_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for label, value in FOLLOW_CHOICES:
            self.follow_combo.addItem(label, value)
        self.follow_combo.setToolTip("How the view follows the playhead: off, jump a page when it leaves the view, scroll smoothly, or keep it centred.")
        self.follow_combo.currentIndexChanged.connect(lambda _i: self._set_state(follow_playhead=self.follow_combo.currentData()))
        self.autoscroll_box = QCheckBox("Auto-scroll")
        self.autoscroll_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.autoscroll_box.setToolTip("Scroll the view when a drag reaches the edge of the timeline.")
        self.autoscroll_box.clicked.connect(lambda checked=False: self._set_state(auto_scroll=bool(checked)))
        flow.addWidget(self._group(self.follow_combo, self.autoscroll_box))

        # media + tracks
        self.add_selected_btn = _ModButton("Add selected media")
        self.add_selected_btn.setProperty("tl", True)
        self.add_selected_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_selected_btn.setObjectName("tlAddMedia")
        self.add_selected_btn.clicked.connect(self._add_selected)
        self.add_track_btn = QPushButton("+ Track")
        self.add_track_btn.setProperty("tl", True)
        self.add_track_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_track_btn.setToolTip("Add a track")
        menu = QMenu(self.add_track_btn)
        for label, kind in ADD_TRACK_KINDS:
            act = QAction(label, menu)
            act.triggered.connect(lambda _checked=False, k=kind: self.add_track(k))
            menu.addAction(act)
        self.add_track_btn.setMenu(menu)
        flow.addWidget(self._group(self.add_selected_btn, self.add_track_btn))

        # QC markers, shortcuts
        self.marker_mode = QComboBox()
        self.marker_mode.setObjectName("qcMarkerMode")
        self.marker_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for label, mode in (("QC markers: show", "all"), ("QC markers: critical only", "critical_only"), ("QC markers: hide", "hidden")):
            self.marker_mode.addItem(label, mode)
        self.marker_mode.setToolTip("Quality-control findings on the timeline")
        self.marker_mode.activated.connect(self._marker_mode_chosen)
        self.keys_btn = self._button("Keys…", "Keyboard shortcuts: see and change the timeline keys.", name="tlKeys")
        self.keys_btn.clicked.connect(lambda _c=False: self.shortcuts.open_dialog())
        flow.addWidget(self._group(self.marker_mode, self.keys_btn))

    def _build_snap_menu(self) -> None:
        menu = self.snap_menu
        menu.setToolTipsVisible(True)
        title = menu.addAction("Snap distance")
        title.setEnabled(False)
        self.tolerance_group = QActionGroup(menu)
        self.tolerance_actions: dict[int, QAction] = {}
        for n in TOLERANCE_CHOICES:
            act = menu.addAction(f"{n} frames")
            act.setCheckable(True)
            self.tolerance_group.addAction(act)
            act.triggered.connect(lambda _c=False, v=n: self._set_tolerance(v))
            self.tolerance_actions[n] = act
        self.custom_tolerance_action = menu.addAction("Custom…")
        self.custom_tolerance_action.setCheckable(True)
        self.tolerance_group.addAction(self.custom_tolerance_action)
        self.custom_tolerance_action.triggered.connect(lambda _c=False: self._custom_tolerance())
        menu.addSeparator()
        head = menu.addAction("Snap to")
        head.setEnabled(False)
        self.target_actions: dict[str, QAction] = {}
        for kind in SNAP_KINDS:
            act = menu.addAction(TARGET_LABELS[kind])
            act.setCheckable(True)
            act.triggered.connect(lambda checked=False, k=kind: self._set_target(k, bool(checked)))
            self.target_actions[kind] = act

    def apply_theme(self) -> None:
        """The panel's own style rules (compact toolbar buttons, the ON look of toggles, the red outline of an invalid timecode) from the current theme."""
        c = palette(self.ctx.ws.settings.theme)
        self.setStyleSheet(f"""
QPushButton[tl="true"], QToolButton[tl="true"] {{ padding: 3px 9px; border-radius: 5px; }}
QToolButton[tl="true"] {{ background: {c['panel2']}; border: 1px solid {c['border']}; }}
QToolButton[tl="true"]:hover {{ border-color: {c['accent']}; }}
QPushButton[tl="true"]:checked, QToolButton[tl="true"]:checked {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']}; font-weight: 600; }}
QPushButton#rippleToggle:checked {{ background: {RIPPLE_COLOR}; color: #1b1205; border-color: {RIPPLE_COLOR}; font-weight: 700; }}
QPushButton#tlPlay:checked {{ background: {c['ok']}; color: #06150d; border-color: {c['ok']}; }}
QLineEdit#timecodeEdit {{ font-family: monospace; padding: 3px 6px; }}
QLineEdit#timecodeEdit[invalid="true"] {{ border: 2px solid {c['danger']}; }}
QLabel#timelineModes {{ color: {c['muted']}; padding: 2px 8px; }}
""")
        self.headers.update()

    # ------------------------------------------------------------ reload
    def request_reload(self) -> None:
        """Reload now, but fold a burst of change events inside one event-loop turn (a multi-step edit publishes one per command) into one trailing reload."""
        if self._reload_gate:
            self._reload_pending = True
            return
        self._reload_gate = True
        QTimer.singleShot(0, self, self._release_reload_gate)
        self.reload()

    def _release_reload_gate(self) -> None:
        self._reload_gate = False
        if self._reload_pending:
            self._reload_pending = False
            self.request_reload()

    def reload(self) -> None:
        self._reload_pending = False
        self.canvas.reload()
        self._sync_marker_mode()
        self.headers.reload()
        self._update_time()
        self.update_controls()
        self._sync_history()

    def _marker_mode_chosen(self, _index: int) -> None:
        mode = self.marker_mode.currentData()
        self.ctx.guard(self, lambda: self.ctx.ws.qc.set_marker_mode(mode))
        self.canvas.reload_qc_markers()

    def _sync_marker_mode(self) -> None:
        p = self.ctx.ws.project
        if p is None:
            return
        i = self.marker_mode.findData(p.qc_settings.marker_mode)
        if i >= 0 and i != self.marker_mode.currentIndex():
            self.marker_mode.blockSignals(True)
            self.marker_mode.setCurrentIndex(i)
            self.marker_mode.blockSignals(False)

    # ------------------------------------------------------------ state -> widgets
    def update_controls(self) -> None:
        """Enable / disable the buttons from the real state (project, selection, clipboard, locks, library selection) and say why in the tooltip. Cheap: counts only."""
        ws = self.ctx.ws
        has = ws.project is not None
        st = self.shortcuts.snapshot()
        for btn, aid in self._gated.items():
            btn.setEnabled(self.shortcuts.why_not(aid, st) is None)
        self.add_track_btn.setEnabled(has)
        self.timecode.setEnabled(has)
        self.snap_btn.setEnabled(True)
        picked = list(self.selected_provider() or []) if has else []
        self.add_selected_btn.setEnabled(has and bool(picked))
        self.add_selected_btn.setToolTip("Open or create a project first." if not has else
                                         "Select media in the library first." if not picked else
                                         f"Add the {len(picked)} selected media item{'s' if len(picked) != 1 else ''} at the playhead (one undo step). "
                                         "Shift+click: insert (later clips move right). Alt+click: overwrite.")
        self.selection_label.setText(self._selection_text())
        self._refresh_tips(st)

    def _selection_text(self) -> str:
        project = self.ctx.ws.project
        if project is None:
            return ""
        tl, sel = project.timeline, self.ctx.ws.selection
        n = len(sel)
        span = sel.span(tl) if n else None
        if not n or span is None:
            return "No selection"
        return f"{n} clip{'s' if n != 1 else ''} selected, {span[1] - span[0]:.1f} s"

    def _refresh_tips(self, state=None) -> None:
        st = state or self.shortcuts.snapshot()
        for w, base, aid in self._tips:
            if w is self.play_btn:
                base = ("Pause" if self.transport.playing else "Play") + ": moves the playhead in real time so the preview follows it. The timeline has no audio output; to hear audio use the voice-over player."
            keys = self.shortcuts.key_text(aid)
            tip = f"{base} ({keys})" if keys else base
            if w in self._gated:
                why = self.shortcuts.why_not(aid, st)
                if why:
                    tip += f"\nNot available: {why}"
            w.setToolTip(tip)

    def _sync_edit_state(self) -> None:
        s = self.ctx.ws.edit_state
        self.select_btn.setChecked(s.tool == "select")
        self.blade_btn.setChecked(s.tool == "blade")
        self.snap_btn.setChecked(s.snap.enabled)
        self.link_btn.setChecked(s.linked_selection)
        self.ripple_btn.setChecked(s.ripple_on)
        self.ripple_btn.setText("Ripple ON" if s.ripple_on else "Ripple OFF")
        for combo, value in ((self.scope_combo, s.ripple_scope), (self.follow_combo, s.follow_playhead)):
            i = combo.findData(value)
            if i >= 0 and i != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
        self.autoscroll_box.setChecked(s.auto_scroll)
        for n, act in self.tolerance_actions.items():
            act.setChecked(s.snap.tolerance_frames == n)
        custom = s.snap.tolerance_frames not in self.tolerance_actions
        self.custom_tolerance_action.setChecked(custom)
        self.custom_tolerance_action.setText(f"Custom ({s.snap.tolerance_frames} frames)…" if custom else "Custom…")
        for kind, act in self.target_actions.items():
            act.setChecked(s.snap.wants(kind))
        self.snap_btn.setText("Snap")
        scope = dict((v, k) for k, v in SCOPE_CHOICES)[s.ripple_scope].replace("Ripple: ", "")
        parts = [f"{'Blade' if s.tool == 'blade' else 'Select'} tool",
                 f"Snapping {'on (' + str(s.snap.tolerance_frames) + ' frames)' if s.snap.enabled else 'off'}",
                 f"Linked selection {'on' if s.linked_selection else 'off'}",
                 f"<b style='color:{RIPPLE_COLOR}'>Ripple ON ({scope})</b>" if s.ripple_on else "Ripple off"]
        if self.transport.playing:
            parts.append("<b>Playing</b>")
        self.status_line.setText(" · ".join(parts))
        self._refresh_tips()

    def _set_state(self, **changes) -> None:
        """Write a toolbar choice to the shared editing state; the toolbar then shows what the state really is (a refused or unchanged value snaps back)."""
        try:
            self.ctx.ws.edit_state.update(**changes)
        except ValueError as exc:
            self.ctx.status(str(exc))
        self._sync_edit_state()

    def _set_snap_enabled(self, enabled: bool) -> None:
        self._set_state(snap=self.ctx.ws.edit_state.snap.replace(enabled=bool(enabled)))

    def _set_tolerance(self, frames: int) -> None:
        self._set_state(snap=self.ctx.ws.edit_state.snap.replace(tolerance_frames=int(frames)))

    def _custom_tolerance(self) -> None:
        s = self.ctx.ws.edit_state
        n, ok = QInputDialog.getInt(self, "Snap distance", f"Snap distance in frames ({MIN_TOLERANCE}–{MAX_TOLERANCE}):", s.snap.tolerance_frames, MIN_TOLERANCE, MAX_TOLERANCE)
        if ok:
            self._set_tolerance(n)
        else:
            self._sync_edit_state()

    def _set_target(self, kind: str, on: bool) -> None:
        self._set_state(snap=self.ctx.ws.edit_state.snap.with_target(kind, on))

    def set_history_actions(self, undo: QAction, redo: QAction) -> None:
        """The main window's Undo / Redo actions: the toolbar buttons run them (so the window's own pre-undo work still happens) and mirror their text."""
        self._undo_action, self._redo_action = undo, redo
        self._sync_history()

    def _history(self, action: QAction | None, fallback) -> None:
        if action is not None:
            action.trigger()
        else:
            self.ctx.guard(self, fallback)

    def _sync_history(self) -> None:
        cmds, has = self.ctx.ws.commands, self.ctx.ws.project is not None
        self.undo_btn.setEnabled(has and cmds.can_undo)
        self.redo_btn.setEnabled(has and cmds.can_redo)
        undo_text, redo_text = getattr(cmds, "undo_text", ""), getattr(cmds, "redo_text", "")
        self.undo_btn.setToolTip(f"Undo {undo_text} (Ctrl+Z)" if has and cmds.can_undo and undo_text else "Nothing to undo." if has else "Open or create a project first.")
        self.redo_btn.setToolTip(f"Redo {redo_text} (Ctrl+Shift+Z)" if has and cmds.can_redo and redo_text else "Nothing to redo." if has else "Open or create a project first.")

    # ------------------------------------------------------------ playhead / timecode field
    def _update_time(self, force: bool = False) -> None:
        project = self.ctx.ws.project
        if project is None:
            self.length_label.setText("")
            if force or not self.timecode.hasFocus():
                self.timecode.setText("--:--:--:--")
            return
        tb = timebase_of(project)
        self.length_label.setText(f"Length {tb.format(project.timeline.duration)}")
        if force or not self.timecode.hasFocus():  # never overwrite what the user is typing
            self.timecode.setText(tb.format(self.canvas.playhead))
            self._set_timecode_invalid(False)

    def _set_timecode_invalid(self, invalid: bool) -> None:
        if bool(self.timecode.property("invalid")) != invalid:
            self.timecode.setProperty("invalid", invalid)
            self.timecode.style().unpolish(self.timecode)
            self.timecode.style().polish(self.timecode)

    def _commit_timecode(self) -> None:
        """Enter in the timecode field: jump to the typed time. Garbage leaves the playhead alone, outlines the field in red and says what is wrong."""
        project = self.ctx.ws.project
        if project is None:
            return
        tb = timebase_of(project)
        try:
            t = tb.parse(self.timecode.text())
            if t < 0:
                raise ValueError("A time cannot be negative.")
            if t > MAX_TIME:
                raise ValueError("That is beyond 24 hours; enter a time inside the timeline.")
        except ValueError as exc:
            self._set_timecode_invalid(True)
            self.ctx.status(str(exc))
            return
        self._set_timecode_invalid(False)
        self.canvas.set_playhead(tb.snap(t))
        self.canvas.ensure_time_visible(self.canvas.playhead)
        self.canvas.setFocus()
        self._update_time(force=True)

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        if obj is self.timecode:
            if ev.type() == QEvent.Type.FocusOut:
                self._set_timecode_invalid(False)
                QTimer.singleShot(0, self._update_time)  # focus has left: show the playhead again
            elif ev.type() == QEvent.Type.KeyPress and ev.key() == Qt.Key.Key_Escape:
                self.canvas.setFocus()
                self._set_timecode_invalid(False)
                self._update_time(force=True)
                return True
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------ playback
    def toggle_playback(self) -> None:
        self.transport.toggle()

    def stop_playback(self) -> None:
        self.transport.stop()

    def _playing_changed(self, playing: bool) -> None:
        self.play_btn.setChecked(playing)
        self.play_btn.setText("|| Pause" if playing else "▶ Play")
        self._refresh_tips()
        self._sync_edit_state()

    def hideEvent(self, event) -> None:  # noqa: N802
        self.transport.stop()  # playback is only meaningful while the timeline is on screen
        super().hideEvent(event)

    # ------------------------------------------------------------ actions
    def add_track(self, kind: TrackKind) -> bool:
        tracks = self.ctx.ws.project.timeline.tracks if self.ctx.ws.project else []
        n = sum(t.kind is kind for t in tracks) + 1
        name = {TrackKind.TEXT: f"T{n} Text", TrackKind.CAPTIONS: f"CC{n} Captions"}.get(kind)  # the default naming only knows video / image / graphics / audio
        return self.ctx.guard(self, lambda: self.ctx.ws.edit.add_track(kind, name), modal=True, title="Add track")

    def _add_selected(self) -> None:
        """Put the media selected in the library on the timeline at the playhead as ONE undo step (plain click: first free spot; Shift: insert; Alt: overwrite)."""
        ids = list(self.selected_provider() or [])
        if not ids:
            self.ctx.status("Select media in the library first.")
            return
        mods = self.add_selected_btn.take_modifiers()
        mode = "insert" if mods & Qt.KeyboardModifier.ShiftModifier else "overwrite" if mods & Qt.KeyboardModifier.AltModifier else "free"
        project = self.ctx.ws.project
        if project is None:
            return
        kinds = {project.assets.get(i).type.value == "audio" for i in ids if project.assets.get(i) is not None}
        if len(kinds) > 1:
            self.ctx.status("Audio and picture media go on different tracks: select one kind at a time.")
            return
        self.ctx.guard(self, lambda: self.ctx.ws.edit.place_assets(ids, self.canvas.playhead, mode=mode), modal=True, title="Add to timeline")
