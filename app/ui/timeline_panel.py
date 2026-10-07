"""Timeline panel: track headers + scrollable canvas + toolbar."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from app.core.timecode import format_timecode
from app.timeline.track import TrackKind
from app.ui.context import UiContext
from app.ui.theme import palette
from app.ui.timeline_canvas import MAX_PPS, MIN_PPS, ROW_H, RULER_H, TimelineCanvas

HEADER_W = 200
BUTTONS = (("hidden", "H", "Hide / show track"), ("muted", "M", "Mute track"), ("locked", "L", "Lock / unlock track"))
BTN = 20


class TrackHeaders(QWidget):
    """Left column: track names with Hide / Mute / Lock toggles. Right-click for more."""

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setFixedWidth(HEADER_W)
        self.setMouseTracking(True)
        self.setObjectName("trackHeaders")

    def reload(self) -> None:
        n = len(self.ctx.ws.project.timeline.tracks) if self.ctx.ws.project else 0
        self.setFixedHeight(RULER_H + max(1, n) * ROW_H + 1)
        self.update()

    def _button_rect(self, row: int, i: int):
        x = HEADER_W - 8 - (len(BUTTONS) - i) * (BTN + 3)
        return x, RULER_H + row * ROW_H + (ROW_H - BTN) // 2, BTN, BTN

    def _hit(self, x: float, y: float) -> tuple[int, int | None] | None:
        project = self.ctx.ws.project
        if project is None or y < RULER_H:
            return None
        row = int((y - RULER_H) // ROW_H)
        if not 0 <= row < len(project.timeline.tracks):
            return None
        for i in range(len(BUTTONS)):
            bx, by, bw, bh = self._button_rect(row, i)
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return row, i
        return row, None

    def paintEvent(self, event) -> None:  # noqa: N802
        c = palette(self.ctx.ws.settings.theme)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(c["panel"]))
        project = self.ctx.ws.project
        p.setPen(QColor(c["muted"]))
        p.drawText(10, 0, HEADER_W - 10, RULER_H, Qt.AlignmentFlag.AlignVCenter, "Tracks")
        if project is None:
            return
        fm = QFontMetrics(self.font())
        for i, track in enumerate(project.timeline.tracks):
            y = RULER_H + i * ROW_H
            p.fillRect(0, y, HEADER_W, ROW_H, QColor(c["panel"] if i % 2 == 0 else c["panel2"]))
            p.setPen(QColor(c["border"]))
            p.drawLine(0, y + ROW_H, HEADER_W, y + ROW_H)
            p.setPen(QColor(c["muted"] if track.hidden else c["text"]))
            name_w = HEADER_W - 16 - len(BUTTONS) * (BTN + 3)
            p.drawText(10, y, name_w, ROW_H, Qt.AlignmentFlag.AlignVCenter, fm.elidedText(track.name, Qt.TextElideMode.ElideRight, name_w))
            for b, (flag, letter, _tip) in enumerate(BUTTONS):
                bx, by, bw, bh = self._button_rect(i, b)
                on = getattr(track, flag)
                p.setBrush(QColor(c["accent"]) if on else QColor(c["bg"]))
                p.setPen(QPen(QColor(c["accent"] if on else c["border"])))
                p.drawRoundedRect(bx, by, bw, bh, 4, 4)
                p.setPen(QColor(c["accent_text"] if on else c["muted"]))
                p.drawText(bx, by, bw, bh, Qt.AlignmentFlag.AlignCenter, letter)

    def event(self, e) -> bool:
        if e.type() == e.Type.ToolTip:
            hit = self._hit(e.pos().x(), e.pos().y())
            if hit and hit[1] is not None:
                QToolTip.showText(e.globalPos(), BUTTONS[hit[1]][2], self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        hit = self._hit(e.position().x(), e.position().y())
        if hit is None:
            return
        row, button = hit
        track = self.ctx.ws.project.timeline.tracks[row]
        if e.button() == Qt.MouseButton.RightButton:
            self._menu(track, e.globalPosition().toPoint())
        elif button is not None:
            flag = BUTTONS[button][0]
            self.ctx.guard(self, lambda: self.ctx.ws.timeline.set_track_flag(track.id, flag, not getattr(track, flag)))

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        hit = self._hit(e.position().x(), e.position().y())
        if hit and hit[1] is None:
            self.rename(self.ctx.ws.project.timeline.tracks[hit[0]].id)

    def rename(self, track_id: str) -> None:
        track = self.ctx.ws.project.timeline.get_track(track_id)
        name, ok = QInputDialog.getText(self, "Rename track", "Track name:", text=track.name)
        if ok:
            self.ctx.guard(self, lambda: self.ctx.ws.timeline.rename_track(track_id, name), modal=True, title="Rename track")

    def _menu(self, track, pos) -> None:
        svc = self.ctx.ws.timeline
        menu = QMenu(self)
        menu.addAction("Rename…", lambda: self.rename(track.id))
        menu.addAction("Show track" if track.hidden else "Hide track",
                       lambda: self.ctx.guard(self, lambda: svc.set_track_flag(track.id, "hidden", not track.hidden)))
        menu.addAction("Unmute track" if track.muted else "Mute track",
                       lambda: self.ctx.guard(self, lambda: svc.set_track_flag(track.id, "muted", not track.muted)))
        menu.addAction("Unlock track" if track.locked else "Lock track",
                       lambda: self.ctx.guard(self, lambda: svc.set_track_flag(track.id, "locked", not track.locked)))
        menu.addSeparator()
        menu.addAction("Delete track", lambda: self.ctx.guard(self, lambda: svc.remove_track(track.id), modal=True, title="Delete track"))
        menu.exec(pos)


class TimelinePanel(QWidget):
    preview_requested = Signal(str)  # asset id

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.headers = TrackHeaders(ctx)
        self.canvas = TimelineCanvas(ctx)
        self.canvas.clip_double_clicked.connect(self.preview_requested)

        # toolbar
        self.add_track_btn = QPushButton("+ Track")
        menu = QMenu(self.add_track_btn)
        for label, kind in (("Video track", TrackKind.VIDEO), ("Image track", TrackKind.IMAGE),
                            ("Graphics track", TrackKind.GRAPHICS), ("Audio track", TrackKind.AUDIO)):
            menu.addAction(label, lambda k=kind: ctx.guard(self, lambda: ctx.ws.timeline.add_track(k), modal=True, title="Add track"))
        self.add_track_btn.setMenu(menu)
        self.add_selected_btn = QPushButton("Add selected media")
        self.add_selected_btn.setToolTip("Adds the media selected in the library at the playhead")
        self.delete_btn = QPushButton("Delete clip")
        self.time_label = QLabel()
        self.time_label.setObjectName("muted")
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(int(MIN_PPS), int(MAX_PPS))
        self.zoom.setValue(int(self.canvas.pps))
        self.zoom.setFixedWidth(140)
        self.zoom.setToolTip("Zoom (Ctrl+wheel)")
        self.selected_provider = lambda: []  # set by the main window (library selection)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Timeline"))
        bar.addWidget(self.add_track_btn)
        bar.addWidget(self.add_selected_btn)
        bar.addWidget(self.delete_btn)
        bar.addStretch(1)
        bar.addWidget(self.time_label)
        bar.addWidget(QLabel("Zoom"))
        bar.addWidget(self.zoom)

        self.inner = QScrollArea()
        self.inner.setWidget(self.canvas)
        self.inner.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner.setFrameShape(QScrollArea.Shape.NoFrame)
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

        note = QLabel("The playhead marks where new media is inserted. Composited timeline playback arrives with the render/preview engine.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.outer, 1)
        layout.addWidget(note)

        self.add_selected_btn.clicked.connect(self._add_selected)
        self.delete_btn.clicked.connect(self.canvas.delete_selected)
        self.zoom.valueChanged.connect(lambda v: self.canvas.set_zoom(float(v)) if abs(v - self.canvas.pps) > 1 else None)
        self.canvas.zoom_changed.connect(lambda v: self.zoom.setValue(int(v)) if int(v) != self.zoom.value() else None)
        self.canvas.playhead_changed.connect(lambda _t: self._update_time())

        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self.reload())
        b.on("project.changed", lambda p: self.reload() if p.get("scope") in ("timeline", "assets") else None)
        b.on("selection.changed", lambda p: (self.canvas.update(), self._update_buttons()))
        self.reload()

    def reload(self) -> None:
        self.canvas.reload()
        self.headers.reload()
        self._update_time()
        self._update_buttons()

    def _update_buttons(self) -> None:
        has = self.ctx.ws.project is not None
        self.add_track_btn.setEnabled(has)
        self.add_selected_btn.setEnabled(has)
        self.delete_btn.setEnabled(has and self.ctx.ws.selected_clip_id is not None)

    def _update_time(self) -> None:
        project = self.ctx.ws.project
        total = project.timeline.duration if project else 0.0
        fps = project.settings.fps if project else None
        self.time_label.setText(f"Playhead {format_timecode(self.canvas.playhead, fps)}   Length {format_timecode(total, fps)}")

    def _add_selected(self) -> None:
        ids = self.selected_provider()
        if not ids:
            self.ctx.status("Select media in the library first.")
            return
        for asset_id in ids:
            self.ctx.guard(self, lambda a=asset_id: self.ctx.ws.timeline.add_asset(a, None, self.canvas.playhead), modal=True, title="Add to timeline")
