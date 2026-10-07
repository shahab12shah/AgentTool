"""Painted timeline: ruler, track rows, clips; mouse editing (move / trim / select / drop).

All geometry <-> time conversion goes through ``x_to_time``/``time_to_x``; the widget never
stores anything in display units. Edits are committed through ``TimelineService`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget

from app.core.constants import MIN_CLIP_DURATION
from app.core.timecode import format_timecode
from app.media.asset import AssetType
from app.timeline.clip import Clip
from app.ui.context import UiContext
from app.ui.media_library import ASSET_MIME
from app.ui.theme import palette

ROW_H = 36
RULER_H = 28
EDGE_PX = 7
SNAP_PX = 8
MIN_PPS, MAX_PPS = 5.0, 600.0


@dataclass
class _Drag:
    mode: str  # "move" | "trim_start" | "trim_end" | "playhead"
    clip_id: str = ""
    press_x: float = 0.0
    orig: Clip | None = None
    start: float = 0.0  # preview values
    end: float = 0.0
    track_index: int = 0
    moved: bool = False


class TimelineCanvas(QWidget):
    playhead_changed = Signal(float)
    clip_double_clicked = Signal(str)  # asset id
    zoom_changed = Signal(float)

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.pps = 80.0
        self.playhead = 0.0
        self._drag: _Drag | None = None
        self._drop_preview: tuple[int, float, float] | None = None  # track idx, start, duration
        self._missing: set[str] = set()
        self._wf_asked: set[str] = set()
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setObjectName("timelineCanvas")
        self.reload()

    # ------------------------------------------------------------ geometry
    def tracks(self):
        project = self.ctx.ws.project
        return project.timeline.tracks if project else []

    def time_to_x(self, t: float) -> float:
        return t * self.pps

    def x_to_time(self, x: float) -> float:
        return max(0.0, x / self.pps)

    def track_index_at(self, y: float) -> int | None:
        if y < RULER_H:
            return None
        idx = int((y - RULER_H) // ROW_H)
        return idx if 0 <= idx < len(self.tracks()) else None

    def row_rect_y(self, index: int) -> float:
        return RULER_H + index * ROW_H

    def clip_rect(self, clip: Clip, index: int, start: float | None = None, duration: float | None = None) -> QRectF:
        s = clip.timeline_start if start is None else start
        d = clip.duration if duration is None else duration
        return QRectF(self.time_to_x(s), self.row_rect_y(index) + 3, max(2.0, d * self.pps), ROW_H - 6)

    def content_height(self) -> int:
        return RULER_H + max(1, len(self.tracks())) * ROW_H + 1

    def reload(self) -> None:
        """Recompute size/missing-media cache after the project or its content changed."""
        project = self.ctx.ws.project
        self._missing = {a.id for a in project.missing_assets()} if project else set()
        duration = project.timeline.duration if project else 0.0
        self.setFixedSize(int(self.time_to_x(duration + 60)) + 200, self.content_height())
        self.update()

    def set_zoom(self, pps: float) -> None:
        self.pps = max(MIN_PPS, min(MAX_PPS, pps))
        self.reload()
        self.zoom_changed.emit(self.pps)

    def set_playhead(self, seconds: float) -> None:
        self.playhead = max(0.0, seconds)
        self.playhead_changed.emit(self.playhead)
        self.update()

    # ------------------------------------------------------------ painting
    def paintEvent(self, event) -> None:  # noqa: N802
        c = palette(self.ctx.ws.settings.theme)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(c["bg"]))
        project = self.ctx.ws.project
        if project is None:
            return
        w = self.width()
        fm = QFontMetrics(self.font())

        # rows
        for i, track in enumerate(project.timeline.tracks):
            y = self.row_rect_y(i)
            p.fillRect(QRectF(0, y, w, ROW_H), QColor(c["panel"] if i % 2 == 0 else c["panel2"]))
            p.setPen(QPen(QColor(c["border"])))
            p.drawLine(0, int(y + ROW_H), w, int(y + ROW_H))

        # ruler
        p.fillRect(QRectF(0, 0, w, RULER_H), QColor(c["panel2"]))
        step = self._tick_step()
        p.setPen(QColor(c["muted"]))
        t = 0.0
        while self.time_to_x(t) < w:
            x = self.time_to_x(t)
            p.drawLine(int(x), RULER_H - 8, int(x), RULER_H)
            p.drawText(int(x) + 3, RULER_H - 11, _ruler_label(t, step))
            t += step
        sub = step / 5
        t = 0.0
        while self.time_to_x(t) < w and sub * self.pps >= 6:
            x = self.time_to_x(t)
            p.drawLine(int(x), RULER_H - 4, int(x), RULER_H)
            t += sub

        # clips
        selected = self.ctx.ws.selected_clip_id
        for i, track in enumerate(project.timeline.tracks):
            for clip in track.clips:
                if self._drag and self._drag.clip_id == clip.id and self._drag.mode != "playhead":
                    continue  # drawn as drag preview below
                self._paint_clip(p, c, fm, clip, i, track, clip.id == selected)
        if self._drag and self._drag.mode != "playhead" and self._drag.orig:
            d = self._drag
            tracks = project.timeline.tracks
            track = tracks[d.track_index]
            self._paint_clip(p, c, fm, d.orig, d.track_index, track, True, d.start, d.end - d.start, ghost=True)
        if self._drop_preview:
            idx, start, dur = self._drop_preview
            r = QRectF(self.time_to_x(start), self.row_rect_y(idx) + 3, max(2.0, dur * self.pps), ROW_H - 6)
            p.setPen(QPen(QColor(c["accent"]), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 4, 4)

        # playhead
        x = self.time_to_x(self.playhead)
        p.setPen(QPen(QColor(c["danger"]), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(QColor(c["danger"]))
        p.drawPolygon([QPointF(x - 6, 0), QPointF(x + 6, 0), QPointF(x, 10)])

    def _tick_step(self) -> float:
        for step in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if step * self.pps >= 70:
                return step
        return 1200

    def _paint_clip(self, p: QPainter, c, fm: QFontMetrics, clip: Clip, index: int, track, selected: bool,
                    start: float | None = None, duration: float | None = None, ghost: bool = False) -> None:
        project = self.ctx.ws.project
        assert project is not None
        asset = project.assets.get(clip.asset_id)
        kind = asset.type if asset else AssetType.VIDEO
        color = QColor(c["clip_audio"] if kind is AssetType.AUDIO else c["clip_image"] if kind is AssetType.IMAGE else c["clip_video"])
        if clip.kind == "caption":
            color = QColor("#2c8c8c")
        elif clip.kind == "text":
            color = QColor("#7a5bbf")
        elif clip.kind == "graphic":
            color = QColor("#b9772f")
        if track.hidden:
            color.setAlpha(80)
        elif ghost:
            color.setAlpha(190)
        r = self.clip_rect(clip, index, start, duration)
        p.setBrush(color)
        missing = clip.asset_id in self._missing
        pen = QPen(QColor(c["selection"]) if selected else QColor(c["danger"]) if missing else color.darker(150), 2 if selected or missing else 1)
        p.setPen(pen)
        p.drawRoundedRect(r, 4, 4)
        if track.locked:  # diagonal hatching
            p.save()
            p.setClipRect(r)
            p.setPen(QPen(QColor(0, 0, 0, 90), 1))
            x = r.left() - r.height()
            while x < r.right():
                p.drawLine(QPointF(x, r.bottom()), QPointF(x + r.height(), r.top()))
                x += 10
            p.restore()
        label = (asset.name if asset else clip.asset_id) + (" ⚠ missing" if missing else "")
        if clip.kind == "caption" and clip.text:
            label = "CC  " + str(clip.text.get("text", ""))
        elif clip.kind == "text" and clip.text:
            label = "T  " + str(clip.text.get("content", ""))
        elif clip.kind == "graphic":
            label = "▭ highlight"
        decision = (project.editing_decisions.get(clip.ai_decision_id) or project.presentation_decisions.get(clip.ai_decision_id)) if clip.ai_decision_id else None
        if track.is_audio and clip.kind == "media" and asset is not None:
            self._paint_waveform(p, r, clip, asset, track)
        tags = []
        role = clip.audio.get("role")
        if role in ("MUSIC", "SFX"):
            tags.append("♪" if role == "MUSIC" else "SFX")
        if clip.created_by == "AI":
            tags.append("AI")
        elif clip.created_by == "USER" and clip.scene_id:
            tags.append("✎")
        if clip.locked:
            tags.append("🔒")
        if decision is not None and decision.confidence < 70:
            tags.append("⚠")  # low-confidence AI decision
        if tags:
            label = " ".join(tags) + "  " + label
        p.setPen(QColor("#ffffff"))
        p.save()
        p.setClipRect(r.adjusted(4, 0, -4, 0))
        p.drawText(r.adjusted(6, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   fm.elidedText(label, Qt.TextElideMode.ElideRight, int(r.width()) - 10))
        p.restore()

    def _paint_waveform(self, p: QPainter, r: QRectF, clip: Clip, asset, track) -> None:
        """Waveform of the clip's source range (cached peaks, generated in a background job). Clipped buckets are drawn red."""
        try:
            wf = self.ctx.ws.presentation.waveform(asset.id, request=False)
            if wf is None:
                if asset.id not in self._wf_asked:  # starting a job inside a paint event would re-enter the UI: defer it
                    self._wf_asked.add(asset.id)
                    QTimer.singleShot(0, lambda aid=asset.id: self.ctx.ws.presentation.waveform(aid))
                return
            if r.width() < 3:
                return
            n = max(1, min(4000, int(r.width() // 2)))
            rows = wf.range(clip.source_in, clip.source_out, n)
        except Exception:  # a waveform problem must never break painting
            return
        mid, half = r.center().y(), (r.height() - 6) / 2
        vol = float(clip.audio.get("volume", 1.0)) * track.volume
        p.save()
        try:
            p.setClipRect(r)
            analysis = self.ctx.ws.project.audio_analysis if self.ctx.ws.project else None
            if analysis is not None and asset.id == analysis.asset_id and clip.audio.get("role") == "VOICE":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 38))  # silence regions
                for a, b in analysis.silence_regions:
                    x0, x1 = r.left() + (a - clip.source_in) / clip.speed * self.pps, r.left() + (b - clip.source_in) / clip.speed * self.pps
                    if x1 > r.left() and x0 < r.right():
                        p.drawRect(QRectF(max(x0, r.left()), r.top(), min(x1, r.right()) - max(x0, r.left()), r.height()))
            step = r.width() / n
            for k, (mn, mx, clipped) in enumerate(rows):
                x = r.left() + k * step
                p.setPen(QPen(QColor("#ff4d4d") if clipped else QColor(255, 255, 255, 170), 1))
                p.drawLine(QPointF(x, mid - min(1.0, abs(mx) * vol) * half), QPointF(x, mid + min(1.0, abs(mn) * vol) * half))
        finally:
            p.restore()

    # ------------------------------------------------------------ hit testing
    def _hit_clip(self, x: float, y: float) -> tuple[Clip, int, str] | None:
        idx = self.track_index_at(y)
        if idx is None:
            return None
        track = self.tracks()[idx]
        for clip in reversed(track.clips):
            r = self.clip_rect(clip, idx)
            if r.contains(x, y):
                zone = "trim_start" if x - r.left() <= EDGE_PX else "trim_end" if r.right() - x <= EDGE_PX else "move"
                return clip, idx, zone
        return None

    # ------------------------------------------------------------ snapping
    def _snap_points(self, exclude_clip_id: str) -> list[float]:
        pts = [0.0, self.playhead]
        for clip in self.ctx.ws.project.timeline.all_clips() if self.ctx.ws.project else []:
            if clip.id != exclude_clip_id:
                pts += [clip.timeline_start, clip.timeline_end]
        return pts

    def _snap(self, t: float, points: list[float]) -> float:
        best = min(points, key=lambda q: abs(q - t), default=t)
        return best if abs(best - t) * self.pps <= SNAP_PX else t

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.setFocus()
        pos = e.position()
        if e.button() == Qt.MouseButton.RightButton:
            self._context_menu(e)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if pos.y() < RULER_H:
            self._drag = _Drag("playhead")
            self.set_playhead(self.x_to_time(pos.x()))
            return
        hit = self._hit_clip(pos.x(), pos.y())
        if hit is None:
            self.ctx.ws.select_clip(None)
            self.update()
            return
        clip, idx, zone = hit
        self.ctx.ws.select_clip(clip.id)
        self._drag = _Drag(zone, clip.id, pos.x(), clip.snapshot(), clip.timeline_start, clip.timeline_end, idx)
        self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        pos = e.position()
        d = self._drag
        if d is None:
            hit = self._hit_clip(pos.x(), pos.y()) if pos.y() >= RULER_H else None
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor if hit and hit[2] != "move" else
                                   Qt.CursorShape.OpenHandCursor if hit else Qt.CursorShape.ArrowCursor))
            return
        if d.mode == "playhead":
            self.set_playhead(self.x_to_time(pos.x()))
            return
        orig = d.orig
        assert orig is not None
        if abs(pos.x() - d.press_x) > 2 or d.moved:
            d.moved = True
        if not d.moved:
            return
        dt = (pos.x() - d.press_x) / self.pps
        pts = self._snap_points(orig.id)
        project = self.ctx.ws.project
        assert project is not None
        if d.mode == "move":
            raw = orig.timeline_start + dt
            by_start = self._snap(raw, pts)
            by_end = self._snap(raw + orig.duration, pts) - orig.duration  # snap the clip's end instead
            start = max(0.0, by_start if abs(by_start - raw) <= abs(by_end - raw) else by_end)
            d.start, d.end = start, start + orig.duration
            idx = self.track_index_at(pos.y())
            if idx is not None:
                d.track_index = idx
        elif d.mode == "trim_start":
            asset = project.assets.get(orig.asset_id)
            max_src = None if asset is None or asset.type is AssetType.IMAGE else asset.duration
            r = project.timeline.compute_trim(orig, new_start=self._snap(orig.timeline_start + dt, pts), max_source=max_src)
            d.start, d.end = r.start, r.start + r.duration
        elif d.mode == "trim_end":
            asset = project.assets.get(orig.asset_id)
            max_src = None if asset is None or asset.type is AssetType.IMAGE else asset.duration
            r = project.timeline.compute_trim(orig, new_end=self._snap(orig.timeline_end + dt, pts), max_source=max_src)
            d.start, d.end = r.start, r.start + r.duration
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        d, self._drag = self._drag, None
        if d is None or d.mode == "playhead" or e.button() != Qt.MouseButton.LeftButton:
            self.update()
            return
        orig = d.orig
        assert orig is not None
        if d.moved:
            svc = self.ctx.ws.timeline
            tracks = self.tracks()
            new_track = tracks[d.track_index].id
            if d.mode == "move":
                if abs(d.start - orig.timeline_start) > 1e-6 or new_track != orig.track_id:
                    self.ctx.guard(self, lambda: svc.move_clip(orig.id, d.start, new_track if new_track != orig.track_id else None))
            elif d.mode == "trim_start":
                self.ctx.guard(self, lambda: svc.trim_clip(orig.id, new_start=d.start))
            elif d.mode == "trim_end":
                self.ctx.guard(self, lambda: svc.trim_clip(orig.id, new_end=d.end))
        self.update()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        hit = self._hit_clip(e.position().x(), e.position().y())
        if hit:
            if hit[0].kind == "media":
                self.clip_double_clicked.emit(hit[0].asset_id)

    def wheelEvent(self, e) -> None:  # noqa: N802
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(self.pps * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15))
            e.accept()
        else:
            e.ignore()  # let the scroll area scroll

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def delete_selected(self) -> None:
        cid = self.ctx.ws.selected_clip_id
        if cid:
            self.ctx.guard(self, lambda: self.ctx.ws.timeline.delete_clip(cid))

    def _context_menu(self, e) -> None:
        hit = self._hit_clip(e.position().x(), e.position().y())
        if not hit:
            return
        clip = hit[0]
        self.ctx.ws.select_clip(clip.id)
        menu = QMenu(self)
        if clip.kind == "media":
            menu.addAction("Preview media", lambda: self.clip_double_clicked.emit(clip.asset_id))
        t = self.playhead
        split = menu.addAction("Split at playhead", lambda: self.ctx.guard(self, lambda: self.ctx.ws.timeline.split_clip(clip.id, t), modal=True, title="Split"))
        split.setEnabled(clip.timeline_start < t < clip.timeline_end)
        if clip.scene_id:
            menu.addAction("Unlock clip" if clip.locked else "Lock clip (protect from AI regeneration)",
                           lambda: self.ctx.guard(self, lambda: self.ctx.ws.editing.lock_clip(clip.id, not clip.locked), modal=True, title="Lock"))
        menu.addAction("Delete clip", self.delete_selected)
        menu.exec(e.globalPosition().toPoint())

    # ------------------------------------------------------------ drop from library
    def _drop_target(self, e) -> tuple[int, float, float] | None:
        md = e.mimeData()
        project = self.ctx.ws.project
        if project is None or not md.hasFormat(ASSET_MIME):
            return None
        asset = project.assets.get(bytes(md.data(ASSET_MIME)).decode())
        idx = self.track_index_at(e.position().y())
        if asset is None or idx is None:
            return None
        track = project.timeline.tracks[idx]
        if not track.accepts(asset.type):
            return None
        dur = asset.duration or 5.0
        return idx, self._snap(self.x_to_time(e.position().x()), self._snap_points("")), dur

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasFormat(ASSET_MIME):
            e.acceptProposedAction()

    def dragMoveEvent(self, e) -> None:  # noqa: N802
        self._drop_preview = self._drop_target(e)
        e.acceptProposedAction() if self._drop_preview else e.ignore()
        self.update()

    def dragLeaveEvent(self, e) -> None:  # noqa: N802
        self._drop_preview = None
        self.update()

    def dropEvent(self, e) -> None:  # noqa: N802
        target = self._drop_target(e)
        self._drop_preview = None
        if target is None:
            e.ignore()
            self.update()
            return
        idx, start, _ = target
        asset_id = bytes(e.mimeData().data(ASSET_MIME)).decode()
        track_id = self.tracks()[idx].id
        self.ctx.guard(self, lambda: self.ctx.ws.timeline.add_asset(asset_id, track_id, start))
        e.acceptProposedAction()
        self.update()


def _ruler_label(t: float, step: float) -> str:
    if step >= 1:
        total = int(round(t))
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"
    return f"{t:.2f}s"
