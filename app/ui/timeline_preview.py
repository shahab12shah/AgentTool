"""Scrubbable timeline preview: paints what the composer says is on screen at time ``t`` (visuals, motion, text, transitions)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from app.editing.compose import FrameState, Layer
from app.preview.frames import FrameProvider
from app.ui.context import UiContext

STYLE_COLORS = {"NUMBER_CARD": "#f2c14e", "WARNING": "#e5584f", "LOWER_THIRD": "#ffffff", "ENTITY_NAME": "#ffffff", "DATE": "#9ad1ff", "HEADLINE": "#ffffff"}


class TimelinePreview(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.time = 0.0
        self.frame: FrameState | None = None
        self._pix: dict[str, QPixmap] = {}
        self.frames = FrameProvider(lambda: ctx.ws.project, lambda: ctx.ws.settings.ffmpeg_path)
        self.setMinimumSize(480, 270)
        self.setObjectName("timelinePreview")

    def show_time(self, t: float) -> FrameState | None:
        self.time = max(0.0, t)
        project = self.ctx.ws.project
        self.frame = self.ctx.ws.editing.composer().frame_at(self.time) if project is not None else None
        self.update()
        return self.frame

    # ------------------------------------------------------------------ painting
    def _canvas_rect(self) -> QRectF:
        project = self.ctx.ws.project
        ratio = (project.settings.width / project.settings.height) if project else 16 / 9
        w = self.width()
        h = w / ratio
        if h > self.height():
            h = self.height()
            w = h * ratio
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _pixmap(self, layer: Layer) -> QPixmap | None:
        project = self.ctx.ws.project
        asset = project.assets.get(layer.asset_id) if project and layer.asset_id else None
        if asset is None:
            return None
        path = self.frames.frame_path(asset, layer.src_time)
        if path is None:
            return None
        key = str(path)
        if key not in self._pix:
            if len(self._pix) > 60:
                self._pix.clear()
            self._pix[key] = QPixmap(key)
        pm = self._pix[key]
        return pm if not pm.isNull() else None

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0d11"))
        r = self._canvas_rect()
        p.fillRect(r, QColor("#000000"))
        project = self.ctx.ws.project
        if project is None or self.frame is None:
            p.setPen(QColor("#8d95a3"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No preview")
            return
        k = r.width() / project.settings.width  # canvas pixels -> widget pixels
        p.save()
        p.setClipRect(r)
        for layer in self.frame.layers:
            if layer.kind == "media":
                self._paint_media(p, layer, r, k)
        for layer in self.frame.layers:
            if layer.kind == "graphic":
                self._paint_highlight(p, layer, r)
            elif layer.kind == "text":
                self._paint_text(p, layer, r, k)
        p.restore()
        p.setPen(QPen(QColor("#2b3140"), 1))
        p.drawRect(r)

    def _paint_media(self, p: QPainter, layer: Layer, r: QRectF, k: float) -> None:
        pm = self._pixmap(layer)
        p.save()
        p.setOpacity(max(0.0, min(1.0, layer.opacity)))
        if layer.reveal < 1.0:
            p.setClipRect(QRectF(r.left(), r.top(), r.width() * layer.reveal, r.height()), Qt.ClipOperation.IntersectClip)
        if pm is None:
            p.fillRect(r, QColor("#1b2030"))
            p.setPen(QColor("#8d95a3"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "frame unavailable")
            p.restore()
            return
        ratio = pm.width() / pm.height()
        cover = layer.fit != "contain"
        w, h = (r.width(), r.width() / ratio)
        if (h < r.height()) == cover:
            h, w = r.height(), r.height() * ratio
        w, h = w * layer.scale, h * layer.scale
        c = r.center() + QPointF(layer.x * k, layer.y * k)
        p.translate(c)
        p.rotate(layer.rotation)
        p.drawPixmap(QRectF(-w / 2, -h / 2, w, h), pm, QRectF(pm.rect()))
        p.restore()

    def _paint_highlight(self, p: QPainter, layer: Layer, r: QRectF) -> None:
        hl = layer.highlight or {}
        x, y, w, h = hl.get("region", (0.2, 0.3, 0.6, 0.2))
        box = QRectF(r.left() + x * r.width(), r.top() + y * r.height(), w * r.width(), h * r.height())
        p.save()
        p.setOpacity(max(0.0, min(1.0, layer.opacity)))
        if hl.get("darken_surround"):
            p.fillRect(QRectF(r.left(), r.top(), r.width(), box.top() - r.top()), QColor(0, 0, 0, 120))
            p.fillRect(QRectF(r.left(), box.bottom(), r.width(), r.bottom() - box.bottom()), QColor(0, 0, 0, 120))
            p.fillRect(QRectF(r.left(), box.top(), box.left() - r.left(), box.height()), QColor(0, 0, 0, 120))
            p.fillRect(QRectF(box.right(), box.top(), r.right() - box.right(), box.height()), QColor(0, 0, 0, 120))
        p.setPen(QPen(QColor("#f2c14e"), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        p.restore()

    def _paint_text(self, p: QPainter, layer: Layer, r: QRectF, k: float) -> None:
        t = layer.text or {}
        content = str(t.get("content", ""))
        if not content:
            return
        pos = t.get("position", (0.5, 0.5))
        font = QFont(str(t.get("font", "Sans")))
        font.setPixelSize(max(8, int(float(t.get("size", 48)) * k)))
        font.setBold(t.get("emphasis") in ("BOLD_TEXT", "PUNCH_TEXT", "NUMBER_CARD", "WARNING_TEXT"))
        p.save()
        p.setOpacity(max(0.0, min(1.0, layer.opacity * float(t.get("opacity", 1.0)))))
        p.setFont(font)
        fm = p.fontMetrics()
        tw, th = fm.horizontalAdvance(content), fm.height()
        cx, cy = r.left() + pos[0] * r.width(), r.top() + pos[1] * r.height()
        left = cx if t.get("alignment") == "left" else cx - tw / 2
        box = QRectF(left - 10, cy - th / 2 - 4, tw + 20, th + 8)
        if t.get("background", "none") != "none":
            p.fillRect(box, QColor(0, 0, 0, 170))
        p.setPen(QColor(STYLE_COLORS.get(str(t.get("style")), "#ffffff")))
        p.drawText(QPointF(left, cy + fm.ascent() / 2 - 2), content)
        p.restore()
