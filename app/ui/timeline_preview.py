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
        self.frames = FrameProvider(lambda: ctx.ws.project, lambda: ctx.ws.settings.ffmpeg_path,
                                    lambda a: ctx.ws.render.proxies.proxy_path_for(a) if ctx.ws.settings.use_proxies else None)
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
            elif layer.kind == "caption":
                self._paint_caption(p, layer, r, project)
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
        content = layer.counter_text or str(t.get("content", ""))
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

    def _paint_caption(self, p: QPainter, layer: Layer, r: QRectF, project) -> None:
        """A caption with its style, safe-area position and word-level highlight (progressive reveal or highlight)."""
        from app.captions.styles import effective_style, style_for

        d = layer.text or {}
        cs = project.caption_settings
        st = effective_style(style_for(project.caption_styles, d.get("style_id", cs.style_id)), cs, d.get("style_overrides"))
        lines = d.get("lines") or [d.get("text", "")]
        words = d.get("words", [])
        mode = d.get("highlight_mode", cs.highlight_mode)
        cur = layer.word_index
        font = QFont(st.font)
        font.setPixelSize(max(8, int(st.size_rel * r.height())))
        font.setBold(st.weight == "bold")
        p.save()
        p.setOpacity(max(0.0, min(1.0, layer.opacity * st.opacity)))
        p.setFont(font)
        fm = p.fontMetrics()
        marks = {m["word_index"]: m for m in d.get("emphasis", [])}
        # word positions across the lines
        idx = 0
        shown: list[list[tuple[str, int]]] = []
        for line in lines:
            toks = line.split()
            shown.append([(tok, idx + i) for i, tok in enumerate(toks)])
            idx += len(toks)
        up = st.uppercase
        lh = fm.height() * st.line_spacing
        total_h = lh * len(lines)
        pos = d.get("position", cs.position)
        xy = d.get("position_xy") or []
        cx = r.left() + (xy[0] if pos == "custom" and len(xy) == 2 else 0.5) * r.width()
        if pos == "top":
            top = r.top() + cs.safe_margin_top * r.height()
        elif pos == "center":
            top = r.center().y() - total_h / 2
        elif pos == "custom" and len(xy) == 2:
            top = r.top() + xy[1] * r.height() - total_h / 2
        else:
            top = r.bottom() - cs.safe_margin_bottom * r.height() - total_h
        widths = [sum(fm.horizontalAdvance((t.upper() if up else t) + " ") for t, _ in ln) for ln in shown]
        if st.background == "box" and widths:
            bw = max(widths) + 24
            p.fillRect(QRectF(cx - bw / 2, top - 6, bw, total_h + 12), QColor(int(st.background_color[1:3], 16), int(st.background_color[3:5], 16), int(st.background_color[5:7], 16),
                                                                                int(255 * st.background_opacity)))
        for li, ln in enumerate(shown):
            x = cx - widths[li] / 2
            y = top + li * lh + fm.ascent()
            for tok, wi in ln:
                text = tok.upper() if up else tok
                if mode == "PROGRESSIVE" and wi > cur:
                    x += fm.horizontalAdvance(text + " ")
                    continue
                color = QColor(st.color)
                m = marks.get(wi)
                if (mode == "HIGHLIGHT" and wi == cur) or m:
                    color = QColor(st.highlight_color)
                if st.shadow:
                    p.setPen(QColor(0, 0, 0, 160))
                    p.drawText(QPointF(x + 2, y + 2), text)
                p.setPen(color)
                p.drawText(QPointF(x, y), text)
                if m and m.get("style") == "UNDERLINE":
                    p.drawLine(QPointF(x, y + 3), QPointF(x + fm.horizontalAdvance(text), y + 3))
                x += fm.horizontalAdvance(text + " ")
        p.restore()
        _ = words
