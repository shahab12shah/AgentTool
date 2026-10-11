"""Painted timeline: ruler, track rows, clips; mouse editing (move / trim / select / drop).

All geometry <-> time conversion goes through ``x_to_time``/``time_to_x``; the widget never
stores anything in display units. Edits are committed through ``TimelineService`` only.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QMenu, QToolTip, QWidget

from app.performance.profiler import profiler
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
TINY_PX = 3.0  # clips narrower than this are flat fills (and runs of them inside one pixel column are drawn once)
SMALL_PX = 14.0  # below this: outline only, no hatching / tags / text / waveform
TEXT_PX = 30.0  # text needs at least this much room
WAVE_BUCKET_PX = 2.0
MAX_WIDGET_PX = (1 << 24) - 1  # QWIDGETSIZE_MAX


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


class _SnapList(list):
    """A sorted list of snap times (0, the playhead and every clip edge), so the nearest one is a bisect away."""


class TimelineCanvas(QWidget):
    playhead_changed = Signal(float)
    clip_double_clicked = Signal(str)  # asset id
    zoom_changed = Signal(float)
    marker_clicked = Signal(str)  # QC issue id

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._qc_markers: list[dict] = []  # derived from the project's QC issues (never stored here)
        self._mk_sorted: list[dict] = []
        self._mk_times: list[float] = []
        self._mk_maxend: list[float] = []
        self.qc_selected: str | None = None
        self.pps = 80.0
        self.playhead = 0.0
        self._drag: _Drag | None = None
        self._drop_preview: tuple[int, float, float] | None = None  # track idx, start, duration
        self._missing: set[str] = set()
        self._wf_asked: set[str] = set()
        self._expose: tuple[float, float] = (0.0, 1e12)  # x range of the paint event in progress (waveforms only draw what is exposed)
        self._wave_cache: dict[tuple, tuple[object, list]] = {}
        self._elide_cache: dict[tuple, str] = {}
        self._qcolors: dict[tuple, QColor] = {}
        self._snap_cache: tuple[tuple, _SnapList] | None = None
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
        """Recompute size/missing-media cache after the project or its content changed. (Zoom and scroll do not come through here.)"""
        project = self.ctx.ws.project
        self._missing = {a.id for a in project.missing_assets()} if project else set()
        self._wave_cache.clear()
        self.reload_qc_markers()
        self._apply_size()
        self.update()

    def _apply_size(self) -> None:
        project = self.ctx.ws.project
        duration = project.timeline.duration if project else 0.0
        self.setFixedSize(min(MAX_WIDGET_PX, int(self.time_to_x(duration + 60)) + 200), self.content_height())

    # ------------------------------------------------------------ QC markers
    @property
    def qc_markers(self) -> list[dict]:
        return self._qc_markers

    @qc_markers.setter
    def qc_markers(self, markers: list[dict]) -> None:
        self._qc_markers = markers
        self._mk_sorted = sorted(markers, key=lambda m: m["time"])
        self._mk_times = [m["time"] for m in self._mk_sorted]
        top, self._mk_maxend = float("-inf"), []
        for m in self._mk_sorted:
            e = m.get("end")
            top = max(top, m["time"], e if e is not None else m["time"])
            self._mk_maxend.append(top)

    def _markers_between(self, t0: float, t1: float) -> list[dict]:
        """Markers whose flag or span can touch [t0, t1] (seconds); the rest are not even looked at."""
        hi = bisect_right(self._mk_times, t1)
        lo = bisect_left(self._mk_maxend, t0)
        return self._mk_sorted[lo:hi]

    def reload_qc_markers(self) -> None:
        """Re-read the QC markers (severity flags on the ruler; a thin span bar on the affected track). Cheap: they come from the stored issues."""
        try:
            self.qc_markers = self.ctx.ws.qc.markers() if self.ctx.ws.project is not None else []
        except Exception:  # noqa: BLE001  (markers are decoration: never let them break the timeline)
            self.qc_markers = []
        self.update()

    def _marker_flag(self, m: dict) -> QRectF:
        x = self.time_to_x(m["time"])
        return QRectF(x - 6, RULER_H - 13, 12, 13)

    def _marker_span(self, m: dict) -> QRectF | None:
        end = m.get("end")
        tracks = self.tracks()
        idx = next((i for i, t in enumerate(tracks) if t.id == m.get("track_id")), None)
        if end is None or end <= m["time"] + 1e-6 or idx is None:
            return None
        return QRectF(self.time_to_x(m["time"]), self.row_rect_y(idx) + ROW_H - 4, max(3.0, (end - m["time"]) * self.pps), 3)

    def _hit_marker(self, pos) -> dict | None:
        slack = 10.0 / self.pps
        for m in self._markers_between(pos.x() / self.pps - slack, pos.x() / self.pps + slack):
            if self._marker_flag(m).adjusted(-2, 0, 2, 0).contains(pos):
                return m
            span = self._marker_span(m)
            if span is not None and span.adjusted(0, -3, 0, 3).contains(pos):
                return m
        return None

    def _paint_qc_markers(self, p: QPainter, x0: float, x1: float) -> None:
        from app.qc.severity import COLORS, Severity  # noqa: PLC0415

        for m in self._markers_between((x0 - 8) / self.pps, (x1 + 8) / self.pps):
            col = QColor(COLORS[Severity(m["severity"])])
            span = self._marker_span(m)
            if span is not None:
                fill = QColor(col)
                fill.setAlpha(170)
                p.fillRect(span, fill)
            r = self._marker_flag(m)
            p.setPen(QPen(QColor("#ffffff") if m["issue_id"] == self.qc_selected else QColor(0, 0, 0, 120), 1.5 if m["issue_id"] == self.qc_selected else 1))
            p.setBrush(col)
            p.drawPolygon([QPointF(r.left(), r.top()), QPointF(r.right(), r.top()), QPointF(r.center().x(), r.bottom())])

    @profiler.timed("timeline.zoom")
    def set_zoom(self, pps: float) -> None:
        self.pps = max(MIN_PPS, min(MAX_PPS, pps))
        self._apply_size()
        self.update()
        self.zoom_changed.emit(self.pps)

    def set_playhead(self, seconds: float) -> None:
        old = self.playhead
        self.playhead = max(0.0, seconds)
        self.playhead_changed.emit(self.playhead)
        h = self.height()
        for t in (old, self.playhead):  # only the strips around the old and new playhead need repainting
            self.update(QRect(int(self.time_to_x(t)) - 9, 0, 19, h))

    # ------------------------------------------------------------ painting
    def _qcolor(self, name: str, alpha: int | None = None, darker: int | None = None) -> QColor:
        key = (name, alpha, darker)
        col = self._qcolors.get(key)
        if col is None:
            col = QColor(name)
            if darker is not None:
                col = col.darker(darker)
            if alpha is not None:
                col.setAlpha(alpha)
            if len(self._qcolors) > 512:
                self._qcolors.clear()
            self._qcolors[key] = col
        return col

    @profiler.timed("timeline.paint")
    def paintEvent(self, event) -> None:  # noqa: N802
        c = palette(self.ctx.ws.settings.theme)
        p = QPainter(self)
        rect = event.rect()
        x0, x1 = float(rect.left()), float(rect.right() + 1)
        y0, y1 = float(rect.top()), float(rect.bottom() + 1)
        self._expose = (x0, x1)
        p.fillRect(rect, self._qcolor(c["bg"]))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        project = self.ctx.ws.project
        if project is None:
            return
        tracks = project.timeline.tracks
        fm = QFontMetrics(self.font())

        # rows (only those the exposed rectangle touches)
        first = max(0, int((y0 - RULER_H) // ROW_H))
        last = min(len(tracks) - 1, int((y1 - RULER_H) // ROW_H))
        for i in range(first, last + 1):
            y = self.row_rect_y(i)
            p.fillRect(QRectF(x0, y, x1 - x0, ROW_H), self._qcolor(c["panel"] if i % 2 == 0 else c["panel2"]))
            p.setPen(QPen(self._qcolor(c["border"])))
            p.drawLine(int(x0), int(y + ROW_H), int(x1), int(y + ROW_H))

        # ruler (ticks inside the exposed x range, plus a margin for labels that start left of it)
        if y0 < RULER_H:
            p.fillRect(QRectF(x0, 0, x1 - x0, RULER_H), self._qcolor(c["panel2"]))
            step = self._tick_step()
            p.setPen(self._qcolor(c["muted"]))
            k = max(0, int((x0 - 90) / (step * self.pps)))
            while k * step * self.pps < x1:
                t = k * step
                x = self.time_to_x(t)
                p.drawLine(int(x), RULER_H - 8, int(x), RULER_H)
                p.drawText(int(x) + 3, RULER_H - 11, _ruler_label(t, step))
                k += 1
            sub = step / 5
            if sub * self.pps >= 6:
                k = max(0, int(x0 / (sub * self.pps)))
                while k * sub * self.pps < x1:
                    x = self.time_to_x(k * sub)
                    p.drawLine(int(x), RULER_H - 4, int(x), RULER_H)
                    k += 1

        # clips: per visible row, only the clips that overlap the exposed time range
        selected = self.ctx.ws.selected_clip_id
        drag = self._drag if self._drag and self._drag.mode != "playhead" else None
        t0, t1 = (x0 - 6) / self.pps, (x1 + 4) / self.pps
        for i in range(first, last + 1):
            track = tracks[i]
            last_col = None
            for clip in project.timeline.clips_in_range(track.id, t0, t1):
                if drag and drag.clip_id == clip.id:
                    continue  # drawn as drag preview below
                if clip.duration * self.pps < TINY_PX and clip.id != selected and clip.asset_id not in self._missing:
                    col = int(clip.timeline_start * self.pps)
                    if col == last_col:
                        continue  # a run of sub-pixel clips inside one pixel column is drawn once
                    last_col = col
                self._paint_clip(p, c, fm, clip, i, track, clip.id == selected)
        if drag and drag.orig:
            d = drag
            track = tracks[d.track_index]
            self._paint_clip(p, c, fm, d.orig, d.track_index, track, True, d.start, d.end - d.start, ghost=True)
        if self._drop_preview:
            idx, start, dur = self._drop_preview
            r = QRectF(self.time_to_x(start), self.row_rect_y(idx) + 3, max(2.0, dur * self.pps), ROW_H - 6)
            p.setPen(QPen(QColor(c["accent"]), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 4, 4)

        self._paint_qc_markers(p, x0, x1)

        # playhead
        x = self.time_to_x(self.playhead)
        if x0 - 10 <= x <= x1 + 10:
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
        base = c["clip_audio"] if kind is AssetType.AUDIO else c["clip_image"] if kind is AssetType.IMAGE else c["clip_video"]
        if clip.kind == "caption":
            base = "#2c8c8c"
        elif clip.kind == "text":
            base = "#7a5bbf"
        elif clip.kind == "graphic":
            base = "#b9772f"
        alpha = 80 if track.hidden else 190 if ghost else None
        color = self._qcolor(base, alpha)
        r = self.clip_rect(clip, index, start, duration)
        missing = clip.asset_id in self._missing
        if r.width() < TINY_PX and not selected and not missing:
            p.fillRect(r, color)
            return
        p.setBrush(color)
        pen = QPen(self._qcolor(c["selection"]) if selected else self._qcolor(c["danger"]) if missing else self._qcolor(base, alpha, 150), 2 if selected or missing else 1)
        p.setPen(pen)
        p.drawRoundedRect(r, 4, 4)
        if r.width() < SMALL_PX:
            return
        if track.locked:  # diagonal hatching
            p.save()
            p.setClipRect(r)
            p.setPen(QPen(QColor(0, 0, 0, 90), 1))
            x = r.left() - r.height()
            while x < r.right():
                p.drawLine(QPointF(x, r.bottom()), QPointF(x + r.height(), r.top()))
                x += 10
            p.restore()
        if track.is_audio and clip.kind == "media" and asset is not None:
            self._paint_waveform(p, r, clip, asset, track)
        if r.width() < TEXT_PX:
            return
        label = (asset.name if asset else clip.asset_id) + (" ⚠ missing" if missing else "")
        if clip.kind == "caption" and clip.text:
            label = "CC  " + str(clip.text.get("text", ""))
        elif clip.kind == "text" and clip.text:
            label = "T  " + str(clip.text.get("content", ""))
        elif clip.kind == "graphic":
            label = "▭ highlight"
        decision = (project.editing_decisions.get(clip.ai_decision_id) or project.presentation_decisions.get(clip.ai_decision_id)) if clip.ai_decision_id else None
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
        p.setPen(self._qcolor("#ffffff"))
        p.save()
        p.setClipRect(r.adjusted(4, 0, -4, 0))
        p.drawText(r.adjusted(6, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._elided(fm, label, int(r.width()) - 10))
        p.restore()

    def _elided(self, fm: QFontMetrics, label: str, width: int) -> str:
        key = (label, width)
        out = self._elide_cache.get(key)
        if out is None:
            if len(self._elide_cache) > 4096:
                self._elide_cache.clear()
            out = self._elide_cache[key] = fm.elidedText(label, Qt.TextElideMode.ElideRight, width)
        return out

    def _paint_waveform(self, p: QPainter, r: QRectF, clip: Clip, asset, track) -> None:
        """Waveform of the part of the clip that is exposed (2 px buckets on a grid fixed to the clip, so partial repaints line up); peaks are cached per bucket range. Clipped buckets are drawn red."""
        left, width = r.left(), r.width()
        vis_l, vis_r = max(left, self._expose[0] - 2), min(r.right(), self._expose[1] + 2)
        if vis_r <= vis_l or width < 3:
            return
        try:
            wf = self.ctx.ws.presentation.waveform(asset.id, request=False)
            if wf is None:
                if asset.id not in self._wf_asked:  # starting a job inside a paint event would re-enter the UI: defer it
                    self._wf_asked.add(asset.id)
                    QTimer.singleShot(0, lambda aid=asset.id: self.ctx.ws.presentation.waveform(aid))
                return
            j0 = int((vis_l - left) // WAVE_BUCKET_PX)
            j1 = max(j0 + 1, int(-(-(vis_r - left) // WAVE_BUCKET_PX)))
            span = clip.source_out - clip.source_in
            key = (asset.id, clip.source_in, clip.source_out, round(width, 3), j0, j1)
            hit = self._wave_cache.get(key)
            if hit is not None and hit[0] is wf:
                rows = hit[1]
            else:
                ta = clip.source_in + span * (j0 * WAVE_BUCKET_PX / width)
                tb = clip.source_in + span * (min(width, j1 * WAVE_BUCKET_PX) / width)
                rows = wf.range(ta, tb, j1 - j0)
                if len(self._wave_cache) > 256:
                    self._wave_cache.clear()
                self._wave_cache[key] = (wf, rows)
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
                    x0, x1 = left + (a - clip.source_in) / clip.speed * self.pps, left + (b - clip.source_in) / clip.speed * self.pps
                    if x1 > vis_l and x0 < vis_r:
                        p.drawRect(QRectF(max(x0, left), r.top(), min(x1, r.right()) - max(x0, left), r.height()))
            normal, red = QPen(QColor(255, 255, 255, 170), 1), QPen(QColor("#ff4d4d"), 1)
            for k, (mn, mx, clipped) in enumerate(rows):
                x = left + (j0 + k) * WAVE_BUCKET_PX
                p.setPen(red if clipped else normal)
                p.drawLine(QPointF(x, mid - min(1.0, abs(mx) * vol) * half), QPointF(x, mid + min(1.0, abs(mn) * vol) * half))
        finally:
            p.restore()

    # ------------------------------------------------------------ hit testing
    def _hit_clip(self, x: float, y: float) -> tuple[Clip, int, str] | None:
        idx = self.track_index_at(y)
        project = self.ctx.ws.project
        if idx is None or project is None:
            return None
        track = project.timeline.tracks[idx]
        tx = x / self.pps
        slack = 2.0 / self.pps + 1e-6  # clips are drawn at least 2 px wide
        for clip in reversed(project.timeline.clips_in_range(track.id, tx - slack, tx + 1e-6)):
            r = self.clip_rect(clip, idx)
            if r.contains(x, y):
                zone = "trim_start" if x - r.left() <= EDGE_PX else "trim_end" if r.right() - x <= EDGE_PX else "move"
                return clip, idx, zone
        return None

    # ------------------------------------------------------------ snapping
    def _snap_points(self, exclude_clip_id: str) -> list[float]:
        """0, the playhead and every clip edge but the excluded clip's, sorted. Cached per timeline revision (a drag asks for it on every mouse move)."""
        project = self.ctx.ws.project
        if project is None:
            return _SnapList([0.0, self.playhead] if self.playhead > 0 else [0.0])
        key = (id(project.timeline), project.timeline.revision, exclude_clip_id, self.playhead)
        if self._snap_cache is not None and self._snap_cache[0] == key:
            return self._snap_cache[1]
        pts = _SnapList(project.timeline.snap_points(exclude_clip_id))
        for v in (0.0, self.playhead):
            pts.insert(bisect_left(pts, v), v)
        self._snap_cache = (key, pts)
        return pts

    def _snap(self, t: float, points: list[float]) -> float:
        if isinstance(points, _SnapList):
            i = bisect_left(points, t)
            points = points[max(0, i - 1): i + 1]
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
        marker = self._hit_marker(pos)
        if marker is not None:  # a QC marker opens its issue (tested before the ruler's playhead drag)
            self.qc_selected = marker["issue_id"]
            self.set_playhead(marker["time"])
            self.marker_clicked.emit(marker["issue_id"])
            self.update()
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
            marker = self._hit_marker(pos)
            if marker is not None:
                QToolTip.showText(e.globalPosition().toPoint(), f"{marker['severity'].title()} · {marker['title']}  (confidence {marker['confidence']:.0f}%) — from the last QC run", self)
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                return
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


@lru_cache(maxsize=4096)
def _ruler_label(t: float, step: float) -> str:
    if step >= 1:
        total = int(round(t))
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"
    return f"{t:.2f}s"
