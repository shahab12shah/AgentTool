"""Painted timeline: ruler, markers, track rows, clips - and the professional editing surface (selection, move, trim, roll / slip / slide, blade, drops, menus).

THE RULE: the canvas never edits a clip and never keeps timing of its own. Every edit is one call on ``ctx.ws.edit`` (``TimelineEditService``) wrapped in ``ctx.guard``. While a
mouse gesture is running the canvas only asks the engine for *dry runs* (``preview_move`` / ``preview_trim`` / ``ws.edit.check``) and draws what comes back; the gesture
commits ONCE on mouse release (= one undo step), Escape cancels it, and a drag below 3 px is a click. All geometry <-> time goes through ``x_to_time`` / ``time_to_x``.

Mouse (select tool)
  * click = select (linked / grouped relatives follow when "linked selection" is on) - Ctrl+click toggles - Shift+click selects the range on the track (Ctrl+Shift adds) - click on
    empty space clears - drag on empty space = rubber band (Shift adds, Ctrl toggles) - double-click a media clip = preview it.
  * drag a clip body = move the whole selection rigidly; vertical drag = other track (only if every moved clip fits) - Ctrl+drag = drag a COPY (same tracks) - Alt = invert snapping
    for this drag - drag an edge = trim (ripple trim when ripple mode is on).
  * Ctrl+Alt+drag an edge between two touching clips = ROLL; Alt+Shift+drag a clip body = SLIP (the content moves under the cut); Ctrl+Alt+drag a clip body = SLIDE.
  * middle-button drag pans; Ctrl+wheel zooms around the pointer; Shift+wheel scrolls sideways; the plain wheel scrolls the page.
Ruler: click / drag = playhead (snaps to markers and clip edges), markers can be dragged (double-click edits, Delete removes), the in / out range has draggable handles.
Blade tool: a vertical line follows the (snapped) pointer over the hovered track; click cuts that track (and linked clips), Shift+click cuts every unlocked track.
"""

from __future__ import annotations

import zlib
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from app.core.exceptions import TimelineError
from app.media.asset import AssetType
from app.performance.profiler import profiler
from app.timeline import ops_basic as ob
from app.timeline import ops_ripple as orp
from app.timeline.clip import KIND_CAPTION, KIND_MEDIA, LOCK_KINDS, Clip
from app.timeline.markers import MARKER_KINDS, Marker
from app.timeline.ops_common import link_members, group_members, relatives
from app.timeline.plan import EditPlan, clip_fits_track
from app.timeline.snapping import SnapResult, SnapSession
from app.timeline.timebase import Timebase, timebase_of
from app.timeline.track import Track, TrackKind
from app.ui.context import UiContext
from app.ui.media_library import ASSET_MIME, asset_ids_from_mime
from app.ui.theme import palette
from app.ui.timeline_layout import COLLAPSED_ROW_H, DEFAULT_ROW_H as ROW_H, RULER_H, TrackLayout  # noqa: F401  (ROW_H is re-exported: other modules and tests import it from here)

EDGE_PX = 7
SNAP_PX = 8  # the Phase 9 fixed snap distance (kept for the legacy ``_snap`` helper; gestures use the engine's frame tolerance)
MIN_PPS, MAX_PPS = 5.0, 600.0
TINY_PX = 3.0  # clips narrower than this are flat fills (and runs of them inside one pixel column are drawn once)
SMALL_PX = 14.0  # below this: outline only, no hatching / tags / text / waveform
TEXT_PX = 30.0  # text needs at least this much room
WAVE_BUCKET_PX = 2.0
MAX_WIDGET_PX = (1 << 24) - 1  # QWIDGETSIZE_MAX
DRAG_PX = 3.0  # a gesture shorter than this is a click
NEW_TRACK_STRIP_H = 24  # empty band under the last row: dropping library media here creates a new track
AUTOSCROLL_MARGIN_PX = 36  # the pointer this close to a viewport edge scrolls it
AUTOSCROLL_MAX_PX = 36  # ... by at most this many pixels per tick
AUTOSCROLL_MS = 30
RANGE_HANDLE_PX = 6.0
PENNANT_H = 13.0  # user-marker flag on the ruler
ZOOM_STEP = 1.25
MOVE_KINDS = ("move",)
AUTO_KINDS = ("move", "trim", "roll", "slip", "slide", "rubber", "playhead", "marker", "range_start", "range_end")
SNAP_COLORS = {"playhead": "#ff5d5d", "markers": "#4aa3ff", "clip_edges": "#ffd24d", "scenes": "#b58cff", "captions": "#2cc7c7", "range": "#7ddc7a", "sentences": "#ff9f43", "words": "#ff9f43"}


# ------------------------------------------------------------------------------------------------------------------------ small pure helpers
def edge_scroll_speed(pos: float, extent: float, margin: float = AUTOSCROLL_MARGIN_PX, max_px: int = AUTOSCROLL_MAX_PX) -> int:
    """Pixels to scroll per tick for a pointer at ``pos`` in a viewport ``extent`` pixels long: 0 in the middle, growing towards the edge, never more than ``max_px`` (also when the
    pointer has left the viewport)."""
    margin = min(float(margin), extent / 4.0)
    if margin <= 0:
        return 0
    if pos < margin:
        return -max(1, round(max_px * min(1.0, (margin - pos) / margin)))
    if pos > extent - margin:
        return max(1, round(max_px * min(1.0, (pos - (extent - margin)) / margin)))
    return 0


def format_delta(tb: Timebase, delta: float) -> str:
    """A signed time difference for the drag HUD, compact timecode: ``+2:05`` (2 s 5 frames), ``-0:12``, ``+1:02:03:04`` for long ones."""
    parts = tb.format(abs(delta)).split(":")
    while len(parts) > 2 and int(parts[0]) == 0:
        parts.pop(0)
    parts[0] = str(int(parts[0]))
    return ("-" if delta < -1e-9 else "+") + ":".join(parts)


def _dry(ws, build) -> tuple[EditPlan | None, str | None]:
    """Build a plan and ask the engine whether it would apply: (plan or None, reason or None). Nothing is changed."""
    try:
        plan = build()
    except TimelineError as exc:
        return None, exc.user_message
    return plan, ws.edit.check(plan)


@lru_cache(maxsize=4096)
def _ruler_label(t: float, step: float, tb: Timebase | None = None) -> str:
    if step >= 1:
        total = int(round(t))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    if tb is not None:
        return tb.format(t)
    return f"{t:.2f}s"


class _SnapList(list):
    """A sorted list of snap times (0, the playhead and every clip edge), so the nearest one is a bisect away."""


@dataclass
class _Gesture:
    kind: str  # move | trim | roll | slip | slide | rubber | playhead | marker | range_start | range_end | pan
    press: QPointF
    mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
    moved: bool = False
    clip_id: str = ""
    edge: str = ""  # trim: "start" | "end"
    sel_ids: list[str] = field(default_factory=list)  # the clips the user selected (these may change tracks)
    ids: list[str] = field(default_factory=list)  # ... plus linked / grouped relatives (these only move in time)
    idset: frozenset = frozenset()
    items: list[tuple[str, float, float, int]] = field(default_factory=list)  # id, start, end, row of every moved clip when the gesture began
    span0: tuple[float, float] = (0.0, 0.0)
    primary_row: int = 0
    session: SnapSession | None = None
    toggle_on_release: bool = False
    collapse_on_release: bool = False
    # live state
    delta: float = 0.0
    value: float = 0.0
    row_delta: int = 0
    track_moves: dict | None = None
    copy: bool = False
    ripple: bool = False
    plan: EditPlan | None = None
    reason: str | None = None
    hud: str = ""
    hud_pos: QPointF = field(default_factory=QPointF)
    guide: tuple[float, str, str] | None = None  # time, snap kind, label
    ghosts: list = field(default_factory=list)  # (clip state, row, start, duration)
    target_row: int | None = None
    key: tuple | None = None
    row_cache: dict = field(default_factory=dict)
    bounds: QRectF = field(default_factory=QRectF)
    first: bool = True
    pair: tuple[str, str] | None = None
    limits: tuple[float, float] | None = None
    impossible: str = ""
    marker_id: str = ""
    edge_time: float = 0.0
    other: float = 0.0
    cur: QPointF = field(default_factory=QPointF)
    mode: str = "replace"
    pan_origin: tuple = ()


@dataclass
class _DropState:
    row: int | None  # the row under the pointer; None over the "new track" strip (or when nothing can be dropped)
    ok: bool
    reason: str = ""
    ghosts: list[tuple[float, float]] = field(default_factory=list)  # start, duration
    mode: str = "free"
    at: float = 0.0
    asset_ids: list[str] = field(default_factory=list)
    pos: QPointF = field(default_factory=QPointF)
    strip: bool = False  # over the empty band under the last row: a new track would be created


class MarkerDialog(QDialog):
    """Name, type, colour and note of a timeline marker."""

    def __init__(self, marker: Marker, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit marker")
        self.name = QLineEdit(marker.name)
        self.name.setPlaceholderText("Marker name")
        self.kind = QComboBox()
        for key, (label, _color) in MARKER_KINDS.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(marker.kind)))
        self.color = QLineEdit(marker.color)
        self.color.setPlaceholderText("#RRGGBB - empty = the type's colour")
        pick = QPushButton("Pick…")
        pick.clicked.connect(self._pick)
        self.note = QPlainTextEdit(marker.note)
        self.note.setFixedHeight(80)
        row = QHBoxLayout()
        row.addWidget(self.color, 1)
        row.addWidget(pick)
        form = QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Type", self.kind)
        form.addRow("Colour", row)
        form.addRow("Note", self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def _pick(self) -> None:
        col = QColorDialog.getColor(QColor(self.color.text() or "#4aa3ff"), self, "Marker colour")
        if col.isValid():
            self.color.setText(col.name())

    def values(self) -> dict[str, str]:
        return {"name": self.name.text().strip(), "kind": str(self.kind.currentData()), "color": self.color.text().strip(), "note": self.note.toPlainText()}


class TimelineCanvas(QWidget):
    playhead_changed = Signal(float)
    clip_double_clicked = Signal(str)  # asset id
    zoom_changed = Signal(float)
    marker_clicked = Signal(str)  # QC issue id
    request_inspector = Signal(str)  # clip id: show it in the inspector (the panel connects this)
    request_speed_dialog = Signal(str)  # clip id
    request_replace_dialog = Signal(str)  # clip id

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
        self._g: _Gesture | None = None
        self._drop: _DropState | None = None
        self._blade: tuple[int, float] | None = None  # row, time of the blade hover line
        self._hover_marker: str | None = None
        self._sel_marker: str | None = None
        self._missing: set[str] = set()
        self._wf_asked: set[str] = set()
        self._expose: tuple[float, float] = (0.0, 1e12)  # x range of the paint event in progress (waveforms only draw what is exposed)
        self._wave_cache: dict[tuple, tuple[object, list]] = {}
        self._elide_cache: dict[tuple, str] = {}
        self._qcolors: dict[tuple, QColor] = {}
        self._link_colors: dict[str, QColor] = {}
        self._name_w: dict[str, int] = {}
        self._snap_cache: tuple[tuple, _SnapList] | None = None
        self._um_cache: tuple | None = None  # (key, marker times) for the user markers on the ruler
        self._tb_cache: tuple | None = None
        self._row_index: dict[str, int] = {}
        self._scroll_ref: tuple[QPointF, int, int] = (QPointF(), 0, 0)
        self._last_global = QPoint()
        self._last_mods = Qt.KeyboardModifier.NoModifier
        self.track_layout = TrackLayout()  # row positions / heights (shared with the header column); rebuilt in reload()
        self._auto = QTimer(self)
        self._auto.setInterval(AUTOSCROLL_MS)
        self._auto.timeout.connect(self._autoscroll_tick)
        ctx.bridge.on("timeline.playhead", lambda payload: self._sync_playhead(payload.get("time", 0.0)))  # someone else (a shortcut, the panel) moved the playhead
        ctx.bridge.on("selection.changed", lambda payload: self.update())
        unsub = ctx.ws.edit_state.subscribe(self._on_edit_state)
        self.destroyed.connect(lambda *_: unsub())
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setObjectName("timelineCanvas")
        self.reload()

    # ------------------------------------------------------------ small accessors
    @property
    def _ws(self):
        return self.ctx.ws

    def _timeline(self):
        project = self.ctx.ws.project
        return project.timeline if project else None

    def timebase(self) -> Timebase:
        project = self.ctx.ws.project
        fps = project.settings.fps if project else 30
        c = self._tb_cache
        if c is None or c[0] != fps:
            c = (fps, timebase_of(project) if project else Timebase(30))
            self._tb_cache = c
        return c[1]

    @property
    def gesture_kind(self) -> str | None:
        """What the mouse is doing right now ("move", "trim", ... or None)."""
        return self._g.kind if self._g else None

    @property
    def hud_text(self) -> str:
        """The text of the tooltip that follows the pointer during a gesture (empty when there is none)."""
        return self._g.hud if self._g else ""

    @property
    def snap_guide(self) -> tuple[float, str] | None:
        """(time, snap kind) of the guide line shown during a gesture, or None."""
        g = self._g
        return (g.guide[0], g.guide[1]) if g and g.guide else None

    @property
    def rubber_rect(self) -> QRectF | None:
        g = self._g
        return QRectF(g.press, g.cur).normalized() if g and g.kind == "rubber" and g.moved else None

    @property
    def drop_state(self) -> _DropState | None:
        return self._drop

    # ------------------------------------------------------------ geometry
    def tracks(self):
        project = self.ctx.ws.project
        return project.timeline.tracks if project else []

    def time_to_x(self, t: float) -> float:
        return t * self.pps

    def x_to_time(self, x: float) -> float:
        return max(0.0, x / self.pps)

    def track_index_at(self, y: float) -> int | None:
        return self.track_layout.index_at(y)

    def row_rect_y(self, index: int) -> float:
        return self.track_layout.top(index)

    def row_height(self, index: int) -> int:
        return self.track_layout.height(index)

    def clip_rect(self, clip: Clip, index: int, start: float | None = None, duration: float | None = None) -> QRectF:
        s = clip.timeline_start if start is None else start
        d = clip.duration if duration is None else duration
        return QRectF(self.time_to_x(s), self.row_rect_y(index) + 3, max(2.0, d * self.pps), self.row_height(index) - 6)

    def content_height(self) -> int:
        return self.track_layout.total_height

    def _row_of(self, track_id: str) -> int:
        i = self._row_index.get(track_id)
        if i is None:
            i = next((k for k, t in enumerate(self.tracks()) if t.id == track_id), 0)
        return i

    def reload(self) -> None:
        """Recompute size/missing-media cache after the project or its content changed. (Zoom and scroll do not come through here.)"""
        project = self.ctx.ws.project
        self._missing = {a.id for a in project.missing_assets()} if project else set()
        self.track_layout.update(self.tracks())
        self._row_index = {t.id: i for i, t in enumerate(self.tracks())}
        self._wave_cache.clear()
        self._um_cache = None
        if self._sel_marker and (self._timeline() is None or self._timeline().get_marker(self._sel_marker) is None):
            self._sel_marker = None
        self.reload_qc_markers()
        self._apply_size()
        self.update()

    def _apply_size(self) -> None:
        project = self.ctx.ws.project
        duration = project.timeline.duration if project else 0.0
        self.setFixedSize(min(MAX_WIDGET_PX, int(self.time_to_x(duration + 60)) + 200), self.content_height() + NEW_TRACK_STRIP_H)

    # ------------------------------------------------------------ QC markers (findings from the last QC run)
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
        return QRectF(self.time_to_x(m["time"]), self.track_layout.bottom(idx) - 4, max(3.0, (end - m["time"]) * self.pps), 3)

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

    # ------------------------------------------------------------ user markers and the in / out range (the ruler)
    def _pennant_width(self, m: Marker) -> float:
        if not m.name:
            return 9.0
        w = self._name_w.get(m.name)
        if w is None:
            if len(self._name_w) > 2048:
                self._name_w.clear()
            w = self._name_w[m.name] = min(110, QFontMetrics(self.font()).horizontalAdvance(m.name))
        return 12.0 + w

    def _pennant_rect(self, m: Marker) -> QRectF:
        return QRectF(self.time_to_x(m.time), 1.0, self._pennant_width(m), PENNANT_H)

    def _user_markers_between(self, t0: float, t1: float) -> list[Marker]:
        """User markers whose flag can touch [t0, t1]: a bisect over the sorted marker times (the times list is rebuilt only when the timeline changed)."""
        tl = self._timeline()
        if tl is None or not tl.markers:
            return []
        ms = tl.markers
        key = (id(ms), len(ms), tl.revision)
        if self._um_cache is None or self._um_cache[0] != key:
            self._um_cache = (key, [m.time for m in ms])
        times = self._um_cache[1]
        return ms[bisect_left(times, t0): bisect_right(times, t1)]

    def _hit_user_marker(self, pos) -> Marker | None:
        if pos.y() >= RULER_H:
            return None
        slack = 130.0 / self.pps
        best = None
        for m in self._user_markers_between(pos.x() / self.pps - slack, pos.x() / self.pps + 6.0 / self.pps):
            r = self._pennant_rect(m)
            if r.adjusted(-5, 0, 0, 0).contains(pos) or (abs(pos.x() - r.left()) <= 3 and pos.y() < RULER_H):
                best = m  # the later marker is drawn on top
        return best

    def _range_now(self) -> tuple[float, float] | None:
        """The marked range, with the live edit of a handle drag applied."""
        tl = self._timeline()
        if tl is None or tl.range is None:
            return None
        g = self._g
        if g is not None and g.kind in ("range_start", "range_end") and g.moved:
            return (g.value, g.other) if g.kind == "range_start" else (g.other, g.value)
        return tl.range

    def _hit_range_handle(self, pos) -> str | None:
        r = self._range_now()
        if r is None or pos.y() >= RULER_H:
            return None
        da, db = abs(pos.x() - self.time_to_x(r[0])), abs(pos.x() - self.time_to_x(r[1]))
        if min(da, db) > RANGE_HANDLE_PX:
            return None
        return "range_start" if da <= db else "range_end"

    def _marker_color(self, m: Marker) -> QColor:
        return self._qcolor(m.display_color)

    def _paint_user_markers(self, p: QPainter, c, x0: float, x1: float, fm: QFontMetrics) -> None:
        g = self._g
        live = g.marker_id if g is not None and g.kind == "marker" and g.moved else ""
        for m in self._user_markers_between((x0 - 8) / self.pps - 130.0 / self.pps, (x1 + 8) / self.pps):
            r = self._pennant_rect(m)
            col = self._marker_color(m)
            dim = m.id == live
            if not dim:
                p.setPen(QPen(self._qcolor(m.display_color, 70), 1))
                p.drawLine(QPointF(r.left(), RULER_H), QPointF(r.left(), self.height()))
            self._paint_pennant(p, c, fm, m, r, col, selected=m.id == self._sel_marker, hovered=m.id == self._hover_marker, dim=dim)
        if live and g is not None:
            m = self._timeline().get_marker(live)
            if m is not None:
                r = QRectF(self.time_to_x(g.value), 1.0, self._pennant_width(m), PENNANT_H)
                p.setPen(QPen(self._qcolor(m.display_color), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(r.left(), RULER_H), QPointF(r.left(), self.height()))
                self._paint_pennant(p, c, fm, m, r, self._marker_color(m), selected=True, hovered=False, dim=False)

    def _paint_pennant(self, p: QPainter, c, fm: QFontMetrics, m: Marker, r: QRectF, col: QColor, *, selected: bool, hovered: bool, dim: bool) -> None:
        if dim:
            col = QColor(col)
            col.setAlpha(60)
        p.setPen(QPen(self._qcolor("#ffffff") if selected else self._qcolor(c["border"]), 1.5 if selected or hovered else 1))
        p.setBrush(col)
        p.drawRect(r)
        p.setPen(QPen(col, 2))
        p.drawLine(QPointF(r.left(), 1.0), QPointF(r.left(), RULER_H))
        if m.name and r.width() > 20 and not dim:
            lum = 0.299 * col.red() + 0.587 * col.green() + 0.114 * col.blue()
            p.setPen(QColor("#111111") if lum > 150 else QColor("#ffffff"))
            p.drawText(QRectF(r.left() + 6, r.top(), r.width() - 7, r.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._elided(fm, m.name, int(r.width()) - 8))

    def _paint_range(self, p: QPainter, c, x0: float, x1: float) -> None:
        r = self._range_now()
        if r is None:
            return
        a, b = self.time_to_x(r[0]), self.time_to_x(r[1])
        if b < x0 - 10 or a > x1 + 10:
            return
        band = QColor(c["accent"])
        band.setAlpha(60)
        p.fillRect(QRectF(a, 0, max(1.0, b - a), RULER_H), band)
        p.setPen(QPen(QColor(c["accent"]), 2))
        for x in (a, b):
            p.drawLine(QPointF(x, 0), QPointF(x, RULER_H))
        p.setBrush(QColor(c["accent"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(a, 0, 4, 6))
        p.drawRect(QRectF(b - 4, 0, 4, 6))

    # ------------------------------------------------------------ viewport: zoom / fit / scroll (the scroll area is the canvas' parent's parent)
    @profiler.timed("timeline.zoom")
    def set_zoom(self, pps: float) -> None:
        self.pps = max(MIN_PPS, min(MAX_PPS, pps))
        self._apply_size()
        self.update()
        self.zoom_changed.emit(self.pps)

    def scroll_area(self):
        w = self.parentWidget()
        while w is not None and not isinstance(w, QScrollArea):
            w = w.parentWidget()
        return w

    def _outer_scroll_area(self):
        sa = self.scroll_area()
        w = sa.parentWidget() if sa is not None else None
        while w is not None and not isinstance(w, QScrollArea):
            w = w.parentWidget()
        return w

    def viewport_width(self) -> int:
        sa = self.scroll_area()
        return sa.viewport().width() if sa is not None else self.width()

    def visible_time_range(self) -> tuple[float, float]:
        sa = self.scroll_area()
        x0 = sa.horizontalScrollBar().value() if sa is not None else 0
        return x0 / self.pps, (x0 + self.viewport_width()) / self.pps

    def scroll_to_time(self, t: float, align: str = "left") -> None:
        """Scroll horizontally so ``t`` is at the left edge (``align="left"``, with a small margin) or in the middle (``"center"``)."""
        sa = self.scroll_area()
        if sa is None:
            return
        x = self.time_to_x(t)
        sa.horizontalScrollBar().setValue(int(x - (self.viewport_width() / 2 if align == "center" else 60)))

    def ensure_time_visible(self, t: float, margin_px: int = 40) -> bool:
        """Scroll the least amount that brings ``t`` into view. Returns True when it scrolled."""
        sa = self.scroll_area()
        if sa is None:
            return False
        bar, x = sa.horizontalScrollBar(), self.time_to_x(t)
        left, right = bar.value(), bar.value() + self.viewport_width()
        if x < left + margin_px:
            bar.setValue(int(x - margin_px))
        elif x > right - margin_px:
            bar.setValue(int(x - self.viewport_width() + margin_px))
        else:
            return False
        return True

    def zoom_at(self, pps: float, anchor_time: float | None = None, anchor_viewport_x: float | None = None) -> None:
        """Zoom so that ``anchor_time`` stays under ``anchor_viewport_x`` (default: the viewport centre) - zooming around the pointer / playhead, not around 0."""
        sa = self.scroll_area()
        if sa is None:
            self.set_zoom(pps)
            return
        bar = sa.horizontalScrollBar()
        vx = self.viewport_width() / 2 if anchor_viewport_x is None else anchor_viewport_x
        t = (bar.value() + vx) / self.pps if anchor_time is None else anchor_time
        self.set_zoom(pps)
        bar.setValue(int(self.time_to_x(t) - vx))

    def _zoom_by(self, factor: float) -> None:
        """Zoom around the playhead when it is in view, otherwise around the centre of the view."""
        sa = self.scroll_area()
        left, right = self.visible_time_range()
        if sa is not None and left <= self.playhead <= right:
            self.zoom_at(self.pps * factor, self.playhead, self.time_to_x(self.playhead) - sa.horizontalScrollBar().value())
        else:
            self.zoom_at(self.pps * factor)

    def zoom_in(self) -> None:
        self._zoom_by(ZOOM_STEP)

    def zoom_out(self) -> None:
        self._zoom_by(1.0 / ZOOM_STEP)

    def zoom_to_range(self, t0: float, t1: float, margin_px: int = 40) -> None:
        """Fit [t0, t1] into the viewport and scroll it to the left edge."""
        span = max(t1 - t0, 0.05)
        pps = max(MIN_PPS, min(MAX_PPS, (self.viewport_width() - 2 * margin_px) / span))
        self.set_zoom(pps)
        self.scroll_to_time(max(0.0, t0 - margin_px / self.pps), "left")
        sa = self.scroll_area()
        if sa is not None:
            sa.horizontalScrollBar().setValue(int(self.time_to_x(t0) - margin_px))

    def zoom_to_fit(self) -> None:
        """Show the whole timeline (0 .. its end)."""
        project = self.ctx.ws.project
        end = project.timeline.duration if project else 0.0
        self.zoom_to_range(0.0, end if end > 0 else 10.0)

    def zoom_to_selection(self) -> bool:
        """Fit the selected clips (their combined span). Returns False when nothing is selected."""
        project = self.ctx.ws.project
        if project is None:
            return False
        span = self.ctx.ws.selection.span(project.timeline)
        if not span:
            return False
        self.zoom_to_range(span[0], span[1])
        return True

    def follow_playhead_scroll(self) -> bool:
        """Scroll the view after the playhead moved, as ``edit_state.follow_playhead`` says: ``off`` | ``page`` (jump a page when it leaves the view) | ``smooth`` (keep it away
        from the edges) | ``center``. Never while the playhead itself is being dragged (the drag has its own auto-scroll). Returns True when the view scrolled."""
        mode = self.ctx.ws.edit_state.follow_playhead
        sa = self.scroll_area()
        if mode == "off" or sa is None or (self._g is not None and self._g.kind == "playhead"):
            return False
        bar, vw, x = sa.horizontalScrollBar(), self.viewport_width(), self.time_to_x(self.playhead)
        left = bar.value()
        right = left + vw
        target = None
        if mode == "page":
            if x > right - 2:
                target = x - 0.1 * vw  # moving forward: the next page starts at the playhead
            elif x < left:
                target = x - 0.9 * vw  # moving back: the previous page ends at the playhead
        elif mode == "smooth":
            m = 0.2 * vw
            if x > right - m:
                target = x - (vw - m)
            elif x < left + m:
                target = x - m
        elif mode == "center":
            target = x - vw / 2
        if target is None:
            return False
        bar.setValue(int(max(0.0, target)))
        return bar.value() != left

    def _sync_playhead(self, seconds: float) -> None:
        if abs(seconds - self.playhead) > 1e-9:
            old, self.playhead = self.playhead, max(0.0, seconds)
            self.playhead_changed.emit(self.playhead)
            h = self.height()
            for t in (old, self.playhead):
                self.update(QRect(int(self.time_to_x(t)) - 9, 0, 19, h))
            self.follow_playhead_scroll()

    def set_playhead(self, seconds: float) -> None:
        old = self.playhead
        self.playhead = max(0.0, seconds)
        self.ctx.ws.edit.set_playhead(self.playhead)  # the one authoritative playhead (the snap engine, split, paste, markers and shortcuts read it)
        self.playhead_changed.emit(self.playhead)
        h = self.height()
        for t in (old, self.playhead):  # only the strips around the old and new playhead need repainting
            self.update(QRect(int(self.time_to_x(t)) - 9, 0, 19, h))
        self.follow_playhead_scroll()

    def _on_edit_state(self, _state, changed) -> None:
        if "tool" in changed:
            self._blade = None
            self._refresh_cursor()
            self.update()
        if "follow_playhead" in changed:
            self.follow_playhead_scroll()

    def _refresh_cursor(self) -> None:
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if self.ctx.ws.edit_state.tool == "blade" else Qt.CursorShape.ArrowCursor))

    # ------------------------------------------------------------ auto-scroll while a gesture is near the viewport edge
    def _scene_now(self) -> QPointF:
        """The pointer's position in canvas coordinates now: the last mouse position corrected by how far the views have scrolled since."""
        pos, h0, v0 = self._scroll_ref
        sa, outer = self.scroll_area(), self._outer_scroll_area()
        dh = (sa.horizontalScrollBar().value() - h0) if sa is not None else 0
        dv = (outer.verticalScrollBar().value() - v0) if outer is not None else 0
        return QPointF(pos.x() + dh, pos.y() + dv)

    def _remember(self, e) -> None:
        sa, outer = self.scroll_area(), self._outer_scroll_area()
        self._last_global = e.globalPosition().toPoint()
        self._last_mods = e.modifiers()
        self._scroll_ref = (QPointF(e.position()), sa.horizontalScrollBar().value() if sa is not None else 0, outer.verticalScrollBar().value() if outer is not None else 0)

    def _autoscroll_step(self) -> tuple[int, int]:
        """(dx, dy) the views should scroll on this tick, from where the pointer is in their viewports (0, 0 when auto-scroll is off)."""
        sa, outer = self.scroll_area(), self._outer_scroll_area()
        dx = dy = 0
        if sa is not None:
            vp = sa.viewport()
            dx = edge_scroll_speed(vp.mapFromGlobal(self._last_global).x(), vp.width())
        if outer is not None and outer.verticalScrollBar().maximum() > 0:
            ovp = outer.viewport()
            dy = edge_scroll_speed(ovp.mapFromGlobal(self._last_global).y(), ovp.height())
        return dx, dy

    def _autoscroll_tick(self) -> None:
        g = self._g
        if g is None or not g.moved or g.kind not in AUTO_KINDS:
            self._auto.stop()
            return
        if not self.ctx.ws.edit_state.auto_scroll:
            return
        dx, dy = self._autoscroll_step()
        if not (dx or dy):
            return
        sa, outer = self.scroll_area(), self._outer_scroll_area()
        moved = False
        if dx and sa is not None:
            bar = sa.horizontalScrollBar()
            before = bar.value()
            bar.setValue(before + dx)
            moved = bar.value() != before
        if dy and outer is not None:
            bar = outer.verticalScrollBar()
            before = bar.value()
            bar.setValue(before + dy)
            moved = moved or bar.value() != before
        if moved:  # the clip must stay under the pointer: recompute the whole gesture from the pointer's position in the scrolled view
            self._gesture_move(g, self._scene_now(), self._last_mods)

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

    def _id_color(self, key: str, value: int = 255) -> QColor:
        col = self._link_colors.get(key)
        if col is None:
            if len(self._link_colors) > 2048:
                self._link_colors.clear()
            col = self._link_colors[key] = QColor.fromHsv(zlib.crc32(key.encode()) % 360, 190, value)
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
        tl = project.timeline
        tracks = tl.tracks
        fm = QFontMetrics(self.font())
        tb = self.timebase()
        sel = self.ctx.ws.selection
        g = self._g
        drop = self._drop

        # rows (only those the exposed rectangle touches)
        rows = self.track_layout.visible(y0, y1)
        for i in rows:
            y, h = self.row_rect_y(i), self.row_height(i)
            p.fillRect(QRectF(x0, y, x1 - x0, h), self._qcolor(c["panel"] if i % 2 == 0 else c["panel2"]))
            p.setPen(QPen(self._qcolor(c["border"])))
            p.drawLine(int(x0), int(y + h), int(x1), int(y + h))
        # the row under a cross-track drag / a library drop: green = allowed, red = not allowed
        hl = g.target_row if g is not None and g.moved else (drop.row if drop is not None else None)
        if hl is not None and hl in rows:
            ok = (g.reason is None) if g is not None and g.moved else drop.ok
            col = QColor("#35c46a" if ok else "#e5484d")
            col.setAlpha(60)
            p.fillRect(QRectF(x0, self.row_rect_y(hl), x1 - x0, self.row_height(hl)), col)
        if drop is not None and drop.strip:  # the "new track" strip under the last row
            sy = float(self.content_height())
            col = QColor("#35c46a" if drop.ok else "#e5484d")
            col.setAlpha(70)
            p.fillRect(QRectF(x0, sy, x1 - x0, NEW_TRACK_STRIP_H), col)
            p.setPen(self._qcolor(c["text"]))
            p.drawText(QRectF(x0 + 8, sy, max(10.0, x1 - x0 - 8), NEW_TRACK_STRIP_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "New track")

        # ruler (ticks inside the exposed x range, plus a margin for labels that start left of it)
        if y0 < RULER_H:
            p.fillRect(QRectF(x0, 0, x1 - x0, RULER_H), self._qcolor(c["panel2"]))
            self._paint_range(p, c, x0, x1)
            step = self._tick_step(tb)
            p.setPen(self._qcolor(c["muted"]))
            k = max(0, int((x0 - 90) / (step * self.pps)))
            while k * step * self.pps < x1:
                t = k * step
                x = self.time_to_x(t)
                p.drawLine(int(x), RULER_H - 8, int(x), RULER_H)
                p.drawText(int(x) + 3, RULER_H - 11, _ruler_label(t, step, tb))
                k += 1
            sub = step / 5
            if sub * self.pps >= 6:
                k = max(0, int(x0 / (sub * self.pps)))
                while k * sub * self.pps < x1:
                    x = self.time_to_x(k * sub)
                    p.drawLine(int(x), RULER_H - 4, int(x), RULER_H)
                    k += 1
        self._paint_user_markers(p, c, x0, x1, fm)

        # clips: per visible row, only the clips that overlap the exposed time range
        primary = sel.primary
        dim_ids = g.idset if g is not None and g.kind in MOVE_KINDS and g.moved else None
        t0, t1 = (x0 - 6) / self.pps, (x1 + 4) / self.pps
        for i in rows:
            track = tracks[i]
            last_col = None
            for clip in tl.clips_in_range(track.id, t0, t1):
                is_sel = clip.id in sel  # a dict lookup: no copy of the selection per clip
                if clip.duration * self.pps < TINY_PX and not is_sel and clip.asset_id not in self._missing:
                    col = int(clip.timeline_start * self.pps)
                    if col == last_col:
                        continue  # a run of sub-pixel clips inside one pixel column is drawn once
                    last_col = col
                self._paint_clip(p, c, fm, clip, i, track, is_sel, primary=clip.id == primary, dim=dim_ids is not None and clip.id in dim_ids)
        if g is not None and g.ghosts:
            for state, row, s, d in g.ghosts:
                if row not in rows:
                    continue
                gx = s * self.pps
                if gx > x1 + 6 or gx + d * self.pps < x0 - 6:
                    continue
                self._paint_clip(p, c, fm, state, row, tracks[row], True, s, d, ghost=True, invalid=g.reason is not None)
        if drop is not None and drop.ghosts:
            row = drop.row
            ry = self.row_rect_y(row) if row is not None else float(self.content_height()) + 2
            rh = self.row_height(row) if row is not None else NEW_TRACK_STRIP_H - 4
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#35c46a" if drop.ok else "#e5484d"), 2, Qt.PenStyle.DashLine))
            for s, d in drop.ghosts:
                p.drawRoundedRect(QRectF(self.time_to_x(s), ry + 3, max(2.0, d * self.pps), rh - 6), 4, 4)

        self._paint_qc_markers(p, x0, x1)

        # snap guide, blade line, rubber band
        if g is not None and g.guide is not None:
            gx = self.time_to_x(g.guide[0])
            if x0 - 2 <= gx <= x1 + 2:
                p.setPen(QPen(QColor(SNAP_COLORS.get(g.guide[1], c["accent"])), 1.5, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(gx, 0), QPointF(gx, self.height()))
        if self._blade is not None and self.ctx.ws.edit_state.tool == "blade":
            row, bt = self._blade
            if 0 <= row < self.track_layout.count:
                bx = self.time_to_x(bt)
                p.setPen(QPen(QColor(c["danger"]), 1.5))
                p.drawLine(QPointF(bx, self.row_rect_y(row)), QPointF(bx, self.track_layout.bottom(row)))
        rr = self.rubber_rect
        if rr is not None:
            fill = QColor(c["selection"])
            fill.setAlpha(45)
            p.setPen(QPen(QColor(c["selection"]), 1, Qt.PenStyle.DashLine))
            p.setBrush(fill)
            p.drawRect(rr)

        # playhead
        x = self.time_to_x(self.playhead)
        if x0 - 10 <= x <= x1 + 10:
            p.setPen(QPen(QColor(c["danger"]), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setBrush(QColor(c["danger"]))
            p.drawPolygon([QPointF(x - 6, 0), QPointF(x + 6, 0), QPointF(x, 10)])

        # the HUD that follows the pointer
        if g is not None and g.hud:
            self._paint_hud(p, c, fm, g.hud, g.hud_pos, bad=g.reason is not None)
        elif drop is not None and (drop.reason or drop.ok):
            self._paint_hud(p, c, fm, self._drop_hud(drop), drop.pos, bad=not drop.ok)

    def _hud_rect(self, text: str, pos: QPointF) -> QRectF:
        fm = QFontMetrics(self.font())
        w = fm.horizontalAdvance(text) + 14
        x, y = pos.x() + 14, pos.y() + 18
        sa = self.scroll_area()
        if sa is not None:
            right = sa.horizontalScrollBar().value() + sa.viewport().width()
            if x + w > right - 4:
                x = max(sa.horizontalScrollBar().value() + 4.0, pos.x() - w - 10)
        return QRectF(x, y, w, fm.height() + 8)

    def _paint_hud(self, p: QPainter, c, fm: QFontMetrics, text: str, pos: QPointF, *, bad: bool) -> None:
        r = self._hud_rect(text, pos)
        p.setPen(QPen(QColor("#ff6b6b") if bad else QColor(c["border"]), 1))
        p.setBrush(QColor(20, 20, 24, 235))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QColor("#ffb4b4") if bad else QColor("#ffffff"))
        p.drawText(r.adjusted(7, 0, -7, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def _tick_step(self, tb: Timebase | None = None) -> float:
        """Seconds between labelled ticks: whole frames at fine zoom (so the labels are exact timecode), then 1 s, 2 s, ... up to 20 min."""
        tb = tb or self.timebase()
        fine = [tb.frame_s * n for n in (1, 2, 3, 5, 10, 15) if tb.frame_s * n < 1.0 - 1e-9]
        for step in fine:
            if step * self.pps >= 96:
                return step
        for step in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
            if step * self.pps >= 70:
                return float(step)
        return 1200.0

    def _paint_clip(self, p: QPainter, c, fm: QFontMetrics, clip: Clip, index: int, track, selected: bool,
                    start: float | None = None, duration: float | None = None, ghost: bool = False, *, primary: bool = False, invalid: bool = False, dim: bool = False) -> None:
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
        alpha = 80 if track.hidden else 190 if ghost else 70 if dim else None
        if invalid:
            base, alpha = c["danger"], 170
        color = self._qcolor(base, alpha)
        r = self.clip_rect(clip, index, start, duration)
        missing = clip.asset_id in self._missing
        if r.width() < TINY_PX and not selected and not missing:
            p.fillRect(r, color)
            return
        p.setBrush(color)
        if invalid:
            pen = QPen(self._qcolor("#ff8a8a"), 2, Qt.PenStyle.DashLine)
        elif selected:
            pen = QPen(self._qcolor(c["selection"]), 3 if primary else 2)
        elif missing:
            pen = QPen(self._qcolor(c["danger"]), 2)
        else:
            pen = QPen(self._qcolor(base, alpha, 150), 1)
        p.setPen(pen)
        p.drawRoundedRect(r, 4, 4)
        if selected and not ghost:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._qcolor(c["selection"], 70 if primary else 45))
            p.drawRoundedRect(r, 4, 4)
            if primary and r.width() > 8:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255, 200), 1))
                p.drawRoundedRect(r.adjusted(2, 2, -2, -2), 3, 3)
        if track.collapsed or self.row_height(index) <= COLLAPSED_ROW_H:
            return  # a collapsed row is a thin colour bar: no labels, badges or waveform
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
        if clip.transition:
            tw = min(max(6.0, float((clip.transition or {}).get("duration", 0.5) or 0.5) * self.pps), r.width() / 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 80))
            p.drawPolygon([QPointF(r.left() + 1, r.bottom() - 1), QPointF(r.left() + 1 + tw, r.top() + 1), QPointF(r.left() + 1, r.top() + 1)])
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
        if decision is not None and decision.confidence < 70:
            tags.append("⚠")  # low-confidence AI decision
        if tags:
            label = " ".join(tags) + "  " + label
        p.setPen(self._qcolor("#ffffff"))
        p.save()
        p.setClipRect(r.adjusted(4, 0, -4, 0))
        p.drawText(r.adjusted(6, 0, -(4 + (14 if clip.protected else 0)), 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._elided(fm, label, int(r.width()) - 10 - (14 if clip.protected else 0)))
        p.restore()
        self._paint_badges(p, fm, clip, r, track, ghost)

    def _paint_badges(self, p: QPainter, fm: QFontMetrics, clip: Clip, r: QRectF, track, ghost: bool) -> None:
        """Lock, link, group, speed, volume / mute and keyframe marks of one clip (only drawn when the clip is wide enough to carry them)."""
        if clip.protected:  # solid lock = whole item locked; outlined lock = only some aspects (or protected from AI regeneration)
            x, y = r.right() - 14, r.top() + 3
            solid = "item" in clip.locks
            p.setPen(QPen(QColor(255, 255, 255, 235), 1.3))
            p.setBrush(QColor(255, 255, 255, 235) if solid else QColor(0, 0, 0, 90))
            p.drawRoundedRect(QRectF(x, y + 4, 10, 7), 1.5, 1.5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(x + 2, y, 6, 8), 0, 180 * 16)
        if clip.link_id or clip.group_id:
            p.save()
            by = r.bottom() - 11
            x = r.left() + 5
            if clip.link_id:
                col = self._id_color(clip.link_id)
                p.fillRect(QRectF(r.left() + 1, r.top() + 4, 3, r.height() - 8), col)  # the same colour tick on every member of the link
                p.setPen(QPen(col.lighter(130), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(QRectF(x, by + 2, 6, 5), 2.5, 2.5)
                p.drawRoundedRect(QRectF(x + 4, by + 2, 6, 5), 2.5, 2.5)
                x += 14
            if clip.group_id:
                col = self._id_color(clip.group_id)
                p.setPen(QPen(col.lighter(130), 1.2, Qt.PenStyle.DotLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(x, by, 8, 8))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col.lighter(130))
                p.drawEllipse(QPointF(x + 4, by + 4), 1.5, 1.5)
            p.restore()
        # speed / volume pills at the bottom right
        pills = []
        if abs(clip.speed - 1.0) > 1e-6:
            pills.append(f"{clip.speed:g}×")
        if clip.kind == KIND_MEDIA and (track.is_audio or clip.audio):
            vol = float(clip.audio.get("volume", 1.0))
            if clip.audio.get("muted") or vol <= 1e-4:
                pills.append("muted")
            elif abs(vol - 1.0) > 0.005:
                pills.append(f"{vol * 100:.0f}%")
        x = r.right() - 4
        for text in pills:
            w = fm.horizontalAdvance(text) + 8
            if x - w < r.left() + 28:
                break
            pr = QRectF(x - w, r.bottom() - 13, w, 11)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(pr, 3, 3)
            p.setPen(QColor("#ffcf6b") if text == "muted" else QColor("#ffffff"))
            p.drawText(pr, Qt.AlignmentFlag.AlignCenter, text)
            x -= w + 3
        # keyframe diamonds on the bottom edge (tall rows, or zoomed in)
        if clip.keyframes and r.height() >= 24 and (r.height() >= 40 or self.pps >= 100):
            size = 4.0 if r.height() >= 40 else 3.0
            p.setPen(QPen(QColor(0, 0, 0, 160), 1))
            p.setBrush(QColor("#ffd24d"))
            seen: set[int] = set()
            lo, hi = self._expose[0] - 6, self._expose[1] + 6
            y = r.bottom() - size - 1
            for kf in clip.keyframes[:400]:
                kx = r.left() + kf.time * self.pps
                if kx < r.left() or kx > r.right() or kx < lo or kx > hi or int(kx) in seen:
                    continue
                seen.add(int(kx))
                p.drawPolygon([QPointF(kx, y - size), QPointF(kx + size, y), QPointF(kx, y + size), QPointF(kx - size, y)])

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

    # ------------------------------------------------------------ snapping (the Phase 9 helpers stay for callers that want a fixed-pixel snap; gestures use ws.edit.snap)
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

    def _snap_mode(self, mods) -> tuple[bool, bool]:
        """(snap this gesture?, force it?). Alt inverts the snapping switch for one gesture: with snapping on it turns it off, with snapping off it snaps anyway."""
        enabled = self.ctx.ws.edit_state.snap.enabled
        if mods & Qt.KeyboardModifier.AltModifier:
            return (not enabled), (not enabled)
        return enabled, False

    def _snap_label(self, res: SnapResult) -> str:
        tl = self._timeline()
        kind = res.target_kind
        if kind == "markers" and tl is not None:
            m = tl.get_marker(res.target_id)
            return f"marker '{m.name}'" if m is not None and m.name else "marker"
        if kind in ("clip_edges", "captions"):
            clip = tl.get_clip(res.target_id) if tl is not None and res.target_id else None
            if clip is None:
                return "the timeline start"
            return ("caption edge" if kind == "captions" else "clip edge") + f" ({self._clip_name(clip)})"
        return {"playhead": "the playhead", "range": "the in / out range", "scenes": "a scene boundary", "sentences": "a sentence boundary", "words": "a word boundary"}.get(kind, kind)

    def _clip_name(self, clip: Clip) -> str:
        project = self.ctx.ws.project
        asset = project.assets.get(clip.asset_id) if project else None
        if asset is not None:
            return asset.name
        if clip.text:
            text = str(clip.text.get("text") or clip.text.get("content") or "").strip()
            if text:
                return f"“{text[:24]}”"
        return clip.kind

    # ------------------------------------------------------------ mouse: press
    def mousePressEvent(self, e) -> None:  # noqa: N802
        self.setFocus()
        self._remember(e)
        pos = e.position()
        btn = e.button()
        if btn == Qt.MouseButton.RightButton:
            self._context_menu(e)
            return
        if btn == Qt.MouseButton.MiddleButton:
            sa, outer = self.scroll_area(), self._outer_scroll_area()
            self._g = _Gesture("pan", QPointF(pos), moved=True, pan_origin=(e.globalPosition().toPoint(), sa.horizontalScrollBar().value() if sa else 0, outer.verticalScrollBar().value() if outer else 0))
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return
        if btn != Qt.MouseButton.LeftButton or self.ctx.ws.project is None:
            return
        if self._g is not None:
            self.cancel_gesture()  # a gesture whose release never arrived (focus lost, a dialog took the mouse) must not block the next press
        mods = e.modifiers()
        marker = self._hit_marker(pos)
        if marker is not None:  # a QC marker opens its issue (tested before the ruler's playhead drag)
            self.qc_selected = marker["issue_id"]
            self.set_playhead(marker["time"])
            self.marker_clicked.emit(marker["issue_id"])
            self.update()
            return
        if pos.y() < RULER_H:
            self._press_ruler(pos, mods)
            return
        if self.ctx.ws.edit_state.tool == "blade":
            self._press_blade(pos, mods)
            return
        hit = self._hit_clip(pos.x(), pos.y())
        self._sel_marker = None
        if hit is None:
            mode = "toggle" if mods & Qt.KeyboardModifier.ControlModifier else "add" if mods & Qt.KeyboardModifier.ShiftModifier else "replace"
            self._g = _Gesture("rubber", QPointF(pos), mods, cur=QPointF(pos), mode=mode)
            return
        self._press_clip(hit, pos, mods)

    def _press_ruler(self, pos: QPointF, mods) -> None:
        um = self._hit_user_marker(pos)
        if um is not None:
            self._sel_marker = um.id
            self._g = _Gesture("marker", QPointF(pos), mods, marker_id=um.id, value=um.time, edge_time=um.time, session=self._ws.edit.snap.session())
            self.update()
            return
        handle = self._hit_range_handle(pos)
        if handle is not None:
            r = self._range_now()
            assert r is not None
            self._g = _Gesture(handle, QPointF(pos), mods, value=r[0] if handle == "range_start" else r[1], edge_time=r[0] if handle == "range_start" else r[1],
                               other=r[1] if handle == "range_start" else r[0], session=self._ws.edit.snap.session())
            return
        self._sel_marker = None
        self._g = _Gesture("playhead", QPointF(pos), mods, moved=True, session=self._ws.edit.snap.session())
        self._auto.start()
        self._gesture_move(self._g, QPointF(pos), mods)

    def _press_blade(self, pos: QPointF, mods) -> None:
        """Blade click: cut the clip under the line on the hovered track (and its linked clips); Shift cuts every unlocked track."""
        idx = self.track_index_at(pos.y())
        if idx is None:
            return
        ws, tl = self._ws, self._timeline()
        t = self._blade_time(pos.x(), mods)
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self._edit(lambda: ws.edit.blade(t, None))
            return
        track = self.tracks()[idx]
        eps = self.timebase().frame_s * 0.5
        under = tl.clips_in_range(track.id, t - eps, t + eps)
        if not under:
            self.ctx.status("There is no clip under the blade on this track.")
            return
        # split() with the clip's id cuts the linked clips with it (blade() on a single track only cuts the clips lying on that track)
        target = next((c for c in under if c.timeline_start < t < c.timeline_end), under[0])
        self._edit(lambda: ws.edit.split(t, [target.id]))

    def _blade_time(self, x: float, mods) -> float:
        t = self.x_to_time(x)
        use, force = self._snap_mode(mods)
        if use:
            res = self._ws.edit.snap.snap(t, pps=self.pps, force=force)
            if res.snapped:
                return res.time
        return t

    def _press_clip(self, hit: tuple[Clip, int, str], pos: QPointF, mods) -> None:
        clip, idx, zone = hit
        ws, tl = self._ws, self._ws.project.timeline
        sel, st = ws.selection, ws.edit_state
        linked = st.linked_selection
        ctrl, shift, alt = (bool(mods & m) for m in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.AltModifier))
        if ctrl and alt:
            if zone == "move":
                self._begin_slide(clip, pos, mods)
            else:
                self._begin_roll(clip, "start" if zone == "trim_start" else "end", pos, mods)
            return
        if alt and shift and zone == "move":
            self._begin_slip(clip, pos, mods)
            return
        if shift:
            sel.extend_to(tl, clip.id, additive=ctrl)
            return
        toggle = ctrl
        collapse = False
        if not ctrl:
            if clip.id not in sel:
                sel.click(tl, clip.id, "replace", linked=linked, grouped=linked)
            else:
                collapse = True  # pressing a selected clip keeps the whole selection for a drag; a plain click on it narrows the selection to it
        if toggle and zone != "move":
            self._g = _Gesture("click", QPointF(pos), mods, clip_id=clip.id, toggle_on_release=True)
            return
        if zone == "move":
            self._begin_move(clip, idx, pos, mods, toggle, collapse)
        else:
            self._begin_trim(clip, idx, "start" if zone == "trim_start" else "end", pos, mods)

    # ------------------------------------------------------------ gesture setup
    def _begin_move(self, clip: Clip, idx: int, pos: QPointF, mods, toggle: bool, collapse: bool) -> None:
        ws = self._ws
        tl, sel, linked = ws.project.timeline, ws.selection, ws.edit_state.linked_selection
        sel_ids = [i for i in sel.ids if tl.get_clip(i) is not None]
        if clip.id not in sel_ids:
            sel_ids.append(clip.id)  # Ctrl+press on a clip that is not selected: it joins the drag (the selection itself changes on release / drag)
        try:
            ids = relatives(tl, sel_ids, linked=linked, grouped=linked)
        except TimelineError:
            ids = list(sel_ids)
        items = []
        for cid in ids:
            track, c = tl.find_clip(cid)
            items.append((cid, c.timeline_start, c.timeline_end, self._row_of(track.id)))
        g = _Gesture("move", QPointF(pos), mods, clip_id=clip.id, sel_ids=sel_ids, ids=ids, idset=frozenset(ids), items=items,
                     span0=(min(i[1] for i in items), max(i[2] for i in items)), primary_row=idx, session=ws.edit.snap.session(),
                     toggle_on_release=toggle, collapse_on_release=collapse)
        g.hud_pos = QPointF(pos)
        self._g = g

    def _begin_trim(self, clip: Clip, idx: int, edge: str, pos: QPointF, mods) -> None:
        ws = self._ws
        tl, linked = ws.project.timeline, ws.edit_state.linked_selection
        try:
            ids = relatives(tl, [clip.id], linked=linked, grouped=False)
        except TimelineError:
            ids = [clip.id]
        self._g = _Gesture("trim", QPointF(pos), mods, clip_id=clip.id, edge=edge, ids=ids, idset=frozenset(ids), primary_row=idx, session=ws.edit.snap.session(),
                           edge_time=clip.timeline_start if edge == "start" else clip.timeline_end)

    def _roll_pair(self, clip: Clip, edge: str) -> tuple[Clip, Clip] | None:
        """The two touching clips around the cut at ``clip``'s edge (left, right), or None when nothing touches that edge."""
        tl = self._timeline()
        fs = self.timebase().frame_s
        boundary = clip.timeline_start if edge == "start" else clip.timeline_end
        for other in tl.clips_in_range(clip.track_id, boundary - 1.5 * fs, boundary + 1.5 * fs):
            if other.id == clip.id:
                continue
            if edge == "end" and abs(other.timeline_start - boundary) <= fs + 1e-6:
                return clip, other
            if edge == "start" and abs(other.timeline_end - boundary) <= fs + 1e-6:
                return other, clip
        return None

    def _begin_roll(self, clip: Clip, edge: str, pos: QPointF, mods) -> None:
        ws = self._ws
        g = _Gesture("roll", QPointF(pos), mods, clip_id=clip.id, edge=edge, primary_row=self._row_of(clip.track_id), session=ws.edit.snap.session(),
                     edge_time=clip.timeline_start if edge == "start" else clip.timeline_end)
        pair = self._roll_pair(clip, edge)
        if pair is None:
            g.impossible = "There is no clip touching this edge, so there is no cut to roll. Ctrl+Alt+drag an edge between two clips that touch."
        else:
            g.pair = (pair[0].id, pair[1].id)
            g.ids = list(g.pair)
            g.idset = frozenset(relatives(ws.project.timeline, g.ids, linked=ws.edit_state.linked_selection, grouped=False))
            try:
                g.limits = ws.edit.roll_limits(*g.pair)
            except TimelineError as exc:
                g.impossible = exc.user_message
        self._g = g

    def _begin_slip(self, clip: Clip, pos: QPointF, mods) -> None:
        ws = self._ws
        g = _Gesture("slip", QPointF(pos), mods, clip_id=clip.id, ids=[clip.id], idset=frozenset([clip.id]), primary_row=self._row_of(clip.track_id))
        try:
            g.limits = ws.edit.slip_limits(clip.id)
        except TimelineError as exc:
            g.impossible = exc.user_message
        self._g = g

    def _begin_slide(self, clip: Clip, pos: QPointF, mods) -> None:
        ws = self._ws
        g = _Gesture("slide", QPointF(pos), mods, clip_id=clip.id, ids=[clip.id], idset=frozenset([clip.id]), primary_row=self._row_of(clip.track_id))
        try:
            g.limits = ws.edit.slide_limits(clip.id)
        except TimelineError as exc:
            g.impossible = exc.user_message
        self._g = g

    # ------------------------------------------------------------ mouse: move
    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self._remember(e)
        pos = e.position()
        g = self._g
        if g is None:
            self._hover(e, pos)
            return
        if g.kind == "pan":
            gp, h0, v0 = g.pan_origin
            d = e.globalPosition().toPoint() - gp
            sa, outer = self.scroll_area(), self._outer_scroll_area()
            if sa is not None:
                sa.horizontalScrollBar().setValue(h0 - d.x())
            if outer is not None:
                outer.verticalScrollBar().setValue(v0 - d.y())
            return
        if g.kind == "click":
            return
        if not g.moved:
            if abs(pos.x() - g.press.x()) < DRAG_PX and abs(pos.y() - g.press.y()) < DRAG_PX:
                return
            self._start_moving(g)
        self._gesture_move(g, QPointF(pos), e.modifiers())

    def _start_moving(self, g: _Gesture) -> None:
        g.moved = True
        if g.kind == "move" and g.toggle_on_release:  # Ctrl+drag: the clip is part of what is dragged; a drag is not a toggle
            ws = self._ws
            if g.clip_id not in ws.selection:
                ws.selection.click(ws.project.timeline, g.clip_id, "add", linked=ws.edit_state.linked_selection, grouped=ws.edit_state.linked_selection)
            g.toggle_on_release = False
        if g.kind in AUTO_KINDS:
            self._auto.start()
        self.update()

    def _gesture_move(self, g: _Gesture, pos: QPointF, mods) -> None:
        self._last_mods = mods
        g.hud_pos = QPointF(pos)
        if g.kind == "move":
            self._update_move(g, pos, mods)
        elif g.kind == "trim":
            self._update_trim(g, pos, mods)
        elif g.kind == "roll":
            self._update_roll(g, pos, mods)
        elif g.kind == "slip":
            self._update_slip(g, pos)
        elif g.kind == "slide":
            self._update_slide(g, pos)
        elif g.kind == "rubber":
            g.cur = QPointF(pos)
            g.hud = ""
        elif g.kind == "playhead":
            self._update_playhead(g, pos, mods)
        elif g.kind == "marker":
            self._update_marker(g, pos, mods)
        elif g.kind in ("range_start", "range_end"):
            self._update_range(g, pos, mods)
        self._invalidate(g)

    def _invalidate(self, g: _Gesture) -> None:
        """Repaint only what the gesture touched: the old and the new bounds of its ghosts, guide line, row highlight, rubber band and HUD."""
        r = QRectF()
        for _state, row, s, d in g.ghosts:
            r = r.united(QRectF(self.time_to_x(s), self.row_rect_y(row), max(2.0, d * self.pps), self.row_height(row)))
        if g.guide is not None:
            r = r.united(QRectF(self.time_to_x(g.guide[0]) - 3, 0, 6, self.height()))
        if g.target_row is not None and g.target_row < self.track_layout.count:
            r = r.united(QRectF(0, self.row_rect_y(g.target_row), self.width(), self.row_height(g.target_row)))
        if g.kind == "rubber":
            r = r.united(QRectF(g.press, g.cur).normalized().adjusted(-2, -2, 2, 2))
        if g.kind in ("marker", "playhead", "range_start", "range_end"):
            r = r.united(QRectF(self.time_to_x(g.value) - 3, 0, 150, RULER_H))
            r = r.united(QRectF(self.time_to_x(g.value) - 3, 0, 6, self.height()))
        if g.hud:
            r = r.united(self._hud_rect(g.hud, g.hud_pos))
        if g.first and g.kind in MOVE_KINDS:
            g.first = False
            self.update()  # the originals turn translucent: repaint everything once
            g.bounds = r
            return
        full = r.united(g.bounds)
        g.bounds = r
        if not full.isNull():
            self.update(full.toAlignedRect().adjusted(-4, -4, 4, 4))

    # ---- move
    def _row_move(self, g: _Gesture, y: float) -> tuple[int, dict | None, str]:
        """Row delta of the pointer relative to the grabbed clip's row, the track moves it implies and the reason when vertical movement has to be ignored."""
        target = self.track_layout.nearest_index(y)
        if target is None:
            return 0, None, ""
        rd = target - g.primary_row
        if rd == 0:
            return 0, None, ""
        hit = g.row_cache.get(rd)
        if hit is None:
            hit = g.row_cache[rd] = self._plan_rows(g, rd)
        return hit

    def _plan_rows(self, g: _Gesture, rd: int) -> tuple[int, dict | None, str]:
        tl, tracks = self._timeline(), self.tracks()
        project = self._ws.project
        rows = {i[0]: i[3] for i in g.items}
        moves: dict[str, str] = {}
        for cid in g.sel_ids:
            row = rows.get(cid, g.primary_row) + rd
            if not 0 <= row < len(tracks):
                return 0, None, "There is no track there, so the movement stays on the current track."
            clip = tl.get_clip(cid)
            track = tracks[row]
            asset = project.assets.get(clip.asset_id) if clip is not None else None
            if clip is None or not clip_fits_track(clip, track, asset.type.value if asset else None):
                who = "these clips" if len(g.sel_ids) > 1 else (f"“{self._clip_name(clip)}”" if clip else "this clip")
                return 0, None, f"Track “{track.name}” cannot hold {who}, so the movement stays on the current track."
            moves[cid] = track.id
        return rd, moves, ""

    def _update_move(self, g: _Gesture, pos: QPointF, mods) -> None:
        ws, tb = self._ws, self.timebase()
        st = ws.edit_state
        s0, e0 = g.span0
        start = max(0.0, s0 + (pos.x() - g.press.x()) / self.pps)  # the whole selection stops at 0:00
        snap_label = ""
        g.guide = None
        use, force = self._snap_mode(mods)
        if use and g.session is not None:
            sp = g.session.snap_span(start, start + (e0 - s0), exclude_ids=g.idset, pps=self.pps, force=force)
            if sp.snapped:
                start = sp.start
                g.guide = (sp.result.target_time if sp.result.target_time is not None else sp.result.time, sp.result.target_kind, sp.result.target_id)
                snap_label = self._snap_label(sp.result)
        delta = start - s0
        copy = bool(mods & Qt.KeyboardModifier.ControlModifier)
        rd, moves, vnote = self._row_move(g, pos.y())
        if copy and rd:
            rd, moves, vnote = 0, None, "Copies stay on their tracks."
        ripple = bool(st.ripple_on) and not copy
        key = (round(delta, 7), rd, copy, ripple)
        if key != g.key:  # an identical pointer state (the pointer is held by a snap) costs nothing
            g.key = key
            g.delta, g.row_delta, g.track_moves, g.copy, g.ripple = delta, rd, moves, copy, ripple
            g.plan, g.reason = self._preview_move(g)
            g.ghosts = self._move_ghosts(g)
        g.target_row = g.primary_row + g.row_delta if g.row_delta else None
        parts = [("Copy " if g.copy else "") + format_delta(tb, g.delta)]
        if snap_label:
            parts.append("snaps to " + snap_label)
        if g.row_delta:
            parts.append(self.tracks()[g.primary_row + g.row_delta].name)
        if vnote:
            parts.append(vnote)
        if g.ripple and g.reason is None:
            parts.append("ripple" + (f" ({g.plan.shifted} clips shift)" if g.plan is not None and g.plan.shifted else ""))
        if g.reason:
            parts.append(g.reason)
        g.hud = " · ".join(parts)

    def _preview_move(self, g: _Gesture) -> tuple[EditPlan | None, str | None]:
        ws = self._ws
        ed, st = ws.edit, ws.edit_state
        if g.copy:
            linked = st.linked_selection
            return _dry(ws, lambda: ob.plan_duplicate(ed.env(), list(g.sel_ids), offset=g.delta, linked=linked, grouped=linked))
        tracks = {self._timeline().find_clip(i)[0].id for i in g.sel_ids if self._timeline().get_clip(i)}
        if g.ripple and not g.track_moves and len(tracks) == 1:  # the same decision ws.edit.move_clips makes: a ripple move is a single-track move
            linked = st.linked_selection
            return _dry(ws, lambda: orp.plan_ripple_move(ed.env(), list(g.sel_ids), g.delta, scope=st.ripple_scope, selected_track_ids=sorted(tracks), linked=linked, grouped=linked))
        return ed.preview_move(g.sel_ids, g.delta, g.track_moves)

    def _move_ghosts(self, g: _Gesture) -> list:
        plan = g.plan
        if plan is not None and not plan.empty:
            states = list(plan.adds) if g.copy else [s for cid, s in plan.updates.items() if cid in g.idset]
            if states:
                return [(s, self._row_of(s.track_id), s.timeline_start, s.duration) for s in states]
        tl = self._timeline()
        out = []
        for cid, s, e, row in g.items:
            clip = tl.get_clip(cid)
            if clip is None:
                continue
            nrow = min(max(row + (g.row_delta if cid in g.sel_ids else 0), 0), max(0, self.track_layout.count - 1))
            out.append((clip, nrow, s + g.delta, e - s))
        return out

    # ---- trim
    def _update_trim(self, g: _Gesture, pos: QPointF, mods) -> None:
        ws, tb = self._ws, self.timebase()
        t = max(0.0, g.edge_time + (pos.x() - g.press.x()) / self.pps)  # the edge follows how far the pointer moved, not where it is (the grab point is a few pixels off the edge)
        g.guide = None
        label = ""
        use, force = self._snap_mode(mods)
        if use and g.session is not None:
            res = g.session.snap(t, exclude_ids=g.idset, pps=self.pps, force=force)
            if res.snapped:
                t = res.time
                g.guide = (res.target_time if res.target_time is not None else res.time, res.target_kind, res.target_id)
                label = self._snap_label(res)
        key = (round(t, 7), ws.edit_state.ripple_on)
        if key != g.key:
            g.key = key
            g.value = t
            g.plan, g.reason = ws.edit.preview_trim(g.clip_id, g.edge, t)
            g.ghosts = self._plan_ghosts(g)
        parts = [f"Trim {g.edge}", tb.format(self._trimmed_edge(g, t)), f"({format_delta(tb, self._trimmed_edge(g, t) - g.edge_time)})"]
        out = [" ".join(parts)]
        if label:
            out.append("snaps to " + label)
        if ws.edit_state.ripple_on:
            out.append("ripple")
        if g.reason:
            out.append(g.reason)
        elif g.plan is not None and g.plan.notes:
            out.append(g.plan.notes[0])
        g.hud = " · ".join(out)

    def _trimmed_edge(self, g: _Gesture, fallback: float) -> float:
        """Where the trimmed edge ends up according to the engine's plan (it may clamp), else the requested time."""
        state = g.plan.updates.get(g.clip_id) if g.plan is not None else None
        if state is None:
            return fallback
        return state.timeline_start if g.edge == "start" else state.timeline_end

    def _plan_ghosts(self, g: _Gesture) -> list:
        plan = g.plan
        if plan is None or plan.empty:
            return []
        keep = g.idset or None
        return [(s, self._row_of(s.track_id), s.timeline_start, s.duration) for cid, s in plan.updates.items() if keep is None or cid in keep]

    # ---- roll
    def _update_roll(self, g: _Gesture, pos: QPointF, mods) -> None:
        ws, tb = self._ws, self.timebase()
        g.guide = None
        if g.impossible or g.pair is None or g.limits is None:
            g.reason, g.ghosts, g.plan = g.impossible or "This cut cannot be rolled.", [], None
            g.hud = "Roll: " + g.reason
            return
        t = max(0.0, g.edge_time + (pos.x() - g.press.x()) / self.pps)
        label = ""
        use, force = self._snap_mode(mods & ~Qt.KeyboardModifier.AltModifier)  # Ctrl+Alt is the roll gesture itself, so Alt cannot also mean "invert snapping" here
        if use and g.session is not None:
            res = g.session.snap(t, exclude_ids=g.idset, pps=self.pps, force=force)
            if res.snapped:
                t = res.time
                g.guide = (res.target_time if res.target_time is not None else res.time, res.target_kind, res.target_id)
                label = self._snap_label(res)
        lo, hi = g.limits
        t = min(max(t, lo), hi)  # the engine tells how far the cut may go
        if g.key != (round(t, 7),):
            g.key = (round(t, 7),)
            g.value = t
            left, right = g.pair
            linked = ws.edit_state.linked_selection
            g.plan, g.reason = _dry(ws, lambda: orp.plan_roll(ws.edit.env(), left, right, t, linked=linked))
            g.ghosts = self._plan_ghosts(g)
        base = self._timeline().get_clip(g.pair[0]).timeline_end
        parts = [f"Roll {tb.format(t)} ({format_delta(tb, t - base)})"]
        if label:
            parts.append("snaps to " + label)
        parts.append(g.reason or (g.plan.notes[0] if g.plan is not None and g.plan.notes else f"the cut can move between {tb.format(lo)} and {tb.format(hi) if hi < 1e9 else 'the end of the media'}"))
        g.hud = " · ".join(parts)

    # ---- slip
    def _update_slip(self, g: _Gesture, pos: QPointF) -> None:
        ws, tb = self._ws, self.timebase()
        g.guide = None
        if g.impossible or g.limits is None:
            g.reason, g.ghosts, g.plan = g.impossible or "This clip cannot be slipped.", [], None
            g.hud = "Slip: " + g.reason
            return
        clip = self._timeline().get_clip(g.clip_id)
        lo, hi = g.limits
        dsrc = min(max(-(pos.x() - g.press.x()) / self.pps * (clip.speed if clip else 1.0), lo), hi)  # dragging right shows earlier media
        if g.key != (round(dsrc, 7),):
            g.key = (round(dsrc, 7),)
            g.value = dsrc
            linked = ws.edit_state.linked_selection
            g.plan, g.reason = _dry(ws, lambda: orp.plan_slip(ws.edit.env(), g.clip_id, dsrc, linked=linked))
            g.ghosts = self._plan_ghosts(g)
        g.hud = f"Slip {format_delta(tb, g.value)}" + (f" · {g.reason}" if g.reason else " · the content moves under the cut; the clip stays where it is")

    # ---- slide
    def _update_slide(self, g: _Gesture, pos: QPointF) -> None:
        ws, tb = self._ws, self.timebase()
        g.guide = None
        if g.impossible or g.limits is None:
            g.reason, g.ghosts, g.plan = g.impossible or "This clip cannot be slid.", [], None
            g.hud = "Slide: " + g.reason
            return
        lo, hi = g.limits
        d = min(max((pos.x() - g.press.x()) / self.pps, lo), hi)
        if g.key != (round(d, 7),):
            g.key = (round(d, 7),)
            g.value = d
            linked = ws.edit_state.linked_selection
            g.plan, g.reason = _dry(ws, lambda: orp.plan_slide(ws.edit.env(), g.clip_id, d, linked=linked))
            g.ghosts = self._plan_ghosts(g)
        g.hud = f"Slide {format_delta(tb, g.value)}" + (f" · {g.reason}" if g.reason else " · the neighbours absorb the move")

    # ---- ruler gestures
    def _snapped_time(self, g: _Gesture, x: float, mods, *, skip=frozenset(), exclude=frozenset()) -> tuple[float, SnapResult | None]:
        t = self.x_to_time(x)
        use, force = self._snap_mode(mods)
        if use and g.session is not None:
            res = g.session.snap(t, exclude_ids=exclude, pps=self.pps, skip_kinds=skip, force=force)
            if res.snapped:
                g.guide = (res.target_time if res.target_time is not None else res.time, res.target_kind, res.target_id)
                return res.time, res
        g.guide = None
        return t, None

    def _update_playhead(self, g: _Gesture, pos: QPointF, mods) -> None:
        t, res = self._snapped_time(g, pos.x(), mods, skip=frozenset({"playhead"}))
        if res is None or res.frame_aligned:
            t = self.timebase().snap(t)
        g.value = t
        g.hud = ""
        self.set_playhead(t)

    def _update_marker(self, g: _Gesture, pos: QPointF, mods) -> None:
        t, res = self._snapped_time(g, self.time_to_x(g.edge_time) + (pos.x() - g.press.x()), mods, exclude=frozenset({g.marker_id}))  # follows how far the pointer moved, not where it is
        if res is None or res.frame_aligned:
            t = self.timebase().snap(t)
        g.value = t
        g.hud = f"Marker {self.timebase().format(t)}" + (f" · snaps to {self._snap_label(res)}" if res is not None else "")

    def _update_range(self, g: _Gesture, pos: QPointF, mods) -> None:
        tb = self.timebase()
        t, res = self._snapped_time(g, self.time_to_x(g.edge_time) + (pos.x() - g.press.x()), mods, exclude=frozenset({"range_in", "range_out"}))
        if res is None or res.frame_aligned:
            t = tb.snap(t)
        t = max(0.0, min(t, g.other - tb.frame_s)) if g.kind == "range_start" else max(t, g.other + tb.frame_s)
        g.value = t
        a, b = (t, g.other) if g.kind == "range_start" else (g.other, t)
        g.hud = f"Range {tb.format(a)} - {tb.format(b)} ({format_delta(tb, b - a).lstrip('+')})"

    # ------------------------------------------------------------ mouse: hover
    def _hover(self, e, pos: QPointF) -> None:
        tool = self.ctx.ws.edit_state.tool
        marker = self._hit_marker(pos)
        if marker is not None:
            QToolTip.showText(e.globalPosition().toPoint(), f"{marker['severity'].title()} · {marker['title']}  (confidence {marker['confidence']:.0f}%) — from the last QC run", self)
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            return
        um = self._hit_user_marker(pos)
        hover_id = um.id if um is not None else None
        if hover_id != self._hover_marker:
            self._hover_marker = hover_id
            self.update(QRect(0, 0, self.width(), RULER_H))
        if um is not None:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            return
        if pos.y() < RULER_H:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor if self._hit_range_handle(pos) else Qt.CursorShape.ArrowCursor))
            return
        if tool == "blade":
            old = self._blade
            idx = self.track_index_at(pos.y())
            self._blade = None if idx is None else (idx, self._blade_time(pos.x(), e.modifiers()))
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            if old != self._blade:
                for b in (old, self._blade):
                    if b is not None and b[0] < self.track_layout.count:
                        self.update(QRect(int(self.time_to_x(b[1])) - 3, int(self.row_rect_y(b[0])), 7, self.row_height(b[0]) + 1))
            return
        hit = self._hit_clip(pos.x(), pos.y())
        alt_ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier) and bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
        self.setCursor(QCursor(Qt.CursorShape.SplitHCursor if hit and hit[2] != "move" and alt_ctrl else Qt.CursorShape.SizeHorCursor if hit and hit[2] != "move" else
                               Qt.CursorShape.OpenHandCursor if hit else Qt.CursorShape.ArrowCursor))

    def leaveEvent(self, e) -> None:  # noqa: N802
        if self._blade is not None:
            self._blade = None
            self.update()
        super().leaveEvent(e)

    # ------------------------------------------------------------ mouse: release / cancel
    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        g = self._g
        if g is None:
            return
        if g.kind == "pan":
            if e.button() == Qt.MouseButton.MiddleButton:
                self._g = None
                self._refresh_cursor()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._remember(e)
        if g.moved:
            self._gesture_move(g, QPointF(e.position()), e.modifiers())  # the final pointer position decides
        self._g = None
        self._auto.stop()
        self.unsetCursor()
        self._refresh_cursor()
        try:
            self._finish(g)
        finally:
            self.update()

    def _finish(self, g: _Gesture) -> None:
        ws = self._ws
        tl = ws.project.timeline if ws.project else None
        if tl is None:
            return
        if not g.moved:
            if g.kind == "move" or g.kind == "click":
                if g.toggle_on_release:
                    ws.selection.click(tl, g.clip_id, "toggle", linked=ws.edit_state.linked_selection, grouped=ws.edit_state.linked_selection)
                elif g.collapse_on_release:
                    ws.selection.click(tl, g.clip_id, "replace", linked=ws.edit_state.linked_selection, grouped=ws.edit_state.linked_selection)
            elif g.kind == "rubber":
                if g.mode == "replace":
                    ws.selection.select_none()  # a click on empty space clears the selection
            elif g.kind == "marker":
                m = tl.get_marker(g.marker_id)
                if m is not None:
                    self.set_playhead(m.time)  # a click on a marker moves the playhead there
            return
        kind = g.kind
        if kind == "rubber":
            self._finish_rubber(g)
        elif kind == "move":
            self._finish_move(g)
        elif kind in ("trim", "roll", "slip", "slide"):
            self._finish_edit(g)
        elif kind == "marker":
            m = tl.get_marker(g.marker_id)
            if m is not None and abs(g.value - m.time) > 1e-9:
                self._ok(lambda: ws.edit.move_marker(g.marker_id, g.value))
        elif kind in ("range_start", "range_end"):
            a, b = (g.value, g.other) if kind == "range_start" else (g.other, g.value)
            if tl.range is None or abs(a - tl.range[0]) > 1e-9 or abs(b - tl.range[1]) > 1e-9:
                self._ok(lambda: ws.edit.set_range(a, b))

    def _finish_rubber(self, g: _Gesture) -> None:
        ws, tl = self._ws, self._ws.project.timeline
        r = QRectF(g.press, g.cur).normalized()
        first, last = self.track_layout.nearest_index(r.top()), self.track_layout.nearest_index(r.bottom())
        if first is None or last is None:
            return
        ws.selection.select_rect(tl, self.x_to_time(r.left()), self.x_to_time(r.right()), first, last, g.mode)

    def _finish_move(self, g: _Gesture) -> None:
        ws = self._ws
        if g.reason:
            self.ctx.status(g.reason)  # nothing is committed
            return
        if abs(g.delta) < 1e-9 and not g.row_delta:
            return
        if g.copy:
            plan = self._run(lambda: ws.edit.duplicate(g.sel_ids, offset=g.delta))
            if plan is not None and plan.adds:
                ws.selection.select([c.id for c in plan.adds], "replace")
            return
        self._run(lambda: ws.edit.move_clips(g.sel_ids, g.delta, g.track_moves or None))

    def _finish_edit(self, g: _Gesture) -> None:
        ws = self._ws
        if g.reason:
            self.ctx.status(g.reason)
            return
        if g.plan is None or g.plan.empty:
            return
        if g.kind == "trim":
            self._run(lambda: ws.edit.trim(g.clip_id, g.edge, g.value))
        elif g.kind == "roll" and g.pair:
            self._run(lambda: ws.edit.roll(g.pair[0], g.pair[1], g.value))
        elif g.kind == "slip":
            self._run(lambda: ws.edit.slip(g.clip_id, g.value))
        elif g.kind == "slide":
            self._run(lambda: ws.edit.slide(g.clip_id, g.value))

    def cancel_gesture(self) -> bool:
        """Abandon the running mouse gesture without changing anything (Escape). Returns True when there was one."""
        g, self._g = self._g, None
        self._auto.stop()
        if g is None:
            return False
        self._refresh_cursor()
        self.update()
        return True

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        pos = e.position()
        um = self._hit_user_marker(pos)
        if um is not None:
            self.edit_marker_dialog(um.id)
            return
        hit = self._hit_clip(pos.x(), pos.y())
        if hit:
            if hit[0].kind == "media":
                self.clip_double_clicked.emit(hit[0].asset_id)

    # ------------------------------------------------------------ wheel / keyboard
    def wheelEvent(self, e) -> None:  # noqa: N802
        mods = e.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            pos = self.mapTo(self.scroll_area().viewport(), e.position().toPoint()) if self.scroll_area() is not None else None
            self.zoom_at(self.pps * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15), self.x_to_time(e.position().x()), pos.x() if pos is not None else None)  # zoom around the pointer
            e.accept()
        elif mods & Qt.KeyboardModifier.ShiftModifier and self.scroll_area() is not None:
            bar = self.scroll_area().horizontalScrollBar()
            dy = e.angleDelta().y() or e.angleDelta().x()
            bar.setValue(bar.value() - int(dy / 120.0 * max(40, bar.singleStep() * 3)))
            e.accept()
        else:
            e.ignore()  # let the scroll area scroll

    def keyPressEvent(self, e) -> None:  # noqa: N802
        key, mods = e.key(), e.modifiers()
        if key in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            self._modifiers_changed(mods)
            super().keyPressEvent(e)
            return
        shift, ctrl = bool(mods & Qt.KeyboardModifier.ShiftModifier), bool(mods & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Escape:
            if self.escape():
                e.accept()
                return
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            marker = self._marker_target()
            if marker is not None:
                self.delete_marker(marker)
            else:
                self.delete_selection(ripple=shift)
            e.accept()
            return
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            sign = -1 if key == Qt.Key.Key_Left else 1
            if ctrl:
                self.jump_to_edge(sign)
            else:
                self.nudge(sign * (10 if shift else 1))
            e.accept()
            return
        elif key == Qt.Key.Key_A and ctrl:
            self.select_all()
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            self._modifiers_changed(e.modifiers())
        super().keyReleaseEvent(e)

    def _modifiers_changed(self, mods) -> None:
        """A modifier was pressed or released mid-gesture (Alt = snapping, Ctrl = copy): recompute the gesture from the pointer where it is."""
        g = self._g
        if g is not None and g.moved and g.kind in ("move", "trim", "roll", "playhead", "marker", "range_start", "range_end"):
            self._gesture_move(g, self._scene_now(), mods)

    # ------------------------------------------------------------ editing API (every method wraps ws.edit in ctx.guard and returns whether it did something)
    def _run(self, fn, *, modal: bool = False, title: str = "") -> EditPlan | None:
        """Run an engine call that returns an ``EditPlan``: the plan when something changed, None when it was refused (the reason is already in the status line) or changed nothing."""
        box: list = []
        ok = self.ctx.guard(self, lambda: box.append(fn()), modal=modal, title=title)
        return box[0] if ok and box else None

    def _edit(self, fn, *, modal: bool = False, title: str = "") -> bool:
        return self._run(fn, modal=modal, title=title) is not None

    def _ok(self, fn, *, modal: bool = False, title: str = "") -> bool:
        """Run an engine call whose success is 'did not raise' (markers, ranges, tracks)."""
        return self.ctx.guard(self, fn, modal=modal, title=title)

    def _selected_ids(self) -> list[str]:
        tl = self._timeline()
        return [i for i in self._ws.selection.ids if tl is not None and tl.get_clip(i) is not None]

    def _need_selection(self) -> list[str] | None:
        ids = self._selected_ids()
        if not ids:
            self.ctx.status("Select a clip first.")
            return None
        return ids

    def nudge(self, frames: int) -> bool:
        """Move the selection by whole frames (never rippling)."""
        ids = self._need_selection()
        if ids is None or not frames:
            return False
        delta = self.timebase().to_seconds(abs(int(frames))) * (1 if frames > 0 else -1)
        return self._edit(lambda: self._ws.edit.move_clips(ids, delta, None, ripple=False))

    def step_playhead(self, frames: int) -> bool:
        """Move the playhead by whole frames."""
        if self.ctx.ws.project is None:
            return False
        t = max(0.0, self.timebase().add_frames(self.playhead, int(frames)))
        if abs(t - self.playhead) < 1e-12:
            return False
        self.set_playhead(t)
        return True

    def jump_to_edge(self, direction: int) -> bool:
        """Playhead to the previous (-1) / next (+1) clip edge on any track."""
        if self.ctx.ws.project is None:
            return False
        snap = self._ws.edit.snap
        eps = self.timebase().frame_s * 0.5
        best = None
        for kind in ("clip_edges", "captions"):
            times = snap.target_times(kind)
            if direction > 0:
                i = bisect_right(times, self.playhead + eps)
                cand = times[i] if i < len(times) else None
                best = cand if cand is not None and (best is None or cand < best) else best
            else:
                i = bisect_left(times, self.playhead - eps) - 1
                cand = times[i] if i >= 0 else None
                best = cand if cand is not None and (best is None or cand > best) else best
        if best is None:
            self.ctx.status("There is no further edit in that direction.")
            return False
        self.set_playhead(best)
        return True

    def jump_to_marker(self, direction: int) -> bool:
        """Playhead to the previous (-1) / next (+1) marker."""
        if self.ctx.ws.project is None:
            return False
        m = self._ws.edit.next_marker() if direction > 0 else self._ws.edit.prev_marker()
        if m is None:
            self.ctx.status("There is no further marker in that direction.")
            return False
        self.set_playhead(m.time)
        return True

    def jump_to_start(self) -> bool:
        if self.ctx.ws.project is None:
            return False
        self.set_playhead(0.0)
        return True

    def jump_to_end(self) -> bool:
        if self.ctx.ws.project is None:
            return False
        self.set_playhead(self.ctx.ws.project.timeline.duration)
        return True

    def begin_blade_at_playhead(self) -> bool:
        """Cut every unlocked track at the playhead (the blade-all shortcut)."""
        return self._edit(lambda: self._ws.edit.blade(self.playhead, None))

    def split_at_playhead(self, ids: list[str] | None = None) -> bool:
        return self._edit(lambda: self._ws.edit.split(self.playhead, ids), modal=False)

    def delete_selection(self, ripple: bool | None = False) -> bool:
        """Delete the selected clips (``ripple=True`` closes the gap, ``False`` leaves it, ``None`` follows the ripple mode)."""
        ids = self._need_selection()
        return ids is not None and self._edit(lambda: self._ws.edit.delete(ids, ripple=ripple))

    def delete_selected(self) -> bool:  # kept for the toolbar button and older callers
        return self.delete_selection(ripple=False)

    def select_all(self) -> bool:
        tl = self._timeline()
        return tl is not None and bool(self._ws.selection.select_all(tl))

    def invert_selection(self) -> bool:
        tl = self._timeline()
        return tl is not None and bool(self._ws.selection.invert(tl))

    def select_track_clips(self, track_id: str, mode: str = "replace") -> bool:
        """Select every clip on one track."""
        box: list[bool] = []
        ok = self.ctx.guard(self, lambda: box.append(bool(self._ws.selection.select_track(self._timeline(), track_id, mode))))
        return ok and bool(box and box[0])

    def select_from_playhead(self, direction: int = 1, track_id: str | None = None) -> bool:
        """Select the clips from the playhead to the end (``direction`` > 0) or from the start up to the playhead (< 0), on one track or on all of them."""
        tl = self._timeline()
        if tl is None:
            return False
        ids = [track_id] if track_id else None
        t0, t1 = (self.playhead, tl.duration + 1.0) if direction > 0 else (0.0, self.playhead)
        return bool(self._ws.selection.select_range(tl, t0, t1, ids, "replace"))

    def copy_selection(self) -> bool:
        ids = self._need_selection()
        return ids is not None and self._ok(lambda: self._ws.edit.copy(ids))

    def cut_selection(self) -> bool:
        ids = self._need_selection()
        return ids is not None and self._edit(lambda: self._ws.edit.cut(ids))

    def paste_at_playhead(self) -> bool:
        return self._edit(lambda: self._ws.edit.paste(self.playhead))

    def duplicate_selection(self) -> bool:
        ids = self._need_selection()
        if ids is None:
            return False
        plan = self._run(lambda: self._ws.edit.duplicate(ids))
        if plan is not None and plan.adds:
            self._ws.selection.select([c.id for c in plan.adds], "replace")
        return plan is not None

    def escape(self) -> bool:
        """Escape: cancel the running gesture, else a library drop preview, else clear the marker / clip selection. Returns True when it did something."""
        if self.cancel_gesture():
            return True
        if self._drop is not None:
            self._drop = None
            self.update()
            return True
        if self._sel_marker is not None:
            self._sel_marker = None
            self.update()
            return True
        if self._ws.selection:
            self._ws.selection.select_none()
            return True
        return False

    # ---- markers and the range
    def _marker_target(self) -> str | None:
        """The marker the Delete key acts on: the one under the pointer, else the one selected by the last click."""
        tl = self._timeline()
        for mid in (self._hover_marker, self._sel_marker):
            if mid and tl is not None and tl.get_marker(mid) is not None:
                return mid
        return None

    def add_marker(self, t: float | None = None) -> bool:
        at = self.playhead if t is None else t
        box: list[str] = []
        ok = self.ctx.guard(self, lambda: box.append(self._ws.edit.add_marker(at)))
        if ok and box:
            self._sel_marker = box[0]
            self.update(QRect(0, 0, self.width(), RULER_H))
        return ok

    def delete_marker(self, marker_id: str) -> bool:
        ok = self._ok(lambda: self._ws.edit.delete_marker(marker_id))
        if ok:
            self._sel_marker = self._hover_marker = None
            self.update()
        return ok

    def edit_marker_dialog(self, marker_id: str) -> bool:
        """Open the marker editor; applies the changes (one undo step) when accepted."""
        tl = self._timeline()
        m = tl.get_marker(marker_id) if tl is not None else None
        if m is None:
            return False
        dlg = MarkerDialog(m, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        new = dlg.values()
        changes = {k: v for k, v in new.items() if v != getattr(m, k)}
        if not changes:
            return False
        return self._ok(lambda: self._ws.edit.edit_marker(marker_id, **changes), modal=True, title="Edit marker")

    def set_range_from_selection(self) -> bool:
        tl = self._timeline()
        span = self._ws.selection.span(tl) if tl is not None else None
        if not span:
            self.ctx.status("Select clips first: the range is set from them.")
            return False
        return self._ok(lambda: self._ws.edit.set_range(span[0], span[1]))

    def clear_range(self) -> bool:
        return self._ok(lambda: self._ws.edit.clear_range())

    # ------------------------------------------------------------ context menus
    def _act(self, menu: QMenu, text: str, fn, *, enabled: bool = True, tip: str = "", checkable: bool = False, checked: bool = False) -> QAction:
        act = menu.addAction(text)
        act.setEnabled(enabled)
        act.setToolTip(tip or text)
        if checkable:
            act.setCheckable(True)
            act.setChecked(checked)
        act.triggered.connect(lambda _checked=False: fn())
        return act

    def _lock_reason(self, clips: list[Clip]) -> str | None:
        tl = self._timeline()
        for c in clips:
            track = tl.get_track(c.track_id)
            if track.locked:
                return f"Track “{track.name}” is locked. Unlock it to edit."
            if "item" in c.locks:
                return f"“{self._clip_name(c)}” is locked. Unlock it first."
        return None

    def clip_menu(self, clip_id: str) -> QMenu:
        """The context menu of a clip, acting on the current selection (which includes the clip). Disabled entries say why in their tooltip."""
        ws, tl = self._ws, self._timeline()
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        clip = tl.get_clip(clip_id)
        if clip is None:
            return menu
        ids = self._selected_ids() or [clip_id]
        clips = [tl.get_clip(i) for i in ids]
        clips = [c for c in clips if c is not None]
        n = len(clips)
        locked = self._lock_reason(clips)
        ph = self.playhead
        edit = ws.edit
        self._act(menu, "Preview media", lambda: self.clip_double_clicked.emit(clip.asset_id), enabled=clip.kind == KIND_MEDIA, tip="Open the clip's source media in the preview." if clip.kind == KIND_MEDIA else "Only media clips have a source to preview.")
        under = [c for c in clips if c.timeline_start < ph < c.timeline_end]
        self._act(menu, "Split at playhead", lambda: self._edit(lambda: edit.split(ph, ids)), enabled=bool(under) and locked is None,
                  tip=locked or ("Cut the selected clips at the playhead." if under else "Move the playhead inside a selected clip to split it."))
        menu.addSeparator()
        self._act(menu, "Delete", lambda: self._edit(lambda: edit.delete(ids, ripple=False)), enabled=locked is None, tip=locked or "Delete the selected clips and leave a gap.")
        self._act(menu, "Ripple delete", lambda: self._edit(lambda: edit.delete(ids, ripple=True)), enabled=locked is None, tip=locked or "Delete the selected clips and close the gap.")
        self._act(menu, "Duplicate", self.duplicate_selection, tip="Copy the selected clips to just after them.")
        self._act(menu, "Copy", self.copy_selection)
        self._act(menu, "Cut", self.cut_selection, enabled=locked is None, tip=locked or "Copy the selected clips and delete them.")
        self._act(menu, "Paste", self.paste_at_playhead, enabled=edit.clipboard is not None, tip="Paste the copied clips at the playhead." if edit.clipboard is not None else "There is nothing to paste: copy or cut clips first.")
        self._act(menu, "Copy attributes", lambda: self._ok(lambda: edit.copy_attributes(clip_id)), tip="Remember this clip's look and audio settings.")
        self._act(menu, "Paste attributes", lambda: self._edit(lambda: edit.paste_attributes(ids)), enabled=locked is None, tip=locked or "Apply the copied attributes to the selected clips.")
        menu.addSeparator()
        self._act(menu, "Link", lambda: self._edit(lambda: edit.link(ids)), enabled=n >= 2, tip="Link the selected clips so they move, trim and cut together." if n >= 2 else "Select two or more clips to link them.")
        self._act(menu, "Unlink", lambda: self._edit(lambda: edit.unlink(ids)), enabled=any(c.link_id for c in clips), tip="Remove the link." if any(c.link_id for c in clips) else "These clips are not linked.")
        self._act(menu, "Group", lambda: self._edit(lambda: edit.group(ids)), enabled=n >= 2, tip="Group the selected clips so they are selected and moved together." if n >= 2 else "Select two or more clips to group them.")
        self._act(menu, "Ungroup", lambda: self._edit(lambda: edit.ungroup(ids)), enabled=any(c.group_id for c in clips), tip="Dissolve the group." if any(c.group_id for c in clips) else "These clips are not grouped.")
        all_locked = all("item" in c.locks for c in clips)
        self._act(menu, "Unlock clip" if all_locked else "Lock clip", lambda: self._edit(lambda: edit.toggle_item_lock(ids)), tip="Locked clips cannot be edited by hand and are protected from AI regeneration.")
        sub = menu.addMenu("Lock aspects")
        sub.setToolTipsVisible(True)
        for aspect in LOCK_KINDS:
            if aspect == "item":
                continue
            on = all(aspect in c.locks for c in clips)
            self._act(sub, aspect.capitalize(), lambda a=aspect, o=on: self._edit(lambda: edit.set_locks(ids, a, "remove" if o else "add")), checkable=True, checked=on,
                      tip={"position": "Stop the clip from being moved.", "timing": "Stop trimming, speed and source changes.", "transform": "Stop position / scale / rotation / opacity changes.",
                           "audio": "Stop volume, fade and pan changes.", "style": "Stop text, effect and transition changes."}[aspect])
        menu.addSeparator()
        speedable = clip.kind == KIND_MEDIA and (ws.project.assets.get(clip.asset_id) is not None and ws.project.assets.get(clip.asset_id).type is not AssetType.IMAGE)
        self._act(menu, "Speed…", lambda: self.request_speed_dialog.emit(clip_id), enabled=speedable and locked is None,
                  tip=locked or ("Change the playback speed of this clip." if speedable else "Speed applies to video and audio clips."))
        replaceable = clip.kind == KIND_MEDIA and n == 1
        self._act(menu, "Replace media…", lambda: self.request_replace_dialog.emit(clip_id), enabled=replaceable and locked is None,
                  tip=locked or ("Swap the source media, keeping the clip's place." if replaceable else "Select a single media clip to replace its media."))
        self._act(menu, "Properties", lambda: self.request_inspector.emit(clip_id), tip="Show this clip in the inspector.")
        menu.addSeparator()
        self._act(menu, "Select all on track", lambda: self.select_track_clips(clip.track_id), tip="Select every clip on this track.")
        self._act(menu, "Add marker at clip start", lambda: self.add_marker(clip.timeline_start), tip="Put a marker where this clip starts.")
        if clip.kind == KIND_CAPTION:
            caps = [c for c in clips if c.kind == KIND_CAPTION]
            self._act(menu, "Merge captions", lambda: self._edit(lambda: edit.merge_captions([c.id for c in caps])), enabled=len(caps) >= 2 and locked is None,
                      tip=locked or ("Join the selected captions into one." if len(caps) >= 2 else "Select two or more captions to merge them."))
        return menu

    def track_menu(self, track_index: int, t: float) -> QMenu:
        """The context menu of empty space on a track row."""
        ws = self._ws
        track = self.tracks()[track_index]
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        have = ws.edit.clipboard is not None
        why = f"Track “{track.name}” is locked. Unlock it to edit." if track.locked else ""
        self._act(menu, "Paste here", lambda: self._edit(lambda: ws.edit.paste(t, track.id)), enabled=have and not track.locked,
                  tip=why or ("Paste the copied clips starting here." if have else "There is nothing to paste: copy or cut clips first."))
        self._act(menu, "Add marker here", lambda: self.add_marker(t), tip="Put a marker at this time.")
        self._act(menu, "Select all on track", lambda: self.select_track_clips(track.id), enabled=bool(track.clips), tip="Select every clip on this track." if track.clips else "This track has no clips.")
        add = menu.addMenu("Add track…")
        for label, kind in (("Video track", TrackKind.VIDEO), ("Image track", TrackKind.IMAGE), ("Graphics track", TrackKind.GRAPHICS), ("Audio track", TrackKind.AUDIO)):
            self._act(add, label, lambda k=kind: self._ok(lambda: ws.edit.add_track(k), modal=True, title="Add track"))
        return menu

    def ruler_menu(self, t: float) -> QMenu:
        """The context menu of the ruler: markers and the in / out range."""
        ws, tl = self._ws, self._timeline()
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        self._act(menu, "Add marker here", lambda: self.add_marker(t), tip="Put a marker at this time.")
        span = ws.selection.span(tl) if tl is not None else None
        self._act(menu, "Set range from selection", self.set_range_from_selection, enabled=bool(span), tip="Mark the in / out range around the selected clips." if span else "Select clips first: the range is set from them.")
        has = tl is not None and tl.range is not None
        self._act(menu, "Clear range", self.clear_range, enabled=has, tip="Remove the in / out range." if has else "There is no range to clear.")
        return menu

    def marker_menu(self, marker_id: str) -> QMenu:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        self._act(menu, "Edit marker…", lambda: self.edit_marker_dialog(marker_id), tip="Change the marker's name, type, colour and note.")
        self._act(menu, "Delete marker", lambda: self.delete_marker(marker_id), tip="Remove this marker.")
        return menu

    def _show_menu(self, menu: QMenu, global_pos) -> None:
        menu.exec(global_pos)

    def _context_menu(self, e) -> None:
        pos = e.position()
        gp = e.globalPosition().toPoint()
        if self.ctx.ws.project is None:
            return
        if pos.y() < RULER_H:
            um = self._hit_user_marker(pos)
            if um is not None:
                self._sel_marker = um.id
                self._show_menu(self.marker_menu(um.id), gp)
            else:
                self._show_menu(self.ruler_menu(self.timebase().snap(self.x_to_time(pos.x()))), gp)
            return
        hit = self._hit_clip(pos.x(), pos.y())
        if hit:
            clip = hit[0]
            ws, tl = self._ws, self._ws.project.timeline
            if clip.id not in ws.selection:
                linked = ws.edit_state.linked_selection
                ws.selection.click(tl, clip.id, "replace", linked=linked, grouped=linked)
            self._show_menu(self.clip_menu(clip.id), gp)
            return
        idx = self.track_index_at(pos.y())
        if idx is not None:
            self._show_menu(self.track_menu(idx, self.timebase().snap(self.x_to_time(pos.x()))), gp)

    # ------------------------------------------------------------ tooltips
    def clip_tooltip(self, clip: Clip) -> str:
        """Everything worth knowing about a clip on hover: name, times, source, speed, track, locks, link / group and who made it."""
        ws, tl, tb = self._ws, self._timeline(), self.timebase()
        track = tl.get_track(clip.track_id)
        lines = [f"{self._clip_name(clip)}  ({clip.kind})", f"Track: {track.name}" + ("  (locked)" if track.locked else ""),
                 f"Start {tb.format(clip.timeline_start)}  ·  Duration {tb.format(clip.duration)}  ·  End {tb.format(clip.timeline_end)}"]
        if clip.kind == KIND_MEDIA:
            lines.append(f"Source in {tb.format(clip.source_in)}  ·  out {tb.format(clip.source_out)}")
        if abs(clip.speed - 1.0) > 1e-6:
            lines.append(f"Speed {clip.speed:g}×")
        if clip.transition:
            lines.append(f"Transition in: {str(clip.transition.get('type', '')).title()} {float(clip.transition.get('duration', 0.0)):.2f} s")
        if clip.locks:
            lines.append("Locked: " + ("the whole clip" if "item" in clip.locks else ", ".join(sorted(clip.locks))))
        if clip.locked:
            lines.append("Protected from AI regeneration")
        if clip.link_id:
            lines.append(f"Linked with {max(0, len(link_members(tl, clip.link_id)) - 1)} other clip(s)")
        if clip.group_id:
            lines.append(f"Grouped with {max(0, len(group_members(tl, clip.group_id)) - 1)} other clip(s)")
        if clip.asset_id in self._missing:
            lines.append("Media file is missing")
        owner = {"AI": "Created by the AI", "USER": "Edited by you" if clip.scene_id else "Created by you", "SYSTEM": "Created by the system"}.get(clip.created_by, clip.created_by)
        lines.append(owner)
        if ws.project is None:
            return lines[0]
        return "\n".join(lines)

    def marker_tooltip(self, m: Marker) -> str:
        label = MARKER_KINDS.get(m.kind, MARKER_KINDS["generic"])[0]
        out = [f"{m.name or label}  ·  {self.timebase().format(m.time)}", label]
        if m.note:
            out.append(m.note)
        return "\n".join(out)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip and self.ctx.ws.project is not None and self._g is None:
            pos = QPointF(e.pos())
            um = self._hit_user_marker(pos)
            if um is not None:
                QToolTip.showText(e.globalPos(), self.marker_tooltip(um), self)
                return True
            hit = self._hit_clip(pos.x(), pos.y()) if pos.y() >= RULER_H else None
            if hit is not None:
                QToolTip.showText(e.globalPos(), self.clip_tooltip(hit[0]), self)
                return True
            QToolTip.hideText()
            e.ignore()
            return True
        return super().event(e)

    # ------------------------------------------------------------ drop from library
    def _row_state(self, pos: QPointF) -> tuple[int | None, bool]:
        """(row under the pointer, over the 'new track' strip)."""
        idx = self.track_index_at(pos.y())
        if idx is not None:
            return idx, False
        return None, pos.y() >= self.track_layout.total_height - 1

    def _drop_state_at(self, pos: QPointF, mods, ids: list[str]) -> _DropState | None:
        ws, project = self._ws, self.ctx.ws.project
        if project is None or not ids:
            return None
        assets = [project.assets.get(i) for i in ids]
        if any(a is None for a in assets):
            return _DropState(None, False, "That media is no longer in the project.", asset_ids=ids, pos=QPointF(pos))
        row, strip = self._row_state(pos)
        if row is None and not strip:
            return None
        mode = "insert" if mods & Qt.KeyboardModifier.ControlModifier else "overwrite" if mods & Qt.KeyboardModifier.ShiftModifier else "free"
        at = self.x_to_time(pos.x())
        use, force = self._snap_mode(mods & ~Qt.KeyboardModifier.ShiftModifier)
        if use:
            res = ws.edit.snap.snap(at, pps=self.pps, force=force)
            if res.snapped:
                at = res.time
        at = self.timebase().snap(at)
        state = _DropState(row, True, "", [], mode, at, ids, QPointF(pos), strip=row is None)
        if row is None:  # a new track of the first item's kind; the other items must fit it too
            kind = {AssetType.AUDIO: TrackKind.AUDIO, AssetType.IMAGE: TrackKind.IMAGE}.get(assets[0].type, TrackKind.VIDEO)
            probe = Track("probe", "New track", kind)
            bad = next((a for a in assets if not probe.accepts(a.type)), None)
            t = at
            for a in assets:
                d = a.duration or 5.0
                state.ghosts.append((t, d))
                t += d
            if bad is not None:
                state.ok, state.reason = False, f"“{bad.name}” cannot go on the same new track as the first item."
            return state
        track = self.tracks()[row]
        plan, reason = _dry(ws, lambda: self._place_plan(ids, at, track, mode))
        if plan is not None and plan.adds:
            state.ghosts = [(c.timeline_start, c.duration) for c in plan.adds]
        else:
            t = at
            for a in assets:
                d = a.duration or 5.0
                state.ghosts.append((t, d))
                t += d
        if reason:
            state.ok, state.reason = False, reason
        return state

    def _place_plan(self, ids: list[str], at: float, track: Track, mode: str) -> EditPlan:
        """The plan ``ws.edit.place_assets`` would run (for the live preview and the red / green row; the drop itself calls the service)."""
        ed = self._ws.edit
        clips = ed.clips_for_assets(ids, track, at)
        env, st = ed.env(), ed.state
        if mode == "insert":
            return orp.plan_ripple_insert(env, clips, at, track.id, scope=st.ripple_scope if st.ripple_on else "track", selected_track_ids=[track.id])
        if mode == "overwrite":
            return ob.plan_overwrite(env, clips)
        return ob.plan_insert_free(env, clips, at, track.id)

    def _drop_hud(self, d: _DropState) -> str:
        if not d.ok:
            return d.reason
        what = f"{len(d.asset_ids)} items" if len(d.asset_ids) > 1 else "1 item"
        where = "a new track" if d.row is None else self.tracks()[d.row].name
        return f"{self.timebase().format(d.at)} · {what} → {where}" + {"insert": " · insert (ripple)", "overwrite": " · overwrite", "free": ""}[d.mode]

    def _drop_dirty(self) -> None:
        self.update()

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasFormat(ASSET_MIME) and self.ctx.ws.project is not None:
            e.acceptProposedAction()

    def dragMoveEvent(self, e) -> None:  # noqa: N802
        ids = asset_ids_from_mime(e.mimeData())
        self._drop = self._drop_state_at(e.position(), e.modifiers(), ids)
        if self._drop is not None and self._drop.ok:
            e.acceptProposedAction()
        else:
            e.ignore()
        self._drop_dirty()

    def dragLeaveEvent(self, e) -> None:  # noqa: N802
        self._drop = None
        self.update()

    def dropEvent(self, e) -> None:  # noqa: N802
        ids = asset_ids_from_mime(e.mimeData())
        state = self._drop_state_at(e.position(), e.modifiers(), ids)
        self._drop = None
        self.update()
        if state is None:
            e.ignore()
            return
        if not state.ok:
            self.ctx.status(state.reason)  # nothing was added
            e.ignore()
            return
        ws = self._ws
        if state.row is None:
            done = self._drop_on_new_track(state)
        else:
            track_id = self.tracks()[state.row].id
            done = self._edit(lambda: ws.edit.place_assets(ids, state.at, track_id, mode=state.mode))
        if done:
            e.acceptProposedAction()
        else:
            e.ignore()

    def _drop_on_new_track(self, state: _DropState) -> bool:
        """Create a track of the right kind, then place the media on it. If placing fails the new track is undone, so a failed drop changes nothing.
        (Two undo steps when it succeeds: the engine has no combined 'new track + place' call.)"""
        ws = self._ws
        first = ws.project.assets.get(state.asset_ids[0])
        kind = {AssetType.AUDIO: TrackKind.AUDIO, AssetType.IMAGE: TrackKind.IMAGE}.get(first.type, TrackKind.VIDEO)
        box: list[Track] = []
        if not self.ctx.guard(self, lambda: box.append(ws.edit.add_track(kind)), modal=True, title="Add track"):
            return False
        track = box[0]
        if self._edit(lambda: ws.edit.place_assets(state.asset_ids, state.at, track.id, mode="free")):
            return True
        ws.commands.undo()  # the placement failed (the reason is in the status line): take the new track away again
        return False
