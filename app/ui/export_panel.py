"""Export page: settings, preflight check, render queue with real progress, history, proxies and preview.

The panel never builds FFmpeg commands or touches render files: everything goes through ``ws.render`` (the RenderService). Long work runs
in the render queue / background jobs; the UI only shows state and reacts to ``render.updated`` events.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.exceptions import AppError
from app.core.timecode import format_timecode
from app.rendering import presets as P
from app.rendering.models import PreflightReport, RenderStatus
from app.rendering.queue import RenderJob
from app.ui.context import UiContext
from app.ui.dialogs.relink_dialog import RelinkDialog

QUEUE_COLUMNS = ("Render", "Status", "Stage", "Progress", "Speed", "ETA", "Elapsed", "Size")
HISTORY_COLUMNS = ("When", "Status", "Output", "Size", "Timeline", "Settings")
PROXY_COLUMNS = ("Asset", "Size", "Status", "Resolution", "Proxy size")
STATUS_TEXT = {RenderStatus.QUEUED: "Queued", RenderStatus.RUNNING: "Running", RenderStatus.PAUSED: "Paused", RenderStatus.CANCELING: "Canceling…", RenderStatus.COMPLETED: "Completed",
               RenderStatus.FAILED: "FAILED", RenderStatus.CANCELED: "Canceled"}
QUALITY_LABELS = {"draft": "Draft", "standard": "Standard", "high": "High", "maximum": "Maximum", "custom": "Custom"}


def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{max(0, n) / 1024:.0f} KB"


def fmt_clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class ExportPanel(QWidget):
    return_to_editor = Signal()
    open_qc_requested = Signal()

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._loading = False
        self._output: Path | None = None
        self._current_job: str | None = None
        self._rows: dict[str, int] = {}
        self._report: PreflightReport | None = None
        self._allow_proxy: set[str] = set()
        self._preview_path: Path | None = None
        self._check_token = 0
        self._qc_pending = False  # quality control is running on behalf of "Start export": it continues by itself when the run finishes

        title = QLabel("EXPORT VIDEO")
        title.setObjectName("title")
        note = QLabel("The MP4 is an output. Your project and its editable timeline are never changed by an export.")
        note.setObjectName("muted")
        note.setWordWrap(True)

        # ---------------------------------------------------------------- settings form
        self.preset = QComboBox()
        self.preset.setObjectName("exportPreset")
        for pid, pr in P.EXPORT_PRESETS.items():
            self.preset.addItem(pr.name, pid)
            self.preset.setItemData(self.preset.count() - 1, pr.description, Qt.ItemDataRole.ToolTipRole)
        self.preset.addItem("Custom", "custom")
        self.resolution = QComboBox()
        self.resolution.setObjectName("exportResolution")
        for k, label in P.RESOLUTION_LABELS.items():
            self.resolution.addItem(label, k)
        self.fps = QComboBox()
        self.fps.setObjectName("exportFps")
        self.fps.addItem("Project FPS", 0)
        for f in P.FPS_CHOICES:
            self.fps.addItem(f"{f} FPS", f)
        self.quality = QComboBox()
        self.quality.setObjectName("exportQuality")
        for q in P.QUALITY_LEVELS:
            self.quality.addItem(QUALITY_LABELS[q], q)
        self.codec = QComboBox()
        self.codec.setObjectName("exportCodec")
        self.audio = QComboBox()
        self.audio.setObjectName("exportAudio")
        for a in P.AUDIO_CODECS:
            self.audio.addItem(P.AUDIO_LABELS[a], a)
        self.hardware = QComboBox()
        self.hardware.setObjectName("exportHardware")
        for h, label in (("auto", "Auto"), ("cpu", "CPU"), ("hardware", "Hardware")):
            self.hardware.addItem(label, h)
        self.container = QComboBox()
        self.container.setObjectName("exportContainer")
        for c in P.CONTAINERS:
            self.container.addItem(c.upper(), c)
        self.notice = QLabel("")
        self.notice.setObjectName("exportNotice")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #e5a24a;")
        # advanced (only meaningful for Custom quality)
        self.crf = QSpinBox()
        self.crf.setRange(0, 63)
        self.crf.setSpecialValueText("auto")
        self.bitrate = QSpinBox()
        self.bitrate.setRange(0, 200000)
        self.bitrate.setSuffix(" kbps")
        self.bitrate.setSpecialValueText("constant quality")
        self.audio_bitrate = QSpinBox()
        self.audio_bitrate.setRange(32, 640)
        self.audio_bitrate.setSuffix(" kbps")
        self.enc_preset = QComboBox()
        self.enc_preset.setEditable(True)
        self.enc_preset.addItems(["", "ultrafast", "veryfast", "fast", "medium", "slow", "slower"])
        self.use_proxies = QCheckBox("Export from proxy media (lower quality — not recommended)")
        self.use_proxies.setObjectName("exportUseProxies")
        adv = QGroupBox("Advanced")
        adv.setCheckable(True)
        adv.setChecked(False)
        self.advanced = adv
        af = QFormLayout()
        for label, w in (("CRF", self.crf), ("Video bitrate", self.bitrate), ("Audio bitrate", self.audio_bitrate), ("Encoder preset", self.enc_preset)):
            af.addRow(label, w)
        af.addRow("", self.use_proxies)
        self.adv_body = QWidget()
        self.adv_body.setLayout(af)
        adv_l = QVBoxLayout(adv)
        adv_l.addWidget(self.adv_body)
        adv.toggled.connect(self.adv_body.setVisible)
        self.adv_body.setVisible(False)

        form = QFormLayout()
        for label, w in (("Preset", self.preset), ("Resolution", self.resolution), ("FPS", self.fps), ("Quality", self.quality), ("Codec", self.codec), ("Audio", self.audio),
                         ("Hardware Acceleration", self.hardware), ("Container", self.container)):
            form.addRow(label, w)
        self.output_label = QLabel("")
        self.output_label.setObjectName("exportOutput")
        self.output_label.setWordWrap(True)
        self.choose_btn = QPushButton("Choose Location…")
        self.choose_btn.clicked.connect(self.choose_output)
        self.overwrite = QCheckBox("Replace the file if it exists")
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_label, 1)
        out_row.addWidget(self.choose_btn)

        # ---------------------------------------------------------------- preflight
        self.preflight_view = QLabel("")
        self.preflight_view.setObjectName("exportPreflight")
        self.preflight_view.setTextFormat(Qt.TextFormat.PlainText)
        self.preflight_view.setWordWrap(True)
        self.preflight_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.fix_btn = QPushButton("Fix Issues")
        self.fix_btn.setObjectName("exportFix")
        self.fix_btn.clicked.connect(self.fix_issues)
        self.recheck_btn = QPushButton("Check again")
        self.recheck_btn.clicked.connect(self.run_preflight)
        pre = QGroupBox("PREFLIGHT CHECK")
        pl = QVBoxLayout(pre)
        pl.addWidget(self.preflight_view)
        pr = QHBoxLayout()
        pr.addWidget(self.fix_btn)
        pr.addWidget(self.recheck_btn)
        pr.addStretch(1)
        pl.addLayout(pr)
        self.qc_line = QLabel("")
        self.qc_line.setObjectName("exportQcStatus")
        self.qc_line.setWordWrap(True)
        self.qc_line.setTextFormat(Qt.TextFormat.PlainText)
        self.qc_btn = QPushButton("Open AI Quality Control")
        self.qc_btn.setObjectName("exportOpenQc")
        self.qc_btn.clicked.connect(self.open_qc_requested.emit)
        qr = QHBoxLayout()
        qr.addWidget(self.qc_line, 1)
        qr.addWidget(self.qc_btn)
        pl.addLayout(qr)

        self.start_btn = QPushButton("START EXPORT")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_export)
        self.draft_btn = QPushButton("Draft export")
        self.draft_btn.setToolTip("A fast low-resolution render to check timing, captions, graphics and audio. Not production quality.")
        self.draft_btn.clicked.connect(self.start_draft)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn, 1)
        buttons.addWidget(self.draft_btn)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(title)
        ll.addWidget(note)
        ll.addLayout(form)
        ll.addWidget(self.notice)
        ll.addWidget(QLabel("Output"))
        ll.addLayout(out_row)
        ll.addWidget(self.overwrite)
        ll.addWidget(adv)
        ll.addWidget(pre)
        ll.addLayout(buttons)
        ll.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(380)

        # ---------------------------------------------------------------- result cards
        self.result_card = self._card("exportComplete")
        self.result_title = QLabel("Export Complete")
        self.result_title.setObjectName("title")
        self.result_info = QLabel("")
        self.result_info.setObjectName("exportCompleteInfo")
        self.result_info.setTextFormat(Qt.TextFormat.PlainText)
        self.render_check = QLabel("")
        self.render_check.setObjectName("exportRenderCheck")
        self.render_check.setWordWrap(True)
        self.render_check.setTextFormat(Qt.TextFormat.PlainText)
        self.open_video_btn, self.open_folder_btn, self.again_btn, self.back_btn = (QPushButton(t) for t in ("Open Video", "Open Folder", "Export Again", "Return to Editor"))
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.again_btn.clicked.connect(self._reset_cards)
        self.back_btn.clicked.connect(self.return_to_editor.emit)
        self._fill_card(self.result_card, self.result_title, self.result_info, (self.open_video_btn, self.open_folder_btn, self.again_btn, self.back_btn))
        self.result_card.layout().insertWidget(2, self.render_check)  # under the file facts: what the second QC pass found in the exported video
        self.fail_card = self._card("exportFailed")
        self.fail_title = QLabel("Render Failed")
        self.fail_title.setObjectName("title")
        self.fail_title.setStyleSheet("color: #ff6b6b;")
        self.fail_info = QLabel("")
        self.fail_info.setObjectName("exportFailedInfo")
        self.fail_info.setTextFormat(Qt.TextFormat.PlainText)
        self.fail_info.setWordWrap(True)
        self.retry_btn, self.cpu_btn, self.logs_btn, self.settings_btn, self.dismiss_btn = (QPushButton(t) for t in ("Retry", "Retry on CPU", "Open Logs", "Change Settings", "Cancel"))
        self.retry_btn.clicked.connect(lambda: self.retry(False))
        self.cpu_btn.clicked.connect(lambda: self.retry(True))
        self.logs_btn.clicked.connect(self.open_logs)
        self.settings_btn.clicked.connect(self._reset_cards)
        self.dismiss_btn.clicked.connect(self._reset_cards)
        self._fill_card(self.fail_card, self.fail_title, self.fail_info, (self.retry_btn, self.cpu_btn, self.logs_btn, self.settings_btn, self.dismiss_btn))

        # ---------------------------------------------------------------- queue tab
        self.queue_table = QTableWidget(0, len(QUEUE_COLUMNS))
        self.queue_table.setObjectName("renderQueue")
        self.queue_table.setHorizontalHeaderLabels(QUEUE_COLUMNS)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.queue_table.setColumnWidth(3, 140)
        self.queue_table.itemSelectionChanged.connect(self._show_detail)
        self.detail = QPlainTextEdit()
        self.detail.setObjectName("renderDetail")
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(170)
        self.cancel_btn, self.pause_btn, self.retry_sel_btn, self.log_sel_btn, self.remove_btn = (QPushButton(t) for t in ("Cancel", "Pause", "Retry", "Open Log", "Remove"))
        self.cancel_btn.clicked.connect(self.cancel_selected)
        self.pause_btn.clicked.connect(self.pause_selected)
        self.retry_sel_btn.clicked.connect(lambda: self.retry(False))
        self.log_sel_btn.clicked.connect(self.open_logs)
        self.remove_btn.clicked.connect(self.remove_selected)
        qb = QHBoxLayout()
        for b in (self.cancel_btn, self.pause_btn, self.retry_sel_btn, self.log_sel_btn, self.remove_btn):
            qb.addWidget(b)
        qb.addStretch(1)
        queue_page = QWidget()
        ql = QVBoxLayout(queue_page)
        ql.addWidget(self.result_card)
        ql.addWidget(self.fail_card)
        ql.addWidget(self.queue_table, 1)
        ql.addLayout(qb)
        ql.addWidget(self.detail)

        # ---------------------------------------------------------------- history tab
        self.history_table = QTableWidget(0, len(HISTORY_COLUMNS))
        self.history_table.setObjectName("renderHistory")
        self.history_table.setHorizontalHeaderLabels(HISTORY_COLUMNS)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hb = QHBoxLayout()
        self.h_open, self.h_folder, self.h_log, self.h_clear = (QPushButton(t) for t in ("Open Video", "Open Folder", "Open Log", "Clear cache"))
        self.h_open.clicked.connect(lambda: self._open_record("video"))
        self.h_folder.clicked.connect(lambda: self._open_record("folder"))
        self.h_log.clicked.connect(lambda: self._open_record("log"))
        self.h_clear.setToolTip("Delete cached video sections and audio mixes (exports, proxies and media are kept).")
        self.h_clear.clicked.connect(self.clear_cache)
        for b in (self.h_open, self.h_folder, self.h_log, self.h_clear):
            hb.addWidget(b)
        hb.addStretch(1)
        history_page = QWidget()
        hl = QVBoxLayout(history_page)
        hl.addWidget(self.history_table, 1)
        hl.addLayout(hb)

        # ---------------------------------------------------------------- proxies tab
        self.proxy_table = QTableWidget(0, len(PROXY_COLUMNS))
        self.proxy_table.setObjectName("proxyTable")
        self.proxy_table.setHorizontalHeaderLabels(PROXY_COLUMNS)
        self.proxy_table.verticalHeader().setVisible(False)
        self.proxy_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.proxy_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.proxy_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.proxy_res = QComboBox()
        self.proxy_res.setObjectName("proxyResolution")
        for r in P.PROXY_RESOLUTIONS:
            self.proxy_res.addItem(r, r)
        self.proxy_res.setCurrentText("720p")
        self.proxy_gen, self.proxy_cancel, self.proxy_retry, self.proxy_delete, self.proxy_regen = (QPushButton(t) for t in ("Generate Proxies", "Cancel", "Retry", "Delete Proxies", "Regenerate"))
        self.proxy_gen.clicked.connect(self.generate_proxies)
        self.proxy_cancel.clicked.connect(lambda: self.ctx.ws.render.proxies.cancel())
        self.proxy_retry.clicked.connect(self.retry_proxy)
        self.proxy_delete.clicked.connect(self.delete_proxies)
        self.proxy_regen.clicked.connect(self.regenerate_proxies)
        self.proxy_note = QLabel("Proxies make editing smooth. Exports always use the original media.")
        self.proxy_note.setObjectName("muted")
        pb = QHBoxLayout()
        pb.addWidget(QLabel("Proxy size"))
        pb.addWidget(self.proxy_res)
        for b in (self.proxy_gen, self.proxy_cancel, self.proxy_retry, self.proxy_delete, self.proxy_regen):
            pb.addWidget(b)
        pb.addStretch(1)
        proxy_page = QWidget()
        xl = QVBoxLayout(proxy_page)
        xl.addWidget(self.proxy_note)
        xl.addLayout(pb)
        xl.addWidget(self.proxy_table, 1)

        # ---------------------------------------------------------------- preview tab
        self.preview_mode = QComboBox()
        self.preview_mode.setObjectName("previewMode")
        for mid, m in ctx.ws.render.preview_modes.items():
            self.preview_mode.addItem(m.label, mid)
            self.preview_mode.setItemData(self.preview_mode.count() - 1, m.description, Qt.ItemDataRole.ToolTipRole)
        self.preview_mode.setCurrentIndex(1)
        self.preview_scope = QComboBox()
        self.preview_scope.setObjectName("previewScope")
        self.preview_btn = QPushButton("Render Preview")
        self.preview_btn.setObjectName("previewRender")
        self.preview_btn.clicked.connect(self.render_preview)
        self.preview_check = QPushButton("Check cache")
        self.preview_check.clicked.connect(self.check_preview_cache)
        self.preview_open = QPushButton("Play Preview")
        self.preview_open.setEnabled(False)
        self.preview_open.clicked.connect(self.open_preview)
        self.preview_status = QLabel("Preview sections are cached: editing one scene re-renders only that scene.")
        self.preview_status.setObjectName("previewStatus")
        self.preview_status.setWordWrap(True)
        pv = QHBoxLayout()
        for w in (QLabel("Mode"), self.preview_mode, QLabel("Scope"), self.preview_scope, self.preview_btn, self.preview_check, self.preview_open):
            pv.addWidget(w)
        pv.addStretch(1)
        preview_page = QWidget()
        vl = QVBoxLayout(preview_page)
        vl.addLayout(pv)
        vl.addWidget(self.preview_status)
        vl.addStretch(1)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("exportTabs")
        for page, name in ((queue_page, "Render Queue"), (history_page, "History"), (proxy_page, "Proxies"), (preview_page, "Preview")):
            self.tabs.addTab(page, name)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_scroll)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 760])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split)
        self._reset_cards()

        for w in (self.resolution, self.fps, self.quality, self.codec, self.audio, self.hardware, self.container, self.enc_preset):
            (w.currentIndexChanged if isinstance(w, QComboBox) and not w.isEditable() else w.currentTextChanged).connect(self._on_edited)
        self.preset.activated.connect(self._on_preset)
        for w in (self.crf, self.bitrate, self.audio_bitrate):
            w.valueChanged.connect(self._on_edited)
        self.use_proxies.toggled.connect(self._on_edited)
        ctx.bridge.on("render.updated", lambda p: self._on_job(p["job"]))
        ctx.bridge.on("render.history_changed", lambda p: self.refresh_history())
        ctx.bridge.on("proxy.changed", lambda p: self.refresh_proxies())
        ctx.bridge.on("qc.updated", self._on_qc_event)
        ctx.bridge.on("project.opened", lambda p: (setattr(self, "_qc_pending", False), self.refresh()))
        ctx.bridge.on("project.closed", lambda p: self._clear())

    # ================================================================ small builders
    @staticmethod
    def _card(name: str) -> QFrame:
        f = QFrame()
        f.setObjectName(name)
        f.setFrameShape(QFrame.Shape.StyledPanel)
        return f

    @staticmethod
    def _fill_card(card: QFrame, title: QLabel, info: QLabel, buttons) -> None:
        lay = QVBoxLayout(card)
        lay.addWidget(title)
        lay.addWidget(info)
        row = QHBoxLayout()
        for b in buttons:
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

    def _reset_cards(self) -> None:
        self.result_card.setVisible(False)
        self.fail_card.setVisible(False)

    def _clear(self) -> None:
        self._qc_pending = False
        self._current_job = None
        self.qc_line.setText("")
        self.render_check.setText("")
        self.queue_table.setRowCount(0)
        self._rows.clear()
        self.history_table.setRowCount(0)
        self.proxy_table.setRowCount(0)
        self.preflight_view.setText("")
        self._reset_cards()

    # ================================================================ settings <-> widgets
    def refresh(self) -> None:
        project = self.ctx.ws.project
        if project is None:
            self._clear()
            return
        self._load_settings()
        self.refresh_history()
        self.refresh_proxies()
        self._load_scopes()
        self._sync_queue()
        self._refresh_qc_line()
        self.run_preflight()

    def _refresh_qc_line(self) -> None:
        """One line about the quality-control state (the export gate): READY / AVAILABLE / BLOCKED, or that QC has not run for the current project."""
        ws = self.ctx.ws
        if ws.project is None:
            self.qc_line.setText("")
            return
        if self._qc_pending:
            self.qc_line.setText("Quality control is running before the export… it starts by itself when the check is done.")
            self.qc_line.setStyleSheet("")
            return
        try:
            g = ws.qc.export_gate()
        except Exception:  # noqa: BLE001  (informational only)
            self.qc_line.setText("")
            return
        if g.needs_run:
            self.qc_line.setText("Quality control: " + ("not run yet" if not ws.project.qc_runs else "the project changed since the last run") + (" — it runs automatically before export." if ws.project.qc_settings.run_before_export else "."))
            self.qc_line.setStyleSheet("")
        else:
            self.qc_line.setText("Quality control: " + g.decision.message)
            self.qc_line.setStyleSheet("color: #ff9d9d;" if g.decision.blocked else "color: #e5a24a;" if g.decision.status == "AVAILABLE" else "color: #6fcf97;")

    def _on_qc_event(self, payload: dict) -> None:
        """A QC run finished / was ignored / fixed, or the file check arrived: the gate line and the exported file's check follow at once."""
        if self.ctx.ws.project is None or payload.get("kind") == "progress":
            return
        if payload.get("kind") in ("run_finished", "run_canceled", "run_failed") and self._qc_pending:
            self._end_qc_wait()
        self._refresh_qc_line()
        self._refresh_render_check()

    def _end_qc_wait(self) -> None:
        self._qc_pending = False
        self.start_btn.setEnabled(bool(self._report is None or self._report.can_start))

    def _refresh_render_check(self) -> None:
        """The second QC pass (on the exported file) as one line on the result card: pending, or its verdict, or that it is switched off."""
        ws = self.ctx.ws
        job = ws.render.job(self._current_job) if (self._current_job and ws.project is not None) else None
        if job is None or job.status is not RenderStatus.COMPLETED or job.record.kind != "export":
            self.render_check.setText("")
            return
        res = ws.qc.render_results().get(job.id)
        if res:
            summary = res.get("summary", "")
            self.render_check.setText(f"File check: {str(res.get('status', '?')).title()}" + (f" — {summary}" if summary else "") + "  (details on the Quality page, “Rendered file”)")
            bad = str(res.get("status", "")).upper() in ("FAILED", "ERROR")
            self.render_check.setStyleSheet("color: #ff9d9d;" if bad else "color: #e5a24a;" if str(res.get("status", "")).upper() == "WARNINGS" else "color: #6fcf97;")
        elif ws.project.qc_settings.post_render_qc:
            self.render_check.setText("File check: checking the exported video…")
            self.render_check.setStyleSheet("")
        else:
            self.render_check.setText("File check: switched off in the QC settings.")
            self.render_check.setStyleSheet("")

    def _set(self, combo: QComboBox, data) -> None:
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _load_settings(self) -> None:
        s = self.ctx.ws.render.settings
        caps = self.ctx.ws.render.capabilities()
        self._loading = True
        try:
            self.codec.clear()
            for c in P.CODECS:
                info = caps["video_codecs"].get(c, {"available": True})
                self.codec.addItem(P.CODEC_LABELS[c] + ("" if info["available"] else "  (not available here)"), c)
            self._set(self.preset, s.preset_id if s.preset_id in P.EXPORT_PRESETS else "custom")
            self._set(self.resolution, s.resolution)
            self._set(self.fps, s.fps)
            self._set(self.quality, s.quality)
            self._set(self.codec, s.video_codec)
            self._set(self.audio, s.audio_codec)
            self._set(self.hardware, s.hardware_acceleration)
            self._set(self.container, s.container)
            self.crf.setValue(s.crf)
            self.bitrate.setValue(s.bitrate_kbps)
            self.audio_bitrate.setValue(s.audio_bitrate_kbps)
            self.enc_preset.setCurrentText(s.encoder_preset)
            self.use_proxies.setChecked(s.use_proxies)
            hw = [k for k, v in caps.get("hardware", {}).items() if v]
            self.hardware.setToolTip("Hardware encoders found: " + (", ".join(hw) if hw else "none — Auto uses the CPU") if caps.get("ffmpeg_ok") else "FFmpeg was not found")
            self.fps.setItemText(0, f"Project FPS ({self.ctx.ws.project.settings.fps})")  # type: ignore[union-attr]
        finally:
            self._loading = False
        self._update_output_label()
        self._update_notice(caps)

    def _gather(self) -> dict:
        return {"resolution": self.resolution.currentData(), "fps": self.fps.currentData(), "quality": self.quality.currentData(), "video_codec": self.codec.currentData(),
                "audio_codec": self.audio.currentData(), "hardware_acceleration": self.hardware.currentData(), "container": self.container.currentData(), "crf": self.crf.value(),
                "bitrate_kbps": self.bitrate.value(), "audio_bitrate_kbps": self.audio_bitrate.value(), "encoder_preset": self.enc_preset.currentText().strip(),
                "use_proxies": self.use_proxies.isChecked()}

    def _on_preset(self) -> None:
        pid = self.preset.currentData()
        if pid and pid != "custom":
            self.ctx.guard(self, lambda: self.ctx.ws.render.apply_preset(pid))
            self._load_settings()
            self.run_preflight()

    def _on_edited(self, *_a) -> None:
        if self._loading or self.ctx.ws.project is None:
            return
        self.ctx.guard(self, lambda: self.ctx.ws.render.update_settings(**self._gather()))
        self._loading = True
        self._set(self.preset, self.ctx.ws.render.settings.preset_id if self.ctx.ws.render.settings.preset_id in P.EXPORT_PRESETS else "custom")
        self._loading = False
        self._update_notice(self.ctx.ws.render.capabilities())
        self.run_preflight()

    def _update_notice(self, caps: dict) -> None:
        s = self.ctx.ws.render.settings
        msgs = []
        info = caps.get("video_codecs", {}).get(s.video_codec)
        if info is not None and not info["available"]:
            alts = [v["label"] for k, v in caps["video_codecs"].items() if v["available"]]
            msgs.append(f"{P.CODEC_LABELS[s.video_codec]} is not available in this FFmpeg build. Supported: {', '.join(alts) or 'none'}.")
        msgs += P.compatibility_problems(s)
        self.notice.setText("  ".join(msgs))

    # ================================================================ output
    def _update_output_label(self) -> None:
        if self._output is not None:
            self.output_label.setText(str(self._output))
        else:
            self.output_label.setText(f"{self.ctx.ws.render.output_dir()}  (name chosen automatically, never overwriting)")

    def choose_output(self) -> None:
        proj = self.ctx.ws.project
        if proj is None:
            return
        start = str(self._output or (self.ctx.ws.render.output_dir() / f"{proj.project_name}.{self.ctx.ws.render.settings.container}"))
        path, _ = QFileDialog.getSaveFileName(self, "Choose where to save the video", start, "Video (*.mp4 *.mkv *.webm)")
        if path:
            self._output = Path(path)
            self._update_output_label()
            self.run_preflight()

    # ================================================================ preflight
    def run_preflight(self) -> None:
        if self.ctx.ws.project is None:
            return
        self._check_token += 1
        token = self._check_token
        self.preflight_view.setText("Checking the project…")
        self.start_btn.setEnabled(False)

        def done(rep: PreflightReport) -> None:
            if token == self._check_token:
                self._show_report(rep)

        self.ctx.guard(self, lambda: self.ctx.ws.render.preflight_async(done, self._output, set(self._allow_proxy)))

    def _show_report(self, rep: PreflightReport) -> None:
        self._report = rep
        self.preflight_view.setText(rep.text().split("\n", 2)[2] if rep.text().count("\n") >= 2 else rep.text())
        self.start_btn.setEnabled(rep.can_start and not self._qc_pending)
        self.draft_btn.setEnabled(True)
        self.fix_btn.setVisible(bool(rep.errors))
        if rep.errors:
            self.preflight_view.setStyleSheet("color: #ff9d9d;")
        else:
            self.preflight_view.setStyleSheet("")

    # ================================================================ fix issues / missing media
    def fix_issues(self) -> None:
        rep = self._report
        if rep is None:
            return
        if rep.proxy_only_assets:
            self._ask_missing_original(rep.proxy_only_assets[0])
        elif rep.missing_assets:
            RelinkDialog(self.ctx, self).exec()
            self.run_preflight()
        else:
            self.tabs.setCurrentIndex(0)
            self.status("Open the Timeline page to fix the listed problems.")

    def _ask_missing_original(self, asset_id: str) -> None:
        a = self.ctx.ws.require_project().assets.require(asset_id)
        box = QMessageBox(self)
        box.setWindowTitle("Original media missing")
        box.setText(f"Original media missing.\n\nAsset:\n{a.name}\n\nProxy available.")
        locate = box.addButton("Locate Original", QMessageBox.ButtonRole.ActionRole)
        use = box.addButton("Use Proxy Anyway", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel Render", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is locate:
            RelinkDialog(self.ctx, self).exec()
        elif box.clickedButton() is use:
            self._allow_proxy.add(asset_id)
        self.run_preflight()

    def status(self, msg: str) -> None:
        self.ctx.status(msg)

    # ================================================================ start / queue
    def start_export(self) -> None:
        self._submit(draft=False)

    def start_draft(self) -> None:
        self._submit(draft=True)

    def _submit(self, draft: bool, *, qc_checked: bool = False, qc_override: bool = False) -> None:
        ws = self.ctx.ws
        if ws.project is None or (self._qc_pending and not draft):
            return
        self._reset_cards()
        if not draft and not qc_checked and ws.project.qc_settings.run_before_export:
            gate = ws.qc.export_gate()
            if gate.needs_run:  # QC has not looked at this version of the project: run it first, then continue exporting
                self.status("Running quality control before the export…")
                project_id = ws.project.project_id

                def after_qc(_run) -> None:
                    # the export continues only for the project that was checked, and only if it still is what QC looked at (an edit made while QC ran would export unchecked work)
                    self._end_qc_wait()
                    cur = self.ctx.ws.project
                    if cur is None or cur.project_id != project_id:
                        return
                    if self.ctx.ws.qc.export_gate().needs_run:
                        self._refresh_qc_line()
                        self.status("The project changed while quality control was running: press Start export again.")
                        return
                    self._submit(False, qc_checked=True)

                def qc_failed(job) -> None:
                    self._end_qc_wait()
                    self.status(f"Quality control failed: {job.error}")

                self._qc_pending = True
                try:
                    ws.qc.run_full_qc(trigger="export", on_done=after_qc, on_error=qc_failed)
                except AppError as exc:
                    self._qc_pending = False
                    QMessageBox.warning(self, "Quality control", exc.user_message)
                    return
                self.start_btn.setEnabled(False)
                self.qc_line.setText("Quality control is running before the export… it starts by itself when the check is done.")
                self.qc_line.setStyleSheet("")
                return

        def go() -> None:
            ws.render.update_settings(**self._gather())
            if draft:
                job = ws.render.start_draft()
            else:
                job = ws.render.start_export(self._output, overwrite=self.overwrite.isChecked(), allow_proxy_assets=set(self._allow_proxy), qc_override=qc_override)
            self._current_job = job.id
            self._on_job(job)
            self.tabs.setCurrentIndex(0)

        from app.services.render_service import PreflightFailed, QCGateBlocked

        try:
            go()
        except QCGateBlocked as exc:
            self._refresh_qc_line()
            self._show_blocked(exc, draft)
        except PreflightFailed as exc:
            self._show_report(exc.report)
            self.status(exc.user_message)
        except AppError as exc:
            QMessageBox.warning(self, "Cannot start the export", exc.user_message)

    def _show_blocked(self, exc, draft: bool) -> None:
        """EXPORT BLOCKED: list what blocks, offer Open Quality Control, and — only where the QC settings allow it and nothing is Critical — an explicit "Export anyway"."""
        ws = self.ctx.ws
        names = []
        for iid in exc.issue_ids[:8]:
            try:
                i = ws.qc.issue(iid)
                names.append(f"• {i.severity.value.title()} — {i.title}")
            except AppError:
                continue
        if len(exc.issue_ids) > 8:
            names.append(f"• … and {len(exc.issue_ids) - 8} more")
        text = exc.user_message + (chr(10) * 2 + chr(10).join(names) if names else "")
        box = QMessageBox(QMessageBox.Icon.Warning, "Export blocked", text, QMessageBox.StandardButton.NoButton, self)
        box.setObjectName("exportBlockedDialog")
        open_btn = box.addButton("Open Quality Control", QMessageBox.ButtonRole.AcceptRole)
        anyway = box.addButton("Export anyway", QMessageBox.ButtonRole.DestructiveRole) if exc.overridable else None
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_btn:
            self.open_qc_requested.emit()
        elif anyway is not None and clicked is anyway:
            self._submit(draft, qc_checked=True, qc_override=True)

    def _on_job(self, job: RenderJob) -> None:
        row = self._rows.get(job.id)
        if row is None:
            row = self.queue_table.rowCount()
            self.queue_table.insertRow(row)
            self._rows[job.id] = row
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(True)
            self.queue_table.setCellWidget(row, 3, bar)
            for c in (0, 1, 2, 4, 5, 6, 7):
                self.queue_table.setItem(row, c, QTableWidgetItem(""))
            self.queue_table.item(row, 0).setData(Qt.ItemDataRole.UserRole, job.id)
        p = job.progress
        self.queue_table.item(row, 0).setText(job.title)
        self.queue_table.item(row, 1).setText(STATUS_TEXT[job.status])
        self.queue_table.item(row, 2).setText(p.stage.value if job.status in (RenderStatus.RUNNING, RenderStatus.PAUSED, RenderStatus.CANCELING) else "")
        bar = self.queue_table.cellWidget(row, 3)
        bar.setValue(int(p.overall * 1000))  # type: ignore[union-attr]
        bar.setFormat(f"{p.overall * 100:.0f}%")  # type: ignore[union-attr]
        self.queue_table.item(row, 4).setText(f"{p.speed:.1f}x" if p.speed else "")
        self.queue_table.item(row, 5).setText(fmt_clock(p.eta) if job.status is RenderStatus.RUNNING else "")
        self.queue_table.item(row, 6).setText(fmt_clock(p.elapsed))
        self.queue_table.item(row, 7).setText(fmt_bytes(p.output_bytes) if p.output_bytes else "")
        sel = self._selected_job_id()
        if sel in (None, job.id):
            if sel is None:
                self.queue_table.selectRow(row)
            self._show_detail()
        if job.status.is_terminal and job.id == self._current_job:
            self._finished(job)
        self._buttons()

    def _sync_queue(self) -> None:
        for job in self.ctx.ws.render.jobs():
            self._on_job(job)

    def _selected_job_id(self) -> str | None:
        row = self.queue_table.currentRow()
        item = self.queue_table.item(row, 0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selected_job(self) -> RenderJob | None:
        jid = self._selected_job_id()
        return self.ctx.ws.render.job(jid) if jid else None

    def _buttons(self) -> None:
        job = self._selected_job()
        st = job.status if job else None
        self.cancel_btn.setEnabled(bool(job) and not st.is_terminal and st is not RenderStatus.CANCELING)  # type: ignore[union-attr]
        self.pause_btn.setEnabled(bool(job) and not st.is_terminal and st is not RenderStatus.CANCELING)  # type: ignore[union-attr]
        self.pause_btn.setText("Resume" if job and job.pause_event.is_set() else "Pause")
        self.retry_sel_btn.setEnabled(bool(job) and st in (RenderStatus.FAILED, RenderStatus.CANCELED))
        self.log_sel_btn.setEnabled(bool(job))
        self.remove_btn.setEnabled(bool(job) and st.is_terminal)  # type: ignore[union-attr]

    def _show_detail(self) -> None:
        job = self._selected_job()
        self._buttons()
        if job is None:
            self.detail.setPlainText("")
            return
        p = job.progress
        lines = []
        if job.status is RenderStatus.RUNNING or job.status is RenderStatus.PAUSED:
            lines.append("Rendering..." if p.stage.value == "Rendering" else f"{p.stage.value}...")
            lines.append("")
            if p.scene_total:
                lines.append(f"Scene {max(1, p.scene_index)} / {p.scene_total}")
            lines.append(f"Video composition: {p.video * 100:.0f}%" + (f"  (section {p.chunk_index}/{p.chunk_total}, {p.cached_chunks} reused)" if p.chunk_total else ""))
            lines.append("Audio mix: " + ("complete" if p.audio >= 1.0 else f"{p.audio * 100:.0f}%" if p.audio else "waiting"))
            lines.append(f"Encoding: {p.encode * 100:.0f}%")
            lines.append("")
            lines.append(f"Speed: {p.speed:.1f}x" if p.speed else "Speed: —")
            lines.append(f"ETA: {fmt_clock(p.eta)}" + ("  (estimated)" if p.eta is not None else ""))
            lines.append(f"Elapsed: {fmt_clock(p.elapsed)}   Output so far: {fmt_bytes(p.output_bytes)}")
        elif job.status is RenderStatus.COMPLETED:
            lines.append(f"Completed: {job.record.output_path}")
        elif job.status is RenderStatus.FAILED and job.error:
            lines += ["Render Failed", "", f"Stage: {job.error.stage}", f"Possible issue: {job.error.possible_issue or job.error.user_message}"]
        elif job.status is RenderStatus.CANCELED:
            lines.append("Canceled. The project and timeline are unchanged.")
        else:
            lines.append(STATUS_TEXT[job.status])
        self.detail.setPlainText("\n".join(lines))

    def _finished(self, job: RenderJob) -> None:
        if job.status is RenderStatus.COMPLETED and job.result is not None:
            r = job.result
            self.result_info.setText(f"File:\n{r.output_path.name}\n\nDuration:\n{format_timecode(r.duration)}\n\nResolution:\n{r.plan.output_resolution[0]}×{r.plan.output_resolution[1]}\n\n"
                                     f"FPS:\n{r.plan.fps}" + (f"\n\nNotes:\n" + "\n".join(r.warnings[:4]) if r.warnings else ""))
            self.fail_card.setVisible(False)
            self.result_card.setVisible(True)
            self._refresh_render_check()
        elif job.status is RenderStatus.FAILED and job.error is not None:
            e = job.error
            self.fail_info.setText(f"Stage:\n{e.stage}\n\nPossible issue:\n{e.possible_issue or e.user_message}\n\n{e.user_message}")
            self.cpu_btn.setVisible(bool(e.can_fallback_cpu))
            self.result_card.setVisible(False)
            self.fail_card.setVisible(True)

    # ================================================================ queue actions
    def cancel_selected(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self.ctx.ws.render.cancel(jid)

    def pause_selected(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        (self.ctx.ws.render.resume if job.pause_event.is_set() else self.ctx.ws.render.pause)(job.id)
        self._on_job(job)

    def remove_selected(self) -> None:
        jid = self._selected_job_id()
        if jid and self.ctx.ws.render.remove(jid):
            row = self._rows.pop(jid)
            self.queue_table.removeRow(row)
            self._rows = {self.queue_table.item(r, 0).data(Qt.ItemDataRole.UserRole): r for r in range(self.queue_table.rowCount())}
            self._buttons()

    def retry(self, cpu: bool) -> None:
        job = self._selected_job() or (self.ctx.ws.render.job(self._current_job) if self._current_job else None)
        if job is None:
            return
        try:
            new = self.ctx.ws.render.retry(job.id, cpu_fallback=cpu, allow_proxy_assets=set(self._allow_proxy))
        except AppError as exc:
            QMessageBox.warning(self, "Cannot retry", exc.user_message)
            return
        self._reset_cards()
        self._current_job = new.id
        self._on_job(new)

    def open_logs(self) -> None:
        job = self._selected_job() or (self.ctx.ws.render.job(self._current_job) if self._current_job else None)
        if job is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.ctx.ws.render.log_path(job.id))))

    def open_video(self) -> None:
        job = self.ctx.ws.render.job(self._current_job) if self._current_job else None
        if job is not None and job.output_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(job.output_path)))

    def open_folder(self) -> None:
        job = self.ctx.ws.render.job(self._current_job) if self._current_job else None
        if job is not None:
            self.ctx.ws.render.open_location(job.output_path)

    # ================================================================ history
    def refresh_history(self) -> None:
        if self.ctx.ws.project is None:
            return
        recs = list(reversed(self.ctx.ws.render.history()))
        self.history_table.setRowCount(len(recs))
        for r, rec in enumerate(recs):
            res = f"{rec.settings.get('resolution', '')} · {rec.settings.get('video_codec', '')} · {QUALITY_LABELS.get(rec.settings.get('quality', ''), '')}"
            vals = (rec.created_at.replace("T", " ")[:19], rec.status.title(), Path(rec.output_path).name if rec.output_path else "", fmt_bytes(rec.size_bytes) if rec.size_bytes else "",
                    f"v{rec.timeline_version} · {rec.timeline_hash[:8]}", res)
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, rec.render_id)
                self.history_table.setItem(r, c, item)

    def _selected_record(self):
        row = self.history_table.currentRow()
        item = self.history_table.item(row, 0) if row >= 0 else None
        rid = item.data(Qt.ItemDataRole.UserRole) if item else None
        return next((r for r in self.ctx.ws.render.history() if r.render_id == rid), None)

    def _open_record(self, what: str) -> None:
        rec = self._selected_record()
        if rec is None:
            return
        if what == "video" and rec.output_path and Path(rec.output_path).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(rec.output_path))
        elif what == "folder" and rec.output_path:
            self.ctx.ws.render.open_location(Path(rec.output_path))
        elif what == "log":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.ctx.ws.render.log_path(rec.render_id))))

    def clear_cache(self) -> None:
        n = self.ctx.ws.render.clear_render_cache()
        self.status(f"Removed {n} cached render file(s).")

    # ================================================================ proxies
    def refresh_proxies(self) -> None:
        project = self.ctx.ws.project
        if project is None:
            return
        videos = [a for a in project.assets.all() if a.type.value == "video"]
        recs = self.ctx.ws.render.proxies.records()
        self.proxy_table.setRowCount(len(videos))
        for r, a in enumerate(videos):
            rec = recs.get(a.id)
            vals = (a.name, f"{a.width}×{a.height}" if a.width else "", rec.proxy_status.title() if rec else "None", rec.proxy_resolution if rec else "",
                    fmt_bytes(rec.size_bytes) if rec and rec.size_bytes else "")
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, a.id)
                if c == 2 and rec and rec.error:
                    item.setToolTip(rec.error)
                self.proxy_table.setItem(r, c, item)
        s = self.ctx.ws.render.proxies.summary()
        self.proxy_note.setText(f"{s['ready']} proxy file(s) ready, {s['queued']} in progress, {s['failed']} failed ({fmt_bytes(s['bytes'])}). Exports always use the original media.")

    def _proxy_ids(self) -> list[str] | None:
        rows = sorted({i.row() for i in self.proxy_table.selectedItems()})
        ids = [self.proxy_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]
        return ids or None

    def generate_proxies(self) -> None:
        self.ctx.guard(self, lambda: self.ctx.ws.render.proxies.generate(self._proxy_ids(), self.proxy_res.currentData(), only_large=self._proxy_ids() is None), modal=True)
        self.refresh_proxies()

    def retry_proxy(self) -> None:
        for aid in self._proxy_ids() or [r.asset_id for r in self.ctx.ws.render.proxies.records().values() if r.proxy_status in ("FAILED", "CANCELED", "STALE")]:
            self.ctx.ws.render.proxies.retry(aid)

    def delete_proxies(self) -> None:
        self.ctx.ws.render.proxies.delete(self._proxy_ids())
        self.refresh_proxies()

    def regenerate_proxies(self) -> None:
        self.ctx.guard(self, lambda: self.ctx.ws.render.proxies.regenerate(self._proxy_ids(), self.proxy_res.currentData()), modal=True)
        self.refresh_proxies()

    # ================================================================ preview
    def _load_scopes(self) -> None:
        self.preview_scope.clear()
        self.preview_scope.addItem("Whole timeline", None)
        for sc in self.ctx.ws.require_project().scenes:
            self.preview_scope.addItem(f"Scene {sc.label}  ({format_timecode(sc.start)}–{format_timecode(sc.end)})", sc.id)

    def check_preview_cache(self) -> None:
        def go() -> None:
            plan = self.ctx.ws.render.preview_plan(self.preview_mode.currentData())
            self.preview_status.setText(f"{plan.cached_sections} of {len(plan.sections)} preview section(s) are cached; {len(plan.stale_sections)} would be rendered "
                                        f"({plan.width}×{plan.height}).")

        self.ctx.guard(self, go, modal=True)

    def render_preview(self) -> None:
        scope = self.preview_scope.currentData()
        self.preview_status.setText("Rendering the preview…")
        self.preview_btn.setEnabled(False)

        def done(res) -> None:
            self._preview_path = res.path
            self.preview_open.setEnabled(True)
            self.preview_btn.setEnabled(True)
            self.preview_status.setText(f"Preview ready ({res.rendered} section(s) rendered, {res.reused} reused from the cache"
                                        + (", proxy media used" if res.uses_proxy else "") + f"): {res.path.name}")

        def failed(job) -> None:
            self.preview_btn.setEnabled(True)
            self.preview_status.setText(job.error or "The preview could not be rendered.")

        self.ctx.guard(self, lambda: self.ctx.ws.render.build_preview(self.preview_mode.currentData(), [scope] if scope else None, done, failed), modal=True)

    def open_preview(self) -> None:
        if self._preview_path is not None and self._preview_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._preview_path)))

    # kept for the main window: the old page had a ``check`` entry point
    def check(self) -> None:
        self.refresh()
