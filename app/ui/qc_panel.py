"""AI QUALITY CONTROL page: run QC in the background, read the scores, review issues, preview / apply / undo fixes, ignore, compare runs, tune the settings.

The panel never analyses or edits anything itself: everything goes through ``ws.qc`` (the QCService). It shows findings; fixes are executed by the service as undoable
commands; the page stays responsive because QC runs as a background job and the panel is driven by ``qc.updated`` events (coalesced).
"""

from __future__ import annotations

import json

from PySide6.QtCore import QItemSelectionModel, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
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

from app.core.timecode import format_timecode
from app.qc.errors import QCError
from app.qc.issue_model import GROUP_LABELS, SCORE_GROUPS, CheckerState, FixRoute, IssueStatus, QCCategory, QCIssue
from app.qc.qc_engine import LABELS as CHECKER_LABELS
from app.qc.qc_engine import PIPELINE
from app.qc.settings import FIX_KINDS_CONFIRM, FIX_KINDS_SAFE, SETTING_DEFS
from app.qc.severity import BLOCK_LABELS, COLORS, ORDER, PLURAL, MEANING, Severity, confidence_label
from app.ui.context import UiContext

ISSUE_COLUMNS = ("Severity", "Scene", "Time", "Category", "Issue", "Confidence")
STAGE_COLUMNS = ("Stage", "Status", "Detail")
HISTORY_COLUMNS = ("Run", "When", "Score", "Status", "Critical", "Errors", "Warnings", "Trigger")
STATE_TEXT = {CheckerState.PENDING.value: "pending", CheckerState.RUNNING.value: "running", CheckerState.DONE.value: "✓", CheckerState.CACHED.value: "✓ (cached)",
              CheckerState.FAILED.value: "FAILED", CheckerState.SKIPPED.value: "skipped", CheckerState.CANCELED.value: "canceled"}
EXPORT_TEXT = {"READY": "READY FOR EXPORT", "AVAILABLE": "EXPORT AVAILABLE", "BLOCKED": "EXPORT BLOCKED"}
STATUS_TEXT = {"READY": "READY", "REVIEW": "REVIEW RECOMMENDED", "FIX_REQUIRED": "FIX REQUIRED", "BLOCKED": "EXPORT BLOCKED"}
STATUS_COLOR = {"READY": "#4cc38a", "REVIEW": "#e5c04a", "FIX_REQUIRED": "#ff8a3d", "BLOCKED": "#ff4d4d"}
try:
    from app.qc.fix_engine import NEEDS_CONFIRM
except ImportError:  # the fix engine is optional (analysis works without it)
    NEEDS_CONFIRM = "This fix changes your edit: confirm it to apply."
STATUS_FILTERS = (("Open issues", "open"), ("Fixed", "fixed"), ("Ignored", "ignored"), ("All statuses", "all"))
CONFIDENCE_FILTERS = (("Any confidence", ""), ("High (80%+)", "high"), ("Medium (55-79%)", "medium"), ("Low (below 55%)", "low"))
VISUAL_CATEGORIES = ("VISUAL_ACCURACY", "VISUAL_REPETITION", "CONTINUITY", "SCENE_COVERAGE", "FACT_REVIEW")
BATCHES = (("fixSafeAll", "Fix All Safe Issues", None), ("fixCaptionTiming", "Fix All Caption Timing", "sync.caption"), ("fixDucking", "Fix All Safe Audio Ducking", "audio.insufficient_ducking"),
           ("fixMargins", "Fix All Safe Margin Issues", "caption.safe_margin"))


def fmt_time(t: float | None) -> str:
    return "—" if t is None else format_timecode(t)


def ask_reason(parent: QWidget | None, title: str = "Ignore this issue") -> tuple[str, bool]:
    text, ok = QInputDialog.getText(parent, title, "Why are you keeping it? (optional, e.g. “Intentional recurring visual”)")
    return text.strip(), bool(ok)


class QCPanel(QWidget):
    open_timeline_requested = Signal(float, str)  # time, clip id
    open_scene_requested = Signal(str)  # scene id (Review page)
    replace_visual_requested = Signal(str)  # scene id
    search_again_requested = Signal(str)  # scene id
    open_settings_requested = Signal(str)

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._issues: dict[str, QCIssue] = {}
        self._selected: str | None = None
        self._history_pick: list[int] = []
        self._loading = False
        self._paused = False  # the page is not shown: events only mark it dirty (the main window calls refresh() when the page opens)
        self._dirty = False
        self._settings_key = ""  # what the settings editor was last loaded from: edits the user has not saved are never overwritten by a refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._refresh_now)

        title = QLabel("AI QUALITY CONTROL")
        title.setObjectName("title")
        note = QLabel("Checks the whole project before export: integrity, scene coverage, narration sync, visual accuracy, pacing, captions, audio, continuity and render readiness. "
                      "QC never changes your project by itself: every fix is previewed and can be undone.")
        note.setObjectName("muted")
        note.setWordWrap(True)

        # ---------------------------------------------------------------- run controls + progress
        self.run_btn = QPushButton("Run QC")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(lambda: self.run_qc(False))
        self.rerun_btn = QPushButton("Re-run everything")
        self.rerun_btn.setObjectName("qcRerun")
        self.rerun_btn.setToolTip("Ignore cached results and analyse the whole project again")
        self.rerun_btn.clicked.connect(lambda: self.run_qc(True))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("qcCancel")
        self.cancel_btn.clicked.connect(self.cancel_run)
        self.retry_btn = QPushButton("Retry failed checks")
        self.retry_btn.setObjectName("qcRetry")
        self.retry_btn.clicked.connect(self.retry_failed)
        self.scene_btn = QPushButton("Re-check selected scene")
        self.scene_btn.setObjectName("qcSceneRun")
        self.scene_btn.clicked.connect(self.run_scene)
        self.category = QComboBox()
        self.category.setObjectName("qcCategory")
        for cid in PIPELINE:
            self.category.addItem(CHECKER_LABELS.get(cid, cid), cid)
        self.category_btn = QPushButton("Re-run this check")
        self.category_btn.setObjectName("qcCategoryRun")
        self.category_btn.clicked.connect(self.run_category)
        self.progress = QProgressBar()
        self.progress.setObjectName("qcProgress")
        self.progress.setRange(0, 100)
        self.stage_label = QLabel("")
        self.stage_label.setObjectName("qcStage")
        self.stages = self._table("qcStages", STAGE_COLUMNS, 0)
        self.stages.setMaximumHeight(210)

        run_box = QGroupBox("RUN")
        rl = QVBoxLayout(run_box)
        r1 = QHBoxLayout()
        for b in (self.run_btn, self.rerun_btn, self.cancel_btn, self.retry_btn, self.scene_btn):
            r1.addWidget(b)
        r1.addStretch(1)
        r1.addWidget(self.category)
        r1.addWidget(self.category_btn)
        rl.addLayout(r1)
        pr = QHBoxLayout()
        pr.addWidget(self.progress, 1)
        pr.addWidget(self.stage_label)
        rl.addLayout(pr)
        rl.addWidget(self.stages)

        # ---------------------------------------------------------------- scores
        self.overall_label = QLabel("Not analysed yet")
        self.overall_label.setObjectName("qcOverall")
        self.overall_label.setStyleSheet("font-size: 22px; font-weight: 600;")
        self.status_label = QLabel("")
        self.status_label.setObjectName("qcStatus")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.export_label = QLabel("")
        self.export_label.setObjectName("qcExport")
        self.export_label.setWordWrap(True)
        self.count_labels: dict[str, QLabel] = {}
        counts = QHBoxLayout()
        for sev in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.NOTICE):
            lab = QLabel(f"{PLURAL[sev]}  0")
            lab.setObjectName(f"qcCount_{sev.value}")
            lab.setStyleSheet(f"color: {COLORS[sev]}; font-weight: 600;")
            lab.setToolTip(MEANING[sev])
            self.count_labels[sev.value] = lab
            counts.addWidget(lab)
        counts.addStretch(1)
        self.group_table = self._table("qcGroups", ("Category", "Score"), 0)
        self.group_table.setMaximumHeight(230)
        self.stale_label = QLabel("")
        self.stale_label.setObjectName("qcStale")
        self.stale_label.setWordWrap(True)
        self.stale_label.setStyleSheet("color: #e5a24a;")
        score_box = QGroupBox("RESULT")
        sl = QVBoxLayout(score_box)
        sl.addWidget(self.overall_label)
        sl.addWidget(self.status_label)
        sl.addWidget(self.export_label)
        sl.addLayout(counts)
        sl.addWidget(self.stale_label)
        sl.addWidget(self.group_table)

        # ---------------------------------------------------------------- issues
        self.sev_filter = QComboBox()
        self.sev_filter.setObjectName("qcSeverityFilter")
        self.sev_filter.addItem("All severities", "")
        for s in ORDER:
            self.sev_filter.addItem(PLURAL[s] if s is not Severity.CRITICAL else "Critical", s.value)
        self.sev_filter.currentIndexChanged.connect(self._fill_issues)
        self.cat_filter = QComboBox()
        self.cat_filter.setObjectName("qcCategoryFilter")
        self.cat_filter.addItem("All categories", "")
        for c in QCCategory:
            self.cat_filter.addItem(c.value.replace("_", " ").title(), c.value)
        self.scene_filter = QComboBox()
        self.scene_filter.setObjectName("qcSceneFilter")
        self.scene_filter.addItem("All scenes", "")
        self.status_filter = QComboBox()
        self.status_filter.setObjectName("qcStatusFilter")
        for label, val in STATUS_FILTERS:
            self.status_filter.addItem(label, val)
        self.conf_filter = QComboBox()
        self.conf_filter.setObjectName("qcConfidenceFilter")
        for label, val in CONFIDENCE_FILTERS:
            self.conf_filter.addItem(label, val)
        for f in (self.cat_filter, self.scene_filter, self.status_filter, self.conf_filter):
            f.currentIndexChanged.connect(self._fill_issues)
        # not shown: the status filter replaced it; kept so callers that toggle "show ignored / fixed" keep working (it drives the status filter)
        self.show_closed = QCheckBox("Show ignored / fixed")
        self.show_closed.setObjectName("qcShowClosed")
        self.show_closed.toggled.connect(self._on_show_closed)
        self.issues_hint = QLabel("")
        self.issues_hint.setObjectName("qcIssuesHint")
        self.issues_hint.setWordWrap(True)
        self.issues_hint.setStyleSheet("color: #9aa3b2;")
        self.issue_table = self._table("qcIssues", ISSUE_COLUMNS, 0)
        self.issue_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.issue_table.itemSelectionChanged.connect(self._on_select)
        self.issue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.issue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        issues_head = QHBoxLayout()
        issues_head.addWidget(QLabel("ISSUES"))
        issues_head.addStretch(1)
        issues_head.addWidget(self.sev_filter)
        issues_head.addWidget(self.cat_filter)
        issues_head.addWidget(self.scene_filter)
        issues_head.addWidget(self.status_filter)
        issues_head.addWidget(self.conf_filter)
        self.batch_btns: dict[str, QPushButton] = {}
        batch_row = QHBoxLayout()
        self.fix_selected_btn = QPushButton("Fix Selected")
        self.fix_selected_btn.setObjectName("qcFixSelected")
        self.fix_selected_btn.clicked.connect(self.fix_selected)
        batch_row.addWidget(self.fix_selected_btn)
        for oid, label, prefix in BATCHES:
            b = QPushButton(label)
            b.setObjectName(oid)
            b.clicked.connect(lambda _c=False, pre=prefix: self.fix_safe(pre))
            self.batch_btns[oid] = b
            batch_row.addWidget(b)
        batch_row.addStretch(1)
        self.report_btn = QPushButton("Export report…")
        self.report_btn.setObjectName("qcReport")
        self.report_btn.clicked.connect(self.export_report)
        batch_row.addWidget(self.report_btn)
        issues_box = QWidget()
        il = QVBoxLayout(issues_box)
        il.setContentsMargins(0, 0, 0, 0)
        il.addLayout(issues_head)
        il.addWidget(self.issues_hint)
        il.addWidget(self.issue_table, 1)
        il.addLayout(batch_row)

        # ---------------------------------------------------------------- issue detail
        self.detail = QPlainTextEdit()
        self.detail.setObjectName("qcDetail")
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("Select an issue to see what was detected, why it matters and how to fix it.")
        self.fix_btn = QPushButton("Fix")
        self.fix_btn.setObjectName("qcFix")
        self.fix_btn.clicked.connect(lambda: self.apply_fix(False))
        self.preview_btn = QPushButton("Preview Fix")
        self.preview_btn.setObjectName("qcPreview")
        self.preview_btn.clicked.connect(self.preview_fix)
        self.ignore_btn = QPushButton("Ignore")
        self.ignore_btn.setObjectName("qcIgnore")
        self.ignore_btn.clicked.connect(lambda: self.ignore(False))
        self.ignore_type_btn = QPushButton("Ignore This Type")
        self.ignore_type_btn.setObjectName("qcIgnoreType")
        self.ignore_type_btn.clicked.connect(lambda: self.ignore(True))
        self.similar_btn = QPushButton("Fix All Similar")
        self.similar_btn.setObjectName("qcFixSimilar")
        self.similar_btn.clicked.connect(self.fix_similar)
        self.timeline_btn = QPushButton("Open Timeline")
        self.timeline_btn.setObjectName("qcOpenTimeline")
        self.timeline_btn.clicked.connect(self.open_timeline)
        self.scene_open_btn = QPushButton("Open Scene")
        self.scene_open_btn.setObjectName("qcOpenScene")
        self.scene_open_btn.clicked.connect(self.open_scene)
        self.replace_btn = QPushButton("Replace Visual")
        self.replace_btn.setObjectName("qcReplaceVisual")
        self.replace_btn.clicked.connect(self.replace_visual)
        self.search_btn = QPushButton("Search Again")
        self.search_btn.setObjectName("qcSearchAgain")
        self.search_btn.clicked.connect(self.search_again)
        self.fix_note = QLabel("")
        self.fix_note.setObjectName("qcFixNote")
        self.fix_note.setWordWrap(True)
        self.fix_note.setTextFormat(Qt.TextFormat.PlainText)
        detail_box = QGroupBox("ISSUE")
        dl = QVBoxLayout(detail_box)
        dl.addWidget(self.detail, 1)
        dl.addWidget(self.fix_note)
        g = QGridLayout()
        for i, b in enumerate((self.fix_btn, self.preview_btn, self.ignore_btn, self.ignore_type_btn, self.similar_btn, self.timeline_btn, self.scene_open_btn, self.replace_btn, self.search_btn)):
            g.addWidget(b, i // 3, i % 3)
        dl.addLayout(g)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(issues_box)
        split.addWidget(detail_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setMinimumHeight(360)
        dash = QWidget()
        dv = QVBoxLayout(dash)
        dv.addWidget(run_box)
        dv.addWidget(score_box)
        dv.addWidget(split, 1)
        dash_scroll = QScrollArea()
        dash_scroll.setWidgetResizable(True)
        dash_scroll.setWidget(dash)
        dash_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # ---------------------------------------------------------------- history
        self.history_table = self._table("qcHistory", HISTORY_COLUMNS, 0)
        self.history_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.history_table.itemSelectionChanged.connect(self._on_history_select)
        self.compare_btn = QPushButton("Compare selected runs")
        self.compare_btn.setObjectName("qcCompare")
        self.compare_btn.clicked.connect(self.compare_runs)
        self.compare_view = QPlainTextEdit()
        self.compare_view.setObjectName("qcCompareView")
        self.compare_view.setReadOnly(True)
        self.compare_view.setMaximumHeight(180)
        self.ignored_table = self._table("qcIgnored", ("Ignored", "Scope", "Reason"), 0)
        self.unignore_btn = QPushButton("Stop ignoring")
        self.unignore_btn.setObjectName("qcUnignore")
        self.unignore_btn.clicked.connect(self.unignore)
        hist = QWidget()
        hl = QVBoxLayout(hist)
        hl.addWidget(QLabel("QC RUNS"))
        hl.addWidget(self.history_table, 1)
        hl.addWidget(self.compare_btn)
        hl.addWidget(self.compare_view)
        hl.addWidget(QLabel("IGNORED ISSUES (kept on purpose)"))
        hl.addWidget(self.ignored_table)
        hl.addWidget(self.unignore_btn)

        # ---------------------------------------------------------------- rendered file QC
        self.render_table = self._table("qcRender", ("Render", "Status", "Summary"), 0)
        self.render_detail = QPlainTextEdit()
        self.render_detail.setObjectName("qcRenderDetail")
        self.render_detail.setReadOnly(True)
        self.render_table.itemSelectionChanged.connect(self._on_render_select)
        rend = QWidget()
        rl2 = QVBoxLayout(rend)
        rl2.addWidget(QLabel("RENDERED FILE CHECKS (a second pass on the exported video; kept alongside the timeline QC)"))
        rl2.addWidget(self.render_table, 1)
        rl2.addWidget(self.render_detail, 1)

        # ---------------------------------------------------------------- settings
        self.set_widgets: dict[str, QWidget] = {}
        form_box = QWidget()
        sg = QVBoxLayout(form_box)
        gate = QGroupBox("Export gate")
        gf = QFormLayout(gate)
        self.block_level = QComboBox()
        self.block_level.setObjectName("qcBlockLevel")
        from app.qc.severity import BlockLevel  # noqa: PLC0415

        for b in BlockLevel:
            self.block_level.addItem(BLOCK_LABELS[b], b.value)
        gf.addRow("Block export on", self.block_level)
        self.allow_override = QCheckBox("Allow continuing past blocking Errors / Warnings (never past Critical)")
        self.allow_override.setObjectName("qcAllowOverride")
        self.run_before = QCheckBox("Run QC automatically before every export")
        self.run_before.setObjectName("qcRunBefore")
        self.post_render = QCheckBox("Check the rendered file after every export")
        self.post_render.setObjectName("qcPostRender")
        self.ai_review = QCheckBox("AI editorial review (provider-independent; the local rule-based reviewer needs no API)")
        self.ai_review.setObjectName("qcAiReview")
        for w in (self.allow_override, self.run_before, self.post_render, self.ai_review):
            gf.addRow(w)
        sg.addWidget(gate)
        checks = QGroupBox("Checks to run")
        cl = QVBoxLayout(checks)
        self.check_boxes: dict[str, QCheckBox] = {}
        for cid in PIPELINE:
            cb = QCheckBox(CHECKER_LABELS.get(cid, cid))
            cb.setObjectName(f"qcCheck_{cid}")
            if cid == "preflight":
                cb.setChecked(True)
                cb.setEnabled(False)
                cb.setToolTip("Always runs: it protects every other check from a project that cannot be analysed")
            self.check_boxes[cid] = cb
            cl.addWidget(cb)
        sg.addWidget(checks)
        self.perm_table = self._table("qcPermissions", ("Fix", "Permission"), 0)
        self.perm_combos: dict[str, QComboBox] = {}
        kinds = [*FIX_KINDS_SAFE, *FIX_KINDS_CONFIRM]
        self.perm_table.setRowCount(len(kinds))
        for r, k in enumerate(kinds):
            self.perm_table.setItem(r, 0, QTableWidgetItem(k))
            cb = QComboBox()
            cb.setObjectName(f"qcPerm_{k}")
            for label, val in (("Apply automatically (safe fixes only)", "auto"), ("Ask me first", "confirm"), ("Never", "never")):
                cb.addItem(label, val)
            self.perm_combos[k] = cb
            self.perm_table.setCellWidget(r, 1, cb)
        self.perm_table.setMinimumHeight(220)
        pbox = QGroupBox("Auto-fix permissions")
        pl = QVBoxLayout(pbox)
        pl.addWidget(self.perm_table)
        sg.addWidget(pbox)
        adv = QGroupBox("Thresholds (advanced)")
        af = QFormLayout(adv)
        for d in SETTING_DEFS:
            if d.kind in ("choice", "bool") and d.path in ("block_level", "allow_export_override", "run_before_export", "post_render_qc"):
                continue
            w: QWidget
            if d.kind == "int":
                w = QSpinBox()
                w.setRange(int(d.minimum), int(d.maximum))
                w.setSingleStep(max(1, int(d.step)))
            elif d.kind == "bool":
                w = QCheckBox()
            else:
                w = QDoubleSpinBox()
                w.setRange(d.minimum, d.maximum)
                w.setSingleStep(d.step)
                w.setDecimals(2 if d.step < 1 else 1)
            w.setObjectName("qcSet_" + d.path.replace(".", "_"))
            w.setToolTip(d.help or d.label)
            self.set_widgets[d.path] = w
            af.addRow(f"{d.group} · {d.label}", w)
        sg.addWidget(adv)
        self.save_settings_btn = QPushButton("Save QC settings")
        self.save_settings_btn.setObjectName("qcSaveSettings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        self.revert_settings_btn = QPushButton("Revert changes")
        self.revert_settings_btn.setObjectName("qcRevertSettings")
        self.revert_settings_btn.setToolTip("Discard what you changed in this form and show the saved settings again")
        self.revert_settings_btn.clicked.connect(self.revert_settings)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_settings_btn)
        save_row.addWidget(self.revert_settings_btn)
        save_row.addStretch(1)
        sg.addLayout(save_row)
        sg.addStretch(1)
        set_scroll = QScrollArea()
        set_scroll.setWidgetResizable(True)
        set_scroll.setWidget(form_box)
        set_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("qcTabs")
        self.tabs.addTab(dash_scroll, "Dashboard")
        self.tabs.addTab(hist, "History")
        self.tabs.addTab(rend, "Rendered file")
        self.tabs.addTab(set_scroll, "Settings")
        root = QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(note)
        root.addWidget(self.tabs, 1)

        b = ctx.bridge
        b.on("qc.updated", self._on_qc_event)
        b.on("project.changed", lambda p: self._schedule())  # any edit can make the result stale (the QC fingerprint covers audio, captions, visuals and settings too)
        b.on("project.opened", lambda p: self._schedule())
        b.on("project.closed", lambda p: self._clear())
        b.on("job.updated", self._on_job)
        self._sync_buttons()

    # ================================================================ helpers
    @staticmethod
    def _table(name: str, columns: tuple[str, ...], rows: int) -> QTableWidget:
        t = QTableWidget(rows, len(columns))
        t.setObjectName(name)
        t.setHorizontalHeaderLabels(columns)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        t.setMinimumHeight(70)
        return t

    @property
    def ws(self):
        return self.ctx.ws

    def status(self, message: str) -> None:
        self.ctx.status(message)

    def _schedule(self) -> None:
        if self._paused:  # not on screen: rebuilding tables on every edit would only slow the editor down (refresh() runs when the page opens)
            self._dirty = True
            return
        self._refresh_timer.start()

    def _on_qc_event(self, payload: dict) -> None:
        if payload.get("kind") == "progress":  # many per second while a run is going: only the progress widgets change
            if not self._paused and self.ws.project is not None:
                self._show_progress()
                self._sync_buttons()
            return
        self._schedule()

    def pause(self) -> None:
        self._refresh_timer.stop()
        self._paused = True

    # ================================================================ running
    def run_qc(self, force: bool) -> None:
        self.ctx.guard(self, lambda: self.ws.qc.run_full_qc(force=force), modal=True, title="AI Quality Control")
        self._sync_buttons()
        self._show_progress()

    def cancel_run(self) -> None:
        self.ws.qc.cancel()
        self._sync_buttons()

    def retry_failed(self) -> None:
        self.ctx.guard(self, lambda: self.ws.qc.retry_failed_check(), modal=True, title="Retry failed checks")
        self._sync_buttons()

    def run_scene(self) -> None:
        i = self._current()
        if i is None or not i.scene_id:
            self.status("Select an issue that belongs to a scene first.")
            return
        sid = i.scene_id
        self.ctx.guard(self, lambda: self.ws.qc.run_scene_qc(sid), modal=True, title="Scene QC")
        self._sync_buttons()

    def run_category(self) -> None:
        cid = self.category.currentData()
        self.ctx.guard(self, lambda: self.ws.qc.run_category_qc(cid), modal=True, title="QC check")
        self._sync_buttons()

    def _on_job(self, payload: dict) -> None:
        job = payload.get("job")
        if job is None or job.type not in ("qc.run", "qc.render"):
            return
        self._show_progress()
        self._sync_buttons()

    def _show_progress(self) -> None:
        if self.ws.project is None:
            return
        pr = self.ws.qc.progress
        self.progress.setValue(int(pr.fraction * 100))
        self.stage_label.setText(pr.message if pr.state != "IDLE" or pr.fraction < 1 or pr.message == "Canceled" else "")  # a finished run says nothing; a canceled one says so
        self._fill_stages(pr.checkers)

    def _fill_stages(self, statuses: dict) -> None:
        if not statuses and self.ws.project is not None and self.ws.project.qc_runs:
            last = self.ws.project.qc_runs[-1].get("checkers", {})
            rows = [(cid, v.get("state", "PENDING"), v.get("error") or (f"{v.get('issues', 0)} finding(s)" if v.get("issues") else "")) for cid, v in last.items()]
        else:
            rows = [(cid, st.state.value, st.error or st.message) for cid, st in statuses.items()]
        order = {cid: i for i, cid in enumerate(PIPELINE)}
        rows.sort(key=lambda r: order.get(r[0], 99))
        self.stages.setRowCount(len(rows))
        for r, (cid, state, detail) in enumerate(rows):
            running = state == "RUNNING"
            txt = STATE_TEXT.get(state, state)
            if running:
                txt = f"{int(self.ws.qc.progress.fraction * 100)}%"
            for c, text in enumerate((CHECKER_LABELS.get(cid, cid), txt, detail)):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, cid)
                if state == "FAILED":
                    item.setForeground(QColor(COLORS[Severity.ERROR]))
                self.stages.setItem(r, c, item)

    # ================================================================ refresh
    def _clear(self) -> None:
        self._issues = {}
        self._selected = None
        self._history_pick = []
        self._settings_key = ""
        for t in (self.issue_table, self.history_table, self.stages, self.group_table, self.ignored_table, self.render_table):
            t.setRowCount(0)
        self.overall_label.setText("Not analysed yet")
        for w in (self.status_label, self.export_label, self.stale_label, self.stage_label, self.issues_hint):
            w.setText("")
        for sev, lab in self.count_labels.items():
            lab.setText(f"{PLURAL[Severity(sev)]}  0")
        self.progress.setValue(0)
        for box in (self.detail, self.compare_view, self.render_detail):
            box.clear()
        self.fix_note.setText("")
        self._sync_buttons()

    def refresh(self) -> None:
        """Show the current state (the main window calls this when the page opens, which also ends a pause)."""
        self._paused = False
        self._dirty = False
        self._refresh_now()

    def _refresh_now(self) -> None:
        p = self.ws.project
        if p is None:
            self._clear()
            return
        self._loading = True
        try:
            self._fill_scores()
            self._fill_scene_filter()
            self._fill_issues()
            self._fill_history()
            self._fill_render_results()
            self._load_settings()
            self._show_progress()
        finally:
            self._loading = False
        self._sync_buttons()

    def _fill_scores(self) -> None:
        p = self.ws.project
        s = p.qc_scores if p else None
        if s is None or not p.qc_runs:
            self.overall_label.setText("Not analysed yet")
            self.status_label.setText("")
            self.export_label.setText("Run QC to check the project before exporting.")
            self.stale_label.setText("")
            self.group_table.setRowCount(0)
            for sev, lab in self.count_labels.items():
                lab.setText(f"{PLURAL[Severity(sev)]}  0")
            return
        self.overall_label.setText(f"Overall Score: {s.overall:.0f}/100")
        self.status_label.setText(STATUS_TEXT.get(s.status, s.status))
        self.status_label.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {STATUS_COLOR.get(s.status, '#ffffff')};")
        gate = self.ws.qc.export_gate()
        self.export_label.setText(gate.message or EXPORT_TEXT.get(s.export, s.export))
        stale = not gate.run_current
        failed = p.qc_runs[-1].get("failed", [])
        notes = []
        if stale:
            notes.append("The project changed since this QC run: run QC again to refresh the results." if p.qc_runs[-1].get("content_hash")
                         else "This QC run did not cover the whole project (a scene or category run, or a canceled run): run QC again for a complete result.")
        if failed:
            notes.append("These checks did not complete: " + ", ".join(CHECKER_LABELS.get(f, f) for f in failed) + ". Their scores are shown as — (not 100).")
        self.stale_label.setText("\n".join(notes))
        for sev, lab in self.count_labels.items():
            lab.setText(f"{PLURAL[Severity(sev)]}  {s.counts.get(sev, 0)}")
        self.group_table.setRowCount(len(SCORE_GROUPS))
        for r, g in enumerate(SCORE_GROUPS):
            self.group_table.setItem(r, 0, QTableWidgetItem(GROUP_LABELS[g]))
            val = QTableWidgetItem("—" if g in s.unavailable else f"{s.groups.get(g, 0):.0f}")
            val.setData(Qt.ItemDataRole.UserRole, g)
            self.group_table.setItem(r, 1, val)

    def _fill_scene_filter(self) -> None:
        p = self.ws.project
        keep = self.scene_filter.currentData() or ""
        self.scene_filter.blockSignals(True)
        self.scene_filter.clear()
        self.scene_filter.addItem("All scenes", "")
        for sc in (p.scenes if p else []):
            self.scene_filter.addItem(f"Scene {sc.label}", sc.id)
        j = self.scene_filter.findData(keep)
        self.scene_filter.setCurrentIndex(j if j >= 0 else 0)
        self.scene_filter.blockSignals(False)

    def _filters_active(self) -> bool:
        return bool(self.sev_filter.currentData() or self.cat_filter.currentData() or self.scene_filter.currentData() or self.conf_filter.currentData()
                    or (self.status_filter.currentData() or "open") != "open")

    def reset_filters(self, status: str = "open") -> None:
        """Show everything open again (or everything, with ``status='all'``): used when an issue must be found that the filters hide."""
        for combo, val in ((self.sev_filter, ""), (self.cat_filter, ""), (self.scene_filter, ""), (self.conf_filter, ""), (self.status_filter, status)):
            combo.blockSignals(True)
            j = combo.findData(val)
            combo.setCurrentIndex(j if j >= 0 else 0)
            combo.blockSignals(False)
        self._sync_show_closed()

    def _on_show_closed(self, checked: bool) -> None:
        j = self.status_filter.findData("all" if checked else "open")
        self.status_filter.blockSignals(True)
        self.status_filter.setCurrentIndex(j)
        self.status_filter.blockSignals(False)
        self._fill_issues()

    def _sync_show_closed(self) -> None:
        self.show_closed.blockSignals(True)
        self.show_closed.setChecked((self.status_filter.currentData() or "open") != "open")
        self.show_closed.blockSignals(False)

    @staticmethod
    def _passes(i: QCIssue, status: str, conf: str) -> bool:
        ignored = i.ignored_by_user or i.status is IssueStatus.IGNORED
        if status == "open" and not (i.status is IssueStatus.OPEN and not ignored):
            return False
        if status == "fixed" and i.status is not IssueStatus.FIXED:
            return False
        if status == "ignored" and not ignored:
            return False
        label = confidence_label(i.confidence)
        return not conf or label.lower() == conf

    def _fill_issues(self, *_a) -> None:
        p = self.ws.project
        if p is None:
            return
        self._sync_show_closed()
        status = self.status_filter.currentData() or "open"
        conf = self.conf_filter.currentData() or ""
        closed = status != "open"
        issues = self.ws.qc.issues(severity=self.sev_filter.currentData() or None, category=self.cat_filter.currentData() or None, scene_id=self.scene_filter.currentData() or None,
                                   include_ignored=closed, include_fixed=closed)
        issues = [i for i in issues if self._passes(i, status, conf)]
        self._issues = {i.issue_id: i for i in issues}
        scene_labels = {s.id: f"Scene {s.label}" for s in p.scenes}
        keep_ids = [i for i in self._selected_ids() if i in self._issues]  # a refresh must not drop what the user selected
        keep = self._selected
        self.issue_table.blockSignals(True)
        self.issue_table.setRowCount(len(issues))
        for r, i in enumerate(issues):
            conf_text = "" if i.confidence >= 99.5 else f"{i.confidence:.0f}% ({confidence_label(i.confidence)})"
            state = " (ignored)" if i.ignored_by_user else " (fixed)" if i.status is IssueStatus.FIXED else ""
            vals = (i.severity.value + state, scene_labels.get(i.scene_id or "", "—") if i.scene_id else "—", fmt_time(i.start_time), i.category.value.replace("_", " ").title(), i.title, conf_text)
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, i.issue_id)
                if c == 0:
                    item.setForeground(QColor(COLORS[i.severity]))
                self.issue_table.setItem(r, c, item)
        self.issue_table.blockSignals(False)
        self._fill_hint(len(issues))
        if keep_ids or (keep and keep in self._issues):
            ids = keep_ids or [keep]
            self.issue_table.blockSignals(True)
            sm = self.issue_table.selectionModel()
            sm.clearSelection()
            for r in range(self.issue_table.rowCount()):
                it = self.issue_table.item(r, 0)
                if it is not None and it.data(Qt.ItemDataRole.UserRole) in ids:
                    sm.select(self.issue_table.model().index(r, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            self.issue_table.blockSignals(False)
            self._selected = keep if keep in ids else ids[0]
            self._show_detail(self._issues.get(self._selected))
        else:
            self._selected = None
            self._show_detail(None)

    def _fill_hint(self, shown: int) -> None:
        p = self.ws.project
        if p is None or shown:
            self.issues_hint.setText("")
        elif not p.qc_runs:
            self.issues_hint.setText("QC has not run yet: press “Run QC” to check the project.")
        elif self._filters_active():
            hidden = len(p.qc_issues)
            self.issues_hint.setText(f"No issue matches the filters ({hidden} issue(s) in total). Choose “All …” in the filters to see them.")
        else:
            self.issues_hint.setText("No open issues." if not p.qc_runs[-1].get("failed") else "No open issues, but some checks did not complete (see the stages above).")

    def _fill_history(self) -> None:
        p = self.ws.project
        runs = list(reversed(p.qc_runs)) if p else []
        self.history_table.setRowCount(len(runs))
        for r, rec in enumerate(runs):
            c = rec.get("counts", {})
            vals = (f"#{rec.get('number', '?')}", (rec.get("finished_at") or rec.get("created_at", "")).replace("T", " ")[:19], f"{rec.get('overall', 0):.0f}", rec.get("status", ""),
                    str(c.get("CRITICAL", 0)), str(c.get("ERROR", 0)), str(c.get("WARNING", 0)), rec.get("trigger", ""))
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, rec.get("number"))
                self.history_table.setItem(r, col, item)
        ign = self.ws.qc.ignored() if p else []
        self.ignored_table.setRowCount(len(ign))
        for r, rec in enumerate(ign):
            for col, text in enumerate((rec.title or rec.code, "all of this type" if rec.scope == "type" else "this issue", rec.reason)):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, rec.ignore_id)
                self.ignored_table.setItem(r, col, item)

    def _fill_render_results(self) -> None:
        res = self.ws.qc.render_results() if self.ws.project is not None else {}
        self.render_table.setRowCount(len(res))
        for r, (rid, rr) in enumerate(res.items()):
            for col, text in enumerate((rid, rr.get("status", "?"), rr.get("summary", ""))):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, rid)
                self.render_table.setItem(r, col, item)

    def _on_render_select(self) -> None:
        row = self.render_table.currentRow()
        item = self.render_table.item(row, 0) if row >= 0 else None
        res = self.ws.qc.render_results().get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        if not res:
            self.render_detail.clear()
            return
        lines = [f"{res.get('status', '?')} — {res.get('summary', '')}", ""]
        for c in res.get("checks", []):
            lines.append(f"[{c.get('status', '?').upper():4}] {c.get('label', c.get('id'))}: {c.get('message', '')}")
        self.render_detail.setPlainText("\n".join(lines))

    # ================================================================ selection / detail
    def _current(self) -> QCIssue | None:
        return self._issues.get(self._selected) if self._selected else None

    def _selected_ids(self) -> list[str]:
        rows = sorted({ix.row() for ix in self.issue_table.selectedIndexes()})
        out = []
        for r in rows:
            it = self.issue_table.item(r, 0)
            if it is not None:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def select_issue(self, issue_id: str) -> None:
        """Select (and scroll to) an issue by id: used by the timeline markers. Filters that hide it are cleared."""
        if issue_id not in self._issues:
            self._reveal(issue_id)
        for r in range(self.issue_table.rowCount()):
            it = self.issue_table.item(r, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) == issue_id:
                self.issue_table.selectRow(r)
                self.issue_table.scrollToItem(it)
                self._selected = issue_id
                self._show_detail(self._issues.get(issue_id))
                return

    def _reveal(self, issue_id: str) -> None:
        try:
            i = self.ws.qc.issue(issue_id)
        except Exception:  # noqa: BLE001
            return
        self.reset_filters("all" if (i.ignored_by_user or i.status is not IssueStatus.OPEN) else "open")
        self._fill_issues()

    def _on_select(self) -> None:
        ids = self._selected_ids()
        self._selected = ids[0] if ids else None
        self._show_detail(self._current())
        self._sync_buttons()

    def _show_detail(self, i: QCIssue | None) -> None:
        if i is None:
            self.detail.clear()
            self.fix_note.setText("")
            self._sync_buttons()
            return
        p = self.ws.project
        scene = next((s for s in p.scenes if s.id == i.scene_id), None) if p else None
        lines = [f"{i.title}", f"Severity: {i.severity.value}    Confidence: {i.confidence:.0f}% ({confidence_label(i.confidence)})    Source: {i.detection_source}", "",
                 f"Scene: {('Scene ' + scene.label) if scene else '—'}", f"Time range: {fmt_time(i.start_time)} – {fmt_time(i.end_time)}" if i.end_time is not None else f"Time: {fmt_time(i.start_time)}", ""]
        if i.description:
            lines += ["What was detected", i.description, ""]
        if i.why_it_matters:
            lines += ["Why it matters", i.why_it_matters, ""]
        if i.current_value:
            lines += [f"Current value: {i.current_value}"]
        if i.recommended_value:
            lines += [f"Recommended value: {i.recommended_value}"]
        if i.original_research_score is not None or i.current_qc_score is not None:
            lines += [f"Research score (original, unchanged): {i.original_research_score if i.original_research_score is not None else '—'}   Current QC score: {i.current_qc_score if i.current_qc_score is not None else '—'}"]
        if i.suggested_fix:
            lines += ["", "Suggested fix", i.suggested_fix]
        if i.affected_elements:
            lines += ["", "Affected: " + ", ".join(i.affected_elements)]
        if i.ignored_by_user:
            lines += ["", f"Ignored by you{': ' + i.ignore_reason if i.ignore_reason else ''}."]
        self.detail.setPlainText("\n".join(lines))
        if i.auto_fix_available and i.fix is not None and i.fix.route is FixRoute.COMMAND:
            self.fix_note.setText("Auto-fix available — " + ("safe: it can be applied without confirmation." if i.auto_fix_safe else "needs your confirmation."))
        elif i.fix_blocked_reason:
            self.fix_note.setText(i.fix_blocked_reason)
        else:
            self.fix_note.setText("No automatic fix: use the buttons to open the place where you decide." if i.fix is not None else "No automatic fix for this issue.")
        self._sync_buttons()

    # ================================================================ actions
    def _require(self) -> QCIssue | None:
        i = self._current()
        if i is None:
            self.status("Select an issue first.")
        return i

    def preview_fix(self) -> None:
        i = self._require()
        if i is None:
            return
        pv = {}

        def go() -> None:
            pv["p"] = self.ws.qc.preview_fix(i.issue_id)

        if not self.ctx.guard(self, go, modal=True, title="Preview fix") or "p" not in pv:
            return
        p = pv["p"]
        before = "\n".join(f"  {k}: {v}" for k, v in p.before.items()) or "  —"
        after = "\n".join(f"  {k}: {v}" for k, v in p.after.items()) or "  —"
        text = f"{p.summary}\n\nBefore:\n{before}\n\nAfter:\n{after}"
        if p.changes:
            text += "\n\n" + "\n".join("• " + c for c in p.changes)
        if p.blocked_reason:
            text += f"\n\nNot available: {p.blocked_reason}"
        box = QMessageBox(QMessageBox.Icon.Question, "Preview fix", text, QMessageBox.StandardButton.NoButton, self)
        box.setObjectName("qcPreviewDialog")
        if not p.blocked_reason:
            box.addButton("Apply", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Reject", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is not None and clicked.text() == "Apply":
            self.apply_fix(True)

    def _confirm(self, name: str, title: str, text: str) -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, self)
        box.setObjectName(name)
        return box.exec() == QMessageBox.StandardButton.Ok

    def _confirm_fix(self, issue: QCIssue) -> bool:
        return self._confirm("qcConfirmFix", "Apply fix", f"{(issue.fix.summary if issue.fix else '') or issue.suggested_fix}\n\nThis changes your edit. You can undo it with Ctrl+Z.")

    def apply_fix(self, confirmed: bool) -> None:
        i = self._require()
        if i is None:
            return
        if i.fix is not None and not i.auto_fix_safe and not confirmed:
            if not self._confirm_fix(i):
                return
            confirmed = True
        asked = {"again": False}

        def go(ok: bool) -> None:
            try:
                self.ws.qc.apply_fix(i.issue_id, confirmed=ok)
            except QCError as exc:
                if not ok and exc.user_message == NEEDS_CONFIRM:  # the flag said safe, but the plan for the project as it is now is not: the user decides
                    asked["again"] = True
                    return
                raise

        done = self.ctx.guard(self, lambda: go(confirmed), modal=True, title="Apply fix")
        if asked["again"]:
            done = self._confirm_fix(i) and self.ctx.guard(self, lambda: go(True), modal=True, title="Apply fix")
        if done:
            self.status("Fix applied. Undo (Ctrl+Z) restores the previous state.")
            self._refresh_now()

    @staticmethod
    def _command_fix(i: QCIssue | None) -> bool:
        return bool(i is not None and i.status is IssueStatus.OPEN and not i.ignored_by_user and i.fix is not None and i.fix.route is FixRoute.COMMAND and i.auto_fix_available)

    def fix_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            self.status("Select one or more issues first.")
            return
        chosen = [self._issues[iid] for iid in ids if self._command_fix(self._issues.get(iid))]
        safe = [i for i in chosen if i.auto_fix_safe]
        careful = [i for i in chosen if not i.auto_fix_safe]
        if careful and not self._confirm("qcConfirmFixSelected", "Apply fixes",
                                         f"{len(careful)} of the selected fix(es) change your edit and need your confirmation:\n\n"
                                         + "\n".join("• " + ((i.fix.summary if i.fix else "") or i.title) for i in careful[:8]) + ("\n• …" if len(careful) > 8 else "")
                                         + "\n\nEach can be undone separately with Ctrl+Z."):
            careful = []  # the user declined: only the safe ones go ahead
        done = 0
        if safe:  # all the safe ones are ONE undo step
            res = {}
            if self.ctx.guard(self, lambda: res.update(r=self.ws.qc.apply_safe_fixes([i.issue_id for i in safe])), modal=False, title="Fix selected"):
                done += len(res.get("r") or [])
        for i in careful:
            if self.ctx.guard(self, lambda iid=i.issue_id: self.ws.qc.apply_fix(iid, confirmed=True), modal=False, title="Fix selected"):
                done += 1
        self.status(f"Applied {done} fix(es); skipped the rest (they need another page or your decision)." + self._skipped_note())
        self._refresh_now()

    def fix_safe(self, code_prefix: str | None) -> None:
        res = {}

        def go() -> None:
            res["r"] = self.ws.qc.apply_safe_fixes(None, code_prefix=code_prefix)

        if self.ctx.guard(self, go, modal=True, title="Fix all safe issues"):
            self.status(f"Applied {len(res.get('r') or [])} safe fix(es) as one undo step. Semantic and creative changes are never applied automatically." + self._skipped_note())
            self._refresh_now()

    def fix_similar(self) -> None:
        i = self._require()
        if i is None:
            return
        if not i.auto_fix_safe:
            n = sum(1 for o in self._issues.values() if o.code == i.code and self._command_fix(o))
            if not self._confirm("qcConfirmFixSimilar", "Fix all similar", f"{(i.fix.summary if i.fix else '') or i.title}\n\nThis changes your edit in up to {max(n, 1)} place(s). "
                                 "Each one is checked again first; locked or user-made elements are skipped. One Ctrl+Z undoes them all."):
                return
        if self.ctx.guard(self, lambda: self.ws.qc.fix_similar(i.issue_id, confirmed=not i.auto_fix_safe), modal=True, title="Fix all similar"):
            note = self._skipped_note()
            if note:
                self.status("Fixed the similar issues that could be fixed." + note)
        self._refresh_now()

    def _skipped_note(self) -> str:
        """What the last batch left out and why (the fix engine records one reason per skipped issue)."""
        skipped = list(getattr(getattr(self.ws.qc, "fixes", None), "last_skipped", None) or [])
        if not skipped:
            return ""
        return f" Skipped {len(skipped)}: {skipped[0].reason}" + (" (and others)" if len(skipped) > 1 else "")

    def ignore(self, whole_type: bool) -> None:
        i = self._require()
        if i is None:
            return
        if i.severity is Severity.CRITICAL:
            self.status("A critical issue cannot be ignored: fix it, or the export stays blocked.")
            return
        reason, ok = ask_reason(self, "Ignore this type of issue" if whole_type else "Ignore this issue")
        if not ok:
            return
        self.ctx.guard(self, lambda: self.ws.qc.ignore_type(i.issue_id, reason) if whole_type else self.ws.qc.ignore_issue(i.issue_id, reason), modal=True, title="Ignore")
        self._refresh_now()

    def unignore(self) -> None:
        row = self.ignored_table.currentRow()
        it = self.ignored_table.item(row, 0) if row >= 0 else None
        if it is None:
            return
        self.ctx.guard(self, lambda: self.ws.qc.unignore(it.data(Qt.ItemDataRole.UserRole)), modal=True, title="Stop ignoring")
        self._refresh_now()

    def open_timeline(self) -> None:
        i = self._require()
        if i is not None and i.start_time is not None:
            self.open_timeline_requested.emit(i.start_time, i.timeline_item_id or "")

    def open_scene(self) -> None:
        i = self._require()
        if i is not None and i.scene_id:
            self.open_scene_requested.emit(i.scene_id)

    def replace_visual(self) -> None:
        i = self._require()
        if i is not None and i.scene_id:
            self.replace_visual_requested.emit(i.scene_id)

    def search_again(self) -> None:
        i = self._require()
        if i is not None and i.scene_id:
            self.search_again_requested.emit(i.scene_id)

    def export_report(self) -> None:
        path = {}
        if self.ctx.guard(self, lambda: path.update(p=self.ws.qc.save_report()), modal=True, title="QC report"):
            self.status(f"QC report saved: {path['p']}")

    # ================================================================ history
    def _on_history_select(self) -> None:
        self._history_pick = sorted({self.history_table.item(ix.row(), 0).data(Qt.ItemDataRole.UserRole) for ix in self.history_table.selectedIndexes()})

    def compare_runs(self) -> None:
        pick = self._history_pick
        if len(pick) < 2:
            self.compare_view.setPlainText("Select two runs in the list to compare them.")
            return
        res = {}
        if self.ctx.guard(self, lambda: res.update(c=self.ws.qc.compare_runs(pick[0], pick[-1])), modal=True, title="Compare QC runs"):
            self.compare_view.setPlainText("\n".join(res["c"].lines()))

    # ================================================================ settings
    def _settings_fingerprint(self) -> str:
        p = self.ws.project
        return f"{p.project_id}:{json.dumps(p.qc_settings.to_dict(), sort_keys=True, default=str)}" if p else ""

    def _load_settings(self, force: bool = False) -> None:
        """Fill the editor from the project. A refresh (a finished run, an edit elsewhere) leaves the user's unsaved edits alone: it reloads only when the stored settings changed."""
        p = self.ws.project
        if p is None:
            return
        key = self._settings_fingerprint()
        if not force and key == self._settings_key:
            return
        self._settings_key = key
        s = p.qc_settings
        i = self.block_level.findData(s.block_level)
        if i >= 0:
            self.block_level.setCurrentIndex(i)
        self.allow_override.setChecked(s.allow_export_override)
        self.run_before.setChecked(s.run_before_export)
        self.post_render.setChecked(s.post_render_qc)
        self.ai_review.setChecked(s.ai_review_enabled)
        for cid, box in self.check_boxes.items():
            box.setChecked(cid == "preflight" or cid in s.enabled_checkers)
        for k, cb in self.perm_combos.items():
            j = cb.findData(s.permission(k))
            if j >= 0:
                cb.setCurrentIndex(j)
        for path, w in self.set_widgets.items():
            try:
                v = s.get_path(path)
            except AttributeError:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            else:
                w.setValue(v)  # type: ignore[attr-defined]

    def revert_settings(self) -> None:
        self._load_settings(force=True)
        self.status("QC settings editor reset to the saved values.")

    def save_settings(self) -> None:
        p = self.ws.project
        if p is None:
            return
        cur = p.qc_settings
        wanted: dict = {"block_level": self.block_level.currentData(), "allow_export_override": self.allow_override.isChecked(), "run_before_export": self.run_before.isChecked(),
                        "post_render_qc": self.post_render.isChecked(), "ai_review_enabled": self.ai_review.isChecked()}
        perms = dict(cur.fix_permissions)
        for k, cb in self.perm_combos.items():
            perms[k] = cb.currentData()
        wanted["fix_permissions"] = perms
        wanted["enabled_checkers"] = [cid for cid, box in self.check_boxes.items() if box.isChecked() or cid == "preflight"]
        for path, w in self.set_widgets.items():
            wanted[path] = w.isChecked() if isinstance(w, QCheckBox) else w.value()  # type: ignore[attr-defined]
        changes: dict = {}
        for k, v in wanted.items():
            try:
                old = cur.get_path(k) if "." in k else getattr(cur, k)
            except AttributeError:
                old = None
            if k == "enabled_checkers":
                if sorted(old or []) != sorted(v):
                    changes[k] = v
                continue
            same = abs(float(old) - float(v)) < 1e-9 if isinstance(v, float) and isinstance(old, (int, float)) and not isinstance(old, bool) else old == v
            if not same:
                changes[k] = v  # only what the user changed: saving must not round or touch anything else (and an unchanged form is not an undo step)
        if not changes:
            self.status("No QC settings were changed.")
            return
        if self.ctx.guard(self, lambda: self.ws.qc.update_settings(**changes), modal=True, title="QC settings"):
            self.status("QC settings saved. Run QC again to apply the new thresholds.")

    # ================================================================ buttons
    def _fixable_count(self, prefix: str | None) -> int:
        """Open safe fixes a batch button would try (the engine re-checks each one when it runs)."""
        p = self.ws.project
        if p is None:
            return 0
        return sum(1 for i in p.qc_issues if self._command_fix(i) and i.auto_fix_safe and (not prefix or i.code.startswith(prefix)))

    def _sync_buttons(self) -> None:
        p = self.ws.project
        has = p is not None
        running = bool(has and self.ws.qc.running)
        self.run_btn.setEnabled(has and not running)
        self.rerun_btn.setEnabled(has and not running)
        self.cancel_btn.setEnabled(running)
        failed = bool(has and p.qc_runs and p.qc_runs[-1].get("failed"))
        self.retry_btn.setEnabled(has and not running and failed)
        i = self._current()
        open_ = bool(i and i.status is IssueStatus.OPEN and not i.ignored_by_user)
        self.scene_btn.setEnabled(has and not running and bool(i and i.scene_id))
        self.category_btn.setEnabled(has and not running)
        can_cmd = bool(self._command_fix(i) and not running)  # never for protected / navigate-only / already closed issues (auto_fix_available is off for those)
        self.fix_btn.setEnabled(can_cmd)
        self.preview_btn.setEnabled(can_cmd)
        self.similar_btn.setEnabled(can_cmd)
        can_ignore = bool(open_ and not running and i.severity is not Severity.CRITICAL)  # a CRITICAL is never ignorable
        self.ignore_btn.setEnabled(can_ignore)
        self.ignore_type_btn.setEnabled(can_ignore)
        why = "A critical issue cannot be ignored: fix it, or the export stays blocked." if i is not None and i.severity is Severity.CRITICAL else ""
        self.ignore_btn.setToolTip(why)
        self.ignore_type_btn.setToolTip(why or "Stop reporting every issue of this kind")
        self.timeline_btn.setEnabled(bool(i and i.start_time is not None))
        self.scene_open_btn.setEnabled(bool(i and i.scene_id))
        visual = bool(open_ and i.scene_id and (i.category.value in VISUAL_CATEGORIES or (i.fix is not None and i.fix.kind in ("visual.replace", "visual.search_again"))))
        self.replace_btn.setEnabled(visual)
        self.search_btn.setEnabled(visual)
        chosen = [self._issues.get(x) for x in self._selected_ids()]
        self.fix_selected_btn.setEnabled(has and not running and any(self._command_fix(c) for c in chosen))
        has_fix_engine = bool(has and self.ws.qc.fixes is not None)
        for oid, _label, prefix in BATCHES:
            self.batch_btns[oid].setEnabled(has_fix_engine and not running and self._fixable_count(prefix) > 0)
        self.report_btn.setEnabled(bool(has and p.qc_runs))
        self.compare_btn.setEnabled(bool(has and len(p.qc_runs) >= 2))
        self.unignore_btn.setEnabled(bool(has and p.qc_ignored_issues))
