"""Reference page: import a reference video, analyse its editing style, review it, customize and apply it as abstract AI-editing preferences.

The panel never analyses or edits anything itself: everything goes through ``ws.reference`` (the ReferenceService). The reference is analysis
input only; applying a style changes preferences (shot length, motion, captions, text, audio), never the timeline, and copies nothing.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.reference.application import APPLY_NOTICE, MODES, STRENGTHS
from app.reference.style_model import DIMENSION_LABELS, DIMENSIONS, ReferenceStyleProfile  # noqa: F401
from app.ui.context import UiContext

PROFILE_COLUMNS = ("Characteristic", "Reading", "Score", "Confidence")
COMPARE_COLUMNS = ("Characteristic", "Reference", "Current Project")
SIM_COLUMNS = ("Characteristic", "Current", "After style", "Reference")
MODE_LABELS = {"FULL": "Full — every characteristic", "BALANCED": "Balanced — pacing, density, motion", "CUSTOM": "Custom — choose characteristics"}
STATUS_TEXT = {"NONE": "Not analyzed", "RUNNING": "Analyzing…", "COMPLETED": "Analyzed", "PARTIAL": "Analyzed (partial)", "FAILED": "Analysis failed", "CANCELED": "Canceled"}
OVERRIDE_LABELS = {
    "target_shot_duration": ("Target shot length", "{:.1f} s"), "min_shot_duration": ("Shortest shot", "{:.1f} s"), "max_shot_duration": ("Longest shot", "{:.1f} s"),
    "motion_intensity": ("Motion", "{:.0%}"), "transition_frequency": ("Transitions", "{:.0%}"), "text_density": ("Text graphics", "{:.0%}"), "hook_seconds": ("Opening length", "{:.0f} s"),
    "hook_shot_factor": ("Opening shot factor", "×{:.2f}"), "caption_style": ("Caption style", "{}"), "caption_position": ("Caption position", "{}"), "caption_density": ("Caption density", "{:.0%}"),
    "caption_max_words": ("Words per caption", "{}"), "keyword_emphasis_rate": ("Keyword highlighting", "{:.0%}"), "music_level": ("Music level", "{:.0%}"),
    "ducking_strength": ("Music ducking", "{:.0%}"), "sfx_per_minute": ("Sound effects", "{:.1f} /min"), "pause_usage": ("Pause usage", "{:.0%}"),
}


def confirm_apply_style(parent: QWidget | None) -> bool:
    box = QMessageBox(QMessageBox.Icon.Question, "Apply reference style", APPLY_NOTICE, QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, parent)
    box.setObjectName("applyStyleConfirm")
    box.button(QMessageBox.StandardButton.Ok).setText("Apply Style")
    return box.exec() == QMessageBox.StandardButton.Ok


def profile_details(p: ReferenceStyleProfile) -> str:
    """Aggregate statistics only (no per-shot timings, no text): what the reading is based on."""
    st, mo, cap, tx, tr, au, hk = p.shot_stats, p.motion, p.caption_style, p.text, p.transitions, p.audio, p.hook
    lines = [
        f"SHOTS  {st.count} shots · average {st.average_shot_duration:.1f} s · median {st.median_shot_duration:.1f} s · min {st.minimum_shot_duration:.1f} s · max {st.maximum_shot_duration:.1f} s · "
        f"{st.cuts_per_minute:.1f} cuts/min ({st.cut_frequency_class})",
        "       distribution: " + ", ".join(f"{k}: {v}" for k, v in st.shot_duration_distribution.items()) if st.shot_duration_distribution else "       distribution: —",
        f"MOTION {mo.motion_class} · {mo.motion_events_per_minute:.1f} events/min · zoom {mo.zoom_frequency:.1f}/min (average ×{mo.zoom.average_scale:.2f}, max ×{mo.zoom.maximum_scale:.2f}) · "
        f"pan {mo.pan_frequency:.1f}/min · static shots {mo.static_shot_share:.0%}",
    ]
    if cap.caption_present:
        lines.append(f"CAPTIONS {cap.style_class} · coverage {cap.caption_coverage:.0%} · {cap.captions_per_minute:.1f}/min · ~{cap.average_words_per_caption:.1f} words (estimate) · "
                     f"{cap.caption_line_count:.1f} lines · {cap.caption_position} · highlighted {cap.caption_emphasis_rate:.0%} · animated {cap.caption_animation_rate:.0%}")
    else:
        lines.append("CAPTIONS none detected")
    lines.append(f"TEXT   {tx.text_events_per_minute:.1f} overlays/min · headlines {tx.headline_frequency:.1f}/min · numbers {tx.number_graphic_frequency:.1f}/min · lower thirds {tx.lower_third_frequency:.1f}/min")
    lines.append(f"TRANSITIONS {tr.non_cut_share:.0%} of boundaries are not hard cuts · " + ", ".join(f"{k} {v:.0%}" for k, v in tr.transition_distribution.items() if v > 0))
    if au.has_audio:
        lines.append(f"AUDIO  voice {au.voice_dominance:.0%} · music {au.music_behavior} ({au.music_presence:.0%}) · ducking {au.music_ducking_strength:.0%} · SFX {au.sfx_class} ({au.sfx_per_minute:.1f}/min) · "
                     f"silence {au.silence_percentage:.0f}% · average pause {au.average_pause_duration:.2f} s · dynamic range {au.audio_dynamic_range:.0f} dB")
    else:
        lines.append("AUDIO  no audio stream")
    if hk.windows:
        lines.append("HOOK   " + " · ".join(f"first {w.seconds}s: {w.shot_rate:.0f} cuts/min" for w in hk.windows) + (" · " + ", ".join(hk.traits) if hk.traits else ""))
    if p.section_mix:
        lines.append("STRUCTURE " + ", ".join(f"{k.title()} {v:.0%}" for k, v in sorted(p.section_mix.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)


class ReferencePanel(QWidget):
    go_to_ai_edit = Signal()

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._loading = False
        self._plan = None
        self._plan_error = ""
        self._replan_timer = QTimer(self)
        self._replan_timer.setSingleShot(True)
        self._replan_timer.setInterval(120)
        self._replan_timer.timeout.connect(self._replan)

        title = QLabel("REFERENCE STYLE")
        title.setObjectName("title")
        note = QLabel("Analyze a reference video's editing style — rhythm, visual density, motion, captions, text and audio — and let the AI editor use it as abstract preferences for YOUR content. "
                      "Footage, text, graphics and exact shot sequences are never copied.")
        note.setObjectName("muted")
        note.setWordWrap(True)

        # ---------------------------------------------------------------- 1. the reference
        self.import_btn = QPushButton("Import Reference Video…")
        self.import_btn.setObjectName("refImport")
        self.import_btn.clicked.connect(self.choose_file)
        self.link_check = QCheckBox("Link the file instead of copying it into the project")
        self.link_check.setObjectName("refLink")
        self.ref_combo = QComboBox()
        self.ref_combo.setObjectName("refList")
        self.ref_combo.currentIndexChanged.connect(self._on_ref_selected)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setObjectName("refAnalyze")
        self.analyze_btn.clicked.connect(lambda: self.analyze(False))
        self.reanalyze_btn = QPushButton("Re-analyze")
        self.reanalyze_btn.setObjectName("refReanalyze")
        self.reanalyze_btn.clicked.connect(lambda: self.analyze(True))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("refCancel")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.retry_btn = QPushButton("Retry")
        self.retry_btn.setObjectName("refRetry")
        self.retry_btn.clicked.connect(self.retry)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setObjectName("refRemove")
        self.remove_btn.clicked.connect(self.remove_reference)
        self.progress = QProgressBar()
        self.progress.setObjectName("refProgress")
        self.progress.setRange(0, 100)
        self.stage_label = QLabel("")
        self.stage_label.setObjectName("refStage")
        self.status_label = QLabel("")
        self.status_label.setObjectName("refStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("refLog")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(110)
        self.thumb_row = QHBoxLayout()
        self.thumb_row.setSpacing(4)
        ref_box = QGroupBox("1 · REFERENCE VIDEO")
        rl = QVBoxLayout(ref_box)
        r1 = QHBoxLayout()
        r1.addWidget(self.import_btn)
        r1.addWidget(self.link_check, 1)
        rl.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(self.ref_combo, 1)
        for b in (self.analyze_btn, self.reanalyze_btn, self.cancel_btn, self.retry_btn, self.remove_btn):
            r2.addWidget(b)
        rl.addLayout(r2)
        pr = QHBoxLayout()
        pr.addWidget(self.progress, 1)
        pr.addWidget(self.stage_label)
        rl.addLayout(pr)
        rl.addWidget(self.status_label)
        rl.addLayout(self.thumb_row)
        rl.addWidget(self.log_view)

        # ---------------------------------------------------------------- 2. the style profile
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("refSummary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        self.warning_label = QLabel("")
        self.warning_label.setObjectName("refWarnings")
        self.warning_label.setWordWrap(True)
        self.warning_label.setTextFormat(Qt.TextFormat.PlainText)
        self.warning_label.setStyleSheet("color: #e5a24a;")
        self.profile_table = self._table("refProfile", PROFILE_COLUMNS, 0)
        self.details_view = QPlainTextEdit()
        self.details_view.setObjectName("refDetails")
        self.details_view.setReadOnly(True)
        self.details_view.setMaximumHeight(170)
        self.reco_label = QLabel("")
        self.reco_label.setObjectName("refRecommendations")
        self.reco_label.setWordWrap(True)
        self.reco_label.setTextFormat(Qt.TextFormat.PlainText)
        prof_box = QGroupBox("2 · STYLE PROFILE")
        pl = QVBoxLayout(prof_box)
        pl.addWidget(self.summary_label)
        pl.addWidget(self.profile_table)
        pl.addWidget(self.warning_label)
        pl.addWidget(self.details_view)
        pl.addWidget(QLabel("Suggestions for your edit"))
        pl.addWidget(self.reco_label)

        # ---------------------------------------------------------------- 3. apply
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("refMode")
        for m in MODES:
            self.mode_combo.addItem(MODE_LABELS[m], m)
        self.strength_combo = QComboBox()
        self.strength_combo.setObjectName("refStrength")
        for s in STRENGTHS:
            self.strength_combo.addItem(f"{int(s * 100)}%", s)
        self.preserve_check = QCheckBox("Keep my own settings and locked edits (recommended)")
        self.preserve_check.setObjectName("refPreserve")
        self.dim_checks: dict[str, QCheckBox] = {}
        dims_row = QGridLayout()
        for i, d in enumerate(DIMENSIONS):
            cb = QCheckBox(DIMENSION_LABELS[d])
            cb.setObjectName(f"refDim_{d}")
            cb.setChecked(True)
            cb.toggled.connect(self._on_controls_changed)
            self.dim_checks[d] = cb
            dims_row.addWidget(cb, i // 4, i % 4)
        self.custom_dims = QWidget()
        self.custom_dims.setLayout(dims_row)
        self.mode_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.strength_combo.currentIndexChanged.connect(self._on_controls_changed)
        self.preserve_check.toggled.connect(self._on_controls_changed)

        self.apply_btn = QPushButton("Apply Style")
        self.apply_btn.setObjectName("refApply")
        self.apply_btn.setStyleSheet("font-weight: 600;")
        self.apply_btn.clicked.connect(self.apply_style)
        self.customize_btn = QPushButton("Customize")
        self.customize_btn.setObjectName("refCustomize")
        self.customize_btn.clicked.connect(self.toggle_customize)
        self.cancel_style_btn = QPushButton("Cancel")
        self.cancel_style_btn.setObjectName("refCancelStyle")
        self.cancel_style_btn.clicked.connect(self.cancel_customize)
        self.remove_style_btn = QPushButton("Remove Applied Style")
        self.remove_style_btn.setObjectName("refRemoveStyle")
        self.remove_style_btn.clicked.connect(self.remove_style)
        self.undo_style_btn = QPushButton("Undo Last Application")
        self.undo_style_btn.setObjectName("refUndoStyle")
        self.undo_style_btn.clicked.connect(self.undo_last)
        self.applied_label = QLabel("")
        self.applied_label.setObjectName("refApplied")
        self.applied_label.setWordWrap(True)
        self.applied_label.setTextFormat(Qt.TextFormat.PlainText)
        self.plan_view = QLabel("")
        self.plan_view.setObjectName("refPlan")
        self.plan_view.setWordWrap(True)
        self.plan_view.setTextFormat(Qt.TextFormat.PlainText)

        # customize sliders: Reference vs User Target
        self.custom_group = QGroupBox("CUSTOMIZE — Reference vs your target")
        self.custom_group.setObjectName("refCustomGroup")
        cg = QGridLayout(self.custom_group)
        for c, h in enumerate(("Characteristic", "Reference", "Follow", "Your target", "", "Result")):
            lab = QLabel(h)
            lab.setObjectName("muted")
            cg.addWidget(lab, 0, c)
        self.target_sliders: dict[str, QSlider] = {}
        self.follow_checks: dict[str, QCheckBox] = {}
        self.ref_values: dict[str, QLabel] = {}
        self.target_values: dict[str, QLabel] = {}
        self.result_values: dict[str, QLabel] = {}
        for r, d in enumerate(DIMENSIONS, start=1):
            cg.addWidget(QLabel(DIMENSION_LABELS[d]), r, 0)
            rv = QLabel("—")
            rv.setObjectName(f"refRef_{d}")
            self.ref_values[d] = rv
            cg.addWidget(rv, r, 1)
            fc = QCheckBox()
            fc.setObjectName(f"refFollow_{d}")
            fc.setChecked(True)
            fc.setToolTip("Follow the reference value for this characteristic")
            self.follow_checks[d] = fc
            cg.addWidget(fc, r, 2)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setObjectName(f"refTarget_{d}")
            sl.setRange(0, 100)
            sl.setEnabled(False)
            self.target_sliders[d] = sl
            cg.addWidget(sl, r, 3)
            tv = QLabel("")
            tv.setMinimumWidth(32)
            self.target_values[d] = tv
            cg.addWidget(tv, r, 4)
            ev = QLabel("—")
            ev.setObjectName(f"refResult_{d}")
            self.result_values[d] = ev
            cg.addWidget(ev, r, 5)
            fc.toggled.connect(lambda on, dim=d: self._on_follow(dim, on))
            sl.valueChanged.connect(lambda v, dim=d: self._on_slider(dim, v))
        cg.setColumnStretch(3, 1)
        self.custom_group.setVisible(False)

        # style request (OriginalityGuard)
        self.request_edit = QLineEdit()
        self.request_edit.setObjectName("refRequest")
        self.request_edit.setPlaceholderText("Optional: describe the style you want, e.g. “faster pacing with subtle zooms”")
        self.request_btn = QPushButton("Use")
        self.request_btn.setObjectName("refRequestUse")
        self.request_btn.clicked.connect(self.submit_request)
        self.request_feedback = QLabel("")
        self.request_feedback.setObjectName("refRequestFeedback")
        self.request_feedback.setWordWrap(True)
        self.request_feedback.setTextFormat(Qt.TextFormat.PlainText)

        apply_box = QGroupBox("3 · APPLY AS EDITING PREFERENCES")
        al = QVBoxLayout(apply_box)
        form = QFormLayout()
        form.addRow("Mode", self.mode_combo)
        form.addRow("Style strength", self.strength_combo)
        al.addLayout(form)
        al.addWidget(self.custom_dims)
        al.addWidget(self.preserve_check)
        al.addWidget(self.custom_group)
        rr = QHBoxLayout()
        rr.addWidget(self.request_edit, 1)
        rr.addWidget(self.request_btn)
        al.addLayout(rr)
        al.addWidget(self.request_feedback)
        al.addWidget(self.plan_view)
        br = QHBoxLayout()
        for b in (self.apply_btn, self.customize_btn, self.cancel_style_btn):
            br.addWidget(b)
        br.addStretch(1)
        br.addWidget(self.undo_style_btn)
        br.addWidget(self.remove_style_btn)
        al.addLayout(br)
        al.addWidget(self.applied_label)

        # ---------------------------------------------------------------- 4. comparison + simulation
        self.compare_table = self._table("refCompare", COMPARE_COLUMNS, 0)
        self.similarity_label = QLabel("")
        self.similarity_label.setObjectName("refSimilarity")
        self.similarity_label.setWordWrap(True)
        self.similarity_label.setTextFormat(Qt.TextFormat.PlainText)
        self.sim_check = QCheckBox("Show “Current Edit vs Reference Style” simulation")
        self.sim_check.setObjectName("refSimToggle")
        self.sim_check.toggled.connect(self._on_sim_toggle)
        self.sim_table = self._table("refSim", SIM_COLUMNS, 0)
        self.sim_table.setVisible(False)
        self.sim_label = QLabel("")
        self.sim_label.setObjectName("refSimSummary")
        self.sim_label.setWordWrap(True)
        self.sim_label.setTextFormat(Qt.TextFormat.PlainText)
        cmp_box = QGroupBox("4 · REFERENCE vs YOUR PROJECT")
        cl = QVBoxLayout(cmp_box)
        cl.addWidget(self.compare_table)
        cl.addWidget(self.similarity_label)
        cl.addWidget(self.sim_check)
        cl.addWidget(self.sim_table)
        cl.addWidget(self.sim_label)

        self.next_btn = QPushButton("Go to AI Edit")
        self.next_btn.setObjectName("refNext")
        self.next_btn.setToolTip("Applying a style does not change the timeline; generate or regenerate it in AI Edit.")
        self.next_btn.clicked.connect(self.go_to_ai_edit.emit)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.addWidget(title)
        bl.addWidget(note)
        bl.addWidget(ref_box)
        bl.addWidget(prof_box)
        bl.addWidget(apply_box)
        bl.addWidget(cmp_box)
        nr = QHBoxLayout()
        nr.addStretch(1)
        nr.addWidget(self.next_btn)
        bl.addLayout(nr)
        bl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        b = ctx.bridge
        b.on("reference.updated", lambda p: self.refresh())
        b.on("reference.style_changed", lambda p: self._refresh_applied())
        b.on("project.opened", lambda p: self.refresh())
        b.on("project.closed", lambda p: self._clear())
        b.on("job.updated", self._on_job)
        b.on("commands.changed", lambda p: self._refresh_applied())
        self._sync_enabled()

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
        t.setMinimumHeight(60)
        return t

    @property
    def ws(self):
        return self.ctx.ws

    def status(self, message: str) -> None:
        self.ctx.status(message)

    def _current_id(self) -> str | None:
        return self.ref_combo.currentData()

    def pause(self) -> None:
        self._replan_timer.stop()

    # ================================================================ import / analyze
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import a reference video", "", "Videos (*.mp4 *.mov *.mkv *.webm *.avi *.m4v)")
        if path:
            self.import_path(Path(path))

    def import_path(self, path: Path) -> None:
        self.status_label.setText(f"Importing “{path.name}”…")
        self.ctx.guard(self, lambda: self.ws.reference.import_reference_async(path, link=self.link_check.isChecked(), on_done=self._imported, on_error=self._import_failed), modal=True,
                       title="Reference video")

    def _imported(self, asset) -> None:
        self.refresh()
        self.status_label.setText(f"Imported “{asset.name}”. Choose Analyze to read its editing style.")

    def _import_failed(self, job) -> None:
        self.status_label.setText(job.error or "The reference could not be imported.")
        self.refresh()

    def analyze(self, force: bool) -> None:
        rid = self._current_id()
        if not rid:
            return
        self.progress.setValue(0)
        self.log_view.clear()
        self.ctx.guard(self, lambda: self.ws.reference.analyze(rid, force=force, on_done=lambda a: self.refresh(), on_error=lambda j: self.refresh()), modal=True, title="Reference analysis")
        self.refresh()

    def cancel_analysis(self) -> None:
        if self._current_id():
            self.ws.reference.cancel_analysis(self._current_id())

    def retry(self) -> None:
        rid = self._current_id()
        if rid:
            self.ctx.guard(self, lambda: self.ws.reference.retry_analysis(rid), modal=True, title="Reference analysis")
            self.refresh()

    def remove_reference(self) -> None:
        rid = self._current_id()
        if not rid:
            return
        box = QMessageBox(QMessageBox.Icon.Question, "Remove reference", "Remove this reference video and its analysis from the project? Applied style preferences stay until you remove them.",
                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, self)
        if box.exec() == QMessageBox.StandardButton.Yes:
            self.ctx.guard(self, lambda: self.ws.reference.remove_reference(rid), modal=True, title="Reference video")

    def _on_job(self, payload: dict) -> None:
        job = payload.get("job")
        if job is None or not str(job.type).startswith("reference."):
            return
        rid = self._current_id()
        mine = self.ws.reference.job_for(rid) if rid and self.ws.project is not None else None
        if job.type == "reference.analyze" and mine is not None and job.id == mine.id:
            self.progress.setValue(int(job.progress))
            self.stage_label.setText(job.message or "")
            if job.status.is_terminal:
                self.stage_label.setText({"COMPLETED": "Done", "FAILED": "Failed", "CANCELLED": "Canceled"}.get(job.status.value, ""))
        elif job.type == "reference.import":
            self.progress.setValue(int(job.progress))
            self.stage_label.setText(job.message or "")

    def _on_ref_selected(self) -> None:
        if self._loading or self.ws.project is None:
            return
        rid = self._current_id()
        if rid and rid != self.ws.project.reference_settings.active_reference_id:
            self.ctx.guard(self, lambda: self.ws.reference.set_active(rid))
        self._show_profile()
        self._schedule_plan()

    # ================================================================ refresh
    def _clear(self) -> None:
        self._loading = True
        self.ref_combo.clear()
        self._loading = False
        for t in (self.profile_table, self.compare_table, self.sim_table):
            t.setRowCount(0)
        for w in (self.summary_label, self.warning_label, self.reco_label, self.status_label, self.applied_label, self.plan_view, self.similarity_label, self.sim_label):
            w.setText("")
        self.details_view.clear()
        self._plan = None
        self._sync_enabled()

    def refresh(self) -> None:
        p = self.ws.project
        if p is None:
            self._clear()
            return
        self._loading = True
        try:
            rs = p.reference_settings
            self.ref_combo.clear()
            for a in self.ws.reference.references():
                self.ref_combo.addItem(f"{a.name} — {STATUS_TEXT.get(a.analysis_status, a.analysis_status)}", a.reference_id)
            i = self.ref_combo.findData(rs.active_reference_id)
            if i >= 0:
                self.ref_combo.setCurrentIndex(i)
            self._set(self.mode_combo, rs.application_mode)
            self._set(self.strength_combo, min(STRENGTHS, key=lambda s: abs(s - rs.style_strength)))
            self.preserve_check.setChecked(rs.preserve_user_edits)
            for d, cb in self.dim_checks.items():
                cb.setChecked(d in rs.custom_dimensions)
            for d in DIMENSIONS:
                v = rs.adjustments.get(d)
                self.follow_checks[d].setChecked(v is None)
                self.target_sliders[d].setEnabled(v is not None)
                if v is not None:
                    self.target_sliders[d].setValue(int(round(v)))
                self.target_values[d].setText("" if v is None else str(int(round(v))))
        finally:
            self._loading = False
        self._show_status()
        self._show_profile()
        self._refresh_applied()
        self._sync_enabled()
        self._schedule_plan()

    @staticmethod
    def _set(combo: QComboBox, data) -> None:
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _asset(self):
        rid = self._current_id()
        return self.ws.project.reference_assets.get(rid) if rid and self.ws.project else None

    def _show_status(self) -> None:
        a = self._asset()
        if a is None:
            self.status_label.setText("Import a reference video to begin." if self.ws.project is not None else "")
            self.stage_label.setText("")
            self.progress.setValue(0)
            self._show_thumbs(None)
            return
        text = STATUS_TEXT.get(a.analysis_status, a.analysis_status)
        if a.error:
            text += f": {a.error}"
        if self.ws.reference.is_stale(a.reference_id):
            text += " — the analysis is out of date (video, version or settings changed); analyze again."
        self.status_label.setText(text)
        if a.analysis_status in ("COMPLETED", "PARTIAL"):
            self.progress.setValue(100)
            self.stage_label.setText("")
        self.log_view.setPlainText(self.ws.reference.read_log(a.reference_id))
        self._show_thumbs(a)

    def _show_thumbs(self, asset) -> None:
        while self.thumb_row.count():
            item = self.thumb_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if asset is None or self.ws.project is None:
            return
        for rel in asset.thumbnails[:6]:
            f = self.ws.project.root / rel  # type: ignore[operator]
            if f.is_file():
                lab = QLabel()
                lab.setPixmap(QPixmap(str(f)).scaledToHeight(54, Qt.TransformationMode.SmoothTransformation))
                lab.setToolTip("A frame of your reference, kept only inside this project's references folder")
                self.thumb_row.addWidget(lab)
        self.thumb_row.addStretch(1)

    def _show_profile(self) -> None:
        prof = self.ws.reference.profile(self._current_id()) if self.ws.project is not None and self._current_id() else None
        self.profile_table.setRowCount(0)
        if prof is None:
            self.summary_label.setText("")
            self.warning_label.setText("")
            self.details_view.clear()
            self.reco_label.setText("")
            for d in DIMENSIONS:
                self.ref_values[d].setText("—")
            self.compare_table.setRowCount(0)
            self.similarity_label.setText("")
            return
        analysis = self.ws.reference.analysis(self._current_id())
        self.summary_label.setText(analysis.summary if analysis else "\n".join(prof.summary_lines()))
        rows = prof.rows()
        self.profile_table.setRowCount(len(rows))
        for r, (dim, label, score, reading, conf) in enumerate(rows):
            vals = (label, reading.upper() if reading == "UNAVAILABLE" else reading, "—" if reading == "UNAVAILABLE" else f"{score:.0f}", conf)
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, dim)
                if conf in ("Low", "LOW") or (reading != "UNAVAILABLE" and prof.dimension_confidence(dim) < 0.5):
                    item.setToolTip("Low confidence: treat this reading as an estimate.")
                self.profile_table.setItem(r, c, item)
            self.ref_values[dim].setText("—" if reading == "UNAVAILABLE" else f"{score:.0f}")
        self.warning_label.setText("\n".join(prof.warnings))
        self.details_view.setPlainText(profile_details(prof))
        self.reco_label.setText("\n".join("• " + t for t in (analysis.recommendations if analysis else [])))
        self._fill_compare()

    def _fill_compare(self) -> None:
        try:
            rows, sim = self.ws.reference.comparison(self._current_id())
        except Exception:  # noqa: BLE001  (the comparison is informational: never let it break the page)
            rows, sim = [], None
        self.compare_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate((row.label, row.reference, row.project)):
                self.compare_table.setItem(r, c, QTableWidgetItem(str(text)))
        if sim is not None:
            self.similarity_label.setText(f"Editing-style similarity: {sim.overall:.0f}%  (pacing {sim.pacing_match:.0f}% · motion {sim.motion_match:.0f}% · captions {sim.caption_match:.0f}% · "
                                          f"audio {sim.audio_match:.0f}%). This compares editing features only — not content, and not a copyright check.")
        else:
            self.similarity_label.setText("")

    # ================================================================ controls -> plan
    def _on_controls_changed(self, *_args) -> None:
        if self._loading:
            return
        self._sync_enabled()
        self._persist_settings()
        self._schedule_plan()

    def _on_follow(self, dim: str, follow: bool) -> None:
        self.target_sliders[dim].setEnabled(not follow)
        if not follow and not self._loading:
            ref = self.ws.reference.profile(self._current_id()) if self._current_id() else None
            if ref is not None:
                self.target_sliders[dim].setValue(int(round(ref.score(dim))))
        self.target_values[dim].setText("" if follow else str(self.target_sliders[dim].value()))
        if not self._loading:
            self._persist_settings()
            self._schedule_plan()

    def _on_slider(self, dim: str, value: int) -> None:
        self.target_values[dim].setText(str(value))
        if not self._loading and not self.follow_checks[dim].isChecked():
            self._persist_settings()
            self._schedule_plan()

    def _collect(self) -> dict:
        adj = {d: float(self.target_sliders[d].value()) for d in DIMENSIONS if not self.follow_checks[d].isChecked()}
        return {"application_mode": self.mode_combo.currentData(), "style_strength": float(self.strength_combo.currentData()),
                "preserve_user_edits": self.preserve_check.isChecked(), "custom_dimensions": [d for d, cb in self.dim_checks.items() if cb.isChecked()], "adjustments": adj}

    def _persist_settings(self) -> None:
        if self.ws.project is None or self._loading:
            return
        s = self._collect()
        cur = self.ws.project.reference_settings
        if all(getattr(cur, k) == v for k, v in s.items()):
            return
        self.ctx.guard(self, lambda: self.ws.reference.update_settings(**s))

    def _schedule_plan(self) -> None:
        self._replan_timer.start()

    def _replan(self) -> None:
        self._plan, self._plan_error = None, ""
        if self.ws.project is None or not self._current_id() or self.ws.reference.profile(self._current_id()) is None:
            self.plan_view.setText("")
            for d in DIMENSIONS:
                self.result_values[d].setText("—")
            self.sim_table.setRowCount(0)
            self.sim_label.setText("")
            self._sync_enabled()
            return
        s = self._collect()
        try:
            self._plan = self.ws.reference.plan(self._current_id(), adjustments=s["adjustments"], mode=s["application_mode"], strength=s["style_strength"],
                                                custom_dimensions=s["custom_dimensions"], preserve_user_edits=s["preserve_user_edits"])
        except Exception as exc:  # noqa: BLE001
            self._plan_error = str(getattr(exc, "user_message", exc))
        self._show_plan()
        self._sync_enabled()

    def _show_plan(self) -> None:
        plan = self._plan
        if plan is None:
            self.plan_view.setText(self._plan_error)
            for d in DIMENSIONS:
                self.result_values[d].setText("—")
            self.sim_table.setRowCount(0)
            self.sim_label.setText("")
            return
        lines = []
        for k, v in plan.overrides.active().items():
            label, fmt = OVERRIDE_LABELS.get(k, (k, "{}"))
            lines.append(f"{label}: {fmt.format(v)}")
        text = ("Will set: " + " · ".join(lines)) if lines else "Nothing would change with these choices."
        if plan.skipped:
            text += "\nNot applied: " + "; ".join(f"{DIMENSION_LABELS.get(k, k)} ({v})" for k, v in plan.skipped.items())
        if plan.notes:
            text += "\n" + "\n".join("• " + n for n in plan.notes[:6])
        if plan.warnings:
            text += "\n" + "\n".join("⚠ " + w for w in plan.warnings)
        self.plan_view.setText(text)
        for d in DIMENSIONS:
            eff = plan.effective_targets.get(d)
            self.result_values[d].setText("—" if eff is None else f"{eff:.0f}")
            if d in plan.skipped:
                self.result_values[d].setToolTip(plan.skipped[d])
            else:
                self.result_values[d].setToolTip("")
        self._fill_simulation()

    def _on_sim_toggle(self, on: bool) -> None:
        self.sim_table.setVisible(on)
        self.sim_label.setVisible(on)
        self._fill_simulation()

    def _fill_simulation(self) -> None:
        plan = self._plan
        if plan is None or plan.projected is None or not self.sim_check.isChecked():
            self.sim_table.setRowCount(0)
            self.sim_label.setText("")
            return
        proj = plan.projected
        mine = self.ws.reference.project_profile()
        ref = self.ws.reference.profile(self._current_id())
        self.sim_table.setRowCount(len(DIMENSIONS))
        for r, d in enumerate(DIMENSIONS):
            vals = (DIMENSION_LABELS[d], f"{mine.score(d):.0f}", f"{proj.scores.get(d, mine.score(d)):.0f}", "—" if ref is None or not ref.is_available(d) else f"{ref.score(d):.0f}")
            for c, text in enumerate(vals):
                self.sim_table.setItem(r, c, QTableWidgetItem(text))
        self.sim_label.setText(f"Style similarity to the reference: {proj.similarity_before:.0f}% now → {proj.similarity_after:.0f}% after applying. "
                               "A preview of editing tendencies only; the timeline is not changed until you regenerate it in AI Edit.")

    # ================================================================ apply / customize
    def toggle_customize(self) -> None:
        self.custom_group.setVisible(not self.custom_group.isVisible())

    def cancel_customize(self) -> None:
        """Discard the slider targets (back to following the reference) and close the Customize area. Nothing is applied."""
        self._loading = True
        try:
            for d in DIMENSIONS:
                self.follow_checks[d].setChecked(True)
                self.target_sliders[d].setEnabled(False)
                self.target_values[d].setText("")
        finally:
            self._loading = False
        self.custom_group.setVisible(False)
        self._persist_settings()
        self._schedule_plan()
        self._replan_timer.stop()
        self._replan()

    def apply_style(self) -> None:
        if self._plan is None:
            self._replan()
        plan = self._plan
        if plan is None:
            self.status(self._plan_error or "Analyze a reference first.")
            return
        if not confirm_apply_style(self):
            return
        if self.ctx.guard(self, lambda: self.ws.reference.apply_style(plan), modal=True, title="Apply reference style"):
            self.status("Reference style applied to your editing preferences. Regenerate in AI Edit to use it.")
            self._refresh_applied()

    def remove_style(self) -> None:
        self.ctx.guard(self, self.ws.reference.clear_style, modal=True, title="Reference style")
        self._refresh_applied()

    def undo_last(self) -> None:
        self.ctx.guard(self, self.ws.reference.revert_last_application, modal=True, title="Reference style")
        self._refresh_applied()

    def submit_request(self) -> None:
        text = self.request_edit.text().strip()
        if not text:
            return
        res = {}

        def run() -> None:
            res["r"] = self.ws.reference.guard_request(text)

        if self.ctx.guard(self, run, modal=True, title="Style request") and "r" in res:
            r = res["r"]
            if r.flagged:
                self.request_feedback.setText("This asks to reproduce specific content, so it was turned into an abstract editing instruction:\n“" + r.abstract_instruction + "”")
            else:
                self.request_feedback.setText("Noted: " + (r.abstract_instruction or text))

    def _refresh_applied(self) -> None:
        p = self.ws.project
        if p is None:
            self.applied_label.setText("")
            self._sync_enabled()
            return
        rs, ov = p.reference_settings, p.reference_style_overrides
        if rs.enabled and not ov.is_empty:
            parts = [f"{OVERRIDE_LABELS.get(k, (k, '{}'))[0]} {OVERRIDE_LABELS.get(k, (k, '{}'))[1].format(v)}" for k, v in ov.active().items()]
            self.applied_label.setText(f"APPLIED — {rs.application_mode.title()} at {int(rs.style_strength * 100)}%: " + " · ".join(parts) +
                                       "\nThe timeline is unchanged until you generate or regenerate it in AI Edit; your locked and manual edits are preserved.")
        else:
            self.applied_label.setText("No reference style is applied. The AI editor uses your own settings.")
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        p = self.ws.project
        has = p is not None
        a = self._asset() if has else None
        job = self.ws.reference.job_for(a.reference_id) if a is not None else None
        running = bool(job is not None and not job.status.is_terminal)
        analyzed = bool(a is not None and a.analysis_status in ("COMPLETED", "PARTIAL"))
        custom = self.mode_combo.currentData() == "CUSTOM"
        self.import_btn.setEnabled(has)
        self.link_check.setEnabled(has)
        self.analyze_btn.setEnabled(a is not None and not running and not analyzed)
        self.reanalyze_btn.setEnabled(a is not None and not running)
        self.cancel_btn.setEnabled(running)
        self.retry_btn.setEnabled(a is not None and not running and a.analysis_status in ("FAILED", "CANCELED"))
        self.remove_btn.setEnabled(a is not None and not running)
        self.custom_dims.setVisible(custom)
        for w in (self.mode_combo, self.strength_combo, self.preserve_check, self.customize_btn, self.cancel_style_btn, self.request_edit, self.request_btn):
            w.setEnabled(has and analyzed)
        self.apply_btn.setEnabled(has and analyzed and self._plan is not None and not self._plan.overrides.is_empty)
        self.remove_style_btn.setEnabled(has and bool(p and (p.reference_settings.enabled or not p.reference_style_overrides.is_empty)))
        self.undo_style_btn.setEnabled(has and bool(p and p.style_application_history and p.style_application_history[-1].before is not None))
        for d in DIMENSIONS:
            prof = self.ws.reference.profile(self._current_id()) if has and analyzed else None
            avail = prof is not None and prof.is_available(d)
            self.follow_checks[d].setEnabled(avail)
            if not avail:
                self.target_sliders[d].setEnabled(False)
