"""Scene review: run analysis, inspect scenes, split / merge / edit / approve."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.analysis.models import Scene, SceneStatus, VisualType
from app.analysis.segmenter import SegmentationParams
from app.core.exceptions import AppError, UserEditsPresentError
from app.core.timecode import format_timecode
from app.jobs.job import Job, JobStatus
from app.preview.player import QtPreviewPlayer
from app.services.scene_service import SceneState
from app.transcription.status import TranscriptStatus, transcript_status
from app.ui.context import UiContext
from app.ui.theme import palette

COLUMNS = ("Scene", "Time", "Topic", "Visual", "Importance", "Confidence", "Status")
GRANULARITY = {"Fewer, longer scenes": 0.75, "Balanced": 0.60, "More, shorter scenes": 0.45}
STATUS_TEXT = {SceneStatus.READY: "Ready", SceneStatus.NEEDS_REVIEW: "Needs review", SceneStatus.APPROVED: "Approved ✓",
               SceneStatus.PENDING: "Pending", SceneStatus.FAILED: "FAILED"}


def _short(t: float) -> str:
    """Compact time for table cells (m:ss.s); full precision is shown in the detail pane."""
    return f"{int(t // 60)}:{t % 60:04.1f}"


class ScenePanel(QWidget):
    def __init__(self, ctx: UiContext, voice_player: QtPreviewPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.player = voice_player
        self._job: Job | None = None
        self._selected: str | None = None
        self._play_until: float | None = None
        self._loading = False

        title = QLabel("Scenes")
        title.setObjectName("title")
        self.analyze_btn = QPushButton("Run Scene Analysis")
        self.analyze_btn.setObjectName("primary")
        self.granularity = QComboBox()
        self.granularity.addItems(list(GRANULARITY))
        self.granularity.setCurrentText("Balanced")
        self.granularity.setToolTip("How readily the AI starts a new scene")
        self.state_label = QLabel()
        self.state_label.setObjectName("sceneState")
        self.state_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setObjectName("sceneProgress")
        self.progress_msg = QLabel()
        self.progress_msg.setObjectName("muted")
        self.failure_box = QFrame()
        self.failure_box.setObjectName("failureBox")
        fl = QHBoxLayout(self.failure_box)
        self.failure_label = QLabel()
        self.failure_label.setWordWrap(True)
        self.retry_btn = QPushButton("Retry From Scene")
        self.retry_btn.setObjectName("primary")
        fl.addWidget(self.failure_label, 1)
        fl.addWidget(self.retry_btn)
        self.review_only = QCheckBox("Needs review only")

        top = QHBoxLayout()
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(QLabel("Granularity"))
        top.addWidget(self.granularity)
        top.addWidget(self.analyze_btn)

        # ---- scene table ----
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("sceneTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for col, w in ((0, 50), (1, 118), (3, 100), (4, 90), (5, 90), (6, 100)):
            self.table.setColumnWidth(col, w)
        self.table.setMinimumWidth(640)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.review_only)
        ll.addWidget(self.table, 1)

        # ---- detail ----
        self.d_title = QLabel("Select a scene")
        self.d_title.setObjectName("title")
        self.d_time = QLabel()
        self.d_time.setObjectName("muted")
        self.d_metrics = QLabel()
        self.narration = QPlainTextEdit()
        self.narration.setReadOnly(True)
        self.narration.setMaximumHeight(90)
        self.topic = QLineEdit()
        self.summary = QPlainTextEdit()
        self.summary.setMaximumHeight(62)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Your notes")
        form = QFormLayout()
        form.addRow("Narration", self.narration)
        form.addRow("Topic", self.topic)
        form.addRow("Summary", self.summary)
        form.addRow("Notes", self.notes)

        self.vtype = QComboBox()
        self.vtype.addItems([t.value for t in VisualType])
        self.primary = QLineEdit()
        self.secondary = QLineEdit()
        self.action = QLineEdit()
        self.context = QLineEdit()
        self.preferred = QLabel()
        self.preferred.setWordWrap(True)
        self.avoid = QLabel()
        self.avoid.setWordWrap(True)
        self.intent_note = QLabel()
        self.intent_note.setObjectName("muted")
        intent = QGroupBox("Visual intent")
        fi = QFormLayout(intent)
        fi.addRow("Type", self.vtype)
        fi.addRow("Primary subject", self.primary)
        fi.addRow("Secondary subject", self.secondary)
        fi.addRow("Action", self.action)
        fi.addRow("Context", self.context)
        fi.addRow("Suggested visuals", self.preferred)
        fi.addRow("Avoid", self.avoid)
        fi.addRow("", self.intent_note)

        self.extracted = QPlainTextEdit()
        self.extracted.setReadOnly(True)
        self.extracted.setObjectName("sceneExtracted")
        self.extracted.setMinimumHeight(110)
        self.rationale = QLabel()
        self.rationale.setObjectName("muted")
        self.rationale.setWordWrap(True)

        self.play_btn = QPushButton("Play scene")
        self.merge_prev = QPushButton("Merge with previous")
        self.merge_next = QPushButton("Merge with next")
        self.split_at = QDoubleSpinBox()
        self.split_at.setDecimals(3)
        self.split_at.setRange(0, 100000)
        self.split_at.setSuffix(" s")
        self.split_at.setKeyboardTracking(False)
        self.use_playhead = QPushButton("Use playback position")
        self.split_btn = QPushButton("Split")
        self.apply_btn = QPushButton("Apply Edit")
        self.approve_btn = QPushButton("Approve")
        self.approve_btn.setObjectName("primary")
        row1 = QHBoxLayout()
        for w in (self.play_btn, self.merge_prev, self.merge_next):
            row1.addWidget(w)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        for w in (QLabel("Split at"), self.split_at, self.use_playhead, self.split_btn):
            row2.addWidget(w)
        row2.addStretch(1)
        row2.addWidget(self.apply_btn)
        row2.addWidget(self.approve_btn)

        detail = QWidget()
        dl = QVBoxLayout(detail)
        for w in (self.d_title, self.d_time, self.d_metrics):
            dl.addWidget(w)
        dl.addLayout(form)
        dl.addWidget(intent)
        dl.addWidget(QLabel("Entities, claims and numbers"))
        dl.addWidget(self.extracted)
        dl.addWidget(self.rationale)
        dl.addLayout(row1)
        dl.addLayout(row2)
        dl.addStretch(1)
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setWidget(detail)
        self.detail = detail

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.detail_scroll)
        split.setSizes([780, 600])
        split.setStretchFactor(0, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.state_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_msg)
        layout.addWidget(self.failure_box)
        layout.addWidget(split, 1)

        # ---- wiring ----
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.retry_btn.clicked.connect(lambda: self._start(lambda: self.ctx.ws.scenes.retry_failed()))
        self.review_only.toggled.connect(lambda _v: self.refresh())
        self.table.itemSelectionChanged.connect(self._on_select)
        self.play_btn.clicked.connect(self._play_scene)
        self.merge_prev.clicked.connect(lambda: self._edit(lambda s: ctx.ws.scenes.merge_with_previous(s)))
        self.merge_next.clicked.connect(lambda: self._edit(lambda s: ctx.ws.scenes.merge_with_next(s)))
        self.split_btn.clicked.connect(lambda: self._edit(lambda s: ctx.ws.scenes.split_scene(s, self.split_at.value())))
        self.use_playhead.clicked.connect(lambda: self.split_at.setValue(self.player.position))
        self.apply_btn.clicked.connect(self._apply_edit)
        self.approve_btn.clicked.connect(self._toggle_approve)
        self.player.position_changed.connect(self._on_position)

        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self._reset())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("scenes", "transcript", "script", "voice_over", "assets") else None)
        b.on("job.updated", self._on_job)
        self._reset()

    # ------------------------------------------------------------ analysis
    def _params(self) -> SegmentationParams:
        return SegmentationParams(threshold=GRANULARITY[self.granularity.currentText()])

    def run_analysis(self) -> None:
        def go() -> None:
            ws = self.ctx.ws
            overwrite = False
            touched = ws.scenes.user_touched_labels()
            if touched:
                shown = ", ".join(touched[:12]) + ("…" if len(touched) > 12 else "")
                answer = QMessageBox.warning(
                    self, "Replace your scene edits?",
                    f"Scenes you edited, approved or split ({shown}) will be replaced by a new AI analysis.\n"
                    "You can undo this afterwards. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
                if answer != QMessageBox.StandardButton.Yes:
                    return
                overwrite = True
            self._job = ws.scenes.analyze(force=bool(ws.project and ws.project.scenes), overwrite_user_edits=overwrite, params=self._params())

        self._start(go)

    def _start(self, action) -> None:
        self.ctx.guard(self, action, modal=True, title="Scene analysis")
        self.refresh()

    def _running(self) -> bool:
        return self._job is not None and self._job.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    def _on_job(self, payload: dict) -> None:
        job: Job = payload["job"]
        if self._job is not None and job.id == self._job.id:
            self.refresh()

    # ------------------------------------------------------------ view
    def _reset(self) -> None:
        self._job = None
        self._selected = None
        self.refresh()

    def _visible_scenes(self) -> list[Scene]:
        project = self.ctx.ws.project
        scenes = project.scenes if project else []
        if self.review_only.isChecked():
            scenes = [s for s in scenes if s.status in (SceneStatus.NEEDS_REVIEW, SceneStatus.FAILED, SceneStatus.PENDING)]
        return scenes

    def refresh(self) -> None:
        ws = self.ctx.ws
        project = ws.project
        running = self._running()
        has_tr = bool(project and project.transcription.transcript)
        t_status = transcript_status(project) if project else TranscriptStatus.NOT_STARTED
        self.analyze_btn.setEnabled(bool(project) and has_tr and t_status is TranscriptStatus.COMPLETE and not running)
        self.analyze_btn.setText("Re-run Scene Analysis" if project and project.scenes else "Run Scene Analysis")
        self.progress.setVisible(running)
        self.progress_msg.setVisible(running)
        if running and self._job:
            self.progress.setValue(int(self._job.progress))
            self.progress_msg.setText(self._job.message)

        # status line
        if project is None:
            msg = ""
        elif not has_tr:
            msg = "Transcribe the voice-over first (Voice page). Scenes are aligned to the spoken timing."
        elif t_status is not TranscriptStatus.COMPLETE:
            msg = "⚠ The transcript is outdated because the voice-over changed. Re-transcribe before analysing scenes."
        elif running:
            msg = "Analysing scenes…"
        elif not project.scenes:
            msg = "No scenes yet. Run Scene Analysis."
        else:
            state = ws.scenes.state()
            n_review = sum(s.status in (SceneStatus.NEEDS_REVIEW, SceneStatus.FAILED, SceneStatus.PENDING) for s in project.scenes)
            base = f"{len(project.scenes)} scenes • {n_review} need review • overall topic: {project.scene_analysis.overall_topic or '—'}"
            msg = {SceneState.UP_TO_DATE: base, SceneState.PARTIAL: base,
                   SceneState.OUTDATED: base + "  ⚠ Outdated: the transcript, script or settings changed since these scenes were generated "
                                               "(your scenes are untouched; re-run to refresh)."}.get(state, base)
        self.state_label.setText(msg)

        # failure box
        failed = bool(project and project.scene_analysis.status == "PARTIAL" and project.scenes)
        self.failure_box.setVisible(failed and not running)
        if failed and project:
            done = sum(s.status not in (SceneStatus.PENDING, SceneStatus.FAILED) for s in project.scenes)
            first = next((i for i, s in enumerate(project.scenes) if s.status in (SceneStatus.FAILED, SceneStatus.PENDING)), 0)
            self.failure_label.setText(
                f"Scenes 1–{done} complete.\nScene {first + 1} failed: {project.scene_analysis.failed_error or 'unknown error'}")
            self.retry_btn.setText(f"Retry From Scene {first + 1}")

        self._fill_table()

    def _fill_table(self) -> None:
        project = self.ctx.ws.project
        c = palette(self.ctx.ws.settings.theme)
        scenes = self._visible_scenes()
        self._loading = True
        try:
            self.table.setRowCount(len(scenes))
            for r, s in enumerate(scenes):
                intent = project.visual_intents.get(s.id) if project else None
                cells = [s.label, f"{_short(s.start)} → {_short(s.end)}", s.topic or "—",
                         intent.type.value if intent else "—",
                         f"{s.importance:.0%}" if intent else "—", f"{s.segmentation_confidence:.0%}", STATUS_TEXT[s.status]]
                color = {SceneStatus.NEEDS_REVIEW: "#e0a030", SceneStatus.APPROVED: c["ok"], SceneStatus.FAILED: c["danger"],
                         SceneStatus.PENDING: c["muted"]}.get(s.status)
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, s.id)
                    if color and col == 6:
                        item.setForeground(QBrush(QColor(color)))
                    if s.user_edited_fields and col == 2:
                        item.setToolTip("Edited by you: " + ", ".join(s.user_edited_fields))
                    self.table.setItem(r, col, item)
                if s.id == self._selected:
                    self.table.selectRow(r)
        finally:
            self._loading = False
        if not any(s.id == self._selected for s in scenes):
            self._selected = None
        self._show_detail()

    # ------------------------------------------------------------ selection / detail
    def select_scene(self, scene_id: str) -> None:
        self._selected = scene_id
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == scene_id:
                self.table.selectRow(r)
                return
        self._show_detail()

    def _on_select(self) -> None:
        if self._loading:
            return
        rows = self.table.selectionModel().selectedRows()
        self._selected = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole) if rows else None
        self._show_detail()

    def current_scene(self) -> Scene | None:
        project = self.ctx.ws.project
        return next((s for s in project.scenes if s.id == self._selected), None) if project and self._selected else None

    def _show_detail(self) -> None:
        s = self.current_scene()
        project = self.ctx.ws.project
        self.detail.setEnabled(s is not None)
        self._loading = True
        try:
            if s is None or project is None:
                self.d_title.setText("Select a scene")
                for w in (self.d_time, self.d_metrics, self.rationale, self.preferred, self.avoid, self.intent_note):
                    w.setText("")
                for w in (self.narration, self.summary, self.extracted):
                    w.setPlainText("")
                for w in (self.topic, self.notes, self.primary, self.secondary, self.action, self.context):
                    w.setText("")
                return
            intent = project.visual_intents.get(s.id)
            analysed = s.status not in (SceneStatus.PENDING, SceneStatus.FAILED)
            self.d_title.setText(f"Scene {s.label}")
            self.d_time.setText(f"{format_timecode(s.start)} → {format_timecode(s.end)}   ({s.duration:.1f}s)")
            self.d_metrics.setText(
                f"Importance: {s.importance:.0%}    Confidence: {s.segmentation_confidence:.0%}    Status: {STATUS_TEXT[s.status]}"
                if analysed else f"Status: {STATUS_TEXT[s.status]} — not analysed yet")
            self.narration.setPlainText(s.narration)
            self.topic.setText(s.topic)
            self.summary.setPlainText(s.summary)
            self.notes.setText(s.notes)
            if intent:
                self.vtype.setCurrentText(intent.type.value)
                self.primary.setText(intent.primary_subject)
                self.secondary.setText(intent.secondary_subject)
                self.action.setText(intent.action)
                self.context.setText(intent.context)
                self.preferred.setText("\n".join("• " + v for v in intent.preferred_visuals) or "—")
                self.avoid.setText("\n".join("• " + v for v in intent.avoid) or "—")
                others = ", ".join(t.value for t in intent.secondary_types)
                self.intent_note.setText(
                    f"{'Set by you' if intent.author.value == 'USER' else 'AI suggestion'} • type confidence {intent.confidence:.0%}"
                    + (f" • also: {others}" if others else "")
                    + (f" • context inherited from {intent.inherited_from}" if intent.inherited_from else ""))
            lines = []
            if s.entities:
                lines.append("Entities: " + ", ".join(f"{e.text} ({e.type.value.replace('_', ' ').title()})" for e in s.entities))
            for cl in s.claims:
                ev = "needs evidence" if cl.requires_evidence else "no evidence needed"
                lines.append(f"Claim [{cl.type.value}] {cl.text}  —  {ev}; evidence: {cl.evidence_status.replace('_', ' ').lower()}")
            if s.numbers:
                lines.append("Numbers & dates: " + ", ".join(f"{n.text} ({n.kind.value.replace('_', ' ').title()})" for n in s.numbers))
            self.extracted.setPlainText("\n".join(lines) or "Nothing extracted.")
            self.rationale.setText("Why: " + " • ".join(s.rationale) if s.rationale else "")
            self.approve_btn.setText("Unapprove" if s.status is SceneStatus.APPROVED else "Approve")
            idx = project.scenes.index(s)
            for w in (self.vtype, self.primary, self.secondary, self.action, self.context):
                w.setEnabled(intent is not None)
            self.merge_prev.setEnabled(idx > 0)
            self.merge_next.setEnabled(idx < len(project.scenes) - 1)
            for w in (self.split_btn, self.apply_btn, self.approve_btn, self.merge_prev, self.merge_next):
                w.setEnabled(w.isEnabled() and analysed)
            self.split_at.setRange(s.start, s.end)
            self.split_at.setValue(s.start + s.duration / 2)
        finally:
            self._loading = False

    # ------------------------------------------------------------ actions
    def _edit(self, action) -> None:
        s = self.current_scene()
        if s is None:
            return
        self.ctx.guard(self, lambda: action(s.id), modal=True, title="Scene edit")
        self.refresh()

    def _apply_edit(self) -> None:
        s = self.current_scene()
        project = self.ctx.ws.project
        if s is None or project is None:
            return
        intent = project.visual_intents.get(s.id)
        changes: dict = {}
        if intent:
            for key, widget in (("primary_subject", self.primary), ("secondary_subject", self.secondary),
                                ("action", self.action), ("context", self.context)):
                if widget.text() != getattr(intent, key):
                    changes[key] = widget.text()
            if self.vtype.currentText() != intent.type.value:
                changes["type"] = self.vtype.currentText()
        kwargs = {"topic": self.topic.text() if self.topic.text() != s.topic else None,
                  "summary": self.summary.toPlainText() if self.summary.toPlainText() != s.summary else None,
                  "notes": self.notes.text() if self.notes.text() != s.notes else None,
                  "intent": changes or None}
        if all(v is None for v in kwargs.values()):
            self.ctx.status("No changes to apply.")
            return
        self._edit(lambda sid: self.ctx.ws.scenes.edit_scene(sid, **kwargs))

    def _toggle_approve(self) -> None:
        s = self.current_scene()
        if s is not None:
            self._edit(lambda sid: self.ctx.ws.scenes.set_approved(sid, s.status is not SceneStatus.APPROVED))

    def _play_scene(self) -> None:
        s = self.current_scene()
        if s is not None:
            self._play_until = s.end
            self.player.seek(s.start)
            self.player.play()

    def _on_position(self, t: float) -> None:
        if self._play_until is not None and t >= self._play_until:
            self._play_until = None
            self.player.pause()

    def pause(self) -> None:
        self._play_until = None
        self.player.pause()
