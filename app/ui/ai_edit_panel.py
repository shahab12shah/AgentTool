"""AI Edit page: style and controls, progress, scene status, scrubbing preview and the AI Decision Inspector."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.editing.models import Creator, DecisionType, EditingDecision, PanKind, SceneEditStatus, TextStyle, TransitionType, ZoomKind, confidence_label
from app.editing.presets import PRESETS
from app.preview.player import QtPreviewPlayer
from app.ui.context import UiContext
from app.ui.timeline_preview import TimelinePreview

SCENE_COLUMNS = ("Scene", "Time", "Visual", "Edit", "Decisions", "Min. confidence")
EDIT_TEXT = {SceneEditStatus.PENDING: "Pending", SceneEditStatus.COMPLETE: "Edited ✓", SceneEditStatus.FAILED: "FAILED", SceneEditStatus.NEEDS_VISUAL: "Needs visual",
             SceneEditStatus.OUTDATED: "Scene changed", SceneEditStatus.LOCKED: "Locked 🔒"}
VISUAL_TEXT = {"APPROVED": "Approved", "MISSING": "MISSING", "UNAPPROVED": "Not approved", "SKIPPED": "Skipped", "MISSING_MEDIA": "MISSING MEDIA"}
LOW = 70.0

# decision type -> editable parameters: (key path, label, kind, extra)
ENUMS = {"zoom": [k.value for k in ZoomKind], "pan": [k.value for k in PanKind], "transition": [t.value for t in TransitionType], "style": [s.value for s in TextStyle]}
PARAM_SPECS: dict[DecisionType, list[tuple[str, str, str, object]]] = {
    DecisionType.VISUAL_TIMING: [("start", "Start (s)", "float", (0, 100000)), ("duration", "Duration (s)", "float", (0.05, 100000)),
                                 ("source_in", "Source start (s)", "float", (0, 100000))],
    DecisionType.ZOOM: [("kind", "Zoom", "enum", "zoom"), ("start_scale", "Start scale", "float", (0.1, 6)), ("end_scale", "End scale", "float", (0.1, 6))],
    DecisionType.PAN: [("kind", "Movement", "enum", "pan"), ("start_scale", "Start scale", "float", (0.1, 6)), ("end_scale", "End scale", "float", (0.1, 6)),
                       ("start_pos.0", "Start X (px)", "float", (-9999, 9999)), ("start_pos.1", "Start Y (px)", "float", (-9999, 9999)),
                       ("end_pos.0", "End X (px)", "float", (-9999, 9999)), ("end_pos.1", "End Y (px)", "float", (-9999, 9999))],
    DecisionType.TEXT: [("content", "Text", "str", None), ("style", "Style", "enum", "style"), ("start", "Start (s)", "float", (0, 100000)),
                        ("duration", "Duration (s)", "float", (0.05, 100)), ("size", "Size", "int", (8, 400)), ("position.0", "Position X", "float", (0, 1)),
                        ("position.1", "Position Y", "float", (0, 1))],
    DecisionType.EVIDENCE_FOCUS: [("region.0", "Region X", "float", (0, 1)), ("region.1", "Region Y", "float", (0, 1)), ("region.2", "Region width", "float", (0.02, 1)),
                                  ("region.3", "Region height", "float", (0.02, 1)), ("zoom_scale", "Zoom", "float", (1, 6)), ("highlight", "Highlight box", "bool", None),
                                  ("darken", "Darken surround", "bool", None)],
    DecisionType.TRANSITION: [("type", "Transition", "enum", "transition"), ("duration", "Duration (s)", "float", (0, 5))],
    DecisionType.AUDIO_DUCK: [("start", "Start (s)", "float", (0, 100000)), ("end", "End (s)", "float", (0, 100000)), ("music_level", "Music level", "float", (0, 1))],
    DecisionType.CAPTION_EMPHASIS: [("words", "Emphasis words (comma separated)", "list", None)],
    DecisionType.TRIM: [("source_in", "Source start (s)", "float", (0, 100000))],
}
PARAM_SPECS[DecisionType.NUMBER_EMPHASIS] = PARAM_SPECS[DecisionType.TEXT]


def _get(params: dict, path: str):
    cur = params
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _set(params: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = params
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(cur, dict):
            if part not in cur or cur[part] is None:
                cur[part] = [0.0, 0.0, 0.0, 0.0] if nxt.isdigit() else {}
            cur = cur[part]
        else:
            cur = cur[int(part)]
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        cur[int(last)] = value


class AIEditPanel(QWidget):
    def __init__(self, ctx: UiContext, voice_player: QtPreviewPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.player = voice_player
        self.scene_id: str | None = None
        self.decision_id: str | None = None
        self._loading = False
        self._editors: dict[str, tuple[QWidget, str]] = {}
        self.go_to_research = None  # set by the main window: callable(scene_id)

        title = QLabel("AI ADVANCED EDITING")
        title.setObjectName("title")
        note = QLabel("The AI creates the edit; you own it. Everything becomes the normal editable timeline plus an inspectable decision for each element.")
        note.setObjectName("muted")
        note.setWordWrap(True)

        # ---- settings
        self.style = QComboBox()
        for key, preset in PRESETS.items():
            self.style.addItem(preset.label, key)
        self.style.setToolTip("\n".join(f"{p.label}: {p.description}" for p in PRESETS.values()))
        self.motion = QSlider(Qt.Orientation.Horizontal)
        self.pacing = QSlider(Qt.Orientation.Horizontal)
        for s in (self.motion, self.pacing):
            s.setRange(0, 100)
        self.transitions_freq = QSlider(Qt.Orientation.Horizontal)
        self.transitions_freq.setRange(0, 100)
        self.text_emphasis = QCheckBox("Text emphasis")
        self.number_emphasis = QCheckBox("Number emphasis")
        self.evidence = QCheckBox("Evidence treatment")
        self.smart_transitions = QCheckBox("Smart transitions")
        self.smart_audio = QCheckBox("Smart audio ducking")
        self.captions = QCheckBox("Caption placement instructions")
        form = QFormLayout()
        form.addRow("Editing style", self.style)
        row = QHBoxLayout()
        row.addWidget(QLabel("Low"))
        row.addWidget(self.motion, 1)
        row.addWidget(QLabel("High"))
        form.addRow("Motion intensity", row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Slow"))
        row.addWidget(self.pacing, 1)
        row.addWidget(QLabel("Fast"))
        form.addRow("Visual pacing", row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Rare"))
        row.addWidget(self.transitions_freq, 1)
        row.addWidget(QLabel("Frequent"))
        form.addRow("Transition frequency", row)
        for cb in (self.text_emphasis, self.number_emphasis, self.evidence, self.smart_transitions, self.smart_audio, self.captions):
            form.addRow("", cb)
        settings_box = QGroupBox("Editing settings")
        settings_box.setLayout(form)

        self.generate_btn = QPushButton("Generate AI Edit")
        self.generate_btn.setObjectName("primary")
        self.generate_sel_btn = QPushButton("Generate Selected Scenes")
        self.regen_all_btn = QPushButton("Regenerate Entire Edit")
        self.retry_btn = QPushButton("Retry Failed Scenes")
        self.cancel_btn = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.progress_label = QLabel()
        self.operation_label = QLabel()
        self.operation_label.setObjectName("muted")
        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.setPlaceholderText("Log")
        run_box = QGroupBox("Run")
        rl = QVBoxLayout(run_box)
        for b in (self.generate_btn, self.generate_sel_btn, self.regen_all_btn, self.retry_btn, self.cancel_btn):
            rl.addWidget(b)
        for w in (self.progress, self.progress_label, self.operation_label, self.state_label, self.log):
            rl.addWidget(w)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(settings_box)
        ll.addWidget(run_box)
        ll.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(330)

        # ---- scenes
        self.table = QTableWidget(0, len(SCENE_COLUMNS))
        self.table.setObjectName("editScenes")
        self.table.setHorizontalHeaderLabels(SCENE_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for col, w in ((0, 50), (1, 110), (2, 110), (4, 80), (5, 110)):
            self.table.setColumnWidth(col, w)
        self.missing_box = QFrame()
        self.missing_box.setObjectName("failureBox")
        ml = QVBoxLayout(self.missing_box)
        self.missing_label = QLabel()
        self.missing_label.setWordWrap(True)
        ml.addWidget(self.missing_label)
        mrow = QHBoxLayout()
        self.return_btn = QPushButton("Return to Research")
        self.manual_btn = QPushButton("Replace Manually…")
        self.skip_btn = QPushButton("Skip Visual")
        for b in (self.return_btn, self.manual_btn, self.skip_btn):
            mrow.addWidget(b)
        mrow.addStretch(1)
        ml.addLayout(mrow)

        # ---- preview
        self.preview = TimelinePreview(ctx)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.play_btn = QPushButton("Play")
        self.time_label = QLabel("0:00.0")
        self.preview_note = QLabel()
        self.preview_note.setObjectName("muted")
        self.preview_note.setWordWrap(True)
        prow = QHBoxLayout()
        prow.addWidget(self.play_btn)
        prow.addWidget(self.slider, 1)
        prow.addWidget(self.time_label)

        # ---- decisions + inspector
        self.decisions = QTableWidget(0, 5)
        self.decisions.setObjectName("decisionList")
        self.decisions.setHorizontalHeaderLabels(("Decision", "Start", "Duration", "Confidence", "By"))
        self.decisions.verticalHeader().setVisible(False)
        self.decisions.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.decisions.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.decisions.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.decisions.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.decisions.setMaximumHeight(190)

        self.i_title = QLabel("AI Decision")
        self.i_title.setObjectName("title")
        self.i_info = QLabel()
        self.i_info.setWordWrap(True)
        self.i_info.setTextFormat(Qt.TextFormat.RichText)
        self.param_form = QFormLayout()
        self.apply_btn = QPushButton("Apply changes")
        self.apply_btn.setObjectName("primary")
        self.lock_visual = QPushButton("Lock visual")
        self.lock_text = QPushButton("Lock text")
        self.lock_motion = QPushButton("Lock motion")
        self.lock_scene = QPushButton("Lock scene")
        self.regen_scene_btn = QPushButton("Regenerate Scene")
        self.replace_btn = QPushButton("Replace visual…")
        for b in (self.lock_visual, self.lock_text, self.lock_motion, self.lock_scene):
            b.setCheckable(True)
        self.inspector = QGroupBox("AI Decision Inspector")
        il = QVBoxLayout(self.inspector)
        il.addWidget(self.i_info)
        il.addLayout(self.param_form)
        il.addWidget(self.apply_btn)
        lrow = QHBoxLayout()
        for b in (self.lock_visual, self.lock_text, self.lock_motion, self.lock_scene):
            lrow.addWidget(b)
        il.addLayout(lrow)
        rrow = QHBoxLayout()
        rrow.addWidget(self.replace_btn)
        rrow.addWidget(self.regen_scene_btn)
        il.addLayout(rrow)

        right = QWidget()
        rl2 = QVBoxLayout(right)
        rl2.addWidget(self.preview, 3)
        rl2.addLayout(prow)
        rl2.addWidget(self.preview_note)
        rl2.addWidget(QLabel("Decisions for the selected scene"))
        rl2.addWidget(self.decisions)
        rl2.addWidget(self.inspector)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right)

        mid = QWidget()
        ml2 = QVBoxLayout(mid)
        ml2.addWidget(QLabel("Scenes"))
        ml2.addWidget(self.table, 1)
        ml2.addWidget(self.missing_box)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_scroll)
        split.addWidget(mid)
        split.addWidget(right_scroll)
        split.setSizes([340, 430, 620])
        split.setChildrenCollapsible(False)
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addWidget(split, 1)

        # ---- wiring
        for w in (self.text_emphasis, self.number_emphasis, self.evidence, self.smart_transitions, self.smart_audio, self.captions):
            w.toggled.connect(self._settings_changed)
        self.style.currentIndexChanged.connect(self._settings_changed)
        for s in (self.motion, self.pacing, self.transitions_freq):
            s.sliderReleased.connect(self._settings_changed)
            s.valueChanged.connect(lambda _v, s=s: self._settings_changed() if not s.isSliderDown() else None)
        self.generate_btn.clicked.connect(self._generate)
        self.generate_sel_btn.clicked.connect(self._generate_selected)
        self.regen_all_btn.clicked.connect(self._regen_all)
        self.retry_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.editing.retry_failed(), "Retry"))
        self.cancel_btn.clicked.connect(lambda: self.ctx.ws.editing.cancel())
        self.table.itemSelectionChanged.connect(self._on_select_scene)
        self.decisions.itemSelectionChanged.connect(self._on_select_decision)
        self.apply_btn.clicked.connect(self._apply)
        self.regen_scene_btn.clicked.connect(lambda: self._regen_scene())
        self.lock_visual.clicked.connect(lambda on: self._lock("VISUAL", on))
        self.lock_text.clicked.connect(lambda on: self._lock("TEXT", on))
        self.lock_motion.clicked.connect(lambda on: self._lock("MOTION", on))
        self.lock_scene.clicked.connect(lambda on: self._lock("SCENE", on))
        self.return_btn.clicked.connect(lambda: self.go_to_research(self.scene_id) if self.go_to_research and self.scene_id else None)
        self.manual_btn.clicked.connect(self._manual_add)
        self.replace_btn.clicked.connect(self._replace_visual)
        self.skip_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.research.skip_visual(self.scene_id), "Skip visual"))
        self.slider.valueChanged.connect(self._scrub)
        self.play_btn.clicked.connect(self._toggle_play)
        self.player.position_changed.connect(self._on_position)
        self.player.playing_changed.connect(lambda on: self.play_btn.setText("Pause" if on else "Play"))

        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self._reset())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("timeline", "research", "scenes", "editing", "assets", "voice_over") else None)
        b.on("job.updated", self._on_job)
        b.on("selection.changed", lambda p: self._on_clip_selected(p.get("clip_id")))
        self._reset()

    # ------------------------------------------------------------------ helpers
    def act(self, action, title: str = "AI edit") -> bool:
        ok = self.ctx.guard(self, action, modal=True, title=title)
        self.refresh()
        return ok

    def _selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows if self.table.item(r, 0)]

    def _reset(self) -> None:
        self.scene_id = self.decision_id = None
        self.refresh()

    # ------------------------------------------------------------------ settings
    def _load_settings(self) -> None:
        project = self.ctx.ws.project
        s = project.editing_settings if project else None
        self._loading = True
        try:
            if s is not None:
                self.style.setCurrentIndex(max(0, self.style.findData(s.style)))
                self.motion.setValue(int(round(s.motion_intensity * 100)))
                self.pacing.setValue(int(round(s.pacing * 100)))
                self.transitions_freq.setValue(int(round(s.transition_frequency * 100)))
                for cb, v in ((self.text_emphasis, s.text_emphasis), (self.number_emphasis, s.number_emphasis), (self.evidence, s.evidence_treatment),
                              (self.smart_transitions, s.smart_transitions), (self.smart_audio, s.smart_audio_ducking), (self.captions, s.caption_mode == "ENABLED")):
                    cb.setChecked(v)
        finally:
            self._loading = False

    def _settings_changed(self, *_a) -> None:
        if self._loading or self.ctx.ws.project is None:
            return
        self.ctx.guard(self, lambda: self.ctx.ws.editing.update_settings(
            style=self.style.currentData(), motion_intensity=self.motion.value() / 100, pacing=self.pacing.value() / 100,
            transition_frequency=self.transitions_freq.value() / 100, text_emphasis=self.text_emphasis.isChecked(),
            number_emphasis=self.number_emphasis.isChecked(), evidence_treatment=self.evidence.isChecked(),
            smart_transitions=self.smart_transitions.isChecked(), smart_audio_ducking=self.smart_audio.isChecked(),
            caption_mode="ENABLED" if self.captions.isChecked() else "DISABLED"))

    # ------------------------------------------------------------------ run
    def _generate(self) -> None:
        self.act(lambda: self.ctx.ws.editing.generate(), "Generate AI Edit")

    def _generate_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            self.ctx.status("Select one or more scenes first.")
            return
        self.act(lambda: self.ctx.ws.editing.generate(ids, force=True, scope="SELECTED"), "Generate selected scenes")

    def _regen_all(self) -> None:
        project = self.ctx.ws.project
        user_owned = sum(1 for d in project.editing_decisions.values() if d.created_by is Creator.USER or d.locked) if project else 0
        text = "Regenerate the entire edit?\n\nAI-created elements are replaced. "
        text += (f"Your {user_owned} edited or locked decision(s) are preserved." if user_owned else "You have not edited anything yet.") + "\nYou can undo this."
        if QMessageBox.question(self, "Regenerate entire edit", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes:
            self.act(lambda: self.ctx.ws.editing.regenerate_all(), "Regenerate")

    def _regen_scene(self) -> None:
        if self.scene_id:
            sid = self.scene_id
            self.act(lambda: self.ctx.ws.editing.regenerate_scene(sid), "Regenerate scene")

    def _on_job(self, payload: dict) -> None:
        if payload["job"].type == "ai_edit":
            self._refresh_progress()

    def _refresh_progress(self) -> None:
        ed = self.ctx.ws.editing
        pr = ed.progress
        running = ed.running
        self.progress.setVisible(running)
        self.progress_label.setVisible(running)
        self.operation_label.setVisible(running)
        if running:
            job = next((j for j in self.ctx.ws.jobs.active_jobs() if j.type == "ai_edit"), None)
            self.progress.setValue(int(job.progress) if job else 0)
            self.progress_label.setText(f"Scene {pr['scene']} / {pr['total']}")
            self.operation_label.setText(pr.get("operation", ""))
        project = self.ctx.ws.project
        s = project.editing_sessions[-1] if project and project.editing_sessions else None
        self.log.setPlainText("\n".join(s.log[-40:]) if s else "")
        if project is None:
            msg = ""
        elif s is None:
            msg = "No AI edit yet. Choose a style and press Generate AI Edit."
        elif running:
            msg = "Creating the edit in the background…"
        elif s.status == "COMPLETED":
            msg = f"Edit complete (version {project.timeline_version}). Scrub the preview, then inspect any scene."
        elif s.status == "FAILED":
            msg = f"⚠ {s.error or 'The edit failed.'}" + (f"\nFailed at scene {self._label(s.failed_scene)}; earlier scenes were kept. Use Retry." if s.failed_scene else "") + (
                "\n" + "\n".join(s.validation_errors[:4]) if s.validation_errors else "")
        elif s.status == "CANCELED":
            msg = "Canceled — nothing was changed."
        else:
            msg = s.status
        self.state_label.setText(msg)

    def _label(self, scene_id: str) -> str:
        project = self.ctx.ws.project
        return next((s.label for s in project.scenes if s.id == scene_id), scene_id) if project else scene_id

    # ------------------------------------------------------------------ table
    def refresh(self) -> None:
        project = self.ctx.ws.project
        ed = self.ctx.ws.editing
        has = bool(project and project.scenes)
        running = ed.running
        self.generate_btn.setEnabled(has and not running)
        self.generate_sel_btn.setEnabled(has and not running)
        self.regen_all_btn.setEnabled(has and not running)
        self.cancel_btn.setEnabled(running)
        failed = bool(project and any(g.status in (SceneEditStatus.FAILED, SceneEditStatus.PENDING) and g.scene_id in {s.id for s in project.scenes}
                                      for g in project.timeline_generation.scenes.values()) and project.timeline_generation.status in ("PARTIAL", "FAILED"))
        self.retry_btn.setEnabled(failed and not running)
        self._load_settings()
        self._refresh_progress()
        rows = ed.scene_rows() if has else []
        keep = set(self._selected_ids())
        self._loading = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                conf = row["min_confidence"]
                cells = (row["label"], f"{row['start']:.1f}–{row['end']:.1f}s", VISUAL_TEXT.get(row["visual_status"], row["visual_status"]), EDIT_TEXT[row["status"]],
                         str(row["decisions"]), f"{conf:.0f}% {confidence_label(conf)}" if conf is not None else "—")
                color = {SceneEditStatus.FAILED: "#e5584f", SceneEditStatus.NEEDS_VISUAL: "#e0a030", SceneEditStatus.OUTDATED: "#e0a030"}.get(row["status"])
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, row["scene_id"])
                    if col == 2 and row["visual_status"] != "APPROVED":
                        item.setForeground(QBrush(QColor("#e5584f")))
                    if col == 3 and color:
                        item.setForeground(QBrush(QColor(color)))
                    if col == 5 and conf is not None and conf < LOW:
                        item.setForeground(QBrush(QColor("#e0a030")))
                    if row["error"] and col == 3:
                        item.setToolTip(row["error"])
                    self.table.setItem(r, col, item)
                if row["scene_id"] in keep or (not keep and row["scene_id"] == self.scene_id):
                    self.table.setRangeSelected(__import__("PySide6.QtWidgets", fromlist=["QTableWidgetSelectionRange"]).QTableWidgetSelectionRange(r, 0, r, len(SCENE_COLUMNS) - 1), True)
        finally:
            self._loading = False
        dur = max((s.end for s in project.scenes), default=0.0) if project else 0.0
        self.slider.setRange(0, int(dur * 10))
        self._on_select_scene()

    def _on_select_scene(self) -> None:
        if self._loading:
            return
        ids = self._selected_ids()
        self.scene_id = ids[0] if len(ids) == 1 else (self.scene_id if ids else None)
        self._fill_decisions()
        self._show_missing()
        project = self.ctx.ws.project
        sc = next((s for s in project.scenes if s.id == self.scene_id), None) if project else None
        if sc is not None and len(ids) == 1:
            self.slider.setValue(int(sc.start * 10))

    def _show_missing(self) -> None:
        project = self.ctx.ws.project
        vs = self.ctx.ws.editing.visual_status(self.scene_id)["status"] if project and self.scene_id else "APPROVED"
        bad = vs != "APPROVED"
        self.missing_box.setVisible(bad)
        if bad:
            self.missing_label.setText(f"Scene {self._label(self.scene_id)}: {'Missing Asset' if vs == 'MISSING_MEDIA' else 'No approved visual'} ({VISUAL_TEXT.get(vs, vs)}).\n"
                                       "Nothing is substituted automatically. Choose what to do:")

    # ------------------------------------------------------------------ decisions & inspector
    def _fill_decisions(self) -> None:
        ed = self.ctx.ws.editing
        decs = ed.decisions_for_scene(self.scene_id) if self.scene_id and self.ctx.ws.project else []
        self._loading = True
        try:
            self.decisions.setRowCount(len(decs))
            for r, d in enumerate(decs):
                label = d.type.value.replace("_", " ").title() + (f" — {d.parameters.get('content')}" if d.type in (DecisionType.TEXT, DecisionType.NUMBER_EMPHASIS) else
                                                                  f" — {d.parameters.get('kind') or d.parameters.get('type')}" if d.type in (
                                                                      DecisionType.ZOOM, DecisionType.PAN, DecisionType.TRANSITION) else "")
                cells = (label, f"{d.start:.2f}", f"{d.duration:.2f}", f"{d.confidence:.0f}% {confidence_label(d.confidence)}", d.created_by.value)
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, d.decision_id)
                    if d.confidence < LOW:
                        item.setForeground(QBrush(QColor("#e0a030")))
                    if d.created_by is Creator.USER:
                        f = item.font()
                        f.setBold(True)
                        item.setFont(f)
                    self.decisions.setItem(r, col, item)
                if d.decision_id == self.decision_id:
                    self.decisions.selectRow(r)
        finally:
            self._loading = False
        if not any(d.decision_id == self.decision_id for d in decs):
            self.decision_id = None
        self._show_inspector()

    def _on_select_decision(self) -> None:
        if self._loading:
            return
        rows = self.decisions.selectionModel().selectedRows()
        self.decision_id = self.decisions.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole) if rows else None
        self._show_inspector()

    def select_decision(self, decision_id: str) -> None:
        project = self.ctx.ws.project
        d = project.editing_decisions.get(decision_id) if project else None
        if d is None:
            return
        self.scene_id, self.decision_id = d.scene_id, decision_id
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == d.scene_id:
                self.table.selectRow(r)
                break
        self._fill_decisions()

    def _on_clip_selected(self, clip_id: str | None) -> None:
        project = self.ctx.ws.project
        if not clip_id or project is None:
            return
        d = self.ctx.ws.editing.decision_for_clip(clip_id)
        if d is not None:
            self.select_decision(d.decision_id)

    def _clear_form(self) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._editors.clear()

    def _show_inspector(self) -> None:
        project = self.ctx.ws.project
        self._clear_form()
        d = project.editing_decisions.get(self.decision_id) if project and self.decision_id else None
        sc = next((s for s in project.scenes if s.id == self.scene_id), None) if project and self.scene_id else None
        for w in (self.apply_btn, self.lock_visual, self.lock_text, self.lock_motion, self.lock_scene, self.regen_scene_btn):
            w.setEnabled(sc is not None)
        gen = project.timeline_generation if project else None
        self._loading = True
        try:
            self.lock_scene.setChecked(bool(gen and self.scene_id in gen.locked_scenes))
            scene_decs = [x for x in project.editing_decisions.values() if x.scene_id == self.scene_id] if project else []
            self.lock_visual.setChecked(any(x.type is DecisionType.VISUAL_TIMING and x.locked for x in scene_decs))
            self.lock_text.setChecked(any(x.type in (DecisionType.TEXT, DecisionType.NUMBER_EMPHASIS) and x.locked for x in scene_decs))
            self.lock_motion.setChecked(any(x.type in (DecisionType.ZOOM, DecisionType.PAN) and x.locked for x in scene_decs))
        finally:
            self._loading = False
        self.apply_btn.setEnabled(d is not None and d.type in PARAM_SPECS)
        if d is None:
            self.i_info.setText("Select a scene, then one of its decisions, to inspect and change it." if sc is None else
                                f"Scene {sc.label}: select a decision above.")
            return
        orig = self.ctx.ws.editing.override_of(d.decision_id)
        src = ""
        clip = project.timeline.get_clip(d.target_id) if d.target_id else None
        if clip is not None and clip.asset_id:
            a = project.assets.get(clip.asset_id)
            src = f"{a.name} ({a.source_type.value.replace('_', ' ').title()})" if a else clip.asset_id
        va = project.visual_assignments.get(d.scene_id)
        acc = f"{va.accuracy_score:.0f}" if va and va.accuracy_score is not None else "—"
        self.i_info.setText(
            f"<b>Type:</b> {d.type.value.replace('_', ' ').title()} &nbsp; <b>Scene:</b> {self._label(d.scene_id)}<br>"
            f"<b>Source:</b> {src or '—'} &nbsp; <b>Visual accuracy:</b> {acc}<br>"
            f"<b>Start:</b> {d.start:.2f}s &nbsp; <b>Duration:</b> {d.duration:.2f}s<br>"
            f"<b>Confidence:</b> {d.confidence:.0f}% ({confidence_label(d.confidence)}) &nbsp; <b>Created by:</b> {d.created_by.value}"
            + (f" &nbsp; <i>(overrides {orig.decision_id})</i>" if orig else "") + f"<br><b>Reason:</b> “{d.reason}”")
        for path, label, kind, extra in PARAM_SPECS.get(d.type, []):
            value = _get(d.parameters, path)
            if value is None and path in ("start", "duration"):
                value = d.start if path == "start" else d.duration
            w: QWidget
            if kind == "float":
                w = QDoubleSpinBox()
                w.setRange(*extra)  # type: ignore[misc]
                w.setDecimals(3)
                w.setSingleStep(0.05)
                w.setValue(float(value or 0.0))
            elif kind == "int":
                w = QSpinBox()
                w.setRange(*extra)  # type: ignore[misc]
                w.setValue(int(value or 0))
            elif kind == "enum":
                w = QComboBox()
                w.addItems(ENUMS[str(extra)])
                w.setCurrentText(str(value))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(value))
            elif kind == "list":
                w = QLineEdit(", ".join(value or []))
            else:
                w = QLineEdit(str(value or ""))
            w.setObjectName(f"param_{path}")
            self.param_form.addRow(label, w)
            self._editors[path] = (w, kind)

    def _collect(self, d: EditingDecision) -> dict:
        out: dict = {}
        for path, (w, kind) in self._editors.items():
            if kind in ("float", "int"):
                v = w.value()  # type: ignore[attr-defined]
            elif kind == "enum":
                v = w.currentText()  # type: ignore[attr-defined]
            elif kind == "bool":
                v = w.isChecked()  # type: ignore[attr-defined]
            elif kind == "list":
                v = [x.strip() for x in w.text().split(",") if x.strip()]  # type: ignore[attr-defined]
            else:
                v = w.text()  # type: ignore[attr-defined]
            old = _get(d.parameters, path)
            if path in ("start", "duration") and old is None:
                old = d.start if path == "start" else d.duration
            if old is None or v != old and not (isinstance(old, float) and abs(v - old) < 1e-9):
                top = path.split(".")[0]
                if "." in path:  # nested: send the whole updated container
                    cont = list(d.parameters.get(top) or [0, 0, 0, 0] if top == "region" else d.parameters.get(top) or [0, 0])
                    cont[int(path.split(".")[1])] = v
                    out[top] = cont
                else:
                    out[path] = v
        return out

    def _apply(self) -> None:
        project = self.ctx.ws.project
        d = project.editing_decisions.get(self.decision_id) if project and self.decision_id else None
        if d is None:
            return
        changes = self._collect(d)
        if not changes:
            self.ctx.status("No changes to apply.")
            return
        did = d.decision_id
        result: list = []

        def go() -> None:
            nd = self.ctx.ws.editing.update_decision(did, changes)
            result.append(nd.decision_id)

        if self.act(go, "Edit decision") and result:
            self.decision_id = result[0]
            self._fill_decisions()
            self.slider.setValue(self.slider.value())
            self._scrub(self.slider.value())

    def _lock(self, aspect: str, on: bool) -> None:
        if self._loading or not self.scene_id:
            return
        sid = self.scene_id
        self.act(lambda: self.ctx.ws.editing.set_lock(sid, aspect, on), "Lock")

    def _replace_visual(self) -> None:
        project = self.ctx.ws.project
        d = project.editing_decisions.get(self.decision_id) if project and self.decision_id else None
        clip = project.timeline.get_clip(d.target_id) if d and d.target_id else None
        if clip is None or clip.kind != "media":
            self.ctx.status("Select a visual decision first.")
            return
        from PySide6.QtWidgets import QInputDialog

        assets = [a for a in project.assets.all() if a.type.value != "audio" and a.id != clip.asset_id]
        if not assets:
            self.ctx.status("There is no other visual asset in the project to use.")
            return
        names = [f"{a.name}  ({a.id})" for a in assets]
        choice, ok = QInputDialog.getItem(self, "Replace visual", "Use this project asset instead:", names, 0, False)
        if ok:
            self.act(lambda: self.ctx.ws.editing.replace_clip_asset(clip.id, assets[names.index(choice)].id), "Replace visual")

    def _manual_add(self) -> None:
        sid = self.scene_id
        project = self.ctx.ws.project
        if not sid or project is None:
            return
        from PySide6.QtWidgets import QInputDialog

        assets = [a for a in project.assets.all() if a.type.value != "audio"]
        if not assets:
            path, _ = QFileDialog.getOpenFileName(self, "Choose a visual", "", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.mkv *.webm)")
            if path:
                from pathlib import Path

                self.act(lambda: self.ctx.ws.research.add_manual_visual(sid, Path(path)), "Manual Add")
            return
        names = [f"{a.name}  ({a.id})" for a in assets]
        choice, ok = QInputDialog.getItem(self, "Replace manually", "Use this project asset for the scene:", names, 0, False)
        if ok:
            self.act(lambda: self.ctx.ws.editing.assign_scene_visual(sid, assets[names.index(choice)].id), "Manual Add")

    # ------------------------------------------------------------------ preview / scrubbing
    def _scrub(self, value: int) -> None:
        t = value / 10.0
        self.time_label.setText(f"{int(t // 60)}:{t % 60:04.1f}")
        fs = self.preview.show_time(t)
        if fs is not None:
            bits = [f"{len([l for l in fs.layers if l.kind == 'media'])} visual layer(s)", f"{len([l for l in fs.layers if l.kind == 'text'])} text",
                    f"music {fs.music_level:.0%}", "voice" if fs.voice_active else "no voice"]
            if fs.transition:
                bits.append(f"transition: {fs.transition.lower()}")
            self.preview_note.setText("Preview of timeline data (frames are stills at the playhead; no render needed): " + " · ".join(bits))
        if self.player.position is not None and abs(self.player.position - t) > 0.25 and not self._playing():
            self.player.seek(t)

    def _playing(self) -> bool:
        return self.play_btn.text() == "Pause"

    def _toggle_play(self) -> None:
        if self._playing():
            self.player.pause()
        else:
            self.player.seek(self.slider.value() / 10.0)
            self.player.play()

    def _on_position(self, t: float) -> None:
        if self._playing() and self.isVisible():
            self.slider.blockSignals(True)
            self.slider.setValue(int(t * 10))
            self.slider.blockSignals(False)
            self._scrub(int(t * 10))

    def pause(self) -> None:
        if self._playing():
            self.player.pause()
