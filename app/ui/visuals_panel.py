"""Visual source preferences and rules. Research itself arrives in Phase 3; these settings are saved with the project."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.ui.context import UiContext
from app.ui.theme import palette
from app.visual.preferences import SOURCE_LABELS, SourceKind, VisualPreferences

RULES = (
    ("prefer_real_visuals", "Prefer real visuals"),
    ("prefer_ai_visuals", "Prefer AI visuals"),
    ("prefer_evidence", "Prefer evidence"),
    ("match_narration_literally", "Match narration literally"),
    ("allow_visual_interpretation", "Allow visual interpretation"),
    ("avoid_repeated_visuals", "Avoid repeated visuals"),
)


class VisualsPanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._loading = False
        title = QLabel("Visual Sources")
        title.setObjectName("title")
        note = QLabel("These preferences are saved in the project and guide visual research (Review page). Percentages are soft targets: the best visual for a scene always wins "
                      "(e.g. a screenshot is used when it is the best evidence, even if screenshots are “over budget”).")
        note.setObjectName("muted")
        note.setWordWrap(True)

        self.enabled: dict[SourceKind, QCheckBox] = {}
        self.target: dict[SourceKind, QSpinBox] = {}
        self.priority: dict[SourceKind, QSpinBox] = {}
        grid = QGridLayout()
        grid.addWidget(QLabel("Source"), 0, 0)
        grid.addWidget(QLabel("Target %"), 0, 1)
        grid.addWidget(QLabel("Priority (1–5)"), 0, 2)
        for row, kind in enumerate(SourceKind, start=1):
            cb = QCheckBox(SOURCE_LABELS[kind])
            cb.setObjectName(f"enable_{kind.value}")
            sp = QSpinBox()
            sp.setObjectName(f"target_{kind.value}")
            sp.setRange(0, 100)
            sp.setSuffix(" %")
            sp.setKeyboardTracking(True)
            self.enabled[kind], self.target[kind] = cb, sp
            grid.addWidget(cb, row, 0)
            grid.addWidget(sp, row, 1)
            pr = QSpinBox()
            pr.setObjectName(f"priority_{kind.value}")
            pr.setRange(1, 5)
            pr.setToolTip("A soft nudge: higher-priority sources are preferred when candidates score about the same")
            self.priority[kind] = pr
            grid.addWidget(pr, row, 2)
            pr.valueChanged.connect(self._changed)
            cb.toggled.connect(self._changed)
            sp.valueChanged.connect(self._changed)
        self.total_label = QLabel()
        self.total_label.setObjectName("totalTarget")
        self.warning_label = QLabel()
        self.warning_label.setObjectName("prefWarning")
        self.warning_label.setWordWrap(True)
        src_box = QGroupBox("Sources")
        sl = QVBoxLayout(src_box)
        sl.addLayout(grid)
        sl.addWidget(self.total_label)
        sl.addWidget(self.warning_label)

        self.accuracy = QSpinBox()
        self.accuracy.setObjectName("minAccuracy")
        self.accuracy.setRange(0, 100)
        self.accuracy.valueChanged.connect(self._changed)
        self.rules: dict[str, QCheckBox] = {}
        rules_box = QGroupBox("Rules")
        rl = QVBoxLayout(rules_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Minimum accuracy score"))
        row.addWidget(self.accuracy)
        row.addStretch(1)
        rl.addLayout(row)
        for key, text in RULES:
            cb = QCheckBox(text)
            cb.setObjectName(f"rule_{key}")
            cb.toggled.connect(self._changed)
            self.rules[key] = cb
            rl.addWidget(cb)
        self.reset_btn = QPushButton("Reset to defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(note)
        cols = QHBoxLayout()
        cols.addWidget(src_box, 1)
        cols.addWidget(rules_box, 1)
        layout.addLayout(cols)
        layout.addWidget(self.reset_btn)
        layout.addStretch(1)

        for topic in ("project.opened", "project.closed"):
            ctx.bridge.on(topic, lambda p: self.refresh())
        ctx.bridge.on("project.changed", lambda p: self.refresh() if p.get("scope") == "preferences" else None)
        self.refresh()

    def _collect(self) -> VisualPreferences:
        project = self.ctx.ws.project
        prefs = VisualPreferences.from_dict(project.visual_preferences.to_dict()) if project else VisualPreferences()
        for kind in SourceKind:
            s = prefs.setting(kind)
            s.enabled, s.target_percent = self.enabled[kind].isChecked(), float(self.target[kind].value())
            s.priority = self.priority[kind].value()
        prefs.min_accuracy_score = self.accuracy.value()
        for key, cb in self.rules.items():
            setattr(prefs, key, cb.isChecked())
        return prefs

    def _changed(self, *_args) -> None:
        if self._loading or self.ctx.ws.project is None:
            return
        prefs = self._collect()
        self._show_report(prefs)
        self.ctx.guard(self, lambda: self.ctx.ws.set_visual_preferences(prefs))  # never blocks, even above 100%

    def _reset_defaults(self) -> None:
        if self.ctx.ws.project is not None:
            self.ctx.guard(self, lambda: self.ctx.ws.set_visual_preferences(VisualPreferences()))

    def _show_report(self, prefs: VisualPreferences) -> None:
        r = prefs.report()
        c = palette(self.ctx.ws.settings.theme)
        self.total_label.setText(f"Total target:\n{r.total:g}%")
        color = c["ok"] if r.is_balanced else "#e0a030"
        self.total_label.setStyleSheet(f"font-weight: 600; font-size: 15px; color: {color};")
        self.warning_label.setText("\n".join(("Warning:\n" if i == 0 else "") + w for i, w in enumerate(r.warnings)))
        self.warning_label.setStyleSheet("color: #e0a030;")

    def refresh(self) -> None:
        project = self.ctx.ws.project
        prefs = project.visual_preferences if project else VisualPreferences()
        self._loading = True
        try:
            for kind in SourceKind:
                s = prefs.setting(kind)
                self.enabled[kind].setChecked(s.enabled)
                self.target[kind].setValue(int(round(s.target_percent)))
                self.priority[kind].setValue(int(s.priority))
            self.accuracy.setValue(prefs.min_accuracy_score)
            for key, cb in self.rules.items():
                cb.setChecked(getattr(prefs, key))
        finally:
            self._loading = False
        for w in (*self.enabled.values(), *self.target.values(), *self.priority.values(), *self.rules.values(), self.accuracy, self.reset_btn):
            w.setEnabled(project is not None)
        self._show_report(prefs)
