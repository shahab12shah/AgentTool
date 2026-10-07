"""Script editor. Edits are committed (debounced) as undoable commands."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from app.ui.context import UiContext

COMMIT_DELAY_MS = 600


class _ScriptEdit(QPlainTextEdit):
    """Plain editor whose Ctrl+Z / Ctrl+Shift+Z go to the project-level undo stack."""

    def event(self, e) -> bool:  # noqa: N802
        if e.type() == QEvent.Type.ShortcutOverride and (e.matches(QKeySequence.StandardKey.Undo) or e.matches(QKeySequence.StandardKey.Redo)):
            e.ignore()
            return True
        return super().event(e)


class ScriptPanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.editor = _ScriptEdit()
        self.editor.setObjectName("scriptEditor")
        self.editor.setUndoRedoEnabled(False)  # the project undo stack owns history
        self.editor.setPlaceholderText("Paste or write your script here…")
        font = self.editor.font()
        font.setPointSize(13)
        self.editor.setFont(font)
        self.counts = QLabel("0 words • 0 characters")
        self.counts.setObjectName("muted")
        self.analyze = QPushButton("Analyze Script — Coming in Phase 2")
        self.analyze.setEnabled(False)
        self.analyze.setToolTip("AI script analysis is planned for Phase 2.")

        title = QLabel("Script")
        title.setObjectName("title")
        bar = QHBoxLayout()
        bar.addWidget(self.counts)
        bar.addStretch(1)
        bar.addWidget(self.analyze)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.editor, 1)
        layout.addLayout(bar)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(COMMIT_DELAY_MS)
        self._timer.timeout.connect(self.flush)
        self.editor.textChanged.connect(self._on_text_changed)

        for topic in ("project.opened", "project.closed"):
            ctx.bridge.on(topic, lambda p: self._load())
        ctx.bridge.on("project.changed", lambda p: self._load() if p.get("scope") == "script" else None)
        self._load()

    def _on_text_changed(self) -> None:
        self._update_counts()
        if self.ctx.ws.project is not None and not self._loading:
            self._timer.start()

    _loading = False

    def flush(self) -> None:
        """Commit pending typing to the project now (called before save/undo/close)."""
        self._timer.stop()
        if self.ctx.ws.project is not None:
            self.ctx.guard(self, lambda: self.ctx.ws.set_script(self.editor.toPlainText()))

    def _load(self) -> None:
        project = self.ctx.ws.project
        text = project.script.text if project else ""
        self.editor.setEnabled(project is not None)
        if self.editor.toPlainText() != text:
            self._timer.stop()
            self._loading = True
            try:
                cursor = self.editor.textCursor().position()
                self.editor.setPlainText(text)
                c = self.editor.textCursor()
                c.setPosition(min(cursor, len(text)))
                self.editor.setTextCursor(c)
            finally:
                self._loading = False
        self._update_counts()

    def _update_counts(self) -> None:
        text = self.editor.toPlainText()
        self.counts.setText(f"{len(text.split())} words • {len(text)} characters")
