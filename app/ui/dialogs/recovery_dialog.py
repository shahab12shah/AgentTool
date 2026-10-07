"""Startup prompt offering to recover an interrupted session."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.project.recovery import RecoveryEntry


class RecoveryDialog(QDialog):
    RECOVER = 1
    DISCARD = 2

    def __init__(self, entry: RecoveryEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project recovery")
        self.setModal(True)
        self.setMinimumWidth(380)
        title = QLabel("Project recovery available.")
        title.setObjectName("title")
        detail = QLabel(
            f"Project:\n<b>{entry.project_name}</b>\n\nLast saved:\n<b>{entry.saved_at_local}</b>\n\nRecover?"
        )
        note = QLabel("Recovering loads your unsaved work. Your saved project file is not changed until you save.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        recover = QPushButton("Recover")
        recover.setObjectName("primary")
        recover.setDefault(True)
        discard = QPushButton("Discard")
        recover.clicked.connect(lambda: self.done(self.RECOVER))
        discard.clicked.connect(lambda: self.done(self.DISCARD))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(recover)
        row.addWidget(discard)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addWidget(note)
        layout.addLayout(row)
