"""User-facing message dialogs."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


def show_error(parent: QWidget | None, message: str, details: str | None = None, title: str = "Something went wrong") -> None:
    box = QMessageBox(QMessageBox.Icon.Warning, title, message, QMessageBox.StandardButton.Ok, parent)
    if details:
        box.setDetailedText(details)
    box.exec()


def ask_save_changes(parent: QWidget | None, project_name: str) -> QMessageBox.StandardButton:
    box = QMessageBox(
        QMessageBox.Icon.Question,
        "Unsaved changes",
        f"Save changes to “{project_name}” before closing?",
        QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        parent,
    )
    box.setDefaultButton(QMessageBox.StandardButton.Save)
    return QMessageBox.StandardButton(box.exec())
