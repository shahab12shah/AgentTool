"""Relink missing media: locate a file, search a folder, or accept exact matches. A weak match is never used without a confirmation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.exceptions import AppError
from app.media.metadata import file_dialog_filter
from app.rendering.relink import RelinkError
from app.ui.context import UiContext


class RelinkDialog(QDialog):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Relink missing media")
        self.resize(720, 360)
        info = QLabel("These files were moved or deleted. Locate each one, search a folder, or let the app match files with identical content.")
        info.setWordWrap(True)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("relinkTable")
        self.table.setHorizontalHeaderLabels(("Asset", "Type", "Last known location"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.locate_btn, self.search_btn, self.auto_btn, close = QPushButton("Locate file…"), QPushButton("Search folder…"), QPushButton("Match exact copies in folder…"), QPushButton("Close")
        self.locate_btn.clicked.connect(self.locate)
        self.search_btn.clicked.connect(self.search)
        self.auto_btn.clicked.connect(self.auto)
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        for b in (self.locate_btn, self.search_btn, self.auto_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        col = QVBoxLayout(self)
        col.addWidget(info)
        col.addWidget(self.table, 1)
        col.addLayout(row)
        self.refresh()

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        missing = self.ctx.ws.render.relink.missing()
        self.table.setRowCount(len(missing))
        project = self.ctx.ws.require_project()
        for r, a in enumerate(missing):
            for c, text in enumerate((a.name, a.type.value, str(project.asset_path(a)))):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, a.id)
                self.table.setItem(r, c, item)
        if missing:
            self.table.selectRow(0)
        for b in (self.locate_btn, self.search_btn):
            b.setEnabled(bool(missing))

    def _selected(self) -> str | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _relink(self, asset_id: str, path: Path) -> bool:
        svc = self.ctx.ws.render.relink
        try:
            svc.relink(asset_id, path)
        except RelinkError as exc:
            if "does not clearly match" not in exc.user_message:
                QMessageBox.warning(self, "Cannot use that file", exc.user_message)
                return False
            if QMessageBox.question(self, "Use this file?", exc.user_message + "\n\nUse it anyway?") != QMessageBox.StandardButton.Yes:
                return False
            try:
                svc.relink(asset_id, path, confirmed=True)
            except AppError as exc2:
                QMessageBox.warning(self, "Cannot use that file", exc2.user_message)
                return False
        self.refresh()
        return True

    # ------------------------------------------------------------------ actions
    def locate(self) -> None:
        aid = self._selected()
        if aid is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Locate the original file", "", file_dialog_filter())
        if path:
            self._relink(aid, Path(path))

    def search(self) -> None:
        aid = self._selected()
        if aid is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Search this folder (and its sub-folders)")
        if not folder:
            return
        asset = self.ctx.ws.require_project().assets.require(aid)
        cands = self.ctx.ws.render.relink.find_candidates(asset, [Path(folder)])
        if not cands:
            QMessageBox.information(self, "No match", f"No file in that folder looks like “{asset.name}”.")
            return
        labels = [f"{c.score:.0f}%  {c.path}  ({', '.join(c.reasons)})" for c in cands[:20]]
        choice, ok = QInputDialog.getItem(self, "Choose the replacement", f"Matches for “{asset.name}”:", labels, 0, False)
        if ok:
            self._relink(aid, cands[labels.index(choice)].path)

    def auto(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder to search for exact copies")
        if not folder:
            return
        done = self.ctx.ws.render.relink.auto_relink([Path(folder)])
        QMessageBox.information(self, "Relink", f"{len(done)} file(s) were matched by identical content and relinked." if done else "No identical copies were found.")
        self.refresh()
