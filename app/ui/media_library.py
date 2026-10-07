"""Media library: thumbnails, search, sort, import, remove, drag to timeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.timecode import format_duration_short
from app.media.asset import Asset, AssetType
from app.media.importer import LINK_COPY, LINK_REFERENCE
from app.media.metadata import file_dialog_filter
from app.ui.context import UiContext
from app.ui.theme import palette

ASSET_MIME = "application/x-agenttool-asset"
ICON_SIZE = QSize(128, 72)
SORTS = ("Date added", "Name", "Type", "Duration")


class _AssetList(QListWidget):
    """List that starts drags carrying the asset id (for dropping on the timeline)."""

    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(ICON_SIZE)
        self.setGridSize(QSize(140, 118))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setWordWrap(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setUniformItemSizes(True)

    def mimeData(self, items) -> QMimeData:  # noqa: N802
        data = QMimeData()
        if items:
            data.setData(ASSET_MIME, items[0].data(Qt.ItemDataRole.UserRole).encode())
        return data

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            self.files_dropped.emit([Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()])
            event.acceptProposedAction()


class MediaLibrary(QWidget):
    asset_activated = Signal(str)  # double-click -> preview
    add_to_timeline_requested = Signal(str)

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._icons: dict[str, QIcon] = {}

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search media…")
        self.search.setClearButtonEnabled(True)
        self.sort = QComboBox()
        self.sort.addItems(SORTS)
        self.sort.setToolTip("Sort by")
        self.import_btn = QPushButton("Import…")
        self.import_btn.setObjectName("primary")
        self.remove_btn = QPushButton("Remove")
        self.list = _AssetList()
        self.list.setObjectName("assetList")
        self.empty = QLabel("No media yet.\nImport files or drop them here.")
        self.empty.setObjectName("placeholder")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.sort)
        bottom = QHBoxLayout()
        bottom.addWidget(self.import_btn)
        bottom.addWidget(self.remove_btn)
        bottom.addStretch(1)
        header = QLabel("Media Library")
        header.setObjectName("muted")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header)
        layout.addLayout(top)
        layout.addWidget(self.empty)
        layout.addWidget(self.list, 1)
        layout.addLayout(bottom)

        self.search.textChanged.connect(self.refresh)
        self.sort.currentIndexChanged.connect(self.refresh)
        self.import_btn.clicked.connect(self.import_dialog)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.list.itemDoubleClicked.connect(lambda it: self.asset_activated.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.files_dropped.connect(self.import_paths)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)

        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self._reset())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") == "assets" else None)
        b.on("media.thumbnail_ready", self._thumb_ready)
        self._update_buttons()
        self.refresh()

    # ----- data -----
    def _assets(self) -> list[Asset]:
        project = self.ctx.ws.project
        if project is None:
            return []
        assets = project.assets.all()
        needle = self.search.text().strip().lower()
        if needle:
            assets = [a for a in assets if needle in a.name.lower() or needle in a.type.value]
        key = self.sort.currentText()
        if key == "Name":
            assets.sort(key=lambda a: a.name.lower())
        elif key == "Type":
            assets.sort(key=lambda a: (a.type.value, a.name.lower()))
        elif key == "Duration":
            assets.sort(key=lambda a: a.duration or 0.0, reverse=True)
        return assets  # "Date added" keeps registry (import) order

    def _reset(self) -> None:
        self._icons.clear()
        self.search.clear()
        self.refresh()

    def refresh(self) -> None:
        selected = {i.data(Qt.ItemDataRole.UserRole) for i in self.list.selectedItems()}
        self.list.clear()
        assets = self._assets()
        project = self.ctx.ws.project
        for asset in assets:
            sub = asset.type.value.capitalize()
            if asset.duration:
                sub += f" • {format_duration_short(asset.duration)}"
            missing = project is not None and not project.asset_path(asset).is_file()
            item = QListWidgetItem(self._icon_for(asset, missing), f"{asset.name}\n{sub}{' • MISSING' if missing else ''}")
            item.setData(Qt.ItemDataRole.UserRole, asset.id)
            item.setToolTip(self._tooltip(asset))
            item.setSizeHint(QSize(140, 118))
            self.list.addItem(item)
            if asset.id in selected:
                item.setSelected(True)
        self.empty.setVisible(project is not None and not len(project.assets))
        self.list.setVisible(not self.empty.isVisible())
        self.setEnabled(project is not None)
        self._update_buttons()

    @staticmethod
    def _tooltip(asset: Asset) -> str:
        parts = [asset.name, f"ID: {asset.id}"]
        if asset.width:
            parts.append(f"{asset.width}×{asset.height}" + (f" @ {asset.fps:.2f} fps" if asset.fps else ""))
        if asset.codec:
            parts.append(f"Codec: {asset.codec}")
        parts.append("Linked (not copied)" if asset.link_mode == "reference" else "Stored in project")
        return "\n".join(parts)

    def _icon_for(self, asset: Asset, missing: bool) -> QIcon:
        thumb = self.ctx.ws.media.thumbnail_file(asset)
        key = f"{asset.id}:{thumb.stat().st_mtime_ns if thumb else 0}:{missing}"
        if key in self._icons:
            return self._icons[key]
        pm = QPixmap(ICON_SIZE)
        c = palette(self.ctx.ws.settings.theme)
        pm.fill(QColor(c["panel2"]))
        painter = QPainter(pm)
        if thumb and not missing:
            src = QPixmap(str(thumb))
            if not src.isNull():
                src = src.scaled(ICON_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap((ICON_SIZE.width() - src.width()) // 2, (ICON_SIZE.height() - src.height()) // 2, src)
        else:
            painter.setPen(QColor(c["danger"] if missing else c["muted"]))
            painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "MISSING" if missing else asset.type.value.upper())
        painter.end()
        icon = QIcon(pm)
        self._icons[key] = icon
        return icon

    def _thumb_ready(self, payload: dict) -> None:
        self.refresh()

    # ----- actions -----
    def selected_ids(self) -> list[str]:
        return [i.data(Qt.ItemDataRole.UserRole) for i in self.list.selectedItems()]

    def _update_buttons(self) -> None:
        has_project = self.ctx.ws.project is not None
        self.import_btn.setEnabled(has_project)
        self.remove_btn.setEnabled(has_project and bool(self.list.selectedItems()))

    def import_dialog(self, link: bool = False) -> None:
        if self.ctx.ws.project is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Import media", "", file_dialog_filter())
        if files:
            self.import_paths([Path(f) for f in files], link=link is True)

    def import_paths(self, paths: list[Path], link: bool = False) -> None:
        self.ctx.guard(
            self, lambda: self.ctx.ws.media.import_files(paths, LINK_REFERENCE if link else LINK_COPY), modal=True, title="Import"
        )

    def remove_selected(self) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        project = self.ctx.ws.project
        used = sum(len(project.timeline.clips_for_asset(i)) for i in ids) if project else 0
        text = f"Remove {len(ids)} item(s) from the project?\nThe media files on disk are not deleted."
        if used:
            text += f"\n{used} timeline clip(s) using them will be removed too (you can undo this)."
        if QMessageBox.question(self, "Remove media", text) != QMessageBox.StandardButton.Yes:
            return
        for asset_id in ids:
            self.ctx.guard(self, lambda a=asset_id: self.ctx.ws.media.remove_asset(a), modal=True, title="Remove media")

    def _context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("Preview", lambda: self.asset_activated.emit(asset_id))
        menu.addAction("Add to timeline", lambda: self.add_to_timeline_requested.emit(asset_id))
        menu.addSeparator()
        menu.addAction("Remove from project…", self.remove_selected)
        menu.exec(self.list.mapToGlobal(pos))
