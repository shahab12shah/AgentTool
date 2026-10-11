"""Media library: thumbnails, search, sort, import, remove, drag to timeline."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, QTimer, Signal
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
from app.media.asset import Asset
from app.media.importer import LINK_COPY, LINK_REFERENCE
from app.media.metadata import file_dialog_filter
from app.performance.memory_monitor import BoundedLRU
from app.performance.profiler import profiler
from app.ui.context import UiContext
from app.ui.theme import palette

ASSET_MIME = "application/x-agenttool-asset"
ICON_SIZE = QSize(128, 72)
SORTS = ("Date added", "Name", "Type", "Duration")
GRID = QSize(140, 118)
ICON_BYTES = ICON_SIZE.width() * ICON_SIZE.height() * 4
MIN_ICONS, DEFAULT_ICONS, MAX_ICONS = 200, 600, 2000
SCROLL_DEBOUNCE_MS = 40
REFRESH_BURST_MS = 120
MISSING_TTL = 5.0
STATE_LABEL = {"pending": None, "failed": "CANNOT READ", "missing_source": "MISSING"}


class _AssetList(QListWidget):
    """List that starts drags carrying the asset id (for dropping on the timeline)."""

    files_dropped = Signal(list)
    viewport_changed = Signal()  # scrolled or resized: the visible rows may have changed

    def __init__(self) -> None:
        super().__init__()
        self.verticalScrollBar().valueChanged.connect(lambda _v: self.viewport_changed.emit())
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(ICON_SIZE)
        self.setGridSize(GRID)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setWordWrap(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setUniformItemSizes(True)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.viewport_changed.emit()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.viewport_changed.emit()

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
        self._icons: BoundedLRU[str, tuple[tuple, QIcon]] = BoundedLRU(DEFAULT_ICONS, 0, on_evict=self._icon_evicted)  # real thumbnails of rows that were visible recently
        self._items: dict[str, QListWidgetItem] = {}
        self._visible: set[str] = set()
        self._reported: tuple[str, ...] = ()
        self._missing: dict[str, tuple[bool, float]] = {}
        self._placeholders: dict[tuple, QIcon] = {}
        self._last_refresh = 0.0
        self._last_cost = 0.0  # how long the last rebuild took: bursts are spaced at several times that
        self.refresh_count = 0

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

        self._fill_timer = QTimer(self)
        self._fill_timer.setSingleShot(True)
        self._fill_timer.setInterval(SCROLL_DEBOUNCE_MS)
        self._fill_timer.timeout.connect(self._fill_visible)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(REFRESH_BURST_MS)
        self._refresh_timer.timeout.connect(self.refresh)
        self.list.viewport_changed.connect(self._fill_timer.start)
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
        b.on("project.changed", lambda p: self._refresh_soon() if p.get("scope") == "assets" else None)
        b.on("media.thumbnail_ready", self._thumb_ready)
        b.on("media.thumbnail_failed", self._thumb_failed)
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
        self._missing.clear()
        self._reported = ()
        self.search.clear()
        self.refresh()

    def _refresh_soon(self) -> None:
        """A burst of asset changes (importing 100 files) refreshes once at the start and once at the end, not 100 times."""
        if time.monotonic() - self._last_refresh >= max(REFRESH_BURST_MS / 1000.0, 3 * self._last_cost) and not self._refresh_timer.isActive():
            self.refresh()
        else:
            self._refresh_timer.start()

    def refresh(self) -> None:
        """Rebuild the list from the registry: cheap items only (text, id, tooltip). Icons and the missing-file check are done lazily for the rows that are on screen."""
        with profiler.timer("ui.library.refresh"):
            self._refresh_timer.stop()
            self.refresh_count += 1
            began = time.monotonic()
            selected = {i.data(Qt.ItemDataRole.UserRole) for i in self.list.selectedItems()}
            self._configure_icon_budget()
            self.list.setUpdatesEnabled(False)
            try:
                self.list.clear()
                self._items = {}
                assets = self._assets()
                project = self.ctx.ws.project
                keep = {a.id for a in assets}
                for aid in [k for k in self._missing if k not in keep]:
                    del self._missing[aid]
                for asset in assets:
                    item = QListWidgetItem(self._label(asset))
                    item.setIcon(self._placeholder(asset.type.value, "pending"))
                    item.setData(Qt.ItemDataRole.UserRole, asset.id)
                    item.setToolTip(self._tooltip(asset))
                    item.setSizeHint(GRID)
                    self.list.addItem(item)
                    self._items[asset.id] = item
                    if asset.id in selected:
                        item.setSelected(True)
            finally:
                self.list.setUpdatesEnabled(True)
            self.empty.setVisible(project is not None and not len(project.assets))
            self.list.setVisible(not self.empty.isVisible())
            self.setEnabled(project is not None)
            self._update_buttons()
            self._visible = set()
            self._fill_visible()
            self._last_cost = time.monotonic() - began
            self._last_refresh = time.monotonic()  # measured from the END of a refresh: a slow rebuild must not make the next event of a burst look "late"

    def _label(self, asset: Asset) -> str:
        sub = asset.type.value.capitalize()
        if asset.duration:
            sub += f" • {format_duration_short(asset.duration)}"
        miss = self._missing.get(asset.id)
        return f"{asset.name}\n{sub}{' • MISSING' if miss and miss[0] else ''}"

    @staticmethod
    def _tooltip(asset: Asset) -> str:
        parts = [asset.name, f"ID: {asset.id}"]
        if asset.width:
            parts.append(f"{asset.width}×{asset.height}" + (f" @ {asset.fps:.2f} fps" if asset.fps else ""))
        if asset.codec:
            parts.append(f"Codec: {asset.codec}")
        parts.append("Linked (not copied)" if asset.link_mode == "reference" else "Stored in project")
        return "\n".join(parts)

    # ----- lazy icons for the rows that are on screen -----
    def _configure_icon_budget(self) -> None:
        lim = None
        try:
            lim = self.ctx.ws.performance.limits()
        except Exception:  # noqa: BLE001 - the performance service is optional plumbing
            pass
        n = DEFAULT_ICONS if lim is None else int(lim.memory_cache_bytes * 0.02 // ICON_BYTES)
        n = max(MIN_ICONS, min(MAX_ICONS, n))
        if n != self._icons.max_items:
            self._icons.resize(max_items=n)

    def _visible_ids(self) -> list[str]:
        """Ids of the rows in (and one row around) the viewport: a handful of ``itemAt`` probes on the grid, independent of the library size."""
        if self.list.count() == 0:
            return []
        vp = self.list.viewport().rect()
        gw, gh = GRID.width(), GRID.height()
        out: list[str] = []
        seen: set[str] = set()
        for y in range(-gh + 4, vp.height() + gh, gh):  # one row above and below the viewport so a scroll finds its icons ready
            for x in range(gw // 2, max(vp.width(), gw), gw):
                it = self.list.itemAt(QPoint(x, y))
                if it is not None:
                    aid = it.data(Qt.ItemDataRole.UserRole)
                    if aid not in seen:
                        seen.add(aid)
                        out.append(aid)
        return out

    def _fill_visible(self) -> None:
        project = self.ctx.ws.project
        if project is None:
            return
        with profiler.timer("ui.library.fill_visible"):
            ids = self._visible_ids()
            self._visible = set(ids)
            for aid in ids:
                asset = project.assets.get(aid)
                if asset is not None:
                    self._update_row(asset)
            key = tuple(ids)
            if key != self._reported:  # tell the thumbnail queue what is on screen (those jump the queue; the ones scrolled away are dropped)
                self._reported = key
                try:
                    self.ctx.ws.media.set_visible_assets(ids)
                except Exception:  # noqa: BLE001
                    pass

    def _is_missing(self, asset: Asset) -> bool:
        project = self.ctx.ws.project
        hit = self._missing.get(asset.id)
        now = time.monotonic()
        if hit is not None and now - hit[1] < MISSING_TTL:
            return hit[0]
        missing = project is not None and not project.asset_path(asset).is_file()
        self._missing[asset.id] = (missing, now)
        return missing

    def _update_row(self, asset: Asset) -> None:
        item = self._items.get(asset.id)
        if item is None:
            return
        media = self.ctx.ws.media
        missing = self._is_missing(asset)
        state = "missing_source" if missing else media.thumbnail_state(asset.id)
        thumb = media.thumbnail_file(asset) if state == "ready" and not missing else None
        try:
            stamp = (thumb.stat().st_mtime_ns if thumb else 0, state, missing, self.ctx.ws.settings.theme)
        except OSError:
            thumb, stamp = None, (0, state, missing, self.ctx.ws.settings.theme)
        cached = self._icons.get(asset.id)
        if cached is not None and cached[0] == stamp:
            icon = cached[1]
            profiler.cache_hit("library.icons")
        elif thumb is not None:
            profiler.cache_miss("library.icons")
            icon = self._thumb_icon(thumb)
            self._icons.put(asset.id, (stamp, icon))
        else:
            icon = self._placeholder(asset.type.value, state)
            if cached is not None:
                self._icons.pop(asset.id)
        item.setIcon(icon)
        label = self._label(asset)
        if item.text() != label:
            item.setText(label)
        tip = self._tooltip(asset)
        if state == "failed":
            tip += f"\nThumbnail could not be created: {media.thumbnail_failure(asset.id) or 'unknown error'}\nRight-click to retry."
        item.setToolTip(tip)

    def _thumb_icon(self, thumb: Path) -> QIcon:
        pm = QPixmap(ICON_SIZE)
        c = palette(self.ctx.ws.settings.theme)
        pm.fill(QColor(c["panel2"]))
        painter = QPainter(pm)
        src = QPixmap(str(thumb))
        if not src.isNull():
            src = src.scaled(ICON_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap((ICON_SIZE.width() - src.width()) // 2, (ICON_SIZE.height() - src.height()) // 2, src)
        painter.end()
        return QIcon(pm)

    def _placeholder(self, kind: str, state: str) -> QIcon:
        """A shared icon with the type (or the problem) as text: pending, cannot read, missing."""
        theme = self.ctx.ws.settings.theme
        key = (kind, state, theme)
        icon = self._placeholders.get(key)
        if icon is None:
            pm = QPixmap(ICON_SIZE)
            c = palette(theme)
            pm.fill(QColor(c["panel2"]))
            painter = QPainter(pm)
            painter.setPen(QColor(c["danger"] if state in ("failed", "missing_source") else c["muted"]))
            painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, STATE_LABEL.get(state) or kind.upper())
            painter.end()
            icon = self._placeholders[key] = QIcon(pm)
        return icon

    def _icon_evicted(self, aid: str, _value) -> None:
        item = self._items.get(aid)
        if item is not None and aid not in self._visible:
            asset = self.ctx.ws.project.assets.get(aid) if self.ctx.ws.project else None
            if asset is not None:
                item.setIcon(self._placeholder(asset.type.value, "pending"))  # release the pixmap of a row nobody is looking at

    def _thumb_ready(self, payload: dict) -> None:
        """One thumbnail arrived: update that one row (if it is on screen); the next scroll picks it up otherwise. No list rebuild."""
        aid = payload.get("asset_id")
        project = self.ctx.ws.project
        if aid is None or project is None or aid not in self._items:
            return
        self._icons.pop(aid)
        if aid in self._visible:
            asset = project.assets.get(aid)
            if asset is not None:
                self._update_row(asset)

    def _thumb_failed(self, payload: dict) -> None:
        aid = payload.get("asset_id")
        project = self.ctx.ws.project
        if aid in self._visible and project is not None:
            asset = project.assets.get(aid)
            if asset is not None:
                self._missing.pop(aid, None)
                self._update_row(asset)

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

    def retry_thumbnail(self, asset_id: str) -> None:
        self._missing.pop(asset_id, None)
        self.ctx.ws.media.retry_thumbnail(asset_id)
        item = self._items.get(asset_id)
        project = self.ctx.ws.project
        asset = project.assets.get(asset_id) if project else None
        if item is not None and asset is not None:
            item.setIcon(self._placeholder(asset.type.value, "pending"))

    def _context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("Preview", lambda: self.asset_activated.emit(asset_id))
        menu.addAction("Add to timeline", lambda: self.add_to_timeline_requested.emit(asset_id))
        if self.ctx.ws.media.thumbnail_state(asset_id) in ("failed", "missing_source"):
            menu.addAction("Retry thumbnail", lambda: self.retry_thumbnail(asset_id))
        menu.addSeparator()
        menu.addAction("Remove from project…", self.remove_selected)
        menu.exec(self.list.mapToGlobal(pos))
