"""Project page: home screen (new / open / recent) and the open-project workspace."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import (
    ASPECT_RATIOS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    FPS_OPTIONS,
    RESOLUTION_PRESETS,
)
from app.ui.context import UiContext


class ProjectView(QStackedWidget):
    open_requested = Signal(str)  # path chosen from the recent list / browse
    close_requested = Signal()

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.addWidget(self._build_home())
        self.addWidget(self._build_workspace())
        b = ctx.bridge
        for topic in ("project.opened", "project.closed", "project.saved", "project.dirty_changed"):
            b.on(topic, lambda p: self.refresh())
        b.on("project.changed", lambda p: self.refresh())  # keeps the media / clip counts current
        self.refresh()

    # ------------------------------------------------------------ home
    def _build_home(self) -> QWidget:
        page = QWidget()
        title = QLabel("AgentTool")
        title.setObjectName("title")
        sub = QLabel("AI Video Director • Visual Researcher • Editor")
        sub.setObjectName("muted")

        self.name = QLineEdit()
        self.name.setPlaceholderText("My Finance Video")
        self.location = QLineEdit(self.ctx.ws.settings.default_project_location)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_location)
        loc = QHBoxLayout()
        loc.addWidget(self.location, 1)
        loc.addWidget(browse)
        self.resolution = QComboBox()
        self.resolution.addItems(list(RESOLUTION_PRESETS))
        self.resolution.setCurrentText(DEFAULT_RESOLUTION)
        self.fps = QComboBox()
        self.fps.addItems([f"{f} FPS" for f in FPS_OPTIONS])
        self.fps.setCurrentText(f"{DEFAULT_FPS} FPS")
        self.aspect = QComboBox()
        self.aspect.addItems(list(ASPECT_RATIOS))
        self.aspect.setCurrentText(DEFAULT_ASPECT_RATIO)
        self.create_btn = QPushButton("Create Project")
        self.create_btn.setObjectName("primary")
        self.create_btn.clicked.connect(self._create)
        self.name.returnPressed.connect(self._create)

        new_box = QGroupBox("New Project")
        form = QFormLayout(new_box)
        form.addRow("Project name", self.name)
        form.addRow("Location", loc)
        form.addRow("Resolution", self.resolution)
        form.addRow("FPS", self.fps)
        form.addRow("Aspect ratio", self.aspect)
        form.addRow("", self.create_btn)

        self.open_btn = QPushButton("Open Project…")
        self.open_btn.clicked.connect(self._browse_open)
        self.recent = QListWidget()
        self.recent.setObjectName("recentList")
        self.recent.itemDoubleClicked.connect(lambda it: self.open_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
        open_box = QGroupBox("Open")
        ol = QVBoxLayout(open_box)
        ol.addWidget(self.open_btn)
        ol.addWidget(QLabel("Recent projects"))
        ol.addWidget(self.recent, 1)

        row = QHBoxLayout()
        row.addWidget(new_box, 1)
        row.addWidget(open_box, 1)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(16)
        layout.addLayout(row, 1)
        return page

    def _browse_location(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Project location", self.location.text())
        if d:
            self.location.setText(d)

    def _browse_open(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open project folder", self.ctx.ws.settings.default_project_location)
        if d:
            self.open_requested.emit(d)

    def _create(self) -> None:
        from app.ui.dialogs.message import show_error  # noqa: F401 (guard shows errors)

        fps = int(self.fps.currentText().split()[0])
        self.ctx.guard(
            self,
            lambda: self.ctx.ws.new_project(
                self.name.text(), Path(self.location.text().strip() or self.ctx.ws.settings.default_project_location),
                self.resolution.currentText(), fps, self.aspect.currentText(),
            ),
            modal=True,
            title="Create project",
        )
        if self.ctx.ws.project is not None:
            self.name.clear()

    # ------------------------------------------------------------ open project workspace
    def _build_workspace(self) -> QWidget:
        page = QWidget()
        self.project_title = QLabel()
        self.project_title.setObjectName("title")
        self.project_info = QLabel()
        self.project_info.setObjectName("muted")
        self.project_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.close_btn = QPushButton("Close Project")
        self.close_btn.clicked.connect(self.close_requested)
        head = QHBoxLayout()
        col = QVBoxLayout()
        col.addWidget(self.project_title)
        col.addWidget(self.project_info)
        head.addLayout(col, 1)
        head.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignTop)
        self.library_slot = QWidget()
        self.library_slot.setLayout(QVBoxLayout())
        self.library_slot.layout().setContentsMargins(0, 0, 0, 0)
        self.preview_slot = QWidget()
        self.preview_slot.setLayout(QVBoxLayout())
        self.preview_slot.layout().setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.library_slot)
        split.addWidget(self.preview_slot)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.addLayout(head)
        layout.addWidget(split, 1)
        return page

    # ------------------------------------------------------------ state
    def refresh(self) -> None:
        project = self.ctx.ws.project
        if project is None:
            self.setCurrentIndex(0)
            self.location.setText(self.ctx.ws.settings.default_project_location) if not self.location.hasFocus() else None
            self.recent.clear()
            for r in self.ctx.ws.projects.recent_projects():
                item = QListWidgetItem(f"{r['name']}\n{r['path']}")
                item.setData(Qt.ItemDataRole.UserRole, r["path"])
                self.recent.addItem(item)
            return
        self.setCurrentIndex(1)
        s = project.settings
        self.project_title.setText(project.project_name + (" •" if project.dirty else ""))
        self.project_info.setText(
            f"{project.root}\n{s.width}×{s.height} • {s.fps} fps • {s.aspect_ratio} • "
            f"{len(project.assets)} media • {len(project.timeline.all_clips())} clips"
        )
