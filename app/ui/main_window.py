"""Main window: toolbar, navigation, pages and the background-job strip."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import APP_NAME
from app.ui.context import UiContext
from app.ui.dialogs.message import ask_save_changes, show_error
from app.ui.dialogs.recovery_dialog import RecoveryDialog
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.export_panel import ExportPanel
from app.ui.inspector_panel import InspectorPanel
from app.ui.jobs_panel import JobsPanel, JobStatusBar
from app.ui.media_library import MediaLibrary
from app.ui.preview_panel import PreviewPanel
from app.ui.project_view import ProjectView
from app.ui.script_panel import ScriptPanel
from app.ui.theme import stylesheet
from app.ui.timeline_panel import TimelinePanel
from app.ui.ai_edit_panel import AIEditPanel
from app.ui.presentation_panel import PresentationPanel
from app.ui.review_panel import ReviewPanel
from app.ui.scene_panel import ScenePanel
from app.ui.visuals_panel import VisualsPanel
from app.ui.voice_panel import VoicePanel

NAV = ("Project", "Script", "Voice", "Scenes", "Visuals", "Review", "AI Edit", "Audio & Captions", "Edit", "Timeline", "Export")
PLACEHOLDERS = {  # name -> (badge, description)
    "Edit": ("Coming in a later phase", "Caption rendering, audio mixing, music/SFX libraries and final export are planned for later phases."),
}


def _slot() -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    return w


class PlaceholderPage(QWidget):
    def __init__(self, name: str, badge: str, description: str) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel(name)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        soon = QLabel(badge)
        soon.setObjectName("placeholder")
        soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc = QLabel(description)
        desc.setObjectName("muted")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        for w in (title, soon, desc):
            lay.addWidget(w)


class MainWindow(QMainWindow):
    def __init__(self, ctx: UiContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.ws = ctx.ws
        self.setWindowTitle(APP_NAME)
        self.resize(1480, 900)
        self.setAcceptDrops(True)

        # ---- shared widgets (re-parented between pages) ----
        self.library = MediaLibrary(ctx)
        self.preview = PreviewPanel(ctx)
        self.inspector = InspectorPanel(ctx)
        self.timeline_panel = TimelinePanel(ctx)
        self.timeline_panel.selected_provider = self.library.selected_ids
        self.script_panel = ScriptPanel(ctx)
        self.voice_panel = VoicePanel(ctx)
        self.scene_panel = ScenePanel(ctx, self.voice_panel.player)  # the voice-over player is shared
        self.visuals_panel = VisualsPanel(ctx)
        self.review_panel = ReviewPanel(ctx)
        self.ai_edit_panel = AIEditPanel(ctx, self.voice_panel.player)
        self.ai_edit_panel.go_to_research = self._research_scene
        self.presentation_panel = PresentationPanel(ctx, self.voice_panel.player)
        self.export_panel = ExportPanel(ctx)
        self.project_view = ProjectView(ctx)
        self.jobs_panel = JobsPanel(ctx)
        self.job_bar = JobStatusBar(ctx, self.jobs_panel)

        self._build_actions()
        self._build_toolbar()
        self._build_pages()
        self._build_layout()
        self._wire()

        self._errors: list[str] = []
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.ws.autosave_tick)
        self._apply_settings()
        self._sync_state()
        self._place_shared("Project")

    # ------------------------------------------------------------ construction
    def _build_actions(self) -> None:
        def act(text: str, slot, shortcut=None, tip: str = "") -> QAction:
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut is not None:
                a.setShortcut(shortcut)
            if tip:
                a.setToolTip(tip)
            return a

        self.new_action = act("New Project…", self.new_project, QKeySequence.StandardKey.New)
        self.open_action = act("Open Project…", self.open_project_dialog, QKeySequence.StandardKey.Open)
        self.save_action = act("Save", self.save, QKeySequence.StandardKey.Save, "Save project (Ctrl+S)")
        self.save_as_action = act("Save As…", self.save_as, QKeySequence("Ctrl+Shift+S"))
        self.close_action = act("Close Project", self.close_project, QKeySequence("Ctrl+W"))
        self.import_action = act("Import Media…", lambda: self.library.import_dialog(), QKeySequence("Ctrl+I"))
        self.link_action = act("Link External Media (don't copy)…", lambda: self.library.import_dialog(link=True))
        self.quit_action = act("Quit", self.close, QKeySequence.StandardKey.Quit)
        self.undo_action = act("Undo", self.undo, QKeySequence.StandardKey.Undo, "Undo (Ctrl+Z)")
        self.redo_action = act("Redo", self.redo, None, "Redo (Ctrl+Shift+Z)")
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.settings_action = act("Settings", self.open_settings, QKeySequence("Ctrl+,"))

        file_menu = self.menuBar().addMenu("&File")
        for a in (self.new_action, self.open_action, self.save_action, self.save_as_action, self.close_action):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(self.import_action)
        file_menu.addAction(self.link_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.settings_action)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setObjectName("mainToolbar")
        self.addToolBar(bar)
        logo = QLabel("◆ AgentTool")
        logo.setObjectName("logo")
        self.name_label = QLabel("No project")
        self.name_label.setObjectName("projectname")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(logo)
        bar.addWidget(self.name_label)
        bar.addWidget(spacer)
        for a in (self.save_action, self.undo_action, self.redo_action, self.settings_action):
            bar.addAction(a)

    def _build_pages(self) -> None:
        self.pages = QStackedWidget()
        self.page_index: dict[str, int] = {}

        # Timeline page
        self.tl_library_slot, self.tl_preview_slot = _slot(), _slot()
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.tl_library_slot)
        top.addWidget(self.tl_preview_slot)
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inspector_scroll.setWidget(self.inspector)
        inspector_scroll.setMinimumWidth(270)
        top.addWidget(inspector_scroll)
        top.setSizes([330, 560, 300])
        top.setChildrenCollapsible(False)
        self.tl_library_slot.setMinimumHeight(140)
        self.tl_preview_slot.setMinimumHeight(140)
        self.timeline_panel.setMinimumHeight(330)
        timeline_page = QSplitter(Qt.Orientation.Vertical)
        timeline_page.addWidget(top)
        timeline_page.addWidget(self.timeline_panel)
        timeline_page.setStretchFactor(0, 2)
        timeline_page.setStretchFactor(1, 3)
        timeline_page.setChildrenCollapsible(False)

        widgets: dict[str, QWidget] = {
            "Project": self.project_view,
            "Script": self.script_panel,
            "Voice": self.voice_panel,
            "Scenes": self.scene_panel,
            "Visuals": self.visuals_panel,
            "Review": self.review_panel,
            "AI Edit": self.ai_edit_panel,
            "Audio & Captions": self.presentation_panel,
            "Timeline": timeline_page,
            "Export": self.export_panel,
        }
        for name in NAV:
            w = widgets.get(name) or PlaceholderPage(name, *PLACEHOLDERS[name])
            self.page_index[name] = self.pages.addWidget(w)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(150)
        self.nav.addItems(NAV)
        self.nav.setCurrentRow(0)

    def _build_layout(self) -> None:
        center = QWidget()
        row = QHBoxLayout(center)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.nav)
        row.addWidget(self.pages, 1)
        root = QWidget()
        col = QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(center, 1)
        self.jobs_panel.setMaximumHeight(190)
        col.addWidget(self.jobs_panel)
        col.addWidget(self.job_bar)
        self.setCentralWidget(root)

    def _wire(self) -> None:
        b = self.ctx.bridge
        self.nav.currentRowChanged.connect(self._on_nav)
        self.library.asset_activated.connect(self.preview_asset)
        self.library.add_to_timeline_requested.connect(self._add_to_timeline)
        self.timeline_panel.preview_requested.connect(self.preview_asset)
        self.script_panel.analyze_requested.connect(lambda: self.go_to("Scenes"))
        self.project_view.open_requested.connect(self.open_project)
        self.export_panel.return_to_editor.connect(lambda: self.go_to("Timeline"))
        self.project_view.close_requested.connect(self.close_project)
        for topic in ("project.opened", "project.closed", "project.dirty_changed", "project.saved"):
            b.on(topic, lambda p: self._sync_state())
        b.on("project.opened", lambda p: self.nav.setCurrentRow(0))
        b.on("commands.changed", self._on_commands_changed)
        b.on("app.error", self._queue_error)

    # ------------------------------------------------------------ state
    def _sync_state(self) -> None:
        project = self.ws.project
        has = project is not None
        name = project.project_name if project else "No project"
        dirty = bool(project and project.dirty)
        self.name_label.setText(name + (" •" if dirty else ""))
        self.setWindowTitle(f"{name}{'*' if dirty else ''} — {APP_NAME}" if has else APP_NAME)
        for a in (self.save_action, self.save_as_action, self.close_action, self.import_action, self.link_action):
            a.setEnabled(has)
        self.save_action.setEnabled(has)
        for i, label in enumerate(NAV):
            item = self.nav.item(i)
            enabled = has or label == "Project"
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled if enabled else item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        if not has:
            self.undo_action.setEnabled(False)
            self.redo_action.setEnabled(False)

    def _on_commands_changed(self, p: dict) -> None:
        self.undo_action.setEnabled(p["can_undo"] and self.ws.project is not None)
        self.redo_action.setEnabled(p["can_redo"] and self.ws.project is not None)
        self.undo_action.setText(f"Undo {p['undo_text']}" if p["undo_text"] else "Undo")
        self.redo_action.setText(f"Redo {p['redo_text']}" if p["redo_text"] else "Redo")

    def _apply_settings(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet(self.ws.settings.theme))
        self.autosave_timer.start(self.ws.settings.autosave_interval_seconds * 1000)

    # ------------------------------------------------------------ navigation
    def _research_scene(self, scene_id: str) -> None:
        self.go_to("Review")
        self.review_panel.scene_id = scene_id
        for r in range(self.review_panel.table.rowCount()):
            if self.review_panel.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == scene_id:
                self.review_panel.table.selectRow(r)
                break

    def go_to(self, name: str) -> None:
        self.nav.setCurrentRow(self.page_index[name])

    def _on_nav(self, row: int) -> None:
        if row < 0:
            return
        name = NAV[row]
        self.script_panel.flush()
        self.preview.pause()
        self.voice_panel.pause()
        self.scene_panel.pause()
        self.ai_edit_panel.pause()
        self.presentation_panel.pause()
        self.pages.setCurrentIndex(row)
        self._place_shared(name)
        if name == "Review":
            self.review_panel.refresh()
        if name == "AI Edit":
            self.ai_edit_panel.refresh()
        if name == "Audio & Captions":
            self.presentation_panel.refresh()
        if name == "Export":
            self.export_panel.refresh()

    def _place_shared(self, page: str) -> None:
        """The library and preview exist once; show them on whichever page needs them."""
        if page == "Project":
            slots = (self.project_view.library_slot, self.project_view.preview_slot)
        elif page == "Timeline":
            slots = (self.tl_library_slot, self.tl_preview_slot)
        else:
            return
        for widget, slot in zip((self.library, self.preview), slots):
            old = widget.parentWidget()
            if old is not None and old.layout() is not None:
                old.layout().removeWidget(widget)
            slot.layout().addWidget(widget)
            widget.show()

    def preview_asset(self, asset_id: str) -> None:
        project = self.ws.project
        asset = project.assets.get(asset_id) if project else None
        if asset is None:
            return
        if self.pages.currentIndex() not in (self.page_index["Project"], self.page_index["Timeline"]):
            self.go_to("Project")
        self.preview.show_asset(asset)

    def _add_to_timeline(self, asset_id: str) -> None:
        self.ctx.guard(self, lambda: self.ws.timeline.add_asset(asset_id, None, self.timeline_panel.canvas.playhead), modal=True, title="Add to timeline")

    # ------------------------------------------------------------ project actions
    def _confirm_close_project(self) -> bool:
        project = self.ws.project
        if project is None:
            return True
        self.script_panel.flush()
        if not project.dirty:
            return True
        choice = ask_save_changes(self, project.project_name)
        if choice == QMessageBox.StandardButton.Save:
            return self.save()
        return choice == QMessageBox.StandardButton.Discard

    def new_project(self) -> None:
        if self._confirm_close_project():
            self.ws.close_project()
            self.go_to("Project")
            self.project_view.name.setFocus()

    def open_project_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open project folder", self.ws.settings.default_project_location)
        if d:
            self.open_project(d)

    def open_project(self, path: str | Path) -> None:
        if not self._confirm_close_project():
            return
        ok = self.ctx.guard(self, lambda: self.ws.open_project(Path(path)), modal=True, title="Open project")
        if not ok:
            self._offer_backup(Path(path))

    def _offer_backup(self, path: Path) -> None:
        backup = path / "project.json.bak"
        if backup.is_file() and QMessageBox.question(
            self, "Open backup?", "The project file could not be opened, but a backup copy exists.\nOpen the backup instead?"
        ) == QMessageBox.StandardButton.Yes:
            self.ctx.guard(self, lambda: self.ws.open_project(path, from_backup=True), modal=True, title="Open backup")

    def save(self) -> bool:
        if self.ws.project is None:
            return False
        self.script_panel.flush()
        return self.ctx.guard(self, self.ws.save, modal=True, title="Save project")

    def save_as(self) -> None:
        project = self.ws.project
        if project is None:
            return
        self.script_panel.flush()
        name, ok = QInputDialog.getText(self, "Save As", "New project name:", text=f"{project.project_name} copy")
        if not ok or not name.strip():
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose where to save the copy", str(project.root.parent) if project.root else "")
        if parent:
            self.ctx.guard(self, lambda: self.ws.save_as(Path(parent), name), modal=True, title="Save As")

    def close_project(self) -> None:
        if self._confirm_close_project():
            self.ws.close_project()
            self.go_to("Project")

    def undo(self) -> None:
        self.script_panel.flush()
        self.ctx.guard(self, self.ws.undo)

    def redo(self) -> None:
        self.script_panel.flush()
        self.ctx.guard(self, self.ws.redo)

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.ws.settings, self.ws.describe_ffmpeg, self.ws.transcripts.provider_report, self.ws.research.provider_report, self)
        if dlg.exec():
            self.ctx.guard(self, lambda: self.ws.update_settings(dlg.result_settings()), modal=True, title="Settings")
            self._apply_settings()

    # ------------------------------------------------------------ recovery
    def check_recovery(self) -> None:
        for entry in self.ws.pending_recovery():
            dlg = RecoveryDialog(entry, self)
            result = dlg.exec()
            if result == RecoveryDialog.RECOVER:
                if self.ctx.guard(self, lambda e=entry: self.ws.recover(e.project_id), modal=True, title="Recovery"):
                    self.ctx.status("Project recovered — save to keep these changes.")
                    return
                self.ws.discard_recovery(entry.project_id)  # unusable data; do not offer it forever
            elif result == RecoveryDialog.DISCARD:
                self.ws.discard_recovery(entry.project_id)

    # ------------------------------------------------------------ errors
    def _queue_error(self, p: dict) -> None:
        self._errors.append(p.get("message", "An error occurred."))
        if len(self._errors) == 1:
            QTimer.singleShot(200, self._flush_errors)

    def _flush_errors(self) -> None:
        errors, self._errors = self._errors, []
        shown = "\n\n".join(errors[:6]) + (f"\n\n…and {len(errors) - 6} more." if len(errors) > 6 else "")
        if errors:
            show_error(self, shown, title="Problem")

    # ------------------------------------------------------------ drag & drop / close
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if self.ws.project is not None and e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.library.import_paths(paths)
            e.acceptProposedAction()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._confirm_close_project():
            event.ignore()
            return
        self.autosave_timer.stop()
        self.preview.player.unload()
        self.voice_panel.player.unload()
        self.ws.close_project()  # clean close: removes recovery data
        self.ws.shutdown()
        event.accept()
