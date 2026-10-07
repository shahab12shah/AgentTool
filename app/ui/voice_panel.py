"""Voice-over panel: import, inspect, play, replace, remove."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.core.constants import AUDIO_EXTENSIONS
from app.core.timecode import format_timecode
from app.preview.player import QtPreviewPlayer
from app.ui.context import UiContext
from app.ui.preview_panel import TransportControls


class VoicePanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.player = QtPreviewPlayer(self)
        self.player.failed.connect(ctx.status)

        title = QLabel("Voice-over")
        title.setObjectName("title")
        self.filename = QLabel("—")
        self.duration = QLabel("—")
        self.status = QLabel("")
        self.status.setObjectName("muted")
        info = QGroupBox("Current voice-over")
        form = QFormLayout(info)
        form.addRow("File", self.filename)
        form.addRow("Duration", self.duration)
        form.addRow("", self.status)

        self.import_btn = QPushButton("Import voice-over…")
        self.import_btn.setObjectName("primary")
        self.remove_btn = QPushButton("Remove")
        self.timeline_btn = QPushButton("Add to timeline")
        self.transcribe_btn = QPushButton("Transcribe — Coming in Phase 2")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setToolTip("Voice-over transcription is planned for Phase 2.")
        buttons = QHBoxLayout()
        for b in (self.import_btn, self.remove_btn, self.timeline_btn, self.transcribe_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.transport = TransportControls(self.player)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(buttons)
        layout.addWidget(info)
        layout.addWidget(self.transport)
        layout.addStretch(1)

        self.import_btn.clicked.connect(self.import_dialog)
        self.remove_btn.clicked.connect(lambda: ctx.guard(self, ctx.ws.media.remove_voice_over, modal=True))
        self.timeline_btn.clicked.connect(self._add_to_timeline)
        for topic in ("project.opened", "project.closed"):
            ctx.bridge.on(topic, lambda p: self.refresh())
        ctx.bridge.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("voice_over", "assets") else None)
        self.refresh()

    def import_dialog(self) -> None:
        pattern = " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))
        file, _ = QFileDialog.getOpenFileName(self, "Import voice-over", "", f"Audio ({pattern})")
        if file:
            self.import_path(Path(file))

    def import_path(self, path: Path) -> None:
        self.ctx.guard(self, lambda: self.ctx.ws.media.import_voice_over(path), modal=True, title="Voice-over")

    def _add_to_timeline(self) -> None:
        vo = self.ctx.ws.project.voice_over if self.ctx.ws.project else None
        if vo and vo.asset_id:
            self.ctx.guard(self, lambda: self.ctx.ws.timeline.add_asset(vo.asset_id), modal=True, title="Timeline")

    def refresh(self) -> None:
        project = self.ctx.ws.project
        vo = project.voice_over if project else None
        has = bool(vo and vo.asset_id)
        self.import_btn.setEnabled(project is not None)
        self.import_btn.setText("Replace voice-over…" if has else "Import voice-over…")
        self.remove_btn.setEnabled(has)
        self.timeline_btn.setEnabled(has)
        self.transport.set_enabled(False)
        self.player.unload()
        if not has or project is None:
            self.filename.setText("No voice-over imported")
            self.duration.setText("—")
            self.status.setText("")
            return
        asset = project.assets.get(vo.asset_id)
        self.filename.setText(vo.filename or "—")
        self.duration.setText(format_timecode(vo.duration) if vo.duration else "—")
        if asset is None or not project.asset_path(asset).is_file():
            self.status.setText("The voice-over file is missing from the project.")
            return
        self.status.setText("Transcript: not generated (Phase 2)")
        self.transport.set_enabled(True)
        self.player.load(project.asset_path(asset))

    def pause(self) -> None:
        self.player.pause()
