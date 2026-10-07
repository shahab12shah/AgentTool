"""Preview panel: plays a single media file (video, audio) or shows an image."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.timecode import format_timecode
from app.media.asset import Asset, AssetType
from app.preview.player import QtPreviewPlayer
from app.ui.context import UiContext


class TransportControls(QWidget):
    """Play / pause / stop, seek bar, time read-out and volume for a ``QtPreviewPlayer``."""

    def __init__(self, player: QtPreviewPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = player
        self._seeking = False
        self.play_btn = QPushButton("Play")
        self.play_btn.setObjectName("play")
        self.stop_btn = QPushButton("Stop")
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.time = QLabel("00:00:00.000 / --:--")
        self.time.setMinimumWidth(190)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(player.volume * 100))
        self.volume.setFixedWidth(90)
        self.volume.setToolTip("Volume")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for w in (self.play_btn, self.stop_btn, self.seek, self.time, QLabel("Vol"), self.volume):
            row.addWidget(w)
        row.setStretchFactor(self.seek, 1)

        self.play_btn.clicked.connect(self._toggle)
        self.stop_btn.clicked.connect(player.stop)
        self.seek.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.seek.sliderReleased.connect(self._seek_released)
        self.seek.sliderMoved.connect(lambda v: player.seek(v / 1000.0))
        self.volume.valueChanged.connect(lambda v: player.set_volume(v / 100.0))
        player.position_changed.connect(self._on_position)
        player.duration_changed.connect(self._on_duration)
        player.playing_changed.connect(lambda playing: self.play_btn.setText("Pause" if playing else "Play"))
        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        for w in (self.play_btn, self.stop_btn, self.seek):
            w.setEnabled(enabled)

    def _toggle(self) -> None:
        self._player.pause() if self._player.is_playing else self._player.play()

    def _seek_released(self) -> None:
        self._seeking = False
        self._player.seek(self.seek.value() / 1000.0)

    def _on_position(self, seconds: float) -> None:
        if not self._seeking:
            self.seek.setValue(int(seconds * 1000))
        self._update_time(seconds)

    def _on_duration(self, seconds: float) -> None:
        self.seek.setRange(0, int(seconds * 1000))
        self._update_time(self._player.position)

    def _update_time(self, pos: float) -> None:
        self.time.setText(f"{format_timecode(pos)} / {format_timecode(self._player.duration)}")


class PreviewPanel(QWidget):
    """Single-asset preview. A future preview engine replaces ``QtPreviewPlayer`` behind ``PreviewBackend``."""

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.player = QtPreviewPlayer(self)
        self.player.failed.connect(self._on_failed)
        self.title = QLabel("Preview")
        self.title.setObjectName("muted")

        self.placeholder = QLabel("Double-click media in the library to preview it")
        self.placeholder.setObjectName("placeholder")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.video = QVideoWidget()
        self.video.setObjectName("videoWidget")
        self.player.set_video_output(self.video)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(160, 90)
        self._pixmap: QPixmap | None = None

        self.stack = QStackedWidget()
        for w in (self.placeholder, self.video, self.image):
            self.stack.addWidget(w)
        self.transport = TransportControls(self.player)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.transport)
        self.current_asset_id: str | None = None

        ctx.bridge.on("project.closed", lambda p: self.clear())
        ctx.bridge.on("project.opened", lambda p: self.clear())

    # ----- public API -----
    def show_asset(self, asset: Asset) -> None:
        project = self.ctx.ws.project
        if project is None:
            return
        path = project.asset_path(asset)
        if not path.is_file():
            self.clear(f"Media file is missing:\n{path}")
            return
        self.player.unload()
        self.current_asset_id = asset.id
        self.title.setText(f"Preview — {asset.name}")
        if asset.type is AssetType.IMAGE:
            pm = QPixmap(str(path))
            if pm.isNull():
                self.clear(f"“{asset.name}” could not be displayed.")
                return
            self._pixmap = pm
            self._fit_image()
            self.stack.setCurrentWidget(self.image)
            self.transport.set_enabled(False)
            return
        if asset.type is AssetType.AUDIO:
            thumb = self.ctx.ws.media.thumbnail_file(asset)
            self._pixmap = QPixmap(str(thumb)) if thumb else None
            if self._pixmap and not self._pixmap.isNull():
                self._fit_image()
                self.stack.setCurrentWidget(self.image)
            else:
                self.placeholder.setText(f"♪  {asset.name}")
                self.stack.setCurrentWidget(self.placeholder)
        else:
            self.stack.setCurrentWidget(self.video)
        self.transport.set_enabled(True)
        self.player.load(path)

    def clear(self, message: str = "Double-click media in the library to preview it") -> None:
        self.player.unload()
        self.current_asset_id = None
        self._pixmap = None
        self.title.setText("Preview")
        self.placeholder.setText(message)
        self.stack.setCurrentWidget(self.placeholder)
        self.transport.set_enabled(False)

    def pause(self) -> None:
        self.player.pause()

    # ----- internals -----
    def _on_failed(self, message: str) -> None:
        self.ctx.status(message)
        self.placeholder.setText(message)
        self.stack.setCurrentWidget(self.placeholder)
        self.transport.set_enabled(False)

    def _fit_image(self) -> None:
        if self._pixmap and not self._pixmap.isNull():
            self.image.setPixmap(
                self._pixmap.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.stack.currentWidget() is self.image:
            self._fit_image()
