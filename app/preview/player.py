"""Preview playback.

``PreviewBackend`` is the seam for a future professional preview engine (proxy-based,
frame-accurate, multi-track). Phase 1 ships ``QtPreviewPlayer`` on top of QtMultimedia,
which plays a single media file at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from app.logging.logger import get_logger

_log = get_logger(__name__)


class PreviewBackend(Protocol):
    """What the UI needs from any preview engine."""

    def load(self, path: Path) -> None: ...
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def stop(self) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def set_volume(self, volume: float) -> None: ...
    @property
    def duration(self) -> float: ...
    @property
    def position(self) -> float: ...


class QtPreviewPlayer(QObject):
    """Seconds-based wrapper around ``QMediaPlayer`` (positions are float seconds)."""

    position_changed = Signal(float)
    duration_changed = Signal(float)
    playing_changed = Signal(bool)
    failed = Signal(str)  # user-friendly message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.8)
        self._player.positionChanged.connect(lambda ms: self.position_changed.emit(ms / 1000.0))
        self._player.durationChanged.connect(lambda ms: self.duration_changed.emit(ms / 1000.0))
        self._player.playbackStateChanged.connect(
            lambda st: self.playing_changed.emit(st == QMediaPlayer.PlaybackState.PlayingState)
        )
        self._player.errorOccurred.connect(self._on_error)
        self._path: Path | None = None

    def set_video_output(self, widget) -> None:
        self._player.setVideoOutput(widget)

    def _on_error(self, error, message: str) -> None:
        _log.warning("Preview error: %s", message, extra={"path": str(self._path)})
        name = self._path.name if self._path else "this file"
        self.failed.emit(f"“{name}” cannot be previewed on this system ({message or 'unsupported format'}).")

    # ----- transport -----
    def load(self, path: Path) -> None:
        self._path = Path(path)
        self._player.setSource(QUrl.fromLocalFile(str(path)))

    def unload(self) -> None:
        self._player.stop()
        self._player.setSource(QUrl())
        self._path = None

    def play(self) -> None:
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def stop(self) -> None:
        self._player.stop()

    def seek(self, seconds: float) -> None:
        self._player.setPosition(int(max(0.0, seconds) * 1000))

    def set_volume(self, volume: float) -> None:
        self._audio.setVolume(max(0.0, min(1.0, volume)))

    @property
    def volume(self) -> float:
        return float(self._audio.volume())

    @property
    def duration(self) -> float:
        return self._player.duration() / 1000.0

    @property
    def position(self) -> float:
        return self._player.position() / 1000.0

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
