"""Settings dialog (autosave, default location, proxies, theme, FFmpeg paths)."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.config import THEMES, Settings
from app.core.constants import MIN_AUTOSAVE_SECONDS
from app.core.exceptions import AppError


class SettingsDialog(QDialog):
    def __init__(
        self, settings: Settings, ffmpeg_tester: Callable[[str], str], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._test = ffmpeg_tester  # service call: the UI never runs FFmpeg itself
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self._settings = settings

        self.autosave = QSpinBox()
        self.autosave.setRange(MIN_AUTOSAVE_SECONDS, 3600)
        self.autosave.setSuffix(" seconds")
        self.autosave.setValue(settings.autosave_interval_seconds)
        self.location = QLineEdit(settings.default_project_location)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_location)
        loc_row = QHBoxLayout()
        loc_row.addWidget(self.location, 1)
        loc_row.addWidget(browse)
        self.proxies = QCheckBox("Prefer proxy media for preview")
        self.proxies.setChecked(settings.use_proxies)
        self.proxies.setToolTip("Stored as a preference. Proxy generation arrives with the professional preview engine.")
        self.theme = QComboBox()
        self.theme.addItems(THEMES)
        self.theme.setCurrentText(settings.theme)
        self.ffmpeg = QLineEdit(settings.ffmpeg_path)
        self.ffmpeg.setPlaceholderText("Empty = find ffmpeg on PATH")
        self.ffprobe = QLineEdit(settings.ffprobe_path)
        self.ffprobe.setPlaceholderText("Empty = find ffprobe on PATH")
        test = QPushButton("Test FFmpeg")
        test.clicked.connect(self._test_ffmpeg)
        self.test_result = QLabel("")
        self.test_result.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Autosave every", self.autosave)
        form.addRow("Default project location", loc_row)
        form.addRow("", self.proxies)
        form.addRow("Theme", self.theme)
        form.addRow("FFmpeg path", self.ffmpeg)
        form.addRow("FFprobe path", self.ffprobe)
        form.addRow("", test)
        form.addRow("", self.test_result)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse_location(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Default project location", self.location.text())
        if d:
            self.location.setText(d)

    def _test_ffmpeg(self) -> None:
        try:
            self.test_result.setText(f"OK — {self._test(self.ffmpeg.text().strip())}")
        except AppError as exc:
            self.test_result.setText(exc.user_message)

    def result_settings(self) -> Settings:
        return replace(
            self._settings,
            autosave_interval_seconds=self.autosave.value(),
            default_project_location=self.location.text().strip() or self._settings.default_project_location,
            use_proxies=self.proxies.isChecked(),
            theme=self.theme.currentText(),
            ffmpeg_path=self.ffmpeg.text().strip(),
            ffprobe_path=self.ffprobe.text().strip(),
        )
