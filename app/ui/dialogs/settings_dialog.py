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
        self, settings: Settings, ffmpeg_tester: Callable[[str], str],
        provider_report: Callable[[], list] | None = None, parent: QWidget | None = None,
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

        # --- transcription (Phase 2). API keys are never entered or stored here: only the env var NAME. ---
        self.provider = QComboBox()
        self.provider.addItems(["auto", "faster-whisper", "api", "pocketsphinx"])
        self.provider.setCurrentText(settings.transcription_provider)
        self.language = QLineEdit(settings.transcription_language)
        self.language.setPlaceholderText("en  (empty = detect)")
        self.whisper_model = QLineEdit(settings.whisper_model)
        self.whisper_model.setPlaceholderText("e.g. small.en, or a model folder")
        self.api_url = QLineEdit(settings.api_base_url)
        self.api_url.setPlaceholderText("https://api.openai.com/v1")
        self.api_model = QLineEdit(settings.api_model)
        self.api_key_env = QLineEdit(settings.api_key_env)
        key_note = QLabel("The API key itself is read from that environment variable and is never saved by this application.")
        key_note.setObjectName("muted")
        key_note.setWordWrap(True)
        form.addRow("Transcription engine", self.provider)
        form.addRow("Language", self.language)
        form.addRow("Whisper model", self.whisper_model)
        form.addRow("API base URL", self.api_url)
        form.addRow("API model", self.api_model)
        form.addRow("API key variable", self.api_key_env)
        form.addRow("", key_note)
        self.provider_status = QLabel()
        self.provider_status.setWordWrap(True)
        self.provider_status.setObjectName("muted")
        if provider_report:
            lines = []
            for name, ok, why, note in provider_report():
                lines.append(f"{'✓' if ok else '✗'} {name}: " + ("available" + (f" — {note}" if note else "") if ok else why))
            self.provider_status.setText("\n".join(lines))
        form.addRow("Engine status", self.provider_status)
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
            transcription_provider=self.provider.currentText(),
            transcription_language=self.language.text().strip(),
            whisper_model=self.whisper_model.text().strip(),
            api_base_url=self.api_url.text().strip(),
            api_model=self.api_model.text().strip() or "whisper-1",
            api_key_env=self.api_key_env.text().strip() or "OPENAI_API_KEY",
        )
