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
        provider_report: Callable[[], list] | None = None,
        research_report: Callable[[], list] | None = None, parent: QWidget | None = None,
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
        # --- visual research (Phase 3). Keys are never entered or stored: only the env var NAME. ---
        def line(value: str, placeholder: str = "") -> QLineEdit:
            e = QLineEdit(value)
            e.setPlaceholderText(placeholder)
            return e

        self.stock_dir = line(settings.local_stock_dir, "Folder of licensed/owned stock media")
        stock_browse = QPushButton("Browse…")
        stock_browse.clicked.connect(self._browse_stock)
        stock_row = QHBoxLayout()
        stock_row.addWidget(self.stock_dir, 1)
        stock_row.addWidget(stock_browse)
        self.wikimedia_url = line(settings.wikimedia_api_url, "Empty = Wikimedia Commons")
        self.youtube_url = line(settings.youtube_api_url, "Empty = YouTube Data API")
        self.youtube_env = line(settings.youtube_key_env)
        self.pexels_url = line(settings.pexels_api_url, "Empty = Pexels API")
        self.pexels_env = line(settings.pexels_key_env)
        self.ai_url = line(settings.ai_image_base_url, "Image generation API base URL")
        self.ai_model = line(settings.ai_image_model)
        self.ai_env = line(settings.ai_image_key_env)
        self.chromium = line(settings.chromium_path, "Empty = auto-detect (used for page screenshots)")
        form.addRow("Local stock folder", stock_row)
        form.addRow("Wikimedia API URL", self.wikimedia_url)
        form.addRow("YouTube API URL", self.youtube_url)
        form.addRow("YouTube key variable", self.youtube_env)
        form.addRow("Pexels API URL", self.pexels_url)
        form.addRow("Pexels key variable", self.pexels_env)
        form.addRow("AI image API URL", self.ai_url)
        form.addRow("AI image model", self.ai_model)
        form.addRow("AI image key variable", self.ai_env)
        form.addRow("Chromium path", self.chromium)
        self.research_status = QLabel()
        self.research_status.setWordWrap(True)
        self.research_status.setObjectName("muted")
        if research_report:
            self.research_status.setText("\n".join(
                f"{'✓' if r['available'] else '✗'} {r['label']}: " + ("configured" if r["available"] else r["reason"])
                + ("" if r["verified_live"] else "  (not verified against the live service)") for r in research_report()))
        form.addRow("Research sources", self.research_status)
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

    def _browse_stock(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Local stock folder", self.stock_dir.text())
        if d:
            self.stock_dir.setText(d)

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
            local_stock_dir=self.stock_dir.text().strip(),
            wikimedia_api_url=self.wikimedia_url.text().strip(),
            youtube_api_url=self.youtube_url.text().strip(),
            youtube_key_env=self.youtube_env.text().strip() or "YOUTUBE_API_KEY",
            pexels_api_url=self.pexels_url.text().strip(),
            pexels_key_env=self.pexels_env.text().strip() or "PEXELS_API_KEY",
            ai_image_base_url=self.ai_url.text().strip(),
            ai_image_model=self.ai_model.text().strip() or "gpt-image-1",
            ai_image_key_env=self.ai_env.text().strip() or "OPENAI_API_KEY",
            chromium_path=self.chromium.text().strip(),
        )
