"""User settings, persisted as JSON outside any project."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from app.core.constants import DEFAULT_AUTOSAVE_SECONDS, MIN_AUTOSAVE_SECONDS
from app.logging.logger import get_logger
from app.storage.atomic import atomic_write_text

_log = get_logger(__name__)
THEMES = ("dark", "light")


@dataclass
class Settings:
    autosave_interval_seconds: int = DEFAULT_AUTOSAVE_SECONDS
    default_project_location: str = str(Path.home() / "AgentToolProjects")
    use_proxies: bool = True  # editing previews read proxy media when a proxy exists (exports always read the originals)
    theme: str = "dark"
    ffmpeg_path: str = ""  # empty = discover on PATH
    ffprobe_path: str = ""
    # Phase 2: transcription (API keys are never stored here; only the *name* of the environment variable)
    transcription_provider: str = "auto"  # auto | faster-whisper | api | pocketsphinx
    transcription_language: str = "en"  # empty = let the provider detect
    whisper_model: str = ""  # faster-whisper model size/name or local folder
    api_base_url: str = ""  # OpenAI-compatible endpoint, e.g. https://api.openai.com/v1
    api_model: str = "whisper-1"
    api_key_env: str = "OPENAI_API_KEY"
    # Phase 3: visual research. Only the NAMES of environment variables are stored, never keys.
    local_stock_dir: str = ""  # folder of licensed/owned stock media searched by the local provider
    wikimedia_api_url: str = ""  # empty = https://commons.wikimedia.org/w/api.php
    youtube_api_url: str = ""
    youtube_key_env: str = "YOUTUBE_API_KEY"
    pexels_api_url: str = ""
    pexels_key_env: str = "PEXELS_API_KEY"
    ai_image_base_url: str = ""
    ai_image_model: str = "gpt-image-1"
    ai_image_key_env: str = "OPENAI_API_KEY"
    chromium_path: str = ""  # empty = auto-detect

    def sanitized(self) -> "Settings":
        """Return a copy with out-of-range values replaced by safe ones."""
        s = Settings(**asdict(self))
        if not isinstance(s.autosave_interval_seconds, int) or isinstance(s.autosave_interval_seconds, bool):
            s.autosave_interval_seconds = DEFAULT_AUTOSAVE_SECONDS
        s.autosave_interval_seconds = max(MIN_AUTOSAVE_SECONDS, s.autosave_interval_seconds)
        if s.theme not in THEMES:
            s.theme = "dark"
        for name in ("default_project_location", "ffmpeg_path", "ffprobe_path", "transcription_language", "whisper_model",
                     "api_base_url", "api_model", "api_key_env", "local_stock_dir", "wikimedia_api_url", "youtube_api_url",
                     "youtube_key_env", "pexels_api_url", "pexels_key_env", "ai_image_base_url", "ai_image_model",
                     "ai_image_key_env", "chromium_path"):
            if not isinstance(getattr(s, name), str):
                setattr(s, name, getattr(Settings(), name))
        if s.transcription_provider not in ("auto", "faster-whisper", "api", "pocketsphinx"):
            s.transcription_provider = "auto"
        s.use_proxies = bool(s.use_proxies)
        return s


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> Settings:
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Settings()
        except (OSError, ValueError):
            _log.warning("Settings file unreadable; using defaults", extra={"path": str(self.path)})
            return Settings()
        if not isinstance(data, dict):
            return Settings()
        known = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in data.items() if k in known}).sanitized()

    def save(self, settings: Settings) -> None:
        atomic_write_text(self.path, json.dumps(asdict(settings.sanitized()), indent=2))
