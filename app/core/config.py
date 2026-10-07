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
    use_proxies: bool = False  # preference only; proxy generation arrives with the pro preview engine
    theme: str = "dark"
    ffmpeg_path: str = ""  # empty = discover on PATH
    ffprobe_path: str = ""

    def sanitized(self) -> "Settings":
        """Return a copy with out-of-range values replaced by safe ones."""
        s = Settings(**asdict(self))
        if not isinstance(s.autosave_interval_seconds, int) or isinstance(s.autosave_interval_seconds, bool):
            s.autosave_interval_seconds = DEFAULT_AUTOSAVE_SECONDS
        s.autosave_interval_seconds = max(MIN_AUTOSAVE_SECONDS, s.autosave_interval_seconds)
        if s.theme not in THEMES:
            s.theme = "dark"
        for name in ("default_project_location", "ffmpeg_path", "ffprobe_path"):
            if not isinstance(getattr(s, name), str):
                setattr(s, name, getattr(Settings(), name))
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
