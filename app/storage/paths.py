"""Filesystem locations: per-user application directories and per-project layout."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.constants import APP_NAME, PROJECT_FILE, PROJECT_SUBDIRS

ENV_HOME_OVERRIDE = "AGENTTOOL_HOME"


@dataclass(frozen=True)
class AppPaths:
    """Per-user directories (settings, logs, recovery data). Lives outside any project."""

    config_dir: Path
    data_dir: Path

    @classmethod
    def default(cls) -> "AppPaths":
        override = os.environ.get(ENV_HOME_OVERRIDE)
        if override:
            base = Path(override)
            return cls(base / "config", base / "data")
        home = Path.home()
        if sys.platform.startswith("win"):
            root = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / APP_NAME
            return cls(root, root)
        if sys.platform == "darwin":
            root = home / "Library" / "Application Support" / APP_NAME
            return cls(root, root)
        config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / APP_NAME
        data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / APP_NAME
        return cls(config, data)

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def recent_projects_file(self) -> Path:
        return self.config_dir / "recent_projects.json"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def recovery_dir(self) -> Path:
        return self.data_dir / "recovery"

    def ensure(self) -> None:
        for d in (self.config_dir, self.data_dir, self.log_dir, self.recovery_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProjectPaths:
    """Layout of a single project folder."""

    root: Path

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def thumbnails_dir(self) -> Path:
        return self.root / "thumbnails"

    def media_subdir(self, asset_type: str) -> Path:
        return self.media_dir / {"video": "video", "image": "images", "audio": "audio"}.get(asset_type, "video")

    def create_structure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in PROJECT_SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_or_absolute: str) -> Path:
        """Resolve a stored asset path (relative to the project, or absolute for linked media)."""
        p = Path(relative_or_absolute)
        return p if p.is_absolute() else self.root / p
