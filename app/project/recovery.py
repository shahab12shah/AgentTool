"""Crash-recovery snapshots, stored outside the project folder.

A snapshot exists only while a project has unsaved changes. A normal save or a clean
close removes it, so finding one at startup means the previous session ended abnormally.
Recovering never overwrites the main ``project.json``; the user must save explicitly.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.constants import PROJECT_FILE
from app.core.exceptions import InvalidProjectError, RecoveryError
from app.logging.logger import get_logger, log_event
from app.project.project import Project
from app.storage.atomic import atomic_write_text

_log = get_logger(__name__)
RECOVERY_FILE = "recovery.json"
RECOVERY_VERSION = 1


@dataclass(frozen=True)
class RecoveryEntry:
    project_id: str
    project_name: str
    project_root: Path
    saved_at: str  # ISO-8601 UTC

    @property
    def saved_at_local(self) -> str:
        try:
            return datetime.fromisoformat(self.saved_at).astimezone().strftime("%H:%M:%S")
        except ValueError:
            return self.saved_at


class RecoveryManager:
    def __init__(self, recovery_dir: Path) -> None:
        self.dir = Path(recovery_dir)

    def _file(self, project_id: str) -> Path:
        return self.dir / project_id / RECOVERY_FILE

    def write_snapshot(self, project: Project) -> None:
        """Persist a snapshot of ``project``."""
        if project.root is not None:
            self.write_document(project.project_id, project.project_name, project.root, project.to_document())

    def write_document(self, project_id: str, name: str, root: Path, document: dict[str, Any]) -> None:
        payload = {
            "recovery_version": RECOVERY_VERSION,
            "project_id": project_id,
            "project_name": name,
            "project_root": str(root),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "document": document,
        }
        atomic_write_text(self._file(project_id), json.dumps(payload, ensure_ascii=False))

    def list_entries(self) -> list[RecoveryEntry]:
        entries: list[RecoveryEntry] = []
        if not self.dir.is_dir():
            return entries
        for sub in sorted(self.dir.iterdir()):
            file = sub / RECOVERY_FILE
            if not file.is_file():
                continue
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                entries.append(
                    RecoveryEntry(data["project_id"], data["project_name"], Path(data["project_root"]), data["saved_at"])
                )
            except (OSError, ValueError, KeyError, TypeError):
                _log.warning("Discarding unreadable recovery data", extra={"path": str(file)})
                shutil.rmtree(sub, ignore_errors=True)
        return entries

    def load_project(self, project_id: str) -> Project:
        """Rebuild the project from its snapshot. The result is marked dirty and not yet saved."""
        file = self._file(project_id)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            root = Path(data["project_root"])
            if not (root / PROJECT_FILE).is_file() and not root.is_dir():
                raise RecoveryError(
                    f"The project folder “{root}” no longer exists, so it cannot be recovered."
                )
            project = Project.from_document(data["document"], root=root)
        except RecoveryError:
            raise
        except InvalidProjectError as exc:
            raise RecoveryError("The recovery data is damaged and cannot be used.", details=str(exc)) from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RecoveryError("The recovery data could not be read.", details=str(exc)) from exc
        project.dirty = True
        log_event(_log, "recovery.loaded", project_id=project_id)
        return project

    def discard(self, project_id: str) -> None:
        shutil.rmtree(self.dir / project_id, ignore_errors=True)
