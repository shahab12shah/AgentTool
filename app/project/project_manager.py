"""Project lifecycle: create, open, save, save-as, close, recent projects."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.performance.profiler import profiler
from app.core.constants import (
    PROJECT_BACKUP_SUFFIX,
    PROJECT_FILE,
    PROJECT_SUBDIRS,
    PROJECT_TMP_SUFFIX,
)
from app.core.events import EventBus, Topics
from app.core.exceptions import InvalidProjectError, ProjectError
from app.logging.logger import get_logger, log_event
from app.project.project import Project, utc_now
from app.project.project_schema import ProjectSettings, validate_document
from app.storage.atomic import atomic_write_text
from app.storage.paths import ProjectPaths

_log = get_logger(__name__)
MAX_RECENT = 10


class ProjectManager:
    """Owns the currently open project and everything about persisting it."""

    def __init__(self, bus: EventBus, recent_file: Path | None = None) -> None:
        self._bus = bus
        self._recent_file = recent_file
        self.current: Project | None = None

    # ----- dirty state -----
    def set_dirty(self, dirty: bool = True) -> None:
        project = self.current
        if project is not None and project.dirty != dirty:
            project.dirty = dirty
            self._bus.publish(Topics.DIRTY_CHANGED, dirty=dirty)

    # ----- create -----
    def create(self, name: str, parent_dir: Path, settings: ProjectSettings | None = None) -> Project:
        name = name.strip()
        if not name:
            raise ProjectError("Please enter a project name.")
        if any(ch in name for ch in '<>:"/\\|?*\0'):
            raise ProjectError('The project name cannot contain any of < > : " / \\ | ? *')
        root = Path(parent_dir).expanduser() / name
        if root.exists() and any(root.iterdir()):
            raise ProjectError(f"The folder “{root}” already exists and is not empty. Choose another name or location.")
        project = Project.new(name, settings)
        project.root = root
        try:
            ProjectPaths(root).create_structure()
        except PermissionError as exc:
            raise ProjectError(f"No permission to create a project in {parent_dir}.", details=str(exc)) from exc
        except OSError as exc:
            raise ProjectError(f"Could not create the project folder: {exc.strerror or exc}", details=str(exc)) from exc
        self._write(project)
        self._adopt(project)
        log_event(_log, "project.created", project_id=project.project_id, path=str(root))
        return project

    # ----- open -----
    def open(self, path: Path, *, from_backup: bool = False) -> Project:
        """Open a project folder (or its ``project.json``)."""
        path = Path(path).expanduser()
        root = path.parent if path.is_file() else path
        file = root / PROJECT_FILE
        if from_backup:
            file = file.with_name(file.name + PROJECT_BACKUP_SUFFIX)
        if not file.is_file():
            raise ProjectError(f"“{root}” is not a project folder (no {PROJECT_FILE} found).")
        project = self.load_file(file, root)
        self._clean_partial_files(root)
        self._adopt(project)
        self._remember(root)
        log_event(_log, "project.opened", project_id=project.project_id, path=str(root))
        return project

    @staticmethod
    def load_file(file: Path, root: Path) -> Project:
        try:
            raw = file.read_text(encoding="utf-8")
        except PermissionError as exc:
            raise ProjectError(f"No permission to read {file}.", details=str(exc)) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ProjectError(f"The project file could not be read: {exc}", details=str(exc)) from exc
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            hint = " A backup (project.json.bak) exists." if file.with_name(file.name + PROJECT_BACKUP_SUFFIX).is_file() else ""
            raise InvalidProjectError(
                f"The project file is corrupt and cannot be opened.{hint}", details=f"JSON error: {exc}"
            ) from exc
        return Project.from_document(doc, root=root)

    # ----- save -----
    @profiler.timed("project.save")
    def save(self, project: Project | None = None) -> Project:
        project = project or self.current
        if project is None:
            raise ProjectError("There is no open project to save.")
        project.validate()
        project.updated_at = utc_now()
        self._write(project)
        if project is self.current:
            self.set_dirty(False)
        self._bus.publish(Topics.PROJECT_SAVED, project_id=project.project_id)
        log_event(_log, "project.saved", project_id=project.project_id, path=str(project.root))
        return project

    def _write(self, project: Project) -> None:
        file = project.paths.project_file
        project.validate()
        text = json.dumps(project.to_document(), indent=2, ensure_ascii=False)

        def verify(written: str) -> None:
            validate_document(json.loads(written))

        if file.is_file():  # keep one known-good generation
            try:
                shutil.copy2(file, file.with_name(file.name + PROJECT_BACKUP_SUFFIX))
            except OSError:
                _log.warning("Could not refresh project backup", exc_info=True)
        atomic_write_text(file, text, tmp_suffix=PROJECT_TMP_SUFFIX, verify=verify)

    def save_as(self, new_parent: Path, new_name: str, project: Project | None = None, progress=None) -> Project:
        """Copy the project folder to ``new_parent/new_name`` and switch to the copy.

        May copy large media: run it inside a job, not on the UI thread.
        """
        project = project or self.current
        if project is None or project.root is None:
            raise ProjectError("There is no open project to save.")
        new_name = new_name.strip()
        if not new_name:
            raise ProjectError("Please enter a project name.")
        target = Path(new_parent).expanduser() / new_name
        if target.resolve() == project.root.resolve():
            raise ProjectError("Choose a different folder or name for “Save As”.")
        if target.exists() and any(target.iterdir()):
            raise ProjectError(f"The folder “{target}” already exists and is not empty.")
        project.validate()
        files = [p for p in project.root.rglob("*") if p.is_file() and not p.name.endswith((".part", PROJECT_TMP_SUFFIX))]
        try:
            for i, src in enumerate(files):
                rel = src.relative_to(project.root)
                if rel.as_posix() in (PROJECT_FILE, PROJECT_FILE + PROJECT_BACKUP_SUFFIX):
                    continue
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                if progress:
                    progress((i + 1) / max(len(files), 1), f"Copying {rel.name}")
            for sub in PROJECT_SUBDIRS:
                (target / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProjectError(f"Could not copy the project: {exc.strerror or exc}", details=str(exc)) from exc
        copy = Project.from_document(project.to_document(), root=target)
        copy.project_name = new_name
        self._write(copy)
        return copy

    def switch_to(self, project: Project) -> None:
        """Make ``project`` (e.g. the result of ``save_as``) the current project."""
        self._adopt(project)
        if project.root:
            self._remember(project.root)

    # ----- close -----
    def close(self) -> None:
        project = self.current
        self.current = None
        if project is not None:
            self._bus.publish(Topics.PROJECT_CLOSED, project_id=project.project_id)
            log_event(_log, "project.closed", project_id=project.project_id)

    def _adopt(self, project: Project) -> None:
        self.current = project
        project.dirty = False
        self._bus.publish(Topics.PROJECT_OPENED, project_id=project.project_id)

    @staticmethod
    def _clean_partial_files(root: Path) -> None:
        """Remove leftovers from interrupted copies."""
        for part in (root / "media").rglob("*.part"):
            try:
                part.unlink()
            except OSError:
                pass
        (root / (PROJECT_FILE + PROJECT_TMP_SUFFIX)).unlink(missing_ok=True)

    # ----- recent projects -----
    def recent_projects(self) -> list[dict[str, str]]:
        if self._recent_file is None:
            return []
        try:
            items = json.loads(self._recent_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and (Path(item.get("path", "")) / PROJECT_FILE).is_file():
                out.append({"name": str(item.get("name", "")), "path": str(item["path"])})
        return out

    def _remember(self, root: Path) -> None:
        if self._recent_file is None or self.current is None:
            return
        entry = {"name": self.current.project_name, "path": str(root.resolve())}
        items = [e for e in self.recent_projects() if e["path"] != entry["path"]]
        try:
            atomic_write_text(self._recent_file, json.dumps([entry] + items[: MAX_RECENT - 1], indent=2))
        except ProjectError:
            _log.warning("Could not update recent projects", exc_info=True)
