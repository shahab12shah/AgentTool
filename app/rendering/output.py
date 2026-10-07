"""OutputManager: where a render is written, without ever overwriting a file by accident, plus the sidecar metadata."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path

from app.rendering.errors import RenderError
from app.rendering.models import disk_free
from app.rendering.presets import ResolvedOutput
from app.storage.atomic import atomic_write_text

_reserve_lock = threading.Lock()
_reserved: set[str] = set()  # outputs promised to queued/running jobs (so two jobs never pick the same name)
INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(name: str) -> str:
    s = INVALID.sub("_", name).strip(" .") or "video"
    return re.sub(r"\s+", "_", s)[:80]


class OutputManager:
    def __init__(self, default_dir: Path) -> None:
        self.default_dir = Path(default_dir)

    def filename(self, project_name: str, out: ResolvedOutput, kind: str = "export") -> str:
        res = f"{min(out.width, out.height)}p"
        tag = "_draft" if kind == "draft" else ""
        return f"{safe_stem(project_name)}_{res}_{out.fps}fps{tag}.{out.container}"

    def validate_target(self, directory: Path) -> str | None:
        """None when ``directory`` can receive a file, otherwise a user-facing reason."""
        d = Path(directory)
        if d.exists() and not d.is_dir():
            return f"“{d}” is a file, not a folder."
        probe = d
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if not os.access(probe, os.W_OK):
            return f"The folder “{d}” is not writable."
        return None

    def reserve(self, project_name: str, out: ResolvedOutput, directory: Path | None = None, *, kind: str = "export", overwrite: bool = False,
                explicit: Path | None = None) -> Path:
        """Choose the output path. Existing files get ``_01``, ``_02``... unless the user explicitly chose to overwrite."""
        if explicit is not None:
            target = Path(explicit)
            directory = target.parent
            name = target.name
            if target.suffix.lower() != f".{out.container}":
                name = target.stem + f".{out.container}"
        else:
            directory = Path(directory or self.default_dir)
            name = self.filename(project_name, out, kind)
        problem = self.validate_target(directory)
        if problem:
            raise RenderError(problem, stage="Validating Project", kind="invalid_path", possible_issue="The output location cannot be used.")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RenderError(f"The output folder could not be created: {exc.strerror or exc}", stage="Validating Project", kind="invalid_path") from exc
        stem, suffix = Path(name).stem, Path(name).suffix
        with _reserve_lock:
            cand, n = directory / name, 0
            while not overwrite and (cand.exists() or str(cand) in _reserved):
                n += 1
                cand = directory / f"{stem}_{n:02d}{suffix}"
            if overwrite and str(cand) in _reserved:
                raise RenderError("Another render is already writing that file.", stage="Validating Project", kind="invalid_path")
            _reserved.add(str(cand))
        return cand

    @staticmethod
    def release(path: Path) -> None:
        with _reserve_lock:
            _reserved.discard(str(path))

    @staticmethod
    def commit(tmp: Path, final: Path) -> Path:
        """Move the finished file into place (atomic where the filesystem allows; otherwise copy + replace)."""
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(tmp, final)
        except OSError:
            part = final.with_name(final.name + ".part")
            shutil.copy2(tmp, part)
            os.replace(part, final)
            tmp.unlink(missing_ok=True)
        return final

    @staticmethod
    def save_metadata(folder: Path, record: dict) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / "metadata.json"
        atomic_write_text(p, json.dumps(record, indent=2, ensure_ascii=False))
        return p

    @staticmethod
    def disk_check(directory: Path, required: int) -> tuple[bool, int]:
        free = disk_free(directory)
        return (free < 0 or free >= required), free

    @staticmethod
    def open_location(path: Path) -> bool:
        """Open the folder in the system file manager (the UI offers this after an export)."""
        import subprocess
        import sys

        target = path if path.is_dir() else path.parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
            return True
        except OSError:
            return False
