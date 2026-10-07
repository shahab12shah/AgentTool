"""Atomic file writes: write to a temp file, verify, then replace."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Callable

from app.core.exceptions import ProjectError


def atomic_write_text(
    path: Path,
    text: str,
    *,
    tmp_suffix: str = ".tmp",
    verify: Callable[[str], None] | None = None,
) -> None:
    """Write ``text`` to ``path`` so readers never observe a partial file.

    ``verify`` receives the text read back from the temp file and may raise to abort
    the replacement (the existing file is then left untouched).
    """
    path = Path(path)
    tmp = path.with_name(path.name + tmp_suffix)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if verify is not None:
            verify(tmp.read_text(encoding="utf-8"))
        os.replace(tmp, path)
    except PermissionError as exc:
        _cleanup(tmp)
        raise ProjectError(f"No permission to write to {path.parent}.", details=str(exc)) from exc
    except OSError as exc:
        _cleanup(tmp)
        if exc.errno == errno.ENOSPC:
            raise ProjectError("The disk is full. Free some space and try again.", details=str(exc)) from exc
        raise ProjectError(f"Could not write {path.name}: {exc.strerror or exc}", details=str(exc)) from exc
    except BaseException:
        _cleanup(tmp)
        raise


def _cleanup(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
