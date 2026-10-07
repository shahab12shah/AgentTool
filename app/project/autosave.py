"""Background autosave into the recovery area (never into the main project file)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.logging.logger import get_logger, log_event
from app.project.project import Project
from app.project.recovery import RecoveryManager

_log = get_logger(__name__)


class AutosaveService:
    """Serialises on the caller's thread (cheap) and writes on a background thread.

    Requests are coalesced; ``clear`` guarantees a stale snapshot can never appear after a
    successful save or a clean close.
    """

    def __init__(self, recovery: RecoveryManager, on_error: Callable[[str], None] | None = None) -> None:
        self._recovery = recovery
        self._on_error = on_error
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autosave")
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._pending: tuple[int, str, str, Any, dict[str, Any]] | None = None
        self._generation = 0
        self._scheduled = False

    def request(self, project: Project | None, *, force: bool = False) -> bool:
        """Queue a snapshot if ``project`` has unsaved changes. Returns True if queued."""
        if project is None or project.root is None or not (project.dirty or force):
            return False
        document = project.to_document()  # serialise on the calling thread; avoids racing with edits
        with self._lock:
            self._pending = (self._generation, project.project_id, project.project_name, project.root, document)
            if self._scheduled:
                return True
            self._scheduled = True
            self._idle.clear()
        self._executor.submit(self._drain)
        return True

    def _drain(self) -> None:
        while True:
            with self._lock:
                pending, self._pending = self._pending, None
                if pending is None:
                    self._scheduled = False
                    self._idle.set()
                    return
                gen, pid, name, root, doc = pending
                if gen != self._generation:
                    continue
                try:  # written under the lock so ``clear`` cannot interleave
                    self._recovery.write_document(pid, name, root, doc)
                    log_event(_log, "autosave.written", project_id=pid)
                except Exception as exc:
                    _log.exception("Autosave failed")
                    if self._on_error:
                        self._on_error(f"Autosave failed: {getattr(exc, 'user_message', exc)}")

    def clear(self, project_id: str) -> None:
        """Drop any snapshot (and queued write) for ``project_id``."""
        with self._lock:
            self._generation += 1
            self._pending = None
            self._recovery.discard(project_id)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return self._idle.wait(timeout)

    def shutdown(self) -> None:
        self.wait_idle(5.0)
        self._executor.shutdown(wait=False)
