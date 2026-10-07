"""Job data model."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Job:
    """A unit of background work. ``func`` receives a ``JobContext`` and returns a result."""

    type: str
    title: str
    func: Callable[[Any], Any] = field(repr=False)
    id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0..100
    message: str = ""
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: Any = None
    attempts: int = 0
    on_complete: Callable[["Job"], None] | None = field(default=None, repr=False)
    on_error: Callable[["Job"], None] | None = field(default=None, repr=False)
    on_cancel: Callable[["Job"], None] | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    run_token: int = field(default=0, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "attempts": self.attempts,
        }
