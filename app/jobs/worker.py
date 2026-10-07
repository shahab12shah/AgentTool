"""Executes a single job on a worker thread and records the outcome."""

from __future__ import annotations

import time
from typing import Callable

from app.core.exceptions import AppError, JobCancelled
from app.jobs.job import Job, JobStatus, _now
from app.logging.logger import get_logger, log_event

_log = get_logger(__name__)
NotifyFn = Callable[[Job], None]


class JobContext:
    """Handed to job functions for progress reporting and cooperative cancellation."""

    def __init__(self, job: Job, notify: NotifyFn) -> None:
        self._job = job
        self._notify = notify
        self._last_emit = 0.0

    @property
    def job(self) -> Job:
        return self._job

    def is_cancelled(self) -> bool:
        return self._job.cancel_requested

    def check_cancelled(self) -> None:
        if self._job.cancel_requested:
            raise JobCancelled()

    def report(self, progress: float | None = None, message: str | None = None) -> None:
        """Report progress in percent (0-100) and/or a status message. Throttled to ~20 Hz."""
        if progress is not None:
            self._job.progress = max(0.0, min(100.0, float(progress)))
        if message is not None:
            self._job.message = message
        now = time.monotonic()
        if now - self._last_emit >= 0.05 or self._job.progress >= 100.0:
            self._last_emit = now
            self._notify(self._job)


def run_job(job: Job, notify: NotifyFn) -> None:
    """Run ``job.func`` and move the job to COMPLETED / FAILED / CANCELLED. Never raises."""
    job.status = JobStatus.RUNNING
    job.started_at = _now()
    job.finished_at = None
    job.error = None
    job.attempts += 1
    notify(job)
    log_event(_log, "job.started", job_id=job.id, job_type=job.type, title=job.title)
    try:
        job.result = job.func(JobContext(job, notify))
        if job.cancel_requested:
            raise JobCancelled()
        job.progress = 100.0
        job.status = JobStatus.COMPLETED
    except JobCancelled:
        job.status = JobStatus.CANCELLED
        job.message = "Cancelled"
    except AppError as exc:
        job.status = JobStatus.FAILED
        job.error = exc.user_message
        job.message = exc.user_message
        _log.warning("Job failed: %s", exc, extra={"job_id": job.id, "job_type": job.type, "details": exc.details})
    except Exception as exc:  # unexpected: log the traceback, show something friendly
        job.status = JobStatus.FAILED
        job.error = f"Unexpected error: {exc}"
        job.message = "Unexpected error — see the log for details."
        _log.exception("Job crashed", extra={"job_id": job.id, "job_type": job.type})
    finally:
        job.finished_at = _now()
    log_event(_log, "job.finished", job_id=job.id, job_type=job.type, status=job.status.value)
