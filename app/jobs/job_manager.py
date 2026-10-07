"""Thread-pool backed job queue with cancel / retry / pause and completion callbacks."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from app.core.events import EventBus, Topics
from app.core.exceptions import JobError
from app.jobs.job import Job, JobStatus, _now
from app.jobs.worker import JobContext, run_job
from app.logging.logger import get_logger

_log = get_logger(__name__)
Dispatcher = Callable[[Callable[[], None]], None]


class JobManager:
    """Runs jobs outside the UI thread.

    ``dispatcher`` decides on which thread completion callbacks run. The UI passes a
    dispatcher that marshals to the Qt main thread; by default callbacks run on the worker.
    """

    def __init__(self, bus: EventBus | None = None, max_workers: int = 3, dispatcher: Dispatcher | None = None) -> None:
        self._bus = bus
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._dispatch: Dispatcher = dispatcher or (lambda fn: fn())
        self._shutdown = False
        self._inflight = 0  # enqueued runs + undelivered callbacks (for wait_idle)

    def set_dispatcher(self, dispatcher: Dispatcher) -> None:
        self._dispatch = dispatcher

    def dispatch(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` where completion callbacks run (the UI thread in the app; the calling thread by default)."""
        self._dispatch(fn)

    # ----- submission -----
    def submit(
        self,
        job_type: str,
        func: Callable[[JobContext], Any],
        *,
        title: str,
        on_complete: Callable[[Job], None] | None = None,
        on_error: Callable[[Job], None] | None = None,
        on_cancel: Callable[[Job], None] | None = None,
    ) -> Job:
        if self._shutdown:
            raise JobError("The application is shutting down; no new jobs can be started.")
        job = Job(type=job_type, title=title, func=func, on_complete=on_complete, on_error=on_error, on_cancel=on_cancel)
        job.message = "Queued"
        with self._lock:
            self._jobs[job.id] = job
        self._publish(Topics.JOB_ADDED, job)
        self._enqueue(job)
        return job

    def _enqueue(self, job: Job) -> None:
        with self._lock:
            job.run_token += 1
            token = job.run_token
            self._inflight += 1
        try:
            self._executor.submit(self._run, job, token)
        except RuntimeError:  # executor already shut down
            self._release()

    def _release(self) -> None:
        with self._lock:
            self._inflight -= 1

    def _run(self, job: Job, token: int) -> None:
        try:
            self._run_inner(job, token)
        finally:
            self._release()

    def _run_inner(self, job: Job, token: int) -> None:
        with self._lock:
            if token != job.run_token or job.status is not JobStatus.QUEUED:
                return  # cancelled, paused or superseded while waiting
            if self._shutdown:
                return
        run_job(job, self._on_update)
        self._on_update(job)
        callback = {
            JobStatus.COMPLETED: job.on_complete,
            JobStatus.FAILED: job.on_error,
            JobStatus.CANCELLED: job.on_cancel,
        }.get(job.status)
        if callback is not None:
            self._schedule_callback(callback, job)

    def _schedule_callback(self, callback: Callable[[Job], None], job: Job) -> None:
        with self._lock:
            self._inflight += 1
        self._dispatch(lambda: self._safe_callback(callback, job))

    def _safe_callback(self, callback: Callable[[Job], None], job: Job) -> None:
        try:
            callback(job)
        except Exception:
            _log.exception("Job callback failed", extra={"job_id": job.id})
        finally:
            self._release()

    # ----- control -----
    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._require(job_id)
            if job.status.is_terminal:
                return False
            job.cancel_event.set()
            if job.status in (JobStatus.QUEUED, JobStatus.PAUSED):
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled"
                job.finished_at = _now()
                queued_cancel = True
            else:
                queued_cancel = False  # running: the job observes the flag and stops itself
                job.message = "Cancelling…"
        self._on_update(job)
        if queued_cancel and job.on_cancel:
            self._schedule_callback(job.on_cancel, job)
        return True

    def retry(self, job_id: str) -> Job:
        with self._lock:
            job = self._require(job_id)
            if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
                raise JobError("Only failed or cancelled jobs can be retried.")
            job.cancel_event.clear()
            job.status = JobStatus.QUEUED
            job.progress = 0.0
            job.error = None
            job.message = "Queued (retry)"
            job.started_at = job.finished_at = None
        self._on_update(job)
        self._enqueue(job)
        return job

    def pause(self, job_id: str) -> bool:
        """Pause a job that has not started yet. Running jobs cannot be paused in Phase 1."""
        with self._lock:
            job = self._require(job_id)
            if job.status is not JobStatus.QUEUED:
                return False
            job.status = JobStatus.PAUSED
            job.message = "Paused"
        self._on_update(job)
        return True

    def resume(self, job_id: str) -> bool:
        with self._lock:
            job = self._require(job_id)
            if job.status is not JobStatus.PAUSED:
                return False
            job.status = JobStatus.QUEUED
            job.message = "Queued"
        self._on_update(job)
        self._enqueue(job)
        return True

    def clear_finished(self) -> None:
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if j.status.is_terminal]:
                del self._jobs[jid]

    # ----- queries -----
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def active_jobs(self) -> list[Job]:
        return [j for j in self.jobs() if j.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED)]

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until no job is queued or running (paused jobs are ignored). For tests/shutdown."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._inflight == 0:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def shutdown(self, cancel_running: bool = True) -> None:
        self._shutdown = True
        if cancel_running:
            for job in self.active_jobs():
                self.cancel(job.id)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ----- internals -----
    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobError("That job no longer exists.")
        return job

    def _on_update(self, job: Job) -> None:
        self._publish(Topics.JOB_UPDATED, job)

    def _publish(self, topic: str, job: Job) -> None:
        if self._bus:
            self._bus.publish(topic, job=job)
