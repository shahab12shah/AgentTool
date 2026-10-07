"""RenderQueue: renders run one after another (or a few at a time) on worker threads — never on the UI thread.

Each job owns a unique working folder, output file and log, so jobs cannot corrupt each other's temporary files, outputs or the shared chunk
cache (cache files are written atomically). States: QUEUED, RUNNING, PAUSED, CANCELING, COMPLETED, FAILED, CANCELED.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.logging.logger import get_logger
from app.project.project_schema import RenderSettings
from app.rendering.engine import RenderEngine
from app.rendering.errors import RenderCancelled, RenderError
from app.rendering.executor import JobSpec, RenderResult
from app.rendering.models import RenderProgress, RenderRecord, RenderStatus, now_iso

_log = get_logger(__name__)


@dataclass
class RenderJob:
    id: str
    title: str
    spec: JobSpec
    record: RenderRecord
    settings: RenderSettings
    status: RenderStatus = RenderStatus.QUEUED
    progress: RenderProgress = field(default_factory=RenderProgress)
    error: RenderError | None = None
    result: RenderResult | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    pause_event: threading.Event = field(default_factory=threading.Event, repr=False)
    finished_event: threading.Event = field(default_factory=threading.Event, repr=False)  # set after the record, metadata and history were written
    created_at: str = field(default_factory=now_iso)
    started_at: str = ""
    finished_at: str = ""
    force_cpu: bool = False
    attempts: int = 0

    @property
    def output_path(self):
        return self.spec.output_path


class RenderQueue:
    def __init__(self, engine: RenderEngine, on_update: Callable[[RenderJob], None], on_finished: Callable[[RenderJob], None], max_parallel: int = 1) -> None:
        self.engine, self._on_update, self._on_finished = engine, on_update, on_finished
        self._cv = threading.Condition()
        self._jobs: dict[str, RenderJob] = {}
        self._order: list[str] = []
        self._shutdown = False
        self._active = 0
        self._threads = [threading.Thread(target=self._loop, name=f"render-{i}", daemon=True) for i in range(max(1, max_parallel))]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ api
    def submit(self, job: RenderJob) -> RenderJob:
        with self._cv:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._cv.notify_all()
        self._notify(job)
        return job

    def jobs(self) -> list[RenderJob]:
        with self._cv:
            return [self._jobs[i] for i in self._order if i in self._jobs]

    def get(self, job_id: str) -> RenderJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.status.is_terminal:
                return False
            job.cancel_event.set()
            job.pause_event.clear()
            if job.status in (RenderStatus.QUEUED, RenderStatus.PAUSED) and job.started_at == "":
                job.status = RenderStatus.CANCELED  # never started: nothing to stop
                job.finished_at = now_iso()
                direct = True
            else:
                job.status = RenderStatus.CANCELING
                direct = False
            self._cv.notify_all()
        self._notify(job)
        if direct:
            self._finish(job)
        return True

    def pause(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.status.is_terminal or job.status is RenderStatus.CANCELING:
                return False
            job.pause_event.set()
            if job.status is RenderStatus.QUEUED:
                job.status = RenderStatus.PAUSED
        self._notify(job)
        return True

    def resume(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or not job.pause_event.is_set():
                return False
            job.pause_event.clear()
            if job.status is RenderStatus.PAUSED and not job.started_at:
                job.status = RenderStatus.QUEUED
            self._cv.notify_all()
        self._notify(job)
        return True

    def remove(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or not job.status.is_terminal:
                return False
            del self._jobs[job_id]
            self._order.remove(job_id)
        return True

    def wait_idle(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cv:
                busy = self._active > 0 or any(j.status in (RenderStatus.QUEUED, RenderStatus.RUNNING, RenderStatus.CANCELING) for j in self._jobs.values())
            if not busy:
                return True
            time.sleep(0.02)
        return False

    def shutdown(self) -> None:
        with self._cv:
            self._shutdown = True
            for j in self._jobs.values():
                if not j.status.is_terminal:
                    j.cancel_event.set()
                    j.pause_event.clear()
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=10)

    # ------------------------------------------------------------------ worker
    def _next(self) -> RenderJob | None:
        for i in self._order:
            j = self._jobs.get(i)
            if j is not None and j.status is RenderStatus.QUEUED and not j.pause_event.is_set():
                return j
        return None

    def _loop(self) -> None:
        while True:
            with self._cv:
                job = None
                while not self._shutdown:
                    job = self._next()
                    if job is not None:
                        job.status = RenderStatus.RUNNING
                        job.started_at = now_iso()
                        self._active += 1
                        break
                    self._cv.wait(0.5)
                if self._shutdown and job is None:
                    return
            try:
                self._run(job)  # type: ignore[arg-type]
            finally:
                with self._cv:
                    self._active -= 1
                    self._cv.notify_all()

    def _run(self, job: RenderJob) -> None:
        job.attempts += 1
        self._notify(job)
        last = [0.0]
        stage = [None]

        def prog(p: RenderProgress) -> None:
            job.progress = p
            now = time.monotonic()
            if now - last[0] >= 0.2 or p.overall >= 1.0 or p.stage is not stage[0]:  # a new stage is always shown at once
                last[0], stage[0] = now, p.stage
                self._notify(job)

        def checkpoint() -> None:
            if job.pause_event.is_set() and not job.cancel_event.is_set():
                job.status = RenderStatus.PAUSED
                self._notify(job)
                while job.pause_event.is_set() and not job.cancel_event.is_set():
                    time.sleep(0.1)
                if not job.cancel_event.is_set():
                    job.status = RenderStatus.RUNNING
                    self._notify(job)

        try:
            job.result = self.engine.run(job.spec, job.cancel_event, prog, checkpoint)
            job.status = RenderStatus.COMPLETED
            job.progress.overall = 1.0
        except RenderCancelled:
            job.status = RenderStatus.CANCELED
        except RenderError as exc:
            job.status, job.error = RenderStatus.FAILED, exc
        except Exception as exc:  # never lose a job to an unexpected error: report it as a failed render
            _log.exception("Render crashed", extra={"render_id": job.id})
            job.status = RenderStatus.FAILED
            job.error = RenderError(f"The render stopped unexpectedly: {exc}", stage=job.progress.stage.value, kind="unexpected", details=f"{type(exc).__name__}: {exc}")
        job.finished_at = now_iso()
        self._finish(job)

    def _notify(self, job: RenderJob) -> None:
        try:
            self._on_update(job)
        except Exception:
            _log.debug("render update handler failed", exc_info=True)

    def _finish(self, job: RenderJob) -> None:
        try:
            self._on_update(job)
            self._on_finished(job)
        except Exception:
            _log.exception("render finish handler failed")
        finally:
            job.finished_event.set()
