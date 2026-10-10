"""Priority, resource-aware job scheduler with cancel / retry / pause and completion callbacks.

Scheduling model (all state under one lock; a small bounded pool of daemon worker threads pulls from per-priority heaps):

* Three classes (``Priority.HIGH/MEDIUM/LOW``), FIFO inside a class, plus aging: a job that waited ``aging_seconds`` is ranked one class higher
  (per elapsed interval) for *ordering only*, so a LOW job can never starve behind a stream of newer work. Aging never lets it use a reserved slot.
* Slots: at most ``max_workers`` non-HIGH jobs run at once; ``foreground_workers`` more threads exist that only HIGH jobs may use, so a flood of
  thumbnails/proxies cannot delay a preview request. Total threads are bounded by ``max_workers + foreground_workers``.
* ``set_resource_limit(name, n)`` caps how many jobs submitted with ``resource=name`` run together (e.g. one proxy encode at a time).
* ``set_pressure_provider`` (the ``ResourceMonitor.pressure`` shape): ELEVATED holds LOW and shrinks the MEDIUM cap by ``elevated_factor``;
  CRITICAL holds LOW and MEDIUM (they stay QUEUED with a "Waiting" message and resume by themselves). Running jobs are never killed.
* ``dedupe_key``: submitting while a job with that key is QUEUED/RUNNING returns that job (promoted if the new priority is higher). The coalesced
  submit's ``on_complete`` / ``on_error`` / ``on_cancel`` are chained after the existing ones; all of them run, a failing one never blocks the rest.
* ``retries=N``: a job function may raise ``RecoverableJobError``; it is re-queued with exponential backoff (``retry_delay * 2**k``) up to N times.
  Any other error fails at once; a cancel always wins. ``Job.attempts`` counts every run.
* ``owner``: callers tag jobs with the project id they belong to and call ``cancel_owner(old_id)`` when the project is closed or switched.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import deque
from typing import Any, Callable

from app.core.events import EventBus, Topics
from app.core.exceptions import JobError
from app.jobs.job import Job, JobStatus, Priority, _now
from app.jobs.worker import JobContext, run_job
from app.logging.logger import get_logger, log_event
from app.performance.profiler import profiler
from app.performance.resource_monitor import CRITICAL, ELEVATED, NORMAL

_log = get_logger(__name__)
Dispatcher = Callable[[Callable[[], None]], None]
WAIT_MESSAGE = "Waiting: low memory/disk"
_NCLS = len(Priority)


class _Entry:
    __slots__ = ("job", "token", "seq", "enq", "cls", "not_before", "alive")

    def __init__(self, job: Job, token: int, seq: int, enq: float, cls: int, not_before: float) -> None:
        self.job, self.token, self.seq, self.enq, self.cls, self.not_before, self.alive = job, token, seq, enq, cls, not_before, True

    def __lt__(self, other: "_Entry") -> bool:
        return self.seq < other.seq


def _chain(first: Callable[[Job], None] | None, second: Callable[[Job], None] | None) -> Callable[[Job], None] | None:
    if first is None or second is None:
        return first or second

    def both(job: Job) -> None:
        for cb in (first, second):
            try:
                cb(job)
            except Exception:
                _log.exception("Job callback failed", extra={"job_id": job.id})

    return both


class JobManager:
    """Runs jobs outside the UI thread.

    ``dispatcher`` decides on which thread completion callbacks run. The UI passes a
    dispatcher that marshals to the Qt main thread; by default callbacks run on the worker.
    """

    def __init__(self, bus: EventBus | None = None, max_workers: int = 3, dispatcher: Dispatcher | None = None, *, foreground_workers: int = 1,
                 aging_seconds: float = 60.0, elevated_factor: float = 0.5, clock: Callable[[], float] = time.monotonic) -> None:
        self._bus = bus
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._dispatch: Dispatcher = dispatcher or (lambda fn: fn())
        self._shutdown = False
        self._inflight = 0  # queue entries + running jobs + undelivered callbacks (for wait_idle)
        self._clock = clock
        self.aging_seconds = float(aging_seconds)
        self.elevated_factor = float(elevated_factor)
        self.max_retry_delay = 300.0
        self.idle_timeout = 30.0  # an idle worker thread exits after this long (a new one is spawned on demand)
        self.pressure_min_interval = 0.25  # job completions re-read the pressure at most this often
        self._max_workers = max(1, int(max_workers))
        self._fg_workers = max(0, int(foreground_workers))
        self._res_limits: dict[str, int] = {}
        self._type_priority: dict[str, Priority] = {}
        # queue
        self._heaps: list[list[_Entry]] = [[] for _ in range(_NCLS)]
        self._live = [0] * _NCLS
        self._dead = [0] * _NCLS
        self._entries: dict[str, _Entry] = {}
        self._seq = 0
        # running
        self._running: dict[str, tuple[Job, bool, str]] = {}  # id -> (job, counts against max_workers, resource)
        self._nonhigh_running = 0
        self._res_running: dict[str, int] = {}
        self._threads: list[threading.Thread] = []
        self._thread_seq = 0
        self._idle = 0
        self._dedupe: dict[str, Job] = {}
        # pressure
        self._provider: Callable[[], dict] | None = None
        self._pressure = NORMAL
        self._pressure_at = 0.0
        self._provider_failed = False
        self._poll_interval = 2.0
        self._poll_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._throttle_logged: set[tuple] = set()
        # stats
        self._retried = 0
        self._coalesced = 0
        self._throttled = 0
        self._failed_recent: deque[dict[str, Any]] = deque(maxlen=50)

    def set_dispatcher(self, dispatcher: Dispatcher) -> None:
        self._dispatch = dispatcher

    def dispatch(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` where completion callbacks run (the UI thread in the app; the calling thread by default)."""
        self._dispatch(fn)

    # ----- configuration -----
    def set_limits(self, *, max_workers: int | None = None, foreground_workers: int | None = None) -> None:
        """Change the worker limits at runtime. Takes effect for jobs that start from now on; running jobs are never interrupted."""
        with self._cond:
            if max_workers is not None:
                self._max_workers = max(1, int(max_workers))
            if foreground_workers is not None:
                self._fg_workers = max(0, int(foreground_workers))
            self._kick_locked()

    def set_resource_limit(self, resource: str, limit: int | None) -> None:
        """Cap concurrent jobs submitted with ``resource=resource``. ``None`` or ``<= 0`` removes the cap."""
        with self._cond:
            if limit is None or int(limit) <= 0:
                self._res_limits.pop(resource, None)
            else:
                self._res_limits[resource] = int(limit)
            self._throttle_logged.discard(("resource", resource))
            self._kick_locked()

    def set_priority_policy(self, *, aging_seconds: float | None = None, elevated_factor: float | None = None, type_priorities: dict[str, Priority] | None = None) -> None:
        """``type_priorities`` maps a job type to its default priority (used when ``submit`` gets no explicit ``priority``)."""
        with self._cond:
            if aging_seconds is not None:
                self.aging_seconds = float(aging_seconds)
            if elevated_factor is not None:
                self.elevated_factor = max(0.0, min(1.0, float(elevated_factor)))
            if type_priorities is not None:
                self._type_priority = {k: self._coerce_priority(v) for k, v in type_priorities.items()}
            self._kick_locked()

    def apply_resolved_limits(self, limits: Any) -> None:
        """Adopt a ``performance.settings.ResolvedLimits`` (worker counts and the thumbnail / proxy concurrency)."""
        self.set_limits(max_workers=limits.background_workers, foreground_workers=limits.foreground_workers)
        self.set_resource_limit("thumbnail", limits.thumbnail_concurrency)
        self.set_resource_limit("proxy", limits.proxy_concurrency)

    def set_pressure_provider(self, provider: Callable[[], dict] | None, *, poll_interval: float | None = None) -> None:
        """``provider()`` returns the ``ResourceMonitor.pressure()`` shape (``{"overall": "normal|elevated|critical", ...}``)."""
        with self._cond:
            self._provider = provider
            if poll_interval is not None:
                self._poll_interval = max(0.01, float(poll_interval))
            if provider is not None and not self._shutdown and (self._poll_thread is None or not self._poll_thread.is_alive()):
                self._poll_thread = threading.Thread(target=self._poll_loop, name="job-pressure", daemon=True)
                self._poll_thread.start()
        self.reevaluate()

    def reevaluate(self) -> str:
        """Re-read the pressure now and let workers re-check what may start. Returns the current level."""
        level = self._refresh_pressure(force=True)
        self.wake()
        return level

    def wake(self) -> None:
        """Make idle workers re-check the queue (e.g. after the clock moved past a retry delay)."""
        with self._cond:
            self._kick_locked()

    @staticmethod
    def _coerce_priority(p: Any) -> Priority:
        return Priority(max(0, min(_NCLS - 1, int(p))))

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
        priority: Priority | int | None = None,
        dedupe_key: str | None = None,
        resource: str = "",
        retries: int = 0,
        retry_delay: float = 1.0,
        owner: str | None = None,
    ) -> Job:
        with self._cond:
            if self._shutdown:
                raise JobError("The application is shutting down; no new jobs can be started.")
            prio = self._coerce_priority(priority) if priority is not None else self._type_priority.get(job_type, Priority.MEDIUM)
            if dedupe_key is not None:
                existing = self._dedupe.get(dedupe_key)
                if existing is not None and existing.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    existing.on_complete = _chain(existing.on_complete, on_complete)
                    existing.on_error = _chain(existing.on_error, on_error)
                    existing.on_cancel = _chain(existing.on_cancel, on_cancel)
                    promoted = prio < existing.priority and existing.status is JobStatus.QUEUED and self._reprioritize_locked(existing, prio)
                    self._coalesced += 1
                    profiler.incr("jobs.coalesced")
                    self._kick_locked()
                else:
                    existing = None
            else:
                existing = None
            if existing is None:
                job = Job(type=job_type, title=title, func=func, on_complete=on_complete, on_error=on_error, on_cancel=on_cancel, priority=prio, dedupe_key=dedupe_key,
                          resource=resource or "", owner=owner, retries=max(0, int(retries)), retry_delay=max(0.0, float(retry_delay)))
                job.message = "Queued"
                self._jobs[job.id] = job
                if dedupe_key is not None:
                    self._dedupe[dedupe_key] = job
        if existing is not None:
            if promoted:
                self._on_update(existing)
            return existing
        self._publish(Topics.JOB_ADDED, job)
        self._enqueue(job)
        return job

    def _enqueue(self, job: Job) -> None:
        notify: list[Job] = []
        with self._cond:
            if job.status is not JobStatus.QUEUED:
                return  # cancelled or paused before it was queued
            if self._shutdown:
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled (shutting down)"
                job.finished_at = _now()
                self._release_dedupe_locked(job)
                notify.append(job)
            else:
                self._enqueue_locked(job, 0.0)
                notify.extend(self._held_notice_locked(job))
        for j in notify:
            self._on_update(j)

    def _enqueue_locked(self, job: Job, delay: float) -> None:
        job.run_token += 1
        self._inflight += 1
        now = self._clock()
        self._seq += 1
        self._push_locked(_Entry(job, job.run_token, self._seq, now, int(job.priority), now + delay))
        self._kick_locked()

    def _push_locked(self, entry: _Entry) -> None:
        heapq.heappush(self._heaps[entry.cls], entry)
        self._live[entry.cls] += 1
        self._entries[entry.job.id] = entry
        profiler.gauge("jobs.queued", sum(self._live))

    def _drop_entry_locked(self, job_id: str, *, release: bool) -> _Entry | None:
        """Remove a queued job from the heap now (lazy-deleted inside the heap; counts and wait_idle see it gone immediately)."""
        e = self._entries.pop(job_id, None)
        if e is None:
            return None
        e.alive = False
        self._live[e.cls] -= 1
        self._dead[e.cls] += 1
        if self._dead[e.cls] > 256 and self._dead[e.cls] > self._live[e.cls]:
            heap = [x for x in self._heaps[e.cls] if x.alive]
            heapq.heapify(heap)
            self._heaps[e.cls], self._dead[e.cls] = heap, 0
        if release:
            self._inflight -= 1
        profiler.gauge("jobs.queued", sum(self._live))
        return e

    def _reprioritize_locked(self, job: Job, prio: Priority) -> bool:
        e = self._drop_entry_locked(job.id, release=False)
        job.priority = prio
        if e is not None:
            self._push_locked(_Entry(job, e.token, e.seq, e.enq, int(prio), e.not_before))
        return True

    def _release(self) -> None:
        with self._lock:
            self._inflight -= 1

    def _release_dedupe_locked(self, job: Job) -> None:
        if job.dedupe_key is not None and self._dedupe.get(job.dedupe_key) is job:
            del self._dedupe[job.dedupe_key]

    def _register_dedupe_locked(self, job: Job) -> None:
        if job.dedupe_key is not None and job.dedupe_key not in self._dedupe:
            self._dedupe[job.dedupe_key] = job

    # ----- scheduling core -----
    def _total_threads(self) -> int:
        return self._max_workers + self._fg_workers

    def _kick_locked(self) -> None:
        if not self._shutdown:
            for _ in range(max(0, min(sum(self._live) - self._idle, self._total_threads() - len(self._threads)))):  # threads are created on demand, never beyond the limit
                self._thread_seq += 1
                t = threading.Thread(target=self._worker_loop, name=f"job-{self._thread_seq}", daemon=True)
                self._threads.append(t)
                t.start()
        self._cond.notify_all()

    def _held(self, cls: int, level: str) -> bool:
        return (level == CRITICAL and cls >= Priority.MEDIUM) or (level == ELEVATED and cls == Priority.LOW)

    def _nonhigh_cap(self, level: str) -> int:
        if level == ELEVATED:
            return max(1, int(self._max_workers * self.elevated_factor))
        return self._max_workers

    def _note_throttle_locked(self, key: tuple, **fields: Any) -> None:
        if key in self._throttle_logged:
            return
        self._throttle_logged.add(key)
        self._throttled += 1
        profiler.incr("jobs.throttled")
        log_event(_log, "perf.worker_limit_reached", **fields)

    def _res_blocked(self, job: Job) -> bool:
        r = job.resource
        if not r:
            return False
        cap = self._res_limits.get(r)
        return cap is not None and self._res_running.get(r, 0) >= cap

    def _find_locked(self, cls: int, now: float) -> tuple[_Entry | None, float | None]:
        """Pop the oldest startable entry of a class (None if there is none) and the earliest time a delayed one becomes startable."""
        heap = self._heaps[cls]
        skipped: list[_Entry] = []
        found: _Entry | None = None
        wake: float | None = None
        while heap:
            e = heap[0]
            if not e.alive:
                heapq.heappop(heap)
                self._dead[cls] -= 1
                continue
            if e.not_before > now:
                wake = e.not_before if wake is None else min(wake, e.not_before)
                skipped.append(heapq.heappop(heap))
                continue
            if self._res_blocked(e.job):
                self._note_throttle_locked(("resource", e.job.resource), reason="resource_cap", resource=e.job.resource, limit=self._res_limits.get(e.job.resource))
                skipped.append(heapq.heappop(heap))
                continue
            found = heapq.heappop(heap)
            break
        for s in skipped:
            heapq.heappush(heap, s)
        return found, wake

    def _pick_locked(self) -> tuple[_Entry | None, float | None]:
        """Choose the next job to start (marks it RUNNING) or return (None, seconds until a delayed job is due)."""
        if len(self._running) >= self._total_threads():
            return None, None
        now = self._clock()
        level = self._pressure
        nonhigh_open = self._nonhigh_running < self._nonhigh_cap(level)
        found: dict[int, _Entry] = {}
        wake: float | None = None
        for cls in range(_NCLS):
            if not self._live[cls]:
                continue
            if cls >= Priority.MEDIUM and not nonhigh_open:
                if cls == Priority.MEDIUM and level == ELEVATED and self._nonhigh_cap(level) < self._max_workers:
                    self._note_throttle_locked(("pressure_cap", level), reason="pressure", level=level, priority="MEDIUM", cap=self._nonhigh_cap(level))
                continue
            if self._held(cls, level):
                self._note_throttle_locked(("pressure", level, cls), reason="pressure", level=level, priority=Priority(cls).name, queued=self._live[cls])
                continue
            e, w = self._find_locked(cls, now)
            if w is not None:
                wake = w - now if wake is None else min(wake, w - now)
            if e is not None:
                found[cls] = e
        if not found:
            return None, wake
        aging = self.aging_seconds

        def rank(e: _Entry) -> tuple[int, int]:
            aged = int((now - e.enq) / aging) if aging > 0 else 0
            return max(0, e.cls - aged), e.seq

        best = min(found.values(), key=rank)
        for e in found.values():
            if e is not best:
                heapq.heappush(self._heaps[e.cls], e)
        self._live[best.cls] -= 1
        del self._entries[best.job.id]
        job = best.job
        counts = best.cls != Priority.HIGH
        self._running[job.id] = (job, counts, job.resource)
        if counts:
            self._nonhigh_running += 1
        if job.resource:
            self._res_running[job.resource] = self._res_running.get(job.resource, 0) + 1
        job.status = JobStatus.RUNNING
        profiler.gauge("jobs.queued", sum(self._live))
        profiler.metrics.record("jobs.queue_wait", max(0.0, now - max(best.enq, best.not_before)))
        return best, None

    def _worker_loop(self) -> None:
        me = threading.current_thread()
        while True:
            with self._cond:
                while True:
                    if self._shutdown:
                        self._threads.remove(me)
                        return
                    entry, wake = self._pick_locked()
                    if entry is not None:
                        break
                    self._idle += 1
                    try:
                        signalled = self._cond.wait(timeout=wake if wake is not None else self.idle_timeout)
                    finally:
                        self._idle -= 1
                    if not signalled and wake is None and not self._shutdown:
                        self._threads.remove(me)  # idle for a long time: let the thread go, one is spawned when needed
                        return
            self._execute(entry)

    def _execute(self, entry: _Entry) -> None:
        job = entry.job
        t0 = time.perf_counter()
        try:
            run_job(job, self._on_update)
            elapsed = time.perf_counter() - t0
            ok = job.status is not JobStatus.FAILED
            profiler.metrics.record("jobs.run", elapsed, ok)
            profiler.metrics.record(f"jobs.run.{job.type}", elapsed, ok)
        finally:
            try:
                self._finish(entry)
            finally:
                self._release()
        self._refresh_pressure()

    def _finish(self, entry: _Entry) -> None:
        job = entry.job
        callback: Callable[[Job], None] | None = None
        with self._cond:
            _, counts, resource = self._running.pop(job.id)
            if counts:
                self._nonhigh_running -= 1
            if resource:
                self._res_running[resource] -= 1
                if self._res_running[resource] <= 0:
                    del self._res_running[resource]
                if not self._res_blocked(job):
                    self._throttle_logged.discard(("resource", resource))
            if job.retry_pending:
                job.retry_pending = False
                if job.status is JobStatus.RUNNING and not job.cancel_requested and not self._shutdown:
                    job.auto_retries_used += 1
                    self._retried += 1
                    profiler.incr("jobs.retried")
                    delay = min(self.max_retry_delay, job.retry_delay * (2 ** (job.auto_retries_used - 1)))
                    job.status = JobStatus.QUEUED
                    job.progress = 0.0
                    job.finished_at = None
                    job.message = f"Retrying ({job.auto_retries_used}/{job.retries})…"
                    self._enqueue_locked(job, delay)
                else:
                    job.status = JobStatus.CANCELLED
                    job.message = "Cancelled"
            if not job.status.is_terminal:
                self._kick_locked()
            else:
                self._release_dedupe_locked(job)
                if job.status is JobStatus.FAILED:
                    self._failed_recent.append({"id": job.id, "type": job.type, "error": job.error, "finished_at": job.finished_at, "attempts": job.attempts})
                callback = {JobStatus.COMPLETED: job.on_complete, JobStatus.FAILED: job.on_error, JobStatus.CANCELLED: job.on_cancel}.get(job.status)
                self._kick_locked()
        self._on_update(job)
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

    # ----- pressure -----
    def _refresh_pressure(self, force: bool = False) -> str:
        provider = self._provider
        if provider is None:
            return self._pressure
        now = time.monotonic()
        if not force and now - self._pressure_at < self.pressure_min_interval:
            return self._pressure
        self._pressure_at = now
        try:
            level = str(provider().get("overall", NORMAL)).lower()
            if level not in (NORMAL, ELEVATED, CRITICAL):
                level = NORMAL
            self._provider_failed = False
        except Exception:  # a broken provider must not stop the scheduler: behave as if nothing is constrained
            level = NORMAL
            if not self._provider_failed:
                self._provider_failed = True
                _log.warning("Job pressure provider failed; assuming normal pressure", exc_info=True)
        changed: list[Job] = []
        with self._cond:
            if level != self._pressure:
                log_event(_log, "jobs.pressure_changed", old=self._pressure, new=level)
                self._pressure = level
                self._throttle_logged = {k for k in self._throttle_logged if k[0] not in ("pressure", "pressure_cap")}
            for e in self._entries.values():
                held = self._held(e.cls, level)
                if held and e.job.message != WAIT_MESSAGE:
                    e.job.message = WAIT_MESSAGE
                    changed.append(e.job)
                elif not held and e.job.message == WAIT_MESSAGE:
                    e.job.message = "Queued"
                    changed.append(e.job)
            self._kick_locked()
        for j in changed:
            self._on_update(j)
        return level

    def _held_notice_locked(self, job: Job) -> list[Job]:
        if self._held(int(job.priority), self._pressure) and job.message != WAIT_MESSAGE:
            job.message = WAIT_MESSAGE
            return [job]
        return []

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            if self._provider is not None and (sum(self._live) or self._pressure != NORMAL):
                self._refresh_pressure(force=True)

    # ----- control -----
    def cancel(self, job_id: str) -> bool:
        with self._cond:
            job = self._require(job_id)
            if job.status.is_terminal:
                return False
            job.cancel_event.set()
            if job.status in (JobStatus.QUEUED, JobStatus.PAUSED):
                self._drop_entry_locked(job.id, release=True)
                self._release_dedupe_locked(job)
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

    def cancel_where(self, pred: Callable[[Job], bool] | None = None, *, dedupe_key: str | None = None, resource: str | None = None, job_type: str | None = None,
                     owner: str | None = None, priority: Priority | None = None, include_running: bool = True) -> list[str]:
        """Cancel every queued / paused (and, unless ``include_running=False``, running) job matching ALL given conditions. Returns the cancelled ids."""
        wanted = (JobStatus.QUEUED, JobStatus.PAUSED) + ((JobStatus.RUNNING,) if include_running else ())
        done: list[str] = []
        for job in self.jobs():
            if job.status not in wanted:
                continue
            if dedupe_key is not None and job.dedupe_key != dedupe_key or resource is not None and job.resource != resource:
                continue
            if job_type is not None and job.type != job_type or owner is not None and job.owner != owner or priority is not None and job.priority != priority:
                continue
            if pred is not None and not pred(job):
                continue
            if self.cancel(job.id):
                done.append(job.id)
        return done

    def cancel_owner(self, owner: str) -> list[str]:
        """Cancel everything tagged with ``owner`` (call with the old project id when a project is closed or switched)."""
        return self.cancel_where(owner=owner)

    def cancel_all_low(self) -> list[str]:
        return self.cancel_where(priority=Priority.LOW)

    def retry(self, job_id: str) -> Job:
        with self._cond:
            job = self._require(job_id)
            if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
                raise JobError("Only failed or cancelled jobs can be retried.")
            job.cancel_event.clear()
            job.status = JobStatus.QUEUED
            job.progress = 0.0
            job.error = None
            job.message = "Queued (retry)"
            job.started_at = job.finished_at = None
            job.auto_retries_used = 0
            self._register_dedupe_locked(job)
        self._on_update(job)
        self._enqueue(job)
        return job

    def pause(self, job_id: str) -> bool:
        """Pause a job that has not started yet. Running jobs cannot be paused in Phase 1."""
        with self._cond:
            job = self._require(job_id)
            if job.status is not JobStatus.QUEUED:
                return False
            self._drop_entry_locked(job.id, release=True)
            self._release_dedupe_locked(job)  # a paused job is not "in flight": an identical request starts a fresh job
            job.status = JobStatus.PAUSED
            job.message = "Paused"
        self._on_update(job)
        return True

    def resume(self, job_id: str) -> bool:
        with self._cond:
            job = self._require(job_id)
            if job.status is not JobStatus.PAUSED:
                return False
            job.status = JobStatus.QUEUED
            job.message = "Queued"
            self._register_dedupe_locked(job)
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

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued_by_priority": {p.name: self._live[int(p)] for p in Priority},
                "queued": sum(self._live),
                "running": len(self._running),
                "running_by_resource": dict(self._res_running),
                "workers": {"threads": len(self._threads), "idle": self._idle, "max_workers": self._max_workers, "foreground_workers": self._fg_workers},
                "limits": {"max_workers": self._max_workers, "foreground_workers": self._fg_workers, "resources": dict(self._res_limits), "aging_seconds": self.aging_seconds,
                           "elevated_factor": self.elevated_factor},
                "pressure": self._pressure,
                "failed_recent": list(self._failed_recent),
                "retried": self._retried,
                "coalesced": self._coalesced,
                "throttled": self._throttled,
            }

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

    def shutdown(self, cancel_running: bool = True, timeout: float = 2.0) -> None:
        """No new jobs; queued jobs are dropped (``on_cancel`` runs); running jobs are signalled (unless ``cancel_running=False``) and the worker threads are joined for up to ``timeout``."""
        with self._cond:
            self._shutdown = True
            self._stop.set()
            self._cond.notify_all()
        for job in self.active_jobs():
            if cancel_running or job.status is not JobStatus.RUNNING:
                self.cancel(job.id)
        with self._cond:
            self._cond.notify_all()
            threads = [t for t in self._threads if t is not threading.current_thread()]
            poll = self._poll_thread
        if poll is not None and poll is not threading.current_thread():
            threads.append(poll)
        deadline = time.monotonic() + max(0.0, timeout)
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))

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
