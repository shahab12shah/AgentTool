"""``PriorityWorkQueue``: a bounded, deduplicated, priority-ordered queue of small background tasks (thumbnails, frame prefetch, ...).

The queue does NOT create one ``Job`` per task. It owns the tasks itself and runs at most ``concurrency()`` *drain* jobs (resource ``thumbnail`` by default,
so the scheduler's resource cap and pressure rules apply too); each drain job takes the best task, runs it, and repeats until nothing startable is left.
Opening a project with 1,500 assets therefore creates a handful of jobs instead of 1,500, and a visible item that is requested later still jumps the queue
because the ordering is decided when a task is *taken*, not when it was queued.

Priorities (lower runs first): ``VISIBLE`` (on screen right now), ``REQUESTED`` (the user or an import asked for it), ``IDLE`` (speculative; held while idle work is
switched off or the machine is under pressure). A task that was only requested because it was visible is dropped again when it scrolls out of view.
"""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from app.core.exceptions import JobCancelled
from app.jobs.job import Job, JobStatus, Priority
from app.logging.logger import get_logger, log_event
from app.performance.profiler import profiler

_log = get_logger(__name__)
VISIBLE, REQUESTED, IDLE = 0, 1, 2
_MAX_FAILED = 5000
IDLE_SETTLE_S = 0.4
PRIORITY_CLASS = {Priority.HIGH: VISIBLE, Priority.MEDIUM: REQUESTED, Priority.LOW: IDLE}


def as_class(priority: Any) -> int:
    """Accept a scheduler ``Priority`` or one of this module's class numbers."""
    if isinstance(priority, Priority):
        return PRIORITY_CLASS[priority]
    return max(VISIBLE, min(IDLE, int(priority)))


@dataclass
class _Task:
    key: str
    payload: Any
    prio: int
    base: int | None  # the priority it was explicitly requested with (None: queued only because it is visible)
    seq: int
    version: int = 0
    running: bool = False
    visible: bool = False
    removed: bool = False


@dataclass
class _Slot:
    index: int
    gen: int
    key: str
    job: Job | None = None


@dataclass
class QueueStats:
    submitted: int = 0
    coalesced: int = 0
    started: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0
    max_running: int = 0
    jobs_created: int = 0
    batch_done: int = 0  # since the queue was last idle (for the aggregated message)
    batch_failed: int = 0
    batch_reasons: dict[str, int] = field(default_factory=dict)


class PriorityWorkQueue:
    def __init__(self, jobs, *, owner: str, run: Callable[[Any, Callable[[], bool]], Any], name: str = "thumbnails", job_type: str = "media.thumbnails", resource: str = "thumbnail",
                 concurrency: Callable[[], int] = lambda: 2, idle_allowed: Callable[[], bool] = lambda: True, pressure_ok: Callable[[], bool] = lambda: True,
                 on_done: Callable[[str, Any, Any], None] | None = None, on_failed: Callable[[str, Any, str], None] | None = None,
                 on_idle: Callable[[QueueStats], None] | None = None, title: str = "Preparing thumbnails", urgent_priority: Priority = Priority.MEDIUM) -> None:
        self._jobs, self.owner, self._run = jobs, owner, run
        self._urgent = urgent_priority  # scheduler priority of a drain job while VISIBLE / REQUESTED work is waiting (a frame queue uses HIGH)
        self.name, self._type, self._resource, self._title = name, job_type, resource, title
        self._concurrency, self._idle_allowed, self._pressure_ok = concurrency, idle_allowed, pressure_ok
        self._on_done, self._on_failed, self._on_idle = on_done, on_failed, on_idle
        self._lock = threading.RLock()
        self._tasks: dict[str, _Task] = {}
        self._heap: list[tuple[int, int, int, str]] = []  # (prio, seq, version, key); stale entries are skipped when popped
        self._counts = [0, 0, 0]  # waiting (not running) tasks per class
        self._failed: dict[str, str] = {}
        self._slots: dict[int, _Slot] = {}
        self._gen = 0
        self._seq = 0
        self._running = 0
        self._closed = False
        self._timer: threading.Timer | None = None
        self.stats = QueueStats()

    # ------------------------------------------------------------------ requests
    def submit(self, key: str, payload: Any, priority: Any = REQUESTED) -> bool:
        """Queue ``key``; a second request for the same key coalesces (and may raise its priority). Returns True when a new task was created."""
        return self.submit_many([(key, payload)], priority) > 0

    def submit_many(self, items: Iterable[tuple[str, Any]], priority: Any = REQUESTED) -> int:
        """Queue a batch under one lock and wake the workers once (so a fast worker cannot drain the queue between two items of the same batch). Returns the new tasks."""
        cls = as_class(priority)
        made = 0
        with self._lock:
            if self._closed:
                return 0
            for key, payload in items:
                made += self._add(key, payload, cls)
        self.kick()
        return made

    def _add(self, key: str, payload: Any, cls: int) -> int:
        t = self._tasks.get(key)
        if t is not None:
            self.stats.coalesced += 1
            profiler.incr(f"{self.name}.coalesced")
            t.base = cls if t.base is None else min(t.base, cls)
            self._retarget(t, min(t.prio, cls))
            return 0
        self._failed.pop(key, None)
        self._seq += 1
        t = _Task(key, payload, cls, cls, self._seq)
        self._tasks[key] = t
        self._counts[cls] += 1
        heapq.heappush(self._heap, (cls, t.seq, t.version, key))
        self.stats.submitted += 1
        return 1

    def set_visible(self, keys: Iterable[str], payload_for: Callable[[str], Any | None]) -> None:
        """Keys on screen now: they run first. Keys that were only queued for being visible and are no longer are dropped; the rest fall back to their own priority."""
        want = list(dict.fromkeys(keys))
        wanted = set(want)
        with self._lock:
            if self._closed:
                return
            for t in list(self._tasks.values()):
                if t.visible and t.key not in wanted:
                    t.visible = False
                    if t.running:
                        continue
                    if t.base is None:
                        self._remove(t)
                    else:
                        self._retarget(t, t.base)
            for k in want:
                t = self._tasks.get(k)
                if t is None:
                    if k in self._failed:
                        continue
                    payload = payload_for(k)
                    if payload is None:
                        continue
                    self._seq += 1
                    t = _Task(k, payload, VISIBLE, None, self._seq, visible=True)
                    self._tasks[k] = t
                    self._counts[VISIBLE] += 1
                    heapq.heappush(self._heap, (VISIBLE, t.seq, t.version, k))
                    self.stats.submitted += 1
                else:
                    t.visible = True
                    self._retarget(t, VISIBLE)
        self.kick()

    def cancel(self, keys: Iterable[str]) -> int:
        n = 0
        with self._lock:
            for k in keys:
                t = self._tasks.get(k)
                if t is not None:
                    self._remove(t)
                    n += 1
        self.stats.cancelled += n
        return n

    def cancel_all(self) -> int:
        with self._lock:
            n = len(self._tasks)
            for t in list(self._tasks.values()):
                self._remove(t)
        self.stats.cancelled += n
        return n

    def close(self) -> None:
        """Drop every task, cancel the drain jobs and refuse new work (project closed or switched)."""
        with self._lock:
            self._closed = True
            slots = list(self._slots.values())
            self._slots.clear()
            tm, self._timer = self._timer, None
        self.cancel_all()
        if tm is not None:
            tm.cancel()
        for s in slots:
            if s.job is not None:
                try:
                    self._jobs.cancel(s.job.id)
                except Exception:  # noqa: BLE001
                    pass

    def retry(self, key: str, payload: Any, priority: Any = REQUESTED) -> bool:
        with self._lock:
            self._failed.pop(key, None)
        return self.submit(key, payload, priority)

    def forget_failure(self, key: str) -> None:
        with self._lock:
            self._failed.pop(key, None)

    # ------------------------------------------------------------------ queries
    def state(self, key: str) -> str | None:
        with self._lock:
            if key in self._tasks:
                return "running" if self._tasks[key].running else "pending"
            return "failed" if key in self._failed else None

    def failure(self, key: str) -> str | None:
        with self._lock:
            return self._failed.get(key)

    def failed_keys(self) -> list[str]:
        with self._lock:
            return list(self._failed)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._tasks)

    def current_job(self) -> Job | None:
        with self._lock:
            return next((s.job for s in self._slots.values() if s.job is not None), None)

    @property
    def closed(self) -> bool:
        return self._closed

    def pending(self) -> int:
        with self._lock:
            return len(self._tasks)

    def waiting(self) -> int:
        with self._lock:
            return sum(self._counts)

    def priority_of(self, key: str) -> int | None:
        with self._lock:
            t = self._tasks.get(key)
            return t.prio if t else None

    # ------------------------------------------------------------------ internals (lock held)
    def _retarget(self, t: _Task, cls: int) -> None:
        if t.running or t.removed or cls == t.prio:
            return
        self._counts[t.prio] -= 1
        t.prio = cls
        self._counts[cls] += 1
        t.version += 1
        heapq.heappush(self._heap, (cls, t.seq, t.version, t.key))

    def _remove(self, t: _Task) -> None:
        if self._tasks.get(t.key) is t:
            del self._tasks[t.key]
        if not t.running and not t.removed:
            self._counts[t.prio] -= 1
        t.removed = True
        t.version += 1  # invalidates its heap entries

    def _startable(self) -> int:
        n = self._counts[VISIBLE] + self._counts[REQUESTED]
        if self._counts[IDLE] and self._idle_allowed() and self._pressure_ok():
            n += self._counts[IDLE]
        return n

    def _take(self, slot: _Slot) -> _Task | None:
        """The best startable task (marked running), or None -- in which case the caller's slot is released atomically so a later submit starts a fresh drain job."""
        with self._lock:
            allow_idle = self._idle_allowed()
            pressure_ok = self._pressure_ok()
            held_idle = False
            found: _Task | None = None
            while self._heap and not self._closed:
                entry = self._heap[0]
                prio, _seq, version, key = entry
                t = self._tasks.get(key)
                if t is None or t.removed or t.running or t.version != version:
                    heapq.heappop(self._heap)
                    continue
                if prio == IDLE and not (allow_idle and pressure_ok):
                    held_idle = True
                    break  # everything left is IDLE (heap order): nothing startable
                heapq.heappop(self._heap)
                found = t
                break
            if found is not None:
                found.running = True
                self._counts[found.prio] -= 1
                self._running += 1
                self.stats.started += 1
                self.stats.max_running = max(self.stats.max_running, self._running)
                return found
            if self._slots.get(slot.index) is slot:
                del self._slots[slot.index]
            if held_idle and allow_idle and not pressure_ok:
                self._schedule_recheck(3.0)
            return None

    def _schedule_recheck(self, delay: float) -> None:
        if self._timer is not None or self._closed:
            return

        def fire() -> None:
            with self._lock:
                self._timer = None
            self.kick()

        t = threading.Timer(delay, fire)
        t.daemon = True
        self._timer = t
        t.start()

    # ------------------------------------------------------------------ scheduling
    def kick(self) -> None:
        """Make sure enough drain jobs exist for the waiting work (also the way to resume held IDLE tasks after a setting changed)."""
        starts: list[_Slot] = []
        promote: list[_Slot] = []
        with self._lock:
            if self._closed:
                return
            startable = self._startable()
            if startable <= 0:
                return
            want = max(1, min(int(self._concurrency() or 1), startable + self._running))
            urgent = (self._counts[VISIBLE] + self._counts[REQUESTED]) > 0
            for i in range(want):
                s = self._slots.get(i)
                if s is None:
                    self._gen += 1
                    s = _Slot(i, self._gen, f"{self.name}:{self.owner}:{i}:{self._gen}")
                    self._slots[i] = s
                    starts.append(s)
                elif urgent and s.job is not None and s.job.priority > self._urgent and s.job.status is JobStatus.QUEUED:
                    promote.append(s)
        for s in starts + promote:
            self._submit_slot(s, self._urgent if urgent else Priority.LOW)

    def _submit_slot(self, slot: _Slot, prio: Priority) -> None:
        def func(ctx) -> int:
            return self._drain(slot, ctx)

        def ended(job: Job) -> None:
            with self._lock:
                if self._slots.get(slot.index) is slot:
                    del self._slots[slot.index]
            self.kick()

        try:
            job = self._jobs.submit(self._type, func, title=self._title, priority=prio, dedupe_key=slot.key, resource=self._resource, owner=self.owner,
                                    on_complete=ended, on_error=ended, on_cancel=ended)
        except Exception:  # noqa: BLE001 - the scheduler refuses new jobs while shutting down
            _log.debug("Could not start a %s drain job", self.name, exc_info=True)
            with self._lock:
                if self._slots.get(slot.index) is slot:
                    del self._slots[slot.index]
            return
        if slot.job is None:
            self.stats.jobs_created += 1
        slot.job = job

    def _drain(self, slot: _Slot, ctx) -> int:
        n = 0
        while True:
            ctx.check_cancelled()
            t = self._take(slot)
            if t is None:
                break
            n += 1
            self._execute(t, ctx)
            if self._idle_check():
                self._notify_idle()
        return n

    def _execute(self, t: _Task, ctx) -> None:
        cancelled = False
        failure: str | None = None
        result: Any = None
        try:
            result = self._run(t.payload, lambda: ctx.is_cancelled() or t.removed)
        except JobCancelled:
            cancelled = True
        except Exception as exc:  # noqa: BLE001 - one broken item must not stop the queue
            failure = (getattr(exc, "user_message", None) or str(exc) or exc.__class__.__name__)[:300]
            _log.debug("%s task failed: %s", self.name, failure, exc_info=True)
        with self._lock:
            self._running -= 1
            current = self._tasks.get(t.key) is t
            removed = t.removed or not current
            if current:
                del self._tasks[t.key]
            if cancelled and not removed and not self._closed:
                # interrupted by a job-level cancel (not by cancel()): put it back so the work is not lost
                t.running = False
                self._tasks[t.key] = t
                self._counts[t.prio] += 1
                t.version += 1
                heapq.heappush(self._heap, (t.prio, t.seq, t.version, t.key))
            elif failure is not None and not removed:
                self._failed[t.key] = failure
                if len(self._failed) > _MAX_FAILED:
                    self._failed.pop(next(iter(self._failed)))
                self.stats.failed += 1
                self.stats.batch_failed += 1
                self.stats.batch_reasons[failure] = self.stats.batch_reasons.get(failure, 0) + 1
            elif not cancelled and failure is None and not removed:
                self.stats.done += 1
                self.stats.batch_done += 1
        if cancelled:
            ctx.check_cancelled()
            return
        if removed:
            return
        if failure is not None:
            if self._on_failed:
                self._jobs.dispatch(lambda k=t.key, p=t.payload, f=failure: self._call(self._on_failed, k, p, f))
        elif self._on_done:
            self._jobs.dispatch(lambda k=t.key, p=t.payload, r=result: self._call(self._on_done, k, p, r))

    @staticmethod
    def _call(fn: Callable, *a: Any) -> None:
        try:
            fn(*a)
        except Exception:  # noqa: BLE001
            _log.debug("queue callback failed", exc_info=True)

    def _idle_check(self) -> bool:
        with self._lock:
            return not self._tasks and self._running == 0 and (self.stats.batch_done or self.stats.batch_failed) > 0

    def _notify_idle(self) -> None:
        """The batch looks finished: report it once the queue has stayed idle for a moment (a burst of requests that is faster than the worker must not produce several reports)."""
        with self._lock:
            if self._tasks or self._running:
                return
            mark = self.stats.started
        t = threading.Timer(IDLE_SETTLE_S, self._settle, args=(mark,))
        t.daemon = True
        t.start()

    def _settle(self, mark: int) -> None:
        with self._lock:
            if self._closed or self._tasks or self._running or self.stats.started != mark or not (self.stats.batch_done or self.stats.batch_failed):
                return
            snap = QueueStats(**{**self.stats.__dict__, "batch_reasons": dict(self.stats.batch_reasons)})
            self.stats.batch_done = self.stats.batch_failed = 0
            self.stats.batch_reasons = {}
        log_event(_log, f"{self.name}.batch_finished", done=snap.batch_done, failed=snap.batch_failed)
        if self._on_idle:
            self._jobs.dispatch(lambda s=snap: self._call(self._on_idle, s))
