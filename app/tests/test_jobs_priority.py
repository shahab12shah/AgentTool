"""Priority / resource-aware scheduling in app.jobs.JobManager. Deterministic: threading.Event gates and a fake clock, no timing assertions."""

from __future__ import annotations

import threading
import time

import pytest

from app.core.events import EventBus
from app.core.exceptions import AppError, JobError, RecoverableJobError
from app.jobs.job import JobStatus, Priority
from app.jobs.job_manager import WAIT_MESSAGE, JobManager
from app.performance.profiler import profiler


def wait_until(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.002)
    return cond()


def job_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("job-") and t.is_alive()]


class Gate:
    """A job body that records that it started and blocks until released."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, ctx):
        self.started.set()
        assert self.release.wait(10)
        return "done"


@pytest.fixture
def make():
    made: list[JobManager] = []

    def factory(**kw):
        kw.setdefault("max_workers", 1)
        kw.setdefault("foreground_workers", 0)
        m = JobManager(EventBus(), **kw)
        made.append(m)
        return m

    yield factory
    for m in made:
        m.shutdown(timeout=3)


def blocker(jm: JobManager, **kw):
    g = Gate()
    job = jm.submit("blocker", g, title="blocker", **kw)
    assert g.started.wait(5)
    return g, job


def recorder(order: list, name: str):
    return lambda ctx: order.append(name)


def test_defaults_are_backward_compatible(make):
    jm = make(max_workers=3, foreground_workers=1)
    job = jm.submit("t", lambda ctx: 5, title="t")
    assert jm.wait_idle(5)
    assert job.priority is Priority.MEDIUM and job.dedupe_key is None and job.resource == "" and job.status is JobStatus.COMPLETED
    d = job.to_dict()
    assert d["priority"] == "MEDIUM" and d["dedupe_key"] is None and d["resource"] == "" and d["status"] == "COMPLETED"


def test_priority_order_then_fifo(make):
    jm = make()
    g, _ = blocker(jm)
    order: list[str] = []
    jm.submit("t", recorder(order, "low"), title="l", priority=Priority.LOW)
    jm.submit("t", recorder(order, "med1"), title="m1")
    jm.submit("t", recorder(order, "high"), title="h", priority=Priority.HIGH)
    jm.submit("t", recorder(order, "med2"), title="m2", priority=Priority.MEDIUM)
    jm.submit("t", recorder(order, "high2"), title="h2", priority=Priority.HIGH)
    assert jm.stats()["queued_by_priority"] == {"HIGH": 2, "MEDIUM": 2, "LOW": 1}
    g.release.set()
    assert jm.wait_idle(5)
    # one slot, HIGH may use it when free: strict priority, FIFO inside a class
    assert order == ["high", "high2", "med1", "med2", "low"]


def test_type_priority_policy_sets_default(make):
    jm = make()
    jm.set_priority_policy(type_priorities={"thumbnail": Priority.LOW})
    g, _ = blocker(jm)
    order: list[str] = []
    jm.submit("thumbnail", recorder(order, "thumb"), title="t")
    jm.submit("other", recorder(order, "other"), title="o")
    explicit = jm.submit("thumbnail", recorder(order, "forced"), title="f", priority=Priority.HIGH)
    assert explicit.priority is Priority.HIGH
    g.release.set()
    assert jm.wait_idle(5)
    assert order == ["forced", "other", "thumb"]


def test_foreground_slot_is_reserved_for_high_priority(make):
    jm = make(max_workers=2, foreground_workers=1)
    gates = [blocker(jm, priority=Priority.LOW)[0] for _ in range(2)]
    flood = [jm.submit("thumb", lambda ctx: time.sleep(0), title="x", priority=Priority.LOW) for _ in range(40)]
    flood += [jm.submit("proxy", lambda ctx: None, title="y") for _ in range(10)]
    assert jm.stats()["running"] == 2
    hi = Gate()
    job = jm.submit("preview", hi, title="preview", priority=Priority.HIGH)
    assert hi.started.wait(5), "a HIGH job must start although every normal slot is busy"
    assert job.status is JobStatus.RUNNING and jm.stats()["running"] == 3
    assert all(j.status is JobStatus.QUEUED for j in flood)  # the flood never got the foreground slot
    hi.release.set()
    for g in gates:
        g.release.set()
    assert jm.wait_idle(10)
    assert all(j.status is JobStatus.COMPLETED for j in flood)


def test_aging_prevents_starvation(make):
    now = [0.0]
    jm = make(aging_seconds=10.0, clock=lambda: now[0])
    g, _ = blocker(jm)
    order: list[str] = []
    jm.submit("t", recorder(order, "old_low"), title="l", priority=Priority.LOW)
    now[0] = 25.0  # waited 2.5 intervals: ranked like HIGH, and it is older than the newcomers
    jm.submit("t", recorder(order, "new_med"), title="m")
    jm.submit("t", recorder(order, "new_high"), title="h", priority=Priority.HIGH)
    g.release.set()
    assert jm.wait_idle(5)
    assert order == ["old_low", "new_high", "new_med"]


def test_without_enough_waiting_low_stays_behind(make):
    now = [0.0]
    jm = make(aging_seconds=100.0, clock=lambda: now[0])
    g, _ = blocker(jm)
    order: list[str] = []
    jm.submit("t", recorder(order, "low"), title="l", priority=Priority.LOW)
    now[0] = 5.0
    jm.submit("t", recorder(order, "med"), title="m")
    g.release.set()
    assert jm.wait_idle(5)
    assert order == ["med", "low"]


def test_aged_low_job_still_cannot_use_the_foreground_slot(make):
    now = [0.0]
    jm = make(max_workers=1, foreground_workers=1, aging_seconds=1.0, clock=lambda: now[0])
    g, _ = blocker(jm, priority=Priority.LOW)
    ran = threading.Event()
    jm.submit("t", lambda ctx: ran.set(), title="aged", priority=Priority.LOW)
    now[0] = 1000.0
    jm.wake()
    assert not ran.wait(0.1) and jm.stats()["running"] == 1
    g.release.set()
    assert jm.wait_idle(5) and ran.is_set()


def test_resource_cap_limits_concurrency_but_not_other_jobs(make):
    jm = make(max_workers=4, foreground_workers=0)
    jm.set_resource_limit("proxy", 1)
    lock = threading.Lock()
    cur = peak = 0

    def proxy(ctx):
        nonlocal cur, peak
        with lock:
            cur += 1
            peak = max(peak, cur)
        gate.wait(10)
        with lock:
            cur -= 1

    gate = threading.Event()
    proxies = [jm.submit("proxy", proxy, title="p", resource="proxy", priority=Priority.LOW) for _ in range(4)]
    assert wait_until(lambda: cur == 1)
    free = Gate()
    jm.submit("other", free, title="o")  # no resource: uncapped, runs next to the proxy job
    assert free.started.wait(5)
    st = jm.stats()
    assert st["running_by_resource"] == {"proxy": 1} and st["running"] == 2 and st["limits"]["resources"] == {"proxy": 1}
    assert sum(1 for j in proxies if j.status is JobStatus.QUEUED) == 3
    free.release.set()
    gate.set()
    assert jm.wait_idle(10)
    assert peak == 1 and all(j.status is JobStatus.COMPLETED for j in proxies)


def test_resource_limit_can_be_raised_at_runtime(make):
    jm = make(max_workers=4, foreground_workers=0)
    jm.set_resource_limit("thumbnail", 1)
    gates = [Gate() for _ in range(3)]
    for g in gates:
        jm.submit("th", g, title="t", resource="thumbnail")
    assert gates[0].started.wait(5) and not gates[1].started.wait(0.05)
    jm.set_resource_limit("thumbnail", 3)
    assert gates[1].started.wait(5) and gates[2].started.wait(5)
    jm.set_resource_limit("thumbnail", None)
    for g in gates:
        g.release.set()
    assert jm.wait_idle(5)


def test_dedupe_returns_existing_job_and_chains_callbacks(make):
    jm = make()
    g, _ = blocker(jm)
    calls: list[str] = []
    first = jm.submit("t", lambda ctx: 1, title="a", dedupe_key="k", on_complete=lambda j: calls.append("c1"), on_cancel=lambda j: calls.append("x1"))
    second = jm.submit("t", lambda ctx: 2, title="b", dedupe_key="k", on_complete=lambda j: calls.append("c2"), on_error=lambda j: calls.append("e2"))

    def bad(job):
        raise RuntimeError("callback bug")

    third = jm.submit("t", lambda ctx: 3, title="c", dedupe_key="k", on_complete=bad)
    fourth = jm.submit("t", lambda ctx: 3, title="c", dedupe_key="k", on_complete=lambda j: calls.append("c4"))
    assert first is second is third is fourth
    assert jm.stats()["coalesced"] == 3 and jm.stats()["queued"] == 1
    g.release.set()
    assert jm.wait_idle(5)
    assert first.result == 1 and calls == ["c1", "c2", "c4"]  # the first function ran once; every callback ran, in submit order, despite one failing
    fresh = jm.submit("t", lambda ctx: 9, title="again", dedupe_key="k")
    assert fresh is not first
    assert jm.wait_idle(5) and fresh.result == 9


def test_dedupe_coalesces_onto_a_running_job(make):
    jm = make()
    g = Gate()
    first = jm.submit("t", g, title="a", dedupe_key="k")
    assert g.started.wait(5)
    seen = []
    again = jm.submit("t", lambda ctx: None, title="a", dedupe_key="k", on_complete=seen.append)
    assert again is first
    g.release.set()
    assert jm.wait_idle(5) and seen == [first]


def test_dedupe_promotes_to_the_higher_priority(make):
    jm = make(max_workers=1, foreground_workers=1)
    g, _ = blocker(jm, priority=Priority.LOW)
    low = jm.submit("t", lambda ctx: "x", title="a", dedupe_key="k", priority=Priority.LOW)
    other = jm.submit("t", lambda ctx: "y", title="o", priority=Priority.MEDIUM)
    assert jm.stats()["queued_by_priority"] == {"HIGH": 0, "MEDIUM": 1, "LOW": 1}
    promoted = jm.submit("t", lambda ctx: None, title="a", dedupe_key="k", priority=Priority.HIGH)
    assert promoted is low and low.priority is Priority.HIGH
    assert jm.stats()["queued_by_priority"] == {"HIGH": 1, "MEDIUM": 1, "LOW": 0}
    assert wait_until(lambda: low.status is JobStatus.COMPLETED)  # runs on the foreground slot while the normal slot is still busy
    assert other.status is JobStatus.QUEUED
    # a lower priority submit never demotes
    again = jm.submit("t", lambda ctx: None, title="a", dedupe_key="k2", priority=Priority.MEDIUM)
    assert jm.submit("t", lambda ctx: None, title="a", dedupe_key="k2", priority=Priority.LOW) is again and again.priority is Priority.MEDIUM
    g.release.set()
    assert jm.wait_idle(5)


def test_dedupe_key_is_released_by_cancel_and_pause(make):
    jm = make()
    g, _ = blocker(jm)
    a = jm.submit("t", lambda ctx: 1, title="a", dedupe_key="k")
    assert jm.cancel(a.id)
    b = jm.submit("t", lambda ctx: 2, title="b", dedupe_key="k")
    assert b is not a
    assert jm.pause(b.id)
    c = jm.submit("t", lambda ctx: 3, title="c", dedupe_key="k")
    assert c is not b
    g.release.set()
    assert jm.wait_idle(5) and c.result == 3 and a.status is JobStatus.CANCELLED and b.status is JobStatus.PAUSED


class FakePressure:
    def __init__(self) -> None:
        self.level = "normal"

    def __call__(self):
        return {"memory": self.level, "disk": "normal", "cpu": "normal", "overall": self.level}


def test_elevated_pressure_holds_low_and_reduces_medium(make):
    jm = make(max_workers=4, foreground_workers=0, elevated_factor=0.5)
    p = FakePressure()
    p.level = "elevated"
    jm.set_pressure_provider(p)
    gates = [Gate() for _ in range(4)]
    for g in gates:
        jm.submit("m", g, title="m")
    low = jm.submit("l", lambda ctx: None, title="low", priority=Priority.LOW)
    assert gates[0].started.wait(5) and gates[1].started.wait(5)
    assert not gates[2].started.wait(0.1) and jm.stats()["running"] == 2  # int(4 * 0.5)
    assert low.status is JobStatus.QUEUED and low.message == WAIT_MESSAGE
    p.level = "normal"
    jm.reevaluate()
    assert gates[2].started.wait(5) and gates[3].started.wait(5)
    for g in gates:
        g.release.set()
    assert jm.wait_idle(5) and low.status is JobStatus.COMPLETED and low.message != WAIT_MESSAGE


def test_critical_pressure_only_high_runs_and_jobs_resume_automatically(make):
    jm = make(max_workers=2, foreground_workers=1)
    p = FakePressure()
    running = Gate()
    keep = jm.submit("pre", running, title="already running")
    assert running.started.wait(5)
    p.level = "critical"
    jm.set_pressure_provider(p, poll_interval=0.02)
    med = jm.submit("m", lambda ctx: "m", title="med")
    low = jm.submit("l", lambda ctx: "l", title="low", priority=Priority.LOW)
    high = jm.submit("h", lambda ctx: "h", title="high", priority=Priority.HIGH)
    assert wait_until(lambda: high.status is JobStatus.COMPLETED)
    assert med.status is JobStatus.QUEUED and low.status is JobStatus.QUEUED
    assert med.message == WAIT_MESSAGE and low.message == WAIT_MESSAGE
    assert keep.status is JobStatus.RUNNING  # throttling never touches running work
    running.release.set()
    assert wait_until(lambda: keep.status is JobStatus.COMPLETED)
    assert med.status is JobStatus.QUEUED  # still held: the pressure has not changed
    p.level = "normal"  # no explicit call: the poll thread notices
    assert jm.wait_idle(10) and wait_until(lambda: med.status is JobStatus.COMPLETED and low.status is JobStatus.COMPLETED)
    assert med.result == "m" and med.message != WAIT_MESSAGE


def test_throttle_decisions_are_counted_once_per_episode(make):
    jm = make(max_workers=2, foreground_workers=0)
    p = FakePressure()
    p.level = "critical"
    jm.set_pressure_provider(p)
    before = profiler.metrics.counter("jobs.throttled")
    jobs = [jm.submit("m", lambda ctx: None, title="m") for _ in range(5)]
    assert wait_until(lambda: jm.stats()["throttled"] == 1)
    for _ in range(5):
        jm.reevaluate()
    assert not wait_until(lambda: jm.stats()["throttled"] > 1, 0.1) and profiler.metrics.counter("jobs.throttled") == before + 1
    p.level = "normal"
    jm.reevaluate()
    assert jm.wait_idle(5) and all(j.status is JobStatus.COMPLETED for j in jobs)
    p.level = "critical"
    jm.reevaluate()
    jm.submit("m", lambda ctx: None, title="m")
    jm.reevaluate()
    assert wait_until(lambda: jm.stats()["throttled"] == 2)  # a new episode logs again
    p.level = "normal"
    jm.reevaluate()
    assert jm.wait_idle(5)


def test_broken_pressure_provider_is_treated_as_normal(make):
    jm = make()

    def boom():
        raise RuntimeError("monitor down")

    jm.set_pressure_provider(boom)
    job = jm.submit("m", lambda ctx: 1, title="m")
    assert jm.wait_idle(5) and job.status is JobStatus.COMPLETED


def test_recoverable_failure_is_retried_with_attempts_counted(make):
    jm = make()
    calls = []

    def flaky(ctx):
        calls.append(1)
        if len(calls) < 3:
            raise RecoverableJobError("file busy")
        return "ok"

    done = []
    job = jm.submit("t", flaky, title="t", retries=3, retry_delay=0.0, on_complete=done.append)
    assert jm.wait_idle(5)
    assert job.status is JobStatus.COMPLETED and job.result == "ok" and job.attempts == 3 and done == [job] and jm.stats()["retried"] == 2


def test_retries_are_bounded_and_failure_reported_once(make):
    jm = make()
    errors = []

    def always(ctx):
        raise RecoverableJobError("still busy")

    job = jm.submit("t", always, title="t", retries=2, retry_delay=0.0, on_error=errors.append)
    assert jm.wait_idle(5)
    assert job.status is JobStatus.FAILED and job.attempts == 3 and job.error == "still busy" and errors == [job]
    assert jm.stats()["failed_recent"][-1]["id"] == job.id


def test_ordinary_errors_and_default_submits_never_retry(make):
    jm = make()
    runs = []

    def bad(ctx):
        runs.append(1)
        raise AppError("nope")

    a = jm.submit("t", bad, title="t", retries=5, retry_delay=0.0)
    b = jm.submit("t", lambda ctx: 1 / 0, title="t", retries=5, retry_delay=0.0)
    c = jm.submit("t", lambda ctx: (_ for _ in ()).throw(RecoverableJobError("x")), title="t")  # retries=0
    assert jm.wait_idle(5)
    assert a.attempts == 1 and a.status is JobStatus.FAILED and b.attempts == 1 and b.status is JobStatus.FAILED and c.attempts == 1 and c.status is JobStatus.FAILED
    assert runs == [1]


def test_retry_backoff_grows_and_waits_for_the_clock(make):
    now = [0.0]
    jm = make(clock=lambda: now[0])
    attempts = []

    def flaky(ctx):
        attempts.append(now[0])
        if len(attempts) < 3:
            raise RecoverableJobError("busy")

    job = jm.submit("t", flaky, title="t", retries=2, retry_delay=10.0)
    assert wait_until(lambda: len(attempts) == 1 and job.status is JobStatus.QUEUED and job.message.startswith("Retrying (1/2)"))
    now[0] = 9.0
    jm.wake()
    assert not wait_until(lambda: len(attempts) > 1, 0.1)  # not due yet
    now[0] = 10.0
    jm.wake()
    assert wait_until(lambda: len(attempts) == 2 and job.status is JobStatus.QUEUED)
    now[0] = 29.0  # second delay is 20 s after the second failure at t=10
    jm.wake()
    assert not wait_until(lambda: len(attempts) > 2, 0.1)
    now[0] = 30.0
    jm.wake()
    assert jm.wait_idle(5) and job.status is JobStatus.COMPLETED and job.attempts == 3


def test_cancel_wins_over_retry(make):
    now = [0.0]
    jm = make(clock=lambda: now[0])
    cancelled = []
    n = []

    def fails(ctx):
        n.append(1)
        raise RecoverableJobError("busy")

    job = jm.submit("t", fails, title="t", retries=5, retry_delay=1000.0, on_cancel=cancelled.append)
    assert wait_until(lambda: len(n) == 1 and job.status is JobStatus.QUEUED)
    assert jm.cancel(job.id)  # waiting for its backoff: removed at once
    assert job.status is JobStatus.CANCELLED and jm.stats()["queued"] == 0
    assert jm.wait_idle(5) and cancelled == [job] and len(n) == 1

    # cancelled while running and failing recoverably: CANCELLED, not retried
    jm2 = make()
    started = threading.Event()
    go = threading.Event()
    m = []

    def fails2(ctx):
        m.append(1)
        started.set()
        go.wait(5)
        raise RecoverableJobError("busy")

    job2 = jm2.submit("t", fails2, title="t", retries=5, retry_delay=0.0)
    assert started.wait(5)
    jm2.cancel(job2.id)
    go.set()
    assert jm2.wait_idle(5) and job2.status is JobStatus.CANCELLED and len(m) == 1


def test_manual_retry_still_works_after_automatic_retries_are_used_up(make):
    jm = make()
    state = {"fail": True}

    def f(ctx):
        if state["fail"]:
            raise RecoverableJobError("busy")
        return "fine"

    job = jm.submit("t", f, title="t", retries=1, retry_delay=0.0)
    assert jm.wait_idle(5) and job.status is JobStatus.FAILED and job.attempts == 2
    state["fail"] = False
    jm.retry(job.id)
    assert jm.wait_idle(5) and job.status is JobStatus.COMPLETED and job.attempts == 3


def test_cancel_queued_job_leaves_the_queue_immediately(make):
    jm = make()
    g, _ = blocker(jm)  # every slot is busy: no worker can be woken to discard anything
    cancelled = []
    jobs = [jm.submit("t", lambda ctx: None, title="q", on_cancel=cancelled.append) for _ in range(5)]
    assert jm.stats()["queued"] == 5
    assert jm.cancel(jobs[2].id)
    st = jm.stats()
    assert st["queued"] == 4 and jobs[2].status is JobStatus.CANCELLED and cancelled == [jobs[2]]
    g.release.set()
    assert jm.wait_idle(5)
    assert [j.status for j in jobs].count(JobStatus.COMPLETED) == 4


def test_cancel_where_filters(make):
    jm = make()
    g, _ = blocker(jm)
    a = jm.submit("proxy", lambda ctx: None, title="a", resource="proxy", dedupe_key="pa", owner="p1")
    b = jm.submit("proxy", lambda ctx: None, title="b", resource="proxy", owner="p2")
    c = jm.submit("thumb", lambda ctx: None, title="c", owner="p1", priority=Priority.LOW)
    d = jm.submit("other", lambda ctx: None, title="d")
    assert jm.cancel_where(dedupe_key="pa") == [a.id]
    assert jm.cancel_where(resource="proxy") == [b.id]
    assert jm.cancel_where(lambda j: j.title == "d") == [d.id]
    assert jm.cancel_where(job_type="nothing") == []
    assert jm.cancel_where(owner="p1") == [c.id]
    assert all(j.status is JobStatus.CANCELLED for j in (a, b, c, d)) and jm.stats()["queued"] == 0
    g.release.set()
    assert jm.wait_idle(5)


def test_cancel_owner_and_all_low_signal_running_jobs_too(make):
    jm = make(max_workers=3, foreground_workers=0)
    started = [threading.Event() for _ in range(3)]

    def coop(i):
        def run(ctx):
            started[i].set()
            while True:
                ctx.check_cancelled()
                time.sleep(0.005)

        return run

    old = jm.submit("old", coop(0), title="old project", owner="project-1")
    keep = jm.submit("keep", coop(1), title="other project", owner="project-2")
    low = jm.submit("thumb", coop(2), title="low", priority=Priority.LOW)
    for e in started:
        assert e.wait(5)
    queued_old = jm.submit("old", lambda ctx: None, title="queued old", owner="project-1")  # all slots are busy: stays queued
    assert set(jm.cancel_owner("project-1")) == {old.id, queued_old.id}
    assert wait_until(lambda: old.status is JobStatus.CANCELLED) and queued_old.status is JobStatus.CANCELLED
    assert keep.status is JobStatus.RUNNING
    assert jm.cancel_all_low() == [low.id]
    assert wait_until(lambda: low.status is JobStatus.CANCELLED)
    jm.cancel(keep.id)
    assert jm.wait_idle(5)


def test_shutdown_drops_queued_stops_running_and_leaks_no_threads():
    base = set(job_threads())
    jm = JobManager(EventBus(), max_workers=1, foreground_workers=1)
    p = FakePressure()
    jm.set_pressure_provider(p, poll_interval=0.02)
    started = threading.Event()
    cancelled = []

    def coop(ctx):
        started.set()
        while True:
            ctx.check_cancelled()
            time.sleep(0.005)

    running = jm.submit("r", coop, title="r", on_cancel=cancelled.append)
    queued = [jm.submit("q", lambda ctx: None, title="q", on_cancel=cancelled.append) for _ in range(3)]
    assert started.wait(5)
    jm.shutdown(timeout=5)
    assert running.status is JobStatus.CANCELLED
    assert all(j.status is JobStatus.CANCELLED for j in queued)
    assert wait_until(lambda: set(job_threads()) <= base), "worker or poll threads leaked"
    assert len(cancelled) == 4
    with pytest.raises(JobError):
        jm.submit("t", lambda ctx: None, title="late")
    jm.shutdown()  # idempotent


def test_idle_workers_are_not_created_without_work_and_bounded_by_limits(make):
    base = set(job_threads())
    jm = make(max_workers=2, foreground_workers=1)
    assert set(job_threads()) == base
    gates = [Gate() for _ in range(6)]
    for g in gates:
        jm.submit("t", g, title="t")
    assert wait_until(lambda: jm.stats()["running"] == 2)
    assert len(set(job_threads()) - base) <= 3
    for g in gates:
        g.release.set()
    assert jm.wait_idle(5)


def test_set_limits_takes_effect_without_killing_running_jobs(make):
    jm = make(max_workers=3, foreground_workers=0)
    gates = [Gate() for _ in range(6)]
    for g in gates:
        jm.submit("t", g, title="t")
    assert wait_until(lambda: jm.stats()["running"] == 3)
    jm.set_limits(max_workers=1)
    assert jm.stats()["running"] == 3  # running jobs untouched
    gates[0].release.set()
    gates[1].release.set()
    assert wait_until(lambda: jm.stats()["running"] == 1 and sum(1 for g in gates if g.started.is_set()) == 3)  # nothing new starts while above the new cap
    jm.set_limits(max_workers=4, foreground_workers=1)
    assert wait_until(lambda: all(g.started.is_set() for g in gates))
    assert jm.stats()["limits"]["max_workers"] == 4
    for g in gates:
        g.release.set()
    assert jm.wait_idle(5)


def test_concurrency_never_exceeds_limits(make):
    jm = make(max_workers=3, foreground_workers=1)
    lock = threading.Lock()
    state = {"cur": 0, "peak": 0, "cur_nonhigh": 0, "peak_nonhigh": 0}
    barrier = threading.Barrier(3, timeout=10)

    def work(high: bool):
        def run(ctx):
            with lock:
                state["cur"] += 1
                state["peak"] = max(state["peak"], state["cur"])
                if not high:
                    state["cur_nonhigh"] += 1
                    state["peak_nonhigh"] = max(state["peak_nonhigh"], state["cur_nonhigh"])
            try:
                barrier.wait()  # three jobs must be running at the same moment to pass; a fourth would break the barrier cycle
            finally:
                with lock:
                    state["cur"] -= 1
                    if not high:
                        state["cur_nonhigh"] -= 1

        return run

    jobs = [jm.submit("m", work(False), title="m") for _ in range(9)]
    assert jm.wait_idle(15)
    assert all(j.status is JobStatus.COMPLETED for j in jobs)
    assert state["peak_nonhigh"] == 3 and state["peak"] <= 4


def test_two_thousand_trivial_jobs_complete_with_bounded_threads(make):
    base = set(job_threads())
    jm = make(max_workers=3, foreground_workers=1)
    seen = []
    done = []

    def trivial(ctx):
        if len(seen) < 50:
            seen.append(len(set(job_threads()) - base))
        done.append(1)

    prios = list(Priority)
    jobs = [jm.submit("t", trivial, title="t", priority=prios[i % 3]) for i in range(2000)]
    assert jm.wait_idle(60)
    assert len(done) == 2000 and all(j.status is JobStatus.COMPLETED for j in jobs)
    assert max(seen) <= 4 and jm.stats()["queued"] == 0 and jm.stats()["running"] == 0


def test_many_cancelled_jobs_do_not_leave_the_queue_inflated(make):
    jm = make()
    g, _ = blocker(jm)
    jobs = [jm.submit("t", lambda ctx: None, title="t") for _ in range(1000)]
    for j in jobs[:900]:
        jm.cancel(j.id)
    assert jm.stats()["queued"] == 100
    g.release.set()
    assert jm.wait_idle(10)
    assert sum(1 for j in jobs if j.status is JobStatus.COMPLETED) == 100


def test_stats_shape_and_failed_recent_is_bounded(make):
    jm = make(max_workers=2, foreground_workers=0)
    jm.set_resource_limit("proxy", 1)
    g = Gate()
    jm.submit("p", g, title="p", resource="proxy", priority=Priority.LOW)
    assert g.started.wait(5)
    st = jm.stats()
    assert set(st) >= {"queued_by_priority", "running", "running_by_resource", "workers", "limits", "failed_recent", "retried", "coalesced", "throttled"}
    assert st["running"] == 1 and st["running_by_resource"] == {"proxy": 1} and st["workers"]["max_workers"] == 2
    g.release.set()
    for i in range(60):
        jm.submit("bad", lambda ctx: (_ for _ in ()).throw(AppError("x")), title="bad")
    assert jm.wait_idle(10)
    fr = jm.stats()["failed_recent"]
    assert len(fr) == 50 and all(f["type"] == "bad" and f["error"] == "x" for f in fr)


def test_profiler_records_queue_wait_run_time_and_queue_gauge(make):
    jm = make()
    profiler.metrics.reset()
    g, _ = blocker(jm)
    jm.submit("special_type", lambda ctx: None, title="x")
    assert profiler.snapshot()["gauges"]["jobs.queued"] == 1
    g.release.set()
    assert jm.wait_idle(5)
    snap = profiler.snapshot()
    assert snap["operations"]["jobs.queue_wait"]["count"] == 2
    assert snap["operations"]["jobs.run.special_type"]["count"] == 1 and snap["operations"]["jobs.run"]["count"] == 2
    assert snap["gauges"]["jobs.queued"] == 0


def test_pause_resume_cancel_semantics_with_priorities(make):
    jm = make()
    g, _ = blocker(jm)
    ran = []
    hi = jm.submit("t", lambda ctx: ran.append("hi"), title="hi", priority=Priority.HIGH)
    lo = jm.submit("t", lambda ctx: ran.append("lo"), title="lo", priority=Priority.LOW)
    assert jm.pause(hi.id) and jm.stats()["queued_by_priority"]["HIGH"] == 0
    g.release.set()
    assert wait_until(lambda: ran == ["lo"] and lo.status is JobStatus.COMPLETED)
    assert hi.status is JobStatus.PAUSED and jm.wait_idle(5)
    assert jm.resume(hi.id) and jm.wait_idle(5) and ran == ["lo", "hi"]
