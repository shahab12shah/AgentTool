from __future__ import annotations

import threading
import time

import pytest

from app.core.events import EventBus, Topics
from app.core.exceptions import AppError, JobError
from app.jobs.job import JobStatus
from app.jobs.job_manager import JobManager


@pytest.fixture
def jm():
    m = JobManager(EventBus(), max_workers=2)
    yield m
    m.shutdown()


def test_job_completes_with_progress_and_callback(jm):
    seen, done = [], []
    jm.bus = None
    jm._bus = EventBus()
    jm._bus.subscribe(Topics.JOB_UPDATED, lambda t, p: seen.append((p["job"].status.value, p["job"].progress)))

    def work(ctx):
        ctx.report(50, "half")
        return 42

    job = jm.submit("test", work, title="t", on_complete=done.append)
    assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED)
    assert jm.wait_idle(5)
    assert job.status is JobStatus.COMPLETED and job.result == 42 and job.progress == 100
    assert job.started_at and job.finished_at and job.error is None
    assert done == [job]
    assert any(s == "RUNNING" for s, _ in seen) and any(s == "COMPLETED" for s, _ in seen)


def test_job_runs_off_the_calling_thread(jm):
    names = []
    jm.submit("t", lambda ctx: names.append(threading.current_thread().name), title="t")
    assert jm.wait_idle(5)
    assert names and names[0] != threading.current_thread().name


def test_failed_job_records_error_and_calls_on_error(jm):
    errors = []

    def boom(ctx):
        raise AppError("Friendly message")

    job = jm.submit("t", boom, title="t", on_error=errors.append)
    assert jm.wait_idle(5)
    assert job.status is JobStatus.FAILED and job.error == "Friendly message" and errors == [job]


def test_unexpected_exception_is_contained(jm):
    job = jm.submit("t", lambda ctx: 1 / 0, title="t")
    assert jm.wait_idle(5)
    assert job.status is JobStatus.FAILED and "division" in job.error


def test_cancel_running_job(jm):
    started = threading.Event()
    cancelled = []

    def work(ctx):
        started.set()
        while True:
            ctx.check_cancelled()
            time.sleep(0.01)

    job = jm.submit("t", work, title="t", on_cancel=cancelled.append)
    assert started.wait(5)
    assert jm.cancel(job.id)
    assert jm.wait_idle(5)
    assert job.status is JobStatus.CANCELLED and cancelled == [job]
    assert not jm.cancel(job.id)  # already finished


def test_cancel_queued_job_never_runs(jm):
    gate = threading.Event()
    ran = []
    blockers = [jm.submit("b", lambda ctx: gate.wait(5), title="b") for _ in range(2)]
    queued = jm.submit("q", lambda ctx: ran.append(1), title="q")
    time.sleep(0.05)
    assert queued.status is JobStatus.QUEUED
    assert jm.cancel(queued.id)
    assert queued.status is JobStatus.CANCELLED
    gate.set()
    assert jm.wait_idle(5)
    assert ran == [] and queued.status is JobStatus.CANCELLED


def test_retry_failed_job_succeeds_second_time(jm):
    calls = []

    def flaky(ctx):
        calls.append(1)
        if len(calls) == 1:
            raise AppError("first try fails")
        return "ok"

    job = jm.submit("t", flaky, title="t")
    assert jm.wait_idle(5)
    assert job.status is JobStatus.FAILED
    jm.retry(job.id)
    assert jm.wait_idle(5)
    assert job.status is JobStatus.COMPLETED and job.result == "ok" and job.attempts == 2 and job.error is None


def test_retry_cancelled_queued_job_runs_exactly_once(jm):
    gate = threading.Event()
    ran = []
    for _ in range(2):
        jm.submit("b", lambda ctx: gate.wait(5), title="b")
    job = jm.submit("q", lambda ctx: ran.append(1), title="q")
    time.sleep(0.05)
    jm.cancel(job.id)
    jm.retry(job.id)
    gate.set()
    assert jm.wait_idle(5)
    assert ran == [1]


def test_retry_rejects_non_failed_jobs(jm):
    job = jm.submit("t", lambda ctx: 1, title="t")
    assert jm.wait_idle(5)
    with pytest.raises(JobError):
        jm.retry(job.id)


def test_pause_and_resume_queued_job(jm):
    gate = threading.Event()
    ran = []
    for _ in range(2):
        jm.submit("b", lambda ctx: gate.wait(5), title="b")
    job = jm.submit("q", lambda ctx: ran.append(1), title="q")
    time.sleep(0.05)
    assert jm.pause(job.id) and job.status is JobStatus.PAUSED
    gate.set()
    time.sleep(0.2)
    assert ran == [] and job.status is JobStatus.PAUSED
    assert jm.resume(job.id)
    assert jm.wait_idle(5) and ran == [1] and job.status is JobStatus.COMPLETED


def test_dispatcher_controls_callback_thread():
    marshalled = []
    m = JobManager(EventBus(), dispatcher=lambda fn: (marshalled.append(1), fn()))
    got = []
    m.submit("t", lambda ctx: 1, title="t", on_complete=got.append)
    assert m.wait_idle(5)
    assert marshalled == [1] and len(got) == 1
    m.shutdown()


def test_callback_exception_does_not_break_manager(jm):
    def bad(job):
        raise RuntimeError("callback bug")

    jm.submit("t", lambda ctx: 1, title="t", on_complete=bad)
    assert jm.wait_idle(5)
    ok = jm.submit("t", lambda ctx: 2, title="t")
    assert jm.wait_idle(5) and ok.status is JobStatus.COMPLETED
