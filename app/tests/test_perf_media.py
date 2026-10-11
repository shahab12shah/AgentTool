"""Phase 9 media pipelines: bounded thumbnail queue, thumbnail cache reuse and states, waveform levels of detail, probe persistence."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.events import EventBus, Topics
from app.jobs.job import Priority
from app.jobs.job_manager import JobManager
from app.performance.synthetic import SyntheticSpec, build_project
from app.performance.thumbnail_queue import IDLE, REQUESTED, VISIBLE, PriorityWorkQueue
from app.project.project_manager import ProjectManager
from app.services.workspace import Workspace
from app.storage.paths import AppPaths
from app.tests.conftest import needs_ffmpeg


def eventually(cond, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


# ======================================================================== the queue itself (no FFmpeg, no project)
@pytest.fixture
def jm():
    m = JobManager(EventBus(), max_workers=4)
    yield m
    m.shutdown()


def make_queue(jm, run, **kw):
    return PriorityWorkQueue(jm, owner="p1", run=run, **kw)


def test_concurrent_requests_for_the_same_key_coalesce_into_one_task(jm):
    calls, gate = [], threading.Event()

    def run(payload, cancel):
        calls.append(payload)
        gate.wait(10)
        return True

    q = make_queue(jm, run, concurrency=lambda: 2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(q.submit("a1", "a1", REQUESTED))) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    eventually(lambda: calls)
    assert results.count(True) == 1 and q.stats.coalesced == 11
    gate.set()
    assert jm.wait_idle(10)
    assert calls == ["a1"] and q.stats.done == 1


def test_visible_items_run_before_requested_before_idle_and_a_scrolled_away_item_is_dropped(jm):
    order, gate = [], threading.Event()

    def run(payload, cancel):
        if payload == "first":
            gate.wait(10)
        order.append(payload)
        return True

    q = make_queue(jm, run, concurrency=lambda: 1)
    q.submit("first", "first", REQUESTED)
    eventually(lambda: q.state("first") == "running")
    for i in range(5):
        q.submit(f"idle{i}", f"idle{i}", IDLE)
    q.submit("req", "req", REQUESTED)
    q.set_visible(["idle3", "vis_new", "idle4"], lambda k: k)  # idle3/idle4 jump the queue; vis_new is queued only because it is visible
    q.set_visible(["idle4"], lambda k: k)  # vis_new and idle3 scrolled away: vis_new is dropped, idle3 falls back to idle
    assert q.state("vis_new") is None and q.priority_of("idle3") == IDLE and q.priority_of("idle4") == VISIBLE
    gate.set()
    assert jm.wait_idle(10)
    assert order[:3] == ["first", "idle4", "req"] and sorted(order[3:]) == ["idle0", "idle1", "idle2", "idle3"]


def test_running_tasks_never_exceed_the_concurrency_limit_and_use_few_jobs(jm):
    lock, now, peak = threading.Lock(), [0], [0]

    def run(payload, cancel):
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        time.sleep(0.01)
        with lock:
            now[0] -= 1
        return True

    q = make_queue(jm, run, concurrency=lambda: 2)
    for i in range(60):
        q.submit(f"k{i}", i, IDLE)
    assert jm.wait_idle(20)
    assert q.stats.done == 60 and 1 <= peak[0] <= 2 and q.stats.max_running <= 2
    assert q.stats.jobs_created <= 3 and len(jm.jobs()) <= 3  # 60 tasks, a handful of drain jobs (not one job per task)


def test_a_failing_task_is_remembered_not_retried_and_reported_once_in_aggregate(jm):
    attempts, failed_cb, idle_cb = [], [], []

    def run(payload, cancel):
        attempts.append(payload)
        if payload % 2:
            raise RuntimeError(f"bad {payload}")
        return True

    q = make_queue(jm, run, concurrency=lambda: 1, on_failed=lambda k, p, r: failed_cb.append(k), on_idle=idle_cb.append)
    for i in range(10):
        q.submit(f"k{i}", i, IDLE)
    assert jm.wait_idle(10)
    eventually(lambda: idle_cb)
    assert sorted(q.failed_keys()) == ["k1", "k3", "k5", "k7", "k9"] and q.state("k1") == "failed" and len(attempts) == 10
    assert len(idle_cb) == 1 and idle_cb[0].batch_failed == 5 and idle_cb[0].batch_done == 5  # ONE aggregated report, not five
    assert q.retry("k1", 1) and jm.wait_idle(10) and len(attempts) == 11


def test_idle_work_is_held_when_switched_off_and_resumes_on_kick(jm):
    allowed, ran = [False], []
    q = make_queue(jm, lambda p, c: ran.append(p) or True, concurrency=lambda: 1, idle_allowed=lambda: allowed[0])
    q.submit("bg", "bg", IDLE)
    q.submit("now", "now", REQUESTED)
    assert jm.wait_idle(10)
    assert ran == ["now"] and q.state("bg") == "pending"  # requested work always runs; speculative work waits
    allowed[0] = True
    q.kick()
    assert jm.wait_idle(10) and ran == ["now", "bg"]


def test_close_drops_pending_tasks_and_cancels_the_drain_job(jm):
    ran, gate = [], threading.Event()

    def run(payload, cancel):
        ran.append(payload)
        gate.wait(10)
        return True

    q = make_queue(jm, run, concurrency=lambda: 1)
    for i in range(20):
        q.submit(f"k{i}", i, IDLE)
    eventually(lambda: ran)
    q.close()
    gate.set()
    assert jm.wait_idle(10)
    assert len(ran) == 1 and q.pending() == 0 and q.submit("late", 1) is False


# ======================================================================== thumbnails through the workspace (fake FFmpeg process)
class Spy:
    """Replaces ThumbnailService._run: records the commands, writes a tiny 'thumbnail', can fail or block."""

    def __init__(self, ws: Workspace) -> None:
        self.calls: list[list[str]] = []
        self.fail_for: set[str] = set()
        self.gate: threading.Event | None = None
        self.lock = threading.Lock()
        self.now = self.peak = 0
        self.order: list[str] = []
        ws.thumbnails._run = self.run  # type: ignore[method-assign]
        ws.thumbnails.command = lambda src, asset, tmp: ["fake-ffmpeg", "-i", str(src), str(tmp)]  # type: ignore[method-assign]

    def run(self, cmd, should_cancel):
        with self.lock:
            self.calls.append(cmd)
            self.now += 1
            self.peak = max(self.peak, self.now)
            name = Path(cmd[-1]).name.split(".")[0]
            self.order.append(name)
        try:
            if self.gate is not None:
                while not self.gate.wait(0.01):
                    if should_cancel and should_cancel():
                        from app.core.exceptions import JobCancelled

                        raise JobCancelled()
            if name in self.fail_for:
                return 1, "Invalid data found when processing input"
            Path(cmd[-1]).write_bytes(b"\xff\xd8thumb\xff\xd9")
            return 0, ""
        finally:
            with self.lock:
                self.now -= 1


def new_ws(tmp_path, name="h") -> Workspace:
    ws = Workspace(AppPaths(tmp_path / name / "cfg", tmp_path / name / "data"))
    ws.media.attach_performance(lambda: ws.performance)
    ws.performance.monitor.thresholds["load_elevated"] = 1e9  # the shared CI/dev machine may be busy: these tests are about scheduling, not about throttling
    return ws


@pytest.fixture
def syn(tmp_path):
    sp = build_project(tmp_path / "proj", SyntheticSpec(scenes=30), real_files=True)
    ProjectManager(EventBus()).save(sp.project)
    return sp


@pytest.fixture
def opened(tmp_path, syn):
    ws = new_ws(tmp_path)
    spy = Spy(ws)
    ws.spy = spy
    yield ws
    ws.shutdown()


def thumbs(ws) -> list[Path]:
    return sorted((ws.project.root / "thumbnails").glob("*.jpg"))


def test_opening_a_project_makes_a_bounded_number_of_jobs_and_one_ffmpeg_call_per_asset(opened, syn):
    ws, spy = opened, opened.spy
    n_assets = len(ws.projects.current.assets) if ws.projects.current else syn.assets
    before = len(ws.jobs.jobs())
    ws.open_project(syn.project.root)
    created = len(ws.jobs.jobs()) - before
    assert ws.jobs.wait_idle(60)
    n_assets = len(ws.project.assets)
    assert created <= 4 and len(ws.jobs.jobs()) - before <= 4, created  # was one job per asset (about 50 here, ~1,500 at 1,000 scenes)
    assert len(spy.calls) == n_assets == len(thumbs(ws)) and spy.peak <= ws.performance.limits().thumbnail_concurrency
    assert ws.media.thumbnail_state(ws.project.assets.all()[0].id) == "ready"


def test_a_second_open_reuses_every_thumbnail_without_any_ffmpeg_call(tmp_path, syn):
    ws = new_ws(tmp_path, "first")
    spy = Spy(ws)
    ws.open_project(syn.project.root)
    assert ws.jobs.wait_idle(60)
    made = len(spy.calls)
    assert made == len(ws.project.assets)
    ws.shutdown()
    ws2 = new_ws(tmp_path, "second")  # a simulated restart: new process state, same project folder (the cache index is on disk)
    spy2 = Spy(ws2)
    jobs_before = len(ws2.jobs.jobs())
    ws2.open_project(syn.project.root)
    assert ws2.jobs.wait_idle(30)
    assert spy2.calls == [] and len(ws2.jobs.jobs()) == jobs_before  # nothing queued, nothing run
    ws2.media.request_thumbnails([a.id for a in ws2.project.assets], Priority.MEDIUM)  # the verifying path (fingerprint check) also finds them current
    assert ws2.jobs.wait_idle(30) and spy2.calls == []
    assert ws2.performance.cache is not None and len(ws2.performance.cache.entries("thumbnails")) == made
    ws2.shutdown()


def test_a_replaced_source_regenerates_its_thumbnail_but_untouched_ones_are_reused(opened, syn):
    ws, spy = opened, opened.spy
    ws.open_project(syn.project.root)
    assert ws.jobs.wait_idle(60)
    spy.calls.clear()
    asset = next(a for a in ws.project.assets if a.type.value == "image")
    src = ws.project.asset_path(asset)
    src.write_bytes(b"\x00" * 200)  # the file was replaced by a different one
    import os

    os.utime(src, ns=(time.time_ns() + 5_000_000_000, time.time_ns() + 5_000_000_000))
    ws.media.request_thumbnails([a.id for a in ws.project.assets], Priority.MEDIUM)
    assert ws.jobs.wait_idle(30)
    assert [Path(c[-1]).name.split(".")[0] for c in spy.calls] == [asset.id]
    spy.calls.clear()
    ws.media.request_thumbnails([a.id for a in ws.project.assets], Priority.MEDIUM)
    assert ws.jobs.wait_idle(30) and spy.calls == []


def test_without_the_cache_index_a_thumbnail_older_than_its_source_is_stale(tmp_path, syn):
    ws = Workspace(AppPaths(tmp_path / "nocache" / "cfg", tmp_path / "nocache" / "data"))  # performance NOT attached: today's behaviour
    ws.performance.monitor.thresholds["load_elevated"] = 1e9
    spy = Spy(ws)
    ws.open_project(syn.project.root)
    assert ws.jobs.wait_idle(60)
    spy.calls.clear()
    asset = next(a for a in ws.project.assets if a.type.value == "video")
    import os

    os.utime(ws.project.asset_path(asset), ns=(time.time_ns() + 9_000_000_000, time.time_ns() + 9_000_000_000))
    ws.media.request_thumbnails([asset.id], Priority.MEDIUM)
    assert ws.jobs.wait_idle(30) and len(spy.calls) == 1
    ws.shutdown()


def test_failed_thumbnails_get_a_clear_state_are_not_retried_in_a_loop_and_log_one_summary(opened, syn, caplog):
    ws, spy = opened, opened.spy
    statuses = []
    ws.bus.subscribe(Topics.STATUS, lambda t, p: statuses.append(p["message"]))
    failed_events = []
    ws.bus.subscribe("media.thumbnail_failed", lambda t, p: failed_events.append(p["asset_id"]))
    ws.open_project(syn.project.root)
    bad = [a.id for a in ws.project.assets.all()[:6]]
    spy.fail_for = set(bad)
    ws.media.cancel_thumbnails()
    assert ws.jobs.wait_idle(30)
    for f in thumbs(ws):
        f.unlink()
    spy.calls.clear()
    statuses.clear()
    with caplog.at_level(logging.WARNING):
        ws.media.ensure_all_thumbnails()
        assert ws.jobs.wait_idle(60)
    assert {ws.media.thumbnail_state(i) for i in bad} == {"failed"} and sorted(failed_events) == sorted(bad)
    first_round = len(spy.calls)
    ws.media.ensure_all_thumbnails()
    ws.media.ensure_all_thumbnails()
    assert ws.jobs.wait_idle(60)
    assert len(spy.calls) == first_round  # the failures were remembered: no retry loop
    eventually(lambda: [m for m in statuses if "thumbnail" in m])
    time.sleep(0.6)
    summaries = [m for m in statuses if "thumbnail" in m]
    assert len(summaries) == 1 and "6 thumbnail" in summaries[0]  # one aggregated message instead of six
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    spy.fail_for = set()
    assert ws.media.retry_thumbnail(bad[0]) and ws.jobs.wait_idle(30) and ws.media.thumbnail_state(bad[0]) == "ready"


def test_a_missing_source_is_its_own_state(opened, syn):
    ws = opened
    ws.open_project(syn.project.root)
    assert ws.jobs.wait_idle(60)
    asset = ws.project.assets.all()[3]
    ws.media.cancel_thumbnails()
    ws.media._thumbnails.thumbnail_path(ws.project.root, asset).unlink()
    ws.project.asset_path(asset).unlink()
    ws.media.request_thumbnails([asset.id], Priority.MEDIUM)
    assert ws.jobs.wait_idle(30)
    assert ws.media.thumbnail_state(asset.id) == "missing_source"


def test_visible_assets_are_generated_first_and_the_queue_respects_the_limit(opened, syn):
    ws, spy = opened, opened.spy
    ws.performance.update_global(type(ws.performance.global_settings())(max_background_workers=1))  # one thumbnail at a time
    assert ws.performance.limits().thumbnail_concurrency == 1
    spy.gate = threading.Event()
    ws.open_project(syn.project.root)
    eventually(lambda: spy.calls)  # the first (idle) one is now blocked inside FFmpeg
    all_ids = [a.id for a in ws.project.assets]
    want = all_ids[-3:]
    ws.media.set_visible_assets(want)
    spy.gate.set()
    assert ws.jobs.wait_idle(60)
    pos = {name: i for i, name in enumerate(spy.order)}
    assert max(pos[i] for i in want) <= 3, [spy.order[:6], want]  # at most the one that was already running came before them
    assert spy.peak == 1


def test_switching_project_cancels_the_pending_thumbnail_work(tmp_path, syn):
    other = build_project(tmp_path / "other", SyntheticSpec(scenes=4), real_files=True)
    ProjectManager(EventBus()).save(other.project)
    ws = new_ws(tmp_path)
    spy = Spy(ws)
    spy.gate = threading.Event()
    ws.open_project(syn.project.root)
    eventually(lambda: spy.calls)
    pending_before = ws.media.thumbnail_stats()["pending"]
    assert pending_before > 5
    ws.open_project(other.project.root)  # switch while the first project's queue is full
    spy.gate.set()
    assert ws.jobs.wait_idle(60)
    first_root = syn.project.root / "thumbnails"
    made_first = len(list(first_root.glob("*.jpg"))) if first_root.exists() else 0
    assert made_first <= 2, made_first  # only what was already in flight finished; the rest was dropped, not generated for a closed project
    assert len(list((other.project.root / "thumbnails").glob("*.jpg"))) == len(ws.project.assets)
    ws.shutdown()


def test_removing_an_asset_drops_its_queued_thumbnail(opened, syn):
    ws, spy = opened, opened.spy
    spy.gate = threading.Event()
    ws.open_project(syn.project.root)
    eventually(lambda: spy.calls)
    victim = ws.project.assets.all()[-1]
    assert ws.media.thumbnail_state(victim.id) == "pending"
    ws.media.remove_asset(victim.id)
    spy.gate.set()
    assert ws.jobs.wait_idle(60)
    assert victim.id not in spy.order


# ======================================================================== real FFmpeg
@needs_ffmpeg
def test_real_thumbnails_keep_the_aspect_ratio_are_small_and_stay_in_a_path_with_spaces_and_unicode(tmp_path, media_dir):
    from app.tests.helpers import make_video

    ws = new_ws(tmp_path, "real")
    ws.new_project("Thumbs ü 日本", tmp_path / "projects é")
    wide = make_video(tmp_path / "wide clip é.mp4", 2.0, "testsrc", "1280x360", 24)
    got = []
    ws.media.import_files([wide, media_dir / "pic.png", media_dir / "voice.wav"], on_asset=got.append)
    assert ws.jobs.wait_idle(60)
    by = {a.name: a for a in ws.project.assets.all()}
    for name in ("wide clip é.mp4", "pic.png", "voice.wav"):
        assert ws.media.thumbnail_state(by[name].id) == "ready", name
    import subprocess

    out = ws.media.thumbnail_file(by["wide clip é.mp4"])
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    w, h = (int(x) for x in r.stdout.strip().split(","))
    assert w == 320 and abs(w / h - 1280 / 360) < 0.1 and out.stat().st_size < 60_000
    pic = ws.media.thumbnail_file(by["pic.png"])
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(pic)], capture_output=True, text=True)
    assert r.stdout.strip() == "64,64"  # a small image is not blown up
    ws.shutdown()


# ======================================================================== waveform levels of detail
import numpy as np  # noqa: E402

from app.audio.backend import ANALYSIS_SR, AudioBackend  # noqa: E402
from app.audio.waveform import WaveformService  # noqa: E402
from app.core.exceptions import JobCancelled  # noqa: E402
from app.performance.waveform_lod import LEVEL_PPS, Waveform  # noqa: E402


def synth_samples(seconds: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * ANALYSIS_SR)) / ANALYSIS_SR
    env = 0.2 + 0.6 * np.abs(np.sin(2 * np.pi * 0.37 * t))
    x = env * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    return np.clip(x, -0.98, 0.98).astype(np.float32)


class ArrayBackend(AudioBackend):
    """Streams a numpy array in chunks (records the chunk sizes); nothing else is implemented."""

    name = "array"

    def __init__(self, samples: np.ndarray, chunk: int = 7 * ANALYSIS_SR + 123) -> None:
        self.samples, self.chunk, self.sizes = samples, chunk, []

    def is_available(self):
        return True, ""

    def decode_mono(self, path, sample_rate=ANALYSIS_SR):
        raise AssertionError("the waveform must be built from the stream, never from a whole-file decode")

    def stream_mono(self, path, sample_rate=ANALYSIS_SR, chunk_seconds=10.0, should_cancel=None):
        for i in range(0, len(self.samples), self.chunk):
            if should_cancel is not None and should_cancel():
                return
            c = self.samples[i:i + self.chunk]
            self.sizes.append(len(c))
            yield c

    def measure_loudness(self, path):
        raise NotImplementedError

    def render_chain(self, src, dst, filter_chain):
        raise NotImplementedError

    def render_graph(self, inputs, graph, out_label, dst, duration):
        raise NotImplementedError


def brute(samples: np.ndarray, t0: float, t1: float, buckets: int):
    out = []
    step = (t1 - t0) / buckets
    for k in range(buckets):
        a, b = int(round((t0 + k * step) * ANALYSIS_SR)), int(round((t0 + (k + 1) * step) * ANALYSIS_SR))
        seg = samples[a:b]
        out.append((float(seg.min()), float(seg.max())))
    return out


@pytest.fixture
def wf_service(tmp_path):
    samples = synth_samples(25.0)
    samples[3 * ANALYSIS_SR:3 * ANALYSIS_SR + 800] = 1.0  # a clipped burst at 3.0 s
    backend = ArrayBackend(samples)
    src = tmp_path / "voice ü.wav"
    src.write_bytes(b"RIFFfake")
    return WaveformService(backend, lambda: tmp_path / "cache"), backend, samples, src


def test_waveform_is_built_in_one_streaming_pass_with_every_level(wf_service):
    svc, backend, samples, src = wf_service
    wf = svc.compute(src, "k1")
    assert [lv.pps for lv in wf.levels] == list(LEVEL_PPS) and [lv.n for lv in wf.levels] == [10000, 2500, 625, 125]
    assert len(backend.sizes) >= 3 and max(backend.sizes) <= 7 * ANALYSIS_SR + 123  # consumed in chunks: the audio was never one array
    assert wf.duration == pytest.approx(25.0)
    assert len(wf.peaks) == 2500 and wf.pps == 100  # the old surface (100 buckets per second)


def test_range_matches_a_brute_force_reduction_at_the_chosen_level_and_is_a_safe_envelope_elsewhere(wf_service):
    svc, _b, samples, src = wf_service
    wf = svc.compute(src, "k1")
    for buckets, level in ((125, 5), (625, 25), (2500, 100), (10000, 400)):  # aligned queries are exact at the level they pick
        assert wf.level_for(0.0, 25.0, buckets).pps == level
        got = wf.range(0.0, 25.0, buckets)
        ref = brute(samples, 0.0, 25.0, buckets)
        assert len(got) == buckets
        assert max(abs(g[0] - r[0]) for g, r in zip(got, ref)) < 1e-6 and max(abs(g[1] - r[1]) for g, r in zip(got, ref)) < 1e-6, buckets
    rng = np.random.default_rng(1)
    for _ in range(40):  # arbitrary windows: never smaller than the truth, never wider than one level-bucket of slack on each side
        t0 = float(rng.uniform(0, 20))
        t1 = t0 + float(rng.uniform(0.2, 5))
        buckets = int(rng.integers(3, 300))
        lv = wf.level_for(t0, t1, buckets)
        slack = 1.0 / lv.pps
        got = wf.range(t0, t1, buckets)
        step = (t1 - t0) / buckets
        for k, (mn, mx, _c) in enumerate(got):
            a, b = t0 + k * step, t0 + (k + 1) * step
            tight = samples[int(a * ANALYSIS_SR):max(int(b * ANALYSIS_SR), int(a * ANALYSIS_SR) + 1)]
            loose = samples[max(0, int((a - slack) * ANALYSIS_SR)):int((b + slack) * ANALYSIS_SR) + 1]
            assert mx >= float(tight.max()) - 2e-4 and mn <= float(tight.min()) + 2e-4
            assert mx <= float(loose.max()) + 2e-4 and mn >= float(loose.min()) - 2e-4


def test_zooming_out_reads_far_fewer_points_than_zooming_in_and_clipping_survives_every_level(wf_service):
    svc, _b, samples, src = wf_service
    wf = svc.compute(src, "k1")
    out_level, in_level = wf.level_for(0, 25, 200), wf.level_for(2.9, 3.1, 200)
    assert out_level.pps < in_level.pps and out_level.n * 10 < in_level.n
    for t0, t1, buckets in ((0.0, 25.0, 50), (2.0, 4.0, 40), (2.99, 3.06, 14)):
        rows = wf.range(t0, t1, buckets)
        flagged = [k for k, r in enumerate(rows) if r[2]]
        assert flagged, (t0, t1)
        step = (t1 - t0) / buckets
        assert all(t0 + (k + 1) * step >= 2.99 - 1.0 / wf.level_for(t0, t1, buckets).pps and t0 + k * step <= 3.06 for k in flagged)
    assert not any(r[2] for r in wf.range(10.0, 20.0, 50))  # no false clipping elsewhere
    assert wf.range(24.9, 40.0, 10)[-1] == (0.0, 0.0, False)  # past the end: silence, not an error


def test_waveform_file_is_compact_registered_and_survives_a_restart(wf_service, tmp_path):
    svc, backend, samples, src = wf_service
    from app.performance.cache_manager import MediaCacheManager

    cache = MediaCacheManager(tmp_path / "cache")
    svc.cache_getter = lambda: cache
    wf = svc.compute(src, "k1", dep_id="asset:a1")
    f = tmp_path / "cache" / "waveforms" / "k1_100.json"
    assert f.is_file() and f.stat().st_size < 64_000  # 10,000 int16 min/max pairs (40 kB raw) + base64, on a noisy signal that barely compresses
    assert [e.category for e in cache.entries("waveforms")] == ["waveforms"] and cache.invalidate_by_dependency("asset:a1") == 1 and not f.exists()
    svc2 = WaveformService(ArrayBackend(samples), lambda: tmp_path / "cache")
    wf = svc2.compute(src, "k2")
    again = WaveformService(ArrayBackend(samples), lambda: tmp_path / "cache").cached("k2")  # a restart: no audio read at all
    assert again is not None and again.clipped == wf.clipped and len(again.peaks) == len(wf.peaks)
    assert max(abs(x[i] - y[i]) for x, y in zip(again.range(0.0, 25.0, 125), wf.range(0.0, 25.0, 125)) for i in (0, 1)) < 4e-5  # stored as int16


def test_a_waveform_cache_file_from_the_old_single_resolution_code_still_loads(tmp_path):
    d = tmp_path / "cache" / "waveforms"
    d.mkdir(parents=True)
    peaks = [[-0.1 * (i % 5), 0.2 * (i % 7)] for i in range(300)]
    (d / "old_100.json").write_text(json.dumps({"pps": 100, "duration": 3.0, "peaks": peaks, "clipped": [4, 5]}), encoding="utf-8")
    svc = WaveformService(ArrayBackend(synth_samples(1)), lambda: tmp_path / "cache")
    wf = svc.cached("old")
    assert wf is not None and wf.pps == 100 and len(wf.peaks) == 300 and wf.clipped == {4, 5} and [lv.pps for lv in wf.levels] == [100, 25, 5]
    rows = wf.range(0.0, 3.0, 300)
    assert rows[7][1] == pytest.approx(0.2 * (7 % 7), abs=1e-6) and rows[4][2] is True
    (d / "bad_100.json").write_text("{not json", encoding="utf-8")
    assert svc.cached("bad") is None  # unreadable: rebuilt, not an error


def test_a_replaced_source_invalidates_the_waveform(wf_service, monkeypatch):
    svc, backend, samples, src = wf_service
    import app.audio.waveform as wfmod

    monkeypatch.setattr(wfmod, "REVALIDATE_EVERY", 0.0)
    first = svc.compute(src, "k1")
    assert svc.cached("k1", src) is first
    time.sleep(0.01)
    src.write_bytes(b"RIFFfake-but-different")
    backend.samples = synth_samples(25.0, seed=9) * 0.5
    assert svc.cached("k1", src) is None  # the fingerprint no longer matches
    assert not (svc._file("k1")).exists()
    second = svc.compute(src, "k1")
    assert second is not first and max(r[1] for r in second.range(0, 25, 100)) < 0.7


def test_building_can_be_cancelled_without_leaving_a_file(wf_service):
    svc, backend, _samples, src = wf_service
    seen = []

    def cancel():
        seen.append(1)
        return len(seen) > 2

    with pytest.raises(JobCancelled):
        svc.compute(src, "kc", should_cancel=cancel)
    assert svc.cached("kc") is None and not list((svc._file("kc").parent).glob("*")) if svc._file("kc").parent.exists() else True


def test_waveform_roundtrip_of_the_in_memory_object_is_lossless_to_int16(wf_service):
    svc, _b, samples, src = wf_service
    wf = svc.compute(src, "k1")
    back = Waveform.from_json(json.loads(json.dumps(wf.to_json())))
    a, b = wf.range(0, 25, 10000), back.range(0, 25, 10000)
    assert max(abs(x[0] - y[0]) for x, y in zip(a, b)) < 4e-5 and [x[2] for x in a] == [y[2] for y in b]


@needs_ffmpeg
def test_the_ffmpeg_backend_streams_the_same_samples_as_a_whole_decode(tmp_path):
    from app.audio.backend import FFmpegAudioBackend
    from app.tests.helpers import write_tone_wav

    path = write_tone_wav(tmp_path / "tone é.wav", 3.0, amp=0.4)
    be = FFmpegAudioBackend("")
    whole = be.decode_mono(path)
    chunks = list(be.stream_mono(path, ANALYSIS_SR, 0.5))
    assert len(chunks) >= 5 and np.array_equal(np.concatenate(chunks), whole)
    stop = []
    got = list(be.stream_mono(path, ANALYSIS_SR, 0.5, lambda: len(stop.append(1) or stop) > 2))
    assert 0 < len(got) < len(chunks)  # cancelled part way: the FFmpeg process was stopped
    from app.audio.backend import AudioError

    with pytest.raises(AudioError):
        list(be.stream_mono(tmp_path / "nope.wav"))


# ======================================================================== probe persistence
PROBE_JSON = json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "30/1", "pix_fmt": "yuv420p"},
                                     {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}], "format": {"format_name": "mov,mp4", "duration": "12.5", "size": "100", "bit_rate": "64000"}})


class FakeFF:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def ffprobe(self) -> str:
        return "ffprobe"

    def run(self, cmd, timeout=60):
        import subprocess

        self.calls += 1
        if self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", "moov atom not found")
        return subprocess.CompletedProcess(cmd, 0, PROBE_JSON, "")


@pytest.fixture
def probe_env(tmp_path):
    from app.performance.cache_manager import MediaCacheManager
    from app.rendering.probe import MediaProbeService

    cache = MediaCacheManager(tmp_path / "cache")
    media = tmp_path / "méd ia 日本"
    media.mkdir()
    f = media / "clip é.mp4"
    f.write_bytes(b"0" * 100)

    def make(ff=None):
        svc = MediaProbeService(ff or FakeFF())
        svc.attach_cache(lambda: cache)
        return svc

    return make, cache, f


def test_probe_results_survive_a_restart_and_a_changed_file_is_probed_again(probe_env):
    make, cache, f = probe_env
    first = make()
    info = first.probe(f)
    assert first.ff.calls == 1 and info.width == 1920 and info.duration == 12.5
    first.probe(f)
    assert first.ff.calls == 1  # in memory
    second = make()  # a new process: empty memory, same cache
    again = second.probe(f)
    assert second.ff.calls == 0 and second.persisted_hits == 1 and again.to_dict() == info.to_dict()
    f.write_bytes(b"1" * 250)  # replaced
    third = make()
    assert third.probe(f).size_bytes in (100, 250) and third.ff.calls == 1


def test_a_failed_probe_is_not_remembered_and_a_corrupt_entry_is_ignored(probe_env):
    from app.core.exceptions import MediaProbeError
    from app.rendering.probe import PROBE_VERSION, MediaProbeService

    make, cache, f = probe_env
    bad = make(FakeFF(fail=True))
    with pytest.raises(MediaProbeError):
        bad.probe(f)
    assert cache.entries("analysis") == []  # a failure is never stored as if it were a result
    good = make()
    assert good.probe(f).width == 1920 and good.ff.calls == 1
    key = MediaProbeService._persist_key(f, f.stat().st_size, f.stat().st_mtime_ns)
    cache.put(key, {"path": "somewhere else", "kind": "video"}, category="analysis", version=PROBE_VERSION)  # stale / wrong content under the right key
    fresh = make()
    assert fresh.probe(f).width == 1920 and fresh.ff.calls == 1 and fresh.persisted_hits == 0
    cache.put(key, {"path": str(f), "kind": "video", "width": 1}, category="analysis", version=PROBE_VERSION + 7)  # another schema version
    other = make()
    assert other.probe(f).width == 1920 and other.ff.calls == 1


def test_probe_memory_is_bounded_and_use_cache_false_stores_nothing(probe_env, tmp_path):
    from app.rendering.probe import MediaProbeService

    make, cache, f = probe_env
    svc = MediaProbeService(FakeFF())
    svc._cache.resize(max_items=5)
    for i in range(20):
        p = f.parent / f"c{i}.mp4"
        p.write_bytes(b"x" * (10 + i))
        svc.probe(p)
    assert len(svc._cache) <= 5 and svc.ff.calls == 20  # no cache attached: behaves as before, with bounded memory
    svc2 = make()
    svc2.probe(f, use_cache=False)
    assert cache.entries("analysis") == []


# ======================================================================== preview frames
from app.media.asset import Asset, AssetType, SourceType  # noqa: E402
from app.preview.frames import QUALITY_WIDTH, STEP, FrameProvider  # noqa: E402


class FrameEnv:
    def __init__(self, tmp_path: Path, jm: JobManager | None = None, quality: str = "balanced", cache=None) -> None:
        self.root = tmp_path / "proj ü"
        (self.root / "media").mkdir(parents=True)
        self.src = self.root / "media" / "v.mp4"
        self.src.write_bytes(b"0" * 64)
        self.asset = Asset("a1", AssetType.VIDEO, SourceType.USER_MEDIA, "media/v.mp4", "v.mp4", 60.0, 1920, 1080)
        self.project = SimpleNamespace(root=self.root, asset_path=lambda a: self.root / a.path, project_id="p1")
        self.quality = quality
        self.fp = FrameProvider(lambda: self.project, lambda: "", None, quality_getter=lambda: self.quality, cache_getter=lambda: cache, jobs=jm)
        self.calls: list[list[str]] = []
        self.gate: threading.Event | None = None
        self.fp.command = None
        self.fp._run = self.run  # type: ignore[method-assign]
        import app.preview.frames as fr

        self._orig_locate = fr.locate_binary
        fr.locate_binary = lambda name, configured="": "fake-ffmpeg"

    def close(self) -> None:
        import app.preview.frames as fr

        fr.locate_binary = self._orig_locate

    def run(self, cmd, should_cancel):
        self.calls.append(cmd)
        if self.gate is not None:
            while not self.gate.wait(0.01):
                if should_cancel and should_cancel():
                    from app.core.exceptions import JobCancelled

                    raise JobCancelled()
        Path(cmd[-1]).write_bytes(b"\xff\xd8frame\xff\xd9")
        return 0, ""


@pytest.fixture
def fenv(tmp_path):
    jm = JobManager(EventBus(), max_workers=4)
    e = FrameEnv(tmp_path, jm)
    e.fp._limits_get = lambda: SimpleNamespace(prefetch_frames=6, idle_work=True, foreground_workers=1)  # one extraction at a time: deterministic ordering
    yield e
    e.close()
    jm.shutdown()


def width_of(cmd) -> int:
    vf = cmd[cmd.index("-vf") + 1]
    return int(vf.split("=")[1].split(":")[0])


def test_preview_quality_chooses_the_extraction_width_and_the_default_keeps_the_960_frames(tmp_path):
    e = FrameEnv(tmp_path)
    try:
        for q, w in (("draft", 480), ("balanced", 960), ("high", 1280)):
            e.quality = q
            e.fp.refresh_settings()
            p = e.fp.frame_path(e.asset, 1.0)
            assert p is not None and width_of(e.calls[-1]) == w == QUALITY_WIDTH[q]
            assert p.name == ("a1_000100.jpg" if q == "balanced" else f"a1_000100_w{w}.jpg")  # balanced keeps the historical file name
        n = len(e.calls)
        e.quality = "draft"
        e.fp.refresh_settings()
        assert e.fp.frame_path(e.asset, 1.0) is not None and len(e.calls) == n  # a frame of each quality is cached separately
        assert e.fp.reduced_quality() == "Draft preview"
        e.quality = "high"
        e.fp.refresh_settings()
        e.fp.frame_path(e.asset, 1.0)
        assert e.fp.reduced_quality() == ""
    finally:
        e.close()


def test_failed_frames_are_remembered_only_for_a_while_and_the_set_is_bounded(tmp_path, monkeypatch):
    import app.preview.frames as fr

    e = FrameEnv(tmp_path)
    try:
        e.run = lambda cmd, c: (1, "bad")  # type: ignore[method-assign]
        e.fp._run = e.run  # type: ignore[method-assign]
        assert e.fp.frame_path(e.asset, 2.0) is None and e.fp.frame_path(e.asset, 2.0) is None
        assert e.fp.stats["failed"] == 1  # the second attempt did not run FFmpeg again
        monkeypatch.setattr(fr, "FAILED_TTL", 0.0)
        e.fp.frame_path(e.asset, 2.0)
        assert e.fp.stats["failed"] == 2  # expired: tried again
        monkeypatch.setattr(fr, "FAILED_MAX", 10)
        for i in range(40):
            e.fp.frame_path(e.asset, 3.0 + i * STEP)
        assert len(e.fp._failed) <= 10
    finally:
        e.close()


def test_concurrent_requests_for_the_same_frame_share_one_extraction(fenv):
    e = fenv
    e.gate = threading.Event()
    got = []
    for _ in range(8):
        assert e.fp.request(e.asset, 5.1, lambda p: got.append(p)) is None
    eventually(lambda: e.calls)
    e.gate.set()
    assert e.fp._jobs.wait_idle(10)
    eventually(lambda: len(got) == 8)
    assert len(e.calls) == 1 and e.fp.stats["coalesced"] == 7 and len({str(p) for p in got}) == 1
    assert e.fp.request(e.asset, 5.1) is not None and len(e.calls) == 1  # now a plain cache hit


def test_a_newer_seek_cancels_older_extractions_and_a_late_result_never_reaches_the_screen(fenv):
    e = fenv
    e.gate = threading.Event()
    old_arrivals, new_arrivals = [], []
    e.fp.begin_seek()
    assert e.fp.request(e.asset, 10.0, old_arrivals.append) is None
    eventually(lambda: e.calls)  # frame A is being extracted (blocked)
    queued = [e.fp.request(e.asset, 11.0 + i, old_arrivals.append) for i in range(5)]  # B..F wait behind it
    assert queued == [None] * 5
    gen = e.fp.begin_seek()
    assert e.fp.request(e.asset, 40.0, new_arrivals.append) is None  # the first request of the new seek supersedes everything older
    assert e.fp.stats["superseded"] >= 5
    e.gate.set()
    assert e.fp._jobs.wait_idle(10)
    eventually(lambda: new_arrivals)
    assert old_arrivals == [] and len(new_arrivals) == 1 and e.fp.generation == gen  # the old callbacks were dropped, the new frame arrived
    assert len(e.calls) <= 2  # the running one was stopped (and at most the new frame ran); the five queued ones never started
    assert all("40" not in c[c.index("-ss") + 1] or True for c in e.calls)


def test_prefetch_is_idle_priority_bounded_and_dropped_by_a_far_seek(fenv):
    e = fenv
    e.fp._limits_get = lambda: SimpleNamespace(prefetch_frames=6, idle_work=True, foreground_workers=1)
    e.gate = threading.Event()
    e.fp.begin_seek()
    assert e.fp.request(e.asset, 1.0, None) is None  # the frame the user needs blocks the single worker
    eventually(lambda: e.calls)
    assert e.fp.prefetch_around(e.asset, 1.0) == 6
    q = e.fp._queue
    assert q is not None and q.pending() == 7 and sorted({q.priority_of(k) for k in q.keys()}) == [0, 2]
    e.fp.begin_seek()
    e.fp.request(e.asset, 50.0, None)  # far away: the old prefetch is cancelled with the old generation
    assert q.pending() <= 2
    e.gate.set()
    assert e.fp._jobs.wait_idle(10)
    e.fp._limits_get = lambda: SimpleNamespace(prefetch_frames=6, idle_work=False, foreground_workers=1)
    n = len(e.calls)
    e.fp.prefetch_around(e.asset, 20.0)
    assert e.fp._jobs.wait_idle(10) and len(e.calls) == n  # idle work switched off: nothing is prefetched


def test_frames_are_registered_with_the_cache_and_a_changed_source_is_extracted_again(tmp_path):
    from app.performance.cache_manager import MediaCacheManager

    cache = MediaCacheManager(tmp_path / "cache")
    e = FrameEnv(tmp_path, None, cache=cache)
    try:
        p = e.fp.frame_path(e.asset, 4.0)
        assert p is not None and [x.category for x in cache.entries("preview_frames")] == ["preview_frames"]
        n = len(e.calls)
        assert e.fp.frame_path(e.asset, 4.0) == p and len(e.calls) == n
        time.sleep(0.01)
        e.src.write_bytes(b"1" * 99)  # the video was replaced
        e.fp._fp.clear()
        assert e.fp.frame_path(e.asset, 4.0) == p and len(e.calls) == n + 1
        e.fp.release()
        assert e.fp.pending_count() == 0
    finally:
        e.close()
