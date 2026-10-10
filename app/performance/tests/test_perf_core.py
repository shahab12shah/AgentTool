"""Performance foundation: metrics, profiler, resource/memory monitors, bounded LRU, settings resolution, bottleneck report, synthetic projects, benchmarks."""

from __future__ import annotations

import json
import threading
import time

import pytest

from app.performance import bottleneck_report as br
from app.performance.memory_monitor import BoundedLRU, MemoryMonitor
from app.performance.metrics import MetricsRegistry
from app.performance.profiler import Profiler
from app.performance.resource_monitor import CRITICAL, ELEVATED, NORMAL, ResourceMonitor, ResourceSample, disk_usage
from app.performance.settings import CACHE_CATEGORIES, MB, PerformanceSettings, resolve
from app.performance.synthetic import SIZES, SyntheticSpec, build_project


# ------------------------------------------------------------------ metrics / profiler
def test_metrics_aggregate_and_percentiles():
    m = MetricsRegistry()
    for i in range(1, 101):
        m.record("op", i / 1000.0, ok=(i != 7))
    st = m.op("op")
    assert st["count"] == 100 and st["errors"] == 1 and st["max_s"] == 0.1 and st["min_s"] == 0.001
    assert 0.045 <= st["p50_s"] <= 0.055 and 0.09 <= st["p95_s"] <= 0.1


def test_metrics_disabled_records_nothing_and_ring_is_bounded():
    m = MetricsRegistry(enabled=False)
    m.record("x", 1.0)
    m.incr("c")
    m.gauge("g", 3)
    assert m.op("x") is None and m.counter("c") == 0 and m.snapshot()["gauges"] == {}
    m.enabled = True
    for _ in range(5000):
        m.record("y", 0.001)
    assert m.op("y")["count"] == 5000 and len(m._ops["y"].recent) == 256


def test_cache_hit_rates_and_thread_safety():
    m = MetricsRegistry()

    def work():
        for _ in range(500):
            m.cache_hit("thumbnails")
            m.cache_miss("thumbnails")
            m.record("t", 0.001)

    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    r = m.hit_rates()["thumbnails"]
    assert r["hits"] == 4000 and r["misses"] == 4000 and r["rate"] == 0.5 and m.op("t")["count"] == 4000


def test_timer_records_failures_and_slow_log_is_capped(caplog):
    p = Profiler(slow_threshold_s=0.0001)
    with pytest.raises(ValueError):
        with p.timer("boom"):
            time.sleep(0.001)
            raise ValueError
    assert p.metrics.op("boom")["errors"] == 1

    @p.timed("deco")
    def f(x):
        time.sleep(0.0005)
        return x * 2

    assert [f(i) for i in range(30)][-1] == 58
    assert p.metrics.op("deco")["count"] == 30 and p._slow_logged["deco"] == 20  # logged at most 20 times


def test_disabled_profiler_does_not_read_the_clock(monkeypatch):
    p = Profiler(enabled=False)
    calls = []
    monkeypatch.setattr("app.performance.operation_timer.time.perf_counter", lambda: calls.append(1) or 0.0)
    with p.timer("a"):
        pass
    assert calls == [] and p.metrics.op("a") is None


# ------------------------------------------------------------------ resources
def test_resource_sample_has_core_fields_and_pressure_levels(tmp_path):
    mon = ResourceMonitor(tmp_path)
    s = mon.sample()
    assert s.cpu_count >= 1 and s.threads >= 1 and s.disk_free_bytes is not None
    assert mon.pressure(s)["overall"] in (NORMAL, ELEVATED, CRITICAL)
    low = ResourceSample(0, 1, 1000, 30, 4, 0.1, None, 10 * MB, 100 * MB * 1000, 1)  # 97% memory used, 10 MB disk free
    p = mon.pressure(low)
    assert p["memory"] == CRITICAL and p["disk"] == CRITICAL and p["overall"] == CRITICAL
    unknown = ResourceSample(0, None, None, None, 4, None, None, None, None, 1)
    assert mon.pressure(unknown)["overall"] == NORMAL  # unknown probes never stall the app


def test_disk_usage_walks_up_to_an_existing_parent(tmp_path):
    free, total = disk_usage(tmp_path / "a" / "b" / "missing")
    assert free and total and free <= total


def test_monitor_background_sampling_starts_and_stops(tmp_path):
    mon = ResourceMonitor(tmp_path)
    mon.start(interval=0.02)
    deadline = time.monotonic() + 3
    while len(mon.history()) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    mon.stop()
    n = len(mon.history())
    assert n >= 3
    time.sleep(0.1)
    assert len(mon.history()) == n and not any(t.name == "resource-monitor" and t.is_alive() for t in threading.enumerate())


def test_memory_monitor_growth_and_watch():
    mm = MemoryMonitor()
    assert mm.mark_baseline() is not None
    with mm.watch() as w:
        junk = bytearray(30 * MB)
        junk[::4096] = b"x" * len(junk[::4096])
    assert w["delta"] is not None and w["delta"] > 10 * MB
    del junk
    assert mm.growth_bytes() is not None


# ------------------------------------------------------------------ bounded LRU
def test_lru_evicts_by_items_and_weight_and_calls_release():
    released = []
    c = BoundedLRU(max_items=3, on_evict=lambda k, v: released.append(k))
    for i in range(5):
        c.put(i, i)
    assert list(c._d) == [2, 3, 4] and released == [0, 1] and c.stats()["evictions"] == 2
    c.get(2)
    c.put(5, 5)
    assert 3 not in c and 2 in c  # 2 was refreshed

    w = BoundedLRU(max_weight=100, weigher=len)
    w.put("a", "x" * 60)
    w.put("b", "y" * 60)
    assert "a" not in w and w.weight == 60
    w.put("huge", "z" * 500)  # a single oversized item is kept (never an empty cache) but evicts the rest
    assert list(w._d) == ["huge"]


def test_lru_discard_where_resize_and_failing_release_hook():
    def bad(_k, _v):
        raise RuntimeError("release failed")

    c = BoundedLRU(max_items=10, on_evict=bad)
    for i in range(6):
        c.put(("a" if i < 3 else "b", i), i)
    assert c.discard_where(lambda k: k[0] == "a") == 3 and len(c) == 3  # a failing hook never breaks the cache
    c.resize(max_items=1)
    assert len(c) == 1
    c.clear()
    assert len(c) == 0 and c.weight == 0


def test_lru_thread_safety():
    c = BoundedLRU(max_items=50)

    def work(n):
        for i in range(2000):
            c.put((n, i % 80), i)
            c.get((n, (i * 7) % 80))

    ts = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(c) <= 50


# ------------------------------------------------------------------ settings
def test_settings_sanitize_roundtrip_and_overrides():
    s = PerformanceSettings.from_dict({"profile": "turbo", "preview_quality": "high", "cache_limit_mb": -5, "category_limits_mb": {"thumbnails": 10, "nope": 5, "proxies": -1}, "extra": 1})
    assert s.profile == "balanced" and s.preview_quality == "high" and s.cache_limit_mb == 0 and s.category_limits_mb == {"thumbnails": 10}
    assert PerformanceSettings.from_dict(s.to_dict()) == s
    o = s.with_overrides({"profile": "performance", "unknown": 1, "render_backend": "bogus"})
    assert o.profile == "performance" and o.render_backend == "auto" and o.preview_quality == "high"
    assert PerformanceSettings.from_dict(None) == PerformanceSettings()


def test_resolve_scales_with_machine_and_never_uses_everything():
    big = resolve(PerformanceSettings(profile="performance"), cpu_count=16, ram_bytes=64 * 1024 * MB, disk_free_bytes=2000 * 1024 * MB)
    small = resolve(PerformanceSettings(profile="power_saver"), cpu_count=2, ram_bytes=4 * 1024 * MB, disk_free_bytes=20 * 1024 * MB)
    assert big.background_workers == 12 < 16 and big.memory_cache_bytes <= 0.25 * 64 * 1024 * MB
    assert small.background_workers == 1 and small.thumbnail_concurrency == 1 and not small.idle_work and not small.background_proxy
    assert small.cache_total_bytes <= 0.9 * 20 * 1024 * MB and all(v <= small.cache_total_bytes for v in small.category_bytes.values())
    assert set(big.category_bytes) == set(CACHE_CATEGORIES) and "never reduced" in big.note
    explicit = resolve(PerformanceSettings(max_background_workers=64, cache_limit_mb=100), cpu_count=4, ram_bytes=None, disk_free_bytes=None)
    assert explicit.background_workers == 4 and explicit.cache_total_bytes == 100 * MB  # an override still respects the CPU count
    assert resolve(PerformanceSettings(), cpu_count=0, ram_bytes=0, disk_free_bytes=0).background_workers >= 1


# ------------------------------------------------------------------ report
def test_report_ranks_flags_and_sanitizes(tmp_path):
    p = Profiler()
    for _ in range(30):
        p.metrics.record("timeline.paint", 0.3)
    p.metrics.record("project.open", 0.01)
    for _ in range(40):
        p.cache_miss("thumbnails")
    p.cache_hit("thumbnails")
    rep = br.build_report(p, ResourceMonitor(tmp_path), settings={"api_key": "sk-123", "path": str(tmp_path.home() / "x")}, hardware={"summary": "4 cores"}, jobs={"failed_recent": 5})
    assert rep["bottlenecks"][0]["operation"] == "timeline.paint" and rep["bottlenecks"][0]["over_budget"]
    text = " ".join(rep["findings"])
    assert "timeline.paint" in text and "thumbnails" in text and "background jobs failed" in text
    assert rep["settings"]["api_key"] == "***" and rep["settings"]["path"].startswith("~")
    assert "Slowest operations" in br.render_text(rep)
    out = br.export_report(rep, tmp_path / "r" / "report.json")
    assert json.loads(out.read_text())["bottlenecks"]
    assert br.export_report(rep, tmp_path / "report.txt", as_text=True).read_text().startswith("Performance report")


# ------------------------------------------------------------------ synthetic projects / benchmarks
def test_synthetic_projects_are_valid_deterministic_and_scale(tmp_path):
    a = build_project(tmp_path / "a", SIZES["small"])
    b = build_project(tmp_path / "b", SIZES["small"])
    assert a.summary() == b.summary() and a.project.to_document()["scenes"] == b.project.to_document()["scenes"]
    a.project.validate()
    assert a.clips == len(a.project.timeline.all_clips()) and a.words == len(a.project.transcription.transcript.words)
    big = build_project(tmp_path / "c", SyntheticSpec(scenes=200, extra_tracks=3, keyframes_per_visual=12))
    assert len(big.project.timeline.tracks) == 12 and big.clips > a.clips * 8
    kinds = {c.kind for c in big.project.timeline.all_clips()}
    assert {"media", "caption", "text", "graphic"} <= kinds


def test_synthetic_project_saves_and_reopens_identically(tmp_path):
    from app.core.events import EventBus
    from app.project.project_manager import ProjectManager

    sp = build_project(tmp_path / "p", 30, real_files=True)
    pm = ProjectManager(EventBus())
    pm.current = sp.project
    pm.save(sp.project)
    again = ProjectManager.load_file(sp.project.paths.project_file, sp.project.root)
    assert again.to_document()["timeline"] == sp.project.to_document()["timeline"] and len(again.scenes) == 30


def test_benchmark_suite_smoke_and_compare(tmp_path):
    from app.performance import benchmarks

    res = benchmarks.run_suite(["small"], repeat=1, workspace=True, ui=False, root=tmp_path)
    m = res["results"]["small"]["metrics"]
    for k in ("project.load_s", "project.save_s", "timeline.model_build_s", "qc.snapshot_s", "workspace.open_project_s", "workspace.jobs_created_on_open", "memory.growth_mb"):
        assert k in m
    cmp = benchmarks.compare(res, res)
    assert all(abs(v - 1.0) < 1e-6 for v in cmp["small"].values())


def test_from_plain_type_hint_plan_is_cached_and_equivalent():
    from app.analysis.models import Scene
    from app.core import serialization as ser
    from app.core.serialization import from_plain, to_plain
    from app.performance.synthetic import build_project
    import tempfile
    from pathlib import Path

    sp = build_project(Path(tempfile.mkdtemp()), 5)
    scene = sp.project.scenes[0]
    plain = to_plain(scene)
    ser._PLAN.pop(Scene, None)
    first = from_plain(Scene, plain)
    assert Scene in ser._PLAN
    assert to_plain(first) == plain == to_plain(from_plain(Scene, plain))
