"""Phase 9 proxies and probe persistence: policy decisions, disk validation, automatic mode, storage report, clean-up, regenerate, exports use originals."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.events import EventBus, Topics
from app.jobs.job import Priority
from app.performance.proxy_policy import BITRATE_KBPS, ProxyPolicy, estimate_bytes
from app.performance.settings import PerformanceSettings
from app.performance.synthetic import SyntheticSpec, build_project
from app.project.project_manager import ProjectManager
from app.rendering.errors import RenderError
from app.rendering.ffmpeg_service import ExecResult
from app.services.workspace import Workspace
from app.storage.paths import AppPaths

GB = 1024 ** 3


def eventually(cond, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


class FakeEncoder:
    """Replaces FFmpegService.run_progress / probe for the proxy encode: writes a small 'proxy', can block (cancel test) and records concurrency."""

    def __init__(self, ws: Workspace) -> None:
        self.calls: list[list[str]] = []
        self.gate: threading.Event | None = None
        self.lock = threading.Lock()
        self.now = self.peak = 0
        ws.render.proxies.ff.run_progress = self.run  # type: ignore[method-assign]
        ws.render.proxies.probe.probe = lambda path, use_cache=True: SimpleNamespace(width=960, height=540)  # type: ignore[method-assign]

    def run(self, args, *, on_progress=None, cancel=None, on_line=None, stall_timeout=300.0, cwd=None):
        with self.lock:
            self.calls.append(args)
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            if self.gate is not None:
                while not self.gate.wait(0.01):
                    if cancel is not None and cancel.is_set():
                        Path(args[-1]).write_bytes(b"partial")
                        return ExecResult(-1, cancelled=True)
            Path(args[-1]).write_bytes(b"\0" * 5000)
            return ExecResult(0)
        finally:
            with self.lock:
                self.now -= 1


@pytest.fixture
def syn(tmp_path):
    sp = build_project(tmp_path / "proj", SyntheticSpec(scenes=12, assets_per_scene=1.0, fraction_images=0.2, fraction_4k=0.5), real_files=True)
    ProjectManager(EventBus()).save(sp.project)
    return sp


def new_ws(tmp_path, name="h") -> Workspace:
    ws = Workspace(AppPaths(tmp_path / name / "cfg", tmp_path / name / "data"))
    from app.performance.media_wiring import wire_media_performance

    wire_media_performance(ws)
    ws.performance.monitor.thresholds["load_elevated"] = 1e9  # a busy shared machine must not change what these tests check
    return ws


@pytest.fixture
def ws(tmp_path, syn):
    w = new_ws(tmp_path)
    w.open_project(syn.project.root)
    w.jobs.wait_idle(30)
    w.enc = FakeEncoder(w)
    yield w
    w.shutdown()


def videos(ws):
    return [a for a in ws.project.assets.all() if a.type.value == "video"]


def set_policy(ws, **kw):
    ws.performance.update_global(PerformanceSettings(**kw))


# ======================================================================== policy decisions (pure)
def test_policy_recommends_by_resolution_codec_and_timeline_complexity(syn):
    p = syn.project
    pol = ProxyPolicy(lambda _p: 100 * GB)
    vids = [a for a in p.assets.all() if a.type.value == "video"]
    fourk = next(a for a in vids if a.width == 3840)
    hd = next(a for a in vids if a.width == 1920)
    rec = pol.recommend(p, vids, PerformanceSettings(proxy_policy="automatic", proxy_profile="balanced"))
    d = {x.asset_id: x for x in rec.decisions}
    assert rec.resolution == "720p" and d[fourk.id].proxy and any("4K" in r for r in d[fourk.id].reasons)
    assert not d[hd.id].proxy and "smoothly" in d[hd.id].skipped  # plain FullHD h264 on a non-busy timeline needs none
    perf = pol.recommend(p, vids, PerformanceSettings(proxy_policy="automatic", proxy_profile="performance"))
    assert perf.resolution == "540p" and all(x.proxy for x in perf.decisions) and any("performance profile" in r for r in perf.why(hd.id))
    qual = pol.recommend(p, vids, PerformanceSettings(proxy_policy="automatic", proxy_profile="quality"))
    assert qual.resolution == "1080p" and hd.id not in qual.asset_ids and fourk.id in qual.asset_ids
    hd.codec = "hevc"
    hevc = pol.recommend(p, [hd], PerformanceSettings(proxy_policy="automatic", proxy_profile="balanced"), {"decoders": {"working": {}}})
    assert hevc.decisions[0].proxy and "HEVC" in " ".join(hevc.decisions[0].reasons) and "no hardware decoding" in " ".join(hevc.decisions[0].reasons)
    hd.codec = "h264"
    small = pol.recommend(p, vids, PerformanceSettings(), resolution="1080p")
    assert small.resolution == "1080p"
    hd.extra = {"probe": {"has_alpha": True}}
    assert "transparency" in pol.recommend(p, [hd], PerformanceSettings(proxy_profile="performance")).decisions[0].skipped  # a proxy would lose the alpha channel


def test_a_busy_timeline_makes_full_hd_sources_worth_a_proxy(syn):
    p = syn.project
    pol = ProxyPolicy(lambda _p: 100 * GB)
    from app.performance.proxy_policy import peak_layers

    base = peak_layers(p)
    hd = next(a for a in p.assets.all() if a.type.value == "video" and a.width == 1920)
    from app.timeline.clip import Clip
    from app.timeline.timeline import new_clip_id

    for tid in ("track_v1", "track_v2", "track_v3"):  # three more layers on top of the same second
        p.timeline.get_track(tid).clips.append(Clip(new_clip_id(), tid, hd.id, 1000.0, 5.0, source_out=5.0))
    assert peak_layers(p) >= 3 and peak_layers(p) > base
    rec = pol.recommend(p, [hd], PerformanceSettings(proxy_policy="automatic", proxy_profile="balanced"))
    assert rec.decisions[0].proxy and any("busy timeline" in r for r in rec.decisions[0].reasons)


def test_the_disk_estimate_is_duration_times_bitrate_and_a_low_disk_refuses_clearly():
    assert estimate_bytes(100.0, "720p") == int(100 * BITRATE_KBPS["720p"] * 1000 / 8)
    assert estimate_bytes(None, "540p") > 0  # unknown duration: a documented default, not zero
    big = int(10 * GB)
    low = ProxyPolicy(lambda _p: 4 * GB).check_disk(big, "/x")
    assert not low.ok and "Not enough free disk space" in low.message and "GB" in low.message
    plenty = ProxyPolicy(lambda _p: 500 * GB).check_disk(big, "/x")
    assert plenty.ok and not plenty.warning
    tight = ProxyPolicy(lambda _p: 20 * GB).check_disk(big, "/x")
    assert tight.ok and "need about" in tight.warning
    assert ProxyPolicy(lambda _p: None).check_disk(big, "/x").ok  # free space unknown: allowed, with a caveat
    over = ProxyPolicy(lambda _p: 500 * GB).check_disk(big, "/x", cache_limit_bytes=5 * GB, existing_proxy_bytes=GB)
    assert over.ok and over.over_cache_limit


# ======================================================================== ProxyManager: manual, off, disk validation
def test_manual_generation_refuses_when_it_would_fill_the_disk_and_creates_nothing(ws):
    mgr = ws.render.proxies
    mgr.policy = ProxyPolicy(lambda _p: GB // 2)
    jobs_before, ids = len(ws.jobs.jobs()), [a.id for a in videos(ws)][:3]
    with pytest.raises(RenderError) as e:
        mgr.generate(ids, "720p", only_large=False)
    assert e.value.kind == "disk_space" and "Not enough free disk space" in str(e.value)
    assert len(ws.jobs.jobs()) == jobs_before and mgr.records() == {} and ws.enc.calls == []


def test_off_policy_creates_nothing_and_previews_ignore_existing_proxies(ws):
    mgr = ws.render.proxies
    vid = videos(ws)[0]
    mgr.generate([vid.id], "720p", only_large=False)
    assert ws.jobs.wait_idle(30) and mgr.proxy_path_for(vid) is not None
    set_policy(ws, proxy_policy="off")
    assert mgr.proxy_path_for(vid) is None and mgr.refs()[vid.id].status == "NONE"  # nothing reads a proxy while they are off
    with pytest.raises(RenderError) as e:
        mgr.generate([videos(ws)[1].id], "720p", only_large=False)
    assert e.value.kind == "proxies_off"
    assert mgr.run_automatic()["state"] == "off" and len(ws.enc.calls) == 1
    set_policy(ws, proxy_policy="manual")
    assert mgr.proxy_path_for(vid) is not None  # switching back reuses the file that is still valid


def test_manual_policy_never_starts_proxies_on_its_own(ws):
    set_policy(ws, proxy_policy="manual")
    assert ws.render.proxies.run_automatic()["state"] == "manual" and ws.render.proxies.records() == {}
    ws.media.request_thumbnails([])
    assert ws.enc.calls == []


# ======================================================================== automatic mode
def test_automatic_policy_queues_low_priority_proxies_for_expensive_sources_with_their_reasons(ws):
    mgr = ws.render.proxies
    set_policy(ws, proxy_policy="automatic", proxy_profile="balanced")
    gate = ws.enc.gate = threading.Event()
    st = mgr.run_automatic()
    fourk = [a for a in videos(ws) if a.width == 3840]
    assert st["state"] == "queued" and st["queued"] == len(fourk) > 1
    eventually(lambda: ws.enc.calls)
    jobs = [j for j in ws.jobs.jobs() if j.type == "proxy.generate"]
    assert all(j.priority is Priority.LOW and j.resource == "proxy" and j.dedupe_key for j in jobs)
    assert ws.performance.limits().proxy_concurrency == 1
    gate.set()
    assert ws.jobs.wait_idle(30) and ws.enc.peak == 1  # the resource cap: one encode at a time
    assert {a.id for a in fourk} == {k for k, r in mgr.records().items() if r.proxy_status == "READY"}
    assert mgr.run_automatic()["state"] == "up_to_date" and len(ws.enc.calls) == len(fourk)  # nothing is made twice
    assert all(not ws.project.asset_path(a).read_bytes() != b"\0" * 64 for a in fourk)  # the originals were not touched


def test_automatic_policy_is_deferred_while_idle_work_is_off_or_the_machine_is_busy(ws, monkeypatch):
    mgr = ws.render.proxies
    set_policy(ws, proxy_policy="automatic", idle_cache_generation=False)
    st = mgr.run_automatic()
    assert st["state"] == "deferred" and "switched off" in st["deferred"] and ws.enc.calls == []
    set_policy(ws, proxy_policy="automatic")
    monkeypatch.setattr(mgr, "_pressure_normal", lambda: False)
    scheduled = []
    monkeypatch.setattr(mgr, "_schedule_auto", lambda d: scheduled.append(d))
    st = mgr.run_automatic()
    assert st["state"] == "deferred" and "busy" in st["deferred"] and scheduled and ws.enc.calls == []  # looked at again later
    monkeypatch.setattr(mgr, "_pressure_normal", lambda: True)
    assert mgr.run_automatic()["state"] == "queued"
    assert ws.jobs.wait_idle(30)


def test_automatic_policy_skips_with_a_visible_reason_when_the_disk_or_the_cache_limit_is_too_small(ws):
    mgr = ws.render.proxies
    msgs = []
    ws.bus.subscribe(Topics.STATUS, lambda t, p: msgs.append(p["message"]))
    set_policy(ws, proxy_policy="automatic")
    mgr.policy = ProxyPolicy(lambda _p: GB)  # 1 GB free: below the reserve
    st = mgr.run_automatic()
    assert st["state"] == "skipped" and st["skipped"] and "Not enough free disk space" in st["skipped"][0]["reason"] and ws.enc.calls == []
    mgr.run_automatic()
    assert len([m for m in msgs if "Automatic proxies were not created" in m]) == 1  # said once, never silently, never repeatedly
    mgr.policy = ProxyPolicy(lambda _p: 500 * GB)
    set_policy(ws, proxy_policy="automatic", category_limits_mb={"proxies": 1})
    st = mgr.run_automatic()
    assert st["state"] == "skipped" and "cache limit" in st["skipped"][0]["reason"] and ws.enc.calls == []


def test_a_failed_or_cancelled_proxy_is_not_retried_by_the_background_policy(ws):
    mgr = ws.render.proxies
    set_policy(ws, proxy_policy="automatic")
    ws.enc.gate = threading.Event()
    mgr.run_automatic()
    eventually(lambda: ws.enc.calls)
    assert mgr.cancel() >= 1
    ws.enc.gate.set()
    assert ws.jobs.wait_idle(30)
    calls = len(ws.enc.calls)
    assert any(r.proxy_status == "CANCELED" for r in mgr.records().values())
    mgr.run_automatic()
    assert ws.jobs.wait_idle(30)
    canceled = {k for k, r in mgr.records().items() if r.proxy_status == "CANCELED"}
    assert canceled and len(ws.enc.calls) - calls <= len(videos(ws))  # only never-attempted assets were queued; the cancelled ones wait for an explicit Retry
    assert all(mgr.records()[k].proxy_status == "CANCELED" for k in canceled)


# ======================================================================== storage, reuse, clean-up
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_storage_report_counts_bytes_by_status_orphans_and_partial_files(ws):
    mgr = ws.render.proxies
    a, b = videos(ws)[:2]
    mgr.generate([a.id, b.id], "720p", only_large=False)
    assert ws.jobs.wait_idle(30)
    folder = ws.project.root / "proxies"
    (folder / "stray_720p.mp4").write_bytes(b"x" * 100)
    (folder / f".{a.id}_540p.part.mp4").write_bytes(b"y" * 50)
    rep = mgr.storage_report()
    assert rep["by_status"]["READY"] == {"count": 2, "bytes": 10000} and len(rep["proxies"]) == 2
    assert [o["bytes"] for o in rep["orphans"]] == [100] and [x["bytes"] for x in rep["partial"]] == [50]
    assert rep["total_bytes"] == 10150 and rep["free_bytes"] and mgr.status()["total_bytes"] == 10150
    assert mgr.clean_orphans() == 2 and not (folder / "stray_720p.mp4").exists() and len(list(folder.glob("*.mp4"))) == 2


def test_the_same_asset_and_size_is_never_made_twice_and_a_size_change_replaces_the_old_file(ws):
    mgr = ws.render.proxies
    a = videos(ws)[0]
    ws.enc.gate = threading.Event()
    j1 = mgr.generate([a.id], "720p", only_large=False)
    j2 = mgr.generate([a.id], "720p", only_large=False)
    assert len(j1) == 1 and j2 == []
    ws.enc.gate.set()
    assert ws.jobs.wait_idle(30)
    assert mgr.generate([a.id], "720p", only_large=False) == []  # valid and the same size: reused
    folder = ws.project.root / "proxies"
    assert [f.name for f in folder.glob("*.mp4")] == [f"{a.id}_720p.mp4"]
    mgr.generate([a.id], "540p", only_large=False)
    assert ws.jobs.wait_idle(30)
    assert [f.name for f in folder.glob("*.mp4")] == [f"{a.id}_540p.mp4"] and mgr.record(a.id).proxy_resolution == "540p"  # one file per asset, the 720p one is gone
    assert mgr.invalidate_other_resolutions("540p") == 0
    assert mgr.invalidate_other_resolutions("1080p") == 1 and mgr.record(a.id).proxy_status == "STALE" and not list(folder.glob("*.mp4"))


def test_a_cancelled_encode_leaves_no_partial_file_and_the_original_untouched(ws):
    mgr = ws.render.proxies
    a = videos(ws)[0]
    orig = ws.project.asset_path(a)
    h = sha(orig)
    ws.enc.gate = threading.Event()
    mgr.generate([a.id], "720p", only_large=False)
    eventually(lambda: ws.enc.calls)
    assert mgr.cancel(a.id) == 1
    assert ws.jobs.wait_idle(30)
    assert mgr.record(a.id).proxy_status == "CANCELED" and not list((ws.project.root / "proxies").glob("*")) and sha(orig) == h
    ws.enc.gate = None
    assert mgr.retry(a.id) is not None and ws.jobs.wait_idle(30) and mgr.record(a.id).proxy_status == "READY"


def test_partial_files_from_a_crash_are_removed_when_the_project_opens(tmp_path, syn):
    folder = syn.project.root / "proxies"
    folder.mkdir(exist_ok=True)
    (folder / ".media_0001_720p.part.mp4").write_bytes(b"half")
    w = new_ws(tmp_path)
    w.open_project(syn.project.root)
    assert not list(folder.glob(".*part*"))
    w.shutdown()


def test_a_missing_or_evicted_proxy_file_means_regenerate(ws):
    mgr = ws.render.proxies
    a = videos(ws)[0]
    mgr.generate([a.id], "720p", only_large=False)
    assert ws.jobs.wait_idle(30)
    Path(mgr.record(a.id).proxy_path).unlink()  # the cache cleaned it up
    assert mgr.proxy_path_for(videos(ws)[0]) is None and mgr.refs()[a.id].status == "STALE"
    assert mgr.storage_report()["by_status"].get("STALE", {}).get("count") == 1
    assert len(mgr.generate([a.id], "720p", only_large=False)) == 1 and ws.jobs.wait_idle(30) and mgr.proxy_path_for(a) is not None
    cache = ws.performance.cache
    assert cache is not None and any(e.category == "proxies" for e in cache.entries())  # registered with the cache manager


def test_remove_selected_only_removes_regenerable_proxies_that_nothing_is_using(ws):
    mgr = ws.render.proxies
    a, b, c = videos(ws)[:3]
    mgr.generate([a.id, b.id, c.id], "720p", only_large=False)
    assert ws.jobs.wait_idle(30)
    ws.project.asset_path(b).rename(ws.project.asset_path(b).with_suffix(".gone"))
    pc = Path(mgr.record(c.id).proxy_path)
    mgr.in_use = lambda p: Path(p) == pc
    out = mgr.remove_selected([a.id, b.id, c.id, "media_none"])
    assert out["removed"] == [a.id] and out["freed_bytes"] == 5000
    assert "original is missing" in out["skipped"][b.id] and "in use" in out["skipped"][c.id] and out["skipped"]["media_none"] == "no proxy"
    assert Path(mgr.record(b.id).proxy_path).is_file() and pc.is_file() and mgr.record(a.id) is None
    mgr.in_use = None
    assert mgr.regenerate([a.id]) and ws.jobs.wait_idle(30) and mgr.record(a.id).proxy_status == "READY"


def test_per_project_proxy_policy_is_an_undoable_override(ws):
    set_policy(ws, proxy_policy="manual")
    ws.render.proxies.set_project_policy("automatic", "performance")
    s = ws.performance.settings()
    assert (s.proxy_policy, s.proxy_profile) == ("automatic", "performance") and ws.performance.global_settings().proxy_policy == "manual"
    ws.undo()
    assert ws.performance.settings().proxy_policy == "manual"


# ======================================================================== the export reads the originals
def test_export_inputs_are_the_originals_even_when_a_proxy_is_ready(ws):
    from app.rendering.compiler import TimelineCompiler
    from app.rendering.fonts import FontResolver
    from app.rendering.models import RenderSettings
    from app.rendering.planner import plan_chunks
    from app.rendering.sources import SourceSelector, prepare_assets

    mgr = ws.render.proxies
    used = {c.asset_id for t in ws.project.timeline.tracks for c in t.clips if c.asset_id}
    vids = [a for a in videos(ws) if a.id in used][:3]
    mgr.generate([a.id for a in vids], "720p", only_large=False)
    assert ws.jobs.wait_idle(30) and all(mgr.proxy_path_for(a) for a in vids)
    snap = ws.render.snapshot()
    assert all(snap.proxies[a.id].status == "READY" for a in vids) and snap.settings.use_proxies is False
    originals = {a.id: str(ws.project.asset_path(a)) for a in vids}
    sel = SourceSelector(snap, ws.render.engine.probe, "final", None, snap.settings.use_proxies)
    for a in vids:
        info = sel.resolve(snap.assets[a.id])
        assert info.path == originals[a.id] and info.path != snap.proxies[a.id].path
    assert sel.used_proxies == set()
    compiler = TimelineCompiler(snap, sel, FontResolver(""), 30)
    chunks = plan_chunks(snap, 30, 30.0, compiler.media_clips_in, None)
    inputs = {i for c in chunks for cc in [compiler.compile_chunk(c)] for i in [str(x) for x in getattr(cc, "inputs", [])] + [getattr(s, "path", "") for s in getattr(cc, "sources", [])]}
    assert not any("/proxies/" in i for i in inputs) and sel.used_proxies == set()
    assert RenderSettings().use_proxies is False  # the default
    explicit = SourceSelector(snap, ws.render.engine.probe, "final", None, True)  # only an explicit opt-in reads the proxy
    assert explicit.resolve(snap.assets[vids[0].id]).path == snap.proxies[vids[0].id].path
    del prepare_assets
