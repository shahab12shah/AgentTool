"""Phase 9 acceptance: the whole performance workflow on a large project and on a real renderable project, through the services the UI uses.

launch -> hardware capabilities -> open a 500-scene project -> timeline -> browse a large media library -> thumbnails (generated, then reused) -> proxies (generated, then reused)
-> edit -> only the relevant cache entries go stale -> resources -> save -> close -> reopen -> timeline and decisions identical -> incremental QC == full QC -> preview sections
-> render -> validate.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from app.jobs.job import Priority
from app.performance.change_tracker import scene_layout_dep
from app.performance.synthetic import SyntheticSpec, build_project
from app.project.project_manager import ProjectManager
from app.rendering.models import RenderStatus
from app.services.workspace import Workspace
from app.storage.paths import AppPaths
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import make_image, make_video, write_tone_wav
from app.tests.render_helpers import wait_job

pytestmark = needs_ffmpeg


def until(cond, timeout: float = 120.0, step: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


@pytest.fixture(scope="module")
def pool(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("pool")
    return {"image": make_image(d / "i.png", "testsrc", "320x180"), "video": make_video(d / "v.mp4", 2.0, "testsrc2", "320x180"), "audio": write_tone_wav(d / "a.wav", 3.0, 0.3, 330.0, 48000)}


def new_ws(tmp_path: Path, name: str) -> Workspace:
    ws = Workspace(AppPaths(tmp_path / name / "cfg", tmp_path / name / "data"))
    ws.performance.start()
    return ws


def open_ws(ws: Workspace, root: Path):
    t = time.perf_counter()
    jobs_before = len(ws.jobs.jobs())
    p = ws.open_project(root)
    return p, time.perf_counter() - t, len(ws.jobs.jobs()) - jobs_before


@pytest.fixture
def big(tmp_path, pool):
    """A saved 500-scene project (about 2,300 clips, 750 assets incl. declared-4K video) in a folder whose name has spaces and non-ASCII characters."""
    root = tmp_path / "Large Project ünï 日本"
    sp = build_project(root, SyntheticSpec(scenes=500, fraction_4k=0.3), media_pool=pool)
    pm = ProjectManager(__import__("app.core.events", fromlist=["EventBus"]).EventBus())
    pm.current = sp.project
    pm.save(sp.project)
    return root, sp


def test_the_large_project_workflow(tmp_path, big, monkeypatch):
    root, sp = big
    ws = new_ws(tmp_path, "w1")
    try:
        # ---- 1. launch + hardware capabilities (a background job; startup never waits for it) ----
        job = ws.performance.detect_hardware_async()
        assert job is not None and until(lambda: job.status.is_terminal)
        hw = ws.performance.hardware_summary()
        assert hw["text"] and ws.performance.hardware.get_recommended_profile(allow_detect=False)["profile"] in ("power_saver", "balanced", "performance")

        # ---- 2. open the large project: nothing expensive starts by itself ----
        p, secs, jobs_created = open_ws(ws, root)
        assert len(p.scenes) == 500 and len(p.timeline.all_clips()) == sp.clips
        assert jobs_created <= 8, f"opening created {jobs_created} jobs (it used to create one per asset)"
        assert not ws.render.proxies.records()  # no proxies are generated just because a project opened
        assert ws.qc.running is False and not p.qc_runs  # nor QC
        assert ws.performance.cache is not None and (root / "cache").is_dir()

        # ---- 3. timeline model answers like brute force ----
        tl = p.timeline
        clips = tl.all_clips()
        for c in clips[::97]:
            assert tl.find_clip(c.id)[1] is c
        assert tl.duration == max(c.timeline_end for c in clips)

        # ---- 4. browse the library: visible assets first, the rest at idle priority ----
        assets = p.assets.all()
        visible = [a.id for a in assets[:30]]
        ws.media.set_visible_assets(visible)
        assert until(lambda: all(ws.media.thumbnail_state(i) == "ready" for i in visible))
        ws.media.request_thumbnails([a.id for a in assets], Priority.LOW)
        assert until(lambda: all(ws.media.thumbnail_state(a.id) == "ready" for a in assets), 240)
        stats = ws.media.thumbnail_stats()
        assert stats.get("max_running", 1) <= ws.performance.limits().thumbnail_concurrency

        # ---- 5. proxies for two declared-4K videos (the real files are tiny); reuse; originals untouched ----
        vids = [a for a in assets if a.type.value == "video" and (a.width or 0) >= 3840][:2]
        assert len(vids) == 2
        before = {a.id: p.asset_path(a).stat().st_mtime_ns for a in vids}
        jobs = ws.render.proxies.generate([a.id for a in vids], "540p", only_large=False)
        assert len(jobs) == 2 and until(lambda: all(j.status.is_terminal for j in jobs))
        assert all(ws.render.proxies.record(a.id).proxy_status == "READY" for a in vids)
        assert ws.render.proxies.generate([a.id for a in vids], "540p", only_large=False) == []  # valid proxies are reused
        assert {a.id: p.asset_path(a).stat().st_mtime_ns for a in vids} == before
        assert ws.render.proxies.status()["total_bytes"] > 0

        # ---- 6. edit one scene: only that scene (and its neighbours) are affected; unrelated cache entries still hit ----
        ws.changes.consume()
        cache = ws.performance.cache
        s_edit, s_far = p.scenes[250], p.scenes[10]
        cache.put("probe-edit", {"v": 1}, category="analysis", deps={scene_layout_dep(s_edit.id): "1"})
        cache.put("probe-far", {"v": 1}, category="analysis", deps={scene_layout_dep(s_far.id): "1"})
        vis = next(c for c in clips if c.scene_id == s_edit.id and c.kind == "media")
        ws.timeline.set_clip_properties(vis.id, opacity=0.5)
        cs = ws.changes.consume()
        assert not cs.all_scenes and s_edit.id in cs.scene_ids and s_far.id not in cs.scene_ids and len(cs.scene_ids) <= 6
        assert cache.get("probe-edit", deps={scene_layout_dep(s_edit.id): "1"}) is None  # the edited scene's entry went stale
        assert cache.get("probe-far", deps={scene_layout_dep(s_far.id): "1"}) is not None  # unrelated results stay reusable
        ws.undo()
        assert vis.opacity == 1.0

        # ---- 7. resources are visible ----
        diag = ws.performance.diagnostics()
        assert diag["resources"]["process_rss_bytes"] and "jobs" in diag and diag["cache"]["available"]
        assert ws.performance.cache_stats()["categories"]["thumbnails"]["entries"] >= len(assets) * 0.9

        # ---- 8. performance override for this project only, then save / close ----
        ws.performance.set_project_overrides({"preview_quality": "draft"})
        before_tl = json.dumps(p.timeline.to_dict(), sort_keys=True)
        decisions = json.dumps(p.to_document()["scenes"], sort_keys=True)
        ws.projects.save()
        ws.close_project()
    finally:
        ws.shutdown()
    assert not [t for t in threading.enumerate() if t.name.startswith("job") and t.is_alive()], "worker threads leaked after shutdown"

    # ---- 9. reopen in a fresh application: identical project, caches reused (no FFmpeg for thumbnails), override remembered ----
    calls: list[list[str]] = []
    import app.media.thumbnails as th

    real = th.run_cancellable
    monkeypatch.setattr(th, "run_cancellable", lambda cmd, *a, **k: (calls.append(cmd), real(cmd, *a, **k))[1])
    ws2 = new_ws(tmp_path, "w2")
    try:
        p2, _secs, jobs2 = open_ws(ws2, root)
        assert jobs2 <= 8
        assert json.dumps(p2.timeline.to_dict(), sort_keys=True) == before_tl and json.dumps(p2.to_document()["scenes"], sort_keys=True) == decisions
        assert ws2.performance.settings().preview_quality == "draft" and ws2.performance.global_settings().preview_quality == "balanced"
        ws2.media.request_thumbnails([a.id for a in p2.assets.all()], Priority.LOW)
        assert until(lambda: ws2.media.thumbnail_stats().get("pending", 0) == 0)
        assert calls == [], f"{len(calls)} FFmpeg thumbnail runs after reopening (valid thumbnails must be reused): {[c[c.index('-i') + 1] if '-i' in c else c for c in calls]}"
        assert all(ws2.render.proxies.record(a.id).proxy_status == "READY" for a in [x for x in p2.assets.all() if x.id in ws2.render.proxies.records()])
        # deleting the disposable cache never damages the project: it is rebuilt
        ws2.performance.clear_rebuildable_cache()
        assert len(p2.scenes) == 500 and all(Path(p2.asset_path(a)).is_file() for a in p2.assets.all())
    finally:
        ws2.shutdown()


def test_cpu_only_machine_keeps_everything_available(tmp_path, monkeypatch):
    """Scenario E: no GPU, no hardware encoder -> detection reports it, hardware is not offered, the CPU path is selected and the app works."""
    ws = new_ws(tmp_path, "cpu")
    try:
        hw = ws.render.engine.hardware
        monkeypatch.setattr(hw, "detect_gpu", lambda: [])
        monkeypatch.setattr(hw, "known_hardware_encoders", lambda: {})
        monkeypatch.setattr(hw, "tested_hardware_encoders", lambda candidates: {c: False for c in candidates})
        prof = hw.get_recommended_profile(allow_detect=False)
        assert prof["render_backend"] in ("auto", "cpu") and prof["render_backend"] != "hardware"
        ws.new_project("Cpu", tmp_path / "projects")
        s = ws.project.render_settings
        assert s.hardware_acceleration in ("auto", "cpu")
        ws.performance.update_global(ws.performance.global_settings().__class__.from_dict({"render_backend": "cpu"}))
        assert ws.project.render_settings.hardware_acceleration == "cpu"  # the preference reaches the export settings (undoable)
        ws.undo()
        assert ws.project.render_settings.hardware_acceleration == s.hardware_acceleration
    finally:
        ws.shutdown()


def test_demo_project_incremental_qc_preview_and_render(render_ws, tmp_path):
    """Scenarios D/F on a real renderable project: incremental QC == full QC, only the edited preview sections go stale, the export reads originals and validates."""
    from app.qc.tests.qc_helpers import add_scene

    ws = render_ws
    p = ws.project
    for t in p.timeline.tracks:  # consistent source ranges, as the Phase 8 acceptance fixture prepares them
        for c in t.clips:
            if c.kind == "media" and t.kind.value in ("video", "image"):
                c.source_out = c.source_in + c.duration * c.speed
    for i in range(4):  # the demo has no scenes: give it four, so the preview is cut into scene-aligned sections
        add_scene(p, 2.0 * i, 2.0 * (i + 1), "")

    def run(**kw):
        j = ws.qc.run_full_qc(**kw)
        assert wait_job_qc(ws, j)
        return p.qc_runs[-1]

    def wait_job_qc(w, j):
        return w.jobs.wait_idle(180) and j.status.value == "COMPLETED"

    run()
    assert ws.qc.qc_staleness()["stale"] is False
    # a proxy exists for the biggest video, but the export must keep using originals
    main = next(a for a in p.assets.all() if a.name == "main.mp4")
    [j] = ws.render.proxies.generate([main.id], "540p", only_large=False)
    assert until(lambda: j.status.is_terminal) and ws.render.proxies.record(main.id).proxy_status == "READY"
    assert p.render_settings.use_proxies is False

    # preview: render once, edit a clip in the middle, plan again -> only the affected sections are stale
    pr = ws.render.build_preview("draft")
    assert ws.jobs.wait_idle(240) and pr.status.value == "COMPLETED"
    full = ws.render.preview_plan("draft")
    assert full.stale_sections == [] and full.cached_sections == len(full.sections)
    still = next(c for c in p.timeline.get_track("track_v3").clips)
    ws.timeline.set_clip_properties(still.id, opacity=0.4)
    plan = ws.render.preview_plan("draft")
    assert 0 < len(plan.stale_sections) < len(plan.sections), (len(plan.stale_sections), len(plan.sections))

    # incremental QC after the edit equals a forced full run
    assert ws.qc.qc_staleness()["stale"] is True
    j = ws.qc.run_incremental_qc()
    assert j is not None and wait_job_qc(ws, j)
    inc = p.qc_runs[-1]
    inc_issues = sorted((i.code, i.fingerprint, i.severity.value) for i in p.qc_issues)
    inc_score = p.qc_scores.overall
    full_run = run(force=True)
    assert sorted((i.code, i.fingerprint, i.severity.value) for i in p.qc_issues) == inc_issues and p.qc_scores.overall == inc_score
    assert inc["content_hash"] == full_run["content_hash"]

    # render a test export with the AUTO backend and validate it
    job = wait_job(ws.render.start_export(settings=None, kind="export"))
    assert job.status is RenderStatus.COMPLETED
    out = Path(job.output_path) if getattr(job, "output_path", None) else None
    assert out is not None and out.is_file() and out.stat().st_size > 1000
    probe = ws.render.engine.probe.probe(out, use_cache=False)
    assert abs((probe.duration or 0) - p.timeline.duration) < 0.6 and (probe.width, probe.height) == (1920, 1080)
