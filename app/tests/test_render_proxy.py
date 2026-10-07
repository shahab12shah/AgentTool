"""Phase 6: proxies (generate / use for preview / never for final), the preview engine and its cache, and media relinking."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import pytest

from app.analysis.models import Scene
from app.rendering.proxy import ProxyRecord
from app.rendering.relink import RelinkError
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import make_video
from app.tests.render_helpers import caption_clip, frame_at, media_clip, put, solid_image, text_clip, wait_job

pytestmark = needs_ffmpeg


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def proxy_ws(project_ws, tmp_path):
    """A project with a real 4K 30 fps clip and a small clip on the timeline."""
    ws = project_ws
    d = tmp_path / "px"
    d.mkdir()
    big = make_video(d / "big_4k.mp4", 3.0, "testsrc2", "3840x2160", 30)
    small = make_video(d / "small.mp4", 2.0, "testsrc", "640x360", 24)
    ws.media.import_files([big, small])
    assert ws.jobs.wait_idle(120)
    ws.by = {a.name: a for a in ws.project.assets.all()}
    put(ws, "track_v1", media_clip(ws.by["big_4k.mp4"], 0.0, 3.0))
    put(ws, "track_v2", media_clip(ws.by["small.mp4"], 1.0, 1.5, scale=0.4))
    put(ws, "track_v5", text_clip(0.3, 1.5, "PROXY TEST"))
    put(ws, "track_v6", caption_clip(0.5, ["Four", "kay", "source", "media"], 0.4))
    ws.project.scenes = [Scene("s1", "1", 0.0, 1.5), Scene("s2", "2", 1.5, 3.0)]
    ws.render.update_settings(resolution="480p", quality="draft", preset_id="custom")
    return ws


def settle_proxies(ws, timeout=120):
    assert ws.jobs.wait_idle(timeout)


# ================================================================ proxy generation and mapping
def test_proxy_for_a_4k_source_is_720p_with_the_same_timing_and_the_asset_id_mapping(proxy_ws):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    original = ws.project.asset_path(big)
    h_before = sha(original)
    assert [a.name for a in ws.render.proxies.candidates()] == ["big_4k.mp4"]  # the 640x360 clip needs none
    jobs = ws.render.proxies.generate()
    assert len(jobs) == 1 and ws.render.proxies.record(big.id).proxy_status == "QUEUED"
    settle_proxies(ws)
    rec = ws.render.proxies.record(big.id)
    assert isinstance(rec, ProxyRecord) and (rec.asset_id, rec.original_path, rec.proxy_status, rec.proxy_resolution) == (big.id, str(original), "READY", "720p")
    assert (rec.width, rec.height) == (1280, 720) and Path(rec.proxy_path).is_file() and Path(rec.proxy_path).parent == ws.project.root / "proxies" and rec.size_bytes > 0
    info = ws.render.engine.probe.probe(Path(rec.proxy_path))
    orig = ws.render.engine.probe.probe(original)
    assert (info.width, info.height) == (1280, 720) and info.fps == pytest.approx(orig.fps) and info.duration == pytest.approx(orig.duration, abs=0.05) and not info.has_audio
    assert sha(original) == h_before  # the original is never touched
    # the timeline still references the asset, not the proxy
    clip = ws.project.timeline.get_track("track_v1").clips[0]
    assert clip.asset_id == big.id and "proxy" not in str(clip.to_dict())
    assert ws.project.proxies[big.id]["proxy_status"] == "READY"  # persisted with the project
    # nothing to do the second time; regenerating replaces it
    assert ws.render.proxies.generate() == []
    assert ws.render.proxies.regenerate([big.id])
    settle_proxies(ws)
    assert ws.render.proxies.record(big.id).proxy_status == "READY"


def test_540p_and_1080p_proxy_sizes_and_unsupported_sizes(proxy_ws):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    ws.render.proxies.generate([big.id], "540p")
    settle_proxies(ws)
    assert (ws.render.proxies.record(big.id).width, ws.render.proxies.record(big.id).height) == (960, 540)
    ws.render.proxies.generate([big.id], "1080p", regenerate=True)
    settle_proxies(ws)
    assert ws.render.proxies.record(big.id).height == 1080
    from app.rendering.errors import RenderError

    with pytest.raises(RenderError, match="Unsupported proxy size"):
        ws.render.proxies.generate([big.id], "999p")


def test_proxies_can_be_cancelled_retried_and_deleted(proxy_ws):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    ws.render.proxies.generate([big.id])
    assert ws.render.proxies.cancel() == 1
    settle_proxies(ws)
    rec = ws.render.proxies.record(big.id)
    assert rec.proxy_status in ("CANCELED", "READY")
    assert not list((ws.project.root / "proxies").glob(".*part*"))
    if rec.proxy_status == "CANCELED":
        assert ws.render.proxies.retry(big.id) is not None
        settle_proxies(ws)
    assert ws.render.proxies.record(big.id).proxy_status == "READY"
    path = Path(ws.render.proxies.record(big.id).proxy_path)
    assert ws.render.proxies.delete() == 1 and not path.exists() and ws.render.proxies.record(big.id) is None and ws.project.asset_path(big).is_file()


def test_a_failed_proxy_is_recorded_with_its_reason_and_can_be_retried(proxy_ws):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    path = ws.project.asset_path(big)
    hidden = path.with_suffix(".hidden")
    path.rename(hidden)
    ws.render.proxies.generate([big.id])
    settle_proxies(ws)
    rec = ws.render.proxies.record(big.id)
    assert rec.proxy_status == "FAILED" and "missing" in rec.error.lower()
    hidden.rename(path)
    assert ws.render.proxies.retry(big.id) is not None
    settle_proxies(ws)
    assert ws.render.proxies.record(big.id).proxy_status == "READY"


# ================================================================ preview uses proxies, final never does
def test_editing_preview_reads_the_proxy_but_the_export_reads_the_original(proxy_ws):
    from app.preview.frames import FrameProvider

    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    original = ws.project.asset_path(big)
    fp = FrameProvider(lambda: ws.project, lambda: ws.settings.ffmpeg_path, lambda a: ws.render.proxies.proxy_path_for(a))
    assert fp.frame_path(big, 0.5) is not None and fp.last_source == original  # no proxy yet: the original
    (ws.project.root / "previews" / "frames").mkdir(exist_ok=True)
    ws.render.proxies.generate()
    settle_proxies(ws)
    proxy = Path(ws.render.proxies.record(big.id).proxy_path)
    assert fp.frame_path(big, 1.5) is not None and fp.last_source == proxy  # scrubbing now reads the 720p proxy
    # --- final export: original media, no proxy in any command
    job = wait_job(ws.render.start_export())
    assert job.status.value == "COMPLETED" and job.result.used_proxy_assets == [] and not job.record.proxy_used
    log = ws.render.read_log(job.id)
    assert str(original) in log and str(proxy) not in log
    info = ws.render.engine.probe.probe(Path(job.record.output_path))
    assert (info.width, info.height) == (854, 480)
    # --- the draft preview, in contrast, is made from the proxy (lower resolution, same timing)
    res = ws.render.preview.build(ws.render.snapshot(), "draft")
    assert res.uses_proxy == [big.id] and res.path.is_file() and ws.render.engine.probe.probe(res.path).duration == pytest.approx(3.0, abs=0.1)
    hq = ws.render.preview.build(ws.render.snapshot(), "high")
    assert hq.uses_proxy == [] and ws.render.engine.probe.probe(hq.path).height == 1080  # High Quality: original media


def test_preview_falls_back_to_the_original_when_a_proxy_is_missing_or_stale(proxy_ws):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    ws.render.proxies.generate()
    settle_proxies(ws)
    Path(ws.render.proxies.record(big.id).proxy_path).unlink()  # the proxy file vanished
    assert ws.render.proxies.proxy_path_for(big) is None and ws.render.snapshot().proxies[big.id].status != "READY"  # the vanished file is not offered
    res = ws.render.preview.build(ws.render.snapshot(), "realtime")
    assert res.uses_proxy == [] and res.path.is_file()  # still works: the original is used
    ws.render.proxies.regenerate([big.id])
    settle_proxies(ws)
    assert ws.render.proxies.proxy_path_for(big) is not None
    path = ws.project.asset_path(big)
    shutil.copy2(ws.project.asset_path(ws.by["small.mp4"]), path)  # the original was replaced by a different file
    assert ws.render.proxies.refresh_staleness() == 1 and ws.render.proxies.record(big.id).proxy_status == "STALE" and ws.render.proxies.proxy_path_for(big) is None


def test_original_missing_with_proxy_available_never_silently_uses_the_proxy_for_final(proxy_ws, tmp_path):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    ws.render.proxies.generate()
    settle_proxies(ws)
    path = ws.project.asset_path(big)
    moved = tmp_path / "gone" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    rep = ws.render.preflight()
    assert not rep.can_start and rep.proxy_only_assets == [big.id] and "proxy available" in " ".join(rep.item("visuals").details)
    from app.services.render_service import PreflightFailed

    with pytest.raises(PreflightFailed):
        ws.render.start_export()
    job = wait_job(ws.render.start_export(skip_preflight=True))  # even when forced, FFmpeg is not pointed at the proxy behind the user's back
    assert job.status.value == "FAILED" and job.error.kind == "missing_media" and big.id in job.error.asset_ids
    # the explicit choice: "Use Proxy Anyway"
    rep = ws.render.preflight(allow_proxy_assets={big.id})
    assert rep.can_start and rep.item("proxy_final").status == "warning"
    ok = wait_job(ws.render.start_export(allow_proxy_assets={big.id}))
    assert ok.status.value == "COMPLETED" and ok.result.used_proxy_assets == [big.id] and ok.record.proxy_used
    assert any("proxy" in w.lower() for w in ok.result.warnings) or ok.result.used_proxy_assets
    # or: locate the original again
    ws.render.relink.relink(big.id, moved)
    final = wait_job(ws.render.start_export())
    assert final.status.value == "COMPLETED" and final.result.used_proxy_assets == []


def test_exporting_from_proxies_is_an_explicit_setting(proxy_ws):
    ws = proxy_ws
    big = ws.by["big_4k.mp4"]
    ws.render.proxies.generate()
    settle_proxies(ws)
    ws.render.update_settings(use_proxies=True)
    assert ws.render.preflight().item("proxy_final").status == "warning"
    job = wait_job(ws.render.start_export())
    assert job.result.used_proxy_assets == [big.id] and job.record.proxy_used


# ================================================================ preview engine: cached sections and smart invalidation
def test_preview_sections_are_cached_per_scene_and_only_affected_ones_are_invalidated(proxy_ws):
    ws = proxy_ws
    put(ws, "track_v6", caption_clip(2.2, ["Only", "in", "scene", "two"], 0.3))  # a caption that lives entirely in scene 2
    ws.render.proxies.generate()
    settle_proxies(ws)
    pe = ws.render.preview
    snap = ws.render.snapshot()
    plan = pe.plan(snap, "draft")
    assert [s.scene_ids for s in plan.sections] == [["s1"], ["s2"]] and plan.cached_sections == 0 and len(plan.stale_sections) == 2
    first = pe.build(snap, "draft")
    assert first.rendered == 2 and first.reused == 0 and first.path.is_file()
    again = pe.plan(ws.render.snapshot(), "draft")
    assert again.cached_sections == 2 and again.audio_cached is False  # no audio on this timeline: nothing to cache
    # a caption edit in scene 2 invalidates scene 2 only
    cap = [c for c in ws.project.timeline.all_clips() if c.kind == "caption" and c.timeline_start > 2.0][0]
    cap.text["lines"], cap.text["text"] = ["EDITED IN SCENE TWO"], "EDITED IN SCENE TWO"
    plan2 = pe.plan(ws.render.snapshot(), "draft")
    assert [s.cached for s in plan2.sections] == [True, False]
    second = pe.build(ws.render.snapshot(), "draft")
    assert second.rendered == 1 and second.reused == 1  # scene 1 was not rendered again
    # a project-wide change (frame rate) invalidates everything
    ws.project.settings.fps = 24
    assert pe.plan(ws.render.snapshot(), "draft").cached_sections == 0
    assert pe.build(ws.render.snapshot(), "draft").rendered == 2
    # modes have their own caches; clearing removes them
    assert pe.plan(ws.render.snapshot(), "realtime").cached_sections == 0
    assert pe.cache_size() > 0 and pe.clear() > 0 and pe.plan(ws.render.snapshot(), "draft").cached_sections == 0


def test_a_scene_range_preview_renders_only_those_scenes_in_the_background(proxy_ws):
    ws = proxy_ws
    done = []
    job = ws.render.build_preview("realtime", ["s2"], on_done=done.append)
    assert ws.jobs.wait_idle(120) and job.status.value == "COMPLETED" and done
    res = done[0]
    assert (res.start, res.end) == (1.5, 3.0) and res.rendered == 1 and ws.render.engine.probe.probe(res.path).duration == pytest.approx(1.5, abs=0.1)
    assert ws.render.engine.probe.probe(res.path).height == 480 and "Draft" not in ws.render.preview_modes["realtime"].label


def test_preview_keeps_timing_accurate_for_cuts(project_ws, tmp_path):
    ws = project_ws
    d = tmp_path / "cut"
    d.mkdir()
    ws.media.import_files([solid_image(d / "red.png", "red"), solid_image(d / "blue.png", "blue")])
    ws.jobs.wait_idle(30)
    by = {a.name: a for a in ws.project.assets.all()}
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 2.0))
    put(ws, "track_v1", media_clip(by["blue.png"], 2.0, 1.0))
    res = ws.render.preview.build(ws.render.snapshot(), "realtime")
    from app.tests.render_helpers import BLUE, RED, near, px

    assert near(px(frame_at(res.path, n=59), 0.5, 0.5), RED, 30) and near(px(frame_at(res.path, n=60), 0.5, 0.5), BLUE, 30)


# ================================================================ relinking
def test_relink_candidates_are_ranked_and_weak_matches_need_confirmation(proxy_ws, tmp_path):
    ws = proxy_ws
    asset = ws.by["small.mp4"]
    path = ws.project.asset_path(asset)
    folder = tmp_path / "search"
    (folder / "deep" / "er").mkdir(parents=True)
    shutil.copy2(path, folder / "deep" / "er" / "renamed_copy.mp4")  # identical content, other name
    make_video(folder / "small.mp4", 5.0, "testsrc2")  # same name, different content
    make_video(folder / "unrelated.mp4", 2.5, "testsrc2", "640x360", 24)  # nothing in common except some media facts
    path.unlink()
    assert [a.id for a in ws.render.relink.missing()] == [asset.id]
    cands = ws.render.relink.find_candidates(asset, [folder])
    assert cands[0].path.name == "renamed_copy.mp4" and cands[0].exact and cands[0].score == 100.0 and cands[0].confident
    assert cands[1].path.name == "small.mp4" and not cands[1].exact and not cands[1].confident and "same file name" in cands[1].reasons
    assert all(c.path.name != "unrelated.mp4" for c in cands)  # metadata alone is never enough to suggest a file
    with pytest.raises(RelinkError, match="does not clearly match"):
        ws.render.relink.relink(asset.id, folder / "small.mp4")  # low confidence: refused without confirmation
    assert not ws.project.asset_path(asset).is_file()
    ws.render.relink.relink(asset.id, folder / "small.mp4", confirmed=True)
    assert ws.project.asset_path(asset) == folder / "small.mp4" and asset.link_mode == "reference" and asset.duration == pytest.approx(5.0, abs=0.1)
    ws.undo()  # relinking is an undoable edit
    assert not ws.project.asset_path(asset).is_file() and asset.link_mode == "copy"
    with pytest.raises(RelinkError, match="video file is needed"):
        ws.render.relink.relink(asset.id, solid_image(tmp_path / "x.png"), confirmed=True)
    with pytest.raises(RelinkError, match="does not exist"):
        ws.render.relink.relink(asset.id, tmp_path / "nope.mp4", confirmed=True)


def test_auto_relink_only_replaces_exact_content_matches(proxy_ws, tmp_path):
    ws = proxy_ws
    a, b = ws.by["small.mp4"], ws.by["big_4k.mp4"]
    folder = tmp_path / "found"
    folder.mkdir()
    shutil.copy2(ws.project.asset_path(a), folder / "elsewhere.mp4")
    make_video(folder / "big_4k.mp4", 3.0, "testsrc", "640x360", 30)  # same name as b, different content
    ws.project.asset_path(a).unlink()
    ws.project.asset_path(b).unlink()
    done = ws.render.relink.auto_relink([folder])
    assert list(done) == [a.id] and ws.project.asset_path(a) == folder / "elsewhere.mp4"
    assert not ws.project.asset_path(b).is_file()  # the same-name-different-content file was NOT used on a guess
    assert [m.id for m in ws.render.relink.missing()] == [b.id]
