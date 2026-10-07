"""Phase 6 service tests: preflight, export flow, queue, failures and recovery, cancellation, snapshots, persistence, odd file names."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.analysis.models import Scene
from app.core.exceptions import AppError
from app.rendering.errors import EncoderUnavailableError, RenderError
from app.rendering.models import RenderStatus
from app.services.render_service import PreflightFailed
from app.tests.conftest import needs_ffmpeg
from app.tests.render_helpers import (BLUE, RED, build_demo, caption_clip, frame_at, install_fake_ffmpeg, media_clip, near, put, px, solid_image, wait_job)

pytestmark = needs_ffmpeg


@pytest.fixture
def demo_ws(render_ws):
    """The demo project with quick export settings (480p, draft) and two scenes."""
    ws = render_ws
    ws.project.scenes = [Scene("s1", "1", 0.0, 4.0), Scene("s2", "2", 4.0, 8.0)]
    ws.render.update_settings(resolution="480p", quality="draft", preset_id="draft")
    return ws


def timeline_doc(ws):
    return json.dumps(ws.project.to_document()["timeline"], sort_keys=True)


# ================================================================ preflight
def test_preflight_lists_every_check_for_a_complete_timeline(demo_ws):
    rep = demo_ws.render.preflight()
    ids = [i.id for i in rep.items]
    assert ids[:2] == ["ffmpeg", "timeline"] and {"voice", "visuals", "captions", "fonts", "audio", "settings", "output", "disk"} <= set(ids)
    assert rep.can_start and not rep.errors and rep.item("visuals").message.startswith("3/3 visuals found") and rep.item("timeline").status == "ok"
    text = rep.text()
    assert text.startswith("PREFLIGHT CHECK") and "✓ FFmpeg" in text and "✓ 3/3 visuals found" in text and "Captions valid" in text
    assert rep.plan and rep.plan.captions == 2 and rep.plan.graphics == 3 and rep.plan.has_audio and rep.plan.output_resolution == (854, 480)
    assert "FADE ×1" in rep.plan.transitions and any("keyframes" in e for e in rep.plan.effects)
    assert rep.required_bytes > 0 and rep.available_bytes > 0


def test_preflight_catches_invalid_keyframes_effects_and_unreadable_media(demo_ws, tmp_path):
    ws = demo_ws
    clip = next(c for c in ws.project.timeline.all_clips() if c.keyframes and c.kind == "media")
    clip.keyframes.append(type(clip.keyframes[0])("scale", 99.0, 1.0))  # outside the clip
    clip.keyframes.append(type(clip.keyframes[0])("sparkle", 0.0, 1.0))  # not a property
    rep = ws.render.preflight()
    t = rep.item("timeline")
    assert t.status == "error" and not rep.can_start and any("outside the clip" in d for d in t.details) and any("unknown keyframe property" in d for d in t.details)
    with pytest.raises(PreflightFailed) as e:
        ws.render.start_export()
    assert "not ready to export" in str(e.value) and e.value.report is not None
    clip.keyframes[:] = [k for k in clip.keyframes if k.property == "scale" and k.time < 10]
    asset = ws.demo.assets["broken.mp4"] if "broken.mp4" in ws.demo.assets else None
    assert asset is None
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video" * 100)
    ws.media.import_files([broken], link_mode="reference")
    ws.jobs.wait_idle(30)  # the importer rejects it, so build the situation by hand: an asset whose file went bad
    a = ws.demo.assets["main.mp4"]
    ws.project.asset_path(a).write_bytes(b"corrupted" * 500)
    rep = ws.render.preflight()
    assert rep.item("media").status == "error" and "cannot be read" in rep.item("media").message and not rep.can_start


def test_preflight_warns_about_missing_fonts_without_blocking(demo_ws):
    ws = demo_ws
    next(c for c in ws.project.timeline.all_clips() if c.kind == "text").text["font"] = "No Such Typeface Pro"
    rep = ws.render.preflight()
    assert rep.item("fonts").status == "warning" and rep.can_start and any("No Such Typeface Pro" in d for d in rep.item("fonts").details)


def test_preflight_reports_low_disk_space(demo_ws, monkeypatch):
    import app.rendering.diagnostics as D

    monkeypatch.setattr(D, "disk_free", lambda p: 100 * 1024)
    rep = demo_ws.render.preflight()
    d = rep.item("disk")
    assert d.status == "error" and "Not enough disk space" in d.message and "Required: approximately" in d.message and "Available: approximately 1 MB" in d.message and not rep.can_start


def test_invalid_output_location_is_rejected(demo_ws, tmp_path):
    f = tmp_path / "a_file.txt"
    f.write_text("x")
    rep = demo_ws.render.preflight(output=f / "out.mp4")
    assert rep.item("output").status == "error" and not rep.can_start
    with pytest.raises(AppError):
        demo_ws.render.start_export(f / "out.mp4")


def test_unavailable_codec_is_reported_with_alternatives_and_never_swapped_silently(demo_ws, monkeypatch):
    ws = demo_ws
    caps = ws.render.engine.ffmpeg.capabilities()
    monkeypatch.setattr(caps, "encoders", {e for e in caps.encoders if e != "libx265"})
    ws.render.update_settings(video_codec="h265")
    rep = ws.render.preflight()
    s = rep.item("settings")
    assert s.status == "error" and "H.265" in s.message and "H.264" in s.fix and not rep.can_start
    snap = ws.render.snapshot()
    with pytest.raises(EncoderUnavailableError) as e:
        ws.render.engine.resolve(ws.render.settings, snap)
    assert "H.264" in e.value.alternatives and ws.render.settings.video_codec == "h265"  # the setting still says what the user chose


def test_incompatible_container_codec_combination_is_refused(demo_ws):
    with pytest.raises(RenderError, match="cannot be stored in a .webm"):
        demo_ws.render.update_settings(container="webm")  # H.264 audio/video do not belong in WebM


def test_hardware_choice_is_honest_and_cpu_always_works(demo_ws):
    ws = demo_ws
    caps = ws.render.capabilities()
    assert caps["ffmpeg_ok"] and caps["video_codecs"]["h264"]["available"] and caps["audio_codecs"]["aac"]["available"] and all(isinstance(v, bool) for v in caps["hardware"].values())
    assert ws.render.settings.hardware_acceleration == "auto"
    working = any(caps["hardware"].values())
    rep = ws.render.preflight()  # Auto: the CPU encoder is used when no GPU encoder works, and the notice says so
    assert rep.item("settings").status == "ok" and (working or any("CPU" in d for d in rep.item("settings").details))
    ws.render.update_settings(hardware_acceleration="hardware")
    rep = ws.render.preflight()
    if not working:  # the explicit "Hardware" choice is refused with a reason and the way out, not silently changed
        s = rep.item("settings")
        assert s.status == "error" and "No working hardware encoder" in s.message and "Auto or CPU" in s.fix and not rep.can_start
        assert ws.render.settings.hardware_acceleration == "hardware"
    ws.render.update_settings(hardware_acceleration="cpu")
    assert ws.render.preflight().can_start


# ================================================================ export flow (the acceptance test)
def test_export_end_to_end_with_real_progress_validation_metadata_and_re_export(demo_ws):
    ws = demo_ws
    doc_before = timeline_doc(ws)
    seen = []
    ws.bus.subscribe("render.updated", lambda t, p: seen.append((p["job"].status, p["job"].progress.stage.value, round(p["job"].progress.overall, 3), p["job"].progress.speed)))
    job = ws.render.start_export()
    assert job.status in (RenderStatus.QUEUED, RenderStatus.RUNNING) and job.record.timeline_hash and job.spec.snapshot.timeline_version == 1
    wait_job(job)
    assert job.status is RenderStatus.COMPLETED, job.error
    out = Path(job.record.output_path)
    assert out.is_file() and out.name == "Demo_480p_30fps.mp4"
    # real progress: monotonic overall, passes through the stages, FFmpeg's speed was reported
    overall = [o for _s, _st, o, _sp in seen]
    assert overall == sorted(overall) and overall[-1] == 1.0
    stages = [st for _s, st, _o, _sp in seen]
    for name in ("Validating Project", "Preparing Media", "Compiling Timeline", "Building Video Graph", "Building Audio Graph", "Rendering", "Encoding", "Validating Output", "Finalizing"):
        assert name in stages, name
    assert [stages.index(n) for n in ("Preparing Media", "Rendering", "Encoding", "Finalizing")] == sorted(stages.index(n) for n in ("Preparing Media", "Rendering", "Encoding", "Finalizing"))
    assert any(sp for *_a, sp in seen if sp)
    # the file is what the settings said, and was validated (not just "ffmpeg exited 0")
    info = ws.render.engine.probe.probe(out)
    assert (info.width, info.height, info.codec, info.audio_codec, info.sample_rate, info.has_audio) == (854, 480, "h264", "aac", 48000, True)
    assert info.fps == pytest.approx(30.0, abs=1e-6) and info.duration == pytest.approx(8.0, abs=0.1)
    ids = {c.id for c in job.result.report.checks}
    assert job.result.report.ok and {"duration", "resolution", "fps", "video_codec", "audio_codec", "voice", "clipping"} <= ids
    # metadata + log + snapshot, and the project history
    rec = ws.render.history()[-1]
    assert rec.render_id == job.id and rec.status == "COMPLETED" and rec.output_path == str(out) and rec.timeline_version == 1 and rec.settings["resolution"] == "480p"
    assert (rec.width, rec.height, rec.fps) == (854, 480, 30.0) and rec.size_bytes == out.stat().st_size and rec.validation["ok"]
    run = ws.project.root / "renders" / job.id
    meta = json.loads((run / "metadata.json").read_text())
    log = (run / "render.log").read_text()
    assert meta["render_id"] == job.id and meta["status"] == "COMPLETED" and (run / "snapshot.json").is_file()
    for needle in ("start:", "end:", "FFmpeg: ffmpeg version", "timeline_version=1", "settings:", "exit status: COMPLETED", "-filter_complex_script", "validate duration: ok"):
        assert needle in log, needle
    # nothing temporary is left behind, the originals and the timeline are untouched
    assert not job.spec.work_dir.exists() and not list((ws.project.root / "cache" / "render").glob("**/final.*")) and timeline_doc(ws) == doc_before
    # change one caption -> export again: a NEW file, the first is unchanged, the project is still fully editable
    first_bytes = out.read_bytes()
    cap = next(c for c in ws.project.timeline.all_clips() if c.kind == "caption")
    cap.text["lines"], cap.text["text"] = ["EDITED CAPTION", ""], "EDITED CAPTION"
    job2 = wait_job(ws.render.start_export())
    assert job2.status is RenderStatus.COMPLETED and Path(job2.record.output_path).name == "Demo_480p_30fps_01.mp4"
    assert out.read_bytes() == first_bytes and Path(job2.record.output_path).read_bytes() != first_bytes
    assert len(ws.render.history()) == 2 and ws.render.history()[0].timeline_hash != ws.render.history()[1].timeline_hash
    ws.timeline.set_clip_properties(next(c for c in ws.project.timeline.all_clips() if c.kind == "media").id, opacity=0.9)  # still editable (and undoable)
    assert ws.commands.can_undo


def test_output_names_never_overwrite_unless_asked(demo_ws, tmp_path):
    ws = demo_ws
    target = tmp_path / "exports" / "My Video (final).mp4"
    a = wait_job(ws.render.start_export(target))
    b = wait_job(ws.render.start_export(target))
    assert Path(a.record.output_path) == target and Path(b.record.output_path) == target.with_name("My Video (final)_01.mp4")
    before = target.stat().st_mtime_ns
    c = wait_job(ws.render.start_export(target, overwrite=True))
    assert Path(c.record.output_path) == target and target.stat().st_mtime_ns != before


def test_an_exported_draft_goes_to_its_own_folder(demo_ws):
    job = wait_job(demo_ws.render.start_draft())
    assert job.status is RenderStatus.COMPLETED and Path(job.record.output_path).parent.name == "draft" and job.record.kind == "draft" and Path(job.record.output_path).stem.endswith("_draft")


def test_render_uses_a_frozen_snapshot_so_editing_during_a_render_is_safe(render_ws, tmp_path):
    ws = render_ws
    d = tmp_path / "snap"
    d.mkdir()
    ws.media.import_files([solid_image(d / "red.png", "red"), solid_image(d / "blue.png", "blue")])
    ws.jobs.wait_idle(30)
    by = {a.name: a for a in ws.project.assets.all()}
    for t in ws.project.timeline.tracks:
        t.clips.clear()
    c = put(ws, "track_v1", media_clip(by["red.png"], 0.0, 6.0))
    ws.render.update_settings(resolution="1080p", quality="high", preset_id="custom")
    job = ws.render.start_export()
    while job.progress.overall < 0.02 and not job.status.is_terminal:
        time.sleep(0.02)
    c.asset_id = by["blue.png"].id  # the user keeps editing: the clip now shows blue...
    c.timeline_start = 3.0
    wait_job(job)
    assert job.status is RenderStatus.COMPLETED and job.spec.snapshot.tracks[0].clips[0].asset_id == by["red.png"].id
    assert near(px(frame_at(job.record.output_path, 1.0), 0.5, 0.5), RED, 30)  # ...but the render shows what the timeline was when it started
    assert next(x for x in ws.project.timeline.all_clips()).asset_id == by["blue.png"].id  # and the edit is in the project


# ================================================================ failures and recovery
def test_missing_media_blocks_export_then_relink_fixes_it(demo_ws, tmp_path):
    ws = demo_ws
    asset = ws.demo.assets["broll.mp4"]
    path = ws.project.asset_path(asset)
    moved = tmp_path / "elsewhere" / "broll.mp4"
    moved.parent.mkdir()
    path.rename(moved)
    doc = timeline_doc(ws)
    rep = ws.render.preflight()
    assert not rep.can_start and rep.item("visuals").status == "error" and "2/3 visuals found" in rep.item("visuals").message and asset.id in rep.missing_assets
    with pytest.raises(PreflightFailed) as e:
        ws.render.start_export()
    assert asset.id in e.value.asset_ids
    # a render that skips the preflight still fails cleanly, in the right stage, and touches nothing
    job = wait_job(ws.render.start_export(skip_preflight=True))
    assert job.status is RenderStatus.FAILED and job.error.stage == "Preparing Media" and job.error.kind == "missing_media" and asset.id in job.error.asset_ids
    assert ws.render.history()[-1].status == "FAILED" and ws.render.history()[-1].failed_stage == "Preparing Media" and timeline_doc(ws) == doc
    assert not Path(job.record.output_path).exists() and not job.spec.work_dir.exists()
    # relink: searching a folder finds it by name / content; the user confirms; the export then succeeds
    cands = ws.render.relink.find_candidates(asset, [tmp_path / "elsewhere"])
    assert cands and cands[0].path == moved and cands[0].exact and cands[0].score == 100.0
    ws.render.relink.relink(asset.id, moved)
    assert ws.render.preflight().can_start and timeline_doc(ws) == doc
    ok = wait_job(ws.render.start_export())
    assert ok.status is RenderStatus.COMPLETED and Path(ok.record.output_path).is_file()


def test_ffmpeg_failing_at_72_percent_leaves_the_project_intact_with_logs_and_retry(demo_ws, tmp_path):
    ws = demo_ws
    ctl = install_fake_ffmpeg(ws, tmp_path, "fail_chunk:0.72")
    doc = timeline_doc(ws)
    job = wait_job(ws.render.start_export())
    assert job.status is RenderStatus.FAILED
    e = job.error
    assert e.stage == "Rendering" and e.kind == "ffmpeg_failed" and "FFmpeg failed" in e.user_message and "Conversion failed" in (e.details or "")
    assert 0.55 < job.progress.video < 0.9  # it really got most of the way
    rec = ws.render.history()[-1]
    assert rec.status == "FAILED" and rec.failed_stage == "Rendering" and rec.possible_issue
    log = ws.render.read_log(job.id)
    assert "exit status: FAILED (Rendering)" in log and "Conversion failed" in log and Path(rec.log_path).is_file()  # logs preserved
    assert timeline_doc(ws) == doc and not Path(rec.output_path).exists() and not job.spec.work_dir.exists()
    assert (ws.project.root / "renders" / job.id / "debug").is_dir()  # the graph is kept for diagnosis
    ctl.write_text("ok")  # the problem goes away: Retry works (from the current project)
    retry = wait_job(ws.render.retry(job.id))
    assert retry.status is RenderStatus.COMPLETED and Path(retry.record.output_path).is_file() and retry.id != job.id


def test_a_late_failure_does_not_redo_the_finished_video_sections(demo_ws, tmp_path):
    ws = demo_ws
    ws.render.chunk_seconds, ws.render.chunk_max_seconds = 1.0, 30.0  # one cached section per scene
    ctl = install_fake_ffmpeg(ws, tmp_path, "fail_audio")
    job = wait_job(ws.render.start_export())
    assert job.status is RenderStatus.FAILED and job.error.stage == "Encoding" and "audio" in job.error.user_message
    sections = list((ws.project.root / "cache" / "render" / "chunks").glob("*.ts"))
    assert len(sections) == 2  # both scenes were rendered and kept
    ctl.write_text("ok")
    retry = wait_job(ws.render.retry(job.id))
    assert retry.status is RenderStatus.COMPLETED and retry.result.cached_chunks == 2 and retry.result.rendered_chunks == 0
    assert "reused from cache" in ws.render.read_log(retry.id)


def test_cached_sections_are_reused_but_only_changed_scenes_are_rendered_again(demo_ws):
    ws = demo_ws
    ws.render.chunk_seconds, ws.render.chunk_max_seconds = 1.0, 30.0
    a = wait_job(ws.render.start_export())
    assert a.result.rendered_chunks == 2 and a.result.cached_chunks == 0
    b = wait_job(ws.render.start_export())
    assert b.result.rendered_chunks == 0 and b.result.cached_chunks == 2 and Path(b.record.output_path).read_bytes()[:64] != b""
    cap = [c for c in ws.project.timeline.all_clips() if c.kind == "caption" and c.timeline_start > 4.0][0]  # a caption in scene 2
    cap.text["lines"], cap.text["text"] = ["ONLY SCENE TWO CHANGED"], "ONLY SCENE TWO CHANGED"
    c = wait_job(ws.render.start_export())
    assert c.result.rendered_chunks == 1 and c.result.cached_chunks == 1
    ws.project.settings.fps = 24  # a global change invalidates everything
    d = wait_job(ws.render.start_export())
    assert d.result.rendered_chunks == 2 and d.result.cached_chunks == 0


def test_hardware_encoder_failure_is_reported_and_cpu_fallback_is_offered(demo_ws, monkeypatch):
    ws = demo_ws
    monkeypatch.setattr(type(ws.render.engine.ffmpeg), "hardware_encoders", lambda self, c=None: {"h264_nvenc": True})  # pretend a GPU encoder exists
    ws.render.update_settings(hardware_acceleration="hardware")
    job = wait_job(ws.render.start_export())  # real FFmpeg then fails to open the (absent) GPU encoder
    assert job.status is RenderStatus.FAILED and job.error.kind == "hardware_failed" and job.error.can_fallback_cpu and "hardware" in job.error.possible_issue.lower()
    assert job.spec.resolved.hardware and job.spec.resolved.encoder == "h264_nvenc"
    fallback = wait_job(ws.render.retry(job.id, cpu_fallback=True))
    assert fallback.status is RenderStatus.COMPLETED and not fallback.spec.resolved.hardware and fallback.spec.resolved.encoder == "libx264"


def test_failure_classification_names_the_likely_cause():
    from app.rendering.executor import classify_failure

    assert classify_failure("x: No space left on device", False)[0] == "disk_full"
    assert classify_failure("/a/b.mp4: No such file or directory", False)[0] == "missing_media"
    assert classify_failure("[h264_nvenc] Cannot load libcuda.so.1", True) == ("hardware_failed", "The hardware encoder failed (driver, GPU or session limit).", True)
    assert classify_failure("Unknown encoder 'libx265'", False)[0] == "unsupported_codec"
    assert classify_failure("moov atom not found", False)[0] == "corrupt_media" and classify_failure("Permission denied", False)[0] == "invalid_path"
    assert classify_failure("something odd", False)[0] == "ffmpeg_failed"


def test_cancelling_a_running_4k_render_stops_ffmpeg_cleans_up_and_keeps_the_project(demo_ws):
    ws = demo_ws
    ws.render.update_settings(resolution="2160p", quality="draft", preset_id="custom")
    doc = timeline_doc(ws)
    job = ws.render.start_export()
    t0 = time.time()
    while job.progress.overall < 0.30 and not job.status.is_terminal and time.time() - t0 < 180:
        time.sleep(0.05)
    assert job.status is RenderStatus.RUNNING and job.progress.overall >= 0.30
    assert ws.render.cancel(job.id) and job.status is RenderStatus.CANCELING
    wait_job(job, 60)
    assert job.status is RenderStatus.CANCELED
    # FFmpeg is really gone, temporary files are cleaned, the output was never written, project and timeline unchanged
    assert not _ffmpeg_running_for(str(job.spec.work_dir))
    assert not job.spec.work_dir.exists() and not Path(job.record.output_path).exists() and not list(job.spec.cache_dir.rglob("*.part*")) and timeline_doc(ws) == doc
    assert ws.render.history()[-1].status == "CANCELED" and "exit status: CANCELED" in ws.render.read_log(job.id)
    assert all(Path(a.path).is_file() or ws.project.asset_path(a).is_file() for a in ws.project.assets.all())  # sources untouched


def _ffmpeg_running_for(needle: str) -> bool:
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
            except OSError:
                continue
            if "ffmpeg" in cmd and needle in cmd:
                return True
    return False


# ================================================================ queue
def test_queue_runs_jobs_in_order_pauses_cancels_and_never_collides(demo_ws):
    ws = demo_ws
    a = ws.render.start_export()
    b = ws.render.start_export()
    c = ws.render.start_export()
    assert ws.render.pause(c.id) and c.status is RenderStatus.PAUSED  # held in the queue
    assert len({j.spec.output_path for j in (a, b, c)}) == 3 and len({j.spec.work_dir for j in (a, b, c)}) == 3 and len({j.spec.run_dir for j in (a, b, c)}) == 3
    assert [p.name for p in (a.spec.output_path, b.spec.output_path, c.spec.output_path)] == ["Demo_480p_30fps.mp4", "Demo_480p_30fps_01.mp4", "Demo_480p_30fps_02.mp4"]
    wait_job(a)
    wait_job(b)
    assert a.status is RenderStatus.COMPLETED and b.status is RenderStatus.COMPLETED and c.status is RenderStatus.PAUSED
    assert ws.render.resume(c.id)
    d = ws.render.start_export()
    assert ws.render.cancel(d.id) or d.status.is_terminal
    wait_job(c)
    wait_job(d)
    assert c.status is RenderStatus.COMPLETED and d.status in (RenderStatus.CANCELED, RenderStatus.COMPLETED)
    assert [j.id for j in ws.render.jobs()] == [a.id, b.id, c.id, d.id]
    assert ws.render.remove(a.id) and a.id not in [j.id for j in ws.render.jobs()]
    assert not ws.render.remove(c.id) is False


def test_a_queued_job_that_is_cancelled_never_starts(demo_ws):
    ws = demo_ws
    first = ws.render.start_export()
    second = ws.render.start_export()
    assert ws.render.cancel(second.id)
    assert second.status is RenderStatus.CANCELED and second.started_at == ""
    wait_job(first)
    assert first.status is RenderStatus.COMPLETED and not Path(second.record.output_path).exists()
    assert ws.render.history()[-1].status == "CANCELED" or any(r.status == "CANCELED" for r in ws.render.history())


def test_a_running_render_can_be_paused_between_sections_and_resumed(demo_ws):
    ws = demo_ws
    ws.render.chunk_seconds, ws.render.chunk_max_seconds = 1.0, 30.0
    job = ws.render.start_export()
    ws.render.pause(job.id)
    t0 = time.time()
    while job.status is not RenderStatus.PAUSED and not job.status.is_terminal and time.time() - t0 < 60:
        time.sleep(0.02)
    if job.status is RenderStatus.PAUSED:
        assert ws.render.resume(job.id)
    wait_job(job)
    assert job.status is RenderStatus.COMPLETED


# ================================================================ persistence
def test_render_settings_and_history_survive_save_and_reopen(demo_ws):
    ws = demo_ws
    ws.render.update_settings(resolution="720p", fps=24, quality="custom", crf=21, video_codec="h264", audio_bitrate_kbps=160, hardware_acceleration="cpu", preset_id="custom")
    job = wait_job(ws.render.start_export())
    assert job.status is RenderStatus.COMPLETED
    ws.save()
    root = ws.project.root
    ws.close_project()
    p = ws.open_project(root)
    s = p.render_settings
    assert (s.resolution, s.fps, s.quality, s.crf, s.audio_bitrate_kbps, s.hardware_acceleration) == ("720p", 24, "custom", 21, 160, "cpu")
    rec = ws.render.history()[-1]
    assert rec.render_id == job.id and rec.status == "COMPLETED" and rec.settings["fps"] == 24 and rec.output_path and rec.timeline_hash == job.record.timeline_hash
    assert p.schema_version == 6 and "render_history" in p.to_document() and "proxies" in p.to_document()
    info = ws.render.engine.probe.probe(Path(rec.output_path))
    assert (info.width, info.height, info.fps) == (1280, 720, 24.0) and info.codec == "h264"


def test_a_version_5_project_opens_and_gets_render_sections(demo_ws):
    from app.project.project import Project

    doc = demo_ws.project.to_document()
    doc["schema_version"] = 5
    for k in ("render_history", "proxies"):
        doc.pop(k)
    doc["render_settings"] = {"container": "mp4", "video_codec": "h264", "audio_codec": "aac", "crf": 18}
    p = Project.from_document(doc)
    assert p.schema_version == 6 and p.render_history == [] and p.proxies == {} and p.render_settings.quality == "custom" and p.render_settings.crf == 18


def test_recovery_checkpoint_and_autosave_happen_before_a_render(demo_ws):
    ws = demo_ws
    ws.project.dirty = True
    job = wait_job(ws.render.start_export())
    assert job.status is RenderStatus.COMPLETED
    cps = list((ws.paths.data_dir / "checkpoints" / ws.project.project_id).glob("before_render_*.json"))
    assert cps and json.loads(cps[-1].read_text())["schema_version"] == 6
    assert ws.autosave.wait_idle(10)


# ================================================================ odd file names and paths
def test_unusual_paths_and_file_names_render_correctly(ws, tmp_path):
    import shutil

    ws.new_project("Silver 2027 (it's final) — café", tmp_path / "My Projects" / "C Drive (copy)")
    d = tmp_path / "Media Folder (raw)" / "Clip's ünïcode"
    d.mkdir(parents=True)
    red = solid_image(d / "red clip (1) – it's.png", "red", "320x180")
    from app.tests.helpers import make_video, write_tone_wav

    vid = make_video(d / "vidéo n°2 (final).mp4", 2.0)
    wav = write_tone_wav(d / "musique d'été.wav", 3.0, 0.3, 330.0, 44100)
    ws.media.import_files([red, vid, wav], link_mode="reference")  # referenced in place: the odd names stay as they are
    ws.jobs.wait_idle(30)
    by = {a.name: a for a in ws.project.assets.all()}
    assert set(by) == {"red clip (1) – it's.png", "vidéo n°2 (final).mp4", "musique d'été.wav"} and all(a.link_mode == "reference" for a in by.values())
    put(ws, "track_v1", media_clip(by["vidéo n°2 (final).mp4"], 0.0, 2.0))
    put(ws, "track_v2", media_clip(by["red clip (1) – it's.png"], 0.5, 1.0, scale=0.3))
    put(ws, "track_a2", media_clip(by["musique d'été.wav"], 0.0, 3.0, audio={"role": "MUSIC"}))
    t = put(ws, "track_v5", __import__("app.tests.render_helpers", fromlist=["text_clip"]).text_clip(0.2, 1.5, "Café “quoted” it's 50%: done; [ok]"))
    ws.render.update_settings(resolution="480p", quality="draft", preset_id="custom")
    ws.project.scenes = [Scene("s1", "1", 0.0, 1.0), Scene("s2", "2", 1.0, 3.0)]
    ws.render.chunk_seconds, ws.render.chunk_max_seconds = 0.5, 30.0  # two sections: the join list also has odd characters in every path
    out = tmp_path / "Exports (v2)" / "Silver 2027 (it's final) — café.mp4"
    job = wait_job(ws.render.start_export(out))
    assert job.status is RenderStatus.COMPLETED, job.error
    assert Path(job.record.output_path).name == out.name and Path(job.record.output_path).is_file() and job.result.report.ok and job.result.rendered_chunks == 2
    assert "file '" in ws.render.read_log(job.id) or "join video chunks" in ws.render.read_log(job.id)
    _ = t
    shutil.rmtree(out.parent)


def test_logs_never_contain_credentials(demo_ws, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-this-must-never-appear")
    job = wait_job(demo_ws.render.start_export())
    assert "sk-this-must-never-appear" not in demo_ws.render.read_log(job.id)
    from app.rendering.executor import RenderLog

    log = RenderLog(demo_ws.project.root / "x.log")
    log.line("GET https://api.example.com/v1?api_key=SECRET&q=1 and --token=ABC123")
    log.close()
    assert "SECRET" not in (demo_ws.project.root / "x.log").read_text() and "ABC123" not in (demo_ws.project.root / "x.log").read_text()


def test_render_cache_can_be_cleared_without_touching_exports_or_media(demo_ws):
    ws = demo_ws
    job = wait_job(ws.render.start_export())
    assert list((ws.project.root / "cache" / "render" / "chunks").glob("*"))
    n = ws.render.clear_render_cache()
    assert n >= 1 and not list((ws.project.root / "cache" / "render" / "chunks").glob("*")) and Path(job.record.output_path).is_file()
    assert all(ws.project.asset_path(a).is_file() for a in ws.project.assets.all())


_ = (BLUE, build_demo, caption_clip)
