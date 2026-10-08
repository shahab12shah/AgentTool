"""Pre-flight checker: the deterministic gate before any expensive analysis."""

from __future__ import annotations

import pytest

from app.qc.preflight import PreflightChecker
from app.qc.tests.conftest import needs_ffmpeg
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, new_project, narrate, qc_ctx, run_checker
from app.qc.severity import Severity
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.models import PreflightItem, PreflightReport
from app.rendering.probe import MediaProbeService
from app.tests.helpers import make_video


def good_project(tmp_path, **kw):
    p = new_project(tmp_path, seconds=20, **kw)
    a = add_asset(p, "city.mp4", "video", duration=30)
    s = add_scene(p, 0, 10, "Silver prices rose sharply.")
    add_clip(p, "track_v1", a, 0, 10, scene=s)
    narrate(p)
    return p, a, s


def run(p, **kw):
    return run_checker(PreflightChecker(), qc_ctx(p, **kw))


def test_valid_project_passes_without_issues(tmp_path):
    p, *_ = good_project(tmp_path)
    out = run(p)
    assert out.issues == []
    assert out.metrics["integrity_ok"] is True and out.metrics["blocking"] == 0


def test_empty_timeline_is_critical_and_breaks_integrity(tmp_path):
    p = new_project(tmp_path)
    out = run(p)
    assert "preflight.no_timeline" in codes(out)
    assert out.metrics["integrity_ok"] is False
    assert all(i.severity is Severity.CRITICAL for i in find(out, "preflight.no_timeline"))


def test_no_visual_clip_is_critical(tmp_path):
    p = new_project(tmp_path)
    mus = add_asset(p, "music.mp3", "audio", duration=30, w=None, h=None)
    add_clip(p, "track_a2", mus, 0, 10)  # something on the timeline, but nothing to see
    out = run(p)
    assert "preflight.no_visuals" in codes(out) and out.metrics["integrity_ok"] is False


def test_missing_voice_over_is_critical(tmp_path):
    p, *_ = good_project(tmp_path, voice=False)
    out = run(p)
    iss = find(out, "preflight.no_voice")
    assert len(iss) == 1 and iss[0].severity is Severity.CRITICAL and iss[0].title == "No voice-over"
    assert out.metrics["integrity_ok"] is False


def test_voice_without_duration_is_critical(tmp_path):
    p, *_ = good_project(tmp_path)
    p.voice_over.duration = 0
    out = run(p)
    assert "preflight.voice_duration" in codes(out) and out.metrics["integrity_ok"] is False


def test_inconsistent_document_is_reported_once_with_the_problems(tmp_path):
    p, a, s = good_project(tmp_path)
    add_clip(p, "track_v1", a, 20, 5, scene=s, id="dup")
    add_clip(p, "track_v1", a, 26, 5, scene=s, id="dup")  # duplicate id
    add_clip(p, "track_v2", None, 0, 4, scene=s)  # reference to an unknown asset
    add_clip(p, "track_v2", a, 8, -1.0, scene=s)  # negative duration
    out = run(p)
    iss = find(out, "preflight.document")
    assert len(iss) == 1 and iss[0].severity is Severity.CRITICAL
    text = " ".join(iss[0].affected_elements)
    assert "duplicate clip id dup" in text and "unknown asset" in text and "invalid time range" in text
    assert out.metrics["integrity_ok"] is False


@pytest.mark.parametrize("w,h,fps,needle", [(1920, 1080, 500, "frame rate"), (4, 1080, 30, "resolution"), (1921, 1080, 30, "odd side"), (20000, 1080, 30, "resolution")])
def test_invalid_project_settings(tmp_path, w, h, fps, needle):
    p, *_ = good_project(tmp_path)
    p.settings.width, p.settings.height, p.settings.fps = w, h, fps
    out = run(p)
    iss = find(out, "preflight.project_settings")
    assert len(iss) == 1 and needle in iss[0].description and out.metrics["integrity_ok"] is False


def test_invalid_export_settings_one_issue_each(tmp_path):
    p, *_ = good_project(tmp_path)
    p.render_settings.audio_sample_rate = 12345
    p.render_settings.container, p.render_settings.video_codec = "webm", "h264"
    out = run(p)
    texts = [i.description for i in find(out, "preflight.export_settings")]
    assert any("sample" in t.lower() for t in texts) and any("cannot be stored" in t for t in texts)
    assert all(i.severity is Severity.CRITICAL for i in find(out, "preflight.export_settings"))


def test_valid_default_export_settings_have_no_issue(tmp_path):
    p, *_ = good_project(tmp_path)
    assert find(run(p), "preflight.export_settings") == []


def test_stale_proxies_warn_only_when_proxies_are_used(tmp_path):
    p, *_ = good_project(tmp_path)
    assert find(run(p), "preflight.proxy") == []
    p.render_settings.use_proxies = True
    iss = find(run(p), "preflight.proxy")
    assert len(iss) == 1 and iss[0].severity is Severity.WARNING and "city.mp4" in iss[0].description


def test_missing_and_undecodable_media_break_integrity_but_are_reported_by_the_asset_checker(tmp_path):
    p, a, _ = good_project(tmp_path)
    (p.root / a.path).unlink()
    out = run(p)
    assert out.metrics["integrity_ok"] is False
    assert not [i for i in out.issues if i.code.startswith("asset.")]  # no duplicate of the per-asset issue
    assert any("missing" in n for n in out.notes)


def test_does_not_modify_the_project(tmp_path):
    p, *_ = good_project(tmp_path)
    before = p.to_document()
    run(p)
    assert p.to_document() == before


def test_cache_key_follows_free_disk_space_and_ffmpeg(tmp_path, monkeypatch):
    p, *_ = good_project(tmp_path)
    ctx = qc_ctx(p)
    c = PreflightChecker()
    base = c.input_hash(ctx)
    assert c.input_hash(ctx) == base
    monkeypatch.setattr("app.qc.preflight.disk_free", lambda _p: 7 * 1024 ** 3)
    assert c.input_hash(ctx) != base


# ---------------------------------------------------------------- with FFmpeg (the render diagnostics are reused)
def _ff_ctx(p, **kw):
    ff = FFmpegService()
    return qc_ctx(p, ffmpeg=ff, probe=MediaProbeService(ff), **kw)


def _item(i, status, msg, fix=""):
    return PreflightItem(i, i, status, msg, fix)


@needs_ffmpeg
def test_real_project_with_real_media_is_ready(tmp_path):
    p = new_project(tmp_path, seconds=3, voice=False)
    import wave

    wav = tmp_path / "media" / "audio" / "voice.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(48000)
        w.writeframes(b"\0\0" * 48000 * 3)
    vid = tmp_path / "media" / "video" / "v.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    make_video(vid, seconds=4, size="640x360")
    va = add_asset(p, "v.mp4", "video", duration=4, w=640, h=360, write=False)
    va.path = "media/video/v.mp4"
    va.size_bytes = vid.stat().st_size
    vo = add_asset(p, "voice.wav", "audio", duration=3, w=None, h=None, write=False)
    vo.path = "media/audio/voice.wav"
    vo.size_bytes = wav.stat().st_size
    p.voice_over.asset_id, p.voice_over.duration = vo.id, 3.0
    s = add_scene(p, 0, 3, "Silver rose.")
    add_clip(p, "track_v1", va, 0, 3, scene=s)
    out = run_checker(PreflightChecker(), _ff_ctx(p))
    assert out.metrics["integrity_ok"] is True, out.notes
    assert not [i for i in out.issues if i.severity is Severity.CRITICAL], [(i.code, i.description) for i in out.issues]


@needs_ffmpeg
def test_corrupt_media_breaks_integrity(tmp_path):
    p, a, _ = good_project(tmp_path)
    (p.root / a.path).write_text("this is not a video")
    out = run_checker(PreflightChecker(), _ff_ctx(p))
    assert out.metrics["integrity_ok"] is False and any("cannot be decoded" in n for n in out.notes)


def test_missing_ffmpeg_is_critical(tmp_path):
    p, *_ = good_project(tmp_path)
    ff = FFmpegService(lambda: "/nonexistent/ffmpeg", lambda: "/nonexistent/ffprobe")
    out = run_checker(PreflightChecker(), qc_ctx(p, ffmpeg=ff, probe=MediaProbeService(ff)))
    iss = find(out, "preflight.ffmpeg")
    assert len(iss) == 1 and iss[0].severity is Severity.CRITICAL and out.metrics["integrity_ok"] is False


@needs_ffmpeg
def test_render_diagnostics_items_are_mapped_to_issues(tmp_path, monkeypatch):
    p, *_ = good_project(tmp_path)
    rep = PreflightReport(items=[
        _item("disk", "error", "Not enough disk space. Required: approximately 4.0 GB. Available: approximately 1.0 GB.", "Free some space or choose another drive."),
        _item("settings", "error", "The H.265 encoder is not installed.", "Choose a supported codec: H.264"),
        _item("fonts", "warning", "1 font(s) are not installed; a fallback will be used", "Install the font."),
        _item("output", "ok", "Output location valid"),
        _item("timeline", "error", "Timeline has 2 problems"),  # owned by the timeline checker: not repeated
    ])
    monkeypatch.setattr("app.qc.preflight.RenderDiagnostics.run", lambda self, *a, **k: rep)
    out = run_checker(PreflightChecker(), _ff_ctx(p))
    by = {i.code: i for i in out.issues}
    assert by["preflight.disk"].severity is Severity.CRITICAL and "4.0 GB" in by["preflight.disk"].description
    assert by["preflight.disk"].suggested_fix == "Free some space or choose another drive."
    assert by["preflight.encoder"].severity is Severity.CRITICAL
    assert by["preflight.fonts"].severity is Severity.WARNING
    assert "preflight.output" not in by and not [c for c in by if "timeline" in c]
    assert out.metrics["blocking"] == 2
    assert not [r for r in out.metrics["integrity_reasons"] if "disk" in r or "encoder" in r]  # disk / encoder block the export, not the analysis (the placeholder media here are undecodable)
