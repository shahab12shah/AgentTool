"""Render readiness (dry run) and post-render inspection of a real exported file."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.qc.context import QCContext
from app.qc.issue_model import QCCategory
from app.qc.render_checker import RenderedFileChecker, RenderReadinessChecker, _warning_kind
from app.qc.severity import Severity
from app.qc.tests.conftest import needs_ffmpeg
from app.qc.tests.qc_helpers import find
from app.rendering.errors import EncoderUnavailableError
from app.rendering.presets import output_size
from app.tests.render_helpers import put, render_to, text_clip

pytestmark = needs_ffmpeg


def ctx_of(ws, **kw) -> QCContext:
    eng = ws.render.engine
    return QCContext.build(ws.project, ffmpeg=eng.ffmpeg, probe=eng.probe, **kw)


def readiness(ws, shared=None, **kw):
    ctx = ctx_of(ws, **kw)
    ctx.shared.update(shared or {})
    return RenderReadinessChecker().run(ctx, lambda f, m: None)


def snapshot_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


# ---------------------------------------------------------------- readiness
def test_valid_demo_is_ready_and_nothing_is_written(render_ws):
    ws = render_ws
    before = snapshot_files(ws.project.root)
    out = readiness(ws)
    assert out.issues == [] and out.metrics["ready"] is True and out.metrics["blocking"] == 0 and out.metrics["chunks"] >= 1 and out.metrics["commands"] >= 2
    assert snapshot_files(ws.project.root) == before  # a dry run: no render folders, no cache, no files


def test_invalid_export_settings_are_reported_when_the_preflight_did_not_run(render_ws):
    ws = render_ws
    ws.project.render_settings.audio_sample_rate = 12345
    ws.project.render_settings.container, ws.project.render_settings.video_codec = "webm", "h264"
    out = readiness(ws)
    assert [i for i in out.issues if i.code == "render.codec_unavailable" and i.severity is Severity.CRITICAL] and out.metrics["ready"] is False
    assert all(i.category is QCCategory.RENDER_READINESS for i in out.issues)


def test_nothing_is_repeated_when_the_preflight_already_reported_it(render_ws):
    from app.qc.preflight import PreflightChecker

    ws = render_ws
    ws.project.render_settings.audio_sample_rate = 12345
    ctx = ctx_of(ws)
    pre = PreflightChecker().run(ctx, lambda f, m: None)
    assert any(i.code == "preflight.export_settings" for i in pre.issues)
    out = readiness(ws, {"preflight": pre})
    assert [i for i in out.issues if i.code == "render.codec_unavailable"] == []


def test_missing_encoder_is_critical_with_alternatives(render_ws, monkeypatch):
    def boom(self, s, snap, *, force_cpu=False):
        raise EncoderUnavailableError("H.265 encoding is not available in this FFmpeg build.", ["H.264", "VP9"])

    monkeypatch.setattr("app.qc.render_checker.EncoderSelector.resolve", boom)
    out = readiness(render_ws)
    i = find(out, "render.codec_unavailable")
    assert len(i) == 1 and i[0].severity is Severity.CRITICAL and "H.264, VP9" in i[0].description + i[0].suggested_fix and out.metrics["ready"] is False


def test_missing_media_is_critical_but_not_reported_twice(render_ws):
    ws = render_ws
    Path(ws.project.asset_path(ws.demo.assets["main.mp4"])).unlink()
    out = readiness(ws)
    i = find(out, "render.graph_failed")
    assert len(i) == 1 and i[0].severity is Severity.CRITICAL and out.metrics["ready"] is False
    # with the asset checker's finding in the shared results, the readiness check stays quiet but still says "not ready"
    from app.qc.asset_checker import AssetChecker

    ctx = ctx_of(ws)
    ctx.shared["asset"] = AssetChecker().run(ctx, lambda f, m: None)
    assert any(x.code == "asset.missing" for x in ctx.shared["asset"].issues)
    out2 = RenderReadinessChecker().run(ctx, lambda f, m: None)
    assert find(out2, "render.graph_failed") == [] and out2.metrics["ready"] is False and out2.metrics["blocked_by"] == "missing media"


def test_a_missing_font_is_a_warning_with_the_substitute(render_ws):
    ws = render_ws
    tc = text_clip(5.0, 1.5, "Hello")
    tc.text["font"] = "No Such Font 123"
    put(ws, "track_v5", tc)
    out = readiness(ws)
    got = [i for i in out.issues if i.code in ("render.font_missing", "render.effect_unsupported")]
    assert got and all(i.severity is Severity.WARNING for i in got) and out.metrics["ready"] is True


def test_compiler_warnings_are_mapped_to_readiness_codes():
    assert _warning_kind("“Foo” is not installed; Sans will be used.")[0] == "render.font_missing"
    assert _warning_kind("Clip x: stretching is not allowed; the clip is filled without distortion instead.")[0] == "render.effect_unsupported"
    assert _warning_kind("Clip x: blur keyframes are rendered as one constant blur (2).")[0] == "render.effect_unsupported"
    assert _warning_kind("caption cap1 could not be rendered: bad style") == ("render.caption_unrenderable", Severity.ERROR)
    assert _warning_kind("The wipe transition on clip x is not supported")[0] == "render.transition_unsupported"


def test_without_ffmpeg_the_check_says_so_and_does_not_crash(render_ws):
    ctx = QCContext.build(render_ws.project)
    out = RenderReadinessChecker().run(ctx, lambda f, m: None)
    assert any("FFmpeg is not available" in n for n in out.notes) and out.metrics["ready"] in (True, False)


# ---------------------------------------------------------------- post-render
@pytest.fixture
def rendered(render_ws, tmp_path):
    ws = render_ws
    res, spec = render_to(ws.render.engine, ws, tmp_path, "qc")
    ctx = ctx_of(ws, rendered_file=res.output_path)
    w, h = output_size(*ctx.canvas, "480p")
    expected = {"duration": 8.0, "width": w, "height": h, "fps": 30, "has_audio": True}
    return ws, Path(res.output_path), ctx, expected


def inspect(ctx, path, expected):
    return RenderedFileChecker().inspect(ctx, path, render_id="r1", expected=expected, report=lambda f, m: None)


def status(r, cid):
    return next(c["status"] for c in r["checks"] if c["id"] == cid)


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *map(str, args)], check=True)


def test_a_good_render_passes_every_check(rendered):
    ws, path, ctx, exp = rendered
    before = path.read_bytes()
    r = inspect(ctx, path, exp)
    assert r["status"] == "PASSED" and r["render_id"] == "r1" and r["summary"].startswith("The rendered file passed")
    ids = {c["id"] for c in r["checks"]}
    assert {"exists", "duration", "resolution", "fps", "video_codec", "audio_stream", "sample_rate", "black_frames", "frozen_frames", "unexpected_silence", "loudness", "corruption"} <= ids
    assert all(c["status"] == "pass" for c in r["checks"]) and r["issues"] == []
    assert r["measured"]["lufs"] is not None and r["measured"]["true_peak_dbfs"] is not None and r["path"] == str(path)
    assert path.read_bytes() == before  # the file is never modified
    import json

    json.dumps(r)  # plain JSON: it is stored in the project


def test_a_short_file_fails_the_duration_check(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    short = tmp_path / "short.mp4"
    ff("-i", path, "-t", 3, "-c", "copy", short)
    r = inspect(ctx, short, exp)
    assert r["status"] == "FAILED" and status(r, "duration") == "fail"
    assert any(i["code"] == "postrender.duration" and i["severity"] == "ERROR" for i in r["issues"])


def test_wrong_resolution_fails(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    big = tmp_path / "big.mp4"
    ff("-i", path, "-vf", "scale=640:360", "-c:a", "copy", big)
    r = inspect(ctx, big, exp)
    assert r["status"] == "FAILED" and status(r, "resolution") == "fail"


def test_missing_audio_stream_fails(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    mute = tmp_path / "noaudio.mp4"
    ff("-i", path, "-an", "-c:v", "copy", mute)
    r = inspect(ctx, mute, exp)
    assert r["status"] == "FAILED" and status(r, "audio_stream") == "fail"


def test_silent_audio_fails(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    silent = tmp_path / "silent.mp4"
    ff("-i", path, "-af", "volume=0", "-c:v", "copy", silent)
    r = inspect(ctx, silent, exp)
    assert r["status"] == "FAILED" and (status(r, "mute") == "fail" or status(r, "loudness") == "fail")


def test_silence_where_the_transcript_has_speech_is_reported(rendered, tmp_path):
    from app.qc.tests.qc_helpers import narrate
    from app.analysis.models import Origin, Scene, SceneStatus

    ws, path, ctx, exp = rendered
    ws.project.scenes.append(Scene("scene_001", "1", 0.0, 8.0, "Silver is running out of supply and the price could climb much higher soon.", [], "silver", "", 0.5, 0.9, SceneStatus.READY, Origin.AI))
    narrate(ws.project)
    holed = tmp_path / "holed.mp4"
    ff("-i", path, "-af", "volume=enable='between(t,2,5)':volume=0", "-c:v", "copy", holed)
    ctx2 = ctx_of(ws, rendered_file=holed)
    r = inspect(ctx2, holed, exp)
    assert status(r, "unexpected_silence") == "fail" and any(i["code"] == "postrender.unexpected_silence" for i in r["issues"])
    # the same file judged without a transcript: no word is expected there, so a quiet stretch is not an error
    ws.project.transcription.transcript = None
    r2 = inspect(ctx_of(ws, rendered_file=holed), holed, exp)
    assert status(r2, "unexpected_silence") == "pass"


def test_black_frames_in_the_output_are_found(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    blk = tmp_path / "black.mp4"
    ff("-i", path, "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,2,5)'", "-c:a", "copy", blk)
    r = inspect(ctx_of(ws, rendered_file=blk), blk, exp)
    assert status(r, "black_frames") == "fail" and r["status"] == "FAILED"
    assert any(i["code"] == "frames.black" and i["category"] == "FRAMES" for i in r["issues"])
    ws.project.qc_settings.frames.intentional_black = [[1.5, 5.5]]
    r2 = inspect(ctx_of(ws, rendered_file=blk), blk, exp)
    assert status(r2, "black_frames") == "pass"


def test_a_corrupt_file_fails(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    bad = tmp_path / "bad.mp4"
    data = bytearray(path.read_bytes())
    mid = len(data) // 2
    data[mid:mid + 3000] = bytes((i * 7) % 256 for i in range(3000))
    bad.write_bytes(bytes(data))
    r = inspect(ctx_of(ws, rendered_file=bad), bad, exp)
    assert r["status"] in ("FAILED", "WARNINGS")  # a damaged middle is either flagged by the decode pass or concealed by the decoder; it must never be reported as clean
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video at all" * 100)
    r2 = inspect(ctx, garbage, exp)
    assert r2["status"] == "FAILED" and status(r2, "readable") == "fail"


def test_a_missing_file_fails_at_once(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    r = inspect(ctx, tmp_path / "nope.mp4", exp)
    assert r["status"] == "FAILED" and status(r, "exists") == "fail" and len(r["checks"]) == 1


def test_clipping_audio_is_a_warning_with_the_measured_values(rendered, tmp_path):
    ws, path, ctx, exp = rendered
    hot = tmp_path / "hot.mp4"
    ff("-i", path, "-af", "volume=30dB,alimiter=limit=1:level=disabled", "-c:v", "copy", hot)
    r = inspect(ctx_of(ws, rendered_file=hot), hot, exp)
    assert status(r, "loudness") in ("warn", "pass") and r["measured"]["true_peak_dbfs"] is not None
    if status(r, "loudness") == "warn":
        assert r["status"] in ("WARNINGS", "FAILED") and any(i["code"] == "postrender.true_peak" for i in r["issues"])


def test_result_is_stored_the_way_the_service_expects(rendered):
    ws, path, ctx, exp = rendered
    r = inspect(ctx, path, exp)
    assert set(r) >= {"render_id", "path", "checked_at", "status", "summary", "checks", "issues", "measured"}
    assert all(set(c) >= {"id", "label", "status", "message", "measured", "expected"} for c in r["checks"])
