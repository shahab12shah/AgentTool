"""Black / frozen frame detection on real (tiny, generated) videos."""

from __future__ import annotations

import subprocess

import pytest

from app.analysis.models import VisualIntent, VisualType
from app.qc import frame_checker
from app.qc.frame_checker import FrameChecker, black_pix_th, freeze_noise_db
from app.qc.severity import Severity
from app.qc.tests.conftest import needs_ffmpeg
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, find, narrate, new_project, qc_ctx, run_checker
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.probe import MediaProbeService
from app.tests.helpers import make_image

# 0-2 s picture | 2-4 s black | 4-6 s picture | 6-8 s almost black | 8-11 s frozen red | 11-12 s picture
PARTS = [("testsrc", 2), ("color=c=black", 2), ("testsrc2", 2), ("color=c=0x242424", 2), ("color=c=red", 3), ("testsrc", 1)]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    path = tmp_path_factory.mktemp("frames") / "demo.mp4"
    args = ["ffmpeg", "-y", "-v", "error"]
    for src, d in PARTS:
        args += ["-f", "lavfi", "-i", f"{src}{':' if '=' in src else '='}s=320x180:d={d}:r=24"]
    args += ["-filter_complex", "".join(f"[{i}]" for i in range(len(PARTS))) + f"concat=n={len(PARTS)}:v=1:a=0", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    return path


def project(tmp_path, demo, *, at=0.0, source=(0.0, 12.0), speed=1.0, seconds=30.0):
    p = new_project(tmp_path, seconds=seconds)
    a = add_asset(p, "demo.mp4", "video", duration=12.0, w=320, h=180, write=False)
    a.path = str(demo)
    a.link_mode = "reference"
    s = add_scene(p, 0, seconds, "A story about silver.")
    narrate(p)
    dur = (source[1] - source[0]) / speed
    c = add_clip(p, "track_v1", a, at, dur, scene=s, source_in=source[0], source_out=source[1], speed=speed, created_by="AI")
    return p, a, c


def run(p, **kw):
    ff = FFmpegService()
    return run_checker(FrameChecker(), qc_ctx(p, ffmpeg=ff, probe=MediaProbeService(ff), **kw))


def spans(out, code):
    return sorted((round(i.start_time, 1), round(i.end_time, 1)) for i in find(out, code))


def test_sensitivity_mappings_are_monotone_and_documented_defaults():
    assert black_pix_th(0.5) == pytest.approx(0.10) and black_pix_th(1.0) < black_pix_th(0.0) + 1 and black_pix_th(0.0) > black_pix_th(1.0)
    assert freeze_noise_db(0.5) == -60.0 and freeze_noise_db(1.0) > freeze_noise_db(0.0)


@needs_ffmpeg
def test_black_near_black_and_frozen_are_found_and_mapped_to_timeline_time(tmp_path, demo):
    p, a, c = project(tmp_path, demo, at=10.0)
    out = run(p)
    assert spans(out, "frames.black") == [(12.0, 14.0)]  # source 2-4 s, clip starts at 10 s
    assert spans(out, "frames.near_black") == [(16.0, 18.0)]
    assert spans(out, "frames.frozen") == [(18.0, 21.0)]
    black = find(out, "frames.black")[0]
    assert black.severity is Severity.ERROR and black.timeline_item_id == c.id and "demo.mp4" in black.description and black.confidence >= 90
    assert find(out, "frames.near_black")[0].severity is Severity.NOTICE and find(out, "frames.frozen")[0].severity is Severity.WARNING
    assert out.metrics["sources_scanned"] == 1


@needs_ffmpeg
def test_only_the_used_range_is_judged(tmp_path, demo):
    p, a, c = project(tmp_path, demo, source=(4.0, 6.0))  # just the clean stretch
    out = run(p)
    assert [i for i in out.issues if i.code in ("frames.black", "frames.near_black", "frames.frozen", "frames.decode_error")] == []
    p2, *_ = project(tmp_path / "b", demo, source=(1.0, 5.0), at=0.0)
    out2 = run(p2)
    assert spans(out2, "frames.black") == [(1.0, 3.0)] and find(out2, "frames.frozen") == []


@needs_ffmpeg
def test_playback_speed_scales_the_mapping(tmp_path, demo):
    p, a, c = project(tmp_path, demo, speed=2.0)
    assert spans(run(p), "frames.black") == [(1.0, 2.0)]


@needs_ffmpeg
def test_short_black_is_a_warning_and_long_black_an_error(tmp_path, demo):
    p, *_ = project(tmp_path, demo)
    p.qc_settings.frames.black_min_seconds = 0.5
    assert find(run(p), "frames.black")[0].severity is Severity.ERROR  # 2 s
    p2, a, c = project(tmp_path / "b", demo, source=(1.5, 2.9))  # 0.9 s of black inside the range
    assert find(run(p2), "frames.black")[0].severity is Severity.WARNING


@needs_ffmpeg
def test_declared_blackouts_and_fades_through_black_are_intentional(tmp_path, demo):
    p, a, c = project(tmp_path, demo, at=10.0)
    p.qc_settings.frames.intentional_black = [[11.5, 14.5]]
    out = run(p)
    assert find(out, "frames.black") == [] and find(out, "frames.black_intentional")[0].severity is Severity.INFO
    assert out.metrics["intentional_black"] == 1
    p2, a2, c2 = project(tmp_path / "fade", demo, source=(2.0, 5.0))
    c2.transition = {"type": "FADE", "duration": 2.0}  # the clip fades up from black
    out2 = run(p2)
    assert find(out2, "frames.black") == [] and find(out2, "frames.black_intentional")


@needs_ffmpeg
def test_still_images_are_never_frozen(tmp_path, demo):
    p = new_project(tmp_path, seconds=10)
    img = make_image(tmp_path / "pic.png", "red")
    a = add_asset(p, "pic.png", "image", duration=None, w=64, h=64, write=False)
    a.path, a.link_mode = str(img), "reference"
    s = add_scene(p, 0, 10, "x")
    add_clip(p, "track_v1", a, 0, 10, scene=s)
    out = run(p)
    assert [i for i in out.issues if i.code.startswith("frames.")] == [] and out.metrics["sources_scanned"] == 0


@needs_ffmpeg
def test_a_deliberate_hold_on_evidence_is_not_frozen(tmp_path, demo):
    p, a, c = project(tmp_path, demo, source=(8.0, 11.0))
    assert find(run(p), "frames.frozen")
    s = p.scenes[0]
    p.visual_intents[s.id] = VisualIntent(s.id, VisualType.EVIDENCE)
    assert find(run(p), "frames.frozen") == []


@needs_ffmpeg
def test_scan_budget_skips_and_reports(tmp_path, demo):
    p, a, c = project(tmp_path, demo)
    p.qc_settings.frames.max_scan_seconds = 5.0
    out = run(p)
    assert out.complete is False and any("scan budget" in n for n in out.notes) and out.metrics["sources_scanned"] == 0


@needs_ffmpeg
def test_results_are_cached_per_content_range_and_thresholds(tmp_path, demo, monkeypatch):
    p, a, c = project(tmp_path, demo)
    first = run(p)
    assert (p.root / "cache" / "qc" / "frames").is_dir() and list((p.root / "cache" / "qc" / "frames").glob("*.json"))
    monkeypatch.setattr(frame_checker, "scan_video", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must read the cache")))
    second = run(p)
    assert spans(second, "frames.black") == spans(first, "frames.black")
    p.qc_settings.frames.black_sensitivity = 0.9  # new thresholds: a new scan is needed
    with pytest.raises(AssertionError):
        run(p)


@needs_ffmpeg
def test_decoding_errors_are_reported(tmp_path, demo):
    bad = tmp_path / "broken.mp4"
    data = bytearray(demo.read_bytes())
    for i in range(len(data) // 3, len(data) // 3 + 4000):
        data[i] = (i * 31) % 256
    bad.write_bytes(bytes(data))
    p, a, c = project(tmp_path / "p", bad)
    out = run(p)
    # a damaged middle either breaks decoding (error) or is concealed silently; the checker must not crash and must say what it saw
    assert out.metrics["sources_scanned"] == 1 and all(i.code.startswith("frames.") for i in out.issues)
    for i in find(out, "frames.decode_error"):
        assert i.severity is Severity.ERROR and i.fix.kind == "asset.replace"


def test_black_stretch_on_the_timeline_outside_the_narration(tmp_path):
    p = new_project(tmp_path, seconds=20)
    a = add_asset(p, "city.mp4", "video", duration=60)
    s = add_scene(p, 0, 8, "Silver prices rose sharply.")
    narrate(p)
    add_clip(p, "track_v1", a, 0, 8, scene=s)  # the picture ends with the narration at 8 s, the video lasts 20 s
    out = run_checker(FrameChecker(), qc_ctx(p))  # no FFmpeg service: only the timeline check can run
    i = find(out, "frames.black_timeline")
    assert len(i) == 1 and i[0].severity is Severity.ERROR and i[0].start_time == pytest.approx(8.0, abs=0.5) and i[0].end_time == pytest.approx(20.0)
    assert any("FFmpeg is not available" in n for n in out.notes)
    p.qc_settings.frames.intentional_black = [[8.0, 20.0]]
    out2 = run_checker(FrameChecker(), qc_ctx(p))
    assert find(out2, "frames.black_timeline") == [] and find(out2, "frames.black_intentional")


def test_a_picture_gap_inside_the_narration_is_left_to_the_timeline_checker(tmp_path):
    p = new_project(tmp_path, seconds=10)
    a = add_asset(p, "city.mp4", "video", duration=60)
    s = add_scene(p, 0, 10, "Silver prices rose sharply last week and analysts were surprised.")
    narrate(p)
    add_clip(p, "track_v1", a, 0, 4, scene=s)
    add_clip(p, "track_v1", a, 6, 4, scene=s, source_in=20.0)  # a 2 s hole under the narration
    assert find(run_checker(FrameChecker(), qc_ctx(p)), "frames.black_timeline") == []


def test_contract(tmp_path):
    p = new_project(tmp_path)
    c = FrameChecker()
    assert c.expensive and not c.scene_local and c.settings_sections == ("frames",)
    before = p.to_document()
    run_checker(c, qc_ctx(p))
    assert p.to_document() == before
