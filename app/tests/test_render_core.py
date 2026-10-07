"""Phase 6 core tests: FFprobe, expressions, presets, timeline compilation and the *pixels and samples* the renderer produces.

Everything here renders for real with FFmpeg (small 480p frames) and then looks at the output: colours at positions for transforms, keyframes,
transitions and cuts; levels over time for ducking and fades.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from app.presentation.animation import spec as anim_spec
from app.rendering import presets as P
from app.rendering.builder import to_geq
from app.rendering.expressions import evaluate, keyframe_curve
from app.rendering.ffmpeg_service import escape_filter_path, escape_filter_value, redact
from app.rendering.models import RenderSnapshot
from app.rendering.probe import MediaProbeService
from app.tests.conftest import needs_ffmpeg
from app.tests.render_helpers import (BLACK, BLUE, GREEN, RED, caption_clip, color_video, decode_audio, frame_at, half_transparent_png, media_clip, near, put, px, render_to, rms,
                                      run_col, run_row, solid_image, span_col, span_row, text_clip, two_color_image)
from app.timeline.clip import Clip
from app.timeline.keyframes import Keyframe, value_at

pytestmark = needs_ffmpeg


@pytest.fixture
def solids(project_ws, tmp_path):
    """A project with solid-colour media imported (so tests can read colours back out of rendered frames)."""
    ws = project_ws
    d = tmp_path / "solids"
    d.mkdir()
    files = [solid_image(d / "red.png", "red", "400x225"), solid_image(d / "blue.png", "blue", "400x225"), solid_image(d / "green.png", "green", "400x225"),
             solid_image(d / "redbox.png", "red", "200x100"), two_color_image(d / "two.png"), half_transparent_png(d / "alpha.png"), color_video(d / "rgb.mp4", each=1.0)]
    ws.media.import_files(files)
    assert ws.jobs.wait_idle(60)
    ws.by = {a.name: a for a in ws.project.assets.all()}
    return ws


def frame(res, t: float):
    return frame_at(res.output_path, t)


# ================================================================ FFprobe / media compatibility
def test_probe_reads_video_image_audio_and_alpha_facts(solids, media_dir, tmp_path, render_engine):
    pr = MediaProbeService(render_engine.ffmpeg)
    v = pr.probe(media_dir / "clip.mp4")
    assert (v.kind, v.width, v.height, v.codec, v.pix_fmt, v.has_audio, v.container.split(",")[0]) == ("video", 320, 180, "h264", "yuv420p", True, "mov")
    assert v.fps == pytest.approx(30.0) and v.duration == pytest.approx(3.0, abs=0.1) and v.frame_count in (90, 91) and v.sample_rate == 44100 and v.channels == 1 and v.bitrate
    png = pr.probe(media_dir / "pic.png")
    assert (png.kind, png.width, png.height, png.duration, png.has_alpha) == ("image", 64, 64, None, False)
    assert pr.probe(tmp_path / "solids" / "alpha.png").has_alpha is True
    wav = pr.probe(media_dir / "voice.wav")
    assert (wav.kind, wav.has_video, wav.has_audio, wav.sample_rate, wav.codec) == ("audio", False, True, 44100, "pcm_s16le") and wav.duration == pytest.approx(4.0, abs=0.05)
    mov = tmp_path / "x.mov"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media_dir / "clip.mp4"), "-c", "copy", str(mov)], check=True)
    assert pr.probe(mov).container.startswith("mov")
    for name, args in (("x.mkv", ["-c", "copy"]), ("x.avi", ["-c:v", "mpeg4", "-an"]), ("x.webm", ["-c:v", "libvpx", "-an"])):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media_dir / "clip.mp4"), *args, str(tmp_path / name)], check=True)
        assert pr.probe(tmp_path / name).has_video
    for name, extra in (("x.flac", []), ("x.mp3", []), ("x.m4a", [])):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media_dir / "voice.wav"), *extra, str(tmp_path / name)], check=True)
        assert pr.probe(tmp_path / name).kind == "audio"
    for name in ("x.jpg", "x.webp", "x.tiff"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media_dir / "pic.png"), str(tmp_path / name)], check=True)
        assert pr.probe(tmp_path / name).kind == "image"


def test_probe_reports_rotation_so_the_display_size_is_what_the_renderer_uses(tmp_path, media_dir, render_engine):
    out = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation", "90", "-i", str(media_dir / "clip.mp4"), "-c", "copy", str(out)], check=True)
    info = MediaProbeService(render_engine.ffmpeg).probe(out)
    assert info.rotation in (90, 270) and (info.coded_width, info.coded_height) == (320, 180) and (info.width, info.height) == (180, 320)


def test_unsupported_and_broken_media_give_clear_errors(tmp_path, media_dir, render_engine):
    from app.core.exceptions import MediaProbeError, UnsupportedMediaError

    pr = MediaProbeService(render_engine.ffmpeg)
    with pytest.raises(UnsupportedMediaError, match="not a supported file type"):
        pr.probe(media_dir / "notes.txt")
    with pytest.raises(MediaProbeError, match="corrupt or use an unsupported codec"):
        pr.probe(media_dir / "broken.mp4")
    info, err = pr.try_probe(tmp_path / "missing.mp4")
    assert info is None and "cannot be read" in err


def test_imported_assets_carry_the_probe_facts(solids):
    a = solids.by["alpha.png"]
    assert a.extra["probe"]["has_alpha"] is True and "pix_fmt" in a.extra["probe"] and solids.by["rgb.mp4"].extra["probe"]["frame_count"] in (72, 73)


# ================================================================ expressions, presets, escaping
def test_keyframe_expressions_match_the_editors_interpolation_exactly():
    kfs = [Keyframe("scale", 0.0, 1.0, "ease_in_out"), Keyframe("scale", 2.0, 1.5, "linear"), Keyframe("scale", 4.0, 1.2, "ease_out"), Keyframe("scale", 5.0, 1.2)]
    c = keyframe_curve(kfs, "scale", 10.0, 1.0)
    assert not c.is_const
    for lt in (-1.0, 0.0, 0.3, 1.0, 2.0, 2.7, 3.5, 4.0, 4.5, 9.0):
        assert evaluate(c.expr, 10.0 + lt) == pytest.approx(value_at(kfs, "scale", lt), abs=1e-5)
    assert keyframe_curve([], "scale", 0.0, 1.0).is_const and keyframe_curve([Keyframe("scale", 0, 2.0), Keyframe("scale", 3, 2.0)], "scale", 0.0, 1.0).value == 2.0
    assert to_geq("clip((t-2)/1,0,1)*lt(t,3)") == "clip((T-2)/1,0,1)*lt(T,3)"  # only a standalone ``t`` is renamed for geq


def test_output_sizes_keep_the_projects_aspect_ratio_for_every_resolution():
    assert P.output_size(1920, 1080, "1080p") == (1920, 1080) and P.output_size(1920, 1080, "2160p") == (3840, 2160) and P.output_size(1920, 1080, "480p") == (854, 480)
    assert P.output_size(1080, 1920, "1080p") == (1080, 1920) and P.output_size(1080, 1920, "2160p") == (2160, 3840)
    assert P.output_size(1080, 1080, "720p") == (720, 720) and P.output_size(3840, 2160, "720p") == (1280, 720)
    for w, h in ((1920, 1080), (1080, 1920), (1080, 1080)):
        for r in P.RESOLUTIONS:
            ow, oh = P.output_size(w, h, r)
            assert ow % 2 == 0 and oh % 2 == 0 and abs(ow / oh - w / h) < 0.01


def test_presets_and_compatibility_rules():
    from app.project.project_schema import RenderSettings

    s = P.apply_preset(RenderSettings(), "youtube_4k")
    assert (s.resolution, s.video_codec, s.audio_codec, s.container, s.preset_id) == ("2160p", "h264", "aac", "mp4", "youtube_4k")
    d = P.apply_preset(s, "draft")
    assert d.resolution == "480p" and d.quality == "draft" and P.apply_preset(s, "nope").preset_id == "custom"
    assert P.compatibility_problems(RenderSettings()) == []
    assert P.compatibility_problems(replace(RenderSettings(), container="webm", video_codec="h264"))  # H.264 does not belong in WebM
    assert P.compatibility_problems(replace(RenderSettings(), fps=25)) and P.compatibility_problems(replace(RenderSettings(), resolution="8k"))
    r = RenderSettings.from_dict({"container": "mp4", "video_codec": "h264", "audio_codec": "aac", "crf": 18})  # a Phase 1 document
    assert r.quality == "custom" and r.crf == 18 and RenderSettings.from_dict(r.to_dict()) == r


def test_concat_list_uses_forward_slashes_and_escapes_apostrophes(render_engine):
    from pathlib import Path, PureWindowsPath

    from app.rendering.builder import FFmpegCommandBuilder

    b = FFmpegCommandBuilder("ffmpeg", P.ResolvedOutput(854, 480, 30, "mp4", "h264", "libx264", False, "aac", "aac", 128, 48000, "draft", "yuv420p", 28, 0, "ultrafast"), type("S", (), {"canvas_w": 1920})(), None)  # type: ignore[arg-type]
    cmd = b.concat([PureWindowsPath(r"C:\Projects\My Video\Silver 2027\a (1).ts"), Path("/tmp/it's here/b.ts")], "chunks.txt", Path("out.ts"), 5.0)
    assert cmd.files["chunks.txt"] == "file 'C:/Projects/My Video/Silver 2027/a (1).ts'\nfile '/tmp/it'\\''s here/b.ts'\n"
    assert cmd.args[:1] == ["ffmpeg"] and all(isinstance(a, str) for a in cmd.args) and "-safe" in cmd.args  # an argument list, never a shell string


def test_windows_paths_and_special_characters_are_escaped_not_concatenated():
    assert escape_filter_path(r"C:\Projects\My Video\Silver 2027\sub (1).ass") == r"C\:/Projects/My Video/Silver 2027/sub (1).ass"
    assert escape_filter_value("it's a: test, [x]") == r"it\'s a\: test\, \[x\]"
    assert redact("https://x/api?api_key=SECRET123&a=1") == "https://x/api?api_key=***&a=1" and redact("--token=abc") == "--token=***"


# ================================================================ timeline compilation
def test_compiler_layers_stack_by_track_and_carry_geometry_curves(solids, render_engine):
    from app.rendering.compiler import TimelineCompiler
    from app.rendering.models import ChunkPlan
    from app.rendering.sources import SourceSelector

    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 2.0))
    put(ws, "track_v3", media_clip(by["blue.png"], 0.5, 1.0, scale=0.5, position=(100.0, 0.0), rotation=15.0, opacity=0.5))
    k = put(ws, "track_v2", media_clip(by["green.png"], 1.0, 1.0))
    k.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 1.0, 2.0)]
    put(ws, "track_v5", text_clip(0.5, 1.0, "HELLO"))
    put(ws, "track_v6", caption_clip(0.6, ["Some", "words", "here"]))
    snap = RenderSnapshot.from_project(ws.project, ws.project.render_settings)
    from app.rendering.executor import prepare_assets  # noqa: F401  (re-exported for symmetry)
    from app.rendering.sources import prepare_assets as prep

    prep(snap, render_engine.probe)
    comp = TimelineCompiler(snap, SourceSelector(snap, render_engine.probe), render_engine.fonts, 30)
    cc = comp.compile_chunk(ChunkPlan(0, 0.0, 2.0, 60))
    order = [(s.kind, s.layer.track_id if s.layer else "") for s in cc.steps]
    assert order == [("layer", "track_v1"), ("layer", "track_v2"), ("layer", "track_v3"), ("ass", "")]  # V1 below V2 below V3; text + captions on top
    v1, v2, v3 = cc.layers
    assert v1.scale.is_const and v1.scale.value == 1.0 and v1.opacity.is_const
    assert not v2.scale.is_const and evaluate(v2.scale.expr, 1.5) == pytest.approx(1.5)  # keyframe curve on the timeline clock
    assert v3.scale.value == 0.5 and v3.x.value == 100.0 and v3.rot.value == 15.0 and v3.opacity.value == 0.5 and v3.rotated
    assert v3.t0 == pytest.approx(0.5) and v3.t1 == pytest.approx(1.5) and v1.base == pytest.approx(max(1920 / by["red.png"].width, 1080 / by["red.png"].height))  # cover fit
    assert len(comp.ass.events) > 3 and any(e.layer == 2 for e in comp.ass.events) and any(e.layer == 1 for e in comp.ass.events)
    # a window cuts clips: only what overlaps it, with the source start moved forward accordingly
    cut = comp.compile_chunk(ChunkPlan(1, 1.2, 1.8, 18))
    assert [l.track_id for l in cut.layers] == ["track_v1", "track_v2", "track_v3"] and cut.layers[0].t0 == pytest.approx(1.2) and cut.layers[0].t1 == pytest.approx(1.8)


def test_chunk_plan_splits_at_scenes_and_covers_every_frame(project_ws, tmp_path):
    from app.analysis.models import Scene
    from app.rendering.planner import plan_chunks

    ws = project_ws
    ws.project.scenes = [Scene(f"s{i}", str(i), i * 10.0, (i + 1) * 10.0) for i in range(6)]
    put(ws, "track_v1", Clip("c1", "", "x", 0.0, 60.0))
    snap = RenderSnapshot.from_project(ws.project, ws.project.render_settings)
    ch = plan_chunks(snap, 30, 25.0)
    assert [(c.start, c.end) for c in ch] == [(0.0, 30.0), (30.0, 60.0)] or sum(c.frames for c in ch) == 1800
    assert sum(c.frames for c in ch) == 1800 and all(a.end == b.start for a, b in zip(ch, ch[1:])) and ch[0].scene_ids
    small = plan_chunks(snap, 24, 1.0, max_seconds=12.0)
    assert len(small) == 6 and sum(c.frames for c in small) == 1440  # one section per scene when scenes are the natural cut points


# ================================================================ video: cuts, frame accuracy, fps
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_cuts_land_on_the_exact_frame_for_every_project_frame_rate(solids, render_engine, tmp_path, fps):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 2.0))
    put(ws, "track_v1", media_clip(by["blue.png"], 2.0, 1.0))
    res, spec = render_to(render_engine, ws, tmp_path, f"cut{fps}", fps=fps)
    assert spec.resolved.fps == fps and res.duration == pytest.approx(3.0) and res.report.ok
    n = 2 * fps
    assert near(px(frame_at(res.output_path, n=n - 1), 0.5, 0.5), RED, 25) and near(px(frame_at(res.output_path, n=n), 0.5, 0.5), BLUE, 25)  # the very frame of the cut
    info = render_engine.probe.probe(res.output_path)
    assert info.fps == pytest.approx(fps, abs=1e-6) and info.frame_count == 3 * fps and (info.width, info.height) == (854, 480)


def test_source_in_out_trim_and_speed_are_honoured(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["rgb.mp4"], 0.0, 1.0, source_in=1.0))  # starts in the green second
    c = put(ws, "track_v1", media_clip(by["rgb.mp4"], 1.0, 1.5, speed=2.0))  # 3 source seconds in 1.5 s: red, green, blue
    c.source_out = 3.0
    res, _ = render_to(render_engine, ws, tmp_path, "trim")
    assert near(px(frame(res, 0.2), 0.5, 0.5), GREEN, 40) and near(px(frame(res, 0.8), 0.5, 0.5), GREEN, 40)
    assert near(px(frame(res, 1.2), 0.5, 0.5), RED, 40) and near(px(frame(res, 1.75), 0.5, 0.5), GREEN, 40) and near(px(frame(res, 2.25), 0.5, 0.5), BLUE, 40)


def test_source_with_a_different_frame_rate_keeps_its_timing(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["rgb.mp4"], 0.0, 3.0))  # a 24 fps source in a 30 fps project
    res, _ = render_to(render_engine, ws, tmp_path, "fps24in30")
    info = render_engine.probe.probe(res.output_path)
    assert info.frame_count == 90 and near(px(frame(res, 0.5), 0.5, 0.5), RED, 40) and near(px(frame(res, 1.5), 0.5, 0.5), GREEN, 40) and near(px(frame(res, 2.5), 0.5, 0.5), BLUE, 40)
    assert near(px(frame_at(res.output_path, n=29), 0.5, 0.5), RED, 40) and near(px(frame_at(res.output_path, n=30), 0.5, 0.5), GREEN, 40)  # the switch at 1.000 s


@pytest.mark.parametrize("resolution,size", [("1080p", (1920, 1080)), ("2160p", (3840, 2160))])
def test_1080p_and_4k_exports_validate(solids, render_engine, tmp_path, resolution, size):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 1.0))
    put(ws, "track_v5", text_clip(0.2, 0.6, "4K OK"))
    res, spec = render_to(render_engine, ws, tmp_path, f"res{resolution}", resolution=resolution)
    info = render_engine.probe.probe(res.output_path)
    assert (info.width, info.height) == size and info.codec == "h264" and res.report.ok and spec.resolved.encoder == "libx264"


@pytest.mark.parametrize("aspect,size_1080", [("9:16", (1080, 1920)), ("1:1", (1080, 1080))])
def test_vertical_and_square_projects_render_without_stretching(ws, tmp_path, render_engine, aspect, size_1080):
    ws.new_project("Aspect", tmp_path / "p", aspect_ratio=aspect)
    d = tmp_path / "m"
    d.mkdir()
    ws.media.import_files([two_color_image(d / "two.png"), solid_image(d / "wide.png", "red", "640x360")])
    assert ws.jobs.wait_idle(30)
    by = {a.name: a for a in ws.project.assets.all()}
    put(ws, "track_v1", media_clip(by["wide.png"], 0.0, 1.0, effects={"fit": "contain"}))
    res, spec = render_to(render_engine, ws, tmp_path, "aspect", resolution="1080p")
    info = render_engine.probe.probe(res.output_path)
    assert (info.width, info.height) == size_1080
    img = frame(res, 0.5)
    if aspect == "9:16":  # a 16:9 picture "contained" in a portrait frame: full width, black bars above and below, never distorted
        w_red = run_row(img, 0.5, RED)
        h_red = run_col(img, 0.5, RED)
        assert w_red > img.width * 0.95 and abs(h_red / w_red - 9 / 16) < 0.03 and near(px(img, 0.5, 0.05), BLACK)


def test_fill_and_fit_modes_scale_without_distortion(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 1.0))  # fills the canvas (cover)
    put(ws, "track_v2", media_clip(by["redbox.png"], 0.0, 1.0, effects={"fit": "contain"}, scale=0.5))  # 2:1 picture contained, then half size
    res, _ = render_to(render_engine, ws, tmp_path, "fit")
    img = frame(res, 0.5)
    w, h = run_row(img, 0.5, RED), run_col(img, 0.5, RED)
    assert near(px(img, 0.02, 0.02), BLUE) and near(px(img, 0.98, 0.98), BLUE)
    assert w == pytest.approx(img.width * 0.5, abs=6) and h == pytest.approx(img.width * 0.25, abs=6) and w / h == pytest.approx(2.0, abs=0.12)  # 2:1 preserved


def test_crop_selects_part_of_the_picture(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["two.png"], 0.0, 1.0, effects={"crop": [0.0, 0.0, 0.5, 1.0]}))  # the red half only
    res, _ = render_to(render_engine, ws, tmp_path, "crop")
    img = frame(res, 0.5)
    assert near(px(img, 0.05, 0.5), RED, 30) and near(px(img, 0.95, 0.5), RED, 30) and not any(near(px(img, x / 10, 0.5), BLUE, 60) for x in range(1, 10))


def test_transparent_png_keeps_its_alpha(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 1.0))
    put(ws, "track_v2", media_clip(by["alpha.png"], 0.0, 1.0, effects={"fit": "contain"}))
    res, _ = render_to(render_engine, ws, tmp_path, "alpha")
    img = frame(res, 0.5)
    assert near(px(img, 0.25, 0.5), RED, 30) and near(px(img, 0.75, 0.5), BLUE, 30)  # the right half of the PNG is transparent: the blue below shows through


# ================================================================ transforms and keyframes
def test_position_scale_rotation_and_opacity_are_rendered(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    put(ws, "track_v2", media_clip(by["redbox.png"], 0.0, 2.0, effects={"fit": "contain"}, scale=0.25, position=(480.0, 0.0)))  # 25% of the width, 480 canvas px to the right
    put(ws, "track_v3", media_clip(by["redbox.png"], 0.0, 2.0, effects={"fit": "contain"}, scale=0.25, position=(-480.0, 0.0), opacity=0.5))
    res, _ = render_to(render_engine, ws, tmp_path, "xform")
    img = frame(res, 1.0)
    W = img.width
    cx = int(W * (0.5 + 480 / 1920))
    assert near(img.getpixel((cx, img.height // 2)), RED, 30) and near(px(img, 0.5, 0.5), BLUE, 30)
    left = img.getpixel((int(W * (0.5 - 480 / 1920)), img.height // 2))
    assert near(left, (127, 0, 127), 25)  # 50% red over blue
    # rotation: a 2:1 box turned 90 degrees is 1:2
    put(ws, "track_v4", media_clip(by["redbox.png"], 0.0, 2.0, effects={"fit": "contain"}, scale=0.4, rotation=90.0, position=(0.0, 0.0)))
    res2, _ = render_to(render_engine, ws, tmp_path, "rot")
    img2 = frame(res2, 1.0)
    # the 90° box is tall: its height ≈ 0.4 * canvas width (864 px of 1920 -> 0.45 of the 854 px frame width... clipped by the frame), its width ≈ half of that
    col, row = span_col(img2, 0.5, RED, 0.5, 50), span_row(img2, 0.5, RED, 0.5, 50)
    assert col and row
    h, w = col[1] - col[0] + 1, row[1] - row[0] + 1
    assert h == pytest.approx(0.4 * 1920 * (img2.width / 1920), abs=10) and w == pytest.approx(h / 2, abs=10)  # a 2:1 box turned 90 degrees is 1:2


def test_keyframed_scale_position_and_opacity_follow_the_curve(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    c = put(ws, "track_v2", media_clip(by["redbox.png"], 0.0, 2.0, effects={"fit": "contain"}, scale=0.2))
    c.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 2.0, 2.0), Keyframe("position_x", 0.0, -400.0), Keyframe("position_x", 2.0, 400.0)]
    res, _ = render_to(render_engine, ws, tmp_path, "kf")
    W = frame(res, 0.0).width
    for t, scale, x in ((0.1, 1.05, -380.0), (1.0, 1.5, 0.0), (1.9, 1.95, 380.0)):
        img = frame(res, t)
        width = run_row(img, 0.5, RED, 60)
        assert width == pytest.approx(0.2 * scale * 1920 * (W / 1920), abs=0.04 * W), f"t={t}"
        reds = [x_ for x_ in range(img.width) if near(img.getpixel((x_, img.height // 2)), RED, 60)]
        assert (reds[0] + reds[-1]) / 2 == pytest.approx(W / 2 + x * W / 1920, abs=0.03 * W), f"t={t}"
    c2 = put(ws, "track_v3", media_clip(by["green.png"], 0.0, 2.0))
    c2.keyframes = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 2.0, 1.0)]
    res2, _ = render_to(render_engine, ws, tmp_path, "kfop")
    p = px(frame(res2, 1.0), 0.02, 0.02)  # corner: only blue + green layers there
    assert 45 <= p[1] <= 80 and 100 <= p[2] <= 150  # about half green (#008000) over blue


def test_transitions_come_from_the_timeline_with_their_own_durations(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    fade = put(ws, "track_v1", media_clip(by["red.png"], 2.0, 2.0))
    fade.transition = {"type": "FADE", "duration": 1.0}
    res, _ = render_to(render_engine, ws, tmp_path, "fade")
    assert near(px(frame(res, 2.5), 0.5, 0.5), (127, 0, 0), 30) and near(px(frame(res, 3.4), 0.5, 0.5), RED, 25) and near(px(frame(res, 1.9), 0.5, 0.5), BLUE, 25)  # not hard-coded: 1 s ramp
    fade.transition = {"type": "DISSOLVE", "duration": 1.0}
    res, _ = render_to(render_engine, ws, tmp_path, "dissolve")
    mid = px(frame(res, 2.5), 0.5, 0.5)
    assert mid[0] > 100 and mid[2] > 30 and mid[1] < 30  # red coming in over blue going out
    fade.transition = {"type": "WIPE", "duration": 1.0}
    res, _ = render_to(render_engine, ws, tmp_path, "wipe")
    img = frame(res, 2.5)
    assert near(px(img, 0.25, 0.5), RED, 30) and not near(px(img, 0.75, 0.5), RED, 80) and near(px(frame(res, 3.2), 0.9, 0.5), RED, 30)
    fade.transition = {"type": "SLIDE", "duration": 1.0}
    res, _ = render_to(render_engine, ws, tmp_path, "slide")
    img = frame(res, 2.5)
    assert near(px(img, 0.75, 0.5), RED, 40) and not near(px(img, 0.2, 0.5), RED, 80)  # the new shot enters from the right
    fade.transition = {"type": "CUT", "duration": 0.0}
    res, _ = render_to(render_engine, ws, tmp_path, "cut")
    assert near(px(frame(res, 2.05), 0.5, 0.5), RED, 25)


def test_timeline_data_is_untouched_by_rendering(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 1.0))
    put(ws, "track_v6", caption_clip(0.2, ["One", "two", "three"]))
    before = ws.project.to_document()["timeline"]
    render_to(render_engine, ws, tmp_path, "keep")
    assert ws.project.to_document()["timeline"] == before


# ================================================================ text, captions, graphics
def _bright(img, box) -> int:
    x0, y0, x1, y1 = (int(box[0] * img.width), int(box[1] * img.height), int(box[2] * img.width), int(box[3] * img.height))
    return sum(1 for x in range(x0, x1) for y in range(y0, y1) if sum(img.getpixel((x, y))) > 600)


def test_headline_text_is_drawn_where_and_when_the_timeline_says(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 3.0))
    put(ws, "track_v5", text_clip(1.0, 1.5, "HEADLINE", pos=(0.5, 0.25)))
    res, _ = render_to(render_engine, ws, tmp_path, "text")
    assert _bright(frame(res, 0.5), (0.2, 0.1, 0.8, 0.4)) < 20 and _bright(frame(res, 1.8), (0.2, 0.1, 0.8, 0.4)) > 150 and _bright(frame(res, 1.8), (0.2, 0.6, 0.8, 0.95)) < 20
    assert _bright(frame(res, 2.9), (0.2, 0.1, 0.8, 0.4)) < 20  # gone after its end
    # the text moved with the timeline data: another position, another place
    clip = next(c for c in ws.project.timeline.all_clips() if c.kind == "text")
    clip.text["position"] = [0.5, 0.75]
    res2, _ = render_to(render_engine, ws, tmp_path, "text2")
    assert _bright(frame(res2, 1.8), (0.2, 0.1, 0.8, 0.4)) < 20 and _bright(frame(res2, 1.8), (0.2, 0.6, 0.8, 0.95)) > 150


def test_number_counter_counts_up_and_ends_on_the_real_figure(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 3.0))
    counter = {"from": 0.0, "to": 1500.0, "decimals": 0, "prefix": "$", "suffix": "", "thousands": True}
    put(ws, "track_v5", text_clip(0.5, 2.0, "$1,500", "NUMBER_CARD", (0.5, 0.5), 110, counter=counter, anim={"in": anim_spec("counter", duration=1.0), "out": anim_spec("fade_out")}))
    res, spec = render_to(render_engine, ws, tmp_path, "counter")
    a, b, c = frame(res, 0.8), frame(res, 1.1), frame(res, 2.0)
    box = (0.2, 0.35, 0.8, 0.65)
    assert _bright(a, box) != _bright(c, box) and a.tobytes() != b.tobytes() and b.tobytes() != c.tobytes()  # the figure changes while counting
    assert frame(res, 1.7).tobytes() == frame(res, 1.9).tobytes() or abs(_bright(frame(res, 1.7), box) - _bright(frame(res, 1.9), box)) < 5  # then holds still


def test_captions_render_with_style_and_word_emphasis(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 3.0))
    put(ws, "track_v6", caption_clip(0.5, "Silver is running out of supply".split(), dt=0.3))
    res, _ = render_to(render_engine, ws, tmp_path, "caps")
    on, off = frame(res, 1.2), frame(res, 0.2)
    assert on.tobytes() != off.tobytes()
    area = (0.15, 0.78, 0.85, 0.97)
    assert _bright(on, area) > 100  # white text near the bottom safe area
    assert _bright(off, area) < 5
    # the last word carries the emphasis colour (#F2C14E in the professional style) once it is reached
    late = frame(res, 2.05)
    gold = sum(1 for x in range(late.width) for y in range(int(late.height * 0.7), late.height) if near(late.getpixel((x, y)), (242, 193, 78), 35))
    assert gold > 30
    # captions are only burned into the *export*: the project still has editable caption objects
    assert sum(1 for c in ws.project.timeline.all_clips() if c.kind == "caption") == 1
    ws.project.caption_settings.enabled = False
    res2, _ = render_to(render_engine, ws, tmp_path, "nocaps")
    assert _bright(frame(res2, 1.2), area) < 5  # turning captions off removes them from the export (the clips stay)


def test_caption_style_overrides_and_position_change_the_picture(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    cap = put(ws, "track_v6", caption_clip(0.2, ["Top", "caption", "here"], dt=0.4))
    cap.text["position"] = "top"
    res, _ = render_to(render_engine, ws, tmp_path, "captop")
    assert _bright(frame(res, 1.0), (0.15, 0.03, 0.85, 0.25)) > 100 and _bright(frame(res, 1.0), (0.15, 0.78, 0.85, 0.97)) < 5


def test_evidence_highlight_draws_a_box_and_dims_the_surroundings(solids, render_engine, tmp_path):
    from app.tests.render_helpers import highlight_clip

    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 3.0))
    put(ws, "track_v4", highlight_clip(0.5, 2.0, region=(0.3, 0.3, 0.4, 0.4), dim=True))
    res, _ = render_to(render_engine, ws, tmp_path, "evidence")
    img = frame(res, 1.5)
    inside, outside = px(img, 0.5, 0.5), px(img, 0.05, 0.05)
    assert near(inside, RED, 25) and outside[0] < 200 and outside[0] > 80  # the surroundings are dimmed, the region is not
    assert near(px(frame(res, 0.2), 0.05, 0.05), RED, 25)  # before the highlight starts
    y0 = int(0.3 * img.height)
    gold = max(sum(1 for x in range(img.width) if near(img.getpixel((x, y)), (242, 193, 78), 60)) for y in range(y0 - 3, y0 + 4))
    assert gold > img.width * 0.25  # the frame edge across the top of the region (0.4 of the width)


def test_missing_font_falls_back_and_is_reported_not_fatal(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    t = put(ws, "track_v5", text_clip(0.2, 1.5, "FALLBACK"))
    t.text["font"] = "Definitely Not Installed Display XL"
    res, spec = render_to(render_engine, ws, tmp_path, "font")
    assert _bright(frame(res, 1.0), (0.2, 0.1, 0.8, 0.5)) > 100  # still drawn
    assert any("Definitely Not Installed Display XL" in w and "using" in w for w in res.warnings)
    assert "not installed" in (spec.run_dir / "render.log").read_text()


# ================================================================ audio
def _audio_project(ws, tmp_path, seconds=8.0):
    from app.tests.helpers import write_tone_wav

    d = tmp_path / "au"
    d.mkdir()
    ws.media.import_files([write_tone_wav(d / "bed.wav", seconds + 1, 0.5, 220.0, 44100), write_tone_wav(d / "voice.wav", seconds, 0.4, 330.0, 48000),
                           write_tone_wav(d / "hit.wav", 0.5, 0.6, 880.0, 22050), solid_image(d / "bg.png", "blue")])
    assert ws.jobs.wait_idle(30)
    return {a.name: a for a in ws.project.assets.all()}


def test_audio_mix_applies_gain_ducking_keyframes_fades_and_sfx_timing(project_ws, render_engine, tmp_path):
    ws = project_ws
    by = _audio_project(ws, tmp_path)
    put(ws, "track_v1", media_clip(by["bg.png"], 0.0, 8.0))
    music = put(ws, "track_a2", media_clip(by["bed.wav"], 0.0, 8.0, audio={"role": "MUSIC", "volume": 1.0, "fade_in": 1.0, "fade_out": 1.0}))
    music.keyframes = [Keyframe("volume", 0.0, 0.5), Keyframe("volume", 2.0, 0.1, "ease_out"), Keyframe("volume", 5.0, 0.1), Keyframe("volume", 6.0, 0.5, "ease_in")]  # ducked 2–5 s
    put(ws, "track_a1", media_clip(by["voice.wav"], 0.0, 8.0, audio={"role": "VOICE", "volume": 1.0}))
    put(ws, "track_a3", media_clip(by["hit.wav"], 3.0, 0.5, audio={"role": "SFX", "volume": 1.0}))
    ws.timeline.set_track_flag("track_a1", "muted", True)  # music and SFX alone, to measure them
    res, spec = render_to(render_engine, ws, tmp_path, "mix")
    a = decode_audio(res.output_path)
    tone = 0.5 / 2 ** 0.5
    normal, ducked = rms(a, 0.6, 0.9), rms(a, 4.0, 4.4)
    assert ducked == pytest.approx(0.1 * tone, rel=0.2) and ducked < rms(a, 6.6, 6.9) * 0.4 and rms(a, 6.6, 6.9) == pytest.approx(0.5 * tone * 0.9, rel=0.25)
    assert rms(a, 0.0, 0.1) < normal * 0.4 and rms(a, 7.9, 8.0) < rms(a, 7.0, 7.1) * 0.5  # fade in / fade out
    # the effect starts when the timeline says (3.0 s), within a few milliseconds
    onset = next(i for i in range(int(2.5 * 16000), len(a)) if abs(a[i]) > 0.2 and i / 16000 > 2.9) / 16000
    assert 3.0 <= onset <= 3.03
    assert abs(a).max() <= 0.98  # never clips (limiter)
    assert res.report.ok and render_engine.probe.probe(res.output_path).sample_rate == 48000


def test_voice_music_and_sfx_mix_stays_below_full_scale_and_voice_stays_dominant(project_ws, render_engine, tmp_path):
    ws = project_ws
    by = _audio_project(ws, tmp_path)
    put(ws, "track_v1", media_clip(by["bg.png"], 0.0, 4.0))
    put(ws, "track_a1", media_clip(by["voice.wav"], 0.0, 4.0, audio={"role": "VOICE", "volume": 1.0}))
    m = put(ws, "track_a2", media_clip(by["bed.wav"], 0.0, 4.0, audio={"role": "MUSIC", "volume": 0.18}))
    put(ws, "track_a3", media_clip(by["hit.wav"], 1.0, 0.5, audio={"role": "SFX", "volume": 0.9}))
    res, _ = render_to(render_engine, ws, tmp_path, "mix2")
    a = decode_audio(res.output_path)
    assert abs(a).max() <= 0.98 and rms(a, 2.0, 3.0) > 0.2  # voice dominates
    ws.timeline.set_track_flag("track_a2", "muted", True)
    res2, _ = render_to(render_engine, ws, tmp_path, "mix_nomusic")
    b = decode_audio(res2.output_path)
    assert rms(b, 2.0, 3.0) < rms(a, 2.0, 3.0)  # without the music bed there is less energy
    assert abs(rms(a, 2.0, 3.0) - rms(b, 2.0, 3.0)) > 0.005  # muting a track changes the mix; solo is honoured too
    ws.timeline.set_track_flag("track_a2", "muted", False)
    ws.timeline.set_track_flag("track_a3", "solo", True)
    res3, _ = render_to(render_engine, ws, tmp_path, "mix_solo")
    c = decode_audio(res3.output_path)
    assert rms(c, 2.0, 3.0) < 0.005 and rms(c, 1.05, 1.4) > 0.1 and m.track_id == "track_a2"


def test_track_gain_and_clip_gain_scale_the_output(project_ws, render_engine, tmp_path):
    ws = project_ws
    by = _audio_project(ws, tmp_path)
    put(ws, "track_v1", media_clip(by["bg.png"], 0.0, 3.0))
    put(ws, "track_a2", media_clip(by["bed.wav"], 0.0, 3.0, audio={"role": "MUSIC", "volume": 1.0}))
    full, _ = render_to(render_engine, ws, tmp_path, "g1")
    ws.timeline.set_track_volume("track_a2", 0.5)
    half, _ = render_to(render_engine, ws, tmp_path, "g2")
    assert rms(decode_audio(half.output_path), 1.0, 2.0) == pytest.approx(rms(decode_audio(full.output_path), 1.0, 2.0) * 0.5, rel=0.1)


def test_a_silent_voice_over_fails_the_export_instead_of_shipping_a_broken_video(project_ws, render_engine, tmp_path):
    from app.core.exceptions import AppError
    from app.tests.helpers import write_tone_wav

    ws = project_ws
    d = tmp_path / "sil"
    d.mkdir()
    ws.media.import_files([write_tone_wav(d / "voice.wav", 4.0, 0.0, 220.0, 48000), solid_image(d / "bg.png", "blue")])
    assert ws.jobs.wait_idle(30)
    by = {a.name: a for a in ws.project.assets.all()}
    put(ws, "track_v1", media_clip(by["bg.png"], 0.0, 4.0))
    put(ws, "track_a1", media_clip(by["voice.wav"], 0.0, 4.0, audio={"role": "VOICE"}))
    ws.project.voice_over.asset_id = by["voice.wav"].id
    with pytest.raises(AppError, match="voice-over is silent"):
        render_to(render_engine, ws, tmp_path, "silentvoice")
    assert not (tmp_path / "out" / "silentvoice.mp4").exists()


def test_output_validator_catches_wrong_duration_codec_and_missing_audio(tmp_path, media_dir, render_engine):
    from app.rendering.validator import Expectations

    v = render_engine.validator
    exp = Expectations(3.0, 320, 180, 30.0, "h264", "aac", True, 44100, [], 0.5)
    rep = v.validate(media_dir / "clip.mp4", exp)
    assert rep.ok and {c.id for c in rep.checks} >= {"exists", "readable", "duration", "resolution", "fps", "video_codec", "audio_codec"}
    bad = v.validate(media_dir / "clip.mp4", replace(exp, duration=9.0, width=1920, height=1080, fps=24.0, video_codec="hevc", expect_audio=True, sample_rate=48000))
    ids = {c.id for c in bad.errors}
    assert not bad.ok and {"duration", "resolution", "fps", "video_codec", "sample_rate"} <= ids and "aspect" not in ids
    noaudio = tmp_path / "na.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media_dir / "clip.mp4"), "-an", "-c:v", "copy", str(noaudio)], check=True)
    assert "audio_stream" in {c.id for c in v.validate(noaudio, exp).errors}
    assert not v.validate(tmp_path / "ghost.mp4", exp).ok and not v.validate(media_dir / "broken.mp4", exp).ok
    z = tmp_path / "empty.mp4"
    z.write_bytes(b"")
    assert v.validate(z, exp).errors[0].id == "exists"


# ================================================================ more codecs, graphics, processing and safety nets
@pytest.mark.parametrize("codec,container,audio,vname", [("h265", "mp4", "aac", "hevc"), ("vp9", "webm", "opus", "vp9"), ("av1", "mkv", "opus", "av1"), ("h264", "mkv", "flac", "h264")])
def test_optional_codecs_and_containers_export_and_validate(solids, render_engine, tmp_path, codec, container, audio, vname):
    from app.rendering.errors import EncoderUnavailableError

    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["red.png"], 0.0, 1.0))
    try:
        res, spec = render_to(render_engine, ws, tmp_path, f"codec_{codec}", video_codec=codec, container=container, audio_codec=audio, hardware_acceleration="cpu")
    except EncoderUnavailableError:
        pytest.skip(f"{codec} encoder not available in this FFmpeg build")
    info = render_engine.probe.probe(res.output_path)
    assert info.codec == vname and res.report.ok and res.output_path.suffix == f".{container}" and abs(info.fps - 30) < 0.1 and spec.resolved.segment_ext == ("ts" if codec in ("h264", "h265") else "mkv")


def test_lower_third_with_title_and_subtitle(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    lt = put(ws, "track_v5", text_clip(0.2, 1.6, "Elon Musk", "LOWER_THIRD", (0.2, 0.8), 54, anim={"in": anim_spec("slide_up", duration=0.3), "out": anim_spec("fade_out", duration=0.3)}))
    lt.text["alignment"], lt.text["background"] = "left", "box"
    res, _ = render_to(render_engine, ws, tmp_path, "lt1")
    img = frame(res, 1.0)
    assert _bright(img, (0.1, 0.7, 0.6, 0.9)) > 80 and _bright(img, (0.1, 0.1, 0.9, 0.5)) < 5
    lt.text["subtitle"] = "CEO of Tesla"
    res2, _ = render_to(render_engine, ws, tmp_path, "lt2")
    assert _bright(frame(res2, 1.0), (0.1, 0.7, 0.6, 0.95)) > _bright(img, (0.1, 0.7, 0.6, 0.95)) + 20  # the second line is there


def test_voice_processing_settings_are_applied_to_the_exported_voice(project_ws, render_engine, tmp_path):
    ws = project_ws
    by = _audio_project(ws, tmp_path)
    put(ws, "track_v1", media_clip(by["bg.png"], 0.0, 3.0))
    put(ws, "track_a1", media_clip(by["voice.wav"], 0.0, 3.0, audio={"role": "VOICE", "volume": 1.0}))
    plain, _ = render_to(render_engine, ws, tmp_path, "vp0")
    ws.project.audio_processing.enabled, ws.project.audio_processing.gain_db = True, -12.0
    quiet, _ = render_to(render_engine, ws, tmp_path, "vp1")
    a, b = decode_audio(plain.output_path), decode_audio(quiet.output_path)
    assert rms(b, 1.0, 2.0) == pytest.approx(rms(a, 1.0, 2.0) * 0.251, rel=0.12)  # -12 dB; the voice file itself is unchanged
    assert ws.project.asset_path(by["voice.wav"]).is_file()


def test_reduced_motion_softens_zooms_in_the_export_like_in_the_preview(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 2.0))
    c = put(ws, "track_v2", media_clip(by["redbox.png"], 0.0, 2.0, effects={"fit": "contain"}, scale=0.2))
    c.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 2.0, 2.0)]
    res, _ = render_to(render_engine, ws, tmp_path, "motion_full")
    ws.project.caption_settings.reduced_motion = True
    soft, _ = render_to(render_engine, ws, tmp_path, "motion_soft")
    full_w, soft_w = run_row(frame(res, 1.95), 0.5, RED), run_row(frame(soft, 1.95), 0.5, RED)
    assert soft_w < full_w * 0.8 and soft_w == pytest.approx(full_w * (1 + 0.95 * 0.4) / (1 + 0.95), rel=0.12)  # the keyframed zoom: 1 + (s-1) * 0.4, on top of the clip's own scale


def test_a_clip_longer_than_its_source_holds_the_last_frame(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["rgb.mp4"], 0.0, 4.5))  # the source is only 3 s: red, green, blue
    res, _ = render_to(render_engine, ws, tmp_path, "hold")
    assert near(px(frame(res, 2.8), 0.5, 0.5), BLUE, 40) and near(px(frame(res, 4.2), 0.5, 0.5), BLUE, 40) and res.duration == pytest.approx(4.5)


def test_hidden_tracks_are_not_rendered(solids, render_engine, tmp_path):
    ws, by = solids, solids.by
    put(ws, "track_v1", media_clip(by["blue.png"], 0.0, 1.0))
    put(ws, "track_v2", media_clip(by["red.png"], 0.0, 1.0))
    ws.timeline.set_track_flag("track_v2", "hidden", True)
    res, _ = render_to(render_engine, ws, tmp_path, "hidden")
    assert near(px(frame(res, 0.5), 0.5, 0.5), BLUE, 30)


def test_a_multi_section_render_equals_a_single_section_render(render_ws, render_engine, tmp_path):
    from dataclasses import replace as rep

    from app.analysis.models import Scene
    from app.rendering.executor import JobSpec
    from app.tests.render_helpers import make_spec

    ws = render_ws
    ws.project.scenes = [Scene("s1", "1", 0.0, 3.0), Scene("s2", "2", 3.0, 5.5), Scene("s3", "3", 5.5, 8.0)]
    one, _ = render_to(render_engine, ws, tmp_path, "single")
    spec, cancel, _ = make_spec(render_engine, ws, tmp_path, "multi")
    spec = rep(spec, chunk_seconds=1.0, chunk_max_seconds=30.0)
    multi = render_engine.run(spec, cancel, lambda p: None)
    assert one.rendered_chunks == 1 and multi.rendered_chunks == 3 and multi.report.ok
    a, b = render_engine.probe.probe(one.output_path), render_engine.probe.probe(multi.output_path)
    assert a.frame_count == b.frame_count == 240 and a.duration == pytest.approx(b.duration, abs=0.05) and b.fps == pytest.approx(30.0, abs=1e-6)
    from PIL import ImageChops

    for t in (0.5, 2.9, 3.1, 5.4, 5.6, 7.5):  # including right before and right after the section joins
        diff = ImageChops.difference(frame(one, t), frame(multi, t)).convert("L")
        assert diff.getextrema()[1] < 80 and __import__('numpy').asarray(diff).mean() < 2.0, t
    x, y = decode_audio(one.output_path), decode_audio(multi.output_path)
    assert abs(len(x) - len(y)) < 400 and rms(x, 2.0, 3.0) == pytest.approx(rms(y, 2.0, 3.0), rel=0.05)
    _ = JobSpec


def test_silent_audio_is_an_error_unless_the_timeline_is_silent_on_purpose(project_ws, render_engine, tmp_path, media_dir):
    from app.rendering.validator import Expectations

    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=320x180:r=30:d=2", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(silent)], check=True)
    exp = Expectations(2.0, 320, 180, 30.0, "h264", "aac", True, 48000, [], 0.5)
    rep = render_engine.validator.validate(silent, exp)
    assert any(c.id == "mute" and c.status == "error" for c in rep.checks) and not rep.ok
    assert render_engine.validator.validate(silent, replace(exp, expect_audible=False)).ok
    assert any(c.id == "audio_duration" and c.status == "ok" for c in rep.checks)
    # every audio clip deliberately at zero volume: the export succeeds (silence was asked for)
    ws = project_ws
    by = _audio_project(ws, tmp_path)
    put(ws, "track_v1", media_clip(by["bg.png"], 0.0, 2.0))
    put(ws, "track_a2", media_clip(by["bed.wav"], 0.0, 2.0, audio={"role": "MUSIC", "volume": 0.0}))
    res, _ = render_to(render_engine, ws, tmp_path, "zero_volume")
    assert res.report.ok and rms(decode_audio(res.output_path), 0.5, 1.5) < 0.001
