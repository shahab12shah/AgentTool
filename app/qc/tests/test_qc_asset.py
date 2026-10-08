"""Asset integrity and media quality checker."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.qc.asset_checker import AssetChecker
from app.qc.issue_model import FixRoute
from app.qc.severity import Severity
from app.qc.tests.conftest import needs_ffmpeg
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, narrate, new_project, qc_ctx, run_checker
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.probe import MediaProbeService
from app.tests.helpers import make_video


def run(p, **kw):
    return run_checker(AssetChecker(), qc_ctx(p, **kw))


def ff_run(p, **kw):
    ff = FFmpegService()
    return run_checker(AssetChecker(), qc_ctx(p, ffmpeg=ff, probe=MediaProbeService(ff), **kw))


def project_with_clip(tmp_path, *, size=(1920, 1080), asset_size=(1920, 1080), kind="video", **clip_kw):
    p = new_project(tmp_path, seconds=20, size=size)
    a = add_asset(p, "city.mp4" if kind == "video" else "pic.png", kind, duration=30 if kind == "video" else None, w=asset_size[0], h=asset_size[1])
    s = add_scene(p, 0, 10, "Silver prices rose sharply.")
    narrate(p)
    c = add_clip(p, "track_v1", a, 0, 10, scene=s, **clip_kw)
    return p, a, s, c


def real_voice(p):
    """Replace the placeholder voice-over file with a decodable WAV of the length the project recorded."""
    import wave

    asset = p.assets.get(p.voice_over.asset_id)
    seconds = int(asset.duration)
    path = p.root / asset.path
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(48000)
        w.writeframes(b"\0\0" * 48000 * seconds)
    p.assets.get(p.voice_over.asset_id).size_bytes = path.stat().st_size


def real_video(p, name="real.mp4", seconds=4, size="640x360"):
    path = p.root / "media" / "video" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    make_video(path, seconds=seconds, size=size)
    a = add_asset(p, name, "video", duration=float(seconds), w=int(size.split("x")[0]), h=int(size.split("x")[1]), write=False)
    a.path, a.size_bytes = f"media/video/{name}", path.stat().st_size
    return a, path


def real_image(p, name, img: Image.Image):
    path = p.root / "media" / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    a = add_asset(p, name, "image", duration=None, w=img.width, h=img.height, write=False)
    a.path, a.size_bytes = f"media/images/{name}", path.stat().st_size
    return a


# ---------------------------------------------------------------- missing assets
def test_missing_asset_is_critical_and_names_the_scene_and_clip(tmp_path):
    p, a, s, c = project_with_clip(tmp_path)
    (p.root / a.path).unlink()
    out = run(p)
    iss = find(out, "asset.missing")
    assert len(iss) == 1
    i = iss[0]
    assert i.severity is Severity.CRITICAL and i.title == "Missing asset"
    assert i.scene_id == s.id and i.timeline_item_id == c.id and i.track_id == "track_v1"
    assert f"Scene {s.label} · {c.id}" in i.affected_elements and a.id in i.affected_elements
    for action in ("Relink", "replace", "remove", "search", "skip"):
        assert action.lower() in i.suggested_fix.lower()
    assert i.fix is not None and i.fix.kind == "asset.replace" and i.fix.route is FixRoute.NAVIGATE and not i.auto_fix_safe
    assert out.metrics["missing"] == 1


def test_one_issue_per_missing_asset_even_when_used_many_times(tmp_path):
    p, a, s, c = project_with_clip(tmp_path)
    add_clip(p, "track_v1", a, 12, 4, scene=s, source_in=5)
    (p.root / a.path).unlink()
    iss = find(run(p), "asset.missing")
    assert len(iss) == 1 and sum(1 for e in iss[0].affected_elements if e.startswith("Scene")) == 2


def test_missing_voice_over_file_is_named_as_such(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    (p.root / p.assets.get(p.voice_over.asset_id).path).unlink()
    iss = find(run(p), "asset.missing")
    assert len(iss) == 1 and iss[0].title == "Missing voice-over" and iss[0].severity is Severity.CRITICAL


def test_unused_missing_asset_is_not_reported(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    spare = add_asset(p, "spare.mp4", "video")
    (p.root / spare.path).unlink()
    assert find(run(p), "asset.missing") == []


def _hash_asset(p, a, data: bytes):
    """Make the asset's recorded facts describe ``data`` and delete its file (so the only copy left is the one a test places)."""
    (p.root / a.path).write_bytes(data)
    a.content_hash, a.size_bytes = hashlib.sha256(data).hexdigest(), len(data)
    (p.root / a.path).unlink()


def test_exact_relink_offered_when_an_identical_copy_exists(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    data = b"identical media bytes" * 20
    _hash_asset(p, a, data)
    copy = p.root / "media" / "elsewhere" / "backup_of_city.mp4"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(data)
    i = find(run(p), "asset.missing")[0]
    assert i.fix.kind == "asset.relink" and i.fix.params["exact"] is True and i.fix.params["new_path"] == str(copy) and i.fix.params["asset_id"] == a.id
    assert i.auto_fix_available and i.auto_fix_safe


def test_exact_relink_survives_a_user_owned_clip(tmp_path):
    """Relinking to an identical file changes none of the user's edits, so the user's ownership of the clip must not disable it."""
    p, a, s, c = project_with_clip(tmp_path, created_by="USER")
    data = b"same bytes" * 30
    _hash_asset(p, a, data)
    copy = p.root / "media" / "elsewhere" / "other_name.mp4"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(data)
    i = find(run(p), "asset.missing")[0]
    assert i.fix.kind == "asset.relink" and i.auto_fix_available


def test_no_relink_fix_for_a_different_file_with_the_same_name(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    _hash_asset(p, a, b"original bytes" * 20)
    other = p.root / "media" / "elsewhere" / "city.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"a different video entirely" * 20)
    i = find(run(p), "asset.missing")[0]
    assert i.fix.kind == "asset.replace" and i.fix.route is FixRoute.NAVIGATE and not i.auto_fix_safe


def test_relink_never_offered_when_the_hash_is_unknown(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    a.content_hash = None
    (p.root / a.path).unlink()
    other = p.root / "media" / "elsewhere" / "city.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"\0" * 64)
    assert find(run(p), "asset.missing")[0].fix.kind == "asset.replace"


# ---------------------------------------------------------------- decodability (real FFmpeg)
@needs_ffmpeg
def test_corrupt_media_is_critical(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    real_voice(p)
    (p.root / a.path).write_text("this is not a video")
    out = ff_run(p)
    iss = find(out, "asset.corrupt")
    assert len(iss) == 1 and iss[0].severity is Severity.CRITICAL and "cannot be decoded" in iss[0].description
    assert out.metrics["corrupt"] == 1


@needs_ffmpeg
def test_unsupported_extension_is_reported_as_unsupported(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    real_voice(p)
    a.path = "media/video/clip.xyz"
    (p.root / a.path).write_bytes(b"\0" * 64)
    assert codes(ff_run(p)).count("asset.unsupported_format") == 1


@needs_ffmpeg
def test_valid_media_has_no_integrity_or_quality_issue(tmp_path):
    p = new_project(tmp_path, seconds=4, size=(640, 360))
    real_voice(p)
    a, _ = real_video(p, size="640x360")
    s = add_scene(p, 0, 4, "Silver rose.")
    add_clip(p, "track_v1", a, 0, 4, scene=s)
    p.render_settings.resolution = "480p"
    out = ff_run(p)
    assert [c for c in codes(out) if c.startswith("asset.") or c in ("media.low_resolution", "media.aspect", "media.upscale")] == []
    assert out.metrics["assets_checked"] >= 1 and out.metrics["corrupt"] == 0


@needs_ffmpeg
def test_file_shorter_than_recorded_is_an_error_when_clips_run_past_its_end(tmp_path):
    p = new_project(tmp_path, seconds=10, size=(640, 360))
    real_voice(p)
    a, _ = real_video(p, seconds=2, size="640x360")
    a.duration = 6.0  # the project believes the file is 6 s long
    s = add_scene(p, 0, 6, "Silver rose.")
    add_clip(p, "track_v1", a, 0, 5, scene=s)
    iss = find(ff_run(p), "asset.modified")
    assert len(iss) == 1 and iss[0].severity is Severity.ERROR and "run past the end" in iss[0].description


def test_changed_file_size_is_a_warning(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    (p.root / a.path).write_bytes(b"\0" * 500)
    iss = find(run(p), "asset.modified")
    assert len(iss) == 1 and iss[0].severity is Severity.WARNING and "size changed" in iss[0].description


def test_invalid_recorded_duration(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    a.duration = 0
    assert find(run(p), "asset.duration_invalid")[0].severity is Severity.ERROR


# ---------------------------------------------------------------- media quality
def test_low_resolution_source_in_a_4k_export_is_a_warning_with_the_numbers(tmp_path):
    p, a, *_ = project_with_clip(tmp_path, asset_size=(640, 360))
    p.render_settings.resolution = "2160p"
    i = find(run(p), "media.low_resolution")[0]
    assert i.severity is Severity.WARNING and "4K" in i.title
    assert "Source 640x360, timeline 3840x2160" in i.description
    assert i.fix.kind == "visual.search_again" and i.fix.route is FixRoute.RESEARCH and not i.auto_fix_safe


def test_low_resolution_threshold_is_a_setting(tmp_path):
    p, a, *_ = project_with_clip(tmp_path, asset_size=(1280, 720))
    assert find(run(p), "media.low_resolution") == []  # 67% of full HD: fine at the default 50%
    p.qc_settings.media.min_source_ratio = 0.8
    assert len(find(run(p), "media.low_resolution")) == 1


def test_full_resolution_source_is_clean(tmp_path):
    p, *_ = project_with_clip(tmp_path)
    assert not [c for c in codes(run(p)) if c.startswith("media.") and c != "media.thumbnail_missing"]


def test_media_is_never_rejected_unless_the_user_asks(tmp_path):
    p, *_ = project_with_clip(tmp_path, asset_size=(320, 180))
    assert find(run(p), "media.rejected") == []
    p.qc_settings.media.reject_below_ratio = 0.3
    out = run(p)
    assert find(out, "media.rejected")[0].severity is Severity.ERROR and find(out, "media.low_resolution") == []


def test_zoom_beyond_the_limit_is_reported_separately(tmp_path):
    p, a, s, c = project_with_clip(tmp_path)
    assert find(run(p), "media.upscale") == []
    c.scale = 3.2
    i = find(run(p), "media.upscale")[0]
    assert i.severity is Severity.WARNING and "3.2" in i.description


def test_portrait_media_in_a_landscape_frame_is_cropped(tmp_path):
    p, a, s, c = project_with_clip(tmp_path, asset_size=(1080, 1920))
    i = find(run(p), "media.aspect")[0]
    assert i.severity is Severity.WARNING and "9:16" in i.description and "cropped" in i.description
    c.effects["fit"] = "contain"
    assert find(run(p), "media.aspect") == []  # letterboxed on purpose


def test_slightly_different_aspect_within_tolerance_is_fine(tmp_path):
    p, *_ = project_with_clip(tmp_path, asset_size=(1920, 1000))
    assert find(run(p), "media.aspect") == []


def _shapes(w=640, h=360, seed=1):
    """Random hard-edged blocks: a picture with real structure (std well above a flat graphic's), sharp until it is blurred."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (90, 120, 160))
    d = ImageDraw.Draw(im)
    for _ in range(60):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        d.rectangle([x, y, x + int(rng.integers(10, 80)), y + int(rng.integers(10, 60))], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return im


def test_blurry_still_image_is_a_low_confidence_notice(tmp_path):
    p = new_project(tmp_path, seconds=20)
    soft = _shapes().filter(ImageFilter.GaussianBlur(9))
    a = real_image(p, "soft.png", soft)
    s = add_scene(p, 0, 10, "x")
    add_clip(p, "track_v1", a, 0, 10, scene=s)
    i = find(run(p), "media.blurry")[0]
    assert i.severity is Severity.NOTICE and i.confidence == 65.0


def test_sharp_and_flat_images_are_not_reported_as_blurry(tmp_path):
    p = new_project(tmp_path, seconds=20)
    s = add_scene(p, 0, 10, "x")
    sharp = real_image(p, "sharp.png", _shapes(seed=3))
    flat = real_image(p, "flat.png", Image.new("RGB", (640, 360), (200, 30, 30)))
    add_clip(p, "track_v1", sharp, 0, 5, scene=s)
    add_clip(p, "track_v1", flat, 5, 5, scene=s)
    assert find(run(p), "media.blurry") == []


def test_missing_thumbnails_are_one_notice_and_cached_ones_are_not(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    i = find(run(p), "media.thumbnail_missing")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE
    from app.media.thumbnails import ThumbnailService

    t = ThumbnailService.thumbnail_path(p.root, a)
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(b"jpg")
    assert find(run(p), "media.thumbnail_missing") == []


# ---------------------------------------------------------------- plumbing
def test_checker_declares_its_inputs_and_does_not_modify_the_project(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    c = AssetChecker()
    assert set(c.categories) and c.settings_sections == ("media",) and "assets" in c.domains and not c.scene_local
    before = p.to_document()
    ctx = qc_ctx(p)
    h1 = c.input_hash(ctx)
    run_checker(c, ctx)
    assert p.to_document() == before
    (p.root / a.path).unlink()
    assert c.input_hash(qc_ctx(p)) != h1  # a vanished file invalidates the cached result


def test_fingerprints_are_stable_between_runs(tmp_path):
    p, a, *_ = project_with_clip(tmp_path)
    (p.root / a.path).unlink()
    assert find(run(p), "asset.missing")[0].fingerprint == find(run(p), "asset.missing")[0].fingerprint


def test_cancellation_is_honoured(tmp_path):
    from app.qc.context import QCCancelled

    p, a, *_ = project_with_clip(tmp_path)
    ctx = qc_ctx(p)
    ctx.cancel.set()
    with pytest.raises(QCCancelled):
        AssetChecker().run(ctx, lambda f, m: None)
