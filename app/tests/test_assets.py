from __future__ import annotations

import pytest

from app.core.exceptions import MediaError, MediaProbeError, UnsupportedMediaError
from app.media.asset import Asset, AssetType, SourceType
from app.media.asset_registry import AssetRegistry
from app.media.importer import LINK_REFERENCE
from app.media.metadata import asset_type_for
from app.tests.conftest import import_and_wait, needs_ffmpeg, settle


def _asset(i: str, h: str | None = None) -> Asset:
    return Asset(id=i, type=AssetType.VIDEO, source_type=SourceType.USER_MEDIA, path=f"media/video/{i}.mp4", name=f"{i}.mp4", content_hash=h)


def test_registry_add_get_remove_and_stable_ids():
    reg = AssetRegistry()
    first = reg.new_id()
    assert first == "media_00001"
    reg.add(_asset(first, "h1"))
    assert reg.get(first) is not None and first in reg and len(reg) == 1
    with pytest.raises(MediaError):
        reg.add(_asset(first))
    reg.remove(first)
    assert reg.get(first) is None
    assert reg.new_id() == "media_00002"  # ids are never reused
    with pytest.raises(MediaError):
        reg.remove("media_00001")


def test_registry_duplicate_lookup_and_roundtrip():
    reg = AssetRegistry([_asset("media_00007", "abc")])
    assert reg.find_by_hash("abc").id == "media_00007"
    assert reg.find_by_hash("zzz") is None
    assert reg.new_id() == "media_00008"
    clone = [Asset.from_dict(d) for d in reg.to_list()]
    assert clone[0] == reg.get("media_00007")


def test_asset_type_classification():
    assert asset_type_for("a.MP4") is AssetType.VIDEO
    assert asset_type_for("a.webp") is AssetType.IMAGE
    assert asset_type_for("a.m4a") is AssetType.AUDIO
    with pytest.raises(UnsupportedMediaError):
        asset_type_for("a.exe")


@needs_ffmpeg
def test_import_video_image_audio_with_real_metadata(project_ws, media_dir):
    ws = project_ws
    v = import_and_wait(ws, media_dir / "clip.mp4")
    i = import_and_wait(ws, media_dir / "pic.png")
    a = import_and_wait(ws, media_dir / "voice.wav")
    assert v.type is AssetType.VIDEO and (v.width, v.height) == (320, 180)
    assert v.fps == pytest.approx(30, abs=0.01) and v.duration == pytest.approx(3, abs=0.2)
    assert v.has_audio and v.codec == "h264"
    assert i.type is AssetType.IMAGE and i.duration is None and (i.width, i.height) == (64, 64)
    assert a.type is AssetType.AUDIO and a.duration == pytest.approx(4, abs=0.1) and a.sample_rate
    root = ws.project.root
    assert (root / v.path).is_file() and v.path.startswith("media/video/")
    assert i.path.startswith("media/images/") and a.path.startswith("media/audio/")
    assert v.source_type is SourceType.USER_MEDIA and v.link_mode == "copy"
    assert [x.id for x in ws.project.assets] == ["media_00001", "media_00002", "media_00003"]


@needs_ffmpeg
def test_duplicate_import_is_not_copied_twice(project_ws, media_dir, tmp_path):
    ws = project_ws
    first = import_and_wait(ws, media_dir / "clip.mp4")
    renamed = tmp_path / "same_content_other_name.mp4"
    renamed.write_bytes((media_dir / "clip.mp4").read_bytes())
    second = import_and_wait(ws, renamed)
    assert second.id == first.id
    assert len(ws.project.assets) == 1
    assert len(list((ws.project.root / "media" / "video").iterdir())) == 1


@needs_ffmpeg
def test_same_file_twice_in_one_batch_registers_once(project_ws, media_dir):
    ws = project_ws
    ws.media.import_files([media_dir / "other.mp4", media_dir / "other.mp4"])
    settle(ws)
    assert len(ws.project.assets) == 1
    assert len(list((ws.project.root / "media" / "video").glob("*.mp4"))) == 1


@needs_ffmpeg
def test_import_same_filename_different_content_gets_unique_name(project_ws, media_dir, tmp_path):
    ws = project_ws
    import_and_wait(ws, media_dir / "clip.mp4")
    impostor = tmp_path / "clip.mp4"
    impostor.write_bytes((media_dir / "other.mp4").read_bytes())
    b = import_and_wait(ws, impostor)
    assert b.path != "media/video/clip.mp4" and (ws.project.root / b.path).is_file()


@needs_ffmpeg
def test_import_errors_are_reported_not_raised(project_ws, media_dir):
    ws = project_ws
    errors = []
    ws.bus.subscribe("app.error", lambda t, p: errors.append(p["message"]))
    for name in ("broken.mp4", "notes.txt", "missing.mp4"):
        ws.media.import_files([media_dir / name])
    settle(ws)
    assert len(errors) == 3
    assert any("could not be read" in e for e in errors)
    assert any("not a supported" in e for e in errors)
    assert len(ws.project.assets) == 0
    assert list((ws.project.root / "media" / "video").iterdir()) == []  # nothing half-copied


@needs_ffmpeg
def test_linked_import_does_not_copy(project_ws, media_dir):
    ws = project_ws
    got = []
    ws.media.import_files([media_dir / "pic.png"], link_mode=LINK_REFERENCE, on_asset=got.append)
    settle(ws)
    assert got[0].link_mode == "reference"
    assert ws.project.asset_path(got[0]) == (media_dir / "pic.png").resolve()
    assert list((ws.project.root / "media" / "images").iterdir()) == []


@needs_ffmpeg
def test_remove_asset_keeps_file_and_cascades_and_undoes(project_ws, media_dir):
    ws = project_ws
    v = import_and_wait(ws, media_dir / "clip.mp4")
    clip = ws.timeline.add_asset(v.id)
    f = ws.project.asset_path(v)
    ws.media.remove_asset(v.id)
    assert v.id not in ws.project.assets and ws.project.timeline.get_clip(clip.id) is None
    assert f.is_file()  # never deletes media from disk
    ws.undo()
    assert v.id in ws.project.assets and ws.project.timeline.get_clip(clip.id) is not None


@needs_ffmpeg
def test_media_probe_failure_message(ws, tmp_path, media_dir):
    with pytest.raises(MediaProbeError):
        ws.prober.probe(media_dir / "broken.mp4")


@needs_ffmpeg
def test_thumbnails_are_generated_in_background_and_cached(project_ws, media_dir):
    ws = project_ws
    ready = []
    ws.bus.subscribe("media.thumbnail_ready", lambda t, p: ready.append(p["asset_id"]))
    v = import_and_wait(ws, media_dir / "clip.mp4")
    a = import_and_wait(ws, media_dir / "voice.wav")
    i = import_and_wait(ws, media_dir / "pic.png")
    settle(ws)
    for asset in (v, a, i):
        thumb = ws.media.thumbnail_file(asset)
        assert thumb is not None and thumb.stat().st_size > 0
    assert set(ready) == {v.id, a.id, i.id}
    # Cached: no new thumbnail job when asking again.
    before = len(ws.jobs.jobs())
    ws.media.ensure_all_thumbnails()
    assert len(ws.jobs.jobs()) == before
    # Missing: regenerated.
    ws.media.thumbnail_file(v).unlink()
    ws.media.ensure_all_thumbnails()
    settle(ws)
    assert ws.media.thumbnail_file(v) is not None
