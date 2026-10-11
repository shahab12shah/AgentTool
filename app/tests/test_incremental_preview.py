"""Preview invalidation: the section cache keys change exactly for the scenes an edit can affect, for every section when something global changes, and the cached frames of an asset follow
the asset's file."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.analysis.models import Scene
from app.preview.frames import FrameProvider
from app.rendering.commands import SetRenderSettingsCommand
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import make_video, write_tone_wav
from app.tests.render_helpers import caption_clip, media_clip, put, text_clip

pytestmark = needs_ffmpeg
SCENES = ("s1", "s2", "s3", "s4")


@pytest.fixture
def pv(project_ws, tmp_path):
    """Four 2-second scenes: one video clip each on V1, captions in s1-s3 (none in s4), a title in s1, a music bed under everything."""
    ws = project_ws
    d = tmp_path / "pv"
    d.mkdir()
    files = [make_video(d / "a.mp4", 8.0, "testsrc", "320x180", 30), make_video(d / "b.mp4", 8.0, "testsrc2", "320x180", 30), write_tone_wav(d / "bed.wav", 8.0, 0.3, 110.0, 48000),
             write_tone_wav(d / "hit.wav", 0.6, 0.5, 880.0, 48000)]
    ws.media.import_files(files)
    assert ws.jobs.wait_idle(120)
    ws.by = {a.name: a for a in ws.project.assets.all()}
    ws.clips = {}
    for i, sid in enumerate(SCENES):
        ws.clips[sid] = put(ws, "track_v1", media_clip(ws.by["a.mp4"], 2.0 * i, 2.0, source_in=2.0 * i, scene_id=sid))
    ws.caps = {sid: put(ws, "track_v6", caption_clip(2.0 * i + 0.2, ["Scene", "number", str(i + 1), "speaks"], 0.3, scene_id=sid)) for i, sid in enumerate(SCENES[:3])}
    put(ws, "track_v5", text_clip(0.2, 1.2, "TITLE"))
    ws.music = put(ws, "track_a2", media_clip(ws.by["bed.wav"], 0.0, 8.0, audio={"role": "MUSIC", "volume": 1.0}))
    ws.project.scenes = [Scene(sid, str(i + 1), 2.0 * i, 2.0 * i + 2.0) for i, sid in enumerate(SCENES)]
    ws.project.voice_over.duration = 8.0
    ws.render.update_settings(resolution="480p", quality="draft", preset_id="custom")
    return ws


def plan(ws, mode: str = "draft"):
    p = ws.render.preview.plan(ws.render.snapshot(), mode)
    assert [s.scene_ids for s in p.sections] == [[s] for s in SCENES]
    return {s.scene_ids[0]: s.key for s in p.sections}, p.audio_key, p


def changed(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {k for k in before if before[k] != after[k]}


def test_a_replaced_visual_changes_only_its_scene_section(pv):
    ws = pv
    k0, a0, _ = plan(ws)
    assert len(set(k0.values())) == 4
    ws.clips["s2"].asset_id = ws.by["b.mp4"].id
    k1, a1, _ = plan(ws)
    assert changed(k0, k1) == {"s2"} and a1 == a0


def test_a_clip_timing_change_changes_only_its_scene_section(pv):
    ws = pv
    k0, a0, _ = plan(ws)
    ws.timeline.trim_clip(ws.clips["s3"].id, new_end=ws.clips["s3"].timeline_end - 0.5)
    k1, a1, _ = plan(ws)
    assert changed(k0, k1) == {"s3"}
    ws.timeline.move_clip(ws.caps["s1"].id, ws.caps["s1"].timeline_start + 0.3)
    k2, _a2, _ = plan(ws)
    assert changed(k1, k2) == {"s1"}


def test_a_caption_edit_and_its_layout_change_only_that_scene(pv):
    ws = pv
    k0, a0, _ = plan(ws)
    ws.caps["s2"].text["text"], ws.caps["s2"].text["lines"] = "something else entirely", ["something else", "entirely"]
    k1, a1, _ = plan(ws)
    assert changed(k0, k1) == {"s2"} and a1 == a0
    ws.caps["s2"].text["lines"] = ["something", "else entirely"]  # same words, different line break: a layout change alone
    k2, _a, _ = plan(ws)
    assert changed(k1, k2) == {"s2"}
    ws.caps["s3"].text["position"] = "top"
    k3, _a, _ = plan(ws)
    assert changed(k2, k3) == {"s3"}


def test_a_caption_that_straddles_two_sections_invalidates_both(pv):
    ws = pv
    straddle = put(ws, "track_v6", caption_clip(3.6, ["Across", "the", "cut", "line"], 0.2))  # 3.6 s - 4.4 s: the s2/s3 boundary is at 4.0 s
    k0, _a, _ = plan(ws)
    straddle.text["text"], straddle.text["lines"] = "Edited across the cut", ["Edited across", "the cut"]
    k1, _a, _ = plan(ws)
    assert changed(k0, k1) == {"s2", "s3"}


def test_a_global_caption_setting_changes_every_section_that_shows_captions(pv):
    ws = pv
    k0, a0, _ = plan(ws)
    ws.project.caption_settings = replace(ws.project.caption_settings, large_text=True)
    k1, a1, _ = plan(ws)
    assert changed(k0, k1) == {"s1", "s2", "s3"} and a1 == a0  # s4 has no overlay at all: nothing of it depends on caption styles


def test_audio_changes_move_the_audio_key_and_no_video_section(pv):
    ws = pv
    k0, a0, p0 = plan(ws)
    assert a0 and p0.audio_cached is False
    ws.timeline.set_track_volume("track_a2", 0.4)
    k1, a1, _ = plan(ws)
    assert k1 == k0 and a1 != a0  # the mix is re-made, the pictures are not
    put(ws, "track_a3", media_clip(ws.by["hit.wav"], 3.8, 0.6, audio={"role": "SFX", "volume": 0.6}))  # a sound that crosses the s2/s3 boundary
    k2, a2, _ = plan(ws)
    assert k2 == k0 and a2 != a1


def test_a_dissolve_depends_on_the_shot_before_it(pv):
    ws = pv
    k0, _a, _ = plan(ws)
    ws.clips["s3"].transition = {"type": "DISSOLVE", "duration": 0.5}
    k1, _a, _ = plan(ws)
    assert changed(k0, k1) == {"s3"}  # the transition itself lives in the section that starts the shot
    ws.clips["s2"].asset_id = ws.by["b.mp4"].id  # the held last frame of the previous shot is part of the dissolve
    k2, _a, _ = plan(ws)
    assert changed(k1, k2) == {"s2", "s3"}
    ws.clips["s3"].transition = None
    k3, _a, _ = plan(ws)
    assert changed(k2, k3) == {"s3"}


def test_export_settings_do_not_touch_the_preview_but_project_wide_changes_touch_every_section(pv):
    ws = pv
    k0, a0, _ = plan(ws)
    ws.commands.execute(SetRenderSettingsCommand(ws.project, replace(ws.project.render_settings, resolution="2160p", fps=24, video_codec="h265", quality="maximum", use_proxies=True)))
    k1, a1, _ = plan(ws)
    assert (k1, a1) == (k0, a0), "a preview is made with its own mode settings: the export choices must not throw it away"
    ws.project.settings.fps = 24  # the project's frame rate is the master clock of every section
    k2, a2, _ = plan(ws)
    assert changed(k1, k2) == set(SCENES)
    ws.project.settings.fps = 30
    ws.project.settings.width, ws.project.settings.height = 1280, 720
    k3, _a, _ = plan(ws)
    assert changed(k0, k3) == set(SCENES)


def test_preview_modes_never_share_sections(pv):
    ws = pv
    kd, _a, pd = plan(ws, "draft")
    kr, _a, pr = plan(ws, "realtime")
    kh, _a, ph = plan(ws, "high")
    assert not (set(kd.values()) & set(kr.values())) and not (set(kd.values()) & set(kh.values())) and not (set(kr.values()) & set(kh.values()))
    assert pd.cached_sections == pr.cached_sections == ph.cached_sections == 0


# ---------------------------------------------------------------------------------------------------------- frame cache
def test_cached_frames_follow_the_assets_file(pv, tmp_path):
    ws = pv
    asset = ws.by["a.mp4"]
    fp = FrameProvider(lambda: ws.project, lambda: ws.settings.ffmpeg_path)
    f1 = fp.frame_path(asset, 1.0)
    assert f1 is not None and f1.is_file() and f1.parent == ws.project.root / "previews" / "frames"
    assert fp.frame_path(asset, 1.0) == f1  # unchanged file: the same cached frame
    other = fp.frame_path(ws.by["b.mp4"], 1.0)
    assert other is not None and other != f1
    before = sorted(p.name for p in f1.parent.glob(f"{asset.id}_*.jpg"))
    # the file behind the asset is replaced (a relink to a different take): the old pictures must not be served any more
    src = ws.project.asset_path(asset)
    make_video(tmp_path / "replacement.mp4", 8.0, "testsrc2", "320x180", 30)
    src.write_bytes((tmp_path / "replacement.mp4").read_bytes())
    f2 = fp.frame_path(asset, 1.0)
    assert f2 is not None and f2 != f1 and f2.is_file()
    assert not f1.is_file(), "the frames cut from the old file are removed"
    assert other.is_file(), "another asset's frames are untouched"
    assert len(before) == 1 and sorted(p.name for p in f1.parent.glob(f"{asset.id}_*.jpg")) == [f2.name]
    # explicit invalidation (asset removed / relinked elsewhere)
    fp._failed.add(str(f2.parent / f"{asset.id}_deadbeef_000100.jpg"))
    assert fp.invalidate_asset(asset.id) == 1 and not f2.is_file() and other.is_file()
    assert not any(k for k in fp._failed if Path(k).name.startswith(asset.id))


def test_frames_of_an_old_version_without_fingerprint_are_swept_not_served(pv):
    ws = pv
    asset = ws.by["a.mp4"]
    frames = ws.project.root / "previews" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    legacy = frames / f"{asset.id}_000100.jpg"
    legacy.write_bytes(b"old frame from before the fingerprint was part of the name")
    fp = FrameProvider(lambda: ws.project, lambda: ws.settings.ffmpeg_path)
    f = fp.frame_path(asset, 1.0)
    assert f is not None and f != legacy and not legacy.exists()
