from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.logging.logger import setup_logging  # noqa: E402
from app.services.workspace import Workspace  # noqa: E402
from app.storage.paths import AppPaths  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def _run(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging(tmp_path_factory):
    setup_logging(tmp_path_factory.mktemp("logs"), console=False)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    """Tiny real media files generated with ffmpeg."""
    d = tmp_path_factory.mktemp("sample_media")
    if HAS_FFMPEG:
        _run(["-f", "lavfi", "-i", "testsrc=duration=3:size=320x180:rate=30", "-f", "lavfi",
              "-i", "sine=frequency=440:duration=3", "-shortest", "-pix_fmt", "yuv420p", str(d / "clip.mp4")])
        _run(["-f", "lavfi", "-i", "testsrc2=duration=2:size=320x180:rate=24", "-pix_fmt", "yuv420p", str(d / "other.mp4")])
        _run(["-f", "lavfi", "-i", "color=c=red:size=64x64", "-frames:v", "1", str(d / "pic.png")])
        _run(["-f", "lavfi", "-i", "sine=frequency=220:duration=4", str(d / "voice.wav")])
    (d / "broken.mp4").write_bytes(b"this is not a video" * 50)
    (d / "notes.txt").write_text("hello")
    return d


@pytest.fixture
def app_paths(tmp_path) -> AppPaths:
    return AppPaths(tmp_path / "cfg", tmp_path / "data")


@pytest.fixture
def ws(app_paths):
    workspace = Workspace(app_paths)
    yield workspace
    workspace.shutdown()


@pytest.fixture
def project_ws(ws, tmp_path):
    """Workspace with a freshly created project."""
    ws.new_project("Demo", tmp_path / "projects")
    return ws


def settle(ws: Workspace) -> None:
    assert ws.jobs.wait_idle(20), "jobs did not finish"
    assert ws.autosave.wait_idle(10)


def import_and_wait(ws: Workspace, path: Path):
    """Import one file and return the resulting asset."""
    got = []
    jobs = ws.media.import_files([path], on_asset=got.append)
    assert ws.jobs.wait_idle(20)
    assert jobs[0].error is None, jobs[0].error
    assert got, "asset was not registered"
    return got[0]


@pytest.fixture
def voice_ws(project_ws, tmp_path):
    """Project with a real WAV imported as the voice-over and the narration set as the script."""
    from app.tests.helpers import NARRATION, ScriptedProvider, make_audio

    ws = project_ws
    provider = ScriptedProvider(NARRATION)
    audio = make_audio(tmp_path / "narration.wav", provider.words[-1].end + 1.0)
    ws.media.import_voice_over(audio)
    assert ws.jobs.wait_idle(30)
    assert ws.project.voice_over.asset_id
    ws.set_script(NARRATION)
    ws.provider = provider
    return ws


def pipeline_ws(ws, tmp_path, narration: str, threshold: float = 0.6, min_scene: float = 2.0):
    """Project -> voice-over -> transcript -> scenes for ``narration`` (test provider; real audio file)."""
    from app.analysis.segmenter import SegmentationParams
    from app.tests.helpers import ScriptedProvider, make_audio, run_scenes, run_transcription

    ws.new_project("Research", tmp_path / "projects")
    provider = ScriptedProvider(narration)
    ws.media.import_voice_over(make_audio(tmp_path / "vo.wav", provider.words[-1].end + 1.0))
    assert ws.jobs.wait_idle(30)
    ws.set_script(narration)
    run_transcription(ws, provider)
    run_scenes(ws, params=SegmentationParams(threshold=threshold, min_scene_seconds=min_scene))
    return ws


@pytest.fixture
def research_ws(ws, tmp_path):
    """A workspace with the standard 16-topic narration analysed into scenes (Phase 2 pipeline, for real)."""
    from app.tests.helpers import NARRATION

    return pipeline_ws(ws, tmp_path, NARRATION)


@pytest.fixture
def edit_ws(research_ws, tmp_path):
    """Phase 4 starting point: scenes + approved visual assignments backed by real project assets (2 videos, 4 stills)."""
    from app.research.models import Acquisition, VisualAssignment
    from app.project.phase3_commands import SceneDecisionCommand
    from app.tests.helpers import make_image, make_video

    ws = research_ws
    d = tmp_path / "edit_media"
    d.mkdir()
    files = [make_video(d / "v_long.mp4", 12.0), make_video(d / "v_short.mp4", 3.0, "testsrc2"),
             make_image(d / "wide.png", "testsrc", "1600x600"), make_image(d / "portrait.png", "testsrc2", "600x900"),
             make_image(d / "plain.png", "gradient", "1280x720"), make_image(d / "doc.png", "mandel", "800x1000")]
    ws.media.import_files(files)
    assert ws.jobs.wait_idle(60)
    p = ws.project
    by_name = {a.name: a for a in p.assets.all()}
    ws.assets_by = by_name
    pool = [by_name[n] for n in ("v_long.mp4", "wide.png", "portrait.png", "plain.png", "v_short.mp4", "doc.png")]
    for i, sc in enumerate(p.scenes):
        evid = p.visual_intents.get(sc.id) and p.visual_intents[sc.id].type.value == "EVIDENCE"
        asset = by_name["doc.png"] if evid else pool[i % (len(pool) - 1)]
        a = VisualAssignment(sc.id, None, asset.id, "USER", 90.0, True, False, acquisition=Acquisition.LOCAL, source_type=asset.source_type)
        ws.apply_command(SceneDecisionCommand(p, sc.id, "assign", assignment=a))
    return ws
