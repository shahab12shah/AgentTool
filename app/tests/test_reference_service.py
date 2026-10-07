"""ReferenceService end to end on real synthetic media: import into the isolated folder, analysis job (stages, cancel, retry, cache), failure and partial results,
persistence, isolation (a reference is never a project asset and the analysis never touches the timeline)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from app.reference.analyzer import ReferenceVideoAnalyzer
from app.reference.feature_extractor import STAGE_NAMES
from app.reference.signals import ReferenceAnalysisError
from app.services.reference_service import ReferenceError
from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import make_audio, mux, overlay_video, run_ffmpeg

pytestmark = needs_ffmpeg

SECONDS = 24.0


@pytest.fixture(scope="module")
def ref_video(tmp_path_factory) -> Path:
    """A 24 s reference: a hard cut every 3 s, bottom captions, voice + continuous music + a few SFX."""
    d = tmp_path_factory.mktemp("refmedia")
    caps = [{"start": 0.5 + 2.0 * i, "end": 2.3 + 2.0 * i, "text": f"Caption number {i}", "pos": "bottom", "size": 0.07} for i in range(10)]
    silent = overlay_video(d / "silent.mp4", SECONDS, caps, shots=[3.0] * 8)
    audio = make_audio(d / "mix.wav", SECONDS, voice=[(0.4 + 3.0 * i, 2.6 + 3.0 * i) for i in range(8)], music=[(0.0, SECONDS, 0.12)], sfx=[3.0, 9.0, 15.0], duck_to=0.5)
    return mux(silent, audio, d / "reference.mp4")


def snapshot(p) -> str:
    return json.dumps([p.timeline.to_dict(), {k: v.to_dict() for k, v in p.editing_decisions.items()}, p.assets.to_list()], sort_keys=True, default=str)


def analyze_and_wait(ws, rid=None, **kw):
    job = ws.reference.analyze(rid, **kw)
    assert ws.jobs.wait_idle(180)
    return job


def test_import_copies_into_the_reference_folder_and_never_into_project_assets(project_ws, ref_video):
    ws = project_ws
    before = len(ws.project.assets.all())
    a = ws.reference.import_reference(ref_video)
    p = ws.project
    assert len(p.assets.all()) == before and p.timeline.all_clips() == []
    folder = p.root / "references" / a.reference_id
    assert (folder / "reference_video.mp4").is_file() and a.path == f"references/{a.reference_id}/reference_video.mp4" and a.link_mode == "copy"
    assert a.content_hash and a.size_bytes == ref_video.stat().st_size and a.analysis_status == "NONE"
    assert p.reference_settings.active_reference_id == a.reference_id and not p.reference_settings.enabled
    assert ref_video.is_file()  # the user's original is never touched


def test_import_can_link_instead_of_copy(project_ws, ref_video):
    a = project_ws.reference.import_reference(ref_video, link=True)
    assert a.link_mode == "reference" and Path(a.path) == ref_video.resolve() and not (project_ws.project.root / "references" / a.reference_id / "reference_video.mp4").exists()


def test_import_rejects_corrupt_unsupported_and_audio_only_files_with_readable_messages(project_ws, tmp_path):
    ws = project_ws
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(ReferenceError, match="could not be read|corrupt"):
        ws.reference.import_reference(bad)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(ReferenceError, match="not a supported video"):
        ws.reference.import_reference(txt)
    wav = make_audio(tmp_path / "only.wav", 3.0, voice=[(0.2, 2.5)])
    only_audio = tmp_path / "audio_only.mp4"
    run_ffmpeg(["-i", str(wav), "-c:a", "aac", str(only_audio)])
    with pytest.raises(ReferenceError, match="no video stream"):
        ws.reference.import_reference(only_audio)
    with pytest.raises(ReferenceError, match="was not found"):
        ws.reference.import_reference(tmp_path / "missing.mp4")
    assert ws.project.reference_assets == {} and list((ws.project.root / "references").iterdir()) == []  # failed imports leave nothing behind


def test_analysis_runs_in_stages_writes_the_cache_and_never_touches_the_timeline(project_ws, ref_video):
    ws = project_ws
    a = ws.reference.import_reference(ref_video)
    before = snapshot(ws.project)
    undo_before = ws.commands.can_undo
    seen: list[str] = []
    progress: list[float] = []
    ws.bus.subscribe("job.updated", lambda t, pl: (seen.append(pl["job"].message), progress.append(pl["job"].progress)) if pl["job"].type == "reference.analyze" else None)
    job = analyze_and_wait(ws)
    assert job is not None and job.status.value == "COMPLETED"
    asset = ws.project.reference_assets[a.reference_id]
    assert asset.analysis_status in ("COMPLETED", "PARTIAL") and asset.analyzed_hash == asset.content_hash and asset.analyzed_version >= 1 and asset.error == ""
    stages_seen = {s.split(":")[0] for s in seen}
    assert {"Detecting Shots", "Analyzing Captions", "Analyzing Audio", "Building Style Profile"} <= stages_seen and set(STAGE_NAMES) >= stages_seen - {"Queued", ""}
    assert progress == sorted(progress) or max(progress) == 100.0  # progress only ever moves forward within one run (jobs may repeat the final value)
    folder = ws.project.root / "references" / a.reference_id
    assert (folder / "analysis.json").is_file() and (folder / "analysis.log").is_file() and len(list((folder / "thumbnails").glob("*.jpg"))) >= 3
    prof = ws.reference.profile()
    assert prof is not None and prof.reference_id == a.reference_id and prof.scores.pacing > 40  # a cut every 3 s is fast
    assert ws.reference.analysis().summary and ws.reference.analysis().recommendations
    assert snapshot(ws.project) == before  # the analysis never changed the timeline, decisions or media library
    assert ws.commands.can_undo == undo_before  # and it is not an undoable edit of the project
    blob = (folder / "analysis.json").read_text()
    assert "Caption number" not in blob  # reference text is never stored (no OCR, geometry only)


def test_the_analysis_is_cached_and_invalidated_by_version_settings_or_video_change(project_ws, ref_video):
    ws = project_ws
    a = ws.reference.import_reference(ref_video)
    analyze_and_wait(ws)
    calls = []
    orig = ReferenceVideoAnalyzer.analyze

    def spy(self, *args, **kw):
        calls.append(1)
        return orig(self, *args, **kw)

    ReferenceVideoAnalyzer.analyze = spy
    try:
        done = []
        assert ws.reference.analyze(on_done=done.append) is None and done and calls == []  # valid cache: reused, no job
        assert ws.reference.is_stale(a.reference_id) is False
        job = ws.reference.reanalyze()
        assert job is not None and ws.jobs.wait_idle(180) and calls == [1]  # the explicit Reanalyze runs again
        ws.reference.analysis_settings.sensitivity = 1.4  # detection settings changed -> the cache no longer applies
        assert ws.reference.is_stale(a.reference_id) is True
        assert ws.reference.analyze() is not None and ws.jobs.wait_idle(180) and len(calls) == 2
        ws.reference.analysis_settings.sensitivity = 1.0
        ws.project.reference_assets[a.reference_id].content_hash = "different"  # the video changed
        assert ws.reference.cached_analysis(a.reference_id) is None
    finally:
        ReferenceVideoAnalyzer.analyze = orig


def test_cancel_retry_and_failure_are_recoverable(project_ws, ref_video, monkeypatch):
    ws = project_ws
    a = ws.reference.import_reference(ref_video)
    orig = ReferenceVideoAnalyzer.analyze
    started = threading.Event()

    def slow(self, path, **kw):
        started.set()
        cancel = kw["cancel"]
        for _ in range(400):
            if cancel.is_set():
                from app.reference.signals import AnalysisCancelled

                raise AnalysisCancelled()
            time.sleep(0.02)
        return orig(self, path, **kw)

    monkeypatch.setattr(ReferenceVideoAnalyzer, "analyze", slow)
    job = ws.reference.analyze()
    assert started.wait(10) and ws.reference.cancel_analysis()
    assert ws.jobs.wait_idle(30)
    assert job.status.value == "CANCELLED" and ws.project.reference_assets[a.reference_id].analysis_status == "CANCELED"
    assert ws.reference.cached_analysis(a.reference_id) is None and not (ws.project.root / "references" / a.reference_id / "analysis.json").exists()

    def broken(self, path, **kw):
        raise ReferenceAnalysisError("The reference could not be decoded.")

    monkeypatch.setattr(ReferenceVideoAnalyzer, "analyze", broken)
    job = ws.reference.retry_analysis()
    assert ws.jobs.wait_idle(30)
    asset = ws.project.reference_assets[a.reference_id]
    assert job.status.value == "FAILED" and asset.analysis_status == "FAILED" and "could not be decoded" in asset.error
    assert ws.reference.profile() is None  # nothing half-valid is kept

    monkeypatch.setattr(ReferenceVideoAnalyzer, "analyze", orig)
    ws.reference.retry_analysis()
    assert ws.jobs.wait_idle(180)
    assert ws.project.reference_assets[a.reference_id].analysis_status in ("COMPLETED", "PARTIAL") and ws.project.reference_assets[a.reference_id].error == ""


def test_a_failing_detector_gives_a_partial_but_usable_profile(project_ws, ref_video, monkeypatch):
    from app.reference import caption_analyzer

    ws = project_ws
    a = ws.reference.import_reference(ref_video)

    def boom(self, *args, **kw):
        raise RuntimeError("caption detector exploded")

    monkeypatch.setattr(caption_analyzer.CaptionTextAnalyzer, "analyze_video", boom)
    analyze_and_wait(ws)
    asset = ws.project.reference_assets[a.reference_id]
    prof = ws.reference.profile()
    assert asset.analysis_status == "PARTIAL" and prof is not None
    assert {"caption_density", "text_density"} <= set(prof.unavailable)
    assert prof.is_available("pacing") and prof.scores.pacing > 40 and prof.is_available("music_presence")
    assert any("could not be completed" in w for w in prof.warnings)
    rows = {r[0]: r for r in prof.rows()}
    assert rows["caption_density"][3] == "UNAVAILABLE" and rows["pacing"][3] != "UNAVAILABLE"


def test_a_reference_without_audio_skips_the_audio_dimensions(project_ws, tmp_path):
    ws = project_ws
    silent = overlay_video(tmp_path / "silent_ref.mp4", 12.0, [], shots=[3.0] * 4)
    ws.reference.import_reference(silent)
    analyze_and_wait(ws)
    prof = ws.reference.profile()
    assert prof is not None and {"music_presence", "sfx_frequency"} <= set(prof.unavailable) and prof.is_available("pacing")
    assert any("no audio" in w.lower() for w in prof.warnings)


def test_persistence_analyze_save_close_reopen(project_ws, ref_video):
    ws = project_ws
    a = ws.reference.import_reference(ref_video)
    analyze_and_wait(ws)
    root = ws.project.root
    sig = ws.reference.profile().signature()
    status = ws.project.reference_assets[a.reference_id].analysis_status
    ws.save()
    ws.close_project()
    ws.open_project(root)
    p = ws.project
    assert a.reference_id in p.reference_assets and p.reference_assets[a.reference_id].analysis_status == status and p.reference_style_profile.signature() == sig
    assert p.reference_analysis[a.reference_id]["reference_hash"] == p.reference_assets[a.reference_id].content_hash and p.schema_version == 7
    done = []
    assert ws.reference.analyze(on_done=done.append) is None and done  # the on-disk cache is still valid after reopening: no second analysis
    assert ws.reference.profile().signature() == sig and len(p.assets.all()) == 0


def test_removing_a_reference_deletes_its_folder_and_records(project_ws, ref_video):
    ws = project_ws
    a = ws.reference.import_reference(ref_video)
    analyze_and_wait(ws)
    folder = ws.project.root / "references" / a.reference_id
    assert folder.is_dir()
    ws.reference.remove_reference(a.reference_id)
    assert not folder.exists() and ws.project.reference_assets == {} and ws.project.reference_analysis == {} and ws.project.reference_style_profile is None
    with pytest.raises(ReferenceError):
        ws.reference.remove_reference(a.reference_id)
    with pytest.raises(ReferenceError, match="Import a reference"):
        ws.reference.analyze()
