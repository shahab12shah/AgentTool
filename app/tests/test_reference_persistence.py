"""Phase 7 persistence: analyse -> customize -> apply -> save -> close -> reopen restores the reference profile, the adjustments, the overrides and the history;
projects saved before Phase 7 (or without the reference sections) still open; a reference whose files are gone does not break the project."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.project.project import Project
from app.services.reference_service import ReferenceError
from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import make_audio, mux, overlay_video

pytestmark = needs_ffmpeg

SECONDS = 16.0
PHASE7_KEYS = ("reference_settings", "reference_assets", "reference_analysis", "reference_style_profile", "reference_style_overrides", "style_application_history")


@pytest.fixture(scope="module")
def ref_video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("persref")
    caps = [{"start": 0.4 + 2.0 * i, "end": 2.0 + 2.0 * i, "text": f"Caption {i}", "pos": "center", "size": 0.08} for i in range(7)]
    silent = overlay_video(d / "silent.mp4", SECONDS, caps, shots=[2.0] * 8)
    audio = make_audio(d / "mix.wav", SECONDS, voice=[(0.4 + 3.0 * i, 2.6 + 3.0 * i) for i in range(5)], music=[(0.0, SECONDS, 0.12)], sfx=[3.0, 9.0], duck_to=0.5)
    return mux(silent, audio, d / "reference.mp4")


@pytest.fixture
def applied_ws(project_ws, ref_video):
    ws = project_ws
    asset = ws.reference.import_reference(ref_video)
    ws.reference.analyze()
    assert ws.jobs.wait_idle(180)
    assert ws.project.reference_assets[asset.reference_id].analysis_status in ("COMPLETED", "PARTIAL")
    ws.reference.update_settings(application_mode="CUSTOM", custom_dimensions=["pacing", "motion_intensity", "caption_density"], style_strength=0.5, adjustments={"motion_intensity": 35.0})
    ws.reference.apply_style(strength=0.5, mode="CUSTOM", custom_dimensions=["pacing", "motion_intensity", "caption_density"], adjustments={"motion_intensity": 35.0})
    ws.rid = asset.reference_id
    return ws


def test_everything_the_user_set_up_survives_save_close_reopen(applied_ws):
    ws = applied_ws
    p = ws.project
    root = p.root
    before = {
        "sig": ws.reference.profile().signature(), "settings": p.reference_settings.to_dict(), "overrides": p.reference_style_overrides.to_dict(),
        "history": [a.to_dict() for a in p.style_application_history], "asset": p.reference_assets[ws.rid].to_dict(), "record": json.dumps(p.reference_analysis, sort_keys=True),
    }
    assert before["settings"]["enabled"] and before["settings"]["adjustments"] == {"motion_intensity": 35.0} and before["settings"]["application_mode"] == "CUSTOM"
    assert len(before["history"]) == 1 and before["history"][0]["before"] is not None and before["history"][0]["after"] == before["overrides"]
    ws.save()
    ws.close_project()
    ws.open_project(root)
    q = ws.project
    assert ws.reference.profile().signature() == before["sig"]
    assert q.reference_settings.to_dict() == before["settings"] and q.reference_style_overrides.to_dict() == before["overrides"]
    assert [a.to_dict() for a in q.style_application_history] == before["history"]
    assert q.reference_assets[ws.rid].to_dict() == before["asset"] and json.dumps(q.reference_analysis, sort_keys=True) == before["record"]
    assert len(q.assets.all()) == 0 and q.timeline.all_clips() == []  # still no reference content in the production project
    # the stored analysis is still valid: no second analysis, and the same plan comes out
    done = []
    assert ws.reference.analyze(on_done=done.append) is None and done
    plan = ws.reference.plan()
    assert plan.overrides.to_dict() == before["overrides"] or plan.overrides.signature() == q.reference_style_overrides.signature()
    # the applied style is still reversible after a reopen
    assert ws.reference.revert_last_application()
    assert q.reference_style_overrides.is_empty
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert ws.project.reference_style_overrides.is_empty and len(ws.project.style_application_history) == 2


def test_clearing_the_style_is_remembered(applied_ws):
    ws = applied_ws
    root = ws.project.root
    ws.reference.clear_style()
    assert not ws.project.reference_settings.enabled and ws.project.reference_style_overrides.is_empty
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert not ws.project.reference_settings.enabled and ws.project.reference_style_overrides.is_empty
    assert ws.project.style_application_history[-1].mode == "CLEARED"
    assert ws.project.reference_style_profile is not None  # the analysis itself is kept


def test_a_project_saved_before_phase_7_opens_and_stays_usable(applied_ws, project_ws):
    ws = applied_ws
    doc = json.loads(json.dumps(ws.project.to_document()))
    old = dict(doc)
    old["schema_version"] = 6
    for k in PHASE7_KEYS:
        old.pop(k)
    for k in ("qc_settings", "qc_runs", "qc_issues", "qc_scores", "qc_ignored_issues", "qc_fixes", "qc_history", "qc_cache", "render_qc_results"):
        old.pop(k, None)
    q = Project.from_document(old)
    assert q.schema_version == 8 and q.reference_assets == {} and q.reference_style_profile is None and q.reference_style_overrides.is_empty
    assert not q.reference_settings.enabled and q.style_application_history == [] and q.reference_analysis == {}
    # the same through the file on disk
    root = ws.project.root
    ws.save()
    ws.close_project()
    f = root / "project.json"
    raw = json.loads(f.read_text(encoding="utf-8"))
    raw["schema_version"] = 6
    for k in PHASE7_KEYS:
        raw.pop(k, None)
    f.write_text(json.dumps(raw), encoding="utf-8")
    ws.open_project(root)
    assert ws.project.reference_assets == {} and ws.project.reference_style_overrides.is_empty
    # and a reference can be imported into it as usual
    with pytest.raises(ReferenceError, match="Import a reference"):
        ws.reference.analyze()


def test_missing_reference_files_do_not_break_the_project(applied_ws):
    ws = applied_ws
    root = ws.project.root
    ov = ws.project.reference_style_overrides.to_dict()
    ws.save()
    ws.close_project()
    shutil.rmtree(root / "references" / ws.rid)  # the user deleted the folder by hand
    ws.open_project(root)
    p = ws.project
    assert ws.rid in p.reference_assets and p.reference_style_overrides.to_dict() == ov and p.reference_style_profile is not None  # the abstract style outlives the files
    with pytest.raises(ReferenceError, match="is missing"):
        ws.reference.analyze()
    assert ws.reference.cached_analysis(ws.rid) is None
    assert ws.reference.plan().overrides is not None  # the stored profile is enough to plan again
    ws.reference.remove_reference(ws.rid)  # and the entry can still be removed
    assert ws.rid not in ws.project.reference_assets


def test_a_corrupt_analysis_cache_is_ignored_and_rebuilt(applied_ws):
    ws = applied_ws
    cache = ws.project.root / "references" / ws.rid / "analysis.json"
    ws.reference._cache.clear()
    cache.write_text("{ this is not json", encoding="utf-8")
    assert ws.reference.cached_analysis(ws.rid) is None
    job = ws.reference.analyze()
    assert job is not None and ws.jobs.wait_idle(180)
    assert ws.reference.cached_analysis(ws.rid) is not None and json.loads(cache.read_text(encoding="utf-8"))["reference_id"] == ws.rid
