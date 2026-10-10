"""Phase 7 acceptance (spec section 55): import a reference -> analyse -> style plan -> customize -> apply -> generate the AI edit and the presentation ->
the project differs measurably from the no-reference baseline -> the reference content is nowhere in the project -> reopen -> remove the reference -> baseline again.

The reference is synthetic (``reference_helpers``): fast cuts, centred captions, voice + music + effects, so every expected direction of change is known.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.editing.effective import effective_editing_settings
from app.reference.originality import OriginalityGuard
from app.reference.project_metrics import project_style_profile
from app.reference.style_adapter import similarity_between
from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import make_audio, mux, overlay_video
from app.tests.test_reference_effective import captions, structure, visual
from app.timeline.clip import KIND_TEXT

pytestmark = needs_ffmpeg

SECONDS = 24.0
REFERENCE_TEXT = "Secret reference caption"


@pytest.fixture(scope="module")
def ref_video(tmp_path_factory) -> Path:
    """A fast competitor-style reference: a hard cut every 2 s, bold centred captions of 3-4 words, a headline, voice, ducked music and a few effects."""
    d = tmp_path_factory.mktemp("accref")
    caps = [{"start": 0.3 + 2.0 * i, "end": 1.9 + 2.0 * i, "text": f"{REFERENCE_TEXT} {i}", "pos": "center", "size": 0.09} for i in range(11)]
    silent = overlay_video(d / "silent.mp4", SECONDS, caps, shots=[2.0] * 12)
    audio = make_audio(d / "mix.wav", SECONDS, voice=[(0.4 + 3.0 * i, 2.6 + 3.0 * i) for i in range(8)], music=[(0.0, SECONDS, 0.14)], sfx=[2.0, 6.0, 10.0, 14.0, 18.0], duck_to=0.4)
    return mux(silent, audio, d / "reference.mp4")


def generate_all(ws) -> None:
    ws.editing.generate(force=True)
    assert ws.jobs.wait_idle(120)
    ws.presentation.generate()
    assert ws.jobs.wait_idle(180)


def measures(ws) -> dict:
    p = ws.project
    v, caps = visual(p), captions(p)
    words = [len(c.text["words"]) for c in caps]
    return {"shots": len(v), "avg_shot": sum(c.duration for c in v) / len(v), "captions": len(caps), "avg_words": sum(words) / len(words), "positions": {c.text["position"] for c in caps},
            "styles": {c.text["style_id"] for c in caps}, "moving": sum(1 for c in v if any(k.property == "scale" for k in c.keyframes))}


def test_the_complete_reference_workflow(pres_ws, ref_video, tmp_path):
    ws = pres_ws
    p = ws.project
    generate_all(ws)  # --- the baseline: the user's own settings, no reference
    base_structure, base = structure(p), measures(ws)
    base_profile = project_style_profile(p)
    assert base["positions"] == {"bottom"} and base["styles"] == {"professional"} and not p.reference_settings.enabled
    assets_before = [a.id for a in p.assets.all()]

    # --- import: the reference goes to references/<id>/, never into the media library or onto the timeline
    ref = ws.reference.import_reference(ref_video)
    folder = p.root / "references" / ref.reference_id
    assert (folder / "reference_video.mp4").is_file() and [a.id for a in p.assets.all()] == assets_before and structure(p) == base_structure

    # --- analyse (background job, cached) and review
    ws.reference.analyze()
    assert ws.jobs.wait_idle(240)
    prof = ws.reference.profile()
    assert prof is not None and prof.reference_id == ref.reference_id and prof.scores.pacing > 60 and prof.confidence["shot_detection"] > 0.5
    assert prof.shot_stats.average_shot_duration == pytest.approx(2.0, abs=0.5)
    assert structure(p) == base_structure and p.reference_style_overrides.is_empty  # analysing changes neither the timeline nor the settings
    rows, sim = ws.reference.comparison()
    assert len(rows) == 10 and sim is not None
    row = {r.key: r for r in rows}
    assert row["cuts_per_minute"].reference_value > row["cuts_per_minute"].project_value  # the reference cuts faster than this project

    # --- style plan: a preview with a strength, a customize target and the simulated outcome; still nothing changes
    plan = ws.reference.plan(strength=1.0, adjustments={"sfx_frequency": 10.0})
    ov = plan.overrides
    assert not ov.is_empty and ov.target_shot_duration < 4.8 and ov.caption_max_words < 12 and ov.caption_position == "center"
    assert plan.projected.similarity_after > plan.projected.similarity_before and plan.notice.startswith("This will update AI editing preferences")
    assert OriginalityGuard().audit_overrides(ov) == [] and OriginalityGuard().audit_profile(prof) == []
    assert structure(p) == base_structure and p.reference_style_overrides.is_empty

    # --- apply: one undoable step, checkpointed; no re-render, no timeline change
    ws.reference.apply_style(plan)
    assert p.reference_settings.enabled and p.reference_style_overrides.to_dict() == ov.to_dict() and len(p.style_application_history) == 1
    assert p.style_application_history[0].checkpoint and not p.render_history and structure(p) == base_structure
    assert ws.commands.can_undo and not ws.render.jobs()

    # --- the next AI edit and presentation use it: measurably different, still valid and editable
    generate_all(ws)
    styled = measures(ws)
    assert styled["shots"] > base["shots"] * 1.2 and styled["avg_shot"] < base["avg_shot"]
    assert styled["captions"] > base["captions"] and styled["avg_words"] < base["avg_words"] and styled["positions"] == {"center"}
    assert structure(p) != base_structure
    p.validate()
    mine = project_style_profile(p)
    assert similarity_between(mine, prof).pacing_match > similarity_between(base_profile, prof).pacing_match  # the project moved toward the reference
    assert p.caption_settings.max_words == 12 and p.caption_settings.position == "bottom" and p.editing_settings.motion_intensity == 0.5  # the user's own settings were not rewritten

    # --- nothing of the reference ended up in the user's video
    audit = OriginalityGuard().audit_project(p)
    assert audit.ok, audit.errors
    assert [a.id for a in p.assets.all()] == assets_before
    assert not any(a.path.startswith("references") for a in p.assets.all()) and all(c.asset_id in ("", *assets_before) for c in p.timeline.all_clips())
    blob = json.dumps([p.timeline.to_dict(), p.to_document()["editing_decisions"], p.to_document()["presentation_decisions"], p.to_document()["caption_segments"]], default=str)
    assert REFERENCE_TEXT not in blob and "Secret" not in blob  # the reference's words are not in captions, texts or decisions
    texts = [c.text.get("content", "") for c in p.timeline.all_clips() if c.kind == KIND_TEXT]
    narration = " ".join(s.narration.lower() for s in p.scenes)
    assert all(t.lower().strip(".,;:") in narration or all(w in narration for w in t.lower().split()) for t in texts if t)  # on-screen text comes from the user's narration

    # --- save, close, reopen: the whole reference setup is restored and the next generation gives the same result
    root = p.root
    sig = ws.reference.profile().signature()
    styled_structure = structure(p)
    ws.save()
    ws.close_project()
    ws.open_project(root)
    p = ws.project
    assert ws.reference.profile().signature() == sig and p.reference_settings.enabled and p.reference_style_overrides.to_dict() == ov.to_dict()
    generate_all(ws)
    assert structure(p) == styled_structure

    # --- a setting the user makes now beats the style (the next generation uses the user's motion, the rest of the style still applies)
    ws.editing.update_settings(motion_intensity=0.2)
    assert effective_editing_settings(p).motion_intensity == 0.2 and effective_editing_settings(p).reference.target_shot_duration is not None
    ws.editing.update_settings(motion_intensity=0.5)
    assert "motion_intensity" in p.editing_settings.user_set

    # --- undo of the application restores the previous strategy; removing the reference and the style returns the project to the baseline
    ws.reference.remove_reference(ref.reference_id)
    assert not folder.exists() and p.reference_assets == {} and p.reference_style_profile is None
    assert p.reference_settings.enabled  # the abstract preferences stay until the user removes them ...
    ws.reference.clear_style()
    assert not p.reference_settings.enabled and p.reference_style_overrides.is_empty
    generate_all(ws)
    assert structure(p) == base_structure and measures(ws) == base  # ... and then everything is exactly as it was without a reference
    p.validate()
    assert OriginalityGuard().audit_project(p).ok


def test_undoing_the_application_restores_the_previous_strategy(pres_ws, ref_video):
    ws = pres_ws
    p = ws.project
    ws.reference.import_reference(ref_video)
    ws.reference.analyze()
    assert ws.jobs.wait_idle(240)
    ws.reference.apply_style(strength=0.5)
    first = p.reference_style_overrides.to_dict()
    ws.reference.apply_style(strength=1.0)
    assert p.reference_style_overrides.to_dict() != first and len(p.style_application_history) == 2
    ws.commands.undo()
    assert p.reference_style_overrides.to_dict() == first and len(p.style_application_history) == 1
    ws.commands.undo()
    assert p.reference_style_overrides.is_empty and not p.reference_settings.enabled and p.style_application_history == []
    ws.commands.redo()
    assert p.reference_style_overrides.to_dict() == first
    ws.reference.apply_style(strength=1.0)
    assert ws.reference.revert_last_application() and p.reference_style_overrides.to_dict() == first


def test_applying_nothing_is_refused_with_a_reason(pres_ws, ref_video):
    from app.services.reference_service import ReferenceError

    ws = pres_ws
    ws.reference.import_reference(ref_video)
    ws.reference.analyze()
    assert ws.jobs.wait_idle(240)
    with pytest.raises(ReferenceError, match="nothing to apply"):
        ws.reference.apply_style(mode="CUSTOM", custom_dimensions=[])
    p = ws.project
    p.editing_settings.user_set = ["pacing", "motion_intensity", "transition_frequency", "text_emphasis"]
    p.caption_settings.user_set = ["style_id", "position", "max_words", "keyword_highlight"]
    p.audio_settings.user_set = ["music_level", "important_level", "max_sfx_per_minute", "pause_level"]
    with pytest.raises(ReferenceError, match="nothing to apply"):
        ws.reference.apply_style()  # the user set everything: the style contributes nothing
    ws.reference.apply_style(preserve_user_edits=False)  # unless the user allows it
    assert not p.reference_style_overrides.is_empty


def test_a_request_to_copy_becomes_an_abstract_instruction_in_the_project(pres_ws, ref_video):
    ws = pres_ws
    ws.reference.import_reference(ref_video)
    res = ws.reference.guard_request("Copy the competitor's exact opening and their script.")
    assert res.flagged and ws.project.reference_settings.style_request.startswith("Use a fast, high-information opening")
    assert "exact opening" not in ws.project.reference_settings.style_request.lower() and "Write your own text" in ws.project.reference_settings.style_request
