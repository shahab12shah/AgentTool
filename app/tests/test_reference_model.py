"""The reference data model: scales, scores, labels, profile building (incl. failed detectors), similarity, comparison, serialisation, schema v7."""

from __future__ import annotations

import json

import pytest

from app.editing.overrides import EditingStrategyOverrides
from app.project.project import Project
from app.project.phase7_commands import (
    ApplyStyleCommand, ClearStyleCommand, RegisterReferenceCommand, RemoveReferenceCommand, StoreAnalysisCommand,
)
from app.reference.application import ReferenceAsset, ReferenceSettings, StyleApplication
from app.reference.style_model import (
    DIMENSIONS, PACING_POINTS, AudioProfile, DetectorStatus, ReferenceFeatures, ReferenceMetadata, ReferenceStyleProfile, Shot, ShotStats, StyleFeatures,
    build_profile, build_profile_from_features, compare, compute_scores, density_label, distribution_buckets, motion_class, pacing_class, scale, similarity, unscale,
)


def features(**kw) -> StyleFeatures:
    base = dict(duration=120.0, average_shot_duration=4.0, median_shot_duration=3.5, cuts_per_minute=15.0, major_changes_per_minute=3.0, overlay_events_per_minute=4.0,
                text_events_per_minute=3.0, graphic_events_per_minute=1.0, motion_events_per_minute=6.0, zoom_events_per_minute=3.0, average_motion_intensity=0.3,
                caption_coverage=0.5, captions_per_minute=10.0, average_words_per_caption=4.0, transition_events_per_minute=1.0, non_cut_share=0.1, voice_dominance=0.8,
                music_presence=0.6, music_ducking_strength=0.5, sfx_per_minute=3.0)
    base.update(kw)
    return StyleFeatures(**base)


def reference(cuts=15.0, **detector_failures) -> ReferenceFeatures:
    ref = ReferenceFeatures(metadata=ReferenceMetadata(duration=120.0, width=1920, height=1080, has_audio=True))
    ref.shot_stats = ShotStats(count=30, average_shot_duration=60 / cuts, median_shot_duration=60 / cuts, cuts_per_minute=cuts)
    ref.audio = AudioProfile(has_audio=True, voice_dominance=0.8, music_presence=0.5, music_behavior="Continuous", sfx_per_minute=2.0, sfx_class="Subtle")
    ref.confidence = {k: 0.9 for k in ("shot_detection", "motion_detection", "caption_detection", "text_detection", "transition_detection", "audio_detection", "structure_detection")}
    for k in ref.confidence:
        ref.detectors.append(DetectorStatus(k, ok=k not in detector_failures, confidence=0.0 if k in detector_failures else 0.9, message="x" if k in detector_failures else ""))
    return ref


# ------------------------------------------------------------------ scales and labels
def test_scale_is_monotonic_clamped_and_invertible():
    assert scale(-5, PACING_POINTS) == 0 and scale(1000, PACING_POINTS) == 100
    prev = -1.0
    for v in range(0, 60):
        s = scale(float(v), PACING_POINTS)
        assert s >= prev
        prev = s
    for score in (5, 25, 50, 80, 95):
        assert scale(unscale(score, PACING_POINTS), PACING_POINTS) == pytest.approx(score, abs=1e-6)


def test_labels_follow_the_spec_scales():
    assert [pacing_class(x) for x in (2, 8, 15, 30)] == ["Slow", "Moderate", "Fast", "Very Fast"]
    assert [density_label(x) for x in (10, 30, 50, 70, 90)] == ["Low", "Moderate-Low", "Medium", "High", "Very High"]
    assert [motion_class(x) for x in (5, 30, 50, 70, 90)] == ["Minimal", "Subtle", "Moderate", "Strong", "Aggressive"]
    assert distribution_buckets([0.5, 1.5, 3.0, 5.0, 10.0, 20.0]) == {"<1s": 1, "1-2s": 1, "2-4s": 1, "4-8s": 1, "8-15s": 1, ">15s": 1}


def test_scores_are_zero_to_hundred_and_more_cuts_means_faster_and_denser():
    slow, fast = compute_scores(features(cuts_per_minute=4.0)), compute_scores(features(cuts_per_minute=30.0))
    for s in (slow, fast):
        assert all(0 <= s.get(d) <= 100 for d in DIMENSIONS)
    assert fast.pacing > slow.pacing and fast.visual_density > slow.visual_density
    quiet = compute_scores(features(music_presence=0.0, sfx_per_minute=0.0, caption_coverage=0.0, captions_per_minute=0.0, text_events_per_minute=0.0))
    assert quiet.music_presence == 0 and quiet.sfx_frequency == 0 and quiet.caption_density == 0 and quiet.text_density == 0


# ------------------------------------------------------------------ profile
def test_profile_has_labels_confidence_and_a_readable_table():
    prof = build_profile(reference(), "ref_1")
    assert prof.reference_id == "ref_1" and prof.unavailable == [] and prof.confidence["overall"] == pytest.approx(0.9)
    rows = prof.rows()
    assert [r[0] for r in rows] == list(DIMENSIONS) and all(r[2] >= 0 for r in rows)
    assert prof.categories["pacing"] == "Fast"
    assert len(prof.summary_lines()) == 8


def test_a_failed_detector_makes_its_dimensions_unavailable_instead_of_guessing():
    prof = build_profile(reference(motion_detection=1, audio_detection=1), "r")
    assert set(prof.unavailable) == {"motion_intensity", "music_presence", "sfx_frequency"}
    assert prof.is_available("pacing") and not prof.is_available("motion_intensity")
    assert all(r[3] == "UNAVAILABLE" for r in prof.rows() if r[0] in prof.unavailable)
    assert prof.confidence["overall"] == pytest.approx(0.9)  # the failed detectors do not drag the confidence of what was measured
    lost = build_profile(reference(shot_detection=1), "r")
    assert "pacing" in lost.unavailable and "visual_density" in lost.unavailable  # density needs the cuts


def test_low_confidence_is_flagged_not_stated_as_fact():
    ref = reference()
    ref.confidence["caption_detection"] = 0.3
    prof = build_profile(ref, "r")
    assert any("low confidence" in w.lower() and "Captions" in w for w in prof.warnings)
    assert prof.dimension_confidence("caption_density") == pytest.approx(0.3)


def test_profile_round_trips_through_json_and_signature_is_stable():
    prof = build_profile(reference(), "r")
    again = ReferenceStyleProfile.from_dict(json.loads(json.dumps(prof.to_dict())))
    assert again.scores.as_dict() == prof.scores.as_dict() and again.signature() == prof.signature() and again.categories == prof.categories
    assert build_profile(reference(cuts=30.0), "r").signature() != prof.signature()


def test_profile_never_contains_per_shot_data():
    ref = reference()
    ref.shots = [Shot(f"s{i}", i * 4.0, i * 4.0 + 4.0) for i in range(5)]
    blob = json.dumps(build_profile(ref, "r").to_dict())
    assert "shot_id" not in blob and '"shots"' not in blob


# ------------------------------------------------------------------ similarity / comparison
def test_similarity_is_100_for_identical_scores_and_skips_unavailable_dimensions():
    a = compute_scores(features())
    same = similarity(a, a)
    assert same.overall == 100.0 and same.compared == list(DIMENSIONS)
    other = compute_scores(features(cuts_per_minute=2.0, music_presence=0.0))
    full, partial = similarity(a, other), similarity(a, other, skip=["pacing", "music_presence"])
    assert full.overall < 100 and partial.overall > full.overall and "pacing" not in partial.compared


def test_compare_rows_cover_every_dimension_on_the_same_scale():
    ref, mine = build_profile(reference(), "r"), build_profile_from_features(features(cuts_per_minute=5.0), "project")
    rows = compare(ref, mine)
    keys = [r.key for r in rows]
    assert {"average_shot_duration", "cuts_per_minute", *DIMENSIONS} <= set(keys)
    pacing = next(r for r in rows if r.key == "pacing")
    assert pacing.reference == "Fast" and pacing.project == "Slow" and pacing.reference_value > pacing.project_value


# ------------------------------------------------------------------ schema v7 + commands
def project_with_reference() -> Project:
    p = Project.new("x")
    p.reference_assets["r1"] = ReferenceAsset("r1", "clip", "references/r1/reference_video.mp4", content_hash="abc")
    p.reference_settings.active_reference_id = "r1"
    return p


def test_schema_v7_round_trips_and_older_documents_migrate():
    p = project_with_reference()
    p.reference_style_profile = build_profile(reference(), "r1")
    ov = EditingStrategyOverrides(target_shot_duration=3.0, applied_fields=["target_shot_duration"])
    p.reference_style_overrides = ov
    p.style_application_history.append(StyleApplication("a1", "r1", before=EditingStrategyOverrides(), after=ov))
    doc = json.loads(json.dumps(p.to_document()))
    assert doc["schema_version"] == 8 and {"reference_settings", "reference_assets", "reference_analysis", "reference_style_profile", "reference_style_overrides",
                                            "style_application_history"} <= set(doc)
    q = Project.from_document(doc)
    assert q.reference_assets["r1"].content_hash == "abc" and q.reference_style_profile.signature() == p.reference_style_profile.signature()
    assert q.reference_style_overrides.target_shot_duration == 3.0 and q.style_application_history[0].after.target_shot_duration == 3.0
    old = dict(doc)
    old["schema_version"] = 6
    for k in ("reference_settings", "reference_assets", "reference_analysis", "reference_style_profile", "reference_style_overrides", "style_application_history"):
        old.pop(k)
    m = Project.from_document(old)
    assert m.schema_version == 8 and m.reference_assets == {} and m.reference_style_profile is None and m.reference_style_overrides.is_empty and not m.reference_settings.enabled


def test_a_reference_is_never_a_project_asset():
    p = project_with_reference()
    assert len(p.assets.all()) == 0 and p.timeline.get_clip("r1") is None
    assert "references/r1" not in json.dumps(p.to_document()["assets"])


def test_apply_and_clear_commands_are_undoable_and_record_history():
    p = project_with_reference()
    s = ReferenceSettings(enabled=True, active_reference_id="r1", application_mode="BALANCED", style_strength=0.5)
    ov = EditingStrategyOverrides(target_shot_duration=3.0, motion_intensity=0.4, applied_fields=["target_shot_duration", "motion_intensity"])
    cmd = ApplyStyleCommand(p, s, ov, StyleApplication("a1", "r1"))
    cmd.do()
    assert p.reference_settings.enabled and p.reference_style_overrides.target_shot_duration == 3.0 and len(p.style_application_history) == 1
    assert p.style_application_history[0].before.is_empty and p.style_application_history[0].after.motion_intensity == 0.4
    clear = ClearStyleCommand(p, StyleApplication("a2", "r1", mode="CLEARED"))
    clear.do()
    assert not p.reference_settings.enabled and p.reference_style_overrides.is_empty and len(p.style_application_history) == 2
    clear.undo()
    assert p.reference_settings.enabled and p.reference_style_overrides.target_shot_duration == 3.0 and len(p.style_application_history) == 1
    cmd.undo()
    assert not p.reference_settings.enabled and p.reference_style_overrides.is_empty and p.style_application_history == []


def test_register_remove_and_store_analysis_commands_round_trip():
    p = Project.new("x")
    asset = ReferenceAsset("r9", "clip", "references/r9/reference_video.mp4")
    reg = RegisterReferenceCommand(p, asset)
    reg.do()
    assert "r9" in p.reference_assets and p.reference_settings.active_reference_id == "r9" and len(p.assets.all()) == 0
    prof = build_profile(reference(), "r9")
    store = StoreAnalysisCommand(p, "r9", {"analysis_status": "COMPLETED", "analyzed_hash": "h"}, {"status": "COMPLETED"}, prof)
    store.do()
    assert p.reference_assets["r9"].analysis_status == "COMPLETED" and p.reference_analysis["r9"]["status"] == "COMPLETED" and p.reference_style_profile is not None
    store.undo()
    assert p.reference_assets["r9"].analysis_status == "NONE" and "r9" not in p.reference_analysis and p.reference_style_profile is None
    store.do()
    rem = RemoveReferenceCommand(p, "r9")
    rem.do()
    assert p.reference_assets == {} and p.reference_style_profile is None and p.reference_settings.active_reference_id == ""
    rem.undo()
    assert "r9" in p.reference_assets and p.reference_style_profile is not None and p.reference_settings.active_reference_id == "r9"


def test_reference_settings_modes_select_dimensions():
    s = ReferenceSettings()
    assert s.dimensions() == list(DIMENSIONS)
    s.application_mode = "BALANCED"
    assert s.dimensions() == ["pacing", "visual_density", "motion_intensity"]
    s.application_mode, s.custom_dimensions = "CUSTOM", ["caption_density", "sfx_frequency", "bogus"]
    assert s.dimensions() == ["caption_density", "sfx_frequency"]
