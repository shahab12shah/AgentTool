"""Phase 7 style adapter: reference profile + user adjustments + project content -> editing strategy overrides.

The spec's rules under test: the reference style never outranks the user's own decisions or the content (narration, readability); the strength dial scales
the movement toward the reference; customize targets win over the reference value; unavailable / uncertain readings are not applied; the overrides are abstract.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.editing.overrides import PARAMETERS, EditingStrategyOverrides
from app.reference.application import AdaptationBaseline, ProjectContent, ReferenceSettings, StyleAdjustments
from app.reference.originality import OriginalityGuard
from app.reference.style_adapter import (
    ReferenceStyleAdapter,
    content_limits,
    predicted_scores,
    similarity_between,
    simulate,
)
from app.reference.style_model import (
    DIMENSIONS,
    AudioProfile,
    CaptionStats,
    HookProfile,
    HookWindow,
    ReferenceStyleProfile,
    StyleFeatures,
    build_profile_from_features,
)


# ---------------------------------------------------------------------------------------------- builders
def reference(**kw) -> ReferenceStyleProfile:
    """A fast, busy, captioned reference with a fast opening and a ducked music bed."""
    f = dict(duration=180.0, average_shot_duration=2.0, median_shot_duration=1.8, cuts_per_minute=30.0, major_changes_per_minute=8.0, overlay_events_per_minute=7.0, text_events_per_minute=6.0,
             graphic_events_per_minute=1.0, motion_events_per_minute=12.0, zoom_events_per_minute=6.0, average_motion_intensity=0.5, caption_coverage=0.6, captions_per_minute=20.0,
             average_words_per_caption=3.5, caption_emphasis_rate=0.5, transition_events_per_minute=1.0, non_cut_share=0.1, voice_dominance=0.7, music_presence=0.8,
             music_ducking_strength=0.6, music_dynamics=0.1, sfx_per_minute=5.0, silence_percentage=4.0, average_pause_duration=0.6, long_pause_frequency=2.0, audio_dynamic_range=12.0)
    f.update(kw)
    p = build_profile_from_features(StyleFeatures(**f), "ref_1")
    p.caption_style = CaptionStats(True, 0.6, 20.0, 3.5, 14.0, 1.2, "center", 0.5, 0.4, 0.08, "Bold", ["large_text"], False)
    p.hook = HookProfile([HookWindow(5, 40, 3, 0.5, 0.8, 40, 1, 2.0), HookWindow(10, 30, 2, 0.4, 0.7, 30, 1, 1.5), HookWindow(15, 20, 1, 0.3, 0.6, 20, 0, 1.0)], ["Fast visual switching"], 0.6)
    p.audio = AudioProfile(True, 0.7, 0.8, 0.6, "Continuous", 0.1, 0, 5.0, "Moderate", 0.4, 0, 0, 0, 12, 4.0, 0.6, 2.0, 12)
    p.confidence = {**p.confidence, "structure_detection": 0.6}
    return p


def content(**kw) -> ProjectContent:
    c = dict(scene_count=12, duration=120.0, median_scene_seconds=10.0, evidence_scene_share=0.1, average_information_density=0.3, median_words_per_second=2.6, number_scene_share=0.3,
             still_image_share=0.5, median_sentence_seconds=4.0)
    c.update(kw)
    return ProjectContent(**c)


def adapt(prof=None, adj=None, cfg=None, cont=None, base=None):
    return ReferenceStyleAdapter().adapt(prof or reference(), StyleAdjustments(adj or {}), cfg or ReferenceSettings(), cont or content(), base or AdaptationBaseline())


def settings(**kw) -> ReferenceSettings:
    return ReferenceSettings(**kw)


# ---------------------------------------------------------------------------------------------- the translation
def test_a_fast_busy_reference_becomes_abstract_strategy_parameters():
    r = adapt()
    ov = r.overrides
    assert isinstance(ov, EditingStrategyOverrides) and not ov.is_empty and ov.source == "reference" and ov.strength == 1.0
    base = AdaptationBaseline()
    assert ov.target_shot_duration < base.base_shot_duration and ov.max_shot_duration < base.max_shot_duration  # faster than the 4.8 s preset
    assert ov.motion_intensity is not None and 0.5 < ov.motion_intensity <= 1.0
    assert ov.hook_seconds in (5.0, 10.0, 15.0) and 0.45 <= ov.hook_shot_factor < 0.95  # a faster opening was measured
    assert ov.caption_max_words < base.caption_max_words and ov.caption_style == "bold" and ov.caption_position == "center" and ov.keyword_emphasis_rate > 0.5
    assert ov.music_level > base.music_level and ov.ducking_strength == pytest.approx(0.6, abs=0.02) and ov.sfx_per_minute > base.sfx_per_minute
    assert set(ov.applied_fields) == {k for k in PARAMETERS if getattr(ov, k) is not None}
    assert r.effective_targets and set(r.effective_targets) <= set(DIMENSIONS) and all(0 <= v <= 100 for v in r.effective_targets.values())
    assert r.notes and not r.warnings
    assert OriginalityGuard().audit_overrides(ov) == []  # nothing but numbers and style ids


def test_the_result_is_deterministic_and_does_not_compound():
    a, b = adapt(), adapt()
    assert a.overrides.to_dict() == b.overrides.to_dict() and a.effective_targets == b.effective_targets
    assert a.overrides.signature() == b.overrides.signature()
    # the baseline is the user's own settings, never the already adapted ones: the same input gives the same output however often it is applied
    assert adapt(base=AdaptationBaseline()).overrides.to_dict() == a.overrides.to_dict()


def test_a_slow_calm_reference_lengthens_shots_and_calms_motion():
    prof = reference(cuts_per_minute=5.0, average_shot_duration=11.0, median_shot_duration=10.0, motion_events_per_minute=1.0, average_motion_intensity=0.05, sfx_per_minute=0.0,
                     text_events_per_minute=0.5, music_presence=0.2)
    prof.hook = HookProfile([HookWindow(5, 5, 0, 0.05, 0.4, 5, 0, 1.0)], [], 0.0)
    ov = adapt(prof).overrides
    base = AdaptationBaseline()
    assert ov.target_shot_duration > base.base_shot_duration and ov.motion_intensity < base.motion_intensity and ov.sfx_per_minute < base.sfx_per_minute and ov.text_density < 0.5
    assert ov.hook_seconds is None and ov.hook_shot_factor is None  # no faster opening was measured: none is invented
    assert ov.max_shot_duration > base.max_shot_duration  # the ceiling rises so a long shot is not cut up again


# ---------------------------------------------------------------------------------------------- strength
def test_strength_scales_how_far_the_project_moves_toward_the_reference():
    levels = {s: adapt(cfg=settings(style_strength=s)).overrides for s in (0.25, 0.5, 0.75, 1.0)}
    base = AdaptationBaseline()
    shots = [levels[s].target_shot_duration for s in (0.25, 0.5, 0.75, 1.0)]
    assert shots == sorted(shots, reverse=True) and shots[0] < base.base_shot_duration and shots[-1] < shots[0]  # monotone toward the faster reference
    motion = [levels[s].motion_intensity for s in (0.25, 0.5, 0.75, 1.0)]
    assert motion == sorted(motion) and motion[0] > base.motion_intensity
    words = [levels[s].caption_max_words for s in (0.25, 0.5, 0.75, 1.0)]
    assert words == sorted(words, reverse=True) and words[0] <= base.caption_max_words
    assert [levels[s].strength for s in levels] == [0.25, 0.5, 0.75, 1.0]
    # 25 % is a subtle influence: well under half of the way (geometric) for the shot length
    full, quarter = levels[1.0].target_shot_duration, levels[0.25].target_shot_duration
    assert (base.base_shot_duration - quarter) < 0.5 * (base.base_shot_duration - full) + 0.3
    assert "caption_style" not in levels[0.25].active() and levels[1.0].caption_style == "bold"  # a look is only taken over at 50 % or more


def test_zero_strength_changes_nothing_that_matters():
    ov = adapt(cfg=settings(style_strength=0.0)).overrides
    base = AdaptationBaseline()
    assert ov.target_shot_duration == pytest.approx(base.base_shot_duration, abs=0.06)
    assert ov.motion_intensity == pytest.approx(base.motion_intensity, abs=0.01)


# ---------------------------------------------------------------------------------------------- user adjustments (Customize sliders)
def test_a_customize_target_replaces_the_reference_value_and_is_used_as_set():
    follow = adapt(cfg=settings(style_strength=0.25)).overrides
    slider = adapt(adj={"pacing": 20.0, "motion_intensity": 90.0}, cfg=settings(style_strength=0.25))
    ov = slider.overrides
    assert ov.target_shot_duration > follow.target_shot_duration * 2  # the user asked for slow pacing: the fast reference does not win
    assert ov.motion_intensity == pytest.approx(0.90, abs=0.02)  # the typed target is used as set (strength scales only the reference's own values)
    assert slider.effective_targets["motion_intensity"] == pytest.approx(90.0, abs=2.0)
    assert any("your target" in n.lower() for n in slider.notes)
    assert ov.hook_seconds is None  # with the user's own pacing the reference's opening is not copied on top


def test_a_user_target_never_breaks_the_content_limits():
    r = adapt(adj={"pacing": 100.0}, cont=content(median_sentence_seconds=12.0))
    c = content(median_sentence_seconds=12.0)
    lim = content_limits(c)
    assert r.overrides.target_shot_duration * lim.shot_factor >= lim.shot_floor - 0.1  # even a user target of "as fast as possible" respects the narration


# ---------------------------------------------------------------------------------------------- user lock / user settings win
def test_user_settings_win_over_the_reference():
    base = AdaptationBaseline(user_set={"editing": ["motion_intensity", "pacing"], "caption": ["style_id"], "audio": ["music_level"]})
    r = adapt(base=base)
    ov = r.overrides
    assert ov.motion_intensity is None and ov.target_shot_duration is None and ov.min_shot_duration is None and ov.max_shot_duration is None
    assert ov.hook_seconds is None and ov.hook_shot_factor is None and ov.caption_style is None and ov.music_level is None
    assert ov.transition_frequency is not None and ov.caption_position == "center" and ov.sfx_per_minute is not None  # untouched settings are still adapted
    assert "pacing" in r.skipped and "motion_intensity" in r.skipped and "yourself" in r.skipped["pacing"]
    assert any("Kept your own settings" in n for n in r.notes)


def test_without_keep_my_settings_the_style_may_override_them():
    base = AdaptationBaseline(user_set={"editing": ["motion_intensity", "pacing"]})
    r = adapt(base=base, cfg=settings(preserve_user_edits=False))
    assert r.overrides.motion_intensity is not None and r.overrides.target_shot_duration is not None and "pacing" not in r.skipped


def test_every_parameter_is_protected_by_its_own_setting():
    from app.editing.overrides import PROTECTED_BY

    fields = {"editing": {"pacing", "motion_intensity", "transition_frequency", "text_emphasis"}, "caption": {"style_id", "position", "max_words", "keyword_highlight"},
              "audio": {"music_level", "important_level", "max_sfx_per_minute", "pause_level"}}
    assert all(name in fields[kind] for kind, name in PROTECTED_BY.values())  # the table only names settings that exist
    everything = AdaptationBaseline(user_set={k: sorted(v) for k, v in fields.items()})
    assert adapt(base=everything).overrides.is_empty  # the user set everything: the style contributes nothing


# ---------------------------------------------------------------------------------------------- content wins
def test_shots_are_never_shorter_than_the_narration_and_reading_time_allow():
    fast = reference(cuts_per_minute=40.0, average_shot_duration=1.2, median_shot_duration=1.0)
    plain = adapt(fast, cont=content(median_sentence_seconds=0.0, evidence_scene_share=0.0))
    slow_speech = adapt(fast, cont=content(median_sentence_seconds=10.0, evidence_scene_share=0.0))
    documents = adapt(fast, cont=content(median_sentence_seconds=4.0, evidence_scene_share=0.8))
    lim = {"plain": content_limits(content(median_sentence_seconds=0.0, evidence_scene_share=0.0)), "slow": content_limits(content(median_sentence_seconds=10.0, evidence_scene_share=0.0)),
           "docs": content_limits(content(median_sentence_seconds=4.0, evidence_scene_share=0.8))}
    final = {k: r.overrides.target_shot_duration * lim[k].shot_factor for k, r in (("plain", plain), ("slow", slow_speech), ("docs", documents))}
    assert final["plain"] >= 1.4 - 0.1  # the readable minimum
    assert final["slow"] >= 0.45 * 10.0 - 0.1 and final["slow"] > final["plain"]  # long sentences: no cut every second
    assert final["docs"] >= 3.0 and final["docs"] > final["plain"]  # documents and data need reading time
    assert any("limited" in n for n in slow_speech.notes) and any("limited" in n for n in documents.notes)


def test_hook_shots_and_minimum_shot_stay_watchable():
    ov = adapt(reference(cuts_per_minute=60.0, average_shot_duration=1.0, median_shot_duration=1.0)).overrides
    assert ov.min_shot_duration is None or ov.min_shot_duration >= 1.0
    lim = content_limits(content())
    assert ov.target_shot_duration * lim.shot_factor * (ov.hook_shot_factor or 1.0) >= 1.0


def test_motion_is_limited_on_document_heavy_content():
    prof = reference(motion_events_per_minute=25.0, average_motion_intensity=1.0)
    free = adapt(prof, cont=content(evidence_scene_share=0.0)).overrides.motion_intensity
    docs = adapt(prof, cont=content(evidence_scene_share=0.8))
    assert docs.overrides.motion_intensity < free and docs.overrides.motion_intensity <= 0.55 and any("Motion: limited" in n for n in docs.notes)


def test_captions_never_get_shorter_than_a_readable_duration():
    prof = reference()
    prof.caption_style = CaptionStats(True, 0.6, 40.0, 1.5, 8.0, 1.0, "center", 0.5, 0.4, 0.08, "Bold", [], False)  # one-and-a-half-word captions
    fast_speech = adapt(prof, cont=content(median_words_per_second=3.4))
    ov = fast_speech.overrides
    assert ov.caption_max_words >= 5  # at least ceil(1.3 x words per second): each caption stays on screen about a second
    assert any("at least" in n and "words per caption" in n for n in fast_speech.notes)
    slow_speech = adapt(prof, cont=content(median_words_per_second=1.8)).overrides
    assert slow_speech.caption_max_words < ov.caption_max_words


def test_text_density_only_follows_the_amount_never_the_words():
    r = adapt(reference(text_events_per_minute=12.0))
    assert r.overrides.text_density > 0.9
    assert any("every word still comes from your narration" in n for n in r.notes)
    off = adapt(base=AdaptationBaseline(text_density=0.0))
    assert off.overrides.text_density is None and "text graphics are switched off" in off.skipped["text_density"]


# ---------------------------------------------------------------------------------------------- modes
def test_balanced_mode_applies_only_the_major_characteristics():
    r = adapt(cfg=settings(application_mode="BALANCED"))
    ov = r.overrides
    assert ov.target_shot_duration is not None and ov.motion_intensity is not None
    assert ov.caption_max_words is None and ov.caption_style is None and ov.music_level is None and ov.sfx_per_minute is None and ov.keyword_emphasis_rate is None
    for d in ("caption_density", "music_presence", "sfx_frequency"):
        assert "Balanced" in r.skipped[d]


def test_custom_mode_applies_the_chosen_characteristics_only():
    r = adapt(cfg=settings(application_mode="CUSTOM", custom_dimensions=["pacing", "motion_intensity", "caption_density"]))
    ov = r.overrides
    assert ov.target_shot_duration is not None and ov.motion_intensity is not None and ov.caption_max_words is not None
    assert ov.music_level is None and ov.sfx_per_minute is None and ov.transition_frequency is None and ov.text_density is None
    assert all("Custom" in r.skipped[d] for d in ("music_presence", "sfx_frequency", "transition_frequency", "text_density"))


def test_visual_density_moves_the_dimensions_that_are_not_applied_otherwise():
    prof = reference(text_events_per_minute=12.0, overlay_events_per_minute=14.0, non_cut_share=0.5, transition_events_per_minute=5.0)
    r = adapt(prof, cfg=settings(application_mode="BALANCED"))
    assert r.overrides.text_density is not None or r.overrides.transition_frequency is not None  # the density target is reached through them
    assert any("Visual density" in n for n in r.notes)
    full = adapt(prof)
    assert full.overrides.text_density is not None and "visual_density" in full.effective_targets  # in Full mode density is simply the sum of the others


# ---------------------------------------------------------------------------------------------- confidence and availability
def test_unavailable_dimensions_are_skipped_not_guessed():
    prof = reference()
    prof.unavailable = ["caption_density", "text_density", "music_presence", "sfx_frequency"]
    r = adapt(prof)
    ov = r.overrides
    assert ov.caption_max_words is None and ov.text_density is None and ov.music_level is None and ov.sfx_per_minute is None and ov.target_shot_duration is not None
    assert all("could not be measured" in r.skipped[d] for d in prof.unavailable)


def test_low_confidence_is_applied_at_half_strength_and_flagged_very_low_is_skipped():
    prof = reference()
    prof.confidence = {**prof.confidence, "motion_detection": 0.4, "transition_detection": 0.1}
    r = adapt(prof)
    full = adapt(reference()).overrides.motion_intensity
    base = AdaptationBaseline().motion_intensity
    assert base < r.overrides.motion_intensity < full  # half of the way
    assert any("Motion" in w and "low confidence" in w for w in r.warnings)
    assert "transition_frequency" in r.skipped and "uncertain" in r.skipped["transition_frequency"] and r.overrides.transition_frequency is None


def test_a_reference_without_captions_leaves_the_captions_alone():
    prof = reference()
    prof.caption_style = CaptionStats(False)
    r = adapt(prof)
    assert r.overrides.caption_max_words is None and r.overrides.caption_style is None and "no captions" in r.skipped["caption_density"]


def test_switched_off_music_and_sfx_are_not_turned_on_by_a_style():
    r = adapt(base=AdaptationBaseline(music_level=0.0, sfx_per_minute=0.0))
    assert r.overrides.music_level is None and r.overrides.sfx_per_minute is None
    assert "switched off" in r.skipped["music_presence"] and "switched off" in r.skipped["sfx_frequency"]


def test_an_empty_plan_has_reasons():
    r = adapt(cfg=settings(application_mode="CUSTOM", custom_dimensions=[]))
    assert r.overrides.is_empty and len(r.skipped) == len(DIMENSIONS)


# ---------------------------------------------------------------------------------------------- similarity and simulation
def test_style_similarity_compares_editing_features_only():
    a = reference()
    same = similarity_between(a, reference())
    other = similarity_between(a, reference(cuts_per_minute=5.0, motion_events_per_minute=1.0, average_motion_intensity=0.0, music_presence=0.0, sfx_per_minute=0.0))
    assert same.overall == 100.0 and other.overall < same.overall - 20 and other.pacing_match < 60 and other.audio_match < 60
    b = reference()
    b.unavailable = ["music_presence", "sfx_frequency"]
    excl = similarity_between(a, b)
    assert "music_presence" not in excl.compared and excl.overall == 100.0  # what one side could not measure takes no part


def test_simulation_moves_the_project_toward_the_reference_without_changing_it():
    ref = reference()
    base, cont = AdaptationBaseline(), content()
    mine = build_profile_from_features(StyleFeatures(duration=120.0, average_shot_duration=5.0, median_shot_duration=4.5, cuts_per_minute=11.0, major_changes_per_minute=3.0,
                                                     overlay_events_per_minute=3.0, text_events_per_minute=3.0, motion_events_per_minute=3.0, average_motion_intensity=0.15,
                                                     caption_coverage=0.85, captions_per_minute=18.0, music_presence=0.5, sfx_per_minute=1.0))
    proj = simulate(ref, StyleAdjustments(), settings(), cont, base, mine)
    assert proj.similarity_after > proj.similarity_before + 3
    assert proj.scores["pacing"] > proj.current["pacing"] and set(proj.scores) == set(DIMENSIONS) and proj.changes and proj.applied
    none = simulate(ref, StyleAdjustments(), settings(application_mode="CUSTOM", custom_dimensions=[]), cont, base, mine)
    assert none.scores == none.current and none.similarity_after == none.similarity_before
    # the profile objects were only read
    assert mine.score("pacing") == pytest.approx(build_profile_from_features(mine.features).score("pacing"))


def test_predicted_scores_follow_the_settings():
    base = AdaptationBaseline()
    faster = predicted_scores(AdaptationBaseline(base_shot_duration=2.5), content())
    assert faster["pacing"] > predicted_scores(base, content())["pacing"]
    assert predicted_scores(AdaptationBaseline(motion_intensity=0.9), content())["motion_intensity"] == pytest.approx(90.0)
    assert predicted_scores(AdaptationBaseline(sfx_per_minute=0.0), content())["sfx_frequency"] == 0.0


# ---------------------------------------------------------------------------------------------- robustness
def test_any_reasonable_profile_gives_valid_parameters():
    rng = np.random.default_rng(7)
    for _ in range(60):
        f = dict(cuts_per_minute=float(rng.uniform(0, 60)), average_shot_duration=float(rng.uniform(0.5, 20)), median_shot_duration=float(rng.uniform(0.5, 20)),
                 motion_events_per_minute=float(rng.uniform(0, 30)), average_motion_intensity=float(rng.uniform(0, 1)), text_events_per_minute=float(rng.uniform(0, 15)),
                 caption_coverage=float(rng.uniform(0, 1)), captions_per_minute=float(rng.uniform(0, 40)), average_words_per_caption=float(rng.uniform(0, 14)),
                 non_cut_share=float(rng.uniform(0, 1)), transition_events_per_minute=float(rng.uniform(0, 8)), music_presence=float(rng.uniform(0, 1)),
                 music_ducking_strength=float(rng.uniform(0, 1)), sfx_per_minute=float(rng.uniform(0, 20)), long_pause_frequency=float(rng.uniform(0, 12)))
        strength = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
        mode = str(rng.choice(["FULL", "BALANCED", "CUSTOM"]))
        dims = [d for d in DIMENSIONS if rng.random() < 0.6]
        adj = {d: float(rng.uniform(0, 100)) for d in DIMENSIONS if rng.random() < 0.2}
        cont = content(evidence_scene_share=float(rng.uniform(0, 1)), median_sentence_seconds=float(rng.uniform(0, 14)), median_words_per_second=float(rng.uniform(1.5, 4.0)))
        r = adapt(reference(**f), adj, settings(style_strength=strength, application_mode=mode, custom_dimensions=dims), cont)
        ov = r.overrides
        for name in ("motion_intensity", "transition_frequency", "text_density", "caption_density", "keyword_emphasis_rate", "ducking_strength", "pause_usage", "hook_shot_factor"):
            v = getattr(ov, name)
            assert v is None or 0.0 <= v <= 1.0, (name, v)
        assert ov.target_shot_duration is None or 0.5 <= ov.target_shot_duration <= 40
        assert ov.min_shot_duration is None or ov.min_shot_duration >= 1.0
        assert ov.max_shot_duration is None or ov.max_shot_duration <= 16.0
        assert ov.caption_max_words is None or 3 <= ov.caption_max_words <= 14
        assert ov.music_level is None or 0.05 <= ov.music_level <= 0.40
        assert ov.sfx_per_minute is None or 0.0 <= ov.sfx_per_minute <= 10.0
        assert all(0.0 <= v <= 100.0 for v in r.effective_targets.values())
        assert OriginalityGuard().audit_overrides(ov) == []
