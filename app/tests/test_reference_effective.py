"""Phase 7 -> Phase 4/5: the applied style reaches the editing engines only through ``app.editing.effective``.

The contract: with no applied style the engines produce exactly what they always did; with a style they produce different output (shot lengths, motion, captions,
audio levels); a setting the user made on purpose beats the style at run time; locked / user-owned content and the stored settings are never touched.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from app.editing.context import build_context
from app.editing.effective import (
    active_overrides,
    effective_audio_settings,
    effective_caption_settings,
    effective_editing_settings,
    hook_factor,
    keyword_rate,
    preset_with_reference,
    protected_parameters,
    style_signature,
)
from app.editing.models import EditingSettings
from app.editing.overrides import PROTECTED_BY, EditingStrategyOverrides
from app.editing.presets import PRESETS, preset_for, shot_factor
from app.project.project import Project
from app.tests.conftest import needs_ffmpeg
from app.timeline.clip import KIND_CAPTION, KIND_MEDIA, KIND_TEXT


def styled(p: Project, **ov) -> Project:
    p.reference_settings.enabled = True
    p.reference_style_overrides = EditingStrategyOverrides(**ov)
    return p


# ---------------------------------------------------------------------------------------------- pure functions
def test_protected_parameters_follow_the_users_own_settings():
    assert protected_parameters({"editing": ["motion_intensity"], "caption": [], "audio": []}, True) == {"motion_intensity"}
    assert protected_parameters({"editing": ["pacing"]}, True) == {"target_shot_duration", "min_shot_duration", "max_shot_duration", "hook_seconds", "hook_shot_factor"}
    assert protected_parameters({"caption": ["max_words"]}, True) == {"caption_max_words", "caption_density"}
    assert protected_parameters({"audio": ["important_level"]}, True) == {"ducking_strength"}
    assert protected_parameters({"editing": ["pacing", "motion_intensity"], "caption": ["style_id"], "audio": ["music_level"]}, False) == set()
    assert protected_parameters({}, True) == set()
    assert set(PROTECTED_BY) <= set(EditingStrategyOverrides().active()) | {n for n in PROTECTED_BY}


def test_no_applied_style_means_no_overrides_and_unchanged_settings():
    p = Project.new("x")
    assert active_overrides(p) is None and style_signature(p) == "" and keyword_rate(p) is None
    es = effective_editing_settings(p)
    assert es.reference is None and es.to_dict() == p.editing_settings.to_dict()
    assert effective_caption_settings(p) == p.caption_settings and effective_audio_settings(p) == p.audio_settings
    p.reference_style_overrides = EditingStrategyOverrides(motion_intensity=0.9)  # overrides stored but never applied (enabled is False)
    assert active_overrides(p) is None
    p.reference_settings.enabled = True
    p.reference_style_overrides = EditingStrategyOverrides()  # applied but empty
    assert active_overrides(p) is None


def test_the_style_is_filtered_by_the_users_settings_at_run_time():
    p = styled(Project.new("x"), motion_intensity=0.9, transition_frequency=0.8, caption_max_words=5)
    assert set(active_overrides(p).active()) == {"motion_intensity", "transition_frequency", "caption_max_words"}
    p.editing_settings.user_set = ["motion_intensity"]  # the user touched the motion slider after applying the style
    assert set(active_overrides(p).active()) == {"transition_frequency", "caption_max_words"}
    p.reference_settings.preserve_user_edits = False
    assert set(active_overrides(p).active()) == {"motion_intensity", "transition_frequency", "caption_max_words"}
    p.reference_settings.preserve_user_edits = True
    p.editing_settings.user_set, p.caption_settings.user_set = ["motion_intensity", "transition_frequency"], ["max_words"]
    assert active_overrides(p) is None  # nothing left of the style
    assert p.reference_style_overrides.motion_intensity == 0.9  # the stored overrides are untouched


def test_effective_editing_settings_lay_the_style_over_a_copy():
    p = styled(Project.new("x"), motion_intensity=0.9, transition_frequency=0.1, target_shot_duration=2.5)
    before = p.editing_settings.to_dict()
    es = effective_editing_settings(p)
    assert es.motion_intensity == 0.9 and es.transition_frequency == 0.1 and es.reference is not None and es.reference.target_shot_duration == 2.5
    assert p.editing_settings.to_dict() == before and p.editing_settings.reference is None and p.editing_settings.motion_intensity == 0.5
    assert es.pacing == p.editing_settings.pacing and es.style == p.editing_settings.style  # everything else is the user's
    explicit = effective_editing_settings(p, EditingSettings(style="dynamic", motion_intensity=0.2))
    assert explicit.style == "dynamic" and explicit.motion_intensity == 0.9


def test_the_preset_takes_shot_lengths_text_and_music_from_the_style():
    es = EditingSettings()
    base = PRESETS["professional"]
    assert preset_for(es) is base  # no style: the very same object as before
    es.reference = EditingStrategyOverrides(target_shot_duration=2.4, min_shot_duration=1.1, max_shot_duration=4.5, text_density=1.0, music_level=0.3)
    p = preset_for(es)
    assert p.base_shot * shot_factor(es) == pytest.approx(2.4) and p.min_shot == 1.1 and p.max_shot == 4.5 and p.text_per_minute == pytest.approx(12.0) and p.music_level == 0.3
    assert p.motion_ratio == base.motion_ratio and p.transitions == base.transitions and p.punch == base.punch  # nothing else moves
    es.pacing = 1.0  # the slider multiplies the target again, so the target stays the target
    assert preset_for(es).base_shot * shot_factor(es) == pytest.approx(2.4)
    es.reference = EditingStrategyOverrides(min_shot_duration=9.0)
    assert preset_for(es).min_shot <= preset_for(es).max_shot  # an impossible pair is repaired
    es.reference = EditingStrategyOverrides(motion_intensity=0.9)
    assert preset_for(es) is base  # parameters that are not preset values leave the preset alone
    assert preset_with_reference(base, EditingSettings()) is base


def test_hook_factor_only_applies_inside_the_opening():
    es = EditingSettings()
    assert hook_factor(es, 0.0) == 1.0
    es.reference = EditingStrategyOverrides(hook_seconds=10.0, hook_shot_factor=0.6)
    assert hook_factor(es, 0.0) == 0.6 and hook_factor(es, 9.9) == 0.6 and hook_factor(es, 10.0) == 1.0 and hook_factor(es, 30.0) == 1.0


def test_caption_settings_take_style_position_and_length_from_the_style():
    p = styled(Project.new("x"), caption_style="bold", caption_position="center", caption_max_words=5, keyword_emphasis_rate=0.8)
    cs = effective_caption_settings(p)
    assert (cs.style_id, cs.position, cs.max_words) == ("bold", "center", 5) and keyword_rate(p) == 0.8
    assert (p.caption_settings.style_id, p.caption_settings.position, p.caption_settings.max_words) == ("professional", "bottom", 12)  # stored settings are the user's
    unknown = effective_caption_settings(styled(Project.new("x"), caption_style="does-not-exist", caption_position="nowhere"))
    assert (unknown.style_id, unknown.position) == ("professional", "bottom")
    dense = effective_caption_settings(styled(Project.new("x"), caption_density=1.0))
    sparse = effective_caption_settings(styled(Project.new("x"), caption_density=0.2))
    assert dense.max_words < sparse.max_words  # density drives the length when no explicit word count is given
    own = styled(Project.new("x"), caption_style="news", caption_max_words=4)
    own.caption_settings.user_set = ["style_id"]
    assert effective_caption_settings(own).style_id == "professional" and effective_caption_settings(own).max_words == 4


def test_audio_settings_take_levels_ducking_pauses_and_sfx_from_the_style():
    p = styled(Project.new("x"), music_level=0.30, ducking_strength=1.0, pause_usage=1.0, sfx_per_minute=7.0)
    a = effective_audio_settings(p)
    assert a.music_level == pytest.approx(0.30) and a.important_level == pytest.approx(0.30 * 10 ** (-12 / 20), abs=1e-3) and a.pause_level == pytest.approx(0.30 * 1.66, abs=1e-3)
    assert a.max_sfx_per_minute == 7.0 and a.important_level <= a.music_level <= a.pause_level
    assert p.audio_settings.music_level == 0.18 and p.audio_settings.max_sfx_per_minute == 3.0  # stored settings untouched
    only_level = effective_audio_settings(styled(Project.new("x"), music_level=0.09))  # a quieter bed keeps the user's ducking depth relative to it
    assert only_level.important_level == pytest.approx(0.045, abs=1e-3) and only_level.pause_level == pytest.approx(0.12, abs=1e-3)
    mine = styled(Project.new("x"), music_level=0.30, sfx_per_minute=1.0)
    mine.audio_settings.important_level, mine.audio_settings.user_set = 0.05, ["important_level"]
    out = effective_audio_settings(mine)
    assert out.important_level == 0.05 and out.music_level == pytest.approx(0.30)  # a level the user set on purpose stays


def test_the_keyword_budget_scales_with_the_style_rate():
    from types import SimpleNamespace

    from app.captions.keywords import KeywordService

    words = [SimpleNamespace(word_id=f"w{i}", text=t) for i, t in enumerate("penalty fine audit risk danger mandatory violation illegal warning penalties".split())]
    scene = SimpleNamespace(numbers=[], entities=[])
    svc = KeywordService()
    default = svc.detect(scene, words)
    assert len(default) == len(svc.detect(scene, words, rate=0.5)) == 2  # the default budget is unchanged
    assert len(svc.detect(scene, words, rate=1.0)) > len(default) > len(svc.detect(scene, words, rate=0.0)) == 1


def test_a_changed_style_changes_the_cache_fingerprints():
    a = styled(Project.new("x"), target_shot_duration=2.5)
    b = styled(Project.new("x"), target_shot_duration=3.5)
    assert style_signature(a) and style_signature(a) != style_signature(b) != ""
    from app.editing.context import scene_hash  # noqa: F401  (the key is part of every scene hash; see the engine tests)


# ---------------------------------------------------------------------------------------------- the engines
def structure(p: Project) -> list:
    """The timeline as data (clip ids are random, so they stay out): enough to see any difference in what the engines produced."""
    rows = []
    for t in p.timeline.tracks:
        for c in t.clips:
            kf = tuple((k.property, round(k.time, 3), round(k.value, 3)) for k in c.keyframes)
            txt = (c.text or {}).get("content") or (c.text or {}).get("text") or ""
            rows.append((t.id, c.kind, round(c.timeline_start, 3), round(c.duration, 3), c.scene_id, c.slot, c.created_by, kf, (c.transition or {}).get("type", ""), txt,
                         json.dumps(c.audio, sort_keys=True, default=str), json.dumps(c.animation, sort_keys=True, default=str)[:80]))
    return sorted(rows, key=lambda r: (r[0], r[2], r[5], r[1]))


def visual(p: Project) -> list:
    return sorted((c for c in p.timeline.all_clips() if c.kind == KIND_MEDIA and c.track_id in ("track_v1", "track_v2", "track_v3")), key=lambda c: c.timeline_start)


def regen(ws) -> None:
    ws.editing.generate(force=True)
    assert ws.jobs.wait_idle(120)


@needs_ffmpeg
def test_without_a_style_the_editing_engine_is_unchanged_and_deterministic(edit_ws):
    ws = edit_ws
    p = ws.project
    ctx = build_context(p)
    assert ctx.settings.reference is None and ctx.settings.to_dict() == p.editing_settings.to_dict()
    regen(ws)
    first = structure(p)
    regen(ws)
    assert structure(p) == first  # same input, same timeline
    hashes = [s.input_hash for s in build_context(p).scenes]
    p.reference_settings.enabled = True  # a style switched on but empty changes nothing either
    assert [s.input_hash for s in build_context(p).scenes] == hashes
    p.reference_style_overrides = EditingStrategyOverrides(motion_intensity=0.9)
    assert [s.input_hash for s in build_context(p).scenes] != hashes  # an applied style changes the scene cache keys, so cached plans are not reused


@needs_ffmpeg
def test_a_fast_style_cuts_more_and_a_slow_style_cuts_less(edit_ws):
    ws = edit_ws
    p = ws.project
    regen(ws)
    base = visual(p)
    base_sig = structure(p)
    styled(p, target_shot_duration=1.8, min_shot_duration=1.2, max_shot_duration=3.2)
    regen(ws)
    fast = visual(p)
    assert len(fast) > len(base) * 1.25
    assert sum(c.duration for c in fast) / len(fast) < sum(c.duration for c in base) / len(base)
    assert all(c.duration >= 1.0 for c in fast[:-1])  # readable minimum
    styled(p, target_shot_duration=9.0, min_shot_duration=3.0, max_shot_duration=14.0)
    regen(ws)
    slow = visual(p)
    assert len(slow) < len(base)
    p.reference_settings.enabled = False  # style off: back to exactly the baseline
    regen(ws)
    assert structure(p) == base_sig
    ws.project.validate()


@needs_ffmpeg
def test_motion_transitions_and_text_follow_the_style(edit_ws):
    ws = edit_ws
    p = ws.project
    regen(ws)

    def counts():
        v = visual(p)
        moving = sum(1 for c in v if any(k.property in ("scale", "position_x", "position_y") for k in c.keyframes))
        trans = sum(1 for c in v if c.transition and c.transition.get("type") != "CUT")
        text = sum(1 for c in p.timeline.all_clips() if c.kind == KIND_TEXT)
        return moving, trans, text

    base = counts()
    styled(p, motion_intensity=1.0, transition_frequency=1.0, text_density=1.0)
    regen(ws)
    high = counts()
    styled(p, motion_intensity=0.0, transition_frequency=0.0, text_density=0.0)
    regen(ws)
    low = counts()
    assert high[0] > low[0] and high[1] >= low[1] and high[2] >= low[2]
    assert high[0] >= base[0] >= low[0] and high[2] >= base[2] >= low[2] and low[1] <= base[1] <= high[1]
    assert (high[0], high[1], high[2]) != (low[0], low[1], low[2])


@needs_ffmpeg
def test_the_users_own_settings_beat_the_style_in_the_engine(edit_ws):
    ws = edit_ws
    p = ws.project
    regen(ws)
    base_sig = structure(p)
    styled(p, motion_intensity=1.0, transition_frequency=1.0, target_shot_duration=1.8, max_shot_duration=3.0)
    ws.editing.update_settings(motion_intensity=0.5, transition_frequency=0.5, pacing=0.5, text_emphasis=False)  # touches nothing -> still defaults
    assert p.editing_settings.user_set == ["text_emphasis"]
    ws.editing.update_settings(motion_intensity=0.7)
    ws.editing.update_settings(motion_intensity=0.5)  # the user chose a value on purpose, even if it equals the default
    assert set(p.editing_settings.user_set) == {"text_emphasis", "motion_intensity"}
    eff = effective_editing_settings(p)
    assert eff.motion_intensity == 0.5 and eff.transition_frequency == 1.0 and eff.reference.target_shot_duration == 1.8  # the user's motion wins, the rest of the style applies
    p.editing_settings.user_set.append("pacing")
    eff = effective_editing_settings(p)
    assert eff.reference is not None and eff.reference.target_shot_duration is None and eff.reference.max_shot_duration is None  # the user's pacing wins over the whole shot-length group
    p.editing_settings.user_set += ["transition_frequency"]
    assert active_overrides(p) is None or "transition_frequency" not in active_overrides(p).active()
    p.reference_settings.enabled = False
    regen(ws)
    assert structure(p) != base_sig  # (text emphasis was switched off by the user above)
    p.reference_settings.enabled = True
    p.editing_settings.user_set = ["motion_intensity", "transition_frequency", "pacing"]
    regen(ws)
    no_style = structure(p)
    p.reference_settings.enabled = False
    regen(ws)
    assert structure(p) == no_style  # with every setting user-owned the style changes nothing at all


@needs_ffmpeg
def test_locked_scenes_are_not_touched_by_a_style(edit_ws):
    ws = edit_ws
    p = ws.project
    regen(ws)
    sid = p.scenes[1].id
    ws.editing.set_lock(sid, "SCENE")
    locked_before = [r for r in structure(p) if r[4] == sid]
    styled(p, target_shot_duration=1.6, max_shot_duration=2.8, motion_intensity=1.0)
    regen(ws)
    assert [r for r in structure(p) if r[4] == sid] == locked_before


# ---------------------------------------------------------------------------------------------- phase 5
def captions(p: Project) -> list:
    return sorted((c for c in p.timeline.all_clips() if c.kind == KIND_CAPTION), key=lambda c: c.timeline_start)


@needs_ffmpeg
def test_captions_follow_the_style_and_return_to_the_users_settings(pres_ws):
    ws = pres_ws
    p = ws.project

    def run():
        ws.presentation.generate(["CAPTIONS"])
        assert ws.jobs.wait_idle(120)

    run()
    base = captions(p)
    base_sig = structure(p)
    assert base and all(c.text["style_id"] == "professional" and c.text["position"] == "bottom" for c in base)
    styled(p, caption_style="bold", caption_position="center", caption_max_words=4)
    assert ws.presentation.staleness()["settings_changed"]  # the captions on the timeline were made with other settings
    run()
    short = captions(p)
    assert len(short) > len(base) and all(c.text["style_id"] == "bold" and c.text["position"] == "center" for c in short)
    assert max(len(c.text["words"]) for c in short) <= 4
    assert p.caption_settings.style_id == "professional" and p.caption_settings.max_words == 12  # the stored caption settings stay the user's
    ws.presentation.update_caption_settings(style_id="news")  # the user picks a style: from now on it wins
    assert "style_id" in p.caption_settings.user_set
    run()
    mixed = captions(p)
    assert all(c.text["style_id"] == "news" and c.text["position"] == "center" for c in mixed) and max(len(c.text["words"]) for c in mixed) <= 4
    p.reference_settings.enabled = False
    ws.presentation.update_caption_settings(style_id="professional")
    run()
    assert structure(p) == base_sig


@needs_ffmpeg
def test_music_ducking_and_sfx_follow_the_style(pres_ws):
    ws = pres_ws
    p = ws.project
    ws.presentation.add_music(ws.audio_by["music_bed.wav"].id)

    def levels():
        music = [c for c in p.timeline.all_clips() if c.audio.get("role") == "MUSIC"]
        vals = [k.value for c in music for k in c.keyframes if k.property == "volume"]
        sfx = [c for c in p.timeline.all_clips() if c.audio.get("role") == "SFX" and c.created_by == "AI"]
        return (max(vals), min(vals)) if vals else (0.0, 0.0), len(sfx)

    ws.presentation.generate(["AUDIO"])
    assert ws.jobs.wait_idle(120)
    (hi, lo), n_sfx = levels()
    assert hi > 0 and lo < hi and n_sfx >= 1
    styled(p, music_level=0.32, ducking_strength=1.0, pause_usage=1.0, sfx_per_minute=0.0)
    ws.presentation.generate(["AUDIO"])
    assert ws.jobs.wait_idle(120)
    (hi2, lo2), n2 = levels()
    assert hi2 > hi and lo2 / hi2 < lo / hi and n2 == 0  # louder bed, relatively deeper duck, no sound effects
    assert p.audio_settings.music_level == 0.18
    p.reference_settings.enabled = False
    ws.presentation.generate(["AUDIO"])
    assert ws.jobs.wait_idle(120)
    assert levels() == ((hi, lo), n_sfx)


def test_the_effective_helpers_never_mutate_what_they_are_given():
    p = styled(Project.new("x"), motion_intensity=0.9, caption_max_words=4, music_level=0.3)
    snap = deepcopy((p.editing_settings.to_dict(), p.caption_settings, p.audio_settings, p.reference_style_overrides.to_dict()))
    effective_editing_settings(p), effective_caption_settings(p), effective_audio_settings(p), active_overrides(p)
    assert (p.editing_settings.to_dict(), p.caption_settings, p.audio_settings, p.reference_style_overrides.to_dict()) == snap
