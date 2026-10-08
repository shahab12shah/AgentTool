"""Phase 7 project metrics: the user's own project measured on the same style dimensions as a reference (so the two can be compared)."""

from __future__ import annotations

import math

import pytest

from app.editing.effective import duck_level_for, ducking_strength
from app.media.asset import Asset, AssetType, SourceType
from app.project.project import Project
from app.reference.project_metrics import (
    adaptation_baseline,
    project_content,
    project_duration,
    project_style_features,
    project_style_profile,
)
from app.reference.style_adapter import similarity_between
from app.reference.style_model import DIMENSIONS, build_profile_from_features
from app.tests.conftest import needs_ffmpeg
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT, Clip
from app.timeline.keyframes import Keyframe


# ---------------------------------------------------------------------------------------------- builders
def project(width: int = 1920) -> Project:
    p = Project.new("Metrics")
    p.settings.width, p.settings.height = width, int(width * 9 / 16)
    return p


def add_asset(p: Project, aid: str, kind: AssetType = AssetType.VIDEO, duration: float | None = 60.0) -> None:
    if p.assets.get(aid) is not None:
        return
    p.assets.add(Asset(aid, kind, SourceType.USER_MEDIA, f"media/{aid}.mp4", aid, duration, 1920, 1080, 24.0))


def visual(p: Project, n: int, length: float = 3.0, start: float = 0.0, track: str = "track_v1", **kw) -> list[Clip]:
    add_asset(p, "v")
    out = []
    for i in range(n):
        c = Clip(f"{track}_{i}", track, "v", start + i * length, length, **kw)
        p.timeline.get_track(track).clips.append(c)
        out.append(c)
    return out


def cap(p: Project, i: int, start: float, dur: float, words: int, emphasis: int = 0, position: str = "bottom") -> Clip:
    c = Clip(f"cap_{i}", "track_v6", "", start, dur, kind=KIND_CAPTION, text={"words": [{"word_id": f"w{j}", "text": "x", "start": start, "end": start + dur} for j in range(words)],
                                                                          "lines": ["x " * words], "emphasis": [{"word_index": 0}] * emphasis, "position": position},
             animation={"in": {"preset": "fade_in"}})
    p.timeline.get_track("track_v6").clips.append(c)
    return c


# ---------------------------------------------------------------------------------------------- features
def test_an_empty_project_has_nothing_to_measure_and_says_so():
    p = project()
    f = project_style_features(p)
    assert f.duration == 0 and f.cuts_per_minute == 0 and f.average_shot_duration == 0
    prof = project_style_profile(p)
    assert set(prof.unavailable) == {"pacing", "visual_density", "motion_intensity", "transition_frequency"} and prof.categories["pacing"] == "No timeline yet"
    assert prof.confidence["overall"] == 0.0 and project_duration(p) == 0.0


def test_cut_rhythm_comes_from_the_visual_clips():
    p = project()
    visual(p, 10, 3.0)  # a cut every 3 s for 30 s
    f = project_style_features(p)
    assert f.duration == pytest.approx(30.0) and f.cuts_per_minute == pytest.approx(18.0)  # 9 cuts in half a minute
    assert f.average_shot_duration == pytest.approx(3.0) and f.median_shot_duration == pytest.approx(3.0)
    prof = project_style_profile(p)
    assert prof.is_available("pacing") and prof.categories["pacing"] == "Fast" and prof.shot_stats.count == 10 and prof.shot_stats.minimum_shot_duration == pytest.approx(3.0)
    assert prof.confidence["overall"] == 1.0


def test_shot_lengths_are_not_just_an_average():
    p = project()
    t = 0.0
    for i, d in enumerate([2.0, 2.0, 10.0, 2.0, 2.0, 10.0]):
        add_asset(p, "v")
        p.timeline.get_track("track_v1").clips.append(Clip(f"c{i}", "track_v1", "v", t, d))
        t += d
    f = project_style_features(p)
    assert f.average_shot_duration == pytest.approx(28.0 / 6) and f.median_shot_duration == pytest.approx(2.0)


def test_clips_starting_together_are_one_shot_and_hidden_or_other_tracks_do_not_count():
    p = project()
    visual(p, 4, 5.0)
    visual(p, 4, 5.0, track="track_v2")  # a second layer starting at the same instants
    f = project_style_features(p)
    assert f.cuts_per_minute == pytest.approx(3 / (20 / 60))
    p.timeline.get_track("track_v2").hidden = True
    assert project_style_features(p).cuts_per_minute == pytest.approx(f.cuts_per_minute)


def test_text_graphics_captions_and_transitions_are_counted():
    p = project()
    clips = visual(p, 6, 10.0)  # 60 s
    for i in range(4):
        p.timeline.get_track("track_v5").clips.append(Clip(f"t{i}", "track_v5", "", i * 10.0, 2.0, kind=KIND_TEXT, text={"style": "HEADLINE" if i < 2 else "NUMBER_CARD", "content": "x", "position": [0.5, 0.15], "size": 60}))
    p.timeline.get_track("track_v4").clips.append(Clip("g0", "track_v4", "", 5.0, 2.0, kind=KIND_GRAPHIC))
    for i in range(10):
        cap(p, i, i * 6.0, 3.0, 5, emphasis=1 if i < 4 else 0)
    clips[2].transition = {"type": "DISSOLVE", "duration": 0.6}
    clips[4].transition = {"type": "CUT", "duration": 0.0}
    f = project_style_features(p)
    assert f.text_events_per_minute == pytest.approx(4.0) and f.graphic_events_per_minute == pytest.approx(1.0) and f.overlay_events_per_minute == pytest.approx(5.0)
    assert f.captions_per_minute == pytest.approx(10.0) and f.caption_coverage == pytest.approx(0.5) and f.average_words_per_caption == pytest.approx(5.0)
    assert f.caption_emphasis_rate == pytest.approx(0.4)
    assert f.transition_events_per_minute == pytest.approx(1.0) and f.non_cut_share == pytest.approx(1 / 5)
    prof = project_style_profile(p)
    cs, tx = prof.caption_style, prof.text
    assert cs.caption_present and cs.caption_position == "bottom" and cs.caption_emphasis_rate == pytest.approx(0.4) and cs.caption_animation_rate == 1.0
    assert tx.headline_frequency == pytest.approx(2.0) and tx.number_graphic_frequency == pytest.approx(2.0) and tx.position_share.get("top") == 1.0


def test_motion_comes_from_keyframes_and_uses_the_reference_intensity_scale():
    p = project()
    clips = visual(p, 6, 10.0)
    clips[0].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 10.0, 1.15)]  # a 15 % zoom over 10 s
    clips[1].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 2.0, 1.2)]  # a quick punch-in
    clips[2].keyframes = [Keyframe("position_x", 0.0, 0.0), Keyframe("position_x", 10.0, 400.0)]  # a pan over 21 % of the width
    clips[3].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 10.0, 1.01)]  # below the minimum of a zoom event
    f = project_style_features(p)
    assert f.zoom_events_per_minute == pytest.approx(2.0) and f.motion_events_per_minute == pytest.approx(3.0)
    assert 0.0 < f.average_motion_intensity < 0.5
    gentle = f.average_motion_intensity
    clips[0].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 10.0, 2.0)]  # a much stronger zoom over the same time -> stronger intensity
    assert project_style_features(p).average_motion_intensity > gentle
    prof = project_style_profile(p)
    assert prof.motion.zoom.events == 2 and prof.motion.static_shot_share == pytest.approx(3 / 6) and prof.categories["motion_intensity"] in ("Minimal", "Subtle", "Moderate", "Strong")


def test_a_zoom_in_and_back_out_is_two_events():
    p = project()
    c = visual(p, 1, 8.0)[0]
    c.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 3.0, 1.8), Keyframe("scale", 6.0, 1.8), Keyframe("scale", 8.0, 1.0)]  # the evidence zoom
    assert project_style_profile(p).motion.zoom.events == 2


def test_audio_is_measured_from_the_clips_and_settings():
    p = project()
    visual(p, 10, 6.0)  # 60 s
    add_asset(p, "m", AssetType.AUDIO, 30.0)
    add_asset(p, "s", AssetType.AUDIO, 1.0)
    p.assets.add(Asset("vo", AssetType.AUDIO, SourceType.USER_MEDIA, "media/vo.wav", "vo", 60.0))
    p.timeline.get_track("track_a1").clips.append(Clip("voice", "track_a1", "vo", 0.0, 60.0, slot="voice"))
    p.timeline.get_track("track_a2").clips.append(Clip("m1", "track_a2", "m", 0.0, 30.0, audio={"role": "MUSIC", "volume": 1.0}))
    for i in range(6):
        p.timeline.get_track("track_a3").clips.append(Clip(f"s{i}", "track_a3", "s", 5.0 + 9.0 * i, 1.0, audio={"role": "SFX"}))
    f = project_style_features(p)
    assert f.music_presence == pytest.approx(0.5) and f.sfx_per_minute == pytest.approx(6.0) and f.voice_dominance == pytest.approx(1.0)
    assert f.music_ducking_strength == pytest.approx(0.5, abs=0.02)  # the default music level 0.18 ducks to 0.09 (6 dB)
    p.audio_settings.auto_ducking = False
    assert project_style_features(p).music_ducking_strength == 0.0
    prof = project_style_profile(p)
    assert prof.audio.music_behavior in ("Continuous", "Minimal", "Dynamic") and prof.audio.sfx_class == "Moderate" and prof.audio.has_audio


def test_pauses_and_silence_come_from_the_voice_analysis():
    from app.presentation.models import VoiceAnalysis

    p = project()
    visual(p, 10, 6.0)
    p.audio_analysis = VoiceAnalysis(silence_regions=[[10.0, 12.0], [30.0, 31.0]], pauses=[[5.0, 5.5], [20.0, 21.5], [40.0, 42.0]], dynamic_range_db=9.0, speech_ratio=0.8)
    f = project_style_features(p)
    assert f.silence_percentage == pytest.approx(100 * 3 / 60) and f.average_pause_duration == pytest.approx((0.5 + 1.5 + 2.0) / 3)
    assert f.long_pause_frequency == pytest.approx(2.0) and f.audio_dynamic_range == 9.0 and f.voice_dominance == pytest.approx(0.8)


def test_the_opening_intensity_compares_the_first_seconds_with_the_rest():
    p = project()
    t = 0.0
    for i in range(8):  # eight 1.2 s shots, then 4 s shots
        add_asset(p, "v")
        p.timeline.get_track("track_v1").clips.append(Clip(f"a{i}", "track_v1", "v", t, 1.2))
        t += 1.2
    for i in range(8):
        p.timeline.get_track("track_v1").clips.append(Clip(f"b{i}", "track_v1", "v", t, 4.0))
        t += 4.0
    assert project_style_features(p).hook_intensity > 0.3
    q = project()
    visual(q, 12, 3.0)
    assert project_style_features(q).hook_intensity == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------------------------- same scale as a reference
def test_project_and_reference_profiles_are_comparable():
    p = project()
    visual(p, 20, 3.0)
    mine = project_style_profile(p)
    ref = build_profile_from_features(project_style_features(p), "ref")
    assert mine.scores.as_dict() == ref.scores.as_dict()  # one pipeline for both
    sim = similarity_between(mine, ref)
    assert sim.overall == 100.0 or set(sim.compared) < set(DIMENSIONS)  # unavailable dimensions (none here) would be excluded rather than scored 0
    q = project()
    visual(q, 5, 12.0)
    assert similarity_between(mine, project_style_profile(q)).pacing_match < 75


# ---------------------------------------------------------------------------------------------- baseline
def test_the_baseline_holds_the_users_own_settings_in_override_units():
    p = project()
    b = adaptation_baseline(p)
    assert b.base_shot_duration == pytest.approx(4.8) and b.min_shot_duration == 1.8 and b.max_shot_duration == 8.0
    assert b.motion_intensity == 0.5 and b.transition_frequency == 0.5 and b.text_density == 0.5 and b.text_per_minute == 6
    assert b.caption_max_words == 12 and b.caption_style == "professional" and b.caption_position == "bottom" and b.keyword_emphasis_rate == 0.5
    assert b.music_level == 0.18 and b.ducking_strength == pytest.approx(0.5, abs=0.02) and b.pause_usage == pytest.approx(0.5, abs=0.02) and b.sfx_per_minute == 3.0
    assert b.user_set == {"editing": [], "caption": [], "audio": []}


def test_the_baseline_follows_pacing_style_and_switches():
    p = project()
    p.editing_settings.style, p.editing_settings.pacing = "dynamic", 1.0
    p.editing_settings.text_emphasis = p.editing_settings.number_emphasis = False
    p.editing_settings.user_set = ["pacing"]
    p.audio_settings.music_enabled, p.audio_settings.sfx_enabled = False, False
    p.caption_settings.max_words = 6
    b = adaptation_baseline(p)
    assert b.base_shot_duration == pytest.approx(3.2 * 0.55) and b.max_shot_duration == 5.5 and b.text_density == 0.0
    assert b.music_level == 0.0 and b.sfx_per_minute == 0.0 and b.caption_max_words == 6 and b.user_set["editing"] == ["pacing"]


def test_the_baseline_never_includes_an_already_applied_style():
    from app.editing.overrides import EditingStrategyOverrides

    p = project()
    p.reference_settings.enabled = True
    p.reference_style_overrides = EditingStrategyOverrides(target_shot_duration=2.0, motion_intensity=0.9, music_level=0.3)
    b = adaptation_baseline(p)
    assert b.base_shot_duration == pytest.approx(4.8) and b.motion_intensity == 0.5 and b.music_level == 0.18


def test_ducking_maths_round_trips():
    for s in (0.0, 0.25, 0.5, 1.0):
        assert ducking_strength(0.2, duck_level_for(0.2, s)) == pytest.approx(s, abs=1e-6)
    assert ducking_strength(0.18, 0.09) == pytest.approx(20 * math.log10(2) / 12)
    assert ducking_strength(0.0, 0.1) == 0.0 and ducking_strength(0.2, 0.0) == 1.0 and ducking_strength(0.1, 0.5) == 0.0  # a "duck" above the music level is no duck


# ---------------------------------------------------------------------------------------------- real projects
@needs_ffmpeg
def test_content_comes_from_the_scenes_the_transcript_and_the_visuals(edit_ws):
    p = edit_ws.project
    c = project_content(p)
    assert c.scene_count == len(p.scenes) and c.duration > 10 and c.median_scene_seconds > 0
    assert 0.0 <= c.evidence_scene_share <= 1.0 and 0.0 <= c.number_scene_share <= 1.0 and 0.0 <= c.still_image_share <= 1.0
    assert 1.0 < c.median_words_per_second < 5.0 and c.median_sentence_seconds > 0.5 and c.average_information_density > 0
    assert c.still_image_share > 0  # the demo project assigns stills
    empty = project_content(Project.new("none"))
    assert empty.scene_count == 0 and empty.median_words_per_second == 2.5


@needs_ffmpeg
def test_a_generated_ai_edit_is_measured_on_every_dimension(edit_ws):
    ws = edit_ws
    ws.editing.generate()
    assert ws.jobs.wait_idle(60)
    p = ws.project
    prof = project_style_profile(p)
    assert prof.unavailable == [] and prof.features.cuts_per_minute > 0 and prof.shot_stats.count >= len(p.scenes) // 2
    assert {d: prof.score(d) for d in DIMENSIONS}["text_density"] > 0  # the AI edit places number and name overlays
    mine = ws.reference.project_profile()
    assert mine.scores.as_dict() == prof.scores.as_dict()
    b = adaptation_baseline(p)
    assert b.base_shot_duration == pytest.approx(4.8)
