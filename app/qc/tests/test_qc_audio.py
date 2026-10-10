"""Audio checker: voice, music, effects and silence, measured on real (synthetic) audio."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from app.presentation.models import VoiceAnalysis
from app.qc.audio_checker import AudioChecker
from app.qc.severity import Severity
from app.qc.tests.conftest import needs_ffmpeg
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, narrate, new_project, qc_ctx, run_checker
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.probe import MediaProbeService
from app.timeline.keyframes import Keyframe

SR = 16000
TEXT1 = "Silver prices rose sharply last week and analysts were surprised by the speed."
TEXT2 = "Industrial demand from solar panel makers is the main reason behind the move."


def write_wav(path, x: np.ndarray, channels: int = 1):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def speech_signal(words, total, amp=0.4, noise=0.0, scale_after=None, hole=None, clip=False):
    n = int(total * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float64)
    if noise:
        x += np.random.default_rng(3).standard_normal(n) * noise
    for w in words:
        a, b = int(w.start * SR), min(n, int(w.end * SR))
        env = np.ones(b - a)
        r = min(160, (b - a) // 4)
        if r > 0:
            env[:r], env[-r:] = np.linspace(0, 1, r), np.linspace(1, 0, r)
        x[a:b] += amp * np.sin(2 * np.pi * 220 * t[a:b]) * env
    if scale_after:
        x[int(scale_after[0] * SR):] *= scale_after[1]
    if hole:
        x[int(hole[0] * SR):int(hole[1] * SR)] = 0.0
    return np.clip(x * (4.0 if clip else 1.0), -1, 1)


def audio_project(tmp_path, *, total=18.0, scenes=((0, 6, TEXT1), (9, 15, TEXT2)), **sig):
    """Voice speaking 0-6 s and 9-15 s with real pauses between (and after), a real WAV, a voice clip on track_a1."""
    p = new_project(tmp_path, seconds=total)
    for a, b, text in scenes:
        add_scene(p, a, b, text)
    narrate(p)
    va = p.assets.get(p.voice_over.asset_id)
    write_wav(p.root / va.path, speech_signal(p.transcription.transcript.words, total, **sig))
    va.size_bytes = (p.root / va.path).stat().st_size
    add_clip(p, "track_a1", va, 0, total, created_by="SYSTEM", slot="voice", audio={"role": "VOICE", "volume": 1.0})
    return p, va


def music(p, *, start=0.0, dur=18.0, volume=0.18, gain_kf=None, name="music.wav", amp=0.3, role="MUSIC", track="track_a2", created_by="AI", **kw):
    path = p.root / "media" / "audio" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(dur * SR)) / SR
    write_wav(path, amp * np.sin(2 * np.pi * 180 * t))
    a = add_asset(p, name, "audio", duration=dur, w=None, h=None, write=False)
    a.path, a.size_bytes = f"media/audio/{name}", path.stat().st_size
    c = add_clip(p, track, a, start, dur, audio={"role": role, "volume": volume}, metadata={"assignment_id": f"as_{name}"}, created_by=created_by, **kw)
    if gain_kf:
        c.keyframes = [Keyframe("volume", float(t0), float(v)) for t0, v in gain_kf]
    return a, c


def duck_kf(base_gap=1.0, under=0.15):
    """Dips under the two spoken stretches (0-6 s and 9-15 s) with 0.25 s ramps."""
    return [(0, under), (6.0, under), (6.4, base_gap), (8.6, base_gap), (9.0, under), (15.0, under), (15.4, base_gap), (17.9, base_gap)]


def ff_ctx(p, **kw):
    ff = FFmpegService()
    return qc_ctx(p, ffmpeg=ff, probe=MediaProbeService(ff), **kw)


def run(p, **kw):
    return run_checker(AudioChecker(), ff_ctx(p, **kw))


def real(out):
    return [i for i in out.issues if i.severity is not Severity.INFO]


pytestmark = needs_ffmpeg


# ---------------------------------------------------------------- clean and basics
def test_clean_project_has_no_audio_problem_and_pauses_are_intentional(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=0.5, gain_kf=duck_kf(1.0, 0.15))
    out = run(p)
    assert real(out) == [], [(i.code, i.description) for i in out.issues]
    assert codes(out) == ["silence.intentional"]
    assert out.metrics["speech_ratio"] > 0.5 and out.metrics["duck_depth_db"]["min"] > 5


def test_checker_contract(tmp_path):
    p, _ = audio_project(tmp_path)
    c = AudioChecker()
    assert c.expensive and not c.scene_local and c.settings_sections == ("audio", "intentional_gaps") and {"audio", "transcript"} <= set(c.domains)
    before = p.to_document()
    run(p)
    assert p.to_document() == before


# ---------------------------------------------------------------- voice
def test_voice_clipping_is_detected_and_severe_clipping_is_an_error(tmp_path):
    p, _ = audio_project(tmp_path, amp=0.5, clip=True)
    i = find(run(p), "audio.voice_clipping")
    assert len(i) == 1 and i[0].severity is Severity.ERROR and i[0].confidence == 100.0
    p2, _ = audio_project(tmp_path / "clean", amp=0.3)
    assert find(run(p2), "audio.voice_clipping") == []


def test_inaudible_voice_is_critical(tmp_path):
    p, _ = audio_project(tmp_path)
    p.timeline.get_track("track_a1").muted = True
    i = find(run(p), "audio.voice_missing")
    assert len(i) == 1 and i[0].severity is Severity.CRITICAL and "muted" in i[0].description


def test_voice_too_quiet(tmp_path):
    p, _ = audio_project(tmp_path, amp=0.003)
    i = find(run(p), "audio.voice_quiet")
    assert len(i) == 1 and i[0].severity in (Severity.WARNING, Severity.ERROR) and "dBFS" in i[0].description
    p2, _ = audio_project(tmp_path / "ok", amp=0.2)
    assert find(run(p2), "audio.voice_quiet") == []


def test_voice_gain_counts_when_judging_loudness(tmp_path):
    p, _ = audio_project(tmp_path, amp=0.004)
    p.timeline.get_track("track_a1").clips[0].audio["volume"] = 30.0  # +29.5 dB makes the quiet recording fine
    assert find(run(p), "audio.voice_quiet") == []


def test_sudden_voice_level_change(tmp_path):
    p, _ = audio_project(tmp_path, amp=0.1, scale_after=(8.0, 6.0))
    i = find(run(p), "audio.voice_jump")
    assert len(i) == 1 and "dB" in i[0].description and i[0].start_time == pytest.approx(6.0, abs=0.7)
    p2, _ = audio_project(tmp_path / "even", amp=0.1)
    assert find(run(p2), "audio.voice_jump") == []


def test_background_noise_between_words(tmp_path):
    p, _ = audio_project(tmp_path, noise=0.03)
    i = find(run(p), "audio.voice_noise")
    assert len(i) == 1 and i[0].metrics["noise_floor_db"] > -45
    p2, _ = audio_project(tmp_path / "quiet", noise=0.0)
    assert find(run(p2), "audio.voice_noise") == []


def test_voice_clip_shorter_than_the_recording(tmp_path):
    p, _ = audio_project(tmp_path)
    c = p.timeline.get_track("track_a1").clips[0]
    c.duration, c.source_out = 12.0, 12.0
    i = find(run(p), "audio.voice_duration")
    assert len(i) == 1 and i[0].severity is Severity.ERROR and "6.0 s missing" in i[0].description


def test_silent_channel_and_inverted_channel(tmp_path):
    p, va = audio_project(tmp_path)
    mono = speech_signal(p.transcription.transcript.words, 18.0, amp=0.3)
    va.channels = 2
    write_wav(p.root / va.path, np.stack([mono, np.zeros_like(mono)], axis=1).reshape(-1), channels=2)
    i = find(run(p), "audio.channel")
    assert len(i) == 1 and "right channel" in i[0].description
    write_wav(p.root / va.path, np.stack([mono, -mono], axis=1).reshape(-1), channels=2)
    i = find(run(p), "audio.channel")
    assert len(i) == 1 and "out of phase" in i[0].title
    write_wav(p.root / va.path, np.stack([mono, mono], axis=1).reshape(-1), channels=2)
    assert find(run(p), "audio.channel") == []


# ---------------------------------------------------------------- music
def test_music_at_full_level_covers_the_voice(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=1.0, amp=0.6)  # +4 dB against the voice
    i = find(run(p), "audio.music_masks_speech")
    assert len(i) == 1 and i[0].severity is Severity.ERROR and i[0].fix.kind == "audio.duck"
    assert i[0].fix.params["role"] == "MUSIC" and i[0].fix.params["spans"] and i[0].auto_fix_available


def test_undermined_ducking_is_reported_with_a_safe_duck_fix(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=0.5)  # constant level: a duck depth of 0 dB
    out = run(p)
    i = find(out, "audio.insufficient_ducking")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and "0.0 dB" in i[0].current_value
    assert i[0].fix.kind == "audio.duck" and i[0].auto_fix_available and i[0].auto_fix_safe
    assert 0 < i[0].fix.params["target_gain"] <= p.audio_settings.music_level
    assert find(out, "audio.music_masks_speech") == [] and find(out, "audio.music_loud") == []  # one cause, one issue


def test_properly_ducked_music_is_clean(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=0.5, gain_kf=duck_kf())
    out = run(p)
    assert not [c for c in codes(out) if c.startswith("audio.music") or c == "audio.insufficient_ducking"]


def test_ducked_but_still_too_loud_music(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=1.0, amp=0.45, gain_kf=duck_kf(1.0, 0.45))  # 7 dB of ducking (enough) but still only -5 dB under the voice
    i = find(run(p), "audio.music_loud")
    assert len(i) == 1 and i[0].fix.kind == "audio.level"


def test_music_fix_is_disabled_for_user_owned_and_mix_locked_music(tmp_path):
    p, _ = audio_project(tmp_path)
    a, c = music(p, volume=0.5)
    c.created_by = "USER"
    i = find(run(p), "audio.insufficient_ducking")[0]
    assert i.locked and not i.auto_fix_available and "you" in i.fix_blocked_reason.lower()
    c.created_by, c.locked = "AI", True  # MIX lock
    j = find(run(p), "audio.insufficient_ducking")[0]
    assert j.locked and not j.auto_fix_available


def test_abrupt_music_level_change(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=0.5, gain_kf=[(0, 1.0), (3.0, 1.0), (3.05, 0.05), (10, 0.05), (17.9, 0.05)])
    i = find(run(p), "audio.music_jump")
    assert len(i) == 1 and i[0].start_time == pytest.approx(3.0, abs=0.2)
    p2, _ = audio_project(tmp_path / "smooth")
    music(p2, volume=0.5, gain_kf=[(0, 1.0), (3.0, 1.0), (3.4, 0.05), (10, 0.05), (17.9, 0.05)])
    assert find(run(p2), "audio.music_jump") == []


def test_music_disabled_track_is_ignored(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, volume=1.0)
    p.timeline.get_track("track_a2").muted = True
    assert not [c for c in codes(run(p)) if c.startswith("audio.music")]


# ---------------------------------------------------------------- sound effects
def test_loud_effect_over_speech(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, start=2.0, dur=1.0, volume=1.0, name="hit.wav", amp=0.9, role="SFX", track="track_a3")
    i = find(run(p), "audio.sfx_loud")
    assert len(i) == 1 and i[0].fix.kind == "audio.level" and i[0].start_time == pytest.approx(2.0)
    assert i[0].fix.params["volume"] < 1.0


def test_quiet_effect_in_a_pause_is_fine(tmp_path):
    p, _ = audio_project(tmp_path)
    music(p, start=7.0, dur=1.0, volume=1.0, name="hit.wav", amp=0.9, role="SFX", track="track_a3")  # in the 6-9 s pause
    music(p, start=2.0, dur=1.0, volume=0.02, name="soft.wav", amp=0.3, role="SFX", track="track_a3")
    assert not [c for c in codes(run(p)) if c.startswith("audio.sfx")]


def test_effects_that_come_too_often_or_repeat_quickly(tmp_path):
    p, _ = audio_project(tmp_path)
    p.qc_settings.audio.sfx_repeat_per_minute = 3
    for k in range(5):
        music(p, start=0.5 + k * 3.0, dur=0.5, volume=0.02, name="pop.wav", amp=0.3, role="SFX", track="track_a3")
    out = run(p)
    kinds = sorted(i.severity.value for i in find(out, "audio.sfx_repeat"))
    assert len(kinds) == 2  # one "too many per minute" and one "same effect twice within the minimum gap"
    assert any("within" in i.description for i in find(out, "audio.sfx_repeat"))


# ---------------------------------------------------------------- silence
def test_silence_where_the_script_has_words_is_an_error(tmp_path):
    p, _ = audio_project(tmp_path, hole=(10.0, 12.0))
    i = find(run(p), "silence.accidental")
    assert len(i) == 1 and i[0].severity is Severity.ERROR and i[0].start_time == pytest.approx(10.0, abs=0.4) and "transcript has" in i[0].description
    assert i[0].fix is None and not i[0].auto_fix_available  # silence is never fixed automatically


def test_pause_after_a_sentence_is_intentional_not_a_problem(tmp_path):
    p, _ = audio_project(tmp_path)
    out = run(p)
    assert find(out, "silence.accidental") == [] and find(out, "silence.unnatural") == [] and find(out, "silence.excessive") == []
    assert find(out, "silence.intentional")[0].severity is Severity.INFO


def test_user_declared_gap_is_never_reported(tmp_path):
    p, _ = audio_project(tmp_path, hole=(10.0, 12.0))
    p.qc_settings.intentional_gaps = [[9.5, 12.5]]
    assert find(run(p), "silence.accidental") == []


def test_very_long_silence_after_a_sentence_is_excessive(tmp_path):
    p, _ = audio_project(tmp_path, total=30.0, scenes=((0, 6, TEXT1), (18, 24, TEXT2)))
    i = find(run(p), "silence.excessive")
    assert len(i) >= 1 and i[0].severity is Severity.WARNING and "after a finished sentence" in i[0].description


def test_silence_inside_a_sentence_is_unnatural(tmp_path):
    cut = "Silver prices rose sharply last week and analysts were surprised"  # no sentence end before the gap
    p, _ = audio_project(tmp_path, scenes=((0, 6, cut), (9, 15, TEXT2)))
    i = find(run(p), "silence.unnatural")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and i[0].confidence <= 80


# ---------------------------------------------------------------- without FFmpeg
def test_without_ffmpeg_the_stored_analysis_is_used_and_nothing_crashes(tmp_path):
    p, va = audio_project(tmp_path, amp=0.3)
    p.audio_analysis = VoiceAnalysis(asset_id=va.id, audio_hash=va.content_hash or "", duration=18.0, peak_db=-6.0, rms_db=-50.0, clipped_samples=0)
    out = run_checker(AudioChecker(), qc_ctx(p))  # no ffmpeg service at all
    assert any("stored voice analysis" in n for n in out.notes)
    q = find(out, "audio.voice_quiet")
    assert len(q) == 1 and q[0].confidence == 60.0


def test_without_ffmpeg_and_without_analysis_it_reports_what_it_can(tmp_path):
    p, _ = audio_project(tmp_path)
    p.timeline.get_track("track_a1").muted = True
    out = run_checker(AudioChecker(), qc_ctx(p))
    assert "audio.voice_missing" in codes(out) and out.metrics["peak_db"] is None


def test_music_loud_wording_never_contradicts_its_own_numbers(tmp_path):
    """A stretch above the limit under a typical level below it was described as "averages -17.5 dB (limit -8 dB)": too loud, and quieter than the limit in the same sentence."""
    p, _ = audio_project(tmp_path)
    music(p, volume=1.0, amp=0.45, gain_kf=duck_kf(1.0, 0.45))
    i = find(run(p), "audio.music_loud")[0]
    peak = i.metrics["peak_relative_db"]
    assert peak > i.metrics["relative_db"] or peak == i.metrics["relative_db"]
    assert "averages" not in i.description and f"{peak:+.1f} dB" in i.description and f"{peak:+.1f} dB" in i.current_value  # what is reported as too loud is the loud part


def test_a_voice_clip_the_timeline_checker_reports_is_not_reported_again_as_cut_short(tmp_path):
    """A voice clip that does not play in full was two ERRORs: ``timeline.voice.misaligned`` and ``audio.voice_duration``. Alone, the audio checker still says it; after the timeline checker it defers."""
    from app.qc.checker_base import CheckerOutput
    from app.qc.issue_model import QCCategory
    from app.qc.timeline_checker import TimelineChecker

    p, _ = audio_project(tmp_path)
    c = p.timeline.get_track("track_a1").clips[0]
    c.duration, c.source_out = 12.0, 12.0
    assert find(run(p), "audio.voice_duration")
    ctx = qc_ctx(p)
    alone = AudioChecker().input_hash(ctx)
    tl = run_checker(TimelineChecker(), ctx)
    assert [i.timeline_item_id for i in find(tl, "timeline.voice.misaligned")] == [c.id]
    ctx.shared["timeline"] = tl
    assert find(run_checker(AudioChecker(), ctx), "audio.voice_duration") == []
    assert AudioChecker().input_hash(ctx) != alone  # the answer depends on it, so the cache key does too
    other = CheckerOutput(issues=[AudioChecker().issue("timeline.voice.misaligned", QCCategory.TIMELINE, Severity.ERROR, "x", clip=None)])
    ctx2 = qc_ctx(p)
    ctx2.shared["timeline"] = other  # reported for no clip we know of: the audio finding stays
    assert find(run_checker(AudioChecker(), ctx2), "audio.voice_duration")
