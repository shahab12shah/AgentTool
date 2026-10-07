"""Phase 7 audio analysis: voice / music / SFX / silence / ducking structure of a reference soundtrack, and ``load_audio``.

Ground truth comes from ``reference_helpers`` (synth_voice / synth_music / synth_sfx / smooth_gain) mixed in memory (the analysis tests need no FFmpeg),
so every number asserted here is known by construction:

* the *voice* is exact digital silence between phrases, so the true pauses / silences are read off the generated array itself;
* the *music* is a chord pad whose coverage, level steps, pulse and chord changes are chosen per test;
* the *SFX* are placed at known times (whoosh / hit / tick) and the shot / text / change lists are chosen to hit or miss them by known margins;
* the *duck* is a known gain (``duck_to`` = 0.3 is a 10.5 dB drop, 0.5 is 6 dB, 0.7 is 3.1 dB).

``load_audio`` is checked on real FFmpeg media (``make_audio`` + ``mux``) and on a file without an audio stream.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from app.core.serialization import to_plain
from app.reference.audio_analyzer import AudioAnalyzer, AudioResult, load_audio
from app.reference.signals import AnalysisCancelled, ReferenceAnalysisError
from app.reference.style_model import AudioProfile, music_class, sfx_class
from app.rendering.ffmpeg_service import FFmpegService
from app.tests.conftest import needs_ffmpeg
from app.tests.reference_helpers import SR, make_audio, mux, shots_video, smooth_gain, synth_music, synth_sfx, synth_voice

VOICE = [(1, 6), (7, 12), (13.5, 19), (20, 26), (27, 29)]  # 30 s, pauses 1.0 / 1.5 / 1.0 / 1.0 s (+ 1 s head, 1 s tail)
SFX_AT = [3.0, 6.5, 9.0, 16.0, 23.5]  # inside speech (3, 9, 16, 23.5) and in a pause (6.5)
DUCK_VOICE = [(4, 10), (14, 20), (24, 30), (34, 38)]  # 40 s, 4 s gaps


# ---------------------------------------------------------------------------------------------- helpers
def mix(seconds: float, voice=None, music=None, sfx=None, kind: str = "whoosh", duck_to: float | None = None, pulse: float = 0.0, sfx_amp: float = 0.5,
        sr: int = SR, voice_amp: float = 0.35, f0: float = 130.0, seed: int = 1) -> np.ndarray:
    """Same mix as ``make_audio`` (x0.9) but in memory, with the voice / SFX level and pitch adjustable."""
    n = int(seconds * sr)
    out = np.zeros(n, dtype=np.float32)
    if voice:
        out += synth_voice(seconds, voice, f0=f0, amp=voice_amp, sr=sr, seed=seed)
    if music:
        m = synth_music(seconds, music, sr=sr, pulse=pulse)
        if duck_to is not None and voice:
            m = m * smooth_gain(n, voice, duck_to, sr=sr)
        out += m
    if sfx:
        out += synth_sfx(seconds, sfx, kind, amp=sfx_amp, sr=sr)
    return (out * 0.9).astype(np.float32)


_CACHE: dict[str, AudioResult] = {}


def analyze(key: str, samples: np.ndarray | None = None, sr: int = SR, **kw) -> AudioResult:
    """Analyse once per scenario key (the scenarios are shared by several tests)."""
    if key not in _CACHE:
        assert samples is not None, key
        _CACHE[key] = AudioAnalyzer().analyze(samples, sr, **kw)
    return _CACHE[key]


def zero_runs(x: np.ndarray, min_s: float = 0.25, sr: int = SR) -> list[tuple[float, float]]:
    """Runs of exact digital silence of at least ``min_s`` in a generated signal: the true gaps."""
    z = np.concatenate(([0], (np.abs(x) < 1e-9).astype(np.int8), [0]))
    d = np.diff(z)
    return [(a / sr, b / sr) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if (b - a) / sr >= min_s]


def matched(truth: list[float], found: list[float], tol: float) -> int:
    """How many true times have a detection within ``tol`` seconds (each detection used once)."""
    pool = list(found)
    hit = 0
    for t in truth:
        best = min(pool, key=lambda f: abs(f - t), default=None)
        if best is not None and abs(best - t) <= tol:
            pool.remove(best)
            hit += 1
    return hit


# ---------------------------------------------------------------------------------------------- contract
def test_result_contract_and_ranges():
    r = analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)]))
    assert isinstance(r, AudioResult) and isinstance(r.profile, AudioProfile)
    p = r.profile
    assert p.has_audio is True
    for name in ("voice_dominance", "music_presence", "music_ducking_strength", "music_dynamics", "sfx_intensity", "sfx_on_transitions", "sfx_on_text", "sfx_on_reveals"):
        assert 0.0 <= getattr(p, name) <= 1.0, name
    assert 0.0 <= p.silence_percentage <= 100.0
    for name in ("music_changes_per_minute", "sfx_per_minute", "silence_frequency", "average_pause_duration", "long_pause_frequency", "audio_dynamic_range",
                 "loudness_changes_per_minute"):
        assert getattr(p, name) >= 0.0, name
    assert 0.0 <= r.confidence <= 1.0
    assert p.music_behavior == music_class(p.music_presence, p.music_dynamics)
    assert p.sfx_class == sfx_class(p.sfx_per_minute)
    assert isinstance(r.silences, list) and isinstance(r.sfx_times, list) and isinstance(r.music_change_times, list) and isinstance(r.notes, list)


def test_series_is_two_per_second_and_normalised():
    r = analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)]))
    assert abs(len(r.series) - 60) <= 1
    times = [row[0] for row in r.series]
    assert all(len(row) == 4 for row in r.series)
    assert all(b > a for a, b in zip(times, times[1:]))
    assert abs((times[1] - times[0]) - 0.5) < 1e-6 and 0.0 <= times[0] <= 0.5 and times[-1] <= 30.0
    for _, loud, voice, music in r.series:
        assert 0.0 <= loud <= 1.0 and 0.0 <= voice <= 1.0 and 0.0 <= music <= 1.0
    inside = [v for t, _, v, _ in r.series if any(a + 0.6 <= t <= b - 0.6 for a, b in VOICE)]
    outside = [v for t, _, v, _ in r.series if 6.3 <= t <= 6.7 or 19.3 <= t <= 19.7 or 12.3 <= t <= 13.2]
    assert np.mean(inside) > 0.8 and np.mean(outside) < 0.2  # the voice curve follows the phrases
    assert np.mean([m for _, _, _, m in r.series]) > 0.8  # and the music curve the bed


def test_only_aggregates_leave_the_module():
    """Golden rule: numbers and event times only, never samples or per-sample arrays."""
    r = analyze("voice_music_sfx", mix(30, voice=VOICE, music=[(0, 30, 0.15)], sfx=SFX_AT))
    assert not any(isinstance(v, np.ndarray) for v in vars(r).values())
    assert len(r.series) < 3 * 30 and len(r.silences) < 30 and len(r.sfx_times) < 30
    plain = to_plain(r.profile)
    assert all(isinstance(v, (bool, int, float, str)) for v in plain.values())
    assert set(plain) == set(AudioProfile.__dataclass_fields__)


def test_deterministic():
    x = mix(30, voice=VOICE, music=[(0, 30, 0.15)], sfx=SFX_AT, kind="hit")
    a, b = AudioAnalyzer().analyze(x, SR), AudioAnalyzer().analyze(x.copy(), SR)
    assert to_plain(a.profile) == to_plain(b.profile) and a.sfx_times == b.sfx_times and a.series == b.series and a.confidence == b.confidence


# ---------------------------------------------------------------------------------------------- degenerate input
def test_empty_and_tiny_input_never_raises():
    for x in (np.zeros(0, np.float32), np.zeros(100, np.float32), np.ones(5000, np.float32) * 0.1, np.array([], dtype=np.int16)):
        r = AudioAnalyzer().analyze(x, SR)
        assert r.confidence == 0.0 and r.notes
        assert r.profile.sfx_per_minute == 0.0 and r.profile.music_presence == 0.0 and r.profile.voice_dominance == 0.0
        assert r.silences == [] and r.sfx_times == [] and r.series == []
    assert AudioAnalyzer().analyze(np.zeros(0, np.float32), SR).profile.has_audio is False
    r = AudioAnalyzer().analyze(np.zeros(10, np.float32), 0)
    assert r.confidence == 0.0 and r.profile.has_audio is False


def test_digital_silence_is_all_silence_with_low_confidence():
    r = AudioAnalyzer().analyze(np.zeros(20 * SR, np.float32), SR)
    p = r.profile
    assert p.has_audio is True and p.silence_percentage >= 99.0 and r.silences and r.silences[0][0] == 0.0 and r.silences[0][1] >= 19.5
    assert p.voice_dominance == 0.0 and p.music_presence == 0.0 and p.sfx_per_minute == 0.0 and p.music_behavior == "None" and p.sfx_class == "None"
    assert p.music_ducking_strength == 0.0 and p.average_pause_duration == 0.0 and p.audio_dynamic_range == 0.0
    assert r.confidence <= 0.3 and any("silent" in n.lower() for n in r.notes)


def test_almost_inaudible_audio_is_treated_as_silent():
    r = AudioAnalyzer().analyze(mix(30, voice=VOICE) * 0.0005, SR)
    assert r.profile.silence_percentage >= 90.0 and r.confidence <= 0.3


def test_constant_signal_and_white_noise_are_not_invented_into_structure():
    for key, x in (("dc", np.full(20 * SR, 0.1, np.float32)), ("noise", np.random.default_rng(0).normal(0, 0.05, 20 * SR).astype(np.float32))):
        r = analyze(key, x)
        p = r.profile
        assert p.voice_dominance == 0.0 and p.music_presence == 0.0 and p.sfx_per_minute == 0.0 and p.music_changes_per_minute == 0.0, key
        assert p.music_behavior == "None" and p.sfx_class == "None", key
        assert r.confidence < 0.6 and r.notes, key  # something is audible but there is no speech/music/effect structure to report


def test_odd_array_shapes_and_dtypes_give_the_same_answer():
    x = mix(30, voice=VOICE, music=[(0, 30, 0.15)])
    ref = AudioAnalyzer().analyze(x, SR).profile
    variants = {
        "stereo (2,N)": np.stack([x, 0.5 * x]),
        "stereo (N,2)": np.stack([x, 0.5 * x]).T,
        "int16": (x * 32767).astype(np.int16),
        "float64": x.astype(np.float64),
    }
    bad = x.copy()
    bad[1000], bad[2000], bad[3000] = np.nan, np.inf, -np.inf
    variants["nan/inf"] = bad
    for name, v in variants.items():
        p = AudioAnalyzer().analyze(v, SR).profile
        assert abs(p.voice_dominance - ref.voice_dominance) < 0.03, name
        assert abs(p.music_presence - ref.music_presence) < 0.03 and p.music_behavior == ref.music_behavior, name


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
def test_other_sample_rates(sr):
    x = mix(30, voice=VOICE, music=[(0, 30, 0.15)], sfx=[3.0, 9.0, 16.0, 23.0], sr=sr)
    r = AudioAnalyzer().analyze(x, sr)
    assert r.profile.voice_dominance > 0.65 and r.profile.music_presence > 0.95
    assert len(r.sfx_times) == 4 and matched([3.0, 9.0, 16.0, 23.0], r.sfx_times, 0.2) == 4


# ---------------------------------------------------------------------------------------------- voice
@pytest.mark.parametrize("f0,amp,seed", [(130, 0.35, 1), (200, 0.2, 2), (100, 0.6, 3), (240, 0.1, 4)])
def test_voice_only(f0, amp, seed):
    r = analyze(f"voice_{f0}", mix(30, voice=VOICE, voice_amp=amp, f0=f0, seed=seed))
    p = r.profile
    assert p.voice_dominance >= 0.9
    assert p.music_presence <= 0.02 and p.music_behavior == "None" and p.music_changes_per_minute == 0.0 and r.music_change_times == []
    assert p.sfx_per_minute == 0.0 and p.sfx_class == "None" and r.sfx_times == []  # a voice's own consonants / syllables are not effects
    assert p.music_ducking_strength == 0.0
    assert r.confidence >= 0.75


def test_speech_share_follows_the_amount_of_talking():
    """voice_dominance is the share of the audio time where speech dominates: a mostly talking file vs. a mostly empty one."""
    talk = analyze("voice_130", mix(30, voice=VOICE))
    sparse = analyze("voice_sparse", mix(30, voice=[(2, 5), (20, 23)]))
    # the generated voice is silent between phrases; speech is dominant for (nearly) all non-silent time in both, silence is excluded from the denominator
    assert talk.profile.voice_dominance >= 0.9 and sparse.profile.voice_dominance >= 0.85
    assert sparse.profile.silence_percentage > 75 > talk.profile.silence_percentage


def test_voice_with_music_is_still_speech():
    r = analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)]))
    speech_time = sum(b - a for a, b in VOICE) / 30.0  # 0.78
    assert abs(r.profile.voice_dominance - speech_time) < 0.12
    assert r.confidence >= 0.75


# ---------------------------------------------------------------------------------------------- music
@pytest.mark.parametrize("level", [0.03, 0.15])
def test_voice_with_continuous_music(level):
    r = analyze(f"voice_music_{level}", mix(30, voice=VOICE, music=[(0, 30, level)]))
    p = r.profile
    assert p.music_presence >= 0.95
    assert p.music_behavior == "Continuous" and p.music_dynamics < 0.3
    assert p.music_changes_per_minute == 0.0 and p.sfx_per_minute == 0.0
    assert p.music_ducking_strength < 0.2  # same bed under and between the phrases: no duck
    assert r.confidence >= 0.7


def test_music_coverage_is_measured_and_its_start_and_stop_are_changes():
    r = analyze("music_10_25", mix(30, voice=VOICE, music=[(10, 25, 0.15)]))
    assert abs(r.profile.music_presence - 0.5) < 0.08
    assert len(r.music_change_times) == 2 and abs(r.music_change_times[0] - 10.0) < 1.0 and abs(r.music_change_times[1] - 25.0) < 1.0
    assert r.profile.music_behavior == "Continuous" and r.profile.music_dynamics < 0.3  # a flat bed that starts and stops is not 'dynamic'
    assert abs(r.profile.music_changes_per_minute - 4.0) < 0.6
    two = analyze("music_two", mix(30, voice=VOICE, music=[(0, 12, 0.15), (20, 30, 0.15)]))
    assert abs(two.profile.music_presence - 22 / 30) < 0.08


def test_music_that_covers_little_is_minimal():
    r = analyze("music_minimal", mix(30, voice=VOICE, music=[(12, 20, 0.15)]))
    assert 0.18 < r.profile.music_presence < 0.34 and r.profile.music_behavior == "Minimal"
    short = analyze("music_blip", mix(30, voice=VOICE, music=[(10, 11.2, 0.15)]))
    assert short.profile.music_presence < 0.1 and short.profile.music_behavior == "None"  # a tone blip is not a bed


def test_music_only_soundtrack():
    r = analyze("music_only", mix(30, music=[(0, 30, 0.3)]))
    p = r.profile
    assert p.voice_dominance == 0.0 and p.music_presence >= 0.95 and p.music_behavior == "Continuous"
    assert p.sfx_per_minute == 0.0 and p.silence_percentage == 0.0 and p.average_pause_duration == 0.0 and p.long_pause_frequency == 0.0
    assert p.music_ducking_strength == 0.0 and r.confidence >= 0.75


def test_flat_pad_vs_pulse_vs_level_steps_order_the_dynamics():
    flat = analyze("pad_flat", mix(40, music=[(0, 40, 0.3)]))
    pulse = analyze("pad_pulse", mix(40, music=[(0, 40, 0.3)], pulse=2.0))
    steps = analyze("pad_steps", mix(40, music=[(0, 10, 0.1), (10, 20, 0.5), (20, 30, 0.1), (30, 40, 0.5)]))
    assert flat.profile.music_behavior == "Continuous" and flat.profile.music_dynamics < 0.3
    assert pulse.profile.music_dynamics > flat.profile.music_dynamics + 0.08  # a rhythmic pulse is more intense than a flat pad ...
    assert pulse.profile.music_behavior in ("Dynamic", "Dramatic") and pulse.sfx_times == []  # ... and its beats are rhythm, not effects
    assert steps.profile.music_behavior == "Dramatic" and steps.profile.music_dynamics >= 0.65  # 14 dB level steps
    assert steps.profile.music_dynamics > pulse.profile.music_dynamics > flat.profile.music_dynamics


def test_level_steps_are_music_changes_at_the_right_times():
    r = analyze("pad_steps", mix(40, music=[(0, 10, 0.1), (10, 20, 0.5), (20, 30, 0.1), (30, 40, 0.5)]))
    assert matched([10.0, 20.0, 30.0], r.music_change_times, 1.5) == 3 and len(r.music_change_times) == 3
    assert abs(r.profile.music_changes_per_minute - 4.5) < 0.6
    flat = analyze("pad_flat", mix(40, music=[(0, 40, 0.3)]))
    assert flat.music_change_times == [] and flat.profile.music_changes_per_minute == 0.0


def test_a_new_chord_is_a_texture_change_at_constant_level():
    a = synth_music(40, [(0, 40, 0.3)])
    b = synth_music(40, [(0, 40, 0.3)], chord=(220.0, 277.2, 329.6, 440.0))
    x = (np.concatenate([a[: 20 * SR], b[20 * SR:]]) * 0.9).astype(np.float32)
    r = analyze("chord_change", x)
    assert len(r.music_change_times) == 1 and abs(r.music_change_times[0] - 20.0) < 1.5
    assert r.profile.music_presence >= 0.95 and r.profile.music_behavior == "Continuous"


# ---------------------------------------------------------------------------------------------- ducking
def _duck(duck_to):
    return analyze(f"duck_{duck_to}", mix(40, voice=DUCK_VOICE, music=[(0, 40, 0.25)], duck_to=duck_to))


def test_ducked_music_vs_unducked():
    ducked, flat = _duck(0.3), _duck(None)
    assert ducked.profile.music_presence >= 0.95 and flat.profile.music_presence >= 0.95
    assert 0.72 <= ducked.profile.music_ducking_strength <= 1.0  # a 10.5 dB drop = 0.875 on the 0..12 dB scale
    assert flat.profile.music_ducking_strength <= 0.15
    assert ducked.profile.music_ducking_strength - flat.profile.music_ducking_strength > 0.6


def test_ducking_strength_follows_the_depth_of_the_duck():
    s = {d: _duck(d).profile.music_ducking_strength for d in (None, 0.7, 0.5, 0.3, 0.1)}
    assert s[None] < s[0.7] < s[0.5] < s[0.3] <= s[0.1]
    assert 0.1 <= s[0.7] <= 0.45  # 3.1 dB
    assert 0.35 <= s[0.5] <= 0.65  # 6 dB
    assert s[0.1] >= 0.95  # 20 dB: full duck


def test_a_duck_is_mixing_not_a_change_of_the_music():
    for d in (0.3, 0.1):
        r = _duck(d)
        assert r.music_change_times == [] and r.profile.music_changes_per_minute == 0.0 and r.profile.music_presence >= 0.95


def test_ducking_is_not_reported_without_enough_material():
    # music only while someone talks (never in the gaps): nothing to compare the level against
    only_under = analyze("duck_only_under", mix(40, voice=DUCK_VOICE, music=[(4, 10, 0.2), (14, 20, 0.2)]))
    assert only_under.profile.music_ducking_strength == 0.0 and any("ducking" in n.lower() for n in only_under.notes)
    # talking almost without pause: the gaps are too short to see the bed come back up
    dense = analyze("duck_dense", mix(30, voice=[(1, 8), (8.4, 15), (15.4, 22), (22.4, 29)], music=[(0, 30, 0.2)], duck_to=0.3))
    assert dense.profile.music_ducking_strength == 0.0 and any("ducking" in n.lower() for n in dense.notes)
    assert dense.confidence < analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)])).confidence
    # no music at all / no speech at all: nothing to duck
    assert analyze("voice_130", mix(30, voice=VOICE)).profile.music_ducking_strength == 0.0
    assert analyze("music_only", mix(30, music=[(0, 30, 0.3)])).profile.music_ducking_strength == 0.0


# ---------------------------------------------------------------------------------------------- SFX
@pytest.mark.parametrize("kind", ["whoosh", "hit", "tick"])
@pytest.mark.parametrize("with_music", [False, True])
def test_sfx_at_known_times(kind, with_music):
    x = mix(30, voice=VOICE, music=[(0, 30, 0.15)] if with_music else None, sfx=SFX_AT, kind=kind)
    r = analyze(f"sfx_{kind}_{with_music}", x)
    assert abs(len(r.sfx_times) - len(SFX_AT)) <= 1
    assert matched(SFX_AT, r.sfx_times, 0.3) >= len(SFX_AT) - 1
    assert abs(r.profile.sfx_per_minute - 10.0) <= 2.1  # 5 effects in half a minute
    assert r.profile.sfx_class == "Heavy" and r.profile.sfx_intensity >= 0.5
    assert r.profile.voice_dominance > 0.6  # an effect never turns the narration into 'not speech'
    assert (r.profile.music_presence >= 0.9) == with_music


@pytest.mark.parametrize("count,expected_class", [(0, "None"), (2, "Subtle"), (5, "Moderate"), (12, "Heavy")])
def test_sfx_rate_and_class(count, expected_class):
    voice = [(1, 10), (12, 25), (27, 40), (42, 58)]
    times = {0: [], 2: [5.0, 35.0], 5: [5.0, 16.0, 28.0, 39.0, 50.0], 12: [4.0, 8.0, 12.0, 17.0, 21.0, 26.0, 31.0, 35.0, 40.0, 45.0, 50.0, 55.0]}[count]
    r = analyze(f"sfx_rate_{count}", mix(60, voice=voice, sfx=times))
    assert abs(len(r.sfx_times) - count) <= 1
    assert r.profile.sfx_class == expected_class or (count == 12 and r.profile.sfx_class in ("Moderate", "Heavy"))
    assert abs(r.profile.sfx_per_minute - count) <= 1.01


def test_sfx_only_soundtrack():
    times = [3.0, 9.0, 16.0, 23.0]
    r = analyze("sfx_only", mix(30, sfx=times))
    assert matched(times, r.sfx_times, 0.2) == 4 and len(r.sfx_times) == 4
    p = r.profile
    assert p.voice_dominance == 0.0 and p.music_presence == 0.0 and p.silence_percentage > 85.0


def test_dense_effects_under_speech_and_music_are_all_found():
    times = [2.0, 4.5, 6.5, 8.0, 10.0, 13.0, 15.0, 17.0, 19.5, 22.0, 24.0, 26.5]
    r = analyze("sfx_dense", mix(30, voice=VOICE, music=[(0, 30, 0.15)], sfx=times))
    assert len(r.sfx_times) >= 10 and matched(times, r.sfx_times, 0.3) >= 10
    assert r.profile.sfx_class == "Heavy" and r.profile.sfx_per_minute >= 20


def test_effects_within_0p3_s_merge_into_one_event():
    r = analyze("sfx_merge", mix(30, voice=VOICE, sfx=[3.0, 3.2, 23.5]))
    assert len(r.sfx_times) == 2 and abs(r.sfx_times[0] - 3.0) < 0.3 and abs(r.sfx_times[1] - 23.5) < 0.3


def test_a_rhythmic_pulse_and_plain_speech_are_not_effects():
    pulse = analyze("pad_pulse", mix(40, music=[(0, 40, 0.3)], pulse=2.0))
    voice_pulse = analyze("voice_pulse", mix(40, voice=[(2, 18), (22, 38)], music=[(0, 40, 0.2)], pulse=2.0))
    assert pulse.sfx_times == [] and voice_pulse.sfx_times == []
    assert pulse.profile.sfx_class == "None" and voice_pulse.profile.sfx_class == "None"


def test_effect_intensity_follows_its_loudness_relative_to_speech():
    loud = analyze("sfx_amp_0.6", mix(30, voice=VOICE, sfx=[3.0, 9.0, 16.0, 23.5], sfx_amp=0.6))
    mid = analyze("sfx_amp_0.15", mix(30, voice=VOICE, sfx=[3.0, 9.0, 16.0, 23.5], sfx_amp=0.15))
    assert len(loud.sfx_times) == 4 and len(mid.sfx_times) == 4
    assert loud.profile.sfx_intensity >= 0.9 and mid.profile.sfx_intensity <= 0.7
    assert loud.profile.sfx_intensity - mid.profile.sfx_intensity > 0.3
    # a whisper-quiet effect under the voice is simply not heard as an effect
    assert analyze("sfx_amp_0.05", mix(30, voice=VOICE, sfx=[3.0, 9.0, 16.0, 23.5], sfx_amp=0.05)).sfx_times == []


def test_sfx_alignment_with_shots_text_and_reveals():
    times = [3.0, 9.0, 16.0, 23.5]
    x = mix(30, voice=VOICE, music=[(0, 30, 0.15)], sfx=times)
    base = analyze("align_none", x)
    assert len(base.sfx_times) == 4
    assert base.profile.sfx_on_transitions == base.profile.sfx_on_text == base.profile.sfx_on_reveals == 0.0  # lists not given
    r = AudioAnalyzer().analyze(x, SR, shot_times=[3.0, 16.0], text_times=[8.8, 23.0], change_times=[2.8, 9.0, 16.0, 23.4])
    assert r.profile.sfx_on_transitions == pytest.approx(0.5, abs=1e-6)  # the effects at 3 and 16 sit on a cut
    assert r.profile.sfx_on_text == pytest.approx(0.25, abs=1e-6)  # 8.8 -> 9.03 is within 0.4 s after; 23.0 -> 23.53 is not
    assert r.profile.sfx_on_reveals == pytest.approx(1.0, abs=1e-6)
    late = AudioAnalyzer().analyze(x, SR, shot_times=[2.0, 8.0, 15.0, 22.5], text_times=[1.0], change_times=[0.5])  # every effect 1 s after the mark
    assert late.profile.sfx_on_transitions == late.profile.sfx_on_text == late.profile.sfx_on_reveals == 0.0
    early = AudioAnalyzer().analyze(x, SR, shot_times=[3.4, 9.4, 16.4, 23.9])  # the effects come 0.4 s BEFORE the cut: not 'on' it
    assert early.profile.sfx_on_transitions == 0.0
    assert AudioAnalyzer().analyze(x, SR, shot_times=[]).profile.sfx_on_transitions == 0.0
    assert AudioAnalyzer().analyze(mix(30, voice=VOICE), SR, shot_times=[3.0, 9.0]).profile.sfx_on_transitions == 0.0  # no effects at all


# ---------------------------------------------------------------------------------------------- silence and pauses
PAUSE_VOICE = [(0.0, 4.0), (4.5, 9.0), (12.0, 17.0), (17.3, 22.0), (24.0, 28.0)]  # pauses ~0.5, 3.0, 0.3, 2.0 s


def test_silence_statistics_match_the_generated_gaps():
    v = synth_voice(30, PAUSE_VOICE)
    truth = zero_runs(v)  # every exact-zero run >= 0.25 s, including the 2 s tail
    between = [(a, b) for a, b in truth if 0.1 < a and b < 29.9]
    r = analyze("pauses", v)
    p = r.profile
    # silence list: same gaps, within one analysis hop at each edge
    assert len(r.silences) == len(truth)
    for (ta, tb), (fa, fb) in zip(truth, r.silences):
        assert abs(ta - fa) < 0.08 and abs(tb - fb) < 0.08
    assert abs(p.silence_percentage - 100.0 * sum(b - a for a, b in truth) / 30.0) < 2.5
    assert abs(p.silence_frequency - len(truth) / 0.5) < 0.1
    # pauses between speech: mean of the generated gaps, and the count of those >= 1 s per minute
    mean_gap = float(np.mean([b - a for a, b in between]))
    assert len(between) == 4 and abs(p.average_pause_duration - mean_gap) < 0.12
    assert abs(p.long_pause_frequency - 2 / 0.5) < 0.1  # the 3 s and 2 s pauses (the 0.5 s and 0.3 s ones are short)
    assert p.voice_dominance >= 0.9 and r.confidence >= 0.75


def test_pauses_filled_with_music_are_pauses_but_not_silence():
    v = synth_voice(30, PAUSE_VOICE)
    between = [(a, b) for a, b in zero_runs(v) if 0.1 < a and b < 29.9]
    r = analyze("pauses_music", ((v + synth_music(30, [(0, 30, 0.12)])) * 0.9).astype(np.float32))
    assert r.profile.silence_percentage < 3.0 and r.silences == []
    assert abs(r.profile.average_pause_duration - float(np.mean([b - a for a, b in between]))) < 0.15
    assert abs(r.profile.long_pause_frequency - 4.0) < 0.1
    assert r.profile.music_presence >= 0.95


def test_pause_threshold_between_speech_runs():
    v = synth_voice(30, [(1, 7), (7.05, 13), (13.4, 19), (19.55, 25), (25.05, 29)])
    gaps = [(a, b) for a, b in zero_runs(v, 0.05) if 0.5 < a and b < 29.5]  # the generated gaps: two ~0.1 s (breaths inside a phrase), two 0.5-0.6 s pauses
    pauses = [b - a for a, b in gaps if b - a >= 0.4]
    assert len(pauses) == 2 and all(b - a < 0.15 for a, b in gaps if b - a < 0.4)
    short = analyze("pauses_short", v)
    assert len(short.pauses) == 2  # the short gaps are inside the speech run, only the real pauses count
    assert short.profile.long_pause_frequency == 0.0
    assert abs(short.profile.average_pause_duration - float(np.mean(pauses))) < 0.1
    none = analyze("pauses_none", synth_voice(30, [(1, 29)]))
    assert none.profile.average_pause_duration == 0.0 and none.profile.long_pause_frequency == 0.0
def test_silence_threshold_adapts_to_the_files_own_level():
    v = synth_voice(30, VOICE)
    truth = zero_runs(v)
    for gain, noise in ((1.0, 0.0), (0.05, 0.0), (1.0, 0.003), (0.05, 0.0002)):
        x = (v * gain + np.random.default_rng(3).normal(0, noise, len(v))).astype(np.float32)
        r = AudioAnalyzer().analyze(x, SR)
        assert abs(r.profile.silence_percentage - 100.0 * sum(b - a for a, b in truth) / 30.0) < 4.0, (gain, noise)
        assert abs(r.profile.average_pause_duration - 1.125) < 0.15, (gain, noise)


def test_a_loud_noise_bed_is_not_silence_and_not_music():
    v = synth_voice(30, VOICE)
    r = analyze("noisy_room", (v + np.random.default_rng(0).normal(0, 0.02, len(v))).astype(np.float32))
    assert r.profile.silence_percentage < 3.0 and r.profile.music_presence == 0.0 and r.profile.sfx_per_minute == 0.0
    assert r.profile.voice_dominance > 0.7 and abs(r.profile.average_pause_duration - 1.125) < 0.2  # pauses are still found between the phrases
    assert any("bed" in n.lower() or "noise" in n.lower() or "hiss" in n.lower() for n in r.notes) or r.confidence <= 0.92


def test_silence_only_soundtrack_reports_zero_rates():
    r = analyze("digital_silence", np.zeros(20 * SR, np.float32))
    assert r.profile.silence_frequency == pytest.approx(3.0, abs=0.2) and r.profile.long_pause_frequency == 0.0


# ---------------------------------------------------------------------------------------------- loudness
def test_loudness_dynamic_range_and_changes():
    flat = analyze("pad_flat", mix(40, music=[(0, 40, 0.3)]))
    steps = analyze("pad_steps", mix(40, music=[(0, 10, 0.1), (10, 20, 0.5), (20, 30, 0.1), (30, 40, 0.5)]))
    assert flat.profile.audio_dynamic_range < 3.0 and flat.profile.loudness_changes_per_minute == 0.0
    assert steps.profile.audio_dynamic_range >= 9.0  # 14 dB steps (the 10th..95th percentile of 3 s loudness)
    assert abs(steps.profile.loudness_changes_per_minute - 4.5) <= 1.6
    assert steps.profile.audio_dynamic_range > flat.profile.audio_dynamic_range + 6.0


def test_a_steady_voice_does_not_change_loudness():
    r = analyze("voice_130", mix(30, voice=VOICE))
    assert r.profile.loudness_changes_per_minute == 0.0
    # narration whose second half is 12 dB louder does
    v = synth_voice(40, [(1, 19), (21, 39)])
    v[20 * SR:] *= 4.0
    r = analyze("voice_step", (v * 0.9 / max(1.0, float(np.abs(v).max()) / 0.9)).astype(np.float32))
    assert r.profile.loudness_changes_per_minute >= 1.0 and r.profile.audio_dynamic_range >= 6.0


# ---------------------------------------------------------------------------------------------- confidence
def test_confidence_is_lower_for_short_clipped_and_ambiguous_audio():
    full = analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)]))
    short = analyze("short_8s", mix(8, voice=[(0.5, 7.5)], music=[(0, 8, 0.15)]))
    tiny = analyze("short_3s", mix(3, voice=[(0.3, 2.7)]))
    assert short.confidence < full.confidence - 0.2 and tiny.confidence < short.confidence
    assert any("short" in n.lower() for n in short.notes)
    assert short.profile.voice_dominance > 0.6  # still measured, just flagged
    clipped = analyze("clipped", np.clip(mix(30, voice=VOICE, music=[(0, 30, 0.15)]) * 12.0, -1.0, 1.0))
    assert clipped.confidence < full.confidence - 0.1 and any("clip" in n.lower() for n in clipped.notes)
    quiet = analyze("quiet", mix(30, voice=VOICE, music=[(0, 30, 0.15)]) * 0.004)
    assert quiet.confidence < full.confidence - 0.2


def test_confidence_drops_near_a_classification_threshold():
    far = analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)]))
    near = analyze("sfx_4", mix(60, voice=[(1, 58)], sfx=[10.0, 25.0, 40.0], music=None))  # 3 effects per minute: exactly the Subtle / Moderate edge
    assert near.confidence < far.confidence and any("threshold" in n.lower() for n in near.notes)


def test_speech_that_cannot_be_separated_from_loud_music_is_flagged_or_clean():
    """Honesty check: when the pad is much louder than the voice the answer may be wrong, but the confidence must not exceed that of the clean case."""
    x = mix(30, voice=VOICE, music=[(0, 30, 0.5)], voice_amp=0.1)
    r = analyze("loud_music", x)
    clean = analyze("voice_music", mix(30, voice=VOICE, music=[(0, 30, 0.15)]))
    assert r.profile.music_presence >= 0.9
    assert r.confidence <= clean.confidence


# ---------------------------------------------------------------------------------------------- load_audio (FFmpeg)
@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio_media")
    wav = make_audio(d / "mix.wav", 20, voice=[(1, 8), (10, 18)], music=[(0, 20, 0.15)], sfx=[4.0, 14.0], sfx_kind="hit")
    video = shots_video(d / "v.mp4", [5, 5, 5, 5], fps=12)
    return {"dir": d, "wav": wav, "silent_video": video, "muxed": mux(video, wav, d / "muxed.mp4")}


@needs_ffmpeg
def test_load_audio_from_muxed_media(media):
    x = load_audio(FFmpegService(), media["muxed"])
    assert x is not None and x.ndim == 1 and x.dtype == np.float32
    assert abs(len(x) / 16000 - 20.0) < 0.2
    assert float(np.abs(x).max()) <= 1.0 and float(np.abs(x).max()) > 0.1 and np.isfinite(x).all()
    # the decoded audio carries the same structure as the generated one
    r = AudioAnalyzer().analyze(x, 16000)
    direct = AudioAnalyzer().analyze(mix(20, voice=[(1, 8), (10, 18)], music=[(0, 20, 0.15)], sfx=[4.0, 14.0], kind="hit"), SR)
    assert abs(r.profile.voice_dominance - direct.profile.voice_dominance) < 0.1
    assert r.profile.music_presence >= 0.9 and abs(r.profile.music_presence - direct.profile.music_presence) < 0.1
    assert matched([4.0, 14.0], r.sfx_times, 0.35) == 2 and len(r.sfx_times) == 2  # AAC does not hide the effects


@needs_ffmpeg
def test_load_audio_sample_rate_and_plain_wav(media):
    x16 = load_audio(FFmpegService(), media["wav"])
    x8 = load_audio(FFmpegService(), media["wav"], sr=8000)
    assert x16 is not None and x8 is not None
    assert abs(len(x16) - 20 * 16000) < 400 and abs(len(x8) - 20 * 8000) < 200
    assert float(np.abs(x16).max()) <= 1.0


@needs_ffmpeg
def test_load_audio_returns_none_for_a_file_without_audio(media):
    assert load_audio(FFmpegService(), media["silent_video"]) is None


@needs_ffmpeg
def test_load_audio_errors(media):
    ff = FFmpegService()
    with pytest.raises(ReferenceAnalysisError):
        load_audio(ff, media["dir"] / "missing.mp4")
    garbage = media["dir"] / "garbage.mp4"
    garbage.write_bytes(b"this is not a media file" * 100)
    with pytest.raises(ReferenceAnalysisError):
        load_audio(ff, garbage)


@needs_ffmpeg
def test_load_audio_can_be_cancelled(media):
    ev = threading.Event()
    ev.set()
    with pytest.raises(AnalysisCancelled):
        load_audio(FFmpegService(), media["muxed"], cancel=ev)


@needs_ffmpeg
def test_load_audio_digital_silence_is_audio_not_missing_audio(media):
    d: Path = media["dir"]
    wav = make_audio(d / "silent.wav", 6)  # no voice, no music: a stream of zeros
    x = load_audio(FFmpegService(), wav)
    assert x is not None and len(x) > 5 * 16000 and float(np.abs(x).max()) == 0.0
    r = AudioAnalyzer().analyze(x, 16000)
    assert r.profile.has_audio is True and r.profile.silence_percentage >= 99.0
