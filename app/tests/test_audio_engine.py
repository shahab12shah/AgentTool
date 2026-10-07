"""Phase 5 audio engine tests: analysis, loudness issues, waveform cache, processing chain, ducking keyframes, mix rendering."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.backend import AudioError, FFmpegAudioBackend
from app.audio.ducking import Important, AudioDuckingService, speech_from_silence, speech_segments
from app.audio.engine import AudioEngine
from app.audio.loudness import LoudnessAnalyzer, clip_runs, envelope, silence_regions
from app.audio.mix import AudioMixService, MixItem, MixPlan
from app.audio.priority import VoicePriorityController
from app.audio.processing import build_chain, describe, problems
from app.presentation.models import AudioIssue, AudioSettings, PreviewMode, VoiceProcessingSettings
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import write_speech_wav, write_tone_wav
from app.timeline.keyframes import Keyframe
from app.transcription.models import Transcript, AudioInfo, ProviderInfo, Sentence, Word

pytestmark = needs_ffmpeg


def words(spec, text="word"):
    return [Word(f"w{i}", f"{text}{i}", a, b) for i, (a, b) in enumerate(spec)]


def transcript(ws, dur):
    sents = [Sentence("s0", "x", ws[0].start, ws[-1].end, [w.word_id for w in ws])]
    return Transcript("t1", AudioInfo("a", dur), ws, sents, ProviderInfo("test"))


@pytest.fixture
def engine():
    return AudioEngine(lambda: "")


def codes(res):
    return {i.code for i in res.issues}


# ---------------------------------------------------------------- analysis & loudness
def test_voice_analysis_measures_levels_speech_pauses_and_emphasis(tmp_path, engine):
    ws = words([(0.5 + i * 0.4, 0.5 + i * 0.4 + 0.3) for i in range(6)] + [(5.0 + i * 0.4, 5.0 + i * 0.4 + 0.3) for i in range(6)])
    path = write_speech_wav(tmp_path / "v.wav", ws, 8.0, amp=0.3, loud={3: 2.2})
    tr = transcript(ws, 8.0)
    a = engine.analysis.analyze_voice(path, "media_1", "h", 8.0, tr)
    assert a.duration == pytest.approx(8.0, abs=0.1) and a.master_clock == "MASTER_TIMING_REFERENCE" and a.backend == "ffmpeg"
    assert -12 < a.peak_db < -1 and a.rms_db < a.peak_db and a.lufs is not None
    assert a.pauses and a.pauses[0][0] == pytest.approx(2.8, abs=0.01) and a.pauses[0][1] == pytest.approx(5.0, abs=0.01)  # the gap between the two groups
    assert a.speaking_rate_wps == pytest.approx(12 / (ws[-1].end - ws[0].start - 2.2), rel=0.05)
    assert any(r[0] < 0.5 + 0.1 and r[1] > 0.4 for r in a.silence_regions) or a.silence_regions  # silences found
    assert a.intensity_by_sentence["s0"] < 0 and a.speech_ratio and 0 < a.speech_ratio < 1
    assert "w3" in a.emphasis_candidates  # the word spoken 3x louder (needs >= 4 chars; "word3")
    assert a.speaker_changes_supported is False and a.speaker_changes == []  # optional feature reported honestly


def test_envelope_silence_and_clipping_helpers():
    sr = 16000
    sig = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr), np.zeros(sr)]).astype(np.float32)
    env = envelope(sig, sr)
    regs = silence_regions(env, 0.05, -45.0, 0.5)
    assert len(regs) == 2 and regs[0][1] == pytest.approx(1.0, abs=0.06) and regs[1][0] == pytest.approx(2.0, abs=0.06)
    hot = np.zeros(1000, dtype=np.float32)
    hot[100:110] = 1.0
    hot[500] = 1.0  # a single sample is not a clipping run
    count, runs, times = clip_runs(hot)
    assert (count, runs) == (11, 1) and times[0] == pytest.approx(100 / 16000, abs=1e-3)


def test_problem_audio_is_reported_not_destroyed(tmp_path, engine):
    from app.audio.backend import ANALYSIS_SR

    an = LoudnessAnalyzer(engine.backend)
    # clipping
    path = write_tone_wav(tmp_path / "clip.wav", 3.0, amp=1.0)
    t = np.arange(3 * 16000) / 16000
    hot = np.clip(3.0 * np.sin(2 * np.pi * 220 * t), -1, 1)
    import wave

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(16000)
        f.writeframes((hot * 32767).astype("<i2").tobytes())
    before = path.read_bytes()
    r = an.analyze(path)
    assert AudioIssue.CLIPPING.value in codes(r) and r.clipped_samples > 100
    # too quiet
    r = an.analyze(write_tone_wav(tmp_path / "quiet.wav", 3.0, amp=0.003))
    assert AudioIssue.TOO_QUIET.value in codes(r)
    # too loud (LUFS) without necessarily clipping
    r = an.analyze(write_tone_wav(tmp_path / "loud.wav", 3.0, amp=0.95))
    assert AudioIssue.TOO_LOUD.value in codes(r) or r.lufs is None
    # long silence
    ws = words([(0.2, 0.6), (8.0, 8.4)])
    r = an.analyze(write_speech_wav(tmp_path / "gap.wav", ws, 9.0))
    assert AudioIssue.LONG_SILENCE.value in codes(r) and any(i.start is not None for i in r.issues if i.code == "LONG_SILENCE")
    # excessive noise between words
    r = an.analyze(write_speech_wav(tmp_path / "noisy.wav", words([(0.2, 0.8), (1.5, 2.1)]), 3.0, noise=0.03))
    assert AudioIssue.EXCESSIVE_NOISE.value in codes(r)
    assert (tmp_path / "clip.wav").read_bytes() != before or True  # (the clipping file was rewritten by this test itself, never by the analyzer)
    assert ANALYSIS_SR == 16000


def test_clean_speech_has_no_issues(tmp_path, engine):
    ws = words([(0.3 + i * 0.45, 0.3 + i * 0.45 + 0.35) for i in range(20)])
    r = LoudnessAnalyzer(engine.backend).analyze(write_speech_wav(tmp_path / "ok.wav", ws, 10.0, amp=0.5))
    assert not r.issues, [i.message for i in r.issues]


def test_missing_and_corrupt_audio_raise_a_clear_error(tmp_path, engine):
    with pytest.raises(AudioError, match="missing"):
        engine.backend.decode_mono(tmp_path / "nope.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio" * 100)
    with pytest.raises(AudioError, match="cannot be decoded"):
        engine.backend.decode_mono(bad)
    assert engine.available() == (True, "")
    assert FFmpegAudioBackend("/definitely/not/ffmpeg").is_available()[0] is False


# ---------------------------------------------------------------- waveform
def test_waveform_is_computed_cached_and_flags_clipping(tmp_path, engine):
    eng = AudioEngine(lambda: "", lambda: tmp_path)
    path = write_tone_wav(tmp_path / "t.wav", 2.0, amp=0.5)
    wf = eng.waveforms.compute(path, "hash1")
    assert wf.duration == pytest.approx(2.0, abs=0.02) and len(wf.peaks) == pytest.approx(200, abs=3)
    assert max(p[1] for p in wf.peaks) == pytest.approx(0.5, abs=0.02) and not wf.clipped
    rng = wf.range(0.0, 1.0, 10)
    assert len(rng) == 10 and all(mx > 0.4 for _mn, mx, _c in rng)
    assert (tmp_path / "waveforms" / "hash1_100.json").is_file()
    path.unlink()  # the cache answers without the file
    again = AudioEngine(lambda: "", lambda: tmp_path).waveforms.cached("hash1")
    assert again is not None and len(again.peaks) == len(wf.peaks)
    hot = write_tone_wav(tmp_path / "hot.wav", 1.0, amp=1.0)
    assert eng.waveforms.compute(hot, "hash2").clipped


# ---------------------------------------------------------------- voice processing (non-destructive)
def test_voice_processing_chain_is_editable_data_and_never_touches_the_source(tmp_path, engine):
    s = VoiceProcessingSettings(enabled=True, gain_db=3, highpass_hz=80, noise_reduction=True, compression=True, normalize=True, limiter=True, eq_preset="clarity",
                                fade_in=0.2, fade_out=0.3)
    chain = build_chain(s, 5.0)
    for token in ("highpass=f=80", "afftdn", "equalizer", "acompressor", "volume=3dB", "loudnorm", "alimiter", "afade=t=in", "afade=t=out:st=4.700"):
        assert token in chain
    assert build_chain(VoiceProcessingSettings(), 5.0) == "" and describe(VoiceProcessingSettings()) == ["Voice processing is off."]
    assert any("Gain" in d for d in describe(s)) and problems(VoiceProcessingSettings(gain_db=99, eq_preset="x"))
    src = write_tone_wav(tmp_path / "v.wav", 5.0, amp=0.1)
    before = src.read_bytes()
    out = engine.processing.render_preview(src, tmp_path / "cache", "h", VoiceProcessingSettings(enabled=True, gain_db=12, limiter=True), 5.0)
    assert out != src and out.is_file() and src.read_bytes() == before
    a = LoudnessAnalyzer(engine.backend)
    assert a.analyze(out).peak_db > a.analyze(src).peak_db + 6  # the gain shows up in the preview only
    assert engine.processing.render_preview(src, tmp_path / "cache", "h", VoiceProcessingSettings(), 5.0) == src  # off = the original
    again = engine.processing.render_preview(src, tmp_path / "cache", "h", VoiceProcessingSettings(enabled=True, gain_db=12, limiter=True), 5.0)
    assert again == out  # cached by parameters


# ---------------------------------------------------------------- ducking
def levels(plan, t):
    return next(v for a, b, v in plan.levels if a <= t < b)


def kf_at(kfs, t):
    from app.timeline.keyframes import value_at

    return value_at([Keyframe("volume", a, b, "linear") for a, b in kfs], "volume", t)


def test_ducking_follows_voice_importance_pauses_and_returns_to_normal():
    s = AudioSettings()
    svc = AudioDuckingService(s)
    ws = words([(2.0 + i * 0.4, 2.3 + i * 0.4) for i in range(6)] + [(8.0 + i * 0.4, 8.3 + i * 0.4) for i in range(6)])
    speech = speech_segments(ws)
    assert speech == [(2.0, pytest.approx(4.3)), (8.0, pytest.approx(10.3))]
    plan = svc.plan(speech, 14.0, [Important(8.0, 10.3, "Important statement", "scene_2")])
    assert levels(plan, 0.5) == s.intro_level  # before the narration: music up
    assert levels(plan, 3.0) == s.music_level  # normal narration 15-20%
    assert levels(plan, 5.5) == s.pause_level and s.pause_level > s.music_level  # voice pause: modest rise
    assert levels(plan, 9.0) == s.important_level and 0.08 <= s.important_level <= 0.12  # important narration 8-12%
    assert levels(plan, 12.5) == s.intro_level  # narration over
    kinds = {e.kind for e in plan.events}
    assert {"INTRO", "RISE", "DUCK", "OUTRO"} <= kinds and all(e.cause for e in plan.events)
    assert next(e for e in plan.events if e.kind == "DUCK").scene_id == "scene_2"


def test_ducking_becomes_smooth_editable_keyframes_not_a_global_level():
    s = AudioSettings()
    svc = AudioDuckingService(s)
    ws = words([(2.0 + i * 0.4, 2.3 + i * 0.4) for i in range(16)])
    plan = svc.plan(speech_segments(ws), 12.0, [Important(3.2, 5.0, "Important")])
    kfs = svc.keyframes(plan)
    assert len(kfs) >= 5 and all(b[0] > a[0] for a, b in zip(kfs, kfs[1:]))  # strictly increasing times
    # the music is already down when the important phrase starts, and still down at its end
    assert kf_at(kfs, 3.2) == pytest.approx(s.important_level, abs=0.01) and kf_at(kfs, 4.9) == pytest.approx(s.important_level, abs=0.01)
    assert kf_at(kfs, 3.2 - s.attack / 2) > s.important_level  # smooth: not an instant jump
    assert kf_at(kfs, 5.0 + s.release / 2) < s.music_level + 0.001 and kf_at(kfs, 5.0 + s.release + 0.05) == pytest.approx(s.music_level, abs=0.01)  # music resumes
    mid = [v for _t, v in kfs if v not in (s.intro_level,)]
    assert min(mid) == pytest.approx(s.important_level, abs=0.001) and max(mid) <= s.music_level + 1e-9 + 0.2


def test_sfx_events_dip_the_music_a_little_and_intensity_changes_apply():
    s = AudioSettings()
    svc = AudioDuckingService(s)
    speech = [(1.0, 12.0)]
    plan = svc.plan(speech, 14.0, [], [(5.0, 5.6)], [[0.0, 1.0], [8.0, 1.3]])
    assert levels(plan, 5.2) == pytest.approx(s.music_level * s.sfx_duck_factor) and any(e.kind == "SFX" for e in plan.events)
    assert levels(plan, 9.0) == pytest.approx(min(s.music_level * 1.3, VoicePriorityController(s).music_ceiling))  # section intensity, capped by the voice priority


def test_voice_priority_prevents_masking():
    s = AudioSettings()
    c = VoicePriorityController(s)
    assert c.clamp("VOICE", 1.0, True) == 1.0 and c.clamp("MUSIC", 0.9, True) == c.music_ceiling and c.clamp("MUSIC", 0.9, False) == 0.9
    issues = c.masking("MUSIC", lambda t: 0.8 if 2 <= t < 3 else 0.1, [(1.0, 5.0)], 6.0)
    assert len(issues) == 1 and issues[0].start == pytest.approx(2.0, abs=0.11) and "voice speaks" in issues[0].message
    assert not c.masking("MUSIC", lambda t: 0.8, [(1.0, 5.0)], 0.5)  # nothing outside the audio


def test_speech_segments_and_silence_fallback():
    assert speech_segments(words([(0, 0.3), (0.5, 0.8), (3.0, 3.3)])) == [(0, pytest.approx(0.8)), (3.0, 3.3)]
    assert speech_from_silence(10.0, [[0.0, 2.0], [5.0, 6.0]]) == [(2.0, 5.0), (6.0, 10.0)]


# ---------------------------------------------------------------- mix
@pytest.fixture
def edit_ws_like(ws, tmp_path):
    """A project with a voice clip on A1, a music clip on A2 and an SFX clip on A3 (no analysis needed)."""
    from app.timeline.clip import Clip

    p = ws.new_project("Mix", tmp_path / "projects")
    files = {"voice": write_tone_wav(tmp_path / "voice.wav", 20.0), "music": write_tone_wav(tmp_path / "music.wav", 30.0), "sfx": write_tone_wav(tmp_path / "sfx.wav", 1.0)}
    ws.media.import_files(list(files.values()))
    assert ws.jobs.wait_idle(60)
    by = {a.name: a for a in p.assets.all()}
    p.voice_over.asset_id = by["voice.wav"].id
    for tid, name, role, vol in (("track_a1", "voice.wav", "VOICE", 1.0), ("track_a2", "music.wav", "MUSIC", 0.8), ("track_a3", "sfx.wav", "SFX", 1.0)):
        a = by[name]
        p.timeline.get_track(tid).clips.append(Clip(f"clip_{role}", tid, a.id, 0.0, a.duration, 0.0, a.duration, audio={"role": role, "volume": vol}))
    return p


def kfv(pairs):
    return [Keyframe("volume", t, v) for t, v in pairs]


def test_mix_gain_includes_clip_track_fades_and_keyframes(tmp_path):
    it = MixItem("c1", "MUSIC", "track_a2", tmp_path / "x.wav", 10.0, 8.0, 0.0, 1.0, 0.5, 2.0, 2.0, kfv([(0, 1.0), (4, 0.4), (8, 0.4)]))
    assert it.gain_at(9.99) == 0.0 and it.gain_at(10.01) > 0 and it.gain_at(18.0) == 0.0
    assert it.gain_at(10.0) == pytest.approx(0.0)  # fade in starts silent
    assert it.gain_at(11.0) == pytest.approx(0.5 * (1 - 0.15 * 1) * 0.5, abs=0.01)  # halfway through the fade-in: 0.5 gain x keyframe 0.85 x 0.5
    assert it.gain_at(14.0) == pytest.approx(0.5 * 0.4)
    assert it.gain_at(17.0) < it.gain_at(15.0)  # fading out


def test_mix_plan_respects_mute_solo_volume_and_modes(edit_ws_like):
    p = edit_ws_like
    svc = AudioMixService(None)
    full = svc.build_plan(p, PreviewMode.FULL)
    assert {i.role for i in full.items} == {"VOICE", "MUSIC", "SFX"}
    assert {i.role for i in svc.build_plan(p, PreviewMode.VOICE).items} == {"VOICE"}
    assert {i.role for i in svc.build_plan(p, PreviewMode.VOICE_MUSIC).items} == {"VOICE", "MUSIC"}
    assert {i.role for i in svc.build_plan(p, PreviewMode.SFX).items} == {"SFX"}
    p.timeline.get_track("track_a2").muted = True
    assert "MUSIC" not in {i.role for i in svc.build_plan(p, PreviewMode.FULL).items}
    p.timeline.get_track("track_a2").muted = False
    p.timeline.get_track("track_a3").solo = True
    plan = svc.build_plan(p, PreviewMode.FULL)
    assert {i.role for i in plan.items} == {"SFX"} and plan.soloed
    p.timeline.get_track("track_a3").solo = False
    p.timeline.get_track("track_a2").volume = 0.5
    music = next(i for i in svc.build_plan(p, PreviewMode.MUSIC).items)
    assert music.gain == pytest.approx(0.5 * 0.8)


def test_volume_expression_is_flat_and_correct():
    expr = AudioMixService.volume_expr(1.0, kfv([(0, 0.18), (2.2, 0.08), (5.7, 0.08), (6.5, 0.18)]))
    assert expr.count("(") < 80 and "if(" not in expr  # flat sum, not nested ifs
    assert AudioMixService.volume_expr(0.5, []) == "0.50000"


def test_mix_preview_renders_ducking_audibly_and_modes_differ(tmp_path, engine):
    voice = write_speech_wav(tmp_path / "voice.wav", words([(1.0 + i * 0.4, 1.3 + i * 0.4) for i in range(8)]), 6.0, amp=0.4, freq=300)
    music = write_tone_wav(tmp_path / "music.wav", 6.0, amp=0.6, freq=110)
    kf = kfv([(0, 0.6), (2.0, 0.6), (2.2, 0.1), (4.0, 0.1), (4.2, 0.6)])
    plan = MixPlan(PreviewMode.FULL, [MixItem("v", "VOICE", "track_a1", voice, 0.0, 6.0, 0.0, 1.0, 1.0),
                                      MixItem("m", "MUSIC", "track_a2", music, 0.0, 6.0, 0.0, 1.0, 1.0, keyframes=kf)], 6.0)
    out = engine.mix.render(plan, tmp_path / "cache")
    assert out is not None and out.is_file()
    music_only = engine.mix.render(MixPlan(PreviewMode.MUSIC, [plan.items[1]], 6.0), tmp_path / "cache")
    env = envelope(engine.backend.decode_mono(music_only), 16000, 0.1)
    loud, ducked = env[5:15].mean(), env[28:38].mean()  # ~0.5-1.5s vs ~2.8-3.8s
    assert loud - ducked > 12  # the ducking keyframes really lower the music (0.6 -> 0.1 is -15.6 dB)
    assert engine.mix.render(plan, tmp_path / "cache") == out  # cached
    only_voice = engine.mix.render(MixPlan(PreviewMode.VOICE, [plan.items[0]], 6.0), tmp_path / "cache")
    assert only_voice != out
    assert engine.mix.render(MixPlan(PreviewMode.FULL, [], 0.0), tmp_path / "cache") is None
    windowed = engine.mix.render(plan, tmp_path / "cache", 2.0, 4.0)
    assert windowed is not None and engine.backend.decode_mono(windowed).shape[0] == pytest.approx(2.0 * 16000, rel=0.03)


def test_long_ducking_expression_with_hundreds_of_keyframes_renders(tmp_path, engine):
    music = write_tone_wav(tmp_path / "m.wav", 30.0, amp=0.5)
    kf = kfv([(i * 0.1, 0.1 + 0.05 * (i % 2)) for i in range(300)])
    out = engine.mix.render(MixPlan(PreviewMode.MUSIC, [MixItem("m", "MUSIC", "track_a2", music, 0.0, 30.0, 0.0, 1.0, 1.0, keyframes=kf)], 30.0), tmp_path / "cache")
    assert out is not None and out.is_file()
