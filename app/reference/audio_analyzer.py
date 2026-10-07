"""AudioAnalyzer: the voice / music / SFX / silence / ducking *structure* of a reference's soundtrack, as abstract numbers (numpy only).

Only aggregate measurements and time lists (silences, SFX times, music-change times) come out. No recording, melody, fingerprint or audio sample is kept or
derived: the analysis learns *how the soundtrack is edited* (how much is speech, whether music sits underneath, how far it ducks, how often effects hit,
how long the pauses are), never what is said or played.

How it works (everything is relative to the soundtrack's own level, so it needs no per-video tuning):

* **Spectrogram.** 50 ms Hann frames every 25 ms, power spectrum up to 8 kHz. From it: frame energy (dB), a *harmonicity* score (the cepstral peak at a voice
  pitch, 80-400 Hz: voiced speech has a comb of harmonics, noise bursts / sweeps / thumps / sustained chords do not at that quefrency) and band energies.
* **Music = the stationary floor.** A sustained bed shows up as energy that is *always there*: per frequency bin, a low percentile (10 %) of the power over a
  1.5 s window ("minimum statistics") follows the bed and ignores speech, whose harmonics move and whose syllables stop. A floor made of *persistent narrow
  peaks* (>= 8 dB over their spectral surroundings) is tonal, i.e. music; a flat, structureless floor is room tone / noise and is NOT counted as music. The
  floor level is the music level, also underneath speech, so the level during speech can be compared with the level in the gaps (ducking).
* **Speech** = runs of voiced frames (harmonic, above the activity gate) that bridge gaps shorter than 0.25 s, last >= 0.3 s and contain syllable-rate
  energy peaks. Gaps between speech runs are *pauses*.
* **SFX** = short (< 1.2 s) bursts of energy ABOVE the stationary floor that are not speech: any burst in non-speech context, and (inside speech) only bursts
  that stick out of their high-band / low-band neighbourhood (noise whooshes, clicks, thumps; sibilants do not). Detections within 0.3 s merge; regular
  trains (a beat) are rhythm, not effects.
* **Silence** = frames below an adaptive floor (a few dB above the file's own quiet level, never above 25 dB under its loud level).

Confidence is honest: short audio, near-threshold classifications, speech that cannot be told from music, clipped or very quiet audio and a noise bed
all lower it. Empty, silent or too-short input never raises: it returns zeros and confidence 0 (or a low confidence for silence) with a note.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.core.exceptions import FFmpegUnavailableError
from app.reference.signals import AnalysisCancelled, ReferenceAnalysisError
from app.reference.style_model import AudioProfile, clamp, music_class, sfx_class
from app.rendering.ffmpeg_service import FFmpegService

_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}

# ---------------------------------------------------------------------------------------------- tunables
TARGET_SR = 16000  # analysis rate: anything much above is decimated first (speech/music structure lives below 8 kHz)
HOP_S = 0.025
STEP_S = 0.5  # one value per step in the result series (2 per second) and the unit of the music/ducking decisions
MAX_HZ = 8000.0
MIN_SECONDS = 1.0  # shorter audio carries no structure to measure
SHORT_SECONDS = 20.0  # below this the rates and shares are noisy: confidence is scaled down
EPS = 1e-12
DB_FLOOR = -120.0  # what digital silence reads as

FLOOR_HALF_S = 0.75  # the stationary floor looks at +-0.75 s: long enough to span syllables, short enough to follow a level step or a duck
FLOOR_Q = 0.10
GAP_GUARD_S = 0.35  # frames this close to speech are not used for the floor of a window that sits in a gap
TONAL_LO_HZ, TONAL_HI_HZ = 60.0, 5000.0
PEAK_CONTRAST_DB = 8.0  # a floor bin this far above its spectral surroundings is a tonal peak
PEAK_SURROUND_HZ = 375.0  # half-width of the surroundings the contrast is measured against
TONAL_ACTIVE = 0.50  # share of floor energy in tonal peaks that makes a window musical (a little more is asked while speech is on)
TONAL_SPEECH_EXTRA = 0.15
MUSIC_ABS_GATE_DB = -65.0  # the floor must be audible: above this and within 42 dB of the loud level
MUSIC_REL_GATE_DB = 42.0
MUSIC_MIN_RUN = 3  # steps (1.5 s): shorter "music" is a tone blip
MUSIC_FILL_GAP = 2  # steps: a dropout this short (a hit, a loud syllable burst) does not end the bed

CPP_LO_HZ, CPP_HI_HZ = 70.0, 4000.0
CPP_VOICED = 5.0  # cepstral peak / cepstral level: noise, chords and bursts sit at 1.4-3.7, voiced speech at 7-20
ACF_VOICED = 0.55  # normalised autocorrelation peak at a voice pitch (80-350 Hz): voiced speech 0.6-0.8, chords <= 0.5, noise / bursts / thumps ~0
ACF_LO_HZ, ACF_HI_HZ = 80.0, 350.0
ACTIVITY_REL_DB = 45.0  # a frame this far under the loud level (or under -72 dBFS) is not speech
BRIDGE_S = 0.25  # silent gaps shorter than this stay inside a speech run (consonants, breaths)
SPEECH_MIN_S = 0.3
SPEECH_MIN_DENSITY = 0.25
SYLLABLE_PROMINENCE_DB = 2.5
MAX_PAUSE_S = 5.0  # a longer speechless stretch is a passage without narration, not a pause
MIN_PAUSE_S = 0.25
LONG_PAUSE_S = 1.0

SILENCE_MIN_S = 0.25
SILENCE_ABS_MIN_DB = -72.0
SILENCE_BELOW_LOUD_DB = 25.0

FG_OVER = 1.5  # foreground = power above 1.5 x the stationary floor
SFX_DROP_DB = 15.0
SFX_RISE_TOT_DB = 12.0
SFX_RISE_BAND_DB = 14.0
SFX_OVER_BED_DB = 3.0  # in non-speech context an effect must also exceed the bed's floor by this much
SFX_STICKOUT_DB = 6.0  # inside speech, a band burst must exceed the neighbourhood's 90th percentile by this much
SFX_MAX_S = 1.2
SFX_MIN_S = 0.03
SFX_MERGE_S = 0.3
SFX_GATE_REL_DB = 40.0
SFX_BAND_GATE_REL_DB = 10.0  # a band burst (high / low band only) must reach within 10 dB of the loud level to be an effect rather than speech texture (a voice's own
                            # high band sits ~20 dB under its loud level, a sibilant ~12 dB, a whoosh within a few dB)
SFX_SPEECH_DILATE_S = 0.2  # speech context reaches this far past a speech run (unvoiced consonants hang off its edges)
SFX_SPEECH_COVER = 0.25  # an event that overlaps speech this much is judged by its band signature only
SFX_ALIGN_BEFORE_S, SFX_ALIGN_AFTER_S = 0.15, 0.4  # an effect is "on" a boundary when it starts up to 0.15 s before .. 0.4 s after it

DUCK_FULL_DB = 12.0  # a drop of this much (or more) = ducking strength 1
DUCK_MIN_SPEECH_S, DUCK_MIN_GAP_S = 3.0, 2.0  # music-under-speech / music-in-gaps material needed to measure a duck

MUSIC_STEP_DB = 5.0  # an abrupt level change of the bed
MUSIC_TEXTURE_DIST = 0.40  # 1 - cosine similarity of the floor spectra either side of a point: a new chord / instrumentation
MUSIC_CHANGE_MERGE_S = 2.0
DYNAMICS_SPREAD_DB = 12.0  # level spread (10th..95th percentile) of the bed that counts as full variation
LOUDNESS_CHANGE_DB = 6.0
LOUDNESS_CHANGE_MERGE_S = 3.0


@dataclass
class AudioResult:
    """What the audio analysis produced. ``series`` = (time, loudness 0..1, voice 0..1, music 0..1) about twice a second; the lists are timing only."""

    profile: AudioProfile = field(default_factory=AudioProfile)
    confidence: float = 0.0
    silences: list[tuple[float, float]] = field(default_factory=list)  # true silence: below the adaptive floor, >= 0.25 s
    sfx_times: list[float] = field(default_factory=list)
    music_change_times: list[float] = field(default_factory=list)
    series: list[tuple[float, float, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pauses: list[tuple[float, float]] = field(default_factory=list)  # gaps between speech runs (silent OR filled by music), 0.25-5 s; feeds average_pause_duration


# ---------------------------------------------------------------------------------------------- loading
def _has_audio_stream(ffmpeg: FFmpegService, path: Path) -> bool | None:
    """True/False from ffprobe; None when ffprobe cannot be used (the decode itself then decides). An unreadable file is an error, not 'no audio'."""
    try:
        probe = ffmpeg.ffprobe()
    except FFmpegUnavailableError:
        return None
    try:
        r = subprocess.run([probe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                           timeout=60, encoding="utf-8", errors="replace", **_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReferenceAnalysisError("The reference file could not be inspected.", details=str(exc)) from exc
    if r.returncode != 0:
        raise ReferenceAnalysisError("The reference file could not be read.", details=(r.stderr or "").strip()[-400:])
    return bool(r.stdout.strip())


def load_audio(ffmpeg: FFmpegService, path: Path, sr: int = TARGET_SR, cancel: threading.Event | None = None) -> np.ndarray | None:
    """The first audio stream of ``path`` as mono float32 in [-1, 1] at ``sr`` Hz, streamed through an FFmpeg pipe (nothing is written to disk).

    ``None`` = the file has no audio stream (a normal case for silent clips). A file that cannot be decoded raises ``ReferenceAnalysisError``;
    a truncated file yields what could be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise ReferenceAnalysisError("The reference file was not found.", details=str(path))
    has = _has_audio_stream(ffmpeg, path)
    if has is False:
        return None
    try:
        exe = ffmpeg.ffmpeg()
    except FFmpegUnavailableError as exc:
        raise ReferenceAnalysisError("FFmpeg could not be found.", details=str(exc)) from exc
    args = [exe, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-sn", "-dn", "-map", "0:a:0?", "-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "pipe:1"]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_NO_WINDOW)
    except OSError as exc:
        raise ReferenceAnalysisError("FFmpeg could not be started.", details=str(exc)) from exc
    tail: deque[str] = deque(maxlen=20)

    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.decode("utf-8", "replace").rstrip())

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    chunks: list[np.ndarray] = []
    rest = b""
    try:
        assert proc.stdout is not None
        while True:
            if cancel is not None and cancel.is_set():
                raise AnalysisCancelled()
            buf = proc.stdout.read(1 << 18)
            if not buf:
                break
            buf = rest + buf
            usable = len(buf) - len(buf) % 4
            rest = buf[usable:]
            if usable:
                chunks.append(np.frombuffer(buf[:usable], dtype="<f4").astype(np.float32))
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        t.join(timeout=2)
    out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if len(out) == 0:
        text = "\n".join(tail)
        if has is None and "does not contain any stream" in text:
            return None  # ffprobe was unavailable and FFmpeg reports the missing stream itself
        if proc.returncode != 0:
            raise ReferenceAnalysisError("The reference audio could not be decoded.", details=text)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------- small helpers
def _db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(power, EPS))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the True runs."""
    m = np.asarray(mask, dtype=bool)
    if not m.size:
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps of at most ``max_gap`` samples that lie BETWEEN True samples."""
    out = np.asarray(mask, dtype=bool).copy()
    runs = _runs(~out)
    for a, b in runs:
        if a > 0 and b < len(out) and b - a <= max_gap:
            out[a:b] = True
    return out


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for a, b in _runs(out):
        if b - a < min_len:
            out[a:b] = False
    return out


def _count_peaks(x: np.ndarray, prominence: float, radius: int = 6) -> int:
    """Local maxima of ``x`` that rise at least ``prominence`` above the lowest point within ``radius`` samples on both sides (syllables of an energy envelope)."""
    n = len(x)
    if n < 3:
        return 0
    k = 0
    for i in range(1, n - 1):
        if x[i] < x[i - 1] or x[i] <= x[i + 1]:
            continue
        lo = x[max(0, i - radius):i].min()
        hi = x[i + 1:i + 1 + radius].min()
        if x[i] - max(lo, hi) >= prominence:
            k += 1
    return k


def _decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Low-pass (windowed sinc, cutoff just under the new Nyquist) then keep every ``factor``-th sample."""
    taps = 8 * factor + 1
    n = np.arange(taps) - taps // 2
    h = np.sinc(n / factor) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h.astype(np.float32), mode="same")[::factor].astype(np.float32)


def _prepare(samples: object) -> np.ndarray:
    """Any numeric array (mono, or channels x samples / samples x channels) -> mono float32; integers are scaled to [-1, 1]; NaN/inf become 0."""
    a = np.asarray(samples)
    if a.ndim == 0 or a.size == 0:
        return np.zeros(0, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=0) if a.shape[0] <= 8 and a.shape[-1] > a.shape[0] else a.reshape(a.shape[0], -1).mean(axis=1)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / float(max(abs(np.iinfo(a.dtype).min), np.iinfo(a.dtype).max))
    return np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class _Spec:
    """The per-frame measurements everything else is built on (frame t is centred at ``tc[t]``)."""

    sr: int
    hop: int
    P: np.ndarray  # (T, K) one-sided power per bin (30 Hz .. 8 kHz), ~mean-square units
    freqs: np.ndarray  # (K,)
    tc: np.ndarray  # (T,) frame-centre times (s)
    E: np.ndarray  # (T,) frame energy, dB
    cpp: np.ndarray  # (T,) harmonicity (cepstral peak at a voice pitch / cepstral level)
    acf: np.ndarray  # (T,) periodicity (normalised autocorrelation peak at a voice pitch, after the first trough so a low tone's slow decay does not count)

    @property
    def hop_s(self) -> float:
        return self.hop / self.sr

    def mask(self, lo: float, hi: float) -> np.ndarray:
        return (self.freqs >= lo) & (self.freqs <= hi)


def _acf_peak(frames: np.ndarray, w: np.ndarray, nfft: int, lag_lo: int, lag_hi: int) -> np.ndarray:
    """Periodicity of each frame: the highest normalised autocorrelation between ``lag_lo`` and ``lag_hi`` samples, searched only AFTER the first trough of the
    autocorrelation (a low-frequency tone or rumble decays slowly from lag 0 and would otherwise read as 'periodic'). 0 for silent frames."""
    f = frames - frames.mean(axis=1, keepdims=True)
    X = np.fft.rfft(f * w, nfft, axis=1)
    ac = np.fft.irfft(X.real ** 2 + X.imag ** 2, nfft, axis=1)[:, :lag_hi + 2]
    e = ac[:, 0]
    ac = ac / np.maximum(e, 1e-12)[:, None]
    trough = (ac[:, 1:-1] < ac[:, :-2]) & (ac[:, 1:-1] <= ac[:, 2:])
    first = np.where(trough.any(axis=1), trough.argmax(axis=1) + 1, lag_hi)
    lags = np.arange(ac.shape[1])[None, :]
    ok = (lags >= np.maximum(lag_lo, first)[:, None]) & (lags <= lag_hi)
    pk = np.where(ok, ac, -1.0).max(axis=1)
    return np.where(e > 1e-9, np.clip(pk, 0.0, 1.0), 0.0).astype(np.float32)


def _spectrogram(x: np.ndarray, sr: int) -> _Spec:
    hop = max(1, int(round(HOP_S * sr)))
    win = 2 * hop
    nfft = 1 << (win - 1).bit_length()
    df = sr / nfft
    kmax = min(nfft // 2, int(MAX_HZ / df)) + 1
    w = np.hanning(win + 2)[1:-1].astype(np.float32)
    norm = 2.0 / (nfft * float(np.sum(w ** 2)))
    T = (len(x) - win) // hop + 1
    freqs = (np.arange(kmax) * df).astype(np.float32)
    P = np.empty((T, kmax), dtype=np.float32)
    cpp = np.empty(T, dtype=np.float32)
    acf = np.empty(T, dtype=np.float32)
    lag_lo, lag_hi = max(2, int(round(sr / ACF_HI_HZ))), int(round(sr / ACF_LO_HZ))
    nacf = 1 << (win + lag_hi).bit_length()
    sel = (freqs >= CPP_LO_HZ) & (freqs <= CPP_HI_HZ)
    nsel = int(sel.sum())
    cwin = np.hanning(nsel).astype(np.float32)
    q_sec = np.fft.rfftfreq(4 * nsel, d=1.0) / df  # quefrency of each cepstral bin (s)
    qpk = (q_sec >= 1.0 / 400.0) & (q_sec <= 1.0 / 80.0)
    qbase = (q_sec >= 0.0005) & (q_sec <= 0.02)
    for a in range(0, T, 1024):
        b = min(T, a + 1024)
        idx = hop * np.arange(a, b)[:, None] + np.arange(win)[None, :]
        X = np.fft.rfft(x[idx] * w, nfft, axis=1)[:, :kmax]
        p = (X.real ** 2 + X.imag ** 2) * norm
        p[:, 0] *= 0.5
        p[:, freqs < 30.0] = 0.0  # DC offset / rumble is not audible content
        P[a:b] = p
        # harmonicity: cepstrum of the log spectrum over the voice band
        L = 10.0 * np.log10(p[:, sel] + EPS)
        L = np.maximum(L, L.max(axis=1, keepdims=True) - 50.0)
        L -= L.mean(axis=1, keepdims=True)
        C = np.abs(np.fft.rfft(L * cwin, 4 * nsel, axis=1))
        cpp[a:b] = C[:, qpk].max(axis=1) / (C[:, qbase].mean(axis=1) + 1e-9)
        acf[a:b] = _acf_peak(x[idx], w, nacf, lag_lo, lag_hi)
    tc = (np.arange(T) + 1) * hop / sr
    E = _db(P.sum(axis=1)).astype(np.float32)
    return _Spec(sr, hop, P, freqs, tc, E, cpp, acf)


# ---------------------------------------------------------------------------------------------- the analyzer
class AudioAnalyzer:
    """Reduces a mono soundtrack to an ``AudioProfile`` plus timing lists. Deterministic; never raises on odd input (empty, silent, constant, short)."""

    def analyze(self, samples: np.ndarray, sr: int, *, shot_times: list[float] | None = None, text_times: list[float] | None = None,
                change_times: list[float] | None = None) -> AudioResult:
        x = _prepare(samples)
        sr = int(sr)
        if len(x) == 0 or sr <= 0:
            return self._empty("No audio samples: the reference has no usable soundtrack.", has_audio=False)
        if sr >= 2 * TARGET_SR - 1:
            f = sr // TARGET_SR
            clipped = float(np.mean(np.abs(x) >= 0.999))
            x, sr = _decimate(x, f), sr // f
        else:
            clipped = float(np.mean(np.abs(x) >= 0.999))
        dur = len(x) / sr
        if dur < MIN_SECONDS or sr < 4000:
            return self._empty(f"The audio is too short or too low-quality to analyse ({dur:.1f} s at {sr} Hz).", has_audio=True)
        return _Run(x, sr, dur, clipped, shot_times or [], text_times or [], change_times or []).result()

    @staticmethod
    def _empty(note: str, has_audio: bool) -> AudioResult:
        return AudioResult(AudioProfile(has_audio=has_audio), 0.0, [], [], [], [], [note])


class _Run:
    """One analysis pass over one soundtrack (kept apart from the analyzer so the stages can share their intermediate arrays)."""

    def __init__(self, x: np.ndarray, sr: int, dur: float, clipped: float, shots: list[float], texts: list[float], changes: list[float]) -> None:
        self.x, self.sr, self.dur, self.clipped = x, sr, dur, clipped
        self.shots, self.texts, self.changes = shots, texts, changes
        self.notes: list[str] = []
        self.spec = _spectrogram(x, sr)
        s = self.spec
        self.T = len(s.tc)
        self.minutes = dur / 60.0
        self.W = max(1, int(dur / STEP_S + 0.5))
        edges = np.arange(self.W + 1) * STEP_S
        self.w_lo = np.searchsorted(s.tc, edges[:-1], side="left")
        self.w_hi = np.searchsorted(s.tc, edges[1:], side="left")
        self.w_hi[-1] = self.T
        self.w_mid = (np.arange(self.W) + 0.5) * STEP_S
        self.frame_win = np.minimum(np.searchsorted(edges[1:], s.tc, side="right"), self.W - 1)
        audible = s.E[s.E > -100.0]
        self.loud = float(np.percentile(audible, 95)) if len(audible) else DB_FLOOR  # the soundtrack's own loud level (dB)
        self.audible_share = float(np.mean(s.E > -72.0)) if self.T else 0.0

    # ------------------------------------------------------------------ orchestration
    def result(self) -> AudioResult:
        s = self.spec
        if self.loud < -75.0:
            return self._silent_result()
        self.thr_sil = self._silence_threshold()
        self._speech()
        self._floor_and_music()
        self._silences_and_pauses()
        self._ducking()
        self._music_stats()
        self._sfx()
        self._loudness()
        return self._finish()

    def _silent_result(self) -> AudioResult:
        prof = AudioProfile(has_audio=True, silence_frequency=1.0 / self.minutes if self.dur else 0.0, silence_percentage=100.0)
        notes = ["The soundtrack is silent (digital silence or inaudibly quiet): there is nothing to measure."]
        series = [(float(t), 0.0, 0.0, 0.0) for t in self.w_mid]
        return AudioResult(prof, 0.25, [(0.0, self.dur)], [], [], series, notes)

    # ------------------------------------------------------------------ silence threshold
    def _silence_threshold(self) -> float:
        """Adaptive: a few dB over the file's own quiet level, clamped so that neither a constant bed (everything near the loud level) nor a very quiet file
        makes the whole thing 'silent'."""
        e = self.spec.E
        quiet = float(np.percentile(e, 10))
        return float(max(SILENCE_ABS_MIN_DB, min(quiet + 8.0, self.loud - SILENCE_BELOW_LOUD_DB)))

    # ------------------------------------------------------------------ speech
    def _speech(self) -> None:
        s = self.spec
        act = max(-72.0, self.loud - ACTIVITY_REL_DB)
        voiced = (s.E >= act) & ((s.acf >= ACF_VOICED) | (s.cpp >= CPP_VOICED))
        bridge = max(1, int(round(BRIDGE_S / s.hop_s)) - 1)
        merged = _close_gaps(voiced, bridge)
        mask = np.zeros(self.T, dtype=bool)
        smooth = np.convolve(s.E, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        for a, b in _runs(merged):
            n_v = int(voiced[a:b].sum())
            length = (b - a) * s.hop_s
            if length < SPEECH_MIN_S or n_v / (b - a) < SPEECH_MIN_DENSITY:
                continue
            if _count_peaks(smooth[a:b], SYLLABLE_PROMINENCE_DB) < 2:
                continue  # one sustained harmonic sound (a held note, a siren) is not syllabic speech
            mask[a:b] = True
        self.voiced, self.speech = voiced, mask
        self.speech_runs = _runs(mask)
        # speech share per step
        cs = np.concatenate(([0], np.cumsum(mask)))
        n = np.maximum(self.w_hi - self.w_lo, 1)
        self.speech_frac = (cs[self.w_hi] - cs[self.w_lo]) / n

    # ------------------------------------------------------------------ stationary floor, tonality, music
    def _floor_and_music(self) -> None:
        s = self.spec
        T, W, K = self.T, self.W, s.P.shape[1]
        F = np.empty((W, K), dtype=np.float32)
        # frames within 0.35 s of speech carry speech (and the bed's duck ramps): a window that is in a gap between speech takes its floor from the clean
        # frames only, so a short gap still shows the bed's real (un-ducked) level
        guard = int(round(GAP_GUARD_S / s.hop_s))
        near = np.convolve(self.speech.astype(np.float32), np.ones(2 * guard + 1, dtype=np.float32), mode="same") > 0
        for w in range(W):
            a = int(np.searchsorted(s.tc, self.w_mid[w] - FLOOR_HALF_S, side="left"))
            b = int(np.searchsorted(s.tc, self.w_mid[w] + FLOOR_HALF_S, side="right"))
            blk = s.P[a:b]
            if self.speech_frac[w] < 0.5:
                clean = blk[~near[a:b]]
                if len(clean) >= 12:
                    blk = clean
            if len(blk) < 4:
                blk = s.P[max(0, a - 2):b + 2]
            k = int(round(FLOOR_Q * (len(blk) - 1)))
            F[w] = np.partition(blk, k, axis=0)[k]
        self.F = F
        # the floor used to *subtract* foreground: the loudest floor within +-1 s, so a bed that fades in/out or steps never leaves a "foreground" behind
        Fp = np.pad(F, ((2, 2), (0, 0)), mode="edge")
        self.F_fg = np.maximum.reduce([Fp[i:i + W] for i in range(5)])
        tonal = s.mask(TONAL_LO_HZ, TONAL_HI_HZ)
        self.tonal_sel = tonal
        Ft = F[:, tonal]
        df = float(s.freqs[1] - s.freqs[0])
        R = max(2, int(round(PEAK_SURROUND_HZ / df)))
        lf = _db(Ft)
        tf = np.zeros(W, dtype=np.float32)
        for a in range(0, W, 64):
            blk = lf[a:a + 64]
            pad = np.pad(blk, ((0, 0), (R, R)), mode="edge")
            med = np.median(sliding_window_view(pad, 2 * R + 1, axis=1), axis=2)
            peaks = (blk - med) >= PEAK_CONTRAST_DB
            tot = Ft[a:a + 64].sum(axis=1)
            tf[a:a + 64] = np.where(tot > 0, (Ft[a:a + 64] * peaks).sum(axis=1) / np.maximum(tot, 1e-30), 0.0)
        self.tonal_frac = tf
        self.floor_db = _db(Ft.sum(axis=1)).astype(np.float32)  # the bed's level (dB), per step
        gate = max(MUSIC_ABS_GATE_DB, self.loud - MUSIC_REL_GATE_DB)
        self.floor_gate = gate
        active = (tf >= TONAL_ACTIVE + TONAL_SPEECH_EXTRA * self.speech_frac) & (self.floor_db >= gate)
        active = _close_gaps(active, MUSIC_FILL_GAP)
        active = _drop_short(active, MUSIC_MIN_RUN)
        self.music = self._extend_music_edges(active)

    def _extend_music_edges(self, active: np.ndarray) -> np.ndarray:
        """The minimum-statistics floor only 'sees' the bed once ~90 % of its window is inside it, so a bed that starts / stops inside the file is found
        about 0.7 s late / early. Each such edge is moved out by one step when the raw frames there still hold the bed's tonal energy."""
        s = self.spec
        out = active.copy()
        for a, b in _runs(active):
            tmpl = self.F[a:b][:, self.tonal_sel]
            tmpl = np.median(tmpl, axis=0)
            tmask = tmpl >= tmpl.max() * 0.05
            ref = float(tmpl[tmask].sum())
            for side, w in ((0, a - 1), (1, b)):
                if w < 0 or w >= self.W or (side == 0 and a == 0) or (side == 1 and b == self.W):
                    continue
                lo, hi = self.w_lo[w], max(self.w_hi[w], self.w_lo[w] + 1)
                p = s.P[lo:hi][:, self.tonal_sel][:, tmask].sum(axis=1)
                k = int(round(0.25 * (len(p) - 1)))
                level = float(np.partition(p, k)[k])
                if level >= 0.35 * ref and not self.speech_frac[w] > 0.5:
                    out[w] = True
        return out

    # ------------------------------------------------------------------ silence and pauses
    def _silences_and_pauses(self) -> None:
        s = self.spec
        hs = s.hop_s
        sil = s.E < self.thr_sil
        min_f = int(round(SILENCE_MIN_S / hs))
        sils: list[tuple[float, float]] = []
        for a, b in _runs(sil):
            if b - a < min_f:
                continue
            t0 = 0.0 if a == 0 else float(s.tc[a] - hs / 2)
            t1 = self.dur if b == self.T else float(s.tc[b - 1] + hs / 2)
            sils.append((round(t0, 3), round(min(t1, self.dur), 3)))
        self.silences = sils
        self.silent_time = float(sum(b - a for a, b in sils))
        self.sil_frames = sil
        pauses: list[tuple[float, float]] = []
        runs = self.speech_runs
        for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
            t0 = float(s.tc[b0 - 1] + hs / 2)
            t1 = float(s.tc[a1] - hs / 2)
            gap = t1 - t0
            if MIN_PAUSE_S <= gap <= MAX_PAUSE_S:
                pauses.append((round(t0, 3), round(t1, 3)))
        self.pauses = pauses

    # ------------------------------------------------------------------ ducking
    def _ducking(self) -> None:
        """How far the bed falls while speech runs, relative to its level in the gaps next to that speech. Paired per speech run (a slow drift of the bed
        cancels out); needs music in both situations, else 0 with a note."""
        self.duck_db = 0.0
        self.duck_ok = False
        m, lvl, sp = self.music, self.floor_db.astype(float), self.speech_frac >= 0.5
        if m.sum() < 4:
            return
        runs: list[tuple[bool, int, int]] = []
        state = np.where(sp, 1, 0)
        a = 0
        for i in range(1, self.W + 1):
            if i == self.W or state[i] != state[a]:
                runs.append((bool(state[a]), a, i))
                a = i
        gap_level: dict[int, float] = {}
        speech_level: dict[int, float] = {}
        for j, (is_sp, a, b) in enumerate(runs):
            ws = [w for w in range(a, b) if m[w]]
            if is_sp:
                if len(ws) * STEP_S >= 1.5 and (b - a) * STEP_S >= 2.0:
                    speech_level[j] = float(np.median(lvl[ws]))
            elif len(ws) * STEP_S >= 1.0:
                gap_level[j] = float(np.max(lvl[ws]))  # smear can only lower a gap's floor: its best step is the honest one
        drops, sp_time, gap_time = [], 0.0, 0.0
        for j, v in speech_level.items():
            refs = [gap_level[k] for k in (j - 1, j + 1) if k in gap_level]
            if not refs:
                continue
            drops.append(float(np.max(refs)) - v)  # the gap's level can only be under-read (smear, fades), so the louder neighbour is the honest reference
            _, a, b = runs[j]
            sp_time += sum(1 for w in range(a, b) if m[w]) * STEP_S
            gap_time += sum(sum(1 for w in range(runs[k][1], runs[k][2]) if m[w]) for k in (j - 1, j + 1) if k in gap_level) * STEP_S
        if drops and sp_time >= DUCK_MIN_SPEECH_S and gap_time >= DUCK_MIN_GAP_S:
            self.duck_db = max(0.0, float(np.median(drops)))
            self.duck_ok = True
            self.duck_pairs = len(drops)
        elif m.any() and (sp & m).any():
            self.notes.append("Music ducking could not be measured: too little music-with-speech and music-without-speech material alternates (strength reported as 0, low confidence).")

    # ------------------------------------------------------------------ music statistics
    def _speech_edge_zone(self, margin: float) -> np.ndarray:
        """Steps whose centre is within ``margin`` s of the start or end of a speech run (where the bed ducks and the floor estimate is smeared)."""
        s = self.spec
        edges = np.array([t for a, b in self.speech_runs for t in (s.tc[a], s.tc[b - 1])], dtype=float)
        if not len(edges):
            return np.zeros(self.W, dtype=bool)
        return np.abs(self.w_mid[:, None] - edges[None, :]).min(axis=1) <= margin

    def _music_stats(self) -> None:
        W = self.W
        m = self.music
        s = self.spec
        self.music_presence = float(m.sum() / W) if W else 0.0
        lvl = self.floor_db.astype(float)
        zone = self._speech_edge_zone(1.0)
        self.edge_zone = zone
        spread = 0.0
        crest = 0.0
        if m.sum() >= 4:
            # level variation of the bed itself: speech-edge steps are excluded (the duck lives there) and the duck is added back inside speech
            stable = m & ~zone
            if stable.sum() < 4:
                stable = m
            v = (lvl + (self.duck_db if self.duck_ok else 0.0) * (self.speech_frac >= 0.5))[stable]
            spread = float(np.percentile(v, 90) - np.percentile(v, 10))
            # texture intensity: how far the bed's mean power stands above its stationary floor in speech-free, effect-free steps (a beat or swell leaves the
            # mean well above the floor; a flat pad does not)
            cs = np.concatenate(([0.0], np.cumsum(s.P.sum(axis=1))))
            n = np.maximum(self.w_hi - self.w_lo, 1)
            mean_db = _db((cs[self.w_hi] - cs[self.w_lo]) / n)
            clean = m & (self.speech_frac < 0.05) & ~zone
            if clean.sum() >= 3:
                crest = float(np.median(mean_db[clean] - lvl[clean]))
        self.spread, self.crest = spread, crest
        self.music_dynamics = clamp(spread / DYNAMICS_SPREAD_DB + 0.5 * clamp((crest - 1.5) / 4.0)) if m.sum() >= 4 else 0.0
        self._music_changes()

    def _music_changes(self) -> None:
        """Abrupt changes of the bed: a level step (>= 5 dB between 1 s blocks), a new texture (the floor's spectral shape changes: chord / instrument), the
        bed starting or stopping. Fades at the very start / end of the file and the duck at speech edges are not changes."""
        W, m, dur = self.W, self.music, self.dur
        lvl = self.floor_db.astype(float)
        zone = self.edge_zone
        times: list[float] = []
        self.music_change_times = []
        if m.sum() < 3:
            return

        def inside(t: float) -> bool:
            return 1.5 <= t <= dur - 1.5

        def step(v: int) -> float:
            if v < 2 or v + 2 > W or not (m[v - 2:v].all() and m[v:v + 2].all()):
                return 0.0
            return float(np.mean(lvl[v:v + 2]) - np.mean(lvl[v - 2:v]))

        for w in range(2, W - 1):
            d = step(w)
            need = MUSIC_STEP_DB if not zone[w] else DUCK_FULL_DB  # next to speech only a step bigger than any duck counts
            if abs(d) < need or not inside(w * STEP_S):
                continue
            if abs(d) < max(abs(step(v)) for v in range(max(2, w - 2), min(W - 1, w + 3))) - 1e-9:
                continue  # not the sharpest point of this step
            times.append(w * STEP_S)
        # texture: the floor's spectral shape either side of a point
        Ft = np.sqrt(self.F[:, self.tonal_sel])
        nrm = np.linalg.norm(Ft, axis=1)

        def dist(v: int) -> float:
            if v < 2 or v + 2 > W or not (m[v - 2] and m[v + 1]) or nrm[v - 2] <= 0 or nrm[v + 1] <= 0:
                return 0.0
            return 1.0 - float(Ft[v - 2] @ Ft[v + 1]) / float(nrm[v - 2] * nrm[v + 1])

        for w in range(2, W - 2):
            d = dist(w)
            if d >= MUSIC_TEXTURE_DIST and inside((w + 0.5) * STEP_S) and d >= max(dist(v) for v in range(max(2, w - 2), min(W - 2, w + 3))) - 1e-9:
                times.append((w + 0.5) * STEP_S)
        # the bed starting or stopping inside the file (>= 3 s of bed)
        for a, b in _runs(m):
            if (b - a) * STEP_S < 3.0:
                continue
            if a > 0 and inside(a * STEP_S):
                times.append(a * STEP_S)
            if b < W and inside(b * STEP_S):
                times.append(b * STEP_S)
        times.sort()
        merged: list[float] = []
        for t in times:
            if merged and t - merged[-1] < MUSIC_CHANGE_MERGE_S:
                continue
            merged.append(t)
        self.music_change_times = [round(float(t), 3) for t in merged]

    # ------------------------------------------------------------------ sound effects
    def _sfx(self) -> None:
        s = self.spec
        T, hs = self.T, s.hop_s
        sp_dil = np.convolve(self.speech.astype(np.float32), np.ones(2 * int(round(SFX_SPEECH_DILATE_S / hs)) + 1, dtype=np.float32), mode="same") > 0
        tot_m = s.freqs >= 40.0
        lb_m = (s.freqs >= 30.0) & (s.freqs <= 120.0)
        hb_hi = min(MAX_HZ, 0.98 * self.sr / 2)
        hb_m = (s.freqs >= 3500.0) & (s.freqs <= hb_hi)
        n_tot, n_lb, n_hb = np.zeros(T), np.zeros(T), np.zeros(T)
        for w in range(self.W):
            a, b = self.w_lo[w], self.w_hi[w]
            if b <= a:
                continue
            sub = np.maximum(s.P[a:b] - FG_OVER * self.F_fg[w], 0.0)
            n_tot[a:b] = sub[:, tot_m].sum(axis=1)
            n_lb[a:b] = sub[:, lb_m].sum(axis=1)
            if hb_m.any():
                n_hb[a:b] = sub[:, hb_m].sum(axis=1)

        def smooth_db(p: np.ndarray) -> np.ndarray:
            return _db(np.convolve(p, np.ones(3) / 3.0, mode="same"))

        s_tot, s_lb, s_hb = smooth_db(n_tot), smooth_db(n_lb), smooth_db(n_hb)
        gate = max(SILENCE_ABS_MIN_DB, self.loud - SFX_GATE_REL_DB)
        floor_frame = _db(self.F_fg[:, self.tonal_sel].sum(axis=1))[self.frame_win]  # the bed's (neighbourhood-maximum) level at every frame
        events: list[tuple[int, int, int, float]] = []  # (onset frame, offset frame, peak frame, strength dB)
        for ser, rise, kind in ((s_tot, SFX_RISE_TOT_DB, "tot"), (s_lb, SFX_RISE_BAND_DB, "lb"), (s_hb, SFX_RISE_BAND_DB, "hb")):
            if kind == "hb" and not hb_m.any():
                continue
            g = gate if kind == "tot" else max(SILENCE_ABS_MIN_DB, self.loud - SFX_BAND_GATE_REL_DB)
            for on, off, pk in self._bursts(ser, g, rise):
                cover = float(sp_dil[on:off + 1].mean())
                if kind == "tot":
                    if cover >= SFX_SPEECH_COVER:
                        continue  # speech (its own syllables) - judged by the band series only
                    if ser[pk] < floor_frame[pk] + SFX_OVER_BED_DB:
                        continue  # not clearly louder than the bed: a beat or swell of the music, not an effect
                else:
                    ctx = np.concatenate((ser[max(0, pk - 80):max(0, pk - 20)], ser[pk + 20:pk + 80]))
                    if len(ctx) >= 20 and (ser[pk] < float(np.percentile(ctx, 90)) + SFX_STICKOUT_DB or ser[pk] < float(ctx.max()) + 2.0):
                        continue  # no louder than its own neighbourhood (a sibilant, a bass note)
                events.append((on, off, pk, float(ser[pk])))
        events.sort(key=lambda e: e[0])
        merged: list[list[int | float]] = []
        for on, off, pk, st in events:
            t = s.tc[on]
            if merged and t - s.tc[int(merged[-1][0])] < SFX_MERGE_S:
                if st > merged[-1][3]:
                    merged[-1][2], merged[-1][3] = pk, st
                merged[-1][1] = max(int(merged[-1][1]), off)
                continue
            merged.append([on, off, pk, st])
        starts = np.array([s.tc[int(e[0])] for e in merged])
        keep = self._off_beat(starts)
        ev = [e for e, k in zip(merged, keep) if k]
        self.sfx_times = [round(float(max(0.0, s.tc[int(e[0])] - hs / 2)), 3) for e in ev]
        self.sfx_frames = [(int(e[0]), int(e[1]), int(e[2])) for e in ev]
        self.sfx_per_minute = len(ev) / self.minutes if self.minutes > 0 else 0.0
        # intensity: the effect's peak foreground level against the speech foreground level (or the file's loud level when nobody speaks)
        if ev:
            peak = np.array([_db(float(n_tot[int(e[0]):int(e[1]) + 1].max())) for e in ev])
            ref_frames = n_tot[self.speech]
            ref = float(_db(float(np.percentile(ref_frames, 75)))) if self.speech.sum() >= 20 else self.loud
            self.sfx_intensity = float(np.mean(np.clip(1.0 + (peak - ref) / 20.0, 0.0, 1.0)))
        else:
            self.sfx_intensity = 0.0
        self.sfx_align = {
            "shots": self._share_on(self.sfx_times, self.shots),
            "text": self._share_on(self.sfx_times, self.texts),
            "reveals": self._share_on(self.sfx_times, self.changes),
        }

    def _off_beat(self, starts: np.ndarray) -> list[bool]:
        """False for events that belong to a regular pulse: most gaps between consecutive events share one spacing (0.25-1.2 s, +-0.08 s) - a beat or
        metronome, i.e. rhythm of the music, not effects. Runs of a beat that speech interrupts are still recognised (the modal spacing is global)."""
        n = len(starts)
        keep = [True] * n
        if n < 6:
            return keep
        gaps = np.diff(starts)
        rel = gaps[(gaps >= 0.25) & (gaps <= 1.2)]
        if len(rel) < 5:
            return keep
        hist, edges = np.histogram(rel, bins=np.arange(0.25, 1.25, 0.04))
        k = int(np.argmax(hist))
        period = float(np.median(rel[(rel >= edges[k] - 0.04) & (rel <= edges[k + 1] + 0.04)]))
        match = np.abs(gaps - period) <= 0.08
        if match.sum() >= 5 and match.sum() >= 0.5 * len(gaps):
            for i in range(n):
                if (i > 0 and match[i - 1]) or (i < n - 1 and match[i]):
                    keep[i] = False
            self.notes.append("A regular rhythmic pulse was treated as music, not as sound effects.")
        return keep

    @staticmethod
    def _share_on(times: list[float], marks: list[float]) -> float:
        if not times or not marks:
            return 0.0
        mk = np.sort(np.asarray(marks, dtype=float))
        hit = 0
        for t in times:
            i = int(np.searchsorted(mk, t + SFX_ALIGN_BEFORE_S, side="right")) - 1  # latest mark that is not more than 0.15 s after the effect
            if i >= 0 and -SFX_ALIGN_BEFORE_S <= t - mk[i] <= SFX_ALIGN_AFTER_S:
                hit += 1
        return hit / len(times)

    def _bursts(self, s: np.ndarray, gate: float, rise_db: float) -> list[tuple[int, int, int]]:
        """Short bursts in a dB series: a local maximum (+-0.1 s) above ``gate`` that rose ``rise_db`` over the previous 0.3 s, decays and is < 1.2 s long
        measured 15 dB under its peak. Returns (onset frame, offset frame, peak frame)."""
        T = len(s)
        if T < 8:
            return []
        k = 4
        rmax = sliding_window_view(np.pad(s, (k, k), mode="edge"), 2 * k + 1).max(axis=1)
        out = []
        for p in np.flatnonzero((s >= rmax) & (s >= gate)).tolist():
            if p > 0 and s[p] <= s[p - 1]:
                continue  # the first sample of a plateau only
            lo = max(0, p - 12)
            pre = float(s[lo:max(lo + 1, p - 1)].min())
            if s[p] - pre < rise_db:
                continue
            on = p
            while on > 0 and s[on - 1] > s[p] - SFX_DROP_DB and p - on < 20:
                on -= 1
            off = p
            while off < T - 1 and s[off + 1] > s[p] - SFX_DROP_DB and off - p < 60:
                off += 1
            dur = (off - on + 1) * self.spec.hop_s
            if SFX_MIN_S <= dur <= SFX_MAX_S:
                out.append((on, off, p))
        return out

    # ------------------------------------------------------------------ loudness
    def _loudness(self) -> None:
        s = self.spec
        cs = np.concatenate(([0.0], np.cumsum(s.P.sum(axis=1))))
        n = np.maximum(self.w_hi - self.w_lo, 1)
        pw = (cs[self.w_hi] - cs[self.w_lo]) / n  # mean power per step
        self.step_power = pw
        self.step_db = _db(pw)
        # short-term (3 s) loudness range, EBU-R128 style: 3 s power average, relative gate 20 dB under the mean, 10th..95th percentile
        k = 6
        st = np.array([pw[max(0, w - k // 2):w + k // 2].mean() for w in range(self.W)])
        st_db = _db(st)
        gate = float(_db(float(pw.mean()))) - 20.0
        sel = st_db[st_db >= gate]
        self.dynamic_range = float(np.percentile(sel, 95) - np.percentile(sel, 10)) if len(sel) >= 4 else 0.0
        # loudness changes: the loud envelope (90th percentile over 3 s, so pauses between phrases do not register) jumping >= 6 dB between 2 s blocks
        env = np.array([np.percentile(self.step_db[max(0, w - 3):w + 4], 90) for w in range(self.W)])
        env = np.maximum(env, self.thr_sil)
        ch: list[float] = []
        for w in range(1, self.W):
            a0, b0 = max(0, w - 4), min(self.W, w + 4)
            if w - a0 < 2 or b0 - w < 2:
                continue
            d = float(np.mean(env[w:b0]) - np.mean(env[a0:w]))
            if abs(d) >= LOUDNESS_CHANGE_DB:
                nb = [abs(float(np.mean(env[v:min(self.W, v + 4)]) - np.mean(env[max(0, v - 4):v]))) for v in range(max(1, w - 2), min(self.W, w + 3))]
                if abs(d) >= max(nb) - 1e-9:
                    ch.append(w * STEP_S)
        merged: list[float] = []
        for t in ch:
            if merged and t - merged[-1] < LOUDNESS_CHANGE_MERGE_S:
                continue
            merged.append(t)
        self.loudness_changes = len(merged)

    # ------------------------------------------------------------------ assemble
    def _voice_dominance(self) -> float:
        s = self.spec
        nonsil = ~self.sil_frames
        if nonsil.sum() == 0:
            return 0.0
        dom_w = np.zeros(self.W, dtype=bool)
        sp_pow = self.step_power - np.where(self.music, 10 ** (self.floor_db / 10.0), 0.0)
        sp_db = _db(np.maximum(sp_pow, 0.0))
        for w in range(self.W):
            if self.speech_frac[w] < 0.5:
                continue
            dom_w[w] = (not self.music[w]) or sp_db[w] >= self.floor_db[w] - 3.0
        dom_frames = self.speech & dom_w[self.frame_win]
        return float(clamp(dom_frames.sum() / max(1, int(nonsil.sum()))))

    def _finish(self) -> AudioResult:
        s = self.spec
        dur, minutes = self.dur, self.minutes
        prof = AudioProfile(has_audio=True)
        prof.voice_dominance = round(self._voice_dominance(), 4)
        prof.music_presence = round(self.music_presence, 4)
        prof.music_ducking_strength = round(clamp(self.duck_db / DUCK_FULL_DB), 4) if self.duck_ok else 0.0
        prof.music_dynamics = round(self.music_dynamics, 4)
        prof.music_behavior = music_class(prof.music_presence, prof.music_dynamics)
        prof.music_changes_per_minute = round(len(self.music_change_times) / minutes, 3)
        prof.sfx_per_minute = round(self.sfx_per_minute, 3)
        prof.sfx_class = sfx_class(prof.sfx_per_minute)
        prof.sfx_intensity = round(self.sfx_intensity, 4)
        prof.sfx_on_transitions = round(self.sfx_align["shots"], 4)
        prof.sfx_on_text = round(self.sfx_align["text"], 4)
        prof.sfx_on_reveals = round(self.sfx_align["reveals"], 4)
        prof.silence_frequency = round(len(self.silences) / minutes, 3)
        prof.silence_percentage = round(clamp(self.silent_time / dur, 0.0, 1.0) * 100.0, 2)
        pd = [b - a for a, b in self.pauses]
        prof.average_pause_duration = round(float(np.mean(pd)), 3) if pd else 0.0
        prof.long_pause_frequency = round(sum(1 for d in pd if d >= LONG_PAUSE_S) / minutes, 3)
        prof.audio_dynamic_range = round(self.dynamic_range, 2)
        prof.loudness_changes_per_minute = round(self.loudness_changes / minutes, 3)
        series = self._series()
        conf = self._confidence(prof)
        return AudioResult(prof, round(conf, 3), self.silences, self.sfx_times, self.music_change_times, series, self.notes, self.pauses)

    def _series(self) -> list[tuple[float, float, float, float]]:
        db = self.step_db
        live = db[db > self.thr_sil]
        ref = float(np.percentile(live, 95)) if len(live) else self.loud
        out = []
        for w in range(self.W):
            loud = clamp(1.0 + (float(db[w]) - ref) / 45.0) if db[w] > DB_FLOOR + 1 else 0.0
            music = clamp((float(self.tonal_frac[w]) - 0.3) / 0.4) if self.music[w] else 0.0
            out.append((round(float(self.w_mid[w]), 3), round(loud, 3), round(float(self.speech_frac[w]), 3), round(music, 3)))
        return out

    # ------------------------------------------------------------------ confidence
    def _confidence(self, prof: AudioProfile) -> float:
        dur = self.dur
        c = 0.92
        if dur < SHORT_SECONDS:
            c *= clamp(0.35 + 0.65 * (dur - 3.0) / (SHORT_SECONDS - 3.0), 0.3, 1.0)
            self.notes.append(f"Short audio ({dur:.0f} s): rates and shares are estimates (confidence reduced).")
        if self.loud < -50.0:
            c *= 0.5
            self.notes.append("The soundtrack is extremely quiet: speech/music separation is unreliable.")
        elif self.loud < -40.0:
            c *= 0.8
        if self.clipped >= 0.05:
            c *= 0.55
            self.notes.append("Heavily clipped audio: spectral decisions are less reliable.")
        elif self.clipped >= 0.01:
            c *= 0.75
            self.notes.append("Some clipping in the audio.")
        if self.audible_share < 0.05:
            c = min(c, 0.3)
            self.notes.append("Almost nothing is audible: there is little to measure.")
        # speech/music separability: steps whose evidence sits between the two classes
        speech_any = self.speech_frac >= 0.5
        both = bool(speech_any.any() and self.music.any())
        amb = (self.tonal_frac > 0.3) & (self.tonal_frac < 0.6) & (self.floor_db >= self.floor_gate)
        amb_share = float(amb.sum()) / max(1, int((speech_any | self.music | amb).sum()))
        if amb_share > 0.1:
            c *= 1.0 - 0.5 * min(1.0, amb_share)
            self.notes.append("Part of the audio sits between 'music' and 'not music': the music figures are less certain.")
        if both:
            overlap = float((speech_any & self.music).sum()) / max(1, int(speech_any.sum()))
            if overlap > 0.3 and not self.duck_ok:
                c *= 0.85  # speech and music overlap a lot and the duck could not be measured
        # nothing recognised although something is audible
        if prof.voice_dominance < 0.02 and prof.music_presence < 0.05 and not self.sfx_times and self.audible_share >= 0.05:
            c *= 0.5
            self.notes.append("The audio contains sound that is neither speech, music nor effects (noise, ambience): the structure could not be classified.")
        # a stationary but structureless bed (room tone, hiss, fan) is reported, not counted as music
        plain = (~self.music) & (self.speech_frac < 0.2) & (self.floor_db >= max(-60.0, self.loud - 38.0)) & (self.floor_db >= self.step_db - 6.0)
        self.noise_bed = float(np.mean(plain))
        if self.noise_bed > 0.5 and prof.music_presence < 0.1:
            c *= 0.9
            self.notes.append("A steady structureless bed (room tone / hiss) is present: it is not counted as music.")
        # near the classification thresholds
        near = 0
        for v, edge, tol in ((prof.music_presence, 0.1, 0.03), (prof.music_presence, 0.35, 0.04)):
            if abs(v - edge) < tol:
                near += 1
        if prof.music_presence >= 0.1:
            for edge in (0.3, 0.65):
                if abs(prof.music_dynamics - edge) < 0.05:
                    near += 1
        for edge in (0.25, 3.0, 8.0):
            if abs(prof.sfx_per_minute - edge) < 0.1 * edge:
                near += 1
        if near:
            c *= 1.0 - 0.07 * min(near, 3)
            self.notes.append("A classification is close to its threshold (music presence / dynamics or effect rate).")
        return float(clamp(c, 0.0, 1.0))
