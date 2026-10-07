"""LoudnessAnalyzer: peak, RMS, LUFS (where the backend can measure it), dynamic range, clipping and silence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.audio.backend import ANALYSIS_SR, AudioBackend, LoudnessResult
from app.presentation.models import AudioIssue, AudioIssueRecord

EPS = 1e-9
WIN = 0.05  # envelope window (s)
SILENCE_DB = -45.0
CLIP_LEVEL = 0.999


def db(x: float) -> float:
    return 20.0 * float(np.log10(max(x, EPS)))


@dataclass
class LoudnessThresholds:
    too_quiet_lufs: float = -28.0
    too_quiet_rms_db: float = -34.0
    too_loud_lufs: float = -10.0
    clip_runs: int = 3  # runs of >= 3 consecutive full-scale samples
    noise_floor_db: float = -42.0
    long_silence_s: float = 3.0
    min_silence_s: float = 0.5


@dataclass
class Loudness:
    peak_db: float
    rms_db: float
    lufs: float | None
    loudness_range: float | None
    dynamic_range_db: float
    clipped_samples: int
    clipped_runs: int
    noise_floor_db: float
    silence_regions: list[list[float]]
    envelope_db: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    issues: list[AudioIssueRecord] = field(default_factory=list)


def envelope(samples: np.ndarray, sr: int = ANALYSIS_SR, win: float = WIN) -> np.ndarray:
    """RMS level (dBFS) per ``win`` seconds."""
    n = max(1, int(sr * win))
    usable = (len(samples) // n) * n
    if usable == 0:
        return np.array([db(float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0)])
    frames = samples[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, EPS))


def silence_regions(env_db: np.ndarray, win: float = WIN, threshold: float = SILENCE_DB, min_len: float = 0.5) -> list[list[float]]:
    out, start = [], None
    for i, v in enumerate(env_db):
        if v < threshold:
            if start is None:
                start = i
        elif start is not None:
            if (i - start) * win >= min_len:
                out.append([round(start * win, 3), round(i * win, 3)])
            start = None
    if start is not None and (len(env_db) - start) * win >= min_len:
        out.append([round(start * win, 3), round(len(env_db) * win, 3)])
    return out


def clip_runs(samples: np.ndarray, level: float = CLIP_LEVEL, run: int = 3) -> tuple[int, int, list[float]]:
    """(clipped sample count, number of runs of >= ``run`` consecutive clipped samples, times of those runs)."""
    hot = np.abs(samples) >= level
    count = int(hot.sum())
    if count == 0:
        return 0, 0, []
    padded = np.concatenate(([0], hot.view(np.int8), [0]))
    diff = np.diff(padded)
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    runs = [(s, e) for s, e in zip(starts, ends) if e - s >= run]
    return count, len(runs), [round(s / ANALYSIS_SR, 3) for s, _ in runs[:200]]


class LoudnessAnalyzer:
    def __init__(self, backend: AudioBackend, thresholds: LoudnessThresholds | None = None) -> None:
        self.backend, self.t = backend, thresholds or LoudnessThresholds()

    def analyze(self, path: Path, samples: np.ndarray | None = None, sr: int = ANALYSIS_SR) -> Loudness:
        samples = self.backend.decode_mono(path) if samples is None else samples
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        env = envelope(samples, sr)
        active = env[env > SILENCE_DB]
        dyn = float(np.percentile(active, 95) - np.percentile(active, 10)) if len(active) > 5 else 0.0
        floor = float(np.percentile(env, 10)) if len(env) else -90.0
        count, runs, run_times = clip_runs(samples)
        sil = silence_regions(env, WIN, SILENCE_DB, self.t.min_silence_s)
        ld: LoudnessResult = self.backend.measure_loudness(path)
        res = Loudness(db(peak), db(rms), ld.lufs, ld.loudness_range, round(dyn, 2), count, runs, floor, sil, env)
        res.issues = self.issues(res, run_times)
        return res

    def issues(self, r: Loudness, clip_times: list[float] | None = None) -> list[AudioIssueRecord]:
        t, out = self.t, []
        if r.clipped_samples and r.clipped_runs >= 1:
            out.append(AudioIssueRecord(AudioIssue.CLIPPING.value, "error", f"The audio clips ({r.clipped_samples} full-scale samples, {r.clipped_runs} runs). "
                                        "Lower the gain or enable the limiter; the original file is not changed.",
                                        clip_times[0] if clip_times else None, None))
        quiet = (r.lufs is not None and r.lufs < t.too_quiet_lufs) or (r.lufs is None and r.rms_db < t.too_quiet_rms_db)
        if quiet:
            out.append(AudioIssueRecord(AudioIssue.TOO_QUIET.value, "warning", "The voice-over is very quiet" + (f" ({r.lufs:.1f} LUFS)" if r.lufs is not None else f" ({r.rms_db:.1f} dB RMS)")
                                        + ". Raise the gain or enable normalisation."))
        if r.lufs is not None and r.lufs > t.too_loud_lufs:
            out.append(AudioIssueRecord(AudioIssue.TOO_LOUD.value, "warning", f"The voice-over is very loud ({r.lufs:.1f} LUFS). Lower the gain or use the limiter."))
        if r.noise_floor_db > t.noise_floor_db and not quiet:
            out.append(AudioIssueRecord(AudioIssue.EXCESSIVE_NOISE.value, "warning",
                                        f"The background level between words is high ({r.noise_floor_db:.0f} dB). Try noise reduction."))
        for a, b in r.silence_regions:
            if b - a >= t.long_silence_s:
                out.append(AudioIssueRecord(AudioIssue.LONG_SILENCE.value, "warning", f"{b - a:.1f}s of silence at {a:.1f}s.", a, b))
        return out
