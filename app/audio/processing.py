"""AudioProcessingService: the non-destructive voice chain.

``VoiceProcessingSettings`` are parameters stored in the project and always editable. They are turned into an FFmpeg filter chain only
to render a *preview* file in the cache; the original voice-over is never modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from app.audio.backend import AudioBackend
from app.presentation.models import VoiceProcessingSettings

EQ_PRESETS = {
    "clarity": ["equalizer=f=200:t=q:w=1:g=-2", "equalizer=f=3000:t=q:w=1:g=3"],
    "warm": ["equalizer=f=120:t=q:w=1:g=2", "equalizer=f=6000:t=q:w=1:g=-2"],
    "broadcast": ["equalizer=f=100:t=q:w=1:g=3", "equalizer=f=3500:t=q:w=1:g=3", "equalizer=f=9000:t=q:w=1:g=-1.5"],
}


def build_chain(s: VoiceProcessingSettings, duration: float | None = None) -> str:
    """The FFmpeg ``-af`` chain for ``s`` (empty when processing is off)."""
    if not s.enabled:
        return ""
    f: list[str] = []
    if s.highpass_hz > 0:
        f.append(f"highpass=f={s.highpass_hz:g}")
    if s.noise_reduction:
        f.append(f"afftdn=nr={max(0.01, min(97, s.noise_reduction_db)):g}:nf=-40")
    f += EQ_PRESETS.get(s.eq_preset, [])
    if s.compression:
        f.append(f"acompressor=threshold={10 ** (s.comp_threshold_db / 20):.5f}:ratio={max(1.0, s.comp_ratio):g}:attack={s.comp_attack_ms:g}:release={s.comp_release_ms:g}")
    if abs(s.gain_db) > 1e-9:
        f.append(f"volume={s.gain_db:g}dB")
    if s.normalize:
        f.append(f"loudnorm=I={s.target_lufs:g}:TP=-1.5:LRA=11")
    if s.limiter:
        f.append(f"alimiter=limit={10 ** (s.limiter_ceiling_db / 20):.5f}")
    if s.fade_in > 0:
        f.append(f"afade=t=in:st=0:d={s.fade_in:g}")
    if s.fade_out > 0 and duration:
        f.append(f"afade=t=out:st={max(0.0, duration - s.fade_out):.3f}:d={s.fade_out:g}")
    return ",".join(f)


def describe(s: VoiceProcessingSettings) -> list[str]:
    """Human-readable list of the active steps (shown in the UI; every one stays editable)."""
    if not s.enabled:
        return ["Voice processing is off."]
    out = []
    if s.highpass_hz > 0:
        out.append(f"High-pass {s.highpass_hz:g} Hz")
    if s.noise_reduction:
        out.append(f"Noise reduction {s.noise_reduction_db:g} dB")
    if s.eq_preset != "none":
        out.append(f"EQ: {s.eq_preset}")
    if s.compression:
        out.append(f"Compression {s.comp_ratio:g}:1 above {s.comp_threshold_db:g} dB")
    if abs(s.gain_db) > 1e-9:
        out.append(f"Gain {s.gain_db:+g} dB")
    if s.normalize:
        out.append(f"Normalize to {s.target_lufs:g} LUFS")
    if s.limiter:
        out.append(f"Limiter {s.limiter_ceiling_db:g} dB")
    if s.fade_in or s.fade_out:
        out.append(f"Fades in {s.fade_in:g}s / out {s.fade_out:g}s")
    return out or ["Enabled, but no step is active."]


def problems(s: VoiceProcessingSettings) -> list[str]:
    out = []
    if not (-24 <= s.gain_db <= 24):
        out.append("Gain must be between -24 and +24 dB.")
    if s.compression and (s.comp_ratio < 1 or s.comp_ratio > 20):
        out.append("Compression ratio must be between 1 and 20.")
    if s.limiter and not (-12 <= s.limiter_ceiling_db <= 0):
        out.append("Limiter ceiling must be between -12 and 0 dB.")
    if s.highpass_hz < 0 or s.highpass_hz > 500:
        out.append("High-pass must be between 0 and 500 Hz.")
    if s.fade_in < 0 or s.fade_out < 0:
        out.append("Fade lengths cannot be negative.")
    if s.eq_preset not in ("none", *EQ_PRESETS):
        out.append(f"Unknown EQ preset “{s.eq_preset}”.")
    return out


class AudioProcessingService:
    def __init__(self, backend: AudioBackend) -> None:
        self.backend = backend

    @staticmethod
    def key(src_hash: str, s: VoiceProcessingSettings, duration: float | None) -> str:
        return hashlib.sha1(json.dumps([src_hash, asdict(s), duration], sort_keys=True).encode()).hexdigest()[:16]

    def render_preview(self, src: Path, cache_dir: Path, src_hash: str, s: VoiceProcessingSettings, duration: float | None = None) -> Path:
        """Processed copy in the cache (a preview only). Returns ``src`` itself when processing is off."""
        chain = build_chain(s, duration)
        if not chain:
            return src
        out = cache_dir / "audio" / f"voice_{self.key(src_hash, s, duration)}.wav"
        if out.is_file():
            return out
        return self.backend.render_chain(src, out, chain)
