"""AudioMixService: the mix as data (what plays when, at which level) plus a *preview* render of it.

``MixPlan`` is derived from the timeline clips — track mute/solo/volume, clip gain, fades and volume keyframes — and nothing else.
``render`` turns a plan into a preview WAV in the cache (voice only, music only, SFX only, combinations or the full mix).
It is not the final export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.audio.backend import AudioBackend
from app.presentation.models import PreviewMode
from app.timeline.keyframes import value_at
from app.timeline.track import TrackKind

ROLE_OF_TRACK = {"track_a1": "VOICE", "track_a2": "MUSIC", "track_a3": "SFX"}
MODE_ROLES = {
    PreviewMode.VOICE: {"VOICE"}, PreviewMode.MUSIC: {"MUSIC"}, PreviewMode.SFX: {"SFX"}, PreviewMode.VOICE_MUSIC: {"VOICE", "MUSIC"},
    PreviewMode.VOICE_SFX: {"VOICE", "SFX"}, PreviewMode.FULL: {"VOICE", "MUSIC", "SFX", "OTHER"},
}


@dataclass
class MixItem:
    clip_id: str
    role: str
    track_id: str
    path: Path
    start: float
    duration: float
    source_in: float
    speed: float
    gain: float  # clip volume x track volume
    fade_in: float = 0.0
    fade_out: float = 0.0
    keyframes: list = field(default_factory=list)  # Keyframe objects (property "volume", clip-local time)

    @property
    def end(self) -> float:
        return self.start + self.duration

    def gain_at(self, t: float) -> float:
        """Effective linear gain at timeline time ``t`` (0 outside the item)."""
        if not (self.start <= t < self.end):
            return 0.0
        local = t - self.start
        g = self.gain * value_at(self.keyframes, "volume", local)
        if self.fade_in > 0 and local < self.fade_in:
            g *= local / self.fade_in
        if self.fade_out > 0 and self.duration - local < self.fade_out:
            g *= max(0.0, (self.duration - local) / self.fade_out)
        return g


@dataclass
class MixPlan:
    mode: PreviewMode
    items: list[MixItem]
    duration: float
    soloed: bool = False

    def gain_at(self, role: str, t: float) -> float:
        return sum(i.gain_at(t) for i in self.items if i.role == role)

    def signature(self) -> str:
        data = [(i.clip_id, str(i.path), i.start, i.duration, i.source_in, i.speed, i.gain, i.fade_in, i.fade_out, [(k.time, k.value, k.interpolation) for k in i.keyframes])
                for i in self.items]
        return hashlib.sha1(json.dumps([self.mode.value, self.duration, data], default=str).encode()).hexdigest()[:16]


def role_of(track, clip) -> str:
    return str(clip.audio.get("role") or ROLE_OF_TRACK.get(track.id, "OTHER")).upper()


class AudioMixService:
    def __init__(self, backend: AudioBackend) -> None:
        self.backend = backend

    # ------------------------------------------------------------------ plan
    def build_plan(self, project, mode: PreviewMode = PreviewMode.FULL, voice_path: Path | None = None) -> MixPlan:
        tl = project.timeline
        audio_tracks = [t for t in tl.tracks if t.kind is TrackKind.AUDIO]
        soloed = any(t.solo for t in audio_tracks)
        roles = MODE_ROLES[mode]
        items: list[MixItem] = []
        for t in audio_tracks:
            if t.muted or t.hidden or (soloed and not t.solo):
                continue
            for c in t.clips:
                role = role_of(t, c)
                asset = project.assets.get(c.asset_id) if c.kind == "media" else None
                if asset is None or role not in roles:
                    continue
                path = voice_path if (voice_path and role == "VOICE" and asset.id == project.voice_over.asset_id) else project.asset_path(asset)
                items.append(MixItem(c.id, role, t.id, path, c.timeline_start, c.duration, c.source_in, c.speed,
                                     float(c.audio.get("volume", 1.0)) * t.volume, float(c.audio.get("fade_in", 0.0)), float(c.audio.get("fade_out", 0.0)),
                                     [k for k in c.keyframes if k.property == "volume"]))
        duration = max((i.end for i in items), default=0.0)
        return MixPlan(mode, sorted(items, key=lambda i: (i.start, i.role)), duration, soloed)

    # ------------------------------------------------------------------ ffmpeg graph
    @staticmethod
    def volume_expr(gain: float, kfs: list, offset: float = 0.0) -> str:
        """Flat (non-nested) piecewise-linear volume expression in clip-local time; ``offset`` shifts the window start."""
        T = f"(max(t,0)+{offset:.4f})" if offset else "max(t,0)"  # max(t,0) also turns FFmpeg's start-up NaN into 0
        pts = sorted(((k.time, k.value) for k in kfs), key=lambda p: p[0])
        if not pts:
            return f"{gain:.5f}"
        terms = [f"lt({T},{pts[0][0]:.4f})*{pts[0][1]:.5f}"]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t1 - t0 < 1e-6:
                continue
            terms.append(f"gte({T},{t0:.4f})*lt({T},{t1:.4f})*({v0:.5f}+({v1 - v0:.5f})*({T}-{t0:.4f})/{t1 - t0:.4f})")
        terms.append(f"gte({T},{pts[-1][0]:.4f})*{pts[-1][1]:.5f}")
        return f"{gain:.5f}*(" + "+".join(terms) + ")"

    def graph(self, plan: MixPlan, w0: float, w1: float) -> tuple[str, list[Path]]:
        inputs: list[Path] = []
        chains: list[str] = []
        labels: list[str] = []
        for it in plan.items:
            a, b = max(it.start, w0), min(it.end, w1)
            if b - a < 0.02:
                continue
            skip = a - it.start
            i = len(inputs)
            inputs.append(it.path)
            s_in = it.source_in + skip * it.speed
            s_out = s_in + (b - a) * it.speed
            parts = [f"[{i}:a]atrim=start={s_in:.4f}:end={s_out:.4f}", "asetpts=PTS-STARTPTS"]
            if abs(it.speed - 1.0) > 1e-6:
                parts.append(f"atempo={min(2.0, max(0.5, it.speed)):.4f}")
            parts.append(f"volume='{self.volume_expr(it.gain, it.keyframes, skip)}':eval=frame")
            local_dur = b - a
            if it.fade_in > 0 and skip < it.fade_in:
                parts.append(f"afade=t=in:st=0:d={max(0.01, it.fade_in - skip):.3f}")
            if it.fade_out > 0 and local_dur > 0:
                st = max(0.0, (it.duration - it.fade_out) - skip)
                if st < local_dur:
                    parts.append(f"afade=t=out:st={st:.3f}:d={min(it.fade_out, local_dur - st):.3f}")
            delay = int(round((a - w0) * 1000))
            parts.append(f"adelay={delay}|{delay}")
            labels.append(f"[m{i}]")
            chains.append(",".join(parts) + f"[m{i}]")
        if not chains:
            return "", inputs
        mix = "".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,alimiter=limit=0.97[out]"
        return ";".join(chains + [mix]), inputs

    def render(self, plan: MixPlan, cache_dir: Path, start: float = 0.0, end: float | None = None, progress=None) -> Path | None:
        """Preview WAV of the plan (cached by its signature). ``None`` when nothing in the plan is audible."""
        w1 = plan.duration if end is None else min(end, plan.duration)
        if w1 - start < 0.05 or not plan.items:
            return None
        out = cache_dir / "audio" / f"mix_{plan.signature()}_{start:.2f}_{w1:.2f}.wav"
        if out.is_file():
            return out
        if progress:
            progress(20, "Building the mix")
        g, inputs = self.graph(plan, start, w1)
        if not g:
            return None
        res = self.backend.render_graph(inputs, g, "out", out, w1 - start)
        if progress:
            progress(100, "Done")
        return res
