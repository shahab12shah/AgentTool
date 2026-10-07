"""RenderPlanner: decides *what will be done* before FFmpeg starts — output size, encoder, chunks and the human-readable plan."""

from __future__ import annotations

import math
from dataclasses import replace

from app.project.project_schema import RenderSettings
from app.rendering import presets as P
from app.rendering.errors import EncoderUnavailableError, RenderError
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.models import ChunkPlan, RenderPlan, RenderSnapshot, has_audio_clips
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT
from app.timeline.track import TrackKind

MAX_LAYERS_PER_CHUNK = 40
FINAL_CHUNK_SECONDS = 30.0
PREVIEW_CHUNK_SECONDS = 12.0


class EncoderSelector:
    """Settings -> a concrete encoder, never silently changing the user's codec choice."""

    def __init__(self, ffmpeg: FFmpegService) -> None:
        self.ff = ffmpeg

    def resolve(self, s: RenderSettings, snapshot: RenderSnapshot, *, force_cpu: bool = False) -> P.ResolvedOutput:
        problems = P.compatibility_problems(s)
        if problems:
            raise RenderError(problems[0], stage="Validating Project", kind="invalid_settings", details="; ".join(problems))
        caps = self.ff.capabilities()
        notes: list[str] = []
        codec = s.video_codec
        soft = next((e for e in P.SOFTWARE_ENCODERS[codec] if e in caps.encoders), "")
        mode = "cpu" if force_cpu else s.hardware_acceleration
        hw_enc = ""
        if mode in ("auto", "hardware"):
            working = self.ff.hardware_encoders(P.HARDWARE_ENCODERS.get(codec, []))
            hw_enc = next((e for e in P.HARDWARE_ENCODERS.get(codec, []) if working.get(e)), "")
            if not hw_enc:
                if mode == "hardware":
                    raise RenderError(f"No working hardware encoder was found for {P.CODEC_LABELS[codec]} on this computer.", stage="Validating Project", kind="hardware_unavailable",
                                      possible_issue="The GPU or its driver is not available to FFmpeg.", can_fallback_cpu=bool(soft))
                notes.append("Hardware encoding is not available here; using the CPU encoder.")
        use_hw = bool(hw_enc)
        encoder = hw_enc if use_hw else soft
        if not encoder:
            alts = [P.CODEC_LABELS[c] for c in P.CODECS if any(e in caps.encoders for e in P.SOFTWARE_ENCODERS[c]) and c != codec]
            raise EncoderUnavailableError(f"{P.CODEC_LABELS[codec]} encoding is not available in this FFmpeg build.", alts,
                                          possible_issue="FFmpeg was built without this encoder.", details=f"missing encoders: {P.SOFTWARE_ENCODERS[codec]}")
        a_enc = {"aac": "aac", "opus": "libopus", "flac": "flac"}[s.audio_codec]
        if a_enc not in caps.encoders:
            alts = [P.AUDIO_LABELS[a] for a, e in (("aac", "aac"), ("opus", "libopus"), ("flac", "flac")) if e in caps.encoders and a != s.audio_codec]
            raise EncoderUnavailableError(f"{P.AUDIO_LABELS[s.audio_codec]} audio encoding is not available in this FFmpeg build.", alts, kind="unsupported_codec")
        w, h = P.output_size(snapshot.canvas_w, snapshot.canvas_h, s.resolution)
        fps = s.fps or snapshot.fps
        q = s.quality
        base_q = "high" if q == "custom" else q
        if use_hw:
            crf = s.crf if (q == "custom" and s.crf) else P.HW_QUALITY[base_q]
            preset = s.encoder_preset
        else:
            crf = s.crf if (q == "custom" and s.crf) else P.CRF[encoder][base_q]
            preset = s.encoder_preset or P.SPEED_PRESET[encoder][base_q]
        return P.ResolvedOutput(w, h, fps, s.container, codec, encoder, use_hw, s.audio_codec, a_enc, s.audio_bitrate_kbps, s.audio_sample_rate, q, "yuv420p", crf,
                                s.bitrate_kbps if q == "custom" else 0, preset, notes)


def plan_chunks(snapshot: RenderSnapshot, fps: int, target_seconds: float, layer_counter=None, max_seconds: float | None = None) -> list[ChunkPlan]:
    """Split the timeline at scene boundaries into windows of about ``target_seconds`` (frame aligned).

    Each window is rendered by its own FFmpeg run and cached by content, so an edit only re-renders the windows it touches and a failure late in
    the render does not repeat the early ones. Windows also bound how many input files one FFmpeg process opens.
    """
    total_frames = max(1, int(round(snapshot.duration * fps)))
    edges = {0, total_frames}
    for sc in snapshot.scenes:
        f = int(round(sc.start * fps))
        if 0 < f < total_frames:
            edges.add(f)
    bounds = sorted(edges)
    chosen = [0]
    for f in bounds[1:]:
        span = (f - chosen[-1]) / fps
        too_big = layer_counter is not None and layer_counter(chosen[-1] / fps, f / fps) > MAX_LAYERS_PER_CHUNK
        if f == total_frames:
            chosen.append(f)
        elif span >= target_seconds or too_big:
            chosen.append(f)
    if chosen[-1] != total_frames:
        chosen.append(total_frames)
    # a very long window without scene boundaries is cut into regular pieces
    final: list[int] = [chosen[0]]
    for f in chosen[1:]:
        cap = max_seconds or target_seconds * 2.0
        while (f - final[-1]) / fps > cap:
            final.append(final[-1] + int(min(cap, max(target_seconds, 1.0)) * fps))
        final.append(f)
    out = []
    for i, (a, b) in enumerate(zip(final, final[1:])):
        scene_ids = [sc.id for sc in snapshot.scenes if sc.start * fps < b and sc.end * fps > a]
        out.append(ChunkPlan(i, a / fps, b / fps, b - a, scene_ids))
    return out


def make_plan(snapshot: RenderSnapshot, resolved: P.ResolvedOutput, chunks: list[ChunkPlan]) -> RenderPlan:
    vt = [{"id": t.id, "name": t.name, "kind": t.kind.value, "clips": len(t.clips)} for t in snapshot.tracks if t.kind is not TrackKind.AUDIO and t.clips]
    at = [{"id": t.id, "name": t.name, "clips": len(t.clips), "muted": t.muted, "volume": t.volume} for t in snapshot.tracks if t.kind is TrackKind.AUDIO and t.clips]
    graphics = captions = 0
    effects: set[str] = set()
    transitions: dict[str, int] = {}
    for t, c in snapshot.clips():
        if t.kind is TrackKind.AUDIO:
            continue
        if c.kind in (KIND_TEXT, KIND_GRAPHIC):
            graphics += 1
        elif c.kind == KIND_CAPTION:
            captions += 1
        if c.kind == KIND_MEDIA:
            if c.keyframes:
                effects.add("keyframes: " + ", ".join(sorted({k.property for k in c.keyframes})))
            if abs(c.speed - 1.0) > 1e-9:
                effects.add("speed change")
            if abs(c.rotation) > 1e-9:
                effects.add("rotation")
            if c.opacity < 0.999:
                effects.add("opacity")
            if c.effects.get("crop"):
                effects.add("crop")
            if c.transition and str(c.transition.get("type", "CUT")) != "CUT":
                transitions[str(c.transition["type"])] = transitions.get(str(c.transition["type"]), 0) + 1
    return RenderPlan(snapshot.project_id, snapshot.timeline_version, (resolved.width, resolved.height), resolved.fps, snapshot.duration, vt, at, graphics, captions,
                      sorted(effects), [f"{k} ×{v}" for k, v in sorted(transitions.items())], resolved.container, resolved.quality, resolved.video_codec, resolved.encoder,
                      resolved.hardware, resolved.audio_codec, sum(c.frames for c in chunks), chunks, [], has_audio_clips(snapshot), [], list(resolved.notes))


_ = (replace, math)


class RenderPlanner:
    """Decides what will be done before FFmpeg starts: the output (size, fps, encoder), the scene-aligned sections and the human-readable ``RenderPlan``."""

    def __init__(self, selector: EncoderSelector) -> None:
        self.selector = selector

    def resolve(self, settings: RenderSettings, snapshot: RenderSnapshot, force_cpu: bool = False) -> P.ResolvedOutput:
        return self.selector.resolve(settings, snapshot, force_cpu=force_cpu)

    def sections(self, snapshot: RenderSnapshot, fps: int, target_seconds: float = FINAL_CHUNK_SECONDS, layer_counter=None, max_seconds: float | None = None) -> list[ChunkPlan]:
        return plan_chunks(snapshot, fps, target_seconds, layer_counter, max_seconds)

    def plan(self, snapshot: RenderSnapshot, settings: RenderSettings, force_cpu: bool = False) -> tuple[RenderPlan, P.ResolvedOutput]:
        resolved = self.resolve(settings, snapshot, force_cpu)
        return make_plan(snapshot, resolved, self.sections(snapshot, resolved.fps)), resolved
