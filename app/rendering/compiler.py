"""TimelineCompiler: the editable timeline -> a renderable description.

Pure data in, pure data out (no FFmpeg, no files). For a time window it produces, bottom to top, the video layers (with geometry as
``Curve`` expressions, so keyframes are interpolated exactly like the editor's preview) and the ASS overlay passes, and for the whole
timeline the audio mix items. The timeline is never modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from app.audio.mix import AudioMixService, ROLE_OF_TRACK
from app.rendering.ass import AssBuilder, AssDocument
from app.rendering.expressions import Curve, fnum, keyframe_curve, ramp
from app.rendering.fonts import FontResolver
from app.rendering.models import AssetRef, ChunkPlan, RenderSnapshot
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.track import Track, TrackKind

REDUCED_MOTION_FACTOR = 0.4  # same softening the editor's preview applies


@dataclass
class SourceInfo:
    path: str
    used_proxy: bool
    stream_w: int  # decoded size of the file actually read (a proxy is smaller than the original)
    stream_h: int


class SourceResolver(Protocol):
    def resolve(self, asset: AssetRef) -> SourceInfo: ...


@dataclass
class VideoLayer:
    layer_id: str
    clip_id: str
    track_id: str
    kind: str  # video | image | hold
    path: str
    used_proxy: bool
    t0: float  # first / last+1 frame times on the timeline (frame aligned)
    t1: float
    seek: float  # source time of the first frame (file time)
    src_span: float  # seconds of source to read
    speed: float
    box_w: float  # display size of the (cropped) original: geometry is always relative to this, so proxies look identical
    box_h: float
    stream_w: int
    stream_h: int
    crop: tuple[float, float, float, float] | None
    has_alpha: bool
    base: float  # canvas pixels per source pixel for the fit mode
    scale: Curve
    x: Curve
    y: Curve
    rot: Curve
    opacity: Curve
    reveal: Curve
    blur: float = 0.0
    fade_in: tuple[float, float] | None = None  # (start, duration) on the timeline
    fade_out: tuple[float, float] | None = None
    tpad: float = 0.0  # seconds of frozen last frame when the source is shorter than the clip
    wipe: bool = False

    @property
    def needs_mask(self) -> bool:
        return not self.opacity.is_const or not self.reveal.is_const

    @property
    def rotated(self) -> bool:
        return not self.rot.is_const or abs(self.rot.value or 0.0) > 1e-6  # type: ignore[arg-type]


@dataclass
class Step:
    kind: str  # layer | ass
    layer: VideoLayer | None = None


@dataclass
class CompiledChunk:
    chunk: ChunkPlan
    steps: list[Step] = field(default_factory=list)
    ass_groups: int = 0

    @property
    def layers(self) -> list[VideoLayer]:
        return [s.layer for s in self.steps if s.kind == "layer" and s.layer is not None]


@dataclass
class AudioItem:
    clip_id: str
    role: str
    track_id: str
    path: str
    start: float
    duration: float
    source_in: float
    speed: float
    gain: float
    fade_in: float
    fade_out: float
    pan: float
    keyframes: list
    is_voice: bool = False


@dataclass
class AudioPlan:
    items: list[AudioItem]
    duration: float
    soloed: bool = False

    @property
    def audible(self) -> bool:
        return bool(self.items)


class TimelineCompiler:
    def __init__(self, snapshot: RenderSnapshot, resolver: SourceResolver, fonts: FontResolver, out_fps: int) -> None:
        self.s = snapshot
        self.resolver = resolver
        self.fps = out_fps
        self.W, self.H = snapshot.canvas_w, snapshot.canvas_h
        self.warnings: list[str] = []
        self.reduced = bool(getattr(snapshot.caption_settings, "reduced_motion", False))
        # overlays are generated once for the whole timeline; chunks take the events that overlap them
        self._ass = self._build_ass(fonts)

    # ------------------------------------------------------------------ helpers
    def snap(self, t: float) -> float:
        return round(t * self.fps) / self.fps

    def _visible_tracks(self) -> list[tuple[int, Track]]:
        return [(i, t) for i, t in enumerate(self.s.tracks) if t.kind is not TrackKind.AUDIO and not t.hidden]

    def _build_ass(self, fonts: FontResolver) -> AssDocument:
        b = AssBuilder(self.s, fonts, self.fps)
        clips = [(i, c) for i, t in self._visible_tracks() for c in t.clips if c.kind in (KIND_TEXT, KIND_GRAPHIC, KIND_CAPTION)]
        return b.build(clips)

    @property
    def ass(self) -> AssDocument:
        return self._ass

    # ------------------------------------------------------------------ chunk-independent statistics
    def media_clips_in(self, w0: float, w1: float) -> int:
        n = 0
        for _i, t in self._visible_tracks():
            for c in t.clips:
                if c.kind == KIND_MEDIA and c.timeline_end > w0 and c.timeline_start < w1:
                    n += 1
        return n

    # ------------------------------------------------------------------ video
    def compile_chunk(self, chunk: ChunkPlan) -> CompiledChunk:
        out = CompiledChunk(chunk)
        w0, w1 = chunk.start, chunk.end
        pending_ass = False
        for _i, t in self._visible_tracks():
            has_overlay = False
            for c in t.clips:
                if c.timeline_end <= w0 or c.timeline_start >= w1:
                    continue
                if c.kind == KIND_MEDIA:
                    for layer in self._media_layers(t, c, w0, w1):
                        if pending_ass:
                            out.steps.append(Step("ass"))
                            out.ass_groups += 1
                            pending_ass = False
                        out.steps.append(Step("layer", layer))
                elif c.kind in (KIND_TEXT, KIND_GRAPHIC, KIND_CAPTION):
                    has_overlay = True
            if has_overlay:
                pending_ass = True
        if pending_ass:
            out.steps.append(Step("ass"))
            out.ass_groups += 1
        return out

    def _media_layers(self, track: Track, c: Clip, w0: float, w1: float) -> list[VideoLayer]:
        asset = self.s.assets.get(c.asset_id)
        if asset is None or asset.type not in ("video", "image"):
            self.warnings.append(f"Clip {c.id} was skipped: its media is not a video or image.")
            return []
        t0, t1 = max(self.snap(c.timeline_start), w0), min(self.snap(c.timeline_end), w1)
        if t1 - t0 < 0.5 / self.fps:
            return []
        src = self.resolver.resolve(asset)
        kind = "image" if asset.type == "image" else "video"
        crop = _crop(c.effects.get("crop"))
        aw, ah = float(asset.width or src.stream_w), float(asset.height or src.stream_h)
        box_w, box_h = aw * (crop[2] if crop else 1.0), ah * (crop[3] if crop else 1.0)
        sw, sh = int(src.stream_w * (crop[2] if crop else 1.0)), int(src.stream_h * (crop[3] if crop else 1.0))
        fit = str(c.effects.get("fit", "cover"))
        if fit == "stretch":
            self.warnings.append(f"Clip {c.id}: stretching is not allowed; the clip is filled without distortion instead.")
        fx, fy = self.W / max(1.0, box_w), self.H / max(1.0, box_h)
        base = min(fx, fy) if fit in ("contain", "fit") else max(fx, fy)
        kf = c.keyframes
        KS = keyframe_curve(kf, "scale", c.timeline_start, 1.0)
        KX = keyframe_curve(kf, "position_x", c.timeline_start, 0.0)
        KY = keyframe_curve(kf, "position_y", c.timeline_start, 0.0)
        if self.reduced:  # accessibility: strong zooms and drifts (the keyframed motion) are softened, as in the preview; the clip's own scale/position are not
            KS = Curve.const(1.0 + (KS.value - 1.0) * REDUCED_MOTION_FACTOR) if KS.is_const else KS.map("(1+({}-1)*" + fnum(REDUCED_MOTION_FACTOR) + ")")  # type: ignore[operator]
            KX, KY = KX.scaled(REDUCED_MOTION_FACTOR), KY.scaled(REDUCED_MOTION_FACTOR)
        S = Curve.const(c.scale).times(KS)
        X = Curve.const(c.position[0]).plus(KX)
        Y = Curve.const(c.position[1]).plus(KY)
        R = Curve.const(c.rotation).plus(keyframe_curve(kf, "rotation", c.timeline_start, 0.0))
        O = Curve.const(c.opacity).times(keyframe_curve(kf, "opacity", c.timeline_start, 1.0))
        RV = keyframe_curve(kf, "reveal", c.timeline_start, 1.0)
        blur_vals = [k.value for k in kf if k.property == "blur"]
        blur = max(blur_vals) if blur_vals else 0.0
        if blur_vals and len({round(v, 6) for v in blur_vals}) > 1:
            self.warnings.append(f"Clip {c.id}: blur keyframes are rendered as one constant blur ({blur:g}).")
        tr = c.transition
        fade_in = wipe = None
        hold: VideoLayer | None = None
        if tr and str(tr.get("type", "CUT")) != "CUT" and float(tr.get("duration", 0.0)) > 0:
            d = min(float(tr["duration"]), c.duration)
            a = c.timeline_start
            kind_t = str(tr["type"])
            if kind_t in ("FADE", "DISSOLVE"):
                fade_in = (a, d)
                if kind_t == "DISSOLVE":
                    hold = self._hold_layer(c, w0, w1, a, d)
            elif kind_t == "WIPE":
                RV = RV.times(Curve(None, ramp(a, d)))
                wipe = True
            elif kind_t == "SLIDE":
                X = X.plus(Curve(None, f"((1-{ramp(a, d)})*{fnum(self.W)})"))
        seek = max(0.0, c.source_in + (t0 - c.timeline_start) * c.speed)
        span = (t1 - t0) * c.speed + 2.5 / self.fps
        tpad = 0.0
        if kind == "video" and asset.duration:
            need_end = seek + (t1 - t0) * c.speed
            if need_end > asset.duration - 0.02:
                tpad = need_end - asset.duration + 0.5
                seek = min(seek, max(0.0, asset.duration - 0.1))
        layer = VideoLayer(c.id, c.id, track.id, kind, src.path, src.used_proxy, t0, t1, seek, span, c.speed, box_w, box_h, sw, sh, crop, asset.has_alpha, base,
                           S, X, Y, R, O, RV, blur, fade_in, None, tpad, bool(wipe))
        return ([hold] if hold else []) + [layer]

    def _hold_layer(self, c: Clip, w0: float, w1: float, a: float, d: float) -> VideoLayer | None:
        """DISSOLVE: the previous shot's last frame, held and faded out while the new shot fades in (the same look as the editor's preview)."""
        prev = None
        for _i, t in self._visible_tracks():
            for o in t.clips:
                if o.kind == KIND_MEDIA and o.id != c.id and abs(o.timeline_end - c.timeline_start) < 0.05 and (prev is None or o.timeline_end > prev.timeline_end):
                    prev = o
        if prev is None:
            return None
        asset = self.s.assets.get(prev.asset_id)
        if asset is None or asset.type not in ("video", "image"):
            return None
        t0, t1 = max(self.snap(a), w0), min(self.snap(a + d), w1)
        if t1 - t0 < 0.5 / self.fps:
            return None
        src = self.resolver.resolve(asset)
        crop = _crop(prev.effects.get("crop"))
        aw, ah = float(asset.width or src.stream_w), float(asset.height or src.stream_h)
        box_w, box_h = aw * (crop[2] if crop else 1.0), ah * (crop[3] if crop else 1.0)
        fit = str(prev.effects.get("fit", "cover"))
        fx, fy = self.W / max(1.0, box_w), self.H / max(1.0, box_h)
        base = min(fx, fy) if fit in ("contain", "fit") else max(fx, fy)
        end = prev.duration - 1.0 / self.fps
        from app.timeline.keyframes import value_at

        S = prev.scale * value_at(prev.keyframes, "scale", end)
        X, Y = prev.position[0] + value_at(prev.keyframes, "position_x", end), prev.position[1] + value_at(prev.keyframes, "position_y", end)
        seek = max(0.0, prev.source_in + end * prev.speed) if asset.type == "video" else 0.0  # a still has one frame: nothing to seek to
        if asset.duration and asset.type == "video":
            seek = min(seek, max(0.0, asset.duration - 0.1))
        sw, sh = int(src.stream_w * (crop[2] if crop else 1.0)), int(src.stream_h * (crop[3] if crop else 1.0))
        return VideoLayer(c.id + "_hold", prev.id, "", "hold", src.path, src.used_proxy, t0, t1, seek, 0.5, 1.0, box_w, box_h, sw, sh, crop, asset.has_alpha, base,
                          Curve.const(S), Curve.const(X), Curve.const(Y), Curve.const(prev.rotation), Curve.const(prev.opacity), Curve.const(1.0), 0.0, None,
                          (self.snap(a), d), 0.0)

    # ------------------------------------------------------------------ audio
    def audio_plan(self) -> AudioPlan:
        """Every audible clip with the gain, fades and (ducking) volume keyframes the timeline gives it. Mute / solo / track volume are applied here."""
        audio_tracks = [t for t in self.s.tracks if t.kind is TrackKind.AUDIO]
        soloed = any(t.solo for t in audio_tracks)
        items: list[AudioItem] = []
        for t in audio_tracks:
            if t.muted or t.hidden or (soloed and not t.solo):
                continue
            for c in t.clips:
                asset = self.s.assets.get(c.asset_id) if c.kind == KIND_MEDIA else None
                if asset is None or c.duration <= 0.0:
                    continue
                if asset.type == "image":
                    continue
                role = str(c.audio.get("role") or ROLE_OF_TRACK.get(t.id, "OTHER")).upper()
                items.append(AudioItem(c.id, role, t.id, asset.path, c.timeline_start, c.duration,
                                       c.source_in, c.speed, float(c.audio.get("volume", 1.0)) * t.volume, float(c.audio.get("fade_in", 0.0)),
                                       float(c.audio.get("fade_out", 0.0)), float(c.audio.get("pan", 0.0)), [k for k in c.keyframes if k.property == "volume"],
                                       role == "VOICE" or asset.id == self.s.voice_asset_id))
        items.sort(key=lambda i: (i.start, i.role))
        return AudioPlan(items, self.s.duration, soloed)


def _crop(v) -> tuple[float, float, float, float] | None:
    try:
        x, y, w, h = (float(n) for n in v)
    except (TypeError, ValueError):
        return None
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1) or x + w > 1.0001 or y + h > 1.0001:
        return None
    return (x, y, w, h)


def window_frames(w0: float, w1: float, fps: int) -> int:
    return max(1, int(round((w1 - w0) * fps)))


_ = math
