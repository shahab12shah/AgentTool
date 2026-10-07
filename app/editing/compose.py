"""PreviewComposer: evaluates the timeline at a time ``t`` (visuals, motion, text, transitions, music instructions).

Qt-free and render-free: it answers "what is on screen and what are the audio levels right now" from the timeline data, which is what a
scrubbing preview needs. It is not the final renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.editing.models import DecisionType
from app.media.asset import AssetType
from app.timeline.clip import Clip
from app.timeline.keyframes import value_at
from app.timeline.track import TrackKind


@dataclass
class Layer:
    kind: str  # media | text | graphic
    clip_id: str
    track_id: str
    asset_id: str = ""
    src_time: float = 0.0
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    blur: float = 0.0
    reveal: float = 1.0  # 0..1 of the width revealed (wipe)
    fit: str = "cover"
    text: dict | None = None
    highlight: dict | None = None
    role: str = "main"  # main | outgoing
    is_still: bool = False
    locked: bool = False
    created_by: str = "USER"


@dataclass
class FrameState:
    time: float
    scene_id: str = ""
    layers: list[Layer] = field(default_factory=list)
    voice_active: bool = False
    music_level: float = 0.0
    transition: str = ""  # transition currently running ("" = none)
    active_decisions: list[str] = field(default_factory=list)


class PreviewComposer:
    def __init__(self, project) -> None:
        self.p = project

    # ------------------------------------------------------------------ audio instructions
    def music_level_at(self, t: float) -> float:
        audio = self.p.editing_strategy.audio
        level = audio.music_level
        fade = 1.0
        for d in self.p.editing_decisions.values():
            if d.type is not DecisionType.AUDIO_DUCK:
                continue
            a, b = float(d.parameters.get("start", d.start)), float(d.parameters.get("end", d.start + d.duration))
            kind, target, ramp = d.parameters.get("kind", "DUCK"), float(d.parameters.get("music_level", level)), float(d.parameters.get("ramp", 0.3))
            if kind == "FADE_IN" and a <= t <= b and b > a:
                fade = min(fade, (t - a) / (b - a))
            elif kind == "FADE_IN" and t < a:
                fade = 0.0
            elif kind == "FADE_OUT" and t >= a:
                fade = min(fade, max(0.0, 1.0 - (t - a) / max(1e-6, b - a)))
            elif kind in ("DUCK", "RISE") and a - ramp <= t <= b + ramp:
                edge = 1.0 if a <= t <= b else max(0.0, 1.0 - (a - t if t < a else t - b) / max(1e-6, ramp))
                level = level + (target - level) * edge
        return round(level * fade, 4)

    # ------------------------------------------------------------------ frame
    def frame_at(self, t: float) -> FrameState:
        tl, assets = self.p.timeline, self.p.assets
        fs = FrameState(t)
        for sc in self.p.scenes:
            if sc.start <= t < sc.end:
                fs.scene_id = sc.id
                break
        for track in tl.tracks:
            if track.hidden:
                continue
            if track.kind is TrackKind.AUDIO:
                for c in track.clips:
                    if c.timeline_start <= t < c.timeline_end and c.asset_id == self.p.voice_over.asset_id and not track.muted:
                        fs.voice_active = True
                continue
            for c in track.clips:
                if c.timeline_start <= t < c.timeline_end:
                    fs.layers.extend(self._layers(c, track.id, t))
                    if c.ai_decision_id:
                        fs.active_decisions.append(c.ai_decision_id)
                    if c.transition and c.transition.get("type", "CUT") != "CUT" and t - c.timeline_start < float(c.transition.get("duration", 0)):
                        fs.transition = str(c.transition["type"])
        fs.music_level = self.music_level_at(t)
        return fs

    def _layers(self, c: Clip, track_id: str, t: float) -> list[Layer]:
        asset = self.p.assets.get(c.asset_id) if c.asset_id else None
        local = t - c.timeline_start
        kf = c.keyframes
        scale = c.scale * value_at(kf, "scale", local)
        x = c.position[0] + value_at(kf, "position_x", local)
        y = c.position[1] + value_at(kf, "position_y", local)
        op = c.opacity * value_at(kf, "opacity", local)
        layer = Layer(c.kind, c.id, track_id, c.asset_id, c.source_in + local * c.speed, x, y, scale, c.rotation + value_at(kf, "rotation", local), op,
                      value_at(kf, "blur", local), 1.0, c.effects.get("fit", "cover"), c.text, (c.effects.get("highlight") if c.kind == "graphic" else None),
                      "main", bool(asset and asset.type is AssetType.IMAGE), c.locked, c.created_by)
        out = [layer]
        if c.kind in ("text", "graphic"):  # fade in/out of overlays
            d = float(c.animation.get("duration", 0.25)) if c.animation else 0.25
            layer.opacity *= min(1.0, local / d if d > 0 else 1.0, (c.duration - local) / d if d > 0 else 1.0)
        tr = c.transition
        if tr and tr.get("type", "CUT") != "CUT" and float(tr.get("duration", 0)) > 0 and local < float(tr["duration"]):
            p = local / float(tr["duration"])
            kind = tr["type"]
            W = self.p.settings.width
            if kind in ("FADE", "DISSOLVE"):
                layer.opacity *= p
            elif kind == "WIPE":
                layer.reveal = p
            elif kind == "SLIDE":
                layer.x += (1 - p) * W
            if kind == "DISSOLVE":
                prev = self._previous(c)
                if prev is not None:
                    pl = self._layers_static(prev)
                    pl.opacity *= (1 - p)
                    pl.role = "outgoing"
                    out.insert(0, pl)
        return out

    def _previous(self, c: Clip) -> Clip | None:
        best = None
        for t in self.p.timeline.tracks:
            if t.kind is TrackKind.AUDIO:
                continue
            for o in t.clips:
                if o.kind == "media" and o.id != c.id and abs(o.timeline_end - c.timeline_start) < 0.05 and (best is None or o.timeline_end > best.timeline_end):
                    best = o
        return best

    def _layers_static(self, c: Clip) -> Layer:
        asset = self.p.assets.get(c.asset_id)
        end = c.duration - 1e-3
        return Layer(c.kind, c.id, c.track_id, c.asset_id, c.source_in + end * c.speed, c.position[0] + value_at(c.keyframes, "position_x", end),
                     c.position[1] + value_at(c.keyframes, "position_y", end), c.scale * value_at(c.keyframes, "scale", end), c.rotation,
                     c.opacity, 0.0, 1.0, c.effects.get("fit", "cover"), None, None, "outgoing", bool(asset and asset.type is AssetType.IMAGE))
