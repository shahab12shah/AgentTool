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
    word_index: int = -1  # captions: the word being spoken at this time (-1 = none yet)
    progress: float = 1.0  # counters: 0..1
    scale_text: float = 1.0
    counter_text: str | None = None
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
    audio: dict = field(default_factory=dict)  # effective linear levels per role at this time (VOICE / MUSIC / SFX)
    active_decisions: list[str] = field(default_factory=list)


class PreviewComposer:
    def __init__(self, project) -> None:
        self.p = project

    # ------------------------------------------------------------------ audio instructions
    def music_level_at(self, t: float) -> float:
        clips = [c for tr in self.p.timeline.tracks for c in tr.clips if tr.kind is TrackKind.AUDIO and c.audio.get("role") == "MUSIC" and not tr.muted]
        if clips:  # Phase 5: the real music clips (gain, fades, volume keyframes)
            total = 0.0
            for c in clips:
                if c.timeline_start <= t < c.timeline_end:
                    local = t - c.timeline_start
                    g = float(c.audio.get("volume", 1.0)) * value_at(c.keyframes, "volume", local)
                    fi, fo = float(c.audio.get("fade_in", 0)), float(c.audio.get("fade_out", 0))
                    if fi > 0 and local < fi:
                        g *= local / fi
                    if fo > 0 and c.duration - local < fo:
                        g *= max(0.0, (c.duration - local) / fo)
                    total += g
            return round(total, 4)
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
        fs.audio = self._audio_levels(t)
        return fs

    def _audio_levels(self, t: float) -> dict:
        out = {"VOICE": 0.0, "MUSIC": 0.0, "SFX": 0.0}
        solo = any(tr.solo for tr in self.p.timeline.tracks if tr.kind is TrackKind.AUDIO)
        for tr in self.p.timeline.tracks:
            if tr.kind is not TrackKind.AUDIO or tr.muted or (solo and not tr.solo):
                continue
            for c in tr.clips:
                if c.kind != "media" or not (c.timeline_start <= t < c.timeline_end):
                    continue
                role = str(c.audio.get("role") or {"track_a1": "VOICE", "track_a2": "MUSIC", "track_a3": "SFX"}.get(tr.id, "OTHER")).upper()
                local = t - c.timeline_start
                g = float(c.audio.get("volume", 1.0)) * tr.volume * value_at(c.keyframes, "volume", local)
                fi, fo = float(c.audio.get("fade_in", 0)), float(c.audio.get("fade_out", 0))
                if fi > 0 and local < fi:
                    g *= local / fi
                if fo > 0 and c.duration - local < fo:
                    g *= max(0.0, (c.duration - local) / fo)
                if role in out:
                    out[role] += g
        return {k: round(v, 4) for k, v in out.items()}

    def _layers(self, c: Clip, track_id: str, t: float) -> list[Layer]:
        asset = self.p.assets.get(c.asset_id) if c.asset_id else None
        local = t - c.timeline_start
        kf = c.keyframes
        reduced = bool(getattr(self.p, "caption_settings", None) and self.p.caption_settings.reduced_motion)
        k_scale, k_x, k_y = value_at(kf, "scale", local), value_at(kf, "position_x", local), value_at(kf, "position_y", local)
        if reduced and c.kind == "media":  # accessibility: strong zooms and drifts (the keyframed motion) are softened; the clip's own scale and position are not
            k_scale, k_x, k_y = 1.0 + (k_scale - 1.0) * 0.4, k_x * 0.4, k_y * 0.4
        scale = c.scale * k_scale
        x = c.position[0] + k_x
        y = c.position[1] + k_y
        op = c.opacity * value_at(kf, "opacity", local)
        layer = Layer(c.kind, c.id, track_id, c.asset_id, c.source_in + local * c.speed, x, y, scale, c.rotation + value_at(kf, "rotation", local), op,
                      value_at(kf, "blur", local), 1.0, c.effects.get("fit", "cover"), c.text, (c.effects.get("highlight") if c.kind == "graphic" else None),
                      "main", bool(asset and asset.type is AssetType.IMAGE), c.locked, c.created_by)
        out = [layer]
        if c.kind in ("text", "graphic", "caption"):
            from app.presentation.animation import animation_state, counter_text

            st = animation_state(c.animation, local, c.duration, float(self.p.settings.height))
            k = self.p.settings.width / 1920.0
            layer.opacity *= st.opacity
            layer.scale *= st.scale
            layer.x += st.dx * k
            layer.y += st.dy * k
            layer.reveal = st.reveal
            layer.progress = st.counter
            if c.kind == "text" and c.text and c.text.get("counter") and st.counter < 1.0:
                layer.counter_text = counter_text(c.text["counter"], st.counter)
            if c.kind == "caption" and c.text:
                words = c.text.get("words", [])
                layer.word_index = max([i for i, w in enumerate(words) if w["start"] <= t] or [-1])
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
