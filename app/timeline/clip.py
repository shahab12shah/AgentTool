"""Timeline clip. All times are seconds (float); display formatting lives elsewhere."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from app.timeline.index import CLIP_WATCHED, bump
from app.timeline.keyframes import Keyframe

KIND_MEDIA, KIND_TEXT, KIND_GRAPHIC, KIND_CAPTION = "media", "text", "graphic", "caption"


@dataclass
class Clip:
    id: str
    track_id: str
    asset_id: str
    timeline_start: float
    duration: float
    source_in: float = 0.0
    source_out: float = 0.0  # always source_in + duration * speed
    position: tuple[float, float] = (0.0, 0.0)  # offset from canvas centre, in pixels
    scale: float = 1.0
    rotation: float = 0.0  # degrees
    opacity: float = 1.0
    speed: float = 1.0
    # --- Phase 4: AI-assembled timeline metadata (all optional; plain clips ignore them) ---
    kind: str = KIND_MEDIA  # "media" | "text" | "graphic" | "caption" (non-media clips have no asset)
    scene_id: str = ""
    slot: str = ""  # role inside the scene ("visual:0", "text:1"...): what regeneration matches on
    created_by: str = "USER"  # AI | USER | SYSTEM
    ai_decision_id: str = ""
    locked: bool = False  # protected from AI regeneration
    keyframes: list[Keyframe] = field(default_factory=list)
    effects: dict[str, Any] = field(default_factory=dict)  # fit, focus region, highlight box...
    text: dict[str, Any] | None = None
    animation: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    transition: dict[str, Any] | None = None  # transition INTO this clip
    metadata: dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in CLIP_WATCHED:
            bump()  # tells every TimelineIndex that geometry/identity changed in place

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.duration

    def snapshot(self) -> "Clip":
        c = copy.copy(self)
        c.keyframes = [copy.copy(k) for k in self.keyframes]
        c.effects, c.animation, c.audio, c.metadata = (copy.deepcopy(x) for x in (self.effects, self.animation, self.audio, self.metadata))
        c.text, c.transition = copy.deepcopy(self.text), copy.deepcopy(self.transition)
        return c

    def to_dict(self) -> dict[str, Any]:
        d = {
            "clip_id": self.id,
            "track_id": self.track_id,
            "asset_id": self.asset_id,
            "timeline_start": self.timeline_start,
            "duration": self.duration,
            "source_in": self.source_in,
            "source_out": self.source_out,
            "position": [self.position[0], self.position[1]],
            "scale": self.scale,
            "rotation": self.rotation,
            "opacity": self.opacity,
            "speed": self.speed,
        }
        if (self.kind, self.scene_id, self.slot, self.created_by, self.ai_decision_id, self.locked) != (KIND_MEDIA, "", "", "USER", "", False):
            d.update(kind=self.kind, scene_id=self.scene_id, slot=self.slot, created_by=self.created_by,
                     ai_decision_id=self.ai_decision_id, locked=self.locked)
        if self.keyframes:
            d["keyframes"] = [k.to_dict() for k in self.keyframes]
        for name in ("effects", "animation", "audio", "metadata"):
            if getattr(self, name):
                d[name] = copy.deepcopy(getattr(self, name))
        if self.text is not None:
            d["text"] = copy.deepcopy(self.text)
        if self.transition is not None:
            d["transition"] = copy.deepcopy(self.transition)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Clip":
        pos = d.get("position") or [0.0, 0.0]
        return cls(
            id=d["clip_id"],
            track_id=d["track_id"],
            asset_id=d.get("asset_id", ""),
            timeline_start=float(d["timeline_start"]),
            duration=float(d["duration"]),
            source_in=float(d.get("source_in", 0.0)),
            source_out=float(d.get("source_out", d.get("source_in", 0.0) + d["duration"])),
            position=(float(pos[0]), float(pos[1])),
            scale=float(d.get("scale", 1.0)),
            rotation=float(d.get("rotation", 0.0)),
            opacity=float(d.get("opacity", 1.0)),
            speed=float(d.get("speed", 1.0)),
            kind=str(d.get("kind", KIND_MEDIA)), scene_id=str(d.get("scene_id", "")), slot=str(d.get("slot", "")),
            created_by=str(d.get("created_by", "USER")), ai_decision_id=str(d.get("ai_decision_id", "")), locked=bool(d.get("locked", False)),
            keyframes=[Keyframe.from_dict(k) for k in d.get("keyframes", [])],
            effects=dict(d.get("effects") or {}), text=copy.deepcopy(d["text"]) if d.get("text") is not None else None,
            animation=dict(d.get("animation") or {}), audio=dict(d.get("audio") or {}),
            transition=copy.deepcopy(d["transition"]) if d.get("transition") is not None else None,
            metadata=dict(d.get("metadata") or {}),
        )
