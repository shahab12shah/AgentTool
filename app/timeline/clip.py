"""Timeline clip. All times are seconds (float); display formatting lives elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


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

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.duration

    def snapshot(self) -> "Clip":
        return replace(self)

    def to_dict(self) -> dict[str, Any]:
        return {
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Clip":
        pos = d.get("position") or [0.0, 0.0]
        return cls(
            id=d["clip_id"],
            track_id=d["track_id"],
            asset_id=d["asset_id"],
            timeline_start=float(d["timeline_start"]),
            duration=float(d["duration"]),
            source_in=float(d.get("source_in", 0.0)),
            source_out=float(d.get("source_out", d.get("source_in", 0.0) + d["duration"])),
            position=(float(pos[0]), float(pos[1])),
            scale=float(d.get("scale", 1.0)),
            rotation=float(d.get("rotation", 0.0)),
            opacity=float(d.get("opacity", 1.0)),
            speed=float(d.get("speed", 1.0)),
        )
