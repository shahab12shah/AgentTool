"""Editable keyframes. Times are seconds relative to the start of the clip they belong to."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PROPERTIES = ("position_x", "position_y", "scale", "rotation", "opacity", "volume", "blur")
INTERPOLATIONS = ("linear", "ease_in", "ease_out", "ease_in_out")
DEFAULTS = {"position_x": 0.0, "position_y": 0.0, "scale": 1.0, "rotation": 0.0, "opacity": 1.0, "volume": 1.0, "blur": 0.0}


@dataclass
class Keyframe:
    property: str
    time: float
    value: float
    interpolation: str = "linear"  # how the value travels from this keyframe to the next
    decision_id: str = ""  # the editing decision that owns it ("" = user keyframe)

    def to_dict(self) -> dict[str, Any]:
        d = {"property": self.property, "time": self.time, "value": self.value, "interpolation": self.interpolation}
        if self.decision_id:
            d["decision_id"] = self.decision_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Keyframe":
        return cls(str(d["property"]), float(d["time"]), float(d["value"]), str(d.get("interpolation", "linear")), str(d.get("decision_id", "")))

    def problems(self, duration: float) -> list[str]:
        out = []
        if self.property not in PROPERTIES:
            out.append(f"unknown keyframe property {self.property!r}")
        if self.interpolation not in INTERPOLATIONS:
            out.append(f"unknown interpolation {self.interpolation!r}")
        if not (math.isfinite(self.time) and math.isfinite(self.value)):
            out.append("keyframe has a non-finite value")
        elif self.time < -1e-6 or self.time > duration + 1e-6:
            out.append(f"keyframe at {self.time:.2f}s is outside the clip")
        return out


def ease(kind: str, u: float) -> float:
    u = min(1.0, max(0.0, u))
    if kind == "ease_in":
        return u * u
    if kind == "ease_out":
        return 1 - (1 - u) ** 2
    if kind == "ease_in_out":
        return 3 * u * u - 2 * u ** 3
    return u


def value_at(keyframes: list[Keyframe], prop: str, t: float, default: float | None = None) -> float:
    """Interpolated value of ``prop`` at clip-local time ``t`` (holds the first/last value outside the range)."""
    pts = sorted((k for k in keyframes if k.property == prop), key=lambda k: k.time)
    if not pts:
        return DEFAULTS[prop] if default is None else default
    if t <= pts[0].time:
        return pts[0].value
    for a, b in zip(pts, pts[1:]):
        if t <= b.time:
            span = b.time - a.time
            u = 1.0 if span <= 1e-9 else (t - a.time) / span
            return a.value + (b.value - a.value) * ease(a.interpolation, u)
    return pts[-1].value
