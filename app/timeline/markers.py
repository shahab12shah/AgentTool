"""Timeline markers: user annotations on the timeline (not captions, not rendered graphics, not QC findings)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

MARKER_KINDS: dict[str, tuple[str, str]] = {  # kind -> (label, default colour)
    "generic": ("Marker", "#4aa3ff"),
    "scene": ("Scene transition", "#ffb84d"),
    "narration": ("Important narration", "#ff6b9d"),
    "evidence": ("Evidence point", "#7ddc7a"),
    "section": ("Section start", "#b58cff"),
    "review": ("Review note", "#ff5d5d"),
    "export": ("Export range boundary", "#e8e8e8"),
}


def new_marker_id() -> str:
    return f"marker_{uuid.uuid4().hex[:10]}"


@dataclass
class Marker:
    id: str
    time: float
    name: str = ""
    kind: str = "generic"
    color: str = ""  # "" = the kind's default colour
    note: str = ""

    @property
    def display_color(self) -> str:
        return self.color or MARKER_KINDS.get(self.kind, MARKER_KINDS["generic"])[1]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "time": self.time}
        for k in ("name", "kind", "color", "note"):
            v = getattr(self, k)
            if v and not (k == "kind" and v == "generic"):
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Marker":
        kind = str(d.get("kind", "generic"))
        return cls(str(d["id"]), float(d["time"]), str(d.get("name", "")), kind if kind in MARKER_KINDS else "generic", str(d.get("color", "")), str(d.get("note", "")))
