"""Timeline track."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.media.asset import AssetType
from app.timeline.clip import Clip


class TrackKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    GRAPHICS = "graphics"
    TEXT = "text"
    AUDIO = "audio"


@dataclass
class Track:
    id: str
    name: str
    kind: TrackKind
    hidden: bool = False
    muted: bool = False
    locked: bool = False
    clips: list[Clip] = field(default_factory=list)

    @property
    def is_audio(self) -> bool:
        return self.kind is TrackKind.AUDIO

    def accepts(self, asset_type: AssetType) -> bool:
        """Which media can be placed here. Text tracks hold generated titles (later phase)."""
        if self.kind is TrackKind.AUDIO:
            return asset_type is AssetType.AUDIO
        if self.kind is TrackKind.TEXT:
            return False
        return asset_type in (AssetType.VIDEO, AssetType.IMAGE)

    def sort(self) -> None:
        self.clips.sort(key=lambda c: (c.timeline_start, c.id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "hidden": self.hidden,
            "muted": self.muted,
            "locked": self.locked,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Track":
        track = cls(
            id=d["id"],
            name=d["name"],
            kind=TrackKind(d["kind"]),
            hidden=bool(d.get("hidden", False)),
            muted=bool(d.get("muted", False)),
            locked=bool(d.get("locked", False)),
            clips=[Clip.from_dict(c) for c in d.get("clips", [])],
        )
        track.sort()
        return track
