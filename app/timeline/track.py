"""Timeline track."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.media.asset import AssetType
from app.timeline.index import bump, watched
from app.timeline.clip import Clip


class TrackKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    GRAPHICS = "graphics"
    TEXT = "text"
    CAPTIONS = "captions"
    AUDIO = "audio"


@dataclass
class Track:
    id: str
    name: str
    kind: TrackKind
    hidden: bool = False
    muted: bool = False
    locked: bool = False
    solo: bool = False
    volume: float = 1.0  # audio tracks: track gain
    clips: list[Clip] = field(default_factory=list)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "clips":
            value = watched(value)
        object.__setattr__(self, name, value)
        if name in ("clips", "id"):
            bump()

    @property
    def is_audio(self) -> bool:
        return self.kind is TrackKind.AUDIO

    def accepts(self, asset_type: AssetType) -> bool:
        """Which media can be placed here. Text tracks hold generated titles (later phase)."""
        if self.kind is TrackKind.AUDIO:
            return asset_type is AssetType.AUDIO
        if self.kind in (TrackKind.TEXT, TrackKind.CAPTIONS):
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
            **({"solo": True} if self.solo else {}),
            **({"volume": self.volume} if abs(self.volume - 1.0) > 1e-9 else {}),
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
            solo=bool(d.get("solo", False)),
            volume=float(d.get("volume", 1.0)),
            clips=[Clip.from_dict(c) for c in d.get("clips", [])],
        )
        track.sort()
        return track
