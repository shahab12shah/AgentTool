"""Asset domain model. Assets are identified by stable IDs, never by filename."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssetType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class SourceType(str, Enum):
    """Where an asset came from. Phase 1 only produces ``USER_MEDIA``."""

    USER_MEDIA = "USER_MEDIA"
    YOUTUBE = "YOUTUBE"
    STOCK_IMAGE = "STOCK_IMAGE"
    STOCK_VIDEO = "STOCK_VIDEO"
    WEB_IMAGE = "WEB_IMAGE"
    WEB_VIDEO = "WEB_VIDEO"
    SCREENSHOT = "SCREENSHOT"
    AI_GENERATED = "AI_GENERATED"


@dataclass
class Asset:
    id: str
    type: AssetType
    source_type: SourceType
    path: str  # relative to the project root, or absolute for linked (non-copied) media
    name: str  # display name (original filename)
    duration: float | None = None  # seconds; None for still images
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    size_bytes: int = 0
    content_hash: str | None = None  # sha256 of the file; used for duplicate detection
    link_mode: str = "copy"  # "copy" (inside project) or "reference" (external file)
    imported_at: str = ""
    source_url: str | None = None  # future: origin of web/stock/YouTube assets
    extra: dict[str, Any] = field(default_factory=dict)  # future: licence, attribution, AI prompt...

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_type": self.source_type.value,
            "path": self.path,
            "name": self.name,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "codec": self.codec,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "link_mode": self.link_mode,
            "imported_at": self.imported_at,
            "source_url": self.source_url,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Asset":
        return cls(
            id=d["id"],
            type=AssetType(d["type"]),
            source_type=SourceType(d.get("source_type", "USER_MEDIA")),
            path=d["path"],
            name=d.get("name") or d["path"].rsplit("/", 1)[-1],
            duration=d.get("duration"),
            width=d.get("width"),
            height=d.get("height"),
            fps=d.get("fps"),
            codec=d.get("codec"),
            has_audio=bool(d.get("has_audio", False)),
            audio_codec=d.get("audio_codec"),
            sample_rate=d.get("sample_rate"),
            channels=d.get("channels"),
            size_bytes=int(d.get("size_bytes", 0)),
            content_hash=d.get("content_hash"),
            link_mode=d.get("link_mode", "copy"),
            imported_at=d.get("imported_at", ""),
            source_url=d.get("source_url"),
            extra=dict(d.get("extra") or {}),
        )
