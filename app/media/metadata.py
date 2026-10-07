"""Media classification and probed metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.constants import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.exceptions import UnsupportedMediaError
from app.media.asset import AssetType

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


def asset_type_for(path: Path) -> AssetType:
    """Classify by extension; raises ``UnsupportedMediaError`` for anything else."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return AssetType.VIDEO
    if ext in IMAGE_EXTENSIONS:
        return AssetType.IMAGE
    if ext in AUDIO_EXTENSIONS:
        return AssetType.AUDIO
    supported = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
    raise UnsupportedMediaError(
        f"“{Path(path).name}” is not a supported file type. Supported: {supported}."
    )


def file_dialog_filter() -> str:
    def pat(exts: frozenset[str]) -> str:
        return " ".join(f"*{e}" for e in sorted(exts))

    return (
        f"All supported media ({pat(SUPPORTED_EXTENSIONS)});;"
        f"Video ({pat(VIDEO_EXTENSIONS)});;Images ({pat(IMAGE_EXTENSIONS)});;Audio ({pat(AUDIO_EXTENSIONS)})"
    )


@dataclass
class MediaInfo:
    """Facts obtained from ffprobe; nothing here is assumed."""

    type: AssetType
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
