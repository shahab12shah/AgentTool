"""FFmpeg / ffprobe discovery and media probing."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.core.exceptions import FFmpegUnavailableError, MediaProbeError
from app.logging.logger import get_logger
from app.media.asset import AssetType
from app.media.metadata import MediaInfo, asset_type_for

_log = get_logger(__name__)
_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}


def locate_binary(name: str, configured: str = "") -> str:
    """Find ``ffmpeg``/``ffprobe``. ``configured`` may be an executable or its directory."""
    if configured:
        p = Path(configured).expanduser()
        candidates = [p, p / name, p / f"{name}.exe"]
        for c in candidates:
            if c.is_file():
                return str(c)
        raise FFmpegUnavailableError(
            f"The configured {name} path does not exist: {configured}. Fix it in Settings.",
        )
    found = shutil.which(name)
    if not found:
        raise FFmpegUnavailableError(
            f"{name} was not found. Install FFmpeg and add it to PATH, or set its location in Settings."
        )
    return found


def run_process(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", **_NO_WINDOW
    )


def _fraction(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(value)
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MediaProber:
    """Wraps ``ffprobe``. Paths are resolved lazily so Settings changes take effect."""

    def __init__(self, ffprobe_path: str = "") -> None:
        self._configured = ffprobe_path

    def configure(self, ffprobe_path: str) -> None:
        self._configured = ffprobe_path

    def binary(self) -> str:
        return locate_binary("ffprobe", self._configured)

    def probe(self, path: Path) -> MediaInfo:
        path = Path(path)
        asset_type = asset_type_for(path)
        cmd = [self.binary(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        try:
            result = run_process(cmd, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise MediaProbeError(f"Reading “{path.name}” timed out.", details=str(exc)) from exc
        except OSError as exc:
            raise FFmpegUnavailableError("ffprobe could not be started.", details=str(exc)) from exc
        if result.returncode != 0:
            _log.warning("ffprobe failed", extra={"path": str(path), "stderr": result.stderr[-500:]})
            raise MediaProbeError(
                f"“{path.name}” could not be read. It may be corrupt or not a valid media file.",
                details=result.stderr.strip()[-500:],
            )
        try:
            data = json.loads(result.stdout)
        except ValueError as exc:
            raise MediaProbeError(f"“{path.name}” returned unreadable metadata.", details=str(exc)) from exc
        return self._parse(path, asset_type, data)

    @staticmethod
    def _parse(path: Path, asset_type: AssetType, data: dict[str, Any]) -> MediaInfo:
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        info = MediaInfo(type=asset_type)
        if asset_type is AssetType.AUDIO:
            if audio is None:
                raise MediaProbeError(f"“{path.name}” contains no audio stream.")
        elif video is None:
            raise MediaProbeError(f"“{path.name}” contains no video/image stream.")
        if video is not None:
            info.width = int(video.get("width") or 0) or None
            info.height = int(video.get("height") or 0) or None
            info.codec = video.get("codec_name")
            if asset_type is AssetType.VIDEO:
                info.fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
        if audio is not None:
            info.has_audio = True
            info.audio_codec = audio.get("codec_name")
            info.sample_rate = int(audio["sample_rate"]) if str(audio.get("sample_rate", "")).isdigit() else None
            info.channels = audio.get("channels")
            if asset_type is AssetType.AUDIO:
                info.codec = audio.get("codec_name")
        if asset_type is not AssetType.IMAGE:
            duration = _num(fmt.get("duration"))
            if duration is None and (video or audio):
                duration = _num((video or audio).get("duration"))  # type: ignore[union-attr]
            if duration is None or duration <= 0:
                raise MediaProbeError(f"Could not determine the duration of “{path.name}”.")
            info.duration = duration
        return info
