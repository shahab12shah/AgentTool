"""MediaProbeService: FFprobe inspection for rendering (and for filling the asset registry).

Gives the renderer the facts it must not guess: the *display* size (after rotation metadata), the real frame rate, the pixel format
(does it have alpha?), audio sample rate and channels. Results are cached by (path, size, mtime) so a 100-asset timeline is probed once.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

from app.core.constants import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.exceptions import FFmpegUnavailableError, MediaProbeError, UnsupportedMediaError
from app.performance.dependencies import stable_key
from app.performance.memory_monitor import BoundedLRU
from app.performance.profiler import profiler
from app.rendering.ffmpeg_service import FFmpegService

PROBE_VERSION = 1  # bump when ProbeInfo changes meaning: persisted results of another version are probed again
MEMORY_ENTRIES = 4096
INTERNAL_VIDEO = {".ts"}  # the renderer's own intermediate sections (not an importable media type)
ALPHA_PREFIXES = ("rgba", "bgra", "argb", "abgr", "yuva", "gbrap", "ya8", "ya16", "gray16a")


@dataclass
class ProbeInfo:
    path: str
    kind: str = "video"  # video | image | audio
    container: str = ""
    duration: float | None = None
    size_bytes: int = 0
    bitrate: int | None = None
    width: int | None = None  # display size (rotation applied)
    height: int | None = None
    coded_width: int | None = None
    coded_height: int | None = None
    rotation: int = 0
    fps: float | None = None
    codec: str | None = None
    pix_fmt: str | None = None
    has_alpha: bool = False
    frame_count: int | None = None
    has_video: bool = False
    has_audio: bool = False
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    audio_bitrate: int | None = None
    audio_duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def extra(self) -> dict[str, Any]:
        """The facts stored in ``asset.extra["probe"]`` (the registry's basic fields stay as they are)."""
        return {k: v for k, v in self.to_dict().items() if k in ("container", "bitrate", "rotation", "pix_fmt", "has_alpha", "frame_count", "coded_width", "coded_height", "audio_bitrate")
                and v not in (None, "")}


def _fraction(v: str | None) -> float | None:
    if not v or v in ("0/0", "N/A"):
        return None
    try:
        if "/" in v:
            a, b = v.split("/", 1)
            return float(a) / float(b) if float(b) else None
        return float(v)
    except ValueError:
        return None


def _int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def kind_for(path: Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS or ext in INTERNAL_VIDEO:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    raise UnsupportedMediaError(f"“{Path(path).name}” is not a supported file type ({ext or 'no extension'}).")


def parse_probe(path: Path, data: dict[str, Any]) -> ProbeInfo:
    kind = kind_for(path)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None) or next(
        (s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    info = ProbeInfo(str(path), kind, str(fmt.get("format_name", "")), _float(fmt.get("duration")), _int(fmt.get("size")) or 0, _int(fmt.get("bit_rate")))
    if kind == "audio":
        if audio is None:
            raise MediaProbeError(f"“{Path(path).name}” contains no audio stream.")
    elif video is None:
        raise MediaProbeError(f"“{Path(path).name}” contains no video/image stream.")
    if video is not None:
        info.has_video = True
        info.coded_width, info.coded_height = _int(video.get("width")), _int(video.get("height"))
        rot = 0
        tags = video.get("tags") or {}
        if "rotate" in tags:
            rot = _int(tags["rotate"]) or 0
        for sd in video.get("side_data_list") or []:
            if "rotation" in sd:
                rot = _int(sd["rotation"]) or rot
        info.rotation = (-rot) % 360  # ffprobe reports the counter-clockwise correction; this is the clockwise display rotation
        w, h = info.coded_width, info.coded_height
        info.width, info.height = (h, w) if info.rotation in (90, 270) else (w, h)
        info.codec = video.get("codec_name")
        info.pix_fmt = video.get("pix_fmt")
        info.has_alpha = _has_alpha(info.pix_fmt) or str(tags.get("alpha_mode", "")) == "1"
        if kind == "video":
            info.fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
            info.frame_count = _int(video.get("nb_frames"))
            if info.duration is None:
                info.duration = _float(video.get("duration"))
    if audio is not None:
        info.has_audio = True
        info.audio_codec = audio.get("codec_name")
        info.sample_rate = _int(audio.get("sample_rate"))
        info.channels = _int(audio.get("channels"))
        info.audio_bitrate = _int(audio.get("bit_rate"))
        info.audio_duration = _float(audio.get("duration"))
        if kind == "audio":
            info.codec = info.audio_codec
        if kind == "audio" and info.duration is None:
            info.duration = _float(audio.get("duration"))
    if kind != "image" and (info.duration is None or info.duration <= 0):
        raise MediaProbeError(f"Could not determine the duration of “{Path(path).name}”.")
    if kind == "image":
        info.duration = None
    elif kind == "video" and info.frame_count is None and info.fps and info.duration:
        info.frame_count = int(round(info.fps * info.duration))
    return info


def _has_alpha(pix_fmt: str | None) -> bool:
    return bool(pix_fmt) and pix_fmt.startswith(ALPHA_PREFIXES)  # type: ignore[union-attr]


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class MediaProbeService:
    """FFprobe results, remembered in memory (bounded) and, when a cache manager is attached, across restarts (category ``analysis``).

    A persisted result is keyed by the file's path, size and modification time, so a changed file is probed again by itself. Failures are never stored.
    """

    def __init__(self, ffmpeg: FFmpegService) -> None:
        self.ff = ffmpeg
        self._cache: BoundedLRU[tuple[str, int, int], ProbeInfo] = BoundedLRU(MEMORY_ENTRIES)
        self._lock = threading.Lock()
        self._cache_getter: Callable[[], Any] | None = None
        self.persisted_hits = 0

    def attach_cache(self, getter: Callable[[], Any]) -> None:
        """``getter()`` returns the open project's ``MediaCacheManager`` (or None): probe results are then also kept on disk."""
        self._cache_getter = getter

    def _persistent(self) -> Any:
        try:
            return self._cache_getter() if self._cache_getter else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _persist_key(path: Path, size: int, mtime_ns: int) -> str:
        return stable_key("probe", Path(os.path.abspath(path)), size, mtime_ns)

    @staticmethod
    def _from_stored(path: Path, d: Any) -> ProbeInfo | None:
        """A stored dict back into a ProbeInfo; anything that does not look right is ignored (the file is probed again)."""
        try:
            if not isinstance(d, dict) or str(d.get("path")) != str(path):
                return None
            names = {f.name for f in fields(ProbeInfo)}
            info = ProbeInfo(**{k: v for k, v in d.items() if k in names})
            return info if info.kind in ("video", "image", "audio") else None
        except (TypeError, ValueError):
            return None

    def probe(self, path: Path, use_cache: bool = True) -> ProbeInfo:
        path = Path(path)
        kind_for(path)  # unsupported extensions fail early with a clear message
        try:
            st = path.stat()
        except OSError as exc:
            raise MediaProbeError(f"“{path.name}” cannot be read: {exc.strerror or exc}") from exc
        key = (str(path), st.st_size, int(st.st_mtime_ns))
        store = self._persistent() if use_cache else None
        pkey = self._persist_key(path, st.st_size, int(st.st_mtime_ns)) if store is not None else ""
        if use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                profiler.cache_hit("probe")
                return hit
            if store is not None:
                entry = store.get(pkey, category="analysis", version=PROBE_VERSION)
                info = self._from_stored(path, entry.inline) if entry is not None else None
                if info is not None:
                    self._cache.put(key, info)
                    self.persisted_hits += 1
                    return info
                if entry is not None:  # stored, but unusable: drop it
                    store.invalidate(pkey)
            profiler.cache_miss("probe")
        cmd = [self.ff.ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        try:
            with profiler.timer("probe.ffprobe"):
                r = self.ff.run(cmd, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise MediaProbeError(f"Reading “{path.name}” timed out.", details=str(exc)) from exc
        except OSError as exc:
            raise FFmpegUnavailableError("ffprobe could not be started.", details=str(exc)) from exc
        if r.returncode != 0:
            raise MediaProbeError(f"“{path.name}” could not be read. It may be corrupt or use an unsupported codec.", details=r.stderr.strip()[-500:])
        try:
            data = json.loads(r.stdout)
        except ValueError as exc:
            raise MediaProbeError(f"“{path.name}” returned unreadable metadata.", details=str(exc)) from exc
        info = parse_probe(path, data)
        info.size_bytes = info.size_bytes or st.st_size
        if use_cache:
            self._cache.put(key, info)
            if store is not None:
                store.put(pkey, info.to_dict(), category="analysis", data_type="probe", version=PROBE_VERSION)
        return info

    def try_probe(self, path: Path) -> tuple[ProbeInfo | None, str]:
        try:
            return self.probe(path), ""
        except (MediaProbeError, UnsupportedMediaError, FFmpegUnavailableError) as exc:
            return None, exc.user_message
