"""Cached thumbnail generation (video frame, scaled image, or audio waveform).

One small JPEG per asset (``<project>/thumbnails/<asset id>.jpg``) is reused by every screen. FFmpeg decodes and scales ONE frame (fast input seek), so nothing
large is ever decoded into Python. A thumbnail is reused only while it is current: the source file's fingerprint (size + mtime) is recorded with the cache
manager when one is available, otherwise the thumbnail must be at least as new as the source.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from app.core.exceptions import JobCancelled, MediaError
from app.logging.logger import get_logger
from app.media.asset import Asset, AssetType
from app.media.media_probe import locate_binary
from app.performance.dependencies import asset_dep, file_fingerprint, stable_key
from app.performance.profiler import profiler
from app.storage.paths import ProjectPaths

_log = get_logger(__name__)
THUMB_WIDTH = 320
_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}


class MissingSourceError(MediaError):
    """The media file a thumbnail would be made from is not there (a different state from "FFmpeg could not read it")."""


def thumbnail_key(asset_id: str) -> str:
    return stable_key("thumbnail", asset_id)


def run_cancellable(cmd: list[str], timeout: float, should_cancel: Callable[[], bool] | None = None) -> tuple[int, str]:
    """Run an FFmpeg command, polling for cancellation so a cancelled thumbnail does not keep a process alive. Returns (returncode, stderr tail)."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, **_NO_WINDOW)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                _out, err = p.communicate(timeout=0.2)
                return p.returncode, (err or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired:
                if should_cancel is not None and should_cancel():
                    p.kill()
                    p.communicate()
                    raise JobCancelled() from None
                if time.monotonic() > deadline:
                    p.kill()
                    p.communicate()
                    return -1, "timed out"
    finally:
        if p.poll() is None:
            p.kill()
            p.communicate()


class ThumbnailService:
    def __init__(self, ffmpeg_path: str = "") -> None:
        self._configured = ffmpeg_path

    def configure(self, ffmpeg_path: str) -> None:
        self._configured = ffmpeg_path

    @staticmethod
    def thumbnail_path(project_root: Path, asset: Asset) -> Path:
        return ProjectPaths(project_root).thumbnails_dir / f"{asset.id}.jpg"

    def is_cached(self, project_root: Path, asset: Asset) -> bool:
        p = self.thumbnail_path(project_root, asset)
        try:
            return p.stat().st_size > 0
        except OSError:
            return False

    def source_path(self, project_root: Path, asset: Asset) -> Path:
        return ProjectPaths(project_root).resolve(asset.path)

    def is_current(self, project_root: Path, asset: Asset, cache: Any = None) -> bool:
        """The cached thumbnail exists and was made from the source as it is now. Counts a cache hit / miss."""
        out = self.thumbnail_path(project_root, asset)
        try:
            t_st = out.stat()
        except OSError:
            profiler.cache_miss("thumbnails")
            return False
        if t_st.st_size <= 0:
            profiler.cache_miss("thumbnails")
            return False
        try:
            s_st = self.source_path(project_root, asset).stat()
        except OSError:
            profiler.cache_hit("thumbnails")
            return True  # the source is gone: keep showing what we have
        key, dep = thumbnail_key(asset.id), asset_dep(asset.id)
        fp = f"s:{s_st.st_size}:{s_st.st_mtime_ns}"
        e = cache.peek(key) if cache is not None else None
        if e is not None and e.path is not None and os.path.normcase(str(e.path)) == os.path.normcase(str(out)):
            return cache.get(key, deps={dep: fp}, category="thumbnails") is not None  # counts the hit/miss; a changed source makes the cache drop the stale file
        if t_st.st_mtime_ns >= s_st.st_mtime_ns:  # no fingerprint recorded (older project / no cache): trust a thumbnail that is newer than its source
            profiler.cache_hit("thumbnails")
            if cache is not None:
                cache.register_external("thumbnails", out, key=key, deps={dep: fp})
            return True
        profiler.cache_miss("thumbnails")
        return False

    def _run(self, cmd: list[str], should_cancel: Callable[[], bool] | None) -> tuple[int, str]:
        return run_cancellable(cmd, 60, should_cancel)

    def command(self, src: Path, asset: Asset, tmp: Path) -> list[str]:
        w = THUMB_WIDTH
        scale = f"scale='min({w},iw)':-2"
        cmd = [locate_binary("ffmpeg", self._configured), "-y", "-v", "error", "-nostdin"]
        if asset.type is AssetType.VIDEO:
            seek = min(1.0, (asset.duration or 0) * 0.1)
            cmd += ["-ss", f"{seek:.3f}", "-i", str(src), "-an", "-sn", "-frames:v", "1", "-vf", scale]
        elif asset.type is AssetType.IMAGE:
            cmd += ["-i", str(src), "-an", "-frames:v", "1", "-vf", scale]
        else:
            cmd += ["-i", str(src), "-filter_complex", f"aformat=channel_layouts=mono,showwavespic=s={w}x180:colors=#4aa3ff", "-frames:v", "1"]
        return cmd + ["-update", "1", str(tmp)]

    def ensure(self, project_root: Path, asset: Asset, should_cancel=None, cache: Any = None) -> Path:
        """Return the current thumbnail, generating it only if it is missing or stale (one FFmpeg process, one decoded frame)."""
        return self.ensure_ex(project_root, asset, should_cancel, cache)[0]

    def ensure_ex(self, project_root: Path, asset: Asset, should_cancel=None, cache: Any = None) -> tuple[Path, bool]:
        """Like ``ensure``; the flag says whether a thumbnail was generated now (False: the cached one was current)."""
        out = self.thumbnail_path(project_root, asset)
        if self.is_current(project_root, asset, cache):
            return out, False
        src = self.source_path(project_root, asset)
        if not src.is_file():
            raise MissingSourceError(f"Media file is missing: {asset.name}")
        if should_cancel and should_cancel():
            raise JobCancelled()
        fp = file_fingerprint(src)  # taken before reading: a file that changes meanwhile is stale next time
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{asset.id}.tmp.jpg")
        cmd = self.command(src, asset, tmp)
        try:
            with profiler.timer("thumbnail.generate"):
                code, err = self._run(cmd, should_cancel)
            if code != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
                raise MediaError(f"Could not create a thumbnail for {asset.name}.", details=err.strip()[-500:])
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        if cache is not None:
            cache.register_external("thumbnails", out, key=thumbnail_key(asset.id), deps={asset_dep(asset.id): fp})
        return out, True
