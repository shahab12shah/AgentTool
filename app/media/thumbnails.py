"""Cached thumbnail generation (video frame, scaled image, or audio waveform)."""

from __future__ import annotations

import os
from pathlib import Path

from app.core.exceptions import JobCancelled, MediaError
from app.logging.logger import get_logger
from app.media.asset import Asset, AssetType
from app.media.media_probe import locate_binary, run_process
from app.storage.paths import ProjectPaths

_log = get_logger(__name__)
THUMB_WIDTH = 320


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
        return p.is_file() and p.stat().st_size > 0

    def ensure(self, project_root: Path, asset: Asset, should_cancel=None) -> Path:
        """Return the cached thumbnail, generating it only if it is missing."""
        out = self.thumbnail_path(project_root, asset)
        if self.is_cached(project_root, asset):
            return out
        src = ProjectPaths(project_root).resolve(asset.path)
        if not src.is_file():
            raise MediaError(f"Media file is missing: {asset.name}")
        if should_cancel and should_cancel():
            raise JobCancelled()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{asset.id}.tmp.jpg")
        cmd = [locate_binary("ffmpeg", self._configured), "-y", "-v", "error"]
        if asset.type is AssetType.VIDEO:
            seek = min(1.0, (asset.duration or 0) * 0.1)
            cmd += ["-ss", f"{seek:.3f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2"]
        elif asset.type is AssetType.IMAGE:
            cmd += ["-i", str(src), "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2"]
        else:
            cmd += [
                "-i", str(src), "-filter_complex",
                f"aformat=channel_layouts=mono,showwavespic=s={THUMB_WIDTH}x180:colors=#4aa3ff",
                "-frames:v", "1",
            ]
        cmd += ["-update", "1", str(tmp)]
        try:
            result = run_process(cmd, timeout=60)
            if result.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
                raise MediaError(
                    f"Could not create a thumbnail for {asset.name}.", details=result.stderr.strip()[-500:]
                )
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        return out
