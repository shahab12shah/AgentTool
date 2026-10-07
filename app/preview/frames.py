"""Frames for the timeline preview: stills come straight from the asset, video frames are extracted on demand and cached.

The preview is not the final render: it shows the poster frame at the requested source time so editing decisions (cuts, motion,
text, transitions) can be checked while scrubbing, without rendering anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.core.exceptions import FFmpegUnavailableError
from app.logging.logger import get_logger
from app.media.asset import Asset, AssetType
from app.media.media_probe import locate_binary, run_process

_log = get_logger(__name__)
STEP = 0.25  # seconds between cached frames of one video


class FrameProvider:
    def __init__(self, project_getter: Callable[[], object], ffmpeg_getter: Callable[[], str]) -> None:
        self._project, self._ffmpeg = project_getter, ffmpeg_getter
        self._failed: set[str] = set()

    def frame_path(self, asset: Asset, src_time: float) -> Path | None:
        project = self._project()
        if project is None or project.root is None:
            return None
        src = project.asset_path(asset)
        if not src.is_file():
            return None
        if asset.type is AssetType.IMAGE:
            return src
        if asset.type is not AssetType.VIDEO:
            return None
        q = round(max(0.0, min(src_time, (asset.duration or src_time) - 0.05)) / STEP) * STEP
        out = project.root / "previews" / "frames" / f"{asset.id}_{int(q * 100):06d}.jpg"
        if out.is_file():
            return out
        key = str(out)
        if key in self._failed:
            return None
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            exe = locate_binary("ffmpeg", self._ffmpeg())
            r = run_process([exe, "-y", "-v", "error", "-ss", f"{q:.2f}", "-i", str(src), "-frames:v", "1", "-vf", "scale=960:-2", str(out)], timeout=15)
            if r.returncode == 0 and out.is_file():
                return out
        except (FFmpegUnavailableError, OSError, Exception):  # a missing preview frame must never break scrubbing
            _log.debug("Frame extraction failed", exc_info=True)
        self._failed.add(key)
        return None
