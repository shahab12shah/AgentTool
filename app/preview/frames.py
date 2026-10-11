"""Frames for the timeline preview: stills come straight from the asset, video frames are extracted on demand and cached.

The preview is not the final render: it shows the poster frame at the requested source time so editing decisions (cuts, motion,
text, transitions) can be checked while scrubbing, without rendering anything.

* ``frame_path`` is the simple synchronous call (extract on a miss). ``request`` is what the UI uses: it returns a cached frame at once, otherwise it queues ONE
  extraction (concurrent requests for the same frame coalesce) and calls back when it is ready.
* Seeks supersede each other: ``begin_seek()`` starts a new generation and the first request of that generation cancels every extraction still queued or running
  for an older one, so a fast scrub never builds a backlog and a late result for an old position never reaches the screen.
* ``preview_quality`` (draft / balanced / high) only chooses the width of the extracted preview frame (480 / 960 / 1280). Exports never read these frames.
* Frames are registered with the cache manager (category ``preview_frames``, which enforces the size limit) and decoded pixmaps live in a bounded memory tier.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.exceptions import FFmpegUnavailableError, JobCancelled
from app.jobs.job import Priority
from app.logging.logger import get_logger
from app.media.asset import Asset, AssetType
from app.media.media_probe import locate_binary
from app.media.thumbnails import run_cancellable
from app.performance.cache_manager import tmp_for
from app.performance.dependencies import asset_dep, file_fingerprint, stable_key
from app.performance.memory_monitor import BoundedLRU
from app.performance.profiler import profiler
from app.performance.thumbnail_queue import IDLE, VISIBLE, PriorityWorkQueue

_log = get_logger(__name__)
STEP = 0.25  # seconds between cached frames of one video
QUALITY_WIDTH = {"draft": 480, "balanced": 960, "high": 1280}
DEFAULT_QUALITY = "balanced"
FAILED_TTL = 30.0  # a frame that could not be extracted is not retried for this long (then it may be: the file may have been fixed)
FAILED_MAX = 512
CLEANUP_EVERY = 200  # new frames between two cache clean-ups
VALIDATE_EVERY = 2.0  # seconds between fingerprint re-validations of the same asset


@dataclass
class _FrameTask:
    key: str
    asset: Asset
    src: Path
    q: float
    out: Path
    width: int
    gen: int
    prefetch: bool
    fp: str
    callbacks: list[Callable[[Path], None]] = field(default_factory=list)


class FrameProvider:
    def __init__(self, project_getter: Callable[[], object], ffmpeg_getter: Callable[[], str], proxy_getter: Callable[[Asset], Path | None] | None = None, *,
                 quality_getter: Callable[[], str] | None = None, cache_getter: Callable[[], object | None] | None = None, limits_getter: Callable[[], object | None] | None = None,
                 pressure_ok: Callable[[], bool] | None = None, jobs=None) -> None:
        self._project, self._ffmpeg = project_getter, ffmpeg_getter
        self._proxy = proxy_getter  # editing reads the proxy when there is one: the timeline itself only ever references the asset id
        self._quality, self._cache_get, self._limits_get = quality_getter, cache_getter, limits_getter
        self._pressure_ok = pressure_ok or (lambda: True)
        self._jobs = jobs
        self.last_source: Path | None = None
        self.last_used_proxy = False
        self._quality_value, self._quality_at = DEFAULT_QUALITY, 0.0
        self._failed: dict[str, float] = {}
        self._lock = threading.RLock()
        self._pending: dict[str, _FrameTask] = {}
        self._queue: PriorityWorkQueue | None = None
        self._queue_project: object | None = None
        self._gen = 0
        self._swept = 0
        self._fp: dict[str, tuple[str, float]] = {}
        self._known: BoundedLRU[str, float] = BoundedLRU(4096)
        self._tier: BoundedLRU | None = None
        self._new_frames = 0
        self.stats = {"extracted": 0, "coalesced": 0, "superseded": 0, "stale_discarded": 0, "failed": 0, "cached": 0}

    # ------------------------------------------------------------------ settings
    @property
    def quality(self) -> str:
        now = time.monotonic()
        if now - self._quality_at > 1.0:  # the setting is read at most once a second (it is asked for on every paint)
            try:
                q = self._quality() if self._quality else DEFAULT_QUALITY
            except Exception:  # noqa: BLE001
                q = DEFAULT_QUALITY
            self._quality_value, self._quality_at = (q if q in QUALITY_WIDTH else DEFAULT_QUALITY), now
        return self._quality_value

    def refresh_settings(self) -> None:
        self._quality_at = 0.0

    @property
    def width(self) -> int:
        return QUALITY_WIDTH[self.quality]

    @property
    def generation(self) -> int:
        return self._gen

    def reduced_quality(self) -> str:
        """Why the preview is not the full-quality picture ('' when it is): shown to the user as a small tag. Never affects the export."""
        if self.last_used_proxy:
            return "Proxy preview"
        return "Draft preview" if self.quality == "draft" else ""

    def prefetch_count(self) -> int:
        lim = self._limits_get() if self._limits_get else None
        return int(getattr(lim, "prefetch_frames", 12)) if lim is not None else 12

    def memory_tier(self) -> BoundedLRU:
        """The bounded in-memory cache for decoded frames (pixmaps): sized from the memory limit through the cache manager when there is one."""
        if self._tier is None:
            cache = self._cache()
            weigher = lambda pm: int(pm.width() * pm.height() * 4) if hasattr(pm, "width") else 1024  # noqa: E731
            if cache is not None:
                self._tier = cache.memory_tier("preview_frames", weigher=weigher, share=0.25)
            else:
                self._tier = BoundedLRU(64, 128 * 1024 * 1024, weigher)
        return self._tier

    def _cache(self):
        try:
            return self._cache_get() if self._cache_get else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ locating a frame
    def _resolve(self, asset: Asset):
        """(project, source path, used_proxy, frame width) or None when this asset has no extractable frames."""
        project = self._project()
        if project is None or project.root is None or asset.type is not AssetType.VIDEO:
            return None
        src = project.asset_path(asset)
        proxy = self._proxy(asset) if self._proxy else None
        used_proxy = proxy is not None and proxy.is_file()
        if used_proxy:
            src = proxy
        if not src.is_file():
            return None
        return project, src, used_proxy, self.width

    def _out_path(self, project, asset: Asset, q: float, width: int) -> Path:
        stem = f"{asset.id}_{int(q * 100):06d}"
        return project.root / "previews" / "frames" / (f"{stem}.jpg" if width == QUALITY_WIDTH[DEFAULT_QUALITY] else f"{stem}_w{width}.jpg")

    @staticmethod
    def _quantize(asset: Asset, src_time: float) -> float:
        return round(max(0.0, min(src_time, (asset.duration or src_time) - 0.05)) / STEP) * STEP

    def _fingerprint(self, project, asset: Asset) -> str:
        now = time.monotonic()
        hit = self._fp.get(asset.id)
        if hit is not None and now - hit[1] < VALIDATE_EVERY:
            return hit[0]
        fp = file_fingerprint(project.asset_path(asset))
        if len(self._fp) > 2048:
            self._fp.clear()
        self._fp[asset.id] = (fp, now)
        return fp

    def _valid(self, project, asset: Asset, key: str, out: Path) -> bool:
        """The frame file exists and (with a cache index) was extracted from the media as it is now."""
        try:
            if os.stat(out).st_size <= 0:
                return False
        except OSError:
            return False
        cache = self._cache()
        if cache is None:
            profiler.cache_hit("preview_frames")
            return True
        e = cache.peek(key)
        if e is None:
            cache.register_external("preview_frames", out, key=key, deps={asset_dep(asset.id): self._fingerprint(project, asset)})  # a frame from before the index existed
            profiler.cache_hit("preview_frames")
            return True
        return cache.get(key, deps={asset_dep(asset.id): self._fingerprint(project, asset)}, category="preview_frames") is not None

    def cached_path(self, asset: Asset, src_time: float) -> Path | None:
        """A frame that is already on disk (never extracts)."""
        project = self._project()
        if project is None or project.root is None:
            return None
        if asset.type is AssetType.IMAGE:
            src = project.asset_path(asset)
            return src if src.is_file() else None
        res = self._resolve(asset)
        if res is None:
            return None
        _p, _src, used_proxy, width = res
        self.last_used_proxy = used_proxy
        q = self._quantize(asset, src_time)
        out = self._out_path(project, asset, q, width)
        return out if self._valid(project, asset, self._key(asset, q, width), out) else None

    @staticmethod
    def _key(asset: Asset, q: float, width: int) -> str:
        return stable_key("frame", asset.id, int(q * 100), width)

    # ------------------------------------------------------------------ extraction
    def _run(self, cmd: list[str], should_cancel: Callable[[], bool] | None) -> tuple[int, str]:
        return run_cancellable(cmd, 15, should_cancel)

    def _extract(self, task: _FrameTask, should_cancel: Callable[[], bool] | None = None) -> bool:
        out = task.out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_for(out)
        cmd = [locate_binary("ffmpeg", self._ffmpeg()), "-y", "-v", "error", "-nostdin", "-ss", f"{task.q:.2f}", "-i", str(task.src), "-an", "-sn", "-frames:v", "1",
               "-vf", f"scale={task.width}:-2", str(tmp)]
        try:
            with profiler.timer("preview.frame_extract"):
                code, _err = self._run(cmd, should_cancel)
            if code != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
                return False
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        self.stats["extracted"] += 1
        cache = self._cache()
        if cache is not None:
            cache.register_external("preview_frames", out, key=task.key, deps={asset_dep(task.asset.id): task.fp})
            self._new_frames += 1
            if self._new_frames >= CLEANUP_EVERY:
                self._new_frames = 0
                cache.cleanup()  # enforces the preview_frames size limit (least recently used frames go first)
        return True

    def _is_failed(self, out: Path) -> bool:
        t = self._failed.get(str(out))
        if t is None:
            return False
        if time.monotonic() - t > FAILED_TTL:
            del self._failed[str(out)]
            return False
        return True

    def _mark_failed(self, out: Path) -> None:
        if len(self._failed) >= FAILED_MAX:
            self._failed.pop(next(iter(self._failed)))
        self._failed[str(out)] = time.monotonic()
        self.stats["failed"] += 1

    def clear_failed(self) -> None:
        self._failed.clear()

    def frame_path(self, asset: Asset, src_time: float) -> Path | None:
        """Synchronous: the frame at ``src_time`` (extracted now on a miss), or None when it cannot be made."""
        project = self._project()
        if project is None or project.root is None:
            return None
        if asset.type is AssetType.IMAGE:
            src = project.asset_path(asset)
            return src if src.is_file() else None
        res = self._resolve(asset)
        if res is None:
            return None
        _p, src, used_proxy, width = res
        q = self._quantize(asset, src_time)
        out = self._out_path(project, asset, q, width)
        key = self._key(asset, q, width)
        self.last_used_proxy = used_proxy
        if self._valid(project, asset, key, out):
            self.stats["cached"] += 1
            return out
        if self._is_failed(out):
            return None
        self.last_source = src
        task = _FrameTask(key, asset, src, q, out, width, self._gen, False, self._fingerprint(project, asset))
        try:
            if self._extract(task):
                return out
        except (FFmpegUnavailableError, OSError, Exception):  # a missing preview frame must never break scrubbing
            _log.debug("Frame extraction failed", exc_info=True)
        self._mark_failed(out)
        return None

    # ------------------------------------------------------------------ asynchronous API (UI)
    def begin_seek(self) -> int:
        """Start a new seek generation: extractions queued for older positions are cancelled by the first request of the new one."""
        with self._lock:
            self._gen += 1
            return self._gen

    def _queue_for(self, project) -> PriorityWorkQueue | None:
        if self._jobs is None:
            return None
        q = self._queue
        if q is not None and (self._queue_project is not project or q.closed):
            q.close()
            self._pending.clear()
            q = None
        if q is None:
            lim = lambda: max(1, min(2, int(getattr(self._limits_get() if self._limits_get else None, "foreground_workers", 2) or 2)))  # noqa: E731
            idle = lambda: bool(getattr(self._limits_get() if self._limits_get else None, "idle_work", True))  # noqa: E731
            q = PriorityWorkQueue(self._jobs, owner=f"frames:{getattr(project, 'project_id', '')}", run=self._run_task, name="frames", job_type="preview.frames", resource="",
                                  concurrency=lim, idle_allowed=idle, pressure_ok=self._pressure_ok, on_done=self._task_done, on_failed=self._task_failed, title="Preparing preview frames",
                                  urgent_priority=Priority.HIGH)
            self._queue, self._queue_project = q, project
        return q

    def _run_task(self, task: _FrameTask, cancel: Callable[[], bool]) -> bool:
        if cancel():
            raise JobCancelled()
        ok = self._extract(task, cancel)
        if not ok:
            raise RuntimeError("frame extraction failed")
        return True

    def _supersede(self, q: PriorityWorkQueue) -> None:
        stale = [k for k, t in self._pending.items() if t.gen < self._gen]
        if stale:
            q.cancel(stale)
            for k in stale:
                self._pending.pop(k, None)
            self.stats["superseded"] += len(stale)
            profiler.incr("preview.superseded", len(stale))

    def request(self, asset: Asset, src_time: float, on_ready: Callable[[Path], None] | None = None, *, prefetch: bool = False) -> Path | None:
        """The frame if it is already cached; otherwise None, with one extraction queued and ``on_ready(path)`` called (on the UI thread) when it arrives -- unless a newer
        seek has superseded this request by then."""
        project = self._project()
        if project is None or project.root is None:
            return None
        if asset.type is AssetType.IMAGE:
            return self.cached_path(asset, src_time)
        res = self._resolve(asset)
        if res is None:
            return None
        _p, src, used_proxy, width = res
        q = self._quantize(asset, src_time)
        out = self._out_path(project, asset, q, width)
        key = self._key(asset, q, width)
        if not prefetch:
            self.last_used_proxy = used_proxy
        if self._valid(project, asset, key, out):
            self.stats["cached"] += 1
            return out
        if self._is_failed(out):
            return None
        queue = self._queue_for(project)
        if queue is None:
            return self.frame_path(asset, src_time)
        with self._lock:
            if self._swept != self._gen and not prefetch:
                self._swept = self._gen
                self._supersede(queue)
            t = self._pending.get(key)
            if t is not None:
                self.stats["coalesced"] += 1
                if not prefetch:
                    t.gen, t.prefetch = self._gen, False
                if on_ready is not None and on_ready not in t.callbacks:
                    t.callbacks.append(on_ready)
            else:
                self.last_source = src
                t = _FrameTask(key, asset, src, q, out, width, self._gen, prefetch, self._fingerprint(project, asset), [on_ready] if on_ready else [])
                self._pending[key] = t
        queue.submit(key, t, IDLE if prefetch else VISIBLE)  # a prefetched frame the user now needs is promoted by the same call
        return None

    def prefetch_around(self, asset: Asset, src_time: float, count: int | None = None, direction: int = 1) -> int:
        """Queue the next ``count`` frames of this asset (mostly ahead of the playhead) at idle priority. Held by the queue while idle work is off or the machine is busy."""
        if asset.type is not AssetType.VIDEO:
            return 0
        n = self.prefetch_count() if count is None else count
        q0 = self._quantize(asset, src_time)
        ahead = max(1, int(n * 0.75))
        queued = 0
        for i in range(1, n + 1):
            t = q0 + (i * STEP if i <= ahead else -(i - ahead) * STEP) * (1 if direction >= 0 else -1)
            if t < 0 or (asset.duration and t > asset.duration):
                continue
            self.request(asset, t, None, prefetch=True)
            queued += 1
        return queued

    def _task_done(self, key: str, task: _FrameTask, _result) -> None:
        with self._lock:
            self._pending.pop(key, None)
            current = task.gen == self._gen or task.prefetch
            cbs = list(task.callbacks) if current else []
        if not current:
            self.stats["stale_discarded"] += 1
            return
        for cb in cbs:
            try:
                cb(task.out)
            except Exception:  # noqa: BLE001
                _log.debug("frame callback failed", exc_info=True)

    def _task_failed(self, key: str, task: _FrameTask, _reason: str) -> None:
        with self._lock:
            self._pending.pop(key, None)
        self._mark_failed(task.out)

    def pending_count(self) -> int:
        return len(self._pending)

    def release(self) -> None:
        """The project changed or closed: drop queued work, the failed set and the decoded frames."""
        with self._lock:
            q, self._queue, self._queue_project = self._queue, None, None
            self._pending.clear()
        if q is not None:
            q.close()
        self._failed.clear()
        self._fp.clear()
        self._known.clear()
        if self._tier is not None:
            self._tier.clear()
