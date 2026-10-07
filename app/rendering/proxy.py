"""ProxyManager: small, fast stand-ins for large media so editing stays smooth. Final renders always read the originals.

The timeline references the asset id only. ``project.proxies[asset_id]`` holds ``asset_id / original_path / proxy_path / proxy_status /
proxy_resolution`` (plus the original's size and modification time, so a proxy of a replaced file is detected as stale). Proxies are made with
the same frame rate and timing as the original, video only.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.core.events import EventBus, Topics
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger
from app.media.asset import Asset, AssetType
from app.project.project import Project
from app.rendering.commands import SetProxyRecordCommand
from app.rendering.errors import RenderError
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.models import ProxyRef
from app.rendering.presets import PROXY_RESOLUTIONS
from app.rendering.probe import MediaProbeService

_log = get_logger(__name__)
STATUSES = ("NONE", "QUEUED", "READY", "FAILED", "CANCELED", "STALE")


@dataclass
class ProxyRecord:
    asset_id: str
    original_path: str
    proxy_path: str = ""
    proxy_status: str = "NONE"
    proxy_resolution: str = "720p"
    width: int = 0
    height: int = 0
    source_size: int = 0
    source_mtime_ns: int = 0
    size_bytes: int = 0
    error: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProxyRecord":
        from dataclasses import fields

        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class ProxyManager:
    def __init__(self, project_getter: Callable[[], Project], jobs: JobManager, bus: EventBus, apply_command: Callable, ffmpeg: FFmpegService, probe: MediaProbeService) -> None:
        self._project, self._jobs, self._bus, self._apply = project_getter, jobs, bus, apply_command
        self.ff, self.probe = ffmpeg, probe
        self._jobs_by_asset: dict[str, Job] = {}

    # ------------------------------------------------------------------ queries
    def record(self, asset_id: str) -> ProxyRecord | None:
        d = self._project().proxies.get(asset_id)
        return ProxyRecord.from_dict(d) if d else None

    def records(self) -> dict[str, ProxyRecord]:
        return {k: ProxyRecord.from_dict(v) for k, v in self._project().proxies.items()}

    def refs(self) -> dict[str, ProxyRef]:
        """Usable proxies as the snapshot sees them."""
        out = {}
        for k, r in self.records().items():
            out[k] = ProxyRef(r.proxy_path, r.proxy_resolution, r.proxy_status if self._fresh(r) else "STALE", r.width, r.height, r.source_size, r.source_mtime_ns)
        return out

    def _fresh(self, r: ProxyRecord) -> bool:
        if r.proxy_status != "READY":
            return False
        return bool(r.proxy_path) and Path(r.proxy_path).is_file()

    def proxy_path_for(self, asset: Asset) -> Path | None:
        """The proxy to show while editing (None = use the original)."""
        r = self.record(asset.id)
        if r and self._fresh(r) and self._original_unchanged(asset, r):
            return Path(r.proxy_path)
        return None

    def _original_unchanged(self, asset: Asset, r: ProxyRecord) -> bool:
        try:
            st = self._project().asset_path(asset).stat()
        except OSError:
            return True
        return not r.source_size or (st.st_size == r.source_size and st.st_mtime_ns == r.source_mtime_ns)

    def candidates(self, resolution: str | None = None, only_large: bool = True) -> list[Asset]:
        """Video assets worth a proxy: bigger than the proxy size (all videos with ``only_large=False``); alpha videos are skipped (a proxy would lose transparency)."""
        res = resolution or self._project().render_settings.proxy_resolution
        h = PROXY_RESOLUTIONS.get(res, 720)
        out = []
        for a in self._project().assets.all():
            if a.type is not AssetType.VIDEO:
                continue
            if (a.extra or {}).get("probe", {}).get("has_alpha"):
                continue
            if only_large and min(a.width or 0, a.height or 0) <= h:
                continue
            out.append(a)
        return out

    def summary(self) -> dict:
        recs = self.records()
        running = sum(1 for j in self._jobs_by_asset.values() if not j.status.is_terminal)
        return {"total": len(recs), "ready": sum(1 for r in recs.values() if r.proxy_status == "READY"), "queued": running, "failed": sum(1 for r in recs.values() if r.proxy_status == "FAILED"),
                "bytes": sum(r.size_bytes for r in recs.values() if r.proxy_status == "READY")}

    # ------------------------------------------------------------------ generate
    def generate(self, asset_ids: list[str] | None = None, resolution: str | None = None, only_large: bool = True, regenerate: bool = False) -> list[Job]:
        project = self._project()
        res = resolution or project.render_settings.proxy_resolution
        if res not in PROXY_RESOLUTIONS:
            raise RenderError(f"Unsupported proxy size “{res}”. Use 540p, 720p or 1080p.", kind="invalid_settings")
        assets = [project.assets.require(i) for i in asset_ids] if asset_ids else self.candidates(res, only_large)
        jobs = []
        for a in assets:
            if a.type is not AssetType.VIDEO:
                continue
            cur = self.record(a.id)
            if cur and cur.proxy_status == "READY" and cur.proxy_resolution == res and self._fresh(cur) and self._original_unchanged(a, cur) and not regenerate:
                continue
            old = self._jobs_by_asset.get(a.id)
            if old is not None and not old.status.is_terminal:
                continue
            jobs.append(self._submit(a, res))
        return jobs

    def _submit(self, asset: Asset, res: str) -> Job:
        project = self._project()
        src = project.asset_path(asset)
        try:
            st = src.stat()
            size, mtime = st.st_size, st.st_mtime_ns
        except OSError:
            size = mtime = 0
        out_dir = project.root / "proxies"  # type: ignore[operator]
        final = out_dir / f"{asset.id}_{res}.mp4"
        rec = ProxyRecord(asset.id, str(src), str(final), "QUEUED", res, 0, 0, size, mtime, 0, "", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._set(asset.id, rec)
        duration = asset.duration or 0.0

        def work(ctx) -> dict:
            if not src.is_file():
                raise RenderError(f"The original “{asset.name}” is missing, so no proxy can be made.", stage="Proxy", kind="missing_media")
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / f".{asset.id}_{res}.part.mp4"
            tmp.unlink(missing_ok=True)
            h = PROXY_RESOLUTIONS[res]
            args = [self.ff.ffmpeg(), "-hide_banner", "-nostats", "-v", "warning", "-progress", "pipe:1", "-stats_period", "0.25", "-i", str(src), "-map", "0:v:0", "-an",
                    "-vf", f"scale=-2:'min({h},ih)':flags=bicubic", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "15",
                    "-movflags", "+faststart", "-y", str(tmp)]
            res_ = self.ff.run_progress(args, on_progress=lambda p: ctx.report(min(99.0, 100.0 * p.out_time / duration) if duration else None, f"Proxy: {asset.name}"),
                                        cancel=ctx.job.cancel_event, stall_timeout=600)
            if res_.cancelled:
                tmp.unlink(missing_ok=True)
                from app.core.exceptions import JobCancelled

                raise JobCancelled()
            if res_.returncode != 0 or not tmp.is_file():
                tmp.unlink(missing_ok=True)
                raise RenderError(f"The proxy for “{asset.name}” could not be created.", stage="Proxy", kind="ffmpeg_failed", details=res_.stderr_tail[-800:])
            info = self.probe.probe(tmp, use_cache=False)
            shutil.move(str(tmp), str(final))
            return {"w": info.width or 0, "h": info.height or 0, "bytes": final.stat().st_size}

        def done(job: Job) -> None:
            r = job.result
            cur = self.record(asset.id) or rec
            cur.proxy_status, cur.width, cur.height, cur.size_bytes, cur.error = "READY", r["w"], r["h"], r["bytes"], ""
            self._set(asset.id, cur)

        def failed(job: Job) -> None:
            cur = self.record(asset.id) or rec
            cur.proxy_status, cur.error = "FAILED", job.error or "The proxy could not be created."
            self._set(asset.id, cur)

        def cancelled(job: Job) -> None:
            cur = self.record(asset.id) or rec
            cur.proxy_status, cur.error = "CANCELED", ""
            self._set(asset.id, cur)

        job = self._jobs.submit("proxy.generate", work, title=f"Proxy {res}: {asset.name}", on_complete=done, on_error=failed, on_cancel=cancelled)
        self._jobs_by_asset[asset.id] = job
        return job

    def _set(self, asset_id: str, rec: ProxyRecord | None) -> None:
        self._apply(SetProxyRecordCommand(self._project(), asset_id, rec.to_dict() if rec else None))
        self._bus.publish(Topics.PROXY_CHANGED, asset_id=asset_id)

    # ------------------------------------------------------------------ control
    def cancel(self, asset_id: str | None = None) -> int:
        n = 0
        for aid, job in list(self._jobs_by_asset.items()):
            if (asset_id is None or aid == asset_id) and not job.status.is_terminal:
                n += bool(self._jobs.cancel(job.id))
        return n

    def retry(self, asset_id: str) -> Job | None:
        rec = self.record(asset_id)
        if rec is None or rec.proxy_status not in ("FAILED", "CANCELED", "STALE"):
            return None
        jobs = self.generate([asset_id], rec.proxy_resolution, only_large=False, regenerate=True)
        return jobs[0] if jobs else None

    def delete(self, asset_ids: list[str] | None = None) -> int:
        """Remove proxy files and records. Originals are never touched."""
        self.cancel()
        n = 0
        for aid, rec in self.records().items():
            if asset_ids is not None and aid not in asset_ids:
                continue
            if rec.proxy_path:
                try:
                    Path(rec.proxy_path).unlink(missing_ok=True)
                except OSError:
                    pass
            self._set(aid, None)
            n += 1
        return n

    def regenerate(self, asset_ids: list[str] | None = None, resolution: str | None = None) -> list[Job]:
        ids = asset_ids or list(self.records())
        self.delete(ids)
        return self.generate(ids or None, resolution, only_large=False if ids else True, regenerate=True)

    def refresh_staleness(self) -> int:
        """Mark proxies whose original changed (or whose file vanished) as STALE. Returns how many changed."""
        n = 0
        project = self._project()
        for aid, rec in self.records().items():
            a = project.assets.get(aid)
            if rec.proxy_status == "READY" and (a is None or not self._original_unchanged(a, rec) or not Path(rec.proxy_path).is_file()):
                rec.proxy_status = "STALE"
                self._set(aid, rec)
                n += 1
        return n
