"""ProxyManager: small, fast stand-ins for large media so editing stays smooth. Final renders always read the originals.

The timeline references the asset id only. ``project.proxies[asset_id]`` holds ``asset_id / original_path / proxy_path / proxy_status /
proxy_resolution`` (plus the original's size and modification time, so a proxy of a replaced file is detected as stale). Proxies are made with
the same frame rate and timing as the original, video only.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.core.events import EventBus, Topics
from app.jobs.job import Job, Priority
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger
from app.media.asset import Asset, AssetType
from app.performance.dependencies import asset_dep, stable_key
from app.performance.profiler import profiler
from app.performance.proxy_policy import ProxyPolicy, ProxyRecommendation, estimate_bytes
from app.performance.settings import PerformanceSettings
from app.project.project import Project
from app.rendering.commands import SetProxyRecordCommand
from app.rendering.errors import RenderError
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.models import ProxyRef
from app.rendering.presets import PROXY_RESOLUTIONS
from app.rendering.probe import MediaProbeService

_log = get_logger(__name__)
STATUSES = ("NONE", "QUEUED", "READY", "FAILED", "CANCELED", "STALE")
PERFORMANCE_UPDATED = "performance.updated"  # published by the performance service (same literal; imported there would invert the layering)
AUTO_DELAY_S = 1.0  # events that can change what an automatic policy wants are coalesced for this long
AUTO_RETRY_S = 30.0  # how long a deferred automatic run waits before it looks again


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
        self.policy = ProxyPolicy()
        self.in_use: Callable[[Path], bool] | None = None  # installed by the render service: a running render / preview is reading this proxy
        self._perf_getter: Callable[[], Any] | None = None
        self._auto_timer: threading.Timer | None = None
        self._auto_lock = threading.Lock()
        self._auto_notified: set[str] = set()
        self.last_automatic: dict[str, Any] = {"state": "idle", "queued": 0, "skipped": [], "deferred": ""}
        for topic in (Topics.PROJECT_OPENED, Topics.PROJECT_CHANGED, PERFORMANCE_UPDATED):
            bus.subscribe(topic, self._on_event)

    # ------------------------------------------------------------------ performance plumbing (optional)
    def attach_performance(self, getter: Callable[[], Any]) -> None:
        """Use the performance service for the policy / profile (global + project overrides), limits, pressure and the cache index. Without it proxies behave as before (manual)."""
        self._perf_getter = getter

    def _perf(self) -> Any:
        try:
            return self._perf_getter() if self._perf_getter else None
        except Exception:  # noqa: BLE001
            return None

    def perf_settings(self) -> PerformanceSettings:
        perf = self._perf()
        try:
            return perf.settings() if perf is not None else PerformanceSettings()
        except Exception:  # noqa: BLE001
            return PerformanceSettings()

    def _limits(self) -> Any:
        perf = self._perf()
        try:
            return perf.limits() if perf is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _cache(self) -> Any:
        perf = self._perf()
        return getattr(perf, "cache", None) if perf is not None else None

    def _pressure_normal(self) -> bool:
        mon = getattr(self._perf(), "monitor", None)
        try:
            s = mon.latest() if mon is not None else None
            return True if s is None else mon.pressure(s).get("overall", "normal") == "normal"
        except Exception:  # noqa: BLE001
            return True

    def _hardware(self) -> dict | None:
        perf = self._perf()
        try:
            return perf.hardware_summary() if perf is not None else None
        except Exception:  # noqa: BLE001
            return None

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
    def proxy_dir(self) -> Path:
        return self._project().root / "proxies"  # type: ignore[operator]

    def _needs(self, a: Asset, res: str, regenerate: bool = False, automatic: bool = False) -> bool:
        cur = self.record(a.id)
        if cur is not None and not regenerate:
            if cur.proxy_status == "READY" and cur.proxy_resolution == res and self._fresh(cur) and self._original_unchanged(a, cur):
                return False  # a valid proxy of the same size is reused; a missing / evicted file or another size is regenerated
            if automatic and cur.proxy_status in ("FAILED", "CANCELED"):
                return False  # the background policy never retries what failed or was cancelled by the user: that is a visible choice (Retry)
        old = self._jobs_by_asset.get(a.id)
        return old is None or old.status.is_terminal

    def plan(self, asset_ids: list[str] | None = None, resolution: str | None = None, only_large: bool = True, regenerate: bool = False) -> ProxyRecommendation:
        """What a batch would cost: the policy's decision per asset (the reasons), the estimated size and whether it fits the disk and the proxy cache limit."""
        project = self._project()
        res = resolution or project.render_settings.proxy_resolution
        if res not in PROXY_RESOLUTIONS:
            raise RenderError(f"Unsupported proxy size “{res}”. Use 540p, 720p or 1080p.", kind="invalid_settings")
        assets = [project.assets.require(i) for i in asset_ids] if asset_ids else self.candidates(res, only_large)
        todo = [a for a in assets if a.type is AssetType.VIDEO and self._needs(a, res, regenerate)]
        rec = self.policy.recommend(project, todo, self.perf_settings(), self._hardware(), resolution=res)
        explicit = {a.id for a in todo}
        for d in rec.decisions:  # an explicit request is honoured even when the policy would not suggest it on its own
            if d.asset_id in explicit and not d.proxy:
                d.proxy, d.estimated_bytes = True, estimate_bytes(project.assets.require(d.asset_id).duration, res)
                d.reasons.append("requested")
        rec.estimated_bytes = sum(d.estimated_bytes for d in rec.decisions if d.proxy)
        lim = self._limits()
        rec.disk = self.policy.check_disk(rec.estimated_bytes, self.proxy_dir(), cache_limit_bytes=(lim.category_bytes.get("proxies") if lim else None),
                                          existing_proxy_bytes=self.storage_report()["total_bytes"])
        return rec

    def generate(self, asset_ids: list[str] | None = None, resolution: str | None = None, only_large: bool = True, regenerate: bool = False, *,
                 priority: Priority = Priority.MEDIUM, automatic: bool = False) -> list[Job]:
        project = self._project()
        res = resolution or project.render_settings.proxy_resolution
        if res not in PROXY_RESOLUTIONS:
            raise RenderError(f"Unsupported proxy size “{res}”. Use 540p, 720p or 1080p.", kind="invalid_settings")
        assets = [project.assets.require(i) for i in asset_ids] if asset_ids else self.candidates(res, only_large)
        todo = [a for a in assets if a.type is AssetType.VIDEO and self._needs(a, res, regenerate, automatic)]
        if not todo:
            return []
        est = sum(estimate_bytes(a.duration, res) for a in todo)
        lim = self._limits()
        chk = self.policy.check_disk(est, self.proxy_dir(), cache_limit_bytes=(lim.category_bytes.get("proxies") if lim else None), existing_proxy_bytes=self.storage_report()["total_bytes"])
        if not chk.ok:
            raise RenderError(chk.message, stage="Proxy", kind="disk_space", possible_issue="Free some disk space, choose a smaller proxy size, or create proxies for fewer clips.")
        if chk.warning or chk.over_cache_limit:
            self._bus.publish(Topics.STATUS, message=chk.warning or "These proxies exceed the proxy cache limit; the oldest unused proxies may be cleaned up.")
        self.cleanup_partial()
        return [self._submit(a, res, priority) for a in todo]

    def _submit(self, asset: Asset, res: str, priority: Priority = Priority.MEDIUM) -> Job:
        project = self._project()
        src = project.asset_path(asset)
        try:
            st = src.stat()
            size, mtime = st.st_size, st.st_mtime_ns
        except OSError:
            size = mtime = 0
        out_dir = project.root / "proxies"  # type: ignore[operator]
        final = out_dir / f"{asset.id}_{res}.mp4"
        previous = self.record(asset.id)
        replaced = Path(previous.proxy_path) if previous and previous.proxy_path and Path(previous.proxy_path) != final else None  # another size of the same asset: removed once the new one is READY
        rec = ProxyRecord(asset.id, str(src), str(final), "QUEUED", res, 0, 0, size, mtime, 0, "", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._set(asset.id, rec)
        duration = asset.duration or 0.0
        tmp = out_dir / f".{asset.id}_{res}.part.mp4"

        def work(ctx) -> dict:
            if not src.is_file():
                raise RenderError(f"The original “{asset.name}” is missing, so no proxy can be made.", stage="Proxy", kind="missing_media")
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp.unlink(missing_ok=True)
            h = PROXY_RESOLUTIONS[res]
            args = [self.ff.ffmpeg(), "-hide_banner", "-nostats", "-v", "warning", "-progress", "pipe:1", "-stats_period", "0.25", "-i", str(src), "-map", "0:v:0", "-an",
                    "-vf", f"scale=-2:'min({h},ih)':flags=bicubic", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "15",
                    "-movflags", "+faststart", "-y", str(tmp)]
            with profiler.timer("proxy.generate"):
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
            cache = self._cache()
            if cache is not None:
                cache.register_external("proxies", final, key=stable_key("proxy", asset.id, res), deps={asset_dep(asset.id): f"s:{size}:{mtime}"})
            if replaced is not None and not self.is_protected(replaced):
                replaced.unlink(missing_ok=True)

        def failed(job: Job) -> None:
            tmp.unlink(missing_ok=True)
            cur = self.record(asset.id) or rec
            cur.proxy_status, cur.error = "FAILED", job.error or "The proxy could not be created."
            self._set(asset.id, cur)

        def cancelled(job: Job) -> None:
            tmp.unlink(missing_ok=True)  # a partial file never stays behind
            cur = self.record(asset.id) or rec
            cur.proxy_status, cur.error = "CANCELED", ""
            self._set(asset.id, cur)

        job = self._jobs.submit("proxy.generate", work, title=f"Proxy {res}: {asset.name}", on_complete=done, on_error=failed, on_cancel=cancelled, priority=priority,
                                dedupe_key=f"proxy:{project.project_id}:{asset.id}:{res}", resource="proxy", owner=project.project_id)
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

    def is_protected(self, path: Path | str) -> bool:
        """A proxy that must not be removed now: a render / preview is reading it, or it is still being made."""
        p = os.path.normcase(os.path.abspath(path))
        for aid, job in self._jobs_by_asset.items():
            if not job.status.is_terminal:
                rec = self.record(aid)
                if rec is not None and rec.proxy_path and os.path.normcase(os.path.abspath(rec.proxy_path)) == p:
                    return True
        try:
            return bool(self.in_use and self.in_use(Path(path)))
        except Exception:  # noqa: BLE001 - when unsure, keep the file
            return True

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

    # ------------------------------------------------------------------ storage
    def storage_report(self) -> dict[str, Any]:
        """Disk use of the proxies: per proxy, by status, files nobody references (orphans) and leftovers of an interrupted encode (partial)."""
        project = self._project()
        folder = self.proxy_dir()
        recs = self.records()
        referenced: set[str] = set()
        rows: list[dict[str, Any]] = []
        by_status: dict[str, dict[str, int]] = {}
        for aid, r in recs.items():
            p = Path(r.proxy_path) if r.proxy_path else None
            exists = bool(p and p.is_file())
            size = p.stat().st_size if exists else 0
            if p is not None:
                referenced.add(os.path.normcase(os.path.abspath(p)))
            a = project.assets.get(aid)
            orig_ok = bool(a is not None and project.asset_path(a).is_file())
            st = r.proxy_status if (r.proxy_status != "READY" or exists) else "STALE"  # READY on paper but the file is gone
            rows.append({"asset_id": aid, "name": a.name if a else aid, "resolution": r.proxy_resolution, "status": st, "bytes": size, "path": r.proxy_path, "exists": exists,
                         "regenerable": orig_ok, "in_use": bool(p and exists and self.is_protected(p))})
            b = by_status.setdefault(st, {"count": 0, "bytes": 0})
            b["count"] += 1
            b["bytes"] += size
        orphans: list[dict[str, Any]] = []
        partial: list[dict[str, Any]] = []
        if folder.is_dir():
            for f in folder.iterdir():
                try:
                    if not f.is_file():
                        continue
                    if f.name.startswith(".") and ".part" in f.name:
                        partial.append({"path": str(f), "bytes": f.stat().st_size})
                    elif f.suffix.lower() == ".mp4" and os.path.normcase(os.path.abspath(f)) not in referenced:
                        orphans.append({"path": str(f), "bytes": f.stat().st_size})
                except OSError:
                    continue
        total = sum(r["bytes"] for r in rows) + sum(o["bytes"] for o in orphans) + sum(x["bytes"] for x in partial)
        free = self.policy._disk_free(folder)
        lim = self._limits()
        return {"folder": str(folder), "proxies": rows, "by_status": by_status, "orphans": orphans, "partial": partial, "total_bytes": total,
                "free_bytes": free, "cache_limit_bytes": lim.category_bytes.get("proxies") if lim else None}

    def cleanup_partial(self) -> int:
        """Delete half-written ``.part`` files of encodes that are not running (after a crash, a cancel or a failure). Returns how many were removed."""
        folder = self.proxy_dir()
        if not folder.is_dir():
            return 0
        busy = {os.path.normcase(os.path.abspath(folder / f".{aid}_{self.record(aid).proxy_resolution}.part.mp4")) for aid, j in self._jobs_by_asset.items()
                if not j.status.is_terminal and self.record(aid) is not None}
        n = 0
        for f in folder.iterdir():
            if f.is_file() and f.name.startswith(".") and ".part" in f.name and os.path.normcase(os.path.abspath(f)) not in busy:
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
        return n

    def clean_orphans(self) -> int:
        """Delete proxy-folder files no record points to (e.g. the old size after a size change that never finished). Only ``<project>/proxies`` is touched."""
        rep = self.storage_report()
        n = 0
        for o in rep["orphans"] + rep["partial"]:
            p = Path(o["path"])
            if not self.is_protected(p):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        return n

    def remove_selected(self, asset_ids: list[str]) -> dict[str, Any]:
        """Delete these assets' proxies when that is safe: the original must still exist (so the proxy can be made again) and no render / preview may be using the proxy."""
        project = self._project()
        removed: list[str] = []
        skipped: dict[str, str] = {}
        freed = 0
        for aid in asset_ids:
            rec = self.record(aid)
            a = project.assets.get(aid)
            if rec is None:
                skipped[aid] = "no proxy"
                continue
            p = Path(rec.proxy_path) if rec.proxy_path else None
            if a is None or not project.asset_path(a).is_file():
                skipped[aid] = "the original is missing: this proxy could not be made again"
                continue
            job = self._jobs_by_asset.get(aid)
            if (job is not None and not job.status.is_terminal) or (p is not None and self.is_protected(p)):
                skipped[aid] = "in use by a running proxy, render or preview"
                continue
            size = 0
            if p is not None:
                try:
                    size = p.stat().st_size if p.is_file() else 0
                    p.unlink(missing_ok=True)
                except OSError as exc:
                    skipped[aid] = f"could not be deleted: {exc.strerror or exc}"
                    continue
            self._set(aid, None)
            removed.append(aid)
            freed += size
        return {"removed": removed, "skipped": skipped, "freed_bytes": freed}

    def invalidate_other_resolutions(self, resolution: str) -> int:
        """Proxies made at another size than ``resolution`` are stale: marked STALE and their files removed unless in use. Same-size proxies are kept (reused)."""
        n = 0
        for aid, r in self.records().items():
            if r.proxy_status == "READY" and r.proxy_resolution != resolution:
                p = Path(r.proxy_path) if r.proxy_path else None
                if p is not None and p.is_file() and not self.is_protected(p):
                    p.unlink(missing_ok=True)
                r.proxy_status = "STALE"
                self._set(aid, r)
                n += 1
        return n

    # ------------------------------------------------------------------ policy: status and the automatic mode
    def status(self) -> dict[str, Any]:
        """Everything the UI needs in one call: the policy and profile in effect, counts, bytes, the last automatic decision and the reasons behind it."""
        s = self.perf_settings()
        rep = self.storage_report()
        return {"policy": s.proxy_policy, "profile": s.proxy_profile, "resolution": self.policy.resolution(s, None, self._project().render_settings.proxy_resolution), "summary": self.summary(),
                "total_bytes": rep["total_bytes"], "free_bytes": rep["free_bytes"], "partial_files": len(rep["partial"]), "orphan_files": len(rep["orphans"]), "automatic": dict(self.last_automatic)}

    def set_project_policy(self, policy: str | None = None, profile: str | None = None) -> None:
        """Per-project proxy policy / profile (stored as performance overrides on the project; undoable)."""
        perf = self._perf()
        if perf is None:
            raise RenderError("Performance settings are not available.", kind="invalid_settings")
        cur = dict(perf.project_overrides())
        for k, v in (("proxy_policy", policy), ("proxy_profile", profile)):
            if v is not None:
                cur[k] = v
        perf.set_project_overrides(cur)

    def _on_event(self, topic: str, payload: dict) -> None:
        if topic == Topics.PROJECT_OPENED:
            try:
                self.cleanup_partial()  # leftovers of an encode that was interrupted by a crash / forced quit
            except Exception:  # noqa: BLE001
                _log.debug("partial proxy clean-up failed", exc_info=True)
        if topic == Topics.PROJECT_CHANGED and payload.get("scope") != "assets":
            return
        try:
            if self.perf_settings().proxy_policy != "automatic":
                return
        except Exception:  # noqa: BLE001 - no project open / settings unavailable
            return
        self._schedule_auto(AUTO_DELAY_S)

    def _schedule_auto(self, delay: float) -> None:
        with self._auto_lock:
            if self._auto_timer is not None:
                return
            t = threading.Timer(delay, self._auto_fire)
            t.daemon = True
            self._auto_timer = t
            t.start()

    def _auto_fire(self) -> None:
        with self._auto_lock:
            self._auto_timer = None
        self._jobs.dispatch(self.run_automatic)  # commands must be applied on the thread callbacks run on (the UI thread in the app)

    def run_automatic(self) -> dict[str, Any]:
        """Apply the AUTOMATIC policy now: queue the recommended proxies at LOW priority, or say (visibly) why not. OFF / MANUAL do nothing here."""
        st: dict[str, Any] = {"state": "idle", "queued": 0, "skipped": [], "deferred": ""}
        self.last_automatic = st
        try:
            project = self._project()
        except Exception:  # noqa: BLE001
            return st
        s = self.perf_settings()
        if s.proxy_policy != "automatic":
            st["state"] = s.proxy_policy
            return st
        lim = self._limits()
        if lim is not None and not (lim.background_proxy and lim.idle_work):
            st.update(state="deferred", deferred="Background proxy creation is switched off by the performance profile; create proxies manually.")
            return st
        if not self._pressure_normal():
            st.update(state="deferred", deferred="The computer is busy; proxies will be created when it is idle.")
            self._schedule_auto(AUTO_RETRY_S)
            return st
        res = self.policy.resolution(s, project)
        self.invalidate_other_resolutions(res)  # a size change: proxies of the old size are stale (same-size ones are reused)
        rec = self.policy.recommend(project, None, s, self._hardware(), resolution=res)
        todo = [project.assets.require(d.asset_id) for d in rec.decisions if d.proxy and self._needs(project.assets.require(d.asset_id), res, False, True)]
        if not todo:
            st["state"] = "up_to_date"
            return st
        est = sum(estimate_bytes(a.duration, res) for a in todo)
        chk = self.policy.check_disk(est, self.proxy_dir(), cache_limit_bytes=(lim.category_bytes.get("proxies") if lim else None), existing_proxy_bytes=self.storage_report()["total_bytes"])
        reason = chk.message if not chk.ok else ("These proxies would exceed the proxy cache limit; clean up proxies or raise the limit." if chk.over_cache_limit else "")
        if reason:
            st.update(state="skipped", skipped=[{"asset_id": a.id, "name": a.name, "reason": reason} for a in todo])
            if reason not in self._auto_notified:  # said once, not on every event
                self._auto_notified.add(reason)
                self._bus.publish(Topics.STATUS, message=f"Automatic proxies were not created: {reason}")
            return st
        jobs = self.generate([a.id for a in todo], res, only_large=False, priority=Priority.LOW, automatic=True)
        st.update(state="queued", queued=len(jobs))
        return st
