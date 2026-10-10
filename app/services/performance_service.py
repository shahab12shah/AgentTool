"""PerformanceService: the one place that turns the performance settings into behaviour.

It owns the resource monitor and the per-project ``MediaCacheManager``, resolves the settings (global + project overrides) into concrete limits, pushes them
into the job scheduler / profiler / cache, runs hardware detection as a LOW-priority background job (never at startup), and produces the diagnostic report.
Everything here is optional plumbing: if any part fails the application keeps working with the previous behaviour.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings
from app.core.events import EventBus, Topics
from app.jobs.job import Job, Priority
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.performance import bottleneck_report as report
from app.performance.cache_manager import MediaCacheManager
from app.performance.memory_monitor import MemoryMonitor
from app.performance.profiler import Profiler, profiler as default_profiler
from app.performance.resource_monitor import ResourceMonitor
from app.performance.settings import CACHE_CATEGORIES, PerformanceSettings, ResolvedLimits, resolve
from app.project.phase9_commands import SetPerformanceOverridesCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager

_log = get_logger(__name__)
PERFORMANCE_UPDATED = "performance.updated"  # payload: kind = settings|hardware|cache|limits


class PerformanceService:
    def __init__(self, projects: ProjectManager, jobs: JobManager, bus: EventBus, execute_command: Callable, settings_getter: Callable[[], Settings],
                 settings_saver: Callable[[Settings], None], hardware_getter: Callable[[], Any] | None = None, profiler: Profiler = default_profiler,
                 proxy_in_use: Callable[[Path], bool] | None = None) -> None:
        self._projects, self._jobs, self._bus, self._execute = projects, jobs, bus, execute_command
        self._settings, self._save_settings = settings_getter, settings_saver
        self._hardware_getter = hardware_getter
        self.profiler = profiler
        self.monitor = ResourceMonitor(lambda: self._projects.current.root if self._projects.current else None)
        self.memory = MemoryMonitor()
        self.cache: MediaCacheManager | None = None
        self._cache_root: Path | None = None
        self._proxy_in_use = proxy_in_use
        self._detect_job: Job | None = None
        self.last_cleanup: str = ""
        self._started = False
        bus.subscribe(Topics.PROJECT_OPENED, self._on_project_opened)
        bus.subscribe(Topics.PROJECT_CLOSED, self._on_project_closed)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Apply the settings and start light background sampling. Safe to call more than once; never blocks."""
        if self._started:
            return
        self._started = True
        self.memory.mark_baseline()
        self.apply()
        try:
            self.monitor.start(interval=10.0)
        except Exception:  # noqa: BLE001
            _log.warning("Resource monitor could not start", exc_info=True)

    def shutdown(self) -> None:
        try:
            self.monitor.stop()
        finally:
            self._close_cache()

    # ------------------------------------------------------------------ settings
    def global_settings(self) -> PerformanceSettings:
        return PerformanceSettings.from_dict(self._settings().performance)

    def project_overrides(self) -> dict[str, Any]:
        p = self._projects.current
        return dict(p.performance_overrides) if p is not None else {}

    def settings(self) -> PerformanceSettings:
        """What is in effect: the global settings with the open project's overrides applied."""
        return self.global_settings().with_overrides(self.project_overrides())

    def limits(self) -> ResolvedLimits:
        s = self.monitor.latest() or self.monitor.sample()
        return resolve(self.settings(), cpu_count=os.cpu_count() or 1, ram_bytes=s.system_total_bytes, disk_free_bytes=s.disk_free_bytes)

    def update_global(self, new: PerformanceSettings) -> None:
        cfg = self._settings()
        cfg.performance = new.sanitized().to_dict()
        self._save_settings(cfg)
        self.apply()
        self._bus.publish(PERFORMANCE_UPDATED, kind="settings")

    def set_project_overrides(self, overrides: dict[str, Any]) -> None:
        p = self._projects.current
        if p is None:
            return
        clean = {k: v for k, v in (overrides or {}).items() if k in PerformanceSettings.from_dict({}).to_dict()}
        self._execute(SetPerformanceOverridesCommand(p, clean))
        self.apply()
        self._bus.publish(PERFORMANCE_UPDATED, kind="settings")

    def apply(self) -> ResolvedLimits:
        """Push the effective settings into the profiler, the scheduler and the cache. Failures are logged, never raised."""
        s = self.settings()
        lim = self.limits()
        try:
            self.profiler.configure(enabled=s.metrics_enabled, slow_threshold_s=s.slow_operation_ms / 1000.0)
            self._jobs.apply_resolved_limits(lim)
            self._jobs.set_pressure_provider(self.monitor.pressure)
            if self.cache is not None:
                self.cache.refresh_limits()
        except Exception:  # noqa: BLE001
            _log.warning("Applying performance settings failed", exc_info=True)
        log_event(_log, "perf.limits_applied", profile=lim.profile, workers=lim.background_workers, memory_cache_mb=lim.memory_cache_bytes // 1048576)
        return lim

    # ------------------------------------------------------------------ cache
    def _on_project_opened(self, topic: str, payload: dict) -> None:  # noqa: ARG002
        self._close_cache()
        p = self._projects.current
        if p is None or p.root is None:
            return
        try:
            root = p.root / "cache"
            self.cache = MediaCacheManager(root, limits=self.limits, monitor=self.monitor, deny_roots=lambda: [r for r in (self._project_media_roots()) if r],
                                           protected=self._protected)
            self._cache_root = root
        except Exception:  # noqa: BLE001 - a cache problem must never stop a project from opening
            _log.warning("Cache manager could not start", exc_info=True)
            self.cache = None
        self.apply()

    def _on_project_closed(self, topic: str, payload: dict) -> None:  # noqa: ARG002
        pid = payload.get("project_id")
        try:
            if pid:
                self._jobs.cancel_owner(pid)
            self._jobs.cancel_all_low()
        except Exception:  # noqa: BLE001
            _log.debug("Cancelling background jobs on close failed", exc_info=True)
        self._close_cache()

    def _close_cache(self) -> None:
        c, self.cache = self.cache, None
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                _log.debug("Cache close failed", exc_info=True)

    def _project_media_roots(self) -> list[Path | None]:
        p = self._projects.current
        return [p.root / "media" if p and p.root else None, p.root / "renders" if p and p.root else None]

    def _protected(self, entry: Any) -> bool:
        if self._proxy_in_use is None:
            return False
        try:
            return bool(entry.path) and self._proxy_in_use(Path(entry.path))
        except Exception:  # noqa: BLE001
            return True  # when unsure, keep it

    def cache_stats(self) -> dict[str, Any]:
        if self.cache is None:
            return {"categories": {c: {"entries": 0, "bytes": 0} for c in CACHE_CATEGORIES}, "total_bytes": 0, "available": False}
        st = self.cache.get_cache_stats()
        st["available"] = True
        st["last_cleanup"] = self.last_cleanup or st.get("last_cleanup", "")
        return st

    def cleanup_cache(self, force: bool = True) -> dict[str, Any]:
        if self.cache is None:
            return {}
        rep = self.cache.cleanup(force=force)
        self.last_cleanup = time.strftime("%Y-%m-%d %H:%M:%S")
        self._bus.publish(PERFORMANCE_UPDATED, kind="cache")
        return rep.to_dict() if hasattr(rep, "to_dict") else {}

    def clear_rebuildable_cache(self, categories: list[str] | None = None) -> dict[str, Any]:
        if self.cache is None:
            return {}
        rep = self.cache.clear_rebuildable_cache(categories)
        self.last_cleanup = time.strftime("%Y-%m-%d %H:%M:%S")
        self._bus.publish(PERFORMANCE_UPDATED, kind="cache")
        return rep.to_dict() if hasattr(rep, "to_dict") else {}

    # ------------------------------------------------------------------ hardware
    @property
    def hardware(self) -> Any | None:
        return self._hardware_getter() if self._hardware_getter else None

    def detect_hardware_async(self, force: bool = False) -> Job | None:
        """Probe the machine in a LOW-priority job (the heavy part tests encoders / decoders). Never called at startup."""
        hw = self.hardware
        if hw is None:
            return None

        def work(ctx):
            if force:
                hw.forget()
            return hw.detect_all(lambda f, m: ctx.report(100.0 * f, m), ctx.job.cancel_event)

        def done(job: Job) -> None:
            self._bus.publish(PERFORMANCE_UPDATED, kind="hardware")

        return self._jobs.submit("performance.hardware", work, title="Checking hardware capabilities", priority=Priority.LOW, dedupe_key="performance.hardware", on_complete=done)

    def hardware_summary(self) -> dict[str, Any]:
        hw = self.hardware
        if hw is None:
            return {"text": "Hardware detection is not available.", "pending": ["all"]}
        try:
            return hw.summary(allow_detect=False)
        except Exception:  # noqa: BLE001
            _log.warning("Hardware summary failed", exc_info=True)
            return {"text": "Hardware detection failed; the CPU path is used.", "pending": ["all"]}

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict[str, Any]:
        lim = self.limits()
        st = self._jobs.stats() if hasattr(self._jobs, "stats") else {}
        from dataclasses import asdict  # noqa: PLC0415

        return report.build_report(self.profiler, self.monitor, cache_stats=self.cache_stats(), hardware=self.hardware_summary(), jobs=st, settings=self.settings().to_dict(),
                                   limits=asdict(lim), extra={"memory_growth_bytes": self.memory.growth_bytes(), "memory_trend_bytes_per_min": self.memory.trend_bytes_per_minute()})

    def export_diagnostics(self, path: Path, as_text: bool = False) -> Path:
        return report.export_report(self.diagnostics(), Path(path), as_text=as_text)

    def project(self) -> Project | None:
        return self._projects.current
