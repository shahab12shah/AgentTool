"""Performance settings: one small, validated contract shared by the UI, the scheduler, the cache manager and the render/preview code.

``PerformanceSettings`` is stored globally (``Settings.performance`` in settings.json) and may be overridden per project
(``Project.performance_overrides``: only the keys the user changed). ``resolve`` turns the profile + overrides + the detected machine into
``ResolvedLimits`` — the concrete numbers the rest of the code uses. A profile never changes final export quality (see ``resolve``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

PROFILES = ("power_saver", "balanced", "performance")
PREVIEW_QUALITIES = ("draft", "balanced", "high")
PROXY_POLICIES = ("off", "manual", "automatic")
PROXY_PROFILES = ("performance", "balanced", "quality")
RENDER_BACKENDS = ("auto", "cpu", "hardware")  # same vocabulary as RenderSettings.hardware_acceleration
CACHE_CATEGORIES = ("thumbnails", "preview_frames", "proxies", "waveforms", "analysis", "render_previews", "temporary")
MB = 1024 * 1024

# proxy profile -> the existing Phase 6 proxy height key (PROXY_RESOLUTIONS); configurable, not assumed to suit every source
PROXY_PROFILE_RESOLUTION = {"performance": "540p", "balanced": "720p", "quality": "1080p"}
DEFAULT_CATEGORY_LIMITS_MB = {"thumbnails": 512, "preview_frames": 512, "proxies": 20_000, "waveforms": 256, "analysis": 512, "render_previews": 8_000, "temporary": 2_000}


@dataclass
class PerformanceSettings:
    profile: str = "balanced"
    preview_quality: str = "balanced"
    proxy_policy: str = "manual"
    proxy_profile: str = "balanced"
    render_backend: str = "auto"
    cache_limit_mb: int = 0  # 0 = derive from the free disk space; otherwise a hard total ceiling for rebuildable caches
    category_limits_mb: dict[str, int] = field(default_factory=dict)  # per category; missing = default
    max_background_workers: int = 0  # 0 = derive from the profile and the CPU count
    background_proxy: bool = True  # proxy generation may run in the background (never when resources are constrained)
    idle_cache_generation: bool = True  # thumbnails / waveforms / frames may be prepared when the app is idle
    metrics_enabled: bool = True  # cheap aggregate timers; no per-call logging
    slow_operation_ms: int = 1500  # an operation slower than this is logged once as perf.slow_operation
    memory_limit_mb: int = 0  # 0 = derive from RAM; soft ceiling for in-memory caches (thumbnails/frames)

    def sanitized(self) -> "PerformanceSettings":
        s = PerformanceSettings(**{f.name: getattr(self, f.name) for f in fields(self)})
        for name, allowed, default in (("profile", PROFILES, "balanced"), ("preview_quality", PREVIEW_QUALITIES, "balanced"), ("proxy_policy", PROXY_POLICIES, "manual"),
                                       ("proxy_profile", PROXY_PROFILES, "balanced"), ("render_backend", RENDER_BACKENDS, "auto")):
            if getattr(s, name) not in allowed:
                setattr(s, name, default)
        for name in ("cache_limit_mb", "max_background_workers", "memory_limit_mb", "slow_operation_ms"):
            v = getattr(s, name)
            setattr(s, name, int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else getattr(PerformanceSettings(), name))
        for name in ("background_proxy", "idle_cache_generation", "metrics_enabled"):
            setattr(s, name, bool(getattr(s, name)))
        if not isinstance(s.category_limits_mb, dict):
            s.category_limits_mb = {}
        s.category_limits_mb = {k: int(v) for k, v in s.category_limits_mb.items() if k in CACHE_CATEGORIES and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
        return s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.sanitized())

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PerformanceSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known}).sanitized()

    def with_overrides(self, overrides: dict[str, Any] | None) -> "PerformanceSettings":
        """Global settings with a project's override keys applied (unknown keys are ignored)."""
        merged = self.to_dict()
        merged.update({k: v for k, v in (overrides or {}).items() if k in merged})
        return PerformanceSettings.from_dict(merged)


@dataclass(frozen=True)
class ResolvedLimits:
    """The concrete numbers derived from the settings and the machine. Never exceeds what the machine has."""

    profile: str
    background_workers: int  # total worker threads for low/medium priority work
    foreground_workers: int  # reserved for high-priority work (previews, visible timeline)
    thumbnail_concurrency: int
    proxy_concurrency: int
    prefetch_frames: int  # frames kept ahead/behind the playhead
    memory_cache_bytes: int  # soft ceiling for in-memory rebuildable caches (decoded thumbnails / frames)
    cache_total_bytes: int  # ceiling for all on-disk rebuildable caches
    category_bytes: dict[str, int] = field(default_factory=dict)
    idle_work: bool = True
    background_proxy: bool = True
    preview_quality: str = "balanced"
    note: str = "Profiles change editing responsiveness and background work only; final export quality is never reduced."


def resolve(settings: PerformanceSettings, *, cpu_count: int, ram_bytes: int | None, disk_free_bytes: int | None) -> ResolvedLimits:
    """Concrete limits for this machine. ``PERFORMANCE`` never means "every core / all RAM": caps are fractions of what exists."""
    s = settings.sanitized()
    cpus = max(1, int(cpu_count or 1))
    ram = ram_bytes if ram_bytes and ram_bytes > 0 else 8 * 1024 * MB  # unknown RAM: assume a modest machine
    base = {"power_saver": max(1, cpus // 4), "balanced": max(1, cpus // 2), "performance": max(1, (cpus * 3) // 4)}[s.profile]
    workers = min(s.max_background_workers, cpus) if s.max_background_workers > 0 else base
    workers = max(1, workers)
    fg = 1 if cpus < 4 else 2
    mem_frac = {"power_saver": 0.03, "balanced": 0.06, "performance": 0.10}[s.profile]
    mem = s.memory_limit_mb * MB if s.memory_limit_mb > 0 else int(ram * mem_frac)
    mem = max(64 * MB, min(mem, int(ram * 0.25)))
    free = disk_free_bytes if disk_free_bytes and disk_free_bytes > 0 else 50 * 1024 * MB
    total = s.cache_limit_mb * MB if s.cache_limit_mb > 0 else int(free * 0.25)
    total = min(total, int(free * 0.9)) if disk_free_bytes else total
    cats = {c: int(min(s.category_limits_mb.get(c, DEFAULT_CATEGORY_LIMITS_MB[c]) * MB, total)) for c in CACHE_CATEGORIES}
    prefetch = {"power_saver": 4, "balanced": 12, "performance": 30}[s.profile]
    thumbs = {"power_saver": 1, "balanced": 2, "performance": 3}[s.profile]
    return ResolvedLimits(s.profile, workers, fg, min(thumbs, workers), 1 if s.profile != "performance" else min(2, workers), prefetch, mem, total, cats,
                          s.idle_cache_generation and s.profile != "power_saver", s.background_proxy and s.profile != "power_saver", s.preview_quality)
