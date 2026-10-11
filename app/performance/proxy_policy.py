"""Proxy policy: which assets deserve a proxy, why, how much disk that needs, and whether it is safe to start.

The policy (``off`` / ``manual`` / ``automatic``) and the profile (``performance`` / ``balanced`` / ``quality`` -> a proxy height) come from the performance settings and may be
overridden per project. ``ProxyPolicy.recommend`` is pure (no I/O besides what the caller passes in), so the decisions are testable and can be shown to the user before anything runs.
Proxies are only a stand-in while editing: a proxy never changes the export, which reads the original media unless the user explicitly chose otherwise.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from app.media.asset import Asset, AssetType
from app.performance.settings import PROXY_PROFILE_RESOLUTION, PerformanceSettings
from app.rendering.presets import PROXY_RESOLUTIONS

# generous average video bitrates of the x264 veryfast / crf 23 proxies, by proxy height (kilobits per second)
BITRATE_KBPS = {"540p": 1800, "720p": 3200, "1080p": 6500}
SAFETY_MARGIN = 1.3  # estimate * margin is what must fit
RESERVE_BYTES = 2 * 1024 ** 3  # never plan to fill the disk below this much free space
WARN_FRACTION = 0.5  # warn when the batch would use more than this share of the free space
UNKNOWN_DURATION_S = 120.0
EXPENSIVE_CODECS = frozenset({"hevc", "h265", "prores", "av1", "vp9", "dnxhd", "dnxhr", "cfhd", "mjpeg", "mpeg2video", "vc1"})
LAYERS_BUSY = 3  # this many visual layers on screen at once make the timeline "busy"
UHD_MIN_SIDE = 1900  # shorter side at/above this counts as 4K-class (UHD 2160, DCI 4K 2160, 5K...) rather than FullHD


@dataclass
class ProxyDecision:
    asset_id: str
    name: str
    proxy: bool
    reasons: list[str] = field(default_factory=list)
    resolution: str = "720p"
    estimated_bytes: int = 0
    skipped: str = ""  # why a proxy is NOT recommended (small source, alpha, ...)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class DiskCheck:
    ok: bool
    needed_bytes: int
    free_bytes: int | None
    message: str = ""
    warning: str = ""
    over_cache_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ProxyRecommendation:
    policy: str
    profile: str
    resolution: str
    decisions: list[ProxyDecision]
    estimated_bytes: int
    disk: DiskCheck | None = None
    note: str = ""

    @property
    def asset_ids(self) -> list[str]:
        return [d.asset_id for d in self.decisions if d.proxy]

    def why(self, asset_id: str) -> list[str]:
        d = next((x for x in self.decisions if x.asset_id == asset_id), None)
        return list(d.reasons) if d else []

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "profile": self.profile, "resolution": self.resolution, "estimated_bytes": self.estimated_bytes, "note": self.note,
                "disk": self.disk.to_dict() if self.disk else None, "decisions": [d.to_dict() for d in self.decisions]}


def resolution_for(profile: str) -> str:
    return PROXY_PROFILE_RESOLUTION.get(profile, "720p")


def estimate_bytes(duration_s: float | None, resolution: str) -> int:
    d = duration_s if duration_s and duration_s > 0 else UNKNOWN_DURATION_S
    return int(d * BITRATE_KBPS.get(resolution, BITRATE_KBPS["720p"]) * 1000 / 8)


def peak_layers(project: Any) -> int:
    """The most visual (video / image) clips on screen at the same time anywhere on the timeline: a sweep over clip start/end events."""
    events: list[tuple[float, int]] = []
    try:
        for track in project.timeline.tracks:
            if getattr(track.kind, "value", str(track.kind)).lower() != "video":
                continue
            for c in track.clips:
                if c.asset_id and c.duration > 0:
                    events.append((c.timeline_start, 1))
                    events.append((c.timeline_start + c.duration, -1))
    except Exception:  # noqa: BLE001 - complexity is only a hint
        return 0
    events.sort(key=lambda e: (e[0], e[1]))  # an end before a start at the same instant
    cur = peak = 0
    for _t, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


class ProxyPolicy:
    """Decides; never runs anything."""

    def __init__(self, disk_free: Callable[[Path | str], int | None] | None = None) -> None:
        self._disk_free = disk_free or self._real_free

    @staticmethod
    def _real_free(path: Path | str) -> int | None:
        p = Path(path)
        for cand in (p, *p.parents):
            try:
                if cand.exists():
                    return int(shutil.disk_usage(cand).free)
            except OSError:
                continue
        return None

    # ------------------------------------------------------------------ decisions
    def resolution(self, settings: PerformanceSettings, project: Any = None, manual_resolution: str | None = None) -> str:
        """``manual_resolution`` (the project's own proxy size) applies to manual proxies; the profile decides otherwise."""
        if settings.proxy_policy == "manual" and manual_resolution in PROXY_RESOLUTIONS:
            return manual_resolution  # type: ignore[return-value]
        return resolution_for(settings.proxy_profile)

    def recommend(self, project: Any, assets: Iterable[Asset] | None, settings: PerformanceSettings, hardware_info: dict | None = None, *, resolution: str | None = None) -> ProxyRecommendation:
        s = settings.sanitized()
        res = resolution if resolution in PROXY_RESOLUTIONS else self.resolution(s, project, getattr(getattr(project, "render_settings", None), "proxy_resolution", None))
        height = PROXY_RESOLUTIONS[res]
        pool = list(assets if assets is not None else project.assets.all())
        layers = peak_layers(project) if project is not None else 0
        busy = layers >= LAYERS_BUSY
        hw = hardware_info or {}
        cores = ((hw.get("cpu") or {}).get("logical_cores")) or 0
        decode_ok = bool(any(((hw.get("decoders") or {}).get("working") or {}).values()))
        decisions: list[ProxyDecision] = []
        total = 0
        for a in pool:
            if a.type is not AssetType.VIDEO:
                continue
            d = ProxyDecision(a.id, a.name, False, [], res)
            decisions.append(d)
            probe = (a.extra or {}).get("probe", {}) if a.extra else {}
            if probe.get("has_alpha"):
                d.skipped = "has transparency: a proxy would lose it"
                continue
            side = min(a.width or 0, a.height or 0)
            if side <= height:
                d.skipped = f"already {side}p or smaller: a {res} proxy would not be lighter" if side else "size unknown"
                continue
            codec = (a.codec or "").lower()
            if side >= UHD_MIN_SIDE:
                d.reasons.append(f"4K-class source ({a.width}x{a.height}) above the {res} proxy size")
            if codec in EXPENSIVE_CODECS:
                d.reasons.append(f"{codec.upper()} is expensive to decode" + ("" if decode_ok else " (no hardware decoding detected)"))
            if busy:
                d.reasons.append(f"busy timeline ({layers} visual layers play at once)")
            if s.proxy_profile == "performance":
                d.reasons.append(f"performance profile proxies every source above {res}")
            if cores and cores <= 2 and not decode_ok and not d.reasons:
                d.reasons.append("few CPU cores and no hardware decoding")
            if not d.reasons:
                d.skipped = f"{a.width}x{a.height} plays smoothly on this setup"
                continue
            d.proxy = True
            d.estimated_bytes = estimate_bytes(a.duration, res)
            total += d.estimated_bytes
        note = {"off": "Proxies are switched off.", "manual": "Proxies are only created when you ask for them.", "automatic": "Proxies are created in the background when the machine is idle."}[s.proxy_policy]
        return ProxyRecommendation(s.proxy_policy, s.proxy_profile, res, decisions, total, None, note)

    # ------------------------------------------------------------------ disk
    def check_disk(self, estimated_bytes: int, folder: Path | str, *, cache_limit_bytes: int | None = None, existing_proxy_bytes: int = 0) -> DiskCheck:
        """Will a batch of ``estimated_bytes`` fit? ``ok`` is False when it would (nearly) fill the disk; ``over_cache_limit`` when it exceeds the proxies cache limit."""
        need = int(estimated_bytes * SAFETY_MARGIN)
        free = self._disk_free(folder)
        over = cache_limit_bytes is not None and cache_limit_bytes > 0 and existing_proxy_bytes + need > cache_limit_bytes
        if free is None:
            return DiskCheck(True, need, None, "", "Free disk space could not be determined; the proxy size is only an estimate.", over)
        if need + RESERVE_BYTES > free:
            return DiskCheck(False, need, free, f"Not enough free disk space for the proxies: about {_gb(need)} are needed (estimate) and {_gb(free)} are free, and {_gb(RESERVE_BYTES)} are kept in reserve.", "", over)
        warn = f"These proxies need about {_gb(need)} of the {_gb(free)} free." if need > free * WARN_FRACTION else ""
        return DiskCheck(True, need, free, "", warn, over)


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 100 * 1024 ** 2 else f"{n / 1024 ** 2:.0f} MB"
