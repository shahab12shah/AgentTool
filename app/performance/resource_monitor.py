"""Resource sampling with the standard library only (no psutil): process memory, system memory, CPU load, disk space and thread count.

Every probe is wrapped: an unavailable measurement is ``None`` (never an exception, never a guess). ``pressure()`` turns a sample into a level the
scheduler uses to throttle background work; unknown measurements count as NORMAL so a platform without a probe never stalls the app.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

NORMAL, ELEVATED, CRITICAL = "normal", "elevated", "critical"
_LEVEL = {NORMAL: 0, ELEVATED: 1, CRITICAL: 2}


@dataclass(frozen=True)
class ResourceSample:
    taken_at: float
    process_rss_bytes: int | None
    system_total_bytes: int | None
    system_available_bytes: int | None
    cpu_count: int
    load_1m: float | None  # normalised by CPU count (1.0 = every core busy), None where unavailable
    process_cpu_percent: float | None  # of ONE core, since the previous sample
    disk_free_bytes: int | None
    disk_total_bytes: int | None
    threads: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def memory_used_fraction(self) -> float | None:
        if self.system_total_bytes and self.system_available_bytes is not None:
            return 1.0 - self.system_available_bytes / self.system_total_bytes
        return None


def process_rss_bytes() -> int | None:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", "rb") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):  # type: ignore[attr-defined]
                return int(pmc.WorkingSetSize)
            return None
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS: bytes (this is the PEAK, the best portable value there)
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024
    except Exception:  # noqa: BLE001 - a missing probe is "unknown", not an error
        return None


def system_memory() -> tuple[int | None, int | None]:
    """(total, available) bytes of physical memory."""
    try:
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo", "rb") as f:
                for line in f:
                    k, _, v = line.decode("ascii", "ignore").partition(":")
                    info[k] = int(v.split()[0]) * 1024
            return info.get("MemTotal"), info.get("MemAvailable", info.get("MemFree"))
        if sys.platform.startswith("win"):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
                return int(ms.ullTotalPhys), int(ms.ullAvailPhys)
            return None, None
        if sys.platform == "darwin":
            total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
            return total, None  # available memory needs vm_stat parsing; unknown is honest
    except Exception:  # noqa: BLE001
        pass
    return None, None


def disk_usage(path: Path | str) -> tuple[int | None, int | None]:
    """(free, total) bytes of the volume holding ``path`` (nearest existing parent)."""
    p = Path(path)
    for cand in (p, *p.parents):
        try:
            if cand.exists():
                u = shutil.disk_usage(cand)
                return int(u.free), int(u.total)
        except OSError:
            continue
    return None, None


class ResourceMonitor:
    """Takes samples on demand (cheap) or on a background timer (``start``); keeps a short history for growth/trend questions."""

    def __init__(self, watch_path: Callable[[], Path | None] | Path | None = None, history: int = 120) -> None:
        self._watch = watch_path
        self._hist: list[ResourceSample] = []
        self._max = history
        self._lock = threading.Lock()
        self._last_cpu: tuple[float, float] | None = None  # (wall, process cpu seconds)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.thresholds = {"mem_elevated": 0.80, "mem_critical": 0.92, "disk_elevated_bytes": 5 * 1024**3, "disk_critical_bytes": 1024**3, "load_elevated": 1.5}

    def set_watch_path(self, watch: Callable[[], Path | None] | Path | None) -> None:
        self._watch = watch

    def _path(self) -> Path:
        w = self._watch() if callable(self._watch) else self._watch
        return Path(w) if w else Path.home()

    def sample(self) -> ResourceSample:
        now = time.monotonic()
        cpu_t = time.process_time()
        pct = None
        with self._lock:
            if self._last_cpu is not None and now > self._last_cpu[0]:
                pct = max(0.0, 100.0 * (cpu_t - self._last_cpu[1]) / (now - self._last_cpu[0]))
            self._last_cpu = (now, cpu_t)
        total, avail = system_memory()
        n = os.cpu_count() or 1
        try:
            load = os.getloadavg()[0] / n
        except (AttributeError, OSError):
            load = None
        free, dtotal = disk_usage(self._path())
        s = ResourceSample(time.time(), process_rss_bytes(), total, avail, n, load, pct, free, dtotal, threading.active_count())
        with self._lock:
            self._hist.append(s)
            del self._hist[: max(0, len(self._hist) - self._max)]
        return s

    def history(self) -> list[ResourceSample]:
        with self._lock:
            return list(self._hist)

    def latest(self) -> ResourceSample | None:
        with self._lock:
            return self._hist[-1] if self._hist else None

    def pressure(self, sample: ResourceSample | None = None) -> dict[str, str]:
        """Per-resource level plus an ``overall`` level (the worst). Unknown measurements are NORMAL."""
        s = sample or self.sample()
        t = self.thresholds
        mem = NORMAL
        f = s.memory_used_fraction
        if f is not None:
            mem = CRITICAL if f >= t["mem_critical"] else ELEVATED if f >= t["mem_elevated"] else NORMAL
        disk = NORMAL
        if s.disk_free_bytes is not None:
            disk = CRITICAL if s.disk_free_bytes < t["disk_critical_bytes"] else ELEVATED if s.disk_free_bytes < t["disk_elevated_bytes"] else NORMAL
        cpu = ELEVATED if (s.load_1m is not None and s.load_1m >= t["load_elevated"]) else NORMAL
        overall = max((mem, disk, cpu), key=_LEVEL.__getitem__)
        return {"memory": mem, "disk": disk, "cpu": cpu, "overall": overall}

    # ---- optional background sampling
    def start(self, interval: float = 5.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.sample()
                except Exception:  # noqa: BLE001 - monitoring must never crash the app
                    pass

        self._thread = threading.Thread(target=loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
