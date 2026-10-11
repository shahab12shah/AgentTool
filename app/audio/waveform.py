"""WaveformService: min/max peaks per time bucket at several levels of detail, cached by content hash. Generation runs in a background job.

The audio is decoded as a stream in one pass (``app.performance.waveform_lod``), the result is stored as a compact file under ``<cache>/waveforms`` and registered with
the cache manager (category ``waveforms``) when there is one. The file records the source file's fingerprint: a replaced source rebuilds the waveform.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

from app.audio.backend import ANALYSIS_SR, AudioBackend
from app.core.exceptions import JobCancelled
from app.logging.logger import get_logger
from app.performance.dependencies import file_fingerprint, stable_key
from app.performance.memory_monitor import BoundedLRU
from app.performance.profiler import profiler
from app.performance.waveform_lod import CLIP, COMPAT_PPS, LEVEL_PPS, FORMAT_VERSION, PeakBuilder, Waveform

__all__ = ["Waveform", "WaveformService", "PPS", "CLIP"]
_log = get_logger(__name__)
PPS = COMPAT_PPS  # the cache file keeps its historical name (``<key>_100.json``)
REVALIDATE_EVERY = 3.0  # seconds between two fingerprint checks of the same source
MEM_BYTES = 96 * 1024 * 1024


class WaveformService:
    def __init__(self, backend: AudioBackend, cache_dir, cache_getter: Callable[[], object | None] | None = None) -> None:
        self.backend, self._cache_dir = backend, cache_dir  # cache_dir: callable -> Path | None
        self.cache_getter = cache_getter  # MediaCacheManager | None (the performance service's cache for the open project)
        self._mem: BoundedLRU[str, Waveform] = BoundedLRU(64, MEM_BYTES, lambda w: w.nbytes)
        self._fp: dict[str, tuple[str, float]] = {}  # key -> (source fingerprint recorded in the file, last time it was compared)
        self._lock = threading.Lock()

    def _file(self, key: str) -> Path | None:
        d = self._cache_dir()
        return Path(d) / "waveforms" / f"{key}_{PPS}.json" if d else None

    def _cache(self):
        try:
            return self.cache_getter() if self.cache_getter else None
        except Exception:  # noqa: BLE001
            return None

    def forget(self, key: str) -> None:
        self._mem.pop(key)
        self._fp.pop(key, None)

    def _stale(self, key: str, path: Path | None) -> bool:
        """The source file changed since this waveform was made (checked at most every few seconds; unknown fingerprint = trusted)."""
        if path is None:
            return False
        rec = self._fp.get(key)
        if rec is None or not rec[0]:
            return False
        now = time.monotonic()
        if now - rec[1] < REVALIDATE_EVERY:
            return False
        fp = file_fingerprint(path)
        self._fp[key] = (rec[0], now)
        return fp not in (rec[0], "missing")

    def cached(self, key: str, path: Path | None = None) -> Waveform | None:
        """The stored waveform, or None. With ``path`` the source's fingerprint is checked too: a replaced file invalidates the waveform."""
        wf = self._mem.get(key)
        if wf is not None:
            if not self._stale(key, path):
                profiler.cache_hit("waveforms")
                return wf
            self.invalidate(key)
            return None
        f = self._file(key)
        try:
            if f and f.is_file():
                d = json.loads(f.read_text(encoding="utf-8"))
                wf = Waveform.from_json(d)
                self._fp[key] = (str(d.get("fp", "")), 0.0 if d.get("fp") else time.monotonic())
                if self._stale(key, path):
                    self.invalidate(key)
                    profiler.cache_miss("waveforms")
                    return None
                self._mem.put(key, wf)
                profiler.cache_hit("waveforms")
                return wf
        except (OSError, ValueError, KeyError, TypeError):
            _log.warning("Ignoring unreadable waveform cache %s", f)
        profiler.cache_miss("waveforms")
        return None

    def invalidate(self, key: str) -> None:
        self.forget(key)
        f = self._file(key)
        cache = self._cache()
        if cache is not None:
            cache.invalidate(stable_key("waveform", key))
        if f is not None:
            f.unlink(missing_ok=True)

    def compute(self, path: Path, key: str, progress=None, should_cancel: Callable[[], bool] | None = None, dep_id: str | None = None,
                duration_hint: float | None = None) -> Waveform:
        wf = self.cached(key, path)
        if wf is not None:
            return wf
        if progress:
            progress(5, "Reading audio")
        fp = file_fingerprint(path)  # taken before reading: a file that changes meanwhile is stale next time
        pb = PeakBuilder(ANALYSIS_SR, LEVEL_PPS[0])
        with profiler.timer("waveform.build"):
            for chunk in self.backend.stream_mono(path, ANALYSIS_SR, 10.0, should_cancel):
                if should_cancel is not None and should_cancel():
                    raise JobCancelled()
                pb.feed(chunk)
                if progress:
                    progress(min(95.0, 5.0 + 90.0 * pb.duration / duration_hint) if duration_hint else 50.0, "Reading audio")
        if should_cancel is not None and should_cancel():
            raise JobCancelled()
        wf = pb.finish()
        self._mem.put(key, wf)
        self._fp[key] = (fp, time.monotonic())
        self._store(key, wf, fp, dep_id)
        if progress:
            progress(100, "Done")
        return wf

    def _store(self, key: str, wf: Waveform, fp: str, dep_id: str | None) -> None:
        f = self._file(key)
        if f is None:
            return
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f".{f.stem}.part{f.suffix}")
            tmp.write_text(json.dumps({**wf.to_json(), "fp": fp, "format": FORMAT_VERSION}), encoding="utf-8")
            os.replace(tmp, f)
        except OSError:
            _log.debug("Could not cache waveform", exc_info=True)
            return
        cache = self._cache()
        if cache is not None:
            cache.put(stable_key("waveform", key), category="waveforms", path=f, deps={dep_id or f"waveform:{key}": fp}, version=FORMAT_VERSION)
