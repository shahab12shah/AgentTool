"""WaveformService: min/max peaks per time bucket, cached by content hash. Generation runs in a background job."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.audio.backend import ANALYSIS_SR, AudioBackend
from app.logging.logger import get_logger

_log = get_logger(__name__)
PPS = 100  # peak buckets per second
CLIP = 0.99


class Waveform:
    def __init__(self, pps: int, peaks: list[list[float]], clipped: list[int], duration: float) -> None:
        self.pps, self.peaks, self.clipped, self.duration = pps, peaks, set(clipped), duration

    def range(self, t0: float, t1: float, buckets: int) -> list[tuple[float, float, bool]]:
        """(min, max, clipped) for ``buckets`` equal slices of [t0, t1] (what a painter needs at a given zoom)."""
        out = []
        if buckets <= 0 or t1 <= t0:
            return out
        step = (t1 - t0) / buckets
        for k in range(buckets):
            a = max(0, int((t0 + k * step) * self.pps))
            b = min(len(self.peaks), max(a + 1, int((t0 + (k + 1) * step) * self.pps)))
            if a >= len(self.peaks):
                out.append((0.0, 0.0, False))
                continue
            seg = self.peaks[a:b]
            out.append((min(p[0] for p in seg), max(p[1] for p in seg), any(i in self.clipped for i in range(a, b))))
        return out

    def to_json(self) -> dict:
        return {"pps": self.pps, "duration": self.duration, "peaks": self.peaks, "clipped": sorted(self.clipped)}


class WaveformService:
    def __init__(self, backend: AudioBackend, cache_dir) -> None:
        self.backend, self._cache_dir = backend, cache_dir  # cache_dir: callable -> Path | None
        self._mem: dict[str, Waveform] = {}

    def _file(self, key: str) -> Path | None:
        d = self._cache_dir()
        return Path(d) / "waveforms" / f"{key}_{PPS}.json" if d else None

    def cached(self, key: str) -> Waveform | None:
        if key in self._mem:
            return self._mem[key]
        f = self._file(key)
        try:
            if f and f.is_file():
                d = json.loads(f.read_text(encoding="utf-8"))
                self._mem[key] = Waveform(d["pps"], d["peaks"], d.get("clipped", []), d["duration"])
                return self._mem[key]
        except (OSError, ValueError, KeyError):
            _log.warning("Ignoring unreadable waveform cache %s", f)
        return None

    def compute(self, path: Path, key: str, progress=None) -> Waveform:
        wf = self.cached(key)
        if wf is not None:
            return wf
        if progress:
            progress(10, "Reading audio")
        samples = self.backend.decode_mono(path)
        bucket = ANALYSIS_SR // PPS
        usable = (len(samples) // bucket) * bucket
        if usable == 0:
            wf = Waveform(PPS, [[0.0, 0.0]], [], len(samples) / ANALYSIS_SR)
        else:
            frames = samples[:usable].reshape(-1, bucket)
            mins, maxs = frames.min(axis=1), frames.max(axis=1)
            clipped = np.where(np.maximum(np.abs(mins), np.abs(maxs)) >= CLIP)[0].tolist()
            wf = Waveform(PPS, [[round(float(a), 4), round(float(b), 4)] for a, b in zip(mins, maxs)], clipped, len(samples) / ANALYSIS_SR)
        self._mem[key] = wf
        f = self._file(key)
        if f is not None:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(wf.to_json()), encoding="utf-8")
            except OSError:
                _log.debug("Could not cache waveform", exc_info=True)
        if progress:
            progress(100, "Done")
        return wf
