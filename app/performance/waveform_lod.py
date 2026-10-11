"""Multi-resolution waveform summaries (levels of detail).

The audio is read ONCE, in chunks, into the finest level (``LEVEL_PPS[0]`` min/max buckets per second); the coarser levels are reduced from it (a vectorised
min/max over groups of buckets), so zooming out never touches the fine data and zooming in never re-reads the audio. ``Waveform.range`` answers with the coarsest
level that is still at least as fine as the request, so a screen-wide query costs a few hundred array reads whatever the length of the audio.

Persistence is a JSON envelope holding the finest level as zlib-compressed int16 (min/max) plus a bit-packed clipping flag per bucket, keyed by content hash and the
source file's fingerprint. A cache file written by the old single-resolution code (``{"pps": 100, "peaks": [[min, max], ...]}``) still loads.
"""

from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

FORMAT_VERSION = 2
LEVEL_PPS = (400, 100, 25, 5)  # buckets per second, finest first (each is an integer fraction of the one before)
COMPAT_PPS = 100  # the resolution the old API (``Waveform.pps`` / ``.peaks``) exposes
CLIP = 0.99
_Q = 32767.0


@dataclass
class Level:
    pps: int
    mins: np.ndarray  # float32, len n (+1 padding element so reduceat may index n)
    maxs: np.ndarray
    clip: np.ndarray  # bool

    @property
    def n(self) -> int:
        return len(self.mins) - 1

    @property
    def nbytes(self) -> int:
        return int(self.mins.nbytes + self.maxs.nbytes + self.clip.nbytes)


def make_level(pps: int, mins: np.ndarray, maxs: np.ndarray, clip: np.ndarray) -> Level:
    pad = np.zeros(1, dtype=np.float32)
    return Level(pps, np.concatenate([np.asarray(mins, dtype=np.float32), pad]), np.concatenate([np.asarray(maxs, dtype=np.float32), pad]),
                 np.concatenate([np.asarray(clip, dtype=bool), np.zeros(1, dtype=bool)]))


def reduce_level(src: Level, pps: int) -> Level:
    """A coarser level from a finer one: min of mins / max of maxs / any clipped over groups of ``src.pps / pps`` buckets."""
    n = src.n
    if n == 0:
        return make_level(pps, np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool))
    f = max(1, int(round(src.pps / pps)))
    idx = np.arange(0, n, f)
    return make_level(pps, np.minimum.reduceat(src.mins[:n], idx), np.maximum.reduceat(src.maxs[:n], idx), np.logical_or.reduceat(src.clip[:n], idx))


class PeakBuilder:
    """Streams float32 mono samples (any chunking) into the finest level without ever holding the audio."""

    def __init__(self, sample_rate: int, pps: int = LEVEL_PPS[0]) -> None:
        self.sample_rate, self.pps = sample_rate, pps
        self.bucket = max(1, sample_rate // pps)
        self.total = 0
        self._carry = np.zeros(0, dtype=np.float32)
        self._mins: list[np.ndarray] = []
        self._maxs: list[np.ndarray] = []

    def feed(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        self.total += len(chunk)
        data = np.concatenate([self._carry, chunk]) if len(self._carry) else np.asarray(chunk, dtype=np.float32)
        usable = (len(data) // self.bucket) * self.bucket
        self._carry = data[usable:].copy()
        if usable:
            frames = data[:usable].reshape(-1, self.bucket)
            self._mins.append(frames.min(axis=1).astype(np.float32))
            self._maxs.append(frames.max(axis=1).astype(np.float32))

    @property
    def duration(self) -> float:
        return self.total / self.sample_rate

    def finish(self) -> "Waveform":
        if not self._mins:
            return Waveform.from_arrays(self.pps, np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool), self.duration)  # shorter than one bucket: silence
        mins, maxs = np.concatenate(self._mins), np.concatenate(self._maxs)
        clip = np.maximum(np.abs(mins), np.abs(maxs)) >= CLIP
        return Waveform.from_arrays(self.pps, mins, maxs, clip, self.duration)


class Waveform:
    """Min/max peaks per time bucket at several resolutions. The legacy constructor (``pps``, ``peaks``, ``clipped``, ``duration``) still works."""

    def __init__(self, pps: int, peaks: list[list[float]], clipped: Iterable[int], duration: float) -> None:
        arr = np.asarray(peaks, dtype=np.float32).reshape(-1, 2)
        clip = np.zeros(len(arr), dtype=bool)
        for i in clipped:
            if 0 <= i < len(clip):
                clip[i] = True
        self._init(pps, arr[:, 0] if len(arr) else np.zeros(0), arr[:, 1] if len(arr) else np.zeros(0), clip, duration)

    @classmethod
    def from_arrays(cls, pps: int, mins, maxs, clip, duration: float) -> "Waveform":
        self = cls.__new__(cls)
        self._init(pps, np.asarray(mins), np.asarray(maxs), np.asarray(clip), duration)
        return self

    def _init(self, pps: int, mins, maxs, clip, duration: float) -> None:
        self.duration = float(duration)
        finest = make_level(pps, mins, maxs, clip)
        levels = [finest]
        for p in LEVEL_PPS:
            if p < pps:
                levels.append(reduce_level(levels[-1], p))
        self.levels: list[Level] = levels  # finest first
        self._compat: tuple[list[list[float]], set[int]] | None = None

    # ------------------------------------------------------------------ compatibility surface
    @property
    def finest(self) -> Level:
        return self.levels[0]

    def level(self, pps: int) -> Level | None:
        return next((lv for lv in self.levels if lv.pps == pps), None)

    @property
    def pps(self) -> int:
        lv = self.level(COMPAT_PPS) or self.finest
        return lv.pps

    @property
    def peaks(self) -> list[list[float]]:
        """``[[min, max], ...]`` at ``pps`` (built on first use; the painter never needs it)."""
        if self._compat is None:
            lv = self.level(COMPAT_PPS) or self.finest
            self._compat = ([[round(float(a), 4), round(float(b), 4)] for a, b in zip(lv.mins[: lv.n], lv.maxs[: lv.n])], set(np.flatnonzero(lv.clip[: lv.n]).tolist()))
        return self._compat[0]

    @property
    def clipped(self) -> set[int]:
        self.peaks  # noqa: B018 - fills the compat cache
        return self._compat[1]  # type: ignore[index]

    @property
    def nbytes(self) -> int:
        return sum(lv.nbytes for lv in self.levels)

    # ------------------------------------------------------------------ queries
    def level_for(self, t0: float, t1: float, buckets: int) -> Level:
        """The coarsest level that has at least ``buckets`` buckets over [t0, t1] (the finest when none is fine enough)."""
        if buckets <= 0 or t1 <= t0:
            return self.finest
        need = buckets / (t1 - t0)
        best = self.finest
        for lv in self.levels:  # fine -> coarse
            if lv.pps >= need:
                best = lv
        return best

    def range(self, t0: float, t1: float, buckets: int) -> list[tuple[float, float, bool]]:
        """(min, max, clipped) for ``buckets`` equal slices of [t0, t1] (what a painter needs at a given zoom)."""
        if buckets <= 0 or t1 <= t0:
            return []
        lv = self.level_for(t0, t1, buckets)
        n = lv.n
        if n == 0:
            return [(0.0, 0.0, False)] * buckets
        step = (t1 - t0) / buckets
        pos = (t0 + np.arange(buckets + 1) * step) * lv.pps
        a = np.maximum(np.floor(pos[:-1] + 1e-6).astype(np.int64), 0)  # 1e-6: an edge that is exactly a bucket boundary must not fall into the bucket before / after it
        b0 = np.ceil(pos[1:] - 1e-6).astype(np.int64)  # the end edge rounds UP: a slice always covers every level bucket it touches (a safe envelope, never smaller than the truth)
        inside = a < n
        a = np.minimum(a, n - 1)
        b = np.minimum(np.maximum(b0, a + 1), n)
        pairs = np.empty(2 * buckets, dtype=np.int64)
        pairs[0::2], pairs[1::2] = a, b
        mn = np.minimum.reduceat(lv.mins, pairs)[0::2]
        mx = np.maximum.reduceat(lv.maxs, pairs)[0::2]
        cl = np.logical_or.reduceat(lv.clip, pairs)[0::2]
        mn, mx, cl = np.where(inside, mn, 0.0), np.where(inside, mx, 0.0), np.where(inside, cl, False)
        return list(zip(mn.tolist(), mx.tolist(), cl.tolist()))

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        lv = self.finest
        q = np.stack([lv.mins[: lv.n], lv.maxs[: lv.n]], axis=1)
        return {"version": FORMAT_VERSION, "duration": self.duration, "levels": [{
            "pps": lv.pps, "n": lv.n, "peaks": _pack(np.round(q * _Q).astype("<i2").tobytes()), "clip": _pack(np.packbits(lv.clip[: lv.n]).tobytes())}]}

    @classmethod
    def from_json(cls, d: dict) -> "Waveform":
        if int(d.get("version", 1)) < 2:  # the old single-resolution format
            return cls(int(d["pps"]), d["peaks"], d.get("clipped", []), float(d["duration"]))
        raw = d["levels"][0]
        n = int(raw["n"])
        q = np.frombuffer(_unpack(raw["peaks"]), dtype="<i2").reshape(-1, 2).astype(np.float32) / _Q
        clip = np.unpackbits(np.frombuffer(_unpack(raw["clip"]), dtype=np.uint8))[:n].astype(bool)
        if len(q) != n or len(clip) != n:
            raise ValueError("inconsistent waveform file")
        return cls.from_arrays(int(raw["pps"]), q[:, 0], q[:, 1], clip, float(d["duration"]))


def _pack(b: bytes) -> str:
    return base64.b64encode(zlib.compress(b, 6)).decode("ascii")


def _unpack(s: str) -> bytes:
    return zlib.decompress(base64.b64decode(s))


def build_from_chunks(chunks: Iterable[np.ndarray], sample_rate: int, progress: Callable[[float], None] | None = None, total_seconds: float | None = None,
                      should_cancel: Callable[[], bool] | None = None) -> Waveform | None:
    """One streaming pass over ``chunks`` (None when cancelled)."""
    pb = PeakBuilder(sample_rate)
    for ch in chunks:
        if should_cancel is not None and should_cancel():
            return None
        pb.feed(ch)
        if progress is not None and total_seconds:
            progress(min(0.99, pb.duration / total_seconds))
    return pb.finish()
