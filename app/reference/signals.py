"""Frame sampling and per-frame signals: the single pass over the reference video that every visual detector builds on.

``FrameSampler`` streams small frames from FFmpeg (never the whole video in memory). ``compute_signals`` reduces them to compact arrays
(``FrameSignals``): brightness, frame-to-frame difference, histogram change, edge density, a tiny signature of every frame, and the global
motion between frames half a second apart. Detectors (shots, motion, structure) read these arrays; no frame is kept.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from app.core.exceptions import AppError
from app.rendering.ffmpeg_service import FFmpegService
from app.reference.motion_estimator import estimate_global_motion

_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
SIG_W, SIG_H = 16, 9  # per-frame signature size
SIGNAL_W, SIGNAL_H = 128, 72  # the size frames are analysed at


class ReferenceAnalysisError(AppError):
    """The reference video could not be analysed (or a stage of it failed)."""


class AnalysisCancelled(Exception):
    """The user cancelled the analysis."""


@dataclass
class SampledFrame:
    index: int
    time: float
    pixels: np.ndarray  # (H, W) uint8 gray, or (H, W, 3) uint8 RGB


class FrameSampler:
    """Streams frames from FFmpeg at a fixed sample rate and size. ``gray=True`` gives 1 channel (cheaper)."""

    def __init__(self, ffmpeg: FFmpegService) -> None:
        self.ff = ffmpeg

    def frames(self, path: Path, fps: float, width: int, height: int, *, gray: bool = True, start: float = 0.0, duration: float | None = None,
               cancel: threading.Event | None = None, stderr_tail: int = 20) -> Iterator[SampledFrame]:
        fmt = "gray" if gray else "rgb24"
        ch = 1 if gray else 3
        vf = f"fps={fps:g},scale={width}:{height}:flags=area"
        args = [self.ff.ffmpeg(), "-v", "error", "-nostdin"]
        if start > 0:
            args += ["-ss", f"{start:.3f}"]
        args += ["-i", str(path)]
        if duration:
            args += ["-t", f"{duration:.3f}"]
        args += ["-an", "-sn", "-vf", vf, "-pix_fmt", fmt, "-f", "rawvideo", "-"]
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_NO_WINDOW)
        except OSError as exc:
            raise ReferenceAnalysisError("FFmpeg could not be started.", details=str(exc)) from exc
        tail: deque[str] = deque(maxlen=stderr_tail)

        def drain() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                tail.append(line.decode("utf-8", "replace").rstrip())

        t = threading.Thread(target=drain, daemon=True)
        t.start()
        size = width * height * ch
        i = 0
        try:
            assert proc.stdout is not None
            while True:
                if cancel is not None and cancel.is_set():
                    raise AnalysisCancelled()
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8)
                yield SampledFrame(i, start + i / fps, arr.reshape(height, width) if gray else arr.reshape(height, width, 3))
                i += 1
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            t.join(timeout=2)
        if i == 0:
            raise ReferenceAnalysisError("No video frames could be decoded from the reference.", details="\n".join(tail))


@dataclass
class FrameSignals:
    fps: float
    times: np.ndarray  # (N,) seconds
    luma: np.ndarray  # (N,) mean brightness 0..1
    luma_std: np.ndarray  # (N,) contrast 0..1
    diff: np.ndarray  # (N,) mean absolute difference to the previous sample, 0..1 (diff[0] = 0)
    hist_diff: np.ndarray  # (N,) half L1 distance between 32-bin luma histograms of consecutive samples, 0..1
    edge_density: np.ndarray  # (N,) share of pixels with a strong gradient
    sig: np.ndarray  # (N, SIG_H, SIG_W) float32 tiny signature of each frame (0..1)
    lag: int  # motion is measured between frame i and frame i - lag
    dx: np.ndarray  # (N,) horizontal content shift per SECOND, as a fraction of the frame width
    dy: np.ndarray  # (N,) vertical shift per second, fraction of the height
    log_scale: np.ndarray  # (N,) natural log of the zoom factor per SECOND (positive = zooming in)
    motion_conf: np.ndarray  # (N,) 0..1
    duration: float = 0.0
    extras: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.times)


def _hist(g: np.ndarray, bins: int = 32) -> np.ndarray:
    h, _ = np.histogram(g, bins=bins, range=(0.0, 1.0))
    return h / max(1, g.size)


def _signature(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    return g[: h - h % SIG_H, : w - w % SIG_W].reshape(SIG_H, h // SIG_H, SIG_W, w // SIG_W).mean(axis=(1, 3)).astype(np.float32)


def compute_signals(frames: Iterable[SampledFrame], fps: float, *, lag_seconds: float = 0.5, progress: Callable[[float], None] | None = None, expected_frames: int | None = None,
                    cancel: threading.Event | None = None) -> FrameSignals:
    """One pass over the sampled gray frames (uint8, SIGNAL_W x SIGNAL_H). Memory is O(frames x signature), never the frames themselves."""
    lag = max(1, int(round(lag_seconds * fps)))
    ring: deque[np.ndarray] = deque(maxlen=lag + 1)
    prev: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    times, luma, luma_std, diff, hdiff, edge, sigs, dxs, dys, lss, confs = ([] for _ in range(11))
    n = 0
    for fr in frames:
        if cancel is not None and cancel.is_set():
            raise AnalysisCancelled()
        g = fr.pixels.astype(np.float32) / 255.0
        hh = _hist(g)
        times.append(fr.time)
        luma.append(float(g.mean()))
        luma_std.append(float(g.std()))
        gx, gy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
        edge.append(float(((gx[:-1, :] + gy[:, :-1]) > 0.18).mean()))
        sigs.append(_signature(g))
        if prev is None:
            diff.append(0.0)
            hdiff.append(0.0)
        else:
            diff.append(float(np.abs(g - prev).mean()))
            hdiff.append(float(0.5 * np.abs(hh - prev_hist).sum()))
        if len(ring) >= lag:
            ref = ring[-lag]  # the frame ``lag`` samples ago
            dx, dy, ls, conf = estimate_global_motion(ref, g)
            sec = lag / fps
            dxs.append(dx / sec)
            dys.append(dy / sec)
            lss.append(ls / sec)
            confs.append(conf)
        else:
            dxs.append(0.0), dys.append(0.0), lss.append(0.0), confs.append(0.0)
        ring.append(g)
        prev, prev_hist = g, hh
        n += 1
        if progress is not None and expected_frames and n % 50 == 0:
            progress(min(1.0, n / expected_frames))
    if n == 0:
        raise ReferenceAnalysisError("No video frames could be decoded from the reference.")
    arr = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    return FrameSignals(fps, np.asarray(times, dtype=np.float64), arr(luma), arr(luma_std), arr(diff), arr(hdiff), arr(edge), np.stack(sigs), lag, arr(dxs), arr(dys), arr(lss), arr(confs),
                        float(times[-1] + 1.0 / fps))


def signals_from_gray_frames(frames: list[np.ndarray], fps: float, **kw) -> FrameSignals:
    """Build signals from in-memory frames (uint8 or float 0..1 of SIGNAL_H x SIGNAL_W). For tests and synthetic data."""
    def to_u8(a: np.ndarray) -> np.ndarray:
        return a if a.dtype == np.uint8 else np.clip(a * 255.0, 0, 255).astype(np.uint8)

    return compute_signals((SampledFrame(i, i / fps, to_u8(f)) for i, f in enumerate(frames)), fps, **kw)
