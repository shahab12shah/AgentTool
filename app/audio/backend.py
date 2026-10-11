"""Audio backend interface (provider-independent) and its FFmpeg implementation.

Nothing above this module runs audio tools; the engine talks to an ``AudioBackend`` so another backend (a library, a service) can replace
FFmpeg without touching the analysis, processing, mixing or ducking logic. Source files are only ever read.
"""

from __future__ import annotations

import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from app.core.exceptions import AppError, FFmpegUnavailableError
from app.logging.logger import get_logger
from app.media.media_probe import locate_binary, run_process

_log = get_logger(__name__)
ANALYSIS_SR = 16000


class AudioError(AppError):
    """Audio could not be read, analysed or processed."""


@dataclass
class LoudnessResult:
    lufs: float | None
    loudness_range: float | None
    true_peak_db: float | None


class AudioBackend(ABC):
    name = "base"

    @abstractmethod
    def is_available(self) -> tuple[bool, str]: ...

    @abstractmethod
    def decode_mono(self, path: Path, sample_rate: int = ANALYSIS_SR) -> np.ndarray:
        """Float32 mono samples in [-1, 1]."""

    def stream_mono(self, path: Path, sample_rate: int = ANALYSIS_SR, chunk_seconds: float = 10.0, should_cancel: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        """Mono float32 samples in chunks, so a long file never has to be held in memory at once. The default slices ``decode_mono``; backends override it with a real stream."""
        samples = self.decode_mono(path, sample_rate)
        step = max(1, int(chunk_seconds * sample_rate))
        for i in range(0, len(samples), step):
            if should_cancel is not None and should_cancel():
                return
            yield samples[i:i + step]

    @abstractmethod
    def measure_loudness(self, path: Path) -> LoudnessResult: ...

    @abstractmethod
    def render_chain(self, src: Path, dst: Path, filter_chain: str) -> Path: ...

    @abstractmethod
    def render_graph(self, inputs: list[Path], graph: str, out_label: str, dst: Path, duration: float) -> Path: ...


class FFmpegAudioBackend(AudioBackend):
    name = "ffmpeg"

    def __init__(self, ffmpeg_path: Callable[[], str] | str = "") -> None:
        self._path = ffmpeg_path if callable(ffmpeg_path) else (lambda p=ffmpeg_path: p)

    def _exe(self) -> str:
        return locate_binary("ffmpeg", self._path())

    def is_available(self) -> tuple[bool, str]:
        try:
            self._exe()
            return True, ""
        except FFmpegUnavailableError as exc:
            return False, exc.user_message

    def _run(self, args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
        exe = self._exe()
        r = subprocess.run([exe, "-hide_banner", "-nostdin", *args], capture_output=True, timeout=timeout)
        return r

    def decode_mono(self, path: Path, sample_rate: int = ANALYSIS_SR) -> np.ndarray:
        if not Path(path).is_file():
            raise AudioError(f"The audio file is missing: {Path(path).name}")
        r = self._run(["-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"])
        if r.returncode != 0 or not r.stdout:
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unsupported or corrupt audio"]
            raise AudioError(f"{Path(path).name} cannot be decoded ({msg[0]}).")
        return np.frombuffer(r.stdout, dtype="<f4").copy()

    def stream_mono(self, path: Path, sample_rate: int = ANALYSIS_SR, chunk_seconds: float = 10.0, should_cancel: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
        if not Path(path).is_file():
            raise AudioError(f"The audio file is missing: {Path(path).name}")
        import tempfile

        step = max(1, int(chunk_seconds * sample_rate)) * 4
        flags = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
        with tempfile.TemporaryFile() as err:
            p = subprocess.Popen([self._exe(), "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
                                 stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL, **flags)
            assert p.stdout is not None
            carry = b""
            total = 0
            try:
                while True:
                    if should_cancel is not None and should_cancel():
                        return
                    block = p.stdout.read(step)
                    if not block:
                        break
                    block = carry + block
                    usable = len(block) - len(block) % 4
                    carry = block[usable:]
                    if usable:
                        total += usable
                        yield np.frombuffer(block[:usable], dtype="<f4").copy()
            finally:
                if p.poll() is None:
                    p.kill()
                p.wait()
            if p.returncode != 0 or total == 0:
                err.seek(0)
                lines = err.read().decode("utf-8", "replace").strip().splitlines()[-1:] or ["unsupported or corrupt audio"]
                raise AudioError(f"{Path(path).name} cannot be decoded ({lines[0]}).")

    def measure_loudness(self, path: Path) -> LoudnessResult:
        try:
            r = self._run(["-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"], timeout=300)
        except Exception:  # pragma: no cover - loudness is optional
            return LoudnessResult(None, None, None)
        text = r.stderr.decode("utf-8", "replace")
        tail = text[text.rfind("Summary:"):] if "Summary:" in text else ""

        def grab(label: str) -> float | None:
            m = re.search(rf"{label}:\s+(-?[0-9.]+)\s", tail)
            try:
                return float(m.group(1)) if m else None
            except ValueError:
                return None

        lufs = grab("I")
        if lufs is not None and lufs < -69:  # silence reads as -70 LUFS
            lufs = None
        return LoudnessResult(lufs, grab("LRA"), grab("Peak"))

    def render_chain(self, src: Path, dst: Path, filter_chain: str) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = self._run(["-y", "-v", "error", "-i", str(src), "-vn", "-af", filter_chain or "anull", "-ar", "44100", str(dst)])
        if r.returncode != 0 or not dst.is_file():
            raise AudioError("The voice processing chain could not be applied.", details=r.stderr.decode("utf-8", "replace")[-300:])
        return dst

    def render_graph(self, inputs: list[Path], graph: str, out_label: str, dst: Path, duration: float) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        args = ["-y", "-v", "error"]
        for p in inputs:
            args += ["-i", str(p)]
        args += ["-filter_complex", graph, "-map", f"[{out_label}]", "-t", f"{duration:.3f}", "-ar", "44100", "-ac", "2", str(dst)]
        r = self._run(args, timeout=600)
        if r.returncode != 0 or not dst.is_file():
            raise AudioError("The audio preview could not be mixed.", details=r.stderr.decode("utf-8", "replace")[-300:])
        return dst
