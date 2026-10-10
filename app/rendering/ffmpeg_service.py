"""FFmpegService: the only place that starts FFmpeg/FFprobe for rendering.

Commands are always structured argument lists (never shell strings), so paths with spaces, parentheses, apostrophes or non-ASCII
characters need no quoting. The service detects the binaries and what they can do, runs a command while parsing ``-progress`` output,
honours cancellation (ask FFmpeg to quit, then terminate, then kill) and returns the tail of stderr for diagnostics.
"""

from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.exceptions import FFmpegUnavailableError
from app.logging.logger import get_logger
from app.media.media_probe import locate_binary

_log = get_logger(__name__)
_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
MIN_VERSION = (4, 4)
STDERR_TAIL = 60


@dataclass
class FFmpegVersion:
    text: str
    major: int = 0
    minor: int = 0
    parsed: bool = False  # False for git/nightly builds whose number cannot be compared

    def at_least(self, major: int, minor: int = 0) -> bool:
        return True if not self.parsed else (self.major, self.minor) >= (major, minor)


@dataclass
class Capabilities:
    encoders: set[str] = field(default_factory=set)
    filters: set[str] = field(default_factory=set)
    muxers: set[str] = field(default_factory=set)

    def has_filters(self, *names: str) -> list[str]:
        """The names that are *missing*."""
        return [n for n in names if n not in self.filters]


@dataclass
class ProgressInfo:
    out_time: float = 0.0  # seconds of output produced so far
    frame: int = 0
    fps: float = 0.0
    speed: float | None = None  # x realtime
    total_size: int = 0
    ended: bool = False


@dataclass
class ExecResult:
    returncode: int
    stderr_tail: str = ""
    elapsed: float = 0.0
    cancelled: bool = False
    stalled: bool = False
    last: ProgressInfo = field(default_factory=ProgressInfo)
    command: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.cancelled and not self.stalled


ProgressFn = Callable[[ProgressInfo], None]
LineFn = Callable[[str], None]


def _parse_time(value: str) -> float | None:
    m = re.match(r"^(-?)(\d+):(\d+):(\d+(?:\.\d+)?)$", value.strip())
    if not m:
        return None
    s = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))
    return -s if m.group(1) else s


def escape_filter_value(text: str) -> str:
    """Escape ``text`` for use as a *quoted-free* filtergraph option value (``\\``, ``'``, ``:``, ``,``, ``;``, ``[``, ``]`` and newlines)."""
    out = []
    for ch in text:
        if ch in "\\':,;[]=":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        else:
            out.append(ch)
    return "".join(out)


def escape_filter_path(path: str | Path) -> str:
    """A file path as a filtergraph option value. Windows paths (``C:\\Projects\\My Video\\...``) become ``C\\:/Projects/My Video/...``.

    The renderer avoids absolute paths in filtergraphs altogether (it runs FFmpeg in a working directory and uses relative names), so this is
    only a safety net for the rare place where an absolute path is unavoidable.
    """
    p = str(path).replace("\\", "/")
    return escape_filter_value(p)


def redact(arg: str) -> str:
    """Never write credentials to logs: URL query secrets and ``key=``/``token=`` values are masked."""
    return re.sub(r"(?i)((?:api[_-]?key|token|secret|password|auth)=)[^&\s]+", r"\1***", arg)


class FFmpegService:
    def __init__(self, ffmpeg_path_getter: Callable[[], str] = lambda: "", ffprobe_path_getter: Callable[[], str] = lambda: "") -> None:
        self._ffmpeg_cfg, self._ffprobe_cfg = ffmpeg_path_getter, ffprobe_path_getter
        self._lock = threading.RLock()
        self._version: dict[str, FFmpegVersion] = {}
        self._caps: dict[str, Capabilities] = {}
        self._hw: dict[str, dict[str, bool]] = {}

    # ------------------------------------------------------------------ detection
    def ffmpeg(self) -> str:
        return locate_binary("ffmpeg", self._ffmpeg_cfg())

    def ffprobe(self) -> str:
        return locate_binary("ffprobe", self._ffprobe_cfg())

    def detect(self) -> tuple[bool, str]:
        """(usable, message) — checks both binaries and the version."""
        try:
            self.ffprobe()
            v = self.version()
        except FFmpegUnavailableError as exc:
            return False, exc.user_message
        if not v.at_least(*MIN_VERSION):
            return False, f"FFmpeg {v.major}.{v.minor} is too old; {MIN_VERSION[0]}.{MIN_VERSION[1]} or newer is required."
        return True, v.text

    def _run(self, args: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", **_NO_WINDOW)

    def version(self) -> FFmpegVersion:
        exe = self.ffmpeg()
        with self._lock:
            if exe in self._version:
                return self._version[exe]
        try:
            out = self._run([exe, "-version"], 10).stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            raise FFmpegUnavailableError("FFmpeg was found but could not be run.", details=str(exc)) from exc
        line = out[0] if out else ""
        m = re.search(r"version\s+n?(\d+)\.(\d+)", line)
        v = FFmpegVersion(line.strip() or exe, int(m.group(1)), int(m.group(2)), True) if m else FFmpegVersion(line.strip() or exe)
        with self._lock:
            self._version[exe] = v
        return v

    def capabilities(self) -> Capabilities:
        exe = self.ffmpeg()
        with self._lock:
            if exe in self._caps:
                return self._caps[exe]
        caps = Capabilities()
        for flag, target, rx in (("-encoders", caps.encoders, _ENC_RE), ("-filters", caps.filters, _FILTER_RE), ("-muxers", caps.muxers, _MUX_RE)):
            try:
                text = self._run([exe, "-hide_banner", flag], 20).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            for line in text.splitlines():
                m = rx.match(line)
                if m:
                    target.add(m.group(1))
        with self._lock:
            self._caps[exe] = caps
        return caps

    def encoder_available(self, name: str) -> bool:
        return name in self.capabilities().encoders

    def hardware_encoders(self, candidates: list[str] | None = None) -> dict[str, bool]:
        """Which hardware encoders really work on this machine. Listed encoders are test-encoded once (a driver or GPU is needed, not just a build flag)."""
        exe = self.ffmpeg()
        caps = self.capabilities()
        names = [n for n in (candidates or _HW_NAMES) if n in caps.encoders]
        with self._lock:
            cache = self._hw.setdefault(exe, {})
        for n in names:
            if n in cache:
                continue
            args = [exe, "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=10:d=0.5", "-frames:v", "3", "-c:v", n, "-f", "null", "-"]
            try:
                ok = self._run(args, 20).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
            with self._lock:
                cache[n] = ok
        return {n: cache.get(n, False) for n in names}

    def seed_hardware_encoders(self, results: dict[str, bool]) -> None:
        """Pre-fill the test-encode cache from a previous run (HardwareCapabilityService's on-disk cache) so a restart does not repeat the tests. Names already tested stay as they are."""
        exe = self.ffmpeg()
        with self._lock:
            cache = self._hw.setdefault(exe, {})
            for n, ok in results.items():
                cache.setdefault(str(n), bool(ok))

    def forget(self) -> None:
        """Drop cached detection (after the user changed the FFmpeg path in Settings)."""
        with self._lock:
            self._version.clear()
            self._caps.clear()
            self._hw.clear()

    # ------------------------------------------------------------------ execution
    def run(self, args: list[str], timeout: float = 60.0, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        """Run FFmpeg/FFprobe to completion (short commands). ``args[0]`` must already be the executable."""
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", cwd=str(cwd) if cwd else None, **_NO_WINDOW)

    def run_progress(self, args: list[str], *, on_progress: ProgressFn | None = None, cancel: threading.Event | None = None, on_line: LineFn | None = None,
                     stall_timeout: float = 300.0, cwd: Path | None = None) -> ExecResult:
        """Run a long FFmpeg command. ``args`` must contain ``-progress pipe:1`` for progress; stderr lines go to ``on_line``."""
        started = time.monotonic()
        res = ExecResult(-1, command=list(args))
        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                    cwd=str(cwd) if cwd else None, **_NO_WINDOW)
        except OSError as exc:
            res.stderr_tail = f"Could not start FFmpeg: {exc}"
            return res
        tail: deque[str] = deque(maxlen=STDERR_TAIL)
        events: queue.Queue = queue.Queue()

        def read_err() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip("\r\n")
                if line:
                    tail.append(line)
                    if on_line:
                        try:
                            on_line(line)
                        except Exception:  # a logging problem must never break the render
                            pass

        def read_out() -> None:
            assert proc.stdout is not None
            cur = ProgressInfo()
            for line in proc.stdout:
                k, _, v = line.strip().partition("=")
                if k == "out_time_us" or (k == "out_time_ms" and not getattr(cur, "_us", False)):
                    try:
                        cur.out_time = max(0.0, int(v) / 1e6)
                    except ValueError:
                        pass
                elif k == "out_time":
                    t = _parse_time(v)
                    if t is not None and cur.out_time == 0.0:
                        cur.out_time = max(0.0, t)
                elif k == "frame":
                    cur.frame = int(v) if v.isdigit() else cur.frame
                elif k == "fps":
                    try:
                        cur.fps = float(v)
                    except ValueError:
                        pass
                elif k == "speed":
                    try:
                        cur.speed = float(v.strip().rstrip("x"))
                    except ValueError:
                        cur.speed = None
                elif k == "total_size":
                    cur.total_size = int(v) if v.isdigit() else cur.total_size
                elif k == "progress":
                    cur.ended = v == "end"
                    events.put(ProgressInfo(cur.out_time, cur.frame, cur.fps, cur.speed, cur.total_size, cur.ended))
            events.put(None)

        threads = [threading.Thread(target=read_err, daemon=True), threading.Thread(target=read_out, daemon=True)]
        for t in threads:
            t.start()
        last_change = time.monotonic()
        stdout_done = False
        while True:
            try:
                ev = events.get(timeout=0.1)
                if ev is None:
                    stdout_done = True
                else:
                    if ev.out_time != res.last.out_time or ev.frame != res.last.frame:
                        last_change = time.monotonic()
                    res.last = ev
                    if on_progress:
                        try:
                            on_progress(ev)
                        except Exception:
                            _log.debug("progress callback failed", exc_info=True)
            except queue.Empty:
                pass
            if cancel is not None and cancel.is_set() and not res.cancelled:
                res.cancelled = True
                self._stop(proc)
            if not res.stalled and not res.cancelled and stall_timeout and time.monotonic() - last_change > stall_timeout and proc.poll() is None:
                res.stalled = True
                self._stop(proc, graceful=False)
            if proc.poll() is not None and stdout_done:
                break
            if proc.poll() is not None and not threads[1].is_alive():
                break
        for t in threads:
            t.join(timeout=2)
        res.returncode = proc.returncode if proc.returncode is not None else -1
        res.stderr_tail = "\n".join(tail)
        res.elapsed = time.monotonic() - started
        return res

    @staticmethod
    def _stop(proc: subprocess.Popen, graceful: bool = True) -> None:
        """Ask FFmpeg to finish (``q``), then terminate, then kill. Output files may be incomplete afterwards: callers delete them."""
        try:
            if graceful and proc.stdin:
                try:
                    proc.stdin.write("q\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                for _ in range(20):
                    if proc.poll() is not None:
                        return
                    time.sleep(0.1)
            proc.terminate()
            for _ in range(30):
                if proc.poll() is not None:
                    return
                time.sleep(0.1)
            proc.kill()
        except OSError:
            pass


_ENC_RE = re.compile(r"^\s*[VAS][F.][S.][X.][B.][D.]\s+(\S+)")
_FILTER_RE = re.compile(r"^\s*[T.][S.][C.]\s+(\S+)\s+\S*->\S*")
_MUX_RE = re.compile(r"^\s*[D ]?E\s+(\S+)")
_HW_NAMES = ["h264_nvenc", "hevc_nvenc", "av1_nvenc", "h264_qsv", "hevc_qsv", "av1_qsv", "h264_amf", "hevc_amf", "av1_amf", "h264_videotoolbox", "hevc_videotoolbox"]
