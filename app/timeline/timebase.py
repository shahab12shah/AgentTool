"""Frame-exact time arithmetic for the editor.

Clip positions stay plain seconds in the project file (no schema change for existing projects); this module is how editing code turns seconds into
*integer frame numbers* and back, so repeated trims / moves never accumulate floating-point error: every frame-aligned edit is computed from an integer frame
index with exact rational arithmetic (``Fraction``) and only converted to a float at the end, so ``to_frame(to_seconds(n)) == n`` for every ``n``.

The timebase distinguishes the things the editor must not conflate:
  * video frame boundaries (``to_frame`` / ``snap``): used for picture clips, captions and graphics;
  * audio sample positions (``snap_samples``): audio clips keep sample precision and are NOT forced onto frame boundaries;
  * project timestamps (seconds on the timeline) and source-media timestamps (``source_in/out`` of a clip, in the source's own seconds).

Fractional rates (23.976, 29.97, 59.94) are supported through ``Fraction``; timecode labels are non-drop-frame ("NDF": frame numbers count at the nominal rate).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

_EPS = Fraction(1, 1_000_000)  # tolerance (in frames) for "already on a frame boundary" decisions
_COMMON = {23.976: Fraction(24000, 1001), 29.97: Fraction(30000, 1001), 59.94: Fraction(60000, 1001), 119.88: Fraction(120000, 1001)}


def to_rate(fps: float | int | Fraction | tuple[int, int] | str | None) -> Fraction:
    """A positive rational frame rate from an int / float / ``"30000/1001"`` / ``(num, den)`` (``None`` or invalid -> 30)."""
    try:
        if isinstance(fps, tuple):
            r = Fraction(int(fps[0]), int(fps[1]))
        elif isinstance(fps, str):
            r = Fraction(fps)
        elif isinstance(fps, Fraction):
            r = fps
        elif fps is None:
            r = Fraction(30)
        else:
            f = float(fps)
            r = next((v for k, v in _COMMON.items() if abs(f - k) < 0.005), None) or Fraction(f).limit_denominator(1001)
        return r if r > 0 else Fraction(30)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return Fraction(30)


class Timebase:
    __slots__ = ("rate", "fps", "nominal", "frame_s")

    def __init__(self, fps: float | int | Fraction | tuple[int, int] | str | None = 30) -> None:
        self.rate = to_rate(fps)
        self.fps = float(self.rate)
        self.nominal = max(1, int(round(self.fps)))  # frames per timecode second
        self.frame_s = float(1 / self.rate)

    def __repr__(self) -> str:
        return f"Timebase({self.rate})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timebase) and other.rate == self.rate

    def __hash__(self) -> int:
        return hash(self.rate)

    # ------------------------------------------------------------------ frames <-> seconds
    def to_frame(self, t: float, mode: str = "round") -> int:
        """Frame index of a time. ``mode``: ``round`` (nearest, ties up) | ``floor`` | ``ceil``. A time already within a millionth of a frame of a boundary counts as on it."""
        x = Fraction(float(t)) * self.rate
        n = math.floor(x + Fraction(1, 2)) if mode == "round" else math.floor(x + _EPS) if mode == "floor" else math.ceil(x - _EPS)
        return int(n)

    def to_seconds(self, frame: int) -> float:
        return float(Fraction(int(frame)) / self.rate)

    def snap(self, t: float, mode: str = "round") -> float:
        return self.to_seconds(self.to_frame(t, mode))

    def is_on_frame(self, t: float) -> bool:
        x = Fraction(float(t)) * self.rate
        return abs(x - round(x)) <= Fraction(1, 10_000)

    def frames(self, seconds: float) -> int:
        """A duration in frames (rounded to the nearest)."""
        return self.to_frame(seconds, "round")

    def add_frames(self, t: float, n: int) -> float:
        """``t`` moved by ``n`` frames (from t's own frame index, so a chain of calls never drifts)."""
        return self.to_seconds(self.to_frame(t, "round") + int(n))

    # ------------------------------------------------------------------ audio
    def snap_samples(self, t: float, sample_rate: int = 48000) -> float:
        """Audio edits keep sample precision (never rounded to a video frame)."""
        sr = max(1, int(sample_rate))
        return float(Fraction(math.floor(Fraction(float(t)) * sr + Fraction(1, 2)), sr))

    # ------------------------------------------------------------------ timecode
    def format(self, t: float | None, with_frames: bool = True) -> str:
        """``HH:MM:SS:FF`` (non-drop). With ``with_frames=False``: ``HH:MM:SS.mmm``."""
        if t is None:
            return "--:--:--:--"
        t = max(0.0, float(t))
        if not with_frames:
            ms = int(round(t * 1000))
            h, rem = divmod(ms, 3_600_000)
            m, rem = divmod(rem, 60_000)
            s, ms = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        n = self.to_frame(t, "round")
        secs, ff = divmod(n, self.nominal)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"

    _TC = re.compile(r"^\s*(?:(\d+):)?(?:(\d+):)?(\d+):(\d+)\s*$")
    _FRAMES = re.compile(r"^\s*(-?\d+)\s*f\s*$", re.I)
    _SECS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*s?\s*$", re.I)

    def parse(self, text: str) -> float:
        """Seconds from what a user types: ``1:02:03:04`` / ``02:03:04`` / ``3:04`` (``[[HH:]MM:]SS:FF``), ``12f`` (frames), ``1.25`` or ``1.25s`` (seconds). Raises ``ValueError`` with a plain-language message."""
        s = (text or "").strip()
        if not s:
            raise ValueError("Enter a time.")
        m = self._FRAMES.match(s)
        if m:
            return self.to_seconds(int(m.group(1)))
        m = self._SECS.match(s)
        if m and ":" not in s:
            return float(m.group(1))
        m = self._TC.match(s)
        if m and s.count(":") >= 1:
            parts = [int(p) for p in s.replace(" ", "").split(":")]
            ff = parts[-1]
            if ff >= self.nominal:
                raise ValueError(f"Frames must be 0–{self.nominal - 1} at {self.fps:g} fps.")
            secs = 0
            for p in parts[:-1]:
                secs = secs * 60 + p
            return self.to_seconds(secs * self.nominal + ff)
        raise ValueError("Use HH:MM:SS:FF, a number of seconds (1.25), or frames (12f).")


def timebase_of(project) -> Timebase:
    """The timebase of a project (its ``settings.fps``)."""
    try:
        return Timebase(project.settings.fps)
    except AttributeError:
        return Timebase(30)
