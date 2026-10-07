"""Display formatting for time values. Never used for calculations."""

from __future__ import annotations


def format_timecode(seconds: float | None, fps: float | None = None) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` (or ``HH:MM:SS:FF`` when ``fps`` is given)."""
    if seconds is None:
        return "--:--"
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if fps:
        frames = int(ms / 1000.0 * fps)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_duration_short(seconds: float | None) -> str:
    """Compact duration for list views: ``m:ss`` or ``h:mm:ss``."""
    if seconds is None:
        return ""
    total = int(round(max(0.0, seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
