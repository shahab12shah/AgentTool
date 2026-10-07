"""RenderValidator: trust the file, not FFmpeg's exit code.

After a render FFprobe inspects the output: it must exist, be readable, have the expected streams, duration, size, frame rate, codec and
aspect ratio, and — if the timeline has a voice-over — that voice must actually be audible. A failed check makes the render FAILED.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.exceptions import AppError
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.probe import MediaProbeService, ProbeInfo

CODEC_NAMES = {"h264": {"h264"}, "h265": {"hevc"}, "vp9": {"vp9"}, "av1": {"av1"}, "aac": {"aac"}, "opus": {"opus"}, "flac": {"flac"}}


@dataclass
class Expectations:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    expect_audio: bool
    sample_rate: int = 48000
    voice_ranges: list[tuple[float, float]] = field(default_factory=list)
    tolerance: float = 0.5
    expect_audible: bool = True  # False when every audio clip is muted / at zero volume on purpose


@dataclass
class Check:
    id: str
    label: str
    status: str  # ok | warning | error
    message: str = ""


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    probe: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.status == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, cid: str, label: str, status: str, message: str = "") -> None:
        self.checks.append(Check(cid, label, status, message))

    def summary(self) -> str:
        if self.ok:
            return "Output validated" + (f" with {len(self.warnings)} warning(s)." if self.warnings else ".")
        return "; ".join(c.message for c in self.errors)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks], "probe": self.probe, "measured": self.measured}


_VOL = re.compile(r"(mean_volume|max_volume):\s*(-?[\d.]+|-inf)\s*dB")


class RenderValidator:
    def __init__(self, ffmpeg: FFmpegService, probe: MediaProbeService) -> None:
        self.ff, self.probe = ffmpeg, probe

    def validate(self, path: Path, exp: Expectations) -> ValidationReport:
        rep = ValidationReport()
        path = Path(path)
        if not path.is_file():
            rep.add("exists", "File exists", "error", "The output file was not created.")
            return rep
        if path.stat().st_size <= 0:
            rep.add("exists", "File exists", "error", "The output file is empty.")
            return rep
        rep.add("exists", "File exists", "ok")
        try:
            info = self.probe.probe(path, use_cache=False)
        except AppError as exc:
            rep.add("readable", "File readable", "error", f"The output file cannot be read back ({exc.user_message}).")
            return rep
        rep.add("readable", "File readable", "ok")
        rep.probe = info.to_dict()
        self._video(rep, info, exp)
        self._audio(rep, info, exp, path)
        return rep

    # ------------------------------------------------------------------ video
    def _video(self, rep: ValidationReport, info: ProbeInfo, exp: Expectations) -> None:
        if not info.has_video:
            rep.add("video_stream", "Video stream", "error", "The output has no video stream.")
            return
        rep.add("video_stream", "Video stream", "ok")
        dur = info.duration or 0.0
        if dur <= 0:
            rep.add("duration", "Duration", "error", "The output has zero duration.")
        else:
            diff = abs(dur - exp.duration)
            hard = max(exp.tolerance * 4, 2.0, exp.duration * 0.05)
            if diff <= exp.tolerance:
                rep.add("duration", "Duration", "ok", f"{dur:.2f}s (expected {exp.duration:.2f}s)")
            elif diff <= hard:
                rep.add("duration", "Duration", "warning", f"The output is {dur:.2f}s but the timeline is {exp.duration:.2f}s.")
            else:
                rep.add("duration", "Duration", "error", f"The output is {dur:.2f}s long but the timeline is {exp.duration:.2f}s — part of the video is missing or extra.")
        rep.measured["duration"] = dur
        if (info.width, info.height) != (exp.width, exp.height):
            rep.add("resolution", "Resolution", "error", f"The output is {info.width}×{info.height}, expected {exp.width}×{exp.height}.")
        else:
            rep.add("resolution", "Resolution", "ok", f"{info.width}×{info.height}")
        ar_out, ar_exp = (info.width or 1) / (info.height or 1), exp.width / exp.height
        if abs(ar_out - ar_exp) > 0.005:
            rep.add("aspect", "Aspect ratio", "error", f"The aspect ratio of the output ({ar_out:.3f}) differs from the project ({ar_exp:.3f}).")
        else:
            rep.add("aspect", "Aspect ratio", "ok")
        if info.fps is None or abs(info.fps - exp.fps) > 0.05:
            rep.add("fps", "Frame rate", "error", f"The output runs at {info.fps or 0:.3f} fps, expected {exp.fps:g}.")
        else:
            rep.add("fps", "Frame rate", "ok", f"{info.fps:g} fps")
        if info.codec not in CODEC_NAMES.get(exp.video_codec, {exp.video_codec}):
            rep.add("video_codec", "Video codec", "error", f"The video codec is {info.codec}, expected {exp.video_codec}.")
        else:
            rep.add("video_codec", "Video codec", "ok", str(info.codec))
        if info.pix_fmt and info.pix_fmt not in ("yuv420p", "yuvj420p"):
            rep.add("pix_fmt", "Pixel format", "warning", f"The pixel format is {info.pix_fmt}; some players only support yuv420p.")
        else:
            rep.add("pix_fmt", "Pixel format", "ok", str(info.pix_fmt))

    # ------------------------------------------------------------------ audio
    def _audio(self, rep: ValidationReport, info: ProbeInfo, exp: Expectations, path: Path) -> None:
        if not exp.expect_audio:
            rep.add("audio_stream", "Audio stream", "ok", "no audio in the timeline")
            return
        if not info.has_audio:
            rep.add("audio_stream", "Audio stream", "error", "The timeline has audio but the output has no audio stream.")
            return
        rep.add("audio_stream", "Audio stream", "ok")
        if info.audio_codec not in CODEC_NAMES.get(exp.audio_codec, {exp.audio_codec}):
            rep.add("audio_codec", "Audio codec", "error", f"The audio codec is {info.audio_codec}, expected {exp.audio_codec}.")
        else:
            rep.add("audio_codec", "Audio codec", "ok", str(info.audio_codec))
        if info.sample_rate != exp.sample_rate:
            rep.add("sample_rate", "Audio sample rate", "error", f"The audio sample rate is {info.sample_rate} Hz, expected {exp.sample_rate} Hz.")
        else:
            rep.add("sample_rate", "Audio sample rate", "ok", f"{info.sample_rate} Hz")
        if info.channels not in (1, 2):
            rep.add("channels", "Audio channels", "error", f"Unexpected channel count {info.channels}.")
        if info.audio_duration is not None:
            adiff = abs(info.audio_duration - exp.duration)
            if adiff > max(exp.tolerance * 4, 2.0):
                rep.add("audio_duration", "Audio duration", "error", f"The audio is {info.audio_duration:.2f}s long but the timeline is {exp.duration:.2f}s.")
            elif adiff > exp.tolerance:
                rep.add("audio_duration", "Audio duration", "warning", f"The audio is {info.audio_duration:.2f}s but the timeline is {exp.duration:.2f}s.")
            else:
                rep.add("audio_duration", "Audio duration", "ok", f"{info.audio_duration:.2f}s")
        loud = self._volume(path, 0.0, None)
        rep.measured["peak_db"] = loud.get("max_volume")
        rep.measured["mean_db"] = loud.get("mean_volume")
        peak = loud.get("max_volume")
        if exp.expect_audible and (peak is None or peak < -80.0):
            rep.add("mute", "Audio audible", "error", "The audio is completely silent although the timeline has audible audio.")
        else:
            rep.add("mute", "Audio audible", "ok")
        if peak is not None and peak >= -0.05:
            rep.add("clipping", "Clipping", "warning", f"The audio peaks at {peak:.2f} dB — it may be clipping.")
        else:
            rep.add("clipping", "Clipping", "ok", "" if peak is None else f"peak {peak:.1f} dB")
        if exp.voice_ranges:
            worst = None
            for a, b in exp.voice_ranges:
                m = self._volume(path, a, min(b - a, 90.0))
                mean = m.get("mean_volume")
                if mean is None or mean < -65.0:
                    worst = (a, mean)
                    break
            if worst is not None:
                rep.add("voice", "Voice-over audible", "error", f"The voice-over is silent in the output (around {worst[0]:.0f}s). The export was rejected instead of delivering a broken video.")
            else:
                rep.add("voice", "Voice-over audible", "ok")

    def _volume(self, path: Path, start: float, length: float | None) -> dict[str, float | None]:
        args = [self.ff.ffmpeg(), "-hide_banner", "-nostats"]
        if start > 0:
            args += ["-ss", f"{start:.3f}"]
        if length:
            args += ["-t", f"{length:.3f}"]
        args += ["-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"]
        try:
            r = self.ff.run(args, timeout=300)
        except Exception:
            return {}
        out: dict[str, float | None] = {}
        for k, v in _VOL.findall(r.stderr or ""):
            out[k] = None if v == "-inf" else float(v)
            if v == "-inf":
                out[k] = -120.0
        return out
