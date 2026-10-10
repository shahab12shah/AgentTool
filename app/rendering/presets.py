"""Export presets, quality levels and the pure functions that turn ``RenderSettings`` into concrete encoder choices.

Nothing here runs FFmpeg: it is the table of "what does High mean for H.264 / HEVC / VP9 / AV1 / NVENC...", plus the rules for output size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.project.project_schema import RenderSettings

QUALITY_LEVELS = ("draft", "standard", "high", "maximum", "custom")
RESOLUTIONS = {"480p": 480, "720p": 720, "1080p": 1080, "2160p": 2160}
RESOLUTION_LABELS = {"480p": "854 × 480 (Draft)", "720p": "1280 × 720 (HD)", "1080p": "1920 × 1080 (Full HD)", "2160p": "3840 × 2160 (4K)"}
FPS_CHOICES = (24, 30, 60)
CODECS = ("h264", "h265", "vp9", "av1")
CODEC_LABELS = {"h264": "H.264", "h265": "H.265 / HEVC", "vp9": "VP9", "av1": "AV1"}
AUDIO_CODECS = ("aac", "opus", "flac")
AUDIO_LABELS = {"aac": "AAC", "opus": "Opus", "flac": "FLAC"}
CONTAINERS = ("mp4", "mkv", "webm")
HARDWARE_MODES = ("auto", "cpu", "hardware")
CONTAINER_VIDEO = {"mp4": {"h264", "h265", "av1", "vp9"}, "mkv": {"h264", "h265", "av1", "vp9"}, "webm": {"vp9", "av1"}}
CONTAINER_AUDIO = {"mp4": {"aac", "opus", "flac"}, "mkv": {"aac", "opus", "flac"}, "webm": {"opus"}}
PROXY_RESOLUTIONS = {"540p": 540, "720p": 720, "1080p": 1080}

# software encoder per codec (the first one that exists in the FFmpeg build is used)
SOFTWARE_ENCODERS = {"h264": ["libx264"], "h265": ["libx265"], "vp9": ["libvpx-vp9"], "av1": ["libsvtav1", "libaom-av1"]}
HARDWARE_ENCODERS = {
    "h264": ["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"],
    "h265": ["hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox"],
    "av1": ["av1_nvenc", "av1_qsv", "av1_amf"],
    "vp9": [],
}
# constant-quality value per quality level (lower = better for CRF/CQ encoders)
CRF = {
    "libx264": {"draft": 30, "standard": 23, "high": 19, "maximum": 15},
    "libx265": {"draft": 32, "standard": 27, "high": 23, "maximum": 19},
    "libvpx-vp9": {"draft": 42, "standard": 34, "high": 29, "maximum": 23},
    "libsvtav1": {"draft": 42, "standard": 34, "high": 28, "maximum": 22},
    "libaom-av1": {"draft": 42, "standard": 34, "high": 28, "maximum": 22},
}
SPEED_PRESET = {
    "libx264": {"draft": "ultrafast", "standard": "medium", "high": "slow", "maximum": "slower"},
    "libx265": {"draft": "ultrafast", "standard": "medium", "high": "medium", "maximum": "slow"},
    "libvpx-vp9": {"draft": "8", "standard": "4", "high": "2", "maximum": "1"},  # -cpu-used
    "libsvtav1": {"draft": "12", "standard": "8", "high": "6", "maximum": "4"},
    "libaom-av1": {"draft": "8", "standard": "6", "high": "4", "maximum": "3"},
}
HW_QUALITY = {"draft": 34, "standard": 28, "high": 23, "maximum": 19}


def resolve_quality(s: RenderSettings, encoder: str, hardware: bool) -> tuple[int, str]:
    """(constant-quality value, speed preset) for ``encoder``. Hardware encoders use their own scale (``HW_QUALITY``), never libx264's CRF table."""
    base_q = "high" if s.quality == "custom" else s.quality
    custom = s.quality == "custom" and bool(s.crf)
    if hardware:
        return (s.crf if custom else HW_QUALITY[base_q]), s.encoder_preset
    return (s.crf if custom else CRF[encoder][base_q]), (s.encoder_preset or SPEED_PRESET[encoder][base_q])


def hw_rate_args(encoder: str, q: int, bitrate_kbps: int = 0, preset: str = "") -> list[str]:
    """Rate-control options for a hardware encoder: NVENC ``-rc vbr -cq``, QSV ``-global_quality``, AMF ``-rc cqp -qp_i/-qp_p``, VideoToolbox ``-q:v`` (an explicit bitrate wins)."""
    args: list[str] = []
    if "nvenc" in encoder:
        args += ["-preset", preset or "p5", "-rc", "vbr", "-cq", str(q), "-b:v", f"{bitrate_kbps}k" if bitrate_kbps else "0"]
    elif "qsv" in encoder:
        args += ["-global_quality", str(q), "-preset", preset or "slow"]
        if bitrate_kbps:
            args += ["-b:v", f"{bitrate_kbps}k"]
    elif "amf" in encoder:
        args += ["-quality", "quality", "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q)] if not bitrate_kbps else ["-b:v", f"{bitrate_kbps}k"]
    elif "videotoolbox" in encoder:
        args += ["-b:v", f"{bitrate_kbps}k"] if bitrate_kbps else ["-q:v", str(max(1, min(100, 100 - q * 2)))]
    return args


def hw_kind(encoder: str) -> str:
    """``nvenc | qsv | amf | videotoolbox`` for a hardware encoder name, ``cpu`` for anything else."""
    return next((k for k in ("nvenc", "qsv", "amf", "videotoolbox") if k in encoder), "cpu")


@dataclass(frozen=True)
class ExportPreset:
    id: str
    name: str
    description: str
    values: dict = field(default_factory=dict)


EXPORT_PRESETS: dict[str, ExportPreset] = {
    "youtube_1080p": ExportPreset("youtube_1080p", "YouTube 1080p", "1920×1080 · H.264 · AAC · project FPS",
                                  {"resolution": "1080p", "video_codec": "h264", "audio_codec": "aac", "container": "mp4", "quality": "high", "fps": 0}),
    "youtube_4k": ExportPreset("youtube_4k", "YouTube 4K", "3840×2160 · H.264 · AAC · project FPS",
                               {"resolution": "2160p", "video_codec": "h264", "audio_codec": "aac", "container": "mp4", "quality": "high", "fps": 0}),
    "high_quality": ExportPreset("high_quality", "High Quality", "Project resolution · H.264 · maximum quality",
                                 {"resolution": "1080p", "video_codec": "h264", "audio_codec": "aac", "container": "mp4", "quality": "maximum", "audio_bitrate_kbps": 320, "fps": 0}),
    "draft": ExportPreset("draft", "Draft", "Fast low-resolution check of timing, captions, graphics and audio",
                          {"resolution": "480p", "video_codec": "h264", "audio_codec": "aac", "container": "mp4", "quality": "draft", "audio_bitrate_kbps": 96, "fps": 0}),
}


def apply_preset(settings: RenderSettings, preset_id: str) -> RenderSettings:
    """A copy of ``settings`` with the preset's values applied (everything stays editable afterwards)."""
    from dataclasses import replace

    p = EXPORT_PRESETS.get(preset_id)
    if p is None:
        return replace(settings, preset_id="custom")
    return replace(settings, preset_id=preset_id, **p.values)


def output_size(canvas_w: int, canvas_h: int, resolution: str) -> tuple[int, int]:
    """Pixel size for a short-edge resolution (``1080p``) keeping the project's aspect ratio exactly (even numbers for the encoder)."""
    short_target = RESOLUTIONS.get(resolution, min(canvas_w, canvas_h))
    short, long_ = min(canvas_w, canvas_h), max(canvas_w, canvas_h)
    long_target = int(round(short_target * long_ / short / 2.0)) * 2
    short_target = int(round(short_target / 2.0)) * 2
    return (long_target, short_target) if canvas_w >= canvas_h else (short_target, long_target)


def resolution_for(canvas_w: int, canvas_h: int) -> str:
    """The closest named resolution for a canvas (used to pick a sensible default)."""
    short = min(canvas_w, canvas_h)
    return min(RESOLUTIONS, key=lambda k: abs(RESOLUTIONS[k] - short))


def aspect_label(w: int, h: int) -> str:
    r = w / h
    for label, v in (("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0), ("4:3", 4 / 3), ("21:9", 21 / 9)):
        if abs(r - v) < 0.01:
            return label
    return f"{w}:{h}"


@dataclass
class ResolvedOutput:
    """Everything the command builder needs to encode, decided from the settings and the machine's encoders."""

    width: int
    height: int
    fps: int
    container: str
    video_codec: str
    encoder: str
    hardware: bool
    audio_codec: str
    audio_encoder: str
    audio_bitrate_kbps: int
    sample_rate: int
    quality: str
    pix_fmt: str = "yuv420p"
    crf: int = 0
    bitrate_kbps: int = 0
    speed_preset: str = ""
    notes: list[str] = field(default_factory=list)  # why choices were made (shown in the preflight and written to the log)

    def video_args(self) -> list[str]:
        keyint = str(max(2, int(round(self.fps * 2))))
        tail = ["-g", keyint, "-pix_fmt", self.pix_fmt]
        if self.hardware:
            return ["-c:v", self.encoder, *self.rate_args(), *tail, *(["-tag:v", "hvc1"] if "hevc" in self.encoder and self.container == "mp4" else [])]
        return ["-c:v", self.encoder, *self.rate_args(), *tail]

    def rate_args(self) -> list[str]:
        """Quality / rate-control / speed options of the chosen encoder (everything between ``-c:v <encoder>`` and ``-g``). Each encoder family has its own vocabulary: libx264's ``-crf`` is never reused for a hardware encoder."""
        e, q = self.encoder, self.crf
        if self.hardware:
            return hw_rate_args(e, q, self.bitrate_kbps, self.speed_preset)
        args: list[str] = []
        rate = ["-b:v", f"{self.bitrate_kbps}k", "-maxrate", f"{int(self.bitrate_kbps * 1.5)}k", "-bufsize", f"{self.bitrate_kbps * 2}k"] if self.bitrate_kbps else []
        if e == "libx264":
            args += ["-preset", self.speed_preset, *(rate or ["-crf", str(q)]), "-profile:v", "high"]
        elif e == "libx265":
            args += ["-preset", self.speed_preset, *(rate or ["-crf", str(q)]), "-x265-params", "log-level=error"]
            if self.container == "mp4":
                args += ["-tag:v", "hvc1"]
        elif e == "libvpx-vp9":
            args += ["-cpu-used", self.speed_preset, "-row-mt", "1", "-deadline", "realtime" if self.quality == "draft" else "good",
                     *(rate or ["-crf", str(q), "-b:v", "0"])]
        elif e == "libsvtav1":
            args += ["-preset", self.speed_preset, *(rate or ["-crf", str(q)])]
        elif e == "libaom-av1":
            args += ["-cpu-used", self.speed_preset, "-row-mt", "1", *(rate or ["-crf", str(q), "-b:v", "0"])]
        else:
            args += rate or ["-crf", str(q)]
        return args

    def audio_args(self) -> list[str]:
        a = ["-c:a", self.audio_encoder, "-ar", str(self.sample_rate), "-ac", "2"]
        if self.audio_encoder != "flac":
            a += ["-b:a", f"{self.audio_bitrate_kbps}k"]
        return a

    @property
    def segment_ext(self) -> str:
        """Container of the intermediate video sections. MPEG-TS has a 90 kHz clock, so every frame time is exact for 24/30/60 fps (Matroska's 1 ms clock is not)."""
        return "ts" if self.video_codec in ("h264", "h265") else "mkv"

    def segment_args(self) -> list[str]:
        return ["-muxdelay", "0", "-muxpreload", "0", "-f", "mpegts"] if self.segment_ext == "ts" else []

    def container_args(self) -> list[str]:
        return ["-movflags", "+faststart"] if self.container == "mp4" else []

    def summary(self) -> dict:
        return {"size": f"{self.width}x{self.height}", "fps": self.fps, "container": self.container, "video_encoder": self.encoder, "hardware": self.hardware,
                "audio_encoder": self.audio_encoder, "quality": self.quality, "crf": self.crf, "bitrate_kbps": self.bitrate_kbps, "preset": self.speed_preset,
                "sample_rate": self.sample_rate, "audio_bitrate_kbps": self.audio_bitrate_kbps, "notes": list(self.notes)}


def compatibility_problems(s: RenderSettings) -> list[str]:
    out: list[str] = []
    if s.container not in CONTAINERS:
        out.append(f"Unknown container “{s.container}”.")
        return out
    if s.video_codec not in CODECS:
        out.append(f"Unknown video codec “{s.video_codec}”.")
    elif s.video_codec not in CONTAINER_VIDEO[s.container]:
        out.append(f"{CODEC_LABELS[s.video_codec]} cannot be stored in a .{s.container} file.")
    if s.audio_codec not in AUDIO_CODECS:
        out.append(f"Unknown audio codec “{s.audio_codec}”.")
    elif s.audio_codec not in CONTAINER_AUDIO[s.container]:
        out.append(f"{AUDIO_LABELS[s.audio_codec]} audio cannot be stored in a .{s.container} file.")
    if s.resolution not in RESOLUTIONS:
        out.append(f"Unsupported resolution “{s.resolution}”.")
    if s.fps not in (0, *FPS_CHOICES):
        out.append(f"Unsupported frame rate {s.fps}. Use 24, 30 or 60 FPS.")
    if s.quality not in QUALITY_LEVELS:
        out.append(f"Unknown quality level “{s.quality}”.")
    if s.hardware_acceleration not in HARDWARE_MODES:
        out.append(f"Unknown hardware setting “{s.hardware_acceleration}”.")
    if s.quality == "custom" and s.crf and not (0 <= s.crf <= 63):
        out.append("CRF must be between 0 and 63.")
    if s.audio_bitrate_kbps < 32 or s.audio_bitrate_kbps > 640:
        out.append("Audio bitrate must be between 32 and 640 kbps.")
    if s.audio_sample_rate not in (44100, 48000, 96000):
        out.append("Audio sample rate must be 44100, 48000 or 96000 Hz.")
    return out
