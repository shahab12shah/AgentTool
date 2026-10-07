"""Application-wide constants. This module must stay dependency-free."""

from __future__ import annotations

APP_NAME = "AgentTool"
APP_VERSION = "0.1.0"
SCHEMA_VERSION = 2  # 2 = Phase 2: transcription, alignment, scenes, visual intents, preferences

PROJECT_FILE = "project.json"
PROJECT_BACKUP_SUFFIX = ".bak"
PROJECT_TMP_SUFFIX = ".tmp"

# Directory layout created for every new project (relative to project root).
PROJECT_SUBDIRS: tuple[str, ...] = (
    "media/video",
    "media/images",
    "media/audio",
    "media/screenshots",
    "generated",
    "proxies",
    "thumbnails",
    "waveforms",
    "previews",
    "renders",
    "cache",
)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".aac", ".m4a"})

# Project creation options.
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "1920×1080": (1920, 1080),
    "3840×2160": (3840, 2160),
}
FPS_OPTIONS: tuple[int, ...] = (24, 30, 60)
ASPECT_RATIOS: tuple[str, ...] = ("16:9", "9:16", "1:1")
DEFAULT_RESOLUTION = "1920×1080"
DEFAULT_FPS = 30
DEFAULT_ASPECT_RATIO = "16:9"

# Timeline.
MIN_CLIP_DURATION = 0.05  # seconds
DEFAULT_IMAGE_DURATION = 5.0  # seconds
TIME_EPSILON = 1e-6

DEFAULT_AUTOSAVE_SECONDS = 60
MIN_AUTOSAVE_SECONDS = 10
UNDO_LIMIT = 500


def resolve_dimensions(preset: str, aspect_ratio: str) -> tuple[int, int]:
    """Pixel size for a resolution preset oriented by aspect ratio.

    The preset names the long edge: 1920×1080 + 9:16 -> 1080×1920, + 1:1 -> 1080×1080.
    """
    long_edge, short_edge = max(RESOLUTION_PRESETS[preset]), min(RESOLUTION_PRESETS[preset])
    if aspect_ratio == "9:16":
        return short_edge, long_edge
    if aspect_ratio == "1:1":
        return short_edge, short_edge
    return long_edge, short_edge
