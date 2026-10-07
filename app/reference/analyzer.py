"""ReferenceVideoAnalyzer: probe + extract + profile for one local reference video, as a pure function of the file (no project access).

Never touches a project, a timeline or the media library. The result (``ReferenceAnalysis``) is what the service caches in
``references/<id>/analysis.json`` and what the style profile / adapter read.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.exceptions import AppError, UnsupportedMediaError
from app.core.serialization import from_plain, to_plain
from app.reference.feature_extractor import (
    AnalysisSettings, ExtractionResult, LogFn, ProgressFn, ReferenceFeatureExtractor, StageProgress, aspect_ratio_text,
)
from app.reference.signals import AnalysisCancelled, ReferenceAnalysisError
from app.reference.style_model import (
    ANALYSIS_VERSION, ReferenceEvents, ReferenceFeatures, ReferenceMetadata, ReferenceStyleProfile, build_profile, now_iso,
)
from app.reference.summary import human_summary, recommendations
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.probe import MediaProbeService

log = logging.getLogger(__name__)

REFERENCE_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
HASH_CHUNK = 1 << 20


def file_hash(path: Path) -> str:
    """Streaming SHA-1 of the whole file (cheap enough for a local video, and exact: it decides whether a cached analysis still applies)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ReferenceAnalysis:
    """The complete, cacheable result of analysing one reference."""

    reference_id: str = ""
    reference_hash: str = ""
    analysis_version: int = ANALYSIS_VERSION
    settings_hash: str = ""
    created_at: str = field(default_factory=now_iso)
    status: str = "COMPLETED"  # COMPLETED | PARTIAL
    features: ReferenceFeatures = field(default_factory=ReferenceFeatures)
    events: ReferenceEvents = field(default_factory=ReferenceEvents)  # internal observation lists (cache only; never exposed to the style profile)
    profile: ReferenceStyleProfile = field(default_factory=ReferenceStyleProfile)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceAnalysis":
        return from_plain(cls, d)

    def record(self) -> dict[str, Any]:
        """The compact summary kept inside project.json (``reference_analysis[id]``); the full data stays in the cache file."""
        return {
            "reference_hash": self.reference_hash, "analysis_version": self.analysis_version, "settings_hash": self.settings_hash, "created_at": self.created_at,
            "status": self.status, "duration": round(self.features.metadata.duration, 2), "shots": len(self.features.shots),
            "confidence": {k: round(v, 3) for k, v in self.profile.confidence.items()}, "unavailable": list(self.profile.unavailable),
            "detectors": [{"name": d.name, "ok": d.ok, "skipped": d.skipped, "confidence": round(d.confidence, 3), "message": d.message} for d in self.features.detectors],
            "summary": self.summary, "warnings": list(self.profile.warnings),
        }


def describe_probe_failure(path: Path, exc: Exception) -> str:
    """A message a person can act on, for a file that could not be probed."""
    if isinstance(exc, UnsupportedMediaError):
        return f"“{path.name}” is not a supported video file. Choose an MP4, MOV, MKV, WebM or AVI video."
    msg = str(getattr(exc, "user_message", exc))
    if "no video" in msg.lower():
        return f"“{path.name}” has no video stream, so there is no editing style to analyse."
    return f"“{path.name}” could not be read. It may be corrupt or use an unsupported codec."


class ReferenceVideoAnalyzer:
    def __init__(self, ffmpeg: FFmpegService, probe: MediaProbeService | None = None, settings: AnalysisSettings | None = None) -> None:
        self.ff = ffmpeg
        self.probe = probe or MediaProbeService(ffmpeg)
        self.settings = settings or AnalysisSettings()

    def read_metadata(self, path: Path) -> ReferenceMetadata:
        """Probe the file. Raises ``ReferenceAnalysisError`` (with a readable message) for a missing, corrupt, unsupported or video-less file."""
        if not path.is_file():
            raise ReferenceAnalysisError(f"The reference video “{path.name}” was not found.")
        if path.suffix.lower() not in REFERENCE_EXTENSIONS:
            raise ReferenceAnalysisError(f"“{path.name}” is not a supported video file. Choose an MP4, MOV, MKV, WebM or AVI video.")
        try:
            info = self.probe.probe(path)
        except AppError as exc:
            raise ReferenceAnalysisError(describe_probe_failure(path, exc), details=str(exc)) from exc
        if not info.has_video or not info.width or not info.height:
            raise ReferenceAnalysisError(f"“{path.name}” has no video stream, so there is no editing style to analyse.")
        if not info.duration or info.duration < 1.0:
            raise ReferenceAnalysisError(f"“{path.name}” is too short ({info.duration or 0:.1f} s) to read an editing style from.")
        return ReferenceMetadata(float(info.duration), int(info.width), int(info.height), float(info.fps or 0.0), aspect_ratio_text(int(info.width), int(info.height)), info.codec or "",
                                 info.audio_codec or "", int(info.sample_rate or 0), True, bool(info.has_audio), int(info.size_bytes or 0), info.container or "")

    def analyze(self, path: Path, *, reference_id: str = "", content_hash: str = "", progress: ProgressFn | None = None, log_fn: LogFn | None = None,
                cancel: threading.Event | None = None) -> ReferenceAnalysis:
        pg = StageProgress(progress, log_fn)
        pg.start("Loading Reference", f"Loading {path.name}")
        if cancel is not None and cancel.is_set():
            raise AnalysisCancelled()
        content_hash = content_hash or file_hash(path)
        pg.finish("Loading Reference")
        pg.start("Probing Media")
        meta = self.read_metadata(path)
        pg.note(f"{meta.width}×{meta.height} {meta.aspect_ratio}, {meta.duration:.1f} s, {meta.fps:.2f} fps, audio: {'yes' if meta.has_audio else 'no'}.")
        pg.finish("Probing Media")
        res: ExtractionResult = ReferenceFeatureExtractor(self.ff, self.settings).extract(path, meta, progress=pg, cancel=cancel)
        feats = res.features
        if all(d.ok is False for d in feats.detectors if d.name in ("shot_detection", "motion_detection", "caption_detection", "audio_detection")):
            raise ReferenceAnalysisError("None of the analysers could read this video. It may be corrupt or use an unsupported codec.")
        pg.start("Generating Recommendations")
        profile = build_profile(feats, reference_id)
        status = "PARTIAL" if any((not d.ok) for d in feats.detectors) else "COMPLETED"
        out = ReferenceAnalysis(reference_id, content_hash, ANALYSIS_VERSION, self.settings.settings_hash(), now_iso(), status, feats, res.events, profile,
                                human_summary(profile), recommendations(profile), res.log)
        pg.finish("Generating Recommendations")
        return out

