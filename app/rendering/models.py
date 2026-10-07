"""Render data model: stages, status, the frozen ``RenderSnapshot``, the ``RenderPlan``, preflight reports and render records."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from app.core.serialization import to_plain
from app.project.project_schema import RenderSettings
from app.timeline.track import Track, TrackKind


class RenderStage(str, Enum):
    VALIDATING = "Validating Project"
    PREPARING = "Preparing Media"
    COMPILING = "Compiling Timeline"
    VIDEO_GRAPH = "Building Video Graph"
    AUDIO_GRAPH = "Building Audio Graph"
    RENDERING = "Rendering"
    ENCODING = "Encoding"
    VALIDATING_OUTPUT = "Validating Output"
    FINALIZING = "Finalizing"


STAGE_ORDER = list(RenderStage)


class RenderStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CANCELING = "CANCELING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

    @property
    def is_terminal(self) -> bool:
        return self in (RenderStatus.COMPLETED, RenderStatus.FAILED, RenderStatus.CANCELED)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------------------------- snapshot
@dataclass
class AssetRef:
    """What the renderer needs to know about one asset. ``path`` is absolute: the snapshot never resolves paths again."""

    id: str
    type: str  # video | image | audio
    name: str
    path: str
    duration: float | None = None
    width: int | None = None  # display size (rotation applied); filled while preparing media
    height: int | None = None
    fps: float | None = None
    has_audio: bool = False
    has_alpha: bool = False
    rotation: int = 0
    content_hash: str | None = None
    size_bytes: int = 0
    exists: bool = True


@dataclass
class ProxyRef:
    path: str = ""
    resolution: str = ""
    status: str = "NONE"
    width: int = 0
    height: int = 0
    source_size: int = 0  # fingerprint of the original the proxy was made from
    source_mtime_ns: int = 0


@dataclass
class SceneRef:
    id: str
    start: float
    end: float
    title: str = ""


@dataclass
class RenderSnapshot:
    """The project as the render sees it. Built once when the render starts and never changed: editing continues freely while it renders."""

    snapshot_id: str
    project_id: str
    project_name: str
    timeline_version: int
    created_at: str
    canvas_w: int
    canvas_h: int
    fps: int
    tracks: list[Track]
    assets: dict[str, AssetRef]
    voice_asset_id: str
    scenes: list[SceneRef]
    caption_settings: Any
    caption_styles: dict[str, Any]
    audio_settings: Any
    audio_processing: Any
    settings: RenderSettings
    proxies: dict[str, ProxyRef] = field(default_factory=dict)
    project_root: str = ""

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_project(cls, project, settings: RenderSettings, proxies: dict[str, ProxyRef] | None = None) -> "RenderSnapshot":
        assets: dict[str, AssetRef] = {}
        used = {c.asset_id for t in project.timeline.tracks for c in t.clips if c.asset_id}
        for a in project.assets.all():
            if a.id not in used and a.id != project.voice_over.asset_id:
                continue
            path = project.asset_path(a)
            probe = (a.extra or {}).get("probe", {})
            rot = int(probe.get("rotation", 0) or 0)
            w, h = a.width, a.height
            if rot in (90, 270) and w and h:
                w, h = h, w
            try:
                st = path.stat()
                size, exists = st.st_size, True
            except OSError:
                size, exists = a.size_bytes, False
            assets[a.id] = AssetRef(a.id, a.type.value, a.name, str(path), a.duration, w, h, a.fps, a.has_audio, bool(probe.get("has_alpha", False)), rot,
                                    a.content_hash, size, exists)
        return cls(
            snapshot_id=new_id("snap"), project_id=project.project_id, project_name=project.project_name, timeline_version=project.timeline_version,
            created_at=now_iso(), canvas_w=project.settings.width, canvas_h=project.settings.height, fps=project.settings.fps,
            tracks=[_copy_track(t) for t in project.timeline.tracks], assets=assets, voice_asset_id=project.voice_over.asset_id or "",
            scenes=[SceneRef(s.id, s.start, s.end, getattr(s, "topic", "") or getattr(s, "title", "")) for s in project.scenes],
            caption_settings=copy.deepcopy(project.caption_settings), caption_styles=copy.deepcopy(project.caption_styles),
            audio_settings=copy.deepcopy(project.audio_settings), audio_processing=copy.deepcopy(project.audio_processing),
            settings=copy.deepcopy(settings), proxies=dict(proxies or {}), project_root=str(project.root) if project.root else "")

    # ------------------------------------------------------------------ queries
    @property
    def duration(self) -> float:
        return max((c.timeline_end for t in self.tracks for c in t.clips), default=0.0)

    def track(self, track_id: str) -> Track | None:
        return next((t for t in self.tracks if t.id == track_id), None)

    def clips(self):
        for t in self.tracks:
            for c in t.clips:
                yield t, c

    def content_hash(self) -> str:
        """Fingerprint of everything that changes the picture or the sound (not of ids, timestamps or the output location)."""
        doc = {
            "canvas": [self.canvas_w, self.canvas_h, self.fps], "voice": self.voice_asset_id,
            "tracks": [t.to_dict() for t in self.tracks],
            "assets": {k: [a.path, a.size_bytes, a.content_hash, a.duration, a.width, a.height] for k, a in sorted(self.assets.items())},
            "captions": to_plain(self.caption_settings), "styles": to_plain(self.caption_styles), "audio": to_plain(self.audio_settings),
            "processing": to_plain(self.audio_processing),
        }
        return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, "project_id": self.project_id, "project_name": self.project_name, "timeline_version": self.timeline_version,
                "created_at": self.created_at, "canvas": [self.canvas_w, self.canvas_h], "fps": self.fps, "content_hash": self.content_hash(),
                "tracks": [t.to_dict() for t in self.tracks], "assets": {k: asdict(a) for k, a in self.assets.items()}, "voice_asset_id": self.voice_asset_id,
                "scenes": [asdict(s) for s in self.scenes], "caption_settings": to_plain(self.caption_settings), "caption_styles": to_plain(self.caption_styles),
                "audio_settings": to_plain(self.audio_settings), "audio_processing": to_plain(self.audio_processing), "render_settings": self.settings.to_dict(),
                "proxies": {k: asdict(v) for k, v in self.proxies.items()}}


def _copy_track(t: Track) -> Track:
    c = copy.copy(t)
    c.clips = [clip.snapshot() for clip in t.clips]
    return c


# ---------------------------------------------------------------------------------------------- plan
@dataclass
class ChunkPlan:
    index: int
    start: float  # timeline seconds (on the frame grid)
    end: float
    frames: int
    scene_ids: list[str] = field(default_factory=list)


@dataclass
class RenderPlan:
    """Exactly what will be done before FFmpeg is started (shown in the log and the preflight)."""

    project_id: str
    timeline_version: int
    output_resolution: tuple[int, int]
    fps: int
    duration: float
    video_tracks: list[dict[str, Any]]
    audio_tracks: list[dict[str, Any]]
    graphics: int
    captions: int
    effects: list[str]
    transitions: list[str]
    output_format: str
    quality: str
    video_codec: str = ""
    encoder: str = ""
    hardware: bool = False
    audio_codec: str = ""
    total_frames: int = 0
    chunks: list[ChunkPlan] = field(default_factory=list)
    uses_proxy_assets: list[str] = field(default_factory=list)
    has_audio: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_resolution"] = list(self.output_resolution)
        return d


# ---------------------------------------------------------------------------------------------- preflight
@dataclass
class PreflightItem:
    id: str
    label: str
    status: str  # ok | warning | error
    message: str = ""
    fix: str = ""  # what the user can do
    asset_ids: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def mark(self) -> str:
        return {"ok": "✓", "warning": "!", "error": "✗"}[self.status]


@dataclass
class PreflightReport:
    items: list[PreflightItem] = field(default_factory=list)
    plan: RenderPlan | None = None
    required_bytes: int = 0
    available_bytes: int = 0
    missing_assets: list[str] = field(default_factory=list)
    proxy_only_assets: list[str] = field(default_factory=list)  # original missing but a proxy exists

    @property
    def errors(self) -> list[PreflightItem]:
        return [i for i in self.items if i.status == "error"]

    @property
    def warnings(self) -> list[PreflightItem]:
        return [i for i in self.items if i.status == "warning"]

    @property
    def can_start(self) -> bool:
        return not self.errors

    def item(self, item_id: str) -> PreflightItem | None:
        return next((i for i in self.items if i.id == item_id), None)

    def text(self) -> str:
        lines = ["PREFLIGHT CHECK", ""]
        for i in self.items:
            lines.append(f"{i.mark} {i.message or i.label}")
            lines.extend(f"    {d}" for d in i.details[:6])
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- progress / record
@dataclass
class RenderProgress:
    stage: RenderStage = RenderStage.VALIDATING
    overall: float = 0.0  # 0..1, derived from real FFmpeg progress
    video: float = 0.0
    audio: float = 0.0
    encode: float = 0.0
    scene_index: int = 0
    scene_total: int = 0
    chunk_index: int = 0
    chunk_total: int = 0
    speed: float | None = None
    elapsed: float = 0.0
    eta: float | None = None
    output_bytes: int = 0
    message: str = ""
    cached_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


@dataclass
class RenderRecord:
    """One render, as stored in ``project.render_history``: which timeline, which settings, which file."""

    render_id: str
    project_id: str
    timeline_version: int
    timeline_hash: str
    created_at: str
    status: str = RenderStatus.QUEUED.value
    settings: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    finished_at: str = ""
    duration: float = 0.0
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    error: str = ""
    failed_stage: str = ""
    possible_issue: str = ""
    log_path: str = ""
    snapshot_path: str = ""
    warnings: list[str] = field(default_factory=list)
    kind: str = "export"  # export | draft | preview
    proxy_used: bool = False
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderRecord":
        from dataclasses import fields

        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def has_audio_clips(snapshot: RenderSnapshot) -> bool:
    return any(t.kind is TrackKind.AUDIO and not t.muted and t.clips for t in snapshot.tracks)


def disk_free(path: Path) -> int:
    import shutil

    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return -1


def same_filesystem(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False
