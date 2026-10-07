"""Rendering service boundary.

Phase 1 implements validation and plan building only. ``render`` raises
``NotAvailableInPhase``: nothing is faked or copied around to look like a render.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import FFmpegUnavailableError, NotAvailableInPhase
from app.media.media_probe import locate_binary
from app.project.project import Project


@dataclass
class RenderIssue:
    severity: str  # "error" | "warning"
    message: str


@dataclass
class RenderSegment:
    clip_id: str
    track_id: str
    asset_path: Path
    timeline_start: float
    duration: float
    source_in: float
    source_out: float
    speed: float
    is_audio: bool


@dataclass
class RenderPlan:
    width: int
    height: int
    fps: int
    duration: float
    segments: list[RenderSegment] = field(default_factory=list)


class Renderer(ABC):
    @abstractmethod
    def validate(self, project: Project) -> list[RenderIssue]: ...

    @abstractmethod
    def build_render_plan(self, project: Project) -> RenderPlan: ...

    @abstractmethod
    def render(self, project: Project, output: Path, plan: RenderPlan | None = None) -> Path: ...


class FFmpegRenderer(Renderer):
    def __init__(self, ffmpeg_path: str = "") -> None:
        self._ffmpeg_path = ffmpeg_path

    def validate(self, project: Project) -> list[RenderIssue]:
        issues: list[RenderIssue] = []
        try:
            locate_binary("ffmpeg", self._ffmpeg_path)
        except FFmpegUnavailableError as exc:
            issues.append(RenderIssue("error", exc.user_message))
        if not project.timeline.all_clips():
            issues.append(RenderIssue("error", "The timeline is empty — add at least one clip."))
        for asset in project.missing_assets():
            issues.append(RenderIssue("error", f"Media file is missing: {asset.name}"))
        for track in project.timeline.tracks:
            if track.hidden and track.clips:
                issues.append(RenderIssue("warning", f"Track “{track.name}” is hidden and will be skipped."))
        return issues

    def build_render_plan(self, project: Project) -> RenderPlan:
        segments: list[RenderSegment] = []
        for track in project.timeline.tracks:
            if track.hidden:
                continue
            for clip in track.clips:
                if clip.kind != "media":  # text/graphic overlays are timeline data for the future renderer
                    continue
                asset = project.assets.require(clip.asset_id)
                segments.append(
                    RenderSegment(
                        clip.id, track.id, project.asset_path(asset), clip.timeline_start, clip.duration,
                        clip.source_in, clip.source_out, clip.speed, track.is_audio,
                    )
                )
        s = project.settings
        return RenderPlan(s.width, s.height, s.fps, project.timeline.duration, segments)

    def render(self, project: Project, output: Path, plan: RenderPlan | None = None) -> Path:
        raise NotAvailableInPhase("Rendering is not implemented yet (planned for Phase 3).")
