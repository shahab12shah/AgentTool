"""Render errors. Every failure names the stage it happened in and the most likely cause, so the UI can say something useful."""

from __future__ import annotations

from app.core.exceptions import AppError


class RenderError(AppError):
    """A render could not be completed. The project is never touched by a failed render."""

    def __init__(self, user_message: str, *, stage: str = "", kind: str = "render_failed", possible_issue: str = "", details: str | None = None,
                 can_fallback_cpu: bool = False, asset_ids: list[str] | None = None) -> None:
        super().__init__(user_message, details=details)
        self.stage = stage
        self.kind = kind  # missing_media | unsupported_codec | invalid_path | ffmpeg_failed | validation_failed | preflight_failed | disk_full | hardware_failed ...
        self.possible_issue = possible_issue
        self.can_fallback_cpu = can_fallback_cpu
        self.asset_ids = asset_ids or []


class RenderCancelled(Exception):
    """Raised inside the render pipeline when the user cancelled."""


class EncoderUnavailableError(RenderError):
    """The selected codec/encoder is not available in this FFmpeg build."""

    def __init__(self, user_message: str, alternatives: list[str] | None = None, **kw) -> None:
        super().__init__(user_message, kind="unsupported_codec", **kw)
        self.alternatives = alternatives or []
