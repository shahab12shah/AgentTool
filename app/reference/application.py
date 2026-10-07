"""Settings, user adjustments and history for applying a reference style, plus the reference-asset record.

A ``ReferenceAsset`` is *not* a project asset: it lives in ``project.reference_assets`` and in ``<project>/references/``, is never added to the media
library and is never placed on the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.editing.overrides import EditingStrategyOverrides
from app.reference.style_model import DIMENSIONS, now_iso

STRENGTHS = (0.25, 0.5, 0.75, 1.0)
MODES = ("FULL", "BALANCED", "CUSTOM")
BALANCED_DIMENSIONS = ("pacing", "visual_density", "motion_intensity")  # "Balanced": only the major characteristics
APPLY_NOTICE = ("This will update AI editing preferences.\n\nIt will NOT:\n- copy reference footage\n- copy reference text\n- copy reference graphics\n- overwrite locked user edits")


@dataclass
class ReferenceSettings:
    enabled: bool = False  # True once a style has been applied (the editing engines only consult the overrides when enabled)
    analysis_version: int = 0
    application_mode: str = "FULL"  # FULL | BALANCED | CUSTOM
    custom_dimensions: list[str] = field(default_factory=lambda: list(DIMENSIONS))
    preserve_user_edits: bool = True  # user-set settings (and every locked / user-owned timeline object) win over the reference style
    style_strength: float = 1.0  # one of STRENGTHS
    active_reference_id: str = ""
    adjustments: dict[str, float] = field(default_factory=dict)  # Customize slider targets (0..100) per dimension; absent = follow the reference
    style_request: str = ""  # optional free text, shown after the OriginalityGuard turned it into an abstract instruction

    def dimensions(self) -> list[str]:
        """The style dimensions the current mode applies."""
        if self.application_mode == "BALANCED":
            return list(BALANCED_DIMENSIONS)
        if self.application_mode == "CUSTOM":
            return [d for d in DIMENSIONS if d in self.custom_dimensions]
        return list(DIMENSIONS)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceSettings":
        return from_plain(cls, d)


@dataclass
class StyleAdjustments:
    """The user's targets on the Customize sliders (0..100). ``None`` = follow the reference value for that dimension."""

    targets: dict[str, float] = field(default_factory=dict)

    def get(self, dim: str) -> float | None:
        v = self.targets.get(dim)
        return None if v is None else float(v)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StyleAdjustments":
        return from_plain(cls, d)


@dataclass
class ReferenceAsset:
    """A reference video that belongs to the *analysis*, not to the production media."""

    reference_id: str
    name: str
    path: str  # relative to the project root: references/<reference_id>/reference_video.<ext> (or absolute when linked)
    link_mode: str = "copy"  # copy | reference
    content_hash: str = ""
    size_bytes: int = 0
    imported_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)  # ReferenceMetadata as a dict
    analysis_status: str = "NONE"  # NONE | RUNNING | COMPLETED | PARTIAL | FAILED | CANCELED
    analyzed_hash: str = ""  # content hash the stored analysis belongs to
    analyzed_version: int = 0
    analyzed_settings_hash: str = ""
    analyzed_at: str = ""
    error: str = ""
    thumbnails: list[str] = field(default_factory=list)  # relative paths under references/<id>/thumbnails (a few frames, for the user's own review)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceAsset":
        return from_plain(cls, d)


@dataclass
class StyleApplication:
    """One entry of ``style_application_history``: what was applied, with which strength, and what it replaced (so it can be undone and audited)."""

    application_id: str
    reference_id: str
    applied_at: str = field(default_factory=now_iso)
    mode: str = "FULL"
    strength: float = 1.0
    dimensions: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)
    profile_signature: str = ""
    before: EditingStrategyOverrides | None = None
    after: EditingStrategyOverrides | None = None
    before_settings: ReferenceSettings | None = None
    checkpoint: str = ""  # file name of the project checkpoint written before applying
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StyleApplication":
        return from_plain(cls, d)


@dataclass
class ProjectContent:
    """What the user's own content needs (read from the Phase 2/4 analysis): the adapter never lets a style demand more than this allows."""

    scene_count: int = 0
    duration: float = 0.0
    median_scene_seconds: float = 0.0
    evidence_scene_share: float = 0.0  # documents / data visuals: need reading time
    average_information_density: float = 0.0  # 0..1
    median_words_per_second: float = 2.5
    number_scene_share: float = 0.0
    still_image_share: float = 0.0


@dataclass
class AdaptationBaseline:
    """The project's current (non-reference) settings the style is blended *from*. All plain numbers: no engine types."""

    base_shot_duration: float = 4.8
    min_shot_duration: float = 1.8
    max_shot_duration: float = 8.0
    motion_intensity: float = 0.5
    transition_frequency: float = 0.5
    text_density: float = 0.5
    text_per_minute: float = 6.0  # the preset's own text-event budget (events/min) that text_density 0.5 stands for
    caption_max_words: int = 12
    caption_style: str = "professional"
    caption_position: str = "bottom"
    keyword_emphasis_rate: float = 0.5
    music_level: float = 0.18
    ducking_strength: float = 0.5
    sfx_per_minute: float = 3.0
    pause_usage: float = 0.5
    user_set: dict[str, list[str]] = field(default_factory=dict)  # {"editing": [...], "caption": [...], "audio": [...]}: fields the user set on purpose


@dataclass
class AdaptationResult:
    overrides: EditingStrategyOverrides
    effective_targets: dict[str, float] = field(default_factory=dict)  # per dimension: the 0..100 target actually used after strength / content limits
    skipped: dict[str, str] = field(default_factory=dict)  # dimension -> reason it was not applied (unavailable, low confidence, user-protected, mode...)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
