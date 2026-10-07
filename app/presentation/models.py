"""Phase 5 data model: audio, captions, graphics and the presentation decisions that explain them.

Plain dataclasses (Qt-free, JSON-serialisable through ``core.serialization``). The timeline stays the source of truth for timing,
geometry and content of every object; these records hold settings, analysis results and the *why* (decisions, overrides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.editing.models import Creator, now_iso


# ------------------------------------------------------------------ enums
class PresentationType(str, Enum):
    CAPTION = "CAPTION"
    KEYWORD_EMPHASIS = "KEYWORD_EMPHASIS"
    NUMBER_GRAPHIC = "NUMBER_GRAPHIC"
    DATE_GRAPHIC = "DATE_GRAPHIC"
    LOWER_THIRD = "LOWER_THIRD"
    HEADLINE = "HEADLINE"
    TEXT_GRAPHIC = "TEXT_GRAPHIC"
    MOTION_GRAPHIC = "MOTION_GRAPHIC"
    EVIDENCE_GRAPHIC = "EVIDENCE_GRAPHIC"
    MUSIC = "MUSIC"
    SFX = "SFX"
    DUCKING = "DUCKING"
    VOICE_PROCESSING = "VOICE_PROCESSING"


class Part(str, Enum):
    CAPTIONS = "CAPTIONS"
    GRAPHICS = "GRAPHICS"
    AUDIO = "AUDIO"


class HighlightMode(str, Enum):
    NONE = "NONE"
    PROGRESSIVE = "PROGRESSIVE"  # words appear as they are spoken
    HIGHLIGHT = "HIGHLIGHT"  # the whole line shows, the spoken word is highlighted


class EmphasisStyle(str, Enum):
    COLOR_CHANGE = "COLOR_CHANGE"
    BOLD = "BOLD"
    SCALE = "SCALE"
    BACKGROUND_BOX = "BACKGROUND_BOX"
    UNDERLINE = "UNDERLINE"
    GLOW = "GLOW"
    POP = "POP"


class KeywordCategory(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    NUMBER = "NUMBER"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENTAGE = "PERCENTAGE"
    WARNING = "WARNING"
    DEADLINE = "DEADLINE"
    PRODUCT = "PRODUCT"
    PROCESS = "PROCESS"
    CLAIM = "CLAIM"
    CONCEPT = "CONCEPT"


class AudioIssue(str, Enum):
    CLIPPING = "CLIPPING"
    TOO_QUIET = "TOO_QUIET"
    TOO_LOUD = "TOO_LOUD"
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    LONG_SILENCE = "LONG_SILENCE"


class PreviewMode(str, Enum):
    VOICE = "VOICE"
    MUSIC = "MUSIC"
    SFX = "SFX"
    VOICE_MUSIC = "VOICE_MUSIC"
    VOICE_SFX = "VOICE_SFX"
    FULL = "FULL"


class SfxCategory(str, Enum):
    WHOOSH = "WHOOSH"
    IMPACT = "IMPACT"
    CLICK = "CLICK"
    NOTIFICATION = "NOTIFICATION"
    PAPER = "PAPER"
    CAMERA = "CAMERA"
    TRANSITION = "TRANSITION"
    TICK = "TICK"
    WARNING = "WARNING"
    DIGITAL = "DIGITAL"
    AMBIENT = "AMBIENT"


# ------------------------------------------------------------------ settings
@dataclass
class CaptionSettings:
    enabled: bool = True
    style_id: str = "professional"
    position: str = "bottom"  # bottom | center | top | custom
    custom_x: float = 0.5  # normalised anchor for position == custom
    custom_y: float = 0.85
    keyword_highlight: bool = True
    number_emphasis: bool = True
    highlight_mode: str = HighlightMode.HIGHLIGHT.value
    max_lines: int = 2
    max_words: int = 12
    uppercase: bool = False
    safe_margin_left: float = 0.06  # fractions of the canvas
    safe_margin_right: float = 0.06
    safe_margin_top: float = 0.06
    safe_margin_bottom: float = 0.08
    # accessibility
    large_text: bool = False
    high_contrast: bool = False
    reading_speed: float = 1.0  # multiplies the allowed reading rate (< 1 = slower captions)
    reduced_motion: bool = False
    # provenance: what the current captions were generated from
    generated_transcript_id: str = ""
    generated_audio_hash: str = ""
    stale_acknowledged_hash: str = ""  # the user chose "Keep existing" for this voice-over hash


@dataclass
class CaptionStyle:
    style_id: str
    name: str
    font: str = "Sans"
    size_rel: float = 0.052  # font height as a fraction of the canvas height
    weight: str = "bold"  # normal | bold
    alignment: str = "center"
    line_spacing: float = 1.12
    color: str = "#FFFFFF"
    highlight_color: str = "#F2C14E"
    background: str = "none"  # none | box
    background_color: str = "#000000"
    background_opacity: float = 0.6
    shadow: bool = True
    shadow_color: str = "#000000"
    outline_width: float = 0.0  # fraction of the font size
    outline_color: str = "#000000"
    opacity: float = 1.0
    uppercase: bool = False
    emphasis: dict[str, str] = field(default_factory=dict)  # KeywordCategory -> EmphasisStyle


@dataclass
class AudioSettings:
    music_enabled: bool = True
    sfx_enabled: bool = True
    auto_ducking: bool = True
    voice_enhancement: bool = False
    voice_level: float = 1.0
    music_level: float = 0.18  # normal narration
    important_level: float = 0.09  # important narration
    pause_level: float = 0.24  # voice pause: modest rise
    intro_level: float = 0.30  # before the voice starts / after it ends
    sfx_level: float = 0.35
    attack: float = 0.25  # seconds to duck
    release: float = 0.50  # seconds to come back
    important_threshold: float = 0.75
    pause_rise_min: float = 0.8  # pauses shorter than this do not raise the music
    sfx_duck_factor: float = 0.8
    max_sfx_per_minute: float = 3.0
    min_sfx_gap: float = 6.0
    music_fade_in: float = 1.5
    music_fade_out: float = 2.0
    loop_music: bool = True
    target_lufs: float = -16.0


@dataclass
class VoiceProcessingSettings:
    """Non-destructive voice chain. Parameters only: the original file is never modified."""

    enabled: bool = False
    gain_db: float = 0.0
    normalize: bool = False
    target_lufs: float = -16.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    compression: bool = False
    comp_threshold_db: float = -18.0
    comp_ratio: float = 3.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 120.0
    limiter: bool = False
    limiter_ceiling_db: float = -1.0
    noise_reduction: bool = False
    noise_reduction_db: float = 12.0
    highpass_hz: float = 0.0  # 0 = off
    eq_preset: str = "none"  # none | clarity | warm | broadcast


# ------------------------------------------------------------------ analysis results
@dataclass
class AudioIssueRecord:
    code: str
    severity: str  # warning | error
    message: str
    start: float | None = None
    end: float | None = None


@dataclass
class VoiceAnalysis:
    asset_id: str = ""
    audio_hash: str = ""
    duration: float = 0.0
    peak_db: float | None = None
    rms_db: float | None = None
    lufs: float | None = None  # None where the backend cannot measure it
    loudness_range: float | None = None
    dynamic_range_db: float | None = None
    clipped_samples: int = 0
    noise_floor_db: float | None = None
    silence_regions: list[list[float]] = field(default_factory=list)  # [start, end]
    pauses: list[list[float]] = field(default_factory=list)  # between spoken words (needs a transcript)
    speaking_rate_wps: float | None = None
    speech_ratio: float | None = None
    intensity_by_sentence: dict[str, float] = field(default_factory=dict)  # sentence id -> RMS dB
    emphasis_candidates: list[str] = field(default_factory=list)  # word ids spoken noticeably louder than their sentence
    speaker_changes: list[float] = field(default_factory=list)
    speaker_changes_supported: bool = False
    issues: list[AudioIssueRecord] = field(default_factory=list)
    backend: str = ""
    created_at: str = field(default_factory=now_iso)
    master_clock: str = "MASTER_TIMING_REFERENCE"


# ------------------------------------------------------------------ captions
@dataclass
class CaptionWord:
    word_id: str
    text: str
    start: float
    end: float


@dataclass
class EmphasisMark:
    word_index: int
    category: str  # KeywordCategory
    style: str  # EmphasisStyle
    reason: str = ""


@dataclass
class CaptionSegment:
    caption_id: str
    scene_id: str
    start: float
    end: float
    text: str
    lines: list[str] = field(default_factory=list)
    words: list[CaptionWord] = field(default_factory=list)
    emphasis: list[EmphasisMark] = field(default_factory=list)
    style_id: str = "professional"
    style_overrides: dict[str, Any] = field(default_factory=dict)
    position: str = "bottom"
    position_xy: list[float] = field(default_factory=list)
    animation: dict[str, Any] = field(default_factory=dict)
    highlight_mode: str = HighlightMode.HIGHLIGHT.value
    reading_cps: float = 0.0

    @property
    def emphasis_words(self) -> list[str]:
        return [self.words[m.word_index].text for m in self.emphasis if 0 <= m.word_index < len(self.words)]

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptionSegment":
        return from_plain(cls, d)


# ------------------------------------------------------------------ assignments
@dataclass
class MusicAssignment:
    assignment_id: str
    asset_id: str
    start: float = 0.0
    end: float = 0.0
    volume: float = 1.0  # clip gain; ducking keyframes are on top
    fade_in: float = 1.5
    fade_out: float = 2.0
    loop: bool = True
    ducking: bool = True
    clip_ids: list[str] = field(default_factory=list)
    created_by: str = "USER"
    locked: bool = False
    reason: str = ""
    confidence: float = 100.0
    intensity: list[list[float]] = field(default_factory=list)  # [time, multiplier] section intensity changes


@dataclass
class SfxAssignment:
    sfx_id: str
    asset_id: str
    scene_id: str = ""
    timestamp: float = 0.0
    duration: float = 0.0
    volume: float = 0.35
    fade_in: float = 0.0
    fade_out: float = 0.05
    category: str = SfxCategory.IMPACT.value
    reason: str = ""
    confidence: float = 85.0
    created_by: str = "AI"
    clip_id: str = ""
    locked: bool = False
    trigger: str = ""


@dataclass
class DuckingEvent:
    event_id: str
    start: float
    end: float
    level: float
    kind: str = "DUCK"  # DUCK | RISE | INTRO | OUTRO | SFX
    cause: str = ""
    scene_id: str = ""
    created_by: str = "AI"


# ------------------------------------------------------------------ decisions & ownership
@dataclass
class PresentationDecision:
    decision_id: str
    scene_id: str
    type: PresentationType
    slot: str = ""
    target_id: str = ""  # clip id ("" for pure instructions)
    start: float = 0.0
    duration: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # one concise sentence; never chain-of-thought
    confidence: float = 85.0
    created_by: Creator = Creator.AI
    overrides_decision_id: str = ""
    locked: bool = False
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PresentationDecision":
        return from_plain(cls, d)


@dataclass
class PresentationOverride:
    override_id: str
    original: PresentationDecision
    at: str = field(default_factory=now_iso)


@dataclass
class ScenePresentationPlan:
    scene_id: str
    caption_plan: dict[str, Any] = field(default_factory=dict)
    keyword_plan: list[dict[str, Any]] = field(default_factory=list)
    number_graphics: list[dict[str, Any]] = field(default_factory=list)
    lower_thirds: list[dict[str, Any]] = field(default_factory=list)
    text_graphics: list[dict[str, Any]] = field(default_factory=list)
    motion_graphics: list[dict[str, Any]] = field(default_factory=list)
    music_state: dict[str, Any] = field(default_factory=dict)
    sfx_events: list[dict[str, Any]] = field(default_factory=list)
    ducking_events: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    input_hash: str = ""


@dataclass
class PartState:
    version: int = 0
    status: str = "NONE"  # NONE | COMPLETE | FAILED
    generated_at: str = ""
    input_hash: str = ""  # what it was generated from (voice/transcript/settings)
    error: str = ""


@dataclass
class PresentationGeneration:
    captions: PartState = field(default_factory=PartState)
    graphics: PartState = field(default_factory=PartState)
    audio: PartState = field(default_factory=PartState)
    locked_scenes: list[str] = field(default_factory=list)
    suppressed_slots: list[str] = field(default_factory=list)  # "scene|slot" the user deleted
    scene_status: dict[str, dict[str, str]] = field(default_factory=dict)  # part -> scene id -> COMPLETE | FAILED | PENDING
    version: int = 0


@dataclass
class PresentationSession:
    session_id: str
    parts: list[str] = field(default_factory=list)
    scope: str = "ALL"
    scene_ids: list[str] = field(default_factory=list)
    status: str = "RUNNING"  # RUNNING | COMPLETED | FAILED | CANCELED
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    error: str = ""
    failed_scene: str = ""
    failed_part: str = ""
    completed: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    checkpoint: str = ""
    log: list[str] = field(default_factory=list)
