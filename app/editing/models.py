"""Phase 4 data model: editing settings, scene editing briefs, plans, decisions, generation state.

Everything here is plain dataclasses (Qt-free, JSON-serialisable through ``core.serialization``). The timeline stays the
source of truth; these records explain *why* each timeline element exists and who owns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.editing.overrides import EditingStrategyOverrides


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DecisionType(str, Enum):
    VISUAL_TIMING = "VISUAL_TIMING"
    CUT = "CUT"
    TRIM = "TRIM"
    ZOOM = "ZOOM"
    PAN = "PAN"
    KEYFRAME = "KEYFRAME"
    TEXT = "TEXT"
    NUMBER_EMPHASIS = "NUMBER_EMPHASIS"
    EVIDENCE_FOCUS = "EVIDENCE_FOCUS"
    TRANSITION = "TRANSITION"
    AUDIO_DUCK = "AUDIO_DUCK"
    CAPTION_EMPHASIS = "CAPTION_EMPHASIS"


class Creator(str, Enum):
    AI = "AI"
    USER = "USER"
    SYSTEM = "SYSTEM"


class Operation(str, Enum):
    """Non-destructive timeline operations a segment can be the result of."""

    CUT = "CUT"
    TRIM = "TRIM"
    SPLIT = "SPLIT"
    EXTEND = "EXTEND"
    SHORTEN = "SHORTEN"
    REPLACE = "REPLACE"
    HOLD = "HOLD"


class ZoomKind(str, Enum):
    NO_ZOOM = "NO_ZOOM"
    SUBTLE_ZOOM = "SUBTLE_ZOOM"
    PUNCH_IN = "PUNCH_IN"
    PUNCH_OUT = "PUNCH_OUT"
    CUSTOM = "CUSTOM"


class PanKind(str, Enum):
    PAN_LEFT = "PAN_LEFT"
    PAN_RIGHT = "PAN_RIGHT"
    PAN_UP = "PAN_UP"
    PAN_DOWN = "PAN_DOWN"
    ZOOM_IN = "ZOOM_IN"
    ZOOM_OUT = "ZOOM_OUT"
    CUSTOM = "CUSTOM"


class TransitionType(str, Enum):
    CUT = "CUT"
    FADE = "FADE"
    DISSOLVE = "DISSOLVE"
    WIPE = "WIPE"
    SLIDE = "SLIDE"


class TextStyle(str, Enum):
    HEADLINE = "HEADLINE"
    LOWER_THIRD = "LOWER_THIRD"
    NUMBER_CARD = "NUMBER_CARD"
    WARNING = "WARNING"
    LABEL = "LABEL"
    DEFINITION = "DEFINITION"
    COMPARISON = "COMPARISON"
    DATE = "DATE"
    LOCATION = "LOCATION"
    ENTITY_NAME = "ENTITY_NAME"


class Emphasis(str, Enum):
    SUBTLE_HIGHLIGHT = "SUBTLE_HIGHLIGHT"
    BOLD_TEXT = "BOLD_TEXT"
    PUNCH_TEXT = "PUNCH_TEXT"
    SCREEN_CENTER_TEXT = "SCREEN_CENTER_TEXT"
    LOWER_THIRD = "LOWER_THIRD"
    WARNING_TEXT = "WARNING_TEXT"
    NUMBER_CARD = "NUMBER_CARD"


class VisualStatus(str, Enum):
    APPROVED = "APPROVED"  # an approved visual with a usable project asset
    MISSING = "MISSING"  # nothing approved: never invented silently
    UNAPPROVED = "UNAPPROVED"  # a candidate was chosen but not approved yet
    SKIPPED = "SKIPPED"  # the user chose no visual
    MISSING_MEDIA = "MISSING_MEDIA"  # approved, but the media file cannot be found


class SceneEditStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    NEEDS_VISUAL = "NEEDS_VISUAL"  # edited without a visual (text/audio only)
    OUTDATED = "OUTDATED"  # the scene changed since it was edited
    LOCKED = "LOCKED"


def confidence_label(confidence: float) -> str:
    """90–100 High, 80–89 Good, 70–79 Review, <70 Low."""
    return "High" if confidence >= 90 else "Good" if confidence >= 80 else "Review" if confidence >= 70 else "Low"


# ------------------------------------------------------------------ settings / presets
@dataclass
class EditingSettings:
    style: str = "professional"  # documentary | professional | dynamic
    pacing: float = 0.5  # 0 = slow ... 1 = fast
    motion_intensity: float = 0.5  # 0 = low ... 1 = high
    transition_frequency: float = 0.5
    text_emphasis: bool = True
    number_emphasis: bool = True
    evidence_treatment: bool = True
    smart_transitions: bool = True
    smart_audio_ducking: bool = True
    caption_mode: str = "ENABLED"  # ENABLED | DISABLED (instructions only; nothing is burned in)
    provider: str = "rule_based"
    user_set: list[str] = field(default_factory=list)  # fields the user changed on purpose: a reference style never overrides these (unless the user allows it)
    reference: EditingStrategyOverrides | None = None  # filled in for the engine run only (see editing/effective.py); never saved with the project settings

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditingSettings":
        return from_plain(cls, d)


@dataclass
class VideoProfile:
    """Whole-video context computed once before any scene is edited (no scene is an isolated edit)."""

    topic: str = ""
    scene_count: int = 0
    duration: float = 0.0
    median_wps: float = 2.5  # words per second of speech
    median_scene_seconds: float = 0.0
    evidence_scene_ids: list[str] = field(default_factory=list)
    section_starts: list[str] = field(default_factory=list)  # scene ids that open a new section
    importance_mean: float = 0.5
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ scene analysis
@dataclass
class SceneEditingBrief:
    scene_id: str
    importance: float = 0.5
    narration_speed: float = 2.5  # words/second while speaking
    speed_class: str = "NORMAL"  # SLOW | NORMAL | FAST
    pause_seconds: float = 0.0
    longest_pause: float = 0.0
    sentence_count: int = 1
    visual_complexity: float = 0.4  # 0..1
    information_density: float = 0.4  # 0..1
    emotional_intensity: float = 0.3  # 0..1
    recommended_pacing: str = "NORMAL"
    recommended_motion: str = "SUBTLE_ZOOM"
    recommended_text: list[str] = field(default_factory=list)
    recommended_evidence_treatment: str = "NONE"
    recommended_transition: str = "CUT"
    # structured answers (never free-form reasoning)
    keep_static: bool = False
    should_move: bool = True
    should_zoom: bool = True
    should_pan: bool = False
    change_during_sentence: bool = False
    text_needed: bool = False
    evidence_treatment_needed: bool = False
    has_number: bool = False
    has_date: bool = False
    introduces_entity: bool = False
    needs_visual_change: bool = False
    continue_previous: bool = False
    overlap_next: bool = False
    visual_status: str = VisualStatus.MISSING.value
    factors: list[str] = field(default_factory=list)  # concise decision factors


# ------------------------------------------------------------------ plan pieces (provider output)
@dataclass
class VisualSegment:
    visual_segment_id: str
    scene_id: str
    asset_id: str
    start: float  # timeline seconds
    duration: float
    reason: str = ""
    source_in: float = 0.0
    source_out: float = 0.0
    speed: float = 1.0
    operation: str = Operation.CUT.value
    slot: str = "visual:0"
    candidate_id: str = ""
    fit: str = "cover"  # cover | contain (how the renderer should base-fit the media)
    reuse_count: int = 0
    previous_scene_id: str = ""
    reuse_reason: str = ""
    continues_previous: bool = False
    confidence: float = 85.0


@dataclass
class MotionPlan:
    kind: str = ZoomKind.NO_ZOOM.value  # ZoomKind or PanKind value
    family: str = "ZOOM"  # ZOOM | PAN
    start_scale: float = 1.0
    end_scale: float = 1.0
    start_pos: tuple[float, float] = (0.0, 0.0)  # pixels from canvas centre
    end_pos: tuple[float, float] = (0.0, 0.0)
    interpolation: str = "ease_in_out"
    reason: str = ""
    confidence: float = 85.0


@dataclass
class EvidencePlan:
    region: tuple[float, float, float, float] = (0.2, 0.25, 0.6, 0.3)  # normalised x, y, w, h
    region_detected: bool = False  # True only when a real source located it (never guessed silently)
    zoom_scale: float = 1.8
    highlight: bool = True
    darken: bool = True
    hold_wide: float = 0.8
    reason: str = ""
    confidence: float = 70.0


@dataclass
class TextGraphic:
    text_id: str
    content: str
    start: float
    duration: float
    position: tuple[float, float] = (0.5, 0.5)  # normalised canvas coordinates of the anchor
    style: str = TextStyle.LABEL.value
    emphasis: str = ""
    animation: str = "fade"
    importance: float = 0.5
    source_scene: str = ""
    source_ref: str = ""  # where the content came from: "script" | "transcript" | word ids
    font: str = "Sans"
    size: int = 56
    alignment: str = "center"
    opacity: float = 1.0
    background: str = "none"  # none | box | gradient

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextGraphic":
        return from_plain(cls, d)


@dataclass
class TransitionPlan:
    type: str = TransitionType.CUT.value
    duration: float = 0.0
    reason: str = ""
    confidence: float = 90.0


@dataclass
class DuckPlan:
    start: float
    end: float
    music_level: float
    kind: str = "DUCK"  # DUCK | RISE | FADE_IN | FADE_OUT
    ramp: float = 0.3
    reason: str = ""
    confidence: float = 85.0


@dataclass
class PlannedSegment:
    segment: VisualSegment
    motion: MotionPlan | None = None
    evidence: EvidencePlan | None = None
    transition: TransitionPlan | None = None  # only the first segment of a scene


@dataclass
class PlannedText:
    graphic: TextGraphic
    decision_type: str = DecisionType.TEXT.value
    slot: str = ""
    reason: str = ""
    confidence: float = 85.0


@dataclass
class ScenePlan:
    scene_id: str
    brief: SceneEditingBrief
    segments: list[PlannedSegment] = field(default_factory=list)
    texts: list[PlannedText] = field(default_factory=list)
    ducks: list[DuckPlan] = field(default_factory=list)
    caption_emphasis: list[str] = field(default_factory=list)
    caption_region: str = "bottom_safe_area"
    input_hash: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenePlan":
        return from_plain(cls, d)


# ------------------------------------------------------------------ decisions & ownership
@dataclass
class EditingDecision:
    decision_id: str
    scene_id: str
    type: DecisionType
    slot: str = ""
    target_id: str = ""  # clip id (or "" for pure instructions such as audio ducking)
    start: float = 0.0
    duration: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # one concise sentence; never chain-of-thought
    confidence: float = 85.0  # 0..100
    created_by: Creator = Creator.AI
    overrides_decision_id: str = ""
    locked: bool = False
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditingDecision":
        return from_plain(cls, d)


@dataclass
class OverrideRecord:
    """The AI decision a user decision replaced (kept so nothing the AI proposed is lost)."""

    override_id: str  # the USER decision
    original: EditingDecision
    at: str = field(default_factory=now_iso)


@dataclass
class AudioPlan:
    voice_level: float = 1.0
    music_level: float = 0.18
    sfx_level: float = 0.30
    duck_level: float = 0.10
    rise_level: float = 0.24
    fade_in: float = 1.5
    fade_out: float = 2.0
    priority: list[str] = field(default_factory=lambda: ["VOICE", "SFX", "MUSIC"])
    note: str = "Instructions only: the audio engine (a later phase) mixes music and effects."


@dataclass
class CaptionPlan:
    mode: str = "ENABLED"
    region: str = "bottom_safe_area"
    max_lines: int = 2
    emphasis: dict[str, list[str]] = field(default_factory=dict)  # scene id -> words to emphasise
    region_by_scene: dict[str, str] = field(default_factory=dict)
    burn_in: bool = False  # never: captions stay timeline data


@dataclass
class EditingStrategy:
    """The video-level strategy plus everything derived per scene (briefs, segments, audio and caption instructions)."""

    profile: VideoProfile = field(default_factory=VideoProfile)
    style: str = "professional"
    briefs: dict[str, SceneEditingBrief] = field(default_factory=dict)
    segments: dict[str, list[VisualSegment]] = field(default_factory=dict)
    audio: AudioPlan = field(default_factory=AudioPlan)
    captions: CaptionPlan = field(default_factory=CaptionPlan)
    scene_visuals: dict[str, list[str]] = field(default_factory=dict)  # extra assets the user added to a scene ("Manual Add")
    provider: str = "rule_based"


@dataclass
class SceneGeneration:
    scene_id: str
    status: SceneEditStatus = SceneEditStatus.PENDING
    error: str = ""
    input_hash: str = ""
    visual_status: str = VisualStatus.MISSING.value
    attempts: int = 0
    generated_at: str = ""


@dataclass
class EditingSession:
    session_id: str
    scope: str = "ALL"  # ALL | SELECTED | SCENE | RETRY | CHANGED
    scene_ids: list[str] = field(default_factory=list)
    status: str = "QUEUED"  # QUEUED | RUNNING | COMPLETED | FAILED | CANCELED
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed_scene: str = ""
    error: str = ""
    validation_errors: list[str] = field(default_factory=list)
    checkpoint: str = ""
    log: list[str] = field(default_factory=list)


@dataclass
class TimelineGeneration:
    version: int = 0
    status: str = "NONE"  # NONE | COMPLETE | PARTIAL | FAILED
    style: str = ""
    last_session_id: str = ""
    scenes: dict[str, SceneGeneration] = field(default_factory=dict)
    locked_scenes: list[str] = field(default_factory=list)
    suppressed_slots: list[str] = field(default_factory=list)  # "scene_id|slot" the user deleted: the AI does not re-add them
    generated_at: str = ""
