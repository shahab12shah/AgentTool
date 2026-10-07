"""Structured-narrative models: entities, claims, numbers, visual intent, scenes, analysis state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain


class EntityType(str, Enum):
    PERSON = "PERSON"
    COMPANY = "COMPANY"
    ORGANIZATION = "ORGANIZATION"
    COUNTRY = "COUNTRY"
    CITY = "CITY"
    PRODUCT = "PRODUCT"
    OBJECT = "OBJECT"
    FINANCIAL_INSTRUMENT = "FINANCIAL_INSTRUMENT"
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"
    TECHNOLOGY = "TECHNOLOGY"
    OTHER = "OTHER"


class ClaimType(str, Enum):
    FACT = "FACT"
    NUMBER = "NUMBER"
    DATE = "DATE"
    LAW = "LAW"
    RULE = "RULE"
    QUOTE = "QUOTE"
    PREDICTION = "PREDICTION"
    OPINION = "OPINION"
    QUESTION = "QUESTION"


class NumberKind(str, Enum):
    PRICE = "PRICE"
    DOLLAR_AMOUNT = "DOLLAR_AMOUNT"
    PERCENTAGE = "PERCENTAGE"
    QUANTITY = "QUANTITY"
    DATE = "DATE"
    YEAR = "YEAR"
    DEADLINE = "DEADLINE"
    AGE = "AGE"


class VisualType(str, Enum):
    LITERAL = "LITERAL"
    PROCESS = "PROCESS"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    DATA = "DATA"
    EVIDENCE = "EVIDENCE"
    ABSTRACT = "ABSTRACT"
    COMPARISON = "COMPARISON"
    OBJECT = "OBJECT"
    EVENT = "EVENT"


class SceneStatus(str, Enum):
    PENDING = "PENDING"  # segmented, not yet enriched
    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    FAILED = "FAILED"


class Origin(str, Enum):
    AI = "AI"
    USER = "USER"


@dataclass
class Entity:
    text: str
    type: EntityType
    canonical: str = ""
    mentions: int = 1
    word_ids: list[str] = field(default_factory=list)  # transcript words that mention it (provenance)


@dataclass
class Claim:
    claim_id: str
    text: str
    type: ClaimType
    sentence_id: str
    requires_evidence: bool = True
    domain: str | None = None  # e.g. "market", "legal"
    # Phase 2 never researches or invents evidence:
    evidence_status: str = "NOT_RESEARCHED"
    evidence: list[str] = field(default_factory=list)


@dataclass
class NumericMention:
    text: str
    kind: NumberKind
    value: float | None = None
    sentence_id: str = ""
    spoken: bool = False  # True when it was spoken as words ("one hundred dollars")
    word_ids: list[str] = field(default_factory=list)


@dataclass
class VisualIntent:
    scene_id: str
    type: VisualType
    primary_subject: str = ""
    secondary_subject: str = ""
    action: str = ""
    context: str = ""
    preferred_visuals: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    secondary_types: list[VisualType] = field(default_factory=list)
    type_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    inherited_from: str | None = None  # scene id whose context this intent borrowed
    author: Origin = Origin.AI

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VisualIntent":
        return from_plain(cls, d)


@dataclass
class ListItem:
    """One element of a spoken enumeration ("solar panels, electronics and electric vehicles")."""

    text: str
    first_word_id: str  # first word of the item itself
    lead_word_id: str  # word where a visual cut for this item would start (e.g. the preceding "and")


@dataclass
class SentenceAnalysis:
    sentence_id: str
    entities: list[Entity] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    numbers: list[NumericMention] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)  # content stems, for cohesion and topic work
    cues: list[str] = field(default_factory=list)  # discourse/semantic flags (see analysis/lexicon.py)
    type_scores: dict[str, float] = field(default_factory=dict)
    word_count: int = 0
    list_items: list[ListItem] = field(default_factory=list)  # present when the sentence enumerates 3+ visual things
    text: str = ""  # analysis text (script casing transferred where aligned)


@dataclass
class Scene:
    id: str
    label: str  # human label: "14", or "14A"/"14B" after a manual split
    start: float
    end: float
    narration: str = ""  # actual spoken words in [start, end)
    sentence_ids: list[str] = field(default_factory=list)
    topic: str = ""
    summary: str = ""
    importance: float = 0.0
    segmentation_confidence: float = 0.0
    status: SceneStatus = SceneStatus.PENDING
    origin: Origin = Origin.AI
    user_edited_fields: list[str] = field(default_factory=list)  # AI never overwrites these
    boundaries_locked: bool = False  # user chose these boundaries
    entities: list[Entity] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    numbers: list[NumericMention] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)  # why the AI cut/merged here
    script_text: str = ""  # script words aligned to this span (display only)
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_user_touched(self) -> bool:
        return (
            self.origin is Origin.USER or bool(self.user_edited_fields)
            or self.boundaries_locked or self.status is SceneStatus.APPROVED
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scene":
        return from_plain(cls, d)


@dataclass
class SceneAnalysisState:
    """Pipeline bookkeeping + cached per-sentence analysis. Scenes themselves live in ``Project.scenes``."""

    status: str = "NONE"  # NONE | COMPLETE | PARTIAL (stopped on a failure)
    input_hash: str = ""  # transcript id + script hash + analyzer + params; same hash => up to date
    sentence_key: str = ""  # transcript id + script hash + analyzer: when equal, per-sentence results are reused
    transcript_id: str = ""
    analyzer: str = ""
    analyzer_version: str = ""
    overall_topic: str = ""
    sentence_analysis: dict[str, SentenceAnalysis] = field(default_factory=dict)
    scene_counter: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    failed_scene_id: str | None = None
    failed_error: str | None = None
    created_at: str = ""
