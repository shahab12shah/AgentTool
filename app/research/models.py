"""Research data model: brief, queries, normalised candidates, scores, ranking, sessions, assignments.

Everything here is plain data (JSON round-trippable through ``core.serialization``). Provider-specific
response formats never leak past the providers: the rest of the application only sees ``Candidate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.media.asset import SourceType


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class QueryType(str, Enum):
    LITERAL = "LITERAL"
    CONTEXT = "CONTEXT"
    PROCESS = "PROCESS"
    EVIDENCE = "EVIDENCE"
    ENTITY = "ENTITY"
    LOCATION = "LOCATION"
    DATA = "DATA"
    NEWS = "NEWS"
    DOCUMENT = "DOCUMENT"
    ALTERNATIVE = "ALTERNATIVE"


class EvidenceLevel(str, Enum):
    NONE = "NONE"
    POSSIBLE = "POSSIBLE"
    REQUIRED = "REQUIRED"


class ResearchStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RESEARCHING = "RESEARCHING"
    CANDIDATES_READY = "CANDIDATES_READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ERROR = "ERROR"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ScoreCategory(str, Enum):
    EXCELLENT = "EXCELLENT"  # 90-100
    GOOD = "GOOD"  # 80-89
    REVIEW = "REVIEW"  # 70-79
    WEAK = "WEAK"  # 60-69
    REJECT = "REJECT"  # < 60


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"  # e.g. an AI concept that has not been generated yet
    READY = "READY"
    REJECTED = "REJECTED"  # rejected by the user
    SELECTED = "SELECTED"
    ACQUIRED = "ACQUIRED"  # media is a project asset
    FAILED = "FAILED"  # acquisition failed


class EvidenceKind(str, Enum):
    DECORATIVE = "DECORATIVE"  # illustrates the narration
    EVIDENCE = "EVIDENCE"  # a real document/page/chart that supports a claim


class Acquisition(str, Enum):
    LOCAL = "LOCAL"  # a file we can copy
    DOWNLOAD = "DOWNLOAD"  # a direct media URL we can fetch
    CAPTURE = "CAPTURE"  # produced by capturing a page
    GENERATE = "GENERATE"  # produced by an AI image service on request
    REFERENCE_ONLY = "REFERENCE_ONLY"  # cannot be acquired by this application (e.g. YouTube); reference + segment only


# --------------------------------------------------------------------------- brief & queries
@dataclass
class SceneContext:
    scene_id: str = ""
    label: str = ""
    topic: str = ""
    primary_subject: str = ""
    narration: str = ""
    visual_type: str = ""


@dataclass
class ResearchBrief:
    scene_id: str
    topic: str = ""
    primary_subject: str = ""
    secondary_subject: str = ""
    action: str = ""
    context: str = ""
    visual_type: str = "LITERAL"
    secondary_types: list[str] = field(default_factory=list)
    narration: str = ""
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    entity_types: dict[str, str] = field(default_factory=dict)
    claims: list[str] = field(default_factory=list)
    claim_types: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.NONE
    preferred_sources: list[str] = field(default_factory=list)  # SourceType values, best first
    avoid: list[str] = field(default_factory=list)
    # context memory
    previous: SceneContext | None = None
    next: SceneContext | None = None
    video_topic: str = ""
    key_terms: list[str] = field(default_factory=list)  # the scene's own salient nouns ("paperwork"); never searched alone when anaphoric
    avoid_terms: list[str] = field(default_factory=list)  # words that signal a generic/wrong visual (e.g. "coin" for solar-silver)
    context_terms: list[str] = field(default_factory=list)  # resolved terms from neighbouring scenes / video topic
    anaphoric: bool = False  # the scene leans on earlier context ("That means...")
    # timing
    scene_start: float = 0.0
    scene_end: float = 0.0
    importance: float = 0.5
    project_width: int = 1920
    project_height: int = 1080

    @property
    def evidence_needed(self) -> bool:
        return self.evidence_level is EvidenceLevel.REQUIRED

    @property
    def scene_duration(self) -> float:
        return max(0.0, self.scene_end - self.scene_start)

    def to_dict(self) -> dict[str, Any]:
        d = to_plain(self)
        d["evidence_needed"] = self.evidence_needed
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchBrief":
        return from_plain(cls, d)


@dataclass
class ResearchQuery:
    query_id: str
    scene_id: str
    text: str
    type: QueryType
    purpose: str
    priority: int = 3  # 1 = most important
    source_preferences: list[str] = field(default_factory=list)  # SourceType values
    generation: int = 0  # 0 = first search; increments for every "search again"
    strategy: str = "initial"


# --------------------------------------------------------------------------- candidates
@dataclass
class LicenseInfo:
    """What the provider *states*. This application never verifies rights."""

    name: str | None = None
    url: str | None = None
    attribution: str | None = None
    status: str = "UNKNOWN"  # UNKNOWN | PROVIDER_STATED
    verified: bool = False  # always False unless a human verified it (not implemented)


@dataclass
class ClipSegment:
    start: float | None = None
    end: float | None = None
    basis: str = "UNKNOWN"  # UNKNOWN | CHAPTER | DEFAULT | FULL

    @property
    def length(self) -> float | None:
        return None if self.start is None or self.end is None else self.end - self.start


@dataclass
class Candidate:
    candidate_id: str
    scene_id: str
    source_type: SourceType
    kind: str  # "IMAGE" | "VIDEO"
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    duration: float | None = None  # seconds of the source media (None for images)
    width: int | None = None
    height: int | None = None
    provider: str = ""
    provider_id: str = ""  # the provider's own id (YouTube video id, Pexels id, file title...)
    source_reference: str = ""  # page URL / file path a human can open
    media_url: str = ""  # direct media URL when downloadable
    thumbnail_url: str = ""
    thumbnail_path: str = ""  # project-relative cached thumbnail
    local_path: str = ""  # absolute path when the media is already on this machine
    license: LicenseInfo = field(default_factory=LicenseInfo)
    segment: ClipSegment = field(default_factory=ClipSegment)
    evidence_kind: EvidenceKind = EvidenceKind.DECORATIVE
    acquisition: Acquisition = Acquisition.REFERENCE_ONLY
    status: CandidateStatus = CandidateStatus.READY
    query_ids: list[str] = field(default_factory=list)  # every query that returned it
    prompt: str = ""  # AI candidates: the prompt used / proposed
    fingerprint: str = ""  # perceptual hash of the thumbnail (near-duplicate detection)
    asset_id: str | None = None  # set once acquired as a project asset
    metadata: dict[str, Any] = field(default_factory=dict)  # provider extras (kept out of the common model)
    found_at: str = field(default_factory=now_iso)

    @property
    def text(self) -> str:
        return " ".join([self.title, self.description, " ".join(self.tags)]).strip()

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Candidate":
        return from_plain(cls, d)


@dataclass
class ScoreComponents:
    semantic: float = 0.0
    subject: float = 0.0
    context: float = 0.0
    action: float = 0.0
    timing: float = 0.0
    quality: float = 0.0
    source: float = 0.0


@dataclass
class CandidateScore:
    candidate_id: str
    scene_id: str
    overall: float
    components: ScoreComponents
    category: ScoreCategory
    confidence: Confidence
    reason: str = ""
    factors: list[str] = field(default_factory=list)  # concise decision factors ("✓ ..." / "✗ ...")
    basis: str = "METADATA"  # what was evaluated: METADATA | PROMPT (the pixels were not inspected)
    recommended_duration: float | None = None
    partial_coverage: bool = False  # the media is shorter than the scene
    crop_hint: str = ""
    orientation: str = ""
    focus: str = ""  # only when a provider/evaluator can actually tell (otherwise empty)
    evaluated_at: str = field(default_factory=now_iso)
    min_accuracy: float = 85.0  # the threshold in force when it was scored


@dataclass
class RankedEntry:
    candidate_id: str
    rank_score: float
    accuracy: float
    adjustments: dict[str, float] = field(default_factory=dict)  # named ranking adjustments (+/-)
    role: str = "ALTERNATIVE"  # BEST | ALTERNATIVE | OTHER


@dataclass
class SceneResearchState:
    scene_id: str
    status: ResearchStatus = ResearchStatus.NOT_STARTED
    message: str = ""
    best_id: str | None = None
    alternatives: list[str] = field(default_factory=list)
    ranked: list[RankedEntry] = field(default_factory=list)
    session_id: str | None = None
    generation: int = 0
    expanded_sources: list[str] = field(default_factory=list)
    brief_hash: str = ""  # hash of the scene content this research was done for (detects 'scene changed since')
    previous_status: str = ""  # status before RESEARCHING (restored on cancel)
    updated_at: str = field(default_factory=now_iso)


@dataclass
class ProviderReport:
    provider: str
    status: str  # SUCCESS | FAILED | SKIPPED | UNAVAILABLE
    candidates: int = 0
    queries: int = 0
    error: str = ""
    seconds: float = 0.0
    from_cache: int = 0


@dataclass
class ResearchSession:
    session_id: str
    scene_id: str
    strategy: str = "initial"
    fresh: bool = False
    expanded: bool = False
    brief: ResearchBrief | None = None
    query_ids: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    provider_reports: list[ProviderReport] = field(default_factory=list)
    status: str = "COMPLETE"  # COMPLETE | PARTIAL | FAILED
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""


@dataclass
class VisualAssignment:
    """The decision that the editing phase will consume: scene -> visual."""

    scene_id: str
    candidate_id: str | None = None
    asset_id: str | None = None  # project asset (None while not acquired / for reference-only sources)
    selected_by: str = "AI"  # "AI" | "USER"
    accuracy_score: float | None = None
    approved: bool = False
    skipped: bool = False
    evidence_kind: EvidenceKind = EvidenceKind.DECORATIVE
    acquisition: Acquisition = Acquisition.REFERENCE_ONLY
    segment: ClipSegment | None = None
    recommended_duration: float | None = None
    source_type: SourceType | None = None
    note: str = ""
    selected_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VisualAssignment":
        return from_plain(cls, d)


DEFAULT_WEIGHTS = {"semantic": 0.35, "subject": 0.20, "context": 0.15, "action": 0.10, "timing": 0.10, "quality": 0.05, "source": 0.05}


@dataclass
class ResearchSettings:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    alternatives: int = 5
    alt_min_score: float = 60.0  # alternatives below this are "Reject" quality and are not shown
    max_results_per_query: int = 8
    max_queries: int = 8
    max_candidates: int = 40
    cache_days: int = 7
    concurrency: int = 4
    default_clip_seconds: float = 5.0
    candidate_counter: int = 0
    query_counter: int = 0
    session_counter: int = 0
