"""Data contracts of the editorial intelligence layer. Plain dataclasses / str-enums, JSON-safe through ``to_plain`` / ``from_plain`` (additive: unknown keys are ignored, missing keys default)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------------------------------ vocabularies
class NarrativeRole(str, Enum):
    HOOK = "HOOK"
    INTRODUCTION = "INTRODUCTION"
    BACKGROUND = "BACKGROUND"
    CONTEXT = "CONTEXT"
    PROBLEM = "PROBLEM"
    QUESTION = "QUESTION"
    INVESTIGATION = "INVESTIGATION"
    EXPLANATION = "EXPLANATION"
    EVIDENCE = "EVIDENCE"
    DISCOVERY = "DISCOVERY"
    COMPARISON = "COMPARISON"
    CONSEQUENCE = "CONSEQUENCE"
    TURNING_POINT = "TURNING_POINT"
    REVELATION = "REVELATION"
    CONCLUSION = "CONCLUSION"
    CALL_TO_ACTION = "CALL_TO_ACTION"
    TRANSITION = "TRANSITION"


class RelationType(str, Enum):
    CONTINUATION = "CONTINUATION"
    TOPIC_TRANSITION = "TOPIC_TRANSITION"
    CAUSE_EFFECT = "CAUSE_EFFECT"
    QUESTION_ANSWER = "QUESTION_ANSWER"
    CLAIM_EVIDENCE = "CLAIM_EVIDENCE"
    BEFORE_AFTER = "BEFORE_AFTER"
    COMPARISON = "COMPARISON"
    CONTRAST = "CONTRAST"
    SUPPORTING_EXAMPLE = "SUPPORTING_EXAMPLE"
    REPEATED_REFERENCE = "REPEATED_REFERENCE"
    CALLBACK = "CALLBACK"


class VisualRole(str, Enum):
    ESTABLISHING = "ESTABLISHING"
    EXPLANATORY = "EXPLANATORY"
    EVIDENCE = "EVIDENCE"
    ILLUSTRATIVE = "ILLUSTRATIVE"
    COMPARATIVE = "COMPARATIVE"
    EMOTIONAL = "EMOTIONAL"
    TRANSITIONAL = "TRANSITIONAL"
    REVEALING = "REVEALING"
    CONTEXTUAL = "CONTEXTUAL"


class EvidenceGrade(str, Enum):
    DIRECT = "DIRECT"  # the visual is the thing the claim is about (the report itself, the chart of those numbers)
    CONTEXTUAL = "CONTEXTUAL"  # the right place / institution / subject, but not proof of the claim
    ILLUSTRATIVE = "ILLUSTRATIVE"  # drawn, generated or staged to show the idea
    DECORATIVE = "DECORATIVE"  # generic footage with no stated relation to the claim
    UNVERIFIED = "UNVERIFIED"  # it presents itself as evidence but nothing in the project establishes that it is


class Category(str, Enum):
    VISUAL_RELEVANCE = "VISUAL_RELEVANCE"
    VISUAL_CONTINUITY = "VISUAL_CONTINUITY"
    VISUAL_DIVERSITY = "VISUAL_DIVERSITY"
    NARRATIVE_FLOW = "NARRATIVE_FLOW"
    PACING = "PACING"
    EMPHASIS = "EMPHASIS"
    EVIDENCE_ALIGNMENT = "EVIDENCE_ALIGNMENT"
    STYLE_CONSISTENCY = "STYLE_CONSISTENCY"
    AUDIO_VISUAL_ALIGNMENT = "AUDIO_VISUAL_ALIGNMENT"
    CAPTION_HIERARCHY = "CAPTION_HIERARCHY"
    REFERENCE_STYLE_CONFLICT = "REFERENCE_STYLE_CONFLICT"


class Priority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    OPTIONAL = "OPTIONAL"


PRIORITY_ORDER = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2, Priority.OPTIONAL: 3}


class RecStatus(str, Enum):
    PENDING = "PENDING"  # shown to the user, nothing done
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"  # the user chose Ignore: it is not offered again unless the situation changes materially
    OUTDATED = "OUTDATED"  # the project changed under it: re-run the analysis before acting on it
    INVALID = "INVALID"  # failed validation: kept only so the reason can be shown
    FAILED = "FAILED"  # an apply attempt failed and was rolled back
    BLOCKED = "BLOCKED"  # a manual override / lock protects the target: reviewable, never auto-applied


class ActionKind(str, Enum):
    REPLACE_VISUAL = "REPLACE_VISUAL"
    SET_SHOT_DURATION = "SET_SHOT_DURATION"
    MOVE_CLIP_START = "MOVE_CLIP_START"
    SET_TRANSITION = "SET_TRANSITION"
    REMOVE_TRANSITION = "REMOVE_TRANSITION"
    ADD_PUNCH_IN = "ADD_PUNCH_IN"
    REMOVE_MOTION = "REMOVE_MOTION"
    CAPTION_EMPHASIS = "CAPTION_EMPHASIS"
    ADD_TEXT_EMPHASIS = "ADD_TEXT_EMPHASIS"
    RELABEL_ILLUSTRATIVE = "RELABEL_ILLUSTRATIVE"
    # not edits: routes to the existing services (the user decides; nothing expensive starts by itself)
    SEARCH_AGAIN = "SEARCH_AGAIN"
    EXPAND_SOURCES = "EXPAND_SOURCES"
    GENERATE_AI_VISUAL = "GENERATE_AI_VISUAL"
    USE_EXISTING_EVIDENCE = "USE_EXISTING_EVIDENCE"
    MANUAL_SELECTION = "MANUAL_SELECTION"
    KEEP_CURRENT = "KEEP_CURRENT"
    REVIEW_ONLY = "REVIEW_ONLY"  # a structural / narrative observation: nothing to apply


EDIT_ACTIONS = frozenset({ActionKind.REPLACE_VISUAL, ActionKind.SET_SHOT_DURATION, ActionKind.MOVE_CLIP_START, ActionKind.SET_TRANSITION, ActionKind.REMOVE_TRANSITION, ActionKind.ADD_PUNCH_IN,
                          ActionKind.REMOVE_MOTION, ActionKind.CAPTION_EMPHASIS, ActionKind.ADD_TEXT_EMPHASIS, ActionKind.RELABEL_ILLUSTRATIVE})


class AnalysisMode(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    CREATIVE = "CREATIVE"


class ContentType(str, Enum):
    AUTO = "AUTO"
    DOCUMENTARY = "DOCUMENTARY"  # documentary / crime / investigation
    FINANCE_EDUCATIONAL = "FINANCE_EDUCATIONAL"
    NEWS = "NEWS"
    EXPLAINER = "EXPLAINER"
    GENERAL = "GENERAL"


class Basis(str, Enum):
    COMPUTED = "COMPUTED"  # a measurement (counts, durations, hashes, shares)
    JUDGMENT = "JUDGMENT"  # an editorial opinion (rule-based or AI): carries a confidence and evidence references


class OverrideType(str, Enum):
    SOURCE_OVERRIDE = "SOURCE_OVERRIDE"
    VISUAL_OVERRIDE = "VISUAL_OVERRIDE"
    TIMING_OVERRIDE = "TIMING_OVERRIDE"
    MOTION_OVERRIDE = "MOTION_OVERRIDE"
    TRANSITION_OVERRIDE = "TRANSITION_OVERRIDE"
    CAPTION_OVERRIDE = "CAPTION_OVERRIDE"
    AUDIO_OVERRIDE = "AUDIO_OVERRIDE"
    STYLE_OVERRIDE = "STYLE_OVERRIDE"


STAGE_IDS = ("context", "narrative", "story_arc", "relationships", "sequence", "continuity", "diversity", "evidence", "pacing", "emphasis", "style", "recommendations")
STAGE_LABELS = {
    "context": "Project context", "narrative": "Narrative classification", "story_arc": "Story arc", "relationships": "Scene relationships", "sequence": "Visual sequence",
    "continuity": "Visual continuity", "diversity": "Diversity and source balance", "evidence": "Evidence alignment", "pacing": "Pacing", "emphasis": "Emphasis and audio-visual alignment",
    "style": "Style consistency", "recommendations": "Recommendations",
}


# ------------------------------------------------------------------------------------------------------------------------ story understanding
@dataclass
class RoleScore:
    role: NarrativeRole
    confidence: float  # 0..100


@dataclass
class SceneRole:
    scene_id: str
    roles: list[RoleScore] = field(default_factory=list)  # strongest first; a scene may have several, or none when nothing can be said
    evidence_terms: list[str] = field(default_factory=list)  # the words / cues that led here (what a reviewer can check)
    uncertain: bool = False
    basis: str = "RULES"  # RULES | AI
    note: str = ""

    @property
    def primary(self) -> NarrativeRole | None:
        return self.roles[0].role if self.roles else None


@dataclass
class StorySection:
    section_id: str
    title: str
    scene_ids: list[str]
    start: float = 0.0
    end: float = 0.0
    role: str = ""  # the dominant narrative role of the section
    confidence: float = 0.0
    note: str = ""


@dataclass
class KeyMoment:
    scene_id: str
    kind: str  # OPENING | CLIMAX | TURNING_POINT | KEY_EVIDENCE | RESOLUTION | ENDING
    time: float = 0.0
    confidence: float = 0.0
    note: str = ""


@dataclass
class StoryStructure:
    model: str = ""  # QUESTION_EXPLANATION | INVESTIGATIVE | PROBLEM_SOLUTION | CHRONOLOGICAL | LIST | UNDETERMINED
    confidence: float = 0.0
    topic: str = ""
    audience_takeaway: str = ""
    sections: list[StorySection] = field(default_factory=list)
    key_moments: list[KeyMoment] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)  # questions raised and not answered later (scene ids in the note)
    content_type: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class Relation:
    from_scene: str
    to_scene: str
    type: RelationType
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    distance: int = 1  # scenes between them (1 = neighbours)


@dataclass
class VisualStrategy:
    scene_id: str
    kind: VisualRole = VisualRole.ILLUSTRATIVE
    secondary: list[VisualRole] = field(default_factory=list)
    primary_subject: str = ""
    secondary_subject: str = ""
    needs_evidence: bool = False
    continue_previous: bool = False  # the same visual may carry on from the previous scene
    several_visuals: bool = False  # distinct ideas inside the scene: more than one visual is justified
    confidence: float = 0.0
    reason: str = ""


# ------------------------------------------------------------------------------------------------------------------------ findings, recommendations, actions
@dataclass
class TargetRef:
    scene_id: str = ""
    clip_id: str = ""
    slot: str = ""
    asset_id: str = ""
    candidate_id: str = ""
    track_id: str = ""


@dataclass
class ActionSpec:
    kind: ActionKind = ActionKind.REVIEW_ONLY
    params: dict[str, Any] = field(default_factory=dict)  # kind-specific, JSON-safe (e.g. {"duration": 4.2}, {"candidate_id": "..."}, {"words": [...]})
    targets: list[TargetRef] = field(default_factory=list)
    summary: str = ""  # one plain sentence: what applying does


@dataclass
class Finding:
    """What an analyzer found. The recommendation engine turns findings into recommendations; analyzers never touch the timeline."""

    finding_id: str
    stage: str
    category: Category
    scene_ids: list[str] = field(default_factory=list)
    timeline_item_ids: list[str] = field(default_factory=list)
    title: str = ""
    detail: str = ""  # the reason, one or two concrete sentences about THIS project (never generic)
    confidence: float = 0.0  # 0..100: how sure the analyzer is that this is a real issue
    impact: float = 50.0  # 0..100: how much it matters if it is (not the same thing as confidence)
    basis: Basis = Basis.JUDGMENT
    evidence_refs: list[str] = field(default_factory=list)  # scene ids, asset ids, word ids, QC issue ids the finding rests on
    action: ActionSpec | None = None
    options: list[ActionKind] = field(default_factory=list)  # fall-back routes when no valid replacement exists
    expected_benefit: str = ""
    time_range: list[float] = field(default_factory=list)  # [start, end] seconds of the affected region
    related_qc_issue_ids: list[str] = field(default_factory=list)
    serious: bool = False  # an evidence-integrity style problem: never averaged away by a good overall score
    intentional: bool = False  # the analyzer judged it to be deliberate (kept for the report, not recommended against)
    key: str = ""  # stable identity (stage + target + kind) so a re-run recognises the same finding and keeps its status


@dataclass
class ValidationCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ValidationResult:
    valid: bool = False
    checks: list[ValidationCheck] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # why it is not valid (plain language)
    options: list[ActionKind] = field(default_factory=list)  # what the user can do instead
    blocked_by_override: list[str] = field(default_factory=list)  # OverrideType values that protect the target
    conflicts_with: list[str] = field(default_factory=list)  # other pending recommendation ids
    plan_notes: list[str] = field(default_factory=list)  # what the dry run said (adjustments the planner would make)
    validated_at: str = ""


@dataclass
class Recommendation:
    recommendation_id: str
    category: Category
    priority: Priority
    scene_ids: list[str] = field(default_factory=list)
    affected_timeline_item_ids: list[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    reason: str = ""
    confidence: float = 0.0  # 0..100, NOT the priority
    expected_benefit: str = ""
    suggested_action: ActionSpec = field(default_factory=ActionSpec)
    affected_duration: float = 0.0  # seconds of the video the change touches
    requires_user_approval: bool = True
    status: RecStatus = RecStatus.PENDING
    created_at: str = field(default_factory=now_iso)
    # ---- bookkeeping
    key: str = ""  # the finding key: identity across analysis runs
    stage: str = ""
    basis: Basis = Basis.JUDGMENT
    time_range: list[float] = field(default_factory=list)
    options: list[ActionKind] = field(default_factory=list)
    related_qc_issue_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    serious: bool = False
    mode: str = ""  # the analysis mode it was produced under
    validation: ValidationResult | None = None
    blocked_by_override: list[str] = field(default_factory=list)
    status_note: str = ""
    status_changed_at: str = ""
    based_on: str = ""  # the analysis run it came from
    inputs_hash: str = ""  # hash of the scenes it rests on: a changed hash outdates it

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recommendation":
        return from_plain(cls, d)


# ------------------------------------------------------------------------------------------------------------------------ overrides and preferences
@dataclass
class EditorialOverride:
    """An explicit decision about the editorial layer (everything else the user did is read from the existing ownership marks by the OverrideResolver)."""

    override_id: str
    type: OverrideType
    scene_id: str = ""
    clip_id: str = ""
    reason: str = ""  # KEEP (a recommendation the user declined for good) | RESET (the user gave a manual override back to the AI)
    action: str = "KEEP"  # KEEP | RESET
    recommendation_key: str = ""
    created_at: str = field(default_factory=now_iso)
    note: str = ""


@dataclass
class PreferenceSignal:
    kind: str  # REPLACED_VISUAL | KEPT_LONG_SHOT | REJECTED_ZOOM | REJECTED_RECOMMENDATION | APPLIED_RECOMMENDATION | CHANGED_CAPTION_STYLE | LOCKED_PROPERTY | CHOSE_SOURCE | ...
    scene_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    explicit: bool = False  # the user stated it (a setting, an explicit "always") vs inferred from behaviour
    at: str = field(default_factory=now_iso)
    category: str = ""  # which preference it is evidence for (e.g. "preferred_visual_sources")
    value: str = ""


@dataclass
class UserEditorialPreferences:
    scope: str = "PROJECT"  # PROJECT | GLOBAL
    learning_enabled: bool = True
    excluded_from_learning: bool = False  # this project does not teach (or is not taught by) the global preferences
    preferred_visual_sources: dict[str, float] = field(default_factory=dict)  # SourceType -> weight -1..1
    rejected_visual_patterns: list[str] = field(default_factory=list)
    preferred_shot_duration: float | None = None
    preferred_motion_intensity: float | None = None  # 0..1
    transition_preferences: dict[str, float] = field(default_factory=dict)  # transition type / "frequency" -> weight
    caption_preferences: dict[str, Any] = field(default_factory=dict)
    graphic_preferences: dict[str, Any] = field(default_factory=dict)
    audio_preferences: dict[str, Any] = field(default_factory=dict)
    visual_diversity_preferences: dict[str, Any] = field(default_factory=dict)
    reference_style_strength: float | None = None
    edited_by_user: list[str] = field(default_factory=list)  # fields the user typed in (never overwritten by learning)
    confidence: dict[str, float] = field(default_factory=dict)  # per field 0..1
    signals: list[PreferenceSignal] = field(default_factory=list)  # a bounded local log (the evidence the learned values came from)
    last_updated: str = ""


# ------------------------------------------------------------------------------------------------------------------------ settings, history, scorecard
@dataclass
class EditorialSettings:
    mode: AnalysisMode = AnalysisMode.BALANCED
    content_type: ContentType = ContentType.AUTO
    stages: list[str] = field(default_factory=lambda: list(STAGE_IDS))
    use_reference_style: bool = True
    min_confidence: float = 0.0  # 0 = the mode's own threshold
    max_recommendations: int = 150
    provider: str = "local"  # local | api | unavailable
    ai_enabled: bool = False  # the local deterministic analysis is always used; AI judgement only when switched on
    window: int = 2  # scenes around a changed scene that are analysed again
    auto_analyze_on_open: bool = False


@dataclass
class StageRecord:
    stage: str
    status: str = "PENDING"  # PENDING | RUNNING | DONE | PARTIAL | FAILED | SKIPPED | CANCELED | REUSED
    started_at: str = ""
    finished_at: str = ""
    seconds: float = 0.0
    scenes_analysed: int = 0
    scenes_reused: int = 0
    findings: int = 0
    message: str = ""
    error: str = ""
    input_hash: str = ""


@dataclass
class ScoreItem:
    dimension: str  # Narrative Coherence | Visual Sequence Quality | ...
    score: float | None = None  # 0..100, None = not measurable for this project
    basis: Basis = Basis.JUDGMENT
    confidence: float = 0.0
    measured: dict[str, float] = field(default_factory=dict)  # the computed numbers behind it (counts, shares, ...)
    evidence_refs: list[str] = field(default_factory=list)
    note: str = ""
    serious_warnings: int = 0


@dataclass
class Scorecard:
    items: list[ScoreItem] = field(default_factory=list)
    overall: float | None = None
    overall_note: str = ""  # explains a cap (a serious evidence warning cannot be hidden by a high average)
    capped_by: list[str] = field(default_factory=list)


@dataclass
class AnalysisRun:
    run_id: str
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    status: str = "RUNNING"  # RUNNING | COMPLETE | PARTIAL | FAILED | CANCELED
    mode: str = ""
    trigger: str = "manual"  # manual | incremental | retry
    scope: dict[str, Any] = field(default_factory=dict)  # {"scenes": [...], "stages": [...], "all": bool}
    stages: list[StageRecord] = field(default_factory=list)
    recommendations: int = 0
    scorecard: Scorecard | None = None
    content_hash: str = ""
    message: str = ""
    provider: str = "local"
    log: list[str] = field(default_factory=list)  # concise lines (what ran, what was reused, what failed)


@dataclass
class EditorialState:
    """Everything the project stores for this phase (``Project.editorial``). Large payloads (raw provider responses) live in ``cache/editorial/``, not here."""

    editorial_settings: EditorialSettings = field(default_factory=EditorialSettings)
    narrative_structure: StoryStructure | None = None
    scene_roles: dict[str, SceneRole] = field(default_factory=dict)  # part of the narrative classification
    story_sections: list[StorySection] = field(default_factory=list)
    scene_relationships: list[Relation] = field(default_factory=list)
    visual_strategy: dict[str, VisualStrategy] = field(default_factory=dict)
    visual_sequence_analysis: dict[str, Any] = field(default_factory=dict)
    visual_consistency_results: dict[str, Any] = field(default_factory=dict)
    visual_diversity_results: dict[str, Any] = field(default_factory=dict)
    pacing_recommendations: dict[str, Any] = field(default_factory=dict)
    emphasis_results: dict[str, Any] = field(default_factory=dict)
    evidence_alignment_results: dict[str, Any] = field(default_factory=dict)
    style_consistency_results: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    editorial_recommendations: list[Recommendation] = field(default_factory=list)
    editorial_overrides: list[EditorialOverride] = field(default_factory=list)
    user_editorial_preferences: UserEditorialPreferences = field(default_factory=UserEditorialPreferences)
    editorial_analysis_history: list[AnalysisRun] = field(default_factory=list)
    scorecard: Scorecard | None = None
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)  # stage id -> its merged result {"scenes": {scene_id: ...}, "global": {...}}: the canonical store; the named fields above are typed views of it
    stage_cache: dict[str, Any] = field(default_factory=dict)  # stage -> {"gsig": project-wide input signature, "scene_hashes": {scene_id: hash}}: what a re-run compares against
    project_hash: str = ""  # the content hash of the last complete analysis (staleness)

    @property
    def is_empty(self) -> bool:
        return (not self.editorial_recommendations and not self.editorial_analysis_history and not self.editorial_overrides and self.narrative_structure is None
                and not self.scene_roles and not self.findings and not self.sections and self.editorial_settings == EditorialSettings()
                and self.user_editorial_preferences == UserEditorialPreferences())

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditorialState":
        return from_plain(cls, d) if d else cls()


# ------------------------------------------------------------------------------------------------------------------------ analyzer contract
@dataclass
class StageOutput:
    """What one analyzer returns. ``section`` is JSON-safe and stored under the stage's own name; per-scene stages put their results under ``section["scenes"][scene_id]`` so a re-run
    for some scenes can be merged by the engine, anything else under ``section["global"]``."""

    stage: str
    section: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    scene_hashes: dict[str, str] = field(default_factory=dict)  # the input hash of every scene analysed (empty for global stages)
    status: str = "DONE"  # DONE | PARTIAL | FAILED
    message: str = ""
    error: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    scenes_analysed: list[str] = field(default_factory=list)


def finding_key(stage: str, kind: str, *parts: object) -> str:
    """A stable identity for a finding: the same problem in the same place keeps its key across runs, so a rejected recommendation is not offered again."""
    import hashlib

    raw = "|".join([stage, kind, *[str(p) for p in parts]])
    return f"{stage}:{kind}:" + hashlib.sha1(raw.encode()).hexdigest()[:10]
