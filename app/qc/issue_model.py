"""The QC data model: categories, issues, fix recipes, checker status, run results, ignore/fix records.

Everything here is plain data (JSON-serialisable through ``core.serialization``). QC never stores anything that could replace the editable project:
issues *describe* problems and *recommend* fixes; a fix is only ever executed later, as an undoable command, by the QCFixEngine.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.qc.severity import Severity, priority_key


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_json(value: Any) -> Any:
    """Make a value safe for project.json: NaN / infinity (which json writes as invalid JSON) become None; nested containers are cleaned recursively; values json cannot
    write at all (numpy numbers, sets, paths, enums ...) become plain numbers / lists / strings, so one odd metric can never make the whole project unsavable."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((clean_json(v) for v in value), key=str)
    if isinstance(value, Enum):
        return clean_json(value.value)
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        try:
            return clean_json(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def new_issue_id() -> str:
    return f"qci_{uuid.uuid4().hex[:10]}"


def new_run_id() -> str:
    return f"qcr_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------------------------- categories
class QCCategory(str, Enum):
    """What an issue is about. (A *score group* — see ``SCORE_GROUPS`` — is what the dashboard totals up.)"""

    PREFLIGHT = "PREFLIGHT"
    TIMELINE = "TIMELINE"
    SCENE_COVERAGE = "SCENE_COVERAGE"
    SYNC = "SYNC"
    VISUAL_ACCURACY = "VISUAL_ACCURACY"
    VISUAL_REPETITION = "VISUAL_REPETITION"
    CONTINUITY = "CONTINUITY"
    PACING = "PACING"
    CUT_TIMING = "CUT_TIMING"
    CAPTION = "CAPTION"
    TEXT = "TEXT"
    MOTION = "MOTION"
    TRANSITION = "TRANSITION"
    AUDIO = "AUDIO"
    SILENCE = "SILENCE"
    MEDIA_QUALITY = "MEDIA_QUALITY"
    ASSET = "ASSET"
    FRAMES = "FRAMES"
    RENDER_READINESS = "RENDER_READINESS"
    EDITORIAL = "EDITORIAL"
    STYLE = "STYLE"
    FACT_REVIEW = "FACT_REVIEW"


CATEGORY_LABELS: dict[QCCategory, str] = {
    QCCategory.PREFLIGHT: "Preflight", QCCategory.TIMELINE: "Timeline", QCCategory.SCENE_COVERAGE: "Scene coverage", QCCategory.SYNC: "Narration sync",
    QCCategory.VISUAL_ACCURACY: "Visual accuracy", QCCategory.VISUAL_REPETITION: "Visual repetition", QCCategory.CONTINUITY: "Continuity", QCCategory.PACING: "Pacing",
    QCCategory.CUT_TIMING: "Cut timing", QCCategory.CAPTION: "Captions", QCCategory.TEXT: "Text & graphics", QCCategory.MOTION: "Motion", QCCategory.TRANSITION: "Transitions",
    QCCategory.AUDIO: "Audio", QCCategory.SILENCE: "Silence & pauses", QCCategory.MEDIA_QUALITY: "Media quality", QCCategory.ASSET: "Assets", QCCategory.FRAMES: "Black / frozen frames",
    QCCategory.RENDER_READINESS: "Render readiness", QCCategory.EDITORIAL: "Editorial review", QCCategory.STYLE: "Reference style", QCCategory.FACT_REVIEW: "Facts to review",
}

# The nine dashboard scores (spec section 27). ``overall`` is computed from the other eight.
SCORE_GROUPS = ("visual_accuracy", "sync", "pacing", "captions", "audio", "continuity", "timeline", "technical")
GROUP_LABELS = {"overall": "Overall Quality", "visual_accuracy": "Visual Accuracy", "sync": "Narration Sync", "pacing": "Pacing", "captions": "Caption Quality", "audio": "Audio Quality",
                "continuity": "Visual Continuity", "timeline": "Timeline Integrity", "technical": "Technical Readiness"}
CATEGORY_GROUP: dict[QCCategory, str] = {
    QCCategory.PREFLIGHT: "technical", QCCategory.ASSET: "technical", QCCategory.MEDIA_QUALITY: "technical", QCCategory.FRAMES: "technical", QCCategory.RENDER_READINESS: "technical",
    QCCategory.TIMELINE: "timeline", QCCategory.SCENE_COVERAGE: "timeline",
    QCCategory.SYNC: "sync",
    QCCategory.VISUAL_ACCURACY: "visual_accuracy", QCCategory.FACT_REVIEW: "visual_accuracy",
    QCCategory.VISUAL_REPETITION: "continuity", QCCategory.CONTINUITY: "continuity", QCCategory.MOTION: "continuity", QCCategory.TRANSITION: "continuity",
    QCCategory.PACING: "pacing", QCCategory.CUT_TIMING: "pacing", QCCategory.STYLE: "pacing",
    QCCategory.CAPTION: "captions", QCCategory.TEXT: "captions",
    QCCategory.AUDIO: "audio", QCCategory.SILENCE: "audio",
    QCCategory.EDITORIAL: "continuity",  # an editorial finding names its own group via ``QCIssue.group_hint``
}


class IssueStatus(str, Enum):
    OPEN = "OPEN"
    FIXED = "FIXED"  # a fix was applied (the next run confirms it)
    IGNORED = "IGNORED"  # the user chose to keep it
    OBSOLETE = "OBSOLETE"  # no longer detected in a newer analysis (kept only in history)


# ---------------------------------------------------------------------------------------------- fixes
class FixRoute(str, Enum):
    COMMAND = "COMMAND"  # executed by QCFixEngine as an undoable command
    NAVIGATE = "NAVIGATE"  # opens another page (Replace Visual, Open Scene, Relink …): the user decides there
    RESEARCH = "RESEARCH"  # search again for a visual (the user then approves the result)


@dataclass
class QCFixSpec:
    """A machine-readable recipe for the recommended fix. ``kind`` selects the handler in the QCFixEngine; ``params`` are its inputs (plain JSON)."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    safe: bool = False  # deterministic, tiny, semantics-preserving AND permitted "auto" in the settings: may be applied without confirmation
    intrinsic_safe: bool = False  # safe by construction (kind + size of the change), whatever the user's permission table says; ``safe`` = this and permission "auto"
    needs_confirmation: bool = True
    route: FixRoute = FixRoute.COMMAND
    summary: str = ""  # one line: "Move the caption 0.34 s earlier"


@dataclass
class QCIssue:
    issue_id: str
    code: str  # stable issue type, e.g. "caption.drift", "timeline.gap.unintended" — what "Ignore this type" and "Fix all similar" key on
    category: QCCategory
    severity: Severity
    title: str
    description: str = ""
    scene_id: str | None = None
    timeline_item_id: str | None = None
    track_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    confidence: float = 100.0  # 0..100; deterministic findings are 100, judgements are lower and said to be judgements
    detection_source: str = "deterministic"  # "deterministic:<checker>" | "ai:<provider>"
    affected_elements: list[str] = field(default_factory=list)  # human-readable names/ids of what is involved
    why_it_matters: str = ""
    current_value: str = ""
    recommended_value: str = ""
    suggested_fix: str = ""
    fix: QCFixSpec | None = None
    auto_fix_available: bool = False
    auto_fix_safe: bool = False
    fix_blocked_reason: str = ""  # e.g. "Caption style is locked by you: auto-fix is disabled"
    status: IssueStatus = IssueStatus.OPEN
    created_at: str = field(default_factory=now_iso)
    run_id: str = ""
    checker: str = ""
    fingerprint: str = ""  # stable identity across runs (type + place + content signature); what an "ignore" is matched on
    viewer_impact: float = 0.5  # 0..1
    scene_importance: float = 0.5  # 0..1
    ignored_by_user: bool = False
    ignore_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    original_research_score: float | None = None  # visual accuracy: the stored Phase 3 score, never rewritten
    current_qc_score: float | None = None  # visual accuracy: the in-context re-evaluation
    group_hint: str = ""  # overrides the score group of the category (editorial findings)
    locked: bool = False  # touches a user-locked / user-owned element: the issue is reported, never auto-fixed

    # ------------------------------------------------------------------ derived
    @property
    def duration(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return max(0.0, self.end_time - self.start_time)

    @property
    def score_group(self) -> str:
        return self.group_hint or CATEGORY_GROUP[self.category]

    @property
    def sort_key(self) -> tuple:
        return priority_key(self.severity, self.viewer_impact, self.duration, self.confidence, self.scene_importance) + (self.start_time or 0.0, self.issue_id)

    @property
    def active(self) -> bool:
        """Counts towards scores and the export gate (not ignored, not fixed, not obsolete)."""
        return self.status is IssueStatus.OPEN and not self.ignored_by_user

    def make_fingerprint(self, signature: str = "") -> str:
        """Stable across runs: the issue type, where it is, and ``signature`` (the content that makes it *this* issue: rounded values, text, asset ids)."""
        sig = signature or json.dumps([self.metrics.get("signature", ""), round(self.start_time or 0.0, 0), round(self.end_time or 0.0, 0)], default=str)
        raw = "|".join([self.code, self.scene_id or "", self.timeline_item_id or "", self.track_id or "", sig])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return clean_json(to_plain(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QCIssue":
        return from_plain(cls, d)

    def compact(self) -> dict[str, Any]:
        """The few fields kept for every past run (history comparison): enough to tell what was new / resolved, nothing more."""
        return {"fp": self.fingerprint, "code": self.code, "cat": self.category.value, "sev": self.severity.value, "scene": self.scene_id, "t": self.start_time, "title": self.title,
                "ignored": self.ignored_by_user, "status": self.status.value}


# ---------------------------------------------------------------------------------------------- user decisions
@dataclass
class IgnoreRecord:
    """The user chose to keep something QC flagged. Matched on ``fingerprint`` (this issue) or on ``code`` (this type, optionally within one scene)."""

    ignore_id: str
    scope: str = "issue"  # issue | type
    fingerprint: str = ""
    code: str = ""
    scene_id: str | None = None
    reason: str = ""
    created_at: str = field(default_factory=now_iso)
    title: str = ""

    def matches(self, issue: QCIssue) -> bool:
        if self.scope == "type":
            return issue.code == self.code and (self.scene_id is None or issue.scene_id == self.scene_id)
        return bool(self.fingerprint) and issue.fingerprint == self.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IgnoreRecord":
        return from_plain(cls, d)


@dataclass
class FixRecord:
    """One applied fix: what, to which issue, and enough to explain it afterwards (the undo itself is on the command stack)."""

    fix_id: str
    issue_id: str
    code: str
    kind: str
    fingerprint: str = ""
    scene_id: str | None = None
    summary: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    safe: bool = False
    confirmed_by_user: bool = False
    applied_at: str = field(default_factory=now_iso)
    run_id: str = ""
    checkpoint: str = ""
    reverted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FixRecord":
        return from_plain(cls, d)


# ---------------------------------------------------------------------------------------------- run results
class CheckerState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CACHED = "CACHED"  # inputs unchanged since the last run: its issues were reused
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"  # not run on purpose (preflight failed, category not selected, disabled in settings)
    CANCELED = "CANCELED"


@dataclass
class CheckerStatus:
    checker: str
    label: str = ""
    state: CheckerState = CheckerState.PENDING
    progress: float = 0.0
    message: str = ""
    error: str = ""
    issue_count: int = 0
    seconds: float = 0.0
    input_hash: str = ""  # what the result was computed from (cache key)
    scene_hashes: dict[str, str] = field(default_factory=dict)  # per scene inputs, for scene-level re-analysis
    reused_scenes: int = 0
    analyzed_scenes: int = 0

    @property
    def ok(self) -> bool:
        return self.state in (CheckerState.DONE, CheckerState.CACHED, CheckerState.SKIPPED)


@dataclass
class QCScores:
    overall: float = 100.0
    groups: dict[str, float] = field(default_factory=dict)  # SCORE_GROUPS -> 0..100
    status: str = "READY"  # READY | REVIEW | FIX_REQUIRED | BLOCKED
    status_label: str = ""  # "92/100 — Ready"
    counts: dict[str, int] = field(default_factory=dict)  # severity -> number of active issues
    export: str = "READY"  # READY | AVAILABLE | BLOCKED
    unavailable: list[str] = field(default_factory=list)  # groups whose checkers failed: shown as "—", not as 100

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QCScores":
        return from_plain(cls, d)


@dataclass
class QCRun:
    run_id: str
    number: int = 1  # QC Run #n (per project)
    created_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    trigger: str = "manual"  # manual | export | scene | category | retry | auto
    scope: dict[str, Any] = field(default_factory=dict)  # {"categories": [...], "scene_ids": [...]}
    project_version: str = ""  # application version + schema version the project had
    timeline_version: int = 0
    settings_version: str = ""
    content_hash: str = ""  # fingerprint of everything QC looks at (timeline, scenes, transcript, assets, audio, captions, settings): a run is *current* while it still matches
    state: str = "COMPLETED"  # COMPLETED | PARTIAL (a checker failed or the run was canceled) | CANCELED | FAILED
    scores: QCScores = field(default_factory=QCScores)
    issues: list[QCIssue] = field(default_factory=list)
    checkers: dict[str, CheckerStatus] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)  # per-checker measurements (pacing curve, loudness, ...): shown, never required
    log: list[str] = field(default_factory=list)
    seconds: float = 0.0
    fixes: list[str] = field(default_factory=list)  # fix ids applied after this run
    cache_hits: int = 0

    def failed_checkers(self) -> list[str]:
        """The checkers that did not complete (failed or canceled): their score groups are 'not analysed' and the export gate says so."""
        return [k for k, c in self.checkers.items() if c.state in (CheckerState.FAILED, CheckerState.CANCELED)]

    def active_issues(self) -> list[QCIssue]:
        return [i for i in self.issues if i.active]

    def record(self) -> dict[str, Any]:
        """The lightweight summary kept in ``project.qc_runs`` (no issue details)."""
        return clean_json({
            "run_id": self.run_id, "number": self.number, "created_at": self.created_at, "finished_at": self.finished_at, "trigger": self.trigger, "scope": self.scope,
            "project_version": self.project_version, "timeline_version": self.timeline_version, "settings_version": self.settings_version, "content_hash": self.content_hash, "state": self.state,
            "overall": round(self.scores.overall, 1), "groups": {k: round(v, 1) for k, v in self.scores.groups.items()}, "status": self.scores.status, "status_label": self.scores.status_label,
            "export": self.scores.export, "counts": dict(self.scores.counts), "failed": self.failed_checkers(), "seconds": round(self.seconds, 2), "cache_hits": self.cache_hits,
            "checkers": {k: {"state": c.state.value, "issues": c.issue_count, "error": c.error} for k, c in self.checkers.items()},
        })

    def history_entry(self, ignored: list[str], fixes: list[str]) -> dict[str, Any]:
        """The archive entry for ``project.qc_history``: scores, compact issues, the fixes and ignored fingerprints at the time."""
        return {
            "qc_run_id": self.run_id, "number": self.number, "timestamp": self.finished_at or self.created_at, "project_version": self.project_version,
            "timeline_version": self.timeline_version, "overall_score": round(self.scores.overall, 1), "category_scores": {k: round(v, 1) for k, v in self.scores.groups.items()},
            "status": self.scores.status, "state": self.state, "counts": dict(self.scores.counts), "issues": [i.compact() for i in self.issues], "fixes": list(fixes), "ignored_issues": list(ignored),
        }


def sorted_issues(issues: list[QCIssue]) -> list[QCIssue]:
    return sorted(issues, key=lambda i: i.sort_key)
