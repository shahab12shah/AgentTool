"""The checker contract: every analyser is a ``BaseChecker`` that reads a QCContext and returns QCIssues.

A checker never mutates the project, never raises for a *finding* (that is an issue), and may raise for a *failure* (a bug, an unreadable file): the engine
records the failure for that checker only and carries on with the rest (partial-failure recovery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCFixSpec, QCIssue, new_issue_id
from app.qc.severity import Severity
from app.timeline.clip import Clip
from app.timeline.track import Track


@dataclass
class CheckerOutput:
    issues: list[QCIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)  # measurements worth showing (pacing curve, loudness ...): never required by anyone
    notes: list[str] = field(default_factory=list)  # one-line log entries ("checked 84 scenes", "skipped 3 videos: scan budget")
    complete: bool = True  # False when the checker deliberately analysed less than asked (budget) - said in notes


class BaseChecker:
    id: str = "base"
    label: str = "Base"
    categories: tuple[QCCategory, ...] = ()
    domains: tuple[str, ...] = ("timeline",)  # which parts of the project the result depends on (ctx.domain_hash names)
    settings_sections: tuple[str, ...] = ()  # QCSettings sections it reads (the cache key covers exactly these)
    scene_local: bool = False  # True: it can analyse single scenes (ctx.scene_filter) and its scene-scoped issues can be reused per scene
    expensive: bool = False  # skipped when preflight found broken project integrity (no expensive analysis of a broken project)
    uses_shared: bool = False  # reads the other checkers' results (the editorial review): its cache key includes theirs
    weight: float = 1.0  # relative share of the run's progress bar

    # ------------------------------------------------------------------ cache keys
    def input_hash(self, ctx: QCContext) -> str:
        """Everything this checker's answer depends on. Equal hash => the previous result is still valid."""
        return sha([ctx.domain_hash(d) for d in self.domains], ctx.settings.subset_hash(*self.settings_sections), self.id, self.version, ctx.shared_signature() if self.uses_shared else "",
                   ctx.basis_hash())

    version: str = "1"  # bump when the checker's rules change (invalidates cached results)

    def scene_input_hash(self, ctx: QCContext, scene_id: str) -> str:
        return sha(ctx.scene_signature(scene_id), ctx.settings.subset_hash(*self.settings_sections), self.id, self.version, ctx.basis_hash(), ctx.global_signature(self.domains))

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:  # pragma: no cover - overridden
        raise NotImplementedError

    # ------------------------------------------------------------------ issue factory
    def issue(self, code: str, category: QCCategory, severity: Severity, title: str, *, description: str = "", scene_id: str | None = None, clip: Clip | None = None,
              track: Track | None = None, start: float | None = None, end: float | None = None, confidence: float = 100.0, source: str | None = None,
              affected: list[str] | None = None, why: str = "", current: str = "", recommended: str = "", suggested_fix: str = "", fix: QCFixSpec | None = None,
              fix_blocked: str = "", viewer_impact: float = 0.5, importance: float | None = None, metrics: dict[str, Any] | None = None, signature: str = "",
              group_hint: str = "", locked: bool = False, ctx: QCContext | None = None) -> QCIssue:
        """Build a fully-populated issue. Auto-fix flags are derived from ``fix`` and the protection of the element, so no checker can offer a fix for something the user owns."""
        protected_reason = fix_blocked
        if ctx is not None and clip is not None and not protected_reason:
            prot, reason = ctx.is_protected(track, clip)
            if prot:
                protected_reason, locked = reason + ": auto-fix is disabled (the issue is still reported)", True
        if ctx is not None and scene_id and importance is None:
            s = ctx.scene(scene_id)
            importance = s.importance if s else 0.5
        has_fix = fix is not None and not protected_reason
        if has_fix and ctx is not None:
            perm = ctx.settings.permission(fix.kind)  # type: ignore[union-attr]
            if perm == "never":
                has_fix, protected_reason = False, "Disabled in QC settings"
        iss = QCIssue(
            issue_id=new_issue_id(), code=code, category=category, severity=severity, title=title, description=description, scene_id=scene_id or (clip.scene_id if clip and clip.scene_id else None),
            timeline_item_id=clip.id if clip is not None else None, track_id=(track.id if track is not None else (clip.track_id if clip is not None else None)), start_time=start, end_time=end,
            confidence=float(max(0.0, min(100.0, confidence))), detection_source=source or f"deterministic:{self.id}", affected_elements=list(affected or []), why_it_matters=why,
            current_value=current, recommended_value=recommended, suggested_fix=suggested_fix or (fix.summary if fix else ""), fix=fix if (fix is not None) else None,
            auto_fix_available=bool(has_fix), auto_fix_safe=bool(has_fix and fix is not None and fix.safe and ctx is not None and ctx.settings.permission(fix.kind) == "auto"),
            fix_blocked_reason=protected_reason, checker=self.id, viewer_impact=max(0.0, min(1.0, viewer_impact)), scene_importance=importance if importance is not None else 0.5,
            metrics=dict(metrics or {}), group_hint=group_hint, locked=locked)
        if clip is not None and start is None:
            iss.start_time, iss.end_time = clip.timeline_start, clip.timeline_end
        iss.fingerprint = iss.make_fingerprint(signature)
        return iss
