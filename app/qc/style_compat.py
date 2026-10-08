"""Reference-style compatibility (spec 48): shared by the checkers that measure a style dimension (pacing, captions, motion, transitions).

A reference style is a *soft preference*. It never outranks the content: priority is user lock > narration/story > visual accuracy > readability > user settings >
reference style > AI defaults. So a deviation from the applied style is mentioned (NOTICE), explained away when the content needs it (INFO), and flagged as a
CONFLICT when the style itself would hurt readability. QC never recommends forcing the style, and never reports anything when no reference style is applied.

    dev = style_deviation(ctx, "pacing", measured_score)            # None unless a style is applied for this dimension and the gap exceeds the tolerance
    issue = style_issue(checker, ctx, "pacing", measured_score, explained_by="...", conflict="...")
"""

from __future__ import annotations

from dataclasses import dataclass

from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker
from app.qc.context import QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.severity import Severity

try:  # the reference engine is a separate subsystem: QC works without it
    from app.reference.style_model import DIMENSION_LABELS, DIMENSIONS
except ImportError:  # pragma: no cover
    DIMENSIONS, DIMENSION_LABELS = (), {}

CODES = {"pacing": "style.pacing_deviation", "caption_density": "style.caption_deviation", "motion_intensity": "style.motion_deviation", "transition_frequency": "style.transition_deviation"}
MIN_CONFIDENCE = 0.3  # a dimension the reference analysis was unsure about is not held against the project


@dataclass
class StyleDeviation:
    dimension: str
    label: str
    measured: float  # 0..100 on the Phase 7 scale
    target: float
    delta: float  # measured - target
    tolerance: float
    explained_by: str = ""
    conflict: str = ""

    @property
    def direction(self) -> str:
        return "higher" if self.delta > 0 else "lower"


def applied_dimensions(ctx: QCContext) -> list[str]:
    """The dimensions of the reference style that are actually in force (empty when no style is applied)."""
    p = ctx.project
    rs, prof = p.reference_settings, p.reference_style_profile
    if not ctx.settings.style.check or not rs.enabled or prof is None:
        return []
    return [d for d in rs.dimensions() if d in DIMENSIONS and prof.is_available(d) and prof.dimension_confidence(d) >= MIN_CONFIDENCE]


def target_score(ctx: QCContext, dimension: str) -> float | None:
    """What the project was asked to look like on this dimension: the user's customised target when there is one, else the reference's own score."""
    if dimension not in applied_dimensions(ctx):
        return None
    p = ctx.project
    adj = p.reference_settings.adjustments.get(dimension)
    return float(adj) if adj is not None else float(p.reference_style_profile.score(dimension))  # type: ignore[union-attr]


def style_deviation(ctx: QCContext, dimension: str, measured: float, *, explained_by: str = "", conflict: str = "") -> StyleDeviation | None:
    target = target_score(ctx, dimension)
    if target is None:
        return None
    delta = float(measured) - target
    if abs(delta) <= ctx.settings.style.tolerance and not conflict:
        return None  # a conflict is reported even when the project matches the style: the style itself is the cause
    return StyleDeviation(dimension, DIMENSION_LABELS.get(dimension, dimension), round(float(measured), 1), round(target, 1), round(delta, 1), ctx.settings.style.tolerance, explained_by, conflict)


def style_issue(checker: BaseChecker, ctx: QCContext, dimension: str, measured: float, *, explained_by: str = "", conflict: str = "", scene_id: str | None = None,
                start: float | None = None, end: float | None = None, viewer_impact: float | None = None) -> QCIssue | None:
    dev = style_deviation(ctx, dimension, measured, explained_by=explained_by, conflict=conflict)
    if dev is None:
        return None
    gap = f"{abs(dev.delta):.0f} points {dev.direction}; tolerance {dev.tolerance:.0f}" if abs(dev.delta) > dev.tolerance else "the project follows the style"
    numbers = f"{dev.label}: the project scores {dev.measured:.0f}/100, the applied reference style {dev.target:.0f}/100 ({gap})."
    sig = sha(dimension, round(dev.measured / 5), round(dev.target / 5), bool(conflict), bool(explained_by))
    if conflict:
        return checker.issue(
            "style.conflict", QCCategory.STYLE, Severity.WARNING, "The applied reference style conflicts with readability here", description=f"{numbers} {conflict}", scene_id=scene_id, start=start, end=end,
            why="Readability and the story come before a reference style, so the style is not being forced.", current=f"{dev.label} {dev.measured:.0f} vs style {dev.target:.0f}", recommended="readable first",
            suggested_fix="Keep the readable version; lower the reference strength or customise this dimension in the style settings.", fix=fx.navigate("settings.open", "Open the reference style settings", page="reference"),
            confidence=85.0, viewer_impact=0.5 if viewer_impact is None else viewer_impact, signature=sig, ctx=ctx, group_hint=_group(dimension))
    if explained_by:
        return checker.issue(
            CODES.get(dimension, "style.deviation"), QCCategory.STYLE, Severity.INFO, f"{dev.label} differs from the reference style (content-driven)",
            description=f"{numbers} It deviates from the reference style because {explained_by}.", scene_id=scene_id, start=start, end=end, why="The content decides; the reference style is a soft preference.",
            current=f"{dev.label} {dev.measured:.0f}", recommended=f"{dev.label} {dev.target:.0f} where the content allows", suggested_fix="Nothing to do.", viewer_impact=0.0, signature=sig, ctx=ctx,
            group_hint=_group(dimension))
    return checker.issue(
        CODES.get(dimension, "style.deviation"), QCCategory.STYLE, Severity.NOTICE, f"{dev.label} differs from the applied reference style", description=numbers, scene_id=scene_id, start=start, end=end,
        why="The edit does not follow the reference style you applied on this dimension.", current=f"{dev.label} {dev.measured:.0f}", recommended=f"{dev.label} {dev.target:.0f}",
        suggested_fix="Re-apply the style, or keep the edit if you prefer it (the style is a preference, not a rule).", fix=fx.navigate("settings.open", "Open the reference style settings", page="reference"),
        confidence=80.0, viewer_impact=0.2 if viewer_impact is None else viewer_impact, signature=sig, ctx=ctx, group_hint=_group(dimension))


def _group(dimension: str) -> str:
    return {"pacing": "pacing", "caption_density": "captions", "motion_intensity": "continuity", "transition_frequency": "continuity"}.get(dimension, "pacing")
