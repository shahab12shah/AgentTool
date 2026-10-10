"""Aggregation: issues -> eight group scores, the overall score, the QC status and the export decision.

A weighted score never hides a blocking problem: the *status* and the *export decision* are decided by the issues themselves (any active CRITICAL blocks;
errors prevent "Ready"), and the label always shows the counts next to the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.qc.issue_model import GROUP_LABELS, SCORE_GROUPS, QCIssue, QCScores
from app.qc.settings import QCSettings
from app.qc.severity import BLOCKING, BlockLevel, Severity, blocks_export, parse_block_level

BASE_PENALTY: dict[Severity, float] = {Severity.CRITICAL: 40.0, Severity.ERROR: 12.0, Severity.WARNING: 4.0, Severity.NOTICE: 1.0, Severity.INFO: 0.0}
READY_AT, REVIEW_AT, FIX_AT = 90.0, 70.0, 40.0
STATUS_TEXT = {"READY": "Ready", "REVIEW": "Review Recommended", "FIX_REQUIRED": "Fix Required", "BLOCKED": "Export Blocked"}


def size_factor(scene_count: int) -> float:
    """Longer projects have proportionally more small findings: warnings/notices are divided by this (1 for <= 12 scenes, up to 4). Criticals and errors are never reduced below 1x."""
    return max(1.0, min(4.0, math.sqrt(max(scene_count, 1) / 12.0)))


def penalty(issue: QCIssue, total_duration: float, scene_count: int) -> float:
    if not issue.active:
        return 0.0
    base = BASE_PENALTY[issue.severity] * (max(0.0, min(100.0, issue.confidence)) / 100.0)
    impact = 0.5 + 0.5 * max(0.0, min(1.0, issue.viewer_impact))
    span = 1.0 + 0.5 * min(1.0, issue.duration / max(8.0, 0.1 * max(total_duration, 1.0)))
    p = base * impact * span
    if issue.severity in (Severity.WARNING, Severity.NOTICE):
        p /= size_factor(scene_count)
    return p


def group_scores(issues: list[QCIssue], total_duration: float, scene_count: int, unavailable: list[str] | None = None) -> dict[str, float]:
    unavailable = set(unavailable or [])
    totals = {g: 0.0 for g in SCORE_GROUPS}
    for i in issues:
        totals[i.score_group if i.score_group in totals else "timeline"] += penalty(i, total_duration, scene_count)
    return {g: (0.0 if g in unavailable else round(max(0.0, 100.0 - totals[g]), 1)) for g in SCORE_GROUPS}


def counts(issues: list[QCIssue]) -> dict[str, int]:
    out = {s.value: 0 for s in Severity}
    for i in issues:
        if i.active:
            out[i.severity.value] += 1
    return out


@dataclass
class ExportDecision:
    """The export gate. ``status``: READY (nothing open) | AVAILABLE (only non-blocking findings remain) | BLOCKED."""

    status: str = "READY"
    level: str = BlockLevel.CRITICAL_ERROR.value
    blocking_ids: list[str] = field(default_factory=list)
    critical: int = 0
    remaining: dict[str, int] = field(default_factory=dict)  # severity -> open issues not blocking
    overridable: bool = False  # the user may continue anyway (never true while a CRITICAL is open)
    message: str = ""

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"


def decide_export(issues: list[QCIssue], level: "BlockLevel | str", allow_override: bool = False, failed_checkers: list[str] | None = None) -> ExportDecision:
    level = parse_block_level(level)
    active = [i for i in issues if i.active]
    blocking = [i for i in active if blocks_export(i.severity, level)]
    crit = sum(1 for i in active if i.severity is Severity.CRITICAL)
    rest = {s.value: sum(1 for i in active if i.severity is s and not blocks_export(s, level)) for s in Severity}
    rest = {k: v for k, v in rest.items() if v}
    if blocking:
        overridable = allow_override and crit == 0
        n = len(blocking)
        what = "critical issue" if crit and crit == n else "blocking issue"
        msg = f"EXPORT BLOCKED — {n} {what}{'s' if n != 1 else ''} must be fixed" + (" (or explicitly accepted)" if overridable else "")
        return ExportDecision("BLOCKED", level.value, [i.issue_id for i in blocking], crit, rest, overridable, msg)
    failed = list(failed_checkers or [])
    if rest.get("WARNING") or rest.get("ERROR") or failed:
        warn = rest.get("WARNING", 0) + rest.get("ERROR", 0)
        extra = f" — {len(failed)} check(s) did not complete" if failed else ""
        return ExportDecision("AVAILABLE", level.value, [], 0, rest, True, f"EXPORT AVAILABLE — warnings remain: {warn}{extra}")
    return ExportDecision("READY", level.value, [], 0, rest, True, "READY FOR EXPORT")


def compute_scores(issues: list[QCIssue], settings: QCSettings, total_duration: float, scene_count: int, failed_groups: list[str] | None = None, failed_checkers: list[str] | None = None) -> QCScores:
    groups = group_scores(issues, total_duration, scene_count, failed_groups)
    avail = {g: w for g, w in settings.group_weights.items() if g in groups and g not in (failed_groups or []) and w > 0}
    if not avail:  # every weight is 0 (or negative): count the analysed groups equally instead of scoring a clean project 0
        avail = {g: 1.0 for g in groups if g not in (failed_groups or [])}
    wsum = sum(avail.values())
    overall = round(sum(groups[g] * w for g, w in avail.items()) / wsum, 1) if wsum else 0.0
    c = counts(issues)
    decision = decide_export(issues, settings.block_level, settings.allow_export_override, failed_checkers)
    weakest = min((groups[g] for g in avail), default=0.0)  # one badly failing area must not be averaged away by seven good ones
    if c["CRITICAL"]:
        status = "BLOCKED"
    elif overall >= READY_AT and not c["ERROR"] and weakest >= REVIEW_AT:
        status = "READY"
    elif overall >= REVIEW_AT and weakest >= FIX_AT:
        status = "REVIEW"
    else:
        status = "FIX_REQUIRED"
    if failed_groups and status == "READY":
        status = "REVIEW"  # a partial analysis is never "Ready"
    if status == "READY" and decision.blocked:
        status = "REVIEW"  # the user's block level stops the export on what is left (e.g. warnings): "Ready" next to "Export blocked" would contradict itself
    label = f"{overall:.0f}/100 — {STATUS_TEXT[status]}"
    if c["CRITICAL"]:
        label += f" ({c['CRITICAL']} critical)"
    return QCScores(overall, groups, status, label, c, decision.status, sorted(failed_groups or []))


def describe(scores: QCScores) -> str:
    """The dashboard header: overall line plus the nine scores."""
    rows = [f"Overall Quality  {scores.status_label}"]
    for g in SCORE_GROUPS:
        rows.append(f"{GROUP_LABELS[g]:<20}{'—' if g in scores.unavailable else f'{scores.groups.get(g, 0):.0f}'}")
    return "\n".join(rows)


_ = BLOCKING
