"""Severity levels, their meaning, sort order, export-blocking rules and confidence handling.

CRITICAL  the video cannot safely proceed to a final render
ERROR     major editorial / technical problem that needs attention
WARNING   the video can render but quality may suffer
NOTICE    an optimisation opportunity
INFO      useful, non-blocking information
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    NOTICE = "NOTICE"
    INFO = "INFO"


ORDER: tuple[Severity, ...] = (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.NOTICE, Severity.INFO)
RANK: dict[Severity, int] = {s: i for i, s in enumerate(ORDER)}  # 0 = most severe
LABELS: dict[Severity, str] = {Severity.CRITICAL: "Critical", Severity.ERROR: "Error", Severity.WARNING: "Warning", Severity.NOTICE: "Notice", Severity.INFO: "Info"}
PLURAL: dict[Severity, str] = {Severity.CRITICAL: "Critical", Severity.ERROR: "Errors", Severity.WARNING: "Warnings", Severity.NOTICE: "Notices", Severity.INFO: "Info"}
COLORS: dict[Severity, str] = {Severity.CRITICAL: "#ff4d4d", Severity.ERROR: "#ff8a3d", Severity.WARNING: "#e5c04a", Severity.NOTICE: "#4aa3e5", Severity.INFO: "#8a94a6"}
MEANING: dict[Severity, str] = {
    Severity.CRITICAL: "The video cannot safely proceed to a final render.",
    Severity.ERROR: "A major editorial or technical problem that needs attention.",
    Severity.WARNING: "The video can render, but quality may suffer.",
    Severity.NOTICE: "An optimisation opportunity.",
    Severity.INFO: "Useful, non-blocking information.",
}


def parse_severity(value: "Severity | str") -> Severity:
    if isinstance(value, Severity):
        return value
    try:
        return Severity(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"Unknown severity {value!r}") from exc


def at_least(a: Severity, b: Severity) -> bool:
    """True when ``a`` is as severe as, or more severe than, ``b``."""
    return RANK[a] <= RANK[b]


def worse(a: Severity, b: Severity) -> Severity:
    return a if RANK[a] <= RANK[b] else b


def milder(a: Severity, b: Severity) -> Severity:
    return a if RANK[a] >= RANK[b] else b


# ---------------------------------------------------------------------------------------------- export blocking
class BlockLevel(str, Enum):
    """Which severities stop an export (project setting; default CRITICAL_ERROR)."""

    CRITICAL = "CRITICAL"
    CRITICAL_ERROR = "CRITICAL_ERROR"
    CRITICAL_ERROR_WARNING = "CRITICAL_ERROR_WARNING"


BLOCK_LABELS: dict[BlockLevel, str] = {BlockLevel.CRITICAL: "Critical only", BlockLevel.CRITICAL_ERROR: "Critical + Errors", BlockLevel.CRITICAL_ERROR_WARNING: "Critical + Errors + Warnings"}
BLOCKING: dict[BlockLevel, frozenset[Severity]] = {
    BlockLevel.CRITICAL: frozenset({Severity.CRITICAL}),
    BlockLevel.CRITICAL_ERROR: frozenset({Severity.CRITICAL, Severity.ERROR}),
    BlockLevel.CRITICAL_ERROR_WARNING: frozenset({Severity.CRITICAL, Severity.ERROR, Severity.WARNING}),
}


def parse_block_level(value: "BlockLevel | str") -> BlockLevel:
    if isinstance(value, BlockLevel):
        return value
    try:
        return BlockLevel(str(value).upper())
    except ValueError:
        return BlockLevel.CRITICAL_ERROR


def blocks_export(severity: Severity, level: "BlockLevel | str") -> bool:
    return severity in BLOCKING[parse_block_level(level)]


# ---------------------------------------------------------------------------------------------- confidence
def confidence_label(confidence: float) -> str:
    return "High" if confidence >= 80 else "Medium" if confidence >= 55 else "Low"


def cap_for_confidence(severity: Severity, confidence: float, caps: tuple[tuple[float, str], ...]) -> Severity:
    """AI judgements are never presented as absolute: below each confidence threshold the severity may not exceed the given cap.

    ``caps`` is ordered by threshold, e.g. ((50, "NOTICE"), (70, "WARNING")): confidence < 50 -> at most NOTICE, < 70 -> at most WARNING.
    """
    out = severity
    for threshold, cap in sorted(caps):
        if confidence < threshold:
            out = milder(out, parse_severity(cap))
            break
    return out


# ---------------------------------------------------------------------------------------------- priority
def priority_key(severity: Severity, viewer_impact: float, affected_seconds: float, confidence: float, scene_importance: float) -> tuple:
    """Sort key for the dashboard (smaller = first): severity, then viewer impact, affected duration, confidence, scene importance."""
    return (RANK[severity], -round(viewer_impact, 3), -round(affected_seconds, 2), -round(confidence, 1), -round(scene_importance, 3))
