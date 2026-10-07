"""QC history: every run is archived (``project.qc_history``) and any two runs can be compared.

An archive entry holds the scores, the compact issue list (fingerprints, never the project content), the fixes applied and the ignored fingerprints, so
"QC Run #4 vs #5" can say what improved and what is new without keeping a copy of the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.qc.issue_model import GROUP_LABELS, SCORE_GROUPS
from app.qc.severity import ORDER, Severity


@dataclass
class RunComparison:
    older: int
    newer: int
    overall: tuple[float, float] = (0.0, 0.0)
    groups: dict[str, tuple[float, float]] = field(default_factory=dict)
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    new_issues: list[dict[str, Any]] = field(default_factory=list)  # present now, not before
    resolved_issues: list[dict[str, Any]] = field(default_factory=list)  # present before, gone now
    persisting: int = 0

    def delta(self, group: str) -> float:
        a, b = self.groups.get(group, (0.0, 0.0))
        return round(b - a, 1)

    @property
    def improved(self) -> bool:
        return self.overall[1] > self.overall[0]

    def lines(self) -> list[str]:
        out = [f"QC Run #{self.older} vs #{self.newer}", f"Overall: {self.overall[0]:.0f} → {self.overall[1]:.0f}"]
        for g in SCORE_GROUPS:
            if g in self.groups and self.groups[g][0] != self.groups[g][1]:
                out.append(f"{GROUP_LABELS[g]}: {self.groups[g][0]:.0f} → {self.groups[g][1]:.0f}")
        for sev in ORDER:
            a, b = self.counts.get(sev.value, (0, 0))
            if a != b:
                out.append(f"{sev.value.title()}: {a} → {b}")
        out.append(f"Resolved: {len(self.resolved_issues)} · New: {len(self.new_issues)} · Still open: {self.persisting}")
        return out


def entry_for(history: list[dict[str, Any]], key: "str | int") -> dict[str, Any] | None:
    """Find an archive entry by run id or by run number."""
    for h in history:
        if h.get("qc_run_id") == key or h.get("number") == key:
            return h
    return None


def _active(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for i in entry.get("issues", []) if i.get("status", "OPEN") == "OPEN" and not i.get("ignored")]


def compare_entries(a: dict[str, Any], b: dict[str, Any]) -> RunComparison:
    """``a`` is the older run, ``b`` the newer one."""
    c = RunComparison(int(a.get("number", 0)), int(b.get("number", 0)), (float(a.get("overall_score", 0)), float(b.get("overall_score", 0))))
    ga, gb = a.get("category_scores", {}), b.get("category_scores", {})
    for g in SCORE_GROUPS:
        if g in ga or g in gb:
            c.groups[g] = (float(ga.get(g, 0.0)), float(gb.get(g, 0.0)))
    ca, cb = a.get("counts", {}), b.get("counts", {})
    for s in Severity:
        c.counts[s.value] = (int(ca.get(s.value, 0)), int(cb.get(s.value, 0)))
    fa = {i["fp"]: i for i in _active(a) if i.get("fp")}
    fb = {i["fp"]: i for i in _active(b) if i.get("fp")}
    c.new_issues = [i for k, i in fb.items() if k not in fa]
    c.resolved_issues = [i for k, i in fa.items() if k not in fb]
    c.persisting = sum(1 for k in fb if k in fa)
    return c
