"""Human-readable QC report (Markdown). Concise and actionable: what was found, how sure QC is, what to do — never any hidden reasoning."""

from __future__ import annotations

from typing import Any

from app.qc.issue_model import CATEGORY_LABELS, GROUP_LABELS, SCORE_GROUPS, FixRecord, IgnoreRecord, QCIssue, QCScores, sorted_issues
from app.qc.scoring import decide_export
from app.qc.severity import ORDER, PLURAL, Severity, confidence_label


def _t(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(max(0.0, seconds), 60.0)
    return f"{int(m):02d}:{s:04.1f}"


def _where(i: QCIssue, scene_names: dict[str, str]) -> str:
    parts = []
    if i.scene_id:
        parts.append(scene_names.get(i.scene_id, i.scene_id))
    if i.start_time is not None:
        parts.append(_t(i.start_time) + (f"–{_t(i.end_time)}" if i.end_time is not None and i.end_time > i.start_time else ""))
    return " · ".join(parts) or "whole project"


def _issue_block(i: QCIssue, scene_names: dict[str, str]) -> list[str]:
    conf = "" if i.confidence >= 99.5 else f" · confidence {i.confidence:.0f}% ({confidence_label(i.confidence)})"
    lines = [f"- **{i.title}** — {_where(i, scene_names)}{conf}"]
    if i.description:
        lines.append(f"  - Detected: {i.description}")
    if i.why_it_matters:
        lines.append(f"  - Why it matters: {i.why_it_matters}")
    if i.current_value or i.recommended_value:
        lines.append(f"  - Current: {i.current_value or '—'} → Recommended: {i.recommended_value or '—'}")
    if i.suggested_fix:
        tag = " (safe auto-fix available)" if i.auto_fix_available and i.auto_fix_safe else " (fix available — needs your approval)" if i.auto_fix_available else ""
        lines.append(f"  - Suggested fix: {i.suggested_fix}{tag}")
    if i.fix_blocked_reason:
        lines.append(f"  - Note: {i.fix_blocked_reason}")
    return lines


def build_report(project_name: str, record: dict[str, Any], scores: QCScores, issues: list[QCIssue], fixes: list[FixRecord], ignored: list[IgnoreRecord], scene_names: dict[str, str] | None = None,
                 block_level: str = "CRITICAL_ERROR", render_results: dict[str, Any] | None = None, allow_override: bool = False) -> str:
    scene_names = scene_names or {}
    active = sorted_issues([i for i in issues if i.active])
    out: list[str] = [f"# AI Quality Control report — {project_name}", ""]
    out += ["## Project", f"- Name: {project_name}", f"- Project version: {record.get('project_version', '—')} · timeline version {record.get('timeline_version', 0)}", ""]
    out += ["## QC run", f"- Run #{record.get('number', '?')} ({record.get('trigger', 'manual')}) · {record.get('finished_at') or record.get('created_at', '')}",
            f"- State: {record.get('state', 'COMPLETED')} · {record.get('seconds', 0):.1f} s" + (f" · {record.get('cache_hits', 0)} cached check(s)" if record.get("cache_hits") else "")]
    failed = [k for k, v in (record.get("checkers") or {}).items() if v.get("state") == "FAILED"]
    if failed:
        out.append(f"- **Checks that did not complete:** {', '.join(failed)} (their results are missing, not clean)")
    out.append("")
    out += ["## Overall score", f"**{scores.status_label}**", "", "| " + " | ".join(PLURAL[s] for s in ORDER) + " |", "|" + "---|" * len(ORDER),
            "| " + " | ".join(str(scores.counts.get(s.value, 0)) for s in ORDER) + " |", ""]
    for sev in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.NOTICE):
        group = [i for i in active if i.severity is sev]
        out.append(f"## {PLURAL[sev] if sev is not Severity.CRITICAL else 'Critical issues'} ({len(group)})")
        if not group:
            out.append("None.")
        for i in group:
            out += _issue_block(i, scene_names)
        out.append("")
    infos = [i for i in active if i.severity is Severity.INFO]
    if infos:
        out += [f"## Information ({len(infos)})"] + [f"- {i.title} — {_where(i, scene_names)}" for i in infos] + [""]
    out += ["## Category scores"]
    for g in SCORE_GROUPS:
        out.append(f"- {GROUP_LABELS[g]}: {'—  (not analysed)' if g in scores.unavailable else f'{scores.groups.get(g, 0):.0f}'}")
    out.append("")
    out += ["## Scene-by-scene findings"]
    by_scene: dict[str, list[QCIssue]] = {}
    for i in active:
        if i.scene_id:
            by_scene.setdefault(i.scene_id, []).append(i)
    if not by_scene:
        out.append("No scene-specific findings.")
    for sid, group in by_scene.items():
        out.append(f"- **{scene_names.get(sid, sid)}**: " + "; ".join(f"{i.severity.value.title()} — {i.title}" for i in group))
    out.append("")
    out += ["## Applied fixes"] + ([f"- {f.summary or f.kind} ({f.code})" + (" — undone" if f.reverted else "") for f in fixes] or ["None."]) + [""]
    out += ["## Ignored issues"] + ([f"- {r.title or r.code} — {r.reason or 'no reason given'} ({'all of this type' if r.scope == 'type' else 'this issue'})" for r in ignored] or ["None."]) + [""]
    d = decide_export(issues, block_level, allow_override, failed)
    out += ["## Export readiness", f"**{d.message}**", f"- Blocking level: {block_level.replace('_', ' + ').title()}"]
    if d.blocking_ids:
        out.append(f"- {len(d.blocking_ids)} issue(s) must be resolved first.")
    for rid, rr in (render_results or {}).items():
        out.append(f"- Rendered file check {rid}: {rr.get('status', '?')} — {rr.get('summary', '')}")
    out.append("")
    out += ["## Recommendations"]
    recs = [i for i in active if i.suggested_fix and i.severity in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING)][:12]
    out += [f"{n}. {i.suggested_fix} ({i.title}, {_where(i, scene_names)})" for n, i in enumerate(recs, 1)] or ["Nothing required before export."]
    out += ["", "_QC checks technical and editorial quality. It does not verify facts: items marked for review are possible inconsistencies, not corrections._"]
    return "\n".join(out) + "\n"


_ = CATEGORY_LABELS
