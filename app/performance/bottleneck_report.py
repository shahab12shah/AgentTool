"""Diagnostic report: where the time and resources actually go. Built from measured data only (no guessing), safe to export.

The report contains operation timings, cache statistics, the resource picture, the job queue state and the hardware/FFmpeg summary the caller supplies.
It never contains project content, API keys or full user paths: ``sanitize`` replaces the home directory with ``~`` and masks any value whose key looks like a secret.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from app.performance.profiler import Profiler
from app.performance.resource_monitor import ResourceMonitor
from app.storage.atomic import atomic_write_text

# Budgets are guidance for flagging, not pass/fail promises: they are generous and machine-independent in spirit.
DEFAULT_BUDGETS_S: dict[str, float] = {
    "app.startup": 5.0, "project.open": 3.0, "project.save": 2.0, "project.close": 1.0, "asset.index": 3.0, "timeline.load": 2.0, "timeline.paint": 0.08, "timeline.zoom": 0.15,
    "timeline.scroll": 0.05, "timeline.select": 0.05, "preview.seek": 0.4, "thumbnail.load": 0.1, "proxy.generate": 600.0, "scene.navigate": 0.2, "qc.start": 0.5,
    "qc.run": 120.0, "ai.orchestrate": 5.0, "render.start": 3.0, "render.run": 3600.0,
}
_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|credential)")


def sanitize(obj: Any, home: str | None = None) -> Any:
    """Deep copy with the home directory replaced by ``~`` and secret-looking values masked."""
    home = home or str(Path.home())

    def clean(v: Any, key: str = "") -> Any:
        if _SECRET_KEY.search(key):
            return "***"
        if isinstance(v, dict):
            return {str(k): clean(x, str(k)) for k, x in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [clean(x) for x in v]
        if isinstance(v, str):
            return v.replace(home, "~") if home and home != "/" else v
        if isinstance(v, Path):
            return str(v).replace(home, "~")
        return v

    return clean(obj)


def rank_bottlenecks(operations: dict[str, dict[str, Any]], budgets: dict[str, float] | None = None, limit: int = 12) -> list[dict[str, Any]]:
    """Operations ordered by total time spent; each row says whether its p95 exceeds the (optional) budget for that operation."""
    b = {**DEFAULT_BUDGETS_S, **(budgets or {})}
    rows = []
    for name, st in operations.items():
        budget = b.get(name)
        over = budget is not None and st.get("p95_s", 0.0) > budget
        rows.append({"operation": name, "count": st["count"], "total_s": st["total_s"], "mean_s": st["mean_s"], "p95_s": st["p95_s"], "max_s": st["max_s"], "errors": st["errors"],
                     "budget_s": budget, "over_budget": over})
    rows.sort(key=lambda r: (not r["over_budget"], -r["total_s"]))
    return rows[:limit]


def build_report(profiler: Profiler, monitor: ResourceMonitor | None = None, *, cache_stats: dict[str, Any] | None = None, hardware: dict[str, Any] | None = None,
                 jobs: dict[str, Any] | None = None, settings: dict[str, Any] | None = None, limits: dict[str, Any] | None = None, extra: dict[str, Any] | None = None,
                 budgets: dict[str, float] | None = None) -> dict[str, Any]:
    snap = profiler.snapshot()
    sample = monitor.sample() if monitor else None
    rates = snap.get("cache_hit_rates", {})
    findings: list[str] = []
    ranked = rank_bottlenecks(snap["operations"], budgets)
    for r in ranked:
        if r["over_budget"]:
            findings.append(f"{r['operation']}: p95 {r['p95_s']:.3f}s exceeds the {r['budget_s']}s guide ({r['count']} calls).")
    for cat, r in rates.items():
        if r["rate"] is not None and (r["hits"] + r["misses"]) >= 20 and r["rate"] < 0.5:
            findings.append(f"cache '{cat}': hit rate {r['rate']:.0%} over {r['hits'] + r['misses']} lookups — check invalidation or cache size.")
    if monitor:
        pr = monitor.pressure(sample)
        for k in ("memory", "disk", "cpu"):
            if pr[k] != "normal":
                findings.append(f"{k} pressure is {pr[k]}.")
    if jobs and jobs.get("failed_recent", 0) >= 3:
        findings.append(f"{jobs['failed_recent']} recent background jobs failed.")
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "metrics_enabled": snap["enabled"], "uptime_s": snap["uptime_s"], "bottlenecks": ranked, "findings": findings,
              "operations": snap["operations"], "counters": snap["counters"], "gauges": snap["gauges"], "cache_hit_rates": rates, "cache": cache_stats or {},
              "resources": sample.to_dict() if sample else None, "pressure": monitor.pressure(sample) if monitor else None, "jobs": jobs or {}, "hardware": hardware or {},
              "settings": settings or {}, "limits": limits or {}}
    if extra:
        report["extra"] = extra
    return sanitize(report)


def render_text(report: dict[str, Any]) -> str:
    out = [f"Performance report — {report.get('generated_at', '')}", ""]
    if report.get("findings"):
        out.append("Findings")
        out += [f"  - {f}" for f in report["findings"]]
        out.append("")
    out.append("Slowest operations (by total time)")
    for r in report.get("bottlenecks", [])[:10]:
        flag = "  [over budget]" if r["over_budget"] else ""
        out.append(f"  {r['operation']:<28} n={r['count']:<6} total={r['total_s']:.3f}s mean={r['mean_s'] * 1000:.1f}ms p95={r['p95_s'] * 1000:.1f}ms{flag}")
    rates = report.get("cache_hit_rates") or {}
    if rates:
        out += ["", "Cache hit rates"]
        out += [f"  {c:<18} {('%.0f%%' % (100 * r['rate'])) if r['rate'] is not None else 'n/a':>6}  ({r['hits']} hits, {r['misses']} misses)" for c, r in rates.items()]
    res = report.get("resources")
    if res:
        mb = lambda v: "n/a" if v is None else f"{v / 1048576:.0f} MB"  # noqa: E731
        out += ["", f"Process memory {mb(res['process_rss_bytes'])}; system available {mb(res['system_available_bytes'])} of {mb(res['system_total_bytes'])}; disk free {mb(res['disk_free_bytes'])}; threads {res['threads']}"]
    hw = report.get("hardware") or {}
    if hw.get("summary"):
        out += ["", "Hardware: " + str(hw["summary"])]
    return "\n".join(out)


def export_report(report: dict[str, Any], path: Path, as_text: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, render_text(report) if as_text else json.dumps(report, indent=2, default=str))
    return path
