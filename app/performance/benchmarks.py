"""Repeatable benchmarks over synthetic projects (``app.performance.synthetic``). Used for before/after comparison on the SAME machine.

Absolute times depend on the computer; the suite records the environment next to the numbers and ``compare`` reports ratios only. Tests assert generous
structural bounds (e.g. "bounded work", "no regression beyond 3x of a stored baseline on this machine"), never fixed wall-clock budgets.

    python -m app.performance.benchmarks --sizes small medium large stress --out result.json [--ui]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from app.performance.resource_monitor import process_rss_bytes
from app.performance.synthetic import SIZES, SyntheticProject, build_project


def _best(fn: Callable[[], Any], repeat: int = 3) -> float:
    """Median of ``repeat`` runs (seconds) — more stable than the mean on a shared machine."""
    runs = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t)
    return statistics.median(runs)


def environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(), "machine": platform.machine()}


def bench_core(sp: SyntheticProject, root: Path, repeat: int = 3) -> dict[str, float]:
    """Qt-free measurements: serialisation, load, timeline model, lookups, QC startup."""
    from app.project.project import Project
    from app.project.project_manager import ProjectManager
    from app.core.events import EventBus
    from app.qc.context import QCContext, snapshot_project
    from app.timeline.timeline import Timeline
    from app.media.thumbnails import ThumbnailService

    p = sp.project
    rng = random.Random(1)
    out: dict[str, float] = {}
    out["project.to_document_s"] = _best(p.to_document, repeat)
    doc = p.to_document()
    out["project.json_dumps_s"] = _best(lambda: json.dumps(doc, indent=2, ensure_ascii=False), repeat)
    out["project.json_bytes"] = float(len(json.dumps(doc, indent=2, ensure_ascii=False)))
    pm = ProjectManager(EventBus())
    pm.current = p
    out["project.save_s"] = _best(lambda: pm.save(p), repeat)
    out["project.load_s"] = _best(lambda: ProjectManager.load_file(p.paths.project_file, p.root), repeat)
    out["project.from_document_s"] = _best(lambda: Project.from_document(doc, root=p.root), repeat)
    td = p.timeline.to_dict()
    out["timeline.model_build_s"] = _best(lambda: Timeline.from_dict(td), repeat)

    ids = [c.id for c in p.timeline.all_clips()]
    sample = [rng.choice(ids) for _ in range(200)]
    out["timeline.find_clip_ms"] = 1000 * _best(lambda: [p.timeline.find_clip(i) for i in sample], repeat) / len(sample)
    tracks = p.timeline.tracks
    probes = [(rng.choice(tracks).id, rng.uniform(0, sp.duration)) for _ in range(200)]

    def check() -> None:
        for tid, t in probes:
            try:
                p.timeline.check_free(tid, t, 0.5)
            except Exception:
                pass

    out["timeline.check_free_ms"] = 1000 * _best(check, repeat) / len(probes)
    out["timeline.duration_ms"] = 1000 * _best(lambda: p.timeline.duration, repeat)

    def snap_points() -> list[float]:  # what the canvas rebuilds on every mouse move while dragging
        pts = [0.0, 0.0]
        for c in p.timeline.all_clips():
            pts += [c.timeline_start, c.timeline_end]
        return pts

    out["timeline.snap_points_ms"] = 1000 * _best(snap_points, repeat)
    pts = snap_points()
    out["timeline.snap_nearest_ms"] = 1000 * _best(lambda: min(pts, key=lambda q: abs(q - 123.4)), repeat)
    assets = p.assets.all()
    out["asset.registry_all_ms"] = 1000 * _best(p.assets.all, repeat)
    ts = ThumbnailService("")
    out["thumbnail.cached_lookup_ms"] = 1000 * _best(lambda: [ts.is_cached(p.root, a) for a in assets], repeat)
    out["qc.snapshot_s"] = _best(lambda: snapshot_project(p), repeat)
    out["qc.context_build_s"] = _best(lambda: QCContext.build(p, detach=False), repeat)

    def hashes() -> None:
        ctx = QCContext.build(p, detach=False)
        for d in ("timeline", "scenes", "transcript", "assets", "audio", "captions", "visual", "reference", "render"):
            ctx.domain_hash(d)

    out["qc.content_hash_s"] = _best(hashes, repeat)
    rss = process_rss_bytes()
    out["memory.rss_mb"] = round(rss / 1048576, 1) if rss else 0.0
    return out


def bench_workspace(sp: SyntheticProject, root: Path) -> dict[str, float]:
    """Opening the saved project through the real Workspace: includes the thumbnail scheduling that runs on open (FFmpeg disabled so only scheduling is measured)."""
    from app.services.workspace import Workspace
    from app.storage.paths import AppPaths

    home = Path(tempfile.mkdtemp(prefix="perf_home_"))
    os.environ["AGENTTOOL_HOME"] = str(home)
    ws = Workspace(AppPaths.default())
    ws.settings.ffmpeg_path = str(home / "no-such-ffmpeg")
    ws.thumbnails.configure(ws.settings.ffmpeg_path)
    out: dict[str, float] = {}
    try:
        before = len(ws.jobs.jobs())
        t = time.perf_counter()
        ws.open_project(sp.project.root)
        out["workspace.open_project_s"] = time.perf_counter() - t
        out["workspace.jobs_created_on_open"] = float(len(ws.jobs.jobs()) - before)
        t = time.perf_counter()
        ws.jobs.wait_idle(60)
        out["workspace.thumbnail_jobs_drain_s"] = time.perf_counter() - t
        t = time.perf_counter()
        ws.projects.save()
        out["workspace.save_s"] = time.perf_counter() - t
        t = time.perf_counter()
        ws.close_project()
        out["workspace.close_s"] = time.perf_counter() - t
    finally:
        ws.shutdown()
    return out


def bench_ui(sp: SyntheticProject) -> dict[str, float]:
    """Timeline canvas in an offscreen Qt: first paint, repaint, hit-test and drag snapping. Needs PySide6."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.main import create_window
    from app.storage.paths import AppPaths

    app = QApplication.instance() or QApplication([])
    home = Path(tempfile.mkdtemp(prefix="perf_home_"))
    os.environ["AGENTTOOL_HOME"] = str(home)
    window, ws = create_window(AppPaths.default())
    ws.settings.ffmpeg_path = str(home / "no-such-ffmpeg")
    ws.thumbnails.configure(ws.settings.ffmpeg_path)
    out: dict[str, float] = {}
    try:
        window.show()
        ws.open_project(sp.project.root)
        window.go_to("Timeline")
        app.processEvents()
        canvas = window.timeline_panel.canvas
        canvas.resize(1600, canvas.content_height())
        t = time.perf_counter()
        canvas.grab()
        out["ui.timeline_first_paint_s"] = time.perf_counter() - t
        out["ui.timeline_paint_s"] = _best(canvas.grab, 3)
        canvas.set_zoom(canvas.pps * 4)
        out["ui.timeline_paint_zoomed_s"] = _best(canvas.grab, 3)
        rng = random.Random(3)
        pts = [(rng.uniform(0, 1500), rng.uniform(30, canvas.content_height())) for _ in range(200)]
        out["ui.hit_test_ms"] = 1000 * _best(lambda: [canvas._hit_clip(x, y) for x, y in pts], 3) / len(pts)
        for z in (0.25, 1.0, 4.0):
            canvas.set_zoom(max(0.5, canvas.pps * z))
            app.processEvents()
        out["ui.zoom_cycle_s"] = _best(lambda: (canvas.set_zoom(canvas.pps * 1.0001), canvas.grab()), 3)
    finally:
        ws.close_project()
        ws.shutdown()
        window.close()
    return out


def run_suite(sizes: list[str], *, repeat: int = 3, workspace: bool = True, ui: bool = False, root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"environment": environment(), "results": {}}
    for name in sizes:
        base = Path(root) if root else Path(tempfile.mkdtemp(prefix=f"perf_{name}_"))
        rss0 = process_rss_bytes() or 0
        t = time.perf_counter()
        sp = build_project(base / name, SIZES[name], real_files=True)
        build = time.perf_counter() - t
        res: dict[str, float] = {"synthetic.build_s": build}
        res.update(bench_core(sp, base, repeat))
        if workspace:
            res.update(bench_workspace(sp, base))
        if ui:
            res.update(bench_ui(sp))
        rss1 = process_rss_bytes() or 0
        res["memory.growth_mb"] = round((rss1 - rss0) / 1048576, 1)
        result["results"][name] = {"summary": sp.summary(), "metrics": {k: round(v, 6) for k, v in res.items()}}
    return result


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per size and metric: after/before (below 1.0 = faster / smaller). Only metrics present in both."""
    out: dict[str, dict[str, float]] = {}
    for size, b in before.get("results", {}).items():
        a = after.get("results", {}).get(size)
        if not a:
            continue
        out[size] = {k: round(a["metrics"][k] / v, 3) for k, v in b["metrics"].items() if k in a["metrics"] and v}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AgentTool performance benchmarks (synthetic projects)")
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"], choices=list(SIZES))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-workspace", action="store_true")
    ap.add_argument("--ui", action="store_true", help="also measure the timeline canvas (offscreen Qt)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run_suite(a.sizes, repeat=a.repeat, workspace=not a.no_workspace, ui=a.ui)
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
