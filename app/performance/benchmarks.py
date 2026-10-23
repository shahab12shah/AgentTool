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


def bench_edit(sp: SyntheticProject, root: Path) -> dict[str, float]:
    """Phase 10 manual editing through the real Workspace (project opened from disk, FFmpeg disabled): drag previews, commits, undo/redo, snapping and selection on the
    busiest track. Everything here is what a mouse gesture costs; none of it may scale with a full rebuild of the timeline."""
    from app.services.workspace import Workspace
    from app.storage.paths import AppPaths
    from app.timeline.index import epoch

    home = Path(tempfile.mkdtemp(prefix="perf_home_"))
    os.environ["AGENTTOOL_HOME"] = str(home)
    ws = Workspace(AppPaths.default())
    ws.settings.ffmpeg_path = str(home / "no-such-ffmpeg")
    ws.thumbnails.configure(ws.settings.ffmpeg_path)
    out: dict[str, float] = {}
    try:
        ws.open_project(sp.project.root)
        ws.jobs.wait_idle(120)
        tl = ws.project.timeline
        rng = random.Random(5)
        busiest = max(tl.tracks, key=lambda t: len(t.clips))
        ids = [c.id for c in busiest.clips]
        out["edit.clips_on_busiest_track"] = float(len(ids))
        end = max(tl.duration, 1.0)
        ws.edit.state.update(linked_selection=False, ripple_on=False)

        t = time.perf_counter()
        ws.selection.select_rect(tl, 0.0, end, 0, len(tl.tracks) - 1)
        out["edit.select_rect_all_ms"] = 1000 * (time.perf_counter() - t)
        out["edit.selected_clips"] = float(len(ws.selection))
        ws.selection.select_none()

        out["edit.preview_move_all_ms"] = 1000 * _best(lambda: ws.edit.preview_move(ids, 1.0), 5)
        out["edit.preview_move_200_ms"] = 1000 * _best(lambda: ws.edit.preview_move(ids[-200:], 1.0), 5)  # the last 200 clips of the track: a typical large selection (nothing sits after them)
        out["edit.preview_trim_ms"] = 1000 * _best(lambda: ws.edit.preview_trim(ids[len(ids) // 2], "end", busiest.clips[len(ids) // 2].timeline_end - 0.2), 5)
        e1 = epoch()
        for _ in range(20):
            ws.edit.preview_move(ids, 1.0)
            ws.edit.preview_trim(ids[0], "end", busiest.clips[0].timeline_end - 0.1)
        out["edit.epoch_bumps_per_20_previews"] = float(epoch() - e1)  # 0 = a drag never invalidates the timeline index / snap caches

        t = time.perf_counter()
        ws.edit.move_clips(ids, 1.0)
        out["edit.commit_move_all_ms"] = 1000 * (time.perf_counter() - t)
        t = time.perf_counter()
        ws.commands.undo()
        out["edit.undo_move_all_ms"] = 1000 * (time.perf_counter() - t)
        t = time.perf_counter()
        ws.commands.redo()
        out["edit.redo_move_all_ms"] = 1000 * (time.perf_counter() - t)
        ws.commands.undo()

        sess = ws.edit.snap.session()
        t = time.perf_counter()
        sess.snap(end / 2, pps=80.0)
        out["edit.snap_first_query_ms"] = 1000 * (time.perf_counter() - t)  # builds the target arrays once
        qs = [rng.uniform(0, end) for _ in range(2000)]
        t = time.perf_counter()
        for q in qs:
            ws.edit.snap.snap(q, pps=80.0)
        out["edit.snap_query_us"] = 1e6 * (time.perf_counter() - t) / len(qs)
        out["edit.snap_builds_total"] = float(sum(ws.edit.snap.builds.values()))

        head = ids[: min(100, len(ids))]
        ws.edit.state.update(ripple_on=True)
        t = time.perf_counter()
        ws.edit.delete(head)
        out["edit.ripple_delete_100_ms"] = 1000 * (time.perf_counter() - t)
        ws.commands.undo()
        ws.edit.state.update(ripple_on=False)

        mid = busiest.clips[len(busiest.clips) // 2]
        t = time.perf_counter()
        try:
            ws.edit.blade(mid.timeline_start + mid.duration / 2)
            out["edit.blade_all_tracks_ms"] = 1000 * (time.perf_counter() - t)
            ws.commands.undo()
        except Exception:  # noqa: BLE001  (a locked/odd synthetic layout: record nothing rather than a wrong number)
            pass

        t = time.perf_counter()
        for i in range(500):
            ws.edit.add_marker(i * (end / 500), f"m{i}")
        out["edit.add_500_markers_s"] = time.perf_counter() - t
        t = time.perf_counter()
        for q in qs[:500]:
            ws.edit.snap.snap(q, pps=80.0)
        out["edit.snap_query_with_500_markers_us"] = 1e6 * (time.perf_counter() - t) / 500

        t = time.perf_counter()
        ws.projects.save()
        out["edit.save_after_edits_s"] = time.perf_counter() - t
        ws.close_project()
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
        from PySide6.QtCore import QRect

        out["ui.timeline_viewport_paint_s"] = _best(lambda: canvas.grab(QRect(0, 0, 1600, canvas.height())), 3)  # what a user sees: one screen of a possibly very wide widget
        canvas.set_zoom(canvas.pps * 4)
        out["ui.timeline_paint_zoomed_s"] = _best(canvas.grab, 3)
        out["ui.timeline_viewport_paint_zoomed_s"] = _best(lambda: canvas.grab(QRect(0, 0, 1600, canvas.height())), 3)
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


def run_suite(sizes: list[str], *, repeat: int = 3, workspace: bool = True, ui: bool = False, edit: bool = False, root: Path | None = None) -> dict[str, Any]:
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
        if edit:
            res.update(bench_edit(sp, base))
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
    ap.add_argument("--edit", action="store_true", help="also measure Phase 10 manual editing (previews, commits, undo, snapping, selection)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run_suite(a.sizes, repeat=a.repeat, workspace=not a.no_workspace, ui=a.ui, edit=a.edit)
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
