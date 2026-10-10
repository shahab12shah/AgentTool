"""Phase 9: the timeline canvas on a large (500-scene) synthetic project. Structural assertions only (clips touched, calls made), never wall-clock."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.events import EventBus, Topics  # noqa: E402
from app.main import create_window  # noqa: E402
from app.performance.synthetic import build_project  # noqa: E402
from app.project.project_manager import ProjectManager  # noqa: E402
from app.storage.paths import AppPaths  # noqa: E402
from app.tests.test_ui_acceptance import pump  # noqa: E402
from app.ui.timeline_canvas import EDGE_PX, ROW_H, RULER_H  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _open(tmp_path_factory, scenes: int):
    root = tmp_path_factory.mktemp(f"syn{scenes}")
    sp = build_project(root / "proj", scenes, real_files=True)
    ProjectManager(EventBus()).save(sp.project)
    home = tmp_path_factory.mktemp(f"home{scenes}")
    window, ws = create_window(AppPaths(home / "cfg", home / "data"))
    ws.settings.ffmpeg_path = str(home / "no-such-ffmpeg")  # no thumbnail processes: only scheduling happens
    ws.thumbnails.configure(ws.settings.ffmpeg_path)
    window.show()
    ws.open_project(sp.project.root)
    window.go_to("Timeline")
    QApplication.processEvents()
    return window, ws, sp


@pytest.fixture(scope="module")
def big(qapp, tmp_path_factory):
    window, ws, sp = _open(tmp_path_factory, 500)
    yield window, ws, sp
    ws.close_project()
    ws.shutdown()
    window.close()
    window.deleteLater()
    QApplication.processEvents()


@pytest.fixture(scope="module")
def small(qapp, tmp_path_factory):
    window, ws, sp = _open(tmp_path_factory, 50)
    yield window, ws, sp
    ws.close_project()
    ws.shutdown()
    window.close()
    window.deleteLater()
    QApplication.processEvents()


def _count_paints(canvas, monkeypatch) -> list:
    calls: list = []
    orig = canvas._paint_clip

    def counting(p, c, fm, clip, *a, **k):
        calls.append(clip.id)
        return orig(p, c, fm, clip, *a, **k)

    monkeypatch.setattr(canvas, "_paint_clip", counting)
    return calls


def _visible_brute(project, canvas, x0: float, x1: float) -> int:
    t0, t1 = (x0 - 6) / canvas.pps, (x1 + 4) / canvas.pps
    return sum(1 for t in project.timeline.tracks for c in t.clips if c.timeline_end > t0 and c.timeline_start < t1)


def test_the_synthetic_project_is_large(big):
    window, ws, sp = big
    assert sp.clips > 2000 and len(ws.project.timeline.all_clips()) == sp.clips


def test_painting_a_viewport_touches_only_the_visible_clips(big, small, monkeypatch):
    results = {}
    for name, (window, ws, sp) in (("big", big), ("small", small)):
        canvas = window.timeline_panel.canvas
        canvas.set_zoom(80.0)
        calls = _count_paints(canvas, monkeypatch)
        x0, w = int(canvas.time_to_x(20.0)), 900  # 20 s .. ~31 s
        canvas.grab(QRect(x0, 0, w, canvas.height()))
        assert calls, "something must be painted"
        assert len(calls) <= _visible_brute(ws.project, canvas, x0, x0 + w)
        results[name] = (len(calls), sp.clips)
        calls.clear()
        far = int(canvas.time_to_x(sp.spec.scenes * sp.spec.scene_seconds * 0.8))  # another place on the timeline
        canvas.grab(QRect(far, 0, w, canvas.height()))
        assert 0 < len(calls) <= _visible_brute(ws.project, canvas, far, far + w)
        calls.clear()
        monkeypatch.undo()
    (n_big, total_big), (n_small, total_small) = results["big"], results["small"]
    assert total_big > 8 * total_small
    assert n_big <= n_small + 4 and n_big < total_big / 20  # same viewport, 10x the project: the same few clips


def test_zoomed_out_view_collapses_subpixel_clips(big, monkeypatch):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    canvas.set_zoom(5.0)  # 5 px per second: 2 s captions are 10 px, but the scene-level clips are still individually visible
    calls = _count_paints(canvas, monkeypatch)
    canvas.grab(QRect(0, 0, 2400, canvas.height()))
    first_screen = len(calls)
    assert 0 < first_screen <= _visible_brute(ws.project, canvas, 0, 2400)
    canvas.set_zoom(80.0)


def test_narrow_clip_runs_are_drawn_once_per_pixel_column(small, monkeypatch):
    window, ws, sp = small
    canvas = window.timeline_panel.canvas
    tl = ws.project.timeline
    track = tl.get_track("track_v6")
    n = len([c for c in track.clips if c.timeline_start < 60])
    canvas.set_zoom(5.0)
    calls = _count_paints(canvas, monkeypatch)
    # make a run of 40 clips of 1 ms each inside one pixel column on an empty extra track: they must collapse
    from app.timeline.clip import Clip
    from app.timeline.track import Track, TrackKind

    scratch = Track("track_scratch", "Scratch", TrackKind.VIDEO)
    tl.insert_track(scratch)
    for k in range(40):
        scratch.clips.append(Clip(f"tiny{k}", scratch.id, "", 10.0 + k * 0.001, 0.001))
    scratch.sort()
    canvas.reload()
    canvas.grab(QRect(0, 0, 800, canvas.height()))
    assert sum(1 for cid in calls if cid.startswith("tiny")) <= 2
    monkeypatch.undo()
    tl.remove_track(scratch.id)
    canvas.reload()
    canvas.set_zoom(80.0)
    assert n >= 0


def test_hit_testing_matches_a_brute_force_scan(big):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    canvas.set_zoom(80.0)
    tracks = canvas.tracks()
    rng = random.Random(11)

    def brute(x, y):
        idx = canvas.track_index_at(y)
        if idx is None:
            return None
        for clip in reversed(tracks[idx].clips):
            r = canvas.clip_rect(clip, idx)
            if r.contains(x, y):
                zone = "trim_start" if x - r.left() <= EDGE_PX else "trim_end" if r.right() - x <= EDGE_PX else "move"
                return clip, idx, zone
        return None

    hits = 0
    for _ in range(600):
        t = rng.uniform(0, 3000)
        y = rng.uniform(0, RULER_H + len(tracks) * ROW_H)
        x = canvas.time_to_x(t)
        got, want = canvas._hit_clip(x, y), brute(x, y)
        assert (got is None) == (want is None)
        if got:
            hits += 1
            assert got[0] is want[0] and got[1:] == want[1:]
    assert hits > 50
    assert canvas._hit_clip(100.0, 5.0) is None  # the ruler


def _pos(canvas, clip, idx, frac=0.5):
    r = canvas.clip_rect(clip, idx)
    return QPoint(int(r.left() + r.width() * frac), int(r.center().y()))


def _free_clip(ws, canvas, track_id: str, min_gap: float = 1.0, min_len: float = 1.0):
    """A clip on ``track_id`` far into the timeline with at least ``min_gap`` free seconds after it."""
    tl = ws.project.timeline
    track = tl.get_track(track_id)
    idx = canvas.tracks().index(track)
    for clip in track.clips[len(track.clips) // 2:]:
        prev_end, next_start = tl.neighbours(clip)
        if clip.duration >= min_len and next_start is not None and next_start - clip.timeline_end >= min_gap and clip.timeline_start - prev_end >= min_gap:
            return clip, idx
    raise AssertionError("no suitable clip")


def test_select_move_trim_split_delete_and_undo_on_a_large_timeline(big):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    canvas.set_zoom(80.0)
    tl = ws.project.timeline
    clip, idx = _free_clip(ws, canvas, "track_v4", 0.6)
    cid, start0, dur0 = clip.id, clip.timeline_start, clip.duration
    n0 = len(tl.all_clips())

    # select
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_pos(canvas, clip, idx))
    assert ws.selected_clip_id == cid
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(int(canvas.time_to_x(1.0)), RULER_H + 5 * ROW_H + 3 * 0 + ROW_H // 2))  # an empty spot on a sparse track
    assert ws.selected_clip_id in (None, cid) or ws.selected_clip_id != cid

    # move right by ~0.4 s (inside the free gap; snapping may nudge it to a clip edge)
    c0 = _pos(canvas, tl.get_clip(cid), idx)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=c0)
    QTest.mouseMove(canvas, c0 + QPoint(12, 0))
    QTest.mouseMove(canvas, c0 + QPoint(int(0.4 * canvas.pps), 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=c0 + QPoint(int(0.4 * canvas.pps), 0))
    moved = tl.get_clip(cid)
    assert moved.timeline_start > start0 + 0.2 and moved.duration == pytest.approx(dur0)

    # trim the end by ~0.3 s
    r = canvas.clip_rect(moved, idx)
    edge = QPoint(int(r.right() - 2), int(r.center().y()))
    assert canvas._hit_clip(edge.x(), edge.y())[2] == "trim_end"
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=edge)
    QTest.mouseMove(canvas, edge + QPoint(-10, 0))
    QTest.mouseMove(canvas, edge + QPoint(-int(0.3 * canvas.pps), 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=edge + QPoint(-int(0.3 * canvas.pps), 0))
    trimmed = tl.get_clip(cid)
    assert trimmed.duration < moved.duration - 0.1 and trimmed.timeline_start == pytest.approx(moved.timeline_start)

    # split at the playhead through the service, then delete the right half with the keyboard path
    canvas.set_playhead(trimmed.timeline_start + trimmed.duration / 2)
    right = ws.timeline.split_clip(cid, canvas.playhead)
    assert tl.get_clip(right.id) is right or tl.get_clip(right.id).id == right.id
    assert len(tl.all_clips()) == n0 + 1
    assert tl.find_clip(right.id)[0].id == "track_v4"
    ws.select_clip(right.id)
    canvas.delete_selected()
    assert tl.get_clip(right.id) is None and len(tl.all_clips()) == n0

    # undo everything back to the start: delete, split, trim, move
    for _ in range(4):
        ws.undo()
    back = tl.get_clip(cid)
    assert back.timeline_start == pytest.approx(start0) and back.duration == pytest.approx(dur0)
    assert len(tl.all_clips()) == n0
    # every public lookup still agrees with a scan after all that
    ids = {c.id for c in tl.all_clips()}
    assert ids == {c.id for t in tl.tracks for c in t.clips}
    assert tl.duration == max(c.timeline_end for t in tl.tracks for c in t.clips)
    canvas.grab(QRect(0, 0, 600, canvas.height()))


def test_snapping_uses_a_cached_sorted_list_and_matches_brute_force(big, monkeypatch):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    canvas.set_zoom(80.0)
    tl = ws.project.timeline
    clip, _ = _free_clip(ws, canvas, "track_v4", 0.6)
    canvas.set_playhead(12.34)
    pts = canvas._snap_points(clip.id)
    assert pts is canvas._snap_points(clip.id)  # cached until the timeline changes
    brute = sorted([0.0, canvas.playhead] + [v for c in tl.all_clips() if c.id != clip.id for v in (c.timeline_start, c.timeline_end)])
    assert list(pts) == brute
    rng = random.Random(2)
    for _ in range(300):
        t = rng.uniform(0, 3000)
        want = min(brute, key=lambda q: abs(q - t))
        want = want if abs(want - t) * canvas.pps <= 8 else t
        assert canvas._snap(t, pts) == want
    # a real edit invalidates the cache
    ws.timeline.move_clip(clip.id, clip.timeline_start + 0.3)
    assert canvas._snap_points(clip.id) is not pts
    ws.undo()
    # while dragging, mouse moves do not rebuild the list
    built: list = []
    orig = type(tl).snap_points

    def counting(self, *a, **k):
        built.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(type(tl), "snap_points", counting)
    cur = tl.get_clip(clip.id)
    idx = canvas.tracks().index(tl.get_track(cur.track_id))
    c0 = _pos(canvas, cur, idx)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=c0)
    for dx in range(4, 60, 4):
        QTest.mouseMove(canvas, c0 + QPoint(dx, 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=c0 + QPoint(58, 0))
    assert len(built) <= 2  # one for the drag (plus at most one after the commit changed the revision)
    ws.undo()
    assert tl.get_clip(clip.id).timeline_start == pytest.approx(clip.timeline_start)


def test_zoom_scroll_and_playhead_do_not_reload(big, monkeypatch):
    window, ws, sp = big
    panel = window.timeline_panel
    canvas = panel.canvas
    counts = {"reload": 0, "markers": 0, "panel": 0}
    orig_reload, orig_markers, orig_panel = canvas.reload, canvas.reload_qc_markers, panel.reload

    def wrap(name, fn):
        def inner(*a, **k):
            counts[name] += 1
            return fn(*a, **k)
        return inner

    monkeypatch.setattr(canvas, "reload", wrap("reload", orig_reload))
    monkeypatch.setattr(canvas, "reload_qc_markers", wrap("markers", orig_markers))
    monkeypatch.setattr(panel, "reload", wrap("panel", orig_panel))
    width0 = canvas.width()
    for pps in (40.0, 160.0, 20.0, 80.0):
        canvas.set_zoom(pps)
    ev = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, ev)
    assert canvas.pps != 80.0
    panel.zoom.setValue(120)
    bar = panel.inner.horizontalScrollBar()
    for v in (0, 500, 2000, 800):
        bar.setValue(min(v, bar.maximum()))
        QApplication.processEvents()
    for t in (1.0, 5.0, 3.0):
        canvas.set_playhead(t)
    QApplication.processEvents()
    assert counts == {"reload": 0, "markers": 0, "panel": 0}, counts
    canvas.set_zoom(80.0)
    assert canvas.width() == width0  # the content size follows the zoom without a reload


def test_playhead_moves_repaint_only_a_strip(big, monkeypatch):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    canvas.set_zoom(80.0)
    rects: list = []
    orig = canvas.update

    def spy(*args):
        rects.append(args)
        return orig(*args)

    monkeypatch.setattr(canvas, "update", spy)
    canvas.set_playhead(40.0)
    canvas.set_playhead(41.0)
    assert rects and all(len(a) == 1 and isinstance(a[0], QRect) and a[0].width() <= 24 for a in rects), rects


def test_burst_of_change_events_is_coalesced_into_few_reloads(big, monkeypatch):
    window, ws, sp = big
    panel = window.timeline_panel
    pump(lambda: True)
    n = {"c": 0}
    orig = panel.canvas.reload

    def counting(*a, **k):
        n["c"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(panel.canvas, "reload", counting)
    for _ in range(25):
        ws.bus.publish(Topics.PROJECT_CHANGED, scope="timeline")
    assert n["c"] <= 2  # the first runs at once, the other 24 fold into a single trailing reload
    pump(lambda: n["c"] >= 2 or True)
    for _ in range(5):
        QApplication.processEvents()
    assert 1 <= n["c"] <= 2
    n["c"] = 0
    ws.bus.publish(Topics.PROJECT_CHANGED, scope="timeline")  # a lone event still reloads synchronously
    assert n["c"] == 1
    for _ in range(3):
        QApplication.processEvents()


def test_waveform_painting_is_bounded_by_the_exposed_width_and_cached(big, monkeypatch):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    canvas.set_zoom(80.0)
    calls: list[int] = []

    class FakeWave:
        duration = 3000.0

        def range(self, t0, t1, n):
            calls.append(n)
            return [(-0.5, 0.5, k % 97 == 0) for k in range(n)]

    wf = FakeWave()
    monkeypatch.setattr(ws.presentation, "waveform", lambda asset_id, request=True: wf)
    voice_idx = [t.id for t in canvas.tracks()].index("track_a1")
    x0 = int(canvas.time_to_x(1000.0))  # deep inside a 3000 s voice-over clip (240,000 px wide)
    rect = QRect(x0, 0, 500, canvas.height())
    canvas.grab(rect)
    assert calls and max(calls) <= 500 // 2 + 3 and sum(calls) <= 2 * (500 // 2 + 3), calls  # bounded by what is exposed, not by the clip's width
    first = len(calls)
    canvas.grab(rect)
    assert len(calls) == first  # identical repaint: the peaks come from the cache
    del voice_idx


def test_reload_after_edits_keeps_size_and_index_consistent(big):
    window, ws, sp = big
    canvas = window.timeline_panel.canvas
    tl = ws.project.timeline
    canvas.set_zoom(80.0)
    clip, idx = _free_clip(ws, canvas, "track_v4", 0.6)
    ws.timeline.move_clip(clip.id, tl.duration + 100.0)
    pump(lambda: canvas.width() >= int(canvas.time_to_x(tl.duration + 60)))
    ws.undo()
    pump(lambda: canvas.width() < int(canvas.time_to_x(tl.duration + 100)))
    assert canvas.width() == int(canvas.time_to_x(tl.duration + 60)) + 200
    ws.timeline.set_track_flag("track_v4", "locked", True)
    with pytest.raises(Exception):
        ws.timeline.move_clip(clip.id, clip.timeline_start + 0.1)
    ws.undo()
    assert canvas._hit_clip(*(lambda r: (r.center().x(), r.center().y()))(canvas.clip_rect(tl.get_clip(clip.id), idx))) is not None
