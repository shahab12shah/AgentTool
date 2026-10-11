"""Phase 9: the asset browser on a 1,500-asset library (lazy icons, single-row updates, failure state, debounced refreshes) and the preview widgets' quality tag and seek handling.
Structural assertions only (rows touched, icons held, calls made), never wall-clock. Waits use ``pump``, never QTest.qWait."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.events import EventBus, Topics  # noqa: E402
from app.main import create_window  # noqa: E402
from app.performance.synthetic import SyntheticSpec, build_project  # noqa: E402
from app.project.project_manager import ProjectManager  # noqa: E402
from app.storage.paths import AppPaths  # noqa: E402
from app.tests.test_ui_acceptance import pump, wait_ms  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def lib_env(qapp, tmp_path_factory):
    root = tmp_path_factory.mktemp("libsyn")
    sp = build_project(root / "proj", SyntheticSpec(scenes=1000, assets_per_scene=1.5), real_files=True)
    ProjectManager(EventBus()).save(sp.project)
    home = tmp_path_factory.mktemp("libhome")
    window, ws = create_window(AppPaths(home / "cfg", home / "data"))
    ws.performance.monitor.thresholds["load_elevated"] = 1e9
    jpg = home / "t.jpg"
    img = QImage(320, 180, QImage.Format.Format_RGB32)
    img.fill(QColor("#336699"))
    img.save(str(jpg))
    gen = SimpleNamespace(calls=0, fail=set())

    def fake_run(cmd, should_cancel):
        gen.calls += 1
        name = Path(cmd[-1]).name.split(".")[0]
        if name in gen.fail:
            return 1, "Invalid data found when processing input"
        Path(cmd[-1]).write_bytes(jpg.read_bytes())
        return 0, ""

    ws.thumbnails._run = fake_run
    ws.thumbnails.command = lambda src, asset, tmp: ["fake", str(tmp)]
    window.show()
    ws.open_project(sp.project.root)
    pump(lambda: ws.jobs.wait_idle(0.0), timeout=120)
    for _ in range(5):
        QApplication.processEvents()
    yield window, ws, sp, gen
    ws.close_project()
    ws.shutdown()
    window.close()
    window.deleteLater()
    QApplication.processEvents()


def test_opening_a_1500_asset_project_creates_few_jobs_and_lists_every_asset(lib_env):
    window, ws, sp, gen = lib_env
    lib = window.library
    n = len(ws.project.assets)
    assert n >= 1500 and lib.list.count() == n
    thumb_jobs = [j for j in ws.jobs.jobs() if j.type == "media.thumbnails"]
    assert 1 <= len(thumb_jobs) <= 6 and len(ws.jobs.jobs()) <= 12, len(ws.jobs.jobs())  # was one job per asset: 1,503
    assert gen.calls == n == len(list((ws.project.root / "thumbnails").glob("*.jpg")))  # one decode per asset, none repeated


def test_only_the_visible_rows_hold_a_real_icon_and_the_cache_is_bounded(lib_env):
    window, ws, sp, gen = lib_env
    lib = window.library
    window.go_to("Project")
    lib.refresh()
    pump(lambda: lib._visible, timeout=10)
    vis = lib._visible
    assert 0 < len(vis) < 200
    assert len(vis) <= len(lib._icons) <= lib._icons.max_items <= 2000  # icons are made for the rows on screen and held within a budget, not for all 1,500
    real = sum(1 for aid, it in lib._items.items() if it.icon().cacheKey() != lib._placeholder(ws.project.assets.get(aid).type.value, 'pending').cacheKey())
    assert real <= len(vis)
    lib._icons.resize(max_items=40)
    sb = lib.list.verticalScrollBar()
    for pos in range(0, sb.maximum(), max(1, sb.maximum() // 12)):
        sb.setValue(pos)
        pump(lambda: lib._fill_timer.isActive() is False, timeout=5)
        lib._fill_visible()
        assert len(lib._icons) <= 40
    placeholders = sum(1 for aid, it in lib._items.items() if aid not in lib._visible and it.icon().cacheKey() == lib._placeholder(ws.project.assets.get(aid).type.value, "pending").cacheKey())
    assert placeholders >= len(lib._items) - 41 - len(lib._visible)  # rows that scrolled away gave their pixmaps back


def test_the_visible_rows_are_reported_to_the_thumbnail_queue_and_scrolling_updates_the_report(lib_env, monkeypatch):
    window, ws, sp, gen = lib_env
    lib = window.library
    seen = []
    monkeypatch.setattr(ws.media, "set_visible_assets", lambda ids: seen.append(list(ids)))
    lib._reported = ()
    lib._fill_visible()
    assert seen and set(seen[-1]) == lib._visible
    sb = lib.list.verticalScrollBar()
    sb.setValue(sb.maximum() // 2)
    pump(lambda: len(seen) >= 2, timeout=5)
    assert set(seen[-1]) != set(seen[0]) and set(seen[-1]) == lib._visible
    sb.setValue(0)
    pump(lambda: lib._visible == set(seen[0]) or len(seen) >= 3, timeout=5)


def test_a_thumbnail_arriving_updates_one_row_without_rebuilding_the_list(lib_env):
    window, ws, sp, gen = lib_env
    lib = window.library
    lib.list.verticalScrollBar().setValue(0)
    lib.refresh()
    aid = next(iter(lib._visible))
    asset = ws.project.assets.get(aid)
    item = lib._items[aid]
    lib._icons.pop(aid)
    item.setIcon(lib._placeholder(asset.type.value, "pending"))
    before_refreshes, before_item = lib.refresh_count, lib.list.item(0)
    ws.bus.publish(Topics.THUMBNAIL_READY, asset_id=aid)
    pump(lambda: item.icon().cacheKey() != lib._placeholder(asset.type.value, "pending").cacheKey(), timeout=5)
    assert lib.refresh_count == before_refreshes and lib.list.item(0) is before_item  # the same item objects: nothing was rebuilt
    far = next(a for a in lib._items if a not in lib._visible)
    ws.bus.publish(Topics.THUMBNAIL_READY, asset_id=far)
    wait_ms(50)
    assert lib.refresh_count == before_refreshes  # an off-screen row is picked up when it scrolls into view


def test_a_failed_thumbnail_shows_a_clear_state_and_the_retry_action_regenerates_it(lib_env):
    window, ws, sp, gen = lib_env
    lib = window.library
    lib.list.verticalScrollBar().setValue(0)
    lib.refresh()
    aid = next(iter(lib._visible))
    asset = ws.project.assets.get(aid)
    ws.media._thumbnails.thumbnail_path(ws.project.root, asset).unlink()
    gen.fail = {aid}
    ws.media.request_thumbnails([aid])
    pump(lambda: ws.jobs.wait_idle(0.0), timeout=20)
    pump(lambda: ws.media.thumbnail_state(aid) == "failed", timeout=10)
    pump(lambda: "could not be created" in lib._items[aid].toolTip(), timeout=10)
    assert lib._items[aid].icon().cacheKey() == lib._placeholder(asset.type.value, "failed").cacheKey()
    gen.fail = set()
    lib.retry_thumbnail(aid)
    pump(lambda: ws.media.thumbnail_state(aid) == "ready", timeout=10)
    pump(lambda: "could not be created" not in lib._items[aid].toolTip(), timeout=10)


def test_a_burst_of_asset_changes_refreshes_the_list_a_couple_of_times_not_once_per_event(lib_env):
    window, ws, sp, gen = lib_env
    lib = window.library
    wait_ms(200)
    start = lib.refresh_count
    for _ in range(60):  # 60 asset-change events in one tick (the handler the bridge calls for "project.changed")
        lib._refresh_soon()
    pump(lambda: lib.refresh_count > start, timeout=5)
    wait_ms(600)
    assert 1 <= lib.refresh_count - start <= 3
    for _ in range(3):  # and real events through the bus still reach it
        ws.bus.publish(Topics.PROJECT_CHANGED, scope="assets", command=None, action="do")
    pump(lambda: lib.refresh_count > start + 1, timeout=10)


def test_search_sort_and_selection_still_work_on_the_large_library(lib_env):
    window, ws, sp, gen = lib_env
    lib = window.library
    lib.search.setText("zzz-no-such")
    assert lib.list.count() == 0
    lib.search.setText("")
    n = lib.list.count()
    assert n == len(ws.project.assets)
    lib.sort.setCurrentText("Name")
    names = [lib.list.item(i).text().split("\n")[0] for i in range(n)]
    assert names == sorted(names, key=str.lower)
    lib.list.item(3).setSelected(True)
    lib.refresh()
    assert lib.selected_ids() == [lib.list.item(3).data(Qt.ItemDataRole.UserRole)]
    lib.sort.setCurrentText("Date added")


# ======================================================================== preview widgets
def test_the_timeline_preview_tags_reduced_quality_and_each_seek_starts_a_new_generation(lib_env):
    window, ws, sp, gen = lib_env
    from app.ui.timeline_preview import TimelinePreview

    prev = TimelinePreview(window.ctx)
    prev.resize(640, 360)
    g0 = prev.frames.generation
    prev.show_time(1.0)
    prev.show_time(2.0)
    assert prev.frames.generation == g0 + 2
    assert prev.frames.reduced_quality() == "" or prev.frames.quality == "draft"
    prev.frames._quality, prev.frames._quality_at = (lambda: "draft"), 0.0
    assert prev.frames.reduced_quality() == "Draft preview"
    prev.frames._quality_at = 0.0
    prev.grab()
    assert "export" in prev.toolTip().lower()  # the tag explains that the export is unaffected
    prev.frames._quality = lambda: "high"
    prev.frames._quality_at = 0.0
    prev.grab()
    assert prev.toolTip() == ""
    prev.release()
    prev.deleteLater()


def test_dragging_the_seek_slider_applies_only_the_newest_position(qapp):
    from app.ui.preview_panel import TransportControls

    class Player:
        volume = 1.0
        is_playing = False
        position = 0.0
        duration = 100.0

        def __init__(self):
            from PySide6.QtCore import QObject, Signal

            class Sig(QObject):
                position_changed = Signal(float)
                duration_changed = Signal(float)
                playing_changed = Signal(bool)

            self.sig = Sig()
            self.position_changed, self.duration_changed, self.playing_changed = self.sig.position_changed, self.sig.duration_changed, self.sig.playing_changed
            self.seeks = []

        def seek(self, s):
            self.seeks.append(s)

        def stop(self): ...
        def play(self): ...
        def pause(self): ...
        def set_volume(self, v): ...

    p = Player()
    tc = TransportControls(p)  # type: ignore[arg-type]
    tc.seek.setRange(0, 100000)
    for v in range(1000, 21000, 1000):
        tc.seek.sliderMoved.emit(v)
    assert len(p.seeks) == 1  # the first move is applied at once, the rest wait for the next tick
    pump(lambda: len(p.seeks) == 2, timeout=5)
    assert p.seeks[-1] == 20.0  # the newest wins; the 18 in between were superseded, not queued
    tc.deleteLater()
