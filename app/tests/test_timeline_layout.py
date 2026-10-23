"""Phase 10: the shared vertical geometry of the timeline (variable and collapsed track rows)."""

from __future__ import annotations

from types import SimpleNamespace as T

from app.ui.timeline_layout import COLLAPSED_ROW_H, DEFAULT_ROW_H, MAX_ROW_H, MIN_ROW_H, RULER_H, TrackLayout, row_height


def tracks(*specs):
    return [T(id=f"t{i}", height=h, collapsed=c) for i, (h, c) in enumerate(specs)]


def test_row_heights_default_custom_collapsed_and_clamped():
    assert row_height(T(height=0, collapsed=False)) == DEFAULT_ROW_H
    assert row_height(T(height=80, collapsed=False)) == 80
    assert row_height(T(height=80, collapsed=True)) == COLLAPSED_ROW_H
    assert row_height(T(height=5, collapsed=False)) == MIN_ROW_H and row_height(T(height=999, collapsed=False)) == MAX_ROW_H


def test_offsets_and_hit_testing():
    L = TrackLayout()
    ts = tracks((0, False), (60, False), (0, True))
    assert L.update(ts) and not L.update(ts)  # rebuilt once, then cached
    assert [L.top(i) for i in range(3)] == [RULER_H, RULER_H + 36, RULER_H + 96]
    assert L.total_height == RULER_H + 36 + 60 + COLLAPSED_ROW_H + 1
    assert L.index_at(RULER_H - 1) is None and L.index_at(RULER_H) == 0 and L.index_at(RULER_H + 35) == 0 and L.index_at(RULER_H + 36) == 1
    assert L.index_at(RULER_H + 96 + COLLAPSED_ROW_H) is None  # below the last row
    assert L.nearest_index(10_000) == 2 and L.nearest_index(0) == 0


def test_visible_rows_and_change_detection():
    L = TrackLayout()
    ts = tracks((0, False), (0, False), (0, False), (0, False))
    L.update(ts)
    assert list(L.visible(RULER_H + 40, RULER_H + 80)) == [1, 2]
    assert list(L.visible(0, RULER_H - 1)) == []
    ts[1].height = 100
    assert L.update(ts) and L.top(2) == RULER_H + 36 + 100  # a height change rebuilds
    ts[1].collapsed = True
    assert L.update(ts) and L.top(2) == RULER_H + 36 + COLLAPSED_ROW_H


def test_empty_timeline_keeps_room_for_a_row():
    L = TrackLayout()
    L.update([])
    assert L.count == 0 and L.total_height > RULER_H and L.index_at(RULER_H + 5) is None
