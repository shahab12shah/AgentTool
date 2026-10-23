"""Phase 10 performance: manual editing must not scale with the size of the timeline for what a mouse gesture costs (structural bounds, never fixed wall-clock budgets)."""

from __future__ import annotations

import pytest

from app.performance.benchmarks import run_suite


@pytest.fixture(scope="module")
def numbers():
    res = run_suite(["medium", "large"], repeat=1, workspace=False, edit=True)["results"]
    return {size: v["metrics"] for size, v in res.items()}


def test_a_drag_preview_never_invalidates_the_timeline_indexes(numbers):
    for size in ("medium", "large"):
        assert numbers[size]["edit.epoch_bumps_per_20_previews"] == 0.0, size  # no full rebuild of the index / snap caches per mouse move


def test_preview_cost_follows_the_selection_not_the_timeline(numbers):
    m, l = numbers["medium"], numbers["large"]
    assert l["edit.preview_move_200_ms"] < 60.0  # 200 selected clips in a 1,500-clip track: far below one frame budget multiples
    assert l["edit.preview_move_200_ms"] < 8 * max(m["edit.preview_move_200_ms"], 1.0)  # 5x the timeline does not cost 5x per move
    assert l["edit.preview_trim_ms"] < 20.0


def test_snapping_builds_its_targets_once_and_queries_stay_cheap(numbers):
    for size in ("medium", "large"):
        assert numbers[size]["edit.snap_builds_total"] <= 8  # a few target arrays (clips, markers, scenes ...), each built once per edit, not per query
        assert numbers[size]["edit.snap_query_us"] < 2000.0
        assert numbers[size]["edit.snap_query_with_500_markers_us"] < 5000.0


def test_commits_scale_roughly_with_the_clips_they_touch(numbers):
    m, l = numbers["medium"], numbers["large"]
    clips_ratio = l["edit.clips_on_busiest_track"] / m["edit.clips_on_busiest_track"]
    assert l["edit.commit_move_all_ms"] < 4 * clips_ratio * max(m["edit.commit_move_all_ms"], 5.0)
    assert l["edit.select_rect_all_ms"] < 500.0 and l["edit.save_after_edits_s"] < 10.0


def test_a_burst_of_marker_edits_is_not_dominated_by_autosave_serialisation(numbers):
    assert numbers["large"]["edit.add_500_markers_s"] < 10.0  # was 41.7 s before edit-triggered autosaves were throttled
