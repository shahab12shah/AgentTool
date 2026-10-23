"""Phase 10 (W1-D): the snapping engine - targets, priority, frame tolerance, exclusion, hysteresis, caching and a brute-force property test."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from app.analysis.models import Scene
from app.timeline.clip import Clip
from app.timeline.markers import Marker
from app.timeline.snapping import (
    MAX_SNAP_PX,
    PRIORITY,
    SNAP_KINDS,
    STICKY_FACTOR,
    SnapEngine,
    SnapSettings,
)
from app.timeline.timebase import Timebase
from app.timeline.timeline import Timeline
from app.timeline.track import TrackKind
from app.transcription.models import AudioInfo, ProviderInfo, Sentence, Transcript, TranscriptionState, Word

ALL_KINDS = frozenset(SNAP_KINDS)


# ---------------------------------------------------------------- builders
def mk(cid: str, track: str, start: float, dur: float, kind: str = "media", **kw) -> Clip:
    return Clip(id=cid, track_id=track, asset_id=kw.pop("asset_id", "a1"), timeline_start=start, duration=dur, kind=kind, **kw)


def mk_transcript(words: list[tuple[float, float]], sentences: list[tuple[float, float]]) -> Transcript:
    ws = [Word(f"w{i}", "x", s, e) for i, (s, e) in enumerate(words)]
    ss = [Sentence(f"s{i}", "x", s, e, [ws[0].word_id] if ws else []) for i, (s, e) in enumerate(sentences)]
    return Transcript("t1", AudioInfo("aud", 100.0), ws, ss, ProviderInfo("test"))


class Rig:
    """An engine wired to mutable state, the way the workspace wires it."""

    def __init__(self, tl: Timeline | None = None, *, fps: float = 30, playhead: float = 0.0, settings: SnapSettings | None = None, project=None) -> None:
        self.tl = tl or Timeline.default()
        self.project = project
        self.playhead = playhead
        self.settings = settings or SnapSettings(targets=ALL_KINDS)
        self.tb = Timebase(fps)
        self.engine = SnapEngine(lambda: self.tl, lambda: self.project, lambda: self.playhead, lambda: self.settings, lambda: self.tb)

    def only(self, *kinds: str) -> "Rig":
        self.settings = SnapSettings(targets=frozenset(kinds))
        return self

    def snap(self, t: float, pps: float = 100.0, **kw):
        return self.engine.snap(t, pps=pps, **kw)


# ---------------------------------------------------------------- settings
def test_settings_defaults_and_sanitising():
    s = SnapSettings()
    assert s.enabled and s.tolerance_frames == 4
    assert s.targets == {"clip_edges", "playhead", "markers", "range", "scenes", "captions"}  # sentences / words are opt-in
    assert not s.wants("words") and not s.wants("sentences") and s.wants("scenes")
    assert SnapSettings(tolerance_frames=999).tolerance_frames == 60
    assert SnapSettings(tolerance_frames=0).tolerance_frames == 1
    assert SnapSettings(tolerance_frames="x").tolerance_frames == 4
    assert SnapSettings(tolerance_frames=True).tolerance_frames == 4
    assert SnapSettings(targets=["markers", "bogus"]).targets == {"markers"}
    assert s.with_target("words", True).wants("words") and not s.with_target("scenes", False).wants("scenes")


def test_settings_dict_roundtrip_and_tolerant_loading():
    s = SnapSettings(False, 8, frozenset({"markers", "words"}))
    assert SnapSettings.from_dict(s.to_dict()) == s
    assert SnapSettings.from_dict(None) == SnapSettings()
    assert SnapSettings.from_dict({"targets": "nope", "tolerance_frames": None, "enabled": 0}) == SnapSettings(False)
    assert SnapSettings.from_dict({"targets": ["clip_edges", 5, "x"]}).targets == {"clip_edges"}


# ---------------------------------------------------------------- each target kind
def test_clip_edges_and_timeline_start():
    r = Rig().only("clip_edges")
    r.tl.insert_clip(mk("c1", "track_v1", 2.0, 3.0))
    r.tl.insert_clip(mk("c2", "track_a1", 6.0, 1.0))
    res = r.snap(5.02)
    assert (res.snapped, res.target_kind, res.target_time, res.time, res.target_id) == (True, "clip_edges", 5.0, 5.0, "c1")
    assert res.distance_frames == pytest.approx(0.6, abs=1e-6)
    assert r.snap(1.99).target_time == 2.0
    assert r.snap(0.03).target_time == 0.0 and r.snap(0.03).target_id == ""  # the timeline start behaves like an edge
    far = r.snap(4.0)
    assert not far.snapped and far.time == 4.0 and far.target_kind == "" and far.target_time is None
    assert r.snap(6.99).target_time == 7.0  # other tracks count too


def test_caption_edges_are_their_own_kind():
    r = Rig()
    r.tl.insert_clip(mk("cap1", "track_v6", 3.0, 1.0, kind="caption", asset_id=""))
    r.tl.insert_clip(mk("m1", "track_v1", 10.0, 1.0))
    r.only("clip_edges")
    assert not r.snap(3.01).snapped  # captions are not clip edges
    assert r.snap(10.01).snapped
    r.only("captions")
    res = r.snap(4.02)
    assert (res.snapped, res.target_kind, res.target_time, res.target_id) == (True, "captions", 4.0, "cap1")
    assert not r.snap(10.01).snapped


def test_markers_and_range():
    r = Rig()
    r.tl.insert_marker(Marker("mk1", 7.0))
    r.tl.insert_marker(Marker("mk2", 9.0))
    r.tl.set_range((12.0, 15.0))
    r.only("markers")
    assert r.snap(7.03).target_id == "mk1" and r.snap(9.0).target_time == 9.0 and not r.snap(12.0).snapped
    r.only("range")
    a, b = r.snap(12.02), r.snap(14.98)
    assert (a.target_kind, a.target_time, a.target_id) == ("range", 12.0, "range_in")
    assert (b.target_time, b.target_id) == (15.0, "range_out")
    r.tl.set_range(None)
    assert not r.snap(12.02).snapped


def test_playhead_and_skip_kinds():
    r = Rig(playhead=8.0).only("playhead")
    res = r.snap(8.05)
    assert (res.snapped, res.target_kind, res.target_time) == (True, "playhead", 8.0)
    r.playhead = 20.0
    assert not r.snap(8.05).snapped and r.snap(20.04).snapped  # read live on every query
    assert not r.snap(20.04, skip_kinds={"playhead"}).snapped  # dragging the playhead must not snap to itself


def test_scene_boundaries():
    scenes = [Scene("s1", "1", 0.0, 4.0), Scene("s2", "2", 4.0, 9.5)]
    r = Rig(project=SimpleNamespace(scenes=scenes, transcription=TranscriptionState())).only("scenes")
    assert r.snap(4.03).target_time == 4.0 and r.snap(9.52).target_time == 9.5 and r.snap(0.02).target_time == 0.0
    assert not r.snap(6.0).snapped
    assert r.snap(4.03).target_kind == "scenes"


def test_scene_boundaries_off_grid_are_reported_not_rounded():
    r = Rig(project=SimpleNamespace(scenes=[Scene("s1", "1", 0.0, 3.217)], transcription=TranscriptionState())).only("scenes")
    res = r.snap(3.21)
    assert res.snapped and res.time == 3.217 and res.frame_aligned is False
    r.project = SimpleNamespace(scenes=[Scene("s1", "1", 0.0, 3.2)], transcription=TranscriptionState())  # 96 frames at 30 fps
    assert r.snap(3.21).frame_aligned is True


def test_sentence_and_word_boundaries_are_opt_in():
    tr = mk_transcript(words=[(1.0, 1.4), (1.5, 2.0), (2.6, 3.0)], sentences=[(1.0, 2.0), (2.6, 3.0)])
    project = SimpleNamespace(scenes=[], transcription=TranscriptionState(transcript=tr))
    r = Rig(project=project, settings=SnapSettings())  # default settings: sentences / words off
    assert not r.snap(2.6).snapped and not r.snap(1.5).snapped
    r.settings = SnapSettings(targets={"sentences"})
    assert r.snap(2.62).target_kind == "sentences" and r.snap(2.62).target_time == 2.6 and not r.snap(1.5).snapped
    r.settings = SnapSettings(targets={"words"})
    assert r.snap(1.52).target_kind == "words" and r.snap(1.52).target_time == 1.5 and r.snap(1.52).target_id == "w1"


def test_missing_project_or_timeline_never_raises():
    e = SnapEngine(lambda: None, lambda: None, lambda: 0.0, lambda: SnapSettings(targets=ALL_KINDS), lambda: Timebase(30))
    assert not e.snap(1.0, pps=100).snapped and e.candidates(1.0, pps=100) == []
    assert e.target_times("clip_edges") == [] and e.target_times("playhead") == [0.0]
    r = Rig(project=SimpleNamespace(scenes=None, transcription=None)).only("scenes", "words", "sentences")
    assert not r.snap(1.0).snapped
    assert not r.snap(float("nan")).snapped
    assert r.engine.candidates(float("nan"), pps=100) == []


# ---------------------------------------------------------------- priority and ties
def _loaded_rig() -> Rig:
    """Every kind has a target near t=10 (frames: 30 fps; tolerance 4 frames = 0.133 s)."""
    scenes = [Scene("sc", "1", 0.0, 10.04)]
    tr = mk_transcript(words=[(9.0, 10.05)], sentences=[(9.0, 10.06)])
    r = Rig(playhead=10.07, project=SimpleNamespace(scenes=scenes, transcription=TranscriptionState(transcript=tr)))
    r.tl.insert_clip(mk("clip", "track_v1", 5.0, 5.03))  # edge at 10.03
    r.tl.insert_clip(mk("cap", "track_v6", 10.02, 1.0, kind="caption", asset_id=""))
    r.tl.insert_marker(Marker("mk", 10.08))
    r.tl.set_range((10.01, 20.0))
    return r


def test_priority_order_playhead_markers_clip_edges_scenes_captions_range_sentences_words():
    r = _loaded_rig()
    assert list(PRIORITY) == ["playhead", "markers", "clip_edges", "scenes", "captions", "range", "sentences", "words"]
    expected = ["playhead", "markers", "clip_edges", "scenes", "captions", "range", "sentences", "words"]
    kinds = set(expected)
    for want in expected:
        r.settings = SnapSettings(targets=frozenset(kinds))
        res = r.snap(10.0)
        assert (res.snapped, res.target_kind) == (True, want)  # the farther higher-priority target beats the nearer lower-priority ones
        kinds.discard(want)
    r.settings = SnapSettings(targets=frozenset())
    assert not r.snap(10.0).snapped


def test_higher_priority_beyond_tolerance_does_not_win():
    r = Rig(playhead=10.5)
    r.tl.insert_clip(mk("c", "track_v1", 5.0, 5.0))
    r.settings = SnapSettings(targets={"playhead", "clip_edges"})
    assert r.snap(10.0).target_kind == "clip_edges"  # the playhead is half a second away


def test_same_kind_ties_nearest_then_earlier():
    r = Rig().only("markers")
    r.settings = SnapSettings(tolerance_frames=8, targets={"markers"})  # 0.267 s = 26.7 px at 100 px/s
    r.tl.insert_marker(Marker("a", 10.0))
    r.tl.insert_marker(Marker("b", 10.5))
    assert r.snap(10.3).target_id == "b"  # nearer
    assert r.snap(10.2).target_id == "a"
    assert r.snap(10.25).target_id == "a"  # exactly halfway (all values exact in binary) -> the earlier target
    r.tl.insert_marker(Marker("a2", 10.0))
    assert r.snap(10.1).target_time == 10.0


def test_candidates_are_ordered_by_priority_distance_time():
    r = _loaded_rig()
    cands = r.engine.candidates(10.0, pps=100)
    assert [c.kind for c in cands] == ["playhead", "markers", "clip_edges", "scenes", "captions", "range", "sentences", "words"]
    assert r.snap(10.0).target_kind == cands[0].kind and r.snap(10.0).target_time == cands[0].time
    r.settings = SnapSettings(targets={"markers"})
    r.tl.insert_marker(Marker("far", 10.1))
    r.tl.insert_marker(Marker("near", 9.99))
    got = r.engine.candidates(10.0, pps=100)
    assert [c.target_id for c in got] == ["near", "mk", "far"] and got[0].distance_frames == pytest.approx(0.3)


# ---------------------------------------------------------------- tolerance in frames
@pytest.mark.parametrize("fps", [24, 30, 29.97, 60])
@pytest.mark.parametrize("pps", [20.0, 100.0, 400.0])
def test_tolerance_is_frames_converted_through_pps_and_capped_in_pixels(fps, pps):
    r = Rig(fps=fps).only("clip_edges")
    r.tl.insert_clip(mk("c", "track_v1", 50.0, 10.0))
    frame = r.tb.frame_s
    expect_s = min(4 * frame, MAX_SNAP_PX / pps)
    assert r.engine.tolerance_seconds(pps) == pytest.approx(expect_s)
    assert r.engine.tolerance_px(pps) == pytest.approx(expect_s * pps) and r.engine.tolerance_px(pps) <= MAX_SNAP_PX + 1e-9
    assert r.snap(50.0 + expect_s * 0.98, pps).snapped
    assert not r.snap(50.0 + expect_s * 1.05, pps).snapped
    assert r.snap(50.0 - expect_s * 0.98, pps).snapped
    if 4 * frame < MAX_SNAP_PX / pps:  # frame-limited: the distance is reported in frames
        assert r.snap(50.0 + 2 * frame, pps).distance_frames == pytest.approx(2.0)


def test_known_conversions():
    r = Rig(fps=24)
    assert r.engine.tolerance_seconds(100) == pytest.approx(4 / 24) and r.engine.tolerance_px(100) == pytest.approx(16.6667, abs=1e-3)
    r = Rig(fps=29.97)
    assert r.engine.tolerance_seconds(100) == pytest.approx(4 * 1001 / 30000) and r.engine.tolerance_px(100) == pytest.approx(13.3467, abs=1e-3)
    r.settings = SnapSettings(tolerance_frames=8, targets=ALL_KINDS)
    assert r.engine.tolerance_px(100) == pytest.approx(26.693, abs=1e-3)
    assert r.engine.tolerance_px(2000) == pytest.approx(30.0)  # zoomed far in: capped at 30 px
    r.settings = SnapSettings(tolerance_frames=2, targets=ALL_KINDS)
    assert r.engine.tolerance_px(0.5) == pytest.approx(2 * 1001 / 30000 * 0.5)  # zoomed far out: a tolerance of 2 frames is a fraction of a pixel (frame based, no pixel floor)
    assert r.engine.tolerance_seconds(0) == pytest.approx(2 * 1001 / 30000)  # no scale known: plain frames


def test_zoomed_out_does_not_grab_everything():
    r = Rig().only("clip_edges")
    for i in range(100):
        r.tl.insert_clip(mk(f"c{i}", "track_v1", i * 1.0, 0.5))  # edges every half second
    # 1 px per second: 4 frames is 0.13 s, so a pointer 0.3 s from the nearest edge stays free
    assert not r.snap(0.8, 1.0).snapped and r.snap(1.04, 1.0).snapped


# ---------------------------------------------------------------- exclusion, locks, switches
def test_excluded_ids_never_snap_to_themselves_but_others_do():
    r = Rig().only("clip_edges")
    r.tl.insert_clip(mk("drag", "track_v1", 5.0, 2.0))
    r.tl.insert_clip(mk("other", "track_v2", 7.0, 1.0))  # starts where `drag` ends
    r.tl.insert_clip(mk("far", "track_v2", 20.0, 1.0))
    assert r.snap(5.02).target_id == "drag"
    assert not r.snap(5.02, exclude_ids={"drag"}).snapped
    res = r.snap(7.02, exclude_ids={"drag"})
    assert res.snapped and res.target_id == "other"  # the shared time is still a target through the other clip
    assert r.snap(7.02, exclude_ids=["drag", "other"]).snapped is False
    assert r.snap(20.02, exclude_ids=("drag", "other")).target_id == "far"


def test_excluded_run_next_to_the_pointer_is_skipped():
    r = Rig().only("clip_edges")
    for i in range(10):
        r.tl.insert_clip(mk(f"s{i}", "track_v1" if i % 2 == 0 else "track_v2", 10.0 + i * 0.001, 1.0))
    ex = {f"s{i}" for i in range(9)}
    res = r.snap(10.0, exclude_ids=ex)
    assert res.target_id in ("s9",) and res.target_time == pytest.approx(10.009)


def test_locked_clip_edges_are_valid_targets():
    r = Rig().only("clip_edges")
    r.tl.insert_clip(mk("lk", "track_v1", 5.0, 2.0, locked=True))
    r.tl.insert_clip(mk("lk2", "track_v2", 9.0, 1.0, locks=frozenset({"item"})))
    r.tl.get_track("track_v1").locked = True
    assert r.snap(7.02).target_id == "lk" and r.snap(9.01).target_id == "lk2"


def test_disabled_and_force():
    r = Rig(playhead=3.0)
    r.tl.insert_clip(mk("c", "track_v1", 5.0, 2.0))
    r.settings = SnapSettings(enabled=False, targets=ALL_KINDS)
    res = r.snap(5.01)
    assert not res.snapped and res.time == 5.01
    assert r.snap(5.01, force=True).snapped
    assert r.engine.candidates(5.01, pps=100) == [] and r.engine.snap_span(5.01, 6.0, pps=100).edge is None
    s = r.engine.session()
    assert not s.snap(5.01, pps=100).snapped and s.last_target is None


# ---------------------------------------------------------------- hysteresis
def test_hysteresis_no_oscillation_between_two_close_targets():
    r = Rig().only("markers")
    r.tl.insert_marker(Marker("A", 10.0))
    r.tl.insert_marker(Marker("B", 10.0 + 3 / 30))  # three frames apart; tolerance is four
    pps = 100.0
    fs = r.tb.frame_s

    def sweep(frames, session=True):
        s = r.engine.session()
        out = []
        for f in frames:
            t = f * fs
            res = s.snap(t, pps=pps) if session else r.snap(t, pps)
            out.append(res.target_id if res.snapped else "-")
        return out

    def switches(seq):
        pruned = [x for x in seq if x != "-"]
        return sum(1 for a, b in zip(pruned, pruned[1:]) if a != b)

    right = sweep(range(290, 321))
    assert switches(right) == 1 and right[:4] == ["-"] * 4 and right[-1] == "-"
    assert "A" in right and "B" in right and right.index("A") < right.index("B")
    left = sweep(range(320, 289, -1))
    assert switches(left) == 1 and left.index("B") < left.index("A")
    # sticky: halfway frame 301 is nearer to B (2 frames) than A (1 frame...) - the session keeps whichever it came from
    assert right[302 - 290] == "A" and left[320 - 301] == "B"  # still stuck where the plain nearest target is already the other one
    assert r.snap(301 * fs, pps).target_id == "A" and r.snap(302 * fs, pps).target_id == "B"  # plain nearest
    # jitter around the midpoint: plain nearest flips back and forth, the session does not
    wiggle = [301 + (k % 2) for k in range(60)]
    assert switches(sweep(wiggle, session=False)) >= 20
    assert switches(sweep(wiggle, session=True)) == 0


def test_session_releases_beyond_one_and_a_half_tolerances_and_resets():
    r = Rig().only("markers")
    r.tl.insert_marker(Marker("A", 10.0))
    pps, tol = 100.0, 4 / 30
    s = r.engine.session()
    assert s.snap(10.0 + 0.9 * tol, pps=pps).target_id == "A"
    inside = s.snap(10.0 + 1.4 * tol, pps=pps)  # outside the tolerance but within 1.5x: still stuck
    assert inside.snapped and inside.time == 10.0 and inside.distance_frames > 4
    assert STICKY_FACTOR == 1.5
    assert not s.snap(10.0 + 1.6 * tol, pps=pps).snapped and s.last_target is None
    assert not s.snap(10.0 + 1.4 * tol, pps=pps).snapped  # a fresh query needs the plain tolerance again
    s.snap(10.0, pps=pps)
    s.reset()
    assert s.last_target is None


def test_session_drops_a_target_that_disappears_or_gets_excluded():
    r = Rig().only("markers")
    r.tl.insert_marker(Marker("A", 10.0))
    s = r.engine.session()
    assert s.snap(10.0, pps=100).snapped
    assert not s.snap(10.01, pps=100, exclude_ids={"A"}).snapped
    s.snap(10.0, pps=100)
    r.tl.remove_marker("A")
    assert not s.snap(10.01, pps=100).snapped


# ---------------------------------------------------------------- span snapping
def test_snap_span_uses_the_closer_edge():
    r = Rig().only("clip_edges")
    r.tl.insert_clip(mk("x", "track_v1", 0.0, 10.0))
    r.tl.insert_clip(mk("y", "track_v1", 20.0, 5.0))
    sp = r.engine.snap_span(9.95, 14.95, pps=100)  # start is 0.05 from x's end
    assert (sp.edge, sp.start, sp.end, sp.snapped) == ("start", 10.0, 15.0, True)
    assert sp.delta == pytest.approx(0.05) and sp.result.target_id == "x"
    sp = r.engine.snap_span(14.0, 19.97, pps=100)  # end is 0.03 from y's start
    assert (sp.edge, sp.end) == ("end", 20.0) and sp.start == pytest.approx(14.03)
    assert sp.end - sp.start == pytest.approx(19.97 - 14.0)  # rigid: the duration never changes
    none = r.engine.snap_span(12.0, 15.0, pps=100)
    assert (none.edge, none.start, none.end, none.snapped, none.delta) == (None, 12.0, 15.0, False, 0.0)


def test_snap_span_prefer_and_equal_distance_rules():
    r = Rig().only("clip_edges")
    r.tl.insert_clip(mk("x", "track_v1", 0.0, 10.0))
    r.tl.insert_clip(mk("y", "track_v1", 30.0, 5.0))
    near_both = (10.04, 29.97)  # start is 0.04 and end 0.03 away
    assert r.engine.snap_span(*near_both, pps=100).edge == "end"
    assert r.engine.snap_span(*near_both, pps=100, prefer="start").edge == "start"
    assert r.engine.snap_span(*near_both, pps=100, prefer="end").edge == "end"
    only_start = r.engine.snap_span(10.04, 20.0, pps=100, prefer="end")
    assert only_start.edge == "start"  # the preferred edge has nothing: fall back to the other
    with pytest.raises(ValueError):
        r.engine.snap_span(1, 2, pps=100, prefer="middle")
    # exactly equal distance: higher priority kind wins, otherwise the start edge
    r.settings = SnapSettings(targets={"clip_edges", "markers"})
    r.tl.insert_marker(Marker("mk", 40.02))
    r.tl.insert_clip(mk("z", "track_v2", 60.0, 1.0))
    eq = r.engine.snap_span(10.02, 40.0, pps=100)
    assert eq.edge == "end" and eq.result.target_kind == "markers"  # both 0.02 away; markers outrank clip edges
    eq = r.engine.snap_span(10.02, 20.0 + 0.0, pps=100)
    assert eq.edge == "start"


def test_multi_clip_drag_snaps_the_union_span_and_excludes_all_dragged_clips():
    r = Rig().only("clip_edges")
    a, b = mk("a", "track_v1", 4.0, 2.0), mk("b", "track_v2", 7.0, 3.0)  # the selection spans 4..10
    far = mk("far", "track_v1", 20.0, 1.0)
    for c in (a, b, far):
        r.tl.insert_clip(c)
    union = (min(a.timeline_start, b.timeline_start), max(a.timeline_end, b.timeline_end))
    sp = r.engine.snap_span(union[0] + 9.97, union[1] + 9.97, exclude_ids={"a", "b"}, pps=100)  # dragged right by ~10 s: the front edge (19.97) is near `far`'s start
    assert (sp.snapped, sp.edge, sp.result.target_id) == (True, "end", "far")
    assert sp.end == 20.0 and sp.start == pytest.approx(14.0) and sp.delta == pytest.approx(0.03)
    shifted = r.engine.snap_span(4.02, 10.02, exclude_ids={"a", "b"}, pps=100)
    assert not shifted.snapped  # the dragged clips' own edges are not targets
    assert r.engine.snap_span(4.02, 10.02, pps=100).snapped


def test_span_session_hysteresis():
    r = Rig().only("markers")
    r.tl.insert_marker(Marker("A", 10.0))
    r.tl.insert_marker(Marker("B", 10.1))
    s = r.engine.session()
    fs = r.tb.frame_s
    first = s.snap_span(300 * fs - 5.0, 300 * fs, pps=100)  # end on A
    assert first.edge == "end" and first.result.target_id == "A"
    again = s.snap_span(302 * fs - 5.0, 302 * fs, pps=100)  # plain nearest would say B (one frame away); the session keeps A
    assert again.result.target_id == "A" and again.end == pytest.approx(10.0)
    plain = r.engine.snap_span(302 * fs - 5.0, 302 * fs, pps=100)
    assert plain.result.target_id == "B"


# ---------------------------------------------------------------- caching
def test_cache_invalidation_when_things_change():
    scenes = [Scene("s1", "1", 0.0, 4.0)]
    tr = mk_transcript(words=[(1.0, 1.4)], sentences=[(1.0, 1.4)])
    project = SimpleNamespace(scenes=scenes, transcription=TranscriptionState(transcript=tr))
    r = Rig(project=project)
    c = mk("c", "track_v1", 5.0, 2.0)
    r.tl.insert_clip(c)
    assert r.snap(7.02).target_time == 7.0
    c.timeline_start = 6.0  # in-place geometry change
    assert not r.snap(7.02).snapped and r.snap(8.02).target_time == 8.0
    r.tl.detach_clip("c")
    assert not r.snap(8.02).snapped
    r.tl.insert_clip(mk("d", "track_v1", 30.0, 1.0))
    assert r.snap(30.01).target_id == "d"
    # markers: add, remove, and an in-place time edit
    m = Marker("m", 50.0)
    r.tl.insert_marker(m)
    assert r.snap(50.02).target_kind == "markers"
    m.time = 55.0
    assert not r.snap(50.02).snapped and r.snap(55.02).target_id == "m"
    r.tl.remove_marker("m")
    assert not r.snap(55.02).snapped
    # range
    r.tl.set_range((60.0, 70.0))
    assert r.snap(60.02).target_kind == "range"
    r.tl.set_range((61.0, 70.0))
    assert not r.snap(60.02).snapped and r.snap(61.02).target_kind == "range"
    # scenes: a boundary edited in place, a scene added
    scenes[0].end = 4.5
    assert r.snap(4.52).target_time == 4.5 and not r.snap(4.02).snapped
    scenes.append(Scene("s2", "2", 4.5, 90.0))
    assert r.snap(90.01).target_time == 90.0
    # transcript replaced: new identity -> rebuilt
    r.settings = SnapSettings(targets={"words", "sentences"})
    assert r.snap(1.42).target_kind == "sentences" and r.snap(1.42).target_time == 1.4
    project.transcription = TranscriptionState(transcript=mk_transcript(words=[(2.0, 2.2)], sentences=[(2.0, 2.2)]))
    assert not r.snap(1.42).snapped and r.snap(2.21).snapped
    # explicit invalidation for in-place edits the keys cannot see
    project.transcription.transcript.words[0].end = 2.3
    r.engine.invalidate("words")
    r.settings = SnapSettings(targets={"words"})
    assert r.snap(2.31).target_time == 2.3
    # project switch: another timeline object
    r.settings = SnapSettings(targets={"clip_edges"})
    assert r.snap(30.01).target_id == "d"
    r.tl = Timeline.default()
    assert not r.snap(30.01).snapped and r.snap(0.01).target_time == 0.0


def test_caches_build_once_per_state_and_not_per_query():
    r = Rig(project=SimpleNamespace(scenes=[Scene("s", "1", 1.0, 2.0)], transcription=TranscriptionState(transcript=mk_transcript([(1, 2)], [(1, 2)]))))
    r.tl.insert_clip(mk("c", "track_v1", 5.0, 1.0))
    r.tl.insert_marker(Marker("m", 3.0))
    r.tl.set_range((7.0, 8.0))
    for i in range(500):
        r.snap(i * 0.02)
    assert dict(r.engine.builds) == {"clips": 1, "markers": 1, "range": 1, "scenes": 1, "sentences": 1, "words": 1}


def test_5000_clip_timeline_ten_thousand_queries_rebuild_a_bounded_number_of_times(monkeypatch):
    tl = Timeline.default()
    tracks = [t for t in tl.tracks if t.kind in (TrackKind.VIDEO, TrackKind.AUDIO)]
    n = 5000
    per = n // len(tracks)
    for ti, t in enumerate(tracks):
        t.clips = [mk(f"c{ti}_{i}", t.id, i * 2.0, 1.5) for i in range(per)]
    assert len(tl.all_clips()) == per * len(tracks)
    r = Rig(tl)
    builds = []
    orig = SnapEngine._build_clip_pass

    def spy(self, timeline):
        builds.append(1)
        return orig(self, timeline)

    monkeypatch.setattr(SnapEngine, "_build_clip_pass", spy)
    rng = random.Random(7)
    r.snap(10.0)  # warm-up builds the arrays
    assert len(builds) == 1
    monkeypatch.setattr(Timeline, "all_clips", lambda self: (_ for _ in ()).throw(AssertionError("a query scanned every clip")))
    monkeypatch.setattr(Timeline, "snap_points", lambda self, *a: (_ for _ in ()).throw(AssertionError("a query rebuilt the snap list")))
    hits = 0
    for _ in range(10_000):
        hits += r.snap(rng.uniform(0, per * 2.0), pps=rng.choice([5, 50, 500]), exclude_ids={"c0_3"}).snapped
    assert len(builds) == 1 and hits > 0
    monkeypatch.undo()
    monkeypatch.setattr(SnapEngine, "_build_clip_pass", spy)
    tl.tracks[0].clips[0].timeline_start = 0.25  # one edit -> exactly one rebuild, however many queries follow
    for _ in range(1000):
        r.snap(rng.uniform(0, 100))
    assert len(builds) == 2
    # the edited clip's edge is now a target at 0.25
    assert r.snap(0.26).target_time == 0.25


# ---------------------------------------------------------------- brute-force property test
def _brute(r: Rig, t: float, pps: float, ex: set[str], skip: set[str]):
    tl, s = r.tl, r.settings
    tol = r.engine.tolerance_seconds(pps)
    pts: dict[str, list[float]] = {k: [] for k in SNAP_KINDS}
    pts["playhead"] = [r.playhead]
    pts["markers"] = [m.time for m in tl.markers if m.id not in ex]
    pts["clip_edges"] = [0.0] + [v for c in tl.all_clips() if c.kind != "caption" and c.id not in ex for v in (c.timeline_start, c.timeline_end)]
    pts["captions"] = [v for c in tl.all_clips() if c.kind == "caption" and c.id not in ex for v in (c.timeline_start, c.timeline_end)]
    pts["range"] = list(tl.range) if tl.range else []
    pts["scenes"] = [v for sc in r.project.scenes for v in (sc.start, sc.end)]
    tr = r.project.transcription.transcript
    pts["sentences"] = [v for x in tr.sentences for v in (x.start, x.end)]
    pts["words"] = [v for x in tr.words for v in (x.start, x.end)]
    best = None
    for kind in SNAP_KINDS:
        if kind not in s.targets or kind in skip or not s.enabled:
            continue
        near = [(abs(v - t), v) for v in pts[kind] if abs(v - t) <= tol + 1e-9]
        if near:
            d, v = min(near)  # nearest, then the earlier time
            cand = (PRIORITY[kind], d, v, kind)
            best = cand if best is None else min(best, cand)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_random_queries_agree_with_a_brute_force_model(seed):
    rng = random.Random(seed)
    tl = Timeline.default()
    ids = []
    for i in range(rng.randint(10, 60)):
        tid = rng.choice(["track_v1", "track_v2", "track_v3", "track_a1", "track_v6"])
        kind = "caption" if tid == "track_v6" else "media"
        c = mk(f"c{i}", tid, round(rng.uniform(0, 60), 3), round(rng.uniform(0.2, 8), 3), kind=kind)
        tl.insert_clip(c)
        ids.append(c.id)
    for i in range(rng.randint(0, 6)):
        tl.insert_marker(Marker(f"m{i}", round(rng.uniform(0, 60), 3)))
        ids.append(f"m{i}")
    if rng.random() < 0.7:
        a = round(rng.uniform(0, 30), 3)
        tl.set_range((a, a + round(rng.uniform(1, 30), 3)))
    scenes = [Scene(f"sc{i}", str(i), round(i * 7.3, 3), round((i + 1) * 7.3, 3)) for i in range(8)]
    words = [(round(i * 0.37, 3), round(i * 0.37 + 0.3, 3)) for i in range(150)]
    sents = [(round(i * 3.7, 3), round(i * 3.7 + 3.5, 3)) for i in range(15)]
    project = SimpleNamespace(scenes=scenes, transcription=TranscriptionState(transcript=mk_transcript(words, sents)))
    r = Rig(tl, fps=rng.choice([24, 29.97, 30, 60]), playhead=round(rng.uniform(0, 60), 3), project=project)
    checked = snapped = 0
    for q in range(400):
        r.settings = SnapSettings(rng.random() < 0.9, rng.choice([2, 4, 8, 3, 15]), frozenset(k for k in SNAP_KINDS if rng.random() < 0.7))
        if q % 40 == 0:
            r.playhead = round(rng.uniform(0, 60), 3)
        t = round(rng.uniform(-1, 62), 3)
        pps = rng.choice([2, 20, 100, 300, 1500])
        ex = set(rng.sample(ids, rng.randint(0, min(8, len(ids)))))
        skip = {"playhead"} if rng.random() < 0.2 else set()
        want = _brute(r, t, pps, ex, skip)
        got = r.snap(t, pps, exclude_ids=ex, skip_kinds=skip)
        checked += 1
        if want is None:
            assert not got.snapped and got.time == t, (seed, q)
        else:
            snapped += 1
            assert got.snapped and got.target_kind == want[3] and got.target_time == want[2] and got.time == want[2], (seed, q, got, want)
            assert got.distance_frames == pytest.approx(want[1] / r.tb.frame_s)
            cands = r.engine.candidates(t, exclude_ids=ex, pps=pps, skip_kinds=skip)
            assert cands and cands[0].kind == got.target_kind and cands[0].time == got.target_time
    assert checked == 400 and snapped > 20


def test_random_session_walks_follow_the_stickiness_model():
    rng = random.Random(99)
    r = Rig().only("markers", "clip_edges")
    for i in range(40):
        r.tl.insert_marker(Marker(f"m{i}", round(rng.uniform(0, 20), 3)))
    for i in range(25):
        r.tl.insert_clip(mk(f"c{i}", rng.choice(["track_v1", "track_v2"]), round(rng.uniform(0, 20), 3), round(rng.uniform(0.1, 1.0), 3)))
    pps = 100.0
    tol = r.engine.tolerance_seconds(pps)
    s = r.engine.session()
    t = 10.0
    last = None
    for _ in range(600):
        t = min(20.0, max(0.0, t + rng.uniform(-0.08, 0.08)))
        got = s.snap(t, pps=pps)
        plain = r.snap(t, pps)
        if last is not None and abs(last.target_time - t) <= tol * STICKY_FACTOR + 1e-9:
            assert got.snapped and got.target_time == last.target_time and got.target_kind == last.target_kind  # stuck
        else:
            assert (got.snapped, got.target_time, got.target_kind) == (plain.snapped, plain.target_time, plain.target_kind)
        if got.snapped:
            assert abs(got.target_time - t) <= tol * STICKY_FACTOR + 1e-9
        last = got if got.snapped else None
