"""Phase 9: the lazily built timeline index must agree with brute-force scans after every kind of mutation (commands, undo/redo, in-place edits, reopen)."""

from __future__ import annotations

import bisect
import copy
import random

import pytest

from app.core.commands import CommandStack, CompositeCommand
from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline import index as ix_mod
from app.timeline.clip import Clip
from app.timeline.timeline import Timeline, new_clip_id, new_track_id
from app.timeline.timeline_commands import (
    AddClipCommand,
    AddTrackCommand,
    DeleteClipCommand,
    MoveClipCommand,
    RemoveTrackCommand,
    SetClipPropertiesCommand,
    SetTrackFlagCommand,
    SplitClipCommand,
    TrimClipCommand,
)
from app.timeline.track import Track, TrackKind


# ---------------------------------------------------------------- brute force reference (the pre-Phase-9 algorithms, written against the raw lists)
def b_find(tl: Timeline, cid):
    for t in tl.tracks:
        for c in t.clips:
            if c.id == cid:
                return t, c
    return None


def b_all(tl):
    return [c for t in tl.tracks for c in t.clips]


def b_duration(tl):
    return max((c.timeline_end for c in b_all(tl)), default=0.0)


def b_free(tl, tid, start, dur, ignore=None) -> bool:
    end = start + dur
    for c in next(t for t in tl.tracks if t.id == tid).clips:
        if c.id != ignore and start < c.timeline_end - TIME_EPSILON and end > c.timeline_start + TIME_EPSILON:
            return False
    return True


def b_first_free(tl, tid, start, dur):
    t = max(0.0, start)
    for c in next(x for x in tl.tracks if x.id == tid).clips:
        if t + dur <= c.timeline_start + TIME_EPSILON:
            break
        if t < c.timeline_end - TIME_EPSILON:
            t = c.timeline_end
    return t


def b_neighbours(tl, clip):
    prev_end, next_start = 0.0, None
    for c in next(x for x in tl.tracks if x.id == clip.track_id).clips:
        if c.id == clip.id:
            continue
        if c.timeline_end <= clip.timeline_start + TIME_EPSILON:
            prev_end = max(prev_end, c.timeline_end)
        elif c.timeline_start >= clip.timeline_end - TIME_EPSILON:
            next_start = c.timeline_start if next_start is None else min(next_start, c.timeline_start)
    return prev_end, next_start


def b_range(tl, tid, t0, t1):
    tracks = [t for t in tl.tracks if tid is None or t.id == tid]
    return {c.id for t in tracks for c in t.clips if c.timeline_end > t0 and c.timeline_start < t1}


def b_snap(tl, exclude):
    return sorted(v for c in b_all(tl) if c.id != exclude for v in (c.timeline_start, c.timeline_end))


def make_timeline(rng: random.Random, n_tracks=4, per_track=12) -> Timeline:
    tl = Timeline([Track(f"t{i}", f"T{i}", TrackKind.VIDEO) for i in range(n_tracks)])
    for t in tl.tracks:
        x = rng.uniform(0, 2)
        for _ in range(per_track):
            d = round(rng.uniform(0.3, 3.0), 3)
            t.clips.append(Clip(new_clip_id(), t.id, rng.choice(["a1", "a2", "a3", ""]), round(x, 3), d, 0.0, d))
            x += d + rng.choice([0.0, 0.0, rng.uniform(0.05, 2.0)])
        t.sort()
    return tl


def verify(tl: Timeline, rng: random.Random, force: bool) -> None:
    """Compare every public answer with brute force. ``force`` builds the index first so the indexed paths are what is checked."""
    if force:
        tl.index()
        for _ in range(8):  # also make sure the "stale until looked at often" path is covered by the unforced run
            tl.duration  # noqa: B018
    clips = b_all(tl)
    ids = [c.id for c in clips]
    assert [c.id for c in tl.all_clips()] == ids
    assert tl.duration == b_duration(tl)
    for cid in rng.sample(ids, min(12, len(ids))) + ["no-such-clip"]:
        want = b_find(tl, cid)
        if want is None:
            with pytest.raises(TimelineError, match="That clip no longer exists"):
                tl.find_clip(cid)
            assert tl.get_clip(cid) is None
        else:
            got = tl.find_clip(cid)
            assert got[0] is want[0] and got[1] is want[1]
            assert tl.get_clip(cid) is want[1]
    for aid in ("a1", "a2", "a3", "", "zzz"):
        assert [c.id for c in tl.clips_for_asset(aid)] == [c.id for c in clips if c.asset_id == aid]
    for t in tl.tracks:
        for _ in range(10):
            start = round(rng.uniform(-0.5, tl.duration + 2), 3)
            dur = round(rng.uniform(0.0, 4.0), 3)
            ignore = rng.choice(ids) if ids and rng.random() < 0.4 else None
            ok = b_free(tl, t.id, start, dur, ignore) if start >= -TIME_EPSILON and dur >= MIN_CLIP_DURATION - TIME_EPSILON else None
            try:
                tl.check_free(t.id, start, dur, ignore)
                assert ok is True
            except TimelineError as exc:
                assert ok is not True, exc
            if dur > 0:
                assert tl.first_free_start(t.id, start, dur) == b_first_free(tl, t.id, start, dur)
        for c in t.clips[:: max(1, len(t.clips) // 6)]:
            assert tl.neighbours(c) == b_neighbours(tl, c)
        for _ in range(6):
            a = rng.uniform(-1, tl.duration + 1)
            b = a + rng.uniform(0, 12)
            assert {c.id for c in tl.clips_in_range(t.id, a, b)} == b_range(tl, t.id, a, b)
            starts = [c.timeline_start for c in tl.clips_in_range(t.id, a, b)]
            assert starts == sorted(starts)
    a = rng.uniform(0, max(1.0, tl.duration))
    assert {c.id for c in tl.clips_in_range(None, a, a + 5)} == b_range(tl, None, a, a + 5)
    ex = rng.choice(ids) if ids else None
    assert tl.snap_points(ex) == b_snap(tl, ex)
    assert tl.snap_points() == b_snap(tl, None)
    assert tl.snap_points(ex) is not tl.snap_points(ex)  # callers may keep / mutate their copy
    with pytest.raises(TimelineError):
        tl.check_free("no-such-track", 0, 1)


# ---------------------------------------------------------------- the randomised run
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_randomised_operations_match_brute_force(seed):
    rng = random.Random(seed)
    tl = make_timeline(rng)
    stack = CommandStack()
    verify(tl, rng, True)
    performed = {"add": 0, "move": 0, "trim": 0, "split": 0, "delete": 0, "ripple": 0, "track": 0, "undo": 0, "redo": 0, "lock": 0, "props": 0, "raw": 0}

    def pick():
        cs = b_all(tl)
        return rng.choice(cs) if cs else None

    for step in range(260):
        r = rng.random()
        c = pick()
        try:
            if r < 0.12:
                t = rng.choice(tl.tracks)
                d = round(rng.uniform(0.3, 3), 3)
                stack.execute(AddClipCommand(tl, Clip(new_clip_id(), t.id, rng.choice(["a1", "a2"]), tl.first_free_start(t.id, rng.uniform(0, tl.duration + 3), d), d, 0.0, d)))
                performed["add"] += 1
            elif r < 0.30 and c:
                new_track = rng.choice(tl.tracks).id if rng.random() < 0.4 else None
                stack.execute(MoveClipCommand(tl, c.id, round(rng.uniform(0, tl.duration + 2), 3), new_track))
                performed["move"] += 1
            elif r < 0.42 and c:
                if rng.random() < 0.5:
                    stack.execute(TrimClipCommand(tl, c.id, new_start=c.timeline_start + rng.uniform(-1, 1)))
                else:
                    stack.execute(TrimClipCommand(tl, c.id, new_end=c.timeline_end + rng.uniform(-1, 1)))
                performed["trim"] += 1
            elif r < 0.52 and c:
                stack.execute(SplitClipCommand(tl, c.id, c.timeline_start + c.duration * rng.uniform(0.2, 0.8), new_clip_id()))
                performed["split"] += 1
            elif r < 0.60 and c:
                stack.execute(DeleteClipCommand(tl, c.id))
                performed["delete"] += 1
            elif r < 0.66 and c:  # ripple delete: remove the clip and pull every later clip on its track left by its length
                track, clip = tl.find_clip(c.id)
                cmds = [DeleteClipCommand(tl, clip.id)] + [MoveClipCommand(tl, k.id, k.timeline_start - clip.duration) for k in track.clips if k.timeline_start >= clip.timeline_end - 1e-9 and k.id != clip.id]
                stack.execute(CompositeCommand("Ripple delete", cmds, scope="timeline"))
                performed["ripple"] += 1
            elif r < 0.70:
                if rng.random() < 0.5 or len(tl.tracks) < 3:
                    stack.execute(AddTrackCommand(tl, Track(new_track_id(), "New", TrackKind.VIDEO), rng.randint(0, len(tl.tracks))))
                else:
                    stack.execute(RemoveTrackCommand(tl, rng.choice(tl.tracks).id))
                performed["track"] += 1
            elif r < 0.74:
                t = rng.choice(tl.tracks)
                stack.execute(SetTrackFlagCommand(tl, t.id, "locked", not t.locked))
                performed["lock"] += 1
            elif r < 0.78 and c:
                stack.execute(SetClipPropertiesCommand(tl, c.id, speed=rng.choice([0.5, 1.0, 2.0])))
                performed["props"] += 1
            elif r < 0.88:
                stack.undo()
                performed["undo"] += 1
            elif r < 0.94:
                stack.redo()
                performed["redo"] += 1
            elif c:  # raw in-place edits the way assembly / presentation code does them
                kind = rng.randint(0, 3)
                tr = next(t for t in tl.tracks if t.id == c.track_id)
                if kind == 0:
                    c.timeline_start = round(c.timeline_start + rng.uniform(-0.2, 0.2), 3) % 60
                elif kind == 1:
                    c.duration = round(max(0.3, c.duration + rng.uniform(-0.3, 0.3)), 3)
                elif kind == 2:
                    tr.clips = [k for k in tr.clips if k.id != c.id]
                else:
                    tr.clips.append(Clip(new_clip_id(), tr.id, "a3", round(rng.uniform(0, 30), 3), 1.0, 0.0, 1.0))
                    tr.sort()
                stack = CommandStack()  # raw edits are not undoable: history before them no longer applies
                performed["raw"] += 1
        except TimelineError:
            pass  # rejected edits (locked track, overlap...) are part of the game
        verify(tl, rng, force=step % 2 == 0)
    assert sum(performed.values()) > 100 and all(performed[k] > 0 for k in ("add", "move", "trim", "split", "delete", "ripple", "track", "undo", "redo", "raw")), performed
    # the whole history rewinds and replays without the index ever disagreeing
    while stack.can_undo:
        stack.undo()
        verify(tl, rng, True)
    while stack.can_redo:
        stack.redo()
        verify(tl, rng, True)


def test_reopen_round_trip_and_copies_have_their_own_index():
    rng = random.Random(9)
    tl = make_timeline(rng, 5, 20)
    tl.index()
    again = Timeline.from_dict(copy.deepcopy(tl.to_dict()))
    verify(again, rng, True)
    assert again.duration == tl.duration and again.snap_points() == tl.snap_points()
    clone = copy.deepcopy(tl)  # QC / the fix engine deep-copy timelines: the index is derived data and must not be shared
    assert clone._ix is None
    verify(clone, rng, True)
    victim = clone.all_clips()[0]
    clone.detach_clip(victim.id)
    assert tl.get_clip(victim.id) is not None and clone.get_clip(victim.id) is None
    verify(tl, rng, True)
    verify(clone, rng, True)


def test_in_place_mutation_is_seen_without_any_explicit_touch():
    rng = random.Random(4)
    tl = make_timeline(rng, 3, 10)
    tl.index()
    c = tl.tracks[1].clips[3]
    before = tl.revision
    c.timeline_start += 100.0  # what assembly code does
    assert tl.revision != before
    assert tl.duration == b_duration(tl) >= 100.0
    c.track_id = "t2"  # identity edit alone does not move it between lists, but must not break any answer either
    verify(tl, rng, True)
    tl.tracks[0].clips.pop()
    verify(tl, rng, True)
    tl.tracks[2].clips[0] = Clip("swapped", "t2", "a1", 50.0, 2.0, 0.0, 2.0)
    assert tl.get_clip("swapped") is not None
    tl.tracks = [t for t in tl.tracks if t.id != "t1"]
    assert tl.get_clip(c.id) is None
    verify(tl, rng, True)


def test_locked_tracks_still_answer_queries_and_commands_still_refuse():
    rng = random.Random(5)
    tl = make_timeline(rng, 2, 6)
    stack = CommandStack()
    clip = tl.tracks[0].clips[2]
    stack.execute(SetTrackFlagCommand(tl, "t0", "locked", True))
    for cmd in (MoveClipCommand(tl, clip.id, 40.0), DeleteClipCommand(tl, clip.id), TrimClipCommand(tl, clip.id, new_end=clip.timeline_end - 0.1)):
        with pytest.raises(TimelineError, match="locked"):
            stack.execute(cmd)
    verify(tl, rng, True)
    assert tl.get_clip(clip.id) is clip
    stack.undo()
    stack.execute(MoveClipCommand(tl, clip.id, tl.duration + 5))
    assert tl.duration == tl.get_clip(clip.id).timeline_end
    verify(tl, rng, True)


def test_degenerate_clips_fall_back_to_scans_and_stay_correct():
    tl = Timeline([Track("t", "T", TrackKind.VIDEO)])
    tl.tracks[0].clips += [Clip("a", "t", "x", 0.0, 2.0), Clip("b", "t", "x", 1.0, 5.0), Clip("nan", "t", "x", float("nan"), 1.0)]
    tl.tracks[0].clips.pop()  # NaN start: a scan's answer is order dependent, so just make sure nothing raises
    for _ in range(8):
        tl.duration  # noqa: B018
        tl.check_free("t", 10.0, 1.0)
    tl.tracks[0].clips.append(Clip("neg", "t", "x", 3.0, -1.0))
    for _ in range(8):
        assert tl.duration == b_duration(tl)
    assert tl.find_clip("neg")[1].id == "neg"
    with pytest.raises(TimelineError):
        tl.check_free("t", 1.5, 1.0)  # overlapping clips on one track (bad data) are still detected


def test_overlapping_clips_on_one_track_are_handled():
    tl = Timeline([Track("t", "T", TrackKind.VIDEO)])
    tl.tracks[0].clips += [Clip("long", "t", "x", 0.0, 100.0), Clip("s1", "t", "x", 10.0, 1.0), Clip("s2", "t", "x", 20.0, 1.0)]
    tl.tracks[0].sort()
    tl.index()
    assert {c.id for c in tl.clips_in_range("t", 50.0, 60.0)} == {"long"}  # a long clip hides behind later short ones in a start-sorted array
    with pytest.raises(TimelineError):
        tl.check_free("t", 50.0, 1.0)
    assert tl.first_free_start("t", 5.0, 1.0) == b_first_free(tl, "t", 5.0, 1.0) == 100.0


# ---------------------------------------------------------------- scaling (operation counts, not wall-clock)
class _Counter:
    """Counts clip attribute reads during a call: a linear scan touches every clip, an indexed lookup a handful."""

    def __init__(self, tl: Timeline):
        self.tl = tl
        self.reads = 0
        self._orig = Clip.__getattribute__

    def __enter__(self):
        counter = self

        def counting(obj, name):
            if name in ("timeline_start", "duration", "id"):
                counter.reads += 1
            return counter._orig(obj, name)

        Clip.__getattribute__ = counting  # type: ignore[method-assign]
        return self

    def __exit__(self, *a):
        Clip.__getattribute__ = self._orig  # type: ignore[method-assign]


def _big(n: int) -> Timeline:
    tl = Timeline([Track(f"t{i}", f"T{i}", TrackKind.VIDEO) for i in range(5)])
    per = n // 5
    for t in tl.tracks:
        t.clips += [Clip(f"{t.id}_c{k}", t.id, f"a{k % 7}", k * 2.0, 1.5, 0.0, 1.5) for k in range(per)]
    return tl


def test_indexed_queries_do_a_bounded_amount_of_work_regardless_of_size():
    reads = {}
    for n in (500, 2000):
        tl = _big(n)
        tl.index()
        mid = (n // 5) * 1.0  # the middle of each track
        with _Counter(tl) as cnt:
            for _ in range(20):
                tl.check_free("t2", mid + 1.55, 0.3)
                tl.neighbours(tl.tracks[2].clips[len(tl.tracks[2].clips) // 2])
                tl.first_free_start("t2", mid, 0.4)  # fits in the next gap; a longer clip would legitimately walk the whole packed chain
                tl.find_clip(f"t3_c{n // 10}")
                tl.clips_in_range("t1", mid, mid + 20)
                tl.duration  # noqa: B018
        reads[n] = cnt.reads
    assert reads[2000] <= reads[500] * 1.5 + 50, reads  # 4x the clips, (almost) the same work
    tl = _big(2000)
    with _Counter(tl) as cnt:
        for _ in range(20):
            for c in tl.all_clips():  # the brute-force baseline for comparison
                c.timeline_end  # noqa: B018
    assert cnt.reads > 20 * 1500


def test_visible_range_query_cost_is_proportional_to_the_result():
    sizes = {}
    for n in (500, 4000):
        tl = _big(n)
        tl.index()
        out = tl.clips_in_range("t0", 30.0, 40.0)
        sizes[n] = len(out)
        with _Counter(tl) as cnt:
            tl.clips_in_range("t0", 30.0, 40.0)
        assert cnt.reads <= 4 * max(1, len(out)) + 8
    assert sizes[500] == sizes[4000]


def test_mutate_then_look_up_loops_do_not_rebuild_the_index_every_step():
    tl = _big(2000)
    tl.index()
    builds = []
    orig = ix_mod.TimelineIndex.__init__

    def counting(self, *a, **k):
        builds.append(1)
        orig(self, *a, **k)

    ix_mod.TimelineIndex.__init__ = counting  # type: ignore[method-assign]
    try:
        for k in range(60):  # assembly-style loop: insert, look up, insert...
            t = tl.tracks[k % 5]
            tl.insert_clip(Clip(f"n{k}", t.id, "a", 5000.0 + k * 3, 1.0, 0.0, 1.0))
            tl.find_clip(f"n{k}")
            tl.check_free(t.id, 6000.0 + k, 1.0)
        assert len(builds) <= 1
        for _ in range(10):  # read-heavy phase (paint / drag): one rebuild, then reuse
            tl.find_clip("t0_c5")
        assert len(builds) <= 2
    finally:
        ix_mod.TimelineIndex.__init__ = orig  # type: ignore[method-assign]
    assert tl.find_clip("n59")[1].id == "n59"


def test_revision_is_monotonic_and_touch_invalidates():
    tl = make_timeline(random.Random(2), 2, 3)
    r0 = tl.revision
    tl.touch()
    r1 = tl.revision
    tl.tracks[0].clips[0].duration += 1
    r2 = tl.revision
    assert r0 < r1 < r2
    ix = tl.index()
    assert tl.index() is ix
    tl.touch()
    assert tl.index() is not ix


def test_bisect_helper_agrees_with_stdlib_on_snap_nearest():
    tl = _big(500)
    pts = tl.snap_points()
    assert pts == sorted(pts)
    for t in (0.0, 13.37, 99.99, 1e6):
        i = bisect.bisect_left(pts, t)
        best = min(pts[max(0, i - 1): i + 1], key=lambda q: abs(q - t))
        assert best == min(pts, key=lambda q: abs(q - t))
