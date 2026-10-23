"""Ripple, roll, slip and slide planners (Phase 10).

Every function is a PURE planner ``plan_xxx(env, ...) -> EditPlan``: it reads ``env.timeline``, never mutates it, and returns the complete result - the primary change *and*
every clip it pushes, trims or splits - as ONE plan, so a ripple over a thousand clips is a single undo step that either fully happens or does not happen at all
(``PlanCommand`` validates the whole result first). Impossible requests raise ``TimelineError`` with a plain-language reason; every adjustment (a clamp at a source limit,
a fragment too short to keep, a locked track that was left out) is explained in ``plan.notes``.

Vocabulary (see ``ops_common``): LINKED clips (``link_id``) follow ripple / roll / slip / slide; GROUPED clips (``group_id``) follow when a clip is shifted or deleted but a trim
stays independent; LOCKED tracks / clips are never touched.

Ripple scope (``RIPPLE_SCOPES``)
  * ``"track"``   only the track of the edited clip(s);
  * ``"tracks"``  the edited clip's track plus an explicit list (``selected_track_ids``: normally the tracks holding the selection);
  * ``"all"``     every UNLOCKED track (locked tracks are left out and listed in ``plan.notes`` - "Track X is locked and was not shifted").
  Linked and grouped clips follow regardless of scope: wherever a clip they belong to is shifted / trimmed / split, they receive the same treatment by their own position.
  A locked track or a locked clip that the ripple would REALLY have to change (explicit scope, a linked follower, a clip that has to move) rejects the whole edit, naming it.

The one mechanism behind delete / trim / insert / move ("operations on the time axis", in ORIGINAL coordinates)
  * ``cut [a, b)``  removes the time range from every in-scope track: clips after it shift left by ``b - a``; a clip crossing one edge of the range is TRIMMED there (head or
    tail removed, ``source_in`` advanced for a head); a clip crossing the whole range is SPLIT and closed up (left part stays, right part shifts); a clip inside it is removed.
  * ``gap [t, t + span)``  opens room: clips at / after ``t`` shift right; a clip crossing ``t`` is SPLIT there (right part shifts).
  Pieces shorter than the minimum clip length are dropped (noted); a clip that is only shifted is never touched by that rule. Text pieces are split between words using their
  word timing (``ops_basic`` rules); a block of text without word timing cannot be split and rejects the edit.

Time policy: shifts are computed from integer frame indices when the clip and the shift are on the frame grid (no drift over repeated edits); audio-track edits keep sample
precision; times that come from the caller (``new_time``, ``at``, ``delta``) are snapped to the frame grid (picture, captions, graphics) or the sample grid (audio tracks).
``source_out - source_in == duration * speed`` is kept for every media clip.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterable

from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import KIND_MEDIA, LOCK_KINDS, Clip
from app.timeline.keyframes import window_keyframes
from app.timeline.ops_basic import (  # the cut / lock rules live in ops_basic; reused so a ripple splits and trims exactly like a plain split / trim
    Grid, _cut_clip, _lock_problem, _make_piece, _r, _source_info, _word_index, sync_text_timing,
)
from app.timeline.ops_common import link_members, new_link_id, quantize_delta, relatives
from app.timeline.plan import EditPlan, PlanEnv, changed_aspects
from app.timeline.timeline import Timeline, new_clip_id
from app.timeline.track import Track, TrackKind

RIPPLE_SCOPES = ("track", "tracks", "all")

__all__ = [
    "RIPPLE_SCOPES", "resolve_scope", "plan_ripple_delete", "plan_ripple_trim", "plan_ripple_insert", "plan_ripple_move", "plan_roll", "plan_slip", "plan_slide",
    "roll_limits", "slip_limits", "slide_limits",
]

_EPS = TIME_EPSILON
_INF = math.inf


# ------------------------------------------------------------------------------------------------------------------------ small helpers
def _fmt(t: float) -> str:
    return f"{t:.2f} s"


def _plural(n: int, word: str = "clip") -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _plus(env: PlanEnv, t: float, d: float) -> float:
    """``t + d`` computed from integer frame indices when both are on the frame grid (repeated edits never drift), else plain addition."""
    tb = env.timebase
    if tb.is_on_frame(t) and tb.is_on_frame(d):
        return tb.to_seconds(tb.to_frame(t) + tb.to_frame(d))
    return t + d


def _diff(env: PlanEnv, a: float, b: float) -> float:
    """``b - a`` (frame exact when both ends are on the frame grid)."""
    tb = env.timebase
    if tb.is_on_frame(a) and tb.is_on_frame(b):
        return tb.to_seconds(tb.to_frame(b) - tb.to_frame(a))
    return b - a


def _num(value: float, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TimelineError(f"The {what} is not a number.") from None
    if not math.isfinite(v):
        raise TimelineError(f"The {what} is not a valid number.")
    return v


def _is_audio(tl: Timeline, track_id: str) -> bool:
    return tl.get_track(track_id).kind is TrackKind.AUDIO


def _check_scope(scope: str) -> None:
    if scope not in RIPPLE_SCOPES:
        raise TimelineError(f"Unknown ripple scope “{scope}”. Use one of: {', '.join(RIPPLE_SCOPES)}.")


def resolve_scope(tl: Timeline, scope: str, base_track_ids: Iterable[str], selected_track_ids: Iterable[str] | None = None) -> tuple[list[Track], list[Track]]:
    """``(tracks_in_scope, skipped_locked_tracks)`` as ``Track`` objects in timeline order.

    ``"track"``: the base tracks (the tracks of the edited clips). ``"tracks"``: base tracks plus ``selected_track_ids``. ``"all"``: every track, but LOCKED tracks are moved to
    the skipped list (they are never shifted; the planners mention them in ``plan.notes``). For the two explicit scopes a locked track stays in the list: asking to ripple a
    locked track is an error only if something on it would really have to change (the planner then raises, naming the track). Unknown track ids raise ``TimelineError``."""
    _check_scope(scope)
    base = list(dict.fromkeys(base_track_ids))
    extra = list(dict.fromkeys(selected_track_ids or ()))
    for tid in (*base, *extra):
        tl.get_track(tid)
    if scope == "track":
        ids = set(base)
    elif scope == "tracks":
        ids = set(base) | set(extra)
    else:
        ids = {t.id for t in tl.tracks}
    tracks = [t for t in tl.tracks if t.id in ids]
    if scope != "all":
        return tracks, []
    return [t for t in tracks if not t.locked], [t for t in tracks if t.locked]


# ------------------------------------------------------------------------------------------------------------------------ operations on the time axis
@dataclass(frozen=True)
class _Spec:
    """One operation on the ORIGINAL time axis. ``cut``: remove [a, b). ``gap``: open ``b - a`` of room at ``a``. ``audio_only``: every clip behind it lives on an audio track,
    so picture tracks round it to whole frames (otherwise it is used exactly as it is - closing a gap must be exact)."""

    kind: str
    a: float
    b: float
    audio_only: bool = False


def _merge_cuts(items: Iterable[tuple[float, float, bool]]) -> list[_Spec]:
    out: list[_Spec] = []
    for a, b, audio in sorted(items):
        if out and a <= out[-1].b + _EPS:
            last = out[-1]
            out[-1] = _Spec("cut", last.a, max(last.b, b), last.audio_only and audio)
        else:
            out.append(_Spec("cut", a, b, audio))
    return out


class _Ops:
    """The operations as sorted, disjoint intervals with prefix sums: the shift of any point and the pieces of any clip are O(log n)."""

    def __init__(self, env: PlanEnv, specs: list[_Spec], picture: bool) -> None:
        tb = env.timebase
        cuts: list[tuple[float, float]] = []
        gaps: list[tuple[float, float]] = []
        for sp in specs:
            a, b = (tb.snap(sp.a), tb.snap(sp.b)) if picture and sp.audio_only else (sp.a, sp.b)
            if b - a > _EPS:
                (cuts if sp.kind == "cut" else gaps).append((a, b))
        cuts.sort()
        gaps.sort()
        self.cuts, self.gaps = cuts, gaps
        self.cut_b = [b for _a, b in cuts]
        self.gap_a = [a for a, _b in gaps]
        self.cut_pre = [0.0]
        for a, b in cuts:
            self.cut_pre.append(self.cut_pre[-1] + (b - a))
        self.gap_pre = [0.0]
        for a, b in gaps:
            self.gap_pre.append(self.gap_pre[-1] + (b - a))
        self.first = min(cuts[0][0] if cuts else _INF, gaps[0][0] if gaps else _INF)

    def shift_at(self, x: float) -> float:
        """How far material that starts at ``x`` (original coordinates) ends up from where it was."""
        return self.gap_pre[bisect_right(self.gap_a, x + _EPS)] - self.cut_pre[bisect_right(self.cut_b, x + _EPS)]

    def _in_cut(self, m: float) -> bool:
        i = bisect_right(self.cut_b, m)
        return i < len(self.cuts) and self.cuts[i][0] <= m

    def pieces(self, s: float, e: float) -> list[tuple[float, float, float]] | None:
        """What is left of the clip [s, e): ``(piece start, piece end, shift)`` in original coordinates; ``None`` when nothing happens to the clip; ``[]`` when it is cut away."""
        if e <= self.first + _EPS:
            return None
        pts = [s, e]
        i = bisect_right(self.cut_b, s + _EPS)
        while i < len(self.cuts) and self.cuts[i][0] < e - _EPS:
            a, b = self.cuts[i]
            if a > s + _EPS:
                pts.append(a)
            if b < e - _EPS:
                pts.append(b)
            i += 1
        j = bisect_right(self.gap_a, s + _EPS)
        while j < len(self.gaps) and self.gaps[j][0] < e - _EPS:
            pts.append(self.gaps[j][0])
            j += 1
        pts.sort()
        out = [(x, y, self.shift_at(x)) for x, y in zip(pts, pts[1:]) if y - x > _EPS and not self._in_cut((x + y) / 2)]
        if len(out) == 1 and out[0][0] == s and out[0][1] == e and abs(out[0][2]) <= _EPS:
            return None
        return out


class _Run:
    """Collects one plan: applies the operations to the clips of the scope tracks, lets linked / grouped clips follow, and tracks everything the notes need."""

    def __init__(self, env: PlanEnv, *, linked: bool, grouped: bool) -> None:
        self.env, self.tl = env, env.timeline
        self.linked, self.grouped = linked, grouped
        self.updates: dict[str, Clip] = {}
        self.adds: list[Clip] = []
        self.removes: list[str] = []
        self.notes: list[str] = []
        self.done: set[str] = set()  # ids that are handled (or deliberately left to the caller); never processed twice
        self.translated: set[str] = set()
        self.translated_tracks: set[str] = set()
        self.slivers = self.cut_away = self.splits = self.text_dropped = 0
        self._ops_cache: dict[tuple[int, bool], _Ops] = {}
        self._queue: list[tuple[Clip, list[_Spec]]] = []
        self._rel: tuple[dict[str, list[Clip]], dict[str, list[Clip]]] | None = None
        self._new: list[tuple[Clip, str, int]] = []  # (new right-hand piece, link id of its source clip, piece number)
        self._audio = {t.id: t.kind is TrackKind.AUDIO for t in self.tl.tracks}
        self._grids = {False: Grid(env, False), True: Grid(env, True)}

    # ---------------------------------------------------------------- locks
    def guard(self, old: Clip, new: Clip | None, action: str, extra: Iterable[str] = ()) -> None:
        aspects = set(LOCK_KINDS) if new is None else set(changed_aspects(old, new)) | set(extra)
        msg = _lock_problem(self.env, old, tuple(sorted(aspects)), action)
        if msg:
            raise TimelineError(f"{msg} The ripple would have to change it; a narrower ripple scope leaves it out.")

    # ---------------------------------------------------------------- relations
    def _relations(self) -> tuple[dict[str, list[Clip]], dict[str, list[Clip]]]:
        if self._rel is None:
            by_link: dict[str, list[Clip]] = {}
            by_group: dict[str, list[Clip]] = {}
            for c in self.tl.all_clips():
                if c.link_id:
                    by_link.setdefault(c.link_id, []).append(c)
                if c.group_id:
                    by_group.setdefault(c.group_id, []).append(c)
            self._rel = (by_link, by_group)
        return self._rel

    def enqueue(self, clip: Clip, specs: list[_Spec], translated: bool) -> None:
        """Linked relatives of a changed clip follow it; grouped ones only when the clip itself was shifted."""
        want_link = self.linked and clip.link_id
        want_group = self.grouped and translated and clip.group_id
        if not (want_link or want_group):
            return
        by_link, by_group = self._relations()
        partners = [*(by_link.get(clip.link_id, ()) if want_link else ()), *(by_group.get(clip.group_id, ()) if want_group else ())]
        for p in partners:
            if p.id not in self.done:
                self._queue.append((p, specs))

    def follow(self) -> None:
        while self._queue:
            clip, specs = self._queue.pop()
            self.process(clip, specs)

    # ---------------------------------------------------------------- the engine
    def _ops(self, specs: list[_Spec], picture: bool) -> _Ops:
        key = (id(specs), picture)
        ops = self._ops_cache.get(key)
        if ops is None:
            ops = self._ops_cache[key] = _Ops(self.env, specs, picture)
        return ops

    def run_scope(self, tracks: Iterable[Track], specs_for: Callable[[str], list[_Spec] | None]) -> None:
        for t in tracks:
            specs = specs_for(t.id)
            if specs:
                for c in list(t.clips):
                    self.process(c, specs)

    def process(self, clip: Clip, specs: list[_Spec]) -> None:
        if clip.id in self.done:
            return
        self.done.add(clip.id)
        env = self.env
        audio = self._audio[clip.track_id]
        res = self._ops(specs, not audio).pieces(clip.timeline_start, clip.timeline_end)
        if res is None:
            return
        s, e = clip.timeline_start, clip.timeline_end
        md = max(self._grids[audio].min_duration, MIN_CLIP_DURATION)
        kept = [p for p in res if (p[1] - p[0] >= md - _EPS) or (p[0] == s and p[1] == e)]
        self.slivers += len(res) - len(kept)
        words: list[tuple[int, int] | None] = [None] * len(kept)
        if clip.kind != KIND_MEDIA and len(kept) > 1:
            raw = clip.text.get("words") if isinstance(clip.text, dict) else None
            if not (isinstance(raw, list) and raw):
                raise TimelineError(f"“{env.label(clip)}” would have to be split by this ripple, but it is a single block of text without word timing. "
                                    "Use a ripple scope that leaves it out, or shorten it first.")
            words = []
            for ks, ke, _sh in kept:
                lo = 0 if abs(ks - s) < 1e-9 else _word_index(raw, ks)[0]
                hi = len(raw) if abs(ke - e) < 1e-9 else _word_index(raw, ke)[0]
                words.append((lo, hi) if hi > lo else None)
            before = len(kept)
            kept = [p for p, w in zip(kept, words) if w is not None]
            words = [w for w in words if w is not None]
            self.text_dropped += before - len(kept)
        if not kept:
            self.guard(clip, None, "removed")
            self.removes.append(clip.id)
            self.cut_away += 1
            return
        built = [self._piece(clip, ks, ke, sh, k, words[k]) for k, (ks, ke, sh) in enumerate(kept)]
        first = built[0]
        if len(kept) > 1:
            action = "split"
        else:
            action = "moved" if abs(first.duration - clip.duration) <= _EPS else "trimmed"
        self.guard(clip, first, action, extra=("timing",) if len(kept) > 1 else ())
        self.updates[clip.id] = first
        translated = abs(first.timeline_start - s) > _EPS
        if translated:
            self.translated.add(clip.id)
            self.translated_tracks.add(clip.track_id)
        for k, c in enumerate(built[1:], 1):
            self.adds.append(c)
            self._new.append((c, clip.link_id, k))
            self.splits += 1
        self.enqueue(clip, specs, translated)

    def _piece(self, clip: Clip, ks: float, ke: float, shift: float, k: int, wr: tuple[int, int] | None) -> Clip:
        env = self.env
        new_start = _plus(env, ks, shift)
        if -_EPS < new_start < 0:
            new_start = 0.0
        if ks == clip.timeline_start and ke == clip.timeline_end:
            c = clip.snapshot()
        else:
            c = _cut_clip(env, clip, ks, ke, words=wr)
            c.duration = _diff(env, ks, ke)
        c.timeline_start = new_start
        sync_text_timing(c, new_start - ks)
        if k:
            _make_piece(c, clip, new_clip_id())
        return c

    # ---------------------------------------------------------------- result
    def finish(self, description: str, *, skipped: Iterable[Track] = (), first_pos: float = 0.0, notes: Iterable[str] = ()) -> EditPlan:
        groups: dict[tuple[str, int], list[Clip]] = {}
        for c, old_link, k in self._new:
            if old_link:
                groups.setdefault((old_link, k), []).append(c)
        for members in groups.values():  # the new right-hand pieces of linked clips are linked to each other, not to the left pieces
            fresh = new_link_id() if len(members) >= 2 else ""
            for c in members:
                c.link_id = fresh
        out = list(notes)
        if self.translated:
            out.append(f"Shifted {_plural(len(self.translated))} on {_plural(len(self.translated_tracks), 'track')}.")
        if self.splits:
            out.append(f"Split {_plural(self.splits, 'clip')} where the edit crossed them.")
        if self.cut_away:
            out.append(f"{_plural(self.cut_away)} on other tracks had nothing left after the edit and {'was' if self.cut_away == 1 else 'were'} removed with it.")
        if self.slivers:
            out.append(f"{_plural(self.slivers, 'fragment')} shorter than the minimum clip length {'was' if self.slivers == 1 else 'were'} removed.")
        if self.text_dropped:
            out.append(f"{_plural(self.text_dropped, 'text piece')} without words {'was' if self.text_dropped == 1 else 'were'} dropped.")
        for t in skipped:
            if any(c.timeline_end > first_pos + _EPS for c in t.clips):
                out.append(f"Track “{t.name}” is locked and was not shifted.")
        return EditPlan(description, updates=self.updates, adds=self.adds, removes=self.removes, notes=out, shifted=len(self.translated))


# ------------------------------------------------------------------------------------------------------------------------ ripple delete
def plan_ripple_delete(env: PlanEnv, ids: list[str], *, scope: str = "track", selected_track_ids: Iterable[str] | None = None, linked: bool = True, grouped: bool = True,
                       close_gap: bool = True) -> EditPlan:
    """Delete clips (plus linked / grouped relatives) and close the time they occupied.

    The removed clips' time ranges (merged per track) are cut out of every in-scope track: later clips shift left by what was freed (spec: A 0-10, B 10-20, C 20-30; delete B -> C
    10-20); a clip crossing the cut is trimmed at the cut or split-and-closed; clips on OTHER in-scope tracks lying entirely inside the cut are removed with it. With scope
    ``"track"`` each edited track closes by what ITS removed clips freed; with ``"tracks"`` / ``"all"`` the cut is the union of the edited clips' ranges, applied to every track in scope
    (linked / grouped relatives that live on other tracks are deleted too, and move with their partners but do not widen the cut). ``close_gap=False`` is a plain delete."""
    _check_scope(scope)
    tl = env.timeline
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise TimelineError("Select at least one clip to delete.")
    prim = {i: tl.find_clip(i) for i in ids}
    all_ids = relatives(tl, ids, linked=linked, grouped=grouped)
    removed = {i: tl.find_clip(i)[1] for i in all_ids}
    run = _Run(env, linked=linked, grouped=grouped)
    for c in removed.values():
        run.guard(c, None, "removed")
    run.done.update(all_ids)
    run.removes.extend(all_ids)
    name = env.label(removed[ids[0]]) if len(all_ids) == 1 else f"{len(all_ids)} clips"
    desc = f"Ripple delete {name}" if close_gap else f"Delete {name}"
    if not close_gap:
        return run.finish(desc)
    base = list(dict.fromkeys(t.id for t, _c in prim.values()))
    sources = [c for c in removed.values() if c.track_id in base]  # the edited clips and relatives sharing their track define what closes
    audio_of = {tid: _is_audio(tl, tid) for tid in base}
    if scope == "track":
        per = {tid: _merge_cuts((c.timeline_start, c.timeline_end, audio_of[tid]) for c in sources if c.track_id == tid) for tid in base}
        lists, specs_for = list(per.values()), per.get
    else:
        allspecs = _merge_cuts((c.timeline_start, c.timeline_end, audio_of[c.track_id]) for c in sources)
        lists, specs_for = [allspecs], (lambda _tid: allspecs)
    tracks, skipped = resolve_scope(tl, scope, base, selected_track_ids)
    run.run_scope(tracks, specs_for)
    run.follow()
    return run.finish(desc, skipped=skipped, first_pos=min((sp.a for lst in lists for sp in lst), default=0.0))


# ------------------------------------------------------------------------------------------------------------------------ edge limits (own source / minimum length only)
def _edge_range(env: PlanEnv, clip: Clip, edge: str) -> tuple[float, str, float, str]:
    """(lowest, why, highest, why): where this edge may sit judged by the clip ITSELF (its minimum length and its source media), rounded inward onto the grid, never stricter than
    where the edge is now. Neighbouring clips are not considered (the callers move them out of the way)."""
    grid = Grid(env, _is_audio(env.timeline, clip.track_id))
    s, e = clip.timeline_start, clip.timeline_end
    bound, total = _source_info(env, clip)
    if edge == "start":
        lo, lo_why = -_INF, ""
        if bound:
            lo, lo_why = s - clip.source_in / clip.speed, "the start of the source media"
        hi, hi_why = e - grid.min_duration, "the minimum clip length"
        cur = s
    else:
        lo, lo_why = s + grid.min_duration, "the minimum clip length"
        hi, hi_why = _INF, ""
        if bound and total is not None:
            hi, hi_why = s + (total - clip.source_in) / clip.speed, "the end of the source media"
        cur = e
    lo_q, hi_q = grid.ceil(lo), grid.floor(hi)
    if cur >= lo - _EPS:
        lo_q = min(lo_q, cur)
    if cur <= hi + _EPS:
        hi_q = max(hi_q, cur)
    return lo_q, lo_why, hi_q, hi_why


def _trim_edge(env: PlanEnv, clip: Clip, edge: str, t: float) -> Clip:
    """``clip`` with one edge moved to timeline time ``t`` - the ``plan_trim`` rules: a start trim advances ``source_in`` by ``delta * speed`` (``source_out`` stays), an end trim
    moves ``source_out`` with the duration, keyframes are windowed, text timing follows; a picture has no source timeline to map."""
    c = clip.snapshot()
    s, e = clip.timeline_start, clip.timeline_end
    bound = _source_info(env, clip)[0]
    still = clip.kind == KIND_MEDIA and not bound and env.asset_info(clip.asset_id) is not None
    if edge == "start":
        c.timeline_start, c.duration = t, _diff(env, t, e)
        if still:
            c.source_out = _r(clip.source_in + c.duration * clip.speed)
        elif clip.kind == KIND_MEDIA:
            c.source_in = max(0.0, _r(clip.source_in + (t - s) * clip.speed))
        if clip.keyframes and abs(t - s) > 1e-12:
            c.keyframes = window_keyframes(clip.keyframes, t - s, clip.duration)
    else:
        c.duration = _diff(env, s, t)
        if clip.kind == KIND_MEDIA:
            c.source_out = _r(clip.source_in + c.duration * clip.speed)
        if clip.keyframes and c.duration < clip.duration - 1e-12:
            c.keyframes = window_keyframes(clip.keyframes, 0.0, c.duration)
    sync_text_timing(c)
    return c


def _apply_edge(env: PlanEnv, clip: Clip, edge: str, t: float, *, who: str = "") -> Clip:
    """``clip`` with one edge moved to ``t`` (the trim rules of ``plan_trim``); raises with the reason if the clip itself does not allow it."""
    lo, lo_why, hi, hi_why = _edge_range(env, clip, edge)
    name = f"{who}“{env.label(clip)}”"
    if t < lo - _EPS:
        if lo_why == "the minimum clip length":
            raise TimelineError(f"{name} would be shorter than the minimum clip length ({MIN_CLIP_DURATION:g} s).")
        room = clip.timeline_start - lo
        raise TimelineError(f"{name} has only {_fmt(max(room, 0.0))} of source media before its current start.")
    if t > hi + _EPS:
        if hi_why == "the minimum clip length":
            raise TimelineError(f"{name} would be shorter than the minimum clip length ({MIN_CLIP_DURATION:g} s).")
        room = hi - clip.timeline_end
        raise TimelineError(f"{name} has only {_fmt(max(room, 0.0))} of source media left after its current end.")
    return _trim_edge(env, clip, edge, t)


# ------------------------------------------------------------------------------------------------------------------------ ripple trim
def plan_ripple_trim(env: PlanEnv, clip_id: str, edge: str, new_time: float, *, scope: str = "track", selected_track_ids: Iterable[str] | None = None,
                     linked: bool = True, grouped: bool = True) -> EditPlan:
    """Trim one edge of a clip to a timeline time and move everything after the edit by the duration change.

    ``edge="end"``: the tail is cut / extended; later clips shift by the change. ``edge="start"`` (classic ripple trim-in): the clip's START stays where it is, its head is removed
    (``source_in`` advances) or extended (``source_in`` goes back) and later clips shift; ``new_time`` is where the new in-point would sit in the un-rippled timeline. Same trim
    rules as ``plan_trim``: limited by the minimum length and the source media (clamped, explained in the notes); linked clips whose edge coincides with this one trim by the same
    amount. Other in-scope tracks are cut (shorter) or opened (longer) at the edit, see the module notes."""
    _check_scope(scope)
    if edge not in ("start", "end"):
        raise TimelineError("Choose the start or the end edge of the clip.")
    tl = env.timeline
    track, clip = tl.find_clip(clip_id)
    desc = f"Ripple trim clip {edge}"
    cur = clip.timeline_start if edge == "start" else clip.timeline_end
    grid = Grid(env, track.kind is TrackKind.AUDIO)
    want = grid.snap(_num(new_time, "trim time")) - cur
    tol = env.timebase.frame_s + _EPS
    members = [clip]
    if linked and clip.link_id:
        for r in link_members(tl, clip.link_id):
            if r.id != clip.id and abs((r.timeline_start if edge == "start" else r.timeline_end) - cur) <= tol:
                members.append(r)
    if abs(want) <= _EPS:
        return EditPlan(desc, notes=["The edge is already there."])
    info = []
    for m in members:
        lo, lo_w, hi, hi_w = _edge_range(env, m, edge)
        pos = m.timeline_start if edge == "start" else m.timeline_end
        info.append((m, lo - pos, lo_w, hi - pos, hi_w, pos))
    lo_b, hi_b = max(info, key=lambda x: x[1]), min(info, key=lambda x: x[3])
    dmin, dmax = lo_b[1], hi_b[3]
    if dmin > dmax + _EPS:
        raise TimelineError(f"“{env.label(lo_b[0])}” and “{env.label(hi_b[0])}” are linked but cannot be trimmed together: one is limited by {lo_b[2]}, the other by {hi_b[4]}.")
    d = min(max(want, dmin), dmax)
    notes: list[str] = []
    if abs(d) <= _EPS:
        who = lo_b if want < 0 else hi_b
        raise TimelineError(f"“{env.label(who[0])}” cannot be trimmed any further: it is limited by {who[2] if want < 0 else who[4]}.")
    if abs(d - want) > _EPS:
        binding, why = (lo_b, lo_b[2]) if d > want else (hi_b, hi_b[4])
        notes.append(f"The {edge} edge stopped at {_fmt(cur + d)} instead of {_fmt(cur + want)} because of {why}"
                     + (f" (“{env.label(binding[0])}”)." if len(members) > 1 or binding[0] is not clip else "."))
    run = _Run(env, linked=linked, grouped=grouped)
    audio = _is_audio(tl, clip.track_id)
    shorter = (d < 0) == (edge == "end")  # the end edge moving back, or the start edge moving forward, removes time
    if not shorter:
        spec = _Spec("gap", cur, _plus(env, cur, abs(d)), audio)
    elif edge == "end":
        spec = _Spec("cut", _plus(env, cur, d), cur, audio)
    else:
        spec = _Spec("cut", cur, _plus(env, cur, d), audio)
    for m, _lo, _lw, _hi, _hw, pos in info:  # the edited clips themselves: plain trim rules, start (or end) stays, the rest of the ripple follows from the cut / gap
        t = _plus(env, pos, d)
        n = _trim_edge(env, m, edge, t)
        if edge == "start":
            n.timeline_start = m.timeline_start
            sync_text_timing(n, m.timeline_start - t)
        run.guard(m, n, "trimmed")
        run.updates[m.id] = n
        run.done.add(m.id)
    specs = [spec]
    tracks, skipped = resolve_scope(tl, scope, [track.id], selected_track_ids)
    run.run_scope(tracks, lambda _tid: specs)
    for m in members:
        run.enqueue(m, specs, False)
    run.follow()
    return run.finish(desc, skipped=skipped, first_pos=spec.a, notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ ripple insert
def plan_ripple_insert(env: PlanEnv, clips: list[Clip], at: float, track_id: str, *, scope: str = "track", selected_track_ids: Iterable[str] | None = None,
                       linked: bool = True, grouped: bool = True) -> EditPlan:
    """Insert ``clips`` at ``at`` and shift everything at / after ``at`` in scope right by the inserted span.

    The clips keep their layout relative to each other: the earliest start lands on ``at`` (snapped to the frame / sample grid) and the span is the whole group's length. A clip
    keeps its own ``track_id`` when that names an existing track, otherwise it goes to ``track_id``; the tracks of all inserted clips are the base of the ripple scope. A clip crossing
    ``at`` on an in-scope track is SPLIT there (right half shifts, text only between words). Ids that are empty or already used get fresh ones."""
    _check_scope(scope)
    tl = env.timeline
    if not clips:
        raise TimelineError("There is nothing to insert.")
    target = tl.get_track(track_id)
    new = [c.snapshot() for c in clips]
    existing = {c.id for c in tl.all_clips()}
    seen: set[str] = set()
    known = {t.id for t in tl.tracks}
    for c in new:
        if c.track_id not in known:
            c.track_id = target.id
        if not c.id or c.id in existing or c.id in seen:
            c.id = new_clip_id()
        seen.add(c.id)
        tr = tl.get_track(c.track_id)
        if tr.locked:
            raise TimelineError(f"Track “{tr.name}” is locked. Unlock it to insert clips there.")
    audio_only = all(_is_audio(tl, c.track_id) for c in new)
    at_q = Grid(env, audio_only).snap(_num(at, "insert time"))
    if at_q < -_EPS:
        raise TimelineError("Clips cannot be inserted before 0:00.")
    lo = min(c.timeline_start for c in new)
    hi = max(c.timeline_end for c in new)
    off = _diff(env, lo, at_q)
    for c in new:
        old_start = c.timeline_start
        c.timeline_start = _plus(env, old_start, off)
        sync_text_timing(c, c.timeline_start - old_start)
    span = _diff(env, lo, hi)
    spec = _Spec("gap", at_q, at_q + span, audio_only)
    specs = [spec]
    run = _Run(env, linked=linked, grouped=grouped)
    run.adds.extend(new)
    tracks, skipped = resolve_scope(tl, scope, [target.id, *(c.track_id for c in new)], selected_track_ids)
    run.run_scope(tracks, lambda _tid: specs)
    run.follow()
    desc = "Ripple insert clip" if len(new) == 1 else f"Ripple insert {len(new)} clips"
    return run.finish(desc, skipped=skipped, first_pos=at_q)


# ------------------------------------------------------------------------------------------------------------------------ ripple move
def plan_ripple_move(env: PlanEnv, ids: list[str], delta: float, *, scope: str = "track", selected_track_ids: Iterable[str] | None = None, linked: bool = True,
                     grouped: bool = True) -> EditPlan:
    """Move clips along their own track with ripple: the gap they leave closes and room opens where they land.

    All ``ids`` must sit on one track (a move onto another track is rejected - that is not a ripple move); linked / grouped relatives move by the same amount. The edited clips form
    one block (no other clip may lie between them). ``delta`` (whole frames, or samples on an audio track) says where the block's start would land in the CURRENT layout; the block
    is lifted out, the gap closes, and the block is inserted at that point of the closed layout (a clip crossing the insertion point is split). Landing inside the block's own
    span changes nothing (empty plan). Only the clip's own track ripples (linked relatives follow); other scopes are accepted but treated as ``"track"``, noted in the plan."""
    _check_scope(scope)
    tl = env.timeline
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise TimelineError("Select at least one clip to move.")
    found = {i: tl.find_clip(i) for i in ids}
    bases = {t.id for t, _c in found.values()}
    if len(bases) != 1:
        raise TimelineError("A ripple move works on clips of one track. Select clips on a single track, or move them without ripple.")
    base = next(iter(found.values()))[0]
    all_ids = relatives(tl, ids, linked=linked, grouped=grouped)
    moved = {i: tl.find_clip(i)[1] for i in all_ids}
    desc = "Ripple move clip" if len(all_ids) == 1 else f"Ripple move {len(all_ids)} clips"
    on_base = [c for c in moved.values() if c.track_id == base.id]
    a = min(c.timeline_start for c in on_base)
    b = max(c.timeline_end for c in on_base)
    for c in base.clips:
        if c.id not in moved and c.timeline_end > a + _EPS and c.timeline_start < b - _EPS:
            raise TimelineError(f"“{env.label(c)}” lies between the clips being moved. Select clips that are next to each other for a ripple move.")
    audio = base.kind is TrackKind.AUDIO
    d = quantize_delta(env, _num(delta, "move distance"), audio=audio)
    if abs(d) <= _EPS:
        return EditPlan(desc, notes=["Nothing moved."])
    notes: list[str] = []
    if scope != "track":
        notes.append("A ripple move only shifts the clip's own track (linked clips follow); the other tracks were not shifted.")
    p = _plus(env, a, d)
    if p < -_EPS:
        notes.append(f"The move was stopped at the start of the timeline: it moved {_fmt(-a)} instead of {_fmt(d)}.")
        p = 0.0
    span = _diff(env, a, b)
    if p > a + _EPS and p < b - _EPS:
        return EditPlan(desc, notes=[*notes, "The clips would land inside their own span, so their order does not change."])
    landing = _plus(env, p, -span) if p >= b - _EPS else p
    net = _diff(env, a, landing)
    if abs(net) <= _EPS:
        return EditPlan(desc, notes=[*notes, "The clips would land between the same neighbours, so nothing changed."])
    run = _Run(env, linked=linked, grouped=grouped)
    for c in moved.values():
        n = c.snapshot()
        n.timeline_start = max(0.0, _plus(env, c.timeline_start, net))
        sync_text_timing(n, n.timeline_start - c.timeline_start)
        run.guard(c, n, "moved")
        run.updates[c.id] = n
        run.done.add(c.id)
    audio_only = all(_is_audio(tl, c.track_id) for c in moved.values())
    specs = [_Spec("cut", a, b, audio_only), _Spec("gap", p, p + span, audio_only)]
    run.run_scope([base], lambda _tid: specs)
    run.follow()
    return run.finish(desc, first_pos=min(a, p), notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ roll
def _adjacent_tolerance(env: PlanEnv) -> float:
    return env.timebase.frame_s + _EPS


def roll_limits(env: PlanEnv, left_id: str, right_id: str) -> tuple[float, float]:
    """(lowest, highest) timeline time the shared cut of two adjacent clips may take, judged by the two clips' own limits (minimum length, source media). For live clamping in the UI."""
    tl = env.timeline
    left, right = tl.find_clip(left_id)[1], tl.find_clip(right_id)[1]
    llo, _a, lhi, _b = _edge_range(env, left, "end")
    rlo, _c, rhi, _d = _edge_range(env, right, "start")
    return max(llo, rlo), min(lhi, rhi)


def plan_roll(env: PlanEnv, left_id: str, right_id: str, new_boundary_time: float, *, linked: bool = True) -> EditPlan:
    """Move the cut between two ADJACENT clips of one track: the left clip's end / ``source_out`` and the right clip's start / ``source_in`` / duration change together, so the
    combined duration is unchanged (spec: a 10 s cut rolled to 9.5 s keeps the total length). Rejected, with the reason, when the clips are not adjacent (within a frame), when the
    left clip would run out of source media or the right clip would start before its media begins, or when a clip would drop below the minimum length. Linked clips follow: the
    link partners of the left clip move their end, those of the right clip move their start, by the same amount."""
    tl = env.timeline
    if left_id == right_id:
        raise TimelineError("Choose two different clips to roll between.")
    ltrack, left = tl.find_clip(left_id)
    rtrack, right = tl.find_clip(right_id)
    ln, rn = env.label(left), env.label(right)
    if ltrack.id != rtrack.id:
        raise TimelineError(f"“{ln}” and “{rn}” are on different tracks, so there is no shared cut to roll.")
    gap = right.timeline_start - left.timeline_end
    if abs(gap) > _adjacent_tolerance(env):
        raise TimelineError(f"“{ln}” and “{rn}” are not next to each other (they are {_fmt(abs(gap))} apart, in that order), so there is no shared cut to roll.")
    grid = Grid(env, ltrack.kind is TrackKind.AUDIO)
    t = grid.snap(_num(new_boundary_time, "cut time"))
    dl, dr = t - left.timeline_end, t - right.timeline_start
    desc = "Roll edit"
    if abs(dl) <= _EPS and abs(dr) <= _EPS:
        return EditPlan(desc, notes=["The cut is already there."])
    lo, hi = roll_limits(env, left_id, right_id)
    try:
        updates = {left.id: _apply_edge(env, left, "end", t), right.id: _apply_edge(env, right, "start", t)}
    except TimelineError as ex:
        raise TimelineError(f"{ex.user_message} The cut can only move between {_fmt(lo)} and {_fmt(hi)}.") from None
    if linked and (left.link_id or right.link_id):
        pairs: dict[str, tuple[Clip, str]] = {}
        for c in [*link_members(tl, left.link_id), *link_members(tl, right.link_id)]:
            if c.id in (left.id, right.id) or c.id in pairs:
                continue
            if c.link_id == left.link_id and c.link_id != right.link_id:
                edge = "end"
            elif c.link_id == right.link_id and c.link_id != left.link_id:
                edge = "start"
            else:
                edge = "end" if abs(c.timeline_end - left.timeline_end) <= abs(c.timeline_start - right.timeline_start) else "start"
            pairs[c.id] = (c, edge)
        for c, edge in pairs.values():
            pos = c.timeline_end if edge == "end" else c.timeline_start
            updates[c.id] = _apply_edge(env, c, edge, _plus(env, pos, dl if edge == "end" else dr), who="Linked clip ")
    return EditPlan(desc, updates=updates)


# ------------------------------------------------------------------------------------------------------------------------ slip
def slip_limits(env: PlanEnv, clip_id: str) -> tuple[float, float]:
    """(lowest, highest) source-seconds delta a slip may use: ``-source_in`` .. ``media length - source_out`` (``+inf`` when the media length is unknown). Raises for clips with
    nothing to slip (stills, text, graphics, captions)."""
    clip = env.timeline.find_clip(clip_id)[1]
    bound, total = _source_info(env, clip)
    if not bound:
        raise TimelineError(f"Nothing to slip: “{env.label(clip)}” has no source media (still images, text, graphics and captions have no content to slide under their cut).")
    return 0.0 - clip.source_in, (_INF if total is None else max(total - clip.source_out, 0.0))


def plan_slip(env: PlanEnv, clip_id: str, delta_source: float, *, linked: bool = True) -> EditPlan:
    """Slip the content: ``source_in`` and ``source_out`` both move by ``delta_source`` (source seconds; whole frames on picture clips, samples on audio) while the clip's timeline
    position, duration and speed stay. Bounded by 0 and the media length: a larger slip is rejected with the allowed range. Keyframes are clip-local and untouched. Linked media
    clips slip by the same amount (a relative with nothing to slip, e.g. a caption, is left alone)."""
    tl = env.timeline
    track, clip = tl.find_clip(clip_id)
    lo, hi = slip_limits(env, clip_id)
    d = quantize_delta(env, _num(delta_source, "slip distance"), audio=track.kind is TrackKind.AUDIO)
    half, notes = env.timebase.frame_s / 2 + _EPS, []
    if hi < d <= hi + half or lo - half <= d < lo:  # a request a rounding step beyond the media: stop at the media (a whole frame beyond is rejected below)
        d = hi if d > hi else lo
        notes.append("The slip was stopped at the end of the source media." if d == hi else "The slip was stopped at the start of the source media.")
    if abs(d) <= _EPS:
        return EditPlan("Slip clip", notes=["The content is already there."])
    if d < lo - _EPS or d > hi + _EPS:
        raise TimelineError(f"“{env.label(clip)}” can only slip between {lo:+.2f} s and {hi:+.2f} s (the limits of its source media); {d:+.2f} s is out of range.")
    updates: dict[str, Clip] = {}
    members = [clip]
    if linked and clip.link_id:
        members += [r for r in link_members(tl, clip.link_id) if r.id != clip.id]
    for m in members:
        try:
            mlo, mhi = slip_limits(env, m.id)
        except TimelineError:
            if m is clip:
                raise
            continue
        if d < mlo - _EPS or d > mhi + _EPS:
            raise TimelineError(f"Linked clip “{env.label(m)}” can only slip between {mlo:+.2f} s and {mhi:+.2f} s, so the slip of {d:+.2f} s is out of range.")
        n = m.snapshot()
        n.source_in, n.source_out = _r(m.source_in + d), _r(m.source_out + d)
        updates[m.id] = n
    return EditPlan("Slip clip", updates=updates, notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ slide
def _slide_neighbours(env: PlanEnv, track: Track, clip: Clip) -> tuple[Clip | None, Clip | None]:
    """The clips directly before and after on the track, each required to touch ``clip`` (within a frame); a missing neighbour is fine, a gap is not."""
    order = sorted(track.clips, key=lambda c: (c.timeline_start, c.timeline_end))
    i = next(k for k, c in enumerate(order) if c.id == clip.id)
    prev = order[i - 1] if i > 0 else None
    nxt = order[i + 1] if i + 1 < len(order) else None
    tol = _adjacent_tolerance(env)
    name = env.label(clip)
    if prev is not None and abs(clip.timeline_start - prev.timeline_end) > tol:
        raise TimelineError(f"“{env.label(prev)}” is not directly before “{name}” ({_fmt(abs(clip.timeline_start - prev.timeline_end))} apart): a slide needs the clips to touch. "
                            "Move the clip instead, or close the gap first.")
    if nxt is not None and abs(nxt.timeline_start - clip.timeline_end) > tol:
        raise TimelineError(f"“{env.label(nxt)}” is not directly after “{name}” ({_fmt(abs(nxt.timeline_start - clip.timeline_end))} apart): a slide needs the clips to touch. "
                            "Move the clip instead, or close the gap first.")
    return prev, nxt


def slide_limits(env: PlanEnv, clip_id: str) -> tuple[float, float]:
    """(lowest, highest) timeline delta a slide of this clip may use, judged by the clip's own neighbours (their minimum lengths and source media) and 0:00 when nothing precedes it.
    Linked clips are not considered. Raises when the clip is not adjacent to its neighbours."""
    tl = env.timeline
    track, clip = tl.find_clip(clip_id)
    prev, nxt = _slide_neighbours(env, track, clip)
    lo, hi = (0.0 - clip.timeline_start if prev is None else -_INF), _INF
    if prev is not None:
        plo, _a, phi, _b = _edge_range(env, prev, "end")
        lo, hi = max(lo, plo - prev.timeline_end), min(hi, phi - prev.timeline_end)
    if nxt is not None:
        nlo, _c, nhi, _d = _edge_range(env, nxt, "start")
        lo, hi = max(lo, nlo - nxt.timeline_start), min(hi, nhi - nxt.timeline_start)
    return lo, hi


def plan_slide(env: PlanEnv, clip_id: str, delta: float, *, linked: bool = True) -> EditPlan:
    """Slide a clip along its track by ``delta`` (whole frames, or samples on audio) while its neighbours absorb the move: the previous clip's end and the next clip's start move
    with it (end trim / head trim with ``source_in`` advance), the slid clip's own source range and duration are unchanged. Both neighbours must touch the clip (within a frame) or be
    absent; a gap is rejected. Rejected too when a neighbour would drop below the minimum length or run out of source media, or when the clip would start before 0:00. Linked clips
    slide by the same amount, and the link partners of the two neighbours move their matching edge too (so linked audio stays in sync)."""
    tl = env.timeline
    track, clip = tl.find_clip(clip_id)
    prev, nxt = _slide_neighbours(env, track, clip)
    d = quantize_delta(env, _num(delta, "slide distance"), audio=track.kind is TrackKind.AUDIO)
    if abs(d) <= _EPS:
        return EditPlan("Slide clip", notes=["Nothing moved."])
    lo, hi = slide_limits(env, clip_id)
    new_start, new_end = _plus(env, clip.timeline_start, d), _plus(env, clip.timeline_end, d)
    updates: dict[str, Clip] = {}

    def put(c: Clip, n: Clip) -> None:
        if c.id in updates:
            raise TimelineError(f"“{env.label(c)}” would be changed twice by this slide (it is linked to more than one of the clips involved).")
        updates[c.id] = n

    def translate(c: Clip) -> Clip:
        n = c.snapshot()
        n.timeline_start = _plus(env, c.timeline_start, d)
        if n.timeline_start < -_EPS:
            raise TimelineError(f"“{env.label(c)}” cannot start before 0:00.")
        sync_text_timing(n, n.timeline_start - c.timeline_start)
        return n

    try:
        slid = [clip, *(r for r in (link_members(tl, clip.link_id) if linked and clip.link_id else ()) if r.id != clip.id)]
        for c in slid:
            put(c, translate(c))
        if prev is not None:
            put(prev, _apply_edge(env, prev, "end", new_start))
        if nxt is not None:
            put(nxt, _apply_edge(env, nxt, "start", new_end))
        if linked:
            for nb, edge, shift in ((prev, "end", d), (nxt, "start", d)):
                if nb is None or not nb.link_id:
                    continue
                for r in link_members(tl, nb.link_id):
                    if r.id != nb.id:
                        pos = r.timeline_end if edge == "end" else r.timeline_start
                        put(r, _apply_edge(env, r, edge, _plus(env, pos, shift), who="Linked clip "))
    except TimelineError as ex:
        if lo - _EPS <= d <= hi + _EPS:
            raise
        raise TimelineError(f"{ex.user_message} The clip can slide between {lo:+.2f} s and {hi:+.2f} s.") from None
    return EditPlan("Slide clip", updates=updates)
