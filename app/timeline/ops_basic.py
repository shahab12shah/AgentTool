"""Basic edit planners: move, numeric edits, trim, split, delete, duplicate, overwrite, replace, free insert.

Every function here is a PURE planner: ``plan_xxx(env, ...) -> EditPlan``. It reads ``env.timeline`` and never mutates it; ``PlanCommand`` validates the complete
result (locks, track compatibility, overlaps, source bounds, keyframes) and applies it atomically. Planners raise ``TimelineError`` with a plain-language reason for
impossible requests and explain every adjustment they made in ``plan.notes`` (a trim that hits a neighbour, a duration limited by the source, a text split between words...).

Vocabulary used throughout (see ``ops_common``): LINKED clips (``link_id``) follow move / trim / split / delete; GROUPED clips (``group_id``) follow move / delete / copy
but trim and split stay independent.

Time policy
  * Picture clips, captions and graphics: every edited time sits on the project's frame grid, computed from integer frame indices (``Timebase``) so repeated edits never drift.
    The *delta* of a multi-clip move is a whole number of frames, so the relative spacing of the moved clips is exactly preserved.
  * Clips on an audio track keep sample precision (48 kHz grid) and are never forced onto frame boundaries.
  * ``source_in`` / ``source_out`` are SOURCE-media seconds. ``source_out - source_in == duration * speed`` for media clips; start trims advance ``source_in`` by
    ``delta * speed`` and leave ``source_out`` alone, end trims leave ``source_in`` alone and move ``source_out``. Source values are rounded to a nanosecond so a trim
    followed by its inverse returns bit-identical numbers. Caption / text / graphic clips have no source range (their source fields are never touched).
  * Keyframes follow ``keyframes.window_keyframes`` (inside the kept window = kept; a boundary keyframe holding the interpolated value is inserted where keyframes were cut away).
  * Text timing stored inside ``clip.text`` (caption ``start`` / ``end`` / ``words[*].start|end``, graphic ``start`` / ``duration``) is kept in step with the clip
    (``sync_text_timing``): a moved caption carries its word timing with it, otherwise the word highlight would run at the old time.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from fractions import Fraction

from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import (
    KIND_CAPTION,
    KIND_GRAPHIC,
    KIND_MEDIA,
    KIND_TEXT,
    LOCK_KINDS,
    Clip,
)
from app.timeline.keyframes import window_keyframes
from app.timeline.ops_common import new_group_id, new_link_id, quantize_delta, relatives
from app.timeline.plan import planner, EditPlan, PlanEnv, clip_fits_track, validate_plan
from app.timeline.timeline import new_clip_id
from app.timeline.timeline_commands import split_keyframes
from app.timeline.track import TrackKind

SAMPLE_RATE = 48000
SOURCE_SPECIFIC_EFFECTS = ("crop", "focus_region")  # belong to one piece of media: dropped when the media is replaced

__all__ = [
    "Grid",
    "fresh_copy",
    "plan_delete",
    "plan_duplicate",
    "plan_insert_free",
    "plan_move",
    "plan_overwrite",
    "plan_replace",
    "plan_set_duration",
    "plan_set_source_range",
    "plan_set_start",
    "plan_split",
    "plan_trim",
    "remap_ids",
    "sync_text_timing",
]


# ------------------------------------------------------------------------------------------------------------------------ grid / small helpers
class Grid:
    """Rounds times to the frame grid (picture, captions, graphics) or to the sample grid (audio-track clips)."""

    __slots__ = ("audio", "min_duration", "sr", "tb")

    def __init__(self, env: PlanEnv, audio: bool, sample_rate: int = SAMPLE_RATE) -> None:
        self.audio, self.tb, self.sr = audio, env.timebase, sample_rate
        self.min_duration = MIN_CLIP_DURATION if audio else self.tb.to_seconds(max(1, math.ceil(MIN_CLIP_DURATION * self.tb.fps - 1e-9)))

    def snap(self, t: float) -> float:
        return self.tb.snap_samples(t, self.sr) if self.audio else self.tb.snap(t)

    def floor(self, t: float) -> float:
        if not math.isfinite(t):
            return t
        return self._samples(t, math.floor, Fraction(1, 1_000_000)) if self.audio else self.tb.snap(t, "floor")

    def ceil(self, t: float) -> float:
        if not math.isfinite(t):
            return t
        return self._samples(t, math.ceil, Fraction(-1, 1_000_000)) if self.audio else self.tb.snap(t, "ceil")

    def _samples(self, t: float, fn, off: Fraction) -> float:
        return float(Fraction(fn(Fraction(float(t)) * self.sr + off), self.sr))


def _is_audio_track(env: PlanEnv, track_id: str) -> bool:
    return env.timeline.get_track(track_id).kind is TrackKind.AUDIO


def _grid(env: PlanEnv, clip: Clip, track_id: str | None = None) -> Grid:
    return Grid(env, _is_audio_track(env, track_id or clip.track_id))


def _finite(value: float, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TimelineError(f"The {what} is not a number.") from None
    if not math.isfinite(v):
        raise TimelineError(f"The {what} is not a valid number.")
    return v


def _r(x: float) -> float:
    """Source-media seconds are kept to a nanosecond: additions of frame deltas never accumulate floating-point noise."""
    return round(x, 9)


def _fmt(t: float) -> str:
    return f"{t:.2f} s"


def _plural(n: int, word: str = "clip") -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _source_info(env: PlanEnv, clip: Clip) -> tuple[bool, float | None]:
    """(does the clip play a limited source range, length of the source media or None). Images and caption / text / graphic clips have no source limit."""
    if clip.kind != KIND_MEDIA:
        return False, None
    info = env.asset_info(clip.asset_id)
    if info is None:
        return True, None
    if info[0] == "image":
        return False, None
    return True, info[1]


def _lock_problem(env: PlanEnv, clip: Clip, aspects: tuple[str, ...], action: str) -> str | None:
    tr = env.timeline.get_track(clip.track_id)
    name = env.label(clip)
    if tr.locked:
        return f"Track “{tr.name}” is locked, so “{name}” cannot be {action}. Unlock the track first."
    if clip.locks and (("item" in clip.locks) or (set(aspects) & clip.locks)):
        return f"“{name}” is locked ({', '.join(sorted(clip.locks))}), so it cannot be {action}. Unlock it first."
    return None


def _check_editable(env: PlanEnv, clip: Clip, *aspects: str, action: str = "edited") -> None:
    msg = _lock_problem(env, clip, aspects, action)
    if msg:
        raise TimelineError(msg)


def _check_destination(env: PlanEnv, clip: Clip, track_id: str) -> None:
    dest = env.timeline.get_track(track_id)
    name = env.label(clip)
    if dest.locked:
        raise TimelineError(f"Track “{dest.name}” is locked, so “{name}” cannot be moved onto it. Unlock the track first.")
    info = env.asset_info(clip.asset_id) if clip.kind == KIND_MEDIA else None
    if clip.kind == KIND_MEDIA and info is None:
        raise TimelineError(f"The media of “{name}” is missing from the project.")
    if not clip_fits_track(clip, dest, info[0] if info else None):
        raise TimelineError(f"“{name}” cannot be moved to track “{dest.name}”: that track does not hold this kind of clip.")


# ------------------------------------------------------------------------------------------------------------------------ text timing inside clips
def sync_text_timing(clip: Clip, shift: float = 0.0) -> None:
    """Keep the timing copies stored in ``clip.text`` in step with the clip (in place; the clip must be a private snapshot).

    ``shift`` is how far the clip moved on the timeline: caption word timings move with it. Trimming passes 0 (the words stay where they were spoken)."""
    t = clip.text
    if not isinstance(t, dict):
        return
    if clip.kind == KIND_CAPTION:
        if "start" in t:
            t["start"] = clip.timeline_start
        if "end" in t:
            t["end"] = clip.timeline_end
        if abs(shift) > 1e-12 and isinstance(t.get("words"), list):
            for w in t["words"]:
                if isinstance(w, dict):
                    for k in ("start", "end"):
                        if isinstance(w.get(k), (int, float)):
                            w[k] = round(w[k] + shift, 6)
    elif clip.kind in (KIND_TEXT, KIND_GRAPHIC):
        if "start" in t:
            t["start"] = clip.timeline_start
        if "duration" in t:
            t["duration"] = clip.duration


def _word_index(words: list[dict], t: float) -> tuple[int, bool]:
    """How many words belong before a cut at absolute time ``t``, and whether a word straddled the cut (it then goes to the nearer side)."""
    k, straddled = 0, False
    for w in words:
        a, b = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        if b <= t + TIME_EPSILON:
            k += 1
        elif a < t - TIME_EPSILON:
            straddled = True
            if (t - a) >= (b - t):
                k += 1
            break
        else:
            break
    return k, straddled


def _text_slice(t: dict, lo: int, hi: int) -> dict:
    """The caption / text dict restricted to words ``lo:hi`` (text, lines, emphasis marks re-indexed)."""
    out = copy.deepcopy(t)
    words = copy.deepcopy(t["words"][lo:hi])
    out["words"] = words
    joined = " ".join(str(w.get("text", "")) for w in words)
    for key in ("text", "content"):
        if key in out:
            out[key] = joined
    lines = t.get("lines")
    if isinstance(lines, list) and sum(len(str(x).split()) for x in lines) == len(t["words"]):
        new, idx = [], 0
        for line in lines:
            toks = str(line).split()
            sub = toks[max(lo - idx, 0):max(min(hi - idx, len(toks)), 0)]
            if sub:
                new.append(" ".join(sub))
            idx += len(toks)
        out["lines"] = new or [joined]
    elif "lines" in out:
        out["lines"] = [joined]
    if isinstance(t.get("emphasis"), list):
        out["emphasis"] = [{**copy.deepcopy(m), "word_index": int(m["word_index"]) - lo} for m in t["emphasis"] if isinstance(m, dict) and lo <= int(m.get("word_index", -1)) < hi]
    return out


# ------------------------------------------------------------------------------------------------------------------------ pieces of a clip
def _cut_clip(env: PlanEnv, src: Clip, s: float, e: float, *, kfs=None, words: tuple[int, int] | None = None) -> Clip:
    """A copy of ``src`` limited to the timeline window [s, e]: source range mapped, keyframes windowed, audio fades kept only at the original edges, transition only at the original start."""
    c = src.snapshot()
    c.timeline_start, c.duration = s, e - s
    at_start, at_end = abs(s - src.timeline_start) < 1e-9, abs(e - src.timeline_end) < 1e-9
    if src.kind == KIND_MEDIA:
        c.source_in = src.source_in if at_start else _r(src.source_in + (s - src.timeline_start) * src.speed)
        c.source_out = src.source_out if at_end else _r(src.source_in + (e - src.timeline_start) * src.speed)
    if kfs is not None:
        c.keyframes = list(kfs)
    elif src.keyframes and not (at_start and at_end):
        c.keyframes = window_keyframes(src.keyframes, s - src.timeline_start, e - src.timeline_start)
    if not at_start:
        c.transition = None
        if c.audio.get("fade_in"):
            c.audio["fade_in"] = 0.0
    if not at_end and c.audio.get("fade_out"):
        c.audio["fade_out"] = 0.0
    if words is not None and isinstance(src.text, dict) and isinstance(src.text.get("words"), list):
        c.text = _text_slice(src.text, words[0], words[1])
    sync_text_timing(c)
    if words is not None and isinstance(c.text, dict) and "reading_cps" in c.text:
        c.text["reading_cps"] = round(len(str(c.text.get("text", ""))) / max(c.duration, 0.25), 2)
    return c


def _make_piece(c: Clip, src: Clip, new_id: str) -> Clip:
    """Turn a cut copy into the NEW right-hand piece of ``src`` (same rules as ``SplitClipCommand``)."""
    c.id = new_id
    c.slot = (src.slot + ".r") if src.slot else ""
    c.ai_decision_id = ""
    c.locked = False
    c.created_by = "USER" if src.scene_id else src.created_by
    c.transition = None
    return c


def remap_ids(clips: list[Clip], attr: str, fresh) -> None:
    """Replace every ``link_id`` / ``group_id`` by a fresh id shared by exactly the clips that shared the old one; an id left on a single clip is cleared (no dangling relations)."""
    counts = Counter(getattr(c, attr) for c in clips if getattr(c, attr))
    mapping = {old: fresh() for old, n in counts.items() if n >= 2}
    for c in clips:
        setattr(c, attr, mapping.get(getattr(c, attr), ""))


def fresh_copy(src: Clip, *, start: float, track_id: str) -> Clip:
    """An independent copy of ``src`` for duplicate / paste: new id, USER-owned, no AI link, no locks. Link / group ids are NOT remapped here (``remap_ids`` does it for the whole set)."""
    c = src.snapshot()
    c.id = new_clip_id()
    c.track_id = track_id
    shift = start - src.timeline_start
    c.timeline_start = start
    c.created_by, c.ai_decision_id, c.locked, c.locks = "USER", "", False, frozenset()
    c.slot = (src.slot + ".copy") if src.slot else ""
    sync_text_timing(c, shift)
    return c


# ------------------------------------------------------------------------------------------------------------------------ move
def _shifted(env: PlanEnv, clip: Clip, d: float, grid: Grid) -> float:
    t = clip.timeline_start
    if grid.audio:
        return grid.snap(t + d)
    nf = round(d * grid.tb.fps)
    if abs(d - grid.tb.to_seconds(nf)) < 1e-9 and grid.tb.is_on_frame(t):
        return grid.tb.add_frames(t, nf)
    return t + d


def _move_plan(env: PlanEnv, clips: list[Clip], d: float, dest: dict[str, str], description: str, *, clamp_at_zero: bool, audio_only: bool, exact: dict[str, float] | None = None) -> EditPlan:
    grids = {c.id: _grid(env, c, dest[c.id]) for c in clips}
    exact = exact or {}

    def starts_for(delta: float) -> dict[str, float]:
        return {c.id: exact.get(c.id, _shifted(env, c, delta, grids[c.id])) for c in clips}

    notes: list[str] = []
    starts = starts_for(d)
    low_id = min(starts, key=starts.get)
    if starts[low_id] < -TIME_EPSILON:
        who = env.label(next(c for c in clips if c.id == low_id))
        if not clamp_at_zero or exact:
            raise TimelineError(f"“{who}” cannot start before 0:00. Move it less far to the left.")
        need = -starts[low_id]
        fix = env.timebase.to_seconds(env.timebase.to_frame(need, "ceil")) if not audio_only else need
        starts = starts_for(d + fix)
        if min(starts.values()) < -TIME_EPSILON:
            raise TimelineError(f"“{who}” cannot start before 0:00. Move it less far to the left.")
        notes.append(f"The selection was stopped at the start of the timeline: it moved {_fmt(d + fix)} instead of {_fmt(d)}.")
    updates: dict[str, Clip] = {}
    for c in clips:
        new_start, new_track = starts[c.id], dest[c.id]
        if abs(new_start - c.timeline_start) <= 1e-12 and new_track == c.track_id:
            continue
        n = c.snapshot()
        n.timeline_start, n.track_id = new_start, new_track
        sync_text_timing(n, new_start - c.timeline_start)
        updates[c.id] = n
    if not updates:
        return EditPlan(description, notes=[*notes, "Nothing moved."])
    return EditPlan(description, updates=updates, notes=notes)


@planner
def plan_move(env: PlanEnv, ids: list[str], delta: float, *, track_moves: dict[str, str] | None = None, linked: bool = True, grouped: bool = True,
              clamp_at_zero: bool = False, description: str = "Move clips") -> EditPlan:
    """Move every clip (plus linked / grouped relatives) by the SAME delta, optionally onto other tracks (``track_moves``: clip id -> destination track id).

    The delta is rounded to whole frames unless every moved clip ends up on an audio track (then it keeps sample precision), so relative spacing is exact. A move that would
    put a clip before 0:00 is rejected, or - with ``clamp_at_zero`` - shortened for the whole selection (spacing preserved) and explained in the notes. Overlaps and track
    locks are judged on the RESULT by ``PlanCommand``, so a selection may move onto its own old positions."""
    tl = env.timeline
    delta = _finite(delta, "move distance")
    all_ids = relatives(tl, list(ids), linked=linked, grouped=grouped)
    if not all_ids:
        return EditPlan(description, notes=["No clips are selected."])
    clips = [tl.find_clip(i)[1] for i in all_ids]
    wanted = set(all_ids)
    moves = {k: v for k, v in (track_moves or {}).items() if k in wanted}
    dest = {c.id: moves.get(c.id, c.track_id) for c in clips}
    for c in clips:
        _check_editable(env, c, "position", action="moved")
        if dest[c.id] != c.track_id:
            _check_destination(env, c, dest[c.id])
    audio_only = all(_is_audio_track(env, dest[c.id]) for c in clips)
    d = quantize_delta(env, delta, audio=audio_only)
    if len(clips) == 1 and description == "Move clips":
        description = "Move clip"
    return _move_plan(env, clips, d, dest, description, clamp_at_zero=clamp_at_zero, audio_only=audio_only)


@planner
def plan_set_start(env: PlanEnv, clip_id: str, new_start: float, *, linked: bool = True, grouped: bool = False) -> EditPlan:
    """Numeric edit of a clip's timeline start (the inspector). The start snaps to the frame (or sample) grid; linked clips follow by the same amount.
    Moving never changes the source range or the duration."""
    tl = env.timeline
    clip = tl.find_clip(clip_id)[1]
    name = env.label(clip)
    target = _grid(env, clip).snap(_finite(new_start, "start time"))
    if target < -TIME_EPSILON:
        raise TimelineError(f"“{name}” cannot start before 0:00.")
    all_ids = relatives(tl, [clip_id], linked=linked, grouped=grouped)
    clips = [tl.find_clip(i)[1] for i in all_ids]
    for c in clips:
        _check_editable(env, c, "position", action="moved")
    dest = {c.id: c.track_id for c in clips}
    audio_only = all(_is_audio_track(env, c.track_id) for c in clips)
    return _move_plan(env, clips, target - clip.timeline_start, dest, "Set clip start", clamp_at_zero=False, audio_only=audio_only, exact={clip_id: target})


# ------------------------------------------------------------------------------------------------------------------------ trim
def _edge_limits(env: PlanEnv, clip: Clip, edge: str, grid: Grid) -> tuple[float, str, float, str]:
    """(lowest, why, highest, why): where this edge may sit. Bounds are rounded INWARD onto the grid, but never stricter than the current position."""
    prev_end, next_start = env.timeline.neighbours(clip)
    s, e = clip.timeline_start, clip.timeline_end
    bound, total = _source_info(env, clip)
    if edge == "start":
        lo, lo_why = (prev_end, "the previous clip") if prev_end > TIME_EPSILON else (0.0, "the start of the timeline")
        if bound and s - clip.source_in / clip.speed > lo:
            lo, lo_why = s - clip.source_in / clip.speed, "the start of the source media"
        hi, hi_why = e - grid.min_duration, "the minimum clip length"
        cur = s
    else:
        lo, lo_why = s + grid.min_duration, "the minimum clip length"
        hi, hi_why = math.inf, ""
        if next_start is not None:
            hi, hi_why = next_start, "the next clip"
        if bound and total is not None and s + (total - clip.source_in) / clip.speed < hi:
            hi, hi_why = s + (total - clip.source_in) / clip.speed, "the end of the source media"
        cur = e
    lo_q, hi_q = grid.ceil(lo), grid.floor(hi)
    if cur >= lo - TIME_EPSILON:
        lo_q = min(lo_q, cur)
    if cur <= hi + TIME_EPSILON:
        hi_q = max(hi_q, cur)
    return lo_q, lo_why, hi_q, hi_why


def _with_edge(env: PlanEnv, clip: Clip, edge: str, t: float) -> Clip:
    """``clip`` with one edge moved to timeline time ``t`` (source range, keyframes and text timing follow)."""
    c = clip.snapshot()
    s, e = clip.timeline_start, clip.timeline_end
    still = clip.kind == KIND_MEDIA and not _source_info(env, clip)[0] and env.asset_info(clip.asset_id) is not None  # a picture: no source timeline to map
    if edge == "start":
        c.timeline_start, c.duration = t, e - t
        if still:
            c.source_out = _r(clip.source_in + c.duration * clip.speed)
        elif clip.kind == KIND_MEDIA:
            c.source_in = max(0.0, _r(clip.source_in + (t - s) * clip.speed))  # source_out stays
        if clip.keyframes and abs(t - s) > 1e-12:
            c.keyframes = window_keyframes(clip.keyframes, t - s, clip.duration)
    else:
        c.duration = t - s
        if clip.kind == KIND_MEDIA:
            c.source_out = _r(clip.source_in + c.duration * clip.speed)
        if clip.keyframes and c.duration < clip.duration - 1e-12:
            c.keyframes = window_keyframes(clip.keyframes, 0.0, c.duration)
    sync_text_timing(c)
    return c


@planner
def plan_trim(env: PlanEnv, clip_id: str, edge: str, new_time: float, *, linked: bool = True, snap: bool = True, description: str | None = None) -> EditPlan:
    """Trim one edge of a clip to a TIMELINE time.

    ``edge="start"``: ``timeline_start`` and ``source_in`` (by ``delta * speed``) and ``duration`` change, ``source_out`` stays. ``edge="end"``: ``duration`` and ``source_out`` change, the rest stays.
    ``snap`` rounds the time to the frame (picture) / sample (audio) grid; it does not look for other clips' edges (the snapping engine does that before calling).
    The edge is clamped to: the minimum clip length, the source media (images have no limit), the neighbouring clip on the track and 0:00 - explained in ``notes``.
    Linked clips trim by the same amount, limited by whichever member allows the least; if nothing can move, ``TimelineError`` names the limiting clip."""
    if edge not in ("start", "end"):
        raise TimelineError("Choose the start or the end edge of the clip.")
    tl = env.timeline
    clip = tl.find_clip(clip_id)[1]
    members = [tl.find_clip(i)[1] for i in relatives(tl, [clip_id], linked=linked, grouped=False)]
    aspects = ("timing", "position") if edge == "start" else ("timing",)
    for m in members:
        _check_editable(env, m, *aspects, action="trimmed")
    new_time = _finite(new_time, "trim time")
    primary_grid = _grid(env, clip)
    cur_primary = clip.timeline_start if edge == "start" else clip.timeline_end
    target = primary_grid.snap(new_time) if snap else new_time
    want = target - cur_primary
    desc = description or (f"Trim clip {edge}" if len(members) == 1 else f"Trim {_plural(len(members))} {edge}")
    if abs(want) <= 1e-12:
        return EditPlan(desc, notes=["The edge is already there."])
    info = []  # (clip, grid, lowest abs, why, highest abs, why, current edge)
    for m in members:
        g = _grid(env, m)
        lo, lo_w, hi, hi_w = _edge_limits(env, m, edge, g)
        info.append((m, g, lo, lo_w, hi, hi_w, m.timeline_start if edge == "start" else m.timeline_end))
    lo_b = max(info, key=lambda x: x[2] - x[6])
    hi_b = min(info, key=lambda x: x[4] - x[6])
    dmin, dmax = lo_b[2] - lo_b[6], hi_b[4] - hi_b[6]
    if dmin > dmax + TIME_EPSILON:
        raise TimelineError(f"“{env.label(lo_b[0])}” and “{env.label(hi_b[0])}” are linked but cannot be trimmed together: one is limited by {lo_b[3]}, the other by {hi_b[5]}.")
    delta = min(max(want, dmin), dmax)
    binding = lo_b if delta > want else hi_b
    why = binding[3] if delta > want else binding[5]
    if abs(delta) <= TIME_EPSILON:
        raise TimelineError(f"“{env.label(binding[0])}” cannot be trimmed any further: it is limited by {why}.")
    notes: list[str] = []
    if abs(delta - want) > TIME_EPSILON:
        who = env.label(binding[0])
        notes.append(f"The {edge} edge stopped at {_fmt(cur_primary + delta)} instead of {_fmt(target)} because of {why}" + (f" (“{who}”)." if len(members) > 1 or binding[0] is not clip else "."))
    updates: dict[str, Clip] = {}
    for m, g, lo, _lw, hi, _hw, cur in info:
        t = cur + delta
        if snap:
            t = g.snap(t)
        t = min(max(t, lo), hi)
        updates[m.id] = _with_edge(env, m, edge, t)
    return EditPlan(desc, updates=updates, notes=notes)


@planner
def plan_set_duration(env: PlanEnv, clip_id: str, duration: float, *, linked: bool = True) -> EditPlan:
    """Numeric duration (the inspector): moves the END edge. ``source_out`` follows (``source_in + duration * speed``); the duration is limited by the source media and the next clip
    (clamped, with a note) and by the minimum clip length."""
    clip = env.timeline.find_clip(clip_id)[1]
    duration = _finite(duration, "duration")
    if duration <= 0:
        raise TimelineError("The duration must be longer than zero.")
    return plan_trim(env, clip_id, "end", clip.timeline_start + duration, linked=linked, snap=True, description="Set clip duration")


@planner
def plan_set_source_range(env: PlanEnv, clip_id: str, source_in: float | None = None, source_out: float | None = None) -> EditPlan:
    """Numeric edit of the SOURCE in / out points (source-media seconds). The timeline start stays; the duration becomes ``(source_out - source_in) / speed`` (rounded to the
    frame grid), so the END edge moves. Only this clip is changed (the source belongs to its own media). Out-of-range numbers are rejected; a range that runs past the end of the
    media or into the next clip is clamped and explained. To slide the content without changing the duration use the slip tool."""
    tl = env.timeline
    clip = tl.find_clip(clip_id)[1]
    name = env.label(clip)
    bound, total = _source_info(env, clip)
    if not bound:
        raise TimelineError(f"“{name}” has no source range to edit (still images and text have none). Change its duration instead.")
    if source_in is None and source_out is None:
        raise TimelineError("Give a source in or out point.")
    _check_editable(env, clip, "timing", action="re-timed")
    si = clip.source_in if source_in is None else _finite(source_in, "source in point")
    so = clip.source_out if source_out is None else _finite(source_out, "source out point")
    if si < 0:
        raise TimelineError("The source in point cannot be before the start of the media.")
    if so <= si:
        raise TimelineError("The source out point must be after the source in point.")
    notes: list[str] = []
    if total is not None:
        if si >= total - TIME_EPSILON:
            raise TimelineError(f"The source in point is past the end of the media ({_fmt(total)}).")
        if so > total + TIME_EPSILON:
            so = total
            notes.append(f"The source out point was limited to the end of the media ({_fmt(total)}).")
    grid = _grid(env, clip)
    s = clip.timeline_start
    end_t = grid.snap(s + (so - si) / clip.speed)
    if total is not None and si + (end_t - s) * clip.speed > total + 1e-9:
        end_t = grid.floor(s + (total - si) / clip.speed)
    _prev, nxt = tl.neighbours(clip)
    if nxt is not None and end_t > nxt + TIME_EPSILON:
        end_t = max(grid.floor(nxt), min(end_t, clip.timeline_end))
        notes.append("The clip was limited by the next clip on the track, so its source out point is earlier than requested.")
    if end_t - s < grid.min_duration - TIME_EPSILON:
        raise TimelineError(f"That source range is shorter than the minimum clip length ({MIN_CLIP_DURATION:g} s).")
    n = clip.snapshot()
    n.duration = end_t - s
    n.source_in, n.source_out = _r(si), _r(si + n.duration * clip.speed)
    if abs(n.source_out - so) > 1e-4:
        notes.append(f"The source out point moved to {_fmt(n.source_out)} to end on a whole frame.")
    if clip.keyframes and n.duration < clip.duration - 1e-12:
        n.keyframes = window_keyframes(clip.keyframes, 0.0, n.duration)
    sync_text_timing(n)
    return EditPlan("Set source range", updates={clip_id: n}, notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ split
def _split_one(env: PlanEnv, clip: Clip, cut: float, grid: Grid, notes: list[str]):
    """(left, right) halves of ``clip`` at ``cut``, or a plain-language reason string when it cannot be split there."""
    s, e = clip.timeline_start, clip.timeline_end
    name = env.label(clip)
    if not (s < cut < e):
        return f"The cut point is not inside “{name}”."
    md = max(grid.min_duration, MIN_CLIP_DURATION)
    if cut - s < md - TIME_EPSILON or e - cut < md - TIME_EPSILON:
        return f"The cut point is too close to the edge of “{name}”: each part must be at least {md:.2f} s long."
    wl = wr = None
    if clip.kind != KIND_MEDIA:
        words = clip.text.get("words") if isinstance(clip.text, dict) else None
        if not isinstance(words, list) or not words:
            return f"“{name}” is a single block of text without word timing, so it cannot be split. Change its duration or duplicate it instead."
        k, straddled = _word_index(words, cut)
        if k <= 0 or k >= len(words):
            return f"“{name}” cannot be split there: one part would have no words."
        if straddled:
            notes.append(f"“{name}”: the cut falls inside a word, so the text was split between the nearest words.")
        wl, wr = (0, k), (k, len(words))
    kl, kr = (None, None)
    if clip.keyframes:
        a, b = split_keyframes(clip.keyframes, cut - s, clip.duration)
        kl, kr = sorted(a, key=lambda k: (k.property, k.time)), sorted(b, key=lambda k: (k.property, k.time))
    left = _cut_clip(env, clip, s, cut, kfs=kl, words=wl)
    right = _make_piece(_cut_clip(env, clip, cut, e, kfs=kr, words=wr), clip, new_clip_id())
    return left, right


@planner
def plan_split(env: PlanEnv, ids: list[str], at: float, *, linked: bool = True, include_intersecting_on: list[str] | None = None, snap: bool = True) -> EditPlan:
    """Split clips at ONE timeline time.

    Targets: ``ids`` plus linked relatives plus every clip on the tracks of ``include_intersecting_on`` that spans ``at`` (locked tracks / clips there are skipped, with a note).
    The cut snaps to the frame grid (sample grid when only audio is involved) so linked halves stay aligned. Each right half gets a fresh id, the same asset and a continuous
    source mapping (left ``source_out`` == right ``source_in``), split keyframes, no transition and the AI fields handled like ``SplitClipCommand``. The new right halves of linked
    clips share a NEW link id (the left halves keep the old one). Captions / texts are split between words using their word timing (a cut inside a word is moved to the nearest
    word boundary and noted); a text block without word timing cannot be split. Clips that cannot be cut at that time are skipped with a note; if none can, the reason is raised."""
    tl = env.timeline
    at = _finite(at, "split time")
    explicit = relatives(tl, list(ids), linked=linked, grouped=False)
    extras: list[str] = []
    notes: list[str] = []
    skipped_locked = 0
    for tid in include_intersecting_on or ():
        tr = tl.get_track(tid)
        for c in tl.clips_in_range(tid, at - TIME_EPSILON, at + TIME_EPSILON):
            if c.id in explicit or c.id in extras or not (c.timeline_start + TIME_EPSILON < at < c.timeline_end - TIME_EPSILON):
                continue
            if tr.locked or _lock_problem(env, c, ("timing",), "split"):
                skipped_locked += 1
                continue
            extras.append(c.id)
    if linked and extras:  # a blade on one track also cuts the linked partners (the picture's own audio) that span the cut
        for rid in relatives(tl, list(extras), linked=True, grouped=False):
            if rid in explicit or rid in extras:
                continue
            mate = tl.find_clip(rid)[1]
            if not (mate.timeline_start + TIME_EPSILON < at < mate.timeline_end - TIME_EPSILON):
                continue
            if tl.get_track(mate.track_id).locked or _lock_problem(env, mate, ("timing",), "split"):
                skipped_locked += 1
                continue
            extras.append(rid)
    if skipped_locked:
        notes.append(f"{_plural(skipped_locked)} on locked tracks or with locked timing {'was' if skipped_locked == 1 else 'were'} left unsplit.")
    targets = [tl.find_clip(i)[1] for i in [*explicit, *extras]]
    if not targets:
        raise TimelineError("No clip is selected to split." if not skipped_locked else "Every clip under that point is locked.")
    if snap:
        cut = Grid(env, all(_is_audio_track(env, c.track_id) for c in targets)).snap(at)
    else:
        cut = at
    pairs: list[tuple[Clip, Clip]] = []
    reasons: list[str] = []
    for c in targets:
        r = _split_one(env, c, cut, _grid(env, c), notes)
        if isinstance(r, str):
            reasons.append(r)
        else:
            _check_editable(env, c, "timing", action="split")
            pairs.append(r)
    if not pairs:
        raise TimelineError(reasons[0])
    notes.extend(r for r in reasons if r not in notes)
    rights = [r for _l, r in pairs]
    by_link: dict[str, list[Clip]] = {}
    for (left, right) in pairs:
        if left.link_id:
            by_link.setdefault(left.link_id, []).append(right)
    for group in by_link.values():
        fresh = new_link_id() if len(group) >= 2 else ""
        for r in group:
            r.link_id = fresh
    n = len(pairs)
    return EditPlan("Split clip" if n == 1 else f"Split {n} clips", updates={left.id: left for left, _r in pairs}, adds=rights, notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ delete / duplicate
@planner
def plan_delete(env: PlanEnv, ids: list[str], *, linked: bool = True, grouped: bool = True) -> EditPlan:
    """Remove clips plus linked / grouped relatives (no ripple: the gap stays)."""
    tl = env.timeline
    all_ids = relatives(tl, list(ids), linked=linked, grouped=grouped)
    if not all_ids:
        return EditPlan("Delete clips", notes=["No clips are selected."])
    for i in all_ids:
        _check_editable(env, tl.find_clip(i)[1], *LOCK_KINDS, action="deleted")
    return EditPlan("Delete clip" if len(all_ids) == 1 else f"Delete {len(all_ids)} clips", removes=all_ids)


def _place_copies(env: PlanEnv, src: list[Clip], shift_start: float, tracks: dict[str, str], description: str) -> EditPlan:
    """Copies of ``src`` whose earliest start becomes ``shift_start``; link / group ids are remapped so the copy is an independent set."""
    t0 = min(c.timeline_start for c in src)
    off = shift_start - t0
    adds: list[Clip] = []
    for c in sorted(src, key=lambda c: (c.timeline_start, c.id)):
        tid = tracks.get(c.id, c.track_id)
        g = _grid(env, c, tid)
        new_start = _shifted(env, c, off, g)
        adds.append(fresh_copy(c, start=new_start, track_id=tid))
    if min(c.timeline_start for c in adds) < -TIME_EPSILON:
        raise TimelineError("The copy would start before 0:00.")
    remap_ids(adds, "link_id", new_link_id)
    remap_ids(adds, "group_id", new_group_id)
    plan = EditPlan(description, adds=adds)
    validate_plan(env, plan)  # a copy that would collide / land on a locked or unsuitable track is refused here with the reason
    return plan


@planner
def plan_duplicate(env: PlanEnv, ids: list[str], *, offset: float | None = None, to_start: float | None = None, target_tracks: dict[str, str] | None = None,
                   linked: bool = True, grouped: bool = True) -> EditPlan:
    """Independent copies of the clips (plus linked / grouped relatives): fresh ids, USER-owned, no AI decision, no locks, link / group ids remapped to fresh ids, ``scene_id`` kept,
    a non-empty ``slot`` gets a ``.copy`` suffix so AI regeneration never mistakes the copy for the original. Placement: ``to_start`` (earliest copy starts there), else ``offset``
    seconds after the originals, else right after the selection's end on the same tracks (``target_tracks``: clip id -> track id). Rejected with the reason if the copies collide."""
    tl = env.timeline
    if offset is not None and to_start is not None:
        raise TimelineError("Give either an offset or a start time for the copy, not both.")
    all_ids = relatives(tl, list(ids), linked=linked, grouped=grouped)
    if not all_ids:
        raise TimelineError("Select at least one clip to duplicate.")
    src = [tl.find_clip(i)[1] for i in all_ids]
    wanted = set(all_ids)
    tracks = {k: v for k, v in (target_tracks or {}).items() if k in wanted}
    for c in src:
        if tracks.get(c.id, c.track_id) != c.track_id:
            _check_destination(env, c, tracks[c.id])
    t0, t1 = min(c.timeline_start for c in src), max(c.timeline_end for c in src)
    audio_only = all(_is_audio_track(env, tracks.get(c.id, c.track_id)) for c in src)
    if to_start is not None:
        start0 = Grid(env, audio_only).snap(_finite(to_start, "start time"))
    elif offset is not None:
        start0 = t0 + quantize_delta(env, _finite(offset, "offset"), audio=audio_only)
    else:
        start0 = t0 + quantize_delta(env, t1 - t0, audio=audio_only)
    if start0 < -TIME_EPSILON:
        raise TimelineError("The copy would start before 0:00.")
    return _place_copies(env, src, start0, tracks, "Duplicate clip" if len(src) == 1 else f"Duplicate {len(src)} clips")


# ------------------------------------------------------------------------------------------------------------------------ overwrite
def _subtract(s: float, e: float, ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cur = s
    for a, b in ivs:
        if b <= cur + TIME_EPSILON:
            continue
        if a >= e - TIME_EPSILON:
            break
        if a > cur + TIME_EPSILON:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < e - TIME_EPSILON:
        out.append((cur, e))
    return out


@planner
def plan_overwrite(env: PlanEnv, new_clips: list[Clip], *, description: str = "Overwrite") -> EditPlan:
    """Place ``new_clips`` (positions and tracks already set) over whatever is there: existing clips on the same track that they overlap are trimmed (start trim advances
    ``source_in``), split in two (when a new clip lies inside) or removed (when fully covered). Leftovers shorter than the minimum clip length are removed with a note. A locked
    track or clip in the way rejects the whole edit naming it. One plan: adds + updates + removes (one undo step)."""
    tl = env.timeline
    new = [c.snapshot() for c in new_clips]
    plan = EditPlan(description, adds=list(new))
    if not new:
        plan.notes.append("There is nothing to place.")
        return plan
    counts: Counter[str] = Counter()
    by_track: dict[str, list[Clip]] = {}
    for c in new:
        by_track.setdefault(c.track_id, []).append(c)
    for tid, news in by_track.items():
        tr = tl.get_track(tid)
        if tr.locked:
            raise TimelineError(f"Track “{tr.name}” is locked. Unlock it to place clips there.")
        ivs = sorted((c.timeline_start, c.timeline_end) for c in news)
        mine = {c.id for c in news}
        for old in tl.clips_in_range(tid, ivs[0][0], max(b for _a, b in ivs)):
            if old.id in mine:
                continue
            s, e = old.timeline_start, old.timeline_end
            pieces = _subtract(s, e, ivs)
            if len(pieces) == 1 and abs(pieces[0][0] - s) < 1e-9 and abs(pieces[0][1] - e) < 1e-9:
                continue
            name = env.label(old)
            md = max(_grid(env, old).min_duration, MIN_CLIP_DURATION)
            keep = [p for p in pieces if p[1] - p[0] >= md - TIME_EPSILON]
            if len(keep) < len(pieces):
                plan.notes.append(f"A leftover piece of “{name}” was removed because it was shorter than the minimum clip length.")
            if not keep:
                _check_editable(env, old, *LOCK_KINDS, action="overwritten")
                plan.removes.append(old.id)
                counts["removed"] += 1
                continue
            _check_editable(env, old, "timing", *(("position",) if abs(keep[0][0] - s) > 1e-9 else ()), action="overwritten")
            words = old.text.get("words") if isinstance(old.text, dict) else None
            split_text = old.kind != KIND_MEDIA and len(keep) > 1
            if split_text and not (isinstance(words, list) and words):
                raise TimelineError(f"“{name}” would have to be split to make room, but it is a single block of text. Move the new clip or shorten that text clip first.")
            placed = False
            for ps, pe in keep:
                wr = None
                if split_text:
                    lo = 0 if abs(ps - s) < 1e-9 else _word_index(words, ps)[0]
                    hi = len(words) if abs(pe - e) < 1e-9 else _word_index(words, pe)[0]
                    if hi <= lo:
                        plan.notes.append(f"A piece of “{name}” was dropped because it would have no words.")
                        continue
                    wr = (lo, hi)
                piece = _cut_clip(env, old, ps, pe, words=wr)
                if not placed:  # the first surviving piece stays the same clip (same id); further pieces are new clips
                    plan.updates[old.id] = piece
                    placed = True
                else:
                    plan.adds.append(_make_piece(piece, old, new_clip_id()))
            if not placed:
                _check_editable(env, old, *LOCK_KINDS, action="overwritten")
                plan.removes.append(old.id)
            counts["split" if len(keep) > 1 else "trimmed"] += 1
    if counts:
        plan.notes.append("Overwrote existing clips: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + ".")
    return plan


# ------------------------------------------------------------------------------------------------------------------------ replace
@planner
def plan_replace(env: PlanEnv, clip_id: str, new_asset_id: str, *, keep_duration: bool = True) -> EditPlan:
    """Swap the media of a clip, keeping its place, track, transform, effects, animation, keyframes, audio settings and transition.

    The source range restarts at 0 (the old ``source_in`` is kept when the new media is long enough). A shorter new media shortens the clip (noted); ``keep_duration=False`` makes
    the clip as long as the new media allows (limited by the next clip). Speed returns to 1.0 when the clip changes between video and image. ``crop`` / ``focus_region`` effects
    belonged to the old picture and are dropped (noted). The replacement must fit the track. Any lock on the clip blocks the replacement."""
    tl = env.timeline
    track, clip = tl.find_clip(clip_id)
    name = env.label(clip)
    if clip.kind != KIND_MEDIA:
        raise TimelineError("Only clips that play media can be replaced.")
    _check_editable(env, clip, *LOCK_KINDS, action="replaced")
    new_info = env.asset_info(new_asset_id)
    if new_info is None:
        raise TimelineError("The replacement media is not in the project.")
    ntype, ndur, nname = new_info
    if new_asset_id == clip.asset_id:
        return EditPlan("Replace media", notes=[f"“{name}” already uses “{nname}”."])
    if not clip_fits_track(clip, track, ntype):
        raise TimelineError(f"Track “{track.name}” cannot hold {ntype} media, so “{nname}” cannot replace “{name}”.")
    old_info = env.asset_info(clip.asset_id)
    old_type = old_info[0] if old_info else None
    notes: list[str] = []
    n = clip.snapshot()
    n.asset_id = new_asset_id
    if (old_type == "image") != (ntype == "image") and abs(n.speed - 1.0) > 1e-9:
        n.speed = 1.0
        notes.append("The speed was reset to 1.0× because the clip changed between video and picture.")
    grid = _grid(env, clip)
    s, speed = clip.timeline_start, n.speed
    dur = clip.duration
    if ntype == "image":
        si = 0.0
    else:
        si = clip.source_in if old_type != "image" else 0.0
        if not keep_duration:
            si = 0.0
            if ndur:
                dur = grid.floor(ndur / speed)
        if ndur is not None and si + dur * speed > ndur + 1e-9:
            si = 0.0
            if dur * speed > ndur + 1e-9:
                dur = grid.floor(ndur / speed)
                notes.append(f"“{nname}” is only {_fmt(ndur)} long, so the clip was shortened from {_fmt(clip.duration)} to {_fmt(dur)}.")
    _p, nxt = tl.neighbours(clip)
    if nxt is not None and s + dur > nxt + TIME_EPSILON:
        dur = max(grid.floor(nxt - s), grid.min_duration)
        notes.append("The clip was limited by the next clip on the track.")
    if dur < grid.min_duration - TIME_EPSILON:
        raise TimelineError(f"“{nname}” is too short to replace “{name}”.")
    n.duration = dur
    n.source_in, n.source_out = _r(si), _r(si + dur * speed)
    if clip.keyframes and dur < clip.duration - 1e-12:
        n.keyframes = window_keyframes(clip.keyframes, 0.0, dur)
    dropped = [k for k in SOURCE_SPECIFIC_EFFECTS if k in n.effects]
    for k in dropped:
        del n.effects[k]
    if dropped:
        notes.append(f"The {' and '.join(dropped)} setting of the old media was removed.")
    return EditPlan("Replace media", updates={clip_id: n}, notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ free insert
@planner
def plan_insert_free(env: PlanEnv, clips: list[Clip], at: float, track_id: str) -> EditPlan:
    """Non-ripple placement for media dropped from the library: the clips go onto ``track_id`` one after another, each at the first free start from ``at`` (frame/sample grid)."""
    tl = env.timeline
    tr = tl.get_track(track_id)
    if tr.locked:
        raise TimelineError(f"Track “{tr.name}” is locked. Unlock it to add clips.")
    at = _finite(at, "time")
    grid = Grid(env, tr.kind is TrackKind.AUDIO)
    t = max(0.0, grid.snap(at))
    adds: list[Clip] = []
    notes: list[str] = []
    for src in clips:
        c = src.snapshot()
        c.track_id = tr.id
        want = t
        guard = 0
        while True:
            free = tl.first_free_start(tr.id, t, c.duration)
            if free <= t + TIME_EPSILON:
                break
            t = grid.ceil(free)
            guard += 1
            if guard > len(tr.clips) + 2:
                raise TimelineError("No free space was found on that track.")
        if t > want + TIME_EPSILON:
            notes.append(f"“{env.label(c)}” was placed at {_fmt(t)} because the track is occupied before that.")
        c.timeline_start = t
        sync_text_timing(c, t - src.timeline_start)
        adds.append(c)
        t = grid.ceil(c.timeline_end)
    return EditPlan("Add clip" if len(adds) == 1 else f"Add {len(adds)} clips", adds=adds, notes=notes)
