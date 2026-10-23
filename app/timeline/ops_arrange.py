"""Arrangement planners: align several clips to a time or to each other, and distribute them evenly. Every clip (plus the linked / grouped relatives that follow it) moves by a whole number of frames
(samples on audio), so the result is frame-exact and the clips keep their durations, source ranges and tracks. Built on ``plan_move``: overlaps, locks and track compatibility are judged on the
RESULT by ``PlanCommand`` / ``validate_plan`` like every other edit."""

from __future__ import annotations

from typing import Mapping

from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.ops_basic import plan_move
from app.timeline.ops_common import is_audio_track, relatives
from app.timeline.plan import EditPlan, PlanEnv, planner

ALIGN_MODES = ("start", "end", "center", "first_start", "last_end")  # start/end/center need a target time (or a time per clip); the other two use the selection itself
DISTRIBUTE_MODES = ("gaps", "starts", "butt")  # equal gaps between clips | equally spaced starts | close the gaps (each clip starts where the previous one ends)
_EPS = 1e-9


def _clips(env: PlanEnv, ids) -> list[Clip]:
    tl = env.timeline
    out = [tl.find_clip(i)[1] for i in dict.fromkeys(ids)]
    if not out:
        raise TimelineError("Select at least one clip.")
    return out


def _move_each(env: PlanEnv, clips: list[Clip], deltas: Mapping[str, float], description: str, notes: list[str], *, linked: bool, grouped: bool) -> EditPlan:
    """One ``plan_move`` per distinct delta, merged. A clip and its relatives must not be asked to go to two different places."""
    tl = env.timeline
    claimed: dict[str, str] = {}  # relative id -> the selected clip that owns it
    for c in clips:
        for rid in relatives(tl, [c.id], linked=linked, grouped=grouped):
            owner = claimed.setdefault(rid, c.id)
            if owner != c.id and abs(deltas[owner] - deltas[c.id]) > _EPS:
                raise TimelineError(f"“{env.label(c)}” and “{env.label(tl.find_clip(owner)[1])}” are linked or grouped, so they cannot be moved to different places. Unlink them, or select only one of them.")
    by_delta: dict[float, list[str]] = {}
    for c in clips:
        d = deltas[c.id]
        if abs(d) <= _EPS:
            continue
        key = next((k for k in by_delta if abs(k - d) <= _EPS), d)
        by_delta.setdefault(key, []).append(c.id)
    if not by_delta:
        return EditPlan(description, notes=[*notes, "Nothing needed to move."])
    plan: EditPlan | None = None
    for d, ids in by_delta.items():
        part = plan_move(env, ids, d, linked=linked, grouped=grouped, description=description)
        plan = part if plan is None else plan.merge(part)
    assert plan is not None
    plan.description, plan.notes = description, [*notes, *plan.notes]
    return plan


@planner
def plan_align(env: PlanEnv, ids: list[str], mode: str, target: float | Mapping[str, float] | None = None, *, linked: bool = True, grouped: bool = True) -> EditPlan:
    """Align clips: ``start`` / ``end`` / ``center`` to ``target`` (one time for all, or a time per clip id), ``first_start`` to the earliest start of the selection, ``last_end`` so every clip
    ends where the last one ends. Clips on the same track would end up on top of each other, which the validator refuses with the names of the clips involved."""
    if mode not in ALIGN_MODES:
        raise TimelineError(f"Unknown alignment “{mode}”. Choose one of: {', '.join(ALIGN_MODES)}.")
    clips = _clips(env, ids)
    if mode in ("start", "end", "center") and target is None:
        raise TimelineError("An alignment to a time needs that time.")
    first, last = min(c.timeline_start for c in clips), max(c.timeline_end for c in clips)
    deltas: dict[str, float] = {}
    for c in clips:
        t = target.get(c.id) if isinstance(target, Mapping) else target
        if t is None and mode in ("start", "end", "center"):
            raise TimelineError(f"There is no time to align “{env.label(c)}” to.")
        if mode == "start":
            want = t
        elif mode == "end":
            want = t - c.duration
        elif mode == "center":
            want = t - c.duration / 2
        elif mode == "first_start":
            want = first
        else:
            want = last - c.duration
        deltas[c.id] = float(want) - c.timeline_start
    label = {"start": "starts", "end": "ends", "center": "centres", "first_start": "starts", "last_end": "ends"}[mode]
    return _move_each(env, clips, deltas, f"Align {len(clips)} clip{'s' if len(clips) != 1 else ''} ({label})", [], linked=linked, grouped=grouped)


@planner
def plan_distribute(env: PlanEnv, ids: list[str], mode: str = "gaps", *, linked: bool = True, grouped: bool = True) -> EditPlan:
    """Spread clips along the time axis in start order (tracks do not matter). ``gaps``: the first and last clip stay, the ones between get equal gaps; ``starts``: equal distance between
    starts; ``butt``: the first clip stays and every later one starts where the previous one ends (the gaps between the selected clips close). Positions land on the frame grid, so
    gaps may differ by a frame - said in the notes."""
    if mode not in DISTRIBUTE_MODES:
        raise TimelineError(f"Unknown distribution “{mode}”. Choose one of: {', '.join(DISTRIBUTE_MODES)}.")
    clips = sorted(_clips(env, ids), key=lambda c: (c.timeline_start, c.timeline_end, c.id))
    need = 3 if mode in ("gaps", "starts") else 2
    if len(clips) < need:
        raise TimelineError(f"Select at least {need} clips to distribute them" + (" (the first and the last stay where they are)." if need == 3 else "."))
    tb = env.timebase
    new_starts: list[float] = [clips[0].timeline_start]
    notes: list[str] = []
    if mode == "gaps":
        span = clips[-1].timeline_end - clips[0].timeline_start
        gap = (span - sum(c.duration for c in clips)) / (len(clips) - 1)
        if gap < -_EPS:
            raise TimelineError("The clips are longer together than the space between the first and the last one, so they cannot be spread out. Move the last clip further away first.")
        cursor = clips[0].timeline_start
        for c in clips[1:]:
            cursor += clips[len(new_starts) - 1].duration + gap
            new_starts.append(cursor)
    elif mode == "starts":
        step = (clips[-1].timeline_start - clips[0].timeline_start) / (len(clips) - 1)
        new_starts = [clips[0].timeline_start + i * step for i in range(len(clips))]
    else:
        cursor = clips[0].timeline_end
        for c in clips[1:]:
            new_starts.append(cursor)
            cursor += c.duration
    deltas: dict[str, float] = {}
    for c, s in zip(clips, new_starts):
        if c is clips[-1] and mode == "gaps":
            deltas[c.id] = 0.0  # the last clip is the fixed end of the span
        else:
            deltas[c.id] = (s if is_audio_track(env.timeline, c.track_id) else tb.snap(s)) - c.timeline_start  # audio keeps sample precision (plan_move rounds it to samples)
    if mode in ("gaps", "starts"):
        notes.append("Positions are rounded to whole frames, so the spaces can differ by one frame.")
    return _move_each(env, clips, deltas, f"Distribute {len(clips)} clips ({ {'gaps': 'equal gaps', 'starts': 'equal spacing', 'butt': 'no gaps'}[mode] })", notes, linked=linked, grouped=grouped)
