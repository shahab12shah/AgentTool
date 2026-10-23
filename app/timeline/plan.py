"""EditPlan: a pure description of a (possibly multi-clip) timeline edit, and the single transactional applier every advanced edit goes through.

Architecture (Phase 10):

    planner function  ->  EditPlan  ->  validate_plan  ->  PlanCommand.do (atomic apply)  ->  undo stack
    (pure, no mutation)   (states)     (locks, tracks,      (one rebuild per touched track,
                                        overlaps, source     rollback on any failure)
                                        bounds, keyframes)

A planner reads the timeline and returns the *complete new state* of every clip it changes plus clips to add / remove; it never mutates. ``PlanCommand`` validates the
whole result before anything is touched ("validate -> capture -> apply -> verify"), applies it in one step and stores exactly the before-states needed to undo it, so
a multi-clip move, a ripple delete over a thousand clips and a roll edit are each one undo step and either fully happen or do not happen at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

from app.core.commands import Command
from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.timebase import Timebase
from app.timeline.timeline import Timeline
from app.timeline.track import Track, TrackKind

SPEED_RANGE = (0.1, 8.0)
SRC_TOL = 1e-3  # seconds of slack on source-range consistency (float round trips through frame arithmetic)


# ------------------------------------------------------------------------------------------------------------------------ environment
@dataclass
class PlanEnv:
    """What validation needs besides the timeline: asset facts and the project timebase."""

    timeline: Timeline
    asset_info: Callable[[str], tuple[str, float | None, str] | None] = lambda _id: None  # asset id -> (type value, duration or None, display name)
    timebase: Timebase = field(default_factory=lambda: Timebase(30))

    @classmethod
    def for_project(cls, project) -> "PlanEnv":
        from app.timeline.timebase import timebase_of  # noqa: PLC0415

        def info(aid: str):
            a = project.assets.get(aid)
            return None if a is None else (a.type.value, a.duration, a.name)

        return cls(project.timeline, info, timebase_of(project))

    def label(self, clip: Clip) -> str:
        if clip.kind == KIND_MEDIA:
            i = self.asset_info(clip.asset_id)
            if i:
                return i[2]
        if isinstance(clip.text, dict):
            t = str(clip.text.get("text") or clip.text.get("content") or "").strip()
            if t:
                return f"“{t[:24]}”"
        return clip.id


# ------------------------------------------------------------------------------------------------------------------------ plan
@dataclass
class EditPlan:
    description: str
    updates: dict[str, Clip] = field(default_factory=dict)  # clip id -> complete new state (the id never changes; the track may)
    adds: list[Clip] = field(default_factory=list)  # new clips (fresh ids)
    removes: list[str] = field(default_factory=list)  # ids to delete
    notes: list[str] = field(default_factory=list)  # plain-language explanation of what the plan does / adjusted
    major: bool = True  # triggers an autosave when committed
    shifted: int = 0  # how many clips a ripple moved (for the confirmation text)

    @property
    def empty(self) -> bool:
        return not (self.updates or self.adds or self.removes)

    def affected_ids(self) -> set[str]:
        return set(self.updates) | set(self.removes) | {c.id for c in self.adds}

    def merge(self, other: "EditPlan") -> "EditPlan":
        """Combine two plans over disjoint clips (e.g. a move plus its linked members)."""
        dup = (set(self.updates) | set(self.removes)) & (set(other.updates) | set(other.removes))
        if dup:
            raise TimelineError("Two parts of the same edit change the same clip.", details=str(sorted(dup)))
        return EditPlan(self.description, {**self.updates, **other.updates}, self.adds + other.adds, self.removes + other.removes, self.notes + other.notes, self.major or other.major, self.shifted + other.shifted)


# ------------------------------------------------------------------------------------------------------------------------ rules
def changed_aspects(old: Clip, new: Clip) -> set[str]:
    """Which lock kinds an edit from ``old`` to ``new`` touches."""
    out: set[str] = set()
    if old.track_id != new.track_id or abs(old.timeline_start - new.timeline_start) > TIME_EPSILON:
        out.add("position")
    if abs(old.duration - new.duration) > TIME_EPSILON or abs(old.source_in - new.source_in) > TIME_EPSILON or abs(old.source_out - new.source_out) > TIME_EPSILON \
            or abs(old.speed - new.speed) > 1e-9:
        out.add("timing")
    if old.position != new.position or old.scale != new.scale or old.rotation != new.rotation or old.opacity != new.opacity:
        out.add("transform")
    if old.audio != new.audio:
        out.add("audio")
    if old.text != new.text or old.animation != new.animation or old.effects != new.effects or old.transition != new.transition:
        out.add("style")
    if [k.__dict__ for k in old.keyframes] != [k.__dict__ for k in new.keyframes]:
        changed = {k.property for k in old.keyframes} ^ {k.property for k in new.keyframes}
        if not changed:  # same properties: find which ones differ
            a = {(k.property, k.time, k.value, k.interpolation) for k in old.keyframes}
            b = {(k.property, k.time, k.value, k.interpolation) for k in new.keyframes}
            changed = {x[0] for x in a ^ b}
        out.add("audio" if changed and changed <= {"volume"} else "transform")
    return out


def clip_fits_track(clip: Clip, track: Track, asset_type: str | None) -> bool:
    if clip.kind == KIND_MEDIA:
        from app.media.asset import AssetType  # noqa: PLC0415

        try:
            return asset_type is not None and track.accepts(AssetType(asset_type))
        except ValueError:
            return False
    if clip.kind == KIND_CAPTION:
        return track.kind is TrackKind.CAPTIONS
    if clip.kind in (KIND_TEXT, KIND_GRAPHIC):
        return track.kind in (TrackKind.TEXT, TrackKind.GRAPHICS)
    return True


def _verb(aspects: Iterable[str]) -> str:
    return {"position": "move it", "timing": "change its timing", "transform": "change its transform", "audio": "change its audio", "style": "change its style", "item": "edit it"}.get(next(iter(sorted(aspects))), "edit it")


def check_clip_numbers(env: PlanEnv, clip: Clip, old: Clip | None) -> None:
    """Numeric sanity of one new / changed clip state (raises ``TimelineError`` with the reason)."""
    name = env.label(clip)
    for v in (clip.timeline_start, clip.duration, clip.source_in, clip.source_out, clip.speed):
        if not math.isfinite(v):
            raise TimelineError(f"“{name}” would get an invalid time value.")
    if clip.timeline_start < -TIME_EPSILON:
        raise TimelineError(f"“{name}” cannot start before 0:00.")
    if clip.duration < MIN_CLIP_DURATION - TIME_EPSILON:
        raise TimelineError(f"“{name}” would be shorter than the minimum clip length ({MIN_CLIP_DURATION:g} s).")
    if not (SPEED_RANGE[0] - 1e-9 <= clip.speed <= SPEED_RANGE[1] + 1e-9):
        raise TimelineError(f"Speed for “{name}” must be between {SPEED_RANGE[0]:g}× and {SPEED_RANGE[1]:g}×.")
    timing_changed = old is None or abs(old.duration - clip.duration) > TIME_EPSILON or abs(old.source_in - clip.source_in) > TIME_EPSILON \
        or abs(old.source_out - clip.source_out) > TIME_EPSILON or abs(old.speed - clip.speed) > 1e-9
    if timing_changed and clip.kind == KIND_MEDIA:
        info = env.asset_info(clip.asset_id)
        if info is not None and info[0] != "image":
            if clip.source_in < -SRC_TOL:
                raise TimelineError(f"“{name}” would start before the beginning of its source media.")
            if clip.source_out <= clip.source_in + 1e-9:
                raise TimelineError(f"The source out point of “{name}” must be after its source in point.")
            if abs((clip.source_out - clip.source_in) - clip.duration * clip.speed) > SRC_TOL + 1e-6 * clip.duration:
                raise TimelineError(f"The source range of “{name}” does not match its duration and speed.")
            if info[1] is not None and clip.source_out > info[1] + SRC_TOL:
                raise TimelineError(f"“{name}” would run past the end of its source media ({info[1]:.2f} s available).")
    if old is None or [k.__dict__ for k in old.keyframes] != [k.__dict__ for k in clip.keyframes] or abs(old.duration - clip.duration) > TIME_EPSILON:
        for k in clip.keyframes:
            if not (math.isfinite(k.time) and math.isfinite(k.value)):
                raise TimelineError(f"“{name}” has a keyframe with an invalid value.")
            if k.time < -TIME_EPSILON or k.time > clip.duration + TIME_EPSILON:
                raise TimelineError(f"A {k.property} keyframe of “{name}” (at {k.time:.2f} s) would fall outside the clip.")


def validate_plan(env: PlanEnv, plan: EditPlan) -> dict[str, Clip]:
    """Check the whole plan against the current timeline *before* anything is changed. Returns the before-states of updated / removed clips.

    Raises ``TimelineError`` with a plain-language reason for: a vanished clip, a locked track / clip / aspect, a track that cannot hold the media, bad numbers or source
    ranges, keyframes outside a clip, and any overlap that would result on a touched track."""
    tl = env.timeline
    before: dict[str, Clip] = {}
    if set(plan.updates) & set(plan.removes):
        raise TimelineError("The edit both changes and deletes the same clip.")
    for cid in [*plan.updates, *plan.removes]:
        before[cid] = tl.find_clip(cid)[1]  # raises "That clip no longer exists."
    existing = {c.id for c in tl.all_clips()}
    for c in plan.adds:
        if c.id in existing or c.id in plan.updates:
            raise TimelineError("A new clip would reuse an existing clip id.", details=c.id)
    if len({c.id for c in plan.adds}) != len(plan.adds):
        raise TimelineError("The edit adds the same clip twice.")

    def track_of(tid: str) -> Track:
        return tl.get_track(tid)

    for cid, new in plan.updates.items():
        old = before[cid]
        if new.id != cid:
            raise TimelineError("An edit may not change a clip's id.")
        name = env.label(old)
        for tid in {old.track_id, new.track_id}:
            tr = track_of(tid)
            if tr.locked:
                raise TimelineError(f"Track “{tr.name}” is locked, so “{name}” cannot be edited. Unlock the track first.")
        aspects = changed_aspects(old, new)
        if aspects and old.locks:
            hit = ({"item"} & old.locks) or (aspects & old.locks)
            if hit:
                raise TimelineError(f"“{name}” is locked ({', '.join(sorted(old.locks))}); unlock it to {_verb(aspects & old.locks or aspects)}.")
        if new.track_id != old.track_id:
            info = env.asset_info(new.asset_id) if new.kind == KIND_MEDIA else None
            if not clip_fits_track(new, track_of(new.track_id), info[0] if info else None):
                raise TimelineError(f"Track “{track_of(new.track_id).name}” cannot hold “{name}”.")
        check_clip_numbers(env, new, old)
    for cid in plan.removes:
        old = before[cid]
        tr = track_of(old.track_id)
        if tr.locked:
            raise TimelineError(f"Track “{tr.name}” is locked, so “{env.label(old)}” cannot be removed. Unlock the track first.")
        if old.locks:
            raise TimelineError(f"“{env.label(old)}” is locked ({', '.join(sorted(old.locks))}); unlock it to remove it.")
    for c in plan.adds:
        tr = track_of(c.track_id)
        if tr.locked:
            raise TimelineError(f"Track “{tr.name}” is locked. Unlock it to add clips.")
        info = env.asset_info(c.asset_id) if c.kind == KIND_MEDIA else None
        if c.kind == KIND_MEDIA and info is None:
            raise TimelineError("The media for a new clip is missing from the project.")
        if not clip_fits_track(c, tr, info[0] if info else None):
            raise TimelineError(f"Track “{tr.name}” cannot hold “{env.label(c)}”.")
        check_clip_numbers(env, c, None)
    _check_overlaps(env, plan, before)
    return before


def _check_overlaps(env: PlanEnv, plan: EditPlan, before: dict[str, Clip]) -> None:
    tl = env.timeline
    gone = set(plan.updates) | set(plan.removes)
    placed = [*plan.updates.values(), *plan.adds]
    per_track: dict[str, list[Clip]] = {}
    for c in placed:
        per_track.setdefault(c.track_id, []).append(c)
    # tracks that lose a clip need no overlap check (removing cannot create an overlap)
    for tid, mine in per_track.items():
        lo = min(c.timeline_start for c in mine) - TIME_EPSILON
        hi = max(c.timeline_end for c in mine) + TIME_EPSILON
        others = [c for c in tl.clips_in_range(tid, lo, hi) if c.id not in gone]
        both = sorted([*others, *mine], key=lambda c: (c.timeline_start, c.timeline_end))
        for a, b in zip(both, both[1:]):
            if b.timeline_start < a.timeline_end - TIME_EPSILON:
                mover = b if (b.id in plan.updates or b in plan.adds) else a
                other = a if mover is b else b
                raise TimelineError(f"“{env.label(mover)}” would overlap “{env.label(other)}” on track “{tl.get_track(tid).name}”.")


# ------------------------------------------------------------------------------------------------------------------------ apply
@dataclass
class PlanRecord:
    """Everything needed to undo (and redo) a plan: the states before it, and the ids it added."""

    before_updates: dict[str, Clip]
    before_removes: dict[str, Clip]
    added_ids: list[str]
    final_updates: dict[str, Clip]
    added: list[Clip]
    removed_ids: list[str]


def _rebuild(tl: Timeline, remove_ids: set[str], put: list[Clip], hint_tracks: Iterable[str]) -> None:
    """Remove ``remove_ids`` and place ``put`` (copies) — one sorted rebuild per touched track, all-or-nothing."""
    touched = set(hint_tracks) | {c.track_id for c in put}
    saved: dict[str, list[Clip]] = {}
    try:
        by_track: dict[str, list[Clip]] = {}
        for c in put:
            by_track.setdefault(c.track_id, []).append(c.snapshot())
        for tid in touched:
            t = tl.get_track(tid)
            saved[tid] = list(t.clips)
            keep = [c for c in t.clips if c.id not in remove_ids]
            keep += by_track.get(tid, [])
            keep.sort(key=lambda c: (c.timeline_start, c.id))
            t.clips = keep
        tl.touch()
    except Exception:
        for tid, lst in saved.items():
            tl.get_track(tid).clips = lst
        tl.touch()
        raise


def apply_plan(tl: Timeline, plan: EditPlan, before: dict[str, Clip]) -> PlanRecord:
    b_up = {cid: before[cid].snapshot() for cid in plan.updates}
    b_rm = {cid: before[cid].snapshot() for cid in plan.removes}
    final = {cid: c.snapshot() for cid, c in plan.updates.items()}
    added = [c.snapshot() for c in plan.adds]
    hint = {c.track_id for c in b_up.values()} | {c.track_id for c in b_rm.values()}
    _rebuild(tl, set(plan.updates) | set(plan.removes), [*final.values(), *added], hint)
    return PlanRecord(b_up, b_rm, [c.id for c in added], final, added, list(plan.removes))


def redo_plan(tl: Timeline, rec: PlanRecord) -> None:
    hint = {c.track_id for c in rec.before_updates.values()} | {c.track_id for c in rec.before_removes.values()}
    _rebuild(tl, set(rec.before_updates) | set(rec.removed_ids), [*rec.final_updates.values(), *rec.added], hint)


def undo_plan(tl: Timeline, rec: PlanRecord) -> None:
    hint = {c.track_id for c in rec.final_updates.values()} | {c.track_id for c in rec.added}
    _rebuild(tl, set(rec.before_updates) | set(rec.added_ids), [*rec.before_updates.values(), *rec.before_removes.values()], hint)


class PlanCommand(Command):
    """One undoable transaction for any ``EditPlan``."""

    scope = "timeline"

    def __init__(self, env: PlanEnv, plan: EditPlan) -> None:
        self.env, self.plan = env, plan
        self.description = plan.description
        self.major = plan.major
        self._rec: PlanRecord | None = None

    @property
    def timeline(self) -> Timeline:
        return self.env.timeline

    def affected_clip_ids(self) -> set[str]:
        return self.plan.affected_ids()

    def states(self) -> tuple[list[Clip], list[Clip]]:
        """(before-states, after-states) of the changed clips — for caches / QC / the change tracker (valid after ``do``)."""
        r = self._rec
        if r is None:
            return [], []
        return [*r.before_updates.values(), *r.before_removes.values()], [*r.final_updates.values(), *r.added]

    def do(self) -> None:
        if self._rec is None:
            before = validate_plan(self.env, self.plan)
            self._rec = apply_plan(self.timeline, self.plan, before)
        else:
            redo_plan(self.timeline, self._rec)

    def undo(self) -> None:
        assert self._rec is not None
        undo_plan(self.timeline, self._rec)
