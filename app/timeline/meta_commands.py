"""Undoable commands for timeline *metadata*: markers, the marked range and the track list (order, display, duplicate, kind, safe delete).

Every command validates in ``do()`` (so ``CommandStack.execute`` records nothing when it is refused), keeps exactly what it needs to undo, and uses the raw
``Timeline`` mutators. Clip-set edits do not live here: they are planners + ``PlanCommand`` (``ops_*.py``).

Track flags, for the UI and the docs (one concept each, no hidden second meaning):
  * ``hidden`` = the track is excluded from the preview AND from the export (the compiler skips hidden tracks);
  * ``muted`` / ``solo`` = audio only (they never change what is drawn);
  * ``locked`` = no manual edit of the track or its clips; ``height`` / ``collapsed`` are display preferences stored with the track.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

from app.core.constants import TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import Clip
from app.timeline.markers import MARKER_KINDS, Marker, new_marker_id
from app.timeline.ops_common import new_group_id, new_link_id
from app.timeline.plan import PlanEnv, clip_fits_track
from app.timeline.timeline import Timeline, new_clip_id, new_track_id
from app.timeline.timeline_commands import _TimelineCommand
from app.timeline.track import Track, TrackKind

MAX_MARKERS = 5000
MAX_NAME_LEN = 200
MAX_NOTE_LEN = 4000
TRACK_HEIGHT_RANGE = (18, 240)  # pixels; 0 means "the default height"
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# ------------------------------------------------------------------------------------------------------------------------ markers
def _check_time(t: float, what: str = "A marker") -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise TimelineError(f"{what} needs a valid time.") from None
    if not math.isfinite(t):
        raise TimelineError(f"{what} needs a valid time.")
    if t < -TIME_EPSILON:
        raise TimelineError(f"{what} cannot be placed before 0:00.")
    return max(0.0, t)


def _check_marker_fields(name: str, kind: str, color: str, note: str) -> None:
    if kind not in MARKER_KINDS:
        raise TimelineError(f"Unknown marker type “{kind}”. Choose one of: {', '.join(MARKER_KINDS)}.")
    if len(name) > MAX_NAME_LEN:
        raise TimelineError(f"A marker name can have at most {MAX_NAME_LEN} characters.")
    if len(note) > MAX_NOTE_LEN:
        raise TimelineError(f"A marker note can have at most {MAX_NOTE_LEN} characters.")
    if color and not _COLOR_RE.match(color):
        raise TimelineError("A marker colour must look like #RRGGBB.")


def _get_marker(tl: Timeline, marker_id: str) -> Marker:
    m = tl.get_marker(marker_id)
    if m is None:
        raise TimelineError("That marker no longer exists.", details=marker_id)
    return m


class AddMarkerCommand(_TimelineCommand):
    description = "Add marker"

    def __init__(self, timeline: Timeline, time: float, name: str = "", kind: str = "generic", color: str = "", note: str = "", *, marker_id: str | None = None) -> None:
        super().__init__(timeline)
        self.time, self.name, self.kind, self.color, self.note = time, str(name).strip(), kind, str(color).strip(), str(note)
        self.marker_id = marker_id or new_marker_id()  # fixed now so redo recreates the very same marker
        self._marker: Marker | None = None

    def do(self) -> None:
        if self._marker is None:
            t = _check_time(self.time)
            _check_marker_fields(self.name, self.kind, self.color, self.note)
            if self.timeline.get_marker(self.marker_id) is not None:
                raise TimelineError("A marker with that id already exists.")
            if len(self.timeline.markers) >= MAX_MARKERS:
                raise TimelineError(f"A timeline can have at most {MAX_MARKERS:,} markers.")
            self._marker = Marker(self.marker_id, t, self.name, self.kind, self.color, self.note)
        self.timeline.insert_marker(replace(self._marker))

    def undo(self) -> None:
        self.timeline.remove_marker(self.marker_id)


class MoveMarkerCommand(_TimelineCommand):
    description = "Move marker"

    def __init__(self, timeline: Timeline, marker_id: str, new_time: float) -> None:
        super().__init__(timeline)
        self.marker_id, self.new_time = marker_id, new_time
        self._old: Marker | None = None
        self._new: Marker | None = None

    def do(self) -> None:
        if self._new is None:
            old = _get_marker(self.timeline, self.marker_id)
            self._old, self._new = replace(old), replace(old, time=_check_time(self.new_time))
        self.timeline.remove_marker(self.marker_id)
        self.timeline.insert_marker(replace(self._new))

    def undo(self) -> None:
        assert self._old is not None
        self.timeline.remove_marker(self.marker_id)
        self.timeline.insert_marker(replace(self._old))


class EditMarkerCommand(_TimelineCommand):
    """Change name / type / colour / note (``None`` = leave as is; ``color=""`` = the type's default colour)."""

    description = "Edit marker"

    def __init__(self, timeline: Timeline, marker_id: str, *, name: str | None = None, kind: str | None = None, color: str | None = None, note: str | None = None) -> None:
        super().__init__(timeline)
        self.marker_id = marker_id
        self.changes = {k: (str(v).strip() if k in ("name", "color") else str(v)) for k, v in (("name", name), ("kind", kind), ("color", color), ("note", note)) if v is not None}
        self._old: Marker | None = None
        self._new: Marker | None = None

    def do(self) -> None:
        if self._new is None:
            old = _get_marker(self.timeline, self.marker_id)
            new = replace(old, **self.changes)
            _check_marker_fields(new.name, new.kind, new.color, new.note)
            self._old, self._new = replace(old), new
        self.timeline.remove_marker(self.marker_id)
        self.timeline.insert_marker(replace(self._new))

    def undo(self) -> None:
        assert self._old is not None
        self.timeline.remove_marker(self.marker_id)
        self.timeline.insert_marker(replace(self._old))


class DeleteMarkerCommand(_TimelineCommand):
    description = "Delete marker"

    def __init__(self, timeline: Timeline, marker_id: str) -> None:
        super().__init__(timeline)
        self.marker_id = marker_id
        self._removed: Marker | None = None

    def do(self) -> None:
        _get_marker(self.timeline, self.marker_id)
        self._removed = self.timeline.remove_marker(self.marker_id)

    def undo(self) -> None:
        assert self._removed is not None
        self.timeline.insert_marker(self._removed)


# ------------------------------------------------------------------------------------------------------------------------ range
class SetRangeCommand(_TimelineCommand):
    """Mark the in / out range (seconds, ``end > start >= 0``). Snapping to frames is the caller's job."""

    description = "Set range"

    def __init__(self, timeline: Timeline, start: float, end: float) -> None:
        super().__init__(timeline)
        self.start, self.end = start, end
        self._old: tuple[float, float] | None = None

    def do(self) -> None:
        try:
            a, b = float(self.start), float(self.end)
        except (TypeError, ValueError):
            raise TimelineError("The range needs a valid start and end.") from None
        if not (math.isfinite(a) and math.isfinite(b)):
            raise TimelineError("The range needs a valid start and end.")
        if a < -TIME_EPSILON:
            raise TimelineError("The range cannot start before 0:00.")
        a = max(0.0, a)
        if b <= a + TIME_EPSILON:
            raise TimelineError("The end of the range must be after its start.")
        self._old = self.timeline.range
        self.timeline.set_range((a, b))

    def undo(self) -> None:
        self.timeline.set_range(self._old)


class ClearRangeCommand(_TimelineCommand):
    description = "Clear range"

    def __init__(self, timeline: Timeline) -> None:
        super().__init__(timeline)
        self._old: tuple[float, float] | None = None

    def do(self) -> None:
        self._old = self.timeline.range
        if self._old is not None:
            self.timeline.set_range(None)

    def undo(self) -> None:
        if self._old is not None:
            self.timeline.set_range(self._old)


# ------------------------------------------------------------------------------------------------------------------------ tracks
def _index_of(tl: Timeline, track: Track) -> int:
    return next(i for i, t in enumerate(tl.tracks) if t is track)  # by identity: dataclass equality would compare every clip


class MoveTrackCommand(_TimelineCommand):
    """Reorder one track. ``new_index`` is its position in the list afterwards (0 = top). Any order is allowed; every track and clip stays."""

    description = "Move track"

    def __init__(self, timeline: Timeline, track_id: str, new_index: int) -> None:
        super().__init__(timeline)
        self.track_id, self.new_index = track_id, new_index
        self._old_index = -1

    def do(self) -> None:
        tl = self.timeline
        track = tl.get_track(self.track_id)
        if isinstance(self.new_index, bool) or not isinstance(self.new_index, int) or not (0 <= self.new_index < len(tl.tracks)):
            raise TimelineError("That is not a valid position for the track.", details=str(self.new_index))
        self._old_index = _index_of(tl, track)
        self._place(track, self.new_index)

    def undo(self) -> None:
        self._place(self.timeline.get_track(self.track_id), self._old_index)

    def _place(self, track: Track, index: int) -> None:
        tl = self.timeline
        if _index_of(tl, track) == index:
            return
        tl.tracks.pop(_index_of(tl, track))
        tl.tracks.insert(index, track)
        tl.touch()


class SetTrackDisplayCommand(_TimelineCommand):
    """Row height (0 = default, otherwise 18..240 px) and / or collapsed state. A display preference: not a major edit."""

    description = "Change track display"
    major = False

    def __init__(self, timeline: Timeline, track_id: str, height: int | None = None, collapsed: bool | None = None) -> None:
        super().__init__(timeline)
        self.track_id, self.height, self.collapsed = track_id, height, collapsed
        self._old: tuple[int, bool] | None = None

    def do(self) -> None:
        track = self.timeline.get_track(self.track_id)
        h = self.height
        if h is not None:
            if isinstance(h, bool) or not isinstance(h, (int, float)) or not math.isfinite(h) or int(h) != h or not (h == 0 or TRACK_HEIGHT_RANGE[0] <= h <= TRACK_HEIGHT_RANGE[1]):
                raise TimelineError(f"A track height must be 0 (default) or between {TRACK_HEIGHT_RANGE[0]} and {TRACK_HEIGHT_RANGE[1]} pixels.")
        self._old = (track.height, track.collapsed)
        if h is not None:
            track.height = int(h)
        if self.collapsed is not None:
            track.collapsed = bool(self.collapsed)

    def undo(self) -> None:
        assert self._old is not None
        track = self.timeline.get_track(self.track_id)
        track.height, track.collapsed = self._old


def make_duplicate_ids(timeline: Timeline, track_id: str) -> tuple[str, dict[str, str]]:
    """Fresh ids for ``DuplicateTrackCommand``: (new track id, {old clip id: new clip id})."""
    track = timeline.get_track(track_id)
    return new_track_id(), {c.id: new_clip_id() for c in track.clips}


class DuplicateTrackCommand(_TimelineCommand):
    """Copy a track (named "<name> copy") right below the original. Copied clips get the ids in ``id_map``, link / group ids are remapped so the copy never shares a link or
    group with the original (a link that would keep a single member is dropped), user locks are cleared and the copies belong to the USER (no AI decision owns them)."""

    description = "Duplicate track"

    def __init__(self, timeline: Timeline, track_id: str, new_track_id: str, id_map: dict[str, str] | None = None, include_clips: bool = True) -> None:
        super().__init__(timeline)
        self.track_id, self.new_track_id, self.include_clips = track_id, new_track_id, include_clips
        self.id_map = dict(id_map or {})
        self._built: Track | None = None
        self._index = 0

    def _build(self) -> Track:
        tl = self.timeline
        src = tl.get_track(self.track_id)
        if any(t.id == self.new_track_id for t in tl.tracks):
            raise TimelineError("A track with that id already exists.")
        new = Track(self.new_track_id, f"{src.name} copy", src.kind, hidden=src.hidden, muted=src.muted, locked=False, solo=False, volume=src.volume,
                    height=src.height, collapsed=src.collapsed)
        if not self.include_clips:
            return new
        taken = {c.id for c in tl.all_clips()}
        new_ids = []
        for c in src.clips:
            nid = self.id_map.get(c.id)
            if not nid:
                raise TimelineError("The track copy has no new id for one of its clips.", details=c.id)
            new_ids.append(nid)
        if len(set(new_ids)) != len(new_ids) or taken & set(new_ids):
            raise TimelineError("The copied clips would reuse existing clip ids.")
        links: dict[str, str] = {}
        groups: dict[str, str] = {}
        copies: list[Clip] = []
        for c, nid in zip(src.clips, new_ids):
            k = c.snapshot()
            k.id, k.track_id = nid, new.id
            k.locks, k.locked, k.created_by, k.ai_decision_id = frozenset(), False, "USER", ""
            for kf in k.keyframes:
                kf.decision_id = ""
            if k.transition:
                k.transition.pop("decision_id", None)
            if k.link_id:
                k.link_id = links.setdefault(k.link_id, new_link_id())
            if k.group_id:
                k.group_id = groups.setdefault(k.group_id, new_group_id())
            copies.append(k)
        for attr in ("link_id", "group_id"):  # a relationship with one member left means nothing
            counts: dict[str, int] = {}
            for k in copies:
                if getattr(k, attr):
                    counts[getattr(k, attr)] = counts.get(getattr(k, attr), 0) + 1
            for k in copies:
                if getattr(k, attr) and counts[getattr(k, attr)] < 2:
                    setattr(k, attr, "")
        new.clips = copies
        new.sort()
        return new

    def do(self) -> None:
        tl = self.timeline
        if self._built is None:
            self._built = self._build()
            self._index = _index_of(tl, tl.get_track(self.track_id)) + 1
        tl.insert_track(self._built, min(self._index, len(tl.tracks)))

    def undo(self) -> None:
        self.timeline.remove_track(self.new_track_id)


class TrackHasContent(TimelineError):
    """Deleting a track that still holds clips needs the user's confirmation. ``clip_count`` / ``track_name`` let the UI ask the question."""

    def __init__(self, user_message: str, *, track_name: str, clip_count: int) -> None:
        super().__init__(user_message)
        self.track_name, self.clip_count = track_name, clip_count


class SetTrackKindCommand(_TimelineCommand):
    """Change what a track is (video / image / graphics / text / captions / audio) — only when every clip on it is valid for the new kind.

    Media clips are checked with their asset type, so pass ``asset_info`` (the same callable ``PlanEnv`` uses: asset id -> (type, duration, name)); without it a track
    that holds media clips cannot be verified and the change is refused."""

    description = "Change track type"

    def __init__(self, timeline: Timeline, track_id: str, kind: TrackKind | str, asset_info=None) -> None:
        super().__init__(timeline)
        self.track_id, self.kind, self.asset_info = track_id, kind, asset_info
        self._old: TrackKind | None = None

    def do(self) -> None:
        tl = self.timeline
        track = tl.require_unlocked(self.track_id)
        try:
            kind = self.kind if isinstance(self.kind, TrackKind) else TrackKind(str(self.kind))
        except ValueError:
            raise TimelineError(f"Unknown track type “{self.kind}”.") from None
        if kind is track.kind:
            self._old = track.kind
            return
        probe = Track(track.id, track.name, kind)
        env = PlanEnv(tl, self.asset_info or (lambda _id: None))
        for c in track.clips:
            info = env.asset_info(c.asset_id) if c.kind == "media" else None
            if c.kind == "media" and info is None:
                raise TimelineError(f"“{env.label(c)}” cannot be checked against a {kind.value} track because its media is unknown.")
            if not clip_fits_track(c, probe, info[0] if info else None):
                raise TimelineError(f"“{env.label(c)}” cannot be on a {kind.value} track, so track “{track.name}” stays a {track.kind.value} track.")
        self._old = track.kind
        track.kind = kind
        tl.touch()

    def undo(self) -> None:
        assert self._old is not None
        track = self.timeline.get_track(self.track_id)
        if track.kind is not self._old:
            track.kind = self._old
            self.timeline.touch()


class RemoveTrackSafeCommand(_TimelineCommand):
    """Delete a track. Refuses (``TrackHasContent`` with the clip count) while the track holds clips unless ``confirm_content`` is True; a locked track and a track that holds a
    locked clip are always refused. Undo puts back the track, its clips and its position."""

    description = "Delete track"

    def __init__(self, timeline: Timeline, track_id: str, *, confirm_content: bool = False) -> None:
        super().__init__(timeline)
        self.track_id, self.confirm_content = track_id, confirm_content
        self._removed: tuple[Track, int] | None = None

    def do(self) -> None:
        tl = self.timeline
        track = tl.require_unlocked(self.track_id)
        n = len(track.clips)
        if n and not self.confirm_content:
            raise TrackHasContent(f"Track “{track.name}” still holds {n} clip{'s' if n != 1 else ''}. Deleting the track deletes them too.", track_name=track.name, clip_count=n)
        env = PlanEnv(tl)
        for c in track.clips:
            if c.locks:
                raise TimelineError(f"“{env.label(c)}” on track “{track.name}” is locked ({', '.join(sorted(c.locks))}); unlock it before deleting the track.")
        self._removed = tl.remove_track(self.track_id)

    def undo(self) -> None:
        assert self._removed is not None
        track, index = self._removed
        self.timeline.insert_track(track, index)
