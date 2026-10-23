"""Qt-free multi-clip selection for the timeline.

The selection is a set of stable clip ids kept in a stable order, plus a ``primary`` (the clip the inspector shows: the last one clicked) and an ``anchor`` (where a
shift-click range starts). It stores ids only, so zooming, scrolling and track reordering never touch it, and it survives undo / redo for as long as the ids still exist
(``prune`` drops the ones that do not; the workspace calls it after every project change).

Invariant: ``primary`` is None exactly when the selection is empty, otherwise it is one of ``ids``.

Modes (``select`` and every bulk selector): ``replace`` | ``toggle`` | ``add`` | ``remove``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from app.core.exceptions import TimelineError
from app.logging.logger import get_logger
from app.timeline.ops_common import relatives
from app.timeline.timeline import Timeline

MODES = ("replace", "toggle", "add", "remove")
_PRUNE_SCAN_ABOVE = 8  # up to this many ids are looked up one by one (indexed); above it one pass over the clips builds an id set
_log = get_logger(__name__)


def _norm(ids: Iterable[str] | str | None) -> list[str]:
    if ids is None:
        return []
    if isinstance(ids, str):
        ids = [ids]
    return list(dict.fromkeys(i for i in ids if isinstance(i, str) and i))


class SelectionModel:
    def __init__(self) -> None:
        self._order: dict[str, None] = {}  # insertion-ordered set
        self._primary: str | None = None
        self._anchor: str | None = None
        self._listeners: list[Callable[["SelectionModel"], None]] = []

    # ------------------------------------------------------------------ reading
    @property
    def ids(self) -> list[str]:
        """The selected clip ids in selection order (a copy)."""
        return list(self._order)

    @property
    def primary(self) -> str | None:
        return self._primary

    @property
    def anchor(self) -> str | None:
        return self._anchor

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, clip_id: object) -> bool:
        return clip_id in self._order

    def __iter__(self):
        return iter(list(self._order))

    def __bool__(self) -> bool:
        return bool(self._order)

    @property
    def is_multi(self) -> bool:
        return len(self._order) > 1

    def span(self, tl: Timeline) -> tuple[float, float] | None:
        """(earliest start, latest end) of the selected clips that still exist - the span a multi-clip drag snaps as one block."""
        clips = [c for c in (tl.get_clip(i) for i in self._order) if c is not None]
        if not clips:
            return None
        return min(c.timeline_start for c in clips), max(c.timeline_end for c in clips)

    # ------------------------------------------------------------------ listeners
    def subscribe(self, listener: Callable[["SelectionModel"], None]) -> Callable[[], None]:
        """``listener(model)`` runs after every change of ``ids`` or ``primary``. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _commit(self, order: dict[str, None], primary: str | None, anchor: str | None) -> bool:
        if primary not in order:
            primary = next(reversed(order), None)
        if anchor not in order:
            anchor = None
        changed = list(order) != list(self._order) or primary != self._primary
        self._order, self._primary, self._anchor = order, primary, anchor
        if changed:
            for fn in list(self._listeners):
                try:
                    fn(self)
                except Exception:  # noqa: BLE001  (a broken listener must not break the editor)
                    _log.exception("selection listener failed")
        return changed

    # ------------------------------------------------------------------ explicit selection
    def _combine(self, new: list[str], mode: str) -> tuple[dict[str, None], str | None]:
        """The selection after applying ``new`` in ``mode`` and the last id that was newly selected (None if nothing was)."""
        if mode not in MODES:
            raise ValueError(f"unknown selection mode: {mode!r}")
        if mode == "replace":
            return dict.fromkeys(new), (new[-1] if new else None)
        order = dict(self._order)
        last_added: str | None = None
        for i in new:
            if mode == "remove" or (mode == "toggle" and i in order):
                order.pop(i, None)
            else:
                if i not in order:
                    order[i] = None
                last_added = i
        return order, last_added

    def select(self, ids: Iterable[str] | str | None, mode: str = "replace", *, primary: str | None = None) -> bool:
        """Change the selection. Returns True when the ids or the primary changed. ``primary`` (when it is among the result) becomes the primary clip; otherwise the last id
        newly selected does, or - if nothing was added - the previous primary / the last remaining id."""
        new = _norm(ids)
        order, last = self._combine(new, mode)
        if primary is not None and primary in order and mode != "remove":
            p = primary
        elif last is not None:
            p = last
        else:
            p = self._primary
        return self._commit(order, p, p if (mode == "replace" or last is not None) else self._anchor)  # a plain click / add moves the shift-click anchor

    def select_none(self) -> bool:
        return self._commit({}, None, None)

    clear = select_none

    def select_clip(self, clip_id: str | None) -> bool:
        """Replace the selection by one clip (None clears)."""
        return self.select([clip_id] if clip_id else [], "replace")

    def click(self, tl: Timeline, clip_id: str, mode: str = "replace", *, linked: bool = True, grouped: bool = True) -> bool:
        """A click on a clip: its linked / grouped relatives are selected with it (switch ``linked`` / ``grouped`` off to click a single clip). ``toggle`` removes the whole
        family when the clicked clip is already selected, otherwise adds it. The clicked clip becomes the primary."""
        family = relatives(tl, [clip_id], linked=linked, grouped=grouped)
        if mode == "toggle":
            mode = "remove" if clip_id in self._order else "add"
        return self.select(family, mode, primary=clip_id)

    # ------------------------------------------------------------------ bulk selectors (marquee, track header, select all)
    def _bulk(self, ids: list[str], mode: str) -> bool:
        order, _last = self._combine(_norm(ids), mode)
        p = self._primary if self._primary in order else next(iter(order), None)  # a bulk selection keeps the old primary when it is still selected, else the first clip
        return self._commit(order, p, p)

    def select_all(self, tl: Timeline, mode: str = "replace") -> bool:
        return self._bulk([c.id for c in tl.all_clips()], mode)

    def select_track(self, tl: Timeline, track_id: str, mode: str = "replace") -> bool:
        track = tl.get_track(track_id)
        return self._bulk([c.id for c in sorted(track.clips, key=lambda c: (c.timeline_start, c.id))], mode)

    def _ids_in_range(self, tl: Timeline, t0: float, t1: float, track_ids: Iterable[str] | None) -> list[str]:
        if t0 > t1:
            t0, t1 = t1, t0
        if track_ids is None:
            return [c.id for c in tl.clips_in_range(None, t0, t1)]
        out: list[str] = []
        for tid in dict.fromkeys(track_ids):
            out += [c.id for c in tl.clips_in_range(tid, t0, t1)]
        return out

    def select_range(self, tl: Timeline, t0: float, t1: float, track_ids: Iterable[str] | None = None, mode: str = "replace") -> bool:
        """Every clip overlapping (t0, t1) on ``track_ids`` (all tracks when None)."""
        return self._bulk(self._ids_in_range(tl, t0, t1, track_ids), mode)

    def select_rect(self, tl: Timeline, t0: float, t1: float, first_track_idx: int, last_track_idx: int, mode: str = "replace") -> bool:
        """The marquee: clips overlapping the time span (t0, t1) on tracks ``first_track_idx..last_track_idx`` (inclusive, any order, clamped to the existing tracks)."""
        n = len(tl.tracks)
        if n == 0:
            return self._bulk([], mode)
        lo, hi = sorted((int(first_track_idx), int(last_track_idx)))
        lo, hi = max(0, lo), min(n - 1, hi)
        if lo > hi:
            return self._bulk([], mode)
        return self._bulk(self._ids_in_range(tl, t0, t1, [t.id for t in tl.tracks[lo:hi + 1]]), mode)

    def extend_to(self, tl: Timeline, clip_id: str, *, additive: bool = False) -> bool:
        """Shift-click: select every clip on the clicked clip's track between the anchor (last plain click) and it, in time order. Without a usable anchor (none, deleted, or
        on another track) it behaves like a plain click. The anchor stays where it was so repeated shift-clicks pivot around it; ``additive`` keeps the existing selection."""
        _track, clip = tl.find_clip(clip_id)
        anchor = tl.get_clip(self._anchor) if self._anchor else None
        if anchor is None or anchor.track_id != clip.track_id:
            return self.select([clip_id], "add" if additive else "replace", primary=clip_id)
        ordered = sorted(tl.get_track(clip.track_id).clips, key=lambda c: (c.timeline_start, c.id))
        pos = {c.id: i for i, c in enumerate(ordered)}
        a, b = sorted((pos[anchor.id], pos[clip.id]))
        span = [c.id for c in ordered[a:b + 1]]
        order = dict(self._order) if additive else {}
        order.update(dict.fromkeys(span))
        return self._commit(order, clip_id, self._anchor)

    def invert(self, tl: Timeline) -> bool:
        """Select exactly the clips that are not selected now."""
        return self._bulk([c.id for c in tl.all_clips() if c.id not in self._order], "replace")

    # ------------------------------------------------------------------ keeping it consistent
    def prune(self, tl: Timeline) -> bool:
        """Drop ids that no longer exist in ``tl`` (call after every project change). Returns True when something was dropped."""
        if not self._order:
            return False
        if len(self._order) <= _PRUNE_SCAN_ABOVE:
            keep = [i for i in self._order if tl.get_clip(i) is not None]
        else:
            alive = {c.id for c in tl.all_clips()}
            keep = [i for i in self._order if i in alive]
        if len(keep) == len(self._order):
            return False
        return self._commit(dict.fromkeys(keep), self._primary, self._anchor)

    def expand_relatives(self, tl: Timeline, linked_on: bool = True, grouped_on: bool = True) -> list[str]:
        """The selected ids followed by every clip linked (``linked_on``) or grouped (``grouped_on``) with one of them - what a move / trim / delete acts on. Does not change the
        selection. Ids that no longer exist are ignored."""
        alive = [i for i in self._order if tl.get_clip(i) is not None]
        if not alive:
            return []
        try:
            return relatives(tl, alive, linked=linked_on, grouped=grouped_on)
        except TimelineError:  # a clip vanished between the two lookups
            return alive

    def selected_clips(self, tl: Timeline) -> list[Any]:
        return [c for c in (tl.get_clip(i) for i in self._order) if c is not None]
