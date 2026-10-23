"""Vertical geometry of the timeline: the ruler plus one row per track, each with its own height (Phase 10: resizable and collapsible tracks).

One ``TrackLayout`` is shared by the canvas (rows, hit testing, painting) and the track-header column, so they can never disagree about where a row is. Qt-free. The offsets
are cached and rebuilt only when the list of tracks, a height or a collapsed flag changes (``signature``), so a mouse move never recomputes them.
"""

from __future__ import annotations

from bisect import bisect_right

RULER_H = 28
DEFAULT_ROW_H = 36
COLLAPSED_ROW_H = 18
MIN_ROW_H, MAX_ROW_H = 18, 240  # the range the model accepts for Track.height (0 = default)


def row_height(track) -> int:
    """Pixels a track row takes: the thin strip when collapsed, its own height when set, otherwise the default."""
    if getattr(track, "collapsed", False):
        return COLLAPSED_ROW_H
    h = int(getattr(track, "height", 0) or 0)
    return max(MIN_ROW_H, min(MAX_ROW_H, h)) if h else DEFAULT_ROW_H


class TrackLayout:
    def __init__(self, ruler_h: int = RULER_H) -> None:
        self.ruler_h = ruler_h
        self._sig: tuple | None = None
        self._tops: list[int] = []  # y of each row's top edge
        self._heights: list[int] = []
        self._total = ruler_h + DEFAULT_ROW_H + 1

    # ------------------------------------------------------------ building
    @staticmethod
    def signature(tracks) -> tuple:
        return tuple((t.id, row_height(t)) for t in tracks)

    def update(self, tracks) -> bool:
        """Recompute the offsets if the tracks' ids / heights / collapsed flags changed. Returns True when they did."""
        sig = self.signature(tracks)
        if sig == self._sig:
            return False
        self._sig = sig
        y, tops, heights = self.ruler_h, [], []
        for _tid, h in sig:
            tops.append(y)
            heights.append(h)
            y += h
        self._tops, self._heights = tops, heights
        self._total = y + 1 if sig else self.ruler_h + DEFAULT_ROW_H + 1  # an empty timeline still shows one empty row's worth of space
        return True

    # ------------------------------------------------------------ queries
    @property
    def count(self) -> int:
        return len(self._tops)

    @property
    def total_height(self) -> int:
        return self._total

    def top(self, index: int) -> int:
        return self._tops[index]

    def height(self, index: int) -> int:
        return self._heights[index]

    def bottom(self, index: int) -> int:
        return self._tops[index] + self._heights[index]

    def index_at(self, y: float) -> int | None:
        """Row under the vertical pixel ``y`` (None on the ruler or below the last row)."""
        if y < self.ruler_h or not self._tops:
            return None
        i = bisect_right(self._tops, y) - 1
        return i if 0 <= i < len(self._tops) and y < self._tops[i] + self._heights[i] else None

    def nearest_index(self, y: float) -> int | None:
        """Row index for a drag that has left the rows: clamps above the first row / below the last one."""
        if not self._tops:
            return None
        if y < self._tops[0]:
            return 0
        return self.index_at(y) if self.index_at(y) is not None else len(self._tops) - 1

    def visible(self, y0: float, y1: float) -> range:
        """Row indexes whose rectangle touches the vertical span [y0, y1]."""
        if not self._tops or y1 < self.ruler_h:
            return range(0)
        first = max(0, bisect_right(self._tops, max(y0, self.ruler_h)) - 1)
        last = bisect_right(self._tops, y1) - 1
        return range(first, min(last, len(self._tops) - 1) + 1)
