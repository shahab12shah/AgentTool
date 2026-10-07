"""Per-frame text-line finder for the caption/text analyzer: where in ONE frame is there a dense cluster of thin, high-contrast strokes laid out like a line
of text? (numpy only, no OCR: nothing here ever reads a word.)

What a frame is reduced to (``FrameObs``): coarse brightness / stroke-energy grids (a few KB, used later to tell whether the background changed under a
region) and a short list of ``LineObs`` -- geometry plus abstract measurements of each text-like line (height, width, edge density, a 1-D column profile,
box / colour-contrast flags). Pixels, crops and glyph shapes are never stored; the column profile is a 1-D count of edge pixels per image column (a projection, from which no glyph can be rebuilt), used only to
ask "is this the same line as one frame earlier?" and it lives only for the duration of the analysis.

How a line is found
* **edges**: colour-gradient magnitude (largest channel difference between the pixels two apart, so it also sees red-on-green). The threshold adapts to the
  frame: ``T = 4 x median`` clamped to 0.12..0.28, so a busy picture (high median gradient) must have *stronger* edges before it counts, while a clean picture
  keeps a floor that is far above sensor noise and compression ringing.
* **blocks**: the edge map is averaged into ~6 px blocks; blocks with a text-like edge density (and both horizontal and vertical stroke edges) are closed
  horizontally (words) and vertically (lines) and grouped into connected components.
* **lines**: inside each component a row profile finds the text bands, a column profile the horizontal extent; a band is one line.
* **text-likeness** (0..1) of a line is a soft AND of: height in a plausible range, aspect ratio, edge density (not blank, not solid noise), a balance of
  horizontal and vertical stroke edges, and how *sharply* the band stands out from the rows just above and below it (a text line is a band; texture is not).

This is a heuristic with real limits (see ``caption_analyzer`` module doc): it cannot tell text from a repeating texture by shape alone, which is why the
analyzer only believes a region that behaves like an overlay over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EDGE_FLOOR, EDGE_CEIL, EDGE_GAIN, CLEAN_P75 = 0.12, 0.55, 3.0, 0.06
MAX_LINE_WIDTH = 0.95  # a "line" spanning the whole frame width is a texture band, not a caption
BLOCK_ROWS = 45  # the block grid has about this many rows (6 px blocks at 270 p)
BLOCK_EDGE_MIN = 0.10  # edge density of a text-like block
BLOCK_EDGE_MAX = 1.01
MIN_LINE_PX = 7  # a line shorter than this (at any resolution) cannot be told from noise
MAX_LINES_PER_FRAME = 10
COARSE = 3  # large text is searched again on a frame shrunk by this factor, where its glyphs are the size of ordinary text
COARSE_MIN_H = 180  # frames shorter than this are not shrunk (the coarse frame would be too small to hold text)
STEM_MIN = 0.55  # stem continuity (see ``_stem_continuity``) from which a coarse band is one row of large glyphs
BIG_LINE = 0.10  # a coarse line is "large text" from this height (fraction of the frame height)


# ---------------------------------------------------------------------------------------------- observations
@dataclass
class LineObs:
    """One text-like line in one frame, as measurements only (pixel coordinates; ``x1``/``y1`` exclusive)."""

    x0: int
    y0: int
    x1: int
    y1: int
    text_score: float = 0.0  # 0..1 how text-like the shape / density is
    density: float = 0.0  # share of the box covered by strong edges
    contrast: float = 0.0  # mean gradient magnitude of the edge pixels (0..1): falls while a fade-in is under way
    energy: float = 0.0  # edge pixels x contrast: the line's total stroke energy
    profile: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint16), repr=False)  # edge pixels per column x0..x1 (transient, see module doc)
    box: float = 0.0  # 0..1: a filled rectangle surrounds the line
    emphasis: float = 0.0  # 0..1: share of the line's fill pixels in a clearly different colour
    colour_known: bool = False  # the line was large/bright enough for its colour to be measured at all

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)


@dataclass
class FrameObs:
    """Everything one frame contributes to the analysis. No pixels."""

    lines: list[LineObs]
    luma: np.ndarray  # (gh, gw) uint8 block-mean brightness
    energy: np.ndarray  # (gh, gw) uint8 block-mean stroke energy
    edge_density: float = 0.0  # share of pixels with a strong edge
    busy: float = 0.0  # share of blocks that look like texture but are not part of a detected line (0..1)
    flat: bool = False  # the frame is (almost) one colour: nothing to find


# ---------------------------------------------------------------------------------------------- small helpers
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True in a 1-D bool array."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _close_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in runs:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _dilate(mask: np.ndarray, ry: int, rx: int) -> np.ndarray:
    out = mask.copy()
    for k in range(1, rx + 1):
        out[:, k:] |= mask[:, :-k]
        out[:, :-k] |= mask[:, k:]
    mid = out.copy()
    for k in range(1, ry + 1):
        out[k:, :] |= mid[:-k, :]
        out[:-k, :] |= mid[k:, :]
    return out


def _fill_mask(sub: np.ndarray, k: int, t: float = 0.22) -> np.ndarray:
    """Bright pixels enclosed on both sides (horizontally or vertically, within ``k`` px) by clearly darker pixels: the inside of an outlined / dark-backed stroke."""
    y = sub.max(axis=2)  # the brightest channel, not luma: saturated red or blue text is dark in luma but as bright as white text against its outline
    big = np.float32(2.0)

    def darker_both(axis: int) -> np.ndarray:
        n = y.shape[axis]
        lo = np.full(y.shape, big, dtype=np.float32)  # min of the k pixels before
        hi = np.full(y.shape, big, dtype=np.float32)  # min of the k pixels after
        for d in range(1, k + 1):
            if d >= n:
                break
            if axis == 1:
                lo[:, d:] = np.minimum(lo[:, d:], y[:, :-d])
                hi[:, :-d] = np.minimum(hi[:, :-d], y[:, d:])
            else:
                lo[d:, :] = np.minimum(lo[d:, :], y[:-d, :])
                hi[:-d, :] = np.minimum(hi[:-d, :], y[d:, :])
        return (y - lo > t) & (y - hi > t)

    return (darker_both(1) | darker_both(0)) & (sub.max(axis=2) >= 0.5)


def components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """8-connected components of a small bool grid as (y0, x0, y1, x1, area), found by union-find over row runs (no scipy)."""
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows: list[list[tuple[int, int, int]]] = []
    for y in range(mask.shape[0]):
        cur = []
        for a, b in _runs(mask[y]):
            rid = len(parent)
            parent.append(rid)
            cur.append((a, b, rid))
        if y > 0:
            for a, b, rid in cur:
                for pa, pb, pid in rows[-1]:
                    if pa <= b and pb >= a:  # runs are [a, b): overlap or diagonal touch
                        ra, rb = find(rid), find(pid)
                        if ra != rb:
                            parent[ra] = rb
        rows.append(cur)
    boxes: dict[int, list[int]] = {}
    for y, cur in enumerate(rows):
        for a, b, rid in cur:
            r = find(rid)
            bb = boxes.get(r)
            if bb is None:
                boxes[r] = [y, a, y + 1, b, b - a]
            else:
                bb[0], bb[1], bb[2], bb[3], bb[4] = min(bb[0], y), min(bb[1], a), max(bb[2], y + 1), max(bb[3], b), bb[4] + (b - a)
    return [tuple(v) for v in boxes.values()]  # type: ignore[misc]


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at ``lo`` and below, 1 at ``hi`` and above (``hi`` < ``lo`` ramps the other way)."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _band(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """A trapezoid: 0 below ``lo0``, 1 between ``lo1`` and ``hi1``, 0 above ``hi0``."""
    return min(_ramp(x, lo0, lo1), _ramp(x, hi0, hi1))


def _stem_continuity(ex: np.ndarray) -> float:
    """How continuously vertical strokes run through a box, 0..1: the mean of the three emptiest interior rows' vertical-edge counts relative to the median row. The
    stems of ONE row of large glyphs run through every row (>= 0.75 measured on digits and capitals of many sizes); between two lines of ordinary text the rows of
    the gap carry (almost) no stem edges (<= 0.3)."""
    h = ex.shape[0]
    rp = ex.sum(axis=1).astype(np.float32)
    med = float(np.median(rp))
    lo, hi = int(0.15 * h), int(0.85 * h)
    if h < 12 or hi - lo < 3 or med <= 0:
        return 0.0
    return float(np.sort(rp[lo:hi])[:3].mean() / med)


def _inside(f: LineObs, box: tuple[int, int, int, int]) -> bool:
    """Is at least 60 % of the line's area inside the box (x0, y0, x1, y1)?"""
    ix = min(f.x1, box[2]) - max(f.x0, box[0])
    iy = min(f.y1, box[3]) - max(f.y0, box[1])
    return ix > 0 and iy > 0 and ix * iy >= 0.6 * max(1, f.w * f.h)


# ---------------------------------------------------------------------------------------------- the detector
class TextRegionDetector:
    """Finds text-like lines in frames of one fixed size. Stateless between frames (all temporal logic lives in the analyzer)."""

    def __init__(self, width: int, height: int, *, coarse: bool = False) -> None:
        self.width, self.height = int(width), int(height)
        self.bs = max(3, int(round(self.height / BLOCK_ROWS)))
        self.gh, self.gw = self.height // self.bs, self.width // self.bs
        self.scale = self.height / 270.0
        # Large text (a number card, a giant title) falls apart at full resolution: its thick strokes have an empty interior, so the row profile of ONE glyph has
        # several peaks and valleys that look like separate lines. Shrunk by COARSE it is ordinary text, so a second detector looks for it there.
        self._big = None if coarse or self.height < COARSE_MIN_H else TextRegionDetector(self.width // COARSE, self.height // COARSE, coarse=True)

    # ------------------------------------------------------------------ public
    def detect(self, frame: np.ndarray) -> FrameObs:
        rgb = self._rgb(frame)
        h, w = rgb.shape[:2]
        planes = [np.ascontiguousarray(rgb[..., c]) for c in range(3)]  # planar copies: reductions over a stride-3 axis are several times slower
        luma = 0.299 * planes[0] + 0.587 * planes[1] + 0.114 * planes[2]
        gx = np.zeros((h, w), dtype=np.float32)
        gy = np.zeros((h, w), dtype=np.float32)
        gx[:, 1:-1] = np.maximum(np.maximum(np.abs(planes[0][:, 2:] - planes[0][:, :-2]), np.abs(planes[1][:, 2:] - planes[1][:, :-2])), np.abs(planes[2][:, 2:] - planes[2][:, :-2]))
        gy[1:-1, :] = np.maximum(np.maximum(np.abs(planes[0][2:] - planes[0][:-2]), np.abs(planes[1][2:] - planes[1][:-2])), np.abs(planes[2][2:] - planes[2][:-2]))
        gm = np.maximum(gx, gy)
        p75 = float(np.percentile(gm[::2, ::2], 75))  # a clean picture has p75 ~ 0.03..0.06, foliage / noise / patterns 0.1..0.5; text barely moves it
        thr = float(np.clip(EDGE_FLOOR + EDGE_GAIN * max(0.0, p75 - CLEAN_P75), EDGE_FLOOR, EDGE_CEIL))
        texture = _ramp(p75, 0.07, 0.25)
        ex, ey = gx > thr, gy > thr
        e = ex | ey
        bs, gh, gw = self.bs, self.gh, self.gw
        crop = (slice(0, gh * bs), slice(0, gw * bs))

        def bmean(a: np.ndarray) -> np.ndarray:
            return a[crop].reshape(gh, bs, gw, bs).mean(axis=(1, 3))

        en = np.where(gm > 0.06, np.minimum(gm, 1.0), 0.0).astype(np.float32)
        obs = FrameObs([], np.clip(bmean(luma) * 255, 0, 255).astype(np.uint8), np.clip(bmean(en) * 255, 0, 255).astype(np.uint8), float(e.mean()), texture)
        if float(luma.std()) < 0.004:
            obs.flat = True
            obs.busy = 0.0
            return obs
        ne, nx, ny = bmean(e), bmean(ex), bmean(ey)
        textish = (ne >= BLOCK_EDGE_MIN) & (ne <= BLOCK_EDGE_MAX) & (nx >= 0.04) & (ny >= 0.02)
        closed = _dilate(textish, 1, 2)
        lines: list[LineObs] = []
        for y0, x0, y1, x1, _area in components(closed):
            if int(textish[y0:y1, x0:x1].sum()) < 4:  # a lone corner or speck of a hard edge is not a line of text
                continue
            lines += self._lines_in(e, ex, ey, gm, rgb, luma, x0 * bs, y0 * bs, min(w, x1 * bs), min(h, y1 * bs))
        if self._big is not None:
            lines = self._merge_scales(lines, self._big.detect(self._shrink(rgb)).lines, ex)
        lines.sort(key=lambda ln: -ln.text_score)
        obs.lines = lines[:MAX_LINES_PER_FRAME]
        covered = np.zeros((gh, gw), dtype=bool)
        for ln in obs.lines:
            if ln.text_score >= 0.4:
                covered[ln.y0 // bs:(ln.y1 + bs - 1) // bs, ln.x0 // bs:(ln.x1 + bs - 1) // bs] = True
        inside = sum(int(e[ln.y0:ln.y1, ln.x0:ln.x1].sum()) for ln in obs.lines if ln.text_score >= 0.4)
        outside_density = max(0.0, float(e.sum()) - inside) / e.size
        obs.busy = float(max(texture, (textish & ~covered).mean(), _ramp(outside_density, 0.06, 0.30)))
        return obs

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _shrink(rgb: np.ndarray) -> np.ndarray:
        """Block-average by COARSE (area filter, so strokes thinner than a block fade and thick ones stay)."""
        h, w = rgb.shape[0] // COARSE * COARSE, rgb.shape[1] // COARSE * COARSE
        return rgb[:h, :w].reshape(h // COARSE, COARSE, w // COARSE, COARSE, 3).mean(axis=(1, 3))

    def _merge_scales(self, fine: list[LineObs], big: list[LineObs], ex: np.ndarray) -> list[LineObs]:
        """Large text found at the coarse scale replaces the fragments the full-resolution pass made of it. A coarse line only counts if it is large (>= BIG_LINE of
        the frame height), clearly text-like, filled by full-resolution fragments (not a plate with margins around a smaller line) and built like ONE row of tall
        glyphs: vertical stroke edges (stems) run through all of its rows. Two lines of ordinary
        text that the shrunk frame merged into one band never have that (their stems stop in the gap between the lines). A fine line that already spans
        the coarse one (ordinary text) is kept as it is."""
        out = list(fine)
        for b in big:
            if b.text_score < 0.55 or b.h * COARSE < BIG_LINE * self.height:
                continue
            box = (b.x0 * COARSE, b.y0 * COARSE, b.x1 * COARSE, b.y1 * COARSE)
            bh, bw = box[3] - box[1], box[2] - box[0]
            inside = [f for f in out if _inside(f, box)]
            if not inside or any(f.h >= 0.8 * bh and f.w >= 0.6 * bw for f in inside):
                continue
            if max(f.y1 for f in inside) - min(f.y0 for f in inside) < 0.75 * bh:  # fragments fill their glyph row; text on a plate leaves the plate's margin around it
                continue
            if _stem_continuity(ex[box[1]:box[3], box[0]:box[2]]) < STEM_MIN:
                continue
            prof = np.repeat(b.profile, COARSE)[:bw]
            if prof.size < bw:
                prof = np.pad(prof, (0, bw - prof.size), mode="edge")
            big_line = LineObs(box[0], box[1], box[2], box[3], b.text_score, b.density, b.contrast, b.energy * COARSE * COARSE, prof.astype(np.uint16), b.box, b.emphasis, b.colour_known)
            out = [f for f in out if f not in inside]
            out.append(big_line)
        return out

    @staticmethod
    def _rgb(frame: np.ndarray) -> np.ndarray:
        a = np.asarray(frame)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0 if a.dtype.kind == "f" and a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
        return a.astype(np.float32) / 255.0

    def _lines_in(self, e, ex, ey, gm, rgb, luma, x0: int, y0: int, x1: int, y1: int) -> list[LineObs]:
        """Split one connected region into text lines (row profile -> bands, column profile -> extent) and score each."""
        H, W = e.shape
        # one block of context so a stroke cut by the block grid is still inside
        x0, y0, x1, y1 = max(0, x0 - 2), max(0, y0 - 2), min(W, x1 + 2), min(H, y1 + 2)
        sub = e[y0:y1, x0:x1]
        if sub.size == 0:
            return []
        rp = sub.sum(axis=1).astype(np.float32)
        if rp.max() < 3:
            return []
        rs = np.convolve(rp, np.ones(3) / 3.0, mode="same")
        on = rs >= max(2.0, 0.22 * float(rs.max()))
        out: list[LineObs] = []
        bands: list[tuple[int, int]] = []
        for ra, rb in _close_runs(_runs(on), max(1, int(round(2 * self.scale)))):
            parts = self._split_valleys(rs, ra, rb)
            if len(parts) == 1 and rb - ra >= 2 * 9 * self.scale:
                parts = self._split_by_fill(rgb[y0 + ra:y0 + rb, x0:x1], ra, rb)
            bands += parts
        for ra, rb in bands:
            if rb - ra < max(MIN_LINE_PX * self.scale * 0.8, 4):
                continue
            band = sub[ra:rb]
            cp = band.sum(axis=0)
            band_h = rb - ra
            cols = _runs(cp >= 1)
            # the vertical sides of a background box (a thin column of edges as tall as the band, clearly apart from the text) are not text
            while len(cols) > 1 and cols[0][1] - cols[0][0] <= 3 and cp[cols[0][0]:cols[0][1]].max() >= 0.9 * band_h and cols[1][0] - cols[0][1] >= max(3, 0.25 * band_h):
                cols = cols[1:]
            while len(cols) > 1 and cols[-1][1] - cols[-1][0] <= 3 and cp[cols[-1][0]:cols[-1][1]].max() >= 0.9 * band_h and cols[-1][0] - cols[-2][1] >= max(3, 0.25 * band_h):
                cols = cols[:-1]
            for ca, cb in _close_runs(cols, max(3, int(round(0.9 * band_h)))):
                ln = self._score(e, ex, ey, gm, rgb, luma, x0 + ca, y0 + ra, x0 + cb, y0 + rb)
                if ln is not None:
                    out.append(ln)
        return out

    def _split_valleys(self, rs: np.ndarray, ra: int, rb: int) -> list[tuple[int, int]]:
        """Two lines whose outlines touch form one band; the row profile still dips between them. Split at a dip that is clearly lower than a strong peak on BOTH
        sides (a single line only fades towards its ascenders / descenders: no peak beyond the dip)."""
        min_part = max(5, int(round(MIN_LINE_PX * self.scale * 0.8)))
        if rb - ra < 2 * min_part + 3:
            return [(ra, rb)]
        seg = rs[ra:rb]
        lo, hi = max(min_part, int(0.2 * len(seg))), min(len(seg) - min_part, int(0.8 * len(seg)) + 1)
        if hi <= lo:
            return [(ra, rb)]
        v = lo + int(np.argmin(seg[lo:hi]))
        left, right = float(seg[:max(1, v - 2)].max()), float(seg[v + 3:].max()) if v + 3 < len(seg) else 0.0
        if seg[v] * 1.8 <= min(left, right) and right > 0:
            return self._split_valleys(rs, ra, ra + v) + self._split_valleys(rs, ra + v + 1, rb)
        return [(ra, rb)]

    def _split_by_fill(self, crop: np.ndarray, ra: int, rb: int) -> list[tuple[int, int]]:
        """Outlined lines whose outlines touch have no empty row between them, but their bright fill does: a row (or a few) with practically no fill pixels
        between two rows of plenty. Measured on single lines of many sizes the dip inside a line never falls below ~0.65 of its lower peak, so a dip below 0.40
        is a gap between lines."""
        fp = _fill_mask(crop, max(3, int(round(4 * self.scale)))).sum(axis=1).astype(np.float32)
        fs = np.convolve(fp, np.ones(3) / 3.0, mode="same")
        min_part = max(6, int(round(7 * self.scale)))
        if len(fs) < 2 * min_part + 1:
            return [(ra, rb)]
        v = min_part + int(np.argmin(fs[min_part:len(fs) - min_part]))
        left, right = float(fs[:max(1, v - 1)].max()), float(fs[v + 2:].max()) if v + 2 < len(fs) else 0.0
        if min(left, right) >= 12.0 and fs[v] <= 0.40 * min(left, right):
            return self._split_by_fill(crop[:v], ra, ra + v) + self._split_by_fill(crop[v + 1:], ra + v + 1, rb)
        return [(ra, rb)]

    def _score(self, e, ex, ey, gm, rgb, luma, x0: int, y0: int, x1: int, y1: int) -> LineObs | None:
        H, W = e.shape
        h, w = y1 - y0, x1 - x0
        if w < 2 * MIN_LINE_PX * self.scale * 0.5 or h < MIN_LINE_PX * self.scale * 0.8:
            return None
        # tighten the rows: drop border rows that carry (almost) no edges
        eb = e[y0:y1, x0:x1]
        rp = eb.sum(axis=1)
        keep = np.flatnonzero(rp >= max(1.0, 0.18 * float(rp.max())))
        if keep.size == 0:
            return None
        y0, y1 = y0 + int(keep[0]), y0 + int(keep[-1]) + 1
        eb = e[y0:y1, x0:x1]
        cp = eb.sum(axis=0)
        kc = np.flatnonzero(cp >= 1)
        if kc.size == 0:
            return None
        x0, x1 = x0 + int(kc[0]), x0 + int(kc[-1]) + 1
        eb = e[y0:y1, x0:x1]
        h, w = y1 - y0, x1 - x0
        if h < MIN_LINE_PX * self.scale * 0.8:
            return None
        area = float(h * w)
        density = float(eb.mean())
        nx, ny = float(ex[y0:y1, x0:x1].mean()), float(ey[y0:y1, x0:x1].mean())
        rel_h, aspect = h / H, w / max(1.0, float(h))
        s_h = _band(rel_h, 0.018, 0.035, 0.28, 0.38)
        s_aspect = _ramp(aspect, 0.45, 1.3)
        s_dens = min(_ramp(density, 0.05, 0.12), _ramp(density, 0.99, 0.90))
        s_orient = _ramp(min(nx, ny) / max(nx, ny, 1e-6), 0.10, 0.30)
        # a text line is a band: the rows just above and below it are much emptier than its core
        pad = max(2, int(round(0.25 * h)))
        above = float(e[max(0, y0 - pad):y0, x0:x1].mean()) if y0 > 0 else 0.0
        below = float(e[y1:min(H, y1 + pad), x0:x1].mean()) if y1 < H else 0.0
        s_band = _ramp(density / max(1e-6, 0.5 * (above + below) + 1e-6), 1.4, 3.5) if (y0 > 0 or y1 < H) else 0.5
        gsub = gm[y0:y1, x0:x1]
        mask = eb
        contrast = float(gsub[mask].mean()) if mask.any() else 0.0
        s_con = _ramp(contrast, 0.15, 0.35)  # the hard edges of a picture's own shapes are far weaker than outlined / boxed text
        parts = (s_h, s_aspect, s_dens, s_orient, s_band, s_con)
        score = float(np.prod(parts) ** (1.0 / 5.0)) if min(parts) > 0 else 0.0
        if score < 0.15 or w > MAX_LINE_WIDTH * W:
            return None
        prof = ex[y0:y1, x0:x1].sum(axis=0).astype(np.uint16)  # vertical-edge (stem) columns: far more specific to the text than all edges together
        ln = LineObs(x0, y0, x1, y1, float(score), density, contrast, float(mask.sum()) * contrast, prof)
        if score >= 0.35:
            ln.box = self._box_score(luma, x0, y0, x1, y1)
            ln.emphasis, ln.colour_known = self._emphasis(rgb, x0, y0, x1, y1)
        _ = area
        return ln

    # ------------------------------------------------------------------ box around a line
    def _box_score(self, luma: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
        """0..1: is the line sitting on a filled rectangle? On each of the four sides the mean brightness profile (moving away from the text) is searched for a
        step; a box makes the same-signed step on every side (darker inside, or brighter inside) a few pixels out; texture and soft shadows do not."""
        H, W = luma.shape
        h = y1 - y0
        pmax = int(max(8, min(40, 1.6 * h + 8)))
        pmin = 3
        steps: list[np.ndarray] = []  # signed step (inside - outside) for p = pmin..pmax per side

        def prof(side: str) -> np.ndarray | None:
            d = np.arange(1, pmax + 4)
            if side == "top":
                ys = y0 - d
                ok = ys >= 0
                if ok.sum() < pmax or x1 - x0 < 4:
                    return None
                return luma[np.clip(ys, 0, H - 1), x0:x1].mean(axis=1)
            if side == "bottom":
                ys = y1 - 1 + d
                ok = ys < H
                if ok.sum() < pmax or x1 - x0 < 4:
                    return None
                return luma[np.clip(ys, 0, H - 1), x0:x1].mean(axis=1)
            if side == "left":
                xs = x0 - d
                ok = xs >= 0
                if ok.sum() < pmax:
                    return None
                return luma[y0:y1, np.clip(xs, 0, W - 1)].mean(axis=0)
            xs = x1 - 1 + d
            ok = xs < W
            if ok.sum() < pmax:
                return None
            return luma[y0:y1, np.clip(xs, 0, W - 1)].mean(axis=0)

        for side in ("top", "bottom", "left", "right"):
            p = prof(side)
            if p is None:
                continue
            # p[k] = brightness at distance k+1 from the text. step at edge position q (the box edge lies between distance q and q+1)
            ps = np.arange(pmin, pmax)
            inside = 0.5 * (p[ps - 2] + p[ps - 1])
            outside = 0.5 * (p[ps + 1] + p[ps + 2])
            steps.append(inside - outside)
        if len(steps) < 3:
            return 0.0
        best = 0.0
        for sign in (-1.0, 1.0):
            # an edge must be strong on every side AND roughly at the same distance (a box has even padding, give or take the text bbox's slack)
            per_side = np.stack([np.clip(sign * s, 0, None) for s in steps])
            side_best = per_side.max(axis=1)
            best = max(best, float(side_best.min()))
        return float(min(1.0, max(0.0, (best - 0.04) / 0.08)))  # a soft shading of the picture reaches ~0.05 on every side; a real plate over a mid-tone picture 0.15+

    # ------------------------------------------------------------------ colour emphasis
    def _emphasis(self, rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[float, bool]:
        """Share of the line's *fill* pixels whose colour (brightness-normalised, so antialiasing and shading drop out) differs clearly from the line's dominant
        colour AND forms a coherent horizontal run (a highlighted word), not scattered chroma noise. ``(share, measured)``.

        Fill pixels are the bright pixels that are enclosed on both sides (within about a fifth of the line height) by clearly darker pixels: the inside of an
        outlined or dark-backed stroke. That keeps the (possibly colourful) background out of the measurement. Dark text and very thin strokes have no such
        pixels: their colour is simply not measured (``measured`` False) and the line counts as not emphasised."""
        sub = rgb[y0:y1, x0:x1]
        h, w = sub.shape[:2]
        fill = _fill_mask(sub, int(max(3, min(8, round(0.2 * h)))))
        core = fill.copy()
        core[1:-1, 1:-1] = fill[1:-1, 1:-1] & fill[:-2, 1:-1] & fill[2:, 1:-1] & fill[1:-1, :-2] & fill[1:-1, 2:]
        core[0, :] = core[-1, :] = False
        core[:, 0] = core[:, -1] = False
        if int(core.sum()) < max(60, int(0.25 * fill.sum())):  # thin strokes have no interior: use the whole fill
            core = fill
        # bright background seen through the gaps between letters is enclosed by outlines too: drop pixels that look like the picture just above / below the line
        H = rgb.shape[0]
        refs = [rgb[max(0, y0 - 4):y0 - 1, x0:x1].mean(axis=0)] if y0 >= 5 else []
        if y1 + 4 <= H:
            refs.append(rgb[y1 + 1:y1 + 4, x0:x1].mean(axis=0))
        if refs:
            bgcol = np.mean(refs, axis=0)  # (w, 3)
            core &= np.sqrt(((sub - bgcol[None, :, :]) ** 2).sum(axis=2)) >= 0.18
        n = int(core.sum())
        if n < 30:
            return 0.0, False
        px = sub[core]
        u = px / np.maximum(px.max(axis=1, keepdims=True), 1e-3)
        q = np.minimum((u * 4).astype(np.int32), 3)
        key = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
        main = u[key == int(np.argmax(np.bincount(key, minlength=64)))].mean(axis=0)  # the dominant colour (a median fails when the highlight is about half the line)
        minority = np.sqrt(((u - main) ** 2).sum(axis=1)) > 0.38
        if float(minority.mean()) > 0.5:  # the dominant bin holds fewer pixels than the rest (heavier letters in the other colour): the smaller colour is the highlight
            minority = ~minority
        share = float(minority.mean())
        if share < 0.05 or share > 0.5 or not minority.any() or minority.all():
            return 0.0, True
        # a highlight is a vivid colour against the other one (or the other one is vivid and the highlight is not): pale tints are the picture showing through the letters
        if max(1.0 - float(u[~minority].mean(axis=0).min()), 1.0 - float(u[minority].mean(axis=0).min())) < 0.55:
            return 0.0, True
        mm = np.zeros(core.shape, dtype=bool)
        mm[core] = minority
        col_core = core.sum(axis=0)
        col_flag = (col_core > 0) & (mm.sum(axis=0) >= 0.6 * np.maximum(col_core, 1))
        best = max((b - a for a, b in _close_runs(_runs(col_flag), 2)), default=0)
        return (share if best >= max(4, int(0.09 * w)) else 0.0), True  # a highlighted word is at least ~9 % of its line
