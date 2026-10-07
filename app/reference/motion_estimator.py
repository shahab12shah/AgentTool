"""Global motion between two small grayscale frames: translation and zoom, estimated *jointly* (numpy only).

Model. Content at ``x`` in the earlier frame appears at ``c + s * (x - c) + t`` in the later one (``c`` = the frame centre, ``s`` = zoom factor,
``t`` = how far the centre content moved). The three parameters (tx, ty, ln s) plus a brightness offset are found by Gauss-Newton on the photometric
error (a Lucas-Kanade fit over the whole frame) on a Gaussian pyramid, with Huber weights so that a moving subject or a static caption does not drag
the estimate. It is sub-pixel accurate (a 1.2 % zoom moves the frame border by only 0.8 px) at about 2 ms per 128x72 pair.

Why the baseline was wrong (kept here so nobody re-introduces it): it searched the zoom FIRST, with no translation in the model, and then ran phase
correlation. Zoom about any point other than the frame centre is zoom + translation, and the translation leaks into the scale search (a plain 1 px shift
gave ``log_scale = -0.016``, the wrong sign), so a zoom anchored off-centre came out negative. Fitting scale and translation together removes that
coupling. Phase correlation survives only as an *initialiser* for large moves (fast pans), never as the answer.

Values are in *normalised units*: dx/dy are fractions of the frame width/height (positive = content moves right/down, i.e. the camera moves left/up),
``log_scale`` is the natural log of the zoom factor (positive = zooming in), ``confidence`` is 0..1: how well the fitted motion explains the second
frame. It is high for a static pair (excellent fit, motion ~0) and ~0 across a cut, on flat frames and on frames the model cannot align.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_K5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
MIN_STD = 0.008  # a frame flatter than this (0..1 luma units) carries no motion information
MAX_LOG_SCALE = float(np.log(1.4))  # per pair; a bigger zoom between two samples is not a camera move
ACCEPT_NCC = 0.95  # a zero-initialised fit that explains the later frame at least this well needs no further search
MIN_LEVEL_SIDE = 16  # the coarsest pyramid level is never smaller than this
_GRIDS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_WINDOWS: dict[tuple[int, int], np.ndarray] = {}


@dataclass
class MotionEstimate:
    """One pair's global motion. ``tx``/``ty`` are pixels of the frames passed in, ``log_scale`` is dimensionless, ``ncc`` is how well the fit explains the later frame."""

    tx: float = 0.0
    ty: float = 0.0
    log_scale: float = 0.0
    confidence: float = 0.0
    ncc: float = 0.0
    valid_share: float = 0.0


# ---------------------------------------------------------------------------------------------- pyramid
def _blur5(img: np.ndarray) -> np.ndarray:
    """Separable 1-4-6-4-1 Gaussian (edge-replicated): removes noise and widens the basin of the gradient fit."""
    h, w = img.shape
    p = np.empty((h, w + 4), dtype=np.float32)
    p[:, 2:w + 2] = img
    p[:, :2], p[:, w + 2:] = img[:, :1], img[:, -1:]
    t = _K5[0] * p[:, 0:w] + _K5[1] * p[:, 1:w + 1] + _K5[2] * p[:, 2:w + 2] + _K5[3] * p[:, 3:w + 3] + _K5[4] * p[:, 4:w + 4]
    q = np.empty((h + 4, w), dtype=np.float32)
    q[2:h + 2] = t
    q[:2], q[h + 2:] = t[:1], t[-1:]
    return _K5[0] * q[0:h] + _K5[1] * q[1:h + 1] + _K5[2] * q[2:h + 2] + _K5[3] * q[3:h + 3] + _K5[4] * q[4:h + 4]


def _pyramid(g: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [_blur5(g)]
    for _ in range(levels - 1):
        out.append(np.ascontiguousarray(_blur5(out[-1])[::2, ::2]))
    return out


def _levels_for(h: int, w: int) -> int:
    n = 1
    while min(h, w) // (2 ** n) >= MIN_LEVEL_SIDE and n < 4:
        n += 1
    return n


def _grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(x - cx, y - cy) for every pixel (flattened float32), in pixel units about the frame centre."""
    key = (h, w)
    if key not in _GRIDS:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRIDS[key] = ((xs - (w - 1) / 2.0).ravel(), (ys - (h - 1) / 2.0).ravel())
    return _GRIDS[key]


# ---------------------------------------------------------------------------------------------- the fit
def _sample(b: np.ndarray, p: np.ndarray, margin: float, want_grad: bool):
    """``b`` (flattened) bilinearly sampled where the model sends every pixel of the earlier frame, the mask of samples inside ``b`` (``margin`` px from its
    border) and, if asked, the exact derivative of the bilinear interpolant (gx, gy)."""
    h, w = b.shape
    xc, yc = _grids(h, w)
    s = np.float32(np.exp(p[2]))
    xs = np.float32((w - 1) / 2.0 + p[0]) + s * xc
    ys = np.float32((h - 1) / 2.0 + p[1]) + s * yc
    valid = (xs >= margin) & (xs <= w - 1 - margin) & (ys >= margin) & (ys <= h - 1 - margin)
    x0 = np.clip(xs, 0, w - 2).astype(np.intp)
    y0 = np.clip(ys, 0, h - 2).astype(np.intp)
    fx, fy = xs - x0, ys - y0
    flat = b.ravel()
    idx = y0 * w + x0
    v00, v01, v10, v11 = flat[idx], flat[idx + 1], flat[idx + w], flat[idx + w + 1]
    dx, dy = v01 - v00, v10 - v00
    bw = v00 + fx * dx + fy * dy + (fx * fy) * (v11 - v01 - dy)
    if not want_grad:
        return bw, valid, None, None
    return bw, valid, dx + fy * (v11 - v10 - dx), dy + fx * (v11 - v01 - dy)


def _fit_level(a: np.ndarray, b: np.ndarray, p: np.ndarray, iters: int) -> tuple[np.ndarray, bool]:
    """Gauss-Newton on ``a(x) - [b(c + s (x - c) + t) + off]`` over one pyramid level. ``p`` = (tx, ty, ln s, offset) in this level's pixels.

    The zoom parameter is solved as ``u = ln s * R`` (R = half the frame width) so that all three motion parameters are in pixel units and the normal
    matrix is well conditioned. Returns (params, usable): ``usable`` is False when too little of the frame is covered, or the system cannot be solved."""
    h, w = a.shape
    xc, yc = _grids(h, w)
    big_r = w / 2.0
    p = p.astype(np.float64).copy()
    af = a.ravel()
    sigma = 0.1
    for _ in range(iters):
        bw, valid, gx, gy = _sample(b, p, 1.0, True)
        vm = valid.astype(np.float32)
        cover = float(vm.mean())
        if cover < 0.3:
            return p, False
        r = af - bw - np.float32(p[3])
        ar = np.abs(r)
        sigma = max(1.25 * float((ar * vm).sum()) / (cover * r.size), 0.03)  # robust-ish noise scale (normalised units)
        wt = vm * np.minimum(1.0, 1.7 * sigma / np.maximum(ar, 1e-6))
        s = float(np.exp(p[2]))
        j = np.empty((4, r.size), dtype=np.float32)
        j[0], j[1], j[3] = gx, gy, 1.0
        j[2] = (gx * xc + gy * yc) * np.float32(s / big_r)
        jw = j * wt
        normal = (jw @ j.T).astype(np.float64)
        rhs = (jw @ r).astype(np.float64)
        normal[np.diag_indices(4)] += 1e-4 * float(np.trace(normal)) + 1e-9  # damping: an edge-only scene cannot fix the motion along the edge
        try:
            d = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return p, False
        lim = 0.35 * min(h, w)
        d[0], d[1] = min(max(d[0], -lim), lim), min(max(d[1], -lim), lim)
        d[2] = min(max(d[2], -0.15 * big_r), 0.15 * big_r)
        p[0] += d[0]
        p[1] += d[1]
        p[2] = min(max(p[2] + d[2] / big_r, -MAX_LOG_SCALE), MAX_LOG_SCALE)
        p[3] += d[3]
        if abs(d[0]) < 0.004 and abs(d[1]) < 0.004 and abs(d[2]) < 0.004:
            break
    return p, True


def _ncc_masked(a: np.ndarray, bw: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 16:
        return 0.0
    x, y = a.ravel()[mask], bw[mask]
    x, y = x - x.mean(), y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-9 else 0.0


def _register(pa: list[np.ndarray], pb: list[np.ndarray], p_top: np.ndarray, start_level: int, iters: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    """Coarse-to-fine fit starting at pyramid ``start_level`` with ``p_top`` (pixel units of that level). ``iters`` is indexed by level."""
    p = p_top.copy()
    usable = True
    for lv in range(start_level, -1, -1):
        p, ok = _fit_level(pa[lv], pb[lv], p, iters[lv])
        usable = usable and ok
        if lv > 0:
            p[0] *= 2.0
            p[1] *= 2.0
    return p, usable


def _hann(h: int, w: int) -> np.ndarray:
    if (h, w) not in _WINDOWS:
        _WINDOWS[(h, w)] = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return _WINDOWS[(h, w)]


def _shift_candidates(a: np.ndarray, b: np.ndarray, count: int = 3) -> list[tuple[float, float]]:
    """Likely whole-pixel translations of content from ``a`` to ``b`` (phase correlation peaks): initialisers for big moves, nothing more."""
    h, w = a.shape
    win = _hann(h, w)
    fa, fb = np.fft.rfft2((a - a.mean()) * win), np.fft.rfft2((b - b.mean()) * win)
    cross = fb * np.conj(fa)
    mag = np.abs(cross)
    cross = cross / (mag + 0.1 * float(mag.mean()) + 1e-9)  # soft whitening: sharpens the peak without amplifying noise
    r = np.fft.irfft2(cross, s=(h, w))
    out: list[tuple[float, float]] = []
    work = r.copy()
    for _ in range(count):
        iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
        dy = float(iy if iy <= h // 2 else iy - h)
        dx = float(ix if ix <= w // 2 else ix - w)
        out.append((dx, dy))
        work[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -1e9  # suppress the neighbourhood so the next peak is a different one
    return out


# ---------------------------------------------------------------------------------------------- public
def estimate_motion(prev: np.ndarray, cur: np.ndarray) -> MotionEstimate:
    """Global motion from ``prev`` to ``cur`` (float32 gray frames of the same size, values 0..1)."""
    if prev.shape != cur.shape or prev.ndim != 2 or min(prev.shape) < 16:
        return MotionEstimate()
    sa, sb = float(prev.std()), float(cur.std())
    if sa < MIN_STD or sb < MIN_STD:
        return MotionEstimate()  # flat frames carry no motion information
    h, w = prev.shape
    a = ((prev - prev.mean()) / sa).astype(np.float32)
    b = ((cur - cur.mean()) / sb).astype(np.float32)
    levels = _levels_for(h, w)
    pa, pb = _pyramid(a, levels), _pyramid(b, levels)
    iters = tuple(2 if lv == 0 else 4 for lv in range(levels))
    top = levels - 1
    best_p, best_ncc, best_ok = np.zeros(4), -2.0, False

    def consider(p: np.ndarray, ok: bool) -> None:
        nonlocal best_p, best_ncc, best_ok
        bw, valid, _, _ = _sample(pb[0], p, 2.0, False)
        c = _ncc_masked(pa[0], bw, valid) if ok else -1.0
        if c > best_ncc:
            best_p, best_ncc, best_ok = p, c, ok

    p0, ok0 = _register(pa, pb, np.zeros(4), top, iters)
    consider(p0, ok0)
    if best_ncc < ACCEPT_NCC and levels >= 2:
        lv = 1  # large moves (fast pans): start from the phase-correlation peaks measured on the half-size frames
        for dx, dy in _shift_candidates(pa[lv], pb[lv], 2):
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            p, ok = _register(pa, pb, np.array([dx, dy, 0.0, 0.0]), lv, iters)
            consider(p, ok)
    _, valid, _, _ = _sample(pb[0], best_p, 2.0, False)
    share = float(valid.mean())
    ncc = max(0.0, best_ncc)
    conf = float(np.clip((ncc - 0.5) / 0.4, 0.0, 1.0))
    conf *= float(np.clip((share - 0.3) / 0.4, 0.0, 1.0))  # a fit that only sees a sliver of the frame proves little
    conf *= float(np.clip(min(sa, sb) / 0.03, 0.0, 1.0))  # a nearly flat frame is a weak witness even when normalised
    if not best_ok or abs(best_p[2]) >= MAX_LOG_SCALE - 1e-6 or max(abs(best_p[0]) / w, abs(best_p[1]) / h) > 0.45:
        conf *= 0.25  # the fit ran into its limits: not trustworthy
    return MotionEstimate(float(best_p[0]), float(best_p[1]), float(best_p[2]), conf, ncc, share)


def estimate_global_motion(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float, float]:
    """Return (dx, dy, log_scale, confidence) from ``prev`` to ``cur`` (float32 gray frames of the same size, values 0..1).

    dx/dy are fractions of the frame width/height that the content moved; log_scale is ln of the zoom factor (> 0 = zoom in)."""
    e = estimate_motion(prev, cur)
    h, w = prev.shape
    return e.tx / w, e.ty / h, e.log_scale, e.confidence
