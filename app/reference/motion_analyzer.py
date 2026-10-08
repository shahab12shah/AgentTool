"""MotionAnalyzer: how much and what kind of camera / content movement a reference video has (zoom, pan, shake), as abstract numbers.

Input is the per-sample motion signals of ``compute_signals`` (zoom rate, pan rates and a confidence from the joint global-motion fit, plus the
frame-to-frame difference) and the shots. Output is a ``MotionResult``: the ``MotionStats`` that enter the style profile (rates, shares, a class) and,
as internal cache data only, one ``ShotMotion`` per shot, a coarse intensity series and the zoom/pan events found. Nothing here is footage, a frame or
a text; the per-event timings stay in ``MotionResult.events`` and must never reach the profile or anything the user's video consumes.

How it measures
- The fit gives, at every sample, the motion between that frame and the one ``lag`` samples earlier. A window that starts before the shot did crosses
  a cut and has no meaning, so the first ``lag`` samples of a shot are skipped as sources (the rates are held over the head and tail instead).
  Windows with a low confidence, or that disagree with their neighbours (the aliasing of fast shake), are discarded.
- Zoom and pan are integrated over the shot (the zoom ratio is exp(sum of log-scale increments), the pan extent is the sum of translation increments
  as a fraction of the frame width). A translation that moves in proportion to the zoom, like the shift of an off-centre punch-in, belongs to the zoom
  and is not counted as a pan.
- Events inside a shot: a run of zoom rate above a floor that accumulates at least 3 % of scale is one zoom event (a 0.8 s punch-in and a 6 s slow
  zoom are one each); likewise a coherent run of translation that adds up to at least 5 % of the width is one pan event.
- Whatever the global model does not explain (a moving subject, handheld shake, chaos) is read from the frame difference after subtracting the noise
  floor and what the measured camera motion accounts for.

Intensity (0..1, per sample; a shot's intensity is the mean over the shot, the video's the duration-weighted mean over all shots)
    I = 1 - (1 - I_zoom)(1 - I_pan)(1 - I_residual)
    I_zoom  = sqrt((|zoom rate| - 0.004) / 0.27)       zoom rate in ln(scale) per second   : 8 % in 4 s -> 0.24, 15 % in 1 s -> 0.7 at its peak
    I_pan   = sqrt((speed - 0.004) / 0.69)             speed in frame widths per second     : 30 % in 4 s -> 0.32, 50 % in 2 s -> 0.6
    I_resid = piecewise-linear of the unexplained frame difference (per 8 fps sample)         : noise -> 0, handheld 0.05 -> 0.45, shake 0.1+ -> 0.8+
Calibration: a static shot is below 0.05, a gentle zoom 0.2-0.35, a fast pan or punch-in 0.5-0.7, shake or chaos above 0.8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from app.reference.style_model import MOTION_EVENT_POINTS, MotionStats, Shot, ZoomStats, clamp, motion_class, scale
from app.reference.signals import FrameSignals

KINDS = ("static", "zoom_in", "zoom_out", "pan", "high_motion", "mixed")
MIN_CONF = 0.5  # a motion window below this confidence is not used
ZOOM_RATE_FLOOR = 0.004  # ln(scale) per second: below this a window counts as "not zooming"
ZOOM_MIN_CHANGE = math.log(1.03)  # an event must add up to at least a 3 % change of scale
PAN_SPEED_FLOOR = 0.006  # frame widths per second
PAN_MIN_EXTENT = 0.05  # an event must move the content by at least 5 % of the width
MIN_EVENT_SECONDS = 0.4
MAX_FILL = 3  # a gap of this many samples (or fewer) between two good windows is bridged; a longer one is 'unknown', never interpolated into a plateau
MIN_MEASURED_SHARE = 0.7  # an event needs at least this share of its samples to come straight from good windows
HIGH_INTENSITY = 0.6  # per-sample intensity from which a sample counts as high motion
SUSTAINED_EVENT_SECONDS = 5.0  # a long high-motion stretch counts as one event per this many seconds
ZOOM_REF_RATE = 0.27
PAN_REF_SPEED = 0.69
PAN_DEAD_ZONE = 0.004  # frame widths per second of translation that is noise
RESIDUAL_POINTS = ((0.0, 0.0), (0.006, 0.02), (0.02, 0.12), (0.05, 0.45), (0.08, 0.75), (0.12, 0.9), (0.2, 1.0))  # unexplained frame difference (8 fps) -> 0..1
EXPLAIN_PER_PIXEL = 0.02  # frame difference that one pixel of camera travel per frame accounts for (at 128 px wide)
EXPLAIN_CAP = 0.05  # ...but never more than this: aliased or garbage camera speeds must not explain away real shake
SIGNAL_W = 128.0  # the size the signals are measured at: translation in px = fraction of width x this
ZOOM_RADIUS_PX = 40.0  # mean distance of a pixel from the centre at that size: a zoom of rate z moves it by about z x this
ANCHOR_MAX = 3.0  # a translation explained by a zoom about a point farther than this many frame widths from the centre is a pan, not a zoom anchor


@dataclass
class ShotMotion:
    """The camera motion of one shot. ``scale_change`` is the end/start zoom ratio (> 1 in, < 1 out, 1.0 none), ``pan_extent`` the total translation as a
    fraction of the frame width, ``intensity`` the 0..1 score defined in the module docstring. ``kind``: static | zoom_in | zoom_out | pan | high_motion | mixed."""

    shot_id: str
    kind: str = "static"
    intensity: float = 0.0
    scale_change: float = 1.0
    pan_extent: float = 0.0
    duration: float = 0.0
    confidence: float = 0.0


@dataclass
class MotionEvent:
    """One zoom / pan / high-motion event, with timing. INTERNAL cache data (structure analysis, debugging): never part of the style profile."""

    kind: str  # zoom_in | zoom_out | pan | high_motion
    shot_id: str
    start: float
    end: float
    magnitude: float  # zoom: end/start ratio of the event (>1 in, <1 out); pan: extent as a fraction of the width; high_motion: mean intensity
    duration: float = 0.0  # effective duration (the support, de-blurred by the measurement window)


@dataclass
class MotionResult:
    stats: MotionStats = field(default_factory=MotionStats)
    per_shot: list[ShotMotion] = field(default_factory=list)
    motion_series: list[tuple[float, float]] = field(default_factory=list)  # (time, intensity 0..1), about one value per second
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    events: list[MotionEvent] = field(default_factory=list)  # internal: see MotionEvent


# ---------------------------------------------------------------------------------------------- small numeric helpers
def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0 or k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)


def _moving_min(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.min(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0) if len(x) else x.copy()


def _moving_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average over ``k`` samples (the edges average over what exists)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) - k // 2 + k, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _interp(x: np.ndarray, known: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill the unknown samples of ``x`` by linear interpolation across gaps of at most ``max_gap`` samples (inside the known range) and by holding the
    nearest known value over at most ``max_gap`` samples at either end. Returns (filled, still_known); unknown samples are left 0."""
    out, ok = np.where(known, x, 0.0), known.copy()
    idx = np.flatnonzero(known)
    if len(idx) == 0:
        return out, ok
    n = len(x)
    gap_start = None
    for i in range(idx[0] + 1, idx[-1] + 1):
        if not known[i] and gap_start is None:
            gap_start = i
        if known[i] and gap_start is not None:
            if i - gap_start <= max_gap:
                out[gap_start:i] = np.interp(np.arange(gap_start, i), [gap_start - 1, i], [x[gap_start - 1], x[i]])
                ok[gap_start:i] = True
            gap_start = None
    head, tail = int(idx[0]), int(idx[-1])
    h = min(head, max_gap)
    out[head - h:head], ok[head - h:head] = x[head], True
    t = min(n - 1 - tail, max_gap)
    out[tail + 1:tail + 1 + t], ok[tail + 1:tail + 1 + t] = x[tail], True
    return out, ok


def _well_measured(measured: np.ndarray, i: int, j: int, span: tuple[int, int], min_n: int) -> bool:
    """True when at least ``min_n`` of the samples i..j that a window can describe were really measured, and ``MIN_MEASURED_SHARE`` of them (the held head and
    tail of a shot do not count against an event, but they cannot prove one either)."""
    lo, hi = max(i, span[0]), min(j, span[1])
    if hi < lo:
        return False
    seg = measured[lo:hi + 1]
    return int(seg.sum()) >= min_n and float(seg.mean()) >= MIN_MEASURED_SHARE


def _sanitised(sig: FrameSignals) -> FrameSignals:
    """A copy whose motion arrays hold finite numbers only (NaN/inf -> 0, confidence clipped to 0..1)."""
    def clean(a: np.ndarray) -> np.ndarray:
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    return replace(sig, diff=clean(sig.diff), dx=clean(sig.dx), dy=clean(sig.dy), log_scale=clean(sig.log_scale), luma_std=clean(sig.luma_std),
                   motion_conf=np.clip(np.nan_to_num(sig.motion_conf, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))


def _runs(active: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Index ranges [i, j] of consecutive True samples, bridging gaps of up to ``max_gap`` False samples."""
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > max_gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


# ---------------------------------------------------------------------------------------------- the analyzer
class MotionAnalyzer:
    """``analyze(signals, shots) -> MotionResult``. Deterministic, pure numpy, linear in the number of samples."""

    def __init__(self, *, min_confidence: float = MIN_CONF) -> None:
        self.min_conf = min_confidence

    # ------------------------------------------------------------------ public
    def analyze(self, signals: FrameSignals, shots: list[Shot]) -> MotionResult:
        n = len(signals)
        if n < 2 or signals.fps <= 0:
            dur = float(signals.duration) if n else 0.0
            given = sorted((x for x in shots if x.end > x.start), key=lambda x: x.start) or [Shot("S1", 0.0, dur)]
            return MotionResult(per_shot=[ShotMotion(x.shot_id, "static", 0.0, 1.0, 0.0, x.duration, 0.0) for x in given], notes=["Too few frames to measure motion."])
        signals = _sanitised(signals)  # a garbage value must not poison a whole video
        fps = float(signals.fps)
        dt = 1.0 / fps
        ar = signals.sig.shape[1] / signals.sig.shape[2] if signals.sig.ndim == 3 and signals.sig.shape[2] else 9.0 / 16.0  # frame height / width
        duration = float(signals.duration) or float(signals.times[-1] + dt)
        spans = self._spans(signals, shots, duration)
        # frame difference per 8 fps sample, with isolated spikes (cuts) removed, and the noise floor of this video
        d8 = np.clip(_median_filter(signals.diff.astype(np.float64), 3) * fps / 8.0, 0.0, 1.0)
        floor = float(np.clip(np.percentile(d8[1:], 10), 0.0, 0.01))
        inten = np.zeros(n)
        counted = np.zeros(n, dtype=bool)
        per_shot: list[ShotMotion] = []
        events: list[MotionEvent] = []
        for k, (shot, a, b) in enumerate(spans):
            nxt = spans[k + 1][0] if k + 1 < len(spans) else None
            head = int(math.ceil(shot.transition_duration * fps)) if shot.transition_type != "CUT" and shot.transition_duration > 0 else 0
            tail = int(math.ceil(0.5 * nxt.transition_duration * fps)) if nxt is not None and nxt.transition_type != "CUT" and nxt.transition_duration > 0 else 0
            if b < a:  # shorter than a sample: nothing to measure
                per_shot.append(ShotMotion(shot.shot_id, "static", 0.0, 1.0, 0.0, shot.duration, 0.0))
                continue
            sm, ev, i_s = self._shot(signals, shot, a, b, head, tail, ar, d8, floor)
            per_shot.append(sm)
            events += ev
            inten[a:b + 1], counted[a:b + 1] = i_s, True
        return self._summarise(signals, per_shot, events, inten, counted, duration)

    # ------------------------------------------------------------------ shots -> sample ranges
    @staticmethod
    def _spans(signals: FrameSignals, shots: list[Shot], duration: float) -> list[tuple[Shot, int, int]]:
        times = signals.times
        half = 0.5 / signals.fps
        use = sorted((s for s in shots if s.end > s.start), key=lambda s: s.start) or [Shot("S1", 0.0, duration)]
        out = []
        for s in use:
            a = int(np.searchsorted(times, s.start - half, side="left"))
            b = int(np.searchsorted(times, s.end - half, side="left")) - 1
            out.append((s, a, min(b, len(times) - 1)))
        return out

    # ------------------------------------------------------------------ one shot
    def _shot(self, sig: FrameSignals, shot: Shot, a: int, b: int, head: int, tail: int, ar: float, d8: np.ndarray, floor: float
              ) -> tuple[ShotMotion, list[MotionEvent], np.ndarray]:
        """Measure one shot. ``head``/``tail`` samples next to a dissolve/fade are left out of the measurement (the picture is a blend there) and filled by holding
        the neighbouring values, so a transition never looks like movement."""
        lo, hi = a + head, b - tail
        if hi - lo + 1 < 2:
            lo, hi = a, b
        sm, ev, i_s = self._inner(sig, shot, lo, hi, ar, d8, floor)
        if lo > a or hi < b:
            i_s = np.concatenate([np.full(lo - a, i_s[0]), i_s, np.full(b - hi, i_s[-1])])
        return sm, ev, i_s

    def _inner(self, sig: FrameSignals, shot: Shot, a: int, b: int, ar: float, d8: np.ndarray, floor: float) -> tuple[ShotMotion, list[MotionEvent], np.ndarray]:
        fps = float(sig.fps)
        dt = 1.0 / fps
        m = b - a + 1
        zoom, px, py, known, measured, span, nw, usable_frac, conf_mean = self._rates(sig, a, b, ar)
        # --- residual motion: frame difference not explained by noise or by the measured camera travel
        d_shot = d8[a:b + 1].copy()
        if m > 1:
            d_shot[0] = d_shot[1]  # the first sample's difference is the cut
        i_resid = self._residual(d_shot, floor, zoom, px, py, known, fps)
        chaos = _moving_mean((i_resid >= HIGH_INTENSITY).astype(np.float64), max(3, int(round(fps)))) >= 0.4
        if chaos.any():  # where the picture changes this much the 0.5 s camera fit is aliased: do not read zooms and pans from it
            for arr in (zoom, px, py):
                arr[chaos] = 0.0
            known = known & ~chaos
            measured = measured & ~chaos
            i_resid = self._residual(d_shot, floor, zoom, px, py, known, fps)
        # --- a translation that moves in proportion to the zoom is the zoom's anchor, not a pan
        s_tot = float(zoom.sum() * dt)
        if abs(s_tot) >= ZOOM_MIN_CHANGE:
            px, py = self._remove_anchor(zoom, px, py, dt)
        speed = np.hypot(px, py)
        i_zoom = np.sqrt(np.clip((np.abs(zoom) - ZOOM_RATE_FLOOR) / ZOOM_REF_RATE, 0.0, 1.0))
        i_pan = np.sqrt(np.clip((speed - PAN_DEAD_ZONE) / PAN_REF_SPEED, 0.0, 1.0))
        i_s = 1.0 - (1.0 - i_zoom) * (1.0 - i_pan) * (1.0 - i_resid)
        # --- events
        zoom_events = self._zoom_events(shot, a, zoom, measured, span, sig.times, dt)
        pan_events = self._pan_events(shot, a, px, py, measured, span, sig.times, dt)
        hm_events = self._high_motion_events(shot, a, i_s, chaos, sig.times, dt)
        # --- the shot
        intensity = float(i_s.mean())
        resid_mean = float(i_resid.mean())
        scale_change = math.exp(s_tot) if (zoom_events or abs(s_tot) >= 0.01) else 1.0
        kind = self._kind(zoom_events, pan_events, hm_events, intensity, resid_mean)
        conf_shot = self._shot_confidence(nw, usable_frac, conf_mean, resid_mean, sig, a, b)
        pan_extent = float(sum(e.magnitude for e in pan_events))
        sm = ShotMotion(shot.shot_id, kind, round(intensity, 4), round(scale_change, 4), round(pan_extent, 4), float(shot.duration or m * dt), round(conf_shot, 3))
        return sm, zoom_events + pan_events + hm_events, i_s

    def _rates(self, sig: FrameSignals, a: int, b: int, ar: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int], int, float, float]:
        """Zoom (ln scale / s), pan x and pan y (frame widths / s) per sample of the shot, from the lag windows that are inside it, confident and coherent.

        Returns (zoom, px, py, known, measured, span, windows, usable_fraction, mean_confidence).
        ``measured`` marks samples a good window describes directly; ``span`` is the first and last sample any window can describe (the head and tail of a
        shot never are). ``known`` adds the samples filled in: gaps of up to ``MAX_FILL`` samples are interpolated, the head and tail the windows cannot
        reach are held. The rest are zero = no measurable camera motion."""
        lag = max(1, int(sig.lag))
        m = b - a + 1
        windows = np.arange(a + lag, b + 1)  # window i compares sample i with sample i - lag, both inside the shot
        nw = len(windows)
        rate = np.zeros((3, m))
        known = np.zeros(m, dtype=bool)
        usable_frac = conf_mean = 0.0
        if nw > 0:
            conf = sig.motion_conf[windows].astype(np.float64)
            raw = np.stack([sig.log_scale[windows], sig.dx[windows], sig.dy[windows] * ar]).astype(np.float64)
            med = np.stack([_median_filter(raw[k], 5) for k in range(3)])
            tol = np.maximum(np.array([[0.012], [0.03], [0.03]]), 0.4 * np.abs(med))
            good = (conf >= self.min_conf) & np.all(np.abs(raw - med) <= tol, axis=0)  # fast shake aliases into estimates that disagree with their neighbours
            usable_frac = float(good.mean())
            conf_mean = float(conf[good].mean()) if good.any() else 0.0
            centre = np.clip(windows - lag // 2, a, b) - a  # a window describes the motion around its middle
            for k in range(3):
                rate[k, centre[good]] = raw[k, good]
            known[centre[good]] = True
        filled = np.zeros_like(rate)
        still = known
        for k in range(3):
            filled[k], still = _interp(rate[k], known, MAX_FILL)
        return filled[0], filled[1], filled[2], still, known, (lag - lag // 2, max(lag - lag // 2, m - 1 - lag // 2)), nw, usable_frac, conf_mean

    @staticmethod
    def _remove_anchor(zoom: np.ndarray, px: np.ndarray, py: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """A zoom about a point other than the centre also shifts the picture, in proportion to the zoom (dT = -K dS for a fixed anchor). When that fits the
        measured translation (R^2 >= 0.6, anchor within ``ANCHOR_MAX`` widths of the centre) the translation belongs to the zoom and is not a pan; otherwise the
        translation is a pan (or a pan that happens next to a zoom) and is kept."""
        d_s = zoom * dt
        d_t = np.stack([px, py]) * dt
        den = float((d_s * d_s).sum())
        if den <= 0:
            return px, py
        k_vec = (d_t * d_s).sum(axis=1) / den
        res = d_t - np.outer(k_vec, d_s)
        total = float((d_t * d_t).sum())
        if total > 0 and float((res * res).sum()) <= 0.4 * total and float(np.hypot(*k_vec)) <= ANCHOR_MAX:
            return res[0] / dt, res[1] / dt
        return px, py

    @staticmethod
    def _residual(d_shot: np.ndarray, floor: float, zoom: np.ndarray, px: np.ndarray, py: np.ndarray, known: np.ndarray, fps: float) -> np.ndarray:
        """0..1 per sample: the frame difference left over after the noise floor and what the measured camera travel accounts for."""
        cam_px = np.where(known, np.sqrt((px * SIGNAL_W) ** 2 + (py * SIGNAL_W) ** 2 + (zoom * ZOOM_RADIUS_PX) ** 2) / fps, 0.0)
        cam_px = _moving_min(cam_px, 5)  # only a sustained camera move explains frame difference: isolated, aliased windows earn no credit
        explained = np.minimum(EXPLAIN_PER_PIXEL * cam_px, EXPLAIN_CAP)
        d_res = np.maximum(0.0, d_shot - floor - explained)
        return np.interp(d_res, [p[0] for p in RESIDUAL_POINTS], [p[1] for p in RESIDUAL_POINTS])

    # ------------------------------------------------------------------ events
    @staticmethod
    def _zoom_events(shot: Shot, a: int, zoom: np.ndarray, measured: np.ndarray, span: tuple[int, int], times: np.ndarray, dt: float) -> list[MotionEvent]:
        out: list[MotionEvent] = []
        bridge = max(1, int(round(0.5 / dt)))
        min_n = max(3, int(round(MIN_EVENT_SECONDS / dt)))
        for sign, kind in ((1.0, "zoom_in"), (-1.0, "zoom_out")):
            for i, j in _runs(sign * zoom >= ZOOM_RATE_FLOOR, bridge):
                seg = sign * zoom[i:j + 1]
                change = float(seg.sum() * dt)
                if j - i + 1 < min_n or change < ZOOM_MIN_CHANGE or not _well_measured(measured, i, j, span, min_n):
                    continue
                peak = float(np.percentile(seg, 90))
                dur = min(change / peak if peak > 0 else 0.0, (j - i + 1) * dt)
                out.append(MotionEvent(kind, shot.shot_id, float(times[a + i]), float(times[a + j]) + dt, math.exp(sign * change), dur))
        return out

    @staticmethod
    def _pan_events(shot: Shot, a: int, px: np.ndarray, py: np.ndarray, measured: np.ndarray, span: tuple[int, int], times: np.ndarray, dt: float) -> list[MotionEvent]:
        out: list[MotionEvent] = []
        speed = np.hypot(px, py)
        bridge = max(1, int(round(0.5 / dt)))
        min_n = max(3, int(round(MIN_EVENT_SECONDS / dt)))
        for i, j in _runs(speed >= PAN_SPEED_FLOOR, bridge):
            # split where the direction reverses (a pan left and a pan right are two events)
            cuts = [i]
            ref = np.array([0.0, 0.0])
            for k in range(i, j + 1):
                v = np.array([px[k], py[k]])
                if np.linalg.norm(ref) > 0 and float(v @ ref) < 0 and np.linalg.norm(v) >= PAN_SPEED_FLOOR:
                    cuts.append(k)
                    ref = v.copy()
                else:
                    ref = ref * 0.8 + v
            cuts.append(j + 1)
            for s0, s1 in zip(cuts, cuts[1:]):
                if s1 - s0 < min_n or not _well_measured(measured, s0, s1 - 1, span, min_n):
                    continue
                vec = np.array([px[s0:s1].sum(), py[s0:s1].sum()]) * dt
                path = float(speed[s0:s1].sum() * dt)
                net = float(np.linalg.norm(vec))
                if net < PAN_MIN_EXTENT or net < 0.5 * path:
                    continue
                peak = float(np.percentile(speed[s0:s1], 90))
                dur = min(path / peak if peak > 0 else 0.0, (s1 - s0) * dt)
                out.append(MotionEvent("pan", shot.shot_id, float(times[a + s0]), float(times[a + s1 - 1]) + dt, net, dur))
        return out

    @staticmethod
    def _high_motion_events(shot: Shot, a: int, inten: np.ndarray, chaos: np.ndarray, times: np.ndarray, dt: float) -> list[MotionEvent]:
        """Stretches whose *unexplained* motion is high (shake, chaos). A fast clean pan is a pan event, not a high-motion one."""
        out: list[MotionEvent] = []
        bridge = max(1, int(round(0.6 / dt)))
        min_n = max(3, int(round(0.5 / dt)))
        for i, j in _runs(chaos, bridge):
            if j - i + 1 < min_n:
                continue
            total = (j - i + 1) * dt
            parts = max(1, int(math.ceil(total / SUSTAINED_EVENT_SECONDS - 1e-9)))
            mean_i = float(inten[i:j + 1].mean())
            for p in range(parts):
                s0 = i + int(round(p * (j - i + 1) / parts))
                s1 = i + int(round((p + 1) * (j - i + 1) / parts))
                out.append(MotionEvent("high_motion", shot.shot_id, float(times[a + s0]), float(times[a + max(s0, s1 - 1)]) + dt, mean_i, total / parts))
        return out

    # ------------------------------------------------------------------ classification and confidence
    @staticmethod
    def _kind(zoom_events: list[MotionEvent], pan_events: list[MotionEvent], hm_events: list[MotionEvent], intensity: float, resid_mean: float) -> str:
        if resid_mean >= 0.5 or (hm_events and intensity >= 0.6):
            return "high_motion"
        ins = [e for e in zoom_events if e.kind == "zoom_in"]
        outs = [e for e in zoom_events if e.kind == "zoom_out"]
        if zoom_events and pan_events:
            return "mixed"
        if ins and outs:
            return "mixed"
        if ins:
            return "zoom_in"
        if outs:
            return "zoom_out"
        if pan_events:
            return "pan"
        if intensity >= 0.5:
            return "high_motion"
        if intensity >= 0.12:
            return "mixed"  # movement that is neither a clean zoom nor a clean pan (handheld, moving subjects)
        return "static"

    @staticmethod
    def _shot_confidence(nw: int, usable_frac: float, conf_mean: float, resid_mean: float, sig: FrameSignals, a: int, b: int) -> float:
        flat = float(sig.luma_std[a:b + 1].mean()) < 0.012
        if flat:
            return 0.0
        if nw <= 0:
            return 0.15 if resid_mean < 0.5 else 0.45  # too short for the camera fit: only the frame difference is available
        c = usable_frac * (0.5 + 0.5 * conf_mean)
        if nw < 3:
            c *= 0.5
        if resid_mean >= 0.6:
            c = max(c, 0.7)  # high unexplained change is itself a reliable reading, even when the camera cannot be separated
        return float(clamp(c))

    # ------------------------------------------------------------------ totals
    def _summarise(self, sig: FrameSignals, per_shot: list[ShotMotion], events: list[MotionEvent], inten: np.ndarray, counted: np.ndarray, duration: float) -> MotionResult:
        fps = float(sig.fps)
        dt = 1.0 / fps
        minutes = max(duration, dt) / 60.0
        zoom_ev = [e for e in events if e.kind in ("zoom_in", "zoom_out")]
        pan_ev = [e for e in events if e.kind == "pan"]
        hm_ev = [e for e in events if e.kind == "high_motion"]
        total_dur = sum(s.duration for s in per_shot) or duration
        avg_intensity = sum(s.intensity * s.duration for s in per_shot) / total_dur if total_dur > 0 else 0.0
        static_share = sum(s.duration for s in per_shot if s.kind == "static") / total_dur if total_dur > 0 else 1.0
        high_share = float((inten[counted] >= HIGH_INTENSITY).mean()) if counted.any() else 0.0
        epm = (len(zoom_ev) + len(pan_ev) + len(hm_ev)) / minutes
        score = 0.5 * scale(epm, MOTION_EVENT_POINTS) + 0.5 * 100.0 * clamp(avg_intensity)  # the same formula as style_model.compute_scores
        ins = [e for e in zoom_ev if e.kind == "zoom_in"]
        mags = [e.magnitude for e in ins] or [1.0 / e.magnitude for e in zoom_ev]
        punch = sum(1 for e in zoom_ev if e.duration < 1.5)
        n_shots = len(per_shot)
        zoom = ZoomStats(
            events=len(zoom_ev), frequency_per_minute=len(zoom_ev) / minutes,
            average_scale=float(np.mean(mags)) if mags else 1.0, maximum_scale=float(max(mags)) if mags else 1.0,
            average_duration=float(np.mean([e.duration for e in zoom_ev])) if zoom_ev else 0.0,
            zoom_in_share=len(ins) / len(zoom_ev) if zoom_ev else 1.0,
            context={"static_shots": (sum(1 for s in per_shot if s.kind == "static") / n_shots) if n_shots else 0.0,
                     "punch_in": punch / len(zoom_ev) if zoom_ev else 0.0, "slow_zoom": (len(zoom_ev) - punch) / len(zoom_ev) if zoom_ev else 0.0})
        stats = MotionStats(epm, avg_intensity, len(zoom_ev) / minutes, len(pan_ev) / minutes, static_share, high_share, motion_class(score), zoom)
        series = self._series(sig, inten, counted)
        conf, notes = self._confidence(sig, per_shot, total_dur, duration)
        return MotionResult(stats, per_shot, series, conf, notes, events)

    @staticmethod
    def _series(sig: FrameSignals, inten: np.ndarray, counted: np.ndarray) -> list[tuple[float, float]]:
        """About one value per second: the mean intensity of the samples in that second (gaps between shots are skipped)."""
        out: list[tuple[float, float]] = []
        step = max(1, int(round(sig.fps)))
        for s in range(0, len(sig), step):
            sel = counted[s:s + step]
            if sel.any():
                seg = inten[s:s + step][sel]
                out.append((float(sig.times[s:s + step][sel].mean()), round(float(seg.mean()), 4)))
        return out

    @staticmethod
    def _confidence(sig: FrameSignals, per_shot: list[ShotMotion], total_dur: float, duration: float) -> tuple[float, list[str]]:
        notes: list[str] = []
        if total_dur <= 0 or not per_shot:
            return 0.0, ["No shots to measure."]
        c = sum(s.confidence * s.duration for s in per_shot) / total_dur
        length = duration
        c *= float(np.clip(length / 15.0, 0.2, 1.0))  # a few seconds of material proves little about a style
        if length < 6.0:
            notes.append("Very short material: motion style is a rough estimate.")
        if float(sig.luma_std.mean()) < 0.012:
            notes.append("Flat or very dark picture: camera motion cannot be measured.")
        low = sum(s.duration for s in per_shot if s.confidence < 0.3)
        if low / total_dur > 0.3:
            notes.append("Much of the video could not be measured reliably (very short shots, fast motion or little texture).")
        return float(clamp(c)), notes
