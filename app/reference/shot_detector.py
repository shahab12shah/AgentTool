"""Shot, transition and major-visual-change detection from the per-sample frame signals.

Only abstract timing comes out (boundary times, transition classes and durations, counts): no frame, no picture content and no text is kept.

How it works (everything is relative to the video's *own* activity, so it needs no per-video tuning):

* Each sample is reduced to its 16x9 signature (block averages: pixel noise and small shifts average out of it). The *change* between two samples is
  the mean absolute change of the signature once the step's median brightness offset has been removed (so a lighting step or a caption pop-up does
  not shift every block), plus a small share of the raw brightness step (so a cut between two flat cards still registers). *Coverage* says how much
  of the frame changed: cuts change almost everything; caption pop-ups, lower thirds and moving objects change a small part and are never cuts.
* A **hard cut** is a one-sample jump that is above an absolute floor, several times above the shot's own local activity (a lower quantile of its
  neighbours, so camera shake / noise / action - where *every* step is large - never produces cuts), spread over the frame, and not just a brightness
  gain/offset of the same picture (flicker, exposure). A single-sample flash (and a 2-sample uniform white/black flash) after which the picture is
  back is NOT a cut: it is reported as a major visual change.
* **Gradual transitions** are runs of consecutive elevated steps that start and end in quiet footage. Through a flat frame -> FADE (dip to black /
  white / colour); a big net change spread evenly over the frame -> DISSOLVE; a net change that arrives as a moving edge (every block is either still
  the old or already the new picture) -> WIPE; every step already as large as the whole change and explained by a shift of the picture, for at most
  ~1.5 s -> SLIDE; anything gradual that cannot be told apart -> UNKNOWN. Runs explained by camera motion (the signals' motion estimate keeps a steady
  shift/zoom) are pans and zooms, not transitions. Transitions shorter than ~3 samples look like cuts at this sample rate.
* **Major visual changes** (not cuts) are partial-frame composition changes (a large graphic or box appearing, a person walking in) that are not
  explained by camera motion, big exposure jumps / flashes, and dips whose picture comes back.

Boundary times: a hard cut is the first sample of the new shot; a fade, dissolve, wipe or slide is the midpoint of the transition (for dissolves /
wipes / slides the moment half of the change has happened). Transition durations come from the number of elevated steps. The pixel-level ``diff`` /
``hist_diff`` signals are deliberately not used: they react to sensor noise far more than the signature does.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.reference.signals import FrameSignals
from app.reference.style_model import TRANSITION_TYPES, Section, Shot, ShotStats, TransitionStats, clamp, distribution_buckets, pacing_class

# ---------------------------------------------------------------------------------------------- tunables
# "change" = mean absolute change over the 144 blocks of the 16x9 grey signature (0..1)
CUT_FLOOR = 0.03  # a hard cut changes the picture at least this much (a cut between two similar-looking angles is ~0.03..0.08, unrelated shots 0.15+)
CUT_RATIO = 4.0  # ... and at least this many times more than the shot's own local activity
CUT_COVERAGE = 0.40  # ... and over at least this share of the frame (overlays / moving objects do not reach it)
ACTIVE_FLOOR = 0.008  # a step that can belong to a gradual transition
ACTIVE_RATIO = 2.5
NET_FLOOR = 0.04  # the picture after a gradual transition differs from the one before by at least this much
MAJOR_FLOOR = 0.03  # a partial-frame change that counts as a major visual change
MAJOR_COVERAGE = 0.12
MAJOR_SPACING = 1.0  # seconds: nearby major changes are one event
FLAT_STD = 0.02  # a frame whose (signature) contrast is below this is a flat colour (black / white / card) - the dip of a fade
GLOBAL_LEVEL = 0.25  # share of the raw brightness step that counts as change (structure ignores it)
COV_TAU = 0.02
MAX_GRADUAL_SECONDS = 4.0  # a longer run of elevated steps is continuous activity, not a transition
HOLD_SECONDS = 1.5  # flat holds up to this long inside a fade are part of the fade; longer ones are a shot of their own
MAX_SLIDE_SECONDS = 1.5  # a picture that keeps shifting for longer than this is a camera pan, not a slide transition
CAMERA_SPEED = 0.04  # per second: shift (fraction of the frame) or log zoom at which the motion estimate counts as camera movement
MIN_SAMPLES_RELIABLE = 20
SHORT_SHOT_SECONDS = 0.3


# ---------------------------------------------------------------------------------------------- result
@dataclass
class ShotDetection:
    shots: list[Shot]
    stats: ShotStats
    transitions: TransitionStats
    confidence: float
    major_change_times: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cut_times(self) -> list[float]:
        """Every shot boundary (the start of each shot after the first)."""
        return [s.start for s in self.shots[1:]]

    @property
    def transition_events(self) -> list[tuple[float, str, float]]:
        """``(time, type, duration)`` per boundary, in the shape ``ReferenceEvents.transitions`` stores."""
        return [(s.start, s.transition_type, s.transition_duration) for s in self.shots[1:]]


@dataclass
class _Event:
    kind: str  # CUT | FADE | DISSOLVE | WIPE | SLIDE | UNKNOWN
    time: float
    duration: float = 0.0
    confidence: float = 0.5


# ---------------------------------------------------------------------------------------------- per-sample measurements
class _Track:
    """Everything derived once from ``FrameSignals``: signature changes between samples, flat frames, local baselines."""

    def __init__(self, sig: FrameSignals, sens: float) -> None:
        self.sens = sens
        self.n = len(sig)
        self.fps = float(sig.fps) if sig.fps and sig.fps > 0 else 8.0
        self.t = np.asarray(sig.times, dtype=np.float64)
        s = np.asarray(sig.sig, dtype=np.float64)
        if s.ndim != 3:
            s = s.reshape(self.n, 1, -1)
        self.grid = s
        self.vec = s.reshape(self.n, -1)
        self.std = self.vec.std(axis=1)  # contrast of the 16x9 signature: pixel noise averages out of it, so a noisy black frame still counts as flat
        self.flat = self.std < FLAT_STD
        self.motion = tuple(np.asarray(getattr(sig, k), dtype=np.float64) for k in ("dx", "dy", "log_scale", "motion_conf"))
        self.level = self.vec.mean(axis=1)
        d = self.vec[1:] - self.vec[:-1]
        off = np.median(d, axis=1)  # a global brightness step moves the median; a local change (caption, box, object) does not
        blk = np.maximum(np.abs(d - off[:, None]), GLOBAL_LEVEL * np.abs(off)[:, None])
        c = blk.mean(axis=1)
        self.c = np.concatenate([[0.0], c])  # c[i]: change between sample i-1 and i
        self.cov = np.concatenate([[0.0], (blk > np.maximum(COV_TAU, 0.25 * c)[:, None]).mean(axis=1)])
        # what is left of a step after the best brightness gain + offset: flicker and exposure steps leave ~0, a new picture leaves nearly all of it
        x, y = self.vec[:-1], self.vec[1:]
        xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        vx = (xc * xc).mean(axis=1)
        slope = np.clip(np.where(vx > 1e-8, (xc * yc).mean(axis=1) / np.maximum(vx, 1e-8), 1.0), 0.6, 1.7)
        self.gain_resid = np.concatenate([[0.0], np.abs(yc - slope[:, None] * xc).mean(axis=1)])
        cn = self.c.copy()
        cn[0] = np.nan
        self.q_global = float(np.nanquantile(cn, 0.3)) if self.n > 1 else 0.0
        half = max(3, int(round(0.8 * self.fps)))
        self.base = self._local(cn, half, 0.3)  # activity of the shot around a step (cuts every >= 2 samples leave most neighbours quiet)
        self.base_wide = self._local(cn, max(8, int(round(2.0 * self.fps))), 0.15)  # a wider, lower reference for gradual runs
        qg = max(self.q_global, 0.0)
        self.cut_thr = np.maximum(CUT_FLOOR / sens, max(1.5, CUT_RATIO / sens) * np.maximum(self.base, qg))  # (sensitivity > 1 is more eager)
        self.active_thr = np.maximum(ACTIVE_FLOOR / sens, max(1.4, ACTIVE_RATIO / sens) * np.maximum(self.base_wide, qg))
        self.coverage_needed = float(clamp(CUT_COVERAGE / math.sqrt(sens), 0.25, 0.5))
        self.cut_cand = (self.c >= self.cut_thr) & (self.cov >= self.coverage_needed)
        self.cut_cand[0] = False

    def _local(self, x: np.ndarray, half: int, q: float) -> np.ndarray:
        pad = np.full(len(x) + 2 * half, np.nan)
        pad[half:half + len(x)] = x
        win = sliding_window_view(pad, 2 * half + 1).copy()
        win[:, half] = np.nan  # the step itself must not set its own baseline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanquantile(win, q, axis=1)
        return np.where(np.isnan(out), self.q_global, out)

    def pair(self, i: int, j: int) -> tuple[float, float]:
        """(change, coverage) between two arbitrary samples."""
        d = self.vec[j] - self.vec[i]
        off = float(np.median(d))
        blk = np.maximum(np.abs(d - off), GLOBAL_LEVEL * abs(off))
        net = float(blk.mean())
        return net, float((blk > max(COV_TAU, 0.25 * net)).mean())

    def cut_candidate(self, i: int) -> bool:
        """A big, frame-wide, locally outstanding step (still to be checked for being only a brightness change)."""
        return bool(self.cut_cand[i])

    def gain_explained(self, i: int) -> bool:
        """Flicker / flash / exposure step: the same picture with another brightness gain or offset. (A change between two flat cards has no picture to keep: a cut.)"""
        return bool(max(self.std[i - 1], self.std[i]) >= 0.04 and self.gain_resid[i] < 0.4 * self.c[i])

    def is_cut_step(self, i: int) -> bool:
        return self.cut_candidate(i) and not self.gain_explained(i)

    def pair_gain_resid(self, i: int, j: int) -> float:
        """What is left of the change between two samples after the best brightness gain + offset."""
        x, y = self.vec[i], self.vec[j]
        xc, yc = x - x.mean(), y - y.mean()
        vx = float((xc * xc).mean())
        slope = float(np.clip((xc * yc).mean() / vx, 0.6, 1.7)) if vx > 1e-8 else 1.0
        return float(np.abs(yc - slope * xc).mean())

    def all_flat(self, i: int, j: int) -> bool:
        return j >= i and bool(self.flat[i:j + 1].all())


# ---------------------------------------------------------------------------------------------- the detector
class ShotDetector:
    """Detects shots, how each was entered, and major non-cut visual changes. ``sensitivity`` > 1 finds weaker cuts/transitions (and risks false
    ones), < 1 only the unmistakable ones; it is clamped to 0.25..3."""

    def __init__(self, sensitivity: float = 1.0) -> None:
        self.sensitivity = float(clamp(sensitivity, 0.25, 3.0))

    # ------------------------------------------------------------------ public
    def detect(self, signals: FrameSignals, sections: list[Section] | None = None) -> ShotDetection:
        n = len(signals)
        notes: list[str] = []
        if n == 0:
            return ShotDetection([], ShotStats(), TransitionStats(), 0.0, [], ["No frames were sampled: no shots can be detected."])
        fps = float(signals.fps) if signals.fps and signals.fps > 0 else 8.0
        last = float(signals.times[-1])
        duration = float(signals.duration) if signals.duration and signals.duration > last else last + 1.0 / fps
        if n < 2:
            shot = Shot("shot_0001", 0.0, duration, float(signals.times[0]), "CUT", 0.0, 0.05)
            notes.append("Only one frame was sampled: the whole video is reported as one shot with very low confidence.")
            return ShotDetection([shot], compute_shot_stats([shot], duration, sections), compute_transition_stats([shot], duration), 0.05, [], notes)

        tr = _Track(signals, self.sensitivity)
        events, majors, info = self._events(tr)
        events = self._enforce_min_shot(tr, events, duration)
        shots = self._build_shots(tr, events, duration)
        major_times = self._finish_majors(tr, majors, events)
        stats = compute_shot_stats(shots, duration, sections, major_changes=len(major_times))
        trans = compute_transition_stats(shots, duration)
        confidence = round(self._confidence(tr, shots, events, info, notes), 3)
        if len(shots) == 1:
            shots[0].confidence = confidence  # a lone shot is only as certain as "there is no cut in here"
        return ShotDetection(shots, stats, trans, confidence, major_times, notes)

    # ------------------------------------------------------------------ events
    def _events(self, tr: _Track) -> tuple[list[_Event], list[tuple[float, float]], dict]:
        n = tr.n
        act = (tr.c >= tr.active_thr) | tr.cut_cand
        act[0] = False
        events: list[_Event] = []
        majors: list[tuple[float, float]] = []  # (time, strength)
        info = {"explained": np.zeros(n, dtype=bool), "ignored_edges": 0, "flashes": 0, "unknown": 0}
        cut_steps: list[int] = []
        max_len = max(3, int(round(MAX_GRADUAL_SECONDS * tr.fps)))
        for merged, parts in self._runs(tr, act):
            if len(parts) > 1:  # bridged over flat frames: first try the whole as ONE fade; if it is not one, its parts stand on their own (e.g. a hard black gap)
                a, b = merged[0] - 1, merged[1]
                if 3 <= b - a <= max_len and self._fade(tr, a, b, int(act[merged[0]:merged[1] + 1].sum()), events, majors, info):
                    info["explained"][merged[0]:merged[1] + 1] = True
                    continue
            for r0, r1 in parts:
                self._handle_run(tr, r0, r1, int(act[r0:r1 + 1].sum()), max_len, events, majors, cut_steps, info)
        # flashes: an excursion of one sample (or two uniform ones) that returns to the picture before it is not a cut
        cut_steps = sorted(set(cut_steps))
        keep = [True] * len(cut_steps)
        for x in range(len(cut_steps) - 1):
            i, j = cut_steps[x], cut_steps[x + 1]
            gap = j - i
            if gap > 2 or not keep[x] or i < 1:
                continue
            back, _ = tr.pair(i - 1, j)
            returns = back <= 0.35 * min(tr.c[i], tr.c[j])
            if returns and (gap == 1 or bool((tr.std[i:j] < 0.05).all())):
                keep[x] = keep[x + 1] = False
                majors.append((float(tr.t[i]), float(min(tr.c[i], tr.c[j]))))
                info["flashes"] += 1
                info["explained"][i:j + 1] = True
        for k, ok in zip(cut_steps, keep):
            if ok:
                events.append(self._cut_event(tr, k))
        events.sort(key=lambda e: e.time)
        return events, majors, info

    def _handle_run(self, tr: _Track, r0: int, r1: int, active_steps: int, max_len: int, events: list[_Event], majors: list[tuple[float, float]], cut_steps: list[int],
                    info: dict) -> None:
        """Decide what one run of elevated steps (frames a -> b) is: fade, hard cut, dissolve / wipe / slide, an exposure flash, a major change - or nothing."""
        a, b, length = r0 - 1, r1, r1 - r0 + 1
        gradual_ok = 3 <= length <= max_len
        done = False
        if gradual_ok and self._fade(tr, a, b, active_steps, events, majors, info):
            done = True
        elif self._dominant_cut(tr, r0, r1, cut_steps):
            done = True
        elif gradual_ok and self._gradual(tr, a, b, events, info):
            done = True
        elif length <= 2:
            took = [k for k in range(r0, r1 + 1) if tr.is_cut_step(k)]
            if took:
                cut_steps.extend(took)
                done = True
        if not done:
            exposure = [k for k in range(r0, r1 + 1) if tr.cut_candidate(k) and tr.gain_explained(k)]
            if exposure:  # lights on/off, a bright flash: the picture stayed, its brightness jumped. Big jumps are visual events, small ones are nothing.
                majors.extend((float(tr.t[k]), float(tr.c[k])) for k in exposure if abs(tr.level[k] - tr.level[k - 1]) >= 0.1)
                done = True
            elif self._maybe_major(tr, a, b, majors):
                done = True
        if done:
            info["explained"][r0:r1 + 1] = True

    def _runs(self, tr: _Track, act: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
        """Runs of consecutive elevated steps. Runs separated only by flat frames (<= ~1.5 s) are bridged: ``(merged, [its raw runs])``."""
        raw: list[list[int]] = []
        i = 1
        while i < tr.n:
            if act[i]:
                j = i
                while j + 1 < tr.n and act[j + 1]:
                    j += 1
                raw.append([i, j])
                i = j + 1
            else:
                i += 1
        # let the partial first/last steps of a gradual run join it (the sampling phase makes them a fraction of the others)
        for r in raw:
            if r[1] - r[0] >= 2:
                med = float(np.median(tr.c[r[0]:r[1] + 1]))
                while r[0] > 1 and tr.c[r[0] - 1] >= max(0.25 * med, 2.0 * tr.base_wide[r[0] - 1], 0.5 * ACTIVE_FLOOR):
                    r[0] -= 1
                while r[1] < tr.n - 1 and tr.c[r[1] + 1] >= max(0.25 * med, 2.0 * tr.base_wide[r[1] + 1], 0.5 * ACTIVE_FLOOR):
                    r[1] += 1
        fixed: list[list[int]] = []
        for r in raw:
            if fixed and r[0] <= fixed[-1][1] + 1:
                fixed[-1][1] = max(fixed[-1][1], r[1])
            else:
                fixed.append(r)
        hold = max(1, int(round(HOLD_SECONDS * tr.fps)))
        out: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for r in fixed:
            if out:
                prev_b, next_a = out[-1][0][1], r[0] - 1
                if next_a - prev_b <= hold and tr.all_flat(prev_b, next_a):
                    out[-1] = ((out[-1][0][0], r[1]), out[-1][1] + [(r[0], r[1])])
                    continue
            out.append(((r[0], r[1]), [(r[0], r[1])]))
        return out

    # ---- hard cuts
    def _cut_event(self, tr: _Track, k: int) -> _Event:
        margin = float(tr.c[k] / max(1e-9, tr.cut_thr[k]))
        strength = 1.0 - math.exp(-(margin - 1.0) * 0.9)
        cover = min(1.0, float(tr.cov[k]) / 0.7)
        return _Event("CUT", float(tr.t[k]), 0.0, round(clamp(0.3 + 0.7 * (0.75 * strength + 0.25 * cover)), 3))

    def _dominant_cut(self, tr: _Track, r0: int, r1: int, cut_steps: list[int]) -> bool:
        """A run with one step carrying most of the change is a hard cut followed by compression settle / ghost steps."""
        steps = tr.c[r0:r1 + 1]
        k = r0 + int(np.argmax(steps))
        if r1 == r0 or not tr.is_cut_step(k):
            return False
        rest = np.delete(steps, k - r0)
        if steps.max() >= 0.5 * steps.sum() and steps.max() >= 3.0 * rest.max():
            cut_steps.append(k)
            return True
        return False

    # ---- fades (through a flat frame)
    def _fade(self, tr: _Track, a: int, b: int, active_steps: int, events: list[_Event], majors: list[tuple[float, float]], info: dict) -> bool:
        inner = tr.std[a + 1:b]
        if len(inner) == 0:
            return False
        ref = float(min(tr.std[a], tr.std[b]))
        dip = float(inner.min())
        fps = tr.fps
        if ref >= 0.04 and dip <= min(FLAT_STD, 0.25 * ref):  # content -> flat colour -> content
            if active_steps < 3:
                return False  # two hard edges around a black gap: cuts, not a fade (no ramp)
            net, _ = tr.pair(a, b)
            if net < NET_FLOOR / tr.sens:  # the same picture comes back: a dip / flash frame, not a boundary
                majors.append((float(0.5 * (tr.t[a + 1] + tr.t[b])), net))
                info["flashes"] += 1
                return True
            depth = 1.0 - dip / max(ref, 1e-6)
            conf = clamp(0.5 + 0.35 * depth + 0.15 * min(1.0, (b - a) / 6.0))
            t = float(0.5 * (tr.t[a + 1] + tr.t[b]))  # the midpoint of the whole fade (the centre of the flat dip for a symmetric one)
            events.append(_Event("FADE", t, round((b - a - 1) / fps, 3), round(conf, 3)))
            return True
        # one side flat: a fade to / from a flat shot (an opening fade-in / closing fade-out has no shot on the other side: not a boundary)
        if tr.flat[a] != tr.flat[b] and ref < FLAT_STD and max(float(tr.std[a]), float(tr.std[b])) >= 0.04:
            if (tr.flat[a] and tr.all_flat(0, a)) or (tr.flat[b] and tr.all_flat(b, tr.n - 1)):
                info["ignored_edges"] += 1
                return True
            ramp = np.diff(tr.std[a:b + 1]) * (1.0 if tr.flat[a] else -1.0)  # contrast should rise out of / fall into the flat colour
            if active_steps >= 3 and float((ramp > -0.01).mean()) >= 0.8:
                events.append(_Event("FADE", self._half_time(tr, a + 1, b), round((b - a - 1) / fps, 3), 0.7))
                return True
        return False

    # ---- dissolve / wipe / slide
    def _gradual(self, tr: _Track, a: int, b: int, events: list[_Event], info: dict) -> bool:
        net, covn = tr.pair(a, b)
        if net < NET_FLOOR / tr.sens or covn < tr.coverage_needed or tr.pair_gain_resid(a, b) < 0.5 * net:
            return False  # too small, only part of the frame, or just a brightness ramp (not a change of picture)
        steps = tr.c[a + 1:b + 1]
        path = float(steps.sum())
        med = float(np.median(steps))
        fps = tr.fps
        kind, conf = "UNKNOWN", 0.45
        if med / net >= 0.55:  # every step is already about as large as the whole change: the picture moves (push / slide / whip pan)
            gain, consistent = self._shift_explained(tr, a, b)
            if gain >= 0.5 and consistent:
                if (b - a - 1) / fps > MAX_SLIDE_SECONDS:
                    return False  # keeps shifting for too long: a pan
                kind, conf = "SLIDE", 0.75
            elif net < 2.0 * NET_FLOOR / tr.sens:  # big steps that are neither a clean shift nor a big picture change: jitter / flicker, not a transition
                return False
        else:
            if path > 2.2 * net or self._camera_motion(tr, a, b):  # not a straight old -> new path (something moved there and back), or a camera pan / zoom
                return False
            bm = self._bimodality(tr, a, b)
            if bm >= 0.28:
                kind, conf = "DISSOLVE", 0.85
            elif bm <= 0.15:
                kind, conf = "WIPE", 0.7
        if kind == "UNKNOWN":
            info["unknown"] += 1
        margin = min(1.0, net / (2.5 * NET_FLOOR))
        events.append(_Event(kind, self._half_time(tr, a + 1, b), round((b - a - 1) / fps, 3), round(clamp(conf * (0.75 + 0.25 * margin)), 3)))
        return True

    def _half_time(self, tr: _Track, s0: int, s1: int) -> float:
        """The time at which half of the total change over steps s0..s1 has happened."""
        c = tr.c[s0:s1 + 1]
        total = float(c.sum())
        if total <= 0:
            return float(0.5 * (tr.t[s0 - 1] + tr.t[s1]))
        half, acc = 0.5 * total, 0.0
        for k, v in enumerate(c):
            if acc + v >= half and v > 0:
                f = (half - acc) / v
                return float(tr.t[s0 + k - 1] + f * (tr.t[s0 + k] - tr.t[s0 + k - 1]))
            acc += float(v)
        return float(tr.t[s1])

    def _bimodality(self, tr: _Track, a: int, b: int) -> float:
        """0 for a wipe (every block is either still the old or already the new picture), ~0.5 mid-way through a dissolve (blocks are half-way)."""
        A, B = tr.grid[a].reshape(-1), tr.grid[b].reshape(-1)
        D = B - A
        w = np.abs(D)
        ok = w > max(0.02, 0.25 * float(w.mean()))
        if not ok.any():
            return 0.0
        den = float((w * ok).sum())
        best = 0.0
        for k in range(a + 1, b):
            p = np.clip(((tr.grid[k].reshape(-1) - A) * D) / np.maximum(D * D, 1e-9), 0.0, 1.0)
            best = max(best, float((np.minimum(p, 1.0 - p) * w * ok).sum()) / den)
        return best

    def _shift_explained(self, tr: _Track, a: int, b: int) -> tuple[float, bool]:
        """How much of each step is explained by a whole-picture shift (median gain) and whether the shift keeps its axis and direction."""
        gains, dirs = [], []
        for k in range(a + 2, b):
            g, d = self._best_shift(tr.grid[k - 1], tr.grid[k])
            gains.append(g)
            dirs.append(d)
        if not gains:
            return 0.0, False
        nz = [d for d in dirs if d != (0, 0)]
        consistent = bool(nz) and len(nz) >= 0.7 * len(dirs) and all((d[0] > 0) == (nz[0][0] > 0) and (d[1] > 0) == (nz[0][1] > 0) and (d[0] == 0) == (nz[0][0] == 0) for d in nz)
        return float(np.median(gains)), consistent

    @staticmethod
    def _best_shift(p: np.ndarray, q: np.ndarray) -> tuple[float, tuple[int, int]]:
        h, w = p.shape
        r0 = float(np.abs(q - p).mean())
        if r0 < 1e-9:
            return 0.0, (0, 0)
        best, arg = r0, (0, 0)
        for s in range(1, max(2, w // 3) + 1):
            for sx in (s, -s):
                r = float(np.abs(q[:, sx:] - p[:, :-sx]).mean()) if sx > 0 else float(np.abs(q[:, :sx] - p[:, -sx:]).mean())
                if r < best:
                    best, arg = r, (sx, 0)
        for s in range(1, max(2, h // 3) + 1):
            for sy in (s, -s):
                r = float(np.abs(q[sy:, :] - p[:-sy, :]).mean()) if sy > 0 else float(np.abs(q[:sy, :] - p[-sy:, :]).mean())
                if r < best:
                    best, arg = r, (0, sy)
        return 1.0 - best / r0, arg

    # ---- major (non-cut) visual changes
    def _maybe_major(self, tr: _Track, a: int, b: int, majors: list[tuple[float, float]]) -> bool:
        """A frame-partial composition change (a big graphic or box appearing, a person walking in) that is neither a cut nor camera motion."""
        net, covn = tr.pair(a, b)
        if net < MAJOR_FLOOR / tr.sens or covn < MAJOR_COVERAGE or tr.pair_gain_resid(a, b) < 0.5 * net:
            return False
        if self._camera_motion(tr, a, b):  # camera motion is not a change of what is in the picture
            return False
        majors.append((self._half_time(tr, a + 1, b), net))
        return True

    def _camera_motion(self, tr: _Track, a: int, b: int) -> bool:
        """True when the signals' motion estimate says the camera was moving (pan / tilt / zoom) for the whole run a -> b: a steady shift or zoom that the
        estimate matched with confidence at (almost) every sample. A dissolve or wipe has no consistent shift; a slide's match collapses mid-way."""
        dx, dy, ls, conf = tr.motion
        if not all(len(x) == tr.n for x in tr.motion):
            return False
        lag = max(1, int(round(0.5 * tr.fps)))  # the estimate at i compares samples i-lag and i; before ``lag`` there is none
        lo, hi = a + lag, min(b, tr.n - 1)
        if hi - lo + 1 < 3:
            return False
        seg = slice(lo, hi + 1)
        moving = (conf[seg] >= 0.6) & ((np.hypot(dx[seg], dy[seg]) >= CAMERA_SPEED) | (np.abs(ls[seg]) >= CAMERA_SPEED))
        return bool(moving.mean() >= 0.8)

    def _finish_majors(self, tr: _Track, majors: list[tuple[float, float]], events: list[_Event]) -> list[float]:
        bounds = [e.time for e in events]
        cand = sorted((m for m in majors if all(abs(m[0] - t) > 0.5 for t in bounds)), key=lambda m: m[0])
        out: list[tuple[float, float]] = []
        for t, s in cand:
            if out and t - out[-1][0] < MAJOR_SPACING:
                if s > out[-1][1]:
                    out[-1] = (t, s)
            else:
                out.append((t, s))
        return [round(t, 3) for t, _ in out]

    # ------------------------------------------------------------------ shots
    def _enforce_min_shot(self, tr: _Track, events: list[_Event], duration: float) -> list[_Event]:
        """Shots shorter than ~2 samples cannot be told from glitches: of two boundaries closer than that, only the more certain one stays
        (and none within that distance of either end of the video)."""
        gap = 1.5 / tr.fps
        out: list[_Event] = []
        for e in events:
            if e.time <= tr.t[0] + gap or e.time > duration - gap:
                continue
            if out and e.time - out[-1].time < gap:
                if e.confidence > out[-1].confidence:
                    out[-1] = e
                continue
            out.append(e)
        return out

    def _build_shots(self, tr: _Track, events: list[_Event], duration: float) -> list[Shot]:
        starts = [0.0] + [e.time for e in events]
        ends = starts[1:] + [duration]
        shots: list[Shot] = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            enter = events[i - 1] if i > 0 else None
            leave = events[i] if i < len(events) else None
            lo = s + (0.5 * enter.duration if enter else 0.0)
            hi = e - (0.5 * leave.duration if leave else 0.0)
            if hi - lo < 2.0 / tr.fps:
                lo, hi = s, e
            centre = 0.5 * (lo + hi)
            inside = np.flatnonzero((tr.t >= s) & (tr.t < e))
            if len(inside):
                pick = inside[int(np.argmin(np.abs(tr.t[inside] - centre)))]
                frame_sample = float(tr.t[pick])
            else:
                frame_sample = float(clamp(centre, s, max(s, e)))
            conf_parts = [x.confidence for x in (enter, leave) if x is not None]
            conf = float(sum(conf_parts) / len(conf_parts)) if conf_parts else 1.0
            shots.append(Shot(f"shot_{i + 1:04d}", round(s, 4), round(e, 4), round(frame_sample, 4), enter.kind if enter else "CUT", enter.duration if enter else 0.0, round(conf, 3)))
        return shots

    # ------------------------------------------------------------------ confidence
    def _confidence(self, tr: _Track, shots: list[Shot], events: list[_Event], info: dict, notes: list[str]) -> float:
        n, fps = tr.n, tr.fps
        structured = np.concatenate([[False], np.maximum(tr.std[:-1], tr.std[1:]) >= 0.04])
        left = np.where(structured, np.minimum(tr.c, tr.gain_resid / 0.4), tr.c)  # what no brightness gain explains: flicker / exposure changes are no hidden cuts
        busy_mask = (left >= CUT_FLOOR / tr.sens) & (tr.cov >= 0.75 * tr.coverage_needed) & ~info["explained"]  # strong frame-wide change nobody explained
        busy_mask[0] = False
        busy_share = float(busy_mask[1:].mean()) if n > 1 else 0.0
        if events:
            sep = float(np.median([e.confidence for e in events]))
        else:
            thr = float(np.median(tr.cut_thr[1:]))
            cutlike = (left * np.minimum(1.0, tr.cov / tr.coverage_needed))[1:]  # a small overlay change is no sign of a hidden cut
            rest = cutlike[~info["explained"][1:]]
            top = float(rest.max()) if len(rest) else 0.0
            margin = thr / max(top, 1e-9)
            sep = clamp(1.0 - 0.9 * math.exp(-(margin - 1.0) * 0.9)) if margin > 1 else 0.1
        conf = 0.97 * sep
        conf *= 0.1 + 0.9 * min(1.0, n / MIN_SAMPLES_RELIABLE) ** 1.5
        if n < MIN_SAMPLES_RELIABLE:
            notes.append(f"Only {n} samples were available: shot boundaries are not reliable.")
        if busy_share > 0.15:
            conf *= 1.0 - 0.8 * min(1.0, busy_share * 1.4)
            notes.append(f"Constant strong frame-to-frame change over {busy_share * 100:.0f}% of the video (camera shake, noise, strobing or extremely rapid editing): cuts inside it cannot be separated, "
                         "so it is reported as continuous footage and the confidence is low.")
        durations = [s.duration for s in shots]
        short = [d for d in durations if d < max(SHORT_SHOT_SECONDS, 2.5 / fps)]
        if short or busy_share > 0.3:
            notes.append(f"Shots shorter than ~{SHORT_SHOT_SECONDS:g} s may exist that cannot be resolved: at {fps:g} samples/s the detectable cut rate is limited to about one cut per two samples.")
        if short and len(short) > 0.3 * len(shots) and len(shots) > 3:
            conf *= 0.7
        if len(shots) > 1 and (len(shots) - 1) > (n - 1) / 2.5:
            conf = min(conf, 0.35)
            notes.append("The shot rate is at the limit of the sample rate: the result is probably a lower bound.")
        if info["flashes"]:
            notes.append(f"{info['flashes']} single-frame flash(es) were not counted as cuts (the picture returned); they are reported as major visual changes.")
        if info["unknown"]:
            ambiguous = info["unknown"] / max(1, len(events))
            conf *= 1.0 - 0.4 * ambiguous
            notes.append(f"{info['unknown']} gradual transition(s) could not be classified (reported as UNKNOWN).")
        if info["ignored_edges"]:
            notes.append("An opening fade-in or closing fade-out was found; it is part of the first/last shot, not a boundary.")
        if fps < 4:
            conf *= 0.8
            notes.append(f"The sample rate ({fps:g} fps) is low: transitions shorter than about {3 / fps:.1f} s look like hard cuts.")
        return float(clamp(conf, 0.0, 1.0))


# ---------------------------------------------------------------------------------------------- statistics
def _bimodal(durations: list[float]) -> bool:
    """Two clearly separate groups of shot lengths (every long shot at least twice any short one, at least two of each, widely spread)."""
    n = len(durations)
    if n < 4:
        return False
    d = sorted(max(x, 1e-3) for x in durations)
    ratios = [d[i + 1] / d[i] for i in range(n - 1)]
    k = int(np.argmax(ratios))
    small, large = k + 1, n - k - 1
    need = max(2, int(math.ceil(0.2 * n)))
    mean = float(np.mean(d))
    cv = float(np.std(d)) / mean if mean > 0 else 0.0
    return ratios[k] >= 2.0 and small >= need and large >= need and cv >= 0.35


def compute_shot_stats(shots: list[Shot], duration: float, sections: list[Section] | None = None, major_changes: int = 0) -> ShotStats:
    """Rhythm numbers of a shot list. Average alone misleads (spec 7), so median, min, max, std and the length distribution always come with it.

    ``cuts_per_minute`` counts every shot boundary; ``major_change_per_minute`` counts boundaries *plus* the ``major_changes`` that are not
    boundaries. ``cuts_per_section`` counts boundaries per section (``start <= t < end``) or, without sections, per equal block of the video
    (about one block per 15 s, 1..8 blocks); ``cuts_per_scene`` is their mean."""
    if not shots:
        return ShotStats()
    durs = [max(0.0, float(s.duration)) for s in shots]
    arr = np.asarray(durs, dtype=np.float64)
    cuts = len(shots) - 1
    minutes = duration / 60.0 if duration > 0 else 0.0
    cpm = cuts / minutes if minutes > 0 else 0.0
    bounds = [s.start for s in shots[1:]]
    if sections:
        per = [float(sum(1 for t in bounds if sec.start <= t < sec.end)) for sec in sections]
    else:
        blocks = int(clamp(round(duration / 15.0), 1, 8)) if duration > 0 else 1
        width = duration / blocks if duration > 0 else 1.0
        per = [0.0] * blocks
        for t in bounds:
            per[min(blocks - 1, max(0, int(t / width)))] += 1.0
    return ShotStats(
        count=len(shots), average_shot_duration=float(arr.mean()), median_shot_duration=float(np.median(arr)), minimum_shot_duration=float(arr.min()),
        maximum_shot_duration=float(arr.max()), std_shot_duration=float(arr.std()), shot_duration_distribution=distribution_buckets(durs), cuts_per_minute=cpm,
        cut_frequency_class=pacing_class(cpm), cuts_per_scene=float(np.mean(per)) if per else 0.0, cuts_per_section=per,
        major_change_per_minute=(cuts + max(0, int(major_changes))) / minutes if minutes > 0 else 0.0, alternates_long_and_short=_bimodal(durs))


def compute_transition_stats(shots: list[Shot], duration: float) -> TransitionStats:
    """How the shots were entered. No boundary at all -> an empty distribution (a lone shot has no transitions to describe)."""
    entered = shots[1:]
    if not entered:
        return TransitionStats(0.0, 0.0, {}, 0.0)
    kinds = [s.transition_type if s.transition_type in TRANSITION_TYPES else "UNKNOWN" for s in entered]
    dist = {t: kinds.count(t) / len(kinds) for t in TRANSITION_TYPES}
    non_cut = [s for s, k in zip(entered, kinds) if k != "CUT"]
    minutes = duration / 60.0 if duration > 0 else 0.0
    return TransitionStats(len(non_cut) / minutes if minutes > 0 else 0.0, len(non_cut) / len(entered), dist,
                           float(np.mean([s.transition_duration for s in non_cut])) if non_cut else 0.0)
