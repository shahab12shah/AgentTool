"""StructureAnalyzer: the high-level shape of a reference video (hook, sections, pacing curve, energy arc, section transitions) from the observations the
other analyzers already made. No frame, no audio sample and no text is read here: the input is ``ReferenceEvents`` (cut times, transitions, text/graphic
events as *geometry and timing*, a motion series, an audio series, silences, SFX and music-change times) plus the shots.

What it infers (all of it a heuristic over abstract numbers; the section *names* are guesses and carry a low confidence, never a reading of the script)

* **Sections.** The video is put on a one-second grid; every second gets eight 0..1 levels (cut rate, text-overlay rate, caption coverage, motion, loudness,
  voice, music, silence). The *novelty* of a point is the weighted RMS difference between the mean levels of the stretch before it and the stretch after it.
  Peaks of the novelty curve become section boundaries (they must clear an absolute floor and stand clear of the video's own background novelty). A pause
  across the boundary, a non-cut transition, a text card or a music change next to it adds a little to its novelty (the cues the spec asks for), but cannot
  create a boundary out of nothing. A boundary is snapped to a nearby shot boundary.
* **Section kinds.** Positional and energy rules: a short opening is HOOK (a long one CONTEXT), the section after it CONTEXT, a short closing section with
  more text, less voice or more music than the rest is CTA (otherwise CONCLUSION), the single most energetic interior section REVEAL, slow text-heavy sections
  EVIDENCE, busy moving sections EXAMPLES, the first interior section that is more energetic than the context PROBLEM, everything else EXPLANATION.
* **Pacing curve.** A smoothed cuts-per-minute curve is labelled Slow / Moderate / Fast / Very Fast per second and cut into time segments (short blips are
  absorbed by their neighbours), each with its own cut rate, shot length, motion and text rate.
* **Hook.** The first 5 / 10 / 15 / 30 s (those that fit) are measured on shot rate, text density, motion, audio intensity, visual changes and number emphasis;
  ``intensity`` says how much more intense the opening is than the rest; ``traits`` are abstract phrases ("Fast visual switching", "Strong text").
* **Section transitions.** At every boundary: which transition was used, the pause around it, whether a text card / music change / pacing change goes with it.
* **Energy arc.** A coarse 0..1 energy series and its overall shape (front-loaded, building, peak in the middle, falling, flat).

Everything is deterministic (no randomness, no clock) and pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.reference.style_model import (
    PACING_POINTS, TEXT_POINTS, HookProfile, HookWindow, PacingSegment, ReferenceEvents, Section, SectionTransitionStyle, Shot, TextEvent, clamp, pacing_class, scale,
)

STEP = 1.0  # seconds per grid bin
HOOK_WINDOWS = (5, 10, 15, 30)
MIN_ANALYSIS_SECONDS = 6.0  # shorter videos are one section
MIN_SIDE_BINS = 3  # a boundary needs at least this much material on each side
NOVELTY_FLOOR = 0.16  # absolute: a boundary changes the weighted levels by at least this much (RMS of the 0..1 level differences)
NOVELTY_RATIO = 1.6  # ... and stands this far above the video's median background novelty
CUE_MIN_BASE = 0.06  # cues (pause / transition / text card / music change) add to a real change, they do not make one
CUE_GAIN = {"pause": 0.04, "transition": 0.05, "text_card": 0.04, "music": 0.04}
CUE_RADIUS = 1.5
SNAP_RADIUS = 1.5  # a boundary moves to a shot boundary this close
MAX_SECTIONS = 9
LEVEL_WEIGHTS = {"cuts": 0.30, "text": 0.12, "caption": 0.10, "motion": 0.12, "loud": 0.12, "voice": 0.08, "music": 0.08, "silence": 0.08}
LEVELS = tuple(LEVEL_WEIGHTS)
TEXT_LIKE = ("HEADLINE", "NUMBER_CARD", "LOWER_THIRD", "TEXT")
CARD_KINDS = ("HEADLINE", "NUMBER_CARD")


@dataclass
class StructureResult:
    """What the structure analysis produced. ``sections`` / ``pacing_curve`` / ``hook`` / ``section_transition_style`` enter ``ReferenceFeatures``; the rest is
    cache-only detail (boundary strengths, the energy arc) that explains the result."""

    sections: list[Section] = field(default_factory=list)
    pacing_curve: list[PacingSegment] = field(default_factory=list)
    hook: HookProfile = field(default_factory=HookProfile)
    section_transition_style: SectionTransitionStyle = field(default_factory=SectionTransitionStyle)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    energy_arc: list[tuple[float, float]] = field(default_factory=list)  # (time, 0..1)
    arc_shape: str = "flat"  # front-loaded | building | peak in the middle | falling | flat
    boundaries: list[float] = field(default_factory=list)  # section boundary times (without 0 and the end)
    boundary_strength: list[float] = field(default_factory=list)  # novelty of each boundary
    intro_end: float = 0.0  # end of the opening section
    outro_start: float = 0.0  # start of the closing section
    cta_start: float | None = None  # start of a section judged to be a call to action (None: no evidence of one)


# ---------------------------------------------------------------------------------------------- the grid
class _Grid:
    """Per-second raw observations plus prefix sums, so any stretch can be summarised in O(1)."""

    def __init__(self, ev: ReferenceEvents, cuts: list[float], duration: float) -> None:
        self.duration = duration
        self.n = max(1, int(math.ceil(duration / STEP)))
        n = self.n
        self.cuts = self._count(cuts)
        self.major = self._count(ev.major_change_times)
        self.text = self._count([t.start for t in ev.text_events if t.kind in TEXT_LIKE])
        self.caption = np.zeros(n)
        for t in ev.text_events:
            if t.kind == "CAPTION":
                self._cover(self.caption, t.start, t.end)
        self.silence = np.zeros(n)
        for a, b in ev.silences:
            self._cover(self.silence, a, b)
        self.motion, self.has_motion = self._mean([(float(t), float(v)) for t, v in ev.motion_series])
        series = list(ev.audio_series)
        self.loud, self.has_audio = self._mean([(float(r[0]), float(r[1])) for r in series])
        self.voice, _ = self._mean([(float(r[0]), float(r[2])) for r in series])
        self.music, _ = self._mean([(float(r[0]), float(r[3])) for r in series])
        raw = {"cuts": self.cuts, "text": self.text, "caption": self.caption, "motion": self.motion, "loud": self.loud, "voice": self.voice, "music": self.music,
               "silence": self.silence}
        self.prefix = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in raw.items()}
        self.prefix["major"] = np.concatenate([[0.0], np.cumsum(self.major)])

    # -- construction helpers
    def _bin(self, t: float) -> int:
        return int(min(self.n - 1, max(0, math.floor(t / STEP))))

    def _count(self, times) -> np.ndarray:
        out = np.zeros(self.n)
        for t in times:
            if 0.0 <= float(t) <= self.duration + 1e-6:
                out[self._bin(float(t))] += 1.0
        return out

    def _cover(self, arr: np.ndarray, start: float, end: float) -> None:
        a, b = max(0.0, float(start)), min(self.duration, float(end))
        if b <= a:
            return
        for i in range(self._bin(a), self._bin(b - 1e-9) + 1):
            lo, hi = max(a, i * STEP), min(b, (i + 1) * STEP)
            arr[i] = min(1.0, arr[i] + max(0.0, hi - lo) / STEP)

    def _mean(self, samples: list[tuple[float, float]]) -> tuple[np.ndarray, bool]:
        total, cnt = np.zeros(self.n), np.zeros(self.n)
        for t, v in samples:
            if 0.0 <= t <= self.duration + 1e-6:
                i = self._bin(t)
                total[i] += v
                cnt[i] += 1.0
        out, last = np.zeros(self.n), 0.0  # a bin without a sample keeps the previous value (the series are coarse); before the first sample: 0
        for i in range(self.n):
            if cnt[i] > 0:
                last = total[i] / cnt[i]
            out[i] = last
        return np.clip(out, 0.0, 1.0), bool(cnt.any())

    # -- summaries over [a, b) in seconds (bin edges)
    def _prefix_at(self, key: str, t: float) -> float:
        """Cumulative sum up to time ``t``; a partly covered bin counts proportionally (observations are spread evenly inside their bin)."""
        pos = max(0.0, min(float(self.n), t / STEP))
        i = int(pos)
        p = self.prefix[key]
        return float(p[i] + (pos - i) * (p[i + 1] - p[i])) if i < self.n else float(p[self.n])

    def span(self, key: str, a: float, b: float) -> float:
        return self._prefix_at(key, b) - self._prefix_at(key, a) if b > a else 0.0

    def mean(self, key: str, a: float, b: float) -> float:
        width = max(0.0, min(self.duration, b) - max(0.0, a)) / STEP
        return self.span(key, a, b) / width if width > 0 else 0.0

    def rate_per_min(self, key: str, a: float, b: float) -> float:
        width = max(0.0, min(self.duration, b) - max(0.0, a))
        return self.span(key, a, b) / (width / 60.0) if width > 0 else 0.0

    def levels(self, a: float, b: float) -> np.ndarray:
        """The eight 0..1 levels of a stretch (order = LEVELS)."""
        return np.array([
            scale(self.rate_per_min("cuts", a, b), PACING_POINTS) / 100.0, scale(self.rate_per_min("text", a, b), TEXT_POINTS) / 100.0, self.mean("caption", a, b),
            self.mean("motion", a, b), self.mean("loud", a, b), self.mean("voice", a, b), self.mean("music", a, b), self.mean("silence", a, b)])

    def side_levels(self, b: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Levels of the ``w`` bins just before and just after bin edge ``b``. Counted events (cuts, text overlays) are noisy in a short stretch - one cut more
        or less moves a rate a lot - so the difference between the two sides is reduced by one standard deviation of its Poisson count noise first; a regular
        run of cuts never looks like a change, a real change of rate still does."""
        left, right = self.levels((b - w) * STEP, b * STEP), self.levels(b * STEP, (b + w) * STEP)
        for i, (key, points) in enumerate((("cuts", PACING_POINTS), ("text", TEXT_POINTS))):
            nl, nr = self.span(key, (b - w) * STEP, b * STEP), self.span(key, b * STEP, (b + w) * STEP)
            rl, rr = nl / (w * STEP), nr / (w * STEP)  # per second
            sd = math.sqrt(nl + nr + 1e-9) / (w * STEP)
            diff = rr - rl
            shrunk = math.copysign(max(0.0, abs(diff) - sd), diff)
            mid = 0.5 * (rl + rr)
            left[i] = scale((mid - shrunk / 2.0) * 60.0, points) / 100.0
            right[i] = scale((mid + shrunk / 2.0) * 60.0, points) / 100.0
        return left, right

    def energy(self, a: float, b: float) -> float:
        """A 0..1 composite of how busy a stretch is: cuts, motion, loudness and on-screen text."""
        lv = dict(zip(LEVELS, self.levels(a, b)))
        return float(0.40 * lv["cuts"] + 0.20 * lv["motion"] + 0.25 * lv["loud"] + 0.15 * lv["text"])


# ---------------------------------------------------------------------------------------------- the analyzer
class StructureAnalyzer:
    def analyze(self, events: ReferenceEvents, duration: float, shots: list[Shot] | None = None) -> StructureResult:
        duration = float(duration)
        if duration <= 0:
            return StructureResult(notes=["The video has no duration, so no structure was read."])
        cuts = sorted(float(t) for t in events.cut_times) or sorted(s.start for s in (shots or [])[1:])
        grid = _Grid(events, cuts, duration)
        notes: list[str] = []
        boundaries, strengths = self._boundaries(grid, events, cuts, notes)
        edges = [0.0, *boundaries, duration]
        sections = [self._section(grid, cuts, shots, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        cta_start = self._label(sections, strengths, grid, events, cuts, duration)
        hook = self._hook(grid, events, cuts, duration)
        curve = self._pacing_curve(grid, cuts, shots, duration)
        transitions = self._section_transitions(grid, events, cuts, boundaries)
        arc, shape = self._energy_arc(grid, duration)
        conf = self._confidence(grid, events, sections, duration)
        notes.append("Section names are inferred from changes in pacing, motion, audio and on-screen text events; they are estimates, not a reading of the script.")
        if not grid.has_audio:
            notes.append("No audio observations: boundaries and the closing section use picture changes only.")
        if duration < 15:
            notes.append("The video is short; its structure is only a rough reading.")
        return StructureResult(sections, curve, hook, transitions, conf, notes, arc, shape, [round(b, 2) for b in boundaries], [round(s, 3) for s in strengths],
                               round(sections[0].end, 2), round(sections[-1].start, 2), cta_start)

    # ------------------------------------------------------------------ boundaries
    def _boundaries(self, grid: _Grid, ev: ReferenceEvents, cuts: list[float], notes: list[str]) -> tuple[list[float], list[float]]:
        d, n = grid.duration, grid.n
        if d < MIN_ANALYSIS_SECONDS or n < 2 * MIN_SIDE_BINS:
            return [], []
        mean_gap = d / (len(cuts) + 1)
        half = int(clamp(max(round(d / 12.0), math.ceil(3.0 * mean_gap)), 4, 20))  # a side holds a few shots, so one more or less cut is not a "change"
        novelty = np.zeros(n + 1)
        side = max(MIN_SIDE_BINS, int(math.ceil(0.6 * half)))  # both stretches must be this long, or a few cuts decide the reading
        weights = np.array([LEVEL_WEIGHTS[k] for k in LEVELS])
        for b in range(side, n - side + 1):
            left, right = grid.side_levels(b, min(half, b, n - b))
            novelty[b] = math.sqrt(float((weights * (right - left) ** 2).sum()))
        base = novelty.copy()
        for b in range(side, n - side + 1):
            if base[b] >= CUE_MIN_BASE:
                novelty[b] += self._cue_bonus(ev, b * STEP)
        valid = novelty[side:n - side + 1]
        if valid.size == 0:
            return [], []
        thr = max(NOVELTY_FLOOR, NOVELTY_RATIO * float(np.median(valid)))
        radius = max(2, half // 2)
        min_sep = max(6.0, d / 15.0)
        min_edge = max(4.0, d / 20.0)
        limit = min(MAX_SECTIONS - 1, int(d // 8))
        cands = []
        for b in range(side, n - side + 1):
            lo, hi = max(side, b - radius), min(n - side, b + radius)
            if novelty[b] >= thr and novelty[b] >= float(novelty[lo:hi + 1].max()) - 1e-12:
                cands.append(b)
        cands.sort(key=lambda b: (-novelty[b], b))  # strongest first; ties by time (deterministic)
        picked: list[int] = []
        for b in cands:
            t = b * STEP
            if t < min_edge or d - t < min_edge or any(abs(t - p * STEP) < min_sep for p in picked) or len(picked) >= limit:
                continue
            picked.append(b)
        picked.sort()
        times, strengths = [], []
        for b in picked:
            t = float(b * STEP)
            near = [c for c in cuts if abs(c - t) <= SNAP_RADIUS]
            if near:
                t = min(near, key=lambda c: (abs(c - t), c))
            if times and t - times[-1] < min_sep * 0.5:
                continue
            times.append(round(t, 3))
            strengths.append(float(novelty[b]))
        if times:
            notes.append(f"{len(times) + 1} sections found from changes in pacing, audio and on-screen text.")
        else:
            notes.append("No clear structural change was found: the video is treated as one section.")
        return times, strengths

    @staticmethod
    def _cue_bonus(ev: ReferenceEvents, t: float) -> float:
        bonus = 0.0
        if any(min(b, t + CUE_RADIUS) - max(a, t - CUE_RADIUS) >= 0.8 for a, b in ev.silences):
            bonus += CUE_GAIN["pause"]
        if any(kind != "CUT" and abs(float(tt) - t) <= CUE_RADIUS for tt, kind, _dur in ev.transitions):
            bonus += CUE_GAIN["transition"]
        if any(x.kind in CARD_KINDS and t - 1.0 <= x.start <= t + 2.5 for x in ev.text_events):
            bonus += CUE_GAIN["text_card"]
        if any(abs(float(m) - t) <= CUE_RADIUS for m in ev.music_change_times):
            bonus += CUE_GAIN["music"]
        return bonus

    # ------------------------------------------------------------------ sections
    @staticmethod
    def _section(grid: _Grid, cuts: list[float], shots: list[Shot] | None, a: float, b: float) -> Section:
        inside = [c for c in cuts if a <= c < b]
        dur = max(1e-6, b - a)
        if shots:
            durs = [s.duration for s in shots if a <= (s.start + s.end) / 2.0 < b]
            avg = float(np.mean(durs)) if durs else dur / (len(inside) + 1)
        else:
            avg = dur / (len(inside) + 1)
        return Section(round(a, 3), round(b, 3), "EXPLANATION", 0.3, len(inside) / (dur / 60.0), grid.rate_per_min("text", a, b), grid.mean("motion", a, b),
                       grid.mean("loud", a, b), avg)

    def _label(self, sections: list[Section], strengths: list[float], grid: _Grid, ev: ReferenceEvents, cuts: list[float], d: float) -> float | None:
        """Give every section a kind and a confidence. Returns the start of the CTA section when one was recognised."""
        n = len(sections)
        energies = [grid.energy(s.start, s.end) for s in sections]
        overall_text = grid.rate_per_min("text", 0.0, d)
        overall_cpm = len(cuts) / (d / 60.0) if d > 0 else 0.0
        conf_edge = [0.0, *[clamp(s / (2 * max(NOVELTY_FLOOR, 1e-6)), 0.0, 1.0) for s in strengths]]  # strength of the boundary a section starts at
        if n == 1:
            sections[0].kind, sections[0].confidence = "EXPLANATION", 0.25
            return None
        cta_start: float | None = None
        first, last = sections[0], sections[-1]
        first.kind = "HOOK" if first.duration <= max(25.0, 0.25 * d) else "CONTEXT"
        first.confidence = round(clamp(0.35 + 0.25 * conf_edge[1], 0.2, 0.7), 2)
        # the closing section
        last_len = last.duration
        short = last_len <= max(15.0, 0.18 * d)
        score = 0
        if short:
            score += int(grid.rate_per_min("text", last.start, d) >= max(3.0, 1.3 * overall_text))
            rest_voice, last_voice = grid.mean("voice", 0.0, last.start), grid.mean("voice", last.start, d)
            score += int(grid.has_audio and rest_voice > 0.15 and last_voice < 0.6 * rest_voice)
            rest_music, last_music = grid.mean("music", 0.0, last.start), grid.mean("music", last.start, d)
            score += int(grid.has_audio and last_music > 0.15 and last_music >= 1.2 * rest_music)
        if short and score >= 2:
            last.kind, cta_start = "CTA", last.start
        else:
            last.kind = "CONCLUSION"
        last.confidence = round(clamp(0.3 + 0.2 * conf_edge[n - 1] + (0.1 if short else 0.0) + (0.05 * score if short else 0.0), 0.2, 0.7), 2)
        if n >= 3 and last.kind == "CTA":
            prev = sections[n - 2]
            if prev.kind == "EXPLANATION":
                prev.kind = "CONCLUSION"
        # the interior
        idx = list(range(1, n - 1))
        ctx_energy = energies[1] if n >= 3 and first.kind == "HOOK" else energies[0]
        if first.kind == "HOOK" and idx:
            sections[idx[0]].kind = "CONTEXT"
            idx = idx[1:]
        if idx:
            med = float(np.median([energies[i] for i in idx]))
            peak = max(idx, key=lambda i: (energies[i], -i))
            rest = [energies[i] for i in range(n) if i != peak]
            if energies[peak] >= 0.4 and energies[peak] >= 1.3 * float(np.median(rest)):
                sections[peak].kind = "REVEAL"  # the one interior section that is clearly the most energetic part of the video
            mean_motion = float(np.mean([sections[i].motion for i in idx]))
            problem_done = False
            for i in idx:
                s = sections[i]
                if s.kind == "REVEAL":
                    continue
                slow = overall_cpm > 0 and s.cuts_per_minute <= 0.8 * overall_cpm
                texty = s.text_per_minute >= max(2.0, overall_text)
                if slow and texty:
                    s.kind = "EVIDENCE"
                elif (mean_motion > 0.05 and s.motion >= 1.2 * mean_motion) or (overall_cpm > 0 and s.cuts_per_minute >= 1.15 * overall_cpm and energies[i] > med):
                    s.kind = "EXAMPLES"
                elif not problem_done and energies[i] > 1.1 * ctx_energy and i == idx[0]:
                    s.kind, problem_done = "PROBLEM", True
                else:
                    s.kind = "EXPLANATION"
        for i in range(1, n - 1):
            sections[i].confidence = round(clamp(0.25 + 0.2 * conf_edge[i], 0.2, 0.55), 2)  # an interior name is the weakest guess
        return cta_start

    # ------------------------------------------------------------------ pacing curve
    def _pacing_curve(self, grid: _Grid, cuts: list[float], shots: list[Shot] | None, d: float) -> list[PacingSegment]:
        n = grid.n
        half = clamp(d / 20.0, 4.0, 8.0)
        labels = []
        for i in range(n):
            mid = (i + 0.5) * STEP
            a, b = max(0.0, mid - half), min(d, mid + half)
            labels.append(pacing_class(grid.rate_per_min("cuts", a, b)))
        # run-length segments over the grid
        segs: list[list] = []
        for i, lab in enumerate(labels):
            if segs and segs[-1][2] == lab:
                segs[-1][1] = (i + 1) * STEP
            else:
                segs.append([i * STEP, (i + 1) * STEP, lab])
        segs[-1][1] = d
        min_len = max(8.0, d / 10.0)

        def cpm(s) -> float:
            return grid.rate_per_min("cuts", s[0], s[1])

        changed = True
        while changed and len(segs) > 1:
            changed = False
            for i, s in enumerate(segs):
                if s[1] - s[0] < min_len:
                    cand = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
                    j = min(cand, key=lambda k: (abs(cpm(segs[k]) - cpm(s)), k))
                    lo, hi = min(i, j), max(i, j)
                    merged = [segs[lo][0], segs[hi][1], pacing_class(grid.rate_per_min("cuts", segs[lo][0], segs[hi][1]))]
                    segs[lo:hi + 1] = [merged]
                    changed = True
                    break
            if not changed:
                for i in range(len(segs) - 1):  # neighbours that ended up with the same class are one segment
                    if segs[i][2] == segs[i + 1][2]:
                        segs[i:i + 2] = [[segs[i][0], segs[i + 1][1], segs[i][2]]]
                        changed = True
                        break
        out = []
        for a, b, _ in segs:
            inside = [c for c in cuts if a <= c < b]
            rate = grid.rate_per_min("cuts", a, b)
            if shots:
                durs = [s.duration for s in shots if a <= (s.start + s.end) / 2.0 < b]
                avg = float(np.mean(durs)) if durs else (b - a) / (len(inside) + 1)
            else:
                avg = (b - a) / (len(inside) + 1)
            out.append(PacingSegment(round(a, 3), round(b, 3), pacing_class(rate), round(rate, 2), round(avg, 3), round(grid.mean("motion", a, b), 3), round(grid.rate_per_min("text", a, b), 2)))
        return out

    # ------------------------------------------------------------------ hook
    def _hook(self, grid: _Grid, ev: ReferenceEvents, cuts: list[float], d: float) -> HookProfile:
        windows = [w for w in HOOK_WINDOWS if w <= 0.75 * d]
        if not windows and d >= 2.0:
            windows = [int(max(1, d * 0.5))]
        overall_cpm = len(cuts) / (d / 60.0)
        loud_ref = float(np.percentile(grid.loud, 95)) if grid.has_audio and grid.loud.size else 0.0
        out = []
        for w in windows:
            out.append(self._hook_window(grid, ev, cuts, w, overall_cpm, loud_ref))
        intensity = self._hook_intensity(grid, d)
        traits = self._hook_traits(grid, ev, out, intensity, d)
        return HookProfile(out, traits, round(intensity, 3))

    @staticmethod
    def _hook_window(grid: _Grid, ev: ReferenceEvents, cuts: list[float], w: int, overall_cpm: float, loud_ref: float) -> HookWindow:
        rate = sum(1 for c in cuts if c < w) / (w / 60.0)
        text = sum(1 for t in ev.text_events if t.kind in TEXT_LIKE and t.start < w) / (w / 60.0)
        numbers = sum(1 for t in ev.text_events if _number_like(t) and t.start < w) / (w / 60.0)
        changes = (sum(1 for c in cuts if c < w) + sum(1 for c in ev.major_change_times if c < w)) / (w / 60.0)
        audio = clamp(grid.mean("loud", 0.0, w) / loud_ref, 0.0, 1.0) if loud_ref > 1e-6 else 0.0
        rel = rate / overall_cpm if overall_cpm > 0 else (1.0 if rate == 0 else 2.0)
        return HookWindow(w, round(rate, 2), round(text, 2), round(grid.mean("motion", 0.0, w), 3), round(audio, 3), round(changes, 2), round(numbers, 2), round(rel, 2))

    @staticmethod
    def _hook_intensity(grid: _Grid, d: float) -> float:
        """0..1: how much busier the opening is than the rest (cut rate, text, motion, loudness, visual changes), as a balanced excess (h - r) / (h + r)."""
        h_len = min(clamp(0.2 * d, 5.0, 15.0), 0.5 * d)
        if h_len <= 0 or d - h_len <= 0:
            return 0.0
        parts = []
        for key, weight, rate in (("cuts", 0.35, True), ("text", 0.20, True), ("motion", 0.15, False), ("loud", 0.20, False), ("major", 0.10, True)):
            h = grid.rate_per_min(key, 0.0, h_len) if rate else grid.mean(key, 0.0, h_len)
            r = grid.rate_per_min(key, h_len, d) if rate else grid.mean(key, h_len, d)
            if h + r <= 1e-9:
                continue
            parts.append((weight, (h - r) / (h + r)))
        if not parts:
            return 0.0
        mean_excess = sum(w * e for w, e in parts) / sum(w for w, _ in parts)
        return float(clamp(2.0 * mean_excess, 0.0, 1.0))

    @staticmethod
    def _hook_traits(grid: _Grid, ev: ReferenceEvents, windows: list[HookWindow], intensity: float, d: float) -> list[str]:
        if not windows:
            return []
        w = windows[min(1, len(windows) - 1)]  # the 10 s window when there is one
        overall_text = grid.rate_per_min("text", 0.0, d)
        overall_motion = grid.mean("motion", 0.0, d)
        traits = []
        if w.shot_rate >= 18 and w.relative_pacing >= 1.3:
            traits.append("Fast visual switching")
        if (w.text_density >= 4.0 and w.text_density >= 1.3 * overall_text) or w.number_emphasis > 0:
            traits.append("Strong text")
        if w.number_emphasis > 0:
            traits.append("Number or title emphasis")
        if w.motion >= 0.4 and w.motion >= 1.2 * max(overall_motion, 1e-6):
            traits.append("High motion")
        if grid.has_audio and w.audio_intensity >= 0.75 and intensity > 0.15:
            traits.append("Loud, energetic audio")
        if grid.has_audio and grid.mean("voice", 0.0, min(2.0, d)) >= 0.3:
            traits.append("Narration starts immediately")
        if intensity < 0.1:
            traits.append("Opening paced like the rest of the video")
        return traits

    # ------------------------------------------------------------------ section transitions
    def _section_transitions(self, grid: _Grid, ev: ReferenceEvents, cuts: list[float], boundaries: list[float]) -> SectionTransitionStyle:
        if not boundaries:
            return SectionTransitionStyle()
        kinds: dict[str, int] = {}
        pauses, cards, music, pacing = [], 0, 0, 0
        for b in boundaries:
            near = [(abs(float(t) - b), k) for t, k, _dur in ev.transitions if abs(float(t) - b) <= CUE_RADIUS]
            kind = min(near)[1] if near else "CUT"
            kinds[kind] = kinds.get(kind, 0) + 1
            pause = 0.0
            for a, e in ev.silences:
                pause += max(0.0, min(e, b + CUE_RADIUS) - max(a, b - CUE_RADIUS))
            pauses.append(pause)
            cards += int(any(x.kind in CARD_KINDS and b - 1.0 <= x.start <= b + 2.5 for x in ev.text_events))
            music += int(any(abs(float(m) - b) <= CUE_RADIUS for m in ev.music_change_times))
            before, after = grid.rate_per_min("cuts", max(0.0, b - 6.0), b), grid.rate_per_min("cuts", b, min(grid.duration, b + 6.0))
            hi, lo = max(before, after), min(before, after)
            pacing += int(hi - lo >= 6.0 and hi >= 1.5 * max(lo, 1e-6))
        k = len(boundaries)
        dist = {name: round(cnt / k, 3) for name, cnt in sorted(kinds.items())}
        traits = []
        if dist.get("CUT", 0.0) >= 0.7:
            traits.append("Hard cuts between sections")
        if sum(v for name, v in dist.items() if name != "CUT") >= 0.4:
            traits.append("Fades or dissolves between sections")
        pause = float(np.mean(pauses))
        if pause >= 0.5:
            traits.append("A pause before a new section")
        if cards / k >= 0.5:
            traits.append("Title or text cards introduce sections")
        if music / k >= 0.5:
            traits.append("Music changes mark sections")
        if pacing / k >= 0.5:
            traits.append("Pacing shifts at section changes")
        return SectionTransitionStyle(dist, round(pause, 3), round(cards / k, 3), round(music / k, 3), round(pacing / k, 3), traits)

    # ------------------------------------------------------------------ energy arc
    @staticmethod
    def _energy_arc(grid: _Grid, d: float) -> tuple[list[tuple[float, float]], str]:
        pts = int(clamp(math.ceil(d / 5.0), 3, 24))
        width = d / pts
        arc = [(round((i + 0.5) * width, 2), round(grid.energy(i * width, (i + 1) * width), 3)) for i in range(pts)]
        thirds = [grid.energy(0.0, d / 3.0), grid.energy(d / 3.0, 2 * d / 3.0), grid.energy(2 * d / 3.0, d)]
        a, b, c = thirds
        mean = sum(thirds) / 3.0
        if mean < 0.02:
            return arc, "flat"
        if a >= 1.2 * max(b, c) and a >= 1.2 * mean:
            shape = "front-loaded"
        elif c >= 1.2 * max(a, b):
            shape = "building"
        elif b >= 1.2 * max(a, c):
            shape = "peak in the middle"
        elif a >= 1.2 * c and b >= c:
            shape = "falling"
        elif c >= 1.2 * a and b <= c:
            shape = "building"
        else:
            shape = "flat"
        return arc, shape

    # ------------------------------------------------------------------ confidence
    @staticmethod
    def _confidence(grid: _Grid, ev: ReferenceEvents, sections: list[Section], d: float) -> float:
        """Structure is a heuristic over other detectors' outputs: it is never reported with high confidence."""
        conf = 0.35
        conf += 0.10 if grid.has_audio else 0.0
        conf += 0.05 if grid.has_motion else 0.0
        conf += 0.05 if ev.text_events else 0.0
        conf += 0.10 if d >= 30 else 0.0
        conf += 0.10 if len(sections) >= 3 else 0.0
        if d < 15:
            conf = min(conf, 0.4)
        if len(sections) == 1:
            conf = min(conf, 0.4)
        return round(clamp(conf, 0.0, 0.75), 3)


def _number_like(t: TextEvent) -> bool:
    """A large centred overlay (a figure or a title card): counted as 'number emphasis' in the opening."""
    if t.kind in ("CAPTION", "GRAPHIC"):
        return False
    return t.kind in CARD_KINDS or (t.position == "center" and t.relative_height >= 0.06)
