"""Captions and other on-screen text/graphics measured WITHOUT OCR: where text-like overlays sit, how big they are, how long they stay, how often they
come, whether they animate, sit on a box or carry a highlighted word. Only geometry, timing and counts are produced; no word, glyph, crop or frame is
ever read, stored or logged (the golden rule of the reference engine: learn the editing *principles*, never the content).

Pipeline (numpy only)
1. ``text_regions.TextRegionDetector`` finds text-like lines in every sampled RGB frame (dense clusters of thin high-contrast strokes laid out in a band).
2. Lines are linked over time into tracks (same place, same height), split where the line's stroke profile changes (a new caption in the same band) and
   grouped into events (lines that appear and vanish together form one multi-line event).
3. **Overlay evidence.** A band of strokes proves nothing by itself: a brick wall, a window blind or a printed page in a *static* scene looks just like
   text. An event is believed only if it behaves like an overlay: (a) it APPEARS (or DISAPPEARS) while the rest of the picture stays put and its stroke
   energy jumps, (b) a new text replaces the old one in place over a steady picture, or (c) it PERSISTS in the same place, with the same strokes, while the
   rest of the picture changes (a cut, a pan, motion under a burned-in caption). Static text-like regions with none of these are dropped, and say so in
   the notes. Events that sit over the whole video (logo / watermark-like) are not editing events and are left out of the rates.
4. Events are classified from geometry and timing only (``classify``): CAPTION | HEADLINE | NUMBER_CARD | LOWER_THIRD | TEXT | GRAPHIC.
5. Per-event measurements are folded into ``CaptionStats`` / ``TextStats`` with honest confidences.

Classification rules (all fractions are of the frame; a *line height* is the height of one text line)
* position: ``top`` centre-y < 0.34; ``bottom`` >= 0.62; ``center`` between; ``lower_left`` = lower half, box starts in the left quarter, centre left of 0.45.
* LOWER_THIRD: lower_left, line height <= 0.075 and sitting on a background box.
* NUMBER_CARD: centred, very large (line height >= 0.14), narrow (width <= 0.6), one line, brief (<= 4 s). **The digits are never read**: this is a size /
  shape heuristic, low confidence, and a large single title word can be mistaken for it.
* CAPTION: events at (nearly) the same vertical position (a *band*) that repeat (>= 3 events; 2 are accepted for a centred bottom band of 1-2 medium lines),
  centred, 1-3 lines, line height 0.025..0.16. A top band needs >= 4 events (otherwise it is a series of headlines).
* HEADLINE: upper part (centre-y < 0.42), line height >= 0.045, not part of a caption band.
* TEXT: any other overlay text.
* GRAPHIC: a flat filled rectangle (bar, plate, box) that pops in over a *steady* picture without text on it. Rectangles over moving footage, translucent
  highlights that keep the picture's texture and graphics that fade slowly are NOT found: ``graphic_events_per_minute`` is a lower bound.

Caption style class (``style_class``), first matching rule wins; h = median line height, words = estimated words per caption, cpm = captions per minute
* High-Impact: h >= 0.085 and (centre position, or >= 2 of: emphasis rate >= 0.25, animation rate >= 0.4, words <= 3)
* Social: words <= 4.5 and cpm >= 12 and (animation rate >= 0.3, emphasis rate >= 0.2 or h >= 0.06)
* Bold: h >= 0.07
* News: captions sit on a background box and are at the bottom
* Subtitle-focused: bottom, coverage >= 0.45 and words >= 5 (near-continuous plain transcription)
* Documentary: bottom/centre, no box, h in 0.04..0.075, little animation / emphasis, coverage < 0.45, words >= 4 (sparse, restrained)
* Minimal: everything else (small, sparse or few captions)
Traits: large_text (h >= 0.08), high_contrast (strokes have a very strong edge contrast), frequent_highlighting (emphasis rate >= 0.3), short_caption_segments
(words <= 4), center_position, bottom_position. No typography (font, colour, shape) is recorded.

Limits (stated plainly): without OCR, words per caption and characters per line are ESTIMATES from the width/height ratio of the text lines (about 2.1
characters per unit of width/height, 5.6 characters per word) and are off by +-30 % with other fonts; text smaller than ~3 % of the frame height, text with
very low contrast, translucent or heavily animated text, and captions over strongly textured, moving footage are missed or unreliable (confidence drops
accordingly); emphasis needs bright text on a darker outline / backing; has_box needs a box that differs in brightness from the picture around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from app.reference.signals import AnalysisCancelled, FrameSampler, ReferenceAnalysisError, SampledFrame
from app.reference.style_model import CaptionStats, Shot, TextEvent, TextStats, clamp
from app.reference.text_regions import FrameObs, LineObs, TextRegionDetector, _ramp, components

# ---------------------------------------------------------------------------------------------- vocabulary (the only strings a result may contain)
CAPTION_STYLES = ("Minimal", "Bold", "News", "Documentary", "Social", "High-Impact", "Subtitle-focused")
CAPTION_TRAITS = ("large_text", "high_contrast", "frequent_highlighting", "short_caption_segments", "center_position", "bottom_position")
EVENT_KINDS = ("CAPTION", "HEADLINE", "NUMBER_CARD", "LOWER_THIRD", "TEXT", "GRAPHIC")
EVENT_POSITIONS = ("top", "center", "bottom", "lower_left")
CAPTION_POSITIONS = ("bottom", "center", "top", "mixed")
FIXED_VOCABULARY = frozenset(CAPTION_STYLES + CAPTION_TRAITS + EVENT_KINDS + EVENT_POSITIONS + CAPTION_POSITIONS)

# ---------------------------------------------------------------------------------------------- tunables
MIN_FRAMES = 8  # fewer samples than this cannot show overlay behaviour
MIN_SIZE = (96, 64)  # (width, height) below which text cannot be measured
TRACK_SCORE = 0.40  # a line must be at least this text-like to join a track
LINK_GAP = 2  # a track survives this many samples without a detection (flicker, a fade passing the threshold)
SAME_CORR = 0.75  # profile correlation from which a line counts as *pixel-identical* text (a burned-in overlay is; a structure that merely persists in a zoom is not)
SPLIT_CORR = 0.60  # stroke-profile correlation below which two consecutive observations are different text
EVIDENCE_OK = 0.50  # overlay evidence needed to believe an event
BG_STEP = 0.015  # mean luma change of the surroundings that counts as "the picture changed" (0..1), tolerance below it
BG_FULL = 0.10  # ... and the change that counts as fully convincing
MIN_EVENT_SECONDS = 0.45
MIN_CONTRAST = 0.33  # an event's strokes must have at least this edge contrast (the picture's own shapes are weaker; very low-contrast text is therefore missed)
MIN_DETECTED_SHARE = 0.70  # share of an event's samples in which its line was actually found (a flickering marginal edge is not an overlay)
PERSISTENT_SHARE = 0.85  # an overlay present for more than this share of a video (>= 8 s) is a logo / watermark, not an editing event
CHARS_PER_ASPECT = 2.1  # characters per unit of (line width / line height): an estimate, font dependent
CHARS_PER_WORD = 5.6
TOP_Y, BOTTOM_Y = 0.34, 0.62
GFX_STEP = 0.03  # block-brightness change (0..1) that counts as part of a pop-in rectangle
GFX_UNIFORM = 0.012  # block-brightness spread inside a solid rectangle
GFX_FLAT = 0.05  # mean stroke energy (0..1) inside a graphic: above it the rectangle has something written / drawn on it


# ---------------------------------------------------------------------------------------------- result
@dataclass
class CaptionTextResult:
    captions: CaptionStats = field(default_factory=CaptionStats)
    text: TextStats = field(default_factory=TextStats)
    events: list[TextEvent] = field(default_factory=list)  # geometry / timing only; analysis-cache data, never part of the style profile
    caption_confidence: float = 0.0
    text_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)  # fixed phrases (no per-video content)


def position_of(cx: float, cy: float, x0: float, w: float) -> str:
    """Where an event sits (all arguments as fractions of the frame): top | center | bottom | lower_left."""
    if cy >= 0.5 and x0 <= 0.25 and cx < 0.45 and w <= 0.55:
        return "lower_left"
    return "top" if cy < TOP_Y else "bottom" if cy >= BOTTOM_Y else "center"


# ---------------------------------------------------------------------------------------------- internal records
@dataclass
class _Track:
    frames: list[int]
    lines: list[LineObs]


@dataclass
class _Seg:
    """A stretch of one track that shows one text (the same stroke profile)."""

    frames: list[int]
    lines: list[LineObs]
    intro: bool = False  # the first sample was a smaller / weaker version of the rest (a pop or fade-in)
    replaces: bool = False  # it follows another text in the same place with no gap
    track: int = 0

    @property
    def a(self) -> int:
        return self.frames[0]

    @property
    def b(self) -> int:
        return self.frames[-1]


@dataclass
class _Ev:
    """One candidate event (one or more lines that appear and vanish together) with everything the classifier needs."""

    a: int
    b: int
    box: tuple[int, int, int, int]  # union box in pixels
    line_h: float  # px, median line height
    line_ws: list[float]
    rows: int
    score: float
    contrast: float
    box_flag: bool
    emphasized: bool
    animated: bool
    replaces: bool
    busy: float
    segs: list[_Seg]
    evidence: float = 0.0
    confidence: float = 0.5
    kind: str = "TEXT"
    position: str = "bottom"
    cx: float = 0.5
    cy: float = 0.5
    rel_h: float = 0.0
    rel_w: float = 0.0
    start: float = 0.0
    end: float = 0.0


# ---------------------------------------------------------------------------------------------- helpers
def _profile_corr(p: LineObs, q: LineObs) -> float:
    """Do two lines carry the same text? Correlation of their column-edge profiles over the columns both cover, with the mean removed (lightly smoothed: a pixel
    of jitter is not a change). Removing the mean matters: every dense line of text has a "box-shaped" profile, so two different captions of similar length would
    correlate highly if the box shape were left in; what is compared is the pattern *inside* the line. ~1 for the same text, ~0 +- 0.3 for different text."""
    lo, hi = max(p.x0, q.x0), min(p.x1, q.x1)
    if hi - lo < 24:
        return 0.0
    a = p.profile[lo - p.x0:hi - p.x0].astype(np.float32)
    b = q.profile[lo - q.x0:hi - q.x0].astype(np.float32)
    k = np.ones(3, dtype=np.float32) / 3.0
    a, b = np.convolve(a, k, mode="same"), np.convolve(b, k, mode="same")
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


def _position_similarity(a: LineObs, b: LineObs) -> float:
    """How much two lines (in different frames) look like the same place: 0 = no, up to 1."""
    ix = min(a.x1, b.x1) - max(a.x0, b.x0)
    iy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ix <= 0 or iy <= 0:
        return 0.0
    vo, ho = iy / min(a.h, b.h), ix / min(a.w, b.w)
    hr = min(a.h, b.h) / max(a.h, b.h)
    if vo < 0.55 or ho < 0.4 or hr < 0.6:
        return 0.0
    return 0.5 * (vo + ho) * hr


def _link(per_frame: list[list[LineObs]]) -> list[_Track]:
    active: list[_Track] = []
    done: list[_Track] = []
    for i, lines in enumerate(per_frame):
        cand = [ln for ln in lines if ln.text_score >= TRACK_SCORE]
        still: list[_Track] = []
        for t in active:
            (still if i - t.frames[-1] <= LINK_GAP else done).append(t)
        active = still
        pairs = sorted(((_position_similarity(t.lines[-1], ln), ti, li) for ti, t in enumerate(active) for li, ln in enumerate(cand)), reverse=True)
        used_t: set[int] = set()
        used_l: set[int] = set()
        for s, ti, li in pairs:
            if s <= 0 or ti in used_t or li in used_l:
                continue
            used_t.add(ti)
            used_l.add(li)
            active[ti].frames.append(i)
            active[ti].lines.append(cand[li])
        for li, ln in enumerate(cand):
            if li not in used_l:
                active.append(_Track([i], [ln]))
    return done + active


def _split(track: _Track, tid: int) -> list[_Seg]:
    """Cut a track where the line's stroke profile changes (a new caption in the same band). A single odd sample between two matching ones (a glitch at a
    cut) is not a change. A one-sample prelude that is a smaller version of what follows is a pop-in, merged into it as an intro."""
    f, ln = track.frames, track.lines
    groups: list[list[int]] = [[0]]
    for k in range(1, len(f)):
        gap_ok = f[k] - f[k - 1] <= LINK_GAP
        same = gap_ok and _profile_corr(ln[k - 1], ln[k]) >= SPLIT_CORR
        if not same and gap_ok and k + 1 < len(f) and _profile_corr(ln[k - 1], ln[k + 1]) >= 0.65:
            same = True  # glitch: the line is back as it was
        if same:
            groups[-1].append(k)
        else:
            groups.append([k])
    segs = [_Seg([f[i] for i in g], [ln[i] for i in g], track=tid) for g in groups]
    out: list[_Seg] = []
    for i, s in enumerate(segs):
        if i > 0 and segs[i - 1].b + LINK_GAP >= s.a:
            s.replaces = True
        out.append(s)
    merged: list[_Seg] = []
    i = 0
    while i < len(out):
        s = out[i]
        if len(s.frames) == 1 and i + 1 < len(out) and len(out[i + 1].frames) >= 3 and out[i + 1].a - s.b == 1:
            nxt = out[i + 1]
            n0 = nxt.lines[0]
            if s.lines[0].w * s.lines[0].h <= 0.8 * n0.w * n0.h and abs(s.lines[0].cx - n0.cx) <= 0.15 * n0.w and abs(s.lines[0].cy - n0.cy) <= 0.5 * n0.h:
                nxt.frames = s.frames + nxt.frames
                nxt.lines = s.lines + nxt.lines
                nxt.intro = True
                nxt.replaces = s.replaces
                i += 1
                merged.append(nxt)
                i += 1
                continue
        merged.append(s)
        i += 1
    return merged


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ---------------------------------------------------------------------------------------------- the analyzer
class CaptionTextAnalyzer:
    """Measures the on-screen text of a video from RGB frames sampled at ``fps`` (``width`` x ``height`` px). See the module doc for what it can and cannot do."""

    def __init__(self, fps: float = 4.0, width: int = 480, height: int = 270) -> None:
        self.fps = float(fps) if fps and fps > 0 else 4.0
        self.width = int(width)
        self.height = int(height)

    # ------------------------------------------------------------------ entry points
    def analyze_video(self, sampler: FrameSampler, path: Path, duration: float, shots: list[Shot] | None = None, progress: Callable[[float], None] | None = None,
                      cancel=None) -> CaptionTextResult:
        try:
            frames = sampler.frames(path, self.fps, self.width, self.height, gray=False, cancel=cancel)
            return self.analyze_frames(frames, duration, shots, progress, cancel)
        except ReferenceAnalysisError:
            return self._unusable("The video frames could not be decoded; captions and text were not measured.")

    def analyze_frames(self, frames: Iterable[SampledFrame], duration: float, shots: list[Shot] | None = None, progress: Callable[[float], None] | None = None,
                       cancel=None) -> CaptionTextResult:
        detector: TextRegionDetector | None = None
        obs: list[FrameObs] = []
        times: list[float] = []
        skipped = 0
        expected = max(1, int(round(max(duration, 0.0) * self.fps)))
        for fr in frames:
            if cancel is not None and cancel.is_set():
                raise AnalysisCancelled()
            px = getattr(fr, "pixels", None)
            if px is None or getattr(px, "ndim", 0) not in (2, 3):
                skipped += 1
                continue
            if detector is None:
                h, w = px.shape[:2]
                if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                    return self._unusable("The frames are too small to measure on-screen text.")
                detector = TextRegionDetector(w, h)
            elif px.shape[:2] != (detector.height, detector.width):
                skipped += 1
                continue
            obs.append(detector.detect(px))
            times.append(float(fr.time))
            if progress is not None and len(obs) % 8 == 0:
                progress(min(0.95, len(obs) / expected))
        if progress is not None:
            progress(1.0)
        if detector is None or len(obs) < MIN_FRAMES:
            return self._unusable(f"Too few frames ({len(obs)}) to measure captions and text; nothing was inferred.")
        res = self._analyze(detector, obs, np.asarray(times, dtype=np.float64), float(duration), shots or [])
        if skipped:
            res.notes.append(f"{skipped} unusable frames were skipped.")
        return res

    # ------------------------------------------------------------------ result for "cannot run"
    @staticmethod
    def _unusable(why: str) -> CaptionTextResult:
        return CaptionTextResult(notes=[why, "no captions detected (the detector could not run, so this says nothing about the video)"])

    # ------------------------------------------------------------------ the analysis proper
    def _analyze(self, det: TextRegionDetector, obs: list[FrameObs], times: np.ndarray, duration: float, shots: list[Shot]) -> CaptionTextResult:
        n = len(obs)
        dt = 1.0 / self.fps
        t0 = float(times[0])
        if not duration or duration <= 0 or duration < times[-1] + dt - 1e-6:
            duration = float(times[-1] + dt)
        notes: list[str] = ["Captions and text are found from stroke geometry and timing, not read (no OCR); words per caption and characters per line are estimates."]
        L = np.stack([o.luma for o in obs]).reshape(n, -1)  # uint8 (n, blocks): brightness / stroke energy per block, converted to 0..1 only where used
        E = np.stack([o.energy for o in obs]).reshape(n, -1)
        busy_mean = float(np.mean([o.busy for o in obs]))
        flat_share = float(np.mean([o.flat for o in obs]))
        cuts = {int(round((s.start - t0) * self.fps)) for s in shots[1:]} if shots else set()

        tracks = _link([o.lines for o in obs])
        segs: list[_Seg] = []
        for tid, t in enumerate(tracks):
            segs += _split(t, tid)
        min_frames = max(2, int(round(MIN_EVENT_SECONDS * self.fps)))
        cands = [s for s in segs if len(s.frames) >= min_frames]
        short_candidates = sum(1 for s in segs if len(s.frames) < min_frames and float(np.median([ln.text_score for ln in s.lines])) >= 0.6
                               and float(np.median([ln.contrast for ln in s.lines])) >= MIN_CONTRAST)
        cands = self._dedupe(cands)
        evs = self._group(cands, det, obs)

        ctx = _Ctx(det, L, E, cuts)
        accepted: list[_Ev] = []
        rejected_static = 0
        persistent = 0
        weak = 0
        for ev in evs:
            detected = sum(len(s.frames) for s in ev.segs) / max(1, len(ev.segs)) / max(1, ev.b - ev.a + 1)
            if ev.contrast < MIN_CONTRAST or detected < MIN_DETECTED_SHARE or ev.score < 0.5:
                weak += 1
                continue
            ev.evidence = ctx.evidence(ev)
            span = (ev.b - ev.a + 1) / self.fps
            if ev.evidence < EVIDENCE_OK:
                rejected_static += 1
                continue
            if span >= PERSISTENT_SHARE * duration and duration >= 8.0:
                persistent += 1
                continue
            accepted.append(ev)
        if weak:
            notes.append(f"{weak} faint or flickering text-like region(s) were too weak to count.")
        if rejected_static:
            notes.append(f"{rejected_static} text-like region(s) never behaved like an overlay (no appearance, replacement or persistence over a changing picture) and were ignored.")
        if persistent:
            notes.append(f"{persistent} overlay(s) stayed on screen for almost the whole video (logo / watermark-like) and are not counted as editing events.")

        for ev in accepted:
            self._finish(ev, det, times, t0, dt, duration)
        graphics = self._graphics(det, L, E, accepted, times, t0, dt, duration)
        self._classify(accepted, duration)
        events = [TextEvent(e.start, e.end, e.kind, e.position, round(e.rel_h, 4), round(e.rel_w, 4), e.rows, e.animated, e.emphasized, e.box_flag, round(e.confidence, 3))
                  for e in sorted(accepted, key=lambda e: (e.start, e.cy))]
        events += graphics
        events.sort(key=lambda e: (e.start, e.end))
        captions, cap_conf = self._caption_stats(accepted, duration)
        text, text_conf = self._text_stats(accepted, graphics, duration)

        # --- confidences: how sure is the *detection* (not how much text there is)
        h = det.height
        f_res = 0.3 + 0.7 * _ramp(h, 100, 240)
        f_frames = 0.45 + 0.55 * _ramp(n, MIN_FRAMES, 48)
        f_busy = 1.0 - 0.7 * clamp(busy_mean * 1.5)
        f_flat = 1.0 - 0.8 * flat_share
        base = f_res * f_frames * f_busy * f_flat
        f_unres = 1.0 / (1.0 + 0.4 * (rejected_static + 0.5 * short_candidates + 0.1 * weak))  # unexplained look-alikes make 'nothing there' less certain
        if captions.caption_present:
            ce = [e.confidence for e in accepted if e.kind == "CAPTION"]
            cap_conf = clamp(float(np.mean(ce)) * (0.3 + 0.7 * base) * (0.85 + 0.15 * min(1.0, len(ce) / 4.0)))
        else:
            cap_conf = clamp(0.85 * base * f_unres)
        tx = [e.confidence for e in accepted if e.kind != "CAPTION"] + [g.confidence for g in graphics]
        if tx:
            text_conf = clamp(float(np.mean(tx)) * (0.3 + 0.7 * base))
        else:
            text_conf = clamp(0.8 * base * f_unres)
        if flat_share > 0.9:
            notes.append("The picture is (almost) a single flat colour: nothing could be measured.")
        if busy_mean > 0.25:
            notes.append("The picture is busy or strongly textured: text detection is less reliable and confidence is reduced.")
        if h < 160:
            notes.append("Low frame resolution: small text cannot be detected.")
        if not captions.caption_present:
            notes.append("no captions detected")
        if any(e.kind == "NUMBER_CARD" for e in accepted):
            notes.append("Number cards are a size / shape heuristic (large, centred, brief text): digits are not read and the confidence is low.")
        notes.append("Graphic overlays are only found as flat rectangles that pop in over a steady picture; graphic_events_per_minute is a lower bound.")
        return CaptionTextResult(captions, text, events, round(cap_conf, 3), round(text_conf, 3), notes)

    # ------------------------------------------------------------------ candidates -> events
    @staticmethod
    def _dedupe(segs: list[_Seg]) -> list[_Seg]:
        """Drop fragments: a segment mostly inside a bigger one that lives at the same time (the detector split one line differently in some frames)."""
        order = sorted(segs, key=lambda s: -(len(s.frames) * float(np.median([ln.w * ln.h for ln in s.lines]))))
        kept: list[_Seg] = []
        for s in order:
            bs = (min(ln.x0 for ln in s.lines), min(ln.y0 for ln in s.lines), max(ln.x1 for ln in s.lines), max(ln.y1 for ln in s.lines))
            area = max(1, (bs[2] - bs[0]) * (bs[3] - bs[1]))
            dup = False
            for k in kept:
                ts = _overlap(s.a, s.b + 1, k.a, k.b + 1) / max(1, s.b - s.a + 1)
                if ts < 0.6:
                    continue
                kb = (min(ln.x0 for ln in k.lines), min(ln.y0 for ln in k.lines), max(ln.x1 for ln in k.lines), max(ln.y1 for ln in k.lines))
                inter = _overlap(bs[0], bs[2], kb[0], kb[2]) * _overlap(bs[1], bs[3], kb[1], kb[3])
                if inter / area >= 0.6:
                    dup = True
                    break
            if not dup:
                kept.append(s)
        return sorted(kept, key=lambda s: (s.a, s.lines[0].y0))

    def _group(self, segs: list[_Seg], det: TextRegionDetector, obs: list[FrameObs]) -> list[_Ev]:
        """Lines that appear and vanish together and are stacked / side by side form one multi-line event."""
        groups: list[list[_Seg]] = []
        for s in segs:
            placed = False
            sb = self._seg_box(s)
            sh = float(np.median([ln.h for ln in s.lines]))
            for g in groups:
                if abs(g[0].a - s.a) > 2 or abs(g[0].b - s.b) > 2:
                    continue
                if any(self._adjacent(self._seg_box(m), float(np.median([ln.h for ln in m.lines])), sb, sh) for m in g):
                    g.append(s)
                    placed = True
                    break
            if not placed:
                groups.append([s])
        return [self._make_event(g, det, obs) for g in groups]

    @staticmethod
    def _seg_box(s: _Seg) -> tuple[int, int, int, int]:
        k = len(s.lines)
        sel = s.lines[min(2, k // 3):] if k >= 5 else s.lines  # the steady part (a pop / fade-in is not the line's size)
        return (int(np.median([ln.x0 for ln in sel])), int(np.median([ln.y0 for ln in sel])), int(np.median([ln.x1 for ln in sel])), int(np.median([ln.y1 for ln in sel])))

    @staticmethod
    def _adjacent(a: tuple[int, int, int, int], ah: float, b: tuple[int, int, int, int], bh: float) -> bool:
        gap_y = max(b[1] - a[3], a[1] - b[3])
        gap_x = max(b[0] - a[2], a[0] - b[2])
        hmax = max(ah, bh)
        hr = min(ah, bh) / hmax
        if hr < 0.5:
            return False
        stacked = gap_y <= 1.2 * hmax and (_overlap(a[0], a[2], b[0], b[2]) / max(1, min(a[2] - a[0], b[2] - b[0])) >= 0.25 or abs((a[0] + a[2]) - (b[0] + b[2])) <= 0.3 * max(a[2] - a[0], b[2] - b[0]))
        side = gap_y <= 0 and gap_x <= 3.0 * hmax
        return stacked or side

    def _make_event(self, g: list[_Seg], det: TextRegionDetector, obs: list[FrameObs]) -> _Ev:
        boxes = [self._seg_box(s) for s in g]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        # rows: members whose vertical ranges overlap by more than half are one row
        rows: list[list[int]] = []
        for i in sorted(range(len(g)), key=lambda i: boxes[i][1]):
            for r in rows:
                j = r[0]
                if _overlap(boxes[i][1], boxes[i][3], boxes[j][1], boxes[j][3]) > 0.5 * min(boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1]):
                    r.append(i)
                    break
            else:
                rows.append([i])
        hs = [float(b[3] - b[1]) for b in boxes]
        steady = [ln for s in g for ln in (s.lines[min(2, len(s.lines) // 3):] if len(s.lines) >= 5 else s.lines)]
        return _Ev(
            a=min(s.a for s in g), b=max(s.b for s in g), box=box, line_h=float(np.median(hs)), line_ws=[float(b[2] - b[0]) for b in boxes], rows=max(1, len(rows)),
            score=float(np.median([ln.text_score for ln in steady])), contrast=float(np.median([ln.contrast for ln in steady])),
            box_flag=float(np.median([ln.box for ln in steady])) >= 0.5,
            emphasized=float(np.mean([ln.emphasis >= 0.15 for ln in steady if ln.colour_known] or [0.0])) >= 0.4,
            animated=self._is_animated(g), replaces=any(s.replaces for s in g) and min(s.a for s in g) > 0,
            busy=float(np.mean([obs[i].busy for s in g for i in s.frames])), segs=g)

    @staticmethod
    def _is_animated(g: list[_Seg]) -> bool:
        """The first samples of the event are visibly weaker or smaller than its steady state (a fade or pop-in)."""
        for s in g:
            if s.intro:
                return True
            k = len(s.lines)
            if k < 4:
                continue
            steady = s.lines[min(3, k // 2):]
            c_s = float(np.median([ln.contrast for ln in steady]))
            e_s = float(np.median([ln.energy for ln in steady]))
            a_s = float(np.median([ln.w * ln.h for ln in steady]))
            head = s.lines[: min(3, k // 2)]
            if c_s > 0 and (min(ln.contrast for ln in head) < 0.80 * c_s or min(ln.energy for ln in head) < 0.70 * e_s or min(ln.w * ln.h for ln in head) < 0.80 * a_s):
                return True
        return False

    def _finish(self, ev: _Ev, det: TextRegionDetector, times: np.ndarray, t0: float, dt: float, duration: float) -> None:
        """Times, geometry (as fractions) and the event's own confidence."""
        ev.start = float(max(0.0, times[ev.a] - 0.5 * dt))
        ev.end = float(min(duration, times[ev.b] + 0.5 * dt))
        x0, y0, x1, y1 = ev.box
        W, H = det.width, det.height
        ev.cx, ev.cy = 0.5 * (x0 + x1) / W, 0.5 * (y0 + y1) / H
        ev.rel_h, ev.rel_w = ev.line_h / H, (x1 - x0) / W
        ev.position = position_of(ev.cx, ev.cy, x0 / W, ev.rel_w)
        frames = ev.b - ev.a + 1
        c = 0.20 + 0.35 * ev.evidence + 0.25 * ev.score + 0.10 * min(1.0, frames / 6.0) + 0.10 * _ramp(ev.rel_h, 0.025, 0.05) - 0.35 * ev.busy
        ev.confidence = float(clamp(c, 0.05, 0.95))

    # ------------------------------------------------------------------ classification
    def _classify(self, evs: list[_Ev], duration: float) -> None:
        rest: list[_Ev] = []
        for e in evs:
            if e.position == "lower_left" and e.rel_h <= 0.075 and e.box_flag:
                e.kind = "LOWER_THIRD"
            elif e.position == "center" and e.rel_h >= 0.14 and e.rel_w <= 0.6 and e.rows == 1 and (e.end - e.start) <= 4.0:
                e.kind = "NUMBER_CARD"
                e.confidence = min(e.confidence, 0.45)
            else:
                rest.append(e)
        # caption bands: events at nearly the same vertical position
        bands: list[list[_Ev]] = []
        for e in sorted(rest, key=lambda e: e.cy):
            if bands and e.cy - bands[-1][-1].cy <= 0.05:
                bands[-1].append(e)
            else:
                bands.append([e])
        for g in bands:
            if not self._is_caption_band(g):
                continue
            n = len(g)
            factor = 0.75 if n == 2 else 0.88 if n == 3 else 1.0
            for e in g:
                e.kind = "CAPTION"
                e.confidence = float(clamp(e.confidence * factor))
        for e in rest:
            if e.kind == "CAPTION":
                continue
            e.kind = "HEADLINE" if (e.cy < 0.42 and e.rel_h >= 0.045 and e.rows <= 3) else "TEXT"

    @staticmethod
    def _is_caption_band(g: list[_Ev]) -> bool:
        n = len(g)
        if n < 2:
            return False
        med_h = float(np.median([e.rel_h for e in g]))
        med_rows = float(np.median([e.rows for e in g]))
        off = float(np.median([abs(e.cx - 0.5) for e in g]))
        pos = max(set(e.position for e in g), key=[e.position for e in g].count)
        if not (0.025 <= med_h <= 0.16 and med_rows <= 3 and off <= 0.16):
            return False
        if pos == "lower_left":
            return False
        if n >= 3:
            return pos != "top" or n >= 4
        return pos in ("bottom", "center") and med_rows <= 2 and 0.03 <= med_h <= 0.12

    # ------------------------------------------------------------------ statistics
    def _caption_stats(self, accepted: list[_Ev], duration: float) -> tuple[CaptionStats, float]:
        caps = [e for e in accepted if e.kind == "CAPTION"]
        st = CaptionStats()
        if not caps:
            return st, 0.0
        minutes = max(duration, 1e-6) / 60.0
        iv = sorted((e.start, e.end) for e in caps)
        covered, cur0, cur1 = 0.0, iv[0][0], iv[0][1]
        for a, b in iv[1:]:
            if a <= cur1:
                cur1 = max(cur1, b)
            else:
                covered += cur1 - cur0
                cur0, cur1 = a, b
        covered += cur1 - cur0
        st.caption_present = True
        st.caption_coverage = round(clamp(covered / max(duration, 1e-6)), 4)
        st.captions_per_minute = round(len(caps) / minutes, 3)
        chars = [sum(CHARS_PER_ASPECT * w / max(1.0, e.line_h) for w in e.line_ws) for e in caps]
        per_line = [CHARS_PER_ASPECT * float(np.mean([w / max(1.0, e.line_h) for w in e.line_ws])) for e in caps]
        st.average_words_per_caption = round(float(np.mean(chars)) / CHARS_PER_WORD, 2)
        st.average_chars_per_line = round(float(np.mean(per_line)), 1)
        st.caption_line_count = round(float(np.mean([e.rows for e in caps])), 2)
        pos = [e.position for e in caps]
        top = max(set(pos), key=pos.count)
        share = pos.count(top) / len(pos)
        st.caption_position = (top if top in ("bottom", "center", "top") else "bottom") if share >= 0.7 else "mixed"
        st.caption_emphasis_rate = round(float(np.mean([e.emphasized for e in caps])), 3)
        st.caption_animation_rate = round(float(np.mean([e.animated for e in caps])), 3)
        st.relative_text_height = round(float(np.median([e.rel_h for e in caps])), 4)
        st.has_background_box = float(np.mean([e.box_flag for e in caps])) >= 0.5
        contrast = float(np.median([e.contrast for e in caps]))
        st.style_class = self.style_class(st)
        st.traits = self.traits(st, contrast)
        return st, 1.0

    @staticmethod
    def style_class(st: CaptionStats) -> str:
        """The documented rule list (module doc), first match wins."""
        h, words, cpm, cov = st.relative_text_height, st.average_words_per_caption, st.captions_per_minute, st.caption_coverage
        anim, emph, pos = st.caption_animation_rate, st.caption_emphasis_rate, st.caption_position
        if h >= 0.085 and (pos == "center" or sum([emph >= 0.25, anim >= 0.4, words <= 3.0]) >= 2):
            return "High-Impact"
        if words <= 4.5 and cpm >= 12 and (anim >= 0.3 or emph >= 0.2 or h >= 0.06):
            return "Social"
        if h >= 0.07:
            return "Bold"
        if st.has_background_box and pos == "bottom":
            return "News"
        if pos == "bottom" and cov >= 0.45 and words >= 5:
            return "Subtitle-focused"
        if pos in ("bottom", "center") and not st.has_background_box and 0.04 <= h <= 0.075 and anim < 0.3 and emph < 0.2 and cov < 0.45 and words >= 4:
            return "Documentary"
        return "Minimal"

    @staticmethod
    def traits(st: CaptionStats, contrast: float) -> list[str]:
        t = []
        if st.relative_text_height >= 0.08:
            t.append("large_text")
        if contrast >= 0.55:
            t.append("high_contrast")
        if st.caption_emphasis_rate >= 0.3:
            t.append("frequent_highlighting")
        if 0 < st.average_words_per_caption <= 4.0:
            t.append("short_caption_segments")
        if st.caption_position == "center":
            t.append("center_position")
        if st.caption_position == "bottom":
            t.append("bottom_position")
        return t

    def _text_stats(self, accepted: list[_Ev], graphics: list[TextEvent], duration: float) -> tuple[TextStats, float]:
        txt = [e for e in accepted if e.kind != "CAPTION"]
        st = TextStats()
        minutes = max(duration, 1e-6) / 60.0
        st.graphic_events_per_minute = round(len(graphics) / minutes, 3)
        if not txt:
            return st, 0.0
        st.text_events_per_minute = round(len(txt) / minutes, 3)
        st.average_duration = round(float(np.mean([e.end - e.start for e in txt])), 3)
        st.headline_frequency = round(sum(e.kind == "HEADLINE" for e in txt) / minutes, 3)
        st.number_graphic_frequency = round(sum(e.kind == "NUMBER_CARD" for e in txt) / minutes, 3)
        st.lower_third_frequency = round(sum(e.kind == "LOWER_THIRD" for e in txt) / minutes, 3)
        st.average_relative_size = round(float(np.mean([e.rel_h for e in txt])), 4)
        st.position_share = {p: round(sum(e.position == p for e in txt) / len(txt), 3) for p in EVENT_POSITIONS}
        st.animation_rate = round(float(np.mean([e.animated for e in txt])), 3)
        return st, 1.0

    # ------------------------------------------------------------------ graphics (flat rectangles that pop in over a steady picture)
    def _graphics(self, det: TextRegionDetector, L: np.ndarray, E: np.ndarray, accepted: list[_Ev], times: np.ndarray, t0: float, dt: float, duration: float) -> list[TextEvent]:
        """Flat filled rectangles (bars, plates, boxes) that appear over a steady picture and later vanish, with nothing written on them.

        On the block-brightness grid: at the step where a rectangle pops in, the changed blocks form ONE compact, almost completely filled rectangle, all in the same
        direction (darker or brighter), the rest of the picture does not change, its inside is flat (no strokes: a plate with text is that text's box, not a
        graphic) and it is still there one sample later. It ends when its inside returns to what it was before (or the picture changes). Rectangles over moving
        footage, translucent highlights that keep the picture's texture, and slow fades are not found."""
        n, gh, gw = L.shape[0], det.gh, det.gw
        L3, E3 = L.reshape(n, gh, gw).astype(np.float32) / 255.0, E.reshape(n, gh, gw).astype(np.float32) / 255.0
        bs = det.bs
        W, H = det.width, det.height
        out: list[TextEvent] = []
        total = gh * gw
        t = 1
        while t < n - 1:
            step = L3[t] - L3[t - 1]
            ch = np.abs(step) > GFX_STEP
            cnt = int(ch.sum())
            if cnt < 4 or cnt > 0.4 * total:
                t += 1
                continue
            comps = components(ch)
            y0, x0, y1, x1, area = max(comps, key=lambda c: c[4])
            box_area = (y1 - y0) * (x1 - x0)
            if area / box_area < 0.8 or box_area < 6 or box_area > 0.4 * total or min(y1 - y0, x1 - x0) < 2:
                t += 1
                continue
            inside = np.zeros_like(ch)
            inside[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = True
            if int((ch & ~inside).sum()) > max(2, 0.1 * area):  # something else changed too: not an overlay popping onto a steady picture
                t += 1
                continue
            vals = step[y0:y1, x0:x1][ch[y0:y1, x0:x1]]
            if max(float((vals > 0).mean()), float((vals < 0).mean())) < 0.85:
                t += 1
                continue
            iy0, iy1, ix0, ix1 = (y0 + 1, y1 - 1, x0 + 1, x1 - 1) if (y1 - y0 > 2 and x1 - x0 > 2) else (y0, y1, x0, x1)
            if float(E3[t, iy0:iy1, ix0:ix1].mean()) > GFX_FLAT or float(E3[t + 1, iy0:iy1, ix0:ix1].mean()) > GFX_FLAT:
                t += 1
                continue
            # an onset leaves a UNIFORM rectangle where there was a picture; an offset is the reverse (picture reappears), which must not count as a second graphic
            if float(L3[t, iy0:iy1, ix0:ix1].std()) > GFX_UNIFORM or float(L3[t - 1, iy0:iy1, ix0:ix1].std()) <= GFX_UNIFORM:
                t += 1
                continue
            pre = L3[t - 1, y0:y1, x0:x1]
            dev = [float(np.abs(L3[k, y0:y1, x0:x1] - pre).mean()) for k in range(t, n)]
            amp = float(np.median(dev[:3]))
            if amp < GFX_STEP or dev[1] < 0.5 * amp:  # too faint, or a flash rather than a graphic
                t += 1
                continue
            end = n
            out_mask = ~np.zeros_like(ch)
            out_mask[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = False
            for k in range(t + 2, n):
                if dev[k - t] < 0.4 * amp:
                    end = k
                    break
                if float(np.abs(L3[k] - L3[k - 1])[out_mask].mean()) > 0.05:  # the whole picture changed (a cut): the plate may or may not still be there
                    end = k
                    break
            last = end - 1
            start_t = float(max(0.0, times[t] - 0.5 * dt))
            end_t = float(min(duration, times[last] + 0.5 * dt))
            span = end_t - start_t
            written = any(e.kind != "GRAPHIC" and _overlap(e.start, e.end, start_t, end_t) >= 0.5 * min(span, e.end - e.start) and e.box[0] >= (x0 - 1) * bs and e.box[2] <= (x1 + 1) * bs
                          and e.box[1] >= (y0 - 1) * bs and e.box[3] <= (y1 + 1) * bs for e in accepted)
            if not written and span >= MIN_EVENT_SECONDS and not (span >= PERSISTENT_SHARE * duration and duration >= 8.0):
                cx, cy = 0.5 * (x0 + x1) * bs / W, 0.5 * (y0 + y1) * bs / H
                rw, rh = (x1 - x0) * bs / W, (y1 - y0) * bs / H
                conf = float(clamp(0.30 + 0.25 * (area / box_area) + 0.25 * _ramp(amp, GFX_STEP, 0.15) + 0.15 * (1.0 if end < n else 0.0)))
                out.append(TextEvent(start_t, end_t, "GRAPHIC", position_of(cx, cy, x0 * bs / W, rw), round(rh, 4), round(rw, 4), 0, False, False, True, round(min(conf, 0.7), 3)))
            t = max(t + 1, end + 1)  # the step that ends it looks like a pop-in in reverse
        return out


class _Ctx:
    """Shared arrays for the overlay-evidence tests: block brightness / stroke energy per frame, the grid, and the cut samples."""

    def __init__(self, det: TextRegionDetector, L: np.ndarray, E: np.ndarray, cuts: set[int]) -> None:
        self.det, self.L, self.E, self.cuts = det, L, E, cuts
        self.n = L.shape[0]

    def masks(self, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        det = self.det
        bs, gh, gw = det.bs, det.gh, det.gw
        x0, y0, x1, y1 = box
        bx0, by0, bx1, by1 = x0 // bs, y0 // bs, -(-x1 // bs), -(-y1 // bs)
        inside = np.zeros((gh, gw), dtype=bool)
        inside[max(0, by0 - 1):min(gh, by1 + 1), max(0, bx0 - 1):min(gw, bx1 + 1)] = True
        near = np.zeros((gh, gw), dtype=bool)
        near[max(0, by0 - 3):min(gh, by1 + 3), max(0, bx0 - 3):min(gw, bx1 + 3)] = True
        return np.flatnonzero(inside.ravel()), np.flatnonzero(~near.ravel())

    def evidence(self, ev: _Ev) -> float:
        """0..1: how much this event behaves like an overlay (appears / vanishes / is replaced while the rest stays put, or stays put while the rest changes)."""
        a, b, n = ev.a, ev.b, self.n
        inside, outside = self.masks(ev.box)
        if outside.size < 20:  # the event covers (nearly) the whole frame: there is no "rest of the picture" to compare with
            return 0.0
        w0, w1 = max(0, a - 4), min(n, b + 9)  # only the event and a few samples around it are needed
        Lo = self.L[w0:w1][:, outside].astype(np.float32) / 255.0
        E_in = self.E[w0:w1][:, inside].astype(np.float32).mean(axis=1) / 255.0

        def bg(i: int, j: int) -> float:
            """How much the rest of the picture changed between samples i and j."""
            return float(np.abs(Lo[min(max(i, w0), w1 - 1) - w0] - Lo[min(max(j, w0), w1 - 1) - w0]).mean())

        def factor(ch: float) -> float:
            return 1.0 - clamp((ch - BG_STEP) / (BG_FULL * 0.45))

        e_s = float(np.median(E_in[a - w0:b - w0 + 1]))
        ev_on = ev_off = 0.0
        if a >= 1 and e_s > 1e-6:
            pre = E_in[max(0, a - 4) - w0:a - w0]
            rise = (e_s - float(pre.min())) / e_s
            ch = max(bg(a - 2, a - 1), bg(a - 1, a), bg(a - 1, a + 1))
            ev_on = clamp((rise - 0.3) / 0.4) * factor(ch)
            if a in self.cuts or (a - 1) in self.cuts:
                ev_on *= 0.5
            if ev.replaces:
                ev_on = max(ev_on, 0.8 * factor(ch))
        if b <= n - 2 and e_s > 1e-6:
            post = E_in[b + 1 - w0:min(n, b + 5) - w0]
            drop = (e_s - float(post.min())) / e_s
            ch = max(bg(b, b + 1), bg(b - 1, b + 1), bg(b + 1, b + 2))
            ev_off = clamp((drop - 0.3) / 0.4) * factor(ch)
            if (b + 1) in self.cuts or b in self.cuts:
                ev_off *= 0.5
        # persistence: the same strokes at the same place while the rest of the picture changes
        ref = max(ev.segs, key=lambda s: len(s.frames) * float(np.median([ln.w for ln in s.lines])))
        by_frame = dict(zip(ref.frames, ref.lines))
        idx = ref.frames
        persist = 0.0
        stride = max(1, len(idx) // 80)
        for lag in (1, 2, 3, 5, 8):
            for k in range(0, len(idx), stride):
                i = idx[k]
                ln_j = by_frame.get(i + lag)
                if ln_j is None:
                    continue
                ch = bg(i, i + lag)
                if ch < BG_STEP:
                    continue
                persist = max(persist, clamp(ch / BG_FULL) * clamp((_profile_corr(by_frame[i], ln_j) - SAME_CORR) / (0.95 - SAME_CORR)))
            if persist >= 0.999:
                break
        return float(1.0 - (1.0 - ev_on) * (1.0 - ev_off) * (1.0 - persist))
