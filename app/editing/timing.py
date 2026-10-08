"""ShotTimingService: how long each visual stays, derived from the actual narration (never a fixed duration)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.analysis.models import NumberKind
from app.editing.effective import hook_factor
from app.editing.context import AssetInfo, SceneContext
from app.editing.models import Operation, SceneEditingBrief
from app.editing.presets import StylePreset, shot_factor
from app.editing.models import EditingSettings
from app.transcription.models import Word

PAUSE_GAP = 0.30  # silence between words that counts as a pause
SLOW_WPS, FAST_WPS = 2.1, 3.1
MIN_FRAGMENT = 0.35


@dataclass
class NarrationStats:
    words: int
    wps: float
    pause_total: float
    longest_pause: float
    speech_seconds: float


def narration_stats(words: list[Word], start: float, end: float) -> NarrationStats:
    if not words:
        return NarrationStats(0, 2.5, 0.0, 0.0, max(0.0, end - start))
    pauses = [b.start - a.end for a, b in zip(words, words[1:]) if b.start - a.end > PAUSE_GAP]
    pause_total = sum(pauses)
    speech = max(0.4, (words[-1].end - words[0].start) - pause_total)
    return NarrationStats(len(words), len(words) / speech, pause_total, max(pauses, default=0.0), speech)


def speed_class(wps: float) -> str:
    return "SLOW" if wps < SLOW_WPS else "FAST" if wps > FAST_WPS else "NORMAL"


class ShotTimingService:
    def __init__(self, preset: StylePreset, settings: EditingSettings) -> None:
        self.preset, self.settings = preset, settings

    # ------------------------------------------------------------------ one number: the desired shot length
    def target_shot(self, brief: SceneEditingBrief, ctx: SceneContext) -> float:
        p = self.preset
        t = p.base_shot * shot_factor(self.settings)
        t *= min(1.3, max(0.75, 2.5 / max(0.8, brief.narration_speed)))  # slow narration -> longer shots, fast -> shorter
        t *= 1.0 + 0.45 * brief.visual_complexity
        t *= 1.0 + 0.25 * brief.information_density
        if brief.evidence_treatment_needed:
            t *= 1.15
        short_sentences = brief.sentence_count >= 2 and ctx.scene.duration / max(1, brief.sentence_count) < 2.4
        if brief.importance >= 0.8 and (short_sentences or brief.emotional_intensity > 0.6):
            t *= 0.8  # dramatic statements get punchier
        ref = self.settings.reference
        if ref is not None:
            t *= hook_factor(self.settings, ctx.scene.start)  # an applied reference style may cut the opening faster
        reading = self.reading_time(brief, ctx)
        t = max(t, reading)
        # the style may lower the longest shot to get more cuts, but never below the time a viewer needs to read what is on screen
        return min(p.max_shot if ref is None else max(p.max_shot, reading), max(p.min_shot, t))

    @staticmethod
    def reading_time(brief: SceneEditingBrief, ctx: SceneContext) -> float:
        """Time a viewer needs to take in what is on screen (documents, charts, numbers)."""
        t = 0.0
        if brief.evidence_treatment_needed or ctx.visual_type in ("EVIDENCE", "DATA"):
            t = 2.2 + 3.0 * brief.information_density
        if brief.has_number:
            t = max(t, 2.0)
        return t

    # ------------------------------------------------------------------ cut points
    def cut_candidates(self, ctx: SceneContext, brief: SceneEditingBrief) -> list[tuple[float, str]]:
        """Times where a visual change is natural: sentence starts, list items, key numbers."""
        s = ctx.scene
        cuts: list[tuple[float, str]] = []
        wmap = {w.word_id: w for w in ctx.words}
        for sent in ctx.sentences:
            if s.start + 0.5 < sent.start < s.end - 0.5:
                cuts.append((sent.start, "sentence boundary"))
        for sa in ctx.sentence_analysis.values():
            for item in sa.list_items:
                w = wmap.get(item.lead_word_id)
                if w is not None and s.start + 0.5 < w.start < s.end - 0.5:
                    cuts.append((max(s.start, w.start - 0.10), f"enumeration item “{item.text}”"))
        for n in s.numbers:
            w = wmap.get(n.word_ids[0]) if n.word_ids else None
            if w is not None and s.start + 1.0 < w.start < s.end - 1.0 and brief.change_during_sentence:
                cuts.append((max(s.start, w.start - 0.15), f"key figure “{n.text}”"))
        cuts.sort()
        return cuts

    def plan_visual_times(self, ctx: SceneContext, brief: SceneEditingBrief, assets: list[AssetInfo]) -> list[tuple[float, float, str, str]]:
        """Segment boundaries for the scene: (start, end, operation, reason). Tiles [scene.start, scene.end] exactly."""
        s, p = ctx.scene, self.preset
        d = s.duration
        target = self.target_shot(brief, ctx)
        wants = max(1, int(round(d / target)))
        if len(assets) > 1:  # several visuals: each should appear, as long as every shot stays watchable
            wants = max(wants, min(len(assets), int(d // p.min_shot) or 1))
        elif d <= p.max_shot:
            wants = 1  # one visual, short enough to hold: do not cut just to look busy
        elif wants < 2:
            wants = 2
        if d < 2 * p.min_shot:
            wants = 1
        if wants == 1:
            reason = "Single visual holds for the whole scene" if d <= p.max_shot else "One visual covers the scene"
            return [(s.start, s.end, Operation.HOLD.value if len(assets) == 1 else Operation.CUT.value, f"{reason} ({d:.1f}s narration)")]
        cands = self.cut_candidates(ctx, brief)
        chosen: list[tuple[float, str]] = []
        ideal = [s.start + d * k / wants for k in range(1, wants)]
        pool = list(cands)
        for target_t in ideal:
            best = None
            for t, why in pool:
                if abs(t - target_t) <= d / (2 * wants) and all(abs(t - c[0]) >= p.min_shot for c in chosen) and t - s.start >= p.min_shot and s.end - t >= p.min_shot:
                    if best is None or abs(t - target_t) < abs(best[0] - target_t):
                        best = (t, why)
            if best is None:  # no natural cut nearby: use the closest word boundary
                wb = min((w.start for w in ctx.words if w.start - s.start >= p.min_shot and s.end - w.start >= p.min_shot),
                         key=lambda x: abs(x - target_t), default=target_t)
                best = (wb, "nearest word boundary")
            if all(abs(best[0] - c[0]) >= p.min_shot for c in chosen):
                chosen.append(best)
        chosen.sort()
        edges = [(s.start, "")] + chosen
        out = []
        for i, (a, _why) in enumerate(edges):
            b = edges[i + 1][0] if i + 1 < len(edges) else s.end
            why = edges[i + 1][1] if i + 1 < len(edges) else ""
            op = Operation.SPLIT.value if len(assets) == 1 else Operation.CUT.value
            out.append((a, b, op, (f"Cut at {why}" if why else "Scene end") + f"; shot {b - a:.1f}s"))
        return out


# ------------------------------------------------------------------ mapping a time span onto source media
@dataclass
class SourceFit:
    source_in: float
    source_out: float
    speed: float
    operation: str
    duration: float  # timeline duration actually covered (may be shorter than asked if the source is short)
    note: str = ""


def fit_source(asset: AssetInfo, want: float, offset: float, *, min_speed: float = 0.8) -> SourceFit:
    """Choose the non-destructive source range for ``want`` seconds of timeline. The media itself is never changed."""
    if asset.is_still or not asset.duration:
        return SourceFit(0.0, want, 1.0, Operation.HOLD.value, want)
    total = float(asset.duration)
    offset = max(0.0, min(offset, max(0.0, total - 0.05)))
    if total - offset >= want - 1e-6:
        return SourceFit(offset, offset + want, 1.0, Operation.TRIM.value, want)
    if total >= want:  # the cursor ran past the end: restart earlier inside the clip
        return SourceFit(total - want, total, 1.0, Operation.TRIM.value, want, "Source range moved earlier to fit")
    if total / want >= min_speed:  # slightly short: stretch with a gentle slow-down
        return SourceFit(0.0, total, total / want, Operation.EXTEND.value, want, f"Slowed to {total / want:.2f}x to cover the narration")
    return SourceFit(0.0, total, 1.0, Operation.SHORTEN.value, total, "Source is shorter than the narration; the remainder is covered by another shot")


def stable_unit(*parts: object) -> float:
    """Deterministic pseudo-random value in [0,1) from the inputs (keeps plans reproducible and cacheable)."""
    import hashlib

    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def number_priority(kind: NumberKind) -> int:
    return {NumberKind.DOLLAR_AMOUNT: 5, NumberKind.PERCENTAGE: 5, NumberKind.PRICE: 4, NumberKind.DEADLINE: 4, NumberKind.DATE: 3,
            NumberKind.QUANTITY: 3, NumberKind.AGE: 3, NumberKind.YEAR: 2}.get(kind, 2)


_ = math  # (math kept for future curves)
