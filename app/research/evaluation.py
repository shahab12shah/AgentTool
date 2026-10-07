"""Visual evaluation: how well does this candidate explain what the narration is saying?

HONEST SCOPE: the default evaluator reads *metadata* (title, description, tags, source, size, duration). It never
looks at pixels, so a mislabelled item can fool it. Scores say so (``basis``) and confidence is capped accordingly.
A vision-model evaluator can replace ``MetadataEvaluator`` behind ``CandidateEvaluator``.

Score = weighted sum of seven components on 0-100 (weights configurable in ``ResearchSettings.weights``):
semantic 35, subject 20, context 15, action 10, timing 10, quality 5, source suitability 5.
Single-keyword matches cannot score well: without the primary subject the semantic score is capped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.textutil import STOPWORDS, stem, tokenize
from app.media.asset import SourceType as S
from app.research.concepts import domains_of
from app.research.models import (
    DEFAULT_WEIGHTS,
    Candidate,
    CandidateScore,
    CandidateStatus,
    Confidence,
    EvidenceKind,
    ResearchBrief,
    ScoreCategory,
    ScoreComponents,
)

CLIP_STEPS = (3.5, 5.0, 7.0, 10.0)


def category_for(score: float) -> ScoreCategory:
    if score >= 90:
        return ScoreCategory.EXCELLENT
    if score >= 80:
        return ScoreCategory.GOOD
    if score >= 70:
        return ScoreCategory.REVIEW
    if score >= 60:
        return ScoreCategory.WEAK
    return ScoreCategory.REJECT


def _stems(text: str) -> list[str]:
    return [stem(t.norm) for t in tokenize(text) if t.norm not in STOPWORDS and len(t.norm) > 1]


def _group(text: str) -> list[str]:
    return list(dict.fromkeys(_stems(text)))


class CandidateEvaluator(ABC):
    name = "abstract"

    @abstractmethod
    def evaluate(self, brief: ResearchBrief, candidate: Candidate, min_accuracy: float) -> CandidateScore: ...


@dataclass
class _Match:
    fraction: float  # 0..1, share of the group's terms found (related-domain words count half)
    literal: float  # 0..1, share found as the actual words (no domain credit): this decides whether the subject is really present


class MetadataEvaluator(CandidateEvaluator):
    name = "metadata"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        total = sum(w.values()) or 1.0
        self.weights = {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------------ matching helpers
    @staticmethod
    def _match(terms: list[str], text_stems: set[str], text_domains: set[str]) -> _Match:
        if not terms:
            return _Match(0.0, 0.0)
        got, exact = 0.0, 0
        for t in terms:
            if t in text_stems:
                got += 1.0
                exact += 1
            elif domains_of(t) & text_domains:
                got += 0.5  # same real-world domain (e.g. "photovoltaic" for "solar")
        return _Match(min(1.0, got / len(terms)), exact / len(terms))

    # ------------------------------------------------------------------ main
    def evaluate(self, brief: ResearchBrief, c: Candidate, min_accuracy: float) -> CandidateScore:
        text = c.text if c.status is not CandidateStatus.PROPOSED else " ".join([c.title, c.description, " ".join(c.tags)])
        tstems = set(_stems(text))
        tdomains: set[str] = set()
        for s in tstems:
            tdomains |= domains_of(s)
        basis = "PROMPT" if c.source_type is S.AI_GENERATED else "METADATA"

        primary, secondary = _group(brief.primary_subject), _group(brief.secondary_subject)
        topic = [t for t in _group(brief.topic) if t not in primary + secondary]
        action = _group(brief.action) if brief.action else []
        ents = [_group(e) for e in brief.entities]
        ctx_terms = _group(" ".join([brief.context] + brief.context_terms))
        factors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []

        m_primary = self._match(primary, tstems, tdomains)
        m_secondary = self._match(secondary, tstems, tdomains)
        m_topic = self._match(topic, tstems, tdomains)
        ent_fracs = [self._match(g, tstems, tdomains).fraction for g in ents if g]
        m_ent = sum(ent_fracs) / len(ent_fracs) if ent_fracs else 0.0
        m_action = self._match(action, tstems, tdomains)
        m_ctx = self._match(ctx_terms, tstems, tdomains)

        # ---- semantic: weighted concept coverage, never a single keyword
        groups = [(m_primary.fraction, 3.0 if primary else 0), (m_secondary.fraction, 2.0 if secondary else 0), (m_topic.fraction, 1.0 if topic else 0),
                  (m_ent, 1.5 if ents else 0), (m_action.fraction, 1.0 if action else 0)]
        wsum = sum(w for _, w in groups) or 1.0
        coverage = sum(f * w for f, w in groups) / wsum
        semantic = 100.0 * min(1.0, coverage ** 0.85 * 1.05)
        primary_found = m_primary.literal >= 0.5 if primary else True  # related words never stand in for the subject itself
        keyword_only = bool(primary) and not primary_found and (m_secondary.fraction > 0 or m_ent > 0 or m_topic.fraction > 0)
        if not primary_found:
            semantic = min(semantic, 55.0)  # a side keyword ("silver") without the actual subject ("solar installations")
        if primary and len(primary) > 1 and f" {' '.join(t.norm for t in tokenize(brief.primary_subject))} " in f" {' '.join(t.norm for t in tokenize(text))} ":
            semantic = min(100.0, semantic + 6.0)  # the subject phrase appears intact

        # ---- subject
        subj = 100.0 * (0.7 * m_primary.fraction + 0.3 * (m_secondary.fraction if secondary else m_primary.fraction)) if primary else 70.0
        agencies = [e for e in brief.entities if brief.entity_types.get(e) in ("GOVERNMENT_AGENCY", "PERSON", "COMPANY", "ORGANIZATION")]
        missing_names = [e for e in agencies if not set(_group(e)) & tstems]
        if missing_names and not primary_found:
            subj = max(0.0, subj - 10.0)

        # ---- context + conflicts
        avoid_hits = [a for a in brief.avoid_terms if stem(a) in tstems]
        conflict = bool(avoid_hits) and not primary_found
        context = 100.0 * (0.35 + 0.65 * m_ctx.fraction) if ctx_terms else 70.0
        if brief.context and domains_of(brief.context.split()[0]) & tdomains:
            context = max(context, 65.0)
        if avoid_hits:
            context = max(0.0, context - (45.0 if conflict else 15.0))

        # ---- action
        if action:
            act = 100.0 if m_action.fraction >= 0.99 else 40.0 + 60.0 * m_action.fraction
        else:
            act = 75.0

        timing, partial, rec = self._timing(brief, c)
        quality = self._quality(brief, c)
        source = self._source(brief, c)

        comp = ScoreComponents(*(round(max(0.0, min(100.0, v)), 1) for v in (semantic, subj, context, act, timing, quality, source)))
        w = self.weights
        overall = (w["semantic"] * comp.semantic + w["subject"] * comp.subject + w["context"] * comp.context + w["action"] * comp.action
                   + w["timing"] * comp.timing + w["quality"] * comp.quality + w["source"] * comp.source)

        # ---- caps that keep the score honest
        caps: list[str] = []
        if conflict:
            overall = min(overall, 59.0)
            caps.append("matches a generic/unrelated visual the brief says to avoid")
        if keyword_only:
            overall = min(overall, 64.0)
            caps.append("only a side keyword matches, not the main subject")
        if brief.evidence_needed and c.evidence_kind is not EvidenceKind.EVIDENCE:
            overall = min(overall, 84.0)
            caps.append("the scene needs evidence and this is a decorative visual")
        if c.source_type is S.AI_GENERATED:
            overall = min(overall, 70.0 if brief.evidence_needed else (88.0 if c.local_path else 82.0))
            caps.append("AI-generated imagery is unverified" + (" and cannot serve as evidence" if brief.evidence_needed else ""))
        overall = round(max(0.0, min(100.0, overall)), 1)

        # ---- explanation
        if primary and primary_found:
            positives.append(f"Matches the main subject “{brief.primary_subject}”")
        elif primary:
            negatives.append(f"Does not mention the main subject “{brief.primary_subject}”")
        if secondary and m_secondary.fraction >= 0.5:
            positives.append(f"Matches “{brief.secondary_subject}”")
        if ctx_terms and m_ctx.fraction >= 0.5:
            positives.append(f"Fits the context ({brief.context or ', '.join(brief.context_terms[:2])})")
        elif ctx_terms and m_ctx.fraction == 0:
            negatives.append("No sign of the surrounding context")
        if action and m_action.fraction >= 0.99:
            positives.append(f"Shows the action “{brief.action}”")
        elif action:
            negatives.append(f"Action “{brief.action}” not evident")
        if c.duration and brief.scene_duration:
            (positives if timing >= 85 else negatives).append(
                f"{'Fits' if timing >= 85 else 'Short for'} the scene duration ({c.duration:.1f}s vs {brief.scene_duration:.1f}s)")
        if brief.evidence_needed:
            (positives if c.evidence_kind is EvidenceKind.EVIDENCE else negatives).append(
                "A real source/page that can support the claim" if c.evidence_kind is EvidenceKind.EVIDENCE else "Decorative, not evidence")
        if avoid_hits:
            negatives.append(f"Looks like a generic visual to avoid ({', '.join(avoid_hits[:2])})")
        else:
            positives.append("No obvious conflict with the narration")
        factors = [f"✓ {p}" for p in positives] + [f"✗ {n}" for n in negatives] + [f"⚠ Capped: {x}" for x in caps]
        reason = (positives[0] if positives and overall >= 60 else (negatives[0] if negatives else "Weak overall match")) + "."
        if len(positives) > 1 and overall >= 60:
            reason += f" {positives[1]}."
        if caps:
            reason += f" Limited because {caps[0]}."

        # ---- confidence (metadata can't be fully trusted)
        hits = sum(1 for m in (m_primary, m_secondary, m_ctx, m_action) if m.fraction >= 0.5) + (1 if m_ent >= 0.5 else 0)
        rich = len(tstems) >= 6
        if overall < min_accuracy or keyword_only or conflict or basis == "PROMPT" or len(tstems) < 3:
            conf = Confidence.LOW
        elif hits >= 3 and rich and primary_found:
            conf = Confidence.HIGH
        else:
            conf = Confidence.MEDIUM
        crop, orient = self._crop(brief, c)
        return CandidateScore(c.candidate_id, c.scene_id, overall, comp, category_for(overall), conf, reason, factors, basis, rec, partial,
                              crop, orient, "", min_accuracy=min_accuracy)

    # ------------------------------------------------------------------ components
    @staticmethod
    def _timing(brief: ResearchBrief, c: Candidate) -> tuple[float, bool, float | None]:
        need = brief.scene_duration
        rec = recommend_duration(brief, c)
        if c.kind == "IMAGE" or need <= 0:
            return 95.0, False, rec
        seg_len = c.segment.length if c.segment and c.segment.length else None
        avail = seg_len or c.duration
        if avail is None:
            return 80.0, False, rec
        if c.segment and c.segment.basis == "UNKNOWN" and (c.duration or 0) > need * 3:
            return 82.0, False, rec  # a long video with no known section: usable, but someone must pick the section
        want = max(1.5, need - 0.4)
        if avail >= want:
            return 100.0, False, rec
        return max(40.0, 100.0 * avail / want), True, rec

    @staticmethod
    def _quality(brief: ResearchBrief, c: Candidate) -> float:
        if c.source_type is S.AI_GENERATED and not c.local_path:
            return 75.0
        if not c.width or not c.height:
            return 70.0
        score = 100.0 if c.height >= 1080 else 85.0 if c.height >= 720 else 60.0 if c.height >= 480 else 35.0
        want_ar = brief.project_width / brief.project_height
        ar = c.width / c.height
        if abs(ar - want_ar) / want_ar > 0.35:
            score -= 12.0
        return max(0.0, score)

    @staticmethod
    def _source(brief: ResearchBrief, c: Candidate) -> float:
        t = brief.visual_type
        s = c.source_type
        if brief.evidence_needed:
            if c.evidence_kind is EvidenceKind.EVIDENCE:
                return 100.0
            return 20.0 if s is S.AI_GENERATED else 35.0
        table: dict[str, dict[S, float]] = {
            "EVIDENCE": {S.SCREENSHOT: 100, S.WEB_IMAGE: 85, S.WEB_VIDEO: 70, S.YOUTUBE: 65, S.STOCK_IMAGE: 40, S.STOCK_VIDEO: 40, S.AI_GENERATED: 15},
            "DATA": {S.SCREENSHOT: 95, S.WEB_IMAGE: 85, S.STOCK_VIDEO: 55, S.AI_GENERATED: 30},
            "PROCESS": {S.STOCK_VIDEO: 100, S.YOUTUBE: 90, S.WEB_VIDEO: 90, S.WEB_IMAGE: 75, S.STOCK_IMAGE: 70, S.AI_GENERATED: 65},
            "PERSON": {S.WEB_IMAGE: 100, S.YOUTUBE: 90, S.WEB_VIDEO: 85, S.STOCK_IMAGE: 60, S.AI_GENERATED: 20},
            "EVENT": {S.YOUTUBE: 100, S.WEB_VIDEO: 95, S.WEB_IMAGE: 85, S.STOCK_VIDEO: 60, S.AI_GENERATED: 25},
            "ABSTRACT": {S.STOCK_VIDEO: 90, S.AI_GENERATED: 90, S.STOCK_IMAGE: 85},
        }
        return table.get(t, {}).get(s, 80.0 if s is not S.AI_GENERATED else 65.0)

    @staticmethod
    def _crop(brief: ResearchBrief, c: Candidate) -> tuple[str, str]:
        if not c.width or not c.height:
            return "", ""
        ar, want = c.width / c.height, brief.project_width / brief.project_height
        orient = "square" if abs(ar - 1) < 0.05 else "landscape" if ar > 1 else "portrait"
        if abs(ar - want) / want < 0.05:
            return "Fits the project frame; no crop needed.", orient
        if ar > want:
            return f"Wider than the {brief.project_width}×{brief.project_height} frame: centre-crop the sides.", orient
        return f"Taller than the {brief.project_width}×{brief.project_height} frame: centre-crop top/bottom (or place on a background).", orient


def recommend_duration(brief: ResearchBrief, c: Candidate) -> float | None:
    """How long to show this visual (never a fixed 5 s). Images fill the scene; clips are fitted to the narration.

    This is an editing suggestion only. It says nothing about whether reusing the media is permitted.
    """
    need = brief.scene_duration
    if need <= 0:
        return None
    base = max(1.5, need - 0.4)  # leave a little room for the cut
    if c.kind == "IMAGE":
        return round(need, 2)
    avail = (c.segment.length if c.segment and c.segment.length else None) or c.duration
    if c.source_type is S.YOUTUBE and (avail is None or c.segment.basis in ("UNKNOWN", "DEFAULT")):
        fits = [s for s in CLIP_STEPS if s <= base]
        return fits[-1] if fits else round(base, 2)  # 3.5 / 5 / 7 / 10 s, whichever fits the narration best
    return round(min(base, avail), 2) if avail else round(base, 2)


class VisualEvaluationService:
    """Evaluates candidates against a brief with a pluggable evaluator."""

    def __init__(self, evaluator: CandidateEvaluator | None = None, weights: dict[str, float] | None = None) -> None:
        self.evaluator = evaluator or MetadataEvaluator(weights)

    def evaluate(self, brief: ResearchBrief, candidate: Candidate, min_accuracy: float = 85.0) -> CandidateScore:
        return self.evaluator.evaluate(brief, candidate, min_accuracy)

    def evaluate_all(self, brief: ResearchBrief, candidates: list[Candidate], min_accuracy: float = 85.0) -> dict[str, CandidateScore]:
        return {c.candidate_id: self.evaluate(brief, c, min_accuracy) for c in candidates}
