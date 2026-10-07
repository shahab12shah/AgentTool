"""Ranking: accuracy dominates; preferences, repetition and continuity only nudge.

rank_score = accuracy + adjustments, where positive adjustments are capped at +6 and negative ones at -15.
A candidate that is 7+ accuracy points better therefore always outranks a preferred-source candidate
(accuracy beats source preference), while exact repeats are still pushed down hard but never forbidden.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.analysis.models import VisualType
from app.core.textutil import STOPWORDS, stem, tokenize
from app.media.asset import SourceType as S
from app.research.dedupe import identity_keys
from app.research.models import (
    Candidate,
    CandidateScore,
    CandidateStatus,
    EvidenceKind,
    RankedEntry,
    ResearchBrief,
    ResearchSettings,
)
from app.visual.preferences import SourceKind, VisualPreferences

KIND_OF = {S.YOUTUBE: SourceKind.YOUTUBE, S.STOCK_IMAGE: SourceKind.STOCK_IMAGES, S.STOCK_VIDEO: SourceKind.STOCK_VIDEOS,
           S.AI_GENERATED: SourceKind.AI_IMAGES, S.WEB_IMAGE: SourceKind.WEB_IMAGES, S.WEB_VIDEO: SourceKind.WEB_VIDEOS,
           S.SCREENSHOT: SourceKind.SCREENSHOTS}
TYPE_OF = {v: k for k, v in KIND_OF.items()}
MAX_BONUS, MAX_PENALTY = 6.0, -15.0


def source_kind(t: S) -> SourceKind | None:
    return KIND_OF.get(t)


def title_terms(c: Candidate) -> frozenset[str]:
    return frozenset(stem(t.norm) for t in tokenize(c.title + " " + " ".join(c.tags[:5])) if t.norm not in STOPWORDS)


@dataclass
class UsedVisual:
    scene_index: int
    keys: set[str]
    fingerprint: str
    terms: frozenset[str]
    source_type: S | None


@dataclass
class RankingHistory:
    """What the rest of the video already uses (from approved assignments and chosen candidates)."""

    used: list[UsedVisual] = field(default_factory=list)
    source_counts: Counter = field(default_factory=Counter)  # SourceType -> number of assigned visuals
    scene_index: int = 0
    previous_terms: frozenset[str] = frozenset()  # visual terms of the previous scene's chosen/best visual

    @property
    def total(self) -> int:
        return sum(self.source_counts.values())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class Ranker:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings()

    # ------------------------------------------------------------------ adjustments
    def adjustments(self, c: Candidate, score: CandidateScore, brief: ResearchBrief, prefs: VisualPreferences,
                    history: RankingHistory) -> dict[str, float]:
        adj: dict[str, float] = {}
        # --- source preference (soft): target-share deficit + user priority + scene fit
        kind = source_kind(c.source_type)
        if kind is not None:
            st = prefs.setting(kind)
            enabled = [(k, prefs.setting(k)) for k in SourceKind if prefs.setting(k).enabled]
            total_target = sum(s.target_percent for _, s in enabled) or 1.0
            want = st.target_percent / total_target
            have = history.source_counts.get(c.source_type, 0) / history.total if history.total else 1.0 / max(len(enabled), 1)
            pref = 4.0 * max(-1.0, min(1.0, (want - have) / max(want, 0.15)))
            pref += {1: -1.5, 2: -0.75, 3: 0.0, 4: 0.75, 5: 1.5}.get(getattr(st, "priority", 3), 0.0)
            order = brief.preferred_sources
            if c.source_type.value in order:
                pref += max(0.0, 2.0 - 0.5 * order.index(c.source_type.value))
            if not st.enabled:
                pref -= 2.0  # an expanded-source result is allowed, but is not preferred
            adj["source_preference"] = pref
        # --- user rules
        if c.source_type is S.AI_GENERATED:
            if prefs.prefer_real_visuals:
                adj["prefer_real"] = -3.0
            if prefs.prefer_ai_visuals:
                adj["prefer_ai"] = 3.0
        elif prefs.prefer_real_visuals:
            adj["prefer_real"] = 1.5
        if brief.evidence_needed and prefs.prefer_evidence:
            adj["evidence"] = 3.0 if c.evidence_kind is EvidenceKind.EVIDENCE else -3.0
        if prefs.match_narration_literally and brief.visual_type in (VisualType.LITERAL.value, VisualType.OBJECT.value):
            adj["literal"] = 1.0 if score.components.subject >= 90 else 0.0
        # --- repetition (soft): exact reuse hurts a lot, generic look-alikes a little
        keys = identity_keys(c)
        strength = 1.0 if prefs.avoid_repeated_visuals else 0.25
        rep = 0.0
        for u in history.used:
            same = bool(keys & u.keys) or (c.fingerprint and u.fingerprint and c.fingerprint == u.fingerprint)
            if same:
                rep = min(rep, -12.0 * strength)
            elif u.scene_index >= history.scene_index - 5 and jaccard(title_terms(c), u.terms) >= 0.6:
                rep = min(rep, -4.0 * strength)
        if rep:
            adj["repetition"] = rep
        # --- continuity with the previous scene's visual
        if history.previous_terms and len(title_terms(c) & history.previous_terms) >= 2:
            adj["continuity"] = 2.0
        return adj

    @staticmethod
    def _cap(adj: dict[str, float]) -> float:
        pos = sum(v for v in adj.values() if v > 0)
        neg = sum(v for v in adj.values() if v < 0)
        return min(pos, MAX_BONUS) + max(neg, MAX_PENALTY)

    # ------------------------------------------------------------------ ranking + selection
    def rank(self, pairs: list[tuple[Candidate, CandidateScore]], brief: ResearchBrief, prefs: VisualPreferences,
             history: RankingHistory) -> list[RankedEntry]:
        """Ranked entries: first = BEST, then up to ``alternatives`` meaningfully different ALTERNATIVEs, the rest OTHER."""
        entries: list[tuple[RankedEntry, Candidate]] = []
        for c, sc in pairs:
            if c.status is CandidateStatus.REJECTED:
                continue
            adj = self.adjustments(c, sc, brief, prefs, history)
            entries.append((RankedEntry(c.candidate_id, round(sc.overall + self._cap(adj), 2), sc.overall, adj, "OTHER"), c))
        entries.sort(key=lambda e: (-e[0].rank_score, -e[0].accuracy, e[0].candidate_id))
        if not entries:
            return []
        entries[0][0].role = "BEST"
        chosen: list[Candidate] = [entries[0][1]]
        remaining = entries[1:]
        # alternatives: greedy, penalising look-alikes so the set is varied; never padded with rejects
        floor = self.settings.alt_min_score
        while remaining and sum(1 for e, _ in entries if e.role == "ALTERNATIVE") < self.settings.alternatives:
            best_i, best_val = -1, -1e9
            for i, (e, c) in enumerate(remaining):
                if e.accuracy < floor:
                    continue
                sim = max(
                    (3.0 * (c.source_type == o.source_type and c.kind == o.kind)) + (4.0 * (jaccard(title_terms(c), title_terms(o)) >= 0.5))
                    + (6.0 if c.fingerprint and o.fingerprint and c.fingerprint == o.fingerprint else 0.0) for o in chosen)
                val = e.rank_score - sim
                if val > best_val:
                    best_i, best_val = i, val
            if best_i < 0:
                break
            e, c = remaining.pop(best_i)
            e.role = "ALTERNATIVE"
            chosen.append(c)
        return [e for e, _ in entries if e.role == "BEST"] + sorted([e for e, _ in entries if e.role == "ALTERNATIVE"], key=lambda e: -e.rank_score) \
            + [e for e, _ in entries if e.role == "OTHER"]
