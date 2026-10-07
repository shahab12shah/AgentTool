"""Research query generation: several queries with different purposes, never one keyword search.

Queries are built from the structured brief (subjects, action, context, entities, claims, numbers),
inherit context from neighbouring scenes when the scene leans on them ("That means ... paperwork" ->
"IRS reporting paperwork silver sellers"), and "search again" strategies produce *new* wording.
"""

from __future__ import annotations

from typing import Callable

from app.core.textutil import STOPWORDS, stem, tokenize
from app.media.asset import SourceType as S
from app.research.concepts import related
from app.research.models import QueryType as Q
from app.research.models import ResearchBrief, ResearchQuery, ResearchSettings

STRATEGIES = ("broader", "specific", "process", "evidence", "alternative", "sources")
PROCESS_WORDS = {"PROCESS": "manufacturing process", "OBJECT": "close up", "EVENT": "footage", "LITERAL": "footage", "LOCATION": "aerial view",
                 "PERSON": "portrait", "ABSTRACT": "concept", "DATA": "chart", "EVIDENCE": "document", "COMPARISON": "comparison"}
DOC_WORDS = ("notice", "form", "ruling", "law", "rule", "report", "statement", "filing", "letter", "regulation", "policy")
MODIFIERS = ("documentary", "close up", "wide shot", "detail", "overview", "real world")
AGENCY = "GOVERNMENT_AGENCY"

SOURCES_FOR: dict[Q, list[S]] = {
    Q.LITERAL: [S.STOCK_VIDEO, S.STOCK_IMAGE, S.WEB_IMAGE, S.YOUTUBE],
    Q.CONTEXT: [S.WEB_IMAGE, S.STOCK_VIDEO, S.YOUTUBE, S.STOCK_IMAGE],
    Q.PROCESS: [S.STOCK_VIDEO, S.YOUTUBE, S.WEB_VIDEO],
    Q.EVIDENCE: [S.SCREENSHOT, S.WEB_IMAGE],
    Q.ENTITY: [S.WEB_IMAGE, S.YOUTUBE, S.STOCK_IMAGE],
    Q.LOCATION: [S.STOCK_VIDEO, S.STOCK_IMAGE, S.WEB_IMAGE],
    Q.DATA: [S.SCREENSHOT, S.WEB_IMAGE],
    Q.NEWS: [S.YOUTUBE, S.WEB_VIDEO, S.WEB_IMAGE],
    Q.DOCUMENT: [S.SCREENSHOT, S.WEB_IMAGE],
    Q.ALTERNATIVE: [S.STOCK_IMAGE, S.WEB_IMAGE, S.STOCK_VIDEO, S.YOUTUBE, S.AI_GENERATED],
}


def tokens(text: str) -> list[str]:
    return [t.norm for t in tokenize(text)]


def join_unique(*parts: str | list[str], limit: int = 8) -> str:
    """Concatenate phrases, dropping repeated words, keeping order, capped at ``limit`` words."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        for chunk in ([part] if isinstance(part, str) else part):
            for w in chunk.split():
                k = stem(w.lower().strip(".,;:!?\"'"))
                if k and k not in seen:
                    seen.add(k)
                    out.append(w.strip(".,;:!?\"'"))
    return " ".join(out[:limit])


def token_set(text: str) -> frozenset[str]:
    return frozenset(stem(t) for t in tokens(text) if t not in STOPWORDS)


def similar(a: str, b: str, threshold: float = 0.8) -> bool:
    x, y = token_set(a), token_set(b)
    return bool(x and y) and len(x & y) / len(x | y) >= threshold


class QueryGenerator:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings()

    # ------------------------------------------------------------------ public
    def generate(self, brief: ResearchBrief, strategy: str = "initial", generation: int = 0,
                 previous: list[ResearchQuery] | None = None, id_factory: Callable[[], str] | None = None) -> list[ResearchQuery]:
        ids = id_factory or _counter()
        specs = self._initial(brief) if strategy == "initial" else self._again(brief, strategy)
        used = [q.text for q in (previous or [])]
        out: list[ResearchQuery] = []
        for qtype, text, purpose, prio, sources in sorted(specs, key=lambda s: s[3]):
            text = text.strip()
            same_wording_ok = strategy == "sources"  # that strategy reuses the wording on purpose, with a different source order
            if not text or any(similar(text, u, 0.85) for u in ([] if same_wording_ok else used) + [o.text for o in out]):
                continue
            out.append(ResearchQuery(ids(), brief.scene_id, text, qtype, purpose, prio, [s.value for s in sources], generation, strategy))
            if len(out) >= self.settings.max_queries:
                break
        if not out and strategy != "initial":  # everything was already tried: vary wording instead of repeating
            base = join_unique(brief.primary_subject, brief.secondary_subject, brief.key_terms[:2]) or brief.topic
            for m in MODIFIERS:
                text = f"{base} {m}"
                if not any(similar(text, u, 0.95) for u in used):
                    out.append(ResearchQuery(ids(), brief.scene_id, text, Q.ALTERNATIVE, f"Variation ({m}) because the earlier wording was exhausted",
                                             4, [s.value for s in SOURCES_FOR[Q.ALTERNATIVE]], generation, strategy))
                    break
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _prefix(b: ResearchBrief) -> list[str]:
        """Context words to put in front when the scene leans on earlier scenes (never search 'paperwork' alone)."""
        return [" ".join(t.split()[:2]) for t in b.context_terms[:2]] if b.anaphoric else []

    @staticmethod
    def _entities(b: ResearchBrief, *types: str) -> list[str]:
        return [e for e in b.entities if b.entity_types.get(e) in types]

    def _initial(self, b: ResearchBrief):
        pre, P, S2, A = self._prefix(b), b.primary_subject, b.secondary_subject, b.action
        specs = []
        lit = join_unique(pre, b.key_terms[:2], P, S2) if b.anaphoric else join_unique(P, S2)
        specs.append((Q.LITERAL, lit, "Literal visual: show what the narration names", 1, SOURCES_FOR[Q.LITERAL]))
        ctx = join_unique(P, S2, b.context or b.video_topic, pre)
        specs.append((Q.CONTEXT, ctx, "Contextual visual: the subject in its industry / setting", 2, SOURCES_FOR[Q.CONTEXT]))
        word = A if A and A not in STOPWORDS else PROCESS_WORDS.get(b.visual_type, "")
        if b.visual_type in ("PROCESS", "LITERAL", "OBJECT", "EVENT") or A:
            specs.append((Q.PROCESS, join_unique(S2 or pre, P, word), "Process visual: the action or activity being described", 2, SOURCES_FOR[Q.PROCESS]))
        needs_ev = b.evidence_level.value != "NONE"
        if needs_ev:
            claim_terms = join_unique(*(tokens(c)[:4] for c in b.claims[:1]), limit=4) if b.claims else ""
            specs.append((Q.EVIDENCE, join_unique(pre, b.topic or P, claim_terms, "report"),
                          "Evidence visual: a real source that supports the claim", 1 if b.evidence_needed else 4, SOURCES_FOR[Q.EVIDENCE]))
        for e in self._entities(b, "PERSON", "COMPANY", "ORGANIZATION", AGENCY)[:2]:
            specs.append((Q.ENTITY, join_unique(e, b.context or b.topic), f"Entity visual: {e}", 3, SOURCES_FOR[Q.ENTITY]))
        for e in self._entities(b, "COUNTRY", "CITY")[:1]:
            specs.append((Q.LOCATION, join_unique(e, P or b.topic), f"Location visual: {e}", 4, SOURCES_FOR[Q.LOCATION]))
        if b.numbers or b.visual_type == "DATA":
            specs.append((Q.DATA, join_unique(b.topic or P, "chart", b.numbers[:1]), "Data visual: the numbers behind the claim", 2, SOURCES_FOR[Q.DATA]))
        if b.visual_type == "EVENT" or (needs_ev and self._entities(b, AGENCY, "COMPANY")):
            specs.append((Q.NEWS, join_unique(pre, b.topic or P, "news"), "News visual: reporting on the subject", 4, SOURCES_FOR[Q.NEWS]))
        agencies = self._entities(b, AGENCY)
        if agencies or set(b.claim_types) & {"LAW", "RULE"}:
            doc = next((w for c in b.claims for w in tokens(c) if w in DOC_WORDS), "notice")
            specs.append((Q.DOCUMENT, join_unique(agencies[:1] or pre, doc, P if not agencies else S2),
                          "Document visual: the official paper or page itself", 1 if b.evidence_needed else 3, SOURCES_FOR[Q.DOCUMENT]))
        alt = self._alternative(b)
        if alt:
            specs.append((Q.ALTERNATIVE, alt, "Alternative interpretation using related terms", 5, SOURCES_FOR[Q.ALTERNATIVE]))
        return specs

    def _alternative(self, b: ResearchBrief) -> str:
        rel: list[str] = []
        for w in (b.primary_subject.split() + b.secondary_subject.split()):
            rel += related(w, 3)
        rel = [r for r in rel if r not in tokens(b.primary_subject + " " + b.secondary_subject)]
        if not rel and b.secondary_types:
            return join_unique(b.primary_subject, b.secondary_types[0].lower().replace("_", " "))
        return join_unique(rel[:2], b.secondary_subject or b.primary_subject, b.context.split()[:2] if b.context else [])

    def _again(self, b: ResearchBrief, strategy: str):
        pre, P, S2, A = self._prefix(b), b.primary_subject, b.secondary_subject, b.action
        specs: list = []
        if strategy == "broader":
            specs += [(Q.LITERAL, join_unique(P), "Broader: the main subject alone", 1, SOURCES_FOR[Q.LITERAL]),
                      (Q.CONTEXT, join_unique(b.video_topic or b.context, P.split()[:1]), "Broader: the video's overall topic", 2, SOURCES_FOR[Q.CONTEXT]),
                      (Q.ALTERNATIVE, join_unique(related((P.split() or [""])[0], 2)), "Broader: related general concept", 3, SOURCES_FOR[Q.ALTERNATIVE])]
        elif strategy == "specific":
            nums = b.numbers[:1] + b.dates[:1]
            specs += [(Q.LITERAL, join_unique(P, S2, A, b.entities[:2], nums, pre, limit=10), "More specific: subject + action + entities + numbers", 1, SOURCES_FOR[Q.LITERAL]),
                      (Q.CONTEXT, join_unique(b.topic, b.entities[:2], b.context, limit=10), "More specific: full topic with entities", 2, SOURCES_FOR[Q.CONTEXT])]
            if b.claims:
                specs.append((Q.EVIDENCE, join_unique(tokens(b.claims[0])[:7]), "More specific: the claim's own wording", 2, SOURCES_FOR[Q.EVIDENCE]))
        elif strategy == "process":
            base = S2 or P
            specs += [(Q.PROCESS, join_unique(base, "production line"), "Process focus: production", 1, SOURCES_FOR[Q.PROCESS]),
                      (Q.PROCESS, join_unique(P, "installation how it works"), "Process focus: how it works", 2, SOURCES_FOR[Q.PROCESS]),
                      (Q.PROCESS, join_unique(P, S2, A or "manufacturing", "footage"), "Process focus: footage of the activity", 2, SOURCES_FOR[Q.PROCESS])]
        elif strategy == "evidence":
            ent = (self._entities(b, AGENCY, "COMPANY", "ORGANIZATION") or [b.topic or P])[0]
            specs += [(Q.EVIDENCE, join_unique(ent, b.topic or P, "official report"), "Evidence focus: official report", 1, SOURCES_FOR[Q.EVIDENCE]),
                      (Q.DOCUMENT, join_unique(ent, "official statement", pre), "Evidence focus: official statement", 2, SOURCES_FOR[Q.DOCUMENT]),
                      (Q.DATA, join_unique(b.topic or P, "data statistics chart"), "Evidence focus: statistics", 2, SOURCES_FOR[Q.DATA]),
                      (Q.NEWS, join_unique(ent, b.topic or P, "news"), "Evidence focus: news coverage", 3, SOURCES_FOR[Q.NEWS])]
        elif strategy == "alternative":
            specs.append((Q.ALTERNATIVE, self._alternative(b), "Alternative interpretation using related terms", 1, SOURCES_FOR[Q.ALTERNATIVE]))
            for st in b.secondary_types[:2]:
                specs.append((Q.ALTERNATIVE, join_unique(P, S2, PROCESS_WORDS.get(st, st.lower())), f"Alternative visual type: {st}", 2, SOURCES_FOR[Q.ALTERNATIVE]))
            specs.append((Q.ALTERNATIVE, join_unique(b.video_topic, P, "concept"), "Alternative: conceptual treatment", 3, SOURCES_FOR[Q.ALTERNATIVE]))
        elif strategy == "sources":
            for qtype in (Q.LITERAL, Q.CONTEXT):
                srcs = SOURCES_FOR[qtype][2:] + SOURCES_FOR[qtype][:2]  # rotate: try the less obvious sources first
                specs.append((qtype, join_unique(pre, P, S2, b.context if qtype is Q.CONTEXT else ""), f"Same intent, different source order ({qtype.value.lower()})", 1, srcs))
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
        return specs


def _counter() -> Callable[[], str]:
    n = [0]

    def make() -> str:
        n[0] += 1
        return f"query_{n[0]:05d}"

    return make
