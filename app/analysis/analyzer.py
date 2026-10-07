"""Semantic analyzer interface and the rule-based implementation.

``SemanticAnalyzer`` is the seam for a future model-backed analyzer: anything that can
analyse sentences, segment, and enrich scenes can be dropped in. ``RuleBasedAnalyzer`` is a
transparent heuristic implementation (lexicons + cue words + lexical cohesion). It does not
"understand" language the way a model does; its limits are listed in the README.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from app.analysis.claims import build_claim
from app.analysis.entities import extract_entities, gazetteer_spans
from app.analysis.intent import build_intent, score_types
from app.analysis.lexicon import (
    CONTINUATIONS,
    CUES,
    QUESTION_STARTS,
    TRANSITIONS_STRONG,
    TRANSITIONS_WEAK,
)
from app.analysis.models import (
    Claim,
    ClaimType,
    Entity,
    EntityType,
    ListItem,
    NumberKind,
    NumericMention,
    Origin,
    Scene,
    SceneStatus,
    SentenceAnalysis,
    VisualIntent,
)
from app.analysis.numbers import MONTHS, extract_numbers
from app.analysis.prep import AToken, build_tokens
from app.analysis.segmenter import SceneDraft, SegmentationParams, idf_table, segment_transcript
from app.core.exceptions import JobCancelled
from app.core.textutil import STOPWORDS, content_terms, normalize_token, stem
from app.transcription.models import Transcript

VERB_STOP = {"meanwhile", "actually", "really", "very", "just", "also", "however", "last", "next", "per", "across", "differently",
             "month", "year", "week", "day", "time", "matter", "matters", "remember", "consider", "describes", "describe", "nobody",
             "anything", "something", "everything", "many", "much", "most", "several", "often", "always", "never", "already",
"sent", "said", "has", "had", "was", "were", "been", "get", "got", "went", "made", "took", "saw", "came", "gave", "found",
             "told", "became", "began", "grew", "rose", "fell", "hit", "put", "set", "let", "run", "ran", "paid", "bought", "sold",
             "kept", "left", "meant", "led", "brought", "thought", "knew", "felt", "seem", "seems", "become", "becomes", "reach",
             "reaches", "talk", "talking", "look", "looking", "discuss", "explain", "show", "see", "know", "think", "want", "need", "take", "about", "could", "would", "should", "might", "must", "shall", "says", "say", "does", "did", "done", "going", "being"}
def _norm_phrases(phrases) -> list[str]:
    """Normalise cue phrases exactly like the analysed text ("let's" -> "let"), so lookups match."""
    return [" ".join(normalize_token(w) for w in p.split()) for p in phrases]


_STRONG, _WEAK, _CONT = _norm_phrases(TRANSITIONS_STRONG), _norm_phrases(TRANSITIONS_WEAK), _norm_phrases(CONTINUATIONS)
ANAPHORA = {"it", "its", "they", "them", "their", "he", "she", "his", "her", "those", "these"}
WH = {"what", "why", "how", "who", "when", "where", "which"}
AUX = {"is", "are", "do", "does", "did", "can", "could", "should", "would", "will", "have", "has", "was", "were"}
PRONOUN_2ND = {"you", "we", "they", "it", "this", "that", "there", "i", "he", "she"}
CONNECT = {"and", "or", "plus", "also", "as", "well"}
SENTENCE_CUES = ("process", "data", "evidence", "person", "location", "comparison", "event", "abstract", "emotional",
                 "conclusion", "prediction", "opinion")


@dataclass
class EnrichContext:
    transcript: Transcript
    analyses: dict[str, SentenceAnalysis]
    surfaces: dict[str, str]
    overall_topic: str
    idf: dict[str, float]
    params: SegmentationParams
    index: int = 0
    total: int = 1
    prev: "EnrichedScene | None" = None
    sentence_text: dict[str, str] = field(default_factory=dict)


@dataclass
class EnrichedScene:
    scene: Scene
    intent: VisualIntent


class SemanticAnalyzer(ABC):
    name: str = "abstract"
    version: str = "0"

    @abstractmethod
    def analyze_sentences(self, transcript: Transcript, surfaces: dict[str, str] | None = None,
                          progress: Callable[[float, str], None] | None = None,
                          should_cancel: Callable[[], bool] | None = None) -> dict[str, SentenceAnalysis]: ...

    @abstractmethod
    def segment(self, transcript: Transcript, analyses: dict[str, SentenceAnalysis], params: SegmentationParams) -> list[SceneDraft]: ...

    @abstractmethod
    def overall_topic(self, transcript: Transcript, analyses: dict[str, SentenceAnalysis]) -> str: ...

    @abstractmethod
    def enrich_scene(self, draft: SceneDraft, scene_id: str, label: str, ctx: EnrichContext) -> EnrichedScene: ...


def _phrase(norms: list[str], entities: list[Entity] | None = None) -> str:
    """Join words into a topic phrase, keeping entity casing (IRS, Tesla) and capitalising the first word."""
    display = {w.lower(): w for e in entities or [] for w in e.text.split()}
    words = [display.get(n, n) for n in norms]
    s = " ".join(words)
    return s[:1].upper() + s[1:]


class RuleBasedAnalyzer(SemanticAnalyzer):
    name = "rule-based"
    version = "1"

    # ------------------------------------------------------------ sentences
    def analyze_sentences(self, transcript, surfaces=None, progress=None, should_cancel=None):
        wm = transcript.word_map()
        out: dict[str, SentenceAnalysis] = {}
        n = len(transcript.sentences)
        for k, sent in enumerate(transcript.sentences):
            if should_cancel and should_cancel():
                raise JobCancelled()
            words = [wm[w] for w in sent.word_ids]
            out[sent.sentence_id] = self._analyze_sentence(sent.sentence_id, words, surfaces or {})
            if progress and (k % 20 == 0 or k == n - 1):
                progress((k + 1) / n, f"Analysing sentence {k + 1} of {n}")
        return out

    def _analyze_sentence(self, sid: str, words, surfaces: dict[str, str]) -> SentenceAnalysis:
        tokens = build_tokens(words, surfaces)
        text = " ".join(t.text for t in tokens)
        norms = [t.norm for t in tokens]
        entities = extract_entities(tokens)
        numbers = extract_numbers(tokens, sid)
        claim = build_claim(sid, 0, text, tokens, numbers)
        claims = [claim] if claim else []
        a = SentenceAnalysis(
            sentence_id=sid, entities=entities, claims=claims, numbers=numbers, terms=content_terms(text),
            cues=self._cues(tokens, text), word_count=len(words), text=text,
            list_items=self._list_items(tokens),
        )
        a.type_scores = score_types(norms, entities, numbers, [c.type for c in claims])
        return a

    @staticmethod
    def _cues(tokens: list[AToken], text: str) -> list[str]:
        norms = [t.norm for t in tokens]
        lead = " ".join(norms[:4])
        cues: list[str] = []
        if any(lead == p or lead.startswith(p + " ") for p in _STRONG):
            cues.append("transition_strong")
        elif any(lead == p or lead.startswith(p + " ") for p in _WEAK):
            cues.append("transition_weak")
        if "transition_strong" not in cues and any(lead == p or lead.startswith(p + " ") for p in _CONT):
            cues.append("continuation")
        if "transition_strong" in cues and len(tokens) <= 8:
            cues.append("intro")  # "Now let's talk about X." introduces what follows; it should not stand alone
        first = norms[0] if norms else ""
        second = norms[1] if len(norms) > 1 else ""
        if text.rstrip().endswith("?") or first in WH or (first in AUX and second in PRONOUN_2ND):
            cues.append("question")
        if any(n in ANAPHORA for n in norms):
            cues.append("anaphora")
        stems = {stem(n) for n in norms}
        cues += [c for c in SENTENCE_CUES if stems & CUES[c]]
        return cues

    @staticmethod
    def _list_items(tokens: list[AToken]) -> list[ListItem]:
        spans = gazetteer_spans(tokens)
        best: list[tuple[int, int, int | None]] = []  # (start, end, lead token index)
        run: list[tuple[int, int, int | None]] = []
        for k, (a, b) in enumerate(spans):
            if not run:
                run = [(a, b, None)]
                continue
            pa, pb, _ = run[-1]
            gap = [t.norm for t in tokens[pb:a]]
            comma = tokens[pb - 1].text.rstrip().endswith(",") if pb > 0 else False
            if (gap and len(gap) <= 3 and all(g in CONNECT for g in gap)) or (not gap and comma):
                run.append((a, b, pb if gap else None))
            else:
                if len(run) > len(best):
                    best = run
                run = [(a, b, None)]
        if len(run) > len(best):
            best = run
        if len(best) < 3:
            return []
        items = []
        for a, b, lead in best:
            text = " ".join(t.text.strip(".,;:!?\"'") for t in tokens[a:b])
            first = tokens[a].word_ids[0]
            items.append(ListItem(text, first, tokens[lead].word_ids[0] if lead is not None else first))
        return items

    # ------------------------------------------------------------ segmentation
    def segment(self, transcript, analyses, params):
        return segment_transcript(transcript, analyses, params)

    def overall_topic(self, transcript, analyses) -> str:
        seq = list(analyses.values())
        idf = idf_table(seq)
        ents: Counter[str] = Counter()
        names: dict[str, str] = {}
        for a in seq:
            for e in a.entities:
                ents[e.canonical] += e.mentions
                names[e.canonical] = e.text
        norms = [n for sent in transcript.sentences for n in [self._norms(transcript, sent.word_ids)]]
        bigrams: Counter[tuple[str, str]] = Counter()
        for ns in norms:
            bigrams.update(self._bigrams(ns))
        best = max(bigrams, key=lambda b: bigrams[b] * (idf.get(stem(b[0]), 1) + idf.get(stem(b[1]), 1)), default=None)
        if best and bigrams[best] >= 2:
            return _phrase(list(best))
        if ents:
            return names[ents.most_common(1)[0][0]]
        return ""

    @staticmethod
    def _norms(transcript: Transcript, word_ids: list[str]) -> list[str]:
        from app.core.textutil import normalize_token

        wm = transcript.word_map()
        return [normalize_token(wm[w].text) or wm[w].text.lower() for w in word_ids]

    @staticmethod
    def _nounish(w: str) -> bool:
        """Cheap filter for words that can be part of a topic phrase (no POS tagger available)."""
        return (w not in STOPWORDS and w not in VERB_STOP and w not in MONTHS and len(w) > 2
                and not any(c.isdigit() for c in w) and not (w.endswith("ed") and len(w) > 4)
                and not (w.endswith("ly") and len(w) > 4))

    @classmethod
    def _bigrams(cls, norms: list[str]):
        for a, b in zip(norms, norms[1:]):
            if cls._nounish(a) and cls._nounish(b):
                yield (a, b)

    # ------------------------------------------------------------ enrichment
    def enrich_scene(self, draft: SceneDraft, scene_id: str, label: str, ctx: EnrichContext) -> EnrichedScene:
        tr = ctx.transcript
        words = tr.words_between(draft.start, draft.end)
        tokens = build_tokens(words, ctx.surfaces)
        narration = " ".join(w.text for w in words)
        norms = [t.norm for t in tokens]
        entities = extract_entities(tokens)
        numbers = extract_numbers(tokens, "")
        sentences = [ctx.analyses[s] for s in draft.sentence_ids if s in ctx.analyses]
        claims: list[Claim] = []
        for a in sentences:
            claims += a.claims
        cues = Counter(c for a in sentences for c in a.cues)

        # topic and subjects
        topic = self._topic(norms, ctx.idf, entities)
        weak = not entities and (len([n for n in norms if n not in STOPWORDS]) < 3 or (
            bool(sentences) and ("continuation" in sentences[0].cues or "anaphora" in sentences[0].cues)))
        prev = ctx.prev
        inherited_from = None
        if weak and prev is not None:  # context memory: "This could become expensive." keeps the previous subject
            inherited_from = prev.scene.id
            topic = prev.scene.topic or topic  # a context-dependent scene is about what came before, not its own stray word
            if not entities and prev.intent.primary_subject:
                entities = [Entity(prev.intent.primary_subject, prev.scene.entities[0].type, prev.intent.primary_subject.lower(), 0)] \
                    if prev.scene.entities else entities
        if not topic:
            topic = ctx.overall_topic

        scores = score_types(norms, entities, numbers, [c.type for c in claims])
        if inherited_from and prev is not None:
            for k, v in prev.intent.type_scores.items():
                scores[k] = round(scores.get(k, 0.0) + 0.4 * v, 4)
        top_terms = [t for t in self._top_terms(norms, ctx.idf) if self._nounish(t)]
        action = next((n for n in norms if stem(n) in CUES["process"] or stem(n) in CUES["event"]), "")
        context = ctx.overall_topic if ctx.overall_topic and ctx.overall_topic.lower() != topic.lower() else (prev.scene.topic if prev else "")
        intent = build_intent(scene_id, scores, entities, topic, top_terms, action, context, inherited_from)

        importance, why_imp = self._importance(ctx, draft, cues, claims, numbers, entities, len(words), intent)
        word_conf = [w.confidence for w in words if w.confidence is not None]
        conf = draft.boundary_conf if not word_conf else 0.75 * draft.boundary_conf + 0.25 * (sum(word_conf) / len(word_conf))
        status = SceneStatus.READY if conf >= ctx.params.review_threshold else SceneStatus.NEEDS_REVIEW
        summary = self._summary(ctx, sentences, narration)
        script_text = " ".join(ctx.surfaces.get(w.word_id, w.text) for w in words)
        rationale = list(draft.rationale)
        if inherited_from:
            rationale.append(f"Context inherited from the previous scene ({prev.scene.label if prev else '?'})")
        if why_imp:
            rationale.append("Importance: " + ", ".join(why_imp))
        scene = Scene(
            id=scene_id, label=label, start=draft.start, end=draft.end, narration=narration,
            sentence_ids=list(draft.sentence_ids), topic=topic, summary=summary, importance=importance,
            segmentation_confidence=max(0.0, min(1.0, conf)), status=status, origin=Origin.AI,
            entities=entities, claims=claims, numbers=numbers, rationale=rationale, script_text=script_text,
        )
        return EnrichedScene(scene, intent)

    @staticmethod
    def _top_terms(norms: list[str], idf: dict[str, float]) -> list[str]:
        scored: dict[str, float] = {}
        for n in norms:
            if n in STOPWORDS or len(n) < 3 or n.isdigit():
                continue
            scored[n] = scored.get(n, 0.0) + idf.get(stem(n), 1.0)
        return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:4]]

    def _topic(self, norms: list[str], idf: dict[str, float], entities: list[Entity]) -> str:
        """Best phrase for what the scene is about: an adjacent content bigram, else "<entity> <term>", else an entity."""
        places = {w for e in entities if e.type in (EntityType.COUNTRY, EntityType.CITY) for w in e.canonical.split()}
        names = {w for e in entities if e.type not in (EntityType.COUNTRY, EntityType.CITY, EntityType.OBJECT)
                 for w in e.canonical.split()}
        best, best_score = None, 0.0
        for a, b in self._bigrams(norms):
            if a in places and b in places:
                continue  # a location is context, not the topic
            score = idf.get(stem(a), 1.0) + idf.get(stem(b), 1.0) + (1.0 if a in names or b in names else 0.0)
            if score > best_score:
                best, best_score = (a, b), score
        subject = max((e for e in entities if e.type not in (EntityType.COUNTRY, EntityType.CITY)),
                      key=lambda e: (e.mentions, len(e.text)), default=None)
        if best is None and subject is not None:
            terms = [t for t in self._top_terms(norms, idf) if self._nounish(t) and t not in subject.canonical.split()]
            if terms:
                return f"{subject.text} {terms[0]}"
        if best:
            return _phrase(list(best), entities)
        if subject:
            return subject.text
        terms = [t for t in self._top_terms(norms, idf) if self._nounish(t)]
        return terms[0].capitalize() if terms else ""

    @staticmethod
    def _summary(ctx: EnrichContext, sentences: list[SentenceAnalysis], narration: str) -> str:
        """Extractive: the most informative sentence (by idf-weighted terms + entities), trimmed."""
        if not sentences:
            return narration[:180]
        best = max(sentences, key=lambda a: sum(ctx.idf.get(t, 1.0) for t in a.terms) + 2 * len(a.entities) + 2 * len(a.numbers))
        text = best.text if best.text else narration
        return text if len(text) <= 180 else text[:177].rsplit(" ", 1)[0] + "…"

    @staticmethod
    def _importance(ctx: EnrichContext, draft: SceneDraft, cues: Counter[str], claims: list[Claim],
                    numbers: list[NumericMention], entities: list[Entity], n_words: int, intent: VisualIntent):
        score, why = 0.30, []
        if any(c.requires_evidence for c in claims):
            score += 0.15
            why.append("major claim")
        if any(c.type in (ClaimType.LAW, ClaimType.RULE) for c in claims):
            score += 0.05
        data = [n for n in numbers if n.kind in (NumberKind.PRICE, NumberKind.PERCENTAGE, NumberKind.DOLLAR_AMOUNT)]
        if data:
            score += 0.15
            why.append("important number")
        elif numbers:
            score += 0.05
        if any("transition_strong" in a for a in [[c for c in cues if c == "transition_strong"]]) or ctx.index == 0:
            score += 0.10
            why.append("topic transition" if ctx.index else "opening hook")
        emo = cues.get("emotional", 0)
        if emo:
            score += 0.12 + (0.06 if emo > 1 else 0.0)
            why.append("emotional language")
        if intent.type.value == "EVIDENCE":
            score += 0.10
            why.append("evidence")
        if cues.get("conclusion"):
            score += 0.15
            why.append("conclusion")
        if ctx.index == ctx.total - 1 and ctx.total > 1:
            score += 0.08
        if len(entities) >= 3:
            score += 0.05
        if n_words < 6 or (not claims and not entities):
            score -= 0.12
            why.append("supporting narration")
        return max(0.0, min(1.0, score)), why


def new_context(analyzer: SemanticAnalyzer, transcript: Transcript, analyses: dict[str, SentenceAnalysis], surfaces: dict[str, str],
                params: SegmentationParams, overall_topic: str | None = None) -> EnrichContext:
    return EnrichContext(
        transcript=transcript, analyses=analyses, surfaces=surfaces,
        overall_topic=analyzer.overall_topic(transcript, analyses) if overall_topic is None else overall_topic,
        idf=idf_table(list(analyses.values())), params=params,
    )
