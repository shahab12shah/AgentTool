"""Visual-requirement scoring and structured visual intent (structure only: no search happens here)."""

from __future__ import annotations

from collections import Counter

from app.analysis.lexicon import CUES
from app.analysis.models import (
    ClaimType,
    Entity,
    EntityType,
    NumberKind,
    NumericMention,
    Origin,
    VisualIntent,
    VisualType,
)
from app.core.textutil import stem

# Tie-break order: the more specific requirement wins.
PRIORITY = [VisualType.EVIDENCE, VisualType.DATA, VisualType.PERSON, VisualType.LOCATION, VisualType.PROCESS,
            VisualType.COMPARISON, VisualType.EVENT, VisualType.OBJECT, VisualType.LITERAL, VisualType.ABSTRACT]
_DATA_NUMBERS = {NumberKind.PRICE, NumberKind.PERCENTAGE, NumberKind.DOLLAR_AMOUNT, NumberKind.QUANTITY}
_PEOPLE = {EntityType.PERSON}
_PLACES = {EntityType.COUNTRY, EntityType.CITY}
_THINGS = {EntityType.OBJECT, EntityType.PRODUCT, EntityType.TECHNOLOGY}
_NAME_TYPES = {EntityType.PERSON, EntityType.COMPANY, EntityType.ORGANIZATION, EntityType.COUNTRY, EntityType.CITY,
               EntityType.GOVERNMENT_AGENCY}


def _n(stems: list[str], cue: str) -> int:
    return sum(1 for s in stems if s in CUES[cue])


def score_types(norms: list[str], entities: list[Entity], numbers: list[NumericMention], claim_types: list[ClaimType]) -> dict[str, float]:
    """Score every VisualType for a stretch of narration (a sentence or a whole scene)."""
    names = {w for e in entities if e.type in _NAME_TYPES for w in e.canonical.split()}
    stems = [stem(n) for n in norms if n not in names]  # "States" in "United States" is not a location cue
    etypes = Counter(e.type for e in entities)
    nkinds = Counter(n.kind for n in numbers)
    ctypes = Counter(claim_types)
    s = {t: 0.0 for t in VisualType}
    s[VisualType.DATA] = 1.1 * sum(nkinds[k] for k in _DATA_NUMBERS) + 0.4 * _n(stems, "data")
    s[VisualType.EVIDENCE] = (0.8 * _n(stems, "evidence") + 1.0 * (ctypes[ClaimType.LAW] + ctypes[ClaimType.RULE] + ctypes[ClaimType.QUOTE])
                              + 0.6 * etypes[EntityType.GOVERNMENT_AGENCY])
    s[VisualType.PERSON] = 1.2 * sum(etypes[t] for t in _PEOPLE) + 0.5 * _n(stems, "person")
    s[VisualType.LOCATION] = 1.2 * sum(etypes[t] for t in _PLACES) + 0.4 * _n(stems, "location")
    s[VisualType.PROCESS] = 0.9 * _n(stems, "process")
    s[VisualType.COMPARISON] = 0.8 * _n(stems, "comparison")
    if _n(stems, "comparison") and len(entities) >= 2:  # "A is higher than B" with two named things: show them side by side
        s[VisualType.COMPARISON] += 1.8
    s[VisualType.EVENT] = 0.8 * _n(stems, "event") + 0.4 * (nkinds[NumberKind.DATE] + nkinds[NumberKind.YEAR])
    s[VisualType.OBJECT] = 0.9 * sum(etypes[t] for t in _THINGS)
    concrete = sum(etypes[t] for t in _THINGS | {EntityType.COMPANY, EntityType.FINANCIAL_INSTRUMENT})
    s[VisualType.LITERAL] = 0.35 * concrete + (0.2 if concrete and not _n(stems, "abstract") else 0.0)
    s[VisualType.ABSTRACT] = (0.6 * _n(stems, "abstract") + 0.5 * (ctypes[ClaimType.PREDICTION] + ctypes[ClaimType.OPINION])
                              + 0.15)
    return {t.value: round(v, 4) for t, v in s.items()}


def pick_type(scores: dict[str, float]) -> tuple[VisualType, list[VisualType], float]:
    ranked = sorted(PRIORITY, key=lambda t: (-scores.get(t.value, 0.0), PRIORITY.index(t)))
    top, second = ranked[0], ranked[1]
    tv, sv = scores[top.value], scores[second.value]
    conf = 0.4 + 0.6 * min(1.0, (tv - sv) / max(tv, 0.01)) * min(1.0, tv / 1.5)
    secondary = [t for t in ranked[1:3] if scores[t.value] >= 0.6 * tv and scores[t.value] > 0.3]
    return top, secondary, max(0.0, min(1.0, conf))


_TEMPLATES: dict[VisualType, list[str]] = {
    VisualType.PROCESS: ["footage of {subj} being {action}", "close-up of {sec}"],
    VisualType.DATA: ["chart or graph showing {subj}", "on-screen number / animated statistic"],
    VisualType.EVIDENCE: ["screenshot or scan of the real {subj}", "close-up of the official document"],
    VisualType.PERSON: ["footage or portrait of {subj}"],
    VisualType.LOCATION: ["establishing shot of {subj}", "map highlighting {subj}"],
    VisualType.COMPARISON: ["side-by-side visual of {subj} and {sec}"],
    VisualType.EVENT: ["news-style footage or photo of {subj}"],
    VisualType.OBJECT: ["clean product/object shot of {subj}"],
    VisualType.LITERAL: ["footage that directly shows {subj}"],
    VisualType.ABSTRACT: ["stylised or metaphorical imagery for {subj}"],
}
_AVOID: dict[VisualType, list[str]] = {
    VisualType.EVIDENCE: ["fabricated or AI-generated documents", "unrelated stock footage"],
    VisualType.DATA: ["decorative footage that ignores the numbers"],
    VisualType.ABSTRACT: ["literal footage that misreads the metaphor"],
}


def suggest_visuals(vtype: VisualType, subject: str, secondary: str = "", action: str = "", topic: str = "") -> list[str]:
    """Descriptive suggestions only (what kind of visual fits). Nothing is searched or fetched."""
    fmt = {"subj": subject or topic or "the subject", "sec": secondary or subject or "the subject",
           "action": action or "used", "topic": topic or subject}
    return [t.format(**fmt) for t in _TEMPLATES[vtype]]


def build_intent(
    scene_id: str,
    scores: dict[str, float],
    entities: list[Entity],
    topic: str,
    top_terms: list[str],
    action: str,
    context: str,
    inherited_from: str | None = None,
) -> VisualIntent:
    vtype, secondary, conf = pick_type(scores)
    ranked = sorted(entities, key=lambda e: (-e.mentions, -len(e.text)))
    subjects = [e.text for e in ranked] + [t for t in top_terms if t.lower() not in {e.text.lower() for e in ranked}]
    primary = subjects[0] if subjects else topic
    sec = subjects[1] if len(subjects) > 1 else ""
    preferred = suggest_visuals(vtype, primary, sec, action, topic)
    return VisualIntent(
        scene_id=scene_id, type=vtype, primary_subject=primary, secondary_subject=sec, action=action, context=context,
        preferred_visuals=preferred, avoid=list(_AVOID.get(vtype, [])), secondary_types=secondary,
        type_scores=scores, confidence=conf, inherited_from=inherited_from, author=Origin.AI,
    )
