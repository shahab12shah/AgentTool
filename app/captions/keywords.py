"""KeywordService: which words deserve emphasis (and which do not).

Sources are the scene analysis (numbers, entities, claims), a small lexicon of warning/deadline words and — optionally — the
acoustic emphasis candidates found in the voice-over. Emphasis is deliberately sparse: a per-scene budget keeps captions calm.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from app.analysis.models import EntityType, NumberKind
from app.presentation.models import KeywordCategory as K

NUMBER_CATEGORY = {NumberKind.DOLLAR_AMOUNT: K.MONEY, NumberKind.PRICE: K.MONEY, NumberKind.PERCENTAGE: K.PERCENTAGE, NumberKind.DATE: K.DATE, NumberKind.YEAR: K.DATE,
                   NumberKind.DEADLINE: K.DEADLINE, NumberKind.QUANTITY: K.NUMBER, NumberKind.AGE: K.NUMBER}
ENTITY_CATEGORY = {EntityType.PERSON: K.PERSON, EntityType.COMPANY: K.ORGANIZATION, EntityType.ORGANIZATION: K.ORGANIZATION, EntityType.GOVERNMENT_AGENCY: K.ORGANIZATION,
                   EntityType.COUNTRY: K.LOCATION, EntityType.CITY: K.LOCATION, EntityType.PRODUCT: K.PRODUCT, EntityType.OBJECT: K.PRODUCT, EntityType.TECHNOLOGY: K.PRODUCT,
                   EntityType.FINANCIAL_INSTRUMENT: K.CONCEPT}
IMPORTANCE = {K.MONEY: 0.95, K.PERCENTAGE: 0.95, K.DEADLINE: 0.9, K.DATE: 0.85, K.WARNING: 0.85, K.NUMBER: 0.8, K.PERSON: 0.7, K.ORGANIZATION: 0.65, K.LOCATION: 0.55,
              K.PRODUCT: 0.5, K.PROCESS: 0.45, K.CLAIM: 0.45, K.CONCEPT: 0.4}
WARN = ("penalty", "penalties", "warning", "fine", "fines", "illegal", "mandatory", "violation", "audit", "risk", "danger")
DEADLINE_WORDS = ("deadline", "due", "expires", "expiry", "expire")
NUMBER_CATEGORIES = {K.MONEY, K.PERCENTAGE, K.NUMBER, K.DATE, K.DEADLINE}


@dataclass
class Keyword:
    category: str
    word_ids: list[str]
    text: str
    importance: float
    reason: str
    source: str = "analysis"  # analysis | lexicon | audio


def _clean(t: str) -> str:
    return re.sub(r"[^\w$%+.,-]", "", t).strip(".,;:!?")


class KeywordService:
    def detect(self, scene, words, audio_emphasis: set[str] | None = None) -> list[Keyword]:
        """Keywords of one scene, best first, within the scene's emphasis budget."""
        by_id = {w.word_id: w for w in words}
        found: list[Keyword] = []
        used: set[str] = set()

        def add(cat: K, ids: list[str], reason: str, source: str = "analysis") -> None:
            ids = [i for i in ids if i in by_id]
            if not ids or any(i in used for i in ids):
                return
            used.update(ids)
            found.append(Keyword(cat.value, ids, " ".join(by_id[i].text for i in ids), IMPORTANCE[cat], reason, source))

        for n in scene.numbers:
            add(NUMBER_CATEGORY.get(n.kind, K.NUMBER), list(n.word_ids), f"{n.kind.value.replace('_', ' ').title()} “{n.text}” is stated in the narration.")
        for e in scene.entities:
            cat = ENTITY_CATEGORY.get(e.type)
            if cat is not None:
                add(cat, list(e.word_ids[:1]) if len(e.word_ids) > 1 and e.type is not EntityType.PERSON else list(e.word_ids[:3]), f"{e.text} is a named {cat.value.lower()}.")
        for w in words:
            c = _clean(w.text).lower()
            if c in WARN:
                add(K.WARNING, [w.word_id], f"The narration warns about “{c}”.", "lexicon")
            elif c in DEADLINE_WORDS:
                add(K.DEADLINE, [w.word_id], f"“{c}” marks a deadline.", "lexicon")
        for wid in sorted(audio_emphasis or (), key=lambda i: by_id[i].start if i in by_id else 0):
            w = by_id.get(wid)
            if w is not None:
                add(K.CONCEPT, [wid], "Spoken with noticeably more emphasis.", "audio")
        budget = max(1, math.ceil(len(words) * 0.08) + 1)
        found.sort(key=lambda k: -k.importance)
        return found[:budget]


@dataclass
class KeywordPlan:
    scene_id: str
    keywords: list[Keyword] = field(default_factory=list)
