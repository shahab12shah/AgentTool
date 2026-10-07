"""Claim extraction. Claims are labelled, never verified: no evidence is ever invented."""

from __future__ import annotations

from app.analysis.lexicon import CUES, QUESTION_STARTS
from app.analysis.models import Claim, ClaimType, NumberKind, NumericMention
from app.analysis.prep import AToken
from app.core.textutil import stem

_NO_EVIDENCE = {ClaimType.OPINION, ClaimType.QUESTION}
IMPERATIVE_STARTS = {"consider", "remember", "imagine", "look", "think", "let", "listen", "watch", "subscribe", "click", "stay", "keep", "make"}
_SKIP_FILLER = {"welcome", "subscribe", "thanks", "thank", "hello", "hi", "hey", "like", "comment", "bell"}


def _count(stems: list[str], cue: str) -> int:
    return sum(1 for s in stems if s in CUES[cue])


def classify_claim(tokens: list[AToken], numbers: list[NumericMention], text: str) -> tuple[ClaimType, str | None] | None:
    """Return (type, domain) for a sentence that makes a checkable assertion, or None for filler."""
    norms = [t.norm for t in tokens]
    stems = [stem(n) for n in norms]
    if len(norms) < 3 or sum(1 for n in norms if n in _SKIP_FILLER) >= 2:
        return None
    domain = "market" if _count(stems, "domain_market") >= 2 else "legal" if _count(stems, "domain_legal") >= 2 else None
    first = norms[0]
    lead = [n for n in norms[:3] if n not in ("next", "now", "so", "and", "but", "okay", "first", "finally")]
    if lead and lead[0] in IMPERATIVE_STARTS and not text.rstrip().endswith("?"):
        return None  # an instruction or discourse move, not a checkable assertion
    if text.rstrip().endswith("?") or (first in QUESTION_STARTS and first in ("what", "why", "how", "who", "when", "where", "which")):
        return ClaimType.QUESTION, domain
    if text.rstrip().endswith("?") is False and first in QUESTION_STARTS and len(norms) > 3 and norms[1] in ("you", "we", "they", "it", "this") \
            and first in ("is", "are", "do", "does", "did", "can", "could", "should", "would", "will", "have", "has"):
        return ClaimType.QUESTION, domain
    if any(p in " ".join(norms) for p in ("i think", "i believe", "in my opinion", "i feel", "in my view")):
        return ClaimType.OPINION, domain
    if '"' in text or "according" in norms or any(n in ("said", "told", "stated", "wrote", "quoted") for n in norms):
        return ClaimType.QUOTE, domain
    if _count(stems, "law") >= 1:
        return ClaimType.LAW, "legal"
    if _count(stems, "rule") >= 1 and ("must" in norms or "required" in norms or "deadline" in norms or "eligible" in norms or _count(stems, "rule") >= 2):
        return ClaimType.RULE, domain or "legal"
    if _count(stems, "prediction") >= 1 and any(n in ("will", "could", "may", "might", "expected", "forecast", "projected", "likely", "predict") for n in norms):
        return ClaimType.PREDICTION, domain
    if _count(stems, "opinion") >= 1 and any(n in ("should", "best", "worst", "terrible", "great", "amazing", "better", "worse") for n in norms):
        return ClaimType.OPINION, domain
    kinds = {n.kind for n in numbers}
    if kinds & {NumberKind.DATE, NumberKind.YEAR, NumberKind.DEADLINE} and not kinds & {
        NumberKind.PRICE, NumberKind.PERCENTAGE, NumberKind.DOLLAR_AMOUNT, NumberKind.QUANTITY}:
        return ClaimType.DATE, domain
    if kinds & {NumberKind.PRICE, NumberKind.PERCENTAGE, NumberKind.DOLLAR_AMOUNT, NumberKind.QUANTITY, NumberKind.AGE}:
        return ClaimType.NUMBER, domain
    if len(norms) < 5:
        return None  # too short to be a meaningful assertion ("Moving on to housing.")
    return ClaimType.FACT, domain


def build_claim(sentence_id: str, index: int, text: str, tokens: list[AToken], numbers: list[NumericMention]) -> Claim | None:
    result = classify_claim(tokens, numbers, text)
    if result is None:
        return None
    ctype, domain = result
    return Claim(f"{sentence_id}_c{index}", text.strip(), ctype, sentence_id, ctype not in _NO_EVIDENCE, domain)
