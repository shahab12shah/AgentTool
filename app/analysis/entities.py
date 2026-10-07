"""Rule-based entity extraction: gazetteer + capitalisation heuristics (no model)."""

from __future__ import annotations

from app.analysis.lexicon import FIRST_NAMES, GAZETTEER, HONORIFICS, MAX_PHRASE, ORG_SUFFIXES, STRICT_CASE
from app.analysis.models import Entity, EntityType
from app.analysis.prep import AToken


def _is_cap(tok: AToken) -> bool:
    return tok.text[:1].isupper()


def extract_entities(tokens: list[AToken]) -> list[Entity]:
    """Longest-match gazetteer lookup, then capitalised proper-noun runs (not sentence-initial common words)."""
    found: dict[str, Entity] = {}
    covered = [False] * len(tokens)

    def add(text: str, etype: EntityType, ids: list[str]) -> None:
        key = text.lower()
        if key in found:
            found[key].mentions += 1
            found[key].word_ids += ids
        else:
            found[key] = Entity(text, etype, canonical=key, word_ids=list(ids))

    i = 0
    while i < len(tokens):
        matched = False
        for n in range(min(MAX_PHRASE, len(tokens) - i), 0, -1):
            key = tuple(t.norm for t in tokens[i : i + n])
            etype = GAZETTEER.get(key)
            if etype is None or any(t.spoken_number for t in tokens[i : i + n]):
                continue
            if n == 1 and key[0] in STRICT_CASE and not _is_cap(tokens[i]):
                continue  # "apple"/"target" as ordinary words
            text = " ".join(t.text.strip(".,;:!?\"'") for t in tokens[i : i + n])
            add(_display(text, etype), etype, [w for t in tokens[i : i + n] for w in t.word_ids])
            for k in range(i, i + n):
                covered[k] = True
            i += n
            matched = True
            break
        if not matched:
            i += 1

    # capitalised runs not covered by the gazetteer
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if covered[i] or not _is_cap(t) or t.spoken_number or t.norm.isdigit():
            i += 1
            continue
        j = i
        while j < len(tokens) and not covered[j] and _is_cap(tokens[j]) and not tokens[j].spoken_number:
            j += 1
        run = tokens[i:j]
        if i == 0 and len(run) == 1 and t.norm not in FIRST_NAMES:
            i = j  # a lone capitalised first word is just the start of the sentence
            continue
        if i == 0 and len(run) > 1 and run[0].norm in _SENTENCE_STARTERS:
            run = run[1:]
        if run:
            text = " ".join(r.text.strip(".,;:!?\"'") for r in run)
            norms = [r.norm for r in run]
            prev = tokens[i - 1].norm if i > 0 else ""
            if norms[-1] in ORG_SUFFIXES:
                etype = EntityType.ORGANIZATION
            elif prev in HONORIFICS or norms[0] in HONORIFICS or (len(run) >= 2 and norms[0] in FIRST_NAMES):
                etype = EntityType.PERSON
            elif len(run) == 1 and norms[0] in FIRST_NAMES:
                etype = EntityType.PERSON
            else:
                etype = EntityType.OTHER
            add(text, etype, [w for r in run for w in r.word_ids])
        i = j
    return list(found.values())


_SENTENCE_STARTERS = {"the", "a", "an", "this", "that", "these", "those", "now", "so", "and", "but", "when", "while", "if", "in", "on",
                      "at", "it", "its", "they", "we", "you", "i", "he", "she", "as", "for", "with", "after", "before", "today"}


def _display(text: str, etype: EntityType) -> str:
    """Upper-case short acronym agencies/tickers; otherwise keep script casing or title-case."""
    if text.isupper() or any(c.isupper() for c in text):
        return text
    if etype is EntityType.GOVERNMENT_AGENCY and len(text) <= 5:
        return text.upper()
    if etype in (EntityType.COUNTRY, EntityType.CITY, EntityType.COMPANY, EntityType.GOVERNMENT_AGENCY):
        return text.title()
    return text.lower() if etype is EntityType.OBJECT else text


def gazetteer_spans(tokens: list[AToken]) -> list[tuple[int, int]]:
    """Token spans ``[i, j)`` of known (gazetteer) terms, longest match first. Used for enumeration detection."""
    spans, i = [], 0
    while i < len(tokens):
        for n in range(min(MAX_PHRASE, len(tokens) - i), 0, -1):
            key = tuple(t.norm for t in tokens[i : i + n])
            if key in GAZETTEER and not any(t.spoken_number for t in tokens[i : i + n]) \
                    and not (n == 1 and key[0] in STRICT_CASE and not _is_cap(tokens[i])):
                spans.append((i, i + n))
                i += n
                break
        else:
            i += 1
    return spans
