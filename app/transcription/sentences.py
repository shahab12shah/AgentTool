"""Builds sentences from words. Three strategies, best available first:

``script-guided``  boundaries follow the user's script sentences (via alignment);
``punctuation``    the provider returned sentence-final punctuation;
``pause``          silence between words (plus a length cap) for unpunctuated output.
"""

from __future__ import annotations

from app.transcription.models import Sentence, Word

TERMINATORS = (".", "?", "!", "…")
ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "inc.", "corp.", "ltd.", "u.s.", "e.g.", "i.e."}
PAUSE_SECONDS = 0.6
MAX_WORDS = 40
MIN_WORDS_FOR_PAUSE_SPLIT = 3


def has_punctuation(words: list[Word]) -> bool:
    ends = sum(1 for w in words if w.text.rstrip("\"')").endswith(TERMINATORS) and w.text.lower() not in ABBREV)
    return ends >= max(1, len(words) // 80)


def _make(index: int, chunk: list[Word]) -> Sentence:
    confs = [w.confidence for w in chunk if w.confidence is not None]
    return Sentence(
        sentence_id=f"sent_{index:04d}",
        text=" ".join(w.text for w in chunk),
        start=chunk[0].start,
        end=chunk[-1].end,
        word_ids=[w.word_id for w in chunk],
        confidence=sum(confs) / len(confs) if confs else None,
    )


def _from_boundaries(words: list[Word], cut_after: set[int]) -> list[Sentence]:
    sentences, chunk = [], []
    for i, w in enumerate(words):
        chunk.append(w)
        if i in cut_after:
            sentences.append(_make(len(sentences), chunk))
            chunk = []
    if chunk:
        sentences.append(_make(len(sentences), chunk))
    return sentences


def by_punctuation(words: list[Word]) -> list[Sentence]:
    cuts = {
        i for i, w in enumerate(words)
        if w.text.rstrip("\"')").endswith(TERMINATORS) and w.text.lower() not in ABBREV
    }
    return _from_boundaries(words, cuts)


def by_pauses(words: list[Word], pause: float = PAUSE_SECONDS, max_words: int = MAX_WORDS) -> list[Sentence]:
    cuts: set[int] = set()
    run = 0
    best_gap, best_i = -1.0, -1  # widest gap seen in the current run (for forced splits)
    for i in range(len(words) - 1):
        run += 1
        gap = words[i + 1].start - words[i].end
        if gap > best_gap:
            best_gap, best_i = gap, i
        if gap >= pause and run >= MIN_WORDS_FOR_PAUSE_SPLIT:
            cuts.add(i)
            run, best_gap, best_i = 0, -1.0, -1
        elif run >= max_words:
            cuts.add(best_i if best_i >= 0 else i)
            run, best_gap, best_i = i - best_i if best_i >= 0 else 0, -1.0, -1
    return _from_boundaries(words, cuts)


def by_script_sentences(words: list[Word], script_sentence_of_word: dict[str, int]) -> list[Sentence]:
    """Cut where the aligned script sentence index changes.

    Words the alignment could not place (added/misrecognised) sit between two aligned words;
    the cut goes at the widest pause among them so they join the sentence they belong to.
    """
    cuts: set[int] = set()
    current: int | None = None
    last_aligned = -1
    for i, w in enumerate(words):
        idx = script_sentence_of_word.get(w.word_id)
        if idx is None:
            continue
        if current is not None and idx != current:
            candidates = range(last_aligned, i)  # cut after word j
            cuts.add(max(candidates, key=lambda j: (words[j + 1].start - words[j].end, -abs(j - (i - 1)))))
        current = idx
        last_aligned = i
    return _from_boundaries(words, cuts)


def build_sentences(words: list[Word], script_sentence_of_word: dict[str, int] | None = None) -> tuple[list[Sentence], str]:
    """Pick the best strategy. Returns ``(sentences, strategy_name)``."""
    if not words:
        return [], "pause"
    if script_sentence_of_word and len(set(script_sentence_of_word.values())) > 1:
        return by_script_sentences(words, script_sentence_of_word), "script-guided"
    if has_punctuation(words):
        return by_punctuation(words), "punctuation"
    return by_pauses(words), "pause"
