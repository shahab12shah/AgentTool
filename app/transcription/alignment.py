"""Aligns the user's script to the words actually spoken. Never modifies either side.

Output is a list of ``AlignmentItem`` (in transcript order, script-only items interleaved at
their script position) saying, for every word: matched / approximately matched / only in
the script (missing from the audio) / only in the transcript (added) / reordered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.core.textutil import STOPWORDS, normalize_token, parse_number_run, split_sentences, stem, tokenize
from app.transcription.models import Transcript, Word


class AlignStatus(str, Enum):
    MATCHED = "MATCHED"
    APPROXIMATE = "APPROXIMATE"
    SCRIPT_ONLY = "SCRIPT_ONLY"  # in the script, not spoken
    TRANSCRIPT_ONLY = "TRANSCRIPT_ONLY"  # spoken, not in the script
    REORDERED = "REORDERED"  # present on both sides but in a different place


class Verdict(str, Enum):
    IDENTICAL = "IDENTICAL"
    MINOR_DIFFERENCES = "MINOR_DIFFERENCES"
    SIGNIFICANT_DIFFERENCES = "SIGNIFICANT_DIFFERENCES"


@dataclass
class AlignmentItem:
    status: AlignStatus
    script_text: str | None = None
    script_index: int | None = None  # index of the (first) script token
    script_sentence: int | None = None
    word_ids: list[str] = field(default_factory=list)
    transcript_text: str | None = None
    similarity: float = 0.0


@dataclass
class AlignmentStats:
    script_words: int = 0
    transcript_words: int = 0
    matched: int = 0
    approximate: int = 0
    script_only: int = 0
    transcript_only: int = 0
    reordered: int = 0
    coverage: float = 0.0  # share of script words that were found in the audio
    similarity: float = 0.0


@dataclass
class DifferenceRegion:
    """A run of consecutive mismatching words long enough to matter."""

    item_start: int
    item_end: int  # exclusive
    script_text: str
    transcript_text: str
    start: float | None = None  # seconds in the audio, when known
    end: float | None = None


@dataclass
class ScriptAlignment:
    script_hash: str
    transcript_id: str
    items: list[AlignmentItem]
    stats: AlignmentStats
    verdict: Verdict
    regions: list[DifferenceRegion] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScriptAlignment":
        return from_plain(cls, d)

    def script_sentence_of_word(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.items:
            if it.status in (AlignStatus.MATCHED, AlignStatus.APPROXIMATE) and it.script_sentence is not None:
                for wid in it.word_ids:
                    out[wid] = it.script_sentence
        return out

    def surface_by_word(self) -> dict[str, str]:
        """script spelling/casing/punctuation for matched words (single-word items only)."""
        return {
            it.word_ids[0]: it.script_text
            for it in self.items
            if it.status in (AlignStatus.MATCHED, AlignStatus.APPROXIMATE)
            and len(it.word_ids) == 1 and it.script_text
        }


def script_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ tokens
@dataclass
class _Tok:
    norm: str
    text: str  # surface text (joined for merged number runs)
    ids: list[int]  # script token indexes OR word indexes
    sentence: int | None = None


def _merge_numbers(norms: list[str], surfaces: list[str]) -> list[_Tok]:
    out: list[_Tok] = []
    i = 0
    while i < len(norms):
        run = parse_number_run(norms, i)
        if run:
            out.append(_Tok(run.text, " ".join(surfaces[i : i + run.consumed]), list(range(i, i + run.consumed))))
            i += run.consumed
        else:
            out.append(_Tok(norms[i], surfaces[i], [i]))
            i += 1
    return out


def _script_tokens(script: str) -> tuple[list[_Tok], list[str]]:
    sent_ranges = [(a, b) for _t, a, b in split_sentences(script)]
    toks = tokenize(script)
    norms = [t.norm for t in toks]
    surfaces = [t.text for t in toks]
    merged = _merge_numbers(norms, surfaces)
    for m in merged:
        pos = toks[m.ids[0]].start
        m.sentence = next((k for k, (a, b) in enumerate(sent_ranges) if a <= pos < b + 1), len(sent_ranges) - 1)
    return merged, surfaces


def _transcript_tokens(words: list[Word]) -> list[_Tok]:
    norms = [normalize_token(w.text) or w.text.lower() for w in words]
    return _merge_numbers(norms, [w.text for w in words])


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if stem(a) == stem(b) and len(a) >= 3:
        return 0.9
    if min(len(a), len(b)) < 4:
        return 0.0
    r = SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.75 else 0.0


# ------------------------------------------------------------------ alignment
def _align_gap(s: list[_Tok], t: list[_Tok], si: int, ti: int) -> list[tuple[int | None, int | None, float]]:
    """Needleman-Wunsch on a small gap. Substitutions are only allowed between similar words."""
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return [(si + i, None, 0.0) for i in range(n)] + [(None, ti + j, 0.0) for j in range(m)]
    if n * m > 60_000:  # pathological gap: no pairing attempt
        return [(si + i, None, 0.0) for i in range(n)] + [(None, ti + j, 0.0) for j in range(m)]
    GAP = -0.4
    sim = [[_similar(s[i].norm, t[j].norm) for j in range(m)] for i in range(n)]
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * GAP
    for j in range(1, m + 1):
        score[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(score[i - 1][j] + GAP, score[i][j - 1] + GAP)
            if sim[i - 1][j - 1] > 0:
                best = max(best, score[i - 1][j - 1] + sim[i - 1][j - 1])
            score[i][j] = best
    out: list[tuple[int | None, int | None, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and sim[i - 1][j - 1] > 0 and abs(score[i][j] - (score[i - 1][j - 1] + sim[i - 1][j - 1])) < 1e-9:
            out.append((si + i - 1, ti + j - 1, sim[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and abs(score[i][j] - (score[i - 1][j] + GAP)) < 1e-9:
            out.append((si + i - 1, None, 0.0))
            i -= 1
        else:
            out.append((None, ti + j - 1, 0.0))
            j -= 1
    out.reverse()
    return out


MIN_REGION_ITEMS = 5


def align_script(script: str, transcript: Transcript) -> ScriptAlignment:
    s_toks, _ = _script_tokens(script)
    t_toks = _transcript_tokens(transcript.words)
    words = transcript.words

    pairs: list[tuple[int | None, int | None, float]] = []
    matcher = SequenceMatcher(None, [x.norm for x in s_toks], [x.norm for x in t_toks], autojunk=False)
    si = ti = 0
    for block in matcher.get_matching_blocks():
        pairs += _align_gap(s_toks[si : block.a], t_toks[ti : block.b], si, ti)
        pairs += [(block.a + k, block.b + k, 1.0) for k in range(block.size)]
        si, ti = block.a + block.size, block.b + block.size

    # reorder pass: unmatched script tokens that were spoken elsewhere
    unmatched_t: dict[str, list[int]] = {}
    for idx, (a, b, _) in enumerate(pairs):
        if a is None and b is not None and t_toks[b].norm not in STOPWORDS and len(t_toks[b].norm) >= 3:
            unmatched_t.setdefault(t_toks[b].norm, []).append(idx)
    drop: set[int] = set()
    reordered: dict[int, int] = {}  # transcript-only pair idx -> script token index
    for idx, (a, b, _) in enumerate(pairs):
        if b is None and a is not None and s_toks[a].norm in unmatched_t and unmatched_t[s_toks[a].norm]:
            target = unmatched_t[s_toks[a].norm].pop(0)
            reordered[target] = a
            drop.add(idx)

    items: list[AlignmentItem] = []
    for idx, (a, b, sim) in enumerate(pairs):
        if idx in drop:
            continue
        wids = [words[k].word_id for k in t_toks[b].ids] if b is not None else []
        ttext = t_toks[b].text if b is not None else None
        if idx in reordered:
            sa = s_toks[reordered[idx]]
            items.append(AlignmentItem(AlignStatus.REORDERED, sa.text, sa.ids[0], sa.sentence, wids, ttext, 1.0))
        elif a is not None and b is not None:
            sa = s_toks[a]
            status = AlignStatus.MATCHED if sim >= 1.0 else AlignStatus.APPROXIMATE
            items.append(AlignmentItem(status, sa.text, sa.ids[0], sa.sentence, wids, ttext, sim))
        elif a is not None:
            sa = s_toks[a]
            items.append(AlignmentItem(AlignStatus.SCRIPT_ONLY, sa.text, sa.ids[0], sa.sentence))
        else:
            items.append(AlignmentItem(AlignStatus.TRANSCRIPT_ONLY, None, None, None, wids, ttext))

    stats = _stats(items, len(s_toks), len(t_toks))
    regions = _regions(items, transcript)
    if stats.script_words == stats.matched and stats.transcript_only == 0:
        verdict = Verdict.IDENTICAL
    elif stats.coverage >= 0.9 and not regions:
        verdict = Verdict.MINOR_DIFFERENCES
    else:
        verdict = Verdict.SIGNIFICANT_DIFFERENCES
    return ScriptAlignment(script_hash(script), transcript.transcript_id, items, stats, verdict, regions)


def _stats(items: list[AlignmentItem], n_script: int, n_trans: int) -> AlignmentStats:
    c = {s: sum(1 for i in items if i.status is s) for s in AlignStatus}
    found = c[AlignStatus.MATCHED] + c[AlignStatus.APPROXIMATE] + c[AlignStatus.REORDERED]
    total = max(n_script, n_trans, 1)
    return AlignmentStats(
        script_words=n_script, transcript_words=n_trans,
        matched=c[AlignStatus.MATCHED], approximate=c[AlignStatus.APPROXIMATE],
        script_only=c[AlignStatus.SCRIPT_ONLY], transcript_only=c[AlignStatus.TRANSCRIPT_ONLY],
        reordered=c[AlignStatus.REORDERED],
        coverage=found / n_script if n_script else 0.0,
        similarity=(c[AlignStatus.MATCHED] + 0.8 * c[AlignStatus.APPROXIMATE] + 0.6 * c[AlignStatus.REORDERED]) / total,
    )


def _regions(items: list[AlignmentItem], transcript: Transcript) -> list[DifferenceRegion]:
    wmap = transcript.word_map()
    bad = (AlignStatus.SCRIPT_ONLY, AlignStatus.TRANSCRIPT_ONLY)
    out: list[DifferenceRegion] = []
    i = 0
    while i < len(items):
        if items[i].status not in bad:
            i += 1
            continue
        j = i
        while j < len(items) and items[j].status in bad:
            j += 1
        if j - i >= MIN_REGION_ITEMS:
            run = items[i:j]
            ids = [w for it in run for w in it.word_ids]
            times = [wmap[w] for w in ids if w in wmap]
            out.append(DifferenceRegion(
                i, j,
                " ".join(it.script_text for it in run if it.script_text),
                " ".join(it.transcript_text for it in run if it.transcript_text),
                times[0].start if times else None, times[-1].end if times else None,
            ))
        i = j
    return out
