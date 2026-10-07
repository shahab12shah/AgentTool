"""Semantic scene segmentation over spoken sentences.

A scene is a span of the voice-over that carries one visual idea. Boundaries come from
*meaning and timing*, never from "one sentence = one scene":

* topic / vocabulary shift (lexical cohesion between neighbouring windows)
* discourse cues ("now", "moving on", questions) vs. continuation cues ("and", "this", "because")
* new entity, new evidence (numbers/quotes/laws) or a change of visual requirement
* the length of the pause in the audio
* enumerations inside one sentence ("A, B and C") may be split into one scene per item

Scenes tile the whole voice-over: scene[i].end == scene[i+1].start exactly.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.analysis.models import EntityType, SentenceAnalysis
from app.transcription.models import Transcript

MAJOR_ENTITY = {EntityType.PERSON, EntityType.COMPANY, EntityType.ORGANIZATION, EntityType.COUNTRY, EntityType.CITY,
                EntityType.GOVERNMENT_AGENCY, EntityType.PRODUCT, EntityType.TECHNOLOGY, EntityType.FINANCIAL_INSTRUMENT}


@dataclass
class SegmentationParams:
    threshold: float = 0.6  # boundary score needed to start a new scene (lower = more scenes)
    min_scene_seconds: float = 2.0
    max_scene_seconds: float = 22.0
    window: int = 2  # sentences on each side used for cohesion
    split_enumerations: bool = True
    min_item_seconds: float = 1.0
    review_threshold: float = 0.6  # scenes below this segmentation confidence need review


@dataclass
class SceneDraft:
    start: float
    end: float
    sentence_ids: list[str]
    rationale: list[str] = field(default_factory=list)
    boundary_conf: float = 0.8
    kind: str = "semantic"  # semantic | enumeration


# ---------------------------------------------------------------- vector helpers
def idf_table(analyses: list[SentenceAnalysis]) -> dict[str, float]:
    n = max(len(analyses), 1)
    df: Counter[str] = Counter()
    for a in analyses:
        df.update(set(a.terms))
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}


def _vec(analyses: list[SentenceAnalysis], idf: dict[str, float]) -> Counter[str]:
    v: Counter[str] = Counter()
    for a in analyses:
        for t in a.terms:
            v[t] += idf.get(t, 1.0)
    return v


def _cosine(a: Counter[str], b: Counter[str]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else None


def _dominant(analyses: list[SentenceAnalysis]) -> tuple[str, float]:
    tot: Counter[str] = Counter()
    for a in analyses:
        tot.update(a.type_scores)
    tot.pop("ABSTRACT", None)  # the default type must not count as a "change"
    return tot.most_common(1)[0] if tot else ("", 0.0)


def _has_evidence(a: SentenceAnalysis) -> bool:
    return bool(a.numbers) or any(c.type.value in ("QUOTE", "LAW", "RULE") for c in a.claims)


# ---------------------------------------------------------------- gap scoring
def score_gaps(transcript: Transcript, analyses: dict[str, SentenceAnalysis], params: SegmentationParams) -> tuple[list[float], list[list[str]]]:
    """Boundary score in [0, 1] for the gap after every sentence, with human-readable reasons."""
    sents = transcript.sentences
    seq = [analyses[s.sentence_id] for s in sents]
    idf = idf_table(seq)
    w = params.window
    scores: list[float] = []
    reasons: list[list[str]] = []
    for i in range(len(sents) - 1):
        prev_w, next_w = seq[max(0, i - w + 1) : i + 1], seq[i + 1 : i + 1 + w]
        nxt = seq[i + 1]
        s, why = 0.0, []
        cos = _cosine(_vec(prev_w, idf), _vec(next_w, idf))
        b_lex = 0.4 if cos is None else 1.0 - cos
        s += 0.35 * b_lex
        if b_lex >= 0.75:
            why.append(f"vocabulary shifts (similarity {1 - b_lex:.2f})")
        if "transition_strong" in nxt.cues:
            s += 0.45
            why.append("explicit topic-transition phrase")
        elif "transition_weak" in nxt.cues:
            s += 0.2
            why.append("transition cue")
        if "question" in nxt.cues and "question" not in seq[i].cues:
            s += 0.2
            why.append("a new question begins")
        seen = {e.canonical for a in prev_w for e in a.entities}
        new = [e for e in nxt.entities if e.type in MAJOR_ENTITY and e.canonical not in seen]
        if new:
            s += 0.15 + (0.1 if len(new) > 1 else 0.0)
            why.append("new subject: " + ", ".join(e.text for e in new[:3]))
        if _has_evidence(nxt) and not any(_has_evidence(a) for a in prev_w):
            s += 0.15
            why.append("new evidence (number/quote/rule)")
        (pt, pv), (nt, nv) = _dominant(prev_w), _dominant(next_w)
        if pt and nt and pt != nt and pv >= 0.8 and nv >= 0.8:
            s += 0.10
            why.append(f"visual requirement changes {pt}→{nt}")
        gap = sents[i + 1].start - sents[i].end
        if gap >= 2.0:
            s += 0.25
            why.append(f"{gap:.1f}s pause")
        elif gap >= 1.2:
            s += 0.15
            why.append(f"{gap:.1f}s pause")
        elif gap >= 0.7:
            s += 0.05
        if "continuation" in nxt.cues:
            s -= 0.3
        if "intro" in seq[i].cues:
            s -= 0.3  # the previous sentence only introduces this one
        if "anaphora" in nxt.cues:
            s -= 0.15  # refers back ("it", "this"): likely the same idea
        if nxt.word_count < 4 or seq[i].word_count < 4:
            s -= 0.15
        scores.append(max(0.0, min(1.0, s)))
        reasons.append(why)
    return scores, reasons


def _margin(score: float, threshold: float) -> float:
    """Confidence in a cut/no-cut decision: ~0.5 right at the threshold, 1.0 far from it."""
    return 0.5 + 0.5 * min(1.0, abs(score - threshold) / 0.25)


# ---------------------------------------------------------------- main entry
def segment_transcript(transcript: Transcript, analyses: dict[str, SentenceAnalysis], params: SegmentationParams | None = None) -> list[SceneDraft]:
    params = params or SegmentationParams()
    sents = transcript.sentences
    if not sents:
        return []
    n = len(sents)
    scores, reasons = score_gaps(transcript, analyses, params)
    T = params.threshold
    ranges: list[list[int]] = []  # inclusive sentence-index ranges
    start = 0
    for i, sc in enumerate(scores):
        if sc >= T:
            ranges.append([start, i])
            start = i + 1
    ranges.append([start, n - 1])
    forced: set[int] = set()  # gap indexes where the score was overridden

    def dur(r: list[int]) -> float:
        return sents[r[1]].end - sents[r[0]].start

    # 1. scenes that are too short merge into the neighbour across the weaker boundary
    changed = True
    while changed and len(ranges) > 1:
        changed = False
        for k, r in enumerate(ranges):
            if dur(r) < params.min_scene_seconds:
                left = scores[r[0] - 1] if k > 0 else None
                right = scores[r[1]] if k < len(ranges) - 1 else None
                if right is None or (left is not None and left <= right):
                    gap = r[0] - 1
                    ranges[k - 1][1] = r[1]
                else:
                    gap = r[1]
                    ranges[k + 1][0] = r[0]
                forced.add(gap)
                del ranges[k]
                changed = True
                break

    # 2. scenes that are too long split at their strongest internal boundary
    k = 0
    while k < len(ranges):
        r = ranges[k]
        if dur(r) > params.max_scene_seconds and r[1] > r[0]:
            options = [g for g in range(r[0], r[1])
                       if sents[g].end - sents[r[0]].start >= params.min_scene_seconds
                       and sents[r[1]].end - sents[g + 1].start >= params.min_scene_seconds]
            if options:
                g = max(options, key=lambda x: scores[x])
                ranges[k : k + 1] = [[r[0], g], [g + 1, r[1]]]
                forced.add(g)
                continue
        k += 1

    # 3. drafts with exact, gap-free timing (cuts sit in the middle of the pause between sentences)
    cuts = [(sents[r[1]].end + sents[r[1] + 1].start) / 2 for r in ranges[:-1]]
    bounds = [0.0] + cuts + [max(transcript.audio.duration or 0.0, sents[-1].end)]
    drafts: list[SceneDraft] = []
    for k, r in enumerate(ranges):
        margins: list[float] = []
        why: list[str] = []
        if k > 0:
            g = r[0] - 1
            margins.append(0.4 if g in forced else _margin(scores[g], T))
            why.append("New scene: " + ("; ".join(reasons[g]) if reasons[g] else "boundary score above threshold"))
        else:
            why.append("Start of the voice-over")
        if k < len(ranges) - 1:
            g = r[1]
            margins.append(0.4 if g in forced else _margin(scores[g], T))
        margins += [_margin(scores[g], T) for g in range(r[0], r[1])]
        if r[1] > r[0]:
            why.append(f"{r[1] - r[0] + 1} sentences kept together as one idea")
        if any(g in forced for g in range(r[0] - 1, r[1] + 1) if 0 <= g < len(scores)):
            why.append("adjusted to respect minimum/maximum scene length")
        drafts.append(SceneDraft(bounds[k], bounds[k + 1], [], why, sum(margins) / len(margins) if margins else 0.8))

    if params.split_enumerations:
        drafts = _split_enumerations(transcript, analyses, drafts, params)
    wm = transcript.word_map()
    for d in drafts:
        d.sentence_ids = [s.sentence_id for s in sents
                          if any(d.start <= (wm[w].start + wm[w].end) / 2 < d.end for w in s.word_ids)]
    return drafts


# ---------------------------------------------------------------- enumerations
def _split_enumerations(transcript: Transcript, analyses: dict[str, SentenceAnalysis], drafts: list[SceneDraft],
                        params: SegmentationParams) -> list[SceneDraft]:
    """One sentence may need several visuals: cut inside it at the start of each list item."""
    wm = transcript.word_map()
    order = {w.word_id: i for i, w in enumerate(transcript.words)}
    out: list[SceneDraft] = []
    for d in drafts:
        points: list[tuple[float, str]] = []
        for s in transcript.sentences:
            a = analyses.get(s.sentence_id)
            if not a or len(a.list_items) < 3:
                continue
            for item in a.list_items[1:]:
                lead = wm[item.lead_word_id]
                prev = transcript.words[order[item.lead_word_id] - 1]
                cut = (prev.end + lead.start) / 2
                if d.start < cut < d.end:
                    points.append((cut, item.text))
        if not points:
            out.append(d)
            continue
        kept, names = [d.start], []
        for cut, text in sorted(points):
            if cut - kept[-1] >= params.min_item_seconds and d.end - cut >= params.min_item_seconds:
                kept.append(cut)
                names.append(text)
        kept.append(d.end)
        if len(kept) == 2:
            out.append(d)
            continue
        for j in range(len(kept) - 1):
            why = list(d.rationale) if j == 0 else [f"Split inside a sentence: the narration enumerates “{names[j - 1]}”"]
            out.append(SceneDraft(kept[j], kept[j + 1], [], why, 0.7 if j else min(d.boundary_conf, 0.8), "enumeration"))
    return out
