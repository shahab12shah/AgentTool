"""CaptionEngine: word/sentence timestamps -> readable caption segments.

Segmentation is a small dynamic programme per sentence, not fixed-length chunks: it prefers breaks at punctuation, pauses and phrase
starts, never splits glued tokens (a date, a number and its unit), keeps reading speed and line length within limits, and only ever
starts or ends a caption on a word boundary. Spoken wording is never changed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.captions.keywords import NUMBER_CATEGORIES, Keyword
from app.captions.styles import effective_style, style_for
from app.presentation.models import CaptionSegment, CaptionSettings, CaptionStyle, CaptionWord, EmphasisMark

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
UNITS = {"percent", "percentage", "dollars", "dollar", "cents", "million", "billion", "trillion", "thousand", "hundred", "years", "year", "months", "days", "ounces",
         "ounce", "tons", "ton", "pounds", "euros", "times", "x"}
TITLES = {"mr", "mrs", "ms", "dr", "prof", "sen", "rep", "gov", "president", "secretary"}
FUNCTION = {"the", "a", "an", "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "and", "or", "over", "across", "about", "into", "than", "under", "between", "during", "after", "before", "through", "while", "because", "but", "that", "this", "its", "their", "which", "who", "whom", "whose", "where", "when", "if", "so", "not"}
PHRASE_START = {"and", "but", "because", "which", "that", "who", "while", "when", "if", "so", "or", "although", "since", "where", "until", "unless", "however"}
MAX_CPS = 17.0  # characters per second a viewer can comfortably read
MAX_WPS = 3.6
MIN_DUR, MAX_DUR = 0.9, 6.5
GAP_PAUSE = 0.35
LONG_PAUSE = 0.9
AVG_CHAR = 0.52  # average glyph width / font height


def _strip(t: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", t).lower()


@dataclass
class Layout:
    chars_per_line: int
    max_chars: int
    max_lines: int
    font_px: float
    box_width: float  # px available between the safe margins


def layout_for(settings: CaptionSettings, style: CaptionStyle, canvas: tuple[int, int]) -> Layout:
    W, H = canvas
    avail = W * (1.0 - settings.safe_margin_left - settings.safe_margin_right)
    px = max(10.0, style.size_rel * H)
    char_w = px * AVG_CHAR * (1.08 if style.weight == "bold" else 1.0) * (1.12 if style.uppercase else 1.0)
    cpl = max(12, int(avail * 0.92 // char_w))  # 8% slack for the background box padding
    return Layout(cpl, cpl * max(1, settings.max_lines), max(1, settings.max_lines), px, avail)


def glued(prev_text: str, next_text: str) -> bool:
    """True when a caption must not be split between these two words."""
    a, b = _strip(prev_text), _strip(next_text)
    if not a or not b:
        return False
    if a in MONTHS and re.fullmatch(r"\d{1,2}(st|nd|rd|th)?", b):
        return True
    if re.fullmatch(r"[$€£]?\d[\d,.]*%?\+?", a) and (b in UNITS or b == "%"):
        return True
    if a in TITLES:
        return True
    if a in ("of", "in") and re.fullmatch(r"\d{4}", b):
        return False
    if re.fullmatch(r"\d{1,2}(st|nd|rd|th)?", a) and b in MONTHS:
        return True
    if a in ("united", "new", "north", "south", "east", "west") and b[:1].isalpha():
        return a == "united" and b in ("states", "kingdom", "nations", "arab")
    return False


def boundary_cost(words: list[CaptionWord], k: int) -> float:
    """Cost of ending a caption after ``words[k]`` (lower = a more natural break)."""
    cur = words[k]
    if k + 1 >= len(words):
        return 0.0
    nxt = words[k + 1]
    if glued(cur.text, nxt.text):
        return 100.0
    cost = 2.0
    if re.search(r"[.?!]['\")]*$", cur.text):
        cost = 0.0
    elif re.search(r"[,;:—–-]['\")]*$", cur.text):
        cost = 0.3
    elif nxt.start - cur.end >= GAP_PAUSE:
        cost = 0.4
    elif _strip(nxt.text) in PHRASE_START:
        cost = 0.9
    if _strip(cur.text) in FUNCTION:
        cost += 3.0  # never leave "the", "of", "to" hanging at the end of a caption
    return cost


def break_lines(words: list[CaptionWord], layout: Layout) -> list[str]:
    """One or two lines with natural phrase grouping (``SILVER IS / RUNNING OUT``)."""
    text = " ".join(w.text for w in words)
    if len(text) <= layout.chars_per_line or layout.max_lines < 2 or len(words) < 2:
        return [text]
    best, best_cost = None, math.inf
    for k in range(len(words) - 1):
        l1, l2 = " ".join(w.text for w in words[:k + 1]), " ".join(w.text for w in words[k + 1:])
        cost = abs(len(l1) - len(l2)) * 0.4
        if len(l1) > layout.chars_per_line or len(l2) > layout.chars_per_line:
            cost += 50.0 + max(len(l1), len(l2)) - layout.chars_per_line
        cost += boundary_cost(words, k) * 2.0
        if k == 0 or k == len(words) - 2:
            cost += 3.0  # a one-word line looks like a list
        if cost < best_cost:
            best, best_cost = (l1, l2), cost
    return list(best) if best else [text]


class CaptionEngine:
    def __init__(self, settings: CaptionSettings, styles: dict[str, CaptionStyle], canvas: tuple[int, int]) -> None:
        self.settings, self.styles, self.canvas = settings, styles, canvas
        self.style = effective_style(style_for(styles, settings.style_id), settings)
        self.layout = layout_for(settings, self.style, canvas)
        self.cps = MAX_CPS * max(0.3, settings.reading_speed)
        self.wps = MAX_WPS * max(0.3, settings.reading_speed)

    # ------------------------------------------------------------------ segmentation
    def _split_sentence(self, words: list[CaptionWord]) -> list[list[CaptionWord]]:
        n = len(words)
        lay, cfg = self.layout, self.settings
        INF = math.inf
        best = [INF] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                seg = words[i:j]
                chars = sum(len(w.text) for w in seg) + len(seg) - 1
                dur = seg[-1].end - seg[0].start
                if (chars > lay.max_chars or dur > MAX_DUR or len(seg) > cfg.max_words) and len(seg) > 1:
                    break  # extending further left only makes it longer
                if len(seg) > 1 and seg[1].start - seg[0].end > LONG_PAUSE:
                    break  # a long silence always starts a new caption
                cost = 1.0 + boundary_cost(words, j - 1) + (0.0 if j == n else 0.0)
                if len(seg) < 3 and n >= 3:
                    cost += 2.0
                if dur < MIN_DUR and n > len(seg):
                    cost += 1.0
                cps = chars / max(dur, 0.25)
                wps = len(seg) / max(dur, 0.25)
                cost += max(0.0, cps - self.cps) * 0.8 + max(0.0, wps - self.wps) * 1.2
                if cps > self.cps * 1.15 and chars > lay.chars_per_line and len(seg) > 1:
                    continue  # dense speech: only one-line captions (easier to take in)
                if i > 0:
                    cost += 0.0
                if best[i] + cost < best[j]:
                    best[j], back[j] = best[i] + cost, i
        out, j = [], n
        while j > 0:
            i = back[j]
            out.append(words[i:j])
            j = i
        return out[::-1]

    def segment(self, scene_id: str, sentences: list[list[CaptionWord]], keywords: list[Keyword] | None = None, id_prefix: str = "cap",
                scene_end: float | None = None, start_n: int = 0) -> list[CaptionSegment]:
        """Caption segments for the given sentences (each a list of timed words). ``keywords`` assign emphasis."""
        groups: list[list[CaptionWord]] = []
        for s in sentences:
            if s:
                groups += self._split_sentence(s)
        segs: list[CaptionSegment] = []
        kw_by_word: dict[str, Keyword] = {}
        for k in keywords or []:
            for wid in k.word_ids:
                kw_by_word.setdefault(wid, k)
        for n, g in enumerate(groups, start=start_n):
            chars = sum(len(w.text) for w in g) + len(g) - 1
            start, end = g[0].start, g[-1].end
            seg = CaptionSegment(f"{id_prefix}_{n + 1:04d}", scene_id, round(start, 3), round(end, 3), " ".join(w.text for w in g), break_lines(g, self.layout),
                                 [CaptionWord(w.word_id, w.text, round(w.start, 3), round(w.end, 3)) for w in g], [], self.settings.style_id, {}, self.settings.position,
                                 [self.settings.custom_x, self.settings.custom_y] if self.settings.position == "custom" else [], self._animation(),
                                 self.settings.highlight_mode, round(chars / max(end - start, 0.25), 2))
            seg.emphasis = self._emphasis(seg, kw_by_word)
            segs.append(seg)
        self._timing(segs, scene_end)
        return segs

    def _animation(self) -> dict:
        return {"in": {"preset": "fade_in", "duration": 0.12, "easing": "ease_out", "delay": 0.0, "start_value": 0.0, "end_value": 1.0},
                "out": {"preset": "fade_out", "duration": 0.10, "easing": "ease_in", "delay": 0.0, "start_value": 1.0, "end_value": 0.0}}

    def _timing(self, segs: list[CaptionSegment], scene_end: float | None) -> None:
        """Hold each caption long enough to read (never past the next caption, never before its first word)."""
        for i, s in enumerate(segs):
            nxt = segs[i + 1].start if i + 1 < len(segs) else (scene_end if scene_end is not None else s.end + 2.0)
            need = max(MIN_DUR, len(s.text) / self.cps)
            end = max(s.end + 0.12, s.start + need)
            s.end = round(max(s.end, min(end, nxt - 0.04, s.end + 1.2)), 3)
            if scene_end is not None:
                s.end = round(min(s.end, scene_end), 3)
            s.reading_cps = round(len(s.text) / max(s.end - s.start, 0.25), 2)

    # ------------------------------------------------------------------ emphasis
    def _emphasis(self, seg: CaptionSegment, kw: dict[str, Keyword]) -> list[EmphasisMark]:
        cfg = self.settings
        marks: list[EmphasisMark] = []
        seen: set[int] = set()
        for idx, w in enumerate(seg.words):
            k = kw.get(w.word_id)
            if k is None:
                continue
            is_num = k.category in {c.value for c in NUMBER_CATEGORIES}
            if (is_num and not cfg.number_emphasis) or (not is_num and not cfg.keyword_highlight):
                continue
            if idx in seen:
                continue
            style = self.style.emphasis.get(k.category, "COLOR_CHANGE")
            marks.append(EmphasisMark(idx, k.category, style, k.reason))
            seen.add(idx)
        marks.sort(key=lambda m: -next((x.importance for x in kw.values() if x.category == m.category), 0.5))
        return sorted(marks[:2], key=lambda m: m.word_index)  # at most two emphasised words per caption


def retime_words(old: list[CaptionWord], new_text: str) -> list[CaptionWord]:
    """Words for an edited caption text. Same word count: the timings are kept. Otherwise the caption's time span is shared out in
    proportion to word length (the user changed what is shown, so the spoken timing can only be approximated)."""
    toks = new_text.split()
    if not toks:
        return []
    if len(toks) == len(old):
        return [CaptionWord(o.word_id, t, o.start, o.end) for o, t in zip(old, toks)]
    a, b = (old[0].start, old[-1].end) if old else (0.0, 1.0)
    total = sum(len(t) + 1 for t in toks)
    out, t0 = [], a
    for i, tok in enumerate(toks):
        t1 = t0 + (b - a) * (len(tok) + 1) / total
        out.append(CaptionWord(f"edit_{i}", tok, round(t0, 3), round(t1 - 0.02 if i < len(toks) - 1 else t1, 3)))
        t0 = t1
    return out
