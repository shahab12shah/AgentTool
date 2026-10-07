"""Rule-based extraction of prices, percentages, amounts, quantities, dates, years, deadlines, ages."""

from __future__ import annotations

import re

from app.analysis.lexicon import CUES
from app.analysis.models import NumberKind, NumericMention
from app.analysis.prep import AToken
from app.core.textutil import stem

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "trillion": 1e12, "bn": 1e9}
_NUM = re.compile(r"^(?P<cur>[$€£])?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<suffix>%|[kKmMbB]|bn)?(?P<plus>\+)?$")
_ORD = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")


def _stem_in(word: str, cue: str) -> bool:
    return stem(word) in CUES[cue]


def extract_numbers(tokens: list[AToken], sentence_id: str) -> list[NumericMention]:
    out: list[NumericMention] = []
    n = len(tokens)
    norms = [t.norm for t in tokens]
    price_ctx = any(_stem_in(w, "price") for w in norms)
    i = 0
    while i < n:
        t = tokens[i]
        text = t.text.strip(".,;:!?\"'()")
        window_before = norms[max(0, i - 4) : i]
        window_after = norms[i + 1 : i + 4]

        # month-based dates: "April 15th", "March 2027", "15 April"
        if t.norm in MONTHS and not (t.norm == "may" and not text[:1].isupper()):
            parts = [text]
            j = i + 1
            if j < n:
                nxt = tokens[j].text.strip(".,;:!?\"'")
                if nxt.isdigit() and len(nxt) <= 2 or _ORD.match(nxt.lower()) or (nxt.isdigit() and len(nxt) == 4):
                    parts.append(nxt)
                    j += 1
                    if j < n and tokens[j].text.strip(".,") .isdigit() and len(tokens[j].text.strip(".,")) == 4:
                        parts.append(tokens[j].text.strip(".,"))
                        j += 1
            kind = NumberKind.DEADLINE if any(_stem_in(w, "deadline") for w in window_before) else NumberKind.DATE
            out.append(NumericMention(" ".join(parts), kind, None, sentence_id, False,
                                      [w for tk in tokens[i:j] for w in tk.word_ids]))
            i = j
            continue

        m = _NUM.match(text.replace(" ", ""))
        if t.spoken_number or m:
            if t.spoken_number:
                value = float(t.norm) if t.norm.replace(".", "", 1).isdigit() else None
                cur, suffix, plus = ("$" if t.unit == "$" else None), ("%" if t.unit == "%" else None), False
                digits = t.norm
            else:
                digits = m.group("num").replace(",", "")
                value = float(digits)
                cur, suffix, plus = m.group("cur"), m.group("suffix"), bool(m.group("plus"))
            consumed = 1
            scale_word = None
            if suffix and suffix.lower() in _SCALE and suffix != "%":
                value *= _SCALE[suffix.lower()]
                suffix = None
            elif i + 1 < n and norms[i + 1] in _SCALE and norms[i + 1] not in ("m", "b", "k"):
                scale_word = norms[i + 1]
                value = (value or 0) * _SCALE[scale_word]
                consumed = 2
            nxt_words = norms[i + consumed : i + consumed + 3]
            surface = text if not scale_word else f"{text} {scale_word}"
            kind: NumberKind | None = None
            if suffix == "%" or (nxt_words[:1] == ["percent"]):
                kind = NumberKind.PERCENTAGE
                if nxt_words[:1] == ["percent"]:
                    consumed += 1
                    surface += " percent"
            elif cur or t.unit == "$" or nxt_words[:1] in (["dollars"], ["dollar"], ["bucks"]):
                kind = NumberKind.PRICE if price_ctx else NumberKind.DOLLAR_AMOUNT
                if nxt_words[:1] in (["dollars"], ["dollar"], ["bucks"]):
                    consumed += 1
                    surface += " dollars"
            elif plus or (nxt_words[:1] in (["years"], ["year"]) and nxt_words[1:2] in (["old"], ["of"])) or \
                    (window_before[-1:] in (["age"], ["aged"]) ) or (nxt_words[:1] == ["yearold"]):
                kind = NumberKind.AGE
            elif digits.isdigit() and len(digits) == 4 and 1900 <= int(digits) <= 2100 and not cur and consumed == 1 \
                    and not (nxt_words[:1] and nxt_words[0] in ("dollars", "percent", "people", "ounces", "units")):
                kind = NumberKind.YEAR
                if any(_stem_in(w, "deadline") for w in window_before):
                    kind = NumberKind.DEADLINE
            elif value is not None and (value >= 10 or scale_word or "." in digits) and not t.is_year:
                kind = NumberKind.QUANTITY
                unit_word = nxt_words[0] if nxt_words and nxt_words[0] not in ("and", "or", "of", "in", "the", "to") else ""
                if unit_word and unit_word.isalpha() and len(unit_word) > 2 and unit_word not in _SCALE and unit_word not in MONTHS:
                    surface += f" {unit_word}"
                    consumed += 1
            elif t.is_year:
                kind = NumberKind.DEADLINE if any(_stem_in(w, "deadline") for w in window_before) else NumberKind.YEAR
            if kind is not None:
                out.append(NumericMention(surface, kind, value, sentence_id, t.spoken_number,
                                          [w for tk in tokens[i : i + consumed] for w in tk.word_ids]))
            i += consumed
            continue
        i += 1
    return out
