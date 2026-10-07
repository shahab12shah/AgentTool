"""Text helpers shared by script alignment and semantic analysis (no model, no I/O)."""

from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being below between both but by
    can could did do does doing down during each few for from further had has have having he her here hers him his how i if
    in into is it its itself just me more most my no nor not now of off on once only or other our out over own same she
    should so some such than that the their them then there these they this those through to too under until up very was we
    were what when where which while who whom why will with would you your yours""".split()
)

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000, "trillion": 10**12}
_UNIT_WORDS = {"dollars": "$", "dollar": "$", "percent": "%", "bucks": "$", "cents": "c"}
_YEAR_LEADS = {"nineteen": 19, "twenty": 20, "eighteen": 18, "seventeen": 17, "sixteen": 16, "fifteen": 15}


@dataclass
class NumberRun:
    value: float
    consumed: int  # tokens used, including a trailing unit word
    unit: str | None  # "$", "%", or None
    is_year: bool = False

    @property
    def text(self) -> str:
        v = int(self.value) if float(self.value).is_integer() else self.value
        return f"{v}"


def _two_digit(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Parse a value 10..99 at tokens[i:]; returns (value, consumed)."""
    if i >= len(tokens):
        return None
    t = tokens[i]
    if t in _TENS:
        if i + 1 < len(tokens) and tokens[i + 1] in _ONES and 1 <= _ONES[tokens[i + 1]] <= 9:
            return _TENS[t] + _ONES[tokens[i + 1]], 2
        return _TENS[t], 1
    if t in _ONES and 10 <= _ONES[t] <= 19:
        return _ONES[t], 1
    return None


def parse_number_run(tokens: list[str], i: int) -> NumberRun | None:
    """Parse a spoken number (``two hundred and fifty``, ``twenty twenty seven``, ``two point five``) at ``tokens[i]``."""
    n = len(tokens)
    if i >= n or not (tokens[i] in _ONES or tokens[i] in _TENS or tokens[i] in _SCALES):
        return None
    # spoken years: "nineteen eighty four", "twenty twenty seven"
    if tokens[i] in _YEAR_LEADS:
        rest = _two_digit(tokens, i + 1)
        after = i + 1 + (rest[1] if rest else 0)
        scale_follows = after < n and tokens[after] in _SCALES
        if rest and not scale_follows:
            return NumberRun(_YEAR_LEADS[tokens[i]] * 100 + rest[0], 1 + rest[1], None, True)
    total, current, j, seen = 0, 0, i, False
    while j < n:
        t = tokens[j]
        if t in _ONES:
            current += _ONES[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == "hundred":
            current = max(current, 1) * 100
        elif t in _SCALES:
            total += max(current, 1) * _SCALES[t]
            current = 0
        elif t == "and" and seen and j + 1 < n and (tokens[j + 1] in _ONES or tokens[j + 1] in _TENS):
            j += 1
            continue
        else:
            break
        seen = True
        j += 1
    value: float = total + current
    if j + 1 < n and tokens[j] == "point" and tokens[j + 1] in _ONES and _ONES[tokens[j + 1]] < 10:
        digits = ""
        k = j + 1
        while k < n and tokens[k] in _ONES and _ONES[tokens[k]] < 10:
            digits += str(_ONES[tokens[k]])
            k += 1
        value = float(f"{int(value)}.{digits}")
        j = k
        if j < n and tokens[j] in _SCALES and tokens[j] != "hundred":
            value *= _SCALES[tokens[j]]
            j += 1
    unit = None
    if j < n and tokens[j] in _UNIT_WORDS and _UNIT_WORDS[tokens[j]] in ("$", "%"):
        unit = _UNIT_WORDS[tokens[j]]
        j += 1
    return NumberRun(value, j - i, unit)


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "inc", "corp", "ltd", "co", "no", "u.s", "e.g", "i.e", "etc"}
_SENT_END = re.compile(r"([.!?]+[\"')\]]*)(\s+)")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into (sentence, start_char, end_char). Guards common abbreviations and decimals."""
    out: list[tuple[str, int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        before = text[start:end].rstrip(".!?\"')]").split()
        last = before[-1].lower().rstrip(".") if before else ""
        if m.group(1).startswith(".") and (last in _ABBREV or (len(last) == 1 and last.isalpha())):
            continue
        out.append((text[start:end].strip(), start, end))
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append((tail, start, len(text)))
    return [(s, a, b) for s, a, b in out if s]


_TOKEN = re.compile(r"[\w$€£%][\w'’.,%$-]*|\S", re.UNICODE)


@dataclass
class TextToken:
    text: str  # surface form including attached punctuation
    norm: str
    start: int
    end: int


def normalize_token(raw: str) -> str:
    t = raw.lower().replace("’", "'")
    t = t.strip(".,;:!?\"()[]{}“”‘’'-—–")
    t = re.sub(r"'s$", "", t).replace("'", "")
    t = re.sub(r"^[$€£]", "", t)
    t = t.rstrip("%+")
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    return t


def tokenize(text: str) -> list[TextToken]:
    tokens = []
    for m in _TOKEN.finditer(text):
        norm = normalize_token(m.group())
        if norm and (norm[0].isalnum()):
            tokens.append(TextToken(m.group(), norm, m.start(), m.end()))
    return tokens


def content_terms(text: str) -> list[str]:
    """Lower-cased, lightly stemmed non-stopword terms (for cohesion / topic work)."""
    out = []
    for t in tokenize(text):
        w = t.norm
        if w in STOPWORDS or len(w) < 2 or w.isdigit():
            continue
        out.append(stem(w))
    return out


def stem(w: str) -> str:
    """Very light suffix stripping so rise/rising/rises and price/prices share a stem."""
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            w = w[: len(w) - len(suf)] + rep
            break
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "ls":
        w = w[:-1]
    return w[:-1] if len(w) >= 4 and w.endswith("e") else w
