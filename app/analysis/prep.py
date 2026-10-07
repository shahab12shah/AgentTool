"""Builds analysis tokens for one sentence from the spoken words (+ script surface forms)."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.textutil import normalize_token, parse_number_run
from app.transcription.models import Word


@dataclass
class AToken:
    text: str  # surface form used for casing decisions ("IRS", "$100")
    norm: str
    word_ids: list[str]
    spoken_number: bool = False
    is_year: bool = False
    unit: str | None = None  # "$" / "%" for merged spoken numbers


def build_tokens(words: list[Word], surfaces: dict[str, str] | None = None) -> list[AToken]:
    """One token per spoken word; spoken number runs become a single digit token.

    ``surfaces`` maps word ids to the user's script spelling so capitalisation and
    punctuation survive even when the recogniser output is lower-case and unpunctuated.
    """
    surfaces = surfaces or {}
    norms = [normalize_token(w.text) or w.text.lower() for w in words]
    out: list[AToken] = []
    i = 0
    while i < len(words):
        run = parse_number_run(norms, i)
        if run:
            ids = [w.word_id for w in words[i : i + run.consumed]]
            text = f"${run.text}" if run.unit == "$" else f"{run.text}%" if run.unit == "%" else run.text
            out.append(AToken(text, run.text, ids, True, run.is_year, run.unit))
            i += run.consumed
            continue
        w = words[i]
        text = surfaces.get(w.word_id) or w.text
        out.append(AToken(text.strip(), norms[i], [w.word_id]))
        i += 1
    return out
