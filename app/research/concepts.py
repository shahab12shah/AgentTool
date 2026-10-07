"""A small concept lexicon: which words belong to the same real-world domain.

It powers three things: related-term credit when scoring, alternative-interpretation queries,
and "generic trap" detection (e.g. coin photos for a scene about solar-panel silver).
This is deliberately small and hand-written; it is NOT a knowledge base and will miss most topics.
"""

from __future__ import annotations

from app.core.textutil import stem

DOMAINS: dict[str, set[str]] = {
    "solar": {"solar", "photovoltaic", "pv", "panel", "cell", "silicon", "wafer", "renewable", "inverter", "installation", "array", "module"},
    "precious_metals": {"silver", "gold", "platinum", "palladium", "bullion", "ounce", "metal", "refinery", "ingot", "bar", "coin", "jewelry"},
    "mining": {"mine", "mining", "ore", "refinery", "smelting", "extraction", "excavation"},
    "tax_government": {"irs", "tax", "taxpayer", "form", "filing", "notice", "reporting", "revenue", "audit", "deadline", "penalty", "treasury",
                       "paperwork", "return", "1099", "agency", "government"},
    "markets": {"market", "price", "chart", "trading", "investor", "stock", "shares", "futures", "commodity", "demand", "supply", "rally", "inflation",
                "yield", "fund", "etf"},
    "ev_battery": {"battery", "lithium", "electric", "vehicle", "ev", "charging", "cell"},
    "manufacturing": {"factory", "manufacturing", "production", "assembly", "plant", "industrial", "machine", "robot", "line", "fabrication"},
    "housing": {"house", "mortgage", "housing", "home", "property", "rent"},
    "space": {"nasa", "lunar", "moon", "space", "rocket", "satellite"},
    "crypto": {"bitcoin", "crypto", "blockchain", "ethereum", "wallet"},
    "retirement": {"retirement", "ira", "401k", "pension", "retiree", "retirees"},
    "legal": {"law", "court", "ruling", "legislation", "statute", "judge", "supreme", "regulation"},
}
_STEM_DOMAINS: dict[str, set[str]] = {k: {stem(w) for w in v} for k, v in DOMAINS.items()}

# Visuals that look right for a *different* meaning of a word: signal "generic/wrong visual" when the
# scene's primary subject is not in the same domain.
TRAPS: dict[str, set[str]] = {
    "precious_metals": {"coin", "coins", "bullion", "jewelry", "jewellery", "ingot", "ingots", "bars", "necklace", "ring"},
    "tax_government": {"paperwork", "calculator", "piggy"},
    "markets": {"casino", "gambling"},
}


def domains_of(word: str) -> set[str]:
    st = stem(word.lower())
    return {d for d, words in _STEM_DOMAINS.items() if st in words}


def related(word: str, limit: int = 4) -> list[str]:
    """Words sharing a domain with ``word`` (excluding itself), most generic first."""
    out: list[str] = []
    for d in domains_of(word):
        for w in sorted(DOMAINS[d]):
            if stem(w) != stem(word.lower()) and w not in out:
                out.append(w)
    return out[:limit]


def shared_domain(a: str, b: str) -> bool:
    return bool(domains_of(a) & domains_of(b))


def traps_for(subject_words: list[str], other_words: list[str]) -> set[str]:
    """Trap words for the domains in ``other_words`` that are not also domains of the subject."""
    subj: set[str] = set()
    for w in subject_words:
        subj |= domains_of(w)
    out: set[str] = set()
    for w in other_words:
        for d in domains_of(w):
            if d not in subj and d in TRAPS:
                out |= TRAPS[d]
    return out
