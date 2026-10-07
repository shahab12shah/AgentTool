from __future__ import annotations

import pytest

from app.analysis.analyzer import RuleBasedAnalyzer
from app.analysis.claims import classify_claim
from app.analysis.entities import extract_entities
from app.analysis.models import ClaimType as C, EntityType as E, NumberKind as N, VisualType as V
from app.analysis.numbers import extract_numbers
from app.analysis.prep import AToken
from app.core.textutil import normalize_token
from app.transcription.models import Word


def toks(s: str) -> list[AToken]:
    return [AToken(w, normalize_token(w), [f"w{i}"]) for i, w in enumerate(s.split())]


def spoken(s: str) -> list[Word]:
    return [Word(f"w{i}", w, i * 0.4, i * 0.4 + 0.3, 0.9) for i, w in enumerate(s.split())]


def analyse(sentence: str, lowercase: bool = False):
    words = spoken(sentence.lower() if lowercase else sentence)
    return RuleBasedAnalyzer()._analyze_sentence("s0", words, {})


# ------------------------------------------------------------------ entities
def test_spec_example_entities():
    ents = {e.text: e.type for e in extract_entities(toks("The IRS said Silver and solar panels matter across the United States."))}
    assert ents["IRS"] is E.GOVERNMENT_AGENCY and ents["Silver"] is E.FINANCIAL_INSTRUMENT
    assert ents["solar panels"] is E.TECHNOLOGY and ents["United States"] is E.COUNTRY


@pytest.mark.parametrize("text,expected", [
    ("Elon Musk visited Tokyo with Tesla executives.", {"Elon Musk": E.PERSON, "Tokyo": E.CITY, "Tesla": E.COMPANY}),
    ("Texas Instruments Inc made chips for the Federal Reserve.", {"Texas Instruments Inc": E.ORGANIZATION, "Federal Reserve": E.GOVERNMENT_AGENCY}),
    ("Dr. Smith studied bitcoin and electric vehicles.", {"bitcoin": E.FINANCIAL_INSTRUMENT, "electric vehicles": E.TECHNOLOGY}),
    ("Ford builds cars in Germany.", {"Ford": E.COMPANY, "cars": E.OBJECT, "Germany": E.COUNTRY}),
])
def test_entity_types(text, expected):
    got = {e.text: e.type for e in extract_entities(toks(text))}
    for k, v in expected.items():
        assert got.get(k) is v, (k, got)


def test_entities_work_on_lowercase_transcripts_via_gazetteer_and_ambiguous_names_need_case():
    a = analyse("the irs sent a notice about silver in the united states", lowercase=True)
    names = {e.canonical for e in a.entities}
    assert {"irs", "silver", "united states"} <= names
    assert not [e for e in analyse("an apple a day").entities if e.type is E.COMPANY]  # ordinary word
    assert [e for e in analyse("Today Apple released a phone").entities if e.type is E.COMPANY]


def test_entity_mentions_are_counted_with_word_provenance():
    es = {e.text: e for e in extract_entities(toks("The IRS warned that the IRS will audit"))}
    assert es["IRS"].mentions == 2 and es["IRS"].word_ids == ["w1", "w5"]


# ------------------------------------------------------------------ numbers & dates
@pytest.mark.parametrize("text,kind,value", [
    ("The price reached $100 today", N.PRICE, 100.0),
    ("He paid $2.5 million for it", N.DOLLAR_AMOUNT, 2.5e6),
    ("Inflation hit 3.5% last year", N.PERCENTAGE, 3.5),
    ("In 2027 everything changes", N.YEAR, 2027.0),
    ("Retirees aged 65+ qualify", N.AGE, 65.0),
    ("About 3 million ounces were sold", N.QUANTITY, 3e6),
    ("The deadline is 2027 for filers", N.DEADLINE, 2027.0),
])
def test_number_kinds(text, kind, value):
    found = [m for m in extract_numbers(toks(text), "s0")]
    assert found and found[0].kind is kind and found[0].value == value, found


def test_dates_and_deadlines():
    ms = extract_numbers(toks("File by April 15th or on March 2027"), "s0")
    assert [(m.text, m.kind) for m in ms] == [("April 15th", N.DEADLINE), ("March 2027", N.DATE)]


def test_spoken_numbers_become_numeric_mentions():
    a = analyse("the price reached one hundred dollars in twenty twenty seven up thirty five percent", lowercase=True)
    got = {(m.kind, m.value, m.spoken) for m in a.numbers}
    assert (N.PRICE, 100.0, True) in got and (N.YEAR, 2027.0, True) in got and (N.PERCENTAGE, 35.0, True) in got


def test_numbers_alone_without_context_do_not_become_noise():
    assert extract_numbers(toks("He bought one car and two dogs"), "s0") == []


# ------------------------------------------------------------------ claims
@pytest.mark.parametrize("text,ctype,needs_evidence", [
    ("Silver demand increased sharply last quarter.", C.FACT, True),
    ("The price reached $100 in March.", C.NUMBER, True),
    ("The notice arrives on April 15th.", C.DATE, True),
    ("The new tax law changes how deductions work.", C.LAW, True),
    ("Filers must submit the form before the deadline.", C.RULE, True),
    ("The minister said the policy will end soon.", C.QUOTE, True),
    ("Prices will probably rise next year.", C.PREDICTION, True),
    ("I think this is the best strategy.", C.OPINION, False),
    ("Why does silver keep rising?", C.QUESTION, False),
])
def test_claim_types(text, ctype, needs_evidence):
    a = analyse(text)
    assert len(a.claims) == 1
    c = a.claims[0]
    assert c.type is ctype and c.requires_evidence is needs_evidence
    assert c.evidence == [] and c.evidence_status == "NOT_RESEARCHED"  # nothing is ever fabricated


def test_market_claim_domain_and_filler_is_not_a_claim():
    assert analyse("Silver demand in the market increased as investors bought more.").claims[0].domain == "market"
    assert analyse("Welcome and thanks for watching").claims == []


# ------------------------------------------------------------------ visual requirement types
@pytest.mark.parametrize("text,expected", [
    ("The IRS sent a notice to affected taxpayers in the United States.", V.EVIDENCE),   # spec example
    ("Silver demand from solar manufacturing increased.", V.PROCESS),                     # spec example
    ("The price reached $100.", V.DATA),                                                  # spec example
    ("Elon Musk announced that he is stepping down as CEO.", V.PERSON),
    ("Mortgage rates in Canada are higher than in Germany.", V.COMPARISON),
    ("Bitcoin crashed and investors panicked.", V.EVENT),
    ("The future of risk is uncertain and the impact is hard to measure.", V.ABSTRACT),
    ("A laptop sits on the desk.", V.OBJECT),
])
def test_visual_requirement_types(text, expected):
    a = analyse(text)
    scores = a.type_scores
    from app.analysis.intent import pick_type

    assert pick_type(scores)[0] is expected, scores


def test_secondary_types_and_confidence_are_reported():
    from app.analysis.intent import pick_type

    top, secondary, conf = pick_type(analyse("The IRS sent a notice to affected taxpayers in the United States.").type_scores)
    assert top is V.EVIDENCE and 0.0 <= conf <= 1.0 and all(isinstance(t, V) for t in secondary)


# ------------------------------------------------------------------ regressions found by looking at the real UI
def test_sentence_initial_words_inside_a_scene_are_not_entities():
    """A scene covers several sentences; 'The'/'Inflation' after a full stop are not proper nouns."""
    ents = extract_entities(toks("Next, consider inflation. Inflation hit 3.5% according to the Federal Reserve. The Supreme Court ruled on tax law."))
    names = {e.text for e in ents}
    assert {"Federal Reserve", "Supreme Court"} <= names
    assert "The" not in names and "Inflation" not in names


def test_instructions_and_fragments_are_not_claims():
    assert analyse("Next, consider inflation.").claims == []
    assert analyse("Moving on to housing.").claims == []
    assert analyse("Remember that nobody can predict the future.").claims == []
    assert analyse("Inflation hit 3.5% according to the Federal Reserve.").claims  # a real claim still is one
