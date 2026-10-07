from __future__ import annotations

from pathlib import Path

import pytest

from app.tests.helpers import NARRATION, ScriptedProvider, timed_words
from app.media.asset import Asset, AssetType, SourceType
from app.transcription.alignment import AlignStatus as S, ScriptAlignment, Verdict, align_script, script_hash
from app.transcription.service import TranscriptionEngine

AUDIO = Asset("media_00001", AssetType.AUDIO, SourceType.USER_MEDIA, "a.wav", "a.wav", duration=300.0, content_hash="h")


def transcript(spoken: str, punctuation: bool = False):
    return TranscriptionEngine().run(ScriptedProvider(spoken, punctuation=punctuation), Path("a.wav"), AUDIO, "en").transcript


def statuses(a: ScriptAlignment):
    return [(i.script_text, i.transcript_text, i.status) for i in a.items]


def test_exact_match():
    a = align_script("Silver demand is rising rapidly.", transcript("silver demand is rising rapidly"))
    assert all(i.status is S.MATCHED for i in a.items) and len(a.items) == 5
    assert a.verdict is Verdict.IDENTICAL and a.stats.coverage == 1.0 and a.stats.transcript_only == 0


def test_spec_example_extra_word_is_transcript_only_and_rest_matches():
    a = align_script("Silver demand is rising rapidly.", transcript("silver demand is rising very rapidly"))
    by = {(i.script_text or "").strip(".").lower() or i.transcript_text: i.status for i in a.items}
    assert by["silver"] is by["demand"] is by["is"] is by["rising"] is S.MATCHED
    assert by["very"] is S.TRANSCRIPT_ONLY
    assert by["rapidly"] in (S.MATCHED, S.APPROXIMATE)
    assert a.verdict is Verdict.MINOR_DIFFERENCES and a.stats.transcript_only == 1


def test_minor_wording_difference_is_approximate():
    a = align_script("Prices are rising.", transcript("price are rise"))
    got = {i.script_text.strip("."): i.status for i in a.items if i.script_text}
    assert got["Prices"] is S.APPROXIMATE and got["are"] is S.MATCHED and got["rising"] is S.APPROXIMATE
    assert a.stats.approximate == 2 and a.verdict is not Verdict.IDENTICAL


def test_missing_words_are_script_only():
    a = align_script("The price of gold climbed twenty percent last year.", transcript("the price of gold climbed last year"))
    missing = [i.script_text for i in a.items if i.status is S.SCRIPT_ONLY]
    assert missing == ["twenty percent"] or set(missing) <= {"twenty", "percent", "twenty percent"} and missing
    assert a.stats.script_only >= 1 and a.stats.coverage < 1.0


def test_added_words_are_transcript_only():
    a = align_script("Gold climbed.", transcript("well you know gold actually climbed"))
    added = [i.transcript_text for i in a.items if i.status is S.TRANSCRIPT_ONLY]
    assert added == ["well", "you", "know", "actually"] and a.stats.transcript_only == 4


def test_reordered_phrase_is_detected():
    a = align_script("First we cover taxes then penalties for late filing.", transcript("first we cover penalties for late filing then taxes"))
    reordered = [i for i in a.items if i.status is S.REORDERED]
    assert [i.script_text for i in reordered] == ["taxes"] and reordered[0].transcript_text == "taxes"
    assert a.stats.reordered == 1


def test_significant_difference_produces_regions_with_timing():
    script = "Silver demand is rising. " + "Our sponsor offers a discount on premium coffee beans today. " + "Gold is steady."
    spoken = "silver demand is rising and then we talk about completely different travel tips for hiking trails gold is steady"
    tr = transcript(spoken)
    a = align_script(script, tr)
    assert a.verdict is Verdict.SIGNIFICANT_DIFFERENCES and a.regions
    r = a.regions[0]
    assert r.start is not None and r.end > r.start and r.start >= tr.words[0].start
    assert "sponsor" in r.script_text and "travel" in r.transcript_text


def test_spoken_numbers_match_written_numbers():
    a = align_script("The price reached $100 in 2027, up 35%.", transcript("the price reached one hundred dollars in twenty twenty seven up thirty five percent"))
    assert a.stats.script_only == 0 and a.stats.transcript_only == 0
    multi = [i for i in a.items if len(i.word_ids) > 1]
    assert [(i.script_text, len(i.word_ids)) for i in multi] == [("$100", 3), ("2027,", 3), ("35%.", 3)]


def test_script_sentence_indexes_and_casing_transfer():
    a = align_script("The IRS sent a notice. Silver rose.", transcript("the irs sent a notice silver rose"))
    tr = transcript("the irs sent a notice silver rose")
    a = align_script("The IRS sent a notice. Silver rose.", tr)
    sent_of = a.script_sentence_of_word()
    assert [sent_of[w.word_id] for w in tr.words] == [0, 0, 0, 0, 0, 1, 1]
    surf = a.surface_by_word()
    assert surf[tr.words[1].word_id] == "IRS"  # script casing transferred to the lower-case transcript word


def test_alignment_never_modifies_script_or_transcript():
    script = "Silver demand is rising rapidly."
    tr = transcript("silver demand is rising very rapidly")
    words_before = [(w.text, w.start, w.end) for w in tr.words]
    a = align_script(script, tr)
    assert script == "Silver demand is rising rapidly." and [(w.text, w.start, w.end) for w in tr.words] == words_before
    assert a.script_hash == script_hash(script) and a.transcript_id == tr.transcript_id


def test_alignment_serialises_and_scales():
    tr = TranscriptionEngine().run(ScriptedProvider(NARRATION), Path("a.wav"), AUDIO, "en").transcript
    a = align_script(NARRATION, tr)
    assert a.verdict is Verdict.IDENTICAL
    assert ScriptAlignment.from_dict(a.to_dict()) == a
    import time

    long_script = (NARRATION + " ") * 12  # ~4k words
    long_tr = TranscriptionEngine().run(ScriptedProvider(long_script), Path("a.wav"), AUDIO, "en").transcript
    t0 = time.perf_counter()
    b = align_script(long_script, long_tr)
    assert b.stats.coverage == 1.0 and time.perf_counter() - t0 < 5.0


def test_empty_script_or_unrelated_script_is_handled():
    tr = transcript("silver demand is rising")
    a = align_script("", tr)
    assert a.stats.script_words == 0 and a.stats.transcript_only == 4
    b = align_script("Completely unrelated banana smoothie recipe.", tr)
    assert b.verdict is Verdict.SIGNIFICANT_DIFFERENCES and b.stats.matched == 0
