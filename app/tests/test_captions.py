"""Phase 5 caption tests: timing, segmentation, line breaking, safe area, styles and keyword emphasis."""

from __future__ import annotations

import pytest

from app.captions.engine import CaptionEngine, MAX_CPS, break_lines, glued, layout_for
from app.captions.keywords import KeywordService
from app.captions.styles import PRESETS, effective_style, style_for
from app.presentation.models import CaptionSegment, CaptionSettings, CaptionWord

CANVAS = (1920, 1080)


def cw(text_start_end):
    return [CaptionWord(f"w{i}", t, a, b) for i, (t, a, b) in enumerate(text_start_end)]


def timed(sentence: str, start=0.0, wps=2.6, gap_after=None):
    """Words of ``sentence`` with evenly spaced timings at ``wps`` words per second."""
    out, t = [], start
    for i, tok in enumerate(sentence.split()):
        d = 0.8 / wps
        out.append((tok, round(t, 3), round(t + d, 3)))
        t += 1.0 / wps
    return cw(out)


def engine(**kw):
    s = CaptionSettings(**kw)
    return CaptionEngine(s, {}, CANVAS)


def seg(eng, sentences, **kw):
    return eng.segment("scene_001", sentences, **kw)


# ---------------------------------------------------------------- timing / segmentation
def test_short_sentence_is_one_caption_on_word_boundaries():
    ws = timed("Silver demand continues to rise.")
    segs = seg(engine(), [ws])
    assert len(segs) == 1
    s = segs[0]
    assert s.start == ws[0].start and s.text == "Silver demand continues to rise." and s.end >= ws[-1].end and s.words[0].word_id == "w0"
    assert [w.text for w in s.words] == [w.text for w in ws]  # spoken wording is never changed


def test_long_sentence_splits_at_natural_phrases_not_fixed_chunks():
    text = ("Silver demand has changed dramatically over the last decade, because solar manufacturers are now consuming more of the metal, "
            "which means that supply is under pressure.")
    ws = timed(text, wps=3.0)
    segs = seg(engine(), [ws])
    assert len(segs) >= 3
    assert all(len(s.lines) <= 2 and all(len(l) <= engine().layout.chars_per_line + 2 for l in s.lines) for s in segs)
    assert any(l.endswith(",") for s in segs for l in s.lines)  # a line/caption break lands on the comma
    assert all(not l.split()[-1].lower().strip(",.") in ("which", "the", "of", "to", "a", "because") for s in segs for l in s.lines)
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start + 1e-6  # no overlap
        assert not a.text.split()[-1].lower().strip(",.") in ("the", "of", "to", "a")  # never ends on a hanging function word
    allwords = [w.word_id for s in segs for w in s.words]
    assert allwords == [w.word_id for w in ws]  # every word exactly once, in order


def test_fast_speech_is_split_more_slowly_spoken_less():
    sentence = "The company announced that production will double next year across every factory"
    fast, slow = seg(engine(), [timed(sentence, wps=5.0)]), seg(engine(), [timed(sentence, wps=1.8)])
    assert len(fast) >= len(slow)
    assert all(len(s.lines) == 1 for s in fast)  # dense speech gets short one-line captions
    assert all(s.reading_cps <= MAX_CPS * 1.15 for s in slow)  # relaxed speech is comfortably readable
    assert slow[0].end - slow[0].start >= 0.9  # held long enough to read


def test_a_pause_starts_a_new_caption():
    a = timed("The deadline is coming", 0.0)
    b = timed("and nobody is ready", a[-1].end + 1.2)
    both = a + b
    both = [CaptionWord(f"w{i}", w.text, w.start, w.end) for i, w in enumerate(both)]
    segs = seg(engine(), [both])
    assert len(segs) >= 2
    assert any(abs(s.start - b[0].start) < 1e-6 for s in segs)  # a caption starts right after the 1.2s pause


def test_multiple_sentences_are_never_merged_and_do_not_overlap():
    s1, s2, s3 = timed("First sentence here.", 0.0), timed("Second one follows soon.", 2.0), timed("And a third.", 4.5)
    allw = [CaptionWord(f"w{i}", w.text, w.start, w.end) for i, w in enumerate(s1 + s2 + s3)]
    n1, n2 = len(s1), len(s2)
    segs = seg(engine(), [allw[:n1], allw[n1:n1 + n2], allw[n1 + n2:]])
    assert [s.text for s in segs] == ["First sentence here.", "Second one follows soon.", "And a third."]
    assert all(a.end <= b.start for a, b in zip(segs, segs[1:]))


def test_caption_end_never_runs_into_the_next_or_past_the_scene():
    ws = timed("One two three four five six seven eight nine ten eleven twelve thirteen fourteen", wps=4.0)
    segs = seg(engine(), [ws], scene_end=ws[-1].end + 0.1)
    assert segs[-1].end <= ws[-1].end + 0.1 + 1e-6 and all(s.end > s.start for s in segs)


def test_density_is_reduced_by_a_slower_reading_speed_setting():
    ws = timed("Investors panicked and sold billions of dollars in a single day after the report", wps=3.5)
    normal = seg(engine(), [ws])
    slow = seg(engine(reading_speed=0.6), [ws])
    assert len(slow) >= len(normal)
    assert max(s.reading_cps for s in slow) <= max(s.reading_cps for s in normal) + 1e-6 or len(slow) > len(normal)


# ---------------------------------------------------------------- line breaking
def lines(text, cpl=18, max_lines=2):
    from app.captions.engine import Layout

    ws = cw([(t, i * 0.3, i * 0.3 + 0.25) for i, t in enumerate(text.split())])
    return break_lines(ws, Layout(cpl, cpl * max_lines, max_lines, 50, 1500))


def test_line_breaking_groups_phrases_instead_of_stacking_words():
    assert lines("SILVER IS RUNNING OUT", 14) == ["SILVER IS", "RUNNING OUT"]
    assert lines("Short text", 30) == ["Short text"]
    two = lines("Silver prices could cross $100 before 2027", 24)
    assert len(two) == 2 and all(len(l) <= 24 for l in two)
    assert "SILVER" not in lines("a b c d e f", 6)  # smoke: tiny words do not crash


def test_line_breaking_with_long_words_numbers_currency_and_dates():
    two = lines("Infrastructure modernization requires approximately extraordinary investment", 40)
    assert len(two) == 2 and all(len(l) <= 40 for l in two)
    d = lines("The filing deadline is October 15 for most taxpayers", 28)
    assert not any(l.endswith("October") for l in d) and not any(l.startswith("15") for l in d)  # a date stays together
    m = lines("It costs about 100 dollars per month for everyone", 24)
    assert not any(l.endswith("100") for l in m)  # a number stays with its unit
    p = lines("Wait, really? Yes - $3,500 or 28% of it, apparently", 26)
    assert all(len(l) <= 26 for l in p) and " ".join(p).replace("  ", " ") == "Wait, really? Yes - $3,500 or 28% of it, apparently"
    assert len(lines("word " * 5, 100)) == 1


def test_glue_rules():
    assert glued("October", "15") and glued("15th", "October") and glued("100", "dollars") and glued("28", "percent") and glued("Dr.", "Smith")
    assert glued("United", "States") and not glued("the", "States") and not glued("silver", "demand")


def test_one_line_mode_never_makes_two_lines():
    eng = engine(max_lines=1)
    ws = timed("Silver demand has changed dramatically over the last decade for many reasons", wps=3.0)
    segs = seg(eng, [ws])
    assert all(len(s.lines) == 1 and len(s.lines[0]) <= eng.layout.chars_per_line + 1 for s in segs)


# ---------------------------------------------------------------- safe area, size and styling
def test_layout_respects_safe_margins_and_font_size():
    from dataclasses import replace

    base = CaptionSettings()
    narrow = replace(base, safe_margin_left=0.25, safe_margin_right=0.25)
    s = style_for({}, "professional")
    a, b = layout_for(base, s, CANVAS), layout_for(narrow, s, CANVAS)
    assert b.chars_per_line < a.chars_per_line and b.box_width < a.box_width
    big = effective_style(s, replace(base, large_text=True))
    assert big.size_rel > s.size_rel and layout_for(replace(base, large_text=True), big, CANVAS).chars_per_line < a.chars_per_line
    hc = effective_style(s, replace(base, high_contrast=True))
    assert hc.background == "box" and hc.background_opacity >= 0.85 and hc.color == "#FFFFFF"
    assert effective_style(s, base, {"size_rel": 0.07, "color": "#FF0000"}).color == "#FF0000"  # per-caption overrides


def test_style_presets_exist_and_are_distinct():
    assert set(PRESETS) == {"professional", "clean", "bold", "news", "documentary", "minimal"}
    assert PRESETS["bold"].uppercase and PRESETS["news"].background == "box" and PRESETS["documentary"].font == "Serif"
    for p in PRESETS.values():
        assert p.size_rel >= 0.04 and p.emphasis and set(p.emphasis.values()) <= {"COLOR_CHANGE", "BOLD", "SCALE", "BACKGROUND_BOX", "UNDERLINE", "GLOW", "POP"}
    assert style_for({}, "nope").style_id == "professional"
    custom = PRESETS["clean"].__class__("mine", "Mine", size_rel=0.07)
    assert style_for({"mine": custom}, "mine").size_rel == 0.07


def test_uppercase_widens_the_text_estimate():
    from dataclasses import replace

    s = style_for({}, "professional")
    lo = layout_for(CaptionSettings(), s, CANVAS)
    up = layout_for(CaptionSettings(), replace(s, uppercase=True), CANVAS)
    assert up.chars_per_line < lo.chars_per_line


def test_segment_roundtrip():
    s = seg(engine(), [timed("Silver demand continues to rise.")])[0]
    assert CaptionSegment.from_dict(s.to_dict()) == s and s.animation["in"]["preset"] == "fade_in"


# ---------------------------------------------------------------- keywords (real pipeline scenes)
@pytest.fixture
def scenes_ws(research_ws):
    return research_ws


def scene_with(ws, text):
    for sc in ws.project.scenes:
        if text in sc.narration:
            return sc
    raise AssertionError(text)


def kinds(ws, sc):
    tr = ws.project.transcription.transcript
    words = tr.words_between(sc.start, sc.end)
    return {k.category: k for k in KeywordService().detect(sc, words)}, words


def test_keywords_cover_people_numbers_dates_percentages_money_and_warnings(scenes_ws):
    ws = scenes_ws
    ks, _ = kinds(ws, scene_with(ws, "Elon Musk"))
    assert "PERSON" in ks and "Elon" in ks["PERSON"].text and ks["PERSON"].reason
    assert "DATE" in ks and "2027" in ks["DATE"].text
    ks, _ = kinds(ws, scene_with(ws, "penalty of 5%"))
    assert "PERCENTAGE" in ks and "5%" in ks["PERCENTAGE"].text and ("WARNING" in ks or "DEADLINE" in ks)
    assert any(k in ks for k in ("DATE", "DEADLINE"))  # April 15th
    ks, _ = kinds(ws, scene_with(ws, "billions of dollars"))
    assert ks  # at least one keyword (bitcoin / dollars)
    ks, _ = kinds(ws, scene_with(ws, "gold climbed 20%"))
    assert "PERCENTAGE" in ks and ks["PERCENTAGE"].importance >= 0.9


def test_keyword_emphasis_is_sparse(scenes_ws):
    ws = scenes_ws
    tr = ws.project.transcription.transcript
    total_words = total_kw = 0
    for sc in ws.project.scenes:
        words = tr.words_between(sc.start, sc.end)
        kws = KeywordService().detect(sc, words)
        total_words += len(words)
        total_kw += sum(len(k.word_ids) for k in kws)
        assert len(kws) <= max(1, int(len(words) * 0.08) + 2)
    assert 0 < total_kw < total_words * 0.2  # never every word


def test_emphasis_marks_follow_toggles_and_limits(scenes_ws):
    ws = scenes_ws
    sc = scene_with(ws, "Elon Musk")
    tr = ws.project.transcription.transcript
    words = tr.words_between(sc.start, sc.end)
    kws = KeywordService().detect(sc, words)
    cws = [CaptionWord(w.word_id, w.text, w.start, w.end) for w in words]
    on = CaptionEngine(CaptionSettings(), {}, CANVAS).segment(sc.id, [cws], kws)
    marks = [m for s in on for m in s.emphasis]
    assert marks and all(len(s.emphasis) <= 2 for s in on) and all(m.style in PRESETS["professional"].emphasis.values() for m in marks)
    assert any(s.emphasis_words for s in on)
    no_num = CaptionEngine(CaptionSettings(number_emphasis=False), {}, CANVAS).segment(sc.id, [cws], kws)
    assert not any(m.category in ("NUMBER", "DATE", "MONEY", "PERCENTAGE", "DEADLINE") for s in no_num for m in s.emphasis)
    off = CaptionEngine(CaptionSettings(number_emphasis=False, keyword_highlight=False), {}, CANVAS).segment(sc.id, [cws], kws)
    assert not any(s.emphasis for s in off)


def test_money_warning_deadline_and_location_keywords_on_a_synthetic_scene():
    from app.analysis.models import Entity, EntityType, NumberKind, NumericMention, Scene
    from app.transcription.models import Word

    text = "Silver prices could cross $100 before 2027 in Texas but a penalty applies and the deadline is near"
    words = [Word(f"w{i}", t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate(text.split())]
    idx = {w.text: w.word_id for w in words}
    sc = Scene("s1", "1", 0.0, words[-1].end, text, numbers=[NumericMention("$100", NumberKind.DOLLAR_AMOUNT, 100.0, "", False, [idx["$100"]]),
                                                              NumericMention("2027", NumberKind.YEAR, 2027.0, "", False, [idx["2027"]])],
               entities=[Entity("Texas", EntityType.CITY, "texas", 1, [idx["Texas"]])])
    ks = {k.category: k for k in KeywordService().detect(sc, words)}
    assert ks["MONEY"].text == "$100" and ks["MONEY"].importance >= 0.9 and ks["DATE"].text == "2027" and ks["DEADLINE"].text == "deadline"
    assert "LOCATION" not in ks and "WARNING" not in ks  # budget keeps a scene sparse: lower-importance candidates are dropped
    sc2 = Scene("s3", "3", 0.0, words[-1].end, text, entities=sc.entities)
    ks2 = {k.category: k for k in KeywordService().detect(sc2, words)}
    assert ks2["WARNING"].text == "penalty" and ks2["WARNING"].source == "lexicon" and ks2["LOCATION"].text == "Texas"
    ranked = [k.category for k in KeywordService().detect(sc, words)]
    assert ranked[0] in ("MONEY", "DEADLINE") and len(ranked) <= max(1, int(len(words) * 0.08) + 2)  # sparse, best first
    audio = KeywordService().detect(Scene("s2", "2", 0.0, 5.0, "x"), words[:6], {words[1].word_id, words[0].word_id})
    assert audio and all(k.source == "audio" and k.category == "CONCEPT" for k in audio)
