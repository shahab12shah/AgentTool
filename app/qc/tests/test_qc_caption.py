"""CaptionChecker: overflow (off-screen / inside the margins), safe margins with their fix, collisions, word timing, density, flicker, reading speed, contrast, font size,
highlights, animation, consistency, locks, scene-local reuse, and "clean captions say nothing".

Two scenes (0-10 s and 10-20 s, voice-over 20 s) with a transcript. Captions are placed directly on track_v6 with exact timings, so every figure in these tests is exact.
"""

from __future__ import annotations

import pytest

from app.captions.engine import CaptionEngine
from app.presentation.animation import spec
from app.presentation.models import CaptionStyle, CaptionWord
from app.qc import fix_catalog
from app.qc.caption_checker import CaptionChecker, caption_summary
from app.qc.geometry import Margins, caption_layout
from app.qc.issue_model import CheckerState, FixRoute, QCCategory
from app.qc.qc_engine import PreviousState, QCEngine
from app.qc.settings import QCSettings
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_clip, add_scene, codes, find, new_project, narrate, qc_ctx, run_checker
from app.tests.conftest import needs_ffmpeg
from app.timeline.clip import Clip

ALLOWED = {"caption.overflow", "caption.collision", "caption.overcrowding", "caption.flicker", "caption.inconsistency", "caption.too_fast", "caption.too_short", "caption.low_contrast",
           "caption.small_font", "caption.safe_margin", "caption.words", "caption.range", "caption.highlight", "caption.animation"}
SHORT = "Silver prices rose sharply"  # 26 characters
FADE = {"in": spec("fade_in", duration=0.12), "out": spec("fade_out", duration=0.10)}


class World:
    def __init__(self, tmp_path):
        self.p = p = new_project(tmp_path, seconds=20)
        self.s1 = add_scene(p, 0, 10, "Silver prices rose sharply last week.", importance=0.8)
        self.s2 = add_scene(p, 10, 20, "Demand keeps growing across Asia.")
        narrate(p)

    def cap(self, start: float, text: str = SHORT, dur: float | None = None, *, by: str = "AI", position: str = "bottom", xy: list[float] | None = None, style_id: str = "professional",
            overrides: dict | None = None, lines: list[str] | None = None, emphasis: list[dict] | None = None, anim: dict | None = None, mode: str = "HIGHLIGHT", words: list[dict] | None = None,
            track: str = "track_v6", **kw) -> Clip:
        dur = dur if dur is not None else max(1.2, len(text) / 13.0)
        toks = text.split()
        step = dur * 0.9 / max(1, len(toks))
        ws = words if words is not None else [{"word_id": f"w{i}", "text": t, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3)} for i, t in enumerate(toks)]
        seg = {"caption_id": "cap", "scene_id": "", "start": start, "end": start + dur, "text": text, "lines": lines if lines is not None else [text], "words": ws, "emphasis": emphasis or [],
               "style_id": style_id, "style_overrides": overrides or {}, "position": position, "position_xy": xy or [], "highlight_mode": mode, "reading_cps": 10.0}
        scene = self.s1 if start + dur / 2 < 10 else self.s2
        return add_clip(self.p, track, None, round(start, 3), dur, scene=scene, kind="caption", created_by=by, slot="caption:0", text=seg, animation=anim if anim is not None else dict(FADE), **kw)

    def box(self, start: float, dur: float, region=(0.3, 0.84, 0.4, 0.10), *, by: str = "AI") -> Clip:
        scene = self.s1 if start < 10 else self.s2
        return add_clip(self.p, "track_v4", None, start, dur, scene=scene, kind="graphic", created_by=by, slot="evidence:0",
                        effects={"highlight": {"region": list(region), "style": "box", "darken_surround": True}})

    def text(self, start: float, dur: float, content: str, pos=(0.5, 0.88), size: int = 60) -> Clip:
        scene = self.s1 if start < 10 else self.s2
        body = {"text_id": "t", "content": content, "start": start, "duration": dur, "position": list(pos), "style": "LABEL", "emphasis": "", "font": "Sans", "size": size, "alignment": "center",
                "opacity": 1.0, "background": "none"}
        return add_clip(self.p, "track_v5", None, start, dur, scene=scene, kind="text", created_by="AI", slot="text:0", text=body)

    def run(self, **kw):
        return run_checker(CaptionChecker(), qc_ctx(self.p, **kw))


def issues_of(out, code):
    return find(out, code)


# ------------------------------------------------------------------ contract / cache key
def test_checker_declares_what_it_is_and_its_cache_key_accurately(tmp_path):
    chk = CaptionChecker()
    assert (chk.id, chk.scene_local, chk.expensive) == ("caption", True, False) and chk.categories == (QCCategory.CAPTION,)
    assert {"timeline", "captions", "transcript"} <= set(chk.domains) and set(chk.settings_sections) == {"caption", "style"}
    w = World(tmp_path)
    w.cap(1.0)
    base = chk.input_hash(qc_ctx(w.p))
    s = QCSettings()
    s.sync.major_ms = 900  # not read by this checker
    assert chk.input_hash(qc_ctx(w.p, s)) == base
    s2 = QCSettings()
    s2.caption.max_cps = 30
    assert chk.input_hash(qc_ctx(w.p, s2)) != base
    w.p.caption_settings.safe_margin_bottom = 0.1  # the project's margins are part of the answer
    assert chk.input_hash(qc_ctx(w.p)) != base


def test_clean_captions_report_nothing_and_measure_reading_speed(tmp_path):
    w = World(tmp_path)
    for start in (0.5, 3.0, 6.0, 12.0, 15.0):
        w.cap(start, SHORT, 2.0)
    out = w.run()
    assert out.issues == []
    assert out.metrics["captions_checked"] == 5 and out.metrics[f"{w.s1.id}.captions_checked"] == 3 and out.metrics["max_cps"] == 13.0
    assert caption_summary(out.metrics)["captions_checked"] == 5 and "5 caption(s)" in out.notes[-1]


@pytest.mark.parametrize("style_id", ["professional", "clean", "bold", "news", "documentary"])
def test_captions_made_by_the_caption_engine_are_clean(tmp_path, style_id):
    p = new_project(tmp_path, seconds=30)
    text = ("Silver prices rose sharply last week as investors rushed into safe haven assets after the central bank announced a surprise decision on interest rates. "
            "The deadline for filing is October 15, 2027, and penalties of 25% apply immediately. Experts expect further gains soon, although some analysts remain cautious.")
    s = add_scene(p, 0, 30, text)
    tr = narrate(p)
    p.caption_settings.style_id = style_id
    engine = CaptionEngine(p.caption_settings, {}, (1920, 1080))
    segs = engine.segment(s.id, [[CaptionWord(w.word_id, w.text, w.start, w.end) for w in tr.words]], scene_end=30)
    assert len(segs) >= 5
    for sg in segs:
        add_clip(p, "track_v6", None, sg.start, sg.end - sg.start, scene=s, kind="caption", created_by="AI", text=sg.to_dict(), animation=sg.animation)
    out = run_checker(CaptionChecker(), qc_ctx(p))
    # the engine fills a line up to the frame width (about 50 characters here): QC's 42-character reading guideline may mention that as information, and nothing else
    assert [i for i in out.issues if not (i.code == "caption.overcrowding" and i.severity is Severity.INFO)] == []


# ------------------------------------------------------------------ overflow and safe margins
def test_a_caption_cut_off_by_the_frame_is_an_error_with_a_safe_move(tmp_path):
    w = World(tmp_path)
    cap = w.cap(1.0, "Silver prices rose", position="custom", xy=[1.0, 0.5])
    (i,) = issues_of(w.run(), "caption.overflow")
    assert i.severity is Severity.ERROR and i.category is QCCategory.CAPTION and i.timeline_item_id == cap.id and i.scene_id == w.s1.id
    assert "cut off" in i.title.lower() and i.metrics["fits_safe_area"] is True
    assert i.fix.kind == "caption.safe_margin" and i.auto_fix_available and i.auto_fix_safe
    # applying the recommended position puts the caption inside the safe margins
    cap.text["position_xy"] = i.fix.params["position_xy"]
    after = caption_layout(cap.text, w.p.caption_settings, {}, (1920, 1080))
    assert after.rect.overshoot(Margins.of(w.p.caption_settings, 0.03).rect) == (0.0, 0.0, 0.0, 0.0)
    assert w.run().issues == []


def test_a_caption_wider_than_the_safe_area_but_on_screen_is_a_warning_without_a_position_fix(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "x" * 58, 4.0)  # about 92% of the frame wide: on screen, but inside the 6% margins
    (i,) = issues_of(w.run(), "caption.overflow")
    assert i.severity is Severity.WARNING and i.fix is None and not i.auto_fix_available and i.metrics["fits_safe_area"] is False
    assert "wider than the safe area" in i.title.lower()


def test_a_caption_in_the_safe_margin_gets_the_safe_margin_fix(tmp_path):
    w = World(tmp_path)
    cap = w.cap(1.0, "Short one", 2.0, position="custom", xy=[0.5, 0.97])
    out = w.run()
    (i,) = issues_of(out, "caption.safe_margin")
    assert codes(out) == ["caption.safe_margin"] and i.severity is Severity.WARNING
    assert i.fix.kind == "caption.safe_margin" and i.fix.route is FixRoute.COMMAND and i.fix.params["clip_id"] == cap.id
    assert i.auto_fix_available and i.auto_fix_safe and i.fix.safe  # a small, deterministic move: permitted "auto" by default
    x, y = i.fix.params["position_xy"]
    assert x == 0.5 and 0.85 < y < 0.92


def test_the_project_margins_and_the_qc_minimum_both_apply(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "Short one", 2.0)  # bottom: its text ends exactly on the project's bottom margin
    assert w.run().issues == []
    w.p.caption_settings.safe_margin_bottom = 0.025  # the project's margin is thinner than QC's 3% minimum
    (i,) = issues_of(w.run(), "caption.safe_margin")
    assert i.severity is Severity.NOTICE and "bottom" in i.description
    s = QCSettings()
    s.caption.min_safe_margin = 0.0
    assert w.run(settings=s).issues == []  # QC's own minimum is a setting: with none, the project's margin is all there is


def test_a_user_owned_caption_is_reported_without_an_auto_fix(tmp_path):
    w = World(tmp_path)
    cap = w.cap(1.0, "Short one", 2.0, position="custom", xy=[0.5, 0.97], by="USER")
    (i,) = issues_of(w.run(), "caption.safe_margin")
    assert i.timeline_item_id == cap.id and i.locked and not i.auto_fix_available and not i.auto_fix_safe and "disabled" in i.fix_blocked_reason
    # the other forms of protection behave the same
    for how in ("clip", "track", "scene"):
        w2 = World(tmp_path / how)
        c = w2.cap(1.0, "Short one", 2.0, position="custom", xy=[0.5, 0.97])
        if how == "clip":
            c.locked = True
        elif how == "track":
            w2.p.timeline.get_track("track_v6").locked = True
        else:
            w2.p.presentation_generation.locked_scenes.append(w2.s1.id)
        (j,) = issues_of(w2.run(), "caption.safe_margin")
        assert j.locked and not j.auto_fix_available and j.fix_blocked_reason


def test_the_fix_permission_table_is_honoured(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "Short one", 2.0, position="custom", xy=[0.5, 0.97])
    s = QCSettings()
    s.fix_permissions["caption.safe_margin"] = "never"
    (i,) = issues_of(w.run(settings=s), "caption.safe_margin")
    assert not i.auto_fix_available and i.fix_blocked_reason == "Disabled in QC settings"
    s2 = QCSettings()
    s2.fix_permissions["caption.safe_margin"] = "confirm"
    (j,) = issues_of(w.run(settings=s2), "caption.safe_margin")
    assert j.auto_fix_available and not j.auto_fix_safe


# ------------------------------------------------------------------ collisions
def test_a_caption_on_top_of_an_evidence_box_collides(tmp_path):
    w = World(tmp_path)
    cap = w.cap(1.0, SHORT, 2.5)
    box = w.box(1.5, 2.0)
    out = w.run()
    (i,) = issues_of(out, "caption.collision")
    assert codes(out) == ["caption.collision"] and i.severity is Severity.WARNING and i.timeline_item_id == cap.id and i.scene_id == w.s1.id
    assert i.metrics["other_clip"] == box.id and i.metrics["overlap_ratio"] > 0.5 and i.metrics["overlap_seconds"] == 2.0 and i.confidence == 90.0
    assert i.fix.route is FixRoute.NAVIGATE and not i.auto_fix_safe


def test_a_caption_over_a_text_graphic_collides_but_separate_places_or_times_do_not(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.5)
    w.text(1.2, 2.0, "Warning", pos=(0.5, 0.88))
    (i,) = issues_of(w.run(), "caption.collision")
    assert i.metrics["other_kind"] == "text graphic" and i.confidence == 75.0  # a text rectangle is an estimate: less certain than an exact highlight region
    w2 = World(tmp_path / "apart")
    w2.cap(1.0, SHORT, 2.5)
    w2.text(1.2, 2.0, "Warning", pos=(0.5, 0.3))  # top half: no overlap in space
    w2.box(1.5, 2.0, region=(0.3, 0.2, 0.4, 0.1))
    w2.box(5.0, 2.0)  # the right place, the wrong time
    assert w2.run().issues == []


def test_two_captions_on_different_tracks_collide_once(tmp_path):
    w = World(tmp_path)
    t = w.p.timeline
    from app.timeline.track import Track, TrackKind  # noqa: PLC0415

    t.tracks.append(Track("track_v6b", "Captions 2", TrackKind.CAPTIONS))
    w.cap(1.0, SHORT, 2.5)
    w.cap(1.5, "Another caption here", 2.0, track="track_v6b")
    found = issues_of(w.run(), "caption.collision")
    assert len(found) == 1 and found[0].metrics["other_kind"] == "caption"


# ------------------------------------------------------------------ word timing
def _words(spans):
    return [{"word_id": f"w{i}", "text": f"w{i}", "start": a, "end": b} for i, (a, b) in enumerate(spans)]


def test_out_of_order_or_invalid_word_times_are_an_error(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "one two three", 3.0, words=_words([(1.0, 1.5), (2.0, 2.4), (1.6, 1.9)]))  # the third word starts before the second
    w.cap(5.0, "four five", 2.0, words=_words([(5.0, 5.6), (6.2, 6.0)]))  # the second ends before it starts
    found = issues_of(w.run(), "caption.words")
    assert len(found) == 2 and all(i.severity is Severity.ERROR and i.fix is None for i in found)
    assert "out of order" in found[0].description and "no usable" in found[1].description


def test_overlapping_words_and_a_caption_that_leaves_before_its_last_word_are_warnings(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "one two three", 3.0, words=_words([(1.0, 1.9), (1.5, 2.4), (2.5, 3.0)]))  # the second word starts 0.4 s before the first ends
    w.cap(5.0, "four five six", 2.0, words=_words([(5.0, 5.5), (5.6, 6.2), (6.3, 7.0)]))  # the caption ends at 7.0: fine
    w.cap(8.0, "seven eight", 1.0, words=_words([(8.0, 8.4), (8.5, 9.6)]))  # the caption leaves at 9.0, its last word ends at 9.6
    found = issues_of(w.run(), "caption.words")
    assert len(found) == 2 and all(i.severity is Severity.WARNING for i in found)
    assert "overlap" in found[0].description and "600 ms before its last word ends" in found[1].description


def test_a_clean_word_list_and_a_plain_shift_are_not_word_findings(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "one two three", 3.0, words=_words([(1.0, 1.9), (1.95, 2.4), (2.5, 3.0)]))
    out = w.run()
    assert issues_of(out, "caption.words") == []
    # a caption that is simply late against the voice is the sync checker's drift, not a word-timing problem
    w2 = World(tmp_path / "late")
    cap = w2.cap(1.0, SHORT, 2.0)
    cap.timeline_start, cap.text["start"] = 2.4, 2.4  # the words stay where they were spoken: 1.4 s earlier
    assert w2.run().issues == []


def test_a_caption_past_the_end_of_the_voice_over_is_reported(tmp_path):
    w = World(tmp_path)
    w.cap(18.5, SHORT, 2.2)  # ends at 20.7: inside the one second of grace
    assert w.run().issues == []
    w.cap(19.0, "Thanks for watching", 2.5)  # ends at 21.5
    w.cap(22.0, "Well beyond the end", 2.0)
    found = issues_of(w.run(), "caption.range")
    assert [i.severity for i in found] == [Severity.WARNING, Severity.ERROR] and all(i.scene_id == w.s2.id for i in found)


# ------------------------------------------------------------------ density
def test_too_many_words_lines_or_characters_are_overcrowding(tmp_path):
    w = World(tmp_path)
    many = " ".join(["ab"] * 18)
    w.cap(0.5, many, 6.0, lines=[" ".join(["ab"] * 9)] * 2)  # 18 words (limit 14), short lines
    w.cap(7.0, "a b c d e f", 3.0, lines=["a b", "c d", "e f"])  # three lines (limit 2)
    w.cap(12.0, "x" * 50, 5.0)  # one line of 50 characters (limit 42), one word
    found = issues_of(w.run(), "caption.overcrowding")
    assert len(found) == 3 and [i.metrics["words"] for i in found] == [18, 6, 1]
    assert "18 words at once (limit 14)" in found[0].description and "3 lines (limit 2)" in found[1].description and "50 characters (limit 42)" in found[2].description
    assert found[0].severity is Severity.WARNING and found[1].severity is Severity.WARNING and found[2].severity is Severity.INFO  # a long line the style's layout fits is information


def test_a_long_line_alone_is_a_notice_until_it_is_far_past_the_limit(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "word " * 8 + "ok", 4.0)  # exactly 42 characters: at the limit is fine
    w.cap(6.0, "y" * 46, 4.0)  # 46 > 42 but the style's own layout (51 characters) fits it: information only
    w.cap(11.0, "y" * 56, 4.0)  # just past the style's layout: a notice
    w.cap(15.0, "y" * 70, 4.0, lines=["y" * 70])  # 1.37 times the layout: a warning
    w.cap(19.0, "y" * 95, 1.0, lines=["y" * 95])  # nearly twice: an error
    found = issues_of(w.run(), "caption.overcrowding")
    assert [i.severity for i in found] == [Severity.INFO, Severity.NOTICE, Severity.WARNING, Severity.ERROR]


def test_overcrowding_offers_a_future_setting_only_when_the_project_setting_allows_it(tmp_path):
    w = World(tmp_path)
    many = " ".join(["ab"] * 18)
    w.cap(0.5, many, 6.0, lines=[" ".join(["ab"] * 9)] * 2)
    assert issues_of(w.run(), "caption.overcrowding")[0].fix is None  # the project's own limit (12 words) is already stricter than QC's 14: nothing to change
    w.p.caption_settings.max_words = 20
    (i,) = issues_of(w.run(), "caption.overcrowding")
    assert i.fix.kind == "caption.restyle" and i.fix.params == {"field": "max_words", "value": 14} and not i.auto_fix_safe and i.fix.needs_confirmation
    assert "from now on" in i.fix.summary
    w.p.caption_settings.user_set.append("max_words")  # a setting the user chose is never changed by QC
    (j,) = issues_of(w.run(), "caption.overcrowding")
    assert j.fix is None and "yourself" in j.fix_blocked_reason


# ------------------------------------------------------------------ reading time
def test_flicker_is_one_finding_for_a_run_of_very_short_captions(tmp_path):
    w = World(tmp_path)
    first = w.cap(1.0, "Yes", 0.3)
    w.cap(1.35, "Right", 0.3)
    w.cap(1.7, "Now", 0.3)
    out = w.run()
    (i,) = issues_of(out, "caption.flicker")
    assert codes(out) == ["caption.flicker"] and i.severity is Severity.WARNING and i.timeline_item_id == first.id and i.metrics["captions"] == 3
    assert i.start_time == 1.0 and i.end_time == 2.0 and i.scene_id == w.s1.id  # no extra "too short" for the same captions


def test_two_short_captions_or_short_captions_separated_by_pauses_do_not_flicker(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "Yes", 0.3)
    w.cap(1.35, "Right", 0.3)  # only two in a row
    w.cap(4.0, "Now", 0.3)
    w.cap(5.5, "Then", 0.3)  # separated by pauses of more than a flicker length
    w.cap(7.0, "Done", 0.3)
    out = w.run()
    assert issues_of(out, "caption.flicker") == [] and len(issues_of(out, "caption.too_short")) == 5


def test_a_caption_that_is_too_short_or_too_fast(tmp_path):
    w = World(tmp_path)
    w.s1.narration = " ".join(["word"] * 40)  # a fast narrator: about four words a second
    narrate(w.p)
    w.cap(1.0, "Yes", 0.3)
    fast = w.cap(5.0, "Silver prices rose sharply last week amid strong demand", 1.5)  # 54 characters in 1.5 s = 36 cps
    out = w.run()
    (short,) = issues_of(out, "caption.too_short")
    (quick,) = issues_of(out, "caption.too_fast")
    assert short.severity is Severity.NOTICE and short.metrics["duration"] == 0.3
    assert quick.severity is Severity.ERROR and quick.timeline_item_id == fast.id and quick.metrics["cps"] == 36.67 and quick.metrics["seconds_needed"] == 2.62
    assert "free after it" in quick.suggested_fix  # nothing follows the caption: it can be held
    assert "narrator speaks about 4.0 words per second" in quick.description  # the transcript says how fast the voice is here
    assert issues_of(w.run(), "caption.too_short")[0].fix is None


def test_reading_speed_bands_and_a_clean_caption(tmp_path):
    w = World(tmp_path)
    w.cap(0.5, "x" * 24, 1.1)  # 21.8 cps: just over the limit: a notice
    w.cap(3.0, "x" * 28, 1.1)  # 25.5 cps: a warning
    w.cap(6.0, "x" * 24, 2.0)  # 12 cps: fine
    found = issues_of(w.run(), "caption.too_fast")
    assert [i.severity for i in found] == [Severity.NOTICE, Severity.WARNING]


def test_a_fast_caption_followed_closely_cannot_be_held_and_the_advice_says_so(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "Silver prices rose sharply last week amid strong demand", 1.5)
    w.cap(2.55, "Next caption", 2.0)
    (i,) = issues_of(w.run(), "caption.too_fast")
    assert "no room" in i.suggested_fix


def test_too_fast_offers_a_slower_reading_speed_for_future_captions_when_the_setting_is_looser(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, "Silver prices rose sharply last week amid strong demand", 1.5)
    assert issues_of(w.run(), "caption.too_fast")[0].fix is None  # the engine's limit (17 characters per second) is already stricter than QC's 21
    w.p.caption_settings.reading_speed = 1.4
    (i,) = issues_of(w.run(), "caption.too_fast")
    assert i.fix.kind == "caption.restyle" and i.fix.params["field"] == "reading_speed" and i.fix.params["value"] == 1.11 and not i.auto_fix_safe  # 1.11 x 17 = 18.9 characters per second: inside QC's 21


# ------------------------------------------------------------------ style: font size, contrast, highlights, animation
def test_a_small_caption_font_is_reported_once_per_scene_and_style(tmp_path):
    w = World(tmp_path)
    for start in (1.0, 4.0, 7.0):
        w.cap(start, SHORT, 2.0, overrides={"size_rel": 0.025})
    w.cap(12.0, SHORT, 2.0, overrides={"size_rel": 0.025})
    found = issues_of(w.run(), "caption.small_font")
    assert [i.scene_id for i in found] == [w.s1.id, w.s2.id] and found[0].metrics["captions"] == 3 and "2 more caption(s)" in found[0].description
    assert found[0].severity is Severity.WARNING and found[0].fix.kind == "caption.restyle" and found[0].fix.params == {"field": "large_text", "value": True}
    tiny = World(tmp_path / "tiny")
    tiny.cap(1.0, SHORT, 2.0, overrides={"size_rel": 0.012})
    (i,) = issues_of(tiny.run(), "caption.small_font")
    assert i.severity is Severity.ERROR and i.fix is None  # large text would still be below the minimum: no fix is offered


def test_low_contrast_text_on_a_known_box_is_certain_and_without_a_box_it_is_a_guess(tmp_path):
    w = World(tmp_path)
    w.p.caption_styles["lowc"] = CaptionStyle("lowc", "Low contrast", color="#FFFFFF", background="box", background_color="#FFFFFF", background_opacity=1.0, highlight_color="#FFFFFF")
    w.cap(1.0, SHORT, 2.0, style_id="lowc")
    (i,) = issues_of(w.run(), "caption.low_contrast")
    assert i.severity is Severity.ERROR and i.confidence == 90.0 and i.metrics["backdrop"] == "box" and i.metrics["contrast"] == 1.0
    assert i.fix.kind == "caption.restyle" and i.fix.params == {"field": "high_contrast", "value": True}
    w2 = World(tmp_path / "nobox")
    w2.p.caption_styles["grey"] = CaptionStyle("grey", "Grey", color="#808080", background="none", shadow=False, outline_width=0.0, highlight_color="#808080")
    w2.cap(1.0, SHORT, 2.0, style_id="grey")
    (j,) = issues_of(w2.run(), "caption.low_contrast")
    assert j.severity is Severity.WARNING and j.confidence == 50.0 and j.metrics["backdrop"] == "none" and "not analysed" in j.description  # the footage under the text is unknown


def test_the_built_in_styles_and_high_contrast_mode_pass_the_contrast_check(tmp_path):
    for style_id in ("professional", "clean", "bold", "news", "documentary", "minimal"):
        w = World(tmp_path / style_id)
        w.cap(1.0, SHORT, 2.0, style_id=style_id)
        assert w.run().issues == [], style_id
    w = World(tmp_path / "hc")
    w.p.caption_settings.high_contrast = True
    w.cap(1.0, SHORT, 2.0)
    assert w.run().issues == []


def test_a_weak_highlight_colour_is_a_notice_and_too_many_highlights_too(tmp_path):
    w = World(tmp_path)
    w.p.caption_styles["pale"] = CaptionStyle("pale", "Pale", color="#FFFFFF", background="none", shadow=False, outline_width=0.0, highlight_color="#909090")
    w.cap(1.0, SHORT, 2.0, style_id="pale")
    (i,) = issues_of(w.run(), "caption.highlight")
    assert i.severity is Severity.NOTICE and "highlight colour" in i.description
    w2 = World(tmp_path / "many")
    marks = [{"word_index": k, "category": "NUMBER", "style": "COLOR_CHANGE", "reason": ""} for k in range(4)]
    w2.cap(1.0, "one two three four five six", 3.0, emphasis=marks)
    w2.cap(5.0, "one two three four five six", 3.0, emphasis=marks[:2])  # two marks: the most the engine ever makes
    (j,) = issues_of(w2.run(), "caption.highlight")
    assert j.metrics == {"highlighted": 4, "words": 6} and j.severity is Severity.NOTICE


def test_a_long_caption_animation_and_motion_under_reduced_motion(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 3.0, anim={"in": spec("fade_in", duration=1.5), "out": spec("fade_out", duration=0.1)})
    (i,) = issues_of(w.run(), "caption.animation")
    assert i.severity is Severity.NOTICE and "1.50 s" in i.description
    w2 = World(tmp_path / "rm")
    w2.p.caption_settings.reduced_motion = True
    w2.cap(1.0, SHORT, 3.0, anim={"in": spec("slide_up", duration=0.3), "out": spec("fade_out", duration=0.1)})
    w2.cap(5.0, SHORT, 3.0)  # fades only
    (j,) = issues_of(w2.run(), "caption.animation")
    assert j.severity is Severity.WARNING and "reduced motion" in j.title.lower() and j.scene_id == w2.s1.id
    w3 = World(tmp_path / "rm-off")
    w3.cap(1.0, SHORT, 3.0, anim={"in": spec("slide_up", duration=0.3), "out": spec("fade_out", duration=0.1)})
    assert w3.run().issues == []  # without reduced motion a short slide is fine


# ------------------------------------------------------------------ consistency
def test_a_caption_that_differs_from_both_neighbours_is_inconsistent(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.0)
    odd = w.cap(3.5, SHORT, 2.0, style_id="bold")
    w.cap(6.0, SHORT, 2.0)
    (i,) = issues_of(w.run(), "caption.inconsistency")
    assert i.severity is Severity.NOTICE and i.timeline_item_id == odd.id and i.confidence == 70.0 and i.metrics["differs"] == ["style", "size"]
    assert i.fix is None
    odd.created_by = "USER"  # the user may have meant it: a weaker judgement
    (j,) = issues_of(w.run(), "caption.inconsistency")
    assert j.confidence == 60.0 and "intentional" in j.description and not j.auto_fix_available


def test_neighbours_that_disagree_or_gaps_between_captions_are_not_inconsistent(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.0)
    w.cap(3.5, SHORT, 2.0, style_id="bold")
    w.cap(6.0, SHORT, 2.0, style_id="news")  # the neighbours differ from each other: nothing to compare against
    assert w.run().issues == []
    w2 = World(tmp_path / "gap")
    w2.cap(0.5, SHORT, 1.5)
    w2.cap(7.0, SHORT, 1.5, style_id="bold")  # more than four seconds from both
    w2.cap(9.0, SHORT, 0.9 + 0.1)
    assert issues_of(w2.run(), "caption.inconsistency") == []


# ------------------------------------------------------------------ scope, determinism, protection
def test_drift_against_the_voice_is_never_reported_here(tmp_path):
    w = World(tmp_path)
    w.cap(1.9, SHORT, 2.0)  # the narration says "Silver" at 0.1 s: this caption trails by 1.8 s
    out = w.run()
    assert out.issues == [] and not [i for i in out.issues if "drift" in i.code or i.code.startswith("sync.")]


def test_every_issue_is_scene_scoped_and_a_scene_filter_analyses_only_that_scene(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.0, position="custom", xy=[0.5, 0.97])
    w.cap(12.0, "x" * 58, 4.0)
    full = w.run()
    assert full.issues and all(i.scene_id for i in full.issues) and {i.scene_id for i in full.issues} == {w.s1.id, w.s2.id}
    only2 = w.run(scene_filter=[w.s2.id])
    assert {i.scene_id for i in only2.issues} == {w.s2.id} and f"{w.s1.id}.captions_checked" not in only2.metrics
    assert {i.code for i in full.issues} <= ALLOWED


def test_only_the_changed_scene_is_reanalysed_and_ignores_survive_by_fingerprint(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.0, position="custom", xy=[0.5, 0.97])
    w.cap(12.0, "x" * 58, 4.0)
    chk = CaptionChecker()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(w.p))
    assert first.run.checkers["caption"].state is CheckerState.DONE and sorted(i.code for i in first.run.issues) == ["caption.overcrowding", "caption.overflow", "caption.safe_margin"]
    h1 = chk.scene_input_hash(qc_ctx(w.p), w.s1.id)
    w.cap(15.0, "x" * 58, 4.0)  # a new problem in scene 2 only
    assert chk.scene_input_hash(qc_ctx(w.p), w.s1.id) == h1
    second = eng.run(qc_ctx(w.p), previous=PreviousState(first.run.issues, first.cache))
    st = second.run.checkers["caption"]
    assert st.reused_scenes == 1 and st.analyzed_scenes == 1 and len(second.run.issues) == 5
    fp = {i.fingerprint for i in first.run.issues}
    assert fp <= {i.fingerprint for i in second.run.issues}  # the same findings keep their identity across runs (that is what an Ignore is matched on)


def test_the_checker_never_mutates_the_project(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.0, position="custom", xy=[0.5, 0.97])
    w.cap(5.0, "x" * 58, 4.0)
    w.box(5.5, 2.0)
    before = w.p.to_document()
    out = w.run()
    assert len(out.issues) >= 3 and w.p.to_document() == before


def test_switched_off_captions_and_hidden_tracks_are_not_checked(tmp_path):
    w = World(tmp_path)
    w.cap(1.0, SHORT, 2.0, position="custom", xy=[0.5, 0.97])
    w.p.caption_settings.enabled = False
    out = w.run()
    assert out.issues == [] and "switched off" in out.notes[0]
    w.p.caption_settings.enabled = True
    w.p.timeline.get_track("track_v6").hidden = True
    assert w.run().issues == []


def test_caption_fixes_use_the_catalog_constructors_and_their_parameter_names(tmp_path):
    w = World(tmp_path)
    cap = w.cap(1.0, "Short one", 2.0, position="custom", xy=[0.5, 0.97])
    (i,) = issues_of(w.run(), "caption.safe_margin")
    expected = fix_catalog.caption_safe_margin(cap.id, i.fix.params["position_xy"], QCSettings())
    assert (i.fix.kind, i.fix.params, i.fix.route) == (expected.kind, expected.params, expected.route)


# ------------------------------------------------------------------ the real pipeline
def _generate(ws, parts):
    ws.presentation.generate(parts)
    assert ws.jobs.wait_idle(120)


@needs_ffmpeg
def test_captions_made_by_the_real_pipeline_raise_nothing_above_information(pres_ws):
    _generate(pres_ws, ["CAPTIONS", "GRAPHICS"])
    out = run_checker(CaptionChecker(), qc_ctx(pres_ws.project))
    assert out.metrics["captions_checked"] >= 20
    # the generator fills a line up to the frame width (about 50 characters), QC's reading guideline is 42: that is information, not a finding
    assert {(i.code, i.severity) for i in out.issues} <= {("caption.overcrowding", Severity.INFO)}
    assert all(i.title == "Caption has a long line" for i in out.issues)
