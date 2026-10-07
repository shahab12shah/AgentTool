"""TextChecker and the geometry helpers: overflow / clipped / safe area, readability, brevity, overlap and evidence collisions, duplicates, timing, hierarchy, excessive text, and
the fact review (a value on screen that is not in the narration is a *review* request: category FACT_REVIEW, never a fix, never a claim that anything is wrong).

Two scenes (0-10 s and 10-20 s). Scene 1's narration says "forty two percent", "$1,250" and "April fifteenth, twenty twenty seven"; Scene 2 says nothing numeric.
"""

from __future__ import annotations

import math

import pytest

from app.analysis.models import Claim, ClaimType, NumberKind, NumericMention, VisualIntent, VisualType
from app.presentation.models import CaptionSettings
from app.qc import fix_catalog
from app.qc import text_checker as tc
from app.qc.geometry import (FRAME, MARGIN_EPS, Margins, Rect, assumed_backdrop, blend, caption_layout, contrast_ratio, fit_inside, luminance, parse_color, region_rect, text_contrast,
                             text_layout, text_width_em)
from app.qc.issue_model import CheckerState, FixRoute, QCCategory
from app.qc.qc_engine import PreviousState, QCEngine
from app.qc.settings import QCSettings
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_clip, add_scene, codes, find, narrate, new_project, qc_ctx, run_checker
from app.qc.text_checker import TextChecker, judge_mention, name_runs, numeric_mentions, spoken_of, text_summary
from app.tests.conftest import needs_ffmpeg
from app.timeline.clip import Clip

NARRATION_1 = "Silver prices rose forty two percent last week. The fee is $1,250 and the deadline is April fifteenth, twenty twenty seven. Jane Smith of the Treasury agreed."
NARRATION_2 = "Demand keeps growing across Asia."
ALLOWED = {"text.overflow", "text.clipped", "text.safe_area", "text.unreadable", "text.too_brief", "text.overlap", "text.collision", "text.duplicate", "text.timing", "text.hierarchy",
           "text.excessive", "text.value_mismatch", "text.evidence_unsupported"}


class World:
    def __init__(self, tmp_path):
        self.p = p = new_project(tmp_path, seconds=20)
        self.s1 = add_scene(p, 0, 10, NARRATION_1, importance=0.8)
        self.s2 = add_scene(p, 10, 20, NARRATION_2)
        narrate(p)
        self.s1.numbers = [NumericMention("forty two percent", NumberKind.PERCENTAGE, 42.0, "", True, []), NumericMention("$1,250", NumberKind.DOLLAR_AMOUNT, 1250.0, "", False, []),
                           NumericMention("April fifteenth", NumberKind.DATE, None, "", True, [])]
        self.s1.claims = [Claim("sent_0000_c0", "Silver prices rose forty two percent last week.", ClaimType.NUMBER, "sent_0000", True)]

    def scene_of(self, start: float):
        return self.s1 if start < 10 else self.s2

    def text(self, start: float, dur: float, content: str, *, style: str = "LABEL", pos=(0.5, 0.42), size: int = 56, by: str = "AI", align: str = "center", background: str = "none",
             counter=None, emphasis: str = "", scene=None, **extra) -> Clip:
        body = {"text_id": "t", "content": content, "start": start, "duration": dur, "position": list(pos), "style": style, "emphasis": emphasis, "animation": "fade", "font": "Sans",
                "size": size, "alignment": align, "opacity": 1.0, "background": background, "variant": style, "counter": counter}
        body.update(extra)
        return add_clip(self.p, "track_v5", None, start, dur, scene=scene or self.scene_of(start), kind="text", created_by=by, slot="text:0", text=body)

    def number(self, start: float, content: str = "42%", dur: float = 2.5, **kw) -> Clip:
        kw.setdefault("style", "NUMBER_CARD")
        kw.setdefault("size", 88)
        kw.setdefault("emphasis", "NUMBER_CARD")
        return self.text(start, dur, content, **kw)

    def box(self, start: float, dur: float, region=(0.3, 0.3, 0.4, 0.2), *, by: str = "AI") -> Clip:
        return add_clip(self.p, "track_v4", None, start, dur, scene=self.scene_of(start), kind="graphic", created_by=by, slot="evidence:0",
                        effects={"highlight": {"region": list(region), "style": "box", "darken_surround": True}})

    def run(self, **kw):
        return run_checker(TextChecker(), qc_ctx(self.p, **kw))


def issues_of(out, code):
    return find(out, code)


# ====================================================================== geometry helper
def test_rect_arithmetic_overlap_and_overshoot():
    a, b = Rect(0.2, 0.2, 0.6, 0.6), Rect(0.5, 0.5, 0.9, 0.9)
    assert math.isclose(a.width, 0.4) and math.isclose(a.area, 0.16) and (a.cx, a.cy) == (0.4, 0.4)
    inter = a.intersection(b)
    assert inter == Rect(0.5, 0.5, 0.6, 0.6) and a.intersection(Rect(0.7, 0.7, 0.8, 0.8)) is None
    assert math.isclose(a.overlap_ratio(b), 0.01 / 0.16) and a.overlap_ratio(Rect(0.3, 0.3, 0.4, 0.4)) == 1.0  # the smaller one lies completely inside
    assert Rect(-0.1, 0.5, 1.2, 0.95).overshoot(FRAME) == (0.1, 0.0, pytest.approx(0.2), 0.0)
    moved = a.moved_to(0.5, 0.5)
    assert [round(v, 9) for v in (moved.x0, moved.y0, moved.x1, moved.y1)] == [0.3, 0.3, 0.7, 0.7] and a.expanded(0.1, 0.0) == Rect(0.1, 0.2, 0.7, 0.6)


def test_margins_use_the_project_setting_but_never_less_than_the_qc_minimum():
    cs = CaptionSettings(safe_margin_left=0.06, safe_margin_right=0.01, safe_margin_top=0.06, safe_margin_bottom=0.08)
    m = Margins.of(cs, 0.03)
    assert (m.left, m.top, m.right, m.bottom) == (0.06, 0.06, 0.03, 0.08)
    assert m.rect == Rect(0.06, 0.06, 0.97, 0.92)


def test_fit_inside_moves_the_least_and_refuses_what_cannot_fit():
    safe = Rect(0.06, 0.06, 0.94, 0.92)
    cx, cy = fit_inside(Rect(0.9, 0.5, 1.1, 0.6), safe)
    assert (round(cx, 3), round(cy, 3)) == (0.84, 0.55)  # pulled left only
    assert fit_inside(Rect(0.0, 0.0, 0.95, 0.1), safe) is None  # wider than the safe area: no position helps


def test_caption_placement_follows_the_renderer_anchor_rules():
    cs = CaptionSettings()
    data = {"text": "Silver prices rose", "lines": ["Silver prices rose"], "position": "bottom"}
    bottom = caption_layout(data, cs, {}, (1920, 1080))
    assert math.isclose(bottom.rect.y1, 1.0 - cs.safe_margin_bottom) and math.isclose(bottom.rect.cx, 0.5) and bottom.boxed and bottom.anchor == "bottom"
    assert bottom.box.width > bottom.rect.width and bottom.box.height > bottom.rect.height  # the box border hides more than the text
    top = caption_layout({**data, "position": "top"}, cs, {}, (1920, 1080))
    assert math.isclose(top.rect.y0, cs.safe_margin_top)
    centre = caption_layout({**data, "position": "center"}, cs, {}, (1920, 1080))
    assert math.isclose(centre.rect.cy, 0.5)
    custom = caption_layout({**data, "position": "custom", "position_xy": [0.3, 0.4]}, cs, {}, (1920, 1080))
    assert math.isclose(custom.rect.cx, 0.3) and math.isclose(custom.rect.cy, 0.4) and custom.anchor == "custom"
    unknown = caption_layout({**data, "position": "somewhere"}, cs, {}, (1920, 1080))
    assert unknown.anchor == "bottom" and caption_layout({"text": "", "lines": []}, cs, {}, (1920, 1080)) is None


def test_caption_size_scales_with_the_font_the_lines_and_the_canvas():
    cs = CaptionSettings()
    one = caption_layout({"text": "x" * 20, "lines": ["x" * 20]}, cs, {}, (1920, 1080))
    two = caption_layout({"text": "x" * 20, "lines": ["x" * 10, "x" * 10]}, cs, {}, (1920, 1080))
    assert two.rect.height > 1.9 * one.rect.height and two.rect.width < one.rect.width
    big = caption_layout({"text": "x" * 20, "lines": ["x" * 20], "style_overrides": {"size_rel": 0.1}}, cs, {}, (1920, 1080))
    assert math.isclose(big.rect.width, 2 * one.rect.width, rel_tol=1e-6)
    portrait = caption_layout({"text": "x" * 20, "lines": ["x" * 20]}, cs, {}, (1080, 1920))  # the same font height in a narrow frame takes more of its width
    assert portrait.rect.width > one.rect.width
    assert text_width_em("abc", bold=True) > text_width_em("abc") and text_width_em("abc", uppercase=True) > text_width_em("abc")


def test_text_clip_geometry_centred_left_aligned_boxed_and_with_a_subtitle():
    d = {"content": "Silver", "position": [0.5, 0.4], "size": 100, "alignment": "center", "background": "none"}
    c = text_layout(d, (1920, 1080))
    assert math.isclose(c.rect.cx, 0.5) and math.isclose(c.rect.cy, 0.4) and not c.boxed and not c.left_aligned
    left = text_layout({**d, "alignment": "left", "position": [0.1, 0.4]}, (1920, 1080))
    assert math.isclose(left.rect.x0, 0.1) and left.left_aligned
    boxed = text_layout({**d, "background": "box"}, (1920, 1080))
    assert boxed.boxed and boxed.box.width > boxed.rect.width
    sub = text_layout({**d, "subtitle": "a longer subtitle line"}, (1920, 1080))
    assert sub.rect.height > c.rect.height and len(sub.lines) == 2
    multi = text_layout({**d, "content": "one\ntwo\nthree"}, (1920, 1080))
    assert math.isclose(multi.rect.height, 3 * c.rect.height)
    bold = text_layout({**d, "emphasis": "NUMBER_CARD"}, (1920, 1080))
    assert bold.bold and bold.rect.width > c.rect.width
    assert text_layout({"content": "  "}, (1920, 1080)) is None and text_layout({"content": "x", "position": ["a", 1]}, (1920, 1080)) is None


def test_evidence_region_matches_what_the_renderer_draws():
    clip = Clip("c", "track_v4", "", 0, 2, kind="graphic", effects={"highlight": {"region": [0.2, 0.3, 0.5, 0.25]}})
    assert region_rect(clip) == Rect(0.2, 0.3, 0.7, 0.55)
    assert region_rect(Clip("c", "track_v4", "", 0, 2, kind="graphic", effects={"highlight": {"style": "box"}})) == Rect(0.2, 0.3, 0.8, 0.5)  # the renderer's default region
    assert region_rect(Clip("c", "track_v4", "", 0, 2, kind="graphic", effects={})) is None  # draws nothing
    assert region_rect(Clip("c", "track_v4", "", 0, 2, kind="graphic", effects={"highlight": {"region": [0.2, 0.3, 0.0, 0.25]}})) is None


def test_contrast_helpers_follow_wcag():
    assert math.isclose(contrast_ratio((255, 255, 255), (0, 0, 0)), 21.0) and math.isclose(contrast_ratio((10, 20, 30), (10, 20, 30)), 1.0)
    assert parse_color("#fff") == (255.0, 255.0, 255.0) and parse_color("#FF8000") == (255.0, 128.0, 0.0) and parse_color("nonsense") is None and parse_color(None) is None
    assert blend((0, 0, 0), (200, 200, 200), 0.5) == (100.0, 100.0, 100.0) and luminance((255, 255, 255)) == pytest.approx(1.0)
    box, kind = assumed_backdrop(box=((0.0, 0.0, 0.0), 0.6))
    assert kind == "box" and box[0] < 60
    halo, kind2 = assumed_backdrop(halo=((0.0, 0.0, 0.0), 0.4))
    assert kind2 == "halo" and assumed_backdrop()[1] == "none" and halo[0] > box[0]
    assert text_contrast((255, 255, 255), 1.0, box) > 7 and text_contrast((255, 255, 255), 0.0, box) == pytest.approx(1.0)  # fully transparent text is invisible
    assert MARGIN_EPS > 0


# ====================================================================== number / date extraction and comparison
def test_numbers_and_dates_are_extracted_with_the_analysis_packages_extractor():
    found = {(m.kind.value, m.value) for m in numeric_mentions("$1,250 and 42% on April 15 2027, or forty two percent")}
    assert ("DOLLAR_AMOUNT", 1250.0) in found and ("PERCENTAGE", 42.0) in found and ("DATE", None) in found
    assert [(m.kind.value, m.value) for m in numeric_mentions("$1.2 million")] == [("DOLLAR_AMOUNT", 1_200_000.0)]  # a written scale word is part of the number
    assert numeric_mentions("") == [] and numeric_mentions("no figures here") == []


def test_spoken_values_come_from_the_narration_the_script_and_the_scene_numbers(tmp_path):
    w = World(tmp_path)
    sp = spoken_of(w.s1)
    assert {42.0, 1250.0, 15.0, 2027.0} <= sp.values and 4 in sp.months and "treasury" in sp.tokens and sp.has_narration
    assert not spoken_of(w.s2).values and spoken_of(w.s2).has_narration
    w.s2.script_text = "Prices fell 7.5% in March."  # the script counts too
    assert 7.5 in spoken_of(w.s2).values and 3 in spoken_of(w.s2).months


def test_judge_mention_distinguishes_a_different_value_from_one_that_is_not_spoken_at_all(tmp_path):
    w = World(tmp_path)
    sp1, sp2 = spoken_of(w.s1), spoken_of(w.s2)
    ment = lambda t: numeric_mentions(t)[0]  # noqa: E731
    assert judge_mention(ment("42%"), sp1) is None and judge_mention(ment("$1,250"), sp1) is None and judge_mention(ment("April 15, 2027"), sp1) is None
    assert judge_mention(ment("43%"), sp1) == "differs" and judge_mention(ment("April 16, 2027"), sp1) == "differs" and judge_mention(ment("May 15, 2027"), sp1) == "differs"
    assert judge_mention(ment("42%"), sp2) == "absent"
    assert judge_mention(ment("1.2 million"), spoken_of(_scene("about one point two million people"))) is None  # spoken numbers are parsed too


def _scene(text: str):
    from app.analysis.models import Origin, Scene, SceneStatus  # noqa: PLC0415

    return Scene("s", "1", 0, 10, text, [], "", "", 0.5, 0.9, SceneStatus.READY, Origin.AI)


def test_name_runs_ignore_text_in_capitals_and_filler_words():
    assert name_runs("Jane Smith, Treasury Secretary") == [["Jane", "Smith"], ["Treasury", "Secretary"]]
    assert name_runs("SILVER IS RUNNING OUT") == [] and name_runs("the market of Asia") == [["Asia"]]


# ====================================================================== the checker: contract and clean cases
def test_checker_declares_what_it_is_and_its_cache_key_accurately(tmp_path):
    chk = TextChecker()
    assert (chk.id, chk.scene_local, chk.expensive) == ("text", True, False) and set(chk.categories) == {QCCategory.TEXT, QCCategory.FACT_REVIEW}
    assert set(chk.domains) == {"timeline", "scenes", "captions"}
    w = World(tmp_path)
    base = chk.input_hash(qc_ctx(w.p))
    s = QCSettings()
    s.sync.major_ms = 900  # not read
    assert chk.input_hash(qc_ctx(w.p, s)) == base
    s2 = QCSettings()
    s2.caption.min_safe_margin = 0.05
    assert chk.input_hash(qc_ctx(w.p, s2)) != base
    sig1 = chk.scene_input_hash(qc_ctx(w.p), w.s1.id)
    w.s1.numbers.append(NumericMention("7%", NumberKind.PERCENTAGE, 7.0, "", False, []))  # the scene's figures are part of the fact review's answer
    assert chk.input_hash(qc_ctx(w.p)) != base and chk.scene_input_hash(qc_ctx(w.p), w.s1.id) != sig1
    assert chk.scene_input_hash(qc_ctx(w.p), w.s2.id) == TextChecker().scene_input_hash(qc_ctx(w.p), w.s2.id)


def test_clean_text_reports_nothing(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "42%")  # spoken as "forty two percent"
    w.text(7.0, 2.5, "April 15, 2027", style="DATE", pos=(0.5, 0.40))
    w.text(4.0, 2.0, "Jane Smith", style="LOWER_THIRD", pos=(0.08, 0.80), align="left", background="box", size=48)  # the name is spoken in the scene
    w.number(12.0, "$1,250", style="NUMBER_CARD")  # not spoken in scene 2, but the scene says nothing numeric: see the fact review tests
    out = w.run()
    assert codes(out) == ["text.value_mismatch"] and out.issues[0].scene_id == w.s2.id
    w.s2.script_text = "The fee is $1,250."  # the script says it: nothing left to review
    out = w.run()
    assert out.issues == [] and out.metrics["texts_checked"] == 4 and out.metrics["values_checked"] == 3
    assert text_summary(out.metrics) == {"texts_checked": 4, "graphics_checked": 0} and "4 text clip(s)" in out.notes[-1]


# ====================================================================== position
def test_text_cut_off_by_the_frame_is_an_error(tmp_path):
    w = World(tmp_path)
    clip = w.text(1.0, 3.0, "Silver prices", pos=(0.97, 0.5))
    (i,) = issues_of(w.run(), "text.clipped")
    assert i.severity is Severity.ERROR and i.category is QCCategory.TEXT and i.timeline_item_id == clip.id and i.scene_id == w.s1.id and "right" in i.description
    assert i.fix.route is FixRoute.NAVIGATE and not i.auto_fix_safe


def test_text_larger_than_the_safe_area_is_a_warning_and_one_in_the_margin_a_smaller_one(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 4.0, "x" * 69, size=48)  # about 90% of the frame wide: on screen, but wider than the 88% safe area
    w.text(6.0, 3.0, "Silver", pos=(0.03, 0.5), align="left")  # starts 3% from the edge: inside the 6% margin, small enough to move
    over = issues_of(w.run(), "text.overflow")
    margin = issues_of(w.run(), "text.safe_area")
    assert len(over) == 1 and over[0].severity is Severity.WARNING and "larger than the safe area" in over[0].title
    assert len(margin) == 1 and margin[0].severity is Severity.WARNING and "left" in margin[0].description and issues_of(w.run(), "text.clipped") == []


def test_a_user_owned_text_is_still_reported_but_never_auto_fixed(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 3.0, "Silver prices", pos=(0.97, 0.5), by="USER")
    (i,) = issues_of(w.run(), "text.clipped")
    assert i.locked and not i.auto_fix_available and not i.auto_fix_safe and "disabled" in i.fix_blocked_reason


# ====================================================================== readability
def test_small_text_is_reported_and_numbers_are_held_to_a_larger_minimum(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 3.0, "tiny label", size=24)  # 2.2% of the frame height: below the 2.8% minimum
    w.number(4.0, "42%", size=40)  # 3.7%: fine for a label, below the 4% a figure needs
    w.text(7.0, 3.0, "microscopic", size=12)  # 1.1%: less than half the minimum
    found = issues_of(w.run(), "text.unreadable")
    assert [i.severity for i in found] == [Severity.NOTICE, Severity.WARNING, Severity.ERROR] and found[1].metrics["important"] is True
    assert "numbers, dates and warnings need" in found[1].description


def test_a_text_colour_that_cannot_be_read_against_its_backdrop(tmp_path, monkeypatch):
    w = World(tmp_path)
    monkeypatch.setitem(tc.TEXT_COLORS, "LABEL", "#303030")
    w.text(1.0, 3.0, "dark on dark")
    w.text(5.0, 3.0, "dark on a box", background="box", pos=(0.5, 0.6))
    found = [i for i in issues_of(w.run(), "text.unreadable") if "contrast" in i.title.lower()]
    assert len(found) == 2 and found[0].confidence == 60.0 and found[1].confidence == 85.0 and found[1].metrics["backdrop"] == "box"
    monkeypatch.setitem(tc.TEXT_COLORS, "LABEL", "#ffffff")
    assert w.run().issues == []


def test_a_figure_needs_time_to_be_read_in_proportion_to_its_length(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "$1,250", dur=0.5)  # 6 characters: 0.9 s needed
    w.number(4.0, "42%", dur=0.7)  # 3 characters: 0.7 s needed: just enough
    w.text(7.0, 0.3, "Hi")
    found = issues_of(w.run(), "text.too_brief")
    assert [i.severity for i in found] == [Severity.WARNING, Severity.NOTICE] and found[0].metrics["seconds_needed"] == 0.88 and found[0].timeline_item_id
    assert "A figure or date" in found[0].why_it_matters


# ====================================================================== overlap, collision, duplicates
def test_text_on_top_of_text_overlaps_but_text_in_another_place_or_time_does_not(tmp_path):
    w = World(tmp_path)
    first = w.text(1.0, 3.0, "Silver prices", pos=(0.5, 0.42))
    second = w.text(2.0, 3.0, "Gold prices", pos=(0.5, 0.43))
    (i,) = issues_of(w.run(), "text.overlap")
    assert i.severity is Severity.WARNING and i.timeline_item_id == first.id and i.metrics["other_clip"] == second.id and i.metrics["overlap_seconds"] == 2.0
    w2 = World(tmp_path / "apart")
    w2.text(1.0, 3.0, "Silver prices", pos=(0.5, 0.2))
    w2.text(2.0, 3.0, "Gold prices", pos=(0.5, 0.7))
    w2.text(5.5, 3.0, "Later", pos=(0.5, 0.2))
    assert issues_of(w2.run(), "text.overlap") == []


def test_text_over_an_evidence_highlight_is_a_collision(tmp_path):
    w = World(tmp_path)
    w.box(1.0, 3.0, region=(0.3, 0.3, 0.4, 0.2))
    w.text(1.5, 2.0, "Silver prices", pos=(0.5, 0.4))
    (i,) = issues_of(w.run(), "text.collision")
    assert i.severity is Severity.WARNING and "evidence highlight" in i.title and i.confidence == 85.0
    w2 = World(tmp_path / "clear")
    w2.box(1.0, 3.0, region=(0.3, 0.3, 0.4, 0.2))
    w2.text(1.5, 2.0, "Silver prices", pos=(0.5, 0.8))
    w2.text(4.5, 2.0, "Later", pos=(0.5, 0.4))
    assert w2.run().issues == []


def test_a_text_that_collides_with_a_caption_is_the_caption_checkers_finding(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 3.0, "Warning", pos=(0.5, 0.88), size=60)
    seg = {"caption_id": "c", "scene_id": "", "start": 1.0, "end": 3.5, "text": "Silver prices rose", "lines": ["Silver prices rose"],
           "words": [{"word_id": "w0", "text": "Silver", "start": 1.0, "end": 1.4}], "emphasis": [], "style_id": "professional", "style_overrides": {}, "position": "bottom",
           "position_xy": [], "highlight_mode": "HIGHLIGHT"}
    add_clip(w.p, "track_v6", None, 1.0, 2.5, scene=w.s1, kind="caption", created_by="AI", text=seg)
    assert "text.overlap" not in codes(w.run()) and "text.collision" not in codes(w.run())


def test_the_same_text_twice_in_a_row_is_a_duplicate_with_a_confirmed_delete_of_the_second(tmp_path):
    w = World(tmp_path)
    first = w.text(1.0, 2.0, "Key point", pos=(0.5, 0.3))
    second = w.text(3.5, 2.0, "key  POINT!", pos=(0.5, 0.3))
    (i,) = issues_of(w.run(), "text.duplicate")
    assert i.severity is Severity.WARNING and i.timeline_item_id == second.id and i.metrics["first_clip"] == first.id
    assert i.fix.kind == "clip.delete" and i.fix.params == {"clip_id": second.id} and i.fix.needs_confirmation and i.auto_fix_available and not i.auto_fix_safe
    w2 = World(tmp_path / "far")
    w2.text(1.0, 2.0, "Key point")
    w2.text(14.0, 2.0, "Key point")  # another scene, far later: a deliberate callback
    assert issues_of(w2.run(), "text.duplicate") == []
    w3 = World(tmp_path / "user")
    w3.text(1.0, 2.0, "Key point")
    w3.text(3.5, 2.0, "Key point", by="USER")
    (j,) = issues_of(w3.run(), "text.duplicate")
    assert j.locked and not j.auto_fix_available and j.fix_blocked_reason


# ====================================================================== timing, hierarchy, amount
def test_text_outside_its_scene_or_on_screen_much_longer_than_needed(tmp_path):
    w = World(tmp_path)
    w.text(7.0, 5.5, "Silver prices")  # midpoint 9.75: scene 1; runs 2.5 s into scene 2
    w.text(12.0, 2.5, "Hello", pos=(0.5, 0.2), scene=w.s1)  # shown in scene 2 but assigned to scene 1
    w.text(15.0, 4.0, "Hi", pos=(0.5, 0.6))  # 4 s for 2 characters: fine
    w.text(0.5, 9.0, "Silver", pos=(0.5, 0.7))  # 9 s for one word
    found = issues_of(w.run(), "text.timing")
    assert sorted((i.severity.value, i.title) for i in found) == [("NOTICE", "Text stays on screen longer than needed"), ("WARNING", "Text is on screen outside its scene"),
                                                                    ("WARNING", "Text is on screen outside its scene")]
    assert any("2.5 s after scene 1 ends" in i.description for i in found) and any("assigned to scene 1" in i.description for i in found)


def test_a_persistent_overlay_is_intentional_and_never_judged_for_length(tmp_path):
    w = World(tmp_path)
    w.text(0.0, 20.0, "Logo", pos=(0.9, 0.08), size=40, scene=w.s1)  # the whole video
    assert w.run().issues == []


def test_a_minor_text_larger_than_a_headline_is_a_hierarchy_notice(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 3.0, "Big idea", style="HEADLINE", pos=(0.5, 0.15), size=48)
    label = w.text(5.0, 3.0, "a small note", style="LABEL", pos=(0.5, 0.7), size=80)
    (i,) = issues_of(w.run(), "text.hierarchy")
    assert i.severity is Severity.NOTICE and i.timeline_item_id == label.id and i.metrics["size"] == 80
    w2 = World(tmp_path / "ok")
    w2.text(1.0, 3.0, "Big idea", style="HEADLINE", pos=(0.5, 0.15), size=64)
    w2.number(5.0, "42%", size=88)  # a number card may be bigger than the headline: it ranks with it
    assert w2.run().issues == []


def test_too_much_text_in_one_scene_is_scene_scoped(tmp_path):
    w = World(tmp_path)
    for k in range(3):
        w.text(0.5 + 3.0 * k, 1.5, f"Point {k + 1}", pos=(0.5, 0.2 + 0.2 * k))
    assert issues_of(w.run(), "text.excessive") == []  # three in ten seconds is fine
    w.text(9.0, 0.9, "Point 4", pos=(0.5, 0.8))
    (i,) = issues_of(w.run(), "text.excessive")
    assert i.severity is Severity.NOTICE and i.scene_id == w.s1.id and i.timeline_item_id is None and i.metrics == {"texts": 4, "allowed": 3}
    for k in range(5, 8):
        w.text(2.0 + 0.1 * k, 0.5, f"More {k}", pos=(0.5, 0.1 + 0.1 * k))
    assert issues_of(w.run(), "text.excessive")[0].severity is Severity.WARNING


# ====================================================================== fact review: never a correction, never a verdict
def test_a_value_on_screen_that_is_not_in_the_narration_is_flagged_for_review_only(tmp_path):
    w = World(tmp_path)
    clip = w.number(1.0, "43%")
    out = w.run()
    (i,) = issues_of(out, "text.value_mismatch")
    assert codes(out) == ["text.value_mismatch"] and i.category is QCCategory.FACT_REVIEW and i.severity is Severity.WARNING and i.confidence <= 80.0
    assert i.title == "Possible inconsistency: on-screen value differs from the script"
    assert i.description.startswith("The on-screen value 43% does not appear in the narration of this scene. Review recommended.")
    assert i.timeline_item_id == clip.id and i.scene_id == w.s1.id and i.group_hint == "" and i.score_group == "visual_accuracy"
    # a review, never a correction: the only fix opens the text, and it is neither safe nor a command
    assert i.fix.kind == "text.change" and i.fix.route is FixRoute.NAVIGATE and i.fix.params == {"clip_id": clip.id} and not i.fix.safe and not i.auto_fix_safe
    # the wording makes no claim about what is true
    words = f"{i.title} {i.description} {i.why_it_matters} {i.suggested_fix}".lower()
    assert not any(bad in words for bad in (" wrong", "incorrect", "is false", "is true", "error in"))


def test_a_value_that_matches_the_narration_the_script_or_the_scene_numbers_is_fine(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "42%")  # narration: "forty two percent"
    w.number(4.0, "$1,250")
    w.text(7.0, 2.5, "April 15, 2027", style="DATE")
    w.s2.script_text = "Imports rose 7.5% in March."
    w.number(12.0, "7.5%")  # only the script says it
    w.s2.numbers = [NumericMention("1.2 million", NumberKind.QUANTITY, 1_200_000.0, "", False, [])]  # only the scene's number list says it
    w.number(15.0, "1.2M")
    assert issues_of(w.run(), "text.value_mismatch") == []


def test_a_figure_when_nothing_numeric_is_spoken_is_still_a_review_but_a_less_certain_one(tmp_path):
    w = World(tmp_path)
    w.number(12.0, "99%")  # scene 2 mentions no figure at all
    (i,) = issues_of(w.run(), "text.value_mismatch")
    assert i.confidence == 65.0 and "Other values are spoken" not in i.description
    w2 = World(tmp_path / "differs")
    w2.number(1.0, "99%")
    (j,) = issues_of(w2.run(), "text.value_mismatch")
    assert j.confidence == 72.0 and "Other values are spoken in this scene" in j.description


def test_dates_are_compared_by_month_and_day_and_a_section_headline_is_not_compared(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 2.5, "April 16, 2027", style="DATE")
    w.text(4.0, 2.5, "TOP 10 MOVES OF 2031", style="HEADLINE", pos=(0.5, 0.14), size=64, derived=True, source_ref="scene_topic")
    found = issues_of(w.run(), "text.value_mismatch")
    assert len(found) == 1 and found[0].metrics["value"] == "April 16 2027"


def test_a_counter_that_ends_on_a_different_figure_is_flagged_for_review(tmp_path):
    w = World(tmp_path)
    ok = {"from": 0.0, "to": 1250.0, "decimals": 0, "prefix": "$", "suffix": "", "thousands": True}
    w.number(1.0, "$1,250", counter=ok)
    assert issues_of(w.run(), "text.value_mismatch") == []
    w2 = World(tmp_path / "bad")
    w2.number(1.0, "$1,250", counter={**ok, "to": 1520.0})
    (i,) = issues_of(w2.run(), "text.value_mismatch")
    assert i.category is QCCategory.FACT_REVIEW and i.metrics["kind"] == "counter" and i.metrics["counter_to"] == 1520.0 and i.fix.kind == "text.change"
    assert "ends on 1520.0" in i.description and "Review recommended" in i.description and i.confidence <= 80.0


def test_a_name_that_is_not_spoken_is_a_notice_and_names_in_capitals_are_not_judged(tmp_path):
    w = World(tmp_path)
    w.text(1.0, 2.5, "Maria Gonzalez", style="LOWER_THIRD", pos=(0.08, 0.8), align="left", size=48)  # not in the narration
    w.text(4.0, 2.5, "Jane Smith", style="LOWER_THIRD", pos=(0.08, 0.8), align="left", size=48)  # spoken
    w.text(7.0, 2.5, "MARIA GONZALEZ", style="LOWER_THIRD", pos=(0.08, 0.8), align="left", size=48)
    w.text(12.0, 2.5, "Breaking", style="LOWER_THIRD", pos=(0.08, 0.8), align="left", size=48)  # one capitalised word: could be any sentence
    found = issues_of(w.run(), "text.value_mismatch")
    assert len(found) == 1 and found[0].severity is Severity.NOTICE and found[0].confidence == 55.0 and found[0].metrics["kind"] == "name" and "Maria Gonzalez" in found[0].description


def test_a_locked_text_is_reviewed_without_any_fix(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "43%", by="USER")
    (i,) = issues_of(w.run(), "text.value_mismatch")
    assert i.locked and not i.auto_fix_available and "disabled" in i.fix_blocked_reason


def test_an_evidence_highlight_without_a_claim_to_support_is_a_notice(tmp_path):
    w = World(tmp_path)
    g = w.box(12.0, 3.0)  # scene 2: "Demand keeps growing across Asia": no claim, no figure
    out = w.run()
    (i,) = issues_of(out, "text.evidence_unsupported")
    assert codes(out) == ["text.evidence_unsupported"] and i.severity is Severity.NOTICE and i.category is QCCategory.FACT_REVIEW and i.timeline_item_id == g.id and i.scene_id == w.s2.id
    assert i.title == "Evidence treatment may not support the spoken claim" and "Review recommended" in i.description and i.confidence <= 80.0
    assert i.fix.kind == "graphics.change" and i.fix.route is FixRoute.NAVIGATE and not i.auto_fix_safe


def test_an_evidence_highlight_is_supported_by_a_claim_a_figure_or_an_evidence_intent(tmp_path):
    w = World(tmp_path)
    w.box(1.0, 3.0)  # scene 1 has a claim that needs evidence
    assert w.run().issues == []
    w.s1.claims, w.s1.numbers = [], []
    assert [i.code for i in w.run().issues] == ["text.evidence_unsupported"]
    w.p.visual_intents[w.s1.id] = VisualIntent(w.s1.id, VisualType.EVIDENCE, "silver", "", "", "", [], [], [], {}, 0.9, None, None)  # type: ignore[arg-type]
    assert w.run().issues == []
    w.s1.claims = [Claim("c", "an opinion", ClaimType.OPINION, "s", False)]
    w.p.visual_intents.pop(w.s1.id)
    assert [i.code for i in w.run().issues] == ["text.evidence_unsupported"]  # an opinion needs no evidence


# ====================================================================== scope, determinism
def test_every_issue_is_scene_scoped_and_codes_stay_in_the_catalogue(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "43%")
    w.text(1.0, 3.0, "Silver prices", pos=(0.97, 0.5))
    w.box(12.0, 3.0)
    w.text(12.0, 2.0, "Hello", pos=(0.5, 0.4), size=12)
    out = w.run()
    assert out.issues and all(i.scene_id for i in out.issues) and {i.code for i in out.issues} <= ALLOWED
    only2 = w.run(scene_filter=[w.s2.id])
    assert {i.scene_id for i in only2.issues} == {w.s2.id} and f"{w.s1.id}.texts_checked" not in only2.metrics


def test_only_the_changed_scene_is_reanalysed(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "43%")
    w.text(12.0, 2.0, "Hello", pos=(0.5, 0.4), size=12)
    chk = TextChecker()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(w.p))
    assert first.run.checkers["text"].state is CheckerState.DONE and sorted(i.code for i in first.run.issues) == ["text.unreadable", "text.value_mismatch"]
    w.text(15.0, 2.0, "Another", pos=(0.5, 0.6), size=12)  # a new problem in scene 2 only
    second = eng.run(qc_ctx(w.p), previous=PreviousState(first.run.issues, first.cache))
    st = second.run.checkers["text"]
    assert st.reused_scenes == 1 and st.analyzed_scenes == 1 and len(second.run.issues) == 3
    assert {i.fingerprint for i in first.run.issues} <= {i.fingerprint for i in second.run.issues}
    w.s1.numbers.append(NumericMention("43%", NumberKind.PERCENTAGE, 43.0, "", False, []))  # the script now contains the value: scene 1 is re-analysed and the review disappears
    third = eng.run(qc_ctx(w.p), previous=PreviousState(second.run.issues, second.cache))
    assert "text.value_mismatch" not in {i.code for i in third.run.issues} and third.run.checkers["text"].analyzed_scenes == 1


def test_the_checker_never_mutates_the_project_and_skips_hidden_tracks(tmp_path):
    w = World(tmp_path)
    w.number(1.0, "43%")
    w.text(1.0, 3.0, "Silver prices", pos=(0.97, 0.5))
    w.box(12.0, 3.0)
    before = w.p.to_document()
    out = w.run()
    assert len(out.issues) >= 3 and w.p.to_document() == before
    w.p.timeline.get_track("track_v5").hidden = True
    w.p.timeline.get_track("track_v4").hidden = True
    assert w.run().issues == []


def test_fix_constructors_match_what_the_checker_attaches(tmp_path):
    w = World(tmp_path)
    clip = w.number(1.0, "43%")
    (i,) = issues_of(w.run(), "text.value_mismatch")
    expected = fix_catalog.navigate("text.change", "Review the on-screen text against the script", clip_id=clip.id)
    assert (i.fix.kind, i.fix.params, i.fix.route) == (expected.kind, expected.params, expected.route)


def test_a_scene_without_narration_or_script_has_nothing_to_compare_the_text_with(tmp_path):
    w = World(tmp_path)
    w.s2.narration = ""
    w.number(12.0, "99%")
    assert w.run().issues == []
    w.s2.script_text = "Imports rose 7.5% in March."  # a script restores the comparison
    assert [i.code for i in w.run().issues] == ["text.value_mismatch"]


def test_both_checkers_run_inside_the_engine_and_feed_the_scores(tmp_path):
    from app.qc.caption_checker import CaptionChecker  # noqa: PLC0415

    w = World(tmp_path)
    w.number(1.0, "43%")  # a fact to review
    w.text(12.0, 2.0, "Hello", pos=(0.97, 0.4))  # clipped
    seg = {"caption_id": "c", "scene_id": "", "start": 1.0, "end": 3.0, "text": "Short one", "lines": ["Short one"], "words": [{"word_id": "w0", "text": "Short", "start": 1.0, "end": 1.4}],
           "emphasis": [], "style_id": "professional", "style_overrides": {}, "position": "custom", "position_xy": [0.5, 0.97], "highlight_mode": "HIGHLIGHT"}
    add_clip(w.p, "track_v6", None, 1.0, 2.0, scene=w.s1, kind="caption", created_by="AI", text=seg)
    result = QCEngine([CaptionChecker(), TextChecker()]).run(qc_ctx(w.p))
    assert result.run.failed_checkers() == [] and result.run.state == "COMPLETED"
    by_code = {i.code: i for i in result.run.issues}
    assert set(by_code) == {"text.value_mismatch", "text.clipped", "caption.safe_margin"}
    assert by_code["text.value_mismatch"].score_group == "visual_accuracy" and by_code["caption.safe_margin"].score_group == "captions" and by_code["text.clipped"].score_group == "captions"
    assert result.run.scores.groups["captions"] < 100.0 and result.run.scores.groups["visual_accuracy"] < 100.0


# ------------------------------------------------------------------ the real pipeline
@needs_ffmpeg
def test_text_and_evidence_graphics_made_by_the_real_pipeline_raise_nothing(pres_ws):
    pres_ws.presentation.generate(["GRAPHICS"])
    assert pres_ws.jobs.wait_idle(120)
    out = run_checker(TextChecker(), qc_ctx(pres_ws.project))
    assert out.metrics["texts_checked"] >= 10 and out.metrics["graphics_checked"] >= 1
    assert out.issues == []  # figures come from the narration, number callouts sit inside their evidence box without hiding it
