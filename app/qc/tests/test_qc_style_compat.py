"""Reference-style compatibility: a soft preference that never outranks the content."""

from __future__ import annotations

from app.qc.pacing_checker import PacingChecker
from app.qc.severity import Severity
from app.qc.style_compat import applied_dimensions, style_deviation, style_issue, target_score
from app.qc.tests.qc_helpers import find, qc_ctx, run_checker
from app.qc.tests.test_qc_pacing import TEXT, build as _build
from app.reference.style_model import PACING_POINTS, ReferenceStyleProfile, StyleScores, scale


NORMAL = TEXT + " Analysts said the move reflects strong industrial demand and tight physical supply across Asia."  # about 2.5 words a second over ten seconds


def build(tmp_path, cuts, **kw):
    kw.setdefault("texts", [NORMAL] * kw.get("scenes", 6))
    return _build(tmp_path, cuts, **kw)


def apply_style(p, pacing=90.0, *, unavailable=(), mode="FULL", dims=None, adjustments=None):
    p.reference_settings.enabled = True
    p.reference_settings.application_mode = mode
    if dims is not None:
        p.reference_settings.custom_dimensions = list(dims)
    p.reference_settings.adjustments = dict(adjustments or {})
    p.reference_style_profile = ReferenceStyleProfile(
        scores=StyleScores(pacing=pacing, caption_density=60.0, motion_intensity=50.0), unavailable=list(unavailable),
        confidence={"shot_detection": 0.9, "caption_detection": 0.9, "motion_detection": 0.9, "overall": 0.9})
    return p


def cuts_for(cpm):
    n = int(cpm * 60 / 60)  # a 60 s video
    return [(k + 1) * 60.0 / (n + 1) for k in range(n)]


class _Probe:
    """Any checker works as the issue factory; the pacing checker is used here."""

    checker = PacingChecker()


def test_no_reference_applied_means_no_style_findings(tmp_path):
    p, _ = build(tmp_path, cuts_for(6))
    ctx = qc_ctx(p)
    assert applied_dimensions(ctx) == [] and target_score(ctx, "pacing") is None and style_deviation(ctx, "pacing", 20.0) is None
    assert style_issue(_Probe.checker, ctx, "pacing", 20.0) is None
    assert not [i for i in run_checker(PacingChecker(), ctx).issues if i.category.value == "STYLE"]


def test_deviation_beyond_the_tolerance_is_a_notice(tmp_path):
    p, _ = build(tmp_path, cuts_for(6))  # 6 cuts a minute: pacing score 25
    apply_style(p, pacing=90.0)
    out = run_checker(PacingChecker(), qc_ctx(p))
    i = find(out, "style.pacing_deviation")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and i[0].category.value == "STYLE"
    assert "25/100" in i[0].description and "90/100" in i[0].description and "points lower" in i[0].description
    assert out.metrics["style"]["pacing_score"] == round(scale(6, PACING_POINTS), 1)
    assert i[0].fix.kind == "settings.open"  # a way to look at the style, never a force-fit


def test_deviation_within_the_tolerance_is_silent(tmp_path):
    p, _ = build(tmp_path, cuts_for(12))  # score 50
    apply_style(p, pacing=60.0)
    assert find(run_checker(PacingChecker(), qc_ctx(p)), "style.pacing_deviation") == []


def test_custom_target_from_the_users_sliders_wins_over_the_reference(tmp_path):
    p, _ = build(tmp_path, cuts_for(6))
    apply_style(p, pacing=90.0, adjustments={"pacing": 25.0})  # the user dialled the pace down to what the edit already has
    ctx = qc_ctx(p)
    assert target_score(ctx, "pacing") == 25.0
    assert find(run_checker(PacingChecker(), ctx), "style.pacing_deviation") == []


def test_dimension_not_applied_or_not_measurable_is_ignored(tmp_path):
    p, _ = build(tmp_path, cuts_for(6))
    apply_style(p, pacing=90.0, unavailable=["pacing"])
    assert "pacing" not in applied_dimensions(qc_ctx(p)) and "caption_density" in applied_dimensions(qc_ctx(p))
    assert find(run_checker(PacingChecker(), qc_ctx(p)), "style.pacing_deviation") == []
    apply_style(p, pacing=90.0, mode="CUSTOM", dims=["caption_density"])  # the style was applied to captions only
    assert "pacing" not in applied_dimensions(qc_ctx(p))
    assert find(run_checker(PacingChecker(), qc_ctx(p)), "style.pacing_deviation") == []


def test_low_confidence_dimension_is_not_held_against_the_project(tmp_path):
    p, _ = build(tmp_path, cuts_for(6))
    apply_style(p, pacing=90.0)
    p.reference_style_profile.confidence["shot_detection"] = 0.1
    assert "pacing" not in applied_dimensions(qc_ctx(p))


def test_style_checks_can_be_switched_off(tmp_path):
    p, _ = build(tmp_path, cuts_for(6))
    apply_style(p, pacing=90.0)
    p.qc_settings.style.check = False
    assert applied_dimensions(qc_ctx(p)) == []


def test_deviation_the_content_asks_for_is_informational(tmp_path):
    slow = "Silver rose."  # a couple of words in ten seconds: very slow narration
    p, _ = _build(tmp_path, cuts_for(6), texts=[slow] * 6)
    apply_style(p, pacing=90.0)
    i = find(run_checker(PacingChecker(), qc_ctx(p)), "style.pacing_deviation")
    assert len(i) == 1 and i[0].severity is Severity.INFO and "because the narration is slow" in i[0].description


def test_style_that_hurts_readability_is_flagged_as_a_conflict_not_forced(tmp_path):
    cuts = [9.4, 9.8, 10.2, 10.6, 11.0, 30.0, 52.0]
    p, _ = build(tmp_path, cuts, scenes=6)
    apply_style(p, pacing=30.0)  # the edit already follows the style (about 7 cuts a minute); the style itself causes the unreadable burst
    out = run_checker(PacingChecker(), qc_ctx(p))
    assert find(out, "cut.micro_cuts")
    i = find(out, "style.conflict")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and "conflicts with readability" in i[0].title
    assert "not being forced" in i[0].why_it_matters and i[0].fix.kind == "settings.open"
    assert not any("increase" in j.suggested_fix.lower() or "speed up" in j.suggested_fix.lower() for j in out.issues if j.category.value == "STYLE")


def test_helper_contract(tmp_path):
    p, _ = build(tmp_path, cuts_for(6), texts=[TEXT] * 6)
    apply_style(p, pacing=90.0)
    ctx = qc_ctx(p)
    dev = style_deviation(ctx, "pacing", 25.0, explained_by="x")
    assert dev and dev.delta == -65.0 and dev.direction == "lower" and dev.explained_by == "x"
    assert style_deviation(ctx, "visual_density", 10.0) is None  # the reference scores 0 there: 10 is inside the tolerance
    for dim, code in (("caption_density", "style.caption_deviation"), ("motion_intensity", "style.motion_deviation")):
        got = style_issue(_Probe.checker, ctx, dim, 0.0)
        assert got is not None and got.code == code
