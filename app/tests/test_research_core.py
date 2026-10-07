from __future__ import annotations

import json
from copy import deepcopy

import pytest

from app.analysis.models import VisualType
from app.core.serialization import from_plain, to_plain
from app.media.asset import SourceType as S
from app.research.dedupe import deduplicate, dhash_gray, fingerprint_file, hamming, normalize_url
from app.research.evaluation import MetadataEvaluator, VisualEvaluationService, category_for, recommend_duration
from app.research.models import (
    Acquisition,
    Candidate,
    CandidateScore,
    CandidateStatus,
    Confidence,
    EvidenceKind,
    EvidenceLevel,
    QueryType,
    ResearchBrief,
    ResearchSettings,
    ScoreCategory,
    VisualAssignment,
)
from app.research.queries import STRATEGIES, QueryGenerator, similar
from app.research.ranking import Ranker, RankingHistory, UsedVisual, title_terms
from app.tests.conftest import needs_ffmpeg, pipeline_ws
from app.tests.helpers import cand, make_image, solar_brief
from app.visual.preferences import SourceKind, VisualPreferences

STRONG = dict(title="Solar panel manufacturing: silver paste screen printing on photovoltaic cells",
              description="Industrial production line in a solar installations factory applying silver conductive paste to silicon wafers",
              tags=["solar", "silver", "manufacturing", "photovoltaic"])


# ================================================================== research brief (from real Phase 2 scenes)
@needs_ffmpeg
def test_brief_carries_scene_context_entities_claims_and_requirements(research_ws):
    ws = research_ws
    scenes = ws.project.scenes
    b = ws.research.brief(scenes[0].id)  # "Silver demand ... Solar manufacturers are consuming ..."
    assert b.scene_id == scenes[0].id and b.narration == scenes[0].narration
    assert b.visual_type == "PROCESS" and b.primary_subject and b.topic == "Silver demand"
    assert "Silver" in b.entities and b.entity_types["Silver"] == "FINANCIAL_INSTRUMENT"
    assert b.claims and b.claim_types and b.evidence_level is EvidenceLevel.POSSIBLE and not b.evidence_needed
    assert b.previous is None and b.next.scene_id == scenes[1].id and b.video_topic == "silver"
    assert b.scene_duration == pytest.approx(scenes[0].duration) and b.project_width == 1920
    assert b.preferred_sources[0] == "STOCK_VIDEO" and "SCREENSHOT" in b.preferred_sources  # evidence possible -> screenshot is an option
    assert b.avoid and any("generic" in a for a in b.avoid)  # explicit "avoid" list, not just keywords
    d = b.to_dict()
    assert d["evidence_needed"] is False and ResearchBrief.from_dict(d) == b


@needs_ffmpeg
def test_evidence_scene_requires_evidence_and_prefers_screenshots(research_ws):
    ws = research_ws
    sc = next(s for s in ws.project.scenes if "IRS sent a notice" in s.narration)
    b = ws.research.brief(sc.id)
    assert b.visual_type == "EVIDENCE" and b.evidence_needed
    assert b.preferred_sources[:2] == ["SCREENSHOT", "WEB_IMAGE"]
    assert "IRS" in b.entities and b.entity_types["IRS"] == "GOVERNMENT_AGENCY"
    assert b.previous.scene_id != sc.id and b.next is not None


@needs_ffmpeg
def test_numbers_and_dates_are_separated_in_the_brief(research_ws):
    ws = research_ws
    sc = next(s for s in ws.project.scenes if "3.5%" in s.narration)
    b = ws.research.brief(sc.id)
    assert any("3.5" in n for n in b.numbers) and any("2027" in d for d in b.dates)
    assert b.visual_type in ("EVIDENCE", "DATA")


@needs_ffmpeg
def test_unanalysed_scene_cannot_be_briefed(ws, tmp_path):
    from app.core.exceptions import AnalysisError

    ws.new_project("X", tmp_path / "p")
    with pytest.raises(AnalysisError):
        ws.research.brief("scene_001")


@needs_ffmpeg
def test_context_inheritance_resolves_what_paperwork_refers_to(ws, tmp_path):
    """The spec's example: 'That means some silver sellers may receive additional paperwork.'"""
    text = ("The IRS introduced a new reporting requirement for precious metals. "
            "That means some silver sellers may receive additional paperwork.")
    pipeline_ws(ws, tmp_path, text, threshold=0.05, min_scene=0.5)
    scenes = ws.project.scenes
    assert len(scenes) >= 2
    b = ws.research.brief(scenes[-1].id)
    assert b.anaphoric and "IRS" in b.context_terms
    assert "paperwork" in b.key_terms
    queries = QueryGenerator().generate(b)
    first = queries[0].text.lower()
    assert "irs" in first and "paperwork" in first and first != "paperwork"  # never the bare word
    assert all("irs" in q.text.lower() or q.type in (QueryType.ALTERNATIVE, QueryType.PROCESS, QueryType.ENTITY) for q in queries[:2])


# ================================================================== queries
def test_multiple_query_types_with_purposes_priorities_and_sources():
    qs = QueryGenerator().generate(solar_brief())
    types = {q.type for q in qs}
    assert {QueryType.LITERAL, QueryType.CONTEXT, QueryType.PROCESS, QueryType.ALTERNATIVE} <= types and len(qs) >= 4
    assert all(q.purpose and q.scene_id == "scene_014" and 1 <= q.priority <= 5 and q.source_preferences for q in qs)
    by = {q.type: q for q in qs}
    assert "solar installations" in by[QueryType.LITERAL].text and "silver" in by[QueryType.LITERAL].text
    assert "industry" in by[QueryType.CONTEXT].text
    assert "manufacturing" in by[QueryType.PROCESS].text
    assert S.SCREENSHOT.value not in by[QueryType.LITERAL].source_preferences and S.STOCK_VIDEO.value in by[QueryType.PROCESS].source_preferences


def test_queries_are_diverse_not_variations_of_one_string():
    qs = QueryGenerator().generate(solar_brief())
    texts = [q.text for q in qs]
    assert len(set(texts)) == len(texts)
    assert not any(similar(a, b, 0.85) for i, a in enumerate(texts) for b in texts[i + 1:])
    assert all(len(t.split()) <= 8 for t in texts)


def test_evidence_scene_gets_evidence_and_document_queries_first():
    b = solar_brief(topic="IRS notice", primary_subject="IRS", secondary_subject="silver sellers", visual_type="EVIDENCE",
                    evidence_level=EvidenceLevel.REQUIRED, entities=["IRS"], entity_types={"IRS": "GOVERNMENT_AGENCY"},
                    claims=["The IRS sent a notice to silver sellers."], claim_types=["FACT"], action="", context="tax reporting")
    qs = QueryGenerator().generate(b)
    assert {QueryType.EVIDENCE, QueryType.DOCUMENT, QueryType.ENTITY} <= {q.type for q in qs}
    top = [q for q in qs if q.priority == 1]
    assert {q.type for q in top} & {QueryType.EVIDENCE, QueryType.DOCUMENT}
    doc = next(q for q in qs if q.type is QueryType.DOCUMENT)
    assert "IRS" in doc.text and doc.source_preferences[0] == S.SCREENSHOT.value


def test_data_scene_gets_a_data_query_with_the_number():
    b = solar_brief(visual_type="DATA", numbers=["$100"], topic="Silver price", primary_subject="silver price", secondary_subject="")
    q = next(q for q in QueryGenerator().generate(b) if q.type is QueryType.DATA)
    assert "chart" in q.text and "$100" in q.text


def test_max_queries_is_respected_and_priority_order_kept():
    qs = QueryGenerator(ResearchSettings(max_queries=3)).generate(solar_brief())
    assert len(qs) == 3 and [q.priority for q in qs] == sorted(q.priority for q in qs)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_again_strategies_produce_new_queries(strategy):
    gen = QueryGenerator()
    first = gen.generate(solar_brief())
    again = gen.generate(solar_brief(), strategy, 1, previous=first)
    assert again, strategy
    if strategy != "sources":  # that one reuses wording on purpose but reorders the sources
        assert not any(similar(a.text, f.text, 0.85) for a in again for f in first), (strategy, [q.text for q in again])
    else:
        assert again[0].source_preferences != next(f for f in first if f.type is again[0].type).source_preferences
    assert all(q.generation == 1 and q.strategy == strategy for q in again)


def test_search_again_never_returns_empty_even_when_everything_was_tried():
    gen = QueryGenerator()
    first = gen.generate(solar_brief())
    seen = list(first)
    for _ in range(4):
        new = gen.generate(solar_brief(), "broader", 1, previous=seen)
        assert new and not any(similar(n.text, s.text, 0.95) for n in new for s in seen)
        seen += new


# ================================================================== candidate model
def test_candidate_serialisation_roundtrip_with_all_nested_parts():
    c = cand("Solar line", "desc", S.YOUTUBE, tags=["a", "b"], license=None) if False else cand("Solar line", "desc", S.YOUTUBE, tags=["a", "b"])
    c.segment.start, c.segment.end, c.segment.basis = 12.5, 17.5, "CHAPTER"
    c.license.name, c.license.status = "Standard YouTube License", "PROVIDER_STATED"
    c.acquisition, c.evidence_kind, c.metadata = Acquisition.REFERENCE_ONLY, EvidenceKind.EVIDENCE, {"channel": "x", "n": [1, 2]}
    back = Candidate.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back == c and back.source_type is S.YOUTUBE and back.segment.length == 5.0 and back.license.verified is False


def test_assignment_and_score_roundtrip():
    sc = MetadataEvaluator().evaluate(solar_brief(), cand(**STRONG), 85)
    back = from_plain(CandidateScore, json.loads(json.dumps(to_plain(sc))))
    assert back == sc
    a = VisualAssignment("scene_014", "candidate_892", "media_293", "USER", 96.0, True)
    assert VisualAssignment.from_dict(json.loads(json.dumps(a.to_dict()))) == a


def test_url_normalisation_and_exact_deduplication_merges_query_ids():
    assert normalize_url("HTTPS://www.Example.com/a/b/?utm_source=x&id=3#frag") == normalize_url("https://example.com/a/b?id=3")
    a = cand("Solar", source_reference="https://www.x.com/p/1?utm_campaign=z", media_url="https://x.com/1.jpg", query_ids=["q1"], cid="c1")
    b = cand("Solar again", source_reference="https://x.com/p/1", media_url="https://x.com/1.jpg", query_ids=["q2"], cid="c2")
    c = cand("Other", source_reference="https://x.com/p/2", media_url="https://x.com/2.jpg", query_ids=["q3"], cid="c3")
    kept, removed = deduplicate([a, b, c])
    assert removed == 1 and [k.candidate_id for k in kept] == ["c1", "c3"]
    assert kept[0].query_ids == ["q1", "q2"] and kept[0].metadata["duplicates"][0]["reason"] == "same URL/asset"


def test_same_provider_id_from_two_queries_is_one_candidate():
    a = cand("Solar", provider_id="p9", source_reference="https://x/1", cid="c1", query_ids=["q1"])
    b = cand("Solar (renamed)", provider_id="p9", source_reference="https://x/other", cid="c2", query_ids=["q2"])
    kept, removed = deduplicate([a, b])
    assert removed == 1 and kept[0].query_ids == ["q1", "q2"]


def test_dhash_math():
    flat = bytes([100] * 72)
    gradient = bytes([(200 - col * 20) for _ in range(8) for col in range(9)])
    assert dhash_gray(flat) == 0 and dhash_gray(gradient) == (1 << 64) - 1 and dhash_gray(b"short") is None
    assert hamming("0" * 16, "f" * 16) == 64 and hamming("00ff" + "0" * 12, "00fe" + "0" * 12) == 1


@needs_ffmpeg
def test_visually_near_identical_images_are_deduplicated_but_different_ones_are_not(tmp_path):
    import subprocess

    a = make_image(tmp_path / "a.png", "testsrc", "640x360")
    near = tmp_path / "near.jpg"  # same picture, re-encoded smaller and lossy: what a re-upload looks like
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(a), "-vf", "scale=400:225", "-q:v", "6", str(near)], check=True)
    other = make_image(tmp_path / "b.png", "mandel", "640x360")
    fps = {n: fingerprint_file(p) for n, p in (("a", a), ("near", near), ("other", other))}
    assert all(fps.values())
    assert hamming(fps["a"], fps["near"]) <= 6 < hamming(fps["a"], fps["other"])
    cs = [cand("Sun A", cid="c1", source_type=S.STOCK_IMAGE, fingerprint=fps["a"], provider_id="1", source_reference="https://a/1"),
          cand("Sun B", cid="c2", source_type=S.WEB_IMAGE, fingerprint=fps["near"], provider_id="2", source_reference="https://b/2"),
          cand("Fractal", cid="c3", source_type=S.STOCK_IMAGE, fingerprint=fps["other"], provider_id="3", source_reference="https://a/3")]
    kept, removed = deduplicate(cs)
    assert removed == 1 and [k.candidate_id for k in kept] == ["c1", "c3"]
    assert kept[0].metadata["duplicates"][0]["reason"] == "visually near-identical"


# ================================================================== scoring
EV = MetadataEvaluator()


def score(brief=None, **kw):
    return EV.evaluate(brief or solar_brief(), cand(**kw), 85)


def test_strong_candidate_scores_high_with_explainable_components():
    s = score(**STRONG)
    assert s.overall >= 85 and s.category in (ScoreCategory.GOOD, ScoreCategory.EXCELLENT)
    c = s.components
    assert c.semantic >= 80 and c.subject >= 85 and c.timing == 100 and c.quality == 100
    assert all(0 <= v <= 100 for v in vars(c).values())
    assert any(f.startswith("✓") and "solar installations" in f for f in s.factors)
    assert any("No obvious conflict" in f for f in s.factors) and s.reason and "Matches the main subject" in s.reason
    assert s.confidence in (Confidence.MEDIUM, Confidence.HIGH) and s.basis == "METADATA"


def test_weak_candidate_is_rejected():
    s = score(title="Sunset over a beach", description="palm trees and waves")
    assert s.overall < 40 and s.category is ScoreCategory.REJECT and s.confidence is Confidence.LOW
    assert any(f.startswith("✗") for f in s.factors)


def test_wrong_subject_keyword_only_match_is_capped_not_rewarded():
    """Searching just 'silver' finds coins. The brief says solar installations, so coins must score badly."""
    s = score(title="Silver coins pile", description="close up of shiny silver bullion coins", tags=["silver", "coin"])
    assert s.components.semantic <= 55 and s.overall <= 59 and s.confidence is Confidence.LOW
    assert any("generic visual to avoid" in f for f in s.factors) and any("Capped" in f for f in s.factors)


def test_silver_alone_without_the_subject_cannot_pass_even_without_trap_words():
    s = score(title="Silver spoon on a table", description="elegant silver cutlery")
    assert s.overall < 65 and s.components.semantic <= 55


def test_wrong_context_scores_below_right_context():
    ctx = solar_brief(topic="IRS reporting", primary_subject="IRS reporting", secondary_subject="silver sellers", visual_type="EVIDENCE",
                      action="", context="tax reporting rules", context_terms=["IRS", "reporting requirement"], entities=["IRS"],
                      entity_types={"IRS": "GOVERNMENT_AGENCY"}, evidence_level=EvidenceLevel.NONE, avoid_terms=[])
    right = EV.evaluate(ctx, cand("IRS reporting form for silver sellers", "tax form explaining the new reporting requirement for sellers"), 85)
    wrong_ctx = EV.evaluate(ctx, cand("Silver sellers at a flea market", "vendors selling silver jewellery at an outdoor market"), 85)
    generic = EV.evaluate(ctx, cand("Stack of paperwork on a desk", "office desk with documents"), 85)
    assert right.overall >= 75 and wrong_ctx.overall <= 64 and generic.overall <= 64
    assert right.overall > wrong_ctx.overall + 10 and right.overall > generic.overall + 10
    assert right.components.context > wrong_ctx.components.context
    assert any("Does not mention the main subject" in f for f in generic.factors)  # "paperwork" alone is not the IRS


def test_timing_mismatch_is_penalised_and_flagged_partial():
    b = solar_brief(scene_end=74.2 + 8.0)  # 8 s scene
    short = EV.evaluate(b, cand(**STRONG, duration=2.0), 85)
    ok = EV.evaluate(b, cand(**STRONG, duration=12.0), 85)
    assert short.components.timing < 50 and short.partial_coverage and ok.components.timing == 100 and not ok.partial_coverage
    assert short.overall < ok.overall


def test_recommended_duration_fits_the_narration_not_a_fixed_five_seconds():
    b = solar_brief()  # 74.2 -> 80.4 = 6.2 s
    assert b.scene_duration == pytest.approx(6.2)
    stock = cand(**STRONG, duration=20.0)
    assert recommend_duration(b, stock) == pytest.approx(5.8)  # "Scene 6.2 -> recommended 5.8"
    img = cand("Solar", source_type=S.STOCK_IMAGE, kind="IMAGE")
    assert recommend_duration(b, img) == pytest.approx(6.2)
    short = cand(**STRONG, duration=3.0)
    assert recommend_duration(b, short) == pytest.approx(3.0)
    yt = cand("Solar factory tour", source_type=S.YOUTUBE, duration=600.0)
    for scene_len, expected in ((4.0, 3.5), (6.2, 5.0), (8.0, 7.0), (14.0, 10.0)):
        assert recommend_duration(solar_brief(scene_end=74.2 + scene_len), yt) == expected  # chosen from {3.5, 5, 7, 10}


def test_category_boundaries_and_configurable_threshold():
    assert [category_for(x) for x in (95, 90, 89.9, 80, 79.9, 70, 69.9, 60, 59.9)] == [
        ScoreCategory.EXCELLENT, ScoreCategory.EXCELLENT, ScoreCategory.GOOD, ScoreCategory.GOOD, ScoreCategory.REVIEW,
        ScoreCategory.REVIEW, ScoreCategory.WEAK, ScoreCategory.WEAK, ScoreCategory.REJECT]
    c = cand(**STRONG)
    at_85 = EV.evaluate(solar_brief(), c, 85)
    at_99 = EV.evaluate(solar_brief(), c, 99)
    assert at_85.overall == at_99.overall and at_99.confidence is Confidence.LOW and at_85.confidence is not Confidence.LOW and at_99.min_accuracy == 99


def test_weights_are_configurable_and_normalised():
    c = cand(**STRONG, duration=1.0)  # poor timing, otherwise strong
    default = MetadataEvaluator().evaluate(solar_brief(), c, 85).overall
    timing_heavy = MetadataEvaluator({"timing": 0.9}).evaluate(solar_brief(), c, 85).overall
    assert timing_heavy < default
    assert sum(MetadataEvaluator({"timing": 0.9}).weights.values()) == pytest.approx(1.0)
    assert sum(MetadataEvaluator().weights.values()) == pytest.approx(1.0)
    assert MetadataEvaluator().weights["semantic"] == pytest.approx(0.35) and MetadataEvaluator().weights["subject"] == pytest.approx(0.20)


def test_decorative_visual_cannot_beat_84_when_evidence_is_required():
    b = solar_brief(evidence_level=EvidenceLevel.REQUIRED)
    deco = EV.evaluate(b, cand(**STRONG), 85)
    real = EV.evaluate(b, cand(**STRONG, evidence_kind=EvidenceKind.EVIDENCE, source_type=S.WEB_IMAGE, kind="IMAGE"), 85)
    assert deco.overall <= 84 and real.overall > deco.overall and deco.components.source <= 35
    assert any("decorative" in f.lower() for f in deco.factors)


def test_ai_generated_is_capped_and_never_evidence():
    ai = cand("AI concept: solar installations", "solar installations manufacturing silver", source_type=S.AI_GENERATED, kind="IMAGE",
              tags=["solar installations", "silver", "manufacturing"], status=CandidateStatus.PROPOSED)
    s = EV.evaluate(solar_brief(), ai, 85)
    assert s.basis == "PROMPT" and s.overall <= 82 and s.confidence is Confidence.LOW
    ev = EV.evaluate(solar_brief(evidence_level=EvidenceLevel.REQUIRED), ai, 85)
    assert ev.overall <= 70 and any("cannot serve as evidence" in f for f in ev.factors) and ev.components.source <= 20


def test_quality_and_crop_hints():
    low = EV.evaluate(solar_brief(), cand(**STRONG, width=320, height=240), 85)
    hd = EV.evaluate(solar_brief(), cand(**STRONG, width=1920, height=1080), 85)
    portrait = EV.evaluate(solar_brief(), cand(**STRONG, width=1080, height=1920), 85)
    assert low.components.quality < 50 and hd.components.quality == 100 and portrait.components.quality < 100
    assert "no crop" in hd.crop_hint and hd.orientation == "landscape" and portrait.orientation == "portrait" and "centre-crop" in portrait.crop_hint
    assert hd.focus == ""  # never invented: no vision model is used


def test_sparse_metadata_means_low_confidence():
    s = EV.evaluate(solar_brief(), cand("Solar", ""), 50)
    assert s.confidence is Confidence.LOW


def test_evaluation_service_scores_every_candidate():
    cs = [cand(**STRONG, cid="a"), cand("Sunset", cid="b")]
    out = VisualEvaluationService().evaluate_all(solar_brief(), cs, 85)
    assert set(out) == {"a", "b"} and out["a"].overall > out["b"].overall


# ================================================================== ranking
def pair(title, accuracy_title_kw, source=S.STOCK_VIDEO, cid=None, **kw):
    c = cand(title, source_type=source, cid=cid or title, **kw)
    return c


def fake_score(c: Candidate, overall: float) -> CandidateScore:
    from app.research.models import ScoreComponents

    return CandidateScore(c.candidate_id, c.scene_id, overall, ScoreComponents(overall, overall, overall, overall, overall, overall, overall),
                          category_for(overall), Confidence.MEDIUM, "t")


def rank(pairs, prefs=None, history=None, brief=None, settings=None):
    return Ranker(settings).rank(pairs, brief or solar_brief(), prefs or VisualPreferences(), history or RankingHistory())


def test_high_accuracy_wins_over_source_preference():
    prefs = VisualPreferences()
    yt, stock = prefs.setting(SourceKind.YOUTUBE), prefs.setting(SourceKind.STOCK_VIDEOS)
    yt.target_percent, yt.priority, stock.target_percent, stock.priority = 80, 5, 0, 1  # user strongly prefers YouTube
    a = cand("Preferred youtube clip", source_type=S.YOUTUBE, cid="yt")
    b = cand("Better stock clip", source_type=S.STOCK_VIDEO, cid="stock")
    ranked = rank([(a, fake_score(a, 80)), (b, fake_score(b, 93))], prefs)
    assert ranked[0].candidate_id == "stock" and ranked[0].role == "BEST"
    assert ranked[0].accuracy == 93 and ranked[0].rank_score - ranked[0].accuracy < 6.5


def test_preferences_break_near_ties_in_both_directions():
    a = cand("Youtube clip of solar line", source_type=S.YOUTUBE, cid="yt")
    b = cand("Stock clip of solar line", source_type=S.STOCK_VIDEO, cid="stock")
    pairs = [(a, fake_score(a, 89)), (b, fake_score(b, 90))]
    want_yt = VisualPreferences()
    want_yt.setting(SourceKind.YOUTUBE).target_percent, want_yt.setting(SourceKind.STOCK_VIDEOS).target_percent = 60, 5
    want_yt.setting(SourceKind.YOUTUBE).priority = 5
    assert rank(pairs, want_yt)[0].candidate_id == "yt"
    want_stock = VisualPreferences()
    want_stock.setting(SourceKind.YOUTUBE).target_percent, want_stock.setting(SourceKind.STOCK_VIDEOS).target_percent = 5, 60
    want_stock.setting(SourceKind.STOCK_VIDEOS).priority = 5
    assert rank(pairs, want_stock)[0].candidate_id == "stock"


def test_soft_balancing_an_over_target_source_is_still_chosen_when_it_is_the_best():
    """Screenshot target 10% is already exceeded, but the official document screenshot is the best evidence: use it."""
    prefs = VisualPreferences()
    hist = RankingHistory(source_counts={S.SCREENSHOT: 6, S.STOCK_VIDEO: 2}, scene_index=9)
    shot = cand("Official IRS notice screenshot", source_type=S.SCREENSHOT, kind="IMAGE", cid="shot", evidence_kind=EvidenceKind.EVIDENCE)
    stock = cand("Generic office paperwork", source_type=S.STOCK_VIDEO, cid="stock")
    ranked = rank([(shot, fake_score(shot, 95)), (stock, fake_score(stock, 78))], prefs, hist, solar_brief(evidence_level=EvidenceLevel.REQUIRED))
    assert ranked[0].candidate_id == "shot"
    assert ranked[0].adjustments["source_preference"] < ranked[1].adjustments.get("source_preference", 0)  # it was penalised... and still won


def test_repetition_penalty_lowers_rank_but_does_not_forbid_reuse():
    reused = cand("Solar factory aerial", cid="reused", source_reference="https://x/aerial", provider_id="aerial")
    fresh = cand("Rooftop solar installation timelapse", cid="fresh")
    used = UsedVisual(2, {f"{reused.provider}:aerial"} | {"url:" + "https://x/aerial".replace("https://", "https://")}, "", title_terms(reused), S.STOCK_VIDEO)
    hist = RankingHistory([used], scene_index=5)
    close = rank([(reused, fake_score(reused, 90)), (fresh, fake_score(fresh, 86))], history=hist)
    assert close[0].candidate_id == "fresh" and next(e for e in close if e.candidate_id == "reused").adjustments["repetition"] <= -10
    far = rank([(reused, fake_score(reused, 97)), (fresh, fake_score(fresh, 70))], history=hist)
    assert far[0].candidate_id == "reused"  # narration may genuinely need it again
    lenient = VisualPreferences(avoid_repeated_visuals=False)
    soft = rank([(reused, fake_score(reused, 90)), (fresh, fake_score(fresh, 86))], lenient, hist)
    assert next(e for e in soft if e.candidate_id == "reused").adjustments["repetition"] > -4


def test_similar_generic_broll_is_penalised_less_than_exact_repeat():
    prev = cand("Solar panel factory production line closeup", cid="prev")
    look_alike = cand("Solar panel factory production line", cid="look", provider_id="other-id")
    hist = RankingHistory([UsedVisual(4, {"fake:prev"}, "", title_terms(prev), S.STOCK_VIDEO)], scene_index=5)
    r = rank([(look_alike, fake_score(look_alike, 90))], history=hist)
    assert -6 < r[0].adjustments["repetition"] < 0
    far_scene = RankingHistory([UsedVisual(0, {"fake:prev"}, "", title_terms(prev), S.STOCK_VIDEO)], scene_index=20)
    assert "repetition" not in rank([(look_alike, fake_score(look_alike, 90))], history=far_scene)[0].adjustments


def test_continuity_with_the_previous_scene_gives_a_small_bonus():
    c = cand("Solar panel factory exterior", cid="a")
    hist = RankingHistory(previous_terms=frozenset(title_terms(cand("Solar factory exterior wide"))))
    assert rank([(c, fake_score(c, 88))], history=hist)[0].adjustments["continuity"] == 2.0


def test_adjustments_are_capped_so_accuracy_always_dominates():
    prefs = VisualPreferences(prefer_real_visuals=True)
    c = cand("Solar factory", cid="a", source_type=S.WEB_IMAGE, kind="IMAGE", evidence_kind=EvidenceKind.EVIDENCE)
    hist = RankingHistory(previous_terms=frozenset(title_terms(c)))
    r = rank([(c, fake_score(c, 80))], prefs, hist, solar_brief(evidence_level=EvidenceLevel.REQUIRED, preferred_sources=["WEB_IMAGE"]))[0]
    assert r.rank_score - r.accuracy <= 6.0 + 1e-9


def test_best_plus_up_to_five_meaningfully_different_alternatives():
    pairs = []
    mix = [("Solar line stock", S.STOCK_VIDEO, 92), ("Solar line stock b", S.STOCK_VIDEO, 91), ("Solar line stock c", S.STOCK_VIDEO, 90.5),
           ("Real factory photo", S.WEB_IMAGE, 88), ("YouTube factory tour", S.YOUTUBE, 87), ("IRS style evidence page", S.SCREENSHOT, 84),
           ("AI concept render", S.AI_GENERATED, 80), ("Another stock image", S.STOCK_IMAGE, 79), ("Weak thing", S.STOCK_IMAGE, 55)]
    for t, s, a in mix:
        c = cand(t, source_type=s, cid=t)
        pairs.append((c, fake_score(c, a)))
    r = rank(pairs, VisualPreferences(prefer_evidence=False))
    assert r[0].role == "BEST" and r[0].candidate_id == "Solar line stock"
    alts = [e for e in r if e.role == "ALTERNATIVE"]
    assert len(alts) == 5 and all(e.accuracy >= 60 for e in alts)
    kinds = {next(c for c, _ in pairs if c.candidate_id == e.candidate_id).source_type for e in [r[0]] + alts}
    assert len(kinds) >= 4  # the alternatives are different kinds of visuals, not three near-identical stock clips
    assert next(e for e in r if e.candidate_id == "Weak thing").role == "OTHER"  # rejects are never promoted to an alternative


def test_alternatives_are_not_padded_with_weak_results():
    a, b, c = (cand(t, source_type=s, cid=t) for t, s in (("One", S.STOCK_VIDEO), ("Two", S.WEB_IMAGE), ("Three", S.STOCK_IMAGE)))
    r = rank([(a, fake_score(a, 91)), (b, fake_score(b, 74)), (c, fake_score(c, 40))])
    assert [e.role for e in r] == ["BEST", "ALTERNATIVE", "OTHER"]


def test_rejected_candidates_are_excluded():
    a, b = cand("Good", cid="a"), cand("Rejected", cid="b", status=CandidateStatus.REJECTED)
    r = rank([(a, fake_score(a, 80)), (b, fake_score(b, 99))])
    assert [e.candidate_id for e in r] == ["a"]


def test_prefer_evidence_and_prefer_real_rules_are_soft():
    shot = cand("Official page", cid="shot", source_type=S.SCREENSHOT, kind="IMAGE", evidence_kind=EvidenceKind.EVIDENCE)
    ai = cand("AI image", cid="ai", source_type=S.AI_GENERATED, kind="IMAGE")
    b = solar_brief(evidence_level=EvidenceLevel.REQUIRED)
    r = rank([(shot, fake_score(shot, 85)), (ai, fake_score(ai, 86))], VisualPreferences(prefer_real_visuals=True), brief=b)
    assert r[0].candidate_id == "shot"
    r2 = rank([(shot, fake_score(shot, 70)), (ai, fake_score(ai, 90))], VisualPreferences(prefer_real_visuals=True), brief=b)
    assert r2[0].candidate_id == "ai"  # 20 points of accuracy beat any preference
