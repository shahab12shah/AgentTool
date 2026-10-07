from __future__ import annotations

import threading
import time

import pytest

from app.core.exceptions import JobCancelled
from app.media.asset import SourceType as S
from app.research.engine import PER_PROVIDER, ResearchEngine, RunContext, decide_status
from app.research.models import Acquisition, CandidateStatus, EvidenceKind, EvidenceLevel, ResearchSettings, ResearchStatus
from app.research.providers.base import ProviderRegistry
from app.research.queries import QueryGenerator
from app.research.ranking import RankingHistory
from app.tests.helpers import FakeProvider, solar_brief
from app.visual.preferences import SourceKind, VisualPreferences

STRONG = {"title": "Solar panel manufacturing: silver paste printing on photovoltaic cells",
          "description": "Solar installations factory production line applying silver paste to silicon wafers", "tags": ["solar", "silver", "manufacturing"]}
WEAKISH = {"title": "Solar panels on a roof", "description": "rooftop installation"}
JUNK = {"title": "Sunset over a beach", "description": "palm trees"}


def ids():
    n = [0]

    def f():
        n[0] += 1
        return f"candidate_{n[0]:05d}"

    return f


def run(tmp_path, providers, brief=None, prefs=None, settings=None, fresh=False, expand=False, cancel=None, existing=None, queries=None):
    reg = ProviderRegistry()
    for p in providers:
        reg.register(p)
    brief = brief or solar_brief()
    settings = settings or ResearchSettings()
    qs = queries or QueryGenerator(settings).generate(brief)
    ctx = RunContext(tmp_path / "research", tmp_path, cancel or (lambda: False), fresh=fresh, expand=expand)
    engine = ResearchEngine(reg)
    out = engine.run(brief, qs, prefs or VisualPreferences(), settings, RankingHistory(), ctx, ids(), existing)
    return out, engine


def real_reports(out):
    """Provider reports excluding the '(SOURCE) no provider installed' notes that these minimal test registries produce."""
    return {r.provider: r for r in out.reports if not r.provider.startswith("(")}


def video_item(spec, **kw):
    return {**spec, "source_type": S.STOCK_VIDEO, **kw}


def test_full_pipeline_produces_best_and_alternatives_with_scores_and_reports(tmp_path):
    vid = FakeProvider([video_item(STRONG, id="v1"), video_item(WEAKISH, id="v2"), video_item(JUNK, id="v3")], [S.STOCK_VIDEO])
    img = FakeProvider([{**STRONG, "title": "Silver paste on solar cells, close-up photo", "id": "i1", "source_type": S.STOCK_IMAGE, "kind": "IMAGE"}], [S.STOCK_IMAGE], name="fake_img")
    out, _ = run(tmp_path, [vid, img])
    assert out.status is ResearchStatus.CANDIDATES_READY and "best" in out.message
    assert out.ranked[0].role == "BEST" and out.scores[out.ranked[0].candidate_id].overall >= 85
    assert all(c.candidate_id.startswith("candidate_") and c.scene_id == "scene_014" for c in out.candidates)
    assert set(real_reports(out)) == {"fake", "fake_img"} and all(r.status == "SUCCESS" and r.candidates > 0 for r in real_reports(out).values())
    assert set(out.scores) == {c.candidate_id for c in out.candidates}
    alts = [e for e in out.ranked if e.role == "ALTERNATIVE"]
    assert 1 <= len(alts) <= 5 and all(e.accuracy >= 60 for e in alts)
    assert "STOCK_VIDEO" in out.searched_sources and "STOCK_IMAGE" in out.searched_sources


def test_same_media_returned_by_several_queries_becomes_one_candidate(tmp_path):
    p = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO])
    out, _ = run(tmp_path, [p])
    assert len(p.calls) >= 2  # several queries hit the provider...
    assert len(out.candidates) == 1  # ...but the same clip is a single candidate
    assert len(out.candidates[0].query_ids) >= 2 and out.duplicates_removed >= 1


def test_same_media_from_two_providers_is_deduplicated(tmp_path):
    a = FakeProvider([video_item(STRONG, id="x1", url="https://same.test/clip")], [S.STOCK_VIDEO], name="a")
    b = FakeProvider([video_item(STRONG, id="y9", url="https://same.test/clip")], [S.STOCK_VIDEO], name="b")
    out, _ = run(tmp_path, [a, b])
    assert len(out.candidates) == 1 and out.duplicates_removed >= 1


# ------------------------------------------------------------------ failures
def test_partial_provider_failure_does_not_fail_the_scene(tmp_path):
    good = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO], name="stock")
    bad = FakeProvider([], [S.WEB_IMAGE], name="web", fail="Web service is down")
    out, _ = run(tmp_path, [good, bad])
    reports = real_reports(out)
    assert reports["stock"].status == "SUCCESS" and reports["web"].status == "FAILED" and "down" in reports["web"].error
    assert out.candidates and out.status is ResearchStatus.NEEDS_REVIEW  # usable, but a human should know a source failed
    assert "web" in out.message


def test_a_provider_that_fails_only_for_some_queries_still_counts_as_success_with_a_note(tmp_path):
    flaky = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO], fail_if=lambda q: "context" in q.purpose.lower() or "contextual" in q.purpose.lower())
    out, _ = run(tmp_path, [flaky])
    r = real_reports(out)["fake"]
    assert r.status == "SUCCESS" and r.error and out.candidates
    assert out.status is ResearchStatus.NEEDS_REVIEW


def test_crashing_provider_is_contained(tmp_path):
    class Boom(FakeProvider):
        def search(self, *a, **k):
            raise RuntimeError("bug in provider")

    good = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO], name="good")
    out, _ = run(tmp_path, [good, Boom([], [S.WEB_IMAGE], name="boom")])
    assert {n: r.status for n, r in real_reports(out).items()} == {"good": "SUCCESS", "boom": "FAILED"} and out.candidates


def test_all_providers_failed_is_an_error_not_random_footage(tmp_path):
    out, _ = run(tmp_path, [FakeProvider([], [S.STOCK_VIDEO], name="a", fail="a down"), FakeProvider([], [S.WEB_IMAGE], name="b", fail="b down")])
    assert out.status is ResearchStatus.ERROR and out.message.startswith("Visual research unavailable.")
    assert out.candidates == [] and out.ranked == [] and "a down" in out.message


def test_unavailable_providers_are_reported_not_hidden(tmp_path):
    keyless = FakeProvider([video_item(STRONG)], [S.STOCK_VIDEO], name="keyless", available=(False, "API key variable X is not set"))
    out, _ = run(tmp_path, [keyless])
    r = real_reports(out)["keyless"]
    assert r.status == "UNAVAILABLE" and "not set" in r.error and keyless.calls == []
    assert out.status is ResearchStatus.ERROR


def test_enabled_source_without_any_provider_is_reported(tmp_path):
    out, _ = run(tmp_path, [FakeProvider([video_item(STRONG)], [S.STOCK_VIDEO])])
    missing = {r.provider for r in out.reports if r.status == "UNAVAILABLE"}
    assert "(YOUTUBE)" in missing or "(WEB_IMAGE)" in missing  # "no provider installed for X" is visible to the user


# ------------------------------------------------------------------ low confidence, evidence
def test_low_confidence_when_every_candidate_is_below_the_minimum(tmp_path):
    p = FakeProvider([video_item(WEAKISH, id="w1", duration=8.0), video_item({"title": "Silver spoon", "description": "dining"}, id="w2")], [S.STOCK_VIDEO])
    out, _ = run(tmp_path, [p])
    best = out.scores[out.ranked[0].candidate_id]
    assert best.overall < 85 and out.status is ResearchStatus.LOW_CONFIDENCE
    assert out.message.startswith("Best available candidate:") and f"{best.overall:.0f}/100" in out.message and "Confidence: LOW" in out.message


def test_threshold_is_configurable(tmp_path):
    p = FakeProvider([video_item(WEAKISH, id="w1")], [S.STOCK_VIDEO])
    strict, _ = run(tmp_path, [p])
    lenient_prefs = VisualPreferences(min_accuracy_score=50)
    lenient, _ = run(tmp_path, [p], prefs=lenient_prefs, fresh=True)
    assert strict.status is ResearchStatus.LOW_CONFIDENCE and lenient.status is ResearchStatus.CANDIDATES_READY


def test_no_candidates_at_all_is_low_confidence_with_guidance(tmp_path):
    out, _ = run(tmp_path, [FakeProvider([], [S.STOCK_VIDEO])])
    assert out.status is ResearchStatus.LOW_CONFIDENCE and "Search Again" in out.message and out.ranked == []


def test_evidence_scene_with_only_decorative_results_needs_review(tmp_path):
    b = solar_brief(evidence_level=EvidenceLevel.REQUIRED, visual_type="EVIDENCE")
    p = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO])
    out, _ = run(tmp_path, [p], brief=b, prefs=VisualPreferences(min_accuracy_score=60))
    assert out.status in (ResearchStatus.NEEDS_REVIEW, ResearchStatus.LOW_CONFIDENCE)
    if out.status is ResearchStatus.NEEDS_REVIEW:
        assert "evidence" in out.message.lower()


# ------------------------------------------------------------------ preferences, expansion
def test_disabled_sources_are_not_searched_unless_the_user_expands(tmp_path):
    stock = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO], name="stock")
    web = FakeProvider([{**STRONG, "source_type": S.WEB_VIDEO, "id": "w1", "title": "Silver paste solar cell footage"}], [S.WEB_VIDEO], name="web")
    prefs = VisualPreferences()
    prefs.setting(SourceKind.WEB_VIDEOS).enabled = False
    out, _ = run(tmp_path, [stock, web], prefs=prefs)
    assert web.calls == [] and "web" not in {r.provider for r in out.reports}
    assert S.WEB_VIDEO.value in out.skipped_disabled
    expanded, _ = run(tmp_path, [stock, web], prefs=prefs, expand=True, fresh=True)
    assert web.calls and any(c.provider == "web" for c in expanded.candidates)
    assert prefs.setting(SourceKind.WEB_VIDEOS).enabled is False  # the user's setting is never changed silently
    adj = next(e.adjustments for e in expanded.ranked if next(c for c in expanded.candidates if c.candidate_id == e.candidate_id).provider == "web")
    assert adj["source_preference"] < adj["source_preference"] + 2  # allowed, but not favoured
    assert not expanded.skipped_disabled


# ------------------------------------------------------------------ cache, concurrency, cancel
def test_research_cache_is_reused_and_fresh_search_bypasses_it(tmp_path):
    p = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO])
    out1, _ = run(tmp_path, [p])
    n1 = len(p.calls)
    out2, _ = run(tmp_path, [p])
    assert len(p.calls) == n1 and real_reports(out2)["fake"].from_cache > 0  # same queries -> no provider calls
    out3, _ = run(tmp_path, [p], fresh=True)
    assert len(p.calls) == 2 * n1  # Fresh Search goes to the provider again
    assert out1.candidates[0].title == out2.candidates[0].title == out3.candidates[0].title


def test_cache_expires(tmp_path):
    p = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO])
    run(tmp_path, [p])
    n = len(p.calls)
    run(tmp_path, [p], settings=ResearchSettings(cache_days=0))  # everything is already older than 0 days
    assert len(p.calls) > n


def test_concurrency_per_provider_is_limited(tmp_path):
    p = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO], delay=0.05)
    run(tmp_path, [p], settings=ResearchSettings(concurrency=8, max_queries=8))
    assert 1 <= p.max_active <= PER_PROVIDER


def test_cancel_stops_research(tmp_path):
    flag = {"stop": False}
    p = FakeProvider([video_item(STRONG, id="v1")], [S.STOCK_VIDEO], delay=0.02)

    def cancel():
        flag["stop"] = True
        return True

    with pytest.raises(JobCancelled):
        run(tmp_path, [p], cancel=cancel)


def test_search_again_keeps_the_pool_and_never_resurrects_rejected_candidates(tmp_path):
    p = FakeProvider([video_item(STRONG, id="v1"), video_item(WEAKISH, id="v2")], [S.STOCK_VIDEO])
    first, engine = run(tmp_path, [p])
    rejected = next(c for c in first.candidates if c.provider_id == "v2")
    rejected.status = CandidateStatus.REJECTED
    again, _ = run(tmp_path, [p], existing=first.candidates, fresh=True)
    v2 = [c for c in again.candidates if c.provider_id == "v2"]
    assert len(v2) == 1 and v2[0].status is CandidateStatus.REJECTED  # the new duplicate merged into the rejected original
    assert all(e.candidate_id != v2[0].candidate_id for e in again.ranked)


def test_rescore_uses_stored_candidates_without_searching(tmp_path):
    p = FakeProvider([video_item(STRONG, id="v1"), video_item(WEAKISH, id="v2")], [S.STOCK_VIDEO])
    out, engine = run(tmp_path, [p])
    n = len(p.calls)
    lenient = VisualPreferences(min_accuracy_score=50)
    again = engine.evaluate_and_rank(solar_brief(), out.candidates, lenient, ResearchSettings(), RankingHistory())
    assert len(p.calls) == n and again.status is ResearchStatus.CANDIDATES_READY and again.scores[again.ranked[0].candidate_id].min_accuracy == 50


def test_decide_status_helper_covers_error_and_ok_paths():
    from app.research.engine import ResearchOutcome

    empty = ResearchOutcome([], {}, [], [], ResearchStatus.NOT_STARTED, "")
    assert decide_status(empty, solar_brief(), VisualPreferences(), False, False)[0] is ResearchStatus.ERROR
    assert decide_status(empty, solar_brief(), VisualPreferences(), True, False, final=False)[0] is ResearchStatus.LOW_CONFIDENCE
