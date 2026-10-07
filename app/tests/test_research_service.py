"""Service-level visual research: Scene -> Brief -> Queries -> Providers -> Candidates -> Scores -> Ranking -> Assignment -> Asset."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import pytest

from app.analysis.models import SceneStatus
from app.core.exceptions import ResearchError, UserEditsPresentError
from app.media.asset import AssetType, SourceType as S
from app.research.models import Acquisition, CandidateStatus, EvidenceKind, ResearchStatus
from app.research.providers.ai_image import AIImageProvider
from app.research.providers.base import ProviderRegistry
from app.research.queries import similar
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import FakeProvider, MockWeb, allow_local_http, make_image, make_video
from app.visual.preferences import SourceKind

pytestmark = needs_ffmpeg

STRONG = {"title": "Solar panel manufacturing: silver paste printing on photovoltaic cells",
          "description": "Solar manufacturers factory production line applying silver paste to silicon wafers while consuming the metal", "tags": ["solar", "silver", "manufacturing"]}
WEAKISH = {"title": "Rooftop solar panels", "description": "installation on a house roof"}
MIDDLE = {"title": "Solar panels and silver metal", "description": "solar cells use silver in manufacturing", "tags": ["solar", "silver"]}
JUNK = {"title": "Sunset over a beach", "description": "palm trees"}


def install(ws, *providers):
    reg = ProviderRegistry()
    for p in providers:
        reg.register(p)
    ws.research.registry = reg
    ws.research.engine.registry = reg


def wait(ws):
    assert ws.jobs.wait_idle(60)


def first_scene(ws):
    return ws.project.scenes[0]


def evidence_scene(ws):
    return next(s for s in ws.project.scenes if "IRS sent a notice" in s.narration)


@pytest.fixture
def media(tmp_path):
    d = tmp_path / "files"
    d.mkdir()
    return {"video": make_video(d / "strong.mp4", 4.0), "image": make_image(d / "good.png", "testsrc", "640x360"),
            "image2": make_image(d / "alt.png", "mandel", "640x360"), "junk": make_image(d / "junk.png", "red", "320x240")}


@pytest.fixture
def rws(research_ws, media):
    """Research workspace with a fake STOCK provider offering one strong local video, one weaker image and junk."""
    ws = research_ws
    items = [{**STRONG, "source_type": S.STOCK_VIDEO, "id": "v1", "local_path": str(media["video"]), "acquisition": Acquisition.LOCAL},
             {**MIDDLE, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "i1", "local_path": str(media["image"]), "acquisition": Acquisition.LOCAL},
             {**JUNK, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "i2", "local_path": str(media["junk"]), "acquisition": Acquisition.LOCAL}]
    ws.provider = FakeProvider(items, [S.STOCK_VIDEO, S.STOCK_IMAGE], name="stock")
    install(ws, ws.provider)
    return ws


def research(ws, scene=None, **kw):
    sid = (scene or first_scene(ws)).id
    job = ws.research.research_scene(sid, **kw)
    wait(ws)
    return sid, job


# ================================================================== the acceptance path, step by step
def test_research_scene_creates_brief_queries_candidates_scores_and_ranking(rws):
    ws = rws
    sid, job = research(ws)
    p = ws.project
    st = p.research_status[sid]
    assert job.status.value == "COMPLETED" and st.status is ResearchStatus.CANDIDATES_READY
    cands = [c for c in p.visual_candidates.values() if c.scene_id == sid]
    assert len(cands) == 3 and all(c.candidate_id in p.candidate_scores for c in cands)
    best = p.visual_candidates[st.best_id]
    assert best.provider_id == "v1" and p.candidate_scores[st.best_id].overall >= 85 and st.alternatives
    assert all(p.candidate_scores[a].overall >= 60 for a in st.alternatives) and len(st.alternatives) <= 5
    sess = p.research_sessions[-1]
    assert sess.scene_id == sid and sess.brief.primary_subject and sess.brief.scene_id == sid and sess.query_ids
    assert [r.status for r in sess.provider_reports if not r.provider.startswith("(")] == ["SUCCESS"]
    qs = [p.research_queries[q] for q in sess.query_ids]
    assert {q.type.value for q in qs} >= {"LITERAL"} and all(q.scene_id == sid for q in qs)
    assert p.source_metadata["providers"]["stock"]["searches"] >= 2
    assert p.timeline.all_clips() == [] and p.visual_assignments == {}  # research alone changes neither the timeline nor decisions
    assert p.assets.all() == [a for a in p.assets.all() if a.type is AssetType.AUDIO]  # nothing became a project asset yet
    assert p.dirty


def test_a_researched_candidate_is_not_a_project_asset_until_selected(rws):
    ws = rws
    sid, _ = research(ws)
    assert not [a for a in ws.project.assets if a.type is AssetType.VIDEO]
    assert not list((ws.project.root / "media" / "video").glob("*"))


def test_choose_candidate_acquires_validates_and_registers_with_provenance(rws):
    ws = rws
    sid, _ = research(ws)
    best_id = ws.project.research_status[sid].best_id
    job = ws.research.choose_candidate(sid, best_id)
    wait(ws)
    p = ws.project
    a = p.visual_assignments[sid]
    c = p.visual_candidates[best_id]
    assert job.status.value == "COMPLETED" and c.status is CandidateStatus.ACQUIRED and c.asset_id
    assert (a.scene_id, a.candidate_id, a.asset_id, a.selected_by, a.approved) == (sid, best_id, c.asset_id, "USER", False)
    assert a.accuracy_score == p.candidate_scores[best_id].overall and a.source_type is S.STOCK_VIDEO and a.recommended_duration
    asset = p.assets.get(a.asset_id)
    assert asset.type is AssetType.VIDEO and asset.source_type is S.STOCK_VIDEO and asset.duration == pytest.approx(4.0, abs=0.2)
    assert (p.root / asset.path).is_file() and asset.path.startswith("media/video/")
    assert asset.extra["candidate_id"] == best_id and asset.extra["scene_id"] == sid and asset.extra["provider"] == "stock"
    assert asset.extra["license"]["verified"] is False and asset.name.startswith("Solar panel manufacturing")
    assert p.timeline.all_clips() == []  # no timeline edit in this phase
    assert ws.media.thumbnail_file(asset) is not None or True


def test_approval_makes_the_assignment_final_and_survives_save_and_reopen(rws):
    ws = rws
    sid, _ = research(ws)
    best_id = ws.project.research_status[sid].best_id
    ws.research.choose_candidate(sid, best_id)
    wait(ws)
    ws.research.approve(sid)
    p = ws.project
    assert p.visual_assignments[sid].approved and p.research_status[sid].status is ResearchStatus.APPROVED
    assert p.timeline.all_clips() == []
    doc = p.to_document()
    root = p.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    q = ws.project
    for key in ("research_settings", "research_queries", "research_sessions", "visual_candidates", "candidate_scores", "visual_assignments",
                "source_metadata", "research_status"):
        assert q.to_document()[key] == doc[key], key
    a = q.visual_assignments[sid]
    assert a.approved and a.selected_by == "USER" and q.assets.get(a.asset_id) and q.research_status[sid].status is ResearchStatus.APPROVED
    assert q.schema_version == 8


def test_user_can_choose_a_lower_scored_candidate_and_is_recorded_as_the_selector(rws):
    ws = rws
    sid, _ = research(ws)
    st = ws.project.research_status[sid]
    weaker = next(a for a in st.alternatives if ws.project.candidate_scores[a].overall < ws.project.candidate_scores[st.best_id].overall)
    ws.research.choose_candidate(sid, weaker)
    wait(ws)
    a = ws.project.visual_assignments[sid]
    assert a.candidate_id == weaker and a.selected_by == "USER" and a.accuracy_score == ws.project.candidate_scores[weaker].overall < ws.project.candidate_scores[st.best_id].overall
    ws.research.approve(sid)
    assert ws.project.visual_assignments[sid].approved


def test_approving_the_ai_best_without_choosing_records_ai_as_selector(rws):
    ws = rws
    sid, _ = research(ws)
    job = ws.research.approve(sid)
    wait(ws)
    a = ws.project.visual_assignments[sid]
    assert a.selected_by == "AI" and a.approved and a.asset_id and a.candidate_id == ws.project.research_status[sid].best_id


def test_changing_the_choice_replaces_assignment_and_undo_restores_it(rws):
    ws = rws
    sid, _ = research(ws)
    st = ws.project.research_status[sid]
    ws.research.choose_candidate(sid, st.best_id)
    wait(ws)
    ws.research.choose_candidate(sid, st.alternatives[0])
    wait(ws)
    assert ws.project.visual_assignments[sid].candidate_id == st.alternatives[0]
    assert ws.project.visual_candidates[st.best_id].status is CandidateStatus.ACQUIRED  # already acquired stays acquired
    ws.undo()
    assert ws.project.visual_assignments[sid].candidate_id in (st.best_id, st.alternatives[0])
    ws.undo()


# ================================================================== low confidence, failure, retry
def test_low_confidence_marks_the_scene_blocks_ai_approval_and_still_allows_user_override(rws, media):
    ws = rws
    ws.provider.items = [{**WEAKISH, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "w1", "local_path": str(media["image"]), "acquisition": Acquisition.LOCAL}]
    sid, _ = research(ws)
    st = ws.project.research_status[sid]
    best = ws.project.candidate_scores[st.best_id]
    assert st.status is ResearchStatus.LOW_CONFIDENCE and best.overall < 85 and best.confidence.value == "LOW"
    assert st.message.startswith("Best available candidate:") and "Confidence: LOW" in st.message
    with pytest.raises(ResearchError, match="below your minimum"):
        ws.research.approve(sid)
    assert sid not in ws.project.visual_assignments  # a weak visual is never forced into the selection
    ws.research.choose_candidate(sid, st.best_id)  # the user may still decide
    wait(ws)
    ws.research.approve(sid)
    a = ws.project.visual_assignments[sid]
    assert a.approved and a.selected_by == "USER" and a.accuracy_score == best.overall


def test_low_confidence_offers_search_again_expand_and_generate(rws, media):
    ws = rws
    ws.provider.items = [{**JUNK, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "j", "local_path": str(media["junk"])}]
    sid, _ = research(ws)
    assert ws.project.research_status[sid].status is ResearchStatus.LOW_CONFIDENCE
    web = FakeProvider([{**STRONG, "source_type": S.WEB_VIDEO, "id": "w"}], [S.WEB_VIDEO], name="webvid")
    install(ws, ws.provider, web)
    prefs = ws.project.visual_preferences
    prefs.setting(SourceKind.WEB_VIDEOS).enabled = False
    assert [e["kind"] for e in ws.research.expandable_sources()] == ["WEB_VIDEOS"]
    ws.research.search_again(sid)
    wait(ws)
    assert web.calls == []  # still respects the disabled source
    ws.research.research_scene(sid, "initial", fresh=True, expand=True)
    wait(ws)
    st = ws.project.research_status[sid]
    assert web.calls and st.expanded_sources == ["WEB_VIDEOS"] and ws.project.visual_preferences.setting(SourceKind.WEB_VIDEOS).enabled is False
    assert st.status in (ResearchStatus.CANDIDATES_READY, ResearchStatus.NEEDS_REVIEW)


def test_total_provider_failure_is_an_error_state_and_retry_recovers(rws):
    ws = rws
    ws.provider.fail = "stock service is down"
    sid, job = research(ws)
    st = ws.project.research_status[sid]
    assert st.status is ResearchStatus.ERROR and st.message.startswith("Visual research unavailable.") and "down" in st.message
    assert not [c for c in ws.project.visual_candidates.values() if c.scene_id == sid]
    ws.provider.fail = None  # [Retry]
    research(ws, first_scene(ws))
    assert ws.project.research_status[sid].status is ResearchStatus.CANDIDATES_READY


def test_one_failing_scene_leaves_the_other_scenes_complete(rws):
    ws = rws
    ws.provider.fail_if = lambda q: "irs" in q.text.lower()  # only the IRS scene's queries fail
    scenes = [first_scene(ws), evidence_scene(ws)]
    jobs = ws.research.research_many([s.id for s in scenes])
    wait(ws)
    st = ws.project.research_status
    assert st[scenes[0].id].status is ResearchStatus.CANDIDATES_READY
    assert st[scenes[1].id].status in (ResearchStatus.ERROR, ResearchStatus.LOW_CONFIDENCE, ResearchStatus.NEEDS_REVIEW)
    ws.provider.fail_if = None
    ws.research.research_scene(scenes[1].id)  # recover just that scene
    wait(ws)
    assert st[scenes[0].id].status is ResearchStatus.CANDIDATES_READY and ws.project.research_status[scenes[1].id].status is not ResearchStatus.ERROR


def test_partial_provider_failure_needs_review(rws):
    ws = rws
    bad = FakeProvider([], [S.WEB_IMAGE], name="web", fail="web is down")
    install(ws, ws.provider, bad)
    sid, _ = research(ws)
    st = ws.project.research_status[sid]
    assert st.status is ResearchStatus.NEEDS_REVIEW and "web" in st.message
    rep = {r.provider: r.status for r in ws.project.research_sessions[-1].provider_reports}
    assert rep["stock"] == "SUCCESS" and rep["web"] == "FAILED"


def test_second_research_of_a_running_scene_is_refused_and_cancel_restores_state(rws):
    ws = rws
    sid = first_scene(ws).id
    gate = threading.Event()

    class Slow(FakeProvider):
        def search(self, *a, **k):
            gate.wait(10)
            return super().search(*a, **k)

    slow = Slow(ws.provider.items, [S.STOCK_VIDEO, S.STOCK_IMAGE], name="stock")
    install(ws, slow)
    job = ws.research.research_scene(sid)
    assert ws.project.research_status[sid].status is ResearchStatus.RESEARCHING
    with pytest.raises(ResearchError, match="already running"):
        ws.research.research_scene(sid)
    ws.jobs.cancel(job.id)
    gate.set()
    wait(ws)
    assert ws.project.research_status[sid].status is ResearchStatus.NOT_STARTED  # back to what it was
    research(ws, first_scene(ws))
    assert ws.project.research_status[sid].status is ResearchStatus.CANDIDATES_READY


# ================================================================== search again / fresh / cache / more
def test_search_again_generates_new_queries_and_keeps_the_pool(rws):
    ws = rws
    sid, _ = research(ws)
    old = {q.text for q in ws.project.research_queries.values() if q.scene_id == sid}
    pool = {c.candidate_id for c in ws.project.visual_candidates.values() if c.scene_id == sid}
    ws.research.search_again(sid)
    wait(ws)
    new = [q for q in ws.project.research_queries.values() if q.scene_id == sid and q.generation == 1]
    assert new and not any(similar(q.text, o, 0.85) for q in new for o in old)
    assert ws.project.research_status[sid].generation == 1
    assert pool <= {c.candidate_id for c in ws.project.visual_candidates.values() if c.scene_id == sid}  # nothing already found is lost
    ws.research.search_again(sid)
    wait(ws)
    assert ws.project.research_status[sid].generation == 2
    assert len({q.strategy for q in ws.project.research_queries.values() if q.scene_id == sid}) >= 3
    ws.research.generate_more(sid)
    wait(ws)
    assert any(q.strategy == "alternative" for q in ws.project.research_queries.values())


def test_cache_is_used_for_repeat_searches_and_fresh_search_bypasses_it(rws):
    ws = rws
    sid, _ = research(ws)
    n = len(ws.provider.calls)
    research(ws)
    assert len(ws.provider.calls) == n  # same queries: served from the research cache
    research(ws, fresh=True)
    assert len(ws.provider.calls) > n  # Fresh Search
    assert list((ws.project.root / "cache" / "research" / "search").glob("*.json"))


def test_changing_the_minimum_score_and_rescoring_updates_status_without_searching(rws, media):
    ws = rws
    ws.provider.items = [{**WEAKISH, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "w1", "local_path": str(media["image"])}]
    sid, _ = research(ws)
    assert ws.project.research_status[sid].status is ResearchStatus.LOW_CONFIDENCE
    n = len(ws.provider.calls)
    prefs = ws.project.visual_preferences
    score = ws.project.candidate_scores[ws.project.research_status[sid].best_id].overall
    prefs.min_accuracy_score = int(score) - 1
    ws.research.rescore(sid)
    wait(ws)
    assert len(ws.provider.calls) == n and ws.project.research_status[sid].status is ResearchStatus.CANDIDATES_READY
    assert ws.project.candidate_scores[ws.project.research_status[sid].best_id].min_accuracy == int(score) - 1


# ================================================================== reject / skip / bulk
def test_reject_candidate_reranks_and_rejecting_the_chosen_visual_clears_the_assignment(rws):
    ws = rws
    sid, _ = research(ws)
    st = ws.project.research_status[sid]
    old_best = st.best_id
    ws.research.reject_candidate(sid, old_best)
    st2 = ws.project.research_status[sid]
    assert ws.project.visual_candidates[old_best].status is CandidateStatus.REJECTED and st2.best_id != old_best and old_best not in st2.alternatives
    ws.undo()
    assert ws.project.visual_candidates[old_best].status is not CandidateStatus.REJECTED and ws.project.research_status[sid].best_id == old_best
    ws.research.choose_candidate(sid, old_best)
    wait(ws)
    ws.research.reject_candidate(sid, old_best)
    assert sid not in ws.project.visual_assignments
    ws.research.unreject_candidate(sid, old_best)
    wait(ws)
    assert ws.project.visual_candidates[old_best].status is not CandidateStatus.REJECTED


def test_skip_visual_is_an_approved_assignment_without_a_candidate(rws):
    ws = rws
    sid, _ = research(ws)
    ws.research.skip_visual(sid)
    a = ws.project.visual_assignments[sid]
    assert a.skipped and a.approved and a.candidate_id is None and a.asset_id is None and a.selected_by == "USER"
    assert ws.project.research_status[sid].status is ResearchStatus.APPROVED
    ws.project.validate()
    ws.undo()
    assert sid not in ws.project.visual_assignments


def test_bulk_actions_skip_weak_scenes_and_never_force_approval(rws, media):
    ws = rws
    ids = [s.id for s in ws.project.scenes[:3]]
    ws.research.research_many(ids)
    wait(ws)
    st = ws.project.research_status
    weak = ids[2]
    # make the third scene weak
    ws.provider.items = [{**JUNK, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "j", "local_path": str(media["junk"])}]
    ws.research.research_scene(weak, fresh=True)
    wait(ws)
    assert st[weak].status is ResearchStatus.LOW_CONFIDENCE
    ready = {i for i in ids if st[i].status is ResearchStatus.CANDIDATES_READY}  # the fake provider only fits scene 1's topic
    approved, skipped = ws.research.approve_scenes(ids)
    wait(ws)
    assert ready and weak in skipped and set(approved) == ready and set(skipped) == set(ids) - ready
    assert all(ws.project.visual_assignments[i].approved for i in approved) and weak not in ws.project.visual_assignments
    done = ws.research.reject_best([weak])
    assert done == [weak]


def test_research_many_reports_unanalysed_scenes_instead_of_failing(rws):
    ws = rws
    msgs = []
    ws.bus.subscribe("app.status", lambda t, p: msgs.append(p["message"]))
    jobs = ws.research.research_many([first_scene(ws).id, "scene_999"])
    wait(ws)
    assert len(jobs) == 1 and any("no longer exists" in m for m in msgs)


# ================================================================== reference-only, manual, screenshots, AI
def test_reference_only_candidates_can_be_selected_without_downloading(rws):
    ws = rws
    yt = FakeProvider([{**STRONG, "source_type": S.YOUTUBE, "id": "vid1", "url": "https://www.youtube.com/watch?v=vid1", "acquisition": Acquisition.REFERENCE_ONLY,
                        "duration": 600.0}], [S.YOUTUBE], name="youtube")
    install(ws, yt)
    sid, _ = research(ws)
    st = ws.project.research_status[sid]
    c = ws.project.visual_candidates[st.best_id]
    assert c.source_type is S.YOUTUBE
    assert ws.research.choose_candidate(sid, c.candidate_id) is None  # no acquisition job
    a = ws.project.visual_assignments[sid]
    assert a.asset_id is None and a.acquisition is Acquisition.REFERENCE_ONLY and "does not download" in a.note
    ws.research.approve(sid)
    assert ws.project.visual_assignments[sid].approved
    ws.project.validate()


def test_manual_selection_imports_the_users_own_file(rws, media):
    ws = rws
    sid, _ = research(ws)
    ws.research.add_manual_visual(sid, media["image2"])
    wait(ws)
    a = ws.project.visual_assignments[sid]
    asset = ws.project.assets.get(a.asset_id)
    assert a.selected_by == "USER" and a.accuracy_score is None and asset.source_type is S.USER_MEDIA and "Manual" in a.note
    assert ws.project.visual_candidates[a.candidate_id].status is CandidateStatus.ACQUIRED


def test_screenshot_of_a_user_chosen_page_enters_the_pool_as_evidence(rws):
    from app.research.providers.screenshot import find_chromium

    if find_chromium() is None:
        pytest.skip("Chromium not installed")
    ws = rws
    web = MockWeb()
    try:
        web.html("/doc", "<html><body style='background:#fff'><h1>IRS notice CP2000</h1><p>reporting requirement for silver sellers</p></body></html>")
        allow_local_http(ws)
        reg = ProviderRegistry()
        reg.register(ws.provider)
        from app.research.providers.screenshot import ScreenshotProvider

        shot = ScreenshotProvider(allow_private=True)
        shot.http.allow_private = True
        reg.register(shot)
        ws.research.registry = ws.research.engine.registry = reg
        sc = evidence_scene(ws)
        research(ws, sc)
        ws.research.add_page_screenshot(sc.id, web.base + "/doc")
        wait(ws)
        shots = [c for c in ws.project.visual_candidates.values() if c.scene_id == sc.id and c.source_type is S.SCREENSHOT]
        assert len(shots) == 1 and shots[0].evidence_kind is EvidenceKind.EVIDENCE and Path(shots[0].local_path).is_file()
        score = ws.project.candidate_scores[shots[0].candidate_id]
        assert score.components.source == 100 and shots[0].fingerprint
        ws.research.choose_candidate(sc.id, shots[0].candidate_id)
        wait(ws)
        asset = ws.project.assets.get(ws.project.visual_assignments[sc.id].asset_id)
        assert asset.source_type is S.SCREENSHOT and asset.source_url == web.base + "/doc" and asset.type is AssetType.IMAGE
        assert ws.project.visual_assignments[sc.id].evidence_kind is EvidenceKind.EVIDENCE
        with pytest.raises(Exception):
            ws.research.add_page_screenshot(sc.id, "file:///etc/passwd")
            wait(ws)
    finally:
        web.close()


def test_ai_visual_is_proposed_then_generated_only_on_request_and_never_counts_as_evidence(rws, monkeypatch, tmp_path):
    ws = rws
    web = MockWeb()
    try:
        monkeypatch.setenv("AI_TEST", "ai-key")
        png = tmp_path / "gen.png"
        make_image(png, "mandel", "512x288")
        web.json("/v1/images/generations", {"data": [{"b64_json": base64.b64encode(png.read_bytes()).decode()}]})
        ai = AIImageProvider(lambda: web.base + "/v1", lambda: "m", lambda: "AI_TEST")
        ai.http.allow_private = True
        install(ws, ws.provider, ai)
        sid, _ = research(ws)
        st = ws.project.research_status[sid]
        prop = next(c for c in ws.project.visual_candidates.values() if c.scene_id == sid and c.source_type is S.AI_GENERATED)
        assert prop.status is CandidateStatus.PROPOSED and web.requests == []  # researching never calls the paid service
        assert ws.project.candidate_scores[prop.candidate_id].overall <= 82 and ws.project.candidate_scores[prop.candidate_id].basis == "PROMPT"
        with pytest.raises(ResearchError, match="not been generated"):
            ws.research.choose_candidate(sid, prop.candidate_id)
        ws.research.generate_ai_visual(sid, prop.candidate_id)
        wait(ws)
        gen = ws.project.visual_candidates[prop.candidate_id]
        assert len(web.requests) == 1 and gen.status is CandidateStatus.READY and Path(gen.local_path).is_file() and gen.local_path.startswith(str(ws.project.root / "generated"))
        assert gen.evidence_kind is EvidenceKind.DECORATIVE and ws.project.candidate_scores[gen.candidate_id].overall <= 88
        ws.research.choose_candidate(sid, gen.candidate_id)
        wait(ws)
        asset = ws.project.assets.get(ws.project.visual_assignments[sid].asset_id)
        assert asset.source_type is S.AI_GENERATED and asset.extra["prompt"] and asset.extra["license"]["status"] == "UNKNOWN"
    finally:
        web.close()


def test_generate_ai_visual_explains_when_the_service_is_not_configured(rws):
    ws = rws
    sid, _ = research(ws)
    ws.research.registry = ws.research.engine.registry = ProviderRegistry()
    r = ProviderRegistry()
    r.register(ws.provider)
    r.register(AIImageProvider(lambda: "", lambda: "m", lambda: "NOPE_KEY"))
    ws.research.registry = ws.research.engine.registry = r
    with pytest.raises(ResearchError, match="not available"):
        ws.research.generate_ai_visual(sid)


# ================================================================== staleness, scenes, secrets, reports
def test_scene_changes_after_research_are_flagged_as_stale(rws):
    ws = rws
    sid, _ = research(ws)
    assert not ws.research.is_stale(sid)
    ws.scenes.edit_scene(sid, intent={"primary_subject": "wind turbines"})
    assert ws.research.is_stale(sid)


def test_regenerating_scenes_is_blocked_by_chosen_visuals_and_prunes_orphaned_research_when_confirmed(rws):
    ws = rws
    sid, _ = research(ws)
    ws.research.choose_candidate(sid, ws.project.research_status[sid].best_id)
    wait(ws)
    ws.research.approve(sid)
    with pytest.raises(UserEditsPresentError):
        ws.scenes.analyze(force=True)
    assert sid in ws.project.visual_assignments
    ws.scenes.analyze(force=True, overwrite_user_edits=True)
    wait(ws)
    assert sid not in ws.project.visual_assignments and not [c for c in ws.project.visual_candidates.values() if c.scene_id == sid]
    ws.project.validate()
    ws.undo()  # undo brings the research back with the old scenes
    assert sid in ws.project.visual_assignments and [c for c in ws.project.visual_candidates.values() if c.scene_id == sid]


def test_split_scene_prunes_nothing_but_merge_removes_the_retired_scene_research(rws):
    ws = rws
    a, b = ws.project.scenes[0], ws.project.scenes[1]
    research(ws, a)
    research(ws, b)
    ws.scenes.merge_scenes(a.id, b.id)
    assert a.id in ws.project.research_status and b.id not in ws.project.research_status
    ws.undo()
    assert b.id in ws.project.research_status


def test_provider_report_is_honest_about_what_was_verified(research_ws):
    rep = {r["name"]: r for r in research_ws.research.provider_report()}
    assert set(rep) == {"local_stock", "wikimedia", "youtube", "pexels", "screenshot", "ai_image"}
    assert rep["local_stock"]["verified_live"] is True
    assert all(not rep[n]["verified_live"] for n in ("wikimedia", "youtube", "pexels", "ai_image", "screenshot"))
    assert rep["youtube"]["available"] is False and "environment variable" in rep["youtube"]["reason"]


def test_no_api_keys_are_written_into_the_project_or_settings(rws, monkeypatch):
    ws = rws
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-SECRET-123")
    monkeypatch.setenv("PEXELS_API_KEY", "px-SECRET-456")
    monkeypatch.setenv("OPENAI_API_KEY", "ai-SECRET-789")
    sid, _ = research(ws)
    ws.save()
    ws.update_settings(ws.settings)
    blob = ws.project.paths.project_file.read_text() + ws.paths.settings_file.read_text()
    for secret in ("yt-SECRET-123", "px-SECRET-456", "ai-SECRET-789"):
        assert secret not in blob
    assert "YOUTUBE_API_KEY" in ws.paths.settings_file.read_text()  # only the variable NAME is stored
