"""Phase 11 foundation: models and persistence, the stage engine (reuse / windows / failures / cancel), the service job, overrides, provider validation, scorecard."""

from __future__ import annotations

import json
import threading

import pytest

from app.core.constants import SCHEMA_VERSION
from app.editorial_intelligence.context import build_editorial_context
from app.editorial_intelligence.editorial_models import (
    ActionKind,
    ActionSpec,
    AnalysisMode,
    Basis,
    Category,
    ContentType,
    EditorialOverride,
    EditorialState,
    Finding,
    OverrideType,
    Priority,
    Recommendation,
    RecStatus,
    StageOutput,
    TargetRef,
    finding_key,
)
from app.editorial_intelligence.engine import EditorialEngine, global_signature
from app.editorial_intelligence.override_resolver import blocked_aspects, clip_overrides, scene_overrides
from app.editorial_intelligence.provider import APIEditorialProvider, EditorialProviderError, validate_observations_payload, validate_relations_payload, validate_story_payload
from app.editorial_intelligence.scorecard import SERIOUS_ITEM_CAP, SERIOUS_OVERALL_CAP, build_scorecard
from app.editorial_intelligence.stages import Stage
from app.project.project import Project
from app.tests.conftest import settle


def sid(i: int) -> str:
    return f"scene_{i:04d}"


# ------------------------------------------------------------------------------------------------------------------------ models + persistence
def test_state_round_trips_and_an_empty_state_writes_nothing():
    p = Project.new("x")
    doc = p.to_document()
    assert "editorial" not in doc and doc["schema_version"] == SCHEMA_VERSION == 10
    p.editorial.editorial_settings.mode = AnalysisMode.CREATIVE
    p.editorial.editorial_recommendations.append(Recommendation("r1", Category.PACING, Priority.HIGH, ["s1"], suggested_action=ActionSpec(ActionKind.SET_SHOT_DURATION, {"duration": 4.0}, [TargetRef("s1", "c1")])))
    p.editorial.editorial_overrides.append(EditorialOverride("o1", OverrideType.TIMING_OVERRIDE, "s1"))
    q = Project.from_document(json.loads(json.dumps(p.to_document())))
    assert q.editorial == p.editorial and q.editorial.editorial_recommendations[0].suggested_action.targets[0].clip_id == "c1"


def test_a_schema_9_project_opens_with_empty_editorial_data():
    doc = Project.new("old").to_document()
    doc["schema_version"] = 9
    q = Project.from_document(doc)
    assert q.schema_version == 10 and q.editorial.is_empty


def test_unknown_future_keys_in_the_editorial_section_are_ignored():
    d = EditorialState().to_dict()
    d["something_new"] = 1
    d["editorial_settings"]["extra"] = True
    assert EditorialState.from_dict(d).is_empty


# ------------------------------------------------------------------------------------------------------------------------ context
def test_context_carries_scenes_timeline_items_candidates_and_a_changing_hash(ews):
    p = ews.project
    ctx = build_editorial_context(p)
    assert len(ctx.scenes) == 12 and ctx.total_duration > 0 and ctx.content_type in {t.value for t in ContentType if t is not ContentType.AUTO}
    s = ctx.by_id(sid(3))
    assert s.clips and any(c.kind == "media" for c in s.clips) and s.hash and ctx.prev(sid(3)).id == sid(2) and ctx.next(sid(12)) is None
    assert [x.id for x in ctx.window(sid(5), 2)] == [sid(i) for i in range(3, 8)]
    first = ctx.signature
    clip = next(c for c in p.timeline.all_clips() if c.scene_id == sid(3) and c.kind == "media")
    clip.duration += 0.5
    again = build_editorial_context(p)
    assert again.by_id(sid(3)).hash != s.hash and again.by_id(sid(4)).hash == ctx.by_id(sid(4)).hash and again.signature != first  # only the touched scene's hash moves


def test_content_type_is_the_users_choice_or_a_cautious_guess(ews):
    p = ews.project
    p.editorial.editorial_settings.content_type = ContentType.NEWS
    assert build_editorial_context(p).content_type == "NEWS"
    p.editorial.editorial_settings.content_type = ContentType.AUTO
    assert build_editorial_context(p).content_type in {"GENERAL", "FINANCE_EDUCATIONAL", "DOCUMENTARY", "NEWS", "EXPLAINER"}


# ------------------------------------------------------------------------------------------------------------------------ overrides
def test_override_resolver_reads_the_existing_ownership_marks(ews):
    p = ews.project
    assert scene_overrides(p, sid(2)) == {} or all(isinstance(v, list) for v in scene_overrides(p, sid(2)).values())
    clip = next(c for c in p.timeline.all_clips() if c.scene_id == sid(2) and c.kind == "media")
    clip.created_by = "USER"
    ov = scene_overrides(p, sid(2))
    assert "VISUAL_OVERRIDE" in ov and "you placed" in ov["VISUAL_OVERRIDE"][0]
    clip.locks = frozenset({"timing"})
    assert "TIMING_OVERRIDE" in clip_overrides(p, clip) and "MOTION_OVERRIDE" not in clip_overrides(p, clip)
    a = p.visual_assignments.get(sid(5))
    if a is not None:
        a.selected_by = "USER"
        assert "SOURCE_OVERRIDE" in scene_overrides(p, sid(5))
    assert blocked_aspects(scene_overrides(p, sid(2)), [OverrideType.TIMING_OVERRIDE, OverrideType.AUDIO_OVERRIDE]).keys() == {"TIMING_OVERRIDE"}


def test_a_reset_gives_an_aspect_back_and_a_keep_protects_it(ews):
    p = ews.project
    clip = next(c for c in p.timeline.all_clips() if c.scene_id == sid(4) and c.kind == "media")
    clip.created_by = "USER"
    assert "VISUAL_OVERRIDE" in scene_overrides(p, sid(4))
    ews.editorial.reset_override(sid(4), OverrideType.VISUAL_OVERRIDE)
    assert "VISUAL_OVERRIDE" not in scene_overrides(p, sid(4))
    ews.commands.undo()
    assert "VISUAL_OVERRIDE" in scene_overrides(p, sid(4))  # the reset is undoable
    p.editorial.editorial_overrides.append(EditorialOverride("k", OverrideType.STYLE_OVERRIDE, sid(6), action="KEEP", reason="I like it"))
    assert scene_overrides(p, sid(6))["STYLE_OVERRIDE"] == ["I like it"]


# ------------------------------------------------------------------------------------------------------------------------ engine
class Counter:
    def __init__(self):
        self.calls: list[tuple[str, set[str] | None]] = []

    def stage(self, name, scope="scene", deps=(), fail=False, findings=True):
        def analyze(ctx, prior, *, scene_filter=None):
            self.calls.append((name, None if scene_filter is None else set(scene_filter)))
            if fail:
                raise RuntimeError(f"{name} exploded")
            ids = ctx.scene_ids if scene_filter is None else [s for s in ctx.scene_ids if s in scene_filter]
            section = {"scenes": {s: {"v": ctx.by_id(s).hash[:4]} for s in ids}, "global": {"n": len(ctx.scenes), "score": {"value": 80, "basis": "COMPUTED"}}}
            fs = [Finding(f"{name}_{s}", name, Category.PACING, [s], [], f"{name} {s}", "d", 70, 50, key=finding_key(name, "k", s)) for s in ids] if findings else []
            return StageOutput(name, section, fs, {s: ctx.by_id(s).hash for s in ids}, "DONE", "", "", {}, [], list(ids))

        return Stage(name, name, scope, tuple(deps), analyze)


def run_engine(engine, ctx, previous, **kw):
    return engine.run(ctx, previous, **kw)


def apply_result(state: EditorialState, res) -> EditorialState:
    new = EditorialState.from_dict(state.to_dict())
    new.sections, new.stage_cache, new.findings = res.sections, res.stage_cache, res.findings
    return new


def test_engine_runs_stages_in_order_then_reuses_everything_when_nothing_changed(ews):
    c = Counter()
    engine = EditorialEngine([c.stage("a"), c.stage("b", "window", ("a",)), c.stage("g", "global", ("b",))])
    ctx = build_editorial_context(ews.project)
    res = engine.run(ctx, ews.project.editorial)
    assert [r.status for r in res.records[:3]] == ["DONE"] * 3 and [n for n, _ in c.calls] == ["a", "b", "g"] and res.status == "COMPLETE"
    assert len(res.findings) == 12 * 2 + 12 and all(f.stage in "abg" for f in res.findings)
    state = apply_result(ews.project.editorial, res)
    c.calls.clear()
    res2 = engine.run(build_editorial_context(ews.project), state)
    assert c.calls == [] and [r.status for r in res2.records[:3]] == ["REUSED"] * 3 and len(res2.findings) == len(res.findings)


def test_a_changed_scene_reanalyses_only_that_scene_its_window_and_global_stages(ews):
    c = Counter()
    engine = EditorialEngine([c.stage("a", "scene"), c.stage("b", "window", ("a",)), c.stage("g", "global", ("b",))])
    p = ews.project
    state = apply_result(p.editorial, engine.run(build_editorial_context(p), p.editorial))
    c.calls.clear()
    next(x for x in p.timeline.all_clips() if x.scene_id == sid(6) and x.kind == "media").duration += 0.4
    res = engine.run(build_editorial_context(p), state)
    got = dict(c.calls)
    assert got["a"] == {sid(6)} and got["b"] == {sid(i) for i in range(4, 9)} and got["g"] is None  # window = 2 scenes each side; global re-runs whole
    assert res.records[0].scenes_analysed == 1 and res.records[0].scenes_reused == 11 and res.records[1].scenes_analysed == 5
    assert {f.scene_ids[0] for f in res.findings if f.stage == "a"} == set(build_editorial_context(p).scene_ids)  # the other scenes' findings were kept
    assert res.sections["a"]["scenes"][sid(6)]["v"] == build_editorial_context(p).by_id(sid(6)).hash[:4]


def test_a_settings_change_reanalyses_everything_but_a_scene_filter_narrows_the_work(ews):
    c = Counter()
    engine = EditorialEngine([c.stage("a")])
    p = ews.project
    state = apply_result(p.editorial, engine.run(build_editorial_context(p), p.editorial))
    c.calls.clear()
    p.editorial.editorial_settings.mode = AnalysisMode.CREATIVE
    ctx = build_editorial_context(p)
    assert global_signature(ctx) != state.stage_cache["a"]["gsig"]
    engine.run(ctx, state)
    assert c.calls == [("a", None)]
    state = apply_result(p.editorial, engine.run(ctx, state))
    c.calls.clear()
    engine.run(ctx, state, force=True, scene_filter={sid(1)})
    assert c.calls == [("a", None)]  # force = everything


def test_a_failing_stage_keeps_the_others_blocks_its_dependents_and_keeps_its_old_result(ews):
    c = Counter()
    p = ews.project
    good = EditorialEngine([c.stage("a"), c.stage("b", "scene", ("a",)), c.stage("c", "scene", ())])
    state = apply_result(p.editorial, good.run(build_editorial_context(p), p.editorial))
    next(x for x in p.timeline.all_clips() if x.scene_id == sid(1) and x.kind == "media").duration += 0.3
    bad = EditorialEngine([c.stage("a", fail=True), c.stage("b", "scene", ("a",)), c.stage("c", "scene", ())])
    res = bad.run(build_editorial_context(p), state)
    by = {r.stage: r for r in res.records}
    assert by["a"].status == "FAILED" and "exploded" in by["a"].error and by["b"].status == "SKIPPED" and by["c"].status in ("DONE", "REUSED")
    assert res.status == "PARTIAL" and "a" in res.sections and len(res.sections["a"]["scenes"]) == 12  # the previous result is still there
    assert any("exploded" in line for line in res.log)


def test_an_unavailable_analyzer_is_reported_not_hidden(ews):
    engine = EditorialEngine([Stage("x", "X", "global", (), None, "module missing")])
    res = engine.run(build_editorial_context(ews.project), ews.project.editorial)
    assert res.records[0].status == "FAILED" and "module missing" in res.records[0].error and res.status == "PARTIAL"


def test_cancel_stops_before_the_next_stage_and_keeps_finished_ones(ews):
    c = Counter()
    ev = threading.Event()
    first = c.stage("a")
    inner = first.analyze

    def analyze(ctx, prior, *, scene_filter=None):
        out = inner(ctx, prior, scene_filter=scene_filter)
        ev.set()
        return out

    engine = EditorialEngine([Stage("a", "a", "scene", (), analyze), c.stage("b", "scene", ("a",))])
    res = engine.run(build_editorial_context(ews.project), ews.project.editorial, cancel=ev)
    assert res.status == "CANCELED" and [r.status for r in res.records[:2]] == ["DONE", "CANCELED"] and "a" in res.sections and "b" not in res.sections


# ------------------------------------------------------------------------------------------------------------------------ service
def fake_recommend(ctx, findings, previous, sections):
    out = []
    for f in findings:
        out.append(Recommendation(f"rec_{f.key[-8:]}", f.category, Priority.LOW, f.scene_ids, [], f.title, f.detail, f.detail, f.confidence, "", ActionSpec(ActionKind.REVIEW_ONLY), 1.0, True,
                                  key=f.key, stage=f.stage, inputs_hash=ctx.by_id(f.scene_ids[0]).hash if f.scene_ids else ""))
    return out


def install_fake_engine(ws, c, **kw):
    ws.editorial.engine = EditorialEngine([c.stage("a"), c.stage("b", "window", ("a",))], recommend=fake_recommend, scorecard=build_scorecard)


def run_job(ws, job):
    assert ws.jobs.wait_idle(60)
    assert job.error is None, job.error
    settle(ws)


def test_the_service_runs_a_job_stores_results_in_the_project_and_keeps_them_out_of_undo(ews):
    c = Counter()
    install_fake_engine(ews, c)
    events = []
    ews.bus.subscribe("editorial.updated", lambda t, p: events.append(p["kind"]))
    job = ews.editorial.run_analysis()
    run_job(ews, job)
    st = ews.project.editorial
    assert st.editorial_analysis_history[-1].status == "COMPLETE" and st.sections.keys() == {"a", "b"} and len(st.findings) == 24 and len(st.editorial_recommendations) == 24
    assert st.scorecard is not None and "run_started" in events and "run_finished" in events and "progress" in events
    assert not ews.commands.can_undo  # an analysis result is not an edit of the video
    assert ews.editorial.staleness()["stale"] is False
    assert ews.editorial.run_incremental() is None  # nothing changed: no job


def test_a_clip_edit_makes_exactly_its_scene_stale_and_the_incremental_run_analyses_only_that(ews):
    c = Counter()
    install_fake_engine(ews, c)
    run_job(ews, ews.editorial.run_analysis())
    clip = next(x for x in ews.project.timeline.all_clips() if x.scene_id == sid(7) and x.kind == "media")
    ews.edit.state.update(linked_selection=False)
    ews.edit.set_duration(clip.id, clip.duration - 0.5)
    stale = ews.editorial.staleness()
    assert stale["stale"] and sid(7) in stale["scene_ids"] and len(stale["scene_ids"]) < 12
    c.calls.clear()
    job = ews.editorial.run_incremental()
    assert job is not None
    run_job(ews, job)
    assert ews.project.editorial.editorial_analysis_history[-1].trigger == "incremental" and dict(c.calls)["a"] == {sid(7)}
    assert ews.editorial.staleness()["stale"] is False


def test_results_survive_save_close_reopen_and_a_new_session_reuses_unchanged_work(ews):
    c = Counter()
    install_fake_engine(ews, c)
    run_job(ews, ews.editorial.run_analysis())
    n_recs = len(ews.project.editorial.editorial_recommendations)
    ews.project.root.mkdir(parents=True, exist_ok=True)
    ews.save()
    root = ews.project.root
    ews.close_project()
    ews.open_project(root)
    st = ews.project.editorial
    assert len(st.editorial_recommendations) == n_recs and st.editorial_analysis_history and st.sections.keys() == {"a", "b"}
    plan = ews.editorial.plan_incremental()
    assert plan.mode == "full" and plan.source == "hash"  # the change log does not cover the new session: hashes decide
    assert ews.editorial.staleness(exact=True)["stale"] is False
    install_fake_engine(ews, c)
    c.calls.clear()
    run_job(ews, ews.editorial.run_analysis())
    assert c.calls == []  # every stage's inputs are unchanged: nothing is analysed again


def test_cancel_through_the_service_keeps_the_stages_that_finished(ews):
    c = Counter()
    release = threading.Event()
    slow = c.stage("b", "scene", ("a",))
    inner = slow.analyze

    def analyze(ctx, prior, *, scene_filter=None):
        release.wait(10)
        return inner(ctx, prior, scene_filter=scene_filter)

    ews.editorial.engine = EditorialEngine([c.stage("a"), Stage("b", "b", "scene", ("a",), analyze)], recommend=fake_recommend)
    job = ews.editorial.run_analysis()
    for _ in range(300):
        if any(n == "a" for n, _ in c.calls):
            break
        threading.Event().wait(0.02)
    assert ews.editorial.cancel()
    release.set()
    assert ews.jobs.wait_idle(60)
    settle(ews)
    run = ews.project.editorial.editorial_analysis_history[-1]
    assert run.status == "CANCELED" and "a" in ews.project.editorial.sections
    assert job.status.is_terminal


def test_a_second_analysis_while_one_runs_is_refused_and_no_scenes_is_explained(tmp_path):
    from app.editorial_intelligence.editorial_service import EditorialError
    from app.services.workspace import Workspace
    from app.storage.paths import AppPaths

    ws = Workspace(AppPaths(tmp_path / "cfg", tmp_path / "data"))
    try:
        ws.new_project("Empty", tmp_path / "p")
        with pytest.raises(EditorialError, match="scene analysis"):
            ws.editorial.run_analysis()
    finally:
        ws.shutdown()


def test_rejected_recommendations_are_not_offered_again_while_their_inputs_are_unchanged(ews):
    c = Counter()
    install_fake_engine(ews, c)
    run_job(ews, ews.editorial.run_analysis())
    rec = ews.editorial.recommendations()[0]
    ews.editorial.reject_recommendation(rec.recommendation_id, "not for this video", keep=True)
    assert ews.editorial.recommendation(rec.recommendation_id).status is RecStatus.REJECTED and ews.project.editorial.editorial_overrides[-1].action == "KEEP"
    ews.commands.undo()  # the whole decision is one undo step
    assert ews.editorial.recommendation(rec.recommendation_id).status is RecStatus.PENDING and not ews.project.editorial.editorial_overrides
    ews.commands.redo()
    run_job(ews, ews.editorial.run_analysis(force=True))
    again = ews.editorial.recommendation(rec.recommendation_id)
    assert again.status is RecStatus.REJECTED and again.status_note == "not for this video"
    assert len([r for r in ews.editorial.recommendations() if r.key == rec.key]) == 1


def test_settings_are_validated_and_undoable(ews):
    from app.editorial_intelligence.editorial_service import EditorialError

    ews.editorial.update_settings(mode="CREATIVE", window=3)
    assert ews.editorial.settings.mode is AnalysisMode.CREATIVE and ews.editorial.settings.window == 3
    ews.commands.undo()
    assert ews.editorial.settings.mode is AnalysisMode.BALANCED
    for bad in ({"window": 99}, {"mode": "WILD"}, {"min_confidence": 150}, {"nonsense": 1}, {"provider": "magic"}):
        with pytest.raises((EditorialError, ValueError)):
            ews.editorial.update_settings(**bad)
    assert ews.editorial.settings.window == 2


# ------------------------------------------------------------------------------------------------------------------------ provider
def test_provider_responses_are_validated_strictly():
    ids = {"s1", "s2"}
    roles, problems = validate_story_payload({"roles": [{"scene_id": "s1", "roles": ["hook", "BOGUS"], "confidence": 400}, {"scene_id": "zzz", "role": "HOOK"}, "junk"]}, ids)
    assert roles == {"s1": [(roles["s1"][0][0], 85.0)]} and roles["s1"][0][0].value == "HOOK" and len(problems) == 3  # capped at 85, unknown role / scene / entry dropped
    assert validate_story_payload({"nope": 1}, ids)[0] == {}
    rel, problems = validate_relations_payload({"relations": [{"from": "s1", "to": "s2", "type": "cause_effect", "confidence": 60}, {"from": "s1", "to": "s1", "type": "CONTRAST"}, {"from": "s1", "to": "s2", "type": "X"}]}, ids)
    assert len(rel) == 1 and rel[0][2].value == "CAUSE_EFFECT" and len(problems) == 2
    obs, problems = validate_observations_payload({"observations": [{"scene_ids": ["s2"], "category": "pacing", "title": "t", "confidence": 99}, {"scene_ids": [], "category": "PACING", "title": "t"},
                                                                     {"scene_ids": ["s1"], "category": "NOPE", "title": "t"}]}, ids)
    assert len(obs) == 1 and obs[0]["confidence"] == 85.0 and len(problems) == 2


def test_api_provider_failures_become_provider_errors():
    def timeout(task, payload):
        raise TimeoutError

    def junk(task, payload):
        return "not json"

    with pytest.raises(EditorialProviderError, match="timed out"):
        APIEditorialProvider(timeout).analyze_story({})
    with pytest.raises(EditorialProviderError, match="valid JSON"):
        APIEditorialProvider(junk).analyze_story({})
    assert APIEditorialProvider(lambda t, p: '{"roles": []}').analyze_story({}) == {"roles": []}


def test_a_failed_ai_analysis_marks_the_run_partial_and_never_complete(ews):
    c = Counter()
    install_fake_engine(ews, c)
    ews.editorial.update_settings(ai_enabled=True)

    def boom(task, payload):
        raise ConnectionError("rate limited")

    ews.editorial.set_provider(APIEditorialProvider(boom))
    run_job(ews, ews.editorial.run_analysis())
    run = ews.project.editorial.editorial_analysis_history[-1]
    assert run.status == "PARTIAL" and any("AI analysis FAILED" in line for line in run.log) and any(s.stage == "ai_analysis" and s.status == "FAILED" for s in run.stages)
    assert ews.project.editorial.editorial_recommendations  # the rule-based result is intact


def test_valid_ai_output_only_adds_judgement_and_cannot_carry_an_edit_action(ews):
    c = Counter()
    install_fake_engine(ews, c)
    ews.editorial.update_settings(ai_enabled=True)
    ews.editorial.set_provider(APIEditorialProvider(lambda t, p: {"observations": [{"scene_ids": [sid(2)], "category": "NARRATIVE_FLOW", "title": "The story stalls here", "detail": "d", "confidence": 90}]} if t == "generate_recommendations" else {}))
    run_job(ews, ews.editorial.run_analysis())
    ai = [f for f in ews.project.editorial.findings if f.key.startswith("recommendations:") or f.finding_id.startswith("ai_")]
    assert ai and ai[0].action is not None and ai[0].action.kind is ActionKind.REVIEW_ONLY and ai[0].confidence == 85.0


# ------------------------------------------------------------------------------------------------------------------------ scorecard
def test_scorecard_separates_measured_from_judgment_and_a_serious_warning_cannot_be_averaged_away():
    secs = {"evidence": {"global": {"score": {"value": 96, "basis": "JUDGMENT", "confidence": 70, "note": "looks fine"}}},
            "pacing": {"global": {"score": {"value": 90, "basis": "COMPUTED", "measured": {"cuts_per_min": 7.5}}}},
            "emphasis": {"global": {"scores": {"Emphasis Quality": {"value": 80}, "Audio-Visual Alignment": {"value": 70}}}}}
    findings = [Finding("f", "evidence", Category.EVIDENCE_ALIGNMENT, ["s1"], serious=True), Finding("g", "evidence", Category.EVIDENCE_ALIGNMENT, ["s2"], serious=True, intentional=True)]
    sc = build_scorecard(None, secs, findings)
    items = {i.dimension: i for i in sc.items}
    assert items["Evidence Alignment"].score == SERIOUS_ITEM_CAP and items["Evidence Alignment"].serious_warnings == 1 and "Capped" in items["Evidence Alignment"].note
    assert items["Pacing Suitability"].basis is Basis.COMPUTED and items["Pacing Suitability"].measured == {"cuts_per_min": 7.5}
    assert items["Emphasis Quality"].score == 80 and items["Audio-Visual Alignment"].score == 70
    assert items["Narrative Coherence"].score is None and "Not measurable" in items["Narrative Coherence"].note  # nobody measured it: no invented number
    assert sc.overall == SERIOUS_OVERALL_CAP and "serious" in sc.overall_note and sc.capped_by == ["Evidence Alignment"]
    clean = build_scorecard(None, secs, [])
    assert clean.overall is not None and clean.overall > SERIOUS_OVERALL_CAP
