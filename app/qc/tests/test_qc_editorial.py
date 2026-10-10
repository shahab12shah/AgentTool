"""AI editorial review: provider abstraction, local rule-based reviewer, strict API validation."""

from __future__ import annotations

import json

import pytest

from app.qc.ai_editorial_checker import (
    QUESTIONS, AIEditorialQCProvider, APIEditorialQCProvider, EditorialChecker, EditorialFinding, EditorialReview, FutureEditorialQCProvider, LocalAIEditorialQCProvider, ProviderError,
    ProviderUnavailable, build_request, parse_review, provider_for)
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.issue_model import QCCategory
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, find, narrate, new_project, qc_ctx, run_checker
from app.timeline.keyframes import Keyframe

TEXT = "Silver prices rose sharply last week after the central bank statement surprised the market."


class _F(BaseChecker):
    id = "x"


def project(tmp_path, n=3):
    p = new_project(tmp_path, seconds=10.0 * n)
    scenes = [add_scene(p, i * 10, (i + 1) * 10, TEXT if i == 0 else f"Scene {i} explains another part of the silver market in detail.", importance=0.5) for i in range(n)]
    narrate(p)
    a = add_asset(p, "silver market footage.mp4", "video", duration=120)
    for i, s in enumerate(scenes):
        add_clip(p, "track_v1", a, i * 10, 10, scene=s, source_in=i * 30.0, created_by="AI")
    return p, scenes


def issue(code, scene, sev=Severity.WARNING, conf=90.0, cat=QCCategory.VISUAL_ACCURACY):
    return _F().issue(code, cat, sev, code, description="x", scene_id=scene, confidence=conf)


def shared(*issues, **metrics):
    by: dict[str, CheckerOutput] = {}
    for i in issues:
        by.setdefault(i.code.split(".")[0], CheckerOutput()).issues.append(i)
    for k, m in metrics.items():
        by.setdefault(k, CheckerOutput()).metrics.update(m)
    return by


def ctx_with(p, sh=None, **kw):
    ctx = qc_ctx(p, **kw)
    ctx.shared.update(sh or {})
    return ctx


def run(p, sh=None, **kw):
    return run_checker(EditorialChecker(), ctx_with(p, sh, **kw))


# ---------------------------------------------------------------- local reviewer
def test_clean_project_gets_positive_answers_and_no_findings(tmp_path):
    p, _ = project(tmp_path)
    out = run(p)
    assert out.issues == [] and out.metrics["provider"] == "local" and set(out.metrics["answers"]) == set(QUESTIONS)
    assert all(a["verdict"] in ("yes", "mostly") for a in out.metrics["answers"].values())
    assert out.metrics["weak_scenes"] == [] and out.metrics["request_size"] > 0 and "rule-based" in out.metrics["provider_label"].lower()


def test_problem_scene_gets_concise_cross_finding_judgements(tmp_path):
    p, scenes = project(tmp_path)
    s2 = scenes[1].id
    sh = shared(issue("visual.mismatch", s2, Severity.ERROR, 80), issue("caption.overflow", s2, Severity.WARNING, 95, QCCategory.CAPTION), issue("pacing.too_fast", s2, Severity.WARNING, 75, QCCategory.PACING),
                issue("visual.repetition", scenes[2].id, Severity.WARNING, 70, QCCategory.VISUAL_REPETITION), continuity={"repetition_score": 60.0}, visual={"mean_current_score": 52.0})
    out = run(p, sh)
    a = out.metrics["answers"]
    assert a["q1"]["verdict"] in ("partly", "no") and a["q4"]["verdict"] in ("partly", "no") and a["q2"]["verdict"] != "yes" and a["q9"]["verdict"] != "yes"
    weak = find(out, "editorial.weak_section")
    assert len(weak) == 1 and weak[0].scene_id == s2 and weak[0].category is QCCategory.EDITORIAL and weak[0].detection_source == "ai:local"
    assert "Potential weak section" in weak[0].description and "Review recommended" in weak[0].description and weak[0].confidence < 100
    assert weak[0].severity in (Severity.NOTICE, Severity.WARNING) and weak[0].score_group in ("visual_accuracy", "captions", "pacing")
    for ans in a.values():  # concise decision factors, never reasoning
        assert len(ans["reason"]) < 260 and ans["reason"].count(".") <= 4 and not any(w in ans["reason"].lower() for w in ("let me", "step by step", "first,", "i think"))
    assert all(len(i.description) < 330 for i in out.issues)


def test_confusing_scene_with_a_number_and_conflicting_signals(tmp_path):
    from app.analysis.models import NumberKind, NumericMention

    p, scenes = project(tmp_path)
    scenes[0].numbers = [NumericMention("5 percent", NumberKind.PERCENTAGE)]
    sh = shared(issue("visual.mismatch", scenes[0].id, Severity.WARNING, 80), issue("sync.caption_early", scenes[0].id, Severity.WARNING, 90, QCCategory.SYNC))
    out = run(p, sh)
    i = find(out, "editorial.confusing")
    assert len(i) == 1 and "Potential confusion" in i[0].description and out.metrics["answers"]["q10"]["verdict"] != "yes"


def test_distracting_scene_combines_motion_transitions_and_graphics(tmp_path):
    p, scenes = project(tmp_path)
    c1 = p.timeline.get_track("track_v1").clips[1]  # scene 2: four quick shots, a move, a wipe and three graphics
    a = p.assets.get(c1.asset_id)
    c1.duration = 2.5
    c1.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 2.0, 1.2)]
    c1.transition = {"type": "WIPE", "duration": 0.5}
    for k, t in enumerate((12.5, 15.0, 17.5)):
        add_clip(p, "track_v1", a, t, 2.5, scene=scenes[1], source_in=70.0 + k * 20, created_by="AI")
    for k in range(3):
        add_clip(p, "track_v5", None, 11.0 + k * 2.5, 1.0, kind="text", text={"text": f"g{k}"})
    out = run(p)
    d = find(out, "editorial.distracting")
    assert len(d) == 1 and "Potential distraction" in d[0].description and out.metrics["answers"]["q5"]["verdict"] != "yes"


def test_the_picture_that_lags_behind_the_narration(tmp_path):
    p = new_project(tmp_path, seconds=20)
    s1 = add_scene(p, 0, 10, "Silver mining output keeps falling at the big mines.", topic="silver mining output")
    s2 = add_scene(p, 10, 20, "The central bank raised interest rates on Tuesday morning.", topic="central bank interest rates")
    narrate(p)
    a = add_asset(p, "silver mining output footage.mp4", "video", duration=120)
    b = add_asset(p, "central bank building.mp4", "video", duration=120)
    add_clip(p, "track_v1", a, 0, 14, scene=s1)  # the mining picture carries on 4 s into the next scene
    add_clip(p, "track_v1", b, 14, 6, scene=s2)
    out = run(p)
    i = find(out, "editorial.topic_lag")
    assert len(i) == 1 and i[0].scene_id == s2.id and "stays on the previous subject" in i[0].title and "4.0 seconds" in i[0].description and i[0].score_group == "sync"
    sh = shared(issue("sync.visual_late", s2.id, Severity.WARNING, 90, QCCategory.SYNC))  # already reported by a deterministic checker
    assert find(run(p, sh), "editorial.topic_lag") == []


def test_local_provider_is_deterministic_and_json_safe(tmp_path):
    p, scenes = project(tmp_path)
    sh = shared(issue("visual.mismatch", scenes[1].id, Severity.ERROR, 80), issue("caption.overflow", scenes[1].id, Severity.WARNING, 95, QCCategory.CAPTION), issue("pacing.too_fast", scenes[1].id, Severity.WARNING, 75, QCCategory.PACING))
    req = build_request(ctx_with(p, sh))
    json.dumps(req.to_dict())
    assert str(p.root) not in json.dumps(req.to_dict())  # no paths in what a reviewer sees
    prov = LocalAIEditorialQCProvider()
    assert prov.review(req).to_dict() == prov.review(req).to_dict()


# ---------------------------------------------------------------- provider abstraction
class FakeProvider(AIEditorialQCProvider):
    name, label = "fake", "Fake reviewer"

    def __init__(self, findings=(), ok=True):
        self.findings, self.ok, self.seen = list(findings), ok, None

    def is_available(self):
        return (self.ok, "" if self.ok else "offline")

    def review(self, request):
        self.seen = request
        return EditorialReview(self.name, {q: {"verdict": "yes", "confidence": 80, "reason": "Fine."} for q in QUESTIONS}, self.findings)


def test_a_custom_provider_plugs_in_through_the_context(tmp_path):
    p, scenes = project(tmp_path)
    fake = FakeProvider([EditorialFinding("odd_cut", "Odd cut", "Potential odd cut. Review recommended.", "WARNING", 85.0, scenes[1].id, group_hint="pacing")])
    out = run_checker(EditorialChecker(), _with_provider(p, fake))
    i = find(out, "editorial.odd_cut")
    assert len(i) == 1 and i[0].detection_source == "ai:fake" and i[0].score_group == "pacing" and out.metrics["provider"] == "fake" and fake.seen.scenes


def _with_provider(p, prov, sh=None):
    ctx = ctx_with(p, sh)
    ctx.ai_provider = prov
    return ctx


def test_findings_never_exceed_warning_and_low_confidence_is_capped_or_hidden(tmp_path):
    p, scenes = project(tmp_path)
    sid = scenes[0].id
    fake = FakeProvider([EditorialFinding("a_error", "Claimed error", "Potential problem.", "ERROR", 95.0, sid), EditorialFinding("b_low", "Low confidence", "Potential problem.", "WARNING", 40.0, sid),
                         EditorialFinding("c_hidden", "Hidden", "Potential problem.", "WARNING", 20.0, sid), EditorialFinding("d_mid", "Mid", "Potential problem.", "WARNING", 60.0, sid)])
    out = run_checker(EditorialChecker(), _with_provider(p, fake))
    by = {i.code: i for i in out.issues}
    assert by["editorial.a_error"].severity is Severity.WARNING  # an AI judgement is never an ERROR
    assert by["editorial.b_low"].severity is Severity.NOTICE  # confidence below 50: at most a notice
    assert by["editorial.d_mid"].severity is Severity.WARNING and "editorial.c_hidden" not in by  # below min_confidence_to_report (35): not shown


def test_findings_repeating_a_deterministic_finding_are_suppressed(tmp_path):
    p, scenes = project(tmp_path)
    sid = scenes[1].id
    fake = FakeProvider([EditorialFinding("pace_again", "Pace", "Potential pacing issue.", "WARNING", 80.0, sid, group_hint="pacing"),
                         EditorialFinding("other", "Other scene", "Potential issue.", "WARNING", 80.0, scenes[2].id, group_hint="pacing")])
    sh = shared(issue("pacing.too_fast", sid, Severity.WARNING, 90, QCCategory.PACING))
    out = run_checker(EditorialChecker(), _with_provider(p, fake, sh))
    assert [i.code for i in out.issues] == ["editorial.other"] and any("repeat" in n for n in out.notes)


def test_provider_failure_is_not_swallowed(tmp_path):
    class Boom(FakeProvider):
        def review(self, request):
            raise RuntimeError("model exploded")

    p, _ = project(tmp_path)
    with pytest.raises(RuntimeError):
        run_checker(EditorialChecker(), _with_provider(p, Boom()))


def test_unavailable_provider_falls_back_to_the_local_one_with_a_note(tmp_path):
    p, _ = project(tmp_path)
    out = run_checker(EditorialChecker(), _with_provider(p, FakeProvider(ok=False)))
    assert out.metrics["provider"] == "local" and any("unavailable (offline)" in n for n in out.notes)
    p.qc_settings.ai_provider = "api"  # no client configured
    out2 = run(p)
    assert out2.metrics["provider"] == "local" and any("No AI client" in n for n in out2.notes)
    p.qc_settings.ai_provider = "future"
    assert run(p).metrics["provider"] == "local"
    p.qc_settings.ai_provider = "does-not-exist"
    assert any("unknown" in n for n in run(p).notes)


def test_switching_the_review_off(tmp_path):
    p, scenes = project(tmp_path)
    p.qc_settings.ai_review_enabled = False
    out = run(p, shared(issue("visual.mismatch", scenes[0].id)))
    assert out.issues == [] and out.metrics["enabled"] is False and any("switched off" in n for n in out.notes)


def test_future_provider_reports_unavailable():
    f = FutureEditorialQCProvider()
    assert f.is_available()[0] is False
    with pytest.raises(ProviderUnavailable):
        f.review(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- API provider (fake client, strict validation)
def good_answer(scene_id):
    return json.dumps({"answers": {q: {"verdict": "mostly", "confidence": 70, "reason": "Looks fine overall. Because of many details."} for q in QUESTIONS},
                       "findings": [{"code": "weak_intro", "scene_id": scene_id, "severity": "WARNING", "confidence": 72, "title": "Weak intro", "reason": "The opening shot is generic.", "suggested_fix": "Use a sharper opener.",
                                     "group": "pacing"}]})


def test_api_provider_valid_json_becomes_issues(tmp_path):
    p, scenes = project(tmp_path)
    calls = []
    prov = APIEditorialQCProvider(lambda prompt: calls.append(prompt) or good_answer(scenes[0].id), name="mock-api")
    out = run_checker(EditorialChecker(), _with_provider(p, prov))
    i = find(out, "editorial.weak_intro")
    assert len(i) == 1 and i[0].detection_source == "ai:mock-api" and i[0].score_group == "pacing" and i[0].severity is Severity.WARNING
    assert out.metrics["answers"]["q1"]["reason"] == "Looks fine overall." and "Questions:" in calls[0] and "DATA:" in calls[0] and "do not judge whether any statement is true" in calls[0]


def test_api_provider_drops_malformed_parts_and_never_crashes(tmp_path):
    p, scenes = project(tmp_path)
    bad = json.dumps({"answers": {"q1": {"verdict": "maybe", "confidence": 70, "reason": "x"}, "q2": {"verdict": "yes", "confidence": "high", "reason": "x"}, "q99": {"verdict": "yes", "confidence": 5, "reason": "x"},
                                  "q3": {"verdict": "yes", "confidence": 250, "reason": "A" * 600}},
                      "findings": [{"code": "BAD CODE", "title": "t", "reason": "r", "confidence": 50}, {"code": "no_title", "reason": "r", "confidence": 50}, {"code": "ghost", "title": "t", "reason": "r.", "confidence": 60, "scene_id": "scene_999"},
                                   {"code": "good_one", "title": "Fine", "reason": "r.", "confidence": 60, "severity": "CRITICAL", "extra_field": {"hidden": "reasoning"}}, "not a dict", 7]})
    prov = APIEditorialQCProvider(lambda prompt: "Sure! Here you go:\n" + bad)
    out = run_checker(EditorialChecker(), _with_provider(p, prov))
    assert set(out.metrics["answers"]) == {"q3"} and out.metrics["answers"]["q3"]["confidence"] == 100.0 and len(out.metrics["answers"]["q3"]["reason"]) <= 240
    assert [i.code for i in out.issues] == ["editorial.good_one"] and out.issues[0].severity is Severity.NOTICE  # CRITICAL is not an allowed AI severity
    assert any("ignored" in n for n in out.notes)


def test_api_answer_with_values_of_the_wrong_type_is_dropped_not_fatal():
    """A scene id that is a list or a record used to raise TypeError (unhashable) and fail the whole checker; infinity used to pass as a confidence of 100."""
    from app.qc.ai_editorial_checker import EditorialRequest

    req = EditorialRequest(scenes=[{"id": "scene_001", "label": "1"}])
    raw = ('{"answers": {"q1": {"verdict": "yes", "confidence": Infinity, "reason": "x"}}, "findings": ['
           '{"code": "a_list", "title": "t", "reason": "r.", "confidence": 60, "scene_id": ["scene_001"]}, '
           '{"code": "a_dict", "title": "t", "reason": "r.", "confidence": 60, "scene_id": {"id": "scene_001"}}, '
           '{"code": "infinite", "title": "t", "reason": "r.", "confidence": Infinity}, '
           '{"code": "fine_one", "title": "t", "reason": "r.", "confidence": 60, "scene_id": "scene_001"}]}')
    rev = parse_review("x", raw, req)
    assert [f.code for f in rev.findings] == ["fine_one"] and rev.answers == {} and any("ignored" in n for n in rev.notes)
    for not_text in (None, {"findings": []}, b"{}"):  # a client that hands back something else than text is an unusable answer, not a crash
        with pytest.raises(ProviderError):
            parse_review("x", not_text, req)  # type: ignore[arg-type]


def test_api_provider_limits_the_number_of_findings():
    from app.qc.ai_editorial_checker import EditorialRequest, MAX_FINDINGS

    many = json.dumps({"answers": {}, "findings": [{"code": f"f_{k:03d}", "title": "t", "reason": "r.", "confidence": 60} for k in range(60)]})
    rev = parse_review("x", many, EditorialRequest())
    assert len(rev.findings) == MAX_FINDINGS and any("ignored" in n for n in rev.notes)


def test_api_provider_error_paths(tmp_path):
    p, _ = project(tmp_path)
    with pytest.raises(ProviderError):
        run_checker(EditorialChecker(), _with_provider(p, APIEditorialQCProvider(lambda prompt: "I cannot do that.")))
    def offline(prompt):
        raise ConnectionError("no route")

    with pytest.raises(ProviderUnavailable):
        run_checker(EditorialChecker(), _with_provider(p, APIEditorialQCProvider(offline)))
    assert APIEditorialQCProvider().is_available()[0] is False
    with pytest.raises(ProviderUnavailable):
        APIEditorialQCProvider().review(build_request(qc_ctx(p)))


def test_prompt_is_bounded(tmp_path):
    p, scenes = project(tmp_path, n=3)
    req = build_request(ctx_with(p))
    req.findings = [{"checker": "x", "code": "a.b", "scene_id": scenes[0].id, "severity": "WARNING", "confidence": 90.0, "title": "t" * 80, "start": 1.0} for _ in range(400)]
    assert len(APIEditorialQCProvider(lambda s: "{}").build_prompt(req)) < 30000


def test_provider_for_prefers_the_services_choice(tmp_path):
    p, _ = project(tmp_path)
    ctx = qc_ctx(p)
    assert provider_for(p.qc_settings, ctx)[0].name == "local"
    ctx.ai_provider = FakeProvider()
    assert provider_for(p.qc_settings, ctx)[0].name == "fake"


def test_contract_and_cache_key(tmp_path):
    p, _ = project(tmp_path)
    c = EditorialChecker()
    assert c.uses_shared and c.expensive and not c.scene_local and "visual" in c.domains
    before = p.to_document()
    h = c.input_hash(qc_ctx(p))
    run(p)
    assert p.to_document() == before
    p.qc_settings.ai_provider = "api"
    assert c.input_hash(qc_ctx(p)) != h
    ctx = qc_ctx(p)
    ctx.ai_provider = FakeProvider()
    assert c.input_hash(ctx) != c.input_hash(qc_ctx(p))
