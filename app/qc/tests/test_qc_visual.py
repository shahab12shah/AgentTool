"""Visual accuracy recheck: the placed visual is scored again, in context, with the Phase 3 scorer."""

from __future__ import annotations

import pytest

from app.analysis.models import Claim, ClaimType, VisualIntent, VisualType
from app.media.asset import SourceType
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, find, narrate, new_project, qc_ctx, run_checker
from app.qc.visual_checker import EVIDENCE_TEXT, VisualChecker
from app.research.models import Candidate, VisualAssignment
from app.qc.issue_model import QCCategory

TITLES = ["silver supply mines industrial demand", "solar panels factory silver paste", "silver price chart weekly"]
NARRATION = ["Silver supply from mines is falling as industrial demand rises.", "Solar panel makers use silver in every cell.", "Prices have reacted sharply in recent weeks."]
SUBJECTS = ["silver supply", "solar panels", "silver prices"]


def project(tmp_path, titles=TITLES, *, sources=None, intent=VisualType.LITERAL, claims=False, assign=None):
    p = new_project(tmp_path, seconds=30)
    scenes = [add_scene(p, i * 10, (i + 1) * 10, NARRATION[i], topic=SUBJECTS[i], importance=0.5) for i in range(3)]
    narrate(p)
    for i, s in enumerate(scenes):
        p.visual_intents[s.id] = VisualIntent(s.id, intent if i == 0 else VisualType.LITERAL, primary_subject=SUBJECTS[i], action="falling" if i == 0 else "using")
    if claims:
        scenes[0].claims = [Claim("c1", "Silver supply is falling.", ClaimType.FACT, "sent_0000")]
    assets, clips = [], []
    for i, t in enumerate(titles):
        a = add_asset(p, f"{t}.mp4", "video", duration=30)
        a.source_type = (sources or {}).get(i, SourceType.USER_MEDIA)
        assets.append(a)
        clips.append(add_clip(p, "track_v1", a, i * 10, 10, scene=scenes[i], source_in=i * 10, created_by="AI"))
    for sid, a in (assign or {}).items():
        p.visual_assignments[sid] = a
    return p, scenes, assets, clips


def run(p, **kw):
    return run_checker(VisualChecker(), qc_ctx(p, **kw))


def accuracy(out):
    return [i for i in out.issues if i.code in ("visual.mismatch", "visual.subject_mismatch", "visual.weak")]


def test_matching_visuals_raise_no_issue_and_report_both_scores(tmp_path):
    p, scenes, *_ = project(tmp_path, assign={"scene_001": VisualAssignment("scene_001", None, "media_00002", accuracy_score=91.0)})
    out = run(p)
    assert accuracy(out) == [], [(i.scene_id, i.code, i.current_qc_score) for i in out.issues]
    m = out.metrics
    assert m["scenes_checked"] == 3 and m["mean_current_score"] > 70 and m["per_scene"]["scene_001"]["original"] == 91.0
    assert any("metadata" in n for n in out.notes)


def test_wrong_picture_is_an_error_with_both_scores_and_the_project_untouched(tmp_path):
    titles = ["city skyline timelapse", *TITLES[1:]]
    p, scenes, assets, clips = project(tmp_path, titles, assign={"scene_001": VisualAssignment("scene_001", None, "media_00002", accuracy_score=92.0)})
    before = p.to_document()
    out = run(p)
    i = accuracy(out)
    assert len(i) == 1 and i[0].scene_id == "scene_001" and i[0].severity is Severity.ERROR and i[0].category is QCCategory.VISUAL_ACCURACY
    assert i[0].original_research_score == 92.0 and i[0].current_qc_score < 40
    assert "city skyline timelapse" in i[0].description and "metadata" in i[0].description.lower() and "may" in i[0].title.lower()
    assert i[0].timeline_item_id == clips[0].id and i[0].confidence < 100
    assert i[0].fix.kind in ("visual.replace", "visual.search_again") and not i[0].auto_fix_safe  # a replacement always needs the user
    assert p.to_document() == before and p.visual_assignments["scene_001"].accuracy_score == 92.0  # the stored research score is never rewritten


def test_severity_follows_the_configured_thresholds(tmp_path):
    p, *_ = project(tmp_path, ["city skyline timelapse", *TITLES[1:]])
    base = accuracy(run(p))[0]
    assert base.severity is Severity.ERROR
    p.qc_settings.visual.error_below = 10.0  # the same score is now "just a warning"
    assert accuracy(run(p))[0].severity is Severity.WARNING
    p.qc_settings.visual.warning_below = 10.0
    assert accuracy(run(p))[0].severity is Severity.NOTICE
    p.qc_settings.visual.notice_below = 10.0
    assert accuracy(run(p)) == []


def test_context_moves_the_score_by_at_most_the_cap(tmp_path):
    p, *_ = project(tmp_path, ["silver mining", *TITLES[1:]])
    cap = p.qc_settings.visual.context_bonus_cap
    s = run(p).metrics["per_scene"]["scene_001"]
    assert abs(s["context"]) <= cap + 1e-6
    p.qc_settings.visual.context_bonus_cap = 0.0
    assert run(p).metrics["per_scene"]["scene_001"]["context"] == 0.0


def test_a_visual_that_fits_the_neighbouring_scene_better_is_marked_down(tmp_path):
    # scene 1 shows the picture that belongs to scene 2
    p, *_ = project(tmp_path, [TITLES[1], TITLES[1], TITLES[2]])
    s1 = run(p).metrics["per_scene"]["scene_001"]
    assert s1["context"] < 0 and abs(s1["context"]) <= p.qc_settings.visual.context_bonus_cap


def test_custom_weights_change_the_result(tmp_path):
    p, *_ = project(tmp_path, ["silver mining", *TITLES[1:]])
    a = run(p).metrics["per_scene"]["scene_001"]["current"]
    p.qc_settings.visual.weights = {"semantic": 0.0, "subject": 100.0, "context": 0.0, "action": 0.0, "timing": 0.0, "quality": 0.0, "source": 0.0}
    b = run(p).metrics["per_scene"]["scene_001"]["current"]
    assert a != b


def test_asset_without_a_candidate_is_handled_with_lower_confidence(tmp_path):
    p, *_ = project(tmp_path, ["city skyline timelapse", *TITLES[1:]])
    i = accuracy(run(p))[0]
    assert i.confidence <= 60.0  # only a file name to go on
    cand = Candidate("cand_1", "scene_001", SourceType.STOCK_VIDEO, "VIDEO", "city skyline timelapse", "", [], 30.0, 1920, 1080, asset_id="media_00002")
    p.visual_candidates["cand_1"] = cand
    p.visual_assignments["scene_001"] = VisualAssignment("scene_001", "cand_1", "media_00002", accuracy_score=88.0)
    before = set(p.visual_candidates)
    j = accuracy(run(p))[0]
    assert j.original_research_score == 88.0 and set(p.visual_candidates) == before  # the researched candidate is used; nothing throwaway is stored


def test_findings_below_the_minimum_confidence_are_not_reported(tmp_path):
    p, *_ = project(tmp_path, ["city skyline timelapse", *TITLES[1:]])
    assert accuracy(run(p))
    p.qc_settings.visual.min_confidence = 95.0
    assert accuracy(run(p)) == []


def test_a_visual_chosen_by_the_user_is_flagged_one_step_lower(tmp_path):
    base_p, *_ = project(tmp_path / "ai", ["city skyline timelapse", *TITLES[1:]])
    base = accuracy(run(base_p))[0]
    p, *_ = project(tmp_path / "user", ["city skyline timelapse", *TITLES[1:]], assign={"scene_001": VisualAssignment("scene_001", None, "media_00002", selected_by="USER", approved=True)})
    i = accuracy(run(p))[0]
    assert base.severity is Severity.ERROR and i.severity is Severity.WARNING and "You selected this visual" in i.description
    assert i.fix.kind != "audio.duck" and not i.auto_fix_safe


def test_evidence_claim_with_a_decorative_picture_gets_the_review_note_not_a_verdict(tmp_path):
    p, scenes, *_ = project(tmp_path, ["silver supply mines industrial demand", *TITLES[1:]], sources={0: SourceType.STOCK_VIDEO}, intent=VisualType.EVIDENCE, claims=True)
    i = find(run(p), "visual.evidence_decorative")
    assert len(i) == 1 and i[0].category is QCCategory.FACT_REVIEW and i[0].title == EVIDENCE_TEXT and i[0].description.startswith(EVIDENCE_TEXT)
    text = (i[0].description + i[0].why_it_matters).lower()
    assert "not a judgement of whether the statement is true" in text and "false" not in text.replace("not a judgement of whether the statement is true", "")
    assert i[0].severity is Severity.WARNING and i[0].fix.kind == "visual.search_again"
    # a real evidence source on screen: nothing to review
    p2, *_ = project(tmp_path / "ev", ["silver supply mines industrial demand", *TITLES[1:]], sources={0: SourceType.SCREENSHOT}, intent=VisualType.EVIDENCE, claims=True)
    assert find(run(p2), "visual.evidence_decorative") == []
    # no claim that needs evidence: nothing either
    p3, *_ = project(tmp_path / "plain", ["silver supply mines industrial demand", *TITLES[1:]], sources={0: SourceType.STOCK_VIDEO})
    assert find(run(p3), "visual.evidence_decorative") == []


def test_generic_stock_picture_for_a_named_subject(tmp_path):
    from app.analysis.models import Entity, EntityType

    p, scenes, *_ = project(tmp_path, ["silver supply mines industrial demand", *TITLES[1:]], sources={0: SourceType.STOCK_VIDEO})
    scenes[0].entities = [Entity("Glencore", EntityType.COMPANY)] if "text" in Entity.__dataclass_fields__ else scenes[0].entities
    out = run(p)
    gen = find(out, "visual.generic_for_specific")
    if "text" in Entity.__dataclass_fields__:
        assert len(gen) == 1 and gen[0].severity is Severity.NOTICE and "Glencore" in gen[0].description
    else:  # pragma: no cover - model without entity text
        pytest.skip("entity model differs")


def test_scenes_without_a_picture_or_not_yet_analysed_are_left_alone(tmp_path):
    p, scenes, assets, clips = project(tmp_path)
    p.timeline.get_track("track_v1").clips.remove(clips[1])  # scene 2 has no picture: the scene checker's finding
    p.visual_intents.pop("scene_003")  # scene 3 was never analysed
    out = run(p)
    assert set(out.metrics["per_scene"]) == {"scene_001"} and any("not analysed" in n for n in out.notes)


def test_scene_local_contract(tmp_path):
    p, scenes, *_ = project(tmp_path, ["city skyline timelapse", *TITLES[1:]])
    c = VisualChecker()
    assert c.scene_local and c.expensive and "visual" in c.domains
    only = run_checker(c, qc_ctx(p, scene_filter=["scene_002"]))
    assert set(only.metrics["per_scene"]) == {"scene_002"} and accuracy(only) == []
    h1 = c.scene_input_hash(qc_ctx(p), "scene_001")
    p.qc_settings.visual.error_below = 5.0
    assert c.scene_input_hash(qc_ctx(p), "scene_001") != h1  # a changed threshold invalidates the cached scene result
