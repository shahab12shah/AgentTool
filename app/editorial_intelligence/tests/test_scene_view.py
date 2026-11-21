"""The scene inspector data and the sequence map: structured, concise, honest about what was not analysed, and read-only."""

from __future__ import annotations

import json

import pytest

from app.editorial_intelligence.tests.scenarios import V, craft
from app.tests.conftest import needs_ffmpeg
from app.editorial_intelligence.editorial_models import (
    AnalysisRun,
    ActionKind,
    ActionSpec,
    Category,
    NarrativeRole,
    Priority,
    Recommendation,
    RecStatus,
    RoleScore,
    SceneRole,
    StorySection,
    StoryStructure,
)
from app.editorial_intelligence.scene_view import build_scene_view, build_sequence_map


pytestmark = needs_ffmpeg

NARRATION = ("The police had already searched the house three times. Nothing was found in 1981. Then a detective noticed something in the official report. "
             "The report listed a second key. That key opened the garage. Inside the garage was the evidence nobody expected.")


@pytest.fixture
def cws(ws, tmp_path):
    craft(ws, tmp_path, NARRATION, {0: V("house", "Exterior of the house"), 1: V("house", "Exterior of the house"), 2: V("report", "Police report", source="SCREENSHOT", kind="IMAGE", evidence="EVIDENCE"),
                                    3: V("key", "A brass key")}, min_scene=1.0, threshold=0.15)
    assert len(ws.project.scenes) >= 4
    return ws


def test_the_inspector_view_before_any_analysis_shows_project_facts_and_says_nothing_was_analysed(cws):
    S = [x.id for x in cws.project.scenes]
    v = build_scene_view(cws.project, S[2])
    assert v["exists"] and v["analysed"] is False and v["role"] is None and v["narration"] and v["visual"]["assigned"] and v["duration"]["current"] > 0
    assert v["previous"]["scene_id"] == S[1] and v["next"]["scene_id"] == S[3] and v["recommendations"] == []
    assert not build_scene_view(cws.project, "nope")["exists"]
    json.dumps(v)


def test_the_inspector_view_includes_the_analysis_overrides_locks_and_open_recommendations(cws):
    p = cws.project
    S = [x.id for x in p.scenes]
    st = p.editorial
    st.editorial_analysis_history.append(AnalysisRun("r1"))
    st.scene_roles[S[2]] = SceneRole(S[2], [RoleScore(NarrativeRole.EVIDENCE, 80), RoleScore(NarrativeRole.EXPLANATION, 55)], ["report"], False)
    st.sections["relationships"] = {"global": {"by_scene": {S[2]: {"previous": "CONTINUATION", "next": "CLAIM_EVIDENCE", "related": [S[0]]}}}}
    st.sections["pacing"] = {"scenes": {S[2]: {"recommended": {"min": 3.0, "target": 5.0, "max": 8.0}, "verdict": "hold_longer"}}}
    st.editorial_recommendations += [Recommendation("r_open", Category.PACING, Priority.HIGH, [S[2]], title="Hold longer", status=RecStatus.PENDING, suggested_action=ActionSpec(ActionKind.SET_SHOT_DURATION)),
                                     Recommendation("r_done", Category.PACING, Priority.LOW, [S[2]], title="x", status=RecStatus.APPLIED)]
    clip = next(c for c in p.timeline.all_clips() if c.scene_id == S[2] and c.kind == "media")
    clip.locks = frozenset({"timing"})
    v = build_scene_view(p, S[2])
    assert v["analysed"] and v["role"]["roles"][0] == {"role": "EVIDENCE", "confidence": 80} and v["next"]["relation"] == "CLAIM_EVIDENCE" and v["related"] == [S[0]]
    assert v["duration"]["recommended"]["target"] == 5.0 and v["duration"]["verdict"] == "hold_longer"
    assert [r["recommendation_id"] for r in v["recommendations"]] == ["r_open"] and "timing" in v["locks"] and "TIMING_OVERRIDE" in v["overrides"]
    json.dumps(v)


def test_the_sequence_map_lists_sections_scenes_repeats_and_changes_nothing(cws):
    p = cws.project
    S = [x.id for x in p.scenes]
    st = p.editorial
    st.narrative_structure = StoryStructure("INVESTIGATIVE", 70.0, sections=[StorySection("sec1", "Opening", [S[0], S[1]], 0.0, 12.0, "HOOK", 60.0)])
    a1, a2 = p.visual_assignments[S[0]], p.visual_assignments[S[1]]
    assert a1.asset_id == a2.asset_id  # the same visual twice (the scenario attaches the same asset to scenes 1 and 2)
    before = json.dumps(p.to_document(), sort_keys=True)
    m = build_sequence_map(p)
    assert m["sections"][0]["title"] == "Opening" and len(m["scenes"]) == len(S) and m["model"] == "INVESTIGATIVE"
    by = {s["scene_id"]: s for s in m["scenes"]}
    assert by[S[0]]["repeated"] and by[S[0]]["repeated_with"] == [S[1]] and not by[S[3]]["repeated"]
    assert json.dumps(p.to_document(), sort_keys=True) == before and "never rearranges" in m["note"]
