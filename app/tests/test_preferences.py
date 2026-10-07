from __future__ import annotations

import pytest

from app.analysis.models import Scene, VisualIntent, VisualType
from app.core.exceptions import NotAvailableInPhase
from app.visual.preferences import DEFAULT_TARGETS, SourceKind, VisualPreferences
from app.visual.research import UnavailableResearchService


def test_defaults_match_the_spec():
    p = VisualPreferences()
    assert {k: s.target_percent for k, s in p.sources.items()} == {
        "YOUTUBE": 20, "STOCK_IMAGES": 15, "STOCK_VIDEOS": 15, "AI_IMAGES": 20, "WEB_IMAGES": 10, "WEB_VIDEOS": 10, "SCREENSHOTS": 10}
    assert all(s.enabled for s in p.sources.values()) and p.min_accuracy_score == 85
    r = p.report()
    assert r.total == 100 and r.is_balanced and r.warnings == []
    assert p.prefer_evidence and p.avoid_repeated_visuals and p.allow_visual_interpretation
    assert not p.prefer_real_visuals and not p.prefer_ai_visuals and not p.match_narration_literally


def test_exceeding_100_is_a_warning_not_an_error():
    p = VisualPreferences()
    p.setting(SourceKind.YOUTUBE).target_percent = 35  # +15
    r = p.report()
    assert r.total == 115 and not r.is_balanced
    assert any("exceed 100%" in w and "soft preferences" in w for w in r.warnings)
    p.setting(SourceKind.YOUTUBE).target_percent = 5
    assert any("less than 100%" in w for w in p.report().warnings) and p.report().total == 85


def test_disabled_sources_do_not_count_and_all_disabled_warns():
    p = VisualPreferences()
    p.setting(SourceKind.AI_IMAGES).enabled = False
    assert p.report().total == 80
    for s in p.sources.values():
        s.enabled = False
    r = p.report()
    assert r.total == 0 and any("No visual source" in w for w in r.warnings)


def test_conflicting_rules_warn_but_are_allowed():
    p = VisualPreferences(prefer_real_visuals=True, prefer_ai_visuals=True, match_narration_literally=True)
    ws = p.report().warnings
    assert any("opposite" in w for w in ws) and any("overlap" in w for w in ws)


def test_sanitize_clamps_without_raising():
    p = VisualPreferences(min_accuracy_score=250)
    p.setting(SourceKind.YOUTUBE).target_percent = -5
    q = p.sanitized()
    assert q.min_accuracy_score == 100 and q.setting(SourceKind.YOUTUBE).target_percent == 0
    assert p.min_accuracy_score == 250  # original untouched
    q.min_accuracy_score = -3
    assert q.sanitized().min_accuracy_score == 0


def test_roundtrip_and_forward_compat():
    p = VisualPreferences(min_accuracy_score=92, prefer_ai_visuals=True)
    p.setting(SourceKind.SCREENSHOTS).target_percent = 40
    assert VisualPreferences.from_dict(p.to_dict()) == p
    legacy = {"min_accuracy_score": 70, "sources": {"YOUTUBE": {"enabled": False, "target_percent": 50}}}
    q = VisualPreferences.from_dict(legacy)
    assert q.setting(SourceKind.YOUTUBE).enabled is False and q.setting(SourceKind.AI_IMAGES).target_percent == DEFAULT_TARGETS[SourceKind.AI_IMAGES]
    assert VisualPreferences.from_dict({}) == VisualPreferences()


def test_preferences_persist_in_the_project_undoably(project_ws):
    ws = project_ws
    prefs = VisualPreferences(min_accuracy_score=91, avoid_repeated_visuals=False)
    prefs.setting(SourceKind.SCREENSHOTS).target_percent = 45
    ws.set_visual_preferences(prefs)
    assert ws.project.dirty and ws.project.visual_preferences.min_accuracy_score == 91
    ws.undo()
    assert ws.project.visual_preferences.min_accuracy_score == 85
    ws.redo()
    root = ws.project.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    p = ws.project.visual_preferences
    assert p.min_accuracy_score == 91 and p.avoid_repeated_visuals is False and p.setting(SourceKind.SCREENSHOTS).target_percent == 45
    assert p.report().total == 135  # saved even though over 100: soft targets never block


def test_rapid_preference_edits_merge_into_one_undo_step(project_ws):
    ws = project_ws
    for v in (86, 87, 88, 89):
        p = VisualPreferences(min_accuracy_score=v)
        ws.set_visual_preferences(p)
    assert ws.project.visual_preferences.min_accuracy_score == 89
    ws.undo()
    assert ws.project.visual_preferences.min_accuracy_score == 85


def test_visual_research_is_an_interface_only_in_phase_2(project_ws):
    svc = UnavailableResearchService()
    scene = Scene("scene_001", "1", 0, 1)
    intent = VisualIntent("scene_001", VisualType.PROCESS)
    with pytest.raises(NotAvailableInPhase, match="Phase 3"):
        svc.search_scene(scene, intent, VisualPreferences())
    with pytest.raises(NotAvailableInPhase):
        svc.search_candidates(scene, intent, SourceKind.YOUTUBE)
    assert isinstance(project_ws.research, UnavailableResearchService)
