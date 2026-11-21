"""Read-only views of the stored editorial analysis for the UI: the scene-level editorial inspector (spec 24) and the visual sequence map (spec 25).

Pure functions over ``Project`` (+ its ``EditorialState``): they assemble what the analysis found and what the project says NOW, and mark anything that is no longer current. Nothing here changes a
project or runs an analysis; the map is informational - it never rearranges the video.
"""

from __future__ import annotations

from typing import Any

from app.editorial_intelligence import override_resolver
from app.editorial_intelligence.editorial_models import EditorialState, Recommendation, RecStatus

_OPEN = (RecStatus.PENDING, RecStatus.BLOCKED, RecStatus.OUTDATED)


def _scene(project: Any, scene_id: str) -> Any:
    return next((s for s in project.scenes if s.id == scene_id), None)


def _section_scene(state: EditorialState, stage: str, scene_id: str) -> dict[str, Any]:
    return ((state.sections.get(stage) or {}).get("scenes") or {}).get(scene_id) or {}


def _rec_row(r: Recommendation) -> dict[str, Any]:
    return {"recommendation_id": r.recommendation_id, "category": r.category.value, "priority": r.priority.value, "title": r.title, "status": r.status.value, "confidence": r.confidence,
            "blocked_by_override": list(r.blocked_by_override), "action": r.suggested_action.kind.value}


def _visual_of(project: Any, scene_id: str) -> dict[str, Any]:
    a = project.visual_assignments.get(scene_id)
    if a is None:
        return {"assigned": False}
    asset = project.assets.get(a.asset_id) if a.asset_id else None
    cand = project.visual_candidates.get(a.candidate_id) if a.candidate_id else None
    score = project.candidate_scores.get(a.candidate_id) if a.candidate_id else None
    return {"assigned": True, "asset_id": a.asset_id or "", "name": (asset.name if asset else (cand.title if cand else "")), "source_type": (a.source_type.value if a.source_type else (asset.source_type.value if asset else "")),
            "selected_by": a.selected_by, "accuracy": a.accuracy_score if a.accuracy_score is not None else (score.overall if score else None), "evidence_kind": a.evidence_kind.value,
            "approved": a.approved, "skipped": a.skipped, "candidate_id": a.candidate_id or "", "title": cand.title if cand else "", "basis": score.basis if score else ""}


def _alternatives(project: Any, scene_id: str, limit: int = 6) -> list[dict[str, Any]]:
    state = project.research_status.get(scene_id)
    out = []
    for r in (state.ranked if state else []):
        if r.role == "BEST" and project.visual_assignments.get(scene_id) is not None and project.visual_assignments[scene_id].candidate_id == r.candidate_id:
            continue
        c = project.visual_candidates.get(r.candidate_id)
        if c is None:
            continue
        out.append({"candidate_id": c.candidate_id, "title": c.title, "source_type": c.source_type.value, "kind": c.kind, "accuracy": r.accuracy, "evidence_kind": c.evidence_kind.value,
                    "acquirable": bool(c.asset_id or c.local_path or c.media_url)})
        if len(out) >= limit:
            break
    return out


def build_scene_view(project: Any, scene_id: str) -> dict[str, Any]:
    """Everything the scene-level editorial inspector shows for one scene. Concise: structured facts and the analysis' own one-line reasons, never reasoning text."""
    st: EditorialState = project.editorial
    sc = _scene(project, scene_id)
    if sc is None:
        return {"scene_id": scene_id, "exists": False}
    order = [s.id for s in project.scenes]
    i = order.index(scene_id)
    role = st.scene_roles.get(scene_id)
    rel_scene = (((st.sections.get("relationships") or {}).get("global") or {}).get("by_scene") or {}).get(scene_id) or {}
    strategy = st.visual_strategy.get(scene_id)
    pacing = _section_scene(st, "pacing", scene_id)
    clips = [c for c in project.timeline.all_clips() if c.scene_id == scene_id]
    visual_clips = [c for c in clips if c.kind == "media" and not project.timeline.get_track(c.track_id).is_audio]
    current = round(sum(c.duration for c in visual_clips), 3)
    rec_dur = pacing.get("recommended") or {}
    recs = [r for r in st.editorial_recommendations if scene_id in r.scene_ids and r.status in _OPEN]
    overrides = override_resolver.scene_overrides(project, scene_id)
    locks = sorted({lk for c in clips for lk in c.locks} | ({"legacy_ai_lock"} if any(getattr(c, "locked", False) for c in clips) else set()))
    analysed = bool(st.editorial_analysis_history)
    return {
        "scene_id": scene_id, "exists": True, "label": sc.label, "index": i, "start": sc.start, "end": sc.end, "narration": sc.narration, "topic": sc.topic, "analysed": analysed,
        "role": {"roles": [{"role": r.role.value, "confidence": r.confidence} for r in role.roles], "uncertain": role.uncertain, "evidence_terms": list(role.evidence_terms)[:8], "basis": role.basis} if role else None,
        "purpose": (strategy.reason if strategy else ""),
        "previous": {"scene_id": order[i - 1] if i else "", "relation": rel_scene.get("previous", "")}, "next": {"scene_id": order[i + 1] if i + 1 < len(order) else "", "relation": rel_scene.get("next", "")},
        "related": list(rel_scene.get("related", []))[:8],
        "visual": _visual_of(project, scene_id), "alternatives": _alternatives(project, scene_id),
        "strategy": {"kind": strategy.kind.value, "needs_evidence": strategy.needs_evidence, "continue_previous": strategy.continue_previous, "several_visuals": strategy.several_visuals,
                     "confidence": strategy.confidence} if strategy else None,
        "continuity": _section_scene(st, "continuity", scene_id), "evidence": _section_scene(st, "evidence", scene_id),
        "duration": {"current": current, "recommended": rec_dur or None, "verdict": pacing.get("verdict", "")},
        "decisions": [{"decision_id": c.ai_decision_id, "slot": c.slot, "created_by": c.created_by} for c in clips if c.ai_decision_id],
        "overrides": overrides, "locks": locks, "user_chosen_visual": bool(project.visual_assignments.get(scene_id) and project.visual_assignments[scene_id].selected_by == "USER"),
        "recommendations": [_rec_row(r) for r in recs],
    }


def build_sequence_map(project: Any) -> dict[str, Any]:
    """The whole video as one informational overview: sections, every scene with its role / visual / source type / evidence point / repetition / pacing change. Click targets are times and scene ids; nothing moves."""
    st: EditorialState = project.editorial
    structure = st.narrative_structure
    scenes: list[dict[str, Any]] = []
    asset_scenes: dict[str, list[str]] = {}
    for sc in project.scenes:
        a = project.visual_assignments.get(sc.id)
        if a is not None and a.asset_id:
            asset_scenes.setdefault(a.asset_id, []).append(sc.id)
    div = ((st.sections.get("diversity") or {}).get("scenes") or {})
    ev = ((st.sections.get("evidence") or {}).get("scenes") or {})
    pace = ((st.sections.get("pacing") or {}).get("scenes") or {})
    medians = sorted(float((p.get("recommended") or {}).get("target", 0.0)) for p in pace.values() if p.get("recommended"))
    median = medians[len(medians) // 2] if medians else 0.0
    prev_target = None
    for sc in project.scenes:
        role = st.scene_roles.get(sc.id)
        v = _visual_of(project, sc.id)
        aid = v.get("asset_id", "")
        repeated = bool(aid) and len(asset_scenes.get(aid, [])) > 1
        grade = (ev.get(sc.id) or {}).get("grade", "")
        target = float(((pace.get(sc.id) or {}).get("recommended") or {}).get("target", 0.0))
        shift = bool(prev_target and target and (target >= 1.6 * prev_target or target <= prev_target / 1.6))  # a large step in the recommended hold: a pacing change worth seeing
        prev_target = target or prev_target
        scenes.append({"scene_id": sc.id, "label": sc.label, "start": sc.start, "end": sc.end, "role": role.primary.value if role and role.primary else "", "roles": [r.role.value for r in role.roles] if role else [],
                       "visual": v.get("name", ""), "source_type": v.get("source_type", ""), "asset_id": aid, "evidence_point": grade in ("DIRECT", "UNVERIFIED") or v.get("evidence_kind") == "EVIDENCE",
                       "evidence_grade": grade, "repeated": repeated, "repeated_with": [s for s in asset_scenes.get(aid, []) if s != sc.id][:5] if repeated else [],
                       "intentional_repeat": bool((div.get(sc.id) or {}).get("intentional")), "pacing_shift": shift,
                       "open_recommendations": sum(1 for r in st.editorial_recommendations if sc.id in r.scene_ids and r.status in _OPEN)})
    sections = [{"section_id": s.section_id, "title": s.title, "scene_ids": list(s.scene_ids), "start": s.start, "end": s.end, "role": s.role, "confidence": s.confidence} for s in (structure.sections if structure else st.story_sections)]
    return {"analysed": bool(st.editorial_analysis_history), "model": structure.model if structure else "", "model_confidence": structure.confidence if structure else 0.0, "sections": sections, "scenes": scenes,
            "median_recommended_hold": round(median, 2), "note": "Informational only: this overview never rearranges the video."}
