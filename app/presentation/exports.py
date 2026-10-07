"""Derived views for the project document.

Timing, geometry and content of every caption, graphic, music and SFX object live on the timeline clips (single source of truth).
The document also lists them in dedicated sections (caption_segments, text_graphics, motion_graphics, music_assignments,
sfx_assignments) so the file is easy to inspect and so other tools can read them. These sections are *derived* when saving and
ignored when loading — restoring the clips restores everything, and nothing can drift out of sync.
"""

from __future__ import annotations

from typing import Any

from app.presentation.models import MusicAssignment, SfxAssignment
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT


def _ai_info(project, clip) -> dict[str, Any]:
    d = project.presentation_decisions.get(clip.ai_decision_id) or project.editing_decisions.get(clip.ai_decision_id)
    return {"reason": d.reason if d else "", "confidence": d.confidence if d else None}


def caption_segments(project) -> list[dict[str, Any]]:
    out = []
    for t in project.timeline.tracks:
        for c in t.clips:
            if c.kind == KIND_CAPTION and c.text:
                out.append({**c.text, "start": c.timeline_start, "end": c.timeline_end, "clip_id": c.id, "created_by": c.created_by, "locked": c.locked,
                            "ai_decision_id": c.ai_decision_id, **_ai_info(project, c)})
    return sorted(out, key=lambda d: d["start"])


def text_graphics(project) -> list[dict[str, Any]]:
    out = []
    for t in project.timeline.tracks:
        for c in t.clips:
            if c.kind == KIND_TEXT and c.text:
                out.append({"clip_id": c.id, "track_id": t.id, "scene_id": c.scene_id, "start": c.timeline_start, "duration": c.duration, "text": c.text,
                            "animation": c.animation, "created_by": c.created_by, "locked": c.locked, "ai_decision_id": c.ai_decision_id, **_ai_info(project, c)})
    return sorted(out, key=lambda d: d["start"])


def motion_graphics(project) -> list[dict[str, Any]]:
    out = []
    for t in project.timeline.tracks:
        for c in t.clips:
            if c.kind == KIND_GRAPHIC:
                out.append({"clip_id": c.id, "track_id": t.id, "scene_id": c.scene_id, "start": c.timeline_start, "duration": c.duration, "effects": c.effects,
                            "animation": c.animation, "created_by": c.created_by, "locked": c.locked, "ai_decision_id": c.ai_decision_id, **_ai_info(project, c)})
    return sorted(out, key=lambda d: d["start"])


def music_assignments(project) -> list[MusicAssignment]:
    groups: dict[str, list] = {}
    for c in project.timeline.get_track("track_a2").clips if _has(project, "track_a2") else []:
        groups.setdefault(c.metadata.get("assignment_id", c.id), []).append(c)
    out = []
    for aid, cl in groups.items():
        cl.sort(key=lambda c: c.timeline_start)
        first = cl[0]
        info = _ai_info(project, first)
        out.append(MusicAssignment(aid, first.asset_id, cl[0].timeline_start, cl[-1].timeline_end, float(first.audio.get("volume", 1.0)),
                                   float(first.audio.get("fade_in", 0.0)), float(cl[-1].audio.get("fade_out", 0.0)), len(cl) > 1 or bool(first.audio.get("loop")),
                                   bool(first.audio.get("ducking", True)), [c.id for c in cl], first.created_by, any(c.locked for c in cl), info["reason"],
                                   info["confidence"] or 100.0))
    return sorted(out, key=lambda a: a.start)


def sfx_assignments(project) -> list[SfxAssignment]:
    out = []
    for c in project.timeline.get_track("track_a3").clips if _has(project, "track_a3") else []:
        info = _ai_info(project, c)
        out.append(SfxAssignment(c.metadata.get("sfx_id", c.id), c.asset_id, c.scene_id, c.timeline_start, c.duration, float(c.audio.get("volume", 1.0)),
                                 float(c.audio.get("fade_in", 0.0)), float(c.audio.get("fade_out", 0.0)), str(c.audio.get("category", "")), info["reason"],
                                 info["confidence"] or 100.0, c.created_by, c.id, c.locked, str(c.metadata.get("trigger", ""))))
    return sorted(out, key=lambda a: a.timestamp)


def _has(project, track_id: str) -> bool:
    return any(t.id == track_id for t in project.timeline.tracks)
