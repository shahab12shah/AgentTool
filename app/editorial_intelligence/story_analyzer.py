"""Stage ``context``: the project-level facts every later stage reads (topic, content type, size, what the voice-over covers, how much of the project the user has already taken over).

Deterministic and cheap. The story structure itself (roles, arc, sections) is produced by the narrative / story-arc stages; this one only states what the project IS so a reviewer can see what the
analysis was based on.
"""

from __future__ import annotations

from app.editorial_intelligence.context import EditorialContext
from app.editorial_intelligence.editorial_models import Basis, StageOutput


def analyze(ctx: EditorialContext, prior, *, scene_filter=None) -> StageOutput:
    scenes = ctx.scenes
    n = len(scenes)
    visual_scenes = [s for s in scenes if s.sc.asset is not None]
    claims = sum(len(s.sc.scene.claims) for s in scenes)
    numbers = sum(len(s.sc.scene.numbers) for s in scenes)
    overridden = sum(1 for s in scenes if s.overrides)
    user_chosen = sum(1 for s in scenes if s.sc.assignment is not None and s.sc.assignment.selected_by == "USER")
    words = sum(len(s.sc.words) for s in scenes)
    section = {"global": {
        "topic": ctx.topic, "content_type": ctx.content_type, "mode": ctx.settings.mode.value, "scenes": n, "scenes_with_visual": len(visual_scenes), "duration": round(ctx.total_duration, 3),
        "words": words, "claims": claims, "numbers": numbers, "scenes_with_user_overrides": overridden, "visuals_chosen_by_user": user_chosen,
        "reference_style_applied": ctx.reference.active, "preferences_signature": ctx.preferences.signature, "qc_open_findings": len(ctx.qc_open),
        "has_transcript": words > 0, "basis": Basis.COMPUTED.value,
    }}
    notes = []
    if n == 0:
        notes.append("There are no scenes yet: run scene analysis first.")
    elif not words:
        notes.append("There is no voice-over transcript, so timing-dependent analysis is limited.")
    return StageOutput("context", section, [], {}, "DONE", "; ".join(notes), "", {"scenes": float(n)}, notes, [s.id for s in scenes])
