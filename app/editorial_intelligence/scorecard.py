"""The editorial scorecard: separate, explainable dimensions; measured numbers kept apart from editorial judgment; a serious warning can never be averaged away.

Each analyzer states its own dimension score in ``section["global"]["score"] = {"value": 0..100 | None, "basis": "COMPUTED" | "JUDGMENT", "confidence": 0..100, "measured": {...}, "note": "...", "evidence": [...]}``.
This module only assembles them: it does not invent a score for a dimension nobody measured (that item is ``None`` and says so), counts the serious findings per dimension, caps an item that has
one (an evidence-integrity problem cannot show 95), and caps the overall score so a good average cannot hide it.
"""

from __future__ import annotations

from typing import Any

from app.editorial_intelligence.editorial_models import Basis, Category, Finding, ScoreItem, Scorecard

DIMENSIONS: dict[str, tuple[str, tuple[Category, ...]]] = {
    "Narrative Coherence": ("story_arc", (Category.NARRATIVE_FLOW,)),
    "Visual Sequence Quality": ("sequence", (Category.VISUAL_RELEVANCE,)),
    "Visual Continuity": ("continuity", (Category.VISUAL_CONTINUITY,)),
    "Visual Diversity": ("diversity", (Category.VISUAL_DIVERSITY,)),
    "Pacing Suitability": ("pacing", (Category.PACING,)),
    "Evidence Alignment": ("evidence", (Category.EVIDENCE_ALIGNMENT,)),
    "Style Consistency": ("style", (Category.STYLE_CONSISTENCY, Category.REFERENCE_STYLE_CONFLICT)),
    "Emphasis Quality": ("emphasis", (Category.EMPHASIS, Category.CAPTION_HIERARCHY)),
    "Audio-Visual Alignment": ("emphasis", (Category.AUDIO_VISUAL_ALIGNMENT,)),
}
_PRIMARY: dict[str, str] = {}
for _dim, (_stage, _c) in DIMENSIONS.items():
    _PRIMARY.setdefault(_stage, _dim)
SERIOUS_ITEM_CAP = 60.0
SERIOUS_OVERALL_CAP = 70.0


def build_scorecard(ctx: Any, sections: dict[str, dict[str, Any]], findings: list[Finding]) -> Scorecard:
    items: list[ScoreItem] = []
    capped: list[str] = []
    for dim, (stage, cats) in DIMENSIONS.items():
        g = (sections.get(stage) or {}).get("global") or {}
        multi = g.get("scores") if isinstance(g.get("scores"), dict) else {}
        sc = multi.get(dim) or (g.get("score") if dim == _PRIMARY[stage] else None)  # a stage with two dimensions states both under "scores"; a single one under "score"
        serious = sum(1 for f in findings if f.serious and f.category in cats and not f.intentional)
        if not isinstance(sc, dict) or sc.get("value") is None:
            items.append(ScoreItem(dim, None, Basis.JUDGMENT, 0.0, {}, [], (sc or {}).get("note") or "Not measurable for this project (or the analysis did not run).", serious))
            continue
        value = max(0.0, min(100.0, float(sc["value"])))
        note = str(sc.get("note", ""))
        if serious and value > SERIOUS_ITEM_CAP:
            value, note = SERIOUS_ITEM_CAP, (note + " " if note else "") + f"Capped at {SERIOUS_ITEM_CAP:.0f}: {serious} serious warning(s)."
            capped.append(dim)
        try:
            basis = Basis(sc.get("basis", "JUDGMENT"))
        except ValueError:
            basis = Basis.JUDGMENT
        items.append(ScoreItem(dim, round(value, 1), basis, float(sc.get("confidence", 0.0)), {k: float(v) for k, v in (sc.get("measured") or {}).items() if isinstance(v, (int, float))},
                               [str(e) for e in (sc.get("evidence") or [])][:12], note, serious))
    values = [i.score for i in items if i.score is not None]
    overall = round(sum(values) / len(values), 1) if values else None
    note = ""
    total_serious = sum(1 for f in findings if f.serious and not f.intentional)
    if overall is not None and total_serious and overall > SERIOUS_OVERALL_CAP:
        overall, note = SERIOUS_OVERALL_CAP, f"The overall score is capped at {SERIOUS_OVERALL_CAP:.0f} because {total_serious} serious warning(s) are open; a good average does not cancel them."
    elif overall is None:
        note = "No dimension could be measured."
    return Scorecard(items, overall, note, capped)
