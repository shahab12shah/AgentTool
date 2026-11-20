"""The stage registry: which analyzer runs when, what it depends on and how a re-run is scoped.

Every analyzer module exposes ``analyze(ctx, prior, *, scene_filter=None) -> StageOutput`` (see ``editorial_models.StageOutput``). ``scope`` tells the engine how much of the previous result a
re-run can keep:

* ``scene``   - a scene's result depends only on that scene (and project-wide settings): only changed scenes are analysed again.
* ``window``  - it also depends on the neighbours: changed scenes plus ``settings.window`` scenes each side are analysed again.
* ``global``  - it needs the whole video (story arc, repetition across the project, style outliers): analysed again whenever any scene changed.

An analyzer given a ``scene_filter`` computes ``section["scenes"][scene_id]`` and the findings of those scenes only, but ``section["global"]`` (project-level numbers, the score) always from the
whole context. Stages whose module is not importable are reported as unavailable instead of silently skipped.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from app.editorial_intelligence.editorial_models import STAGE_IDS, STAGE_LABELS

# stage id -> (module, scope, depends on)
SPEC: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "context": ("story_analyzer", "global", ()),
    "narrative": ("narrative_classifier", "window", ("context",)),
    "story_arc": ("story_arc_analyzer", "global", ("narrative",)),
    "relationships": ("scene_relationship_analyzer", "global", ("narrative",)),
    "sequence": ("visual_sequence_planner", "window", ("narrative", "relationships")),
    "continuity": ("visual_consistency_analyzer", "window", ("relationships",)),
    "diversity": ("visual_diversity_analyzer", "global", ("relationships",)),
    "evidence": ("evidence_alignment", "scene", ("narrative",)),
    "pacing": ("pacing_intelligence", "window", ("narrative",)),
    "emphasis": ("emphasis_planner", "scene", ("narrative",)),
    "style": ("style_consistency", "global", ()),
}
assert set(SPEC) | {"recommendations"} == set(STAGE_IDS)


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    scope: str
    depends_on: tuple[str, ...]
    analyze: Callable | None  # None = the module could not be imported
    problem: str = ""


def default_stages() -> list[Stage]:
    out: list[Stage] = []
    for sid in STAGE_IDS:
        if sid == "recommendations":
            continue
        module, scope, deps = SPEC[sid]
        try:
            fn = importlib.import_module(f"app.editorial_intelligence.{module}").analyze
            out.append(Stage(sid, STAGE_LABELS[sid], scope, deps, fn))
        except (ImportError, AttributeError) as exc:
            out.append(Stage(sid, STAGE_LABELS[sid], scope, deps, None, f"{module} is not available ({exc})"))
    return out
