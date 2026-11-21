"""The action contract: which ``ActionSpec`` kinds exist, what parameters each one takes, which manual-override aspects protect it and how it is applied.

Analyzers build ``ActionSpec`` objects with ``make_action`` (which checks the parameter shape); the recommendation validator re-checks them against the project; the apply builder turns the edit
kinds into structured Phase 10 / presentation / editing operations inside one command transaction. Route kinds (SEARCH_AGAIN ...) change nothing themselves: they point the user at an existing
service and never start expensive work on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.editorial_intelligence.editorial_models import ActionKind, ActionSpec, OverrideType, TargetRef

TRANSITION_TYPES = ("FADE", "DISSOLVE", "WIPE", "SLIDE")  # what the renderer supports (see Phase 10 ops_props)
TEXT_KINDS = ("HEADLINE", "NUMBER_CARD", "LOWER_THIRD", "QUOTE", "DATE")  # on-screen text kinds the presentation service can add (see PresentationService.add_graphic)


@dataclass(frozen=True)
class ActionSpecInfo:
    kind: ActionKind
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    targets: str = "clip"  # clip | caption | scene | none: what the targets must name
    blocked_by: tuple[OverrideType, ...] = ()  # manual-override aspects that make the action refuse (the recommendation is BLOCKED, shown but not applicable)
    applies_via: str = ""  # how it is carried out (documentation + dispatch key for the apply builder)
    route: bool = False  # True = changes nothing, points at an existing service
    summary: str = ""


CATALOG: dict[ActionKind, ActionSpecInfo] = {i.kind: i for i in (
    ActionSpecInfo(ActionKind.REPLACE_VISUAL, ("candidate_id",), ("asset_id", "keep_duration"), "clip", (OverrideType.VISUAL_OVERRIDE, OverrideType.SOURCE_OVERRIDE),
                   "ws.edit.replace_clip / editing.replace_clip_asset (needs a candidate that is already a project asset)", False, "Swap the visual for an alternative the research already found"),
    ActionSpecInfo(ActionKind.SET_SHOT_DURATION, ("duration",), ("ripple",), "clip", (OverrideType.TIMING_OVERRIDE,), "ws.edit.set_duration (+ ripple scope when ripple=True)", False,
                   "Change how long the shot stays on screen (the new length in seconds)"),
    ActionSpecInfo(ActionKind.MOVE_CLIP_START, ("start",), (), "clip", (OverrideType.TIMING_OVERRIDE,), "ws.edit.set_start (absolute timeline seconds)", False, "Move a clip so it starts at the spoken moment"),
    ActionSpecInfo(ActionKind.SET_TRANSITION, ("type",), ("duration",), "clip", (OverrideType.TRANSITION_OVERRIDE,), "ws.edit.set_transition on the incoming clip", False, "Use a calmer / better-fitting transition"),
    ActionSpecInfo(ActionKind.REMOVE_TRANSITION, (), (), "clip", (OverrideType.TRANSITION_OVERRIDE,), "ws.edit.remove_transition", False, "Cut instead of transitioning"),
    ActionSpecInfo(ActionKind.ADD_PUNCH_IN, ("end_scale",), ("start_scale", "at"), "clip", (OverrideType.MOTION_OVERRIDE,), "ws.edit.add_keyframe(scale) x2 (keyframes on the clip)", False, "Slow push-in on the important moment"),
    ActionSpecInfo(ActionKind.REMOVE_MOTION, (), (), "clip", (OverrideType.MOTION_OVERRIDE,), "ws.edit.delete_keyframes (scale / position keyframes of the clip)", False, "Hold the shot still"),
    ActionSpecInfo(ActionKind.CAPTION_EMPHASIS, ("words",), (), "caption", (OverrideType.CAPTION_OVERRIDE,), "presentation.update_caption(emphasis=...)", False, "Emphasise the key word(s) in the caption"),
    ActionSpecInfo(ActionKind.ADD_TEXT_EMPHASIS, ("content", "kind"), ("start", "duration"), "scene", (OverrideType.STYLE_OVERRIDE,), "presentation.add_graphic(kind, content, start, duration, scene_id=...)", False,
                   "Show a number / date / name on screen exactly as spoken"),
    ActionSpecInfo(ActionKind.RELABEL_ILLUSTRATIVE, ("label",), (), "clip", (OverrideType.STYLE_OVERRIDE,), "presentation.add_graphic(LOWER_THIRD, label) over the clip", False,
                   "Label an illustration so it cannot be mistaken for an authentic record"),
    ActionSpecInfo(ActionKind.SEARCH_AGAIN, (), ("query_hint",), "scene", (), "research.search_again (the user starts it)", True, "Search for a better visual"),
    ActionSpecInfo(ActionKind.EXPAND_SOURCES, (), ("sources",), "scene", (), "research.expand_sources (the user starts it)", True, "Look in more sources"),
    ActionSpecInfo(ActionKind.GENERATE_AI_VISUAL, (), ("prompt",), "scene", (), "research AI image candidate (the user starts it; labelled as AI-generated)", True, "Create an illustrative AI image (clearly labelled)"),
    ActionSpecInfo(ActionKind.USE_EXISTING_EVIDENCE, (), ("asset_id", "candidate_id"), "scene", (), "assign an existing project asset / evidence candidate (the user picks)", True, "Use evidence already in the project"),
    ActionSpecInfo(ActionKind.MANUAL_SELECTION, (), (), "scene", (), "open the scene in Visual Review", True, "Choose the visual yourself"),
    ActionSpecInfo(ActionKind.KEEP_CURRENT, (), (), "scene", (), "records a KEEP decision", True, "Keep the current visual"),
    ActionSpecInfo(ActionKind.REVIEW_ONLY, (), (), "none", (), "nothing to apply", True, "Review this observation"),
)}
assert set(CATALOG) == set(ActionKind)


def info(kind: ActionKind) -> ActionSpecInfo:
    return CATALOG[kind]


def shape_problems(action: ActionSpec) -> list[str]:
    """What is wrong with an action's SHAPE (missing / unknown parameters, wrong target kind). The project-level checks are the validator's job."""
    i = CATALOG[action.kind]
    problems = [f"missing parameter “{r}”" for r in i.required if r not in action.params]
    problems += [f"unknown parameter “{k}”" for k in action.params if k not in i.required and k not in i.optional]
    if i.targets == "clip" and not any(t.clip_id for t in action.targets):
        problems.append("the action must name a timeline clip")
    elif i.targets == "caption" and not any(t.clip_id for t in action.targets):
        problems.append("the action must name a caption clip")
    elif i.targets == "scene" and not any(t.scene_id for t in action.targets):
        problems.append("the action must name a scene")
    if action.kind is ActionKind.SET_TRANSITION and action.params.get("type") not in TRANSITION_TYPES:
        problems.append(f"transition type must be one of {', '.join(TRANSITION_TYPES)}")
    if action.kind is ActionKind.ADD_TEXT_EMPHASIS and action.params.get("kind") not in TEXT_KINDS:
        problems.append(f"text kind must be one of {', '.join(TEXT_KINDS)}")
    return problems


def make_action(kind: ActionKind, targets: list[TargetRef], summary: str = "", **params: Any) -> ActionSpec:
    """Build an ActionSpec and check its shape (raises ``ValueError`` on a malformed action: a bug in the analyzer, never user data)."""
    a = ActionSpec(kind, dict(params), list(targets), summary or CATALOG[kind].summary)
    bad = shape_problems(a)
    if bad:
        raise ValueError(f"{kind.value}: " + "; ".join(bad))
    return a


def blocked_aspects_for(kind: ActionKind) -> tuple[OverrideType, ...]:
    return CATALOG[kind].blocked_by


@dataclass
class ActionRoute:
    """What a route kind (no edit) tells the UI to open."""

    kind: ActionKind
    scene_id: str
    params: dict[str, Any] = field(default_factory=dict)
