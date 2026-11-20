"""OverrideResolver: ONE answer to "has the user taken this over?", read from the ownership marks the application already keeps - no second store of truth.

Sources (all existing): ``VisualAssignment.selected_by == "USER"``; ``Clip.created_by == "USER"`` on a clip that belongs to a scene (the edit hook marks every manual edit of an AI clip this way);
Phase 10 user locks (``Clip.locks``, ``Clip.protected``) and ``Track.locked``; the legacy AI lock (``Clip.locked``); ``EditingDecision`` / ``PresentationDecision`` created by the USER (or locked);
``Scene.user_edited_fields`` / ``boundaries_locked``; ``user_set`` on the editing / caption / audio settings; and the explicit ``EditorialOverride`` records of this phase (a recommendation the user
declined for good = KEEP; a manual override the user gave back to the AI = RESET, which cancels the marks above for that aspect).

An override never expires by itself. A recommendation that touches an overridden aspect is still shown (the user may want to read it) but it is BLOCKED: it cannot be applied until the user resets the override.
"""

from __future__ import annotations

from typing import Any

from app.editing.models import Creator, DecisionType
from app.editorial_intelligence.editorial_models import OverrideType

ASPECTS = tuple(t.value for t in OverrideType)
_DECISION_ASPECT = {
    DecisionType.VISUAL_TIMING: OverrideType.TIMING_OVERRIDE, DecisionType.CUT: OverrideType.TIMING_OVERRIDE, DecisionType.TRIM: OverrideType.TIMING_OVERRIDE,
    DecisionType.ZOOM: OverrideType.MOTION_OVERRIDE, DecisionType.PAN: OverrideType.MOTION_OVERRIDE, DecisionType.KEYFRAME: OverrideType.MOTION_OVERRIDE,
    DecisionType.TRANSITION: OverrideType.TRANSITION_OVERRIDE, DecisionType.CAPTION_EMPHASIS: OverrideType.CAPTION_OVERRIDE,
    DecisionType.TEXT: OverrideType.STYLE_OVERRIDE, DecisionType.NUMBER_EMPHASIS: OverrideType.STYLE_OVERRIDE, DecisionType.EVIDENCE_FOCUS: OverrideType.STYLE_OVERRIDE,
    DecisionType.AUDIO_DUCK: OverrideType.AUDIO_OVERRIDE,
}
_LOCK_ASPECT = {"item": tuple(OverrideType), "position": (OverrideType.TIMING_OVERRIDE,), "timing": (OverrideType.TIMING_OVERRIDE,), "transform": (OverrideType.MOTION_OVERRIDE,),
                "audio": (OverrideType.AUDIO_OVERRIDE,), "style": (OverrideType.STYLE_OVERRIDE, OverrideType.TRANSITION_OVERRIDE, OverrideType.CAPTION_OVERRIDE)}


def _add(out: dict[str, list[str]], aspect: OverrideType | str, why: str) -> None:
    key = aspect.value if isinstance(aspect, OverrideType) else aspect
    if why not in out.setdefault(key, []):
        out[key].append(why)


def clip_overrides(project: Any, clip: Any) -> dict[str, list[str]]:
    """Aspect -> reasons, for one timeline clip."""
    out: dict[str, list[str]] = {}
    tl = project.timeline
    try:
        track = tl.get_track(clip.track_id)
    except Exception:  # noqa: BLE001
        track = None
    if track is not None and track.locked:
        for a in OverrideType:
            _add(out, a, f"track “{track.name}” is locked")
    if clip.created_by == "USER" and clip.scene_id:
        if clip.kind == "media":
            _add(out, OverrideType.VISUAL_OVERRIDE, "you placed or edited this clip")
        elif clip.kind == "caption":
            _add(out, OverrideType.CAPTION_OVERRIDE, "you edited this caption")
        else:
            _add(out, OverrideType.STYLE_OVERRIDE, "you created or edited this graphic")
    if getattr(clip, "locked", False):
        for a in (OverrideType.VISUAL_OVERRIDE, OverrideType.TIMING_OVERRIDE, OverrideType.MOTION_OVERRIDE, OverrideType.TRANSITION_OVERRIDE):
            _add(out, a, "the clip is locked against AI changes")
    for lock in getattr(clip, "locks", ()) or ():
        for a in _LOCK_ASPECT.get(lock, ()):
            _add(out, a, f"you locked the clip’s {lock}" if lock != "item" else "you locked this clip")
    did = getattr(clip, "ai_decision_id", "")
    dec = project.editing_decisions.get(did) if did else None
    dec = dec or (project.presentation_decisions.get(did) if did else None)
    if dec is not None and (dec.created_by is Creator.USER or dec.locked):
        dtype = getattr(dec, "type", None)
        aspect = _DECISION_ASPECT.get(dtype) if isinstance(dtype, DecisionType) else None
        if aspect is not None:
            _add(out, aspect, "you changed this decision" if dec.created_by is Creator.USER else "this decision is locked")
    return out


def _settings_overrides(project: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    es, cs, au = project.editing_settings, project.caption_settings, project.audio_settings
    user_set = {"editing": set(es.user_set), "caption": set(cs.user_set), "audio": set(au.user_set)}
    if user_set["editing"] & {"pacing", "min_shot", "max_shot"}:
        _add(out, OverrideType.TIMING_OVERRIDE, "you set the pacing yourself")
    if user_set["editing"] & {"motion_intensity"}:
        _add(out, OverrideType.MOTION_OVERRIDE, "you set the motion level yourself")
    if user_set["editing"] & {"transition_frequency", "smart_transitions"}:
        _add(out, OverrideType.TRANSITION_OVERRIDE, "you set the transitions yourself")
    if user_set["caption"]:
        _add(out, OverrideType.CAPTION_OVERRIDE, "you changed caption settings")
    if user_set["audio"]:
        _add(out, OverrideType.AUDIO_OVERRIDE, "you changed audio settings")
    if user_set["editing"] & {"style", "text_emphasis", "number_emphasis"}:
        _add(out, OverrideType.STYLE_OVERRIDE, "you set the editing style yourself")
    return out


def scene_overrides(project: Any, scene_id: str, *, keep: bool = True) -> dict[str, list[str]]:
    """Aspect -> reasons for a scene (union over its clips, assignment, scene marks and the explicit editorial records, RESET cancelling what it names). ``keep=False`` leaves out the
    "keep as is" records: the hash of a scene must not change just because the user declined a recommendation about it."""
    out: dict[str, list[str]] = {}
    a = project.visual_assignments.get(scene_id)
    if a is not None and a.selected_by == "USER":
        _add(out, OverrideType.VISUAL_OVERRIDE, "you chose this visual")
        _add(out, OverrideType.SOURCE_OVERRIDE, "you chose this source")
    sc = next((s for s in project.scenes if s.id == scene_id), None)
    if sc is not None and (sc.boundaries_locked or sc.user_edited_fields):
        _add(out, OverrideType.TIMING_OVERRIDE, "you set this scene’s boundaries")
    for clip in project.timeline.all_clips():
        if clip.scene_id == scene_id:
            for k, v in clip_overrides(project, clip).items():
                for why in v:
                    _add(out, k, why)
    for k, v in _settings_overrides(project).items():
        for why in v:
            _add(out, k, why)
    for rec in project.editorial.editorial_overrides:
        if rec.scene_id and rec.scene_id != scene_id:
            continue
        if rec.action == "RESET" and (rec.scene_id or not rec.clip_id):
            out.pop(rec.type.value, None)  # the user gave it back: nothing protects that aspect any more
        elif rec.action == "KEEP" and keep:
            _add(out, rec.type, rec.reason or "you asked to keep it as it is")
    return out


def blocked_aspects(overrides: dict[str, list[str]], needed: list[OverrideType | str]) -> dict[str, list[str]]:
    """The subset of ``needed`` aspects that an override protects (with the reasons)."""
    return {(n.value if isinstance(n, OverrideType) else n): overrides[(n.value if isinstance(n, OverrideType) else n)] for n in needed if overrides.get(n.value if isinstance(n, OverrideType) else n)}
