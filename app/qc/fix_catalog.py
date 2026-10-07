"""The fix catalog: the contract between the checkers (who *recommend* a fix) and the QCFixEngine (who *executes* it).

A checker never edits anything: it attaches a ``QCFixSpec`` built with one of the constructors below. The fix engine looks the ``kind`` up in ``HANDLED_KINDS``,
re-resolves every id on the UI thread at execution time (nothing in a spec is trusted to still be valid), refuses anything the user owns or locked, and runs the
change as ONE undoable command together with the bookkeeping that marks the issue fixed.

``safe`` means: deterministic, small, preserves the meaning and the user's content. A spec that is not safe always needs the user's confirmation.
Kinds routed to NAVIGATE / RESEARCH never change the project from QC: they take the user to the place where they decide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.qc.issue_model import FixRoute, QCFixSpec
from app.qc.settings import QCSettings


@dataclass(frozen=True)
class FixKind:
    kind: str
    label: str
    route: FixRoute
    safe_by_design: bool  # may ever be applied without a confirmation click (still subject to the user's fix permissions)
    description: str
    params: tuple[str, ...] = ()


CATALOG: dict[str, FixKind] = {k.kind: k for k in (
    # ---- safe (deterministic, small): may be auto-applied when the user's permission table says "auto"
    FixKind("caption.retime", "Retime caption", FixRoute.COMMAND, True, "Move/resize one caption window onto the spoken words (the text and word timing are never changed).", ("clip_id", "new_start", "new_end", "delta")),
    FixKind("caption.safe_margin", "Move caption inside the safe area", FixRoute.COMMAND, True, "Nudge a caption's position so it respects the safe margins.", ("clip_id", "position_xy")),
    FixKind("audio.duck", "Lower music under speech", FixRoute.COMMAND, True, "Add volume automation (keyframes) so music/SFX sit under the voice; the audio clip itself is untouched.", ("role", "assignment_id", "spans", "target_gain")),
    FixKind("clip.extend", "Extend clip within its source", FixRoute.COMMAND, True, "Make a clip a little longer, only as far as its source media and the free space allow.", ("clip_id", "new_end")),
    FixKind("clip.remove_empty", "Remove empty/zero-length item", FixRoute.COMMAND, True, "Remove an accidental zero-duration or contentless timeline item.", ("clip_id",)),
    FixKind("asset.relink", "Relink to the identical file", FixRoute.COMMAND, True, "Point a missing asset at a file with the same content hash (an exact copy).", ("asset_id", "new_path")),
    FixKind("param.normalize", "Clamp invalid value to valid bounds", FixRoute.COMMAND, True, "Bring an out-of-range opacity / scale / volume / keyframe back inside its valid range.", ("clip_id", "changes")),
    # ---- need confirmation (they change the edit or its meaning): never applied silently
    FixKind("motion.soften", "Soften zoom / movement", FixRoute.COMMAND, False, "Reduce the end scale or lengthen the animation of a keyframed move.", ("clip_id", "keyframes")),
    FixKind("transition.shorten", "Shorten transition", FixRoute.COMMAND, False, "Shorten a transition that is too long.", ("clip_id", "duration")),
    FixKind("gap.close", "Close unintended gap", FixRoute.COMMAND, False, "Extend the clip before the gap (within its source) so the picture is continuous.", ("clip_id", "new_end")),
    FixKind("clip.delete", "Delete clip", FixRoute.COMMAND, False, "Remove a duplicate / orphaned element.", ("clip_id",)),
    FixKind("caption.restyle", "Change caption settings", FixRoute.COMMAND, False, "Change a caption setting (size, position, lines) for future captions.", ("field", "value")),
    FixKind("audio.level", "Change a level", FixRoute.COMMAND, False, "Change the level of a clip/track.", ("clip_id", "volume")),
    FixKind("silence.remove", "Remove silence", FixRoute.COMMAND, False, "Remove a stretch of silence (changes narration timing).", ("start", "end")),
    FixKind("narration.retime", "Change narration timing", FixRoute.NAVIGATE, False, "Narration timing is changed in the voice/scene tools.", ("scene_id",)),
    FixKind("scene.restructure", "Change scene structure", FixRoute.NAVIGATE, False, "Split / merge scenes in the scene editor.", ("scene_id",)),
    FixKind("graphics.change", "Change graphic", FixRoute.NAVIGATE, False, "Review the graphic in the Audio & Captions / Timeline tools.", ("clip_id",)),
    FixKind("text.change", "Review on-screen text", FixRoute.NAVIGATE, False, "QC never corrects text: review it against the script.", ("clip_id",)),
    # ---- navigation / search routes
    FixKind("visual.replace", "Replace visual", FixRoute.NAVIGATE, False, "Choose another visual for this scene in the Review page.", ("scene_id",)),
    FixKind("visual.search_again", "Search again", FixRoute.RESEARCH, False, "Search for another visual for this scene.", ("scene_id",)),
    FixKind("asset.replace", "Replace asset", FixRoute.NAVIGATE, False, "Pick a replacement for a missing or unusable asset.", ("asset_id",)),
    FixKind("asset.remove", "Remove asset from timeline", FixRoute.NAVIGATE, False, "Remove a missing asset and the clips that use it.", ("asset_id",)),
    FixKind("scene.skip", "Skip scene", FixRoute.NAVIGATE, False, "Mark the scene as skipped.", ("scene_id",)),
    FixKind("open.timeline", "Open on timeline", FixRoute.NAVIGATE, False, "Show the item on the timeline.", ("clip_id",)),
    FixKind("open.scene", "Open scene", FixRoute.NAVIGATE, False, "Open the scene.", ("scene_id",)),
    FixKind("settings.open", "Open settings", FixRoute.NAVIGATE, False, "Open the relevant settings page.", ("page",)),
)}
HANDLED_KINDS = tuple(k for k, v in CATALOG.items() if v.route is FixRoute.COMMAND)


def _spec(kind: str, params: dict[str, Any], summary: str, safe: bool, settings: QCSettings | None = None) -> QCFixSpec:
    info = CATALOG[kind]
    perm = settings.permission(kind) if settings else ("auto" if info.safe_by_design else "confirm")
    intrinsic = bool(safe and info.safe_by_design)
    is_safe = intrinsic and perm == "auto"
    return QCFixSpec(kind=kind, params=params, safe=is_safe, intrinsic_safe=intrinsic, needs_confirmation=not is_safe, route=info.route, summary=summary)


# ---- constructors the checkers call (so the parameter names always match what the engine reads)
def caption_retime(clip_id: str, new_start: float, new_end: float, delta: float, settings: QCSettings | None = None) -> QCFixSpec:
    small = abs(delta) <= (settings.max_caption_shift_seconds if settings else 0.6)
    where = "earlier" if delta < 0 else "later"
    return _spec("caption.retime", {"clip_id": clip_id, "new_start": round(new_start, 3), "new_end": round(new_end, 3), "delta": round(delta, 3)},
                 f"Move the caption {abs(delta) * 1000:.0f} ms {where} to match the spoken words", small, settings)


def caption_safe_margin(clip_id: str, position_xy: list[float], settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("caption.safe_margin", {"clip_id": clip_id, "position_xy": [round(position_xy[0], 4), round(position_xy[1], 4)]}, "Move the caption inside the safe area", True, settings)


def audio_duck(role: str, assignment_id: str, spans: list[list[float]], target_gain: float, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("audio.duck", {"role": role, "assignment_id": assignment_id, "spans": [[round(a, 3), round(b, 3)] for a, b in spans], "target_gain": round(target_gain, 4)},
                 f"Lower the {role.lower()} under the voice with volume automation", True, settings)


def clip_extend(clip_id: str, new_end: float, extension: float, settings: QCSettings | None = None) -> QCFixSpec:
    small = extension <= (settings.max_clip_extension_seconds if settings else 1.0)
    return _spec("clip.extend", {"clip_id": clip_id, "new_end": round(new_end, 3)}, f"Extend the clip by {extension:.2f} s (within its source)", small, settings)


def clip_remove_empty(clip_id: str, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("clip.remove_empty", {"clip_id": clip_id}, "Remove the empty / zero-length item", True, settings)


def asset_relink(asset_id: str, new_path: str, exact: bool, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("asset.relink", {"asset_id": asset_id, "new_path": new_path, "exact": bool(exact)}, "Relink to the identical file" if exact else "Relink to a similar file (check it first)", exact, settings)


def param_normalize(clip_id: str, changes: dict[str, Any], summary: str, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("param.normalize", {"clip_id": clip_id, "changes": changes}, summary, True, settings)


def motion_soften(clip_id: str, keyframes: list[dict[str, Any]], summary: str, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("motion.soften", {"clip_id": clip_id, "keyframes": keyframes}, summary, False, settings)


def transition_shorten(clip_id: str, duration: float, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("transition.shorten", {"clip_id": clip_id, "duration": round(duration, 3)}, f"Shorten the transition to {duration:.2f} s", False, settings)


def gap_close(clip_id: str, new_end: float, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("gap.close", {"clip_id": clip_id, "new_end": round(new_end, 3)}, "Extend the previous clip over the gap", False, settings)


def clip_delete(clip_id: str, summary: str = "Delete the duplicate element", settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("clip.delete", {"clip_id": clip_id}, summary, False, settings)


def navigate(kind: str, summary: str = "", **params: Any) -> QCFixSpec:
    info = CATALOG[kind]
    return QCFixSpec(kind=kind, params=dict(params), safe=False, needs_confirmation=True, route=info.route, summary=summary or info.label)


def caption_restyle(field: str, value: Any, summary: str, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("caption.restyle", {"field": field, "value": value}, summary, False, settings)


def audio_level(clip_id: str, volume: float, summary: str, settings: QCSettings | None = None) -> QCFixSpec:
    return _spec("audio.level", {"clip_id": clip_id, "volume": round(volume, 4)}, summary, False, settings)
