"""Planners that change *properties* of existing clips: transform / audio / text properties, speed, transitions, keyframes, links, groups and user locks.

Every planner is a pure function ``plan_xxx(env, ...) -> EditPlan`` (it reads the timeline, never mutates it); ``PlanCommand`` validates the result (track locks, user
locks per aspect, overlaps, numbers) and applies it as one undoable step. Impossible or out-of-range requests raise ``TimelineError`` with a plain-language reason; nothing is
clamped silently — where a plan has to adjust something that follows from the user's request (e.g. fades that no longer fit after a speed change) it says so in ``plan.notes``.
A request that changes nothing returns an *empty* plan (``plan.empty``) with a note, never a no-op command.

What the renderer really reads (checked against ``app/rendering/compiler.py`` and ``ass.py``, so the editor never offers an illusion):
  * picture clips (video / image media): ``position`` ``scale`` ``rotation`` ``opacity`` (+ keyframes of those, ``blur`` and ``reveal``), ``effects["crop"]`` ``effects["fit"]``
    (``cover`` | ``contain``); the transitions CUT / FADE / DISSOLVE / WIPE / SLIDE (``duration`` only: WIPE always reveals left to right, SLIDE enters from the right);
  * audio clips: ``audio["volume"]`` (the clip gain; there is no separate gain), ``fade_in``, ``fade_out``, ``pan`` and volume keyframes. There is NO per-clip mute switch in the
    mixer, so "mute" is volume 0 (the previous volume is kept in ``metadata["premute_volume"]`` for "unmute");
  * text clips: ``text`` keys ``content`` ``subtitle`` ``style`` (decides the colour) ``position`` (0..1 of the frame) ``size`` ``alignment`` ``opacity`` ``background`` ``font``
    ``emphasis``; the clip's own position / scale / rotation / keyframes are not rendered for text, so they are not offered;
  * caption clips: ``text["position"]`` / ``["position_xy"]`` / ``["highlight_mode"]`` (the words themselves are edited by the caption service);
  * keyframe interpolation: linear and the three ease modes only (a "hold / step" keyframe cannot be rendered, so it is not offered).
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Iterable, Sequence

from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import KIND_MEDIA, LOCK_KINDS, Clip
from app.timeline.keyframes import INTERPOLATIONS, PROPERTIES, Keyframe, window_keyframes
from app.timeline.ops_common import group_members, link_members, new_group_id, new_link_id
from app.timeline.plan import SPEED_RANGE, EditPlan, PlanEnv
from app.timeline.track import Track, TrackKind

# ------------------------------------------------------------------------------------------------------------------------ ranges
SCALE_RANGE = (0.01, 20.0)
OPACITY_RANGE = (0.0, 1.0)
ROTATION_RANGE = (-360.0, 360.0)
VOLUME_RANGE = (0.0, 4.0)
PAN_RANGE = (-1.0, 1.0)
POSITION_LIMIT = 20000.0  # pixels from the canvas centre; beyond this the value is a typo, not a design
TEXT_SIZE_RANGE = (6, 500)
MIN_CROP = 0.01
FIT_MODES = ("cover", "contain")  # "stretch" is not allowed by the renderer
TEXT_ALIGNMENTS = ("center", "left")
TEXT_BACKGROUNDS = ("none", "box")
CAPTION_POSITIONS = ("bottom", "center", "top", "custom")

KEYFRAME_RANGES: dict[str, tuple[float, float]] = {
    "position_x": (-POSITION_LIMIT, POSITION_LIMIT), "position_y": (-POSITION_LIMIT, POSITION_LIMIT), "scale": SCALE_RANGE,
    "rotation": (-3600.0, 3600.0),  # additive to the clip rotation; may wind several turns
    "opacity": OPACITY_RANGE, "volume": VOLUME_RANGE, "blur": (0.0, 100.0), "reveal": (0.0, 1.0),
}
VISUAL_KEYFRAME_PROPS = ("position_x", "position_y", "scale", "rotation", "opacity", "blur", "reveal")
AUDIO_KEYFRAME_PROPS = ("volume",)
_PROP_NAME = {"position_x": "horizontal position", "position_y": "vertical position", "scale": "scale", "rotation": "rotation", "opacity": "opacity", "volume": "volume",
              "blur": "blur", "reveal": "reveal"}
KF_CONFLICT_EPS = 1e-6  # two keyframes of one property closer than this are "at the same time"
KF_MATCH_EPS = 1e-4  # how close a requested time must be to an existing keyframe to select it

TRANSITION_TYPES = ("CUT", "FADE", "DISSOLVE", "WIPE", "SLIDE")  # exactly what the compiler renders
LINK_POLICIES = ("keep_linked_audio_in_sync", "independent")
DISSOLVE_ADJACENT = 0.05  # the compiler pairs a dissolve with a clip ending within this many seconds of the new clip's start


class SpeedChoiceRequired(TimelineError):
    """The clip has linked clips: the caller must choose ``link_policy`` (``options``) before the speed can change. ``member_ids`` are the linked clips concerned."""

    def __init__(self, user_message: str, *, options: Sequence[str] = LINK_POLICIES, member_ids: Iterable[str] = ()) -> None:
        super().__init__(user_message)
        self.options = tuple(options)
        self.member_ids = tuple(member_ids)


# ------------------------------------------------------------------------------------------------------------------------ helpers
def clip_category(track: Track, clip: Clip) -> str:
    """``visual`` | ``audio`` (media clips by track) | ``text`` | ``graphic`` | ``caption``."""
    if clip.kind == KIND_MEDIA:
        return "audio" if track.kind is TrackKind.AUDIO else "visual"
    return clip.kind


def _number(label: str, value: Any, lo: float, hi: float, unit: str = "") -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TimelineError(f"{label} must be a number.")
    v = float(value)
    if not math.isfinite(v):
        raise TimelineError(f"{label} must be a finite number.")
    if not (lo - 1e-12 <= v <= hi + 1e-12):
        raise TimelineError(f"{label} must be between {lo:g} and {hi:g}{unit}.")
    return v


def _pair(label: str, value: Any, lo: float, hi: float) -> tuple[float, float]:
    try:
        a, b = value
    except (TypeError, ValueError):
        raise TimelineError(f"{label} needs two numbers (x, y).") from None
    return _number(f"{label} (x)", a, lo, hi), _number(f"{label} (y)", b, lo, hi)


def _empty(description: str, note: str) -> EditPlan:
    return EditPlan(description, notes=[note], major=False)


def _clips(env: PlanEnv, ids: Iterable[str]) -> list[Clip]:
    return [env.timeline.find_clip(cid)[1] for cid in dict.fromkeys(ids)]


def _require_unlocked_tracks(env: PlanEnv, clips: Iterable[Clip], what: str) -> None:
    for c in clips:
        tr = env.timeline.get_track(c.track_id)
        if tr.locked:
            raise TimelineError(f"Track “{tr.name}” is locked, so “{env.label(c)}” cannot be {what}. Unlock the track first.")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ------------------------------------------------------------------------------------------------------------------------ properties
_VISUAL = ("position", "scale", "rotation", "opacity", "crop", "fit")
_AUDIO = ("volume", "muted", "fade_in", "fade_out", "pan")
_TEXT = ("text_content", "text_subtitle", "text_style", "text_position", "text_size", "text_alignment", "text_opacity", "text_background", "text_font", "text_emphasis")
_CAPTION = ("caption_position", "caption_position_xy", "caption_highlight_mode")
_ALLOWED: dict[str, tuple[str, ...]] = {"visual": _VISUAL, "audio": _AUDIO, "text": _TEXT, "caption": _CAPTION, "graphic": ()}
_KNOWN = frozenset(_VISUAL + _AUDIO + _TEXT + _CAPTION)
_LABEL = {"position": "position", "scale": "scale", "rotation": "rotation", "opacity": "opacity", "crop": "crop", "fit": "fit", "volume": "volume", "muted": "mute", "fade_in": "fade in",
          "fade_out": "fade out", "pan": "pan", "text_content": "text", "text_subtitle": "subtitle", "text_style": "text style", "text_position": "text position", "text_size": "text size",
          "text_alignment": "alignment", "text_opacity": "text opacity", "text_background": "background", "text_font": "font", "text_emphasis": "emphasis",
          "caption_position": "caption position", "caption_position_xy": "caption position", "caption_highlight_mode": "highlight mode"}


def _wrong_category(key: str, category: str, name: str) -> str:
    if category == "graphic":
        return f"“{name}” is a graphic highlight; edit it in the Graphics panel. The clip properties here do not apply to it."
    if category == "text" and key in _VISUAL:
        return f"Text is placed with text_position / text_opacity (0–1 of the frame); the clip’s own {key} is not rendered for text."
    if category == "caption" and key in _TEXT + _VISUAL:
        return f"“{key}” cannot be set on a caption here; use caption_position / caption_position_xy / caption_highlight_mode (the words are edited in the caption editor)."
    if category == "audio" and key in _VISUAL:
        return f"“{name}” is an audio clip and has no picture property “{key}”."
    if category == "visual" and key in _AUDIO:
        return f"“{name}” is a picture clip: its own sound is not played, so “{key}” does not apply. Put the sound on an audio track."
    return f"“{key}” cannot be changed on this kind of clip. Available: {', '.join(_ALLOWED.get(category, ())) or 'nothing'}."


def plan_update_clip(env: PlanEnv, clip_id: str, **changes: Any) -> EditPlan:
    """Change several properties of ONE clip as one undo step. Only whitelisted properties that the renderer reads are accepted (see the module docstring); values are
    validated, never clamped. Picture clips: position (x, y px), scale, rotation, opacity, crop ((x, y, w, h) fractions or None), fit. Audio clips: volume, muted, fade_in,
    fade_out, pan. Text clips: text_content / _subtitle / _style / _position / _size / _alignment / _opacity / _background / _font / _emphasis. Captions: caption_position /
    caption_position_xy / caption_highlight_mode."""
    if not changes:
        raise TimelineError("Nothing to change.")
    track, old = env.timeline.find_clip(clip_id)
    cat = clip_category(track, old)
    name = env.label(old)
    for key in changes:
        if key not in _KNOWN:
            raise TimelineError(f"“{key}” is not a clip property that can be edited.")
        if key not in _ALLOWED.get(cat, ()):
            raise TimelineError(_wrong_category(key, cat, name))
    new, notes = old.snapshot(), []
    {"visual": _set_visual, "audio": _set_audio, "text": _set_text, "caption": _set_caption}[cat](env, new, changes, notes)
    if new.to_dict() == old.to_dict():
        return _empty("Edit clip properties", "Nothing changed: the clip already has these values.")
    if changes.get("muted") is not None and len(changes) == 1:
        description = "Mute clip" if changes["muted"] else "Unmute clip"
    else:
        labels = list(dict.fromkeys(_LABEL[k] for k in changes))
        description = "Change " + (", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0])
    return EditPlan(description, {clip_id: new}, notes=notes, major=False)


def _set_visual(env: PlanEnv, new: Clip, ch: dict[str, Any], notes: list[str]) -> None:
    if "position" in ch:
        new.position = _pair("Position", ch["position"], -POSITION_LIMIT, POSITION_LIMIT)
    if "scale" in ch:
        new.scale = _number("Scale", ch["scale"], *SCALE_RANGE)
    if "rotation" in ch:
        new.rotation = _number("Rotation", ch["rotation"], *ROTATION_RANGE, unit="°")
    if "opacity" in ch:
        new.opacity = _number("Opacity", ch["opacity"], *OPACITY_RANGE)
    if "fit" in ch:
        if ch["fit"] not in FIT_MODES:
            raise TimelineError(f"Fit must be one of: {', '.join(FIT_MODES)} (stretching would distort the picture).")
        new.effects["fit"] = ch["fit"]
    if "crop" in ch:
        crop = _crop_value(ch["crop"])
        if crop is None:
            new.effects.pop("crop", None)
        elif crop == [0.0, 0.0, 1.0, 1.0]:
            new.effects.pop("crop", None)
            notes.append("The crop covers the whole picture, so no crop is stored.")
        else:
            new.effects["crop"] = crop


def _crop_value(v: Any) -> list[float] | None:
    """Mirror of the compiler's ``_crop`` (0 <= x,y < 1, 0 < w,h <= 1, inside the picture), plus a minimum size so the crop cannot collapse to zero pixels."""
    if v is None:
        return None
    try:
        x, y, w, h = v
    except (TypeError, ValueError):
        raise TimelineError("A crop needs four numbers (x, y, width, height) as fractions of the picture.") from None
    x, y = _number("Crop x", x, 0.0, 0.99), _number("Crop y", y, 0.0, 0.99)
    w, h = _number("Crop width", w, MIN_CROP, 1.0), _number("Crop height", h, MIN_CROP, 1.0)
    if x + w > 1.0001 or y + h > 1.0001:
        raise TimelineError("The crop must stay inside the picture (x + width and y + height cannot exceed 1).")
    return [x, y, w, h]


def _set_audio(env: PlanEnv, new: Clip, ch: dict[str, Any], notes: list[str]) -> None:
    if "muted" in ch and "volume" in ch:
        raise TimelineError("Set either the volume or mute, not both in one edit.")
    if "volume" in ch:
        new.audio["volume"] = _number("Volume", ch["volume"], *VOLUME_RANGE)
        new.metadata.pop("premute_volume", None)
    if "muted" in ch:
        muted = ch["muted"]
        if not isinstance(muted, bool):
            raise TimelineError("Mute must be on or off.")
        cur = float(new.audio.get("volume", 1.0))
        if muted and cur > 0:
            new.metadata["premute_volume"] = cur
            new.audio["volume"] = 0.0
            notes.append("A clip has no separate mute switch in the mixer, so muting sets its volume to 0 (unmute restores the previous volume).")
        elif not muted and cur <= 0:
            back = float(new.metadata.pop("premute_volume", 1.0))
            new.audio["volume"] = back if back > 0 else 1.0
        elif not muted:
            new.metadata.pop("premute_volume", None)
    for k, label in (("fade_in", "Fade in"), ("fade_out", "Fade out")):
        if k in ch:
            new.audio[k] = _number(label, ch[k], 0.0, new.duration, " s")
    fi, fo = float(new.audio.get("fade_in", 0.0)), float(new.audio.get("fade_out", 0.0))
    if fi + fo > new.duration + 1e-6:
        raise TimelineError(f"The fades ({fi:.2f} s + {fo:.2f} s) are longer than the clip ({new.duration:.2f} s).")
    if "pan" in ch:
        new.audio["pan"] = _number("Pan", ch["pan"], *PAN_RANGE)


def _text_dict(new: Clip) -> dict[str, Any]:
    if new.text is None:
        new.text = {}
    return new.text


def _set_text(env: PlanEnv, new: Clip, ch: dict[str, Any], notes: list[str]) -> None:
    from app.editing.models import Emphasis, TextStyle  # noqa: PLC0415

    t = _text_dict(new)
    if "text_content" in ch:
        s = ch["text_content"]
        if not isinstance(s, str) or not s.strip():
            raise TimelineError("The text cannot be empty.")
        s = s.strip()
        if len(s) > 2000:
            raise TimelineError("The text can have at most 2,000 characters.")
        t["content"] = t["title"] = s
        if t.get("counter"):  # a counting number graphic keeps counting to the NEW figure
            from app.presentation.graphics import counter_spec  # noqa: PLC0415

            spec = counter_spec(s)
            if spec:
                t["counter"] = spec
            else:
                t.pop("counter", None)
                notes.append("The new text is not a plain number, so the counting animation was removed.")
    if "text_subtitle" in ch:
        s = ch["text_subtitle"]
        if not isinstance(s, str) or len(s) > 500:
            raise TimelineError("The subtitle must be text of at most 500 characters.")
        t["subtitle"] = s
    if "text_style" in ch:
        if ch["text_style"] not in {s.value for s in TextStyle}:
            raise TimelineError(f"Unknown text style “{ch['text_style']}”. Choose one of: {', '.join(s.value for s in TextStyle)}.")
        t["style"] = ch["text_style"]
    if "text_position" in ch:
        x, y = _pair("Text position", ch["text_position"], 0.0, 1.0)
        t["position"] = [x, y]
    if "text_size" in ch:
        v = _number("Text size", ch["text_size"], *TEXT_SIZE_RANGE, unit=" px")
        if v != int(v):
            raise TimelineError("The text size must be a whole number of pixels.")
        t["size"] = int(v)
    if "text_alignment" in ch:
        if ch["text_alignment"] not in TEXT_ALIGNMENTS:
            raise TimelineError(f"Alignment must be one of: {', '.join(TEXT_ALIGNMENTS)}.")
        t["alignment"] = ch["text_alignment"]
    if "text_opacity" in ch:
        t["opacity"] = _number("Text opacity", ch["text_opacity"], *OPACITY_RANGE)
    if "text_background" in ch:
        if ch["text_background"] not in TEXT_BACKGROUNDS:
            raise TimelineError(f"Background must be one of: {', '.join(TEXT_BACKGROUNDS)}.")
        t["background"] = ch["text_background"]
    if "text_font" in ch:
        f = ch["text_font"]
        if not isinstance(f, str) or not f.strip() or len(f) > 120:
            raise TimelineError("Choose a font name (at most 120 characters).")
        t["font"] = f.strip()
    if "text_emphasis" in ch:
        if ch["text_emphasis"] not in ("", *(e.value for e in Emphasis)):
            raise TimelineError(f"Unknown emphasis “{ch['text_emphasis']}”. Choose one of: {', '.join(e.value for e in Emphasis)}.")
        t["emphasis"] = ch["text_emphasis"]


def _set_caption(env: PlanEnv, new: Clip, ch: dict[str, Any], notes: list[str]) -> None:
    from app.presentation.models import HighlightMode  # noqa: PLC0415

    t = _text_dict(new)
    if "caption_position" in ch:
        pos = ch["caption_position"]
        if pos not in CAPTION_POSITIONS:
            raise TimelineError(f"Caption position must be one of: {', '.join(CAPTION_POSITIONS)}.")
        if pos == "custom" and "caption_position_xy" not in ch and len(t.get("position_xy") or []) != 2:
            raise TimelineError("A custom caption position needs the point to place it at (caption_position_xy).")
        if pos != "custom" and "caption_position_xy" in ch:
            raise TimelineError("A point can only be given together with the position “custom”.")
        t["position"] = pos
        if pos != "custom":
            t["position_xy"] = []
    if "caption_position_xy" in ch:
        x, y = _pair("Caption position", ch["caption_position_xy"], 0.0, 1.0)
        t["position"], t["position_xy"] = "custom", [x, y]
    if "caption_highlight_mode" in ch:
        if ch["caption_highlight_mode"] not in {m.value for m in HighlightMode}:
            raise TimelineError(f"Highlight mode must be one of: {', '.join(m.value for m in HighlightMode)}.")
        t["highlight_mode"] = ch["caption_highlight_mode"]


# ------------------------------------------------------------------------------------------------------------------------ speed
def _speed_value(speed: Any) -> float:
    if isinstance(speed, bool) or not isinstance(speed, numbers.Real) or not math.isfinite(float(speed)):
        raise TimelineError("Speed must be a number.")
    s = float(speed)
    if s <= 0:
        raise TimelineError(f"Reverse playback and freezing (speed {s:g}) are not available. Use a speed between {SPEED_RANGE[0]:g}× and {SPEED_RANGE[1]:g}×.")
    if not (SPEED_RANGE[0] - 1e-9 <= s <= SPEED_RANGE[1] + 1e-9):
        raise TimelineError(f"Speed must be between {SPEED_RANGE[0]:g}× and {SPEED_RANGE[1]:g}×.")
    return s


def _speed_problem(env: PlanEnv, clip: Clip) -> str:
    """Why this clip has no playback speed ("" when it has one)."""
    if clip.kind != KIND_MEDIA:
        return f"“{env.label(clip)}” is not a video or audio clip, so it has no playback speed."
    info = env.asset_info(clip.asset_id)
    if info is not None and info[0] == "image":
        return f"“{env.label(clip)}” is a picture; it has no playback speed. Change its length instead."
    return ""


def _respeed(env: PlanEnv, clip: Clip, s: float, notes: list[str]) -> Clip:
    """The clip at speed ``s``: same source range, duration = source span / speed. Rejects when it would overlap the next clip or become too short."""
    tl = env.timeline
    name = env.label(clip)
    why = _speed_problem(env, clip)
    if why:
        raise TimelineError(why)
    span = clip.source_out - clip.source_in
    if span <= 1e-9:
        raise TimelineError(f"“{name}” has no source range, so its speed cannot be changed.")
    track = tl.get_track(clip.track_id)
    new = clip.snapshot()
    new.speed, new.duration = s, span / s
    if new.duration < MIN_CLIP_DURATION - TIME_EPSILON:
        raise TimelineError(f"At {s:g}× “{name}” would be only {new.duration:.3f} s long (the shortest clip is {MIN_CLIP_DURATION:g} s). Choose a lower speed.")
    _prev_end, next_start = tl.neighbours(clip)
    if next_start is not None and clip.timeline_start + new.duration > next_start + TIME_EPSILON:
        room = max(next_start - clip.timeline_start, TIME_EPSILON)
        raise TimelineError(f"At {s:g}× “{name}” would be {new.duration:.2f} s long, but only {room:.2f} s are free before the next clip on track “{track.name}” "
                            f"(the slowest possible speed here is {span / room:.2f}×). Trim or move the next clip first.")
    d0, d1 = clip.duration, new.duration
    is_audio = track.kind is TrackKind.AUDIO
    if new.keyframes and abs(d1 - d0) > TIME_EPSILON:
        if is_audio:  # volume automation is tied to what happens on the timeline (ducking under speech), so it stays where it is
            if d1 < d0:
                new.keyframes = window_keyframes(new.keyframes, 0.0, d1)
                notes.append(f"Volume keyframes after the new end of “{name}” were removed (the value at the end is kept).")
        else:  # motion belongs to the picture: it stretches with the clip
            for k in new.keyframes:
                k.time = k.time * d1 / d0
            notes.append(f"The keyframes of “{name}” were stretched with the clip.")
    if is_audio:
        fi, fo = float(new.audio.get("fade_in", 0.0)), float(new.audio.get("fade_out", 0.0))
        if fi + fo > d1 + 1e-6:
            k = d1 / (fi + fo)
            new.audio["fade_in"], new.audio["fade_out"] = fi * k, fo * k
            notes.append(f"The fades of “{name}” were shortened to fit the new length.")
    elif new.transition and float(new.transition.get("duration", 0.0)) > d1 + 1e-6:
        new.transition["duration"] = d1
        notes.append(f"The transition into “{name}” was shortened to the new clip length.")
    if not is_audio and not env.timebase.is_on_frame(d1):
        notes.append(f"“{name}” is now {d1:.4f} s long, which is {d1 * env.timebase.fps:.2f} frames (not a whole number); trim it to a frame boundary if you need one.")
    if is_audio:
        voice = str(clip.audio.get("role", "")).upper() == "VOICE" or clip.slot == "voice"
        notes.append(f"“{name}” is {'the voice-over' if voice else 'an audio clip'}: its playback tempo changes with the speed, so it will sound {'faster' if s > clip.speed else 'slower'}.")
    return new


def plan_set_speed(env: PlanEnv, clip_id: str, speed: float, *, linked: bool = True, link_policy: str | None = None) -> EditPlan:
    """Change a clip's playback speed (``SPEED_RANGE``, never reverse). The source range stays; the duration becomes (source_out - source_in) / speed, limited by the next clip.

    Clips linked to it (``link_id``): with ``linked=True`` and members whose speed would differ the caller MUST pass ``link_policy`` — ``"keep_linked_audio_in_sync"`` gives
    every speed-capable member the same speed, ``"independent"`` changes only this clip (a note says sync may break) — otherwise ``SpeedChoiceRequired`` is raised with both
    options. ``linked=False`` means "only this clip" (same as ``independent``). Only the named clip and its explicit linked members are ever touched."""
    s = _speed_value(speed)
    if link_policy is not None and link_policy not in LINK_POLICIES:
        raise TimelineError(f"Unknown link choice “{link_policy}”. Choose one of: {', '.join(LINK_POLICIES)}.")
    if not linked and link_policy == "keep_linked_audio_in_sync":
        raise TimelineError("The speed cannot both ignore linked clips (linked=False) and keep them in sync.")
    tl = env.timeline
    _track, clip = tl.find_clip(clip_id)
    mates = [m for m in link_members(tl, clip.link_id) if m.id != clip.id] if clip.link_id else []
    notes: list[str] = []
    follow: list[Clip] = []
    if clip.speed and abs(clip.speed - s) <= 1e-9 and not (linked and link_policy == "keep_linked_audio_in_sync"):
        return _empty(f"Change speed to {s:g}×", f"“{env.label(clip)}” already plays at {s:g}×.")
    if mates and linked and link_policy != "independent":
        capable = [m for m in mates if not _speed_problem(env, m)]
        differing = [m for m in capable if abs(m.speed - s) > 1e-9]
        if link_policy is None and differing:
            raise SpeedChoiceRequired(
                f"“{env.label(clip)}” is linked to {_plural(len(mates), 'other clip')} (for example “{env.label(mates[0])}”). Choose how to change the speed: keep the linked clips in sync "
                f"(they get the same {s:g}× speed) or change only this clip (the sync with them may break).", member_ids=[m.id for m in differing])
        follow = differing if link_policy == "keep_linked_audio_in_sync" else []
        capable_ids = {m.id for m in capable}
        left = [m for m in mates if m.id not in capable_ids]
        if left and link_policy == "keep_linked_audio_in_sync":
            notes.append(f"{_plural(len(left), 'linked clip')} that cannot change speed (pictures or text) kept their length.")
    elif mates:
        notes.append(f"The speed of {_plural(len(mates), 'linked clip')} was not changed, so they may drift out of sync with “{env.label(clip)}”.")
    updates: dict[str, Clip] = {}
    if abs(clip.speed - s) > 1e-9:
        updates[clip_id] = _respeed(env, clip, s, notes)
    for m in follow:
        updates[m.id] = _respeed(env, m, s, notes)
    if follow:
        notes.append(f"{_plural(len(follow), 'linked clip')} changed to {s:g}× as well.")
    if not updates:
        return _empty(f"Change speed to {s:g}×", "Nothing changed: every clip already plays at that speed.")
    return EditPlan(f"Change speed to {s:g}×", updates, notes=notes)


# ------------------------------------------------------------------------------------------------------------------------ transitions
def _require_picture(env: PlanEnv, clip_id: str) -> tuple[Track, Clip]:
    track, clip = env.timeline.find_clip(clip_id)
    if clip.kind != KIND_MEDIA or track.kind is TrackKind.AUDIO:
        raise TimelineError(f"“{env.label(clip)}” is not a picture clip. Transitions lead into video and image clips only.")
    return track, clip


def dissolve_source(env: PlanEnv, clip: Clip) -> Clip | None:
    """The clip the renderer holds and fades out under a DISSOLVE into ``clip`` (same rule as ``TimelineCompiler._hold_layer``): the media clip on a visible picture track that
    ends within 0.05 s of the clip's start (the latest end wins), provided its media is a video or image. ``None`` when there is none (the renderer would only fade in)."""
    best: Clip | None = None
    for t in env.timeline.tracks:
        if t.kind is TrackKind.AUDIO or t.hidden:
            continue
        for o in t.clips:
            if o.kind == KIND_MEDIA and o.id != clip.id and abs(o.timeline_end - clip.timeline_start) < DISSOLVE_ADJACENT and (best is None or o.timeline_end > best.timeline_end):
                best = o
    if best is None:
        return None
    info = env.asset_info(best.asset_id)
    return best if info is not None and info[0] in ("video", "image") else None


def plan_set_transition(env: PlanEnv, clip_id: str, type: str | None, duration: float | None = None, params: dict | None = None) -> EditPlan:
    """Set the transition INTO a picture clip: ``FADE`` | ``DISSOLVE`` | ``WIPE`` | ``SLIDE`` (``CUT`` / ``None`` remove it). ``duration`` (seconds, > 0, at most the clip length)
    is required for the visible types. A DISSOLVE needs a picture that ends where this clip starts (the renderer holds its last frame); otherwise the request is rejected rather
    than silently rendered as a fade. The renderer reads no further options, so ``params`` must be empty."""
    _track, clip = _require_picture(env, clip_id)
    if type is None:
        return plan_remove_transition(env, clip_id)
    kind = type.strip().upper() if isinstance(type, str) else ""
    if kind not in TRANSITION_TYPES:
        raise TimelineError(f"Unknown transition “{type}”. Available: {', '.join(TRANSITION_TYPES)}.")
    if params:
        raise TimelineError(f"A {kind.lower()} transition has no options ({', '.join(sorted(map(str, params)))} would be ignored by the renderer).")
    if kind == "CUT":
        return plan_remove_transition(env, clip_id)
    if duration is None:
        raise TimelineError(f"A {kind.lower()} needs a duration.")
    if isinstance(duration, bool) or not isinstance(duration, numbers.Real) or not math.isfinite(float(duration)):
        raise TimelineError(f"The {kind.lower()} duration must be a number of seconds.")
    d, frame = float(duration), env.timebase.frame_s
    if d < frame - 1e-9:
        raise TimelineError(f"The {kind.lower()} must last at least one frame ({frame:.3f} s).")
    if d > clip.duration + TIME_EPSILON:
        raise TimelineError(f"The {kind.lower()} cannot be longer than the clip ({clip.duration:.2f} s).")
    notes: list[str] = []
    if kind == "DISSOLVE":
        prev = dissolve_source(env, clip)
        if prev is None:
            raise TimelineError(f"A dissolve needs a picture that ends exactly where “{env.label(clip)}” starts (its last frame is faded out). There is none here: use a fade or move the clips together.")
        notes.append(f"The dissolve holds the last frame of “{env.label(prev)}” while this clip fades in.")
    elif kind == "WIPE":
        notes.append("The wipe always reveals the clip from left to right.")
    elif kind == "SLIDE":
        notes.append("The slide always enters from the right edge.")
    new = clip.snapshot()
    new.transition = {"type": kind, "duration": d}  # a manual transition: no AI decision owns it
    if new.to_dict() == clip.to_dict():
        return _empty(f"Set {kind.lower()} transition", "The clip already has this transition.")
    return EditPlan(f"Set {kind.lower()} transition", {clip_id: new}, notes=notes)


def plan_remove_transition(env: PlanEnv, clip_id: str) -> EditPlan:
    _track, clip = _require_picture(env, clip_id)
    if clip.transition is None:
        return _empty("Remove transition", "This clip has no transition.")
    new = clip.snapshot()
    new.transition = None
    return EditPlan("Remove transition", {clip_id: new})


# ------------------------------------------------------------------------------------------------------------------------ keyframes
def keyframe_properties(track: Track, clip: Clip) -> tuple[str, ...]:
    """The properties that can be keyframed (and are rendered) on this clip: picture clips animate transform / opacity / blur / reveal, audio clips volume. Text, captions and
    graphics are animated by presets (``clip.animation``), not keyframes — the export does not read their keyframes."""
    cat = clip_category(track, clip)
    return VISUAL_KEYFRAME_PROPS if cat == "visual" else AUDIO_KEYFRAME_PROPS if cat == "audio" else ()


def _kf_target(env: PlanEnv, clip_id: str) -> tuple[Track, Clip, tuple[str, ...]]:
    track, clip = env.timeline.find_clip(clip_id)
    props = keyframe_properties(track, clip)
    if not props:
        raise TimelineError(f"Keyframes are only rendered for picture and audio clips; “{env.label(clip)}” is animated with animation presets instead.")
    return track, clip, props


def _kf_prop(prop: Any, props: tuple[str, ...], clip: Clip, env: PlanEnv) -> str:
    if prop not in PROPERTIES:
        raise TimelineError(f"“{prop}” is not a property that can be animated.")
    if prop not in props:
        what = "audio clips only have volume keyframes" if props == AUDIO_KEYFRAME_PROPS else "picture clips have no volume keyframes (their sound is not played)"
        raise TimelineError(f"{_PROP_NAME[prop].capitalize()} cannot be animated on “{env.label(clip)}”: {what}.")
    return prop


def _kf_interp(interp: Any) -> str:
    if interp not in INTERPOLATIONS:
        raise TimelineError(f"Unknown interpolation “{interp}”. Choose one of: {', '.join(INTERPOLATIONS)} (hold / step keyframes cannot be rendered, so they are not offered).")
    return interp


def _kf_time(clip: Clip, t: Any, what: str = "A keyframe") -> float:
    if isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(float(t)):
        raise TimelineError(f"{what} needs a valid time.")
    t = float(t)
    if t < -TIME_EPSILON or t > clip.duration + TIME_EPSILON:
        raise TimelineError(f"{what} at {t:.2f} s is outside the clip (0 – {clip.duration:.2f} s).")
    return min(max(t, 0.0), clip.duration)


def _kf_value(prop: str, v: Any) -> float:
    return _number(f"The {_PROP_NAME[prop]} value", v, *KEYFRAME_RANGES[prop])


def _kf_sorted(kfs: Iterable[Keyframe]) -> list[Keyframe]:
    order = {p: i for i, p in enumerate(PROPERTIES)}
    return sorted(kfs, key=lambda k: (order.get(k.property, len(order)), k.time))


def _kf_find(kfs: Iterable[Keyframe], prop: str, t: float) -> Keyframe | None:
    best = min((k for k in kfs if k.property == prop), key=lambda k: abs(k.time - t), default=None)
    return best if best is not None and abs(best.time - t) <= KF_MATCH_EPS else None


def _kf_pick(kfs: list[Keyframe], prop: Any, t: Any, clip: Clip) -> Keyframe:
    if isinstance(t, bool) or not isinstance(t, numbers.Real):
        raise TimelineError("A keyframe is selected by its property and time.")
    k = _kf_find(kfs, prop, float(t))
    if k is None:
        raise TimelineError(f"There is no {_PROP_NAME.get(prop, prop)} keyframe at {float(t):.2f} s.")
    return k


def _kf_assert_unique(kfs: Iterable[Keyframe]) -> None:
    by_prop: dict[str, list[float]] = {}
    for k in kfs:
        by_prop.setdefault(k.property, []).append(k.time)
    for prop, times in by_prop.items():
        times.sort()
        for a, b in zip(times, times[1:]):
            if b - a < KF_CONFLICT_EPS:
                raise TimelineError(f"There would be two {_PROP_NAME.get(prop, prop)} keyframes at {a:.2f} s. Move or delete one of them first.")


def plan_add_keyframe(env: PlanEnv, clip_id: str, prop: str, time: float, value: float, interpolation: str = "linear") -> EditPlan:
    """Add one keyframe (clip-local ``time`` in 0..duration). Rejected when the property already has a keyframe at that time."""
    _track, clip, props = _kf_target(env, clip_id)
    prop = _kf_prop(prop, props, clip, env)
    interp = _kf_interp(interpolation)
    t = _kf_time(clip, time)
    v = _kf_value(prop, value)
    if any(k.property == prop and abs(k.time - t) < KF_CONFLICT_EPS for k in clip.keyframes):
        raise TimelineError(f"There is already a {_PROP_NAME[prop]} keyframe at {t:.2f} s. Move or delete it first.")
    new = clip.snapshot()
    new.keyframes = _kf_sorted([*new.keyframes, Keyframe(prop, t, v, interp, "")])
    return EditPlan("Add keyframe", {clip_id: new}, major=False)


def plan_delete_keyframes(env: PlanEnv, clip_id: str, selection: Iterable[tuple[str, float]]) -> EditPlan:
    """Delete the keyframes selected by (property, time)."""
    _track, clip, _props = _kf_target(env, clip_id)
    new = clip.snapshot()
    doomed: list[Keyframe] = []
    for prop, t in list(selection):
        k = _kf_pick(new.keyframes, prop, t, clip)
        if not any(k is d for d in doomed):
            doomed.append(k)
    if not doomed:
        raise TimelineError("Select at least one keyframe.")
    new.keyframes = _kf_sorted(k for k in new.keyframes if not any(k is d for d in doomed))
    return EditPlan("Delete keyframe" if len(doomed) == 1 else f"Delete {len(doomed)} keyframes", {clip_id: new}, major=False)


def plan_move_keyframes(env: PlanEnv, clip_id: str, moves: Iterable[Sequence[Any]]) -> EditPlan:
    """Move / re-value several keyframes at once: ``moves`` = (property, old_time, new_time, new_value | None). The RESULT is checked, so keyframes can swap places in one edit,
    but two keyframes of one property may never end up at the same time. An edited keyframe stops belonging to an AI decision (manual override)."""
    _track, clip, props = _kf_target(env, clip_id)
    new = clip.snapshot()
    picked: list[tuple[Keyframe, float, float | None]] = []
    for mv in list(moves):
        if len(mv) not in (3, 4):
            raise TimelineError("A keyframe move is (property, old time, new time, new value).")
        prop, old_t, new_t = mv[0], mv[1], mv[2]
        new_v = mv[3] if len(mv) == 4 else None
        _kf_prop(prop, props, clip, env)
        k = _kf_pick(new.keyframes, prop, old_t, clip)
        if any(k is p[0] for p in picked):
            raise TimelineError(f"The {_PROP_NAME[prop]} keyframe at {k.time:.2f} s is moved twice in one edit.")
        picked.append((k, _kf_time(clip, new_t, f"The moved {_PROP_NAME[prop]} keyframe"), None if new_v is None else _kf_value(prop, new_v)))
    if not picked:
        raise TimelineError("Select at least one keyframe.")
    changed = 0
    for k, nt, nv in picked:
        if abs(nt - k.time) > 1e-9 or (nv is not None and abs(nv - k.value) > 1e-12):
            k.time = nt
            if nv is not None:
                k.value = nv
            k.decision_id = ""
            changed += 1
    if not changed:
        return _empty("Move keyframe", "The keyframes are already there.")
    _kf_assert_unique(new.keyframes)
    new.keyframes = _kf_sorted(new.keyframes)
    return EditPlan("Move keyframe" if len(picked) == 1 else f"Move {len(picked)} keyframes", {clip_id: new}, major=False)


def plan_set_keyframe_interpolation(env: PlanEnv, clip_id: str, selection: Iterable[tuple[str, float]], interp: str) -> EditPlan:
    """Set how the value travels from each selected keyframe to the next (``keyframes.INTERPOLATIONS``)."""
    _track, clip, _props = _kf_target(env, clip_id)
    interp = _kf_interp(interp)
    new = clip.snapshot()
    chosen: list[Keyframe] = []
    for prop, t in list(selection):
        k = _kf_pick(new.keyframes, prop, t, clip)
        if not any(k is c for c in chosen):
            chosen.append(k)
    if not chosen:
        raise TimelineError("Select at least one keyframe.")
    todo = [k for k in chosen if k.interpolation != interp]
    if not todo:
        return _empty("Change interpolation", f"The selected keyframes already use “{interp}”.")
    for k in todo:
        k.interpolation, k.decision_id = interp, ""
    return EditPlan("Change keyframe interpolation", {clip_id: new}, major=False)


def plan_copy_keyframes(clip: Clip, selection: Iterable[tuple[str, float]] | None = None) -> list[Keyframe]:
    """Copies of the selected keyframes (all of them when ``selection`` is None), sorted, owned by the user. Pure: nothing changes; paste with ``plan_paste_keyframes``."""
    if selection is None:
        picked = list(clip.keyframes)
    else:
        picked = []
        for prop, t in list(selection):
            k = _kf_pick(clip.keyframes, prop, t, clip)
            if not any(k is p for p in picked):
                picked.append(k)
    return _kf_sorted(Keyframe(k.property, k.time, k.value, k.interpolation, "") for k in picked)


def plan_paste_keyframes(env: PlanEnv, clip_id: str, keyframes: Iterable[Keyframe | dict], at_time: float) -> EditPlan:
    """Paste copied keyframes so the earliest lands at ``at_time`` (clip-local), keeping their spacing. Properties the target cannot animate are skipped with a note (rejected when
    none fit); a paste that would run past the end of the clip, or land on an existing keyframe of the same property, is rejected."""
    _track, clip, props = _kf_target(env, clip_id)
    items = [k if isinstance(k, Keyframe) else Keyframe.from_dict(k) for k in keyframes]
    if not items:
        raise TimelineError("There are no copied keyframes to paste.")
    usable = [k for k in items if k.property in props]
    notes: list[str] = []
    skipped = sorted({k.property for k in items if k.property not in props})
    if not usable:
        raise TimelineError(f"None of the copied keyframes ({', '.join(_PROP_NAME.get(p, p) for p in skipped)}) can be pasted on “{env.label(clip)}”.")
    if skipped:
        notes.append(f"Skipped keyframes this clip cannot animate: {', '.join(_PROP_NAME.get(p, p) for p in skipped)}.")
    at = _kf_time(clip, at_time, "The paste position")
    base = min(k.time for k in usable)
    new = clip.snapshot()
    added: list[Keyframe] = []
    for k in usable:
        t = at + (k.time - base)
        if t > clip.duration + TIME_EPSILON:
            raise TimelineError(f"The pasted keyframes would run {t - clip.duration:.2f} s past the end of the clip. Paste earlier or on a longer clip.")
        t = min(t, clip.duration)
        if any(e.property == k.property and abs(e.time - t) < KF_CONFLICT_EPS for e in new.keyframes):
            raise TimelineError(f"There is already a {_PROP_NAME[k.property]} keyframe at {t:.2f} s. Delete it or paste somewhere else.")
        added.append(Keyframe(k.property, t, _kf_value(k.property, k.value), _kf_interp(k.interpolation), ""))
    _kf_assert_unique([*new.keyframes, *added])
    new.keyframes = _kf_sorted([*new.keyframes, *added])
    return EditPlan("Paste keyframe" if len(added) == 1 else f"Paste {len(added)} keyframes", {clip_id: new}, notes=notes, major=False)


# ------------------------------------------------------------------------------------------------------------------------ links / groups / locks
def _relate(env: PlanEnv, ids: Iterable[str], attr: str, make_id, given: str | None, noun: str, verb: str) -> EditPlan:
    tl = env.timeline
    clips = _clips(env, ids)
    if len(clips) < 2:
        raise TimelineError(f"Select at least two clips to {verb}.")
    old = {getattr(c, attr) for c in clips if getattr(c, attr)}
    involved = {c.id: c for c in clips}
    if old:  # clips that were already related to a selected clip join the new relationship, so no old one is torn apart
        for c in tl.all_clips():
            if getattr(c, attr) in old:
                involved.setdefault(c.id, c)
    _require_unlocked_tracks(env, involved.values(), f"{verb}ed")
    if len(old) == 1 and set(involved) == {c.id for c in clips} and all(getattr(c, attr) in old for c in clips) and given in (None, next(iter(old))):
        return _empty(f"{verb.capitalize()} clips", f"These clips are already {verb}ed.")
    rid = given or make_id()
    updates: dict[str, Clip] = {}
    for cid, c in involved.items():
        if getattr(c, attr) != rid:
            n = c.snapshot()
            setattr(n, attr, rid)
            updates[cid] = n
    notes = []
    extra = len(involved) - len(clips)
    if extra:
        notes.append(f"{_plural(extra, 'clip')} already {verb}ed to the selection joined the new {noun}.")
    return EditPlan(f"{verb.capitalize()} {_plural(len(involved), 'clip')}", updates, notes=notes)


def _release(env: PlanEnv, ids: Iterable[str], attr: str, members, include_mates: bool, noun: str, verb: str) -> EditPlan:
    tl = env.timeline
    clips = _clips(env, ids)
    targets = {c.id: c for c in clips if getattr(c, attr)}
    if include_mates:
        for rid in {getattr(c, attr) for c in targets.values()}:
            for m in members(tl, rid):
                targets.setdefault(m.id, m)
    if not targets:
        return _empty(f"{verb.capitalize()} clips", f"None of these clips has a {noun}.")
    _require_unlocked_tracks(env, targets.values(), f"{verb}ed")
    updates = {}
    for cid, c in targets.items():
        n = c.snapshot()
        setattr(n, attr, "")
        updates[cid] = n
    return EditPlan(f"{verb.capitalize()} {_plural(len(updates), 'clip')}", updates)


def plan_link(env: PlanEnv, ids: Iterable[str], *, link_id: str | None = None) -> EditPlan:
    """Link two or more clips (a picture and its production audio): they then follow each other's move / trim / split / delete / ripple. Clips already linked to a selected clip
    join the new link. Clips on locked tracks are rejected. ``link_id`` is optional (tests / callers that need a stable id)."""
    return _relate(env, ids, "link_id", new_link_id, link_id, "link", "link")


def plan_unlink(env: PlanEnv, ids: Iterable[str], *, include_mates: bool = False) -> EditPlan:
    """Remove the link from the given clips (and from every other member of their links when ``include_mates``). Members left alone keep their id (a link of one is harmless)."""
    return _release(env, ids, "link_id", link_members, include_mates, "link", "unlink")


def plan_group(env: PlanEnv, ids: Iterable[str], *, group_id: str | None = None) -> EditPlan:
    """Group two or more clips: they are selected and moved together, but trimmed and split independently."""
    return _relate(env, ids, "group_id", new_group_id, group_id, "group", "group")


def plan_ungroup(env: PlanEnv, ids: Iterable[str], *, include_mates: bool = False) -> EditPlan:
    return _release(env, ids, "group_id", group_members, include_mates, "group", "ungroup")


def plan_set_locks(env: PlanEnv, ids: Iterable[str], locks: Iterable[str] | str, mode: str = "set") -> EditPlan:
    """Edit the user locks (``LOCK_KINDS``) of clips: ``set`` replaces them, ``add`` / ``remove`` change the given kinds only. Works on clips that are already locked (changing a
    lock is not a locked aspect); a locked TRACK still has to be unlocked first."""
    if mode not in ("set", "add", "remove"):
        raise TimelineError("Lock mode must be set, add or remove.")
    kinds = frozenset({locks} if isinstance(locks, str) else locks)
    bad = sorted(kinds - set(LOCK_KINDS))
    if bad:
        raise TimelineError(f"Unknown lock “{bad[0]}”. Choose from: {', '.join(LOCK_KINDS)}.")
    clips = _clips(env, ids)
    if not clips:
        raise TimelineError("Select at least one clip.")
    _require_unlocked_tracks(env, clips, "locked or unlocked")
    updates: dict[str, Clip] = {}
    for c in clips:
        target = kinds if mode == "set" else (c.locks | kinds) if mode == "add" else (c.locks - kinds)
        if target != c.locks:
            n = c.snapshot()
            n.locks = frozenset(target)
            updates[c.id] = n
    if not updates:
        return _empty("Change locks", "The locks are already as requested.")
    unlock = mode == "remove" or (mode == "set" and not kinds)
    return EditPlan(("Unlock " if unlock else "Lock ") + _plural(len(updates), "clip"), updates)
