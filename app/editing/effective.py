"""Effective settings: the user's own settings with the applied reference style (``EditingStrategyOverrides``) laid over them, for one engine run.

    project settings  +  project.reference_style_overrides  ->  effective settings  ->  Phase 4 / Phase 5 engines

This is the single place where a reference style changes what the editing engines do. Everything here is *soft*: a target shot length, a motion level, a text
budget, caption length / style / position, a music level, ducking, pause usage, a number of sound effects. It never touches content: no footage, wording,
timing of the narration, locked objects or user-owned objects. Whatever the user set on purpose (``user_set`` on the editing / caption / audio settings) is left
exactly as it is while "keep my own settings" is on. With no applied style every function returns its input unchanged, so the engines behave exactly as before.

The project's stored settings are never modified: the effective copy lives only for the run (``EditingSettings.reference`` is filled in on the copy).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from app.editing.models import EditingSettings
from app.editing.overrides import PROTECTED_BY, EditingStrategyOverrides

if TYPE_CHECKING:  # pragma: no cover
    from app.editing.presets import StylePreset
    from app.presentation.models import AudioSettings, CaptionSettings
    from app.project.project import Project

DUCK_FULL_DB = 12.0  # ducking_strength 1.0 = the music drops by 12 dB under speech (the scale the reference analyzer measures on)
PAUSE_LEVEL_GAIN = 0.66  # pause_usage 1.0 raises the music in a pause to 1.66 x its normal level (0.5 = 1.33 x, the default)
TEXT_BUDGET_FLOOR = 0.1  # text_density 0 still leaves a tenth of the preset's text budget (the engine always shows at least what the narration needs)


# ---------------------------------------------------------------------------------------------- level maths (shared with the adapter)
def ducking_strength(music_level: float, duck_level: float) -> float:
    """0..1 from the level the music drops to under important speech (12 dB or more = 1.0)."""
    if music_level <= 0:
        return 0.0
    if duck_level <= 0:
        return 1.0
    return max(0.0, min(1.0, 20.0 * math.log10(max(music_level, duck_level) / duck_level) / DUCK_FULL_DB))


def duck_level_for(music_level: float, strength: float) -> float:
    """Inverse of ``ducking_strength``: the ducked level that gives this strength."""
    return music_level * 10.0 ** (-max(0.0, min(1.0, strength)) * DUCK_FULL_DB / 20.0)


def pause_level_for(music_level: float, usage: float) -> float:
    return music_level * (1.0 + PAUSE_LEVEL_GAIN * max(0.0, min(1.0, usage)))


# ---------------------------------------------------------------------------------------------- which parameters the user's own settings protect
def protected_parameters(user_set: dict[str, list[str]], preserve: bool) -> set[str]:
    """Override parameters whose setting the user changed on purpose (empty when the user allowed the style to override everything)."""
    if not preserve:
        return set()
    return {param for param, (kind, name) in PROTECTED_BY.items() if name in (user_set.get(kind) or ())}


def user_set_of(project: "Project") -> dict[str, list[str]]:
    return {"editing": list(project.editing_settings.user_set), "caption": list(project.caption_settings.user_set), "audio": list(project.audio_settings.user_set)}


def active_overrides(project: "Project") -> EditingStrategyOverrides | None:
    """The applied style as the engines may use it right now: ``None`` when no style is applied, otherwise the stored overrides minus every parameter the user's
    own settings protect (checked at run time, so a setting changed after the style was applied still wins)."""
    rs, ov = project.reference_settings, project.reference_style_overrides
    if not rs.enabled or ov.is_empty:
        return None
    protected = protected_parameters(user_set_of(project), rs.preserve_user_edits) & set(ov.active())
    out = ov.without(protected) if protected else deepcopy(ov)
    return None if out.is_empty else out


def style_signature(project: "Project") -> str:
    """Fingerprint of the style the engines will use (``""`` when none): part of every cache key that depends on it."""
    ov = active_overrides(project)
    return ov.signature() if ov is not None else ""


# ---------------------------------------------------------------------------------------------- Phase 4
def effective_editing_settings(project: "Project", settings: EditingSettings | None = None) -> EditingSettings:
    """A copy of the editing settings for one engine run. Sliders the style sets (motion, transitions) are replaced; the rest travels in ``.reference`` and is read
    by ``preset_for`` (shot lengths, text budget, music level) and the shot timing (the opening)."""
    out = deepcopy(settings if settings is not None else project.editing_settings)
    out.reference = None
    ov = active_overrides(project)
    if ov is None:
        return out
    out.reference = ov
    if ov.motion_intensity is not None:
        out.motion_intensity = max(0.0, min(1.0, ov.motion_intensity))
    if ov.transition_frequency is not None:
        out.transition_frequency = max(0.0, min(1.0, ov.transition_frequency))
    return out


def preset_with_reference(preset: "StylePreset", settings: EditingSettings) -> "StylePreset":
    """The editing preset with the style's shot lengths, text budget and music level (``preset_for`` calls this when ``settings.reference`` carries values)."""
    from app.editing.presets import shot_factor  # noqa: PLC0415  (presets import this module lazily; neither imports the other at load time)

    ov = settings.reference
    if ov is None or ov.is_empty:
        return preset
    changes: dict[str, float] = {}
    if ov.target_shot_duration is not None:
        changes["base_shot"] = max(0.3, ov.target_shot_duration) / max(shot_factor(settings), 0.1)  # the pacing slider multiplies it again: the product is the target
    if ov.min_shot_duration is not None:
        changes["min_shot"] = max(0.5, ov.min_shot_duration)
    if ov.max_shot_duration is not None:
        changes["max_shot"] = ov.max_shot_duration
    if ov.text_density is not None:
        changes["text_per_minute"] = preset.text_per_minute * max(TEXT_BUDGET_FLOOR, ov.text_density / 0.5)
    if ov.music_level is not None:
        changes["music_level"] = ov.music_level
    if not changes:
        return preset
    out = replace(preset, **changes)
    if out.min_shot > out.max_shot:
        out = replace(out, min_shot=out.max_shot)
    return out


def hook_factor(settings: EditingSettings, scene_start: float) -> float:
    """< 1 for scenes inside the opening the style asks to cut faster; 1.0 otherwise (and always 1.0 without a style)."""
    ov = settings.reference
    if ov is None or not ov.hook_seconds or not ov.hook_shot_factor or scene_start >= ov.hook_seconds:
        return 1.0
    return float(ov.hook_shot_factor)


def reference_hash_part(settings: EditingSettings) -> list[str]:
    """What a scene's cache key gains from an applied style: nothing (an empty list) without one."""
    ov = settings.reference
    return [ov.signature()] if ov is not None and not ov.is_empty else []


# ---------------------------------------------------------------------------------------------- Phase 5
def effective_caption_settings(project: "Project") -> "CaptionSettings":
    """Caption settings for one generation run: style, position and words per caption from the applied style. The stored settings stay as the user left them."""
    from app.captions.styles import PRESETS  # noqa: PLC0415

    out = deepcopy(project.caption_settings)
    ov = active_overrides(project)
    if ov is None:
        return out
    if ov.caption_style and (ov.caption_style in PRESETS or ov.caption_style in project.caption_styles):
        out.style_id = ov.caption_style
    if ov.caption_position in ("bottom", "center", "top"):
        out.position = ov.caption_position  # type: ignore[assignment]
    if ov.caption_max_words:
        out.max_words = int(ov.caption_max_words)
    elif ov.caption_density is not None:
        out.max_words = int(round(max(3.0, min(14.0, 16.0 - 12.0 * ov.caption_density))))
    return out


def keyword_rate(project: "Project") -> float | None:
    """The style's keyword-highlighting rate (0.5 = the default budget), or ``None``."""
    ov = active_overrides(project)
    return ov.keyword_emphasis_rate if ov is not None else None


def effective_audio_settings(project: "Project") -> "AudioSettings":
    """Audio settings for one generation run: music level, ducking depth, pause rise and sound-effect rate from the applied style. A level the user set on
    purpose is kept (and only clamped so that ducked <= normal <= pause level still holds)."""
    out = deepcopy(project.audio_settings)
    ov = active_overrides(project)
    if ov is None:
        return out
    old = out.music_level
    level = ov.music_level if ov.music_level is not None else old
    kept = protected_parameters(user_set_of(project), project.reference_settings.preserve_user_edits)
    out.music_level = level
    if ov.ducking_strength is not None:
        out.important_level = duck_level_for(level, ov.ducking_strength)
    elif "ducking_strength" not in kept and old > 0 and level != old:
        out.important_level = out.important_level * level / old  # keep the user's ducking depth relative to the new level
    if ov.pause_usage is not None:
        out.pause_level = pause_level_for(level, ov.pause_usage)
    elif "pause_usage" not in kept and old > 0 and level != old:
        out.pause_level = out.pause_level * level / old
    out.important_level = min(out.important_level, out.music_level)
    out.pause_level = max(out.pause_level, out.music_level)
    if ov.sfx_per_minute is not None:
        out.max_sfx_per_minute = float(ov.sfx_per_minute)
    out.important_level, out.pause_level = round(out.important_level, 4), round(out.pause_level, 4)
    out.music_level = round(out.music_level, 4)
    return out
