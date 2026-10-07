"""Editing style presets and how the user's sliders modulate them."""

from __future__ import annotations

from dataclasses import dataclass

from app.editing.models import EditingSettings


@dataclass(frozen=True)
class StylePreset:
    name: str
    label: str
    base_shot: float  # seconds, for normal narration
    min_shot: float
    max_shot: float
    motion_ratio: float  # share of visuals that get any movement
    subtle: float  # scale growth of a subtle zoom
    punch: float  # scale growth of a punch-in
    transition_threshold: float  # 0..1: how strong a story break must be before a transition is used
    max_transition_ratio: float  # at most this share of scene boundaries
    transitions: tuple[str, ...]
    text_per_minute: float
    music_level: float
    description: str


PRESETS: dict[str, StylePreset] = {
    "documentary": StylePreset("documentary", "Documentary", 6.5, 2.5, 11.0, 0.45, 0.05, 0.12, 0.75, 0.12, ("FADE", "DISSOLVE"), 4, 0.20,
                               "Longer shots, subtle movement, minimal transitions, clean typography, evidence-focused."),
    "professional": StylePreset("professional", "Professional", 4.8, 1.8, 8.0, 0.65, 0.08, 0.15, 0.60, 0.20, ("FADE", "DISSOLVE"), 6, 0.18,
                                "Balanced cuts, moderate zoom, clean transitions, strong information hierarchy."),
    "dynamic": StylePreset("dynamic", "Dynamic", 3.2, 1.2, 5.5, 0.90, 0.10, 0.22, 0.45, 0.30, ("FADE", "DISSOLVE", "WIPE", "SLIDE"), 9, 0.22,
                           "Faster cuts, stronger punch-ins, more emphasis animations, frequent movement."),
}
DEFAULT_PRESET = "professional"


def preset_for(settings: EditingSettings) -> StylePreset:
    return PRESETS.get(settings.style, PRESETS[DEFAULT_PRESET])


def shot_factor(settings: EditingSettings) -> float:
    """Pacing slider: 0 (slow) -> 1.45x longer shots, 0.5 -> 1.0x, 1 (fast) -> 0.55x."""
    return 1.45 - 0.9 * min(1.0, max(0.0, settings.pacing))


def motion_scale(settings: EditingSettings) -> float:
    """Motion slider multiplier for movement amounts (0.4 .. 1.6)."""
    return 0.4 + 1.2 * min(1.0, max(0.0, settings.motion_intensity))


def motion_ratio(preset: StylePreset, settings: EditingSettings) -> float:
    return min(1.0, preset.motion_ratio * (0.5 + min(1.0, max(0.0, settings.motion_intensity))))


def transition_threshold(preset: StylePreset, settings: EditingSettings) -> float:
    return preset.transition_threshold * (1.5 - min(1.0, max(0.0, settings.transition_frequency)))
