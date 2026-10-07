"""EditingStrategyOverrides: the *only* way a reference style reaches the editing engine.

Abstract strategy parameters (a target shot length, a motion level, a text budget, caption style hints, music/SFX levels...). Every field is
optional: ``None`` means "no opinion, use the user's setting / the preset". The engines read these values; they never see the reference
video, its analysis data, its text or its footage, and the reference analyzer never touches a timeline.

Priority (highest first): user lock / user override  >  content requirement  >  user style settings  >  reference style  >  AI default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.core.serialization import from_plain, to_plain

# the parameters (everything else is bookkeeping)
PARAMETERS = ("target_shot_duration", "min_shot_duration", "max_shot_duration", "motion_intensity", "transition_frequency", "text_density", "hook_seconds", "hook_shot_factor",
              "caption_style", "caption_position", "caption_density", "caption_max_words", "keyword_emphasis_rate", "music_level", "ducking_strength", "sfx_per_minute", "pause_usage")
# which user-setting field protects which parameter when the user's own value must win (see effective.py)
PROTECTED_BY = {
    "target_shot_duration": ("editing", "pacing"), "min_shot_duration": ("editing", "pacing"), "max_shot_duration": ("editing", "pacing"),
    "hook_seconds": ("editing", "pacing"), "hook_shot_factor": ("editing", "pacing"),
    "motion_intensity": ("editing", "motion_intensity"), "transition_frequency": ("editing", "transition_frequency"),
    "text_density": ("editing", "text_emphasis"),
    "caption_style": ("caption", "style_id"), "caption_position": ("caption", "position"), "caption_max_words": ("caption", "max_words"), "caption_density": ("caption", "max_words"),
    "keyword_emphasis_rate": ("caption", "keyword_highlight"),
    "music_level": ("audio", "music_level"), "ducking_strength": ("audio", "important_level"), "sfx_per_minute": ("audio", "max_sfx_per_minute"), "pause_usage": ("audio", "pause_level"),
}


@dataclass
class EditingStrategyOverrides:
    # ---- Phase 4 (editing)
    target_shot_duration: float | None = None  # seconds, *before* the content adjustments (narration speed, complexity, reading time)
    min_shot_duration: float | None = None
    max_shot_duration: float | None = None
    motion_intensity: float | None = None  # 0..1, same scale as the AI Edit "Motion" slider
    transition_frequency: float | None = None  # 0..1, same scale as the AI Edit "Transitions" slider
    text_density: float | None = None  # 0..1 where 0.5 = the preset's own text budget
    hook_seconds: float | None = None  # the opening that gets tighter pacing
    hook_shot_factor: float | None = None  # < 1 shortens shots inside the hook
    # ---- Phase 5 (captions / graphics)
    caption_style: str | None = None  # a caption style id
    caption_position: str | None = None  # bottom | center | top
    caption_density: float | None = None  # 0..1 informational (drives caption_max_words when that is not given)
    caption_max_words: int | None = None
    keyword_emphasis_rate: float | None = None  # 0..1 where 0.5 = default keyword budget
    # ---- Phase 5 (audio)
    music_level: float | None = None  # 0..1 linear level under narration
    ducking_strength: float | None = None  # 0..1 how far the music drops under speech
    sfx_per_minute: float | None = None
    pause_usage: float | None = None  # 0..1 how much the music rises during pauses
    # ---- bookkeeping
    strength: float = 1.0  # the style strength these values were blended with (25/50/75/100 %)
    source: str = "reference"
    applied_fields: list[str] = field(default_factory=list)  # which parameters carry a value
    notes: list[str] = field(default_factory=list)  # why a value is what it is (content limits, strength, user adjustments)

    # ------------------------------------------------------------------ helpers
    def active(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in PARAMETERS if getattr(self, k) is not None}

    @property
    def is_empty(self) -> bool:
        return not self.active()

    def without(self, names: set[str] | list[str]) -> "EditingStrategyOverrides":
        """A copy with the given parameters removed (the user's own values win for those)."""
        out = from_plain(EditingStrategyOverrides, to_plain(self))
        for n in names:
            if n in PARAMETERS:
                setattr(out, n, None)
        out.applied_fields = [f for f in out.applied_fields if getattr(out, f, None) is not None]
        return out

    def signature(self) -> str:
        """Stable fingerprint of the active parameters (used so scenes edited under a different style are seen as changed)."""
        return hashlib.sha1(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.active().items()}, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditingStrategyOverrides":
        return from_plain(cls, d)
