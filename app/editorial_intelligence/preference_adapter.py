"""Preference adaptation: the user's editorial preferences, local and controllable.

This module defines how preferences are *read* (``effective_preferences``) and how signals / settings change them. Nothing is learned from a single action; learned values only bias
recommendation thresholds and ranking, never write to ``EditingSettings`` / caption / audio settings, never modify an existing project, and stay on this machine (the global copy lives in
the application settings file; projects keep their own). The learning itself (signals -> values) is implemented in ``PreferenceAdapter`` below.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.editorial_intelligence.editorial_models import UserEditorialPreferences


@dataclass
class EffectivePreferences:
    """What the engines look at for one run: the project's preferences laid over the global ones (unless the project is excluded). ``signature`` goes into every cache key that depends on it."""

    prefs: UserEditorialPreferences = field(default_factory=UserEditorialPreferences)
    learning_enabled: bool = True
    scope_used: str = "PROJECT"
    signature: str = ""

    def source_weight(self, source_type: str) -> float:
        return float(self.prefs.preferred_visual_sources.get(source_type, 0.0))

    def rejects(self, pattern: str) -> bool:
        return pattern in self.prefs.rejected_visual_patterns


def signature_of(prefs: UserEditorialPreferences) -> str:
    import hashlib
    import json

    plain = to_plain(prefs)
    plain.pop("signals", None)  # the log is evidence, not an input
    plain.pop("last_updated", None)
    return hashlib.sha1(json.dumps(plain, sort_keys=True, default=str).encode()).hexdigest()[:12]


def merge(project_prefs: UserEditorialPreferences, global_prefs: UserEditorialPreferences | None) -> UserEditorialPreferences:
    """The project's values win; fields the project leaves empty fall back to the global ones."""
    if global_prefs is None or project_prefs.excluded_from_learning or project_prefs.scope == "PROJECT" and not global_prefs.learning_enabled:
        return deepcopy(project_prefs)
    out = deepcopy(global_prefs) if project_prefs.scope == "GLOBAL" else deepcopy(project_prefs)
    out.scope = project_prefs.scope
    if project_prefs.scope == "PROJECT":
        for name in ("preferred_visual_sources", "transition_preferences", "caption_preferences", "graphic_preferences", "audio_preferences", "visual_diversity_preferences"):
            merged = dict(getattr(global_prefs, name))
            merged.update(getattr(project_prefs, name))
            setattr(out, name, merged)
        out.rejected_visual_patterns = sorted(set(global_prefs.rejected_visual_patterns) | set(project_prefs.rejected_visual_patterns))
        for name in ("preferred_shot_duration", "preferred_motion_intensity", "reference_style_strength"):
            if getattr(project_prefs, name) is None:
                setattr(out, name, getattr(global_prefs, name))
    out.learning_enabled = project_prefs.learning_enabled and global_prefs.learning_enabled
    return out


def effective_preferences(project: Any, global_prefs: UserEditorialPreferences | None = None) -> EffectivePreferences:
    pp = project.editorial.user_editorial_preferences
    prefs = merge(pp, global_prefs)
    return EffectivePreferences(prefs, prefs.learning_enabled and not pp.excluded_from_learning, "GLOBAL+PROJECT" if global_prefs is not None and not pp.excluded_from_learning else "PROJECT",
                                signature_of(prefs))


def from_plain_prefs(d: dict[str, Any] | None) -> UserEditorialPreferences:
    return from_plain(UserEditorialPreferences, d) if d else UserEditorialPreferences(scope="GLOBAL")
