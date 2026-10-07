"""Visual source preferences and rules (saved in the project).

Source percentages are SOFT targets: they bias, never restrict. If the best visual for a
scene is a screenshot, it is used even when screenshots are "over budget".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.serialization import from_plain, to_plain

SOFT_TARGET_NOTE = "These values are treated as soft preferences, not limits."


class SourceKind(str, Enum):
    YOUTUBE = "YOUTUBE"
    STOCK_IMAGES = "STOCK_IMAGES"
    STOCK_VIDEOS = "STOCK_VIDEOS"
    AI_IMAGES = "AI_IMAGES"
    WEB_IMAGES = "WEB_IMAGES"
    WEB_VIDEOS = "WEB_VIDEOS"
    SCREENSHOTS = "SCREENSHOTS"


SOURCE_LABELS = {
    SourceKind.YOUTUBE: "YouTube",
    SourceKind.STOCK_IMAGES: "Stock Images",
    SourceKind.STOCK_VIDEOS: "Stock Videos",
    SourceKind.AI_IMAGES: "AI Images",
    SourceKind.WEB_IMAGES: "Real Web Images",
    SourceKind.WEB_VIDEOS: "Real Web Videos",
    SourceKind.SCREENSHOTS: "Screenshots",
}
DEFAULT_TARGETS = {
    SourceKind.YOUTUBE: 20.0, SourceKind.STOCK_IMAGES: 15.0, SourceKind.STOCK_VIDEOS: 15.0, SourceKind.AI_IMAGES: 20.0,
    SourceKind.WEB_IMAGES: 10.0, SourceKind.WEB_VIDEOS: 10.0, SourceKind.SCREENSHOTS: 10.0,
}


@dataclass
class SourceSetting:
    enabled: bool = True
    target_percent: float = 0.0
    priority: int = 3  # 1 (low) .. 5 (high): a soft nudge used by visual research ranking


def _default_sources() -> dict[str, SourceSetting]:
    return {k.value: SourceSetting(True, DEFAULT_TARGETS[k]) for k in SourceKind}


@dataclass
class PreferenceReport:
    total: float  # sum of the targets of enabled sources
    warnings: list[str]

    @property
    def is_balanced(self) -> bool:
        return abs(self.total - 100.0) < 1e-6


@dataclass
class VisualPreferences:
    sources: dict[str, SourceSetting] = field(default_factory=_default_sources)
    min_accuracy_score: int = 85
    prefer_real_visuals: bool = False
    prefer_ai_visuals: bool = False
    prefer_evidence: bool = True
    match_narration_literally: bool = False
    allow_visual_interpretation: bool = True
    avoid_repeated_visuals: bool = True

    def setting(self, kind: SourceKind) -> SourceSetting:
        return self.sources.setdefault(kind.value, SourceSetting(True, DEFAULT_TARGETS[kind]))

    def sanitized(self) -> "VisualPreferences":
        """Clamp values into range and make sure every source exists. Never raises."""
        out = VisualPreferences.from_dict(self.to_dict())
        for k in SourceKind:
            s = out.setting(k)
            s.target_percent = max(0.0, min(1000.0, float(s.target_percent)))
            s.priority = int(max(1, min(5, s.priority)))
        out.min_accuracy_score = int(max(0, min(100, out.min_accuracy_score)))
        return out

    def report(self) -> PreferenceReport:
        total = sum(s.target_percent for s in self.sources.values() if s.enabled)
        warnings: list[str] = []
        if total > 100.0 + 1e-6:
            warnings.append(f"Targets exceed 100%. {SOFT_TARGET_NOTE}")
        elif total < 100.0 - 1e-6:
            warnings.append(f"Targets add up to less than 100%. {SOFT_TARGET_NOTE}")
        if not any(s.enabled for s in self.sources.values()):
            warnings.append("No visual source is enabled, so nothing could be researched.")
        if self.prefer_real_visuals and self.prefer_ai_visuals:
            warnings.append("“Prefer real visuals” and “Prefer AI visuals” pull in opposite directions; both are soft preferences.")
        if self.match_narration_literally and self.allow_visual_interpretation:
            warnings.append("“Match narration literally” and “Allow visual interpretation” overlap; the scene's visual type decides.")
        return PreferenceReport(total, warnings)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VisualPreferences":
        prefs = from_plain(cls, d) if d else cls()
        for k in SourceKind:  # sources added in later versions get defaults
            prefs.sources.setdefault(k.value, SourceSetting(True, DEFAULT_TARGETS[k]))
        return prefs
