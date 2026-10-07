"""Interfaces for the Phase 3 visual research engine. Phase 2 performs NO research."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ai.interfaces import VisualCandidate
from app.analysis.models import Scene, VisualIntent
from app.core.exceptions import NotAvailableInPhase
from app.visual.preferences import SourceKind, VisualPreferences


@dataclass
class SceneResearchResult:
    scene_id: str
    candidates: list[VisualCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class VisualResearchService(ABC):
    """Takes a *structured scene* (never raw script text) and returns visual candidates."""

    @abstractmethod
    def search_scene(self, scene: Scene, intent: VisualIntent, prefs: VisualPreferences) -> SceneResearchResult: ...

    @abstractmethod
    def search_candidates(self, scene: Scene, intent: VisualIntent, source: SourceKind, limit: int = 10) -> list[VisualCandidate]: ...


class UnavailableResearchService(VisualResearchService):
    def search_scene(self, scene, intent, prefs):
        raise NotAvailableInPhase("Visual research is not available yet (coming in Phase 3).")

    def search_candidates(self, scene, intent, source, limit=10):
        raise NotAvailableInPhase("Visual research is not available yet (coming in Phase 3).")
