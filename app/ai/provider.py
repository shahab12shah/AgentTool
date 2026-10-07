"""AI provider abstraction. Phase 1 ships only an explicit "not implemented" provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.ai.interfaces import (
    SceneSegment,
    ScriptAnalysis,
    TranscriptionResult,
    VisualCandidate,
    VisualIntent,
    VisualScore,
)
from app.core.exceptions import NotAvailableInPhase


class AIProvider(ABC):
    """Everything an AI backend must be able to do. Implementations arrive in Phase 2."""

    name: str = "abstract"

    @abstractmethod
    def analyze_script(self, script: str) -> ScriptAnalysis: ...

    @abstractmethod
    def segment_scenes(self, script: str, transcript: TranscriptionResult | None = None) -> list[SceneSegment]: ...

    @abstractmethod
    def generate_visual_intent(self, scene: SceneSegment) -> VisualIntent: ...

    @abstractmethod
    def score_visual(self, intent: VisualIntent, candidate: VisualCandidate) -> VisualScore: ...

    @abstractmethod
    def generate_image(self, prompt: str, width: int, height: int) -> bytes: ...


class UnavailableAIProvider(AIProvider):
    """Placeholder provider: every call fails loudly instead of returning fake results."""

    name = "unavailable"

    @staticmethod
    def _nope(feature: str) -> NotAvailableInPhase:
        return NotAvailableInPhase(f"{feature} is not available yet (coming in Phase 2).")

    def analyze_script(self, script: str) -> ScriptAnalysis:
        raise self._nope("Script analysis")

    def segment_scenes(self, script: str, transcript: TranscriptionResult | None = None) -> list[SceneSegment]:
        raise self._nope("Scene segmentation")

    def generate_visual_intent(self, scene: SceneSegment) -> VisualIntent:
        raise self._nope("Visual intent generation")

    def score_visual(self, intent: VisualIntent, candidate: VisualCandidate) -> VisualScore:
        raise self._nope("Visual scoring")

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        raise self._nope("AI image generation")


class AIProviderRegistry:
    """Named providers; the active one is chosen by the application, never by the UI."""

    def __init__(self) -> None:
        self._providers: dict[str, AIProvider] = {}
        self._active = "unavailable"
        self.register(UnavailableAIProvider())

    def register(self, provider: AIProvider) -> None:
        self._providers[provider.name] = provider

    def activate(self, name: str) -> None:
        if name not in self._providers:
            raise KeyError(name)
        self._active = name

    @property
    def active(self) -> AIProvider:
        return self._providers[self._active]
