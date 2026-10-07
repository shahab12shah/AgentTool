"""Provider-independent transcription interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ProgressFn = Callable[[float, str], None]  # (0..1, message)
CancelFn = Callable[[], bool]


@dataclass
class RawWord:
    """A word as reported by a provider, before ids are assigned and timing is checked."""

    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class RawTranscription:
    words: list[RawWord]
    language: str | None = None
    model: str | None = None


class TranscriptionProvider(ABC):
    """A speech-to-text backend. Implementations must be safe to call from a worker thread."""

    name: str = "abstract"
    kind: str = "local"  # "local" | "api"
    accuracy_note: str | None = None  # shown to the user when this provider is chosen

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(usable now?, human-readable reason when not)."""

    @abstractmethod
    def config_key(self) -> str:
        """Identifies settings that change the output (model, language...) for caching."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
    ) -> RawTranscription:
        """Transcribe ``audio_path`` and return words with timestamps. Raise ``TranscriptionError`` on failure."""


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, TranscriptionProvider] = {}

    def register(self, provider: TranscriptionProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> TranscriptionProvider | None:
        return self._providers.get(name)

    def names(self) -> list[str]:
        return list(self._providers)

    def all(self) -> list[TranscriptionProvider]:
        return list(self._providers.values())
