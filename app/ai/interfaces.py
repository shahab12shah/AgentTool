"""Data contracts for the future AI pipeline (Phase 2+).

Nothing here talks to a model. These types exist so the timeline, project schema and UI
can already be written against stable shapes. Decisions made through these contracts are
stored in ``Project.ai_decisions`` and are always user-editable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ScriptAnalysis:
    summary: str = ""
    tone: str | None = None
    topics: list[str] = field(default_factory=list)


@dataclass
class SceneSegment:
    index: int
    script_text: str
    start: float | None = None
    end: float | None = None


@dataclass
class VisualIntent:
    scene_id: str
    description: str
    keywords: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)  # SourceType names


@dataclass
class VisualCandidate:
    id: str
    scene_id: str
    source_type: str
    url: str | None = None
    asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualScore:
    candidate_id: str
    score: float  # 0..1
    rationale: str = ""


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    language: str | None = None


@runtime_checkable
class Transcriber(Protocol):
    """Voice-over transcription boundary."""

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult: ...


@runtime_checkable
class VisualSource(Protocol):
    """A place visuals can be found (stock library, YouTube, web, screenshots, generators)."""

    source_type: str

    def search(self, intent: VisualIntent, limit: int = 10) -> list[VisualCandidate]: ...
