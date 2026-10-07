"""Source provider interface. Each provider turns a query into *normalised* candidates; nothing else sees raw responses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.exceptions import AcquisitionError
from app.media.asset import SourceType
from app.research.http import HttpClient
from app.research.models import Candidate, ResearchBrief, ResearchQuery


@dataclass
class SearchContext:
    brief: ResearchBrief | None = None
    cache_dir: Path | None = None  # project-local scratch/cache (screenshots, thumbnails)
    should_cancel: Callable[[], bool] = lambda: False
    default_clip_seconds: float = 5.0
    min_width: int = 0


class SourceProvider(ABC):
    name: str = "abstract"
    label: str = "Abstract"
    source_types: tuple[SourceType, ...] = ()
    verified_live: bool = False  # True only if this adapter was exercised against the real service (never claimed by default)

    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(usable now?, human-readable reason when not)."""

    @abstractmethod
    def config_key(self) -> str:
        """Settings that change results (base URL, folder...). Part of the research cache key."""

    @abstractmethod
    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        """Return normalised candidates (``candidate_id`` may be empty: the engine assigns ids)."""

    def fetch_thumbnail(self, candidate: Candidate, dest: Path, http: HttpClient) -> bool:
        """Default: download ``thumbnail_url``. Returns False when there is nothing to fetch."""
        if not candidate.thumbnail_url:
            return False
        http.download(candidate.thumbnail_url, dest, max_bytes=8 * 1024 * 1024)
        return True

    def acquire(self, candidate: Candidate, dest_dir: Path, ctx: SearchContext) -> Path:
        """Produce a local media file for ``candidate`` inside ``dest_dir``. Raises ``AcquisitionError``."""
        raise AcquisitionError("This source cannot be downloaded by the application.")


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SourceProvider] = {}

    def register(self, provider: SourceProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> SourceProvider | None:
        return self._providers.get(name)

    def all(self) -> list[SourceProvider]:
        return list(self._providers.values())

    def for_source(self, source_type: SourceType) -> list[SourceProvider]:
        return [p for p in self._providers.values() if source_type in p.source_types]


def blank_candidate(query: ResearchQuery, source_type: SourceType, kind: str, provider: str) -> Candidate:
    return Candidate(candidate_id="", scene_id=query.scene_id, source_type=source_type, kind=kind, provider=provider,
                     query_ids=[query.query_id])
