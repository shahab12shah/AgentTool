"""Transcript data model: audio metadata, words, sentences.

Timestamps are raw float seconds exactly as the provider reported them; nothing here rounds.
Words are the immutable record of what was spoken. Sentences are a *derived* grouping of
word ids and may be rebuilt without touching the words.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.exceptions import InvalidTranscriptError
from app.core.serialization import from_plain, to_plain


@dataclass
class AudioInfo:
    file_id: str  # asset id of the voice-over
    duration: float  # seconds
    sample_rate: int | None = None
    channels: int | None = None
    content_hash: str | None = None  # sha256 of the audio file; used for change detection
    filename: str | None = None


@dataclass
class Word:
    word_id: str
    text: str
    start: float
    end: float
    confidence: float | None = None  # 0..1 when the provider supplies it


@dataclass
class Sentence:
    sentence_id: str
    text: str
    start: float
    end: float
    word_ids: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class ProviderInfo:
    name: str
    model: str | None = None
    language: str | None = None
    note: str | None = None  # e.g. accuracy caveats


@dataclass
class Transcript:
    transcript_id: str
    audio: AudioInfo
    words: list[Word]
    sentences: list[Sentence]
    provider: ProviderInfo
    sentence_strategy: str = "pause"  # punctuation | pause | script-guided
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # ----- derived views -----
    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def word_map(self) -> dict[str, Word]:
        return {w.word_id: w for w in self.words}

    def sentence_map(self) -> dict[str, Sentence]:
        return {s.sentence_id: s for s in self.sentences}

    def sentence_at(self, t: float) -> Sentence | None:
        """Sentence being spoken at time ``t`` (the previous one during pauses between sentences)."""
        if not self.sentences or t < self.sentences[0].start:
            return None
        starts = [s.start for s in self.sentences]
        return self.sentences[bisect.bisect_right(starts, t) - 1]

    def word_at(self, t: float) -> Word | None:
        if not self.words or t < self.words[0].start:
            return None
        starts = [w.start for w in self.words]
        w = self.words[bisect.bisect_right(starts, t) - 1]
        return w if t <= w.end + 0.25 else None  # brief hold so highlights do not flicker in tiny gaps

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint lies in [start, end). Used to derive scene narration from timing."""
        return [w for w in self.words if start <= (w.start + w.end) / 2 < end]

    # ----- validation -----
    def validate(self) -> None:
        problems = validate_words(self.words)
        if self.audio.duration is None or not math.isfinite(self.audio.duration) or self.audio.duration < 0:
            problems.append("audio duration invalid")
        ids = {w.word_id for w in self.words}
        seen_s: set[str] = set()
        prev_end = 0.0
        for s in self.sentences:
            if s.sentence_id in seen_s:
                problems.append(f"duplicate sentence id {s.sentence_id}")
            seen_s.add(s.sentence_id)
            if not (_finite(s.start) and _finite(s.end)) or s.start < 0 or s.end < s.start:
                problems.append(f"sentence {s.sentence_id} has invalid timing")
            if s.start < prev_end - 1e-9:
                problems.append(f"sentence {s.sentence_id} overlaps the previous sentence")
            prev_end = max(prev_end, s.end)
            missing = [w for w in s.word_ids if w not in ids]
            if missing or not s.word_ids:
                problems.append(f"sentence {s.sentence_id} references unknown/no words")
        if problems:
            raise InvalidTranscriptError("The transcript contains invalid timing data.", details="; ".join(problems[:10]))

    # ----- persistence -----
    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        return from_plain(cls, d)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_words(words: list[Word]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    prev_start = 0.0
    for w in words:
        if w.word_id in seen:
            problems.append(f"duplicate word id {w.word_id}")
        seen.add(w.word_id)
        if not w.text.strip():
            problems.append(f"word {w.word_id} is empty")
        if not (_finite(w.start) and _finite(w.end)):
            problems.append(f"word {w.word_id} has non-finite timing")
            continue
        if w.start < 0 or w.end < w.start:
            problems.append(f"word {w.word_id} has invalid timing ({w.start}->{w.end})")
        if w.start < prev_start - 1e-9:
            problems.append(f"word {w.word_id} starts before the previous word")
        prev_start = max(prev_start, w.start)
        if w.confidence is not None and not (_finite(w.confidence) and 0.0 <= w.confidence <= 1.0):
            problems.append(f"word {w.word_id} confidence out of range")
    return problems


# ---------------------------------------------------------------- project-level state
@dataclass
class TranscriptionState:
    """What the project stores about transcription (the transcript itself plus failure info)."""

    transcript: Transcript | None = None
    last_error: str | None = None  # last failed attempt (cleared on success)
    failed_audio_hash: str | None = None
