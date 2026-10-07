"""Turns a provider's raw output into a validated ``Transcript`` (+ optional script alignment)."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import TranscriptionError
from app.logging.logger import get_logger, log_event
from app.media.asset import Asset
from app.transcription.alignment import ScriptAlignment, align_script
from app.transcription.models import AudioInfo, ProviderInfo, Transcript, Word
from app.transcription.provider import CancelFn, ProgressFn, RawTranscription, RawWord, TranscriptionProvider
from app.transcription.sentences import build_sentences

_log = get_logger(__name__)
MIN_ALIGNMENT_COVERAGE = 0.6  # below this, script sentence boundaries are not trusted


@dataclass
class TranscriptionResult:
    transcript: Transcript
    alignment: ScriptAlignment | None
    repaired_words: int = 0
    dropped_words: int = 0


def normalize_words(raw: list[RawWord]) -> tuple[list[Word], int, int]:
    """Assign ids and repair provider quirks. Returns ``(words, repaired, dropped)``.

    Repairs: end < start -> end = start; negative start -> 0; out-of-order -> sorted;
    confidence outside 0..1 -> clamped. Words with empty text or non-finite times are dropped.
    Timestamps are never rounded.
    """
    repaired = dropped = 0
    cleaned: list[RawWord] = []
    for w in raw:
        text = (w.text or "").strip()
        if not text or not (math.isfinite(w.start) and math.isfinite(w.end)):
            dropped += 1
            continue
        start, end, conf = float(w.start), float(w.end), w.confidence
        if start < 0 or end < start:
            repaired += 1
            start = max(0.0, start)
            end = max(start, end)
        if conf is not None and not (math.isfinite(conf) and 0.0 <= conf <= 1.0):
            repaired += 1
            conf = max(0.0, min(1.0, conf)) if math.isfinite(conf) else None
        cleaned.append(RawWord(text, start, end, conf))
    if any(b.start < a.start for a, b in zip(cleaned, cleaned[1:])):
        repaired += 1
        cleaned.sort(key=lambda w: w.start)
    words = [Word(f"w_{i:06d}", w.text, w.start, w.end, w.confidence) for i, w in enumerate(cleaned)]
    return words, repaired, dropped


class TranscriptionEngine:
    """Provider-agnostic pipeline: provider -> words -> (alignment) -> sentences -> validated transcript."""

    def run(
        self,
        provider: TranscriptionProvider,
        audio_path: Path,
        asset: Asset,
        language: str | None = None,
        script: str = "",
        progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
    ) -> TranscriptionResult:
        def sub(lo: float, hi: float) -> ProgressFn:
            return lambda f, m: progress(lo + (hi - lo) * f, m) if progress else None

        ok, reason = provider.is_available()
        if not ok:
            raise TranscriptionError(reason)
        raw: RawTranscription = provider.transcribe(audio_path, language, sub(0.0, 0.88), should_cancel)
        if progress:
            progress(0.9, "Building word and sentence timing")
        words, repaired, dropped = normalize_words(raw.words)
        if not words:
            raise TranscriptionError("No speech was detected in this audio.")
        if repaired or dropped:
            _log.warning("Provider output repaired", extra={"provider": provider.name, "repaired": repaired, "dropped": dropped})

        transcript = Transcript(
            transcript_id=f"tr_{uuid.uuid4().hex[:10]}",
            audio=AudioInfo(
                file_id=asset.id,
                duration=asset.duration or words[-1].end,
                sample_rate=asset.sample_rate,
                channels=asset.channels,
                content_hash=asset.content_hash,
                filename=asset.name,
            ),
            words=words,
            sentences=[],
            provider=ProviderInfo(provider.name, raw.model, raw.language or language, provider.accuracy_note),
        )
        alignment = None
        script_sentences: dict[str, int] | None = None
        if script.strip():
            alignment = align_script(script, transcript)
            if alignment.stats.coverage >= MIN_ALIGNMENT_COVERAGE:
                script_sentences = alignment.script_sentence_of_word()
        transcript.sentences, transcript.sentence_strategy = build_sentences(words, script_sentences)
        transcript.validate()
        log_event(_log, "transcription.done", provider=provider.name, words=len(words), sentences=len(transcript.sentences))
        if progress:
            progress(1.0, "Transcription complete")
        return TranscriptionResult(transcript, alignment, repaired, dropped)
