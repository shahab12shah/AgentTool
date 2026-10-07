"""Offline PocketSphinx provider (bundled English model, no setup, no network).

Accuracy is LOW compared with Whisper-class models: expect many misrecognised words and no
punctuation or capitalisation. Timings are real (10 ms frames). It exists so the application
works out of the box; use the faster-whisper or API provider for production transcripts.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from app.core.exceptions import JobCancelled, TranscriptionError
from app.media.media_probe import locate_binary, run_process
from app.transcription.provider import CancelFn, ProgressFn, RawTranscription, RawWord, TranscriptionProvider

FRAME_SECONDS = 0.01
CHUNK_BYTES = 16000 * 2 * 5  # 5 s of 16 kHz mono s16le
_SKIP = {"<s>", "</s>", "<sil>"}


class PocketSphinxProvider(TranscriptionProvider):
    name = "pocketsphinx"
    kind = "local"
    accuracy_note = "Offline fallback with low accuracy. For reliable transcripts use faster-whisper or an API provider."

    def __init__(self, ffmpeg_path: Callable[[], str] = lambda: "") -> None:
        self._ffmpeg_path = ffmpeg_path

    def is_available(self) -> tuple[bool, str]:
        try:
            import pocketsphinx  # noqa: F401
        except ImportError:
            return False, "The 'pocketsphinx' package is not installed (pip install pocketsphinx)."
        return True, ""

    def config_key(self) -> str:
        return "en-us-default"

    def transcribe(self, audio_path: Path, language: str | None = None, progress: ProgressFn | None = None,
                   should_cancel: CancelFn | None = None) -> RawTranscription:
        ok, reason = self.is_available()
        if not ok:
            raise TranscriptionError(reason)
        if language and not language.lower().startswith("en"):
            raise TranscriptionError("The offline PocketSphinx provider only supports English.")
        from pocketsphinx import Decoder

        ffmpeg = locate_binary("ffmpeg", self._ffmpeg_path())
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "audio.raw"
            if progress:
                progress(0.02, "Converting audio")
            result = run_process([ffmpeg, "-y", "-v", "error", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "s16le", str(raw)], timeout=3600)
            if result.returncode != 0 or not raw.is_file():
                raise TranscriptionError("The audio could not be decoded.", details=result.stderr[-500:])
            data = raw.read_bytes()
        if not data:
            raise TranscriptionError("The audio file contains no audio.")

        try:
            decoder = Decoder(samprate=16000, loglevel="FATAL")
            decoder.start_utt()
            for pos in range(0, len(data), CHUNK_BYTES):
                if should_cancel and should_cancel():
                    decoder.end_utt()
                    raise JobCancelled()
                decoder.process_raw(data[pos : pos + CHUNK_BYTES], no_search=False, full_utt=False)
                if progress:
                    progress(0.05 + 0.9 * min(1.0, (pos + CHUNK_BYTES) / len(data)), "Recognising speech")
            decoder.end_utt()
            segments = list(decoder.seg() or [])  # seg() is None when nothing was recognised
        except JobCancelled:
            raise
        except Exception as exc:
            raise TranscriptionError("The offline speech engine failed on this audio.", details=f"{type(exc).__name__}: {exc}") from exc

        words: list[RawWord] = []
        for seg in segments:
            text = re.sub(r"\(\d+\)$", "", seg.word)
            if text in _SKIP or text.startswith("[") or text.startswith("++") or text.startswith("<"):
                continue
            words.append(RawWord(text, seg.start_frame * FRAME_SECONDS, (seg.end_frame + 1) * FRAME_SECONDS,
                                 max(0.0, min(1.0, float(seg.prob)))))
        return RawTranscription(words, "en", "pocketsphinx-en-us")
