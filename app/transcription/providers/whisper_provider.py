"""Local faster-whisper provider (the recommended accurate local engine).

Requires ``pip install faster-whisper`` and a model (a size name such as ``small.en`` that
faster-whisper can download, or a local model directory). Word timestamps and per-word
probabilities come straight from the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.core.exceptions import JobCancelled, TranscriptionError
from app.transcription.provider import CancelFn, ProgressFn, RawTranscription, RawWord, TranscriptionProvider


class FasterWhisperProvider(TranscriptionProvider):
    name = "faster-whisper"
    kind = "local"

    def __init__(self, model: Callable[[], str] = lambda: "", device: str = "auto", compute_type: str = "int8",
                 model_factory: Callable[[str], Any] | None = None) -> None:
        self._model_name = model
        self._device, self._compute = device, compute_type
        self._factory = model_factory  # injectable for tests
        self._cache: tuple[str, Any] | None = None

    def is_available(self) -> tuple[bool, str]:
        if self._factory is None:
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                return False, "faster-whisper is not installed (pip install faster-whisper)."
        if not self._model_name().strip():
            return False, "No Whisper model is configured. Set a model name or folder in Settings → Transcription."
        return True, ""

    def config_key(self) -> str:
        return f"{self._model_name()}:{self._compute}"

    def _model(self) -> Any:
        name = self._model_name().strip()
        if self._cache and self._cache[0] == name:
            return self._cache[1]
        try:
            if self._factory:
                model = self._factory(name)
            else:
                from faster_whisper import WhisperModel

                model = WhisperModel(name, device=self._device, compute_type=self._compute)
        except Exception as exc:
            raise TranscriptionError(
                f"The Whisper model “{name}” could not be loaded. Check the model name/folder and your network.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc
        self._cache = (name, model)
        return model

    def transcribe(self, audio_path: Path, language: str | None = None, progress: ProgressFn | None = None,
                   should_cancel: CancelFn | None = None) -> RawTranscription:
        ok, reason = self.is_available()
        if not ok:
            raise TranscriptionError(reason)
        if progress:
            progress(0.02, "Loading speech model")
        model = self._model()
        try:
            segments, info = model.transcribe(str(audio_path), language=language or None, word_timestamps=True, vad_filter=False)
            total = float(getattr(info, "duration", 0) or 0)
            words: list[RawWord] = []
            for seg in segments:  # lazy generator: decoding happens while iterating
                if should_cancel and should_cancel():
                    raise JobCancelled()
                for w in getattr(seg, "words", None) or []:
                    words.append(RawWord(str(w.word).strip(), float(w.start), float(w.end), _prob(getattr(w, "probability", None))))
                if progress and total:
                    progress(0.05 + 0.9 * min(1.0, float(seg.end) / total), "Transcribing")
        except (JobCancelled, TranscriptionError):
            raise
        except Exception as exc:
            raise TranscriptionError("Whisper failed while transcribing this audio.", details=f"{type(exc).__name__}: {exc}") from exc
        return RawTranscription(words, getattr(info, "language", language), self._model_name())


def _prob(p: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(p)))
    except (TypeError, ValueError):
        return None
