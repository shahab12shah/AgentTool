"""API provider for OpenAI-compatible ``/audio/transcriptions`` endpoints (word timestamps).

The API key is read from an environment variable at call time. It is never stored in
settings or project files and never written to logs.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from app.core.exceptions import TranscriptionError
from app.media.media_probe import locate_binary, run_process
from app.transcription.provider import CancelFn, ProgressFn, RawTranscription, RawWord, TranscriptionProvider

MAX_UPLOAD_BYTES = 24 * 1024 * 1024


class OpenAICompatibleProvider(TranscriptionProvider):
    name = "api"
    kind = "api"

    def __init__(self, base_url: Callable[[], str] = lambda: "", model: Callable[[], str] = lambda: "whisper-1",
                 key_env: Callable[[], str] = lambda: "OPENAI_API_KEY", ffmpeg_path: Callable[[], str] = lambda: "",
                 timeout: float = 600.0) -> None:
        self._base, self._model, self._key_env, self._ffmpeg = base_url, model, key_env, ffmpeg_path
        self._timeout = timeout

    def is_available(self) -> tuple[bool, str]:
        if not self._base().strip():
            return False, "No API base URL is configured (Settings → Transcription)."
        if not os.environ.get(self._key_env().strip() or "OPENAI_API_KEY"):
            return False, f"The API key environment variable {self._key_env() or 'OPENAI_API_KEY'} is not set."
        return True, ""

    def config_key(self) -> str:
        return f"{self._base()}|{self._model()}"

    def transcribe(self, audio_path: Path, language: str | None = None, progress: ProgressFn | None = None,
                   should_cancel: CancelFn | None = None) -> RawTranscription:
        ok, reason = self.is_available()
        if not ok:
            raise TranscriptionError(reason)
        key = os.environ[self._key_env().strip() or "OPENAI_API_KEY"]
        with tempfile.TemporaryDirectory() as tmp:
            upload = Path(audio_path)
            if upload.stat().st_size > MAX_UPLOAD_BYTES:  # re-encode small and mono
                if progress:
                    progress(0.05, "Compressing audio for upload")
                small = Path(tmp) / "upload.mp3"
                r = run_process([locate_binary("ffmpeg", self._ffmpeg()), "-y", "-v", "error", "-i", str(audio_path),
                                 "-ac", "1", "-ar", "16000", "-b:a", "48k", str(small)], timeout=3600)
                if r.returncode != 0 or small.stat().st_size > MAX_UPLOAD_BYTES:
                    raise TranscriptionError("The audio is too long to upload to this API in one request.")
                upload = small
            if should_cancel and should_cancel():
                from app.core.exceptions import JobCancelled
                raise JobCancelled()
            if progress:
                progress(0.15, "Uploading audio")
            body, ctype = _multipart(
                {"model": self._model(), "response_format": "verbose_json", "timestamp_granularities[]": "word",
                 **({"language": language} if language else {})},
                "file", upload,
            )
            req = urllib.request.Request(
                self._base().rstrip("/") + "/audio/transcriptions", data=body, method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": ctype},
            )
            if progress:
                progress(0.3, "Waiting for the transcription service")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raise TranscriptionError(_http_message(exc.code), details=f"HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise TranscriptionError("The transcription service could not be reached.", details=str(exc)) from exc
            except ValueError as exc:
                raise TranscriptionError("The transcription service returned an unreadable response.", details=str(exc)) from exc
        words = [RawWord(str(w["word"]).strip(), float(w["start"]), float(w["end"]), None) for w in payload.get("words") or []]
        if not words and payload.get("text"):
            raise TranscriptionError("The service returned text without word timestamps, which this application requires.")
        return RawTranscription(words, payload.get("language"), self._model())


def _http_message(code: int) -> str:
    return {
        401: "The API rejected the key (401). Check the API key environment variable.",
        403: "The API refused the request (403).",
        404: "The API endpoint was not found (404). Check the base URL.",
        413: "The audio file is too large for this API (413).",
        429: "The API rate limit or quota was reached (429). Try again later.",
    }.get(code, f"The transcription service returned an error (HTTP {code}).")


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n".encode() + path.read_bytes() + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
