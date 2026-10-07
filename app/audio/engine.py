"""AudioEngine: the provider-independent audio facade.

    AudioEngine -> AudioAnalysisService -> AudioProcessingService -> AudioMixService   (all on top of an AudioBackend)

The UI and application services call the engine; nothing outside this package runs audio tools.
"""

from __future__ import annotations

from typing import Callable

from app.audio.analysis import AudioAnalysisService
from app.audio.backend import AudioBackend, FFmpegAudioBackend
from app.audio.ducking import AudioDuckingService
from app.audio.mix import AudioMixService
from app.audio.priority import VoicePriorityController
from app.audio.processing import AudioProcessingService
from app.audio.waveform import WaveformService
from app.presentation.models import AudioSettings


class AudioEngine:
    def __init__(self, ffmpeg_path: Callable[[], str] | str = "", cache_dir: Callable[[], object] | None = None, backend: AudioBackend | None = None) -> None:
        self.backend: AudioBackend = backend or FFmpegAudioBackend(ffmpeg_path)
        self.analysis = AudioAnalysisService(self.backend)
        self.processing = AudioProcessingService(self.backend)
        self.mix = AudioMixService(self.backend)
        self.waveforms = WaveformService(self.backend, cache_dir or (lambda: None))

    def available(self) -> tuple[bool, str]:
        return self.backend.is_available()

    @staticmethod
    def priority(settings: AudioSettings) -> VoicePriorityController:
        return VoicePriorityController(settings)

    @staticmethod
    def ducking(settings: AudioSettings) -> AudioDuckingService:
        return AudioDuckingService(settings, VoicePriorityController(settings))
