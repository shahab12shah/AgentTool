"""AudioAnalysisService: voice-over analysis. The voice-over is the MASTER_TIMING_REFERENCE; nothing here moves it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.audio.backend import ANALYSIS_SR, AudioBackend
from app.audio.loudness import LoudnessAnalyzer, db
from app.presentation.models import VoiceAnalysis
from app.transcription.models import Transcript

STOP = {"the", "and", "that", "this", "with", "from", "have", "were", "been", "they", "their", "which", "would", "there", "about", "into", "than", "then",
        "also", "just", "very", "more", "some", "will", "what", "when", "your", "them", "does", "over"}
PAUSE_GAP = 0.30


def _rms_db(samples: np.ndarray, a: float, b: float) -> float | None:
    i, j = int(a * ANALYSIS_SR), int(b * ANALYSIS_SR)
    seg = samples[max(0, i):max(i + 1, j)]
    if len(seg) == 0:
        return None
    return db(float(np.sqrt(np.mean(seg ** 2))))


class AudioAnalysisService:
    def __init__(self, backend: AudioBackend) -> None:
        self.backend = backend
        self.loudness = LoudnessAnalyzer(backend)

    def analyze_voice(self, path: Path, asset_id: str, audio_hash: str, duration: float | None = None, transcript: Transcript | None = None,
                      progress=None) -> VoiceAnalysis:
        if progress:
            progress(5, "Reading the voice-over")
        samples = self.backend.decode_mono(path)
        total = len(samples) / ANALYSIS_SR
        if progress:
            progress(35, "Measuring loudness")
        ld = self.loudness.analyze(path, samples)
        a = VoiceAnalysis(asset_id=asset_id, audio_hash=audio_hash, duration=round(duration or total, 3), peak_db=round(ld.peak_db, 2), rms_db=round(ld.rms_db, 2),
                          lufs=ld.lufs, loudness_range=ld.loudness_range, dynamic_range_db=ld.dynamic_range_db, clipped_samples=ld.clipped_samples,
                          noise_floor_db=round(ld.noise_floor_db, 2), silence_regions=ld.silence_regions, issues=ld.issues, backend=self.backend.name)
        if transcript is not None and transcript.words:
            if progress:
                progress(70, "Measuring speech")
            self._with_words(a, samples, transcript)
        else:
            active = sum(b - s for s, b in [[0, total]]) - sum(b - s for s, b in ld.silence_regions)
            a.speech_ratio = round(max(0.0, active) / total, 3) if total else None
        if progress:
            progress(100, "Done")
        return a

    @staticmethod
    def _with_words(a: VoiceAnalysis, samples: np.ndarray, tr: Transcript) -> None:
        words = tr.words
        pauses = [[round(x.end, 3), round(y.start, 3)] for x, y in zip(words, words[1:]) if y.start - x.end > PAUSE_GAP]
        speech = max(0.4, (words[-1].end - words[0].start) - sum(b - s for s, b in pauses))
        a.pauses = pauses
        a.speaking_rate_wps = round(len(words) / speech, 2)
        a.speech_ratio = round(min(1.0, speech / max(a.duration, 1e-6)), 3)
        wmap = tr.word_map()
        cands: list[str] = []
        for s in tr.sentences:
            lvl = _rms_db(samples, s.start, s.end)
            if lvl is None:
                continue
            a.intensity_by_sentence[s.sentence_id] = round(lvl, 2)
            scored = []
            for wid in s.word_ids:
                w = wmap.get(wid)
                if w is None or len(w.text.strip(".,;:!?")) < 4 or w.text.lower().strip(".,;:!?") in STOP:
                    continue
                wl = _rms_db(samples, w.start, w.end)
                if wl is not None and wl - lvl >= 3.0:
                    scored.append((wl - lvl, wid))
            cands += [wid for _d, wid in sorted(scored, reverse=True)[:2]]
        a.emphasis_candidates = cands
