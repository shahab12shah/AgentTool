"""ReferenceFeatureExtractor: runs the detectors over one reference video and assembles ``ReferenceFeatures`` + ``ReferenceEvents``.

Pipeline (each step isolated: a failing detector marks its own dimensions unavailable and the rest of the analysis carries on):

    one gray frame pass  -> FrameSignals -> ShotDetector -> MotionAnalyzer
    one RGB frame pass   -> CaptionTextAnalyzer (geometry / persistence heuristics, no OCR, no text is read)
    one audio decode     -> AudioAnalyzer (voice / music / SFX / silence / ducking heuristics)
    observation lists    -> StructureAnalyzer (sections, pacing curve, hook, section transitions)

Nothing here modifies a project or a timeline, and nothing it keeps can reproduce the reference: only measurements.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.reference.signals import AnalysisCancelled, FrameSampler, ReferenceAnalysisError, SIGNAL_H, SIGNAL_W, compute_signals
from app.reference.style_model import (
    CONFIDENCE_KEYS, DetectorStatus, ReferenceEvents, ReferenceFeatures, ReferenceMetadata, Section, TextEvent,
)
from app.rendering.ffmpeg_service import FFmpegService

log = logging.getLogger(__name__)

SIGNAL_FPS = 8.0
CAPTION_FPS = 4.0
LOW_CONFIDENCE = 0.5

# (stage name, weight): the order and relative cost of the analysis stages shown to the user
STAGES: tuple[tuple[str, float], ...] = (
    ("Loading Reference", 0.02), ("Probing Media", 0.03), ("Detecting Shots", 0.30), ("Analyzing Motion", 0.08), ("Analyzing Captions", 0.25),
    ("Analyzing Audio", 0.20), ("Building Style Profile", 0.07), ("Generating Recommendations", 0.05),
)
STAGE_NAMES = tuple(n for n, _ in STAGES)

ProgressFn = Callable[[str, float, str], None]  # (stage, overall fraction 0..1, message)
LogFn = Callable[[str], None]


@dataclass
class AnalysisSettings:
    """Detection settings. A change here invalidates a cached analysis (see ``settings_hash``)."""

    sensitivity: float = 1.0  # shot-detector sensitivity (>1 = more cuts)
    signal_fps: float = SIGNAL_FPS
    caption_fps: float = CAPTION_FPS

    def settings_hash(self) -> str:
        import hashlib
        import json

        return hashlib.sha1(json.dumps([round(self.sensitivity, 3), self.signal_fps, self.caption_fps], sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class ExtractionResult:
    features: ReferenceFeatures
    events: ReferenceEvents
    log: list[str] = field(default_factory=list)


class StageProgress:
    """Maps (stage, fraction inside the stage) to one overall 0..1 value and reports it."""

    def __init__(self, report: ProgressFn | None, log_fn: LogFn | None = None) -> None:
        self.report, self.log_fn = report, log_fn
        self.starts: dict[str, float] = {}
        acc = 0.0
        total = sum(w for _, w in STAGES)
        self.weights: dict[str, float] = {}
        for name, w in STAGES:
            self.starts[name] = acc / total
            self.weights[name] = w / total
            acc += w

    def start(self, stage: str, message: str = "") -> None:
        self.update(stage, 0.0, message or stage)
        if self.log_fn:
            self.log_fn(f"{stage}…")

    def update(self, stage: str, fraction: float, message: str = "") -> None:
        if self.report:
            self.report(stage, self.starts[stage] + self.weights[stage] * max(0.0, min(1.0, fraction)), message or stage)

    def finish(self, stage: str, message: str = "") -> None:
        self.update(stage, 1.0, message or stage)

    def note(self, text: str) -> None:
        if self.log_fn:
            self.log_fn(text)


def aspect_ratio_text(w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        return ""
    r = w / h
    for name, v in (("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0), ("4:3", 4 / 3), ("3:4", 3 / 4), ("21:9", 21 / 9), ("4:5", 4 / 5)):
        if abs(r - v) < 0.03:
            return name
    return f"{w}:{h}"


class ReferenceFeatureExtractor:
    def __init__(self, ffmpeg: FFmpegService, settings: AnalysisSettings | None = None) -> None:
        self.ff = ffmpeg
        self.settings = settings or AnalysisSettings()
        self.sampler = FrameSampler(ffmpeg)

    # ------------------------------------------------------------------------------------------ entry point
    def extract(self, path: Path, metadata: ReferenceMetadata, *, progress: StageProgress | None = None, cancel: threading.Event | None = None) -> ExtractionResult:
        from app.reference.audio_analyzer import AudioAnalyzer, load_audio  # noqa: PLC0415  (imported here so a broken detector module cannot break the import of the others)
        from app.reference.caption_analyzer import CaptionTextAnalyzer  # noqa: PLC0415
        from app.reference.motion_analyzer import MotionAnalyzer  # noqa: PLC0415
        from app.reference.shot_detector import ShotDetector, compute_shot_stats  # noqa: PLC0415
        from app.reference.structure_analyzer import StructureAnalyzer  # noqa: PLC0415

        pg = progress or StageProgress(None)
        cfg = self.settings
        feats = ReferenceFeatures(metadata=metadata)
        events = ReferenceEvents()
        duration = float(metadata.duration)
        text_events: list[TextEvent] = []
        sections: list[Section] = []
        lines: list[str] = []

        def say(msg: str) -> None:
            lines.append(msg)
            pg.note(msg)

        def fail(name: str, exc: BaseException) -> None:
            if isinstance(exc, AnalysisCancelled):
                raise exc
            msg = exc.user_message if isinstance(exc, ReferenceAnalysisError) else f"{type(exc).__name__}: {exc}"
            log.warning("reference detector %s failed: %s", name, exc, exc_info=not isinstance(exc, ReferenceAnalysisError))
            feats.detectors.append(DetectorStatus(name, ok=False, confidence=0.0, message=str(msg)))
            feats.warnings.append(f"{name.replace('_', ' ').capitalize()} could not be completed ({msg}); the related readings are unavailable.")
            say(f"{name}: failed — {msg}")

        def ok(name: str, confidence: float, message: str = "", skipped: bool = False) -> None:
            feats.detectors.append(DetectorStatus(name, ok=True, confidence=float(max(0.0, min(1.0, confidence))), message=message, skipped=skipped))
            feats.confidence[name] = float(max(0.0, min(1.0, confidence)))

        signals = None
        shots: list = []

        # ---------------------------------------------------------------- visual pass: signals -> shots, transitions
        pg.start("Detecting Shots")
        try:
            expected = int(duration * cfg.signal_fps) if duration > 0 else None
            frames = self.sampler.frames(path, cfg.signal_fps, SIGNAL_W, SIGNAL_H, gray=True, cancel=cancel)
            signals = compute_signals(frames, cfg.signal_fps, expected_frames=expected, cancel=cancel,
                                      progress=lambda f: pg.update("Detecting Shots", f * 0.9, "Reading frames"))
            if duration <= 0:
                duration = signals.duration
                metadata.duration = duration
            det = ShotDetector(cfg.sensitivity).detect(signals)
            shots = det.shots
            feats.shots, feats.shot_stats, feats.transitions = det.shots, det.stats, det.transitions
            events.cut_times = [s.start for s in shots[1:]]
            events.transitions = [(s.start, s.transition_type, s.transition_duration) for s in shots[1:]]
            events.major_change_times = list(det.major_change_times)
            ok("shot_detection", det.confidence, "; ".join(det.notes))
            ok("transition_detection", min(1.0, det.confidence * 0.85), "Transitions are classified from the frame-difference profile around each cut.")
            say(f"Detected {len(shots)} shots ({feats.shot_stats.cuts_per_minute:.1f} cuts/min).")
        except AnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            fail("shot_detection", exc)
            fail("transition_detection", exc)
            fail("motion_detection", exc)
        pg.finish("Detecting Shots")

        # ---------------------------------------------------------------- motion (reads the signals; cheap)
        pg.start("Analyzing Motion")
        if signals is not None and shots:
            try:
                mres = MotionAnalyzer().analyze(signals, shots)
                feats.motion = mres.stats
                events.motion_series = list(mres.motion_series)
                ok("motion_detection", mres.confidence, "; ".join(mres.notes))
                say(f"Motion: {mres.stats.motion_class} ({mres.stats.motion_events_per_minute:.1f} events/min).")
            except AnalysisCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                fail("motion_detection", exc)
        elif feats.detector("motion_detection") is None:
            feats.detectors.append(DetectorStatus("motion_detection", ok=False, message="No shots were available."))
        pg.finish("Analyzing Motion")

        # ---------------------------------------------------------------- captions / text (second, RGB pass)
        pg.start("Analyzing Captions")
        if metadata.has_video:
            try:
                cres = CaptionTextAnalyzer(cfg.caption_fps).analyze_video(self.sampler, path, duration, shots or None,
                                                                          progress=lambda f: pg.update("Analyzing Captions", f, "Looking for on-screen text regions"), cancel=cancel)
                feats.captions, feats.text = cres.captions, cres.text
                text_events = list(cres.events)
                events.text_events = text_events
                ok("caption_detection", cres.caption_confidence, "; ".join(cres.notes))
                ok("text_detection", cres.text_confidence)
                say(f"Captions: {'present' if cres.captions.caption_present else 'none'}; text events/min {cres.text.text_events_per_minute:.1f}.")
            except AnalysisCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                fail("caption_detection", exc)
                fail("text_detection", exc)
        pg.finish("Analyzing Captions")

        # ---------------------------------------------------------------- audio
        pg.start("Analyzing Audio")
        try:
            samples = load_audio(self.ff, path, 16000, cancel=cancel) if metadata.has_audio else None
            if samples is None or len(samples) < 16000 * 0.5:
                feats.audio.has_audio = False
                ok("audio_detection", 0.0, "The reference has no usable audio stream; the audio dimensions are not measured.", skipped=True)
                feats.warnings.append("The reference has no audio: music, SFX and ducking are not part of its style profile.")
                say("No audio stream.")
            else:
                ares = AudioAnalyzer().analyze(samples, 16000, shot_times=list(events.cut_times), text_times=[t.start for t in text_events],
                                               change_times=list(events.major_change_times))
                feats.audio = ares.profile
                events.silences, events.sfx_times, events.music_change_times, events.audio_series = (
                    list(ares.silences), list(ares.sfx_times), list(ares.music_change_times), list(ares.series))
                ok("audio_detection", ares.confidence, "; ".join(ares.notes))
                say(f"Audio: voice {ares.profile.voice_dominance:.0%}, music {ares.profile.music_behavior}, SFX {ares.profile.sfx_class}.")
        except AnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            fail("audio_detection", exc)
        pg.finish("Analyzing Audio")

        # ---------------------------------------------------------------- structure (works from the observation lists only)
        pg.start("Building Style Profile")
        try:
            sres = StructureAnalyzer().analyze(events, duration, shots or None)
            feats.sections, feats.pacing_curve, feats.hook, feats.section_transition_style = sres.sections, sres.pacing_curve, sres.hook, sres.section_transition_style
            sections = sres.sections
            if shots:  # cuts-per-section needs the sections: recompute the shot statistics with them
                feats.shot_stats = compute_shot_stats(shots, duration, sections, len(events.major_change_times))
            ok("structure_detection", sres.confidence, "; ".join(sres.notes))
            say(f"Structure: {len(sections)} sections; hook intensity {sres.hook.intensity:.2f}.")
        except AnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            fail("structure_detection", exc)
        # pad detector status for keys no step reported (never leave a key silently 0 without a status)
        have = {d.name for d in feats.detectors}
        for k in CONFIDENCE_KEYS:
            if k not in have:
                feats.detectors.append(DetectorStatus(k, ok=False, message="Not run."))
        for k in CONFIDENCE_KEYS:
            feats.confidence.setdefault(k, 0.0)
        for d in feats.detectors:
            if d.ok and not d.skipped and d.confidence < LOW_CONFIDENCE:
                feats.warnings.append(f"{d.name.replace('_', ' ').capitalize()}: low confidence ({d.confidence:.0%}); treat the reading as an estimate.")
        pg.finish("Building Style Profile")
        return ExtractionResult(feats, events, lines)
