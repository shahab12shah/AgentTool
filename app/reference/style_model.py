"""The reference-analysis data model: raw measurements, normalised scores, the style profile and the comparison/similarity maths.

Everything stored here is an *abstract editing measurement* (rates, durations, shares, classes). Nothing in it can reproduce the reference
video: no frames, no text, no graphics, no sequence of shots. (Per-shot timings exist only inside the analysis cache, as observations;
the style profile and everything derived from it contain aggregates.)

The same ``StyleFeatures`` -> ``StyleScores`` pipeline is used for the reference *and* for the user's own project, so the two can be
compared and matched on exactly the same scale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from app.core.serialization import from_plain, to_plain

ANALYSIS_VERSION = 1  # bump when any detector/statistic changes meaning: cached analyses of older versions are re-run

TRANSITION_TYPES = ("CUT", "FADE", "DISSOLVE", "WIPE", "SLIDE", "UNKNOWN")
SECTION_KINDS = ("HOOK", "CONTEXT", "PROBLEM", "EXPLANATION", "EVIDENCE", "EXAMPLES", "REVEAL", "CONCLUSION", "CTA")
DIMENSIONS = ("pacing", "visual_density", "motion_intensity", "caption_density", "text_density", "transition_frequency", "music_presence", "sfx_frequency")
DIMENSION_LABELS = {"pacing": "Pacing", "visual_density": "Visual Density", "motion_intensity": "Motion", "caption_density": "Captions", "text_density": "Text Graphics",
                    "transition_frequency": "Transitions", "music_presence": "Music", "sfx_frequency": "SFX"}
# detector -> which style dimensions it supports (when a detector fails or is unsure, these dimensions are flagged, never invented)
CONFIDENCE_KEYS = ("shot_detection", "motion_detection", "caption_detection", "text_detection", "transition_detection", "audio_detection", "structure_detection")
DIMENSION_CONFIDENCE = {"pacing": "shot_detection", "visual_density": "shot_detection", "motion_intensity": "motion_detection", "caption_density": "caption_detection",
                        "text_density": "text_detection", "transition_frequency": "transition_detection", "music_presence": "audio_detection", "sfx_frequency": "audio_detection"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------- piecewise-linear scales
Points = tuple[tuple[float, float], ...]
PACING_POINTS: Points = ((0, 0), (3, 10), (6, 25), (12, 50), (24, 80), (40, 100))  # cuts per minute -> 0..100
OVERLAY_POINTS: Points = ((0, 0), (2, 20), (6, 50), (12, 80), (20, 100))  # text+graphic events per minute
MOTION_EVENT_POINTS: Points = ((0, 0), (3, 25), (8, 50), (15, 80), (25, 100))  # motion events per minute
MAJOR_CHANGE_POINTS: Points = ((0, 0), (1, 20), (3, 50), (6, 80), (10, 100))  # major visual changes per minute
TRANSITION_EVENT_POINTS: Points = ((0, 0), (1, 30), (3, 60), (6, 100))  # non-cut transitions per minute
CAPTION_RATE_POINTS: Points = ((0, 0), (5, 25), (12, 60), (25, 100))  # captions per minute
TEXT_POINTS: Points = ((0, 0), (1, 15), (3, 40), (6, 70), (12, 100))  # text overlays (not captions) per minute
TRANSITION_SHARE_POINTS: Points = ((0, 0), (0.05, 20), (0.15, 50), (0.35, 80), (0.6, 100))  # share of boundaries that are not hard cuts
SFX_POINTS: Points = ((0, 0), (1, 15), (3, 40), (8, 70), (15, 100))  # sound effects per minute


def scale(value: float, points: Points) -> float:
    """Map a raw measurement onto 0..100 along a piecewise-linear scale (clamped at both ends)."""
    if value <= points[0][0]:
        return float(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if value <= x1:
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return float(points[-1][1])


def unscale(score: float, points: Points) -> float:
    """The raw measurement that maps to ``score`` (inverse of ``scale``)."""
    score = max(points[0][1], min(points[-1][1], score))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if score <= y1:
            return x0 + (x1 - x0) * (score - y0) / (y1 - y0) if y1 > y0 else x0
    return float(points[-1][0])


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------------------------- classes / labels
def pacing_class(cuts_per_minute: float) -> str:
    return "Slow" if cuts_per_minute < 6 else "Moderate" if cuts_per_minute < 12 else "Fast" if cuts_per_minute < 24 else "Very Fast"


def density_label(score: float) -> str:
    """Spec scale: 0–20 Low · 21–40 Moderate-Low · 41–60 Medium · 61–80 High · 81–100 Very High."""
    return "Low" if score <= 20 else "Moderate-Low" if score <= 40 else "Medium" if score <= 60 else "High" if score <= 80 else "Very High"


def motion_class(score: float) -> str:
    return "Minimal" if score < 20 else "Subtle" if score < 40 else "Moderate" if score < 60 else "Strong" if score < 80 else "Aggressive"


def level_label(score: float) -> str:
    """A seven-step reading of any 0..100 score for the review screen."""
    return ("VERY LOW" if score < 15 else "LOW" if score < 35 else "MEDIUM-LOW" if score < 45 else "MEDIUM" if score < 58 else "MEDIUM-HIGH" if score < 70
            else "HIGH" if score < 85 else "VERY HIGH")


def frequency_label(score: float) -> str:
    return "None" if score <= 5 else "Rare" if score < 25 else "Occasional" if score < 50 else "Frequent" if score < 75 else "Constant"


def confidence_label(c: float) -> str:
    return "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low"


def sfx_class(per_minute: float) -> str:
    return "None" if per_minute < 0.25 else "Subtle" if per_minute < 3 else "Moderate" if per_minute < 8 else "Heavy"


def music_class(presence: float, dynamics: float) -> str:
    """presence: share of the video with music (0..1); dynamics: how much its level changes (0..1)."""
    if presence < 0.1:
        return "None"
    if presence < 0.35:
        return "Minimal"
    return "Dramatic" if dynamics >= 0.65 else "Dynamic" if dynamics >= 0.3 else "Continuous"


def distribution_buckets(durations: list[float]) -> dict[str, int]:
    edges = (("<1s", 0.0, 1.0), ("1-2s", 1.0, 2.0), ("2-4s", 2.0, 4.0), ("4-8s", 4.0, 8.0), ("8-15s", 8.0, 15.0), (">15s", 15.0, float("inf")))
    return {name: sum(1 for d in durations if lo <= d < hi) for name, lo, hi in edges}


# ---------------------------------------------------------------------------------------------- measurements
@dataclass
class ReferenceMetadata:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    aspect_ratio: str = ""
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int = 0
    has_video: bool = True
    has_audio: bool = False
    size_bytes: int = 0
    container: str = ""


@dataclass
class Shot:
    shot_id: str
    start: float
    end: float
    frame_sample: float = 0.0  # time of a representative sampled frame (the frame itself is not stored with the analysis)
    transition_type: str = "CUT"  # how this shot was entered (CUT for the first shot)
    transition_duration: float = 0.0
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ShotStats:
    count: int = 0
    average_shot_duration: float = 0.0
    median_shot_duration: float = 0.0
    minimum_shot_duration: float = 0.0
    maximum_shot_duration: float = 0.0
    std_shot_duration: float = 0.0
    shot_duration_distribution: dict[str, int] = field(default_factory=dict)
    cuts_per_minute: float = 0.0
    cut_frequency_class: str = "Slow"
    cuts_per_scene: float = 0.0  # cuts per structural section (see Section)
    cuts_per_section: list[float] = field(default_factory=list)
    major_change_per_minute: float = 0.0  # significant composition/content changes (not every cut)
    alternates_long_and_short: bool = False  # deliberate rhythm: the distribution is bimodal, the average alone would mislead


@dataclass
class PacingSegment:
    start: float
    end: float
    label: str  # Slow | Moderate | Fast | Very Fast
    cuts_per_minute: float = 0.0
    average_shot_duration: float = 0.0
    motion: float = 0.0
    text_per_minute: float = 0.0


@dataclass
class ZoomStats:
    events: int = 0
    frequency_per_minute: float = 0.0
    average_scale: float = 1.0  # average end/start scale ratio of a zoom event (1.08 = 8% growth)
    maximum_scale: float = 1.0
    average_duration: float = 0.0
    zoom_in_share: float = 1.0  # share of zoom events that go in (the rest go out)
    context: dict[str, float] = field(default_factory=dict)  # e.g. {"static_shots": .2, "punch_in": .1} aggregate shares only


@dataclass
class MotionStats:
    motion_events_per_minute: float = 0.0
    average_motion_intensity: float = 0.0  # 0..1
    zoom_frequency: float = 0.0  # per minute
    pan_frequency: float = 0.0  # per minute
    static_shot_share: float = 1.0
    high_motion_share: float = 0.0
    motion_class: str = "Minimal"
    zoom: ZoomStats = field(default_factory=ZoomStats)


@dataclass
class CaptionStats:
    caption_present: bool = False
    caption_coverage: float = 0.0  # share of the video duration with captions on screen (0..1)
    captions_per_minute: float = 0.0
    average_words_per_caption: float = 0.0  # estimate (no text is read when OCR is unavailable)
    average_chars_per_line: float = 0.0
    caption_line_count: float = 0.0  # average number of lines
    caption_position: str = "bottom"  # bottom | center | top | mixed
    caption_emphasis_rate: float = 0.0  # share of captions with a highlighted word (colour change inside the region)
    caption_animation_rate: float = 0.0  # share of captions that appear with a visible animation (pop/fade) rather than at once
    relative_text_height: float = 0.0  # caption height as a share of the frame height
    style_class: str = "Minimal"  # Minimal | Bold | News | Documentary | Social | High-Impact | Subtitle-focused
    traits: list[str] = field(default_factory=list)  # large_text, high_contrast, frequent_highlighting, short_caption_segments, center_position, bottom_position
    has_background_box: bool = False


@dataclass
class TextStats:
    text_events_per_minute: float = 0.0
    average_duration: float = 0.0
    headline_frequency: float = 0.0  # per minute
    number_graphic_frequency: float = 0.0
    lower_third_frequency: float = 0.0
    average_relative_size: float = 0.0
    position_share: dict[str, float] = field(default_factory=dict)  # top / center / bottom / left-lower aggregate shares
    animation_rate: float = 0.0
    graphic_events_per_minute: float = 0.0  # non-text overlays (boxes, highlights, charts) detected as graphics


@dataclass
class TransitionStats:
    transition_frequency: float = 0.0  # non-cut transitions per minute
    non_cut_share: float = 0.0  # share of shot boundaries that are not hard cuts
    transition_distribution: dict[str, float] = field(default_factory=dict)  # shares summing to 1 over TRANSITION_TYPES
    average_transition_duration: float = 0.0


@dataclass
class AudioProfile:
    has_audio: bool = False
    voice_dominance: float = 0.0  # share of the audio time where speech is the dominant element (0..1)
    music_presence: float = 0.0  # share of the video with music underneath (0..1)
    music_ducking_strength: float = 0.0  # 0..1: how far the music level falls while speech is active (relative to its level in gaps)
    music_behavior: str = "None"  # None | Minimal | Continuous | Dynamic | Dramatic
    music_dynamics: float = 0.0  # 0..1 variation of the music level
    music_changes_per_minute: float = 0.0
    sfx_per_minute: float = 0.0
    sfx_class: str = "None"  # None | Subtle | Moderate | Heavy
    sfx_intensity: float = 0.0  # 0..1 average loudness of sound effects relative to speech
    sfx_on_transitions: float = 0.0  # share of SFX events within 0.4 s of a shot boundary
    sfx_on_text: float = 0.0  # share within 0.4 s of a text overlay start
    sfx_on_reveals: float = 0.0  # share within 0.4 s of a strong visual change
    silence_frequency: float = 0.0  # silences per minute
    silence_percentage: float = 0.0  # 0..100
    average_pause_duration: float = 0.0
    long_pause_frequency: float = 0.0  # pauses >= 1.0 s per minute
    audio_dynamic_range: float = 0.0  # dB (loudness range between quiet and loud passages)
    loudness_changes_per_minute: float = 0.0


@dataclass
class VisualDensity:
    score: float = 0.0
    label: str = "Low"
    components: dict[str, float] = field(default_factory=dict)  # each 0..100 (cuts, overlays, motion, major_changes, transitions)


@dataclass
class Section:
    start: float
    end: float
    kind: str = "EXPLANATION"  # one of SECTION_KINDS
    confidence: float = 0.4
    cuts_per_minute: float = 0.0
    text_per_minute: float = 0.0
    motion: float = 0.0
    audio_intensity: float = 0.0
    average_shot_duration: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class HookWindow:
    seconds: int
    shot_rate: float = 0.0  # cuts per minute inside the window
    text_density: float = 0.0  # text events per minute
    motion: float = 0.0  # 0..1
    audio_intensity: float = 0.0  # 0..1 (relative loudness)
    visual_changes: float = 0.0  # major changes per minute
    number_emphasis: float = 0.0  # large centred overlays per minute (numbers/titles)
    relative_pacing: float = 0.0  # window shot_rate relative to the whole video (1.0 = same)


@dataclass
class HookProfile:
    windows: list[HookWindow] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)  # human-readable, abstract: "Fast visual switching", "Strong text", "High motion"
    intensity: float = 0.0  # 0..1 how much more intense the opening is than the rest


@dataclass
class SectionTransitionStyle:
    transitions_between_sections: dict[str, float] = field(default_factory=dict)  # share of section boundaries using CUT / FADE / ...
    pause_before_section: float = 0.0  # average silence (s) around a section boundary
    text_card_rate: float = 0.0  # share of boundaries with a text overlay
    music_change_rate: float = 0.0
    pacing_change_rate: float = 0.0  # share of boundaries where the cut rate changes clearly
    traits: list[str] = field(default_factory=list)


@dataclass
class DetectorStatus:
    """Per-detector outcome. A failed detector never fails the analysis: its dimensions are reported as unavailable."""

    name: str
    ok: bool = True
    confidence: float = 0.0
    message: str = ""
    skipped: bool = False  # e.g. "no audio stream"


@dataclass
class ReferenceFeatures:
    """Everything the analyzers measured (the cacheable analysis result). ``shots`` stays in the cache; the profile never exposes it."""

    metadata: ReferenceMetadata = field(default_factory=ReferenceMetadata)
    shots: list[Shot] = field(default_factory=list)
    shot_stats: ShotStats = field(default_factory=ShotStats)
    pacing_curve: list[PacingSegment] = field(default_factory=list)
    motion: MotionStats = field(default_factory=MotionStats)
    captions: CaptionStats = field(default_factory=CaptionStats)
    text: TextStats = field(default_factory=TextStats)
    transitions: TransitionStats = field(default_factory=TransitionStats)
    audio: AudioProfile = field(default_factory=AudioProfile)
    sections: list[Section] = field(default_factory=list)
    hook: HookProfile = field(default_factory=HookProfile)
    section_transition_style: SectionTransitionStyle = field(default_factory=SectionTransitionStyle)
    confidence: dict[str, float] = field(default_factory=dict)  # CONFIDENCE_KEYS -> 0..1
    detectors: list[DetectorStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def detector(self, name: str) -> DetectorStatus | None:
        return next((d for d in self.detectors if d.name == name), None)

    def available(self, confidence_key: str) -> bool:
        d = self.detector(confidence_key)
        return d is None or (d.ok and not d.skipped)


@dataclass
class TextEvent:
    """One on-screen text/graphic event as *measured* (geometry and timing only — the text itself is never read or stored)."""

    start: float
    end: float
    kind: str = "TEXT"  # CAPTION | HEADLINE | NUMBER_CARD | LOWER_THIRD | TEXT | GRAPHIC
    position: str = "bottom"  # top | center | bottom | lower_left
    relative_height: float = 0.0  # text height / frame height
    relative_width: float = 0.0
    lines: int = 1
    animated: bool = False
    emphasized: bool = False  # a highlighted (differently coloured) region inside the text
    has_box: bool = False
    confidence: float = 0.5

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ReferenceEvents:
    """Time-stamped observations from all analyzers, kept in the analysis cache for the structure analysis and for SFX/transition alignment.
    The style profile and everything derived from it never contain these lists."""

    cut_times: list[float] = field(default_factory=list)  # shot boundaries
    transitions: list[tuple[float, str, float]] = field(default_factory=list)  # (time, type, duration)
    major_change_times: list[float] = field(default_factory=list)
    text_events: list[TextEvent] = field(default_factory=list)
    motion_series: list[tuple[float, float]] = field(default_factory=list)  # (time, intensity 0..1), coarse (one per second or so)
    audio_series: list[tuple[float, float, float, float]] = field(default_factory=list)  # (time, loudness 0..1, voice 0..1, music 0..1)
    silences: list[tuple[float, float]] = field(default_factory=list)  # (start, end)
    sfx_times: list[float] = field(default_factory=list)
    music_change_times: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- raw -> scores
@dataclass
class StyleFeatures:
    """The raw abstract measurements one video is reduced to. Both a reference and a project produce one; scores are computed from it."""

    duration: float = 0.0
    average_shot_duration: float = 0.0
    median_shot_duration: float = 0.0
    cuts_per_minute: float = 0.0
    major_changes_per_minute: float = 0.0
    overlay_events_per_minute: float = 0.0  # text + graphics (not captions)
    text_events_per_minute: float = 0.0
    graphic_events_per_minute: float = 0.0
    motion_events_per_minute: float = 0.0
    zoom_events_per_minute: float = 0.0
    average_motion_intensity: float = 0.0
    caption_coverage: float = 0.0
    captions_per_minute: float = 0.0
    average_words_per_caption: float = 0.0
    caption_emphasis_rate: float = 0.0
    transition_events_per_minute: float = 0.0
    non_cut_share: float = 0.0
    voice_dominance: float = 0.0
    music_presence: float = 0.0
    music_ducking_strength: float = 0.0
    music_dynamics: float = 0.0
    sfx_per_minute: float = 0.0
    silence_percentage: float = 0.0
    average_pause_duration: float = 0.0
    long_pause_frequency: float = 0.0
    audio_dynamic_range: float = 0.0
    hook_intensity: float = 0.0


@dataclass
class StyleScores:
    """The eight comparable 0..100 dimensions (a relative, analytical scale — not a quality score)."""

    pacing: float = 0.0
    visual_density: float = 0.0
    motion_intensity: float = 0.0
    caption_density: float = 0.0
    text_density: float = 0.0
    transition_frequency: float = 0.0
    music_presence: float = 0.0
    sfx_frequency: float = 0.0
    visual_density_components: dict[str, float] = field(default_factory=dict)

    def get(self, dim: str) -> float:
        return float(getattr(self, dim))

    def as_dict(self) -> dict[str, float]:
        return {d: round(self.get(d), 1) for d in DIMENSIONS}


def compute_scores(f: StyleFeatures) -> StyleScores:
    pacing = scale(f.cuts_per_minute, PACING_POINTS)
    comps = {
        "cuts": pacing,
        "overlays": scale(f.overlay_events_per_minute, OVERLAY_POINTS),
        "motion": scale(f.motion_events_per_minute, MOTION_EVENT_POINTS),
        "major_changes": scale(f.major_changes_per_minute, MAJOR_CHANGE_POINTS),
        "transitions": scale(f.transition_events_per_minute, TRANSITION_EVENT_POINTS),
    }
    density = 0.35 * comps["cuts"] + 0.20 * comps["overlays"] + 0.20 * comps["motion"] + 0.15 * comps["major_changes"] + 0.10 * comps["transitions"]
    motion = 0.5 * scale(f.motion_events_per_minute, MOTION_EVENT_POINTS) + 0.5 * 100.0 * clamp(f.average_motion_intensity)
    caption = (0.6 * 100.0 * clamp(f.caption_coverage) + 0.4 * scale(f.captions_per_minute, CAPTION_RATE_POINTS)) if f.caption_coverage > 0 or f.captions_per_minute > 0 else 0.0
    return StyleScores(pacing, density, motion, caption, scale(f.text_events_per_minute, TEXT_POINTS), scale(f.non_cut_share, TRANSITION_SHARE_POINTS),
                       100.0 * clamp(f.music_presence), scale(f.sfx_per_minute, SFX_POINTS), comps)


def features_from_reference(ref: ReferenceFeatures) -> StyleFeatures:
    st, mo, cap, tx, tr, au = ref.shot_stats, ref.motion, ref.captions, ref.text, ref.transitions, ref.audio
    return StyleFeatures(
        ref.metadata.duration, st.average_shot_duration, st.median_shot_duration, st.cuts_per_minute, st.major_change_per_minute,
        tx.text_events_per_minute + tx.graphic_events_per_minute, tx.text_events_per_minute, tx.graphic_events_per_minute, mo.motion_events_per_minute, mo.zoom_frequency,
        mo.average_motion_intensity, cap.caption_coverage, cap.captions_per_minute, cap.average_words_per_caption, cap.caption_emphasis_rate, tr.transition_frequency,
        tr.non_cut_share, au.voice_dominance, au.music_presence, au.music_ducking_strength, au.music_dynamics, au.sfx_per_minute, au.silence_percentage,
        au.average_pause_duration, au.long_pause_frequency, au.audio_dynamic_range, ref.hook.intensity)


# ---------------------------------------------------------------------------------------------- the profile
@dataclass
class ReferenceStyleProfile:
    """The normalised, reviewable style of a reference. Every number comes from the analysis; unavailable dimensions are marked, not guessed."""

    reference_id: str = ""
    analysis_version: int = ANALYSIS_VERSION
    created_at: str = field(default_factory=now_iso)
    features: StyleFeatures = field(default_factory=StyleFeatures)
    scores: StyleScores = field(default_factory=StyleScores)
    categories: dict[str, str] = field(default_factory=dict)  # human-readable class per dimension ("Fast", "High", "Subtle"...)
    levels: dict[str, str] = field(default_factory=dict)  # seven-step reading per dimension
    confidence: dict[str, float] = field(default_factory=dict)  # per detector (CONFIDENCE_KEYS) + "overall"
    unavailable: list[str] = field(default_factory=list)  # dimensions that could not be measured (their scores are 0 and must not be applied)
    shot_stats: ShotStats = field(default_factory=ShotStats)
    pacing_curve: list[PacingSegment] = field(default_factory=list)
    motion: MotionStats = field(default_factory=MotionStats)
    caption_style: CaptionStats = field(default_factory=CaptionStats)
    text: TextStats = field(default_factory=TextStats)
    transitions: TransitionStats = field(default_factory=TransitionStats)
    audio: AudioProfile = field(default_factory=AudioProfile)
    hook: HookProfile = field(default_factory=HookProfile)
    section_transition_style: SectionTransitionStyle = field(default_factory=SectionTransitionStyle)
    section_mix: dict[str, float] = field(default_factory=dict)  # share of the video per section kind (aggregate)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    def score(self, dim: str) -> float:
        return self.scores.get(dim)

    def is_available(self, dim: str) -> bool:
        return dim not in self.unavailable

    def dimension_confidence(self, dim: str) -> float:
        return float(self.confidence.get(DIMENSION_CONFIDENCE[dim], 0.0))

    def rows(self) -> list[tuple[str, str, float, str, str]]:
        """(dimension, label, score, level/category text, confidence label) for the review table."""
        out = []
        for d in DIMENSIONS:
            if d in self.unavailable:
                out.append((d, DIMENSION_LABELS[d], 0.0, "UNAVAILABLE", "—"))
            else:
                out.append((d, DIMENSION_LABELS[d], self.score(d), self.levels.get(d, level_label(self.score(d))), confidence_label(self.dimension_confidence(d))))
        return out

    def summary_lines(self) -> list[str]:
        lines = []
        for d in DIMENSIONS:
            cat = "Unavailable" if d in self.unavailable else self.categories.get(d, "")
            low = "" if d in self.unavailable or self.dimension_confidence(d) >= 0.5 else "  (low confidence)"
            lines.append(f"{DIMENSION_LABELS[d]}: {cat}{low}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceStyleProfile":
        return from_plain(cls, d)

    def signature(self) -> str:
        return hashlib.sha1(json.dumps([self.scores.as_dict(), self.analysis_version, sorted(self.unavailable)], sort_keys=True).encode()).hexdigest()[:12]


def categories_for(ref: ReferenceFeatures, scores: StyleScores, f: StyleFeatures) -> dict[str, str]:
    return {
        "pacing": pacing_class(f.cuts_per_minute),
        "visual_density": density_label(scores.visual_density),
        "motion_intensity": motion_class(scores.motion_intensity),
        "caption_density": frequency_label(scores.caption_density),
        "text_density": frequency_label(scores.text_density) if scores.text_density > 5 else "None",
        "transition_frequency": "Minimal" if scores.transition_frequency < 20 else "Occasional" if scores.transition_frequency < 45 else "Frequent" if scores.transition_frequency < 70 else "Heavy",
        "music_presence": ref.audio.music_behavior if ref.audio.has_audio else "None",
        "sfx_frequency": ref.audio.sfx_class if ref.audio.has_audio else "None",
    }


def build_profile_from_features(f: StyleFeatures, reference_id: str = "", *, music_behavior: str | None = None, sfx_label: str | None = None, has_audio: bool = True,
                                confidence: float = 1.0) -> ReferenceStyleProfile:
    """A profile straight from raw features (used for the user's own project, where nothing needs detecting: confidence is high by construction)."""
    scores = compute_scores(f)
    ref = ReferenceFeatures()
    ref.audio.has_audio = has_audio
    ref.audio.music_behavior = music_behavior or music_class(f.music_presence, f.music_dynamics)
    ref.audio.sfx_class = sfx_label or sfx_class(f.sfx_per_minute)
    ref.shot_stats.cuts_per_minute = f.cuts_per_minute
    ref.shot_stats.average_shot_duration = f.average_shot_duration
    ref.shot_stats.median_shot_duration = f.median_shot_duration
    ref.shot_stats.cut_frequency_class = pacing_class(f.cuts_per_minute)
    return ReferenceStyleProfile(reference_id, ANALYSIS_VERSION, now_iso(), f, scores, categories_for(ref, scores, f), {d: level_label(scores.get(d)) for d in DIMENSIONS},
                                 {**{k: confidence for k in CONFIDENCE_KEYS}, "overall": confidence}, [], ref.shot_stats, [], MotionStats(f.motion_events_per_minute, f.average_motion_intensity,
                                 f.zoom_events_per_minute), CaptionStats(f.caption_coverage > 0, f.caption_coverage, f.captions_per_minute, f.average_words_per_caption),
                                 TextStats(f.text_events_per_minute, graphic_events_per_minute=f.graphic_events_per_minute), TransitionStats(f.transition_events_per_minute, f.non_cut_share),
                                 AudioProfile(has_audio, f.voice_dominance, f.music_presence, f.music_ducking_strength, ref.audio.music_behavior, f.music_dynamics, 0.0, f.sfx_per_minute,
                                              ref.audio.sfx_class, 0.0, 0.0, 0.0, 0.0, 0.0, f.silence_percentage, f.average_pause_duration, f.long_pause_frequency, f.audio_dynamic_range))


def build_profile(ref: ReferenceFeatures, reference_id: str = "") -> ReferenceStyleProfile:
    """Features -> normalised, labelled profile. Dimensions whose detector failed/was skipped are marked unavailable (never guessed)."""
    f = features_from_reference(ref)
    scores = compute_scores(f)
    unavailable = []
    for dim, key in DIMENSION_CONFIDENCE.items():
        d = ref.detector(key)
        if d is not None and (not d.ok or d.skipped):
            unavailable.append(dim)
    if "pacing" in unavailable and "visual_density" not in unavailable:
        unavailable.append("visual_density")  # density needs the cuts
    conf = {k: float(ref.confidence.get(k, 0.0)) for k in CONFIDENCE_KEYS}
    known = [v for k, v in conf.items() if ref.detector(k) is None or (ref.detector(k).ok and not ref.detector(k).skipped)]  # type: ignore[union-attr]
    conf["overall"] = round(sum(known) / len(known), 3) if known else 0.0
    mix: dict[str, float] = {}
    total = sum(s.duration for s in ref.sections) or 0.0
    for s in ref.sections:
        mix[s.kind] = mix.get(s.kind, 0.0) + (s.duration / total if total else 0.0)
    warnings = list(ref.warnings)
    for dim in DIMENSIONS:
        if dim not in unavailable and conf.get(DIMENSION_CONFIDENCE[dim], 0.0) < 0.5:
            warnings.append(f"{DIMENSION_LABELS[dim]}: low confidence — treat this reading as an estimate.")
    return ReferenceStyleProfile(reference_id, ANALYSIS_VERSION, now_iso(), f, scores, categories_for(ref, scores, f),
                                 {d: level_label(scores.get(d)) for d in DIMENSIONS}, conf, unavailable, ref.shot_stats, ref.pacing_curve, ref.motion, ref.captions, ref.text,
                                 ref.transitions, ref.audio, ref.hook, ref.section_transition_style, {k: round(v, 3) for k, v in mix.items()}, warnings)


# ---------------------------------------------------------------------------------------------- comparison and similarity
@dataclass
class ComparisonRow:
    key: str
    label: str
    reference: str
    project: str
    reference_value: float = 0.0
    project_value: float = 0.0
    note: str = ""


@dataclass
class StyleSimilarityScore:
    """How close the *editing features* of two videos are. This is NOT a copyright / content similarity measure."""

    pacing_match: float = 0.0
    motion_match: float = 0.0
    caption_match: float = 0.0
    text_match: float = 0.0
    audio_match: float = 0.0
    visual_density_match: float = 0.0
    transition_match: float = 0.0
    overall: float = 0.0
    compared: list[str] = field(default_factory=list)  # dimensions that took part (unavailable ones are excluded, not scored as 0)


SIMILARITY_WEIGHTS = {"pacing": 0.22, "visual_density": 0.14, "motion_intensity": 0.16, "caption_density": 0.14, "text_density": 0.10, "transition_frequency": 0.06,
                      "music_presence": 0.10, "sfx_frequency": 0.08}


def similarity(ref: StyleScores, other: StyleScores, skip: list[str] | None = None) -> StyleSimilarityScore:
    skip = set(skip or [])

    def match(d: str) -> float:
        return round(100.0 - abs(ref.get(d) - other.get(d)), 1)

    m = {d: match(d) for d in DIMENSIONS}
    used = [d for d in DIMENSIONS if d not in skip]
    wsum = sum(SIMILARITY_WEIGHTS[d] for d in used)
    overall = sum(SIMILARITY_WEIGHTS[d] * m[d] for d in used) / wsum if wsum else 0.0
    audio = [m[d] for d in ("music_presence", "sfx_frequency") if d in used]
    return StyleSimilarityScore(m["pacing"], m["motion_intensity"], m["caption_density"], m["text_density"], round(sum(audio) / len(audio), 1) if audio else 0.0, m["visual_density"],
                                m["transition_frequency"], round(overall, 1), used)


def compare(ref: ReferenceStyleProfile, project: ReferenceStyleProfile) -> list[ComparisonRow]:
    """REFERENCE vs CURRENT PROJECT, on the same abstract measurements (rows the user can act on)."""
    rf, pf = ref.features, project.features

    def row(key: str, label: str, a: float, b: float, fmt: str = "{:.1f}", unit: str = "", note: str = "") -> ComparisonRow:
        return ComparisonRow(key, label, fmt.format(a) + unit, fmt.format(b) + unit, a, b, note)

    return [
        row("average_shot_duration", "Shot Duration", rf.average_shot_duration, pf.average_shot_duration, "{:.1f}", "s"),
        row("cuts_per_minute", "Cuts/min", rf.cuts_per_minute, pf.cuts_per_minute),
        ComparisonRow("pacing", "Pacing", ref.categories.get("pacing", ""), project.categories.get("pacing", ""), ref.score("pacing"), project.score("pacing")),
        ComparisonRow("visual_density", "Visual Density", ref.categories.get("visual_density", ""), project.categories.get("visual_density", ""), ref.score("visual_density"), project.score("visual_density")),
        ComparisonRow("motion_intensity", "Motion", ref.categories.get("motion_intensity", ""), project.categories.get("motion_intensity", ""), ref.score("motion_intensity"), project.score("motion_intensity")),
        ComparisonRow("caption_density", "Captions", ref.categories.get("caption_density", ""), project.categories.get("caption_density", ""), ref.score("caption_density"), project.score("caption_density")),
        ComparisonRow("text_density", "Text", ref.categories.get("text_density", ""), project.categories.get("text_density", ""), ref.score("text_density"), project.score("text_density")),
        ComparisonRow("transition_frequency", "Transitions", ref.categories.get("transition_frequency", ""), project.categories.get("transition_frequency", ""), ref.score("transition_frequency"), project.score("transition_frequency")),
        ComparisonRow("music_presence", "Music", ref.categories.get("music_presence", ""), project.categories.get("music_presence", ""), ref.score("music_presence"), project.score("music_presence")),
        ComparisonRow("sfx_frequency", "SFX", ref.categories.get("sfx_frequency", ""), project.categories.get("sfx_frequency", ""), ref.score("sfx_frequency"), project.score("sfx_frequency")),
    ]


_ = fields
