"""QC settings: every threshold the analysers use lives here (nothing is hard-coded inside a checker), is saved with the project (``project.qc_settings``) and is editable.

Grouped by what they govern. ``SETTING_DEFS`` describes each user-facing threshold (label, range, help) so the UI can offer an advanced editor
without knowing the analysers. Sensitivities are 0..1 (0.5 = default); a checker scales its own thresholds by them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.core.serialization import from_plain, to_plain
from app.qc.severity import BlockLevel

ALL_CHECKERS = ("preflight", "timeline", "scene", "sync", "visual", "continuity", "pacing", "caption", "text", "motion", "transition", "audio", "frames", "asset", "render", "editorial")

# fix kinds the engine knows, and whether each may ever run without a confirmation click (the user's permission table; "never" disables the fix)
FIX_KINDS_SAFE = ("caption.retime", "caption.safe_margin", "audio.duck", "clip.extend", "clip.remove_empty", "asset.relink", "param.normalize")
FIX_KINDS_CONFIRM = ("visual.replace", "scene.restructure", "clip.delete", "narration.retime", "silence.remove", "graphics.change", "text.change", "motion.soften", "gap.close", "transition.shorten",
                     "caption.restyle")


@dataclass
class SyncThresholds:
    minor_ms: float = 100.0  # drift between minor and moderate: NOTICE
    moderate_ms: float = 250.0  # moderate: WARNING
    major_ms: float = 500.0  # above: ERROR
    caption_early_ms: float = 150.0  # a caption that appears this much before its first spoken word
    visual_late_ms: float = 400.0  # a visual that arrives this long after the statement it illustrates
    emphasis_miss_ms: float = 300.0  # an emphasis animation this far from the spoken word


@dataclass
class CoverageThresholds:
    min_covered_ratio: float = 0.98  # share of a scene's narration with a visual on screen
    max_hold_seconds: float = 12.0  # one visual on screen longer than this (stills: ``max_hold_still_seconds``)
    max_hold_still_seconds: float = 9.0
    max_cuts_per_sentence: float = 2.0  # visual changes inside one simple sentence
    important_scene: float = 0.7  # scenes at/above this importance need real visual support for claims, numbers, dates, warnings and evidence
    min_shot_seconds: float = 0.8  # shots shorter than this are "very short"
    gap_error_seconds: float = 1.0  # an unintended picture gap under narration longer than this is an ERROR (shorter: a WARNING)
    error_uncovered_seconds: float = 1.0  # narration without a picture for longer than this (scaled by scene importance) is an ERROR
    hold_warning_factor: float = 1.5  # a hold this many times the limit is a WARNING instead of a NOTICE
    simple_sentence_words: int = 14  # a sentence up to this long is "simple": it carries at most ``max_cuts_per_sentence`` picture changes
    short_cluster: int = 3  # this many consecutive too-short shots in a row are a cluster


@dataclass
class VisualThresholds:
    error_below: float = 40.0  # current QC score below this: ERROR
    warning_below: float = 58.0
    notice_below: float = 70.0
    min_confidence: float = 45.0  # semantic judgements below this confidence are not reported
    weights: dict[str, float] = field(default_factory=lambda: {"semantic": 35.0, "subject": 20.0, "context": 15.0, "action": 10.0, "timing": 10.0, "quality": 5.0, "source": 5.0})
    context_bonus_cap: float = 8.0  # how much the neighbouring scenes may move the score either way


@dataclass
class RepetitionThresholds:
    sensitivity: float = 0.5
    adjacent_scenes: int = 2  # the same asset in scenes this close is "adjacent"
    max_uses: int = 3  # uses of one asset before it is flagged (scaled by sensitivity)
    min_gap_seconds: float = 20.0
    intentional_assets: list[str] = field(default_factory=list)  # asset ids the user marked as deliberate callbacks (logos, recurring person, charts)


@dataclass
class ContinuityThresholds:
    sensitivity: float = 0.5
    topic_jump_similarity: float = 0.10  # lexical similarity between neighbouring scenes below which a hard subject change is suspected
    scale_jump: float = 1.8
    brightness_jump: float = 0.45


@dataclass
class PacingThresholds:
    sensitivity: float = 0.5
    too_fast_cuts_per_minute: float = 36.0
    too_slow_shot_seconds: float = 14.0
    uneven_ratio: float = 2.6  # a window's cut rate against the video's median window
    window_seconds: float = 20.0
    over_edit_events_per_minute: float = 30.0  # cuts + effects + graphics together
    cut_in_phrase_ms: float = 200.0  # a cut this close to the middle of a spoken phrase is "awkward"
    micro_cut_seconds: float = 0.6
    micro_cut_cluster: int = 3  # this many micro-cuts inside 4 s


@dataclass
class CaptionThresholds:
    max_chars_per_line: int = 42
    max_lines: int = 2
    max_words_displayed: int = 14
    max_cps: float = 21.0  # characters per second
    min_duration: float = 0.45
    flicker_seconds: float = 0.4  # consecutive captions each shorter than this
    flicker_count: int = 3
    min_safe_margin: float = 0.03  # share of the frame
    min_relative_font: float = 0.028  # caption font size / frame height
    min_contrast: float = 3.0  # WCAG-like contrast ratio of text against its background box / assumed backdrop
    max_animation_seconds: float = 0.8


@dataclass
class MotionThresholds:
    max_scale_per_second: float = 0.30  # change of scale per second (1.00 -> 1.35 in 0.4 s is 0.875/s)
    max_total_scale: float = 1.6
    max_rotation_per_second: float = 40.0
    keyframe_jump: float = 0.30  # fractional change within one frame
    max_zoom_during_text: float = 0.12  # zoom rate allowed under readable text


@dataclass
class TransitionThresholds:
    max_duration: float = 1.5
    max_per_minute: float = 8.0
    min_spacing: float = 1.2
    important_phrase_margin_ms: float = 150.0


@dataclass
class AudioThresholds:
    clip_dbfs: float = -0.3  # sample peak at/above this is clipping
    clip_samples: int = 3  # consecutive near-full-scale samples that make it audible clipping
    voice_min_rms_dbfs: float = -34.0  # voice quieter than this over a spoken stretch
    voice_jump_db: float = 9.0  # sudden level change between adjacent voiced stretches
    music_over_voice_db: float = -8.0  # music louder than (voice + this) while speaking: insufficient ducking
    min_duck_db: float = 5.0  # the music must fall by at least this under speech
    sfx_over_voice_db: float = -3.0
    sfx_repeat_per_minute: float = 8.0
    unnatural_silence_seconds: float = 2.5  # silence with no pause in the script around it
    accidental_gap_seconds: float = 0.7  # silence where the script says words are spoken
    voice_duration_tolerance: float = 0.5  # |voice duration - timeline end|
    noise_floor_dbfs: float = -45.0


@dataclass
class FrameThresholds:
    black_sensitivity: float = 0.5  # 0.5 -> pix_th 0.10
    black_min_seconds: float = 0.5
    frozen_sensitivity: float = 0.5
    frozen_min_seconds: float = 2.0
    intentional_black: list[list[float]] = field(default_factory=list)  # [[start, end], ...] the user declared deliberate blackouts
    analyze_assets: bool = True  # scan the used ranges of source videos for black / frozen stretches
    max_scan_seconds: float = 600.0  # budget for source scanning (largest first, the rest is skipped and reported)


@dataclass
class MediaThresholds:
    min_source_ratio: float = 0.5  # source width / output width below which the picture will look soft
    reject_below_ratio: float = 0.0  # 0 = never reject automatically
    max_upscale: float = 2.5
    expected_aspect_tolerance: float = 0.12


@dataclass
class StyleThresholds:
    check: bool = True
    tolerance: float = 28.0  # points on the 0..100 style scale before a deviation from the applied reference style is mentioned


@dataclass
class QCSettings:
    # ---- gate / behaviour
    block_level: str = BlockLevel.CRITICAL_ERROR.value
    allow_export_override: bool = False  # may the user continue past a blocking ERROR / WARNING (a CRITICAL is never overridable)
    run_before_export: bool = True
    post_render_qc: bool = True
    enabled_checkers: list[str] = field(default_factory=lambda: list(ALL_CHECKERS))
    ai_review_enabled: bool = True
    ai_provider: str = "local"
    ai_confidence_caps: list[list[Any]] = field(default_factory=lambda: [[50.0, "NOTICE"], [70.0, "WARNING"]])  # confidence < 50 -> at most NOTICE; < 70 -> at most WARNING
    min_confidence_to_report: float = 35.0
    # ---- markers on the timeline
    marker_mode: str = "all"  # all | critical_only | hidden
    # ---- auto-fix permissions: kind -> auto | confirm | never ("auto" is only honoured for kinds that are safe by construction)
    fix_permissions: dict[str, str] = field(default_factory=lambda: {**{k: "auto" for k in FIX_KINDS_SAFE}, **{k: "confirm" for k in FIX_KINDS_CONFIRM}})
    max_caption_shift_seconds: float = 0.6  # a "small" caption timing correction
    max_clip_extension_seconds: float = 1.0
    duck_target_db: float = -14.0  # how far under the voice the fix leaves the music
    # ---- score weights of the eight groups (the overall score is their weighted mean; criticals are never hidden by it)
    group_weights: dict[str, float] = field(default_factory=lambda: {"visual_accuracy": 20.0, "sync": 15.0, "pacing": 10.0, "captions": 10.0, "audio": 15.0, "continuity": 10.0, "timeline": 10.0,
                                                                     "technical": 10.0})
    # ---- intentional things the user declared (never reported as problems)
    intentional_gaps: list[list[float]] = field(default_factory=list)  # [[start, end], ...] deliberate empty stretches
    # ---- thresholds
    sync: SyncThresholds = field(default_factory=SyncThresholds)
    coverage: CoverageThresholds = field(default_factory=CoverageThresholds)
    visual: VisualThresholds = field(default_factory=VisualThresholds)
    repetition: RepetitionThresholds = field(default_factory=RepetitionThresholds)
    continuity: ContinuityThresholds = field(default_factory=ContinuityThresholds)
    pacing: PacingThresholds = field(default_factory=PacingThresholds)
    caption: CaptionThresholds = field(default_factory=CaptionThresholds)
    motion: MotionThresholds = field(default_factory=MotionThresholds)
    transition: TransitionThresholds = field(default_factory=TransitionThresholds)
    audio: AudioThresholds = field(default_factory=AudioThresholds)
    frames: FrameThresholds = field(default_factory=FrameThresholds)
    media: MediaThresholds = field(default_factory=MediaThresholds)
    style: StyleThresholds = field(default_factory=StyleThresholds)

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QCSettings":
        return from_plain(cls, d)

    def section(self, name: str) -> Any:
        return getattr(self, name)

    def subset_hash(self, *names: str) -> str:
        """Fingerprint of the named top-level settings (a checker's cache key covers only what it reads)."""
        d = self.to_dict()
        return hashlib.sha1(json.dumps({n: d.get(n) for n in sorted(names)}, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def version(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]

    # what does NOT change an analysis: how results are *presented or acted on* (gate level, override, markers, fix permissions, score weights).
    # Changing these never makes a QC run stale; the service re-derives scores / fix flags from them at once.
    PRESENTATION_FIELDS = ("block_level", "allow_export_override", "run_before_export", "post_render_qc", "marker_mode", "fix_permissions", "group_weights")

    def analysis_version(self) -> str:
        """Fingerprint of the settings that can change what QC *finds* (thresholds, enabled checkers, AI settings, declared intentional ranges ...)."""
        d = {k: v for k, v in self.to_dict().items() if k not in self.PRESENTATION_FIELDS}
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def set_path(self, path: str, value: Any) -> None:
        """Set ``"sync.major_ms"``-style paths (the advanced editor); the type of the current value decides the conversion."""
        parts = path.split(".")
        obj: Any = self
        for p in parts[:-1]:
            obj = getattr(obj, p)
        cur = getattr(obj, parts[-1])
        setattr(obj, parts[-1], type(cur)(value) if not isinstance(cur, (list, dict)) and cur is not None else value)

    def get_path(self, path: str) -> Any:
        obj: Any = self
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj

    def permission(self, kind: str) -> str:
        return self.fix_permissions.get(kind, "confirm")


@dataclass
class SettingDef:
    path: str
    label: str
    kind: str = "float"  # float | int | bool | choice
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    help: str = ""
    choices: tuple[str, ...] = ()
    group: str = "General"


SETTING_DEFS: tuple[SettingDef, ...] = (
    SettingDef("block_level", "Block export on", "choice", help="Which severities stop an export.", choices=tuple(b.value for b in BlockLevel), group="Export"),
    SettingDef("allow_export_override", "Allow continuing past blocking Errors / Warnings", "bool", group="Export", help="Critical issues can never be overridden."),
    SettingDef("run_before_export", "Run QC before every export", "bool", group="Export"),
    SettingDef("post_render_qc", "Check the rendered file after export", "bool", group="Export"),
    SettingDef("min_confidence_to_report", "Hide AI judgements below confidence", "float", 0, 100, 5, group="AI review"),
    SettingDef("visual.error_below", "Visual accuracy: error below", "float", 0, 100, 1, "Current in-context score under which a visual is an ERROR.", group="Visual accuracy"),
    SettingDef("visual.warning_below", "Visual accuracy: warning below", "float", 0, 100, 1, group="Visual accuracy"),
    SettingDef("sync.minor_ms", "Sync drift: minor (ms)", "float", 10, 2000, 10, group="Synchronisation"),
    SettingDef("sync.moderate_ms", "Sync drift: moderate (ms)", "float", 10, 3000, 10, group="Synchronisation"),
    SettingDef("sync.major_ms", "Sync drift: major (ms)", "float", 10, 5000, 10, group="Synchronisation"),
    SettingDef("coverage.min_shot_seconds", "Shortest acceptable shot (s)", "float", 0.1, 5, 0.1, group="Pacing"),
    SettingDef("coverage.max_hold_seconds", "Longest acceptable hold, video (s)", "float", 2, 60, 1, group="Pacing"),
    SettingDef("repetition.sensitivity", "Repetition sensitivity", "float", 0, 1, 0.05, group="Visual continuity"),
    SettingDef("continuity.sensitivity", "Continuity sensitivity", "float", 0, 1, 0.05, group="Visual continuity"),
    SettingDef("pacing.sensitivity", "Pacing sensitivity", "float", 0, 1, 0.05, group="Pacing"),
    SettingDef("caption.max_chars_per_line", "Caption: max characters per line", "int", 10, 80, 1, group="Captions"),
    SettingDef("caption.max_cps", "Caption: max characters per second", "float", 5, 40, 0.5, group="Captions"),
    SettingDef("audio.clip_dbfs", "Audio: clipping threshold (dBFS)", "float", -6, 0, 0.1, group="Audio"),
    SettingDef("audio.min_duck_db", "Audio: minimum ducking (dB)", "float", 0, 30, 0.5, group="Audio"),
    SettingDef("frames.black_sensitivity", "Black-frame sensitivity", "float", 0, 1, 0.05, group="Frames"),
    SettingDef("frames.frozen_sensitivity", "Frozen-frame sensitivity", "float", 0, 1, 0.05, group="Frames"),
    SettingDef("media.min_source_ratio", "Low-resolution warning below (source/output width)", "float", 0.1, 1, 0.05, group="Media"),
)

