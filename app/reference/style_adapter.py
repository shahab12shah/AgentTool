"""ReferenceStyleAdapter: reference style profile + the user's adjustments + the project's content  ->  EditingStrategyOverrides.

    ReferenceStyleProfile  +  StyleAdjustments (Customize sliders)  +  ProjectContent  +  AdaptationBaseline  ->  abstract strategy parameters

The result is never a copy of the reference: a target shot length, a motion level, a text budget, caption length / style hints, music and SFX levels. The
Phase 4 / Phase 5 engines read them through ``app.editing.effective`` and still decide everything from the user's own narration and visuals.

Priority (highest first), spec sections 30 / 31:   user lock / user setting  >  content requirement  >  customize targets  >  reference style  >  AI default

* **User lock / user setting.** With "keep my settings" on (the default) a parameter whose AI Edit / caption / audio setting the user changed on purpose is never
  produced; locked and manually edited timeline objects are never touched by the engines at all. A Customize slider target the user typed in is used as set.
* **Content requirement.** The style may only move the project *toward* the reference as far as the content allows: shots never get shorter than a share of
  the narration's sentences or the reading time documents need, motion is limited on document-heavy content, captions never get shorter than a readable
  duration, and the readability floors of the engines stay in force. Every limit that bit is reported in ``notes``.
* **Strength.** 25 / 50 / 75 / 100 % is how far the project moves from where it is now toward the (content-limited) reference value, per parameter
  (geometrically for durations). Even at 100 % the two layers above win.
* **Confidence.** A dimension whose detector is unavailable is skipped; one with low confidence is applied at half strength and flagged; nothing uncertain is
  presented as a fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.captions.styles import PRESETS as CAPTION_PRESETS
from app.editing.effective import protected_parameters
from app.editing.overrides import PARAMETERS, EditingStrategyOverrides
from app.reference.application import AdaptationBaseline, AdaptationResult, ProjectContent, ReferenceSettings, StyleAdjustments
from app.reference.style_model import (
    CAPTION_RATE_POINTS, DIMENSION_LABELS, DIMENSIONS, PACING_POINTS, SFX_POINTS, TEXT_POINTS, ReferenceStyleProfile, StyleScores, StyleSimilarityScore, clamp, scale,
    similarity, unscale,
)

MIN_CONFIDENCE = 0.30  # below this a detector's reading is not applied at all
SOFT_CONFIDENCE = 0.50  # below this it is applied at half strength and flagged
STYLE_CONFIDENCE = 0.50  # caption style / position need a reasonably sure caption detection
MAX_SHOT_SECONDS = 20.0
HARD_MIN_SHOT = 1.0  # no shot is ever planned shorter than this, whatever the reference does
READABLE_MIN_SHOT = 1.4  # the average shot of an edit stays above this
SENTENCE_SHARE = 0.45  # an average shot is at least this share of a narration sentence (cutting more than twice per sentence does not follow the speech)
EVIDENCE_MIN_SHOT = 4.0  # where documents / data dominate, shots need reading time
MOTION_EVIDENCE_PENALTY = 0.6
TRANSITION_CAP = 0.85
TEXT_BUDGET_FACTOR = 2.0  # text_density 1.0 = twice the preset's text budget
CAPTION_WORDS_RANGE = (3, 14)
CAPTION_FILL = 0.6  # a caption holds about this share of max_words on average
CAPTION_COVERAGE_NOMINAL = 0.85  # captions follow the narration: share of the time they are on screen
MUSIC_REFERENCE_LEVEL = 0.18
MUSIC_LEVEL_RANGE = (0.05, 0.40)
SFX_CAP = 10.0
HOOK_MIN_RELATIVE = 1.25  # the opening is paced at least this much faster than the rest to be copied as a tendency
DENSITY_DEADZONE = 6.0
DENSITY_MAX_SHIFT = 25.0
DENSITY_WEIGHTS = {"pacing": 0.50, "text_density": 0.20, "motion_intensity": 0.20, "transition_frequency": 0.10}  # how the density score is made up of the dimension scores
DIMENSION_PARAMS: dict[str, tuple[str, ...]] = {
    "pacing": ("target_shot_duration", "min_shot_duration", "max_shot_duration", "hook_seconds", "hook_shot_factor"),
    "motion_intensity": ("motion_intensity",),
    "transition_frequency": ("transition_frequency",),
    "text_density": ("text_density",),
    "caption_density": ("caption_density", "caption_max_words", "caption_style", "caption_position", "keyword_emphasis_rate"),
    "music_presence": ("music_level", "ducking_strength", "pause_usage"),
    "sfx_frequency": ("sfx_per_minute",),
    "visual_density": (),
}
MODE_TEXT = {"FULL": "Full", "BALANCED": "Balanced", "CUSTOM": "Custom"}
REFERENCE_CAPTION_CLASS = {"Minimal": "minimal", "Bold": "bold", "News": "news", "Documentary": "documentary", "Social": "bold", "High-Impact": "bold", "Subtitle-focused": "clean"}


# ---------------------------------------------------------------------------------------------- small maths
def lin(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def geo(a: float, b: float, s: float) -> float:
    return math.exp(lin(math.log(max(a, 1e-6)), math.log(max(b, 1e-6)), s))


def rnd(v: float, step: float) -> float:
    return round(round(v / step) * step, 4)


@dataclass
class _Limits:
    """What the content demands, in the units the parameters use."""

    shot_factor: float  # final average shot / the pre-adjustment target shot (narration speed, visual complexity, information density, evidence)
    shot_floor: float  # seconds: the average shot never goes below this
    shot_floor_reason: str
    motion_cap: float
    words_floor: int  # captions never hold fewer words than this (readable duration)
    wps: float


def content_limits(c: ProjectContent) -> _Limits:
    wps = c.median_words_per_second if c.median_words_per_second > 0 else 2.5
    narration = clamp(2.5 / max(0.8, wps), 0.75, 1.3)
    complexity = c.evidence_scene_share * 0.75 + (1 - c.evidence_scene_share) * (c.still_image_share * 0.25 + (1 - c.still_image_share) * 0.5)
    factor = narration * (1 + 0.45 * complexity) * (1 + 0.25 * clamp(c.average_information_density)) * (1 + 0.15 * c.evidence_scene_share)
    floor, why = READABLE_MIN_SHOT, "a readable minimum"
    if c.median_sentence_seconds > 0 and SENTENCE_SHARE * c.median_sentence_seconds > floor:
        floor, why = SENTENCE_SHARE * c.median_sentence_seconds, f"your narration's sentences ({c.median_sentence_seconds:.1f} s)"
    if c.evidence_scene_share > 0:
        ev = lin(floor, max(floor, EVIDENCE_MIN_SHOT), c.evidence_scene_share)
        if ev > floor:
            floor, why = ev, "the reading time your documents and data need"
    return _Limits(factor, floor, why, clamp(1.0 - MOTION_EVIDENCE_PENALTY * c.evidence_scene_share, 0.4, 1.0), max(3, int(math.ceil(1.3 * wps))), wps)


def _pacing_score(shot_seconds: float) -> float:
    return scale(60.0 / max(shot_seconds, 0.2), PACING_POINTS)


def _text_per_minute(density: float, b: AdaptationBaseline) -> float:
    return b.text_per_minute * density / 0.5


def _music_presence(level: float) -> float:
    return clamp((level / MUSIC_REFERENCE_LEVEL - 0.4) / 1.2)


def _caption_rate_score(words: float, lim: _Limits) -> float:
    return scale(60.0 * lim.wps / max(1.0, CAPTION_FILL * words), CAPTION_RATE_POINTS)


def predicted_scores(b: AdaptationBaseline, c: ProjectContent) -> dict[str, float]:
    """Where the current settings put the project on the eight dimensions (a tendency model: it is what the engines are *set up* to do, not a measurement)."""
    lim = content_limits(c)
    p = _pacing_score(b.base_shot_duration * lim.shot_factor)
    tx = scale(_text_per_minute(b.text_density, b), TEXT_POINTS)
    m, tr = clamp(b.motion_intensity) * 100.0, clamp(b.transition_frequency) * 100.0
    out = {"pacing": p, "motion_intensity": m, "transition_frequency": tr, "text_density": tx,
           "caption_density": 100.0 * (0.6 * CAPTION_COVERAGE_NOMINAL + 0.4 * _caption_rate_score(b.caption_max_words, lim) / 100.0),
           "music_presence": 100.0 * _music_presence(b.music_level) if b.music_level > 0 else 0.0, "sfx_frequency": scale(b.sfx_per_minute, SFX_POINTS)}
    out["visual_density"] = _density(out)
    return out


def _density(s: dict[str, float]) -> float:
    return sum(w * s[d] for d, w in DENSITY_WEIGHTS.items())


# ---------------------------------------------------------------------------------------------- the result of a simulation
@dataclass
class ProjectedStyle:
    """How the project would read on the eight dimensions after the style is applied: a tendency preview, not a render."""

    scores: dict[str, float] = field(default_factory=dict)
    current: dict[str, float] = field(default_factory=dict)
    similarity_before: float = 0.0
    similarity_after: float = 0.0
    changes: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- the adapter
class ReferenceStyleAdapter:
    def adapt(self, profile: ReferenceStyleProfile, adjustments: StyleAdjustments, settings: ReferenceSettings, content: ProjectContent,
              baseline: AdaptationBaseline) -> AdaptationResult:
        run = _Run(profile, adjustments, settings, content, baseline)
        return run.execute()


class _Run:
    def __init__(self, profile: ReferenceStyleProfile, adj: StyleAdjustments, settings: ReferenceSettings, content: ProjectContent, base: AdaptationBaseline) -> None:
        self.prof, self.adj, self.cfg, self.content, self.base = profile, adj, settings, content, base
        self.lim = content_limits(content)
        self.strength = clamp(float(settings.style_strength))
        self.params: dict[str, object] = {}
        self.targets: dict[str, float] = {}
        self.skipped: dict[str, str] = {}
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self.cur = predicted_scores(base, content)
        self.after = dict(self.cur)  # predicted dimension scores once the chosen parameters are in force
        self.eff_strength: dict[str, float] = {}
        self.touched: set[str] = set()  # dimensions whose parameters this run produced (chosen directly, or moved by the visual density)

    # ------------------------------------------------------------------ driver
    def execute(self) -> AdaptationResult:
        applied: list[str] = []
        mode = self.cfg.application_mode
        allowed = set(self.cfg.dimensions())
        for d in DIMENSIONS:
            if d not in allowed:
                self.skipped[d] = f"not part of the {MODE_TEXT.get(mode, mode)} mode"
                continue
            if not self.prof.is_available(d):
                self.skipped[d] = "could not be measured in this reference"
                continue
            conf = self.prof.dimension_confidence(d)
            if conf < MIN_CONFIDENCE:
                self.skipped[d] = f"the reading is too uncertain to apply ({conf:.0%} confidence)"
                continue
            user_target = self.adj.get(d)
            st = 1.0 if user_target is not None else self.strength  # a target the user typed in is used as set; strength scales only the reference's own values
            if conf < SOFT_CONFIDENCE and user_target is None:
                st *= 0.5
                self.warnings.append(f"{DIMENSION_LABELS[d]}: low confidence ({conf:.0%}) - applied at half strength.")
            self.eff_strength[d] = st
            if user_target is not None:
                self.notes.append(f"{DIMENSION_LABELS[d]}: your target ({user_target:.0f}) is used instead of the reference value ({self.prof.score(d):.0f}).")
            applied.append(d)
        for d in applied:
            if d == "visual_density":
                continue
            target = self.adj.get(d)
            handler = getattr(self, "_" + d)
            self.touched.add(d)
            handler(self.prof.score(d) if target is None else target, self.eff_strength[d], target is not None)
        if "visual_density" in applied:
            self._visual_density(applied)
        self._user_settings_win()
        ov = EditingStrategyOverrides(strength=self.strength, source="reference")
        for k, v in self.params.items():
            if k in PARAMETERS:
                setattr(ov, k, v)
        ov.applied_fields = [k for k in PARAMETERS if getattr(ov, k) is not None]
        ov.notes = list(self.notes[:12])
        self.targets = {d: round(v, 1) for d, v in self.targets.items() if d not in self.skipped}
        if applied and not ov.is_empty:
            self.notes.append("Locked and manually edited timeline objects are never changed; the next generation uses these values only for AI-created elements.")
        if self.strength < 1.0:
            self.notes.append(f"Style strength {self.strength:.0%}: the project moves {self.strength:.0%} of the way from its current settings toward the reference.")
        return AdaptationResult(ov, self.targets, self.skipped, self.notes, self.warnings)

    # ------------------------------------------------------------------ pacing -> shot lengths (+ the opening)
    def _pacing(self, score: float, st: float, adjusted: bool) -> None:
        b, lim, f = self.base, self.lim, self.prof.features
        if not adjusted and f.average_shot_duration > 0:
            ref = (f.average_shot_duration + (f.median_shot_duration or f.average_shot_duration)) / 2.0
        else:
            ref = 60.0 / max(unscale(score, PACING_POINTS), 0.5)
        ref = min(ref, MAX_SHOT_SECONDS)
        cur = b.base_shot_duration * lim.shot_factor
        want = geo(cur, ref, st)
        final = want
        if want < cur and want < lim.shot_floor:
            final = min(cur, lim.shot_floor)
            self.notes.append(f"Pacing: shots limited to about {final:.1f} s (the reference's {ref:.1f} s would be shorter than {lim.shot_floor_reason} allows).")
        target = rnd(final / lim.shot_factor, 0.05)
        self.params["target_shot_duration"] = max(0.5, target)
        self.notes.append(f"Pacing: target shot length {target:.2f} s before content adjustments (about {final:.1f} s in the edit; reference {ref:.1f} s).")
        shortest = final
        hook = self._hook(final, st) if not adjusted else None
        if hook:
            self.params["hook_seconds"], self.params["hook_shot_factor"] = hook
            shortest = final * hook[1]
        if shortest * 0.8 < b.min_shot_duration:
            self.params["min_shot_duration"] = rnd(max(HARD_MIN_SHOT, shortest * 0.8), 0.05)
        ceiling = b.max_shot_duration
        if final < cur:  # a faster edit needs the "one visual holds the whole scene" ceiling lowered, or a short scene is never cut
            ceiling = rnd(min(b.max_shot_duration, max(2.4 * final, b.min_shot_duration + 0.5)), 0.1)
        elif final * 1.25 > b.max_shot_duration:
            ceiling = rnd(min(16.0, final * 1.6), 0.1)
        if abs(ceiling - b.max_shot_duration) > 0.05:
            self.params["max_shot_duration"] = ceiling
        self.targets["pacing"] = _pacing_score(final)
        self.after["pacing"] = self.targets["pacing"]

    def _hook(self, final: float, st: float) -> tuple[float, float] | None:
        h = self.prof.hook
        conf = self.prof.confidence.get("structure_detection", 0.0)
        if not h.windows or conf < MIN_CONFIDENCE:
            return None
        fast = [w for w in h.windows if w.relative_pacing >= HOOK_MIN_RELATIVE and w.shot_rate > 0]
        if not fast:
            return None
        # the longest window that is still clearly faster than the rest, counted from the first window
        chosen = fast[0]
        for w in h.windows:
            if w.relative_pacing >= HOOK_MIN_RELATIVE and w.shot_rate > 0:
                chosen = w
            else:
                break
        ref_factor = clamp(1.0 / chosen.relative_pacing, 0.45, 1.0)
        factor = 1.0 - (1.0 - ref_factor) * st
        factor = max(factor, HARD_MIN_SHOT / max(final, HARD_MIN_SHOT))  # opening shots stay watchable
        if factor > 0.92:
            return None
        self.notes.append(f"Opening: the first {chosen.seconds} s are cut about {1 / factor:.1f}x faster than the rest.")
        return float(min(chosen.seconds, 15)), rnd(factor, 0.05)

    # ------------------------------------------------------------------ motion
    def _motion_intensity(self, score: float, st: float, adjusted: bool) -> None:
        b, lim = self.base, self.lim
        m0, ref = clamp(b.motion_intensity), clamp(score / 100.0)
        m = lin(m0, ref, st)
        if m > m0 and m > lim.motion_cap:
            m = max(m0, lim.motion_cap)
            self.notes.append(f"Motion: limited to {m:.0%} - documents and data stay still enough to read.")
        self.params["motion_intensity"] = rnd(m, 0.01)
        self.targets["motion_intensity"] = self.after["motion_intensity"] = m * 100.0

    # ------------------------------------------------------------------ transitions
    def _transition_frequency(self, score: float, st: float, adjusted: bool) -> None:
        t0, ref = clamp(self.base.transition_frequency), clamp(score / 100.0)
        t = lin(t0, ref, st)
        if t > t0 and t > TRANSITION_CAP:
            t = max(t0, TRANSITION_CAP)
            self.notes.append("Transitions: limited so that not every cut becomes a transition.")
        self.params["transition_frequency"] = rnd(t, 0.01)
        self.targets["transition_frequency"] = self.after["transition_frequency"] = t * 100.0

    # ------------------------------------------------------------------ text graphics
    def _text_density(self, score: float, st: float, adjusted: bool) -> None:
        b = self.base
        if b.text_density <= 0 or b.text_per_minute <= 0:
            self.skipped["text_density"] = "text graphics are switched off in AI Edit"
            return
        d0 = clamp(b.text_density)
        d_ref = clamp(0.5 * unscale(score, TEXT_POINTS) / b.text_per_minute)
        d = lin(d0, d_ref, st)
        self.params["text_density"] = rnd(d, 0.01)
        self.targets["text_density"] = self.after["text_density"] = scale(_text_per_minute(d, b), TEXT_POINTS)
        self.notes.append("Text graphics: the amount follows the reference's rhythm, but every word still comes from your narration or script.")

    # ------------------------------------------------------------------ captions
    def _caption_density(self, score: float, st: float, adjusted: bool) -> None:
        cap, b, lim = self.prof.caption_style, self.base, self.lim
        if not cap.caption_present and not adjusted:
            self.skipped["caption_density"] = "no captions were detected in the reference"
            return
        w0 = float(b.caption_max_words)
        if adjusted:
            # the slider is on the combined (coverage + count) scale: move the reference's caption length by the same relative amount
            ratio = clamp(score / max(self.prof.score("caption_density"), 5.0), 0.25, 2.5)
            words_ref = (cap.average_words_per_caption or 0.6 * w0) / ratio
        else:
            words_ref = cap.average_words_per_caption
        ref_max = clamp(round(words_ref / CAPTION_FILL), *CAPTION_WORDS_RANGE) if words_ref > 0 else w0
        final = lin(w0, ref_max, st)
        floor = float(lim.words_floor)
        if final < w0 and final < floor:
            final = min(w0, floor)
            self.notes.append(f"Captions: at least {int(floor)} words per caption so that each stays on screen long enough to read.")
        final = float(clamp(round(final), *CAPTION_WORDS_RANGE))
        self.params["caption_max_words"] = int(final)
        dens_ref = self.prof.score("caption_density") if not adjusted else score
        frac = 1.0 if abs(math.log(max(ref_max, 1) / max(w0, 1))) < 1e-6 else clamp(math.log(max(final, 1) / max(w0, 1)) / math.log(max(ref_max, 1) / max(w0, 1)))
        density = rnd(clamp(lin(self.cur["caption_density"], dens_ref, frac) / 100.0), 0.01)
        self.params["caption_density"] = density
        self.targets["caption_density"] = self.after["caption_density"] = density * 100.0
        conf = self.prof.confidence.get("caption_detection", 0.0)
        if cap.caption_present and conf >= STYLE_CONFIDENCE and not adjusted:
            style = REFERENCE_CAPTION_CLASS.get(cap.style_class)
            if style in CAPTION_PRESETS and st >= 0.5:
                self.params["caption_style"] = style
                self.notes.append(f"Captions: a {cap.style_class.lower()} look ('{style}' style) - your own caption text and wording stay as they are.")
            if cap.caption_position in ("bottom", "center", "top") and st >= 0.5:
                self.params["caption_position"] = cap.caption_position
        if cap.caption_present and not adjusted:
            r = clamp(cap.caption_emphasis_rate / 0.6)
            self.params["keyword_emphasis_rate"] = rnd(lin(clamp(b.keyword_emphasis_rate), 0.15 + 0.85 * r, st), 0.01)

    # ------------------------------------------------------------------ audio: music, ducking, pauses
    def _music_presence(self, score: float, st: float, adjusted: bool) -> None:
        b, au = self.base, self.prof.audio
        if b.music_level <= 0:
            self.skipped["music_presence"] = "music is switched off in the audio settings"
            return
        ref_level = clamp(MUSIC_REFERENCE_LEVEL * (0.4 + 1.2 * clamp(score / 100.0)), *MUSIC_LEVEL_RANGE)
        level = lin(b.music_level, ref_level, st)
        self.params["music_level"] = rnd(clamp(level, *MUSIC_LEVEL_RANGE), 0.005)
        self.targets["music_presence"] = self.after["music_presence"] = 100.0 * _music_presence(level)
        if not adjusted and au.has_audio and au.music_presence >= 0.2:
            duck = rnd(lin(clamp(b.ducking_strength), clamp(au.music_ducking_strength), st), 0.01)
            pause_ref = clamp(0.2 + 0.8 * min(1.0, au.long_pause_frequency / 8.0))
            pause = rnd(lin(clamp(b.pause_usage), pause_ref, st), 0.01)
            self.params["ducking_strength"], self.params["pause_usage"] = duck, pause
            self.notes.append(f"Music: level {level:.0%}, ducking {duck:.0%} under speech, pause usage {pause:.0%}.")

    def _sfx_frequency(self, score: float, st: float, adjusted: bool) -> None:
        b = self.base
        if b.sfx_per_minute <= 0:
            self.skipped["sfx_frequency"] = "sound effects are switched off in the audio settings"
            return
        ref = min(SFX_CAP, unscale(score, SFX_POINTS))
        v = min(SFX_CAP, lin(b.sfx_per_minute, ref, st))
        self.params["sfx_per_minute"] = rnd(v, 0.5)
        self.targets["sfx_frequency"] = self.after["sfx_frequency"] = scale(v, SFX_POINTS)

    # ------------------------------------------------------------------ visual density: made of the other dimensions
    def _visual_density(self, applied: list[str]) -> None:
        score = self.adj.get("visual_density")
        st = self.eff_strength["visual_density"]
        t_ref = self.prof.score("visual_density") if score is None else score
        cur_d = _density(self.cur)
        want = lin(cur_d, t_ref, st)
        now = _density(self.after)
        delta = want - now
        free = [d for d in DENSITY_WEIGHTS if d not in applied and self._free(d)]
        if abs(delta) >= DENSITY_DEADZONE and free:
            wsum = sum(DENSITY_WEIGHTS[d] for d in free)
            for d in free:
                shift = clamp(delta / wsum, -DENSITY_MAX_SHIFT, DENSITY_MAX_SHIFT)
                self.skipped.pop(d, None)  # it was "not part of this mode"; the density moves it after all
                self.touched.add(d)
                self._shift(d, self.after[d] + shift)
            self.notes.append("Visual density: " + ", ".join(DIMENSION_LABELS[d].lower() for d in free) + " were adjusted to bring the overall density to " + f"{_density(self.after):.0f}.")
        elif abs(delta) >= DENSITY_DEADZONE:
            self.notes.append("Visual density follows from the pacing, motion, text and transition settings; no separate setting exists for it.")
        ref_d = self.prof.score("visual_density")
        ref_parts = {d: (self.adj.get(d) if self.adj.get(d) is not None else self.prof.score(d)) for d in DENSITY_WEIGHTS}
        self.targets["visual_density"] = clamp(ref_d + _density(self.after) - _density(ref_parts), 0.0, 100.0)  # the density formula is only a surrogate: report it relative to the reference

    def _free(self, d: str) -> bool:
        """A dimension the density can move on its own: its parameter exists and is not switched off."""
        if d == "text_density":
            return self.base.text_density > 0
        return True

    def _shift(self, d: str, score: float) -> None:
        score = clamp(score, 0.0, 100.0)
        if d == "pacing":
            self._pacing(score, 1.0, True)
        elif d == "motion_intensity":
            self._motion_intensity(score, 1.0, True)
        elif d == "transition_frequency":
            self._transition_frequency(score, 1.0, True)
        elif d == "text_density":
            self._text_density(score, 1.0, True)

    # ------------------------------------------------------------------ user settings win
    def _user_settings_win(self) -> None:
        protected = protected_parameters(self.base.user_set, self.cfg.preserve_user_edits)
        dropped = [k for k in list(self.params) if k in protected]
        for k in dropped:
            del self.params[k]
        if dropped:
            self.notes.append("Kept your own settings for: " + ", ".join(sorted(dropped)) + ".")
        for d, names in DIMENSION_PARAMS.items():
            if names and d in self.touched and d not in self.skipped and not any(n in self.params for n in names):
                self.skipped[d] = "you set this yourself; the reference does not override it" if any(n in dropped for n in names) else "nothing to change"
                self.targets.pop(d, None)


# ---------------------------------------------------------------------------------------------- comparison and simulation
def similarity_between(a: ReferenceStyleProfile, b: ReferenceStyleProfile) -> StyleSimilarityScore:
    """Editing-feature similarity of two profiles (never a copyright or content measure). Dimensions either side could not measure take no part."""
    return similarity(a.scores, b.scores, sorted(set(a.unavailable) | set(b.unavailable)))


def simulate(profile: ReferenceStyleProfile, adjustments: StyleAdjustments, settings: ReferenceSettings, content: ProjectContent, baseline: AdaptationBaseline,
             mine: ReferenceStyleProfile) -> ProjectedStyle:
    """The project's eight scores now and after the style would be applied. A measured timeline keeps its own offset from the settings-based prediction."""
    res = ReferenceStyleAdapter().adapt(profile, adjustments, settings, content, baseline)
    cur_pred = predicted_scores(baseline, content)
    measured = {d: mine.score(d) for d in DIMENSIONS if mine.is_available(d)}
    now = {d: measured.get(d, cur_pred[d]) for d in DIMENSIONS}
    after = dict(now)
    for d, eff in res.effective_targets.items():  # what the style sets up (after the content limits): that is where the regenerated edit is aimed
        after[d] = clamp(eff, 0.0, 100.0)
    skip = sorted(set(profile.unavailable))
    before_sim = similarity(profile.scores, StyleScores(**{d: now[d] for d in DIMENSIONS}), skip)
    after_sim = similarity(profile.scores, StyleScores(**{d: after[d] for d in DIMENSIONS}), skip)
    changes = [f"{DIMENSION_LABELS[d]}: {now[d]:.0f} -> {after[d]:.0f}" + (f"  (reference {profile.score(d):.0f})" if profile.is_available(d) else "") for d in DIMENSIONS if abs(after[d] - now[d]) >= 1.0]
    return ProjectedStyle({d: round(v, 1) for d, v in after.items()}, {d: round(v, 1) for d, v in now.items()}, before_sim.overall, after_sim.overall, changes,
                          list(res.effective_targets), list(res.notes))
