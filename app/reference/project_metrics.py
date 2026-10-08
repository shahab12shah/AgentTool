"""Measure the user's own project with the same abstract style dimensions as a reference.

    project timeline / captions / audio settings  ->  StyleFeatures  ->  StyleScores  ->  ReferenceStyleProfile   (the same pipeline a reference goes through)

so that REFERENCE vs CURRENT PROJECT can be compared row by row, the style similarity is computed on one scale, and the style adapter knows where the project
stands. Nothing here reads a reference, changes a project or touches the timeline: it only reads.

* cut rhythm: the starts of the visual clips (video / image tracks) are the shot boundaries; shot lengths are the clip lengths;
* text overlays / graphics: the text and graphic clips; non-cut transitions: ``Clip.transition`` on the visual clips;
* captions: the caption clips (coverage, count, words per caption, highlighted words, position);
* motion: the keyframes on the visual clips - every scale / position move is a zoom / pan event, rated on the same intensity curve the reference analyzer uses;
* audio: music coverage and ducking from the music clips and the audio settings, SFX from the SFX clips, pauses from the voice analysis.

``project_content`` and ``adaptation_baseline`` are the two inputs the adapter needs beyond the profile: what the user's content demands (reading time,
narration speed, sentence length) and the settings the style is blended *from*.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import TYPE_CHECKING

from app.editing.effective import PAUSE_LEVEL_GAIN, ducking_strength
from app.reference.application import AdaptationBaseline, ProjectContent
from app.reference.motion_analyzer import PAN_REF_SPEED, PAN_SPEED_FLOOR, ZOOM_MIN_CHANGE, ZOOM_RATE_FLOOR, ZOOM_REF_RATE, PAN_MIN_EXTENT
from app.reference.style_model import (
    CaptionStats, MotionStats, ReferenceStyleProfile, StyleFeatures, TextStats, ZoomStats, build_profile_from_features, clamp, motion_class, music_class, pacing_class, sfx_class,
)
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.track import TrackKind

if TYPE_CHECKING:  # pragma: no cover
    from app.project.project import Project

VISUAL_TRACKS = (TrackKind.VIDEO, TrackKind.IMAGE)
CAPTION_CLASS = {"professional": "Subtitle-focused", "clean": "Subtitle-focused", "bold": "Bold", "news": "News", "documentary": "Documentary", "minimal": "Minimal"}
SHOT_MERGE = 0.05  # clips starting within this many seconds of each other start one shot
HOOK_SECONDS = 10.0
WORDS_PER_CAPTION_FILL = 0.6  # a caption holds about this share of ``max_words`` on average


# ---------------------------------------------------------------------------------------------- helpers
def _clips(project: "Project", kinds: tuple[str, ...], tracks: tuple[TrackKind, ...] | None = None) -> list[Clip]:
    out = [c for t in project.timeline.tracks if not t.hidden and (tracks is None or t.kind in tracks) for c in t.clips if c.kind in kinds]
    return sorted(out, key=lambda c: (c.timeline_start, c.id))


def _union_length(spans: list[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for a, b in sorted(spans):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def project_duration(project: "Project") -> float:
    """The length of the edit: the last end of anything on the timeline, or the voice-over / scenes when the timeline is still empty."""
    ends = [c.timeline_end for t in project.timeline.tracks for c in t.clips]
    vo = project.assets.get(project.voice_over.asset_id) if project.voice_over.asset_id else None
    ends += [float(vo.duration or 0.0)] if vo is not None else []
    ends += [s.end for s in project.scenes]
    return float(max(ends, default=0.0))


def _zoom_pan_events(clip: Clip, canvas_w: float) -> list[tuple[str, float, float]]:
    """(kind, seconds, intensity 0..1) for every zoom / pan move in a clip's keyframes: one event per run of keyframe intervals that keeps moving the same way."""
    events: list[tuple[str, float, float]] = []
    by_prop: dict[str, list] = {}
    for k in clip.keyframes:
        by_prop.setdefault(k.property, []).append(k)
    scale_kf = sorted(by_prop.get("scale", []), key=lambda k: k.time)
    run: list[float] = []  # [seconds, log change] of the current zoom run
    for a, b in zip(scale_kf, scale_kf[1:]):
        dt = b.time - a.time
        if dt <= 1e-6 or a.value <= 0 or b.value <= 0:
            continue
        lc = math.log(b.value / a.value)
        if abs(lc) < 1e-6:
            if run:
                events.append(_zoom_event(run))
                run = []
            continue
        if run and (run[1] > 0) == (lc > 0):
            run[0] += dt
            run[1] += lc
        else:
            if run:
                events.append(_zoom_event(run))
            run = [dt, lc]
    if run:
        events.append(_zoom_event(run))
    events = [e for e in events if e[0] != "skip"]
    px, py = sorted(by_prop.get("position_x", []), key=lambda k: k.time), sorted(by_prop.get("position_y", []), key=lambda k: k.time)
    if len(px) >= 2 or len(py) >= 2:
        # a pan: the content travels across the frame (the extent in pixels over the whole clip; zoom-anchored drift is small and falls below the minimum)
        def span(kfs):
            return (max(k.value for k in kfs) - min(k.value for k in kfs), kfs[-1].time - kfs[0].time) if len(kfs) >= 2 else (0.0, 0.0)

        (ex, tx), (ey, ty) = span(px), span(py)
        extent = math.hypot(ex, ey) / max(canvas_w, 1.0)
        dt = max(tx, ty, 1e-6)
        has_zoom = any(e[0] == "zoom" for e in events)
        if extent >= PAN_MIN_EXTENT * (2.0 if has_zoom else 1.0) and dt > 0.3:
            speed = extent / dt
            events.append(("pan", dt, clamp(math.sqrt(max(0.0, speed - PAN_SPEED_FLOOR) / PAN_REF_SPEED))))
    return events


def _zoom_event(run: list[float]) -> tuple[str, float, float]:
    dt, lc = run
    if abs(lc) < ZOOM_MIN_CHANGE:
        return ("skip", 0.0, 0.0)
    rate = abs(lc) / dt
    return ("zoom", dt, clamp(math.sqrt(max(0.0, rate - ZOOM_RATE_FLOOR) / ZOOM_REF_RATE)))


# ---------------------------------------------------------------------------------------------- features
class _Measure:
    """Everything measured from the timeline in one pass (also used to fill the descriptive parts of the profile)."""

    def __init__(self, project: "Project") -> None:
        p = self.p = project
        self.duration = project_duration(p)
        self.minutes = self.duration / 60.0 if self.duration > 0 else 0.0
        self.visual = _clips(p, (KIND_MEDIA,), VISUAL_TRACKS)
        self.texts = _clips(p, (KIND_TEXT,))
        self.graphics = _clips(p, (KIND_GRAPHIC,))
        self.captions = _clips(p, (KIND_CAPTION,))
        self.audio = [c for t in p.timeline.tracks if t.kind is TrackKind.AUDIO and not t.muted for c in t.clips if c.kind == KIND_MEDIA]
        self.canvas_w = float(p.settings.width or 1920)
        self._shots()

    def per_min(self, n: float) -> float:
        return n / self.minutes if self.minutes > 0 else 0.0

    # -- shots
    def _shots(self) -> None:
        starts: list[tuple[float, float]] = []  # (start, length of the longest clip starting there)
        for c in self.visual:
            if starts and c.timeline_start - starts[-1][0] <= SHOT_MERGE:
                starts[-1] = (starts[-1][0], max(starts[-1][1], c.duration))
            else:
                starts.append((c.timeline_start, c.duration))
        self.cut_times = [s for s, _ in starts[1:]]
        durs = []
        for i, (s, ln) in enumerate(starts):
            nxt = starts[i + 1][0] if i + 1 < len(starts) else s + ln
            durs.append(max(0.0, min(nxt - s, ln)) if i + 1 < len(starts) else ln)
        self.shot_durations = durs

    # -- motion
    def motion(self) -> tuple[int, int, int, float, float]:
        """(zoom events, pan events, clips that move, intensity-weighted seconds, visual seconds)"""
        zooms = pans = moving = 0
        weighted = total = 0.0
        for c in self.visual:
            total += c.duration
            ev = _zoom_pan_events(c, self.canvas_w)
            if not ev:
                continue
            moving += 1
            zooms += sum(1 for e in ev if e[0] == "zoom")
            pans += sum(1 for e in ev if e[0] == "pan")
            weighted += sum(e[2] * min(e[1], c.duration) for e in ev)
        return zooms, pans, moving, weighted, total

    # -- audio
    def role(self, name: str) -> list[Clip]:
        return [c for c in self.audio if str(c.audio.get("role", "")).upper() == name]

    def music_dynamics(self) -> float:
        """Intensity changes of the bed that are not ducking: volume keyframes of clips whose ducking is off, and clips with different base volumes."""
        music = self.role("MUSIC")
        if len(music) < 2:
            return 0.0
        vols = [float(c.audio.get("volume", 1.0)) for c in music]
        hi, lo = max(vols), min(vols)
        return clamp((hi - lo) / hi) if hi > 0 else 0.0


def project_style_features(project: "Project") -> StyleFeatures:
    m = _Measure(project)
    f = StyleFeatures(duration=m.duration)
    if m.duration <= 0:
        return f
    durs = m.shot_durations
    if durs:
        f.average_shot_duration = float(sum(durs) / len(durs))
        f.median_shot_duration = float(median(durs))
    f.cuts_per_minute = m.per_min(len(m.cut_times))
    f.text_events_per_minute = m.per_min(len(m.texts))
    f.graphic_events_per_minute = m.per_min(len(m.graphics))
    f.overlay_events_per_minute = f.text_events_per_minute + f.graphic_events_per_minute
    f.major_changes_per_minute = f.cuts_per_minute + f.graphic_events_per_minute
    zooms, pans, _moving, weighted, vis = m.motion()
    f.zoom_events_per_minute = m.per_min(zooms)
    f.motion_events_per_minute = m.per_min(zooms + pans)
    f.average_motion_intensity = clamp(weighted / vis) if vis > 0 else 0.0
    caps = m.captions
    f.caption_coverage = clamp(_union_length([(c.timeline_start, c.timeline_end) for c in caps]) / m.duration)
    f.captions_per_minute = m.per_min(len(caps))
    words = [len((c.text or {}).get("words") or []) or len(str((c.text or {}).get("text", "")).split()) for c in caps]
    f.average_words_per_caption = float(sum(words) / len(words)) if words else 0.0
    f.caption_emphasis_rate = sum(1 for c in caps if (c.text or {}).get("emphasis")) / len(caps) if caps else 0.0
    trans = [c for c in m.visual if c.transition and str(c.transition.get("type", "CUT")).upper() != "CUT"]
    f.transition_events_per_minute = m.per_min(len(trans))
    f.non_cut_share = len(trans) / max(1, len(m.cut_times)) if m.cut_times else 0.0
    f.non_cut_share = clamp(f.non_cut_share)
    # audio
    music, sfx = m.role("MUSIC"), m.role("SFX")
    a = project.audio_settings
    f.music_presence = clamp(_union_length([(c.timeline_start, c.timeline_end) for c in music]) / m.duration)
    f.music_ducking_strength = ducking_strength(a.music_level, a.important_level) if (a.auto_ducking and music) else 0.0
    f.music_dynamics = m.music_dynamics()
    f.sfx_per_minute = m.per_min(len(sfx))
    an = project.audio_analysis
    if an is not None:
        silences = [(s[0], s[1]) for s in an.silence_regions if len(s) >= 2]
        f.silence_percentage = 100.0 * clamp(sum(b - s for s, b in silences) / m.duration)
        pauses = [b - s for s, b in (tuple(x[:2]) for x in an.pauses if len(x) >= 2)]
        f.average_pause_duration = float(sum(pauses) / len(pauses)) if pauses else 0.0
        f.long_pause_frequency = m.per_min(sum(1 for d in pauses if d >= 1.0))
        f.audio_dynamic_range = float(an.dynamic_range_db or an.loudness_range or 0.0)
        f.voice_dominance = clamp(float(an.speech_ratio)) if an.speech_ratio is not None else 0.0
    if not f.voice_dominance:
        voice = [c for c in m.audio if c.slot == "voice" or str(c.audio.get("role", "")).upper() == "VOICE" or c.track_id == "track_a1"]
        f.voice_dominance = clamp(_union_length([(c.timeline_start, c.timeline_end) for c in voice]) / m.duration)
    if len(m.cut_times) >= 3 and m.duration > 2 * HOOK_SECONDS:
        hook = sum(1 for t in m.cut_times if t < HOOK_SECONDS) / (HOOK_SECONDS / 60.0)
        rest = sum(1 for t in m.cut_times if t >= HOOK_SECONDS) / ((m.duration - HOOK_SECONDS) / 60.0)
        f.hook_intensity = clamp(2.0 * (hook - rest) / (hook + rest)) if hook + rest > 0 else 0.0
    return f


# ---------------------------------------------------------------------------------------------- the profile
def project_style_profile(project: "Project") -> ReferenceStyleProfile:
    """The user's project on the reference scale. Dimensions that cannot be measured yet (no visuals on the timeline) are marked unavailable, not scored 0."""
    m = _Measure(project)
    f = project_style_features(project)
    has_visual = bool(m.visual)
    music = m.role("MUSIC")
    prof = build_profile_from_features(f, "", music_behavior=music_class(f.music_presence, f.music_dynamics) if music else "None", sfx_label=sfx_class(f.sfx_per_minute),
                                       has_audio=bool(m.audio), confidence=1.0 if has_visual else 0.0)
    if not has_visual:
        prof.unavailable = ["pacing", "visual_density", "motion_intensity", "transition_frequency"]
        for d in prof.unavailable:
            prof.categories[d] = "No timeline yet"
    # descriptive parts (no reference-only numbers are invented: everything below is counted)
    prof.shot_stats.count = len(m.shot_durations)
    prof.shot_stats.minimum_shot_duration = min(m.shot_durations, default=0.0)
    prof.shot_stats.maximum_shot_duration = max(m.shot_durations, default=0.0)
    prof.shot_stats.cut_frequency_class = pacing_class(f.cuts_per_minute)
    zooms, pans, moving, _w, _v = m.motion()
    mo = MotionStats(f.motion_events_per_minute, f.average_motion_intensity, f.zoom_events_per_minute, m.per_min(pans), 1.0 - moving / len(m.visual) if m.visual else 1.0,
                     0.0, motion_class(prof.scores.motion_intensity), ZoomStats(zooms, f.zoom_events_per_minute))
    prof.motion = mo
    prof.caption_style = _caption_stats(project, m, f)
    prof.text = _text_stats(m, f)
    return prof


def _caption_stats(project: "Project", m: _Measure, f: StyleFeatures) -> CaptionStats:
    caps = m.captions
    cs = project.caption_settings
    from app.captions.styles import PRESETS  # noqa: PLC0415

    style = project.caption_styles.get(cs.style_id) or PRESETS.get(cs.style_id) or PRESETS["professional"]
    positions = Counter(str((c.text or {}).get("position", cs.position)) for c in caps)
    pos = positions.most_common(1)[0][0] if positions else cs.position
    pos = pos if pos in ("bottom", "center", "top") else "bottom"
    lines = [len((c.text or {}).get("lines") or [1]) for c in caps]
    chars = [len(l) for c in caps for l in ((c.text or {}).get("lines") or [])]
    anim = sum(1 for c in caps if (c.animation or {}).get("in")) / len(caps) if caps else 0.0
    traits = []
    if cs.large_text or style.size_rel >= 0.058:
        traits.append("large_text")
    if cs.high_contrast:
        traits.append("high_contrast")
    if f.caption_emphasis_rate >= 0.4:
        traits.append("frequent_highlighting")
    if 0 < f.average_words_per_caption <= 4:
        traits.append("short_caption_segments")
    traits.append(f"{pos}_position" if pos in ("center", "bottom") else "top_position")
    return CaptionStats(bool(caps), f.caption_coverage, f.captions_per_minute, f.average_words_per_caption, float(sum(chars) / len(chars)) if chars else 0.0,
                        float(sum(lines) / len(lines)) if lines else 0.0, pos, f.caption_emphasis_rate, anim, style.size_rel, CAPTION_CLASS.get(style.style_id, "Subtitle-focused"),
                        traits, style.background == "box" or cs.high_contrast)


def _text_stats(m: _Measure, f: StyleFeatures) -> TextStats:
    texts = m.texts
    kinds = Counter(str((c.text or {}).get("style", "")) for c in texts)
    pos = Counter()
    for c in texts:
        y = float(((c.text or {}).get("position") or [0.5, 0.5])[1])
        pos["top" if y < 0.33 else "center" if y < 0.66 else "bottom"] += 1
    n = len(texts)
    return TextStats(f.text_events_per_minute, float(sum(c.duration for c in texts) / n) if n else 0.0, m.per_min(kinds.get("HEADLINE", 0)), m.per_min(kinds.get("NUMBER_CARD", 0)),
                     m.per_min(kinds.get("LOWER_THIRD", 0) + kinds.get("ENTITY_NAME", 0)),
                     float(sum(float((c.text or {}).get("size", 0)) for c in texts) / n / max(1.0, float(m.p.settings.height))) if n else 0.0,
                     {k: round(v / n, 3) for k, v in pos.items()} if n else {}, sum(1 for c in texts if (c.animation or {}).get("in")) / n if n else 0.0, f.graphic_events_per_minute)


# ---------------------------------------------------------------------------------------------- what the user's content needs
def project_content(project: "Project") -> ProjectContent:
    """What the narration and the visuals demand (read from the scene analysis and the transcript): the adapter never lets a style ask for more than this allows."""
    scenes = list(project.scenes)
    c = ProjectContent()
    if not scenes:
        return c
    c.scene_count = len(scenes)
    c.duration = float(max(s.end for s in scenes) - min(s.start for s in scenes))
    c.median_scene_seconds = float(median(s.duration for s in scenes))
    evidence = 0
    nums = 0
    dens, wps = [], []
    tr = project.transcription.transcript
    for s in scenes:
        intent = project.visual_intents.get(s.id)
        if intent is not None and intent.type.value in ("EVIDENCE", "DATA"):
            evidence += 1
        if s.numbers:
            nums += 1
        dens.append(min(1.0, 0.18 * len(s.numbers) + 0.12 * len(s.claims) + 0.05 * len(s.entities)))
        if tr is not None:
            words = tr.words_between(s.start, s.end)
            if len(words) >= 3:
                pauses = sum(b.start - a.end for a, b in zip(words, words[1:]) if b.start - a.end > 0.3)
                speech = max(0.4, (words[-1].end - words[0].start) - pauses)
                wps.append(len(words) / speech)
    c.evidence_scene_share = evidence / len(scenes)
    c.number_scene_share = nums / len(scenes)
    c.average_information_density = float(sum(dens) / len(dens))
    c.median_words_per_second = float(median(wps)) if wps else 2.5
    if tr is not None and tr.sentences:
        c.median_sentence_seconds = float(median(max(0.0, s.end - s.start) for s in tr.sentences))
    assigned = [a for a in project.visual_assignments.values() if a.asset_id and not a.skipped]
    stills = 0
    for a in assigned:
        asset = project.assets.get(a.asset_id)
        if asset is not None and asset.type.value == "image":
            stills += 1
    c.still_image_share = stills / len(assigned) if assigned else 0.0
    return c


# ---------------------------------------------------------------------------------------------- where the style is blended from
def adaptation_baseline(project: "Project") -> AdaptationBaseline:
    """The user's own settings in the units of the override parameters (never the already style-adjusted values, so applying twice does not compound)."""
    from dataclasses import replace  # noqa: PLC0415

    from app.editing.presets import preset_for, shot_factor  # noqa: PLC0415

    es = replace(project.editing_settings, reference=None)
    preset = preset_for(es)
    cs, au = project.caption_settings, project.audio_settings
    text_density = 0.5 if (es.text_emphasis or es.number_emphasis) else 0.0
    ducking = ducking_strength(au.music_level, au.important_level) if au.auto_ducking else 0.0
    pause_usage = clamp((au.pause_level / au.music_level - 1.0) / PAUSE_LEVEL_GAIN) if au.music_level > 0 else 0.5
    return AdaptationBaseline(
        base_shot_duration=preset.base_shot * shot_factor(es), min_shot_duration=preset.min_shot, max_shot_duration=preset.max_shot, motion_intensity=es.motion_intensity,
        transition_frequency=es.transition_frequency if es.smart_transitions else 0.0, text_density=text_density, text_per_minute=preset.text_per_minute,
        caption_max_words=cs.max_words, caption_style=cs.style_id, caption_position=cs.position if cs.position in ("bottom", "center", "top") else "bottom",
        keyword_emphasis_rate=0.5 if cs.keyword_highlight else 0.0, music_level=au.music_level if au.music_enabled else 0.0, ducking_strength=ducking,
        sfx_per_minute=au.max_sfx_per_minute if au.sfx_enabled else 0.0, pause_usage=pause_usage,
        user_set={"editing": list(es.user_set), "caption": list(cs.user_set), "audio": list(au.user_set)})
