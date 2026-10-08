"""Rule-based planners: motion, evidence treatment, text/number emphasis, transitions, audio and caption instructions.

Each planner returns plain plan objects; nothing here touches the timeline. Reasons are concise decision factors,
never chain-of-thought.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.analysis.models import EntityType, NumberKind
from app.editing.context import AssetInfo, SceneContext
from app.editing.effective import duck_level_for, pause_level_for
from app.editing.models import (
    AudioPlan,
    DecisionType,
    DuckPlan,
    Emphasis,
    EvidencePlan,
    MotionPlan,
    PanKind,
    PlannedText,
    SceneEditingBrief,
    TextGraphic,
    TextStyle,
    TransitionPlan,
    TransitionType,
    VisualSegment,
    ZoomKind,
)
from app.editing.presets import StylePreset, motion_ratio, motion_scale, transition_threshold
from app.editing.timing import number_priority, stable_unit
from app.timeline.keyframes import Keyframe

# ====================================================================== keyframe synthesis (shared by planner and inspector)


def motion_keyframes(plan: MotionPlan, duration: float, decision_id: str = "") -> list[Keyframe]:
    """Turn motion parameters into editable keyframes. The parameters stay the source of truth for the AI decision."""
    kfs: list[Keyframe] = []
    if plan.kind == ZoomKind.NO_ZOOM.value:
        return kfs
    pairs = (("scale", plan.start_scale, plan.end_scale), ("position_x", plan.start_pos[0], plan.end_pos[0]),
             ("position_y", plan.start_pos[1], plan.end_pos[1]))
    for prop, a, b in pairs:
        if abs(a - b) > 1e-9 or (prop == "scale" and abs(a - 1.0) > 1e-9):
            kfs.append(Keyframe(prop, 0.0, round(a, 4), plan.interpolation, decision_id))
            kfs.append(Keyframe(prop, round(duration, 4), round(b, 4), plan.interpolation, decision_id))
    return kfs


def evidence_keyframes(ev: EvidencePlan, duration: float, canvas: tuple[int, int], decision_id: str = "") -> list[Keyframe]:
    """Wide view -> zoom to the region -> hold -> back to the wider view (when the shot is long enough)."""
    W, H = canvas
    x, y, w, h = ev.region
    cx, cy = x + w / 2, y + h / 2
    s = max(1.0, ev.zoom_scale)
    px, py = round((0.5 - cx) * W * s, 2), round((0.5 - cy) * H * s, 2)
    hold = min(ev.hold_wide, duration * 0.25)
    zin = min(duration * 0.7, hold + max(0.6, duration * 0.35))
    pts: list[tuple[float, float, float, float]] = [(0.0, 1.0, 0.0, 0.0), (hold, 1.0, 0.0, 0.0), (zin, s, px, py)]
    if duration >= 4.0:
        back = duration - max(0.6, duration * 0.25)
        pts += [(back, s, px, py), (duration, 1.0, 0.0, 0.0)]
    else:
        pts.append((duration, s, px, py))
    out: list[Keyframe] = []
    for t, sc, ox, oy in pts:
        out += [Keyframe("scale", round(t, 4), sc, "ease_in_out", decision_id), Keyframe("position_x", round(t, 4), ox, "ease_in_out", decision_id),
                Keyframe("position_y", round(t, 4), oy, "ease_in_out", decision_id)]
    return out


# ====================================================================== motion
class MotionPlanner:
    def __init__(self, preset: StylePreset, settings) -> None:
        self.preset, self.settings = preset, settings

    def plan(self, seg: VisualSegment, asset: AssetInfo, ctx: SceneContext, brief: SceneEditingBrief, index: int, canvas: tuple[int, int],
             has_evidence: bool) -> MotionPlan | None:
        if brief.keep_static or has_evidence:
            return None
        W, H = canvas
        m = motion_scale(self.settings)
        ratio = motion_ratio(self.preset, self.settings)
        u = stable_unit(ctx.scene.id, "motion", index)
        dur = seg.duration
        important = brief.importance >= 0.8
        if dur < 1.0:
            return None
        if not asset.is_still:  # video already moves: keep it nearly still unless the line deserves a punch-in
            if important and dur <= 6 and u < 0.7:
                return MotionPlan(ZoomKind.PUNCH_IN.value, "ZOOM", 1.0, round(1 + self.preset.punch * m * 0.8, 3), interpolation="ease_out",
                                  reason="Important statement: stronger punch-in on moving footage.", confidence=86)
            if u < ratio * 0.45:
                return MotionPlan(ZoomKind.SUBTLE_ZOOM.value, "ZOOM", 1.0, round(1 + self.preset.subtle * m * 0.6, 3),
                                  reason="Footage already moves: only a very subtle push.", confidence=88)
            return None
        if seg.fit == "contain":  # documents, charts: never crop away information
            if u < ratio * 0.6:
                return MotionPlan(ZoomKind.SUBTLE_ZOOM.value, "ZOOM", 1.0, round(1 + self.preset.subtle * m * 0.7, 3),
                                  reason="Slow push on a document; the full page stays visible.", confidence=84)
            return None
        if important and dur <= 6:
            return MotionPlan(ZoomKind.PUNCH_IN.value, "ZOOM", 1.0, round(1 + self.preset.punch * m, 3), interpolation="ease_out",
                              reason="Important statement requires stronger visual emphasis.", confidence=90)
        if u >= ratio:
            return None  # not every clip moves
        w, h = asset.width or W, asset.height or H
        r, cr = w / h, W / H
        if r > cr * 1.15:  # wide photo: travel across it
            s = round(1.0 + 0.05 + 0.06 * m, 3)
            a = round((s - 1) * W / 2 * 0.85, 1)
            left = stable_unit(ctx.scene.id, "dir", index) < 0.5
            kind = PanKind.PAN_LEFT if left else PanKind.PAN_RIGHT
            return MotionPlan(kind.value, "PAN", s, s, (a if left else -a, 0.0), (-a if left else a, 0.0), "linear",
                              "Wide composition: slow pan reveals the whole frame.", 88)
        if r < 0.9:  # portrait: drift toward the upper part (faces)
            s = round(1.0 + 0.06 + 0.06 * m, 3)
            return MotionPlan(PanKind.ZOOM_IN.value, "PAN", 1.0, s, (0.0, 0.0), (0.0, round((s - 1) * H / 2 * 0.5, 1)),
                              reason="Portrait framing: gentle push toward the upper area.", confidence=84)
        if u < ratio * 0.5:
            s = round(1.0 + self.preset.subtle * m, 3)
            return MotionPlan(ZoomKind.SUBTLE_ZOOM.value, "ZOOM", 1.0, s, reason="Still image: subtle push keeps the shot alive.", confidence=88)
        s = round(1.0 + self.preset.subtle * m, 3)
        return MotionPlan(PanKind.ZOOM_OUT.value, "PAN", s, 1.0, reason="Still image: slow pull-back reveals context.", confidence=86)


# ====================================================================== evidence
EVIDENCE_SOURCES = {"SCREENSHOT"}


def is_evidence_visual(ctx: SceneContext, asset: AssetInfo | None) -> bool:
    if asset is None:
        return False
    if ctx.candidate is not None and ctx.candidate.evidence_kind.value == "EVIDENCE":
        return True
    return ctx.visual_type in ("EVIDENCE", "DATA") and asset.is_still or asset.source_type in EVIDENCE_SOURCES


class EvidencePlanner:
    def plan(self, ctx: SceneContext, seg: VisualSegment, asset: AssetInfo, brief: SceneEditingBrief) -> EvidencePlan | None:
        if not brief.evidence_treatment_needed or seg.duration < 2.0:
            return None
        region = (0.12, 0.28, 0.76, 0.22)
        focus = getattr(ctx.score, "focus", "") if ctx.score else ""
        return EvidencePlan(
            region=region, region_detected=False, zoom_scale=1.8 if seg.duration >= 3 else 1.5, highlight=True, darken=True,
            reason=("Evidence document: zoom to the relevant area, highlight it, then return to the wider view. "
                    "The region was not detected automatically: check it" + (f" ({focus})" if focus else "") + "."),
            confidence=64.0)


# ====================================================================== text / numbers / emphasis
WARN_WORDS = ("penalty", "penalties", "warning", "deadline", "fine", "fines", "illegal", "mandatory", "must", "violation", "audit")
TIME_RE = re.compile(r"[^0-9.]")


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9%$]+", " ", t.lower()).strip()


def _numeric_value(token: str) -> float | None:
    cleaned = TIME_RE.sub("", token.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


class TextPlanner:
    def __init__(self, preset: StylePreset, settings) -> None:
        self.preset, self.settings = preset, settings

    @staticmethod
    def number_content(mention, scene) -> tuple[str, str]:
        """The on-screen text for a figure, taken from the script/transcript only. Returns (content, source)."""
        text = mention.text.strip().strip(".,;:")
        if mention.value is not None and scene.script_text:
            for tok in scene.script_text.split():
                clean = tok.strip(".,;:!?()\"'")
                if any(ch.isdigit() for ch in clean) and _numeric_value(clean) == mention.value and clean:
                    return clean, "script"
        return text, "transcript"

    def plan(self, ctx: SceneContext, brief: SceneEditingBrief, segments: list[VisualSegment], graphics_budget: float = 1.0) -> list[PlannedText]:
        s = ctx.scene
        out: list[PlannedText] = []
        wmap = {w.word_id: w for w in ctx.words}
        narr = _norm(s.narration + " " + s.script_text)
        allowed = max(1, int(round(self.preset.text_per_minute * s.duration / 60.0 * graphics_budget + 0.49)))
        n = 0

        def add(content: str, word_id: str | None, style: TextStyle, emphasis: Emphasis, dtype: DecisionType, source: str, reason: str,
                importance: float, conf: float, lead: float = 0.15, min_dur: float = 1.8, pos: tuple[float, float] | None = None) -> None:
            nonlocal n
            if not content.strip() or _norm(content) not in narr and not all(tok in narr for tok in _norm(content).split()):
                return  # nothing is ever invented: the text must come from the narration/script
            w = wmap.get(word_id) if word_id else None
            start = max(s.start, (w.start if w else s.start) - lead)
            dur = min(4.2, max(min_dur, 1.4 + 0.07 * len(content)))
            if start + dur > s.end:
                dur = s.end - start
                if dur < 0.9:
                    start = max(s.start, s.end - 0.9)
                    dur = s.end - start
            if dur < 0.6:
                return
            clash = next((o for o in out if start < o.graphic.start + o.graphic.duration - 1e-6 and start + dur > o.graphic.start + 1e-6), None)
            if clash is not None:  # one text at a time on the text track: shift after the one in the way, or drop
                start = clash.graphic.start + clash.graphic.duration + 0.05
                if start + 0.9 > s.end or any(start < o.graphic.start + o.graphic.duration - 1e-6 and start + 0.9 > o.graphic.start + 1e-6 for o in out):
                    return
                dur = min(dur, s.end - start)
            positions = {TextStyle.NUMBER_CARD: (0.5, 0.42), TextStyle.LOWER_THIRD: (0.08, 0.80), TextStyle.HEADLINE: (0.5, 0.14),
                         TextStyle.WARNING: (0.5, 0.20), TextStyle.DATE: (0.5, 0.40), TextStyle.LABEL: (0.5, 0.80),
                         TextStyle.ENTITY_NAME: (0.08, 0.80), TextStyle.LOCATION: (0.08, 0.12)}
            g = TextGraphic(
                text_id=f"{s.id}_t{n}", content=content, start=round(start, 3), duration=round(dur, 3), position=pos or positions.get(style, (0.5, 0.8)),
                style=style.value, emphasis=emphasis.value, animation="pop" if emphasis in (Emphasis.PUNCH_TEXT, Emphasis.NUMBER_CARD) else "fade",
                importance=round(importance, 2), source_scene=s.id, source_ref=source,
                size=88 if style is TextStyle.NUMBER_CARD else 64 if emphasis is Emphasis.PUNCH_TEXT else 48,
                alignment="left" if style in (TextStyle.LOWER_THIRD, TextStyle.ENTITY_NAME, TextStyle.LOCATION) else "center",
                background="box" if style in (TextStyle.LOWER_THIRD, TextStyle.WARNING, TextStyle.ENTITY_NAME) else "none")
            out.append(PlannedText(g, dtype.value, f"{'number' if dtype is DecisionType.NUMBER_EMPHASIS else 'text'}:{n}", reason, conf))
            n += 1

        # --- important numbers: only figures actually spoken in this scene
        if self.settings.number_emphasis:
            ranked = sorted(s.numbers, key=lambda m: (-number_priority(m.kind), wmap[m.word_ids[0]].start if m.word_ids and m.word_ids[0] in wmap else 0))
            for m in ranked:
                if n >= min(allowed, 2):
                    break
                if number_priority(m.kind) < 3 and not (brief.importance >= 0.7 and number_priority(m.kind) >= 2):
                    continue
                content, src = self.number_content(m, s)
                wid = m.word_ids[0] if m.word_ids else None
                if m.kind in (NumberKind.DATE, NumberKind.DEADLINE, NumberKind.YEAR):
                    add(content, wid, TextStyle.DATE, Emphasis.BOLD_TEXT, DecisionType.NUMBER_EMPHASIS, src,
                        f"Date “{content}” stated in the narration is shown for orientation.", brief.importance, 88)
                else:
                    add(content, wid, TextStyle.NUMBER_CARD, Emphasis.NUMBER_CARD, DecisionType.NUMBER_EMPHASIS, src,
                        "Important statistic requires stronger visual emphasis.", brief.importance, 92)
        # --- text emphasis: names, warnings, key statements
        if self.settings.text_emphasis:
            if n < allowed:
                for e in ctx.new_entities:
                    if n >= allowed:
                        break
                    wid = e.word_ids[0] if e.word_ids else None
                    if e.type is EntityType.PERSON:
                        add(e.text, wid, TextStyle.LOWER_THIRD, Emphasis.LOWER_THIRD, DecisionType.TEXT, "transcript",
                            f"{e.text} is introduced here for the first time.", 0.6, 88)
                    elif e.type in (EntityType.GOVERNMENT_AGENCY, EntityType.ORGANIZATION, EntityType.COMPANY) and brief.importance >= 0.5:
                        add(e.text, wid, TextStyle.ENTITY_NAME, Emphasis.SUBTLE_HIGHLIGHT, DecisionType.TEXT, "transcript",
                            f"{e.text} is introduced here for the first time.", 0.55, 82)
            low = s.narration.lower()
            hit = next((w for w in WARN_WORDS if re.search(rf"\b{w}\b", low)), None)
            if hit and n < allowed and brief.importance >= 0.5 and not any(p.decision_type == DecisionType.NUMBER_EMPHASIS.value for p in out):
                word = next((w for w in ctx.words if w.text.lower().strip(".,;:!?") == hit), None)
                add(hit.upper(), word.word_id if word else None, TextStyle.WARNING, Emphasis.WARNING_TEXT, DecisionType.TEXT, "transcript",
                    f"The narration warns about “{hit}”.", brief.importance, 80)
            if brief.importance >= 0.8 and n < allowed and not out:
                claim = next((c for c in s.claims if c.requires_evidence), None)
                if claim:
                    phrase = " ".join(claim.text.replace("’", "'").split()[:6]).strip(".,;:")
                    add(phrase, None, TextStyle.HEADLINE, Emphasis.PUNCH_TEXT, DecisionType.TEXT, "transcript",
                        "Key claim of an important scene gets on-screen emphasis.", brief.importance, 78, lead=0.0, min_dur=2.2)
        out.sort(key=lambda t: t.graphic.start)
        return out


# ====================================================================== transitions
TIME_CUES = ("years later", "decades", "back in", "in the past", "later that", "last year", "next year", "long ago", "years ago", "by then")


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b and (a | b) else 0.0


class TransitionPlanner:
    def __init__(self, preset: StylePreset, settings) -> None:
        self.preset, self.settings = preset, settings

    def plan(self, ctx: SceneContext) -> TransitionPlan:
        if ctx.prev is None:
            return TransitionPlan(TransitionType.CUT.value, 0.0, "First scene: the edit opens directly.", 95)
        if not self.settings.smart_transitions:
            return TransitionPlan(TransitionType.CUT.value, 0.0, "Smart transitions are off: hard cut.", 95)
        s, prev = ctx.scene, ctx.prev
        terms = {w for w in "".join(c.lower() if c.isalnum() else " " for c in s.topic + " " + s.summary).split() if len(w) > 3}
        ents = {(e.canonical or e.text).lower() for e in s.entities}
        sim = max(_jaccard(terms, prev.terms), 0.7 * _jaccard(ents, prev.entities))
        gap = (ctx.words[0].start if ctx.words else s.start) - prev.last_word_end
        low = s.narration.lower()
        time_cue = any(c in low for c in TIME_CUES)
        loc_change = ctx.visual_type == "LOCATION" and prev.visual_type != "LOCATION"
        score = 0.40 * (1 - sim) + (0.35 if ctx.starts_section else 0.0) + min(0.15, max(0.0, gap) / 1.5 * 0.15)
        score += 0.10 if ctx.visual_type != prev.visual_type else 0.0
        score += 0.05 if ctx.asset and ctx.asset.source_type != prev.source_type else 0.0
        score += 0.25 if time_cue else 0.0
        score += 0.20 if loc_change else 0.0
        thr = transition_threshold(self.preset, self.settings)
        if score < thr:
            return TransitionPlan(TransitionType.CUT.value, 0.0, "The topic continues: a hard cut keeps the pace.", round(min(97.0, 80 + (thr - score) * 40), 1))
        allowed = self.preset.transitions
        conf = round(min(95.0, 70 + (score - thr) * 80), 1)
        if time_cue or loc_change:
            kind, dur, why = "FADE", 0.8, "Time or location change: fade."
        elif ctx.visual_type == "COMPARISON" and "SLIDE" in allowed:
            kind, dur, why = "SLIDE", 0.35, "Comparison starts: slide."
        elif score >= thr + 0.25 and "WIPE" in allowed:
            kind, dur, why = "WIPE", 0.4, "Strong section break: wipe."
        elif ctx.starts_section or score >= thr + 0.1:
            kind, dur, why = "DISSOLVE", 0.6, "New section: dissolve."
        else:
            kind, dur, why = "FADE", 0.5, "Noticeable topic change: fade."
        if kind not in allowed and kind != "CUT":
            kind = "DISSOLVE" if "DISSOLVE" in allowed else "FADE"
        return TransitionPlan(kind, dur, why, conf)


# ====================================================================== audio + captions
class AudioPlanner:
    def __init__(self, preset: StylePreset, settings) -> None:
        self.preset, self.settings = preset, settings

    def global_plan(self) -> AudioPlan:
        level, duck, rise = self.preset.music_level, self.preset.music_level * 0.55, self.preset.music_level * 1.3
        ref = getattr(self.settings, "reference", None)
        if ref is not None:  # an applied reference style sets the depth of the duck and the rise in pauses (see effective.py)
            if ref.ducking_strength is not None:
                duck = duck_level_for(level, ref.ducking_strength)
            if ref.pause_usage is not None:
                rise = pause_level_for(level, ref.pause_usage)
        return AudioPlan(music_level=level, duck_level=round(duck, 3), rise_level=round(rise, 3))

    def plan(self, ctx: SceneContext, brief: SceneEditingBrief, audio: AudioPlan) -> list[DuckPlan]:
        if not self.settings.smart_audio_ducking:
            return []
        s = ctx.scene
        out: list[DuckPlan] = []
        wmap = {w.word_id: w for w in ctx.words}
        if brief.importance >= 0.75:
            out.append(DuckPlan(s.start, s.end, audio.duck_level, "DUCK", 0.4, "Important statement: music drops so the voice dominates.", 90))
        else:
            for m in s.numbers:
                w = wmap.get(m.word_ids[0]) if m.word_ids else None
                if w is not None and number_priority(m.kind) >= 3:
                    out.append(DuckPlan(max(s.start, w.start - 0.3), min(s.end, w.end + 1.0), audio.duck_level, "DUCK", 0.25,
                                        f"Figure “{m.text}” is spoken: music ducks.", 88))
        for a, b in zip(ctx.words, ctx.words[1:]):
            if b.start - a.end > 0.8:
                out.append(DuckPlan(a.end + 0.2, b.start - 0.2, audio.rise_level, "RISE", 0.3, "Voice pause: music may rise modestly.", 80))
        out.sort(key=lambda d: d.start)
        merged: list[DuckPlan] = []
        for d in out:  # merge overlapping instructions of the same kind
            if merged and d.start <= merged[-1].end and d.kind == merged[-1].kind:
                merged[-1].end = max(merged[-1].end, d.end)
            elif merged and d.start < merged[-1].end:
                d.start = merged[-1].end
                if d.end > d.start:
                    merged.append(d)
            else:
                merged.append(d)
        return merged


def caption_emphasis_words(ctx: SceneContext, texts: list[PlannedText]) -> tuple[list[str], str]:
    words: list[str] = []
    for m in ctx.scene.numbers:
        words.append(m.text.strip(".,;:"))
    for e in ctx.scene.entities:
        words.append(e.text)
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w.upper() if w.isalpha() and len(w) <= 6 else w)
    region = "bottom_safe_area_raised" if any(t.graphic.style in (TextStyle.LOWER_THIRD.value, TextStyle.ENTITY_NAME.value) for t in texts) else "bottom_safe_area"
    return out[:8], region


_ = (dataclass, EvidencePlan)
