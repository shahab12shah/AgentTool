"""Transition quality (spec 18): duration, frequency, placement and fit with the chosen editing style.

A transition is stored on the clip it leads INTO (``clip.transition = {"type", "duration"}``). A hard cut is the default and is never a problem; QC never recommends adding a
transition. It reports transitions that are too long for their neighbours, used too often or too close together, that run over an important spoken phrase (a number or the
middle of a claim), or that do not fit the style the user chose (wipes and slides in a calm documentary, a different effect every time).

Broken values (negative or unknown type, longer than the clip itself) are reported by the timeline checker (``timeline.transition.invalid``) and are not repeated here.
The only recommended change is shortening a long transition; it needs confirmation (``transition.shorten``). Because the usage rate is a property of the whole video this
checker is not scene-local, and it also compares the transition share with an applied reference style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory
from app.qc.severity import Severity
from app.qc.style_compat import style_issue
from app.reference.style_model import TRANSITION_SHARE_POINTS, scale
from app.timeline.clip import Clip
from app.timeline.track import Track

NEIGHBOUR_SHARE = 0.4  # a transition may take at most this share of the shorter clip around it
MIN_DURATION = 0.2  # a recommended shortening never goes below this
CALM_STYLES = ("documentary", "professional")
BUSY_TYPES = ("WIPE", "SLIDE")
EXPECTED_PER_MINUTE = {"documentary": 3.0, "professional": 5.0, "dynamic": 9.0}  # at the middle setting of transition frequency
MAX_PER_CODE = 6


@dataclass
class _Tr:
    track: Track
    clip: Clip
    kind: str
    duration: float
    prev: Clip | None

    @property
    def start(self) -> float:
        return self.clip.timeline_start

    @property
    def end(self) -> float:
        return self.clip.timeline_start + self.duration


class TransitionChecker(BaseChecker):
    id = "transition"
    label = "Transitions"
    categories = (QCCategory.TRANSITION, QCCategory.STYLE)
    domains = ("timeline", "transcript", "scenes", "reference")
    settings_sections = ("transition", "style")
    scene_local = False
    version = "1"

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = ctx.settings.transition
        trs, cuts = self._collect(ctx)
        duration = max(ctx.duration, 1e-6)
        types: dict[str, int] = {}
        for t in trs:
            types[t.kind] = types.get(t.kind, 0) + 1
        per_min = len(trs) / (duration / 60.0)
        out.metrics = {"transitions": len(trs), "per_minute": round(per_min, 2), "types": types, "cuts": cuts}
        if not trs:
            self._style(ctx, out, 0.0, cuts)
            return out
        report(0.2, "Checking transition lengths")
        for t in trs:
            self._too_long(ctx, out, cfg, t)
        report(0.5, "Checking how often transitions are used")
        self._frequency(ctx, out, cfg, trs, per_min)
        report(0.7, "Checking transitions against the narration and the style")
        self._in_phrase(ctx, out, cfg, trs)
        self._fit_with_style(ctx, out, trs, per_min, types)
        self._style(ctx, out, len(trs) / max(1, cuts), cuts)
        report(1.0, "Transition check complete")
        return out

    # ------------------------------------------------------------------ collection
    def _collect(self, ctx: QCContext) -> tuple[list[_Tr], int]:
        trs: list[_Tr] = []
        boundaries = 0
        by_track: dict[str, list[Clip]] = {}
        for t, c in ctx.visual_clips():
            by_track.setdefault(t.id, []).append(c)
        for t, c in ctx.visual_clips():
            seq = by_track[t.id]
            i = seq.index(c)
            prev = seq[i - 1] if i > 0 else None
            if prev is not None:
                boundaries += 1
            tr = c.transition or {}
            kind = str(tr.get("type", "CUT")).upper()
            try:
                d = float(tr.get("duration", 0.0))
            except (TypeError, ValueError):
                continue
            if kind == "CUT" or not math.isfinite(d) or d <= 1e-6 or not math.isfinite(c.timeline_start) or prev is None:
                continue
            trs.append(_Tr(t, c, kind, d, prev))
        return sorted(trs, key=lambda x: x.start), boundaries

    # ------------------------------------------------------------------ findings
    def _too_long(self, ctx: QCContext, out: CheckerOutput, cfg, t: _Tr) -> None:
        near = [t.clip.duration] + ([t.prev.duration] if t.prev is not None else [])
        limit = min(cfg.max_duration, NEIGHBOUR_SHARE * min(near))
        if t.duration <= limit + 1e-6 or t.duration > t.clip.duration + 1e-6:  # beyond the clip itself: timeline.transition.invalid
            return
        target = max(MIN_DURATION, round(limit, 2))
        why_limit = f"{cfg.max_duration:.1f} s" if limit >= cfg.max_duration - 1e-6 else f"{NEIGHBOUR_SHARE:.0%} of the shorter neighbouring clip ({min(near):.1f} s)"
        out.issues.append(self.issue(
            "transition.too_long", QCCategory.TRANSITION, Severity.WARNING, "Transition is too long", description=f"The {t.kind.lower()} into this clip takes {t.duration:.2f} s; the limit here is {limit:.2f} s ({why_limit}).",
            clip=t.clip, track=t.track, start=t.start, end=t.end, why="A long transition eats into the pictures on both sides and slows the story.", current=f"{t.duration:.2f} s", recommended=f"{target:.2f} s or shorter",
            suggested_fix=f"Shorten the transition to {target:.2f} s.", fix=fx.transition_shorten(t.clip.id, target, ctx.settings), viewer_impact=0.35, signature=sha(t.clip.id, round(t.duration, 1)),
            metrics={"duration": round(t.duration, 3), "limit": round(limit, 3)}, ctx=ctx))

    def _frequency(self, ctx: QCContext, out: CheckerOutput, cfg, trs: list[_Tr], per_min: float) -> None:
        worst, at = 0, 0.0
        for i, t0 in enumerate(trs):
            n = sum(1 for t in trs[i:] if t.start < t0.start + 60.0)
            if n > worst:
                worst, at = n, t0.start
        limit = cfg.max_per_minute
        if worst > limit:
            out.issues.append(self.issue(
                "transition.excessive", QCCategory.TRANSITION, Severity.WARNING, "Too many transitions", description=f"{worst} transitions within one minute from {_t(at)} (limit {limit:.0f}); {len(trs)} in the whole video.",
                start=at, end=at + 60.0, why="Constant effects draw attention to the editing instead of the story.", current=f"{worst} per minute", recommended=f"at most {limit:.0f} per minute",
                suggested_fix="Use hard cuts for most scene changes and keep transitions for real changes of topic.", viewer_impact=0.4, signature=sha(round(at / 30)), ctx=ctx))
        close = [(a, b) for a, b in zip(trs, trs[1:]) if b.start - a.end < cfg.min_spacing and b.start - a.start < 60.0]
        for a, b in close[:MAX_PER_CODE]:
            out.issues.append(self.issue(
                "transition.excessive", QCCategory.TRANSITION, Severity.NOTICE, "Transitions too close together", description=f"Two transitions {max(0.0, b.start - a.end):.1f} s apart at {_t(a.start)} and {_t(b.start)} "
                f"(minimum {cfg.min_spacing:.1f} s).", clip=b.clip, track=b.track, start=a.start, end=b.end, why="Back-to-back effects blur into one restless movement.", current=f"{max(0.0, b.start - a.end):.1f} s apart",
                recommended=f"at least {cfg.min_spacing:.1f} s", suggested_fix="Turn one of them into a hard cut.", viewer_impact=0.25, signature=sha(a.clip.id, b.clip.id), confidence=90.0, ctx=ctx))
        _ = per_min

    def _in_phrase(self, ctx: QCContext, out: CheckerOutput, cfg, trs: list[_Tr]) -> None:
        margin = cfg.important_phrase_margin_ms / 1000.0
        words = {w.word_id: w for w in ctx.words}
        sentences = {s.sentence_id: s for s in ctx.sentences}
        done = 0
        for t in trs:
            if done >= MAX_PER_CODE:
                break
            sc = ctx.scene_at(t.start + 1e-3)
            if sc is None:
                continue
            reason = ""
            for n in sc.numbers:
                ws = [words[i] for i in n.word_ids if i in words]
                if ws and t.start - margin < max(w.end for w in ws) and t.end + margin > min(w.start for w in ws):
                    reason = f"the spoken number “{n.text}” ({_t(min(w.start for w in ws))})"
                    break
            if not reason:
                for c in sc.claims:
                    s = sentences.get(c.sentence_id)
                    if s is not None and s.start + margin < t.start < s.end - margin:
                        reason = f"the middle of the claim “{c.text[:60]}”"
                        break
            if not reason:
                continue
            done += 1
            out.issues.append(self.issue(
                "transition.in_phrase", QCCategory.TRANSITION, Severity.NOTICE, "Transition over an important phrase", description=f"The {t.kind.lower()} at {_t(t.start)} ({t.duration:.2f} s) runs over {reason}.",
                scene_id=sc.id, clip=t.clip, track=t.track, start=t.start, end=t.end, why="The picture changes just as the viewer should be taking in the key information.", current=f"{t.kind.lower()} {t.duration:.2f} s",
                recommended="a hard cut or a transition between phrases", suggested_fix="Move the transition to a pause or sentence boundary, or use a hard cut.", confidence=75.0, viewer_impact=0.3,
                signature=sha(t.clip.id, reason[:20]), ctx=ctx))

    def _fit_with_style(self, ctx: QCContext, out: CheckerOutput, trs: list[_Tr], per_min: float, types: dict[str, int]) -> None:
        es = ctx.project.editing_settings
        style = str(es.style or "professional").lower()
        busy = [t for t in trs if t.kind in BUSY_TYPES]
        if busy and style == "documentary":
            for t in busy[:MAX_PER_CODE]:
                out.issues.append(self.issue(
                    "transition.distracting", QCCategory.TRANSITION, Severity.NOTICE, f"A {t.kind.lower()} in a documentary edit", description=f"A {t.kind.lower()} at {_t(t.start)} stands out in the calm documentary style you chose.",
                    clip=t.clip, track=t.track, start=t.start, end=t.end, why="Wipes and slides call attention to the editing; documentary pacing favours cuts and soft dissolves.", current=t.kind.lower(),
                    recommended="cut, fade or dissolve", suggested_fix="Replace it with a dissolve or a hard cut.", confidence=80.0, viewer_impact=0.25, signature=sha(t.clip.id, t.kind), ctx=ctx))
        elif len(busy) >= 3 and style == "professional":
            t = busy[0]
            out.issues.append(self.issue(
                "transition.distracting", QCCategory.TRANSITION, Severity.NOTICE, "Several wipes or slides", description=f"{len(busy)} wipes or slides (first at {_t(t.start)}) in a professional style.",
                clip=t.clip, track=t.track, start=t.start, end=t.end, why="Frequent graphic transitions look dated and pull attention away from the content.", current=f"{len(busy)} wipes/slides", recommended="fades, dissolves and cuts",
                suggested_fix="Keep one signature transition at most.", confidence=70.0, viewer_impact=0.2, signature=sha("busy", len(busy)), ctx=ctx))
        if len(trs) >= 4 and len(types) >= 4:
            out.issues.append(self.issue(
                "transition.distracting", QCCategory.TRANSITION, Severity.NOTICE, "A different transition every time", description=f"{len(types)} different transition types across {len(trs)} transitions ({', '.join(sorted(types))}).",
                why="Consistent transitions feel intentional; a new effect each time feels random.", current=f"{len(types)} types", recommended="one or two types", suggested_fix="Standardise on one or two transition types.",
                confidence=75.0, viewer_impact=0.2, signature=sha(sorted(types)), ctx=ctx))
        expected = EXPECTED_PER_MINUTE.get(style, 5.0) * (0.5 + float(es.transition_frequency))
        too_many = any(i.code == "transition.excessive" and i.severity is Severity.WARNING for i in out.issues)
        if per_min > expected * 1.5 and per_min > 1.0 and not too_many:
            out.issues.append(self.issue(
                "transition.style_mismatch", QCCategory.TRANSITION, Severity.NOTICE, "More transitions than the chosen style calls for",
                description=f"{per_min:.1f} transitions per minute; the {style} style at transition frequency {es.transition_frequency:.0%} expects about {expected:.1f}.", why="The edit feels busier than the style you chose.",
                current=f"{per_min:.1f}/min", recommended=f"about {expected:.1f}/min", suggested_fix="Remove transitions that are not at a change of topic, or change the style settings.",
                confidence=75.0, viewer_impact=0.2, signature=sha(style, round(per_min)), ctx=ctx))
        elif style in CALM_STYLES:
            unmotivated = [t for t in trs if not self._topic_change(ctx, t)]
            if len(unmotivated) >= 3 and len(unmotivated) / len(trs) > 0.5:
                t = unmotivated[0]
                out.issues.append(self.issue(
                    "transition.style_mismatch", QCCategory.TRANSITION, Severity.NOTICE, "Transitions that are not at a change of topic",
                    description=f"{len(unmotivated)} of {len(trs)} transitions lead into scenes that continue the same topic (first at {_t(t.start)}).", clip=t.clip, track=t.track, start=t.start, end=t.end,
                    why=f"In a {style} edit transitions mark a change of topic; between related scenes a cut is expected.", current=f"{len(unmotivated)} unmotivated", recommended="transitions at topic changes only",
                    suggested_fix="Use hard cuts between scenes of the same topic.", confidence=65.0, viewer_impact=0.2, signature=sha(len(unmotivated), len(trs)), ctx=ctx))

    @staticmethod
    def _topic_change(ctx: QCContext, t: _Tr) -> bool:
        sc = ctx.scene_at(t.start + 1e-3)
        if sc is None:
            return True
        idx = ctx.scenes.index(sc)
        if idx == 0:
            return True
        a = {w for w in sc.topic.lower().split() if len(w) > 3}
        b = {w for w in ctx.scenes[idx - 1].topic.lower().split() if len(w) > 3}
        c = ctx.scene_ctx(sc.id)
        if c is not None and c.starts_section:
            return True
        return not a or not b or len(a & b) / len(a | b) < 0.2

    def _style(self, ctx: QCContext, out: CheckerOutput, share: float, cuts: int) -> None:
        measured = scale(share, TRANSITION_SHARE_POINTS)
        out.metrics["non_cut_share"] = round(share, 3)
        iss = style_issue(self, ctx, "transition_frequency", measured)
        if iss is not None:
            out.issues.append(iss)


def _t(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"
