"""Pacing and cut timing (spec 13-14): does the rhythm of the edit serve the narration?

The checker measures the shot structure (shots, cuts per minute, a pacing curve over fixed windows) and compares it with what the narration supports (sentences and
words per second, from the same ``narration_stats`` the editing engine uses). It never optimises for *more* cuts: story and narration come first, so a slow stretch is
at most a NOTICE and a fast one is only reported when the narration cannot carry it. Findings are judgements (confidence below 100) and every fix is left to the user:
changing cuts is a creative decision, so no finding offers an automatic change.

Pace thresholds scale with ``pacing.sensitivity``: an upper limit is multiplied by ``1.5 - sensitivity`` (0.5 keeps the configured value, 1.0 halves it, 0.0 raises it by half).
Per-scene holds and short-shot clusters inside one scene belong to the scene checker (scene.hold.excessive / scene.fragmentation.*); this checker looks at the rhythm across
scenes and at where cuts land in the speech.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

from app.editing.planners import EVIDENCE_SOURCES, is_evidence_visual
from app.editing.timing import narration_stats, speed_class
from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory
from app.qc.severity import Severity
from app.qc.style_compat import style_issue, target_score
from app.reference.style_model import PACING_POINTS, pacing_class, scale
from app.timeline.clip import KIND_GRAPHIC, KIND_TEXT, Clip
from app.timeline.track import Track

CONTIGUOUS = 0.04  # s: source ranges this close are one continuous shot (a split, not a cut)
HOOK_SECONDS = 8.0  # the opening may be faster on purpose
PHRASE_GAP = 0.25  # a gap between words longer than this ends a phrase
NUMBER_RE = re.compile(r"^[\$€£]?\d[\d,\.]*%?$")
UNITS = frozenset({"percent", "per", "million", "billion", "trillion", "thousand", "hundred", "dollars", "dollar", "euros", "ounces", "ounce", "oz", "tons", "tonnes", "years", "year",
                   "months", "days", "kilograms", "kg", "bps", "points", "times", "x", "%"})
MIN_MEDIAN_CPM = 3.0  # floor for the "median window" so a nearly static video does not make every cut look wild
MIN_READ_SECONDS = 1.2  # an information overlay needs about this long on screen to be read
MAX_ISSUES_PER_CODE = 8


@dataclass
class _Shot:
    start: float
    end: float  # visible end (clipped by the next shot's start)
    clip: Clip
    track: Track
    scene_id: str | None
    asset_id: str
    moving: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _Window:
    start: float
    end: float
    cuts: int = 0
    events: int = 0
    sentences: int = 0
    words: int = 0
    wps: float = 0.0

    @property
    def minutes(self) -> float:
        return max(1e-6, (self.end - self.start) / 60.0)

    @property
    def cpm(self) -> float:
        return self.cuts / self.minutes

    @property
    def epm(self) -> float:
        return self.events / self.minutes

    @property
    def sentences_pm(self) -> float:
        return self.sentences / self.minutes


@dataclass
class _Run:
    """Consecutive flagged windows reported as one finding."""

    start: float
    end: float
    worst: float
    windows: list[_Window] = field(default_factory=list)


class PacingChecker(BaseChecker):
    id = "pacing"
    label = "Pacing & cut timing"
    categories = (QCCategory.PACING, QCCategory.CUT_TIMING, QCCategory.STYLE)
    domains = ("timeline", "scenes", "transcript", "reference")
    settings_sections = ("pacing", "coverage", "style")
    scene_local = False
    version = "1"

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = ctx.settings.pacing
        self.f = 1.5 - max(0.0, min(1.0, cfg.sensitivity))  # >1 = more lenient, <1 = stricter
        shots = self._shots(ctx)
        duration = max(ctx.duration, shots[-1].end if shots else 0.0)
        if not shots or duration <= 0:
            out.notes.append("no visual shots to analyse")
            out.metrics = {"shots": 0}
            return out
        words = ctx.words
        stats = narration_stats(words, 0.0, duration)
        windows = self._windows(ctx, shots, duration)
        metrics = self._metrics(shots, windows, duration, stats.wps)
        out.metrics = metrics
        report(0.2, "Checking the pace")
        self._too_fast(ctx, out, cfg, windows)
        self._too_slow(ctx, out, cfg, shots)
        self._uneven(ctx, out, cfg, windows)
        self._over_edited(ctx, out, cfg, windows, stats.wps)
        self._under_emphasized(ctx, out, shots)
        ctx.check_cancel()
        report(0.6, "Checking where the cuts land")
        cuts = [s.start for s in shots[1:]]
        self._awkward_cuts(ctx, out, cfg, shots)
        self._unnecessary_cuts(ctx, out, shots)
        self._micro_cuts(ctx, out, cfg, shots)
        self._before_content(ctx, out, cuts)
        report(0.9, "Comparing with the reference style")
        self._style(ctx, out, metrics, stats.wps)
        out.metrics["style"] = {"pacing_score": round(scale(metrics["cuts_per_minute"], PACING_POINTS), 1), "pacing_class": pacing_class(metrics["cuts_per_minute"])}
        report(1.0, "Pacing check complete")
        return out

    # ------------------------------------------------------------------ shot structure
    def _shots(self, ctx: QCContext) -> list[_Shot]:
        """One shot per picture the viewer sees: clips on the video/image tracks, a continuing clip (same footage, contiguous) merged into the shot it continues."""
        rows = [(t, c) for t, c in ctx.visual_clips() if c.duration > 1e-3 and c.timeline_start >= -1e-6]
        rows.sort(key=lambda r: (r[1].timeline_start, r[1].id))
        shots: list[_Shot] = []
        for t, c in rows:
            prev = shots[-1] if shots else None
            if prev and prev.asset_id == c.asset_id and abs(prev.clip.timeline_end - c.timeline_start) <= ctx.frame * 1.5 and abs(prev.clip.source_out - c.source_in) <= CONTIGUOUS:
                prev.end = max(prev.end, c.timeline_end)
                prev.clip = c  # the merged shot now ends with this clip (so the next one can continue it)
                continue
            shots.append(_Shot(c.timeline_start, c.timeline_end, c, t, ctx.clip_scene_id(c), c.asset_id, moving=self._moves(c)))
        for a, b in zip(shots, shots[1:]):
            a.end = min(a.end, max(a.start, b.start))  # a following picture takes over the screen
        return shots

    @staticmethod
    def _moves(c: Clip) -> bool:
        return any(k.property in ("scale", "position_x", "position_y", "rotation") for k in c.keyframes) or bool(c.animation) or bool(c.effects.get("focus_region") or c.effects.get("highlight"))

    def _events(self, ctx: QCContext) -> list[float]:
        """Times of things that change the picture besides cuts: motion starts, text / graphic overlays, real transitions."""
        ev: list[float] = []
        for t, c in ctx.visual_clips():
            if self._moves(c):
                ev.append(c.timeline_start)
            if c.transition and str(c.transition.get("type", "CUT")).upper() != "CUT":
                ev.append(c.timeline_start)
        ev += [c.timeline_start for _t, c in ctx.clips() if c.kind in (KIND_TEXT, KIND_GRAPHIC)]
        return sorted(ev)

    def _windows(self, ctx: QCContext, shots: list[_Shot], duration: float) -> list[_Window]:
        w = max(5.0, ctx.settings.pacing.window_seconds)
        n = max(1, int(duration // w))
        bounds = [(i * w, (i + 1) * w if i < n - 1 else duration) for i in range(n)]  # the tail shorter than a window joins the last one
        cut_times = [s.start for s in shots[1:]]
        events = self._events(ctx)
        sentences = ctx.sentences
        out = []
        for a, b in bounds:
            ws = _Window(a, b)
            ws.cuts = sum(1 for t in cut_times if a <= t < b)
            ws.events = ws.cuts + sum(1 for t in events if a <= t < b)
            ws.sentences = sum(1 for s in sentences if a <= s.start < b)
            ww = ctx.words_between(a, b)
            ws.words = len(ww)
            ws.wps = narration_stats(ww, a, b).wps if ww else 0.0
            out.append(ws)
        return out

    @staticmethod
    def _metrics(shots: list[_Shot], windows: list[_Window], duration: float, wps: float) -> dict:
        durs = sorted(s.duration for s in shots)
        cuts = len(shots) - 1
        cpm = cuts / (duration / 60.0)
        return {"shots": len(shots), "cuts": cuts, "cuts_per_minute": round(cpm, 2), "average_shot": round(sum(durs) / len(durs), 2), "median_shot": round(median(durs), 2),
                "min_shot": round(durs[0], 2), "max_shot": round(durs[-1], 2), "narration_wps": round(wps, 2), "narration_class": speed_class(wps),
                "curve": [{"start": round(w.start, 1), "end": round(w.end, 1), "cuts_per_minute": round(w.cpm, 1), "label": pacing_class(w.cpm)} for w in windows]}

    # ------------------------------------------------------------------ helpers shared by the findings
    def _justified(self, ctx: QCContext, a: float, b: float) -> str:
        """Why a burst of editing in [a, b) is deliberate: the hook, an important scene, the start of a section, or a high-confidence AI timing decision with a reason."""
        if a < HOOK_SECONDS:
            return "the opening is allowed to be fast"
        for s in ctx.scenes:
            if s.end <= a or s.start >= b:
                continue
            if s.importance >= ctx.settings.coverage.important_scene:
                return f"Scene {s.label} is an important moment"
            sc = ctx.scene_ctx(s.id)
            if sc is not None and sc.starts_section and s.start >= a - 1.0:
                return f"Scene {s.label} starts a new section"
        for t, c in ctx.visual_clips():
            if c.timeline_end <= a or c.timeline_start >= b:
                continue
            d = ctx.project.editing_decisions.get(c.ai_decision_id)
            if d is not None and d.confidence >= 85 and d.reason and str(getattr(d.type, "value", d.type)) in ("VISUAL_TIMING", "CUT"):
                return "the editing decisions give a reason for these cuts"
        return ""

    def _runs(self, windows: list[_Window], flagged: list[tuple[_Window, float]]) -> list[_Run]:
        runs: list[_Run] = []
        for w, v in flagged:
            if runs and abs(runs[-1].end - w.start) < 1e-6:
                runs[-1].end, runs[-1].worst = w.end, max(runs[-1].worst, v)
                runs[-1].windows.append(w)
            else:
                runs.append(_Run(w.start, w.end, v, [w]))
        return runs

    def _scene_of(self, ctx: QCContext, t: float) -> str | None:
        s = ctx.scene_at(t)
        return s.id if s else None

    # ------------------------------------------------------------------ findings: rhythm
    def _too_fast(self, ctx: QCContext, out: CheckerOutput, cfg, windows: list[_Window]) -> None:
        limit = cfg.too_fast_cuts_per_minute * self.f
        flagged = []
        for w in windows:
            support = w.sentences_pm * ctx.settings.coverage.max_cuts_per_sentence  # what the story can carry: a couple of picture changes per sentence
            if w.cpm > limit and w.cpm > support and not self._justified(ctx, w.start, w.end):
                flagged.append((w, w.cpm))
        for r in self._runs(windows, flagged)[:MAX_ISSUES_PER_CODE]:
            support = sum(w.sentences_pm for w in r.windows) / len(r.windows) * ctx.settings.coverage.max_cuts_per_sentence
            avg = sum(w.cpm for w in r.windows) / len(r.windows)
            out.issues.append(self.issue(
                "pacing.too_fast", QCCategory.PACING, Severity.WARNING, "Pacing too fast for the narration",
                description=f"{avg:.0f} cuts per minute between {_t(r.start)} and {_t(r.end)} (limit {limit:.0f}); the narration there carries about {support:.0f} picture changes per minute.",
                start=r.start, end=r.end, scene_id=self._scene_of(ctx, r.start), why="Cutting faster than the story moves tires viewers and hides the pictures.", current=f"{avg:.0f} cuts/min",
                recommended=f"about {min(limit, max(support, 6)):.0f} cuts/min or fewer", suggested_fix="Merge some of the shots or let key visuals stay longer.", confidence=75.0, viewer_impact=0.5,
                signature=sha(round(r.start / 10), round(avg / 5)), metrics={"cuts_per_minute": round(avg, 1)}, ctx=ctx))

    def _too_slow(self, ctx: QCContext, out: CheckerOutput, cfg, shots: list[_Shot]) -> None:
        """A picture that outlasts several sentences while nothing else changes (a hold inside one scene is the scene checker's); evidence and data holds are for reading."""
        limit = cfg.too_slow_shot_seconds * self.f
        sents = ctx.sentences
        done = 0
        for s in shots:
            if s.duration <= limit or done >= MAX_ISSUES_PER_CODE:
                continue
            crossed = sum(1 for x in sents if s.start <= x.start < s.end)
            if crossed < 3 or self._moves_during(ctx, s) or self._reading_hold(ctx, s):
                continue
            done += 1
            out.issues.append(self.issue(
                "pacing.too_slow", QCCategory.PACING, Severity.NOTICE, "The picture stays unchanged while the story moves on",
                description=f"One visual stays for {s.duration:.0f} s ({_t(s.start)}-{_t(s.end)}) while {crossed} sentences are spoken, with no movement, text or graphic.", start=s.start, end=s.end,
                scene_id=s.scene_id, clip=s.clip, track=s.track, why="A static picture under a changing story loses attention; the story may deserve a visual of its own.",
                current=f"{s.duration:.0f} s, {crossed} sentences", recommended=f"under {limit:.0f} s or visible movement", suggested_fix="Consider a second visual or a slow zoom here; a deliberate hold is fine.",
                confidence=65.0, viewer_impact=0.3, signature=sha(s.clip.id), ctx=ctx))

    def _moves_during(self, ctx: QCContext, s: _Shot) -> bool:
        if any(self._moves(c) for _t, c in ctx.clips_in(s.start, s.end) if c.kind == "media" and c.id != s.clip.id):
            return True
        return any(c.timeline_start < s.end and c.timeline_end > s.start for _t, c in ctx.clips() if c.kind in (KIND_TEXT, KIND_GRAPHIC)) or self._moves(s.clip)

    def _reading_hold(self, ctx: QCContext, s: _Shot) -> bool:
        """Evidence (a document, chart, screenshot) is held so it can be read."""
        for sc in ctx.scenes:
            if sc.end <= s.start or sc.start >= s.end:
                continue
            c = ctx.scene_ctx(sc.id)
            if c is None:
                continue
            if c.visual_type in ("DOCUMENT", "CHART", "SCREENSHOT", "DATA", "EVIDENCE") or is_evidence_visual(c, c.asset):
                return True
            brief = ctx.project.editing_strategy.briefs.get(sc.id) if ctx.project.editing_strategy else None
            if brief is not None and (brief.evidence_treatment_needed or brief.keep_static):
                return True
        a = ctx.asset(s.asset_id)
        return bool(a and getattr(a.source_type, "value", "") in EVIDENCE_SOURCES)

    def _uneven(self, ctx: QCContext, out: CheckerOutput, cfg, windows: list[_Window]) -> None:
        if len(windows) < 3:
            return
        med = max(MIN_MEDIAN_CPM, median(w.cpm for w in windows))
        limit = cfg.uneven_ratio * self.f
        flagged = [(w, w.cpm / med) for w in windows if w.cpm > limit * med and w.cpm - med >= 6 and not self._justified(ctx, w.start, w.end)]
        for r in self._runs(windows, flagged)[:MAX_ISSUES_PER_CODE]:
            avg = sum(w.cpm for w in r.windows) / len(r.windows)
            out.issues.append(self.issue(
                "pacing.uneven", QCCategory.PACING, Severity.NOTICE, "Uneven pacing", description=f"{avg:.0f} cuts per minute between {_t(r.start)} and {_t(r.end)}, {avg / med:.1f}x the video's typical {med:.0f} "
                f"(limit {limit:.1f}x) with nothing in the story that calls for it.", start=r.start, end=r.end, scene_id=self._scene_of(ctx, r.start), why="A sudden change of rhythm without a reason feels like a different video.",
                current=f"{avg:.0f} cuts/min vs median {med:.0f}", recommended=f"within {limit:.1f}x of the median", suggested_fix="Smooth the rhythm, or keep it if this part is meant to feel different.",
                confidence=70.0, viewer_impact=0.3, signature=sha(round(r.start / 10)), ctx=ctx))

    def _over_edited(self, ctx: QCContext, out: CheckerOutput, cfg, windows: list[_Window], wps: float) -> None:
        limit = cfg.over_edit_events_per_minute * self.f * min(1.3, max(0.8, wps / 2.5 if wps else 1.0))
        flagged = [(w, w.epm) for w in windows if w.epm > limit and not self._justified(ctx, w.start, w.end)]
        for r in self._runs(windows, flagged)[:MAX_ISSUES_PER_CODE]:
            avg = sum(w.epm for w in r.windows) / len(r.windows)
            out.issues.append(self.issue(
                "pacing.over_edited", QCCategory.PACING, Severity.WARNING, "Too many edits at once", description=f"{avg:.0f} edit events per minute (cuts, motion, transitions and graphics together) between "
                f"{_t(r.start)} and {_t(r.end)}; the limit for this narration speed is {limit:.0f}.", start=r.start, end=r.end, scene_id=self._scene_of(ctx, r.start),
                why="When everything moves, nothing stands out and the viewer cannot follow the words.", current=f"{avg:.0f} events/min", recommended=f"at most {limit:.0f} events/min",
                suggested_fix="Drop decorative effects or graphics here and keep the ones that carry meaning.", confidence=75.0, viewer_impact=0.5, signature=sha(round(r.start / 10)), ctx=ctx))

    def _under_emphasized(self, ctx: QCContext, out: CheckerOutput, shots: list[_Shot]) -> None:
        thr = ctx.settings.coverage.important_scene
        scenes = [s for s in ctx.scenes if s.duration >= 3.0]
        if len(scenes) < 3:
            return
        events = self._events(ctx)
        cut_times = [x.start for x in shots[1:]]
        emph = [(c.timeline_start, c.timeline_end) for _t, c in ctx.caption_clips() if (c.text or {}).get("emphasis")]

        def activity(s) -> float:
            n = sum(1 for t in cut_times + events if s.start <= t < s.end) + sum(1 for a, b in emph if a < s.end and b > s.start)
            return n / s.duration

        rates = {s.id: activity(s) for s in scenes}
        done = 0
        for s in scenes:
            if s.importance < thr or not (s.claims or s.numbers) or done >= MAX_ISSUES_PER_CODE:
                continue
            others = [rates[o.id] for o in scenes if o.id != s.id and o.importance < thr]
            if len(others) < 2:
                continue
            typical = median(others)
            if typical <= 0 or rates[s.id] > 0.35 * typical:
                continue
            done += 1
            out.issues.append(self.issue(
                "pacing.under_emphasized", QCCategory.PACING, Severity.NOTICE, "An important moment gets less attention than its neighbours",
                description=f"Scene {s.label} (importance {s.importance:.0%}, with {len(s.claims)} claim(s) and {len(s.numbers)} number(s)) has {rates[s.id] * 60:.0f} picture changes, motions or graphics per minute, "
                f"while the other scenes average {typical * 60:.0f}.", scene_id=s.id, start=s.start, end=s.end, why="Viewers should be guided to the statements that matter most.",
                current=f"{rates[s.id] * 60:.0f}/min", recommended=f"about {typical * 60:.0f}/min", suggested_fix="Add emphasis here: a number or text treatment, a zoom, or a visual change on the key statement.",
                fix=fx.navigate("open.scene", "Open the scene", scene_id=s.id), confidence=65.0, viewer_impact=0.4, signature=sha(s.id), ctx=ctx))

    # ------------------------------------------------------------------ findings: where the cuts land
    def _phrases(self, ctx: QCContext) -> list[tuple[float, float, list]]:
        """Runs of words that belong together: split at sentence / clause punctuation and at pauses."""
        out, cur = [], []
        for w in ctx.words:
            if cur and w.start - cur[-1].end > PHRASE_GAP:
                out.append((cur[0].start, cur[-1].end, cur))
                cur = []
            cur.append(w)
            if w.text.rstrip().endswith((",", ";", ":", ".", "!", "?")):
                out.append((cur[0].start, cur[-1].end, cur))
                cur = []
        if cur:
            out.append((cur[0].start, cur[-1].end, cur))
        return out

    def _awkward_cuts(self, ctx: QCContext, out: CheckerOutput, cfg, shots: list[_Shot]) -> None:
        phrases = self._phrases(ctx)
        tol = cfg.cut_in_phrase_ms / 1000.0
        done = 0
        for s in shots[1:]:
            if done >= MAX_ISSUES_PER_CODE:
                break
            t = s.start
            ph = next((p for p in phrases if p[0] + 0.1 < t < p[1] - 0.1 and len(p[2]) >= 4), None)
            reason, sev, importance = "", Severity.NOTICE, 0.0
            if ph is not None:
                mid = (ph[0] + ph[1]) / 2
                prev_w = max((w for w in ph[2] if w.end <= t + 0.05), key=lambda w: w.end, default=None)
                next_w = min((w for w in ph[2] if w.start >= t - 0.05), key=lambda w: w.start, default=None)
                if prev_w is not None and next_w is not None and NUMBER_RE.match(prev_w.text.strip(",.;:").lower()) and next_w.text.strip(",.;:").lower() in UNITS:
                    reason, sev = f"it separates “{prev_w.text}” from its unit “{next_w.text}”", Severity.WARNING
                elif abs(t - mid) <= tol:
                    reason = f"it lands {abs(t - mid) * 1000:.0f} ms from the middle of the phrase “{' '.join(w.text for w in ph[2][:8])}”"
                if reason:
                    sc = ctx.scene_at(t)
                    importance = sc.importance if sc else 0.0
                    if importance < ctx.settings.coverage.important_scene and sev is Severity.WARNING:
                        sev = Severity.NOTICE
            if not reason:
                continue
            d = ctx.project.editing_decisions.get(s.clip.ai_decision_id)
            if d is not None and d.confidence >= 85 and d.reason:
                continue  # a deliberate, explained cut
            done += 1
            out.issues.append(self.issue(
                "cut.awkward", QCCategory.CUT_TIMING, sev, "Cut lands in the middle of a phrase", description=f"The cut at {_t(t)} interrupts a spoken phrase: {reason}.", start=t - 0.3, end=t + 0.3,
                scene_id=s.scene_id, clip=s.clip, track=s.track, why="Cuts feel natural on sentence and clause boundaries, pauses and emphasis changes.", current=_t(t),
                recommended="a nearby phrase boundary", suggested_fix="Move the cut to the nearest clause boundary or pause.", confidence=65.0, viewer_impact=0.35 if sev is Severity.NOTICE else 0.5,
                signature=sha(s.clip.id, round(t, 1)), ctx=ctx))

    def _unnecessary_cuts(self, ctx: QCContext, out: CheckerOutput, shots: list[_Shot]) -> None:
        """A jump inside one sentence between two parts of the same footage, with nothing else changing: a cut for the sake of a cut."""
        sents = ctx.sentences
        done = 0
        for a, b in zip(shots, shots[1:]):
            if a.asset_id != b.asset_id or done >= MAX_ISSUES_PER_CODE or abs(a.end - b.start) > ctx.frame * 1.5:
                continue
            t = b.start
            inside = next((x for x in sents if x.start + 0.3 < t < x.end - 0.3), None)
            if inside is None or abs(a.clip.source_out - b.clip.source_in) <= CONTIGUOUS:
                continue
            if self._moves(b.clip) or (b.clip.transition and str(b.clip.transition.get("type", "CUT")).upper() != "CUT") or any(c.timeline_start >= t - 0.5 and c.timeline_start <= t + 0.5 for _t, c in ctx.clips() if c.kind in (KIND_TEXT, KIND_GRAPHIC)):
                continue
            done += 1
            jump = abs(b.clip.source_in - a.clip.source_out)
            out.issues.append(self.issue(
                "cut.unnecessary", QCCategory.CUT_TIMING, Severity.NOTICE, "Cut inside a sentence with no change of picture",
                description=f"At {_t(t)} the edit jumps {jump:.1f} s forward in the same footage in the middle of a sentence, and nothing else changes.", start=t - 0.3, end=t + 0.3, scene_id=b.scene_id,
                clip=b.clip, track=b.track, why="A jump cut with no new information looks like a mistake and breaks the flow of the explanation.", current=f"jump of {jump:.1f} s",
                recommended="one continuous shot, or a different picture", suggested_fix="Join the two parts, or put a different visual or a movement on the second part.", confidence=65.0,
                viewer_impact=0.3, signature=sha(b.clip.id), ctx=ctx))

    def _micro_cuts(self, ctx: QCContext, out: CheckerOutput, cfg, shots: list[_Shot]) -> None:
        """Bursts of tiny shots. Bursts inside one scene belong to scene.fragmentation.short_shots; this reports the ones that cross scene boundaries."""
        tiny = [s for s in shots if s.duration < cfg.micro_cut_seconds]
        done, i = 0, 0
        while i < len(tiny) and done < MAX_ISSUES_PER_CODE:
            group = [tiny[i]]
            for nxt in tiny[i + 1:]:
                if nxt.start - group[0].start <= 4.0:
                    group.append(nxt)
                else:
                    break
            if len(group) >= cfg.micro_cut_cluster:
                i += len(group)
                if len({g.scene_id for g in group}) < 2:
                    continue
                a, b = group[0].start, group[-1].end
                if self._deliberate(ctx, group):
                    continue
                done += 1
                out.issues.append(self.issue(
                    "cut.micro_cuts", QCCategory.CUT_TIMING, Severity.WARNING, "Burst of very short shots", description=f"{len(group)} shots shorter than {cfg.micro_cut_seconds:.1f} s within "
                    f"{b - a:.1f} s ({_t(a)}-{_t(b)}), across several scenes.", start=a, end=b, scene_id=group[0].scene_id, clip=group[0].clip, track=group[0].track,
                    why="Shots this short cannot be read; the picture just flickers.", current=f"{len(group)} shots < {cfg.micro_cut_seconds:.1f} s", recommended="shots long enough to register",
                    suggested_fix="Merge or lengthen the shots, or remove filler pictures; keep it only if this is a deliberate montage.", confidence=80.0, viewer_impact=0.45,
                    signature=sha(round(a, 0), len(group)), ctx=ctx))
            else:
                i += 1

    def _deliberate(self, ctx: QCContext, group: list[_Shot]) -> bool:
        for g in group:
            d = ctx.project.editing_decisions.get(g.clip.ai_decision_id)
            if d is not None and d.confidence >= 85 and d.reason and "montage" in (d.reason + str(d.parameters)).lower():
                return True
        return False

    def _before_content(self, ctx: QCContext, out: CheckerOutput, cuts: list[float]) -> None:
        """An overlay that carries information (text, number, evidence graphic) and is replaced before it can be read."""
        done = 0
        for _track, c in ctx.clips():
            if c.kind not in (KIND_TEXT, KIND_GRAPHIC) or done >= MAX_ISSUES_PER_CODE:
                continue
            if c.duration >= MIN_READ_SECONDS:
                continue
            cut = next((x for x in cuts if abs(x - c.timeline_end) <= 0.1), None)
            if cut is None:
                continue
            done += 1
            out.issues.append(self.issue(
                "cut.before_content", QCCategory.CUT_TIMING, Severity.NOTICE, "Information is cut away before it can be read", description=f"A {c.kind} element is on screen for {c.duration:.1f} s and "
                f"disappears with the cut at {_t(cut)} (about {MIN_READ_SECONDS:.1f} s are needed to read it).", clip=c, track=ctx.timeline.get_track(c.track_id), why="Viewers cannot take in what vanishes right as it becomes useful.",
                current=f"{c.duration:.1f} s", recommended=f"at least {MIN_READ_SECONDS:.1f} s", suggested_fix="Let the element stay through the cut, or delay the cut until it has been read.",
                confidence=70.0, viewer_impact=0.35, signature=sha(c.id), ctx=ctx))

    # ------------------------------------------------------------------ reference style (soft preference)
    def _style(self, ctx: QCContext, out: CheckerOutput, metrics: dict, wps: float) -> None:
        target = target_score(ctx, "pacing")
        if target is None:
            return
        measured = scale(metrics["cuts_per_minute"], PACING_POINTS)
        delta, cls = measured - target, speed_class(wps)
        conflict, explained = "", ""
        if any(i.code == "cut.micro_cuts" for i in out.issues) and delta >= -ctx.settings.style.tolerance:
            conflict = "The style asks for a pace at which some shots are too short to read."
        elif (cls == "SLOW" and delta < 0) or (cls == "FAST" and delta > 0):
            explained = "the narration is " + cls.lower() + " and the pictures follow it"
        iss = style_issue(self, ctx, "pacing", measured, explained_by=explained, conflict=conflict)
        if iss is not None:
            out.issues.append(iss)


def _t(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"

