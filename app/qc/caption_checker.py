"""CaptionChecker: can the viewer read, in time, every caption that is on screen?

What it adds to what already exists: ``PresentationValidator`` checks that a caption is structurally valid when a candidate edit is applied; the sync checker measures
caption *drift* against the voice (never reported here); the timeline checker owns same-track overlaps. This checker answers the viewer's questions: is the text fully
on screen and inside the safe margins, does it stay long enough to be read, is it legible against what is behind it, is it too dense, does it collide with an
evidence box or a text graphic, does the style stay consistent from caption to caption.

Everything is scene-local: a caption belongs to the scene its midpoint lies in, and only that scene's captions (plus the clips within 0.5 s of it, which is what
``QCContext.scene_signature`` hashes) are read, so a result can be reused per scene. Style findings that are the same for every caption of a style (a small font, a
weak contrast) are reported once per scene and style, on the first affected caption, with the number of captions concerned.

Contrast is judged against an *assumed* backdrop (mid grey) unless the style has its own background box, because QC does not look at pixels: the confidence says so.
Geometry comes from ``qc.geometry`` (the renderer's placement rules, the caption engine's glyph width): estimates, with a tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.captions.engine import layout_for
from app.presentation.animation import normalize as normalize_animation
from app.qc import fix_catalog
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext
from app.qc.geometry import (FRAME, MARGIN_EPS, CaptionLayout, Margins, Rect, assumed_backdrop, caption_layout, caption_lines, clip_rect, fit_inside, parse_color, text_contrast)
from app.qc.issue_model import QCCategory, QCFixSpec, QCIssue
from app.qc.severity import RANK as ORDER_RANK
from app.qc.severity import Severity
from app.timeline.clip import KIND_GRAPHIC, Clip
from app.timeline.track import Track

CAT = QCCategory.CAPTION
MOTION_PRESETS = ("slide_up", "slide_down", "slide_left", "slide_right", "scale_in", "scale_out", "pop", "type_on", "reveal", "counter")  # not allowed in reduced-motion mode
MOTION_EMPHASIS = ("POP", "SCALE")
MIN_CHARS_FOR_SPEED = 6  # a caption this short is judged by ``too_short``, never by reading speed
SIDES = ("left", "top", "right", "bottom")
SHADOW_WEIGHT = 0.6  # how much a drop shadow darkens the picture around a glyph, for the contrast estimate
OUTLINE_WEIGHT = 0.85
THIN_OUTLINE = 0.03  # an outline thinner than this share of the font size is just a shadow


def _short(text: str, n: int = 44) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _opt(ctx: QCContext, name: str, default: float) -> float:
    """An optional threshold of the caption section (the contract may not define it): the default applies until it does."""
    return float(getattr(ctx.settings.caption, name, default))


@dataclass
class _Cfg:
    """Every threshold, read once per run from the QC settings."""

    max_chars: int
    max_lines: int
    max_words: int
    max_cps: float
    min_duration: float
    flicker_seconds: float
    flicker_count: int
    min_margin: float
    min_font: float
    min_contrast: float
    max_anim: float
    collision_seconds: float
    collision_overlap: float
    word_overlap: float
    word_window: float
    past_voice: float
    max_highlights: int
    neighbour_gap: float
    flicker_gap: float

    @classmethod
    def of(cls, ctx: QCContext) -> "_Cfg":
        c = ctx.settings.caption
        return cls(int(c.max_chars_per_line), int(c.max_lines), int(c.max_words_displayed), float(c.max_cps), float(c.min_duration), float(c.flicker_seconds), int(c.flicker_count),
                   float(c.min_safe_margin), float(c.min_relative_font), float(c.min_contrast), float(c.max_animation_seconds),
                   _opt(ctx, "collision_seconds", 0.3), _opt(ctx, "collision_overlap", 0.15), _opt(ctx, "word_overlap_seconds", 0.1), _opt(ctx, "word_window_seconds", 0.35),
                   _opt(ctx, "past_voice_seconds", 1.0), int(_opt(ctx, "max_highlighted_words", 3)), _opt(ctx, "neighbour_gap_seconds", 4.0),
                   _opt(ctx, "flicker_gap_seconds", float(c.flicker_seconds)))


@dataclass
class _Cap:
    track: Track
    clip: Clip
    data: dict[str, Any]
    scene_id: str | None
    layout: CaptionLayout | None
    text: str
    chars: int
    words: int

    @property
    def start(self) -> float:
        return self.clip.timeline_start

    @property
    def end(self) -> float:
        return self.clip.timeline_end

    @property
    def duration(self) -> float:
        return self.clip.duration

    @property
    def label(self) -> str:
        return f"“{_short(self.text)}”"


@dataclass
class _Other:
    """Something else on screen that a caption can collide with."""

    track: Track
    clip: Clip
    rect: Rect
    what: str  # "evidence highlight" | "text graphic" | "caption"
    exact: bool  # a highlight region is exact; text and caption rectangles are estimates


# ---------------------------------------------------------------------------------------------- the checker
class CaptionChecker(BaseChecker):
    id = "caption"
    label = "Captions"
    categories = (QCCategory.CAPTION,)
    domains = ("timeline", "scenes", "captions", "transcript")  # scenes: which scene a caption belongs to; transcript: how fast the narrator speaks under a fast caption
    settings_sections = ("caption", "style")
    scene_local = True
    version = "1"

    # ------------------------------------------------------------------ run
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cs = ctx.project.caption_settings
        if not cs.enabled:
            out.notes.append("Captions are switched off in the project: caption checks skipped")
            return out
        cfg = _Cfg.of(ctx)
        caps = self._captions(ctx)
        others = self._others(ctx, caps)
        scenes = ctx.target_scenes()
        out.metrics.update({"captions_total": len(caps)})
        checked = 0
        all_cps: list[float] = []
        for n, scene in enumerate(scenes):
            ctx.check_cancel()
            report(n / max(1, len(scenes)), f"Scene {scene.label}")
            own = [c for c in caps if c.scene_id == scene.id]
            window = [c for c in caps if c.end > scene.start - 0.5 and c.start < scene.end + 0.5]
            near = [o for o in others if o.clip.timeline_end > scene.start - 0.5 and o.clip.timeline_start < scene.end + 0.5]  # what the scene signature covers
            cps = [c.chars / max(c.duration, 0.25) for c in own if c.chars]
            all_cps += cps
            checked += len(own)
            out.metrics[f"{scene.id}.captions_checked"] = len(own)
            out.metrics[f"{scene.id}.max_cps"] = round(max(cps, default=0.0), 1)
            out.metrics[f"{scene.id}.mean_cps"] = round(sum(cps) / len(cps), 1) if cps else 0.0
            if not own:
                continue
            flicker_ids = self._flicker(ctx, cfg, scene.id, own, out.issues)
            for cap in own:
                self._words(ctx, cfg, cap, out.issues)
                self._range(ctx, cfg, cap, out.issues)
                self._crowding(ctx, cfg, cap, out.issues)
                self._timing(ctx, cfg, cap, window, scene.end, flicker_ids, out.issues)
                self._placement(ctx, cfg, cap, out.issues)
                self._collision(ctx, cfg, cap, near, out.issues)
            self._style(ctx, cfg, own, out.issues)
            self._inconsistency(ctx, cfg, own, window, out.issues)
        out.metrics["captions_checked"] = checked
        out.metrics["max_cps"] = round(max(all_cps, default=0.0), 1)
        out.metrics["mean_cps"] = round(sum(all_cps) / len(all_cps), 1) if all_cps else 0.0
        out.notes.append(f"{checked} caption(s) checked in {len(scenes)} scene(s)")
        report(1.0, "")
        return out

    # ------------------------------------------------------------------ gathering
    def _host_scene(self, ctx: QCContext, clip: Clip) -> str | None:
        """The scene a clip is analysed in: the one its midpoint lies in (the nearest one for a clip that ends beyond the last scene), because only that is covered by the scene signature."""
        scenes = ctx.scenes
        if not scenes:
            return None
        mid = clip.timeline_start + clip.duration / 2
        s = ctx.scene_at(mid)
        if s is None:
            s = min(scenes, key=lambda sc: max(sc.start - mid, mid - sc.end, 0.0))
        return s.id

    def _captions(self, ctx: QCContext) -> list[_Cap]:
        cs, styles, canvas = ctx.project.caption_settings, ctx.project.caption_styles, ctx.canvas
        out: list[_Cap] = []
        for track, clip in ctx.caption_clips():
            if track.hidden or clip.duration <= 0 or not isinstance(clip.text, dict):
                continue
            text = " ".join(str(clip.text.get("text", "")).split())
            if not text:
                continue  # an empty caption is the timeline checker's finding
            layout = caption_layout(clip.text, cs, styles, canvas)
            out.append(_Cap(track, clip, clip.text, self._host_scene(ctx, clip), layout, text, len(text), len(text.split())))
        out.sort(key=lambda c: (c.start, c.clip.id))
        return out

    def _others(self, ctx: QCContext, caps: list[_Cap]) -> list[_Other]:
        cs, styles, canvas = ctx.project.caption_settings, ctx.project.caption_styles, ctx.canvas
        out: list[_Other] = []
        for track, clip in ctx.graphic_clips() + ctx.text_clips():
            if track.hidden or clip.duration <= 0 or clip.opacity < 0.2:
                continue
            rect = clip_rect(clip, cs, styles, canvas)
            if rect is not None:
                out.append(_Other(track, clip, rect, "evidence highlight" if clip.kind == KIND_GRAPHIC else "text graphic", clip.kind == KIND_GRAPHIC))
        for c in caps:  # captions on a second caption track (the same track is the timeline checker's overlap rule)
            if c.layout is not None:
                out.append(_Other(c.track, c.clip, c.layout.box, "caption", False))
        return out

    # ------------------------------------------------------------------ one issue
    def _issue(self, ctx: QCContext, cap: _Cap, code: str, severity: Severity, title: str, **kw: Any) -> QCIssue:
        return self.issue(code, CAT, severity, title, scene_id=cap.scene_id, clip=cap.clip, track=cap.track, ctx=ctx, affected=kw.pop("affected", [f"Caption {cap.label}"]), **kw)

    # ------------------------------------------------------------------ word timing integrity
    def _words(self, ctx: QCContext, cfg: _Cfg, cap: _Cap, issues: list[QCIssue]) -> None:
        words = cap.data.get("words") or []
        if not words:
            return
        bad = unordered = overlapping = 0
        valid: list[tuple[float, float]] = []
        prev_start = prev_end = None
        for w in words:
            try:
                s, e = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                bad += 1
                continue
            if not (math.isfinite(s) and math.isfinite(e)) or e < s - 1e-6:
                bad += 1
                continue
            if prev_start is not None and s < prev_start - 1e-6:
                unordered += 1
            elif prev_end is not None and s < prev_end - cfg.word_overlap:
                overlapping += 1
            prev_start, prev_end = s, max(e, prev_end if prev_end is not None else e)
            valid.append((s, e))
        # the window may end before its own words do: the viewer loses the end of the caption. (Where a caption *starts* against the voice is narration drift: the sync
        # checker's finding, so a caption that is merely late or early is not reported here.)
        cut = max([v[1] - cap.end for v in valid if v[0] < cap.end] + [0.0])
        problems = []
        if bad:
            problems.append(f"{bad} word(s) have no usable start / end time")
        if unordered:
            problems.append(f"{unordered} word(s) are out of order")
        if overlapping:
            problems.append(f"{overlapping} word(s) overlap the previous word by more than {cfg.word_overlap * 1000:.0f} ms")
        if cut > cfg.word_window:
            problems.append(f"the caption leaves the screen at {cap.end:.2f} s, {cut * 1000:.0f} ms before its last word ends")
        if not problems:
            return
        severity = Severity.ERROR if (bad or unordered) else Severity.WARNING
        issues.append(self._issue(
            ctx, cap, "caption.words", severity, "Caption word timing is inconsistent",
            description=f"In the caption {cap.label}: " + "; ".join(problems) + ".",
            why="Highlighting, word-by-word reveal and any later retiming read these word times, so a caption with inconsistent word timing cannot follow the voice reliably.",
            current="; ".join(problems), recommended="ordered, non-overlapping words, all spoken before the caption leaves the screen",
            suggested_fix="Regenerate the captions of this scene (or retime the caption) so its words follow the transcript again.", viewer_impact=0.5 if severity is Severity.ERROR else 0.35,
            signature=f"{bad}/{unordered}/{overlapping}/{round(cut, 1)}", metrics={"bad": bad, "unordered": unordered, "overlapping": overlapping, "cut_seconds": round(cut, 3)}))

    def _range(self, ctx: QCContext, cfg: _Cfg, cap: _Cap, issues: list[QCIssue]) -> None:
        voice = ctx.voice_duration
        if not voice or cap.end <= voice + cfg.past_voice:
            return
        late = cap.end - voice
        issues.append(self._issue(
            ctx, cap, "caption.range", Severity.ERROR if cap.start >= voice else Severity.WARNING, "Caption runs past the end of the voice-over",
            description=f"The caption {cap.label} ends at {cap.end:.2f} s, {late:.2f} s after the voice-over ends ({voice:.2f} s).",
            why="Text that outlives the narration hangs on screen over nothing and can delay the end of the video.",
            current=f"ends at {cap.end:.2f} s", recommended=f"ends by {voice + cfg.past_voice:.2f} s at the latest", suggested_fix="Shorten or delete the caption at the end of the timeline.",
            fix=fix_catalog.navigate("open.timeline", "Show the caption on the timeline", clip_id=cap.clip.id), viewer_impact=0.5, signature=f"{round(late, 1)}",
            metrics={"seconds_past_voice": round(late, 3)}))

    # ------------------------------------------------------------------ density: words, line length, line count
    def _crowding(self, ctx: QCContext, cfg: _Cfg, cap: _Cap, issues: list[QCIssue]) -> None:
        cs = ctx.project.caption_settings
        lines = caption_lines(cap.data)
        longest = max((len(x) for x in lines), default=0)
        found: list[tuple[str, float, Severity]] = []  # (what, how many times the limit, its severity)
        if cap.words > cfg.max_words:
            r = cap.words / cfg.max_words
            found.append((f"{cap.words} words at once (limit {cfg.max_words})", r, Severity.ERROR if r >= 1.6 else Severity.WARNING if r >= 1.15 else Severity.NOTICE))
        if longest > cfg.max_chars:
            r = longest / cfg.max_chars
            # a long line alone is a reading guideline, not a defect: the caption engine itself fills a line up to the frame width at the style's size (about 50 characters at
            # the default one), so a line the style's own layout fits is information only, and only a line beyond it is a finding (PresentationValidator treats 1.2 times that as
            # an error)
            cap_chars = layout_for(cs, cap.layout.style, ctx.canvas).chars_per_line if cap.layout else cfg.max_chars
            found.append((f"a line of {longest} characters (limit {cfg.max_chars})", r, Severity.ERROR if longest >= cap_chars * 1.75 else Severity.WARNING if longest >= cap_chars * 1.3 else Severity.NOTICE if longest > cap_chars else Severity.INFO))  # information while the style's own layout fits it
        if len(lines) > cfg.max_lines:
            r = len(lines) / cfg.max_lines
            found.append((f"{len(lines)} lines (limit {cfg.max_lines})", r, Severity.ERROR if r >= 2.0 else Severity.WARNING))
        if not found:
            return
        worst = max(r for _w, r, _s in found)
        severity = min((sev for _w, _r, sev in found), key=lambda v: ORDER_RANK[v])
        if sum(1 for _w, _r, sev in found if sev is not Severity.INFO) > 1 and severity is Severity.NOTICE:
            severity = Severity.WARNING  # several limits exceeded at once
        fix: QCFixSpec | None = None
        blocked = ""
        if cap.words > cfg.max_words and cs.max_words > cfg.max_words:  # the project's own setting allows captions this dense: lowering it helps future captions
            if "max_words" in cs.user_set:
                blocked = "You chose the maximum number of caption words yourself: QC does not change it"
            else:
                fix = fix_catalog.caption_restyle("max_words", cfg.max_words, f"Limit captions to {cfg.max_words} words (applies to captions created from now on)", ctx.settings)
        line_only = all(w.startswith("a line of") for w, _r, _s in found)
        issues.append(self._issue(
            ctx, cap, "caption.overcrowding", severity, "Caption has a long line" if line_only else "Caption is crowded",
            description=f"The caption {cap.label} shows " + "; ".join(w for w, _r, _s in found) + ".",
            why="A long, dense caption takes more time to read than the voice gives it and covers more of the picture.",
            current="; ".join(w for w, _r, _s in found), recommended=f"at most {cfg.max_words} words, {cfg.max_lines} lines of {cfg.max_chars} characters",
            suggested_fix="Split the caption into two or shorten the lines.", fix=fix, fix_blocked=blocked, viewer_impact=min(1.0, 0.35 + 0.3 * (worst - 1.0)),
            signature=f"{cap.words}/{longest}/{len(lines)}", metrics={"words": cap.words, "longest_line": longest, "lines": len(lines)}))

    # ------------------------------------------------------------------ reading time
    def _timing(self, ctx: QCContext, cfg: _Cfg, cap: _Cap, window: list[_Cap], scene_end: float, flicker_ids: set[str], issues: list[QCIssue]) -> None:
        dur = cap.duration
        cps = cap.chars / max(dur, 0.25)
        if cap.chars >= MIN_CHARS_FOR_SPEED and cps > cfg.max_cps:
            ratio = cps / cfg.max_cps
            severity = Severity.ERROR if ratio >= 1.5 else Severity.WARNING if ratio >= 1.2 else Severity.NOTICE
            need = cap.chars / cfg.max_cps
            nxt = next((c.start for c in window if c.start >= cap.end - 1e-6 and c.clip.id != cap.clip.id), None)
            room = (nxt if nxt is not None else scene_end) - cap.end
            heard_to = min(cap.end, scene_end)  # the scene's own narration: what the scene signature covers
            spoken = ctx.words_between(cap.start, heard_to)
            wps = len(spoken) / max(heard_to - cap.start, 0.25)
            if room >= need - dur - 1e-6:
                advice = f"Keep the caption on screen for about {need:.1f} s: there is {room:.1f} s free after it."
            else:
                advice = "Regroup the words over two captions or shorten the text: there is no room to hold the caption longer."
            speech = f" The narrator speaks about {wps:.1f} words per second here." if len(spoken) >= 3 else ""
            cs = ctx.project.caption_settings
            fix: QCFixSpec | None = None
            blocked = ""
            if cs.reading_speed * 17.0 > cfg.max_cps and cs.reading_speed > 0.5:  # the engine's own limit (17 cps x reading speed) is looser than QC's: tighten it for future captions
                if "reading_speed" in cs.user_set:
                    blocked = "You chose the caption reading speed yourself: QC does not change it"
                else:
                    fix = fix_catalog.caption_restyle("reading_speed", round(max(0.5, cfg.max_cps / 17.0 * 0.9), 2), "Slow the caption reading speed (applies to captions created from now on)", ctx.settings)
            issues.append(self._issue(
                ctx, cap, "caption.too_fast", severity, "Caption may be too fast to read",
                description=f"The caption {cap.label} shows {cap.chars} characters in {dur:.2f} s: {cps:.1f} characters per second (limit {cfg.max_cps:g}).{speech}",
                why="Viewers who cannot finish reading before the caption changes stop reading, or stop listening.",
                current=f"{cps:.1f} characters/s", recommended=f"at most {cfg.max_cps:g} characters/s (at least {need:.1f} s for this text)", suggested_fix=advice, fix=fix, fix_blocked=blocked,
                viewer_impact=min(1.0, 0.45 + 0.4 * (ratio - 1.0)), signature=f"{round(cps)}", metrics={"cps": round(cps, 2), "seconds_needed": round(need, 2), "room_after": round(room, 2)}))
            return
        if dur < cfg.min_duration - 1e-9 and cap.clip.id not in flicker_ids:
            issues.append(self._issue(
                ctx, cap, "caption.too_short", Severity.WARNING if dur < cfg.min_duration * 0.65 else Severity.NOTICE, "Caption is on screen for a very short time",
                description=f"The caption {cap.label} is on screen for {dur:.2f} s (minimum {cfg.min_duration:g} s).",
                why="A caption that disappears before the eye lands on it reads as a flash rather than as text.",
                current=f"{dur:.2f} s", recommended=f"at least {cfg.min_duration:g} s", suggested_fix="Hold the caption longer or merge it with the neighbouring caption.",
                viewer_impact=0.3, signature=f"{round(dur, 1)}", metrics={"duration": round(dur, 3)}))

    def _flicker(self, ctx: QCContext, cfg: _Cfg, scene_id: str, own: list[_Cap], issues: list[QCIssue]) -> set[str]:
        """Runs of consecutive very short captions (each below the flicker length, separated by at most the flicker gap). Returns the ids they cover."""
        runs: list[list[_Cap]] = []
        cur: list[_Cap] = []
        for c in own:
            if c.duration < cfg.flicker_seconds - 1e-9 and (not cur or c.start - cur[-1].end <= cfg.flicker_gap + 1e-6):
                cur.append(c)
                continue
            if len(cur) >= cfg.flicker_count:
                runs.append(cur)
            cur = [c] if c.duration < cfg.flicker_seconds - 1e-9 else []
        if len(cur) >= cfg.flicker_count:
            runs.append(cur)
        covered: set[str] = set()
        for run in runs:
            covered |= {c.clip.id for c in run}
            first = run[0]
            issues.append(self.issue(
                "caption.flicker", CAT, Severity.WARNING, "Captions flicker", description=f"{len(run)} consecutive captions are each on screen for less than {cfg.flicker_seconds:g} s "
                f"({first.start:.2f}-{run[-1].end:.2f} s).", scene_id=scene_id, clip=first.clip, track=first.track, start=first.start, end=run[-1].end,
                why="A burst of captions that each vanish almost at once looks like flicker and cannot be read.", current=f"{len(run)} captions under {cfg.flicker_seconds:g} s",
                recommended=f"captions of at least {cfg.min_duration:g} s, grouped into phrases", suggested_fix="Merge the short captions into phrases and hold each long enough to read.",
                affected=[f"Caption {c.label}" for c in run[:6]], viewer_impact=min(1.0, 0.5 + 0.08 * len(run)), signature=f"{first.clip.id}:{len(run)}",
                metrics={"captions": len(run)}, ctx=ctx))
        return covered

    # ------------------------------------------------------------------ screen position
    def _placement(self, ctx: QCContext, cfg: _Cfg, cap: _Cap, issues: list[QCIssue]) -> None:
        lay = cap.layout
        if lay is None:
            return
        safe = Margins.of(ctx.project.caption_settings, cfg.min_margin).rect
        off = lay.rect.overshoot(FRAME)
        over = lay.rect.overshoot(safe)
        clipped = max(off) > MARGIN_EPS
        if not clipped and max(over) <= MARGIN_EPS:
            return
        fit = fit_inside(lay.rect, safe)
        edges = [n for n, v in zip(SIDES, off if clipped else over) if v > MARGIN_EPS]
        depth = max(off) if clipped else max(over)
        fix = fix_catalog.caption_safe_margin(cap.clip.id, [fit[0], fit[1]], ctx.settings) if fit is not None else None
        sig = f"{'|'.join(edges)}:{round(depth, 2)}"
        m = {"overshoot": round(depth, 4), "edges": edges, "fits_safe_area": fit is not None}
        if clipped:
            issues.append(self._issue(
                ctx, cap, "caption.overflow", Severity.ERROR, "Caption is cut off by the edge of the frame",
                description=f"The caption {cap.label} reaches {depth * 100:.1f}% of the frame past the {'/'.join(edges)} edge, so part of the text is not visible.",
                why="Text that leaves the frame is cut off in the final video.", current=f"{depth * 100:.1f}% outside the frame", recommended="inside the safe margins",
                suggested_fix="Move the caption inside the safe area." if fit else "Shorten the caption or break it into more lines.", fix=fix, viewer_impact=0.9, signature=sig, metrics=m))
            return
        if fit is None:  # too large for the safe area wherever it is put
            issues.append(self._issue(
                ctx, cap, "caption.overflow", Severity.WARNING, "Caption is wider than the safe area",
                description=f"The caption {cap.label} needs about {lay.rect.width * 100:.0f}% x {lay.rect.height * 100:.0f}% of the frame; the safe area is {safe.width * 100:.0f}% x {safe.height * 100:.0f}%.",
                why="Text outside the safe margins can be cut off or look crowded against the edge on some screens.", current=f"{depth * 100:.1f}% inside the {'/'.join(edges)} margin",
                recommended="text that fits inside the safe margins", suggested_fix="Shorten the caption or break it into more lines.", viewer_impact=0.5, signature=sig, metrics=m))
            return
        issues.append(self._issue(
            ctx, cap, "caption.safe_margin", Severity.WARNING if depth >= 0.015 else Severity.NOTICE, "Caption is inside the safe margin",
            description=f"The caption {cap.label} reaches {depth * 100:.1f}% of the frame into the {'/'.join(edges)} safe margin.",
            why="Text close to the frame edge can be cut off or covered by the interface of some players.", current=f"{depth * 100:.1f}% inside the margin", recommended="inside the safe margins",
            suggested_fix="Move the caption inside the safe area.", fix=fix, viewer_impact=0.4, signature=sig, metrics=m))

    # ------------------------------------------------------------------ collisions with other things on screen
    def _collision(self, ctx: QCContext, cfg: _Cfg, cap: _Cap, others: list[_Other], issues: list[QCIssue]) -> None:
        lay = cap.layout
        if lay is None:
            return
        for o in others:
            if o.clip.id == cap.clip.id:
                continue
            if o.what == "caption" and (o.track.id == cap.track.id or (o.clip.timeline_start, o.clip.id) < (cap.start, cap.clip.id)):
                continue  # same track: the timeline checker; otherwise report each pair once, from the earlier caption
            seconds = min(cap.end, o.clip.timeline_end) - max(cap.start, o.clip.timeline_start)
            if seconds < cfg.collision_seconds - 1e-9:
                continue
            # over an evidence box what counts is how much of the box the caption hides; between two text elements, how much of the smaller one
            share = lay.box.share_of(o.rect) if o.exact else lay.box.overlap_ratio(o.rect)
            if share < cfg.collision_overlap:
                continue
            what = o.what if o.what != "text graphic" else f"text graphic “{_short(str((o.clip.text or {}).get('content', '')), 30)}”"
            issues.append(self._issue(
                ctx, cap, "caption.collision", Severity.WARNING, f"Caption overlaps a {o.what}",
                description=f"The caption {cap.label} overlaps the {what} for {seconds:.1f} s and covers {share * 100:.0f}% of {'it' if o.exact else 'the smaller of the two'}.",
                why="Two elements on the same part of the screen at once hide each other, so the viewer misses one of them.", current=f"{share * 100:.0f}% overlap for {seconds:.1f} s",
                recommended="captions and graphics in separate parts of the frame", suggested_fix="Move the caption or the graphic so they do not overlap.",
                fix=fix_catalog.navigate("open.timeline", "Show the caption on the timeline", clip_id=cap.clip.id), affected=[f"Caption {cap.label}", f"{o.what} {o.clip.id}"],
                confidence=90.0 if o.exact else 75.0, viewer_impact=min(1.0, 0.45 + share * 0.4), signature=f"{o.clip.id}:{round(share, 1)}",
                metrics={"overlap_ratio": round(share, 3), "overlap_seconds": round(seconds, 2), "other_clip": o.clip.id, "other_kind": o.what}))

    # ------------------------------------------------------------------ style: font size, contrast, highlights, animation
    @staticmethod
    def _backdrop(st: Any) -> tuple[tuple[float, float, float], str]:
        """What is behind the glyphs: the style's own box, else its outline / shadow halo, else an unknown picture."""
        box, outline, shadow = parse_color(st.background_color), parse_color(st.outline_color), parse_color(st.shadow_color)
        if st.background == "box" and box is not None:
            return assumed_backdrop(box=(box, float(st.background_opacity)))
        if float(st.outline_width) >= THIN_OUTLINE and outline is not None:
            return assumed_backdrop(halo=(outline, OUTLINE_WEIGHT))
        if st.shadow and shadow is not None:
            return assumed_backdrop(halo=(shadow, SHADOW_WEIGHT))
        return assumed_backdrop()

    def _style(self, ctx: QCContext, cfg: _Cfg, own: list[_Cap], issues: list[QCIssue]) -> None:
        cs = ctx.project.caption_settings
        groups: dict[tuple, list[_Cap]] = {}
        for c in own:
            if c.layout is None:
                continue
            st = c.layout.style
            key = (round(st.size_rel, 4), st.color, st.highlight_color, st.background, st.background_color, round(st.background_opacity, 2), bool(st.shadow), st.shadow_color,
                   round(float(st.outline_width), 3), st.outline_color, round(float(st.opacity), 2), str(c.data.get("highlight_mode", cs.highlight_mode)), bool(c.data.get("emphasis")))
            groups.setdefault(key, []).append(c)
        for key, caps in groups.items():
            first = caps[0]
            st = first.layout.style  # type: ignore[union-attr]
            n = len(caps)
            many = f" (and {n - 1} more caption(s) with the same style in this scene)" if n > 1 else ""
            # ---- font size
            if st.size_rel < cfg.min_font - 1e-9:
                fix, blocked = self._restyle_fix(ctx, "large_text", True, "Use large caption text (applies to captions created from now on)", cs.large_text is False and st.size_rel * 1.3 >= cfg.min_font)
                issues.append(self._issue(
                    ctx, first, "caption.small_font", Severity.ERROR if st.size_rel < cfg.min_font * 0.6 else Severity.WARNING, "Caption text is small",
                    description=f"The caption {first.label} is drawn at {st.size_rel * 100:.1f}% of the frame height{many}; the minimum is {cfg.min_font * 100:.1f}%.",
                    why="Small text is hard to read on a phone or from a distance.", current=f"{st.size_rel * 100:.1f}% of the frame height", recommended=f"at least {cfg.min_font * 100:.1f}%",
                    suggested_fix="Increase the caption size.", fix=fix, fix_blocked=blocked, affected=[f"Caption {c.label}" for c in caps[:5]], viewer_impact=0.6,
                    signature=f"{st.size_rel:.3f}", metrics={"relative_size": round(st.size_rel, 4), "captions": n}))
            # ---- contrast of the text, and of the highlight colour
            backdrop, known = self._backdrop(st)
            conf = {"box": 90.0, "halo": 65.0, "none": 50.0}[known]
            for what, colour, code, title in (("text", st.color, "caption.low_contrast", "Caption text has low contrast"),
                                              ("highlight", st.highlight_color, "caption.highlight", "Caption highlight colour is hard to read")):
                rgb = parse_color(colour)
                if rgb is None:
                    continue
                if what == "highlight" and not (key[11] in ("HIGHLIGHT", "PROGRESSIVE") or key[12]):
                    continue
                ratio = text_contrast(rgb, float(st.opacity), backdrop)
                if ratio >= cfg.min_contrast:
                    continue
                basis = {"box": "its background box", "halo": "its outline / shadow over an assumed mid-grey picture", "none": "an assumed mid-grey picture (the actual footage is not analysed)"}[known]
                fix, blocked = self._restyle_fix(ctx, "high_contrast", True, "Use high-contrast captions (applies to captions created from now on)", not cs.high_contrast)
                if what == "text":
                    severity = Severity.ERROR if (known == "box" and ratio < cfg.min_contrast / 2) else Severity.WARNING
                else:  # a highlight is a short accent: a notice unless the style's own box makes it certain
                    severity = Severity.WARNING if known == "box" else Severity.NOTICE
                issues.append(self._issue(
                    ctx, first, code, severity, title,
                    description=f"The {what} colour {colour} against {basis} has a contrast ratio of {ratio:.1f}:1{many}; the minimum is {cfg.min_contrast:g}:1.",
                    why="Low contrast makes words hard to pick out, especially in bright scenes and on small screens.", current=f"{ratio:.1f}:1", recommended=f"at least {cfg.min_contrast:g}:1",
                    suggested_fix="Use a darker box behind the text or a stronger outline.", fix=fix, fix_blocked=blocked, affected=[f"Caption {c.label}" for c in caps[:5]], confidence=conf,
                    viewer_impact=0.7 if what == "text" else 0.4, signature=f"{what}:{colour}:{known}", metrics={"contrast": round(ratio, 2), "backdrop": known, "captions": n}))
        self._highlight_count(ctx, cfg, own, issues)
        self._animation(ctx, cfg, own, issues)

    def _restyle_fix(self, ctx: QCContext, field_name: str, value: Any, summary: str, applicable: bool) -> tuple[QCFixSpec | None, str]:
        if not applicable:
            return None, ""
        if field_name in ctx.project.caption_settings.user_set:
            return None, "You chose this caption option yourself: QC does not change it"
        return fix_catalog.caption_restyle(field_name, value, summary, ctx.settings), ""

    def _highlight_count(self, ctx: QCContext, cfg: _Cfg, own: list[_Cap], issues: list[QCIssue]) -> None:
        for c in own:
            marks = c.data.get("emphasis") or []
            n = len(marks)
            if n <= cfg.max_highlights and not (n >= 3 and n / max(1, c.words) > 0.5):
                continue
            cs = ctx.project.caption_settings
            fix, blocked = self._restyle_fix(ctx, "keyword_highlight", False, "Switch keyword highlighting off (applies to captions created from now on)", cs.keyword_highlight)
            issues.append(self._issue(
                ctx, c, "caption.highlight", Severity.NOTICE, "Too many highlighted words in one caption",
                description=f"The caption {c.label} highlights {n} of its {c.words} words.", why="Emphasis only works when it is rare: when most words stand out, none of them does.",
                current=f"{n} highlighted words", recommended=f"at most {cfg.max_highlights}, and less than half of the caption", suggested_fix="Keep the highlight for the one or two words that matter.",
                fix=fix, fix_blocked=blocked, viewer_impact=0.25, signature=f"{n}/{c.words}", metrics={"highlighted": n, "words": c.words}))

    def _animation(self, ctx: QCContext, cfg: _Cfg, own: list[_Cap], issues: list[QCIssue]) -> None:
        cs = ctx.project.caption_settings
        groups: dict[tuple, list[_Cap]] = {}
        for c in own:
            an = normalize_animation(c.clip.animation)
            sides = []
            for side in ("in", "out"):
                s = an.get(side)
                if isinstance(s, dict):
                    try:
                        sides.append((side, str(s.get("preset", "")), round(float(s.get("duration", 0.0)) + float(s.get("delay", 0.0)), 3)))
                    except (TypeError, ValueError):
                        continue
            emph = tuple(sorted({str(m.get("style", "")) for m in (c.data.get("emphasis") or []) if isinstance(m, dict)}))
            groups.setdefault((tuple(sides), emph), []).append(c)
        for (sides, emph), caps in groups.items():
            first = caps[0]
            n = len(caps)
            many = f" ({n} captions in this scene)" if n > 1 else ""
            slow = [(side, preset, d) for side, preset, d in sides if d > cfg.max_anim + 1e-9]
            if slow:
                side, preset, d = max(slow, key=lambda x: x[2])
                issues.append(self._issue(
                    ctx, first, "caption.animation", Severity.NOTICE, "Caption animation is long",
                    description=f"The caption {first.label} uses a {d:.2f} s {side}-animation ({preset}){many}; captions should settle within {cfg.max_anim:g} s.",
                    why="A slow caption animation delays reading and draws attention away from the picture.", current=f"{d:.2f} s", recommended=f"at most {cfg.max_anim:g} s",
                    suggested_fix="Shorten the caption animation.", affected=[f"Caption {c.label}" for c in caps[:5]], viewer_impact=0.2, signature=f"{side}:{preset}:{round(d, 1)}",
                    metrics={"animation_seconds": d, "captions": n}))
            if cs.reduced_motion:
                moving = [f"{side} {preset}" for side, preset, _d in sides if preset in MOTION_PRESETS] + [f"{e} emphasis" for e in emph if e in MOTION_EMPHASIS]
                if moving:
                    fix, blocked = self._restyle_fix(ctx, "reduced_motion", True, "Keep reduced motion on for new captions", False)
                    issues.append(self._issue(
                        ctx, first, "caption.animation", Severity.WARNING, "Caption moves although reduced motion is on",
                        description=f"Reduced motion is switched on in the caption settings, but the caption {first.label} uses {', '.join(moving)}{many}.",
                        why="Reduced motion exists for viewers who are sensitive to movement: captions should only fade.", current=", ".join(moving), recommended="fades only",
                        suggested_fix="Use a fade for this caption.", fix=fix, fix_blocked=blocked, affected=[f"Caption {c.label}" for c in caps[:5]], viewer_impact=0.4,
                        signature="reduced:" + ",".join(moving), metrics={"captions": n}))

    # ------------------------------------------------------------------ consistency between neighbouring captions
    def _inconsistency(self, ctx: QCContext, cfg: _Cfg, own: list[_Cap], window: list[_Cap], issues: list[QCIssue]) -> None:
        """A caption whose style, size, position or highlight mode differs from BOTH its neighbours, which agree with each other, with no reason on record."""
        cs = ctx.project.caption_settings
        track_caps = sorted(window, key=lambda c: (c.start, c.clip.id))
        own_ids = {c.clip.id for c in own}

        def look(c: _Cap) -> dict[str, Any]:
            st = c.layout.style if c.layout else None
            pos = str(c.data.get("position", cs.position))
            xy = [round(float(v), 2) for v in (c.data.get("position_xy") or [])] if pos == "custom" else []
            return {"style": str(c.data.get("style_id", cs.style_id)), "size": round(st.size_rel, 3) if st else None, "position": pos, "anchor": xy,
                    "highlight mode": str(c.data.get("highlight_mode", cs.highlight_mode))}

        for i in range(1, len(track_caps) - 1):
            c, a, b = track_caps[i], track_caps[i - 1], track_caps[i + 1]
            if c.clip.id not in own_ids or c.start - a.end > cfg.neighbour_gap or b.start - c.end > cfg.neighbour_gap:
                continue
            la, lc, lb = look(a), look(c), look(b)
            if la != lb:
                continue
            diff = [k for k in lc if lc[k] != la[k]]
            if not diff:
                continue
            by_user = str(c.clip.created_by).upper() == "USER"
            issues.append(self._issue(
                ctx, c, "caption.inconsistency", Severity.NOTICE, "Caption looks different from its neighbours",
                description=f"The caption {c.label} differs in {', '.join(diff)} from the captions before and after it, which match each other"
                + ("; it was edited by you, so this may be intentional." if by_user else "."),
                why="A caption that changes look for no visible reason makes the captions feel unfinished.", current=", ".join(f"{k}: {lc[k]}" for k in diff),
                recommended=", ".join(f"{k}: {la[k]}" for k in diff), suggested_fix="Use the same caption style, size and position as the captions around it, unless the change is deliberate.",
                confidence=60.0 if by_user else 70.0, viewer_impact=0.2, signature="|".join(f"{k}={lc[k]}" for k in diff), metrics={"differs": diff}))


def caption_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Project totals from the per-scene caption metrics (a scene-level re-run only refreshes its own scenes' keys)."""
    n, worst, total, k = 0, 0.0, 0.0, 0
    for key, v in metrics.items():
        if key.endswith(".captions_checked"):
            sid = key[: -len(".captions_checked")]
            c = int(v)
            n += c
            total += c * float(metrics.get(f"{sid}.mean_cps", 0.0))
            k += c
            worst = max(worst, float(metrics.get(f"{sid}.max_cps", 0.0)))
    return {"captions_checked": n, "max_cps": round(worst, 1), "mean_cps": round(total / k, 1) if k else 0.0}
