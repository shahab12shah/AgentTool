"""Motion quality (spec 17): zoom, pan, rotation and keyframes read from the real keyframes of each picture clip.

Movement should support the story, not distract from it. For every clip with keyframed motion the checker measures the rate of each segment (scale change per second,
rotation degrees per second, jumps between neighbouring frames), the total range, and what is on screen while it moves (captions, text, numbers). The example from the
spec - scale 1.00 -> 1.35 in 0.4 s, which is 0.875 per second - is reported as "Potentially excessive zoom speed" with a gentler alternative. A recommended replacement
(``motion.soften``) always needs the user's confirmation; invalid values (scale <= 0, opacity outside 0..1) are the timeline checker's job and are skipped here.

The checker is scene-local: a clip belongs to the scene it is assigned to (or mostly overlaps) and the answer depends only on the clip, its keyframes and the text on
screen with it. The project-wide comparison with a reference style's motion level lives in the pacing checker, which is not scene-local.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory
from app.qc.media_facts import extras
from app.qc.severity import Severity
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT, Clip
from app.timeline.keyframes import Keyframe, value_at
from app.timeline.track import Track

SOFT_FACTOR = 0.7  # a recommended movement is this fraction of the allowed rate
MOTION_PROPS = ("scale", "position_x", "position_y", "rotation")
MIN_MOTION_CLIP = 1.0  # s: movement on a shorter clip is not seen as movement
MIN_ZOOM_DELTA = 0.03  # a scale change smaller than this is not a zoom at all
EVIDENCE_TYPES = ("DOCUMENT", "CHART", "SCREENSHOT", "DATA", "EVIDENCE")
OVERLAP_MIN = 0.5  # s of overlap with text before motion can hurt reading
REVERSAL_SPAN = 1.5  # s: in -> out -> in within this is "pumping"
MAX_ISSUES_PER_CODE_PER_SCENE = 3


@dataclass
class _Seg:
    prop: str
    t0: float
    t1: float
    v0: float
    v1: float
    interp: str

    @property
    def span(self) -> float:
        return max(1e-6, self.t1 - self.t0)

    @property
    def rate(self) -> float:
        return abs(self.v1 - self.v0) / self.span


class MotionChecker(BaseChecker):
    id = "motion"
    label = "Motion"
    categories = (QCCategory.MOTION,)
    domains = ("timeline", "transcript", "scenes", "assets")  # scenes: which scene a clip belongs to and what kind of picture it shows (a document must be read); assets: the picture size a pan is measured against
    settings_sections = ("motion",)
    scene_local = True
    version = "1"

    def input_hash(self, ctx: QCContext) -> str:
        return sha(super().input_hash(ctx), extras(ctx, "decisions", "facts"))

    def scene_input_hash(self, ctx: QCContext, scene_id: str) -> str:
        """Besides the scene's own signature: the text clips that can overlap its clips and the canvas the pan is measured against."""
        s = ctx.scene(scene_id)
        overlay = [(c.id, round(c.timeline_start, 3), round(c.duration, 3)) for _t, c in ctx.clips_in(s.start - 0.5, s.end + 0.5, kinds=(KIND_CAPTION, KIND_TEXT, KIND_GRAPHIC))] if s else []
        return sha(super().scene_input_hash(ctx, scene_id), overlay, ctx.canvas, ctx.fps, extras(ctx, "decisions", "facts", scene_id=scene_id))

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = ctx.settings.motion
        scenes = ctx.target_scenes()
        wanted = {s.id for s in scenes}
        rates: list[float] = []
        moving = 0
        analysed = 0
        per_scene: dict[str, int] = {}
        for n, (t, c) in enumerate(ctx.visual_clips()):
            ctx.check_cancel()
            sid = ctx.clip_scene_id(c)
            if sid not in wanted or not math.isfinite(c.duration) or c.duration <= 0:
                continue
            analysed += 1
            segs = self._segments(c)
            if not segs and not c.animation:
                continue
            moving += 1
            rates += [s.rate * max(c.scale, 1e-6) for s in segs if s.prop == "scale"]
            found = self._clip(ctx, c, t, sid, segs, cfg)
            for iss in found:
                key = (sid, iss.code)
                per_scene[key] = per_scene.get(key, 0) + 1  # type: ignore[index]
                if per_scene[key] <= MAX_ISSUES_PER_CODE_PER_SCENE:  # type: ignore[index]
                    out.issues.append(iss)
        minutes = max(1e-6, sum(s.duration for s in scenes) / 60.0)
        out.metrics = {"clips_analysed": analysed, "clips_with_motion": moving, "mean_scale_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
                       "max_scale_rate": round(max(rates), 4) if rates else 0.0, "motion_events_per_minute": round(moving / minutes, 2)}
        report(1.0, "Motion check complete")
        return out

    # ------------------------------------------------------------------ keyframe segments
    @staticmethod
    def _segments(c: Clip, skip: frozenset[str] | set[str] = frozenset()) -> list[_Seg]:
        segs = []
        for prop in MOTION_PROPS:
            pts = sorted((k for k in c.keyframes if k.property == prop and math.isfinite(k.time) and math.isfinite(k.value) and k.decision_id not in skip), key=lambda k: k.time)
            segs += [_Seg(prop, a.time, b.time, a.value, b.value, a.interpolation) for a, b in zip(pts, pts[1:]) if abs(b.value - a.value) > 1e-9]
        return segs

    @staticmethod
    def _focus_ids(ctx: QCContext, c: Clip) -> set[str]:
        """The ids of the editing decisions that planned an EVIDENCE focus on this clip (zoom to the region, hold, return): the keyframes they own are a deliberate move, not decoration."""
        def build() -> dict[str, set[str]]:
            m: dict[str, set[str]] = {}
            for d in ctx.project.editing_decisions.values():
                if d.target_id and str(getattr(d.type, "value", d.type)) == "EVIDENCE_FOCUS":
                    m.setdefault(d.target_id, set()).add(d.decision_id)
            return m

        return ctx.memo("motion.focus", build).get(c.id, set())

    def _clip(self, ctx: QCContext, c: Clip, t: Track, sid: str, segs: list[_Seg], cfg) -> list:
        out = []
        base = dict(scene_id=sid, clip=c, track=t, ctx=ctx)
        W, H = ctx.canvas
        focus = self._focus_ids(ctx, c)
        all_segs = segs
        if focus:  # the evidence focus is judged by its readability under text only; its zoom depth and speed are the point of it
            segs = self._segments(c, focus)
        scale_kf = sorted((k for k in c.keyframes if k.property == "scale" and math.isfinite(k.value) and k.value > 0 and k.decision_id not in focus), key=lambda k: k.time)
        cs = max(c.scale, 1e-6) if math.isfinite(c.scale) else 1.0
        # ---- total zoom
        peak = max([cs * k.value for k in scale_kf] + [cs]) if cs > 0 else 1.0
        if peak > cfg.max_total_scale + 1e-9:
            out.append(self.issue(
                "motion.excessive_zoom", QCCategory.MOTION, Severity.WARNING, "Zoom goes further than recommended", description=f"The picture is enlarged to {peak:.2f}x (limit {cfg.max_total_scale:.2f}x).",
                start=c.timeline_start, end=c.timeline_end, why="Heavy zoom magnifies compression and softness and crops away context.", current=f"{peak:.2f}x", recommended=f"at most {cfg.max_total_scale:.2f}x",
                suggested_fix=f"Limit the zoom to {min(cfg.max_total_scale, 1.0 + (peak - 1.0) * 0.6):.2f}x.", fix=self._soften(ctx, c, scale_kf, cfg, cap=min(cfg.max_total_scale, 1.0 + (peak - 1.0) * 0.6)),
                viewer_impact=0.4, signature=sha(c.id, round(peak, 1)), metrics={"peak_scale": round(peak, 3)}, **base))
        # ---- zoom speed / pumping
        fast = [s for s in segs if s.prop == "scale" and s.rate * cs > cfg.max_scale_per_second]
        if fast:
            s = max(fast, key=lambda s: s.rate)
            rate = s.rate * cs
            safe = self._safe_rate_plan(ctx, c, scale_kf, cfg)
            out.append(self.issue(
                "motion.abrupt_zoom", QCCategory.MOTION, Severity.WARNING, "Potentially excessive zoom speed",
                description=f"Scale {s.v0 * cs:.2f} -> {s.v1 * cs:.2f} in {s.span:.1f} s is {rate:.2f} per second (limit {cfg.max_scale_per_second:.2f}).", start=c.timeline_start + s.t0,
                end=c.timeline_start + s.t1, why="A fast zoom pulls the eye away from the words and can feel jarring.", current=f"{rate:.2f}/s ({s.v0 * cs:.2f} -> {s.v1 * cs:.2f} in {s.span:.1f} s)",
                recommended=safe[0], suggested_fix=f"Soften the zoom: {safe[0]}.", fix=safe[1], confidence=85.0, viewer_impact=0.45, signature=sha(c.id, "zoom", round(rate, 2)),
                metrics={"scale_rate": round(rate, 3)}, **base))
        elif self._reverses(scale_kf):
            out.append(self.issue(
                "motion.abrupt_zoom", QCCategory.MOTION, Severity.NOTICE, "Zoom pumps in and out", description=f"The scale changes direction repeatedly within {REVERSAL_SPAN:.1f} s.", start=c.timeline_start,
                end=c.timeline_end, why="Back-and-forth zooming reads as unsteady.", current="direction reversals", recommended="one smooth move", suggested_fix="Use a single slow zoom in one direction.", confidence=75.0,
                viewer_impact=0.3, signature=sha(c.id, "pump"), **base))
        # ---- rotation
        rot = [s for s in segs if s.prop == "rotation" and s.rate > cfg.max_rotation_per_second]
        if rot:
            s = max(rot, key=lambda s: s.rate)
            out.append(self.issue(
                "motion.rotation", QCCategory.MOTION, Severity.WARNING, "Fast or unnatural rotation", description=f"Rotation of {abs(s.v1 - s.v0):.0f} degrees in {s.span:.1f} s is {s.rate:.0f} degrees per second "
                f"(limit {cfg.max_rotation_per_second:.0f}).", start=c.timeline_start + s.t0, end=c.timeline_start + s.t1, why="Quick spins disorient viewers and rarely serve the story.", current=f"{s.rate:.0f} deg/s",
                recommended=f"under {cfg.max_rotation_per_second:.0f} deg/s", suggested_fix="Lengthen the rotation or reduce its angle.", viewer_impact=0.4, signature=sha(c.id, "rot"), confidence=90.0, **base))
        # ---- keyframe jumps: a big change inside one frame
        jumps = self._jumps(c, ctx, cfg, W)
        if jumps:
            prop, k0, k1, frac = jumps[0]
            out.append(self.issue(
                "motion.keyframe_jump", QCCategory.MOTION, Severity.WARNING, "Keyframe makes the picture jump", description=f"{prop} changes by {frac:.0%} of its range between {k0:.2f} s and {k1:.2f} s, "
                f"less than one frame at {ctx.fps} fps (limit {cfg.keyframe_jump:.0%}).", start=c.timeline_start + k0, end=c.timeline_start + k1 + ctx.frame, why="The picture snaps instead of moving; it looks like a glitch.",
                current=f"{frac:.0%} in {(k1 - k0) * 1000:.0f} ms", recommended=f"under {cfg.keyframe_jump:.0%} per frame", suggested_fix="Spread the change over a longer time or remove the stray keyframe.",
                viewer_impact=0.5, signature=sha(c.id, prop, round(k0, 2)), confidence=95.0, **base))
        # ---- animation / keyframes past the clip
        over = [k for k in c.keyframes if k.property in MOTION_PROPS and k.time > c.duration + 1e-3]
        anim = self._anim_overrun(c)
        if over or anim:
            what = f"a keyframe at {max(k.time for k in over):.2f} s" if over else f"the {anim} animation"
            out.append(self.issue(
                "motion.animation_overrun", QCCategory.MOTION, Severity.WARNING, "Movement extends beyond the clip", description=f"{what.capitalize()} is later than the clip's {c.duration:.2f} s, so the move is cut off.",
                start=c.timeline_start, end=c.timeline_end, why="The move never completes, so the picture ends mid-motion.", current=what, recommended=f"inside {c.duration:.2f} s",
                suggested_fix="Shorten the movement so it finishes within the clip.", viewer_impact=0.3, signature=sha(c.id, "overrun"), **base))
        # ---- off the picture
        edge = self._off_image(ctx, c, scale_kf, cs)
        if edge:
            out.append(self.issue(
                "motion.off_image", QCCategory.MOTION, Severity.WARNING, "Movement shows empty edges", description=edge, start=c.timeline_start, end=c.timeline_end,
                why="Moving past the edge of the picture exposes the background.", current="outside the picture", recommended="stay inside the picture", suggested_fix="Zoom in a little or reduce the pan so the picture always fills the frame.",
                viewer_impact=0.5, signature=sha(c.id, "edge"), confidence=85.0, **base))
        # ---- unnecessary motion
        if c.duration < MIN_MOTION_CLIP and any(s.prop == "scale" and abs(s.v1 - s.v0) * cs >= MIN_ZOOM_DELTA or s.prop != "scale" for s in segs):
            out.append(self.issue(
                "motion.unnecessary", QCCategory.MOTION, Severity.NOTICE, "Movement on a very short clip", description=f"The clip lasts {c.duration:.2f} s; movement that short is felt as a twitch, not a move.",
                start=c.timeline_start, end=c.timeline_end, why="Short shots read better static.", current=f"{c.duration:.2f} s with motion", recommended="no movement under 1 s", suggested_fix="Remove the movement from this clip.",
                viewer_impact=0.15, signature=sha(c.id, "short"), confidence=80.0, **base))
        elif self._reading_visual(ctx, c, sid) and max((abs(s.v1 - s.v0) * cs for s in segs if s.prop == "scale"), default=0.0) >= 0.1 and not self._planned_evidence_move(ctx, c):
            out.append(self.issue(
                "motion.unnecessary", QCCategory.MOTION, Severity.NOTICE, "Movement on a picture that has to be read", description="This is a document, chart or screenshot, and it moves while it is on screen.",
                start=c.timeline_start, end=c.timeline_end, why="Viewers read evidence; a moving page is harder to read.", current="moving evidence", recommended="static, or a deliberate focus zoom", suggested_fix="Hold the picture still, or use an evidence focus on the relevant region.",
                viewer_impact=0.3, signature=sha(c.id, "evidence"), confidence=70.0, **base))
        # ---- readability under text
        out += self._readability(ctx, c, t, sid, all_segs, cs, cfg, W, focus)
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _reverses(kfs: list[Keyframe]) -> bool:
        pts = [(k.time, k.value) for k in kfs]
        dirs = [(1 if b[1] > a[1] else -1, b[0]) for a, b in zip(pts, pts[1:]) if abs(b[1] - a[1]) > MIN_ZOOM_DELTA]
        return any(d1 != d2 and d3 != d2 and t3 - t1 <= REVERSAL_SPAN for (d1, t1), (d2, _t2), (d3, t3) in zip(dirs, dirs[1:], dirs[2:])) if len(dirs) >= 3 else False

    @staticmethod
    def _jumps(c: Clip, ctx: QCContext, cfg, width: int) -> list[tuple[str, float, float, float]]:
        ranges = {"scale": 1.0, "rotation": 90.0, "position_x": float(width), "position_y": float(ctx.canvas[1])}  # the span a full-range change would cover
        out = []
        for prop, rng in ranges.items():
            pts = sorted((k for k in c.keyframes if k.property == prop and math.isfinite(k.time) and math.isfinite(k.value)), key=lambda k: k.time)
            for a, b in zip(pts, pts[1:]):
                if b.time - a.time <= ctx.frame + 1e-6 and abs(b.value - a.value) / rng > cfg.keyframe_jump and (b.time - a.time) > 0:
                    out.append((prop, a.time, b.time, abs(b.value - a.value) / rng))
        return out

    @staticmethod
    def _anim_overrun(c: Clip) -> str:
        a = c.animation or {}
        for side in ("in", "out"):
            s = a.get(side)
            if isinstance(s, dict):
                try:
                    if float(s.get("duration", 0)) + float(s.get("delay", 0)) > c.duration + 1e-3:
                        return side
                except (TypeError, ValueError):
                    continue
        return ""

    def _off_image(self, ctx: QCContext, c: Clip, scale_kf: list[Keyframe], cs: float) -> str:
        if str(c.effects.get("fit", "cover")) in ("contain", "fit") or not scale_kf and not any(k.property.startswith("position") for k in c.keyframes):
            return ""
        lo = min([cs * k.value for k in scale_kf] + [cs])
        if lo < 0.98 and lo > 0:
            return f"The picture shrinks to {lo:.2f}x, which leaves empty edges around it."
        asset = ctx.asset(c.asset_id)
        W, H = ctx.canvas
        aw, ah = (asset.width or W, asset.height or H) if asset else (W, H)
        base = max(W / aw, H / ah)
        for k in (k for k in c.keyframes if k.property in ("position_x", "position_y")):
            s = cs * value_at(c.keyframes, "scale", k.time)
            mx, my = max(0.0, (aw * base * s - W) / 2), max(0.0, (ah * base * s - H) / 2)
            off = abs(c.position[0] + (k.value if k.property == "position_x" else value_at(c.keyframes, "position_x", k.time))) if k.property == "position_x" else abs(c.position[1] + k.value)
            if off > (mx if k.property == "position_x" else my) + 1.0:
                return f"At {k.time:.1f} s the picture is moved {off:.0f} px sideways/up, but only {(mx if k.property == 'position_x' else my):.0f} px of margin exist at that zoom, so an empty edge shows."
        return ""

    def _reading_visual(self, ctx: QCContext, c: Clip, sid: str) -> bool:
        sc = ctx.scene_ctx(sid)
        if sc is None:
            return False
        return sc.visual_type in EVIDENCE_TYPES

    @staticmethod
    def _planned_evidence_move(ctx: QCContext, c: Clip) -> bool:
        d = ctx.project.editing_decisions.get(c.ai_decision_id)
        return bool(d is not None and str(getattr(d.type, "value", d.type)) == "EVIDENCE_FOCUS") or bool(c.effects.get("highlight") or c.effects.get("focus_region"))

    def _readability(self, ctx: QCContext, c: Clip, t: Track, sid: str, segs: list[_Seg], cs: float, cfg, width: int, focus: set[str] = frozenset()) -> list:  # type: ignore[assignment]
        out = []
        limit = cfg.max_zoom_during_text
        for tt, o in ctx.clips_in(c.timeline_start, c.timeline_end, kinds=(KIND_CAPTION, KIND_TEXT, KIND_GRAPHIC)):
            if o.kind == KIND_GRAPHIC and focus and o.metadata.get("evidence_decision") in focus:
                continue  # the highlight the focus zoom belongs to: moving towards it is its purpose
            a, b = max(c.timeline_start, o.timeline_start), min(c.timeline_end, o.timeline_end)
            if b - a < OVERLAP_MIN:
                continue
            la, lb = a - c.timeline_start, b - c.timeline_start
            worst = 0.0
            for s in segs:
                if s.prop not in ("scale", "position_x", "position_y") or s.t1 <= la or s.t0 >= lb:
                    continue
                share = (min(s.t1, lb) - max(s.t0, la)) / s.span
                rate = (s.rate * cs if s.prop == "scale" else s.rate / max(1.0, width)) if share > 0 else 0.0  # pan as a share of the frame width per second
                worst = max(worst, rate)
            if worst > limit:
                what = {KIND_CAPTION: "a caption", KIND_TEXT: "text", KIND_GRAPHIC: "a graphic"}[o.kind]
                out.append(self.issue(
                    "motion.readability", QCCategory.MOTION, Severity.NOTICE if o.kind == KIND_CAPTION else Severity.WARNING, "Movement while text is on screen",
                    description=f"The picture moves at {worst:.2f} per second while {what} is shown ({_t(a)}-{_t(b)}); the comfortable limit is {limit:.2f}.", scene_id=sid, clip=c, track=t, ctx=ctx,
                    start=a, end=b, why="Movement under text makes it harder to read.", current=f"{worst:.2f}/s", recommended=f"under {limit:.2f}/s while text is visible",
                    suggested_fix="Slow the movement or let it finish before the text appears.", fix=self._soften(ctx, c, sorted((k for k in c.keyframes if k.property == "scale"), key=lambda k: k.time), cfg, rate=limit * 0.8),
                    confidence=80.0, viewer_impact=0.4, signature=sha(c.id, o.id)))
                break
        return out

    # ------------------------------------------------------------------ recommended alternative (needs confirmation, never automatic)
    def _safe_rate_plan(self, ctx: QCContext, c: Clip, kfs: list[Keyframe], cfg) -> tuple[str, object]:
        allowed = cfg.max_scale_per_second * SOFT_FACTOR
        fix = self._soften(ctx, c, kfs, cfg, rate=allowed)
        new = [k for k in (fix.params["keyframes"] if fix else []) if k["property"] == "scale"]
        cs = max(c.scale, 1e-6)
        if len(new) >= 2:
            a, b = new[0], new[-1]
            return f"{a['value'] * cs:.2f} -> {b['value'] * cs:.2f} over {b['time'] - a['time']:.1f} s", fix
        return f"at most {allowed:.2f} per second", fix

    def _soften(self, ctx: QCContext, c: Clip, kfs: list[Keyframe], cfg, *, rate: float | None = None, cap: float | None = None):
        """The scale keyframes of a gentler move: each fast segment is stretched in time (up to the next keyframe or the end of the clip) and, when it still cannot be slow enough, its
        end value is moved closer to the start; values stay inside [1/clip scale, cap/clip scale]. None when there is nothing sensible to offer."""
        if len(kfs) < 2:
            return None
        cs = max(c.scale, 1e-6)
        rate = rate if rate is not None else cfg.max_scale_per_second * SOFT_FACTOR
        top = (cap if cap is not None else cfg.max_total_scale) / cs
        pts = [[k.time, min(k.value, top), k.interpolation] for k in kfs]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            delta = (b[1] - a[1]) * cs
            if abs(delta) / max(1e-6, b[0] - a[0]) <= rate + 1e-9:
                continue
            room = (pts[i + 2][0] if i + 2 < len(pts) else c.duration) - a[0]
            need = abs(delta) / rate
            b[0] = a[0] + min(need, max(room, b[0] - a[0]))
            if abs(b[1] - a[1]) * cs / max(1e-6, b[0] - a[0]) > rate + 1e-9:  # not enough time in the clip: move the end value closer
                b[1] = a[1] + math.copysign(rate * (b[0] - a[0]) / cs, b[1] - a[1])
        new = [{"property": "scale", "time": round(p[0], 3), "value": round(max(0.05, p[1]), 4), "interpolation": "ease_in_out" if len(pts) == 2 else p[2]} for p in pts]
        if all(abs(n["time"] - k.time) < 1e-3 and abs(n["value"] - k.value) < 1e-3 for n, k in zip(new, kfs)):
            return None
        if any(n["time"] > c.duration + 1e-6 for n in new):
            return None
        end = new[-1]
        return fx.motion_soften(c.id, new, f"Soften the zoom to {new[0]['value'] * cs:.2f} -> {end['value'] * cs:.2f} over {end['time'] - new[0]['time']:.1f} s", ctx.settings)


def _t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"
