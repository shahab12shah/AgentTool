"""Black and frozen frames (spec 22).

Three sources, from cheapest to most telling:

(a) the timeline itself: a stretch where no picture covers the canvas, outside the narration (inside it the timeline checker reports the gap), renders as black;
(b) the *used part* of every source video: ffmpeg's ``blackdetect`` / ``freezedetect`` run on exactly the range the timeline plays, at a small size and with a time budget
    (largest first, the rest is reported as skipped), cached on disk per (content, range, thresholds), and mapped back to timeline time;
(c) a rendered file, when the post-render pass hands one in (the same detectors over the whole file; that pass lives in ``render_checker``).

Black that the user declared (``frames.intentional_black``) or that a fade-through-black transition produces is intentional and reported only as information. A still image is
never "frozen", and a deliberate hold on evidence is not either. Without FFmpeg only (a) runs and the notes say so.

Sensitivity mapping (both 0..1, 0.5 = default):  black ``pix_th = 0.16 - 0.12 * s`` (0.5 -> 0.10: a pixel counts as black up to 10 % luminance; "near black" is detected up to
``pix_th + 0.08``); frozen ``noise = -80 + 40 * s`` dB (0.5 -> -60 dB, higher = flags smaller differences as frozen).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCCancelled, QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.media_facts import ffmpeg_ready
from app.qc.severity import Severity
from app.timeline.clip import Clip
from app.timeline.track import Track

NEAR_MARGIN = 0.08  # near-black is detected this far above the black threshold
DECODE_ERRORS = re.compile(r"(error while decoding|invalid data found|corrupt|missing picture|concealing|non-existing pps|decode_slice_header|invalid nal|overread|reference picture missing)", re.I)
BLACK_RE = re.compile(r"black_start:(-?[\d.]+)\s+black_end:(-?[\d.]+)\s+black_duration:(-?[\d.]+)")
FREEZE_RE = re.compile(r"lavfi\.freezedetect\.freeze_(start|end|duration):\s*(-?[\d.]+)")
SCAN_WIDTH = 160
CACHE_VERSION = 1
MAX_REPORTED = 8


def black_pix_th(sensitivity: float) -> float:
    return round(0.16 - 0.12 * max(0.0, min(1.0, sensitivity)), 4)


def freeze_noise_db(sensitivity: float) -> float:
    return round(-80.0 + 40.0 * max(0.0, min(1.0, sensitivity)), 1)


@dataclass
class Scan:
    """Detector results for one scanned range; all times are seconds from the start of the scanned range."""

    ok: bool = False  # ffmpeg ran
    black: list[tuple[float, float]] = field(default_factory=list)
    near_black: list[tuple[float, float]] = field(default_factory=list)
    frozen: list[tuple[float, float]] = field(default_factory=list)
    decode_errors: list[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "black": self.black, "near_black": self.near_black, "frozen": self.frozen, "decode_errors": self.decode_errors, "duration": self.duration}

    @classmethod
    def from_dict(cls, d: dict) -> "Scan":
        return cls(bool(d.get("ok")), [tuple(x) for x in d.get("black", [])], [tuple(x) for x in d.get("near_black", [])], [tuple(x) for x in d.get("frozen", [])],
                   list(d.get("decode_errors", [])), float(d.get("duration", 0.0)))


# ---------------------------------------------------------------------- the detectors (shared with the post-render pass)
def _run(ctx: QCContext, args: list[str], timeout: float) -> tuple[int, str, bytes]:
    """Run FFmpeg, polling the cancel flag so a long scan stops promptly."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waited = 0.0
    while True:
        try:
            out, err = proc.communicate(timeout=0.25)
            return proc.returncode, err.decode("utf-8", "replace"), out
        except subprocess.TimeoutExpired:
            waited += 0.25
            if ctx.cancel.is_set():
                proc.kill()
                proc.communicate()
                raise QCCancelled() from None
            if waited > timeout:
                proc.kill()
                proc.communicate()
                return -9, "scan timed out", b""


def _mean_luma(ctx: QCContext, path: Path, t: float) -> float | None:
    """Average luminance (0..1) of the frame at ``t`` seconds into ``path``."""
    try:
        rc, _err, out = _run(ctx, [ctx.ffmpeg.ffmpeg(), "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1", "-vf", "scale=16:16,format=gray",
                                   "-f", "rawvideo", "-"], 20)  # type: ignore[union-attr]
    except OSError:
        return None
    return (sum(out) / (255.0 * len(out))) if rc == 0 and out else None


def scan_video(ctx: QCContext, path: Path, start: float, duration: float, *, black_min: float, black_sens: float, freeze_min: float, freeze_sens: float, frozen: bool = True) -> Scan:
    """Run blackdetect (and freezedetect) over ``duration`` seconds of ``path`` starting at ``start``."""
    pix = black_pix_th(black_sens)
    vf = f"scale={SCAN_WIDTH}:-2,blackdetect=d={black_min:g}:pix_th={pix + NEAR_MARGIN:g}:pic_th=0.98"
    if frozen:
        vf += f",freezedetect=n={freeze_noise_db(freeze_sens):g}dB:d={freeze_min:g}"
    args = [ctx.ffmpeg.ffmpeg(), "-hide_banner", "-nostdin", "-v", "info", "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}", "-i", str(path), "-an", "-vf", vf, "-f", "null", "-"]  # type: ignore[union-attr]
    rc, err, _ = _run(ctx, args, max(60.0, duration * 4))
    sc = Scan(ok=rc == 0, duration=duration)
    loose: list[tuple[float, float]] = []
    cur_start = None
    freezes: list[tuple[float, float]] = []
    for line in err.splitlines():
        m = BLACK_RE.search(line)
        if m:
            loose.append((max(0.0, float(m.group(1))), min(duration, float(m.group(2)))))
            continue
        fm = FREEZE_RE.search(line)
        if fm:
            kind, val = fm.group(1), float(fm.group(2))
            if kind == "start":
                cur_start = val
            elif kind == "end" and cur_start is not None:
                freezes.append((cur_start, val))
                cur_start = None
        elif DECODE_ERRORS.search(line) and "blackdetect" not in line and "freezedetect" not in line:
            if len(sc.decode_errors) < 5:
                sc.decode_errors.append(line.strip()[:200])
    if cur_start is not None:  # a freeze that lasts to the end of the range has no end marker
        freezes.append((cur_start, duration))
    for a, b in loose:
        mid = (a + b) / 2
        luma = _mean_luma(ctx, path, start + mid)
        (sc.black if (luma is None or luma <= pix + 1e-3) else sc.near_black).append((a, b))
    sc.frozen = [(a, b) for a, b in freezes if b - a >= freeze_min - 1e-6 and not _covered(a, b, sc.black + sc.near_black)]  # a black stretch is also "frozen": report it once
    if rc != 0 and not sc.decode_errors:
        sc.decode_errors.append(err.strip().splitlines()[-1][:200] if err.strip() else "ffmpeg could not decode this range")
    return sc


def _covered(a: float, b: float, spans: list[tuple[float, float]], share: float = 0.8) -> bool:
    got = sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)
    return got >= share * max(1e-6, b - a)


class FrameChecker(BaseChecker):
    id = "frames"
    label = "Black & frozen frames"
    categories = (QCCategory.FRAMES,)
    domains = ("timeline", "assets", "transcript", "scenes")  # scenes: an evidence / data scene may hold a still picture
    settings_sections = ("frames", "intentional_gaps")
    scene_local = False
    expensive = True
    version = "1"

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = ctx.settings.frames
        self._timeline_black(ctx, out, cfg)
        report(0.2, "Timeline checked for empty stretches")
        scanned = intentional = 0
        if cfg.analyze_assets and ffmpeg_ready(ctx):
            scanned, intentional = self._sources(ctx, out, cfg, report)
        elif not ffmpeg_ready(ctx):
            out.notes.append("source videos were not scanned: FFmpeg is not available (timeline stretches only)")
        out.metrics = {"sources_scanned": scanned, "intentional_black": intentional}
        report(1.0, "Frame check complete")
        return out

    # ------------------------------------------------------------------ (a) the timeline itself
    def _timeline_black(self, ctx: QCContext, out: CheckerOutput, cfg) -> None:
        end = ctx.duration
        if end <= 0:
            return
        spans = sorted((max(0.0, c.timeline_start), min(end, c.timeline_end)) for t, c in ctx.visual_clips() if not t.hidden and c.opacity > 0.01 and c.duration > 0)
        gaps, cur = [], 0.0
        for a, b in spans:
            if a > cur + 1e-6:
                gaps.append((cur, a))
            cur = max(cur, b)
        if cur < end - 1e-6:
            gaps.append((cur, end))
        speech = self._speech(ctx)
        done = 0
        for a, b in gaps:
            for x, y in self._subtract((a, b), speech):  # inside the narration the timeline checker already reports the gap
                if y - x < cfg.black_min_seconds - 1e-6 or done >= MAX_REPORTED:
                    continue
                if ctx.in_intentional_black(x, y) or ctx.in_intentional_gap(x, y):
                    out.issues.append(self._intentional(ctx, x, y, "declared as an intentional blackout"))
                    continue
                done += 1
                out.issues.append(self.issue(
                    "frames.black_timeline", QCCategory.FRAMES, Severity.ERROR, "Black screen: nothing is on the timeline", description=f"No picture covers {_t(x)}-{_t(y)} ({y - x:.1f} s), so the render shows black there.",
                    start=x, end=y, scene_id=(ctx.scene_at((x + y) / 2).id if ctx.scene_at((x + y) / 2) else None), why="A black screen the viewer did not ask for looks like an export error.",
                    current=f"{y - x:.1f} s without picture", recommended="a visual, or a declared intentional blackout", suggested_fix="Extend a neighbouring clip or add a visual; declare it intentional in the QC settings if it is deliberate.",
                    viewer_impact=0.8, signature=sha(round(x, 1), round(y, 1)), ctx=ctx))

    @staticmethod
    def _speech(ctx: QCContext) -> list[tuple[float, float]]:
        from app.audio.ducking import speech_segments  # noqa: PLC0415

        return speech_segments(ctx.words) if ctx.words else []

    @staticmethod
    def _subtract(span: tuple[float, float], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out = [span]
        for c0, c1 in cuts:
            nxt = []
            for a, b in out:
                if c1 <= a or c0 >= b:
                    nxt.append((a, b))
                    continue
                if c0 > a:
                    nxt.append((a, c0))
                if c1 < b:
                    nxt.append((c1, b))
            out = nxt
        return out

    def _intentional(self, ctx: QCContext, a: float, b: float, why: str) -> QCIssue:
        return self.issue("frames.black_intentional", QCCategory.FRAMES, Severity.INFO, "Intentional black screen", description=f"Black from {_t(a)} to {_t(b)} ({why}); left as it is.", start=a, end=b,
                          why="Deliberate black screens are allowed.", current=f"{b - a:.1f} s", recommended="keep", suggested_fix="Nothing to do.", viewer_impact=0.0, signature=sha(round(a, 1), round(b, 1)), ctx=ctx)

    # ------------------------------------------------------------------ (b) the used range of every source video
    def _sources(self, ctx: QCContext, out: CheckerOutput, cfg, report: ProgressFn) -> tuple[int, int]:
        ranges: dict[str, list[tuple[float, float, Clip, Track]]] = {}
        for t, c in ctx.visual_clips():
            a = ctx.asset(c.asset_id)
            if a is None or a.type.value != "video" or t.hidden or c.duration <= 0 or not ctx.asset_path(a).is_file():
                continue
            ranges.setdefault(a.id, []).append((c.source_in, c.source_out, c, t))
        jobs = []
        for aid, rs in ranges.items():
            for lo, hi in self._merge([(r[0], r[1]) for r in rs]):
                jobs.append((aid, lo, hi, [r for r in rs if r[0] < hi and r[1] > lo]))
        jobs.sort(key=lambda j: -(j[2] - j[1]))  # largest first
        budget, scanned, skipped, intentional = cfg.max_scan_seconds, 0, 0, 0
        for n, (aid, lo, hi, users) in enumerate(jobs):
            ctx.check_cancel()
            if hi - lo > budget + 1e-6:
                skipped += 1
                continue
            budget -= hi - lo
            asset = ctx.asset(aid)
            report(0.2 + 0.75 * n / max(1, len(jobs)), f"Scanning {asset.name}")  # type: ignore[union-attr]
            sc = self._cached_scan(ctx, asset, lo, hi - lo, cfg)
            scanned += 1
            for clip_src_in, clip_src_out, clip, track in users:
                intentional += self._report(ctx, out, cfg, sc, lo, clip, track, asset)
        if skipped:
            out.complete = False
            out.notes.append(f"{skipped} source video(s) not scanned: the scan budget of {cfg.max_scan_seconds:.0f} s was used up (largest first)")
        out.notes.append(f"scanned {scanned} source range(s)")
        return scanned, intentional

    @staticmethod
    def _merge(rs: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[list[float]] = []
        for a, b in sorted(rs):
            if out and a <= out[-1][1] + 0.25:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    def _cached_scan(self, ctx: QCContext, asset, lo: float, dur: float, cfg) -> Scan:
        key = sha(asset.content_hash or asset.id, round(lo, 2), round(dur, 2), cfg.black_sensitivity, cfg.black_min_seconds, cfg.frozen_sensitivity, cfg.frozen_min_seconds, CACHE_VERSION)
        cache = (ctx.root / "cache" / "qc" / "frames" / f"{key}.json") if ctx.root else None
        if cache is not None and cache.is_file():
            try:
                return Scan.from_dict(json.loads(cache.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        sc = scan_video(ctx, ctx.asset_path(asset), lo, dur, black_min=cfg.black_min_seconds, black_sens=cfg.black_sensitivity, freeze_min=cfg.frozen_min_seconds, freeze_sens=cfg.frozen_sensitivity)
        if cache is not None and sc.ok:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(sc.to_dict()), encoding="utf-8")
            except OSError:
                pass  # the cache is an optimisation only
        return sc

    def _report(self, ctx: QCContext, out: CheckerOutput, cfg, sc: Scan, lo: float, clip: Clip, track: Track, asset) -> int:
        """Turn one scan into issues for one clip that plays part of it; returns the number of intentional black stretches."""
        n_int = 0

        def to_timeline(span: tuple[float, float]) -> tuple[float, float] | None:
            a, b = lo + span[0], lo + span[1]  # source seconds
            a, b = max(a, clip.source_in), min(b, clip.source_out)
            if b - a < 1e-3:
                return None
            sp = max(clip.speed, 1e-6)
            return clip.timeline_start + (a - clip.source_in) / sp, clip.timeline_start + (b - clip.source_in) / sp

        sid = ctx.clip_scene_id(clip)
        for kind, spans in (("black", sc.black), ("near_black", sc.near_black)):
            for span in spans:
                m = to_timeline(span)
                if m is None or len(out.issues) > 60:
                    continue
                a, b = m
                if ctx.in_intentional_black(a, b) or self._fade_through_black(clip, a, b):
                    n_int += 1
                    out.issues.append(self._intentional(ctx, a, b, "a deliberate blackout" if ctx.in_intentional_black(a, b) else "a fade through black"))
                    continue
                if kind == "black":
                    out.issues.append(self.issue(
                        "frames.black", QCCategory.FRAMES, Severity.ERROR if b - a > 1.0 else Severity.WARNING, "Black frames in the footage", description=f"“{asset.name}” is black for {b - a:.1f} s at {_t(a)}-{_t(b)} "
                        f"(source {_t(lo + span[0])}-{_t(lo + span[1])}).", scene_id=sid, clip=clip, track=track, start=a, end=b, why="Unintended black frames look like a playback error.",
                        current=f"{b - a:.1f} s of black", recommended="no black unless intentional", suggested_fix="Trim the clip past the black part, replace it, or declare it intentional in the QC settings.",
                        fix=fx.navigate("visual.replace", "Choose another visual", scene_id=sid or ""), confidence=95.0, viewer_impact=0.7 if b - a > 1.0 else 0.4, signature=sha(clip.id, round(a, 1)), ctx=ctx))
                else:
                    out.issues.append(self.issue(
                        "frames.near_black", QCCategory.FRAMES, Severity.NOTICE, "Nearly black frames in the footage", description=f"“{asset.name}” is almost black for {b - a:.1f} s at {_t(a)}-{_t(b)}.",
                        scene_id=sid, clip=clip, track=track, start=a, end=b, why="Very dark footage may read as a blank screen on many displays.", current=f"{b - a:.1f} s very dark", recommended="visible picture",
                        suggested_fix="Check the footage; brighten or replace it if the darkness is not intended.", confidence=75.0, viewer_impact=0.3, signature=sha(clip.id, "near", round(a, 1)), ctx=ctx))
        for span in sc.frozen:
            m = to_timeline(span)
            if m is None:
                continue
            a, b = m
            if self._holds_deliberately(ctx, clip, sid):
                continue
            out.issues.append(self.issue(
                "frames.frozen", QCCategory.FRAMES, Severity.WARNING, "Frozen picture in the footage", description=f"“{asset.name}” shows an unchanged picture for {b - a:.1f} s at {_t(a)}-{_t(b)} while the sound goes on.",
                scene_id=sid, clip=clip, track=track, start=a, end=b, why="A frozen frame in moving footage looks like a stall or a dropped signal.", current=f"{b - a:.1f} s unchanged", recommended="moving footage",
                suggested_fix="Trim the clip before the freeze or use another part of the footage.", fix=fx.navigate("visual.replace", "Choose another visual", scene_id=sid or ""), confidence=85.0, viewer_impact=0.45,
                signature=sha(clip.id, "freeze", round(a, 1)), ctx=ctx))
        if sc.decode_errors:
            out.issues.append(self.issue(
                "frames.decode_error", QCCategory.FRAMES, Severity.ERROR, "The footage has decoding errors", description=f"FFmpeg reported errors while decoding “{asset.name}”: {sc.decode_errors[0]}", scene_id=sid, clip=clip,
                track=track, start=clip.timeline_start, end=clip.timeline_end, why="Damaged footage can show glitches, green frames or fail to render.", current=sc.decode_errors[0][:80], recommended="clean footage",
                suggested_fix="Replace the file or re-encode it.", fix=fx.navigate("asset.replace", "Replace the asset", asset_id=asset.id), confidence=90.0, viewer_impact=0.6, signature=sha(asset.id, "decode"), ctx=ctx))
        return n_int

    @staticmethod
    def _fade_through_black(clip: Clip, a: float, b: float) -> bool:
        tr = clip.transition or {}
        d = float(tr.get("duration", 0.0) or 0.0)
        return str(tr.get("type", "CUT")).upper() in ("FADE", "DISSOLVE") and d > 0 and (a <= clip.timeline_start + d + 0.25 or b >= clip.timeline_end - 0.25) and (b - a) <= d + 0.5

    @staticmethod
    def _holds_deliberately(ctx: QCContext, clip: Clip, sid: str | None) -> bool:
        if not sid:
            return False
        sc = ctx.scene_ctx(sid)
        if sc is not None and sc.visual_type in ("EVIDENCE", "DATA"):
            return True
        brief = ctx.project.editing_strategy.briefs.get(sid) if ctx.project.editing_strategy else None
        return bool(brief is not None and (brief.keep_static or brief.evidence_treatment_needed))


def _t(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"

