"""Render readiness (spec 25) and post-render inspection (spec 41).

READINESS answers "if the user pressed Export now, would it work?" without rendering anything. It reuses the Phase 6 machinery: the encoder choice (``EncoderSelector``) and a
dry-run of the render pipeline (``RenderExecutor.prepare``), which validates the timeline, reads every media file's facts, compiles every section and builds every FFmpeg
command and the audio mix - but runs nothing and creates no files. What the pre-flight checker already reported (missing FFmpeg, bad settings, disk space, missing media) is not
reported twice.

POST-RENDER is a second, separate pass on the exported FILE: the Phase 6 ``RenderValidator`` (streams, duration, size, frame rate, codec, audibility) plus black and frozen frames
over the whole file, silence where the transcript says the voice is speaking, final loudness and true peak (reported, no fixed target), and a full decode pass for corruption.
Its result is a plain JSON document stored with the project; the file is never modified and the timeline QC results are never replaced by it.
"""

from __future__ import annotations

import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.audio.ducking import speech_segments
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCCancelled, QCContext, sha
from app.qc.frame_checker import scan_video
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.media_facts import ffmpeg_ready
from app.qc.preflight import PreflightChecker
from app.qc.severity import Severity
from app.rendering import presets as P
from app.rendering.errors import EncoderUnavailableError, RenderCancelled, RenderError
from app.rendering.executor import JobSpec, NullLog, RenderExecutor
from app.rendering.fonts import FontResolver
from app.rendering.models import RenderSnapshot
from app.rendering.planner import EncoderSelector
from app.rendering.validator import Expectations, RenderValidator

# preflight code -> the readiness code used when the pre-flight did not run in this pass
MIRROR = {"preflight.encoder": "render.codec_unavailable", "preflight.export_settings": "render.codec_unavailable", "preflight.ffmpeg_features": "render.codec_unavailable",
          "preflight.output": "render.output_invalid", "preflight.disk": "render.disk_space", "preflight.fonts": "render.font_missing"}
SILENCE_DB = -50.0
SILENCE_MIN = 1.0
MAX_WARNING_LINES = 3


def _warning_kind(text: str) -> tuple[str, Severity]:
    low = text.lower()
    if "caption" in low and "could not be rendered" in low:
        return "render.caption_unrenderable", Severity.ERROR
    if "font" in low or "not installed" in low or "substitut" in low:
        return "render.font_missing", Severity.WARNING
    if "transition" in low:
        return "render.transition_unsupported", Severity.WARNING
    return "render.effect_unsupported", Severity.WARNING


class RenderReadinessChecker(BaseChecker):
    id = "render"
    label = "Render readiness"
    categories = (QCCategory.RENDER_READINESS,)
    domains = ("timeline", "assets", "render", "captions", "audio")  # the dry run compiles the caption styles / fonts and the audio mix too
    settings_sections = ("media",)
    scene_local = False
    expensive = True
    uses_shared = True  # it stands down where earlier checkers already reported the same problem
    version = "1"

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        shared = ctx.shared
        pre = shared.get("preflight")
        if pre is None:  # run alone: mirror the environment findings of the pre-flight under the readiness codes
            pre_out = PreflightChecker().run(ctx, lambda f, m: None)
            for i in pre_out.issues:
                if i.code in MIRROR:
                    out.issues.append(self._mirror(ctx, i))
            known = {i.code for i in pre_out.issues}
        else:
            known = {i.code for i in pre.issues}
        asset_missing = any(i.code == "asset.missing" for i in getattr(shared.get("asset"), "issues", []))
        if not ffmpeg_ready(ctx):
            out.notes.append("the render graph was not compiled: FFmpeg is not available")
            return self._finish(out)
        report(0.2, "Compiling the render graph (dry run)")
        proj = ctx.project
        snap = RenderSnapshot.from_project(proj, proj.render_settings, PreflightChecker._proxy_refs(ctx))
        try:
            resolved = EncoderSelector(ctx.ffmpeg).resolve(proj.render_settings, snap, force_cpu=True)  # type: ignore[arg-type]
        except EncoderUnavailableError as exc:
            if not known & {"preflight.encoder", "preflight.export_settings"}:
                out.issues.append(self._make(ctx, "render.codec_unavailable", Severity.CRITICAL, "The export codec is not available", f"{exc.user_message} Alternatives: {', '.join(exc.alternatives) or 'none found'}.",
                                             "Choose a codec this FFmpeg can encode."))
            return self._finish(out)
        except RenderError as exc:
            if not known & {"preflight.encoder", "preflight.export_settings"}:
                out.issues.append(self._make(ctx, "render.codec_unavailable", Severity.CRITICAL, "The export settings cannot be used", exc.user_message, exc.possible_issue or "Change the export settings."))
            return self._finish(out)
        tmp = Path(tempfile.gettempdir()) / f"qc_dry_{sha(proj.project_id)}"  # never created: prepare() only plans
        spec = JobSpec("qc_dry_run", snap, resolved, tmp / "out.mp4", tmp / "work", tmp / "cache", tmp / "run", "export", "final", set(), 30.0, None, False, proj.render_settings.duration_tolerance)
        executor = RenderExecutor(ctx.ffmpeg, ctx.probe, FontResolver(""), RenderValidator(ctx.ffmpeg, ctx.probe))  # type: ignore[arg-type]
        cancel = ctx.cancel if isinstance(ctx.cancel, threading.Event) else threading.Event()
        try:
            prepared = executor.prepare(spec, cancel, NullLog())
        except RenderCancelled:
            raise QCCancelled() from None
        except RenderError as exc:
            self._graph_failed(ctx, out, exc, known, asset_missing)
            return self._finish(out)
        out.metrics.update(chunks=len(prepared.chunks), commands=len(prepared.cmds) + (1 if prepared.amix else 0))
        report(0.8, "Reading the compiler's warnings")
        self._warnings(ctx, out, list(prepared.plan.warnings), known)
        report(1.0, "Render readiness check complete")
        return self._finish(out)

    # ------------------------------------------------------------------ helpers
    def _finish(self, out: CheckerOutput) -> CheckerOutput:
        blocking = sum(1 for i in out.issues if i.severity is Severity.CRITICAL)
        out.metrics.update(ready=blocking == 0 and "blocked_by" not in out.metrics, blocking=blocking, warnings=sum(1 for i in out.issues if i.severity is Severity.WARNING))
        return out

    def _make(self, ctx: QCContext, code: str, sev: Severity, title: str, desc: str, fix_text: str, *, affected: list[str] | None = None) -> QCIssue:
        return self.issue(code, QCCategory.RENDER_READINESS, sev, title, description=desc, affected=affected or [], why="The export would fail or differ from the preview." if sev is Severity.CRITICAL else "The export will use a fallback.",
                          current=desc[:120], recommended="resolved", suggested_fix=fix_text, viewer_impact=1.0 if sev is Severity.CRITICAL else 0.3, signature=sha(code, desc[:80]), ctx=ctx)

    def _mirror(self, ctx: QCContext, i: QCIssue) -> QCIssue:
        return self._make(ctx, MIRROR[i.code], i.severity, i.title, i.description, i.suggested_fix, affected=i.affected_elements)

    def _graph_failed(self, ctx: QCContext, out: CheckerOutput, exc: RenderError, known: set[str], asset_missing: bool) -> None:
        if exc.kind == "missing_media" and asset_missing:
            out.notes.append("missing media: reported by the asset checker")
            out.metrics["blocked_by"] = "missing media"
            return
        if exc.kind in ("ffmpeg_missing", "ffmpeg_incapable") and known & {"preflight.ffmpeg", "preflight.ffmpeg_features"}:
            return
        if exc.kind == "invalid_timeline" and known & {"preflight.no_timeline", "preflight.document"}:
            return
        out.issues.append(self._make(ctx, "render.graph_failed", Severity.CRITICAL, "The render cannot be prepared", f"{exc.user_message}" + (f" ({exc.possible_issue})" if exc.possible_issue else ""),
                                     exc.possible_issue or "Fix the problem named above, then run QC again.", affected=[str(a) for a in exc.asset_ids][:10]))

    def _warnings(self, ctx: QCContext, out: CheckerOutput, warnings: list[str], known: set[str]) -> None:
        groups: dict[str, tuple[Severity, list[str]]] = {}
        for w in warnings:
            code, sev = _warning_kind(w)
            if code == "render.font_missing" and "preflight.fonts" in known:
                continue
            groups.setdefault(code, (sev, []))[1].append(w)
        titles = {"render.font_missing": "A font is not installed", "render.effect_unsupported": "Part of the edit cannot be rendered exactly", "render.transition_unsupported": "A transition cannot be rendered as designed",
                  "render.caption_unrenderable": "A caption cannot be rendered"}
        fixes = {"render.font_missing": "Install the font or pick another one; the export substitutes a similar font.", "render.effect_unsupported": "Change or remove the effect named above, or accept the simplified result.",
                 "render.transition_unsupported": "Pick another transition.", "render.caption_unrenderable": "Edit or regenerate the caption."}
        for code, (sev, lines) in groups.items():
            out.issues.append(self._make(ctx, code, sev, titles[code], f"{len(lines)} item(s): " + " | ".join(lines[:MAX_WARNING_LINES]) + (" ..." if len(lines) > MAX_WARNING_LINES else ""), fixes[code], affected=lines[:8]))


# ================================================================== post-render
class RenderedFileChecker(BaseChecker):
    id = "render.file"
    label = "Rendered file"
    categories = (QCCategory.RENDER_READINESS, QCCategory.FRAMES)
    version = "1"

    def inspect(self, ctx: QCContext, output_path: Path, *, render_id: str, expected: dict[str, Any], report: ProgressFn) -> dict[str, Any]:
        """The result stored in ``project.render_qc_results[render_id]``; plain JSON."""
        path = Path(output_path)
        checks: list[dict[str, Any]] = []
        issues: list[QCIssue] = []
        measured: dict[str, Any] = {}

        def add(cid: str, label: str, status: str, message: str = "", got: Any = None, want: Any = None) -> None:
            checks.append({"id": cid, "label": label, "status": status, "message": message, "measured": got, "expected": want})

        def issue(cid: str, sev: Severity, title: str, msg: str, **kw) -> None:
            issues.append(self.issue(f"postrender.{cid}", QCCategory.RENDER_READINESS, sev, title, description=msg, why="The exported file is what the viewer gets.", suggested_fix=kw.pop("fix", "Export again after fixing the cause."),
                                     viewer_impact=0.9 if sev in (Severity.CRITICAL, Severity.ERROR) else 0.4, signature=sha(render_id, cid), ctx=ctx, **kw))

        if not ffmpeg_ready(ctx):
            add("ffmpeg", "FFmpeg available", "fail", "FFmpeg is not available, so the file could not be inspected.")
            return self._result(render_id, path, checks, issues, measured)
        rs = ctx.project.render_settings
        w, h = expected.get("width"), expected.get("height")
        if not (w and h) or (w, h) == ctx.canvas:
            w, h = P.output_size(*ctx.canvas, rs.resolution)  # the export is scaled to the chosen resolution
        voice = speech_segments(ctx.words) if ctx.words else []
        exp = Expectations(float(expected.get("duration") or ctx.duration), int(w), int(h), float(expected.get("fps") or rs.fps or ctx.fps), rs.video_codec, rs.audio_codec, bool(expected.get("has_audio", True)),
                           int(rs.audio_sample_rate), [(a, b) for a, b in voice][:200], float(rs.duration_tolerance), True)
        report(0.1, "Checking the file's streams")
        rep = RenderValidator(ctx.ffmpeg, ctx.probe).validate(path, exp)  # type: ignore[arg-type]
        for c in rep.checks:
            st = {"ok": "pass", "warning": "warn", "error": "fail"}[c.status]
            add(c.id, c.label, st, c.message)
            if st != "pass":
                issue(c.id, Severity.ERROR if st == "fail" else Severity.WARNING, c.label + (" problem" if st == "fail" else " warning"), c.message or c.label)
        measured.update(rep.measured)
        measured.update({k: v for k, v in rep.probe.items() if k in ("duration", "width", "height", "fps", "codec", "audio_codec", "sample_rate", "channels")})
        if any(c["id"] in ("exists", "readable") and c["status"] == "fail" for c in checks):
            return self._result(render_id, path, checks, issues, measured)
        ctx.check_cancel()
        dur = float(measured.get("duration") or 0.0)
        report(0.3, "Looking for black and frozen frames")
        self._frames(ctx, path, dur, checks, issues, measured, add)
        ctx.check_cancel()
        if rep.probe.get("has_audio"):
            report(0.6, "Checking silence and loudness")
            self._audio(ctx, path, voice, checks, issues, measured, add, issue)
        ctx.check_cancel()
        report(0.85, "Decoding the whole file to look for corruption")
        self._decode(ctx, path, add, issue)
        report(1.0, "Rendered file checked")
        return self._result(render_id, path, checks, issues, measured)

    # ------------------------------------------------------------------ parts
    def _frames(self, ctx: QCContext, path: Path, dur: float, checks, issues, measured, add) -> None:
        cfg = ctx.settings.frames
        sc = scan_video(ctx, path, 0.0, dur, black_min=cfg.black_min_seconds, black_sens=cfg.black_sensitivity, freeze_min=cfg.frozen_min_seconds, freeze_sens=cfg.frozen_sensitivity)
        black = [(a, b) for a, b in sc.black if not ctx.in_intentional_black(a, b)]
        measured["black_seconds"] = round(sum(b - a for a, b in black), 2)
        measured["frozen_seconds"] = round(sum(b - a for a, b in sc.frozen), 2)
        if black:
            a, b = black[0]
            add("black_frames", "Black frames", "fail" if any(y - x > 1.0 for x, y in black) else "warn", f"{len(black)} black stretch(es), first at {_t(a)} ({b - a:.1f} s).", [list(x) for x in black[:5]], "none")
            issues.append(self.issue("frames.black", QCCategory.FRAMES, Severity.ERROR if any(y - x > 1.0 for x, y in black) else Severity.WARNING, "Black frames in the rendered video",
                                     description=f"The exported file is black at {_t(a)}-{_t(b)} ({len(black)} stretch(es) in total).", start=a, end=b, why="The viewer sees a blank screen.", current=f"{b - a:.1f} s black", recommended="none unless intentional",
                                     suggested_fix="Find the cause on the timeline (a gap or a black source) and export again.", viewer_impact=0.8, signature=sha("rendered-black", round(a, 1)), ctx=ctx))
        else:
            add("black_frames", "Black frames", "pass", "no black stretches", [], "none")
        if sc.frozen:
            a, b = sc.frozen[0]
            add("frozen_frames", "Frozen frames", "warn", f"{len(sc.frozen)} frozen stretch(es), first at {_t(a)} ({b - a:.1f} s).", [list(x) for x in sc.frozen[:5]], "none")
            issues.append(self.issue("frames.frozen", QCCategory.FRAMES, Severity.WARNING, "Frozen picture in the rendered video", description=f"The picture does not change at {_t(a)}-{_t(b)}.", start=a, end=b,
                                     why="A long unchanged picture can look like a stall.", current=f"{b - a:.1f} s", recommended="moving or deliberate", suggested_fix="Check whether the hold is intended.", confidence=80.0, viewer_impact=0.35,
                                     signature=sha("rendered-frozen", round(a, 1)), ctx=ctx))
        else:
            add("frozen_frames", "Frozen frames", "pass", "no frozen stretches", [], "none")

    def _audio(self, ctx: QCContext, path: Path, voice, checks, issues, measured, add, issue) -> None:
        rc, err, _ = _ff(ctx, ["-i", str(path), "-vn", "-af", f"silencedetect=n={SILENCE_DB:g}dB:d={SILENCE_MIN:g},ebur128=peak=true", "-f", "null", "-"], 600)
        silences, start = [], None
        for line in err.splitlines():
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                start = max(0.0, float(m.group(1)))
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m and start is not None:
                silences.append((start, float(m.group(1))))
                start = None
        if start is not None:
            silences.append((start, float(measured.get("duration") or start)))
        bad = [(a, b, sum(max(0.0, min(b, y) - max(a, x)) for x, y in voice)) for a, b in silences]
        bad = [(a, b, ov) for a, b, ov in bad if ov >= ctx.settings.audio.accidental_gap_seconds]
        if bad:
            a, b, ov = bad[0]
            add("unexpected_silence", "Silence under the voice", "fail", f"The audio is silent at {_t(a)}-{_t(b)} where the transcript has speech.", [[x, y] for x, y, _ in bad[:5]], "speech audible")
            issue("unexpected_silence", Severity.ERROR, "Silence where the voice should be", f"The exported audio is silent for {b - a:.1f} s around {_t(a)}, but the transcript has speech for {ov:.1f} s of it.", start=a, end=b)
        else:
            add("unexpected_silence", "Silence under the voice", "pass", "no silence under the narration", [], "none")
        tail = err[err.rfind("Summary:"):] if "Summary:" in err else ""
        mi = re.search(r"I:\s+(-?[\d.]+|-inf)\s+LUFS", tail)
        mp = re.search(r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS", tail)
        lufs = float(mi.group(1)) if mi and mi.group(1) != "-inf" else None
        peak = float(mp.group(1)) if mp and mp.group(1) != "-inf" else None
        measured.update(lufs=lufs, true_peak_dbfs=peak)
        cfg = ctx.settings.audio
        if peak is not None and peak >= cfg.clip_dbfs:
            add("loudness", "Final loudness", "warn", f"True peak {peak:.1f} dBFS reaches full scale (integrated {lufs if lufs is not None else '?'} LUFS).", {"lufs": lufs, "true_peak": peak}, f"true peak below {cfg.clip_dbfs:.1f} dBFS")
            issue("true_peak", Severity.WARNING, "The final audio reaches full scale", f"True peak {peak:.1f} dBFS (limit {cfg.clip_dbfs:.1f}); the sound may clip on some players.")
        elif lufs is None:
            add("loudness", "Final loudness", "fail", "No measurable audio level: the audio is silent.", {"lufs": lufs, "true_peak": peak}, "audible audio")
            issue("silent_audio", Severity.ERROR, "The exported audio is silent", "Loudness measurement found no audio signal.")
        else:
            add("loudness", "Final loudness", "pass", f"integrated {lufs:.1f} LUFS, true peak {peak if peak is not None else '?'} dBFS (reported only: no fixed target)", {"lufs": lufs, "true_peak": peak}, "report only")

    def _decode(self, ctx: QCContext, path: Path, add, issue) -> None:
        rc, err, _ = _ff(ctx, ["-v", "error", "-i", str(path), "-f", "null", "-"], 1800, loglevel=None)
        lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
        if rc != 0 or lines:
            add("corruption", "File decodes cleanly", "fail", (lines[0] if lines else f"ffmpeg exited with code {rc}")[:200], len(lines), 0)
            issue("corruption", Severity.ERROR, "The file reports decoding errors", f"A full decode of the rendered file reported {len(lines) or 1} problem(s): {(lines[0] if lines else 'decoder failure')[:160]}")
        else:
            add("corruption", "File decodes cleanly", "pass", "no decoding errors", 0, 0)

    def _result(self, render_id: str, path: Path, checks, issues: list[QCIssue], measured) -> dict[str, Any]:
        fails = [c for c in checks if c["status"] == "fail"]
        warns = [c for c in checks if c["status"] == "warn"]
        status = "FAILED" if fails else "WARNINGS" if warns else "PASSED"
        summary = ("The rendered file failed " + str(len(fails)) + " check(s): " + "; ".join(c["label"] for c in fails[:4]) + ".") if fails else \
                  (f"The rendered file passed with {len(warns)} warning(s): " + "; ".join(c["label"] for c in warns[:4]) + ".") if warns else "The rendered file passed every check."
        return {"render_id": render_id, "path": str(path), "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": status, "summary": summary, "checks": checks,
                "issues": [i.to_dict() for i in issues], "measured": measured}


def _ff(ctx: QCContext, args: list[str], timeout: float, loglevel: str | None = "info") -> tuple[int, str, bytes]:
    from app.qc.frame_checker import _run  # noqa: PLC0415 - the cancel-aware runner lives with the detectors

    base = [ctx.ffmpeg.ffmpeg(), "-hide_banner", "-nostdin"] + (["-v", loglevel] if loglevel else [])  # type: ignore[union-attr]
    return _run(ctx, base + args, timeout)


def _t(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"

