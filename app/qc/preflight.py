"""Pre-flight: the deterministic checks that run before any expensive analysis (spec 6).

It answers "is this project sound enough to analyse and render at all?". It reports what only a whole-project view can see (an inconsistent document, no timeline,
no picture, no voice-over, settings that cannot be exported, a missing encoder, no disk space) and sets ``metrics["integrity_ok"]``: when that is False the engine
skips the expensive checkers, because analysing a broken project only produces noise. Per-asset findings (missing / corrupt files) are reported by the asset
checker; pre-flight only uses them to decide integrity, so nothing is reported twice. Everything reusable from Phase 6 (RenderDiagnostics, presets) is reused.
"""

from __future__ import annotations

import math
from pathlib import Path

from app.core.exceptions import AppError, InvalidProjectError
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory
from app.qc.media_facts import ffmpeg_ready, probe_used, used_assets
from app.qc.severity import Severity
from app.rendering import presets as P
from app.rendering.diagnostics import RenderDiagnostics
from app.rendering.fonts import FontResolver
from app.rendering.models import ProxyRef, RenderSnapshot, disk_free
from app.rendering.output import OutputManager
from app.rendering.planner import EncoderSelector
from app.rendering.probe import MediaProbeService
from app.rendering.sources import proxy_usable
from app.timeline.track import TrackKind

# Render-diagnostics items that describe the *environment and output* (the rest - timeline, voice, visuals, media, audio, captions - are covered by the QC checkers
# that own those topics, with locations and fixes, so they are not repeated here).
ITEM_CODES = {"ffmpeg": "preflight.ffmpeg", "filters": "preflight.ffmpeg_features", "settings": "preflight.encoder", "output": "preflight.output", "disk": "preflight.disk",
              "fonts": "preflight.fonts"}
DISK_BUCKET = 256 * 1024 * 1024  # free space is part of the cache key, rounded so a few MB of churn does not invalidate the result
MAX_DIM, MIN_DIM, MAX_FPS = 8192, 16, 240


class PreflightChecker(BaseChecker):
    id = "preflight"
    label = "Pre-flight"
    categories = (QCCategory.PREFLIGHT,)
    domains = ("timeline", "scenes", "assets", "render")
    settings_sections = ("media",)
    scene_local = False
    expensive = False
    version = "1"

    # ------------------------------------------------------------------ cache key
    def input_hash(self, ctx: QCContext) -> str:
        """The disk and the FFmpeg install are outside the project, but they decide the verdict: they are part of the key."""
        free = disk_free(self._output_dir(ctx)) if ctx.root else -1
        return sha(super().input_hash(ctx), ffmpeg_ready(ctx), free // DISK_BUCKET if free >= 0 else -1)

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        blocking_integrity: list[str] = []  # reasons deeper analysis would be meaningless
        report(0.05, "Checking the project document")
        self._document(ctx, out, blocking_integrity)
        self._structure(ctx, out, blocking_integrity)
        self._project_settings(ctx, out, blocking_integrity)
        ctx.check_cancel()
        report(0.3, "Checking the media files")
        self._media(ctx, out, blocking_integrity, report)
        ctx.check_cancel()
        report(0.7, "Checking the export setup")
        self._export(ctx, out, blocking_integrity)
        report(1.0, "Pre-flight complete")
        blocking = sum(1 for i in out.issues if i.severity is Severity.CRITICAL)
        out.metrics = {"integrity_ok": not blocking_integrity, "checks": 9, "blocking": blocking, "integrity_reasons": blocking_integrity}
        out.notes.append("integrity OK" if not blocking_integrity else "integrity failed: " + "; ".join(blocking_integrity) + " - expensive analysis skipped")
        return out

    # ------------------------------------------------------------------ 1. the document
    def _document(self, ctx: QCContext, out: CheckerOutput, integrity: list[str]) -> None:
        try:
            ctx.project.validate()
            return
        except InvalidProjectError as exc:
            problems = list(exc.problems) or [exc.user_message]
        shown = problems[:8]
        out.issues.append(self.issue(
            "preflight.document", QCCategory.PREFLIGHT, Severity.CRITICAL, "The project data is inconsistent",
            description=f"{len(problems)} integrity problem(s): " + "; ".join(shown) + (f"; and {len(problems) - len(shown)} more" if len(problems) > len(shown) else ""),
            affected=shown, why="Inconsistent references or time ranges make rendering and every later check unreliable.", current=f"{len(problems)} problem(s)", recommended="none",
            suggested_fix="Fix or remove the listed elements on the timeline (QC does not edit broken project data automatically).", viewer_impact=1.0,
            signature=sha(sorted(problems)), ctx=ctx))
        integrity.append("project data is inconsistent")

    # ------------------------------------------------------------------ 2. timeline / voice-over existence
    def _structure(self, ctx: QCContext, out: CheckerOutput, integrity: list[str]) -> None:
        tl = ctx.timeline
        total = sum(len(t.clips) for t in tl.tracks)
        duration = tl.duration
        if total == 0 or not math.isfinite(duration) or duration <= 0:
            out.issues.append(self.issue(
                "preflight.no_timeline", QCCategory.PREFLIGHT, Severity.CRITICAL, "The timeline is empty" if total == 0 else "The timeline has no valid duration",
                description="There is nothing to render." if total == 0 else f"The timeline duration is {duration!r}.", why="An empty timeline cannot be exported.", current=f"{total} clip(s)",
                recommended="at least one visual clip", suggested_fix="Generate the AI edit or add clips to the timeline.", viewer_impact=1.0, signature="empty", ctx=ctx))
            integrity.append("the timeline is empty")
        elif not any(c.kind == "media" for t, c in ctx.clips(track_kinds=(TrackKind.VIDEO, TrackKind.IMAGE))):
            out.issues.append(self.issue(
                "preflight.no_visuals", QCCategory.PREFLIGHT, Severity.CRITICAL, "There is no picture on the timeline",
                description="No video or image track has a clip, so the export would be a black screen.", why="A video without visuals cannot be published.", current="0 visual clips",
                recommended="visual clips covering the narration", suggested_fix="Run the AI edit or add visuals on a video / image track.", viewer_impact=1.0, signature="no-visuals", ctx=ctx))
            integrity.append("no visuals on the timeline")
        vo = ctx.project.voice_over
        if not vo.asset_id or ctx.asset(vo.asset_id) is None:
            out.issues.append(self.issue(
                "preflight.no_voice", QCCategory.PREFLIGHT, Severity.CRITICAL, "No voice-over",
                description="The project has no voice-over, so there is no narration to synchronise the edit with.", why="Narration is the master clock of the whole edit.", current="no voice-over",
                recommended="an imported voice-over", suggested_fix="Import or record the voice-over on the Voice page.", viewer_impact=1.0, signature="no-voice", ctx=ctx))
            integrity.append("no voice-over")
        elif not vo.duration or vo.duration <= 0 or not math.isfinite(vo.duration):
            out.issues.append(self.issue(
                "preflight.voice_duration", QCCategory.PREFLIGHT, Severity.CRITICAL, "The voice-over has no valid duration", description=f"Recorded voice-over duration: {vo.duration!r}.",
                why="Scene timing and sync checks are measured against the voice-over length.", current=str(vo.duration), recommended="a positive duration", suggested_fix="Re-import the voice-over.",
                viewer_impact=1.0, signature="voice-duration", ctx=ctx))
            integrity.append("the voice-over has no valid duration")

    # ------------------------------------------------------------------ 3. project settings
    def _project_settings(self, ctx: QCContext, out: CheckerOutput, integrity: list[str]) -> None:
        s = ctx.project.settings
        bad = []
        if not (MIN_DIM <= int(s.width) <= MAX_DIM and MIN_DIM <= int(s.height) <= MAX_DIM):
            bad.append(f"resolution {s.width}x{s.height} (allowed {MIN_DIM}-{MAX_DIM} px per side)")
        elif int(s.width) % 2 or int(s.height) % 2:
            bad.append(f"resolution {s.width}x{s.height} has an odd side (H.264 / H.265 need even dimensions)")
        if not (1 <= int(s.fps) <= MAX_FPS):
            bad.append(f"frame rate {s.fps} (allowed 1-{MAX_FPS})")
        if not bad:
            return
        out.issues.append(self.issue(
            "preflight.project_settings", QCCategory.PREFLIGHT, Severity.CRITICAL, "Invalid project settings", description="; ".join(bad) + ".",
            affected=bad, why="Rendering and every time-based check depend on a valid canvas and frame rate.", current="; ".join(bad), recommended="a supported resolution and frame rate",
            suggested_fix="Change the project format in the project settings.", viewer_impact=1.0, signature=sha(bad), ctx=ctx))
        integrity.append("invalid project settings")

    # ------------------------------------------------------------------ 4. media readable / decodable (reported per asset by the asset checker)
    def _media(self, ctx: QCContext, out: CheckerOutput, integrity: list[str], report: ProgressFn) -> None:
        used = probe_used(ctx, lambda f, m: report(0.3 + 0.35 * f, m))
        missing = sorted(u.asset.name for u in used.values() if not u.exists)
        corrupt = sorted(u.asset.name for u in used.values() if u.exists and u.probed and u.info is None)
        if missing:
            integrity.append(f"{len(missing)} required media file(s) missing")
            out.notes.append("missing media: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        if corrupt:
            integrity.append(f"{len(corrupt)} media file(s) cannot be decoded")
            out.notes.append("undecodable media: " + ", ".join(corrupt[:5]) + (" ..." if len(corrupt) > 5 else ""))
        if used and not ffmpeg_ready(ctx):
            out.notes.append("media were not decoded: FFmpeg is not available")

    # ------------------------------------------------------------------ 5. export setup (codec, encoder, output location, disk, fonts, proxies)
    def _export(self, ctx: QCContext, out: CheckerOutput, integrity: list[str]) -> None:
        rs = ctx.project.render_settings
        compat = P.compatibility_problems(rs)
        for text in compat[:6]:
            out.issues.append(self.issue(
                "preflight.export_settings", QCCategory.PREFLIGHT, Severity.CRITICAL, "Invalid export settings", description=text, why="The renderer refuses settings it cannot honour.",
                current=text, recommended="a valid combination", suggested_fix="Open the Export page and pick supported settings.", viewer_impact=1.0, signature=text, ctx=ctx))
        self._proxies(ctx, out)
        if not ffmpeg_ready(ctx):
            if ctx.ffmpeg is not None:  # a context without any FFmpeg service (pure document tests) says nothing about the install
                out.issues.append(self.issue(
                    "preflight.ffmpeg", QCCategory.PREFLIGHT, Severity.CRITICAL, "FFmpeg is not available", description=ctx.ffmpeg.detect()[1] or "FFmpeg could not be started.",
                    why="Nothing can be decoded or rendered without FFmpeg.", current="not found", recommended="a working FFmpeg", suggested_fix="Install FFmpeg or set its location in Settings.",
                    viewer_impact=1.0, signature="ffmpeg", ctx=ctx))
                integrity.append("FFmpeg is not available")
            return
        try:
            report = self._diagnostics(ctx)
        except (AppError, OSError) as exc:
            out.notes.append(f"render diagnostics unavailable: {exc}")
            return
        for item in report.items:
            code = ITEM_CODES.get(item.id)
            if code is None or item.status == "ok":
                continue
            sev = Severity.CRITICAL if item.status == "error" else Severity.WARNING
            if code == "preflight.encoder" and compat:
                continue  # already explained by the setting that causes it
            out.issues.append(self.issue(
                code, QCCategory.PREFLIGHT, sev, _TITLES.get(item.id, item.label), description=" ".join([item.message] + [d for d in item.details[:3]]), affected=list(item.details[:8]),
                why="The export would fail or look different from the preview." if sev is Severity.CRITICAL else "The export will use a fallback.", current=item.message,
                recommended="resolved", suggested_fix=item.fix, viewer_impact=1.0 if sev is Severity.CRITICAL else 0.3, signature=sha(item.id, item.message), ctx=ctx))

    def _diagnostics(self, ctx: QCContext):
        proj = ctx.project
        snap = RenderSnapshot.from_project(proj, proj.render_settings, self._proxy_refs(ctx))
        diag = RenderDiagnostics(ctx.ffmpeg, ctx.probe or MediaProbeService(ctx.ffmpeg), FontResolver(""), EncoderSelector(ctx.ffmpeg), OutputManager(self._output_dir(ctx)))
        cache = (proj.root / "cache" / "render") if proj.root else None
        return diag.run(snap, None, None, cache, check_hardware=False)  # hardware encoders are an optimisation: probing them would test-encode

    @staticmethod
    def _output_dir(ctx: QCContext) -> Path:
        rs = ctx.project.render_settings
        return Path(rs.output_dir) if rs.output_dir else (ctx.project.root or Path(".")) / "renders"

    @staticmethod
    def _proxy_refs(ctx: QCContext) -> dict[str, ProxyRef]:
        refs: dict[str, ProxyRef] = {}
        for aid, d in ctx.project.proxies.items():
            ok = d.get("proxy_status") == "READY" and bool(d.get("proxy_path")) and Path(str(d.get("proxy_path"))).is_file()
            refs[aid] = ProxyRef(str(d.get("proxy_path", "")), str(d.get("proxy_resolution", "")), d.get("proxy_status", "NONE") if ok else "STALE", int(d.get("width", 0) or 0),
                                 int(d.get("height", 0) or 0), int(d.get("source_size", 0) or 0), int(d.get("source_mtime_ns", 0) or 0))
        return refs

    def _proxies(self, ctx: QCContext, out: CheckerOutput) -> None:
        """With 'use proxies' on, every visual should have a current proxy; stale or missing ones fall back to the originals (slow, not wrong)."""
        if not ctx.project.render_settings.use_proxies:
            return
        refs = self._proxy_refs(ctx)
        names = sorted(u.asset.name for u in used_assets(ctx).values() if u.role == "visual" and u.exists and not proxy_usable(refs.get(u.asset.id)))
        if names:
            out.issues.append(self.issue(
                "preflight.proxy", QCCategory.PREFLIGHT, Severity.WARNING, "Proxy media is stale or missing", description=f"{len(names)} visual(s) have no current proxy: " + ", ".join(names[:5]),
                affected=names[:8], why="The export is set to use proxies, but these will be rendered from the (slower) originals or at lower quality.", current=f"{len(names)} without proxy",
                recommended="regenerated proxies", suggested_fix="Regenerate the proxies on the Export page, or turn 'use proxies' off.", viewer_impact=0.2, signature=sha(names), ctx=ctx))


_TITLES = {"ffmpeg": "FFmpeg is not available", "filters": "FFmpeg lacks required features", "settings": "The export encoder is not available", "output": "The output location is not usable",
           "disk": "Not enough disk space", "fonts": "Some fonts are not installed"}
