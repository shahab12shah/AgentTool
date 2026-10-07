"""RenderEngine: the facade over planner, compiler, command builder, FFmpeg service, executor, validator and diagnostics.

``RenderPlanner`` (planner.py) -> ``TimelineCompiler`` (compiler.py) -> ``FFmpegCommandBuilder`` (builder.py) -> ``RenderExecutor`` (executor.py)
-> ``RenderValidator`` (validator.py) -> ``OutputManager`` (output.py). The engine owns no project state: it renders a ``RenderSnapshot``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.core.constants import APP_NAME, APP_VERSION
from app.project.project_schema import RenderSettings
from app.rendering.diagnostics import RenderDiagnostics
from app.rendering.errors import RenderCancelled, RenderError
from app.rendering.executor import JobSpec, RenderExecutor, RenderLog, RenderResult
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.fonts import FontResolver
from app.rendering.models import RenderProgress, RenderSnapshot
from app.rendering.output import OutputManager
from app.rendering.planner import EncoderSelector, RenderPlanner
from app.rendering.presets import ResolvedOutput
from app.rendering.probe import MediaProbeService
from app.rendering.validator import RenderValidator


class RenderEngine:
    def __init__(self, ffmpeg_path: Callable[[], str] = lambda: "", ffprobe_path: Callable[[], str] = lambda: "", fallback_font: str = "") -> None:
        self.ffmpeg = FFmpegService(ffmpeg_path, ffprobe_path)
        self.probe = MediaProbeService(self.ffmpeg)
        self.fonts = FontResolver(fallback_font)
        self.selector = EncoderSelector(self.ffmpeg)
        self.planner = RenderPlanner(self.selector)
        self.validator = RenderValidator(self.ffmpeg, self.probe)
        self.executor = RenderExecutor(self.ffmpeg, self.probe, self.fonts, self.validator)

    def outputs(self, default_dir: Path) -> OutputManager:
        return OutputManager(default_dir)

    def diagnostics(self, default_dir: Path) -> RenderDiagnostics:
        return RenderDiagnostics(self.ffmpeg, self.probe, self.fonts, self.selector, self.outputs(default_dir))

    def resolve(self, settings: RenderSettings, snapshot: RenderSnapshot, force_cpu: bool = False) -> ResolvedOutput:
        return self.planner.resolve(settings, snapshot, force_cpu)

    def reconfigure(self) -> None:
        """Settings changed (FFmpeg path): forget what was detected."""
        self.ffmpeg.forget()

    # ------------------------------------------------------------------ run
    def run(self, spec: JobSpec, cancel: threading.Event, on_progress: Callable[[RenderProgress], None], checkpoint: Callable[[], None] | None = None) -> RenderResult:
        """Render ``spec.snapshot`` to ``spec.output_path``. Writes ``render.log`` and ``snapshot.json`` into ``spec.run_dir``; always cleans the working folder."""
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        log = RenderLog(spec.run_dir / "render.log")
        snap = spec.snapshot
        try:
            (spec.run_dir / "snapshot.json").write_text(json.dumps(snap.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")
        except OSError:
            pass
        started = datetime.now()
        try:
            log.line(f"{APP_NAME} {APP_VERSION} — render {spec.render_id} ({spec.kind})")
            log.line(f"start: {started.isoformat(timespec='seconds')}")
            log.line(f"project: {snap.project_name} ({snap.project_id}) timeline_version={snap.timeline_version} content={snap.content_hash()} snapshot={snap.snapshot_id}")
            log.line(f"settings: {snap.settings.to_dict()}")
            log.line(f"output: {spec.resolved.summary()} -> {spec.output_path}")
            try:
                result = self.executor.execute(spec, cancel, on_progress, log, checkpoint)
            except RenderCancelled:
                log.line("exit status: CANCELED")
                raise
            except RenderError as exc:
                log.error(f"stage={exc.stage} kind={exc.kind}: {exc.user_message} {exc.details or ''}")
                log.line(f"exit status: FAILED ({exc.stage})")
                raise
            except Exception as exc:
                log.error(f"unexpected error: {type(exc).__name__}: {exc}")
                log.line("exit status: FAILED (unexpected)")
                raise
            log.line(f"exit status: COMPLETED in {result.elapsed:.1f}s ({result.rendered_chunks} section(s) rendered, {result.cached_chunks} reused)")
            return result
        finally:
            log.line(f"end: {datetime.now().isoformat(timespec='seconds')}")
            log.close()
