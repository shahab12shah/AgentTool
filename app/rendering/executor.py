"""RenderExecutor: runs a render plan stage by stage and reports honest progress.

Pipeline: validate -> prepare media -> compile timeline -> build graphs -> render video windows (each cached by content) -> join ->
mix audio (cached) -> encode/mux -> validate the output file -> move it into place. The timeline snapshot is the only input; the
project is never touched, and a failure or cancellation leaves only logs behind (temporary media is always removed).
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.logging.logger import get_logger
from app.rendering.builder import BuiltCommand, FFmpegCommandBuilder
from app.rendering.compiler import TimelineCompiler
from app.rendering.errors import RenderCancelled, RenderError
from app.rendering.ffmpeg_service import ExecResult, FFmpegService, ProgressInfo, redact
from app.rendering.fonts import FontResolver
from app.rendering.models import RenderPlan, RenderProgress, RenderSnapshot, RenderStage, has_audio_clips
from app.rendering.output import OutputManager
from app.rendering.planner import make_plan, plan_chunks
from app.rendering.presets import ResolvedOutput
from app.rendering.probe import MediaProbeService
from app.rendering.sources import MissingMediaError, SourceSelector, prepare_assets
from app.rendering.validator import Expectations, RenderValidator, ValidationReport
from app.timeline.track import TrackKind

_log = get_logger(__name__)
REQUIRED_FILTERS = ("color", "overlay", "scale", "fps", "trim", "setpts", "format", "crop", "rotate", "fade", "alphamerge", "geq", "blend", "tpad", "gblur", "colorchannelmixer",
                    "amix", "aresample", "aformat", "volume", "afade", "adelay", "atrim", "apad", "alimiter", "pan", "atempo", "asetpts")
WEIGHTS_AUDIO = {"video": 0.80, "audio": 0.08, "encode": 0.10, "validate": 0.02}
WEIGHTS_SILENT = {"video": 0.88, "audio": 0.0, "encode": 0.10, "validate": 0.02}


@dataclass
class JobSpec:
    render_id: str
    snapshot: RenderSnapshot
    resolved: ResolvedOutput
    output_path: Path
    work_dir: Path
    cache_dir: Path
    run_dir: Path
    kind: str = "export"  # export | draft | preview
    mode: str = "final"  # final | preview  (which media the compiler reads)
    allow_proxy_assets: set[str] = field(default_factory=set)
    chunk_seconds: float = 30.0
    chunk_max_seconds: float | None = None
    use_cache: bool = True
    duration_tolerance: float = 0.5
    chunk_ranges: list[tuple[float, float]] | None = None  # preview: render only these windows
    encode_args: list[str] | None = None  # override the video encoder arguments (previews use fast settings)


@dataclass
class Prepared:
    snap: RenderSnapshot
    out: ResolvedOutput
    selector: SourceSelector
    compiler: TimelineCompiler
    builder: FFmpegCommandBuilder
    chunks: list
    compiled: list
    cmds: list
    aplan: object
    amix: BuiltCommand | None
    plan: RenderPlan
    total_seconds: float


@dataclass
class RenderResult:
    output_path: Path
    report: ValidationReport
    plan: RenderPlan
    duration: float
    size_bytes: int
    cached_chunks: int
    rendered_chunks: int
    warnings: list[str]
    used_proxy_assets: list[str]
    elapsed: float = 0.0


class AnyEvent:
    """Looks like a ``threading.Event`` that is set when any of the given events is (what FFmpegService polls for cancellation)."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)


class NullLog:
    """A log that discards everything (used when only planning, e.g. to compute preview cache keys)."""

    def line(self, text: str, level: str = "INFO") -> None: ...
    def warn(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...
    def close(self) -> None: ...


class RenderLog:
    """``render.log``: start/end, project version, FFmpeg version, settings, errors, warnings, command metadata and the exit status."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def line(self, text: str, level: str = "INFO") -> None:
        with self._lock:
            self._fh.write(f"{datetime.now().isoformat(timespec='milliseconds')} {level:<7} {redact(text)}\n")
            self._fh.flush()

    def warn(self, text: str) -> None:
        self.line(text, "WARNING")

    def error(self, text: str) -> None:
        self.line(text, "ERROR")

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass


def classify_failure(stderr: str, hardware: bool) -> tuple[str, str, bool]:
    """(kind, likely cause, hardware-fallback offered) from FFmpeg's error text."""
    t = stderr.lower()
    if "no space left" in t or "disk full" in t:
        return "disk_full", "The disk is full.", False
    if "no such file or directory" in t or "does not exist" in t:
        return "missing_media", "A source file or folder is missing.", False
    if hardware and any(k in t for k in ("nvenc", "cuda", "qsv", "amf", "videotoolbox", "openencodesession", "no capable devices", "device creation failed", "cannot load")):
        return "hardware_failed", "The hardware encoder failed (driver, GPU or session limit).", True
    if "unknown encoder" in t or "encoder not found" in t or "unknown decoder" in t or "decoder (codec" in t:
        return "unsupported_codec", "A codec is not supported by this FFmpeg build.", False
    if "invalid data found" in t or "moov atom not found" in t or "error while decoding" in t:
        return "corrupt_media", "A source file looks damaged or uses an unsupported codec.", False
    if "permission denied" in t:
        return "invalid_path", "A file or folder cannot be written (permission denied).", False
    if "error parsing" in t or "no such filter" in t or "invalid argument" in t and "filter" in t:
        return "ffmpeg_failed", "The filter graph was rejected by FFmpeg.", hardware
    return "ffmpeg_failed", "FFmpeg stopped with an error.", hardware


class Tracker:
    """Turns real FFmpeg progress into ``RenderProgress``. Nothing is invented: every number comes from an FFmpeg ``out_time`` or a finished step."""

    def __init__(self, snapshot: RenderSnapshot, has_audio: bool, fps: int, total_frames: int, emit: Callable[[RenderProgress], None]) -> None:
        self.snap, self.emit, self.fps, self.total_frames = snapshot, emit, fps, max(1, total_frames)
        self.w = WEIGHTS_AUDIO if has_audio else WEIGHTS_SILENT
        self.p = RenderProgress(scene_total=len(snapshot.scenes))
        self.t0 = time.monotonic()
        self.frames_done = 0.0  # frames of finished + running chunks (cached ones included)
        self.rendered_frames = 0.0  # frames actually produced by FFmpeg in this run
        self.render_started: float | None = None
        self.uncached_total = self.total_frames
        self.audio = self.encode = self.validate = 0.0

    def stage(self, st: RenderStage, message: str = "") -> None:
        self.p.stage = st
        self.p.message = message or st.value
        self._push()

    def video(self, frames_in_chunk: float, chunk_start_t: float, chunk_index: int, chunk_total: int, speed: float | None, out_bytes: int, *, cached: bool = False,
              base_frames: float = 0.0) -> None:
        self.frames_done = base_frames + frames_in_chunk
        if not cached:
            if self.render_started is None:
                self.render_started = time.monotonic()
        self.p.video = min(1.0, self.frames_done / self.total_frames)
        t = chunk_start_t + frames_in_chunk / self.fps
        self.p.scene_index = next((i + 1 for i, s in enumerate(self.snap.scenes) if s.start <= t < s.end), self.p.scene_index)
        self.p.chunk_index, self.p.chunk_total = chunk_index, chunk_total
        self.p.speed, self.p.output_bytes = speed, out_bytes
        self._push()

    def note_rendered(self, frames: float) -> None:
        self.rendered_frames = frames

    def set(self, which: str, frac: float, speed: float | None = None, out_bytes: int = 0) -> None:
        setattr(self, which, max(0.0, min(1.0, frac)))
        if which == "audio":
            self.p.audio = self.audio
        if which == "encode":
            self.p.encode = self.encode
            self.p.output_bytes = out_bytes or self.p.output_bytes
        if speed is not None:
            self.p.speed = speed
        self._push()

    def _push(self) -> None:
        p = self.p
        p.overall = max(0.0, min(1.0, p.video * self.w["video"] + self.audio * self.w["audio"] + self.encode * self.w["encode"] + self.validate * self.w["validate"]))
        p.elapsed = time.monotonic() - self.t0
        p.eta = None
        if p.stage is RenderStage.RENDERING and self.render_started and self.rendered_frames > self.fps * 0.5:
            rate = self.rendered_frames / max(1e-3, time.monotonic() - self.render_started)
            remaining = max(0.0, self.uncached_total - self.rendered_frames)
            p.eta = remaining / rate if rate > 0 else None
        elif p.overall > 0.05 and p.stage in (RenderStage.ENCODING, RenderStage.VALIDATING_OUTPUT):
            p.eta = max(0.0, p.elapsed * (1 - p.overall) / p.overall)
        self.emit(p)


class RenderExecutor:
    def __init__(self, ffmpeg: FFmpegService, probe: MediaProbeService, fonts: FontResolver, validator: RenderValidator) -> None:
        self.ff, self.probe, self.fonts, self.validator = ffmpeg, probe, fonts, validator

    # ------------------------------------------------------------------ entry point
    def execute(self, spec: JobSpec, cancel: threading.Event, on_progress: Callable[[RenderProgress], None], log: RenderLog,
                checkpoint: Callable[[], None] | None = None) -> RenderResult:
        started = time.monotonic()
        snap, out = spec.snapshot, spec.resolved
        stage = RenderStage.VALIDATING
        try:
            spec.work_dir.mkdir(parents=True, exist_ok=True)
            tracker = Tracker(snap, has_audio_clips(snap), out.fps, max(1, int(round(snap.duration * out.fps))), on_progress)
            prep = self.prepare(spec, cancel, log, tracker)
            # ---- rendering video sections (each cached by content); the audio mix is independent of them and runs alongside
            stage = RenderStage.RENDERING
            tracker.stage(stage, f"Rendering {len(prep.cmds)} section(s)")
            stop_audio = threading.Event()
            audio_cancel = AnyEvent(cancel, stop_audio)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-mix") as pool:
                audio_future = pool.submit(self.render_audio, prep, spec, audio_cancel, log, tracker, False)
                try:
                    made, cached, rendered = self.render_sections(prep, spec, cancel, log, tracker, checkpoint)
                except BaseException:
                    stop_audio.set()  # the video failed or was cancelled: do not keep mixing audio
                    raise
                audio_src = audio_future.result()
            self._check_cancel(cancel)
            # ---- join
            builder, total_seconds = prep.builder, prep.total_seconds
            video_src = made[0]
            if len(made) > 1:
                join = builder.concat(made, "chunks.txt", spec.work_dir / f"video.{out.segment_ext}", total_seconds)
                (spec.work_dir / "chunks.txt").write_text(join.files["chunks.txt"], encoding="utf-8")
                log.line(f"{join.description}: {join.printable()}")
                tracker.stage(RenderStage.RENDERING, "Joining sections")
                res = self.ff.run_progress(join.args, cancel=cancel, on_line=lambda ln: self._ffmpeg_line(log, ln), cwd=spec.work_dir)
                self._check_result(res, "join", stage, out.hardware, cancel, log, join, spec.work_dir / f"video.{out.segment_ext}")
                video_src = spec.work_dir / f"video.{out.segment_ext}"
            # ---- encode / mux
            stage = RenderStage.ENCODING
            tracker.stage(stage, "Encoding audio and writing the file")
            final_tmp = spec.work_dir / f"final.{out.container}"
            mux = builder.mux(video_src, audio_src, final_tmp, total_seconds)
            log.line(f"{mux.description}: {mux.printable()}")
            res = self.ff.run_progress(mux.args, on_progress=lambda pi: tracker.set("encode", pi.out_time / max(0.1, total_seconds), pi.speed, pi.total_size), cancel=cancel,
                                       on_line=lambda ln: self._ffmpeg_line(log, ln), cwd=spec.work_dir)
            self._check_result(res, "encode", stage, False, cancel, log, mux, final_tmp)
            tracker.set("encode", 1.0, out_bytes=final_tmp.stat().st_size)
            # ---- validate the file
            stage = RenderStage.VALIDATING_OUTPUT
            tracker.stage(stage)
            voice = [(c.timeline_start, c.timeline_end) for t in snap.tracks if t.kind is TrackKind.AUDIO and not t.muted for c in t.clips
                     if c.asset_id and c.asset_id == snap.voice_asset_id and c.audio.get("role", "VOICE") == "VOICE"]
            audible = any(it.gain > 1e-6 and not (it.keyframes and all(k.value <= 1e-6 for k in it.keyframes)) for it in prep.aplan.items)  # type: ignore[attr-defined]
            exp = Expectations(total_seconds, out.width, out.height, float(out.fps), out.video_codec, out.audio_codec, audio_src is not None, out.sample_rate, voice, spec.duration_tolerance, audible)
            report = self.validator.validate(final_tmp, exp)
            for c in report.checks:
                log.line(f"validate {c.id}: {c.status} {c.message}", "INFO" if c.status == "ok" else c.status.upper())
            if not report.ok:
                raise RenderError("The render finished but the file did not pass validation: " + report.summary(), stage=stage.value, kind="validation_failed",
                                  possible_issue="The output does not match the timeline.", details=report.summary())
            tracker.validate = 1.0
            # ---- finalize
            stage = RenderStage.FINALIZING
            tracker.stage(stage)
            OutputManager.commit(final_tmp, spec.output_path)
            report.probe["path"] = str(spec.output_path)
            size = spec.output_path.stat().st_size
            tracker.p.overall = 1.0
            tracker.emit(tracker.p)
            log.line(f"output: {spec.output_path} ({size} bytes)")
            self._prune_cache(spec.cache_dir, keep=set(made), limit_bytes=4 * 1024 ** 3)
            return RenderResult(spec.output_path, report, prep.plan, total_seconds, size, cached, rendered, prep.plan.warnings + [c.message for c in report.warnings],
                                prep.plan.uses_proxy_assets, time.monotonic() - started)
        except RenderCancelled:
            raise
        except RenderError as exc:
            if not exc.stage:
                exc.stage = stage.value
            raise
        except OSError as exc:
            kind = "disk_full" if getattr(exc, "errno", 0) == 28 else "ffmpeg_failed"
            raise RenderError(f"A file operation failed: {exc.strerror or exc}", stage=stage.value, kind=kind, details=str(exc)) from exc
        finally:
            self._cleanup(spec, log)

    # ------------------------------------------------------------------ building blocks (also used by the preview engine)
    def prepare(self, spec: JobSpec, cancel: threading.Event, log: RenderLog, tracker: Tracker | None = None) -> Prepared:
        """Validate, read the media facts, compile the timeline and build every command (nothing is run)."""
        snap, out = spec.snapshot, spec.resolved
        stage = RenderStage.VALIDATING
        try:
            if tracker:
                tracker.stage(stage)
            ok, msg = self.ff.detect()
            if not ok:
                raise RenderError(msg, stage=stage.value, kind="ffmpeg_missing", possible_issue="Install FFmpeg or set its location in Settings.")
            version = self.ff.version()
            log.line(f"FFmpeg: {version.text}")
            missing_f = self.ff.capabilities().has_filters(*REQUIRED_FILTERS)
            if missing_f:
                raise RenderError("This FFmpeg build lacks filters the renderer needs: " + ", ".join(missing_f) + ".", stage=stage.value, kind="ffmpeg_incapable",
                                  possible_issue="Install a full FFmpeg build (the 'full' or 'gpl' variant).")
            if snap.duration <= 0:
                raise RenderError("The timeline is empty.", stage=stage.value, kind="invalid_timeline", possible_issue="Add clips before exporting.")
            stage = RenderStage.PREPARING
            if tracker:
                tracker.stage(stage)
            missing = prepare_assets(snap, self.probe)
            selector = SourceSelector(snap, self.probe, spec.mode, spec.allow_proxy_assets, snap.settings.use_proxies)
            need = [a for a in missing if not (a.id in spec.allow_proxy_assets and snap.proxies.get(a.id) and snap.proxies[a.id].status == "READY")]
            if need:
                raise MissingMediaError(need, [a.id for a in missing if a.id in snap.proxies and snap.proxies[a.id].status == "READY"], stage.value)
            self._check_cancel(cancel)
            stage = RenderStage.COMPILING
            if tracker:
                tracker.stage(stage)
            compiler = TimelineCompiler(snap, selector, self.fonts, out.fps)
            chunks = plan_chunks(snap, out.fps, spec.chunk_seconds, compiler.media_clips_in, spec.chunk_max_seconds)
            if spec.chunk_ranges is not None:
                chunks = [c for c in chunks if any(c.start < b and c.end > a for a, b in spec.chunk_ranges)]
            plan = make_plan(snap, out, chunks)
            if tracker:
                tracker.total_frames = tracker.uncached_total = max(1, sum(c.frames for c in chunks))
            compiled = [compiler.compile_chunk(c) for c in chunks]
            plan.uses_proxy_assets = sorted(selector.used_proxies)
            plan.warnings += compiler.warnings + compiler.ass.warnings
            for w in plan.warnings:
                log.warn(w)
            log.line("PLAN " + str({k: v for k, v in plan.to_dict().items() if k not in ("chunks", "video_tracks", "audio_tracks")}))
            self._check_cancel(cancel)
            stage = RenderStage.VIDEO_GRAPH
            if tracker:
                tracker.stage(stage)
            flag = "-fps_mode" if version.at_least(5, 1) else "-vsync"
            builder = FFmpegCommandBuilder(self.ff.ffmpeg(), out, snap, compiler, flag)
            cmds = [builder.video_chunk(cc, spec.work_dir / f"c{cc.chunk.index}" / f"out.{out.segment_ext}", spec.encode_args) for cc in compiled]
            stage = RenderStage.AUDIO_GRAPH
            if tracker:
                tracker.stage(stage)
            aplan = compiler.audio_plan()
            missing_audio = [i.path for i in aplan.items if not Path(i.path).is_file()]
            if missing_audio:
                raise RenderError(f"Audio file missing: {Path(missing_audio[0]).name}", stage=stage.value, kind="missing_media", possible_issue="A music, SFX or voice-over file was moved or deleted.")
            amix = builder.audio_mix(aplan, spec.work_dir / "mix.flac")
            total_seconds = sum(c.chunk.frames for c in compiled) / out.fps
            return Prepared(snap, out, selector, compiler, builder, chunks, compiled, cmds, aplan, amix, plan, total_seconds)
        except RenderError as exc:
            if not exc.stage:
                exc.stage = stage.value
            raise

    def render_sections(self, prep: Prepared, spec: JobSpec, cancel: threading.Event, log: RenderLog, tracker: Tracker | None = None,
                        checkpoint: Callable[[], None] | None = None) -> tuple[list[Path], int, int]:
        """Render (or reuse from the cache) every video section. Returns (section files, reused count, rendered count)."""
        out = prep.out
        chunk_dir = spec.cache_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        cached = rendered = produced = 0
        done_frames = 0.0
        made: list[Path] = []
        for cc, cmd in zip(prep.compiled, prep.cmds):
            self._check_cancel(cancel)
            if checkpoint:
                checkpoint()  # pausing happens between sections
            self._check_cancel(cancel)
            ch = cc.chunk
            target = chunk_dir / f"{cmd.cache_key}.{out.segment_ext}"
            if spec.use_cache and self._cache_ok(target, ch.frames, out.fps):
                cached += 1
                if tracker:
                    tracker.uncached_total = max(1, tracker.uncached_total - ch.frames)
                log.line(f"{cmd.description}: reused from cache ({target.name})")
                done_frames += ch.frames
                if tracker:
                    tracker.video(0, ch.start, ch.index + 1, len(prep.cmds), None, 0, cached=True, base_frames=done_frames)
                try:
                    os.utime(target)  # most recently used: pruned last
                except OSError:
                    pass
                made.append(target)
                continue
            cdir = cmd.output.parent  # type: ignore[union-attr]
            self._prepare_dir(cdir, cmd, prep.compiler, spec.work_dir / "fonts")
            log.line(f"{cmd.description}: {cmd.printable()}")

            def prog(pi: ProgressInfo, ch=ch, base=done_frames, idx=ch.index + 1, produced_before=produced) -> None:
                fr = min(float(ch.frames), pi.out_time * out.fps if pi.out_time else float(pi.frame))
                if tracker:
                    tracker.note_rendered(produced_before + fr)
                    tracker.video(fr, ch.start, idx, len(prep.cmds), pi.speed, tracker_bytes(made) + pi.total_size, base_frames=base)

            res = self.ff.run_progress(cmd.args, on_progress=prog, cancel=cancel, on_line=lambda ln: self._ffmpeg_line(log, ln), cwd=cdir)
            self._check_result(res, "chunk", RenderStage.RENDERING, out.hardware, cancel, log, cmd, cmd.output)
            if not self._cache_ok(cmd.output, ch.frames, out.fps, strict=True):
                raise RenderError(f"Section {ch.index + 1} was rendered with the wrong length.", stage=RenderStage.RENDERING.value, kind="ffmpeg_failed",
                                  possible_issue="FFmpeg produced an incomplete section.")
            shutil.move(str(cmd.output), str(target))
            made.append(target)
            rendered += 1
            produced += ch.frames
            done_frames += ch.frames
            if tracker:
                tracker.note_rendered(float(produced))
                tracker.video(0, ch.end, ch.index + 1, len(prep.cmds), None, tracker_bytes(made), base_frames=done_frames)
        if tracker:
            tracker.p.video = 1.0
            tracker.p.cached_chunks = cached
        return made, cached, rendered

    def render_audio(self, prep: Prepared, spec: JobSpec, cancel, log: RenderLog, tracker: Tracker | None = None, announce: bool = True) -> Path | None:
        """The full-timeline audio mix as a lossless file (cached by its graph and inputs); ``None`` when nothing is audible."""
        amix = prep.amix
        if amix is None:
            return None
        if tracker and announce:
            tracker.stage(RenderStage.ENCODING, "Mixing audio")
        a_target = spec.cache_dir / "audio" / f"{amix.cache_key}.flac"
        a_target.parent.mkdir(parents=True, exist_ok=True)
        if spec.use_cache and a_target.is_file() and a_target.stat().st_size > 1000:
            log.line("audio mix: reused from cache")
            if tracker:
                tracker.set("audio", 1.0)
            return a_target
        (spec.work_dir / "graph.txt").write_text(amix.graph, encoding="utf-8")
        log.line(f"audio mix: {amix.printable()}")
        res = self.ff.run_progress(amix.args, on_progress=lambda pi: tracker.set("audio", pi.out_time / max(0.1, amix.expected_seconds), pi.speed) if tracker else None, cancel=cancel,
                                   on_line=lambda ln: self._ffmpeg_line(log, ln), cwd=spec.work_dir)
        self._check_result(res, "audio", RenderStage.ENCODING, False, cancel, log, amix, spec.work_dir / "mix.flac")
        shutil.move(str(spec.work_dir / "mix.flac"), str(a_target))
        if tracker:
            tracker.set("audio", 1.0)
        return a_target

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check_cancel(cancel) -> None:
        if cancel.is_set():
            raise RenderCancelled()

    def _prepare_dir(self, cdir: Path, cmd: BuiltCommand, compiler: TimelineCompiler, fonts_dir: Path) -> None:
        cdir.mkdir(parents=True, exist_ok=True)
        for name, text in cmd.files.items():
            (cdir / name).write_text(text, encoding="utf-8")
        FontResolver.stage(compiler.ass.fonts, cdir / "fonts")

    @staticmethod
    def _ffmpeg_line(log: RenderLog, line: str) -> None:
        low = line.lower()
        log.line(line, "WARNING" if ("warning" in low or "error" in low or "invalid" in low) else "FFMPEG")

    def _check_result(self, res: ExecResult, what: str, stage: RenderStage, hardware: bool, cancel, log: RenderLog, cmd: BuiltCommand, out: Path) -> None:
        if res.cancelled or cancel.is_set():
            raise RenderCancelled()
        if res.stalled:
            raise RenderError("FFmpeg stopped responding and was stopped.", stage=stage.value, kind="ffmpeg_failed", possible_issue="A source file may be unreadable or extremely slow to decode.")
        if res.returncode != 0 or not out.is_file():
            kind, issue, fallback = classify_failure(res.stderr_tail, hardware and what == "chunk")
            log.error(f"{what} failed (exit {res.returncode}): {res.stderr_tail[-800:]}")
            raise RenderError(f"FFmpeg failed while rendering ({what}).", stage=stage.value, kind=kind, possible_issue=issue, details=res.stderr_tail[-1500:], can_fallback_cpu=fallback)

    def _cache_ok(self, path: Path, frames: int, fps: int, strict: bool = False) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < 1000:
                return False
            info = self.probe.probe(path, use_cache=False)
            return bool(info.has_video and info.duration is not None and abs(info.duration * fps - frames) <= (2.5 if strict else 1.5))
        except Exception:
            return False

    @staticmethod
    def _prune_cache(cache_dir: Path, keep: set[Path], limit_bytes: int) -> None:
        files = [p for sub in ("chunks", "audio") if (cache_dir / sub).is_dir() for p in (cache_dir / sub).iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        for p in sorted(files, key=lambda x: x.stat().st_mtime):
            if total <= limit_bytes:
                break
            if p in keep:
                continue
            try:
                total -= p.stat().st_size
                p.unlink()
            except OSError:
                pass

    def _cleanup(self, spec: JobSpec, log: RenderLog) -> None:
        """Always remove the temporary working folder (media files are never kept); the small graph/overlay scripts of a failed render are saved for diagnosis."""
        try:
            dbg = spec.run_dir / "debug"
            for f in spec.work_dir.rglob("*"):
                if f.is_file() and f.suffix in (".txt", ".ass") and f.stat().st_size < 5_000_000:
                    rel = f.relative_to(spec.work_dir)
                    dest = dbg / str(rel).replace(os.sep, "_")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
            shutil.rmtree(spec.work_dir, ignore_errors=True)
        except OSError:
            _log.debug("work dir cleanup failed", exc_info=True)


def tracker_bytes(paths: list[Path]) -> int:
    n = 0
    for p in paths:
        try:
            n += p.stat().st_size
        except OSError:
            pass
    return n


_ = re
