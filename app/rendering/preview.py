"""PreviewEngine: watch the edit without a full final render.

A preview is rendered with the *same* compiler and command builder as an export, at lower resolution, with fast encoder settings, and
(unless "High Quality") from proxy media. It is split at scene boundaries into sections; every section is cached by a content key, so
editing scene 40 re-renders only scene 40's section and the previously rendered scenes are reused. Changing something global (project FPS,
canvas size, preview mode) changes every key, which invalidates the whole preview automatically.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from app.project.project_schema import RenderSettings
from app.rendering.engine import RenderEngine
from app.rendering.errors import RenderCancelled, RenderError
from app.rendering.executor import JobSpec, NullLog, RenderLog, Tracker
from app.rendering.models import RenderProgress, RenderSnapshot, RenderStage, has_audio_clips
from app.rendering.presets import ResolvedOutput

SECTION_TARGET = 1.0
SECTION_MAX = 30.0


@dataclass(frozen=True)
class PreviewMode:
    id: str
    label: str
    resolution: str
    crf: int
    speed: str
    use_proxies: bool
    description: str


PREVIEW_MODES: dict[str, PreviewMode] = {
    "realtime": PreviewMode("realtime", "Realtime", "480p", 36, "ultrafast", True, "Smallest and fastest: proxy media, 480p, minimal encoding."),
    "draft": PreviewMode("draft", "Draft", "720p", 30, "veryfast", True, "Proxy media at 720p: check timing, captions, graphics and audio."),
    "high": PreviewMode("high", "High Quality", "1080p", 23, "fast", False, "Original media at 1080p: closest to the final export."),
}


@dataclass
class PreviewSection:
    index: int
    start: float
    end: float
    scene_ids: list[str]
    key: str
    cached: bool


@dataclass
class PreviewPlan:
    mode: str
    sections: list[PreviewSection]
    audio_key: str
    audio_cached: bool
    duration: float
    width: int
    height: int
    fps: int

    @property
    def cached_sections(self) -> int:
        return sum(1 for s in self.sections if s.cached)

    @property
    def stale_sections(self) -> list[PreviewSection]:
        return [s for s in self.sections if not s.cached]


@dataclass
class PreviewResult:
    path: Path
    plan: PreviewPlan
    rendered: int
    reused: int
    start: float
    end: float
    uses_proxy: list[str] = field(default_factory=list)


class PreviewEngine:
    def __init__(self, engine: RenderEngine, root_getter: Callable[[], Path | None]) -> None:
        self.engine, self._root = engine, root_getter

    # ------------------------------------------------------------------ settings
    @staticmethod
    def settings_for(mode: PreviewMode, base: RenderSettings) -> RenderSettings:
        return replace(base, resolution=mode.resolution, fps=0, quality="custom", crf=mode.crf, encoder_preset=mode.speed, video_codec="h264", container="mp4", audio_codec="aac",
                       audio_bitrate_kbps=128, audio_sample_rate=48000, hardware_acceleration="cpu", bitrate_kbps=0, use_proxies=mode.use_proxies)

    def _dirs(self, mode: str) -> tuple[Path, Path]:
        root = self._root()
        assert root is not None
        return root / "previews" / "render" / mode, root / "cache" / "render"

    def _spec(self, snap: RenderSnapshot, mode: PreviewMode, ranges: list[tuple[float, float]] | None) -> tuple[JobSpec, ResolvedOutput]:
        settings = self.settings_for(mode, snap.settings)
        snap = replace(snap, settings=settings)
        resolved = self.engine.resolve(settings, snap)
        cache, cache_work = self._dirs(mode.id)
        rid = "preview_" + uuid.uuid4().hex[:8]
        spec = JobSpec(rid, snap, resolved, cache / "out" / f"{rid}.mp4", cache_work / rid, cache, cache / "runs" / rid, kind="preview", mode="preview" if mode.use_proxies else "final",
                       chunk_seconds=SECTION_TARGET, chunk_max_seconds=SECTION_MAX, chunk_ranges=ranges, encode_args=None)
        return spec, resolved

    # ------------------------------------------------------------------ plan (nothing is rendered)
    def plan(self, snap: RenderSnapshot, mode_id: str = "draft", ranges: list[tuple[float, float]] | None = None) -> PreviewPlan:
        mode = PREVIEW_MODES[mode_id]
        spec, resolved = self._spec(snap, mode, ranges)
        prep = self.engine.executor.prepare(spec, threading.Event(), NullLog())  # type: ignore[arg-type]
        cache, _w = self._dirs(mode.id)
        sections = [PreviewSection(cc.chunk.index, cc.chunk.start, cc.chunk.end, cc.chunk.scene_ids, cmd.cache_key, (cache / "chunks" / f"{cmd.cache_key}.{resolved.segment_ext}").is_file())
                    for cc, cmd in zip(prep.compiled, prep.cmds)]
        akey = prep.amix.cache_key if prep.amix else ""
        shutil.rmtree(spec.work_dir, ignore_errors=True)
        return PreviewPlan(mode.id, sections, akey, bool(akey) and (cache / "audio" / f"{akey}.flac").is_file(), prep.total_seconds, resolved.width, resolved.height, resolved.fps)

    # ------------------------------------------------------------------ build
    def build(self, snap: RenderSnapshot, mode_id: str = "draft", ranges: list[tuple[float, float]] | None = None, cancel: threading.Event | None = None,
              on_progress: Callable[[RenderProgress], None] | None = None) -> PreviewResult:
        """Render the stale sections (and the audio mix if needed), then join them into one playable file."""
        mode = PREVIEW_MODES[mode_id]
        cancel = cancel or threading.Event()
        spec, resolved = self._spec(snap, mode, ranges)
        cache, _w = self._dirs(mode.id)
        spec.work_dir.mkdir(parents=True, exist_ok=True)
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        log = RenderLog(spec.run_dir / "render.log")
        try:
            tracker = Tracker(spec.snapshot, has_audio_clips(spec.snapshot), resolved.fps, 1, on_progress or (lambda p: None))
            prep = self.engine.executor.prepare(spec, cancel, log, tracker)  # type: ignore[arg-type]
            tracker.stage(RenderStage.RENDERING, "Rendering preview sections")
            made, reused, rendered = self.engine.executor.render_sections(prep, spec, cancel, log, tracker)
            audio = self.engine.executor.render_audio(prep, spec, cancel, log, tracker)
            start, end = prep.chunks[0].start, prep.chunks[-1].end
            key = hashlib.sha1(("|".join(m.stem for m in made) + (audio.stem if audio else "") + f"{start}-{end}").encode()).hexdigest()[:20]
            final = cache / "out" / f"preview_{key}.mp4"
            if not final.is_file():
                final.parent.mkdir(parents=True, exist_ok=True)
                tracker.stage(RenderStage.ENCODING, "Assembling the preview")
                src = made[0]
                if len(made) > 1:
                    join = prep.builder.concat(made, "chunks.txt", spec.work_dir / f"video.{resolved.segment_ext}", prep.total_seconds)
                    (spec.work_dir / "chunks.txt").write_text(join.files["chunks.txt"], encoding="utf-8")
                    res = self.engine.ffmpeg.run_progress(join.args, cancel=cancel, cwd=spec.work_dir)
                    self.engine.executor._check_result(res, "join", RenderStage.RENDERING, False, cancel, log, join, spec.work_dir / f"video.{resolved.segment_ext}")
                    src = spec.work_dir / f"video.{resolved.segment_ext}"
                tmp = spec.work_dir / "preview.mp4"
                args = [self.engine.ffmpeg.ffmpeg(), "-hide_banner", "-nostats", "-v", "warning", "-i", str(src)]
                if audio is not None:
                    args += ["-ss", f"{start:.4f}", "-t", f"{end - start:.4f}", "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *resolved.audio_args()]
                else:
                    args += ["-map", "0:v:0", "-c:v", "copy", "-an"]
                args += ["-movflags", "+faststart", "-y", str(tmp)]
                res = self.engine.ffmpeg.run_progress(args, cancel=cancel, cwd=spec.work_dir)
                self.engine.executor._check_result(res, "preview", RenderStage.ENCODING, False, cancel, log, prep.cmds[0], tmp)
                shutil.move(str(tmp), str(final))
            plan = PreviewPlan(mode.id, [PreviewSection(cc.chunk.index, cc.chunk.start, cc.chunk.end, cc.chunk.scene_ids, c.cache_key, True) for cc, c in zip(prep.compiled, prep.cmds)],
                               prep.amix.cache_key if prep.amix else "", True, prep.total_seconds, resolved.width, resolved.height, resolved.fps)
            return PreviewResult(final, plan, rendered, reused, start, end, prep.plan.uses_proxy_assets)
        except RenderCancelled:
            raise
        except RenderError:
            raise
        finally:
            shutil.rmtree(spec.work_dir, ignore_errors=True)
            log.close()
            shutil.rmtree(spec.run_dir, ignore_errors=True)

    # ------------------------------------------------------------------ cache
    def clear(self, mode_id: str | None = None) -> int:
        n = 0
        for m in ([mode_id] if mode_id else list(PREVIEW_MODES)):
            cache, _w = self._dirs(m)
            for sub in ("chunks", "audio", "out"):
                d = cache / sub
                if d.is_dir():
                    n += sum(1 for _ in d.iterdir())
                    shutil.rmtree(d, ignore_errors=True)
        return n

    def cache_size(self) -> int:
        total = 0
        for m in PREVIEW_MODES:
            cache, _w = self._dirs(m)
            for f in cache.rglob("*") if cache.is_dir() else []:
                if f.is_file():
                    total += f.stat().st_size
        return total
