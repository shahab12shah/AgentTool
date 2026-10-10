# Phase 9 — implementation map (what exists, what is slow, what is missing)

Written before any optimisation, from reading the Phase 1–8 code and measuring a synthetic project (see `baseline_phase8.json`, produced by
`python -m app.performance.benchmarks`). Numbers are machine-specific (4 vCPU Linux container); only ratios between runs on the same machine mean anything.

## Existing systems to reuse (not to duplicate)
| Area | Where | Notes |
|---|---|---|
| Job system | `app/jobs/{job,job_manager,worker}.py` | `ThreadPoolExecutor(max_workers=3)`, FIFO, callbacks dispatched to the UI thread. No priorities, no dedupe, no resource awareness. |
| Proxies | `app/rendering/proxy.py` (`ProxyManager`) | Per-asset jobs, real progress, cancel/retry, staleness by (size, mtime_ns), `ProxyRecord`s live in `project.proxies`, files in `<project>/proxies/`. libx264 only; 540p/720p/1080p. No policy (off/manual/auto), no disk estimate, no storage report. |
| Thumbnails | `app/media/thumbnails.py`, `MediaService.ensure_thumbnail/ensure_all_thumbnails`, `ui/media_library.py` | One JPG per asset in `<project>/thumbnails/`, one FFmpeg process + one Job **per missing asset, scheduled for every asset on open** (no visibility priority, no dedupe beyond an in-flight set, no limit). Library keeps an unbounded `QIcon` dict. |
| Preview frames | `app/preview/frames.py` (`FrameProvider`) | JPEG per (asset, 0.25 s step) under `previews/frames/`, one FFmpeg per miss, unbounded `_failed` set, no size limit, no cancel of obsolete seeks. |
| Section preview | `app/rendering/preview.py` (`PreviewEngine`) | **Already segment-cached by content key** (scene-aligned sections, global settings change every key). Modes realtime/draft/high. |
| Probe | `app/rendering/probe.py` | In-memory cache by (path,size,mtime); lost on restart. |
| Waveforms | `app/audio/waveform.py` (`WaveformService`), `PresentationService.waveform` | Peaks cached on disk by content key, generated in a job; one resolution (`pps`), `range()` resamples. |
| Render backend | `rendering/{presets,planner,ffmpeg_service,executor}.py` | AUTO/CPU/HARDWARE exists (`RenderSettings.hardware_acceleration`); hardware **encoders** are test-encoded once per FFmpeg path; hardware failure → `hardware_failed` error with `can_fallback_cpu`. No decoder detection, no GPU/system summary. |
| QC | `app/qc/*` | Per-checker `input_hash`, scene-local reuse, detached snapshot (deep copy of the whole project document on the UI thread at run start). |
| Settings | `app/core/config.py` (`Settings`), `Project.render_settings`, `QCSettings` | |

## Measured / observed bottlenecks (baseline, before Phase 9)
See the table in the completion report; the structural findings are:
1. **Timeline canvas paints every clip on every repaint** (no viewport culling). Medium (100 scenes, 464 clips): 0.24 s per paint, 0.72 s zoomed in; cost grows linearly with clip count and super-linearly with zoom.
2. **Timeline lookups are linear** (`find_clip`, `get_clip`, `check_free`, `duration`, snap points rebuilt for every mouse move while dragging); `Workspace._on_project_changed` calls `get_clip` on every change.
3. **Opening a project schedules one thumbnail job (and later one FFmpeg process) per asset** — 153 jobs at 100 scenes, ~1,500 at 1,000 scenes.
4. **QC start deep-copies the whole project document on the UI thread** (`snapshot_project`): ~0.3 s at 100 scenes, growing linearly.
5. **Project load** is dominated by `Project.from_document` (~0.17 s at 100 scenes); the JSON is 1 MB at 100 scenes.
6. In-memory caches are unbounded (`QIcon` dict in the media library, `FrameProvider._failed`, probe cache, preview-frame pixmaps in the preview panel).
7. No priority between jobs: a thumbnail storm delays a user-requested preview or proxy.
8. No unified accounting of rebuildable cache size, no cleanup, no corruption handling for a cache index (there is no index).

## Missing capabilities (to build)
Performance metrics/diagnostics; unified cache manager with dependency invalidation; hardware capability service (CPU/RAM/disk/GPU/decoders); priority + resource-aware scheduler; proxy policy/profile/disk estimate; multi-resolution waveform summaries; preview quality settings with obsolete-seek cancellation; timeline index + viewport rendering; incremental QC/analysis invalidation; performance settings UI.

## Risky areas / compatibility constraints
* Project files must keep opening: the only project-document change is the optional `performance_overrides` key (no schema bump; absent = follow global settings).
* The timeline is mutated in place by many commands (`timeline_commands.py`, `phase4/5_commands.py`, editing assembly, QC fixes): any index must be validated against brute force across all of them and across undo/redo.
* Final export must never read proxies or preview caches unless the user explicitly opted in (`RenderSettings.use_proxies`).
* Qt: `QTest.qWait` starves Python worker threads in this environment — tests use `pump`/`wait_ms`.
* No new dependencies: resource probes use the standard library (no psutil).
