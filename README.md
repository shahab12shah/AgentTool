# AgentTool

Desktop **AI Video Director + Visual Researcher + Editor**. This repository currently contains **Phase 1: the desktop foundation and core project system**. No AI, web research or rendering is implemented yet — see [Status](#status).

## Run

```bash
pip install -r requirements.txt     # PySide6, pytest
# FFmpeg + ffprobe must be on PATH (or set their location in Settings)
python -m app.main [path/to/project_folder]
python -m pytest                    # Qt tests run headless via QT_QPA_PLATFORM=offscreen
```

Always start the app as `python -m app.main` (or `python app/main.py`, which is guarded): `app/logging/` must never be put on `sys.path` because it would shadow the standard library.
On Linux the Qt wheels need system GL/EGL libraries (`apt install libegl1 libgl1`).

## Architecture

```
UI (PySide6)  ──►  services/  ──►  domain models  ──►  storage / FFmpeg
```

| Package | Responsibility |
|---|---|
| `ui/` | Widgets only. Call `Workspace`/services, render model state, never touch files, JSON or FFmpeg. |
| `services/` | `Workspace` (facade: lifecycle, dirty tracking, autosave policy, recovery, settings), `MediaService`, `TimelineService`. |
| `core/` | Qt-free infrastructure: `EventBus`, undo/redo `CommandStack`, settings, exceptions, constants, timecode *display* formatting. |
| `project/` | `Project` aggregate, JSON schema + validation + migration hook, `ProjectManager` (atomic save), `AutosaveService`, `RecoveryManager`, project-level commands. |
| `media/` | `Asset`, `AssetRegistry` (stable `media_00001` ids, SHA-256 duplicate detection), ffprobe wrapper, importer (hash/probe/copy – thread-safe), thumbnails. |
| `timeline/` | `Timeline`/`Track`/`Clip` model (time = float seconds) and undoable commands. |
| `jobs/` | `Job`, `JobManager` (thread pool, queue/cancel/retry/pause-queued, callbacks dispatched to a chosen thread), worker. |
| `ai/` | Provider interfaces; they raise `NotAvailableInPhase` rather than fake results. |
| `rendering/` | **Phase 6**: the media pipeline — `FFmpegService`, `MediaProbeService`, `RenderPlanner`, `TimelineCompiler`, `FFmpegCommandBuilder`, `RenderExecutor`, `RenderValidator`, `OutputManager`, `RenderQueue`, `ProxyManager`, `PreviewEngine`, `MediaRelinkService`, `RenderDiagnostics`. Qt-free; the UI only calls `services/render_service.py`. |
| `preview/` | `PreviewBackend` protocol + QtMultimedia player (single file). |
| `storage/`, `logging/` | OS paths, atomic file writes, structured JSON logging with secret redaction. |

Key rules enforced in code: the core is Qt-free; worker threads never mutate the project (importers return a `PreparedMedia`, registration happens on the UI thread); every edit is a `Command`; assets/clips/tracks are referenced by stable ids, never filenames; media is never deleted from disk.

**Persistence.** `project.json` is written to `project.json.tmp`, read back and schema-validated, then `os.replace`d; the previous good file is kept as `project.json.bak`. Autosave/recovery snapshots live in the per-user data dir (never inside the project) and are removed on save or clean close, so a leftover snapshot at startup means an unclean exit. Recovering loads the snapshot as an *unsaved* project; the main file is untouched until the user saves.

User data locations (override with `AGENTTOOL_HOME`): settings/recent projects in the config dir, logs + recovery data in the data dir.

## Phase 2 — voice analysis, transcription and scene segmentation

```
Voice-over ─► TranscriptionService ─► words (id, text, start, end, confidence) ─► sentences ─► script alignment
          ─► semantic analysis (entities · claims · numbers/dates · visual type) ─► scene segmentation
          ─► per-scene enrichment (topic, summary, visual intent, importance, confidence, context memory) ─► user review
```

The voice-over is the timing master: scenes tile `[0, audio duration]` exactly (`scene[i].end == scene[i+1].start`) and every scene's narration is rebuilt from the transcript words in its time window. The script is never edited; `transcription`, `script_alignment` and `scene_analysis` are stored next to it.

**Transcription is provider-independent** (`transcription/provider.py`). Three providers ship:

| provider | needs | status in this repo |
|---|---|---|
| `faster-whisper` (local, recommended) | `pip install faster-whisper` + a model name/folder in Settings | adapter tested with an injected fake model only — **not run against a real model** |
| `api` (OpenAI-compatible `/audio/transcriptions`) | base URL in Settings, key in an env var (only the variable *name* is stored) | tested against a local mock HTTP server — **not run against a real service** |
| `pocketsphinx` (offline, bundled) | nothing | runs for real, but **accuracy is low** (see limitations) |

`auto` picks the first available in that order. Results are cached per (audio hash, provider, settings) in `cache/transcripts/`.

**Scene segmentation** (`analysis/`) is a transparent *rule-based* analyzer behind the `SemanticAnalyzer` interface, so a model-backed analyzer can replace it. Boundaries score lexical cohesion between neighbouring sentence windows, topic-transition / question / continuation cues, new entities, new evidence, visual-type change and the length of the pause in the audio. It merges sentences that form one idea, enforces min/max scene length, and can split one sentence at the items of an enumeration. Manual split/merge/edit/approve are undoable commands; AI output never overwrites scenes you edited, approved or split without confirmation, and a failure on scene *N* keeps scenes 1..N-1 and resumes at *N*.

Change detection: replacing the voice-over marks the transcript `OUTDATED` (derived from the audio hash; the old transcript is kept); editing only the script marks the alignment outdated (cheap re-align, no re-transcription); scenes/analysis are flagged outdated, never silently regenerated.

Schema version is now **2** (additive; version-1 projects migrate on open). Visual source preferences (soft targets, accuracy threshold, rules) are saved in the project; `VisualResearchService` is an interface only — nothing is searched.

## Phase 3 — AI visual research engine

```
Scene ─► Research Brief (+ context memory) ─► multi-purpose queries ─► source providers (isolated, parallel, cached)
      ─► normalised candidates ─► de-duplication ─► evaluation (0–100, 7 weighted components, explainable)
      ─► ranking (accuracy dominates; soft source targets, repetition and continuity adjustments)
      ─► best + up to 5 diverse alternatives ─► user review ─► Visual Assignment ─► project asset
```

Code lives in `research/` (Qt-free) and `services/research_service.py`; the UI is the **Review** page (`ui/review_panel.py`). Research never touches the timeline.

* **Brief** (`brief.py`): subject, action, context, visual type, evidence level, preferred/avoided sources, plus context from neighbouring scenes ("That means…" scenes inherit the subject) and traps to avoid (e.g. *coin* imagery for solar-silver).
* **Queries** (`queries.py`): LITERAL / CONTEXT / PROCESS / EVIDENCE / ENTITY / LOCATION / DATA / NEWS / DOCUMENT / ALTERNATIVE, each with a purpose and source preference. *Search Again* uses a new strategy each time (broader, specific, process, evidence, alternative, sources) — never the same search twice.
* **Providers** (`research/providers/`, `ProviderRegistry`): each implements `search`, `fetch_thumbnail`, `acquire`. A failing provider is reported and isolated; if all fail the scene shows *"Visual research unavailable."* with Retry / Expand Sources / Generate AI Visual / Manual Select / Skip. Random stock is never substituted silently. Expanding to disabled sources always asks first and never changes your preferences.
* **Evaluation** (`evaluation.py`): semantic .35 · subject .20 · context .15 · action .10 · timing .10 · quality .05 · source .05, with concise "why" factors. Categories: ≥90 Excellent, 80–89 Good, 70–79 Review, 60–69 Weak, <60 Reject. Default minimum 85 (configurable on the Visuals page).
* **Ranking** (`ranking.py`): accuracy decides; bounded adjustments (+6 / −15) for source targets, priority, repetition and continuity; alternatives are chosen greedily with a diversity penalty and must score ≥ 60.
* **Decisions**: Use / Replace / Reject / Skip / Approve / More like this / bulk approve (confirmation; scenes below the minimum are skipped, never forced). Choosing a candidate acquires it as a *project asset* (download job, validation, registration); a research result alone is not an asset. All decisions are undoable commands. Approving requires accuracy ≥ minimum unless **you** chose the visual.
* **Schema 3** adds `research_settings`, `research_queries`, `research_sessions`, `visual_candidates`, `candidate_scores`, `visual_assignments`, `source_metadata`, `research_status` (v1/v2 projects migrate on open). Editing a scene prunes/flags its research; the table shows "(scene changed)".
* **Secrets**: only environment-variable *names* are stored (YouTube, Pexels, AI image); keys are never saved, logged or shown.

### Provider status (honest)

| provider | sources | status in this repo |
|---|---|---|
| `local_stock` | stock image/video from a folder you point at | **runs for real** (tests use real files and FFmpeg) |
| `screenshot` | Chromium page capture, official-site homepages | **runs for real against local test pages** with the installed Chromium; real websites were not reachable from the sandbox. Evidence stays "needs verification". |
| `wikimedia` | web images (Commons) | adapter **tested against a mock HTTP server only** |
| `youtube` | YouTube search (Data API) | **mock-tested only**; results are **reference-only** — videos are never downloaded, chapters become suggested segments |
| `pexels` | stock image/video | **mock-tested only** |
| `ai_image` | AI-generated image | **mock-tested only**; proposes a prompt first, generates only on request (may cost money); never counted as evidence |

No provider other than `local_stock` was exercised against its real service. Licences shown are what the source *states*; the application never verifies rights — check licences before publishing.

## Phase 4 — AI advanced editing and automatic timeline assembly

```
Script + voice-over + transcript + scenes + approved visuals + editing settings
  -> EditingStrategyService (provider: rule-based | AI [not available yet])   video profile, SceneEditingBrief per scene
  -> ShotTimingService     shot length from narration speed, pauses, density, complexity, importance, reading time
  -> visual segments       one or several visuals per scene; cuts on word/sentence boundaries; reuse tracked
  -> motion · text/number emphasis · evidence treatment · transitions · audio-ducking and caption instructions
  -> TimelineAssemblyService  -> TimelineValidator  -> ONE undoable command  -> the normal editable timeline
```

Golden rule: **the AI creates the edit; the user owns every decision.** There is no separate AI timeline and nothing is rendered or flattened: the result is the project's real timeline (tracks V1 main, V2 B-roll, V3 images, V4 graphics, V5 text, V6 captions, A1 voice-over, A2 music, A3 SFX) plus structured decisions.

* **Timeline items** (`timeline/clip.py`) gained `kind` (media/text/graphic), `scene_id`, `slot`, `created_by` (AI/USER/SYSTEM), `ai_decision_id`, `locked`, `keyframes`, `effects`, `text`, `animation`, `audio`, `transition`, `metadata`. Source media is never touched: a clip is a source range + transform + keyframes. Keyframes (`position_x/y, scale, rotation, opacity, volume, blur`; linear / ease_in / ease_out / ease_in_out) are plain data the user can edit.
* **Decisions** (`editing/models.py`): `VISUAL_TIMING, CUT, TRIM, ZOOM, PAN, KEYFRAME, TEXT, NUMBER_EMPHASIS, EVIDENCE_FOCUS, TRANSITION, AUDIO_DUCK, CAPTION_EMPHASIS`, each with start, duration, parameters, a one-sentence reason (never chain-of-thought), confidence (≥90 High, 80–89 Good, 70–79 Review, <70 Low — low ones are marked ⚠ on the timeline) and `created_by`.
* **Ownership.** Editing an AI element (inspector, or any normal timeline edit: move/trim/split/properties) makes it USER-owned in the same undo step: a USER decision with `overrides_decision_id` replaces the AI one, and the original is kept in `ai_overrides`. Regeneration (scene / selected / entire edit) replaces only AI-owned, unlocked elements; USER and locked ones survive, a USER zoom/transition/evidence decision is re-applied to a regenerated clip, and elements the user deleted are not re-created. Locks: visual, timing, text, motion, whole scene.
* **Safety.** Every generation is one undo step; a checkpoint of the project is written before it (`<data dir>/checkpoints/<project>/before_ai_edit_<time>.json`, newest 10); the result is validated (`TimelineValidator`: timestamps, durations, source ranges, asset/track references, keyframes, transitions, overlaps, voice-over alignment, scene coverage) and **not committed** if it has errors; autosave covers it; cancel changes nothing.
* **Incremental and resumable.** Per-scene input hashes: unchanged scenes are never re-analysed (plans are cached under `cache/editing/plans`); a failure at scene *N* keeps scenes 1..N-1, marks N FAILED and the rest PENDING, and *Retry* resumes at N.
* **Missing visuals** are never invented: MISSING / UNAPPROVED / SKIPPED / MISSING_MEDIA scenes get text and audio instructions only, with Return to Research, Replace Manually and Skip.
* **Audio and captions** are instructions only (`AUDIO_DUCK` fade-in/out, ducks on important lines and figures, modest rises in pauses; priority voice > SFX > music; `CAPTION_EMPHASIS` words and a safe region). Nothing is mixed or burned in.
* **Presets** Documentary / Professional / Dynamic plus sliders for pacing, motion and transition frequency and toggles for text, number, evidence, transitions and ducking. Schema is now **4** (older projects migrate; the V6 captions track is added).
* **UI**: *AI Edit* page (settings, progress "Scene 18 / 74" with the current operation, scene table, scrubbing preview, decision list and inspector with real editable parameters, locks, regenerate, retry, replace visual); AI badges, lock and low-confidence markers on the timeline; an AI box in the clip inspector; *Split at playhead* on the timeline.

### What Phase 4 does not claim
* The strategy provider is **rule-based**. `AIProvider` is an interface slot that reports itself unavailable; nothing model-based was run.
* **Evidence regions are not detected.** The document zoom/highlight uses a default region and is flagged low-confidence (< 70%) for the user to adjust. Motion for portraits/wide photos uses the image aspect ratio only (no face or saliency detection).
* Scoring inputs are metadata and timing; the preview shows still frames at the playhead (images, or video frames extracted on demand and cached) composed from timeline data — it is **not** a rendered or real-time playback preview, and no proxy media pipeline exists yet. Voice-over audio plays; music/SFX are not mixed.
* Transitions are timeline data: a dissolve is previewed as a cross-fade, wipe/slide approximately. Reference-style analysis is not implemented. No FFmpeg rendering, export, caption renderer, music/SFX library or mastering.
* Short source clips are slowed (≥ 0.8x) or restarted on a word boundary to cover the narration; such shots are marked lower-confidence.

## Phase 5 — professional audio, captions and motion graphics

```
VOICE-OVER (master clock) ─► AudioEngine ─► analysis · loudness · waveforms · processing chain · mix preview · ducking
                         ─► CaptionEngine ─► word/sentence timestamps -> segments · line breaking · keyword/number emphasis · styles
                         ─► MotionGraphicsEngine ─► number/date graphics · lower thirds · headlines · evidence tools · animation presets
                         ─► SFX planner (sparse, explained) ─► PresentationAssembler ─► PresentationValidator ─► ONE undoable command
```

Everything is a normal timeline object — **captions on V6**, graphics on V4/V5, **music on A2**, **SFX on A3** (extra layer tracks on demand), voice-over on A1 — plus a *presentation decision* (reason, confidence, `created_by`). Nothing is rendered or burned in; there is no second timeline.

* **Audio engine** (`audio/`, provider-independent `AudioBackend`, FFmpeg implementation): `AudioAnalysisService` (duration, peak, RMS, LUFS where FFmpeg can measure it, dynamic range, clipping, noise floor, silences, pauses and speaking rate from the transcript, per-sentence intensity, acoustic emphasis candidates), `LoudnessAnalyzer` (reports `CLIPPING`, `TOO_QUIET`, `TOO_LOUD`, `EXCESSIVE_NOISE`, `LONG_SILENCE`; it never alters the audio), `WaveformService` (cached peaks, background job), `AudioProcessingService` (gain, normalize, fades, compression, limiter, noise reduction, high-pass, EQ presets as editable `VoiceProcessingSettings` — rendered to the cache for previews only), `AudioMixService` (the mix as data: track mute/solo/volume, clip gain, fades, volume keyframes; preview renders for Voice / Music / SFX / Voice+Music / Voice+SFX / Full), `VoicePriorityController` (voice 100, music/SFX ceilings while speaking; masking is reported).
* **Ducking** is volume **keyframes**, not a global level: voice activity from word timestamps, important scenes and figures, pauses (music may rise modestly), intro/outro, section intensity and SFX dips, with attack/release ramps. Defaults (configurable): normal ≈ 18%, important ≈ 9%, pause ≈ 24%. Editing the keyframes makes the ducking decision USER-owned.
* **Captions**: a dynamic-programming segmenter per sentence (punctuation, pauses, phrase starts, reading speed, line width from the safe margins and font size; dates, numbers + units and titles are never split), 1–2 line breaking (`SILVER IS / RUNNING OUT`), word-level highlight or progressive reveal, six style presets (Professional, Clean, Bold, News, Documentary, Minimal) plus per-caption overrides, accessibility (large text, strong contrast, reading speed, reduced motion). Captions start and end only on word boundaries and never change the spoken wording; editing the text keeps the word timing (or redistributes it).
* **Keywords / numbers**: from the scene analysis (numbers, dates, money, percentages, people, organisations, places), a warning/deadline lexicon and optional acoustic emphasis; at most two emphasised words per caption and a per-scene budget. Styles: colour, bold, scale, background box, underline, glow, pop.
* **Graphics**: number cards, date/deadline graphics (start when the narration says *deadline*), lower thirds (text + subtitle), section headlines (marked *derived* from the scene topic, lower confidence), warnings, evidence tools (focus box, highlight, underline, pointer, magnify, crop, dim — the document is never altered), counters that always end on the real figure. Text only comes from the narration/script (or the user); no chart or data is ever generated. Animation presets (fade, slide, scale, pop, reveal, type-on, counter, highlight) are editable specs on the clip; LOW/MEDIUM/HIGH motion and reduced-motion choose the defaults.
* **SFX**: library by category, import/tag, place, trim, move, volume, fade, replace, delete, layer. AI SFX are rate-limited (per minute and minimum gap), quiet, never on every cut, only from the library, and each carries scene, time, duration, type, volume, reason and confidence.
* **Ownership / locks / regeneration** work like the AI edit: manual timeline edits and inspector edits make an object USER-owned (the replaced AI decision is kept in `presentation_overrides`); locks for caption, graphic, music, SFX, mix and whole scene; *Regenerate Captions / Graphics / Audio / Selected Scene / All*; deleted AI objects are not re-created; every run is one undo step with a checkpoint, is validated (`PresentationValidator`: timestamps, safe area, line length, sources, keyframes, animations, references) and not committed if invalid; scenes that fail stop the run, earlier scenes are kept and *Retry* resumes there.
* **Voice replacement**: the captions are marked outdated (*“Voice-over changed. Captions need regeneration.”*) with Regenerate Captions / Keep Existing / Review Changes; the analysis and transcript are flagged; nothing is shifted or deleted; *Update voice-over clip* re-points A1.
* **Persistence**: schema **5**. Settings, analysis, processing, ducking events, styles, keyword emphasis, scene presentation plans, decisions and overrides are stored; `caption_segments`, `text_graphics`, `motion_graphics`, `music_assignments` and `sfx_assignments` are *derived views of the timeline clips* (the clips are the single source of truth, so nothing can drift) and are rebuilt on load.
* **UI**: *Audio & Captions* page (global caption and audio controls, generation, stale banner, scene status, caption / graphic / music & SFX editors, voice analysis, processing, mix preview modes); waveforms (cached, drawn red where clipped, silences shaded) and caption/graphic/music/SFX objects on the timeline; Solo and volume on audio tracks; a *Presentation object* box in the clip inspector.

### What Phase 5 does not claim
* **Audio quality is measured, not judged by ear**: no speech or music model was run; "emphasis" is loudness relative to the sentence. Speaker changes are not supported (reported as such). LUFS comes from FFmpeg's `ebur128`.
* The voice processing chain (noise reduction, EQ, compression) is rendered by FFmpeg only for previews; how it *sounds* was not evaluated by a person. Audio playback could not be verified end-to-end in the headless environment (the mix files are rendered and analysed numerically).
* The preview is composed from timeline data (stills at the playhead, captions/graphics painted by Qt); it is not a rendered video and not real-time playback. Fonts are looked up by Qt at paint time; a missing font falls back silently.
* Caption lines are estimated from font size (average glyph width), not measured with the real font; very unusual fonts can wrap slightly differently.
* Section headlines use the scene topic (a heuristic noun phrase) and are flagged for review. Graphics generated by Phase 5 adopt the Phase 4 text overlays that match; regenerating the Phase 4 edit recreates those overlays (run *Regenerate Graphics* afterwards). The Phase 4 edit never touches captions, music or SFX.
* No final render, export, codec settings, mastering, music/SFX library bundled with the app (you import your own), or competitor/reference analysis.

## Phase 6 — rendering, preview, proxy and export engine

```
Editable timeline ─► RenderSnapshot (frozen copy) ─► Preflight ─► RenderPlanner ─► TimelineCompiler ─► FFmpegCommandBuilder
        ─► RenderExecutor (video sections · audio mix · encode) ─► RenderValidator (ffprobe the *file*) ─► OutputManager ─► MP4 + log + metadata
```

**The MP4 is an output, not the project.** A render reads a frozen snapshot and writes a copy; the timeline, the AI/user decisions and every original asset are never modified, flattened or deleted. Before a render starts the project is autosaved, a checkpoint is written (`<data dir>/checkpoints/<project>/before_render_*.json`) and the snapshot is taken, so editing during a render is safe (the render shows what the timeline was when it started).

* **FFmpeg integration.** `FFmpegService` detects ffmpeg/ffprobe, the version (≥ 4.4), which encoders/filters exist and which *hardware* encoders actually work (each is test-encoded once), runs structured argument lists (never shell strings), parses `-progress`, honours cancellation (ask FFmpeg to quit → terminate → kill) and returns the stderr tail. Filter graphs are written to a script file and FFmpeg runs inside the render's working folder, so no path ever needs filtergraph escaping (spaces, parentheses, apostrophes, Unicode and Windows `C:\...` paths are covered by tests/escape helpers). `MediaProbeService` reads duration, display size (after rotation), fps, codec, pixel format/alpha, audio rate/channels, bitrate, frame count, container; the facts are stored in `asset.extra["probe"]`.
* **Compiler.** The timeline becomes layers bottom-to-top (V1…V6 in track order; text/graphics/captions are an ASS pass at the position of their tracks) with every property as an FFmpeg expression of timeline time: position, scale, rotation, opacity, reveal and crop, with keyframes interpolated *exactly* like the editor (`linear`, `ease_in`, `ease_out`, `ease_in_out` — checked against `value_at` in tests). Clips are placed on the frame grid of the project FPS; source in/out and speed are honoured; a source shorter than its clip holds its last frame; fit modes `cover`/`contain` never distort; transitions CUT / FADE / DISSOLVE / WIPE / SLIDE take their duration from the timeline.
* **Text, captions, graphics.** All of V4/V5/V6 is rendered with libass (ASS script generated per render): headlines, lower thirds, numbers (counters end on the real figure), dates, warnings, evidence boxes/dim/highlight/underline/pointer, and captions with the style, safe-margin position, word highlight/progressive reveal and emphasis colours of the timeline objects. Animations are *sampled once per frame* with the same `animation_state` function the preview uses, so the export moves like the preview. Captions only exist in the export; the project keeps editable caption clips (turning captions off removes them from the export, not from the project). Fonts are resolved to real files; a missing font falls back (reported in preflight, plan and log), never crashes.
* **Audio.** One filter graph for the whole timeline: resample to 48 kHz, gain (clip × track), keyframed volume — this is where **ducking is rendered** (the timeline's volume keyframes, never baked into the music file) — fades, pan, atempo for speed, mute/solo, `amix` and a limiter (peak ≤ −0.26 dB) → lossless mix (cached) → AAC/Opus/FLAC. The voice-over processing chain is applied if enabled in the project.
* **Output.** H.264 + AAC + MP4 by default; H.265 / VP9 / AV1 (MP4, MKV, WebM) where the FFmpeg build has the encoder — otherwise a clear message with the supported alternatives; the codec is never changed silently. 1080p and 4K (plus 480p/720p), 16:9 / 9:16 / 1:1 from the project's aspect ratio, 24 / 30 / 60 FPS (project FPS is the default master rate; other rates use FFmpeg's timestamp-based `fps` conversion). Presets: YouTube 1080p, YouTube 4K, High Quality, Draft; quality Draft / Standard / High / Maximum / Custom (CRF, bitrate, audio bitrate, encoder preset). Hardware: Auto / CPU / Hardware; CPU always works; a failing hardware encode is reported as such with a *Retry on CPU* offer, never hidden.
* **Sections, cache and recovery.** The timeline is rendered in scene-aligned **sections** (MPEG-TS for H.264/H.265 so every frame time is exact, Matroska for VP9/AV1), each cached by a hash of its filter graph, overlay script, encoder arguments and input files; sections are joined with stream copy, the audio is mixed alongside, and only the final audio encode/mux writes the file. Edit one caption → only that section is rendered again; change the project FPS → everything is. A failure at a late stage leaves the finished sections in `cache/render/chunks`, and *Retry* reuses them. Working files live in `cache/render/<render_id>/` and are always removed (small graph/overlay scripts of a failed render are kept under `renders/<render_id>/debug/`).
* **Proxies.** `ProxyManager` makes 540p/720p/1080p H.264 stand-ins (same frame rate and timing, video only) with Generate / Cancel / Retry / Delete / Regenerate; `project.proxies[asset_id]` holds `asset_id, original_path, proxy_path, proxy_status, proxy_resolution` (+ the original's size/mtime so a proxy of a replaced file is *stale*). The timeline references the asset id only. Editing previews and the preview engine read the proxy; **final exports read the original**. If an original is missing the export is blocked and the user chooses *Locate Original*, *Use Proxy Anyway* (explicit, per asset, noted in the record) or *Cancel*. "Export from proxy media" is an explicit setting, never a default.
* **Preview engine.** Modes Realtime (480p) / Draft (720p) / High Quality (1080p, original media). Scene-aligned sections are cached by content, so editing scene 40 re-renders scene 40 only; anything global invalidates all. A scene or the whole timeline can be rendered in the background and played in the system player.
* **Render queue.** QUEUED → RUNNING → (PAUSED) → COMPLETED / FAILED / CANCELING → CANCELED. Each job has its own working folder, output name (`Name_1080p_30fps.mp4`, `_01`, `_02`… unless overwrite is chosen) and log. Progress is real: FFmpeg's `out_time` per section, audio and final encode (weights 80/8/10/2 %), the current scene, speed, elapsed, an ETA only once there is data, and the output size. Cancelling stops FFmpeg, removes temporary files and leaves the project untouched. Pausing a running render takes effect between sections.
* **Validation.** *Preflight* (before): FFmpeg, timeline (keyframes, animations, durations, references), voice-over, visuals found (`84/84`), unreadable/unsupported media, captions, fonts, audio, settings (codec/container/hardware), output location, disk space. *Output validation* (after): the file exists and is readable; video and audio streams; duration against the timeline (configurable tolerance); resolution; aspect ratio; fps; codecs; pixel format; sample rate; peak/clipping; and the **voice-over must be audible** — a silent voice-over *fails* the render instead of shipping a broken video. FFmpeg exiting 0 is not success.
* **Records.** `project.render_history` and `renders/<render_id>/{render.log, metadata.json, snapshot.json}`: render id, project id, timeline version **and content hash**, timestamp, settings, resolved encoder, output path, status, validation, warnings, FFmpeg version, commands (credentials redacted) and exit status. `render_settings` are saved with the project (schema **6**; 5 → 6 is additive).
* **Relinking.** `MediaRelinkService` finds replacements by content hash (exact), file name, size and media facts; a weak match is only used after explicit confirmation, `auto_relink` only replaces exact copies, and relinking is an undoable edit that keeps the asset id.
* **UI.** *Export* page: preset, resolution, FPS, quality, codec, audio, hardware, container, advanced (CRF, bitrates, preset), output location, preflight checklist with *Fix Issues*, START EXPORT / Draft export, render queue with real progress and *Rendering… Scene 42 / 87 / Speed / ETA* detail, completion and failure cards (Open Video / Open Folder / Export Again / Return to Editor; Retry / Retry on CPU / Open Logs / Change Settings), history, proxies and preview tabs. The UI never builds FFmpeg commands or touches render files.

### What Phase 6 does not claim
* **Compositing is verified, not perfect.** Tests render real frames and check colours, sizes and positions (transforms, keyframes, transitions, crops, alpha, cuts at the exact frame for 24/30/60 fps) and audio levels over time (gain, ducking, fades, SFX onset). Pixel-exact equivalence with the editor's Qt preview is not claimed: libass draws text (font metrics differ slightly from Qt), a caption background box is drawn per line, and the word-level "background box" emphasis falls back to a colour/bold change.
* **Not implemented:** clip masks (only alpha, opacity and reveal/wipe masks), keyframed blur (a single constant blur is used and a warning is logged), speed *ramps*, frame blending/optical-flow for fps conversion (frames are picked by timestamp), colour management/HDR (8-bit yuv420p output only), hardware *decoding*, VAAPI encoders, parallel rendering of several video sections at once (sections run one after another; the audio mix runs alongside), real-time playback inside the app (a preview is a rendered file played by the system player).
* **Not verified here:** hardware encoders (no GPU in the development sandbox: detection, the "unavailable" path, failure classification and CPU fallback are tested; actual NVENC/QuickSync/AMF/VideoToolbox encodes are not), Windows and macOS (path handling is tested with escape helpers; no Windows ffmpeg was run), AV1/VP9 playback in players, real multi-gigabyte media (streaming design only; the largest test source is a 3-second 4K clip), and loudness against platform targets (no loudness normalisation is applied; the limiter only prevents clipping).
* Per-frame `scale` for animated zooms uses integer sizes (tiny shimmer on very slow zooms of small pictures); very large stills are decoded once but scaled every frame during a zoom.
* Disk-space figures are estimates (quality-typical bitrates), not guarantees. The cache is pruned to 4 GB per render; previews have no size cap (clear them from the History tab).
* No YouTube upload, publishing, cloud rendering, collaboration, colour grading or competitor analysis (explicitly out of scope).

## Status

Implemented (Phase 1, see above for Phase 2): project create/open/save/save-as/close with recent list; media import (copy or link) with probing, duplicate detection and cached background thumbnails; media library (search/sort/remove/drag); script editor; voice-over import/replace/remove/playback; single-file preview; 8-track timeline with add/rename/hide/mute/lock/delete tracks and add/move/trim/delete clips (mouse, inspector, drag-drop, snapping); inspector; undo/redo; autosave + crash recovery; job manager with status bar and jobs panel; settings; logging; renderer validation/plan.

Phases 3–5 add visual research, AI editing and the audio/captions/graphics layer (above); Phase 6 adds rendering, proxies, previews and export. Not implemented: the stand-alone Edit page ("Coming in a later phase").

### Known limitations
**Phase 3**
- Scoring is **metadata/text based** (titles, descriptions, tags, duration, licence, resolution). Pixels are not inspected, so a mislabelled result can score well; every score carries `basis: METADATA`. The preview and your review are the real check.
- Real-service behaviour (rate limits, response variations, API changes) of Wikimedia, YouTube, Pexels and the AI image service is unverified. YouTube videos cannot be acquired (reference-only).
- "More like this" re-searches with a more specific wording; it does not use visual similarity.
- No visual-similarity model: near-duplicate detection uses a perceptual hash of thumbnails only.
- Interactive UI was driven by scripted events in a headless environment; no human has used it on a real display.

**Phase 2**
- No accurate speech model could be obtained in the development sandbox (model hosts were blocked). On clean synthetic speech the bundled PocketSphinx engine recovered only ~24% of the script's words — treat it as a demo fallback. Real-world accuracy of `faster-whisper` and `api` is unverified here.
- The semantic analyzer is heuristic: small built-in lexicons (agencies, countries, cities, companies, commodities, technologies, objects), cue words and lexical cohesion. Unknown names are found only by capitalisation (which un-punctuated transcripts lack unless the script is aligned), topics are noun-phrase guesses, visual types are cue-scored, and it only handles English. Expect mis-typed scenes and awkward topics; that is what the review UI and the `SemanticAnalyzer` interface are for.
- Sentences are fixed when transcribing; adding a script *after* transcription does not rebuild them (re-transcribe, or the script only affects alignment/casing).
- Audio playback could not be verified end to end in the headless test environment (no audio device); tests drive the player position directly and check the highlighting logic and click-to-seek.
- Manual split recomputes AI metadata from the split words; summaries are extractive (a chosen sentence), not abstractive.

**Phase 1**
- Preview plays one file at a time; the timeline playhead is positional only (no composited timeline playback).
- Video frames could not be visually verified in the headless test environment (playback loading/duration is tested; on-screen rendering depends on the platform's Qt multimedia backend).
- Running jobs cannot be paused (only queued ones); jobs are not persisted across restarts. A crash mid-import can leave an unregistered file in `media/` (`*.part` files are cleaned on open).
- Removing media from the project keeps the file on disk (by design; no "delete files" action yet). Linked (non-copied) media breaks if the external file moves (reported as MISSING).
- Clip transform values (position/scale/rotation/opacity) are applied by the Phase 6 renderer (the editor's preview approximates them). `Source In/Out` are read-only (changed by trimming).
- Tests cover logic and a scripted UI workflow with synthetic mouse events; no human has used the GUI on a real display.

## Next: Phase 7
Candidate work (not started): in-app playback of rendered previews, parallel section rendering, loudness targets and mastering presets, colour management, a bundled royalty-free music/SFX library, evidence-region detection, a model-backed editing/keyword provider, and validating the research providers and hardware encoders on real services and machines.
