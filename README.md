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
| `ai/`, `rendering/` | Provider/renderer **interfaces** only; they raise `NotAvailableInPhase` rather than fake results. |
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

## Status

Implemented (Phase 1, see above for Phase 2): project create/open/save/save-as/close with recent list; media import (copy or link) with probing, duplicate detection and cached background thumbnails; media library (search/sort/remove/drag); script editor; voice-over import/replace/remove/playback; single-file preview; 8-track timeline with add/rename/hide/mute/lock/delete tracks and add/move/trim/delete clips (mouse, inspector, drag-drop, snapping); inspector; undo/redo; autosave + crash recovery; job manager with status bar and jobs panel; settings; logging; renderer validation/plan.

Not implemented: visual research/search/download, candidate scoring, AI images, Review/Edit pages ("Coming in Phase 3 / a later phase"), rendering (Export only validates and shows the plan).

### Known limitations
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
- Clip transform values (position/scale/rotation/opacity) are stored and editable but not yet applied by any renderer. `Source In/Out` are read-only (changed by trimming).
- Tests cover logic and a scripted UI workflow with synthetic mouse events; no human has used the GUI on a real display.

## Next: Phase 3
Implement `VisualResearchService` against the structured scenes (never raw script sentences): per-source searchers (stock, YouTube, web, screenshots), candidate scoring against `VisualIntent` and the saved preferences (minimum accuracy, soft source targets, avoid repeats), a review/replace workflow that writes to the timeline as undoable commands, and a real model-backed `SemanticAnalyzer` and Whisper-class transcription validated on real narration.
