# AgentTool

Desktop **AI Video Director + Visual Researcher + Editor**. This repository currently contains **Phase 1: the desktop foundation and core project system**. No AI, web research or rendering is implemented yet — see [Status](#status).

## Run

```bash
pip install -r requirements.txt     # PySide6, pytest
# FFmpeg + ffprobe must be on PATH (or set their location in Settings)
python -m app.main [path/to/project_folder]
python -m pytest                    # 87 tests (Qt tests run headless via QT_QPA_PLATFORM=offscreen)
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

## Status

Implemented (Phase 1): project create/open/save/save-as/close with recent list; media import (copy or link) with probing, duplicate detection and cached background thumbnails; media library (search/sort/remove/drag); script editor; voice-over import/replace/remove/playback; single-file preview; 8-track timeline with add/rename/hide/mute/lock/delete tracks and add/move/trim/delete clips (mouse, inspector, drag-drop, snapping); inspector; undo/redo; autosave + crash recovery; job manager with status bar and jobs panel; settings; logging; renderer validation/plan.

Not implemented (visible but disabled / "Coming in Phase 2"): script analysis, transcription, Visuals/Review/Edit pages, AI providers, rendering (Export only validates and shows the plan).

### Known limitations
- Preview plays one file at a time; the timeline playhead is positional only (no composited timeline playback).
- Video frames could not be visually verified in the headless test environment (playback loading/duration is tested; on-screen rendering depends on the platform's Qt multimedia backend).
- Running jobs cannot be paused (only queued ones); jobs are not persisted across restarts. A crash mid-import can leave an unregistered file in `media/` (`*.part` files are cleaned on open).
- Removing media from the project keeps the file on disk (by design; no "delete files" action yet). Linked (non-copied) media breaks if the external file moves (reported as MISSING).
- Clip transform values (position/scale/rotation/opacity) are stored and editable but not yet applied by any renderer. `Source In/Out` are read-only (changed by trimming).
- Tests cover logic and a scripted UI workflow with synthetic mouse events; no human has used the GUI on a real display.

## Next: Phase 2
Script analysis + scene segmentation behind `AIProvider`, voice-over transcription behind `Transcriber`, scenes/`ai_decisions` populated and editable, the Visuals/Review pages with `VisualSource` implementations (stock, YouTube, web, screenshots), candidate scoring and a replace workflow that writes to the timeline as undoable commands.
