# Phase 10 — implementation plan (written after auditing the Phase 1–9 timeline)

## 1. What already works (reused as is)
| Area | Where | State |
|---|---|---|
| Authoritative model | `app/timeline/{timeline,track,clip,keyframes}.py` | Tracks hold clips; clips have timeline start / duration, `source_in/out`, `speed`, transform, keyframes, effects, `text`, `animation`, `audio`, `transition`, AI metadata (`scene_id`, `created_by`, `locked`, `ai_decision_id`). The renderer consumes the same model (`RenderSnapshot` → compiler). |
| Commands / undo | `app/core/commands.py` (`Command`, `CompositeCommand`, `CommandStack` with merge), `timeline_commands.py` | Single-clip move / trim / split / delete / add / properties; track add / remove / rename / flags / volume. `TimelineService._execute` runs the **edit hook**: a manual edit of an AI clip records an override (ownership) in the same undo step. |
| Phase 9 | `Timeline.index()`, `clips_in_range`, `snap_points`, `revision`; canvas viewport painting; cached snap list | Used unchanged. |
| Preview / render | `PreviewEngine` section keys are content-addressed; compiler already renders `speed`, `crop`, transitions FADE / DISSOLVE / WIPE / SLIDE, audio volume / fades / pan / volume keyframes, track hidden / muted / solo; no reverse playback. | Used unchanged. |
| Caption / graphic / audio services | `PresentationService.update_caption / update_graphic / set_clip_audio / lock_clip`; `EditingService.replace_clip_asset / lock_clip / override_command` | Reused for specialised clips; the new generic commands never bypass the edit hook. |
| UI | `timeline_canvas.py` (single selection, drag move / trim with snapping to edges + playhead, drop from library, QC markers), `timeline_panel.py` (track headers with H/M/S/L, zoom slider), `inspector_panel.py` (single clip numeric fields) | Extended, not rewritten. |

## 2. Partially implemented
* Selection: one clip (`ws.selected_clip_id`).  Snapping: edges + playhead at a fixed 8 px, no markers / scenes / captions, no guides, no hysteresis.  Split: only via the context menu at the playhead, one clip.  Locks: `Clip.locked` only means "protect from AI regeneration".  Keyframes: stored and rendered, editable only by AI / code.  Speed: a single inspector field.  Timecode: display-only, seconds are floats, not frame-exact.  Transitions: stored on the clip, no editor.
## 3. Missing
Multi-selection and group operations; ripple / roll / slip / slide; insert / overwrite / replace / duplicate / clipboard; links and groups; markers and a marked range; track reorder / height / collapse / duplicate / set type; keyframe editor; type-aware inspector with precise timecode input; transition editor; caption split / merge; audio fades / gain in the UI; keyboard shortcuts beyond Delete; auto-scroll; fit / zoom-to-playhead / zoom-to-selection; ripple-mode / snap / link toggles.

## 4. New architecture (small, focused)
```
UI gesture ─► TimelineEditService (app/services) ─► planner (pure: reads timeline, returns EditPlan)
                       │                                  │
                       │ edit hook (AI ownership)         ▼
                       └────────────────────────► PlanCommand  = validate → capture → apply (one rebuild per track, rollback on failure) → undo record
                                                              │
                                              Timeline model (authoritative) ─► preview / render / QC / caches (change tracker)
```
* `app/timeline/timebase.py` — rational frame arithmetic (`Timebase`): clips stay float seconds in the file, but every frame-aware edit is computed from integer frame indices, so repeated trims never drift; audio keeps sample precision; timecode parse / format.
* `app/timeline/plan.py` — `EditPlan`, `validate_plan`, `PlanCommand`: **all** advanced edits (multi-move, ripple, roll, slip, slide, insert, overwrite, replace, duplicate, paste, split-many, delete-many) are planners producing an `EditPlan`; one validator enforces track locks, user item / aspect locks, track compatibility, minimum duration, source bounds, keyframe range and overlaps *on the resulting configuration* before anything is changed.
* Small focused commands for what is not a clip-set edit: markers, range, track operations, user locks, links / groups, keyframes, transitions, properties.
* `app/timeline/snapping.py` — Qt-free snap engine (targets, priority, frame tolerance, hysteresis), fed by the Phase 9 index.
* `SelectionModel` in `Workspace` (ordered ids, primary id kept for old code) and `TimelineEditState` (tool, ripple mode / scope, snap on / tolerance, link on, follow playhead).

## 5. Model changes (schema 8 → 9, additive and optional)
`Clip.link_id`, `Clip.group_id`, `Clip.locks` (subset of item / position / timing / transform / audio / style; `Clip.protected` = legacy `locked` or any lock, now used by every AI regeneration check); `Track.height`, `Track.collapsed`; `Timeline.markers`, `Timeline.range`. Old projects open unchanged (migration only bumps the version); the bump makes older apps refuse a project whose links / markers they would drop.
**Render visibility:** only one concept exists — a hidden track is excluded from preview *and* export (the compiler skips it); the UI says so. Muted / solo affect audio only.
**Sync-lock:** realised as the ripple *scope* (current track / selected tracks / all unlocked tracks, linked and grouped items always follow); there is no separate per-track flag.

## 6. Preview / render interfaces touched
None need new fields: the renderer already reads start / duration / source / speed / keyframes / transition / audio of the same clips. New work is **consistency**: the Phase 9 `ProjectChangeTracker` gets a handler for `PlanCommand` (dirty scenes from before / after states); tests compare the timeline with the compiler's output and with a real export. Not offered because the renderer cannot reproduce it: reverse playback, speed ramps (a keyframed speed), range export (preview of a range is supported).

## 7. Order of work
foundation (done) → pure planners (basic ops, then ripple / roll / slip / slide) in parallel with metadata commands and the snapping / selection / service layer → service facade → UI (canvas, panel, inspector / keyframes) → integration (tracker, persistence, performance, acceptance) → regression.
