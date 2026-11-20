# Phase 11 — AI editorial intelligence: audit, implementation map and architecture

Written after reading the Phase 1–10 code that this phase has to reuse. The rule of the phase: an **analysis → recommendation → user review → structured, undoable edit** layer on top of the
existing systems. Nothing here researches visuals, scores candidates, edits the timeline by itself or renders.

## 1. Implementation map

### Existing and reusable (read, not rebuilt)
| Need | Existing system | How Phase 11 uses it |
|---|---|---|
| Scene understanding | `analysis/models.py` (`Scene`: topic, summary, importance, entities, claims, numbers, `user_edited_fields`, `boundaries_locked`; `SentenceAnalysis`: cues, terms, type scores; `VisualIntent`), `analysis/lexicon.py` cues | roles / relationships are derived from these, never from re-parsing the script |
| Project-wide context | `editing/context.py` `build_context()` → `EditingContext` (per scene: words, sentences, intent, assignment, asset info, candidate, score, prev/next neighbours, `new_entities`, `reuse_count`, `starts_section`, per-scene `input_hash`) | the editorial context **wraps** it and adds timeline items, candidate pool, captions / graphics, reference style, preferences, overrides |
| Visual evidence | `research/models.py` (`Candidate.evidence_kind`, `source_type`, `fingerprint`, `license`, `metadata`; `CandidateScore` with `basis: METADATA`; `SceneResearchState.ranked` alternatives; `VisualAssignment.selected_by`) | evidence alignment, diversity, continuity and replacement options come from this data; no re-scoring |
| Source preferences | `visual/preferences.py` `VisualPreferences` (soft source targets, `avoid_repeated_visuals`, `prefer_real_visuals` …) | source-balance input; the percentages stay preferences |
| Timing facts | `editing/timing.py` `narration_stats`, `speed_class`; `ShotTimingService` | content-aware pacing |
| Reference style | `editing/overrides.py` (`EditingStrategyOverrides`, priority doc: user lock > content > user style > reference > default), `editing/effective.py` (`active_overrides`, `style_signature`, `protected_parameters`), `reference/style_model.py`, `qc/style_compat.py` (`style_deviation`) | reference influence is read through `active_overrides`; the priority chain already exists and is reused verbatim |
| Manual ownership / locks | `Clip.created_by` / `ai_decision_id` / `scene_id` / `slot`, **Phase 10** `Clip.locks` + `Clip.protected` + `Track.locked`; `EditingDecision.created_by` + `OverrideRecord` (`editing/models.py`), `PresentationOverride`; `Scene.user_edited_fields` / `boundaries_locked`; `EditingSettings.user_set` / `CaptionSettings.user_set` / `AudioSettings.user_set`; `VisualAssignment.selected_by == "USER"` | one **OverrideResolver** reads all of these (no second store of truth). Only an explicit "keep this" / "reset" decision about a *recommendation* is stored by Phase 11 |
| QC overlap | `qc/continuity_checker.py` (repetition incl. intentional-reuse rules, scene/subject/location jumps, style clash), `qc/pacing_checker.py`, `qc/ai_editorial_checker.py` (provider abstraction: Local / API (strictly validated JSON) / Future), `qc/style_compat.py` | the editorial layer produces *recommendations with an action*, QC produces *findings*; recommendations carry `related_qc_issue_ids`, never contradict an open QC finding, and the provider pattern (deterministic local + validated API + unavailable) is copied |
| Timeline edits | **Phase 10** `ws.edit` (`TimelineEditService`) and planners (`plan_replace`, `plan_set_duration`, `plan_trim`, `plan_set_transition`, keyframe planners, `plan_move`, `plan_split` …), `PlanCommand`, `validate_plan`, `PlanEnv` | every approved change is one of these, validated by `validate_plan` in a dry run before it is offered |
| Presentation edits | `PresentationService.update_caption(emphasis=…)`, `add_graphic(...)`, `update_graphic`, `set_clip_audio`, `lock_clip`, `override_command`; `EditingService.replace_clip_asset`, `update_decision`, `lock_clip` | caption emphasis / on-screen text / audio changes reuse these (they already record AI-ownership overrides) |
| Background work | `JobManager` (priorities, `dedupe_key`, cancel, progress), `QCService` pattern (engine + staged checkers, partial results, incremental plan from the change tracker, `StoreQCRunCommand`) | the editorial service copies the QC service shape |
| Incremental work | Phase 9 `ProjectChangeTracker` (+ `ChangeSet`), `MediaCacheManager`, `AnalysisCache`, per-scene hashes | stage inputs are hashed per scene; only dirty scenes / windows are re-analysed |
| Persistence | `Project` fields + `project_schema.migrate_document` (additive), `phase8_commands.StoreQCRunCommand` pattern | schema 9 → 10, one `editorial` aggregate with the 13 named sections |
| UI | `main_window.NAV` pages, `QCPanel` / `ReferencePanel` patterns, `timeline_panel` signals, Phase 10 inspector | new "Editorial" page + a scene-level box in the existing inspector |

### Partially implemented
* Repetition / continuity / pacing / style **detection** exists (QC) but only as findings with navigation fixes; there is no story-level model, no recommended *action*, no apply path.
* Reference style influences generation (Phase 4/5) but nothing compares an *existing* timeline with it except QC notices.
* Manual ownership is recorded per clip / decision / setting, but there is no single "is this overridden?" answer per aspect.

### Missing (new in this phase)
Narrative roles, story arc and sections, scene relationships, visual-strategy roles, sequence assessment, temporal / identity / real-vs-illustrative continuity from metadata, content-aware pacing recommendations, emphasis planning, evidence-vs-claim alignment grading, audio-visual alignment, style-consistency outliers, preference adaptation (local, controllable), recommendations + validator + apply + preview, editorial scorecard / report, review UI, incremental editorial analysis, `CommandStack.transaction`.

### Requires modification
* `core/commands.py`: `CommandStack.transaction()` (record several commands as ONE undo step with rollback) so existing services (presentation, editing) can take part in a compound recommendation.
* `project.py` / `project_schema.py` / `constants.py`: schema 10, `Project.editorial`.
* `performance/change_tracker.py`: map the new store / apply commands; scene-level dependency ids for editorial stages.
* `core/config.py`: global (local-only) editorial preferences.
* `main_window.py`: new page, inspector scene box, signals.

### Potential conflicts and how they are resolved
1. **Phase 10 locks vs AI recommendations** — `validate_plan` already refuses locked items; the recommendation validator runs the same dry run and additionally marks `blocked_by_override`, so "Manual override preserved" is shown instead of an Apply button.
2. **Duplicate QC logic** — QC finds, editorial recommends; shared heuristics are imported (intentional-reuse rules, `narration_stats`, `is_evidence_visual`), not copied; open QC issues are linked, not repeated.
3. **Reference style** — only through `active_overrides` (already protected by user-set fields); the priority chain of the spec equals the one already documented in `editing/overrides.py`.
4. **Learning vs. existing settings** — learned preferences never write to `EditingSettings` etc.; they only bias recommendation thresholds and ranking, are local, off-switchable and per project or global.
5. **Cost** — `build_context` deep-copies per scene: editorial analysis runs on a snapshot in a worker job, never on the UI thread, per-scene hashed for incremental runs.
6. **No AI available** — every analyzer is deterministic; the AI provider only adds judgement text / roles when configured and its output is validated; a failed provider run is reported as failed, never as a completed assessment.

## 2. Architecture

```
EditorialService (jobs, staging, cache, persistence, incremental plan)           app/editorial_intelligence/editorial_service.py
  └─ build_editorial_context(project) -> EditorialContext (read-only snapshot)   context.py
  └─ stages (each: analyze(ctx, prior) -> StageOutput, per-scene input hashes)
       1 context          story_analyzer.py           project facts, content type, topic
       2 narrative        narrative_classifier.py     roles per scene (multi-role, confidence, evidence terms)
       3 story_arc        story_arc_analyzer.py       model + sections + key moments (+ structural findings)
       4 relationships    scene_relationship_analyzer.py   edges (prev/next/distant)
       5 sequence         visual_sequence_planner.py  visual strategy per scene + sequence windows + alternatives
       6 continuity       visual_consistency_analyzer.py   period / place / identity / real-vs-illustrative from metadata
       7 diversity        visual_diversity_analyzer.py     repetition (intentional vs unnecessary) + source balance
       8 evidence         evidence_alignment.py       Direct / Contextual / Illustrative / Decorative per claim scene
       9 pacing           pacing_intelligence.py      recommended hold per scene vs actual
      10 emphasis         emphasis_planner.py         grounded targets (numbers, dates, names, claims, reveals)
      11 av_alignment     (in emphasis_planner / pacing)   reveal / graphic timing vs spoken words
      12 style            style_consistency.py        outliers + reference-style conflicts
      13 recommendations  recommendation_engine.py  → recommendation_validator.py (dry run through Phase 10 planners)
  └─ preference_adapter.py (global/project, signals, reset)  recommendation apply: apply_builder.py (ActionSpec → one CommandStack transaction)
  └─ decision_explainer.py (concise factors, never reasoning)
  └─ EditorialIntelligenceProvider (Local deterministic | API validated | Unavailable)
```

Analysis, recommendations and edits stay separate: stages only produce data (`StageOutput`: stored section + `Finding`s); the engine turns findings into `Recommendation`s; applying is a separate user action that builds structured operations and executes them in ONE transaction.

### Data contracts (`editorial_models.py`)
* `SceneRole(scene_id, roles[(role, confidence)], evidence_terms, basis)`; `StoryStructure(model, confidence, sections[StorySection], key_moments, scene_refs)`.
* `Relation(from_scene, to_scene, type, confidence, evidence)`; `VisualStrategy(scene_id, kind, primary_subject, secondary_subject, needs_evidence, continue_previous, …)`.
* `Finding(finding_id, stage, category, scene_ids, timeline_item_ids, title, detail, confidence, impact, basis COMPUTED|JUDGMENT, evidence_refs, action: ActionSpec | None, options, related_qc_issue_ids)`.
* `Recommendation` (the exact fields of the spec §20) + `blocked_by_override`, `validation`, `preview`, `time_ranges`.
* `ActionSpec(kind, params, targets[TargetRef(scene_id, clip_id, slot, asset_id)])`, kinds: REPLACE_VISUAL, SET_SHOT_DURATION, MOVE_CLIP_START, SET_TRANSITION, REMOVE_TRANSITION, ADD_PUNCH_IN, REMOVE_MOTION, CAPTION_EMPHASIS, ADD_TEXT_EMPHASIS, SEARCH_AGAIN, EXPAND_SOURCES, GENERATE_AI_VISUAL, USE_EXISTING_EVIDENCE, MANUAL_SELECTION, KEEP_CURRENT, REVIEW_ONLY.
* `EditorialSettings(mode CONSERVATIVE|BALANCED|CREATIVE, content_type, enabled stages, preference_learning…)`, `Scorecard` (computed vs judgment, evidence refs, serious-warning floor), `EditorialReport`.

### Persistence (schema 9 → 10; `Project.editorial`)
`editorial_settings, narrative_structure, story_sections, scene_relationships, visual_sequence_analysis, visual_consistency_results, visual_diversity_results, pacing_recommendations, evidence_alignment_results, editorial_recommendations, editorial_overrides, user_editorial_preferences, editorial_analysis_history` (+ `stage_cache` hashes). Raw provider responses and any large payload live in `<project>/cache/editorial/` through the Phase 9 cache manager, not in `project.json`. Old projects open with defaults.

### Incremental analysis
Per-stage input hashes are built from per-scene hashes (`EditingContext.input_hash` + timeline items of the scene + candidate pool signature + preferences signature). `plan_incremental()` maps the change tracker's `ChangeSet` to: dirty scenes → window re-analysis (scene ± neighbours) for sequence / continuity / transitions, global diversity only when the asset/source set changed, emphasis only for text changes, story-level stages only when roles / sections changed or narration changed; when a dependency cannot be decided the broader stage reruns. Stale recommendations are marked `OUTDATED`.

## 3. Order of work
1. **Foundation (lead)**: models, `Project.editorial` + schema 10, store/apply commands, `CommandStack.transaction`, context builder, service skeleton with staged job, tracker mapping, deterministic/API provider shell, tests.
2. **Analyzers (3 agents in parallel)**: A = narrative + arc + relationships + content-type strategy; B = sequence + continuity + diversity/source balance + evidence; C = pacing + emphasis + AV alignment + style/reference.
3. **Recommendations (agent D, parallel with contracts)**: preference adapter, recommendation engine, validator, override resolver, apply builder, previews.
4. **UI (agent E)** after the service API is stable.
5. **Integration (lead)**: incremental, persistence round trips, performance at 100/500/1,000 scenes, end-to-end acceptance (spec §38), README, regression, final report.
