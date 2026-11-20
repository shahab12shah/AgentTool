"""EditorialEngine: runs the analyzers in dependency order, reuses what is still valid, keeps partial results.

    previous state + context  ->  per stage: decide what is dirty -> analyze(ctx, prior, scene_filter) -> merge into the previous section -> findings
                              ->  recommendations (separate step: engine only calls it, ``recommendation_engine`` owns it)  ->  scorecard

Re-analysis is decided from content hashes (never from a guess): a scene is dirty for a stage when its hash differs from the one the stage last analysed, or when the project-wide signature
(mode, content type, reference style, preferences, topic) changed - then every scene is. ``window`` stages also re-run the neighbours of a dirty scene, ``global`` stages re-run whenever
anything is dirty. A stage whose dependency failed is SKIPPED (its previous result stays, marked stale in its record), and one failing analyzer never discards another's result.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.serialization import to_plain
from app.editorial_intelligence.context import EditorialContext
from app.editorial_intelligence.editorial_models import (
    STAGE_IDS,
    EditorialState,
    Finding,
    Recommendation,
    Scorecard,
    StageOutput,
    StageRecord,
    now_iso,
)
from app.editorial_intelligence.stages import Stage, default_stages
from app.logging.logger import get_logger

_log = get_logger(__name__)
ProgressFn = Callable[[float, str, list[StageRecord]], None]


class EditorialCancelled(Exception):
    pass


def global_signature(ctx: EditorialContext) -> str:
    """Everything outside a single scene that can change what an analyzer concludes."""
    raw = json.dumps([ctx.settings.mode.value, ctx.content_type, ctx.reference.signature, ctx.preferences.signature, ctx.edit.topic, ctx.settings.use_reference_style,
                      sorted(ctx.settings.stages), ctx.fps], default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


@dataclass
class EngineResult:
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    stage_cache: dict[str, Any] = field(default_factory=dict)
    records: list[StageRecord] = field(default_factory=list)
    recommendations: list[Recommendation] | None = None
    scorecard: Scorecard | None = None
    status: str = "COMPLETE"  # COMPLETE | PARTIAL | FAILED | CANCELED
    log: list[str] = field(default_factory=list)
    analysed_scenes: set[str] = field(default_factory=set)
    outputs: dict[str, StageOutput] = field(default_factory=dict)


def _merge_section(previous: dict[str, Any] | None, new: dict[str, Any], analysed: set[str] | None, live_scenes: set[str]) -> dict[str, Any]:
    """Keep the previous result for scenes that were not analysed again, take the new one for those that were, drop scenes that no longer exist."""
    out: dict[str, Any] = {}
    prev_scenes = dict((previous or {}).get("scenes") or {})
    new_scenes = dict(new.get("scenes") or {})
    if analysed is None:  # a full run replaces everything
        merged = new_scenes
    else:
        merged = {sid: v for sid, v in prev_scenes.items() if sid not in analysed}
        merged.update(new_scenes)
    merged = {sid: v for sid, v in merged.items() if sid in live_scenes}
    if merged or "scenes" in new or "scenes" in (previous or {}):
        out["scenes"] = merged
    out["global"] = deepcopy(new.get("global", (previous or {}).get("global", {})))
    for k, v in new.items():
        if k not in ("scenes", "global"):
            out[k] = deepcopy(v)  # any other top-level keys are whole-stage values
    return out


class EditorialEngine:
    def __init__(self, stages: list[Stage] | None = None, recommend: Callable | None = None, scorecard: Callable | None = None) -> None:
        self.stages = stages if stages is not None else default_stages()
        self._recommend = recommend
        self._scorecard = scorecard

    # ------------------------------------------------------------------ planning
    def dirty_scenes(self, stage: Stage, ctx: EditorialContext, previous: EditorialState, gsig: str, scene_filter: set[str] | None, force: bool) -> set[str] | None:
        """The scenes a stage must analyse. ``None`` = all of them (a full run); an empty set = nothing to do."""
        cache = previous.stage_cache.get(stage.id) or {}
        have = previous.sections.get(stage.id)
        live = ctx.scene_ids
        if force or have is None or cache.get("gsig") != gsig:
            return None
        hashes: dict[str, str] = cache.get("scene_hashes") or {}
        dirty = {s.id for s in ctx.scenes if hashes.get(s.id) != s.hash}
        if scene_filter is not None:
            dirty &= scene_filter | {sid for sid in dirty if sid not in hashes}
        if set(hashes) - set(live):  # a scene disappeared
            dirty |= {sid for sid in live if sid not in hashes}
            if not dirty:
                dirty = set(live) if stage.scope == "global" else set()
        if not dirty:
            return set()
        if stage.scope == "global":
            return None
        if stage.scope == "window":
            k = ctx.settings.window
            expanded: set[str] = set()
            for sid in dirty:
                expanded |= {s.id for s in ctx.window(sid, k)}
            return expanded
        return dirty

    # ------------------------------------------------------------------ run
    def run(self, ctx: EditorialContext, previous: EditorialState, *, selected: set[str] | None = None, scene_filter: set[str] | None = None, force: bool = False,
            cancel: threading.Event | None = None, on_progress: ProgressFn | None = None, previous_recommendations: list[Recommendation] | None = None) -> EngineResult:
        res = EngineResult(sections=deepcopy(previous.sections), stage_cache=deepcopy(previous.stage_cache))
        gsig = global_signature(ctx)
        live = set(ctx.scene_ids)
        stages = [s for s in self.stages if (selected is None or s.id in selected) and (s.id in ctx.settings.stages or s.id not in STAGE_IDS)]  # the user can switch the built-in stages off; custom ones always run
        prior: dict[str, StageOutput] = {}
        findings_by_stage: dict[str, list[Finding]] = {}
        for f in previous.findings:
            findings_by_stage.setdefault(f.stage, []).append(f)
        failed: set[str] = set()
        total = len(stages) + 1
        records = [StageRecord(s.id) for s in stages] + [StageRecord("recommendations")]
        res.records = records

        def tick(i: int, msg: str, frac_in: float = 0.0) -> None:
            if on_progress is not None:
                on_progress(min(1.0, (i + frac_in) / total), msg, records)

        any_failed = False
        for i, stage in enumerate(stages):
            rec = records[i]
            if cancel is not None and cancel.is_set():
                for r in records[i:]:
                    r.status = "CANCELED"
                res.status = "CANCELED"
                res.log.append("Canceled; results of the finished stages are kept.")
                break
            tick(i, f"{stage.label}…")
            t0 = time.monotonic()
            rec.started_at = now_iso()
            if stage.analyze is None:
                rec.status, rec.error, rec.message = "FAILED", stage.problem, "not available"
                any_failed = True
                failed.add(stage.id)
                res.log.append(f"{stage.label}: unavailable ({stage.problem}).")
                continue
            if any(d in failed for d in stage.depends_on):
                rec.status, rec.message = "SKIPPED", "a stage it depends on failed"
                failed.add(stage.id)
                res.log.append(f"{stage.label}: skipped because a stage it depends on failed.")
                self._carry(stage, prior, res, findings_by_stage)
                continue
            dirty = self.dirty_scenes(stage, ctx, previous, gsig, scene_filter, force)
            if dirty is not None and not dirty:
                rec.status, rec.message, rec.scenes_reused = "REUSED", "nothing changed since the last analysis", len(live)
                res.log.append(f"{stage.label}: reused.")
                self._carry(stage, prior, res, findings_by_stage)
                rec.finished_at = now_iso()
                continue
            rec.status = "RUNNING"
            try:
                out = stage.analyze(ctx, prior, scene_filter=dirty)
            except EditorialCancelled:
                rec.status = "CANCELED"
                res.status = "CANCELED"
                break
            except Exception as exc:  # noqa: BLE001  (one analyzer failing must not discard the others)
                _log.exception("editorial stage %s failed", stage.id)
                rec.status, rec.error = "FAILED", f"{type(exc).__name__}: {exc}"[:300]
                rec.message = "failed; the previous result (if any) is kept and marked out of date"
                any_failed = True
                failed.add(stage.id)
                res.log.append(f"{stage.label}: FAILED ({rec.error}).")
                self._carry(stage, prior, res, findings_by_stage)
                rec.seconds = round(time.monotonic() - t0, 3)
                continue
            if out.status == "FAILED":
                rec.status, rec.error, rec.message = "FAILED", out.error[:300], out.message
                any_failed = True
                failed.add(stage.id)
                res.log.append(f"{stage.label}: FAILED ({out.error[:120]}).")
                self._carry(stage, prior, res, findings_by_stage)
                continue
            analysed = None if dirty is None else set(dirty)
            merged = _merge_section(res.sections.get(stage.id), out.section, analysed, live)
            res.sections[stage.id] = merged
            kept = [f for f in findings_by_stage.get(stage.id, []) if analysed is not None and not (set(f.scene_ids) & analysed) and set(f.scene_ids) <= live]
            stage_findings = kept + [f for f in out.findings if analysed is None or not f.scene_ids or set(f.scene_ids) & analysed or stage.scope == "global"]
            findings_by_stage[stage.id] = stage_findings
            cache = res.stage_cache.setdefault(stage.id, {})
            hashes = {} if dirty is None else {sid: h for sid, h in (cache.get("scene_hashes") or {}).items() if sid in live}
            for s in ctx.scenes:
                if dirty is None or s.id in dirty:
                    hashes[s.id] = s.hash
            res.stage_cache[stage.id] = {"gsig": gsig, "scene_hashes": hashes, "analysed_at": now_iso()}
            res.analysed_scenes |= (live if dirty is None else dirty)
            rec.status = "PARTIAL" if out.status == "PARTIAL" else "DONE"
            rec.scenes_analysed, rec.scenes_reused = (len(live), 0) if dirty is None else (len(dirty), len(live) - len(dirty))
            rec.findings = len(stage_findings)
            rec.message = out.message
            rec.input_hash = gsig
            res.log.append(f"{stage.label}: {rec.status.lower()} ({rec.scenes_analysed} analysed, {rec.scenes_reused} reused, {rec.findings} findings).")
            full = StageOutput(stage.id, merged, stage_findings, dict(hashes), out.status, out.message, out.error, dict(out.metrics), list(out.notes), list(out.scenes_analysed))
            prior[stage.id] = full
            res.outputs[stage.id] = full
            rec.finished_at = now_iso()
            rec.seconds = round(time.monotonic() - t0, 3)
        res.findings = [f for fs in findings_by_stage.values() for f in fs if set(f.scene_ids) <= live]
        if res.status != "CANCELED":
            self._finish(ctx, res, prior, records[-1], previous_recommendations or previous.editorial_recommendations, any_failed, tick, len(stages))
        if res.status == "COMPLETE" and any_failed:
            res.status = "PARTIAL"
        if res.status == "COMPLETE" and any(r.status == "PARTIAL" for r in records):
            res.status = "PARTIAL"
        tick(total, "Done")
        return res

    @staticmethod
    def _carry(stage: Stage, prior: dict[str, StageOutput], res: EngineResult, findings_by_stage: dict[str, list[Finding]]) -> None:
        """A stage that did not run still exposes its last result to the stages after it."""
        if stage.id in res.sections:
            prior[stage.id] = StageOutput(stage.id, res.sections[stage.id], findings_by_stage.get(stage.id, []), {}, "DONE")

    def _finish(self, ctx: EditorialContext, res: EngineResult, prior: dict[str, StageOutput], rec: StageRecord, previous_recs: list[Recommendation], any_failed: bool, tick, i: int) -> None:
        rec.started_at = now_iso()
        t0 = time.monotonic()
        if self._recommend is None:
            rec.status, rec.message = "SKIPPED", "no recommendation engine"
            return
        tick(i, "Recommendations…")
        try:
            res.recommendations = self._recommend(ctx, res.findings, previous_recs, res.sections)
            rec.status = "PARTIAL" if any_failed else "DONE"
            rec.findings = len(res.recommendations)
            res.log.append(f"Recommendations: {len(res.recommendations)}.")
        except Exception as exc:  # noqa: BLE001
            _log.exception("editorial recommendations failed")
            rec.status, rec.error = "FAILED", f"{type(exc).__name__}: {exc}"[:300]
            res.log.append(f"Recommendations: FAILED ({rec.error}).")
            res.recommendations = None
        if self._scorecard is not None:
            try:
                res.scorecard = self._scorecard(ctx, res.sections, res.findings)
            except Exception:  # noqa: BLE001
                _log.exception("editorial scorecard failed")
        rec.seconds = round(time.monotonic() - t0, 3)
        rec.finished_at = now_iso()


__all__ = ["EditorialEngine", "EngineResult", "EditorialCancelled", "global_signature", "to_plain"]
