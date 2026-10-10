"""QCEngine: runs the checkers in pipeline order, caches unchanged analysis, isolates failures, aggregates issues, classifies severity, applies the user's
ignores and computes the scores. Pure logic (no Qt, no project mutation): the service installs the result.

    Preflight -> Timeline -> Scene coverage -> Sync -> Visual accuracy -> Continuity/repetition -> Pacing/cut timing -> Captions -> Text -> Motion -> Transitions
    -> Audio -> Frames -> Assets/media -> Render readiness -> AI editorial review -> aggregation -> severity -> scores -> export decision
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import QCCancelled, QCContext
from app.qc.issue_model import (
    CATEGORY_GROUP, CheckerState, CheckerStatus, IgnoreRecord, IssueStatus, QCCategory, QCIssue, QCRun, new_run_id, now_iso,
)
from app.qc.scoring import compute_scores
from app.qc.settings import QCSettings
from app.qc.severity import Severity, cap_for_confidence

log = logging.getLogger(__name__)

# (checker id, module, class): the spec's pipeline order. A module that fails to import becomes a *failed checker*, never a failed engine.
REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("preflight", "app.qc.preflight", "PreflightChecker"),
    ("timeline", "app.qc.timeline_checker", "TimelineChecker"),
    ("scene", "app.qc.scene_checker", "SceneChecker"),
    ("sync", "app.qc.sync_checker", "SyncChecker"),
    ("visual", "app.qc.visual_checker", "VisualChecker"),
    ("continuity", "app.qc.continuity_checker", "ContinuityChecker"),
    ("pacing", "app.qc.pacing_checker", "PacingChecker"),
    ("caption", "app.qc.caption_checker", "CaptionChecker"),
    ("text", "app.qc.text_checker", "TextChecker"),
    ("motion", "app.qc.motion_checker", "MotionChecker"),
    ("transition", "app.qc.transition_checker", "TransitionChecker"),
    ("audio", "app.qc.audio_checker", "AudioChecker"),
    ("frames", "app.qc.frame_checker", "FrameChecker"),
    ("asset", "app.qc.asset_checker", "AssetChecker"),
    ("render", "app.qc.render_checker", "RenderReadinessChecker"),
    ("editorial", "app.qc.ai_editorial_checker", "EditorialChecker"),
)
LABELS = {
    "preflight": "Preflight", "timeline": "Timeline Integrity", "scene": "Scene Coverage", "sync": "Narration Sync", "visual": "Visual Accuracy", "continuity": "Visual Continuity",
    "pacing": "Pacing", "caption": "Captions", "text": "Text & Graphics", "motion": "Motion", "transition": "Transitions", "audio": "Audio", "frames": "Black / Frozen Frames",
    "asset": "Asset Integrity", "render": "Render Readiness", "editorial": "AI Editorial Review",
}
PIPELINE = tuple(r[0] for r in REGISTRY)
CHECKER_CATEGORIES: dict[str, tuple[QCCategory, ...]] = {
    "preflight": (QCCategory.PREFLIGHT,), "timeline": (QCCategory.TIMELINE,), "scene": (QCCategory.SCENE_COVERAGE,), "sync": (QCCategory.SYNC,), "visual": (QCCategory.VISUAL_ACCURACY,),
    "continuity": (QCCategory.CONTINUITY, QCCategory.VISUAL_REPETITION), "pacing": (QCCategory.PACING, QCCategory.CUT_TIMING, QCCategory.STYLE), "caption": (QCCategory.CAPTION,),
    "text": (QCCategory.TEXT, QCCategory.FACT_REVIEW), "motion": (QCCategory.MOTION,), "transition": (QCCategory.TRANSITION,), "audio": (QCCategory.AUDIO, QCCategory.SILENCE),
    "frames": (QCCategory.FRAMES,), "asset": (QCCategory.ASSET, QCCategory.MEDIA_QUALITY), "render": (QCCategory.RENDER_READINESS,), "editorial": (QCCategory.EDITORIAL,),
}

ProgressCb = Callable[[float, str, dict[str, CheckerStatus]], None]  # (overall 0..1, message, all statuses)


class _Failed(BaseChecker):
    """Stands in for a checker whose module could not be imported."""

    def __init__(self, cid: str, error: str) -> None:
        self.id, self.label, self.error = cid, LABELS.get(cid, cid), error

    def run(self, ctx: QCContext, report) -> CheckerOutput:  # noqa: ARG002
        raise RuntimeError(self.error)


def load_checkers(ids: tuple[str, ...] | None = None) -> list[BaseChecker]:
    out: list[BaseChecker] = []
    for cid, module, cls in REGISTRY:
        if ids is not None and cid not in ids:
            continue
        try:
            checker = getattr(importlib.import_module(module), cls)()
            checker.id = cid
            checker.label = LABELS.get(cid, checker.label)
            out.append(checker)
        except Exception as exc:  # noqa: BLE001
            log.warning("QC checker %s could not be loaded: %s", cid, exc, exc_info=not isinstance(exc, ImportError))
            out.append(_Failed(cid, f"The {LABELS.get(cid, cid)} checker could not be loaded ({type(exc).__name__}: {exc})"))
    return out


@dataclass
class PreviousState:
    """What an earlier run left behind: the current issues and each checker's cache entry. Nothing here is trusted blindly: a cache entry is only reused when its input hash matches."""

    issues: list[QCIssue] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)  # checker id -> {"input_hash", "scene_hashes", "metrics", "state"}


@dataclass
class EngineResult:
    run: QCRun
    cache: dict[str, Any]
    outputs: dict[str, CheckerOutput] = field(default_factory=dict)
    complete: bool = True  # every enabled checker's findings were verified against the project as it is now (a scene / category run, a retry or a cancel may leave some unverified)


def merge_metrics(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Metrics of a scene-level re-analysis laid over the previous ones: tables keyed by scene (dict values) are merged entry by entry, so the scenes that were not re-run keep
    their numbers, and the averages that are derived from such a table (visual fit) are recomputed from the whole table."""
    out = dict(old)
    for k, v in new.items():
        out[k] = {**old[k], **v} if isinstance(v, dict) and isinstance(old.get(k), dict) else v
    table = out.get("per_scene")
    if isinstance(table, dict) and table:
        for key, field_ in (("mean_current_score", "current"), ("mean_original_score", "original")):
            vals = [r[field_] for r in table.values() if isinstance(r, dict) and isinstance(r.get(field_), (int, float))]
            if key in out and vals:
                out[key] = round(sum(vals) / len(vals), 1)
    return out


def classify(issue: QCIssue, settings: QCSettings) -> QCIssue:
    """Severity classification: a judgement (anything below 100 % confidence, and every AI finding) is never presented as stronger than its confidence allows."""
    if issue.confidence < 99.5 or issue.detection_source.startswith("ai:"):
        caps = tuple((float(a), str(b)) for a, b in settings.ai_confidence_caps)
        capped = cap_for_confidence(issue.severity, issue.confidence, caps)
        if capped is not issue.severity:
            issue.metrics.setdefault("original_severity", issue.severity.value)
            issue.severity = capped
    return issue


def refresh_fix_flags(issue: QCIssue, settings: QCSettings) -> None:
    """Make an issue's auto-fix flags agree with the *current* fix permissions (a cached issue may have been built under different ones). Protected elements stay protected."""
    if issue.fix is None or issue.locked:
        return
    perm = settings.permission(issue.fix.kind)
    if perm == "never":
        issue.auto_fix_available, issue.auto_fix_safe, issue.fix_blocked_reason = False, False, "Disabled in QC settings"
        return
    if issue.fix_blocked_reason == "Disabled in QC settings":
        issue.fix_blocked_reason = ""
    if not issue.fix_blocked_reason:
        issue.auto_fix_available = True
        issue.auto_fix_safe = bool(issue.fix.intrinsic_safe and perm == "auto")
        issue.fix.safe, issue.fix.needs_confirmation = issue.auto_fix_safe, not issue.auto_fix_safe


def apply_ignores(issues: list[QCIssue], ignores: list[IgnoreRecord]) -> None:
    for i in issues:
        if i.severity is Severity.CRITICAL:
            continue  # a CRITICAL can never be waved through (not even by an "ignore this type" recorded for a milder finding of the same kind)
        rec = next((r for r in ignores if r.matches(i)), None)
        if rec is not None:
            i.ignored_by_user, i.status, i.ignore_reason = True, IssueStatus.IGNORED, rec.reason


def _dedupe(issues: list[QCIssue]) -> list[QCIssue]:
    seen: dict[tuple, QCIssue] = {}
    for i in issues:
        key = (i.code, i.fingerprint or i.issue_id)
        if key not in seen or i.confidence > seen[key].confidence:
            seen[key] = i
    return list(seen.values())


class QCEngine:
    def __init__(self, checkers: list[BaseChecker] | None = None) -> None:
        self.checkers = checkers if checkers is not None else load_checkers()

    # ------------------------------------------------------------------ selection
    def select(self, settings: QCSettings, checker_ids: list[str] | None = None, categories: list[str] | None = None, scenes_only: bool = False) -> set[str]:
        """Checker ids to run. ``checker_ids`` / ``categories`` narrow a run (category QC, retry); ``scenes_only`` keeps the scene-local checkers (scene QC)."""
        chosen = {c.id for c in self.checkers if c.id in settings.enabled_checkers or c.id == "preflight"}
        if checker_ids:
            chosen &= set(checker_ids)
        if categories:
            cats = {str(c).upper() for c in categories}
            chosen &= {c.id for c in self.checkers if cats & {k.value for k in c.categories} or c.id in {x.lower() for x in cats}}
        if scenes_only:
            chosen &= {c.id for c in self.checkers if c.scene_local}
        return chosen

    # ------------------------------------------------------------------ the run
    def run(self, ctx: QCContext, *, previous: PreviousState | None = None, ignores: list[IgnoreRecord] | None = None, selected: set[str] | None = None, trigger: str = "manual",
            number: int = 1, scope: dict[str, Any] | None = None, progress: ProgressCb | None = None, use_cache: bool = True, run_id: str | None = None,
            project_version: str = "", log_cb: Callable[[str], None] | None = None) -> EngineResult:
        t0 = time.monotonic()
        previous = previous or PreviousState()
        settings = ctx.settings
        selected = selected if selected is not None else self.select(settings)
        run = QCRun(run_id or new_run_id(), number, trigger=trigger, scope=dict(scope or {}), project_version=project_version, timeline_version=ctx.project.timeline_version,
                    settings_version=settings.version())
        statuses = {c.id: CheckerStatus(c.id, LABELS.get(c.id, c.label)) for c in self.checkers}
        run.checkers = statuses
        outputs: dict[str, CheckerOutput] = {}
        new_cache: dict[str, Any] = {}
        collected: list[QCIssue] = []
        total_w = sum(c.weight for c in self.checkers) or 1.0
        done_w = 0.0
        integrity_ok = True
        prev_by_checker: dict[str, list[QCIssue]] = {}
        for i in previous.issues:
            prev_by_checker.setdefault(i.checker, []).append(i)
        enabled = self.select(settings)  # what the settings switch on; a checker the user switched off keeps no findings
        carried = {id(i) for i in previous.issues}  # findings re-used from the previous run (not detected again): their FIXED / ignored marks must survive
        unverified: set[str] = set()  # enabled checkers whose findings still come from an earlier state of the project

        def keep_previous(cid: str) -> None:
            """A checker that did not run keeps its last findings and cache entry; the run is only 'current' if those are still valid for this project."""
            collected.extend(prev_by_checker.get(cid, []))
            entry = previous.cache.get(cid)
            if entry:
                new_cache[cid] = entry
                outputs[cid] = CheckerOutput(list(prev_by_checker.get(cid, [])), dict(entry.get("metrics") or {}))
                ctx.shared[cid] = outputs[cid]
            if cid in enabled and not self._verified(chk_by_id[cid], ctx, previous):
                unverified.add(cid)

        chk_by_id = {c.id: c for c in self.checkers}

        def say(msg: str) -> None:
            run.log.append(msg)
            if log_cb:
                log_cb(msg)

        def tick(frac_in_checker: float, msg: str, w: float) -> None:
            if progress:
                progress(min(1.0, (done_w + w * max(0.0, min(1.0, frac_in_checker))) / total_w), msg, statuses)

        canceled = False
        for chk in self.checkers:
            st = statuses[chk.id]
            if canceled:
                st.state, st.message = CheckerState.CANCELED, "Canceled"
                keep_previous(chk.id)  # an unreached checker keeps what the last run found
                continue
            if chk.id not in selected:
                if chk.id in enabled:
                    st.state, st.message = CheckerState.SKIPPED, "Not part of this run"
                    keep_previous(chk.id)
                else:
                    st.state, st.message = CheckerState.SKIPPED, "Switched off in the QC settings"  # its old findings would never be refreshed: they go
                done_w += chk.weight
                tick(0.0, f"{st.label}: skipped", 0.0)
                continue
            if chk.expensive and not integrity_ok:
                st.state, st.message = CheckerState.SKIPPED, "Skipped: fix the preflight problems first"
                collected += prev_by_checker.get(chk.id, [])
                say(f"{st.label}: skipped (project integrity problems)")
                done_w += chk.weight
                tick(0.0, st.message, 0.0)
                continue
            st.state, st.message = CheckerState.RUNNING, "Running"
            tick(0.0, f"{st.label}…", chk.weight)
            started = time.monotonic()
            try:
                ctx.check_cancel()
                out, partial = self._run_one(chk, ctx, previous, prev_by_checker.get(chk.id, []), use_cache, st, lambda f, m: tick(f, f"{st.label}: {m}" if m else st.label, chk.weight))
                if partial:
                    unverified.add(chk.id)
                for i in out.issues:
                    i.checker = chk.id  # the engine owns attribution (cache reuse and "retry this checker" rely on it)
                outputs[chk.id] = out
                ctx.shared[chk.id] = out
                st.seconds = time.monotonic() - started
                st.issue_count = len(out.issues)
                if st.state is CheckerState.RUNNING:
                    st.state = CheckerState.DONE
                st.message = "; ".join(out.notes[:2]) if out.notes else ("Cached" if st.state is CheckerState.CACHED else "Done")
                collected += out.issues
                new_cache[chk.id] = {"input_hash": st.input_hash, "scene_hashes": dict(st.scene_hashes), "metrics": out.metrics, "state": st.state.value, "version": chk.version}
                run.cache_hits += 1 if st.state is CheckerState.CACHED else 0
                if chk.id == "preflight" and out.metrics.get("integrity_ok") is False:
                    integrity_ok = False
                    say("Preflight found blocking integrity problems: deeper analysis is skipped")
                for n in out.notes:
                    say(f"{st.label}: {n}")
            except QCCancelled:
                st.state, st.message = CheckerState.CANCELED, "Canceled"
                canceled = True
                keep_previous(chk.id)
                say(f"{st.label}: canceled")
            except Exception as exc:  # noqa: BLE001  (a failing checker never fails the run)
                st.state, st.error = CheckerState.FAILED, f"{type(exc).__name__}: {exc}"
                st.message = f"Failed: {exc}"
                collected += prev_by_checker.get(chk.id, [])  # conservative: what was known before stays visible
                log.warning("QC checker %s failed", chk.id, exc_info=True)
                say(f"{st.label}: FAILED — {exc}")
            done_w += chk.weight
            tick(1.0, st.message or st.label, 0.0)

        issues = self._aggregate(collected, ctx, previous, ignores or [], run.run_id, carried)
        failed_groups, failed_ids = self._failed_groups(statuses, previous)
        run.issues = issues
        run.scores = compute_scores(issues, settings, ctx.duration, len(ctx.project.scenes), failed_groups, failed_ids)
        run.metrics = {cid: o.metrics for cid, o in outputs.items() if o.metrics}
        run.state = "CANCELED" if canceled else "PARTIAL" if failed_ids or any(s.state is CheckerState.CANCELED for s in statuses.values()) else "COMPLETED"
        run.seconds = time.monotonic() - t0
        run.finished_at = now_iso()
        return EngineResult(run, new_cache, outputs, complete=not unverified)

    # ------------------------------------------------------------------ one checker (cache / scene-level reuse / full)
    @staticmethod
    def _verified(chk: BaseChecker, ctx: QCContext, previous: PreviousState) -> bool:
        """True when the checker's last result was computed from exactly the project (and settings) QC is looking at now."""
        entry = previous.cache.get(chk.id)
        if not entry or entry.get("version") != chk.version or entry.get("state") not in ("DONE", "CACHED") or not entry.get("input_hash"):
            return False
        try:
            return entry["input_hash"] == chk.input_hash(ctx)
        except Exception:  # noqa: BLE001  (a key that cannot be computed verifies nothing)
            return False

    def _run_one(self, chk: BaseChecker, ctx: QCContext, previous: PreviousState, prev_issues: list[QCIssue], use_cache: bool, st: CheckerStatus,
                 report: Callable[[float, str], None]) -> tuple[CheckerOutput, bool]:
        """(output, partial). ``partial``: only some scenes were analysed and the rest keep findings that may be out of date (a scene run after other scenes changed): the cache
        entry is then stored as *not valid for this project* and the run is not 'current', so the next full run re-analyses what was left."""
        entry = previous.cache.get(chk.id) if use_cache else None
        valid_entry = bool(entry and entry.get("version") == chk.version and entry.get("state") in ("DONE", "CACHED"))
        full_hash = chk.input_hash(ctx)
        st.input_hash = full_hash
        explicit_scenes = ctx.scene_filter is not None and chk.scene_local
        same_inputs = bool(valid_entry and entry["input_hash"] == full_hash)  # type: ignore[index]
        if same_inputs and not explicit_scenes:
            st.state = CheckerState.CACHED
            return CheckerOutput(list(prev_issues), dict(entry.get("metrics") or {})), False  # type: ignore[union-attr]
        if chk.scene_local and (valid_entry or explicit_scenes):
            scenes = [s.id for s in ctx.scenes]
            hashes = {sid: chk.scene_input_hash(ctx, sid) for sid in scenes}
            st.scene_hashes = dict(hashes)
            old = (entry or {}).get("scene_hashes") or {} if valid_entry else {}
            global_prev = [i for i in prev_issues if not i.scene_id]
            if explicit_scenes:
                changed = {sid for sid in ctx.scene_filter if sid in hashes}  # type: ignore[union-attr]
                covers_all = changed == set(scenes)
                # scenes this run does not look at but whose inputs differ from what the cache was built on keep OLD findings: remember that they are out of date
                stale = set() if (same_inputs or covers_all) else {sid for sid, h in hashes.items() if sid not in changed and old.get(sid) != h}
                partial = bool(stale) or (bool(global_prev) and not same_inputs and not covers_all)
                saved = ctx.scene_filter
                ctx.scene_filter = set(changed)
                try:
                    out = chk.run(ctx, report)
                finally:
                    ctx.scene_filter = saved
                fresh_global = [i for i in out.issues if not i.scene_id]
                seen = {i.fingerprint for i in fresh_global if i.fingerprint}
                keep = [i for i in prev_issues if i.scene_id in hashes and i.scene_id not in changed]
                if not covers_all:
                    keep += [i for i in global_prev if i.fingerprint not in seen]  # a scene run cannot re-judge what spans the whole project: those findings stay as they were
                st.reused_scenes, st.analyzed_scenes = len(scenes) - len(changed), len(changed)
                merged = CheckerOutput(keep + [i for i in out.issues if i.scene_id in changed or not i.scene_id], merge_metrics((entry or {}).get("metrics", {}), out.metrics),
                                       list(out.notes) + [f"re-analysed {len(changed)} of {len(scenes)} scenes"], out.complete)
                if partial:
                    st.input_hash = ""  # never equal to a real key: the entry does not vouch for the whole project
                    for sid in stale:
                        st.scene_hashes[sid] = ""
                st.state = CheckerState.DONE
                return merged, partial
            changed = {sid for sid, h in hashes.items() if old.get(sid) != h}
            if valid_entry and not global_prev and changed != set(scenes) and old:
                saved = ctx.scene_filter
                ctx.scene_filter = set(changed)
                try:
                    out = chk.run(ctx, report)
                finally:
                    ctx.scene_filter = saved
                keep = [i for i in prev_issues if i.scene_id in hashes and i.scene_id not in changed]
                st.reused_scenes, st.analyzed_scenes = len(scenes) - len(changed), len(changed)
                merged = CheckerOutput(keep + [i for i in out.issues if i.scene_id in changed or not i.scene_id], merge_metrics((entry or {}).get("metrics", {}), out.metrics),
                                       list(out.notes) + [f"re-analysed {len(changed)} of {len(scenes)} scenes"], out.complete)
                st.state = CheckerState.DONE
                return merged, False
            st.analyzed_scenes = len(scenes)
        out = chk.run(ctx, report)
        if chk.scene_local and not st.scene_hashes:
            st.scene_hashes = {s.id: chk.scene_input_hash(ctx, s.id) for s in ctx.scenes}
        return out, False

    # ------------------------------------------------------------------ aggregation
    def _aggregate(self, issues: list[QCIssue], ctx: QCContext, previous: PreviousState, ignores: list[IgnoreRecord], run_id: str, carried: set[int] | None = None) -> list[QCIssue]:
        settings = ctx.settings
        out: list[QCIssue] = []
        for i in issues:
            if i.confidence < settings.min_confidence_to_report and (i.detection_source.startswith("ai:") or i.confidence < 99.5):
                continue  # too uncertain to say anything
            i.run_id = run_id
            if i.status in (IssueStatus.FIXED, IssueStatus.OBSOLETE) and id(i) not in (carried or ()):
                i.status = IssueStatus.OPEN  # detected again by an analysis that ran: the earlier fix did not resolve it (a finding merely carried over keeps its mark)
            if i.status is IssueStatus.IGNORED and not i.ignored_by_user:
                i.status = IssueStatus.OPEN
            i.ignored_by_user, i.ignore_reason = False, ""
            if not i.fingerprint:
                i.fingerprint = i.make_fingerprint()
            refresh_fix_flags(i, settings)
            out.append(classify(i, settings))
        out = _dedupe(out)
        apply_ignores(out, ignores)
        out.sort(key=lambda i: i.sort_key)
        return out

    @staticmethod
    def _failed_groups(statuses: dict[str, CheckerStatus], previous: PreviousState) -> tuple[list[str], list[str]]:  # noqa: ARG004
        """Score groups whose checkers did not complete are 'not analysed' (shown as —), never 100."""
        groups: set[str] = set()
        ids: list[str] = []
        for cid, st in statuses.items():
            if st.state in (CheckerState.FAILED, CheckerState.CANCELED):
                ids.append(cid)
                groups |= {CATEGORY_GROUP[cat] for cat in CHECKER_CATEGORIES.get(cid, ())}
        return sorted(groups), ids





