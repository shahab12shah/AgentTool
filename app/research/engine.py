"""Research engine: queries -> providers -> candidate pool -> dedupe -> evaluation -> ranking -> status.

Qt-free and project-free (inputs in, ``ResearchOutcome`` out), so it can run on a worker thread and be tested alone.
A provider that fails never fails the scene: its error is recorded in a ``ProviderReport`` and the others continue.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.exceptions import JobCancelled, ProviderError
from app.logging.logger import get_logger
from app.media.asset import SourceType as S
from app.research.dedupe import deduplicate, fingerprint_file, identity_keys
from app.research.evaluation import VisualEvaluationService
from app.research.http import HttpClient
from app.research.models import (
    Candidate,
    CandidateScore,
    CandidateStatus,
    EvidenceKind,
    ProviderReport,
    RankedEntry,
    ResearchBrief,
    ResearchQuery,
    ResearchSettings,
    ResearchStatus,
)
from app.research.providers.base import ProviderRegistry, SearchContext, SourceProvider
from app.research.ranking import KIND_OF, TYPE_OF, Ranker, RankingHistory
from app.visual.preferences import SourceKind, VisualPreferences

_log = get_logger(__name__)
PER_PROVIDER = 2  # concurrent requests per provider


@dataclass
class RunContext:
    cache_dir: Path  # project/cache/research
    project_root: Path | None = None
    should_cancel: Callable[[], bool] = lambda: False
    progress: Callable[[float, str], None] = lambda f, m: None
    ffmpeg_path: str = ""
    fresh: bool = False  # bypass the search cache ("Fresh Search")
    expand: bool = False  # user-approved: also search sources they have disabled


@dataclass
class ResearchOutcome:
    candidates: list[Candidate]
    scores: dict[str, CandidateScore]
    ranked: list[RankedEntry]
    reports: list[ProviderReport]
    status: ResearchStatus
    message: str
    duplicates_removed: int = 0
    searched_sources: list[str] = field(default_factory=list)
    skipped_disabled: list[str] = field(default_factory=list)  # enabled-by-nobody sources that might have helped


def allowed_source_types(prefs: VisualPreferences, expand: bool) -> list[S]:
    return [TYPE_OF[k] for k in SourceKind if expand or prefs.setting(k).enabled]


class ResearchEngine:
    def __init__(self, registry: ProviderRegistry, evaluation: VisualEvaluationService | None = None, ranker: Ranker | None = None,
                 http: HttpClient | None = None) -> None:
        self.registry = registry
        self.evaluation = evaluation
        self.ranker = ranker
        self.http = http or HttpClient()
        self._sems: dict[str, threading.Semaphore] = {}

    # ------------------------------------------------------------------ search
    def run(self, brief: ResearchBrief, queries: list[ResearchQuery], prefs: VisualPreferences, settings: ResearchSettings,
            history: RankingHistory, ctx: RunContext, id_factory: Callable[[], str],
            existing: list[Candidate] | None = None) -> ResearchOutcome:
        evaluation = self.evaluation or VisualEvaluationService(weights=settings.weights)
        ranker = self.ranker or Ranker(settings)
        allowed = allowed_source_types(prefs, ctx.expand)
        ctx.cache_dir.mkdir(parents=True, exist_ok=True)
        sctx = SearchContext(brief, ctx.cache_dir, ctx.should_cancel, settings.default_clip_seconds)

        tasks: list[tuple[int, int, ResearchQuery, S, SourceProvider]] = []
        reports: dict[str, ProviderReport] = {}
        for qi, q in enumerate(sorted(queries, key=lambda q: q.priority)):
            order = [S(x) for x in q.source_preferences if x in S._value2member_map_]
            for si, st in enumerate(order):
                if st not in allowed:
                    continue
                providers = self.registry.for_source(st)
                if not providers:
                    reports.setdefault(f"({st.value})", ProviderReport(f"({st.value})", "UNAVAILABLE", error=f"No provider is installed for {st.value}."))
                for p in providers:
                    tasks.append((qi, si, q, st, p))
        # providers that cannot run (missing key...) are reported once, not retried per query
        usable: dict[str, tuple[bool, str]] = {}
        for _, _, _, _, p in tasks:
            usable.setdefault(p.name, p.is_available())
        for name, (ok, why) in usable.items():
            if not ok:
                reports[name] = ProviderReport(name, "UNAVAILABLE", error=why)
        tasks = [t for t in tasks if usable[t[4].name][0]]

        raw: list[tuple[int, int, Candidate]] = []
        errors: dict[str, list[str]] = {}
        successes: dict[str, int] = {}
        cached: dict[str, int] = {}
        counts: dict[str, int] = {}
        qcount: dict[str, int] = {}
        started: dict[str, float] = {}
        total = max(len(tasks), 1)
        done = 0
        lock = threading.Lock()

        def work(task):
            qi, si, q, st, p = task
            if ctx.should_cancel():
                raise JobCancelled()
            sem = self._sems.setdefault(p.name, threading.Semaphore(PER_PROVIDER))
            with sem:
                t0 = time.monotonic()
                key = self._cache_key(p, q, st, settings)
                hit = None if (ctx.fresh or p.name == "ai_image") else self._cache_get(ctx.cache_dir, key, settings.cache_days)
                if hit is not None:
                    return task, hit, True, time.monotonic() - t0, None
                try:
                    found = p.search(q, st, settings.max_results_per_query, sctx)
                except JobCancelled:
                    raise
                except ProviderError as exc:
                    return task, [], False, time.monotonic() - t0, exc.user_message + (f" ({exc.details})" if exc.details else "")
                except Exception as exc:  # a buggy provider must not take the scene down
                    _log.exception("Provider crashed", extra={"provider": p.name})
                    return task, [], False, time.monotonic() - t0, f"{type(exc).__name__}: {exc}"
                if p.name != "ai_image":
                    self._cache_put(ctx.cache_dir, key, found)
                return task, found, False, time.monotonic() - t0, None

        with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
            futures = [pool.submit(work, t) for t in tasks]
            for fut in as_completed(futures):
                task, found, was_cached, secs, err = fut.result()
                qi, si, q, st, p = task
                with lock:
                    done += 1
                    qcount[p.name] = qcount.get(p.name, 0) + 1
                    started[p.name] = started.get(p.name, 0.0) + secs
                    if err:
                        errors.setdefault(p.name, []).append(err)
                    else:
                        successes[p.name] = successes.get(p.name, 0) + 1
                        counts[p.name] = counts.get(p.name, 0) + len(found)
                        if was_cached:
                            cached[p.name] = cached.get(p.name, 0) + len(found)
                        for c in found:
                            raw.append((qi, si, c))
                ctx.progress(0.05 + 0.55 * done / total, f"Searched {done} of {total} (query × source)")
        if ctx.should_cancel():
            raise JobCancelled()

        for p in {t[4].name: t[4] for t in tasks}.values():
            n = p.name
            if n in errors and not successes.get(n):
                status = "FAILED"
            elif n in errors:
                status = "SUCCESS"  # partial: some queries worked; the error is still shown
            else:
                status = "SUCCESS"
            reports[n] = ProviderReport(n, status, counts.get(n, 0), qcount.get(n, 0), "; ".join(dict.fromkeys(errors.get(n, [])))[:300],
                                        round(started.get(n, 0.0), 2), cached.get(n, 0))

        # ---- normalise ids, then pool with what we already had (search again / more)
        raw.sort(key=lambda t: (t[0], t[1]))
        fresh_c: list[Candidate] = []
        for _qi, _si, c in raw:
            c.candidate_id, c.scene_id = id_factory(), brief.scene_id
            if c.status is CandidateStatus.READY and c.acquisition.value == "GENERATE":
                c.status = CandidateStatus.PROPOSED
            fresh_c.append(c)
        pool_all = list(existing or []) + fresh_c
        kept, removed = deduplicate(pool_all)  # exact URL/asset/provider-id duplicates first

        # ---- thumbnails (lazy-loaded in the UI; fetched here with limited concurrency) + near-duplicate detection
        ctx.progress(0.65, "Fetching previews")
        self._thumbnails(kept, ctx, brief)
        for c in kept:
            if not c.fingerprint and c.thumbnail_path:
                c.fingerprint = fingerprint_file(self._abs(c.thumbnail_path, ctx), ctx.ffmpeg_path)
        kept, removed2 = deduplicate(kept)
        removed += removed2
        if ctx.project_root is not None:  # keep stored paths project-relative so the project can be moved
            for c in kept:
                tp = Path(c.thumbnail_path) if c.thumbnail_path else None
                if tp and tp.is_absolute() and tp.is_relative_to(ctx.project_root):
                    c.thumbnail_path = str(tp.relative_to(ctx.project_root))

        # ---- evaluate + rank
        ctx.progress(0.85, "Evaluating candidates")
        out = self.evaluate_and_rank(brief, kept, prefs, settings, history, evaluation, ranker)
        out.reports = list(reports.values())
        out.duplicates_removed = removed
        out.searched_sources = sorted({t[3].value for t in tasks})
        out.skipped_disabled = [] if ctx.expand else [TYPE_OF[k].value for k in SourceKind if not prefs.setting(k).enabled]
        out.status, out.message = decide_status(out, brief, prefs, bool(tasks), ctx.expand)
        ctx.progress(1.0, out.message)
        return out

    # ------------------------------------------------------------------ evaluation only (re-score after a settings change)
    def evaluate_and_rank(self, brief: ResearchBrief, candidates: list[Candidate], prefs: VisualPreferences, settings: ResearchSettings,
                          history: RankingHistory, evaluation: VisualEvaluationService | None = None, ranker: Ranker | None = None) -> ResearchOutcome:
        evaluation = evaluation or self.evaluation or VisualEvaluationService(weights=settings.weights)
        ranker = ranker or self.ranker or Ranker(settings)
        scores = evaluation.evaluate_all(brief, candidates, float(prefs.min_accuracy_score))
        ranked = ranker.rank([(c, scores[c.candidate_id]) for c in candidates], brief, prefs, history)
        keep_ids = {e.candidate_id for e in ranked[: settings.max_candidates]}
        cands = [c for c in candidates if c.candidate_id in keep_ids or c.status is CandidateStatus.REJECTED]
        out = ResearchOutcome(cands, {k: v for k, v in scores.items() if k in {c.candidate_id for c in cands}},
                              [e for e in ranked if e.candidate_id in keep_ids], [], ResearchStatus.NOT_STARTED, "")
        out.status, out.message = decide_status(out, brief, prefs, True, False, final=False)
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cache_key(p: SourceProvider, q: ResearchQuery, st: S, settings: ResearchSettings) -> str:
        raw = f"{p.name}|{p.config_key()}|{st.value}|{' '.join(q.text.lower().split())}|{settings.max_results_per_query}"
        return hashlib.sha1(raw.encode()).hexdigest()[:24]

    @staticmethod
    def _cache_get(cache_dir: Path, key: str, days: int) -> list[Candidate] | None:
        f = cache_dir / "search" / f"{key}.json"
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if time.time() - float(data["saved_at"]) > days * 86400:
                return None
            return [Candidate.from_dict(d) for d in data["candidates"]]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _cache_put(cache_dir: Path, key: str, found: list[Candidate]) -> None:
        f = cache_dir / "search" / f"{key}.json"
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps({"saved_at": time.time(), "candidates": [c.to_dict() for c in found]}), encoding="utf-8")
            tmp.replace(f)
        except OSError:
            _log.warning("Could not write research cache", exc_info=True)

    @staticmethod
    def _abs(path: str, ctx: RunContext) -> Path:
        p = Path(path)
        return p if p.is_absolute() or ctx.project_root is None else ctx.project_root / p

    def _thumbnails(self, cands: list[Candidate], ctx: RunContext, brief: ResearchBrief) -> None:
        todo = [c for c in cands if not c.thumbnail_path or not self._abs(c.thumbnail_path, ctx).is_file()]
        if not todo:
            return

        def one(c: Candidate) -> None:
            if ctx.should_cancel():
                return
            prov = self.registry.get(c.provider)
            if prov is None:
                return
            key = hashlib.sha1("|".join(sorted(identity_keys(c))).encode()).hexdigest()[:16]
            dest = ctx.cache_dir / "thumbs" / f"{key}.jpg"
            try:
                if dest.is_file() or prov.fetch_thumbnail(c, dest, self.http):
                    root = ctx.project_root
                    c.thumbnail_path = str(dest.relative_to(root)) if root and dest.is_relative_to(root) else str(dest)
            except Exception:  # a missing preview must not drop the candidate
                _log.debug("Thumbnail unavailable for %s", c.provider_id, exc_info=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, todo))


def decide_status(out: ResearchOutcome, brief: ResearchBrief, prefs: VisualPreferences, searched: bool, expanded: bool,
                  final: bool = True) -> tuple[ResearchStatus, str]:
    """Scene status from the outcome. Nothing weak is ever promoted to a confident result."""
    minimum = float(prefs.min_accuracy_score)
    visible = [e for e in out.ranked]
    if final:
        ok = [r for r in out.reports if r.status == "SUCCESS"]
        if not ok:
            errs = sorted((r for r in out.reports if r.error), key=lambda r: r.provider.startswith("("))
            why = "; ".join(f"{r.provider}: {r.error}" for r in errs)[:300]
            return ResearchStatus.ERROR, "Visual research unavailable." + (f" {why}" if why else "")
    if not visible:
        return ResearchStatus.LOW_CONFIDENCE, "No usable candidates were found. Try Search Again, Expand Sources or Generate AI Visual."
    best = out.scores[visible[0].candidate_id]
    if best.overall < minimum:
        return ResearchStatus.LOW_CONFIDENCE, (f"Best available candidate: {best.overall:.0f}/100 (below your minimum of {minimum:.0f}). "
                                               f"Confidence: {best.confidence.value}.")
    partial = final and any(r.status == "FAILED" or (r.status == "SUCCESS" and r.error) for r in out.reports)
    best_c = next(c for c in out.candidates if c.candidate_id == visible[0].candidate_id)
    if brief.evidence_needed and best_c.evidence_kind is not EvidenceKind.EVIDENCE:
        return ResearchStatus.NEEDS_REVIEW, "This scene needs evidence but the best visual is decorative. Please review."
    if partial:
        failed = ", ".join(r.provider for r in out.reports if r.status == "FAILED")
        return ResearchStatus.NEEDS_REVIEW, f"Candidates ready, but some sources failed ({failed or 'partial errors'}). Please review."
    return ResearchStatus.CANDIDATES_READY, f"{len(visible)} candidate(s); best {best.overall:.0f}/100, confidence {best.confidence.value}."
