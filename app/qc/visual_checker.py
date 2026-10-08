"""Visual accuracy recheck (spec 10): does the picture that is ON THE TIMELINE still fit what is being said?

The Phase 3 research score was computed when the visual was chosen. Since then the narration may have changed, the user may have swapped the picture, or the
neighbouring scenes may make it look out of place. So the visual that is actually placed is scored again, in context, with the SAME scorer (``MetadataEvaluator``, the
seven Phase 3 components with the project's weights): against the scene's research brief, and against the neighbours' briefs to see whether the picture belongs next door
instead. Context may move the score by at most ``visual.context_bonus_cap`` points. Both numbers are stored on the issue (``original_research_score`` is the stored Phase 3
score, never rewritten; ``current_qc_score`` is the recheck).

HONEST SCOPE: the scorer reads titles, descriptions, tags, source and size - never pixels. Every finding says "based on metadata", carries a confidence derived from how
many signals agree (lower for a bare file with only a name), and is worded as a judgement ("may"). Facts (a missing file, no visual at all) belong to the asset and scene
checkers. A visual the user chose can still be flagged, but one severity step lower, and nothing is ever replaced automatically.

Evidence is technical review, not fact-checking: a claim that needs evidence but is shown with a decorative picture gets "Evidence visual may not directly support the
spoken claim." - QC never says the statement is false and never invents evidence.
"""

from __future__ import annotations

from dataclasses import replace

from app.core.exceptions import AnalysisError
from app.core.textutil import STOPWORDS, stem, tokenize
from app.editing.planners import EVIDENCE_SOURCES
from app.media.asset import Asset, AssetType, SourceType
from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.severity import Severity
from app.research.brief import BriefBuilder
from app.research.evaluation import MetadataEvaluator
from app.research.models import Candidate, CandidateScore, Confidence, EvidenceKind, EvidenceLevel, ResearchBrief
from app.research.models import Acquisition as Acq
from app.timeline.clip import Clip
from app.timeline.track import Track

CONFIDENCE = {Confidence.LOW: 50.0, Confidence.MEDIUM: 70.0, Confidence.HIGH: 85.0}
BARE_ASSET_CAP = 60.0  # a file with only a name to go on is a weaker basis than a researched candidate
GENERIC_SOURCES = (SourceType.STOCK_IMAGE, SourceType.STOCK_VIDEO, SourceType.AI_GENERATED)
SPECIFIC_ENTITY_TYPES = ("PERSON", "COMPANY", "ORGANIZATION", "COUNTRY", "CITY", "PRODUCT", "GOVERNMENT_AGENCY", "FINANCIAL_INSTRUMENT")
EVIDENCE_TEXT = "Evidence visual may not directly support the spoken claim."
STEP_DOWN = {Severity.ERROR: Severity.WARNING, Severity.WARNING: Severity.NOTICE, Severity.NOTICE: Severity.INFO, Severity.INFO: Severity.INFO, Severity.CRITICAL: Severity.ERROR}
NEIGHBOUR_GAP = 20.0  # the visual must fit a neighbour this many points better than its own scene before it counts as misplaced


class VisualChecker(BaseChecker):
    id = "visual"
    label = "Visual accuracy"
    categories = (QCCategory.VISUAL_ACCURACY, QCCategory.FACT_REVIEW)
    domains = ("scenes", "visual", "timeline", "assets")
    settings_sections = ("visual", "coverage")
    scene_local = True
    expensive = True
    version = "1"

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = ctx.settings.visual
        prefs = ctx.project.visual_preferences
        evaluator = MetadataEvaluator(cfg.weights)
        builder = BriefBuilder()
        scenes = ctx.target_scenes()
        per_scene: dict[str, dict] = {}
        skipped = 0
        for n, scene in enumerate(scenes):
            ctx.check_cancel()
            report(n / max(1, len(scenes)), f"Rechecking the visual of scene {scene.label}")
            found = self._main_clip(ctx, scene)
            if found is None:
                continue
            track, clip = found
            asset = ctx.asset(clip.asset_id)
            if asset is None or asset.type is AssetType.AUDIO:
                continue
            try:
                brief = builder.build(ctx.project, scene.id)
            except AnalysisError:
                skipped += 1  # the scene has not been analysed far enough to be judged
                continue
            assign = ctx.project.visual_assignments.get(scene.id)
            cand, bare = self._candidate(ctx, scene.id, asset, assign)
            score = evaluator.evaluate(brief, cand, float(prefs.min_accuracy_score))
            bonus, why_ctx = self._context(ctx, builder, evaluator, scene, cand, score, cfg, prefs.min_accuracy_score)
            current = max(0.0, min(100.0, score.overall + bonus))
            original = self._original(ctx, assign, cand)
            per_scene[scene.id] = {"original": original, "current": round(current, 1), "context": round(bonus, 1)}
            conf = CONFIDENCE.get(score.confidence, 60.0)
            if bare:
                conf = min(conf, BARE_ASSET_CAP)
            user_chosen = bool(assign and (assign.selected_by == "USER" or assign.approved))
            if conf >= cfg.min_confidence:
                iss = self._accuracy(ctx, scene, clip, track, asset, cand, brief, score, current, original, conf, why_ctx, user_chosen, bare)
                if iss is not None:
                    out.issues.append(iss)
            ev = self._evidence(ctx, scene, clip, track, asset, cand, brief, current, original, conf, user_chosen)
            if ev is not None:
                out.issues.append(ev)
        cur = [v["current"] for v in per_scene.values()]
        org = [v["original"] for v in per_scene.values() if v["original"] is not None]
        out.metrics = {"scenes_checked": len(per_scene), "mean_current_score": round(sum(cur) / len(cur), 1) if cur else None,
                       "mean_original_score": round(sum(org) / len(org), 1) if org else None, "per_scene": per_scene}
        if skipped:
            out.notes.append(f"{skipped} scene(s) not analysed far enough to judge their visual")
        out.notes.append("scores are based on metadata (titles, descriptions, tags), not on the pixels")
        report(1.0, "Visual accuracy check complete")
        return out

    # ------------------------------------------------------------------ what is on screen for the scene
    @staticmethod
    def _main_clip(ctx: QCContext, scene) -> tuple[Track, Clip] | None:
        best, best_len = None, 0.0
        for t, c in ctx.visual_clips():
            ov = min(c.timeline_end, scene.end) - max(c.timeline_start, scene.start)
            if ov > best_len + 1e-9:
                best, best_len = (t, c), ov
        return best if best_len >= min(0.5, scene.duration * 0.25) else None

    def _candidate(self, ctx: QCContext, scene_id: str, asset: Asset, assign) -> tuple[Candidate, bool]:
        """The researched Candidate behind the placed asset when there is one; else a throwaway built from what the file itself says (never stored)."""
        stored = None
        for cid in ([assign.candidate_id] if assign and assign.candidate_id else []) + [c.candidate_id for c in ctx.project.visual_candidates.values() if c.asset_id == asset.id]:
            c = ctx.project.visual_candidates.get(cid)
            if c is not None and (c.asset_id in (None, asset.id)):
                stored = c
                break
        if stored is not None:
            return stored, False
        title = asset.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        extra = asset.extra or {}
        tags = [str(t) for t in (extra.get("tags") or [])]
        cand = Candidate(
            f"qc_{asset.id}", scene_id, asset.source_type, "IMAGE" if asset.type is AssetType.IMAGE else "VIDEO", str(extra.get("title") or title), str(extra.get("description") or ""), tags, asset.duration,
            asset.width, asset.height, provider="qc", provider_id=asset.id, local_path=str(ctx.asset_path(asset)), acquisition=Acq.LOCAL,
            evidence_kind=EvidenceKind.EVIDENCE if asset.source_type.value in EVIDENCE_SOURCES else EvidenceKind.DECORATIVE, asset_id=asset.id)
        return cand, True

    @staticmethod
    def _original(ctx: QCContext, assign, cand: Candidate) -> float | None:
        """The stored Phase 3 score, as it was (the project is never changed)."""
        if assign is not None and assign.accuracy_score is not None:
            return round(float(assign.accuracy_score), 1)
        s = ctx.project.candidate_scores.get(cand.candidate_id)
        return round(float(s.overall), 1) if s is not None else None

    # ------------------------------------------------------------------ context: neighbours may move the score, within a cap
    def _context(self, ctx: QCContext, builder: BriefBuilder, evaluator: MetadataEvaluator, scene, cand: Candidate, score: CandidateScore, cfg, min_acc) -> tuple[float, str]:
        cap = cfg.context_bonus_cap
        scenes = ctx.scenes
        i = scenes.index(scene)
        near = []
        for nb in (scenes[i - 1] if i > 0 else None, scenes[i + 1] if i + 1 < len(scenes) else None):
            if nb is None:
                continue
            try:
                b = builder.build(ctx.project, nb.id)
            except AnalysisError:
                continue
            near.append((nb, b, evaluator.evaluate(replace(b, scene_id=nb.id), replace(cand, scene_id=nb.id), float(min_acc)).overall))
        if not near:
            return 0.0, ""
        best_nb, _b, best = max(near, key=lambda n: n[2])
        gap = best - score.overall
        if gap >= NEIGHBOUR_GAP:  # fits next door much better: probably in the wrong place
            return -cap * min(1.0, (gap - NEIGHBOUR_GAP) / 30.0 + 0.3), f"the visual fits scene {best_nb.label} better ({best:.0f}) than its own scene ({score.overall:.0f})"
        if score.overall < 85 and best >= 55:  # neighbours about the same subject make a generic visual more acceptable
            return cap * min(1.0, (best - 55) / 30.0), f"neighbouring scene {best_nb.label} is about the same subject (fit {best:.0f})"
        return 0.0, ""

    # ------------------------------------------------------------------ findings
    def _accuracy(self, ctx: QCContext, scene, clip: Clip, track: Track, asset: Asset, cand: Candidate, brief: ResearchBrief, score: CandidateScore, current: float, original: float | None,
                  conf: float, why_ctx: str, user_chosen: bool, bare: bool) -> QCIssue | None:
        cfg = ctx.settings.visual
        if current >= cfg.notice_below:
            return self._generic(ctx, scene, clip, track, asset, cand, brief, score, current, original, conf, user_chosen, bare)
        sev = Severity.ERROR if current < cfg.error_below else Severity.WARNING if current < cfg.warning_below else Severity.NOTICE
        comp = score.components
        if comp.subject < 40 and brief.primary_subject:
            code, title = "visual.subject_mismatch", "The visual may show a different subject"
            reason = f"the narration is about “{brief.primary_subject}”, but the visual is titled “{cand.title or asset.name}”"
        elif comp.semantic < 50:
            code, title = "visual.mismatch", "The visual may not match the narration"
            reason = f"the narration introduces “{brief.topic or brief.primary_subject}”, but the visual is titled “{cand.title or asset.name}”"
        else:
            code, title = "visual.weak", "Weak visual match"
            reason = "several signals (subject, context, action) agree only partly with the narration"
        if user_chosen:
            sev = STEP_DOWN[sev]
        negatives = [f.lstrip("✗ ").strip() for f in score.factors if f.startswith(("✗", "⚠"))][:3]
        desc = (f"Rechecked in context: {current:.0f}/100" + (f" (research score when chosen: {original:.0f})" if original is not None else "") + f". Reason: {reason}."
                + (f" Context: {why_ctx}." if why_ctx else "") + (" " + "; ".join(negatives) + "." if negatives else "") + " Based on titles, tags and metadata; the picture itself was not inspected."
                + (" You selected this visual, so it is only flagged for your review." if user_chosen else ""))
        return self._make(ctx, scene, clip, track, code, sev, title, desc, current, original, conf if not bare else min(conf, BARE_ASSET_CAP),
                          fx.navigate("visual.replace" if code != "visual.weak" else "visual.search_again", "Choose another visual for this scene", scene_id=scene.id),
                          why="Pictures that do not match the words confuse viewers or undermine trust.", cat=QCCategory.VISUAL_ACCURACY, user_chosen=user_chosen,
                          extra={"components": {k: round(v, 1) for k, v in vars(comp).items()}, "reason": reason, "basis": getattr(score, "basis", "METADATA")})

    def _generic(self, ctx: QCContext, scene, clip: Clip, track: Track, asset: Asset, cand: Candidate, brief: ResearchBrief, score: CandidateScore, current: float, original: float | None,
                 conf: float, user_chosen: bool, bare: bool) -> QCIssue | None:
        """A good-enough score can still hide a stock picture used for a specific named subject or figure."""
        specific = [e for e in brief.entities if brief.entity_types.get(e) in SPECIFIC_ENTITY_TYPES]
        if not specific and not brief.numbers:
            return None
        if asset.source_type not in GENERIC_SOURCES and cand.source_type not in GENERIC_SOURCES:
            return None
        text = {stem(t.norm) for t in tokenize(cand.text) if t.norm not in STOPWORDS}
        mentioned = [e for e in specific if {stem(t.norm) for t in tokenize(e) if t.norm not in STOPWORDS} & text]
        if mentioned or (not specific and current >= 80):
            return None
        sev = Severity.WARNING if scene.importance >= ctx.settings.coverage.important_scene else Severity.NOTICE
        if user_chosen:
            sev = STEP_DOWN[sev]
        what = ", ".join(f"“{e}”" for e in specific[:3]) or "the figures mentioned"
        desc = (f"The narration names {what}, but the placed visual is a generic {asset.source_type.value.replace('_', ' ').lower()} titled “{cand.title or asset.name}” that does not mention it. "
                f"Rechecked score {current:.0f}/100. Based on titles, tags and metadata." + (" You selected this visual." if user_chosen else ""))
        return self._make(ctx, scene, clip, track, "visual.generic_for_specific", sev, "Generic visual for a specific subject", desc, current, original, min(conf, 65.0),
                          fx.navigate("visual.search_again", "Search for a more specific visual", scene_id=scene.id), why="A generic picture under a specific claim weakens it.",
                          cat=QCCategory.VISUAL_ACCURACY, user_chosen=user_chosen, extra={"entities": specific[:5]})

    def _evidence(self, ctx: QCContext, scene, clip: Clip, track: Track, asset: Asset, cand: Candidate, brief: ResearchBrief, current: float, original: float | None, conf: float,
                  user_chosen: bool) -> QCIssue | None:
        if brief.evidence_level is EvidenceLevel.NONE or not brief.claims:
            return None
        evidence = cand.evidence_kind is EvidenceKind.EVIDENCE or asset.source_type.value in EVIDENCE_SOURCES
        if evidence:
            return None
        sev = Severity.WARNING if brief.evidence_level is EvidenceLevel.REQUIRED else Severity.NOTICE
        if user_chosen:
            sev = STEP_DOWN[sev]
        claim = brief.claims[0]
        desc = (f"{EVIDENCE_TEXT} The claim “{claim[:90]}” is the kind that viewers expect to see supported, and the visual is {('a ' + asset.source_type.value.replace('_', ' ').lower()) if asset.source_type is not SourceType.USER_MEDIA else 'a decorative picture'} "
                f"titled “{cand.title or asset.name}”. This is a technical review of the picture, not a judgement of whether the statement is true." + (" You selected this visual." if user_chosen else ""))
        return self._make(ctx, scene, clip, track, "visual.evidence_decorative", sev, EVIDENCE_TEXT, desc, current, original, min(conf, 70.0),
                          fx.navigate("visual.search_again", "Look for a supporting source or document", scene_id=scene.id), why="Claims that need evidence are more convincing with a real source on screen.",
                          cat=QCCategory.FACT_REVIEW, user_chosen=user_chosen, extra={"evidence_level": brief.evidence_level.value, "claim": claim[:120]})

    def _make(self, ctx: QCContext, scene, clip: Clip, track: Track, code: str, sev: Severity, title: str, desc: str, current: float, original: float | None, conf: float, fix, *, why: str,
              cat: QCCategory, user_chosen: bool, extra: dict) -> QCIssue:
        iss = self.issue(
            code, cat, sev, title, description=desc, scene_id=scene.id, start=clip.timeline_start, end=clip.timeline_end, why=why, current=f"fit {current:.0f}/100", recommended=f"at least {ctx.settings.visual.notice_below:.0f}/100",
            suggested_fix="Review the scene and choose or search for a better visual." if not user_chosen else "Your choice stands; review it if you want a second opinion.", fix=fix, confidence=conf,
            viewer_impact=0.7 if sev in (Severity.ERROR, Severity.WARNING) else 0.3, signature=sha(scene.id, clip.asset_id, round(current / 5)), metrics={"signature": sha(scene.id, clip.asset_id, round(current / 5)), **extra}, ctx=ctx)
        iss.original_research_score, iss.current_qc_score = original, round(current, 1)
        iss.timeline_item_id, iss.track_id = clip.id, track.id
        iss.fingerprint = iss.make_fingerprint(iss.metrics["signature"])
        return iss
