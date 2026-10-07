"""EditingStrategyService: provider-independent editing strategy.

    EditingStrategyService -> EditingStrategyProvider -> RuleBasedProvider | AIProvider (not available yet) | future providers

A provider sees the *whole* video (``VideoProfile``) plus the previous/next scene before it plans a scene, so no scene is an
isolated edit. Providers return ``ScenePlan`` objects; they never touch the project.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from statistics import median

from app.analysis.models import NumberKind
from app.core.exceptions import AppError
from app.editing.context import AssetInfo, EditingContext, SceneContext
from app.editing.models import (
    ScenePlan,
    DecisionType,
    Operation,
    PlannedSegment,
    SceneEditingBrief,
    VideoProfile,
    VisualSegment,
    VisualStatus,
    ZoomKind,
)
from app.editing.planners import (
    AudioPlanner,
    EvidencePlanner,
    MotionPlanner,
    TextPlanner,
    TransitionPlanner,
    caption_emphasis_words,
    is_evidence_visual,
)
from app.editing.presets import preset_for
from app.editing.timing import ShotTimingService, fit_source, narration_stats, speed_class
from app.logging.logger import get_logger

_log = get_logger(__name__)
INTENSE = ("crisis", "collapse", "dramatic", "shocking", "massive", "record", "surge", "crash", "urgent", "danger", "huge", "never")


class EditingProviderUnavailable(AppError):
    """The selected provider cannot run (e.g. no model configured)."""


class EditingStrategyProvider(ABC):
    name = "base"
    label = "Base"
    verified_live = False

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def analyze_video(self, ctx: EditingContext) -> VideoProfile: ...

    @abstractmethod
    def plan_scene(self, sc: SceneContext, ctx: EditingContext, profile: VideoProfile) -> ScenePlan: ...


class AIProvider(EditingStrategyProvider):
    """Placeholder for a model-backed provider. Nothing is faked: it reports itself unavailable until one is implemented."""

    name, label = "ai", "AI model"

    def is_available(self) -> tuple[bool, str]:
        return False, "No model-backed editing provider is implemented yet; the rule-based provider is used."

    def analyze_video(self, ctx):  # pragma: no cover
        raise EditingProviderUnavailable(self.is_available()[1])

    def plan_scene(self, sc, ctx, profile):  # pragma: no cover
        raise EditingProviderUnavailable(self.is_available()[1])


class RuleBasedProvider(EditingStrategyProvider):
    name, label = "rule_based", "Rule-based"

    # ------------------------------------------------------------------ video level
    def analyze_video(self, ctx: EditingContext) -> VideoProfile:
        wps, lens, imps, ev, sections = [], [], [], [], []
        for sc in ctx.scenes:
            st = narration_stats(sc.words, sc.scene.start, sc.scene.end)
            if st.words >= 3:
                wps.append(st.wps)
            lens.append(sc.scene.duration)
            imps.append(sc.scene.importance)
            if is_evidence_visual(sc, sc.asset) or any(c.requires_evidence for c in sc.scene.claims):
                ev.append(sc.scene.id)
            if sc.starts_section:
                sections.append(sc.scene.id)
        notes = []
        if not wps:
            notes.append("No word timing available: default pacing assumed.")
        return VideoProfile(ctx.topic, len(ctx.scenes), ctx.total_duration, round(median(wps), 2) if wps else 2.5,
                            round(median(lens), 2) if lens else 0.0, ev, sections, round(sum(imps) / len(imps), 3) if imps else 0.5, notes)

    # ------------------------------------------------------------------ scene analysis
    def build_brief(self, sc: SceneContext, ctx: EditingContext, profile: VideoProfile) -> SceneEditingBrief:
        s, settings = sc.scene, ctx.settings
        st = narration_stats(sc.words, s.start, s.end)
        asset = sc.asset
        evid = is_evidence_visual(sc, asset)
        if asset is None:
            complexity = 0.3
        elif evid or sc.visual_type == "DATA":
            complexity = 0.75
        elif not asset.is_still:
            complexity = 0.5
        else:
            complexity = 0.25
        nums = len(s.numbers)
        density = min(1.0, 0.18 * nums + 0.12 * len(s.claims) + 0.05 * len(s.entities) + max(0.0, st.wps - 2.5) * 0.15)
        low = s.narration.lower()
        emotional = min(1.0, 0.1 + 0.4 * s.importance + 0.15 * low.count("!") + 0.12 * sum(w in low for w in INTENSE))
        cls = speed_class(st.wps)
        has_date = any(n.kind in (NumberKind.DATE, NumberKind.DEADLINE, NumberKind.YEAR) for n in s.numbers)
        has_num = any(n.kind not in (NumberKind.DATE, NumberKind.DEADLINE, NumberKind.YEAR) for n in s.numbers)
        evidence_needed = bool(settings.evidence_treatment and asset is not None and evid and asset.is_still)
        same_prev = bool(sc.prev and asset and sc.prev.asset_id == asset.asset_id)
        keep_static = bool(asset and not evid and complexity >= 0.7) or (density > 0.8 and not evidence_needed and asset is not None and asset.is_still and sc.visual_type == "DATA")
        factors = [f"{cls.title()} narration ({st.wps:.1f} words/s)", f"Scene importance {s.importance:.0%}"]
        if nums:
            factors.append(f"{nums} figure(s) spoken")
        if evid:
            factors.append("Evidence-type visual")
        if st.longest_pause > 0.8:
            factors.append(f"Pause of {st.longest_pause:.1f}s")
        if same_prev:
            factors.append("Same visual as the previous scene")
        if sc.reuse_count:
            factors.append(f"Visual reused (x{sc.reuse_count + 1})")
        b = SceneEditingBrief(
            scene_id=s.id, importance=round(s.importance, 3), narration_speed=round(st.wps, 2), speed_class=cls, pause_seconds=round(st.pause_total, 2),
            longest_pause=round(st.longest_pause, 2), sentence_count=max(1, len(sc.sentences)), visual_complexity=round(complexity, 2),
            information_density=round(density, 2), emotional_intensity=round(emotional, 2),
            recommended_pacing={"SLOW": "RELAXED", "FAST": "BRISK"}.get(cls, "NORMAL"), keep_static=keep_static,
            should_move=not keep_static, has_number=has_num, has_date=has_date, introduces_entity=bool(sc.new_entities),
            evidence_treatment_needed=evidence_needed, text_needed=bool(nums or sc.new_entities), continue_previous=same_prev,
            change_during_sentence=bool(s.duration > 6 and nums), visual_status=sc.visual_status, factors=factors,
            recommended_evidence_treatment="ZOOM_HIGHLIGHT_RETURN" if evidence_needed else "NONE")
        return b

    # ------------------------------------------------------------------ scene plan
    def plan_scene(self, sc: SceneContext, ctx: EditingContext, profile: VideoProfile) -> ScenePlan:
        settings, preset = ctx.settings, preset_for(ctx.settings)
        brief = self.build_brief(sc, ctx, profile)
        plan = ScenePlan(sc.scene.id, brief, input_hash=sc.input_hash)
        timing = ShotTimingService(preset, settings)
        motion, evidence, text = MotionPlanner(preset, settings), EvidencePlanner(), TextPlanner(preset, settings)
        transitions, audio = TransitionPlanner(preset, settings), AudioPlanner(preset, settings)

        assets: list[AssetInfo] = ([sc.asset] if sc.asset else []) + list(sc.extra_assets)
        segments: list[VisualSegment] = []
        if assets and sc.visual_status == VisualStatus.APPROVED.value:
            spans = timing.plan_visual_times(sc, brief, assets)
            segments = self._segments(sc, ctx, assets, spans, brief)
        else:
            plan.notes.append(f"No usable visual ({sc.visual_status}): nothing was invented; text and audio instructions only.")
        first_for_asset: set[str] = set()
        for i, seg in enumerate(segments):
            asset = next(a for a in assets if a.asset_id == seg.asset_id)
            ev = evidence.plan(sc, seg, asset, brief) if i == 0 or seg.duration == max(x.duration for x in segments) else None
            if ev is not None and any(p.evidence is not None for p in plan.segments):
                ev = None
            mo = motion.plan(seg, asset, sc, brief, i, ctx.canvas, ev is not None)
            ps = PlannedSegment(seg, mo, ev, None)
            if i == 0:
                ps.transition = transitions.plan(sc)
            plan.segments.append(ps)
            first_for_asset.add(seg.asset_id)
        graphics_budget = 1.0
        plan.texts = text.plan(sc, brief, segments, graphics_budget) if (settings.text_emphasis or settings.number_emphasis) else []
        ap = ctx_audio_plan(ctx)
        plan.ducks = audio.plan(sc, brief, ap)
        plan.caption_emphasis, plan.caption_region = caption_emphasis_words(sc, plan.texts)
        # fold the outcome back into the brief (concise, structured)
        moves = [p.motion for p in plan.segments if p.motion]
        brief.should_zoom = any(m.family == "ZOOM" for m in moves)
        brief.should_pan = any(m.family == "PAN" and m.kind.startswith("PAN_") for m in moves)
        brief.recommended_motion = moves[0].kind if moves else ZoomKind.NO_ZOOM.value
        brief.should_move = bool(moves) or any(not a.is_still for a in assets)
        brief.needs_visual_change = len(segments) > 1
        brief.recommended_text = [t.graphic.content for t in plan.texts]
        first = plan.segments[0].transition if plan.segments else None
        brief.recommended_transition = first.type if first else "CUT"
        brief.overlap_next = False
        return plan

    @staticmethod
    def _snap_back(sc: SceneContext, t: float, floor: float) -> float:
        """Latest word start at or before ``t`` (but not before ``floor``); ``t`` itself when no word qualifies."""
        cands = [w.start for w in sc.words if floor - 1e-6 <= w.start <= t + 1e-6]
        return max(cands) if cands else t

    def _segments(self, sc: SceneContext, ctx: EditingContext, assets: list[AssetInfo], spans, brief: SceneEditingBrief) -> list[VisualSegment]:
        out: list[VisualSegment] = []
        cursor: dict[str, float] = {}
        prev_asset = sc.prev.asset_id if sc.prev else ""
        for i, (a, b, op, why) in enumerate(spans):
            asset = assets[i % len(assets)]
            continues = bool(i == 0 and prev_asset == asset.asset_id and brief.continue_previous)
            offset = cursor.get(asset.asset_id)
            if offset is None:
                seg = sc.assignment.segment if sc.assignment and sc.assignment.segment and i == 0 and sc.assignment.asset_id == asset.asset_id else None
                offset = float(seg.start) if seg is not None and seg.start is not None else 0.0
            fit = fit_source(asset, b - a, offset)
            if fit.operation == Operation.SHORTEN.value:  # the restart must land on a word boundary, not mid-word
                end = self._snap_back(sc, a + fit.duration, a + ctx_min_shot(ctx))
                if end > a + 0.05:
                    fit.duration = end - a
                    fit.source_out = fit.source_in + fit.duration * fit.speed
            cursor[asset.asset_id] = fit.source_out
            fitting = "contain" if (asset.is_still and (is_evidence_visual(sc, asset) or sc.visual_type in ("EVIDENCE", "DATA"))) else "cover"
            reason = why + (f"; {fit.note}" if fit.note else "") + ("; continues the previous scene's visual" if continues else "")
            conf = 92.0
            if sc.score is not None and sc.score.overall < 85:
                conf = min(conf, float(sc.score.overall))
            if fit.operation in (Operation.EXTEND.value, Operation.SHORTEN.value):
                conf = min(conf, 76.0)
            if sc.reuse_count:
                conf = min(conf, 80.0)
            dur = fit.duration
            out.append(VisualSegment(
                visual_segment_id=f"{sc.scene.id}_v{i}", scene_id=sc.scene.id, asset_id=asset.asset_id, start=round(a, 4), duration=round(dur, 4),
                reason=reason, source_in=round(fit.source_in, 4), source_out=round(fit.source_out, 4), speed=round(fit.speed, 4),
                operation=fit.operation if fit.operation in (Operation.EXTEND.value, Operation.SHORTEN.value)
                else (Operation.TRIM.value if op == Operation.HOLD.value and not asset.is_still else op),
                slot=f"visual:{i}", candidate_id=sc.assignment.candidate_id or "" if sc.assignment else "", fit=fitting,
                reuse_count=sc.reuse_count, previous_scene_id=sc.previous_scene_with_asset,
                reuse_reason=("Same subject continues; the visual stays informative." if sc.reuse_count else ""), continues_previous=continues, confidence=conf))
            covered, remaining, k = dur, (b - a) - dur, 0
            while remaining > 0.05 and k < 6:  # short source: restart the same media until the narration is covered
                k += 1
                fit2 = fit_source(asset, remaining, 0.0)
                if fit2.operation == Operation.SHORTEN.value:
                    end2 = self._snap_back(sc, a + covered + fit2.duration, a + covered + ctx_min_shot(ctx))
                    if end2 > a + covered + 0.05:
                        fit2.duration = end2 - (a + covered)
                        fit2.source_out = fit2.source_in + fit2.duration * fit2.speed
                out.append(VisualSegment(
                    visual_segment_id=f"{sc.scene.id}_v{i}{chr(97 + k)}", scene_id=sc.scene.id, asset_id=asset.asset_id, start=round(a + covered, 4),
                    duration=round(fit2.duration, 4), reason="Source media is shorter than the narration: restarts to cover the rest of the scene",
                    source_in=round(fit2.source_in, 4), source_out=round(fit2.source_out, 4), speed=round(fit2.speed, 4), operation=Operation.HOLD.value,
                    slot=f"visual:{i}{chr(97 + k)}", fit=fitting, reuse_count=sc.reuse_count + 1, confidence=70.0))
                covered += fit2.duration
                remaining -= fit2.duration
        return out


def ctx_min_shot(ctx: EditingContext) -> float:
    return preset_for(ctx.settings).min_shot


def ctx_audio_plan(ctx: EditingContext):
    return AudioPlanner(preset_for(ctx.settings), ctx.settings).global_plan()


# ============================================================================== the service
class EditingStrategyService:
    """Chooses the provider, runs scene planning with a disk cache, and applies video-wide limits (transition budget)."""

    def __init__(self) -> None:
        self.providers: dict[str, EditingStrategyProvider] = {}
        for p in (RuleBasedProvider(), AIProvider()):
            self.register(p)

    def register(self, provider: EditingStrategyProvider) -> None:
        self.providers[provider.name] = provider

    def provider_report(self) -> list[dict]:
        out = []
        for p in self.providers.values():
            ok, why = p.is_available()
            out.append({"name": p.name, "label": p.label, "available": ok, "reason": why})
        return out

    def resolve(self, name: str) -> tuple[EditingStrategyProvider, str]:
        """(provider, note). An unavailable provider falls back to rule-based and says so."""
        p = self.providers.get(name) or self.providers["rule_based"]
        ok, why = p.is_available()
        if ok:
            return p, ""
        return self.providers["rule_based"], f"{p.label} provider unavailable ({why}); used the rule-based provider."

    def plan_scene(self, provider: EditingStrategyProvider, sc: SceneContext, ctx: EditingContext, profile: VideoProfile) -> ScenePlan:
        cached = self._load(ctx, sc, provider)
        if cached is not None:
            return cached
        plan = provider.plan_scene(sc, ctx, profile)
        plan.input_hash = sc.input_hash
        self._store(ctx, sc, provider, plan)
        return plan

    # -- disk cache (reusable analysis: unchanged scenes are never re-analysed)
    @staticmethod
    def _path(ctx: EditingContext, sc: SceneContext, provider: EditingStrategyProvider):
        return ctx.cache_dir / "plans" / f"{provider.name}_{sc.input_hash}.json" if ctx.cache_dir else None

    def _load(self, ctx, sc, provider) -> ScenePlan | None:
        path = self._path(ctx, sc, provider)
        try:
            if path and path.is_file():
                return ScenePlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            _log.warning("Ignoring unreadable editing plan cache %s", path)
        return None

    def _store(self, ctx, sc, provider, plan: ScenePlan) -> None:
        path = self._path(ctx, sc, provider)
        try:
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
        except OSError:
            _log.debug("Could not write editing plan cache", exc_info=True)

    # -- video-wide limits that need more than one scene
    @staticmethod
    def limit_transitions(plans: list[ScenePlan], ctx: EditingContext) -> None:
        """No transition between every pair of scenes: keep the strongest ones within the preset's budget and never back to back."""
        preset = preset_for(ctx.settings)
        total_scenes = max(1, len(ctx.scenes))
        budget = max(1, int(round(preset.max_transition_ratio * total_scenes)))
        order = {sc.scene.id: i for i, sc in enumerate(ctx.scenes)}
        have = [p for p in plans if p.segments and p.segments[0].transition and p.segments[0].transition.type != "CUT"]
        have.sort(key=lambda p: -p.segments[0].transition.confidence)  # type: ignore[union-attr]
        keep: list[ScenePlan] = []
        for p in have:
            idx = order.get(p.scene_id, -1)
            neighbours = {order.get(q.scene_id, -9) for q in keep}
            if len(keep) >= budget or idx - 1 in neighbours or idx + 1 in neighbours:
                continue
            if (idx - 1 >= 0 and ctx.scenes[idx - 1].scene.id in ctx.neighbour_transitions) or (idx + 1 < len(ctx.scenes) and ctx.scenes[idx + 1].scene.id in ctx.neighbour_transitions):
                continue
            keep.append(p)
        for p in have:
            if p not in keep:
                t = p.segments[0].transition
                p.segments[0].transition = type(t)("CUT", 0.0, "Transition budget: hard cut keeps transitions meaningful.", 85.0)  # type: ignore[arg-type]
                p.brief.recommended_transition = "CUT"


_ = (DecisionType,)
