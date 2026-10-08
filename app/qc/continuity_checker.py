"""Visual repetition (spec 11) and visual continuity (spec 12), judged from metadata, structure and keyframes - never from pixels.

REPETITION: the same footage on screen again and again. Different parts of a long stock video are different pictures, so a video counts as a repeat only when the used source
ranges overlap; a still is a repeat whenever it comes back. Reuse that is deliberate is not reported: an asset the user declared intentional, a recurring person / logo / chart /
document tied to the same entity, a callback in the narration ("as we saw", "again"), a clip the editing engine marked as a continuation or gave a reuse reason, and anything
the user placed themselves. Each asset gets a repetition score (0 none .. 100 severe) and the whole video gets one.

CONTINUITY: scene-to-scene flow. The key signal is *relevance*: how much of a visual's own description (name, tags, title) is explained by its scene's narration or by its
neighbours. A picture that explains neither - while the narration does not call for a cutaway ("imagine", "for example") - is flagged as a likely relevance problem
(``continuity.unrelated``), unless the visual accuracy checker has already flagged that scene. Also checked: abrupt subject or location changes, clashing visual styles
(cartoon / AI imagery next to real footage), a fast opening zoom after a static shot, large jumps of scale between consecutive pictures, and (when thumbnails exist) a
sudden brightness jump.

All of this is a judgement: confidence 50-90, concise decision factors, WARNING at most, and the only fixes are navigation to replace or search again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.core.textutil import content_terms
from app.media.asset import Asset, AssetType, SourceType
from app.media.thumbnails import ThumbnailService
from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.severity import Severity
from app.research.concepts import shared_domain
from app.timeline.clip import Clip
from app.timeline.keyframes import value_at
from app.timeline.track import Track

CALLBACK = re.compile(r"\b(as we saw|as we(?:'ve| have) seen|once again|again|back to|returning to|as mentioned|earlier|remember|recall|once more|as before)\b", re.I)
CUTAWAY = re.compile(r"\b(imagine|picture this|think of|for example|for instance|such as|like a|as if|metaphor|it'?s like|consider|say you|what if)\b", re.I)
TRANSITION_WORDS = re.compile(r"^\W*(meanwhile|now|next|however|but|so|on the other hand|in other news|turning to|separately|elsewhere|finally|first|second|third)\b", re.I)
CARTOON = frozenset({"cartoon", "illustration", "vector", "clipart", "animation", "animated", "3d", "render", "drawing", "sketch", "icon", "infographic"})
REAL = frozenset({"footage", "photo", "photograph", "documentary", "stock", "news", "street", "aerial", "interview"})
LOCATION_TYPES = ("CITY", "COUNTRY")
SAME_FOOTAGE_OVERLAP = 0.4  # share of the shorter use that must overlap in the source for two uses of one video to be "the same picture"
MAX_PER_CODE = 6
EXTREME_ZOOM_RATE = 0.6  # per second, in the first second of a shot following a static one
MIN_VISUAL_TERMS = 1


@dataclass
class _Use:
    asset: Asset
    clip: Clip
    track: Track
    scene_idx: int
    scene_id: str | None
    start: float
    end: float

    def source(self) -> tuple[float, float]:
        return (self.clip.source_in, self.clip.source_out)


@dataclass
class _Row:
    """One scene with its main visual and the vocabulary of both."""

    idx: int
    scene: object
    use: _Use
    sterms: set[str]
    vterms: set[str]
    entities: set[str]
    klass: str
    narration: str = ""
    intent: str = "LITERAL"
    starts_section: bool = False
    extras: dict = field(default_factory=dict)


class ContinuityChecker(BaseChecker):
    id = "continuity"
    label = "Visual continuity & repetition"
    categories = (QCCategory.CONTINUITY, QCCategory.VISUAL_REPETITION)
    domains = ("timeline", "scenes", "assets", "visual")
    settings_sections = ("repetition", "continuity", "motion")
    scene_local = False
    expensive = True
    uses_shared = True  # it stands down where the visual accuracy checker has already flagged the scene
    version = "1"

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        uses = self._uses(ctx)
        rows = self._rows(ctx, uses)
        report(0.1, "Looking for repeated pictures")
        rep_score, per_asset = self._repetition(ctx, out, uses, rows)
        report(0.5, "Checking scene-to-scene flow")
        flagged = self._flow(ctx, out, rows)
        out.metrics = {"repetition_score": rep_score, "per_asset": per_asset, "continuity_score": round(max(0.0, 100.0 - 12.0 * flagged / max(1, len(rows)) * 4), 1) if rows else 100.0,
                       "scenes_checked": len(rows)}
        report(1.0, "Continuity check complete")
        return out

    # ------------------------------------------------------------------ gathering
    def _uses(self, ctx: QCContext) -> list[_Use]:
        """Each picture the viewer sees as one use: a clip that simply continues the previous one (same footage, contiguous) is the same use."""
        out: list[_Use] = []
        scenes = ctx.scenes
        for t, c in sorted(ctx.visual_clips(), key=lambda r: (r[1].timeline_start, r[1].id)):
            a = ctx.asset(c.asset_id)
            if a is None or a.type is AssetType.AUDIO or c.duration <= 1e-3:
                continue
            prev = out[-1] if out else None
            if prev and prev.asset.id == a.id and abs(prev.end - c.timeline_start) <= ctx.frame * 1.5 and abs(prev.clip.source_out - c.source_in) <= 0.04:
                prev.end, prev.clip = c.timeline_end, c
                continue
            sid = ctx.clip_scene_id(c)
            idx = next((i for i, s in enumerate(scenes) if s.id == sid), -1)
            out.append(_Use(a, c, t, idx, sid, c.timeline_start, c.timeline_end))
        return out

    def _terms(self, asset: Asset, ctx: QCContext, scene_id: str | None) -> set[str]:
        """What the file says about itself: its name, its descriptive metadata and the researched candidate's title / tags."""
        extra = asset.extra or {}
        text = [asset.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " "), str(extra.get("title") or ""), str(extra.get("description") or ""), " ".join(str(t) for t in extra.get("tags") or [])]
        for c in ctx.project.visual_candidates.values():
            if c.asset_id == asset.id:
                text += [c.title, c.description, " ".join(c.tags)]
                break
        return {w for w in content_terms(" ".join(text)) if len(w) > 2 and not w.isdigit()}

    def _rows(self, ctx: QCContext, uses: list[_Use]) -> list[_Row]:
        rows: list[_Row] = []
        for i, s in enumerate(ctx.scenes):
            best = None
            for u in uses:
                ov = min(u.end, s.end) - max(u.start, s.start)
                if ov > 0 and (best is None or ov > best[0]):
                    best = (ov, u)
            if best is None or best[0] < min(0.5, s.duration * 0.25):
                continue
            u = best[1]
            intent = ctx.project.visual_intents.get(s.id)
            ents = {e.text.lower() for e in s.entities}
            text = " ".join([s.topic, s.narration] + ([intent.primary_subject, intent.secondary_subject] if intent else []) + [e.text for e in s.entities])
            sc = ctx.scene_ctx(s.id)
            vt = self._terms(u.asset, ctx, s.id)
            low = " ".join(vt) + " " + u.asset.name.lower()
            klass = ("ai" if u.asset.source_type is SourceType.AI_GENERATED else "cartoon" if CARTOON & set(low.split()) else "screen" if u.asset.source_type is SourceType.SCREENSHOT
                     else "photo" if u.asset.type is AssetType.IMAGE else "footage")
            rows.append(_Row(i, s, u, {w for w in content_terms(text) if len(w) > 2}, vt, ents, klass, s.narration, intent.type.value if intent else "LITERAL", bool(sc and sc.starts_section)))
        return rows

    # ------------------------------------------------------------------ repetition
    @staticmethod
    def _same_footage(a: _Use, b: _Use) -> bool:
        if a.asset.id != b.asset.id:
            return False
        if a.asset.type is AssetType.IMAGE:
            return True
        (a0, a1), (b0, b1) = a.source(), b.source()
        ov = min(a1, b1) - max(a0, b0)
        return ov >= SAME_FOOTAGE_OVERLAP * max(1e-6, min(a1 - a0, b1 - b0))

    def _intentional(self, ctx: QCContext, group: list[_Use], rows_by_scene: dict[str, _Row]) -> str:
        """Why this reuse is deliberate ('' when nothing says so)."""
        cfg = ctx.settings.repetition
        a = group[0].asset
        if a.id in cfg.intentional_assets:
            return "you marked this asset as an intentional recurring visual"
        later = group[1:]
        for u in later:
            c = u.clip
            if str(c.created_by).upper() == "USER":
                return "you placed it again yourself"
            md = c.metadata or {}
            if md.get("reuse_reason") or md.get("continues_previous"):
                return "the editing decisions give a reason for the reuse"
        names = [rows_by_scene[u.scene_id].entities if u.scene_id in rows_by_scene else set() for u in group]
        common = set.intersection(*names) if names and all(names) else set()
        kinds = {rows_by_scene[u.scene_id].intent for u in group if u.scene_id in rows_by_scene}
        if common and (kinds & {"DATA", "EVIDENCE", "PERSON"} or a.source_type is SourceType.SCREENSHOT):
            return f"it is the recurring {sorted(common)[0]} visual"
        subjects = [{w for w in rows_by_scene[u.scene_id].sterms} for u in group if u.scene_id in rows_by_scene]
        if len(subjects) >= 2 and common:
            return f"the scenes share the same subject ({sorted(common)[0]})"
        for u in later:
            r = rows_by_scene.get(u.scene_id or "")
            if r is not None and CALLBACK.search(r.narration):
                return "the narration calls back to it"
        return ""

    def _repetition(self, ctx: QCContext, out: CheckerOutput, uses: list[_Use], rows: list[_Row]) -> tuple[float, dict]:
        cfg = ctx.settings.repetition
        f = 1.5 - max(0.0, min(1.0, cfg.sensitivity))
        limit = max(2, round(cfg.max_uses * f))
        rows_by_scene = {r.scene.id: r for r in rows}
        per_asset: dict[str, dict] = {}
        by_asset: dict[str, list[_Use]] = {}
        for u in uses:
            by_asset.setdefault(u.asset.id, []).append(u)
        unjustified_repeats, flagged_assets = 0, 0
        for aid, us in by_asset.items():
            ctx.check_cancel()
            if len(us) < 2:
                continue
            us.sort(key=lambda u: u.start)
            groups: list[list[_Use]] = []
            for u in us:
                for g in groups:
                    if any(self._same_footage(u, m) for m in g):
                        g.append(u)
                        break
                else:
                    groups.append([u])
            group = max(groups, key=len)
            if len(group) < 2:
                per_asset[aid] = {"uses": len(us), "score": 0.0, "intentional": False}
                continue
            why_ok = self._intentional(ctx, group, rows_by_scene)
            adj = [(a, b) for a, b in zip(group, group[1:]) if (b.start - a.end < cfg.min_gap_seconds or (a.scene_idx >= 0 and 0 < b.scene_idx - a.scene_idx <= cfg.adjacent_scenes))]
            score = 0.0 if why_ok else min(100.0, 25.0 * (len(group) - 1) + 20.0 * len(adj))
            per_asset[aid] = {"uses": len(group), "score": round(score, 1), "intentional": bool(why_ok), "reason": why_ok}
            if why_ok:
                continue
            unjustified_repeats += len(group) - 1
            conf = min(90.0, 60.0 + 10.0 * ((len(group) > limit) + bool(adj) + (len(group) >= 2 and any(b.start - a.end < cfg.min_gap_seconds for a, b in adj))))
            where = ", ".join(f"{_t(u.start)}" for u in group[:6]) + (" ..." if len(group) > 6 else "")
            name = us[0].asset.name
            base = dict(ctx=ctx, signature=sha(aid, len(group)), group_hint="continuity")
            if len(group) > limit and flagged_assets < MAX_PER_CODE:
                flagged_assets += 1
                anchor = group[limit]
                out.issues.append(self._repeat_issue(
                    "visual.repetition", Severity.WARNING if len(group) > limit + 1 else Severity.NOTICE, "The same visual is used too often",
                    f"“{name}” appears {len(group)} times (limit {limit}): {where}." + (f" {len(adj)} of the repeats are close together." if adj else ""), anchor, conf, score, len(group), limit, **base))
            elif adj and flagged_assets < MAX_PER_CODE:
                flagged_assets += 1
                a, b = adj[0]
                out.issues.append(self._repeat_issue(
                    "visual.repetition_adjacent", Severity.WARNING if b.start - a.end < cfg.min_gap_seconds / 2 else Severity.NOTICE, "The same visual comes back almost immediately",
                    f"“{name}” is shown at {_t(a.start)} and again at {_t(b.start)}, {max(0.0, b.start - a.end):.0f} s later" + (f" ({b.scene_idx - a.scene_idx} scene(s) apart)" if a.scene_idx >= 0 else "") + ".",
                    b, conf, score, len(group), limit, **base))
            elif len(group) >= 2 and self._same_composition(group) and flagged_assets < MAX_PER_CODE:
                flagged_assets += 1
                out.issues.append(self._repeat_issue(
                    "visual.repetition_composition", Severity.NOTICE, "The same shot is framed the same way again", f"“{name}” is used {len(group)} times with identical framing and movement: {where}.", group[1],
                    min(conf, 70.0), score, len(group), limit, **base))
        gen = self._generated(ctx, out)
        total = max(4, len(uses))
        overall = round(min(100.0, 100.0 * (unjustified_repeats + gen) / total * 2.0), 1)
        return overall, per_asset

    def _repeat_issue(self, code: str, sev: Severity, title: str, desc: str, anchor: _Use, conf: float, score: float, n: int, limit: int, *, ctx: QCContext, signature: str, group_hint: str) -> QCIssue:
        iss = self.issue(
            code, QCCategory.VISUAL_REPETITION, sev, title, description=desc + f" Repetition score {score:.0f}/100. Not marked as intentional.", scene_id=anchor.scene_id, start=anchor.start, end=anchor.end,
            why="Seeing the same picture again looks like filler unless it is a deliberate callback.", current=f"{n} use(s)", recommended=f"at most {limit}, spread out", suggested_fix="Replace the repeats with other visuals, or mark the asset as intentional in the QC settings.",
            fix=fx.navigate("visual.search_again", "Search for another visual", scene_id=anchor.scene_id or ""), confidence=conf, viewer_impact=0.4 if sev is Severity.WARNING else 0.2, signature=signature,
            metrics={"signature": signature, "repetition_score": round(score, 1), "uses": n}, group_hint=group_hint, ctx=ctx)
        iss.timeline_item_id, iss.track_id = anchor.clip.id, anchor.track.id
        iss.fingerprint = iss.make_fingerprint(signature)
        return iss

    @staticmethod
    def _same_composition(group: list[_Use]) -> bool:
        def sig(u: _Use):
            c = u.clip
            return (round(c.scale, 2), round(c.position[0]), round(c.position[1]), str(c.effects.get("fit", "cover")), str(c.effects.get("focus_region")),
                    tuple((k.property, round(k.value, 2)) for k in sorted(c.keyframes, key=lambda k: (k.property, k.time))))

        return len({sig(u) for u in group}) < len(group)

    def _generated(self, ctx: QCContext, out: CheckerOutput) -> int:
        """Several AI-generated assets made from near-identical prompts / titles."""
        gen = [a for a in ctx.project.assets.all() if a.source_type is SourceType.AI_GENERATED and any(c.asset_id == a.id for _t, c in ctx.visual_clips())]
        keys = {a.id: {w for w in content_terms(str((a.extra or {}).get("prompt") or a.name.rsplit(".", 1)[0].replace("_", " "))) if len(w) > 2} for a in gen}
        seen: set[str] = set()
        clusters = []
        for a in gen:
            if a.id in seen or not keys[a.id]:
                continue
            members = [b for b in gen if b.id not in seen and keys[b.id] and len(keys[a.id] & keys[b.id]) / len(keys[a.id] | keys[b.id]) >= 0.8]
            seen.update(m.id for m in members)
            if len(members) >= 2:
                clusters.append(members)
        n = 0
        for members in clusters[:MAX_PER_CODE]:
            n += len(members) - 1
            first = next(c for _t, c in ctx.visual_clips() if c.asset_id == members[1].id)
            t = ctx.timeline.get_track(first.track_id)
            iss = self.issue(
                "visual.repetition_generated", QCCategory.VISUAL_REPETITION, Severity.WARNING if len(members) >= 4 else Severity.NOTICE, "AI-generated images that look alike",
                description=f"{len(members)} generated images share almost the same prompt or title: {', '.join(m.name for m in members[:4])}.", scene_id=ctx.clip_scene_id(first), start=first.timeline_start, end=first.timeline_end,
                why="Near-identical generated pictures read as repetition even though they are different files.", current=f"{len(members)} similar images", recommended="varied imagery",
                suggested_fix="Change the prompt for some of them to show a different subject or angle.", fix=fx.navigate("visual.search_again", "Search for another visual", scene_id=ctx.clip_scene_id(first) or ""),
                confidence=70.0, viewer_impact=0.25, signature=sha(sorted(m.id for m in members)), group_hint="continuity", ctx=ctx)
            iss.timeline_item_id, iss.track_id = first.id, t.id if t else None
            out.issues.append(iss)
        return n

    # ------------------------------------------------------------------ continuity
    @staticmethod
    def _fit(visual: set[str], text: set[str]) -> float:
        """Share of the visual's own vocabulary that the text explains (same word stem or same concept domain)."""
        if not visual:
            return 0.0
        hit = sum(1 for v in visual if v in text or any(shared_domain(v, t) for t in text))
        return hit / len(visual)

    def _flow(self, ctx: QCContext, out: CheckerOutput, rows: list[_Row]) -> int:
        cfg = ctx.settings.continuity
        flagged = 0
        shared = ctx.shared.get("visual")
        reported = {i.scene_id for i in getattr(shared, "issues", []) if i.severity in (Severity.ERROR, Severity.WARNING)}
        low = 0.2 + (cfg.sensitivity - 0.5) * 0.2  # a visual explained by less than this share of its surroundings is an outlier
        skipped_by_visual = 0
        for k, r in enumerate(rows):
            ctx.check_cancel()
            prev, nxt = (rows[k - 1] if k > 0 else None), (rows[k + 1] if k + 1 < len(rows) else None)
            if len(r.vterms) >= MIN_VISUAL_TERMS and flagged < MAX_PER_CODE * 3:
                own = self._fit(r.vterms, r.sterms)
                near = max([self._fit(r.vterms, n.sterms | n.vterms) for n in (prev, nxt) if n is not None] or [0.0])
                if own < low and near < low and (prev or nxt):
                    if CUTAWAY.search(r.narration) or r.intent in ("ABSTRACT", "COMPARISON"):
                        pass  # the narration asks for a cutaway: a loosely related picture is the point
                    elif r.scene.id in reported:
                        skipped_by_visual += 1
                    else:
                        flagged += 1
                        out.issues.append(self._unrelated(ctx, r, prev, nxt, own, near, low))
            if prev is not None and r.scene.id not in reported:
                flagged += self._pair(ctx, out, prev, r, cfg)
        if skipped_by_visual:
            out.notes.append(f"{skipped_by_visual} unrelated visual(s) already flagged by the visual accuracy checker")
        flagged += self._styles(ctx, out, rows)
        flagged += self._colour(ctx, out, rows, cfg)
        return flagged

    def _unrelated(self, ctx: QCContext, r: _Row, prev: _Row | None, nxt: _Row | None, own: float, near: float, low: float) -> QCIssue:
        neighbours_related = bool(prev and nxt and (self._fit(prev.vterms, nxt.sterms | nxt.vterms) >= low or self._fit(prev.sterms, nxt.sterms) >= low))
        signals = 1 + (len(r.sterms) >= 4) + neighbours_related + (len(r.vterms) >= 3)
        conf = min(85.0, 40.0 + 12.0 * signals)
        sev = Severity.WARNING if conf >= 65 else Severity.NOTICE
        title_terms = ", ".join(sorted(r.vterms)[:5])
        neigh = " and ".join(f"scene {n.scene.label}" for n in (prev, nxt) if n is not None)
        iss = self.issue(
            "continuity.unrelated", QCCategory.CONTINUITY, sev, "Visual may not belong here",
            description=f"The picture in scene {r.scene.label} ({r.use.asset.name}; terms: {title_terms}) shares {own:.0%} of its description with its own narration and {near:.0%} with {neigh}"
            f"{'; the neighbouring scenes relate to each other' if neighbours_related else ''}. The narration does not call for a cutaway. Based on names, tags and metadata only.",
            scene_id=r.scene.id, start=r.use.start, end=r.use.end, why="A picture unrelated to what is said and to the scenes around it breaks the flow and can mislead.", current=f"fit {own:.0%} / neighbours {near:.0%}",
            recommended="a visual that relates to the narration or its neighbours", suggested_fix="Check this scene; replace the visual unless the cutaway is intentional.",
            fix=fx.navigate("visual.replace", "Choose another visual for this scene", scene_id=r.scene.id), confidence=conf, viewer_impact=0.5, signature=sha(r.scene.id, r.use.asset.id), group_hint="continuity", ctx=ctx)
        iss.timeline_item_id, iss.track_id = r.use.clip.id, r.use.track.id
        iss.fingerprint = iss.make_fingerprint(iss.metrics.get("signature", "") or sha(r.scene.id, r.use.asset.id))
        return iss

    def _pair(self, ctx: QCContext, out: CheckerOutput, a: _Row, b: _Row, cfg) -> int:
        n = 0
        # abrupt subject change: neither the narrations nor the pictures are related, and nothing introduces the change
        sim_text = self._fit(a.sterms, b.sterms) if a.sterms else 0.0
        sim_vis = max(self._fit(b.vterms, a.vterms), self._fit(a.vterms, b.vterms))
        lead = TRANSITION_WORDS.search(b.narration) or b.starts_section or CUTAWAY.search(b.narration) or CALLBACK.search(b.narration)
        if a.sterms and b.sterms and sim_text < cfg.topic_jump_similarity * (1.5 - cfg.sensitivity) and sim_vis < 0.15 and not lead:
            n += 1
            self._add(ctx, out, "continuity.subject_jump", Severity.NOTICE, "Abrupt change of subject", f"Scene {a.scene.label} -> {b.scene.label}: the narration and the pictures share almost no vocabulary "
                      f"({sim_text:.0%} / {sim_vis:.0%}) and nothing introduces the change.", b, 62.0, "Viewers lose the thread when the subject jumps without a signpost.", "Add a bridging sentence or visual, or ignore if the jump is intended.")
        # place: the visual names a location that belongs to another scene
        loc = self._location(ctx, a, b)
        if loc:
            n += 1
            self._add(ctx, out, "continuity.location_jump", Severity.NOTICE, "Location may not match", loc, b, 58.0, "A picture of a different place than the one being discussed misleads.", "Check where the footage was filmed.")
        n += self._motion_scale(ctx, out, a, b, cfg)
        return n

    @staticmethod
    def _location(ctx: QCContext, a: _Row, b: _Row) -> str:
        places = {e.text.lower(): e for r in (a, b) for e in r.scene.entities if e.type.value in LOCATION_TYPES}
        own = {e.text.lower() for e in b.scene.entities if e.type.value in LOCATION_TYPES}
        text = " ".join(sorted(b.vterms)) + " " + b.use.asset.name.lower()
        for name in places:
            if name not in own and name in text:
                return f"The visual in scene {b.scene.label} ({b.use.asset.name}) names “{name.title()}”, but that scene's narration is about {', '.join(sorted(own)) or 'another subject'}. Based on names and tags only."
        return ""

    def _motion_scale(self, ctx: QCContext, out: CheckerOutput, a: _Row, b: _Row, cfg) -> int:
        n = 0
        ca, cb = a.use.clip, b.use.clip
        a_moves = any(k.property in ("scale", "position_x", "position_y") for k in ca.keyframes)
        kf = sorted((k for k in cb.keyframes if k.property == "scale" and k.time <= 1.0 + 1e-6), key=lambda k: k.time)
        rate = max((abs(y.value - x.value) / max(1e-6, y.time - x.time) * max(cb.scale, 1e-6) for x, y in zip(kf, kf[1:])), default=0.0)
        if not a_moves and rate > EXTREME_ZOOM_RATE:
            n += 1
            self._add(ctx, out, "continuity.extreme_motion_jump", Severity.NOTICE, "Fast zoom right after a still shot", f"Scene {b.scene.label} opens with a zoom of {rate:.2f} per second after a static shot "
                      f"(limit {EXTREME_ZOOM_RATE:.2f}).", b, 80.0, "A sudden burst of movement after stillness is jarring.", "Start the movement later or make it slower.", viewer=0.3)
        s_end = max(ca.scale, 1e-6) * value_at(ca.keyframes, "scale", ca.duration)
        s_start = max(cb.scale, 1e-6) * value_at(cb.keyframes, "scale", 0.0)
        ratio = max(s_start / s_end, s_end / s_start)
        if ratio > cfg.scale_jump:
            n += 1
            self._add(ctx, out, "continuity.scale_jump", Severity.NOTICE, "Large jump in picture scale", f"The picture ends at {s_end:.2f}x and the next one starts at {s_start:.2f}x ({ratio:.1f}x apart, limit "
                      f"{cfg.scale_jump:.1f}x).", b, 75.0, "Cutting between very different magnifications feels like a jump cut.", "Match the end scale of one shot to the start of the next, or soften the zoom.", viewer=0.25)
        return n

    def _styles(self, ctx: QCContext, out: CheckerOutput, rows: list[_Row]) -> int:
        """Cartoon or AI imagery directly beside real footage / photos."""
        clashes = [(a, b) for a, b in zip(rows, rows[1:]) if {a.klass, b.klass} & {"ai", "cartoon"} and {a.klass, b.klass} & {"footage", "photo"}]
        if not clashes:
            return 0
        a, b = clashes[0]
        many = len(clashes) >= 3
        sev = Severity.WARNING if many else Severity.NOTICE
        self._add(ctx, out, "continuity.style_clash", sev, "Visual styles clash", f"{a.klass} imagery (scene {a.scene.label}) sits next to {b.klass} imagery (scene {b.scene.label})"
                  + (f"; the same kind of switch happens {len(clashes)} times" if many else "") + ". Based on source type and tags.", b, 70.0 if many else 60.0, "Mixing illustrated or generated pictures with real footage looks inconsistent.",
                  "Keep one visual language, or separate the styles with clear sections.")
        return len(clashes)

    def _colour(self, ctx: QCContext, out: CheckerOutput, rows: list[_Row], cfg) -> int:
        if ctx.root is None:
            return 0
        lum: dict[str, float | None] = {}

        def bright(a: Asset) -> float | None:
            if a.id not in lum:
                p = ThumbnailService.thumbnail_path(ctx.root, a)  # type: ignore[arg-type]
                try:
                    with Image.open(p) as im:
                        lum[a.id] = float(np.asarray(im.convert("L").resize((16, 16)), dtype=np.float64).mean()) / 255.0
                except (OSError, ValueError):
                    lum[a.id] = None
            return lum[a.id]

        n, missing = 0, 0
        for a, b in zip(rows, rows[1:]):
            la, lb = bright(a.use.asset), bright(b.use.asset)
            if la is None or lb is None:
                missing += 1
                continue
            if abs(la - lb) > cfg.brightness_jump and n < MAX_PER_CODE:
                n += 1
                self._add(ctx, out, "continuity.color_mismatch", Severity.NOTICE, "Sudden change of brightness", f"The picture gets {'brighter' if lb > la else 'darker'} by {abs(la - lb):.0%} between scene "
                          f"{a.scene.label} and {b.scene.label} (limit {cfg.brightness_jump:.0%}). Measured on the thumbnails.", b, 65.0, "A sudden flash or blackout between scenes is tiring to watch.",
                          "Match the exposure or put a short dissolve between the two pictures.", viewer=0.2)
        if missing:
            out.notes.append(f"brightness not compared for {missing} pair(s): no thumbnails yet")
        return n

    def _add(self, ctx: QCContext, out: CheckerOutput, code: str, sev: Severity, title: str, desc: str, row: _Row, conf: float, why: str, fix_text: str, viewer: float = 0.3) -> None:
        iss = self.issue(
            code, QCCategory.CONTINUITY, sev, title, description=desc, scene_id=row.scene.id, start=row.use.start, end=row.use.end, why=why, current=title.lower(), recommended="a smooth flow", suggested_fix=fix_text,
            fix=fx.navigate("visual.search_again", "Search for another visual", scene_id=row.scene.id), confidence=conf, viewer_impact=viewer, signature=sha(code, row.scene.id, row.use.asset.id), group_hint="continuity", ctx=ctx)
        iss.timeline_item_id, iss.track_id = row.use.clip.id, row.use.track.id
        iss.fingerprint = iss.make_fingerprint(sha(code, row.scene.id, row.use.asset.id))
        out.issues.append(iss)


def _t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"

