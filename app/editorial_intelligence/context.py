"""EditorialContext: the read-only, per-run snapshot every analyzer works from.

It WRAPS the editing engine's ``EditingContext`` (scenes with words, sentences, intents, assignments, candidate and score, neighbours, reuse counts, per-scene hashes) and adds what the editorial
layer needs on top of it: the timeline items of each scene (media, captions, graphics, transitions, keyframes, ownership and locks), the candidate pool with its evidence kinds and fingerprints,
the explicit override map per scene, the applied reference style, the source preferences, the user's editorial preferences and the open QC findings. Build it on a detached copy of the project
(``snapshot_project``) so a worker thread can read it while the user keeps editing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.editing.context import EditingContext, SceneContext, build_context
from app.editing.effective import active_overrides, style_signature
from app.editorial_intelligence import override_resolver
from app.editorial_intelligence.editorial_models import ContentType, EditorialSettings
from app.editorial_intelligence.preference_adapter import EffectivePreferences, effective_preferences
from app.visual.preferences import VisualPreferences


@dataclass
class ClipView:
    clip_id: str
    track_id: str
    kind: str  # media | caption | text | graphic
    slot: str
    start: float
    duration: float
    asset_id: str = ""
    created_by: str = "USER"
    protected: bool = False
    locks: tuple[str, ...] = ()
    transition: dict[str, Any] | None = None
    keyframe_props: tuple[str, ...] = ()
    speed: float = 1.0
    scale_range: tuple[float, float] | None = None  # (min, max) of the scale keyframes (a zoom / punch-in), None = no scale animation
    text: dict[str, Any] = field(default_factory=dict)  # compact: style, content[:120], words, emphasis (number of marked words)
    is_audio: bool = False
    decision_id: str = ""

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class CandidateView:
    candidate_id: str
    scene_id: str
    source_type: str
    kind: str  # IMAGE | VIDEO
    title: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    evidence_kind: str = "DECORATIVE"
    fingerprint: str = ""
    asset_id: str = ""
    accuracy: float | None = None
    role: str = "OTHER"  # ASSIGNED | BEST | ALTERNATIVE | OTHER
    duration: float | None = None
    license_status: str = "UNKNOWN"
    acquirable: bool = False  # has a project asset, or a local file, or a media URL (a replacement needs one of these)


@dataclass
class EditorialScene:
    sc: SceneContext
    clips: list[ClipView]
    candidates: list[CandidateView]
    overrides: dict[str, list[str]]
    hash: str = ""
    hash_overrides: dict[str, list[str]] = field(default_factory=dict)  # the overrides that count for the hash (no "keep as is" records)

    @property
    def id(self) -> str:
        return self.sc.scene.id

    @property
    def index(self) -> int:
        return self.sc.index

    @property
    def start(self) -> float:
        return self.sc.scene.start

    @property
    def end(self) -> float:
        return self.sc.scene.end

    def visual_clips(self) -> list[ClipView]:
        return [c for c in self.clips if c.kind == "media" and not c.is_audio]

    def of_kind(self, kind: str) -> list[ClipView]:
        return [c for c in self.clips if c.kind == kind]


@dataclass
class ReferenceView:
    active: bool = False
    reference_id: str = ""
    strength: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)  # EditingStrategyOverrides.active() (already minus what the user's own settings protect)
    scores: dict[str, float] = field(default_factory=dict)  # dimension -> 0..100 of the profile
    unavailable: tuple[str, ...] = ()
    signature: str = ""


@dataclass
class EditorialContext:
    edit: EditingContext
    scenes: list[EditorialScene]
    settings: EditorialSettings
    preferences: EffectivePreferences
    reference: ReferenceView
    visual_prefs: VisualPreferences
    qc_open: list[dict[str, Any]]  # open QC findings: {"issue_id", "code", "scene_id", "item_id", "severity"}
    project_id: str
    fps: int
    total_duration: float
    content_type: str  # resolved (never AUTO)
    topic: str = ""
    script_hash: str = ""
    signature: str = ""
    project: Any = field(default=None, repr=False)  # the (detached) project the snapshot was taken from: read-only; planners / validators dry-run against its timeline
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def by_id(self, scene_id: str) -> EditorialScene:
        return self.scenes[self._index[scene_id]]

    def has_scene(self, scene_id: str) -> bool:
        return scene_id in self._index

    def window(self, scene_id: str, k: int) -> list[EditorialScene]:
        i = self._index.get(scene_id)
        return [] if i is None else self.scenes[max(0, i - k): i + k + 1]

    def prev(self, scene_id: str) -> EditorialScene | None:
        i = self._index[scene_id]
        return self.scenes[i - 1] if i > 0 else None

    def next(self, scene_id: str) -> EditorialScene | None:
        i = self._index[scene_id]
        return self.scenes[i + 1] if i + 1 < len(self.scenes) else None

    @property
    def scene_ids(self) -> list[str]:
        return [s.id for s in self.scenes]


# ------------------------------------------------------------------------------------------------------------------------ building
def _clip_view(project: Any, clip: Any) -> ClipView:
    scale = [k.value for k in clip.keyframes if k.property == "scale"]
    text: dict[str, Any] = {}
    if isinstance(clip.text, dict):
        words = clip.text.get("words") if isinstance(clip.text.get("words"), list) else []
        text = {"style": clip.text.get("style_id") or clip.text.get("style") or "", "content": str(clip.text.get("text") or clip.text.get("content") or "")[:120],
                "words": len(words), "emphasis": len(clip.text.get("emphasis") or [])}
    try:
        is_audio = project.timeline.get_track(clip.track_id).is_audio
    except Exception:  # noqa: BLE001
        is_audio = False
    return ClipView(clip.id, clip.track_id, clip.kind, clip.slot, clip.timeline_start, clip.duration, clip.asset_id, clip.created_by, bool(clip.protected), tuple(sorted(clip.locks)),
                    dict(clip.transition) if isinstance(clip.transition, dict) else None, tuple(sorted({k.property for k in clip.keyframes})), clip.speed,
                    (min(scale), max(scale)) if scale else None, text, is_audio, clip.ai_decision_id)


def _candidate_view(project: Any, cand: Any, role: str, accuracy: float | None) -> CandidateView:
    asset = project.assets.get(cand.asset_id) if cand.asset_id else None
    acquirable = bool(asset is not None or cand.local_path or cand.media_url)
    return CandidateView(cand.candidate_id, cand.scene_id, cand.source_type.value, cand.kind, cand.title, cand.description[:300], tuple(cand.tags[:12]), cand.evidence_kind.value, cand.fingerprint,
                         cand.asset_id or "", accuracy, role, cand.duration, cand.license.status, acquirable)


def _pool(project: Any, scene_id: str, assigned_candidate: str | None, limit: int = 8) -> list[CandidateView]:
    out: list[CandidateView] = []
    seen: set[str] = set()
    if assigned_candidate and assigned_candidate in project.visual_candidates:
        sc = project.candidate_scores.get(assigned_candidate)
        out.append(_candidate_view(project, project.visual_candidates[assigned_candidate], "ASSIGNED", sc.overall if sc else None))
        seen.add(assigned_candidate)
    state = project.research_status.get(scene_id)
    for r in (state.ranked if state else [])[:limit]:
        if r.candidate_id in seen or r.candidate_id not in project.visual_candidates:
            continue
        seen.add(r.candidate_id)
        out.append(_candidate_view(project, project.visual_candidates[r.candidate_id], r.role if r.role in ("BEST", "ALTERNATIVE") else "OTHER", r.accuracy))
    return out


def resolve_content_type(project: Any, settings: EditorialSettings, edit: EditingContext) -> str:
    """The user's choice, else a transparent guess from several signals (never one keyword): the visual types the scenes need, the entities and the claims."""
    if settings.content_type is not ContentType.AUTO:
        return settings.content_type.value
    from collections import Counter

    types = Counter(sc.visual_type for sc in edit.scenes)
    ents = Counter(e.type.value for sc in edit.scenes for e in sc.scene.entities)
    claims = Counter(c.type.value for sc in edit.scenes for c in sc.scene.claims)
    n = max(1, len(edit.scenes))
    finance = (ents["FINANCIAL_INSTRUMENT"] + ents["COMPANY"] + claims["NUMBER"]) / n
    legal = (ents["GOVERNMENT_AGENCY"] + ents["PERSON"] + types["EVENT"] + types["EVIDENCE"]) / n
    news = (claims["DATE"] + ents["COUNTRY"] + ents["CITY"]) / n
    process = (types["PROCESS"] + types["DATA"]) / n
    scores = {ContentType.FINANCE_EDUCATIONAL.value: finance, ContentType.DOCUMENTARY.value: legal, ContentType.NEWS.value: news, ContentType.EXPLAINER.value: process}
    best, val = max(scores.items(), key=lambda kv: kv[1])
    second = sorted(scores.values())[-2]
    return best if val >= 0.6 and val >= 1.3 * second else ContentType.GENERAL.value  # a close call is "general": a wrong category would steer the priorities


def build_editorial_context(project: Any, settings: EditorialSettings | None = None, global_prefs: Any = None) -> EditorialContext:
    """Snapshot of one project state for one analysis run. Never mutates ``project``."""
    settings = settings or project.editorial.editorial_settings
    edit = build_context(project)
    by_scene: dict[str, list[ClipView]] = {}
    for clip in project.timeline.all_clips():
        if clip.scene_id:
            by_scene.setdefault(clip.scene_id, []).append(_clip_view(project, clip))
    scenes: list[EditorialScene] = []
    for sc in edit.scenes:
        sid = sc.scene.id
        clips = sorted(by_scene.get(sid, []), key=lambda c: (c.start, c.slot, c.clip_id))
        pool = _pool(project, sid, sc.assignment.candidate_id if sc.assignment else None)
        scenes.append(EditorialScene(sc, clips, pool, override_resolver.scene_overrides(project, sid)))
        scenes[-1].hash_overrides = override_resolver.scene_overrides(project, sid, keep=False)
    prefs = effective_preferences(project, global_prefs)
    ov = active_overrides(project) if settings.use_reference_style else None
    prof = project.reference_style_profile
    ref = ReferenceView(bool(ov is not None), project.reference_settings.active_reference_id, float(project.reference_settings.style_strength), ov.active() if ov is not None else {},
                        {d: prof.score(d) for d in ("pacing", "visual_density", "motion_intensity", "caption_density", "text_density", "transition_frequency")} if (prof is not None and ov is not None) else {},
                        tuple(prof.unavailable) if prof is not None else (), style_signature(project) if ov is not None else "")
    qc_open = [{"issue_id": i.issue_id, "code": i.code, "scene_id": i.scene_id or "", "item_id": i.timeline_item_id or "", "severity": i.severity.value}
               for i in project.qc_issues if i.status.value == "OPEN" and not i.ignored_by_user]
    ctype = resolve_content_type(project, settings, edit)
    script_hash = hashlib.sha1((project.script.text if hasattr(project.script, "text") else "").encode()).hexdigest()[:10]
    ctx = EditorialContext(edit, scenes, settings, prefs, ref, project.visual_preferences, qc_open, project.project_id, project.settings.fps, edit.total_duration, ctype, edit.topic, script_hash)
    ctx.project = project
    ctx._index = {s.id: i for i, s in enumerate(scenes)}
    for es in scenes:  # a scene's hash covers everything an analyzer may read about it: narration + visuals (EditingContext hash), its timeline items, its candidate pool, its overrides
        payload = [es.sc.input_hash, [(c.clip_id, c.kind, round(c.start, 3), round(c.duration, 3), c.asset_id, c.created_by, c.locks, c.transition, c.keyframe_props, round(c.speed, 3),
                                       c.scale_range, c.text) for c in es.clips],
                   [(c.candidate_id, c.role, round(c.accuracy or 0, 1), c.source_type, c.evidence_kind, c.acquirable) for c in es.candidates], sorted((k, tuple(v)) for k, v in es.hash_overrides.items())]
        es.hash = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    ctx.signature = hashlib.sha1(json.dumps([[s.id, s.hash] for s in scenes] + [ref.signature, prefs.signature, settings.mode.value, ctype, edit.topic], default=str).encode()).hexdigest()[:16]
    return ctx
