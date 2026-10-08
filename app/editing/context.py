"""Read-only snapshot of everything the editing engine needs, safe to hand to a worker thread."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from app.analysis.lexicon import TRANSITIONS_STRONG
from app.analysis.models import Entity, Scene, SentenceAnalysis, VisualIntent
from app.editing.effective import effective_editing_settings, reference_hash_part
from app.editing.models import EditingSettings, VisualStatus
from app.media.asset import AssetType
from app.research.models import Candidate, CandidateScore, VisualAssignment
from app.transcription.models import Sentence, Word


@dataclass
class AssetInfo:
    asset_id: str
    type: AssetType
    duration: float | None
    width: int | None
    height: int | None
    source_type: str
    name: str = ""

    @property
    def is_still(self) -> bool:
        return self.type is AssetType.IMAGE


@dataclass
class NeighbourInfo:
    scene_id: str
    topic: str
    visual_type: str
    asset_id: str
    source_type: str
    terms: set[str]
    entities: set[str]
    first_word_start: float
    last_word_end: float
    claims_evidence: bool = False


@dataclass
class SceneContext:
    scene: Scene
    index: int
    words: list[Word]
    sentences: list[Sentence]
    sentence_analysis: dict[str, SentenceAnalysis]
    intent: VisualIntent | None
    visual_status: str
    assignment: VisualAssignment | None
    asset: AssetInfo | None
    extra_assets: list[AssetInfo]
    candidate: Candidate | None
    score: CandidateScore | None
    prev: NeighbourInfo | None
    next: NeighbourInfo | None
    new_entities: list[Entity]  # entities mentioned here for the first time in the video
    reuse_count: int = 0  # how many earlier scenes used the same asset
    previous_scene_with_asset: str = ""
    starts_section: bool = False
    input_hash: str = ""

    @property
    def visual_type(self) -> str:
        return self.intent.type.value if self.intent else "LITERAL"


@dataclass
class EditingContext:
    scenes: list[SceneContext]
    settings: EditingSettings
    canvas: tuple[int, int]
    fps: int
    topic: str
    voice_asset_id: str | None
    total_duration: float
    cache_dir: Path | None = None
    neighbour_transitions: set[str] = field(default_factory=set)  # scene ids whose neighbours already use a transition

    def by_id(self, scene_id: str) -> SceneContext:
        return next(c for c in self.scenes if c.scene.id == scene_id)


def _terms(text: str) -> set[str]:
    return {w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(w) > 3}


def _asset_info(project, asset_id: str | None) -> AssetInfo | None:
    a = project.assets.get(asset_id) if asset_id else None
    if a is None:
        return None
    return AssetInfo(a.id, a.type, a.duration, a.width, a.height, a.source_type.value, a.name)


def visual_status(project, scene_id: str, extra: list[str] | None = None) -> tuple[str, VisualAssignment | None]:
    a = project.visual_assignments.get(scene_id)
    if a is None:
        return (VisualStatus.APPROVED.value if extra else VisualStatus.MISSING.value), None
    if a.skipped:
        return VisualStatus.SKIPPED.value, a
    if not a.approved:
        return VisualStatus.UNAPPROVED.value, a
    asset = project.assets.get(a.asset_id) if a.asset_id else None
    if asset is None:
        return VisualStatus.MISSING.value, a  # approved but never acquired (e.g. a reference-only source)
    try:
        if not project.asset_path(asset).is_file():
            return VisualStatus.MISSING_MEDIA.value, a
    except Exception:
        return VisualStatus.MISSING_MEDIA.value, a
    return VisualStatus.APPROVED.value, a


def scene_hash(sc: SceneContext, settings: EditingSettings, canvas: tuple[int, int]) -> str:
    s = sc.scene
    payload = {
        "n": s.narration, "t": [round(s.start, 3), round(s.end, 3)], "topic": s.topic, "imp": round(s.importance, 3),
        "vt": sc.visual_type, "vs": sc.visual_status,
        "asset": [sc.asset.asset_id, sc.asset.duration, sc.asset.width, sc.asset.height] if sc.asset else None,
        "extra": [a.asset_id for a in sc.extra_assets],
        "cand": [sc.candidate.evidence_kind.value, sc.candidate.source_type.value] if sc.candidate else None,
        "num": [(n.text, n.kind.value) for n in s.numbers], "ent": [(e.text, e.type.value) for e in s.entities],
        "words": [(w.word_id, round(w.start, 3), round(w.end, 3)) for w in sc.words][:400],
        "prev": [sc.prev.topic, sc.prev.asset_id, sc.prev.source_type] if sc.prev else None,
        "settings": [settings.style, settings.pacing, settings.motion_intensity, settings.transition_frequency, settings.text_emphasis,
                     settings.number_emphasis, settings.evidence_treatment, settings.smart_transitions, settings.smart_audio_ducking,
                     settings.caption_mode, settings.provider, *reference_hash_part(settings)],
        "canvas": list(canvas), "section": sc.starts_section,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_context(project, settings: EditingSettings | None = None) -> EditingContext:
    """Snapshot the project (scenes, transcript, intents, approved visuals, assets). Never mutates the project."""
    settings = effective_editing_settings(project, settings)  # the user's settings with an applied reference style laid over them (a copy; identical without a style)
    tr = project.transcription.transcript
    scenes = list(project.scenes)
    extras = project.editing_strategy.scene_visuals
    analysis = project.scene_analysis.sentence_analysis
    seen_entities: set[str] = set()
    used_asset_scene: dict[str, str] = {}
    use_count: dict[str, int] = {}
    ctxs: list[SceneContext] = []
    infos: list[NeighbourInfo] = []
    for i, s in enumerate(scenes):
        words = tr.words_between(s.start, s.end) if tr else []
        sents = [x for x in (tr.sentences if tr else []) if x.end > s.start and x.start < s.end]
        intent = project.visual_intents.get(s.id)
        status, assignment = visual_status(project, s.id, extras.get(s.id))
        asset = _asset_info(project, assignment.asset_id) if status == VisualStatus.APPROVED.value and assignment and assignment.asset_id else None
        extra_assets = [x for x in (_asset_info(project, a) for a in extras.get(s.id, [])) if x is not None and (asset is None or x.asset_id != asset.asset_id)]
        if asset is None and extra_assets and status != VisualStatus.SKIPPED.value:
            asset, extra_assets = extra_assets[0], extra_assets[1:]
            status = VisualStatus.APPROVED.value
        cand = project.visual_candidates.get(assignment.candidate_id) if assignment and assignment.candidate_id else None
        score = project.candidate_scores.get(assignment.candidate_id) if assignment and assignment.candidate_id else None
        new_ents = []
        for e in s.entities:
            key = (e.canonical or e.text).lower()
            if key not in seen_entities:
                seen_entities.add(key)
                new_ents.append(deepcopy(e))
        reuse, prev_with = 0, ""
        if asset is not None:
            reuse = use_count.get(asset.asset_id, 0)
            prev_with = used_asset_scene.get(asset.asset_id, "")
            use_count[asset.asset_id] = reuse + 1
            used_asset_scene[asset.asset_id] = s.id
        starts_section = i > 0 and any(s.narration.lower().lstrip().startswith(p) for p in TRANSITIONS_STRONG)
        sc = SceneContext(deepcopy(s), i, deepcopy(words), deepcopy(sents),
                          {sid: analysis[sid] for sid in s.sentence_ids if sid in analysis}, deepcopy(intent), status,
                          deepcopy(assignment), asset, extra_assets, deepcopy(cand), deepcopy(score), None, None, new_ents, reuse, prev_with,
                          starts_section)
        ctxs.append(sc)
        infos.append(NeighbourInfo(
            s.id, s.topic, sc.visual_type, asset.asset_id if asset else "", asset.source_type if asset else "", _terms(s.topic + " " + s.summary),
            {(e.canonical or e.text).lower() for e in s.entities}, words[0].start if words else s.start, words[-1].end if words else s.end,
            any(c.requires_evidence for c in s.claims)))
    for i, sc in enumerate(ctxs):
        sc.prev = infos[i - 1] if i > 0 else None
        sc.next = infos[i + 1] if i + 1 < len(infos) else None
    ctx = EditingContext(ctxs, settings, (project.settings.width, project.settings.height), project.settings.fps,
                         project.scene_analysis.overall_topic or (scenes[0].topic if scenes else ""), project.voice_over.asset_id,
                         max((s.end for s in scenes), default=0.0), (project.root / "cache" / "editing") if project.root else None)
    for sc in ctxs:
        sc.input_hash = scene_hash(sc, settings, ctx.canvas)
    return ctx
