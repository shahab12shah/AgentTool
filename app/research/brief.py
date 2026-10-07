"""Builds a structured research brief from a scene plus its neighbours (context memory)."""

from __future__ import annotations

from app.analysis.models import ClaimType, EntityType, NumberKind, Scene, SceneStatus, VisualType
from app.core.exceptions import AnalysisError
from app.core.textutil import STOPWORDS, content_terms, normalize_token, tokenize
from app.media.asset import SourceType as S
from app.project.project import Project
from app.research.concepts import domains_of, traps_for
from app.research.models import EvidenceLevel, ResearchBrief, SceneContext

CONTINUATION_LEADS = {"that", "this", "it", "they", "these", "those", "and", "also", "so", "because", "which", "he", "she", "its", "their", "such"}
MAJOR_ENTITIES = {EntityType.GOVERNMENT_AGENCY, EntityType.COMPANY, EntityType.ORGANIZATION, EntityType.PERSON, EntityType.COUNTRY,
                  EntityType.CITY, EntityType.TECHNOLOGY, EntityType.FINANCIAL_INSTRUMENT, EntityType.PRODUCT}

# Source types worth trying for each visual requirement, best first (the user's preferences filter this later).
SOURCES_BY_TYPE: dict[str, list[S]] = {
    "EVIDENCE": [S.SCREENSHOT, S.WEB_IMAGE, S.WEB_VIDEO, S.YOUTUBE],
    "DATA": [S.SCREENSHOT, S.WEB_IMAGE, S.STOCK_VIDEO],
    "PERSON": [S.WEB_IMAGE, S.YOUTUBE, S.WEB_VIDEO, S.STOCK_IMAGE],
    "LOCATION": [S.STOCK_VIDEO, S.STOCK_IMAGE, S.WEB_IMAGE, S.YOUTUBE],
    "PROCESS": [S.STOCK_VIDEO, S.YOUTUBE, S.WEB_VIDEO, S.WEB_IMAGE, S.AI_GENERATED],
    "EVENT": [S.YOUTUBE, S.WEB_VIDEO, S.WEB_IMAGE, S.STOCK_VIDEO],
    "OBJECT": [S.STOCK_IMAGE, S.STOCK_VIDEO, S.WEB_IMAGE, S.AI_GENERATED],
    "LITERAL": [S.STOCK_VIDEO, S.STOCK_IMAGE, S.YOUTUBE, S.WEB_IMAGE, S.AI_GENERATED],
    "COMPARISON": [S.WEB_IMAGE, S.STOCK_IMAGE, S.SCREENSHOT, S.AI_GENERATED],
    "ABSTRACT": [S.STOCK_VIDEO, S.AI_GENERATED, S.STOCK_IMAGE],
}


def _neighbour(scene: Scene | None, project: Project) -> SceneContext | None:
    if scene is None:
        return None
    intent = project.visual_intents.get(scene.id)
    return SceneContext(scene.id, scene.label, scene.topic, intent.primary_subject if intent else "", scene.narration,
                        intent.type.value if intent else "")


GENERIC = {"some", "additional", "new", "means", "mean", "receive", "several", "many", "other", "another", "may", "might", "also", "more", "most",
           "very", "much", "make", "makes", "made", "get", "gets", "take", "takes", "going", "just", "like", "year", "years", "today", "now"}


def salient_terms(narration: str, entity_texts: list[str], limit: int = 8) -> list[str]:
    """The scene's own specific words, most informative first (long, noun-like, not generic filler)."""
    from app.analysis.analyzer import RuleBasedAnalyzer

    ent_words = {w.lower() for e in entity_texts for w in e.split()}
    seen: dict[str, float] = {}
    for t in tokenize(narration):
        w = t.norm
        if w in STOPWORDS or w in GENERIC or w.isdigit() or not RuleBasedAnalyzer._nounish(w):
            continue
        seen.setdefault(w, len(w) + (3 if w in ent_words else 0))
    return [w for w, _ in sorted(seen.items(), key=lambda kv: -kv[1])][:limit]


def _uniq(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


class BriefBuilder:
    def build(self, project: Project, scene_id: str) -> ResearchBrief:
        scenes = project.scenes
        idx = next((i for i, s in enumerate(scenes) if s.id == scene_id), -1)
        if idx < 0:
            raise AnalysisError("That scene no longer exists.")
        scene = scenes[idx]
        intent = project.visual_intents.get(scene.id)
        if intent is None or scene.status in (SceneStatus.PENDING, SceneStatus.FAILED):
            raise AnalysisError(f"Scene {scene.label} has not been analysed yet. Run scene analysis first.")
        prev = scenes[idx - 1] if idx > 0 else None
        nxt = scenes[idx + 1] if idx + 1 < len(scenes) else None
        video_topic = project.scene_analysis.overall_topic or ""

        first = [normalize_token(t.text) for t in tokenize(scene.narration)[:3]]
        key_terms = _uniq([t for t in content_terms(scene.narration)])  # stems; replaced below with surface words
        surface = salient_terms(scene.narration, [e.text for e in scene.entities])
        anaphoric = bool(intent.inherited_from) or (bool(first) and first[0] in CONTINUATION_LEADS) or len(key_terms) < 3

        # context terms: what the neighbouring scenes and the whole video are about (agency/company/subject first)
        ctx: list[str] = []
        if prev is not None:
            pintent = project.visual_intents.get(prev.id)
            ctx += [e.text for e in prev.entities if e.type in MAJOR_ENTITIES]
            if pintent and pintent.primary_subject:
                ctx.append(pintent.primary_subject)
            if prev.topic:
                ctx.append(prev.topic)
        if video_topic:
            ctx.append(video_topic)
        own_names = {e.text.lower() for e in scene.entities}
        context_terms = [c for c in _uniq(ctx) if c.lower() not in own_names or anaphoric][:5]

        claims, claim_types = [c.text for c in scene.claims], [c.type.value for c in scene.claims]
        types = set(claim_types)
        if intent.type is VisualType.EVIDENCE or types & {ClaimType.LAW.value, ClaimType.RULE.value, ClaimType.QUOTE.value}:
            level = EvidenceLevel.REQUIRED
        elif any(c.requires_evidence for c in scene.claims) or intent.type is VisualType.DATA:
            level = EvidenceLevel.POSSIBLE
        else:
            level = EvidenceLevel.NONE

        order = SOURCES_BY_TYPE.get(intent.type.value, SOURCES_BY_TYPE["LITERAL"])
        sources = list(order)
        if level is EvidenceLevel.REQUIRED:  # evidence first, whatever the type
            sources = [S.SCREENSHOT, S.WEB_IMAGE] + [x for x in sources if x not in (S.SCREENSHOT, S.WEB_IMAGE)]
        elif level is EvidenceLevel.POSSIBLE and S.SCREENSHOT not in sources:
            sources.insert(min(2, len(sources)), S.SCREENSHOT)
        if intent.secondary_types:
            for t in intent.secondary_types:
                for x in SOURCES_BY_TYPE.get(t.value, []):
                    if x not in sources:
                        sources.append(x)

        avoid = list(intent.avoid)
        if intent.secondary_subject and intent.primary_subject and intent.secondary_subject.lower() != intent.primary_subject.lower():
            avoid.append(f"generic {intent.secondary_subject} visuals unrelated to {intent.primary_subject}")
        subj_words = [w for w in (intent.primary_subject + " " + scene.topic).split()]
        other_words = [w for w in intent.secondary_subject.split()] + [e.text for e in scene.entities]
        avoid_terms = sorted(traps_for(subj_words, other_words))

        return ResearchBrief(
            scene_id=scene.id, topic=scene.topic, primary_subject=intent.primary_subject, secondary_subject=intent.secondary_subject,
            action=intent.action, context=intent.context or video_topic, visual_type=intent.type.value,
            secondary_types=[t.value for t in intent.secondary_types], narration=scene.narration, summary=scene.summary,
            entities=_uniq([e.text for e in scene.entities]), entity_types={e.text: e.type.value for e in scene.entities},
            claims=claims, claim_types=claim_types,
            numbers=[n.text for n in scene.numbers if n.kind not in (NumberKind.DATE, NumberKind.YEAR, NumberKind.DEADLINE)],
            dates=[n.text for n in scene.numbers if n.kind in (NumberKind.DATE, NumberKind.YEAR, NumberKind.DEADLINE)],
            evidence_level=level, preferred_sources=[s.value for s in sources], avoid=_uniq(avoid), avoid_terms=avoid_terms,
            previous=_neighbour(prev, project), next=_neighbour(nxt, project), video_topic=video_topic,
            key_terms=surface, context_terms=context_terms, anaphoric=anaphoric,
            scene_start=scene.start, scene_end=scene.end, importance=scene.importance,
            project_width=project.settings.width, project_height=project.settings.height,
        )
