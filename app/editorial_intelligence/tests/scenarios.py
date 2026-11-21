"""Test scenarios for the editorial analyzers: run the REAL scene pipeline on narration you write (so entities, claims, numbers, cues, topics and word timings are what the application
really produces), then attach the visuals / candidates / timeline clips you want to judge. No media is decoded: files are tiny placeholders, only metadata matters (as in production,
where candidate scoring is metadata based).

    ws = craft(ws, tmp_path, NARRATION, {0: V("house", "Exterior of the house"), 2: V("house", "Exterior of the house"), 3: V("report", "Police report", source="SCREENSHOT", evidence="EVIDENCE")})
    ctx = build_editorial_context(ws.project)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.media.asset import Asset, AssetType, SourceType
from app.research.models import Acquisition, Candidate, CandidateScore, EvidenceKind, RankedEntry, SceneResearchState, ScoreComponents, ScoreCategory, Confidence, VisualAssignment
from app.timeline.clip import Clip
from app.timeline.timeline import new_clip_id


@dataclass
class V:
    """One visual: its identity (same ``key`` = the same project asset), what the candidate metadata says about it, and who chose it."""

    key: str
    title: str
    description: str = ""
    tags: tuple[str, ...] = ()
    source: str = "STOCK_VIDEO"  # SourceType value
    kind: str = "VIDEO"  # VIDEO | IMAGE
    evidence: str = "DECORATIVE"  # EvidenceKind value: DECORATIVE | EVIDENCE
    accuracy: float = 88.0
    by: str = "AI"  # AI | USER
    fingerprint: str = ""
    duration: float = 12.0
    alts: tuple["V", ...] = ()  # alternatives the research found (candidates; the first is the best alternative)
    acquired_alts: bool = True  # alternatives already have a project asset (so a replacement is possible)
    clip_seconds: float | None = None  # timeline clip length (default: the scene's length)
    created_by: str = "AI"  # clip ownership


def _asset(ws, v: V, cache: dict[str, Asset]) -> Asset:
    if v.key in cache:
        return cache[v.key]
    p = ws.project
    aid = p.assets.new_id()
    is_image = v.kind == "IMAGE"
    rel = f"media/{'images' if is_image else 'video'}/{aid}.{'png' if is_image else 'mp4'}"
    path = p.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 64)
    a = Asset(aid, AssetType.IMAGE if is_image else AssetType.VIDEO, SourceType(v.source), rel, v.title, None if is_image else v.duration, 1920, 1080, None if is_image else 30.0, size_bytes=64,
              content_hash=f"hash_{aid}", extra={"tags": list(v.tags)})
    p.assets.add(a)
    cache[v.key] = a
    return a


def _candidate(ws, scene_id: str, v: V, cache: dict[str, Asset], *, acquired: bool, role: str) -> Candidate:
    p = ws.project
    p.research_settings.candidate_counter += 1
    cid = f"cand_{p.research_settings.candidate_counter:05d}"
    asset = _asset(ws, v, cache) if acquired else None
    c = Candidate(cid, scene_id, SourceType(v.source), v.kind, v.title, v.description, list(v.tags), None if v.kind == "IMAGE" else v.duration, 1920, 1080, "test", cid, f"https://example.test/{v.key}",
                  "" if asset else f"https://example.test/{v.key}.mp4", evidence_kind=EvidenceKind(v.evidence), acquisition=Acquisition.LOCAL if asset else Acquisition.REFERENCE_ONLY,
                  fingerprint=v.fingerprint, asset_id=asset.id if asset else None)
    p.visual_candidates[cid] = c
    p.candidate_scores[cid] = CandidateScore(cid, scene_id, v.accuracy, ScoreComponents(), ScoreCategory.GOOD if hasattr(ScoreCategory, "GOOD") else list(ScoreCategory)[0], Confidence.HIGH if hasattr(Confidence, "HIGH") else list(Confidence)[0],
                                             f"{role}: {v.title}")
    return c


def craft(ws, tmp_path, narration: str, visuals: dict[int, V], *, min_scene: float = 2.0, threshold: float = 0.6) -> Any:
    """``ws`` with a project whose scenes come from the real pipeline on ``narration`` and whose scene ``i`` shows ``visuals[i]`` (0-based). Returns ``ws``."""
    from app.tests.conftest import pipeline_ws

    pipeline_ws(ws, tmp_path, narration, threshold=threshold, min_scene=min_scene)
    p = ws.project
    track = next(t for t in p.timeline.tracks if t.kind.value == "video")
    cache: dict[str, Asset] = {}
    for i, sc in enumerate(p.scenes):
        v = visuals.get(i)
        if v is None:
            continue
        cand = _candidate(ws, sc.id, v, cache, acquired=True, role="ASSIGNED")
        alts = [_candidate(ws, sc.id, a, cache, acquired=v.acquired_alts, role="ALTERNATIVE") for a in v.alts]
        p.visual_assignments[sc.id] = VisualAssignment(sc.id, cand.candidate_id, cand.asset_id, v.by, v.accuracy, True, False, EvidenceKind(v.evidence), Acquisition.LOCAL, None, sc.duration,
                                                       SourceType(v.source))
        ranked = [RankedEntry(cand.candidate_id, v.accuracy, v.accuracy, {}, "BEST")] + [RankedEntry(c.candidate_id, a.accuracy, a.accuracy, {}, "ALTERNATIVE") for c, a in zip(alts, v.alts)]
        p.research_status[sc.id] = SceneResearchState(sc.id, best_id=cand.candidate_id, alternatives=[c.candidate_id for c in alts], ranked=ranked)
        asset = cache[v.key]
        dur = v.clip_seconds if v.clip_seconds is not None else sc.duration
        clip = Clip(new_clip_id(), track.id, asset.id, sc.start, dur, 0.0, dur, scene_id=sc.id, slot="visual:0", created_by=v.created_by)
        track.clips.append(clip)
    track.sort()
    return ws
