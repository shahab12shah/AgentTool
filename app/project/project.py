"""The Project aggregate: everything that is saved in ``project.json``."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.constants import APP_VERSION, SCHEMA_VERSION, TIME_EPSILON
from app.core.exceptions import InvalidProjectError
from app.media.asset import Asset, AssetType
from app.media.asset_registry import AssetRegistry
from app.analysis.models import Scene, SceneAnalysisState, VisualIntent
from app.core.serialization import from_plain, to_plain
from app.project.project_schema import (
    AIDecision,
    ProjectSettings,
    RenderSettings,
    ScriptData,
    VoiceOverData,
    migrate_document,
    validate_document,
)
from app.research.models import (
    Candidate,
    CandidateScore,
    ResearchQuery,
    ResearchSession,
    ResearchSettings,
    SceneResearchState,
    VisualAssignment,
)
from app.editing.models import (
    EditingDecision,
    EditingSession,
    EditingSettings,
    EditingStrategy,
    OverrideRecord,
    TimelineGeneration,
)
from app.storage.paths import ProjectPaths
from app.transcription.alignment import ScriptAlignment
from app.transcription.models import TranscriptionState
from app.visual.preferences import VisualPreferences
from app.timeline.timeline import Timeline


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Project:
    project_id: str
    project_name: str
    created_at: str
    updated_at: str
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    script: ScriptData = field(default_factory=ScriptData)
    voice_over: VoiceOverData = field(default_factory=VoiceOverData)
    scenes: list[Scene] = field(default_factory=list)
    assets: AssetRegistry = field(default_factory=AssetRegistry)
    timeline: Timeline = field(default_factory=Timeline.default)
    render_settings: RenderSettings = field(default_factory=RenderSettings)
    ai_decisions: list[AIDecision] = field(default_factory=list)
    # Phase 2
    transcription: TranscriptionState = field(default_factory=TranscriptionState)
    script_alignment: ScriptAlignment | None = None
    scene_analysis: SceneAnalysisState = field(default_factory=SceneAnalysisState)
    visual_intents: dict[str, VisualIntent] = field(default_factory=dict)
    visual_preferences: VisualPreferences = field(default_factory=VisualPreferences)
    # Phase 3: visual research
    research_settings: ResearchSettings = field(default_factory=ResearchSettings)
    research_queries: dict[str, ResearchQuery] = field(default_factory=dict)
    research_sessions: list[ResearchSession] = field(default_factory=list)
    visual_candidates: dict[str, Candidate] = field(default_factory=dict)
    candidate_scores: dict[str, CandidateScore] = field(default_factory=dict)
    visual_assignments: dict[str, VisualAssignment] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    research_status: dict[str, SceneResearchState] = field(default_factory=dict)
    # Phase 4: AI editing
    editing_settings: EditingSettings = field(default_factory=EditingSettings)
    editing_strategy: EditingStrategy = field(default_factory=EditingStrategy)
    editing_sessions: list[EditingSession] = field(default_factory=list)
    editing_decisions: dict[str, EditingDecision] = field(default_factory=dict)
    timeline_generation: TimelineGeneration = field(default_factory=TimelineGeneration)
    ai_overrides: list[OverrideRecord] = field(default_factory=list)
    timeline_version: int = 0
    schema_version: int = SCHEMA_VERSION
    application_version: str = APP_VERSION
    # Runtime-only state (never serialised):
    root: Path | None = None
    dirty: bool = False

    # ----- construction -----
    @classmethod
    def new(cls, name: str, settings: ProjectSettings | None = None) -> "Project":
        now = utc_now()
        return cls(
            project_id=f"proj_{uuid.uuid4().hex[:12]}",
            project_name=name,
            created_at=now,
            updated_at=now,
            settings=settings or ProjectSettings(),
        )

    @property
    def paths(self) -> ProjectPaths:
        if self.root is None:
            raise RuntimeError("Project has no location yet.")
        return ProjectPaths(self.root)

    def prune_research(self, keep_scene_ids: set[str]) -> dict:
        """Drop research data for scenes that no longer exist; returns what was removed (for undo)."""
        removed: dict = {"candidates": {}, "scores": {}, "status": {}, "assignments": {}}
        for cid, c in list(self.visual_candidates.items()):
            if c.scene_id not in keep_scene_ids:
                removed["candidates"][cid] = self.visual_candidates.pop(cid)
                if cid in self.candidate_scores:
                    removed["scores"][cid] = self.candidate_scores.pop(cid)
        for sid in list(self.research_status):
            if sid not in keep_scene_ids:
                removed["status"][sid] = self.research_status.pop(sid)
        for sid in list(self.visual_assignments):
            if sid not in keep_scene_ids:
                removed["assignments"][sid] = self.visual_assignments.pop(sid)
        return removed

    def restore_research(self, removed: dict) -> None:
        self.visual_candidates.update(removed["candidates"])
        self.candidate_scores.update(removed["scores"])
        self.research_status.update(removed["status"])
        self.visual_assignments.update(removed["assignments"])

    def asset_path(self, asset: Asset) -> Path:
        return self.paths.resolve(asset.path)

    def missing_assets(self) -> list[Asset]:
        return [a for a in self.assets if not self.asset_path(a).is_file()]

    # ----- validation -----
    def validate(self) -> None:
        """Semantic validation performed before every save. Raises ``InvalidProjectError``."""
        problems: list[str] = []
        if not self.project_name.strip():
            problems.append("project name is empty")
        s = self.settings
        if s.width <= 0 or s.height <= 0 or s.fps <= 0:
            problems.append("resolution and fps must be positive")
        track_ids: set[str] = set()
        clip_ids: set[str] = set()
        for t in self.timeline.tracks:
            if t.id in track_ids:
                problems.append(f"duplicate track id {t.id}")
            track_ids.add(t.id)
            last_end = 0.0
            for c in t.clips:
                if c.id in clip_ids:
                    problems.append(f"duplicate clip id {c.id}")
                clip_ids.add(c.id)
                if c.track_id != t.id:
                    problems.append(f"clip {c.id} claims track {c.track_id} but is stored on {t.id}")
                if c.kind == "media" and c.asset_id not in self.assets:
                    problems.append(f"clip {c.id} references unknown asset {c.asset_id}")
                if c.timeline_start < -TIME_EPSILON or c.duration <= 0:
                    problems.append(f"clip {c.id} has an invalid time range")
                if c.timeline_start < last_end - TIME_EPSILON:
                    problems.append(f"clip {c.id} overlaps the previous clip on {t.name}")
                last_end = max(last_end, c.timeline_end)
        scene_ids: set[str] = set()
        prev_end = 0.0
        for sc in self.scenes:
            if sc.id in scene_ids:
                problems.append(f"duplicate scene id {sc.id}")
            scene_ids.add(sc.id)
            if sc.end <= sc.start or sc.start < -TIME_EPSILON:
                problems.append(f"scene {sc.label} has an invalid time range")
            if sc.start < prev_end - 1e-6:
                problems.append(f"scene {sc.label} overlaps the previous scene")
            prev_end = max(prev_end, sc.end)
        for sid in self.visual_intents:
            if sid not in scene_ids:
                problems.append(f"visual intent for unknown scene {sid}")
        for sid, a in self.visual_assignments.items():
            if a.asset_id is not None and a.asset_id not in self.assets:
                problems.append(f"visual assignment for {sid} references unknown asset {a.asset_id}")
            if a.candidate_id is not None and a.candidate_id not in self.visual_candidates:
                problems.append(f"visual assignment for {sid} references unknown candidate {a.candidate_id}")
        vo = self.voice_over.asset_id
        if vo is not None:
            asset = self.assets.get(vo)
            if asset is None or asset.type is not AssetType.AUDIO:
                problems.append("voice-over references a missing or non-audio asset")
        if problems:
            raise InvalidProjectError(
                "The project contains inconsistent data and was not saved.", problems=problems,
                details="; ".join(problems[:10]),
            )

    # ----- (de)serialisation -----
    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project": {
                "id": self.project_id,
                "name": self.project_name,
                "application_version": self.application_version,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
            "settings": self.settings.to_dict(),
            "script": self.script.to_dict(),
            "voice_over": self.voice_over.to_dict(),
            "scenes": [s.to_dict() for s in self.scenes],
            "assets": self.assets.to_list(),
            "timeline": self.timeline.to_dict(),
            "render_settings": self.render_settings.to_dict(),
            "ai_decisions": [d.to_dict() for d in self.ai_decisions],
            "transcription": to_plain(self.transcription),
            "script_alignment": self.script_alignment.to_dict() if self.script_alignment else {},
            "scene_analysis": to_plain(self.scene_analysis),
            "visual_intents": {k: v.to_dict() for k, v in self.visual_intents.items()},
            "visual_preferences": self.visual_preferences.to_dict(),
            "research_settings": to_plain(self.research_settings),
            "research_queries": to_plain(self.research_queries),
            "research_sessions": to_plain(self.research_sessions),
            "visual_candidates": {k: v.to_dict() for k, v in self.visual_candidates.items()},
            "candidate_scores": to_plain(self.candidate_scores),
            "visual_assignments": {k: v.to_dict() for k, v in self.visual_assignments.items()},
            "source_metadata": to_plain(self.source_metadata),
            "research_status": to_plain(self.research_status),
            "editing_settings": self.editing_settings.to_dict(),
            "editing_strategy": to_plain(self.editing_strategy),
            "editing_sessions": to_plain(self.editing_sessions[-50:]),
            "editing_decisions": {k: v.to_dict() for k, v in self.editing_decisions.items()},
            "timeline_generation": to_plain(self.timeline_generation),
            "ai_overrides": to_plain(self.ai_overrides),
            "timeline_version": self.timeline_version,
            "counters": {"asset": self.assets.counter},
        }

    @classmethod
    def from_document(cls, doc: dict[str, Any], root: Path | None = None) -> "Project":
        doc = migrate_document(doc)
        validate_document(doc)
        try:
            meta = doc["project"]
            project = cls(
                project_id=meta["id"],
                project_name=meta["name"],
                created_at=meta["created_at"],
                updated_at=meta["updated_at"],
                settings=ProjectSettings.from_dict(doc["settings"]),
                script=ScriptData.from_dict(doc["script"]),
                voice_over=VoiceOverData.from_dict(doc["voice_over"]),
                scenes=[Scene.from_dict(s) for s in doc["scenes"]],
                transcription=from_plain(TranscriptionState, doc["transcription"]) if doc["transcription"] else TranscriptionState(),
                script_alignment=ScriptAlignment.from_dict(doc["script_alignment"]) if doc["script_alignment"] else None,
                scene_analysis=from_plain(SceneAnalysisState, doc["scene_analysis"]) if doc["scene_analysis"] else SceneAnalysisState(),
                visual_intents={k: VisualIntent.from_dict(v) for k, v in doc["visual_intents"].items()},
                visual_preferences=VisualPreferences.from_dict(doc["visual_preferences"]),
                research_settings=from_plain(ResearchSettings, doc["research_settings"]) if doc["research_settings"] else ResearchSettings(),
                research_queries={k: from_plain(ResearchQuery, v) for k, v in doc["research_queries"].items()},
                research_sessions=[from_plain(ResearchSession, v) for v in doc["research_sessions"]],
                visual_candidates={k: Candidate.from_dict(v) for k, v in doc["visual_candidates"].items()},
                candidate_scores={k: from_plain(CandidateScore, v) for k, v in doc["candidate_scores"].items()},
                visual_assignments={k: VisualAssignment.from_dict(v) for k, v in doc["visual_assignments"].items()},
                source_metadata=dict(doc["source_metadata"]),
                research_status={k: from_plain(SceneResearchState, v) for k, v in doc["research_status"].items()},
                editing_settings=EditingSettings.from_dict(doc["editing_settings"]) if doc["editing_settings"] else EditingSettings(),
                editing_strategy=from_plain(EditingStrategy, doc["editing_strategy"]) if doc["editing_strategy"] else EditingStrategy(),
                editing_sessions=[from_plain(EditingSession, v) for v in doc["editing_sessions"]],
                editing_decisions={k: EditingDecision.from_dict(v) for k, v in doc["editing_decisions"].items()},
                timeline_generation=from_plain(TimelineGeneration, doc["timeline_generation"]) if doc["timeline_generation"] else TimelineGeneration(),
                ai_overrides=[from_plain(OverrideRecord, v) for v in doc["ai_overrides"]],
                timeline_version=int(doc["timeline_version"]),
                assets=AssetRegistry(
                    [Asset.from_dict(a) for a in doc["assets"]], counter=int((doc.get("counters") or {}).get("asset", 0))
                ),
                timeline=Timeline.from_dict(doc["timeline"]),
                render_settings=RenderSettings.from_dict(doc["render_settings"]),
                ai_decisions=[AIDecision.from_dict(d) for d in doc["ai_decisions"]],
                schema_version=doc["schema_version"],
                application_version=meta.get("application_version", APP_VERSION),
                root=root,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidProjectError(
                "The project file is damaged or incomplete and cannot be opened.", details=f"{type(exc).__name__}: {exc}"
            ) from exc
        return project
