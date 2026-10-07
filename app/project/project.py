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
from app.project.project_schema import (
    AIDecision,
    ProjectSettings,
    RenderSettings,
    Scene,
    ScriptData,
    VoiceOverData,
    migrate_document,
    validate_document,
)
from app.storage.paths import ProjectPaths
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
                if c.asset_id not in self.assets:
                    problems.append(f"clip {c.id} references unknown asset {c.asset_id}")
                if c.timeline_start < -TIME_EPSILON or c.duration <= 0:
                    problems.append(f"clip {c.id} has an invalid time range")
                if c.timeline_start < last_end - TIME_EPSILON:
                    problems.append(f"clip {c.id} overlaps the previous clip on {t.name}")
                last_end = max(last_end, c.timeline_end)
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
