"""Project document schema: section dataclasses, validation and migration.

The on-disk format is a JSON document (see ``project.py`` for conversion):

    {schema_version, project{id,name,application_version,created_at,updated_at},
     settings{width,height,fps,aspect_ratio}, script, voice_over, scenes, assets,
     timeline, render_settings, ai_decisions, counters}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import SCHEMA_VERSION
from app.core.exceptions import InvalidProjectError


@dataclass
class ProjectSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    aspect_ratio: str = "16:9"

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "fps": self.fps, "aspect_ratio": self.aspect_ratio}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectSettings":
        return cls(int(d["width"]), int(d["height"]), int(d["fps"]), str(d.get("aspect_ratio", "16:9")))


@dataclass
class ScriptData:
    text: str = ""
    # Phase 2 fills these in; kept so the format does not need to change later.
    analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "analysis": self.analysis}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScriptData":
        return cls(text=str(d.get("text", "")), analysis=d.get("analysis"))


@dataclass
class VoiceOverData:
    asset_id: str | None = None
    filename: str | None = None
    duration: float | None = None
    transcript: dict[str, Any] | None = None  # filled by a future transcription service

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "filename": self.filename,
            "duration": self.duration,
            "transcript": self.transcript,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VoiceOverData":
        return cls(d.get("asset_id"), d.get("filename"), d.get("duration"), d.get("transcript"))


@dataclass
class RenderSettings:
    container: str = "mp4"
    video_codec: str = "h264"
    audio_codec: str = "aac"
    crf: int = 18

    def to_dict(self) -> dict[str, Any]:
        return {
            "container": self.container,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "crf": self.crf,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderSettings":
        base = cls()
        return cls(
            d.get("container", base.container),
            d.get("video_codec", base.video_codec),
            d.get("audio_codec", base.audio_codec),
            int(d.get("crf", base.crf)),
        )


@dataclass
class AIDecision:
    """An editable record of something an AI (or the user) decided.

    The timeline stores the consequence; this stores the *why*, so decisions can be
    inspected, replaced and overridden.
    """

    id: str
    kind: str  # e.g. "scene_split", "visual_choice"
    target_id: str | None = None  # scene / clip the decision applies to
    author: str = "ai"  # "ai" | "user"
    payload: dict[str, Any] = field(default_factory=dict)
    overridden: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target_id": self.target_id,
            "author": self.author,
            "payload": self.payload,
            "overridden": self.overridden,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AIDecision":
        return cls(
            d["id"], d["kind"], d.get("target_id"), d.get("author", "ai"),
            dict(d.get("payload") or {}), bool(d.get("overridden", False)), d.get("created_at", ""),
        )


# ----------------------------------------------------------------- migration
def migrate_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an older document to the current schema. Returns a new dict; the input is untouched."""
    version = doc.get("schema_version")
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise InvalidProjectError(
            f"This project was created by a newer version of the application (schema {version}). "
            "Please update the application to open it."
        )
    doc = dict(doc)
    if version == 1:  # Phase 1 -> Phase 2: purely additive sections
        doc.setdefault("transcription", {})
        doc.setdefault("script_alignment", {})
        doc.setdefault("scene_analysis", {})
        doc.setdefault("visual_intents", {})
        doc.setdefault("visual_preferences", {})
        doc["scenes"] = []  # Phase 1 never produced scenes; the Scene shape changed
        doc["schema_version"] = 2
        version = 2
    if version == 2:  # Phase 2 -> Phase 3: purely additive sections
        doc.setdefault("research_settings", {})
        doc.setdefault("research_queries", {})
        doc.setdefault("research_sessions", [])
        doc.setdefault("visual_candidates", {})
        doc.setdefault("candidate_scores", {})
        doc.setdefault("visual_assignments", {})
        doc.setdefault("source_metadata", {})
        doc.setdefault("research_status", {})
        doc["schema_version"] = 3
        version = 3
    if version == 3:  # Phase 3 -> Phase 4: additive sections; the timeline gains the V6 captions track
        doc.setdefault("editing_settings", {})
        doc.setdefault("editing_strategy", {})
        doc.setdefault("editing_sessions", [])
        doc.setdefault("editing_decisions", {})
        doc.setdefault("timeline_generation", {})
        doc.setdefault("ai_overrides", [])
        doc.setdefault("timeline_version", 0)
        tl = dict(doc.get("timeline") or {})
        tracks = list(tl.get("tracks", []))
        if tracks and not any(t.get("id") == "track_v6" for t in tracks):
            at = next((i for i, t in enumerate(tracks) if t.get("kind") == "audio"), len(tracks))
            tracks.insert(at, {"id": "track_v6", "name": "V6 Captions", "kind": "captions", "hidden": False, "muted": False,
                               "locked": False, "clips": []})
        tl["tracks"] = tracks
        doc["timeline"] = tl
        doc["schema_version"] = 4
    return doc


# ---------------------------------------------------------------- validation
_REQUIRED_SECTIONS: dict[str, type] = {
    "project": dict,
    "settings": dict,
    "script": dict,
    "voice_over": dict,
    "scenes": list,
    "assets": list,
    "timeline": dict,
    "render_settings": dict,
    "ai_decisions": list,
}
_V2_SECTIONS: dict[str, type] = {
    "transcription": dict,
    "script_alignment": dict,
    "scene_analysis": dict,
    "visual_intents": dict,
    "visual_preferences": dict,
}
_V3_SECTIONS: dict[str, type] = {
    "research_settings": dict,
    "research_queries": dict,
    "research_sessions": list,
    "visual_candidates": dict,
    "candidate_scores": dict,
    "visual_assignments": dict,
    "source_metadata": dict,
    "research_status": dict,
}
_V4_SECTIONS: dict[str, type] = {
    "editing_settings": dict,
    "editing_strategy": dict,
    "editing_sessions": list,
    "editing_decisions": dict,
    "timeline_generation": dict,
    "ai_overrides": list,
}


def validate_document(doc: Any) -> None:
    """Structural validation of a project document. Raises ``InvalidProjectError`` listing every problem."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        raise InvalidProjectError("The project file is not a valid project.", problems=["root is not an object"])
    version = doc.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        problems.append("schema_version missing or not an integer")
    elif version > SCHEMA_VERSION:
        raise InvalidProjectError(
            f"This project was created by a newer version of the application (schema {version}). "
            "Please update the application to open it."
        )
    elif version < 1:
        problems.append(f"unsupported schema_version {version}")
    for key, typ in {**_REQUIRED_SECTIONS, **(_V2_SECTIONS if isinstance(version, int) and version >= 2 else {}), **(_V3_SECTIONS if isinstance(version, int) and version >= 3 else {}), **(_V4_SECTIONS if isinstance(version, int) and version >= 4 else {})}.items():
        if key not in doc:
            problems.append(f"missing section '{key}'")
        elif not isinstance(doc[key], typ):
            problems.append(f"section '{key}' must be {typ.__name__}")
    if not problems:
        meta = doc["project"]
        for key in ("id", "name", "created_at", "updated_at"):
            if not isinstance(meta.get(key), str) or not meta.get(key):
                problems.append(f"project.{key} missing")
        s = doc["settings"]
        for key in ("width", "height", "fps"):
            v = s.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                problems.append(f"settings.{key} must be a positive number")
        ids: set[str] = set()
        for i, a in enumerate(doc["assets"]):
            if not isinstance(a, dict) or not a.get("id") or not a.get("path") or not a.get("type"):
                problems.append(f"assets[{i}] is incomplete")
            elif a["id"] in ids:
                problems.append(f"duplicate asset id {a['id']}")
            else:
                ids.add(a["id"])
        tracks = doc["timeline"].get("tracks")
        if not isinstance(tracks, list):
            problems.append("timeline.tracks must be a list")
        else:
            for i, t in enumerate(tracks):
                if not isinstance(t, dict) or not t.get("id") or not t.get("kind") or not isinstance(t.get("clips", []), list):
                    problems.append(f"timeline.tracks[{i}] is incomplete")
    if problems:
        raise InvalidProjectError(
            "The project file is damaged or incomplete and cannot be opened.", problems=problems,
            details="; ".join(problems[:10]),
        )
