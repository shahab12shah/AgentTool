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
    """What the user last chose in the Export screen. Saved with the project; the renderer reads a frozen copy (see ``RenderSnapshot``).

    Zero / empty means "follow the project" or "derive from ``quality``", so the same settings keep working when the project changes.
    """

    preset_id: str = "youtube_1080p"  # youtube_1080p | youtube_4k | high_quality | draft | custom
    resolution: str = "1080p"  # short edge: 480p | 720p | 1080p | 2160p (aspect ratio always follows the project)
    fps: int = 0  # 0 = the project FPS (the master output rate)
    quality: str = "high"  # draft | standard | high | maximum | custom
    container: str = "mp4"  # mp4 | mkv | webm
    video_codec: str = "h264"  # h264 | h265 | vp9 | av1
    audio_codec: str = "aac"  # aac | opus | flac
    crf: int = 0  # custom quality only (0 = automatic)
    bitrate_kbps: int = 0  # custom quality only; 0 = constant quality (CRF)
    audio_bitrate_kbps: int = 192
    audio_sample_rate: int = 48000
    encoder_preset: str = ""  # codec specific speed/efficiency preset; "" = from quality
    hardware_acceleration: str = "auto"  # auto | cpu | hardware
    output_dir: str = ""  # "" = <project>/renders
    use_proxies: bool = False  # explicit opt-in: export from proxy media (never the default)
    proxy_resolution: str = "720p"  # 540p | 720p | 1080p
    duration_tolerance: float = 0.5  # seconds of allowed difference between timeline and rendered duration

    @property
    def output_format(self) -> str:
        return self.container

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderSettings":
        from dataclasses import fields

        base = cls()
        kw = {}
        for f in fields(cls):
            if f.name in d:
                kw[f.name] = type(getattr(base, f.name))(d[f.name]) if not isinstance(getattr(base, f.name), bool) else bool(d[f.name])
        if "crf" in kw and "quality" not in d:  # Phase 1 documents stored crf=18 without a quality level: keep it as a custom choice
            kw["quality"] = "custom"
        return cls(**kw)


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
        version = 4
    if version == 4:  # Phase 4 -> Phase 5: additive sections
        for key, empty in (("audio_settings", {}), ("audio_analysis", {}), ("audio_processing", {}), ("music_assignments", []), ("sfx_assignments", []),
                           ("ducking_events", []), ("caption_settings", {}), ("caption_segments", []), ("caption_styles", {}), ("keyword_emphasis", {}),
                           ("text_graphics", []), ("motion_graphics", []), ("presentation_plans", {}), ("presentation_decisions", {}),
                           ("presentation_overrides", []), ("presentation_generation", {}), ("presentation_sessions", [])):
            doc.setdefault(key, empty)
        doc["schema_version"] = 5
        version = 5
    if version == 5:  # Phase 5 -> Phase 6: render history and proxy mapping (render_settings fields are additive)
        doc.setdefault("render_history", [])
        doc.setdefault("proxies", {})
        doc["schema_version"] = 6
        version = 6
    if version == 6:  # Phase 6 -> Phase 7: reference style analysis (additive; nothing existing changes)
        doc.setdefault("reference_settings", {})
        doc.setdefault("reference_assets", {})
        doc.setdefault("reference_analysis", {})
        doc.setdefault("reference_style_profile", {})
        doc.setdefault("reference_style_overrides", {})
        doc.setdefault("style_application_history", [])
        doc["schema_version"] = 7
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
_V5_SECTIONS: dict[str, type] = {
    "audio_settings": dict,
    "audio_analysis": dict,
    "audio_processing": dict,
    "music_assignments": list,
    "sfx_assignments": list,
    "ducking_events": list,
    "caption_settings": dict,
    "caption_segments": list,
    "caption_styles": dict,
    "keyword_emphasis": dict,
    "text_graphics": list,
    "motion_graphics": list,
    "presentation_plans": dict,
    "presentation_decisions": dict,
    "presentation_overrides": list,
    "presentation_generation": dict,
    "presentation_sessions": list,
}
_V6_SECTIONS: dict[str, type] = {
    "render_history": list,
    "proxies": dict,
}
_V7_SECTIONS: dict[str, type] = {
    "reference_settings": dict,
    "reference_assets": dict,
    "reference_analysis": dict,
    "reference_style_profile": dict,
    "reference_style_overrides": dict,
    "style_application_history": list,
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
    for key, typ in {**_REQUIRED_SECTIONS, **(_V2_SECTIONS if isinstance(version, int) and version >= 2 else {}), **(_V3_SECTIONS if isinstance(version, int) and version >= 3 else {}), **(_V4_SECTIONS if isinstance(version, int) and version >= 4 else {}), **(_V5_SECTIONS if isinstance(version, int) and version >= 5 else {}), **(_V6_SECTIONS if isinstance(version, int) and version >= 6 else {}), **(_V7_SECTIONS if isinstance(version, int) and version >= 7 else {})}.items():
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
