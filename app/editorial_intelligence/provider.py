"""EditorialIntelligenceProvider: the optional AI judgement behind the deterministic analysis.

The deterministic stages ARE the local analysis and always run (no network, reproducible). A provider can only ADD judgement on top - extra narrative roles, relationships and review-only
observations - and everything it returns is validated strictly before it touches a result: unknown scene ids, unknown enum values, malformed or out-of-range fields are dropped (and counted),
confidences are capped, nothing it says can carry an edit action, and it can never remove or overrule a deterministic finding. A provider failure (timeout, rate limit, invalid JSON, nothing
usable) is reported as a failed AI analysis; the run is then PARTIAL, never a completed assessment. Providers return concise findings only, never reasoning.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.core.exceptions import AppError
from app.editorial_intelligence.context import EditorialContext
from app.editorial_intelligence.editorial_models import Category, NarrativeRole, RelationType

AI_CONFIDENCE_CAP = 85.0  # an AI judgement is never presented as more certain than this
MAX_ITEMS = 400


class EditorialProviderError(AppError):
    pass


class EditorialIntelligenceProvider(ABC):
    name = "abstract"
    label = "Abstract"

    def available(self) -> bool:
        return True

    @abstractmethod
    def analyze_story(self, project_context: dict[str, Any]) -> dict[str, Any] | None: ...

    @abstractmethod
    def analyze_scene_relationships(self, project_context: dict[str, Any]) -> dict[str, Any] | None: ...

    @abstractmethod
    def generate_recommendations(self, analysis_context: dict[str, Any]) -> dict[str, Any] | None: ...


class LocalEditorialProvider(EditorialIntelligenceProvider):
    """The default: the deterministic stages are the analysis, so there is nothing to add."""

    name, label = "local", "Local (rule-based)"

    def analyze_story(self, project_context):
        return None

    def analyze_scene_relationships(self, project_context):
        return None

    def generate_recommendations(self, analysis_context):
        return None


class UnavailableEditorialProvider(EditorialIntelligenceProvider):
    name, label = "unavailable", "Not configured"

    def available(self) -> bool:
        return False

    def _no(self):
        raise EditorialProviderError("No AI provider is configured for editorial analysis. The rule-based analysis is unaffected.")

    def analyze_story(self, project_context):
        self._no()

    def analyze_scene_relationships(self, project_context):
        self._no()

    def generate_recommendations(self, analysis_context):
        self._no()


class APIEditorialProvider(EditorialIntelligenceProvider):
    """Asks ``client(task, payload)`` (any model service the host application wires in) for JSON. ``client`` may return a dict or a JSON string."""

    name, label = "api", "AI service"

    def __init__(self, client: Callable[[str, dict[str, Any]], Any]) -> None:
        self._client = client

    def _ask(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = self._client(task, payload)
        except TimeoutError as exc:
            raise EditorialProviderError("The AI service timed out.") from exc
        except EditorialProviderError:
            raise
        except Exception as exc:  # noqa: BLE001  (rate limits and network errors: any client failure is a failed analysis, reported plainly)
            raise EditorialProviderError(f"The AI service failed: {type(exc).__name__}") from exc
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError as exc:
                raise EditorialProviderError("The AI service did not return valid JSON.") from exc
        if not isinstance(raw, dict):
            raise EditorialProviderError("The AI service returned an unexpected response.")
        return raw

    def analyze_story(self, project_context):
        return self._ask("analyze_story", project_context)

    def analyze_scene_relationships(self, project_context):
        return self._ask("analyze_scene_relationships", project_context)

    def generate_recommendations(self, analysis_context):
        return self._ask("generate_recommendations", analysis_context)


# ------------------------------------------------------------------------------------------------------------------------ what a provider is shown
def compact_context(ctx: EditorialContext) -> dict[str, Any]:
    """The minimum a model needs: ids, narration, topic, entities, the current visual's name and source type. No paths, keys, thumbnails or settings."""
    scenes = []
    for s in ctx.scenes:
        sc = s.sc.scene
        scenes.append({"scene_id": sc.id, "start": round(sc.start, 2), "end": round(sc.end, 2), "narration": sc.narration[:600], "topic": sc.topic, "summary": sc.summary[:240],
                       "entities": [e.text for e in sc.entities][:8], "claims": [c.text[:140] for c in sc.claims][:4],
                       "visual": {"name": s.sc.asset.name if s.sc.asset else "", "source_type": s.sc.asset.source_type if s.sc.asset else ""}})
    return {"topic": ctx.topic, "content_type": ctx.content_type, "scenes": scenes[:MAX_ITEMS],
            "roles": [r.value for r in NarrativeRole], "relations": [r.value for r in RelationType]}


# ------------------------------------------------------------------------------------------------------------------------ validation of what comes back
def _num(v: Any, lo: float = 0.0, hi: float = 100.0, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else max(lo, min(hi, f))


def validate_story_payload(data: dict[str, Any], scene_ids: set[str]) -> tuple[dict[str, list[tuple[NarrativeRole, float]]], list[str]]:
    """{"roles": [{"scene_id", "role" | "roles": [...], "confidence"}]} -> {scene_id: [(role, confidence)]} plus a list of what was dropped and why."""
    problems: list[str] = []
    out: dict[str, list[tuple[NarrativeRole, float]]] = {}
    items = data.get("roles")
    if not isinstance(items, list):
        return out, ["the response has no 'roles' list"]
    for it in items[:MAX_ITEMS]:
        if not isinstance(it, dict) or it.get("scene_id") not in scene_ids:
            problems.append("a role entry named an unknown scene")
            continue
        raw = it.get("roles") if isinstance(it.get("roles"), list) else [it.get("role")]
        for r in raw[:3]:
            try:
                role = NarrativeRole(str(r).upper())
            except ValueError:
                problems.append(f"unknown role {str(r)[:30]!r}")
                continue
            out.setdefault(it["scene_id"], []).append((role, min(AI_CONFIDENCE_CAP, _num(it.get("confidence"), default=50.0))))
    return out, problems


def validate_relations_payload(data: dict[str, Any], scene_ids: set[str]) -> tuple[list[tuple[str, str, RelationType, float, str]], list[str]]:
    problems: list[str] = []
    out: list[tuple[str, str, RelationType, float, str]] = []
    items = data.get("relations")
    if not isinstance(items, list):
        return out, ["the response has no 'relations' list"]
    for it in items[:MAX_ITEMS]:
        if not isinstance(it, dict) or it.get("from") not in scene_ids or it.get("to") not in scene_ids or it.get("from") == it.get("to"):
            problems.append("a relation named unknown or identical scenes")
            continue
        try:
            typ = RelationType(str(it.get("type")).upper())
        except ValueError:
            problems.append(f"unknown relation type {str(it.get('type'))[:30]!r}")
            continue
        out.append((it["from"], it["to"], typ, min(AI_CONFIDENCE_CAP, _num(it.get("confidence"), default=50.0)), str(it.get("evidence", ""))[:200]))
    return out, problems


def validate_observations_payload(data: dict[str, Any], scene_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Review-only observations: {"observations": [{"scene_ids": [...], "category", "title", "detail", "confidence"}]}. They never carry an edit action."""
    problems: list[str] = []
    out: list[dict[str, Any]] = []
    items = data.get("observations")
    if not isinstance(items, list):
        return out, ["the response has no 'observations' list"]
    for it in items[:MAX_ITEMS]:
        if not isinstance(it, dict):
            problems.append("an observation was not an object")
            continue
        ids = [s for s in (it.get("scene_ids") or []) if s in scene_ids] if isinstance(it.get("scene_ids"), list) else []
        try:
            cat = Category(str(it.get("category")).upper())
        except ValueError:
            problems.append(f"unknown category {str(it.get('category'))[:30]!r}")
            continue
        title, detail = str(it.get("title", "")).strip()[:140], str(it.get("detail", "")).strip()[:400]
        if not ids or not title:
            problems.append("an observation had no valid scene or no title")
            continue
        out.append({"scene_ids": ids, "category": cat, "title": title, "detail": detail, "confidence": min(AI_CONFIDENCE_CAP, _num(it.get("confidence"), default=50.0))})
    return out, problems
