"""AI image provider (OpenAI-compatible ``/images/generations``).

Searching never calls the service: it only *proposes* a concept built from the research brief (no cost,
no fabricated evidence). Generation happens on explicit request (``generate``). NOT verified against a real
service here; tested against a local mock.

Safety rules baked into the prompt: no real people's likeness, no documents/charts/logos/screenshots, no text
that could pass as factual evidence. For evidence-required scenes the proposal is labelled DECORATIVE only.
"""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import Callable

from app.core.exceptions import AcquisitionError, ProviderError
from app.media.asset import SourceType
from app.research.models import Acquisition, Candidate, CandidateStatus, EvidenceKind, LicenseInfo, ResearchBrief, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate

NEGATIVE_RULES = ("Do not depict real, identifiable people. Do not render documents, forms, charts, graphs, logos, brand marks or "
                  "readable text. Nothing in the image may be presented as factual evidence.")


def build_prompt(brief: ResearchBrief) -> str:
    """Concept prompt derived from the structured brief (not from raw script words)."""
    subject = brief.primary_subject or brief.topic
    detail = ", ".join(x for x in (brief.secondary_subject, brief.action, brief.context) if x)
    style = {"PROCESS": "documentary-style photograph of the real-world process", "OBJECT": "clean studio-quality product photograph",
             "LOCATION": "establishing photograph of the place", "ABSTRACT": "stylised conceptual illustration",
             "EVENT": "photojournalistic scene"}.get(brief.visual_type, "photorealistic documentary-style image")
    avoid = f" Avoid: {'; '.join(brief.avoid[:3])}." if brief.avoid else ""
    return f"{style.capitalize()} of {subject}" + (f" ({detail})" if detail else "") + f", 16:9 composition, professional lighting.{avoid} {NEGATIVE_RULES}"


class AIImageProvider(SourceProvider):
    name = "ai_image"
    label = "AI image generation"
    source_types = (SourceType.AI_GENERATED,)

    def __init__(self, base_url: Callable[[], str] = lambda: "", model: Callable[[], str] = lambda: "gpt-image-1",
                 key_env: Callable[[], str] = lambda: "OPENAI_API_KEY", **kw) -> None:
        super().__init__(**kw)
        self._base, self._model, self._key_env = base_url, model, key_env

    def _key(self) -> str:
        return os.environ.get(self._key_env().strip() or "OPENAI_API_KEY", "")

    def is_available(self) -> tuple[bool, str]:
        if not self._base().strip():
            return False, "No image-generation API base URL is configured (Settings → Visual research)."
        if not self._key():
            return False, f"The API key environment variable {self._key_env() or 'OPENAI_API_KEY'} is not set."
        return True, ""

    def config_key(self) -> str:
        return f"ai:{self._base()}:{self._model()}"

    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        if ctx.brief is None:
            return []
        b = ctx.brief
        c = blank_candidate(query, SourceType.AI_GENERATED, "IMAGE", self.name)
        c.title = f"AI concept: {b.primary_subject or b.topic}"
        c.description = f"Proposed AI-generated visual for “{b.topic}”. Not generated yet."
        c.tags = [t for t in (b.primary_subject, b.secondary_subject, b.action, b.context) if t]
        c.prompt = build_prompt(b)
        c.provider_id = f"concept:{b.scene_id}"
        c.source_reference = "AI-generated (on request)"
        c.status = CandidateStatus.PROPOSED
        c.acquisition = Acquisition.GENERATE
        c.evidence_kind = EvidenceKind.DECORATIVE  # AI imagery is never evidence
        c.license = LicenseInfo("AI-generated: terms depend on the generating service", None, None, "UNKNOWN", False)
        return [c]

    def generate(self, candidate: Candidate, dest_dir: Path) -> Path:
        """Call the image service and save a PNG. Raises ``ProviderError`` / ``AcquisitionError``."""
        ok, why = self.is_available()
        if not ok:
            raise ProviderError(why)
        data = self.http.request_json(self._base().rstrip("/") + "/images/generations",
                                      headers={"Authorization": f"Bearer {self._key()}"},
                                      body={"model": self._model(), "prompt": candidate.prompt, "n": 1, "size": "1536x1024"})
        item = (data.get("data") or [None])[0]
        if not item:
            raise AcquisitionError("The image service returned no image.")
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"ai_{candidate.candidate_id or uuid.uuid4().hex[:8]}.png"
        if item.get("b64_json"):
            out.write_bytes(base64.b64decode(item["b64_json"]))
        elif item.get("url"):
            self.http.download(item["url"], out)
        else:
            raise AcquisitionError("The image service response contained neither image data nor a URL.")
        return out

    def acquire(self, candidate: Candidate, dest_dir: Path, ctx: SearchContext) -> Path:
        if candidate.local_path and Path(candidate.local_path).is_file():
            return Path(candidate.local_path)
        return self.generate(candidate, dest_dir)
