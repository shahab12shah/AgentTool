"""Pexels stock provider (photos and videos). Needs an API key in an environment variable.

NOT verified against the live service in this repository: tested against a local mock of the documented
response shape. Licence information is what Pexels states (the Pexels License); it is not verified here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

from app.core.exceptions import AcquisitionError
from app.media.asset import SourceType
from app.research.models import Acquisition, Candidate, LicenseInfo, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate

DEFAULT_API = "https://api.pexels.com"
LICENSE_URL = "https://www.pexels.com/license/"


def _slug_words(url: str) -> str:
    m = re.search(r"/(?:photo|video)/([^/]+?)(?:-\d+)?/?$", url or "")
    return m.group(1).replace("-", " ") if m else ""


class PexelsProvider(SourceProvider):
    name = "pexels"
    label = "Pexels"
    source_types = (SourceType.STOCK_IMAGE, SourceType.STOCK_VIDEO)

    def __init__(self, api_url: Callable[[], str] = lambda: "", key_env: Callable[[], str] = lambda: "PEXELS_API_KEY", **kw) -> None:
        super().__init__(**kw)
        self._api, self._key_env = api_url, key_env

    def _base(self) -> str:
        return (self._api().strip() or DEFAULT_API).rstrip("/")

    def _key(self) -> str:
        return os.environ.get(self._key_env().strip() or "PEXELS_API_KEY", "")

    def is_available(self) -> tuple[bool, str]:
        if not self._key():
            return False, f"The Pexels API key environment variable {self._key_env() or 'PEXELS_API_KEY'} is not set."
        return True, ""

    def config_key(self) -> str:
        return f"pexels:{self._base()}"

    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        auth = {"Authorization": self._key()}
        video = source_type is SourceType.STOCK_VIDEO
        data = self.http.request_json(f"{self._base()}/{'videos/search' if video else 'v1/search'}",
                                      params={"query": query.text, "per_page": min(limit, 40)}, headers=auth)
        out = []
        for item in data.get("videos" if video else "photos", []):
            c = blank_candidate(query, source_type, "VIDEO" if video else "IMAGE", self.name)
            page = item.get("url", "")
            words = _slug_words(page)
            c.title = (item.get("alt") or words or f"Pexels {item.get('id')}").strip()
            c.description = " ".join(x for x in (item.get("alt"), words) if x)
            c.width, c.height = item.get("width"), item.get("height")
            c.provider_id, c.source_reference = str(item.get("id")), page
            c.license = LicenseInfo("Pexels License", LICENSE_URL, (item.get("photographer") or (item.get("user") or {}).get("name")), "PROVIDER_STATED", False)
            c.acquisition = Acquisition.DOWNLOAD
            if video:
                c.duration = float(item["duration"]) if item.get("duration") else None
                c.thumbnail_url = item.get("image", "")
                files = sorted((f for f in item.get("video_files", []) if f.get("link")), key=lambda f: -(f.get("width") or 0))
                hd = next((f for f in files if (f.get("width") or 0) <= 1920), files[0] if files else None)
                if hd:
                    c.media_url, c.width, c.height = hd["link"], hd.get("width") or c.width, hd.get("height") or c.height
            else:
                src = item.get("src") or {}
                c.media_url, c.thumbnail_url = src.get("large2x") or src.get("large") or src.get("original", ""), src.get("medium", "")
            out.append(c)
        return out

    def acquire(self, candidate: Candidate, dest_dir: Path, ctx: SearchContext) -> Path:
        if not candidate.media_url:
            raise AcquisitionError("This Pexels item has no downloadable file URL.")
        ext = Path(candidate.media_url.split("?")[0]).suffix or (".mp4" if candidate.kind == "VIDEO" else ".jpg")
        return self.http.download(candidate.media_url, dest_dir / f"{candidate.candidate_id}{ext}")
