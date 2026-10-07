"""Wikimedia Commons provider (web images and videos with provider-stated licences, no API key).

NOT verified against the live service in this repository (network access was unavailable during
development): it is tested against a local mock that mimics the documented MediaWiki response shape.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

from app.core.exceptions import AcquisitionError
from app.media.asset import SourceType
from app.research.models import Acquisition, Candidate, LicenseInfo, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate

DEFAULT_API = "https://commons.wikimedia.org/w/api.php"
_TAG = re.compile(r"<[^>]+>")


def _text(v: object) -> str:
    return html.unescape(_TAG.sub("", str(v or ""))).strip()


class WikimediaProvider(SourceProvider):
    name = "wikimedia"
    label = "Wikimedia Commons"
    source_types = (SourceType.WEB_IMAGE, SourceType.WEB_VIDEO)

    def __init__(self, api_url: Callable[[], str] = lambda: "", **kw) -> None:
        super().__init__(**kw)
        self._api = api_url

    def _url(self) -> str:
        return self._api().strip() or DEFAULT_API

    def is_available(self) -> tuple[bool, str]:
        return True, ""  # keyless; failures are reported per search

    def config_key(self) -> str:
        return f"wikimedia:{self._url()}"

    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        video = source_type is SourceType.WEB_VIDEO
        data = self.http.request_json(self._url(), params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": limit,
            "gsrsearch": f"{query.text} filetype:{'video' if video else 'bitmap'}",
            "prop": "imageinfo", "iiprop": "url|size|mime|mediatype|extmetadata", "iiurlwidth": 640,
        })
        pages = ((data or {}).get("query") or {}).get("pages") or {}
        out: list[Candidate] = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            info = (page.get("imageinfo") or [None])[0]
            if not info:
                continue
            meta = info.get("extmetadata") or {}
            mtype = str(info.get("mediatype", "")).upper()
            if video != (mtype in ("VIDEO", "AUDIO") or str(info.get("mime", "")).startswith("video/")):
                continue
            c = blank_candidate(query, source_type, "VIDEO" if video else "IMAGE", self.name)
            title = str(page.get("title", "")).removeprefix("File:")
            c.title = _text(meta.get("ObjectName", {}).get("value")) or re.sub(r"\.\w+$", "", title).replace("_", " ")
            c.description = _text(meta.get("ImageDescription", {}).get("value"))
            c.tags = [t.strip() for t in _text(meta.get("Categories", {}).get("value")).split("|") if t.strip()]
            c.width, c.height = info.get("width"), info.get("height")
            c.duration = float(info["duration"]) if info.get("duration") else None
            c.provider_id = str(page.get("pageid", title))
            c.source_reference = info.get("descriptionurl") or ""
            c.media_url = info.get("url") or ""
            c.thumbnail_url = info.get("thumburl") or ""
            c.acquisition = Acquisition.DOWNLOAD
            c.license = LicenseInfo(_text(meta.get("LicenseShortName", {}).get("value")) or None,
                                    _text(meta.get("LicenseUrl", {}).get("value")) or None,
                                    _text(meta.get("Artist", {}).get("value")) or None, "PROVIDER_STATED", False)
            out.append(c)
        return out

    def acquire(self, candidate: Candidate, dest_dir: Path, ctx: SearchContext) -> Path:
        if not candidate.media_url:
            raise AcquisitionError("This Commons item has no downloadable file URL.")
        ext = Path(candidate.media_url.split("?")[0]).suffix or (".mp4" if candidate.kind == "VIDEO" else ".jpg")
        return self.http.download(candidate.media_url, dest_dir / f"{candidate.candidate_id}{ext}")
