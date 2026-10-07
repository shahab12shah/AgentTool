"""YouTube Data API v3 provider. Produces *references* with suggested segments; it does NOT download videos.

NOT verified against the live API in this repository (no network during development): tested against a
local mock of the documented ``search.list`` / ``videos.list`` response shapes. Needs an API key in the
environment variable named in Settings (the key itself is never stored or logged).

Useful sections: only when the video description contains chapter timestamps ("1:23 Panel assembly")
can a section be proposed. Otherwise the segment is UNKNOWN and the candidate says so.
"""

from __future__ import annotations

import os
import re
from typing import Callable

from app.core.exceptions import AcquisitionError, ProviderError
from app.core.textutil import STOPWORDS, stem, tokenize
from app.media.asset import SourceType
from app.research.models import Acquisition, Candidate, ClipSegment, LicenseInfo, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate

DEFAULT_API = "https://www.googleapis.com/youtube/v3"
_ISO = re.compile(r"^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")
_CHAPTER = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s*[-–—:]?\s*(.+?)\s*$", re.M)
CLIP_STEPS = (3.5, 5.0, 7.0, 10.0)


def parse_iso_duration(s: str) -> float | None:
    m = _ISO.match(s or "")
    if not m or not any(m.groups()):
        return None
    d, h, mi, sec = (int(x or 0) for x in m.groups())
    return float(d * 86400 + h * 3600 + mi * 60 + sec)


def parse_chapters(description: str, duration: float | None) -> list[tuple[float, float | None, str]]:
    """[(start, end, title)] from ``0:45 Title`` lines (YouTube chapter format). Empty when there are none."""
    marks = []
    for m in _CHAPTER.finditer(description or ""):
        h, mi, sec, title = int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), m.group(4)
        marks.append((h * 3600 + mi * 60 + sec, title))
    marks = sorted(set(marks))
    out = []
    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else duration
        out.append((float(start), float(end) if end is not None else None, title))
    return out


class YouTubeProvider(SourceProvider):
    name = "youtube"
    label = "YouTube"
    source_types = (SourceType.YOUTUBE,)

    def __init__(self, api_url: Callable[[], str] = lambda: "", key_env: Callable[[], str] = lambda: "YOUTUBE_API_KEY", **kw) -> None:
        super().__init__(**kw)
        self._api, self._key_env = api_url, key_env

    def _base(self) -> str:
        return (self._api().strip() or DEFAULT_API).rstrip("/")

    def _key(self) -> str:
        return os.environ.get(self._key_env().strip() or "YOUTUBE_API_KEY", "")

    def is_available(self) -> tuple[bool, str]:
        if not self._key():
            return False, f"The YouTube API key environment variable {self._key_env() or 'YOUTUBE_API_KEY'} is not set."
        return True, ""

    def config_key(self) -> str:
        return f"youtube:{self._base()}"

    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        key = self._key()
        found = self.http.request_json(f"{self._base()}/search", params={
            "part": "snippet", "type": "video", "q": query.text, "maxResults": min(limit, 25), "key": key, "safeSearch": "moderate"})
        ids = [i["id"]["videoId"] for i in found.get("items", []) if (i.get("id") or {}).get("videoId")]
        if not ids:
            return []
        details = self.http.request_json(f"{self._base()}/videos", params={"part": "contentDetails,snippet,status", "id": ",".join(ids), "key": key})
        by_id = {v["id"]: v for v in details.get("items", [])}
        q_terms = {stem(t.norm) for t in tokenize(query.text) if t.norm not in STOPWORDS}
        out = []
        for vid in ids:
            v = by_id.get(vid)
            if not v:
                continue
            sn = v.get("snippet", {})
            c = blank_candidate(query, SourceType.YOUTUBE, "VIDEO", self.name)
            c.title, c.description = sn.get("title", ""), sn.get("description", "")
            c.tags = list(sn.get("tags") or [])
            c.duration = parse_iso_duration((v.get("contentDetails") or {}).get("duration", ""))
            c.provider_id, c.source_reference = vid, f"https://www.youtube.com/watch?v={vid}"
            thumbs = sn.get("thumbnails") or {}
            c.thumbnail_url = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")
            c.acquisition = Acquisition.REFERENCE_ONLY  # this application does not download YouTube videos
            lic = (v.get("status") or {}).get("license", "youtube")
            c.license = LicenseInfo("Creative Commons (as labelled by YouTube)" if lic == "creativeCommon" else "Standard YouTube License",
                                    None, sn.get("channelTitle"), "PROVIDER_STATED", False)
            c.metadata = {"channel": sn.get("channelTitle"), "published": sn.get("publishedAt")}
            c.segment = self._segment(c, q_terms, ctx.default_clip_seconds)
            out.append(c)
        return out

    @staticmethod
    def _segment(c: Candidate, q_terms: set[str], clip: float) -> ClipSegment:
        chapters = parse_chapters(c.description, c.duration)
        if not chapters:
            return ClipSegment(None, None, "UNKNOWN")
        def overlap(ch):
            return len(q_terms & {stem(t.norm) for t in tokenize(ch[2]) if t.norm not in STOPWORDS})
        best = max(chapters, key=overlap)
        if overlap(best) == 0:
            return ClipSegment(None, None, "UNKNOWN")
        start, end, _title = best
        length = clip if end is None else min(clip, max(end - start, 1.0))
        return ClipSegment(start, start + length, "CHAPTER")

    def acquire(self, candidate: Candidate, dest_dir, ctx: SearchContext):
        raise AcquisitionError("Downloading YouTube videos is not implemented. The reference and suggested segment are saved with the visual assignment.")
