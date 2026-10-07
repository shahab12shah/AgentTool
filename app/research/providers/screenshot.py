"""Screenshot / evidence provider: captures a web page with headless Chromium.

Where do pages come from? (1) a small registry of OFFICIAL SITE HOMEPAGES for entities named in the brief
(IRS, SEC, ...). These are homepages, not the specific document, so they are offered as *evidence candidates
that need human verification*; (2) URLs the user supplies explicitly (``capture_url``). Capturing a page is
not a licence to reuse it.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from app.core.exceptions import AcquisitionError, ProviderError
from app.core.textutil import normalize_token, tokenize
from app.media.asset import SourceType
from app.research.http import USER_AGENT, is_public_url
from app.research.models import Acquisition, Candidate, EvidenceKind, LicenseInfo, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate

OFFICIAL_SITES: dict[str, tuple[str, str]] = {
    "irs": ("Internal Revenue Service (IRS)", "https://www.irs.gov/"),
    "sec": ("U.S. Securities and Exchange Commission", "https://www.sec.gov/"),
    "fda": ("U.S. Food and Drug Administration", "https://www.fda.gov/"),
    "fbi": ("Federal Bureau of Investigation", "https://www.fbi.gov/"),
    "nasa": ("NASA", "https://www.nasa.gov/"),
    "epa": ("U.S. Environmental Protection Agency", "https://www.epa.gov/"),
    "ftc": ("Federal Trade Commission", "https://www.ftc.gov/"),
    "cdc": ("Centers for Disease Control and Prevention", "https://www.cdc.gov/"),
    "treasury": ("U.S. Department of the Treasury", "https://home.treasury.gov/"),
    "federal reserve": ("Federal Reserve", "https://www.federalreserve.gov/"),
    "fed": ("Federal Reserve", "https://www.federalreserve.gov/"),
    "european central bank": ("European Central Bank", "https://www.ecb.europa.eu/"),
    "ecb": ("European Central Bank", "https://www.ecb.europa.eu/"),
}
VIEWPORT = (1280, 720)


def find_chromium(configured: str = "") -> str | None:
    if configured and Path(configured).is_file():
        return configured
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for p in sorted(Path("/opt/pw-browsers").glob("chromium*/chrome-linux*/chrome")):
        return str(p)
    return None


class ScreenshotProvider(SourceProvider):
    name = "screenshot"
    label = "Page screenshots (Chromium)"
    source_types = (SourceType.SCREENSHOT,)
    verified_live = False  # captures are verified against local pages only; real sites were unreachable

    def __init__(self, chromium_path: Callable[[], str] = lambda: "", extra_sites: dict[str, tuple[str, str]] | None = None,
                 allow_private: bool = False, **kw) -> None:
        super().__init__(**kw)
        self._chromium = chromium_path
        self.sites = {**OFFICIAL_SITES, **(extra_sites or {})}
        self.allow_private = allow_private

    def is_available(self) -> tuple[bool, str]:
        if not find_chromium(self._chromium()):
            return False, "Chromium/Chrome was not found. Install it or set its path in Settings → Visual research."
        return True, ""

    def config_key(self) -> str:
        return f"screenshot:{find_chromium(self._chromium())}:{len(self.sites)}"

    # ------------------------------------------------------------------
    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        words = " ".join(t.norm for t in tokenize(query.text))
        urls: list[tuple[str, str]] = []
        for key, (title, url) in self.sites.items():
            if f" {key} " in f" {words} " and (title, url) not in urls:
                urls.append((title, url))
        out, errors = [], []
        for title, url in urls[:limit]:
            if ctx.should_cancel():
                break
            try:
                out.append(self.capture_url(url, query, ctx, title=f"{title} — official website (homepage)"))
            except ProviderError as exc:
                errors.append(exc.user_message)
        if urls and not out and errors:
            raise ProviderError(errors[0])
        return out

    def capture_url(self, url: str, query: ResearchQuery, ctx: SearchContext, title: str = "") -> Candidate:
        if urlsplit(url).scheme not in ("http", "https"):
            raise ProviderError("Only http(s) pages can be captured.")
        if not (self.allow_private or is_public_url(url)):
            raise ProviderError("That address points to this computer or a private network and was refused.")
        chrome = find_chromium(self._chromium())
        if not chrome or ctx.cache_dir is None:
            raise ProviderError("Chromium is not available for screenshots.")
        out = ctx.cache_dir / "screenshots" / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.is_file():
            self._preflight(url)
            cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--disable-dev-shm-usage",
                   f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}", "--virtual-time-budget=8000", f"--screenshot={out}", url]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except (subprocess.SubprocessError, OSError) as exc:
                raise ProviderError("The page could not be captured.", details=str(exc)) from exc
            if not out.is_file() or out.stat().st_size < 500:
                out.unlink(missing_ok=True)
                raise ProviderError("The page could not be captured (it may be unreachable or blocked).", details=url)
        c = blank_candidate(query, SourceType.SCREENSHOT, "IMAGE", self.name)
        host = urlsplit(url).hostname or url
        c.title = title or f"Screenshot of {host}"
        c.description = f"Screenshot of {url}. Verify that it shows the document or page the narration refers to."
        c.tags = [host]
        c.width, c.height = VIEWPORT
        c.provider_id, c.source_reference = url, url
        c.local_path = c.thumbnail_path = str(out)
        c.acquisition = Acquisition.CAPTURE
        c.evidence_kind = EvidenceKind.EVIDENCE
        c.license = LicenseInfo("Not a licence: screenshots of third-party pages need their own rights check", None, host, "UNKNOWN", False)
        return c

    def _preflight(self, url: str) -> None:
        """Chromium renders its own error page for unreachable/4xx pages and still exits 0, which would look like evidence.
        So check the page answers with a success status first, and that redirects did not lead to a private address."""
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                final = resp.geturl()
                if resp.status >= 400:
                    raise ProviderError(f"The page answered with an error (HTTP {resp.status}).", details=url)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"The page answered with an error (HTTP {exc.code}), so it was not captured.", details=url) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("The page could not be captured (it may be unreachable or blocked).", details=f"{url}: {exc}") from exc
        if not (self.allow_private or is_public_url(final)):
            raise ProviderError("The page redirected to a private network address and was refused.", details=final)

    def fetch_thumbnail(self, candidate: Candidate, dest: Path, http) -> bool:
        src = Path(candidate.local_path)
        if src.is_file():
            shutil.copyfile(src, dest)
            return True
        return False

    def acquire(self, candidate: Candidate, dest_dir: Path, ctx: SearchContext) -> Path:
        src = Path(candidate.local_path)
        if not src.is_file():
            raise AcquisitionError("The captured screenshot file is missing; search again.")
        return src
