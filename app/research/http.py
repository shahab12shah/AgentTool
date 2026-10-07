"""Small, safe HTTP helpers for providers (stdlib only): JSON GET/POST, capped downloads, retries, rate limiting.

* only http/https; downloads are size-capped and written to a ``.part`` file first
* private / loopback hosts are refused unless the caller opts in (guards against pointing a provider at an internal service)
* secrets are passed in headers/params by the caller and are never logged
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from app.core.exceptions import ProviderError
from app.logging.logger import get_logger

_log = get_logger(__name__)
USER_AGENT = "AgentTool/0.3 (visual research; contact: local desktop app)"
MAX_DOWNLOAD = 200 * 1024 * 1024


def is_public_url(url: str) -> bool:
    """True when ``url`` is http(s) and its host does not resolve to a private/loopback/link-local address."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    host = parts.hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # cannot resolve here (e.g. offline): let the request itself fail with a clear network error
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


class HttpClient:
    def __init__(self, timeout: float = 20.0, min_interval: float = 0.0, retries: int = 2, allow_private: bool = False) -> None:
        self.timeout, self.min_interval, self.retries, self.allow_private = timeout, min_interval, retries, allow_private
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _check(self, url: str) -> None:
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise ProviderError("Only http(s) addresses are allowed.", details=url[:80])
        if not self.allow_private and not is_public_url(url):
            raise ProviderError("That address points to this computer or a private network and was refused.", details=url[:80])

    def request_json(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
                     body: dict[str, Any] | None = None) -> Any:
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        self._check(url)
        data = json.dumps(body).encode() if body is not None else None
        hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if data is not None else "GET")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(min(2.0, 0.4 * (attempt + 1)))
                    last = exc
                    continue
                raise ProviderError(_http_text(exc.code), details=f"HTTP {exc.code}", retryable=exc.code in (429, 500, 502, 503, 504)) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise ProviderError("The service could not be reached.", details=str(exc), retryable=True) from exc
            except ValueError as exc:
                raise ProviderError("The service returned an unreadable response.", details=str(exc)) from exc
        raise ProviderError("The service could not be reached.", details=str(last), retryable=True)

    def download(self, url: str, dest: Path, *, headers: dict[str, str] | None = None, max_bytes: int = MAX_DOWNLOAD) -> Path:
        self._check(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        self._throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp, open(part, "wb") as fh:
                length = int(resp.headers.get("Content-Length") or 0)
                if length and length > max_bytes:
                    raise ProviderError("The file is larger than the allowed download size.")
                done = 0
                while chunk := resp.read(1024 * 256):
                    done += len(chunk)
                    if done > max_bytes:
                        raise ProviderError("The file is larger than the allowed download size.")
                    fh.write(chunk)
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as exc:
            raise ProviderError(_http_text(exc.code), details=f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("The file could not be downloaded.", details=str(exc), retryable=True) from exc
        finally:
            part.unlink(missing_ok=True)


def _http_text(code: int) -> str:
    return {
        400: "The service rejected the request (400).",
        401: "The service rejected the API key (401). Check the key environment variable.",
        403: "The service refused the request (403): the key may lack access or the quota is exhausted.",
        404: "The service endpoint was not found (404). Check the configured URL.",
        429: "The service rate limit or quota was reached (429). Try again later.",
    }.get(code, f"The service returned an error (HTTP {code}).")
