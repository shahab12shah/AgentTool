"""Input-fingerprint skip for per-scene analysis: the result of a deterministic computation is stored in the media cache (category ``analysis``) under a key made of
*everything the computation reads*, so running it again with unchanged inputs returns the stored result (a recorded cache hit) and changing one scene's inputs changes only that
scene's key. Entries are content-addressed: they need no dependency invalidation (a changed input is simply a different key), and a user's explicit "force" bypasses the lookup.
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.serialization import from_plain, to_plain
from app.logging.logger import get_logger
from app.performance.dependencies import stable_key

_log = get_logger(__name__)
CATEGORY = "analysis"


class AnalysisCache:
    def __init__(self, cache_getter: Callable[[], Any]) -> None:
        self._cache = cache_getter

    def key(self, kind: str, *inputs: Any) -> str:
        return stable_key(f"analysis.{kind}", *inputs)

    def lookup(self, key: str, version: int = 1) -> Any | None:
        c = self._cache()
        if c is None:
            return None
        e = c.get(key, category=CATEGORY, version=version)
        return e.inline if e is not None else None

    def store(self, key: str, value: Any, *, kind: str, version: int = 1, metadata: dict[str, Any] | None = None) -> bool:
        c = self._cache()
        if c is None:
            return False
        return c.put(key, value, category=CATEGORY, data_type=kind, version=version, metadata=metadata) is not None

    def run(self, kind: str, inputs: tuple, compute: Callable[[], Any], *, cls: type | None = None, force: bool = False, version: int = 1, metadata: dict[str, Any] | None = None) -> tuple[Any, bool]:
        """``(result, hit)``. ``compute`` must be a pure function of ``inputs``; a dataclass result (``cls``) is stored as plain JSON and rebuilt on a hit."""
        key = self.key(kind, *inputs)
        if not force:
            stored = self.lookup(key, version)
            if stored is not None:
                try:
                    return (from_plain(cls, stored) if cls is not None else stored), True
                except Exception:  # noqa: BLE001  (an entry that no longer decodes is a miss)
                    _log.warning("analysis cache entry for %s could not be decoded", kind, exc_info=True)
        result = compute()
        try:
            self.store(key, to_plain(result), kind=kind, version=version, metadata=metadata)
        except Exception:  # noqa: BLE001  (not storing is only a missed speed-up)
            _log.warning("analysis result for %s could not be stored", kind, exc_info=True)
        return result, False
