"""``MediaCacheManager``: one place that knows every rebuildable cache file (thumbnails, preview frames, proxies, waveforms, analysis, render previews, temp).

* Owned files live under ``<cache_root>/<category>/...`` (``path_for`` hands out locations); files that already live elsewhere (``<project>/thumbnails``,
  ``<project>/proxies``, ``previews/frames``) are *registered* and stay where they are — the manager indexes and accounts them.
* Entries are validated on lookup (file present, version, dependency fingerprints); a stale entry is a miss. The index is SQLite, mirrored in memory so a hit
  never touches the database.
* The manager only ever deletes (a) files inside its cache_root whose real parent directory is also inside it, or (b) files registered by ``register_external``
  in a rebuildable category and not under a ``deny_roots`` path. Directories, originals, exports and anything unregistered are never touched.
* A cache problem never raises into callers: it is logged once and behaves as a miss.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from app.logging.logger import get_logger, log_event
from app.performance.cache_index import (STATUS_OK, STATUS_STALE, STATUS_UNVERIFIED, CacheEntry, CacheIndexDB, path_id, rel_inside)
from app.performance.dependencies import stable_key
from app.performance.memory_monitor import BoundedLRU
from app.performance.profiler import profiler as _default_profiler
from app.performance.resource_monitor import disk_usage
from app.performance.settings import CACHE_CATEGORIES, ResolvedLimits

_log = get_logger(__name__)

REBUILDABLE_CATEGORIES = frozenset(CACHE_CATEGORIES)
MAX_INLINE_BYTES = 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_TEMP_SUFFIXES = (".tmp", ".part", ".partial", ".temp")


def is_temp_name(name: str) -> bool:
    low = name.lower()
    return low.startswith(".") or low.endswith(_TEMP_SUFFIXES) or ".tmp." in low or ".part." in low


def tmp_for(final: Path) -> Path:
    """A sibling temp name for writing ``final`` atomically; keeps the extension (FFmpeg picks the muxer from it) and is recognised as a temp file by ``cleanup``."""
    return final.with_name(f".{final.stem}.part{final.suffix}")


@dataclass
class CleanupReport:
    evicted: int = 0
    freed_bytes: int = 0
    orphans_removed: int = 0
    temp_files_removed: int = 0
    failed: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count (protected / vetoed / denied / unsafe / category)
    by_category: dict[str, int] = field(default_factory=dict)  # category -> entries removed
    duration_s: float = 0.0

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"evicted": self.evicted, "freed_bytes": self.freed_bytes, "orphans_removed": self.orphans_removed, "temp_files_removed": self.temp_files_removed, "failed": self.failed,
                "skipped": dict(self.skipped), "by_category": dict(self.by_category), "duration_s": round(self.duration_s, 4)}


class _Store:
    """The state for one cache_root: SQLite handle plus the in-memory mirror and its secondary indexes."""

    def __init__(self, raw: Any, root: Path) -> None:
        self.raw, self.root = raw, root
        self.db: CacheIndexDB | None = None
        self.entries: dict[str, CacheEntry] = {}
        self.by_path: dict[str, str] = {}
        self.dep_index: dict[str, set[str]] = {}
        self.cat_bytes: dict[str, int] = {}
        self.cat_count: dict[str, int] = {}
        self.dirty: set[str] = set()
        self.last_maintenance = 0.0
        self.real_root = os.path.realpath(root)


def _safe(default: Callable[[], Any]):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception as exc:  # noqa: BLE001 - a cache failure must never break the caller
                self._failure(fn.__name__, exc)
                return default()

        return wrapper

    return deco


class MediaCacheManager:
    def __init__(self, cache_root: Path | str | Callable[[], Path | str | None] | None, *, limits: Callable[[], ResolvedLimits | None] | None = None, profiler=_default_profiler,
                 monitor=None, can_remove: Callable[[CacheEntry], bool] | None = None, protected: Callable[[CacheEntry], bool] | None = None,
                 deny_roots: Callable[[], Iterable[Path | str]] | Iterable[Path | str] | None = None, temp_max_age_s: float = 1800.0, maintenance_interval_s: float = 600.0,
                 clock: Callable[[], float] = time.time) -> None:
        self._root_src = cache_root
        self._limits_fn = limits
        self._prof = profiler
        self.monitor = monitor
        self._can_remove = can_remove
        self._protected_cb = protected
        self._deny = deny_roots
        self._prot_keys: set[str] = set()
        self._prot_paths: set[str] = set()
        self._temp_max_age = temp_max_age_s
        self._maint_interval = maintenance_interval_s
        self._clock = clock
        self._lock = threading.RLock()
        self._st: _Store | None = None
        self._last_t = 0.0
        self._failed_logged: set[str] = set()
        self._counters: dict[str, dict[str, int]] = {}
        self._last_cleanup: dict[str, float] = {}
        self._tiers: dict[str, tuple[BoundedLRU, float]] = {}

    # ------------------------------------------------------------------ plumbing
    def _failure(self, where: str, exc: BaseException) -> None:
        if where not in self._failed_logged:
            self._failed_logged.add(where)
            log_event(_log, "cache.error", where=where, error=type(exc).__name__, detail=str(exc)[:200])

    def _tick(self) -> float:
        t = self._clock()
        if t <= self._last_t:
            t = self._last_t + 1e-6  # strictly increasing so LRU order is exact even with a coarse clock
        self._last_t = t
        return t

    def _count(self, cat: str, what: str, n: int = 1) -> None:
        c = self._counters.setdefault(cat, {"hits": 0, "misses": 0, "evictions": 0})
        c[what] += n

    def _store(self) -> _Store | None:
        src = self._root_src
        raw = src() if callable(src) else src
        st = self._st
        if st is not None and st.raw == raw:
            return st
        if raw is None:
            return None
        with self._lock:
            if self._st is not None and self._st.raw == raw:
                return self._st
            if self._st is not None:
                self._close_store(self._st)
                self._st = None
            self._st = self._open_store(raw)
            return self._st

    def _open_store(self, raw: Any) -> _Store:
        root = Path(raw)
        st = _Store(raw, root)
        rows: list[CacheEntry] = []
        try:
            db = CacheIndexDB(root)
            rows = db.open()
            st.db = db
            if db.recovered:
                log_event(_log, "cache.index_recovered", reason=db.recovered[:200])
        except Exception as exc:  # noqa: BLE001 - e.g. read-only folder: keep working with an in-memory index
            self._failure("open_index", exc)
            st.db = None
        for e in rows:
            self._mirror_add(st, e)
        if st.db is None or st.db.fresh:
            self._rebuild(st)
        return st

    def _close_store(self, st: _Store) -> None:
        try:
            self._flush_store(st)
        finally:
            if st.db is not None:
                st.db.close()

    def close(self) -> None:
        with self._lock:
            if self._st is not None:
                self._close_store(self._st)
                self._st = None

    # ---- mirror (caller holds the lock)
    def _mirror_add(self, st: _Store, e: CacheEntry) -> None:
        self._mirror_remove(st, e.key)
        if e.path is not None:
            other = st.by_path.get(path_id(e.path))
            if other is not None and other != e.key:
                self._mirror_remove(st, other, persist_in=st)
            st.by_path[path_id(e.path)] = e.key
        st.entries[e.key] = e
        for d in e.deps:
            st.dep_index.setdefault(d, set()).add(e.key)
        st.cat_bytes[e.category] = st.cat_bytes.get(e.category, 0) + e.size
        st.cat_count[e.category] = st.cat_count.get(e.category, 0) + 1

    def _mirror_remove(self, st: _Store, key: str, persist_in: _Store | None = None) -> CacheEntry | None:
        e = st.entries.pop(key, None)
        if e is None:
            return None
        if e.path is not None and st.by_path.get(path_id(e.path)) == key:
            del st.by_path[path_id(e.path)]
        for d in e.deps:
            s = st.dep_index.get(d)
            if s is not None:
                s.discard(key)
                if not s:
                    del st.dep_index[d]
        st.cat_bytes[e.category] = st.cat_bytes.get(e.category, 0) - e.size
        st.cat_count[e.category] = st.cat_count.get(e.category, 0) - 1
        st.dirty.discard(key)
        if persist_in is not None:
            self._db(st, lambda db: db.delete([key]))
        return e

    def _db(self, st: _Store, op: Callable[[CacheIndexDB], None]) -> None:
        if st.db is None or st.db.conn is None:
            return
        try:
            op(st.db)
        except sqlite3.Error as exc:
            self._recover(st, exc)

    def _recover(self, st: _Store, exc: BaseException) -> None:
        self._failure("index_write", exc)
        try:
            st.db.quarantine_and_recreate()  # type: ignore[union-attr]
            st.db.rewrite(st.entries.values())  # type: ignore[union-attr]
            log_event(_log, "cache.index_recovered", reason=type(exc).__name__)
        except Exception as exc2:  # noqa: BLE001
            self._failure("index_recover", exc2)
            if st.db is not None:
                st.db.close()
            st.db = None

    def _flush_store(self, st: _Store) -> None:
        with self._lock:
            if not st.dirty:
                return
            items = [(st.entries[k].last_access, k) for k in st.dirty if k in st.entries]
            st.dirty.clear()
        if items:
            with self._lock:
                self._db(st, lambda db: db.touch_many(items))

    def flush(self) -> None:
        st = self._st
        if st is not None:
            self._flush_store(st)

    # ---- rescan
    def _owned_dirs(self, st: _Store) -> list[Path]:
        out = []
        for c in CACHE_CATEGORIES:
            d = st.root / c
            if d.is_dir() and not d.is_symlink():
                out.append(d)
        return out

    def _rebuild(self, st: _Store) -> int:
        added: list[CacheEntry] = []
        now = self._clock()
        for d in self._owned_dirs(st):
            cat = d.name
            for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
                for name in filenames:
                    if is_temp_name(name):
                        continue
                    p = Path(dirpath) / name
                    try:
                        size = os.stat(p).st_size
                    except OSError:
                        continue
                    if size <= 0:
                        continue
                    with self._lock:
                        if path_id(p) in st.by_path:
                            continue
                        stem = p.stem
                        key = stem if _HEX40.match(stem) else "file:" + (rel_inside(p, st.root) or name)
                        if key in st.entries:
                            continue
                        try:
                            mt = os.stat(p).st_mtime
                        except OSError:
                            mt = now
                        e = CacheEntry(key, cat, "", p, None, {}, mt, mt, size, 1, STATUS_UNVERIFIED, False)
                        self._mirror_add(st, e)
                        added.append(e)
        if added:
            with self._lock:
                self._db(st, lambda db: db.upsert_many(added))
        return len(added)

    @_safe(lambda: 0)
    def rebuild_index(self) -> int:
        """Rescan the directories the manager owns and index files it does not know (as ``unverified``: a miss until the caller puts them again)."""
        st = self._store()
        return self._rebuild(st) if st is not None else 0

    # ------------------------------------------------------------------ locations
    @property
    def root(self) -> Path | None:
        st = self._store()
        return st.root if st else None

    def path_for(self, category: str, key: str, suffix: str = "") -> Path | None:
        """A file location inside the cache for ``key`` (parent directory created). Write it (via ``tmp_for`` + rename), then ``put(key, path=...)``."""
        st = self._store()
        if st is None:
            return None
        cat = category if _SAFE_NAME.match(category) else re.sub(r"[^A-Za-z0-9_-]", "_", category)[:40] or "misc"
        name = key if _SAFE_NAME.match(key) and not key.startswith(".") else stable_key("name", key)
        p = st.root / cat / name[:2] / f"{name}{suffix}"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._failure("path_for", exc)
            return None
        return p

    # ------------------------------------------------------------------ lookups
    def _miss(self, category: str | None) -> None:
        cat = category or "unknown"
        with self._lock:
            self._count(cat, "misses")
        self._prof.cache_miss(cat)

    def get(self, key: str, *, deps: dict[str, str] | None = None, category: str | None = None, version: int | None = None) -> CacheEntry | None:
        try:
            st = self._store()
            if st is None:
                self._miss(category)
                return None
            with self._lock:
                e = st.entries.get(key)
            if e is None or e.status != STATUS_OK or (version is not None and e.version != version):
                self._miss(category or (e.category if e else None))
                return None
            if e.path is not None:
                try:
                    size = os.stat(e.path).st_size
                except OSError:
                    size = -1
                if size <= 0:
                    self._drop_entries(st, [e], delete_files=False)
                    self._miss(e.category)
                    return None
            if deps is not None and not self._deps_match(e, deps):
                self._discard(st, [e])
                self._miss(e.category)
                return None
            with self._lock:
                if st.entries.get(key) is e:
                    e.last_access = self._tick()
                    st.dirty.add(key)
                    self._count(e.category, "hits")
            self._prof.cache_hit(e.category)
            return e
        except Exception as exc:  # noqa: BLE001
            self._failure("get", exc)
            self._miss(category)
            return None

    @staticmethod
    def _deps_match(e: CacheEntry, current: dict[str, str]) -> bool:
        for d, fp in current.items():
            if e.deps.get(d) != str(fp):
                return False  # changed, or the entry was built without knowing about this dependency
        for d, fp in e.deps.items():
            if d in current and current[d] != fp:
                return False
        return True

    def peek(self, key: str) -> CacheEntry | None:
        """The entry as indexed, without validation, statistics or a last-access update."""
        st = self._store()
        return st.entries.get(key) if st else None

    def has(self, key: str) -> bool:
        e = self.peek(key)
        return e is not None and e.status == STATUS_OK

    def entries(self, category: str | None = None) -> list[CacheEntry]:
        st = self._store()
        if st is None:
            return []
        with self._lock:
            return [e for e in st.entries.values() if category is None or e.category == category]

    # ------------------------------------------------------------------ writes
    @_safe(lambda: None)
    def put(self, key: str, value: Any = None, *, category: str, data_type: str = "", path: Path | str | None = None, deps: dict[str, str] | None = None, version: int = 1,
            metadata: dict[str, Any] | None = None, size: int | None = None) -> CacheEntry | None:
        st = self._store()
        if st is None or not key or not category:
            return None
        inline_size = 0
        if value is not None:
            inline_size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))  # raises (-> None) if not JSON-able
            if inline_size > MAX_INLINE_BYTES:
                log_event(_log, "cache.inline_too_large", category=category, size=inline_size)
                return None
        p: Path | None = None
        file_size = 0
        if path is not None:
            p = Path(os.path.abspath(path))
            try:
                if not p.is_file():
                    return None
                file_size = os.stat(p).st_size
            except OSError:
                return None
            if file_size <= 0:
                return None
            if rel_inside(p, st.root) is None and self._is_denied(p):
                return None
        now = self._tick()
        e = CacheEntry(key, category, data_type, p, value, {str(d): str(f) for d, f in (deps or {}).items()}, now, now, file_size + inline_size if p else (size if size is not None else inline_size),
                       int(version), STATUS_OK, p is not None and rel_inside(p, st.root) is None, dict(metadata or {}))
        old_to_delete: CacheEntry | None = None
        with self._lock:
            old = st.entries.get(key)
            if old is not None and old.path is not None and (p is None or path_id(old.path) != path_id(p)) and not old.external:
                old_to_delete = old
            self._mirror_add(st, e)
            self._db(st, lambda db: db.upsert(e))
            flush = len(st.dirty) > 256
        if old_to_delete is not None and self._blocked(old_to_delete) is None:
            self._delete_file(st, old_to_delete)
        if flush:
            self._flush_store(st)
        return e

    @_safe(lambda: None)
    def register_external(self, category: str, path: Path | str, key: str | None = None, deps: dict[str, str] | None = None, version: int = 1) -> CacheEntry | None:
        """Index (and account) a file that lives outside cache_root. It stays where it is; only a rebuildable category makes it removable."""
        st = self._store()
        if st is None:
            return None
        p = Path(os.path.abspath(path))
        if not p.is_file():
            return None
        if key is None:
            hit = st.by_path.get(path_id(p))
            key = hit if hit is not None and st.entries[hit].category == category else stable_key(category, "ext", p)
        existing = st.entries.get(key)
        if existing is not None and existing.path is not None and path_id(existing.path) == path_id(p) and existing.status == STATUS_OK and deps is None:
            try:
                size = os.stat(p).st_size
            except OSError:
                return None
            with self._lock:
                st.cat_bytes[existing.category] = st.cat_bytes.get(existing.category, 0) + size - existing.size
                existing.size = size
                existing.last_access = self._tick()
                st.dirty.add(key)
            return existing
        return self.put(key, category=category, data_type="external", path=p, deps=deps, version=version)

    # ------------------------------------------------------------------ invalidation
    def _discard(self, st: _Store, entries: list[CacheEntry], delete_files: bool = True) -> int:
        """Forget entries and delete their files where allowed; a file that must stay (in use) keeps a ``stale`` row so it is still accounted and evicted first."""
        return self._drop_entries(st, entries, delete_files=delete_files, keep_stale_if_blocked=True)

    def _drop_entries(self, st: _Store, entries: list[CacheEntry], *, delete_files: bool, keep_stale_if_blocked: bool = False) -> int:
        n = 0
        stale: list[CacheEntry] = []
        gone: list[str] = []
        for e in entries:
            if delete_files and e.path is not None:
                if self._blocked(e) is not None:
                    if keep_stale_if_blocked:
                        e.status = STATUS_STALE
                        stale.append(e)
                        n += 1
                    continue
                outcome = self._delete_file(st, e)
                if outcome == "failed":
                    if keep_stale_if_blocked:
                        e.status = STATUS_STALE
                        stale.append(e)
                        n += 1
                    continue
            with self._lock:
                if st.entries.get(e.key) is e:
                    self._mirror_remove(st, e.key)
                    gone.append(e.key)
            n += 1
        with self._lock:
            if gone:
                self._db(st, lambda db: db.delete(gone))
            live_stale = [e for e in stale if st.entries.get(e.key) is e]
            if live_stale:
                self._db(st, lambda db: db.upsert_many(live_stale))
        self._forget_memory(gone + [e.key for e in stale], {e.category for e in entries})
        return n

    def _forget_memory(self, keys: list[str], categories: set[str]) -> None:
        ks = set(keys)
        for c in categories:
            t = self._tiers.get(c)
            if t is not None and ks:
                t[0].discard_where(lambda k: k in ks)

    @_safe(lambda: False)
    def invalidate(self, key: str, *, delete_file: bool = True) -> bool:
        st = self._store()
        e = st.entries.get(key) if st else None
        if st is None or e is None:
            return False
        self._discard(st, [e], delete_files=delete_file)
        return True

    @_safe(lambda: 0)
    def invalidate_by_dependency(self, dep_id: str) -> int:
        st = self._store()
        if st is None:
            return 0
        with self._lock:
            es = [st.entries[k] for k in st.dep_index.get(dep_id, ()) if k in st.entries]
        return self._discard(st, es)

    @_safe(lambda: 0)
    def invalidate_by_dependency_prefix(self, prefix: str) -> int:
        st = self._store()
        if st is None:
            return 0
        with self._lock:
            keys = {k for d, ks in st.dep_index.items() if d.startswith(prefix) for k in ks}
            es = [st.entries[k] for k in keys if k in st.entries]
        return self._discard(st, es)

    @_safe(lambda: 0)
    def invalidate_category(self, category: str) -> int:
        st = self._store()
        if st is None:
            return 0
        with self._lock:
            es = [e for e in st.entries.values() if e.category == category]
        return self._discard(st, es)

    # ------------------------------------------------------------------ safety
    def set_protected(self, callback: Callable[[CacheEntry], bool] | None) -> None:
        self._protected_cb = callback

    def set_can_remove(self, callback: Callable[[CacheEntry], bool] | None) -> None:
        self._can_remove = callback

    def protect(self, key_or_path: str | os.PathLike) -> None:
        self._prot_keys.add(str(key_or_path))
        self._prot_paths.add(path_id(key_or_path))

    def unprotect(self, key_or_path: str | os.PathLike) -> None:
        self._prot_keys.discard(str(key_or_path))
        self._prot_paths.discard(path_id(key_or_path))

    def _deny_list(self) -> list[Path]:
        d = self._deny() if callable(self._deny) else self._deny
        return [Path(x) for x in (d or [])]

    def _is_denied(self, p: Path) -> bool:
        deny = self._deny_list()
        if not deny:
            return False
        real = os.path.realpath(p)
        for r in deny:
            if rel_inside(p, r) is not None or rel_inside(real, os.path.realpath(r)) is not None:
                return True
        return False

    def _blocked(self, e: CacheEntry) -> str | None:
        """Why this entry must not be removed (None = removable). External callbacks run without the manager lock held."""
        if e.category not in REBUILDABLE_CATEGORIES:
            return "category"
        if e.path is not None:
            if e.key in self._prot_keys or path_id(e.path) in self._prot_paths:
                return "protected"
            if e.external and self._is_denied(e.path):
                return "denied"
        elif e.key in self._prot_keys:
            return "protected"
        if self._protected_cb is not None:
            try:
                if self._protected_cb(e):
                    return "protected"
            except Exception:  # noqa: BLE001 - fail safe: keep the file
                return "protected"
        if self._can_remove is not None:
            try:
                if not self._can_remove(e):
                    return "vetoed"
            except Exception:  # noqa: BLE001
                return "vetoed"
        return None

    def _delete_file(self, st: _Store, e: CacheEntry) -> str:
        """'deleted' (or already gone) / 'unsafe' (refused: path not provably ours) / 'failed' (OS error, try again later)."""
        p = e.path
        if p is None:
            return "deleted"
        try:
            if not os.path.lexists(p):
                return "deleted"
            if os.path.isdir(p) and not os.path.islink(p):
                return "unsafe"
            real_parent = os.path.realpath(p.parent)
            if not e.external:
                if rel_inside(real_parent, st.real_root) is None and os.path.normcase(real_parent) != os.path.normcase(st.real_root):
                    return "unsafe"
            elif e.category not in REBUILDABLE_CATEGORIES or self._is_denied(p):
                return "unsafe"
            os.unlink(p)  # for a symlink this removes the link, never its target
            return "deleted"
        except FileNotFoundError:
            return "deleted"  # someone else got there first
        except OSError as exc:
            self._failure("delete_file", exc)
            return "failed"

    # ------------------------------------------------------------------ cleanup
    def _limits(self) -> ResolvedLimits | None:
        if self._limits_fn is None:
            return None
        try:
            return self._limits_fn()
        except Exception as exc:  # noqa: BLE001
            self._failure("limits", exc)
            return None

    @staticmethod
    def _lru_key(e: CacheEntry) -> tuple:
        return (e.status == STATUS_OK, e.last_access)  # stale / unverified rows are the first to go

    def _rebuildable_bytes(self, st: _Store) -> int:
        return sum(b for c, b in st.cat_bytes.items() if c in REBUILDABLE_CATEGORIES)

    def _evict(self, st: _Store, e: CacheEntry, rep: CleanupReport) -> bool:
        why = self._blocked(e)
        if why is not None:
            rep.skip(why)
            return False
        outcome = self._delete_file(st, e)
        if outcome == "failed":
            rep.failed += 1
            return False
        if outcome == "unsafe":
            rep.skip("unsafe")  # forget the row, never the file
        with self._lock:
            if st.entries.get(e.key) is not e:
                return False
            self._mirror_remove(st, e.key)
            self._db(st, lambda db: db.delete([e.key]))
        if outcome == "deleted":
            rep.evicted += 1
            rep.freed_bytes += e.size
            rep.by_category[e.category] = rep.by_category.get(e.category, 0) + 1
            with self._lock:
                self._count(e.category, "evictions")
            self._prof.incr(f"cache.{e.category}.evicted")
        self._forget_memory([e.key], {e.category})
        return True

    def _sweep_orphans(self, st: _Store, rep: CleanupReport) -> None:
        with self._lock:
            with_files = [e for e in st.entries.values() if e.path is not None]
        gone: list[CacheEntry] = []
        for e in with_files:
            try:
                ok = os.stat(e.path).st_size > 0
            except OSError:
                ok = False
            if not ok:
                gone.append(e)
        with self._lock:
            keys = []
            for e in gone:
                if st.entries.get(e.key) is e:
                    self._mirror_remove(st, e.key)
                    keys.append(e.key)
            if keys:
                self._db(st, lambda db: db.delete(keys))
        rep.orphans_removed += len(gone)
        self._forget_memory([e.key for e in gone], {e.category for e in gone})

    def _sweep_temp(self, st: _Store, rep: CleanupReport) -> None:
        now = time.time()
        for d in self._owned_dirs(st):
            temporary = d.name == "temporary"
            for dirpath, dirnames, filenames in os.walk(d, followlinks=False, topdown=False):
                real = os.path.realpath(dirpath)
                if rel_inside(real, st.real_root) is None:
                    continue
                for name in filenames:
                    p = Path(dirpath) / name
                    with self._lock:
                        if path_id(p) in st.by_path:
                            continue
                    if not (is_temp_name(name) or temporary):
                        continue
                    try:
                        if now - os.lstat(p).st_mtime < self._temp_max_age:
                            continue
                        os.unlink(p)
                        rep.temp_files_removed += 1
                    except OSError:
                        continue
                if Path(dirpath) != d:
                    try:
                        if now - os.stat(dirpath).st_mtime >= self._temp_max_age:
                            os.rmdir(dirpath)  # only succeeds when empty
                    except OSError:
                        pass

    @_safe(lambda: CleanupReport())
    def cleanup(self, force: bool = False) -> CleanupReport:
        """Evict least-recently-used entries over the per-category and total limits. The (slower) orphan and temp-file sweeps run when ``force`` or every ``maintenance_interval_s``."""
        rep = CleanupReport()
        st = self._store()
        if st is None:
            return rep
        t0 = time.perf_counter()
        with self._prof.timer("cache.cleanup"):
            self._flush_store(st)
            now = self._clock()
            if force or now - st.last_maintenance >= self._maint_interval:
                st.last_maintenance = now
                self._sweep_orphans(st, rep)
                self._sweep_temp(st, rep)
            lim = self._limits()
            if lim is not None:
                for cat in list(st.cat_bytes):
                    limit = lim.category_bytes.get(cat)
                    if cat not in REBUILDABLE_CATEGORIES or limit is None or st.cat_bytes.get(cat, 0) <= limit:
                        continue
                    with self._lock:
                        cands = sorted((e for e in st.entries.values() if e.category == cat), key=self._lru_key)
                    for e in cands:
                        if st.cat_bytes.get(cat, 0) <= limit:
                            break
                        self._evict(st, e, rep)
                if self._rebuildable_bytes(st) > lim.cache_total_bytes:
                    with self._lock:
                        cands = sorted((e for e in st.entries.values() if e.category in REBUILDABLE_CATEGORIES), key=self._lru_key)
                    for e in cands:
                        if self._rebuildable_bytes(st) <= lim.cache_total_bytes:
                            break
                        self._evict(st, e, rep)
            stamp = self._clock()
            with self._lock:
                for cat in set(st.cat_bytes) | set(CACHE_CATEGORIES):
                    self._last_cleanup[cat] = stamp
                self._db(st, lambda db: db.set_meta("last_cleanup", str(stamp)))
        rep.duration_s = time.perf_counter() - t0
        if rep.evicted or rep.orphans_removed or rep.temp_files_removed:
            log_event(_log, "cache.cleanup", evicted=rep.evicted, freed_bytes=rep.freed_bytes, orphans=rep.orphans_removed, temp_files=rep.temp_files_removed)
        return rep

    @_safe(lambda: CleanupReport())
    def clear_rebuildable_cache(self, categories: Iterable[str] | None = None) -> CleanupReport:
        """Delete rebuildable cache files (all categories, or the given ones) except protected / vetoed ones. Originals and exports are never in scope."""
        rep = CleanupReport()
        st = self._store()
        if st is None:
            return rep
        t0 = time.perf_counter()
        wanted = REBUILDABLE_CATEGORIES if categories is None else REBUILDABLE_CATEGORIES & set(categories)
        with self._lock:
            cands = sorted((e for e in st.entries.values() if e.category in wanted), key=self._lru_key)
        for e in cands:
            self._evict(st, e, rep)
        rep.duration_s = time.perf_counter() - t0
        log_event(_log, "cache.cleared", evicted=rep.evicted, freed_bytes=rep.freed_bytes, categories=sorted(wanted))
        return rep

    @_safe(lambda: [])
    def eligible_for_removal(self, categories: Iterable[str] | None = None) -> list[CacheEntry]:
        """Entries ``clear_rebuildable_cache`` would delete now, least recently used first."""
        st = self._store()
        if st is None:
            return []
        wanted = REBUILDABLE_CATEGORIES if categories is None else REBUILDABLE_CATEGORIES & set(categories)
        with self._lock:
            cands = sorted((e for e in st.entries.values() if e.category in wanted), key=self._lru_key)
        return [e for e in cands if self._blocked(e) is None]

    # ------------------------------------------------------------------ stats
    @_safe(lambda: {"categories": {}, "total_bytes": 0, "total_limit_bytes": None, "disk_free_bytes": None, "eligible_bytes": 0, "last_cleanup": None})
    def get_cache_stats(self) -> dict[str, Any]:
        st = self._store()
        lim = self._limits()
        cats: dict[str, dict[str, Any]] = {}
        with self._lock:
            names = list(CACHE_CATEGORIES) + [c for c in (set(st.cat_bytes) if st else set()) | set(self._counters) if c not in CACHE_CATEGORIES]
            for c in names:
                cnt = self._counters.get(c, {})
                cats[c] = {"entries": st.cat_count.get(c, 0) if st else 0, "bytes": st.cat_bytes.get(c, 0) if st else 0, "limit_bytes": lim.category_bytes.get(c) if lim else None,
                           "hits": cnt.get("hits", 0), "misses": cnt.get("misses", 0), "evictions": cnt.get("evictions", 0), "last_cleanup": self._last_cleanup.get(c)}
            total = self._rebuildable_bytes(st) if st else 0
            unmanaged = sum(b for c, b in st.cat_bytes.items() if c not in REBUILDABLE_CATEGORIES) if st else 0
        eligible = sum(e.size for e in self.eligible_for_removal())
        free = disk_usage(st.root)[0] if st else None
        out = {"categories": cats, "total_bytes": total, "unmanaged_bytes": unmanaged, "total_limit_bytes": lim.cache_total_bytes if lim else None, "disk_free_bytes": free,
               "eligible_bytes": eligible, "last_cleanup": max(self._last_cleanup.values()) if self._last_cleanup else None,
               "memory": {c: t.stats() for c, (t, _s) in self._tiers.items()}}
        if self.monitor is not None:
            try:
                out["rss_growth_bytes"] = self.monitor.growth_bytes()
            except Exception:  # noqa: BLE001
                pass
        return out

    # ------------------------------------------------------------------ memory tier
    def memory_tier(self, category: str, *, weigher: Callable[[Any], int] | None = None, on_evict: Callable[[Any, Any], None] | None = None, share: float = 0.25) -> BoundedLRU:
        """The in-memory LRU for decoded things of a category (pixmaps, frames), bounded by ``ResolvedLimits.memory_cache_bytes * share``. Same tier on repeated calls."""
        with self._lock:
            hit = self._tiers.get(category)
            if hit is not None:
                return hit[0]
            tier: BoundedLRU = BoundedLRU(0, self._tier_bytes(share), weigher or _default_weigher, on_evict)
            self._tiers[category] = (tier, share)
            return tier

    def _tier_bytes(self, share: float) -> int:
        lim = self._limits()
        return max(1, int(lim.memory_cache_bytes * share)) if lim else 64 * 1024 * 1024

    def refresh_limits(self) -> None:
        """Re-apply the current limits to the memory tiers (call after the performance settings changed)."""
        with self._lock:
            tiers = list(self._tiers.values())
        for tier, share in tiers:
            tier.resize(max_weight=self._tier_bytes(share))

    def trim_memory(self, factor: float = 0.5) -> None:
        """Shrink every memory tier to ``factor`` of its current weight (e.g. under memory pressure); the tiers keep their configured ceiling."""
        with self._lock:
            tiers = list(self._tiers.values())
        for tier, _share in tiers:
            saved = tier.max_weight
            tier.resize(max_weight=max(1, int(tier.weight * max(0.0, min(1.0, factor)))))
            tier.resize(max_weight=saved)

    def clear_memory(self) -> None:
        with self._lock:
            tiers = list(self._tiers.values())
        for tier, _share in tiers:
            tier.clear()


def _default_weigher(v: Any) -> int:
    try:
        return len(v)  # bytes / bytearray / memoryview
    except TypeError:
        n = getattr(v, "nbytes", None) or getattr(v, "sizeInBytes", None)
        try:
            if not n and callable(getattr(v, "width", None)) and callable(getattr(v, "height", None)):
                return int(v.width() * v.height() * 4)
            return int(n() if callable(n) else n) if n else 1024
        except Exception:  # noqa: BLE001
            return 1024
