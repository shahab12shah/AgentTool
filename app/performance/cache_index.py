"""The persistent side of the media cache: ``CacheEntry`` and the SQLite index file (``<cache_root>/cache_index.sqlite``).

The index is a cache of a cache: if it cannot be read it is moved aside (``.corrupt-<ts>``, at most two kept) and a fresh one is started — the manager then
re-discovers the files it owns. Nothing here raises on a damaged database: ``open`` recovers, other calls raise ``sqlite3.Error`` for the manager to handle.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

INDEX_NAME = "cache_index.sqlite"
SCHEMA_VERSION = 1
KEEP_CORRUPT = 2
_COLUMNS = "key, category, data_type, path, inline, deps, created, last_access, size, version, status, external, metadata"

STATUS_OK = "ok"
STATUS_UNVERIFIED = "unverified"  # found by a rescan: the file exists but nothing is known about what it was built from
STATUS_STALE = "stale"  # invalidated, but the file could not (or must not) be deleted yet


@dataclass(eq=False)
class CacheEntry:
    key: str
    category: str
    data_type: str = ""
    path: Path | None = None  # absolute; None for inline-only entries
    inline: Any = None
    deps: dict[str, str] = field(default_factory=dict)
    created: float = 0.0
    last_access: float = 0.0
    size: int = 0
    version: int = 1
    status: str = STATUS_OK
    external: bool = False  # the file lives outside cache_root (registered, not owned)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "category": self.category, "data_type": self.data_type, "path": str(self.path) if self.path else None, "size": self.size,
                "created": self.created, "last_access": self.last_access, "version": self.version, "status": self.status, "external": self.external}


def rel_inside(path: str | os.PathLike, root: str | os.PathLike) -> str | None:
    """``path`` relative to ``root`` (forward slashes) when it lies lexically inside it, else None. Symlinks are not resolved."""
    p, r = os.path.abspath(path), os.path.abspath(root)
    np_, nr = os.path.normcase(p), os.path.normcase(r)
    prefix = nr.rstrip(os.sep) + os.sep
    if np_.startswith(prefix) and len(np_) > len(prefix):
        return p[len(prefix):].replace(os.sep, "/")
    return None


def path_id(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(path))


def entry_to_row(e: CacheEntry, root: Path) -> tuple:
    rel = rel_inside(e.path, root) if e.path is not None else None
    stored = rel if rel is not None else (str(e.path) if e.path is not None else None)
    return (e.key, e.category, e.data_type, stored, json.dumps(e.inline, ensure_ascii=False), json.dumps(e.deps, ensure_ascii=False), e.created, e.last_access, e.size, e.version,
            e.status, 1 if e.external else 0, json.dumps(e.metadata, ensure_ascii=False))


def row_to_entry(row: tuple, root: Path) -> CacheEntry:
    key, category, data_type, stored, inline, deps, created, last_access, size, version, status, external, metadata = row
    path = None
    if stored:
        path = Path(stored) if external else root / stored
    return CacheEntry(key, category, data_type or "", path, json.loads(inline) if inline else None, json.loads(deps) if deps else {}, float(created or 0), float(last_access or 0),
                      int(size or 0), int(version or 1), status or STATUS_OK, bool(external), json.loads(metadata) if metadata else {})


class CacheIndexDB:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / INDEX_NAME
        self.conn: sqlite3.Connection | None = None
        self.fresh = False  # a new (empty) index was created: the owner should rescan its directories
        self.recovered: str | None = None  # why the previous file was moved aside, if it was

    def open(self) -> list[CacheEntry]:
        self.root.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        try:
            return self._open_existing(existed)
        except (sqlite3.Error, ValueError, TypeError) as exc:
            self.recovered = f"{type(exc).__name__}: {exc}"
            self.close()
            self._quarantine()
            self.fresh = True
            return self._open_existing(False)

    def _open_existing(self, existed: bool) -> list[CacheEntry]:
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=5.0)
        if existed:
            check = self.conn.execute("PRAGMA quick_check").fetchone()
            if not check or check[0] != "ok":
                raise sqlite3.DatabaseError("quick_check failed")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        version = self._schema_version()
        if version != SCHEMA_VERSION:
            self._create_schema()
            self.fresh = True
            return []
        return [row_to_entry(r, self.root) for r in self.conn.execute(f"SELECT {_COLUMNS} FROM entries")]

    def _schema_version(self) -> int | None:
        try:
            r = self.conn.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()  # type: ignore[union-attr]
        except sqlite3.OperationalError:
            return None  # no such table: a new or empty file
        return int(r[0]) if r else None

    def _create_schema(self) -> None:
        c = self.conn
        assert c is not None
        c.execute("BEGIN")
        try:
            c.execute("DROP TABLE IF EXISTS entries")
            c.execute("DROP TABLE IF EXISTS meta")
            c.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
            c.execute("CREATE TABLE entries (key TEXT PRIMARY KEY, category TEXT NOT NULL, data_type TEXT, path TEXT, inline TEXT, deps TEXT, created REAL, last_access REAL, "
                      "size INTEGER, version INTEGER, status TEXT, external INTEGER, metadata TEXT)")
            c.execute("CREATE INDEX idx_entries_category ON entries(category, last_access)")
            c.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def _quarantine(self) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        target = self.root / f"{INDEX_NAME}.corrupt-{ts}"
        n = 0
        while target.exists():
            n += 1
            target = self.root / f"{INDEX_NAME}.corrupt-{ts}-{n}"
        try:
            if self.path.exists():
                os.replace(self.path, target)
        except OSError:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
        for side in ("-journal", "-wal", "-shm"):
            try:
                (self.root / (INDEX_NAME + side)).unlink(missing_ok=True)
            except OSError:
                pass
        old = sorted(self.root.glob(f"{INDEX_NAME}.corrupt-*"), key=lambda p: p.name)
        for p in old[:-KEEP_CORRUPT]:
            try:
                p.unlink()
            except OSError:
                pass

    def quarantine_and_recreate(self) -> None:
        """A runtime failure showed the file is unusable: set it aside and start an empty index (the caller rewrites its in-memory rows)."""
        self.close()
        self._quarantine()
        self._open_existing(False)

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
            self.conn = None

    # ---- writes (all raise sqlite3.Error on trouble)
    def upsert(self, e: CacheEntry) -> None:
        self.conn.execute(f"INSERT OR REPLACE INTO entries ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", entry_to_row(e, self.root))  # type: ignore[union-attr]

    def rewrite(self, entries: Iterable[CacheEntry]) -> None:
        c = self.conn
        assert c is not None
        c.execute("BEGIN")
        try:
            c.execute("DELETE FROM entries")
            c.executemany(f"INSERT OR REPLACE INTO entries ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [entry_to_row(e, self.root) for e in entries])
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def upsert_many(self, entries: Iterable[CacheEntry]) -> None:
        c = self.conn
        assert c is not None
        c.execute("BEGIN")
        try:
            c.executemany(f"INSERT OR REPLACE INTO entries ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [entry_to_row(e, self.root) for e in entries])
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def delete(self, keys: Iterable[str]) -> None:
        self.conn.executemany("DELETE FROM entries WHERE key=?", [(k,) for k in keys])  # type: ignore[union-attr]

    def touch_many(self, items: list[tuple[float, str]]) -> None:
        self.conn.executemany("UPDATE entries SET last_access=? WHERE key=?", items)  # type: ignore[union-attr]

    def set_meta(self, k: str, v: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, v))  # type: ignore[union-attr]

    def get_meta(self, k: str) -> str | None:
        r = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()  # type: ignore[union-attr]
        return r[0] if r else None
