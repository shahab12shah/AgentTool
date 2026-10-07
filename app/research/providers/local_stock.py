"""Local stock library: a folder of images/videos the user owns or has licensed.

Real and fully testable offline. Metadata comes from, in order: ``catalog.json`` in the folder
(``{"relative/file.mp4": {"title", "description", "tags", "license": {...}}}``), a ``<file>.json`` sidecar,
then the file name and its folder names. Searching matches query words against that text.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from app.core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.exceptions import AcquisitionError, MediaError
from app.core.textutil import STOPWORDS, stem, tokenize
from app.media.asset import SourceType
from app.media.media_probe import MediaProber, locate_binary, run_process
from app.research.models import Acquisition, Candidate, LicenseInfo, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate


class LocalStockProvider(SourceProvider):
    name = "local_stock"
    label = "Local stock folder"
    source_types = (SourceType.STOCK_IMAGE, SourceType.STOCK_VIDEO)
    verified_live = True  # no remote service involved: exercised for real in tests

    def __init__(self, folder: Callable[[], str] = lambda: "", ffmpeg_path: Callable[[], str] = lambda: "",
                 prober: MediaProber | None = None, **kw) -> None:
        super().__init__(**kw)
        self._folder, self._ffmpeg, self._prober = folder, ffmpeg_path, prober or MediaProber()
        self._meta_cache: dict[tuple[str, float], dict] = {}

    def _root(self) -> Path | None:
        f = self._folder().strip()
        return Path(f).expanduser() if f else None

    def is_available(self) -> tuple[bool, str]:
        root = self._root()
        if root is None:
            return False, "No local stock folder is configured (Settings → Visual research)."
        if not root.is_dir():
            return False, f"The local stock folder does not exist: {root}"
        return True, ""

    def config_key(self) -> str:
        root = self._root()
        return f"local:{root}:{root.stat().st_mtime_ns if root and root.exists() else 0}"

    # ------------------------------------------------------------------
    def _catalog(self, root: Path) -> dict:
        p = root / "catalog.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _entry(self, root: Path, file: Path, catalog: dict) -> dict:
        rel = file.relative_to(root).as_posix()
        entry = dict(catalog.get(rel) or {})
        side = file.with_suffix(file.suffix + ".json")
        if side.is_file():
            try:
                entry = {**json.loads(side.read_text(encoding="utf-8")), **entry}
            except (OSError, ValueError):
                pass
        words = re.split(r"[\W_]+", " ".join(list(file.relative_to(root).parts[:-1]) + [file.stem]))
        entry.setdefault("title", " ".join(w for w in re.split(r"[\W_]+", file.stem) if w))
        entry["_path_words"] = [w for w in words if w]
        return entry

    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext) -> list[Candidate]:
        root = self._root()
        if root is None or not root.is_dir():
            return []
        want_video = source_type is SourceType.STOCK_VIDEO
        exts = VIDEO_EXTENSIONS if want_video else IMAGE_EXTENSIONS
        q = {stem(t.norm) for t in tokenize(query.text) if t.norm not in STOPWORDS}
        if not q:
            return []
        catalog = self._catalog(root)
        scored: list[tuple[float, Path, dict]] = []
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in exts:
                continue
            entry = self._entry(root, file, catalog)
            text = " ".join([str(entry.get("title", "")), str(entry.get("description", "")), " ".join(entry.get("tags", []) or []),
                             " ".join(entry["_path_words"])])
            terms = {stem(t.norm) for t in tokenize(text) if t.norm not in STOPWORDS}
            overlap = len(q & terms)
            if overlap:
                scored.append((overlap / len(q), file, entry))
        scored.sort(key=lambda s: (-s[0], str(s[1])))
        out = []
        for _s, file, entry in scored[:limit]:
            c = blank_candidate(query, source_type, "VIDEO" if want_video else "IMAGE", self.name)
            info = self._probe(file)
            lic = entry.get("license") or {}
            c.title = str(entry.get("title", file.stem))
            c.description = str(entry.get("description", ""))
            c.tags = [str(t) for t in (entry.get("tags") or [])] + [w for w in entry["_path_words"] if w.lower() not in c.title.lower()]
            c.duration, c.width, c.height = info.get("duration"), info.get("width"), info.get("height")
            c.provider_id = file.relative_to(root).as_posix()
            c.source_reference = str(file)
            c.local_path = str(file)
            c.acquisition = Acquisition.LOCAL
            if lic:
                c.license = LicenseInfo(lic.get("name"), lic.get("url"), lic.get("attribution"), "PROVIDER_STATED", False)
            out.append(c)
        return out

    def _probe(self, file: Path) -> dict:
        key = (str(file), file.stat().st_mtime)
        if key not in self._meta_cache:
            try:
                i = self._prober.probe(file)
                self._meta_cache[key] = {"duration": i.duration, "width": i.width, "height": i.height}
            except MediaError:
                self._meta_cache[key] = {}
        return self._meta_cache[key]

    def fetch_thumbnail(self, candidate: Candidate, dest: Path, http) -> bool:
        src = Path(candidate.local_path)
        if not src.is_file():
            return False
        cmd = [locate_binary("ffmpeg", self._ffmpeg()), "-y", "-v", "error"]
        if candidate.kind == "VIDEO":
            cmd += ["-ss", f"{min(1.0, (candidate.duration or 0) * 0.1):.2f}"]
        cmd += ["-i", str(src), "-frames:v", "1", "-vf", "scale=320:-2", "-update", "1", str(dest)]
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = run_process(cmd, timeout=60)
        return r.returncode == 0 and dest.is_file()

    def acquire(self, candidate: Candidate, dest_dir: Path, ctx: SearchContext) -> Path:
        src = Path(candidate.local_path)
        if not src.is_file():
            raise AcquisitionError(f"The stock file is no longer available: {src.name}")
        return src  # the importer copies it into the project
