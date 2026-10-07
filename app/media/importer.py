"""Media import: the slow, thread-safe half (hash, probe, copy).

``MediaImporter.prepare`` never touches the project model, so it can run on a worker
thread. Registering the resulting asset is done afterwards on the main thread
(see ``services/media_service.py``).
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.core.exceptions import JobCancelled, MediaError
from app.logging.logger import get_logger, log_event
from app.media.asset import Asset, AssetType, SourceType
from app.media.media_probe import MediaProber
from app.media.metadata import MediaInfo, asset_type_for
from app.storage.paths import ProjectPaths

_log = get_logger(__name__)
CHUNK = 1024 * 1024
LINK_COPY = "copy"
LINK_REFERENCE = "reference"

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


@dataclass
class PreparedMedia:
    source: Path
    asset_type: AssetType
    content_hash: str
    size_bytes: int
    info: MediaInfo | None  # None when ``duplicate_of`` is set (nothing was probed)
    stored_path: str | None  # path to store in the asset (relative for copies)
    link_mode: str
    duplicate_of: str | None = None
    copied: bool = False  # True if this call created a new file inside the project


def sha256_file(path: Path, progress: ProgressFn | None = None, should_cancel: CancelFn | None = None) -> str:
    total = max(path.stat().st_size, 1)
    done = 0
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            if should_cancel and should_cancel():
                raise JobCancelled()
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(done / total, "Checking file")
    return digest.hexdigest()


def safe_filename(name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = re.sub(r"[^\w\-. ()]+", "_", stem, flags=re.UNICODE).strip(" .") or "media"
    return f"{stem}.{ext.lower()}" if ext else stem


def unique_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


class MediaImporter:
    def __init__(self, prober: MediaProber) -> None:
        self._prober = prober

    def prepare(
        self,
        project_root: Path,
        source: Path,
        *,
        known_hashes: dict[str, str],
        link_mode: str = LINK_COPY,
        progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
    ) -> PreparedMedia:
        source = Path(source)
        asset_type = asset_type_for(source)  # raises UnsupportedMediaError
        if not source.is_file():
            raise MediaError(f"“{source.name}” does not exist or is not a file.")
        try:
            size = source.stat().st_size
            if size == 0:
                raise MediaError(f"“{source.name}” is empty.")
            content_hash = sha256_file(
                source, (lambda f, m: progress(f * 0.3, m)) if progress else None, should_cancel
            )
        except PermissionError as exc:
            raise MediaError(f"No permission to read “{source.name}”.", details=str(exc)) from exc
        except OSError as exc:
            raise MediaError(f"“{source.name}” could not be read: {exc.strerror or exc}", details=str(exc)) from exc

        if content_hash in known_hashes:
            log_event(_log, "media.duplicate", source=str(source), existing=known_hashes[content_hash])
            return PreparedMedia(
                source, asset_type, content_hash, size, None, None, link_mode, duplicate_of=known_hashes[content_hash]
            )

        if progress:
            progress(0.35, "Reading metadata")
        info = self._prober.probe(source)  # fail early, before anything is copied
        if should_cancel and should_cancel():
            raise JobCancelled()

        if link_mode == LINK_REFERENCE:
            return PreparedMedia(source, asset_type, content_hash, size, info, str(source.resolve()), link_mode)

        paths = ProjectPaths(project_root)
        dest_dir = paths.media_subdir(asset_type.value)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_destination(dest_dir, safe_filename(source.name))
        self._copy(source, dest, size, progress, should_cancel)
        rel = dest.relative_to(project_root).as_posix()
        return PreparedMedia(source, asset_type, content_hash, size, info, rel, link_mode, copied=True)

    @staticmethod
    def _copy(src: Path, dest: Path, size: int, progress: ProgressFn | None, should_cancel: CancelFn | None) -> None:
        part = dest.with_name(dest.name + ".part")
        copied = 0
        try:
            with open(src, "rb") as fin, open(part, "wb") as fout:
                while chunk := fin.read(CHUNK):
                    if should_cancel and should_cancel():
                        raise JobCancelled()
                    fout.write(chunk)
                    copied += len(chunk)
                    if progress:
                        progress(0.4 + 0.6 * copied / max(size, 1), "Copying into project")
                fout.flush()
                os.fsync(fout.fileno())
            shutil.copystat(src, part, follow_symlinks=True)
            os.replace(part, dest)
        except BaseException as exc:
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, OSError):
                if exc.errno == errno.ENOSPC:
                    raise MediaError("The disk is full — the file could not be copied into the project.", details=str(exc)) from exc
                raise MediaError(f"Copying “{src.name}” failed: {exc.strerror or exc}", details=str(exc)) from exc
            raise


def build_asset(asset_id: str, prepared: PreparedMedia) -> Asset:
    """Create the ``Asset`` record for prepared media (call on the main thread)."""
    info = prepared.info
    assert info is not None and prepared.stored_path is not None
    return Asset(
        id=asset_id,
        type=prepared.asset_type,
        source_type=SourceType.USER_MEDIA,
        path=prepared.stored_path,
        name=prepared.source.name,
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        codec=info.codec,
        has_audio=info.has_audio,
        audio_codec=info.audio_codec,
        sample_rate=info.sample_rate,
        channels=info.channels,
        size_bytes=prepared.size_bytes,
        content_hash=prepared.content_hash,
        link_mode=prepared.link_mode,
        imported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
