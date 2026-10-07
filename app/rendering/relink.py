"""MediaRelinkService: find and replace missing media — only with a person's confirmation when the match is not certain."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.constants import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.exceptions import AppError, MediaError
from app.media.asset import Asset, AssetType
from app.media.importer import sha256_file
from app.project.project import Project
from app.rendering.commands import RelinkAssetCommand
from app.rendering.probe import MediaProbeService

HIGH_CONFIDENCE = 90.0
EXT_OF = {AssetType.VIDEO: VIDEO_EXTENSIONS, AssetType.IMAGE: IMAGE_EXTENSIONS, AssetType.AUDIO: AUDIO_EXTENSIONS}


@dataclass
class RelinkCandidate:
    asset_id: str
    path: Path
    score: float
    exact: bool  # identical content (hash)
    reasons: list[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.score >= HIGH_CONFIDENCE


class RelinkError(AppError):
    """The replacement was rejected."""


class MediaRelinkService:
    def __init__(self, project_getter: Callable[[], Project], apply_command: Callable, probe: MediaProbeService) -> None:
        self._project, self._apply, self.probe = project_getter, apply_command, probe

    def missing(self) -> list[Asset]:
        p = self._project()
        return [a for a in p.assets.all() if not p.asset_path(a).is_file()]

    # ------------------------------------------------------------------ search
    def find_candidates(self, asset: Asset, folders: list[Path], recursive: bool = True, limit_files: int = 20000) -> list[RelinkCandidate]:
        """Rank files in ``folders`` as replacements for ``asset``: by hash (exact), file name, size and media facts (duration, size, frame rate)."""
        exts = EXT_OF[asset.type]
        name = Path(asset.name).name.lower()
        stem = Path(asset.name).stem.lower()
        want_size = asset.size_bytes
        out: list[RelinkCandidate] = []
        seen = 0
        for folder in folders:
            folder = Path(folder)
            if not folder.is_dir():
                continue
            walker = os.walk(folder) if recursive else [(str(folder), [], [f.name for f in folder.iterdir() if f.is_file()])]
            for root, _d, files in walker:
                for f in files:
                    if Path(f).suffix.lower() not in exts:
                        continue
                    seen += 1
                    if seen > limit_files:
                        break
                    path = Path(root) / f
                    cand = self._score(asset, path, name, stem, want_size)
                    if cand is not None:
                        out.append(cand)
        out.sort(key=lambda c: (-c.score, str(c.path)))
        return out

    def _score(self, asset: Asset, path: Path, name: str, stem: str, want_size: int) -> RelinkCandidate | None:
        score, reasons = 0.0, []
        fname = path.name.lower()
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if fname == name:
            score += 35
            reasons.append("same file name")
        elif path.stem.lower() == stem:
            score += 20
            reasons.append("same name, other extension")
        elif stem and (stem in path.stem.lower() or path.stem.lower() in stem):
            score += 8
            reasons.append("similar name")
        if want_size and size == want_size:
            score += 20
            reasons.append("same size")
        exact = False
        if asset.content_hash and (size == want_size):
            try:
                if sha256_file(path) == asset.content_hash:
                    return RelinkCandidate(asset.id, path, 100.0, True, ["identical content (hash)"] + reasons)
            except OSError:
                pass
        if score < 8:
            return None  # nothing to go on: never suggest a file by metadata alone
        info, _err = self.probe.try_probe(path)
        if info is not None:
            facts = 0
            if asset.duration and info.duration and abs(info.duration - asset.duration) <= 0.05:
                facts += 1
            if asset.width and asset.height and (info.width, info.height) in ((asset.width, asset.height), (asset.height, asset.width)):
                facts += 1
            if asset.fps and info.fps and abs(info.fps - asset.fps) < 0.01:
                facts += 1
            if facts:
                score += 15 * facts
                reasons.append(f"{facts} matching media fact(s)")
        return RelinkCandidate(asset.id, path, min(score, 99.0), exact, reasons)

    # ------------------------------------------------------------------ apply
    def relink(self, asset_id: str, path: Path, confirmed: bool = False) -> Asset:
        """Replace the asset's file. A different file than the original (or a weak match) needs ``confirmed=True``: nothing is replaced on a guess."""
        project = self._project()
        asset = project.assets.require(asset_id)
        path = Path(path)
        if not path.is_file():
            raise RelinkError("That file does not exist.")
        exts = EXT_OF[asset.type]
        if path.suffix.lower() not in exts:
            raise RelinkError(f"A {asset.type.value} file is needed (one of {', '.join(sorted(e.lstrip('.') for e in exts))}).")
        info, err = self.probe.try_probe(path)
        if info is None:
            raise RelinkError(f"That file cannot be used: {err}")
        cand = self._score(asset, path, Path(asset.name).name.lower(), Path(asset.name).stem.lower(), asset.size_bytes)
        score = cand.score if cand else 0.0
        if not (cand and cand.confident) and not confirmed:
            raise RelinkError(f"This file does not clearly match “{asset.name}” (confidence {score:.0f}%). Confirm to use it anyway.")
        facts = {"size_bytes": path.stat().st_size, "duration": info.duration, "width": info.width if info.rotation not in (90, 270) else info.coded_width,
                 "height": info.height if info.rotation not in (90, 270) else info.coded_height, "fps": info.fps, "codec": info.codec,
                 "has_audio": info.has_audio, "audio_codec": info.audio_codec, "sample_rate": info.sample_rate, "channels": info.channels}
        if not (cand and cand.exact):
            try:
                facts["content_hash"] = sha256_file(path)
            except OSError as exc:
                raise MediaError(f"The file could not be read: {exc}") from exc
        self._apply(RelinkAssetCommand(project, asset_id, path, facts))
        return project.assets.require(asset_id)

    def auto_relink(self, folders: list[Path]) -> dict[str, Path]:
        """Relink every missing asset that has exactly one identical-content file in ``folders``. Weaker matches are only reported by ``find_candidates``."""
        done: dict[str, Path] = {}
        for asset in self.missing():
            exact = [c for c in self.find_candidates(asset, folders) if c.exact]
            if len(exact) >= 1:
                self.relink(asset.id, exact[0].path, confirmed=True)
                done[asset.id] = exact[0].path
        return done


_ = MediaError
