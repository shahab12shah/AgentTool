"""Registry of every asset in a project."""

from __future__ import annotations

from typing import Any, Iterator

from app.core.exceptions import MediaError
from app.media.asset import Asset


class AssetRegistry:
    """Ordered collection of assets with stable, never-reused IDs (``media_00001``)."""

    def __init__(self, assets: list[Asset] | None = None, counter: int = 0) -> None:
        self._assets: dict[str, Asset] = {}
        self._counter = counter
        for asset in assets or []:
            self._assets[asset.id] = asset
            self._counter = max(self._counter, _numeric_suffix(asset.id))

    @property
    def counter(self) -> int:
        return self._counter

    def new_id(self) -> str:
        self._counter += 1
        return f"media_{self._counter:05d}"

    def add(self, asset: Asset) -> Asset:
        if asset.id in self._assets:
            raise MediaError(f"Asset id {asset.id} already exists.")
        self._assets[asset.id] = asset
        self._counter = max(self._counter, _numeric_suffix(asset.id))
        return asset

    def remove(self, asset_id: str) -> Asset:
        try:
            return self._assets.pop(asset_id)
        except KeyError:
            raise MediaError(f"Asset {asset_id} is not in this project.") from None

    def get(self, asset_id: str) -> Asset | None:
        return self._assets.get(asset_id)

    def require(self, asset_id: str) -> Asset:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise MediaError(f"Asset {asset_id} is not in this project.")
        return asset

    def find_by_hash(self, content_hash: str) -> Asset | None:
        return next((a for a in self._assets.values() if a.content_hash == content_hash), None)

    def known_hashes(self) -> dict[str, str]:
        """Snapshot ``hash -> asset id`` (safe to hand to a worker thread)."""
        return {a.content_hash: a.id for a in self._assets.values() if a.content_hash}

    def all(self) -> list[Asset]:
        return list(self._assets.values())

    def __iter__(self) -> Iterator[Asset]:
        return iter(list(self._assets.values()))

    def __len__(self) -> int:
        return len(self._assets)

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self._assets

    def to_list(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._assets.values()]


def _numeric_suffix(asset_id: str) -> int:
    tail = asset_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0
