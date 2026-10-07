"""Which file does a clip read? Final renders use the original media; previews prefer proxies. Never a silent mix-up."""

from __future__ import annotations

from pathlib import Path

from app.rendering.compiler import SourceInfo
from app.rendering.errors import RenderError
from app.rendering.models import AssetRef, ProxyRef, RenderSnapshot
from app.rendering.probe import MediaProbeService


class MissingMediaError(RenderError):
    def __init__(self, assets: list[AssetRef], proxy_available: list[str] | None = None, stage: str = "Preparing Media") -> None:
        names = ", ".join(a.name for a in assets[:5]) + (f" and {len(assets) - 5} more" if len(assets) > 5 else "")
        super().__init__(f"Original media is missing: {names}.", stage=stage, kind="missing_media", asset_ids=[a.id for a in assets],
                         possible_issue="A source file was moved, renamed or deleted." + (" A proxy exists for some of it." if proxy_available else ""))
        self.proxy_available = proxy_available or []


def proxy_usable(ref: ProxyRef | None) -> bool:
    return bool(ref and ref.status == "READY" and ref.path and Path(ref.path).is_file())


class SourceSelector:
    """``mode``: ``final`` (original media; a proxy only when the user explicitly allowed it) or ``preview`` (proxy when it is ready and current)."""

    def __init__(self, snapshot: RenderSnapshot, probe: MediaProbeService, mode: str = "final", allow_proxy_assets: set[str] | None = None, use_proxies: bool = False) -> None:
        self.s, self.probe, self.mode = snapshot, probe, mode
        self.allow = allow_proxy_assets or set()
        self.use_proxies = use_proxies or mode == "preview"
        self.used_proxies: set[str] = set()

    def resolve(self, asset: AssetRef) -> SourceInfo:
        ref = self.s.proxies.get(asset.id)
        original = Path(asset.path)
        if asset.type == "audio":
            return SourceInfo(asset.path, False, 0, 0)
        if self.use_proxies and proxy_usable(ref) and original_current(asset, ref):
            self.used_proxies.add(asset.id)
            return SourceInfo(ref.path, True, ref.width or asset.width or 0, ref.height or asset.height or 0)  # type: ignore[union-attr]
        if not original.is_file():
            if proxy_usable(ref) and asset.id in self.allow:
                self.used_proxies.add(asset.id)
                return SourceInfo(ref.path, True, ref.width or asset.width or 0, ref.height or asset.height or 0)  # type: ignore[union-attr]
            raise MissingMediaError([asset], [asset.id] if proxy_usable(ref) else [])
        w, h = asset.width, asset.height
        if not (w and h):
            info = self.probe.probe(original)
            w, h = info.width, info.height
            asset.width, asset.height = w, h
        return SourceInfo(asset.path, False, int(w or 0), int(h or 0))


def original_current(asset: AssetRef, ref: ProxyRef | None) -> bool:
    """A proxy made from a file that has since changed (or was re-linked to another file) is stale; the original is used instead."""
    if ref is None or not ref.source_size:
        return True
    try:
        st = Path(asset.path).stat()
    except OSError:
        return True  # the original is gone: the proxy is all there is
    return st.st_size == ref.source_size and st.st_mtime_ns == ref.source_mtime_ns


def prepare_assets(snapshot: RenderSnapshot, probe: MediaProbeService) -> list[AssetRef]:
    """Read the real facts of every used asset (size after rotation, frame rate, alpha) with FFprobe. Returns the assets whose files are missing."""
    missing: list[AssetRef] = []
    for a in snapshot.assets.values():
        p = Path(a.path)
        if not p.is_file():
            a.exists = False
            missing.append(a)
            continue
        a.exists = True
        if a.type in ("video", "image"):
            info = probe.probe(p)
            a.width, a.height, a.fps, a.has_alpha, a.rotation = info.width, info.height, info.fps, info.has_alpha, info.rotation
            a.duration = a.duration or info.duration
        elif a.type == "audio" and not a.duration:
            a.duration = probe.probe(p).duration
    return missing
