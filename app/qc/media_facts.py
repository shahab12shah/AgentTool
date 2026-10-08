"""Facts about the media the timeline actually uses, gathered once per QC run and shared by the checkers that need them (pre-flight, assets, frames, audio).

Only assets that are *used* (a clip refers to them, or they are the voice-over) are looked at. Probing goes through the probe service, which caches by
(path, size, mtime), so a second run over unchanged files costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.qc.context import QCContext
from app.timeline.clip import KIND_MEDIA, Clip
from app.timeline.track import Track, TrackKind

if TYPE_CHECKING:  # pragma: no cover
    from app.media.asset import Asset
    from app.rendering.probe import ProbeInfo


@dataclass
class UsedAsset:
    asset: "Asset"
    path: Path
    role: str  # visual | audio | voice
    uses: list[tuple[Track, Clip]] = field(default_factory=list)
    exists: bool = False
    info: "ProbeInfo | None" = None
    error: str = ""  # why the file could not be read ("" = readable, or not probed)
    probed: bool = False

    @property
    def first_use(self) -> tuple[Track, Clip] | None:
        return min(self.uses, key=lambda u: (u[1].timeline_start, u[1].id)) if self.uses else None

    @property
    def scene_ids(self) -> list[str]:
        return sorted({c.scene_id for _t, c in self.uses if c.scene_id})


def used_assets(ctx: QCContext) -> dict[str, UsedAsset]:
    """Every asset the timeline (or the voice-over slot) refers to and that exists in the registry, with whether its file is on disk."""

    def build() -> dict[str, UsedAsset]:
        out: dict[str, UsedAsset] = {}
        for t, c in ctx.clips(kind=KIND_MEDIA):
            a = ctx.asset(c.asset_id)
            if a is None:
                continue  # a dangling reference is a document problem (pre-flight), not an asset-file problem
            role = "audio" if t.kind is TrackKind.AUDIO else "visual"
            u = out.get(a.id)
            if u is None:
                u = out[a.id] = UsedAsset(a, ctx.asset_path(a), role)
            elif role == "visual":
                u.role = "visual"
            u.uses.append((t, c))
        vo = ctx.asset(ctx.project.voice_over.asset_id)
        if vo is not None and vo.id not in out:
            out[vo.id] = UsedAsset(vo, ctx.asset_path(vo), "voice")
        elif vo is not None:
            out[vo.id].role = "voice"
        for u in out.values():
            u.exists = u.path.is_file()
        return out

    return ctx.memo("media.used", build)


def ffmpeg_ready(ctx: QCContext) -> bool:
    if ctx.ffmpeg is None or ctx.probe is None:
        return False
    try:
        return bool(ctx.ffmpeg.detect()[0])
    except Exception:  # noqa: BLE001 - a broken FFmpeg install is reported by pre-flight, not here
        return False


def probe_used(ctx: QCContext, progress=None) -> dict[str, UsedAsset]:
    """``used_assets`` with every existing file probed (once per run). Without FFmpeg nothing is probed and ``probed`` stays False."""

    def run() -> dict[str, UsedAsset]:
        used = used_assets(ctx)
        if not ffmpeg_ready(ctx):
            return used
        todo = [u for u in used.values() if u.exists]
        for i, u in enumerate(sorted(todo, key=lambda u: u.asset.id)):
            ctx.check_cancel()
            info, err = ctx.probe.try_probe(u.path)  # type: ignore[union-attr]
            u.info, u.error, u.probed = info, ("" if info is not None else (err or "The file cannot be decoded.")), True
            if progress:
                progress(i / max(1, len(todo)), f"Reading {u.asset.name}")
        return used

    return ctx.memo("media.probed", run)
