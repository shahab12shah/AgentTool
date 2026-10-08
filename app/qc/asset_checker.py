"""Asset integrity and media quality (spec 23-24).

For every asset the timeline uses (and the voice-over): is the file there, can it be decoded, does it still match what the project recorded about it, and is it
good enough for the output size. A missing file is CRITICAL and comes with the real ways out (relink, replace, remove, search, skip); an *identical* copy of it
(same content hash) found in the project or beside the original is offered as a safe, exact relink. Low resolution is only ever a warning: QC never rejects media
on its own unless the user sets ``media.reject_below_ratio``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.core.constants import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.media.asset import AssetType
from app.media.thumbnails import ThumbnailService
from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue
from app.qc.media_facts import UsedAsset, ffmpeg_ready, probe_used
from app.qc.severity import Severity
from app.rendering import presets as P

KNOWN_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS
ALL_ACTIONS = "Relink the file, replace it, remove it from the timeline, search for an alternative visual, or skip the scene."
DURATION_TOLERANCE = 0.25  # seconds: the registry's duration against the file's (container rounding)
BLUR_VARIANCE, FLAT_STD = 30.0, 12.0  # Laplacian variance of a 512 px copy below which a photo is "soft"; below FLAT_STD the picture is a flat graphic (never reported)
MAX_RELINK_SEARCH = 12  # missing assets searched for an identical copy per run (each search walks the project media folder)
RELINK_FILE_LIMIT = 4000


class _NoProbe:
    def try_probe(self, _path: Path):  # relink scoring only needs it for weak matches, which QC never offers
        return None, "not probed"


class AssetChecker(BaseChecker):
    id = "asset"
    label = "Assets & media quality"
    categories = (QCCategory.ASSET, QCCategory.MEDIA_QUALITY)
    domains = ("timeline", "assets")
    settings_sections = ("media",)
    scene_local = False
    version = "1"

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        used = probe_used(ctx, lambda f, m: report(0.05 + 0.3 * f, m))
        if not used:
            out.notes.append("no media in use")
            out.metrics = {"assets_checked": 0, "missing": 0, "corrupt": 0, "low_resolution": 0}
            return out
        if not ffmpeg_ready(ctx):
            out.notes.append("media were not decoded (FFmpeg unavailable): existence and recorded facts only")
        w, h = P.output_size(*ctx.canvas, ctx.project.render_settings.resolution)
        counts = {"missing": 0, "corrupt": 0, "low_resolution": 0}
        searches = 0
        items = sorted(used.values(), key=lambda u: (u.first_use[1].timeline_start if u.first_use else 1e9, u.asset.id))
        for n, u in enumerate(items):
            ctx.check_cancel()
            report(0.35 + 0.6 * n / max(1, len(items)), f"Checking {u.asset.name}")
            if not u.exists:
                counts["missing"] += 1
                searches += 1
                out.issues.append(self._missing(ctx, u, search=searches <= MAX_RELINK_SEARCH))
                continue
            if u.probed and u.info is None:
                counts["corrupt"] += 1
                out.issues.append(self._corrupt(ctx, u))
                continue
            out.issues += self._recorded_facts(ctx, u)
            if u.role != "audio" and u.role != "voice":
                found = self._quality(ctx, u, (w, h))
                counts["low_resolution"] += sum(1 for i in found if i.code == "media.low_resolution")
                out.issues += found
        out.issues += self._thumbnails(ctx, items)
        out.metrics = {"assets_checked": len(items), **counts}
        out.notes.append(f"checked {len(items)} asset(s): {counts['missing']} missing, {counts['corrupt']} undecodable, {counts['low_resolution']} low-resolution")
        if searches > MAX_RELINK_SEARCH:
            out.notes.append(f"searched for identical copies of the first {MAX_RELINK_SEARCH} of {searches} missing assets only")
            out.complete = False
        report(1.0, "Asset check complete")
        return out

    # ------------------------------------------------------------------ location helpers
    def _where(self, ctx: QCContext, u: UsedAsset) -> tuple[list[str], str | None]:
        """Human-readable places the asset is used ("Scene 14 · clip_x") and the first scene."""
        places, first = [], None
        for t, c in sorted(u.uses, key=lambda x: (x[1].timeline_start, x[1].id)):
            sid = ctx.clip_scene_id(c)
            first = first or sid
            places.append(f"{ctx.scene_label(sid)} · {c.id}")
        if u.role == "voice" and not places:
            places.append("voice-over")
        return places, first

    def _place(self, iss: QCIssue, u: UsedAsset) -> QCIssue:
        """Anchor the issue on the asset's first use without making it 'a user clip' (an asset-level fix does not touch the clip, so clip ownership must not block it)."""
        use = u.first_use
        if use is not None:
            t, c = use
            iss.timeline_item_id, iss.track_id, iss.start_time, iss.end_time = c.id, t.id, c.timeline_start, c.timeline_end
            iss.fingerprint = iss.make_fingerprint(iss.metrics.get("signature", ""))
        return iss

    # ------------------------------------------------------------------ missing / corrupt
    def _missing(self, ctx: QCContext, u: UsedAsset, search: bool) -> QCIssue:
        a = u.asset
        places, scene = self._where(ctx, u)
        found = self._identical_copy(ctx, u) if search else None
        if found is not None:
            fix = fx.asset_relink(a.id, str(found), True, ctx.settings)
            how = f"An identical copy was found at {found}: relinking restores it exactly."
        elif u.role == "visual":
            fix = fx.navigate("asset.replace", "Choose a replacement for the missing media", asset_id=a.id)
            how = ALL_ACTIONS
        else:
            fix = fx.navigate("asset.replace", "Locate or replace the missing file", asset_id=a.id)
            how = "Relink the file, replace it, or remove it from the timeline."
        what = {"voice": "voice-over file", "audio": "audio file", "visual": "media file"}[u.role]
        iss = self.issue(
            "asset.missing", QCCategory.ASSET, Severity.CRITICAL, "Missing voice-over" if u.role == "voice" else "Missing asset",
            description=f"The {what} “{a.name}” ({a.path}) is not on disk. Used in: {', '.join(places[:6])}{' ...' if len(places) > 6 else ''}.",
            scene_id=scene, affected=[a.id, a.name, *places[:12]], why="The export would fail, or show a black frame where this media belongs.", current="file not found",
            recommended="the file available (or replaced)", suggested_fix=how, fix=fix, viewer_impact=1.0, signature=a.id, metrics={"signature": a.id, "asset_id": a.id}, ctx=ctx)
        return self._place(iss, u)

    def _identical_copy(self, ctx: QCContext, u: UsedAsset) -> Path | None:
        a = u.asset
        if not a.content_hash or not a.size_bytes:
            return None  # without a recorded hash nothing can be proven identical
        from app.rendering.relink import MediaRelinkService  # noqa: PLC0415 - keeps the render layer out of this module's import graph

        svc = MediaRelinkService(lambda: ctx.project, lambda _c: None, ctx.probe or _NoProbe())  # type: ignore[arg-type]
        seen: set[Path] = set()
        folders: list[tuple[Path, bool]] = [(u.path.parent, False)]
        if ctx.root is not None:
            folders.append((ctx.root / "media", True))
        for folder, recursive in folders:
            ctx.check_cancel()
            if not folder.is_dir() or folder in seen:
                continue
            seen.add(folder)
            for c in svc.find_candidates(a, [folder], recursive=recursive, limit_files=RELINK_FILE_LIMIT):
                if c.exact and c.path != u.path:
                    return c.path
        return None

    def _corrupt(self, ctx: QCContext, u: UsedAsset) -> QCIssue:
        a = u.asset
        places, scene = self._where(ctx, u)
        unsupported = u.path.suffix.lower() not in KNOWN_EXTENSIONS
        code = "asset.unsupported_format" if unsupported else "asset.corrupt"
        iss = self.issue(
            code, QCCategory.ASSET, Severity.CRITICAL, "Unsupported media format" if unsupported else "Corrupt or unreadable media",
            description=f"“{a.name}” cannot be decoded: {u.error}. Used in: {', '.join(places[:6])}.", scene_id=scene, affected=[a.id, a.name, *places[:12]],
            why="FFmpeg cannot read this file, so the export would fail.", current=u.error or "undecodable", recommended="a readable file in a supported format",
            suggested_fix="Replace the file, re-encode it to MP4/H.264, or remove it from the timeline." + ("" if u.role != "visual" else " Or search for an alternative visual."),
            fix=fx.navigate("asset.replace", "Choose a replacement for the unreadable media", asset_id=a.id), viewer_impact=1.0, signature=a.id,
            metrics={"signature": a.id, "asset_id": a.id}, ctx=ctx)
        return self._place(iss, u)

    # ------------------------------------------------------------------ facts recorded at import vs the file now
    def _recorded_facts(self, ctx: QCContext, u: UsedAsset) -> list[QCIssue]:
        a, info, out = u.asset, u.info, []
        places, scene = self._where(ctx, u)
        if a.type is not AssetType.IMAGE and (a.duration is None or a.duration <= 0):
            out.append(self._place(self.issue(
                "asset.duration_invalid", QCCategory.ASSET, Severity.ERROR, "Invalid media duration", description=f"“{a.name}” is recorded with duration {a.duration!r}.", scene_id=scene,
                affected=[a.id, *places[:8]], why="Trimming, looping and sync all depend on the source duration.", current=str(a.duration), recommended="a positive duration",
                suggested_fix="Re-import or replace the file.", fix=fx.navigate("asset.replace", "Replace the asset", asset_id=a.id), viewer_impact=0.8, signature=a.id,
                metrics={"signature": a.id}, ctx=ctx), u))
        reasons: list[str] = []
        try:
            size = u.path.stat().st_size
        except OSError:
            size = 0
        if a.size_bytes and size and size != a.size_bytes:
            reasons.append(f"file size changed ({a.size_bytes:,} -> {size:,} bytes)")
        past_end = 0
        if info is not None and info.duration and a.duration and abs(info.duration - a.duration) > DURATION_TOLERANCE:
            reasons.append(f"duration is {info.duration:.2f} s, the project recorded {a.duration:.2f} s")
            past_end = sum(1 for _t, c in u.uses if c.source_out > info.duration + 0.05)
        if info is not None and info.width and a.width and a.height and (info.width, info.height) not in ((a.width, a.height), (a.height, a.width)):
            reasons.append(f"resolution is {info.width}x{info.height}, the project recorded {a.width}x{a.height}")
        if reasons:
            sev = Severity.ERROR if past_end else Severity.WARNING
            out.append(self._place(self.issue(
                "asset.modified", QCCategory.ASSET, sev, "Media changed since it was imported",
                description=f"“{a.name}” no longer matches what the project recorded: {'; '.join(reasons)}." + (f" {past_end} clip(s) now run past the end of the file." if past_end else ""),
                scene_id=scene, affected=[a.id, *places[:8]], why="The file may have been replaced or re-encoded, so trims and sync could be off.", current="; ".join(reasons),
                recommended="the original file", suggested_fix="Check the media; relink to the original file or replace it.", fix=fx.navigate("asset.replace", "Relink or replace the asset", asset_id=a.id),
                confidence=90.0, viewer_impact=0.6 if past_end else 0.3, signature=sha(a.id, reasons), metrics={"signature": sha(a.id, reasons)}, ctx=ctx), u))
        return out

    # ------------------------------------------------------------------ media quality for the output size
    def _quality(self, ctx: QCContext, u: UsedAsset, out_size: tuple[int, int]) -> list[QCIssue]:
        a, info, cfg = u.asset, u.info, ctx.settings.media
        sw = (info.width if info and info.width else a.width) or 0
        sh = (info.height if info and info.height else a.height) or 0
        if sw <= 0 or sh <= 0:
            return []
        ow, oh = out_size
        cw, ch = ctx.canvas
        found: list[QCIssue] = []
        places, scene = self._where(ctx, u)
        use = u.first_use
        contain = bool(use) and all(str(c.effects.get("fit", "cover")) in ("contain", "fit") for _t, c in u.uses)
        fx_, fy_ = ow / sw, oh / sh
        base = min(fx_, fy_) if contain else max(fx_, fy_)  # how much the renderer enlarges the source to fit the frame
        zoom = max((max(c.scale, 1e-6) for _t, c in u.uses), default=1.0)
        ratio = 1.0 / base if base > 0 else 1.0  # source pixels per output pixel at the fitted size
        label = f"Source {sw}x{sh}, timeline {ow}x{oh}"
        if cfg.reject_below_ratio > 0 and ratio < cfg.reject_below_ratio:
            found.append(self._place(self.issue(
                "media.rejected", QCCategory.MEDIA_QUALITY, Severity.ERROR, "Source resolution below your minimum", description=f"{label}: {ratio * 100:.0f}% of the output size (your minimum is {cfg.reject_below_ratio * 100:.0f}%).",
                scene_id=scene, affected=[a.id, *places[:8]], why="You asked QC to reject media below this resolution.", current=f"{sw}x{sh}", recommended=f"at least {cfg.reject_below_ratio * 100:.0f}% of {ow}x{oh}",
                suggested_fix="Replace it with a higher-resolution visual.", fix=fx.navigate("visual.search_again", "Search for a higher-resolution visual", scene_id=scene or ""), viewer_impact=0.6,
                signature=a.id, metrics={"signature": a.id, "ratio": round(ratio, 3)}, ctx=ctx), u))
        elif ratio < cfg.min_source_ratio:
            found.append(self._place(self.issue(
                "media.low_resolution", QCCategory.MEDIA_QUALITY, Severity.WARNING, f"Low-resolution source may appear soft in {_label(ow, oh)} export",
                description=f"{label}. The picture is enlarged {base:.1f}x to fill the frame.", scene_id=scene, affected=[a.id, a.name, *places[:8]],
                why="Enlarged low-resolution media looks blurry next to sharper shots.", current=f"{sw}x{sh} ({ratio * 100:.0f}% of output)", recommended=f"at least {cfg.min_source_ratio * 100:.0f}% of {ow}x{oh}",
                suggested_fix="Search for a higher-resolution version, or keep it if the softness is acceptable.", fix=fx.navigate("visual.search_again", "Search for a sharper visual", scene_id=scene or ""),
                confidence=90.0, viewer_impact=0.4, signature=sha(a.id, ow, oh), metrics={"signature": sha(a.id, ow, oh), "ratio": round(ratio, 3), "source": [sw, sh], "output": [ow, oh]}, ctx=ctx), u))
        total = base * zoom
        if total > cfg.max_upscale and zoom > 1.05:  # the source alone is not the cause: the clip's own zoom pushes it past the limit
            found.append(self._place(self.issue(
                "media.upscale", QCCategory.MEDIA_QUALITY, Severity.WARNING, "Media is enlarged beyond the recommended limit",
                description=f"{label}: fitted at {base:.1f}x and zoomed to {zoom:.2f}x, a total enlargement of {total:.1f}x (limit {cfg.max_upscale:.1f}x).", scene_id=scene,
                affected=[a.id, *places[:8]], why="Heavy enlargement loses detail and shows compression artefacts.", current=f"{total:.1f}x", recommended=f"at most {cfg.max_upscale:.1f}x",
                suggested_fix="Reduce the zoom on the clip or use a higher-resolution source.", fix=fx.navigate("visual.search_again", "Search for a sharper visual", scene_id=scene or ""),
                viewer_impact=0.4, signature=sha(a.id, round(total, 1)), metrics={"signature": sha(a.id, round(total, 1)), "scale": round(total, 2)}, ctx=ctx), u))
        # a source whose shape differs a lot from the frame is cropped by 'cover' (a contained clip is letterboxed on purpose)
        if not contain and cw > 0 and ch > 0:
            rel = abs((sw / sh) / (cw / ch) - 1.0)
            if rel > cfg.expected_aspect_tolerance:
                cropped = 1.0 - min(sw / sh, cw / ch) / max(sw / sh, cw / ch)
                found.append(self._place(self.issue(
                    "media.aspect", QCCategory.MEDIA_QUALITY, Severity.WARNING if rel > 0.5 else Severity.NOTICE, "Media shape differs from the frame",
                    description=f"{sw}x{sh} ({P.aspect_label(sw, sh)}) in a {P.aspect_label(cw, ch)} frame: about {cropped * 100:.0f}% of the picture is cropped.", scene_id=scene,
                    affected=[a.id, *places[:8]], why="Important parts of the picture may be cut off at the edges.", current=P.aspect_label(sw, sh), recommended=P.aspect_label(cw, ch),
                    suggested_fix="Check the framing, set the clip to 'contain', or choose a source with the right shape.", confidence=85.0, viewer_impact=0.3 if rel <= 0.5 else 0.5,
                    signature=sha(a.id, round(rel, 2)), metrics={"signature": sha(a.id, round(rel, 2)), "crop": round(cropped, 3)}, ctx=ctx), u))
        if a.type is AssetType.IMAGE:
            soft = self._soft_image(ctx, u)
            if soft is not None:
                found.append(self._place(self.issue(
                    "media.blurry", QCCategory.MEDIA_QUALITY, Severity.NOTICE, "Image looks soft or out of focus",
                    description=f"Edge detail measured {soft:.0f} (a sharp photo at this scale is well above {BLUR_VARIANCE:.0f}).", scene_id=scene, affected=[a.id, a.name, *places[:8]],
                    why="Blurry stills look unprofessional, especially when enlarged.", current=f"sharpness {soft:.0f}", recommended="a sharper image", suggested_fix="Search for a sharper image.",
                    fix=fx.navigate("visual.search_again", "Search for a sharper image", scene_id=scene or ""), confidence=65.0, viewer_impact=0.3, signature=a.id,
                    metrics={"signature": a.id, "sharpness": round(soft, 1)}, source="deterministic:asset.sharpness", ctx=ctx), u))
        return found

    @staticmethod
    def _soft_image(ctx: QCContext, u: UsedAsset) -> float | None:
        """Laplacian variance of a 512 px greyscale copy; None when the picture is sharp, flat (a graphic) or unreadable."""
        def measure() -> tuple[float, float] | None:
            try:
                with Image.open(u.path) as im:
                    im = im.convert("L")
                    im.thumbnail((512, 512))
                    g = np.asarray(im, dtype=np.float32)
            except (OSError, ValueError):
                return None
            if g.shape[0] < 8 or g.shape[1] < 8:
                return None
            lap = -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
            return float(lap.var()), float(g.std())

        m = ctx.memo(f"sharp.{u.asset.id}", measure)
        if m is None:
            return None
        var, std = m
        return var if (var < BLUR_VARIANCE and std > FLAT_STD) else None

    # ------------------------------------------------------------------ thumbnails (cosmetic: one NOTICE for all)
    def _thumbnails(self, ctx: QCContext, items: list[UsedAsset]) -> list[QCIssue]:
        if ctx.root is None:
            return []
        def cached(u: UsedAsset) -> bool:
            p = ThumbnailService.thumbnail_path(ctx.root, u.asset)  # type: ignore[arg-type]
            return p.is_file() and p.stat().st_size > 0

        missing = sorted(u.asset.name for u in items if u.exists and u.role == "visual" and not cached(u))
        if not missing:
            return []
        return [self.issue(
            "media.thumbnail_missing", QCCategory.MEDIA_QUALITY, Severity.NOTICE, "Some media have no thumbnail", description=f"{len(missing)} used media file(s) have no cached thumbnail: " + ", ".join(missing[:5]),
            affected=missing[:12], why="Thumbnails make the media library and timeline easier to read; the export is not affected.", current=f"{len(missing)} missing", recommended="thumbnails generated",
            suggested_fix="Open the Media page: thumbnails are generated automatically.", viewer_impact=0.0, signature=sha(missing), ctx=ctx)]


def _label(w: int, h: int) -> str:
    short = min(w, h)
    return "4K" if short >= 2160 else f"{short}p"

