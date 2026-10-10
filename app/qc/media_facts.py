"""Facts about the media the timeline actually uses, gathered once per QC run and shared by the checkers that need them (pre-flight, assets, frames, audio).

Only assets that are *used* (a clip refers to them, or they are the voice-over) are looked at. Probing goes through the probe service, which caches by
(path, size, mtime), so a second run over unchanged files costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.qc.context import QCContext, sha
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


# ---------------------------------------------------------------------------------------------- cache-key extras
# The context's domain hashes cover the timeline, the scene boundaries / text, the transcript and the asset FILES, but a checker also reads facts none of them contain: a scene's
# claims, numbers and entities, an asset's name / source type / tags, the researched candidate's title, the AI decisions, the editing style, thumbnails on disk, a neighbouring
# scene's words. A checker adds exactly the parts it reads with ``extras(...)`` to its ``input_hash`` / ``scene_input_hash``, otherwise a change in one of them would reuse a stale result.
def _scene_facts(ctx: QCContext, scene_id: str | None) -> str:
    def one(s) -> list:
        intent = ctx.project.visual_intents.get(s.id)
        return [s.id, s.narration, s.script_text, s.topic, [[c.claim_id, c.text, c.requires_evidence, c.sentence_id] for c in s.claims],
                [[n.text, n.kind.value, n.value, n.sentence_id, list(n.word_ids)] for n in s.numbers], [[e.text, e.type.value, e.canonical] for e in s.entities],
                [intent.type.value, intent.primary_subject, intent.secondary_subject] if intent else None]

    if scene_id is not None:
        s = ctx.scene(scene_id)
        return sha(one(s)) if s is not None else ""
    return ctx.memo("x.facts", lambda: sha([one(s) for s in ctx.scenes]))


def _asset_ids(ctx: QCContext, scene_id: str | None) -> list[str] | None:
    if scene_id is None:
        return None
    s = ctx.scene(scene_id)
    if s is None:
        return []
    ids = {c.asset_id for _t, c in ctx.clips_in(s.start - 0.5, s.end + 0.5) if c.asset_id}
    a = ctx.project.visual_assignments.get(scene_id)
    if a is not None and a.asset_id:
        ids.add(a.asset_id)
    return sorted(ids)


def _asset_meta(ctx: QCContext, scene_id: str | None) -> str:
    ids = _asset_ids(ctx, scene_id)
    rows = []
    for a in sorted(ctx.project.assets.all(), key=lambda a: a.id):
        if ids is None or a.id in ids:
            rows.append([a.id, a.name, a.path, a.type.value, a.source_type.value, a.extra or {}])
    return sha(rows)


def _candidates(ctx: QCContext, scene_id: str | None) -> str:
    ids = _asset_ids(ctx, scene_id)
    rows = []
    for cid, c in sorted(ctx.project.visual_candidates.items()):
        if ids is None or c.asset_id in ids or (scene_id is not None and c.scene_id == scene_id):
            rows.append([cid, c.scene_id, c.asset_id, c.title, c.description, list(c.tags), c.source_type.value, c.evidence_kind.value])
    return sha(rows)


def _decisions(ctx: QCContext, scene_id: str | None) -> str:  # noqa: ARG001 - decisions are project-wide
    def calc() -> str:
        p = ctx.project
        rows = []
        for store in (p.editing_decisions, p.presentation_decisions):
            for did, d in sorted(store.items()):
                rows.append([did, d.scene_id, getattr(d.type, "value", str(d.type)), d.target_id, d.confidence, d.reason, bool(getattr(d, "locked", False)), str(getattr(d, "created_by", ""))])
        return sha(rows)

    return ctx.memo("x.decisions", calc)


def _editing(ctx: QCContext, scene_id: str | None) -> str:  # noqa: ARG001
    es = ctx.project.editing_settings
    return sha([getattr(es, k, None) for k in ("style", "pacing", "motion_intensity", "transition_frequency", "text_emphasis", "number_emphasis", "evidence_treatment")])


def _thumbs(ctx: QCContext, scene_id: str | None) -> str:
    if ctx.root is None:
        return ""
    from app.media.thumbnails import ThumbnailService  # noqa: PLC0415

    ids = _asset_ids(ctx, scene_id)
    rows = []
    for a in sorted(ctx.project.assets.all(), key=lambda a: a.id):
        if ids is not None and a.id not in ids:
            continue
        try:
            st = ThumbnailService.thumbnail_path(ctx.root, a).stat()
            rows.append([a.id, st.st_size, int(st.st_mtime)])
        except OSError:
            rows.append([a.id, None])
    return sha(rows)


def _neighbours(ctx: QCContext, scene_id: str | None) -> str:
    """What a scene's boundary findings read from the scenes beside it: where the previous narration ends and the next one begins."""
    if scene_id is None:
        return ""
    sc = ctx.scene_ctx(scene_id)
    if sc is None:
        return ""
    nxt = ctx.scene(sc.next.scene_id) if sc.next is not None else None
    nw = ctx.words_between(nxt.start, nxt.end) if nxt is not None else []
    return sha(sc.prev.last_word_end if sc.prev is not None else None, [[w.word_id, round(w.start, 3), round(w.end, 3)] for w in nw[:3]], nxt.start if nxt is not None else None)


def _shared_state(ctx: QCContext, scene_id: str | None) -> str:  # noqa: ARG001
    """What the checkers that ran before produced, down to severity and confidence (``QCContext.shared_signature`` only fingerprints the findings)."""
    return sha([[cid, sorted([i.fingerprint, i.severity.value, round(i.confidence, 1), i.status.value] for i in getattr(o, "issues", []))] for cid, o in sorted(ctx.shared.items())])


_EXTRAS = {"facts": _scene_facts, "assets": _asset_meta, "candidates": _candidates, "decisions": _decisions, "editing": _editing, "thumbs": _thumbs, "neighbours": _neighbours,
           "shared_state": _shared_state}


def extras(ctx: QCContext, *parts: str, scene_id: str | None = None) -> str:
    """Fingerprint of the named facts a checker reads beyond the context's domains (project-wide, or for one scene when ``scene_id`` is given)."""
    return sha([_EXTRAS[p](ctx, scene_id) for p in parts])
