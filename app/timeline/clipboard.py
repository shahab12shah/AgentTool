"""Timeline clipboard: copy / paste of clips and of clip attributes.

``ClipboardData`` is a plain, JSON-serialisable snapshot (clip dicts via ``Clip.to_dict``, start offsets from the earliest copied clip, the vertical track layout, the asset ids and
the link / group ids involved), so it can live on the system clipboard (``MIME_TYPE``) or in memory and still be pasted into another project that has the same assets.
``plan_paste`` is a pure planner like every other edit (see ``ops_basic``): fresh clip ids, fresh link / group ids (remapped consistently, an id left on a single clip is dropped),
USER-owned clips, no locks, no AI decision. ``copy_attributes`` / ``plan_paste_attributes`` copy the look of a clip (transform, opacity, effects, audio volume and fades) onto others.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any

from app.core.constants import TIME_EPSILON
from app.core.exceptions import TimelineError
from app.timeline.clip import KIND_MEDIA, Clip
from app.timeline.ops_basic import (
    SOURCE_SPECIFIC_EFFECTS,
    Grid,
    fresh_copy,
    plan_overwrite,
    remap_ids,
)
from app.timeline.ops_common import new_group_id, new_link_id, relatives
from app.timeline.plan import planner, EditPlan, PlanEnv, clip_fits_track, validate_plan
from app.timeline.timeline import Timeline
from app.timeline.track import TrackKind

MIME_TYPE = "application/x-agenttool-timeline-clips"
CLIPBOARD_VERSION = 1
AUDIO_ATTRIBUTES = ("volume", "fade_in", "fade_out", "pan")


@dataclass
class ClipboardData:
    clips: list[dict[str, Any]] = field(default_factory=list)  # Clip.to_dict() of each copied clip (timeline order)
    offsets: list[float] = field(default_factory=list)  # start of each clip minus the earliest copied start (seconds)
    track_offsets: list[int] = field(default_factory=list)  # track index of each clip minus the topmost source track index
    track_kinds: list[str] = field(default_factory=list)  # TrackKind value of each clip's source track
    track_ids: list[str] = field(default_factory=list)  # source track ids (used when pasting back onto the same tracks)
    asset_ids: list[str] = field(default_factory=list)  # distinct media asset ids the clips use
    link_ids: list[str] = field(default_factory=list)  # distinct link ids among the copied clips
    group_ids: list[str] = field(default_factory=list)  # distinct group ids among the copied clips
    span: float = 0.0  # time from the earliest start to the latest end
    version: int = CLIPBOARD_VERSION

    # ------------------------------------------------------------------ capture
    @classmethod
    def from_clips(cls, tl: Timeline, ids: list[str], *, linked: bool = True, grouped: bool = True) -> ClipboardData:
        all_ids = relatives(tl, list(ids), linked=linked, grouped=grouped)
        if not all_ids:
            raise TimelineError("Select at least one clip to copy.")
        order = {t.id: i for i, t in enumerate(tl.tracks)}
        pairs = sorted((tl.find_clip(i) for i in all_ids), key=lambda tc: (tc[1].timeline_start, order[tc[0].id], tc[1].id))
        t0 = min(c.timeline_start for _t, c in pairs)
        base = min(order[t.id] for t, _c in pairs)
        clips = [c for _t, c in pairs]
        return cls(
            clips=[c.to_dict() for c in clips],
            offsets=[c.timeline_start - t0 for c in clips],
            track_offsets=[order[t.id] - base for t, _c in pairs],
            track_kinds=[t.kind.value for t, _c in pairs],
            track_ids=[t.id for t, _c in pairs],
            asset_ids=sorted({c.asset_id for c in clips if c.kind == KIND_MEDIA and c.asset_id}),
            link_ids=sorted({c.link_id for c in clips if c.link_id}),
            group_ids=sorted({c.group_id for c in clips if c.group_id}),
            span=max(c.timeline_end for c in clips) - t0,
        )

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "clips": copy.deepcopy(self.clips), "offsets": list(self.offsets), "track_offsets": list(self.track_offsets),
            "track_kinds": list(self.track_kinds), "track_ids": list(self.track_ids), "asset_ids": list(self.asset_ids), "link_ids": list(self.link_ids),
            "group_ids": list(self.group_ids), "span": self.span,
        }

    @classmethod
    def from_dict(cls, d: Any) -> ClipboardData:
        try:
            clips = list(d["clips"])
            n = len(clips)
            out = cls(
                clips=copy.deepcopy(clips), offsets=[float(x) for x in d["offsets"]], track_offsets=[int(x) for x in d["track_offsets"]],
                track_kinds=[str(x) for x in d["track_kinds"]], track_ids=[str(x) for x in d.get("track_ids", [])] or [""] * n,
                asset_ids=[str(x) for x in d.get("asset_ids", [])], link_ids=[str(x) for x in d.get("link_ids", [])],
                group_ids=[str(x) for x in d.get("group_ids", [])], span=float(d.get("span", 0.0)), version=int(d.get("version", CLIPBOARD_VERSION)),
            )
            for lst in (out.offsets, out.track_offsets, out.track_kinds, out.track_ids):
                if len(lst) != n:
                    raise ValueError("length mismatch")
            if not n:
                raise ValueError("empty")
            for c in out.clips:
                Clip.from_dict(c)  # shape check
            return out
        except (KeyError, TypeError, ValueError, AttributeError):
            raise TimelineError("The clipboard does not contain timeline clips.") from None

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ClipboardData:
        try:
            return cls.from_dict(json.loads(text))
        except (TypeError, ValueError):
            raise TimelineError("The clipboard does not contain timeline clips.") from None


# ------------------------------------------------------------------------------------------------------------------------ paste
def _shifted_start(env: PlanEnv, base: float, offset: float, grid: Grid) -> float:
    if grid.audio:
        return grid.snap(base + offset)
    if grid.tb.is_on_frame(base) and grid.tb.is_on_frame(offset):
        return grid.tb.add_frames(base, grid.tb.to_frame(offset))
    return base + offset


@planner
def plan_paste(env: PlanEnv, data: ClipboardData, at: float, *, target_track_id: str | None = None, policy: str = "reject") -> EditPlan:
    """Paste ``data`` so that its earliest clip starts at ``at`` (frame grid; sample grid if every target is an audio track).

    Tracks: without ``target_track_id`` each clip goes back to the track it was copied from; with it, the topmost source track maps to the target and the others keep their relative
    track offsets - each destination must exist and hold that kind of clip, otherwise the paste is rejected with the reason. Clips whose media is no longer in the project are refused.
    ``policy``: ``"reject"`` refuses any overlap (the reason names the clips); ``"overwrite"`` trims / splits / removes what is in the way (see ``plan_overwrite``)."""
    if policy not in ("reject", "overwrite"):
        raise TimelineError("Choose whether a paste may overwrite existing clips.")
    tl = env.timeline
    if not data.clips:
        raise TimelineError("The clipboard is empty.")
    at = float(at)
    if not math.isfinite(at):
        raise TimelineError("The paste position is not a valid time.")
    srcs = [Clip.from_dict(d) for d in data.clips]
    missing = sorted({c.asset_id for c in srcs if c.kind == KIND_MEDIA and env.asset_info(c.asset_id) is None})
    if missing:
        raise TimelineError(f"{len(missing)} of the copied clips use media that is no longer in this project, so they cannot be pasted.")
    dest_ids = _destinations(env, data, srcs, target_track_id)
    audio_only = all(tl.get_track(t).kind is TrackKind.AUDIO for t in dest_ids)
    base = max(0.0, Grid(env, audio_only).snap(at))
    adds: list[Clip] = []
    for src, off, tid in zip(srcs, data.offsets, dest_ids):
        grid = Grid(env, tl.get_track(tid).kind is TrackKind.AUDIO)
        adds.append(fresh_copy(src, start=_shifted_start(env, base, off, grid), track_id=tid))
    remap_ids(adds, "link_id", new_link_id)
    remap_ids(adds, "group_id", new_group_id)
    desc = "Paste clip" if len(adds) == 1 else f"Paste {len(adds)} clips"
    if policy == "overwrite":
        plan = plan_overwrite(env, adds, description=desc)
        return plan
    plan = EditPlan(desc, adds=adds)
    validate_plan(env, plan)  # refuses overlaps, locked tracks and unsuitable tracks with the reason
    return plan


def _destinations(env: PlanEnv, data: ClipboardData, srcs: list[Clip], target_track_id: str | None) -> list[str]:
    tl = env.timeline
    out: list[str] = []
    ids = [t.id for t in tl.tracks]
    for i, src in enumerate(srcs):
        if target_track_id is None:
            tid = data.track_ids[i]
            if tid not in ids:
                raise TimelineError("A track the clips were copied from no longer exists. Choose a track to paste into.")
        else:
            if target_track_id not in ids:
                raise TimelineError("That track no longer exists.")
            idx = ids.index(target_track_id) + data.track_offsets[i]
            if not (0 <= idx < len(ids)):
                raise TimelineError("There are not enough tracks there for all the copied clips. Choose a higher track.")
            tid = ids[idx]
        track = tl.get_track(tid)
        info = env.asset_info(src.asset_id) if src.kind == KIND_MEDIA else None
        if not clip_fits_track(src, track, info[0] if info else None):
            raise TimelineError(f"Track “{track.name}” cannot hold {_what(src, info)}. Choose a different track.")
        out.append(tid)
    return out


def _what(clip: Clip, info) -> str:
    if clip.kind == KIND_MEDIA:
        return f"{info[0]} media" if info else "this media"
    return {"caption": "captions", "text": "text", "graphic": "graphics"}.get(clip.kind, "this clip")


# ------------------------------------------------------------------------------------------------------------------------ attributes
def copy_attributes(clip: Clip) -> dict[str, Any]:
    """The look of a clip as plain data: transform (position, scale, rotation), opacity, effects (without the source-specific crop / focus region) and audio volume / fades / pan.
    Speed is NOT included (it changes the duration), nor are timing, source range, text or keyframes."""
    audio = {k: clip.audio[k] for k in AUDIO_ATTRIBUTES if k in clip.audio}
    return {
        "position": [clip.position[0], clip.position[1]], "scale": clip.scale, "rotation": clip.rotation, "opacity": clip.opacity,
        "effects": {k: copy.deepcopy(v) for k, v in clip.effects.items() if k not in SOURCE_SPECIFIC_EFFECTS},
        "audio": audio,
    }


@planner
def plan_paste_attributes(env: PlanEnv, ids: list[str], attrs: dict[str, Any]) -> EditPlan:
    """Apply copied attributes to clips. Audio-track clips only take the audio part; stills and text take no audio; the target's own crop / focus region is kept; fades are scaled
    down (noted) when together they would be longer than the target clip. Clips that would not change are left out."""
    tl = env.timeline
    updates: dict[str, Clip] = {}
    notes: list[str] = []
    for cid in dict.fromkeys(ids):
        track, clip = tl.find_clip(cid)
        n = clip.snapshot()
        is_audio = track.kind is TrackKind.AUDIO
        if not is_audio:
            if "position" in attrs:
                n.position = (float(attrs["position"][0]), float(attrs["position"][1]))
            for k in ("scale", "rotation", "opacity"):
                if k in attrs:
                    setattr(n, k, float(attrs[k]))
        if isinstance(attrs.get("effects"), dict) and not is_audio:
            keep = {k: v for k, v in clip.effects.items() if k in SOURCE_SPECIFIC_EFFECTS}
            n.effects = {**copy.deepcopy(attrs["effects"]), **keep}
        info = env.asset_info(clip.asset_id) if clip.kind == KIND_MEDIA else None
        has_audio = clip.kind == KIND_MEDIA and info is not None and info[0] in ("video", "audio")
        if isinstance(attrs.get("audio"), dict) and has_audio:
            for k, v in attrs["audio"].items():
                if k in AUDIO_ATTRIBUTES:
                    n.audio[k] = v
            fi, fo = float(n.audio.get("fade_in", 0.0) or 0.0), float(n.audio.get("fade_out", 0.0) or 0.0)
            if fi + fo > n.duration + TIME_EPSILON:
                f = n.duration / (fi + fo)
                n.audio["fade_in"], n.audio["fade_out"] = round(fi * f, 6), round(fo * f, 6)
                notes.append(f"The fades were shortened to fit “{env.label(clip)}”.")
        if n.to_dict() != clip.to_dict():
            updates[cid] = n
    if not updates:
        return EditPlan("Paste attributes", notes=[*notes, "Nothing changed."])
    return EditPlan("Paste attributes" if len(updates) > 1 else "Paste attributes onto clip", updates=updates, notes=notes)
