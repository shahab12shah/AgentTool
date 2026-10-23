"""Helpers shared by the edit planners (``ops_*.py``): link / group expansion, frame quantisation, fresh ids.

Vocabulary (kept distinct on purpose):
  * LINKED  (``Clip.link_id``)   edits that are about timing apply to every member: move, trim, split, delete, ripple (a picture and its production audio).
  * GROUPED (``Clip.group_id``)  members are selected and moved together; trims and splits stay independent.
  * LOCKED  (``Clip.locks``, ``Track.locked``)  stops manual edits; a locked clip is also protected from AI regeneration.
"""

from __future__ import annotations

import uuid
from typing import Iterable

from app.timeline.clip import Clip
from app.timeline.plan import PlanEnv
from app.timeline.timeline import Timeline
from app.timeline.track import TrackKind


def new_link_id() -> str:
    return f"link_{uuid.uuid4().hex[:8]}"


def new_group_id() -> str:
    return f"group_{uuid.uuid4().hex[:8]}"


def relatives(tl: Timeline, ids: Iterable[str], *, linked: bool = True, grouped: bool = True) -> list[str]:
    """``ids`` (input order) followed by every other clip that shares a link id (when ``linked``) or a group id (when ``grouped``) with any of them. Unknown ids raise ``TimelineError``."""
    wanted = list(dict.fromkeys(ids))
    clips = {cid: tl.find_clip(cid)[1] for cid in wanted}
    if not (linked or grouped):
        return wanted
    links = {c.link_id for c in clips.values() if linked and c.link_id}
    groups = {c.group_id for c in clips.values() if grouped and c.group_id}
    if not (links or groups):
        return wanted
    seen = set(wanted)
    for c in tl.all_clips():
        if c.id not in seen and ((c.link_id and c.link_id in links) or (c.group_id and c.group_id in groups)):
            wanted.append(c.id)
            seen.add(c.id)
    return wanted


def link_members(tl: Timeline, link_id: str) -> list[Clip]:
    return [c for c in tl.all_clips() if c.link_id == link_id] if link_id else []


def group_members(tl: Timeline, group_id: str) -> list[Clip]:
    return [c for c in tl.all_clips() if c.group_id == group_id] if group_id else []


def is_audio_track(tl: Timeline, track_id: str) -> bool:
    return tl.get_track(track_id).kind is TrackKind.AUDIO


def quantize(env: PlanEnv, t: float, *, audio: bool = False, sample_rate: int = 48000, mode: str = "round") -> float:
    """Snap a time to the project's frame grid (picture, captions, graphics) or to the audio sample grid (audio clips keep sample precision, not frame precision)."""
    return env.timebase.snap_samples(t, sample_rate) if audio else env.timebase.snap(t, mode)


def quantize_delta(env: PlanEnv, delta: float, *, audio: bool = False) -> float:
    """A time *difference* rounded to whole frames (or left alone for audio-only edits), so frame-aligned clips stay frame-aligned and relative spacing is unchanged."""
    if audio:
        return delta
    n = env.timebase.to_frame(abs(delta), "round")
    return env.timebase.to_seconds(n) * (1 if delta >= 0 else -1)
