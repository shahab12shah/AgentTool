"""Caption-specific planners that are not generic clip edits (merge). Splitting a caption is ``ops_basic.plan_split`` (it splits between words)."""

from __future__ import annotations

import copy

from app.core.exceptions import TimelineError
from app.timeline.clip import KIND_CAPTION
from app.timeline.ops_basic import sync_text_timing
from app.timeline.plan import EditPlan, PlanEnv

MAX_MERGE_GAP = 1.5  # seconds: captions further apart than this are not "one sentence on screen"


def plan_merge_captions(env: PlanEnv, ids: list[str]) -> EditPlan:
    """Merge caption clips on ONE track into the first one. They must be neighbours on the track (no other clip between them), carry word timing, and be less than 1.5 s apart.

    The merged clip spans the first start to the last end; words, lines and emphasis marks are concatenated (emphasis indexes shifted). The merged text may need more lines than the
    caption style allows - the Captions page re-wraps it. Manual merging marks the result as the user's (the service's edit hook records the takeover)."""
    tl = env.timeline
    clips = [tl.find_clip(i)[1] for i in dict.fromkeys(ids)]
    if len(clips) < 2:
        raise TimelineError("Select at least two captions to merge.")
    if any(c.kind != KIND_CAPTION for c in clips):
        raise TimelineError("Only caption clips can be merged.")
    if len({c.track_id for c in clips}) != 1:
        raise TimelineError("The captions to merge must be on the same track.")
    clips.sort(key=lambda c: c.timeline_start)
    track = tl.get_track(clips[0].track_id)
    order = [c.id for c in track.clips]
    pos = [order.index(c.id) for c in clips]
    if pos != list(range(pos[0], pos[0] + len(pos))):
        raise TimelineError("There is another clip between the selected captions, so they cannot be merged.")
    for a, b in zip(clips, clips[1:]):
        if b.timeline_start - a.timeline_end > MAX_MERGE_GAP:
            raise TimelineError(f"The captions are {b.timeline_start - a.timeline_end:.1f} s apart; merge only captions that follow each other closely.")
    for c in clips:
        t = c.text
        if not (isinstance(t, dict) and isinstance(t.get("words"), list) and t["words"]):
            raise TimelineError(f"“{env.label(c)}” has no word timing, so it cannot be merged.")
    first = clips[0].snapshot()
    text = copy.deepcopy(first.text)
    words = list(text["words"])
    lines = list(text.get("lines") or [])
    emphasis = list(text.get("emphasis") or [])
    notes: list[str] = []
    for c in clips[1:]:
        t = c.text
        offset = len(words)
        words += copy.deepcopy(t["words"])
        lines += list(t.get("lines") or [str(t.get("text", ""))])
        emphasis += [{**copy.deepcopy(m), "word_index": int(m.get("word_index", 0)) + offset} for m in (t.get("emphasis") or []) if isinstance(m, dict)]
    joined = " ".join(str(w.get("text", "")) for w in words)
    text["words"], text["text"] = words, joined
    if "lines" in text or lines:
        text["lines"] = lines or [joined]
    if emphasis:
        text["emphasis"] = emphasis
    first.text = text
    first.duration = clips[-1].timeline_end - first.timeline_start
    first.source_out = first.source_in + first.duration * first.speed
    sync_text_timing(first)
    if len(text.get("lines") or []) > 2:
        notes.append("The merged caption has more than two lines; re-wrap it on the Captions page if needed.")
    return EditPlan(f"Merge {len(clips)} captions", updates={first.id: first}, removes=[c.id for c in clips[1:]], notes=notes)
