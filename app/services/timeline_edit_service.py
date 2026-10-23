"""TimelineEditService (``ws.edit``): the one API the timeline UI uses for professional editing.

Every method turns a user intent into an ``EditPlan`` with the pure planners in ``app/timeline/ops_*.py``, runs it as ONE undoable ``PlanCommand`` (validate -> capture -> atomic
apply -> rollback on failure) and reports what happened. The UI never edits clips itself and never keeps timing of its own: drag feedback comes from ``preview_*`` (a dry run of the same
planner + validator, nothing is changed), and the gesture commits once on release.

AI ownership: a manual edit of a clip made by the AI runs the same edit hook as the older timeline service (``TimelineService.edit_hook``), recorded in the same undo step, so a later
AI regeneration keeps the user's change. Clips that are merely *shifted* by a ripple are not taken over - only the clips the user actually edited.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from app.core.commands import Command, CommandStack, CompositeCommand
from app.core.constants import DEFAULT_IMAGE_DURATION
from app.core.events import EventBus, Topics
from app.core.exceptions import TimelineError
from app.logging.logger import get_logger
from app.media.asset import Asset
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.timeline import meta_commands as mc
from app.timeline import ops_basic as ob
from app.timeline import ops_props as op
from app.timeline import ops_ripple as orp
from app.timeline.clipboard import ClipboardData, copy_attributes, plan_paste, plan_paste_attributes
from app.timeline.clip import Clip
from app.timeline.markers import Marker
from app.timeline.ops_common import relatives
from app.timeline.ops_text import plan_merge_captions
from app.timeline.plan import EditPlan, PlanCommand, PlanEnv, validate_plan
from app.timeline.snapping import SnapEngine
from app.timeline.timebase import Timebase, timebase_of
from app.timeline.timeline import Timeline, new_clip_id
from app.timeline.track import Track, TrackKind

_log = get_logger(__name__)


class TimelineEditService:
    def __init__(self, projects: ProjectManager, commands: CommandStack, bus: EventBus, selection, edit_state, legacy) -> None:
        self._projects, self._commands, self._bus = projects, commands, bus
        self.selection, self.state = selection, edit_state
        self._legacy = legacy  # the Phase 2 TimelineService: owns edit_hook (AI ownership), track naming and the default-track rules
        self.clipboard: ClipboardData | None = None
        self.playhead = 0.0  # the one playhead (the canvas reads and writes it; the snap engine reads it)
        self.last_notes: list[str] = []
        self._empty = Timeline()
        # the one snapping engine: targets are cached per timeline revision, so a drag never rescans the timeline
        self.snap = SnapEngine(lambda: self.timeline if self._projects.current else self._empty, lambda: self._projects.current, lambda: self.playhead, lambda: self.state.snap,
                               lambda: timebase_of(self._projects.current) if self._projects.current else Timebase(30))

    # ------------------------------------------------------------------ plumbing
    def _project(self) -> Project:
        if self._projects.current is None:
            raise TimelineError("Open or create a project first.")
        return self._projects.current

    @property
    def timeline(self) -> Timeline:
        return self._project().timeline

    def env(self) -> PlanEnv:
        return PlanEnv.for_project(self._project())

    def selected(self) -> list[str]:
        return [i for i in self.selection.ids if self.timeline.get_clip(i) is not None]

    def _ids(self, ids: Iterable[str] | None) -> list[str]:
        out = list(ids) if ids is not None else self.selected()
        if not out:
            raise TimelineError("Select a clip first.")
        return out

    def set_playhead(self, t: float) -> None:
        self.playhead = max(0.0, float(t))
        self._bus.publish("timeline.playhead", time=self.playhead)

    def run(self, plan: EditPlan, *, own: Iterable[str] = (), action: str = "edit") -> EditPlan | None:
        """Execute a plan as one undo step. ``own`` are the clips the user directly edited (AI ownership is recorded for those only). Empty plans do nothing."""
        self.last_notes = list(plan.notes)
        if plan.empty:
            if plan.notes:
                self._bus.publish(Topics.STATUS, message=plan.notes[0])
            return None
        cmd: Command = PlanCommand(self.env(), plan)
        hook = self._legacy.edit_hook
        if hook is not None:
            extras = []
            existing = {c.id for c in self.timeline.all_clips()}
            for cid in dict.fromkeys(own):
                if cid in existing:
                    h = hook(cid, "delete" if cid in plan.removes else action)
                    if h is not None:
                        extras.append(h)
            if extras:
                cmd = CompositeCommand(plan.description, [cmd, *extras], scope="timeline")
        self._commands.execute(cmd)
        note = "; ".join(plan.notes[:2])
        if note:
            self._bus.publish(Topics.STATUS, message=note)
        return plan

    def check(self, plan: EditPlan) -> str | None:
        """Dry run: None when the plan would apply, otherwise the plain-language reason it would be refused (nothing is changed)."""
        try:
            validate_plan(self.env(), plan)
            return None
        except TimelineError as exc:
            return exc.user_message

    def _try(self, build: Callable[[], EditPlan]) -> tuple[EditPlan | None, str | None]:
        try:
            plan = build()
        except TimelineError as exc:
            return None, exc.user_message
        return plan, self.check(plan)

    def _scope_tracks(self, ids: Iterable[str]) -> list[str]:
        tl = self.timeline
        return sorted({tl.find_clip(i)[0].id for i in ids})

    def _expand(self, ids: Iterable[str]) -> list[str]:
        return relatives(self.timeline, list(ids), linked=self.state.linked_selection, grouped=self.state.linked_selection)

    # ------------------------------------------------------------------ move / trim / numeric
    def preview_move(self, ids: Iterable[str], delta: float, track_moves: dict[str, str] | None = None) -> tuple[EditPlan | None, str | None]:
        ids = list(ids)
        return self._try(lambda: ob.plan_move(self.env(), ids, delta, track_moves=track_moves, linked=self.state.linked_selection, grouped=self.state.linked_selection))

    def move_clips(self, ids: Iterable[str] | None, delta: float, track_moves: dict[str, str] | None = None, *, ripple: bool | None = None) -> EditPlan | None:
        ids = self._ids(ids)
        env, linked = self.env(), self.state.linked_selection
        ripple = self.state.ripple_on if ripple is None else ripple
        if ripple and not track_moves and len(self._scope_tracks(ids)) == 1:
            plan = orp.plan_ripple_move(env, ids, delta, scope=self.state.ripple_scope, selected_track_ids=self._scope_tracks(ids), linked=linked, grouped=linked)
        else:
            plan = ob.plan_move(env, ids, delta, track_moves=track_moves, linked=linked, grouped=linked)
        return self.run(plan, own=ids)

    def set_start(self, clip_id: str, new_start: float) -> EditPlan | None:
        return self.run(ob.plan_set_start(self.env(), clip_id, new_start, linked=self.state.linked_selection), own=[clip_id])

    def set_duration(self, clip_id: str, duration: float) -> EditPlan | None:
        return self.run(ob.plan_set_duration(self.env(), clip_id, duration, linked=self.state.linked_selection), own=[clip_id])

    def set_source_range(self, clip_id: str, source_in: float | None = None, source_out: float | None = None) -> EditPlan | None:
        return self.run(ob.plan_set_source_range(self.env(), clip_id, source_in, source_out), own=[clip_id])

    def preview_trim(self, clip_id: str, edge: str, new_time: float) -> tuple[EditPlan | None, str | None]:
        return self._try(lambda: self._trim_plan(clip_id, edge, new_time, None))

    def _trim_plan(self, clip_id: str, edge: str, new_time: float, ripple: bool | None) -> EditPlan:
        env, linked = self.env(), self.state.linked_selection
        if self.state.ripple_on if ripple is None else ripple:
            return orp.plan_ripple_trim(env, clip_id, edge, new_time, scope=self.state.ripple_scope, selected_track_ids=self._scope_tracks([clip_id]), linked=linked, grouped=linked)
        return ob.plan_trim(env, clip_id, edge, new_time, linked=linked)

    def trim(self, clip_id: str, edge: str, new_time: float, *, ripple: bool | None = None) -> EditPlan | None:
        return self.run(self._trim_plan(clip_id, edge, new_time, ripple), own=[clip_id])

    # ------------------------------------------------------------------ split / delete / duplicate / clipboard
    def split(self, at: float | None = None, ids: Iterable[str] | None = None) -> EditPlan | None:
        """Split at ``at`` (default: the playhead). With ids (or a selection): those clips (and linked ones). With neither: every clip under the cut on unlocked tracks."""
        at = self.playhead if at is None else at
        tl, env = self.timeline, self.env()
        wanted = list(ids) if ids is not None else self.selected()
        everywhere = [t.id for t in tl.tracks if not t.locked]
        if wanted:
            under = [i for i in wanted if tl.get_clip(i) and tl.get_clip(i).timeline_start < at < tl.get_clip(i).timeline_end]
            plan = ob.plan_split(env, under or wanted, at, linked=self.state.linked_selection)
        else:
            plan = ob.plan_split(env, [], at, linked=self.state.linked_selection, include_intersecting_on=everywhere)
        return self.run(plan, own=list(plan.updates))

    def blade(self, at: float, track_id: str | None = None) -> EditPlan | None:
        """The blade tool: cut the clip(s) under ``at`` on one track (and linked clips), or on every unlocked track when ``track_id`` is None."""
        tl = self.timeline
        tracks = [track_id] if track_id else [t.id for t in tl.tracks if not t.locked]
        plan = ob.plan_split(self.env(), [], at, linked=self.state.linked_selection, include_intersecting_on=tracks)
        return self.run(plan, own=list(plan.updates))

    def delete(self, ids: Iterable[str] | None = None, *, ripple: bool | None = None) -> EditPlan | None:
        ids = self._ids(ids)
        env, linked = self.env(), self.state.linked_selection
        if self.state.ripple_on if ripple is None else ripple:
            plan = orp.plan_ripple_delete(env, ids, scope=self.state.ripple_scope, selected_track_ids=self._scope_tracks(ids), linked=linked, grouped=linked)
        else:
            plan = ob.plan_delete(env, ids, linked=linked, grouped=linked)
        return self.run(plan, own=ids, action="delete")

    def duplicate(self, ids: Iterable[str] | None = None, *, offset: float | None = None, to_start: float | None = None) -> EditPlan | None:
        linked = self.state.linked_selection
        return self.run(ob.plan_duplicate(self.env(), self._ids(ids), offset=offset, to_start=to_start, linked=linked, grouped=linked))

    def copy(self, ids: Iterable[str] | None = None) -> ClipboardData:
        ids = self._ids(ids)
        linked = self.state.linked_selection
        self.clipboard = ClipboardData.from_clips(self.timeline, ids, linked=linked, grouped=linked)
        return self.clipboard

    def cut(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        self.copy(ids)
        return self.delete(ids, ripple=False)

    def paste(self, at: float | None = None, track_id: str | None = None, *, policy: str = "reject") -> EditPlan | None:
        if self.clipboard is None:
            raise TimelineError("There is nothing to paste: copy or cut clips first.")
        return self.run(plan_paste(self.env(), self.clipboard, self.playhead if at is None else at, target_track_id=track_id, policy=policy))

    def copy_attributes(self, clip_id: str) -> dict[str, Any]:
        self._attrs = copy_attributes(self.timeline.find_clip(clip_id)[1])
        return self._attrs

    def paste_attributes(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        attrs = getattr(self, "_attrs", None)
        if not attrs:
            raise TimelineError("There are no copied attributes: use 'Copy attributes' on a clip first.")
        return self.run(plan_paste_attributes(self.env(), self._ids(ids), attrs), own=self._ids(ids))

    def merge_captions(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        ids = self._ids(ids)
        return self.run(plan_merge_captions(self.env(), ids), own=ids)

    # ------------------------------------------------------------------ media placement
    def clips_for_assets(self, asset_ids: Iterable[str], track: Track, start: float) -> list[Clip]:
        """New clips for library assets, laid out one after another from ``start`` on ``track`` (positions are fixed up by the placement planner)."""
        project, out, t = self._project(), [], start
        for aid in asset_ids:
            asset = project.assets.require(aid)
            if not track.accepts(asset.type):
                raise TimelineError(f"Track “{track.name}” cannot hold {asset.type.value} media.")
            dur = asset.duration or DEFAULT_IMAGE_DURATION
            out.append(Clip(new_clip_id(), track.id, asset.id, t, dur, 0.0, dur))
            t += dur
        return out

    def default_track(self, asset: Asset) -> Track:
        return self._legacy.default_track_for(asset)

    def place_assets(self, asset_ids: Iterable[str], at: float | None = None, track_id: str | None = None, *, mode: str = "free") -> EditPlan | None:
        """Put library media on the timeline. ``mode``: ``free`` (first free spot on the track, nothing moves) | ``insert`` (ripple: later clips move right) | ``overwrite``."""
        asset_ids = list(asset_ids)
        if not asset_ids:
            return None
        project, tl = self._project(), self.timeline
        at = self.playhead if at is None else at
        track = tl.get_track(track_id) if track_id else self.default_track(project.assets.require(asset_ids[0]))
        clips = self.clips_for_assets(asset_ids, track, at)
        env = self.env()
        if mode == "insert":
            plan = orp.plan_ripple_insert(env, clips, at, track.id, scope=self.state.ripple_scope if self.state.ripple_on else "track", selected_track_ids=[track.id])
        elif mode == "overwrite":
            plan = ob.plan_overwrite(env, clips)
        else:
            plan = ob.plan_insert_free(env, clips, at, track.id)
        plan.description = "Add media" if mode == "free" else f"{mode.title()} media"
        return self.run(plan)

    def replace_clip(self, clip_id: str, asset_id: str, *, keep_duration: bool = True) -> EditPlan | None:
        return self.run(ob.plan_replace(self.env(), clip_id, asset_id, keep_duration=keep_duration), own=[clip_id])

    # ------------------------------------------------------------------ advanced trims
    def roll_limits(self, left_id: str, right_id: str) -> tuple[float, float]:
        return orp.roll_limits(self.env(), left_id, right_id)

    def slip_limits(self, clip_id: str) -> tuple[float, float]:
        return orp.slip_limits(self.env(), clip_id)

    def slide_limits(self, clip_id: str) -> tuple[float, float]:
        return orp.slide_limits(self.env(), clip_id)

    def roll(self, left_id: str, right_id: str, boundary: float) -> EditPlan | None:
        return self.run(orp.plan_roll(self.env(), left_id, right_id, boundary, linked=self.state.linked_selection), own=[left_id, right_id])

    def slip(self, clip_id: str, delta_source: float) -> EditPlan | None:
        return self.run(orp.plan_slip(self.env(), clip_id, delta_source, linked=self.state.linked_selection), own=[clip_id])

    def slide(self, clip_id: str, delta: float) -> EditPlan | None:
        return self.run(orp.plan_slide(self.env(), clip_id, delta, linked=self.state.linked_selection), own=[clip_id])

    # ------------------------------------------------------------------ relationships and locks
    def link(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        return self.run(op.plan_link(self.env(), self._ids(ids)))

    def unlink(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        return self.run(op.plan_unlink(self.env(), self._ids(ids)))

    def group(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        return self.run(op.plan_group(self.env(), self._ids(ids)))

    def ungroup(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        return self.run(op.plan_ungroup(self.env(), self._ids(ids)))

    def set_locks(self, ids: Iterable[str] | None, locks: Iterable[str] | str, mode: str = "set") -> EditPlan | None:
        return self.run(op.plan_set_locks(self.env(), self._ids(ids), locks, mode))

    def toggle_item_lock(self, ids: Iterable[str] | None = None) -> EditPlan | None:
        ids = self._ids(ids)
        tl = self.timeline
        all_locked = all("item" in tl.find_clip(i)[1].locks for i in ids)
        return self.set_locks(ids, "item", "remove" if all_locked else "add")

    # ------------------------------------------------------------------ properties / speed / transitions / keyframes
    def update_clip(self, clip_id: str, **changes: Any) -> EditPlan | None:
        return self.run(op.plan_update_clip(self.env(), clip_id, **changes), own=[clip_id])

    def update_clips(self, ids: Iterable[str], **changes: Any) -> EditPlan | None:
        """Bulk change of a compatible property over several clips (one undo step). Clips the property does not apply to are skipped with a note."""
        ids = list(ids)
        merged: EditPlan | None = None
        skipped: list[str] = []
        env = self.env()
        for cid in ids:
            try:
                p = op.plan_update_clip(env, cid, **changes)
            except TimelineError as exc:
                skipped.append(str(exc))
                continue
            if not p.empty:
                merged = p if merged is None else merged.merge(p)
        if merged is None:
            raise TimelineError(skipped[0] if skipped else "Nothing to change.")
        if skipped:
            merged.notes.append(f"{len(skipped)} selected clip(s) were skipped: {skipped[0]}")
        merged.description = "Edit clip properties"
        return self.run(merged, own=ids)

    def set_speed(self, clip_id: str, speed: float, *, link_policy: str | None = None) -> EditPlan | None:
        return self.run(op.plan_set_speed(self.env(), clip_id, speed, linked=self.state.linked_selection, link_policy=link_policy), own=[clip_id])

    def set_transition(self, clip_id: str, type_: str | None, duration: float | None = None, params: dict | None = None) -> EditPlan | None:
        return self.run(op.plan_set_transition(self.env(), clip_id, type_, duration, params), own=[clip_id])

    def remove_transition(self, clip_id: str) -> EditPlan | None:
        return self.run(op.plan_remove_transition(self.env(), clip_id), own=[clip_id])

    def add_keyframe(self, clip_id: str, prop: str, time: float, value: float, interpolation: str = "linear") -> EditPlan | None:
        return self.run(op.plan_add_keyframe(self.env(), clip_id, prop, time, value, interpolation), own=[clip_id])

    def delete_keyframes(self, clip_id: str, selection: Iterable[tuple[str, float]]) -> EditPlan | None:
        return self.run(op.plan_delete_keyframes(self.env(), clip_id, selection), own=[clip_id])

    def move_keyframes(self, clip_id: str, moves: Iterable[Iterable[Any]]) -> EditPlan | None:
        return self.run(op.plan_move_keyframes(self.env(), clip_id, moves), own=[clip_id])

    def set_keyframe_interpolation(self, clip_id: str, selection: Iterable[tuple[str, float]], interp: str) -> EditPlan | None:
        return self.run(op.plan_set_keyframe_interpolation(self.env(), clip_id, selection, interp), own=[clip_id])

    def copy_keyframes(self, clip_id: str, selection: Iterable[tuple[str, float]] | None = None) -> list:
        self._kf_clip = op.plan_copy_keyframes(self.timeline.find_clip(clip_id)[1], selection)
        return self._kf_clip

    def paste_keyframes(self, clip_id: str, at_time: float) -> EditPlan | None:
        kfs = getattr(self, "_kf_clip", None)
        if not kfs:
            raise TimelineError("There are no copied keyframes.")
        return self.run(op.plan_paste_keyframes(self.env(), clip_id, kfs, at_time), own=[clip_id])

    # ------------------------------------------------------------------ markers and range
    def add_marker(self, time: float | None = None, name: str = "", kind: str = "generic", color: str = "", note: str = "") -> str:
        t = self.playhead if time is None else time
        t = self.env().timebase.snap(t)
        cmd = mc.AddMarkerCommand(self.timeline, t, name, kind, color, note)
        self._commands.execute(cmd)
        return cmd.marker_id

    def move_marker(self, marker_id: str, time: float) -> None:
        self._commands.execute(mc.MoveMarkerCommand(self.timeline, marker_id, self.env().timebase.snap(time)))

    def edit_marker(self, marker_id: str, **fields: str) -> None:
        self._commands.execute(mc.EditMarkerCommand(self.timeline, marker_id, **fields))

    def delete_marker(self, marker_id: str) -> None:
        self._commands.execute(mc.DeleteMarkerCommand(self.timeline, marker_id))

    def next_marker(self, t: float | None = None) -> Marker | None:
        return self.timeline.next_marker(self.playhead if t is None else t)

    def prev_marker(self, t: float | None = None) -> Marker | None:
        return self.timeline.prev_marker(self.playhead if t is None else t)

    def set_range(self, start: float, end: float) -> None:
        tb = self.env().timebase
        self._commands.execute(mc.SetRangeCommand(self.timeline, tb.snap(start), tb.snap(end)))

    def clear_range(self) -> None:
        self._commands.execute(mc.ClearRangeCommand(self.timeline))

    def mark_in(self, t: float | None = None) -> None:
        """The ``I`` key: set the range start at the playhead (keeps the end if there is a later one)."""
        t = self.playhead if t is None else t
        r = self.timeline.range
        self.set_range(t, r[1] if r and r[1] > t else t + 1.0)

    def mark_out(self, t: float | None = None) -> None:
        """The ``O`` key: set the range end at the playhead (keeps the start if there is an earlier one)."""
        t = self.playhead if t is None else t
        r = self.timeline.range
        self.set_range(r[0] if r and r[0] < t else max(0.0, t - 1.0), t)

    # ------------------------------------------------------------------ tracks
    def add_track(self, kind: TrackKind = TrackKind.VIDEO, name: str | None = None) -> Track:
        return self._legacy.add_track(kind, name)

    def move_track(self, track_id: str, new_index: int) -> None:
        self._commands.execute(mc.MoveTrackCommand(self.timeline, track_id, new_index))

    def set_track_display(self, track_id: str, height: int | None = None, collapsed: bool | None = None) -> None:
        self._commands.execute(mc.SetTrackDisplayCommand(self.timeline, track_id, height=height, collapsed=collapsed))

    def duplicate_track(self, track_id: str, *, include_clips: bool = True) -> str:
        new_id, id_map = mc.make_duplicate_ids(self.timeline, track_id)
        self._commands.execute(mc.DuplicateTrackCommand(self.timeline, track_id, new_id, id_map, include_clips))
        return new_id

    def set_track_kind(self, track_id: str, kind: TrackKind) -> None:
        self._commands.execute(mc.SetTrackKindCommand(self.timeline, track_id, kind, self.env().asset_info))

    def remove_track(self, track_id: str, *, confirm_content: bool = False) -> None:
        """Delete a track. A track that holds clips raises ``TrackHasContent`` (with ``clip_count``) until the caller confirms; undo brings back the track AND its clips."""
        self._commands.execute(mc.RemoveTrackSafeCommand(self.timeline, track_id, confirm_content=confirm_content))

    def set_track_flag(self, track_id: str, flag: str, value: bool) -> None:
        self._legacy.set_track_flag(track_id, flag, value)

    def rename_track(self, track_id: str, name: str) -> None:
        self._legacy.rename_track(track_id, name)
