"""Application service for timeline editing. All edits go through the undo stack."""

from __future__ import annotations

from app.core.commands import CommandStack
from app.core.constants import DEFAULT_IMAGE_DURATION
from app.core.exceptions import TimelineError
from app.media.asset import Asset, AssetType
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.timeline.clip import Clip
from app.timeline.timeline import Timeline, new_clip_id, new_track_id
from app.timeline.timeline_commands import (
    AddClipCommand,
    AddTrackCommand,
    DeleteClipCommand,
    MoveClipCommand,
    RemoveTrackCommand,
    RenameTrackCommand,
    SetClipPropertiesCommand,
    SetTrackFlagCommand,
    TrimClipCommand,
)
from app.timeline.track import Track, TrackKind


class TimelineService:
    def __init__(self, projects: ProjectManager, commands: CommandStack) -> None:
        self._projects = projects
        self._commands = commands

    def _project(self) -> Project:
        if self._projects.current is None:
            raise TimelineError("Open or create a project first.")
        return self._projects.current

    @property
    def timeline(self) -> Timeline:
        return self._project().timeline

    # ----- tracks -----
    def add_track(self, kind: TrackKind = TrackKind.VIDEO, name: str | None = None) -> Track:
        tl = self.timeline
        if name is None:
            if kind is TrackKind.AUDIO:
                name = f"A{sum(t.is_audio for t in tl.tracks) + 1} Audio"
            else:
                label = {TrackKind.IMAGE: "Images", TrackKind.GRAPHICS: "Graphics", TrackKind.TEXT: "Text"}.get(kind, "Video")
                name = f"V{sum(not t.is_audio for t in tl.tracks) + 1} {label}"
        track = Track(new_track_id(), name, kind)
        # keep video-like tracks above audio tracks
        index = len(tl.tracks) if kind is TrackKind.AUDIO else sum(not t.is_audio for t in tl.tracks)
        self._commands.execute(AddTrackCommand(tl, track, index))
        return track

    def remove_track(self, track_id: str) -> None:
        self._commands.execute(RemoveTrackCommand(self.timeline, track_id))

    def rename_track(self, track_id: str, name: str) -> None:
        self._commands.execute(RenameTrackCommand(self.timeline, track_id, name))

    def set_track_flag(self, track_id: str, flag: str, value: bool) -> None:
        self._commands.execute(SetTrackFlagCommand(self.timeline, track_id, flag, value))

    # ----- clips -----
    def default_track_for(self, asset: Asset) -> Track:
        project = self._project()
        tracks = project.timeline.tracks
        usable = [t for t in tracks if t.accepts(asset.type) and not t.locked]
        if not usable:
            raise TimelineError("There is no unlocked track that can hold this kind of media. Add or unlock a track.")
        if asset.type is AssetType.IMAGE:
            pick = next((t for t in usable if t.kind is TrackKind.IMAGE), None)
        elif asset.type is AssetType.AUDIO:
            audio = [t for t in usable if t.is_audio]
            is_vo = project.voice_over.asset_id == asset.id
            pick = audio[0] if is_vo or len(audio) < 2 else audio[1]
        else:
            pick = next((t for t in usable if t.kind is TrackKind.VIDEO), None)
        return pick or usable[0]

    def add_asset(self, asset_id: str, track_id: str | None = None, start: float | None = None) -> Clip:
        project = self._project()
        asset = project.assets.require(asset_id)
        track = project.timeline.get_track(track_id) if track_id else self.default_track_for(asset)
        if not track.accepts(asset.type):
            raise TimelineError(f"Track “{track.name}” cannot hold {asset.type.value} media.")
        duration = asset.duration or DEFAULT_IMAGE_DURATION
        if start is None:
            start = max((c.timeline_end for c in track.clips), default=0.0)
        start = project.timeline.first_free_start(track.id, start, duration)
        clip = Clip(
            id=new_clip_id(), track_id=track.id, asset_id=asset.id,
            timeline_start=start, duration=duration, source_in=0.0, source_out=duration,
        )
        self._commands.execute(AddClipCommand(project.timeline, clip))
        return clip

    def move_clip(self, clip_id: str, new_start: float, new_track_id: str | None = None) -> None:
        project = self._project()
        if new_track_id is not None:
            track = project.timeline.get_track(new_track_id)
            _, clip = project.timeline.find_clip(clip_id)
            if not track.accepts(project.assets.require(clip.asset_id).type):
                raise TimelineError(f"Track “{track.name}” cannot hold that kind of media.")
        self._commands.execute(MoveClipCommand(project.timeline, clip_id, new_start, new_track_id))

    def trim_clip(self, clip_id: str, *, new_start: float | None = None, new_end: float | None = None) -> None:
        project = self._project()
        _, clip = project.timeline.find_clip(clip_id)
        asset = project.assets.require(clip.asset_id)
        max_source = None if asset.type is AssetType.IMAGE else asset.duration
        self._commands.execute(
            TrimClipCommand(project.timeline, clip_id, new_start=new_start, new_end=new_end, max_source=max_source)
        )

    def delete_clip(self, clip_id: str) -> None:
        self._commands.execute(DeleteClipCommand(self.timeline, clip_id))

    def set_clip_properties(self, clip_id: str, **changes: object) -> None:
        self._commands.execute(SetClipPropertiesCommand(self.timeline, clip_id, **changes))
