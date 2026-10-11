"""ProjectChangeTracker: command -> affected scenes / domains / dependency ids, neighbours, coalescing, undo/redo, project switch, cache invalidation that touches only dependants,
and the preview-section keys that decide what a preview re-renders."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.core.commands import Command, CommandStack, CompositeCommand
from app.core.events import EventBus, Topics
from app.media.asset import Asset, AssetType, SourceType
from app.performance import change_tracker as ct
from app.performance.cache_manager import MediaCacheManager
from app.performance.change_tracker import BROAD_TIMELINE, QC_DOMAINS, ProjectChangeTracker, scene_layout_dep
from app.performance.dependencies import TRANSCRIPT_DEP, DependencyGraph, asset_dep, scene_dep, settings_dep, stable_key, timeline_dep
from app.performance.synthetic import SyntheticSpec, build_project
from app.presentation.assembly import PresState
from app.project.phase2_commands import ReplaceScenesCommand, SetVisualPreferencesCommand
from app.project.phase3_commands import SceneDecisionCommand
from app.project.phase4_commands import MarkUserEditCommand
from app.project.phase5_commands import ApplyPresentationCommand, SetSettingCommand
from app.project.phase8_commands import SetQCSettingsCommand
from app.project.project_commands import AddAssetCommand, RemoveAssetCommand, SetScriptCommand, SetVoiceOverCommand
from app.rendering.commands import RenderRecordCommand, SetRenderSettingsCommand
from app.research.models import VisualAssignment
from app.timeline.timeline_commands import (AddClipCommand, DeleteClipCommand, MoveClipCommand, SetClipPropertiesCommand, SetTrackFlagCommand, SetTrackVolumeCommand,
                                            TrimClipCommand)
from app.timeline.clip import Clip
from app.timeline.timeline import new_clip_id

N = 10


def sid(i: int) -> str:
    return f"scene_{i:04d}"


class SpyCache:
    def __init__(self) -> None:
        self.deps: list[str] = []
        self.prefixes: list[str] = []

    def invalidate_by_dependency(self, dep: str) -> int:
        self.deps.append(dep)
        return 0

    def invalidate_by_dependency_prefix(self, prefix: str) -> int:
        self.prefixes.append(prefix)
        return 0


class Env:
    def __init__(self, tmp_path, scenes: int = N, cache=None) -> None:
        self.sp = build_project(tmp_path / "proj", SyntheticSpec(scenes=scenes, keyframes_per_visual=1))
        self.p = self.sp.project
        self.bus = EventBus()
        self.stack = CommandStack(self.bus)
        self.cache = cache if cache is not None else SpyCache()
        self.tr = ProjectChangeTracker(self.bus, lambda: self.p, lambda: self.cache)

    def clip_of(self, scene: str, kind: str = "media", track: str | None = None) -> Clip:
        for t in self.p.timeline.tracks:
            if track and t.id != track:
                continue
            for c in t.clips:
                if c.scene_id == scene and c.kind == kind:
                    return c
        raise AssertionError((scene, kind))

    def run(self, cmd: Command):
        self.tr.consume("t")
        self.stack.execute(cmd)
        return self.tr.peek(consumer="t")


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# ---------------------------------------------------------------------------------------------------------- table: command -> effect
def case_move_visual(e: Env):
    c = e.clip_of(sid(5))
    return MoveClipCommand(e.p.timeline, c.id, c.timeline_start, c.track_id), {"direct": {sid(5)}, "within": {sid(4), sid(5), sid(6)}, "domains": {"timeline"}, "not_domains": {"captions", "audio"}}


def case_trim_caption(e: Env):
    c = e.clip_of(sid(3), "caption")
    return TrimClipCommand(e.p.timeline, c.id, new_end=c.timeline_end - 0.2), {"direct": {sid(3)}, "within": {sid(2), sid(3), sid(4)}, "domains": {"timeline", "captions"}, "not_domains": {"audio", "render"}}


def case_delete_text(e: Env):
    c = e.clip_of(sid(5), "text")
    return DeleteClipCommand(e.p.timeline, c.id), {"direct": {sid(5)}, "within": {sid(4), sid(5), sid(6)}, "domains": {"timeline"}, "deps": {timeline_dep("track_v5"), scene_layout_dep(sid(5))}}


def case_clip_props(e: Env):
    c = e.clip_of(sid(7))
    return SetClipPropertiesCommand(e.p.timeline, c.id, opacity=0.5), {"direct": {sid(7)}, "within": {sid(6), sid(7), sid(8)}, "domains": {"timeline"}}


def case_track_volume(e: Env):
    return SetTrackVolumeCommand(e.p.timeline, "track_a2", 0.4), {"all_in_scene_ids": True, "domains": {"timeline", "audio"}, "deps": {timeline_dep("track_a2")}}


def case_track_flag(e: Env):
    return SetTrackFlagCommand(e.p.timeline, "track_v5", "hidden", True), {"direct_some": True, "domains": {"timeline"}, "deps": {timeline_dep("track_v5")}}


def case_replace_scene(e: Env):
    s = next(x for x in e.p.scenes if x.id == sid(6))
    new = replace(s, narration="A different narration.")
    return ReplaceScenesCommand(e.p, [sid(6)], [new], {}, "Edit scene"), {"direct": {sid(6)}, "content": {sid(6)}, "within": {sid(5), sid(6), sid(7)}, "domains": {"scenes", "visual"}, "deps": {scene_dep(sid(6))}}


def case_scene_decision(e: Env):
    a = VisualAssignment(sid(2), asset_id=next(iter(a.id for a in e.p.assets.all() if a.type is AssetType.IMAGE)), selected_by="USER")
    return SceneDecisionCommand(e.p, sid(2), "Choose visual", assignment=a), {"direct": {sid(2)}, "content": {sid(2)}, "within": {sid(1), sid(2), sid(3)}, "domains": {"visual"}, "not_domains": {"timeline"}}


def case_script(e: Env):
    return SetScriptCommand(e.p, "New script"), {"no_scenes": True, "domains": {"script"}, "deps": {TRANSCRIPT_DEP}}


def case_render_settings(e: Env):
    return SetRenderSettingsCommand(e.p, replace(e.p.render_settings, resolution="720p")), {"all": True, "domains": {"render"}, "deps": {settings_dep("render")}}


def case_qc_settings(e: Env):
    s = e.p.qc_settings.__class__.from_dict(e.p.qc_settings.to_dict())
    return SetQCSettingsCommand(e.p, s), {"no_scenes": True, "domains": {"qc_settings"}, "deps": {settings_dep("qc")}}


def case_audio_setting(e: Env):
    return SetSettingCommand(e.p, "audio_settings", e.p.audio_settings, "Change audio"), {"all": True, "domains": {"audio", "timeline"}, "deps": {settings_dep("audio_settings")}}


def case_caption_setting(e: Env):
    return SetSettingCommand(e.p, "caption_settings", e.p.caption_settings, "Change captions"), {"all": True, "domains": {"captions"}, "deps": {settings_dep("caption_settings")}}


def case_voice_over(e: Env):
    a = next(x for x in e.p.assets.all() if x.type is AssetType.AUDIO and x.id != e.p.voice_over.asset_id)
    return SetVoiceOverCommand(e.p, a), {"all": True, "domains": {"audio", "transcript"}, "deps": {TRANSCRIPT_DEP}}


def case_visual_prefs(e: Env):
    return SetVisualPreferencesCommand(e.p, e.p.visual_preferences), {"all": True, "domains": {"visual"}, "deps": {settings_dep("visual_preferences")}}


def case_remove_asset(e: Env):
    c = e.clip_of(sid(3))
    return RemoveAssetCommand(e.p, c.asset_id), {"direct_has": {sid(3)}, "domains": {"assets", "timeline"}, "deps": {"asset"}}


def case_add_asset(e: Env):
    a = Asset("asset_new", AssetType.IMAGE, SourceType.USER_MEDIA, "media/images/new.png", "new", None, 10, 10, None, None, False, None, None, None, 1, "h")
    return AddAssetCommand(e.p, a), {"no_scenes": True, "domains": {"assets"}, "deps": {asset_dep("asset_new")}}


def case_presentation_state(e: Env):
    after = PresState.capture(e.p)
    cap = next(c for t in after.timeline.tracks for c in t.clips if c.kind == "caption" and c.scene_id == sid(8))
    cap.text = {**cap.text, "text": "an edited caption"}
    return ApplyPresentationCommand(e.p, after, "Edit caption"), {"direct": {sid(8)}, "within": {sid(7), sid(8), sid(9)}, "domains": {"timeline", "captions"}, "not_domains": {"audio"}}


def case_composite(e: Env):
    c = e.clip_of(sid(2))
    d = e.clip_of(sid(9))
    return CompositeCommand("two edits", [MoveClipCommand(e.p.timeline, c.id, c.timeline_start), MarkUserEditCommand(e.p, d.id)], scope="timeline"), {"direct": {sid(2), sid(9)}, "domains": {"timeline"}}


CASES = [case_move_visual, case_trim_caption, case_delete_text, case_clip_props, case_track_volume, case_track_flag, case_replace_scene, case_scene_decision, case_script,
         case_render_settings, case_qc_settings, case_audio_setting, case_caption_setting, case_voice_over, case_visual_prefs, case_remove_asset, case_add_asset,
         case_presentation_state, case_composite]


@pytest.mark.parametrize("case", CASES, ids=[c.__name__[5:] for c in CASES])
def test_command_maps_to_the_scenes_domains_and_deps_it_can_affect(env, case):
    cmd, want = case(env)
    cs = env.run(cmd)
    assert not cs.unknown, "a known command must not fall back to the broad guess"
    all_ids = {sid(i) for i in range(1, N + 1)}
    if want.get("all"):
        assert cs.all_scenes
    else:
        assert not cs.all_scenes
    if "direct" in want:
        assert cs.direct_scene_ids == want["direct"], (sorted(cs.direct_scene_ids), sorted(want["direct"]))
    if "within" in want:
        assert cs.scene_ids <= want["within"] and cs.direct_scene_ids <= cs.scene_ids
    if want.get("no_scenes"):
        assert not cs.scene_ids and not cs.all_scenes
    if "content" in want:
        assert cs.content_scene_ids == want["content"]
    if "direct_has" in want:
        assert want["direct_has"] <= cs.direct_scene_ids
    if want.get("direct_some"):
        assert cs.direct_scene_ids and cs.direct_scene_ids <= all_ids
    if want.get("all_in_scene_ids"):
        assert cs.scene_ids == all_ids  # a music track under the whole video: every scene
    assert want.get("domains", set()) <= cs.domains
    assert not (want.get("not_domains", set()) & cs.domains)
    for d in want.get("deps", set()):
        if d == "asset":
            assert any(x.startswith("asset:") for x in cs.deps)
        else:
            assert d in cs.deps or d in {x for x in env.cache.deps}  # clip-level layout deps are invalidated, not listed in the change set
    assert cs.revision > cs.since


def test_a_command_without_effect_does_not_bump_the_revision(env):
    r0 = env.tr.revision
    env.stack.execute(RenderRecordCommand(env.p, {"render_id": "r1", "status": "COMPLETED"}))
    assert env.tr.revision == r0 and env.tr.peek().empty


class _Mystery(Command):
    scope = "timeline"
    description = "Mystery"

    def do(self) -> None: ...

    def undo(self) -> None: ...


class _MysteryElsewhere(_Mystery):
    scope = "gadgets"


class _QCMystery(_Mystery):
    scope = "qc"


def test_a_command_it_does_not_understand_is_never_guessed_narrow(env):
    cs = env.run(_Mystery())
    assert cs.unknown and cs.all_scenes and set(BROAD_TIMELINE) <= cs.domains
    cs = env.run(_MysteryElsewhere())
    assert cs.unknown and cs.all_scenes and set(QC_DOMAINS) <= cs.domains  # not even the scope is known: everything QC reads
    spy: SpyCache = env.cache
    assert "scene:" in spy.prefixes and "layout:" in spy.prefixes  # nothing is known about it: every scene-keyed cache entry goes
    r0 = env.tr.revision
    env.stack.execute(_QCMystery())
    assert env.tr.revision == r0  # the QC layer's own bookkeeping never moves the analysed content


def test_an_event_without_a_command_is_broad_too(env):
    env.bus.publish(Topics.PROJECT_CHANGED, scope="meta", command=None, action="do")
    cs = env.tr.peek()
    assert cs.unknown and cs.all_scenes
    env.tr.consume()
    env.bus.publish(Topics.PROJECT_CHANGED, scope="waveform", command=None, action="do")  # the presentation service announces a waveform refresh this way
    assert env.tr.peek().empty


# ---------------------------------------------------------------------------------------------------------- neighbours
def test_neighbours_are_dirty_but_not_direct(env):
    cs = env.run(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(5)).id, opacity=0.8))
    assert cs.direct_scene_ids == {sid(5)} and cs.scene_ids == {sid(4), sid(5), sid(6)}
    assert env.tr.neighbours(sid(1)) == [sid(2)] and env.tr.neighbours(sid(5)) == [sid(4), sid(6)] and env.tr.neighbours("nope") == []
    cs = env.run(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(1)).id, opacity=0.8))
    assert cs.scene_ids == {sid(1), sid(2)}
    cs = env.run(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(N)).id, opacity=0.8))
    assert cs.scene_ids == {sid(N - 1), sid(N)}


def test_a_clip_that_crosses_a_scene_boundary_dirties_both_scenes(env):
    c = env.clip_of(sid(5), "text")
    cs = env.run(MoveClipCommand(env.p.timeline, c.id, env.p.scenes[5].start - 1.0))  # the title now straddles scenes 5 and 6
    assert {sid(5), sid(6)} <= cs.direct_scene_ids


def test_graph_neighbours_follow_scene_order(env):
    g = env.tr.graph
    assert g.affected(scene_dep(sid(5)), include_neighbours=1, include_self=True) == {scene_dep(sid(4)), scene_dep(sid(5)), scene_dep(sid(6))}


# ---------------------------------------------------------------------------------------------------------- coalescing / revision / consumers
def test_a_burst_of_identical_edits_is_one_log_entry_and_the_index_is_built_once(env, monkeypatch):
    built = []
    orig = ct._SceneIndex.__init__

    def counting(self, scenes):
        built.append(1)
        orig(self, scenes)

    monkeypatch.setattr(ct._SceneIndex, "__init__", counting)
    c = env.clip_of(sid(5), "text")
    base = c.timeline_start
    r0 = env.tr.revision
    for i in range(300):  # a drag: the same clip moved a little at a time
        env.stack.execute(MoveClipCommand(env.p.timeline, c.id, base + (0.01 if i % 2 else 0.02)))
    assert env.tr.revision == r0 + 300
    assert len(env.tr._log) == 1
    assert len(built) <= 2


def test_log_is_bounded_and_compaction_only_widens(tmp_path):
    e = Env(tmp_path, scenes=40)
    e.tr._limit = 16
    e.tr.mark(scene_ids=[sid(1)])  # distinct changes: one entry each
    marks = [sid(i) for i in range(2, 31)]
    for s in marks:
        e.tr.mark(scene_ids=[s], domains=[f"d_{s}"])
    assert len(e.tr._log) <= 16
    cs = e.tr.peek(since=0)
    assert {sid(1), *marks} <= cs.direct_scene_ids  # nothing was lost by merging
    late = e.tr.peek(since=e.tr.revision - 1)
    assert sid(30) in late.direct_scene_ids  # a reader that is up to date still sees the newest change


def test_revision_is_monotonic_and_undo_redo_are_changes(env):
    c = env.clip_of(sid(5))
    seen = [env.tr.revision]
    env.stack.execute(MoveClipCommand(env.p.timeline, c.id, c.timeline_start))
    seen.append(env.tr.revision)
    cs = env.tr.consume("q")
    assert sid(5) in cs.direct_scene_ids
    assert env.tr.consume("q").empty
    env.stack.undo()
    seen.append(env.tr.revision)
    cs = env.tr.consume("q")
    assert sid(5) in cs.direct_scene_ids and "timeline" in cs.domains
    env.stack.redo()
    seen.append(env.tr.revision)
    assert sid(5) in env.tr.consume("q").direct_scene_ids
    assert seen == sorted(seen) and len(set(seen)) == len(seen)


def test_consumers_are_independent_and_mark_clean_keeps_later_changes(env):
    env.stack.execute(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(2)).id, opacity=0.5))
    r1 = env.tr.revision
    env.stack.execute(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(8)).id, opacity=0.5))
    assert sid(2) in env.tr.consume("qc").direct_scene_ids
    assert env.tr.peek(consumer="preview").direct_scene_ids == {sid(2), sid(8)}  # the preview consumer has not looked yet
    env.tr.mark_clean("preview", r1)
    assert env.tr.peek(consumer="preview").direct_scene_ids == {sid(8)}
    assert env.tr.peek(since=r1).direct_scene_ids == {sid(8)}
    assert env.tr.dirty_scenes >= {sid(8)} and "timeline" in env.tr.dirty_domains


def test_project_switch_resets_everything(env):
    env.stack.execute(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(2)).id, opacity=0.5))
    e0, r0 = env.tr.epoch, env.tr.revision
    env.bus.publish(Topics.PROJECT_OPENED, project_id="other")
    assert env.tr.epoch == e0 + 1 and env.tr.revision > r0 and env.tr.peek().empty and env.tr.peek(since=0).empty
    env.bus.publish(Topics.PROJECT_CLOSED, project_id="other")
    assert env.tr.epoch == e0 + 2


def test_a_mapping_failure_widens_instead_of_losing_the_change(env, monkeypatch):
    def boom(self, cmd, scope):
        raise RuntimeError("bug")

    monkeypatch.setattr(ProjectChangeTracker, "_effect_for_command", boom)
    cs = env.run(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(2)).id, opacity=0.5))
    assert cs.all_scenes and cs.unknown


# ---------------------------------------------------------------------------------------------------------- cache invalidation
def test_invalidation_targets_only_the_dependants(env):
    spy: SpyCache = env.cache
    c = env.clip_of(sid(5))
    env.stack.execute(MoveClipCommand(env.p.timeline, c.id, c.timeline_start))
    assert timeline_dep(c.track_id) in spy.deps and scene_layout_dep(sid(5)) in spy.deps
    assert not [d for d in spy.deps if d.startswith("asset:") or d.startswith("settings:") or d == TRANSCRIPT_DEP]
    assert scene_dep(sid(5)) not in spy.deps  # a clip move is not a change of the scene's own content (text, timing, assignment)
    spy.deps.clear()
    asset_id = env.clip_of(sid(3)).asset_id
    env.stack.execute(RemoveAssetCommand(env.p, asset_id))
    assert asset_dep(asset_id) in spy.deps


def test_graph_dependants_are_invalidated_transitively(env):
    spy: SpyCache = env.cache
    env.tr.graph.add_edge(asset_dep("a1"), "thumb:a1")
    env.tr.graph.add_edge("thumb:a1", "sprite:a1")
    env.tr.mark(deps=[asset_dep("a1")])
    assert {"thumb:a1", "sprite:a1", asset_dep("a1")} <= set(spy.deps)


def test_real_cache_unrelated_entries_keep_hitting(tmp_path):
    cache = MediaCacheManager(tmp_path / "cache")
    e = Env(tmp_path, cache=cache)
    for i in (3, 7):
        cache.put(f"analysis:{i}", {"scene": i}, category="analysis", deps={scene_dep(sid(i)): "v1"})
        cache.put(f"layout:{i}", {"scene": i}, category="analysis", deps={scene_layout_dep(sid(i)): "v1"})
    cache.put("thumb:x", {"x": 1}, category="analysis", deps={asset_dep("asset_x"): "v1"})
    cache.put("render:settings", {"x": 1}, category="analysis", deps={settings_dep("render"): "v1"})
    a = VisualAssignment(sid(3), asset_id=e.clip_of(sid(3)).asset_id, selected_by="USER")
    e.stack.execute(SceneDecisionCommand(e.p, sid(3), "Choose visual", assignment=a))  # scene 3's content
    assert cache.get("analysis:3") is None and cache.get("layout:3") is None  # scene 3's own entries are gone ...
    assert cache.get("analysis:7") is not None and cache.get("layout:7") is not None  # ... scene 7 and everything unrelated still hit
    assert cache.get("thumb:x") is not None and cache.get("render:settings") is not None
    e.stack.execute(SetClipPropertiesCommand(e.p.timeline, e.clip_of(sid(7)).id, opacity=0.5))
    assert cache.get("layout:7") is None and cache.get("analysis:7") is not None  # a clip edit moves the layout of scene 7 but not its content
    e.stack.execute(SetRenderSettingsCommand(e.p, replace(e.p.render_settings, fps=24)))
    assert cache.get("render:settings") is None and cache.get("thumb:x") is not None
    assert e.tr.invalidated >= 4


def test_a_dead_or_broken_cache_never_breaks_tracking(tmp_path):
    e = Env(tmp_path, cache=None)
    e.cache = None
    e.tr._cache = lambda: None
    e.stack.execute(SetClipPropertiesCommand(e.p.timeline, e.clip_of(sid(2)).id, opacity=0.5))
    assert sid(2) in e.tr.peek().direct_scene_ids

    class Broken:
        def invalidate_by_dependency(self, d):
            raise OSError("disk")

        invalidate_by_dependency_prefix = invalidate_by_dependency

    e.tr._cache = lambda: Broken()
    e.stack.execute(SetClipPropertiesCommand(e.p.timeline, e.clip_of(sid(3)).id, opacity=0.5))
    assert sid(3) in e.tr.peek().direct_scene_ids


def test_large_project_event_cost_is_structural(tmp_path, monkeypatch):
    """No hashing of the project and no scan of the timeline per clip edit: the scene index is built once for 500 edits of a 300-scene project."""
    import app.qc.context as qc_context

    e = Env(tmp_path, scenes=300)
    hashed = []
    monkeypatch.setattr(qc_context, "sha", lambda *a, **k: hashed.append(1) or "")
    built = []
    orig = ct._SceneIndex.__init__
    monkeypatch.setattr(ct._SceneIndex, "__init__", lambda self, s: (built.append(1), orig(self, s))[1])
    clips = [e.clip_of(sid(i)) for i in range(1, 101)]
    for k in range(5):
        for c in clips:
            e.stack.execute(SetClipPropertiesCommand(e.p.timeline, c.id, opacity=0.5 + 0.1 * (k % 2)))
    assert not hashed and len(built) <= 2
    assert len(e.tr.peek(since=0).direct_scene_ids) == 100


def test_tracker_close_unsubscribes(env):
    env.tr.close()
    r = env.tr.revision
    env.stack.execute(SetClipPropertiesCommand(env.p.timeline, env.clip_of(sid(2)).id, opacity=0.5))
    assert env.tr.revision == r
    assert stable_key("x", 1)  # keep the import honest
    assert DependencyGraph is not None and AddClipCommand is not None and Clip is not None and new_clip_id is not None


def test_a_proxy_record_never_invalidates_the_originals_dependants():
    """Found by the Phase 9 acceptance workflow: making a proxy deleted the asset's thumbnail because the tracker treated the proxy record as a change to the original."""
    from types import SimpleNamespace

    from app.performance.change_tracker import _h_proxy
    from app.performance.dependencies import asset_dep, settings_dep

    eff = _h_proxy(SimpleNamespace(), SimpleNamespace(asset_id="media_00006"))
    assert asset_dep("media_00006") not in eff.deps and settings_dep("proxies") in eff.deps and "proxy:media_00006" in eff.deps
    assert not eff.all_scenes and not eff.direct
