"""Phase 4 service tests: assembly, undo, overrides, locks, regeneration, failure recovery, persistence, validation."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

import pytest

from app.core.exceptions import EditingError
from app.editing.assembly import EditState
from app.editing.compose import PreviewComposer
from app.editing.models import Creator, DecisionType, SceneEditStatus
from app.editing.strategy import RuleBasedProvider
from app.project.project import Project
from app.tests.conftest import needs_ffmpeg
from app.timeline.timeline import Timeline

pytestmark = needs_ffmpeg


def gen(ws, *a, **k):
    job = ws.editing.generate(*a, **k)
    assert ws.jobs.wait_idle(60)
    return job


def last(ws):
    return ws.project.editing_sessions[-1]


def clips(ws, scene_id, kind=None):
    return [c for c in ws.project.timeline.all_clips() if c.scene_id == scene_id and (kind is None or c.kind == kind)]


def snapshot(p):
    return json.dumps(p.timeline.to_dict(), sort_keys=True), json.dumps({k: v.to_dict() for k, v in p.editing_decisions.items()}, sort_keys=True, default=str)


def scene_with(ws, predicate):
    p = ws.project
    for sc in p.scenes:
        if predicate(sc):
            return sc
    raise AssertionError("no such scene")


class FailingProvider(RuleBasedProvider):
    name, label = "failing", "Failing"

    def __init__(self, fail_ids):
        self.fail_ids = set(fail_ids)
        self.calls = []

    def plan_scene(self, sc, ctx, profile):
        self.calls.append(sc.scene.id)
        if sc.scene.id in self.fail_ids:
            raise RuntimeError("model exploded")
        return super().plan_scene(sc, ctx, profile)


# ============================================================ the assembled timeline
def test_generate_builds_a_valid_multitrack_timeline_with_structured_decisions(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    s = last(ws)
    assert s.status == "COMPLETED" and len(s.completed) == len(p.scenes) and not s.validation_errors
    tracks = Counter((c.track_id, c.kind) for c in p.timeline.all_clips())
    assert tracks[("track_a1", "media")] == 1  # voice-over
    assert any(t == "track_v1" for t, _ in tracks) and any(t == "track_v3" for t, _ in tracks) and tracks[("track_v5", "text")] > 0 and tracks[("track_v4", "graphic")] > 0
    assert {t.id for t in p.timeline.tracks} >= {"track_v1", "track_v2", "track_v3", "track_v4", "track_v5", "track_v6", "track_a1", "track_a2", "track_a3"}
    kinds = {d.type for d in p.editing_decisions.values()}
    assert kinds >= {DecisionType.VISUAL_TIMING, DecisionType.TRANSITION, DecisionType.NUMBER_EMPHASIS, DecisionType.EVIDENCE_FOCUS, DecisionType.AUDIO_DUCK,
                     DecisionType.CAPTION_EMPHASIS, DecisionType.TRIM, DecisionType.CUT}
    assert kinds & {DecisionType.ZOOM, DecisionType.PAN}
    for d in p.editing_decisions.values():
        assert d.reason and len(d.reason) < 260 and 0 <= d.confidence <= 100 and d.created_by in (Creator.AI, Creator.SYSTEM) and d.scene_id
        if d.target_id:
            assert p.timeline.get_clip(d.target_id) is not None
    assert p.timeline_version == 1 and p.timeline_generation.version == 1 and p.timeline_generation.status == "COMPLETE"
    assert all(g.status is SceneEditStatus.COMPLETE for g in p.timeline_generation.scenes.values())
    assert ws.editing.validate_timeline() == [] or all(i.severity == "warning" for i in ws.editing.validate_timeline())


def test_visuals_tile_every_scene_and_voice_over_stays_the_master_clock(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    for sc in p.scenes:
        vis = sorted([c for c in clips(ws, sc.id, "media")], key=lambda c: c.timeline_start)
        assert vis[0].timeline_start == pytest.approx(sc.start, abs=1e-3) and vis[-1].timeline_end == pytest.approx(sc.end, abs=1e-3)
        for a, b in zip(vis, vis[1:]):
            assert a.timeline_end == pytest.approx(b.timeline_start, abs=1e-3)
    voice = next(c for c in p.timeline.get_track("track_a1").clips)
    asset = p.assets.get(p.voice_over.asset_id)
    assert voice.asset_id == asset.id and voice.timeline_start == 0 and voice.duration == pytest.approx(asset.duration)
    assert voice.created_by == "SYSTEM"


def test_source_media_is_never_modified_and_ranges_are_valid(edit_ws):
    ws = edit_ws
    p = ws.project
    before = {a.id: p.asset_path(a).read_bytes() for a in p.assets.all()}
    gen(ws)
    assert {a.id: p.asset_path(a).read_bytes() for a in p.assets.all()} == before
    for c in p.timeline.all_clips():
        if c.kind == "media" and c.track_id != "track_a1":
            a = p.assets.get(c.asset_id)
            if a.duration:
                assert 0 <= c.source_in < c.source_out <= a.duration + 0.05
    trims = [d for d in p.editing_decisions.values() if d.type is DecisionType.TRIM]
    assert trims and all(d.parameters["source_out"] > d.parameters["source_in"] for d in trims)


def test_one_undo_reverts_the_entire_ai_edit_and_redo_restores_it(edit_ws):
    ws = edit_ws
    p = ws.project
    empty = snapshot(p)
    gen(ws)
    done = snapshot(p)
    assert done != empty
    ws.undo()
    assert snapshot(p) == empty and p.timeline_version == 0 and not p.editing_decisions and not p.timeline.all_clips()
    assert not ws.commands.can_undo or ws.commands.undo_text != "AI edit (16 scenes)"
    ws.redo()
    assert snapshot(p) == done and p.timeline_version == 1


def test_project_validation_accepts_the_generated_project(edit_ws):
    ws = edit_ws
    gen(ws)
    ws.project.validate()  # text/graphic clips without assets are legal


def test_settings_change_the_edit(edit_ws):
    ws = edit_ws
    ws.editing.update_settings(style="documentary", pacing=0.2, motion_intensity=0.1)
    gen(ws)
    doc = Counter(c.kind for c in ws.project.timeline.all_clips() if c.track_id in ("track_v1", "track_v2", "track_v3"))
    n_motion_doc = sum(1 for d in ws.project.editing_decisions.values() if d.type in (DecisionType.ZOOM, DecisionType.PAN))
    ws.editing.update_settings(style="dynamic", pacing=0.9, motion_intensity=0.9)
    gen(ws, force=True)
    dyn = Counter(c.kind for c in ws.project.timeline.all_clips() if c.track_id in ("track_v1", "track_v2", "track_v3"))
    n_motion_dyn = sum(1 for d in ws.project.editing_decisions.values() if d.type in (DecisionType.ZOOM, DecisionType.PAN))
    assert dyn["media"] > doc["media"] and n_motion_dyn > n_motion_doc  # faster cuts and more movement


# ============================================================ multi-visual scenes
def test_a_scene_can_have_several_visuals_and_they_are_tracked(edit_ws):
    ws = edit_ws
    p = ws.project
    sc = max(p.scenes, key=lambda s: s.duration)
    ws.editing.add_scene_visual(sc.id, ws.assets_by["wide.png"].id)
    ws.editing.add_scene_visual(sc.id, ws.assets_by["v_long.mp4"].id)
    gen(ws)
    vis = sorted(clips(ws, sc.id, "media"), key=lambda c: c.timeline_start)
    assert len({c.asset_id for c in vis}) >= 2 and len(vis) >= 3
    segs = p.editing_strategy.segments[sc.id]
    assert len(segs) == len(vis) and all(s.reason and s.visual_segment_id and s.scene_id == sc.id for s in segs)
    assert any(d.type is DecisionType.CUT and d.scene_id == sc.id for d in p.editing_decisions.values())
    assert vis[-1].timeline_end == pytest.approx(sc.end, abs=1e-3)


def test_a_visual_continues_across_scenes_when_the_subject_continues(edit_ws):
    from app.project.phase3_commands import SceneDecisionCommand
    from app.research.models import VisualAssignment

    ws = edit_ws
    p = ws.project
    a0 = p.visual_assignments[p.scenes[5].id]
    nxt = p.scenes[6]
    ws.apply_command(SceneDecisionCommand(p, nxt.id, "same", assignment=VisualAssignment(**{**a0.__dict__, "scene_id": nxt.id})))
    gen(ws)
    c = clips(ws, nxt.id, "media")[0]
    assert c.metadata["reuse_count"] >= 1 and c.metadata["previous_scene_id"] == p.scenes[5].id and c.metadata["reuse_reason"]


# ============================================================ ownership: user edits & regeneration
def pick_motion_scene(ws):
    p = ws.project
    for d in p.editing_decisions.values():
        if d.type in (DecisionType.ZOOM, DecisionType.PAN) and len(clips(ws, d.scene_id, "media")) == 1:
            return d
    raise AssertionError("no scene with motion")


def test_user_changes_to_decisions_survive_regeneration(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    d = pick_motion_scene(ws)
    sid = d.scene_id
    clip = ws.project.timeline.get_clip(d.target_id)
    ai_id = d.decision_id
    new = ws.editing.update_decision(d.decision_id, {"kind": "PUNCH_IN", "start_scale": 1.0, "end_scale": 1.31, "start_pos": [0, 0], "end_pos": [0, 0]})
    assert new.created_by is Creator.USER and new.overrides_decision_id == ai_id and ai_id not in p.editing_decisions
    assert any(o.override_id == new.decision_id and o.original.decision_id == ai_id and o.original.created_by is Creator.AI for o in p.ai_overrides)
    clip = p.timeline.get_clip(clip.id)
    assert [k.value for k in clip.keyframes if k.property == "scale"] == [1.0, 1.31]
    vt = next(x for x in ws.editing.decisions_for_scene(sid) if x.type is DecisionType.VISUAL_TIMING)
    ws.editing.update_decision(vt.decision_id, {"duration": clip.duration - 0.5})
    ws.editing.set_lock(sid, "VISUAL")
    clip = ws.project.timeline.get_clip(clip.id)
    assert clip.locked and clip.created_by == "USER" and clip.duration == pytest.approx(vt.duration - 0.5)
    user_clip_id, user_dur = clip.id, clip.duration
    other = snapshot_scene(ws, [s.id for s in p.scenes if s.id != sid][0])
    gen_job = ws.editing.regenerate_scene(sid)
    assert ws.jobs.wait_idle(60)
    after = ws.project.timeline.get_clip(user_clip_id)
    assert after is not None and after.locked and after.duration == pytest.approx(user_dur) and after.created_by == "USER"  # locked USER visual unchanged
    scale = [(k.time, k.value) for k in after.keyframes if k.property == "scale"]
    assert scale[-1][1] == 1.31  # USER zoom unchanged
    ud = [x for x in ws.editing.decisions_for_scene(sid) if x.type in (DecisionType.ZOOM, DecisionType.PAN)]
    assert ud and ud[0].created_by is Creator.USER and ud[0].parameters["end_scale"] == 1.31
    assert snapshot_scene(ws, [s.id for s in p.scenes if s.id != sid][0]) == other  # unrelated scenes untouched
    assert gen_job is not None


def snapshot_scene(ws, sid):
    return json.dumps([c.to_dict() for c in sorted(clips(ws, sid), key=lambda c: (c.timeline_start, c.track_id, c.slot))], sort_keys=True, default=str)[:20000].replace(
        "clip_", "")[:0] or tuple(sorted((c.track_id, round(c.timeline_start, 3), round(c.duration, 3), c.asset_id, c.slot) for c in clips(ws, sid)))


def test_user_motion_override_is_reapplied_to_a_regenerated_ai_clip(edit_ws):
    ws = edit_ws
    gen(ws)
    d = pick_motion_scene(ws)
    sid = d.scene_id
    ws.editing.update_decision(d.decision_id, {"kind": "PUNCH_IN", "start_scale": 1.0, "end_scale": 1.27})
    old_clip_id = d.target_id
    ws.editing.regenerate_scene(sid)
    assert ws.jobs.wait_idle(60)
    clip = next(c for c in clips(ws, sid, "media") if c.slot == "visual:0")
    assert clip.id != old_clip_id and clip.created_by == "AI"  # the AI-owned visual was regenerated...
    assert max(k.value for k in clip.keyframes if k.property == "scale") == 1.27  # ...and the user's zoom applied to it
    assert any(x.created_by is Creator.USER and x.target_id == clip.id for x in ws.editing.decisions_for_scene(sid))


def test_manual_timeline_edits_take_ownership_and_are_undoable(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    sc = scene_with(ws, lambda s: len([c for c in clips(ws, s.id, "media")]) == 1)
    clip = clips(ws, sc.id, "media")[0]
    dec = p.editing_decisions[clip.ai_decision_id]
    assert clip.created_by == "AI" and dec.created_by is Creator.AI
    ws.timeline.trim_clip(clip.id, new_end=clip.timeline_end - 0.4)
    clip = p.timeline.get_clip(clip.id)
    now = p.editing_decisions[clip.ai_decision_id]
    assert clip.created_by == "USER" and now.created_by is Creator.USER and now.overrides_decision_id == dec.decision_id
    assert now.duration == pytest.approx(clip.duration) and ws.editing.override_of(now.decision_id).decision_id == dec.decision_id
    ws.undo()  # one undo undoes both the trim and the ownership change
    clip = p.timeline.get_clip(clip.id)
    assert clip.created_by == "AI" and p.editing_decisions[clip.ai_decision_id].decision_id == dec.decision_id and not p.ai_overrides
    ws.redo()
    assert p.timeline.get_clip(clip.id).created_by == "USER"


def test_regeneration_never_overwrites_a_manually_edited_clip(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    sc = scene_with(ws, lambda s: len(clips(ws, s.id, "media")) == 1)
    clip = clips(ws, sc.id, "media")[0]
    ws.timeline.trim_clip(clip.id, new_end=clip.timeline_end - 0.5)
    edited = p.timeline.get_clip(clip.id)
    key = (edited.id, edited.timeline_start, edited.duration, edited.asset_id)
    ws.editing.regenerate_all()
    assert ws.jobs.wait_idle(60)
    again = p.timeline.get_clip(edited.id)
    assert again is not None and (again.id, again.timeline_start, again.duration, again.asset_id) == key
    assert ws.editing.validate_timeline() is not None
    assert not [i for i in ws.editing.validate_timeline() if i.severity == "error"]  # the user's gap is a warning, never an error


def test_deleted_ai_elements_are_not_recreated(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    text = next(c for c in p.timeline.all_clips() if c.kind == "text")
    sid, slot = text.scene_id, text.slot
    ws.timeline.delete_clip(text.id)
    assert f"{sid}|{slot}" in p.timeline_generation.suppressed_slots
    ws.editing.regenerate_scene(sid)
    assert ws.jobs.wait_idle(60)
    assert not [c for c in clips(ws, sid, "text") if c.slot == slot]
    ws.undo()  # undo of the regeneration
    ws.undo()  # undo of the delete
    assert p.timeline.get_clip(text.id) is not None and f"{sid}|{slot}" not in p.timeline_generation.suppressed_slots


def test_locks_protect_text_motion_and_whole_scenes(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    sc = scene_with(ws, lambda s: bool(clips(ws, s.id, "text")))
    t = clips(ws, sc.id, "text")[0]
    ws.editing.set_lock(sc.id, "TEXT")
    assert p.timeline.get_clip(t.id).locked
    ws.editing.set_lock(sc.id, "SCENE")
    before = snapshot_scene(ws, sc.id)
    ws.editing.update_settings(style="dynamic")
    ws.editing.regenerate_scene(sc.id)
    assert ws.jobs.wait_idle(60)
    assert snapshot_scene(ws, sc.id) == before  # a locked scene is never touched
    assert {r["scene_id"]: r["status"] for r in ws.editing.scene_rows()}[sc.id] is SceneEditStatus.LOCKED
    ws.editing.set_lock(sc.id, "SCENE", False)
    ws.editing.regenerate_scene(sc.id)
    assert ws.jobs.wait_idle(60)
    assert p.timeline.get_clip(t.id) is not None and p.timeline.get_clip(t.id).locked  # the locked text still survives
    with pytest.raises(EditingError):
        ws.editing.set_lock(sc.id, "BOGUS")
    ws.undo()


def test_inspector_edits_validate_input(edit_ws):
    ws = edit_ws
    gen(ws)
    vt = next(d for d in ws.project.editing_decisions.values() if d.type is DecisionType.VISUAL_TIMING)
    with pytest.raises(EditingError):
        ws.editing.update_decision(vt.decision_id, {"duration": -1})
    long_video = next(d for d in ws.project.editing_decisions.values() if d.type is DecisionType.TRIM and d.parameters.get("asset_duration"))
    with pytest.raises(EditingError):
        ws.editing.update_decision(long_video.decision_id, {"source_in": 999})
    with pytest.raises(EditingError):
        ws.editing.update_decision("dec_nope", {})
    text = next(d for d in ws.project.editing_decisions.values() if d.type in (DecisionType.TEXT, DecisionType.NUMBER_EMPHASIS))
    with pytest.raises(EditingError):
        ws.editing.update_decision(text.decision_id, {"content": "  "})
    ws.project.validate()


def test_text_graphic_is_editable_and_becomes_user_owned(edit_ws):
    ws = edit_ws
    gen(ws)
    text = next(d for d in ws.project.editing_decisions.values() if d.type in (DecisionType.TEXT, DecisionType.NUMBER_EMPHASIS))
    new = ws.editing.update_decision(text.decision_id, {"content": "Edited", "style": "HEADLINE", "size": 70})
    clip = ws.project.timeline.get_clip(new.target_id)
    assert clip.text["content"] == "Edited" and clip.text["size"] == 70 and clip.created_by == "USER"


# ============================================================ failure recovery & incremental processing
def test_failed_scene_keeps_earlier_scenes_and_retry_resumes_there(edit_ws):
    ws = edit_ws
    p = ws.project
    fail_id = p.scenes[9].id
    prov = FailingProvider({fail_id})
    ws.editing.strategy.register(prov)
    ws.editing.update_settings(provider="failing")
    gen(ws)
    order = [s.id for s in p.scenes]
    gens = p.timeline_generation.scenes
    assert last(ws).status == "FAILED" and last(ws).failed_scene == fail_id
    assert all(gens[i].status is SceneEditStatus.COMPLETE for i in order[:9])
    assert gens[fail_id].status is SceneEditStatus.FAILED and "model exploded" in gens[fail_id].error
    assert all(gens[i].status is SceneEditStatus.PENDING for i in order[10:])
    assert p.timeline_generation.status == "PARTIAL"
    assert not clips(ws, fail_id) and all(clips(ws, i, "media") for i in order[:9])
    kept = {c.id for i in order[:9] for c in clips(ws, i)}
    ws.project.validate()
    prov.fail_ids.clear()
    prov.calls.clear()
    ws.editing.retry_failed()
    assert ws.jobs.wait_idle(60)
    assert prov.calls == order[9:]  # resumed at the failed scene; scenes 1-9 were not reprocessed
    gens = p.timeline_generation.scenes
    assert all(gens[i].status is SceneEditStatus.COMPLETE for i in order)
    assert kept <= {c.id for c in p.timeline.all_clips()}  # earlier scenes' clips are the very same objects
    assert p.timeline_generation.status == "COMPLETE" and last(ws).status == "COMPLETED"
    with pytest.raises(EditingError):
        ws.editing.retry_failed()


def test_only_changed_scenes_are_reprocessed(edit_ws):
    ws = edit_ws
    prov = FailingProvider(set())
    ws.editing.strategy.register(prov)
    ws.editing.update_settings(provider="failing")
    gen(ws)
    assert len(prov.calls) == len(ws.project.scenes)
    prov.calls.clear()
    assert gen(ws) is None and prov.calls == []  # nothing changed: nothing is re-analysed
    assert last(ws).log and "up to date" in last(ws).log[0]
    p = ws.project
    sid = p.scenes[4].id
    ws.editing.add_scene_visual(sid, ws.assets_by["wide.png"].id)
    assert ws.editing.outdated_scenes() == [sid]
    gen(ws)
    assert prov.calls == [sid]
    assert ws.editing.outdated_scenes() == []


def test_plans_are_cached_on_disk(edit_ws):
    ws = edit_ws
    gen(ws)
    cache = ws.project.root / "cache" / "editing" / "plans"
    assert len(list(cache.glob("rule_based_*.json"))) == len(ws.project.scenes)
    prov = FailingProvider(set())
    prov.name = "rule_based"  # same cache namespace: a forced regeneration reads the plans instead of re-analysing
    ws.editing.strategy.register(prov)
    ws.editing.regenerate_all()
    assert ws.jobs.wait_idle(60)
    assert prov.calls == []  # served from the cache


def test_cancel_changes_nothing(edit_ws):
    ws = edit_ws
    p = ws.project

    class Cancelling(RuleBasedProvider):
        name = "cancelling"

        def plan_scene(self, sc, ctx, profile):
            if sc.index == 2:
                ws.editing.cancel()
            return super().plan_scene(sc, ctx, profile)

    ws.editing.strategy.register(Cancelling())
    ws.editing.update_settings(provider="cancelling")
    before = snapshot(p)
    gen(ws)
    assert last(ws).status == "CANCELED" and snapshot(p) == before and p.timeline_version == 0
    assert not ws.editing.running


# ============================================================ validation, checkpoints, autosave, persistence
def test_invalid_generated_edit_is_not_committed(edit_ws):
    ws = edit_ws

    class Broken(RuleBasedProvider):
        name = "broken"

        def plan_scene(self, sc, ctx, profile):
            plan = super().plan_scene(sc, ctx, profile)
            if plan.segments and sc.index == 3:
                plan.segments[0].segment.source_out = 9999.0  # impossible source range
                plan.segments[0].segment.source_in = 0.0
                plan.segments[0].segment.duration = 3.0
            return plan

    class Corrupt(Broken):
        name = "corrupt"

    # force an impossible range through the assembler: asset is shorter than the requested range
    ws.editing.strategy.register(Broken())
    ws.editing.update_settings(provider="broken")
    p = ws.project
    sid = p.scenes[3].id
    vid = ws.assets_by["v_short.mp4"]
    from app.project.phase3_commands import SceneDecisionCommand
    from app.research.models import Acquisition, VisualAssignment

    ws.apply_command(SceneDecisionCommand(p, sid, "x", assignment=VisualAssignment(sid, None, vid.id, "USER", 90, True, acquisition=Acquisition.LOCAL)))
    before = snapshot(p)
    gen(ws)
    s = last(ws)
    assert s.status == "FAILED" and s.validation_errors and "gap" in s.validation_errors[0]
    assert snapshot(p) == before and not p.timeline.all_clips()  # the current timeline was not replaced
    assert not ws.editing.running


def test_a_checkpoint_is_written_before_the_ai_edit_and_autosave_runs(edit_ws, app_paths):
    ws = edit_ws
    gen(ws)
    cps = ws.editing.checkpoints()
    assert len(cps) == 1 and cps[0].name.startswith("before_ai_edit_")
    doc = json.loads(cps[0].read_text())
    assert doc["timeline_version"] == 0 and not any(t["clips"] for t in doc["timeline"]["tracks"])  # the state *before* the AI edit
    assert last(ws).checkpoint == cps[0].name
    assert ws.autosave.wait_idle(10)
    assert any(e.project_id == ws.project.project_id for e in ws.recovery.list_entries())  # autosave captured the new edit
    ws.editing.regenerate_all()
    assert ws.jobs.wait_idle(60)
    assert len(ws.editing.checkpoints()) == 2


def test_edit_survives_save_close_and_reopen(edit_ws, app_paths):
    ws = edit_ws
    gen(ws)
    p = ws.project
    d = pick_motion_scene(ws)
    ws.editing.update_decision(d.decision_id, {"kind": "PUNCH_IN", "start_scale": 1.0, "end_scale": 1.2})
    sid = d.scene_id
    ws.editing.set_lock(sid, "VISUAL")
    before = p.to_document()
    comp = PreviewComposer(p)
    samples = [0.5, 8.0, 20.0, 40.0, 70.0, 100.0]
    frames_before = [json.dumps(_frame(comp.frame_at(t)), sort_keys=True) for t in samples]
    root = p.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    p2 = ws.project
    after = p2.to_document()
    for key in ("timeline", "editing_decisions", "editing_strategy", "timeline_generation", "ai_overrides", "timeline_version", "editing_settings", "editing_sessions"):
        assert after[key] == before[key], key
    comp2 = PreviewComposer(p2)
    assert [json.dumps(_frame(comp2.frame_at(t)), sort_keys=True) for t in samples] == frames_before
    assert any(c.locked for c in p2.timeline.all_clips())
    # keeps working after reopen: regenerate honours what was restored
    ws.editing.regenerate_scene(sid)
    assert ws.jobs.wait_idle(60)
    assert any(c.locked and c.scene_id == sid for c in p2.timeline.all_clips())
    p2.validate()


def _frame(fs):
    from dataclasses import asdict

    return asdict(fs)


def test_older_projects_migrate_and_open(edit_ws):
    ws = edit_ws
    doc = ws.project.to_document()
    for k in ("editing_settings", "editing_strategy", "editing_sessions", "editing_decisions", "timeline_generation", "ai_overrides", "timeline_version"):
        del doc[k]
    doc["schema_version"] = 3
    doc["timeline"]["tracks"] = [t for t in doc["timeline"]["tracks"] if t["id"] != "track_v6"]
    p = Project.from_document(doc)
    assert p.schema_version == 6 and any(t.id == "track_v6" for t in p.timeline.tracks) and p.timeline_version == 0 and p.editing_decisions == {}


# ============================================================ missing visuals / media
def test_scene_without_an_approved_visual_is_marked_missing_and_not_invented(edit_ws):
    ws = edit_ws
    p = ws.project
    sid = p.scenes[2].id
    del p.visual_assignments[sid]
    assert ws.editing.visual_status(sid)["status"] == "MISSING"
    gen(ws)
    assert not clips(ws, sid, "media")  # no random substitute
    assert p.timeline_generation.scenes[sid].status is SceneEditStatus.NEEDS_VISUAL and p.timeline_generation.scenes[sid].visual_status == "MISSING"
    rows = {r["scene_id"]: r for r in ws.editing.scene_rows()}
    assert rows[sid]["visual_status"] == "MISSING"
    ws.editing.assign_scene_visual(sid, ws.assets_by["plain.png"].id)  # Manual Add
    assert ws.editing.visual_status(sid)["status"] == "APPROVED"
    assert sid in ws.editing.outdated_scenes() and set(ws.editing.outdated_scenes()) <= {sid, p.scenes[3].id}  # the scene and the one after it (transition context)
    gen(ws)
    assert clips(ws, sid, "media") and p.timeline_generation.scenes[sid].status is SceneEditStatus.COMPLETE
    with pytest.raises(EditingError):
        ws.editing.assign_scene_visual(sid, p.voice_over.asset_id)


def test_skipped_and_unapproved_visuals_are_not_used(edit_ws):
    ws = edit_ws
    p = ws.project
    s1, s2 = p.scenes[1].id, p.scenes[2].id
    ws.research.skip_visual(s1)
    p.visual_assignments[s2].approved = False
    assert ws.editing.visual_status(s1)["status"] == "SKIPPED" and ws.editing.visual_status(s2)["status"] == "UNAPPROVED"
    gen(ws)
    assert not clips(ws, s1, "media") and not clips(ws, s2, "media")


def test_missing_media_is_reported_and_never_substituted(edit_ws):
    ws = edit_ws
    p = ws.project
    sid = p.scenes[5].id
    asset = p.assets.get(p.visual_assignments[sid].asset_id)
    p.asset_path(asset).unlink()
    assert ws.editing.visual_status(sid)["status"] == "MISSING_MEDIA"
    gen(ws)
    assert not clips(ws, sid, "media")
    assert p.timeline_generation.scenes[sid].visual_status == "MISSING_MEDIA"


def test_regenerate_requires_scenes_and_a_project(ws):
    with pytest.raises(EditingError):
        ws.editing.generate()


def test_validator_rejects_corrupt_timelines(edit_ws):
    from app.editing.validator import TimelineValidator
    from app.timeline.clip import Clip
    from app.timeline.keyframes import Keyframe

    ws = edit_ws
    gen(ws)
    p = ws.project
    st = EditState.capture(p)
    v = TimelineValidator(p.assets, p.scenes, p.voice_over.asset_id, st.decisions, st.strategy, st.generation)
    assert not [i for i in v.validate(st.timeline) if i.severity == "error"]

    def errors(mutate):
        s2 = EditState.capture(p)
        mutate(s2)
        return {i.code for i in TimelineValidator(p.assets, p.scenes, p.voice_over.asset_id, s2.decisions, s2.strategy, s2.generation).validate(s2.timeline)
                if i.severity == "error"}

    def first_video(s2):
        return next(c for c in s2.timeline.all_clips() if c.kind == "media" and c.track_id == "track_v1")

    assert "clip.duration" in errors(lambda s: setattr(first_video(s), "duration", -1.0))
    assert "clip.source" in errors(lambda s: (setattr(first_video(s), "source_in", 500.0), setattr(first_video(s), "source_out", 505.0)))
    assert "clip.asset" in errors(lambda s: setattr(first_video(s), "asset_id", "media_99999"))
    assert "clip.track" in errors(lambda s: setattr(first_video(s), "track_id", "track_zzz"))
    assert "keyframe" in errors(lambda s: first_video(s).keyframes.append(Keyframe("scale", 999.0, 1.0)))
    assert "keyframe" in errors(lambda s: first_video(s).keyframes.append(Keyframe("sparkle", 0.0, 1.0)))
    assert "transition" in errors(lambda s: setattr(first_video(s), "transition", {"type": "SPIN", "duration": 1.0}))
    assert "transition" in errors(lambda s: setattr(first_video(s), "transition", {"type": "FADE", "duration": 999.0}))
    assert "clip.nonfinite" in errors(lambda s: setattr(first_video(s), "timeline_start", float("nan")))
    assert "clip.overlap" in errors(lambda s: s.timeline.get_track("track_v1").clips.append(
        Clip("clip_dup", "track_v1", first_video(s).asset_id, first_video(s).timeline_start, 1.0, 0.0, 1.0, scene_id="scene_001", created_by="AI")))
    assert "voice.alignment" in errors(lambda s: setattr(s.timeline.get_track("track_a1").clips[0], "timeline_start", 3.0))
    assert "voice.missing" in errors(lambda s: s.timeline.get_track("track_a1").clips.clear())
    assert "scene.coverage" in errors(lambda s: s.timeline.get_track("track_v3").clips.clear())
    assert "clip.text" in errors(lambda s: next(c for c in s.timeline.all_clips() if c.kind == "text").text.update(content=""))


# ============================================================ plain timeline operations keep working on the AI timeline
def test_insert_split_move_trim_delete_with_undo_redo(edit_ws):
    ws = edit_ws
    p = ws.project
    asset = ws.assets_by["v_long.mp4"]
    c = ws.timeline.add_asset(asset.id, "track_v2", 0.0)  # insert
    assert c.created_by == "USER" and p.timeline.get_clip(c.id) is not None
    right = ws.timeline.split_clip(c.id, 4.0)  # split
    left = p.timeline.get_clip(c.id)
    assert left.duration == pytest.approx(4.0) and right.timeline_start == pytest.approx(4.0) and left.source_out == pytest.approx(right.source_in)
    assert right.source_out == pytest.approx(12.0) and left.source_in == 0.0  # non-destructive: same media, two ranges
    ws.timeline.move_clip(right.id, 6.0)  # move
    assert p.timeline.get_clip(right.id).timeline_start == pytest.approx(6.0)
    ws.timeline.trim_clip(right.id, new_end=10.0)  # trim
    assert p.timeline.get_clip(right.id).duration == pytest.approx(4.0)
    ws.timeline.delete_clip(left.id)  # delete
    assert p.timeline.get_clip(left.id) is None
    for _ in range(4):
        ws.undo()
    assert p.timeline.get_clip(c.id).duration == pytest.approx(12.0) and p.timeline.get_clip(right.id) is None  # back to the unsplit clip
    ws.undo()
    assert p.timeline.get_clip(c.id) is None
    for _ in range(5):
        ws.redo()
    assert p.timeline.get_clip(left.id) is None and p.timeline.get_clip(right.id).duration == pytest.approx(4.0)
    with pytest.raises(Exception):
        ws.timeline.split_clip(right.id, p.timeline.get_clip(right.id).timeline_start)  # not at the very edge
    p.validate()


def test_user_can_split_zoom_and_keep_editing_ai_clips(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    d = pick_motion_scene(ws)
    clip = p.timeline.get_clip(d.target_id)
    mid = clip.timeline_start + clip.duration / 2
    right = ws.timeline.split_clip(clip.id, mid)
    left = p.timeline.get_clip(clip.id)
    assert left.created_by == "USER" and right.created_by == "USER" and left.scene_id == right.scene_id == d.scene_id
    from app.timeline.keyframes import value_at

    assert value_at(left.keyframes, "scale", left.duration) == pytest.approx(value_at(right.keyframes, "scale", 0.0))  # motion stays continuous
    ws.timeline.set_clip_properties(right.id, opacity=0.5)
    ws.timeline.move_clip(right.id, right.timeline_start)  # moving onto itself is fine
    for _ in range(3):  # move, opacity, split
        ws.undo()
    assert p.timeline.get_clip(right.id) is None and p.timeline.get_clip(clip.id).created_by == "AI"
    p.validate()


# ============================================================ preview composition (what the scrubber shows)
def test_preview_composition_shows_visuals_motion_text_transitions_and_music_levels(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    comp = ws.editing.composer()
    sc = p.scenes[3]
    f = comp.frame_at(sc.start + 0.5)
    assert f.scene_id == sc.id and f.voice_active and any(l.kind == "media" for l in f.layers)
    assert comp.frame_at(p.scenes[-1].end + 5).layers == []  # past the end: nothing on screen
    # motion: a clip with keyframes changes scale over time
    d = pick_motion_scene(ws)
    clip = p.timeline.get_clip(d.target_id)
    a = next(l for l in comp.frame_at(clip.timeline_start + 0.01).layers if l.clip_id == clip.id)
    b = next(l for l in comp.frame_at(clip.timeline_end - 0.01).layers if l.clip_id == clip.id)
    assert (a.scale, a.x) != (b.scale, b.x)
    # text appears only while it is on
    t = next(c for c in p.timeline.all_clips() if c.kind == "text")
    assert any(l.kind == "text" and l.text["content"] == t.text["content"] for l in comp.frame_at(t.timeline_start + t.duration / 2).layers)
    assert not any(l.kind == "text" and l.clip_id == t.id for l in comp.frame_at(t.timeline_end + 0.01).layers)
    # transitions: a dissolve cross-fades with the outgoing visual
    dis = next((c for c in p.timeline.all_clips() if c.transition and c.transition["type"] == "DISSOLVE"), None)
    assert dis is not None
    f = comp.frame_at(dis.timeline_start + dis.transition["duration"] / 2)
    main = next(l for l in f.layers if l.clip_id == dis.id)
    out = [l for l in f.layers if l.role == "outgoing"]
    assert 0.2 < main.opacity < 0.8 and out and out[0].opacity == pytest.approx(1 - main.opacity, abs=0.05) and f.transition == "DISSOLVE"
    # audio instructions: music ducks on important lines, fades in at the start and out at the end
    duck = next(x for x in p.editing_decisions.values() if x.type is DecisionType.AUDIO_DUCK and x.parameters["kind"] == "DUCK")
    mid = (duck.parameters["start"] + duck.parameters["end"]) / 2
    base = p.editing_strategy.audio.music_level
    assert comp.music_level_at(mid) < base
    assert comp.music_level_at(0.0) == 0.0 < comp.music_level_at(3.0) and comp.music_level_at(p.scenes[-1].end - 0.05) < base * 0.1
    assert p.editing_strategy.audio.priority == ["VOICE", "SFX", "MUSIC"]


def test_scene_status_rows_and_progress_reporting(edit_ws):
    ws = edit_ws
    rows = ws.editing.scene_rows()
    assert len(rows) == len(ws.project.scenes) and all(r["status"] is SceneEditStatus.PENDING and r["visual_status"] == "APPROVED" for r in rows)
    gen(ws)
    assert ws.editing.progress["state"] == "COMPLETED" and ws.editing.progress["total"] == len(rows)
    rows = ws.editing.scene_rows()
    assert all(r["status"] is SceneEditStatus.COMPLETE and r["decisions"] > 0 and r["min_confidence"] is not None for r in rows)
    assert min(r["min_confidence"] for r in rows) < 70  # evidence regions are honestly flagged low-confidence
    assert any("Scene" in line for line in last(ws).log)


def test_replace_visual_keeps_timing_and_survives_regeneration(edit_ws):
    ws = edit_ws
    gen(ws)
    p = ws.project
    sc = scene_with(ws, lambda s: len(clips(ws, s.id, "media")) == 1 and p.assets.get(clips(ws, s.id, "media")[0].asset_id).type.value == "image")
    clip = clips(ws, sc.id, "media")[0]
    start, dur = clip.timeline_start, clip.duration
    new_asset = ws.assets_by["plain.png"] if clip.asset_id != ws.assets_by["plain.png"].id else ws.assets_by["wide.png"]
    ws.editing.replace_clip_asset(clip.id, new_asset.id)
    now = p.timeline.get_clip(clip.id)
    d = p.editing_decisions[now.ai_decision_id]
    assert now.asset_id == new_asset.id and (now.timeline_start, now.duration) == (start, dur) and now.created_by == "USER"
    assert d.created_by is Creator.USER and d.parameters["operation"] == "REPLACE" and d.overrides_decision_id
    ws.editing.regenerate_scene(sc.id)
    assert ws.jobs.wait_idle(60)
    assert p.timeline.get_clip(clip.id).asset_id == new_asset.id  # the user's replacement is not undone by the AI
    ws.undo()
    ws.undo()
    assert p.timeline.get_clip(clip.id).asset_id == clip.asset_id
    with pytest.raises(EditingError):
        ws.editing.replace_clip_asset(clip.id, p.voice_over.asset_id)
    short = ws.assets_by["v_short.mp4"]
    long_clip = next(c for c in p.timeline.all_clips() if c.kind == "media" and c.track_id in ("track_v1", "track_v3") and c.duration > 3.2 and c.scene_id)
    with pytest.raises(EditingError):
        ws.editing.replace_clip_asset(long_clip.id, short.id)
