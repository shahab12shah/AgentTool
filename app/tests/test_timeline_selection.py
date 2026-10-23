"""Phase 10 (W1-D): the selection model, the timeline edit state and their workspace plumbing."""

from __future__ import annotations

import json
import random

import pytest

from app.core.config import Settings
from app.core.events import Topics
from app.core.exceptions import TimelineError
from app.services.workspace import Workspace
from app.timeline.clip import Clip
from app.timeline.edit_state import FOLLOW_MODES, RIPPLE_SCOPES, TimelineEditState
from app.timeline.selection import MODES, SelectionModel
from app.timeline.snapping import SnapSettings
from app.timeline.timeline import Timeline
from app.timeline.timeline_commands import AddClipCommand, DeleteClipCommand


def mk(cid: str, track: str, start: float, dur: float = 1.0, **kw) -> Clip:
    return Clip(id=cid, track_id=track, asset_id="a1", timeline_start=start, duration=dur, **kw)


@pytest.fixture
def tl() -> Timeline:
    """V1: a b c d (0-1, 2-3, 4-5, 6-7)   V2: e f (1-2, 5-7)   A1: g (0-8)   V3 (images): empty"""
    t = Timeline.default()
    for c in (mk("a", "track_v1", 0), mk("b", "track_v1", 2), mk("c", "track_v1", 4), mk("d", "track_v1", 6),
              mk("e", "track_v2", 1), mk("f", "track_v2", 5, 2), mk("g", "track_a1", 0, 8)):
        t.insert_clip(c)
    return t


# ---------------------------------------------------------------- explicit selection modes
def test_replace_toggle_add_remove_and_primary():
    s = SelectionModel()
    assert s.ids == [] and s.primary is None and not s and len(s) == 0
    assert s.select(["a", "b", "c"]) is True
    assert (s.ids, s.primary) == (["a", "b", "c"], "c")  # replace: the last id is the primary
    assert s.select(["b"], "toggle") is True and s.ids == ["a", "c"] and s.primary == "c"
    assert s.select(["d"], "toggle") and s.ids == ["a", "c", "d"] and s.primary == "d"
    assert s.select(["c"], "toggle") and s.ids == ["a", "d"] and s.primary == "d"
    assert s.select(["e", "a"], "add") and s.ids == ["a", "d", "e"] and s.primary == "a"  # order of first selection is kept, the primary is the last id given
    assert s.select(["a"], "remove") and s.ids == ["d", "e"] and s.primary == "e"  # the primary left: the most recent remaining id takes over
    assert s.select(["zzz"], "remove") is False
    assert s.select(["d", "e"], "replace") is False  # nothing changed -> False (primary was e, stays e)
    assert s.select("x") and s.ids == ["x"]  # a single id string is accepted
    assert s.select(None) and s.ids == [] and s.primary is None
    assert s.select(["", None, "a", "a", 5], "replace") and s.ids == ["a"]  # junk and duplicates are ignored


def test_explicit_primary_and_invariant():
    s = SelectionModel()
    s.select(["a", "b", "c"], primary="a")
    assert s.primary == "a" and s.ids == ["a", "b", "c"]
    s.select(["z"], "add", primary="a")  # a primary that is selected wins over the last added id
    assert s.primary == "a"
    s.select(["q"], "replace", primary="nope")  # an invalid primary is ignored
    assert s.primary == "q"
    with pytest.raises(ValueError):
        s.select(["a"], "sideways")
    assert MODES == ("replace", "toggle", "add", "remove")


def test_listeners_fire_once_per_change_and_can_unsubscribe():
    s = SelectionModel()
    seen = []
    off = s.subscribe(lambda m: seen.append((m.ids, m.primary)))
    s.select(["a"])
    s.select(["a"])  # unchanged
    s.select(["a", "b"], "add")
    s.select(["b"], "remove")
    s.select_none()
    s.select_none()
    assert seen == [(["a"], "a"), (["a", "b"], "b"), (["a"], "a"), ([], None)]
    off()
    s.select(["z"])
    assert len(seen) == 4

    def broken(_m):
        raise RuntimeError("boom")

    s.subscribe(broken)
    s.select(["y"])  # a failing listener never breaks the editor
    assert s.ids == ["y"]


def test_random_operations_match_an_ordered_list_model():
    rng = random.Random(3)
    universe = [f"c{i}" for i in range(12)]
    s = SelectionModel()
    model: list[str] = []
    for _ in range(600):
        mode = rng.choice(MODES)
        ids = rng.sample(universe, rng.randint(0, 4))
        s.select(ids, mode)
        if mode == "replace":
            model = list(ids)
        elif mode == "add":
            model += [i for i in ids if i not in model]
        elif mode == "remove":
            model = [i for i in model if i not in ids]
        else:
            for i in ids:
                model.remove(i) if i in model else model.append(i)
        assert s.ids == model
        assert (s.primary is None) == (not model) and (s.primary in model or not model)


# ---------------------------------------------------------------- bulk selectors on a timeline
def test_select_all_track_and_invert_across_tracks(tl):
    s = SelectionModel()
    assert s.select_all(tl) and set(s.ids) == set("abcdefg") and s.primary in s.ids
    s.select_none()
    assert s.select_track(tl, "track_v1") and s.ids == ["a", "b", "c", "d"]
    assert s.select_track(tl, "track_v2", "add") and s.ids == ["a", "b", "c", "d", "e", "f"]
    assert s.primary == "a"  # a bulk selection keeps the old primary while it stays selected
    assert s.select_track(tl, "track_v2", "remove") and s.ids == ["a", "b", "c", "d"]
    assert s.select_track(tl, "track_v3") is True and s.ids == [] and s.primary is None  # an empty track selects nothing
    with pytest.raises(TimelineError):
        s.select_track(tl, "nope")
    s.select(["a", "g"])
    assert s.invert(tl) and set(s.ids) == set("bcdef")
    assert s.invert(tl) and set(s.ids) == {"a", "g"}


def test_select_range_overlap_is_open_interval_and_track_filtered(tl):
    s = SelectionModel()
    s.select_range(tl, 1.5, 4.5)
    assert set(s.ids) == {"b", "c", "e", "g"}  # a ends at 1.0, d starts at 6; e (1-2) overlaps
    s.select_range(tl, 1.0, 2.0)  # touching clips are not inside the range: a ends at 1.0, b starts at 2.0
    assert set(s.ids) == {"e", "g"}
    s.select_range(tl, 4.5, 1.5, ["track_v1"])  # reversed times are accepted
    assert s.ids == ["b", "c"]
    s.select_range(tl, 0, 100, ["track_v2", "track_a1", "track_v2"])
    assert set(s.ids) == {"e", "f", "g"}
    s.select_range(tl, 0, 100, ["track_a1"], "add")
    assert set(s.ids) == {"e", "f", "g"}
    s.select_range(tl, 5.5, 5.5)  # a zero-width range selects the clips that contain the instant
    assert set(s.ids) == {"f", "g"}


def test_select_rect_geometry_in_time_and_track_space(tl):
    # tracks: 0 V1, 1 V2, 2 V3 images, 3 V4 graphics, 4 V5 text, 5 V6 captions, 6 A1, 7 A2, 8 A3
    s = SelectionModel()
    s.select_rect(tl, 1.5, 4.5, 0, 0)
    assert s.ids == ["b", "c"]
    s.select_rect(tl, 1.5, 4.5, 0, 1)
    assert set(s.ids) == {"b", "c", "e"}
    s.select_rect(tl, 1.5, 4.5, 1, 0)  # reversed rows
    assert set(s.ids) == {"b", "c", "e"}
    s.select_rect(tl, 0, 10, 0, 8)
    assert set(s.ids) == set("abcdefg")
    s.select_rect(tl, 0, 10, 6, 99)  # clamped to the last track
    assert s.ids == ["g"]
    s.select_rect(tl, 0, 10, -5, 0)
    assert s.ids == ["a", "b", "c", "d"]
    s.select_rect(tl, 0, 10, 2, 5)  # only empty tracks
    assert s.ids == []
    s.select_rect(tl, 0.5, 1.5, 0, 1)
    s.select_rect(tl, 5.5, 6.5, 0, 1, "add")  # shift + marquee adds
    assert set(s.ids) == {"a", "e", "f", "d"}
    s.select_rect(tl, 5.5, 6.5, 0, 1, "toggle")  # ctrl + marquee toggles
    assert set(s.ids) == {"a", "e"}
    empty = Timeline([])
    assert s.select_rect(empty, 0, 1, 0, 3) is True and s.ids == []


# ---------------------------------------------------------------- shift-click
def test_extend_to_selects_the_track_range_between_anchor_and_clicked_clip(tl):
    s = SelectionModel()
    s.select(["b"])  # the plain click that sets the anchor
    assert s.extend_to(tl, "d") and s.ids == ["b", "c", "d"] and s.primary == "d" and s.anchor == "b"
    assert s.extend_to(tl, "a") and s.ids == ["a", "b"] and s.primary == "a"  # pivots around the same anchor, replacing the previous range
    assert s.extend_to(tl, "b") and s.ids == ["b"]
    s.select(["b"])
    s.extend_to(tl, "d")
    s.select(["e"], "add")
    assert s.anchor == "e"  # a ctrl-click moves the anchor
    s.select(["a"])
    assert s.extend_to(tl, "e") and s.ids == ["e"]  # different track -> a plain click
    s.select(["a"])
    s.extend_to(tl, "c", additive=True)
    s.extend_to(tl, "f", additive=True)  # f is on another track: added on its own
    assert s.ids == ["a", "b", "c", "f"]


def test_extend_to_without_a_usable_anchor_is_a_plain_click(tl):
    s = SelectionModel()
    assert s.extend_to(tl, "c") and s.ids == ["c"] and s.anchor == "c"
    s.select(["a"])
    tl.detach_clip("a")
    assert s.extend_to(tl, "c") and s.ids == ["c"]  # the anchor was deleted
    with pytest.raises(TimelineError):
        s.extend_to(tl, "nope")


def test_extend_to_is_in_time_order_not_list_order():
    t = Timeline.default()
    for c in (mk("late", "track_v1", 10), mk("early", "track_v1", 0), mk("mid", "track_v1", 5)):
        t.insert_clip(c)
    s = SelectionModel()
    s.select(["late"])
    s.extend_to(t, "early")
    assert s.ids == ["early", "mid", "late"]


# ---------------------------------------------------------------- prune / undo / redo
def test_prune_drops_missing_ids_and_fixes_primary(tl):
    s = SelectionModel()
    s.select(["a", "b", "c"])
    seen = []
    s.subscribe(lambda m: seen.append(list(m.ids)))
    assert s.prune(tl) is False and not seen
    tl.detach_clip("c")
    assert s.prune(tl) and s.ids == ["a", "b"] and s.primary == "b" and seen == [["a", "b"]]
    tl.detach_clip("a")
    tl.detach_clip("b")
    assert s.prune(tl) and s.ids == [] and s.primary is None and s.anchor is None
    assert s.prune(tl) is False
    # a big selection takes the single-pass path
    s.select_all(tl)
    keep = s.ids
    assert len(keep) > 3
    for i in range(12):
        tl.insert_clip(mk(f"x{i}", "track_v3", float(i) * 2))
    s.select([f"x{i}" for i in range(12)] + ["gone"])
    assert s.prune(tl) and s.ids == [f"x{i}" for i in range(12)]


def test_selection_survives_undo_redo_while_ids_exist(tl):
    from app.core.commands import CommandStack

    stack = CommandStack()
    s = SelectionModel()
    stack.execute(AddClipCommand(tl, mk("n", "track_v3", 0)))
    s.select(["a", "n"])
    stack.execute(DeleteClipCommand(tl, "b"))
    s.prune(tl)
    assert s.ids == ["a", "n"]  # b was not selected
    stack.undo()  # b is back
    s.prune(tl)
    assert s.ids == ["a", "n"]  # undoing a delete of an unselected clip leaves the selection alone
    stack.undo()  # the add of n is undone: n disappears
    s.prune(tl)
    assert s.ids == ["a"] and s.primary == "a"
    stack.redo()
    s.prune(tl)
    assert s.ids == ["a"]  # a vanished id is not resurrected by redo
    s.select(["n"])
    stack.redo()  # delete b again
    s.prune(tl)
    assert s.ids == ["n"]


# ---------------------------------------------------------------- relatives
def test_expand_relatives_with_links_and_groups(tl):
    tl.get_clip("a").link_id = "L1"
    tl.get_clip("g").link_id = "L1"  # video + its audio
    tl.get_clip("c").group_id = "G1"
    tl.get_clip("e").group_id = "G1"
    s = SelectionModel()
    s.select(["a", "c"])
    assert s.expand_relatives(tl, True, True) == ["a", "c", "e", "g"]  # the selected ids first, then relatives in timeline order
    assert s.expand_relatives(tl, True, False) == ["a", "c", "g"]
    assert s.expand_relatives(tl, False, True) == ["a", "c", "e"]
    assert s.expand_relatives(tl, False, False) == ["a", "c"]
    assert s.ids == ["a", "c"]  # expanding never changes the selection
    s.select(["a", "gone"])
    assert s.expand_relatives(tl, True, True) == ["a", "g"]  # vanished ids are ignored
    s.select_none()
    assert s.expand_relatives(tl) == []


def test_click_selects_the_family_and_toggle_removes_it(tl):
    tl.get_clip("a").link_id = "L1"
    tl.get_clip("g").link_id = "L1"
    tl.get_clip("c").group_id = "G1"
    tl.get_clip("e").group_id = "G1"
    s = SelectionModel()
    assert s.click(tl, "a") and s.ids == ["a", "g"] and s.primary == "a"
    assert s.click(tl, "a", linked=False) and s.ids == ["a"]  # linked selection switched off
    assert s.click(tl, "c") and s.ids == ["c", "e"] and s.primary == "c"
    assert s.click(tl, "a", "toggle") and s.ids == ["c", "e", "a", "g"] and s.primary == "a"
    assert s.click(tl, "g", "toggle") and s.ids == ["c", "e"]  # toggling a selected member removes the whole family
    assert s.click(tl, "c", "add", grouped=False) and s.ids == ["c", "e"] and s.primary == "c"  # grouping switched off: only the clicked clip is (re)added, it becomes the primary
    with pytest.raises(TimelineError):
        s.click(tl, "nope")


def test_span_and_selected_clips(tl):
    s = SelectionModel()
    assert s.span(tl) is None
    s.select(["b", "f", "gone"])
    assert s.span(tl) == (2.0, 7.0)
    assert [c.id for c in s.selected_clips(tl)] == ["b", "f"]


# ---------------------------------------------------------------- edit state
def test_edit_state_defaults_and_validation():
    st = TimelineEditState()
    assert (st.tool, st.ripple_on, st.ripple_scope, st.linked_selection, st.follow_playhead, st.auto_scroll) == ("select", False, "track", True, "page", True)
    assert st.snap == SnapSettings() and st.snap.enabled
    st.tool = "blade"
    with pytest.raises(ValueError):
        st.tool = "razor"
    with pytest.raises(ValueError):
        st.ripple_scope = "everything"
    with pytest.raises(ValueError):
        st.follow_playhead = "scroll"
    with pytest.raises(ValueError):
        st.ripple_on = "yes"
    with pytest.raises(ValueError):
        st.snap = 5
    with pytest.raises(AttributeError):
        st.colour = "red"
    assert st.tool == "blade" and set(RIPPLE_SCOPES) == {"track", "tracks", "all"} and "smooth" in FOLLOW_MODES
    st.snap = {"enabled": False, "tolerance_frames": 8}  # a plain dict is converted
    assert isinstance(st.snap, SnapSettings) and not st.snap.enabled and st.snap.tolerance_frames == 8


def test_edit_state_listeners_and_atomic_update():
    st = TimelineEditState()
    seen = []
    off = st.subscribe(lambda s, changed: seen.append(set(changed)))
    st.ripple_on = True
    st.ripple_on = True  # unchanged: silent
    assert st.update(ripple_scope="all", linked_selection=False, auto_scroll=True) == {"ripple_scope", "linked_selection"}
    with pytest.raises(ValueError):
        st.update(tool="blade", follow_playhead="bogus")  # all or nothing
    assert st.tool == "select"
    assert st.toggle_ripple() is False and st.toggle_snapping() is False and st.toggle_linked_selection() is True
    assert seen == [{"ripple_on"}, {"ripple_scope", "linked_selection"}, {"ripple_on"}, {"snap"}, {"linked_selection"}]
    off()
    st.tool = "blade"
    assert len(seen) == 5


def test_edit_state_roundtrip_and_tolerant_loading():
    st = TimelineEditState(tool="blade", ripple_on=True, ripple_scope="all", snap=SnapSettings(False, 8, frozenset({"markers", "words"})),
                           linked_selection=False, follow_playhead="center", auto_scroll=False)
    d = st.to_dict()
    assert "tool" not in d  # the tool is per session
    assert json.loads(json.dumps(d)) == d
    back = TimelineEditState.from_dict(json.loads(json.dumps(d)))
    assert back.tool == "select" and back.copy().ripple_scope == "all"
    back.tool = "blade"
    assert back == st
    assert TimelineEditState.from_dict(st.to_dict(include_tool=True)).tool == "blade"
    assert TimelineEditState.from_dict(None) == TimelineEditState() and TimelineEditState.from_dict([1]) == TimelineEditState()
    bad = TimelineEditState.from_dict({"ripple_on": "yes", "ripple_scope": "nope", "snap": "x", "follow_playhead": "smooth", "auto_scroll": 1, "unknown": 1, "linked_selection": False})
    assert bad.ripple_on is False and bad.ripple_scope == "track" and bad.snap == SnapSettings() and bad.follow_playhead == "smooth" and bad.auto_scroll is True and bad.linked_selection is False


# ---------------------------------------------------------------- settings
def test_settings_timeline_ui_field_is_sanitised(tmp_path):
    s = Settings()
    assert s.timeline_ui == {}
    s.timeline_ui = "oops"  # type: ignore[assignment]
    assert s.sanitized().timeline_ui == {}
    s.timeline_ui = {"ripple_on": True}
    assert s.sanitized().timeline_ui == {"ripple_on": True}
    from app.core.config import SettingsStore

    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.save(Settings(timeline_ui={"snap": {"tolerance_frames": 2}}))
    assert store.load().timeline_ui == {"snap": {"tolerance_frames": 2}}
    path.write_text(json.dumps({"theme": "light"}))  # a settings file from before Phase 10
    assert store.load().timeline_ui == {} and store.load().theme == "light"


# ---------------------------------------------------------------- workspace plumbing
def _events(ws, topic):
    got = []
    ws.bus.subscribe(topic, lambda _t, p: got.append(p))
    return got


def test_workspace_select_clip_keeps_working_and_publishes_both_payload_forms(ws):
    got = _events(ws, Topics.SELECTION_CHANGED)
    assert ws.selected_clip_id is None and ws.selection.ids == []
    ws.select_clip("c1")
    ws.select_clip("c1")  # unchanged -> no event
    assert ws.selected_clip_id == "c1" and got == [{"clip_id": "c1", "clip_ids": ["c1"]}]
    ws.select_clip(None)
    assert got[-1] == {"clip_id": None, "clip_ids": []} and ws.selected_clip_id is None and len(got) == 2
    ws.select_clip(None)
    assert len(got) == 2
    ws.selected_clip_id = "c2"  # the old attribute is still assignable
    assert ws.selection.ids == ["c2"] and got[-1]["clip_id"] == "c2"


def test_workspace_set_selection_modes_and_primary(ws):
    got = _events(ws, Topics.SELECTION_CHANGED)
    ws.set_selection(["a", "b", "c"])
    assert ws.selected_clip_id == "c" and ws.selected_clip_ids == ["a", "b", "c"] and got[-1] == {"clip_id": "c", "clip_ids": ["a", "b", "c"]}
    ws.set_selection(["c"], "toggle")
    assert ws.selected_clip_id == "b" and got[-1]["clip_ids"] == ["a", "b"]
    ws.set_selection(["z", "y"], "add", primary="z")
    assert ws.selected_clip_id == "z" and ws.selected_clip_ids == ["a", "b", "z", "y"]
    ws.set_selection(["a"], "replace", primary="a")
    assert ws.selected_clip_id == "a"
    n = len(got)
    ws.set_selection(["a"])
    assert len(got) == n
    # the model is public for the canvas (marquee, shift-click ...) and every change made on it still reaches the bus
    ws.selection.select(["q", "r"], "replace")
    assert got[-1] == {"clip_id": "r", "clip_ids": ["q", "r"]}


def test_workspace_prunes_the_selection_after_project_changes_and_clears_on_project_switch(project_ws, tmp_path):
    ws = project_ws
    tl = ws.project.timeline
    for c in (mk("a", "track_v1", 0), mk("b", "track_v1", 2), mk("c", "track_v2", 0)):
        tl.insert_clip(c)
    got = _events(ws, Topics.SELECTION_CHANGED)
    ws.set_selection(["a", "b", "c"])
    ws.commands.execute(DeleteClipCommand(tl, "b"))  # publishes PROJECT_CHANGED
    assert ws.selection.ids == ["a", "c"] and got[-1] == {"clip_id": "c", "clip_ids": ["a", "c"]}
    ws.undo()  # b is back but was dropped from the selection and stays dropped
    assert ws.selection.ids == ["a", "c"] and tl.get_clip("b") is not None
    ws.set_selection(["c"])
    ws.commands.execute(DeleteClipCommand(tl, "c"))
    assert ws.selected_clip_id is None and ws.selection.ids == []
    ws.undo()
    ws.set_selection(["a", "b"])
    ws.new_project("Second", tmp_path / "projects2")  # a new project starts with nothing selected
    assert ws.selection.ids == [] and ws.selected_clip_id is None and got[-1] == {"clip_id": None, "clip_ids": []}


def test_workspace_edit_state_is_persisted_but_the_tool_and_selection_are_not(app_paths):
    w = Workspace(app_paths)
    try:
        events = _events(w, Topics.TIMELINE_EDIT_STATE_CHANGED)
        assert w.edit_state == TimelineEditState() and w.settings.timeline_ui == {}
        w.edit_state.tool = "blade"
        assert events == [{"changed": ["tool"]}]
        assert not app_paths.settings_file.exists() or json.loads(app_paths.settings_file.read_text()).get("timeline_ui") in ({}, None)  # the tool is never persisted
        w.edit_state.update(ripple_on=True, ripple_scope="tracks")
        w.edit_state.snap = w.edit_state.snap.replace(tolerance_frames=8).with_target("words", True)
        w.select_clip("c1")
        assert events[-1] == {"changed": ["snap"]}
        on_disk = json.loads(app_paths.settings_file.read_text())["timeline_ui"]
        assert on_disk["ripple_on"] is True and on_disk["ripple_scope"] == "tracks" and on_disk["snap"]["tolerance_frames"] == 8 and "words" in on_disk["snap"]["targets"]
        assert "c1" not in json.dumps(on_disk) and "tool" not in on_disk and w.settings.timeline_ui == on_disk
    finally:
        w.shutdown()
    w2 = Workspace(app_paths)
    try:
        assert w2.edit_state.ripple_on and w2.edit_state.ripple_scope == "tracks" and w2.edit_state.tool == "select"
        assert w2.edit_state.snap.tolerance_frames == 8 and w2.edit_state.snap.wants("words")
        assert w2.selection.ids == []
        w2.update_settings(Settings(**{**w2.settings.__dict__, "theme": "light"}))  # saving other settings keeps the stored preferences
        assert json.loads(app_paths.settings_file.read_text())["timeline_ui"]["ripple_on"] is True
    finally:
        w2.shutdown()


def test_workspace_survives_a_corrupt_timeline_ui_setting(app_paths):
    app_paths.ensure()
    app_paths.settings_file.write_text(json.dumps({"timeline_ui": {"ripple_scope": "nope", "snap": 7, "auto_scroll": "maybe"}}))
    w = Workspace(app_paths)
    try:
        assert w.edit_state == TimelineEditState()
    finally:
        w.shutdown()
    app_paths.settings_file.write_text(json.dumps({"timeline_ui": [1, 2]}))
    w = Workspace(app_paths)
    try:
        assert w.edit_state == TimelineEditState() and w.settings.timeline_ui == {}
    finally:
        w.shutdown()
