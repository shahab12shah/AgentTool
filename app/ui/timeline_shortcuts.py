"""Configurable keyboard shortcuts for the timeline (Phase 10).

``TimelineShortcuts`` is the one registry of timeline key bindings. Every action has an id, a label, default key sequences, a handler and the conditions under which it is possible.
The handlers only call the canvas (``panel.canvas``) or ``ws.edit`` / ``ws.edit_state`` / ``ws.selection``: the registry never edits a clip itself.

How keys reach the timeline
  * Each action is a ``QAction`` (shortcut context ``WidgetWithChildrenShortcut``) added to the timeline panel, so a key only acts while keyboard focus is somewhere inside the
    timeline panel (canvas, track headers, toolbar) - never in the media library, the preview or the inspector, and never while another page is shown.
  * Keys are never stolen from text entry. When the focus widget is a ``QLineEdit`` / ``QTextEdit`` / ``QPlainTextEdit`` / spin box / editable combo box (or anything that enables
    input methods), (1) an event filter accepts Qt's ``ShortcutOverride`` for every key this registry has bound, so Qt delivers the key to the widget instead of firing the action,
    and (2) the action handler itself refuses to run while such a widget has focus. Keys that are not bound here (Ctrl+S, ...) are left alone.
  * Undo / redo already exist as main-window actions and are not registered again. A key that a main-window action uses (Ctrl+S, Ctrl+I, ...) cannot be assigned here.
  * The Timeline menu uses separate ``QAction`` objects with ``WidgetShortcut`` context: they display the CURRENT key sequences (read from this registry) but can only be triggered
    with the mouse, so a key press never runs one action twice.

Customisation: ``Settings.timeline_shortcuts`` stores only the overrides ``{action id: [key sequences]}``; it is loaded tolerantly (unknown ids, unparsable or chorded sequences and
sequences already taken are dropped). ``set_bindings`` / ``set_binding`` refuse a sequence that another action uses and name that action; ``apply`` rebinds everything live.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from typing import Callable

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QMenu, QPlainTextEdit, QTextEdit, QWidget

from app.core.exceptions import TimelineError
from app.logging.logger import get_logger
from app.timeline.timebase import timebase_of

_log = get_logger(__name__)
MAX_KEYS = 2  # key sequences per action (the dialog offers two slots)
PORTABLE = QKeySequence.SequenceFormat.PortableText
NATIVE = QKeySequence.SequenceFormat.NativeText
_PREV, _NEXT = -1, 1


class ShortcutConflict(ValueError):
    """A key sequence is already used by another action. ``owner`` is that action's label."""

    def __init__(self, sequence: str, owner: str) -> None:
        super().__init__(f"{native(sequence)} is already used by “{owner}”. Choose another key or change that action first.")
        self.sequence, self.owner = sequence, owner


# ------------------------------------------------------------------------------------------------------------------------ the action table
@dataclass(frozen=True)
class ActionSpec:
    id: str
    label: str
    group: str
    defaults: tuple[str, ...]
    need: tuple[str, ...] = ()  # conditions checked by ``why_not`` (see ``_NEEDS``)
    repeat: bool = False  # holding the key repeats the action (stepping, zooming, nudging)
    menu_label: str = ""  # the Timeline menu's wording when it differs (toggles are shown with a check mark)
    tip: str = ""  # one line for the dialog / tooltips


def _a(id_: str, label: str, group: str, defaults: str | tuple[str, ...], *need: str, repeat: bool = False, menu: str = "", tip: str = "") -> ActionSpec:
    return ActionSpec(id_, label, group, (defaults,) if isinstance(defaults, str) else tuple(defaults), tuple(need), repeat, menu, tip)


G_PLAY, G_TOOLS, G_EDIT, G_SEL, G_MARK, G_MODE, G_VIEW = "Playback and navigation", "Tools", "Editing", "Selection", "Markers and range", "Modes", "View and zoom"
SPECS: tuple[ActionSpec, ...] = (
    _a("play_pause", "Play / pause", G_PLAY, "Space", "duration", tip="Advances the playhead in real time. The timeline has no audio output."),
    _a("go_start", "Go to start", G_PLAY, "Home", "project"),
    _a("go_end", "Go to end", G_PLAY, "End", "project"),
    _a("frame_prev", "Previous frame", G_PLAY, "Left", "project", repeat=True),
    _a("frame_next", "Next frame", G_PLAY, "Right", "project", repeat=True),
    _a("second_prev", "Back 1 second", G_PLAY, "Shift+Left", "project", repeat=True),
    _a("second_next", "Forward 1 second", G_PLAY, "Shift+Right", "project", repeat=True),
    _a("edit_prev", "Previous edit point", G_PLAY, "Up", "duration", tip="The previous clip edge on any track."),
    _a("edit_next", "Next edit point", G_PLAY, "Down", "duration", tip="The next clip edge on any track."),
    _a("marker_prev", "Previous marker", G_PLAY, "Alt+M", "markers"),
    _a("marker_next", "Next marker", G_PLAY, "Shift+M", "markers"),
    _a("tool_select", "Select tool", G_TOOLS, ("S", "V"), menu="Select tool"),
    _a("tool_blade", "Blade tool", G_TOOLS, "B", menu="Blade tool", tip="Click a clip to cut it; Shift+click cuts every unlocked track."),
    _a("delete", "Delete", G_EDIT, ("Delete", "Backspace"), "selection", "unlocked", tip="Deletes the selected clips and leaves the gap."),
    _a("ripple_delete", "Ripple delete", G_EDIT, ("Shift+Delete", "Shift+Backspace"), "selection", "unlocked", tip="Deletes the selected clips and closes the gap."),
    _a("cut", "Cut", G_EDIT, "Ctrl+X", "selection", "unlocked"),
    _a("copy", "Copy", G_EDIT, "Ctrl+C", "selection"),
    _a("paste", "Paste at playhead", G_EDIT, "Ctrl+V", "project", "clipboard"),
    _a("paste_attributes", "Paste attributes", G_EDIT, "Ctrl+Shift+V", "selection", "attrs"),
    _a("duplicate", "Duplicate", G_EDIT, "Ctrl+D", "selection"),
    _a("split", "Split at playhead", G_EDIT, "Ctrl+B", "project", "clips", tip="Splits the selected clips at the playhead, or every clip under it when nothing is selected."),
    _a("blade_all", "Cut all tracks at playhead", G_EDIT, "Ctrl+Shift+B", "project", "clips"),
    _a("nudge_left", "Nudge left 1 frame", G_EDIT, ",", "selection", "unlocked", repeat=True),
    _a("nudge_right", "Nudge right 1 frame", G_EDIT, ".", "selection", "unlocked", repeat=True),
    _a("nudge_left_10", "Nudge left 10 frames", G_EDIT, "Alt+Left", "selection", "unlocked", repeat=True),
    _a("nudge_right_10", "Nudge right 10 frames", G_EDIT, "Alt+Right", "selection", "unlocked", repeat=True),
    _a("link_clips", "Link clips", G_EDIT, "Ctrl+L", "selection2"),
    _a("unlink_clips", "Unlink clips", G_EDIT, "Ctrl+Shift+L", "selection"),
    _a("group", "Group clips", G_EDIT, "Ctrl+G", "selection2"),
    _a("ungroup", "Ungroup clips", G_EDIT, "Ctrl+Shift+G", "selection"),
    _a("lock_clips", "Lock / unlock clips", G_EDIT, "Ctrl+Alt+L", "selection"),
    _a("select_all", "Select all", G_SEL, ("A", "Ctrl+A"), "clips"),
    _a("select_none", "Select none", G_SEL, "Ctrl+Shift+A", "selection"),
    _a("invert_selection", "Invert selection", G_SEL, "Ctrl+Alt+I", "clips"),
    _a("escape", "Cancel / stop", G_SEL, "Esc", tip="Stops playback, or cancels the drag in progress and clears the selection."),
    _a("add_marker", "Add marker", G_MARK, "M", "project"),
    _a("mark_in", "Mark in", G_MARK, "I", "project"),
    _a("mark_out", "Mark out", G_MARK, "O", "project"),
    _a("clear_range", "Clear in / out range", G_MARK, "Alt+X", "range"),
    _a("toggle_snap", "Toggle snapping", G_MODE, "N", menu="Snapping"),
    _a("toggle_ripple", "Toggle ripple editing", G_MODE, "Shift+R", menu="Ripple editing"),
    _a("toggle_link", "Toggle linked selection", G_MODE, "L", menu="Linked selection", tip="Selecting a clip also selects its linked and grouped clips."),
    _a("zoom_in", "Zoom in", G_VIEW, ("Ctrl+=", "Ctrl++"), repeat=True),
    _a("zoom_out", "Zoom out", G_VIEW, "Ctrl+-", repeat=True),
    _a("zoom_fit", "Zoom to fit", G_VIEW, "Shift+Z", "project"),
    _a("zoom_selection", "Zoom to selection", G_VIEW, "Alt+Z", "selection"),
)
BY_ID: dict[str, ActionSpec] = {s.id: s for s in SPECS}

# the Timeline menu: action ids, "-" separators, ("sub", title, [ids]) submenus, ("dialog", title) for the shortcuts dialog
MENU_LAYOUT: tuple = (
    ("sub", "Tools", ("tool_select", "tool_blade")), "-",
    "cut", "copy", "paste", "paste_attributes", "duplicate", "-",
    "split", "blade_all", "delete", "ripple_delete", "-",
    "select_all", "select_none", "invert_selection", "-",
    "link_clips", "unlink_clips", "group", "ungroup", "lock_clips", "-",
    "add_marker", "mark_in", "mark_out", "clear_range", "-",
    ("sub", "Go to", ("go_start", "go_end", "frame_prev", "frame_next", "second_prev", "second_next", "edit_prev", "edit_next", "marker_prev", "marker_next")),
    ("sub", "Zoom", ("zoom_in", "zoom_out", "zoom_fit", "zoom_selection")),
    ("sub", "Nudge selection", ("nudge_left", "nudge_right", "nudge_left_10", "nudge_right_10")), "-",
    "toggle_snap", "toggle_ripple", "toggle_link", "-",
    "play_pause", "-",
    ("dialog", "Keyboard Shortcuts…"),
)


# ------------------------------------------------------------------------------------------------------------------------ helpers
def normalize(seq: str | QKeySequence) -> str:
    """The portable text of a single key combination ("" when it is empty, invalid or a chord of several keys)."""
    ks = QKeySequence(seq, PORTABLE) if isinstance(seq, str) else seq
    return ks.toString(PORTABLE) if ks.count() == 1 else ""


def native(seq: str) -> str:
    return QKeySequence(seq, PORTABLE).toString(NATIVE)


def is_text_entry(w: QWidget | None) -> bool:
    """True for a widget the user types text into: keys must go to it, not to a shortcut."""
    if w is None:
        return False
    if isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
        return True
    if isinstance(w, QComboBox) and w.isEditable():
        return True
    try:
        return bool(w.testAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled))  # custom text widgets enable input methods
    except RuntimeError:  # the widget is being destroyed
        return False


@dataclass
class _State:
    """What the enable conditions look at, read once per refresh (cheap: counts and flags only)."""

    project: bool = False
    selected: int = 0
    editable: int = 0  # selected clips whose track is not locked
    duration: float = 0.0
    clipboard: bool = False
    attrs: bool = False
    has_range: bool = False
    markers: int = 0


_NEEDS: dict[str, Callable[[_State], str | None]] = {
    "project": lambda s: None if s.project else "Open or create a project first.",
    "selection": lambda s: None if s.selected else "Select a clip first.",
    "selection2": lambda s: None if s.selected >= 2 else "Select two or more clips first.",
    "unlocked": lambda s: None if (not s.selected or s.editable) else "The selected clips are on locked tracks. Unlock the track first.",
    "clips": lambda s: None if s.duration > 0 else "The timeline is empty.",
    "duration": lambda s: None if s.duration > 0 else "The timeline is empty.",
    "clipboard": lambda s: None if s.clipboard else "There is nothing to paste: copy or cut clips first.",
    "attrs": lambda s: None if s.attrs else "There are no copied attributes: use “Copy attributes” on a clip first.",
    "range": lambda s: None if s.has_range else "There is no in / out range to clear.",
    "markers": lambda s: None if s.markers else "There are no markers on the timeline.",
}


class _OverrideGuard(QObject):
    """Installed on text-entry widgets inside the timeline panel: a key this registry has bound is delivered to the widget as typing, not run as a shortcut."""

    def __init__(self, owner: "TimelineShortcuts") -> None:
        super().__init__(owner)
        self._owner = owner

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.ShortcutOverride and isinstance(obj, QWidget) and is_text_entry(obj) and self._owner.claims(event):
            event.accept()
            return True
        return False


# ------------------------------------------------------------------------------------------------------------------------ the registry
class TimelineShortcuts(QObject):
    changed = Signal()  # the key bindings changed (rebound): tooltips and menus refresh

    def __init__(self, parent_window: QWidget | None, ctx, panel: QWidget) -> None:
        super().__init__(panel)
        self.ctx, self.panel = ctx, panel
        self._window = parent_window  # None: the panel's top-level window is used
        self.specs: dict[str, ActionSpec] = dict(BY_ID)
        self.table: dict[str, tuple] = {s.id: (s.label, s.defaults, getattr(self, "_do_" + s.id), s.need) for s in SPECS}  # {id: (label, default keys, handler, conditions)}
        self._overrides: dict[str, list[str]] = {}
        self._bindings: dict[str, list[str]] = {}
        self._bound: set[str] = set()
        self._actions: dict[str, QAction] = {}  # shortcut-bearing actions (hosted on the panel)
        self._menu_actions: dict[str, QAction] = {}  # display-only actions of the Timeline menu
        self._menus: list[QMenu] = []
        self._guard = _OverrideGuard(self)
        for spec in SPECS:
            act = QAction(spec.label, panel)
            act.setObjectName("tlshortcut_" + spec.id)
            act.setProperty("tl_registry", True)
            act.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            act.setAutoRepeat(spec.repeat)
            act.triggered.connect(lambda _checked=False, aid=spec.id: self.trigger(aid))
            panel.addAction(act)
            self._actions[spec.id] = act
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)
        self.apply()

    # ------------------------------------------------------------ small accessors
    @property
    def ws(self):
        return self.ctx.ws

    @property
    def canvas(self):
        return self.panel.canvas

    def window(self) -> QWidget:
        return self._window or self.panel.window()

    def label(self, aid: str) -> str:
        return self.specs[aid].label

    def defaults(self) -> dict[str, list[str]]:
        return {s.id: [normalize(k) for k in s.defaults] for s in SPECS}

    def bindings(self) -> dict[str, list[str]]:
        """The effective key sequences of every action (portable text), defaults merged with the user's overrides."""
        return {aid: list(keys) for aid, keys in self._bindings.items()}

    def keys(self, aid: str) -> list[str]:
        return list(self._bindings.get(aid, ()))

    def key_text(self, aid: str, limit: int = 2) -> str:
        """For tooltips: "Ctrl+B" or "S / V" in the platform's notation ("" when unbound)."""
        return " / ".join(native(k) for k in self.keys(aid)[:limit])

    def action(self, aid: str) -> QAction:
        return self._actions[aid]

    def is_overridden(self, aid: str) -> bool:
        return aid in self._overrides

    # ------------------------------------------------------------ availability
    def snapshot(self) -> _State:
        project = self.ws.project
        if project is None:
            return _State()
        tl, sel = project.timeline, self.ws.selection
        ids = sel.ids
        editable = 0
        for cid in ids:
            try:
                if not tl.find_clip(cid)[0].locked:
                    editable += 1
            except TimelineError:
                pass
        return _State(True, len(ids), editable, tl.duration, self.ws.edit.clipboard is not None, bool(getattr(self.ws.edit, "_attrs", None)), tl.range is not None, len(tl.markers))

    def why_not(self, aid: str, state: _State | None = None) -> str | None:
        """None when the action is possible right now, otherwise the plain-language reason it is not."""
        st = state or self.snapshot()
        for tag in self.specs[aid].need:
            reason = _NEEDS[tag](st)
            if reason:
                return reason
        return None

    def is_enabled(self, aid: str) -> bool:
        return self.why_not(aid) is None

    # ------------------------------------------------------------ running
    def trigger(self, aid: str, *, from_menu: bool = False) -> bool:
        """Run an action. Keyboard activations are ignored while a text field has focus; a refusal is reported in the status line. Returns True when it ran without an error."""
        if aid not in self.specs:
            return False
        if not from_menu and is_text_entry(QApplication.focusWidget()):
            return False
        why = self.why_not(aid)
        if why:
            self.ctx.status(why)
            return False
        return self.ctx.guard(self.panel, self.table[aid][2])

    def _on_focus_changed(self, _old, new) -> None:
        if new is not None and isinstance(new, QWidget) and not new.property("_tlGuard") and is_text_entry(new) and self.panel.isAncestorOf(new):
            self.guard_widget(new)

    def guard_widget(self, w: QWidget) -> None:
        """Make a text-entry widget inside the panel receive the keys this registry has bound."""
        w.installEventFilter(self._guard)
        w.setProperty("_tlGuard", True)

    def claims(self, key_event) -> bool:
        """True when the key combination is bound to one of these actions."""
        return normalize(QKeySequence(key_event.keyCombination())) in self._bound

    # ------------------------------------------------------------ bindings: load / validate / apply / persist
    def window_shortcuts(self) -> list[tuple[str, str]]:
        """[(portable key, label)] of the main-window actions (Save, Import, Undo, ...): these keys cannot be assigned to a timeline action."""
        out: list[tuple[str, str]] = []
        win = self.window()
        if win is None:
            return out
        for act in win.findChildren(QAction):
            if act.property("tl_registry") or act.property("tl_menu"):
                continue
            label = act.text().split("\t")[0].replace("&", "").strip().rstrip("…").strip() or "a main-window command"
            for ks in act.shortcuts():
                n = normalize(ks)
                if n:
                    out.append((n, label))
        return out

    def _parse(self, seqs) -> list[str]:
        if isinstance(seqs, str):
            seqs = [seqs]
        if not isinstance(seqs, (list, tuple)):
            return []
        out: list[str] = []
        for s in seqs:
            n = normalize(s) if isinstance(s, str) else ""
            if n and n not in out:
                out.append(n)
        return out[:MAX_KEYS]

    def _load_overrides(self) -> dict[str, list[str]]:
        raw = getattr(self.ws.settings, "timeline_shortcuts", None)
        if not isinstance(raw, dict):
            return {}
        return {aid: self._parse(v) for aid, v in raw.items() if aid in self.specs and isinstance(v, (str, list, tuple))}

    def _effective(self, overrides: dict[str, list[str]]) -> dict[str, list[str]]:
        """Defaults + overrides with duplicates removed: an overridden action wins a contested key over a default one; keys used by main-window actions are dropped."""
        reserved = {n for n, _label in self.window_shortcuts()}
        eff: dict[str, list[str]] = {}
        taken: set[str] = set(reserved)
        for spec in SPECS:  # overrides first
            if spec.id in overrides:
                keep = [k for k in overrides[spec.id] if k not in taken]
                if len(keep) != len(overrides[spec.id]):
                    _log.warning("Timeline shortcut for %s ignored: key already in use", spec.id)
                eff[spec.id] = keep
                taken.update(keep)
        for spec in SPECS:
            if spec.id not in eff:
                keep = [n for n in (normalize(k) for k in spec.defaults) if n and n not in taken]
                eff[spec.id] = keep
                taken.update(keep)
        return {s.id: eff[s.id] for s in SPECS}

    def apply(self) -> None:
        """(Re)load the user's overrides from the settings and rebind every action live."""
        self._overrides = {aid: v for aid, v in self._load_overrides().items() if v != self.defaults()[aid]}
        self._rebind()

    def _rebind(self) -> None:
        self._bindings = self._effective(self._overrides)
        self._bound = {k for keys in self._bindings.values() for k in keys}
        for aid, act in self._actions.items():
            act.setShortcuts([QKeySequence(k, PORTABLE) for k in self._bindings[aid]])
        for aid, act in self._menu_actions.items():
            act.setShortcuts([QKeySequence(k, PORTABLE) for k in self._bindings[aid]])
        self.changed.emit()

    def find_conflict(self, seq: str, aid: str, pending: dict[str, list[str]] | None = None) -> str | None:
        """The label of the action (timeline or main window) that already uses ``seq`` - ignoring ``aid`` itself - or None. ``pending`` replaces the current bindings (the dialog's unsaved edits)."""
        n = normalize(seq)
        if not n:
            return None
        for other, keys in (pending if pending is not None else self._bindings).items():
            if other != aid and other in self.specs and n in keys:
                return self.specs[other].label
        for key, label in self.window_shortcuts():
            if key == n:
                return label
        return None

    def validate(self, new: dict[str, list[str]]) -> dict[str, list[str]]:
        """Check a complete set of bindings; returns the cleaned version or raises ``ShortcutConflict`` / ``ValueError`` naming the problem."""
        clean: dict[str, list[str]] = {}
        for aid in self.specs:
            seqs = new.get(aid, self.defaults()[aid])
            keys: list[str] = []
            for s in (seqs if isinstance(seqs, (list, tuple)) else [seqs]):
                if not isinstance(s, str) or not s.strip():
                    continue
                n = normalize(s)
                if not n:
                    raise ValueError(f"“{s}” is not a single key combination (use one key, optionally with Ctrl, Shift or Alt).")
                if n not in keys:
                    keys.append(n)
            if len(keys) > MAX_KEYS:
                raise ValueError(f"“{self.label(aid)}” can have at most {MAX_KEYS} key sequences.")
            clean[aid] = keys
        owner: dict[str, str] = {}
        for aid, keys in clean.items():
            for k in keys:
                if k in owner:
                    raise ShortcutConflict(k, self.label(owner[k]))
                owner[k] = aid
        reserved = dict(self.window_shortcuts())
        for k in owner:
            if k in reserved:
                raise ShortcutConflict(k, reserved[k])
        return clean

    def set_bindings(self, new: dict[str, list[str]]) -> None:
        """Replace the bindings (actions missing from ``new`` keep their current keys), persist them in the settings and rebind live. Raises ``ShortcutConflict`` and changes nothing on a conflict."""
        merged = {**self._bindings, **{aid: list(v) for aid, v in new.items() if aid in self.specs}}
        clean = self.validate(merged)
        defaults = self.defaults()
        self._overrides = {aid: keys for aid, keys in clean.items() if keys != defaults[aid]}
        self._save()
        self._rebind()

    def set_binding(self, aid: str, keys: list[str] | str) -> None:
        self.set_bindings({aid: [keys] if isinstance(keys, str) else list(keys)})

    def reset(self, aid: str | None = None) -> None:
        """Restore the default keys of one action (or of all)."""
        if aid is None:
            self.set_bindings(self.defaults())
        else:
            self.set_bindings({aid: self.defaults()[aid]})

    def _save(self) -> None:
        ws = self.ws
        ws.settings = dc_replace(ws.settings, timeline_shortcuts={aid: list(keys) for aid, keys in self._overrides.items()})
        try:
            ws.settings_store.save(ws.settings)
        except OSError:
            _log.warning("Timeline shortcuts could not be saved", exc_info=True)

    def open_dialog(self) -> None:
        from app.ui.dialogs.shortcuts_dialog import ShortcutsDialog  # noqa: PLC0415

        ShortcutsDialog(self, self.window()).exec()

    # ------------------------------------------------------------ the Timeline menu
    def menu_action(self, aid: str, parent: QWidget | None = None) -> QAction:
        """A menu item for ``aid``: shows the current key sequence, runs the action when clicked (never by key: the keys belong to the panel's own actions)."""
        spec = self.specs[aid]
        act = QAction(spec.menu_label or spec.label, parent)
        act.setProperty("tl_menu", True)
        act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        act.setShortcuts([QKeySequence(k, PORTABLE) for k in self._bindings.get(aid, ())])
        if aid in ("tool_select", "tool_blade", "toggle_snap", "toggle_ripple", "toggle_link"):
            act.setCheckable(True)
        act.triggered.connect(lambda _checked=False, a=aid: (self.trigger(a, from_menu=True), self.refresh_menu_state()))
        self._menu_actions[aid] = act
        return act

    def populate_menu(self, menu: QMenu) -> None:
        """Fill the (Timeline) menu from ``MENU_LAYOUT``; enable states and check marks are evaluated every time the menu opens."""
        menu.setToolTipsVisible(True)
        for item in MENU_LAYOUT:
            if item == "-":
                menu.addSeparator()
            elif isinstance(item, str):
                menu.addAction(self.menu_action(item, menu))
            elif item[0] == "sub":
                sub = menu.addMenu(item[1])
                sub.setToolTipsVisible(True)
                for aid in item[2]:
                    sub.addAction(self.menu_action(aid, sub))
            elif item[0] == "dialog":
                act = QAction(item[1], menu)
                act.setProperty("tl_menu", True)
                act.triggered.connect(lambda _checked=False: self.open_dialog())
                menu.addAction(act)
        menu.aboutToShow.connect(self.refresh_menu_state)
        self._menus.append(menu)
        self.refresh_menu_state()

    def refresh_menu_state(self) -> None:
        st = self.snapshot()
        state = self.ws.edit_state
        checks = {"tool_select": state.tool == "select", "tool_blade": state.tool == "blade", "toggle_snap": state.snap.enabled,
                  "toggle_ripple": state.ripple_on, "toggle_link": state.linked_selection}
        for aid, act in self._menu_actions.items():
            why = self.why_not(aid, st)
            act.setEnabled(why is None)
            act.setToolTip(why or self.specs[aid].tip or self.specs[aid].label)
            if aid in checks:
                act.setChecked(checks[aid])

    # ------------------------------------------------------------ handlers (each calls the canvas or ws.edit; the engine's refusal reaches the user through ctx.guard)
    def _frames_per_second(self) -> int:
        project = self.ws.project
        return max(1, timebase_of(project).frames(1.0)) if project else 30

    def _do_play_pause(self) -> None:
        self.panel.toggle_playback()

    def _do_go_start(self) -> None:
        self.canvas.jump_to_start()

    def _do_go_end(self) -> None:
        self.canvas.jump_to_end()

    def _do_frame_prev(self) -> None:
        self.canvas.step_playhead(-1)

    def _do_frame_next(self) -> None:
        self.canvas.step_playhead(1)

    def _do_second_prev(self) -> None:
        self.canvas.step_playhead(-self._frames_per_second())

    def _do_second_next(self) -> None:
        self.canvas.step_playhead(self._frames_per_second())

    def _do_edit_prev(self) -> None:
        self.canvas.jump_to_edge(_PREV)

    def _do_edit_next(self) -> None:
        self.canvas.jump_to_edge(_NEXT)

    def _do_marker_prev(self) -> None:
        self.canvas.jump_to_marker(_PREV)

    def _do_marker_next(self) -> None:
        self.canvas.jump_to_marker(_NEXT)

    def _do_tool_select(self) -> None:
        self.ws.edit_state.update(tool="select")

    def _do_tool_blade(self) -> None:
        self.ws.edit_state.update(tool="blade")

    def _do_delete(self) -> None:
        self.canvas.delete_selection(ripple=False)

    def _do_ripple_delete(self) -> None:
        self.canvas.delete_selection(ripple=True)

    def _do_cut(self) -> None:
        self.canvas.cut_selection()

    def _do_copy(self) -> None:
        self.canvas.copy_selection()

    def _do_paste(self) -> None:
        self.canvas.paste_at_playhead()

    def _do_paste_attributes(self) -> None:
        self.ws.edit.paste_attributes()

    def _do_duplicate(self) -> None:
        self.canvas.duplicate_selection()

    def _do_split(self) -> None:
        self.ws.edit.split()

    def _do_blade_all(self) -> None:
        self.canvas.begin_blade_at_playhead()

    def _do_nudge_left(self) -> None:
        self.canvas.nudge(-1)

    def _do_nudge_right(self) -> None:
        self.canvas.nudge(1)

    def _do_nudge_left_10(self) -> None:
        self.canvas.nudge(-10)

    def _do_nudge_right_10(self) -> None:
        self.canvas.nudge(10)

    def _do_link_clips(self) -> None:
        self.ws.edit.link()

    def _do_unlink_clips(self) -> None:
        self.ws.edit.unlink()

    def _do_group(self) -> None:
        self.ws.edit.group()

    def _do_ungroup(self) -> None:
        self.ws.edit.ungroup()

    def _do_lock_clips(self) -> None:
        self.ws.edit.toggle_item_lock()

    def _do_select_all(self) -> None:
        self.canvas.select_all()

    def _do_select_none(self) -> None:
        self.ws.selection.select_none()

    def _do_invert_selection(self) -> None:
        self.canvas.invert_selection()

    def _do_escape(self) -> None:
        if self.panel.transport.playing:
            self.panel.transport.stop()  # Escape stops playback first and leaves the selection alone
        else:
            self.canvas.escape()

    def _do_add_marker(self) -> None:
        self.ws.edit.add_marker()

    def _do_mark_in(self) -> None:
        self.ws.edit.mark_in()

    def _do_mark_out(self) -> None:
        self.ws.edit.mark_out()

    def _do_clear_range(self) -> None:
        self.ws.edit.clear_range()

    def _do_toggle_snap(self) -> None:
        self.ws.edit_state.toggle_snapping()

    def _do_toggle_ripple(self) -> None:
        self.ws.edit_state.toggle_ripple()

    def _do_toggle_link(self) -> None:
        self.ws.edit_state.toggle_linked_selection()

    def _do_zoom_in(self) -> None:
        self.canvas.zoom_in()

    def _do_zoom_out(self) -> None:
        self.canvas.zoom_out()

    def _do_zoom_fit(self) -> None:
        self.canvas.zoom_to_fit()

    def _do_zoom_selection(self) -> None:
        if self.canvas.zoom_to_selection() is False:
            self.ctx.status("Select a clip first.")
