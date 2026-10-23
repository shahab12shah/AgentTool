"""Session-level timeline editing preferences: the active tool, ripple mode and scope, snapping, linked selection and playhead following.

``TimelineEditState`` is a small observable value holder (Qt-free). Assigning a field validates it (``ValueError`` for an impossible choice) and notifies the listeners with
the set of field names that changed; ``update(**changes)`` changes several fields atomically with a single notification. Everything except the active ``tool`` is a
preference and is persisted in the application settings (``Settings.timeline_ui``); loading is tolerant - an invalid or missing value falls back to its default.
"""

from __future__ import annotations

from typing import Any, Callable

from app.logging.logger import get_logger
from app.timeline.snapping import SnapSettings

TOOLS = ("select", "blade")
RIPPLE_SCOPES = ("track", "tracks", "all")  # the clip's own track | the tracks holding the selection | every unlocked track
FOLLOW_MODES = ("off", "page", "smooth", "center")
FIELDS = ("tool", "ripple_on", "ripple_scope", "snap", "linked_selection", "follow_playhead", "auto_scroll")
PERSISTED = tuple(f for f in FIELDS if f != "tool")  # the tool is per session: the app never starts in the blade tool
_DEFAULTS: dict[str, Any] = {"tool": "select", "ripple_on": False, "ripple_scope": "track", "linked_selection": True, "follow_playhead": "page", "auto_scroll": True}
_CHOICES = {"tool": TOOLS, "ripple_scope": RIPPLE_SCOPES, "follow_playhead": FOLLOW_MODES}
_BOOLS = ("ripple_on", "linked_selection", "auto_scroll")
_log = get_logger(__name__)

Listener = Callable[["TimelineEditState", frozenset], None]


def _coerce(name: str, value: Any) -> Any:
    if name in _CHOICES:
        if value not in _CHOICES[name]:
            raise ValueError(f"{name} must be one of {', '.join(_CHOICES[name])} (got {value!r})")
        return value
    if name in _BOOLS:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False (got {value!r})")
        return value
    if name == "snap":
        if isinstance(value, SnapSettings):
            return value
        if isinstance(value, dict):
            return SnapSettings.from_dict(value)
        raise ValueError("snap must be SnapSettings")
    raise AttributeError(f"TimelineEditState has no field {name!r}")


class TimelineEditState:
    tool: str
    ripple_on: bool
    ripple_scope: str
    snap: SnapSettings
    linked_selection: bool
    follow_playhead: str
    auto_scroll: bool

    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_listeners", [])
        merged: dict[str, Any] = {**_DEFAULTS, "snap": SnapSettings(), **values}
        for k, v in merged.items():
            object.__setattr__(self, k, _coerce(k, v))

    # ------------------------------------------------------------------ changing
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self.update(**{name: value})

    def update(self, **changes: Any) -> frozenset:
        """Apply several changes at once (all valid or none) and notify once. Returns the names that actually changed."""
        checked = {k: _coerce(k, v) for k, v in changes.items()}
        changed = frozenset(k for k, v in checked.items() if getattr(self, k, None) != v)
        for k in changed:
            object.__setattr__(self, k, checked[k])
        if changed:
            for fn in list(self._listeners):
                try:
                    fn(self, changed)
                except Exception:  # noqa: BLE001  (a broken listener must not break the editor)
                    _log.exception("edit-state listener failed")
        return changed

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """``listener(state, changed_names)`` runs after a change. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    # ------------------------------------------------------------------ toggles the toolbar / shortcuts use
    def toggle_ripple(self) -> bool:
        self.ripple_on = not self.ripple_on
        return self.ripple_on

    def toggle_snapping(self) -> bool:
        self.snap = self.snap.replace(enabled=not self.snap.enabled)
        return self.snap.enabled

    def toggle_linked_selection(self) -> bool:
        self.linked_selection = not self.linked_selection
        return self.linked_selection

    # ------------------------------------------------------------------ persistence
    def to_dict(self, *, include_tool: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {k: getattr(self, k) for k in PERSISTED if k != "snap"}
        d["snap"] = self.snap.to_dict()
        if include_tool:
            d["tool"] = self.tool
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "TimelineEditState":
        """Tolerant loader: unknown keys are ignored and every invalid value falls back to its default."""
        values: dict[str, Any] = {}
        if isinstance(d, dict):
            for k in FIELDS:
                if k not in d:
                    continue
                try:
                    values[k] = _coerce(k, d[k])
                except ValueError:
                    continue
        return cls(**values)

    def copy(self) -> "TimelineEditState":
        return TimelineEditState(**{k: getattr(self, k) for k in FIELDS})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TimelineEditState) and all(getattr(self, k) == getattr(other, k) for k in FIELDS)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "TimelineEditState(" + ", ".join(f"{k}={getattr(self, k)!r}" for k in FIELDS) + ")"
