"""Keyboard shortcuts dialog for the timeline: every action with its keys, editable, with conflict detection.

Edits are collected in ``pending`` and only written (``TimelineShortcuts.set_bindings``: validated, persisted in the settings, rebound live) when the user presses Apply. A key
that another timeline action or a main-window command (Save, Import, Undo, ...) already uses is refused on the spot, naming that action.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.timeline_shortcuts import MAX_KEYS, PORTABLE, SPECS, ShortcutConflict, TimelineShortcuts, native, normalize

COL_AREA, COL_ACTION, COL_KEY1, COL_KEY2, COL_RESET = range(5)


class ShortcutsDialog(QDialog):
    def __init__(self, registry: TimelineShortcuts, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Keyboard shortcuts")
        self.resize(760, 640)
        self.pending: dict[str, list[str]] = registry.bindings()
        self._busy = False
        self.editors: dict[tuple[str, int], QKeySequenceEdit] = {}
        self.reset_buttons: dict[str, QPushButton] = {}
        self._rows: dict[str, int] = {}

        intro = QLabel("Timeline shortcuts work while the timeline has keyboard focus (click a clip or the ruler first). They never fire while you type in a text field. "
                       "Click a box and press the keys you want; the × clears it. Undo and redo (Ctrl+Z, Ctrl+Y, Ctrl+Shift+Z) belong to the main window.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by action, area or key…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)

        self.table = QTableWidget(len(SPECS), 5)
        self.table.setObjectName("shortcutTable")
        self.table.setHorizontalHeaderLabels(["Area", "Action", "Shortcut", "Alternative", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(COL_AREA, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(COL_ACTION, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(COL_KEY1, QHeaderView.ResizeMode.Fixed)
        head.setSectionResizeMode(COL_KEY2, QHeaderView.ResizeMode.Fixed)
        head.setSectionResizeMode(COL_RESET, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(COL_KEY1, 150)
        self.table.setColumnWidth(COL_KEY2, 150)
        for row, spec in enumerate(SPECS):
            self._rows[spec.id] = row
            area = QTableWidgetItem(spec.group)
            area.setForeground(self.palette().placeholderText())
            name = QTableWidgetItem(spec.label)
            name.setToolTip(spec.tip or spec.label)
            self.table.setItem(row, COL_AREA, area)
            self.table.setItem(row, COL_ACTION, name)
            for slot, col in enumerate((COL_KEY1, COL_KEY2)):
                ed = QKeySequenceEdit()
                ed.setMaximumSequenceLength(1)
                ed.setClearButtonEnabled(True)
                ed.setObjectName(f"key_{spec.id}_{slot}")
                ed.keySequenceChanged.connect(lambda ks, a=spec.id, s=slot: self._edited(a, s, ks))
                self.table.setCellWidget(row, col, ed)
                self.editors[(spec.id, slot)] = ed
            btn = QPushButton("Reset")
            btn.setToolTip("Restore the default keys of this action: " + (", ".join(native(normalize(k)) for k in spec.defaults) or "none"))
            btn.clicked.connect(lambda _checked=False, a=spec.id: self.reset_one(a))
            self.table.setCellWidget(row, COL_RESET, btn)
            self.reset_buttons[spec.id] = btn
        self.table.verticalHeader().setDefaultSectionSize(34)

        self.message = QLabel("")
        self.message.setObjectName("shortcutMessage")
        self.message.setWordWrap(True)
        self.message.setMinimumHeight(22)
        fixed = sorted({f"{native(key)} {label}" for key, label in registry.window_shortcuts()})
        self.fixed_note = QLabel("Used by the main window (cannot be assigned here): " + ", ".join(fixed) if fixed else "")
        self.fixed_note.setObjectName("muted")
        self.fixed_note.setWordWrap(True)

        self.reset_all_btn = QPushButton("Reset all")
        self.reset_all_btn.setToolTip("Restore every timeline shortcut to its default")
        self.reset_all_btn.clicked.connect(self.reset_all)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        self.apply_btn = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_btn.clicked.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self.reset_all_btn)
        bottom.addStretch(1)
        bottom.addWidget(self.buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.filter)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.message)
        layout.addWidget(self.fixed_note)
        layout.addLayout(bottom)
        for aid in self._rows:
            self._refresh_row(aid)
        self._refresh_buttons()

    # ------------------------------------------------------------ editing
    def _say(self, text: str, error: bool = True) -> None:
        self.message.setText(text)
        self.message.setStyleSheet("color: #ff6b6b;" if error and text else "")

    def _slots(self, aid: str) -> list[str]:
        keys = list(self.pending.get(aid, []))[:MAX_KEYS]
        return keys + [""] * (MAX_KEYS - len(keys))

    def _refresh_row(self, aid: str) -> None:
        self._busy = True
        try:
            for slot, key in enumerate(self._slots(aid)):
                self.editors[(aid, slot)].setKeySequence(QKeySequence(key, PORTABLE) if key else QKeySequence())
        finally:
            self._busy = False
        default = self.registry.defaults()[aid]
        changed = self.pending.get(aid, []) != default
        self.reset_buttons[aid].setEnabled(changed)
        row = self._rows[aid]
        font = QFont(self.font())
        font.setBold(changed)
        self.table.item(row, COL_ACTION).setFont(font)

    def _refresh_buttons(self) -> None:
        self.apply_btn.setEnabled(self.pending != self.registry.bindings())

    def _edited(self, aid: str, slot: int, ks: QKeySequence) -> None:
        if self._busy:
            return
        n = normalize(ks) if not ks.isEmpty() else ""
        if ks.count() > 1:
            self._say("Use a single key combination (one key, optionally with Ctrl, Shift or Alt).")
            self._refresh_row(aid)
            return
        slots = self._slots(aid)
        if n:
            owner = self.registry.find_conflict(n, aid, pending=self.pending)
            if owner:
                self._say(str(ShortcutConflict(n, owner)))
                self._refresh_row(aid)
                return
            if n in slots and slots.index(n) != slot:
                self._say(f"{native(n)} is already assigned to this action.")
                self._refresh_row(aid)
                return
        slots[slot] = n
        self.pending[aid] = [k for k in slots if k]
        self._say("")
        self._refresh_row(aid)
        self._refresh_buttons()

    def reset_one(self, aid: str) -> bool:
        defaults = self.registry.defaults()[aid]
        for key in defaults:
            owner = self.registry.find_conflict(key, aid, pending=self.pending)
            if owner:
                self._say(str(ShortcutConflict(key, owner)))
                return False
        self.pending[aid] = list(defaults)
        self._say("")
        self._refresh_row(aid)
        self._refresh_buttons()
        return True

    def reset_all(self) -> None:
        self.pending = self.registry.defaults()
        self._say("")
        for aid in self._rows:
            self._refresh_row(aid)
        self._refresh_buttons()

    def apply(self) -> bool:
        """Validate and persist the edits; they take effect immediately. Returns False (and shows why) when something is refused."""
        try:
            self.registry.set_bindings(self.pending)
        except (ShortcutConflict, ValueError) as exc:
            self._say(str(exc))
            return False
        self.pending = self.registry.bindings()
        for aid in self._rows:
            self._refresh_row(aid)
        self._refresh_buttons()
        self._say("Shortcuts saved.", error=False)
        return True

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for spec in SPECS:
            hay = " ".join([spec.label, spec.group, *(native(k) for k in self.pending.get(spec.id, []))]).lower()
            self.table.setRowHidden(self._rows[spec.id], bool(needle) and needle not in hay)
