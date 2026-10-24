"""Validated inputs for the inspector: a timecode field and a numeric field.

Both follow the same rules, so the inspector never needs its own validation:
  * the widget never changes the project; it emits ``committed`` with the typed value once the user finishes (Enter / leaving the field) and only when the text really changed;
  * nothing is clamped silently: a number outside the engine's range is still committed, the ENGINE refuses it with its plain-language message and the inspector refreshes the
    field back to the real value (``set_model`` / ``set_seconds`` always win over a stale edit once the edit has been committed or discarded);
  * while the user is typing (``is_editing()``) a refresh coming from the model leaves the field alone, so typing is never interrupted;
  * a hint (red outline + tooltip) is shown for text that cannot be understood or that lies outside the range the engine documents; the hint never replaces the engine's check.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit

from app.timeline.timebase import Timebase

DANGER = "#ff6b6b"
MIXED = "—"
_BAD_STYLE = f"border: 1px solid {DANGER};"


class TimecodeEdit(QLineEdit):
    """``HH:MM:SS:FF`` (non-drop) field that also accepts seconds (``1.25`` / ``1.25s``) and frames (``12f``) - exactly what ``Timebase.parse`` accepts."""

    committed = Signal(float)  # a valid, different time (seconds)
    invalid = Signal(str)  # the text could not be understood (plain-language reason)

    def __init__(self, timebase: Timebase | None = None, parent=None) -> None:
        super().__init__(parent)
        self._tb = timebase or Timebase(30)
        self._value: float | None = 0.0
        self._shown = ""
        self._base_tip = "Time as HH:MM:SS:FF, seconds (1.25) or frames (12f)."
        self._flagged = False
        self.setPlaceholderText("00:00:00:00")
        self.setToolTip(self._base_tip)
        self.textEdited.connect(self._live_check)
        self.editingFinished.connect(self._finish)
        self.set_seconds(0.0, force=True)

    # ------------------------------------------------------------------ model -> view
    def set_timebase(self, tb: Timebase) -> None:
        if tb != self._tb:
            self._tb = tb
            self.set_seconds(self._value, force=True)

    def timebase(self) -> Timebase:
        return self._tb

    def is_editing(self) -> bool:
        """True while the user has typed something that was not committed or discarded yet (a model refresh must not touch the field then)."""
        return self.isModified()

    def set_seconds(self, seconds: float | None, *, force: bool = False) -> None:
        """Show ``seconds`` (``None`` = not applicable / mixed values: shows an em dash). Ignored while the user is typing unless ``force``."""
        if self.is_editing() and not force:
            self._value = None if seconds is None else float(seconds)  # remembered: discarding the edit shows the latest model value
            return
        self._value = None if seconds is None else float(seconds)
        self._shown = MIXED if seconds is None else self._tb.format(seconds)
        self.setText(self._shown)  # setText also clears the modified flag
        self._flag("")
        self._update_tip()

    def discard_edit(self) -> None:
        """Throw away an uncommitted edit and show the model value again."""
        self.set_seconds(self._value, force=True)

    def value(self) -> float:
        return 0.0 if self._value is None else self._value

    def setValue(self, seconds: float) -> None:  # compatibility with the old spin box API: shows a value, never commits
        self.set_seconds(seconds, force=True)

    def set_base_tooltip(self, text: str) -> None:
        self._base_tip = text
        self._update_tip()

    def _update_tip(self) -> None:
        if not self._flagged:
            exact = "" if self._value is None else f"  ({self._value:.4f} s)"
            self.setToolTip(self._base_tip + exact)

    # ------------------------------------------------------------------ validation
    def _flag(self, message: str) -> None:
        if bool(message) != self._flagged:
            self.setStyleSheet(_BAD_STYLE if message else "")
        self._flagged = bool(message)
        if message:
            self.setToolTip(message)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.is_editing():
            self.discard_edit()  # Escape throws an uncommitted edit away
            event.accept()
            return
        super().keyPressEvent(event)

    def parse(self, text: str) -> float:
        """Seconds for typed text (raises ``ValueError`` with the reason)."""
        t = self._tb.parse(text)
        if not math.isfinite(t):
            raise ValueError("Enter a finite time.")
        return t

    def _live_check(self, text: str) -> None:
        try:
            self.parse(text)
            self._flag("")
            self._update_tip()
        except ValueError as exc:
            self._flag(str(exc))

    def _finish(self) -> None:
        if not self.is_editing() or self.text() == self._shown:
            if self.text() != self._shown:
                self.set_seconds(self._value, force=True)
            self.setModified(False)
            return
        typed = self.text()
        try:
            t = self.parse(typed)
        except ValueError as exc:
            msg = str(exc)
            self.set_seconds(self._value, force=True)  # restore the old text ...
            self._flag(msg)  # ... but keep the reason visible
            self.invalid.emit(msg)
            return
        if self._value is not None and abs(t - self._value) < 1e-9:
            self.set_seconds(self._value, force=True)
            return
        self.setModified(False)  # the model refresh that follows a successful commit may now update the field
        self._flag("")
        self.committed.emit(t)
        if self.text() == typed and typed != self._shown:  # nobody accepted the value (the engine refused it): show the real value again
            self.set_seconds(self._value, force=True)


class NumberEdit(QDoubleSpinBox):
    """Numeric field for an engine property.

    ``lo`` / ``hi`` are the engine's documented limits in MODEL units (``KEYFRAME_RANGES``, ``SPEED_RANGE`` ...); ``factor`` converts model -> display units (100 shows a 0..1 opacity as
    0..100 %). The spin box itself accepts any typed number, so a value outside the limits reaches the engine, which answers with its own message; only the arrow keys and the wheel stop
    at the limits. Keyboard tracking is off (one commit per edit), the wheel is ignored unless the field has focus, and ``set_model(None)`` shows a mixed value as an em dash."""

    committed = Signal(float)  # model units
    hint = Signal(str)

    def __init__(self, lo: float, hi: float, *, step: float = 0.1, decimals: int = 3, suffix: str = "", factor: float = 1.0, label: str = "Value", parent=None) -> None:
        super().__init__(parent)
        self._factor = float(factor)
        self._lo, self._hi = float(lo), float(hi)
        self._label = label
        self._model: float | None = 0.0
        self._shown_value = 0.0
        self._mixed = False
        self._base_tip = ""
        self._flagged = False
        super().setRange(-1e9, 1e9)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccelerated(False)
        if suffix:
            self.setSuffix(suffix)
        self.lineEdit().textEdited.connect(self._live_check)
        self.editingFinished.connect(self._finish)
        self._refresh_tip()

    # ------------------------------------------------------------------ limits
    def limits(self) -> tuple[float, float]:
        return self._lo, self._hi

    def set_limits(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        self._refresh_tip()

    def set_unit(self, factor: float, suffix: str, decimals: int, step: float) -> None:
        """Change the display unit (a keyframe property switches between px, %, degrees ...); the model value keeps being shown."""
        self._factor = float(factor)
        self.setSuffix(suffix)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self._refresh_tip()

    def _display_limits(self) -> tuple[float, float]:
        a, b = self._lo * self._factor, self._hi * self._factor
        return min(a, b), max(a, b)

    def stepBy(self, steps: int) -> None:
        lo, hi = self._display_limits()
        cur = self.value()
        new = min(max(cur + steps * self.singleStep(), lo), hi)
        self._mixed = False
        if abs(new - cur) > 1e-12 or self.lineEdit().text().strip() in (MIXED, ""):
            self.setValue(new)
            self.lineEdit().setText(self.prefix() + self.textFromValue(new) + self.suffix())
            self.lineEdit().setModified(True)  # a stepped value is an edit in progress until the field is left
            self.lineEdit().selectAll()

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    # ------------------------------------------------------------------ mixed values
    def textFromValue(self, value: float) -> str:
        return MIXED if self._mixed else super().textFromValue(value)

    def valueFromText(self, text: str) -> float:
        if self._mixed and text.replace(self.suffix(), "").strip() in (MIXED, "-", ""):
            return self.value()
        return super().valueFromText(text)

    def validate(self, text: str, pos: int):
        if self._mixed and text.replace(self.suffix(), "").strip() in (MIXED, ""):
            return QValidator.State.Acceptable, text, pos
        return super().validate(text, pos)

    def is_mixed(self) -> bool:
        return self._mixed

    def is_editing(self) -> bool:
        return self.lineEdit().isModified()

    # ------------------------------------------------------------------ model -> view
    def set_model(self, value: float | None, *, force: bool = False) -> None:
        """Show a model value (``None`` = mixed). Ignored while the user is typing unless ``force``."""
        if self.is_editing() and not force:
            self._model = None if value is None else float(value)  # remembered: discarding the edit shows the latest model value
            return
        self._mixed = value is None
        self._model = None if value is None else float(value)
        was_blocked = self.blockSignals(True)
        try:
            if value is not None:
                self.setValue(float(value) * self._factor)
            self.lineEdit().setText(self.prefix() + self.textFromValue(self.value()) + self.suffix())
            self.lineEdit().setModified(False)
        finally:
            self.blockSignals(was_blocked)
        self._shown_value = self.value()
        self._flag("")

    def discard_edit(self) -> None:
        self.set_model(self._model, force=True)

    def model_value(self) -> float:
        return self.value() / self._factor

    def set_base_tooltip(self, text: str) -> None:
        self._base_tip = text
        self._refresh_tip()

    def _refresh_tip(self) -> None:
        if self._flagged:
            return
        lo, hi = self._display_limits()
        rng = f"{self._label}: {lo:g} to {hi:g}{self.suffix().rstrip() and ' ' + self.suffix().strip()}"
        self.setToolTip((self._base_tip + "\n" if self._base_tip else "") + rng)

    # ------------------------------------------------------------------ validation hint
    def _flag(self, message: str) -> None:
        if bool(message) != self._flagged:
            self.setStyleSheet(_BAD_STYLE if message else "")
        self._flagged = bool(message)
        if message:
            self.setToolTip(message)
            self.hint.emit(message)
        else:
            self._refresh_tip()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.is_editing():
            self.discard_edit()  # Escape throws an uncommitted edit away
            event.accept()
            return
        super().keyPressEvent(event)

    def _live_check(self, _text: str) -> None:
        self._mixed = self._mixed and self.lineEdit().text().strip() in (MIXED, "")
        try:
            v = self.valueFromText(self.lineEdit().text())
        except Exception:
            self._flag(f"{self._label} must be a number.")
            return
        lo, hi = self._display_limits()
        if v < lo - 1e-9 or v > hi + 1e-9:
            self._flag(f"{self._label} must be between {lo:g} and {hi:g}.")
        else:
            self._flag("")

    def _finish(self) -> None:
        if self._model is None:  # the model values are mixed: only a number the user typed (or stepped to) counts, the dash alone does not
            if self._mixed:
                return  # entered and left without typing a number: the dash stays
        elif abs(self.value() - self._shown_value) < 1e-12:
            self.lineEdit().setModified(False)
            self._flag("")
            return  # nothing changed (or the number that is already shown was typed)
        value = self.model_value()
        if not math.isfinite(value):
            self.discard_edit()
            return
        self.lineEdit().setModified(False)
        self._flag("")
        self.committed.emit(value)
        if self._model is None and not self._mixed:
            self.set_model(None, force=True)  # still mixed after the commit (the engine refused it): show the dash again
        elif self._model is not None and abs(self.value() - self._shown_value) > 1e-12:
            self.set_model(self._model, force=True)  # nobody accepted it (the engine refused the value): show the real value again
