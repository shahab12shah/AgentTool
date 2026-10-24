"""Speed dialogs: the choice for clips that have linked mates, and the stand-alone "Speed..." dialog.

The engine decides everything (``ws.edit.set_speed``): when a clip has linked clips whose speed would differ it raises ``SpeedChoiceRequired`` with the options; this module turns that
into a question and retries with the chosen ``link_policy``. Nothing here changes a clip itself.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.exceptions import AppError
from app.timeline.ops_props import LINK_POLICIES, SpeedChoiceRequired
from app.timeline.plan import SPEED_RANGE
from app.ui.context import UiContext
from app.ui.timecode_edit import NumberEdit

OPTION_TEXT = {
    "keep_linked_audio_in_sync": ("Keep linked clips in sync", "The linked clips get the same speed, so picture and sound stay together."),
    "independent": ("Change only this clip", "The linked clips keep their speed; the sync with them may break."),
}
PRESETS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0)


def option_text(option: str) -> tuple[str, str]:
    return OPTION_TEXT.get(option, (option.replace("_", " ").capitalize(), ""))


class SpeedChoiceDialog(QDialog):
    """Lists the engine's options for a linked clip; ``choice`` holds the picked ``link_policy`` after the dialog was accepted (``None`` = cancelled)."""

    def __init__(self, message: str, options: Sequence[str] = LINK_POLICIES, members: Sequence[str] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Change speed of linked clips")
        self.choice: str | None = None
        self.buttons: dict[str, QPushButton] = {}
        lay = QVBoxLayout(self)
        text = QLabel(message)
        text.setWordWrap(True)
        lay.addWidget(text)
        if members:
            lay.addWidget(QLabel("Linked clips with a different speed: " + ", ".join(members[:6]) + (f" and {len(members) - 6} more" if len(members) > 6 else "")))
        for opt in options:
            title, detail = option_text(opt)
            b = QPushButton(title + (f"\n{detail}" if detail else ""))
            b.setObjectName(f"speed_option_{opt}")
            b.setToolTip(detail)
            b.clicked.connect(lambda _=False, o=opt: self._pick(o))
            self.buttons[opt] = b
            lay.addWidget(b)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        lay.addWidget(cancel)

    def _pick(self, option: str) -> None:
        self.choice = option
        self.accept()


def choose_link_policy(parent: QWidget | None, exc: SpeedChoiceRequired, labels: Sequence[str] = ()) -> str | None:
    """Ask which link policy to use (``None`` = the user cancelled). Looked up through the module at call time, so tests can replace it."""
    dlg = SpeedChoiceDialog(exc.user_message, exc.options or LINK_POLICIES, labels, parent)
    return dlg.choice if dlg.exec() == QDialog.DialogCode.Accepted else None


def apply_speed(ctx: UiContext, parent: QWidget | None, clip_ids: Sequence[str], speed: float) -> tuple[bool, str]:
    """Change the speed of one clip (``ws.edit.set_speed``) or of several compatible clips (ONE undo step). Returns (done, message): ``message`` is the engine's reason when
    refused, or "" when it worked or the user cancelled the linked-clip question."""
    from app.timeline import (
        ops_props as op,
    )

    ws = ctx.ws
    ids = list(dict.fromkeys(clip_ids))

    def labels(exc: SpeedChoiceRequired) -> list[str]:
        tl = ws.project.timeline if ws.project else None
        out = []
        for cid in exc.member_ids:
            hit = tl.find_clip(cid) if tl and tl.get_clip(cid) else None
            out.append(ws.project.assets.get(hit[1].asset_id).name if hit and ws.project.assets.get(hit[1].asset_id) else cid)
        return out

    def run(policy: str | None) -> None:
        if len(ids) == 1:
            ws.edit.set_speed(ids[0], speed, link_policy=policy)
            return
        env, linked = ws.edit.env(), ws.edit.state.linked_selection
        merged = None
        done: set[str] = set()
        for cid in ids:
            if cid in done:
                continue
            plan = op.plan_set_speed(env, cid, speed, linked=linked, link_policy=policy)
            plan.updates = {k: v for k, v in plan.updates.items() if k not in done}
            done |= set(plan.updates)
            if not plan.empty:
                merged = plan if merged is None else merged.merge(plan)
        if merged is None:
            raise_empty()
        merged.description = f"Change speed to {speed:g}×"
        ws.edit.run(merged, own=ids)

    def raise_empty() -> None:
        from app.core.exceptions import TimelineError

        raise TimelineError("Nothing changed: the clips already play at that speed.")

    result = {"ok": False, "msg": "", "cancelled": False}

    def attempt() -> None:
        try:
            run(None)
        except SpeedChoiceRequired as exc:
            choice = choose_link_policy(parent, exc, labels(exc))  # (looked up in the module at call time, so tests can replace it)
            if choice is None:
                result["cancelled"] = True
                return
            run(choice)
        result["ok"] = True

    def wrapped() -> None:
        try:
            attempt()
        except AppError as exc:
            result["msg"] = exc.user_message
            raise

    ok = ctx.guard(parent, wrapped)
    return (ok and result["ok"]), result["msg"]


class SpeedDialog(QDialog):
    """The "Speed..." dialog of a clip: value, presets and the duration the clip would get. OK applies the change through ``apply_speed``."""

    def __init__(self, ctx: UiContext, clip_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx, self.clip_id = ctx, clip_id
        self.setWindowTitle("Clip speed")
        clip = ctx.ws.project.timeline.get_clip(clip_id) if ctx.ws.project else None
        self._clip = clip
        lay = QVBoxLayout(self)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row = QHBoxLayout()
        row.addWidget(QLabel("Speed"))
        self.speed = NumberEdit(*SPEED_RANGE, step=0.05, decimals=2, suffix="×", label="Speed")
        self.speed.set_model(clip.speed if clip else 1.0)
        self.speed.valueChanged.connect(lambda _v: self._update())
        row.addWidget(self.speed, 1)
        lay.addLayout(row)
        presets = QHBoxLayout()
        self.preset_buttons: dict[float, QPushButton] = {}
        for p in PRESETS:
            b = QPushButton(f"{p:g}×")
            b.setToolTip(f"Set the speed to {p:g}×")
            b.clicked.connect(lambda _=False, v=p: self._preset(v))
            self.preset_buttons[p] = b
            presets.addWidget(b)
        lay.addLayout(presets)
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        lay.addWidget(self.result_label)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ff6b6b;")
        self.error.hide()
        lay.addWidget(self.error)
        buttons = QHBoxLayout()
        self.ok = QPushButton("Apply")
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.apply)
        cancel = QPushButton("Close")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.ok)
        buttons.addWidget(cancel)
        lay.addLayout(buttons)
        if clip is None:
            self.ok.setEnabled(False)
            self.info.setText("This clip no longer exists.")
        self._update()

    def _preset(self, value: float) -> None:
        self.speed.set_model(value, force=True)
        self._update()

    def current_duration(self) -> float | None:
        c = self._clip
        if c is None:
            return None
        span = c.source_out - c.source_in
        v = self.speed.model_value()
        return span / v if v > 0 and span > 0 else None

    def _update(self) -> None:
        c = self._clip
        if c is None:
            return
        d = self.current_duration()
        self.info.setText(f"Current speed {c.speed:g}×, duration {c.duration:.3f} s (source {c.source_out - c.source_in:.3f} s).")
        self.result_label.setText("Resulting duration: " + (f"{d:.3f} s" if d is not None else "—") + "  (the engine rounds and checks the room before the next clip)")

    def apply(self) -> bool:
        ok, msg = apply_speed(self.ctx, self, [self.clip_id], self.speed.model_value())
        self.error.setVisible(bool(msg))
        self.error.setText(msg)
        if ok:
            self.accept()
        return ok

