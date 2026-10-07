"""Inspector for the selected timeline clip."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from app.core.constants import MIN_CLIP_DURATION
from app.timeline.clip import Clip
from app.ui.context import UiContext


def _spin(low: float, high: float, step: float, decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


class InspectorPanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._clip_id: str | None = None
        title = QLabel("Inspector")
        title.setObjectName("muted")
        self.empty = QLabel("Select a clip on the timeline to inspect it.")
        self.empty.setObjectName("placeholder")
        self.empty.setWordWrap(True)

        # Clip / asset
        self.track_label, self.clip_label = QLabel(), QLabel()
        self.asset_label, self.asset_info = QLabel(), QLabel()
        for lbl in (self.asset_label, self.asset_info, self.clip_label):
            lbl.setWordWrap(True)
        clip_box = QGroupBox("Clip")
        f1 = QFormLayout(clip_box)
        f1.addRow("Track", self.track_label)
        f1.addRow("Clip", self.clip_label)
        f1.addRow("Asset", self.asset_label)
        f1.addRow("Source", self.asset_info)

        # Timing
        self.start = _spin(0, 100000, 0.1, suffix=" s")
        self.duration = _spin(MIN_CLIP_DURATION, 100000, 0.1, suffix=" s")
        self.source_in = _spin(0, 100000, 0.1, suffix=" s")
        self.source_out = _spin(0, 100000, 0.1, suffix=" s")
        for ro in (self.source_in, self.source_out):  # changed by trimming, not typed
            ro.setReadOnly(True)
            ro.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        timing = QGroupBox("Timing")
        f2 = QFormLayout(timing)
        f2.addRow("Start", self.start)
        f2.addRow("Duration", self.duration)
        f2.addRow("Source In", self.source_in)
        f2.addRow("Source Out", self.source_out)

        # Transform
        self.pos_x = _spin(-20000, 20000, 1, 1, " px")
        self.pos_y = _spin(-20000, 20000, 1, 1, " px")
        self.scale = _spin(0.01, 20, 0.05, 2)
        self.rotation = _spin(-360, 360, 1, 1, "°")
        self.opacity = _spin(0, 1, 0.05, 2)
        self.speed = _spin(0.1, 8, 0.1, 2, "×")
        transform = QGroupBox("Transform")
        f3 = QFormLayout(transform)
        f3.addRow("Position X", self.pos_x)
        f3.addRow("Position Y", self.pos_y)
        f3.addRow("Scale", self.scale)
        f3.addRow("Rotation", self.rotation)
        f3.addRow("Opacity", self.opacity)
        f3.addRow("Speed", self.speed)

        # AI decision (Phase 4)
        self.ai_info = QLabel()
        self.ai_info.setWordWrap(True)
        self.ai_info.setTextFormat(Qt.TextFormat.RichText)
        self.ai_lock = QCheckBox("Locked (protected from AI regeneration)")
        self.ai_box = QGroupBox("AI decision")
        f4 = QVBoxLayout(self.ai_box)
        f4.addWidget(self.ai_info)
        f4.addWidget(self.ai_lock)
        self.ai_lock.clicked.connect(self._toggle_lock)

        # Presentation object (Phase 5): caption, graphic, music or sound effect
        self.pres_info = QLabel()
        self.pres_info.setWordWrap(True)
        self.pres_info.setTextFormat(Qt.TextFormat.RichText)
        self.pres_text = QLineEdit()
        self.pres_volume = _spin(0, 400, 5, 0, " %")
        self.pres_apply = QPushButton("Apply")
        self.pres_lock = QCheckBox("Locked (protected from regeneration)")
        self.pres_box = QGroupBox("Presentation object")
        f5 = QFormLayout(self.pres_box)
        f5.addRow(self.pres_info)
        f5.addRow("Text", self.pres_text)
        f5.addRow("Volume", self.pres_volume)
        f5.addRow(self.pres_apply)
        f5.addRow(self.pres_lock)
        self.pres_apply.clicked.connect(self._apply_pres)
        self.pres_lock.clicked.connect(lambda on: self._run(lambda: self.ctx.ws.presentation.lock_clip(self._clip_id, on)) if self._clip_id else None)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        for box in (clip_box, self.pres_box, self.ai_box, timing, transform):
            body.addWidget(box)
        body.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)
        layout.addWidget(self.empty)
        layout.addWidget(self.body, 1)

        self.start.editingFinished.connect(self._apply_start)
        self.duration.editingFinished.connect(self._apply_duration)
        for w in (self.pos_x, self.pos_y):
            w.editingFinished.connect(self._apply_position)
        self.scale.editingFinished.connect(lambda: self._apply("scale", self.scale.value()))
        self.rotation.editingFinished.connect(lambda: self._apply("rotation", self.rotation.value()))
        self.opacity.editingFinished.connect(lambda: self._apply("opacity", self.opacity.value()))
        self.speed.editingFinished.connect(lambda: self._apply("speed", self.speed.value()))

        b = ctx.bridge
        b.on("selection.changed", lambda p: self.refresh())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("timeline", "assets") else None)
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self.refresh())
        self.refresh()

    # ----- model -> view -----
    def _current(self) -> Clip | None:
        project = self.ctx.ws.project
        cid = self.ctx.ws.selected_clip_id
        return project.timeline.get_clip(cid) if project and cid else None

    def refresh(self) -> None:
        clip = self._current()
        project = self.ctx.ws.project
        self._clip_id = clip.id if clip else None
        self.body.setVisible(clip is not None)
        self.empty.setVisible(clip is None)
        if clip is None or project is None:
            return
        asset = project.assets.get(clip.asset_id)
        track = project.timeline.get_track(clip.track_id)
        d = project.editing_decisions.get(clip.ai_decision_id) if clip.ai_decision_id else None
        desc = self.ctx.ws.presentation.describe(clip.id)
        self.pres_box.setVisible(desc is not None)
        if desc is not None:
            self._show_pres(desc)
        self.ai_box.setVisible(bool(clip.scene_id or d) and desc is None)
        if self.ai_box.isVisible():
            self.ai_info.setText(
                (f"<b>{d.type.value.replace('_', ' ').title()}</b> — {d.confidence:.0f}% · created by {d.created_by.value}<br><i>“{d.reason}”</i>" if d else
                 f"Created by {clip.created_by}") + "<br>Open the AI Edit page to change its parameters.")
            self.ai_lock.blockSignals(True)
            self.ai_lock.setChecked(clip.locked)
            self.ai_lock.blockSignals(False)
        self.track_label.setText(track.name + ("  (locked)" if track.locked else ""))
        self.clip_label.setText(clip.id)
        self.asset_label.setText(f"{asset.name} ({asset.id})" if asset else clip.asset_id)
        info = []
        if asset:
            info.append(asset.type.value)
            if asset.width:
                info.append(f"{asset.width}×{asset.height}")
            if asset.fps:
                info.append(f"{asset.fps:.2f} fps")
        self.asset_info.setText(" • ".join(info))
        values = {
            self.start: clip.timeline_start, self.duration: clip.duration, self.source_in: clip.source_in,
            self.source_out: clip.source_out, self.pos_x: clip.position[0], self.pos_y: clip.position[1],
            self.scale: clip.scale, self.rotation: clip.rotation, self.opacity: clip.opacity, self.speed: clip.speed,
        }
        for widget, value in values.items():
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        for w in (self.start, self.duration, self.pos_x, self.pos_y, self.scale, self.rotation, self.opacity, self.speed):
            w.setEnabled(not track.locked)

    def _show_pres(self, d: dict) -> None:
        kind = d["kind"]
        rows = [f"<b>{kind.title()}</b> — created by {d['created_by']}" + (f" · {d['confidence']:.0f}%" if d["confidence"] is not None else "")]
        if kind == "CAPTION":
            rows += [f"Start {d['start']:.2f}s · End {d['end']:.2f}s", f"Font {d['font']} · Size {d['size_pct']}% · {d['weight']}",
                     f"Style {d['style_id']} · Position {d['position']} · Highlight {d['highlight_mode']}",
                     f"Animation: {', '.join(str(v.get('preset')) for v in d['animation'].values() if isinstance(v, dict)) or '—'}"]
        elif kind in ("GRAPHIC", "EVIDENCE"):
            t = d["text"] or {}
            rows += [f"Start {d['start']:.2f}s · End {d['end']:.2f}s", f"Type {t.get('variant', kind)} · Style {t.get('style', '—')}",
                     f"Animation: {', '.join(str(v.get('preset')) for v in d['animation'].values() if isinstance(v, dict)) or '—'}"]
        else:
            rows += [f"Asset {d['asset']}", f"Start {d['start']:.2f}s · End {d['end']:.2f}s", f"Fade in {d['fade_in']:.2f}s · out {d['fade_out']:.2f}s"
                     + (f" · {len(d['keyframes'])} ducking keyframes" if kind == "MUSIC" else f" · {d['category']}")]
        if d["reason"]:
            rows.append(f"<i>Reason: “{d['reason']}”</i>")
        self.pres_info.setText("<br>".join(rows))
        texty = kind in ("CAPTION", "GRAPHIC")
        self.pres_text.setVisible(texty)
        self.pres_volume.setVisible(kind in ("MUSIC", "SFX"))
        if texty:
            self.pres_text.setText(d["text"] if kind == "CAPTION" else str((d["text"] or {}).get("content", "")))
        else:
            self.pres_volume.blockSignals(True)
            self.pres_volume.setValue(float(d.get("volume", 1.0)) * 100)
            self.pres_volume.blockSignals(False)
        self.pres_lock.blockSignals(True)
        self.pres_lock.setChecked(d["locked"])
        self.pres_lock.blockSignals(False)
        self.pres_apply.setVisible(kind != "EVIDENCE")

    def _apply_pres(self) -> None:
        clip = self._current()
        d = self.ctx.ws.presentation.describe(clip.id) if clip else None
        if d is None:
            return
        svc = self.ctx.ws.presentation
        if d["kind"] == "CAPTION":
            self._run(lambda: svc.update_caption(clip.id, text=self.pres_text.text()))
        elif d["kind"] == "GRAPHIC":
            self._run(lambda: svc.update_graphic(clip.id, content=self.pres_text.text()))
        elif d["kind"] == "MUSIC" and d["assignment_id"]:
            self._run(lambda: svc.set_music(d["assignment_id"], volume=self.pres_volume.value() / 100))
        elif d["kind"] in ("MUSIC", "SFX"):
            self._run(lambda: svc.set_clip_audio(clip.id, volume=self.pres_volume.value() / 100))

    def _toggle_lock(self, on: bool) -> None:
        clip = self._current()
        if clip:
            self._run(lambda: self.ctx.ws.editing.lock_clip(clip.id, on))

    # ----- view -> model -----
    def _run(self, action) -> None:
        self.ctx.guard(self, action)
        self.refresh()  # always re-sync: failed/clamped edits revert to the real values

    def _apply_start(self) -> None:
        clip = self._current()
        if clip and abs(self.start.value() - clip.timeline_start) > 1e-6:
            self._run(lambda: self.ctx.ws.timeline.move_clip(clip.id, self.start.value()))

    def _apply_duration(self) -> None:
        clip = self._current()
        if clip and abs(self.duration.value() - clip.duration) > 1e-6:
            self._run(lambda: self.ctx.ws.timeline.trim_clip(clip.id, new_end=clip.timeline_start + self.duration.value()))

    def _apply_position(self) -> None:
        clip = self._current()
        pos = (self.pos_x.value(), self.pos_y.value())
        if clip and pos != clip.position:
            self._run(lambda: self.ctx.ws.timeline.set_clip_properties(clip.id, position=pos))

    def _apply(self, name: str, value: float) -> None:
        clip = self._current()
        if clip and abs(getattr(clip, name) - value) > 1e-9:
            self._run(lambda: self.ctx.ws.timeline.set_clip_properties(clip.id, **{name: value}))
