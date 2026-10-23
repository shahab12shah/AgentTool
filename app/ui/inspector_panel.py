"""Inspector for the selected timeline clip(s): type-aware sections that edit ONLY properties the renderer really reads.

The rule of the Phase 10 UI: the inspector never edits a clip itself. Every field commits ONE call on ``ctx.ws.edit`` (``set_start`` / ``set_duration`` / ``set_source_range`` /
``update_clip(s)`` / ``set_speed`` / ``set_transition`` / ``set_locks`` / ``link`` ...), wrapped in ``ctx.guard``; when the engine refuses (locked clip, number out of range, not enough source
media ...) its plain-language message is shown and the field goes back to the real value. Caption and graphic text goes through ``ws.presentation`` so the specialised models and the
manual-override records stay in charge, AI decisions through ``ws.editing``.

Sections show only what applies to the selection: video, image, audio, caption, text and graphic clips each get their own set; with several clips selected the COMMON editable properties
are shown (a mixed value reads as an em dash) and one edit changes all of them in a single undo step. A refresh from the model never touches a field the user is typing in.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.commands import CompositeCommand
from app.core.exceptions import AppError, TimelineError
from app.editing.models import Creator, TextStyle
from app.media.asset import AssetType
from app.presentation.models import HighlightMode
from app.timeline.clip import LOCK_KINDS, Clip
from app.timeline.ops_props import (
    CAPTION_POSITIONS,
    FIT_MODES,
    OPACITY_RANGE,
    PAN_RANGE,
    POSITION_LIMIT,
    ROTATION_RANGE,
    SCALE_RANGE,
    TEXT_ALIGNMENTS,
    TEXT_BACKGROUNDS,
    TEXT_SIZE_RANGE,
    TRANSITION_TYPES,
    VOLUME_RANGE,
    clip_category,
    dissolve_source,
    keyframe_properties,
)
from app.timeline.plan import SPEED_RANGE
from app.timeline.timebase import Timebase, timebase_of
from app.timeline.track import Track
from app.ui.context import UiContext
from app.ui.dialogs import replace_media_dialog, speed_dialog
from app.ui.keyframe_editor import KeyframeEditor
from app.ui.timecode_edit import NumberEdit, TimecodeEdit

SPEED_PRESETS = speed_dialog.PRESETS
DIAMOND = "◆"
TRANSITION_NAMES = {"CUT": "None (cut)", "FADE": "Fade", "DISSOLVE": "Dissolve", "WIPE": "Wipe", "SLIDE": "Slide"}
TRANSITION_FACTS = {"FADE": "Fades the clip in.", "DISSOLVE": "Needs a picture that ends exactly where this clip starts: its last frame is faded out while this clip fades in.",
                    "WIPE": "Always reveals the clip from left to right.", "SLIDE": "Always enters from the right edge.", "CUT": "A plain cut: no transition."}
LOCK_TEXT = {"item": "Lock item", "position": "Position", "timing": "Timing", "transform": "Transform", "audio": "Audio", "style": "Style"}
LOCK_TIP = {"item": "Locks the whole clip: no manual edit is accepted and AI regeneration leaves it alone.",
            "position": "Stops moving the clip.", "timing": "Stops trimming, duration, source and speed changes.", "transform": "Stops position, scale, rotation, opacity and transform keyframes.",
            "audio": "Stops volume, fades, pan, mute and volume keyframes.", "style": "Stops text, caption, crop, fit and transition changes."}
NOT_RESETTABLE = frozenset({"operation", "asset_id", "track_id", "asset_duration", "source_out", "speed", "reason", "slot", "kind_hint"})  # derived or bookkeeping parameters
KIND_LABEL = {"video": "video", "image": "image", "audio": "audio", "caption": "caption", "text": "text", "graphic": "graphic"}


def _spin(low: float, high: float, step: float, decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


def _common(values: list[float], tol: float = 1e-9) -> float | None:
    """The shared value, or None when the values differ (a mixed selection)."""
    first = values[0]
    return first if all(abs(v - first) <= tol for v in values[1:]) else None


def _set_combo(combo: QComboBox, data: Any) -> None:
    combo.blockSignals(True)
    i = combo.findData(data)
    combo.setCurrentIndex(i if i >= 0 else -1)
    combo.blockSignals(False)


def _set_check(box: QCheckBox, on: bool) -> None:
    box.blockSignals(True)
    box.setChecked(on)
    box.blockSignals(False)


class InspectorPanel(QWidget):
    search_alternatives_requested = Signal(str)  # scene id: the main window opens the Research page for it

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._clip_id: str | None = None  # the primary selected clip
        self._ids: list[str] = []  # every selected clip that still exists
        self._busy = False
        self._reset_sig: tuple = ()
        self._tips: dict[QWidget, str] = {}
        self._gates: dict[str, list[QWidget]] = {k: [] for k in ("position", "timing", "transform", "audio", "style")}
        self._key_labels: dict[str, QLabel] = {}
        self.title = QLabel("Inspector")
        self.title.setObjectName("muted")
        self.notice = QLabel()
        self.notice.setObjectName("inspector_notice")
        self.notice.setWordWrap(True)
        self.notice.hide()
        self.empty = QLabel("Select a clip on the timeline to inspect it.")
        self.empty.setObjectName("placeholder")
        self.empty.setWordWrap(True)
        self.multi_label = QLabel()
        self.multi_label.setObjectName("inspector_multi")
        self.multi_label.setWordWrap(True)
        self.no_common = QLabel("These clips have no common editable properties: select clips of the same kind to edit them together.")
        self.no_common.setWordWrap(True)
        self.no_common.setObjectName("muted")

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.multi_label)
        self.sections: dict[str, QGroupBox] = {}
        self._build_info(body)
        self._build_pres(body)
        self._build_ai(body)
        self._build_timing(body)
        self._build_speed(body)
        self._build_transform(body)
        self._build_audio(body)
        self._build_text(body)
        self._build_caption(body)
        self._build_transition(body)
        self._build_locks(body)
        self._build_keyframes(body)
        body.addWidget(self.no_common)
        body.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addWidget(self.notice)
        layout.addWidget(self.empty)
        layout.addWidget(self.body, 1)

        b = ctx.bridge
        b.on("selection.changed", lambda p: (self._clear_notice(), self.refresh()))
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("timeline", "assets", "editing", "presentation") else None)
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self.refresh())
        b.on("timeline.playhead", lambda p: self._on_playhead())
        self.refresh()

    # ================================================================== building
    def _group(self, key: str, title: str, parent: QVBoxLayout) -> QGroupBox:
        box = QGroupBox(title)
        box.setObjectName(f"sec_{key}")
        self.sections[key] = box
        parent.addWidget(box)
        return box

    def _tip(self, w: QWidget, text: str) -> None:
        self._tips[w] = text
        if hasattr(w, "set_base_tooltip"):
            w.set_base_tooltip(text)  # type: ignore[attr-defined]
        else:
            w.setToolTip(text)

    def _gate(self, widgets: list[QWidget], reason: str) -> None:
        """Disable ``widgets`` with ``reason`` as tooltip (empty reason: enable them and restore their own tooltip)."""
        for w in widgets:
            w.setEnabled(not reason)
            text = reason or self._tips.get(w, "")
            if hasattr(w, "set_base_tooltip"):
                w.set_base_tooltip(text)  # type: ignore[attr-defined]
            else:
                w.setToolTip(text)

    def _row(self, form: QFormLayout, text: str, w: QWidget, aspect: str | None = None, key: str | None = None, tip: str = "") -> QLabel:
        lab = QLabel(text)
        form.addRow(lab, w)
        if aspect:
            self._gates[aspect].append(w)
        if key:
            self._key_labels[key] = lab
        if tip:
            self._tip(w, tip)
        return lab

    def _num(self, lo: float, hi: float, *, step: float = 0.1, decimals: int = 3, suffix: str = "", factor: float = 1.0, label: str = "Value") -> NumberEdit:
        return NumberEdit(lo, hi, step=step, decimals=decimals, suffix=suffix, factor=factor, label=label)

    # ----- clip info
    def _build_info(self, body: QVBoxLayout) -> None:
        box = self._group("clip", "Clip", body)
        f = QFormLayout(box)
        self.name_label, self.kind_label = QLabel(), QLabel()
        self.track_label, self.clip_label = QLabel(), QLabel()
        self.asset_label, self.asset_info = QLabel(), QLabel()
        self.owner_label, self.scene_label = QLabel(), QLabel()
        self.info_note = QLabel()
        self.info_note.setObjectName("muted")
        for lbl in (self.name_label, self.asset_label, self.asset_info, self.clip_label, self.track_label, self.info_note):
            lbl.setWordWrap(True)
        f.addRow("Name", self.name_label)
        f.addRow("Kind", self.kind_label)
        f.addRow("Track", self.track_label)
        f.addRow("Clip", self.clip_label)
        f.addRow("Asset", self.asset_label)
        f.addRow("Source", self.asset_info)
        f.addRow("Owner", self.owner_label)
        f.addRow("Scene", self.scene_label)
        f.addRow(self.info_note)

    # ----- presentation object (Phase 5: caption, graphic, music, SFX)
    def _build_pres(self, body: QVBoxLayout) -> None:
        self.pres_info = QLabel()
        self.pres_info.setWordWrap(True)
        self.pres_info.setTextFormat(Qt.TextFormat.RichText)
        self.pres_text = QLineEdit()
        self.pres_volume = _spin(0, 400, 5, 0, " %")
        self.pres_apply = QPushButton("Apply")
        self.pres_lock = QCheckBox("Locked (protected from regeneration)")
        self.pres_box = self._group("presentation", "Presentation object", body)
        self.pres_box.setObjectName("pres_box")
        f = QFormLayout(self.pres_box)
        f.addRow(self.pres_info)
        f.addRow("Text", self.pres_text)
        self.pres_volume_label = QLabel("Volume")
        f.addRow(self.pres_volume_label, self.pres_volume)
        f.addRow(self.pres_apply)
        f.addRow(self.pres_lock)
        self.pres_apply.clicked.connect(self._apply_pres)
        self.pres_lock.clicked.connect(lambda on: self._commit(lambda: self.ctx.ws.presentation.lock_clip(self._clip_id, on)) if self._clip_id else None)

    # ----- AI decision
    def _build_ai(self, body: QVBoxLayout) -> None:
        self.ai_box = self._group("ai", "AI decision", body)
        v = QVBoxLayout(self.ai_box)
        self.ai_badge = QLabel("Manual override")
        self.ai_badge.setObjectName("manual_badge")
        self.ai_badge.setStyleSheet("background: #ffd166; color: #1c2230; border-radius: 4px; padding: 2px 8px; font-weight: bold;")
        self.ai_badge.hide()
        self.ai_info = QLabel()
        self.ai_info.setWordWrap(True)
        self.ai_info.setTextFormat(Qt.TextFormat.RichText)
        self.ai_factors = QLabel()
        self.ai_factors.setWordWrap(True)
        self.ai_factors.setTextFormat(Qt.TextFormat.RichText)
        self.ai_factors.setObjectName("muted")
        v.addWidget(self.ai_badge, 0, Qt.AlignmentFlag.AlignLeft)
        v.addWidget(self.ai_info)
        v.addWidget(self.ai_factors)
        row = QHBoxLayout()
        self.ai_replace = QPushButton("Replace visual…")
        self.ai_search = QPushButton("Search alternatives")
        row.addWidget(self.ai_replace)
        row.addWidget(self.ai_search)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.ai_regen = QPushButton("Regenerate this decision")
        self.ai_manual = QPushButton("Mark as manual override")
        row2.addWidget(self.ai_regen)
        row2.addWidget(self.ai_manual)
        v.addLayout(row2)
        self.ai_resets = QVBoxLayout()
        v.addLayout(self.ai_resets)
        self.ai_lock = QCheckBox("Locked (protected from AI regeneration)")
        self.ai_lock.setToolTip("Keeps this clip when the scene is regenerated. It does not stop your own edits: use “Lock item” under Locks & relations for that.")
        v.addWidget(self.ai_lock)
        self.ai_lock.clicked.connect(self._toggle_lock)
        self.ai_replace.clicked.connect(lambda: self.open_replace_dialog())
        self.ai_search.clicked.connect(self._search_alternatives)
        self.ai_regen.clicked.connect(self._regenerate)
        self.ai_manual.clicked.connect(self._mark_manual)

    # ----- timing
    def _build_timing(self, body: QVBoxLayout) -> None:
        box = self._group("timing", "Timing", body)
        f = QFormLayout(box)
        self.start, self.duration = TimecodeEdit(), TimecodeEdit()
        self.source_in, self.source_out = TimecodeEdit(), TimecodeEdit()
        self.end_label = QLabel()
        self._row(f, "Start", self.start, "position", tip="Where the clip starts on the timeline. The engine snaps it to the frame grid.")
        self._row(f, "Duration", self.duration, "timing", tip="Length on the timeline; moves the end edge.")
        self.end_row = self._row(f, "End", self.end_label)
        self.source_in_row = self._row(f, "Source In", self.source_in, "timing", tip="Where in the source media the clip starts. Changing it moves the end edge, the start stays.")
        self.source_out_row = self._row(f, "Source Out", self.source_out, "timing", tip="Where in the source media the clip ends.")
        self.start.committed.connect(lambda t: self._commit(lambda: self.ctx.ws.edit.set_start(self._clip_id, t)))
        self.duration.committed.connect(lambda t: self._commit(lambda: self.ctx.ws.edit.set_duration(self._clip_id, t)))
        self.source_in.committed.connect(lambda t: self._commit(lambda: self.ctx.ws.edit.set_source_range(self._clip_id, source_in=t)))
        self.source_out.committed.connect(lambda t: self._commit(lambda: self.ctx.ws.edit.set_source_range(self._clip_id, source_out=t)))
        for w in (self.start, self.duration, self.source_in, self.source_out):
            w.invalid.connect(lambda m: self._notify(m, True))

    # ----- speed
    def _build_speed(self, body: QVBoxLayout) -> None:
        box = self._group("speed", "Speed", body)
        f = QFormLayout(box)
        self.speed = self._num(*SPEED_RANGE, step=0.05, decimals=2, suffix="×", label="Speed")
        self.speed_presets = QToolButton()
        self.speed_presets.setText("Presets")
        self.speed_presets.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.speed_presets)
        for p in SPEED_PRESETS:
            a = menu.addAction(f"{p:g}×")
            a.triggered.connect(lambda _=False, v=p: self._apply_speed(v))
        self.speed_presets.setMenu(menu)
        self.speed_result = QLabel()
        self.speed_result.setObjectName("muted")
        self.speed_result.setWordWrap(True)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.speed, 1)
        h.addWidget(self.speed_presets)
        self.speed_row = self._row(f, "Speed", row, "timing")
        self._gates["timing"].extend([self.speed, self.speed_presets])
        self._tip(self.speed, "Playback speed of the media; the clip gets shorter or longer. Never reverse or frozen.")
        self._tip(self.speed_presets, "Common speeds.")
        f.addRow(self.speed_result)
        self.speed.committed.connect(self._apply_speed)
        self.speed.lineEdit().textEdited.connect(lambda _t: self._update_speed_result(self._typed_speed()))
        self.speed.valueChanged.connect(lambda _v: self._update_speed_result(self.speed.model_value()))

    # ----- transform
    def _build_transform(self, body: QVBoxLayout) -> None:
        box = self._group("transform", "Transform", body)
        f = QFormLayout(box)
        lim = POSITION_LIMIT
        self.pos_x = self._num(-lim, lim, step=1, decimals=1, suffix=" px", label="Position X")
        self.pos_y = self._num(-lim, lim, step=1, decimals=1, suffix=" px", label="Position Y")
        self.scale = self._num(*SCALE_RANGE, step=0.05, decimals=3, label="Scale")
        self.rotation = self._num(*ROTATION_RANGE, step=1, decimals=1, suffix="°", label="Rotation")
        self.opacity = self._num(*OPACITY_RANGE, step=5, decimals=0, suffix=" %", factor=100.0, label="Opacity")
        self._row(f, "Position X", self.pos_x, "transform", "position_x", "Offset from the centre of the frame, in pixels of the project size.")
        self._row(f, "Position Y", self.pos_y, "transform", "position_y", "Offset from the centre of the frame, in pixels of the project size.")
        self._row(f, "Scale", self.scale, "transform", "scale", "1 = the size the fit mode gives the picture.")
        self._row(f, "Rotation", self.rotation, "transform", "rotation", "Degrees, clockwise.")
        self._row(f, "Opacity", self.opacity, "transform", "opacity", "0 % is invisible, 100 % fully opaque.")
        self.crop = {k: self._num(0.0, 0.99, step=1, decimals=1, suffix=" %", factor=100.0, label=f"Crop {k}") for k in ("left", "right", "top", "bottom")}
        self.crop_rows = []
        for k, w in self.crop.items():
            self.crop_rows.append(self._row(f, f"Crop {k}", w, "style", tip=f"Cuts this share of the picture away from the {k}."))
        self.fit = QComboBox()
        for m in FIT_MODES:
            self.fit.addItem({"cover": "Cover (fill the frame)", "contain": "Contain (whole picture)"}[m], m)
        self.fit_row = self._row(f, "Fit", self.fit, "style", tip="How the picture is fitted into the frame. Stretching is not available because it would distort the picture.")
        self.pos_x.committed.connect(lambda v: self._set_position(x=v))
        self.pos_y.committed.connect(lambda v: self._set_position(y=v))
        self.scale.committed.connect(lambda v: self._update(scale=v))
        self.rotation.committed.connect(lambda v: self._update(rotation=v))
        self.opacity.committed.connect(lambda v: self._update(opacity=v))
        for w in self.crop.values():
            w.committed.connect(lambda _v: self._set_crop())
        self.fit.activated.connect(lambda _i: self._update(fit=self.fit.currentData()))

    # ----- audio
    def _build_audio(self, body: QVBoxLayout) -> None:
        box = self._group("audio", "Audio", body)
        f = QFormLayout(box)
        self.volume = self._num(*VOLUME_RANGE, step=5, decimals=0, suffix=" %", factor=100.0, label="Volume")
        self.fade_in = self._num(0.0, 3600.0, step=0.1, decimals=2, suffix=" s", label="Fade in")
        self.fade_out = self._num(0.0, 3600.0, step=0.1, decimals=2, suffix=" s", label="Fade out")
        self.pan = self._num(*PAN_RANGE, step=0.1, decimals=2, label="Pan")
        self.mute = QCheckBox("Mute")
        self._row(f, "Volume", self.volume, "audio", "volume", "The clip gain (there is no separate gain): 100 % leaves the sound unchanged.")
        self._row(f, "Fade in", self.fade_in, "audio", tip="Seconds to fade in from silence.")
        self._row(f, "Fade out", self.fade_out, "audio", tip="Seconds to fade out to silence.")
        self._row(f, "Pan", self.pan, "audio", tip="-1 = all left, 0 = centre, 1 = all right.")
        f.addRow(self.mute)
        self._gates["audio"].append(self.mute)
        self._tip(self.mute, "Sets the volume to 0 and remembers the previous volume; unmute brings it back (the mixer has no separate mute switch for a clip).")
        self.audio_note = QLabel("Volume is the clip gain; there is no separate gain control.")
        self.audio_note.setObjectName("muted")
        self.audio_note.setWordWrap(True)
        f.addRow(self.audio_note)
        self.volume.committed.connect(lambda v: self._update(volume=v))
        self.fade_in.committed.connect(lambda v: self._update(fade_in=v))
        self.fade_out.committed.connect(lambda v: self._update(fade_out=v))
        self.pan.committed.connect(lambda v: self._update(pan=v))
        self.mute.clicked.connect(lambda on: self._update(muted=bool(on)))

    # ----- text clips
    def _build_text(self, body: QVBoxLayout) -> None:
        box = self._group("text", "Text", body)
        f = QFormLayout(box)
        self.text_style = QComboBox()
        for s in TextStyle:
            self.text_style.addItem(s.value.replace("_", " ").title(), s.value)
        self.text_size = self._num(*TEXT_SIZE_RANGE, step=2, decimals=0, suffix=" px", label="Text size")
        self.text_opacity = self._num(*OPACITY_RANGE, step=5, decimals=0, suffix=" %", factor=100.0, label="Text opacity")
        self.text_align = QComboBox()
        for a in TEXT_ALIGNMENTS:
            self.text_align.addItem(a.title(), a)
        self.text_bg = QComboBox()
        for a in TEXT_BACKGROUNDS:
            self.text_bg.addItem({"none": "None", "box": "Box"}[a], a)
        self.text_x = self._num(0.0, 1.0, step=1, decimals=0, suffix=" %", factor=100.0, label="Text X")
        self.text_y = self._num(0.0, 1.0, step=1, decimals=0, suffix=" %", factor=100.0, label="Text Y")
        self._row(f, "Style", self.text_style, "style", tip="The graphic style; it decides the colour.")
        self._row(f, "Size", self.text_size, "style", tip="Text size in pixels.")
        self._row(f, "Opacity", self.text_opacity, "style")
        self._row(f, "Alignment", self.text_align, "style")
        self._row(f, "Background", self.text_bg, "style")
        self._row(f, "Position X", self.text_x, "style", tip="Position in the frame, 0 % = left edge, 100 % = right edge.")
        self._row(f, "Position Y", self.text_y, "style", tip="Position in the frame, 0 % = top, 100 % = bottom.")
        note = QLabel("The clip’s own position, scale and rotation are not rendered for text, so they are not offered. Edit the words in the Presentation object box.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        f.addRow(note)
        self.text_style.activated.connect(lambda _i: self._update(text_style=self.text_style.currentData()))
        self.text_size.committed.connect(lambda v: self._update(text_size=v))
        self.text_opacity.committed.connect(lambda v: self._update(text_opacity=v))
        self.text_align.activated.connect(lambda _i: self._update(text_alignment=self.text_align.currentData()))
        self.text_bg.activated.connect(lambda _i: self._update(text_background=self.text_bg.currentData()))
        self.text_x.committed.connect(lambda v: self._set_text_position(x=v))
        self.text_y.committed.connect(lambda v: self._set_text_position(y=v))

    # ----- captions
    def _build_caption(self, body: QVBoxLayout) -> None:
        box = self._group("caption", "Caption", body)
        f = QFormLayout(box)
        self.cap_position = QComboBox()
        self.cap_highlight = QComboBox()
        for h in HighlightMode:
            self.cap_highlight.addItem(h.value.title(), h.value)
        self.cap_words = QListWidget()
        self.cap_words.setObjectName("caption_words")
        self.cap_words.setMaximumHeight(130)
        self.cap_words.setToolTip("When each word is spoken. Word timing is read-only here; it follows the voice-over.")
        self.cap_merge = QPushButton("Merge with next")
        self.cap_split = QPushButton("Split at playhead")
        self._row(f, "Position", self.cap_position, "style", tip="Where the caption sits in the frame.")
        self._row(f, "Highlight", self.cap_highlight, "style", tip="How the spoken word is shown.")
        f.addRow("Words", self.cap_words)
        row = QHBoxLayout()
        row.addWidget(self.cap_merge)
        row.addWidget(self.cap_split)
        f.addRow(row)
        self.cap_position.activated.connect(lambda _i: self._caption_edit(position=self.cap_position.currentData()))
        self.cap_highlight.activated.connect(lambda _i: self._caption_edit(highlight_mode=self.cap_highlight.currentData()))
        self.cap_merge.clicked.connect(self._merge_next)
        self.cap_split.clicked.connect(self._split_at_playhead)

    # ----- transition
    def _build_transition(self, body: QVBoxLayout) -> None:
        box = self._group("transition", "Transition", body)
        f = QFormLayout(box)
        self.trans_type = QComboBox()
        self.trans_type.setObjectName("transition_type")
        for t in TRANSITION_TYPES:
            self.trans_type.addItem(TRANSITION_NAMES[t], t)
        self.trans_duration = self._num(0.0, 3600.0, step=0.1, decimals=2, suffix=" s", label="Transition duration")
        self.trans_remove = QPushButton("Remove")
        self.trans_hint = QLabel()
        self.trans_hint.setObjectName("muted")
        self.trans_hint.setWordWrap(True)
        self._row(f, "Type", self.trans_type, "style", tip="The transition INTO this clip. Only transitions the renderer implements are listed.")
        self._row(f, "Duration", self.trans_duration, "style", tip="How long the transition lasts (at least one frame, at most the clip length).")
        f.addRow(self.trans_remove)
        f.addRow(self.trans_hint)
        self._gates["style"].append(self.trans_remove)
        self.trans_type.activated.connect(lambda _i: self._set_transition(self.trans_type.currentData(), None))
        self.trans_duration.committed.connect(lambda v: self._set_transition(self.trans_type.currentData(), v))
        self.trans_remove.clicked.connect(lambda: self._commit(lambda: self.ctx.ws.edit.remove_transition(self._clip_id)))

    # ----- locks and relations
    def _build_locks(self, body: QVBoxLayout) -> None:
        box = self._group("locks", "Locks & relations", body)
        v = QVBoxLayout(box)
        self.lock_boxes: dict[str, QCheckBox] = {}
        for k in LOCK_KINDS:
            cb = QCheckBox(LOCK_TEXT[k])
            cb.setObjectName(f"lock_{k}")
            cb.setToolTip(LOCK_TIP[k])
            cb.clicked.connect(lambda on, kind=k: self._set_lock(kind, bool(on)))
            self.lock_boxes[k] = cb
            v.addWidget(cb)
        self.track_note = QLabel()
        self.track_note.setWordWrap(True)
        self.track_note.setObjectName("muted")
        v.addWidget(self.track_note)
        self.relation_label = QLabel()
        self.relation_label.setWordWrap(True)
        v.addWidget(self.relation_label)
        row = QHBoxLayout()
        self.link_btn, self.unlink_btn = QPushButton("Link"), QPushButton("Unlink")
        self.group_btn, self.ungroup_btn = QPushButton("Group"), QPushButton("Ungroup")
        for b in (self.link_btn, self.unlink_btn, self.group_btn, self.ungroup_btn):
            row.addWidget(b)
        v.addLayout(row)
        self.link_btn.clicked.connect(lambda: self._commit(lambda: self.ctx.ws.edit.link(self._ids)))
        self.unlink_btn.clicked.connect(lambda: self._commit(lambda: self.ctx.ws.edit.unlink(self._ids)))
        self.group_btn.clicked.connect(lambda: self._commit(lambda: self.ctx.ws.edit.group(self._ids)))
        self.ungroup_btn.clicked.connect(lambda: self._commit(lambda: self.ctx.ws.edit.ungroup(self._ids)))

    # ----- keyframes
    def _build_keyframes(self, body: QVBoxLayout) -> None:
        box = self._group("keyframes", "Keyframes", body)
        v = QVBoxLayout(box)
        self.kf = KeyframeEditor(self.ctx)
        self.kf.message.connect(self._notify)
        self.kf.lock_reason = self._keyframe_lock_reason
        v.addWidget(self.kf)

    # ================================================================== helpers
    def _project(self):
        return self.ctx.ws.project

    def _kind(self, track: Track, clip: Clip) -> str:
        cat = clip_category(track, clip)
        if cat == "visual":
            a = self._project().assets.get(clip.asset_id)
            return "image" if a is not None and a.type is AssetType.IMAGE else "video"
        return cat

    def _clips(self) -> list[Clip]:
        p = self._project()
        return [c for c in (p.timeline.get_clip(i) for i in self._ids) if c is not None] if p else []

    def _primary(self) -> Clip | None:
        p = self._project()
        return p.timeline.get_clip(self._clip_id) if p and self._clip_id else None

    def _current(self) -> Clip | None:  # compatibility with the older inspector
        return self._primary()

    def _lock_reason(self, clips: list[Clip], aspect: str | None) -> str:
        """Why these clips cannot be edited (a locked track, the item lock, or the aspect lock); "" when they can."""
        p = self._project()
        if p is None:
            return ""
        n = len(clips)
        for c in clips:
            tr = p.timeline.get_track(c.track_id)
            if tr.locked:
                return f"Track “{tr.name}” is locked. Unlock the track on the timeline to edit this."
        locked = [c for c in clips if "item" in c.locks]
        if locked:
            return (f"{n} selected clips: {len(locked)} locked (item lock)." if n > 1 else "This clip is locked (item lock).") + " Untick “Lock item” under Locks & relations to edit it."
        if aspect:
            locked = [c for c in clips if aspect in c.locks]
            if locked:
                return (f"{len(locked)} of the selected clips have their {aspect} locked." if n > 1 else f"The {aspect} of this clip is locked.") + \
                    f" Untick “{LOCK_TEXT[aspect]}” under Locks & relations to edit it."
        return ""

    def _keyframe_lock_reason(self, prop: str) -> str:
        return self._lock_reason(self._clips(), "audio" if prop == "volume" else "transform")

    def _notify(self, text: str, error: bool = True) -> None:
        self.notice.setText(text)
        self.notice.setStyleSheet("color: #ff6b6b;" if error else "")
        self.notice.setVisible(bool(text))

    def _clear_notice(self) -> None:
        self.notice.setText("")
        self.notice.hide()

    def _on_playhead(self) -> None:
        if not self._ids:
            return
        self._update_caption_buttons()
        if self.kf.available:
            self.kf.update_playhead()

    # ================================================================== model -> view
    def refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self._refresh()
        finally:
            self._busy = False

    def _refresh(self) -> None:
        ws = self.ctx.ws
        project = ws.project
        clips: list[Clip] = []
        if project is not None:
            clips = [c for c in (project.timeline.get_clip(i) for i in ws.selection.ids) if c is not None]
        self._ids = [c.id for c in clips]
        primary = ws.selected_clip_id
        self._clip_id = primary if primary in self._ids else (self._ids[-1] if self._ids else None)
        self.body.setVisible(bool(clips))
        self.empty.setVisible(not clips)
        if not clips or project is None:
            self.kf.set_clip(None)
            self._clear_notice()
            return
        tl = project.timeline
        tb = timebase_of(project)
        single = len(clips) == 1
        clip = tl.get_clip(self._clip_id) or clips[-1]
        tracks = {c.id: tl.get_track(c.track_id) for c in clips}
        kinds = {c.id: self._kind(tracks[c.id], c) for c in clips}
        kindset = set(kinds.values())
        sec = self.sections

        # ---- header / clip info
        self.multi_label.setVisible(not single)
        if not single:
            counts = {k: sum(1 for v in kinds.values() if v == k) for k in kindset}
            self.multi_label.setText(f"<b>{len(clips)} clips selected</b> · " + ", ".join(f"{n} {KIND_LABEL[k]}" for k, n in sorted(counts.items())))
        sec["clip"].setVisible(single)
        desc = ws.presentation.describe(clip.id) if single else None
        if single:
            self._fill_info(clip, tracks[clip.id], kinds[clip.id], project)

        # ---- presentation + AI boxes
        self.pres_box.setVisible(single and desc is not None)
        if single and desc is not None:
            self._show_pres(desc)
        dec = ws.editing.decision_for_clip(clip.id) if single else None
        has_ai = single and bool(clip.ai_decision_id or clip.created_by == "AI" or dec is not None or clip.scene_id or (desc is not None and desc.get("decision") is not None))
        self.ai_box.setVisible(has_ai)
        if has_ai:
            self._fill_ai(clip, tracks[clip.id], desc, dec)

        # ---- which editing sections apply to ALL selected clips
        visual = kindset <= {"video", "image"}
        media = kindset <= {"video", "audio"}
        audio = kindset == {"audio"}
        sec["timing"].setVisible(single)
        sec["speed"].setVisible(media)
        sec["transform"].setVisible(visual)
        sec["audio"].setVisible(audio)
        sec["text"].setVisible(single and kindset == {"text"})
        sec["caption"].setVisible(single and kindset == {"caption"})
        sec["transition"].setVisible(single and visual)
        sec["locks"].setVisible(True)
        props = keyframe_properties(tracks[clip.id], clip) if single else ()
        sec["keyframes"].setVisible(single and bool(props))
        self.no_common.setVisible(not single and not (media or visual or audio))

        if single:
            self._fill_timing(clip, tracks[clip.id], kinds[clip.id], tb)
        if media:
            self._fill_speed(clips, kinds)
        if visual:
            self._fill_transform(clips, single)
        if audio:
            self._fill_audio(clips)
        if single and kindset == {"text"}:
            self._fill_text(clip)
        if single and kindset == {"caption"}:
            self._fill_caption(clip, tracks[clip.id], tb)
        if single and visual:
            self._fill_transition(clip, tl)
        self._fill_locks(clips, tracks)
        self.kf.set_clip(clip.id if single else None)
        self._apply_gates(clips, tracks)
        self._mark_keyframed(clips)

    def _fill_info(self, clip: Clip, track: Track, kind: str, project) -> None:
        asset = project.assets.get(clip.asset_id)
        name = asset.name if asset else (str((clip.text or {}).get("text") or (clip.text or {}).get("content") or "")[:60] or clip.id)
        self.name_label.setText(name)
        self.kind_label.setText(KIND_LABEL[kind])
        self.track_label.setText(track.name + ("  (locked)" if track.locked else ""))
        self.clip_label.setText(clip.id)
        self.asset_label.setText(f"{asset.name} ({asset.id})" if asset else (clip.asset_id or "—"))
        info = []
        if asset:
            info.append(asset.type.value)
            if asset.width:
                info.append(f"{asset.width}×{asset.height}")
            if asset.fps:
                info.append(f"{asset.fps:.2f} fps")
        self.asset_info.setText(" • ".join(info))
        owner = {"AI": "AI", "USER": "You", "SYSTEM": "System"}.get(clip.created_by, clip.created_by)
        self.owner_label.setText(owner)
        scene = next((s for s in project.scenes if s.id == clip.scene_id), None) if clip.scene_id else None
        self.scene_label.setText((f"{scene.label} ({scene.id})" if scene else clip.scene_id) if clip.scene_id else "—")
        notes = []
        if kind == "video" and asset is not None and asset.has_audio:
            notes.append("The video’s own sound is not played by the renderer: put the sound on an audio track.")
        if kind in ("caption", "text", "graphic"):
            notes.append("Keyframes are not available here: captions, text and graphics are animated with animation presets.")
        self.info_note.setText(" ".join(notes))
        self.info_note.setVisible(bool(notes))

    def _fill_timing(self, clip: Clip, track: Track, kind: str, tb: Timebase) -> None:
        for w in (self.start, self.duration, self.source_in, self.source_out):
            w.set_timebase(tb)
        self.start.set_seconds(clip.timeline_start)
        self.duration.set_seconds(clip.duration)
        self.end_label.setText(tb.format(clip.timeline_end))
        has_source = kind in ("video", "audio")
        for w, lab in ((self.source_in, self.source_in_row), (self.source_out, self.source_out_row)):
            w.setVisible(has_source)
            lab.setVisible(has_source)
        if has_source:
            self.source_in.set_seconds(clip.source_in)
            self.source_out.set_seconds(clip.source_out)

    def _typed_speed(self) -> float:
        try:
            return self.speed.valueFromText(self.speed.lineEdit().text())
        except Exception:  # noqa: BLE001
            return self.speed.model_value()

    def _update_speed_result(self, speed: float) -> None:
        c = self._primary()
        if c is None or len(self._ids) != 1 or speed <= 0:
            self.speed_result.setText("")
            return
        span = c.source_out - c.source_in
        self.speed_result.setText(f"At {speed:g}× the clip would last {span / speed:.3f} s (now {c.duration:.3f} s); the engine checks the room before the next clip.")

    def _fill_speed(self, clips: list[Clip], kinds: dict[str, str]) -> None:
        v = _common([c.speed for c in clips])
        self.speed.set_model(v)
        if len(clips) == 1:
            self._update_speed_result(clips[0].speed)
        else:
            self.speed_result.setText("")

    def _fill_transform(self, clips: list[Clip], single: bool) -> None:
        self.pos_x.set_model(_common([c.position[0] for c in clips]))
        self.pos_y.set_model(_common([c.position[1] for c in clips]))
        self.scale.set_model(_common([c.scale for c in clips]))
        self.rotation.set_model(_common([c.rotation for c in clips]))
        self.opacity.set_model(_common([c.opacity for c in clips]))
        for w in list(self.crop.values()) + [self.fit]:
            w.setVisible(single)
        for lab in (*self.crop_rows, self.fit_row):
            lab.setVisible(single)
        if single:
            c = clips[0]
            x, y, w, h = c.effects.get("crop") or (0.0, 0.0, 1.0, 1.0)
            for k, val in (("left", x), ("top", y), ("right", 1.0 - (x + w)), ("bottom", 1.0 - (y + h))):
                self.crop[k].set_model(max(0.0, val) if abs(val) > 1e-9 else 0.0)
            _set_combo(self.fit, str(c.effects.get("fit", "cover")))

    def _fill_audio(self, clips: list[Clip]) -> None:
        self.volume.set_model(_common([float(c.audio.get("volume", 1.0)) for c in clips]))
        self.fade_in.set_model(_common([float(c.audio.get("fade_in", 0.0)) for c in clips]))
        self.fade_out.set_model(_common([float(c.audio.get("fade_out", 0.0)) for c in clips]))
        self.pan.set_model(_common([float(c.audio.get("pan", 0.0)) for c in clips]))
        dur = min(c.duration for c in clips)
        self.fade_in.set_limits(0.0, dur)
        self.fade_out.set_limits(0.0, dur)
        muted = [float(c.audio.get("volume", 1.0)) <= 0 for c in clips]
        _set_check(self.mute, all(muted))
        self.mute.setText("Mute" if all(muted) or not any(muted) else "Mute (some are muted)")

    def _fill_text(self, clip: Clip) -> None:
        t = clip.text or {}
        _set_combo(self.text_style, str(t.get("style", "")))
        self.text_size.set_model(float(t.get("size", 0) or 0))
        self.text_opacity.set_model(float(t.get("opacity", 1.0)))
        _set_combo(self.text_align, str(t.get("alignment", "center")))
        _set_combo(self.text_bg, str(t.get("background", "none")))
        pos = t.get("position") or [0.5, 0.5]
        self.text_x.set_model(float(pos[0]))
        self.text_y.set_model(float(pos[1]))

    def _fill_caption(self, clip: Clip, track: Track, tb: Timebase) -> None:
        t = clip.text or {}
        pos = str(t.get("position", "bottom"))
        positions = list(CAPTION_POSITIONS[:3]) + (["custom"] if pos == "custom" else [])
        self.cap_position.blockSignals(True)
        self.cap_position.clear()
        for p in positions:
            self.cap_position.addItem(p.title() if p != "custom" else f"Custom ({', '.join(f'{v:.2f}' for v in t.get('position_xy') or [])})", p)
        self.cap_position.setCurrentIndex(max(0, self.cap_position.findData(pos)))
        self.cap_position.blockSignals(False)
        _set_combo(self.cap_highlight, str(t.get("highlight_mode", "HIGHLIGHT")))
        words = t.get("words") or []
        sig = [(w.get("text"), w.get("start"), w.get("end")) for w in words if isinstance(w, dict)]
        if sig != getattr(self, "_word_sig", None):
            self._word_sig = sig
            self.cap_words.clear()
            for text, a, b in sig:
                self.cap_words.addItem(QListWidgetItem(f"{text}    {tb.format(a)} – {tb.format(b)}"))
        self._update_caption_buttons()

    def _next_caption(self, clip: Clip) -> Clip | None:
        p = self._project()
        track = p.timeline.get_track(clip.track_id)
        order = sorted(track.clips, key=lambda c: (c.timeline_start, c.id))
        i = next((k for k, c in enumerate(order) if c.id == clip.id), -1)
        return order[i + 1] if 0 <= i < len(order) - 1 else None

    def _update_caption_buttons(self) -> None:
        clip = self._primary()
        if clip is None or clip.kind != "caption" or not self.sections["caption"].isVisibleTo(self):
            return
        nxt = self._next_caption(clip)
        self.cap_merge.setEnabled(nxt is not None and nxt.kind == "caption")
        self.cap_merge.setToolTip("Join this caption and the one after it into one caption (they must follow each other closely)."
                                  if self.cap_merge.isEnabled() else "There is no caption after this one on the track.")
        head = self.ctx.ws.edit.playhead
        inside = clip.timeline_start + 1e-6 < head < clip.timeline_end - 1e-6
        reason = self._lock_reason([clip], "timing")
        self.cap_split.setEnabled(inside and not reason)
        self.cap_split.setToolTip(reason or ("Cut the caption between two words at the playhead." if inside else "Move the playhead inside this caption first."))

    def _fill_transition(self, clip: Clip, tl) -> None:
        tr = clip.transition or {}
        kind = str(tr.get("type", "CUT")).upper()
        if kind not in TRANSITION_TYPES:
            kind = "CUT"
        _set_combo(self.trans_type, kind)
        has = kind != "CUT"
        self._has_transition = has
        self.trans_duration.set_model(float(tr.get("duration", 0.0)) if has else 0.0)
        self.trans_duration.set_limits(timebase_of(self._project()).frame_s, max(clip.duration, timebase_of(self._project()).frame_s))
        i = self.trans_type.findData("DISSOLVE")
        item = self.trans_type.model().item(i)
        src = dissolve_source(self.ctx.ws.edit.env(), clip)
        item.setEnabled(src is not None or kind == "DISSOLVE")
        self.trans_type.setItemData(i, "" if src is not None else "Needs a picture that ends exactly where this clip starts (its last frame is faded out). There is none here.",
                                    Qt.ItemDataRole.ToolTipRole)
        self.trans_hint.setText(TRANSITION_FACTS.get(kind, ""))

    def _fill_locks(self, clips: list[Clip], tracks: dict[str, Track]) -> None:
        for k, cb in self.lock_boxes.items():
            have = [k in c.locks for c in clips]
            _set_check(cb, all(have))
            cb.setText(LOCK_TEXT[k] + (" (some)" if any(have) and not all(have) else ""))
        locked_tracks = sorted({tracks[c.id].name for c in clips if tracks[c.id].locked})
        self.track_note.setText(f"Track {', '.join('“' + n + '”' for n in locked_tracks)} {'is' if len(locked_tracks) == 1 else 'are'} locked: its clips cannot be edited until the track is unlocked on the timeline."
                                if locked_tracks else "")
        self.track_note.setVisible(bool(locked_tracks))
        p = self._project()
        parts = []
        linked = [c for c in clips if c.link_id]
        grouped = [c for c in clips if c.group_id]
        if linked:
            ids = {c.link_id for c in linked}
            mates = [c for c in p.timeline.all_clips() if c.link_id in ids and c.id not in {x.id for x in clips}] if len(clips) <= 20 else []
            parts.append(f"Linked: {len(linked)} of {len(clips)} selected clip{'s' if len(clips) != 1 else ''}" + (f" (+{len(mates)} more linked clip{'s' if len(mates) != 1 else ''} not selected)" if mates else ""))
        if grouped:
            parts.append(f"Grouped: {len(grouped)} of {len(clips)}")
        self.relation_label.setText(" · ".join(parts) if parts else "Not linked or grouped.")
        n = len(clips)
        self.link_btn.setEnabled(n >= 2)
        self.link_btn.setToolTip("Link the selected clips: they then move, trim, split and delete together." if n >= 2 else "Select at least two clips to link.")
        self.group_btn.setEnabled(n >= 2)
        self.group_btn.setToolTip("Group the selected clips: they are selected and moved together." if n >= 2 else "Select at least two clips to group.")
        self.unlink_btn.setEnabled(bool(linked))
        self.unlink_btn.setToolTip("Remove the link from the selected clips." if linked else "None of the selected clips is linked.")
        self.ungroup_btn.setEnabled(bool(grouped))
        self.ungroup_btn.setToolTip("Remove the selected clips from their group." if grouped else "None of the selected clips is grouped.")

    def _apply_gates(self, clips: list[Clip], tracks: dict[str, Track]) -> None:
        for aspect, widgets in self._gates.items():
            self._gate(widgets, self._lock_reason(clips, aspect))
        # locks themselves: only a locked TRACK blocks changing them
        track_reason = self._lock_reason(clips, None) if any(tracks[c.id].locked for c in clips) else ""
        self._gate(list(self.lock_boxes.values()), track_reason)
        for b in (self.link_btn, self.unlink_btn, self.group_btn, self.ungroup_btn):
            if track_reason:
                b.setEnabled(False)
                b.setToolTip(track_reason)
        # widgets that are only meaningful in a given state
        if self.sections["transition"].isVisibleTo(self):
            reason = self._lock_reason(clips, "style")
            has = getattr(self, "_has_transition", False)
            self.trans_duration.setEnabled(has and not reason)
            self.trans_remove.setEnabled(has and not reason)
            if not reason and not has:
                self.trans_duration.set_base_tooltip("Choose a transition type first.")
                self.trans_remove.setToolTip("This clip has no transition.")
            elif not reason:
                self.trans_remove.setToolTip("Remove the transition into this clip.")
        if self.sections["caption"].isVisibleTo(self):
            self._update_caption_buttons()
            reason = self._lock_reason(clips, "style")
            nxt_ok = self.cap_merge.isEnabled()
            if reason:
                self.cap_merge.setEnabled(False)
                self.cap_merge.setToolTip(reason)
            else:
                self.cap_merge.setEnabled(nxt_ok)
        self._gate_presentation(clips)

    def _gate_presentation(self, clips: list[Clip]) -> None:
        reason = self._lock_reason(clips, "style") if len(clips) == 1 else ""
        c = clips[0] if len(clips) == 1 else None
        if c is not None and c.audio.get("role") in ("MUSIC", "SFX"):
            reason = self._lock_reason(clips, "audio")
            self._gate([self.pres_volume, self.pres_apply], reason)
        elif c is not None:
            self._gate([self.pres_text, self.pres_apply], reason)

    def _mark_keyframed(self, clips: list[Clip]) -> None:
        names = {"position_x": "Position X", "position_y": "Position Y", "scale": "Scale", "rotation": "Rotation", "opacity": "Opacity", "volume": "Volume"}
        for key, lab in self._key_labels.items():
            n = sum(1 for c in clips for k in c.keyframes if k.property == key)
            lab.setText(names[key] + (f" {DIAMOND}" if n else ""))
            lab.setToolTip(f"{n} {names[key].lower()} keyframe{'s' if n != 1 else ''}: the value animates. Edit them under Keyframes." if n else "")

    # ----- presentation box (Phase 5)
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
        self.pres_volume_label.setVisible(kind in ("MUSIC", "SFX"))
        self.pres_volume_label.setText("Volume (all pieces)" if kind == "MUSIC" and d.get("assignment_id") else "Volume")
        if texty:
            if not self.pres_text.isModified():
                self.pres_text.setText(d["text"] if kind == "CAPTION" else str((d["text"] or {}).get("content", "")))
        elif not self.pres_volume.lineEdit().isModified():
            self.pres_volume.blockSignals(True)
            self.pres_volume.setValue(float(d.get("volume", 1.0)) * 100)
            self.pres_volume.blockSignals(False)
        _set_check(self.pres_lock, d["locked"])
        self.pres_apply.setVisible(kind != "EVIDENCE")

    def _apply_pres(self) -> None:
        clip = self._primary()
        d = self.ctx.ws.presentation.describe(clip.id) if clip else None
        if d is None:
            return
        svc = self.ctx.ws.presentation
        self.pres_text.setModified(False)
        self.pres_volume.lineEdit().setModified(False)
        if d["kind"] == "CAPTION":
            self._commit(lambda: svc.update_caption(clip.id, text=self.pres_text.text()))
        elif d["kind"] == "GRAPHIC":
            self._commit(lambda: svc.update_graphic(clip.id, content=self.pres_text.text()))
        elif d["kind"] == "MUSIC" and d["assignment_id"]:
            self._commit(lambda: svc.set_music(d["assignment_id"], volume=self.pres_volume.value() / 100))
        elif d["kind"] in ("MUSIC", "SFX"):
            self._commit(lambda: svc.set_clip_audio(clip.id, volume=self.pres_volume.value() / 100))

    # ----- AI decision box
    def _ai_owned(self, clip: Clip) -> bool:
        ws = self.ctx.ws
        if clip.created_by == "AI":
            return True
        return any(d.created_by is Creator.AI for d in ws.editing.decisions_for_clip(clip.id)) or \
            any(d.created_by is Creator.AI for d in self._presentation_decisions(clip))

    def _presentation_decisions(self, clip: Clip) -> list:
        p = self._project()
        return [d for d in p.presentation_decisions.values() if d.target_id == clip.id]

    def is_manual_override(self, clip: Clip) -> bool:
        """User-owned (``created_by == "USER"`` with a scene) or protected: AI regeneration keeps the clip."""
        return bool(clip.scene_id or clip.ai_decision_id) and (clip.created_by == "USER" or clip.protected)

    def _fill_ai(self, clip: Clip, track: Track, desc: dict | None, dec) -> None:
        manual = self.is_manual_override(clip)
        self.ai_badge.setVisible(manual)
        if manual:
            why = []
            if clip.created_by == "USER":
                why.append("you created or edited it")
            if clip.locked:
                why.append("it is locked")
            if clip.locks:
                why.append("it has an edit lock")
            self.ai_badge.setText("Manual override")
            self.ai_badge.setToolTip("AI regeneration keeps this clip because " + " and ".join(why) + ".")
        pd = desc.get("decision") if desc else None
        if desc is None:
            self.ai_info.setText(
                (f"<b>{dec.type.value.replace('_', ' ').title()}</b> — {dec.confidence:.0f}% · created by {dec.created_by.value}<br><i>“{dec.reason}”</i>" if dec else
                 f"Created by {clip.created_by}") + "<br>Open the AI Edit page to change its parameters.")
        else:
            self.ai_info.setText(f"Presentation decision {pd.decision_id} ({pd.type.value.replace('_', ' ').lower()})." if pd else "This object has no recorded AI decision.")
        self.ai_factors.setText(self._factors_html(clip, dec, pd))
        # actions
        is_visual = clip.kind == "media" and not track.is_audio
        self.ai_replace.setVisible(is_visual)
        self.ai_replace.setEnabled(is_visual)
        self.ai_replace.setToolTip("Choose other media for this clip: it keeps its place and timing and becomes yours.")
        self.ai_search.setEnabled(bool(clip.scene_id))
        self.ai_search.setToolTip("Open the Research page for this scene to find other visuals." if clip.scene_id else "This clip does not belong to a scene.")
        self.ai_regen.setEnabled(bool(clip.scene_id))
        self.ai_regen.setToolTip("Ask the AI for a new proposal for this scene. Parts you edited or locked are kept." if clip.scene_id else "This clip does not belong to a scene.")
        ai_owned = self._ai_owned(clip)
        self.ai_manual.setEnabled(ai_owned)
        self.ai_manual.setToolTip("Take this clip over: AI regeneration will no longer replace it." if ai_owned else "Already a manual override: AI regeneration keeps this clip.")
        self.ai_lock.setVisible(desc is None)
        _set_check(self.ai_lock, clip.locked)
        self._fill_resets(clip, desc)

    def _factors_html(self, clip: Clip, dec, pd) -> str:
        ws = self.ctx.ws
        p = ws.project
        rows: list[str] = []
        scene = next((s for s in p.scenes if s.id == clip.scene_id), None) if clip.scene_id else None
        if scene is not None:
            rows.append(f"Scene {scene.label} ({scene.start:.1f}–{scene.end:.1f} s)")
        d = dec or pd
        if d is not None and d.slot:
            rows.append(f"Role: {d.slot}")
        if dec is not None:
            par = dec.parameters
            bits = []
            for key in ("operation", "kind", "type", "style"):
                if key in par and isinstance(par[key], (str, int, float)):
                    bits.append(f"{key.replace('_', ' ')} {par[key]}")
            if "source_in" in par and "source_out" in par:
                bits.append(f"source {float(par['source_in']):.2f}–{float(par['source_out']):.2f} s")
            if bits:
                rows.append("Decision: " + ", ".join(bits))
            others = [x for x in ws.editing.decisions_for_clip(clip.id) if x.decision_id != dec.decision_id]
            if others:
                rows.append("Also: " + ", ".join(f"{x.type.value.replace('_', ' ').lower()} ({x.created_by.value}, {x.confidence:.0f}%)" for x in others[:5]))
        if clip.scene_id and clip.kind == "media":
            va = p.visual_assignments.get(clip.scene_id)
            if va is not None:
                score = f"accuracy {va.accuracy_score:.0f}%" if va.accuracy_score is not None else "accuracy not scored"
                rows.append(f"Visual: {score}, chosen by {va.selected_by}")
                cs = p.candidate_scores.get(va.candidate_id) if va.candidate_id else None
                if cs is not None:
                    rows.append(f"Match score {cs.overall:.0f} ({cs.category.value.lower() if hasattr(cs.category, 'value') else cs.category})" + (f": {cs.reason}" if cs.reason else ""))
                    rows += [str(f) for f in cs.factors[:4]]
            cands = [c for c in p.visual_candidates.values() if c.scene_id == clip.scene_id]
            usable = [c for c in cands if c.asset_id and p.assets.get(c.asset_id) and c.asset_id != clip.asset_id]
            if cands:
                rows.append(f"Alternatives: {len(cands)} candidate{'s' if len(cands) != 1 else ''} researched, {len(usable)} already in the project.")
        if d is not None:
            o = ws.editing.override_of(d.decision_id) if dec is not None else None
            if o is not None:
                rows.append(f"Replaces the AI decision {o.decision_id} ({o.confidence:.0f}% confidence).")
        return "<br>".join(rows)

    def _reset_candidates(self, clip: Clip, desc: dict | None) -> list[dict]:
        """Properties where the user's value differs from what the AI proposed - only when the original AI decision is stored (nothing is invented)."""
        ws = self.ctx.ws
        out: list[dict] = []

        def same(a: Any, b: Any) -> bool:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                return abs(float(a) - float(b)) < 1e-6
            return a == b

        def show(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.3g}"
            if isinstance(v, (list, tuple)):
                return "(" + ", ".join(show(x) for x in v) + ")"
            return str(v)

        def scalar(v: Any) -> bool:
            return isinstance(v, (str, int, float, bool)) or (isinstance(v, (list, tuple)) and len(v) <= 4 and all(isinstance(x, (int, float)) for x in v))

        for d in ws.editing.decisions_for_clip(clip.id) + ([x for x in [ws.editing.decision_for_clip(clip.id)] if x is not None and x.target_id != clip.id]):
            if d.created_by is not Creator.USER:
                continue
            o = ws.editing.override_of(d.decision_id)
            if o is None:
                continue
            label = d.type.value.replace("_", " ").lower()
            keys: dict[str, tuple[Any, Any]] = {}
            if d.type.value in ("VISUAL_TIMING", "TEXT", "NUMBER_EMPHASIS"):
                for key in ("start", "duration"):
                    keys[key] = (d.parameters.get(key, getattr(d, key)), o.parameters.get(key, getattr(o, key)))
            for key, orig in o.parameters.items():
                if scalar(orig) and key not in keys and key not in NOT_RESETTABLE:
                    keys[key] = (d.parameters.get(key), orig)
            for key, (cur, orig) in keys.items():
                if cur is None or same(cur, orig):
                    continue
                out.append({"label": f"Reset {key.replace('_', ' ')} to AI value ({show(orig)})", "tip": f"The {label} decision: you changed {key.replace('_', ' ')} to {show(cur)}; the AI proposed {show(orig)}.",
                            "key": (d.decision_id, key, show(orig)), "run": lambda did=d.decision_id, k=key, v=orig: ws.editing.update_decision(did, {k: v})})
        pres = self._presentation_resets(clip, desc, same, show)
        return (out + pres)[:8]

    def _presentation_resets(self, clip: Clip, desc: dict | None, same, show) -> list[dict]:
        if desc is None or not isinstance(clip.text, dict):
            return []
        p = self._project()
        svc = self.ctx.ws.presentation
        out = []
        for d in self._presentation_decisions(clip):
            if d.created_by is not Creator.USER:
                continue
            ov = next((o for o in p.presentation_overrides if o.override_id == d.decision_id), None)
            if ov is None:
                continue
            o = ov.original
            if desc["kind"] == "CAPTION":
                fields = {"style_id": ("style_id", lambda v: svc.update_caption(clip.id, style_id=v)),
                          "highlight_mode": ("highlight_mode", lambda v: svc.update_caption(clip.id, highlight_mode=v))}
            elif desc["kind"] == "GRAPHIC":
                fields = {k: (k, lambda v, k=k: svc.update_graphic(clip.id, **{k: v})) for k in ("content", "style", "size", "position")}
            else:
                continue
            for key, (tkey, fn) in fields.items():
                orig = o.parameters.get(key)
                cur = clip.text.get(tkey)
                if orig is None or cur is None or same(cur, orig):
                    continue
                out.append({"label": f"Reset {key.replace('_', ' ')} to AI value ({show(orig)})", "tip": f"You changed {key.replace('_', ' ')} to {show(cur)}; the AI proposed {show(orig)}.",
                            "key": (d.decision_id, key, show(orig)), "run": lambda fn=fn, v=orig: fn(v)})
        return out

    def _fill_resets(self, clip: Clip, desc: dict | None) -> None:
        cands = self._reset_candidates(clip, desc)
        sig = tuple((c["key"], c["label"]) for c in cands)
        if sig == self._reset_sig:
            return
        self._reset_sig = sig
        while self.ai_resets.count():
            it = self.ai_resets.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        self.reset_buttons: list[QPushButton] = []
        for c in cands:
            b = QPushButton(c["label"])
            b.setToolTip(c["tip"])
            b.setObjectName("ai_reset")
            b.clicked.connect(lambda _=False, run=c["run"]: self._commit(run))
            self.ai_resets.addWidget(b)
            self.reset_buttons.append(b)

    # ================================================================== view -> model
    def _commit(self, action: Callable[[], object], *, modal: bool = False) -> bool:
        """Run one engine call through ``ctx.guard``; show the engine's message when it refuses; always re-sync the fields (a refused edit goes back to the real value)."""
        err: list[str] = []

        def wrapped() -> None:
            try:
                action()
            except AppError as exc:
                err.append(exc.user_message)
                raise

        edit = self.ctx.ws.edit
        before = edit.last_notes
        ok = self.ctx.guard(self, wrapped, modal=modal)
        if err:
            self._notify(err[0], True)
        elif ok:
            self._clear_notice()
            if edit.last_notes is not before and edit.last_notes:
                self._notify(" ".join(edit.last_notes[:2]), False)
        self.refresh()
        return ok

    def _toggle_lock(self, on: bool) -> None:
        clip = self._primary()
        if clip:
            self._commit(lambda: self.ctx.ws.editing.lock_clip(clip.id, on))

    def _update(self, **changes) -> None:
        ids = list(self._ids)
        if not ids:
            return
        if len(ids) == 1:
            self._commit(lambda: self.ctx.ws.edit.update_clip(ids[0], **changes))
        else:
            self._commit(lambda: self.ctx.ws.edit.update_clips(ids, **changes))

    def _update_each(self, make: Callable[[Clip], dict]) -> None:
        """One undo step for per-clip values (a position where only x or y was typed): one plan per clip, merged and run by the engine."""
        ids = list(self._ids)
        ws = self.ctx.ws
        if len(ids) == 1:
            clip = ws.project.timeline.get_clip(ids[0])
            self._commit(lambda: ws.edit.update_clip(ids[0], **make(clip)))
            return

        def run() -> None:
            from app.timeline import ops_props as op  # noqa: PLC0415

            env, merged = ws.edit.env(), None
            for cid in ids:
                plan = op.plan_update_clip(env, cid, **make(ws.project.timeline.get_clip(cid)))
                if not plan.empty:
                    merged = plan if merged is None else merged.merge(plan)
            if merged is None:
                raise TimelineError("Nothing changed: the clips already have these values.")
            merged.description = "Edit clip properties"
            ws.edit.run(merged, own=ids)

        self._commit(run)

    def _set_position(self, x: float | None = None, y: float | None = None) -> None:
        self._update_each(lambda c: {"position": (c.position[0] if x is None else x, c.position[1] if y is None else y)})

    def _set_text_position(self, x: float | None = None, y: float | None = None) -> None:
        clip = self._primary()
        pos = (clip.text or {}).get("position") or [0.5, 0.5] if clip else [0.5, 0.5]
        self._update(text_position=(pos[0] if x is None else x, pos[1] if y is None else y))

    def _set_crop(self) -> None:
        vals = {k: w.model_value() for k, w in self.crop.items()}
        x, y = vals["left"], vals["top"]
        self._update(crop=[x, y, 1.0 - x - vals["right"], 1.0 - y - vals["bottom"]])

    def _apply_speed(self, speed: float) -> None:
        ids = list(self._ids)
        if not ids:
            return
        ok, msg = speed_dialog.apply_speed(self.ctx, self, ids, speed)
        if msg:
            self._notify(msg, True)
        elif ok:
            self._clear_notice()
            notes = self.ctx.ws.edit.last_notes
            if notes:
                self._notify(" ".join(notes[:2]), False)
        self.refresh()

    def _set_transition(self, kind: str, duration: float | None) -> None:
        clip = self._primary()
        if clip is None:
            return
        if kind == "CUT":
            if clip.transition is not None:
                self._commit(lambda: self.ctx.ws.edit.remove_transition(clip.id))
            else:
                self.refresh()
            return
        if duration is None:
            cur = float((clip.transition or {}).get("duration", 0.0))
            duration = cur if cur > 0 else min(0.5, clip.duration)
        self._commit(lambda: self.ctx.ws.edit.set_transition(clip.id, kind, duration))

    def _set_lock(self, kind: str, on: bool) -> None:
        ids = list(self._ids)
        if ids:
            self._commit(lambda: self.ctx.ws.edit.set_locks(ids, kind, "add" if on else "remove"))

    def _caption_edit(self, **kw) -> None:
        clip = self._primary()
        if clip is not None:
            self._commit(lambda: self.ctx.ws.presentation.update_caption(clip.id, **kw))

    def _merge_next(self) -> None:
        clip = self._primary()
        nxt = self._next_caption(clip) if clip else None
        if clip is None or nxt is None:
            self._notify("There is no caption after this one on the track.", True)
            return
        self._commit(lambda: self.ctx.ws.edit.merge_captions([clip.id, nxt.id]))

    def _split_at_playhead(self) -> None:
        clip = self._primary()
        if clip is not None:
            self._commit(lambda: self.ctx.ws.edit.split(self.ctx.ws.edit.playhead, [clip.id]))

    # ----- AI actions
    def _search_alternatives(self) -> None:
        clip = self._primary()
        if clip is not None and clip.scene_id:
            self.search_alternatives_requested.emit(clip.scene_id)

    def _regenerate(self) -> None:
        clip = self._primary()
        if clip is None or not clip.scene_id:
            return
        presentation = self.ctx.ws.presentation.describe(clip.id) is not None
        answer = QMessageBox.question(self, "Regenerate this scene",
                                      "Ask the AI for a new proposal for this scene?\n\nAnything you edited or locked is kept (user-owned and locked parts are never overwritten); "
                                      "the AI-owned parts of the scene are replaced. You can undo the result.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        svc = self.ctx.ws.presentation if presentation else self.ctx.ws.editing
        if self._commit(lambda: svc.regenerate_scene(clip.scene_id)):
            self._notify("Regenerating this scene in the background: your edits and locks are kept.", False)

    def _mark_manual(self) -> None:
        clip = self._primary()
        if clip is None:
            return
        ws = self.ctx.ws

        def run() -> None:
            cmds = [c for c in (ws.editing.override_command(clip.id, "edit"), ws.presentation.override_command(clip.id, "edit")) if c is not None]
            if not cmds:
                raise TimelineError("This clip has no AI decision to take over.")
            ws.commands.execute(cmds[0] if len(cmds) == 1 else CompositeCommand("Take ownership", cmds, scope="timeline"))

        if self._commit(run):
            self._notify("This clip is now yours: AI regeneration will keep it.", False)

    # ================================================================== public API for the main window / timeline
    def focus_clip(self, clip_id: str) -> bool:
        """Select ``clip_id`` (replacing the selection), show it and bring the inspector into view. False when the clip does not exist."""
        p = self._project()
        if p is None or p.timeline.get_clip(clip_id) is None:
            return False
        self.ctx.ws.select_clip(clip_id)
        self.refresh()
        area = self.parent()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parent()
        if isinstance(area, QScrollArea):
            area.verticalScrollBar().setValue(0)
        if self.start.isEnabled() and self.sections["timing"].isVisibleTo(self):
            self.start.setFocus()
            self.start.selectAll()
        return True

    def open_speed_dialog(self, clip_id: str | None = None) -> bool:
        """Show the clip in the inspector and open the Speed dialog (video and audio clips only; otherwise say why). Returns True if the dialog was shown."""
        cid = clip_id or self._clip_id
        if not cid or not self.focus_clip(cid):
            self._notify("Select a clip first.", True)
            return False
        p = self._project()
        track, clip = p.timeline.find_clip(cid)
        kind = self._kind(track, clip)
        if kind not in ("video", "audio"):
            self._notify(f"This {KIND_LABEL[kind]} clip has no playback speed: only video and audio clips can change speed.", True)
            return False
        reason = self._lock_reason([clip], "timing")
        if reason:
            self._notify(reason, True)
            return False
        speed_dialog.SpeedDialog(self.ctx, cid, self).exec()
        self.refresh()
        return True

    def open_replace_dialog(self, clip_id: str | None = None) -> bool:
        """Show the clip in the inspector, let the user pick other media and replace it through the engine. Returns True if the media was replaced."""
        cid = clip_id or self._clip_id
        if not cid or not self.focus_clip(cid):
            self._notify("Select a clip first.", True)
            return False
        ws = self.ctx.ws
        p = ws.project
        track, clip = p.timeline.find_clip(cid)
        if clip.kind != "media":
            self._notify("Only clips that play media can be replaced.", True)
            return False
        if track.locked or clip.locks:
            self._notify(self._lock_reason([clip], None) or f"“{clip.id}” is locked ({', '.join(sorted(clip.locks))}); unlock it to replace its media.", True)
            return False
        asset_id = replace_media_dialog.choose_replacement(self, self.ctx, cid)
        if not asset_id:
            return False
        if (clip.scene_id or clip.ai_decision_id) and not track.is_audio:
            return self._commit(lambda: ws.editing.replace_clip_asset(cid, asset_id))
        return self._commit(lambda: ws.edit.replace_clip(cid, asset_id))

