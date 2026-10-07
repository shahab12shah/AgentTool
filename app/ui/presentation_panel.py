"""Audio & Captions page: global caption/audio controls, generation, caption/graphic/music/SFX editing, voice analysis and mix previews."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.captions.styles import PRESETS
from app.editing.models import Creator
from app.presentation.animation import EASINGS, PRESET_NAMES
from app.presentation.graphics import EVIDENCE_TOOLS
from app.presentation.models import HighlightMode, Part, PreviewMode, SfxCategory
from app.preview.player import QtPreviewPlayer
from app.ui.context import UiContext
from app.ui.timeline_preview import TimelinePreview

SCENE_COLUMNS = ("Scene", "Captions", "Graphics", "SFX", "Status")
LOW = 70.0
GRAPHIC_TYPES = ("HEADLINE", "LOWER_THIRD", "NUMBER", "DATE", "WARNING", "LABEL", "DEFINITION", "COMPARISON", "LOCATION", "ENTITY")


def _dspin(lo: float, hi: float, step: float = 0.05, dec: int = 2, suffix: str = "") -> QDoubleSpinBox:
    b = QDoubleSpinBox()
    b.setRange(lo, hi)
    b.setSingleStep(step)
    b.setDecimals(dec)
    b.setKeyboardTracking(False)
    if suffix:
        b.setSuffix(suffix)
    return b


def _combo(items, data=False) -> QComboBox:
    c = QComboBox()
    for it in items:
        if data:
            c.addItem(it[0], it[1])
        else:
            c.addItem(it)
    return c


class PresentationPanel(QWidget):
    def __init__(self, ctx: UiContext, voice_player: QtPreviewPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.voice_player = voice_player
        self.mix_player = QtPreviewPlayer(self)
        self.scene_id: str | None = None
        self.clip_id: str | None = None
        self._loading = False
        title = QLabel("AUDIO, CAPTIONS & GRAPHICS")
        title.setObjectName("title")
        note = QLabel("Voice-over is the master clock. Captions, graphics, music and sound effects are normal timeline objects you can inspect, edit, lock and regenerate.")
        note.setObjectName("muted")
        note.setWordWrap(True)

        # ================================================================ left: global controls + run
        self.cap_enabled = QCheckBox("Captions")
        self.cap_style = _combo([(p.name, k) for k, p in PRESETS.items()], data=True)
        self.cap_position = _combo([("Bottom", "bottom"), ("Center", "center"), ("Top", "top"), ("Custom", "custom")], data=True)
        self.cap_keywords = QCheckBox("Keyword highlight")
        self.cap_numbers = QCheckBox("Number emphasis")
        self.cap_highlight = _combo([("Highlight spoken word", "HIGHLIGHT"), ("Reveal word by word", "PROGRESSIVE"), ("No word timing", "NONE")], data=True)
        self.cap_lines = _combo([("2 lines", 2), ("1 line", 1)], data=True)
        self.cap_margins = {k: _dspin(0, 40, 1, 0, " %") for k in ("left", "right", "top", "bottom")}
        self.cap_large = QCheckBox("Large caption size")
        self.cap_contrast = QCheckBox("Strong contrast")
        self.cap_speed = _dspin(0.3, 2.0, 0.1, 1)
        self.cap_reduced = QCheckBox("Reduced motion")
        cf = QFormLayout()
        cf.addRow("", self.cap_enabled)
        cf.addRow("Style", self.cap_style)
        cf.addRow("Position", self.cap_position)
        cf.addRow("", self.cap_keywords)
        cf.addRow("", self.cap_numbers)
        cf.addRow("Word timing", self.cap_highlight)
        cf.addRow("Lines", self.cap_lines)
        for k, w in self.cap_margins.items():
            cf.addRow(f"Safe margin {k}", w)
        cf.addRow("", self.cap_large)
        cf.addRow("", self.cap_contrast)
        cf.addRow("Reading speed", self.cap_speed)
        cf.addRow("", self.cap_reduced)
        cap_box = QGroupBox("Captions")
        cap_box.setLayout(cf)

        self.music_on = QCheckBox("Music")
        self.sfx_on = QCheckBox("Sound effects")
        self.ducking_on = QCheckBox("Auto ducking")
        self.voice_enh = QCheckBox("Voice enhancement")
        self.lvl = {k: _dspin(0, 150, 1, 0, " %") for k in ("music_level", "important_level", "pause_level", "intro_level", "sfx_level")}
        self.attack = _dspin(0.05, 3, 0.05, 2, " s")
        self.release = _dspin(0.05, 5, 0.05, 2, " s")
        af = QFormLayout()
        for w in (self.music_on, self.sfx_on, self.ducking_on, self.voice_enh):
            af.addRow("", w)
        for label, k in (("Normal narration", "music_level"), ("Important narration", "important_level"), ("Voice pause", "pause_level"), ("Before / after voice", "intro_level"),
                         ("Sound effects", "sfx_level")):
            af.addRow(label + " (music)" if k != "sfx_level" else label, self.lvl[k])
        af.addRow("Duck attack", self.attack)
        af.addRow("Release", self.release)
        aud_box = QGroupBox("Audio")
        aud_box.setLayout(af)

        self.gen_captions_btn = QPushButton("Generate Captions")
        self.gen_graphics_btn = QPushButton("Generate Graphics")
        self.gen_audio_btn = QPushButton("Generate Audio (SFX + ducking)")
        self.gen_all_btn = QPushButton("Generate Presentation")
        self.gen_all_btn.setObjectName("primary")
        self.regen_scene_btn = QPushButton("Regenerate Selected Scene")
        self.regen_all_btn = QPushButton("Regenerate All")
        self.retry_btn = QPushButton("Retry Failed Scenes")
        self.cancel_btn = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.progress_label = QLabel()
        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        self.stale_box = QFrame()
        self.stale_box.setObjectName("failureBox")
        sl = QVBoxLayout(self.stale_box)
        self.stale_label = QLabel()
        self.stale_label.setWordWrap(True)
        sl.addWidget(self.stale_label)
        row = QHBoxLayout()
        self.stale_regen_btn = QPushButton("Regenerate Captions")
        self.stale_keep_btn = QPushButton("Keep Existing")
        self.stale_review_btn = QPushButton("Review Changes")
        for b in (self.stale_regen_btn, self.stale_keep_btn, self.stale_review_btn):
            row.addWidget(b)
        sl.addLayout(row)
        run = QGroupBox("Generate")
        rl = QVBoxLayout(run)
        for b in (self.gen_all_btn, self.gen_captions_btn, self.gen_graphics_btn, self.gen_audio_btn, self.regen_scene_btn, self.regen_all_btn, self.retry_btn, self.cancel_btn):
            rl.addWidget(b)
        for w in (self.progress, self.progress_label, self.state_label, self.log):
            rl.addWidget(w)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(self.stale_box)
        ll.addWidget(cap_box)
        ll.addWidget(aud_box)
        ll.addWidget(run)
        ll.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(340)

        # ================================================================ middle: scenes
        self.table = QTableWidget(0, len(SCENE_COLUMNS))
        self.table.setObjectName("presScenes")
        self.table.setHorizontalHeaderLabels(SCENE_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for col, w in ((0, 50), (1, 70), (2, 70), (3, 50)):
            self.table.setColumnWidth(col, w)
        self.lock_btns = {a: QPushButton(f"Lock {a.title()}") for a in ("CAPTION", "GRAPHIC", "MUSIC", "SFX", "MIX", "SCENE")}
        for b in self.lock_btns.values():
            b.setCheckable(True)
        lk = QHBoxLayout()
        for b in self.lock_btns.values():
            lk.addWidget(b)
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.addWidget(QLabel("Scenes"))
        ml.addWidget(self.table, 1)
        ml.addLayout(lk)

        # ================================================================ right: preview + tabs
        self.preview = TimelinePreview(ctx)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.time_label = QLabel("0:00.0")
        self.play_btn = QPushButton("Play voice")
        prow = QHBoxLayout()
        prow.addWidget(self.play_btn)
        prow.addWidget(self.slider, 1)
        prow.addWidget(self.time_label)
        self.preview_note = QLabel()
        self.preview_note.setObjectName("muted")
        self.preview_note.setWordWrap(True)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._captions_tab(), "Captions")
        self.tabs.addTab(self._graphics_tab(), "Graphics")
        self.tabs.addTab(self._audio_tab(), "Music & SFX")
        self.tabs.addTab(self._voice_tab(), "Voice & mix")
        right = QWidget()
        rl2 = QVBoxLayout(right)
        rl2.addWidget(self.preview, 3)
        rl2.addLayout(prow)
        rl2.addWidget(self.preview_note)
        rl2.addWidget(self.tabs, 4)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_scroll)
        split.addWidget(mid)
        split.addWidget(right_scroll)
        split.setSizes([360, 340, 640])
        split.setChildrenCollapsible(False)
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addWidget(split, 1)

        # ---- wiring
        for w in (self.cap_enabled, self.cap_keywords, self.cap_numbers, self.cap_large, self.cap_contrast, self.cap_reduced):
            w.toggled.connect(self._caption_settings_changed)
        for w in (self.cap_style, self.cap_position, self.cap_highlight, self.cap_lines):
            w.currentIndexChanged.connect(self._caption_settings_changed)
        for w in (*self.cap_margins.values(), self.cap_speed):
            w.valueChanged.connect(self._caption_settings_changed)
        for w in (self.music_on, self.sfx_on, self.ducking_on, self.voice_enh):
            w.toggled.connect(self._audio_settings_changed)
        for w in (*self.lvl.values(), self.attack, self.release):
            w.valueChanged.connect(self._audio_settings_changed)
        self.gen_all_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.generate(), "Generate presentation"))
        self.gen_captions_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.generate([Part.CAPTIONS.value], self._selected() or None), "Generate captions"))
        self.gen_graphics_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.generate([Part.GRAPHICS.value], self._selected() or None), "Generate graphics"))
        self.gen_audio_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.generate([Part.AUDIO.value], self._selected() or None), "Generate audio"))
        self.regen_scene_btn.clicked.connect(self._regen_selected)
        self.regen_all_btn.clicked.connect(self._regen_all)
        self.retry_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.retry_failed(), "Retry"))
        self.cancel_btn.clicked.connect(lambda: self.ctx.ws.presentation.cancel())
        self.stale_regen_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.regenerate_captions(), "Regenerate captions"))
        self.stale_keep_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.acknowledge_stale(), "Keep existing captions"))
        self.stale_review_btn.clicked.connect(self._review)
        self.table.itemSelectionChanged.connect(self._on_select_scene)
        for a, b in self.lock_btns.items():
            b.clicked.connect(lambda on, a=a: self._lock(a, on))
        self.slider.valueChanged.connect(self._scrub)
        self.play_btn.clicked.connect(self._toggle_play)
        self.voice_player.position_changed.connect(self._on_position)
        self.voice_player.playing_changed.connect(lambda on: self.play_btn.setText("Pause" if on else "Play voice"))
        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self._reset())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("timeline", "editing", "scenes", "assets", "voice_over", "waveform") else None)
        b.on("job.updated", self._on_job)
        b.on("selection.changed", lambda p: self._on_clip_selected(p.get("clip_id")))
        self._reset()

    # ================================================================== tab builders
    def _captions_tab(self) -> QWidget:
        self.cap_table = QTableWidget(0, 5)
        self.cap_table.setObjectName("captionList")
        self.cap_table.setHorizontalHeaderLabels(("Start", "End", "Text", "Style", "By"))
        self._table_setup(self.cap_table, 2)
        self.cap_text = QLineEdit()
        self.cap_style_edit = _combo([(p.name, k) for k, p in PRESETS.items()], data=True)
        self.cap_pos_edit = _combo([("Bottom", "bottom"), ("Center", "center"), ("Top", "top")], data=True)
        self.cap_hl_edit = _combo([(m.value.title(), m.value) for m in HighlightMode], data=True)
        self.cap_in = _combo(PRESET_NAMES)
        self.cap_font = QLineEdit()
        self.cap_size = _dspin(2.0, 12.0, 0.2, 1, " %")
        self.cap_weight = _combo(("normal", "bold"))
        self.cap_bg = _combo(("none", "box"))
        self.cap_shadow = QCheckBox("Shadow")
        self.cap_outline = _dspin(0, 30, 1, 0, " %")
        self.cap_opacity = _dspin(0.1, 1.0, 0.05)
        self.cap_spacing = _dspin(0.8, 2.0, 0.05)
        self.cap_align = _combo(("center", "left", "right"))
        self.cap_save_style_btn = QPushButton("Save look as style…")
        self.cap_info = QLabel()
        self.cap_info.setWordWrap(True)
        self.cap_info.setTextFormat(Qt.TextFormat.RichText)
        self.cap_apply_btn = QPushButton("Apply to caption")
        self.cap_apply_btn.setObjectName("primary")
        self.cap_lock_btn = QPushButton("Lock caption")
        self.cap_lock_btn.setCheckable(True)
        self.cap_style_all_btn = QPushButton("Apply style to all AI captions")
        f = QFormLayout()
        f.addRow("Text", self.cap_text)
        f.addRow("Style", self.cap_style_edit)
        f.addRow("Position", self.cap_pos_edit)
        f.addRow("Word timing", self.cap_hl_edit)
        f.addRow("Animation in", self.cap_in)
        f.addRow("Font / size", self._pair(self.cap_font, self.cap_size))
        f.addRow("Weight / alignment", self._pair(self.cap_weight, self.cap_align))
        f.addRow("Background / opacity", self._pair(self.cap_bg, self.cap_opacity))
        f.addRow("Outline / line spacing", self._pair(self.cap_outline, self.cap_spacing))
        f.addRow("", self.cap_shadow)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self.cap_table)
        lay.addWidget(self.cap_info)
        lay.addLayout(f)
        row = QHBoxLayout()
        for b in (self.cap_apply_btn, self.cap_lock_btn, self.cap_style_all_btn, self.cap_save_style_btn):
            row.addWidget(b)
        lay.addLayout(row)
        self.cap_table.itemSelectionChanged.connect(lambda: self._pick(self.cap_table))
        self.cap_apply_btn.clicked.connect(self._apply_caption)
        self.cap_save_style_btn.clicked.connect(self._save_style)
        self.cap_lock_btn.clicked.connect(lambda on: self._lock_clip(on))
        self.cap_style_all_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.apply_caption_style(self.cap_style_edit.currentData(), self._selected() or None), "Apply style"))
        return w

    def _graphics_tab(self) -> QWidget:
        self.gfx_table = QTableWidget(0, 5)
        self.gfx_table.setObjectName("graphicList")
        self.gfx_table.setHorizontalHeaderLabels(("Start", "Duration", "Content", "Type", "By"))
        self._table_setup(self.gfx_table, 2)
        self.gfx_content = QLineEdit()
        self.gfx_subtitle = QLineEdit()
        self.gfx_x, self.gfx_y = _dspin(0, 1, 0.05), _dspin(0, 1, 0.05)
        self.gfx_size = QSpinBox()
        self.gfx_size.setRange(8, 400)
        self.gfx_start, self.gfx_dur = _dspin(0, 100000, 0.1), _dspin(0.1, 100, 0.1)
        self.gfx_in, self.gfx_out = _combo(("none", *PRESET_NAMES)), _combo(("none", *PRESET_NAMES))
        self.gfx_in_dur, self.gfx_ease = _dspin(0.05, 5, 0.05), _combo(EASINGS)
        self.gfx_tool = _combo(EVIDENCE_TOOLS)
        self.gfx_info = QLabel()
        self.gfx_info.setWordWrap(True)
        self.gfx_info.setTextFormat(Qt.TextFormat.RichText)
        self.gfx_apply_btn = QPushButton("Apply to graphic")
        self.gfx_apply_btn.setObjectName("primary")
        self.gfx_lock_btn = QPushButton("Lock graphic")
        self.gfx_lock_btn.setCheckable(True)
        self.gfx_new_type = _combo(GRAPHIC_TYPES)
        self.gfx_new_btn = QPushButton("Add graphic at playhead")
        f = QFormLayout()
        f.addRow("Text", self.gfx_content)
        f.addRow("Subtitle", self.gfx_subtitle)
        f.addRow("Position X / Y", self._pair(self.gfx_x, self.gfx_y))
        f.addRow("Size", self.gfx_size)
        f.addRow("Start / duration", self._pair(self.gfx_start, self.gfx_dur))
        f.addRow("Animation in", self.gfx_in)
        f.addRow("Animation out", self.gfx_out)
        f.addRow("Duration / easing", self._pair(self.gfx_in_dur, self.gfx_ease))
        f.addRow("Evidence tool", self.gfx_tool)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self.gfx_table)
        lay.addWidget(self.gfx_info)
        lay.addLayout(f)
        row = QHBoxLayout()
        row.addWidget(self.gfx_apply_btn)
        row.addWidget(self.gfx_lock_btn)
        row.addWidget(self.gfx_new_type)
        row.addWidget(self.gfx_new_btn)
        lay.addLayout(row)
        self.gfx_table.itemSelectionChanged.connect(lambda: self._pick(self.gfx_table))
        self.gfx_apply_btn.clicked.connect(self._apply_graphic)
        self.gfx_lock_btn.clicked.connect(lambda on: self._lock_clip(on))
        self.gfx_new_btn.clicked.connect(self._new_graphic)
        return w

    def _audio_tab(self) -> QWidget:
        self.music_lib = QListWidget()
        self.sfx_lib = QListWidget()
        self.import_music_btn = QPushButton("Import music…")
        self.import_sfx_btn = QPushButton("Import SFX…")
        self.sfx_cat = _combo([c.value for c in SfxCategory])
        self.add_music_btn = QPushButton("Add music to timeline")
        self.add_music_btn.setObjectName("primary")
        self.add_sfx_btn = QPushButton("Add SFX at playhead")
        self.audio_table = QTableWidget(0, 5)
        self.audio_table.setObjectName("audioList")
        self.audio_table.setHorizontalHeaderLabels(("Start", "End", "Asset", "Volume", "By"))
        self._table_setup(self.audio_table, 2)
        self.au_volume = _dspin(0, 400, 5, 0, " %")
        self.au_fade_in, self.au_fade_out = _dspin(0, 30, 0.1), _dspin(0, 30, 0.1)
        self.au_loop = QCheckBox("Loop")
        self.au_info = QLabel()
        self.au_info.setWordWrap(True)
        self.au_info.setTextFormat(Qt.TextFormat.RichText)
        self.au_apply_btn = QPushButton("Apply to audio clip")
        self.au_apply_btn.setObjectName("primary")
        self.au_lock_btn = QPushButton("Lock")
        self.au_lock_btn.setCheckable(True)
        self.au_delete_btn = QPushButton("Delete")
        self.duck_edit = QPlainTextEdit()
        self.duck_edit.setPlaceholderText("Ducking keyframes: one per line as  seconds, level%  (editable)")
        self.duck_edit.setMaximumHeight(110)
        self.duck_apply_btn = QPushButton("Apply ducking keyframes")
        f = QFormLayout()
        f.addRow("Volume", self.au_volume)
        f.addRow("Fade in / out", self._pair(self.au_fade_in, self.au_fade_out))
        f.addRow("", self.au_loop)
        w = QWidget()
        lay = QVBoxLayout(w)
        libs = QHBoxLayout()
        lm, ls = QVBoxLayout(), QVBoxLayout()
        lm.addWidget(QLabel("Music library"))
        lm.addWidget(self.music_lib)
        lm.addWidget(self.import_music_btn)
        lm.addWidget(self.add_music_btn)
        ls.addWidget(QLabel("SFX library"))
        ls.addWidget(self.sfx_lib)
        ls.addWidget(self.sfx_cat)
        ls.addWidget(self.import_sfx_btn)
        ls.addWidget(self.add_sfx_btn)
        libs.addLayout(lm)
        libs.addLayout(ls)
        lay.addLayout(libs)
        lay.addWidget(QLabel("Music and sound effects on the timeline"))
        lay.addWidget(self.audio_table)
        lay.addWidget(self.au_info)
        lay.addLayout(f)
        row = QHBoxLayout()
        for b in (self.au_apply_btn, self.au_lock_btn, self.au_delete_btn):
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(self.duck_edit)
        lay.addWidget(self.duck_apply_btn)
        self.audio_table.itemSelectionChanged.connect(lambda: self._pick(self.audio_table))
        self.music_lib.itemDoubleClicked.connect(self._play_library_item)
        self.sfx_lib.itemDoubleClicked.connect(self._play_library_item)
        self.import_music_btn.clicked.connect(lambda: self._import("music"))
        self.import_sfx_btn.clicked.connect(lambda: self._import("sfx"))
        self.add_music_btn.clicked.connect(self._add_music)
        self.add_sfx_btn.clicked.connect(self._add_sfx)
        self.au_apply_btn.clicked.connect(self._apply_audio)
        self.au_lock_btn.clicked.connect(lambda on: self._lock_clip(on))
        self.au_delete_btn.clicked.connect(self._delete_audio)
        self.duck_apply_btn.clicked.connect(self._apply_ducking)
        return w

    def _voice_tab(self) -> QWidget:
        self.an_label = QLabel()
        self.an_label.setWordWrap(True)
        self.an_issues = QPlainTextEdit()
        self.an_issues.setReadOnly(True)
        self.an_issues.setMaximumHeight(110)
        self.an_btn = QPushButton("Analyse voice-over")
        self.vp_gain = _dspin(-24, 24, 0.5, 1, " dB")
        self.vp_highpass = _dspin(0, 500, 10, 0, " Hz")
        self.vp_comp = QCheckBox("Compression")
        self.vp_limiter = QCheckBox("Limiter")
        self.vp_noise = QCheckBox("Noise reduction")
        self.vp_norm = QCheckBox("Normalize")
        self.vp_eq = _combo(("none", "clarity", "warm", "broadcast"))
        self.vp_apply_btn = QPushButton("Apply voice processing")
        self.vp_label = QLabel()
        self.vp_label.setObjectName("muted")
        self.vp_label.setWordWrap(True)
        self.preview_mode = _combo([(m.value.replace("_", " + ").title(), m.value) for m in PreviewMode], data=True)
        self.preview_btn = QPushButton("Preview mix")
        self.preview_btn.setObjectName("primary")
        self.preview_status = QLabel()
        self.preview_status.setObjectName("muted")
        self.preview_status.setWordWrap(True)
        self.masking_label = QLabel()
        self.masking_label.setWordWrap(True)
        f = QFormLayout()
        f.addRow("Gain", self.vp_gain)
        f.addRow("High-pass", self.vp_highpass)
        f.addRow("EQ", self.vp_eq)
        for cb in (self.vp_comp, self.vp_limiter, self.vp_noise, self.vp_norm):
            f.addRow("", cb)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Voice-over analysis"))
        lay.addWidget(self.an_label)
        lay.addWidget(self.an_issues)
        lay.addWidget(self.an_btn)
        lay.addWidget(QLabel("Voice processing (non-destructive; the original file is never changed)"))
        lay.addLayout(f)
        lay.addWidget(self.vp_label)
        lay.addWidget(self.vp_apply_btn)
        lay.addWidget(QLabel("Audio preview"))
        r = QHBoxLayout()
        r.addWidget(self.preview_mode, 1)
        r.addWidget(self.preview_btn)
        lay.addLayout(r)
        lay.addWidget(self.preview_status)
        lay.addWidget(self.masking_label)
        lay.addStretch(1)
        self.an_btn.clicked.connect(lambda: self.act(lambda: self.ctx.ws.presentation.analyze_voice(force=True), "Analyse voice-over"))
        self.vp_apply_btn.clicked.connect(self._apply_voice)
        self.preview_btn.clicked.connect(self._preview)
        return w

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    @staticmethod
    def _table_setup(t: QTableWidget, stretch: int) -> None:
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
        t.setMaximumHeight(170)

    # ================================================================== helpers
    def act(self, action, title: str = "Presentation") -> bool:
        ok = self.ctx.guard(self, action, modal=True, title=title)
        self.refresh()
        return ok

    def _selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows if self.table.item(r, 0)]

    def _reset(self) -> None:
        self.scene_id = self.clip_id = None
        self.refresh()

    def _svc(self):
        return self.ctx.ws.presentation

    # ================================================================== settings -> service
    def _load_settings(self) -> None:
        p = self.ctx.ws.project
        if p is None:
            return
        cs, a = p.caption_settings, p.audio_settings
        self._loading = True
        try:
            self.cap_enabled.setChecked(cs.enabled)
            self.cap_style.setCurrentIndex(max(0, self.cap_style.findData(cs.style_id)))
            self.cap_position.setCurrentIndex(max(0, self.cap_position.findData(cs.position)))
            self.cap_keywords.setChecked(cs.keyword_highlight)
            self.cap_numbers.setChecked(cs.number_emphasis)
            self.cap_highlight.setCurrentIndex(max(0, self.cap_highlight.findData(cs.highlight_mode)))
            self.cap_lines.setCurrentIndex(max(0, self.cap_lines.findData(cs.max_lines)))
            for k, w in self.cap_margins.items():
                w.setValue(getattr(cs, f"safe_margin_{k}") * 100)
            self.cap_large.setChecked(cs.large_text)
            self.cap_contrast.setChecked(cs.high_contrast)
            self.cap_speed.setValue(cs.reading_speed)
            self.cap_reduced.setChecked(cs.reduced_motion)
            for w, v in ((self.music_on, a.music_enabled), (self.sfx_on, a.sfx_enabled), (self.ducking_on, a.auto_ducking), (self.voice_enh, a.voice_enhancement)):
                w.setChecked(v)
            for k, w in self.lvl.items():
                w.setValue(getattr(a, k) * 100)
            self.attack.setValue(a.attack)
            self.release.setValue(a.release)
            pr = p.audio_processing
            self.vp_gain.setValue(pr.gain_db)
            self.vp_highpass.setValue(pr.highpass_hz)
            self.vp_comp.setChecked(pr.compression)
            self.vp_limiter.setChecked(pr.limiter)
            self.vp_noise.setChecked(pr.noise_reduction)
            self.vp_norm.setChecked(pr.normalize)
            self.vp_eq.setCurrentText(pr.eq_preset)
        finally:
            self._loading = False

    def _caption_settings_changed(self, *_a) -> None:
        if self._loading or self.ctx.ws.project is None:
            return
        self.ctx.guard(self, lambda: self._svc().update_caption_settings(
            enabled=self.cap_enabled.isChecked(), style_id=self.cap_style.currentData(), position=self.cap_position.currentData(), keyword_highlight=self.cap_keywords.isChecked(),
            number_emphasis=self.cap_numbers.isChecked(), highlight_mode=self.cap_highlight.currentData(), max_lines=self.cap_lines.currentData(),
            safe_margin_left=self.cap_margins["left"].value() / 100, safe_margin_right=self.cap_margins["right"].value() / 100,
            safe_margin_top=self.cap_margins["top"].value() / 100, safe_margin_bottom=self.cap_margins["bottom"].value() / 100, large_text=self.cap_large.isChecked(),
            high_contrast=self.cap_contrast.isChecked(), reading_speed=self.cap_speed.value(), reduced_motion=self.cap_reduced.isChecked()), modal=True, title="Caption settings")
        self._load_settings()

    def _audio_settings_changed(self, *_a) -> None:
        if self._loading or self.ctx.ws.project is None:
            return
        self.ctx.guard(self, lambda: self._svc().update_audio_settings(
            music_enabled=self.music_on.isChecked(), sfx_enabled=self.sfx_on.isChecked(), auto_ducking=self.ducking_on.isChecked(), voice_enhancement=self.voice_enh.isChecked(),
            attack=self.attack.value(), release=self.release.value(), **{k: w.value() / 100 for k, w in self.lvl.items()}), modal=True, title="Audio settings")
        self._load_settings()

    # ================================================================== generation
    def _regen_selected(self) -> None:
        ids = self._selected()
        if not ids:
            self.ctx.status("Select one or more scenes first.")
            return
        self.act(lambda: self._svc().generate(None, ids, "SELECTED"), "Regenerate scene")

    def _regen_all(self) -> None:
        p = self.ctx.ws.project
        owned = sum(1 for d in p.presentation_decisions.values() if d.created_by is Creator.USER or d.locked) if p else 0
        text = "Regenerate the whole presentation?\n\nAI-created captions, graphics, sound effects and ducking are replaced. " + (
            f"Your {owned} edited or locked object(s) are preserved." if owned else "You have not edited anything yet.") + "\nYou can undo this."
        if QMessageBox.question(self, "Regenerate all", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes:
            self.act(lambda: self._svc().regenerate_all(), "Regenerate all")

    def _review(self) -> None:
        r = self._svc().caption_review()
        QMessageBox.information(self, "Review caption changes",
                                f"Current captions: {r['current']}\nKept (edited or locked by you): {r['preserved_user_or_locked']}\nReplaced (AI-made): {r['removed_ai']}\n"
                                f"New AI captions after regeneration: about {r['new_ai_estimate']}\nTranscript words: {r['transcript_words']}\n\n"
                                "Nothing has been changed.")

    def _on_job(self, payload: dict) -> None:
        if payload["job"].type in ("presentation", "audio_analysis", "audio_preview", "waveform"):
            self._refresh_progress()

    def _refresh_progress(self) -> None:
        svc = self._svc()
        running = svc.running
        self.progress.setVisible(running)
        self.progress_label.setVisible(running)
        if running:
            job = next((j for j in self.ctx.ws.jobs.active_jobs() if j.type == "presentation"), None)
            self.progress.setValue(int(job.progress) if job else 0)
            pr = svc.progress
            self.progress_label.setText(f"Scene {pr['scene']} / {pr['total']} — {pr['operation']}")
        p = self.ctx.ws.project
        s = p.presentation_sessions[-1] if p and p.presentation_sessions else None
        self.log.setPlainText("\n".join(s.log[-30:]) if s else "")
        if p is None:
            msg = ""
        elif s is None:
            msg = "Nothing generated yet. Choose a style, then press Generate Presentation."
        elif running:
            msg = "Working in the background…"
        elif s.status == "COMPLETED":
            msg = f"Done: {', '.join(x.title() for x in s.parts)} for {len(s.completed)} scene(s). Scrub the preview and inspect any object."
        elif s.status == "FAILED":
            msg = f"⚠ {s.error or 'The run failed.'}" + (f"\nFailed at scene {self._label(s.failed_scene)} ({s.failed_part.title()}); earlier scenes were kept. Use Retry." if s.failed_scene else "")
            if s.validation_errors:
                msg += "\n" + "\n".join(s.validation_errors[:3])
        elif s.status == "CANCELED":
            msg = "Canceled — nothing was changed."
        else:
            msg = s.status
        self.state_label.setText(msg)

    def _label(self, scene_id: str) -> str:
        p = self.ctx.ws.project
        return next((s.label for s in p.scenes if s.id == scene_id), scene_id) if p else scene_id

    # ================================================================== refresh
    def refresh(self) -> None:
        p = self.ctx.ws.project
        svc = self._svc()
        has = bool(p and p.scenes)
        running = svc.running
        for b in (self.gen_captions_btn, self.gen_graphics_btn, self.gen_audio_btn, self.gen_all_btn, self.regen_scene_btn, self.regen_all_btn):
            b.setEnabled(has and not running)
        self.cancel_btn.setEnabled(running)
        failed = bool(p and any(v in ("FAILED", "PENDING") for part in ("CAPTIONS", "GRAPHICS") for sid, v in p.presentation_generation.scene_status.get(part, {}).items()
                                if sid in {s.id for s in p.scenes}) and p.presentation_sessions and p.presentation_sessions[-1].status == "FAILED")
        self.retry_btn.setEnabled(failed and not running)
        self._load_settings()
        self._refresh_progress()
        st = svc.staleness() if p else {"captions_outdated": False, "message": ""}
        self.stale_box.setVisible(bool(st["captions_outdated"]))
        self.stale_label.setText(st.get("message", ""))
        rows = svc.scene_rows() if has else []
        keep = set(self._selected())
        self._loading = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                status = " / ".join(f"{k[0]}:{v[0]}" for k, v in (("Captions", row["caption_status"]), ("Graphics", row["graphics_status"]), ("Audio", row["audio_status"]))) + (" 🔒" if row["locked"] else "")
                for col, text in enumerate((row["label"], str(row["captions"]), str(row["graphics"]), str(row["sfx"]), status)):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, row["scene_id"])
                    if "F" in status.replace("Failed", "") and col == 4 and "F" in (row["caption_status"][0], row["graphics_status"][0]) and "FAILED" in (row["caption_status"], row["graphics_status"]):
                        item.setForeground(QBrush(QColor("#e5584f")))
                    self.table.setItem(r, col, item)
                if row["scene_id"] in keep or (not keep and row["scene_id"] == self.scene_id):
                    from PySide6.QtWidgets import QTableWidgetSelectionRange

                    self.table.setRangeSelected(QTableWidgetSelectionRange(r, 0, r, len(SCENE_COLUMNS) - 1), True)
        finally:
            self._loading = False
        dur = max((s.end for s in p.scenes), default=0.0) if p else 0.0
        self.slider.setRange(0, int(dur * 10))
        self._fill_lists()
        self._fill_audio_tab()
        self._fill_voice()

    def _on_select_scene(self) -> None:
        if self._loading:
            return
        ids = self._selected()
        self.scene_id = ids[0] if len(ids) == 1 else (self.scene_id if ids else None)
        self._fill_lists()
        p = self.ctx.ws.project
        locked = p.presentation_generation.locked_scenes if p else []
        self._loading = True
        try:
            for a, b in self.lock_btns.items():
                b.setChecked(a == "SCENE" and self.scene_id in locked or (a == "CAPTION" and self._all_locked(KIND="caption")) or (a == "GRAPHIC" and self._all_locked(KIND="text")))
        finally:
            self._loading = False
        sc = next((s for s in p.scenes if s.id == self.scene_id), None) if p else None
        if sc is not None and len(ids) == 1:
            self.slider.setValue(int(sc.start * 10))

    def _all_locked(self, KIND: str) -> bool:
        p = self.ctx.ws.project
        cl = [c for c in p.timeline.all_clips() if c.scene_id == self.scene_id and c.kind == KIND] if p and self.scene_id else []
        return bool(cl) and all(c.locked for c in cl)

    def _lock(self, aspect: str, on: bool) -> None:
        if self._loading or not self.scene_id:
            return
        sid = self.scene_id
        self.act(lambda: self._svc().set_lock(sid, aspect, on), "Lock")

    # ---- lists
    def _scene_clips(self, pred):
        p = self.ctx.ws.project
        if p is None:
            return []
        cl = [c for c in p.timeline.all_clips() if pred(c) and (self.scene_id is None or c.scene_id == self.scene_id)]
        return sorted(cl, key=lambda c: c.timeline_start)

    def _fill_lists(self) -> None:
        p = self.ctx.ws.project
        if p is None:
            return
        self._loading = True
        try:
            for table, pred, cells in (
                (self.cap_table, lambda c: c.kind == "caption", lambda c: (f"{c.timeline_start:.2f}", f"{c.timeline_end:.2f}", str((c.text or {}).get("text", "")), str((c.text or {}).get("style_id", "")), c.created_by)),
                (self.gfx_table, lambda c: c.kind in ("text", "graphic"), lambda c: (f"{c.timeline_start:.2f}", f"{c.duration:.2f}", str((c.text or {}).get("content", "evidence highlight")),
                                                                                     str((c.text or {}).get("variant", c.kind)), c.created_by)),
                (self.audio_table, lambda c: c.audio.get("role") in ("MUSIC", "SFX"), lambda c: (f"{c.timeline_start:.2f}", f"{c.timeline_end:.2f}",
                                                                                                   f"{c.audio.get('role')}: {(p.assets.get(c.asset_id).name if p.assets.get(c.asset_id) else c.asset_id)}",
                                                                                                   f"{float(c.audio.get('volume', 1)) * 100:.0f}%", c.created_by)),
            ):
                clips = self._scene_clips(pred) if table is not self.audio_table else [c for c in p.timeline.all_clips() if pred(c) and (self.scene_id is None or c.scene_id in ("", self.scene_id)
                                                                                                                                 or c.audio.get("role") == "MUSIC")]
                clips.sort(key=lambda c: c.timeline_start)
                table.setRowCount(len(clips))
                for r, c in enumerate(clips):
                    d = p.presentation_decisions.get(c.ai_decision_id)
                    for col, text in enumerate(cells(c)):
                        item = QTableWidgetItem(text)
                        item.setData(Qt.ItemDataRole.UserRole, c.id)
                        if d is not None and d.confidence < LOW:
                            item.setForeground(QBrush(QColor("#e0a030")))
                        if c.created_by == "USER" and col == 4:
                            f = item.font()
                            f.setBold(True)
                            item.setFont(f)
                        table.setItem(r, col, item)
                    if c.id == self.clip_id:
                        table.selectRow(r)
        finally:
            self._loading = False
        self._show_clip()

    def _pick(self, table: QTableWidget) -> None:
        if self._loading:
            return
        rows = table.selectionModel().selectedRows()
        if rows:
            self.clip_id = table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            self._show_clip()
            c = self.ctx.ws.project.timeline.get_clip(self.clip_id)
            if c is not None:
                self.slider.setValue(int(c.timeline_start * 10))

    def _on_clip_selected(self, clip_id: str | None) -> None:
        p = self.ctx.ws.project
        if not clip_id or p is None:
            return
        c = p.timeline.get_clip(clip_id)
        if c is not None and (c.kind in ("caption", "text", "graphic") or c.audio.get("role") in ("MUSIC", "SFX")):
            self.clip_id = clip_id
            self.tabs.setCurrentIndex(0 if c.kind == "caption" else 1 if c.kind in ("text", "graphic") else 2)
            self._fill_lists()

    def _show_clip(self) -> None:
        svc = self._svc()
        d = svc.describe(self.clip_id) if self.clip_id and self.ctx.ws.project else None
        self._loading = True
        try:
            for lock in (self.cap_lock_btn, self.gfx_lock_btn, self.au_lock_btn):
                lock.setChecked(bool(d and d["locked"]))
            if d is None:
                for lab in (self.cap_info, self.gfx_info, self.au_info):
                    lab.setText("Select an object above to inspect and edit it.")
                return
            info = (f"<b>Created by:</b> {d['created_by']} &nbsp; <b>Scene:</b> {self._label(d['scene_id']) if d['scene_id'] else '—'} &nbsp; "
                    f"<b>Confidence:</b> {d['confidence']:.0f}%<br><b>Reason:</b> “{d['reason'] or '—'}”" if d["confidence"] is not None else
                    f"<b>Created by:</b> {d['created_by']}")
            if d["kind"] == "CAPTION":
                self.cap_info.setText(info)
                self.cap_text.setText(d["text"])
                self.cap_style_edit.setCurrentIndex(max(0, self.cap_style_edit.findData(d["style_id"])))
                self.cap_pos_edit.setCurrentIndex(max(0, self.cap_pos_edit.findData(d["position"])))
                self.cap_hl_edit.setCurrentIndex(max(0, self.cap_hl_edit.findData(d["highlight_mode"])))
                self.cap_in.setCurrentText((d["animation"].get("in") or {}).get("preset", "fade_in") if isinstance(d["animation"].get("in"), dict) else "fade_in")
                self.cap_font.setText(d["font"])
                self.cap_size.setValue(d["size_pct"])
                self.cap_weight.setCurrentText(d["weight"])
            elif d["kind"] in ("GRAPHIC", "EVIDENCE"):
                self.gfx_info.setText(info)
                t = d["text"] or {}
                self.gfx_content.setText(str(t.get("content", "")))
                self.gfx_subtitle.setText(str(t.get("subtitle", "")))
                pos = t.get("position", (0.5, 0.5))
                self.gfx_x.setValue(pos[0])
                self.gfx_y.setValue(pos[1])
                self.gfx_size.setValue(int(t.get("size", 56)))
                self.gfx_start.setValue(d["start"])
                self.gfx_dur.setValue(d["end"] - d["start"])
                an = d["animation"] or {}
                ain, aout = an.get("in") if isinstance(an.get("in"), dict) else {}, an.get("out") if isinstance(an.get("out"), dict) else {}
                self.gfx_in.setCurrentText(ain.get("preset", "none"))
                self.gfx_out.setCurrentText(aout.get("preset", "none"))
                self.gfx_in_dur.setValue(float(ain.get("duration", 0.3)))
                self.gfx_ease.setCurrentText(ain.get("easing", "ease_out"))
                self.gfx_tool.setCurrentText((d["effects"].get("evidence") or {}).get("tool", "FOCUS_BOX"))
            else:
                self.au_info.setText(info + (f"<br><b>Asset:</b> {d['asset']} &nbsp; <b>Start:</b> {d['start']:.2f}s &nbsp; <b>End:</b> {d['end']:.2f}s"
                                             + (f" &nbsp; <b>Ducking keyframes:</b> {len(d['keyframes'])}" if d["kind"] == "MUSIC" else f" &nbsp; <b>Category:</b> {d['category']}")))
                self.au_volume.setValue(float(d["volume"]) * 100)
                self.au_fade_in.setValue(float(d["fade_in"]))
                self.au_fade_out.setValue(float(d["fade_out"]))
                self.au_loop.setVisible(d["kind"] == "MUSIC")
                if d["kind"] == "MUSIC" and d["assignment_id"]:
                    self.duck_edit.setPlainText("\n".join(f"{t:.2f}, {v * 100:.1f}" for t, v in svc.ducking_keyframes(d["assignment_id"])[:400]))
        finally:
            self._loading = False

    # ---- edit actions
    def _apply_caption(self) -> None:
        if not self.clip_id:
            return
        cid = self.clip_id
        self.act(lambda: self._svc().update_caption(cid, text=self.cap_text.text(), style_id=self.cap_style_edit.currentData(), position=self.cap_pos_edit.currentData(),
                                                     highlight_mode=self.cap_hl_edit.currentData(), style_overrides=self._style_overrides()), "Edit caption")
        self.ctx.guard(self, lambda: self._svc().set_graphic_animation(cid, "in", self.cap_in.currentText()), modal=True, title="Caption animation")
        self.refresh()

    def _style_overrides(self) -> dict:
        return {"font": self.cap_font.text().strip() or "Sans", "size_rel": self.cap_size.value() / 100, "weight": self.cap_weight.currentText(), "alignment": self.cap_align.currentText(),
                "background": self.cap_bg.currentText(), "background_opacity": 0.6 if self.cap_bg.currentText() == "box" else 0.0, "opacity": self.cap_opacity.value(),
                "shadow": self.cap_shadow.isChecked(), "outline_width": self.cap_outline.value() / 100, "line_spacing": self.cap_spacing.value()}

    def _save_style(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        from app.captions.styles import PRESETS as P

        name, ok = QInputDialog.getText(self, "Save caption style", "Style name:")
        if not ok or not name.strip():
            return
        from dataclasses import replace

        base = replace(P.get(self.cap_style_edit.currentData()) or P["professional"], style_id=name.strip().lower().replace(" ", "_"), name=name.strip(), **self._style_overrides())
        self.act(lambda: self._svc().save_caption_style(base), "Save style")
        self.cap_style_edit.addItem(base.name, base.style_id)

    def _apply_graphic(self) -> None:
        if not self.clip_id:
            return
        cid = self.clip_id
        c = self.ctx.ws.project.timeline.get_clip(cid)

        def go() -> None:
            if c.kind == "text":
                self._svc().update_graphic(cid, content=self.gfx_content.text(), subtitle=self.gfx_subtitle.text(), position=(self.gfx_x.value(), self.gfx_y.value()),
                                           size=self.gfx_size.value(), start=self.gfx_start.value(), duration=self.gfx_dur.value())
            else:
                self._svc().set_evidence_tool(cid, self.gfx_tool.currentText())
            self._svc().set_graphic_animation(cid, "in", self.gfx_in.currentText(), duration=self.gfx_in_dur.value(), easing=self.gfx_ease.currentText())
            self._svc().set_graphic_animation(cid, "out", self.gfx_out.currentText())

        self.act(go, "Edit graphic")

    def _new_graphic(self) -> None:
        t = self.preview.time
        text = self.gfx_content.text().strip()
        self.act(lambda: self._svc().add_graphic(self.gfx_new_type.currentText(), text, t, 2.5, subtitle=self.gfx_subtitle.text()), "Add graphic")

    def _lock_clip(self, on: bool) -> None:
        if self._loading or not self.clip_id:
            return
        cid = self.clip_id
        self.act(lambda: self._svc().lock_clip(cid, on), "Lock")

    def _apply_audio(self) -> None:
        d = self._svc().describe(self.clip_id) if self.clip_id else None
        if d is None:
            return
        cid, vol = self.clip_id, self.au_volume.value() / 100

        def go() -> None:
            if d["kind"] == "MUSIC" and d["assignment_id"]:
                self._svc().set_music(d["assignment_id"], volume=vol, fade_in=self.au_fade_in.value(), fade_out=self.au_fade_out.value(), loop=self.au_loop.isChecked())
            else:
                self._svc().set_clip_audio(cid, volume=vol, fade_in=self.au_fade_in.value(), fade_out=self.au_fade_out.value())

        self.act(go, "Edit audio")

    def _delete_audio(self) -> None:
        d = self._svc().describe(self.clip_id) if self.clip_id else None
        if d is None:
            return
        cid = self.clip_id
        if d["kind"] == "MUSIC" and d["assignment_id"]:
            aid = d["assignment_id"]
            self.act(lambda: self._svc().delete_music(aid), "Delete music")
        else:
            self.act(lambda: self.ctx.ws.timeline.delete_clip(cid), "Delete sound effect")
        self.clip_id = None

    def _apply_ducking(self) -> None:
        d = self._svc().describe(self.clip_id) if self.clip_id else None
        if d is None or d["kind"] != "MUSIC" or not d["assignment_id"]:
            self.ctx.status("Select a music clip first.")
            return
        pts = []
        for line in self.duck_edit.toPlainText().splitlines():
            if line.strip():
                a, b = line.replace(";", ",").split(",")[:2]
                pts.append((float(a), float(b.strip().rstrip("%")) / 100))
        aid = d["assignment_id"]
        self.act(lambda: self._svc().set_ducking_keyframes(aid, pts), "Ducking keyframes")

    def _import(self, role: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Import {role}", "", "Audio (*.wav *.mp3 *.aac *.m4a)")
        if path:
            cat = self.sfx_cat.currentText() if role == "sfx" else None
            self.act(lambda: self._svc().import_audio(Path(path), role, cat), "Import audio")

    def _play_library_item(self, item) -> None:
        """Preview a library asset (double-click)."""
        p = self.ctx.ws.project
        a = p.assets.get(item.data(Qt.ItemDataRole.UserRole)) if p else None
        if a is not None:
            self.mix_player.load(p.asset_path(a))
            self.mix_player.play()
            self.preview_status.setText(f"Playing {a.name}")

    def _add_music(self) -> None:
        item = self.music_lib.currentItem() or (self.music_lib.item(0) if self.music_lib.count() else None)
        if item is None:
            self.ctx.status("Import a music file first.")
            return
        aid = item.data(Qt.ItemDataRole.UserRole)
        self.act(lambda: self._svc().add_music(aid), "Add music")

    def _add_sfx(self) -> None:
        item = self.sfx_lib.currentItem() or (self.sfx_lib.item(0) if self.sfx_lib.count() else None)
        if item is None:
            self.ctx.status("Import a sound effect first.")
            return
        aid, t = item.data(Qt.ItemDataRole.UserRole), self.preview.time
        self.act(lambda: self._svc().add_sfx(aid, t), "Add sound effect")

    def _fill_audio_tab(self) -> None:
        p = self.ctx.ws.project
        if p is None:
            return
        for lst, role in ((self.music_lib, "music"), (self.sfx_lib, "sfx")):
            cur = lst.currentItem().data(Qt.ItemDataRole.UserRole) if lst.currentItem() else None
            lst.clear()
            for a in self._svc().library(role):
                it = QListWidgetItem(f"{a.name}  ({a.duration or 0:.1f}s{', ' + str(a.extra.get('category')) if role == 'sfx' and a.extra.get('category') else ''})")
                it.setData(Qt.ItemDataRole.UserRole, a.id)
                lst.addItem(it)
                if a.id == cur:
                    lst.setCurrentItem(it)

    def _fill_voice(self) -> None:
        p = self.ctx.ws.project
        if p is None:
            return
        a = p.audio_analysis
        if a is None:
            self.an_label.setText("Not analysed yet.")
            self.an_issues.setPlainText("")
        else:
            f = lambda v, u="": "—" if v is None else f"{v:.1f}{u}"  # noqa: E731
            self.an_label.setText(f"Duration {a.duration:.1f}s · Peak {f(a.peak_db, ' dB')} · RMS {f(a.rms_db, ' dB')} · Loudness {f(a.lufs, ' LUFS')} · Dynamic range {f(a.dynamic_range_db, ' dB')}"
                                  f"\nSpeaking rate {f(a.speaking_rate_wps, ' words/s')} · Speech {f((a.speech_ratio or 0) * 100 if a.speech_ratio is not None else None, '%')} · "
                                  f"Pauses {len(a.pauses)} · Silences {len(a.silence_regions)} · Emphasis candidates {len(a.emphasis_candidates)}"
                                  + ("\n⚠ The voice-over changed since this analysis." if self._svc().staleness()["analysis_outdated"] else "")
                                  + "\nSpeaker changes: not supported by this backend.")
            self.an_issues.setPlainText("\n".join(f"[{i.code}] {i.message}" for i in a.issues) or "No audio problems found.")
        from app.audio.processing import describe

        self.vp_label.setText("Chain: " + " → ".join(describe(p.audio_processing)))
        mask = self._svc().masking() if p.timeline.all_clips() else []
        self.masking_label.setText("⚠ " + "\n⚠ ".join(m.message for m in mask[:4]) if mask else "Voice priority: music and sound effects do not mask the voice.")

    def _apply_voice(self) -> None:
        self.act(lambda: self._svc().update_voice_processing(
            gain_db=self.vp_gain.value(), highpass_hz=self.vp_highpass.value(), compression=self.vp_comp.isChecked(), limiter=self.vp_limiter.isChecked(),
            noise_reduction=self.vp_noise.isChecked(), normalize=self.vp_norm.isChecked(), eq_preset=self.vp_eq.currentText()), "Voice processing")

    # ================================================================== preview
    def _preview(self) -> None:
        mode = PreviewMode(self.preview_mode.currentData())
        self.preview_status.setText(f"Rendering {mode.value.replace('_', ' + ').lower()} preview in the background…")
        t0 = self.slider.value() / 10.0

        def ready(path) -> None:
            if path is None:
                self.preview_status.setText("Nothing audible in that mix (or it could not be created).")
                return
            self.preview_status.setText(f"Preview ready: {mode.value.replace('_', ' + ').title()} — {Path(path).name}")
            self.mix_player.load(Path(path))
            self.mix_player.play()

        self.ctx.guard(self, lambda: self._svc().preview_mix(mode, t0, t0 + 20.0, on_ready=ready), modal=True, title="Audio preview")

    def _scrub(self, value: int) -> None:
        t = value / 10.0
        self.time_label.setText(f"{int(t // 60)}:{t % 60:04.1f}")
        fs = self.preview.show_time(t)
        if fs is not None:
            bits = [f"{len([l for l in fs.layers if l.kind == 'media'])} visual", f"{len([l for l in fs.layers if l.kind == 'caption'])} caption",
                    f"{len([l for l in fs.layers if l.kind in ('text', 'graphic')])} graphics", f"voice {fs.audio.get('VOICE', 0):.0%}" if fs.audio else "",
                    f"music {fs.music_level:.0%}", f"sfx {fs.audio.get('SFX', 0):.0%}" if fs.audio else ""]
            self.preview_note.setText("Preview of timeline data: " + " · ".join(b for b in bits if b))
        if self.voice_player.position is not None and abs(self.voice_player.position - t) > 0.25 and self.play_btn.text() != "Pause":
            self.voice_player.seek(t)

    def _toggle_play(self) -> None:
        if self.play_btn.text() == "Pause":
            self.voice_player.pause()
        else:
            self.voice_player.seek(self.slider.value() / 10.0)
            self.voice_player.play()

    def _on_position(self, t: float) -> None:
        if self.play_btn.text() == "Pause" and self.isVisible():
            self.slider.blockSignals(True)
            self.slider.setValue(int(t * 10))
            self.slider.blockSignals(False)
            self._scrub(int(t * 10))

    def pause(self) -> None:
        if self.play_btn.text() == "Pause":
            self.voice_player.pause()
        self.mix_player.pause()
