"""Transcript viewer: searchable, click-to-seek, follows playback (sentence + current word)."""

from __future__ import annotations

import bisect

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from app.core.timecode import format_timecode
from app.transcription.models import Transcript
from app.ui.theme import palette


class _View(QTextBrowser):
    clicked = Signal(int)  # document position
    double_clicked = Signal(int)
    hovered = Signal(int, QPoint)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setOpenLinks(False)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        super().mouseReleaseEvent(e)
        if e.button() == Qt.MouseButton.LeftButton and not self.textCursor().hasSelection():
            self.clicked.emit(self.cursorForPosition(e.position().toPoint()).position())

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        self.double_clicked.emit(self.cursorForPosition(e.position().toPoint()).position())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        super().mouseMoveEvent(e)
        self.hovered.emit(self.cursorForPosition(e.position().toPoint()).position(), e.globalPosition().toPoint())


class TranscriptViewer(QWidget):
    seek_requested = Signal(float)  # seconds

    def __init__(self, theme: str = "dark", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.transcript: Transcript | None = None
        self._sent_ranges: list[tuple[int, int]] = []  # char ranges incl. header
        self._sent_starts: list[int] = []
        self._word_ranges: list[tuple[int, int, int]] = []  # (start_pos, end_pos, word index)
        self._word_starts: list[int] = []
        self._matches: list[tuple[int, int]] = []
        self._t_sent: list[float] = [0.0]
        self._t_word: list[float] = [0.0]
        self._match_index = -1
        self.current_sentence = -1
        self.current_word = -1

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search transcript…")
        self.search.setClearButtonEnabled(True)
        self.prev_btn, self.next_btn = QPushButton("▲"), QPushButton("▼")
        for b in (self.prev_btn, self.next_btn):
            b.setFixedWidth(34)
        self.match_label = QLabel("")
        self.match_label.setObjectName("muted")
        self.timings = QCheckBox("Word timings")
        self.follow = QCheckBox("Follow playback")
        self.follow.setChecked(True)
        self.view = _View()
        self.view.setObjectName("transcriptView")
        self.empty = QLabel("No transcript yet.")
        self.empty.setObjectName("placeholder")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bar = QHBoxLayout()
        for w in (self.search, self.prev_btn, self.next_btn, self.match_label, self.timings, self.follow):
            bar.addWidget(w)
        bar.setStretchFactor(self.search, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.empty)
        layout.addWidget(self.view, 1)

        self.search.textChanged.connect(self._search)
        self.search.returnPressed.connect(lambda: self._step(1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.timings.toggled.connect(lambda _on: self._rebuild())
        self.view.clicked.connect(self._on_click)
        self.view.double_clicked.connect(self._on_double_click)
        self.view.hovered.connect(self._on_hover)
        self.set_transcript(None)

    # ------------------------------------------------------------ content
    def set_transcript(self, transcript: Transcript | None, outdated: bool = False) -> None:
        self.transcript = transcript
        self.empty.setVisible(transcript is None)
        self.view.setVisible(transcript is not None)
        self.current_sentence = self.current_word = -1
        self._rebuild()
        for w in (self.search, self.prev_btn, self.next_btn, self.timings, self.follow):
            w.setEnabled(transcript is not None)

    def _rebuild(self) -> None:
        self.view.clear()
        self._sent_ranges, self._word_ranges, self._matches = [], [], []
        tr = self.transcript
        if tr is None:
            self._t_sent, self._t_word = [0.0], [0.0]
            return
        self._t_sent = [x.start for x in tr.sentences] or [0.0]
        self._t_word = [x.start for x in tr.words] or [0.0]
        c = palette(self._theme)
        head = QTextCharFormat()
        head.setForeground(QColor(c["accent"]))
        head.setFont(QFont(self.view.font().family(), max(8, self.view.font().pointSize() - 2), QFont.Weight.DemiBold))
        body = QTextCharFormat()
        small = QTextCharFormat()
        small.setForeground(QColor(c["muted"]))
        small.setFont(QFont(self.view.font().family(), max(7, self.view.font().pointSize() - 3)))
        cur = QTextCursor(self.view.document())
        wm = tr.word_map()
        index = {w.word_id: i for i, w in enumerate(tr.words)}
        for k, s in enumerate(tr.sentences):
            begin = cur.position()
            cur.insertText(format_timecode(s.start) + "\n", head)
            for wid in s.word_ids:
                w = wm[wid]
                a = cur.position()
                cur.insertText(w.text, body)
                self._word_ranges.append((a, cur.position(), index[wid]))
                if self.timings.isChecked():
                    cur.insertText(f" ‹{w.start:.2f}›", small)
                cur.insertText(" ", body)
            self._sent_ranges.append((begin, cur.position()))
            cur.insertBlock()
        self._sent_starts = [a for a, _ in self._sent_ranges]
        self._word_starts = [a for a, _, _ in self._word_ranges]
        self._search(self.search.text())
        self._selections()

    # ------------------------------------------------------------ lookups
    def _sentence_at_pos(self, pos: int) -> int:
        if not self._sent_starts:
            return -1
        return max(0, bisect.bisect_right(self._sent_starts, pos) - 1)

    def _word_at_pos(self, pos: int) -> int | None:
        i = bisect.bisect_right(self._word_starts, pos) - 1
        if i >= 0:
            a, b, wi = self._word_ranges[i]
            if a <= pos <= b:
                return wi
        return None

    # ------------------------------------------------------------ interaction
    def _on_click(self, pos: int) -> None:
        if self.transcript and self._sent_starts:
            self.seek_requested.emit(self.transcript.sentences[self._sentence_at_pos(pos)].start)

    def _on_double_click(self, pos: int) -> None:
        wi = self._word_at_pos(pos)
        if self.transcript and wi is not None:
            self.seek_requested.emit(self.transcript.words[wi].start)

    def _on_hover(self, pos: int, global_pos: QPoint) -> None:
        wi = self._word_at_pos(pos)
        if self.transcript and wi is not None:
            w = self.transcript.words[wi]
            conf = f"  •  confidence {w.confidence:.0%}" if w.confidence is not None else ""
            QToolTip.showText(global_pos, f"“{w.text}”  {w.start:.3f} → {w.end:.3f} s{conf}", self.view)
        else:
            QToolTip.hideText()

    # ------------------------------------------------------------ playback highlight
    def set_position(self, t: float) -> None:
        tr = self.transcript
        if tr is None or not self._sent_ranges:
            return
        si = bisect.bisect_right(self._t_sent, t) - 1 if t >= self._t_sent[0] else -1
        wi = bisect.bisect_right(self._t_word, t) - 1 if t >= self._t_word[0] else -1
        if wi >= 0 and t > tr.words[wi].end + 0.25:
            wi = -1  # silence: no word is "being spoken"
        if (si, wi) == (self.current_sentence, self.current_word):
            return
        sentence_changed = si != self.current_sentence
        self.current_sentence, self.current_word = si, wi
        self._selections()
        if self.follow.isChecked() and si >= 0 and sentence_changed:
            cur = QTextCursor(self.view.document())
            cur.setPosition(self._sent_ranges[si][0])
            self.view.setTextCursor(cur)
            self.view.ensureCursorVisible()

    def _selections(self) -> None:
        c = palette(self._theme)
        sels: list[QTextEdit.ExtraSelection] = []

        def add(a: int, b: int, bg: str | None = None, fg: str | None = None, bold: bool = False, underline: bool = False):
            sel = QTextEdit.ExtraSelection()
            cur = QTextCursor(self.view.document())
            cur.setPosition(a)
            cur.setPosition(b, QTextCursor.MoveMode.KeepAnchor)
            sel.cursor = cur
            fmt = QTextCharFormat()
            if bg:
                fmt.setBackground(QBrush(QColor(bg)))
            if fg:
                fmt.setForeground(QBrush(QColor(fg)))
            if bold:
                fmt.setFontWeight(QFont.Weight.Bold)
            fmt.setFontUnderline(underline)
            sel.format = fmt
            sels.append(sel)

        if 0 <= self.current_sentence < len(self._sent_ranges):
            a, b = self._sent_ranges[self.current_sentence]
            add(a, b, bg=c["panel2"])
        for i, (a, b) in enumerate(self._matches):
            add(a, b, bg="#f2c94c" if i == self._match_index else "#7a6a2a", fg="#000000" if i == self._match_index else None)
        if self.current_word >= 0:
            for a, b, wi in self._word_ranges:
                if wi == self.current_word:
                    add(a, b, bg=c["accent"], fg=c["accent_text"], bold=True)
                    break
        self.view.setExtraSelections(sels)

    # ------------------------------------------------------------ search
    def _search(self, text: str) -> None:
        self._matches, self._match_index = [], -1
        text = text.strip()
        if text and self.transcript:
            doc = self.view.document()
            cur = QTextCursor(doc)
            while True:
                cur = doc.find(text, cur)
                if cur.isNull():
                    break
                self._matches.append((cur.selectionStart(), cur.selectionEnd()))
        self.match_label.setText("" if not text else (f"{len(self._matches)} match(es)" if self._matches else "no matches"))
        self._selections()
        if self._matches:
            self._step(1)

    def _step(self, direction: int) -> None:
        if not self._matches:
            return
        self._match_index = (self._match_index + direction) % len(self._matches)
        a, _ = self._matches[self._match_index]
        cur = QTextCursor(self.view.document())
        cur.setPosition(a)
        self.view.setTextCursor(cur)
        self.view.ensureCursorVisible()
        self.match_label.setText(f"{self._match_index + 1} / {len(self._matches)}")
        self._selections()
