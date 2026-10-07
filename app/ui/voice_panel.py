"""Voice-over panel: import, playback, transcription (with progress / failure / outdated states) and transcript viewer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import AUDIO_EXTENSIONS
from app.core.exceptions import AppError
from app.core.timecode import format_duration_short, format_timecode
from app.jobs.job import Job, JobStatus
from app.preview.player import QtPreviewPlayer
from app.transcription.alignment import Verdict
from app.transcription.status import AlignmentStatus, TranscriptStatus, alignment_status, transcript_status
from app.ui.context import UiContext
from app.ui.preview_panel import TransportControls
from app.ui.transcript_viewer import TranscriptViewer


class VoicePanel(QWidget):
    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.player = QtPreviewPlayer(self)
        self.player.failed.connect(ctx.status)
        self._loaded: Path | None = None
        self._job: Job | None = None

        title = QLabel("Voice-over")
        title.setObjectName("title")
        self.filename = QLabel("—")
        self.duration = QLabel("—")
        info = QGroupBox("Voice-over")
        form = QFormLayout(info)
        form.addRow("File", self.filename)
        form.addRow("Duration", self.duration)

        self.import_btn = QPushButton("Import voice-over…")
        self.import_btn.setObjectName("primary")
        self.remove_btn = QPushButton("Remove")
        self.timeline_btn = QPushButton("Add to timeline")
        buttons = QHBoxLayout()
        for b in (self.import_btn, self.remove_btn, self.timeline_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.transport = TransportControls(self.player)

        # --- transcription block ---
        self.state_label = QLabel()
        self.state_label.setObjectName("transcriptState")
        self.engine_label = QLabel()
        self.engine_label.setObjectName("muted")
        self.engine_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setObjectName("transcribeProgress")
        self.progress_msg = QLabel()
        self.progress_msg.setObjectName("muted")
        self.summary_label = QLabel()
        self.summary_label.setObjectName("transcriptSummary")
        self.align_label = QLabel()
        self.align_label.setObjectName("muted")
        self.align_label.setWordWrap(True)
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.setObjectName("primary")
        self.cancel_btn = QPushButton("Cancel")
        self.retry_btn = QPushButton("Retry")
        self.choose_btn = QPushButton("Choose Another File")
        self.realign_btn = QPushButton("Re-align script")
        actions = QHBoxLayout()
        for b in (self.transcribe_btn, self.cancel_btn, self.retry_btn, self.choose_btn, self.realign_btn):
            actions.addWidget(b)
        actions.addStretch(1)
        box = QGroupBox("Transcription")
        bl = QVBoxLayout(box)
        for w in (self.state_label, self.progress, self.progress_msg, self.summary_label, self.align_label, self.engine_label):
            bl.addWidget(w)
        bl.addLayout(actions)

        self.viewer = TranscriptViewer(ctx.ws.settings.theme)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(buttons)
        top = QHBoxLayout()
        top.addWidget(info, 1)
        top.addWidget(box, 2)
        layout.addLayout(top)
        layout.addWidget(self.transport)
        layout.addWidget(self.viewer, 1)

        self.import_btn.clicked.connect(self.import_dialog)
        self.choose_btn.clicked.connect(self.import_dialog)
        self.remove_btn.clicked.connect(lambda: ctx.guard(self, ctx.ws.media.remove_voice_over, modal=True))
        self.timeline_btn.clicked.connect(self._add_to_timeline)
        self.transcribe_btn.clicked.connect(lambda: self._transcribe(force=False))
        self.retry_btn.clicked.connect(lambda: self._transcribe(force=True))
        self.cancel_btn.clicked.connect(self._cancel)
        self.realign_btn.clicked.connect(lambda: ctx.guard(self, ctx.ws.transcripts.realign, modal=True, title="Align script"))
        self.viewer.seek_requested.connect(self.seek)
        self.player.position_changed.connect(self.viewer.set_position)

        b = ctx.bridge
        for topic in ("project.opened", "project.closed"):
            b.on(topic, lambda p: self._reset())
        b.on("project.changed", lambda p: self.refresh() if p.get("scope") in ("voice_over", "assets", "transcript", "script") else None)
        b.on("job.updated", self._on_job)
        b.on("transcription.failed", lambda p: self.refresh())
        self._reset()

    # ------------------------------------------------------------ actions
    def import_dialog(self) -> None:
        pattern = " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))
        file, _ = QFileDialog.getOpenFileName(self, "Import voice-over", "", f"Audio ({pattern})")
        if file:
            self.import_path(Path(file))

    def import_path(self, path: Path) -> None:
        self.ctx.guard(self, lambda: self.ctx.ws.media.import_voice_over(path), modal=True, title="Voice-over")

    def _add_to_timeline(self) -> None:
        vo = self.ctx.ws.project.voice_over if self.ctx.ws.project else None
        if vo and vo.asset_id:
            self.ctx.guard(self, lambda: self.ctx.ws.timeline.add_asset(vo.asset_id), modal=True, title="Timeline")

    def _transcribe(self, force: bool) -> None:
        def start() -> None:
            self._job = self.ctx.ws.transcripts.transcribe(force=force)

        self.ctx.guard(self, start, modal=True, title="Transcription")
        self.refresh()

    def _cancel(self) -> None:
        if self._job:
            self.ctx.ws.jobs.cancel(self._job.id)

    def seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        self.viewer.set_position(seconds)

    # ------------------------------------------------------------ state
    def _on_job(self, payload: dict) -> None:
        job: Job = payload["job"]
        if self._job is not None and job.id == self._job.id:
            self.refresh()

    def _reset(self) -> None:
        self._job = None
        self._loaded = None
        self.player.unload()
        self.refresh()

    def _running(self) -> bool:
        return self._job is not None and self._job.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    def refresh(self) -> None:
        ws = self.ctx.ws
        project = ws.project
        vo = project.voice_over if project else None
        has = bool(vo and vo.asset_id)
        asset = project.assets.get(vo.asset_id) if (project and has) else None
        file_ok = bool(asset and project.asset_path(asset).is_file())
        self.import_btn.setEnabled(project is not None and not self._running())
        self.import_btn.setText("Replace voice-over…" if has else "Import voice-over…")
        self.remove_btn.setEnabled(has)
        self.timeline_btn.setEnabled(has)
        self.filename.setText((vo.filename if has else None) or "No voice-over imported")
        self.duration.setText(format_duration_short(vo.duration) if has and vo.duration else "—")
        self._sync_player(project, asset, file_ok)

        status = transcript_status(project) if project else TranscriptStatus.NOT_STARTED
        tr = project.transcription.transcript if project else None
        running = self._running()
        provider_text, provider_ok = self._provider_line()
        self.engine_label.setText(provider_text)
        for w in (self.progress, self.progress_msg):
            w.setVisible(running)
        self.cancel_btn.setVisible(running)
        self.summary_label.setVisible(False)
        self.retry_btn.setVisible(False)
        self.choose_btn.setVisible(False)
        self.transcribe_btn.setVisible(not running)
        self.transcribe_btn.setText("Transcribe")
        can = has and file_ok and provider_ok and not running
        self.transcribe_btn.setEnabled(can)

        if running and self._job:
            self.state_label.setText("Transcribing…")
            self.progress.setValue(int(self._job.progress))
            self.progress_msg.setText(self._job.message)
        elif not has:
            self.state_label.setText("Transcription:  [Not Started]  — import a voice-over first")
        elif status is TranscriptStatus.NOT_STARTED:
            self.state_label.setText("Transcription:  [Not Started]")
        elif status is TranscriptStatus.COMPLETE and tr is not None:
            self.state_label.setText("✓ Transcription Complete")
            self.summary_label.setVisible(True)
            self.summary_label.setText(
                f"Words: {len(tr.words):,}\nSentences: {len(tr.sentences):,}\nDuration: {format_duration_short(tr.audio.duration)}")
            self.transcribe_btn.setText("Re-transcribe")
            self.transcribe_btn.setEnabled(can)
        elif status is TranscriptStatus.OUTDATED:
            self.state_label.setText("⚠ Transcript: OUTDATED — the voice-over changed. Re-transcribe to update it.")
            self.transcribe_btn.setText("Re-transcribe")
            self.transcribe_btn.setEnabled(can)
        elif status is TranscriptStatus.FAILED:
            self.state_label.setText(f"Transcription failed.\n{project.transcription.last_error or ''}")
            self.retry_btn.setVisible(True)
            self.choose_btn.setVisible(True)
            self.retry_btn.setEnabled(can)
            self.transcribe_btn.setVisible(False)

        # alignment line + viewer
        al = project.script_alignment if project else None
        self.realign_btn.setVisible(bool(tr and project and project.script.text.strip() and alignment_status(project) is AlignmentStatus.OUTDATED))
        if tr is not None and al is not None and project is not None:
            note = {Verdict.IDENTICAL: "the narration matches the script exactly",
                    Verdict.MINOR_DIFFERENCES: "minor wording differences",
                    Verdict.SIGNIFICANT_DIFFERENCES: "significant differences — review the transcript against the script"}[al.verdict]
            stale = " (script changed since — Re-align)" if alignment_status(project) is AlignmentStatus.OUTDATED else ""
            st = al.stats
            self.align_label.setText(f"Script alignment: {note}{stale}. {st.coverage:.0%} of script words heard; "
                                     f"{st.script_only} missing, {st.transcript_only} added, {st.reordered} reordered.")
        else:
            self.align_label.setText("Script alignment: add a script on the Script page to compare it with what is spoken." if tr else "")
        self.viewer.set_transcript(tr)

    def _provider_line(self) -> tuple[str, bool]:
        try:
            p = self.ctx.ws.transcripts.resolve_provider()
        except AppError as exc:
            return f"No usable transcription engine: {exc.user_message}", False
        note = f" — {p.accuracy_note}" if p.accuracy_note else ""
        return f"Engine: {p.name} ({p.kind}){note}", True

    def _sync_player(self, project, asset, file_ok: bool) -> None:
        path = project.asset_path(asset) if (project and asset and file_ok) else None
        if path != self._loaded:
            self._loaded = path
            self.player.unload()
            if path is not None:
                self.player.load(path)
        self.transport.set_enabled(path is not None)

    def pause(self) -> None:
        self.player.pause()
