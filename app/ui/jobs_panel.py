"""Background-job UI: a compact status strip plus an expandable list of all jobs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.jobs.job import Job, JobStatus
from app.ui.context import UiContext

COLUMNS = ("Job", "Status", "Progress", "Message", "")


class JobsPanel(QWidget):
    """Table of every job with Cancel / Retry actions."""

    def __init__(self, ctx: UiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("jobsTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 130)
        self.table.setColumnWidth(4, 150)
        clear = QPushButton("Clear finished")
        clear.clicked.connect(self._clear_finished)
        top = QHBoxLayout()
        label = QLabel("Background jobs")
        label.setObjectName("muted")
        top.addWidget(label)
        top.addStretch(1)
        top.addWidget(clear)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addLayout(top)
        layout.addWidget(self.table)
        self._rows: dict[str, int] = {}
        ctx.bridge.on("job.added", lambda p: self.update_job(p["job"]))
        ctx.bridge.on("job.updated", lambda p: self.update_job(p["job"]))

    def update_job(self, job: Job) -> None:
        row = self._rows.get(job.id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[job.id] = row
            self.table.setCellWidget(row, 2, QProgressBar())
            self.table.setCellWidget(row, 4, self._actions(job))
        self.table.setItem(row, 0, QTableWidgetItem(job.title))
        self.table.setItem(row, 1, QTableWidgetItem(job.status.value))
        bar = self.table.cellWidget(row, 2)
        assert isinstance(bar, QProgressBar)
        bar.setValue(int(job.progress))
        self.table.setItem(row, 3, QTableWidgetItem(job.error or job.message))
        self._refresh_actions(job, row)

    def _actions(self, job: Job) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 0, 2, 0)
        cancel, retry = QPushButton("Cancel"), QPushButton("Retry")
        cancel.clicked.connect(lambda: self.ctx.ws.jobs.cancel(job.id))
        retry.clicked.connect(lambda: self.ctx.guard(self, lambda: self.ctx.ws.jobs.retry(job.id)))
        lay.addWidget(cancel)
        lay.addWidget(retry)
        w.cancel, w.retry = cancel, retry  # type: ignore[attr-defined]
        return w

    def _refresh_actions(self, job: Job, row: int) -> None:
        w = self.table.cellWidget(row, 4)
        w.cancel.setVisible(not job.status.is_terminal)  # type: ignore[union-attr]
        w.retry.setVisible(job.status in (JobStatus.FAILED, JobStatus.CANCELLED))  # type: ignore[union-attr]

    def _clear_finished(self) -> None:
        self.ctx.ws.jobs.clear_finished()
        alive = {j.id for j in self.ctx.ws.jobs.jobs()}
        for job_id in [i for i in self._rows if i not in alive]:
            self._rows.pop(job_id)
        jobs = self.ctx.ws.jobs.jobs()
        self.table.setRowCount(0)
        self._rows.clear()
        for job in jobs:
            self.update_job(job)


class JobStatusBar(QFrame):
    """Always-visible strip: current job, progress, cancel, and a toggle for the full list."""

    def __init__(self, ctx: UiContext, panel: JobsPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("jobbar")
        self.ctx, self.panel = ctx, panel
        self.message = QLabel("Ready")
        self.message.setMinimumWidth(240)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setVisible(False)
        self.percent = QLabel("")
        self.cancel = QPushButton("Cancel")
        self.cancel.setVisible(False)
        self.more = QLabel("")
        self.more.setObjectName("muted")
        self.toggle = QPushButton("Jobs ▴")
        self.toggle.setCheckable(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        for w in (self.message, self.progress, self.percent, self.cancel):
            lay.addWidget(w)
        lay.addStretch(1)
        lay.addWidget(self.more)
        lay.addWidget(self.toggle)
        self._job: Job | None = None
        self._idle_text = "Ready"

        self.toggle.toggled.connect(self._toggled)
        self.cancel.clicked.connect(self._cancel)
        panel.setVisible(False)
        ctx.bridge.on("job.added", lambda p: self._update())
        ctx.bridge.on("job.updated", lambda p: self._update(p["job"]))
        ctx.bridge.on("app.status", lambda p: self.set_idle_text(p.get("message", "")))

    def set_idle_text(self, text: str) -> None:
        self._idle_text = text or "Ready"
        if not self.ctx.ws.jobs.active_jobs():
            self.message.setText(self._idle_text)

    def _toggled(self, shown: bool) -> None:
        self.panel.setVisible(shown)
        self._refresh_toggle()

    def _refresh_toggle(self) -> None:
        n = len(self.ctx.ws.jobs.active_jobs())
        arrow = "▾" if self.toggle.isChecked() else "▴"
        self.toggle.setText(f"Jobs ({n}) {arrow}" if n else f"Jobs {arrow}")

    def _cancel(self) -> None:
        if self._job:
            self.ctx.ws.jobs.cancel(self._job.id)

    def _update(self, changed: Job | None = None) -> None:
        self._refresh_toggle()
        active = [j for j in self.ctx.ws.jobs.active_jobs() if j.status is not JobStatus.PAUSED]
        if not active:
            self._job = None
            self.progress.setVisible(False)
            self.percent.setText("")
            self.cancel.setVisible(False)
            self.more.setText("")
            failed = changed is not None and changed.status is JobStatus.FAILED
            self.message.setText(f"{changed.title} failed: {changed.error}" if failed else self._idle_text)
            return
        running = [j for j in active if j.status is JobStatus.RUNNING]
        job = changed if changed in running else (running or active)[0]
        self._job = job
        self.message.setText(job.message or job.title)
        self.progress.setVisible(True)
        self.progress.setValue(int(job.progress))
        self.percent.setText(f"{int(job.progress)}%")
        self.cancel.setVisible(True)
        self.more.setText(f"+{len(active) - 1} more" if len(active) > 1 else "")
