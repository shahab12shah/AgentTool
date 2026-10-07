"""Project changes made by the quality-control layer.

Storing an analysis is a recorded (not undoable) update of the QC sections only. Ignoring an issue, changing QC settings and marking an issue fixed are
undoable. None of them touches the timeline: a fix's timeline change is a separate command that QCFixEngine groups with ``MarkIssueFixedCommand`` into one undo step.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.commands import Command
from app.project.project import Project
from app.qc.issue_model import FixRecord, IgnoreRecord, IssueStatus, QCIssue, QCScores, clean_json
from app.qc.settings import QCSettings

MAX_RUNS = 100
MAX_HISTORY = 30
MAX_ISSUES = 3000  # the current findings are persisted with the project: keep the most severe if a run is extremely noisy


class StoreQCRunCommand(Command):
    """Install the result of a QC run: the current issues, scores, the run record, the history archive and the cache state. Not an edit of the video."""

    scope = "qc"
    major = False
    description = "Store QC results"

    def __init__(self, project: Project, issues: list[QCIssue], scores: QCScores, record: dict[str, Any], history: dict[str, Any], cache: dict[str, Any],
                 replace_run_id: str = "") -> None:
        self.project = project
        keep = sorted(issues, key=lambda i: i.sort_key)[:MAX_ISSUES]
        self.issues, self.scores, self.record, self.history, self.cache = deepcopy(keep), deepcopy(scores), clean_json(deepcopy(record)), clean_json(deepcopy(history)), clean_json(deepcopy(cache))
        self.replace_run_id = replace_run_id  # a retry of one checker updates its own run in place
        self._old: dict[str, Any] = {}

    def do(self) -> None:
        p = self.project
        self._old = {"issues": p.qc_issues, "scores": p.qc_scores, "runs": list(p.qc_runs), "history": list(p.qc_history), "cache": p.qc_cache}
        p.qc_issues = deepcopy(self.issues)
        p.qc_scores = deepcopy(self.scores)
        p.qc_cache = deepcopy(self.cache)
        runs = [r for r in p.qc_runs if r.get("run_id") != self.replace_run_id] if self.replace_run_id else list(p.qc_runs)
        history = [h for h in p.qc_history if h.get("qc_run_id") != self.replace_run_id] if self.replace_run_id else list(p.qc_history)
        runs.append(deepcopy(self.record))
        history.append(deepcopy(self.history))
        p.qc_runs, p.qc_history = runs[-MAX_RUNS:], history[-MAX_HISTORY:]

    def undo(self) -> None:
        p = self.project
        p.qc_issues, p.qc_scores, p.qc_runs, p.qc_history, p.qc_cache = (self._old["issues"], self._old["scores"], self._old["runs"], self._old["history"], self._old["cache"])


class IgnoreIssuesCommand(Command):
    """The user keeps something QC flagged: store the ignore record and mark the matching current issues. Undoable (Unignore)."""

    scope = "qc"
    major = True
    description = "Ignore QC issue"

    def __init__(self, project: Project, record: IgnoreRecord) -> None:
        self.project, self.record = project, deepcopy(record)
        self._marked: list[tuple[str, IssueStatus, bool, str]] = []

    def do(self) -> None:
        p = self.project
        p.qc_ignored_issues.append(deepcopy(self.record))
        self._marked = []
        for i in p.qc_issues:
            if self.record.matches(i) and not i.ignored_by_user:
                self._marked.append((i.issue_id, i.status, i.ignored_by_user, i.ignore_reason))
                i.ignored_by_user = True
                i.status = IssueStatus.IGNORED
                i.ignore_reason = self.record.reason

    def undo(self) -> None:
        p = self.project
        p.qc_ignored_issues = [r for r in p.qc_ignored_issues if r.ignore_id != self.record.ignore_id]
        by_id = {i.issue_id: i for i in p.qc_issues}
        for iid, status, ign, reason in self._marked:
            if iid in by_id:
                by_id[iid].status, by_id[iid].ignored_by_user, by_id[iid].ignore_reason = status, ign, reason


class UnignoreCommand(Command):
    scope = "qc"
    major = True
    description = "Stop ignoring QC issue"

    def __init__(self, project: Project, ignore_id: str) -> None:
        self.project, self.ignore_id = project, ignore_id
        self._record: IgnoreRecord | None = None
        self._marked: list[str] = []

    def do(self) -> None:
        p = self.project
        self._record = next((r for r in p.qc_ignored_issues if r.ignore_id == self.ignore_id), None)
        if self._record is None:
            return
        p.qc_ignored_issues = [r for r in p.qc_ignored_issues if r.ignore_id != self.ignore_id]
        self._marked = []
        for i in p.qc_issues:
            if i.ignored_by_user and self._record.matches(i) and not any(r.matches(i) for r in p.qc_ignored_issues):
                self._marked.append(i.issue_id)
                i.ignored_by_user, i.ignore_reason = False, ""
                if i.status is IssueStatus.IGNORED:
                    i.status = IssueStatus.OPEN

    def undo(self) -> None:
        p = self.project
        if self._record is None:
            return
        p.qc_ignored_issues.append(self._record)
        for i in p.qc_issues:
            if i.issue_id in self._marked:
                i.ignored_by_user, i.status, i.ignore_reason = True, IssueStatus.IGNORED, self._record.reason


class MarkIssueFixedCommand(Command):
    """Part of a fix's undo step: the issue becomes FIXED and the fix is recorded. Undo makes the issue OPEN again and drops the record."""

    scope = "qc"
    major = False
    description = "Mark QC issue fixed"

    def __init__(self, project: Project, issue_id: str, record: FixRecord) -> None:
        self.project, self.issue_id, self.record = project, issue_id, deepcopy(record)
        self._old_status: IssueStatus | None = None

    def do(self) -> None:
        p = self.project
        for i in p.qc_issues:
            if i.issue_id == self.issue_id:
                self._old_status = i.status
                i.status = IssueStatus.FIXED
        p.qc_fixes.append(deepcopy(self.record))

    def undo(self) -> None:
        p = self.project
        for i in p.qc_issues:
            if i.issue_id == self.issue_id and self._old_status is not None:
                i.status = self._old_status
        p.qc_fixes = [f for f in p.qc_fixes if f.fix_id != self.record.fix_id]


class SetQCSettingsCommand(Command):
    scope = "qc"
    major = False
    description = "Change QC settings"

    def __init__(self, project: Project, new: QCSettings) -> None:
        self.project, self.new = project, deepcopy(new)
        self._old: QCSettings | None = None

    def do(self) -> None:
        self._old = self.project.qc_settings
        self.project.qc_settings = deepcopy(self.new)

    def undo(self) -> None:
        assert self._old is not None
        self.project.qc_settings = self._old


class StoreRenderQCCommand(Command):
    """Keep the result of checking a rendered file (separate from, and retained alongside, the timeline QC)."""

    scope = "qc"
    major = False
    description = "Store render QC"

    def __init__(self, project: Project, render_id: str, result: dict[str, Any]) -> None:
        self.project, self.render_id, self.result = project, render_id, deepcopy(result)
        self._old: dict[str, Any] | None = None

    def do(self) -> None:
        self._old = self.project.render_qc_results.get(self.render_id)
        self.project.render_qc_results[self.render_id] = deepcopy(self.result)

    def undo(self) -> None:
        if self._old is None:
            self.project.render_qc_results.pop(self.render_id, None)
        else:
            self.project.render_qc_results[self.render_id] = self._old
