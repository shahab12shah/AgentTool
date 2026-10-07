"""QCFixEngine: executes the fix recipes attached to QC issues as ONE undoable command each (or one for a whole batch).

A checker only *recommends* a fix (a ``QCFixSpec`` built with ``fix_catalog``). Nothing in a spec is trusted: at execution time the engine, on the UI thread,
  1. re-resolves every id against the live project and re-reads the values the issue was about (a changed project => "run QC again"),
  2. refuses anything the user owns or locked (track / clip / decision / scene locks, USER or SYSTEM ownership, the voice-over, the MIX lock),
  3. refuses while an AI edit or a presentation generation is running,
  4. plans the change on a COPY of the timeline and re-runs the existing validators: a fix may not add a new error-level problem, and the tracks stay sorted and overlap-free,
  5. executes ``CompositeCommand("QC fix: ...", [edit, ownership, MarkIssueFixedCommand], scope="timeline")``: Ctrl+Z restores the timeline AND the issue status,
  6. writes a safety checkpoint first for fixes that are not safe and for every batch.

Ownership follows the manual-edit rule: a fix on an AI-created clip goes through the same edit hook the timeline service uses (MarkUserEditCommand /
MarkPresentationEditCommand), so a later regeneration keeps the fix instead of silently discarding it. Ducking is the exception: like the existing ducking editor it takes
over the ``duck:<music>`` decision, not the music clip. Caption words and text are never changed by a timing fix, and audio fixes only add volume keyframes.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.commands import Command, CompositeCommand
from app.core.constants import MIN_CLIP_DURATION
from app.core.exceptions import AppError, TimelineError
from app.editing.models import Creator, DecisionType, EditingDecision, OverrideRecord
from app.editing.validator import TimelineValidator
from app.logging.logger import get_logger
from app.media.asset import AssetType
from app.media.importer import sha256_file
from app.presentation.assembly import PresState
from app.presentation.models import CaptionSettings, HighlightMode, PresentationDecision, PresentationOverride, PresentationType
from app.presentation.validator import MAX_VOLUME, PresentationValidator
from app.project.phase4_commands import MarkUserEditCommand, RecordUserDeleteCommand
from app.project.phase5_commands import MarkPresentationEditCommand, RecordPresentationDeleteCommand, SetSettingCommand
from app.project.phase8_commands import MarkIssueFixedCommand
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.qc.errors import QCError
from app.qc.fix_catalog import CATALOG
from app.qc.issue_model import FixRecord, FixRoute, IssueStatus, QCFixSpec, QCIssue
from app.rendering.commands import RelinkAssetCommand
from app.timeline.clip import KIND_CAPTION, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.keyframes import INTERPOLATIONS, Keyframe, value_at
from app.timeline.timeline import Timeline
from app.timeline.timeline_commands import SCALE_RANGE
from app.timeline.track import Track, TrackKind

_log = get_logger(__name__)

STALE = "The project changed since QC ran: run QC again."
GONE = "That element no longer exists: run QC again."
OPENS_PAGE = "Opens another page"
NEEDS_CONFIRM = "This fix changes your edit: confirm it to apply."
CHECKPOINT_LABEL = "before_qc_fix"

KEYFRAME_RANGES: dict[str, tuple[float, float]] = {"opacity": (0.0, 1.0), "reveal": (0.0, 1.0), "volume": (0.0, MAX_VOLUME), "scale": SCALE_RANGE}
CAPTION_POSITIONS = ("bottom", "center", "top", "custom")
RESTYLE_FIELDS = ("position", "custom_x", "custom_y", "keyword_highlight", "number_emphasis", "highlight_mode", "max_lines", "max_words", "uppercase", "safe_margin_left",
                  "safe_margin_right", "safe_margin_top", "safe_margin_bottom", "large_text", "high_contrast", "reading_speed", "reduced_motion", "style_id")
RESTYLE_ALIASES = {"size": "large_text", "lines": "max_lines", "caption_size": "large_text", "caption_lines": "max_lines", "caption_position": "position"}
UNSUPPORTED = {"silence.remove": "Removing silence changes the narration timing: do it in the voice-over tools, where the result can be heard."}
BATCH_LABELS = {"sync.caption": "Caption Timing", "sync": "Narration Sync", "caption": "Caption", "audio": "Audio", "timeline": "Timeline", "asset": "Asset"}
MOTION_TYPES = (DecisionType.ZOOM, DecisionType.PAN, DecisionType.KEYFRAME)
EDIT_TOL = 0.0045  # rounding slack when an issue's recorded times (3 decimals) are compared with the live clip


# ---------------------------------------------------------------------------------------------- public data
@dataclass
class FixPreview:
    """What a fix WOULD do (nothing is changed to produce it). ``before`` / ``after`` are display strings keyed by what is measured."""

    issue_id: str
    kind: str
    summary: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)
    safe: bool = False  # may be applied without a confirmation click (deterministic, small, permitted "auto")
    needs_confirmation: bool = True
    blocked_reason: str = ""  # non-empty: the fix cannot be applied (and why)
    destructive: bool = False  # removes something


@dataclass
class SkippedFix:
    issue_id: str
    code: str
    reason: str


@dataclass
class _Plan:
    kind: str
    summary: str
    before: dict[str, Any]
    after: dict[str, Any]
    changes: list[str]
    small: bool = True  # the live size of the change is within the "safe" limits of the QC settings
    destructive: bool = False


# ---------------------------------------------------------------------------------------------- small helpers
def _finite(*xs: Any) -> bool:
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    return min(max(float(v), lo), hi) if _finite(v) else default


def _s(x: float) -> str:
    return f"{x:.2f} s"


def _ms(x: float) -> str:
    return f"{x * 1000.0:+.0f} ms"


def _db(g: float) -> str:
    return "silent" if g <= 1e-6 else f"{20.0 * math.log10(g):.1f} dB"


def _owner(x: Any) -> str:
    return str(getattr(x, "value", x)).upper()


def _next_id(prefix: str, keys: Any) -> str:
    n = max((int(k.rsplit("_", 1)[-1]) for k in keys if k.rsplit("_", 1)[-1].isdigit()), default=0)
    return f"{prefix}_{n + 1:05d}"


def _short(text: Any, n: int = 28) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _protection(project: Project, track: Track | None, clip: Clip) -> str:
    """Why QC must not touch this element ("" = it may). QC reports such elements but NEVER modifies them."""
    if track is not None and track.locked:
        return f"Track “{track.name}” is locked: QC does not change it."
    if clip.locked:
        return "This element is locked by you: QC does not change it."
    owner = _owner(clip.created_by)
    role = str((clip.audio or {}).get("role") or "").upper()
    if owner == "SYSTEM" or clip.slot == "voice" or role == "VOICE" or (clip.asset_id and clip.asset_id == project.voice_over.asset_id):
        return "The voice-over and other system elements are never changed by QC."
    if owner == "USER":
        return "You created or edited this element: QC does not change it."
    if clip.scene_id and clip.scene_id in (set(project.timeline_generation.locked_scenes) | set(project.presentation_generation.locked_scenes)):
        return "The scene is locked: QC does not change it."
    for decisions in (project.editing_decisions, project.presentation_decisions):
        for d in decisions.values():
            mine = d.decision_id == clip.ai_decision_id or d.target_id == clip.id or bool(clip.slot and d.slot == clip.slot and d.scene_id == clip.scene_id)
            if mine and (d.locked or (_owner(d.created_by) == "USER" and not _qc_owned(d))):
                return "The decision behind this element is locked or was set by you: QC does not change it."
    return ""


def _qc_owned(d: Any) -> bool:
    """A ducking decision created by an accepted QC fix (and not edited by the user since): QC may adjust the ducking again, e.g. for another stretch of speech."""
    return getattr(d, "type", None) is PresentationType.DUCKING and bool(d.parameters.get("qc_fix")) and not d.parameters.get("edited") and not d.locked


# ---------------------------------------------------------------------------------------------- the working copy
class _Work:
    """The candidate state a fix (or a whole batch) is planned on: a COPY of the timeline plus the decision tables, never the live objects.

    Everything a plan touches is recorded here, so the commands can be built once at the end. ``clone`` gives a cheap trial copy: a member of a batch that fails keeps no trace.
    """

    def __init__(self, project: Project) -> None:
        self.project = project
        self.timeline: Timeline = copy.deepcopy(project.timeline)
        self.edited: dict[str, Clip | None] = {}  # clip id -> state after the fix (None = removed)
        self.flip: list[str] = []  # clips whose ownership is handed to the user (the edit hook)
        self.deleted: list[str] = []
        self.extra: list[tuple[str, Command]] = []  # (scope, command) for changes that are not clip edits: relink, caption settings
        self._ed: dict[str, EditingDecision] | None = None
        self._ao: list[OverrideRecord] | None = None
        self._pd: dict[str, PresentationDecision] | None = None
        self._po: list[PresentationOverride] | None = None

    def clone(self) -> "_Work":
        w = _Work.__new__(_Work)
        w.project = self.project
        w.timeline = copy.deepcopy(self.timeline)
        w.edited = {k: (v.snapshot() if v is not None else None) for k, v in self.edited.items()}
        w.flip, w.deleted, w.extra = list(self.flip), list(self.deleted), list(self.extra)
        w._ed, w._ao = (copy.deepcopy(self._ed), copy.deepcopy(self._ao)) if self._ed is not None else (None, None)
        w._pd, w._po = (copy.deepcopy(self._pd), copy.deepcopy(self._po)) if self._pd is not None else (None, None)
        return w

    # ---- clips
    def put(self, after: Clip, flip: bool = True) -> None:
        self.timeline.restore_clip(after)
        self.edited[after.id] = after.snapshot()
        if flip and after.id not in self.flip:
            self.flip.append(after.id)

    def remove(self, clip_id: str) -> None:
        self.timeline.detach_clip(clip_id)
        self.edited[clip_id] = None
        if clip_id not in self.deleted:
            self.deleted.append(clip_id)

    # ---- decision tables (copied on first use)
    def editing(self) -> dict[str, EditingDecision]:
        if self._ed is None:
            self._ed, self._ao = copy.deepcopy(self.project.editing_decisions), copy.deepcopy(self.project.ai_overrides)
        return self._ed

    def overrides(self) -> list[OverrideRecord]:
        self.editing()
        assert self._ao is not None
        return self._ao

    def presentation(self) -> dict[str, PresentationDecision]:
        if self._pd is None:
            self._pd, self._po = copy.deepcopy(self.project.presentation_decisions), copy.deepcopy(self.project.presentation_overrides)
        return self._pd

    def pres_overrides(self) -> list[PresentationOverride]:
        self.presentation()
        assert self._po is not None
        return self._po

    def own_editing(self, d: EditingDecision) -> EditingDecision:
        """The USER decision that replaces an AI decision (the original is kept in ``ai_overrides``), like a manual edit does."""
        if d.created_by is Creator.USER:
            return d
        table = self.editing()
        new = copy.deepcopy(d)
        new.decision_id, new.created_by, new.overrides_decision_id = _next_id("dec", table), Creator.USER, d.decision_id
        del table[d.decision_id]
        table[new.decision_id] = new
        self.overrides().append(OverrideRecord(new.decision_id, copy.deepcopy(d)))
        return new

    def own_presentation(self, d: PresentationDecision) -> PresentationDecision:
        if d.created_by is Creator.USER:
            return d
        table = self.presentation()
        new = copy.deepcopy(d)
        new.decision_id, new.created_by, new.overrides_decision_id = _next_id("pdec", table), Creator.USER, d.decision_id
        del table[d.decision_id]
        table[new.decision_id] = new
        self.pres_overrides().append(PresentationOverride(new.decision_id, copy.deepcopy(d)))
        return new

    @property
    def changes_timeline(self) -> bool:
        return bool(self.edited) or self._ed is not None or self._pd is not None


# ---------------------------------------------------------------------------------------------- the one edit command
class _EditCommand(Command):
    """Installs the planned clip states and decision tables. Atomic (a failure restores what it touched) and replayable: it applies precomputed states, never recomputes."""

    scope = "timeline"
    major = True

    def __init__(self, project: Project, description: str, work: _Work) -> None:
        self.project, self.description = project, description
        self.clips = {k: (v.snapshot() if v is not None else None) for k, v in work.edited.items()}
        self.editing = (copy.deepcopy(work._ed), copy.deepcopy(work._ao)) if work._ed is not None else None
        self.presentation = (copy.deepcopy(work._pd), copy.deepcopy(work._po)) if work._pd is not None else None
        self._before: dict[str, Any] | None = None

    def _capture(self) -> dict[str, Any]:
        p = self.project
        return {"clips": {cid: (c.snapshot() if (c := p.timeline.get_clip(cid)) is not None else None) for cid in self.clips},
                "editing": (copy.deepcopy(p.editing_decisions), copy.deepcopy(p.ai_overrides)) if self.editing else None,
                "presentation": (copy.deepcopy(p.presentation_decisions), copy.deepcopy(p.presentation_overrides)) if self.presentation else None}

    def _install(self, clips: dict[str, Clip | None], editing: Any, presentation: Any) -> None:
        p = self.project
        for cid, state in clips.items():
            if state is None:
                if p.timeline.get_clip(cid) is not None:
                    p.timeline.detach_clip(cid)
            else:
                p.timeline.restore_clip(state)
        if editing is not None:
            p.editing_decisions.clear()
            p.editing_decisions.update(copy.deepcopy(editing[0]))
            p.ai_overrides[:] = copy.deepcopy(editing[1])
        if presentation is not None:
            p.presentation_decisions.clear()
            p.presentation_decisions.update(copy.deepcopy(presentation[0]))
            p.presentation_overrides[:] = copy.deepcopy(presentation[1])

    def do(self) -> None:
        if self._before is None:
            self._before = self._capture()
        try:
            self._install(self.clips, self.editing, self.presentation)
        except Exception:
            self.undo()
            raise

    def undo(self) -> None:
        b = self._before
        if b is not None:
            self._install(b["clips"], b["editing"], b["presentation"])


# ---------------------------------------------------------------------------------------------- the engine
class QCFixEngine:
    """See the module docstring. ``services`` is any object with the attributes ``editing``, ``presentation``, ``timeline`` and ``render`` (the Workspace services); only
    ``editing.running``, ``presentation.running``, ``timeline.edit_hook`` and (to relink to a *similar* file) ``render.relink.probe`` are used, and a missing one degrades
    to the documented fallback or a clear QCError."""

    def __init__(self, projects: ProjectManager, execute: Callable[[Command], object], checkpoint: Callable[[Project, str], str], services: object) -> None:
        self._projects, self._execute, self._checkpoint, self._services = projects, execute, checkpoint, services
        self.last_skipped: list[SkippedFix] = []  # what the last batch call left out, with the reason
        self._handlers: dict[str, Callable[[_Work, QCIssue, QCFixSpec], _Plan]] = {
            "caption.retime": self._caption_retime, "caption.safe_margin": self._caption_safe_margin, "audio.duck": self._audio_duck, "clip.extend": self._clip_extend,
            "clip.remove_empty": self._clip_remove_empty, "asset.relink": self._asset_relink, "param.normalize": self._param_normalize, "motion.soften": self._motion_soften,
            "transition.shorten": self._transition_shorten, "gap.close": self._gap_close, "clip.delete": self._clip_delete, "caption.restyle": self._caption_restyle,
            "audio.level": self._audio_level}

    # ================================================================== public API
    def preview_fix(self, issue_id: str) -> FixPreview:
        """What the fix would do. Never changes anything; a fix that cannot run comes back with ``blocked_reason`` (and, for pages, "Opens another page")."""
        project = self._project()
        issue = self._issue(project, issue_id)
        spec = issue.fix
        if spec is None:
            return FixPreview(issue_id, "", "No automatic fix for this issue.", blocked_reason="There is no automatic fix for this issue.")
        info = CATALOG.get(spec.kind)
        if info is not None and info.route is not FixRoute.COMMAND:
            return FixPreview(issue_id, spec.kind, spec.summary or info.label, blocked_reason=OPENS_PAGE)
        try:
            self._guard_busy()
            plan, safe, _work = self._prepare(project, issue)
        except QCError as exc:
            return FixPreview(issue_id, spec.kind, spec.summary or (info.label if info else spec.kind), blocked_reason=exc.user_message)
        return FixPreview(issue_id, spec.kind, plan.summary, dict(plan.before), dict(plan.after), list(plan.changes), safe, not safe, "", plan.destructive)

    def can_fix(self, issue: QCIssue) -> tuple[bool, str]:
        """(possible, reason). Confirmation is a separate question: a fix that needs it is still possible."""
        try:
            project = self._project()
            live = self._issue(project, issue.issue_id)
            spec = live.fix
            if spec is None:
                return False, "There is no automatic fix for this issue."
            info = CATALOG.get(spec.kind)
            if info is not None and info.route is not FixRoute.COMMAND:
                return False, OPENS_PAGE
            self._guard_busy()
            self._prepare(project, live)
            return True, ""
        except QCError as exc:
            return False, exc.user_message

    def apply_fix(self, issue_id: str, confirmed: bool = False) -> FixRecord:
        """Apply one fix as one undo step. Raises QCError (user-safe text) when blocked, unconfirmed, stale or no longer present."""
        project = self._project()
        issue = self._issue(project, issue_id)
        self._guard_busy()
        plan, safe, work = self._prepare(project, issue)
        if not safe and not confirmed:
            raise QCError(NEEDS_CONFIRM)
        rec = self._commit(project, work, [(issue, plan, safe, bool(confirmed))], plan.summary, checkpoint=not safe, strict_checkpoint=plan.destructive)[0]
        return rec

    def apply_safe_fixes(self, issue_ids: list[str] | None = None, *, code_prefix: str | None = None) -> list[FixRecord]:
        """Apply every fix that is safe AND permitted "auto", in ONE undo step. Everything else is skipped (see ``last_skipped``)."""
        project = self._project()
        wanted = set(issue_ids) if issue_ids is not None else None
        members = [i for i in project.qc_issues if i.active and i.fix is not None and (wanted is None or i.issue_id in wanted) and (not code_prefix or i.code.startswith(code_prefix))]
        label = "Fix All " + self._batch_label(code_prefix) if code_prefix else "Fix all safe issues"
        return self._batch(project, members, confirmed=False, safe_only=True, label=label)

    def fix_similar(self, issue_id: str, confirmed: bool = False) -> list[FixRecord]:
        """The same kind of fix for every open issue of the same type, as one undo step. Each one is re-checked on its own; locked ones are skipped."""
        project = self._project()
        seed = self._issue(project, issue_id)
        if seed.fix is None:
            raise QCError("There is no automatic fix for this issue.")
        if CATALOG.get(seed.fix.kind) is not None and CATALOG[seed.fix.kind].route is not FixRoute.COMMAND:
            raise QCError("This suggestion opens another page: use its button instead.")
        members = [i for i in project.qc_issues if i.active and i.code == seed.code and i.fix is not None and i.fix.kind == seed.fix.kind]
        records = self._batch(project, members, confirmed=confirmed, safe_only=False, label=f"Fix all “{_short(seed.title, 40)}”")
        if not records and self.last_skipped:
            raise QCError("Nothing could be fixed: " + self.last_skipped[0].reason)
        return records

    # ================================================================== planning (shared by preview / apply / batch)
    def _project(self) -> Project:
        p = self._projects.current
        if p is None:
            raise QCError("Open a project first.")
        return p

    @staticmethod
    def _issue(project: Project, issue_id: str) -> QCIssue:
        issue = next((i for i in project.qc_issues if i.issue_id == issue_id), None)
        if issue is None:
            raise QCError("That issue is no longer in the QC results: run QC again.")
        if issue.status is not IssueStatus.OPEN or issue.ignored_by_user:
            raise QCError("That issue is already " + ("ignored." if issue.ignored_by_user or issue.status is IssueStatus.IGNORED else "fixed or resolved."))
        return issue

    def _guard_busy(self) -> None:
        for name, what in (("editing", "An AI edit"), ("presentation", "The captions, graphics and audio")):
            svc = getattr(self._services, name, None)
            if svc is not None and bool(getattr(svc, "running", False)):
                raise QCError(f"{what} is being generated. Wait for it to finish, then try again.")

    def _prepare(self, project: Project, issue: QCIssue, work: _Work | None = None, baseline: dict | None = None) -> tuple[_Plan, bool, _Work]:
        work = work if work is not None else _Work(project)
        plan = self._plan(work, issue)
        self._check_structure(work)
        self._validate(project, work, baseline if baseline is not None else self._errors(project, project.timeline))
        return plan, self._is_safe(project, issue, plan), work

    def _plan(self, work: _Work, issue: QCIssue) -> _Plan:
        spec = issue.fix
        if spec is None:
            raise QCError("There is no automatic fix for this issue.")
        info = CATALOG.get(spec.kind)
        if info is None:
            raise QCError("This kind of fix is not available in this version of the app.")
        if info.route is not FixRoute.COMMAND:
            raise QCError("This suggestion opens another page: use its button instead.")
        if spec.kind in UNSUPPORTED:
            raise QCError(UNSUPPORTED[spec.kind])
        handler = self._handlers.get(spec.kind)
        if handler is None:
            raise QCError("This kind of fix is not available in this version of the app.")
        if work.project.qc_settings.permission(spec.kind) == "never":
            raise QCError("This kind of fix is switched off in the QC settings.")
        try:
            return handler(work, issue, spec)
        except QCError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:  # a malformed recipe is a stale / foreign issue, never a crash
            _log.warning("QC fix %s: unusable parameters (%s)", spec.kind, exc)
            raise QCError(STALE) from exc

    @staticmethod
    def _is_safe(project: Project, issue: QCIssue, plan: _Plan) -> bool:
        spec = issue.fix
        assert spec is not None
        return bool(plan.small and CATALOG[spec.kind].safe_by_design and spec.safe and spec.intrinsic_safe and project.qc_settings.permission(spec.kind) == "auto")

    # ---- validation of the candidate
    def _errors(self, project: Project, timeline: Timeline) -> dict[tuple[str, str, str], str]:
        out: dict[tuple[str, str, str], str] = {}
        try:
            tv = TimelineValidator(project.assets, project.scenes, project.voice_over.asset_id, project.editing_decisions, project.editing_strategy, project.timeline_generation)
            for i in tv.validate(timeline):
                if i.severity == "error":
                    out[("timeline", i.code, i.clip_id or i.message)] = i.message
            state = PresState(timeline, project.presentation_decisions, project.presentation_overrides, project.presentation_plans, project.ducking_events, project.keyword_emphasis,
                              project.presentation_generation, project.caption_settings)
            for i in PresentationValidator(project, state).validate():
                if i.severity == "error":
                    out[("presentation", i.code, i.clip_id or i.message)] = i.message
        except Exception as exc:  # noqa: BLE001 - a project the validators cannot read is not one to edit blindly
            _log.warning("QC fix: validation failed", exc_info=True)
            raise QCError("The edit could not be checked before changing it: run QC again.") from exc
        return out

    def _validate(self, project: Project, work: _Work, baseline: dict[tuple[str, str, str], str]) -> None:
        if not work.changes_timeline:
            return
        new = [msg for key, msg in self._errors(project, work.timeline).items() if key not in baseline]
        if new:
            raise QCError(f"The fix would cause a new problem ({_short(new[0], 90)}), so it was not applied.")

    @staticmethod
    def _check_structure(work: _Work) -> None:
        """What Project.validate() would reject on save: bad times and same-track overlap (tolerance 1e-6) for every clip the fix touched."""
        tl = work.timeline
        for cid, after in work.edited.items():
            if after is None:
                continue
            if not _finite(after.timeline_start, after.duration) or after.duration <= 0 or after.timeline_start < -1e-6:
                raise QCError("The change would leave the element with invalid timing, so it was not applied.")
            try:
                track = tl.get_track(after.track_id)
            except TimelineError as exc:
                raise QCError(GONE) from exc
            for o in track.clips:
                if o.id != cid and after.timeline_start < o.timeline_end - 1e-6 and o.timeline_start < after.timeline_end - 1e-6:
                    raise QCError("That would overlap another clip on the same track.")

    # ---- batch
    def _batch(self, project: Project, members: list[QCIssue], *, confirmed: bool, safe_only: bool, label: str) -> list[FixRecord]:
        self.last_skipped = []
        if not members:
            return []
        self._guard_busy()
        members = sorted(members, key=lambda i: (i.start_time if i.start_time is not None else 1e12, i.issue_id))
        work, applied = _Work(project), []
        baseline = self._errors(project, project.timeline)
        for issue in members:
            trial = work.clone()
            try:
                plan, safe, trial = self._prepare(project, issue, trial, baseline)
                if not safe and (safe_only or not confirmed):
                    raise QCError("Needs your confirmation: it changes your edit.")
            except QCError as exc:
                self.last_skipped.append(SkippedFix(issue.issue_id, issue.code, exc.user_message))
                continue
            work = trial
            applied.append((issue, plan, safe, bool(confirmed and not safe)))
        if not applied:
            return []
        return self._commit(project, work, applied, label, checkpoint=True, strict_checkpoint=any(p.destructive for _i, p, _sf, _cf in applied))

    @staticmethod
    def _batch_label(prefix: str | None) -> str:
        if not prefix:
            return "Safe Issues"
        key = max((k for k in BATCH_LABELS if prefix.startswith(k)), key=len, default="")
        return BATCH_LABELS.get(key, prefix.replace(".", " ").title())

    # ---- execution
    def _commit(self, project: Project, work: _Work, applied: list[tuple[QCIssue, _Plan, bool, bool]], description: str, *, checkpoint: bool, strict_checkpoint: bool) -> list[FixRecord]:
        name = ""
        if checkpoint:
            try:
                name = self._checkpoint(project, CHECKPOINT_LABEL) or ""
            except Exception as exc:  # noqa: BLE001 - a missing safety copy only blocks a fix that removes something
                _log.warning("QC fix: checkpoint failed: %s", exc)
                if strict_checkpoint:
                    raise QCError("A safety copy of the project could not be written, so nothing was removed.") from exc
        records = [FixRecord(f"qcf_{uuid.uuid4().hex[:10]}", i.issue_id, i.code, i.fix.kind if i.fix else "", i.fingerprint, i.scene_id, p.summary, dict(p.before), dict(p.after), safe,
                             confirmed, run_id=i.run_id, checkpoint=name) for i, p, safe, confirmed in applied]
        commands = self._commands(project, work, f"QC fix: {description}")
        commands += [MarkIssueFixedCommand(project, rec.issue_id, rec) for rec in records]
        kinds = {p.kind for _i, p, _sf, _cf in applied}
        scope = "assets" if kinds == {"asset.relink"} else "editing" if kinds == {"caption.restyle"} else "timeline"
        try:
            self._execute(CompositeCommand(f"QC fix: {description}", commands, scope=scope))
        except QCError:
            raise
        except AppError as exc:
            raise QCError(exc.user_message) from exc
        except Exception as exc:  # noqa: BLE001 - CompositeCommand has rolled everything back
            _log.exception("QC fix could not be applied")
            raise QCError("The fix could not be applied; nothing was changed.") from exc
        return records

    def _commands(self, project: Project, work: _Work, description: str) -> list[Command]:
        cmds: list[Command] = []
        if work.changes_timeline:
            cmds.append(_EditCommand(project, description, work))
        cmds += [c for _scope, c in work.extra]
        for cid in work.flip:
            if cid not in work.deleted and work.edited.get(cid) is not None:
                cmd = self._ownership(project, cid, "edit")
                if cmd is not None:
                    cmds.append(cmd)
        for cid in work.deleted:
            cmd = self._ownership(project, cid, "delete")  # built while the clip still exists: it records the slot and the decisions
            if cmd is not None:
                cmds.append(cmd)
        return cmds

    def _ownership(self, project: Project, clip_id: str, action: str) -> Command | None:
        """The command that hands an AI element to the user (the same one a manual edit runs). Uses the timeline service's edit hook, or its rule when there is none."""
        hook = getattr(getattr(self._services, "timeline", None), "edit_hook", None)
        if hook is not None:
            return hook(clip_id, action)
        clip = project.timeline.get_clip(clip_id)
        if clip is None:
            return None
        cmds: list[Command] = []
        if clip.scene_id or clip.ai_decision_id:
            cmds.append(RecordUserDeleteCommand(project, clip_id) if action == "delete" else MarkUserEditCommand(project, clip_id))
        if clip.metadata.get("phase") == 5 or clip.metadata.get("phase5") or any(d.target_id == clip_id for d in project.presentation_decisions.values()):
            cmds.append(RecordPresentationDeleteCommand(project, clip_id) if action == "delete" else MarkPresentationEditCommand(project, clip_id))
        if not cmds:
            return None
        return cmds[0] if len(cmds) == 1 else CompositeCommand("Take ownership", cmds, scope="timeline")

    # ================================================================== resolving elements
    @staticmethod
    def _clip(work: _Work, clip_id: str, kinds: tuple[str, ...] | None = None) -> tuple[Track, Clip, Clip]:
        """(live track, live clip, working clip) for an id the issue carries, after the lock / ownership checks. QC reports protected elements but never modifies them."""
        project = work.project
        try:
            track, live = project.timeline.find_clip(clip_id)
        except TimelineError as exc:
            raise QCError(GONE) from exc
        cur = work.timeline.get_clip(clip_id)
        if cur is None:
            raise QCError("That element was already changed or removed by another fix in this batch.")
        if kinds is not None and live.kind not in kinds:
            raise QCError(STALE)
        reason = _protection(project, track, live)
        if reason:
            raise QCError(reason)
        return track, live, cur

    @staticmethod
    def _free(work: _Work, after: Clip) -> None:
        try:
            work.timeline.check_free(after.track_id, after.timeline_start, after.duration, ignore_clip_id=after.id)
        except TimelineError as exc:
            raise QCError(exc.user_message) from exc

    @staticmethod
    def _pres_decisions_of(work: _Work, clip_id: str, *types: PresentationType) -> list[PresentationDecision]:
        return [d for d in work.presentation().values() if d.target_id == clip_id and (not types or d.type in types)]

    # ================================================================== handlers: captions
    def _caption_retime(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        p = spec.params
        cid, ns, ne, delta = str(p["clip_id"]), float(p["new_start"]), float(p["new_end"]), float(p.get("delta", 0.0))
        _track, live, cur = self._clip(work, cid, (KIND_CAPTION,))
        if abs(live.timeline_start - (ns - delta)) > EDIT_TOL:
            raise QCError(STALE)  # the caption is no longer where QC measured it
        if abs(ns - live.timeline_start) < 0.0005:
            raise QCError("The caption already starts at that time.")
        if ne - ns < MIN_CLIP_DURATION or not _finite(ns, ne):
            raise QCError("The new caption window would be too short.")
        after = cur.snapshot()
        after.timeline_start, after.duration = round(ns, 4), round(ne - ns, 4)
        self._free(work, after)
        text = copy.deepcopy(after.text or {})  # only the window moves: the words, their timing and the text are never touched
        text["start"], text["end"] = after.timeline_start, round(after.timeline_end, 4)
        text["reading_cps"] = round(len(str(text.get("text", ""))) / max(after.duration, 0.25), 2)
        after.text = text
        for d in self._pres_decisions_of(work, cid, PresentationType.CAPTION):
            d.parameters["reading_cps"] = text["reading_cps"]
        work.put(after)
        shift = abs(after.timeline_start - live.timeline_start)
        small = shift <= work.project.qc_settings.max_caption_shift_seconds + 1e-9
        words = len((live.text or {}).get("words") or [])
        return _Plan("caption.retime", spec.summary or "Retime the caption",
                     {"caption start": _s(live.timeline_start), "caption end": _s(live.timeline_end), "drift from the spoken start": _ms(live.timeline_start - ns)},
                     {"caption start": _s(after.timeline_start), "caption end": _s(after.timeline_end), "drift from the spoken start": _ms(after.timeline_start - ns)},
                     [f"The caption “{_short((live.text or {}).get('text'))}” moves {_s(shift)} {'earlier' if after.timeline_start < live.timeline_start else 'later'}.",
                      f"Its text and the timing of its {words} words are not changed."], small)

    def _caption_safe_margin(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid, xy = str(spec.params["clip_id"]), spec.params["position_xy"]
        _track, live, cur = self._clip(work, cid, (KIND_CAPTION,))
        cs = work.project.caption_settings
        x = _clamp(xy[0], cs.safe_margin_left, 1.0 - cs.safe_margin_right, 0.5)
        y = _clamp(xy[1], cs.safe_margin_top, 1.0 - cs.safe_margin_bottom, 0.85)
        old = copy.deepcopy(live.text or {})
        old_xy = list(old.get("position_xy") or [])
        if old.get("position") == "custom" and len(old_xy) == 2 and abs(old_xy[0] - x) < 1e-4 and abs(old_xy[1] - y) < 1e-4:
            raise QCError(STALE)
        after = cur.snapshot()
        text = copy.deepcopy(after.text or {})
        text["position"], text["position_xy"] = "custom", [round(x, 4), round(y, 4)]
        after.text = text
        work.put(after)
        shown = f"x {old_xy[0]:.2f}, y {old_xy[1]:.2f}" if len(old_xy) == 2 else f"“{old.get('position', 'bottom')}” position"
        return _Plan("caption.safe_margin", spec.summary or "Move the caption inside the safe area", {"caption position": shown}, {"caption position": f"x {x:.2f}, y {y:.2f} (inside the safe area)"},
                     [f"The caption “{_short(old.get('text'))}” is nudged inside the safe margins; its text is not changed."])

    def _caption_restyle(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        name = RESTYLE_ALIASES.get(str(spec.params["field"]), str(spec.params["field"]))
        if name not in RESTYLE_FIELDS:
            raise QCError("That caption setting cannot be changed by QC.")
        cs = work.project.caption_settings
        if name in cs.user_set:
            raise QCError("You chose this caption setting yourself: QC does not change it.")
        cur = getattr(cs, name)
        raw = spec.params["value"]
        if isinstance(cur, bool):
            if isinstance(raw, str):
                raw = raw.strip().lower() in ("1", "true", "yes", "on")
            value: Any = bool(raw)
        elif isinstance(cur, (int, float)):
            if not _finite(raw):
                raise QCError("That is not a valid value for this setting.")
            value = type(cur)(raw)
        else:
            value = str(raw)
        if value == cur:
            raise QCError(STALE)
        new = copy.deepcopy(cs)
        setattr(new, name, value)
        problem = self._caption_settings_problem(work.project, new)
        if problem:
            raise QCError(problem)
        if name not in new.user_set:
            new.user_set.append(name)  # a later reference style must not override a setting the user confirmed
        work.extra.append(("editing", SetSettingCommand(work.project, "caption_settings", new, "Change caption settings")))
        return _Plan("caption.restyle", spec.summary or "Change a caption setting", {name: cur}, {name: value},
                     ["Applies to captions created from now on; existing captions are not laid out again."])

    @staticmethod
    def _caption_settings_problem(project: Project, new: CaptionSettings) -> str:
        from app.captions.styles import PRESETS  # noqa: PLC0415 - the preset table is only needed here

        if new.position not in CAPTION_POSITIONS:
            return f"Unknown caption position “{new.position}”."
        if new.style_id not in {**PRESETS, **project.caption_styles}:
            return f"Unknown caption style “{new.style_id}”."
        if new.highlight_mode not in {m.value for m in HighlightMode}:
            return f"Unknown highlight mode “{new.highlight_mode}”."
        if new.max_lines not in (1, 2):
            return "Captions can have 1 or 2 lines."
        if any(not (0.0 <= getattr(new, k) <= 0.4) for k in ("safe_margin_left", "safe_margin_right", "safe_margin_top", "safe_margin_bottom")):
            return "Safe margins must be between 0% and 40%."
        if new.safe_margin_left + new.safe_margin_right >= 0.8:
            return "The safe margins leave no room for text."
        if not (0.3 <= new.reading_speed <= 2.0):
            return "Reading speed must be between 0.3 and 2.0."
        return ""

    # ================================================================== handlers: visuals (extend / close / remove / delete)
    def _extend(self, work: _Work, cid: str, want_end: float, *, strict: bool, kind: str, summary: str, extra_before: dict[str, Any] | None = None) -> _Plan:
        project = work.project
        track, live, cur = self._clip(work, cid, (KIND_MEDIA,))
        if track.is_audio:
            raise QCError("Only pictures and video are extended by QC.")
        if not _finite(want_end) or want_end <= live.timeline_end + 0.001:
            raise QCError(STALE)  # nothing left to extend: the clip already reaches that far
        asset = project.assets.get(cur.asset_id) if cur.asset_id in project.assets else None
        if asset is None:
            raise QCError("The clip's media is missing, so it cannot be extended.")
        max_source = None if asset.type is AssetType.IMAGE or not asset.duration else float(asset.duration)
        r = work.timeline.compute_trim(cur, new_end=want_end, max_source=max_source)
        new_end = r.start + r.duration
        grown = new_end - cur.timeline_end
        short = want_end - new_end
        if grown < 0.005 or (strict and short > 0.001):
            _prev, nxt = work.timeline.neighbours(cur)
            src_limit = cur.timeline_start + (max_source - cur.source_in) / cur.speed if max_source is not None else math.inf
            why = "the source media is too short" if src_limit <= (nxt if nxt is not None else math.inf) + 1e-6 else "another clip on the track is in the way"
            raise QCError(f"The clip cannot be extended by {_s(want_end - cur.timeline_end)}: {why}.")
        after = cur.snapshot()
        after.duration, after.source_in, after.source_out = round(r.duration, 4), r.source_in, r.source_out
        work.put(after)
        limit = work.project.qc_settings.max_clip_extension_seconds
        lines = [f"The clip is extended by {_s(grown)} (it now ends at {_s(after.timeline_end)})."]
        if short > 0.001:
            lines.append(f"Limited to {_s(grown)} by the source media or the free space ({_s(want_end - cur.timeline_end)} was asked for).")
        before = {"clip end": _s(live.timeline_end), "clip duration": _s(live.duration), **(extra_before or {})}
        after_v = {"clip end": _s(after.timeline_end), "clip duration": _s(after.duration)}
        if extra_before and "gap" in extra_before:
            after_v["gap"] = _s(max(0.0, want_end - after.timeline_end))
        return _Plan(kind, summary, before, after_v, lines, grown <= limit + 1e-9)

    def _clip_extend(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        return self._extend(work, str(spec.params["clip_id"]), float(spec.params["new_end"]), strict=False, kind="clip.extend", summary=spec.summary or "Extend the clip")

    def _gap_close(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        project = work.project
        cid, new_end = str(spec.params["clip_id"]), float(spec.params["new_end"])
        _track, live, _cur = self._clip(work, cid, (KIND_MEDIA,))
        tol = max(0.05, 2.0 / max(1, int(project.settings.fps or 30)))
        if issue.start_time is not None and abs(live.timeline_end - issue.start_time) > tol:
            raise QCError(STALE)  # the clip before the gap is no longer the one that ends where the gap starts
        for t in project.timeline.tracks:  # the gap must still be a gap: nothing visible may already cover it
            if t.kind in (TrackKind.VIDEO, TrackKind.IMAGE) and not t.hidden:
                if any(o.id != cid and o.kind == KIND_MEDIA and o.timeline_start < new_end - tol and o.timeline_end > live.timeline_end + tol for o in t.clips):
                    raise QCError(STALE)
        gap = max(0.0, new_end - live.timeline_end)
        plan = self._extend(work, cid, new_end, strict=True, kind="gap.close", summary=spec.summary or "Close the gap", extra_before={"gap": _s(gap)})
        plan.small = False  # closing a gap is a creative change: always confirmed
        return plan

    def _clip_remove_empty(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid = str(spec.params["clip_id"])
        track, live, _cur = self._clip(work, cid)
        why = self._emptiness(work.project, live)
        if not why:
            raise QCError(STALE)  # it has content now: not an empty item any more
        work.remove(cid)
        return _Plan("clip.remove_empty", spec.summary or "Remove the empty item", {"item": f"{live.kind} on {track.name}", "content": why}, {"item": "removed"},
                     [f"The empty {live.kind} item ({why}) is removed from {track.name}."], True, True)

    @staticmethod
    def _emptiness(project: Project, c: Clip) -> str:
        if not _finite(c.duration) or c.duration < MIN_CLIP_DURATION - 1e-9:
            return f"lasts {c.duration:.3f} s"
        if c.kind == KIND_MEDIA and not c.asset_id:
            return "no media attached"
        if c.kind == KIND_TEXT and not (isinstance(c.text, dict) and str(c.text.get("content", "")).strip()):
            return "no text"
        if c.kind == KIND_CAPTION and not (isinstance(c.text, dict) and str(c.text.get("text", "")).strip()):
            return "no text"
        return ""

    def _clip_delete(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid = str(spec.params["clip_id"])
        track, live, _cur = self._clip(work, cid)
        if "duplicate" in issue.code and not self._has_twin(live, track):
            raise QCError(STALE)  # the copy it duplicated is gone: removing this one would remove the only one
        work.remove(cid)
        name = _short((live.text or {}).get("content") or (live.text or {}).get("text") or live.slot or live.kind)
        return _Plan("clip.delete", spec.summary or "Delete the element", {"element": f"{live.kind} “{name}” on {track.name}", "time": f"{_s(live.timeline_start)} to {_s(live.timeline_end)}"},
                     {"element": "deleted"}, [f"“{name}” is removed from {track.name}; Undo brings it back."], False, True)

    @staticmethod
    def _has_twin(c: Clip, track: Track) -> bool:
        def key(x: Clip) -> tuple:
            return (x.kind, x.asset_id, round(x.timeline_start, 3), round(x.duration, 3), round(x.source_in, 3), str((x.text or {}).get("content") or (x.text or {}).get("text") or ""))

        return any(o.id != c.id and key(o) == key(c) for o in track.clips)

    # ================================================================== handlers: assets and parameters
    def _asset_relink(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        project = work.project
        aid, path, exact = str(spec.params["asset_id"]), Path(str(spec.params["new_path"])), bool(spec.params.get("exact"))
        asset = project.assets.get(aid) if aid in project.assets else None
        if asset is None:
            raise QCError(GONE)
        if project.asset_path(asset).is_file():
            raise QCError(STALE)  # the file is back: nothing to relink
        if any(isinstance(c, RelinkAssetCommand) and c.asset_id == aid for _s_, c in work.extra):
            raise QCError("This asset is already being relinked.")
        if not path.is_file():
            raise QCError("The replacement file no longer exists.")
        from app.rendering.relink import EXT_OF  # noqa: PLC0415 - keeps the render layer out of QC's import graph

        if path.suffix.lower() not in EXT_OF[asset.type]:
            raise QCError(f"A {asset.type.value} file is needed for this asset.")
        size = path.stat().st_size
        facts: dict[str, Any] = {"size_bytes": size}
        if exact:
            if not asset.content_hash:
                raise QCError("The original file's fingerprint is unknown, so an identical copy cannot be confirmed.")
            if asset.size_bytes and size != asset.size_bytes:
                raise QCError("That file is not an identical copy (different size).")
            try:
                same = sha256_file(path) == asset.content_hash
            except OSError as exc:
                raise QCError("The replacement file could not be read.") from exc
            if not same:
                raise QCError("That file is not an identical copy (different content).")
            # identical bytes: every stored media fact (duration, size in pixels, ...) stays valid
        else:
            probe = getattr(getattr(getattr(self._services, "render", None), "relink", None), "probe", None)
            if probe is None:
                raise QCError("Relinking to a similar file needs the media tools, which are not available here.")
            info, err = probe.try_probe(path)
            if info is None:
                raise QCError(f"That file cannot be used: {err}")
            rotated = info.rotation in (90, 270)
            facts |= {"duration": info.duration, "width": info.coded_width if rotated else info.width, "height": info.coded_height if rotated else info.height, "fps": info.fps,
                      "codec": info.codec, "has_audio": info.has_audio, "audio_codec": info.audio_codec, "sample_rate": info.sample_rate, "channels": info.channels}
            try:
                facts["content_hash"] = sha256_file(path)
            except OSError as exc:
                raise QCError("The replacement file could not be read.") from exc
        work.extra.append(("assets", RelinkAssetCommand(project, aid, path, facts)))
        return _Plan("asset.relink", spec.summary or "Relink the media", {"file": f"{asset.name} (missing)"}, {"file": f"{path.name} ({'identical copy' if exact else 'similar file'})"},
                     [f"“{asset.name}” is pointed at {path}; every clip that uses it stays where it is."])

    def _param_normalize(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid, changes = str(spec.params["clip_id"]), dict(spec.params.get("changes") or {})
        _track, live, cur = self._clip(work, cid)
        after = cur.snapshot()
        before: dict[str, Any] = {}
        result: dict[str, Any] = {}

        def note(label: str, old: Any, new: Any) -> None:
            before[label], result[label] = (f"{old:g}" if _finite(old) else repr(old)), f"{new:g}"

        for key, want in changes.items():
            if key == "scale":
                if _finite(live.scale) and SCALE_RANGE[0] <= live.scale <= SCALE_RANGE[1]:
                    continue
                after.scale = _clamp(want, *SCALE_RANGE, 1.0)
                note("scale", live.scale, after.scale)
            elif key == "opacity":
                if _finite(live.opacity) and 0.0 <= live.opacity <= 1.0:
                    continue
                after.opacity = _clamp(want, 0.0, 1.0, 1.0)
                note("opacity", live.opacity, after.opacity)
            elif key == "audio.volume":
                vol = (live.audio or {}).get("volume")
                if vol is None or (_finite(vol) and 0.0 <= vol <= MAX_VOLUME):
                    continue
                after.audio["volume"] = _clamp(want, 0.0, MAX_VOLUME, 1.0)
                note("volume", vol, after.audio["volume"])
            elif key == "keyframes":
                for item in want:
                    prop, t = str(item["property"]), float(item["time"])
                    rng = KEYFRAME_RANGES.get(prop)
                    if rng is None:
                        continue
                    for k in (k for k in after.keyframes if k.property == prop and abs(k.time - t) <= 0.002):
                        bad = not _finite(k.value) or (k.value <= 0 or k.value > rng[1] if prop == "scale" else not rng[0] <= k.value <= rng[1])
                        if bad:
                            old = k.value
                            k.value = round(_clamp(item.get("value"), *rng, DEFAULT_KF.get(prop, 1.0)), 4)
                            note(f"{prop} keyframe at {t:.2f} s", old, k.value)
            else:
                raise QCError("That kind of correction is not available.")
        if not result:
            raise QCError(STALE)  # every value is already inside its valid range
        work.put(after)
        return _Plan("param.normalize", spec.summary or "Bring the values back into range", before, result,
                     [f"{k}: {before[k]} becomes {result[k]}." for k in result])

    # ================================================================== handlers: motion and transitions
    def _motion_soften(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid = str(spec.params["clip_id"])
        _track, live, cur = self._clip(work, cid)
        new: list[Keyframe] = []
        for item in spec.params["keyframes"]:
            prop, interp = str(item["property"]), str(item.get("interpolation") or "linear")
            kf = Keyframe(prop, round(float(item["time"]), 3), round(float(item["value"]), 4), interp if interp in INTERPOLATIONS else "linear", "")
            rng = KEYFRAME_RANGES.get(prop)
            if kf.problems(cur.duration) or (rng is not None and not rng[0] <= kf.value <= rng[1]):
                raise QCError(STALE)  # the recommended motion no longer fits this clip
            new.append(kf)
        props = {k.property for k in new}
        old = sorted((k for k in live.keyframes if k.property in props), key=lambda k: (k.property, k.time))
        if [(k.property, round(k.time, 3), round(k.value, 4)) for k in old] == [(k.property, k.time, k.value) for k in sorted(new, key=lambda k: (k.property, k.time))]:
            raise QCError(STALE)  # already softened
        after = cur.snapshot()
        old_ids = {k.decision_id for k in after.keyframes if k.property in props and k.decision_id}
        owner_id = ""
        for d in [d for d in work.editing().values() if d.target_id == cid and d.type in MOTION_TYPES]:
            d2 = work.own_editing(d)
            scale = sorted((k for k in new if k.property == "scale"), key=lambda k: k.time)
            if scale and "end_scale" in d2.parameters:
                d2.parameters.update(start_scale=scale[0].value, end_scale=scale[-1].value)
            if d.decision_id in old_ids:
                owner_id = d2.decision_id
            if after.ai_decision_id == d.decision_id:
                after.ai_decision_id = d2.decision_id
        for k in new:
            k.decision_id = owner_id
        after.keyframes = [k for k in after.keyframes if k.property not in props] + sorted(new, key=lambda k: (k.property, k.time))
        work.put(after)

        def describe(kfs: list[Keyframe]) -> str:
            s = sorted((k for k in kfs if k.property == "scale"), key=lambda k: k.time)
            return f"scale {s[0].value:.2f} to {s[-1].value:.2f} over {s[-1].time - s[0].time:.1f} s" if len(s) >= 2 else f"{len(kfs)} keyframe(s) on {', '.join(sorted(props))}"

        return _Plan("motion.soften", spec.summary or "Soften the movement", {"movement": describe(old)}, {"movement": describe(new)},
                     [f"The movement on this clip is replaced by a gentler one ({describe(new)}); its other keyframes stay."])

    def _transition_shorten(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid, want = str(spec.params["clip_id"]), float(spec.params["duration"])
        _track, live, cur = self._clip(work, cid)
        tr = live.transition
        if not tr or not _finite(tr.get("duration"), want) or float(tr["duration"]) <= want + 0.001:
            raise QCError(STALE)  # no transition, or it is already that short
        new = round(_clamp(want, 0.0, cur.duration, 0.0), 3)
        after = cur.snapshot()
        tr2 = dict(after.transition or {})
        tr2["duration"] = new
        did = tr2.get("decision_id")
        d = work.editing().get(did) if did else None
        if d is not None:
            d2 = work.own_editing(d)
            d2.parameters["duration"] = new
            tr2["decision_id"] = d2.decision_id
            if after.ai_decision_id == d.decision_id:
                after.ai_decision_id = d2.decision_id
        after.transition = tr2
        work.put(after)
        return _Plan("transition.shorten", spec.summary or "Shorten the transition", {"transition": f"{tr.get('type', 'transition')}, {_s(float(tr['duration']))}"},
                     {"transition": f"{tr.get('type', 'transition')}, {_s(new)}"}, [f"The {str(tr.get('type', '')).lower()} into this clip becomes {_s(new)} long."])

    # ================================================================== handlers: audio
    def _audio_level(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        cid, want = str(spec.params["clip_id"]), spec.params["volume"]
        track, live, cur = self._clip(work, cid, (KIND_MEDIA,))
        if not track.is_audio or not _finite(want):
            raise QCError(STALE)
        old = float((live.audio or {}).get("volume", 1.0))
        new = round(_clamp(want, 0.0, MAX_VOLUME, 1.0), 4)
        if abs(old - new) < 1e-4:
            raise QCError(STALE)
        after = cur.snapshot()
        after.audio["volume"] = new
        for d in self._pres_decisions_of(work, cid):
            if "volume" in d.parameters or d.type in (PresentationType.MUSIC, PresentationType.SFX):
                d.parameters["volume"] = new
        work.put(after)
        return _Plan("audio.level", spec.summary or "Change the level", {"clip volume": f"{old:.2f} ({_db(old)})"}, {"clip volume": f"{new:.2f} ({_db(new)})"},
                     [f"The level of this {str((live.audio or {}).get('role', 'audio')).lower()} clip changes from {_db(old)} to {_db(new)}; the audio file is not changed."])

    def _audio_duck(self, work: _Work, issue: QCIssue, spec: QCFixSpec) -> _Plan:
        project = work.project
        p = spec.params
        role, aid, gain = str(p.get("role", "MUSIC")).upper(), str(p.get("assignment_id", "")), p.get("target_gain")
        if role not in ("MUSIC", "SFX"):
            raise QCError("Only music and sound effects are lowered under the voice.")
        if not _finite(gain) or not 0.0 <= float(gain) <= MAX_VOLUME:
            raise QCError(STALE)
        gain = float(gain)
        spans = _merge_spans([(float(a), float(b)) for a, b in p.get("spans") or [] if _finite(a, b) and b > a])
        if not spans:
            raise QCError(STALE)
        pieces: list[tuple[Track, Clip]] = []
        for t in project.timeline.tracks:
            if t.kind is TrackKind.AUDIO:
                pieces += [(t, c) for c in t.clips if c.kind == KIND_MEDIA and str((c.audio or {}).get("role") or "").upper() == role and _matches(c, aid)]
        pieces = [(t, c) for t, c in pieces if any(a < c.timeline_end and b > c.timeline_start for a, b in spans)]
        if not pieces:
            raise QCError(STALE)
        for t, c in pieces:
            reason = _protection(project, t, c)
            if reason:
                raise QCError(reason)
        slot = f"duck:{aid}"
        existing = [d for d in project.presentation_decisions.values() if d.slot == slot]
        if role == "MUSIC":
            for d in existing:
                if d.locked or d.parameters.get("edited") or (_owner(d.created_by) == "USER" and not d.parameters.get("qc_fix")):
                    raise QCError("The ducking of this music is locked or was set by you: QC does not change it.")
        att, rel = float(getattr(project.audio_settings, "attack", 0.25)), float(getattr(project.audio_settings, "release", 0.5))
        worst_before, new_clips = 0.0, []
        decision_id = ""
        if role == "MUSIC":
            decision_id = self._duck_decision(work, aid, slot, sorted((c for _t, c in pieces), key=lambda c: c.timeline_start)[0].id)
        for t, live in pieces:
            cur = work.timeline.get_clip(live.id)
            if cur is None:
                raise QCError("That element was already changed or removed by another fix in this batch.")
            base = float((live.audio or {}).get("volume", 1.0)) * float(t.volume)
            if base <= 1e-6:
                continue  # silent already
            local = [(max(a, cur.timeline_start) - cur.timeline_start, min(b, cur.timeline_end) - cur.timeline_start) for a, b in spans if a < cur.timeline_end and b > cur.timeline_start]
            worst_before = max(worst_before, _worst_gain(live, base, local))
            vol_old = sorted((k for k in cur.keyframes if k.property == "volume"), key=lambda k: k.time)
            pts = _duck_points(vol_old, cur.duration, local, min(MAX_VOLUME, gain / base), att, rel)
            after = cur.snapshot()
            after.keyframes = [k for k in after.keyframes if k.property != "volume"] + [Keyframe("volume", round(tt, 3), round(v, 4), "linear", decision_id) for tt, v in pts]
            if decision_id:
                old_ids = {d.decision_id for d in existing}
                after.ai_decision_id = decision_id if (not after.ai_decision_id or after.ai_decision_id in old_ids) else after.ai_decision_id
            if any(k.problems(after.duration) for k in after.keyframes):
                raise QCError("The ducking would not fit this clip: run QC again.")
            if _worst_gain(after, base, local) > gain + 0.002:
                raise QCError("The music could not be lowered far enough with the available keyframes.")
            new_clips.append(after)
        if not new_clips or worst_before <= gain + 0.002:
            raise QCError(STALE)  # already under the target level
        for after in new_clips:
            work.put(after, flip=(role == "SFX"))  # music: the ducking decision changes hands (like the ducking editor); a sound effect is one clip of its own
        total = sum(b - a for a, b in spans)
        return _Plan("audio.duck", spec.summary or "Lower the music under the voice",
                     {f"{role.lower()} level under the voice": _db(worst_before), "ducking": "too little"}, {f"{role.lower()} level under the voice": _db(gain), "ducking": "volume keyframes added"},
                     [f"{len(spans)} stretch(es) of speech ({_s(total)}) get a smooth dip to {_db(gain)} with {att:.2f} s attack and {rel:.2f} s release.",
                      "Only volume keyframes are added: the audio file and the clip are not changed."])

    def _duck_decision(self, work: _Work, aid: str, slot: str, first_clip_id: str) -> str:
        """The USER-owned ducking decision the new keyframes belong to (created when the music had none, taken over from the AI otherwise): regeneration then leaves it alone."""
        table = work.presentation()
        existing = next((d for d in table.values() if d.slot == slot), None)
        if existing is None:
            d = PresentationDecision(_next_id("pdec", table), "", PresentationType.DUCKING, slot, first_clip_id, 0.0, 0.0, {"assignment_id": aid, "qc_fix": True},
                                     "Music lowered under the voice by an accepted QC fix.", 100.0, Creator.USER)
            table[d.decision_id] = d
            return d.decision_id
        d = work.own_presentation(existing)
        d.parameters.update(qc_fix=True)
        d.reason, d.confidence = "Music lowered under the voice by an accepted QC fix.", 100.0
        return d.decision_id


DEFAULT_KF = {"opacity": 1.0, "reveal": 1.0, "volume": 1.0, "scale": 1.0}


# ---------------------------------------------------------------------------------------------- ducking maths
def _matches(c: Clip, assignment_id: str) -> bool:
    return bool(assignment_id) and assignment_id in (c.metadata.get("assignment_id"), c.metadata.get("sfx_id"), c.id)


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _worst_gain(c: Clip, base: float, spans: list[tuple[float, float]]) -> float:
    """The loudest effective gain (clip volume x track volume x volume keyframes) the clip has inside the spans (clip-local seconds)."""
    ks = [k for k in c.keyframes if k.property == "volume"]
    worst = 0.0
    for a, b in spans:
        step = max(0.05, (b - a) / 400.0)
        n = max(1, int(math.ceil((b - a) / step)))
        worst = max([worst] + [base * value_at(ks, "volume", a + (b - a) * i / n) for i in range(n + 1)])
    return worst


def _duck_points(old: list[Keyframe], dur: float, spans: list[tuple[float, float]], m: float, attack: float, release: float) -> list[tuple[float, float]]:
    """New volume keyframes (clip-local time, multiplier): the old curve, dipped to ``m`` inside each span with an attack ramp before and a release ramp after.

    The result is min(old curve, ducking curve) sampled at every breakpoint: at each of them the value is at most ``m`` inside a span, and between breakpoints it is a straight
    line between two such values, so the gain inside a span never exceeds the target.
    """
    def orig(t: float) -> float:
        return value_at(old, "volume", t)

    windows = _merge_spans([(max(0.0, a - attack), min(dur, b + release)) for a, b in spans])
    # every span belongs to the window that contains it; windows that touch have been merged, so the spans merged with them
    pts: dict[float, float] = {round(k.time, 3): k.value for k in old}
    for w0, w1 in windows:
        inside = [(max(a, w0), min(b, w1)) for a, b in spans if a < w1 and b > w0]
        la, lb = min(a for a, _b in inside), max(b for _a, b in inside)
        v0, v1 = orig(w0), orig(w1)
        times = sorted({round(w0, 3), round(la, 3), round(lb, 3), round(w1, 3)} | {round(k.time, 3) for k in old if w0 < k.time < w1})
        for t in [t for t in pts if w0 - 1e-6 <= t <= w1 + 1e-6]:
            del pts[t]
        for t in times:
            if la <= t <= lb:
                duck = m
            elif t < la:
                duck = v0 + (m - v0) * ((t - w0) / (la - w0)) if la - w0 > 1e-9 else m
            else:
                duck = m + (v1 - m) * ((t - lb) / (w1 - lb)) if w1 - lb > 1e-9 else v1
            pts[t] = min(orig(t), duck)
    return sorted((min(max(t, 0.0), dur), v) for t, v in pts.items())


__all__ = ["QCFixEngine", "FixPreview", "SkippedFix", "STALE", "OPENS_PAGE"]
