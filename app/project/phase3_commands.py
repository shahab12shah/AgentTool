"""Commands for visual research results and decisions.

Research results (from jobs) are installed without undo history, exactly like transcripts. User decisions
(choose / approve / reject / skip / manual) are undoable. A visual assignment is the contract with the future
editing phase: scene -> visual, with who selected it and the accuracy score at the time.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.commands import Command
from app.project.project import Project
from app.research.models import (
    Candidate,
    CandidateScore,
    ResearchQuery,
    ResearchSession,
    SceneResearchState,
    VisualAssignment,
)

_KEEP = object()


class ApplyResearchCommand(Command):
    """Installs the outcome of a research job for one scene (queries, session, candidate pool, scores, state)."""

    description = "Visual research"
    scope = "research"
    major = True

    def __init__(self, project: Project, scene_id: str, queries: list[ResearchQuery], session: ResearchSession | None,
                 candidates: list[Candidate], scores: dict[str, CandidateScore], state: SceneResearchState,
                 provider_stats: dict[str, Any] | None = None) -> None:
        self.project, self.scene_id = project, scene_id
        self.queries, self.session, self.candidates = queries, session, candidates
        self.scores, self.state, self.provider_stats = scores, state, provider_stats or {}
        self._undo: tuple | None = None

    def do(self) -> None:
        p = self.project
        self._undo = (
            deepcopy({c.candidate_id: c for c in p.visual_candidates.values() if c.scene_id == self.scene_id}),
            deepcopy(p.research_status.get(self.scene_id)),
            deepcopy(p.source_metadata),
            len(p.research_sessions),
        )
        for cid in [k for k, c in p.visual_candidates.items() if c.scene_id == self.scene_id]:
            del p.visual_candidates[cid]
            p.candidate_scores.pop(cid, None)
        for q in self.queries:
            p.research_queries[q.query_id] = q
        for c in self.candidates:
            p.visual_candidates[c.candidate_id] = c
        p.candidate_scores.update(self.scores)
        if self.session:
            p.research_sessions.append(self.session)
        p.research_status[self.scene_id] = self.state
        providers = p.source_metadata.setdefault("providers", {})
        for name, stats in self.provider_stats.items():
            cur = providers.setdefault(name, {"searches": 0, "failures": 0})
            cur["searches"] += stats.get("queries", 0)
            cur["failures"] += 1 if stats.get("status") == "FAILED" else 0
            cur["last_status"], cur["last_error"] = stats.get("status"), stats.get("error", "")
            cur["last_used"] = stats.get("when", "")

    def undo(self) -> None:
        assert self._undo is not None
        old_c, old_state, old_meta, n_sessions = self._undo
        p = self.project
        for cid in [k for k, c in p.visual_candidates.items() if c.scene_id == self.scene_id]:
            del p.visual_candidates[cid]
        p.visual_candidates.update(old_c)
        if old_state is None:
            p.research_status.pop(self.scene_id, None)
        else:
            p.research_status[self.scene_id] = old_state
        p.source_metadata = old_meta
        del p.research_sessions[n_sessions:]


class SceneDecisionCommand(Command):
    """An undoable user decision on one scene: state, assignment and/or some candidates change together."""

    scope = "research"

    def __init__(self, project: Project, scene_id: str, description: str, *, state: Any = _KEEP, assignment: Any = _KEEP,
                 candidates: list[Candidate] | None = None) -> None:
        self.project, self.scene_id, self.description = project, scene_id, description
        self.state = deepcopy(state) if state is not _KEEP else _KEEP
        self.assignment = deepcopy(assignment) if assignment is not _KEEP else _KEEP
        self.candidates = deepcopy(candidates or [])
        self._old: tuple | None = None

    def do(self) -> None:
        p = self.project
        self._old = (deepcopy(p.research_status.get(self.scene_id)), deepcopy(p.visual_assignments.get(self.scene_id)),
                     {c.candidate_id: deepcopy(p.visual_candidates.get(c.candidate_id)) for c in self.candidates})
        if self.state is not _KEEP:
            p.research_status[self.scene_id] = deepcopy(self.state)
        if self.assignment is not _KEEP:
            if self.assignment is None:
                p.visual_assignments.pop(self.scene_id, None)
            else:
                p.visual_assignments[self.scene_id] = deepcopy(self.assignment)
        for c in self.candidates:
            p.visual_candidates[c.candidate_id] = deepcopy(c)

    def undo(self) -> None:
        assert self._old is not None
        state, assignment, cands = self._old
        p = self.project
        if self.state is not _KEEP:
            if state is None:
                p.research_status.pop(self.scene_id, None)
            else:
                p.research_status[self.scene_id] = state
        if self.assignment is not _KEEP:
            if assignment is None:
                p.visual_assignments.pop(self.scene_id, None)
            else:
                p.visual_assignments[self.scene_id] = assignment
        for cid, c in cands.items():
            if c is None:
                p.visual_candidates.pop(cid, None)
            else:
                p.visual_candidates[cid] = c
