"""Shared fixtures for the editorial intelligence tests: a synthetic project (real scenes, intents, assignments, candidates, timeline) opened in a real Workspace."""

from __future__ import annotations

import pytest

from app.performance.synthetic import SyntheticSpec, build_project
from app.services.workspace import Workspace
from app.storage.paths import AppPaths


def make_ws(tmp_path, scenes: int = 12, **spec_kw) -> Workspace:
    ws = Workspace(AppPaths(tmp_path / "cfg", tmp_path / "data"))
    sp = build_project(tmp_path / "syn", SyntheticSpec(scenes=scenes, keyframes_per_visual=2, **spec_kw), real_files=True)
    ws.projects.switch_to(sp.project)
    return ws


@pytest.fixture
def ews(tmp_path):
    ws = make_ws(tmp_path)
    yield ws
    ws.shutdown()
