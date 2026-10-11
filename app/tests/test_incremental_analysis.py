"""Analysis skip: re-evaluating a scene's candidates with unchanged inputs returns the stored evaluation (a recorded cache hit); one changed input recomputes only that scene; an explicit
force always recomputes; an evaluator that is not a pure function of its inputs is never cached."""

from __future__ import annotations

import pytest

from app.media.asset import SourceType as S
from app.performance.analysis_cache import AnalysisCache
from app.research.evaluation import MetadataEvaluator, VisualEvaluationService
from app.research.models import Acquisition
from app.tests.conftest import needs_ffmpeg
from app.tests.test_research_service import JUNK, MIDDLE, STRONG, install, media, research, wait  # noqa: F401  (fixtures)
from app.tests.helpers import FakeProvider

pytestmark = needs_ffmpeg


@pytest.fixture
def two_scene_ws(research_ws, media):  # noqa: F811
    ws = research_ws
    items = [{**STRONG, "source_type": S.STOCK_VIDEO, "id": "v1", "local_path": str(media["video"]), "acquisition": Acquisition.LOCAL},
             {**MIDDLE, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "i1", "local_path": str(media["image"]), "acquisition": Acquisition.LOCAL},
             {**JUNK, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "i2", "local_path": str(media["junk"]), "acquisition": Acquisition.LOCAL}]
    ws.provider = FakeProvider(items, [S.STOCK_VIDEO, S.STOCK_IMAGE], name="stock")
    install(ws, ws.provider)
    a, b = ws.project.scenes[0].id, ws.project.scenes[1].id
    research(ws, ws.project.scenes[0])
    research(ws, ws.project.scenes[1])
    calls: list[str] = []
    real = ws.research.engine.evaluate_and_rank

    def spy(brief, candidates, *args, **kw):
        calls.append(brief.scene_id)
        return real(brief, candidates, *args, **kw)

    ws.research.engine.evaluate_and_rank = spy
    ws.calls, ws.a, ws.b = calls, a, b
    return ws


def rescore(ws, sid, **kw):
    ws.research.rescore(sid, **kw)
    wait(ws)


def hits(ws) -> int:
    return ws.performance.cache.get_cache_stats()["categories"].get("analysis", {}).get("hits", 0)


def state_of(ws, sid):
    p = ws.project
    st = p.research_status[sid]
    return (st.status.value, st.best_id, list(st.alternatives), [(e.candidate_id, e.role) for e in st.ranked], {k: round(v.overall, 4) for k, v in p.candidate_scores.items() if p.visual_candidates[k].scene_id == sid})


def test_unchanged_inputs_are_not_evaluated_again_and_give_the_same_result(two_scene_ws):
    ws = two_scene_ws
    rescore(ws, ws.a)
    first = state_of(ws, ws.a)
    assert ws.calls == [ws.a] and hits(ws) == 0
    rescore(ws, ws.a)
    assert ws.calls == [ws.a], "the second evaluation must come from the stored result"
    assert hits(ws) == 1
    assert state_of(ws, ws.a) == first


def test_force_always_recomputes(two_scene_ws):
    ws = two_scene_ws
    rescore(ws, ws.a)
    rescore(ws, ws.a, force=True)
    rescore(ws, ws.a, force=True)
    assert ws.calls == [ws.a, ws.a, ws.a] and hits(ws) == 0
    rescore(ws, ws.a)  # the forced runs stored their (identical) result: a normal run may use it
    assert ws.calls == [ws.a] * 3 and hits(ws) == 1


def test_changing_one_scene_recomputes_only_that_scene(two_scene_ws):
    ws = two_scene_ws
    rescore(ws, ws.a)
    rescore(ws, ws.b)
    assert ws.calls == [ws.a, ws.b]
    ws.research.reject_best([ws.b])  # scene b's candidates change (and with them the ranking history scene a is judged against only through b's assignment, which is untouched)
    wait(ws)
    ws.calls.clear()
    h0 = hits(ws)
    rescore(ws, ws.a)
    assert ws.calls == [] and hits(ws) == h0 + 1  # scene a: nothing it reads changed
    rescore(ws, ws.b)
    assert ws.calls == [ws.b]  # scene b: its candidates changed


def test_a_changed_preference_or_setting_is_a_different_key(two_scene_ws):
    ws = two_scene_ws
    rescore(ws, ws.a)
    n = len(ws.calls)
    ws.project.visual_preferences.min_accuracy_score = int(ws.project.visual_preferences.min_accuracy_score) - 7
    rescore(ws, ws.a)
    assert len(ws.calls) == n + 1
    ws.project.research_settings.max_candidates = max(2, ws.project.research_settings.max_candidates - 1)
    rescore(ws, ws.a)
    assert len(ws.calls) == n + 2
    rescore(ws, ws.a)
    assert len(ws.calls) == n + 2  # and now the new inputs are stored


def test_an_evaluator_that_is_not_a_pure_function_is_never_cached(two_scene_ws):
    ws = two_scene_ws

    class Delegating:  # same answers as the built-in evaluator, but nothing says it is a pure function of its inputs (think: a vision model)
        def __init__(self) -> None:
            self.inner = MetadataEvaluator()

        def evaluate(self, brief, candidate, min_accuracy=85.0):
            return self.inner.evaluate(brief, candidate, min_accuracy)

    moody = Delegating()
    ws.research.engine.evaluation = VisualEvaluationService(evaluator=moody)
    rescore(ws, ws.a)
    rescore(ws, ws.a)
    assert len(ws.calls) == 2 and hits(ws) == 0


def test_without_a_cache_everything_still_works(two_scene_ws):
    ws = two_scene_ws
    ws.research.analysis = AnalysisCache(lambda: None)
    rescore(ws, ws.a)
    rescore(ws, ws.a)
    assert ws.calls == [ws.a, ws.a]


def test_a_corrupt_or_undecodable_entry_is_a_miss_not_an_error(two_scene_ws):
    ws = two_scene_ws
    rescore(ws, ws.a)
    entry = next(e for e in ws.performance.cache.entries("analysis"))
    ws.performance.cache.put(entry.key, {"candidates": "not a list"}, category="analysis", data_type=entry.data_type, version=entry.version)
    rescore(ws, ws.a)
    assert ws.calls == [ws.a, ws.a]
