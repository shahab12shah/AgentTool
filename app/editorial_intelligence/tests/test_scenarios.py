"""The scenario builder itself: real pipeline narration + attached visuals produce the context the analyzers read."""

from __future__ import annotations

import pytest

from app.editorial_intelligence.context import build_editorial_context
from app.editorial_intelligence.tests.scenarios import V, craft
from app.tests.conftest import needs_ffmpeg

pytestmark = needs_ffmpeg

NARRATION = ("The police had already searched the house three times. Nothing was found in 1981. "
             "Then a detective noticed something in the official report. The report listed a second key. "
             "That key opened the garage. Inside the garage was the evidence nobody expected.")


def test_crafted_scenes_carry_the_attached_visuals_candidates_and_clips(ws, tmp_path):
    craft(ws, tmp_path, NARRATION, {0: V("house", "Exterior of the house", tags=("house", "exterior")), 1: V("house", "Exterior of the house"),
                                    2: V("report", "Police report", source="SCREENSHOT", kind="IMAGE", evidence="EVIDENCE", alts=(V("report2", "Police report page 2", source="SCREENSHOT", kind="IMAGE"),))}, min_scene=1.0, threshold=0.15)
    ctx = build_editorial_context(ws.project)
    assert len(ctx.scenes) >= 3
    s0, s1 = ctx.scenes[0], ctx.scenes[1]
    assert s0.sc.asset is not None and s0.sc.asset.asset_id == s1.sc.asset.asset_id and s1.sc.reuse_count == 1  # the same asset twice
    assert s0.clips and s0.clips[0].kind == "media" and s0.candidates and s0.candidates[0].role == "ASSIGNED"
    s2 = ctx.scenes[2]
    assert s2.candidates[0].evidence_kind == "EVIDENCE" and any(c.role == "ALTERNATIVE" and c.acquirable for c in s2.candidates)
    assert s0.sc.scene.narration and s0.sc.words  # the real pipeline produced narration and word timings
