"""Pacing and cut-timing checker (synthetic projects: exact, no media)."""

from __future__ import annotations

import pytest

from app.analysis.models import Claim, ClaimType, NumberKind, NumericMention
from app.editing.models import Creator, DecisionType, EditingDecision, SceneEditingBrief
from app.media.asset import SourceType
from app.qc.pacing_checker import PacingChecker
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, narrate, new_project, qc_ctx, run_checker
from app.timeline.keyframes import Keyframe

TEXT = "Silver prices rose sharply last week after the central bank statement surprised the market."


def run(p, **kw):
    return run_checker(PacingChecker(), qc_ctx(p, **kw))


def pacing_codes(out):
    return [c for c in codes(out) if c.startswith(("pacing.", "cut."))]


def build(tmp_path, cuts, *, total=60.0, scenes=6, texts=None, importance=0.5, assets=3, by="AI"):
    """``cuts`` = times at which a new shot starts (0 is added). Scenes tile [0, total] and each has one narrated sentence."""
    p = new_project(tmp_path, seconds=total)
    step = total / scenes
    for i in range(scenes):
        add_scene(p, i * step, (i + 1) * step, (texts or [TEXT] * scenes)[i], importance=importance)
    narrate(p)
    pool = [add_asset(p, f"a{k}.mp4", "video", duration=600) for k in range(assets)]
    edges = sorted({0.0, *cuts})
    clips = []
    for i, t in enumerate(edges):
        end = edges[i + 1] if i + 1 < len(edges) else total
        sc = next(s for s in p.scenes if s.start - 1e-9 <= t < s.end + 1e-9)
        clips.append(add_clip(p, "track_v1", pool[i % assets], t, end - t, scene=sc, source_in=10.0 * i, created_by=by))
    return p, clips


def even(total=60.0, every=7.5):
    return [k * every for k in range(1, int(total / every))]


# ---------------------------------------------------------------- rhythm
def test_even_pacing_is_clean_and_reports_metrics(tmp_path):
    p, _ = build(tmp_path, even(60.0, 10.0))  # a cut on every scene (sentence) boundary
    out = run(p)
    assert pacing_codes(out) == []
    m = out.metrics
    assert m["shots"] == 6 and m["cuts"] == 5 and m["cuts_per_minute"] == pytest.approx(5.0)
    assert m["average_shot"] == pytest.approx(10.0) and m["median_shot"] == pytest.approx(10.0) and m["min_shot"] == m["max_shot"] == pytest.approx(10.0)
    assert len(m["curve"]) == 3 and all(w["label"] in ("Slow", "Moderate") for w in m["curve"])
    assert {"narration_wps", "style"} <= set(m)


def test_pacing_too_fast_for_the_narration(tmp_path):
    cuts = [20 + k * 0.8 for k in range(50)]  # a cut every 0.8 s from 0:20 to the end: 75 per minute
    p, _ = build(tmp_path, cuts)
    out = run(p)
    i = find(out, "pacing.too_fast")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and i[0].start_time == pytest.approx(20.0) and "cuts per minute" in i[0].description
    assert i[0].fix is None and i[0].confidence < 100  # a judgement, and changing cuts is the user's call


def test_a_fast_opening_is_allowed(tmp_path):
    p, _ = build(tmp_path, [k * 1.0 for k in range(1, 8)] + [30.0, 45.0])
    assert find(run(p), "pacing.too_fast") == []


def test_fast_cutting_that_the_narration_carries_is_not_flagged(tmp_path):
    cuts = [20 + k * 1.5 for k in range(26)]  # 40 cuts/min
    p, _ = build(tmp_path, cuts, scenes=30, texts=[TEXT] * 30)  # 30 sentences a minute: the story changes every 2 s
    assert find(run(p), "pacing.too_fast") == []


def test_sensitivity_scales_the_limits(tmp_path):
    cuts = [20 + k * 2.0 for k in range(20)]  # 30 cuts/min: under the default limit of 36
    p, _ = build(tmp_path, cuts)
    assert find(run(p), "pacing.too_fast") == []
    p.qc_settings.pacing.sensitivity = 1.0
    assert len(find(run(p), "pacing.too_fast")) == 1
    p.qc_settings.pacing.sensitivity = 0.0
    assert find(run(p), "pacing.too_fast") == []


def test_long_hold_across_many_sentences_is_a_slow_notice(tmp_path):
    p, clips = build(tmp_path, [10.0, 50.0], total=60.0)  # one picture from 0:10 to 0:50 under four sentences
    i = find(run(p), "pacing.too_slow")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and i[0].fix is None and "4 sentences" in i[0].description


def test_a_moving_picture_or_an_evidence_hold_is_not_slow(tmp_path):
    p, clips = build(tmp_path, [10.0, 50.0])
    clips[1].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 39.0, 1.3)]
    assert find(run(p), "pacing.too_slow") == []
    p2, clips2 = build(tmp_path / "ev", [10.0, 50.0])
    p2.assets.get(clips2[1].asset_id).source_type = SourceType.SCREENSHOT  # a document / screenshot is held so it can be read
    assert find(run(p2), "pacing.too_slow") == []
    p3, clips3 = build(tmp_path / "keep", [10.0, 50.0])
    for s in p3.scenes:
        p3.editing_strategy.briefs[s.id] = SceneEditingBrief(s.id, keep_static=True)
    assert find(run(p3), "pacing.too_slow") == []


def _burst(start, end, every):
    t, out = start, []
    while t < end:
        out.append(t)
        t += every
    return out


def test_uneven_pacing_and_its_narrative_justification(tmp_path):
    cuts = even(100, 10.0) + _burst(41.0, 59.0, 1.2)
    p, _ = build(tmp_path, cuts, total=100.0, scenes=5, texts=[TEXT] * 5)
    i = find(run(p), "pacing.uneven")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and "typical" in i[0].description
    p2, _ = build(tmp_path / "justified", cuts, total=100.0, scenes=5, texts=[TEXT] * 5, importance=0.9)  # an important moment may be edited harder
    assert find(run(p2), "pacing.uneven") == []


def test_over_editing_counts_graphics_and_motion_too(tmp_path):
    p, clips = build(tmp_path, even())
    for k in range(40):  # 40 text overlays in 0:20-0:40
        add_clip(p, "track_v5", None, 20.5 + k * 0.45, 0.4, kind="text", text={"text": f"t{k}"})
    out = run(p)
    i = find(out, "pacing.over_edited")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and i[0].start_time == pytest.approx(20.0, abs=0.01)
    assert find(run(build(tmp_path / "calm", even())[0]), "pacing.over_edited") == []


def test_important_moment_with_less_attention_than_its_neighbours(tmp_path):
    cuts = [k * 2.5 for k in range(1, 24) if not 30 <= k * 2.5 < 40]  # a cut every 2.5 s except in scene 4 (0:30-0:40)
    p, _ = build(tmp_path, cuts)
    sc = p.scenes[3]
    sc.importance = 0.9
    sc.numbers = [NumericMention("5 percent", NumberKind.PERCENTAGE)]
    sc.claims = [Claim("c1", "Silver rose five percent.", ClaimType.NUMBER, "s4")]
    i = find(run(p), "pacing.under_emphasized")
    assert len(i) == 1 and i[0].scene_id == sc.id and i[0].fix.kind == "open.scene" and i[0].severity is Severity.NOTICE
    sc.numbers, sc.claims = [], []
    assert find(run(p), "pacing.under_emphasized") == []  # nothing to emphasise


# ---------------------------------------------------------------- cut timing
def _word_times(p):
    return {w.text: w for w in p.transcription.transcript.words}


def test_cut_in_the_middle_of_a_phrase(tmp_path):
    p, _ = build(tmp_path, [4.0, 6.5, 17.0, 30.0, 45.0])
    ph = [w for w in p.transcription.transcript.words if 0 <= w.start < 10]
    mid = (ph[0].start + ph[-1].end) / 2
    p, _ = build(tmp_path / "x", [mid, 17.0, 31.0, 52.0])
    i = find(run(p), "cut.awkward")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and "middle of the phrase" in i[0].description and i[0].fix is None
    p2, _ = build(tmp_path / "clean", [10.0, 20.0, 30.0, 40.0, 50.0])  # cuts on scene (sentence) boundaries
    assert find(run(p2), "cut.awkward") == []


def test_cut_between_a_number_and_its_unit_is_a_warning_in_an_important_scene(tmp_path):
    text = "Silver rose 5 percent after the central bank decision on Tuesday morning."
    p, _ = build(tmp_path, [20.0, 40.0], texts=[text] + [TEXT] * 5, importance=0.8)
    w = _word_times(p)
    t = (w["5"].end + w["percent"].start) / 2
    p, _ = build(tmp_path / "x", [t, 20.0, 40.0], texts=[text] + [TEXT] * 5, importance=0.8)
    i = find(run(p), "cut.awkward")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and "unit" in i[0].description
    p, _ = build(tmp_path / "y", [t, 20.0, 40.0], texts=[text] + [TEXT] * 5, importance=0.5)
    assert find(run(p), "cut.awkward")[0].severity is Severity.NOTICE


def test_an_explained_cut_is_not_awkward(tmp_path):
    p0, _ = build(tmp_path / "probe", [20.0])
    ph = [w for w in p0.transcription.transcript.words if 0 <= w.start < 10]
    mid = (ph[0].start + ph[-1].end) / 2
    p, clips = build(tmp_path / "x", [mid, 20.0])
    d = EditingDecision("dec_00001", clips[1].scene_id, DecisionType.VISUAL_TIMING, "visual:1", clips[1].id, mid, 5.0, {}, "Cut on the gesture for emphasis.", 92.0, Creator.AI)
    p.editing_decisions["dec_00001"] = d
    clips[1].ai_decision_id = "dec_00001"
    assert find(run(p), "cut.awkward") == []


def test_cut_inside_a_sentence_that_only_jumps_forward_in_the_same_footage(tmp_path):
    p, clips = build(tmp_path, [4.5], assets=1)
    clips[1].source_in, clips[1].source_out = 50.0, 50.0 + clips[1].duration  # jump of 40 s inside sentence 1
    i = find(run(p), "cut.unnecessary")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and "jumps" in i[0].description
    clips[1].source_in = clips[0].source_out  # a split in continuous footage is invisible
    clips[1].source_out = clips[1].source_in + clips[1].duration
    assert find(run(p), "cut.unnecessary") == []


def test_micro_cut_burst_across_scenes(tmp_path):
    cuts = [9.4, 9.8, 10.2, 10.6, 11.0]
    p, _ = build(tmp_path, cuts + [30.0, 45.0], scenes=6)
    i = find(run(p), "cut.micro_cuts")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and "across several scenes" in i[0].description
    p2, _ = build(tmp_path / "inside", [12.0, 12.4, 12.8, 13.2, 13.6, 30.0, 45.0])  # all inside one scene: the scene checker reports those
    assert find(run(p2), "cut.micro_cuts") == []


def test_information_cut_away_before_it_can_be_read(tmp_path):
    p, _ = build(tmp_path, [10.0, 20.0, 30.0, 40.0, 50.0])
    add_clip(p, "track_v5", None, 18.8, 1.2 - 0.0, kind="text", text={"text": "5%"})  # 1.2 s: fine
    add_clip(p, "track_v5", None, 29.4, 0.6, kind="text", text={"text": "5%"})  # 0.6 s, gone with the cut at 30
    i = find(run(p), "cut.before_content")
    assert len(i) == 1 and i[0].start_time == pytest.approx(29.4)


# ---------------------------------------------------------------- plumbing
def test_checker_contract_and_no_mutation(tmp_path):
    p, _ = build(tmp_path, even())
    c = PacingChecker()
    assert not c.scene_local and set(c.settings_sections) == {"pacing", "coverage", "style"} and "reference" in c.domains
    before = p.to_document()
    h1 = c.input_hash(qc_ctx(p))
    run(p)
    assert p.to_document() == before
    p.qc_settings.pacing.sensitivity = 0.9
    assert c.input_hash(qc_ctx(p)) != h1


def test_no_visuals_is_not_an_error(tmp_path):
    p = new_project(tmp_path)
    out = run(p)
    assert out.issues == [] and out.metrics["shots"] == 0


def test_one_cut_is_not_reported_twice_and_a_restart_is_not_called_a_jump_forward(tmp_path):
    p0, _ = build(tmp_path / "probe", [20.0])
    ph = [w for w in p0.transcription.transcript.words if 0 <= w.start < 10]
    mid = (ph[0].start + ph[-1].end) / 2
    p, clips = build(tmp_path / "x", [mid, 20.0], assets=1)
    clips[0].source_in, clips[0].source_out = 10.0, 10.0 + clips[0].duration
    clips[1].source_in, clips[1].source_out = 60.0, 60.0 + clips[1].duration  # a jump inside the same sentence, in the middle of the phrase
    out = run(p)
    assert len(find(out, "cut.awkward")) == 1 and find(out, "cut.unnecessary") == []  # the cut that interrupts a phrase is reported once, with the phrase
    clips[1].source_in, clips[1].source_out = 0.0, clips[1].duration  # restarting short footage under a long sentence
    p2, clips2 = build(tmp_path / "y", [4.5], assets=1)
    clips2[1].source_in, clips2[1].source_out = 0.0, clips2[1].duration
    i = find(run(p2), "cut.unnecessary")
    assert len(i) == 1 and "again" in i[0].description and "forward" not in i[0].description


def test_a_run_leaves_no_state_on_the_shared_checker_instance(tmp_path):
    """One checker instance serves every run (and thread): the sensitivity of one run must not leak into the next one."""
    cuts = [i * 1.5 for i in range(1, 40)]
    p, _ = build(tmp_path, cuts, total=60.0, assets=3)
    ck = PacingChecker()
    p.qc_settings.pacing.sensitivity = 1.0
    strict = run_checker(ck, qc_ctx(p))
    p.qc_settings.pacing.sensitivity = 0.0
    lenient = run_checker(ck, qc_ctx(p))
    p.qc_settings.pacing.sensitivity = 1.0
    again = run_checker(ck, qc_ctx(p))
    assert vars(ck) == {} and codes(strict) == codes(again) and codes(strict) != codes(lenient)
