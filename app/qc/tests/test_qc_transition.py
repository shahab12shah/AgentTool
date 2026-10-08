"""Transition checker."""

from __future__ import annotations

import pytest

from app.analysis.models import Claim, ClaimType, NumberKind, NumericMention
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, narrate, new_project, qc_ctx, run_checker
from app.qc.tests.test_qc_style_compat import apply_style
from app.qc.transition_checker import TransitionChecker

TEXT = "Silver prices rose 5 percent last week after the central bank statement surprised the whole market."


def run(p, **kw):
    return run_checker(TransitionChecker(), qc_ctx(p, **kw))


def project(tmp_path, n=6, each=10.0, transitions=None, *, style="professional", by="AI", topics=None):
    """``n`` picture clips of ``each`` seconds. ``transitions``: {clip index: (type, duration)} - the transition into that clip."""
    total = n * each
    p = new_project(tmp_path, seconds=total)
    p.editing_settings.style = style
    for i in range(n):
        add_scene(p, i * each, (i + 1) * each, TEXT if i == 0 else f"Scene number {i} talks about something else entirely, topic {(topics or {}).get(i, i)}.", topic=(topics or {}).get(i, f"topic{i}"))
    narrate(p)
    a = add_asset(p, "city.mp4", "video", duration=900)
    clips = [add_clip(p, "track_v1", a, i * each, each, scene=p.scenes[i], source_in=i * 20.0, created_by=by) for i in range(n)]
    for idx, (kind, dur) in (transitions or {}).items():
        clips[idx].transition = {"type": kind, "duration": dur}
    return p, clips


def test_sparse_soft_transitions_are_clean_and_counted(tmp_path):
    p, _ = project(tmp_path, transitions={2: ("DISSOLVE", 0.6), 4: ("FADE", 0.5)})
    out = run(p)
    assert out.issues == []
    assert out.metrics["transitions"] == 2 and out.metrics["types"] == {"DISSOLVE": 1, "FADE": 1} and out.metrics["per_minute"] == pytest.approx(2.0)


def test_no_transitions_means_no_issue_and_none_is_ever_recommended(tmp_path):
    p, _ = project(tmp_path)
    out = run(p)
    assert out.issues == [] and out.metrics["transitions"] == 0


def test_transition_longer_than_the_limit(tmp_path):
    p, clips = project(tmp_path, transitions={2: ("DISSOLVE", 1.8)})
    i = find(run(p), "transition.too_long")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and "1.80 s" in i[0].description
    assert i[0].fix.kind == "transition.shorten" and i[0].fix.params == {"clip_id": clips[2].id, "duration": 1.5}
    assert i[0].fix.needs_confirmation and not i[0].fix.safe and not i[0].auto_fix_safe  # shortening is a creative change


def test_transition_that_takes_too_much_of_a_short_neighbour(tmp_path):
    p, clips = project(tmp_path, n=4, each=2.0, transitions={2: ("FADE", 1.0)})
    i = find(run(p), "transition.too_long")
    assert len(i) == 1 and i[0].fix.params["duration"] == pytest.approx(0.8) and "shorter neighbouring clip" in i[0].description


def test_a_transition_longer_than_its_clip_is_the_timeline_checkers_business(tmp_path):
    p, clips = project(tmp_path, n=3, each=2.0, transitions={1: ("FADE", 5.0)})
    assert find(run(p), "transition.too_long") == []


def test_too_many_transitions_in_a_minute(tmp_path):
    p, _ = project(tmp_path, n=12, each=5.0, transitions={i: ("DISSOLVE", 0.4) for i in range(1, 12)})  # 11 in 60 s
    i = find(run(p), "transition.excessive")
    assert any(x.severity is Severity.WARNING and "within one minute" in x.description for x in i)
    p2, _ = project(tmp_path / "ok", n=12, each=5.0, transitions={i: ("DISSOLVE", 0.4) for i in range(2, 12, 2)})
    assert not [x for x in find(run(p2), "transition.excessive") if x.severity is Severity.WARNING]


def test_two_transitions_back_to_back(tmp_path):
    p, clips = project(tmp_path, n=4, each=1.5, transitions={1: ("FADE", 0.5), 2: ("FADE", 0.4)})  # the second starts 1.5 s after the first ends... use 0.9 s gap
    clips[1].transition = {"type": "FADE", "duration": 1.1}
    i = [x for x in find(run(p), "transition.excessive") if x.severity is Severity.NOTICE]
    assert len(i) == 1 and "too close" in i[0].title


def _number_word(p):
    sc = p.scenes[0]
    ws = [w for w in p.transcription.transcript.words if w.text == "5"]
    sc.numbers = [NumericMention("5 percent", NumberKind.PERCENTAGE, 5.0, "sent_0000", False, [ws[0].word_id])]
    return ws[0]


def test_transition_over_a_spoken_number(tmp_path):
    p, clips = project(tmp_path)
    w = _number_word(p)
    p2, clips2 = project(tmp_path / "t")
    _number_word(p2)
    # a transition starting right at the number inside scene 1 (the clip boundary is moved there)
    clips2[1].timeline_start = w.start
    clips2[0].duration = w.start
    clips2[1].duration = 10.0 - w.start + 10.0
    clips2[1].transition = {"type": "DISSOLVE", "duration": 0.6}
    clips2[1].scene_id = p2.scenes[0].id
    i = find(run(p2), "transition.in_phrase")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE and "“5 percent”" in i[0].description
    clips[1].transition = {"type": "DISSOLVE", "duration": 0.6}  # at the scene boundary, nowhere near the number
    assert find(run(p), "transition.in_phrase") == []


def test_transition_in_the_middle_of_a_claim(tmp_path):
    p, clips = project(tmp_path, n=3, each=10.0)
    p.scenes[0].claims = [Claim("c1", "Silver rose five percent last week.", ClaimType.FACT, "sent_0000")]
    clips[1].timeline_start, clips[0].duration = 4.0, 4.0
    clips[1].duration = 16.0
    clips[1].transition = {"type": "FADE", "duration": 0.5}
    i = find(run(p), "transition.in_phrase")
    assert len(i) == 1 and "claim" in i[0].description


def test_wipes_and_slides_do_not_fit_a_documentary(tmp_path):
    p, _ = project(tmp_path, style="documentary", transitions={2: ("WIPE", 0.5), 4: ("SLIDE", 0.5)})
    assert len(find(run(p), "transition.distracting")) == 2
    p2, _ = project(tmp_path / "dyn", style="dynamic", transitions={2: ("WIPE", 0.5), 4: ("SLIDE", 0.5)})
    assert find(run(p2), "transition.distracting") == []


def test_a_new_effect_every_time(tmp_path):
    p, _ = project(tmp_path, n=6, transitions={1: ("FADE", 0.5), 2: ("DISSOLVE", 0.5), 3: ("WIPE", 0.5), 4: ("SLIDE", 0.5)}, style="dynamic")
    i = find(run(p), "transition.distracting")
    assert len(i) == 1 and "different transition every time" in i[0].title.lower()


def test_more_transitions_than_the_style_asks_for(tmp_path):
    p, _ = project(tmp_path, n=8, each=7.5, style="documentary", transitions={i: ("DISSOLVE", 0.5) for i in range(1, 8)})  # 7 in 60 s: busy for a documentary, under the hard limit
    p.editing_settings.transition_frequency = 0.5
    i = find(run(p), "transition.style_mismatch")
    assert len(i) == 1 and "documentary" in i[0].description and i[0].severity is Severity.NOTICE


def test_transitions_between_related_scenes_in_a_calm_style(tmp_path):
    same = {i: "silver market" for i in range(6)}
    p, _ = project(tmp_path, n=6, style="professional", topics=same, transitions={1: ("DISSOLVE", 0.5), 2: ("DISSOLVE", 0.5), 3: ("DISSOLVE", 0.5), 4: ("DISSOLVE", 0.5)})
    i = find(run(p), "transition.style_mismatch")
    assert len(i) == 1 and "not at a change of topic" in i[0].title


def test_user_owned_and_locked_transitions_are_reported_without_a_fix(tmp_path):
    p, clips = project(tmp_path, transitions={2: ("DISSOLVE", 1.8)}, by="USER")
    i = find(run(p), "transition.too_long")[0]
    assert i.locked and not i.auto_fix_available and "you" in i.fix_blocked_reason.lower()
    p2, clips2 = project(tmp_path / "l", transitions={2: ("DISSOLVE", 1.8)})
    clips2[2].locked = True
    assert not find(run(p2), "transition.too_long")[0].auto_fix_available


def test_transition_share_compared_with_an_applied_reference_style(tmp_path):
    p, _ = project(tmp_path, transitions={2: ("DISSOLVE", 0.5)})
    assert not [i for i in run(p).issues if i.code.startswith("style.")]
    apply_style(p, pacing=50.0)
    p.reference_style_profile.scores.transition_frequency = 90.0
    p.reference_style_profile.confidence["transition_detection"] = 0.9
    from app.reference.style_model import DIMENSION_CONFIDENCE

    p.reference_style_profile.confidence[DIMENSION_CONFIDENCE["transition_frequency"]] = 0.9
    i = find(run(p), "style.transition_deviation")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE


def test_contract_and_no_mutation(tmp_path):
    p, _ = project(tmp_path, transitions={2: ("DISSOLVE", 1.8)})
    c = TransitionChecker()
    assert not c.scene_local and set(c.settings_sections) == {"transition", "style"}
    before = p.to_document()
    h = c.input_hash(qc_ctx(p))
    run(p)
    assert p.to_document() == before
    p.qc_settings.transition.max_duration = 3.0
    assert c.input_hash(qc_ctx(p)) != h
    assert codes(run(p)) == []
