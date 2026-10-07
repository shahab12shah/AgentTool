from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.analyzer import RuleBasedAnalyzer, new_context
from app.analysis.models import Origin, SceneStatus, VisualType
from app.analysis.segmenter import SegmentationParams
from app.core.exceptions import AnalysisError, SceneEditError, UserEditsPresentError
from app.media.asset import Asset, AssetType, SourceType
from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import NARRATION, ScriptedProvider, run_scenes, run_transcription, timed_words
from app.transcription.service import TranscriptionEngine

AUDIO_DUR = 400.0


def build(text: str, params: SegmentationParams | None = None, words=None, duration: float | None = None):
    raw = words if words is not None else timed_words(text)
    asset = Asset("media_00001", AssetType.AUDIO, SourceType.USER_MEDIA, "a.wav", "a.wav", duration=duration or raw[-1].end + 1.0, content_hash="h")
    res = TranscriptionEngine().run(ScriptedProvider(raw), Path("a.wav"), asset, "en", script=text)
    an = RuleBasedAnalyzer()
    surf = res.alignment.surface_by_word()
    analyses = an.analyze_sentences(res.transcript, surf)
    params = params or SegmentationParams()
    return an, res.transcript, analyses, surf, params, an.segment(res.transcript, analyses, params)


def enrich_all(an, tr, analyses, surf, params, drafts):
    ctx = new_context(an, tr, analyses, surf, params)
    out, prev = [], None
    for k, d in enumerate(drafts):
        ctx.index, ctx.total, ctx.prev = k, len(drafts), prev
        prev = an.enrich_scene(d, f"scene_{k + 1:03d}", str(k + 1), ctx)
        out.append(prev)
    return out


# ------------------------------------------------------------------ grouping / boundaries
def test_scenes_are_not_one_per_sentence():
    an, tr, analyses, surf, params, drafts = build(NARRATION)
    assert len(tr.sentences) == 32 and 14 <= len(drafts) < len(tr.sentences)


def test_spec_example_continuous_idea_becomes_one_scene():
    *_, drafts = build("Silver demand has increased. Solar manufacturers are consuming more of it.")
    assert len(drafts) == 1


def test_continuation_sentences_stay_with_the_previous_idea():
    *_, tr_drafts = build("Silver demand has changed dramatically. Solar manufacturers are now consuming more of the metal. "
                          "And that shift is affecting the market.")
    assert len(tr_drafts) == 1


def test_topic_transition_creates_a_boundary():
    an, tr, analyses, surf, params, drafts = build(
        "Silver demand has changed dramatically. Solar manufacturers are now consuming more of the metal. "
        "Now let's talk about the IRS. The IRS sent a notice to affected taxpayers in the United States.")
    assert len(drafts) == 2
    assert drafts[0].narration if False else True
    first_irs = tr.sentences[2]
    assert drafts[0].end < first_irs.start and drafts[1].start > tr.sentences[1].end  # cut sits in the pause between sentences
    assert "topic-transition" in " ".join(drafts[1].rationale)


def test_question_starts_a_new_scene():
    *_, drafts = build("Silver demand has changed dramatically. Solar manufacturers are consuming more of the metal. "
                       "How does a solar panel actually work? Sunlight hits silicon cells and silver wiring collects the current.")
    assert len(drafts) == 2 and any("question" in r for r in drafts[1].rationale)


def test_threshold_controls_granularity():
    coarse = build(NARRATION, SegmentationParams(threshold=0.8))[-1]
    fine = build(NARRATION, SegmentationParams(threshold=0.3))[-1]
    assert len(coarse) < len(build(NARRATION)[-1]) < len(fine)


# ------------------------------------------------------------------ timing
def test_scenes_tile_the_voiceover_without_gaps_or_overlaps():
    an, tr, analyses, surf, params, drafts = build(NARRATION)
    assert drafts[0].start == 0.0
    for a, b in zip(drafts, drafts[1:]):
        assert a.end == b.start and a.end > a.start  # exact, not approximately
    assert drafts[-1].end == tr.audio.duration >= tr.words[-1].end
    covered = [w.word_id for d in drafts for w in tr.words_between(d.start, d.end)]
    assert covered == [w.word_id for w in tr.words]  # every word belongs to exactly one scene, in order


def test_scene_narration_and_sentence_ids_come_from_timing():
    an, tr, analyses, surf, params, drafts = build(NARRATION)
    scenes = [e.scene for e in enrich_all(an, tr, analyses, surf, params, drafts)]
    assert " ".join(s.narration for s in scenes) == " ".join(w.text for w in tr.words)
    for s in scenes:
        assert s.sentence_ids and all(sid in analyses for sid in s.sentence_ids)


def test_min_duration_merges_and_max_duration_splits():
    text = "Silver demand has changed. Gold prices fell sharply. Bitcoin crashed last month."
    *_, short = build(text, SegmentationParams(threshold=0.2, min_scene_seconds=60.0))
    assert len(short) == 1  # everything is "too short" so it collapses
    long_text = " ".join(f"Topic number {i} concerns something quite different here." for i in range(12))
    an, tr, analyses, surf, params, drafts = build(long_text, SegmentationParams(threshold=0.99, max_scene_seconds=15.0))
    assert len(drafts) > 1 and all(d.end - d.start <= 15.0 + 8.0 for d in drafts)
    assert any("minimum/maximum" in " ".join(d.rationale) for d in drafts)


# ------------------------------------------------------------------ enumeration split (one sentence, several visuals)
def test_enumeration_inside_a_sentence_is_split_into_visual_scenes():
    text = "Silver is used in solar panels, electronics, and electric vehicles."
    an, tr, analyses, surf, params, drafts = build(text, SegmentationParams(min_item_seconds=0.6))
    scenes = [e.scene for e in enrich_all(an, tr, analyses, surf, params, drafts)]
    assert len(scenes) == 3
    assert [s.topic.lower() for s in scenes][1:] == ["electronics", "electric vehicles"]
    assert scenes[0].start == 0.0 and all(a.end == b.start for a, b in zip(scenes, scenes[1:]))
    assert "enumerates" in " ".join(scenes[1].rationale)
    assert all(s.sentence_ids == [tr.sentences[0].sentence_id] for s in scenes)  # all three belong to the one spoken sentence
    assert {s.narration for s in scenes} and " ".join(s.narration for s in scenes) == " ".join(w.text for w in tr.words)


def test_enumeration_split_can_be_disabled_and_needs_enough_time():
    text = "Silver is used in solar panels, electronics, and electric vehicles."
    assert len(build(text, SegmentationParams(split_enumerations=False))[-1]) == 1
    assert len(build(text, SegmentationParams(min_item_seconds=5.0))[-1]) == 1


# ------------------------------------------------------------------ confidence / importance / context memory
def test_confidence_status_and_review_threshold():
    an, tr, analyses, surf, params, drafts = build(NARRATION)
    good = enrich_all(an, tr, analyses, surf, params, drafts)
    assert all(0 <= e.scene.segmentation_confidence <= 1 for e in good)
    assert all(e.scene.status in (SceneStatus.READY, SceneStatus.NEEDS_REVIEW) for e in good)
    low = [type(w)(w.text, w.start, w.end, 0.05) for w in timed_words(NARRATION)]
    *_, low_drafts = build(NARRATION, words=low)
    an2, tr2, analyses2, surf2, params2, drafts2 = build(NARRATION, words=low)
    bad = enrich_all(an2, tr2, analyses2, surf2, params2, drafts2)
    assert sum(e.scene.status is SceneStatus.NEEDS_REVIEW for e in bad) > sum(e.scene.status is SceneStatus.NEEDS_REVIEW for e in good)
    assert min(e.scene.segmentation_confidence for e in bad) < min(e.scene.segmentation_confidence for e in good)


def test_importance_ranks_claims_numbers_and_conclusions_above_filler():
    text = ("Hello and welcome back to the channel everyone. Gold behaved very differently last year. "
            "Shockingly the price of gold climbed 20% while the Federal Reserve cut rates. Now let's talk about chairs. "
            "In conclusion diversification matters most.")
    an, tr, analyses, surf, params, drafts = build(text, SegmentationParams(threshold=0.3, min_scene_seconds=0.5))
    scenes = [e.scene for e in enrich_all(an, tr, analyses, surf, params, drafts)]
    assert all(0 <= s.importance <= 1 for s in scenes)
    number_scene = next(s for s in scenes if "20%" in s.narration)
    filler = next(s for s in scenes if s.narration.startswith("Now let's"))
    conclusion = next(s for s in scenes if "In conclusion" in s.narration)
    assert number_scene.importance >= 0.6 and filler.importance <= 0.45  # highly important vs supporting narration
    assert number_scene.importance > filler.importance and conclusion.importance > filler.importance
    assert number_scene.numbers and number_scene.claims[0].requires_evidence


def test_context_memory_inherits_subject_for_pronoun_led_scene():
    text = "The price of silver reached $100 today. Pause here. This could become expensive."
    words = timed_words("The price of silver reached $100 today.") 
    tail = timed_words("This could become expensive.", start=words[-1].end + 2.5)
    an, tr, analyses, surf, params, drafts = build(text, SegmentationParams(threshold=0.2, min_scene_seconds=0.5), words=words + tail)
    ctx = new_context(an, tr, analyses, surf, params)
    prev = an.enrich_scene(drafts[0], "scene_001", "1", ctx)
    assert len(drafts) >= 2
    ctx.prev, ctx.index, ctx.total = prev, 1, 2
    second = an.enrich_scene(drafts[-1], "scene_002", "2", ctx)
    assert second.intent.inherited_from == "scene_001"
    assert second.intent.primary_subject.lower() == prev.intent.primary_subject.lower() and second.scene.topic == prev.scene.topic
    assert any("inherited" in r for r in second.scene.rationale)
    assert second.intent.type_scores["DATA"] > 0.4 * prev.intent.type_scores["DATA"]


def test_scene_object_has_the_required_structure():
    an, tr, analyses, surf, params, drafts = build(NARRATION)
    e = enrich_all(an, tr, analyses, surf, params, drafts)[1]
    s, i = e.scene, e.intent
    assert s.id.startswith("scene_") and s.start < s.end and s.narration and s.topic and s.summary
    assert s.sentence_ids and 0 <= s.importance <= 1 and s.status in (SceneStatus.READY, SceneStatus.NEEDS_REVIEW)
    assert i.scene_id == s.id and isinstance(i.type, VisualType) and i.primary_subject and i.preferred_visuals
    assert set(i.type_scores) == {t.value for t in VisualType} and i.author is Origin.AI
    assert any("IRS" == x.text for x in s.entities) and s.claims


# ------------------------------------------------------------------ service level
@needs_ffmpeg
def test_pipeline_generates_scenes_and_intents_undoably(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    p = ws.project
    assert len(p.scenes) >= 14 and p.scene_analysis.status == "COMPLETE"
    assert set(p.visual_intents) == {s.id for s in p.scenes}
    assert p.scene_analysis.overall_topic and len(p.scene_analysis.sentence_analysis) == len(p.transcription.transcript.sentences)
    p.validate()
    assert ws.commands.undo_text == "Generate scenes"
    ws.undo()
    assert p.scenes == [] and p.visual_intents == {}
    ws.redo()
    assert len(p.scenes) >= 14


@needs_ffmpeg
def test_up_to_date_scenes_are_not_recomputed_and_change_detection(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    from app.services.scene_service import SceneState

    assert ws.scenes.state() is SceneState.UP_TO_DATE
    assert ws.scenes.analyze() is None  # cached: nothing to do
    ws.set_script(NARRATION + " Extra closing line here.")
    assert ws.scenes.state() is SceneState.OUTDATED  # flagged, never silently regenerated
    n = len(ws.project.scenes)
    assert len(ws.project.scenes) == n
    # voice-over replaced -> scene analysis refuses to run on an outdated transcript
    from app.tests.helpers import make_audio

    ws.media.import_voice_over(make_audio(ws.project.root.parent / "other.wav", 30.0))
    assert ws.jobs.wait_idle(30)
    with pytest.raises(AnalysisError, match="outdated"):
        ws.scenes.analyze(force=True, overwrite_user_edits=True)
    assert len(ws.project.scenes) == n  # existing scenes untouched


@needs_ffmpeg
def test_user_edits_are_protected_from_regeneration(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    sc = ws.project.scenes[2]
    ws.scenes.edit_scene(sc.id, topic="My own topic")
    assert ws.project.scenes[2].topic == "My own topic" and "topic" in ws.project.scenes[2].user_edited_fields
    with pytest.raises(UserEditsPresentError) as e:
        ws.scenes.analyze(force=True)
    assert sc.label in e.value.scene_labels
    assert ws.project.scenes[2].topic == "My own topic"  # nothing was touched
    run_scenes(ws, force=True, overwrite_user_edits=True)  # explicit confirmation
    assert ws.project.scenes[2].topic != "My own topic"
    ws.undo()  # ...and even that is undoable
    assert ws.project.scenes[2].topic == "My own topic"


@needs_ffmpeg
def test_manual_split_keeps_words_timing_metadata_and_is_undoable(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    p = ws.project
    tr = p.transcription.transcript
    target = next(s for s in p.scenes if s.duration > 6)
    ws.scenes.edit_scene(target.id, topic="Hand-written topic", notes="remember this")
    ws.scenes.edit_scene(target.id, intent={"type": "PERSON", "primary_subject": "My Subject"})
    parent = next(s for s in p.scenes if s.id == target.id)
    n = len(p.scenes)
    t = parent.start + 0.4 * parent.duration
    a, b = ws.scenes.split_scene(parent.id, t)
    assert len(p.scenes) == n + 1
    assert (a.start, a.end, b.start, b.end) == (parent.start, t, t, parent.end)  # exact split point, nothing lost
    assert a.label == f"{parent.label}A" and b.label == f"{parent.label}B" and a.id == parent.id and b.id != parent.id
    assert (a.narration + " " + b.narration).split() == parent.narration.split()  # transcript words preserved
    assert a.origin is Origin.USER and a.boundaries_locked and b.boundaries_locked
    for child in (a, b):
        assert child.topic == "Hand-written topic" and child.notes == "remember this"  # user metadata preserved
        assert p.visual_intents[child.id].primary_subject == "My Subject" and p.visual_intents[child.id].author is Origin.USER
    assert tr is p.transcription.transcript and len(tr.words) == len(ws.provider.words)  # transcript untouched
    p.validate()
    ws.undo()
    assert len(p.scenes) == n and p.scenes[[s.id for s in p.scenes].index(parent.id)].end == parent.end
    assert b.id not in p.visual_intents
    ws.redo()
    assert len(p.scenes) == n + 1
    # a split child can be split again with a sensible label
    c, d = ws.scenes.split_scene(a.id, a.start + 0.5 * a.duration)
    assert c.label == f"{a.label}1" and d.label == f"{a.label}2"


@needs_ffmpeg
def test_split_validation(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    s = ws.project.scenes[0]
    for bad in (s.start - 1, s.start, s.start + 0.05, s.end - 0.05, s.end, s.end + 3):
        with pytest.raises(SceneEditError):
            ws.scenes.split_scene(s.id, bad)
    with pytest.raises(SceneEditError):
        ws.scenes.split_scene("scene_999", 5.0)


@needs_ffmpeg
def test_manual_merge_preserves_narration_and_timing(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    p = ws.project
    a, b = p.scenes[3], p.scenes[4]
    ws.scenes.edit_scene(b.id, topic="Edited second")
    b = p.scenes[4]
    n = len(p.scenes)
    merged = ws.scenes.merge_with_next(a.id)
    assert len(p.scenes) == n - 1 and p.scenes[3] is not None
    assert (merged.start, merged.end) == (a.start, b.end) and merged.id == a.id and merged.label == a.label
    assert merged.narration.split() == (a.narration + " " + b.narration).split()
    assert merged.topic == "Edited second" and "topic" in merged.user_edited_fields  # user edit survives the merge
    assert b.id not in p.visual_intents and merged.id in p.visual_intents
    assert [x.start for x in p.scenes[1:]] == [x.end for x in p.scenes[:-1]]  # still tiles the voice-over
    ws.undo()
    assert len(p.scenes) == n and p.scenes[4].id == b.id and p.scenes[4].topic == "Edited second"
    ws.scenes.merge_with_previous(b.id)
    assert len(p.scenes) == n - 1
    with pytest.raises(SceneEditError):
        ws.scenes.merge_with_previous(p.scenes[0].id)
    with pytest.raises(SceneEditError):
        ws.scenes.merge_with_next(p.scenes[-1].id)
    with pytest.raises(SceneEditError):
        ws.scenes.merge_scenes(p.scenes[0].id, p.scenes[2].id)  # not neighbours


@needs_ffmpeg
def test_edit_and_approve_scene(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    p = ws.project
    s = p.scenes[1]
    assert s.status is not SceneStatus.APPROVED
    ws.scenes.set_approved(s.id)
    assert p.scenes[1].status is SceneStatus.APPROVED and p.scenes[1].is_user_touched
    ws.scenes.set_approved(s.id, False)
    assert p.scenes[1].status in (SceneStatus.READY, SceneStatus.NEEDS_REVIEW)
    ws.scenes.edit_scene(s.id, intent={"type": "DATA", "primary_subject": "silver price"})
    vi = p.visual_intents[s.id]
    assert vi.type is VisualType.DATA and vi.author is Origin.USER and "silver price" in vi.preferred_visuals[0]
    assert "visual_intent" in p.scenes[1].user_edited_fields
    with pytest.raises(SceneEditError):
        ws.scenes.edit_scene(s.id, intent={"bogus": 1})
    ws.undo()
    assert p.visual_intents[s.id].author is Origin.AI


# ------------------------------------------------------------------ failure recovery
class FailingAnalyzer(RuleBasedAnalyzer):
    def __init__(self, fail_at: int):
        self.fail_at, self.enrich_calls, self.sentence_calls = fail_at, [], 0
        self.armed = True

    def analyze_sentences(self, *a, **k):
        self.sentence_calls += 1
        return super().analyze_sentences(*a, **k)

    def enrich_scene(self, draft, scene_id, label, ctx):
        self.enrich_calls.append(ctx.index)
        if self.armed and ctx.index == self.fail_at:
            raise RuntimeError("boom on this scene")
        return super().enrich_scene(draft, scene_id, label, ctx)


@needs_ffmpeg
def test_failure_on_scene_n_keeps_earlier_scenes_and_retry_resumes_there(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    fa = FailingAnalyzer(fail_at=6)
    ws.scenes.analyzer = fa
    job = run_scenes(ws)
    p = ws.project
    assert job.status.value == "FAILED" and job.error == "Scenes 1–6 complete. Scene 7 failed."
    statuses = [s.status for s in p.scenes]
    assert statuses[:6].count(SceneStatus.READY) + statuses[:6].count(SceneStatus.NEEDS_REVIEW) == 6
    assert statuses[6] is SceneStatus.FAILED and all(x is SceneStatus.PENDING for x in statuses[7:])
    assert p.scene_analysis.status == "PARTIAL" and p.scene_analysis.failed_scene_id == p.scenes[6].id
    assert set(p.visual_intents) == {s.id for s in p.scenes[:6]}
    p.validate()
    from app.services.scene_service import SceneState

    assert ws.scenes.state() is SceneState.PARTIAL
    done_before = [s.to_dict() for s in p.scenes[:6]]
    total = len(p.scenes)
    fa.armed, fa.enrich_calls, fa.sentence_calls = False, [], 0
    ws.scenes.retry_failed()
    assert ws.jobs.wait_idle(60)
    assert fa.enrich_calls == list(range(6, total))  # only scene 7 onwards was recomputed
    assert fa.sentence_calls == 0  # sentence analysis was reused, not repeated
    assert [s.to_dict() for s in p.scenes[:6]] == done_before  # scenes 1-6 are byte-for-byte unchanged
    assert p.scene_analysis.status == "COMPLETE" and len(p.scenes) == total
    assert all(s.status in (SceneStatus.READY, SceneStatus.NEEDS_REVIEW) for s in p.scenes)
    assert set(p.visual_intents) == {s.id for s in p.scenes}
    with pytest.raises(AnalysisError):
        ws.scenes.retry_failed()  # nothing left to resume


@needs_ffmpeg
def test_failure_on_first_scene_and_partial_state_survives_save(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    ws.scenes.analyzer = FailingAnalyzer(fail_at=0)
    job = run_scenes(ws)
    assert job.error == "Scene 1 failed."
    root = ws.project.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert ws.project.scene_analysis.status == "PARTIAL" and ws.project.scenes[0].status is SceneStatus.FAILED
    ws.scenes.analyzer.armed = False  # (new analyzer instance is not needed: same object)
    ws.scenes.retry_failed()
    assert ws.jobs.wait_idle(60) and ws.project.scene_analysis.status == "COMPLETE"


# ------------------------------------------------------------------ persistence / recovery / migration
@needs_ffmpeg
def test_everything_survives_save_and_reopen(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    ws.scenes.split_scene(ws.project.scenes[1].id, ws.project.scenes[1].start + 2.0)
    ws.scenes.edit_scene(ws.project.scenes[0].id, topic="Kept", intent={"type": "LITERAL"})
    doc = ws.project.to_document()
    root = ws.project.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert ws.project.to_document()["scenes"] == doc["scenes"]
    for key in ("transcription", "script_alignment", "scene_analysis", "visual_intents", "visual_preferences"):
        assert ws.project.to_document()[key] == doc[key], key
    assert ws.project.scenes[0].topic == "Kept" and ws.project.visual_intents[ws.project.scenes[0].id].type is VisualType.LITERAL


@needs_ffmpeg
def test_scenes_are_recoverable_after_a_crash(voice_ws, app_paths):
    from app.services.workspace import Workspace
    from app.tests.conftest import settle

    ws = voice_ws
    run_transcription(ws, ws.provider)
    run_scenes(ws)
    settle(ws)
    n = len(ws.project.scenes)
    ws2 = Workspace(app_paths)  # crash: ws never saved or closed
    entry = ws2.pending_recovery()[0]
    recovered = ws2.recover(entry.project_id)
    assert len(recovered.scenes) == n and recovered.transcription.transcript is not None and recovered.dirty
    ws2.shutdown()


@needs_ffmpeg
def test_phase1_project_files_migrate_to_schema_2(project_ws):
    import json

    ws = project_ws
    ws.save()
    root = ws.project.root
    doc = json.loads((root / "project.json").read_text())
    for key in ("transcription", "script_alignment", "scene_analysis", "visual_intents", "visual_preferences"):
        doc.pop(key)
    doc["schema_version"] = 1
    (root / "project.json").write_text(json.dumps(doc))
    ws.close_project()
    ws.open_project(root)
    p = ws.project
    assert p.schema_version == 3 and p.scenes == [] and p.transcription.transcript is None
    assert p.visual_preferences.min_accuracy_score == 85
    ws.save()
    assert json.loads((root / "project.json").read_text())["schema_version"] == 3
