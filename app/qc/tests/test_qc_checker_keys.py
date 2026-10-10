"""Review regressions for the 16 checkers: a result is only reused while everything the checker reads is unchanged (cache keys), and odd data never crashes one."""

from __future__ import annotations

import math

import pytest

from app.analysis.models import Claim, ClaimType, Entity, EntityType, NumberKind, NumericMention, VisualIntent, VisualType
from app.editing.models import Creator, DecisionType, EditingDecision
from app.media.asset import SourceType
from app.presentation.models import PresentationDecision, PresentationType
from app.qc.ai_editorial_checker import EditorialChecker
from app.qc.asset_checker import AssetChecker
from app.qc.audio_checker import AudioChecker
from app.qc.checker_base import CheckerOutput
from app.qc.context import QCContext
from app.qc.continuity_checker import ContinuityChecker
from app.qc.frame_checker import FrameChecker
from app.qc.geometry import region_rect
from app.qc.issue_model import QCCategory
from app.qc.motion_checker import MotionChecker
from app.qc.pacing_checker import PacingChecker
from app.qc.preflight import PreflightChecker
from app.qc.render_checker import RenderReadinessChecker
from app.qc.scene_checker import SceneChecker
from app.qc.severity import Severity
from app.qc.sync_checker import SyncChecker
from app.qc.tests.conftest import needs_ffmpeg
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, narrate, new_project, qc_ctx, run_checker
from app.qc.timeline_checker import TimelineChecker
from app.qc.transition_checker import TransitionChecker
from app.qc.visual_checker import VisualChecker
from app.research.models import Candidate
from app.timeline.clip import KIND_GRAPHIC, Clip


def world(tmp_path):
    """Three narrated scenes, one AI clip each, with an editing decision, a researched candidate, claims, numbers, entities and an intent: everything the keys below must cover."""
    p = new_project(tmp_path, seconds=30)
    scenes = [add_scene(p, i * 10, (i + 1) * 10, t, importance=0.8, topic=t[:30]) for i, t in enumerate(
        ["Silver supply from mines is falling as demand rises.", "The Federal Reserve raised rates to 5 percent this week.", "Solar panel makers use silver in every cell."])]
    narrate(p)
    assets = [add_asset(p, f"clip{i}.mp4", "video", duration=60) for i in range(3)]
    clips = [add_clip(p, "track_v1", assets[i], i * 10, 10, scene=scenes[i], created_by="AI") for i in range(3)]
    for s in scenes:
        p.visual_intents[s.id] = VisualIntent(s.id, VisualType.LITERAL, primary_subject="silver")
    scenes[1].claims = [Claim("c1", "The Fed raised rates.", ClaimType.FACT, "sent_0001")]
    scenes[1].numbers = [NumericMention("5 percent", NumberKind.PERCENTAGE, 5.0, "sent_0001", True, [])]
    scenes[1].entities = [Entity("Federal Reserve", EntityType.ORGANIZATION)]
    p.editing_decisions["dec_1"] = EditingDecision("dec_1", scenes[0].id, DecisionType.VISUAL_TIMING, "visual:0", clips[0].id, 0.0, 10.0, {}, "Cut at the sentence boundary.", 90.0, Creator.AI)
    clips[0].ai_decision_id = "dec_1"
    p.presentation_decisions["pd_1"] = PresentationDecision("pd_1", scenes[2].id, PresentationType.CAPTION, "caption", clips[2].id, 20.0, 10.0, {}, "Captions on.", 90.0, Creator.AI)
    clips[2].ai_decision_id = "pd_1"
    p.visual_candidates["cand_1"] = Candidate("cand_1", scenes[0].id, SourceType.STOCK_VIDEO, "VIDEO", "silver mine", "", [], 30.0, 1920, 1080, asset_id=assets[0].id)
    return p, scenes, assets, clips


def asset_renamed(p, scenes, assets, clips):
    assets[0].name = "federal_reserve_building.mp4"


def asset_tagged(p, scenes, assets, clips):
    assets[0].extra = {"title": "Federal Reserve", "tags": ["interest", "rates"]}


def asset_source(p, scenes, assets, clips):
    assets[0].source_type = SourceType.SCREENSHOT


def candidate_retitled(p, scenes, assets, clips):
    p.visual_candidates["cand_1"].title = "wildlife lions savanna"


def claim_flipped(p, scenes, assets, clips):
    scenes[1].claims[0].requires_evidence = False


def claims_dropped(p, scenes, assets, clips):
    scenes[1].claims = []


def numbers_dropped(p, scenes, assets, clips):
    scenes[1].numbers = []


def entities_dropped(p, scenes, assets, clips):
    scenes[1].entities = []


def subject_changed(p, scenes, assets, clips):
    p.visual_intents[scenes[1].id].primary_subject = "wildlife"


def intent_changed(p, scenes, assets, clips):
    p.visual_intents[scenes[1].id].type = VisualType.EVIDENCE


def decision_deleted(p, scenes, assets, clips):
    del p.editing_decisions["dec_1"]


def presentation_decision_deleted(p, scenes, assets, clips):
    del p.presentation_decisions["pd_1"]


def decision_edited(p, scenes, assets, clips):
    p.editing_decisions["dec_1"].confidence = 40.0


def next_scene_speaks_earlier(p, scenes, assets, clips):
    for w in p.transcription.transcript.words:
        if scenes[1].start <= w.start < scenes[1].start + 1.0:
            w.start, w.end = w.start - 0.3, w.end - 0.3


def previous_scene_runs_on(p, scenes, assets, clips):
    last = [w for w in p.transcription.transcript.words if scenes[0].start <= w.start < scenes[0].end][-1]
    last.end += 0.5


def transition_frequency(p, scenes, assets, clips):
    p.editing_settings.transition_frequency = 0.0


def declared_gap(p, scenes, assets, clips):
    p.qc_settings.intentional_gaps = [[0.0, 3.0]]


def thumbnail_appears(p, scenes, assets, clips):
    from app.media.thumbnails import ThumbnailService

    path = ThumbnailService.thumbnail_path(p.root, assets[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8thumb")


def caption_style_changed(p, scenes, assets, clips):
    p.caption_settings.style_id = "other_style"


CASES = [  # (checker, what changed, scene-scoped key too?)
    (VisualChecker, asset_renamed, True), (VisualChecker, asset_tagged, True), (VisualChecker, asset_source, True), (VisualChecker, candidate_retitled, True),
    (VisualChecker, claim_flipped, True), (VisualChecker, claims_dropped, True), (VisualChecker, subject_changed, True), (VisualChecker, entities_dropped, True),
    (TimelineChecker, decision_deleted, False), (TimelineChecker, presentation_decision_deleted, False),
    (SceneChecker, claims_dropped, True), (SceneChecker, numbers_dropped, True), (SceneChecker, intent_changed, True),
    (SyncChecker, numbers_dropped, True), (SyncChecker, claims_dropped, True), (SyncChecker, next_scene_speaks_earlier, True), (SyncChecker, previous_scene_runs_on, True),
    (MotionChecker, intent_changed, True), (MotionChecker, decision_edited, True),
    (ContinuityChecker, asset_renamed, False), (ContinuityChecker, asset_tagged, False), (ContinuityChecker, candidate_retitled, False), (ContinuityChecker, entities_dropped, False),
    (ContinuityChecker, thumbnail_appears, False),
    (PacingChecker, decision_edited, False), (PacingChecker, asset_source, False), (PacingChecker, claims_dropped, False),
    (TransitionChecker, transition_frequency, False), (TransitionChecker, numbers_dropped, False),
    (AudioChecker, declared_gap, False), (FrameChecker, declared_gap, False),
    (AssetChecker, thumbnail_appears, False), (AssetChecker, asset_renamed, False),
    (EditorialChecker, claims_dropped, False), (EditorialChecker, asset_renamed, False),
    (RenderReadinessChecker, caption_style_changed, False),
]


@pytest.mark.parametrize("checker,change,scoped", CASES, ids=[f"{c.__name__}-{f.__name__}" for c, f, _ in CASES])
def test_a_checker_result_is_not_reused_after_a_fact_it_reads_changes(tmp_path, checker, change, scoped):
    p, scenes, assets, clips = world(tmp_path)
    ck = checker()
    ctx = qc_ctx(p)
    before = ck.input_hash(ctx)
    before_scenes = {s.id: ck.scene_input_hash(ctx, s.id) for s in scenes} if scoped else {}
    change(p, scenes, assets, clips)
    ctx2 = qc_ctx(p)
    assert ck.input_hash(ctx2) != before, f"{checker.__name__} would reuse a stale result after {change.__name__}"
    if scoped:
        assert any(ck.scene_input_hash(ctx2, sid) != h for sid, h in before_scenes.items()), f"{checker.__name__}: no scene key changed after {change.__name__}"


def test_a_scene_key_covers_the_words_of_the_scenes_beside_it(tmp_path):
    """The sync findings at a scene's edges (a picture that starts before the previous narration ended, stays into the next one) read the neighbours' words."""
    p, scenes, assets, clips = world(tmp_path)
    ck = SyncChecker()
    ctx = qc_ctx(p)
    keys = {s.id: ck.scene_input_hash(ctx, s.id) for s in scenes}
    previous_scene_runs_on(p, scenes, assets, clips)
    ctx2 = qc_ctx(p)
    assert ck.scene_input_hash(ctx2, scenes[1].id) != keys[scenes[1].id]  # the scene after the one whose narration changed
    assert ck.scene_input_hash(ctx2, scenes[2].id) == keys[scenes[2].id]  # a scene two steps away is untouched


def test_continuity_is_rerun_when_the_visual_checker_changes_a_severity(tmp_path):
    """It stands down on scenes the visual checker rates WARNING or above, so a changed rating (same finding, same fingerprint) changes its answer."""
    p, scenes, assets, clips = world(tmp_path)
    ck = ContinuityChecker()

    def key(sev: Severity) -> str:
        ctx = qc_ctx(p)
        ctx.shared["visual"] = CheckerOutput(issues=[ck.issue("visual.mismatch", QCCategory.VISUAL_ACCURACY, sev, "x", scene_id=scenes[1].id)])
        return ck.input_hash(ctx)

    assert key(Severity.NOTICE) != key(Severity.WARNING) and key(Severity.NOTICE) == key(Severity.NOTICE)


def test_audio_and_frames_keys_cover_the_declared_gaps(tmp_path):
    assert "intentional_gaps" in AudioChecker.settings_sections and "intentional_gaps" in FrameChecker.settings_sections


# ---------------------------------------------------------------------------------------------- odd data
def test_non_finite_clip_times_do_not_crash_the_sync_and_pacing_checkers(tmp_path):
    p, scenes, assets, clips = world(tmp_path)
    clips[1].duration = math.inf
    clips[2].timeline_start = math.nan
    for ck in (SyncChecker(), PacingChecker(), MotionChecker(), SceneChecker(), VisualChecker(), ContinuityChecker(), TimelineChecker()):
        run_checker(ck, qc_ctx(p))  # must not raise


def test_a_highlight_that_is_not_a_record_is_ignored(tmp_path):
    for effects in ({"highlight": 5}, {"highlight": ["x"]}, {"highlight": {"region": [0.1, 0.1, 0.2, 0.2]}, "evidence": 7}):
        c = Clip("g", "track_v4", "", 0.0, 1.0, kind=KIND_GRAPHIC, effects=effects)
        region_rect(c)  # must not raise


@needs_ffmpeg
def test_preflight_keeps_its_findings_when_the_render_diagnostics_cannot_read_the_timeline(tmp_path):
    from app.rendering.ffmpeg_service import FFmpegService
    from app.rendering.probe import MediaProbeService

    p, scenes, assets, clips = world(tmp_path)
    clips[1].duration = math.inf
    ff = FFmpegService(lambda: "", lambda: "")
    ctx = QCContext.build(p, p.qc_settings, ffmpeg=ff, probe=MediaProbeService(ff), detach=False)
    out = run_checker(PreflightChecker(), ctx)  # used to raise OverflowError and drop the CRITICAL below
    assert any(i.code == "preflight.no_timeline" and i.severity is Severity.CRITICAL for i in out.issues)
    assert out.metrics["integrity_ok"] is False


# ---------------------------------------------------------------------------------------------- settings sections
def test_every_setting_a_checker_reads_is_part_of_its_cache_key():
    """``settings_sections`` must name every QCSettings field the module reads: a changed threshold must not reuse the result found under the old one."""
    import dataclasses
    import importlib
    import inspect
    import re

    from app.qc.qc_engine import REGISTRY
    from app.qc.settings import QCSettings

    fields = {f.name for f in dataclasses.fields(QCSettings)} - set(QCSettings.PRESENTATION_FIELDS) - {"enabled_checkers"}  # presentation fields are re-derived at once (see QCSettings)
    implied = {"in_intentional_gap": "intentional_gaps", "in_intentional_black": "frames"}
    problems = []
    for cid, module, cls in REGISTRY:
        mod = importlib.import_module(module)
        checker = getattr(mod, cls)()
        src = inspect.getsource(getattr(mod, cls))  # the class itself: the post-render RenderedFileChecker shares a module but not a cache entry
        read = {m for m in re.findall(r"ctx\.settings\.(\w+)", src) if m in fields} | {v for k, v in implied.items() if k in src}
        missing = read - set(checker.settings_sections)
        if missing:
            problems.append(f"{cid} reads {sorted(missing)} but its key covers {checker.settings_sections}")
    assert not problems, problems
