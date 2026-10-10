"""SceneChecker: scene coverage on synthetic projects (no ffmpeg).

Every code the checker emits is triggered by a test and has a clean counterpart; every fix is a navigation route; a user-owned element is reported with its fix disabled;
and the scene-local contract (``ctx.scene_filter``) is checked against a full run.
"""

from __future__ import annotations

import json

from app.analysis.models import Claim, ClaimType, NumberKind, NumericMention, VisualIntent, VisualType
from app.qc.qc_engine import PreviousState, QCEngine
from app.qc.scene_checker import INFO, SceneChecker
from app.qc.settings import QCSettings
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, find, narrate, new_project, qc_ctx, run_checker
from app.research.models import EvidenceKind, VisualAssignment
from app.tests.conftest import needs_ffmpeg
from app.timeline.keyframes import Keyframe
from app.transcription.models import AudioInfo, ProviderInfo, Sentence, Transcript, TranscriptionState, Word

TEXT = "Silver prices rose sharply last week."  # 6 words, one sentence per scene


def build(tmp_path, *, importance: float = 0.5, seconds: float = 30.0, assign: bool = True):
    """30 s project, three 10 s scenes. Each scene has its own approved video (a placeholder file) and one AI-owned clip on V1 that covers it exactly: a clean project."""
    p = new_project(tmp_path, seconds=seconds)
    voice = p.assets.get(p.voice_over.asset_id)
    add_clip(p, "track_a1", voice, 0, seconds, created_by="SYSTEM", slot="voice", audio={"role": "VOICE", "volume": 1.0})
    scenes = [add_scene(p, i * 10, i * 10 + 10, TEXT, importance=importance) for i in range(3)]
    narrate(p)
    assets, clips = [], []
    for i, s in enumerate(scenes):
        a = add_asset(p, f"shot{i}.mp4", "video", duration=30)
        assets.append(a)
        clips.append(add_clip(p, "track_v1", a, i * 10, 10, scene=s, created_by="AI"))
        if assign:
            p.visual_assignments[s.id] = VisualAssignment(s.id, None, a.id, "AI", 0.9, True)
    return p, scenes, assets, clips


def check(p, settings: QCSettings | None = None, **kw):
    return run_checker(SceneChecker(), qc_ctx(p, settings, **kw))


def one(out, code: str):
    found = find(out, code)
    assert len(found) == 1, f"{code}: expected exactly one issue, got {[(i.code, i.scene_id, i.description) for i in out.issues]}"
    return found[0]


def none_of(out, *codes: str):
    got = [(i.code, i.description) for i in out.issues if i.code in codes]
    assert not got, f"unexpected {got}"


def speech(p, spans: list[tuple[float, float]]) -> None:
    """Replace the transcript with words at exactly these (start, end) times (one sentence per scene, so pauses and cut positions are exact)."""
    words = [Word(f"w_{i:04d}", f"word{i}", a, b, 0.95) for i, (a, b) in enumerate(spans)]
    sents = []
    for s in p.scenes:
        ids = [w.word_id for w in words if s.start <= (w.start + w.end) / 2 < s.end]
        if ids:
            ws = [w for w in words if w.word_id in ids]
            sents.append(Sentence(f"sent_{len(sents):04d}", " ".join(w.text for w in ws), ws[0].start, ws[-1].end, ids, 0.95))
            s.sentence_ids = [sents[-1].sentence_id]
    tr = Transcript("tr_test", AudioInfo(p.voice_over.asset_id or "", p.voice_over.duration or 30.0, 48000, 2, None, "voice.wav"), words, sents, ProviderInfo("test"), "punctuation")
    p.transcription = TranscriptionState(tr)


def drop(p, clip) -> None:
    p.timeline.get_track(clip.track_id).clips.remove(clip)


# ------------------------------------------------------------------ clean project, contract, metrics
def test_correct_coverage_has_no_issues_and_reports_rows(tmp_path):
    p, scenes, assets, _clips = build(tmp_path)
    out = check(p)
    assert out.issues == [] and out.complete
    m = out.metrics
    assert m["scenes_checked"] == 3 and m["coverage_ratio_mean"] == 1.0 and m["uncovered_seconds"] == 0 and m["holds_over_limit"] == 0
    row = m["scene_rows"][scenes[1].id]
    assert row["assigned_asset"] == assets[1].id and row["actual_assets"] == [assets[1].id] and row["importance"] == 0.5
    assert row["start"] == 10 and row["end"] == 20 and row["visual_seconds"] == 10 and row["gaps"] == [] and row["shots"] == 1 and row["overlap_seconds"] == 0


def test_checker_declares_its_scene_local_cache_key_and_never_mutates_the_project(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    chk = SceneChecker()
    assert chk.id == "scene" and chk.scene_local and chk.categories[0].value == "SCENE_COVERAGE" and {"timeline", "scenes", "visual"} <= set(chk.domains)
    ctx = qc_ctx(p)
    h = [chk.scene_input_hash(ctx, s.id) for s in scenes]
    assert len(set(h)) == 3 and h == [chk.scene_input_hash(qc_ctx(p), s.id) for s in scenes]
    drop(p, clips[2])
    before = json.dumps(p.timeline.to_dict(), sort_keys=True) + json.dumps(p.visual_assignments[scenes[2].id].to_dict(), sort_keys=True)
    run_checker(chk, qc_ctx(p))
    assert json.dumps(p.timeline.to_dict(), sort_keys=True) + json.dumps(p.visual_assignments[scenes[2].id].to_dict(), sort_keys=True) == before
    after = [chk.scene_input_hash(qc_ctx(p), s.id) for s in scenes]
    assert after[2] != h[2] and after[0] == h[0]  # scene 2 is the dropped clip's neighbour (clips within 0.5 s count), scene 1 is out of reach


def test_hiding_a_track_changes_the_scene_cache_key(tmp_path):
    p, scenes, _a, _c = build(tmp_path)
    h = SceneChecker().scene_input_hash(qc_ctx(p), scenes[0].id)
    p.timeline.get_track("track_v1").hidden = True  # nothing in the clips changed, but nothing is on screen any more
    assert SceneChecker().scene_input_hash(qc_ctx(p), scenes[0].id) != h
    out = check(p)
    assert {i.scene_id for i in find(out, "scene.visual.not_on_timeline")} == set() and len(find(out, "scene.coverage.missing")) == 3


def test_every_code_has_a_title_why_fix_and_lowercase_dotted_name():
    for code, (title, why, fix, impact) in INFO.items():
        assert code == code.lower() and code.startswith("scene.") and title and why and fix and 0 <= impact <= 1


# ------------------------------------------------------------------ missing coverage
def test_scene_without_any_visual_is_an_error_with_a_search_route(tmp_path):
    p, scenes, _a, clips = build(tmp_path, assign=False)
    drop(p, clips[1])
    iss = one(check(p), "scene.coverage.missing")
    assert iss.severity is Severity.ERROR and iss.scene_id == scenes[1].id and "whole narration" in iss.description and "No visual is assigned" in iss.description
    assert iss.fix.kind == "visual.search_again" and iss.fix.route.value in ("NAVIGATE", "RESEARCH") and not iss.fix.safe and iss.fix.needs_confirmation and not iss.auto_fix_safe
    assert 10.0 <= iss.start_time < 11.0 and iss.end_time > 19.0 and 0.9 <= iss.viewer_impact <= 1.0


def test_unapproved_and_missing_media_visuals_route_to_replace(tmp_path):
    p, scenes, assets, clips = build(tmp_path)
    drop(p, clips[0])
    p.visual_assignments[scenes[0].id].approved = False
    assert one(check(p), "scene.coverage.missing").fix.kind == "visual.replace" and "not approved" in one(check(p), "scene.coverage.missing").description
    p.visual_assignments[scenes[0].id].approved = True
    (p.root / p.assets.get(assets[0].id).path).unlink()
    iss = one(check(p), "scene.coverage.missing")
    assert iss.fix.kind == "visual.replace" and "cannot be found" in iss.description
    none_of(check(p), "scene.visual.not_on_timeline")  # the missing file is the asset checker's finding; the coverage hole is reported once


def test_partly_covered_scene_severity_is_scaled_by_importance(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    clips[1].duration = 6.0  # scene 2 is narrated until 19.65 but pictured only until 16: 3.65 s uncovered
    iss = one(check(p), "scene.coverage.missing")
    assert iss.severity is Severity.ERROR and "3.7 s" in iss.description and abs(iss.start_time - 16.0) < 1e-6
    clips[1].duration = 9.3  # 0.35 s uncovered: below the error limit for an average scene
    iss = one(check(p), "scene.coverage.missing")
    assert iss.severity is Severity.WARNING
    clips[1].duration = 8.8  # 0.85 s uncovered
    assert one(check(p), "scene.coverage.missing").severity is Severity.WARNING
    scenes[1].importance = 1.0  # the same 0.85 s in the most important scene: the error limit shrinks to 0.5 s
    assert one(check(p), "scene.coverage.missing").severity is Severity.ERROR
    scenes[1].importance = 0.0  # ... and grows to 1.5 s for the least important one
    clips[1].duration = 8.2  # 1.45 s uncovered
    assert one(check(p), "scene.coverage.missing").severity is Severity.WARNING
    scenes[1].importance = 0.2
    clips[1].duration = 9.4  # 0.25 s in a minor scene is only worth a notice
    assert one(check(p), "scene.coverage.missing").severity is Severity.NOTICE


def test_small_hole_within_the_covered_ratio_is_not_reported(tmp_path):
    p, _s, _a, clips = build(tmp_path)
    clips[1].duration = 9.62  # 0.03 s uncovered: under one frame at 30 fps and under 2% of the narration
    none_of(check(p), "scene.coverage.missing")
    s = QCSettings()
    s.coverage.min_covered_ratio = 0.9
    clips[1].duration = 9.0  # 0.65 s uncovered, 6.6% of 9.55 s: allowed by a 90% ratio
    none_of(check(p, s), "scene.coverage.missing")
    s.coverage.min_covered_ratio = 0.98
    assert find(check(p, s), "scene.coverage.missing")


def test_declared_intentional_gap_is_not_reported(tmp_path):
    p, _s, _a, clips = build(tmp_path, assign=False)
    drop(p, clips[1])
    s = QCSettings()
    s.intentional_gaps = [[10.0, 20.0]]
    none_of(check(p, s), "scene.coverage.missing")
    s.intentional_gaps = [[10.0, 14.0]]  # only part of it declared: the rest is still a hole
    iss = one(check(p, s), "scene.coverage.missing")
    assert iss.start_time >= 14.0 - 1e-6


def test_a_deliberate_pause_in_the_narration_is_not_a_hole(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    speech(p, [(0.1, 0.6), (8.0, 8.6), (10.1, 10.6), (16.5, 17.0), (20.1, 20.6), (21.0, 21.5)])  # scene 1: 7.4 s of silence in the middle; scene 2: speech again at 16.5
    clips[0].duration = 2.0
    clips[0].timeline_start = 0.0
    add_clip(p, "track_v1", p.assets.get(clips[0].asset_id), 8.0, 2.0, scene=scenes[0], created_by="AI", source_in=12.0)
    none_of(check(p), "scene.coverage.missing")  # the picture is away between 2 s and 8 s, but nobody speaks then
    clips[1].duration = 5.0  # scene 2 pictured until 15; the word at 16.5 is 5.9 s after the previous one: its own stretch
    iss = one(check(p), "scene.coverage.missing")
    assert iss.scene_id == scenes[1].id and iss.start_time >= 16.0 - 1e-6  # only the narrated stretch counts, not the silent 15-16.5


def test_a_hidden_track_or_transparent_clip_shows_nothing(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    clips[0].opacity = 0.0
    out = check(p)
    assert one(out, "scene.coverage.missing").scene_id == scenes[0].id
    clips[0].opacity = 1.0
    none_of(check(p), "scene.coverage.missing")


# ------------------------------------------------------------------ skipped scenes
def test_skipped_scene_without_a_visual_is_info_only(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    drop(p, clips[1])
    p.visual_assignments[scenes[1].id].skipped = True
    out = check(p)
    iss = one(out, "scene.coverage.skipped")
    assert iss.severity is Severity.INFO and iss.scene_id == scenes[1].id and iss.fix is None
    assert [i.code for i in out.issues if i.scene_id == scenes[1].id] == ["scene.coverage.skipped"]  # nothing stronger: no missing coverage, no "not on the timeline"
    none_of(out, "scene.coverage.missing", "scene.visual.not_on_timeline")


def test_skipped_scene_that_is_covered_anyway_has_no_issue(tmp_path):
    p, scenes, _a, _c = build(tmp_path)
    p.visual_assignments[scenes[1].id].skipped = True
    assert check(p).issues == []


# ------------------------------------------------------------------ approved visual vs timeline
def test_approved_visual_missing_from_the_timeline_is_an_error(tmp_path):
    p, scenes, assets, clips = build(tmp_path)
    drop(p, clips[1])
    iss = one(check(p), "scene.visual.not_on_timeline")
    assert iss.severity is Severity.ERROR and iss.scene_id == scenes[1].id and "shot1.mp4" in iss.description and "shows nothing" in iss.description and "no picture" in iss.description
    assert iss.fix.kind == "open.scene" and iss.viewer_impact == 1.0
    none_of(check(p), "scene.coverage.missing")  # one root cause, one issue
    other = add_asset(p, "other.mp4", "video", duration=30)
    add_clip(p, "track_v1", other, 10, 10, scene=scenes[1], created_by="AI")  # something else covers the scene
    iss = one(check(p), "scene.visual.not_on_timeline")
    assert iss.severity is Severity.ERROR and "other.mp4" in iss.description and "instead" in iss.description and iss.viewer_impact < 1.0
    none_of(check(p), "scene.coverage.missing")


def test_a_visual_you_replaced_yourself_is_only_a_notice(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    drop(p, clips[1])
    mine = add_asset(p, "mine.mp4", "video", duration=30)
    add_clip(p, "track_v1", mine, 10, 10, scene=scenes[1])  # created_by defaults to USER
    iss = one(check(p), "scene.visual.not_on_timeline")
    assert iss.severity is Severity.NOTICE and iss.locked and not iss.auto_fix_available and "edited or created by you" in iss.fix_blocked_reason


def test_placed_approved_visual_is_clean_even_when_it_is_only_a_part_of_the_scene(tmp_path):
    p, scenes, assets, clips = build(tmp_path)
    clips[1].duration = 9.5
    none_of(check(p), "scene.visual.not_on_timeline")  # it is on the timeline (the shortfall is the coverage's business)
    p.visual_assignments[scenes[1].id].approved = False
    clips[1].asset_id = ""  # an unapproved choice is not "approved but missing"
    none_of(check(p), "scene.visual.not_on_timeline")


# ------------------------------------------------------------------ excessive hold
def long_scene(tmp_path, *, seconds: float = 20.0, **kw):
    """One scene of `seconds`, narrated throughout, with one video clip of the same length."""
    p = new_project(tmp_path, seconds=seconds)
    s = add_scene(p, 0, seconds, " ".join(["word"] * int(seconds * 2)), importance=kw.pop("importance", 0.5))
    narrate(p)
    a = add_asset(p, "long.mp4", "video", duration=60)
    c = add_clip(p, "track_v1", a, 0, seconds, scene=s, created_by=kw.pop("created_by", "AI"))
    return p, s, a, c


def test_video_held_longer_than_the_limit_is_a_notice_then_a_warning(tmp_path):
    p, s, _a, c = long_scene(tmp_path, seconds=14.0)
    iss = one(check(p), "scene.hold.excessive")
    assert iss.severity is Severity.NOTICE and iss.timeline_item_id == c.id and "14.0 s" in iss.description and iss.fix.kind == "open.scene" and not iss.fix.safe
    assert abs(iss.start_time) < 1e-6 and abs(iss.end_time - 14.0) < 1e-6
    p2, s2, _a2, _c2 = long_scene(tmp_path / "w", seconds=20.0)
    assert one(check(p2), "scene.hold.excessive").severity is Severity.WARNING  # 20 s > 1.5 x 12 s
    p3, _s3, _a3, _c3 = long_scene(tmp_path / "ok", seconds=11.5)
    none_of(check(p3), "scene.hold.excessive")


def test_hold_limits_come_from_the_settings_and_stills_have_their_own(tmp_path):
    p, _s, _a, c = long_scene(tmp_path, seconds=10.0)
    none_of(check(p), "scene.hold.excessive")
    st = QCSettings()
    st.coverage.max_hold_seconds = 8.0
    assert one(check(p, st), "scene.hold.excessive").metrics["limit_seconds"] == 8.0
    still = add_asset(p, "chart.png", "image", duration=None)
    drop(p, c)
    c2 = add_clip(p, "track_v3", still, 0, 10.0, scene=p.scenes[0], created_by="AI")  # a still: 9 s limit
    iss = one(check(p), "scene.hold.excessive")
    assert iss.metrics["limit_seconds"] == 9.0 and "a still" in iss.description
    c2.keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 10.0, 1.2)]  # a slow push-in keeps a still alive: the video limit (12 s) applies
    none_of(check(p, QCSettings()), "scene.hold.excessive")


def test_deliberate_hold_on_evidence_is_not_flagged(tmp_path):
    p, s, _a, c = long_scene(tmp_path, seconds=14.0)
    p.visual_intents[s.id] = VisualIntent(s.id, VisualType.EVIDENCE)
    shot = add_asset(p, "doc.png", "image", duration=None)
    drop(p, c)
    still = add_clip(p, "track_v3", shot, 0, 14.0, scene=s, created_by="AI")  # a 14 s still on an EVIDENCE scene: the viewer has to read it
    none_of(check(p), "scene.hold.excessive")
    still.asset_id = add_asset(p, "other.png", "image", duration=None).id
    p.visual_intents[s.id] = VisualIntent(s.id, VisualType.LITERAL)
    assert find(check(p), "scene.hold.excessive")  # the same hold on an ordinary picture is flagged
    still.effects = {"highlight": {"region": [0.1, 0.1, 0.5, 0.2]}}
    none_of(check(p), "scene.hold.excessive")  # ... unless the clip carries an evidence highlight


def test_a_cutaway_breaks_a_long_base_clip_into_two_acceptable_holds(tmp_path):
    p, s, _a, base = long_scene(tmp_path, seconds=20.0)
    broll = add_asset(p, "broll.mp4", "video", duration=30)
    add_clip(p, "track_v2", broll, 8, 4, scene=s, created_by="AI")  # opaque B-roll over the middle: the picture changes twice
    out = check(p)
    none_of(out, "scene.hold.excessive")
    assert out.metrics["scene_rows"][s.id]["shots"] == 3 and out.metrics["scene_rows"][s.id]["overlap_seconds"] == 4.0


def test_adjacent_pieces_of_one_continuous_video_are_one_hold(tmp_path):
    p, s, a, c = long_scene(tmp_path, seconds=20.0)
    c.duration, c.source_out = 8.0, 8.0
    add_clip(p, "track_v1", a, 8, 7.0, scene=s, created_by="AI", source_in=8.0)  # continues the same footage: split, not cut
    add_clip(p, "track_v1", a, 15, 5.0, scene=s, created_by="AI", source_in=40.0)  # jumps elsewhere in the media: a real cut
    out = check(p)
    assert one(out, "scene.hold.excessive").metrics["hold_seconds"] == 15.0
    assert out.metrics["scene_rows"][s.id]["shots"] == 2


def test_a_hold_across_two_scenes_is_reported_once_in_the_scene_it_starts_in(tmp_path):
    p = new_project(tmp_path, seconds=20.0)
    s1, s2 = add_scene(p, 0, 10, TEXT), add_scene(p, 10, 20, TEXT)
    narrate(p)
    a = add_asset(p, "long.mp4", "video", duration=60)
    add_clip(p, "track_v1", a, 0, 20.0, scene=s1, created_by="AI")  # 20 s hold: each scene alone shows only 10 s of it
    out = check(p)
    iss = one(out, "scene.hold.excessive")
    assert iss.scene_id == s1.id and abs(iss.end_time - 20.0) < 1e-6
    only2 = check(p, scene_filter={s2.id})
    none_of(only2, "scene.hold.excessive")


def test_hold_on_a_user_owned_clip_is_reported_with_the_fix_disabled(tmp_path):
    p, _s, _a, c = long_scene(tmp_path, seconds=14.0, created_by="USER")
    iss = one(check(p), "scene.hold.excessive")
    assert iss.locked and not iss.auto_fix_available and not iss.auto_fix_safe and "edited or created by you" in iss.fix_blocked_reason and iss.fix is not None
    c.created_by, c.locked = "AI", True
    assert "locked by you" in one(check(p), "scene.hold.excessive").fix_blocked_reason


def test_navigation_fix_permission_never_disables_the_fix(tmp_path):
    p, _s, _a, _c = long_scene(tmp_path, seconds=14.0)
    st = QCSettings()
    st.fix_permissions["open.scene"] = "never"
    iss = one(check(p, st), "scene.hold.excessive")
    assert not iss.auto_fix_available and iss.fix_blocked_reason == "Disabled in QC settings"


# ------------------------------------------------------------------ fragmentation
def cut_scene(tmp_path, cuts: list[float], *, seconds: float = 10.0):
    """One scene [0, seconds] with a single 8-word sentence spoken from 0.5 to 7.5 and one clip per interval between the cut times."""
    p = new_project(tmp_path, seconds=seconds)
    s = add_scene(p, 0, seconds, "one two three four five six seven eight", importance=0.5)
    speech(p, [(0.5 + i * 0.9, 0.5 + i * 0.9 + 0.8) for i in range(8)])
    edges = [0.0, *cuts, seconds]
    clips = [add_clip(p, "track_v1", add_asset(p, f"s{i}.mp4", "video", duration=30), a, b - a, scene=s, created_by="AI") for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    return p, s, clips


def test_too_many_cuts_inside_one_sentence(tmp_path):
    p, s, clips = cut_scene(tmp_path, [2.0, 3.5, 5.0, 6.5])  # four cuts inside a sentence that may take two
    iss = one(check(p), "scene.fragmentation.cuts")
    assert iss.severity is Severity.WARNING and iss.scene_id == s.id and "cut 4 times" in iss.description and iss.metrics["cuts"] == 4 and iss.fix.kind == "open.scene"
    p2, _s2, _c2 = cut_scene(tmp_path / "three", [2.0, 4.0, 6.0])
    assert one(check(p2), "scene.fragmentation.cuts").severity is Severity.NOTICE  # one cut over the limit
    p3, _s3, _c3 = cut_scene(tmp_path / "two", [3.0, 6.0])
    none_of(check(p3), "scene.fragmentation.cuts")  # two cuts is the limit
    st = QCSettings()
    st.coverage.max_cuts_per_sentence = 4.0
    none_of(check(p, st), "scene.fragmentation.cuts")


def test_a_long_sentence_may_carry_proportionally_more_cuts(tmp_path):
    p = new_project(tmp_path, seconds=10.0)
    s = add_scene(p, 0, 10, " ".join(f"w{i}" for i in range(28)), importance=0.5)
    speech(p, [(0.2 + i * 0.33, 0.2 + i * 0.33 + 0.3) for i in range(28)])  # 28 words in one sentence: allowance is 2 x 2
    edges = [0.0, 2.0, 3.5, 5.0, 6.5, 10.0]
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        add_clip(p, "track_v1", add_asset(p, f"s{i}.mp4", "video", duration=30), a, b - a, scene=s, created_by="AI")
    none_of(check(p), "scene.fragmentation.cuts")  # 4 cuts <= 4


def test_cluster_of_very_short_shots(tmp_path):
    p, s, _c = cut_scene(tmp_path, [4.0, 4.5, 5.0, 5.5, 6.0])  # three 0.5 s shots in a row between longer ones
    iss = one(check(p), "scene.fragmentation.short_shots")
    assert iss.severity is Severity.WARNING and iss.metrics["shots"] == 4 and "shorter than 0.8 s" in iss.description and abs(iss.start_time - 4.0) < 1e-6 and abs(iss.end_time - 6.0) < 1e-6
    p2, _s2, _c2 = cut_scene(tmp_path / "two", [4.0, 4.5, 5.0, 8.0])  # only two short shots: a quick beat, not a cluster
    none_of(check(p2), "scene.fragmentation.short_shots")
    p3, _s3, _c3 = cut_scene(tmp_path / "ok", [4.0, 5.0, 6.0, 7.0])  # one-second shots are fine
    none_of(check(p3), "scene.fragmentation.short_shots")
    st = QCSettings()
    st.coverage.min_shot_seconds = 0.4
    none_of(check(p, st), "scene.fragmentation.short_shots")


def test_a_shot_cut_by_the_scene_edge_is_not_short(tmp_path):
    p = new_project(tmp_path, seconds=20.0)
    s1, s2 = add_scene(p, 0, 10, TEXT), add_scene(p, 10, 20, TEXT)
    narrate(p)
    a = add_asset(p, "a.mp4", "video", duration=60)
    add_clip(p, "track_v1", a, 0, 10.2, scene=s1, created_by="AI")  # 0.2 s of it reach into scene 2
    add_clip(p, "track_v1", add_asset(p, "b.mp4", "video", duration=60), 10.2, 9.8, scene=s2, created_by="AI")
    none_of(check(p), "scene.fragmentation.short_shots")


# ------------------------------------------------------------------ under-supported statements
def claim(sid: str = "sent_0000") -> Claim:
    return Claim(f"{sid}_c0", "Silver production fell by a third", ClaimType.FACT, sid, True)


def test_important_claim_with_only_a_decorative_visual_is_a_judgement_warning(tmp_path):
    p, scenes, _a, clips = build(tmp_path, importance=0.8)
    scenes[1].claims = [claim()]
    iss = one(check(p), "scene.support.under_supported")
    assert iss.severity is Severity.WARNING and iss.scene_id == scenes[1].id and iss.confidence < 100 and iss.timeline_item_id == clips[1].id
    assert "Potential under-support detected" in iss.description and "Review recommended" in iss.description and "true" not in iss.description.lower() and "false" not in iss.description.lower()
    assert iss.fix.kind == "visual.replace" and iss.fix.route.value == "NAVIGATE" and not iss.fix.safe and iss.detection_source.startswith("deterministic")


def test_spoken_numbers_and_evidence_intent_trigger_the_same_check(tmp_path):
    p, scenes, _a, _c = build(tmp_path, importance=0.8)
    scenes[0].numbers = [NumericMention("4%", NumberKind.PERCENTAGE, 4.0, "", False, ["w_0001"])]
    scenes[2].numbers = [NumericMention("three", NumberKind.QUANTITY, 3.0, "", True, ["w_0002"])]  # a quantity is too weak a statement
    p.visual_intents[scenes[1].id] = VisualIntent(scenes[1].id, VisualType.DATA)
    out = check(p)
    flagged = {i.scene_id: i for i in find(out, "scene.support.under_supported")}
    assert set(flagged) == {scenes[0].id, scenes[1].id}
    assert flagged[scenes[0].id].confidence > flagged[scenes[1].id].confidence  # a spoken number is firmer than an intent alone
    assert "4%" in flagged[scenes[0].id].description and "data visual intent" in flagged[scenes[1].id].description


def test_support_is_clean_with_a_text_treatment_an_evidence_visual_or_low_importance(tmp_path):
    p, scenes, assets, clips = build(tmp_path, importance=0.8)
    for s in scenes:
        s.claims = [claim()]
    assert len(find(check(p), "scene.support.under_supported")) == 3
    add_clip(p, "track_v5", None, 11, 4, kind="text", text={"content": "Output -33%"}, created_by="AI")  # a text treatment on V5 during scene 2
    add_clip(p, "track_v4", None, 0, 4, kind="graphic", effects={"highlight": {"region": [0.1, 0.1, 0.4, 0.2]}}, created_by="AI")  # a highlight on V4 during scene 1
    p.visual_assignments[scenes[2].id].evidence_kind = EvidenceKind.EVIDENCE
    p.assets.get(assets[2].id)  # (kept: scene 3's visual becomes evidence below)
    clips[2].effects = {"evidence": {"region": [0.1, 0.1, 0.5, 0.2]}}
    assert check(p).issues == []
    p.timeline.get_track("track_v5").hidden = True  # a hidden text track shows nothing
    assert [i.scene_id for i in find(check(p), "scene.support.under_supported")] == [scenes[1].id]
    p.timeline.get_track("track_v5").hidden = False
    scenes[1].importance = 0.69  # below the important-scene threshold
    none_of(check(p), "scene.support.under_supported")
    scenes[1].importance = 0.7
    st = QCSettings()
    st.coverage.important_scene = 0.9
    none_of(check(p, st), "scene.support.under_supported")


def test_an_empty_text_clip_is_no_treatment_and_opinions_are_not_claims(tmp_path):
    p, scenes, _a, _c = build(tmp_path, importance=0.8)
    scenes[1].claims = [claim()]
    add_clip(p, "track_v5", None, 11, 4, kind="text", text={"content": "  "}, created_by="AI")
    assert len(find(check(p), "scene.support.under_supported")) == 1
    scenes[1].claims = [Claim("c1", "I like silver", ClaimType.OPINION, "sent_0000", False)]
    none_of(check(p), "scene.support.under_supported")


def test_a_scene_without_picture_is_not_also_called_under_supported(tmp_path):
    p, scenes, _a, clips = build(tmp_path, importance=0.8)
    scenes[1].claims = [claim()]
    drop(p, clips[1])
    out = check(p)
    assert find(out, "scene.visual.not_on_timeline")
    none_of(out, "scene.support.under_supported")


def test_under_supported_on_a_user_owned_visual_is_reported_with_the_fix_disabled(tmp_path):
    p, scenes, _a, clips = build(tmp_path, importance=0.8)
    scenes[1].claims = [claim()]
    clips[1].created_by = "USER"
    iss = one(check(p), "scene.support.under_supported")
    assert iss.locked and not iss.auto_fix_available and "edited or created by you" in iss.fix_blocked_reason


# ------------------------------------------------------------------ scene-local contract
def test_scene_filter_returns_only_that_scenes_issues_and_the_same_ones_as_a_full_run(tmp_path):
    p, scenes, _a, clips = build(tmp_path, importance=0.8)
    for s in scenes:
        s.claims = [claim()]
    clips[0].duration = 6.0  # a hole in scene 1
    drop(p, clips[2])  # scene 3 not on the timeline
    full = check(p)
    assert {i.scene_id for i in full.issues} == {scenes[0].id, scenes[1].id, scenes[2].id} and all(i.scene_id for i in full.issues)
    for s in scenes:
        part = check(p, scene_filter={s.id})
        assert {i.scene_id for i in part.issues} == {s.id}
        assert sorted((i.code, i.fingerprint, i.severity.value) for i in part.issues) == sorted((i.code, i.fingerprint, i.severity.value) for i in full.issues if i.scene_id == s.id)
        assert part.metrics["scenes_checked"] == 1 and list(part.metrics["scene_rows"]) == [s.id]
    two = check(p, scene_filter={scenes[0].id, scenes[2].id})
    assert {i.scene_id for i in two.issues} == {scenes[0].id, scenes[2].id}


def test_findings_are_deterministic_and_fingerprints_survive_a_rerun(tmp_path):
    p, scenes, _a, clips = build(tmp_path)
    drop(p, clips[1])
    a, b = check(p), check(p)
    assert [i.fingerprint for i in a.issues] == [i.fingerprint for i in b.issues] and all(i.fingerprint for i in a.issues)
    assert [i.issue_id for i in a.issues] != [i.issue_id for i in b.issues]
    clips[0].duration = 8.0  # a different finding in another scene does not change this one's fingerprint
    c = check(p)
    assert one(c, "scene.visual.not_on_timeline").fingerprint == one(a, "scene.visual.not_on_timeline").fingerprint


def test_progress_is_reported_and_cancellation_is_honoured(tmp_path):
    import pytest

    from app.qc.context import QCCancelled

    p, _s, _a, _c = build(tmp_path)
    calls = []
    SceneChecker().run(qc_ctx(p), lambda f, m: calls.append(f))
    assert calls and calls[-1] == 1.0 and calls == sorted(calls)
    ctx = qc_ctx(p)
    ctx.cancel.set()
    with pytest.raises(QCCancelled):
        SceneChecker().run(ctx, lambda f, m: None)


def test_project_without_scenes_or_transcript_is_quietly_empty(tmp_path):
    p = new_project(tmp_path)
    out = check(p)
    assert out.issues == [] and out.metrics["scenes_checked"] == 0 and out.metrics["coverage_ratio_mean"] == 1.0


def test_incremental_engine_run_equals_a_full_run(tmp_path):
    """The engine reuses unchanged scenes by ``scene_input_hash``: after any edit the merged result must equal a from-scratch run (this is what the track-flag / protection keys are for)."""
    p, scenes, _a, clips = build(tmp_path, importance=0.8)
    eng = QCEngine([SceneChecker()])
    first = eng.run(qc_ctx(p))
    assert first.run.issues == []

    def fingerprints(res):
        return sorted((i.code, i.fingerprint, i.severity.value) for i in res.run.issues)

    prev = PreviousState(first.run.issues, first.cache)
    clips[0].duration = 4.0  # a hole at the end of scene 1 (clips within 0.5 s of the next scene may touch it too, so only scene 3 is guaranteed to be reused)
    second = eng.run(qc_ctx(p), previous=prev)
    assert second.run.checkers["scene"].reused_scenes >= 1 and any(i.scene_id == scenes[0].id for i in second.run.issues)
    assert fingerprints(second) == fingerprints(eng.run(qc_ctx(p), use_cache=False))
    prev = PreviousState(second.run.issues, second.cache)
    p.timeline.get_track("track_v1").hidden = True  # no clip changed, yet nothing is on screen any more: no scene may be served from the cache
    third = eng.run(qc_ctx(p), previous=prev)
    assert third.run.checkers["scene"].reused_scenes == 0 and len(find(third.outputs["scene"], "scene.coverage.missing")) == 3
    assert fingerprints(third) == fingerprints(eng.run(qc_ctx(p), use_cache=False))
    p.timeline.get_track("track_v1").hidden = False
    clips[1].created_by = "USER"  # ownership alone changes a finding's fix flags
    clips[1].duration = 4.0
    fourth = eng.run(qc_ctx(p), previous=PreviousState(third.run.issues, third.cache))
    assert fingerprints(fourth) == fingerprints(eng.run(qc_ctx(p), use_cache=False))


# ------------------------------------------------------------------ real pipeline (needs ffmpeg for the media behind the project)
@needs_ffmpeg
def test_real_ai_edited_project_is_clean_and_a_removed_scene_visual_is_found(pres_ws):
    p = pres_ws.project
    out = check(p)
    assert out.issues == [] and out.metrics["scenes_checked"] == len(p.scenes) >= 10 and out.metrics["coverage_ratio_mean"] == 1.0  # the assembler's own output is not flagged
    target = p.scenes[5]
    for t in p.timeline.tracks:
        if t.kind.value in ("video", "image"):
            t.clips[:] = [c for c in t.clips if not (c.timeline_start >= target.start - 1e-6 and c.timeline_end <= target.end + 1e-6)]  # direct edit of the live project: this test never saves
    out = check(p)
    mine = [i for i in out.issues if i.scene_id == target.id]
    assert [i.code for i in mine if i.severity is Severity.ERROR] == ["scene.visual.not_on_timeline"]
    part = check(p, scene_filter={target.id})
    assert {i.scene_id for i in part.issues} == {target.id} and sorted(i.fingerprint for i in part.issues) == sorted(i.fingerprint for i in mine)


def test_a_hole_the_timeline_checker_already_reports_is_reported_once(tmp_path):
    """The same stretch was an ERROR twice: ``timeline.gap.unintended`` and ``scene.coverage.missing``. The timeline finding stays (it carries the fix); a scene without a usable visual keeps
    its own finding, because the cause and the way out (choose a visual) are only known here."""
    from app.qc.timeline_checker import TimelineChecker

    p, scenes, assets, clips = build(tmp_path)
    clips[1].duration = 4.0  # scene 2: its approved visual is on the timeline, but only for the first 4 of its 10 s

    def both():
        ctx = qc_ctx(p)
        ctx.shared["timeline"] = run_checker(TimelineChecker(), ctx)
        return ctx.shared["timeline"], run_checker(SceneChecker(), ctx)

    tl, out = both()
    assert find(tl, "timeline.gap.unintended") and find(out, "scene.coverage.missing") == []
    assert out.metrics["scene_rows"][scenes[1].id]["uncovered_seconds"] > 5  # still measured
    assert find(check(p), "scene.coverage.missing")  # run on its own it speaks up as before
    p.visual_assignments[scenes[1].id] = VisualAssignment(scenes[1].id, None, assets[1].id, "AI", 0.9, False)  # chosen but never approved
    tl, out = both()
    assert find(tl, "timeline.gap.unintended") and find(out, "scene.coverage.missing")
