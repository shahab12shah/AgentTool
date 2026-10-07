"""TimelineChecker: integrity of the whole timeline on synthetic projects (no ffmpeg).

Every code the checker emits is triggered by a test and has a clean counterpart; every fix offered has a safe/confirmation spec and a user-lock case.
"""

from __future__ import annotations

import json
import math

import pytest

from app.editing.models import DecisionType, EditingDecision
from app.presentation import animation
from app.qc.settings import QCSettings
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, find, narrate, new_project, qc_ctx, run_checker
from app.qc.timeline_checker import INFO, TimelineChecker
from app.timeline.keyframes import Keyframe
from app.tests.conftest import needs_ffmpeg
from app.timeline.track import Track, TrackKind


def build(tmp_path, seconds: float = 30.0, fps: int = 30):
    """30 s project: a voice-over clip on A1, three 10 s scenes, one AI-owned video clip per scene on V1 (clean)."""
    p = new_project(tmp_path, seconds=seconds, fps=fps)
    voice = p.assets.get(p.voice_over.asset_id)
    vclip = add_clip(p, "track_a1", voice, 0, seconds, created_by="SYSTEM", slot="voice", audio={"role": "VOICE", "volume": 1.0})
    scenes = [add_scene(p, i * 10, i * 10 + 10, "Silver prices rose sharply last week and then fell again.", importance=0.5) for i in range(3)]
    narrate(p)
    video = add_asset(p, "city.mp4", "video", duration=12)
    clips = [add_clip(p, "track_v1", video, i * 10, 10, scene=s, created_by="AI") for i, s in enumerate(scenes)]
    return p, scenes, video, clips, vclip


def check(p, settings: QCSettings | None = None, **kw):
    return run_checker(TimelineChecker(), qc_ctx(p, settings, **kw))


def one(out, code: str):
    found = find(out, code)
    assert len(found) == 1, f"{code}: expected exactly one issue, got {[(i.code, i.description) for i in out.issues]}"
    return found[0]


def none_of(out, *codes: str):
    got = [i.code for i in out.issues if i.code in codes]
    assert not got, f"unexpected {got}: {[i.description for i in out.issues if i.code in codes]}"


# ------------------------------------------------------------------ clean project, contract, metrics
def test_clean_timeline_has_no_issues_and_reports_metrics(tmp_path):
    p, *_ = build(tmp_path)
    out = check(p)
    assert out.issues == [] and out.complete
    m = out.metrics
    assert m["clips_by_track_kind"]["video"] == 3 and m["clips_by_track_kind"]["audio"] == 1 and m["clip_count"] == 4
    assert m["total_duration"] == 30 and m["visual_covered_ratio"] == 1.0 and m["visual_covered_ratio_while_speaking"] == 1.0 and m["gap_count"] == 0


def test_checker_declares_its_cache_key_and_never_mutates_the_project(tmp_path):
    p, _scenes, _video, clips, _v = build(tmp_path)
    chk = TimelineChecker()
    assert chk.id == "timeline" and not chk.scene_local and {"timeline", "scenes", "assets"} <= set(chk.domains)
    ctx = qc_ctx(p)
    h = chk.input_hash(ctx)
    before = json.dumps(p.timeline.to_dict(), sort_keys=True)
    clips[0].opacity = 3.0  # make it find something so the fix paths run too
    before = json.dumps(p.timeline.to_dict(), sort_keys=True)
    run_checker(chk, qc_ctx(p))
    assert json.dumps(p.timeline.to_dict(), sort_keys=True) == before
    assert chk.input_hash(qc_ctx(p)) != h  # a changed clip changes the key; an unchanged project keeps it
    assert chk.input_hash(qc_ctx(p)) == chk.input_hash(qc_ctx(p))


def test_every_emitted_code_has_a_title_why_and_lowercase_dotted_name():
    for code, (title, why, fix, impact) in INFO.items():
        assert code == code.lower() and code.startswith("timeline.") and title and why and fix and 0 <= impact <= 1


# ------------------------------------------------------------------ whole-timeline CRITICALs
def test_empty_timeline_is_critical_three_ways(tmp_path):
    p = new_project(tmp_path)
    out = check(p)
    for code in ("timeline.empty.visual", "timeline.empty.voice", "timeline.duration.invalid"):
        assert one(out, code).severity is Severity.CRITICAL
    none_of(out, "timeline.gap.unintended")  # an empty timeline is one CRITICAL, not one giant gap


def test_missing_voice_or_picture_alone_is_critical(tmp_path):
    p, _s, _v, clips, vclip = build(tmp_path)
    p.timeline.get_track("track_a1").clips.remove(vclip)
    out = check(p)
    assert one(out, "timeline.empty.voice").severity is Severity.CRITICAL
    none_of(out, "timeline.empty.visual", "timeline.duration.invalid")
    p, _s, _v, clips, vclip = build(tmp_path / "b")
    p.timeline.get_track("track_v1").clips.clear()
    out = check(p)
    assert one(out, "timeline.empty.visual").severity is Severity.CRITICAL
    none_of(out, "timeline.empty.voice")


# ------------------------------------------------------------------ invalid / empty clips and the safe fixes
def test_zero_duration_clip_is_critical_with_a_safe_remove_fix(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    c = add_clip(p, "track_v2", video, 5, 0.0, created_by="AI")
    iss = one(check(p), "timeline.clip.zero_duration")
    assert iss.severity is Severity.CRITICAL and iss.timeline_item_id == c.id
    assert iss.fix.kind == "clip.remove_empty" and iss.fix.safe and not iss.fix.needs_confirmation and iss.auto_fix_available and iss.auto_fix_safe


def test_short_but_valid_clip_is_not_zero_duration(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 5, 0.5, created_by="AI", opacity=0.6)  # an overlay layer of half a second
    none_of(check(p), "timeline.clip.zero_duration", "timeline.clip.empty_content")


def test_user_owned_empty_clip_is_reported_but_its_fix_is_disabled(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 5, 0.0)  # created_by defaults to USER
    iss = one(check(p), "timeline.clip.zero_duration")
    assert iss.locked and not iss.auto_fix_available and not iss.auto_fix_safe and "edited or created by you" in iss.fix_blocked_reason and iss.fix is not None


def test_locked_track_and_locked_scene_also_disable_the_fix(tmp_path):
    p, scenes, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 5, 0.0, created_by="AI")
    p.timeline.get_track("track_v2").locked = True
    assert "locked" in one(check(p), "timeline.clip.zero_duration").fix_blocked_reason
    p.timeline.get_track("track_v2").locked = False
    p.timeline.get_track("track_v2").clips[0].scene_id = scenes[0].id
    p.timeline_generation.locked_scenes.append(scenes[0].id)
    iss = one(check(p), "timeline.clip.zero_duration")
    assert not iss.auto_fix_available and "scene is locked" in iss.fix_blocked_reason


def test_fix_permission_never_disables_the_fix(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 5, 0.0, created_by="AI")
    s = QCSettings()
    s.fix_permissions["clip.remove_empty"] = "never"
    iss = one(check(p, s), "timeline.clip.zero_duration")
    assert not iss.auto_fix_available and iss.fix_blocked_reason == "Disabled in QC settings"


def test_empty_text_and_assetless_media_are_empty_content(tmp_path):
    p, *_ = build(tmp_path)
    t = add_clip(p, "track_v5", None, 2, 3, kind="text", text={"content": "  "}, created_by="AI")
    m = add_clip(p, "track_v2", None, 12, 3, created_by="AI", opacity=0.5)  # a media clip with no asset at all
    out = check(p)
    assert {i.timeline_item_id for i in find(out, "timeline.clip.empty_content")} == {t.id, m.id}
    for i in find(out, "timeline.clip.empty_content"):
        assert i.severity is Severity.ERROR and i.fix.kind == "clip.remove_empty" and i.auto_fix_safe
    none_of(out, "timeline.orphan.asset")  # no asset id at all is an empty item, not a dangling reference


def test_text_with_content_is_clean(tmp_path):
    p, *_ = build(tmp_path)
    add_clip(p, "track_v5", None, 2, 3, kind="text", text={"content": "Silver +4%"}, created_by="AI")
    none_of(check(p), "timeline.clip.empty_content")


def test_nonfinite_time_value_is_critical(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, math.nan, 3, created_by="AI")
    out = check(p)
    assert one(out, "timeline.clip.nonfinite").severity is Severity.CRITICAL
    none_of(check(build(tmp_path / "ok")[0]), "timeline.clip.nonfinite")


def test_invalid_speed_is_critical(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    ok = add_clip(p, "track_v2", video, 4, 3, created_by="AI", speed=1.5, opacity=0.5)
    none_of(check(p), "timeline.clip.speed")
    ok.speed = 0.0
    assert one(check(p), "timeline.clip.speed").severity is Severity.CRITICAL


# ------------------------------------------------------------------ overlaps
def test_same_track_overlap_is_an_error_and_names_the_clip_it_runs_into(tmp_path):
    p, _s, video, clips, _v = build(tmp_path)
    late = add_clip(p, "track_v1", video, 8, 2, created_by="AI")  # sits on the last 2 s of clip 0
    iss = one(check(p), "timeline.overlap.same_track")
    assert iss.severity is Severity.ERROR and iss.timeline_item_id == late.id and abs(iss.start_time - 8) < 1e-6 and abs(iss.end_time - 10) < 1e-6
    assert "2.00 s" in iss.description and iss.affected_elements


def test_adjacent_clips_do_not_overlap(tmp_path):
    p, *_ = build(tmp_path)
    none_of(check(p), "timeline.overlap.same_track", "timeline.overlap.cross_track")


def test_cross_track_overlap_unintended_vs_deliberate_layers(tmp_path):
    p, _s, video, clips, _v = build(tmp_path)
    broll = add_asset(p, "broll.mp4", "video", duration=12)
    staggered = add_clip(p, "track_v2", broll, 8, 4, created_by="AI")  # starts inside clip 0, ends inside clip 1: two shots fighting
    found = find(check(p), "timeline.overlap.cross_track")  # one finding per covered clip: 2 s over clip 0 and 2 s over clip 1
    assert len(found) == 2 and {i.timeline_item_id for i in found} == {staggered.id} and all(i.severity is Severity.ERROR and abs(i.duration - 2.0) < 1e-6 for i in found)
    staggered.created_by = "USER"
    assert {i.severity for i in find(check(p), "timeline.overlap.cross_track")} == {Severity.WARNING}  # the validator's rule: the user may have meant it
    p.timeline.get_track("track_v2").clips.clear()
    deliberate = {
        "cutaway inside the clip": dict(start=2, dur=4),
        "half transparent layer": dict(start=8, dur=4, opacity=0.5),
        "picture in picture": dict(start=8, dur=4, scale=0.4, position=(500.0, 300.0)),
        "flagged overlay": dict(start=8, dur=4, effects={"overlay": True}),
        "cross dissolve": dict(start=9.5, dur=1.0, transition={"type": "DISSOLVE", "duration": 1.0}),
    }
    for name, kw in deliberate.items():
        s, d = kw.pop("start"), kw.pop("dur")
        c = add_clip(p, "track_v2", broll, s, d, created_by="AI", **kw)
        none_of(check(p), "timeline.overlap.cross_track")
        p.timeline.get_track("track_v2").clips.remove(c)
    hidden = add_clip(p, "track_v2", broll, 0, 12, created_by="AI")  # covers clip 0 completely and clip 1 partly
    assert "hidden completely" in " ".join(i.description for i in find(check(p), "timeline.overlap.cross_track"))
    p.timeline.get_track("track_v2").clips.remove(hidden)
    none_of(check(p), "timeline.overlap.cross_track")


# ------------------------------------------------------------------ gaps
def with_gap(tmp_path, *, still: bool = False, user: bool = False, gap: tuple[float, float] = (10.0, 20.0)):
    """V1 holds [0, gap start) and [gap end, 30); the first clip is a still (extendable) or a 12 s video (not extendable by 10 s)."""
    p, scenes, video, clips, vclip = build(tmp_path)
    v1 = p.timeline.get_track("track_v1")
    v1.clips.clear()
    first = add_asset(p, "still.png", "image") if still else video
    kw = {} if user else {"created_by": "AI"}
    prev = add_clip(p, "track_v1", first, 0, gap[0], scene=scenes[0], **kw)
    add_clip(p, "track_v1", video, gap[1], 30 - gap[1], scene=scenes[2], created_by="AI")
    return p, prev


def test_unintended_gap_error_over_a_second_warning_under_it(tmp_path):
    p, _ = with_gap(tmp_path)
    iss = one(check(p), "timeline.gap.unintended")
    assert iss.severity is Severity.ERROR and abs(iss.start_time - 10) < 1e-6 and abs(iss.end_time - 20) < 1e-6 and iss.scene_id == "scene_002"
    p, _ = with_gap(tmp_path / "short", gap=(10.0, 10.5))
    assert one(check(p), "timeline.gap.unintended").severity is Severity.WARNING


def test_gap_shorter_than_a_frame_is_ignored(tmp_path):
    p, _ = with_gap(tmp_path, gap=(10.0, 10.02))  # 30 fps: a frame is 0.033 s
    none_of(check(p), "timeline.gap.unintended")


def test_declared_intentional_gap_is_not_reported_but_the_rest_of_it_is(tmp_path):
    p, _ = with_gap(tmp_path)
    s = QCSettings()
    s.intentional_gaps = [[10.0, 20.0]]
    out = check(p, s)
    none_of(out, "timeline.gap.unintended")
    assert out.metrics["intentional_gap_seconds"] == pytest.approx(10.0)
    s.intentional_gaps = [[10.0, 15.0]]
    iss = one(check(p, s), "timeline.gap.unintended")
    assert abs(iss.start_time - 15) < 1e-6 and abs(iss.end_time - 20) < 1e-6


def test_gap_continued_by_a_longer_clip_on_another_visual_track_is_not_a_gap(tmp_path):
    p, _ = with_gap(tmp_path)
    still = add_asset(p, "held.png", "image")
    add_clip(p, "track_v3", still, 8, 14, created_by="AI")  # V3 holds the picture over the whole V1 gap
    out = check(p)
    none_of(out, "timeline.gap.unintended")
    assert out.metrics["visual_covered_ratio"] == pytest.approx(1.0)


def test_sparse_overlay_tracks_are_not_gaps(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 5, 3, created_by="AI", opacity=0.6)
    add_clip(p, "track_v5", None, 12, 2, kind="text", text={"content": "Silver"}, created_by="AI")
    none_of(check(p), "timeline.gap.unintended")


def test_a_blank_stretch_inside_a_deliberate_narration_pause_is_not_an_error(tmp_path):
    p, _ = with_gap(tmp_path, gap=(12.5, 14.5))
    tr = p.transcription.transcript
    tr.words = [w for w in tr.words if not 12 <= w.start < 15]  # nobody speaks between ~11.9 s and 15.0 s
    out = check(p)
    none_of(out, "timeline.gap.unintended")
    assert out.metrics["pause_gap_seconds"] == pytest.approx(2.0)
    p, _ = with_gap(tmp_path / "overlapping", gap=(11.0, 14.0))
    tr = p.transcription.transcript
    tr.words = [w for w in tr.words if not 12 <= w.start < 15]
    assert one(check(p), "timeline.gap.unintended")  # it starts while the narration is still speaking


def test_gap_close_fix_needs_confirmation_and_respects_the_source_length(tmp_path):
    p, prev = with_gap(tmp_path, still=True)
    iss = one(check(p), "timeline.gap.unintended")
    assert iss.fix.kind == "gap.close" and iss.fix.params == {"clip_id": prev.id, "new_end": 20.0}
    assert not iss.fix.safe and iss.fix.needs_confirmation and iss.auto_fix_available and not iss.auto_fix_safe and iss.timeline_item_id == prev.id
    p, prev = with_gap(tmp_path / "video")  # a 12 s video cannot be stretched by 10 s
    iss = one(check(p), "timeline.gap.unintended")
    assert iss.fix is None or not iss.auto_fix_available
    assert "too short" in iss.fix_blocked_reason


def test_gap_close_fix_is_disabled_for_a_user_owned_clip(tmp_path):
    p, prev = with_gap(tmp_path, still=True, user=True)
    iss = one(check(p), "timeline.gap.unintended")
    assert iss.locked and not iss.auto_fix_available and "edited or created by you" in iss.fix_blocked_reason


# ------------------------------------------------------------------ outside / beyond the narration
def test_clip_before_zero_is_critical_and_after_the_end_is_an_error(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    early = add_clip(p, "track_v2", video, -1.0, 3, created_by="AI", opacity=0.5)
    late = add_clip(p, "track_v5", None, 31.0, 2, kind="text", text={"content": "Late"}, created_by="AI")
    out = check(p)
    by = {i.timeline_item_id: i for i in find(out, "timeline.clip.out_of_bounds")}
    assert by[early.id].severity is Severity.CRITICAL and by[late.id].severity is Severity.ERROR and len(by) == 2


def test_clip_inside_the_video_is_in_bounds(tmp_path):
    p, *_ = build(tmp_path)
    none_of(check(p), "timeline.clip.out_of_bounds", "timeline.clip.beyond_narration")


def test_clip_running_past_the_narration_by_kind_and_amount(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    music = add_asset(p, "music.wav", "audio", duration=60)
    pic = add_clip(p, "track_v2", video, 28, 5, created_by="AI", opacity=0.5)  # 3 s past the end of the narration
    txt = add_clip(p, "track_v5", None, 29, 2.2, kind="text", text={"content": "End"}, created_by="AI")  # 1.2 s past
    mus = add_clip(p, "track_a2", music, 20, 13, created_by="AI", audio={"role": "MUSIC", "volume": 0.2})
    within = add_clip(p, "track_a3", music, 25, 5.3, created_by="AI", audio={"role": "SFX", "volume": 0.2})  # 0.3 s: inside the tolerance
    by = {i.timeline_item_id: i for i in find(check(p), "timeline.clip.beyond_narration")}
    assert by[pic.id].severity is Severity.ERROR and by[txt.id].severity is Severity.WARNING and by[mus.id].severity is Severity.NOTICE and mus.id in by and within.id not in by
    assert by[mus.id].confidence < 100  # a music tail may be a deliberate outro


# ------------------------------------------------------------------ keyframes, opacity, transforms, levels: safe normalisation
def test_keyframe_outside_its_clip_is_critical_a_valid_one_is_clean(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[0].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 9.0, 1.2)]
    none_of(check(p), "timeline.keyframe.invalid", "timeline.keyframe.value")
    clips[0].keyframes.append(Keyframe("scale", 99.0, 1.3))
    assert one(check(p), "timeline.keyframe.invalid").severity is Severity.CRITICAL
    clips[0].keyframes = [Keyframe("sparkle", 1.0, 1.0)]
    assert "unknown keyframe property" in one(check(p), "timeline.keyframe.invalid").description


def test_keyframe_value_out_of_range_has_a_safe_clamp_fix(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[0].keyframes = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 2.0, 1.7), Keyframe("scale", 3.0, -0.5)]
    iss = one(check(p), "timeline.keyframe.value")
    assert iss.severity is Severity.ERROR and iss.fix.kind == "param.normalize" and iss.auto_fix_safe
    assert iss.fix.params["changes"]["keyframes"] == [{"property": "opacity", "time": 2.0, "value": 1.0}, {"property": "scale", "time": 3.0, "value": 0.01}]
    clips[0].created_by = "USER"
    iss = one(check(p), "timeline.keyframe.value")
    assert iss.locked and not iss.auto_fix_available


def test_invalid_opacity_is_critical_with_a_safe_clamp_and_the_lock_case(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[1].opacity = 1.5
    iss = one(check(p), "timeline.opacity.invalid")
    assert iss.severity is Severity.CRITICAL and iss.fix.kind == "param.normalize" and iss.fix.params == {"clip_id": clips[1].id, "changes": {"opacity": 1.0}} and iss.auto_fix_safe
    clips[1].opacity = math.nan
    assert one(check(p), "timeline.opacity.invalid").fix.params["changes"] == {"opacity": 1.0}
    clips[1].opacity = -0.2
    assert one(check(p), "timeline.opacity.invalid").fix.params["changes"] == {"opacity": 0.0}
    clips[1].created_by = "USER"
    iss = one(check(p), "timeline.opacity.invalid")
    assert iss.locked and not iss.auto_fix_available and not iss.auto_fix_safe and iss.fix_blocked_reason
    clips[1].opacity = 1.0
    none_of(check(p), "timeline.opacity.invalid")


def test_invalid_transforms(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[0].scale = 0.0
    iss = one(check(p), "timeline.transform.invalid")
    assert iss.severity is Severity.CRITICAL and iss.fix.params["changes"] == {"scale": 1.0} and iss.auto_fix_safe
    clips[0].scale = 50.0  # absurd, but the render can still draw it
    iss = one(check(p), "timeline.transform.invalid")
    assert iss.severity is Severity.ERROR and iss.fix.params["changes"] == {"scale": 20.0}
    clips[0].scale = 1.0
    clips[0].position = (math.nan, 0.0)
    iss = one(check(p), "timeline.transform.invalid")
    assert iss.severity is Severity.CRITICAL and iss.fix is None  # nothing safe to guess for a position
    clips[0].position = (0.0, 0.0)
    clips[0].scale = 1.4
    none_of(check(p), "timeline.transform.invalid")
    clips[0].scale, clips[0].created_by = 0.0, "USER"
    iss = one(check(p), "timeline.transform.invalid")
    assert iss.locked and not iss.auto_fix_available


def test_invalid_audio_levels(tmp_path):
    p, *_ = build(tmp_path)
    music = add_asset(p, "music.wav", "audio", duration=60)
    m = add_clip(p, "track_a2", music, 0, 30, created_by="AI", audio={"role": "MUSIC", "volume": 7.0})
    iss = one(check(p), "timeline.audio_level.invalid")
    assert iss.severity is Severity.ERROR and iss.fix.params["changes"] == {"audio.volume": 4.0} and iss.auto_fix_safe
    m.audio["volume"] = math.nan
    assert one(check(p), "timeline.audio_level.invalid").fix.params["changes"] == {"audio.volume": 1.0}
    m.audio["volume"] = 0.2
    none_of(check(p), "timeline.audio_level.invalid")
    p.timeline.get_track("track_a2").volume = math.nan
    iss = one(check(p), "timeline.audio_level.invalid")
    assert iss.timeline_item_id is None and iss.track_id == "track_a2" and iss.fix is None
    p.timeline.get_track("track_a2").volume = 1.5
    none_of(check(p), "timeline.audio_level.invalid")
    m.audio["volume"], m.created_by = 9.0, "USER"
    iss = one(check(p), "timeline.audio_level.invalid")
    assert iss.locked and not iss.auto_fix_available


# ------------------------------------------------------------------ duplicates
def test_exact_duplicate_is_reported_once_with_a_confirm_only_delete_fix(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    broll = add_asset(p, "b.mp4", "video", duration=12)
    a = add_clip(p, "track_v2", broll, 3, 4, created_by="AI", id="dup_a", opacity=0.5)
    b = add_clip(p, "track_v2", broll, 3, 4, created_by="AI", id="dup_b", opacity=0.5)
    out = check(p)
    iss = one(out, "timeline.duplicate.element")
    assert iss.severity is Severity.WARNING and iss.fix.kind == "clip.delete" and iss.fix.params == {"clip_id": b.id} and not iss.fix.safe and iss.fix.needs_confirmation
    assert iss.auto_fix_available and not iss.auto_fix_safe
    none_of(out, "timeline.overlap.same_track")  # the overlap a duplicate causes is the same finding
    a.created_by = "AI"
    b.created_by = "USER"  # delete the copy the user does not own
    assert one(check(p), "timeline.duplicate.element").fix.params["clip_id"] == a.id
    a.created_by = "USER"
    iss = one(check(p), "timeline.duplicate.element")
    assert iss.locked and not iss.auto_fix_available
    b.asset_id = video.id  # a different asset at the same time is not a duplicate
    none_of(check(p), "timeline.duplicate.element")


def test_duplicate_audio_is_an_error_and_a_shifted_copy_is_not_a_duplicate(tmp_path):
    p, *_ = build(tmp_path)
    sfx = add_asset(p, "whoosh.wav", "audio", duration=2)
    add_clip(p, "track_a3", sfx, 5, 1, created_by="AI", audio={"role": "SFX", "volume": 0.3})
    add_clip(p, "track_a3", sfx, 5, 1, created_by="AI", audio={"role": "SFX", "volume": 0.3})
    assert one(check(p), "timeline.duplicate.element").severity is Severity.ERROR
    p2, *_ = build(tmp_path / "shifted")
    sfx2 = add_asset(p2, "whoosh.wav", "audio", duration=2)
    add_clip(p2, "track_a3", sfx2, 5, 1, created_by="AI", audio={"role": "SFX"})
    add_clip(p2, "track_a3", sfx2, 9, 1, created_by="AI", audio={"role": "SFX"})
    none_of(check(p2), "timeline.duplicate.element")


def test_duplicate_ids_and_duplicate_tracks(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 3, 2, created_by="AI", id="same", opacity=0.5)
    add_clip(p, "track_v3", video, 6, 2, created_by="AI", id="same", opacity=0.5)
    p.timeline.tracks.append(Track("track_v1", "V1 again", TrackKind.VIDEO))
    out = check(p)
    assert one(out, "timeline.duplicate.id").severity is Severity.ERROR and one(out, "timeline.duplicate.track").severity is Severity.ERROR
    p2, *_ = build(tmp_path / "clean")
    none_of(check(p2), "timeline.duplicate.id", "timeline.duplicate.track")


# ------------------------------------------------------------------ orphans and invalid references
def test_orphaned_asset_scene_decision_and_track_references(tmp_path):
    p, _s, video, clips, _vc = build(tmp_path)
    ghost = add_clip(p, "track_v2", video, 3, 2, created_by="AI", opacity=0.5)
    ghost.asset_id = "asset_gone"
    lost = add_clip(p, "track_v2", video, 6, 2, created_by="AI", opacity=0.5, scene_id="scene_999", ai_decision_id="dec_99999")
    wrong = add_clip(p, "track_v2", video, 9, 2, created_by="AI", opacity=0.5)
    wrong.track_id = "track_zzz"
    p.editing_decisions["dec_00001"] = EditingDecision("dec_00001", "scene_001", DecisionType.CUT, target_id="clip_vanished")
    out = check(p)
    assert one(out, "timeline.orphan.asset").severity is Severity.CRITICAL and one(out, "timeline.orphan.asset").timeline_item_id == ghost.id
    sc = one(out, "timeline.orphan.scene")
    assert sc.severity is Severity.WARNING and sc.timeline_item_id == lost.id and sc.scene_id is None  # never a scene-scoped issue for a scene that does not exist
    decisions = find(out, "timeline.orphan.decision")
    assert len(decisions) == 2 and all(d.severity is Severity.WARNING for d in decisions)
    assert one(out, "timeline.orphan.track").timeline_item_id == wrong.id


def test_valid_references_are_clean(tmp_path):
    p, scenes, video, clips, _vc = build(tmp_path)
    p.editing_decisions["dec_00001"] = EditingDecision("dec_00001", scenes[0].id, DecisionType.CUT, target_id=clips[0].id)
    clips[0].ai_decision_id = "dec_00001"
    none_of(check(p), "timeline.orphan.asset", "timeline.orphan.scene", "timeline.orphan.decision", "timeline.orphan.track")


def test_caption_references(tmp_path):
    p, scenes, *_ = build(tmp_path)
    words = [{"word_id": "w1", "text": "Hello", "start": 2.0, "end": 2.4}, {"word_id": "w2", "text": "world", "start": 2.5, "end": 3.0}]
    good = {"caption_id": "cap1", "scene_id": scenes[0].id, "start": 2.0, "end": 3.0, "text": "Hello world", "lines": ["Hello world"], "words": words}
    ok = add_clip(p, "track_v6", None, 2, 1, kind="caption", text=good, created_by="AI", scene_id=scenes[0].id)
    none_of(check(p), "timeline.caption.invalid", "timeline.orphan.scene")
    bad = add_clip(p, "track_v6", None, 5, 1, kind="caption", text=None, created_by="AI")
    iss = one(check(p), "timeline.caption.invalid")
    assert iss.severity is Severity.ERROR and iss.timeline_item_id == bad.id
    bad.text = {**good, "scene_id": "scene_404", "text": "Hello world", "words": words}
    ok.text = good
    assert one(check(p), "timeline.orphan.scene").timeline_item_id == bad.id


# ------------------------------------------------------------------ mapped from the validators
def test_broken_transition_is_an_error_a_fitting_one_is_clean(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[1].transition = {"type": "FADE", "duration": 0.5}
    none_of(check(p), "timeline.transition.invalid")
    clips[1].transition = {"type": "FADE", "duration": 99.0}
    assert one(check(p), "timeline.transition.invalid").severity is Severity.ERROR
    clips[1].transition = {"type": "FADE", "duration": -0.3}
    assert one(check(p), "timeline.transition.invalid").timeline_item_id == clips[1].id


def test_invalid_effect_parameters(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[0].effects = {"fit": "cover", "focus_region": [0.1, 0.1, 0.5, 0.5]}
    none_of(check(p), "timeline.effect.invalid")
    clips[0].effects = {"fit": "stretch", "focus_region": [0.5, 0.5, 0.9, 0.9]}
    iss = one(check(p), "timeline.effect.invalid")
    assert iss.severity is Severity.ERROR and "stretch" in iss.description and "focus region" in iss.description
    clips[0].effects = {}
    hl = add_clip(p, "track_v4", None, 3, 2, kind="graphic", effects={"highlight": {"region": [0.2, 0.2, 0.9, 0.9], "style": "box"}}, created_by="AI")
    assert one(check(p), "timeline.effect.invalid").timeline_item_id == hl.id


def test_invalid_animation_is_critical_even_on_user_owned_clips(tmp_path):
    p, _s, _v, clips, _vc = build(tmp_path)
    clips[0].animation = {"in": animation.spec("fade_in")}
    none_of(check(p), "timeline.animation.invalid")
    clips[0].animation = {"in": {"preset": "bogus", "duration": 0.3}}
    clips[0].created_by = "USER"  # the presentation validator only warns for owned clips; the render refuses it all the same
    assert one(check(p), "timeline.animation.invalid").severity is Severity.CRITICAL


def test_source_range_and_track_kind_problems_come_from_the_timeline_validator(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    voice_asset = p.assets.get(p.voice_over.asset_id)
    none_of(check(p), "timeline.clip.source_range", "timeline.clip.track_kind")  # the clean project plays inside its media on the right tracks
    bad_src = add_clip(p, "track_v2", video, 5, 3, source_in=10.0, source_out=13.0, created_by="AI", opacity=0.5)  # the media is 12 s long
    wrong = add_clip(p, "track_v3", voice_asset, 20, 2, created_by="AI", opacity=0.5)
    out = check(p)
    assert one(out, "timeline.clip.source_range").timeline_item_id == bad_src.id
    assert one(out, "timeline.clip.track_kind").timeline_item_id == wrong.id


def test_voice_over_that_moved_is_an_error(tmp_path):
    p, _s, _v, _c, vclip = build(tmp_path)
    none_of(check(p), "timeline.voice.misaligned")
    vclip.timeline_start = 2.0
    iss = one(check(p), "timeline.voice.misaligned")
    assert iss.severity is Severity.ERROR and iss.fix is None  # the master clock is never moved by QC
    none_of(check(p), "timeline.empty.voice", "timeline.clip.beyond_narration")


def test_findings_are_deterministic_and_fingerprints_survive_a_rerun(tmp_path):
    p, _s, video, *_ = build(tmp_path)
    add_clip(p, "track_v2", video, 5, 0.0, created_by="AI")
    a, b = check(p), check(p)
    assert [i.fingerprint for i in a.issues] == [i.fingerprint for i in b.issues] and all(i.fingerprint for i in a.issues)
    assert [i.issue_id for i in a.issues] != [i.issue_id for i in b.issues]  # ids are per run, fingerprints are per finding


# ------------------------------------------------------------------ real pipeline (needs ffmpeg for the media behind the project)
@needs_ffmpeg
def test_real_ai_edited_project_is_clean_and_a_removed_clip_becomes_a_gap(pres_ws):
    p = pres_ws.project
    out = check(p)
    assert out.issues == [] and out.complete and out.metrics["visual_covered_ratio"] == 1.0  # TimelineValidator / PresentationValidator and the new checks agree with the assembler
    mid = p.scenes[len(p.scenes) // 2]
    victim = next(c for t, c in [(t, c) for t in p.timeline.tracks for c in t.clips] if t.id in ("track_v1", "track_v2", "track_v3") and c.timeline_start >= mid.start - 1e-6 and c.timeline_end <= mid.end + 1e-6)
    next(t for t in p.timeline.tracks if t.id == victim.track_id).clips.remove(victim)  # direct edit of the live project: the test never saves it
    out = check(p)
    gap = [i for i in out.issues if i.code == "timeline.gap.unintended"]
    assert gap and gap[0].scene_id == mid.id and out.metrics["visual_covered_ratio"] < 1.0
