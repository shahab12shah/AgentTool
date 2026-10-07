"""QCFixEngine: every fix is previewable, undoable in one step, refuses what the user owns or locked, refuses a stale issue, and leaves the validators clean.

Synthetic projects through a real Workspace (no ffmpeg) for the exact cases; the demo project / a generated AI edit only where real media or real AI-owned objects matter.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.editing.models import Creator, DecisionType, EditingDecision
from app.media.importer import sha256_file
from app.presentation.models import PresentationDecision, PresentationType
from app.qc import fix_engine as fe
from app.qc import fix_catalog as fc
from app.qc.errors import QCError
from app.qc.fix_engine import STALE, FixPreview, QCFixEngine
from app.qc.issue_model import FixRecord, IssueStatus, QCCategory, QCIssue
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, narrate
from app.tests.conftest import needs_ffmpeg
from app.timeline.keyframes import Keyframe, value_at


# ---------------------------------------------------------------------------------------------- builders
@pytest.fixture
def fws(project_ws):
    ws = project_ws
    p = ws.project
    p.voice_over.duration = 30.0
    ws.vid = add_asset(p, "a.mp4", "video", duration=12)
    ws.s1 = add_scene(p, 0, 15, "Silver rose sharply this week.", importance=0.8)
    ws.s2 = add_scene(p, 15, 30, "Demand keeps growing steadily.")
    narrate(p)
    return ws


def tl_json(ws) -> str:
    return json.dumps(ws.project.timeline.to_dict(), sort_keys=True)


def dec_json(ws) -> str:
    p = ws.project
    return json.dumps([{k: v.to_dict() for k, v in sorted(d.items())} for d in (p.editing_decisions, p.presentation_decisions)] + [len(p.ai_overrides), len(p.presentation_overrides)], sort_keys=True)


def state_json(ws) -> str:
    return tl_json(ws) + dec_json(ws)


def add_caption(p, start: float, text: str = "Silver prices rose sharply today", *, dt: float = 0.3, created_by: str = "AI", scene=None, **kw):
    words = text.split()
    w = [{"word_id": f"w{start}_{i}", "text": t, "start": round(start + i * dt, 3), "end": round(start + i * dt + dt * 0.9, 3)} for i, t in enumerate(words)]
    end = round(start + len(words) * dt, 3)
    seg = {"caption_id": f"cap_{start}", "scene_id": scene.id if scene else "", "start": start, "end": end, "text": text, "lines": [text], "words": w, "emphasis": [], "style_id": "professional",
           "style_overrides": {}, "position": "bottom", "position_xy": [], "highlight_mode": "HIGHLIGHT", "reading_cps": 12.0}
    return add_clip(p, "track_v6", None, start, end - start, kind="caption", text=seg, created_by=created_by, scene=scene, slot=kw.pop("slot", f"caption:{start}"),
                    metadata={"phase": 5, "caption_id": seg["caption_id"]}, **kw)


def add_pres_decision(p, clip, type_=PresentationType.CAPTION, created_by=Creator.AI, **kw) -> PresentationDecision:
    d = PresentationDecision(kw.pop("decision_id", f"pdec_{len(p.presentation_decisions) + 1:05d}"), clip.scene_id, type_, clip.slot, clip.id, clip.timeline_start, clip.duration,
                             kw.pop("parameters", {"reading_cps": 12.0}), "test", 90.0, created_by, **kw)
    p.presentation_decisions[d.decision_id] = d
    clip.ai_decision_id = clip.ai_decision_id or d.decision_id
    return d


def add_visual(ws, start: float, dur: float, *, scene=None, created_by: str = "AI", asset=None, source_in: float = 0.0, track: str = "track_v1", decision: bool = True, **kw):
    p = ws.project
    c = add_clip(p, track, asset or ws.vid, start, dur, scene=scene or ws.s1, created_by=created_by, slot=kw.pop("slot", f"visual:{start}"), source_in=source_in, **kw)
    if decision:
        d = EditingDecision(f"dec_{len(p.editing_decisions) + 1:05d}", c.scene_id, DecisionType.VISUAL_TIMING, c.slot, c.id, start, dur,
                            {"asset_id": c.asset_id, "source_in": source_in, "source_out": source_in + dur, "speed": 1.0}, "test", 90.0, Creator.AI)
        p.editing_decisions[d.decision_id] = d
        c.ai_decision_id = d.decision_id
    return c


_n = [0]


def issue(ws, code: str, fix, *, clip=None, start=None, end=None, category=QCCategory.CAPTION, **kw) -> QCIssue:
    _n[0] += 1
    i = QCIssue(issue_id=f"qci_t{_n[0]:04d}", code=code, category=category, severity=Severity.WARNING, title=kw.pop("title", "Test issue"), description="d", scene_id=kw.pop("scene_id", None),
                timeline_item_id=clip.id if clip is not None else None, start_time=start if start is not None else (clip.timeline_start if clip is not None else None),
                end_time=end if end is not None else (clip.timeline_end if clip is not None else None), fix=fix, auto_fix_available=True, auto_fix_safe=bool(fix.safe), **kw)
    i.fingerprint = i.make_fingerprint()
    ws.project.qc_issues.append(i)
    return i


def retime_issue(ws, clip, shift: float, **kw) -> QCIssue:
    """A caption that is ``shift`` seconds early (negative: it should move earlier)."""
    ref = clip.timeline_start + shift
    fix = fc.caption_retime(clip.id, ref, clip.timeline_end + shift, shift, ws.project.qc_settings)
    return issue(ws, "sync.caption_drift", fix, clip=clip, **kw)


def engine(ws, **svc) -> QCFixEngine:
    """An engine over the workspace's services, optionally with a stub that says an AI edit is running."""
    services = SimpleNamespace(editing=ws.editing, presentation=ws.presentation, timeline=ws.timeline, render=ws.render)
    services.__dict__.update(svc)
    return QCFixEngine(ws.projects, ws.commands.execute, ws.editing._checkpoint, services)


def errors(ws) -> set:
    return {(i.code, i.clip_id) for i in ws.editing.validate_timeline() + ws.presentation.validate() if i.severity == "error"}


def undo_count(ws) -> int:
    return len(ws.commands._undo)


# ================================================================================================ caption.retime
def test_safe_caption_retime_applies_undoes_and_redoes(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    dec = add_pres_decision(p, cap)
    iss = retime_issue(ws, cap, -0.2)
    assert iss.fix.safe
    before, base_errors = state_json(ws), errors(ws)
    words_before = json.dumps(cap.text["words"]) + cap.text["text"]

    rec = ws.qc.fixes.apply_fix(iss.issue_id)  # no confirmation needed: safe and permitted "auto"

    assert isinstance(rec, FixRecord) and rec.safe and not rec.confirmed_by_user and rec.kind == "caption.retime" and rec.issue_id == iss.issue_id
    assert rec.before["caption start"] == "5.00 s" and rec.after["caption start"] == "4.80 s"
    c = p.timeline.get_clip(cap.id)
    assert c.timeline_start == pytest.approx(4.8) and c.timeline_end == pytest.approx(cap.timeline_end - 0.2)
    assert json.dumps(c.text["words"]) + c.text["text"] == words_before  # the words and their timing are never touched
    assert c.text["start"] == pytest.approx(4.8) and c.text["end"] == pytest.approx(c.timeline_end)
    assert iss.status is IssueStatus.FIXED and [f.fix_id for f in p.qc_fixes] == [rec.fix_id]
    assert c.created_by == "USER"  # ownership follows the manual-edit rule: a regeneration keeps the fix
    assert dec.decision_id not in p.presentation_decisions and any(o.original.decision_id == dec.decision_id for o in p.presentation_overrides)
    new_dec = next(d for d in p.presentation_decisions.values() if d.target_id == cap.id)
    assert new_dec.created_by is Creator.USER and new_dec.start == pytest.approx(4.8)
    assert errors(ws) <= base_errors  # no new error-level validation problem
    p.validate()  # the save gate still passes

    ws.commands.undo()
    assert state_json(ws) == before and iss.status is IssueStatus.OPEN and p.qc_fixes == []
    ws.commands.redo()
    assert p.timeline.get_clip(cap.id).timeline_start == pytest.approx(4.8) and iss.status is IssueStatus.FIXED and len(p.qc_fixes) == 1


def test_large_shift_needs_confirmation(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, 1.5)  # beyond max_caption_shift_seconds
    assert not iss.fix.safe
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.needs_confirmation and not pv.safe and not pv.blocked_reason
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    assert state_json(ws) == before and iss.status is IssueStatus.OPEN and undo_count(ws) == 0

    rec = ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert rec.confirmed_by_user and not rec.safe and rec.checkpoint  # a safety copy is written before a fix that is not safe
    assert p.timeline.get_clip(cap.id).timeline_start == pytest.approx(6.5)
    ws.commands.undo()
    assert state_json(ws) == before


def test_retime_refuses_to_overlap_the_next_caption(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    add_caption(p, 6.3, "Another caption right behind", scene=ws.s1)
    iss = retime_issue(ws, cap, 0.4)
    before = state_json(ws)
    with pytest.raises(QCError, match="overlap"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert state_json(ws) == before


# ================================================================================================ protection
def _lock_clip(ws, cap):
    cap.locked = True


def _user_owned(ws, cap):
    cap.created_by = "USER"


def _system_owned(ws, cap):
    cap.created_by = "SYSTEM"


def _locked_track(ws, cap):
    ws.project.timeline.get_track(cap.track_id).locked = True


def _locked_timeline_scene(ws, cap):
    ws.project.timeline_generation.locked_scenes.append(ws.s1.id)


def _locked_presentation_scene(ws, cap):
    ws.project.presentation_generation.locked_scenes.append(ws.s1.id)


def _locked_decision(ws, cap):
    add_pres_decision(ws.project, cap, locked=True)


def _user_decision(ws, cap):
    add_pres_decision(ws.project, cap, created_by=Creator.USER)


@pytest.mark.parametrize("protect", [_lock_clip, _user_owned, _system_owned, _locked_track, _locked_timeline_scene, _locked_presentation_scene, _locked_decision, _user_decision])
def test_protected_elements_are_reported_but_never_modified(fws, protect):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, -0.2)
    protect(ws, cap)
    before = state_json(ws)
    eng = ws.qc.fixes
    for confirmed in (False, True):
        with pytest.raises(QCError):
            eng.apply_fix(iss.issue_id, confirmed=confirmed)
    pv = eng.preview_fix(iss.issue_id)
    assert pv.blocked_reason and not pv.safe
    ok, why = eng.can_fix(iss)
    assert not ok and why == pv.blocked_reason
    assert eng.apply_safe_fixes() == [] and eng.last_skipped[0].reason == pv.blocked_reason
    assert state_json(ws) == before and iss.status is IssueStatus.OPEN and undo_count(ws) == 0 and p.qc_fixes == []


def test_the_voice_over_clip_is_never_touched(fws):
    ws, p = fws, fws.project
    voice = add_asset(p, "voice.wav", "audio", duration=30, w=None, h=None)
    c = add_clip(p, "track_a1", voice, 0.0, 30.0, created_by="SYSTEM", slot="voice", audio={"role": "VOICE", "volume": 1.0})
    iss = issue(ws, "audio.voice_quiet", fc.audio_level(c.id, 1.5, "Raise the voice", p.qc_settings), clip=c, category=QCCategory.AUDIO)
    before = state_json(ws)
    with pytest.raises(QCError, match="voice"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert state_json(ws) == before


# ================================================================================================ stale issues / busy / missing
def test_stale_issue_is_refused(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, -0.2)
    ws.timeline.move_clip(cap.id, 5.4)  # the user moved it after QC ran (this also makes it USER-owned, so first give it back to the AI to isolate the stale rule)
    p.timeline.get_clip(cap.id).created_by = "AI"
    before = state_json(ws)
    with pytest.raises(QCError) as e:
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert e.value.user_message == STALE and "run QC again" in e.value.user_message
    assert state_json(ws) == before
    assert ws.qc.fixes.preview_fix(iss.issue_id).blocked_reason == STALE


def test_missing_fixed_or_ignored_issue_is_refused(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, -0.2)
    with pytest.raises(QCError, match="no longer"):
        ws.qc.fixes.apply_fix("qci_nope")
    ws.qc.fixes.apply_fix(iss.issue_id)
    with pytest.raises(QCError, match="already"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    cap2 = add_caption(p, 12.0, "A second caption goes here", scene=ws.s1)
    iss2 = retime_issue(ws, cap2, -0.2)
    iss2.ignored_by_user, iss2.status = True, IssueStatus.IGNORED
    with pytest.raises(QCError, match="ignored"):
        ws.qc.fixes.apply_fix(iss2.issue_id)
    gone = retime_issue(ws, add_caption(p, 20.0, "Soon deleted caption here", scene=ws.s2), -0.2)
    p.timeline.get_track("track_v6").clips.pop()
    with pytest.raises(QCError, match="no longer exists"):
        ws.qc.fixes.apply_fix(gone.issue_id)


@pytest.mark.parametrize("which", ["editing", "presentation"])
def test_refuses_while_generation_is_running(fws, which):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, -0.2)
    eng = engine(ws, **{which: SimpleNamespace(running=True)})
    before = state_json(ws)
    with pytest.raises(QCError, match="Wait"):
        eng.apply_fix(iss.issue_id)
    assert eng.preview_fix(iss.issue_id).blocked_reason.startswith(("An AI edit", "The captions"))
    with pytest.raises(QCError, match="Wait"):
        eng.apply_safe_fixes()
    with pytest.raises(QCError, match="Wait"):
        eng.fix_similar(iss.issue_id, confirmed=True)
    ok, why = eng.can_fix(iss)  # can_fix never raises: it answers
    assert not ok and "Wait" in why
    assert state_json(ws) == before and undo_count(ws) == 0


def test_no_project_open_gives_a_clear_error(app_paths):
    from app.services.workspace import Workspace

    w = Workspace(app_paths)
    try:
        eng = QCFixEngine(w.projects, w.commands.execute, w.editing._checkpoint, SimpleNamespace())
        with pytest.raises(QCError, match="Open a project"):
            eng.apply_fix("x")
    finally:
        w.shutdown()


# ================================================================================================ preview
def test_preview_changes_nothing_and_shows_before_after(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    add_pres_decision(p, cap)
    iss = retime_issue(ws, cap, -0.42)
    before, ids = state_json(ws), [(i.issue_id, i.status) for i in p.qc_issues]
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert isinstance(pv, FixPreview) and pv.kind == "caption.retime" and pv.safe and not pv.needs_confirmation and not pv.destructive and not pv.blocked_reason
    assert pv.before["drift from the spoken start"] == "+420 ms" and pv.after["drift from the spoken start"] == "+0 ms"
    assert pv.before["caption start"] == "5.00 s" and pv.after["caption start"] == "4.58 s"
    assert pv.changes and "words" in pv.changes[-1]
    assert state_json(ws) == before and [(i.issue_id, i.status) for i in p.qc_issues] == ids and undo_count(ws) == 0 and p.qc_fixes == []


def test_navigate_and_research_kinds_open_another_page(fws):
    ws = fws
    for kind in ("visual.replace", "visual.search_again", "scene.skip", "open.timeline"):
        i = issue(ws, "visual.test", fc.navigate(kind, "Go there", scene_id=ws.s1.id), category=QCCategory.VISUAL_ACCURACY)
        pv = ws.qc.fixes.preview_fix(i.issue_id)
        assert pv.blocked_reason == "Opens another page" and not pv.safe and pv.kind == kind
        assert not ws.qc.fixes.can_fix(i)[0]
        with pytest.raises(QCError, match="another page"):
            ws.qc.fixes.apply_fix(i.issue_id, confirmed=True)


def test_issue_without_a_fix_and_unsupported_kind(fws):
    ws = fws
    bare = QCIssue("qci_bare", "x.y", QCCategory.TIMELINE, Severity.NOTICE, "t")
    ws.project.qc_issues.append(bare)
    assert ws.qc.fixes.preview_fix("qci_bare").blocked_reason
    sil = issue(ws, "silence.excessive", fc._spec("silence.remove", {"start": 1.0, "end": 2.0}, "Remove silence", False), category=QCCategory.SILENCE)
    pv = ws.qc.fixes.preview_fix(sil.issue_id)
    assert "narration timing" in pv.blocked_reason
    with pytest.raises(QCError, match="narration timing"):
        ws.qc.fixes.apply_fix(sil.issue_id, confirmed=True)


# ================================================================================================ permissions + batch
def test_batch_applies_only_safe_fixes_in_one_undo_step(fws):
    ws, p = fws, fws.project
    caps = [add_caption(p, 2.0 + 3.0 * i, f"Caption number {i} says something", scene=ws.s1) for i in range(3)]
    safe = [retime_issue(ws, c, -0.2) for c in caps]
    big_cap = add_caption(p, 12.0, "A caption that is far off", scene=ws.s1)
    big = retime_issue(ws, big_cap, 1.4)
    locked_cap = add_caption(p, 15.5, "A caption the user locked", scene=ws.s2, locked=True)
    locked = retime_issue(ws, locked_cap, -0.2)
    before, base_errors, depth = state_json(ws), errors(ws), undo_count(ws)

    done = ws.qc.fixes.apply_safe_fixes()

    assert sorted(r.issue_id for r in done) == sorted(i.issue_id for i in safe)
    assert undo_count(ws) == depth + 1  # ONE undo step for the whole batch
    assert all(r.checkpoint for r in done)  # a batch always writes a safety copy first
    assert [i.status for i in safe] == [IssueStatus.FIXED] * 3 and big.status is IssueStatus.OPEN and locked.status is IssueStatus.OPEN
    reasons = {s.issue_id: s.reason for s in ws.qc.fixes.last_skipped}
    assert "confirm" in reasons[big.issue_id].lower() and "locked" in reasons[locked.issue_id].lower()
    assert errors(ws) <= base_errors
    p.validate()
    ws.commands.undo()
    assert state_json(ws) == before and all(i.status is IssueStatus.OPEN for i in p.qc_issues) and p.qc_fixes == []


def test_batch_filters_by_ids_and_code_prefix(fws):
    ws, p = fws, fws.project
    a, b = add_caption(p, 2.0, scene=ws.s1), add_caption(p, 8.0, "Second caption of the batch", scene=ws.s1)
    ia, ib = retime_issue(ws, a, -0.2), retime_issue(ws, b, -0.2)
    other = issue(ws, "caption.safe_margin", fc.caption_safe_margin(add_caption(p, 14.0, "Third caption out of bounds", scene=ws.s1).id, [0.5, 0.95], p.qc_settings))
    assert [r.issue_id for r in ws.qc.fixes.apply_safe_fixes([ia.issue_id])] == [ia.issue_id]
    assert ib.status is IssueStatus.OPEN
    done = ws.qc.fixes.apply_safe_fixes(code_prefix="sync.caption")
    assert [r.issue_id for r in done] == [ib.issue_id] and other.status is IssueStatus.OPEN
    undo_text = ws.commands.undo_text
    assert undo_text == "QC fix: Fix All Caption Timing"


def test_permissions_never_and_confirm_are_respected(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, -0.2)
    before = state_json(ws)

    p.qc_settings.fix_permissions["caption.retime"] = "never"
    with pytest.raises(QCError, match="switched off"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert ws.qc.fixes.apply_safe_fixes() == [] and "switched off" in ws.qc.fixes.last_skipped[0].reason
    assert ws.qc.fixes.preview_fix(iss.issue_id).blocked_reason

    p.qc_settings.fix_permissions["caption.retime"] = "confirm"  # allowed, but never without a click
    assert ws.qc.fixes.apply_safe_fixes() == []
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    assert ws.qc.fixes.preview_fix(iss.issue_id).needs_confirmation
    assert state_json(ws) == before

    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert p.timeline.get_clip(cap.id).timeline_start == pytest.approx(4.8)


def test_batch_rolls_back_completely_when_one_member_fails(fws, monkeypatch):
    ws, p = fws, fws.project
    caps = [add_caption(p, 2.0 + 4.0 * i, f"Rollback caption {i} here", scene=ws.s1) for i in range(3)]
    issues = [retime_issue(ws, c, -0.2) for c in caps]
    before, depth = state_json(ws), undo_count(ws)
    real = fe.MarkIssueFixedCommand

    class Boom(real):
        def do(self):
            if self.issue_id == issues[2].issue_id:
                raise RuntimeError("disk on fire")
            super().do()

    monkeypatch.setattr(fe, "MarkIssueFixedCommand", Boom)
    with pytest.raises(QCError, match="nothing was changed"):
        ws.qc.fixes.apply_safe_fixes()
    assert state_json(ws) == before and undo_count(ws) == depth and all(i.status is IssueStatus.OPEN for i in issues) and p.qc_fixes == []


def test_fix_similar_is_one_undo_step_and_respects_locks(fws):
    ws, p = fws, fws.project
    caps = [add_caption(p, 2.0 + 4.0 * i, f"Similar caption {i} goes here", scene=ws.s1) for i in range(3)]
    issues = [retime_issue(ws, c, 1.4) for c in caps]  # all large: they need confirmation
    caps[1].locked = True
    before, depth = state_json(ws), undo_count(ws)
    with pytest.raises(QCError, match="Nothing could be fixed"):
        ws.qc.fixes.fix_similar(issues[0].issue_id)  # unconfirmed: none of them is safe
    assert state_json(ws) == before
    done = ws.qc.fixes.fix_similar(issues[0].issue_id, confirmed=True)
    assert sorted(r.issue_id for r in done) == sorted([issues[0].issue_id, issues[2].issue_id])  # the locked one is skipped
    assert undo_count(ws) == depth + 1 and all(r.confirmed_by_user for r in done)
    ws.commands.undo()
    assert state_json(ws) == before


# ================================================================================================ caption.safe_margin / restyle
def test_caption_safe_margin_moves_inside_the_margins(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = issue(ws, "caption.safe_margin", fc.caption_safe_margin(cap.id, [0.5, 0.99], p.qc_settings), clip=cap)
    before = state_json(ws)
    ws.qc.fixes.apply_fix(iss.issue_id)
    c = p.timeline.get_clip(cap.id)
    assert c.text["position"] == "custom" and c.text["position_xy"] == [0.5, pytest.approx(1.0 - p.caption_settings.safe_margin_bottom)]
    assert c.text["text"] == cap.text["text"] and c.text["words"] == cap.text["words"]
    ws.commands.undo()
    assert state_json(ws) == before


def test_caption_restyle_needs_confirmation_and_marks_the_field_as_chosen(fws):
    ws, p = fws, fws.project
    iss = issue(ws, "caption.small_font", fc.caption_restyle("size", True, "Use large caption text", p.qc_settings))
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.needs_confirmation and pv.before == {"large_text": False} and pv.after == {"large_text": True} and "from now on" in pv.changes[0]
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert p.caption_settings.large_text and "large_text" in p.caption_settings.user_set
    ws.commands.undo()
    assert not p.caption_settings.large_text and iss.status is IssueStatus.OPEN
    p.caption_settings.user_set.append("large_text")  # the user chose this setting on purpose
    with pytest.raises(QCError, match="chose"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    bad = issue(ws, "caption.x", fc.caption_restyle("max_lines", 5, "Five lines", p.qc_settings))
    with pytest.raises(QCError, match="1 or 2"):
        ws.qc.fixes.apply_fix(bad.issue_id, confirmed=True)


# ================================================================================================ audio
def music(ws, *, volume: float = 1.0, created_by: str = "AI", keyframes=None, duck_decision: bool = True, locked: bool = False):
    p = ws.project
    asset = add_asset(p, "bed.wav", "audio", duration=30, w=None, h=None)
    c = add_clip(p, "track_a2", asset, 0.0, 20.0, created_by=created_by, slot="music:mus1:0", audio={"role": "MUSIC", "volume": volume, "fade_in": 0.5, "fade_out": 1.0, "ducking": True},
                 metadata={"phase": 5, "assignment_id": "mus1"}, keyframes=keyframes or [], locked=locked)
    if duck_decision:
        d = add_pres_decision(p, c, PresentationType.DUCKING, parameters={"assignment_id": "mus1"}, decision_id="pdec_duck01")
        d.slot, c.ai_decision_id = "duck:mus1", ""
    return asset, c


def duck_issue(ws, spans=((5.0, 8.0), (12.0, 14.0)), gain=0.1):
    return issue(ws, "audio.insufficient_ducking", fc.audio_duck("MUSIC", "mus1", [list(s) for s in spans], gain, ws.project.qc_settings), category=QCCategory.AUDIO)


def eff(c, t, track_volume=1.0):
    return c.audio["volume"] * track_volume * value_at([k for k in c.keyframes if k.property == "volume"], "volume", t - c.timeline_start)


def test_audio_duck_adds_volume_keyframes_only(fws):
    ws, p = fws, fws.project
    asset, clip = music(ws, volume=0.5)
    iss = duck_issue(ws)
    assert iss.fix.safe
    before, file_before = state_json(ws), (p.asset_path(asset).read_bytes(), asset.to_dict() if hasattr(asset, "to_dict") else None)
    clip_before = clip.snapshot()

    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.safe and "-6.0 dB" in pv.before["music level under the voice"] and pv.after["music level under the voice"] == "-20.0 dB"
    assert state_json(ws) == before

    rec = ws.qc.fixes.apply_fix(iss.issue_id)

    c = p.timeline.get_clip(clip.id)
    assert rec.safe and rec.kind == "audio.duck"
    for t in (5.0, 6.5, 8.0, 12.0, 13.0, 14.0):
        assert eff(c, t) <= 0.1 + 1e-3
    assert eff(c, 2.0) == pytest.approx(0.5) and eff(c, 18.0) == pytest.approx(0.5)  # level outside the speech is untouched
    for attr in ("asset_id", "timeline_start", "duration", "source_in", "source_out", "speed", "slot", "track_id", "created_by"):
        assert getattr(c, attr) == getattr(clip_before, attr)
    assert c.audio == clip_before.audio and c.metadata == clip_before.metadata  # the clip itself is not modified: only keyframes are added
    assert {k.property for k in c.keyframes} == {"volume"} and not any(k.problems(c.duration) for k in c.keyframes)
    assert (p.asset_path(asset).read_bytes(), asset.to_dict() if hasattr(asset, "to_dict") else None) == file_before
    duck = next(d for d in p.presentation_decisions.values() if d.slot == "duck:mus1")
    assert duck.created_by is Creator.USER and duck.parameters["qc_fix"] and {k.decision_id for k in c.keyframes} == {duck.decision_id}
    assert errors(ws) <= set()
    p.validate()

    ws.commands.undo()
    assert state_json(ws) == before and iss.status is IssueStatus.OPEN


def test_audio_duck_keeps_the_existing_automation_and_is_not_applied_twice(fws):
    ws, p = fws, fws.project
    _asset, clip = music(ws, volume=1.0, keyframes=[Keyframe("volume", 0.0, 0.3), Keyframe("volume", 10.0, 0.3)])
    iss = duck_issue(ws, spans=((5.0, 8.0),), gain=0.1)
    ws.qc.fixes.apply_fix(iss.issue_id)
    c = p.timeline.get_clip(clip.id)
    assert eff(c, 2.0) == pytest.approx(0.3) and eff(c, 6.0) <= 0.1 + 1e-3 and eff(c, 9.5) == pytest.approx(0.3, abs=0.02)
    again = duck_issue(ws, spans=((5.0, 8.0),), gain=0.1)  # already ducked: the live state no longer matches the issue
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(again.issue_id)
    # a different stretch of speech on the same music may still be fixed: the first fix made the ducking decision, not the user's hands
    more = duck_issue(ws, spans=((14.0, 16.0),), gain=0.1)
    ws.qc.fixes.apply_fix(more.issue_id)
    assert eff(p.timeline.get_clip(clip.id), 15.0) <= 0.1 + 1e-3 and eff(p.timeline.get_clip(clip.id), 6.0) <= 0.1 + 1e-3


@pytest.mark.parametrize("lock", ["clip", "decision", "user_decision", "user_edited_decision"])
def test_audio_duck_respects_the_mix_lock_and_the_users_own_ducking(fws, lock):
    ws, p = fws, fws.project
    _asset, clip = music(ws, volume=0.5)
    duck = p.presentation_decisions["pdec_duck01"]
    if lock == "clip":
        clip.locked = True
    elif lock == "decision":
        duck.locked = True
    elif lock == "user_decision":
        duck.created_by = Creator.USER
    else:
        duck.created_by, duck.parameters["edited"], duck.parameters["qc_fix"] = Creator.USER, True, True
    iss = duck_issue(ws)
    before = state_json(ws)
    with pytest.raises(QCError):
        ws.qc.fixes.apply_fix(iss.issue_id)
    assert state_json(ws) == before


def test_audio_duck_never_touches_the_voice_role(fws):
    ws = fws
    iss = issue(ws, "audio.x", fc.audio_duck("VOICE", "v", [[1.0, 2.0]], 0.1, ws.project.qc_settings), category=QCCategory.AUDIO)
    with pytest.raises(QCError):
        ws.qc.fixes.apply_fix(iss.issue_id)


def test_audio_level_changes_the_clip_volume_with_confirmation(fws):
    ws, p = fws, fws.project
    asset = add_asset(p, "hit.wav", "audio", duration=1, w=None, h=None)
    sfx = add_clip(p, "track_a3", asset, 3.0, 0.8, created_by="AI", scene=ws.s1, slot="sfx:impact:3.00", audio={"role": "SFX", "volume": 0.9, "category": "IMPACT"}, metadata={"phase": 5})
    add_pres_decision(p, sfx, PresentationType.SFX, parameters={"volume": 0.9})
    iss = issue(ws, "audio.sfx_loud", fc.audio_level(sfx.id, 0.4, "Lower the sound effect", p.qc_settings), clip=sfx, category=QCCategory.AUDIO)
    before = state_json(ws)
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    c = p.timeline.get_clip(sfx.id)
    assert c.audio["volume"] == pytest.approx(0.4) and c.asset_id == sfx.asset_id and c.created_by == "USER"
    assert next(d for d in p.presentation_decisions.values() if d.target_id == sfx.id).parameters["volume"] == pytest.approx(0.4)
    ws.commands.undo()
    assert state_json(ws) == before


# ================================================================================================ clip.extend / gap.close
def test_clip_extend_is_limited_by_the_source_media(fws):
    ws, p = fws, fws.project
    short = add_asset(p, "short.mp4", "video", duration=10.6)
    clip = add_visual(ws, 0.0, 10.0, asset=short, scene=ws.s1)
    iss = issue(ws, "timeline.clip.short", fc.clip_extend(clip.id, 11.0, 1.0, p.qc_settings), clip=clip, category=QCCategory.TIMELINE)
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.after["clip end"] == "10.60 s" and "Limited" in pv.changes[-1]
    assert pv.needs_confirmation  # less than was recommended: shown and confirmed, never applied silently
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    c = p.timeline.get_clip(clip.id)
    assert c.timeline_end == pytest.approx(10.6) and c.source_out == pytest.approx(10.6) and c.source_out <= short.duration + 1e-6
    d = next(d for d in p.editing_decisions.values() if d.target_id == clip.id)
    assert d.created_by is Creator.USER and d.parameters["source_out"] == pytest.approx(10.6)  # the manual-edit sync
    ws.commands.undo()
    assert state_json(ws) == before


def test_clip_extend_is_limited_by_free_space_and_refused_when_blocked(fws):
    ws, p = fws, fws.project
    clip = add_visual(ws, 0.0, 10.0, scene=ws.s1, asset=add_asset(p, "long.mp4", "video", duration=30))
    add_visual(ws, 10.5, 4.0, scene=ws.s1, asset=ws.vid, slot="visual:next")
    iss = issue(ws, "x.extend", fc.clip_extend(clip.id, 11.5, 1.5, p.qc_settings), clip=clip, category=QCCategory.TIMELINE)
    assert not iss.fix.safe  # 1.5 s is more than max_clip_extension_seconds: confirmation needed
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert p.timeline.get_clip(clip.id).timeline_end == pytest.approx(10.5)  # up to the next clip, not over it
    p.validate()
    again = issue(ws, "x.extend", fc.clip_extend(clip.id, 11.0, 0.5, p.qc_settings), clip=clip, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="does not change it"):
        ws.qc.fixes.apply_fix(again.issue_id, confirmed=True)  # the first fix made the clip the user's: QC leaves it alone from now on
    tight = add_visual(ws, 20.0, 4.0, scene=ws.s2, asset=add_asset(p, "long2.mp4", "video", duration=30), slot="visual:tight")
    add_visual(ws, 24.0, 3.0, scene=ws.s2, slot="visual:wall")
    blocked = issue(ws, "x.extend", fc.clip_extend(tight.id, 24.5, 0.5, p.qc_settings), clip=tight, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="another clip on the track is in the way"):
        ws.qc.fixes.apply_fix(blocked.issue_id, confirmed=True)  # no free space at all


def test_clip_extend_refused_when_the_source_is_exhausted(fws):
    ws, p = fws, fws.project
    exact = add_asset(p, "exact.mp4", "video", duration=10.0)
    clip = add_visual(ws, 0.0, 10.0, asset=exact, scene=ws.s1)
    iss = issue(ws, "x.extend", fc.clip_extend(clip.id, 10.5, 0.5, p.qc_settings), clip=clip, category=QCCategory.TIMELINE)
    before = state_json(ws)
    with pytest.raises(QCError, match="source media is too short"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert state_json(ws) == before


def test_clip_extend_of_a_still_image_is_limited_only_by_free_space(fws):
    ws, p = fws, fws.project
    img = add_asset(p, "pic.png", "image", duration=None)
    clip = add_visual(ws, 0.0, 4.0, asset=img, scene=ws.s1, track="track_v3", decision=False)
    iss = issue(ws, "x.extend", fc.clip_extend(clip.id, 4.8, 0.8, p.qc_settings), clip=clip, category=QCCategory.TIMELINE)
    assert iss.fix.safe
    ws.qc.fixes.apply_fix(iss.issue_id)
    assert p.timeline.get_clip(clip.id).duration == pytest.approx(4.8)


def test_gap_close_needs_confirmation_and_closes_the_gap_exactly(fws):
    ws, p = fws, fws.project
    clip = add_visual(ws, 0.0, 8.0, scene=ws.s1, asset=add_asset(p, "g.mp4", "video", duration=30))
    add_visual(ws, 9.0, 5.0, scene=ws.s1, slot="visual:after")
    iss = issue(ws, "timeline.gap.unintended", fc.gap_close(clip.id, 9.0, p.qc_settings), clip=clip, start=8.0, end=9.0, category=QCCategory.TIMELINE)
    assert not iss.fix.safe
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.needs_confirmation and pv.before["gap"] == "1.00 s" and pv.after["gap"] == "0.00 s"
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert p.timeline.get_clip(clip.id).timeline_end == pytest.approx(9.0)
    p.validate()
    ws.commands.undo()
    assert state_json(ws) == before


def test_gap_close_is_refused_when_it_cannot_close_the_whole_gap_or_is_stale(fws):
    ws, p = fws, fws.project
    short = add_asset(p, "s.mp4", "video", duration=8.4)
    clip = add_visual(ws, 0.0, 8.0, scene=ws.s1, asset=short)
    iss = issue(ws, "timeline.gap.unintended", fc.gap_close(clip.id, 9.0, p.qc_settings), clip=clip, start=8.0, end=9.0, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="source media is too short"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)  # a half-closed gap is not what was confirmed
    add_visual(ws, 8.0, 1.0, scene=ws.s1, track="track_v2", slot="visual:filler")  # the user covered the gap on another track
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)


# ================================================================================================ clip.remove_empty / clip.delete
def test_remove_empty_removes_a_zero_length_item_and_undo_restores_it(fws):
    ws, p = fws, fws.project
    empty = add_clip(p, "track_v1", None, 5.0, 0.0, kind="media", scene=ws.s1, created_by="AI", slot="visual:ghost")
    iss = issue(ws, "timeline.clip.zero_duration", fc.clip_remove_empty(empty.id, p.qc_settings), clip=empty, category=QCCategory.TIMELINE)
    assert iss.fix.safe
    with pytest.raises(Exception):
        p.validate()  # a zero-length clip is what blocks saving
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.destructive and pv.after == {"item": "removed"}
    rec = ws.qc.fixes.apply_fix(iss.issue_id)
    assert p.timeline.get_clip(empty.id) is None and rec.kind == "clip.remove_empty"
    assert f"{ws.s1.id}|visual:ghost" in p.timeline_generation.suppressed_slots  # an AI element removed by a fix is not recreated by a regeneration
    p.validate()
    ws.commands.undo()
    assert state_json(ws) == before and p.timeline.get_clip(empty.id) is not None and iss.status is IssueStatus.OPEN
    assert f"{ws.s1.id}|visual:ghost" not in p.timeline_generation.suppressed_slots


def test_remove_empty_refuses_an_item_that_has_content(fws):
    ws, p = fws, fws.project
    full = add_visual(ws, 0.0, 5.0, scene=ws.s1)
    iss = issue(ws, "timeline.clip.empty_content", fc.clip_remove_empty(full.id, p.qc_settings), clip=full, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    assert p.timeline.get_clip(full.id) is not None


def test_remove_empty_does_not_remove_a_user_item(fws):
    ws, p = fws, fws.project
    empty = add_clip(p, "track_v1", None, 5.0, 0.0, kind="media", created_by="USER")
    iss = issue(ws, "timeline.clip.zero_duration", fc.clip_remove_empty(empty.id, p.qc_settings), clip=empty, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="You created or edited"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    assert p.timeline.get_clip(empty.id) is not None


def test_clip_delete_removes_a_verified_duplicate_with_confirmation(fws):
    ws, p = fws, fws.project
    keep = add_visual(ws, 0.0, 5.0, scene=ws.s1, decision=False, created_by="AI")
    dup = add_clip(p, "track_v1", ws.vid, 0.0, 5.0, scene=ws.s1, created_by="AI", slot="visual:dup", id="clip_dup")
    p.timeline.get_track("track_v1").clips.sort(key=lambda c: (c.timeline_start, c.id))
    iss = issue(ws, "timeline.duplicate.element", fc.clip_delete(dup.id, "Delete the duplicate copy", p.qc_settings), clip=dup, category=QCCategory.TIMELINE)
    assert not iss.fix.safe
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.destructive and pv.needs_confirmation
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    rec = ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert rec.checkpoint and p.timeline.get_clip(dup.id) is None and p.timeline.get_clip(keep.id) is not None
    ws.commands.undo()
    assert state_json(ws) == before
    ws.commands.redo()
    assert p.timeline.get_clip(dup.id) is None
    ws.commands.undo()
    p.timeline.get_clip(keep.id).timeline_start = 1.0  # no longer a duplicate: deleting this one would delete the only copy
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)


# ================================================================================================ asset.relink
def test_relink_to_an_identical_copy_is_safe_and_undoable(fws, tmp_path):
    ws, p = fws, fws.project
    asset = add_asset(p, "clip.mp4", "video", duration=5)
    path = p.asset_path(asset)
    path.write_bytes(b"identical media bytes" * 100)
    asset.content_hash, asset.size_bytes = sha256_file(path), path.stat().st_size
    copy = tmp_path / "elsewhere" / "clip_copy.mp4"
    copy.parent.mkdir()
    copy.write_bytes(path.read_bytes())
    path.unlink()  # the original is gone
    iss = issue(ws, "asset.missing", fc.asset_relink(asset.id, str(copy), True, p.qc_settings), category=QCCategory.ASSET)
    assert iss.fix.safe
    old_path, before = asset.path, state_json(ws)
    rec = ws.qc.fixes.apply_fix(iss.issue_id)
    assert rec.kind == "asset.relink" and p.assets.get(asset.id).path == str(copy) and p.asset_path(p.assets.get(asset.id)).is_file()
    assert p.assets.get(asset.id).duration == 5 and p.assets.get(asset.id).content_hash == asset.content_hash
    ws.commands.undo()
    assert p.assets.get(asset.id).path == old_path and iss.status is IssueStatus.OPEN and state_json(ws) == before


def test_relink_refuses_a_file_that_is_not_an_exact_copy(fws, tmp_path):
    ws, p = fws, fws.project
    asset = add_asset(p, "clip.mp4", "video", duration=5)
    path = p.asset_path(asset)
    path.write_bytes(b"original bytes" * 50)
    asset.content_hash, asset.size_bytes = sha256_file(path), path.stat().st_size
    path.unlink()
    other = tmp_path / "other.mp4"
    other.write_bytes(b"different bytes" * 50)  # same length, different content
    iss = issue(ws, "asset.missing", fc.asset_relink(asset.id, str(other), True, p.qc_settings), category=QCCategory.ASSET)
    old_path = asset.path
    with pytest.raises(QCError, match="not an identical copy"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    assert p.assets.get(asset.id).path == old_path
    with pytest.raises(QCError):
        ws.qc.fixes.apply_fix(issue(ws, "asset.missing", fc.asset_relink(asset.id, str(tmp_path / "gone.mp4"), True, p.qc_settings), category=QCCategory.ASSET).issue_id, confirmed=True)
    wrong_type = tmp_path / "notes.txt"
    wrong_type.write_bytes(b"x")
    with pytest.raises(QCError, match="file is needed"):
        ws.qc.fixes.apply_fix(issue(ws, "asset.missing", fc.asset_relink(asset.id, str(wrong_type), True, p.qc_settings), category=QCCategory.ASSET).issue_id, confirmed=True)


def test_relink_of_a_similar_file_always_needs_confirmation_and_the_media_tools(fws, tmp_path):
    ws, p = fws, fws.project
    asset = add_asset(p, "clip.mp4", "video", duration=5)
    p.asset_path(asset).unlink()
    other = tmp_path / "similar.mp4"
    other.write_bytes(b"not really a video")
    iss = issue(ws, "asset.missing", fc.asset_relink(asset.id, str(other), False, p.qc_settings), category=QCCategory.ASSET)
    assert not iss.fix.safe
    info = SimpleNamespace(duration=5.0, width=1280, height=720, coded_width=1280, coded_height=720, rotation=0, fps=25.0, codec="h264", has_audio=False, audio_codec=None, sample_rate=None,
                           channels=None)
    probe = SimpleNamespace(try_probe=lambda path: (info, ""))
    stub = engine(ws, render=SimpleNamespace(relink=SimpleNamespace(probe=probe)))
    before = state_json(ws)
    with pytest.raises(QCError, match="confirm"):
        stub.apply_fix(iss.issue_id)
    assert state_json(ws) == before
    rec = stub.apply_fix(iss.issue_id, confirmed=True)
    a = p.assets.get(asset.id)
    assert rec.confirmed_by_user and a.path == str(other) and (a.width, a.height) == (1280, 720) and a.content_hash == sha256_file(other)
    ws.commands.undo()
    assert p.assets.get(asset.id).width == 1920 and iss.status is IssueStatus.OPEN
    with pytest.raises(QCError, match="media tools"):
        engine(ws, render=SimpleNamespace()).apply_fix(iss.issue_id, confirmed=True)
    with pytest.raises(QCError, match="cannot be used"):  # the real probe rejects a file that is not media
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)


def test_relink_is_stale_when_the_file_is_back(fws, tmp_path):
    ws, p = fws, fws.project
    asset = add_asset(p, "clip.mp4", "video", duration=5)  # the placeholder file exists: nothing is missing
    other = tmp_path / "copy.mp4"
    other.write_bytes(b"\0" * 64)
    iss = issue(ws, "asset.missing", fc.asset_relink(asset.id, str(other), True, p.qc_settings), category=QCCategory.ASSET)
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(iss.issue_id)


# ================================================================================================ param.normalize
def test_param_normalize_clamps_to_the_valid_bounds(fws):
    ws, p = fws, fws.project
    clip = add_visual(ws, 0.0, 6.0, scene=ws.s1, decision=False, scale=50.0, opacity=1.7)
    clip.keyframes = [Keyframe("opacity", 1.0, 2.5), Keyframe("scale", 2.0, 1.2)]
    changes = {"scale": 100.0, "opacity": 1.9, "keyframes": [{"property": "opacity", "time": 1.0, "value": 2.0}]}  # even a recipe asking for too much is clamped
    iss = issue(ws, "timeline.transform.invalid", fc.param_normalize(clip.id, changes, "Clamp the values", p.qc_settings), clip=clip, category=QCCategory.TIMELINE)
    assert iss.fix.safe
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.before["scale"] == "50" and pv.after["scale"] == "20" and pv.after["opacity"] == "1"
    ws.qc.fixes.apply_fix(iss.issue_id)
    c = p.timeline.get_clip(clip.id)
    assert c.scale == 20.0 and c.opacity == 1.0
    assert [k.value for k in c.keyframes if k.property == "opacity"] == [1.0] and [k.value for k in c.keyframes if k.property == "scale"] == [1.2]  # only the invalid keyframe
    ws.commands.undo()
    assert state_json(ws) == before


def test_param_normalize_fixes_a_clip_volume_and_refuses_valid_values(fws):
    ws, p = fws, fws.project
    asset = add_asset(p, "loud.wav", "audio", duration=10, w=None, h=None)
    loud = add_clip(p, "track_a3", asset, 2.0, 3.0, created_by="AI", scene=ws.s1, slot="sfx:x", audio={"role": "SFX", "volume": 9.0})
    iss = issue(ws, "timeline.audio_level.invalid", fc.param_normalize(loud.id, {"audio.volume": 4.0}, "Clamp the volume", p.qc_settings), clip=loud, category=QCCategory.TIMELINE)
    ws.qc.fixes.apply_fix(iss.issue_id)
    assert p.timeline.get_clip(loud.id).audio["volume"] == 4.0
    fine = add_visual(ws, 0.0, 5.0, scene=ws.s1, decision=False)
    stale = issue(ws, "timeline.opacity.invalid", fc.param_normalize(fine.id, {"opacity": 1.0}, "Clamp the opacity", p.qc_settings), clip=fine, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(stale.issue_id)
    weird = issue(ws, "timeline.x", fc.param_normalize(fine.id, {"speed": 3.0}, "Unknown", p.qc_settings), clip=fine, category=QCCategory.TIMELINE)
    with pytest.raises(QCError, match="not available"):
        ws.qc.fixes.apply_fix(weird.issue_id, confirmed=True)


# ================================================================================================ motion / transition
def test_motion_soften_replaces_only_the_motion_keyframes_and_owns_the_zoom_decision(fws):
    ws, p = fws, fws.project
    clip = add_visual(ws, 0.0, 6.0, scene=ws.s1)
    zoom = EditingDecision("dec_zoom1", ws.s1.id, DecisionType.ZOOM, "motion:0", clip.id, 0.0, 6.0, {"kind": "zoom_in", "start_scale": 1.0, "end_scale": 1.35}, "t", 80.0, Creator.AI)
    p.editing_decisions[zoom.decision_id] = zoom
    clip.keyframes = [Keyframe("scale", 0.0, 1.0, "ease_in_out", "dec_zoom1"), Keyframe("scale", 0.4, 1.35, "linear", "dec_zoom1"), Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 0.5, 1.0)]
    new = [{"property": "scale", "time": 0.0, "value": 1.0, "interpolation": "ease_in_out"}, {"property": "scale", "time": 5.0, "value": 1.1, "interpolation": "linear"}]
    iss = issue(ws, "motion.too_fast", fc.motion_soften(clip.id, new, "Slow the zoom", p.qc_settings), clip=clip, category=QCCategory.MOTION)
    assert not iss.fix.safe
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert "1.00 to 1.35 over 0.4 s" in pv.before["movement"] and "1.00 to 1.10 over 5.0 s" in pv.after["movement"]
    with pytest.raises(QCError, match="confirm"):
        ws.qc.fixes.apply_fix(iss.issue_id)
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    c = p.timeline.get_clip(clip.id)
    assert [(k.time, k.value) for k in c.keyframes if k.property == "scale"] == [(0.0, 1.0), (5.0, 1.1)]
    assert [(k.time, k.value) for k in c.keyframes if k.property == "opacity"] == [(0.0, 0.0), (0.5, 1.0)]  # other animation stays
    owned = next(d for d in p.editing_decisions.values() if d.type is DecisionType.ZOOM)
    assert owned.created_by is Creator.USER and owned.parameters["end_scale"] == pytest.approx(1.1) and owned.overrides_decision_id == "dec_zoom1"
    assert {k.decision_id for k in c.keyframes if k.property == "scale"} == {owned.decision_id}
    assert any(o.original.decision_id == "dec_zoom1" for o in p.ai_overrides)
    ws.commands.undo()
    assert state_json(ws) == before
    p.timeline.get_clip(clip.id).keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 5.0, 1.1)]  # already the recommended motion
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)


def test_motion_soften_refuses_keyframes_outside_the_clip(fws):
    ws, p = fws, fws.project
    clip = add_visual(ws, 0.0, 3.0, scene=ws.s1, decision=False)
    iss = issue(ws, "motion.x", fc.motion_soften(clip.id, [{"property": "scale", "time": 5.0, "value": 1.1}], "Too late", p.qc_settings), clip=clip, category=QCCategory.MOTION)
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)


def test_transition_shorten_updates_the_transition_and_its_decision(fws):
    ws, p = fws, fws.project
    clip = add_visual(ws, 6.0, 5.0, scene=ws.s1)
    t_dec = EditingDecision("dec_tr001", ws.s1.id, DecisionType.TRANSITION, "transition:in", clip.id, 6.0, 1.8, {"type": "FADE", "duration": 1.8}, "t", 80.0, Creator.AI)
    p.editing_decisions[t_dec.decision_id] = t_dec
    clip.transition = {"type": "FADE", "duration": 1.8, "decision_id": "dec_tr001"}
    iss = issue(ws, "transition.too_long", fc.transition_shorten(clip.id, 0.6, p.qc_settings), clip=clip, category=QCCategory.TRANSITION)
    before = state_json(ws)
    pv = ws.qc.fixes.preview_fix(iss.issue_id)
    assert pv.before["transition"] == "FADE, 1.80 s" and pv.after["transition"] == "FADE, 0.60 s" and pv.needs_confirmation
    ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)
    c = p.timeline.get_clip(clip.id)
    assert c.transition["duration"] == 0.6 and c.transition["type"] == "FADE"
    owned = p.editing_decisions[c.transition["decision_id"]]
    assert owned.created_by is Creator.USER and owned.parameters["duration"] == 0.6
    ws.commands.undo()
    assert state_json(ws) == before
    with pytest.raises(QCError, match="run QC again"):  # a transition that is already short enough
        p.timeline.get_clip(clip.id).transition["duration"] = 0.5
        ws.qc.fixes.apply_fix(iss.issue_id, confirmed=True)


# ================================================================================================ real projects
@needs_ffmpeg
def test_caption_retime_on_the_demo_project_keeps_all_validators_clean(render_ws):
    ws = render_ws
    p = ws.project
    cap = next(c for c in p.timeline.get_track("track_v6").clips)
    cap.created_by = "AI"  # the demo clips are user-created: make this one an AI caption
    add_pres_decision(p, cap)
    iss = retime_issue(ws, cap, -0.15)
    before, base = state_json(ws), errors(ws)
    ws.qc.fixes.apply_fix(iss.issue_id)
    assert errors(ws) <= base
    p.validate()
    assert ws.render.preflight().can_start
    ws.commands.undo()
    assert state_json(ws) == before


@needs_ffmpeg
def test_a_generated_ai_edit_keeps_the_fix_when_regenerated(pres_ws):
    ws = pres_ws
    p = ws.project
    ws.presentation.generate(["CAPTIONS"])
    assert ws.jobs.wait_idle(120)
    cap = next(c for t in p.timeline.tracks for c in t.clips if c.kind == "caption" and c.created_by == "AI" and p.timeline.neighbours(c)[0] <= c.timeline_start - 0.2)  # room to move
    base = errors(ws)
    iss = retime_issue(ws, cap, -0.12)
    ws.qc.fixes.apply_fix(iss.issue_id)
    fixed = p.timeline.get_clip(cap.id)
    assert fixed.created_by == "USER" and errors(ws) <= base
    ws.presentation.regenerate_captions()
    assert ws.jobs.wait_idle(60)
    kept = p.timeline.get_clip(cap.id)
    assert kept is not None and kept.timeline_start == pytest.approx(cap.timeline_start - 0.12, abs=1e-3)  # a later regeneration did not discard the fix


# ================================================================================================ batches that touch the same element, hooks, checkpoints, plumbing
def test_two_fixes_on_the_same_caption_in_one_batch_both_survive(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    retime = retime_issue(ws, cap, -0.2)
    margin = issue(ws, "caption.safe_margin", fc.caption_safe_margin(cap.id, [0.5, 0.97], p.qc_settings), clip=cap)
    before, depth = state_json(ws), undo_count(ws)
    done = ws.qc.fixes.apply_safe_fixes()
    assert sorted(r.issue_id for r in done) == sorted([retime.issue_id, margin.issue_id]) and undo_count(ws) == depth + 1
    c = p.timeline.get_clip(cap.id)
    assert c.timeline_start == pytest.approx(4.8) and c.text["position"] == "custom"
    ws.commands.undo()
    assert state_json(ws) == before


def test_two_ducking_fixes_for_the_same_music_in_one_batch(fws):
    ws, p = fws, fws.project
    _asset, clip = music(ws, volume=0.5)
    first, second = duck_issue(ws, spans=((5.0, 8.0),)), duck_issue(ws, spans=((12.0, 14.0),))
    before = state_json(ws)
    done = ws.qc.fixes.apply_safe_fixes()
    assert len(done) == 2
    c = p.timeline.get_clip(clip.id)
    assert eff(c, 6.0) <= 0.1 + 1e-3 and eff(c, 13.0) <= 0.1 + 1e-3 and eff(c, 10.0) == pytest.approx(0.5, abs=0.01)
    assert len([d for d in p.presentation_decisions.values() if d.slot == "duck:mus1"]) == 1
    assert first.status is second.status is IssueStatus.FIXED
    ws.commands.undo()
    assert state_json(ws) == before


def test_two_caption_restyles_in_one_batch_both_apply(fws):
    ws, p = fws, fws.project
    a = issue(ws, "caption.settings", fc.caption_restyle("size", True, "Large text", p.qc_settings), title="Settings")
    b = issue(ws, "caption.settings", fc.caption_restyle("lines", 1, "One line", p.qc_settings), title="Settings")
    # same fix kind, same issue code: "fix all similar" applies both and neither overwrites the other
    done = ws.qc.fixes.fix_similar(a.issue_id, confirmed=True)
    assert len(done) == 2 and p.caption_settings.large_text and p.caption_settings.max_lines == 1
    assert set(p.caption_settings.user_set) >= {"large_text", "max_lines"}
    ws.commands.undo()
    assert not p.caption_settings.large_text and p.caption_settings.max_lines == 2 and not p.caption_settings.user_set
    assert a.status is b.status is IssueStatus.OPEN


def test_ownership_falls_back_to_the_manual_edit_commands_without_an_edit_hook(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    dec = add_pres_decision(p, cap)
    iss = retime_issue(ws, cap, -0.2)
    bare = engine(ws, timeline=SimpleNamespace())
    before = state_json(ws)
    bare.apply_fix(iss.issue_id)
    assert p.timeline.get_clip(cap.id).created_by == "USER" and dec.decision_id not in p.presentation_decisions
    ws.commands.undo()
    assert state_json(ws) == before


def test_a_failing_checkpoint_only_blocks_fixes_that_remove_something(fws):
    ws, p = fws, fws.project

    def broken(project, label):
        raise OSError("disk full")

    eng = QCFixEngine(ws.projects, ws.commands.execute, broken, SimpleNamespace(editing=ws.editing, presentation=ws.presentation, timeline=ws.timeline))
    cap = add_caption(p, 5.0, scene=ws.s1)
    big = retime_issue(ws, cap, 1.4)
    rec = eng.apply_fix(big.issue_id, confirmed=True)  # not destructive: it proceeds, without a safety copy
    assert rec.checkpoint == "" and p.timeline.get_clip(cap.id).timeline_start == pytest.approx(6.4)
    keep = add_visual(ws, 0.0, 5.0, scene=ws.s1, decision=False, created_by="AI")
    dup = add_clip(p, "track_v1", ws.vid, 0.0, 5.0, scene=ws.s1, created_by="AI", id="clip_zdup")
    p.timeline.get_track("track_v1").clips.sort(key=lambda c: (c.timeline_start, c.id))
    rm = issue(ws, "timeline.duplicate.element", fc.clip_delete(dup.id, "Delete the duplicate", p.qc_settings), clip=dup, category=QCCategory.TIMELINE)
    before = state_json(ws)
    with pytest.raises(QCError, match="safety copy"):
        eng.apply_fix(rm.issue_id, confirmed=True)
    assert state_json(ws) == before and p.timeline.get_clip(keep.id) is not None


def test_the_service_passes_fixes_through_and_the_record_is_serialisable(fws):
    ws, p = fws, fws.project
    cap = add_caption(p, 5.0, scene=ws.s1)
    iss = retime_issue(ws, cap, -0.2)
    other = retime_issue(ws, add_caption(p, 12.0, "Another caption for the service", scene=ws.s1), 1.4)
    assert ws.qc.preview_fix(iss.issue_id).safe
    ws.qc.apply_fix(iss.issue_id)
    doc = json.loads(json.dumps(p.to_document()))  # the fix record and the issue status persist with the project
    assert [f["issue_id"] for f in doc["qc_fixes"]] == [iss.issue_id]
    assert [i["status"] for i in doc["qc_issues"] if i["issue_id"] == iss.issue_id] == ["FIXED"]
    assert ws.qc.fix_similar(other.issue_id, confirmed=True) and other.status is IssueStatus.FIXED
    more = retime_issue(ws, add_caption(p, 20.0, "Safe caption for the service", scene=ws.s2), -0.2)
    assert [r.issue_id for r in ws.qc.apply_safe_fixes(code_prefix="sync.caption")] == [more.issue_id]


def test_navigate_issues_are_not_part_of_a_safe_batch(fws):
    ws = fws
    nav = issue(ws, "visual.x", fc.navigate("visual.replace", "Replace it", scene_id=ws.s1.id), category=QCCategory.VISUAL_ACCURACY)
    assert ws.qc.fixes.apply_safe_fixes() == [] and ws.qc.fixes.last_skipped == [] and nav.status is IssueStatus.OPEN


def test_a_fix_that_would_add_a_validation_error_is_refused_and_the_rest_of_the_batch_goes_ahead(fws):
    ws, p = fws, fws.project
    voice = add_asset(p, "voice.wav", "audio", duration=30, w=None, h=None)
    p.voice_over.asset_id = voice.id
    good = add_caption(p, 4.0, scene=ws.s1)
    late = add_caption(p, 29.0, scene=ws.s2)  # ends at 30.5: moving it 0.6 s later would end after the voice-over + 1 s
    ok_issue, bad_issue = retime_issue(ws, good, -0.2), retime_issue(ws, late, 0.6)
    assert bad_issue.fix.safe
    before, base = state_json(ws), errors(ws)
    with pytest.raises(QCError, match="new problem"):
        ws.qc.fixes.apply_fix(bad_issue.issue_id)
    assert state_json(ws) == before
    done = ws.qc.fixes.apply_safe_fixes()
    assert [r.issue_id for r in done] == [ok_issue.issue_id]
    assert "new problem" in ws.qc.fixes.last_skipped[0].reason and bad_issue.status is IssueStatus.OPEN
    assert errors(ws) <= base
    ws.commands.undo()
    assert state_json(ws) == before


def test_every_command_kind_of_the_catalog_has_a_handler_or_a_clear_reason(fws):
    eng = fws.qc.fixes
    assert set(fc.HANDLED_KINDS) == set(eng._handlers) | set(fe.UNSUPPORTED)
    for kind in fe.UNSUPPORTED:
        assert kind in fc.HANDLED_KINDS


def test_a_malformed_recipe_is_refused_as_stale_not_a_crash(fws):
    ws = fws
    spec = fc.caption_retime("x", 1.0, 2.0, 0.1, ws.project.qc_settings)
    spec.params.pop("clip_id")
    i = issue(ws, "sync.caption_drift", spec)
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(i.issue_id, confirmed=True)
    assert ws.qc.fixes.preview_fix(i.issue_id).blocked_reason == STALE


def test_restyle_and_level_fixes_that_are_already_in_place_are_stale(fws):
    ws, p = fws, fws.project
    same = issue(ws, "caption.small_font", fc.caption_restyle("large_text", False, "No change", p.qc_settings))
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(same.issue_id, confirmed=True)
    asset = add_asset(p, "hit.wav", "audio", duration=1, w=None, h=None)
    sfx = add_clip(p, "track_a3", asset, 3.0, 0.8, created_by="AI", scene=ws.s1, slot="sfx:x", audio={"role": "SFX", "volume": 0.4})
    lvl = issue(ws, "audio.sfx_loud", fc.audio_level(sfx.id, 0.4, "Already there", p.qc_settings), clip=sfx, category=QCCategory.AUDIO)
    with pytest.raises(QCError, match="run QC again"):
        ws.qc.fixes.apply_fix(lvl.issue_id, confirmed=True)


def test_a_sound_effect_over_speech_is_ducked_with_keyframes_and_handed_to_the_user(fws):
    ws, p = fws, fws.project
    asset = add_asset(p, "hit.wav", "audio", duration=4, w=None, h=None)
    sfx = add_clip(p, "track_a3", asset, 10.0, 3.0, created_by="AI", scene=ws.s1, slot="sfx:impact:10.00", audio={"role": "SFX", "volume": 0.8, "category": "IMPACT"}, metadata={"phase": 5, "sfx_id": "sfx1"})
    dec = add_pres_decision(p, sfx, PresentationType.SFX)
    iss = issue(ws, "audio.sfx_over_speech", fc.audio_duck("SFX", "sfx1", [[10.5, 12.0]], 0.2, p.qc_settings), clip=sfx, category=QCCategory.AUDIO)
    before = state_json(ws)
    ws.qc.fixes.apply_fix(iss.issue_id)
    c = p.timeline.get_clip(sfx.id)
    assert eff(c, 11.0) <= 0.2 + 1e-3 and c.audio == sfx.audio and c.created_by == "USER"
    assert dec.decision_id not in p.presentation_decisions
    ws.commands.undo()
    assert state_json(ws) == before
