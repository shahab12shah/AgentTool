"""QC foundations: severity, scoring, issue model, settings, the engine (cache, scene reuse, failure isolation, cancel), history, report, persistence and the snapshot."""

from __future__ import annotations

import json

import pytest

from app.project.phase8_commands import IgnoreIssuesCommand, MarkIssueFixedCommand, SetQCSettingsCommand, StoreQCRunCommand, UnignoreCommand
from app.project.project import Project
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import QCCancelled, QCContext, snapshot_project
from app.qc.issue_model import CheckerState, FixRecord, IgnoreRecord, IssueStatus, QCCategory, QCFixSpec, QCIssue
from app.qc.qc_engine import PreviousState, QCEngine, classify
from app.qc.qc_history import compare_entries
from app.qc.report_generator import build_report
from app.qc.scoring import compute_scores, decide_export, size_factor
from app.qc.settings import QCSettings
from app.qc.severity import RANK, BlockLevel, Severity, blocks_export, cap_for_confidence, priority_key
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, narrate, new_project, qc_ctx


def issue(sev=Severity.WARNING, code="x.y", cat=QCCategory.SYNC, scene="scene_001", conf=100.0, start=1.0, end=2.0, **kw) -> QCIssue:
    i = QCIssue(f"qci_{code}_{scene}_{start}", code, cat, sev, "Title " + code, "desc", scene, kw.pop("item", None), None, start, end, conf, kw.pop("source", "deterministic:test"), **kw)
    i.fingerprint = i.make_fingerprint()
    return i


# ------------------------------------------------------------------ severity
def test_severity_order_blocking_levels_and_confidence_caps():
    assert [RANK[s] for s in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.NOTICE, Severity.INFO)] == [0, 1, 2, 3, 4]
    assert blocks_export(Severity.CRITICAL, "CRITICAL") and not blocks_export(Severity.ERROR, "CRITICAL")
    assert blocks_export(Severity.ERROR, BlockLevel.CRITICAL_ERROR) and not blocks_export(Severity.WARNING, BlockLevel.CRITICAL_ERROR)
    assert blocks_export(Severity.WARNING, "CRITICAL_ERROR_WARNING") and not blocks_export(Severity.NOTICE, "CRITICAL_ERROR_WARNING")
    caps = ((50.0, "NOTICE"), (70.0, "WARNING"))
    assert cap_for_confidence(Severity.ERROR, 40, caps) is Severity.NOTICE and cap_for_confidence(Severity.ERROR, 65, caps) is Severity.WARNING
    assert cap_for_confidence(Severity.ERROR, 90, caps) is Severity.ERROR and cap_for_confidence(Severity.NOTICE, 10, caps) is Severity.NOTICE  # a cap never raises severity
    # same severity: viewer impact, then duration, then confidence, then scene importance
    assert priority_key(Severity.ERROR, 0.9, 1, 80, 0.5) < priority_key(Severity.ERROR, 0.5, 9, 99, 0.9) < priority_key(Severity.WARNING, 1.0, 99, 100, 1.0)


def test_classify_caps_judgements_but_never_deterministic_findings():
    s = QCSettings()
    ai = classify(issue(Severity.ERROR, conf=60, source="ai:local"), s)
    assert ai.severity is Severity.WARNING and ai.metrics["original_severity"] == "ERROR"
    assert classify(issue(Severity.ERROR, conf=100), s).severity is Severity.ERROR


# ------------------------------------------------------------------ scoring and the export gate
def test_a_good_score_never_hides_a_critical_issue():
    crit = [issue(Severity.CRITICAL, "preflight.voice_missing", QCCategory.PREFLIGHT, None, start=None, end=None) for _ in range(2)]
    scores = compute_scores(crit, QCSettings(), 60.0, 20)
    assert scores.counts["CRITICAL"] == 2 and scores.status == "BLOCKED" and scores.export == "BLOCKED" and "critical" in scores.status_label and scores.overall > 60  # the number alone looks fine
    assert decide_export(crit, "CRITICAL", True).blocked and not decide_export(crit, "CRITICAL", True).overridable  # a CRITICAL is never overridable


def test_export_decision_follows_the_block_level_and_override_setting():
    errs = [issue(Severity.ERROR)]
    warns = [issue(Severity.WARNING, "w.x")]
    assert decide_export(errs, "CRITICAL_ERROR").blocked and not decide_export(errs, "CRITICAL").blocked
    assert decide_export(errs, "CRITICAL_ERROR", True).overridable and not decide_export(errs, "CRITICAL_ERROR", False).overridable
    d = decide_export(warns, "CRITICAL_ERROR")
    assert d.status == "AVAILABLE" and "warnings remain: 1" in d.message
    assert decide_export(warns, "CRITICAL_ERROR_WARNING").blocked
    assert decide_export([], "CRITICAL_ERROR").status == "READY"


def test_scores_ignored_and_fixed_issues_stop_counting_and_unavailable_groups_are_not_100():
    a, b = issue(Severity.ERROR, "e.1"), issue(Severity.ERROR, "e.2", QCCategory.CAPTION)
    base = compute_scores([a, b], QCSettings(), 60, 12)
    a.status = IssueStatus.FIXED
    b.ignored_by_user = True
    assert compute_scores([a, b], QCSettings(), 60, 12).overall > base.overall and compute_scores([a, b], QCSettings(), 60, 12).counts["ERROR"] == 0
    partial = compute_scores([], QCSettings(), 60, 12, failed_groups=["audio"], failed_checkers=["audio"])
    assert partial.groups["audio"] == 0 and "audio" in partial.unavailable and partial.status != "READY"  # a partial analysis is never "Ready"
    assert size_factor(5) == 1.0 and size_factor(400) == 4.0


def test_status_labels():
    s = QCSettings()
    assert compute_scores([], s, 60, 10).status_label == "100/100 — Ready"
    warns = [issue(Severity.WARNING, f"w.{i}", QCCategory.PACING) for i in range(40)]
    assert compute_scores(warns, s, 60, 12).status in ("REVIEW", "FIX_REQUIRED")


# ------------------------------------------------------------------ issue model, ignores, settings
def test_issue_round_trip_fingerprint_stability_and_ignore_matching():
    i = issue(fix=None)
    i.fix = QCFixSpec("caption.retime", {"clip_id": "c1"}, True, False)
    again = QCIssue.from_dict(json.loads(json.dumps(i.to_dict())))
    assert again.fix.kind == "caption.retime" and again.severity is Severity.WARNING and again.category is QCCategory.SYNC and again.fingerprint == i.fingerprint
    moved = issue(start=1.2, end=2.2)  # a tiny shift in time is still the same issue
    assert moved.fingerprint == i.fingerprint
    assert issue(start=9.0, end=10.0).fingerprint != i.fingerprint
    assert IgnoreRecord("g1", "issue", i.fingerprint).matches(i) and not IgnoreRecord("g2", "issue", "other").matches(i)
    assert IgnoreRecord("g3", "type", code="x.y").matches(i) and not IgnoreRecord("g4", "type", code="x.y", scene_id="scene_999").matches(i)
    assert i.compact()["fp"] == i.fingerprint and "description" not in i.compact()


def test_settings_round_trip_paths_and_subset_hash():
    s = QCSettings()
    s.set_path("sync.major_ms", 800)
    s.set_path("caption.max_chars_per_line", "36")
    assert s.get_path("sync.major_ms") == 800.0 and s.get_path("caption.max_chars_per_line") == 36
    again = QCSettings.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again.to_dict() == s.to_dict() and again.version() == s.version()
    h = s.subset_hash("sync")
    s.set_path("caption.max_cps", 30)
    assert s.subset_hash("sync") == h and s.subset_hash("caption") != QCSettings().subset_hash("caption")  # a checker's cache key covers only what it reads


# ------------------------------------------------------------------ the engine, with fake checkers
class Fake(BaseChecker):
    scene_local = False
    domains = ("timeline",)

    def __init__(self, cid="fake", issues=None, fail=False, expensive=False, cats=(QCCategory.TIMELINE,), integrity=True, cancel_after=False):
        self.id, self.label, self.categories, self.expensive = cid, cid.title(), cats, expensive
        self._issues, self.fail, self.integrity, self.cancel_after = issues or [], fail, integrity, cancel_after
        self.runs = 0

    def run(self, ctx, report):
        self.runs += 1
        ctx.check_cancel()
        if self.fail:
            raise RuntimeError("boom")
        return CheckerOutput([QCIssue.from_dict(i.to_dict()) for i in self._issues], {"integrity_ok": self.integrity, "n": len(self._issues)}, ["checked"])


@pytest.fixture
def proj(tmp_path):
    p = new_project(tmp_path, seconds=30)
    a = add_asset(p, "a.mp4", "video", duration=20)
    s1 = add_scene(p, 0, 15, "Silver rose sharply this week.", importance=0.8)
    s2 = add_scene(p, 15, 30, "Demand keeps growing steadily.")
    narrate(p)
    add_clip(p, "track_v1", a, 0, 15, scene=s1)
    add_clip(p, "track_v1", a, 15, 15, scene=s2)
    p.s1, p.s2, p.a = s1, s2, a
    return p


def test_one_failing_checker_does_not_fail_the_run_and_others_continue(proj):
    ok, bad, after = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)]), Fake("audio", fail=True, cats=(QCCategory.AUDIO,)), Fake("caption", cats=(QCCategory.CAPTION,))
    res = QCEngine([ok, bad, after]).run(qc_ctx(proj))
    st = res.run.checkers
    assert st["audio"].state is CheckerState.FAILED and "boom" in st["audio"].error and st["timeline"].state is CheckerState.DONE and st["caption"].state is CheckerState.DONE
    assert res.run.state == "PARTIAL" and res.run.failed_checkers() == ["audio"] and len(res.run.issues) == 1
    assert "audio" in res.run.scores.unavailable and res.run.scores.status != "READY"
    assert after.runs == 1  # it ran after the failure


def test_unchanged_inputs_are_served_from_cache_and_changes_rerun(proj):
    chk = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)])
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(proj))
    assert chk.runs == 1 and first.run.checkers["timeline"].state is CheckerState.DONE
    second = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
    assert chk.runs == 1 and second.run.checkers["timeline"].state is CheckerState.CACHED and second.run.cache_hits == 1 and len(second.run.issues) == 1
    add_clip(proj, "track_v2", proj.a, 3, 2)  # the timeline changed
    third = eng.run(qc_ctx(proj), previous=PreviousState(second.run.issues, second.cache))
    assert chk.runs == 2 and third.run.checkers["timeline"].state is CheckerState.DONE
    forced = eng.run(qc_ctx(proj), previous=PreviousState(third.run.issues, third.cache), use_cache=False)
    assert chk.runs == 3 and forced.run.checkers["timeline"].state is CheckerState.DONE


class SceneFake(Fake):
    scene_local = True
    domains = ("timeline", "scenes")

    def __init__(self):
        super().__init__("scene", cats=(QCCategory.SCENE_COVERAGE,))
        self.seen: list[list[str]] = []

    def run(self, ctx, report):
        self.runs += 1
        ids = [s.id for s in ctx.target_scenes()]
        self.seen.append(ids)
        return CheckerOutput([self._mk(sid) for sid in ids], {"n": len(ids)})

    @staticmethod
    def _mk(sid):
        i = issue(Severity.WARNING, "scene.x", QCCategory.SCENE_COVERAGE, sid)
        i.checker = "scene"
        return i


def test_only_the_changed_scene_is_reanalysed(proj):
    chk = SceneFake()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(proj))
    assert chk.seen == [[proj.s1.id, proj.s2.id]] and len(first.run.issues) == 2
    add_clip(proj, "track_v2", proj.a, 20, 3, scene=proj.s2)  # touches scene 2 only
    second = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
    assert chk.seen[-1] == [proj.s2.id] and second.run.checkers["scene"].reused_scenes == 1 and {i.scene_id for i in second.run.issues} == {proj.s1.id, proj.s2.id}
    # run_scene_qc: an explicit scene filter re-analyses exactly that scene and keeps everything else
    third = eng.run(qc_ctx(proj, scene_filter=[proj.s1.id]), previous=PreviousState(second.run.issues, second.cache))
    assert chk.seen[-1] == [proj.s1.id] and len(third.run.issues) == 2


def test_broken_integrity_skips_expensive_checkers_but_cheap_ones_still_run(proj):
    pre = Fake("preflight", [issue(Severity.CRITICAL, "preflight.x", QCCategory.PREFLIGHT, None, start=None, end=None)], cats=(QCCategory.PREFLIGHT,), integrity=False)
    cheap, costly = Fake("timeline"), Fake("visual", cats=(QCCategory.VISUAL_ACCURACY,), expensive=True)
    res = QCEngine([pre, cheap, costly]).run(qc_ctx(proj))
    assert res.run.checkers["visual"].state is CheckerState.SKIPPED and "preflight" in res.run.checkers["visual"].message.lower() and cheap.runs == 1 and costly.runs == 0
    assert res.run.scores.counts["CRITICAL"] == 1


def test_cancel_keeps_finished_checkers_and_previous_findings_for_the_rest(proj):
    a, b, c = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)]), Fake("scene", cats=(QCCategory.SCENE_COVERAGE,)), Fake("sync", cats=(QCCategory.SYNC,))
    prev_issue = issue(Severity.ERROR, "sync.old", QCCategory.SYNC)
    prev_issue.checker = "sync"
    ctx = qc_ctx(proj)

    class Canceler(Fake):
        def run(self, ctx, report):
            ctx.cancel.set()
            ctx.check_cancel()

    res = QCEngine([a, Canceler("scene", cats=(QCCategory.SCENE_COVERAGE,)), c]).run(ctx, previous=PreviousState([prev_issue], {}))
    st = res.run.checkers
    assert st["timeline"].state is CheckerState.DONE and st["scene"].state is CheckerState.CANCELED and st["sync"].state is CheckerState.CANCELED
    assert res.run.state == "CANCELED" and any(i.code == "sync.old" for i in res.run.issues) and any(i.code == "t.1" for i in res.run.issues)
    _ = (b, QCCancelled)


def test_category_selection_and_previous_findings_of_unselected_checkers_are_kept(proj):
    t, c = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)]), Fake("caption", cats=(QCCategory.CAPTION,))
    eng = QCEngine([t, c])
    full = eng.run(qc_ctx(proj))
    sel = eng.select(proj.qc_settings, categories=["CAPTION"])
    assert sel == {"caption"}
    part = eng.run(qc_ctx(proj), previous=PreviousState(full.run.issues, full.cache), selected=sel, use_cache=False)
    assert part.run.checkers["timeline"].state is CheckerState.SKIPPED and any(i.code == "t.1" for i in part.run.issues)


def test_aggregation_applies_ignores_drops_low_confidence_ai_noise_and_dedupes(proj):
    i1 = issue(Severity.ERROR, "dup.x")
    i2 = issue(Severity.ERROR, "dup.x")  # same fingerprint
    noise = issue(Severity.WARNING, "ai.noise", QCCategory.EDITORIAL, conf=20, source="ai:local")
    keep = issue(Severity.WARNING, "ai.keep", QCCategory.EDITORIAL, conf=60, source="ai:local")
    ignore = IgnoreRecord("ig", "type", code="ai.keep", reason="intentional")
    res = QCEngine([Fake("timeline", [i1, i2, noise, keep])]).run(qc_ctx(proj), ignores=[ignore])
    codes = {i.code: i for i in res.run.issues}
    assert set(codes) == {"dup.x", "ai.keep"} and codes["ai.keep"].ignored_by_user and codes["ai.keep"].status is IssueStatus.IGNORED and not codes["ai.keep"].active
    assert res.run.scores.counts["ERROR"] == 1 and res.run.scores.counts["WARNING"] == 0


# ------------------------------------------------------------------ persistence, commands, history, report, snapshot
def test_schema_v8_round_trip_and_migration(proj):
    i = issue()
    i.fix = QCFixSpec("caption.retime", {"clip_id": "c"}, True, False)
    res = QCEngine([Fake("timeline", [i])]).run(qc_ctx(proj))
    proj.qc_issues, proj.qc_scores = res.run.issues, res.run.scores
    proj.qc_runs, proj.qc_history, proj.qc_cache = [res.run.record()], [res.run.history_entry([], [])], res.cache
    proj.qc_ignored_issues = [IgnoreRecord("g", "issue", i.fingerprint, reason="keep")]
    proj.qc_fixes = [FixRecord("f1", i.issue_id, i.code, "caption.retime", summary="moved")]
    proj.render_qc_results = {"render_1": {"status": "PASSED"}}
    doc = json.loads(json.dumps(proj.to_document()))
    assert doc["schema_version"] == 8 and {"qc_settings", "qc_runs", "qc_issues", "qc_scores", "qc_ignored_issues", "qc_fixes", "qc_history", "render_qc_results"} <= set(doc)
    q = Project.from_document(doc, root=proj.root)
    assert q.qc_issues[0].fix.kind == "caption.retime" and q.qc_runs[0]["run_id"] == res.run.run_id and q.qc_ignored_issues[0].reason == "keep" and q.render_qc_results["render_1"]["status"] == "PASSED"
    assert q.qc_scores.overall == res.run.scores.overall and q.qc_fixes[0].summary == "moved" and q.qc_history[0]["qc_run_id"] == res.run.run_id
    old = dict(doc)
    old["schema_version"] = 7
    for k in ("qc_settings", "qc_runs", "qc_issues", "qc_scores", "qc_ignored_issues", "qc_fixes", "qc_history", "qc_cache", "render_qc_results"):
        old.pop(k)
    m = Project.from_document(old, root=proj.root)
    assert m.schema_version == 8 and m.qc_issues == [] and m.qc_scores is None and m.qc_settings.block_level == "CRITICAL_ERROR"


def test_qc_commands_are_undoable_and_leave_the_timeline_alone(proj):
    before = json.dumps(proj.timeline.to_dict(), sort_keys=True)
    i = issue()
    res = QCEngine([Fake("timeline", [i])]).run(qc_ctx(proj))
    store = StoreQCRunCommand(proj, res.run.issues, res.run.scores, res.run.record(), res.run.history_entry([], []), res.cache)
    store.do()
    assert len(proj.qc_issues) == 1 and len(proj.qc_runs) == 1
    rec = IgnoreRecord("ig1", "issue", proj.qc_issues[0].fingerprint, reason="intentional")
    ign = IgnoreIssuesCommand(proj, rec)
    ign.do()
    assert proj.qc_issues[0].ignored_by_user and proj.qc_issues[0].status is IssueStatus.IGNORED and proj.qc_ignored_issues[0].reason == "intentional"
    ign.undo()
    assert not proj.qc_issues[0].ignored_by_user and proj.qc_issues[0].status is IssueStatus.OPEN and proj.qc_ignored_issues == []
    ign.do()
    un = UnignoreCommand(proj, "ig1")
    un.do()
    assert not proj.qc_issues[0].ignored_by_user and proj.qc_ignored_issues == []
    un.undo()
    assert proj.qc_issues[0].ignored_by_user
    fixed = MarkIssueFixedCommand(proj, proj.qc_issues[0].issue_id, FixRecord("f1", proj.qc_issues[0].issue_id, "x.y", "caption.retime"))
    fixed.do()
    assert proj.qc_issues[0].status is IssueStatus.FIXED and len(proj.qc_fixes) == 1
    fixed.undo()
    assert proj.qc_issues[0].status is IssueStatus.IGNORED and proj.qc_fixes == []
    s = QCSettings()
    s.block_level = "CRITICAL"
    sc = SetQCSettingsCommand(proj, s)
    sc.do()
    assert proj.qc_settings.block_level == "CRITICAL"
    sc.undo()
    assert proj.qc_settings.block_level == "CRITICAL_ERROR"
    store.undo()
    assert proj.qc_issues == [] and proj.qc_runs == []
    assert json.dumps(proj.timeline.to_dict(), sort_keys=True) == before


def test_history_comparison_shows_what_improved(proj):
    eng = QCEngine([Fake("timeline", [issue(Severity.ERROR, "e.1", QCCategory.TIMELINE), issue(Severity.WARNING, "w.1", QCCategory.TIMELINE)])])
    r1 = eng.run(qc_ctx(proj), number=4)
    r2 = QCEngine([Fake("timeline", [issue(Severity.WARNING, "w.1", QCCategory.TIMELINE), issue(Severity.NOTICE, "n.1", QCCategory.TIMELINE)])]).run(qc_ctx(proj), number=5)
    c = compare_entries(r1.run.history_entry([], []), r2.run.history_entry([], []))
    assert (c.older, c.newer) == (4, 5) and c.improved and c.overall[1] > c.overall[0] and c.counts["ERROR"] == (1, 0)
    assert [i["code"] for i in c.resolved_issues] == ["e.1"] and [i["code"] for i in c.new_issues] == ["n.1"] and c.persisting == 1
    assert any(line.startswith("Overall:") and "→" in line for line in c.lines()) and "QC Run #4 vs #5" in c.lines()[0]


def test_report_has_every_section_and_no_hidden_reasoning(proj):
    i = issue(Severity.ERROR, "sync.major", QCCategory.SYNC, why_it_matters="Viewers notice", suggested_fix="Move the caption earlier", current_value="+620 ms", recommended_value="+80 ms")
    res = QCEngine([Fake("timeline", [i])]).run(qc_ctx(proj))
    text = build_report("Demo", res.run.record(), res.run.scores, res.run.issues, [FixRecord("f", "i", "c.x", "caption.retime", summary="moved a caption")], [IgnoreRecord("g", "issue", "z", reason="deliberate")],
                        {"scene_001": "Scene 1"})
    for heading in ("## Project", "## QC run", "## Overall score", "## Critical issues", "## Errors", "## Warnings", "## Notices", "## Category scores", "## Scene-by-scene findings", "## Applied fixes",
                    "## Ignored issues", "## Export readiness", "## Recommendations"):
        assert heading in text, heading
    assert "Why it matters: Viewers notice" in text and "+620 ms" in text and "moved a caption" in text and "deliberate" in text and "EXPORT BLOCKED" in text
    assert "chain of thought" not in text.lower() and "does not verify facts" in text


def test_snapshot_is_detached_from_the_live_project(proj):
    ctx = QCContext.build(proj)  # detach=True
    before = json.dumps(ctx.timeline.to_dict(), sort_keys=True)
    add_clip(proj, "track_v2", proj.a, 3, 2)  # the live project keeps changing
    proj.scenes[0].importance = 0.1
    assert json.dumps(ctx.timeline.to_dict(), sort_keys=True) == before and ctx.scene(proj.s1.id).importance == 0.8
    assert len(ctx.visual_clips()) == 2 and ctx.root == proj.root and ctx.duration == 30.0 and ctx.frame == pytest.approx(1 / 30)
    snap = snapshot_project(proj)
    assert snap is not proj and snap.qc_issues == []


def test_context_protection_rules(proj):
    c1 = add_clip(proj, "track_v1", proj.a, 40, 1, created_by="USER")
    c2 = add_clip(proj, "track_v3", proj.a, 40, 1, locked=True)
    c3 = add_clip(proj, "track_v2", proj.a, 41, 1, created_by="AI")
    ctx = qc_ctx(proj)
    t = lambda tid: proj.timeline.get_track(tid)  # noqa: E731
    assert ctx.is_protected(t("track_v1"), c1)[0] and ctx.is_protected(t("track_v3"), c2)[0] and not ctx.is_protected(t("track_v2"), c3)[0]
    t("track_v2").locked = True
    assert ctx.is_protected(t("track_v2"), c3) == (True, "Track V2 B-Roll is locked")
    t("track_v2").locked = False
    proj.timeline_generation.locked_scenes.append("scene_009")
    c3.scene_id = "scene_009"
    assert ctx.is_protected(t("track_v2"), c3)[0]


def test_issue_factory_disables_auto_fix_for_protected_elements(proj):
    from app.qc import fix_catalog as fc

    class Chk(BaseChecker):
        id = "x"

    c = add_clip(proj, "track_v6", None, 5, 2, kind="caption", created_by="USER", scene_id=proj.s1.id)
    ai = add_clip(proj, "track_v6", None, 8, 2, kind="caption", created_by="AI", scene_id=proj.s1.id)
    ctx = qc_ctx(proj)
    chk = Chk()
    locked = chk.issue("caption.drift", QCCategory.SYNC, Severity.WARNING, "Drift", clip=c, track=proj.timeline.get_track("track_v6"), fix=fc.caption_retime(c.id, 5.1, 7.1, 0.1, ctx.settings), ctx=ctx)
    assert locked.locked and not locked.auto_fix_available and "auto-fix is disabled" in locked.fix_blocked_reason
    free = chk.issue("caption.drift", QCCategory.SYNC, Severity.WARNING, "Drift", clip=ai, track=proj.timeline.get_track("track_v6"), fix=fc.caption_retime(ai.id, 8.1, 10.1, 0.1, ctx.settings), ctx=ctx)
    assert free.auto_fix_available and free.auto_fix_safe and free.scene_id == proj.s1.id and free.fingerprint
    big = chk.issue("caption.drift", QCCategory.SYNC, Severity.ERROR, "Big drift", clip=ai, track=proj.timeline.get_track("track_v6"), fix=fc.caption_retime(ai.id, 9.0, 11.0, 1.0, ctx.settings), ctx=ctx)
    assert big.auto_fix_available and not big.auto_fix_safe  # a large shift needs confirmation
    ctx.settings.fix_permissions["caption.retime"] = "never"
    off = chk.issue("caption.drift", QCCategory.SYNC, Severity.WARNING, "Drift", clip=ai, track=proj.timeline.get_track("track_v6"), fix=fc.caption_retime(ai.id, 8.1, 10.1, 0.1, ctx.settings), ctx=ctx)
    assert not off.auto_fix_available and "Disabled in QC settings" in off.fix_blocked_reason


# ------------------------------------------------------------------ review round: stale caches, carried findings, gate semantics
class CoverageFake(SceneFake):
    """A scene-local checker whose answer depends on the project: a CRITICAL for every scene whose picture is shorter than the scene."""

    def run(self, ctx, report):
        self.runs += 1
        ids = [s.id for s in ctx.target_scenes()]
        self.seen.append(ids)
        out = []
        for sid in ids:
            s = ctx.scene(sid)
            if sum(c.duration for _t, c in ctx.visual_clips() if c.scene_id == sid) < (s.end - s.start) - 0.01:
                i = issue(Severity.CRITICAL, "scene.uncovered", QCCategory.SCENE_COVERAGE, sid)
                i.checker = "scene"
                out.append(i)
        return CheckerOutput(out, {})


def test_a_scene_run_does_not_vouch_for_scenes_it_did_not_look_at(proj):
    """Regression: a scene-only run stored the CURRENT hashes for every scene, so the next full run served the other scenes' OLD findings from the cache (a CRITICAL stayed hidden)."""
    chk = CoverageFake()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(proj))
    assert first.run.issues == [] and first.complete
    proj.timeline.get_track("track_v1").clips[1].duration = 5.0  # scene 2 is now uncovered ...
    scene_run = eng.run(qc_ctx(proj, scene_filter=[proj.s1.id]), previous=PreviousState(first.run.issues, first.cache))  # ... but QC only looked at scene 1
    assert chk.seen[-1] == [proj.s1.id] and scene_run.run.issues == [] and not scene_run.complete  # the run says so: it is not a verdict on the whole project
    full = eng.run(qc_ctx(proj), previous=PreviousState(scene_run.run.issues, scene_run.cache))
    assert [(i.code, i.scene_id) for i in full.run.issues] == [("scene.uncovered", proj.s2.id)] and full.run.scores.status == "BLOCKED" and full.complete
    assert chk.seen[-1] == [proj.s2.id]  # only the scene that was left behind is analysed, scene 1 is still reused


class WithGlobal(CoverageFake):
    def run(self, ctx, report):
        out = super().run(ctx, report)
        g = issue(Severity.NOTICE, "scene.global", QCCategory.SCENE_COVERAGE, None, start=None, end=None)
        g.checker = "scene"
        out.issues.append(g)
        return out


def test_a_scene_run_keeps_every_other_scenes_findings_even_when_a_global_finding_exists(proj):
    proj.timeline.get_track("track_v1").clips[1].duration = 5.0
    chk = WithGlobal()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(proj))
    assert {(i.code, i.scene_id) for i in first.run.issues} == {("scene.global", None), ("scene.uncovered", proj.s2.id)}
    again = eng.run(qc_ctx(proj, scene_filter=[proj.s1.id]), previous=PreviousState(first.run.issues, first.cache))
    assert {(i.code, i.scene_id) for i in again.run.issues} == {("scene.global", None), ("scene.uncovered", proj.s2.id)}  # scene 2's CRITICAL is not dropped
    assert again.complete  # nothing changed since the full run: the scene run is still a verdict on the whole project


def test_findings_carried_over_keep_their_fixed_mark_but_a_redetected_one_reopens(proj):
    chk = CoverageFake()
    eng = QCEngine([chk])
    proj.timeline.get_track("track_v1").clips[1].duration = 5.0
    first = eng.run(qc_ctx(proj))
    fixed = next(i for i in first.run.issues if i.scene_id == proj.s2.id)
    fixed.status = IssueStatus.FIXED  # the user applied a fix to scene 2
    carried = eng.run(qc_ctx(proj, scene_filter=[proj.s1.id]), previous=PreviousState(first.run.issues, first.cache))
    assert [i.status for i in carried.run.issues] == [IssueStatus.FIXED] and carried.run.scores.counts["CRITICAL"] == 0  # not re-detected: still fixed
    cached = eng.run(qc_ctx(proj), previous=PreviousState(carried.run.issues, carried.cache))
    assert cached.run.checkers["scene"].state is CheckerState.CACHED and cached.run.issues[0].status is IssueStatus.FIXED
    redetected = eng.run(qc_ctx(proj), previous=PreviousState(cached.run.issues, cached.cache), use_cache=False)
    assert redetected.run.issues[0].status is IssueStatus.OPEN  # analysed again and still wrong: the fix did not resolve it


def test_a_category_run_or_a_cancel_is_complete_only_while_the_rest_still_matches_the_project(proj):
    t, c = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)]), Fake("caption", cats=(QCCategory.CAPTION,))
    eng = QCEngine([t, c])
    full = eng.run(qc_ctx(proj))
    same = eng.run(qc_ctx(proj), previous=PreviousState(full.run.issues, full.cache), selected={"caption"}, use_cache=False)
    assert same.complete and "timeline" in same.cache  # the timeline result is still valid for this project: kept, with its cache entry
    add_clip(proj, "track_v2", proj.a, 3, 2)
    moved = eng.run(qc_ctx(proj), previous=PreviousState(same.run.issues, same.cache), selected={"caption"}, use_cache=False)
    assert not moved.complete  # the timeline finding comes from before the edit: this run cannot vouch for the project
    assert any(i.code == "t.1" for i in moved.run.issues)


def test_a_canceled_run_reports_the_canceled_checks_as_not_completed(proj):
    class Canceler(Fake):
        def run(self, ctx, report):
            ctx.cancel.set()
            ctx.check_cancel()

    a = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)])
    first = QCEngine([a, Fake("sync", cats=(QCCategory.SYNC,))]).run(qc_ctx(proj))
    res = QCEngine([a, Canceler("sync", cats=(QCCategory.SYNC,))]).run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache), use_cache=False)
    assert res.run.state == "CANCELED" and res.run.failed_checkers() == ["sync"] and "sync" in res.run.scores.unavailable and res.run.record()["failed"] == ["sync"]
    assert "sync" in res.cache and res.complete  # the sync result of the earlier run is still valid for the unchanged project and stays cached
    add_clip(proj, "track_v2", proj.a, 3, 2)
    stale = QCEngine([a, Canceler("sync", cats=(QCCategory.SYNC,))]).run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache), use_cache=False)
    assert not stale.complete


def test_a_checker_switched_off_in_the_settings_loses_its_findings(proj):
    t, c = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)]), Fake("caption", [issue(Severity.ERROR, "c.1", QCCategory.CAPTION, None, start=None, end=None)], cats=(QCCategory.CAPTION,))
    eng = QCEngine([t, c])
    first = eng.run(qc_ctx(proj))
    assert {i.code for i in first.run.issues} == {"t.1", "c.1"}
    proj.qc_settings.enabled_checkers = [x for x in proj.qc_settings.enabled_checkers if x != "caption"]
    second = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
    assert {i.code for i in second.run.issues} == {"t.1"} and second.run.checkers["caption"].state is CheckerState.SKIPPED and "caption" not in second.cache and second.complete


def test_a_global_fact_every_checker_reads_invalidates_every_cache_entry(proj):
    """Regression: the voice-over length (ctx.duration), the locks, the export resolution and the confidence filters were in no cache key, so a changed value was served from the cache."""
    chk = Fake("timeline", [issue(Severity.WARNING, "t.1", QCCategory.TIMELINE)])
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(proj))
    for change in (lambda p: setattr(p.voice_over, "duration", 25.0), lambda p: p.timeline_generation.locked_scenes.append(p.s1.id),
                   lambda p: setattr(p.render_settings, "resolution", "2160p"), lambda p: setattr(p.qc_settings, "min_confidence_to_report", 80.0),
                   lambda p: setattr(p.qc_settings, "ai_confidence_caps", [[90.0, "INFO"]])):
        runs = chk.runs
        before = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
        assert before.run.checkers["timeline"].state is CheckerState.CACHED and chk.runs == runs
        change(proj)
        after = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
        assert after.run.checkers["timeline"].state is CheckerState.DONE and chk.runs == runs + 1
        first = after


def test_a_critical_issue_is_never_waved_through_by_an_ignore_record(proj):
    crit, warn = issue(Severity.CRITICAL, "same.code", QCCategory.TIMELINE), issue(Severity.WARNING, "same.code", QCCategory.TIMELINE, "scene_002")
    res = QCEngine([Fake("timeline", [crit, warn])]).run(qc_ctx(proj), ignores=[IgnoreRecord("t", "type", code="same.code", reason="intentional")])
    by_sev = {i.severity: i for i in res.run.issues}
    assert by_sev[Severity.WARNING].ignored_by_user and not by_sev[Severity.CRITICAL].ignored_by_user and res.run.scores.status == "BLOCKED" and res.run.scores.export == "BLOCKED"
    proj.qc_issues = res.run.issues
    IgnoreIssuesCommand(proj, IgnoreRecord("ig", "issue", crit.fingerprint)).do()
    assert not next(i for i in proj.qc_issues if i.severity is Severity.CRITICAL).ignored_by_user


def test_ready_never_sits_next_to_a_blocked_export_and_all_zero_weights_do_not_score_zero():
    s = QCSettings()
    s.block_level = BlockLevel.CRITICAL_ERROR_WARNING.value
    sc = compute_scores([issue(Severity.WARNING, "w.1", QCCategory.PACING)], s, 120, 12)
    assert sc.export == "BLOCKED" and sc.status == "REVIEW" and sc.overall >= 90  # the score is fine, the user's block level is not satisfied
    z = QCSettings()
    z.group_weights = {k: 0.0 for k in z.group_weights}
    assert compute_scores([], z, 60, 12).status_label == "100/100 — Ready"


def test_the_history_archive_and_noisy_runs_are_bounded(proj):
    from app.project.phase8_commands import MAX_HISTORY_ISSUES, MAX_ISSUES

    noisy = [issue(Severity.NOTICE, f"n.{k}", QCCategory.TIMELINE, None, start=float(k), end=float(k) + 1) for k in range(MAX_ISSUES + 50)]
    for k, i in enumerate(noisy):
        i.checker, i.fingerprint = "timeline", f"fp{k}"
    res = QCEngine([Fake("timeline", noisy)]).run(qc_ctx(proj))
    StoreQCRunCommand(proj, res.run.issues, res.run.scores, res.run.record(), res.run.history_entry([], []), res.cache).do()
    assert len(proj.qc_issues) == MAX_ISSUES and len(proj.qc_history[-1]["issues"]) == MAX_HISTORY_ISSUES and proj.qc_history[-1]["issues_total"] == MAX_ISSUES + 50
    assert "timeline" not in proj.qc_cache  # findings were cut: the cache must not vouch for them
    json.dumps(proj.to_document(), allow_nan=False)  # still a valid, savable document


def test_values_json_cannot_write_never_make_the_project_unsavable():
    import numpy as np
    from pathlib import Path

    from app.qc.issue_model import clean_json

    odd = {"n": np.int64(3), "f": np.float32(1.5), "s": {2, 1}, "p": Path("a/b"), "nan": float("nan"), "t": (1, 2), "sev": Severity.ERROR}
    assert json.loads(json.dumps(clean_json(odd), allow_nan=False)) == {"n": 3, "f": 1.5, "s": [1, 2], "p": str(Path("a/b")), "nan": None, "t": [1, 2], "sev": "ERROR"}
    i = issue()
    i.metrics = {"count": np.int64(7)}
    assert json.loads(json.dumps(i.to_dict(), allow_nan=False))["metrics"] == {"count": 7}


class CaptionSceneFake(SceneFake):
    """Scene-local, reads the project-wide caption settings and the frame size (neither belongs to a single scene)."""

    domains = ("timeline", "scenes", "captions")


def test_a_project_wide_value_re_analyses_every_scene_not_just_the_changed_ones(proj):
    """Regression: with a changed caption margin or frame size every scene's own key was unchanged, so the scene-level reuse served the old findings of all scenes."""
    chk = CaptionSceneFake()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(proj))
    assert chk.seen == [[proj.s1.id, proj.s2.id]]
    for change in (lambda p: setattr(p.caption_settings, "safe_margin_left", 0.3), lambda p: setattr(p.settings, "width", 1080), lambda p: setattr(p.settings, "fps", 24)):
        change(proj)
        chk.seen.clear()
        again = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
        assert chk.seen == [[proj.s1.id, proj.s2.id]] and again.complete
        first = again
    chk.seen.clear()
    quiet = eng.run(qc_ctx(proj), previous=PreviousState(first.run.issues, first.cache))
    assert chk.seen == [] and quiet.run.checkers["scene"].state is CheckerState.CACHED
