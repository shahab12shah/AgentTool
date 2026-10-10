"""QCService through a real Workspace (no ffmpeg needed): background run, results installed in the project, cache, scene run, cancel with partial results, retry of a failed
check, ignores (undoable, persistent), history comparison, the export gate, persistence and "QC never changes the timeline"."""

from __future__ import annotations

import json
import threading
import time

import pytest

from app.core.events import Topics
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.errors import QCError
from app.project.phase8_commands import MarkIssueFixedCommand
from app.qc.issue_model import CheckerState, FixRecord, IssueStatus, QCCategory, QCIssue
from app.qc.qc_engine import QCEngine
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, narrate
from app.services.render_service import QCGateBlocked


class Fake(BaseChecker):
    domains = ("timeline", "scenes")

    def __init__(self, cid, issues=(), cats=(QCCategory.TIMELINE,), fail=None, block=None, scene_local=False):
        self.id, self.label, self.categories, self.scene_local = cid, cid.title(), cats, scene_local
        self._issues, self.fail, self.block, self.runs = list(issues), fail, block, 0
        self.fail_now = fail is not None

    def run(self, ctx, report):
        self.runs += 1
        if self.block is not None:
            self.block.set()
            for _ in range(500):
                ctx.check_cancel()
                time.sleep(0.01)
        if self.fail_now:
            raise RuntimeError(self.fail)
        out = []
        for i in self._issues:
            if self.scene_local and i.scene_id and ctx.scene_filter is not None and i.scene_id not in ctx.scene_filter:
                continue
            out.append(QCIssue.from_dict(i.to_dict()))
        return CheckerOutput(out, {"n": len(out)})


def mk(sev, code, cat=QCCategory.TIMELINE, scene=None, start=1.0, end=2.0, **kw) -> QCIssue:
    i = QCIssue(f"qci_{code}_{scene}", code, cat, sev, "Title " + code, "desc", scene, kw.pop("item", None), None, start, end, kw.pop("conf", 100.0), "deterministic:fake", **kw)
    i.fingerprint = i.make_fingerprint()
    return i


@pytest.fixture
def qws(project_ws):
    ws = project_ws
    p = ws.project
    p.voice_over.duration = 30.0
    a = add_asset(p, "a.mp4", "video", duration=30)
    s1 = add_scene(p, 0, 15, "Silver rose sharply this week.", importance=0.8)
    s2 = add_scene(p, 15, 30, "Demand keeps growing steadily.")
    narrate(p)
    add_clip(p, "track_v1", a, 0, 15, scene=s1)
    add_clip(p, "track_v1", a, 15, 15, scene=s2)
    ws.s1, ws.s2, ws.a = s1, s2, a
    return ws


def run_qc(ws, **kw):
    job = ws.qc.run_full_qc(**kw)
    assert ws.jobs.wait_idle(30)
    return job


def timeline_json(ws) -> str:
    return json.dumps(ws.project.timeline.to_dict(), sort_keys=True)


def test_a_full_run_installs_findings_scores_and_history_without_touching_the_timeline(qws):
    ws = qws
    before = timeline_json(ws)
    events: list[str] = []
    ws.bus.subscribe(Topics.QC_UPDATED, lambda t, p: events.append(p["kind"]))
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "t.w", scene=ws.s1.id), mk(Severity.ERROR, "t.e", QCCategory.SYNC, scene=ws.s2.id)]),
                             Fake("caption", cats=(QCCategory.CAPTION,))])
    job = run_qc(ws)
    p = ws.project
    assert job.status.value == "COMPLETED" and len(p.qc_runs) == 1 and len(p.qc_history) == 1 and len(p.qc_issues) == 2
    assert p.qc_runs[0]["number"] == 1 and p.qc_runs[0]["state"] == "COMPLETED" and p.qc_runs[0]["checkers"]["timeline"]["issues"] == 2
    assert p.qc_scores.counts["ERROR"] == 1 and p.qc_scores.counts["WARNING"] == 1 and p.qc_scores.export == "BLOCKED" and p.qc_runs[0]["content_hash"]
    assert {"run_started", "progress", "run_finished"} <= set(events)
    assert timeline_json(ws) == before  # QC analysed; it did not edit
    res = ws.qc.get_results()
    assert [i.title for i in res["issues"]][0] == "Title t.e" and res["scores"].overall == p.qc_scores.overall  # most severe first
    assert ws.qc.issues(severity="WARNING")[0].code == "t.w" and ws.qc.issues(scene_id=ws.s2.id)[0].code == "t.e"
    assert [m["severity"] for m in ws.qc.markers()] == ["ERROR", "WARNING"] or len(ws.qc.markers()) == 2
    assert ws.qc.markers("hidden") == [] and all(m["severity"] == "ERROR" for m in ws.qc.markers("critical_only")) or ws.qc.markers("critical_only") == []


def test_second_run_reuses_unchanged_analysis_and_a_changed_timeline_reruns_it(qws):
    ws = qws
    chk = Fake("timeline", [mk(Severity.WARNING, "t.w", scene=ws.s1.id)])
    ws.qc.engine = QCEngine([chk])
    run_qc(ws)
    run_qc(ws)
    assert chk.runs == 1 and ws.project.qc_runs[-1]["checkers"]["timeline"]["state"] == "CACHED" and ws.project.qc_runs[-1]["cache_hits"] == 1 and ws.project.qc_runs[-1]["number"] == 2
    add_clip(ws.project, "track_v2", ws.a, 3, 2)
    run_qc(ws)
    assert chk.runs == 2
    run_qc(ws, force=True)
    assert chk.runs == 3 and len(ws.project.qc_runs) == 4


def test_scene_run_only_touches_scene_local_checkers(qws):
    ws = qws
    sc = Fake("scene", [mk(Severity.WARNING, "s.1", QCCategory.SCENE_COVERAGE, scene=ws.s1.id), mk(Severity.WARNING, "s.2", QCCategory.SCENE_COVERAGE, scene=ws.s2.id)],
              cats=(QCCategory.SCENE_COVERAGE,), scene_local=True)
    glob = Fake("timeline", [mk(Severity.NOTICE, "t.n")])
    ws.qc.engine = QCEngine([glob, sc])
    run_qc(ws)
    g_runs = glob.runs
    job = ws.qc.run_scene_qc(ws.s1.id)
    assert ws.jobs.wait_idle(30) and job.status.value == "COMPLETED"
    assert glob.runs == g_runs and len(ws.project.qc_issues) == 3  # the global finding is kept, not re-run
    assert ws.project.qc_runs[-1]["trigger"] == "scene" and ws.project.qc_runs[-1]["scope"]["scene_ids"] == [ws.s1.id]
    with pytest.raises(QCError, match="does not exist"):
        ws.qc.run_scene_qc("scene_999")


def test_cancel_keeps_finished_results_and_stores_a_partial_run(qws):
    ws = qws
    started = threading.Event()
    done = Fake("timeline", [mk(Severity.WARNING, "t.w")])
    blocker = Fake("sync", cats=(QCCategory.SYNC,), block=started)
    ws.qc.engine = QCEngine([done, blocker])
    job = ws.qc.run_full_qc()
    assert started.wait(10) and ws.qc.cancel()
    assert ws.jobs.wait_idle(30)
    p = ws.project
    assert job.status.value == "CANCELLED" and p.qc_runs[-1]["state"] == "CANCELED" and p.qc_runs[-1]["checkers"]["timeline"]["state"] == "DONE" and p.qc_runs[-1]["checkers"]["sync"]["state"] == "CANCELED"
    assert [i.code for i in p.qc_issues] == ["t.w"] and ws.qc.progress.state == "IDLE" and not ws.qc.running


def test_one_failing_check_does_not_fail_the_run_and_retry_updates_the_same_run(qws):
    ws = qws
    bad = Fake("audio", cats=(QCCategory.AUDIO,), fail="decoder exploded")
    ok = Fake("timeline", [mk(Severity.WARNING, "t.w")])
    ws.qc.engine = QCEngine([ok, bad])
    run_qc(ws)
    rec = ws.project.qc_runs[-1]
    assert rec["state"] == "PARTIAL" and rec["failed"] == ["audio"] and rec["checkers"]["audio"]["state"] == "FAILED" and "decoder exploded" in rec["checkers"]["audio"]["error"]
    assert "audio" in ws.project.qc_scores.unavailable and ws.project.qc_scores.status != "READY"
    run_id, ok_runs = rec["run_id"], ok.runs
    bad.fail_now = False
    bad._issues = [mk(Severity.ERROR, "a.clip", QCCategory.AUDIO)]
    job = ws.qc.retry_failed_check()
    assert ws.jobs.wait_idle(30) and job.status.value == "COMPLETED"
    rec2 = ws.project.qc_runs[-1]
    assert rec2["run_id"] == run_id and len(ws.project.qc_runs) == 1 and rec2["state"] == "COMPLETED" and rec2["checkers"]["audio"]["state"] == "DONE"
    assert ok.runs == ok_runs  # the healthy checker was not restarted
    assert {i.code for i in ws.project.qc_issues} == {"t.w", "a.clip"} and "audio" not in ws.project.qc_scores.unavailable
    with pytest.raises(QCError, match="Nothing failed"):
        ws.qc.retry_failed_check()


def test_ignore_is_undoable_persistent_and_survives_a_rerun_until_the_content_changes(qws):
    ws = qws
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "visual.repetition", QCCategory.VISUAL_REPETITION, scene=ws.s1.id),
                                               mk(Severity.WARNING, "visual.repetition", QCCategory.VISUAL_REPETITION, scene=ws.s2.id)])])
    run_qc(ws)
    first = ws.qc.issues()[0]
    base = ws.project.qc_scores.overall
    rec = ws.qc.ignore_issue(first.issue_id, "Intentional recurring visual")
    assert ws.project.qc_ignored_issues[0].reason == "Intentional recurring visual" and len(ws.qc.issues()) == 1 and ws.project.qc_scores.counts["WARNING"] == 1 and ws.project.qc_scores.overall >= base
    run_qc(ws, force=True)  # a new analysis does not flag it again
    again = [i for i in ws.project.qc_issues if i.fingerprint == first.fingerprint][0]
    assert again.ignored_by_user and again.status is IssueStatus.IGNORED and again.ignore_reason == "Intentional recurring visual" and ws.project.qc_scores.counts["WARNING"] == 1
    ws.undo()  # Ctrl+Z of the ignore: it is an ordinary undoable edit...
    assert not ws.project.qc_ignored_issues and ws.project.qc_scores.counts["WARNING"] == 2  # ...and the scores follow at once
    ws.redo()
    assert ws.project.qc_ignored_issues and ws.project.qc_scores.counts["WARNING"] == 1
    ws.qc.unignore(rec.ignore_id)
    assert ws.project.qc_scores.counts["WARNING"] == 2
    # ignore the whole type: every issue of that kind stops counting
    ws.qc.ignore_type(ws.qc.issues()[0].issue_id, "Recurring on purpose")
    assert ws.project.qc_scores.counts["WARNING"] == 0 and all(i.ignored_by_user for i in ws.project.qc_issues)


def test_history_comparison_and_report(qws):
    ws = qws
    chk = Fake("timeline", [mk(Severity.ERROR, "t.e"), mk(Severity.WARNING, "t.w")])
    ws.qc.engine = QCEngine([chk])
    run_qc(ws)
    chk._issues = [mk(Severity.WARNING, "t.w")]
    run_qc(ws, force=True)
    c = ws.qc.compare_runs(1, 2)
    assert (c.older, c.newer) == (1, 2) and c.improved and [i["code"] for i in c.resolved_issues] == ["t.e"] and c.counts["ERROR"] == (1, 0)
    assert ws.qc.compare_runs(2, 1).older == 1  # order does not matter
    with pytest.raises(QCError):
        ws.qc.compare_runs(1, 9)
    text = ws.qc.report()
    assert "## Overall score" in text and "Title t.w" in text
    path = ws.qc.save_report()
    assert path.is_file() and path.parent.name == "qc" and path.read_text(encoding="utf-8") == text
    old = ws.qc.get_results(ws.project.qc_runs[0]["run_id"])
    assert old["archived"] and {i["code"] for i in old["issues"]} == {"t.e", "t.w"}


def test_export_gate_blocks_only_while_the_blocking_run_is_current(qws):
    ws = qws
    assert ws.qc.export_gate().needs_run and not ws.qc.export_gate().blocked  # never run: nothing blocks (the UI runs QC first)
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.ERROR, "t.e"), mk(Severity.WARNING, "t.w")])])
    run_qc(ws)
    g = ws.qc.export_gate()
    assert g.run_current and g.blocked and "EXPORT BLOCKED" in g.message
    with pytest.raises(QCGateBlocked) as exc:
        ws.render.start_export()
    assert exc.value.kind == "qc_blocked" and not exc.value.overridable and exc.value.issue_ids
    add_clip(ws.project, "track_v2", ws.a, 3, 2)  # the project changed since: the old result no longer blocks, QC must run again
    g2 = ws.qc.export_gate()
    assert g2.needs_run and not g2.blocked and "changed since the last QC run" in g2.message
    run_qc(ws)
    ws.qc.update_settings(block_level="CRITICAL")  # Errors no longer block
    g3 = ws.qc.export_gate()
    assert not g3.blocked and g3.decision.status == "AVAILABLE" and "warnings remain" in g3.decision.message
    ws.qc.update_settings(block_level="CRITICAL_ERROR_WARNING")
    assert ws.qc.export_gate().blocked


def test_override_is_allowed_only_where_settings_permit_and_never_for_critical(qws):
    ws = qws
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.ERROR, "t.e")])])
    run_qc(ws)
    with pytest.raises(QCGateBlocked):
        ws.render.start_export(qc_override=True)  # the setting does not allow overriding
    ws.qc.update_settings(allow_export_override=True)
    gate = ws.qc.export_gate()
    assert gate.blocked and gate.decision.overridable
    ws.render._check_qc_gate(True)  # no exception: the explicit override is honoured
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.CRITICAL, "t.c", QCCategory.PREFLIGHT)], cats=(QCCategory.PREFLIGHT,))])
    run_qc(ws, force=True)
    with pytest.raises(QCGateBlocked) as exc:
        ws.render._check_qc_gate(True)
    assert not exc.value.overridable  # a CRITICAL cannot be overridden


def test_settings_are_validated_undoable_and_saved_with_the_project(qws, tmp_path):
    ws = qws
    ws.qc.update_settings(block_level="CRITICAL", **{"sync.major_ms": 900.0, "caption.max_cps": 18.0})
    s = ws.project.qc_settings
    assert s.block_level == "CRITICAL" and s.sync.major_ms == 900.0 and s.caption.max_cps == 18.0
    for bad in ({"block_level": "NEVER"}, {"marker_mode": "sometimes"}, {"sync.minor_ms": 9000.0}, {"nonsense": 1}, {"fix_permissions": {"caption.retime": "maybe"}}):
        with pytest.raises(QCError):
            ws.qc.update_settings(**bad)
    assert ws.project.qc_settings.block_level == "CRITICAL"
    ws.undo()
    assert ws.project.qc_settings.caption.max_cps == 21.0 or ws.project.qc_settings.sync.major_ms != 900.0
    ws.save()
    root = ws.project.root
    ws.close_project()
    ws.open_project(root)
    assert ws.project.schema_version == 8 and ws.project.qc_settings.block_level in ("CRITICAL", "CRITICAL_ERROR")


def test_qc_history_issues_and_scores_survive_save_and_reopen(qws):
    ws = qws
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "t.w", scene=ws.s1.id)])])
    run_qc(ws)
    ws.qc.ignore_issue(ws.qc.issues()[0].issue_id, "keep it")
    snap = (ws.project.qc_scores.overall, len(ws.project.qc_runs), len(ws.project.qc_history), ws.project.qc_runs[-1]["run_id"])
    ws.save()
    root = ws.project.root
    ws.close_project()
    ws.open_project(root)
    p = ws.project
    assert (p.qc_scores.overall, len(p.qc_runs), len(p.qc_history), p.qc_runs[-1]["run_id"]) == snap
    assert p.qc_ignored_issues[0].reason == "keep it" and p.qc_issues[0].ignored_by_user
    assert ws.qc.export_gate().run_current  # nothing changed: the stored run still matches the project
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "t.w", scene=ws.s1.id)])])
    run_qc(ws)
    assert p.qc_runs[-1]["number"] == 2 and p.qc_issues[0].ignored_by_user  # the ignore still applies after reopening


def test_a_second_run_cannot_start_while_one_is_running(qws):
    ws = qws
    started = threading.Event()
    ws.qc.engine = QCEngine([Fake("timeline", block=started)])
    ws.qc.run_full_qc()
    assert started.wait(10) and ws.qc.running
    with pytest.raises(QCError, match="already running"):
        ws.qc.run_full_qc()
    ws.qc.cancel()
    assert ws.jobs.wait_idle(30)


def test_category_run_and_unknown_category(qws):
    ws = qws
    cap = Fake("caption", [mk(Severity.NOTICE, "c.n", QCCategory.CAPTION)], cats=(QCCategory.CAPTION,))
    ws.qc.engine = QCEngine([Fake("timeline"), cap])
    run_qc(ws)
    ws.qc.run_category_qc("CAPTION")
    assert ws.jobs.wait_idle(30) and cap.runs == 2 and ws.project.qc_runs[-1]["trigger"] == "category"
    with pytest.raises(QCError, match="no QC check"):
        ws.qc.run_category_qc("NOT_A_CATEGORY")


def test_no_project_gives_a_clear_error(ws):
    from app.core.exceptions import ProjectError

    with pytest.raises(ProjectError):
        ws.qc.run_full_qc()
    assert not ws.qc.running


_ = CheckerState


# ------------------------------------------------------------------ review round: partial runs, ignores, settings, older documents
class CoverageChk(BaseChecker):
    """Scene-local: a CRITICAL for every scene whose V1 pictures are shorter than the scene (it reads the project, so a stale cache entry shows)."""

    domains = ("timeline", "scenes")
    scene_local = True

    def __init__(self, cid="scene", gate: threading.Event | None = None, release: threading.Event | None = None, severity=Severity.CRITICAL):
        self.id, self.label, self.categories, self.runs, self.gate, self.release, self.severity = cid, cid, (QCCategory.SCENE_COVERAGE,), 0, gate, release, severity

    def run(self, ctx, report):
        self.runs += 1
        if self.gate is not None:
            self.gate.set()
            assert self.release.wait(10)
            ctx.check_cancel()
        out = []
        for s in ctx.target_scenes():
            if sum(c.duration for _t, c in ctx.visual_clips() if c.scene_id == s.id) < (s.end - s.start) - 0.01:
                out.append(self.issue("scene.uncovered", QCCategory.SCENE_COVERAGE, self.severity, "Uncovered", scene_id=s.id, start=s.start, end=s.end, ctx=ctx))
        return CheckerOutput(out, {})


def test_a_scene_run_never_hides_a_problem_in_another_scene_from_the_gate_or_the_next_full_run(qws):
    ws = qws
    ws.qc.engine = QCEngine([CoverageChk()])
    run_qc(ws)
    assert ws.qc.export_gate().run_current and not ws.qc.export_gate().blocked
    ws.project.timeline.get_track("track_v1").clips[1].duration = 5.0  # scene 2 breaks; QC is asked about scene 1 only
    ws.qc.run_scene_qc(ws.s1.id)
    assert ws.jobs.wait_idle(30)
    g = ws.qc.export_gate()
    assert not g.run_current and g.needs_run and ws.project.qc_runs[-1]["content_hash"] == "" and "did not cover the whole project" in g.message  # the scene run is not a verdict on the project
    run_qc(ws)
    assert [(i.code, i.scene_id) for i in ws.project.qc_issues] == [("scene.uncovered", ws.s2.id)]  # the full run really looks at scene 2
    g2 = ws.qc.export_gate()
    assert g2.run_current and g2.blocked


def test_a_scene_run_after_a_fix_keeps_the_other_scenes_fixed_marks(qws):
    ws = qws
    ws.project.timeline.get_track("track_v1").clips[1].duration = 5.0
    ws.qc.engine = QCEngine([CoverageChk()])
    run_qc(ws)
    issue = ws.project.qc_issues[0]
    ws.commands.execute(MarkIssueFixedCommand(ws.project, issue.issue_id, FixRecord("f1", issue.issue_id, issue.code, "gap.close")))
    ws.qc._recount(ws.project)  # what QCService.apply_fix does after a fix
    assert issue.status is IssueStatus.FIXED and ws.project.qc_scores.counts["CRITICAL"] == 0
    ws.qc.run_scene_qc(ws.s1.id)
    assert ws.jobs.wait_idle(30)
    assert [i.status for i in ws.project.qc_issues] == [IssueStatus.FIXED] and ws.project.qc_scores.counts["CRITICAL"] == 0  # carried over, not re-opened


def test_cancel_is_not_a_verdict_and_the_gate_says_the_check_did_not_complete(qws):
    ws = qws
    started, release = threading.Event(), threading.Event()
    sync = CoverageChk("sync", started, release)
    sync.gate = None
    ws.qc.engine = QCEngine([Fake("timeline"), sync])
    run_qc(ws)
    assert ws.qc.export_gate().run_current
    sync.gate, sync.release = started, release
    ws.project.timeline.get_track("track_v1").clips[1].duration = 5.0
    job = ws.qc.run_full_qc()
    assert started.wait(10) and ws.qc.cancel()
    release.set()
    assert ws.jobs.wait_idle(30) and job.status.value == "CANCELLED"
    rec = ws.project.qc_runs[-1]
    assert rec["state"] == "CANCELED" and rec["failed"] == ["sync"] and not ws.qc.export_gate().run_current  # the canceled check would have found the broken scene
    assert "sync" in ws.project.qc_scores.unavailable


def test_a_critical_issue_cannot_be_ignored_but_a_milder_one_of_the_same_kind_can(qws):
    ws = qws
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.CRITICAL, "t.same", scene=ws.s1.id), mk(Severity.WARNING, "t.same", scene=ws.s2.id)])])
    run_qc(ws)
    crit = next(i for i in ws.project.qc_issues if i.severity is Severity.CRITICAL)
    warn = next(i for i in ws.project.qc_issues if i.severity is Severity.WARNING)
    with pytest.raises(QCError, match="critical"):
        ws.qc.ignore_issue(crit.issue_id, "I know")
    ws.qc.ignore_type(warn.issue_id, "the same kind, on purpose")
    assert warn.ignored_by_user and not crit.ignored_by_user and ws.project.qc_scores.counts["CRITICAL"] == 1 and ws.qc.export_gate().blocked
    run_qc(ws, force=True)
    assert [i.ignored_by_user for i in sorted(ws.project.qc_issues, key=lambda i: i.severity.value)] == [False, True] and ws.qc.export_gate().blocked


def test_an_ignore_made_while_the_run_is_in_flight_survives_the_install(qws):
    ws = qws
    started, release = threading.Event(), threading.Event()
    chk = CoverageChk("scene", severity=Severity.WARNING)
    ws.project.timeline.get_track("track_v1").clips[1].duration = 5.0
    ws.qc.engine = QCEngine([chk])
    run_qc(ws)
    issue = ws.project.qc_issues[0]
    chk.gate, chk.release = started, release
    ws.qc.run_full_qc(force=True)
    assert started.wait(10)
    ws.qc.ignore_issue(issue.issue_id, "decided while QC was running")  # the user acts on the old findings while the job works on its snapshot
    release.set()
    assert ws.jobs.wait_idle(30)
    [now] = ws.project.qc_issues
    assert now.ignored_by_user and now.status is IssueStatus.IGNORED and ws.project.qc_scores.counts["WARNING"] == 0


def test_only_the_latest_run_has_a_full_report_and_settings_are_validated(qws):
    ws = qws
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "t.w")])])
    run_qc(ws)
    first = ws.project.qc_runs[-1]["run_id"]
    run_qc(ws, force=True)
    assert "Run #2" in ws.qc.report() and ws.qc.report(ws.project.qc_runs[-1]["run_id"]) == ws.qc.report()
    with pytest.raises(QCError, match="latest"):
        ws.qc.report(first)
    for bad in ({"enabled_checkers": ["nonsense"]}, {"group_weights": {"audio": -1.0}}, {"group_weights": {"audio": float("nan")}}, {"ai_confidence_caps": [[50.0, "BAD"]]},
                {"min_confidence_to_report": 140.0}, {"caption.max_cps": float("inf")}, {"sync.minor_ms": float("nan")}):
        with pytest.raises(QCError):
            ws.qc.update_settings(**bad)


@pytest.mark.parametrize("version", [7, 6, 5, 4, 3, 2, 1])
def test_a_document_saved_before_phase_8_opens_with_empty_qc_sections_and_saves_as_schema_8(qws, version):
    from app.project import project_schema as ps
    from app.storage.paths import ProjectPaths

    ws = qws
    ws.save()
    root = ws.project.root
    f = ProjectPaths(root).project_file
    doc = json.loads(f.read_text(encoding="utf-8"))
    qc_keys = ("qc_settings", "qc_runs", "qc_issues", "qc_scores", "qc_ignored_issues", "qc_fixes", "qc_history", "qc_cache", "render_qc_results")
    drop = set(qc_keys)
    for limit, section in ((6, ps._V7_SECTIONS), (5, ps._V6_SECTIONS), (4, ps._V5_SECTIONS), (3, ps._V4_SECTIONS), (2, ps._V3_SECTIONS), (1, ps._V2_SECTIONS)):
        if version <= limit:
            drop |= set(section)
    if version <= 3:
        drop.add("timeline_version")
        doc["timeline"]["tracks"] = [t for t in doc["timeline"]["tracks"] if t["id"] != "track_v6"]
    if version <= 1:
        doc["scenes"] = []
    for k in drop:
        doc.pop(k, None)
    doc["schema_version"] = version
    f.write_text(json.dumps(doc), encoding="utf-8")
    ws.close_project()
    ws.open_project(root)
    p = ws.project
    assert p.schema_version == 8 and p.qc_issues == [] and p.qc_runs == [] and p.qc_scores is None and p.qc_history == [] and p.qc_cache == {} and p.qc_settings.block_level == "CRITICAL_ERROR"
    gate = ws.qc.export_gate()
    assert gate.needs_run and not gate.blocked  # never analysed: the export page runs QC first
    ws.save()
    saved = json.loads(f.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 8 and all(k in saved for k in qc_keys)
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "t.w")])])
    run_qc(ws)
    assert len(ws.project.qc_runs) == 1 and ws.project.qc_runs[0]["number"] == 1


_ = (FixRecord, MarkIssueFixedCommand)


def test_post_render_qc_measures_the_file_against_the_snapshot_that_was_rendered(qws, tmp_path, monkeypatch):
    """Regression: the expected size / length / frame rate came from the project as it is when the render FINISHES, so editing or changing the export resolution during a long
    render made a correct file look wrong."""
    from types import SimpleNamespace

    from app.project.project_schema import RenderSettings
    from app.qc import render_checker as rc

    seen: list[dict] = []

    def fake_inspect(self, ctx, path, *, render_id, expected, report):  # noqa: ARG001
        seen.append(dict(expected))
        return {"render_id": render_id, "status": "PASSED", "checks": [], "summary": ""}

    monkeypatch.setattr(rc.RenderedFileChecker, "inspect", fake_inspect)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    ws = qws
    ws.project.render_settings.resolution = "2160p"  # chosen AFTER the render started
    ws.project.voice_over.duration = 99.0
    snap = SimpleNamespace(settings=RenderSettings(resolution="720p", fps=0), canvas_w=1920, canvas_h=1080, fps=30, duration=12.5, voice_asset_id="asset_voice", tracks=[])
    ws.qc.run_post_render_qc("r1", out, snapshot=snap)
    assert ws.jobs.wait_idle(30)
    assert seen[-1] == {"duration": 12.5, "width": 1280, "height": 720, "fps": 30, "has_audio": True}
    assert ws.project.render_qc_results["r1"]["status"] == "PASSED"
    ws.qc.run_post_render_qc("r2", out)  # no snapshot (an older caller): the project as it is
    assert ws.jobs.wait_idle(30)
    assert seen[-1]["width"] == 3840 and seen[-1]["duration"] == 99.0


def test_two_ignores_of_the_same_issue_get_distinct_ids_even_within_one_second(qws):
    """Regression: the id was a hash of the issue id and the time to the second, so 'ignore this' followed by 'ignore this type' collided and Stop-ignoring removed both."""
    ws = qws
    ws.qc.engine = QCEngine([Fake("timeline", [mk(Severity.WARNING, "t.w", scene=ws.s1.id)])])
    run_qc(ws)
    iid = ws.qc.issues()[0].issue_id
    a = ws.qc.ignore_issue(iid, "this one")
    b = ws.qc.ignore_type(iid, "all of them")
    assert a.ignore_id != b.ignore_id and len(ws.project.qc_ignored_issues) == 2
    ws.qc.unignore(b.ignore_id)
    assert [r.ignore_id for r in ws.project.qc_ignored_issues] == [a.ignore_id] and ws.project.qc_issues[0].ignored_by_user  # the first ignore still holds


def test_canceling_the_rendered_file_check_cancels_the_job_quietly(qws, tmp_path, monkeypatch):
    """Regression: the check raised QCCancelled, which the job runner reported as FAILED ("Unexpected error") with a logged traceback."""
    from app.qc import render_checker as rc
    from app.qc.context import QCCancelled

    def cancelled(self, ctx, path, **kw):  # noqa: ARG001
        raise QCCancelled()

    monkeypatch.setattr(rc.RenderedFileChecker, "inspect", cancelled)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    job = qws.qc.run_post_render_qc("r9", out)
    assert qws.jobs.wait_idle(30)
    assert job.status.value == "CANCELLED" and job.error is None and "r9" not in qws.project.render_qc_results
