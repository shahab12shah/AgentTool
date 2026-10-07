"""Phase 3 acceptance workflow through the real main window: research -> candidates -> scores -> choose -> asset -> save -> reopen."""

from __future__ import annotations

import pytest

from app.tests.conftest import needs_ffmpeg
from app.tests.helpers import NARRATION, FakeProvider, ScriptedProvider, make_audio, make_image, make_video

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QMessageBox, QPushButton  # noqa: E402

from app.main import create_window  # noqa: E402
from app.media.asset import AssetType, SourceType as S  # noqa: E402
from app.research.models import Acquisition, ResearchStatus  # noqa: E402
from app.research.providers.base import ProviderRegistry  # noqa: E402
from app.tests.test_research_service import JUNK, MIDDLE, STRONG  # noqa: E402
from app.tests.test_ui_acceptance import create_project, pump, qapp, win  # noqa: E402,F401  (fixtures)
from app.ui.review_panel import CandidateCard  # noqa: E402

pytestmark = needs_ffmpeg


def install(window, provider):
    reg = ProviderRegistry()
    reg.register(provider)
    window.ws.research.registry = window.ws.research.engine.registry = reg


def make_provider(tmp_path):
    d = tmp_path / "files"
    d.mkdir(exist_ok=True)
    video = make_video(d / "strong.mp4", 4.0)
    image = make_image(d / "good.png", "testsrc", "640x360")
    junk = make_image(d / "junk.png", "red", "320x240")
    items = [{**STRONG, "source_type": S.STOCK_VIDEO, "id": "v1", "local_path": str(video), "acquisition": Acquisition.LOCAL},
             {**MIDDLE, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "i1", "local_path": str(image), "acquisition": Acquisition.LOCAL},
             {**JUNK, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "i2", "local_path": str(junk), "acquisition": Acquisition.LOCAL}]
    return FakeProvider(items, [S.STOCK_VIDEO, S.STOCK_IMAGE], name="stock"), junk


def cards(panel):
    return panel.detail.findChildren(CandidateCard)


def button(widget, text):
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text and b.isVisible())


def test_phase3_acceptance_workflow(win, tmp_path, app_paths, qapp, monkeypatch):
    ws = win.ws
    provider, junk = make_provider(tmp_path)
    project = create_project(win, tmp_path, "Phase3")
    voice = ScriptedProvider(NARRATION)
    ws.transcripts.registry.register(voice)
    ws.transcripts.resolve_provider = lambda name=None: voice
    ws.set_script(NARRATION)
    win.voice_panel.import_path(make_audio(tmp_path / "vo.wav", voice.words[-1].end + 1.0))
    pump(lambda: project.voice_over.asset_id is not None and ws.jobs.wait_idle(0))
    ws.transcripts.transcribe(provider_name=voice.name)
    pump(lambda: project.transcription.transcript is not None and ws.jobs.wait_idle(0))
    from app.analysis.segmenter import SegmentationParams

    ws.scenes.analyze(params=SegmentationParams(threshold=0.6, min_scene_seconds=2.0))
    pump(lambda: len(project.scenes) > 0 and ws.jobs.wait_idle(0))
    install(win, provider)
    panel = win.review_panel
    win.go_to("Review")
    assert panel.table.rowCount() == len(project.scenes) >= 14
    assert "Not started" in panel.table.item(0, 2).text()

    # ---- scene -> brief -> queries -> providers -> candidates -> scores -> best + alternatives ----
    panel.table.selectRow(0)
    sid = project.scenes[0].id
    assert panel.scene_id == sid
    QTest.mouseClick(panel.research_btn, Qt.MouseButton.LeftButton)
    pump(lambda: project.research_status.get(sid) and project.research_status[sid].status is ResearchStatus.CANDIDATES_READY)
    pump(lambda: len(cards(panel)) >= 2)
    st = project.research_status[sid]
    shown = cards(panel)
    assert shown[0].property("role") == "BEST" and len(shown) - 1 == len(st.alternatives) <= 5
    texts = " ".join(l.text() for l in shown[0].findChildren(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel))
    assert "Why this visual?" in texts and "/100" in texts
    assert panel.table.item(0, 3).text() == f"{project.candidate_scores[st.best_id].overall:.0f}"
    assert project.visual_assignments == {} and not [a for a in project.assets.all() if a.source_type is S.STOCK_VIDEO]  # research is not an asset

    # ---- the user picks an alternative (not the AI's best) ----
    alt_card = shown[1]
    QTest.mouseClick(button(alt_card, "Use"), Qt.MouseButton.LeftButton)
    pump(lambda: sid in project.visual_assignments and project.visual_assignments[sid].asset_id and ws.jobs.wait_idle(0))
    a = project.visual_assignments[sid]
    assert a.candidate_id == st.alternatives[0] and a.selected_by == "USER"
    asset = project.assets.get(a.asset_id)
    assert asset is not None and asset.type is AssetType.IMAGE and asset.source_type is S.STOCK_IMAGE
    pump(lambda: "Chosen" in " ".join(b.text() for b in panel.detail.findChildren(QPushButton)))
    QTest.mouseClick(button(panel.detail, "Approve"), Qt.MouseButton.LeftButton)
    pump(lambda: project.visual_assignments[sid].approved)
    assert "Approved" in panel.table.item(0, 2).text()
    assert project.timeline.all_clips() == []  # no timeline editing in this phase

    # ---- save, reopen, restored ----
    before = {k: project.to_document()[k] for k in ("research_status", "visual_candidates", "candidate_scores", "visual_assignments", "research_queries")}
    root = project.root
    assert win.save()
    win.close()
    ws.shutdown()
    win2, ws2 = create_window(app_paths)
    win2.show()
    install(win2, provider)
    win2.open_project(root)
    p2 = ws2.project
    for key, value in before.items():
        assert p2.to_document()[key] == value, key
    win2.go_to("Review")
    pump(lambda: win2.review_panel.table.rowCount() > 0)
    assert "Approved" in win2.review_panel.table.item(0, 2).text()
    win2.review_panel.table.selectRow(0)
    assert cards(win2.review_panel) and "Chosen" in " ".join(b.text() for b in win2.review_panel.detail.findChildren(QPushButton))
    assert p2.assets.get(p2.visual_assignments[sid].asset_id) is not None

    # ---- low-confidence flow ----
    provider.items = [{**JUNK, "source_type": S.STOCK_IMAGE, "kind": "IMAGE", "id": "j", "local_path": str(junk), "acquisition": Acquisition.LOCAL}]
    sid2 = p2.scenes[1].id
    win2.review_panel.table.selectRow(1)
    QTest.mouseClick(win2.review_panel.research_btn, Qt.MouseButton.LeftButton)
    pump(lambda: p2.research_status.get(sid2) and p2.research_status[sid2].status is ResearchStatus.LOW_CONFIDENCE)
    pump(lambda: any(b.text() == "Manual Select…" for b in win2.review_panel.detail.findChildren(QPushButton)))
    labels = {b.text() for b in win2.review_panel.detail.findChildren(QPushButton)}
    assert {"Search Again", "Expand Sources…", "Generate AI Visual", "Manual Select…", "Skip"} <= labels
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.Yes)[1])
    win2.review_panel.table.selectAll()
    QTest.mouseClick(win2.review_panel.approve_sel_btn, Qt.MouseButton.LeftButton)
    assert asked and "nothing weak is approved" in asked[0]
    assert sid2 not in p2.visual_assignments  # bulk approval never forces a weak visual in
    win2.review_panel.table.clearSelection()

    # ---- total provider failure shows the required message with recovery actions ----
    provider.fail = "service down"
    sid3 = p2.scenes[2].id
    win2.review_panel.table.selectRow(2)
    QTest.mouseClick(win2.review_panel.research_btn, Qt.MouseButton.LeftButton)
    pump(lambda: p2.research_status.get(sid3) and p2.research_status[sid3].status is ResearchStatus.ERROR)
    pump(lambda: any(b.text() == "Retry" for b in win2.review_panel.detail.findChildren(QPushButton)))
    assert p2.research_status[sid3].message.startswith("Visual research unavailable.")
    ws2.close_project()
    ws2.shutdown()
    win2.close()


def test_settings_dialog_exposes_research_sources_without_storing_keys(win, monkeypatch):
    from app.ui.dialogs.settings_dialog import SettingsDialog

    monkeypatch.setenv("PEXELS_API_KEY", "super-secret-value")
    dlg = SettingsDialog(win.ws.settings, win.ws.describe_ffmpeg, win.ws.transcripts.provider_report, win.ws.research.provider_report)
    dlg.pexels_env.setText("MY_PEXELS_VAR")
    dlg.stock_dir.setText("/some/stock")
    out = dlg.result_settings()
    assert out.pexels_key_env == "MY_PEXELS_VAR" and out.local_stock_dir == "/some/stock"
    assert "super-secret-value" not in repr(out) and "super-secret-value" not in dlg.research_status.text()
    assert "Screenshot" in dlg.research_status.text() or "screenshot" in dlg.research_status.text().lower()
    assert "not verified against the live service" in dlg.research_status.text()  # honest provider status
