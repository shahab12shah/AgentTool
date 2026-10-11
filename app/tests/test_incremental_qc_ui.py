"""The QC page names the scenes that are stale after an edit (from the change log) and keeps the existing banner."""

from __future__ import annotations

import pytest

from app.qc.tests.test_qc_ui import qwin, run_and_wait, win  # noqa: F401  (fixtures)
from app.tests.conftest import needs_ffmpeg

pytest.importorskip("PySide6")
from app.tests.test_ui_acceptance import pump, qapp  # noqa: E402,F401  (fixtures)

pytestmark = needs_ffmpeg


def test_the_stale_banner_lists_the_scenes_to_re_check(qwin):
    win = qwin
    win.go_to("Quality")
    run_and_wait(win)
    panel = win.qc_panel
    assert "Scenes to re-check" not in panel.stale_label.text()
    win.ws.timeline.move_clip(win.c2.id, 16.0)
    pump(lambda: "changed since this QC run" in panel.stale_label.text())
    st = win.ws.qc.qc_staleness()
    assert st["stale"] and st["source"] == "tracker" and win.s2.id in st["scene_ids"]
    real = win.ws.qc.qc_staleness
    win.ws.qc.qc_staleness = lambda exact=False: {**real(exact), "scene_ids": [win.s2.id]}  # the project has two scenes: say that only the second is stale
    panel.refresh()
    assert f"Scenes to re-check: {win.s2.label}" in panel.stale_label.text() and "changed since this QC run" in panel.stale_label.text()
    win.ws.qc.qc_staleness = real
    run_and_wait(win)
    assert panel.stale_label.text() == ""
