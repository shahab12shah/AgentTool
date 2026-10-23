"""Phase 10 UI: the type-aware inspector, the keyframe editor and the AI decision actions (offscreen Qt, real media, real engine)."""

from __future__ import annotations

import pytest

from app.tests.conftest import import_and_wait, needs_ffmpeg

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.context import UiContext  # noqa: E402
from app.ui.inspector_panel import InspectorPanel  # noqa: E402
from app.ui.qt_bridge import UiBridge  # noqa: E402

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(qapp, project_ws, media_dir):
    ws = project_ws
    v = import_and_wait(ws, media_dir / "clip.mp4")
    i = import_and_wait(ws, media_dir / "pic.png")
    a = import_and_wait(ws, media_dir / "voice.wav")
    panel = InspectorPanel(UiContext(ws, UiBridge(ws.bus)))
    panel.resize(440, 1500)
    panel.show()
    ws.edit.place_assets([v.id], at=0.0)
    ws.edit.place_assets([i.id], at=4.0)
    ws.edit.place_assets([a.id], at=0.0)
    clips = {c.asset_id: c for c in ws.project.timeline.all_clips()}
    qapp.processEvents()
    yield type("Env", (), dict(ws=ws, panel=panel, video=clips[v.id], image=clips[i.id], audio=clips[a.id], assets=(v, i, a)))
    panel.close()


def test_smoke(env, qapp):
    ws, panel = env.ws, env.panel
    for c in ws.project.timeline.all_clips():
        ws.select_clip(c.id)
        qapp.processEvents()
        print(c.id, c.track_id, {k: g.isVisibleTo(panel) for k, g in panel.sections.items()})
        print("start", panel.start.text(), "dur", panel.duration.text())
