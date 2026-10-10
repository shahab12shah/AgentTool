import traceback
from app.tests.test_ui_timeline_perf import *
def test_dbg(big, monkeypatch):
    window, ws, sp = big
    panel = window.timeline_panel
    orig = panel.reload
    def inner(*a, **k):
        traceback.print_stack(limit=8)
        return orig(*a, **k)
    monkeypatch.setattr(panel, "reload", inner)
    canvas=panel.canvas
    canvas.set_zoom(40.0)
    QApplication.processEvents()
    print("--- zoom done")
    panel.zoom.setValue(120)
    QApplication.processEvents()
    print("--- slider done")
    bar = panel.inner.horizontalScrollBar()
    bar.setValue(min(500, bar.maximum()))
    QApplication.processEvents()
    print('--- end')
