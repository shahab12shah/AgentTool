"""Application entry point: ``python -m app.main [project_folder]``."""

from __future__ import annotations

import sys
from pathlib import Path


def _drop_script_dir_from_path() -> None:
    """``python app/main.py`` would put ``app/`` on sys.path, where ``app/logging`` shadows the stdlib."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    root = str(here.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_drop_script_dir_from_path()

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.constants import APP_NAME, APP_VERSION  # noqa: E402
from app.logging.logger import get_logger, log_event, setup_logging  # noqa: E402
from app.services.workspace import Workspace  # noqa: E402
from app.storage.paths import AppPaths  # noqa: E402
from app.ui.context import UiContext  # noqa: E402
from app.ui.dialogs.message import show_error  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.qt_bridge import UiBridge  # noqa: E402
from app.ui.theme import stylesheet  # noqa: E402


def create_window(paths: AppPaths | None = None) -> tuple[MainWindow, Workspace]:
    """Build the workspace and main window (used by ``main`` and by UI tests)."""
    ws = Workspace(paths)
    bridge = UiBridge(ws.bus)
    ws.jobs.set_dispatcher(bridge.dispatch)  # job callbacks run on the UI thread
    QApplication.instance().setStyleSheet(stylesheet(ws.settings.theme))  # type: ignore[union-attr]
    return MainWindow(UiContext(ws, bridge)), ws


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    paths = AppPaths.default()
    setup_logging(paths.log_dir)
    log = get_logger(__name__)
    log_event(log, "app.startup", version=APP_VERSION)

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    def excepthook(exc_type, exc, tb) -> None:
        log.error("Unhandled exception", exc_info=(exc_type, exc, tb))
        try:
            show_error(None, "An unexpected error occurred. The application will keep running; details are in the log.", str(exc))
        except Exception:
            pass

    sys.excepthook = excepthook

    window, ws = create_window(paths)
    window.show()
    if len(argv) > 1 and Path(argv[1]).exists():
        QTimer.singleShot(0, lambda: window.open_project(argv[1]))
    else:
        QTimer.singleShot(0, window.check_recovery)
    code = app.exec()
    log_event(log, "app.exit", code=code)
    return code


if __name__ == "__main__":
    sys.exit(main())
