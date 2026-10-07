"""QC tests reuse the application's shared fixtures (ws, project_ws, edit_ws, pres_ws, render_ws, needs_ffmpeg …)."""

from app.tests.conftest import *  # noqa: F401,F403
from app.tests.conftest import HAS_FFMPEG, needs_ffmpeg  # noqa: F401
