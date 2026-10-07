from __future__ import annotations

import json
import logging

import pytest

from app.ai.interfaces import SceneSegment
from app.ai.provider import AIProviderRegistry, UnavailableAIProvider
from app.core.commands import Command, CommandStack, CompositeCommand
from app.core.config import Settings, SettingsStore
from app.core.events import EventBus
from app.core.exceptions import NotAvailableInPhase
from app.core.timecode import format_duration_short, format_timecode
from app.logging.logger import get_logger, log_event, redact, setup_logging
from app.rendering.renderer import FFmpegRenderer
from app.tests.conftest import import_and_wait, needs_ffmpeg


# ---------------------------------------------------------------- settings
def test_settings_defaults_roundtrip_and_sanitising(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    s = store.load()
    assert s.autosave_interval_seconds == 60 and s.theme == "dark" and s.ffmpeg_path == ""
    s.autosave_interval_seconds = 120
    s.use_proxies = True
    store.save(s)
    again = store.load()
    assert again.autosave_interval_seconds == 120 and again.use_proxies is True
    (tmp_path / "s.json").write_text(json.dumps({"autosave_interval_seconds": 1, "theme": "neon", "bogus": 1}))
    bad = store.load()
    assert bad.autosave_interval_seconds == 10 and bad.theme == "dark"  # clamped / defaulted, unknown keys ignored


def test_settings_corrupt_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "s.json").write_text("{{{")
    assert SettingsStore(tmp_path / "s.json").load() == Settings()


def test_describe_ffmpeg_via_service(ws):
    from app.core.exceptions import FFmpegUnavailableError

    if __import__("shutil").which("ffmpeg"):
        assert ws.describe_ffmpeg().lower().startswith("ffmpeg version")
    with pytest.raises(FFmpegUnavailableError):
        ws.describe_ffmpeg("/definitely/not/here")


def test_update_settings_persists_and_applies_ffmpeg_path(project_ws):
    ws = project_ws
    s = ws.settings
    s.ffmpeg_path = "/nonexistent/ffmpeg"
    ws.update_settings(s)
    assert SettingsStore(ws.paths.settings_file).load().ffmpeg_path == "/nonexistent/ffmpeg"  # persisted
    issues = ws.renderer.validate(ws.project)  # renderer picked up the new path
    assert any("configured ffmpeg path" in i.message for i in issues)
    s.ffmpeg_path = ""
    ws.update_settings(s)
    assert not any("configured" in i.message for i in ws.renderer.validate(ws.project))


# ---------------------------------------------------------------- logging
def test_redaction_masks_secrets():
    assert "abc123" not in redact("calling with api_key=abc123 now")
    assert "hunter2" not in redact("password: hunter2")
    assert redact("nothing secret here") == "nothing secret here"


def test_log_files_are_structured_and_redacted(tmp_path):
    setup_logging(tmp_path, console=False)
    log = get_logger("app.test")
    log_event(log, "unit.test", name="reserved-key-ok", token="tok_12345")  # 'name' is a reserved LogRecord attribute
    log.warning("auth failed token=supersecret")
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("boom api_key=XYZ")
    for h in logging.getLogger("agenttool").handlers:
        h.flush()
    lines = [json.loads(l) for l in (tmp_path / "agenttool.log").read_text().splitlines()]
    assert lines[0]["event"] == "unit.test" and lines[0]["name_"] == "reserved-key-ok"
    text = (tmp_path / "agenttool.log").read_text()
    assert "supersecret" not in text and "XYZ" not in text
    assert any("exception" in l for l in lines)


# ---------------------------------------------------------------- events / commands
def test_event_bus_isolates_broken_handlers():
    bus = EventBus()
    got = []
    bus.subscribe("t", lambda t, p: 1 / 0)
    unsub = bus.subscribe("t", lambda t, p: got.append(p["x"]))
    bus.publish("t", x=1)
    unsub()
    bus.publish("t", x=2)
    assert got == [1]


class _Inc(Command):
    description = "inc"

    def __init__(self, box, fail=False):
        self.box, self.fail = box, fail

    def do(self):
        if self.fail:
            raise ValueError("nope")
        self.box.append(1)

    def undo(self):
        self.box.pop()


def test_command_stack_failed_do_is_not_recorded_and_composite_rolls_back():
    box: list[int] = []
    stack = CommandStack()
    stack.execute(_Inc(box))
    with pytest.raises(ValueError):
        stack.execute(_Inc(box, fail=True))
    assert box == [1] and stack.undo_text == "inc"
    with pytest.raises(ValueError):
        stack.execute(CompositeCommand("combo", [_Inc(box), _Inc(box), _Inc(box, fail=True)]))
    assert box == [1]  # the two successful sub-commands were rolled back
    stack.undo()
    assert box == [] and stack.can_redo and not stack.can_undo
    assert stack.undo() is None


def test_timecode_formatting_is_display_only():
    assert format_timecode(83.5) == "00:01:23.500"
    assert format_timecode(3723.25, fps=30) == "01:02:03:07"
    assert format_timecode(None) == "--:--"
    assert format_duration_short(8.2) == "0:08" and format_duration_short(3661) == "1:01:01"


# ---------------------------------------------------------------- AI / renderer boundaries
def test_ai_provider_is_explicitly_unavailable():
    p = UnavailableAIProvider()
    for call in (
        lambda: p.analyze_script("x"),
        lambda: p.segment_scenes("x"),
        lambda: p.generate_visual_intent(SceneSegment(0, "x")),
        lambda: p.generate_image("x", 1, 1),
    ):
        with pytest.raises(NotAvailableInPhase):
            call()
    reg = AIProviderRegistry()
    assert reg.active.name == "unavailable"


@needs_ffmpeg
def test_renderer_validates_and_plans_but_does_not_fake_rendering(project_ws, media_dir, tmp_path):
    ws = project_ws
    r = ws.renderer
    assert any("empty" in i.message for i in r.validate(ws.project))
    a = import_and_wait(ws, media_dir / "clip.mp4")
    ws.timeline.add_asset(a.id)
    assert r.validate(ws.project) == []
    plan = r.build_render_plan(ws.project)
    assert (plan.width, plan.height, plan.fps) == (1920, 1080, 30) and len(plan.segments) == 1
    assert plan.segments[0].asset_path.is_file() and plan.duration == pytest.approx(a.duration)
    with pytest.raises(NotAvailableInPhase):
        r.render(ws.project, tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists()
    # missing media is reported
    ws.project.asset_path(a).unlink()
    assert any("missing" in i.message.lower() for i in r.validate(ws.project))


def test_renderer_reports_missing_ffmpeg(project_ws):
    issues = FFmpegRenderer("/definitely/not/here").validate(project_ws.project)
    assert any(i.severity == "error" and "path" in i.message.lower() for i in issues)


# ---------------------------------------------------------------- app-level behaviour
def test_new_project_failure_keeps_current_project(project_ws, tmp_path):
    from app.core.exceptions import ProjectError

    ws = project_ws
    ws.timeline.add_track()
    before = ws.project
    with pytest.raises(ProjectError):
        ws.new_project("Bad/Name", tmp_path / "x")
    assert ws.project is before and ws.commands.can_undo  # session untouched


def test_save_as_job_switches_to_copy(project_ws, tmp_path):
    ws = project_ws
    ws.set_script("hello")
    job = ws.save_as(tmp_path / "copies", "Demo Copy")
    assert ws.jobs.wait_idle(20)
    assert ws.project.project_name == "Demo Copy" and ws.project.script.text == "hello"
    assert (tmp_path / "copies" / "Demo Copy" / "project.json").is_file()
    assert (tmp_path / "projects" / "Demo" / "project.json").is_file()  # original left alone
    assert not ws.project.dirty


def test_selection_cleared_when_clip_disappears(project_ws, media_dir):
    ws = project_ws
    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg")
    a = import_and_wait(ws, media_dir / "clip.mp4")
    c = ws.timeline.add_asset(a.id)
    ws.select_clip(c.id)
    ws.undo()
    assert ws.selected_clip_id is None
