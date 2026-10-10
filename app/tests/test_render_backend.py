"""Render backend selection wired into the planner (EncoderSelector -> select_backend): Auto / CPU / Hardware, fallbacks, logging, rate-control vocabulary."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from app.performance.hardware import HardwareCapabilityService
from app.project.project_schema import RenderSettings
from app.rendering import presets as P
from app.rendering.errors import EncoderUnavailableError, RenderError
from app.rendering.planner import EncoderSelector, RenderPlanner

SNAP = SimpleNamespace(canvas_w=1920, canvas_h=1080, fps=30)
SNAP_WIDE = SimpleNamespace(canvas_w=5120, canvas_h=2160, fps=30)  # ultra-wide: 2160p export is 5120 pixels wide


@dataclass
class FakeCaps:
    encoders: set = field(default_factory=lambda: {"libx264", "libx265", "libvpx-vp9", "libsvtav1", "aac", "libopus", "flac"})


class FakeFF:
    def __init__(self, hw: dict[str, bool] | None = None) -> None:
        self.hw = hw or {}
        self.caps = FakeCaps()
        self.caps.encoders |= set(self.hw)
        self.hw_calls = 0

    def capabilities(self):
        return self.caps

    def hardware_encoders(self, candidates=None):
        self.hw_calls += 1
        return {n: self.hw.get(n, False) for n in (candidates or list(self.hw)) if n in self.caps.encoders}

    def forget(self):
        pass


def selector(hw=None) -> EncoderSelector:
    return EncoderSelector(FakeFF(hw))


def rs(**kw) -> RenderSettings:
    return replace(RenderSettings(), **kw)


@pytest.fixture
def events():
    got: list[logging.LogRecord] = []

    class H(logging.Handler):
        def emit(self, record):
            if hasattr(record, "event"):
                got.append(record)

    h = H(logging.INFO)
    lg = logging.getLogger("agenttool.rendering.planner")
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    yield got
    lg.removeHandler(h)
    lg.setLevel(old)


def test_cpu_only_machine_resolves_to_software_and_says_why():
    r = selector().resolve(rs(), SNAP)
    assert (r.encoder, r.hardware, r.crf, r.speed_preset) == ("libx264", False, 19, "slow") and any("CPU" in n for n in r.notes)
    assert r.video_args()[:2] == ["-c:v", "libx264"] and "-crf" in r.video_args()


def test_auto_uses_a_tested_hardware_encoder_with_its_own_rate_control():
    r = selector({"h264_nvenc": True}).resolve(rs(), SNAP)
    assert (r.encoder, r.hardware, r.crf) == ("h264_nvenc", True, P.HW_QUALITY["high"])
    a = r.video_args()
    assert a[:2] == ["-c:v", "h264_nvenc"] and a[a.index("-rc") + 1] == "vbr" and a[a.index("-cq") + 1] == "23" and "-crf" not in a
    assert any("NVENC" in n for n in r.notes)


def test_auto_cpu_hardware_decisions_and_user_choice_is_never_overridden():
    sel = selector({"h264_nvenc": True})
    assert sel.resolve(rs(hardware_acceleration="cpu"), SNAP).encoder == "libx264"
    assert sel.resolve(rs(hardware_acceleration="hardware"), SNAP).encoder == "h264_nvenc"
    assert sel.resolve(rs(hardware_acceleration="hardware"), SNAP, force_cpu=True).encoder == "libx264"  # the explicit CPU retry
    assert sel.resolve(rs(video_codec="h265"), SNAP).encoder == "libx265"  # no tested HEVC hardware encoder: CPU
    assert sel.resolve(rs(video_codec="vp9", container="webm", audio_codec="opus"), SNAP).encoder == "libvpx-vp9"


def test_auto_keeps_export_quality_on_the_cpu_when_hardware_cannot_match_it():
    sel = selector({"h264_nvenc": True})
    r = sel.resolve(rs(quality="maximum"), SNAP)
    assert (r.encoder, r.hardware, r.crf) == ("libx264", False, P.CRF["libx264"]["maximum"]) and any("Maximum" in n for n in r.notes)
    r2 = sel.resolve(rs(quality="custom", crf=18), SNAP)
    assert (r2.encoder, r2.crf) == ("libx264", 18)
    assert sel.resolve(rs(quality="maximum", hardware_acceleration="hardware"), SNAP).encoder == "h264_nvenc"


def test_auto_leaves_over_limit_output_to_the_cpu_but_explicit_hardware_is_tried():
    sel = selector({"h264_nvenc": True})
    r = sel.resolve(rs(resolution="2160p"), SNAP_WIDE)
    assert r.encoder == "libx264" and any("4096" in n for n in r.notes)
    assert sel.resolve(rs(resolution="2160p"), SNAP).encoder == "h264_nvenc"  # 3840 wide: within the limit
    forced = sel.resolve(rs(hardware_acceleration="hardware", resolution="2160p"), SNAP_WIDE)
    assert forced.encoder == "h264_nvenc" and any("Warning" in n for n in forced.notes)


def test_hardware_unavailable_is_an_error_with_a_cpu_fallback_offer():
    with pytest.raises(RenderError) as e:
        selector({"h264_nvenc": False}).resolve(rs(hardware_acceleration="hardware"), SNAP)
    assert e.value.kind == "hardware_unavailable" and e.value.can_fallback_cpu and "No working hardware encoder" in e.value.user_message and "H.264" in e.value.user_message
    with pytest.raises(RenderError) as e2:
        selector({"h264_nvenc": True}).resolve(rs(video_codec="vp9", container="webm", audio_codec="opus", hardware_acceleration="hardware"), SNAP)
    assert e2.value.kind == "hardware_unavailable" and e2.value.can_fallback_cpu


def test_error_kinds_that_existed_before_still_apply():
    sel = selector()
    sel.ff.caps.encoders.discard("libx265")
    with pytest.raises(EncoderUnavailableError) as e:
        sel.resolve(rs(video_codec="h265"), SNAP)
    assert "H.264" in e.value.alternatives and e.value.kind == "unsupported_codec"
    sel.ff.caps.encoders.discard("libx264")
    with pytest.raises(RenderError) as e3:
        sel.resolve(rs(hardware_acceleration="hardware"), SNAP)
    assert e3.value.kind == "hardware_unavailable" and not e3.value.can_fallback_cpu  # no software encoder to fall back to
    with pytest.raises(RenderError) as e2:
        sel.resolve(rs(container="webm"), SNAP)
    assert e2.value.kind == "invalid_settings"


def test_selector_and_planner_share_one_service_and_use_its_persisted_results(tmp_path):
    ff = FakeFF({"h264_nvenc": False})
    svc = HardwareCapabilityService(ff, cache_path=tmp_path / "hw.json")
    sel = EncoderSelector(ff, svc)
    assert sel.hardware is svc and RenderPlanner(sel).selector.hardware is svc
    sel.resolve(rs(), SNAP)
    sel.resolve(rs(), SNAP)
    assert ff.hw_calls == 2  # the FFmpeg-side cache (not the fake) is what normally makes this free; the service asks per call


def test_backend_events_are_logged_once_per_distinct_decision(events):
    sel = selector({"h264_nvenc": True})
    sel.resolve(rs(), SNAP)
    sel.resolve(rs(), SNAP)
    assert [r.event for r in events] == ["render.backend_selected"] and events[0].backend == "nvenc" and events[0].encoder == "h264_nvenc"
    sel.resolve(rs(quality="maximum"), SNAP)
    assert [r.event for r in events][-1] == "render.hardware_fallback" and events[-1].reason == "quality_not_honoured" and events[-1].cpu_encoder == "libx264"
    sel.resolve(rs(hardware_acceleration="hardware"), SNAP, force_cpu=True)  # retry on the CPU after a hardware failure: hardware is known to work here
    assert events[-1].event == "render.hardware_fallback" and events[-1].reason == "cpu_requested_for_this_run"
    with pytest.raises(RenderError):
        selector().resolve(rs(hardware_acceleration="hardware"), SNAP)
    for r in events:  # only encoder / backend vocabulary: no paths, names or content
        assert all("/" not in str(getattr(r, k, "")) and "\\" not in str(getattr(r, k, "")) for k in ("backend", "encoder", "requested", "reason", "codec", "cpu_encoder"))


def test_hardware_unavailable_is_logged(events):
    with pytest.raises(RenderError):
        selector({"h264_nvenc": False}).resolve(rs(hardware_acceleration="hardware"), SNAP)
    assert events[-1].event == "render.hardware_unavailable" and events[-1].requested == "hardware"


# ---------------------------------------------------------------- presets refactor keeps every command line identical
@pytest.mark.parametrize("enc", ["h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf", "h264_videotoolbox", "hevc_videotoolbox", "av1_nvenc"])
def test_hardware_video_args_layout(enc):
    r = P.ResolvedOutput(1920, 1080, 30, "mp4", "h265" if "hevc" in enc else "h264", enc, True, "aac", "aac", 192, 48000, "high", "yuv420p", 23, 0, "")
    a = r.video_args()
    assert a[:2] == ["-c:v", enc] and a[2:2 + len(r.rate_args())] == r.rate_args() == P.hw_rate_args(enc, 23, 0, "")
    assert a[2 + len(r.rate_args()):][:4] == ["-g", "60", "-pix_fmt", "yuv420p"] and (a[-2:] == ["-tag:v", "hvc1"]) == ("hevc" in enc)
    assert "-crf" not in a and P.hw_kind(enc) in ("nvenc", "qsv", "amf", "videotoolbox")


def test_resolve_quality_tables():
    assert P.resolve_quality(rs(quality="draft"), "libx264", False) == (30, "ultrafast")
    assert P.resolve_quality(rs(quality="custom", crf=20, encoder_preset="fast"), "libx264", False) == (20, "fast")
    assert P.resolve_quality(rs(quality="custom"), "libx264", False) == (19, "slow")  # custom without a CRF follows "high"
    assert P.resolve_quality(rs(quality="standard"), "h264_nvenc", True) == (28, "")
    assert P.resolve_quality(rs(quality="custom", crf=17, encoder_preset="p7"), "h264_nvenc", True) == (17, "p7")
    assert P.hw_kind("libx264") == "cpu"
