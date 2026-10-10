"""HardwareCapabilityService + select_backend. All detection goes through an injected runner and a fake FFmpeg: no GPU is needed."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from app.performance.hardware import GB, BackendChoice, HardwareCapabilityService, select_backend
from app.project.project_schema import RenderSettings
from app.rendering import presets as P

LSPCI_NVIDIA = '00:02.0 "VGA compatible controller" "Intel Corporation" "UHD Graphics 630" -r02 "Dell" "0x0830"\n01:00.0 "VGA compatible controller" "NVIDIA Corporation" "GA106 [GeForce RTX 3060]" -ra1 "Gigabyte" "0x4090"\n00:1f.3 "Audio device" "Intel Corporation" "Cannon Lake PCH cAVS" -r10 "Dell" "0x0830"\n'
NVIDIA_SMI = "NVIDIA GeForce RTX 3060, 535.154.05\n"
PS_TWO = json.dumps([{"Name": "Intel(R) UHD Graphics 630", "AdapterCompatibility": "Intel Corporation", "DriverVersion": "31.0.101.2111"},
                     {"Name": "NVIDIA GeForce RTX 3060", "AdapterCompatibility": "NVIDIA", "DriverVersion": "31.0.15.3623"}])
PS_AMD = json.dumps({"Name": "AMD Radeon RX 6700 XT", "AdapterCompatibility": "Advanced Micro Devices, Inc.", "DriverVersion": "31.0.21001.45002"})
WMIC = "Node,AdapterCompatibility,DriverVersion,Name\r\nPC,NVIDIA,30.0.15.1,NVIDIA GeForce GTX 1060\r\n\r\n"
SYSPROF = json.dumps({"SPDisplaysDataType": [{"_name": "Apple M2", "sppci_model": "Apple M2", "spdisplays_vendor": "sppci_vendor_Apple"}]})
HWACCELS = "Hardware acceleration methods:\nvdpau\ncuda\nvaapi\nqsv\ndrm\nopencl\nvulkan\n"


@dataclass
class FakeCaps:
    encoders: set = field(default_factory=lambda: {"libx264", "libx265", "libvpx-vp9", "libsvtav1", "aac", "libopus", "flac"})


@dataclass
class FakeVersion:
    text: str = "ffmpeg version 6.1"


class FakeFF:
    """Duck-typed FFmpegService: only what HardwareCapabilityService / select_backend use."""

    def __init__(self, exe: Path, hw: dict[str, bool] | None = None, extra_encoders: set | None = None, version: str = "ffmpeg version 6.1") -> None:
        self.exe = str(exe)
        self.hw = hw or {}
        self.caps = FakeCaps()
        self.caps.encoders |= set(self.hw) | (extra_encoders or set())
        self.version_text = version
        self.hw_calls = 0
        self.seeded: dict[str, bool] = {}
        self.forgot = 0

    def ffmpeg(self) -> str:
        return self.exe

    def version(self) -> FakeVersion:
        return FakeVersion(self.version_text)

    def capabilities(self) -> FakeCaps:
        return self.caps

    def hardware_encoders(self, candidates=None) -> dict[str, bool]:
        self.hw_calls += 1
        names = [n for n in (candidates or list(self.hw)) if n in self.caps.encoders]
        return {n: self.seeded.get(n, self.hw.get(n, False)) for n in names}

    def seed_hardware_encoders(self, results) -> None:
        self.seeded.update(results)

    def forget(self) -> None:
        self.forgot += 1


class Runner:
    """Canned responses keyed by a substring of the joined command; anything else 'fails to start' (-1) like a missing tool. Records every call."""

    def __init__(self, responses: dict[str, tuple[int, str, str]] | None = None, decode_ok: dict[str, tuple[int, str, str]] | None = None) -> None:
        self.responses = responses or {}
        self.decode = decode_ok or {}
        self.calls: list[list[str]] = []

    def __call__(self, args, timeout):
        self.calls.append(list(args))
        cmd = " ".join(args)
        if "-hwaccel " in cmd and "-hwaccels" not in cmd:
            return self.decode.get(args[args.index("-hwaccel") + 1], (1, "", "Device creation failed"))
        for key, val in self.responses.items():
            if key in cmd:
                return val
        return -1, "", "not found"

    def count(self, needle: str) -> int:
        return sum(1 for c in self.calls if needle in " ".join(c))


@pytest.fixture
def exe(tmp_path) -> Path:
    p = tmp_path / "bin" / "ffmpeg"
    p.parent.mkdir()
    p.write_bytes(b"x")
    return p


def svc(ff, runner, tmp_path, **kw) -> HardwareCapabilityService:
    kw.setdefault("system", "linux")
    kw.setdefault("cpuinfo_path", tmp_path / "no_cpuinfo")
    kw.setdefault("drm_root", tmp_path / "no_drm")
    kw.setdefault("temp_root", tmp_path)
    return HardwareCapabilityService(ff, runner=runner, **kw)


def settings(**kw) -> RenderSettings:
    return replace(RenderSettings(), **kw)


class Mon:
    def __init__(self, total, avail, free, dtotal=500 * GB) -> None:
        self.s = type("S", (), {"system_total_bytes": total, "system_available_bytes": avail, "disk_free_bytes": free, "disk_total_bytes": dtotal})()

    def sample(self):
        return self.s


# ---------------------------------------------------------------- CPU-only machine
def test_cpu_only_machine_selects_cpu_and_recommends_cpu(exe, tmp_path):
    ff = FakeFF(exe)
    r = Runner({"-hwaccels": (0, "Hardware acceleration methods:\n", "")})
    s = svc(ff, r, tmp_path, monitor=Mon(16 * GB, 8 * GB, 200 * GB))
    assert s.detect_gpu() == []
    assert s.detect_ffmpeg_encoders()["hardware"] == {}
    assert s.detect_hardware_decoders() == {"hwaccels": [], "working": {}}
    rec = s.get_recommended_profile()
    assert rec["render_backend"] == "cpu" and rec["profile"] in ("balanced", "performance") and rec["reasons"]
    ch = select_backend(settings(), None, ff, s)
    assert (ch.kind, ch.encoder, ch.outcome) == ("cpu", "libx264", "cpu_fallback") and ch.reason == "no_working_hardware_encoder" and any("CPU" in n for n in ch.notes)
    assert "-crf" in ch.quality_args and ch.decode_args == []
    text = s.summary()["text"]
    assert "Graphics: none detected" in text and "Hardware video encoders that work: none" in text


def test_summary_without_detect_is_cheap_and_reports_pending(exe, tmp_path):
    r = Runner()
    s = svc(FakeFF(exe), r, tmp_path)
    out = s.summary(allow_detect=False)
    assert set(out["pending"]) == {"gpu", "encoders", "decoders"} and r.calls == []  # nothing slow was run
    assert out["cpu"]["logical_cores"] >= 1 and isinstance(out["text"], str)


# ---------------------------------------------------------------- GPU detection per platform
def test_nvidia_linux_lspci_and_nvidia_smi(exe, tmp_path):
    r = Runner({"lspci": (0, LSPCI_NVIDIA, ""), "nvidia-smi": (0, NVIDIA_SMI, "")})
    gpus = svc(FakeFF(exe), r, tmp_path).detect_gpu()
    assert [g["vendor"] for g in gpus] == ["intel", "nvidia"]
    nv = gpus[1]
    assert nv["model"] == "NVIDIA GeForce RTX 3060" and nv["driver"] == "535.154.05" and nv["source"] == "nvidia-smi"
    assert gpus[0]["source"] == "lspci" and "UHD" in gpus[0]["model"]
    assert r.calls[1][:2] == ["nvidia-smi", "--query-gpu=name,driver_version"] and r.calls[1][2] == "--format=csv,noheader"


def test_linux_sysfs_fallback_when_no_tools(exe, tmp_path):
    drm = tmp_path / "drm"
    for name, vid in (("card0", "0x8086"), ("card0-HDMI-A-1", "0x8086"), ("card1", "0x1002"), ("renderD128", "0x1002")):
        (drm / name / "device").mkdir(parents=True)
        (drm / name / "device" / "vendor").write_text(vid + "\n")
    gpus = svc(FakeFF(exe), Runner(), tmp_path, drm_root=drm).detect_gpu()
    assert [g["vendor"] for g in gpus] == ["intel", "amd"] and all(g["source"] == "sysfs" for g in gpus)


def test_windows_powershell_list_dict_and_wmic_fallback(exe, tmp_path):
    two = svc(FakeFF(exe), Runner({"Win32_VideoController": (0, PS_TWO, "")}), tmp_path, system="win32").detect_gpu()
    assert [g["vendor"] for g in two] == ["intel", "nvidia"] and two[1]["driver"] == "31.0.15.3623" and two[1]["source"] == "powershell"
    one = svc(FakeFF(exe), Runner({"Win32_VideoController": (0, "﻿" + PS_AMD, "")}), tmp_path, system="win32").detect_gpu()
    assert len(one) == 1 and one[0]["vendor"] == "amd" and "Radeon" in one[0]["model"]
    r = Runner({"Win32_VideoController": (1, "", "boom"), "wmic": (0, WMIC, "")})
    w = svc(FakeFF(exe), r, tmp_path, system="win32").detect_gpu()
    assert w == [{"vendor": "nvidia", "model": "NVIDIA GeForce GTX 1060", "driver": "30.0.15.1", "source": "wmic"}]
    assert r.calls[0][:3] == ["powershell", "-NoProfile", "-NonInteractive"]


def test_macos_system_profiler(exe, tmp_path):
    g = svc(FakeFF(exe), Runner({"system_profiler": (0, SYSPROF, "")}), tmp_path, system="darwin").detect_gpu()
    assert g == [{"vendor": "apple", "model": "Apple M2", "driver": None, "source": "system_profiler"}]


@pytest.mark.parametrize("platform_name", ["linux", "win32", "darwin"])
@pytest.mark.parametrize("failure", ["raises", "timeout", "garbage", "nonzero", "empty"])
def test_detection_failures_are_safe(exe, tmp_path, platform_name, failure):
    def runner(args, timeout):
        if failure == "raises":
            raise RuntimeError("boom")
        return {"timeout": (-2, "", "timeout"), "garbage": (0, "\x00{{{ not json \xff", ""), "nonzero": (3, "x", "y"), "empty": (0, "", "")}[failure]

    s = svc(FakeFF(exe, {"h264_nvenc": False}), runner, tmp_path, system=platform_name)
    assert s.detect_gpu() == []
    cpu = s.detect_cpu()
    assert cpu["logical_cores"] >= 1 and set(cpu) == {"arch", "logical_cores", "physical_cores", "model"}
    assert s.detect_hardware_decoders()["working"] == {}
    out = s.summary()
    assert out["gpu"] == [] and isinstance(out["text"], str)
    assert s.get_recommended_profile()["render_backend"] == "cpu"


def test_no_ffmpeg_at_all_is_safe(tmp_path):
    s = svc(None, Runner(), tmp_path)
    assert s.detect_ffmpeg_encoders() == {"software": [], "hardware": {}} and s.detect_hardware_decoders() == {"hwaccels": [], "working": {}}
    assert s.summary()["ffmpeg_version"] is None and s.get_recommended_profile()["render_backend"] == "cpu"


def test_ffmpeg_that_raises_is_safe(tmp_path):
    class Broken:
        def ffmpeg(self):
            raise OSError("missing")

        def version(self):
            raise OSError("missing")

        def capabilities(self):
            raise OSError("missing")

        def hardware_encoders(self, c=None):
            raise OSError("missing")

    s = svc(Broken(), Runner(), tmp_path)
    assert s.detect_all()["encoders"] == {"software": [], "hardware": {}}
    assert select_backend(settings(hardware_acceleration="hardware"), None, Broken(), s).outcome == "hardware_unavailable"


def test_cpu_linux_cpuinfo_and_windows_cim(exe, tmp_path):
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nmodel name\t: Test CPU @ 3.0GHz\nphysical id\t: 0\ncore id\t: 0\n\nprocessor\t: 1\nmodel name\t: Test CPU @ 3.0GHz\nphysical id\t: 0\ncore id\t: 0\n\n"
                    "processor\t: 2\nmodel name\t: Test CPU @ 3.0GHz\nphysical id\t: 0\ncore id\t: 1\n")
    c = svc(FakeFF(exe), Runner(), tmp_path, cpuinfo_path=info).detect_cpu()
    assert c["model"] == "Test CPU @ 3.0GHz" and (c["physical_cores"] in (2, None))  # None when the sandbox has fewer logical CPUs than the fake file claims
    w = svc(FakeFF(exe), Runner({"Win32_Processor": (0, json.dumps({"Name": "Intel(R) Core(TM) i7", "NumberOfCores": 1}), "")}), tmp_path, system="win32").detect_cpu()
    assert w["model"] == "Intel(R) Core(TM) i7" and w["physical_cores"] == 1
    m = svc(FakeFF(exe), Runner({"brand_string": (0, "Apple M2\n", ""), "hw.physicalcpu": (0, "1\n", "")}), tmp_path, system="darwin").detect_cpu()
    assert m["model"] == "Apple M2" and m["physical_cores"] == 1


def test_memory_and_disk_use_monitor_or_stdlib(exe, tmp_path):
    s = svc(FakeFF(exe), Runner(), tmp_path, monitor=Mon(16 * GB, 4 * GB, 77 * GB))
    assert s.detect_memory() == {"total_bytes": 16 * GB, "available_bytes": 4 * GB} and s.detect_disk() == {"free": 77 * GB, "total": 500 * GB}
    real = svc(FakeFF(exe), Runner(), tmp_path)
    d = real.detect_disk(tmp_path / "does" / "not" / "exist")  # nearest existing parent
    assert d["free"] > 0 and d["total"] >= d["free"]
    assert set(real.detect_memory()) == {"total_bytes", "available_bytes"}


# ---------------------------------------------------------------- hardware decoders
def test_hwaccel_listed_but_test_decode_fails_is_not_working(exe, tmp_path):
    r = Runner({"-hwaccels": (0, HWACCELS, ""), "libx264": (0, "", "")}, decode_ok={"vaapi": (0, "", ""), "cuda": (1, "", "Device creation failed: -1"),
                                                                                    "qsv": (0, "", "[h264 @ 0x1] Failed setup for format qsv: hwaccel initialisation returned error.")})
    dec = svc(FakeFF(exe), r, tmp_path).detect_hardware_decoders()
    assert dec["hwaccels"] == ["vdpau", "cuda", "vaapi", "qsv", "drm", "opencl", "vulkan"]
    assert dec["working"]["vaapi"] is True
    assert dec["working"]["cuda"] is False  # non-zero exit
    assert dec["working"]["qsv"] is False  # exit 0 but FFmpeg reported it fell back to software
    assert "drm" not in dec["working"] and "opencl" not in dec["working"]  # not general decoders: never tested
    assert r.count("-f null") == len(dec["working"])


def test_decode_test_clip_in_path_with_spaces_and_unicode_is_cleaned_up(exe, tmp_path):
    root = tmp_path / "Vidéo Projekte ü 日本語 (test)"
    root.mkdir()
    seen: list[str] = []

    def runner(args, timeout):
        if args[-1].endswith("probe clip.mp4"):
            seen.append(args[-1])
            assert Path(args[-1]).parent.parent == root and Path(args[-1]).parent.exists()
            Path(args[-1]).write_bytes(b"clip") if "libx264" in args else None
            return 0, "", ""
        if "-hwaccels" in args:
            return 0, "Hardware acceleration methods:\nvaapi\n", ""
        assert "-hwaccel" in args and Path(args[args.index("-i") + 1]).exists()  # the clip really is there while it is being decoded
        seen.append(args[args.index("-i") + 1])
        return 0, "", ""

    dec = svc(FakeFF(exe), runner, tmp_path, temp_root=root).detect_hardware_decoders()
    assert dec["working"] == {"vaapi": True} and len(seen) == 2 and all(" " in p for p in seen)
    assert list(root.iterdir()) == []  # temp folder removed
    assert all(isinstance(a, str) for a in seen)


def test_decode_clip_generation_failure_means_nothing_is_working(exe, tmp_path):
    r = Runner({"-hwaccels": (0, HWACCELS, ""), "libx264": (1, "", "Unknown encoder")}, decode_ok={"vaapi": (0, "", "")})
    assert not any(svc(FakeFF(exe), r, tmp_path).detect_hardware_decoders()["working"].values())
    assert list(tmp_path.glob("agenttool hw probe*")) == []


def test_decoder_tests_stop_on_cancel_and_are_not_cached(exe, tmp_path):
    ev = threading.Event()
    r = Runner({"-hwaccels": (0, HWACCELS, ""), "libx264": (0, "", "")}, decode_ok={"cuda": (0, "", ""), "vaapi": (0, "", ""), "qsv": (0, "", "")})
    orig = r.__call__

    def runner(args, timeout):
        out = orig(args, timeout)
        if "-hwaccel" in args and "-hwaccels" not in args:
            ev.set()  # cancelled while the first test runs
        return out

    s = svc(FakeFF(exe), runner, tmp_path, cache_path=tmp_path / "hw.json")
    part = s.detect_hardware_decoders(cancel=ev)
    assert r.count("-f null") == 1 and part["working"] == {"cuda": True, "qsv": False, "vaapi": False, "vulkan": False, "vdpau": False}  # stopped after the first test; the untested ones are not claimed to work
    assert not (tmp_path / "hw.json").exists() and s._dec is None  # partial results are never cached


# ---------------------------------------------------------------- backend selection
HW_ALL = {"h264_nvenc": True, "hevc_nvenc": True, "h264_qsv": False, "h264_amf": False, "h264_videotoolbox": False}


@pytest.mark.parametrize("hw,kind,rate", [
    ({"h264_nvenc": True}, "nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]),
    ({"h264_qsv": True}, "qsv", ["-global_quality", "23", "-preset", "slow"]),
    ({"h264_amf": True}, "amf", ["-quality", "quality", "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"]),
    ({"h264_videotoolbox": True}, "videotoolbox", ["-q:v", "54"]),
])
def test_auto_picks_the_tested_hardware_encoder_with_its_own_rate_control(exe, tmp_path, hw, kind, rate):
    ff = FakeFF(exe, hw)
    ch = select_backend(settings(), None, ff, svc(ff, Runner(), tmp_path), output_size=(1920, 1080), fps=30)
    assert (ch.kind, ch.encoder, ch.outcome, ch.reason) == (kind, next(iter(hw)), "ok", "hardware_tested_ok") and ch.hardware
    assert ch.quality_args == rate and "-crf" not in ch.quality_args
    assert ch.fallback is not None and ch.fallback.encoder == "libx264" and ch.offers_cpu_fallback and "-crf" in ch.fallback.quality_args
    assert ch.decode_args == []  # no sources/filters given


def test_quality_arg_mapping_follows_quality_level_and_bitrate(exe, tmp_path):
    ff = FakeFF(exe, {"h264_nvenc": True, "h264_amf": True, "h264_qsv": True, "h264_videotoolbox": True})
    s = svc(ff, Runner(), tmp_path)
    draft = select_backend(settings(quality="draft"), None, ff, s)
    assert draft.encoder == "h264_nvenc" and draft.quality_args[draft.quality_args.index("-cq") + 1] == str(P.HW_QUALITY["draft"])
    br = select_backend(settings(quality="custom", bitrate_kbps=8000), None, ff, s)
    assert br.encoder == "h264_nvenc" and br.quality_args[-2:] == ["-b:v", "8000k"]
    for enc, expect in (("h264_amf", ["-b:v", "8000k"]), ("h264_videotoolbox", ["-b:v", "8000k"]), ("h264_qsv", ["-b:v", "8000k"])):
        f2 = FakeFF(exe, {enc: True})
        c = select_backend(settings(quality="custom", bitrate_kbps=8000, hardware_acceleration="hardware"), None, f2, svc(f2, Runner(), tmp_path))
        assert c.encoder == enc and c.quality_args[-2:] == expect and "-crf" not in c.quality_args
    c = select_backend(settings(quality="custom", crf=17, hardware_acceleration="hardware"), None, ff, s)  # explicit Hardware: the user's value is passed on in the encoder's own option
    assert c.encoder == "h264_nvenc" and c.quality_args[c.quality_args.index("-cq") + 1] == "17"
    assert select_backend(settings(preset_id="custom", encoder_preset="p7"), None, ff, s).quality_args[1] == "p7"


def test_cpu_mode_is_always_software_and_never_tests_hardware(exe, tmp_path):
    ff = FakeFF(exe, HW_ALL)
    ch = select_backend(settings(hardware_acceleration="cpu"), None, ff, svc(ff, Runner(), tmp_path))
    assert (ch.kind, ch.encoder, ch.outcome, ch.reason) == ("cpu", "libx264", "ok", "cpu_requested") and ff.hw_calls == 0 and ch.decode_args == []


def test_force_cpu_retry_reports_a_fallback_only_when_hardware_was_known_to_work(exe, tmp_path):
    ff = FakeFF(exe, HW_ALL)
    s = svc(ff, Runner(), tmp_path)
    assert select_backend(settings(hardware_acceleration="hardware"), None, ff, s, force_cpu=True).outcome == "ok"  # nothing known yet: cheap path, nothing tested
    s.tested_hardware_encoders(["h264_nvenc"])
    c = select_backend(settings(hardware_acceleration="hardware"), None, ff, s, force_cpu=True)
    assert (c.kind, c.encoder, c.outcome, c.reason) == ("cpu", "libx264", "cpu_fallback", "cpu_requested_for_this_run")


def test_hardware_mode_without_a_working_encoder_is_explicit_and_offers_cpu(exe, tmp_path):
    ff = FakeFF(exe, {"h264_nvenc": False})  # listed in the build but the test encode failed
    ch = select_backend(settings(hardware_acceleration="hardware"), None, ff, svc(ff, Runner(), tmp_path))
    assert ch.outcome == "hardware_unavailable" and ch.encoder == "" and ch.kind == "cpu" and ch.requested == "hardware"
    assert ch.offers_cpu_fallback and ch.fallback.encoder == "libx264" and ch.fallback.kind == "cpu" and ch.fallback.quality_args
    assert any("No working hardware encoder" in n for n in ch.notes)


def test_unsupported_codec_for_hardware(exe, tmp_path):
    ff = FakeFF(exe, HW_ALL)
    s = svc(ff, Runner(), tmp_path)
    vp9 = settings(video_codec="vp9", container="webm", audio_codec="opus")
    hw = select_backend(replace(vp9, hardware_acceleration="hardware"), None, ff, s)
    assert hw.outcome == "hardware_unavailable" and hw.fallback.encoder == "libvpx-vp9"
    auto = select_backend(vp9, None, ff, s)
    assert auto.kind == "cpu" and auto.encoder == "libvpx-vp9" and auto.outcome == "cpu_fallback" and "-crf" in auto.quality_args
    av1 = select_backend(settings(video_codec="av1"), None, FakeFF(exe, {"h264_nvenc": True}, {"libsvtav1"}), s)  # NVENC for H.264 only: AV1 stays on the CPU
    assert av1.encoder == "libsvtav1"
    hevc = select_backend(settings(video_codec="h265"), None, ff, s)
    assert hevc.encoder == "hevc_nvenc" and hevc.kind == "nvenc"


def test_codec_without_any_software_encoder(exe, tmp_path):
    ff = FakeFF(exe)
    ff.caps.encoders.discard("libx264")
    ch = select_backend(settings(), None, ff, svc(ff, Runner(), tmp_path))
    assert ch.encoder == "" and ch.fallback is None and not ch.offers_cpu_fallback


def test_auto_keeps_quality_by_using_cpu_where_hardware_cannot_honour_it(exe, tmp_path):
    ff = FakeFF(exe, {"h264_nvenc": True})
    s = svc(ff, Runner(), tmp_path)
    mx = select_backend(settings(quality="maximum"), None, ff, s)
    assert (mx.kind, mx.outcome, mx.reason) == ("cpu", "cpu_fallback", "quality_not_honoured") and "Maximum" in mx.notes[0]
    crf = select_backend(settings(quality="custom", crf=18), None, ff, s)
    assert crf.kind == "cpu" and crf.reason == "quality_not_honoured" and crf.quality_args[crf.quality_args.index("-crf") + 1] == "18"
    assert select_backend(settings(quality="custom", crf=18, bitrate_kbps=6000), None, ff, s).kind == "nvenc"  # an explicit bitrate maps to every encoder
    assert select_backend(settings(quality="maximum", hardware_acceleration="hardware"), None, ff, s).kind == "nvenc"  # the user chose Hardware: honoured
    assert select_backend(settings(quality="standard"), None, ff, s).kind == "nvenc"


def test_auto_respects_output_limits_but_explicit_hardware_only_warns(exe, tmp_path):
    ff = FakeFF(exe, {"h264_nvenc": True, "hevc_nvenc": True})
    s = svc(ff, Runner(), tmp_path)
    big = select_backend(settings(), None, ff, s, output_size=(7680, 4320), fps=30)
    assert big.kind == "cpu" and big.reason == "output_beyond_hardware_limits" and "4096" in big.notes[0]
    assert select_backend(settings(), None, ff, s, output_size=(1920, 1080), fps=240).reason == "output_beyond_hardware_limits"
    assert select_backend(settings(), None, ff, s, output_size=(3840, 2160), fps=60).kind == "nvenc"
    assert select_backend(settings(video_codec="h265"), None, ff, s, output_size=(7680, 4320), fps=30).kind == "nvenc"
    forced = select_backend(settings(hardware_acceleration="hardware"), None, ff, s, output_size=(7680, 4320), fps=30)
    assert forced.kind == "nvenc" and any("Warning" in n for n in forced.notes)


@dataclass
class Src:
    codec: str = "h264"
    pix_fmt: str = "yuv420p"
    kind: str = "video"
    has_alpha: bool = False
    has_video: bool = True


def test_hardware_decode_args_only_when_tested_safe_and_not_for_export(exe, tmp_path):
    ff = FakeFF(exe, {"h264_nvenc": True})
    r = Runner({"-hwaccels": (0, "Hardware acceleration methods:\ncuda\nvaapi\n", ""), "libx264": (0, "", "")}, decode_ok={"vaapi": (0, "", ""), "cuda": (1, "", "")})
    s = svc(ff, r, tmp_path)
    flt = ["scale", "overlay", "fps", "format"]
    exp = select_backend(settings(), [Src()], ff, s, filters=flt)  # final export
    assert exp.decode_args == [] and any("software" in n for n in exp.notes) and not r.calls  # decided without even running a decode test
    pv = select_backend(settings(), [Src()], ff, s, filters=flt, purpose="preview")
    assert pv.decode_args == ["-hwaccel", "vaapi"] and pv.hwaccel == "vaapi"  # cuda is listed but failed its test
    assert select_backend(settings(), [Src()], ff, s, filters=flt, purpose="export", allow_hw_decode_export=True).decode_args == ["-hwaccel", "vaapi"]
    assert select_backend(settings(), [Src()], ff, s, filters=flt + ["libplacebo"], purpose="preview").decode_args == []
    assert select_backend(settings(), [Src()], ff, s, filters=None, purpose="preview").decode_args == []
    assert select_backend(settings(), [Src(codec="hevc")], ff, s, filters=flt, purpose="preview").decode_args == []
    assert select_backend(settings(), [Src(pix_fmt="yuv420p10le")], ff, s, filters=flt, purpose="preview").decode_args == []
    assert select_backend(settings(), [Src(), Src(has_alpha=True)], ff, s, filters=flt, purpose="preview").decode_args == []
    assert select_backend(settings(), [Src(kind="image", codec="png")], ff, s, filters=flt, purpose="preview").decode_args == []  # no video sources: nothing to decode
    none_ok = svc(ff, Runner({"-hwaccels": (0, "Hardware acceleration methods:\ncuda\n", "")}), tmp_path)
    assert select_backend(settings(), [Src()], ff, none_ok, filters=flt, purpose="preview").decode_args == []
    assert all("-hwaccel_output_format" not in a for a in (pv.decode_args, exp.decode_args))  # frames always come back to system memory


def test_decode_args_with_cpu_encoder_when_user_chose_cpu(exe, tmp_path):
    ff = FakeFF(exe)
    r = Runner({"-hwaccels": (0, "Hardware acceleration methods:\nvaapi\n", ""), "libx264": (0, "", "")}, decode_ok={"vaapi": (0, "", "")})
    ch = select_backend(settings(hardware_acceleration="cpu"), [Src()], ff, svc(ff, r, tmp_path), filters=["scale"], purpose="preview")
    assert ch.kind == "cpu" and ch.decode_args == []  # CPU means software everywhere


def test_choice_to_dict_is_json_serialisable(exe, tmp_path):
    ff = FakeFF(exe, {"h264_nvenc": True})
    d = select_backend(settings(), None, ff, svc(ff, Runner(), tmp_path)).to_dict()
    assert json.loads(json.dumps(d))["fallback"]["encoder"] == "libx264" and isinstance(BackendChoice().to_dict(), dict)


def test_select_backend_without_a_service_uses_ffmpeg_directly(exe):
    ff = FakeFF(exe, {"h264_nvenc": True})
    assert select_backend(settings(), None, ff, None).kind == "nvenc" and select_backend(settings(), [Src()], ff, None, filters=["scale"], purpose="preview").decode_args == []


# ---------------------------------------------------------------- caching
def full_runner() -> Runner:
    return Runner({"lspci": (0, LSPCI_NVIDIA, ""), "nvidia-smi": (0, NVIDIA_SMI, ""), "-hwaccels": (0, "Hardware acceleration methods:\ncuda\n", ""), "libx264": (0, "", "")}, decode_ok={"cuda": (0, "", "")})


def test_memory_cache_makes_repeated_calls_free(exe, tmp_path):
    ff, r = FakeFF(exe, {"h264_nvenc": True}), full_runner()
    s = svc(ff, r, tmp_path)
    s.detect_all()
    calls, hw_calls = len(r.calls), ff.hw_calls
    t0 = time.perf_counter()
    for _ in range(200):
        s.detect_gpu(), s.detect_cpu(), s.detect_ffmpeg_encoders(), s.detect_hardware_decoders(), s.summary(), s.get_recommended_profile()
    assert len(r.calls) == calls and ff.hw_calls == hw_calls  # nothing was probed again
    assert time.perf_counter() - t0 < 5.0


def test_cache_file_survives_restart_and_is_invalidated_by_ffmpeg_change(exe, tmp_path):
    cache = tmp_path / "state" / "hardware.json"
    ff, r = FakeFF(exe, {"h264_nvenc": True, "h264_qsv": False}), full_runner()
    first = svc(ff, r, tmp_path, cache_path=cache)
    out1 = first.detect_all()
    assert cache.is_file() and out1["encoders"]["hardware"] == {"h264_nvenc": True, "h264_qsv": False} and out1["decoders"]["working"] == {"cuda": True}
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["key"]["version"] == "ffmpeg version 6.1" and data["key"]["path"] == str(exe) and data["encoders"]["hardware"]["h264_nvenc"] is True

    # restart: a new service + a new FFmpeg object with the same binary -> nothing is re-tested, the FFmpeg test cache is pre-filled
    ff2, r2 = FakeFF(exe, {"h264_nvenc": False, "h264_qsv": False}), Runner()
    second = svc(ff2, r2, tmp_path, cache_path=cache)
    out2 = second.summary()
    assert r2.calls == [] and ff2.hw_calls == 0 and out2["encoders"]["hardware"]["h264_nvenc"] is True and out2["gpu"][1]["driver"] == "535.154.05" and out2["tested_at"]
    assert ff2.seeded == {"h264_nvenc": True, "h264_qsv": False}
    assert select_backend(settings(), None, ff2, second).kind == "nvenc"  # the planner path uses the persisted result (the fake's own answer would be False)

    # a new FFmpeg version (or a changed binary) is a different key -> tested again
    ff3, r3 = FakeFF(exe, {"h264_nvenc": False}, version="ffmpeg version 7.0"), full_runner()
    third = svc(ff3, r3, tmp_path, cache_path=cache)
    assert third.detect_ffmpeg_encoders()["hardware"] == {"h264_nvenc": False} and ff3.hw_calls == 1
    exe.write_bytes(b"changed binary!")
    ff4 = FakeFF(exe, {"h264_nvenc": True}, version="ffmpeg version 7.0")
    assert svc(ff4, Runner(), tmp_path, cache_path=cache).detect_ffmpeg_encoders()["hardware"] == {"h264_nvenc": True} and ff4.hw_calls == 1


@pytest.mark.parametrize("damage", ["garbage", "empty", "list", "stale", "future", "wrong_schema", "bad_sections"])
def test_corrupt_or_stale_cache_file_means_retest_not_crash(exe, tmp_path, damage):
    cache = tmp_path / "hardware.json"
    ff = FakeFF(exe, {"h264_nvenc": True})
    svc(ff, full_runner(), tmp_path, cache_path=cache).detect_all()
    data = json.loads(cache.read_text(encoding="utf-8"))
    if damage == "garbage":
        cache.write_text("{not json", encoding="utf-8")
    elif damage == "empty":
        cache.write_text("", encoding="utf-8")
    elif damage == "list":
        cache.write_text("[1,2]", encoding="utf-8")
    elif damage == "stale":
        cache.write_text(json.dumps({**data, "tested_at": time.time() - 90 * 86400}), encoding="utf-8")
    elif damage == "future":
        cache.write_text(json.dumps({**data, "tested_at": time.time() + 10 * 86400}), encoding="utf-8")
    elif damage == "wrong_schema":
        cache.write_text(json.dumps({**data, "schema": 99}), encoding="utf-8")
    else:
        cache.write_text(json.dumps({**data, "encoders": "x", "decoders": 5, "gpu": {"a": 1}}), encoding="utf-8")
    ff2 = FakeFF(exe, {"h264_nvenc": True})
    s = svc(ff2, full_runner(), tmp_path, cache_path=cache)
    assert s.detect_ffmpeg_encoders()["hardware"] == {"h264_nvenc": True} and ff2.hw_calls == 1
    assert json.loads(cache.read_text(encoding="utf-8"))["schema"] == 1  # rewritten with fresh results


def test_unwritable_cache_location_is_ignored(exe, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    s = svc(FakeFF(exe, {"h264_nvenc": True}), full_runner(), tmp_path, cache_path=blocker / "sub" / "hw.json")  # parent is a file
    assert s.detect_all()["encoders"]["hardware"] == {"h264_nvenc": True}


def test_forget_and_refresh_retest_and_clear_the_file(exe, tmp_path):
    cache = tmp_path / "hardware.json"
    ff, r = FakeFF(exe, {"h264_nvenc": True}), full_runner()
    s = svc(ff, r, tmp_path, cache_path=cache)
    s.detect_all()
    assert cache.is_file()
    s.forget()
    assert not cache.exists() and ff.forgot == 1 and s.summary(allow_detect=False)["pending"] == ["gpu", "encoders", "decoders"]
    ff.hw = {"h264_nvenc": False}
    ff.seeded.clear()
    out = s.refresh()
    assert out["encoders"]["hardware"] == {"h264_nvenc": False} and cache.is_file() and ff.forgot == 2


def test_detect_all_reports_progress_and_honours_cancel(exe, tmp_path):
    steps: list[tuple[float, str]] = []
    s = svc(FakeFF(exe, {"h264_nvenc": True}), full_runner(), tmp_path)
    out = s.detect_all(progress=lambda f, m: steps.append((f, m)))
    assert [f for f, _ in steps] == sorted(f for f, _ in steps) and steps[-1][0] == 1.0 and out["pending"] == []
    ev = threading.Event()
    ev.set()
    s2 = svc(FakeFF(exe, {"h264_nvenc": True}), full_runner(), tmp_path)
    part = s2.detect_all(cancel=ev)
    assert part["pending"] == ["gpu", "encoders", "decoders"] and s2._gpu is None  # cancelled before the slow probes
    s3 = svc(FakeFF(exe), full_runner(), tmp_path)
    s3.detect_all(progress=lambda f, m: (_ for _ in ()).throw(RuntimeError("ui gone")))  # a broken progress callback never breaks detection
    assert s3._enc is not None


def test_concurrent_detection_runs_each_probe_once(exe, tmp_path):
    ff, r = FakeFF(exe, {"h264_nvenc": True}), full_runner()
    s = svc(ff, r, tmp_path)
    ts = [threading.Thread(target=s.detect_all) for _ in range(4)]
    [t.start() for t in ts]
    [t.join(timeout=20) for t in ts]
    assert r.count("lspci") == 1 and ff.hw_calls == 1 and r.count("-f null") == 1


# ---------------------------------------------------------------- recommended profile
def rec(exe, tmp_path, ram, free, cores=None, hw=None, monkeypatch=None):
    ff = FakeFF(exe, hw or {})
    s = svc(ff, Runner(), tmp_path, monitor=Mon(ram, ram // 2 if ram else None, free))
    if cores is not None:
        s._cpu = {"arch": "x86_64", "logical_cores": cores, "physical_cores": None, "model": "t"}
    return s.get_recommended_profile()


def test_recommended_profile_tiers(exe, tmp_path):
    big = rec(exe, tmp_path, 32 * GB, 400 * GB, cores=16)
    assert big["profile"] == "performance" and big["proxy_policy"] == "automatic" and big["render_backend"] == "cpu"
    mid = rec(exe, tmp_path, 16 * GB, 100 * GB, cores=4)
    assert mid["profile"] == "balanced" and mid["proxy_policy"] == "manual"
    low_ram = rec(exe, tmp_path, 2 * GB, 400 * GB, cores=16)
    assert low_ram["profile"] == "power_saver" and any("memory" in x.lower() for x in low_ram["reasons"])
    few_cores = rec(exe, tmp_path, 32 * GB, 400 * GB, cores=2)
    assert few_cores["profile"] == "power_saver"
    low_disk = rec(exe, tmp_path, 32 * GB, 3 * GB, cores=16)
    assert low_disk["profile"] == "power_saver" and low_disk["proxy_policy"] == "off" and any("disk" in x.lower() for x in low_disk["reasons"])
    some_disk = rec(exe, tmp_path, 32 * GB, 12 * GB, cores=16)
    assert some_disk["profile"] == "balanced" and some_disk["proxy_policy"] == "manual"
    unknown_ram = rec(exe, tmp_path, None, 400 * GB, cores=16)
    assert unknown_ram["profile"] == "balanced" and any("could not be read" in x for x in unknown_ram["reasons"])
    for r in (big, mid, low_ram, few_cores, low_disk, some_disk, unknown_ram):
        assert r["profile"] in ("power_saver", "balanced", "performance") and r["render_backend"] in ("auto", "cpu", "hardware") and r["proxy_policy"] in ("off", "manual", "automatic")


def test_recommendation_never_suggests_hardware_that_was_not_tested(exe, tmp_path):
    failing = rec(exe, tmp_path, 32 * GB, 400 * GB, cores=8, hw={"h264_nvenc": False, "hevc_nvenc": False})
    assert failing["render_backend"] == "cpu" and any("No hardware video encoder passed" in x for x in failing["reasons"])
    passing = rec(exe, tmp_path, 32 * GB, 400 * GB, cores=8, hw={"h264_nvenc": True})
    assert passing["render_backend"] == "auto" and any("h264_nvenc" in x for x in passing["reasons"])  # "auto" (use it when the settings allow), never forced "hardware"
    # a GPU that is merely present (lspci) without a working encoder does not change the recommendation
    ff = FakeFF(exe)
    s = svc(ff, Runner({"lspci": (0, LSPCI_NVIDIA, ""), "nvidia-smi": (0, NVIDIA_SMI, "")}), tmp_path, monitor=Mon(32 * GB, 16 * GB, 400 * GB))
    assert s.detect_gpu() and s.get_recommended_profile()["render_backend"] == "cpu"


# ---------------------------------------------------------------- real FFmpeg
from app.tests.conftest import needs_ffmpeg  # noqa: E402


@needs_ffmpeg
def test_real_ffmpeg_detection_and_software_path(tmp_path):
    from app.rendering.ffmpeg_service import FFmpegService

    ff = FFmpegService()
    s = HardwareCapabilityService(ff, cache_path=tmp_path / "hw.json", temp_root=tmp_path)
    out = s.detect_all()
    assert out["ffmpeg_version"] and "libx264" in out["encoders"]["software"] and out["cpu"]["logical_cores"] >= 1
    assert all(isinstance(v, bool) for v in out["encoders"]["hardware"].values()) and all(isinstance(v, bool) for v in out["decoders"]["working"].values())
    assert set(out["decoders"]["working"]) <= set(out["decoders"]["hwaccels"])
    assert list(tmp_path.glob("agenttool hw probe*")) == []
    cpu = select_backend(settings(hardware_acceleration="cpu"), None, ff, s)
    assert cpu.encoder == "libx264" and cpu.kind == "cpu"
    auto = select_backend(settings(), None, ff, s)
    assert auto.encoder in ("libx264", *P.HARDWARE_ENCODERS["h264"])
    if not any(out["encoders"]["hardware"].values()):
        assert auto.kind == "cpu" and s.get_recommended_profile()["render_backend"] == "cpu"
        assert select_backend(settings(hardware_acceleration="hardware"), None, ff, s).outcome == "hardware_unavailable"
        # restart: the cache file prevents re-testing
        again = HardwareCapabilityService(FFmpegService(), cache_path=tmp_path / "hw.json", runner=lambda a, t: (_ for _ in ()).throw(AssertionError("must not probe")))
        assert again.detect_ffmpeg_encoders() == out["encoders"] and again.detect_hardware_decoders() == out["decoders"]
        pytest.skip("no hardware video encoder is present on this machine: hardware encode/decode tests skipped (software path verified above)")
    # a hardware encoder is present: its test encode must have really produced a result and the choice must name that encoder
    assert auto.hardware and auto.encoder in [k for k, v in out["encoders"]["hardware"].items() if v]


def test_default_runner_never_raises(tmp_path):
    from app.performance import hardware as H

    assert H._default_runner([str(tmp_path / "definitely missing tool")], 1)[0] == -1
    rc, _, err = H._default_runner([__import__("sys").executable, "-c", "import time; time.sleep(5)"], 0.3)
    assert rc == -2 and err == "timeout"
    rc, out, _ = H._default_runner([__import__("sys").executable, "-c", "print('héllo ü')"], 10)
    assert rc == 0 and "llo" in out
