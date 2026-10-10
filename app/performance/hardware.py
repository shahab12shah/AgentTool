"""HardwareCapabilityService: what this computer can do (CPU, memory, disk, GPU, FFmpeg encoders and decoders) and which render backend that justifies.

Everything is detected without administrator rights, every probe has a timeout and a failure means "unknown / not available" (never an exception),
and results are cached in memory and, optionally, in a small JSON file keyed by the FFmpeg binary so a restart does not repeat the test encodes.
Nothing here runs at startup: ``detect_all`` is meant for a background job and the cheap accessors never trigger the heavy tests unless asked.

``select_backend`` turns the export settings plus these facts into a ``BackendChoice``. Hardware is used only when it was *tested* to work, a
hardware request that cannot be honoured is an explicit outcome (with a CPU fallback offered) and nothing silently lowers quality.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from app.logging.logger import get_logger, log_event
from app.performance.resource_monitor import disk_usage, system_memory
from app.rendering import presets as P

_log = get_logger(__name__)
_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
GB = 1024**3
CACHE_SCHEMA = 1
CACHE_MAX_AGE_S = 30 * 86400  # a driver update or a new GPU is picked up at the latest after this long (or on "Re-test")
PROBE_TIMEOUT_S = 6.0
DECODE_TEST_TIMEOUT_S = 20.0
Runner = Callable[[list[str], float], tuple[int, str, str]]

# hwaccels worth testing, in the order they are preferred per platform; others FFmpeg lists (drm, opencl...) are not general-purpose decoders
DECODE_HWACCELS = ("cuda", "qsv", "d3d11va", "dxva2", "d3d12va", "vaapi", "videotoolbox", "vulkan", "vdpau")
DECODE_PREFERENCE = {"win": ("cuda", "qsv", "d3d11va", "dxva2"), "linux": ("cuda", "qsv", "vaapi"), "darwin": ("videotoolbox",)}
# the decode test uses H.264 8-bit 4:2:0, so "working" only vouches for sources of exactly that kind
DECODE_TESTED_CODECS = {"h264"}
DECODE_TESTED_PIX_FMTS = {"yuv420p", "yuvj420p"}
# frames are decoded to system memory (no -hwaccel_output_format), so these ordinary software filters see the same kind of frames as without hwaccel
HW_DECODE_FILTERS = {"color", "overlay", "scale", "fps", "trim", "setpts", "format", "crop", "rotate", "fade", "alphamerge", "geq", "blend", "tpad", "gblur", "colorchannelmixer", "subtitles", "ass",
                     "null", "split", "concat", "pad", "transpose", "vflip", "hflip", "setsar", "setdar", "drawtext"}
HW_FAILURE_MARKERS = ("failed setup for format", "hwaccel initialisation returned error", "device creation failed", "no device available", "failed to set value", "cannot load", "failed to initialise",
                      "failed to initialize", "no hardware", "unsupported hwaccel", "error initializing")
# conservative published encoder limits (long edge in pixels, fps). Over the limit => Auto uses the CPU; an explicit Hardware choice still tries and reports a failure honestly
HW_MAX_LONG_EDGE = {"h264": 4096, "h265": 8192, "av1": 8192}
HW_MAX_FPS = 120
VENDORS = {"nvidia": "nvidia", "intel": "intel", "amd": "amd", "advanced micro devices": "amd", "ati ": "amd", "apple": "apple"}


# ---------------------------------------------------------------------------------------------------- helpers
def _default_runner(args: list[str], timeout: float) -> tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, **_NO_WINDOW)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return -2, "", "timeout"
    except Exception as exc:  # noqa: BLE001 - a probe that cannot run is "unknown"
        return -1, "", str(exc)


def _vendor(*texts: str | None) -> str:
    low = " ".join(t for t in texts if t).lower() + " "
    for key, v in VENDORS.items():
        if key in low:
            return v
    if "radeon" in low or "geforce" in low or "quadro" in low:
        return "amd" if "radeon" in low else "nvidia"
    return "other"


def _as_list(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        return [obj]
    return [o for o in obj if isinstance(o, dict)] if isinstance(obj, list) else []


def _gb(n: int | None) -> str:
    return "unknown" if n is None else f"{n / GB:.1f} GB"


def _plat(system: str | None = None) -> str:
    s = (system or sys.platform).lower()
    return "win" if s.startswith("win") else "darwin" if s.startswith("darwin") else "linux"


# ---------------------------------------------------------------------------------------------------- backend choice
@dataclass
class BackendChoice:
    """What the renderer will use, and why. ``outcome``: ``ok`` (as requested), ``cpu_fallback`` (Auto chose the CPU; ``reason`` says why) or ``hardware_unavailable``
    (the user asked for Hardware and it cannot be used: ``fallback`` is the CPU choice to offer, the caller decides)."""

    kind: str = "cpu"  # cpu | nvenc | qsv | amf | videotoolbox
    encoder: str = ""
    decode_args: list[str] = field(default_factory=list)  # hardware decode options for the INPUTS; empty = software decoding
    quality_args: list[str] = field(default_factory=list)  # rate control in the vocabulary of this encoder
    notes: list[str] = field(default_factory=list)
    fallback: "BackendChoice | None" = None
    requested: str = "auto"
    reason: str = ""
    outcome: str = "ok"
    hwaccel: str = ""

    @property
    def hardware(self) -> bool:
        return self.kind != "cpu"

    @property
    def offers_cpu_fallback(self) -> bool:
        return self.fallback is not None and bool(self.fallback.encoder)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "encoder": self.encoder, "decode_args": list(self.decode_args), "quality_args": list(self.quality_args), "notes": list(self.notes), "requested": self.requested,
                "reason": self.reason, "outcome": self.outcome, "hwaccel": self.hwaccel, "fallback": self.fallback.to_dict() if self.fallback else None}


def _ffmpeg_caps_encoders(ffmpeg: Any) -> set[str]:
    try:
        return set(ffmpeg.capabilities().encoders)
    except Exception:  # noqa: BLE001
        return set()


def _tested_encoders(ffmpeg: Any, service: "HardwareCapabilityService | None", candidates: list[str]) -> dict[str, bool]:
    try:
        return dict(service.tested_hardware_encoders(candidates) if service is not None else ffmpeg.hardware_encoders(candidates))
    except Exception:  # noqa: BLE001 - a failing test is "not available"
        return {}


def _output_limits_problem(codec: str, size: tuple[int, int] | None, fps: int | None) -> str:
    limit = HW_MAX_LONG_EDGE.get(codec)
    if size and limit and max(size) > limit:
        return f"{P.CODEC_LABELS.get(codec, codec)} hardware encoders support up to {limit} pixels on the long edge; this output is {size[0]}×{size[1]}"
    if fps and fps > HW_MAX_FPS:
        return f"hardware encoders are only validated up to {HW_MAX_FPS} fps; this output is {fps} fps"
    return ""


def _auto_quality_problem(s: Any) -> str:
    """Why Auto must not use a hardware encoder for these quality settings ('' = fine)."""
    if s.quality == "maximum":
        return "Maximum quality needs the CPU encoder (a hardware encoder cannot reach the same quality per bit)"
    if s.quality == "custom" and s.crf and not s.bitrate_kbps:
        return f"CRF {s.crf} is a CPU-encoder quality scale with no exact hardware equivalent"
    return ""


def _decode_choice(s: Any, source_infos: Iterable[Any] | None, service: "HardwareCapabilityService | None", filters: Iterable[str] | None, purpose: str, allow_export: bool,
                   notes: list[str]) -> tuple[list[str], str]:
    """(decode args, hwaccel). Hardware decoding is offered only when it is tested, the sources are of the tested kind and every filter is a plain software filter."""
    if service is None or source_infos is None:
        return [], ""
    if purpose == "export" and not allow_export:
        notes.append("Final export decodes in software so every pixel is identical to a CPU render.")
        return [], ""
    infos = [i for i in source_infos if getattr(i, "has_video", True) and getattr(i, "kind", "video") == "video"]
    if not infos:
        return [], ""
    for i in infos:
        if str(getattr(i, "codec", "") or "").lower() not in DECODE_TESTED_CODECS or str(getattr(i, "pix_fmt", "") or "").lower() not in DECODE_TESTED_PIX_FMTS or getattr(i, "has_alpha", False):
            notes.append("Hardware decoding is not used: only 8-bit H.264 sources were validated for it.")
            return [], ""
    if filters is None:
        notes.append("Hardware decoding is not used: the filters on the path are not known.")
        return [], ""
    unsupported = sorted({f for f in filters if f not in HW_DECODE_FILTERS})
    if unsupported:
        notes.append("Hardware decoding is not used: unsupported filter(s) " + ", ".join(unsupported) + ".")
        return [], ""
    dec = service.detect_hardware_decoders()
    working = [h for h in DECODE_PREFERENCE[service.platform] if dec["working"].get(h)]
    if not working:
        return [], ""
    return ["-hwaccel", working[0]], working[0]


def select_backend(render_settings: Any, source_infos: Iterable[Any] | None, ffmpeg: Any, service: "HardwareCapabilityService | None" = None, *, force_cpu: bool = False,
                   output_size: tuple[int, int] | None = None, fps: int | None = None, filters: Iterable[str] | None = None, purpose: str = "export",
                   allow_hw_decode_export: bool = False) -> BackendChoice:
    """Decide the render backend from the export settings and what was *tested* on this machine.

    * CPU: the software encoder, always.
    * Hardware: the tested hardware encoder for the codec, or outcome ``hardware_unavailable`` with the CPU choice in ``fallback`` (nothing is switched silently).
    * Auto: hardware only when a tested encoder exists for the codec, the output is within known limits and the quality settings can be honoured; otherwise the CPU, with the reason.

    ``source_infos`` (ProbeInfo-like) and ``filters`` (names of the FFmpeg filters on the path) are only needed for hardware *decoding* (``decode_args``), which is never used for
    final export unless ``allow_hw_decode_export``; frames are always decoded to system memory so output pixels and formats do not change.
    """
    s = render_settings
    codec = s.video_codec
    requested = "cpu" if force_cpu else s.hardware_acceleration
    caps = _ffmpeg_caps_encoders(ffmpeg)
    soft = next((e for e in P.SOFTWARE_ENCODERS.get(codec, []) if e in caps), "")
    bitrate = s.bitrate_kbps if s.quality == "custom" else 0

    cpu = BackendChoice("cpu", soft, requested=requested)
    if soft:
        try:
            crf, preset = P.resolve_quality(s, soft, False)
            cpu.quality_args = P.ResolvedOutput(0, 0, fps or 30, s.container, codec, soft, False, s.audio_codec, "", 0, 0, s.quality, "yuv420p", crf, bitrate, preset).rate_args()
        except Exception:  # noqa: BLE001 - a custom/unknown quality is validated elsewhere; the choice itself still stands
            cpu.quality_args = []
    if requested == "cpu":
        cpu.reason = "cpu_requested" if s.hardware_acceleration == "cpu" else "cpu_requested_for_this_run"
        if cpu.reason != "cpu_requested" and service is not None and any(known.get(e) for e in P.HARDWARE_ENCODERS.get(codec, []) for known in [service.known_hardware_encoders()]):
            cpu.outcome = "cpu_fallback"  # e.g. the retry after a hardware failure: the user is told the CPU is used although hardware was available
        return cpu

    hw_names = P.HARDWARE_ENCODERS.get(codec, [])
    working = _tested_encoders(ffmpeg, service, hw_names) if hw_names else {}
    hw_enc = next((e for e in hw_names if working.get(e)), "")

    def unavailable(why: str, detail: str) -> BackendChoice:
        if requested == "hardware":
            return BackendChoice("cpu", "", [], [], [f"No working hardware encoder for {P.CODEC_LABELS.get(codec, codec)} ({detail})."], cpu, requested, why, "hardware_unavailable")
        cpu.outcome, cpu.reason = "cpu_fallback", why
        cpu.notes = ["Hardware encoding is not available here; using the CPU encoder."] if why == "no_working_hardware_encoder" else [f"Using the CPU encoder: {detail}."]
        return cpu

    if not hw_enc:
        return unavailable("no_working_hardware_encoder", "the GPU or its driver is not available to FFmpeg" if hw_names else "this codec has no hardware encoder in the supported list")
    limit = _output_limits_problem(codec, output_size, fps)
    notes: list[str] = []
    if requested == "auto":
        problem = _auto_quality_problem(s)
        if problem:
            return unavailable("quality_not_honoured", problem)
        if limit:
            return unavailable("output_beyond_hardware_limits", limit)
    elif limit:
        notes.append(f"Warning: {limit}; the encoder may refuse it.")
    crf, preset = P.resolve_quality(s, hw_enc, True)
    decode_args, hwaccel = _decode_choice(s, source_infos, service, filters, purpose, allow_hw_decode_export, notes)
    kind = P.hw_kind(hw_enc)
    notes.insert(0, f"Using the {kind.upper() if kind != 'videotoolbox' else 'VideoToolbox'} hardware encoder ({hw_enc}).")
    return BackendChoice(kind, hw_enc, decode_args, P.hw_rate_args(hw_enc, crf, bitrate, preset), notes, cpu if soft else None, requested, "hardware_tested_ok", "ok", hwaccel)


# ---------------------------------------------------------------------------------------------------- the service
class HardwareCapabilityService:
    def __init__(self, ffmpeg: Any | None = None, *, runner: Runner | None = None, monitor: Any | None = None, cache_path: Path | str | None = None, system: str | None = None,
                 temp_root: Path | str | None = None, cpuinfo_path: Path | str = "/proc/cpuinfo", drm_root: Path | str = "/sys/class/drm", disk_path: Path | str | None = None) -> None:
        self.ffmpeg = ffmpeg
        self._runner = runner or _default_runner
        self._monitor = monitor
        self._cache_path = Path(cache_path) if cache_path else None
        self.platform = _plat(system)
        self._temp_root = str(temp_root) if temp_root else None
        self._cpuinfo, self._drm = Path(cpuinfo_path), Path(drm_root)
        self._disk_path = Path(disk_path) if disk_path else None
        self._lock = threading.RLock()
        self._detect_lock = threading.Lock()  # one heavy detection at a time (a second caller waits for the first and then hits the cache)
        self._cpu: dict | None = None
        self._gpu: list[dict] | None = None
        self._enc: dict | None = None
        self._dec: dict | None = None
        self._tested_at: float | None = None
        self._file_loaded = False

    # ------------------------------------------------------------------ plumbing
    def _run(self, args: list[str], timeout: float = PROBE_TIMEOUT_S) -> tuple[int, str, str]:
        try:
            r = self._runner(args, timeout)
            rc, out, err = r
            return int(rc), out if isinstance(out, str) else "", err if isinstance(err, str) else ""
        except Exception as exc:  # noqa: BLE001 - an injected or real runner must never break the caller
            return -1, "", str(exc)

    def _exe(self) -> str | None:
        if self.ffmpeg is None:
            return None
        try:
            return str(self.ffmpeg.ffmpeg())
        except Exception:  # noqa: BLE001 - FFmpeg not installed
            return None

    def ffmpeg_version(self) -> str | None:
        try:
            return str(self.ffmpeg.version().text) if self.ffmpeg is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _cache_key(self) -> dict | None:
        exe = self._exe()
        if not exe:
            return None
        try:
            st = os.stat(exe)
            mtime, size = st.st_mtime_ns, st.st_size
        except OSError:
            mtime = size = 0
        return {"path": exe, "version": self.ffmpeg_version() or "", "mtime_ns": mtime, "size": size, "platform": self.platform}

    # ------------------------------------------------------------------ on-disk cache
    def _load_file(self) -> None:
        if self._file_loaded:
            return
        self._file_loaded = True
        if not self._cache_path:
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            key = self._cache_key()
            if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA or key is None or data.get("key") != key:
                return
            at = float(data.get("tested_at", 0))
            if time.time() - at > CACHE_MAX_AGE_S or at > time.time() + 86400:
                return
            enc, dec, gpu = data.get("encoders"), data.get("decoders"), data.get("gpu")
            with self._lock:
                if isinstance(enc, dict) and isinstance(enc.get("software"), list) and isinstance(enc.get("hardware"), dict):
                    self._enc = {"software": [str(x) for x in enc["software"]], "hardware": {str(k): bool(v) for k, v in enc["hardware"].items()}}
                    if self.ffmpeg is not None and hasattr(self.ffmpeg, "seed_hardware_encoders"):
                        self.ffmpeg.seed_hardware_encoders(self._enc["hardware"])
                if isinstance(dec, dict) and isinstance(dec.get("hwaccels"), list) and isinstance(dec.get("working"), dict):
                    self._dec = {"hwaccels": [str(x) for x in dec["hwaccels"]], "working": {str(k): bool(v) for k, v in dec["working"].items()}}
                if isinstance(gpu, list):
                    self._gpu = [g for g in gpu if isinstance(g, dict) and "vendor" in g and "model" in g]
                self._tested_at = at
        except (OSError, ValueError, TypeError):
            return  # missing or corrupt file: detect again

    def _save_file(self) -> None:
        if not self._cache_path:
            return
        key = self._cache_key()
        if key is None or self._enc is None:
            return
        payload = {"schema": CACHE_SCHEMA, "key": key, "tested_at": self._tested_at or time.time(), "gpu": self._gpu, "encoders": self._enc, "decoders": self._dec}
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_name(self._cache_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._cache_path)
        except OSError:
            _log.debug("hardware cache could not be written", exc_info=True)

    # ------------------------------------------------------------------ CPU / memory / disk
    def detect_cpu(self) -> dict:
        with self._lock:
            if self._cpu is not None:
                return dict(self._cpu)
        logical = os.cpu_count() or 1
        physical: int | None = None
        model: str | None = None
        try:
            if self.platform == "linux":
                text = self._cpuinfo.read_text(encoding="utf-8", errors="replace")
                cores: set[tuple[str, str]] = set()
                pid = ""
                for line in text.splitlines():
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k == "model name" and not model:
                        model = v or None
                    elif k == "physical id":
                        pid = v
                    elif k == "core id":
                        cores.add((pid, v))
                physical = len(cores) or None
            elif self.platform == "win":
                rc, out, _ = self._run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores | ConvertTo-Json -Compress"], 10)
                rows = _as_list(json.loads(out)) if rc == 0 and out.strip() else []
                if rows:
                    model = str(rows[0].get("Name") or "").strip() or None
                    n = sum(int(r.get("NumberOfCores") or 0) for r in rows)
                    physical = n or None
            else:
                rc, out, _ = self._run(["sysctl", "-n", "machdep.cpu.brand_string"])
                model = out.strip() or None if rc == 0 else None
                rc, out, _ = self._run(["sysctl", "-n", "hw.physicalcpu"])
                physical = int(out.strip()) if rc == 0 and out.strip().isdigit() else None
        except Exception:  # noqa: BLE001 - unknown stays None
            pass
        model = model or platform.processor() or None
        if physical is not None and not (1 <= physical <= logical):
            physical = None
        res = {"arch": platform.machine() or "unknown", "logical_cores": logical, "physical_cores": physical, "model": model}
        with self._lock:
            self._cpu = res
        return dict(res)

    def detect_memory(self) -> dict:
        total = avail = None
        try:
            if self._monitor is not None:
                s = self._monitor.sample()
                total, avail = s.system_total_bytes, s.system_available_bytes
            else:
                total, avail = system_memory()
        except Exception:  # noqa: BLE001
            total = avail = None
        return {"total_bytes": total, "available_bytes": avail}

    def detect_disk(self, path: Path | str | None = None) -> dict:
        p = Path(path) if path else self._disk_path
        try:
            if p is None and self._monitor is not None:
                s = self._monitor.sample()
                return {"free": s.disk_free_bytes, "total": s.disk_total_bytes}
            free, total = disk_usage(p or Path.home())
        except Exception:  # noqa: BLE001
            free = total = None
        return {"free": free, "total": total}

    # ------------------------------------------------------------------ GPU
    def detect_gpu(self) -> list[dict]:
        self._load_file()
        with self._lock:
            if self._gpu is not None:
                return [dict(g) for g in self._gpu]
        try:
            gpus = {"win": self._gpu_windows, "darwin": self._gpu_macos, "linux": self._gpu_linux}[self.platform]()
        except Exception:  # noqa: BLE001 - garbage output means "no information"
            gpus = []
        with self._lock:
            self._gpu = gpus
        return [dict(g) for g in gpus]

    def _gpu_linux(self) -> list[dict]:
        gpus: list[dict] = []
        rc, out, _ = self._run(["lspci", "-mm"])
        if rc == 0:
            for line in out.splitlines():
                parts = re.findall(r'"([^"]*)"', line)
                if len(parts) >= 3 and any(k in parts[0].lower() for k in ("vga", "3d controller", "display controller")):
                    gpus.append({"vendor": _vendor(parts[1]), "model": f"{parts[1]} {parts[2]}".strip(), "driver": None, "source": "lspci"})
        smi: list[dict] = []
        rc, out, _ = self._run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if rc == 0:
            for row in csv.reader(io.StringIO(out)):
                if len(row) >= 2 and row[0].strip().isprintable() and re.fullmatch(r"\d+(\.\d+)*", row[1].strip()):  # "name, 535.54"; error text and garbage are not a GPU
                    smi.append({"vendor": "nvidia", "model": row[0].strip(), "driver": row[1].strip(), "source": "nvidia-smi"})
        if smi:  # the driver tool is authoritative for NVIDIA boards (exact model and driver version)
            gpus = [g for g in gpus if g["vendor"] != "nvidia"] + smi
        if not gpus:
            ids = {"0x10de": "nvidia", "0x8086": "intel", "0x1002": "amd"}
            try:
                for card in sorted(self._drm.glob("card*")):
                    if "-" in card.name:
                        continue
                    v = ids.get((card / "device" / "vendor").read_text(encoding="utf-8", errors="replace").strip().lower())
                    if v:
                        gpus.append({"vendor": v, "model": f"{v.upper() if v != 'nvidia' else 'NVIDIA'} graphics (model unknown)", "driver": None, "source": "sysfs"})
            except OSError:
                pass
        return gpus

    def _gpu_windows(self) -> list[dict]:
        cmd = "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility,DriverVersion | ConvertTo-Json -Compress"
        rc, out, _ = self._run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], 12)
        gpus: list[dict] = []
        if rc == 0 and out.strip():
            for r in _as_list(json.loads(out.strip().lstrip("﻿"))):
                name = str(r.get("Name") or "").strip()
                if name:
                    gpus.append({"vendor": _vendor(str(r.get("AdapterCompatibility") or ""), name), "model": name, "driver": str(r.get("DriverVersion") or "").strip() or None, "source": "powershell"})
        if not gpus:
            rc, out, _ = self._run(["wmic", "path", "win32_videocontroller", "get", "name,adapterCompatibility,driverversion", "/format:csv"], 12)
            if rc == 0:
                rows = [r for r in csv.DictReader(io.StringIO(out.replace("\r", "").lstrip("﻿").strip()))]
                for r in rows:
                    name = (r.get("Name") or "").strip()
                    if name:
                        gpus.append({"vendor": _vendor(r.get("AdapterCompatibility"), name), "model": name, "driver": (r.get("DriverVersion") or "").strip() or None, "source": "wmic"})
        return gpus

    def _gpu_macos(self) -> list[dict]:
        rc, out, _ = self._run(["system_profiler", "SPDisplaysDataType", "-json"], 12)
        gpus: list[dict] = []
        if rc == 0 and out.strip():
            for r in _as_list(json.loads(out).get("SPDisplaysDataType", [])):
                name = str(r.get("sppci_model") or r.get("_name") or "").strip()
                if name:
                    gpus.append({"vendor": _vendor(str(r.get("spdisplays_vendor") or "").replace("sppci_vendor_", ""), name), "model": name, "driver": None, "source": "system_profiler"})
        return gpus

    # ------------------------------------------------------------------ FFmpeg encoders / decoders
    def detect_ffmpeg_encoders(self) -> dict:
        self._load_file()
        with self._lock:
            if self._enc is not None:
                return {"software": list(self._enc["software"]), "hardware": dict(self._enc["hardware"])}
        software: list[str] = []
        hardware: dict[str, bool] = {}
        if self.ffmpeg is not None:
            try:
                encs = set(self.ffmpeg.capabilities().encoders)
                software = sorted(e for names in P.SOFTWARE_ENCODERS.values() for e in names if e in encs)
                hardware = {str(k): bool(v) for k, v in self.ffmpeg.hardware_encoders().items()}
            except Exception:  # noqa: BLE001
                software, hardware = [], {}
        with self._lock:
            self._enc = {"software": software, "hardware": hardware}
            self._tested_at = self._tested_at or time.time()
        self._save_file()
        return {"software": list(software), "hardware": dict(hardware)}

    def known_hardware_encoders(self) -> dict[str, bool]:
        """Hardware encoder test results already known (memory / cache file); never runs a test."""
        return self._cached_enc()["hardware"]

    def tested_hardware_encoders(self, candidates: list[str]) -> dict[str, bool]:
        """``{encoder: works}`` for the candidates that exist in the FFmpeg build (same contract as ``FFmpegService.hardware_encoders``), using the on-disk cache after a restart."""
        if self.ffmpeg is None:
            return {}
        self._load_file()
        res = self.ffmpeg.hardware_encoders(candidates)
        with self._lock:
            known = (self._enc or {}).get("hardware", {})
            changed = any(known.get(k) != v for k, v in res.items())
            if changed:
                hw = dict(known)
                hw.update(res)
                self._enc = {"software": (self._enc or {}).get("software", []), "hardware": hw}
                self._tested_at = self._tested_at or time.time()
        if changed:
            self._save_file()
        return res

    def detect_hardware_decoders(self, cancel: Any | None = None) -> dict:
        self._load_file()
        with self._lock:
            if self._dec is not None:
                return {"hwaccels": list(self._dec["hwaccels"]), "working": dict(self._dec["working"])}
        exe = self._exe()
        hwaccels: list[str] = []
        working: dict[str, bool] = {}
        if exe:
            rc, out, _ = self._run([exe, "-hide_banner", "-hwaccels"], 10)
            if rc == 0:
                hwaccels = [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.strip().lower().startswith("hardware acceleration")]
            todo = [h for h in DECODE_HWACCELS if h in hwaccels]
            if todo:
                working = self._test_decoders(exe, todo, cancel)
                if cancel is not None and cancel.is_set():
                    return {"hwaccels": hwaccels, "working": working}  # partial: not cached
        with self._lock:
            self._dec = {"hwaccels": hwaccels, "working": working}
            self._tested_at = self._tested_at or time.time()
        self._save_file()
        return {"hwaccels": list(hwaccels), "working": dict(working)}

    def _test_decoders(self, exe: str, names: list[str], cancel: Any | None) -> dict[str, bool]:
        """A tiny generated H.264 clip is really decoded with each hwaccel; 'listed' is not 'works'."""
        res = {n: False for n in names}
        tmp: str | None = None
        try:
            tmp = tempfile.mkdtemp(prefix="agenttool hw probe ", dir=self._temp_root)  # spaces / non-ASCII anywhere in the path are fine: arguments are never shell-parsed
            clip = str(Path(tmp) / "probe clip.mp4")
            rc, _, _ = self._run([exe, "-y", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=10:d=0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], DECODE_TEST_TIMEOUT_S)
            if rc != 0:
                return res
            for n in names:
                if cancel is not None and cancel.is_set():
                    break
                rc, out, err = self._run([exe, "-hide_banner", "-v", "error", "-hwaccel", n, "-i", clip, "-f", "null", "-"], DECODE_TEST_TIMEOUT_S)
                text = (out + err).lower()
                res[n] = rc == 0 and not any(m in text for m in HW_FAILURE_MARKERS)  # FFmpeg can fall back to software and still exit 0; its error lines say so
        except OSError:
            pass
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        return res

    # ------------------------------------------------------------------ whole detection / recommendation
    def detect_all(self, progress: Callable[[float, str], None] | None = None, cancel: Any | None = None) -> dict:
        """Run every probe (the slow ones included). Meant for a background job: reports progress, stops early when ``cancel.is_set()`` and caches only complete results."""
        def step(frac: float, msg: str) -> bool:
            if progress:
                try:
                    progress(frac, msg)
                except Exception:  # noqa: BLE001
                    pass
            return bool(cancel is not None and cancel.is_set())

        with self._detect_lock:
            t0 = time.monotonic()
            for frac, msg, fn in ((0.05, "Checking CPU, memory and disk", lambda: (self.detect_cpu(), self.detect_memory(), self.detect_disk())), (0.15, "Looking for graphics hardware", self.detect_gpu),
                                  (0.4, "Testing hardware video encoders", self.detect_ffmpeg_encoders), (0.8, "Testing hardware video decoding", lambda: self.detect_hardware_decoders(cancel))):
                if step(frac, msg):
                    break
                fn()
            else:
                step(1.0, "Hardware check finished")
            out = self.summary(allow_detect=False)
            if not (cancel is not None and cancel.is_set()):
                log_event(_log, "hardware.detected", seconds=round(time.monotonic() - t0, 2), gpus=len(self._gpu or []), hardware_encoders=sorted(k for k, v in (self._enc or {}).get("hardware", {}).items() if v),
                          hwaccels=sorted(k for k, v in (self._dec or {}).get("working", {}).items() if v))
            return out

    def get_recommended_profile(self, path: Path | str | None = None, allow_detect: bool = True) -> dict:
        cpu = self.detect_cpu()
        mem = self.detect_memory()
        disk = self.detect_disk(path)
        enc = self.detect_ffmpeg_encoders() if allow_detect else self._cached_enc()
        cores = cpu["logical_cores"]
        ram, free = mem["total_bytes"], disk["free"]
        reasons: list[str] = []
        level = 1  # 0 power_saver, 1 balanced, 2 performance
        if cores <= 2:
            level = 0
            reasons.append(f"Only {cores} CPU thread(s): background work is kept light.")
        elif cores >= 8:
            level = 2
        if ram is None:
            level = min(level, 1)
            reasons.append("Memory size could not be read: assuming a modest computer.")
        elif ram < 4 * GB:
            level = 0
            reasons.append(f"Low memory ({_gb(ram)}): background work and caches are kept small.")
        elif ram < 12 * GB:
            level = min(level, 1)
            reasons.append(f"{_gb(ram)} of memory: balanced settings.")
        if free is not None and free < 5 * GB:
            level = 0
            reasons.append(f"Low free disk space ({_gb(free)}): large caches and proxies are avoided.")
        elif free is not None and free < 20 * GB:
            level = min(level, 1)
            reasons.append(f"Limited free disk space ({_gb(free)}).")
        elif free is None:
            level = min(level, 1)
        profile = ("power_saver", "balanced", "performance")[level]
        proxy = "off" if free is not None and free < 5 * GB else "automatic" if level == 2 and free is not None and free >= 50 * GB else "manual"
        hw_ok = sorted(k for k, v in enc["hardware"].items() if v)
        backend = "auto" if hw_ok else "cpu"
        reasons.append(f"Hardware encoders that passed a test: {', '.join(hw_ok)}. Auto uses them when the export settings allow." if hw_ok else "No hardware video encoder passed a test: exports use the CPU.")
        if level == 2:
            reasons.append(f"{cores} CPU threads and {_gb(ram)} of memory: Performance is comfortable.")
        return {"profile": profile, "render_backend": backend, "proxy_policy": proxy, "reasons": reasons}

    def _cached_enc(self) -> dict:
        self._load_file()
        with self._lock:
            return {"software": list((self._enc or {}).get("software", [])), "hardware": dict((self._enc or {}).get("hardware", {}))}

    def summary(self, allow_detect: bool = True) -> dict:
        """Everything in one dict. ``allow_detect=False`` only reports what is already known (cheap, never runs the slow tests): missing parts are empty and listed in ``pending``."""
        self._load_file()
        pending: list[str] = []
        if allow_detect:
            cpu, gpu, enc, dec = self.detect_cpu(), self.detect_gpu(), self.detect_ffmpeg_encoders(), self.detect_hardware_decoders()
        else:
            cpu = self.detect_cpu()
            with self._lock:
                gpu, dec = [dict(g) for g in (self._gpu or [])], {"hwaccels": list((self._dec or {}).get("hwaccels", [])), "working": dict((self._dec or {}).get("working", {}))}
            enc = self._cached_enc()
            pending = [n for n, v in (("gpu", self._gpu), ("encoders", self._enc), ("decoders", self._dec)) if v is None]
        mem = self.detect_memory()
        tested = datetime.fromtimestamp(self._tested_at).isoformat(timespec="seconds") if self._tested_at else None
        cores = f"{cpu['physical_cores']} cores / {cpu['logical_cores']} threads" if cpu["physical_cores"] else f"{cpu['logical_cores']} threads"
        lines = [f"CPU: {cpu['model'] or 'unknown'} ({cores}, {cpu['arch']})", f"Memory: {_gb(mem['total_bytes'])} total, {_gb(mem['available_bytes'])} available"]
        lines.append("Graphics: " + ("; ".join(g["model"] + (f" (driver {g['driver']})" if g.get("driver") else "") for g in gpu) if gpu else ("not detected yet" if "gpu" in pending else "none detected")))
        hw_ok = sorted(k for k, v in enc["hardware"].items() if v)
        hw_bad = sorted(k for k, v in enc["hardware"].items() if not v)
        lines.append("Software video encoders: " + (", ".join(enc["software"]) or ("not detected yet" if "encoders" in pending else "none")))
        lines.append("Hardware video encoders that work: " + (", ".join(hw_ok) or "none") + (f" (listed but failed the test: {', '.join(hw_bad)})" if hw_bad else ""))
        dec_ok = sorted(k for k, v in dec["working"].items() if v)
        dec_bad = [h for h in dec["hwaccels"] if h in dec["working"] and not dec["working"][h]]
        lines.append("Hardware video decoding that works: " + (", ".join(dec_ok) or "none") + (f" (listed but not working: {', '.join(dec_bad)})" if dec_bad else ""))
        lines.append("FFmpeg: " + (self.ffmpeg_version() or "not found"))
        return {"text": "\n".join(lines), "cpu": cpu, "memory": mem, "gpu": gpu, "encoders": enc, "decoders": dec, "ffmpeg_version": self.ffmpeg_version(), "tested_at": tested, "pending": pending}

    # ------------------------------------------------------------------ reset
    def forget(self) -> None:
        """Drop everything detected (memory and the cache file) and FFmpeg's own cached detection (after the FFmpeg path or the hardware changed)."""
        with self._lock:
            self._cpu = self._gpu = self._enc = self._dec = None
            self._tested_at = None
            self._file_loaded = True  # do not re-read the file we are about to delete
        if self._cache_path:
            try:
                self._cache_path.unlink()
            except OSError:
                pass
        if self.ffmpeg is not None and hasattr(self.ffmpeg, "forget"):
            try:
                self.ffmpeg.forget()
            except Exception:  # noqa: BLE001
                pass

    def refresh(self, progress: Callable[[float, str], None] | None = None, cancel: Any | None = None) -> dict:
        self.forget()
        return self.detect_all(progress, cancel)
