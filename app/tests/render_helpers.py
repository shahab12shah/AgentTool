"""Phase 6 test helpers: a small but complete timeline (video, B-roll, image, text, captions, voice, music, SFX) and a one-call render."""

from __future__ import annotations

import threading
from pathlib import Path

from app.presentation.animation import spec
from app.rendering.executor import JobSpec
from app.rendering.models import RenderSnapshot
from app.tests.helpers import make_image, make_video, write_tone_wav
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT, Clip
from app.timeline.keyframes import Keyframe
from app.timeline.timeline import new_clip_id


def put(ws, track_id: str, clip: Clip) -> Clip:
    tl = ws.project.timeline
    clip.track_id = track_id
    t = tl.get_track(track_id)
    t.clips.append(clip)
    t.sort()
    return clip


def media_clip(asset, start: float, duration: float, **kw) -> Clip:
    return Clip(new_clip_id(), "", asset.id, start, duration, source_in=kw.pop("source_in", 0.0), source_out=kw.pop("source_in_out", 0.0) or duration, **kw)


def caption_clip(start: float, words: list[str], dt: float = 0.3, **kw) -> Clip:
    ws_ = [{"word_id": f"w{i}", "text": w, "start": start + i * dt, "end": start + i * dt + dt * 0.9} for i, w in enumerate(words)]
    half = (len(words) + 1) // 2
    lines = [" ".join(words[:half]), " ".join(words[half:])] if len(words) > 3 else [" ".join(words)]
    end = start + len(words) * dt
    seg = {"caption_id": "cap_" + words[0].lower(), "scene_id": "", "start": start, "end": end, "text": " ".join(words), "lines": [x for x in lines if x], "words": ws_,
           "emphasis": [{"word_index": len(words) - 1, "category": "NUMBER", "style": "COLOR_CHANGE", "reason": ""}], "style_id": "professional", "style_overrides": {},
           "position": "bottom", "position_xy": [], "highlight_mode": "HIGHLIGHT", "reading_cps": 12.0}
    return Clip(new_clip_id(), "track_v6", "", start, end - start, kind=KIND_CAPTION, text=seg, animation={"in": spec("fade_in", duration=0.2), "out": spec("fade_out", duration=0.2)}, **kw)


def text_clip(start: float, duration: float, content: str, style: str = "HEADLINE", pos=(0.5, 0.3), size: int = 72, counter=None, anim=None, **kw) -> Clip:
    t = {"text_id": "t_" + content[:4], "content": content, "start": start, "duration": duration, "position": list(pos), "style": style, "emphasis": "", "animation": "fade",
         "font": "Sans", "size": size, "alignment": "center", "opacity": 1.0, "background": "none", "variant": style, "counter": counter}
    return Clip(new_clip_id(), "track_v5", "", start, duration, kind=KIND_TEXT, text=t, animation=anim if anim is not None else {"in": spec("fade_in"), "out": spec("fade_out")}, **kw)


def highlight_clip(start: float, duration: float, region=(0.2, 0.3, 0.5, 0.25), dim: bool = True) -> Clip:
    return Clip(new_clip_id(), "track_v4", "", start, duration, kind=KIND_GRAPHIC, effects={"highlight": {"region": list(region), "style": "box", "darken_surround": dim}},
                animation={"in": spec("reveal", duration=0.4), "out": spec("fade_out", duration=0.3)})


class Demo:
    """Everything a render test needs to know about the demo project."""

    def __init__(self) -> None:
        self.assets: dict[str, object] = {}
        self.duration = 8.0


def build_demo(ws, tmp_path: Path, *, seconds: float = 8.0, with_audio: bool = True, big_video: tuple[int, int] | None = None, wait=None) -> Demo:
    """V1 two video clips (second one fades in and slowly zooms), V2 B-roll at 80% opacity, V3 a rotated image with keyframed opacity, V4 an evidence box, V5 headline +
    counter, V6 captions, A1 voice-over, A2 music with ducking keyframes, A3 a sound effect."""
    d = tmp_path / "demo_media"
    d.mkdir(exist_ok=True)
    demo = Demo()
    demo.duration = seconds
    files = [make_video(d / "main.mp4", 6.0), make_video(d / "broll.mp4", 4.0, "testsrc2"), make_image(d / "still.png", "testsrc", "800x600")]
    if big_video:
        files.append(make_video(d / "big.mp4", 3.0, "testsrc2", f"{big_video[0]}x{big_video[1]}"))
    if with_audio:
        files += [write_tone_wav(d / "voice.wav", seconds, 0.3, 220.0, 48000), write_tone_wav(d / "bed.wav", seconds + 2, 0.4, 110.0, 44100), write_tone_wav(d / "hit.wav", 0.5, 0.5, 880.0, 48000)]
    ws.media.import_files(files)
    if wait is not None:
        wait()  # a UI test pumps the Qt event loop here (job callbacks are delivered on the UI thread)
    else:
        assert ws.jobs.wait_idle(60)
    by = {a.name: a for a in ws.project.assets.all()}
    demo.assets = by
    h = seconds / 2
    c1 = put(ws, "track_v1", media_clip(by["main.mp4"], 0.0, h))
    c2 = put(ws, "track_v1", media_clip(by["main.mp4"], h, h, source_in=1.0))
    c2.transition = {"type": "FADE", "duration": 0.5}
    c2.keyframes = [Keyframe("scale", 0.0, 1.0, "ease_in_out"), Keyframe("scale", h, 1.3, "linear")]
    put(ws, "track_v2", media_clip(by["broll.mp4"], 1.0, 3.0, opacity=0.8, position=(120.0, -60.0), scale=0.5))
    img = put(ws, "track_v3", media_clip(by["still.png"], 4.5, 2.5, rotation=10.0, scale=0.4))
    img.keyframes = [Keyframe("opacity", 0.0, 0.0), Keyframe("opacity", 1.0, 1.0), Keyframe("opacity", 2.5, 0.3)]
    put(ws, "track_v4", highlight_clip(1.0, 2.0))
    put(ws, "track_v5", text_clip(1.0, 2.0, "SILVER DEMAND"))
    put(ws, "track_v5", text_clip(4.0, 2.5, "$1,500", "NUMBER_CARD", (0.5, 0.5), 96, counter={"from": 0.0, "to": 1500.0, "decimals": 0, "prefix": "$", "suffix": "", "thousands": True},
                                  anim={"in": spec("counter", duration=0.9), "out": spec("fade_out")}))
    put(ws, "track_v6", caption_clip(0.5, "Silver is running out of supply".split()))
    put(ws, "track_v6", caption_clip(4.2, "The price could hit 100 dollars".split()))
    if with_audio:
        put(ws, "track_a1", media_clip(by["voice.wav"], 0.0, seconds, audio={"role": "VOICE", "volume": 1.0}))
        music = media_clip(by["bed.wav"], 0.0, seconds, audio={"role": "MUSIC", "volume": 1.0, "fade_in": 0.5, "fade_out": 1.0})
        music.keyframes = [Keyframe("volume", 0.0, 0.3), Keyframe("volume", 1.0, 0.1, "ease_out"), Keyframe("volume", 5.0, 0.1), Keyframe("volume", 6.0, 0.3, "ease_in")]
        put(ws, "track_a2", music)
        put(ws, "track_a3", media_clip(by["hit.wav"], 2.0, 0.5, audio={"role": "SFX", "volume": 0.6}))
    if with_audio:
        ws.project.voice_over.asset_id = by["voice.wav"].id  # the project's voice-over (what a real project has after importing narration)
        ws.project.voice_over.duration = seconds
    ws.project.timeline_version = 1
    return demo


def make_spec(engine, ws, tmp_path: Path, name: str = "out", **overrides) -> tuple[JobSpec, threading.Event, list]:
    from dataclasses import replace

    project = ws.project
    settings = replace(project.render_settings, **{"resolution": "480p", "quality": "draft", **overrides})
    snap = RenderSnapshot.from_project(project, settings)
    resolved = engine.resolve(settings, snap)
    root = project.root
    spec_ = JobSpec("render_" + name, snap, resolved, tmp_path / "out" / f"{name}.{resolved.container}", root / "cache" / "render" / ("render_" + name), root / "cache" / "render",
                    root / "renders" / ("render_" + name))
    return spec_, threading.Event(), []


# ---------------------------------------------------------------------------------------------- pixel / audio probes
import subprocess  # noqa: E402

import numpy as np  # noqa: E402


def solid_image(path: Path, color: str = "red", size: str = "320x180") -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s={size}", "-frames:v", "1", str(path)], check=True)
    return path


def two_color_image(path: Path, left: str = "red", right: str = "blue", size: int = 100) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={left}:s={size}x{size}", "-f", "lavfi", "-i", f"color=c={right}:s={size}x{size}",
                    "-filter_complex", "hstack", "-frames:v", "1", str(path)], check=True)
    return path


def half_transparent_png(path: Path) -> Path:
    from PIL import Image

    im = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    for x in range(100, 200):
        for y in range(100):
            im.putpixel((x, y), (0, 0, 0, 0))
    im.save(path)
    return path


def color_video(path: Path, colors=("red", "green", "blue"), each: float = 1.0, size: str = "320x180", rate: int = 24, extra: list[str] | None = None) -> Path:
    """A video that shows ``colors`` one after another (each for ``each`` seconds): handy for checking trims, speed, cuts and frame accuracy."""
    inputs, labels = [], []
    for i, c in enumerate(colors):
        inputs += ["-f", "lavfi", "-i", f"color=c={c}:s={size}:r={rate}:d={each}"]
        labels.append(f"[{i}:v]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", "".join(labels) + f"concat=n={len(colors)}:v=1:a=0", "-pix_fmt", "yuv420p", *(extra or []), str(path)], check=True)
    return path


def frame_at(path: Path, t: float | None = None, n: int | None = None):
    """One decoded frame as a PIL image: by time or by frame number."""
    from io import BytesIO

    from PIL import Image

    if n is not None:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    return Image.open(BytesIO(data)).convert("RGB")


def px(img, fx: float, fy: float) -> tuple[int, int, int]:
    """Colour at a relative position (0..1) of the frame."""
    return img.getpixel((min(img.width - 1, int(fx * img.width)), min(img.height - 1, int(fy * img.height))))


def near(c, target, tol: int = 40) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(c, target))


RED, GREEN, BLUE, BLACK, WHITE = (255, 0, 0), (0, 128, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)  # FFmpeg's "green" is #008000


def run_row(img, fy: float, color, tol: int = 60) -> int:
    """How many pixels along a row are close to ``color`` (measures sizes of shapes)."""
    y = min(img.height - 1, int(fy * img.height))
    return sum(1 for x in range(img.width) if near(img.getpixel((x, y)), color, tol))


def run_col(img, fx: float, color, tol: int = 60) -> int:
    x = min(img.width - 1, int(fx * img.width))
    return sum(1 for y in range(img.height) if near(img.getpixel((x, y)), color, tol))


def decode_audio(path: Path, sr: int = 16000) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def rms(a: np.ndarray, t0: float, t1: float, sr: int = 16000) -> float:
    seg = a[int(t0 * sr):int(t1 * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def render_to(engine, ws, tmp_path: Path, name: str, **overrides):
    """Render the current project with the engine directly (no queue) and return (result, spec)."""
    spec_, cancel, prog = make_spec(engine, ws, tmp_path, name, **overrides)
    res = engine.run(spec_, cancel, lambda p: prog.append(p))
    return res, spec_


def span_row(img, fy: float, color, around: float = 0.5, tol: int = 60) -> tuple[int, int] | None:
    """(first, last) pixel of the continuous run of ``color`` through the point (around, fy): the size of one shape, ignoring other shapes in the row."""
    y = min(img.height - 1, int(fy * img.height))
    x = min(img.width - 1, int(around * img.width))
    if not near(img.getpixel((x, y)), color, tol):
        return None
    a = b = x
    while a > 0 and near(img.getpixel((a - 1, y)), color, tol):
        a -= 1
    while b < img.width - 1 and near(img.getpixel((b + 1, y)), color, tol):
        b += 1
    return a, b


def span_col(img, fx: float, color, around: float = 0.5, tol: int = 60) -> tuple[int, int] | None:
    x = min(img.width - 1, int(fx * img.width))
    y = min(img.height - 1, int(around * img.height))
    if not near(img.getpixel((x, y)), color, tol):
        return None
    a = b = y
    while a > 0 and near(img.getpixel((x, a - 1)), color, tol):
        a -= 1
    while b < img.height - 1 and near(img.getpixel((x, b + 1)), color, tol):
        b += 1
    return a, b


def install_fake_ffmpeg(ws, tmp_path: Path, mode: str = "ok") -> Path:
    """Point the workspace at the fake ffmpeg wrapper; returns the control file (write a mode into it to change behaviour between runs)."""
    import os
    import shutil
    import stat
    import sys

    from app.core.config import Settings

    ctl = tmp_path / "fake_mode.txt"
    ctl.write_text(mode)
    wrapper = tmp_path / "fake_ffmpeg"
    src = Path(__file__).with_name("fake_ffmpeg.py")
    wrapper.write_text(f"#!{sys.executable}\n" + src.read_text().split("\n", 1)[1])
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.environ["FAKE_FFMPEG_REAL"] = shutil.which("ffmpeg") or "ffmpeg"
    os.environ["FAKE_FFMPEG_MODE"] = str(ctl)
    ws.update_settings(Settings(**{**ws.settings.__dict__, "ffmpeg_path": str(wrapper)}))
    return ctl


def wait_job(job, timeout: float = 120.0):
    assert job.finished_event.wait(timeout), f"job still {job.status}"
    return job
