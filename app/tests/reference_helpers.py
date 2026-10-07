"""Synthetic reference media for Phase 7 tests: videos with *known* editing (cuts, fades, dissolves, zooms, pans, captions, text) and audio with known
structure (voice, music, SFX, ducking, silence). Because the truth is known, detector tests can assert real numbers.

Nothing here is a copy of any real video: shots are random smooth "blob" pictures drawn from a seed.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SIZE = (320, 180)


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


# ---------------------------------------------------------------------------------------------- pictures
def blob_image(path: Path, seed: int, size: tuple[int, int] = SIZE, boxes: int = 3) -> Path:
    """A smooth random picture (distinct for every seed): low-frequency colour blobs plus a few hard-edged rectangles."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    small = rng.random((9, 16, 3)) * 255
    im = Image.fromarray(small.astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)
    d = ImageDraw.Draw(im)
    for _ in range(boxes):
        x0, y0 = int(rng.integers(0, size[0] - 60)), int(rng.integers(0, size[1] - 40))
        d.rectangle([x0, y0, x0 + int(rng.integers(30, 80)), y0 + int(rng.integers(20, 50))], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    im.save(path)
    return path


XFADE = {"fade": "fade", "fadeblack": "fadeblack", "dissolve": "dissolve", "wipe": "wipeleft", "slide": "slideleft"}


def shots_video(path: Path, durations: list[float], transitions: list[str] | None = None, *, fps: int = 24, size: tuple[int, int] = SIZE, trans_dur: float = 1.0,
                seed0: int = 100, noise: float = 3.0, audio: Path | None = None) -> Path:
    """A video made of ``len(durations)`` static distinct pictures. ``transitions[i]`` joins shot i to i+1: ``cut`` | ``fade`` | ``fadeblack`` | ``dissolve`` |
    ``wipe`` | ``slide`` (default all cuts). Cut times are exactly the cumulative durations (a transition overlaps the end of shot i with the start of i+1)."""
    transitions = transitions or ["cut"] * (len(durations) - 1)
    tmp = path.parent / f"_{path.stem}_imgs"
    tmp.mkdir(exist_ok=True)
    inputs, norm = [], []
    for i, d in enumerate(durations):
        img = blob_image(tmp / f"s{i}.png", seed0 + i, size)
        extra = trans_dur if i + 1 < len(durations) and transitions[i] != "cut" else 0.0
        inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{d + extra:.3f}", "-i", str(img)]
        norm.append(f"[{i}:v]settb=1/{fps},fps={fps},format=yuv420p,setsar=1[n{i}]")
    graph = list(norm)
    cur, length = "n0", durations[0] + (trans_dur if len(durations) > 1 and transitions[0] != "cut" else 0.0)
    for i in range(1, len(durations)):
        t = transitions[i - 1]
        nxt = f"c{i}"
        if t == "cut":
            graph.append(f"[{cur}][n{i}]concat=n=2:v=1:a=0,settb=1/{fps},fps={fps}[{nxt}]")
            length += durations[i] + (trans_dur if i + 1 < len(durations) and transitions[i] != "cut" else 0.0)
        else:
            graph.append(f"[{cur}][n{i}]xfade=transition={XFADE[t]}:duration={trans_dur}:offset={length - trans_dur:.3f}[{nxt}]")
            length += durations[i] + (trans_dur if i + 1 < len(durations) and transitions[i] != "cut" else 0.0) - trans_dur
        cur = nxt
    last = f"[{cur}]noise=alls={noise}:allf=t,format=yuv420p[v]" if noise else f"[{cur}]format=yuv420p[v]"
    graph.append(last)
    out_args = ["-filter_complex", ";".join(graph), "-map", "[v]"]
    if audio is not None:
        inputs += ["-i", str(audio)]
        out_args += ["-map", f"{len(durations)}:a", "-c:a", "aac", "-shortest"]
    run_ffmpeg([*inputs, *out_args, "-r", str(fps), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", str(path)])
    return path


def camera_video(path: Path, seconds: float, mode: str = "zoom", amount: float = 0.10, *, fps: int = 24, size: tuple[int, int] = SIZE, seed: int = 7, noise: float = 2.0) -> Path:
    """One long shot of a static picture with *known* camera movement: ``zoom`` (end/start scale = 1+amount), ``zoom_out``, ``pan`` (a pan covering ``amount`` of the
    width), ``shake`` (fast irregular movement) or ``static``."""
    w, h = size
    big = blob_image(path.parent / f"_{path.stem}_big.png", seed, (w * 3, h * 3))
    d = seconds
    if mode == "zoom":
        z = f"(1+{amount}*t/{d})"
        vf = f"scale=w='{w * 3}*{z}':h='{h * 3}*{z}':eval=frame,crop={w}:{h}"
    elif mode == "zoom_out":
        z = f"(1+{amount}*(1-t/{d}))"
        vf = f"scale=w='{w * 3}*{z}':h='{h * 3}*{z}':eval=frame,crop={w}:{h}"
    elif mode == "pan":
        vf = f"crop={w}:{h}:x='({w}*{amount * 3})*t/{d}+{w}':y='{h}'"
    elif mode == "shake":
        vf = f"crop={w}:{h}:x='{w}+{w * 0.35}*sin(2*PI*2.3*t)+{w * 0.15}*sin(2*PI*5.1*t)':y='{h}+{h * 0.3}*sin(2*PI*3.7*t)'"
    else:
        vf = f"crop={w}:{h}:x='{w}':y='{h}'"
    nz = f",noise=alls={noise}:allf=t" if noise else ""
    run_ffmpeg(["-loop", "1", "-framerate", str(fps), "-t", f"{seconds:.3f}", "-i", str(big), "-vf", vf + nz + ",format=yuv420p", "-r", str(fps), "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "16", "-pix_fmt", "yuv420p", str(path)])
    return path


def font_path(bold: bool = True) -> str:
    from app.rendering.fonts import FontResolver

    return FontResolver().resolve("Sans", bold).path


def drawtext_events(events: list[dict], size: tuple[int, int] = SIZE) -> str:
    """A drawtext filter chain. event: {start, end, text, pos: bottom|center|top|lower_third, size: rel. height (default .07), color, box: bool, fade: seconds}"""
    w, h = size
    fp = font_path(True).replace(":", "\\:")
    parts = []
    for e in events:
        fs = int(h * e.get("size", 0.07))
        pos = e.get("pos", "bottom")
        x, y = {"bottom": ("(w-text_w)/2", f"h-h*0.10-text_h"), "center": ("(w-text_w)/2", "(h-text_h)/2"), "top": ("(w-text_w)/2", "h*0.08"),
                "lower_third": ("w*0.06", "h*0.74")}[pos]
        box = ":box=1:boxcolor=black@0.65:boxborderw=8" if e.get("box") else ""
        alpha = f":alpha='min(1,max(0,(t-{e['start']})/{e['fade']}))'" if e.get("fade") else ""
        parts.append(f"drawtext=fontfile='{fp}':text='{e['text']}':fontsize={fs}:fontcolor={e.get('color', 'white')}:borderw={e.get('border', 2)}:bordercolor=black:x={x}:y={y}{box}{alpha}:"
                     f"enable='between(t,{e['start']},{e['end']})'")
    return ",".join(parts)


def overlay_video(path: Path, seconds: float, events: list[dict], *, fps: int = 24, size: tuple[int, int] = SIZE, seed: int = 11, shots: list[float] | None = None) -> Path:
    """A video with on-screen text of known timing/position (captions, headlines, number cards, lower thirds). Underneath: one picture, or hard-cut shots."""
    tmp = path.parent / f"_{path.stem}_base.mp4"
    if shots:
        shots_video(tmp, shots, fps=fps, size=size, seed0=seed)
    else:
        camera_video(tmp, seconds, "static", fps=fps, size=size, seed=seed)
    run_ffmpeg(["-i", str(tmp), "-vf", drawtext_events(events, size) if events else "null", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", str(path)])
    return path


# ---------------------------------------------------------------------------------------------- audio
SR = 22050


def _env(n: int, a: int, r: int) -> np.ndarray:
    e = np.ones(n, dtype=np.float32)
    a, r = min(a, n // 2), min(r, n // 2)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] = np.linspace(1, 0, r)
    return e


def synth_voice(seconds: float, phrases: list[tuple[float, float]], f0: float = 130.0, amp: float = 0.35, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Speech-like signal: syllable bursts (~4.5/s) of a harmonic voice with formant peaks, inside the given phrases; silence between them."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    for a, b in phrases:
        t = a
        while t < b - 0.08:
            d = float(rng.uniform(0.09, 0.2))
            d = min(d, b - t)
            n = int(d * sr)
            tt = np.arange(n) / sr
            f = f0 * (1 + 0.08 * np.sin(2 * np.pi * 5 * (tt + t)) + 0.12 * rng.uniform(-1, 1))
            phase = 2 * np.pi * np.cumsum(f) / sr
            sig = np.zeros(n, dtype=np.float32)
            for k in range(1, 28):
                fk = f0 * k
                gain = np.exp(-((fk - 700) / 450) ** 2) + 0.8 * np.exp(-((fk - 1500) / 600) ** 2) + 0.35 * np.exp(-((fk - 2600) / 700) ** 2) + 0.04
                sig += (gain / k ** 0.3) * np.sin(k * phase)
            sig *= _env(n, int(0.02 * sr), int(0.03 * sr))
            sig += rng.normal(0, 0.02, n).astype(np.float32)
            s0 = int(t * sr)
            out[s0:s0 + n] += sig[: len(out) - s0]
            t += d + float(rng.uniform(0.01, 0.05))
    m = float(np.abs(out).max())
    return out * (amp / m) if m > 0 else out


def synth_music(seconds: float, ranges: list[tuple[float, float, float]], sr: int = SR, chord: tuple[float, ...] = (196.0, 246.9, 293.7, 392.0), pulse: float = 0.0) -> np.ndarray:
    """A sustained pad (a chord with slow tremolo); ``ranges`` = (start, end, level). ``pulse`` > 0 adds a rhythmic beat at that rate (Hz)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    pad = sum(np.sin(2 * np.pi * f * t + i) * (1.0 / (i + 1) ** 0.5) for i, f in enumerate(chord)).astype(np.float32)
    pad *= (0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32)
    if pulse:
        pad *= (0.6 + 0.4 * np.maximum(0, np.sin(2 * np.pi * pulse * t)) ** 2).astype(np.float32)
    pad /= float(np.abs(pad).max())
    gain = np.zeros(n, dtype=np.float32)
    for a, b, lv in ranges:
        i0, i1 = int(a * sr), min(n, int(b * sr))
        gain[i0:i1] = lv * _env(i1 - i0, int(0.5 * sr), int(0.5 * sr))
    return pad * gain


def synth_sfx(seconds: float, times: list[float], kind: str = "whoosh", amp: float = 0.5, sr: int = SR, seed: int = 3) -> np.ndarray:
    """Short sound effects at known times: ``whoosh`` (swept noise, 0.35 s), ``hit`` (low thump + click, 0.4 s), ``tick`` (0.06 s click)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    for t0 in times:
        d = {"whoosh": 0.35, "hit": 0.4, "tick": 0.06}[kind]
        n = int(d * sr)
        tt = np.arange(n) / sr
        if kind == "whoosh":
            sig = rng.normal(0, 1, n) * np.sin(np.pi * tt / d) ** 2
            sig += 0.6 * np.sin(2 * np.pi * (300 * tt + 3000 * tt ** 2 / (2 * d))) * np.sin(np.pi * tt / d)
        elif kind == "hit":
            sig = np.sin(2 * np.pi * 60 * tt) * np.exp(-tt * 9) + 0.5 * rng.normal(0, 1, n) * np.exp(-tt * 40)
        else:
            sig = rng.normal(0, 1, n) * np.exp(-tt * 90)
        sig = (sig / (np.abs(sig).max() + 1e-9) * amp).astype(np.float32)
        i0 = int(t0 * sr)
        out[i0:i0 + n] += sig[: len(out) - i0]
    return out


def smooth_gain(n: int, active: list[tuple[float, float]], low: float, ramp: float = 0.25, sr: int = SR) -> np.ndarray:
    """A gain curve that is 1.0 normally and falls to ``low`` inside the ``active`` spans (with ``ramp`` seconds attack/release): ducking."""
    g = np.ones(n, dtype=np.float32)
    for a, b in active:
        i0, i1 = int(a * sr), min(n, int(b * sr))
        g[i0:i1] = np.minimum(g[i0:i1], low)
        r = int(ramp * sr)
        for k in range(r):
            f = low + (1 - low) * (1 - k / r)
            if i0 - r + k >= 0:
                g[i0 - r + k] = min(g[i0 - r + k], 1 - (1 - low) * (k / r))
            if i1 + k < n:
                g[i1 + k] = min(g[i1 + k], low + (1 - low) * (k / r))
    return g


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    x = np.clip(x, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return path


def make_audio(path: Path, seconds: float, *, voice: list[tuple[float, float]] | None = None, music: list[tuple[float, float, float]] | None = None, sfx: list[float] | None = None,
               sfx_kind: str = "whoosh", duck_to: float | None = None, sr: int = SR, pulse: float = 0.0) -> Path:
    """A mixed soundtrack with known structure. ``duck_to`` < 1 lowers the music inside the voice phrases to that fraction (with 0.25 s ramps)."""
    n = int(seconds * sr)
    mix = np.zeros(n, dtype=np.float32)
    if voice:
        mix += synth_voice(seconds, voice, sr=sr)
    if music:
        m = synth_music(seconds, music, sr=sr, pulse=pulse)
        if duck_to is not None and voice:
            m = m * smooth_gain(n, voice, duck_to, sr=sr)
        mix += m
    if sfx:
        mix += synth_sfx(seconds, sfx, sfx_kind, sr=sr)
    return write_wav(path, mix * 0.9, sr)


def mux(video: Path, audio: Path, out: Path) -> Path:
    run_ffmpeg(["-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)])
    return out
