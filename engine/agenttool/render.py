"""FFmpeg renderer: per-scene effect segments -> xfade chain -> captions + audio mix."""
import os
import random

from .captions import build_ass
from .sfx import ensure_sfx, pick
from .util import run, ffprobe_duration, ffprobe_size, progress

TRANS_DUR = {"hardcut": 0.04, "fade": 0.5, "dissolve": 0.6, "fadeblack": 0.6, "slideleft": 0.4, "slideright": 0.4,
             "wipeleft": 0.45, "circleopen": 0.55, "zoomin": 0.5, "smoothleft": 0.5, "radial": 0.55, "pixelize": 0.4}
XFADE = {"hardcut": "fade"}

GRADES = {
    "none": "",
    "cinematic": "eq=contrast=1.10:saturation=1.12:gamma=0.98,colorbalance=rs=.05:gs=-.01:bs=-.06:rh=.05:bh=-.04",
    "warm": "eq=contrast=1.05:saturation=1.10,colorbalance=rs=.08:bs=-.08:rm=.04:bm=-.04",
    "cool": "eq=contrast=1.06:saturation=1.05,colorbalance=rs=-.05:bs=.08:bh=.04",
    "bw": "hue=s=0,eq=contrast=1.15",
}


def _zoom_expr(mode, n):
    """Return (z, x, y) zoompan expressions for n frames."""
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if mode == "zoom_out":
        return f"1.16-0.16*on/{n}", cx, cy
    if mode == "punch_in":
        return f"1.0+0.22*max(0,1-on/{max(int(n * 0.35), 1)})+0.04*on/{n}", cx, cy
    if mode == "pan_left":
        return "1.14", f"(iw-iw/zoom)*(1-on/{n})", cy
    if mode == "pan_right":
        return "1.14", f"(iw-iw/zoom)*on/{n}", cy
    if mode == "drift":
        return f"1.08+0.05*on/{n}", f"(iw-iw/zoom)*(0.3+0.4*on/{n})", f"(ih-ih/zoom)*(0.6-0.3*on/{n})"
    if mode == "shake":
        return "1.12", "(iw-iw/zoom)/2+22*sin(on*1.9)", "(ih-ih/zoom)/2+16*cos(on*2.3)"
    return f"1.0+0.14*on/{n}", cx, cy  # zoom_in


def _segment_filter(sc, kind, W, H, fps, frames, settings):
    ed = settings["editing"]
    eff = sc["effects"]
    parts = [f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},setsar=1"]
    if ed["zoom"]:
        z, x, y = _zoom_expr(eff["zoom"], frames)
        parts.append(f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={fps}")
    else:
        parts.append(f"scale={W}:{H},fps={fps}")
    grade = GRADES.get(ed["grade"], "")
    if grade:
        parts.append(grade)
    if ed["overlays"]:
        ov = eff.get("overlay")
        if ov == "grain" or ed["grade"] == "cinematic":
            parts.append("noise=alls=7:allf=t")
        if ov == "vignette" or ed["grade"] == "cinematic":
            parts.append("vignette=PI/5")
        if ov == "flash":
            parts.append("eq=brightness='0.55*max(0,1-t/0.16)':eval=frame")
    parts.append("format=yuv420p")
    return ",".join(parts)


def render_segment(sc, path, kind, out, W, H, fps, length, settings, rng):
    frames = max(int(round(length * fps)), 2)
    if kind == "video":
        dur = ffprobe_duration(path)
        pre = ["-stream_loop", "-1"] if dur < length + 0.5 else []
        ss = round(rng.uniform(0, max(dur - length - 0.2, 0)), 2) if dur > length + 0.5 else 0
        if sc.get("clip_start") is not None and dur > length + 0.5:
            ss = min(float(sc["clip_start"]), max(dur - length - 0.1, 0))
        inp = [*pre, "-ss", str(ss), "-t", f"{length:.3f}", "-i", path]
    else:
        inp = ["-loop", "1", "-framerate", str(fps), "-t", f"{length:.3f}", "-i", path]
    vf = _segment_filter(sc, kind, W, H, fps, frames, settings)
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inp, "-an", "-vf", vf, "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", out])


def render(plan, project_dir, settings, out_path, fetch_visual):
    """plan: dict with scenes (each with chosen visual), words, audio. fetch_visual(scene)->(path, kind)."""
    W, H, fps = settings["width"], settings["height"], settings["fps"]
    ed = settings["editing"]
    work = os.path.join(project_dir, "work")
    os.makedirs(work, exist_ok=True)
    rng = random.Random(7)
    scenes = plan["scenes"]
    n = len(scenes)

    # 1) transitions per scene (transition i = from scene i to i+1)
    trans = []
    for i, sc in enumerate(scenes[:-1]):
        name = sc["effects"]["transition"] if ed["transitions"] else "hardcut"
        td = min(TRANS_DUR.get(name, 0.4), (sc["end"] - sc["start"]) * 0.45,
                 (scenes[i + 1]["end"] - scenes[i + 1]["start"]) * 0.45)
        trans.append((name, max(td, 0.04)))

    # 2) segments
    segs = []
    for i, sc in enumerate(scenes):
        progress("render", 5 + 60 * i / n, f"Editing scene {i + 1}/{n}")
        path, kind = fetch_visual(sc)
        length = (sc["end"] - sc["start"]) + (trans[i][1] if i < n - 1 else 0)
        out = os.path.join(work, f"seg_{i:03d}.mp4")
        render_segment(sc, path, kind, out, W, H, fps, length, settings, rng)
        segs.append(out)

    # 3) captions
    ass_rel = None
    if settings["captions"]["enabled"]:
        with open(os.path.join(work, "captions.ass"), "w", encoding="utf-8") as f:
            f.write(build_ass(plan["words"], settings["captions"], W, H))
        ass_rel = "captions.ass"

    # 4) audio inputs: voice, music, sfx
    sfx_paths = ensure_sfx(os.path.join(project_dir, "sfx"), ed.get("sfx_folder", ""))
    inputs = []
    for s in segs:
        inputs += ["-i", os.path.relpath(s, work)]
    voice_idx = len(segs)
    inputs += ["-i", os.path.abspath(plan["audio"])]
    music = ed.get("music")
    music_idx = None
    if music and os.path.exists(music):
        music_idx = voice_idx + 1
        inputs += ["-stream_loop", "-1", "-i", os.path.abspath(music)]
    events = []
    if ed["sfx"]:
        for i, sc in enumerate(scenes):
            name = sc["effects"].get("sfx")
            f = pick(sfx_paths, name, rng) if name else None
            if f:
                offset = sc["start"] - (0.12 if name in ("whoosh", "riser") else 0.0)
                events.append((max(offset, 0), f, name))
    sfx_first = voice_idx + 1 + (1 if music_idx is not None else 0)
    for _, f, _ in events:
        inputs += ["-i", os.path.abspath(f)]

    # 5) filter graph
    fc = []
    label = "[0:v]"
    for i in range(1, n):
        name, td = trans[i - 1]
        offset = scenes[i]["start"]
        fc.append(f"{label}[{i}:v]xfade=transition={XFADE.get(name, name)}:duration={td:.3f}:offset={offset:.3f}[v{i}]")
        label = f"[v{i}]"
    vlabel = label
    if ass_rel:
        fc.append(f"{vlabel}ass={ass_rel}[vout]")
        vlabel = "[vout]"
    elif n == 1:
        fc.append(f"{vlabel}null[vout]")
        vlabel = "[vout]"
    amix_in = [f"[vo]"]
    fc.append(f"[{voice_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[vo][vsc]")
    if music_idx is not None:
        fc.append(f"[{music_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={ed['music_volume']}[m0]")
        fc.append("[m0][vsc]sidechaincompress=threshold=0.04:ratio=8:attack=20:release=400[md]")
        amix_in.append("[md]")
    else:
        fc.append("[vsc]anullsink")
    for k, (t, _, _) in enumerate(events):
        ms = int(t * 1000)
        fc.append(f"[{sfx_first + k}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                  f"volume={ed['sfx_volume']},adelay={ms}|{ms}[s{k}]")
        amix_in.append(f"[s{k}]")
    fc.append(f"{''.join(amix_in)}amix=inputs={len(amix_in)}:normalize=0:dropout_transition=0,alimiter=limit=0.95[aout]")

    script_path = os.path.join(work, "graph.txt")
    with open(script_path, "w") as f:
        f.write(";\n".join(fc))
    progress("render", 70, "Joining scenes, transitions, captions and audio")
    total = plan["duration"]
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex_script", "graph.txt",
         "-map", vlabel, "-map", "[aout]", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", os.path.abspath(out_path)],
        cwd=work)
    progress("render", 100, "Done")
    return out_path
