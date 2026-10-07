import json
import os

from . import sources
from .config import load_settings
from .planner import build_scenes, plan_scenes
from .ranker import rank
from .render import render
from .transcribe import timed_script_words
from .util import progress, run, ffprobe_size


def _plan_path(project_dir):
    return os.path.join(project_dir, "plan.json")


def load_plan(project_dir):
    with open(_plan_path(project_dir)) as f:
        return json.load(f)


def save_plan(project_dir, plan):
    os.makedirs(project_dir, exist_ok=True)
    with open(_plan_path(project_dir), "w") as f:
        json.dump(plan, f, indent=1)


def _gather(scene, source, settings, per_source=6):
    cands, errs = sources.search(source, scene["queries"], per_source, settings)
    return cands, errs


def find_visuals(scene, assigned, settings, context=""):
    """Search the assigned source (falling back to the others), rank, and pick."""
    order = [assigned] if assigned else []
    order += [s for s in sources.active_sources(settings) if s != assigned]
    cands, errors = [], []
    for src in order:
        got, errs = _gather(scene, src, settings)
        cands += got
        errors += errs
        if len(cands) >= 3:
            break
    scene["errors"] = errors
    if not cands:
        scene["candidates"], scene["chosen"], scene["reason"] = [], None, "no visuals found"
        return
    order_idx, reason = rank(scene, cands, settings, context)
    scene["candidates"] = [cands[i] for i in order_idx]
    scene["chosen"] = 0
    scene["reason"] = reason


def analyze(script, audio, project_dir, overrides, context=""):
    settings = load_settings(overrides)
    words, total, method = timed_script_words(script, audio)
    progress("plan", 15, f"Splitting script into scenes (timing: {method})")
    scenes = build_scenes(words, total, settings["editing"]["scene_seconds"])
    scenes = plan_scenes(scenes, settings)
    assigned = sources.assign_sources(len(scenes), settings)
    for i, sc in enumerate(scenes):
        progress("visuals", 25 + 70 * i / len(scenes), f"Finding visuals {i + 1}/{len(scenes)}")
        find_visuals(sc, assigned[i], settings, context or script[:200])
        sc["source"] = assigned[i]
        sc["approved"] = False
    plan = {"script": script, "audio": os.path.abspath(audio), "duration": total, "timing": method,
            "words": words, "scenes": scenes, "context": context}
    save_plan(project_dir, plan)
    progress("done", 100, "Analysis finished - review the picks, then render")
    return plan


def research(project_dir, scene_id, source, queries, overrides):
    settings = load_settings(overrides)
    plan = load_plan(project_dir)
    sc = plan["scenes"][scene_id]
    if queries:
        sc["queries"] = queries
    cands, errs = sources.search(source, sc["queries"], 8, settings)
    have = {c["cid"] for c in sc["candidates"]}
    sc["candidates"] += [c for c in cands if c["cid"] not in have]
    save_plan(project_dir, plan)
    return {"scene": sc, "errors": errs}


def _placeholder(project_dir, i):
    p = os.path.join(project_dir, "cache", f"placeholder_{i % 5}.png")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cols = ["1b2a41", "2d1b41", "1b413a", "41301b", "3a1b1b"]
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             f"color=c=0x{cols[i % 5]}:s=1920x1080", "-frames:v", "1", p])
    return p


def render_plan(project_dir, overrides, out_path=None):
    settings = load_settings(overrides)
    plan = load_plan(project_dir)
    cache = os.path.join(project_dir, "cache")

    def fetch_visual(sc):
        need = sc["end"] - sc["start"]
        order = [sc["chosen"]] if sc.get("chosen") is not None else []
        order += [i for i in range(len(sc["candidates"])) if i not in order]
        for i in order:
            cand = sc["candidates"][i]
            try:
                path = sources.fetch(cand, cache, settings, need)
                sc["clip_start"] = None
                return path, cand["kind"]
            except Exception as exc:
                progress("render", 0, f"scene {sc['id']}: {cand['cid']} failed ({exc}); trying next")
        return _placeholder(project_dir, sc["id"]), "image"

    out_path = out_path or os.path.join(project_dir, "output.mp4")
    render(plan, project_dir, settings, out_path, fetch_visual)
    return {"output": out_path}
