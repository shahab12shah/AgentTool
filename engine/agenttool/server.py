"""JSON-lines RPC over stdio. The Electron app spawns `python -m agenttool serve`."""
import json
import sys
import traceback

from . import pipeline
from .util import emit, have


def deps():
    try:
        import faster_whisper  # noqa: F401
        whisper = True
    except ImportError:
        whisper = False
    return {"ffmpeg": have("ffmpeg"), "yt_dlp": have("yt-dlp"), "faster_whisper": whisper}


def handle(cmd, a):
    if cmd == "deps":
        return deps()
    if cmd == "analyze":
        return pipeline.analyze(a["script"], a["audio"], a["project_dir"], a.get("settings"), a.get("context", ""))
    if cmd == "load":
        return pipeline.load_plan(a["project_dir"])
    if cmd == "save":
        pipeline.save_plan(a["project_dir"], a["plan"])
        return {"ok": True}
    if cmd == "research":
        return pipeline.research(a["project_dir"], a["scene_id"], a["source"], a.get("queries"), a.get("settings"))
    if cmd == "render":
        return pipeline.render_plan(a["project_dir"], a.get("settings"), a.get("out"))
    raise ValueError(f"unknown command {cmd}")


def serve():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            emit("result", id=req["id"], data=handle(req["cmd"], req.get("args", {})))
        except Exception as exc:
            emit("error", id=req["id"], message=str(exc), trace=traceback.format_exc())
