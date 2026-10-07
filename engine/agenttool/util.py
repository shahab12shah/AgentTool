import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.parse

UA = "AgentTool/0.1"


def emit(kind, **data):
    """Write one JSON line to stdout (read by the Electron app)."""
    sys.stdout.write(json.dumps({"event": kind, **data}) + "\n")
    sys.stdout.flush()


def progress(stage, pct, msg=""):
    emit("progress", stage=stage, pct=round(pct, 1), msg=msg)


def have(binary):
    return shutil.which(binary) is not None


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"command failed: {' '.join(cmd[:4])} ...\n{tail}")
    return p


def ffprobe_duration(path):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    return float(p.stdout.strip())


def ffprobe_size(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path])
    w, h = p.stdout.strip().split(",")[:2]
    return int(w), int(h)


def http_json(url, headers=None, data=None, timeout=40):
    h = {"User-Agent": UA}
    h.update(headers or {})
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url, dest, headers=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=90) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)
    return dest


def qs(params):
    return urllib.parse.urlencode(params)


def slug(s, n=40):
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")[:n] or "x"


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
