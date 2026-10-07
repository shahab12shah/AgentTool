"""Visual sources. Each search returns candidate dicts; fetch() downloads the chosen one."""
import glob
import hashlib
import hmac
import json
import os
import re
import subprocess
import time

from .util import http_json, download, qs, slug, have, ffprobe_duration

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _cand(source, kind, cid, **kw):
    d = {"cid": f"{source}:{cid}", "source": source, "kind": kind, "title": "", "thumb": "", "url": "",
         "duration": 0, "w": 0, "h": 0, "credit": "", "page": "", "start": 0}
    d.update(kw)
    return d


def _pick_file(files, want_w=1920):
    files = [f for f in files if f.get("link") and f.get("file_type", "video/mp4") == "video/mp4"]
    if not files:
        return None
    files.sort(key=lambda f: abs((f.get("width") or 0) - want_w))
    return files[0]


def search_pexels_video(q, n, settings):
    key = settings["keys"]["pexels"]
    if not key:
        return []
    r = http_json("https://api.pexels.com/videos/search?" + qs({"query": q, "per_page": n, "orientation": "landscape"}),
                  headers={"Authorization": key})
    out = []
    for v in r.get("videos", []):
        f = _pick_file(v.get("video_files", []))
        if f:
            out.append(_cand("pexels_video", "video", v["id"], title=v.get("url", "").rstrip("/").split("/")[-1].replace("-", " "),
                             thumb=v.get("image", ""), url=f["link"], duration=v.get("duration", 0),
                             w=f.get("width", 0), h=f.get("height", 0), credit="Pexels / " + v.get("user", {}).get("name", ""),
                             page=v.get("url", "")))
    return out


def search_pexels_photo(q, n, settings):
    key = settings["keys"]["pexels"]
    if not key:
        return []
    r = http_json("https://api.pexels.com/v1/search?" + qs({"query": q, "per_page": n, "orientation": "landscape"}),
                  headers={"Authorization": key})
    return [_cand("pexels_photo", "image", p["id"], title=p.get("alt", ""), thumb=p["src"]["medium"],
                  url=p["src"]["large2x"], w=p.get("width", 0), h=p.get("height", 0),
                  credit="Pexels / " + p.get("photographer", ""), page=p.get("url", ""))
            for p in r.get("photos", [])]


def search_pixabay_video(q, n, settings):
    key = settings["keys"]["pixabay"]
    if not key:
        return []
    r = http_json("https://pixabay.com/api/videos/?" + qs({"key": key, "q": q, "per_page": max(n, 3)}))
    out = []
    for h in r.get("hits", [])[:n]:
        vs = h.get("videos", {})
        v = vs.get("large") or vs.get("medium") or vs.get("small")
        if v and v.get("url"):
            out.append(_cand("pixabay_video", "video", h["id"], title=h.get("tags", ""),
                             thumb=(vs.get("medium") or v).get("thumbnail", ""), url=v["url"],
                             duration=h.get("duration", 0), w=v.get("width", 0), h=v.get("height", 0),
                             credit="Pixabay / " + h.get("user", ""), page=h.get("pageURL", "")))
    return out


def search_pixabay_photo(q, n, settings):
    key = settings["keys"]["pixabay"]
    if not key:
        return []
    r = http_json("https://pixabay.com/api/?" + qs({"key": key, "q": q, "per_page": max(n, 3), "image_type": "photo",
                                                    "orientation": "horizontal", "min_width": 1600}))
    return [_cand("pixabay_photo", "image", h["id"], title=h.get("tags", ""), thumb=h.get("webformatURL", ""),
                  url=h.get("largeImageURL", ""), w=h.get("imageWidth", 0), h=h.get("imageHeight", 0),
                  credit="Pixabay / " + h.get("user", ""), page=h.get("pageURL", ""))
            for h in r.get("hits", [])[:n]]


def search_youtube(q, n, settings):
    if not have("yt-dlp"):
        return []
    p = subprocess.run(["yt-dlp", f"ytsearch{n}:{q}", "--flat-playlist", "--dump-json", "--no-warnings"],
                       capture_output=True, text=True, timeout=90)
    out = []
    for line in p.stdout.splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            continue
        dur = v.get("duration") or 0
        if dur and dur < 15:
            continue
        clip = settings["sources"]["youtube"].get("clip_seconds", 6)
        out.append(_cand("youtube", "video", v["id"], title=v.get("title", ""),
                         thumb=f"https://i.ytimg.com/vi/{v['id']}/hqdefault.jpg",
                         url=f"https://www.youtube.com/watch?v={v['id']}", duration=dur,
                         credit="YouTube / " + (v.get("channel") or v.get("uploader") or ""),
                         page=f"https://www.youtube.com/watch?v={v['id']}",
                         start=round(min(max(dur * 0.25, 5), max(dur - clip - 2, 0)), 1)))
    return out


def search_storyblocks(q, n, settings):
    """Storyblocks API v2 (needs an API key + secret from a Storyblocks API plan). Best-effort."""
    key, secret = settings["keys"]["storyblocks"], settings["keys"]["storyblocks_secret"]
    if not (key and secret):
        return []
    resource = "/api/v2/videos/search"
    expires = str(int(time.time()) + 3600)
    sig = hmac.new((secret + expires).encode(), resource.encode(), hashlib.sha256).hexdigest()
    r = http_json("https://api.storyblocks.com" + resource + "?" + qs({
        "api_key": key, "expires": expires, "hmac": sig, "project_id": "agenttool", "user_id": "agenttool",
        "keywords": q, "results_per_page": n, "has_talent": "false"}))
    return [_cand("storyblocks", "video", v["id"], title=v.get("title", ""), thumb=v.get("thumbnail_url", ""),
                  url=(v.get("preview_urls") or {}).get("_720p", ""), duration=v.get("duration", 0),
                  credit="Storyblocks", page=v.get("url", ""))
            for v in r.get("results", [])]


def search_local(q, n, settings):
    folder = settings["sources"]["local"].get("folder")
    if not folder or not os.path.isdir(folder):
        return []
    words = set(re.findall(r"[a-z]{3,}", q.lower()))
    scored = []
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXT + IMAGE_EXT:
            continue
        name = set(re.findall(r"[a-z]{3,}", os.path.basename(path).lower()))
        scored.append((len(words & name), path, ext))
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, path, ext in scored[:n]:
        kind = "video" if ext in VIDEO_EXT else "image"
        out.append(_cand("local", kind, hashlib.md5(path.encode()).hexdigest()[:10], title=os.path.basename(path),
                         thumb="file://" + path if kind == "image" else "", url=path, credit="Your library"))
    return out


SEARCH = {
    "pexels_video": search_pexels_video, "pexels_photo": search_pexels_photo,
    "pixabay_video": search_pixabay_video, "pixabay_photo": search_pixabay_photo,
    "youtube": search_youtube, "storyblocks": search_storyblocks, "local": search_local,
}


def active_sources(settings):
    return {k: v for k, v in settings["sources"].items() if v.get("enabled") and k in SEARCH}


def assign_sources(n, settings):
    """Spread scenes over enabled sources according to their percent shares."""
    act = active_sources(settings)
    if not act:
        return [None] * n
    weights = {k: max(v.get("percent", 0), 0) for k, v in act.items()}
    if sum(weights.values()) == 0:
        weights = {k: 1 for k in act}
    tot = float(sum(weights.values()))
    share = {k: w / tot for k, w in weights.items() if w > 0}
    counts = {k: 0 for k in share}
    out = []
    for i in range(n):
        k = max(share, key=lambda s: share[s] * (i + 1) - counts[s])
        counts[k] += 1
        out.append(k)
    return out


def search(source, queries, n, settings):
    """Returns (candidates, errors)."""
    seen, out, errors = set(), [], []
    for q in queries:
        if len(out) >= n:
            break
        try:
            for c in SEARCH[source](q, n, settings):
                if c["cid"] not in seen:
                    seen.add(c["cid"])
                    c["query"] = q
                    out.append(c)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    return out[:n], errors


def fetch(cand, cache_dir, settings, need_seconds=6.0):
    """Download the candidate and return a local file path."""
    src = cand["source"]
    if src == "local":
        return cand["url"]
    name = slug(cand["cid"], 60)
    if src == "youtube":
        secs = max(need_seconds + 1, settings["sources"]["youtube"].get("clip_seconds", 6))
        s = cand.get("start", 0)
        dest = os.path.join(cache_dir, f"{name}_{int(s)}_{int(secs)}.mp4")
        if not os.path.exists(dest):
            os.makedirs(cache_dir, exist_ok=True)
            subprocess.run(["yt-dlp", cand["url"], "--download-sections", f"*{s}-{s + secs}",
                            "--force-keyframes-at-cuts", "-f", "bv*[height<=1080]/b[height<=1080]",
                            "--merge-output-format", "mp4", "-o", dest, "--no-warnings", "-q"],
                           check=True, timeout=300)
        return dest
    ext = ".mp4" if cand["kind"] == "video" else ".jpg"
    return download(cand["url"], os.path.join(cache_dir, name + ext))
