"""Picks the best candidate for each scene.

With an Anthropic key, Claude looks at the candidate thumbnails together with the scene's line and
decides which one fits best (and why). Without one, a heuristic scores keyword overlap, resolution
and duration so the tool still produces a usable first cut.
"""
import json
import re

from .util import http_json


def _tokens(s):
    return set(re.findall(r"[a-z]{3,}", (s or "").lower()))


def heuristic_rank(scene, cands):
    want = _tokens(" ".join(scene.get("queries", [])) + " " + scene["text"])
    need = scene["end"] - scene["start"]
    scored = []
    for i, c in enumerate(cands):
        s = len(want & _tokens(c["title"])) * 2.0
        s += 1.0 if c["w"] >= 1920 else 0.3 if c["w"] >= 1280 else 0
        if c["kind"] == "video" and c["duration"]:
            s += 1.0 if c["duration"] >= need else 0.2
        if scene.get("kind") in ("video", "image"):
            s += 1.5 if c["kind"] == scene["kind"] else 0
        scored.append((s, i))
    scored.sort(reverse=True)
    return [i for _, i in scored]


def claude_rank(scene, cands, settings, context=""):
    key = settings["keys"].get("anthropic")
    thumbs = [(i, c) for i, c in enumerate(cands) if c["thumb"].startswith("http")][:8]
    if not key or not thumbs:
        return None
    content = [{"type": "text", "text": (
        f"Video context: {context}\nScene voiceover line: \"{scene['text']}\"\n"
        f"Mood: {scene.get('mood')}. Visual wanted: {scene.get('queries')}.\n"
        "Below are candidate visuals. Pick the one that best illustrates the line, is cinematic, relevant, "
        "and avoids watermarks, text overlays and random people staring at camera unless needed. "
        'Reply ONLY JSON: {"ranking":[candidate numbers best-first],"reason":"one sentence"}')}]
    for i, c in thumbs:
        content.append({"type": "text", "text": f"Candidate {i}: [{c['kind']}] {c['title'][:100]}"})
        content.append({"type": "image", "source": {"type": "url", "url": c["thumb"]}})
    resp = http_json("https://api.anthropic.com/v1/messages",
                     headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                     data={"model": settings["ai_model"], "max_tokens": 600,
                           "messages": [{"role": "user", "content": content}]}, timeout=90)
    text = "".join(b.get("text", "") for b in resp.get("content", []))
    d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    rk = [i for i in d["ranking"] if isinstance(i, int) and 0 <= i < len(cands)]
    rest = [i for i in range(len(cands)) if i not in rk]
    return rk + rest, d.get("reason", "")


def rank(scene, cands, settings, context=""):
    """Returns (ordered indices, reason)."""
    try:
        r = claude_rank(scene, cands, settings, context)
        if r:
            return r
    except Exception:
        pass
    return heuristic_rank(scene, cands), "heuristic match (add an Anthropic key for AI picking)"
