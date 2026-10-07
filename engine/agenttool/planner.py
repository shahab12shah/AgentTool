"""Turns the script into scenes and an edit plan (queries, effects, sfx).

The AI planner (Claude) is used when an Anthropic key is configured; otherwise a
rule-based planner produces a varied, sensible default so the tool still works offline.
"""
import json
import re

from .transcribe import split_sentences
from .util import http_json

STOP = set("""a an the and or but if then so of to in on at for from by with as is are was were be been being it its
this that these those i you he she we they them his her our your their my me us not no do does did done have has had
will would can could should may might just about into over under up down out very more most some any all each than
too also there here what which who whom whose when where why how like get got make made""".split())

ZOOMS = ["zoom_in", "zoom_out", "punch_in", "pan_left", "pan_right", "drift", "shake"]
TRANSITIONS = ["fade", "slideleft", "slideright", "wipeleft", "circleopen", "zoomin", "smoothleft",
               "radial", "fadeblack", "hardcut", "hardcut", "hardcut", "dissolve", "pixelize"]
SFX = ["whoosh", "hit", "click", "riser", None]
OVERLAYS = ["grain", "vignette", "flash", None]


def build_scenes(words, total, target):
    """Group script sentences into scenes of roughly `target` seconds."""
    # sentence -> word index ranges
    sentences, idx = [], 0
    text = " ".join(w["w"] for w in words)
    for s in split_sentences(text):
        n = len(s.split())
        sentences.append((idx, idx + n))
        idx += n
    # split very long sentences at commas / midpoints
    units = []
    for a, b in sentences:
        dur = words[b - 1]["end"] - words[a]["start"]
        if dur > target * 2.4 and b - a > 6:
            pieces = max(2, round(dur / (target * 1.3)))
            step = (b - a) / pieces
            cuts = [a + round(step * k) for k in range(pieces)] + [b]
            for x, y in zip(cuts, cuts[1:]):
                # prefer a nearby comma boundary
                for k in range(max(x, y - 3), y):
                    if words[k]["w"].endswith((",", ";", ":")):
                        y = k + 1
                        break
                if y > x:
                    units.append((x, y))
        else:
            units.append((a, b))
    # fix overlaps created by comma snapping
    merged, last = [], 0
    for a, b in units:
        a = max(a, last)
        if b > a:
            merged.append((a, b))
            last = b
    # merge tiny units until they reach the target length
    scenes, cur = [], None
    for a, b in merged:
        if cur is None:
            cur = [a, b]
        else:
            cur[1] = b
        if words[cur[1] - 1]["end"] - words[cur[0]]["start"] >= target * 0.8:
            scenes.append(tuple(cur))
            cur = None
    if cur:
        if scenes and words[cur[1] - 1]["end"] - words[cur[0]]["start"] < target * 0.5:
            scenes[-1] = (scenes[-1][0], cur[1])
        else:
            scenes.append(tuple(cur))
    out = []
    for i, (a, b) in enumerate(scenes):
        start = 0.0 if i == 0 else words[a]["start"]
        end = total if i == len(scenes) - 1 else words[scenes[i + 1][0]]["start"]
        out.append({"id": i, "word_range": [a, b], "start": round(start, 3), "end": round(end, 3),
                    "text": " ".join(w["w"] for w in words[a:b])})
    return out


def _keywords(text, n=3):
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'-]+", text)]
    toks = [t for t in toks if t not in STOP and len(t) > 3]
    seen, out = set(), []
    for t in sorted(toks, key=lambda x: -len(x)):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:n]


def _heuristic_plan(scenes):
    plans = []
    for i, sc in enumerate(scenes):
        kws = _keywords(sc["text"])
        plans.append({
            "queries": [" ".join(kws[:2]) or "abstract background", " ".join(kws[1:3]) or "cinematic b-roll"],
            "mood": "neutral",
            "kind": "any",
            "zoom": ZOOMS[i % len(ZOOMS)],
            "transition": TRANSITIONS[(i * 5 + 1) % len(TRANSITIONS)],
            "sfx": SFX[i % len(SFX)],
            "overlay": OVERLAYS[(i * 3) % len(OVERLAYS)],
        })
    return plans


SYSTEM = """You are a senior video editor planning a YouTube video from a voiceover script.
For EVERY scene return one edit decision. Think about what the viewer should SEE while hearing the line:
concrete, filmable visuals (not abstract concepts), good variety, strong pacing.
Return ONLY JSON: {"scenes":[{"id":int,"queries":[str,str,str],"mood":str,"kind":"video|image|any",
"zoom":"zoom_in|zoom_out|punch_in|pan_left|pan_right|drift|shake","transition":"fade|slideleft|slideright|wipeleft|circleopen|zoomin|smoothleft|radial|fadeblack|hardcut|dissolve|pixelize",
"sfx":"whoosh|hit|click|riser|null","overlay":"grain|vignette|flash|null"}]}
Rules: queries are 2-4 word stock-footage search phrases, most specific first. Use hardcut + hit/flash for
punchy statements, fade/dissolve for calm moments, whoosh with slide/wipe transitions, riser before a reveal.
Never repeat the same zoom or transition more than twice in a row. Use image for facts/people/places that
are better as a still, video for action."""


def claude_plan(scenes, settings):
    key = settings["keys"].get("anthropic")
    if not key:
        return None
    payload = {"scenes": [{"id": s["id"], "text": s["text"], "seconds": round(s["end"] - s["start"], 1)} for s in scenes]}
    resp = http_json(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        data={"model": settings["ai_model"], "max_tokens": 8000, "system": SYSTEM,
              "messages": [{"role": "user", "content": json.dumps(payload)}]},
        timeout=120,
    )
    text = "".join(b.get("text", "") for b in resp.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0))
    by_id = {p["id"]: p for p in data["scenes"]}
    return [by_id.get(s["id"]) for s in scenes]


def plan_scenes(scenes, settings):
    base = _heuristic_plan(scenes)
    ai = None
    try:
        ai = claude_plan(scenes, settings)
    except Exception as exc:
        from .util import progress
        progress("plan", 30, f"AI planner failed ({exc}); using rule-based plan")
    for i, sc in enumerate(scenes):
        p = dict(base[i])
        if ai and ai[i]:
            p.update({k: v for k, v in ai[i].items() if v not in (None, "", [])})
        if p.get("sfx") in ("null", "none"):
            p["sfx"] = None
        if p.get("overlay") in ("null", "none"):
            p["overlay"] = None
        sc["queries"] = p["queries"]
        sc["mood"] = p.get("mood", "neutral")
        sc["kind"] = p.get("kind", "any")
        sc["effects"] = {
            "zoom": p["zoom"] if p.get("zoom") in ZOOMS else "zoom_in",
            "transition": p["transition"] if p.get("transition") in TRANSITIONS else "fade",
            "sfx": p.get("sfx"),
            "overlay": p.get("overlay"),
        }
    return scenes
