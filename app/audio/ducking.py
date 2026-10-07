"""AudioDuckingService: automatic ducking as editable volume keyframes (never a single global level).

Inputs are voice activity (word timestamps), important narration spans, pauses, SFX events and section intensity. Output is a list of
``DuckingEvent`` (the why) and keyframes (the how) that follow them with attack/release ramps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.audio.priority import VoicePriorityController
from app.presentation.models import AudioSettings, DuckingEvent

HANG = 0.35  # gaps shorter than this are part of the same speech segment
INTRO_GUARD = 0.2


@dataclass
class Important:
    start: float
    end: float
    cause: str
    scene_id: str = ""


@dataclass
class DuckPlan:
    events: list[DuckingEvent] = field(default_factory=list)
    levels: list[tuple[float, float, float]] = field(default_factory=list)  # (start, end, level) constant intervals
    speech: list[tuple[float, float]] = field(default_factory=list)


def speech_segments(words, hang: float = HANG) -> list[tuple[float, float]]:
    segs: list[list[float]] = []
    for w in words:
        if segs and w.start - segs[-1][1] <= hang:
            segs[-1][1] = max(segs[-1][1], w.end)
        else:
            segs.append([w.start, w.end])
    return [(a, b) for a, b in segs]


def speech_from_silence(duration: float, silence: list[list[float]]) -> list[tuple[float, float]]:
    out, cur = [], 0.0
    for a, b in silence:
        if a - cur > 0.05:
            out.append((cur, a))
        cur = max(cur, b)
    if duration - cur > 0.05:
        out.append((cur, duration))
    return out


class AudioDuckingService:
    def __init__(self, settings: AudioSettings, priority: VoicePriorityController | None = None) -> None:
        self.s = settings
        self.priority = priority or VoicePriorityController(settings)

    # ------------------------------------------------------------------ level plan
    def plan(self, speech: list[tuple[float, float]], duration: float, important: list[Important] | None = None,
             sfx: list[tuple[float, float]] | None = None, intensity: list[list[float]] | None = None, id_prefix: str = "duck") -> DuckPlan:
        s = self.s
        important = important or []
        sfx = sfx or []
        speech = [(a, min(b, duration)) for a, b in speech if a < duration]
        cuts = {0.0, duration}
        for a, b in speech:
            cuts |= {a, b}
        for imp in important:
            cuts |= {max(0.0, imp.start), min(duration, imp.end)}
        for a, b in sfx:
            cuts |= {max(0.0, a - 0.1), min(duration, b + 0.3)}
        gaps = [(x[1], y[0]) for x, y in zip(speech, speech[1:]) if y[0] - x[1] >= s.pause_rise_min]
        for a, b in gaps:
            cuts |= {a + 0.3, b - 0.3}
        for t, _m in intensity or []:
            cuts.add(min(max(0.0, t), duration))
        pts = sorted(c for c in cuts if 0.0 <= c <= duration)
        first, last = (speech[0][0], speech[-1][1]) if speech else (0.0, 0.0)
        plan = DuckPlan(speech=list(speech))
        spans: list[tuple[float, float, float, str, str, str]] = []  # a, b, level, kind, cause, scene
        for a, b in zip(pts, pts[1:]):
            if b - a < 1e-6:
                continue
            m = (a + b) / 2
            in_speech = any(x <= m < y for x, y in speech)
            imp = next((i for i in important if i.start <= m < i.end), None)
            gap = next(((x, y) for x, y in gaps if x + 0.3 <= m < y - 0.3), None)
            kind, cause, scene = "NORMAL", "", ""
            level = s.music_level
            if not speech or m < first - INTRO_GUARD:
                level, kind, cause = s.intro_level, "INTRO", "Before the narration"
            elif m >= last + 0.3:
                level, kind, cause = s.intro_level, "OUTRO", "After the narration"
            elif imp is not None and in_speech:
                level, kind, cause, scene = s.important_level, "DUCK", imp.cause, imp.scene_id
            elif gap is not None:
                level, kind, cause = s.pause_level, "RISE", f"Voice pause of {gap[1] - gap[0]:.1f}s"
            mult = 1.0
            for t, mm in sorted(intensity or []):
                if t <= m:
                    mult = mm
            level *= mult
            if any(x - 0.1 <= m < y + 0.3 for x, y in sfx):
                level *= s.sfx_duck_factor
                if kind == "NORMAL":
                    kind, cause = "SFX", "A sound effect plays"
            level = self.priority.clamp("MUSIC", level, in_speech)
            spans.append((a, b, round(level, 4), kind, cause, scene))
        merged: list[list] = []
        for sp in spans:
            if merged and abs(merged[-1][2] - sp[2]) < 1e-6 and merged[-1][3] == sp[3] and merged[-1][4] == sp[4]:
                merged[-1][1] = sp[1]
            else:
                merged.append(list(sp))
        plan.levels = [(m[0], m[1], m[2]) for m in merged]
        n = 0
        for a, b, level, kind, cause, scene in merged:
            if kind == "NORMAL":
                continue
            n += 1
            plan.events.append(DuckingEvent(f"{id_prefix}_{n:04d}", round(a, 3), round(b, 3), level, kind, cause, scene, "AI"))
        return plan

    # ------------------------------------------------------------------ keyframes
    def keyframes(self, plan: DuckPlan, fade_in: float | None = None, fade_out: float | None = None, t0: float = 0.0, t1: float | None = None) -> list[tuple[float, float]]:
        """Volume breakpoints (timeline time, linear level) for the window [t0, t1]; ramps follow the ducking events."""
        s = self.s
        lv = [(a, b, v) for a, b, v in plan.levels]
        if not lv:
            return []
        end = lv[-1][1] if t1 is None else t1
        pts: list[tuple[float, float]] = [(lv[0][0], lv[0][2])]
        for (a1, b1, v1), (a2, b2, v2) in zip(lv, lv[1:]):
            edge = b1  # == a2
            if abs(v1 - v2) < 1e-6:
                continue
            if v2 < v1:  # duck: arrive at the lower level exactly when the reason starts
                ramp = min(s.attack, max(0.05, (b1 - a1) / 2))
                pts += [(edge - ramp, v1), (edge, v2)]
            else:  # release
                ramp = min(s.release, max(0.05, (b2 - a2) / 2))
                pts += [(edge, v1), (edge + ramp, v2)]
        pts.append((lv[-1][1], lv[-1][2]))
        # clean up: sort, merge equal times, drop collinear points
        pts.sort(key=lambda p: p[0])
        clean: list[tuple[float, float]] = []
        for t, v in pts:
            if clean and abs(t - clean[-1][0]) < 1e-3:
                clean[-1] = (clean[-1][0], v)
            else:
                clean.append((round(t, 3), round(v, 4)))
        out: list[tuple[float, float]] = []
        for p in clean:
            if len(out) >= 2:
                (ta, va), (tb, vb) = out[-2], out[-1]
                if abs((vb - va) * (p[0] - tb) - (p[1] - vb) * (tb - ta)) < 1e-9:
                    out[-1] = p
                    continue
            out.append(p)
        # fades at the very start / end of the music
        if fade_in and fade_in > 0 and out:
            first_t = out[0][0]
            out = [(first_t, 0.0)] + [(first_t + fade_in, out[0][1])] + [p for p in out[1:] if p[0] > first_t + fade_in]
        if fade_out and fade_out > 0 and out:
            tail = [p for p in out if p[0] < end - fade_out]
            lvl = tail[-1][1] if tail else out[-1][1]
            out = tail + [(max(tail[-1][0] if tail else 0.0, end - fade_out), lvl), (end, 0.0)]
        return [(t, v) for t, v in out if t0 - 1e-6 <= t <= end + 1e-6]
