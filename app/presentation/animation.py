"""Animation presets as timeline data. An animation is a small, editable spec stored on the clip (``clip.animation``):

    {"in": {preset, duration, easing, delay, start_value, end_value}, "out": {...}}

``animation_state`` evaluates it at a clip-local time, so previews and a future renderer read the same numbers. Motion intensity
(LOW / MEDIUM / HIGH) and reduced-motion mode choose the defaults; HIGH is capped so it never becomes chaotic.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.timeline.keyframes import ease

PRESET_NAMES = ("fade_in", "fade_out", "slide_up", "slide_down", "slide_left", "slide_right", "scale_in", "scale_out", "pop", "reveal", "type_on", "counter", "highlight")
EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out")
SLIDE_PX = 48.0  # slide distance at 1080p
MAX_OVERSHOOT = 1.08

# preset -> defaults for (duration, easing, start_value, end_value); start/end meaning depends on the preset (opacity, scale, reveal...)
DEFAULTS = {
    "fade_in": (0.30, "ease_out", 0.0, 1.0), "fade_out": (0.30, "ease_in", 1.0, 0.0),
    "slide_up": (0.35, "ease_out", 0.0, 1.0), "slide_down": (0.35, "ease_out", 0.0, 1.0), "slide_left": (0.35, "ease_out", 0.0, 1.0), "slide_right": (0.35, "ease_out", 0.0, 1.0),
    "scale_in": (0.35, "ease_out", 0.95, 1.0), "scale_out": (0.30, "ease_in", 1.0, 0.95),
    "pop": (0.30, "ease_out", 0.85, 1.0), "reveal": (0.40, "ease_in_out", 0.0, 1.0), "type_on": (0.60, "linear", 0.0, 1.0),
    "counter": (0.90, "ease_out", 0.0, 1.0), "highlight": (0.40, "ease_in_out", 0.0, 1.0),
}


def spec(preset: str, **kw) -> dict:
    d, e, a, b = DEFAULTS[preset]
    out = {"preset": preset, "duration": d, "easing": e, "delay": 0.0, "start_value": a, "end_value": b}
    out.update(kw)
    return out


def problems(anim: dict | None, clip_duration: float) -> list[str]:
    out: list[str] = []
    anim = {k: v for k, v in (anim or {}).items() if k in ("in", "out")}
    if any(isinstance(v, str) for v in anim.values()):  # the Phase 4 string form is valid too
        anim = normalize(anim)
    for side in ("in", "out"):
        s = (anim or {}).get(side)
        if s is None:
            continue
        if not isinstance(s, dict):
            out.append(f"animation “{side}” must be a preset spec")
            continue
        if s.get("preset") not in PRESET_NAMES:
            out.append(f"unknown animation preset {s.get('preset')!r}")
        if s.get("easing", "linear") not in EASINGS:
            out.append(f"unknown easing {s.get('easing')!r}")
        try:
            d, dl = float(s.get("duration", 0)), float(s.get("delay", 0))
        except (TypeError, ValueError):
            out.append("animation duration/delay must be numbers")
            continue
        if d < 0 or dl < 0 or d + dl > clip_duration + 1e-6:
            out.append(f"animation “{side}” ({d + dl:.2f}s) does not fit the clip ({clip_duration:.2f}s)")
    return out


def normalize(anim) -> dict:
    """Accept the Phase 4 string form ({"in": "fade", "duration": .25}) and return preset specs."""
    anim = anim or {}
    out: dict = {}
    legacy = float(anim.get("duration", 0.25)) if not isinstance(anim.get("in"), dict) else None
    for side, default in (("in", "fade_in"), ("out", "fade_out")):
        v = anim.get(side)
        if isinstance(v, dict):
            out[side] = v
        elif isinstance(v, str):
            name = {"fade": default, "pop": "pop" if side == "in" else "fade_out", "none": None}.get(v, v if v in PRESET_NAMES else default)
            if name:
                out[side] = spec(name, duration=legacy or DEFAULTS[name][0])
    return out


# ------------------------------------------------------------------ intensity
def intensity_level(motion_intensity: float) -> str:
    return "LOW" if motion_intensity < 0.34 else "MEDIUM" if motion_intensity < 0.67 else "HIGH"


def default_animation(variant: str, level: str = "MEDIUM", reduced: bool = False, counter_ok: bool = False) -> dict:
    """Editable defaults per graphic type. Reduced motion: fades only, no slide/scale/counter and nothing faster than 0.25s."""
    fade_in, fade_out = spec("fade_in"), spec("fade_out")
    if reduced or level == "LOW":
        return {"in": spec("fade_in", duration=0.35), "out": spec("fade_out", duration=0.30)}
    fast = level == "HIGH"
    d = 0.25 if fast else 0.35
    if variant == "NUMBER":
        return {"in": spec("counter", duration=0.9) if (fast and counter_ok) else spec("pop", duration=d, start_value=0.88 if fast else 0.92), "out": fade_out}
    if variant == "DATE":
        return {"in": spec("scale_in", duration=d, start_value=0.95), "out": fade_out}
    if variant == "LOWER_THIRD":
        return {"in": spec("slide_up", duration=d), "out": spec("fade_out", duration=0.3)}
    if variant == "HEADLINE":
        return {"in": spec("reveal", duration=0.45 if not fast else 0.35), "out": fade_out}
    if variant == "WARNING":
        return {"in": spec("pop", duration=d, start_value=0.9), "out": fade_out}
    if variant == "EVIDENCE":
        return {"in": spec("reveal", duration=0.4), "out": spec("fade_out", duration=0.3)}
    return {"in": fade_in, "out": fade_out}


# ------------------------------------------------------------------ evaluation
@dataclass
class AnimState:
    opacity: float = 1.0
    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    reveal: float = 1.0
    counter: float = 1.0  # 0..1 progress of a number counter
    highlight: float = 1.0


def _phase(s: dict, t: float, out: bool, clip_dur: float) -> tuple[float, bool]:
    """(eased progress 0..1, active) of one side of the animation at clip-local time ``t``."""
    d = max(1e-6, float(s.get("duration", 0.3)))
    delay = float(s.get("delay", 0.0))
    a = (clip_dur - d - delay) if out else delay
    if out and t < a:
        return 0.0, False
    if not out and t > a + d:
        return 1.0, False
    u = min(1.0, max(0.0, (t - a) / d))
    return ease(str(s.get("easing", "linear")), u), True


def animation_state(anim: dict | None, t: float, clip_dur: float, canvas_h: float = 1080.0) -> AnimState:
    st = AnimState()
    an = normalize(anim)
    k = canvas_h / 1080.0
    for side in ("in", "out"):
        s = an.get(side)
        if not s:
            continue
        out = side == "out"
        e, active = _phase(s, t, out, clip_dur)
        if not out and not active and e >= 1.0:
            continue  # the in-animation is finished
        if out and not active:
            continue
        a, b = float(s.get("start_value", 0.0)), float(s.get("end_value", 1.0))
        p = a + (b - a) * e
        name = s["preset"]
        if name in ("fade_in", "fade_out"):
            st.opacity *= p
        elif name.startswith("slide_"):
            dist = min(80.0, SLIDE_PX) * k * (1 - p if not out else p)
            sign = {"slide_up": (0, 1), "slide_down": (0, -1), "slide_left": (1, 0), "slide_right": (-1, 0)}[name]
            st.dx += sign[0] * dist
            st.dy += sign[1] * dist
            st.opacity *= (p if not out else 1 - p)
        elif name in ("scale_in", "scale_out"):
            st.scale *= p
            st.opacity *= e if name == "scale_in" else 1 - e
        elif name == "pop":
            if not out:
                peak = min(MAX_OVERSHOOT, b + 0.06)
                st.scale *= (a + (peak - a) * (e / 0.65)) if e < 0.65 else (peak + (b - peak) * ((e - 0.65) / 0.35))
                st.opacity *= min(1.0, e * 2)
            else:
                st.opacity *= 1 - e
        elif name in ("reveal", "type_on"):
            st.reveal = min(st.reveal, p if not out else 1.0)
            if out:
                st.opacity *= 1 - e
        elif name == "counter":
            st.counter = min(st.counter, e if not out else 1.0)
            st.opacity *= min(1.0, e * 4) if not out else 1 - e
        elif name == "highlight":
            st.highlight = p
    return st


def counter_text(counter: dict | None, progress: float) -> str | None:
    """The number shown while counting up (it always ends on the real figure)."""
    if not counter:
        return None
    to = float(counter.get("to", 0.0))
    dec = int(counter.get("decimals", 0))
    v = to * progress
    body = f"{v:,.{dec}f}" if counter.get("thousands", True) else f"{v:.{dec}f}"
    return f"{counter.get('prefix', '')}{body}{counter.get('suffix', '')}"
