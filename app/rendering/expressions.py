"""Keyframes as FFmpeg expressions.

A ``Curve`` is a time-varying value written as an FFmpeg expression in the variable ``t`` (seconds on the *timeline*, because layer streams carry
timeline timestamps). Constant values stay plain numbers so the command builder can use fast static filters where nothing moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.timeline.keyframes import Keyframe


def fnum(v: float) -> str:
    """A number FFmpeg's expression parser reads the same way everywhere (no exponents, no locale)."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


@dataclass(frozen=True)
class Curve:
    value: float | None  # not None -> constant
    expr: str  # always valid; for constants it is the number

    @property
    def is_const(self) -> bool:
        return self.value is not None

    @staticmethod
    def const(v: float) -> "Curve":
        return Curve(float(v), fnum(v))

    def scaled(self, k: float) -> "Curve":
        if self.is_const:
            return Curve.const(self.value * k)  # type: ignore[operator]
        return Curve(None, f"(({self.expr})*{fnum(k)})")

    def plus(self, other: "Curve") -> "Curve":
        if self.is_const and other.is_const:
            return Curve.const(self.value + other.value)  # type: ignore[operator]
        return Curve(None, f"(({self.expr})+({other.expr}))")

    def times(self, other: "Curve") -> "Curve":
        if self.is_const and other.is_const:
            return Curve.const(self.value * other.value)  # type: ignore[operator]
        if self.is_const and self.value == 1.0:
            return other
        if other.is_const and other.value == 1.0:
            return self
        return Curve(None, f"(({self.expr})*({other.expr}))")

    def map(self, fmt: str) -> "Curve":
        """Wrap the expression: ``fmt`` contains ``{}`` for the inner value (constants are folded by evaluating nothing: only used on non-constants)."""
        return Curve(None, fmt.format(f"({self.expr})"))


def _ease(kind: str, u: str) -> str:
    if kind == "ease_in":
        return f"({u})*({u})"
    if kind == "ease_out":
        return f"(1-(1-({u}))*(1-({u})))"
    if kind == "ease_in_out":
        return f"(({u})*({u})*(3-2*({u})))"
    return f"({u})"


def keyframe_curve(keyframes: list[Keyframe], prop: str, clip_start: float, default: float, offset: float = 0.0) -> Curve:
    """Value of ``prop`` over timeline time ``t``: holds the first/last value outside the keyframes and eases between them exactly like ``value_at``.

    ``clip_start`` is the clip's start on the timeline; ``offset`` shifts keyframe times (used when a clip was cut at a chunk boundary).
    """
    pts = sorted(((k.time, k.value, k.interpolation) for k in keyframes if k.property == prop), key=lambda p: p[0])
    if not pts:
        return Curve.const(default)
    if len({round(v, 9) for _t, v, _i in pts}) == 1:
        return Curve.const(pts[0][1])
    T = f"(t-{fnum(clip_start - offset)})"
    expr = fnum(pts[-1][1])
    for (t0, v0, kind), (t1, v1, _k) in reversed(list(zip(pts, pts[1:]))):
        span = t1 - t0
        if span < 1e-9:
            continue
        seg = f"{fnum(v0)}+({fnum(v1 - v0)})*{_ease(kind, f'({T}-{fnum(t0)})/{fnum(span)}')}"
        expr = f"if(lt({T},{fnum(t1)}),{seg},{expr})"
    expr = f"if(lt({T},{fnum(pts[0][0])}),{fnum(pts[0][1])},{expr})"
    return Curve(None, expr)


def ramp(start: float, duration: float) -> str:
    """0 -> 1 over ``duration`` seconds beginning at timeline time ``start`` (clamped)."""
    return f"clip((t-{fnum(start)})/{fnum(max(duration, 1e-6))},0,1)"


def evaluate(expr: str, t: float) -> float:
    """Evaluate a Curve expression in Python (tests and diagnostics). Supports the small subset the builder emits."""
    import math

    def lt(a, b):
        return 1.0 if a < b else 0.0

    def clip(x, lo, hi):
        return min(hi, max(lo, x))

    def pyexpr(e: str) -> str:
        # translate if(a,b,c) -> (b if a else c): do it by recursive descent on the outermost call
        out, i = [], 0
        while i < len(e):
            if e.startswith("if(", i) and (i == 0 or not (e[i - 1].isalnum() or e[i - 1] == "_")):
                depth, j, parts, cur = 1, i + 3, [], []
                while depth:
                    ch = e[j]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    if ch == "," and depth == 1:
                        parts.append("".join(cur))
                        cur = []
                    else:
                        cur.append(ch)
                    j += 1
                parts.append("".join(cur))
                out.append(f"(({pyexpr(parts[1])}) if ({pyexpr(parts[0])}) else ({pyexpr(parts[2])}))")
                i = j + 1
            else:
                out.append(e[i])
                i += 1
        return "".join(out)

    return float(eval(pyexpr(expr), {"__builtins__": {}}, {"t": t, "lt": lt, "clip": clip, "min": min, "max": max, "PI": math.pi}))
