"""Text, captions and graphics as an ASS subtitle script (rendered by libass through FFmpeg's ``ass`` filter).

Everything on the V4/V5/V6 tracks (evidence highlights, headlines, lower thirds, numbers, dates, captions) becomes timed ASS events.
Animations are *sampled*: ``animation_state`` — the same function the preview uses — is evaluated once per output frame during an animation,
so the export moves exactly like the preview and no preset has to be translated into ASS tags. Nothing is burned into any asset; the
script is generated fresh from the timeline for every render.

Coordinates are project-canvas pixels (``PlayRes`` = the canvas), so the same script renders correctly at any output resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.captions.styles import effective_style, style_for
from app.presentation.animation import AnimState, animation_state, counter_text, normalize
from app.rendering.fonts import FontMatch, FontResolver
from app.rendering.models import RenderSnapshot
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT, Clip

STYLE_COLORS = {"NUMBER_CARD": "#f2c14e", "WARNING": "#e5584f", "LOWER_THIRD": "#ffffff", "ENTITY_NAME": "#ffffff", "DATE": "#9ad1ff", "HEADLINE": "#ffffff"}
BOLD_EMPHASIS = ("BOLD_TEXT", "PUNCH_TEXT", "NUMBER_CARD", "WARNING_TEXT")
HIGHLIGHT_FILL = "#f2c14e"
DIM_OPACITY = 0.47  # 120/255, like the preview


# ---------------------------------------------------------------------------------------------- helpers
def ass_color(hex_color: str) -> str:
    h = (hex_color or "#ffffff").lstrip("#")
    if len(h) != 6:
        h = "ffffff"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{b}{g}{r}&".upper()


def ass_alpha(opacity: float) -> str:
    return f"&H{max(0, min(255, int(round(255 * (1.0 - max(0.0, min(1.0, opacity)))))) ):02X}&"


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\r", "").replace("\n", "\\N")


def num(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


class TextMeasure:
    """Text widths from the real font file (Pillow); falls back to an average glyph width when Pillow or the file is unavailable."""

    def __init__(self) -> None:
        self._fonts: dict[tuple[str, int], object] = {}
        self._ratio: dict[str, float] = {}

    def _font(self, path: str, size: int):
        key = (path, size)
        if key not in self._fonts:
            try:
                from PIL import ImageFont

                self._fonts[key] = ImageFont.truetype(path, size) if path else None
            except Exception:
                self._fonts[key] = None
        return self._fonts[key]

    def width(self, text: str, path: str, px: float) -> float:
        f = self._font(path, max(8, int(round(px))))
        if f is None:
            return len(text) * px * 0.54
        try:
            return float(f.getlength(text)) * px / max(8, int(round(px)))  # type: ignore[attr-defined]
        except Exception:
            return len(text) * px * 0.54

    def cell_ratio(self, path: str) -> float:
        """libass sizes ``\\fs`` by the font's cell height (ascent + descent), Qt by the em. This converts an em size to the equivalent ``\\fs``."""
        if path not in self._ratio:
            f = self._font(path, 100)
            try:
                asc, desc = f.getmetrics()  # type: ignore[union-attr]
                self._ratio[path] = max(0.8, min(1.6, (asc + desc) / 100.0))
            except Exception:
                self._ratio[path] = 1.17
        return self._ratio[path]


@dataclass
class AssEvent:
    start: float
    end: float
    layer: int
    style: str  # Plain | Box
    text: str
    order: int = 0
    clip_id: str = ""


@dataclass
class AssDocument:
    width: int
    height: int
    events: list[AssEvent] = field(default_factory=list)
    fonts: list[FontMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self, w0: float | None = None, w1: float | None = None) -> str:
        """The script text; with a window only the events that overlap it are included (times stay absolute)."""
        evs = [e for e in self.events if w0 is None or (e.end > w0 and e.start < (w1 if w1 is not None else math.inf))]
        evs.sort(key=lambda e: (e.start, e.layer, e.order))
        head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {self.width}\nPlayResY: {self.height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n"
                "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Plain,Sans,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1\n"
                "Style: Box,Sans,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,0,0,5,0,0,0,1\n\n"
                "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        body = "".join(f"Dialogue: {e.layer},{ass_time(e.start)},{ass_time(e.end)},{e.style},,0,0,0,,{e.text}\n" for e in evs)
        return head + body


# ---------------------------------------------------------------------------------------------- builder
class AssBuilder:
    def __init__(self, snapshot: RenderSnapshot, fonts: FontResolver, fps: int) -> None:
        self.s, self.fonts, self.fps = snapshot, fonts, fps
        self.W, self.H = snapshot.canvas_w, snapshot.canvas_h
        self.measure = TextMeasure()
        self.doc = AssDocument(self.W, self.H)
        self._matches: dict[tuple[str, bool], FontMatch] = {}
        self._order = 0

    # ------------------------------------------------------------------ fonts
    def match(self, family: str, bold: bool) -> FontMatch:
        key = (family, bold)
        if key not in self._matches:
            m = self.fonts.resolve(family, bold)
            self._matches[key] = m
            if all(x.family != m.family or x.path != m.path for x in self.doc.fonts):
                self.doc.fonts.append(m)
            if m.substituted and m.reason and m.reason not in self.doc.warnings:
                self.doc.warnings.append(m.reason)
        return self._matches[key]

    def _font_path(self, m: FontMatch, bold: bool) -> str:
        return (m.bold_path if bold and m.bold_path else m.path) or ""

    # ------------------------------------------------------------------ frame-grid helpers
    def _edge(self, t: float) -> float:
        """The instant between the frame that does not show ``t`` yet and the first that does (frames sit on ``i / fps``)."""
        return (math.ceil(t * self.fps - 1e-6) - 0.5) / self.fps

    def _boundaries(self, clip: Clip, anim: dict, extra_times: list[float] | None = None) -> list[float]:
        """Event boundaries (absolute seconds, on frame edges): clip edges, every frame of an active animation and any extra change times."""
        a0, a1 = clip.timeline_start, clip.timeline_end
        b = {self._edge(a0), self._edge(a1)}
        an = normalize(anim)
        for side, spec in an.items():
            d, delay = float(spec.get("duration", 0.3)), float(spec.get("delay", 0.0))
            lo = a0 + delay if side == "in" else a1 - d - delay
            hi = lo + d
            f = math.ceil(max(lo, a0) * self.fps - 1e-6)
            while f / self.fps <= min(hi, a1) + 1e-9:
                b.add((f - 0.5) / self.fps)
                b.add((f + 0.5) / self.fps)
                f += 1
        for t in extra_times or []:
            b.add(self._edge(t))
        lo_edge, hi_edge = self._edge(a0), self._edge(a1)
        return sorted(x for x in b if lo_edge - 1e-9 <= x <= hi_edge + 1e-9)

    def _emit(self, clip: Clip, intervals: list[tuple[float, float, str, str]], layer: int) -> None:
        """Add events, merging neighbours whose text is identical (steady states are one event)."""
        merged: list[list] = []
        for a, b, style, text in intervals:
            if b - a < 0.001:
                continue
            if merged and merged[-1][2] == style and merged[-1][3] == text and abs(merged[-1][1] - a) < 1e-6:
                merged[-1][1] = b
            else:
                merged.append([a, b, style, text])
        for a, b, style, text in merged:
            self._order += 1
            self.doc.events.append(AssEvent(a, b, layer, style, text, self._order, clip.id))

    # ------------------------------------------------------------------ clip dispatch
    def build(self, clips: list[tuple[int, Clip]]) -> AssDocument:
        """``clips``: (track order, clip) for every text/graphic/caption clip on a visible track."""
        for _z, c in sorted(clips, key=lambda x: (x[0], x[1].timeline_start)):
            try:
                if c.kind == KIND_CAPTION:
                    self._caption(c)
                elif c.kind == KIND_TEXT:
                    self._text(c)
                elif c.kind == KIND_GRAPHIC:
                    self._graphic(c)
            except Exception as exc:  # one broken overlay must not stop the whole render; it is reported instead
                self.doc.warnings.append(f"{c.kind} {c.id} could not be rendered: {exc}")
        return self.doc

    # ------------------------------------------------------------------ shared tag pieces
    def _anim_at(self, clip: Clip, t_abs: float) -> AnimState:
        return animation_state(clip.animation, t_abs - clip.timeline_start, clip.duration, float(self.H))

    def _opacity_tags(self, op: float, box_alpha: float = 1.0, shadow: bool = False) -> str:
        vis = ass_alpha(op)
        outline = ass_alpha(op * box_alpha)
        return f"\\1a{vis}\\3a{outline}\\4a{ass_alpha(op * (0.6 if shadow else 0.0))}"

    def _reveal_clip(self, x0: float, x1: float, y0: float, y1: float, reveal: float) -> str:
        if reveal >= 0.999:
            return ""
        return f"\\clip({num(x0)},{num(y0)},{num(x0 + (x1 - x0) * max(0.0, reveal))},{num(y1)})"

    # ------------------------------------------------------------------ text clips
    def _text(self, c: Clip) -> None:
        t = c.text or {}
        content = str(t.get("content", ""))
        if not content:
            return
        pos = t.get("position", (0.5, 0.5))
        family = str(t.get("font", "Sans"))
        bold = t.get("emphasis") in BOLD_EMPHASIS
        m = self.match(family, bold)
        fpath = self._font_path(m, bold)
        em = float(t.get("size", 48))
        fs = em * self.measure.cell_ratio(fpath)
        left = t.get("alignment") == "left"
        cx, cy = float(pos[0]) * self.W, float(pos[1]) * self.H
        color = ass_color(STYLE_COLORS.get(str(t.get("style")), "#ffffff"))
        base_op = float(t.get("opacity", 1.0))
        boxed = t.get("background", "none") != "none"
        counter = t.get("counter")
        subtitle = str(t.get("subtitle", "") or "")
        edges = self._boundaries(c, c.animation)
        intervals = []
        for a, b in zip(edges, edges[1:]):
            mid = (a + b) / 2
            st = self._anim_at(c, mid)
            body = content
            if counter and st.counter < 1.0:
                body = counter_text(counter, st.counter) or content
            op = base_op * st.opacity
            scale = st.scale * 100.0
            px, py = cx + st.dx, cy + st.dy
            width = max((self.measure.width(line, fpath, em) for line in body.split("\n")), default=0.0) * st.scale
            x_left = (px if left else px - width / 2)
            tag = [f"\\an{4 if left else 5}\\pos({num(px)},{num(py)})", f"\\fn{m.family}", f"\\fs{num(fs)}", f"\\b{1 if bold else 0}", f"\\1c{color}",
                   f"\\fscx{num(scale)}\\fscy{num(scale)}"]
            if boxed:
                tag += [f"\\bord{num(em * 0.22)}", "\\shad0", "\\3c&H000000&", self._opacity_tags(op, 170 / 255)]
            else:
                tag += [f"\\bord0", f"\\shad{num(max(1.0, em * 0.04))}", "\\4c&H000000&", self._opacity_tags(op, 1.0, shadow=True)]
            clip_tag = self._reveal_clip(x_left - em * 0.4, x_left + width + em * 0.4, py - em * 1.6, py + em * 1.6, st.reveal)
            text = esc(body)
            if subtitle:
                text += f"\\N{{\\fs{num(fs * 0.6)}\\b0}}{esc(subtitle)}"
            intervals.append((a, b, "Box" if boxed else "Plain", "{" + "".join(tag) + clip_tag + "}" + text))
        self._emit(c, intervals, 1)

    # ------------------------------------------------------------------ evidence graphics
    def _graphic(self, c: Clip) -> None:
        hl = c.effects.get("highlight")
        if not hl:
            return
        ev = c.effects.get("evidence") or {}
        tool = str(ev.get("tool") or ("FOCUS_BOX" if hl.get("darken_surround") else "HIGHLIGHT"))
        x, y, w, h = (hl.get("region") or ev.get("region") or (0.2, 0.3, 0.6, 0.2))
        X0, Y0, X1, Y1 = x * self.W, y * self.H, (x + w) * self.W, (y + h) * self.H
        dim = bool(hl.get("darken_surround")) or tool == "DIM"
        thick = 4.0 * self.H / 1080.0
        gold = ass_color(HIGHLIGHT_FILL)
        edges = self._boundaries(c, c.animation)
        intervals_dim, intervals_box = [], []
        for a, b in zip(edges, edges[1:]):
            st = self._anim_at(c, (a + b) / 2)
            op = st.opacity
            clip_tag = self._reveal_clip(0, self.W, 0, self.H, st.reveal)
            if dim and tool != "HIGHLIGHT":
                rects = [(0, 0, self.W, Y0), (0, Y1, self.W, self.H), (0, Y0, X0, Y1), (X1, Y0, self.W, Y1)]
                path = " ".join(f"m {num(r[0])} {num(r[1])} l {num(r[2])} {num(r[1])} {num(r[2])} {num(r[3])} {num(r[0])} {num(r[3])}" for r in rects if r[2] - r[0] > 0.5 and r[3] - r[1] > 0.5)
                intervals_dim.append((a, b, "Plain", f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H000000&\\1a{ass_alpha(DIM_OPACITY * op)}{clip_tag}}}{path}{{\\p0}}"))
            if tool == "HIGHLIGHT":
                path = f"m {num(X0)} {num(Y0)} l {num(X1)} {num(Y0)} {num(X1)} {num(Y1)} {num(X0)} {num(Y1)}"
                intervals_box.append((a, b, "Plain", f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c{gold}\\1a{ass_alpha(0.28 * op)}{clip_tag}}}{path}{{\\p0}}"))
            elif tool == "UNDERLINE":
                path = f"m {num(X0)} {num(Y1)} l {num(X1)} {num(Y1)} {num(X1)} {num(Y1 + thick)} {num(X0)} {num(Y1 + thick)}"
                intervals_box.append((a, b, "Plain", f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c{gold}\\1a{ass_alpha(op)}{clip_tag}}}{path}{{\\p0}}"))
            elif tool == "POINTER":
                s = min(self.W, self.H) * 0.05
                path = f"m {num(X0 - s * 1.6)} {num(Y0 - s * 1.2)} l {num(X0)} {num(Y0)} {num(X0 - s * 1.2)} {num(Y0 - s * 1.7)}"
                intervals_box.append((a, b, "Plain", f"{{\\an7\\pos(0,0)\\p1\\bord{num(thick)}\\3c{gold}\\3a{ass_alpha(op)}\\shad0\\1c{gold}\\1a{ass_alpha(op)}{clip_tag}}}{path}{{\\p0}}"))
            elif tool != "DIM":  # FOCUS_BOX, MAGNIFY, CROP: an outlined frame (the zoom itself is the host clip's keyframes)
                path = f"m {num(X0)} {num(Y0)} l {num(X1)} {num(Y0)} {num(X1)} {num(Y1)} {num(X0)} {num(Y1)}"
                intervals_box.append((a, b, "Plain", f"{{\\an7\\pos(0,0)\\p1\\bord{num(thick)}\\3c{gold}\\3a{ass_alpha(op)}\\shad0\\1a&HFF&{clip_tag}}}{path}{{\\p0}}"))
        self._emit(c, intervals_dim, 0)
        self._emit(c, intervals_box, 0)

    # ------------------------------------------------------------------ captions
    def _caption(self, c: Clip) -> None:
        d = c.text or {}
        cs = self.s.caption_settings
        if not cs.enabled:
            return
        style = effective_style(style_for(self.s.caption_styles, d.get("style_id", cs.style_id)), cs, d.get("style_overrides"))
        lines = d.get("lines") or [d.get("text", "")]
        words = d.get("words", [])
        mode = d.get("highlight_mode", cs.highlight_mode)
        marks = {m["word_index"]: m for m in d.get("emphasis", [])}
        bold = style.weight == "bold"
        m = self.match(style.font, bold)
        fpath = self._font_path(m, bold)
        em = max(10.0, style.size_rel * self.H)
        fs = em * self.measure.cell_ratio(fpath)
        pos = d.get("position", cs.position)
        xy = d.get("position_xy") or []
        if pos == "top":
            an, px, py = 8, self.W / 2, self.H * cs.safe_margin_top
        elif pos == "center":
            an, px, py = 5, self.W / 2, self.H / 2
        elif pos == "custom" and len(xy) == 2:
            an, px, py = 5, xy[0] * self.W, xy[1] * self.H
        else:
            an, px, py = 2, self.W / 2, self.H * (1.0 - cs.safe_margin_bottom)
        boxed = style.background == "box"
        base_op = style.opacity
        word_times = [float(w["start"]) for w in words] if mode in ("HIGHLIGHT", "PROGRESSIVE") else []
        edges = self._boundaries(c, c.animation, word_times)
        intervals = []
        for a, b in zip(edges, edges[1:]):
            mid = (a + b) / 2
            st = self._anim_at(c, mid)
            cur = max([i for i, w in enumerate(words) if float(w["start"]) <= mid + 1e-9] or [-1])
            op = base_op * st.opacity
            lead = [f"\\an{an}\\pos({num(px + st.dx)},{num(py + st.dy)})", f"\\fn{m.family}", f"\\fs{num(fs)}", f"\\b{1 if bold else 0}"]
            if boxed:
                lead += [f"\\bord{num(em * 0.2)}", "\\shad0", f"\\3c{ass_color(style.background_color)}"]
                alpha_for = lambda o, hidden=False: f"\\1a{ass_alpha(0 if hidden else o)}\\3a{ass_alpha(o * style.background_opacity)}\\4a&HFF&"  # noqa: E731
            else:
                ow = style.outline_width * em
                lead += [f"\\bord{num(ow)}", f"\\3c{ass_color(style.outline_color)}", f"\\shad{num(max(1.0, em * 0.05)) if style.shadow else 0}", f"\\4c{ass_color(style.shadow_color)}"]
                alpha_for = lambda o, hidden=False: f"\\1a{ass_alpha(0 if hidden else o)}\\3a{ass_alpha(0 if hidden else o)}\\4a{ass_alpha(0 if hidden else o * (0.63 if style.shadow else 0.0))}"  # noqa: E731
            idx = 0
            out_lines = []
            for line in lines:
                toks = []
                for tok in line.split():
                    wi = idx
                    idx += 1
                    text = tok.upper() if style.uppercase else tok
                    hidden = mode == "PROGRESSIVE" and wi > cur
                    mk = marks.get(wi)
                    emph = mk.get("style") if mk else None
                    color = style.highlight_color if ((mode == "HIGHLIGHT" and wi == cur) or mk) else style.color
                    tags = [f"\\1c{ass_color(color)}", alpha_for(op, hidden)]
                    tags.append(f"\\b{1 if (bold or emph in ('BOLD', 'POP')) else 0}")
                    tags.append("\\u1" if emph == "UNDERLINE" else "\\u0")
                    sc = 112 if emph == "POP" else 100
                    tags.append(f"\\fscx{sc}\\fscy{sc}")
                    toks.append("{" + "".join(tags) + "}" + esc(text))
                out_lines.append(" ".join(toks))
            text = "\\N".join(out_lines)
            intervals.append((a, b, "Box" if boxed else "Plain", "{" + "".join(lead) + "}" + text))
        self._emit(c, intervals, 2)
