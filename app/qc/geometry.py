"""Where do captions, text and evidence graphics land on the screen?

QC never renders a frame, so it estimates the rectangle each overlay occupies from the clip data alone, with the same placement rules as the renderer
(``rendering/ass.py``): a caption is anchored bottom / top / centre or at a custom point, a text clip is centred (or left aligned) on its normalised position,
an evidence graphic covers its highlight region. Text widths use the caption engine's average glyph width (``captions.engine.AVG_CHAR``), so the estimate agrees
with the layout the engine itself used when it broke the lines.

All rectangles are in normalised canvas units (0..1 on both axes); a coordinate outside 0..1 is off the frame. The numbers are *estimates*: good enough to tell
"clearly outside the safe area" or "clearly on top of the evidence box", not to measure a pixel. Callers keep a tolerance (``MARGIN_EPS``) and lower their
confidence when a verdict rests on the estimate alone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.captions.engine import AVG_CHAR
from app.captions.styles import effective_style, style_for
from app.presentation.models import CaptionSettings, CaptionStyle
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT, Clip

try:  # the renderer's own tables: QC judges what will actually be drawn
    from app.rendering.ass import BOLD_EMPHASIS, STYLE_COLORS
except ImportError:  # pragma: no cover - the rendering package is part of the application
    BOLD_EMPHASIS = ("BOLD_TEXT", "PUNCH_TEXT", "NUMBER_CARD", "WARNING_TEXT")
    STYLE_COLORS = {"NUMBER_CARD": "#f2c14e", "WARNING": "#e5584f", "LOWER_THIRD": "#ffffff", "ENTITY_NAME": "#ffffff", "DATE": "#9ad1ff", "HEADLINE": "#ffffff"}

BOLD_WIDTH = 1.08  # a bold glyph is this much wider (same factor as captions.engine.layout_for)
UPPER_WIDTH = 1.12
TEXT_LINE_HEIGHT = 1.2  # text clips: line height / em
SUBTITLE_SCALE = 0.6  # the renderer draws a text clip's subtitle at 60 % of the title size
CAPTION_BOX_PAD = 0.2  # the renderer's box border around caption text, in em
TEXT_BOX_PAD = 0.22
MARGIN_EPS = 0.004  # normalised: ~4 px on a 1080p frame; smaller overshoots are rounding, not a layout problem
DEFAULT_REGION = (0.2, 0.3, 0.6, 0.2)  # the renderer's fallback when a highlight carries no region
MID_GREY = (128.0, 128.0, 128.0)  # the assumed backdrop when nothing is known about the picture underneath


# ---------------------------------------------------------------------------------------------- rectangles
@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def expanded(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)

    def intersection(self, other: "Rect") -> "Rect | None":
        x0, y0, x1, y1 = max(self.x0, other.x0), max(self.y0, other.y0), min(self.x1, other.x1), min(self.y1, other.y1)
        return Rect(x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def overlap_ratio(self, other: "Rect") -> float:
        """Intersection area as a share of the *smaller* rectangle: 1.0 = one lies completely on top of the other."""
        inter = self.intersection(other)
        smaller = min(self.area, other.area)
        return inter.area / smaller if inter is not None and smaller > 0 else 0.0

    def share_of(self, other: "Rect") -> float:
        """The part of ``other`` that this rectangle covers (0..1): how much of the thing underneath is hidden."""
        inter = self.intersection(other)
        return inter.area / other.area if inter is not None and other.area > 0 else 0.0

    def overshoot(self, bounds: "Rect") -> tuple[float, float, float, float]:
        """How far this rectangle sticks out past ``bounds`` on (left, top, right, bottom); 0 where it stays inside."""
        return (max(0.0, bounds.x0 - self.x0), max(0.0, bounds.y0 - self.y0), max(0.0, self.x1 - bounds.x1), max(0.0, self.y1 - bounds.y1))

    def moved_to(self, cx: float, cy: float) -> "Rect":
        return Rect(cx - self.width / 2, cy - self.height / 2, cx + self.width / 2, cy + self.height / 2)


FRAME = Rect(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class Margins:
    """Safe margins as shares of the frame."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def of(cls, settings: CaptionSettings, floor: float = 0.0) -> "Margins":
        """The project's caption margins, never smaller than ``floor`` (QC's minimum safe margin)."""
        return cls(max(settings.safe_margin_left, floor), max(settings.safe_margin_top, floor), max(settings.safe_margin_right, floor), max(settings.safe_margin_bottom, floor))

    @property
    def rect(self) -> Rect:
        return Rect(self.left, self.top, 1.0 - self.right, 1.0 - self.bottom)


def worst_overshoot(o: Iterable[float]) -> float:
    return max(o, default=0.0)


def fit_inside(rect: Rect, safe: Rect) -> tuple[float, float] | None:
    """The centre that puts ``rect`` inside ``safe`` with the least movement; None when it is too large to fit there at all (no position can help)."""
    if rect.width > safe.width + MARGIN_EPS or rect.height > safe.height + MARGIN_EPS:
        return None
    cx = min(max(rect.cx, safe.x0 + rect.width / 2), safe.x1 - rect.width / 2)
    cy = min(max(rect.cy, safe.y0 + rect.height / 2), safe.y1 - rect.height / 2)
    return cx, cy


# ---------------------------------------------------------------------------------------------- text size
def text_width_em(text: str, *, bold: bool = False, uppercase: bool = False) -> float:
    """Estimated width of one line, in em (font height)."""
    return len(text) * AVG_CHAR * (BOLD_WIDTH if bold else 1.0) * (UPPER_WIDTH if uppercase else 1.0)


# ---------------------------------------------------------------------------------------------- captions
@dataclass(frozen=True)
class CaptionLayout:
    rect: Rect  # the text block: what the safe margins apply to
    box: Rect  # the text block plus the background box's border: what hides the picture
    font_px: float
    lines: tuple[str, ...]
    boxed: bool
    anchor: str  # bottom | top | center | custom
    style: CaptionStyle


def caption_style(data: dict[str, Any], settings: CaptionSettings, styles: dict[str, CaptionStyle]) -> CaptionStyle:
    """The style the renderer draws this caption with (preset or project style, per-caption overrides, accessibility adjustments)."""
    return effective_style(style_for(styles, str(data.get("style_id", settings.style_id))), settings, data.get("style_overrides"))


def caption_lines(data: dict[str, Any]) -> list[str]:
    text = " ".join(str(data.get("text", "")).split())
    return [str(x) for x in (data.get("lines") or []) if str(x).strip()] or ([text] if text else [])


def caption_layout(data: dict[str, Any], settings: CaptionSettings, styles: dict[str, CaptionStyle], canvas: tuple[int, int]) -> CaptionLayout | None:
    """Estimated screen rectangle of a caption (None for a caption without text). Placement follows ``rendering/ass.py``'s ``_caption``."""
    lines = caption_lines(data)
    if not lines:
        return None
    W, H = max(1, canvas[0]), max(1, canvas[1])
    style = caption_style(data, settings, styles)
    em = max(10.0, style.size_rel * H)
    bold = style.weight == "bold"
    width = max(text_width_em(line, bold=bold, uppercase=style.uppercase) for line in lines) * em / W
    height = len(lines) * em * max(1.0, style.line_spacing) / H
    pos = str(data.get("position", settings.position))
    xy = data.get("position_xy") or []
    if pos == "top":
        anchor, cx, y0 = "top", 0.5, settings.safe_margin_top
    elif pos == "center":
        anchor, cx, y0 = "center", 0.5, 0.5 - height / 2
    elif pos == "custom" and len(xy) == 2:
        anchor, cx, y0 = "custom", float(xy[0]), float(xy[1]) - height / 2
    else:  # bottom, and any unknown name: the renderer falls back to the bottom anchor
        anchor, cx, y0 = "bottom", 0.5, 1.0 - settings.safe_margin_bottom - height
    rect = Rect(cx - width / 2, y0, cx + width / 2, y0 + height)
    boxed = style.background == "box"
    box = rect.expanded(CAPTION_BOX_PAD * em / W, CAPTION_BOX_PAD * em / H) if boxed else rect
    return CaptionLayout(rect, box, em, tuple(lines), boxed, anchor, style)


# ---------------------------------------------------------------------------------------------- text clips
@dataclass(frozen=True)
class TextLayout:
    rect: Rect
    box: Rect
    font_px: float  # the clip's ``size`` (px on the output frame)
    lines: tuple[str, ...]
    boxed: bool
    bold: bool
    left_aligned: bool


def text_layout(data: dict[str, Any], canvas: tuple[int, int]) -> TextLayout | None:
    """Estimated screen rectangle of a text clip (title plus subtitle). Placement follows ``rendering/ass.py``'s ``_text``: centred on ``position``, or left edge at it."""
    content = str(data.get("content", ""))
    if not content.strip():
        return None
    W, H = max(1, canvas[0]), max(1, canvas[1])
    try:
        em = max(1.0, float(data.get("size", 48)))
        pos = data.get("position", (0.5, 0.5))
        px, py = float(pos[0]), float(pos[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(em) and math.isfinite(px) and math.isfinite(py)):
        return None
    bold = data.get("emphasis") in BOLD_EMPHASIS
    lines = tuple(content.split("\n"))
    sub = str(data.get("subtitle", "") or "")
    width_em = max(text_width_em(line, bold=bold) for line in lines)
    height_em = len(lines) * TEXT_LINE_HEIGHT
    if sub:
        width_em = max(width_em, text_width_em(sub) * SUBTITLE_SCALE)
        height_em += TEXT_LINE_HEIGHT * SUBTITLE_SCALE
    w, h = width_em * em / W, height_em * em / H
    left = data.get("alignment") == "left"
    x0 = px if left else px - w / 2
    rect = Rect(x0, py - h / 2, x0 + w, py + h / 2)
    boxed = str(data.get("background", "none")) != "none"
    box = rect.expanded(TEXT_BOX_PAD * em / W, TEXT_BOX_PAD * em / H) if boxed else rect
    return TextLayout(rect, box, em, lines + ((sub,) if sub else ()), boxed, bold, left)


def text_color(data: dict[str, Any]) -> str:
    """The colour the renderer gives this text clip (it depends on the style only)."""
    return STYLE_COLORS.get(str(data.get("style")), "#ffffff")


# ---------------------------------------------------------------------------------------------- evidence graphics
def region_rect(clip: Clip) -> Rect | None:
    """The highlighted region of an evidence graphic (None when the clip draws nothing, like the renderer)."""
    hl = clip.effects.get("highlight") if isinstance(clip.effects, dict) else None
    if not hl:
        return None
    ev = clip.effects.get("evidence") or {}
    region = hl.get("region") or ev.get("region") or DEFAULT_REGION
    try:
        x, y, w, h = (float(v) for v in region)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return Rect(x, y, x + w, y + h)


def clip_rect(clip: Clip, settings: CaptionSettings, styles: dict[str, CaptionStyle], canvas: tuple[int, int]) -> Rect | None:
    """The rectangle a caption / text / evidence clip hides on screen (the box included), or None when it cannot be estimated."""
    if clip.kind == KIND_CAPTION:
        lay = caption_layout(clip.text or {}, settings, styles, canvas)
        return lay.box if lay else None
    if clip.kind == KIND_TEXT:
        lay2 = text_layout(clip.text or {}, canvas)
        return lay2.box if lay2 else None
    if clip.kind == KIND_GRAPHIC:
        return region_rect(clip)
    return None


# ---------------------------------------------------------------------------------------------- colour and contrast (WCAG 2.x)
_HEX = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


def parse_color(value: Any) -> tuple[float, float, float] | None:
    m = _HEX.match(str(value).strip()) if value is not None else None
    if not m:
        return None
    h = m.group(1)
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return float(int(h[0:2], 16)), float(int(h[2:4], 16)), float(int(h[4:6], 16))


def blend(fg: tuple[float, float, float], bg: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
    a = max(0.0, min(1.0, alpha))
    return tuple(f * a + b * (1.0 - a) for f, b in zip(fg, bg))  # type: ignore[return-value]


def luminance(rgb: tuple[float, float, float]) -> float:
    def lin(v: float) -> float:
        c = max(0.0, min(255.0, v)) / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def assumed_backdrop(*, box: tuple[tuple[float, float, float], float] | None = None, halo: tuple[tuple[float, float, float], float] | None = None,
                     base: tuple[float, float, float] = MID_GREY) -> tuple[tuple[float, float, float], str]:
    """The colour behind a glyph, and how well that is known ("box" | "halo" | "none").

    A background box of colour C and opacity a covers the picture: C*a + picture*(1-a). A halo (outline or shadow, colour C, weight w) darkens / lightens only the
    pixels around the glyph, so it counts with weight w. The picture itself is unknown to QC: it is assumed mid grey.
    """
    if box is not None:
        return blend(box[0], base, box[1]), "box"
    if halo is not None:
        return blend(halo[0], base, halo[1]), "halo"
    return base, "none"


def text_contrast(color: tuple[float, float, float], opacity: float, backdrop: tuple[float, float, float]) -> float:
    """Contrast of a glyph colour (drawn at ``opacity``) against its backdrop."""
    return contrast_ratio(blend(color, backdrop, opacity), backdrop)
