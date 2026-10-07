"""Caption style presets and the accessibility adjustments applied on top of them."""

from __future__ import annotations

from dataclasses import replace

from app.presentation.models import CaptionSettings, CaptionStyle

EMPH = {"MONEY": "POP", "PERCENTAGE": "POP", "NUMBER": "COLOR_CHANGE", "DATE": "BACKGROUND_BOX", "DEADLINE": "BACKGROUND_BOX", "WARNING": "BOLD",
        "PERSON": "UNDERLINE", "ORGANIZATION": "COLOR_CHANGE", "LOCATION": "UNDERLINE", "PRODUCT": "COLOR_CHANGE", "PROCESS": "UNDERLINE",
        "CLAIM": "BOLD", "CONCEPT": "COLOR_CHANGE"}

PRESETS: dict[str, CaptionStyle] = {
    "professional": CaptionStyle("professional", "Professional", size_rel=0.050, weight="bold", background="box", background_opacity=0.55, shadow=False,
                                 emphasis=dict(EMPH)),
    "clean": CaptionStyle("clean", "Clean", size_rel=0.048, weight="normal", background="none", shadow=True, outline_width=0.0, highlight_color="#7FD1FF",
                          emphasis={**EMPH, "MONEY": "COLOR_CHANGE", "PERCENTAGE": "COLOR_CHANGE", "DATE": "UNDERLINE", "DEADLINE": "UNDERLINE"}),
    "bold": CaptionStyle("bold", "Bold", size_rel=0.062, weight="bold", background="none", shadow=True, outline_width=0.08, uppercase=True,
                         highlight_color="#FFD23F", emphasis={**EMPH, "NUMBER": "POP", "DATE": "POP"}),
    "news": CaptionStyle("news", "News", size_rel=0.046, weight="bold", background="box", background_color="#10243E", background_opacity=0.9, shadow=False,
                         highlight_color="#FFFFFF", emphasis={**EMPH, "MONEY": "BACKGROUND_BOX", "PERCENTAGE": "BACKGROUND_BOX"}),
    "documentary": CaptionStyle("documentary", "Documentary", font="Serif", size_rel=0.045, weight="normal", background="none", shadow=True,
                                highlight_color="#E8D9A8", emphasis={**EMPH, "MONEY": "COLOR_CHANGE", "PERCENTAGE": "COLOR_CHANGE", "DATE": "UNDERLINE"}),
    "minimal": CaptionStyle("minimal", "Minimal", size_rel=0.040, weight="normal", background="none", shadow=True, opacity=0.95, highlight_color="#FFFFFF",
                            emphasis={k: "BOLD" for k in EMPH}),
}


def style_for(styles: dict[str, CaptionStyle], style_id: str) -> CaptionStyle:
    """A project-level (edited or custom) style wins over the built-in preset of the same id."""
    return styles.get(style_id) or PRESETS.get(style_id) or PRESETS["professional"]


def effective_style(base: CaptionStyle, settings: CaptionSettings, overrides: dict | None = None) -> CaptionStyle:
    """Base style + per-caption overrides + accessibility (large text, strong contrast)."""
    st = replace(base)
    for k, v in (overrides or {}).items():
        if hasattr(st, k) and k != "style_id":
            setattr(st, k, v)
    if settings.uppercase:
        st.uppercase = True
    if settings.large_text:
        st.size_rel = round(st.size_rel * 1.3, 4)
    if settings.high_contrast:
        st.color, st.background, st.background_color, st.background_opacity = "#FFFFFF", "box", "#000000", 0.85
        st.highlight_color, st.opacity, st.shadow = "#FFE45C", 1.0, False
    return st
