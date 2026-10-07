"""Human-readable reading of a style profile: one-paragraph summary and abstract editing recommendations.

Everything is derived from aggregate numbers and classes; low-confidence readings are said to be estimates, never stated as fact.
"""

from __future__ import annotations

from app.reference.style_model import DIMENSION_LABELS, DIMENSIONS, ReferenceStyleProfile


def _hedge(profile: ReferenceStyleProfile, dim: str) -> str:
    return " (estimate)" if profile.is_available(dim) and profile.dimension_confidence(dim) < 0.5 else ""


def human_summary(profile: ReferenceStyleProfile) -> str:
    """e.g. "Fast-paced editing with high visual density, subtle motion, frequent bold captions, minimal transitions, continuous music…"."""
    c = profile.categories
    parts: list[str] = []

    def add(dim: str, text: str) -> None:
        if profile.is_available(dim):
            parts.append(text + _hedge(profile, dim))

    add("pacing", f"{c.get('pacing', 'Moderate').lower()}-paced editing" + (f" (a cut about every {profile.features.average_shot_duration:.1f} s)" if profile.features.average_shot_duration > 0 else ""))
    add("visual_density", f"{c.get('visual_density', 'medium').lower()} visual density")
    add("motion_intensity", f"{c.get('motion_intensity', 'minimal').lower()} motion")
    if profile.is_available("caption_density"):
        cap = profile.caption_style
        parts.append((f"{c.get('caption_density', 'no').lower()} captions" + (f" in a {cap.style_class.lower()} style" if cap.caption_present else "")) + _hedge(profile, "caption_density"))
    add("text_density", f"{c.get('text_density', 'no').lower()} text graphics")
    add("transition_frequency", f"{c.get('transition_frequency', 'minimal').lower()} transitions")
    if profile.is_available("music_presence"):
        parts.append(f"{c.get('music_presence', 'no').lower()} music" + (" ducking under speech" if profile.audio.music_ducking_strength > 0.4 and profile.audio.music_presence > 0.2 else "") + _hedge(profile, "music_presence"))
    if profile.is_available("sfx_frequency"):
        parts.append(f"{c.get('sfx_frequency', 'no').lower()} sound effects" + _hedge(profile, "sfx_frequency"))
    if not parts:
        return "The reference could not be measured, so no style summary is available."
    text = parts[0][0].upper() + parts[0][1:] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "") + "."
    if profile.unavailable:
        text += " Not measured: " + ", ".join(DIMENSION_LABELS[d] for d in DIMENSIONS if d in profile.unavailable) + "."
    return text


def recommendations(profile: ReferenceStyleProfile) -> list[str]:
    """Abstract suggestions for the user's own edit — parameters and tendencies, never content."""
    f, out = profile.features, []
    if profile.is_available("pacing") and f.average_shot_duration > 0:
        out.append(f"Pacing: aim for shots around {f.median_shot_duration or f.average_shot_duration:.1f} s where the narration allows; slow down for documents and data that need reading time.")
    if profile.is_available("motion_intensity"):
        out.append(f"Motion: {profile.categories.get('motion_intensity', 'Subtle').lower()} camera movement" + (f" with zooms about every {60.0 / f.zoom_events_per_minute:.0f} s" if f.zoom_events_per_minute > 0.5 else "") + ".")
    if profile.is_available("caption_density") and profile.caption_style.caption_present:
        cs = profile.caption_style
        out.append(f"Captions: {cs.style_class.lower()} style, about {cs.average_words_per_caption:.0f} words per caption in the {cs.caption_position} of the frame" + (", with highlighted key words" if cs.caption_emphasis_rate > 0.25 else "") + ".")
    if profile.is_available("text_density") and f.text_events_per_minute > 0.5:
        out.append(f"Text graphics: about {f.text_events_per_minute:.0f} on-screen text elements per minute (headlines, figures, lower thirds) — written from your own content.")
    if profile.is_available("music_presence") and profile.audio.has_audio:
        out.append(f"Audio: {profile.categories.get('music_presence', 'minimal').lower()} music with {profile.categories.get('sfx_frequency', 'subtle').lower()} sound effects" + (", ducked clearly under the voice" if profile.audio.music_ducking_strength > 0.4 else "") + ".")
    if profile.hook.intensity > 0.25:
        out.append("Opening: the first seconds are edited more intensely than the rest — tighter shots and stronger emphasis.")
    for d in DIMENSIONS:
        if d in profile.unavailable:
            out.append(f"{DIMENSION_LABELS[d]} could not be measured; it will not be changed by this style.")
    return out
