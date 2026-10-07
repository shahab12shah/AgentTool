"""SFXPlanner: sparing, explainable sound-effect recommendations.

Sound effects should improve communication, not make the video noisy: events come only from meaningful moments (a major transition, an
important figure, a warning, an evidence reveal, a text punch), are rate-limited, quiet, and each one carries scene, time, duration,
type, volume, reason and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.presentation.models import AudioSettings, SfxCategory

CATEGORY_VOLUME = {"IMPACT": 0.85, "WHOOSH": 0.6, "CLICK": 0.55, "NOTIFICATION": 0.6, "PAPER": 0.7, "CAMERA": 0.7, "TRANSITION": 0.6, "TICK": 0.5, "WARNING": 0.75,
                   "DIGITAL": 0.55, "AMBIENT": 0.4}
CATEGORY_MAX_DURATION = {"IMPACT": 1.5, "WHOOSH": 1.2, "CLICK": 0.5, "NOTIFICATION": 1.2, "PAPER": 1.5, "CAMERA": 1.0, "TRANSITION": 1.2, "TICK": 0.6, "WARNING": 1.8,
                         "DIGITAL": 1.2, "AMBIENT": 4.0}


@dataclass
class SfxMoment:
    scene_id: str
    time: float
    kind: str  # TRANSITION | NUMBER | WARNING | HEADLINE | EVIDENCE | PUNCH | GRAPHIC
    score: float  # importance of the moment 0..1
    detail: str = ""


@dataclass
class SfxEvent:
    scene_id: str
    time: float
    duration: float
    category: str
    volume: float
    reason: str
    confidence: float
    trigger: str
    asset_id: str | None = None  # None = no library asset of that category (nothing is placed)


MOMENT_TO_CATEGORY = {"TRANSITION": ("TRANSITION", "A subtle transition sound marks the change of topic."),
                      "NUMBER": ("TICK", "A quiet tick accompanies the figure appearing."),
                      "WARNING": ("IMPACT", "Subtle impact reinforces the warning graphic."),
                      "HEADLINE": ("IMPACT", "A soft impact marks the new section title."),
                      "EVIDENCE": ("PAPER", "A paper sound accompanies the document reveal."),
                      "PUNCH": ("IMPACT", "A subtle impact supports the key statement."),
                      "GRAPHIC": ("CLICK", "A light click accompanies the graphic appearing.")}


class SfxPlanner:
    def __init__(self, settings: AudioSettings, dynamic: bool = False) -> None:
        self.s, self.dynamic = settings, dynamic

    def plan(self, moments: list[SfxMoment], library: dict[str, list], duration: float) -> list[SfxEvent]:
        """``library``: category -> [(asset_id, asset_duration)]. Returns the chosen events, best moments first within the rate limit."""
        s = self.s
        budget = max(1, int(round(s.max_sfx_per_minute * duration / 60.0))) if moments else 0
        chosen: list[SfxMoment] = []
        for m in sorted(moments, key=lambda m: -m.score):
            if len(chosen) >= budget:
                break
            if any(abs(m.time - c.time) < s.min_sfx_gap for c in chosen):
                continue
            chosen.append(m)
        events: list[SfxEvent] = []
        for m in sorted(chosen, key=lambda m: m.time):
            cat, why = MOMENT_TO_CATEGORY.get(m.kind, ("CLICK", "A light sound supports the graphic."))
            if m.kind == "TRANSITION" and self.dynamic and library.get("WHOOSH"):
                cat = "WHOOSH"
            pool = library.get(cat) or []
            asset_id, adur = (pool[0][0], pool[0][1]) if pool else (None, None)
            dur = min(float(adur or CATEGORY_MAX_DURATION.get(cat, 1.0)), CATEGORY_MAX_DURATION.get(cat, 1.5))
            vol = round(min(0.5, s.sfx_level * CATEGORY_VOLUME.get(cat, 0.6)), 3)  # quiet by design; never above the voice priority limit
            conf = round(min(95.0, 60.0 + m.score * 35.0), 1) if asset_id else 55.0
            events.append(SfxEvent(m.scene_id, round(m.time, 3), round(dur, 3), cat, vol, why + (f" ({m.detail})" if m.detail else ""), conf, m.kind, asset_id))
        return events


_ = SfxCategory
