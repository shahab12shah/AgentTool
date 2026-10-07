"""MotionGraphicsEngine: plans number/date graphics, lower thirds, headlines, warnings and evidence tools for a scene.

Text content only ever comes from the narration/script (the Phase 4 ``TextPlanner`` enforces that); a section headline is derived from
the scene topic and is marked as such. Timing follows word timestamps, animation follows the motion-intensity level, and nothing is
generated for data that is not there (no fabricated charts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.editing.context import EditingContext, SceneContext
from app.editing.models import DecisionType, Emphasis, SceneEditingBrief, TextGraphic, TextStyle
from app.editing.planners import TextPlanner
from app.editing.presets import preset_for
from app.presentation.animation import default_animation, intensity_level
from app.presentation.models import PresentationType

TRIGGERS = ("deadline", "due", "by", "until", "before", "on", "expires", "expire", "starting", "from")
VARIANT_OF = {TextStyle.NUMBER_CARD.value: "NUMBER", TextStyle.DATE.value: "DATE", TextStyle.LOWER_THIRD.value: "LOWER_THIRD", TextStyle.ENTITY_NAME.value: "LOWER_THIRD",
              TextStyle.HEADLINE.value: "HEADLINE", TextStyle.WARNING.value: "WARNING", TextStyle.LABEL.value: "TEXT", TextStyle.LOCATION.value: "TEXT"}
TYPE_OF = {"NUMBER": PresentationType.NUMBER_GRAPHIC, "DATE": PresentationType.DATE_GRAPHIC, "LOWER_THIRD": PresentationType.LOWER_THIRD, "HEADLINE": PresentationType.HEADLINE,
           "WARNING": PresentationType.HEADLINE, "TEXT": PresentationType.TEXT_GRAPHIC}
COUNTER_RE = re.compile(r"^([^\d]*)(\d[\d,]*\.?\d*)(.*)$")
EVIDENCE_TOOLS = ("FOCUS_BOX", "HIGHLIGHT", "UNDERLINE", "POINTER", "MAGNIFY", "CROP", "DIM")


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9%$]+", " ", t.lower()).strip()


@dataclass
class PlannedGraphic:
    slot: str
    variant: str  # NUMBER | DATE | LOWER_THIRD | HEADLINE | WARNING | TEXT
    graphic: TextGraphic
    type: PresentationType
    reason: str
    confidence: float
    animation: dict
    counter: dict | None = None
    derived: bool = False  # text not taken verbatim from the narration (section headline)
    title: str = ""
    subtitle: str = ""


@dataclass
class EvidenceTool:
    tool: str
    dim: bool
    animation: dict
    reason: str


@dataclass
class GraphicsPlan:
    scene_id: str
    graphics: list[PlannedGraphic] = field(default_factory=list)
    evidence: EvidenceTool | None = None
    notes: list[str] = field(default_factory=list)


def counter_spec(content: str) -> dict | None:
    """A counter that counts up to the *actual* figure in ``content`` (None when the text is not a plain number)."""
    m = COUNTER_RE.match(content.strip())
    if not m:
        return None
    pre, num, suf = m.groups()
    try:
        to = float(num.replace(",", ""))
    except ValueError:
        return None
    dec = len(num.split(".")[1]) if "." in num else 0
    return {"from": 0.0, "to": to, "decimals": dec, "prefix": pre, "suffix": suf, "thousands": "," in num}


class GraphicsPlanner:
    def __init__(self, ctx: EditingContext, reduced_motion: bool = False) -> None:
        self.ctx, self.reduced = ctx, reduced_motion
        self.settings = ctx.settings
        self.level = intensity_level(ctx.settings.motion_intensity)
        self.text = TextPlanner(preset_for(ctx.settings), ctx.settings)

    # ------------------------------------------------------------------ one scene
    def plan_scene(self, sc: SceneContext, brief: SceneEditingBrief) -> GraphicsPlan:
        plan = GraphicsPlan(sc.scene.id)
        taken: list[tuple[float, float]] = []
        n = 0
        if self.settings.text_emphasis or self.settings.number_emphasis:
            for pt in self.text.plan(sc, brief, [], 1.0):
                g = pt.graphic
                variant = VARIANT_OF.get(g.style, "TEXT")
                if variant == "DATE":
                    g = self._align_date(g, sc)
                counter = counter_spec(g.content) if (variant == "NUMBER" and self.level == "HIGH" and not self.reduced) else None
                anim = default_animation(variant, self.level, self.reduced, counter is not None)
                slot = f"p5:{'number' if variant == 'NUMBER' else variant.lower()}:{n}"
                n += 1
                plan.graphics.append(PlannedGraphic(slot, variant, g, TYPE_OF.get(variant, PresentationType.TEXT_GRAPHIC), pt.reason, pt.confidence, anim, counter,
                                                    title=g.content if variant == "LOWER_THIRD" else ""))
                taken.append((g.start, g.start + g.duration))
        head = self._headline(sc, taken)
        if head is not None:
            plan.graphics.append(head)
        plan.notes += self._chart_notes(sc)
        return plan

    # ------------------------------------------------------------------ word-timestamp alignment
    @staticmethod
    def _align_date(g: TextGraphic, sc: SceneContext) -> TextGraphic:
        """Start the date when the narration says its trigger ("the deadline is October 15" -> at "deadline")."""
        first = _norm(g.content).split()[0] if _norm(g.content) else ""
        words = sc.words
        idx = next((i for i, w in enumerate(words) if _norm(w.text) == first and abs(w.start - (g.start + 0.15)) < 1.2), None)
        if idx is None:
            return g
        for back in range(1, 5):
            if idx - back < 0:
                break
            w = words[idx - back]
            if _norm(w.text) in TRIGGERS:
                end_word = words[min(len(words) - 1, idx + len(g.content.split()) - 1)]
                start = max(sc.scene.start, w.start - 0.05)
                g = TextGraphic(**{**g.to_dict(), "start": round(start, 3), "duration": round(min(sc.scene.end - start, max(g.duration, end_word.end - start + 1.2)), 3)})
                break
        return g

    def _headline(self, sc: SceneContext, taken: list[tuple[float, float]]) -> PlannedGraphic | None:
        s = sc.scene
        if not (sc.starts_section and self.settings.text_emphasis and s.topic.strip()):
            return None
        title = " ".join(s.topic.split()[:6]).upper()
        start, dur = s.start + 0.2, min(2.6, max(1.2, s.duration - 0.4))
        if any(a < start + dur and b > start for a, b in taken):
            return None
        g = TextGraphic(f"{s.id}_head", title, round(start, 3), round(dur, 3), (0.5, 0.14), TextStyle.HEADLINE.value, Emphasis.SCREEN_CENTER_TEXT.value, "reveal", 0.6,
                        s.id, "scene_topic", size=64, background="none")
        return PlannedGraphic("p5:headline:0", "HEADLINE", g, PresentationType.HEADLINE, "Section title for a new topic (taken from the scene topic, not the narration).",
                              70.0, default_animation("HEADLINE", self.level, self.reduced), derived=True, title=title)

    # ------------------------------------------------------------------ evidence + data
    def evidence_tool(self, sc: SceneContext) -> EvidenceTool:
        if self.level == "LOW" or self.reduced:
            return EvidenceTool("HIGHLIGHT", False, default_animation("EVIDENCE", "LOW", True), "Gentle highlight; surrounding area stays visible.")
        return EvidenceTool("FOCUS_BOX", True, default_animation("EVIDENCE", self.level, False), "Focus box with the surrounding area dimmed; the document is never altered.")

    @staticmethod
    def _chart_notes(sc: SceneContext) -> list[str]:
        if sc.visual_type != "DATA":
            return []
        a = sc.asset
        if a is None:
            return ["No chart or data visual is available: no chart was generated. Use a text statistic, an evidence screenshot or add a chart manually."]
        return ["A real data visual is assigned: only region highlights and number callouts are offered; no data is generated."]
