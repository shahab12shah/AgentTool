"""TextChecker: is the on-screen text and are the evidence graphics readable, in the right place, in the right amount - and does an on-screen value match the script?

Two jobs, one pass over each scene:

* TEXT (spec 16): text that is cut off or outside the safe area, too small or too brief to read (numbers and dates are held to a stricter standard), text on top of
  other text or on top of an evidence box, the same text twice in a row, text outside its scene or hanging around, a small label bigger than a headline, too many
  text elements in a scene. Captions are the caption checker's: a text clip that collides with a caption is reported there (``caption.collision``), not here.
* FACT_REVIEW: QC does not fact-check and never corrects text. It compares every number, date and price on screen with what the scene's narration and script say,
  and when a value on screen is not among them it asks for a *review* (``text.value_mismatch``): "Potential inconsistency detected ... Review recommended". No fix is
  offered except opening the text for review. The same applies to a counter that ends on a different figure than the text it shows, and, as a notice, to an evidence
  highlight in a scene whose narration makes no claim that needs evidence.

Numbers and dates are extracted with the analysis package's own extractor (``analysis.numbers.extract_numbers`` over ``analysis.prep.build_tokens``), the same code that
produced ``Scene.numbers`` from the narration, so "forty two percent" spoken and "42%" on screen compare equal. Geometry comes from ``qc.geometry``.

Scene-local: a clip is analysed in the scene its midpoint lies in; the pairs it can collide with are the clips within 0.5 s of that scene, which is what
``QCContext.scene_signature`` hashes. The scene's numbers, claims and script text are not part of that signature, so this checker adds them to its cache keys.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.analysis.models import NumericMention
from app.analysis.numbers import MONTHS, extract_numbers
from app.analysis.prep import build_tokens
from app.presentation.graphics import counter_spec
from app.qc import fix_catalog
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.geometry import (FRAME, MARGIN_EPS, STYLE_COLORS, Margins, Rect, TextLayout, assumed_backdrop, fit_inside, parse_color, region_rect, text_contrast, text_layout)
from app.qc.issue_model import QCCategory, QCFixSpec, QCIssue
from app.qc.severity import Severity
from app.timeline.clip import Clip
from app.timeline.track import Track
from app.transcription.models import Word

CAT = QCCategory.TEXT
FACT = QCCategory.FACT_REVIEW
SIDES = ("left", "top", "right", "bottom")
SHADOW_WEIGHT = 0.6  # the renderer draws unboxed text with a drop shadow: it darkens the picture around the glyphs
BOX_COLOR, BOX_OPACITY = (0.0, 0.0, 0.0), 170.0 / 255.0  # the renderer's text box: black at 170/255
IMPORTANT_STYLES = ("NUMBER_CARD", "DATE", "WARNING")
NAME_STYLES = ("LOWER_THIRD", "ENTITY_NAME", "LOCATION")  # text that carries a person / place / organisation name
# visual priority of a text style: a style must never be drawn larger than one that ranks above it
RANK = {"HEADLINE": 3, "NUMBER_CARD": 3, "WARNING": 3, "DATE": 2, "COMPARISON": 2, "DEFINITION": 2, "LOWER_THIRD": 1, "ENTITY_NAME": 1, "LABEL": 1, "LOCATION": 1}
EVIDENCE_INTENTS = ("EVIDENCE", "DATA")
TEXT_COLORS: dict[str, str] = dict(STYLE_COLORS)  # style -> colour the renderer draws it in; a module attribute so a colour override (or a test) can replace entries
TITLE_WORD = re.compile(r"[A-Z][A-Za-z'’-]+")
NAME_FILLER = {"The", "A", "An", "Of", "And", "In", "On", "At", "For", "To", "By", "Mr", "Mrs", "Ms", "Dr"}
DIGITS = re.compile(r"\d[\d,]*\.?\d*")
ORDINAL_WORDS = {w: i for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(), 1)}
ORDINAL_WORDS.update({"twenty-first": 21, "twenty-second": 22, "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25, "twenty-sixth": 26, "twenty-seventh": 27,
                      "twenty-eighth": 28, "twenty-ninth": 29, "thirtieth": 30, "thirty-first": 31})


def _short(text: str, n: int = 40) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%$]+", " ", str(text).lower()).strip()


def _opt(ctx: QCContext, name: str, default: float) -> float:
    """An optional threshold of a (future) ``text`` settings section; the default applies until the contract defines it."""
    return float(getattr(getattr(ctx.settings, "text", None), name, default))


@dataclass
class _Cfg:
    min_margin: float
    min_size: float  # relative text height (size / frame height) below which any text is hard to read
    min_size_important: float
    min_contrast: float
    per_minute: float
    min_per_scene: int
    outside_seconds: float
    hold_floor: float
    hold_factor: float
    duplicate_window: float
    overlap_seconds: float
    overlap_share: float
    hierarchy_ratio: float
    confidence: float

    @classmethod
    def of(cls, ctx: QCContext) -> "_Cfg":
        c = ctx.settings.caption
        floor = float(c.min_relative_font)
        return cls(float(c.min_safe_margin), _opt(ctx, "min_relative_size", floor), _opt(ctx, "min_relative_size_important", max(floor * 1.4, 0.04)), float(c.min_contrast),
                   _opt(ctx, "max_per_minute", 14.0), int(_opt(ctx, "min_allowed_per_scene", 3)), _opt(ctx, "outside_scene_seconds", 0.5), _opt(ctx, "hold_floor_seconds", 6.0),
                   _opt(ctx, "hold_factor", 3.0), _opt(ctx, "duplicate_window_seconds", 10.0), _opt(ctx, "overlap_seconds", 0.3), _opt(ctx, "overlap_share", 0.15),
                   _opt(ctx, "hierarchy_ratio", 1.1), min(80.0, _opt(ctx, "mismatch_confidence", 72.0)))


@dataclass
class _Item:
    """One text clip, or one evidence graphic."""

    track: Track
    clip: Clip
    data: dict[str, Any]
    scene_id: str | None
    layout: TextLayout | None  # text clips
    region: Rect | None  # evidence graphics
    content: str
    style: str
    mentions: list[NumericMention]
    derived: bool

    @property
    def start(self) -> float:
        return self.clip.timeline_start

    @property
    def end(self) -> float:
        return self.clip.timeline_end

    @property
    def is_text(self) -> bool:
        return self.layout is not None or self.region is None

    @property
    def important(self) -> bool:
        return self.style in IMPORTANT_STYLES or str(self.data.get("variant", "")) in ("NUMBER", "DATE", "WARNING") or bool(self.mentions)

    @property
    def label(self) -> str:
        return f"“{_short(self.content)}”" if self.is_text else "evidence highlight"

    @property
    def rect(self) -> Rect | None:
        return self.layout.rect if self.layout is not None else self.region


@dataclass
class _Spoken:
    """What a scene's narration and script say, in comparable form."""

    values: set[float]
    months: set[int]
    tokens: set[str]
    has_narration: bool


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
WRITTEN_SCALE = re.compile(r"([$€£]?)(\d[\d,]*(?:\.\d+)?)\s+(thousand|million|billion|trillion)\b", re.IGNORECASE)


def _scaled(m: "re.Match[str]") -> str:
    value = float(m.group(2).replace(",", "")) * SCALE_WORDS[m.group(3).lower()]
    return m.group(1) + (str(int(value)) if value.is_integer() else str(value))


def numeric_mentions(text: str) -> list[NumericMention]:
    """Numbers, prices, percentages, years and dates in ``text``: the analysis package's extractor, which also understands spoken numbers ("forty two percent").

    A figure written in digits followed by a scale word ("1.2 million") is merged into one number first: the extractor's spoken-number pass would otherwise read the
    scale word on its own as a million.
    """
    words = [Word(f"t{i}", t, 0.0, 0.0, None) for i, t in enumerate(WRITTEN_SCALE.sub(_scaled, str(text)).split())]
    return extract_numbers(build_tokens(words), "") if words else []


def _date_parts(text: str) -> tuple[int | None, list[float]]:
    """(month number, numbers) of a month-based date mention: "April 15th 2027" -> (4, [15.0, 2027.0])."""
    month = next((MONTHS[t] for t in re.findall(r"[a-z]+", text.lower()) if t in MONTHS), None)
    return month, [float(d.replace(",", "")) for d in DIGITS.findall(text)]


def spoken_of(scene: Any) -> _Spoken:
    """Every value a scene's narration, script and number list contain. Deliberately generous: a value counts as spoken when it appears in any of them."""
    values: set[float] = set()
    months: set[int] = set()
    tokens: set[str] = set()
    text = f"{scene.narration} {scene.script_text}".strip()
    for n in scene.numbers:
        if n.value is not None:
            values.add(float(n.value))
        mo, nums = _date_parts(n.text) if n.value is None else (None, [])
        if mo:
            months.add(mo)
        values.update(nums)
    for m in numeric_mentions(text):
        if m.value is not None:
            values.add(float(m.value))
        else:
            mo, nums = _date_parts(m.text)
            if mo:
                months.add(mo)
            values.update(nums)
    for d in DIGITS.findall(text):  # every figure written as digits, however the extractor classified it
        try:
            values.add(float(d.replace(",", "")))
        except ValueError:
            continue
    for tok in re.findall(r"[a-z][a-z'’-]*", text.lower()):
        tokens.add(tok.replace("’", "'"))
        if tok in MONTHS:
            months.add(MONTHS[tok])
        if tok in ORDINAL_WORDS:
            values.add(float(ORDINAL_WORDS[tok]))
    for src in (scene.topic, *(e.text for e in scene.entities)):
        tokens.update(re.findall(r"[a-z][a-z'’-]*", str(src).lower()))
    return _Spoken(values, months, tokens, bool(scene.narration.strip() or scene.script_text.strip()))


def judge_mention(m: NumericMention, sp: _Spoken) -> str | None:
    """None when the on-screen value is among the spoken ones; "differs" when other values are spoken but not this one; "absent" when nothing comparable is spoken."""
    if m.value is not None:
        if any(_same(m.value, v) for v in sp.values):
            return None
        return "differs" if sp.values else "absent"
    month, nums = _date_parts(m.text)
    if month is not None and month not in sp.months:
        return "differs" if sp.months else "absent"
    if any(not any(_same(n, v) for v in sp.values) for n in nums):
        return "differs" if sp.values else "absent"
    return None


def name_runs(content: str) -> list[list[str]]:
    """Runs of capitalised words in text that is not set in capitals ("Jane Smith, Treasury Secretary" -> [["Jane", "Smith"], ["Treasury", "Secretary"]])."""
    if content.isupper():
        return []  # in capitals every word looks like a name: nothing can be concluded
    runs: list[list[str]] = []
    for part in re.split(r"[,;:/()\n]+", content):  # punctuation ends a name: "Jane Smith, Treasury Secretary" is two
        cur: list[str] = []
        for tok in part.split():
            t = tok.strip(".!?\"'")
            if TITLE_WORD.fullmatch(t) and t not in NAME_FILLER:
                cur.append(t)
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
    return runs


# ---------------------------------------------------------------------------------------------- the checker
class TextChecker(BaseChecker):
    id = "text"
    label = "Text & graphics"
    categories = (QCCategory.TEXT, QCCategory.FACT_REVIEW)
    domains = ("timeline", "scenes", "captions")  # captions: the project's safe margins
    settings_sections = ("caption", "text")
    scene_local = True
    version = "1"

    # ------------------------------------------------------------------ cache keys: what the fact review reads beyond the shared fingerprints
    @staticmethod
    def _scene_facts(scene: Any) -> list[Any]:
        return [scene.id, scene.script_text, [[n.text, n.value, n.kind.value] for n in scene.numbers], [[c.text, c.requires_evidence] for c in scene.claims], [e.text for e in scene.entities]]

    def input_hash(self, ctx: QCContext) -> str:
        return sha(super().input_hash(ctx), [self._scene_facts(s) for s in ctx.scenes])

    def scene_input_hash(self, ctx: QCContext, scene_id: str) -> str:
        s = ctx.scene(scene_id)
        return sha(super().scene_input_hash(ctx, scene_id), self._scene_facts(s) if s else None)

    # ------------------------------------------------------------------ run
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = _Cfg.of(ctx)
        items = self._items(ctx)
        scenes = ctx.target_scenes()
        texts_checked = graphics_checked = values_checked = 0
        for n, scene in enumerate(scenes):
            ctx.check_cancel()
            report(n / max(1, len(scenes)), f"Scene {scene.label}")
            own = [i for i in items if i.scene_id == scene.id]
            window = [i for i in items if i.end > scene.start - 0.5 and i.start < scene.end + 0.5]
            sp = spoken_of(scene)
            own_text = [i for i in own if i.is_text]
            own_gfx = [i for i in own if not i.is_text]
            texts_checked += len(own_text)
            graphics_checked += len(own_gfx)
            out.metrics[f"{scene.id}.texts_checked"] = len(own_text)
            out.metrics[f"{scene.id}.graphics_checked"] = len(own_gfx)
            for it in own:
                if it.is_text:
                    self._geometry(ctx, cfg, it, out.issues)
                    self._readability(ctx, cfg, it, out.issues)
                    self._timing(ctx, cfg, scene, it, out.issues)
                    values_checked += self._facts(ctx, cfg, it, sp, out.issues)
            self._pairs(ctx, cfg, own, window, out.issues)
            self._hierarchy(ctx, cfg, own_text, out.issues)
            self._excessive(ctx, cfg, scene, own_text, out.issues)
            self._evidence(ctx, scene, own_gfx, out.issues)
        out.metrics.update({"texts_checked": texts_checked, "graphics_checked": graphics_checked, "values_checked": values_checked})
        out.notes.append(f"{texts_checked} text clip(s) and {graphics_checked} evidence graphic(s) checked in {len(scenes)} scene(s)")
        report(1.0, "")
        return out

    # ------------------------------------------------------------------ gathering
    def _items(self, ctx: QCContext) -> list[_Item]:
        scenes = ctx.scenes
        canvas = ctx.canvas

        def host(c: Clip) -> str | None:
            if not scenes:
                return None
            mid = c.timeline_start + c.duration / 2
            s = ctx.scene_at(mid) or min(scenes, key=lambda sc: max(sc.start - mid, mid - sc.end, 0.0))  # a clip beyond either end belongs to the nearest scene
            return s.id

        out: list[_Item] = []
        for track, clip in ctx.text_clips():
            if track.hidden or clip.duration <= 0 or not isinstance(clip.text, dict):
                continue
            content = str(clip.text.get("content", "")).strip()
            if not content:
                continue  # empty text is the timeline checker's finding
            layout = text_layout(clip.text, canvas)
            derived = bool(clip.text.get("derived")) or clip.text.get("source_ref") == "scene_topic"
            out.append(_Item(track, clip, clip.text, host(clip), layout, None, content, str(clip.text.get("style", "")), [] if derived else numeric_mentions(content), derived))
        for track, clip in ctx.graphic_clips():
            region = region_rect(clip)
            if track.hidden or clip.duration <= 0 or region is None:
                continue
            out.append(_Item(track, clip, {}, host(clip), None, region, "", "EVIDENCE", [], False))
        out.sort(key=lambda i: (i.start, i.clip.id))
        return out

    def _issue(self, ctx: QCContext, it: _Item, code: str, severity: Severity, title: str, *, category: QCCategory = CAT, **kw: Any) -> QCIssue:
        return self.issue(code, category, severity, title, scene_id=it.scene_id, clip=it.clip, track=it.track, ctx=ctx, affected=kw.pop("affected", [f"Text {it.label}"]), **kw)

    @staticmethod
    def _open(it: _Item, text: str = "Show the text on the timeline") -> QCFixSpec:
        return fix_catalog.navigate("open.timeline", text, clip_id=it.clip.id)

    # ------------------------------------------------------------------ position: clipped, too big for the safe area, in the margin
    def _geometry(self, ctx: QCContext, cfg: _Cfg, it: _Item, issues: list[QCIssue]) -> None:
        lay = it.layout
        if lay is None:
            return
        safe = Margins.of(ctx.project.caption_settings, cfg.min_margin).rect
        off = lay.rect.overshoot(FRAME)
        over = lay.rect.overshoot(safe)
        if max(over) <= MARGIN_EPS:
            return
        clipped = max(off) > MARGIN_EPS
        edges = [n for n, v in zip(SIDES, off if clipped else over) if v > MARGIN_EPS]
        depth = max(off) if clipped else max(over)
        fits = fit_inside(lay.rect, safe) is not None
        sig = f"{'|'.join(edges)}:{round(depth, 2)}"
        m = {"overshoot": round(depth, 4), "edges": edges, "width": round(lay.rect.width, 3), "height": round(lay.rect.height, 3)}
        if clipped:
            issues.append(self._issue(
                ctx, it, "text.clipped", Severity.ERROR, "Text is cut off by the edge of the frame",
                description=f"The text {it.label} reaches {depth * 100:.1f}% of the frame past the {'/'.join(edges)} edge, so part of it is not visible.",
                why="Text that leaves the frame is cut off in the final video.", current=f"{depth * 100:.1f}% outside the frame", recommended="inside the safe margins",
                suggested_fix="Move the text inside the frame or make it smaller.", fix=self._open(it), confidence=90.0 if depth > 0.02 else 75.0, viewer_impact=0.9, signature=sig, metrics=m))
        elif not fits:
            issues.append(self._issue(
                ctx, it, "text.overflow", Severity.WARNING, "Text is larger than the safe area",
                description=f"The text {it.label} needs about {lay.rect.width * 100:.0f}% x {lay.rect.height * 100:.0f}% of the frame; the safe area is {safe.width * 100:.0f}% x {safe.height * 100:.0f}%.",
                why="Text outside the safe margins can be cut off or look crowded against the edge on some screens.", current=f"{depth * 100:.1f}% inside the {'/'.join(edges)} margin",
                recommended="text that fits inside the safe margins", suggested_fix="Shorten the text or reduce its size.", fix=self._open(it), confidence=80.0, viewer_impact=0.5, signature=sig, metrics=m))
        else:
            issues.append(self._issue(
                ctx, it, "text.safe_area", Severity.WARNING if depth >= 0.015 else Severity.NOTICE, "Text is inside the safe margin",
                description=f"The text {it.label} reaches {depth * 100:.1f}% of the frame into the {'/'.join(edges)} safe margin.",
                why="Text close to the frame edge can be cut off or covered by the interface of some players.", current=f"{depth * 100:.1f}% inside the margin", recommended="inside the safe margins",
                suggested_fix="Move the text inside the safe area.", fix=self._open(it), confidence=80.0, viewer_impact=0.35, signature=sig, metrics=m))

    # ------------------------------------------------------------------ readability: size, contrast, time on screen
    def _readability(self, ctx: QCContext, cfg: _Cfg, it: _Item, issues: list[QCIssue]) -> None:
        lay = it.layout
        if lay is None:
            return
        H = max(1, ctx.canvas[1])
        rel = lay.font_px / H
        need_size = cfg.min_size_important if it.important else cfg.min_size
        if rel < need_size - 1e-9:
            issues.append(self._issue(
                ctx, it, "text.unreadable", Severity.ERROR if rel < need_size * 0.5 else Severity.WARNING if it.important else Severity.NOTICE, "Text is small",
                description=f"The text {it.label} is {rel * 100:.1f}% of the frame height ({lay.font_px:.0f} px); " + ("numbers, dates and warnings need" if it.important else "text needs")
                + f" at least {need_size * 100:.1f}%.", why="Small text cannot be read on a phone or from a distance." + (" Figures and dates are the part of the video viewers remember." if it.important else ""),
                current=f"{rel * 100:.1f}% of the frame height", recommended=f"at least {need_size * 100:.1f}%", suggested_fix="Increase the text size.", fix=self._open(it), viewer_impact=0.7 if it.important else 0.4,
                signature=f"size:{rel:.3f}", metrics={"relative_size": round(rel, 4), "important": it.important}))
        fg = parse_color(TEXT_COLORS.get(it.style, "#ffffff"))
        if fg is not None:
            backdrop, known = (assumed_backdrop(box=(BOX_COLOR, BOX_OPACITY)) if lay.boxed else assumed_backdrop(halo=((0.0, 0.0, 0.0), SHADOW_WEIGHT)))
            ratio = text_contrast(fg, float(self._opacity(it)), backdrop)
            if ratio < cfg.min_contrast:
                issues.append(self._issue(
                    ctx, it, "text.unreadable", Severity.WARNING, "Text has low contrast",
                    description=f"The text {it.label} has a contrast ratio of {ratio:.1f}:1 against its {'box' if known == 'box' else 'shadow over an assumed mid-grey picture'}; the minimum is {cfg.min_contrast:g}:1.",
                    why="Low contrast makes text hard to pick out, especially in bright scenes.", current=f"{ratio:.1f}:1", recommended=f"at least {cfg.min_contrast:g}:1",
                    suggested_fix="Give the text a background box or a stronger colour.", fix=self._open(it), confidence=85.0 if known == "box" else 60.0, viewer_impact=0.6,
                    signature=f"contrast:{round(ratio, 1)}", metrics={"contrast": round(ratio, 2), "backdrop": known}))
        chars = len(re.sub(r"\s+", "", it.content))
        need = (0.5 + chars / 16.0) if it.important else (0.4 + chars / 22.0)
        dur = it.clip.duration
        if dur < need - 0.05 and not self._persistent(ctx, it):
            issues.append(self._issue(
                ctx, it, "text.too_brief", Severity.WARNING if it.important else Severity.NOTICE, "Text is on screen for a very short time",
                description=f"The text {it.label} ({chars} characters) is on screen for {dur:.1f} s; reading it takes about {need:.1f} s.",
                why="A figure or date that disappears before it can be read is worse than none." if it.important else "Text that vanishes before it can be read only distracts.",
                current=f"{dur:.1f} s", recommended=f"at least {need:.1f} s", suggested_fix="Keep the text on screen a little longer.", fix=self._open(it), viewer_impact=0.5 if it.important else 0.25,
                signature=f"brief:{round(dur, 1)}", metrics={"duration": round(dur, 3), "seconds_needed": round(need, 2)}))

    @staticmethod
    def _opacity(it: _Item) -> float:
        try:
            return max(0.0, min(1.0, float(it.data.get("opacity", 1.0))))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _persistent(ctx: QCContext, it: _Item) -> bool:
        """A logo / watermark style text that stays for a large part of the video is intentional: its length is not judged."""
        return it.clip.duration >= 20.0 and it.clip.duration >= 0.5 * max(ctx.duration, 1e-6)

    # ------------------------------------------------------------------ timing: outside the scene, longer than needed
    def _timing(self, ctx: QCContext, cfg: _Cfg, scene: Any, it: _Item, issues: list[QCIssue]) -> None:
        if self._persistent(ctx, it):
            return
        before, after = max(0.0, scene.start - it.start), max(0.0, it.end - scene.end)
        declared = ctx.scene(it.clip.scene_id) if it.clip.scene_id else None
        elsewhere = declared is not None and declared.id != scene.id  # assigned to one scene, shown during another
        if max(before, after) > cfg.outside_seconds + 1e-9 or elsewhere:
            if elsewhere:
                where = f"during scene {scene.label}, but it is assigned to scene {declared.label}"  # type: ignore[union-attr]
            else:
                where = f"{before:.1f} s before scene {scene.label} starts" if before > after else f"{after:.1f} s after scene {scene.label} ends"
            issues.append(self._issue(
                ctx, it, "text.timing", Severity.WARNING if (elsewhere or max(before, after) > 1.5) else Severity.NOTICE, "Text is on screen outside its scene",
                description=f"The text {it.label} ({it.start:.2f}-{it.end:.2f} s) is on screen {where} ({scene.start:.2f}-{scene.end:.2f} s).",
                why="Text that belongs to one statement and lingers into the next one describes something that is no longer being said.",
                current=f"{max(before, after):.1f} s outside the scene" if not elsewhere else "shown in another scene", recommended=f"within {scene.start:.2f}-{scene.end:.2f} s",
                suggested_fix="Trim or move the text to the scene it belongs to.", fix=self._open(it), confidence=85.0, viewer_impact=0.4,
                signature=f"outside:{round(before, 1)}:{round(after, 1)}:{declared.id if elsewhere else ''}", metrics={"seconds_before": round(before, 2), "seconds_after": round(after, 2)}))
        chars = len(re.sub(r"\s+", "", it.content))
        need = (0.5 + chars / 16.0) if it.important else (0.4 + chars / 22.0)
        limit = max(cfg.hold_floor, cfg.hold_factor * need)
        if it.clip.duration > limit:
            issues.append(self._issue(
                ctx, it, "text.timing", Severity.NOTICE, "Text stays on screen longer than needed",
                description=f"The text {it.label} is on screen for {it.clip.duration:.1f} s; reading it takes about {need:.1f} s (limit {limit:.1f} s).",
                why="Text that stays after it has been read clutters the picture.", current=f"{it.clip.duration:.1f} s", recommended=f"at most {limit:.1f} s",
                suggested_fix="Shorten the text clip.", fix=self._open(it), confidence=70.0, viewer_impact=0.15, signature=f"long:{round(it.clip.duration)}",
                metrics={"duration": round(it.clip.duration, 2), "limit": round(limit, 2)}))

    # ------------------------------------------------------------------ pairs: overlap, collision with evidence, duplicates
    def _pairs(self, ctx: QCContext, cfg: _Cfg, own: list[_Item], window: list[_Item], issues: list[QCIssue]) -> None:
        own_ids = {i.clip.id for i in own}
        for a in window:
            if a.clip.id not in own_ids:
                continue
            for b in window:
                if b.clip.id == a.clip.id or (b.start, b.clip.id) < (a.start, a.clip.id):
                    continue  # each pair once, from the clip that starts first
                if a.is_text and b.is_text:
                    self._duplicate(ctx, cfg, a, b, issues)
                ra, rb = a.rect, b.rect
                if ra is None or rb is None or (not a.is_text and not b.is_text):
                    continue
                seconds = min(a.end, b.end) - max(a.start, b.start)
                if seconds < cfg.overlap_seconds - 1e-9:
                    continue
                both_text = a.is_text and b.is_text
                if both_text:  # how much of the smaller text is covered
                    share = ra.overlap_ratio(rb)
                else:  # how much of the evidence box is hidden (a small number callout inside a big box hides little of it)
                    txt, gfx = (ra, rb) if a.is_text else (rb, ra)
                    share = txt.share_of(gfx)
                if share < cfg.overlap_share:
                    continue
                if both_text:
                    code, title = "text.overlap", "Text overlaps other text"
                    desc = f"The text {a.label} and the text {b.label} are on screen together for {seconds:.1f} s and share {share * 100:.0f}% of the smaller one."
                    why = "Two texts on the same part of the screen hide each other, so one of them is not read."
                else:
                    code, title = "text.collision", "Text overlaps an evidence highlight"
                    desc = f"The text {(a if a.is_text else b).label} lies over the evidence highlight for {seconds:.1f} s and hides {share * 100:.0f}% of the highlighted region."
                    why = "Text on top of the highlighted evidence hides what the highlight points at."
                issues.append(self._issue(
                    ctx, a, code, Severity.WARNING, title, description=desc, why=why, current=f"{share * 100:.0f}% overlap for {seconds:.1f} s", recommended="text and graphics in separate parts of the frame",
                    suggested_fix="Move one of the two so they no longer overlap.", fix=self._open(a), affected=[f"{x.label} {x.clip.id}" for x in (a, b)], confidence=80.0 if both_text else 85.0,
                    viewer_impact=min(1.0, 0.45 + 0.4 * share), signature=f"{b.clip.id}:{round(share, 1)}",
                    metrics={"overlap_ratio": round(share, 3), "overlap_seconds": round(seconds, 2), "other_clip": b.clip.id}))

    def _duplicate(self, ctx: QCContext, cfg: _Cfg, a: _Item, b: _Item, issues: list[QCIssue]) -> None:
        if _norm(a.content) != _norm(b.content) or not _norm(a.content):
            return
        gap = b.start - a.end
        if gap > cfg.duplicate_window:
            return
        issues.append(self._issue(
            ctx, b, "text.duplicate", Severity.WARNING if gap < 2.0 else Severity.NOTICE, "The same text appears twice in a row",
            description=f"The text {b.label} is shown at {a.start:.2f} s and again at {b.start:.2f} s ({max(0.0, gap):.1f} s after the first has gone).",
            why="Showing the same text twice close together reads as a mistake, or wastes the viewer's attention.", current="the same text twice", recommended="each text shown once",
            suggested_fix="Delete the second one, unless the repetition is deliberate.", fix=fix_catalog.clip_delete(b.clip.id, "Delete the repeated text", ctx.settings), confidence=75.0,
            affected=[f"{a.label} {a.clip.id}", f"{b.label} {b.clip.id}"], viewer_impact=0.3, signature=f"{a.clip.id}", metrics={"gap_seconds": round(gap, 2), "first_clip": a.clip.id}))

    # ------------------------------------------------------------------ hierarchy and amount
    def _hierarchy(self, ctx: QCContext, cfg: _Cfg, texts: list[_Item], issues: list[QCIssue]) -> None:
        reported: set[str] = set()
        for low in texts:
            if low.layout is None or low.clip.id in reported:
                continue
            for high in texts:
                if high.layout is None or high.clip.id == low.clip.id:
                    continue
                rl, rh = RANK.get(low.style, 1), RANK.get(high.style, 1)
                if rl >= rh or low.layout.font_px <= high.layout.font_px * cfg.hierarchy_ratio:
                    continue
                reported.add(low.clip.id)
                issues.append(self._issue(
                    ctx, low, "text.hierarchy", Severity.NOTICE, "A minor text is larger than a more important one",
                    description=f"The {low.style.lower().replace('_', ' ')} {low.label} is set at {low.layout.font_px:.0f} px, larger than the {high.style.lower().replace('_', ' ')} {high.label} "
                    f"({high.layout.font_px:.0f} px) in the same scene.", why="Viewers read the biggest text first: the main message should be the biggest.",
                    current=f"{low.layout.font_px:.0f} px", recommended=f"no larger than {high.layout.font_px:.0f} px", suggested_fix="Reduce the size of the minor text or enlarge the main one.",
                    fix=self._open(low), confidence=70.0, affected=[f"{x.label} {x.clip.id}" for x in (low, high)], viewer_impact=0.2, signature=f"{high.clip.id}:{round(low.layout.font_px)}",
                    metrics={"size": low.layout.font_px, "other_size": high.layout.font_px}))
                break

    def _excessive(self, ctx: QCContext, cfg: _Cfg, scene: Any, texts: list[_Item], issues: list[QCIssue]) -> None:
        allowed = max(cfg.min_per_scene, int(round(cfg.per_minute * scene.duration / 60.0 + 0.49)))
        n = len(texts)
        if n <= allowed:
            return
        issues.append(self.issue(
            "text.excessive", CAT, Severity.WARNING if n >= 2 * allowed else Severity.NOTICE, "Too much text in one scene",
            description=f"Scene {scene.label} ({scene.duration:.1f} s) shows {n} text elements; up to {allowed} is comfortable ({cfg.per_minute:g} per minute, at least {cfg.min_per_scene}).",
            scene_id=scene.id, start=scene.start, end=scene.end, why="Many text elements compete with the picture and with the narration for the viewer's attention.",
            current=f"{n} text elements", recommended=f"at most {allowed}", suggested_fix="Keep the text that carries the message and remove the rest.",
            fix=fix_catalog.navigate("open.scene", "Open the scene to review its text", scene_id=scene.id), affected=[f"Text {t.label}" for t in texts[:6]], viewer_impact=min(1.0, 0.3 + 0.1 * (n - allowed)),
            signature=f"{n}/{allowed}", metrics={"texts": n, "allowed": allowed}, ctx=ctx))

    # ------------------------------------------------------------------ facts: values on screen against the script (review, never correction)
    def _facts(self, ctx: QCContext, cfg: _Cfg, it: _Item, sp: _Spoken, issues: list[QCIssue]) -> int:
        checked = 0
        if it.derived:
            return 0  # a section headline taken from the scene topic carries no figure of its own
        if not (sp.has_narration or sp.values):
            return 0  # a scene without narration or script has nothing to compare the text with
        review = fix_catalog.navigate("text.change", "Review the on-screen text against the script", clip_id=it.clip.id)
        for m in it.mentions:
            verdict = judge_mention(m, sp)
            checked += 1
            if verdict is None:
                continue
            issues.append(self._mismatch(ctx, cfg, it, m.text, verdict, review, "value"))
        if it.style in NAME_STYLES and sp.has_narration:
            for run in name_runs(it.content):
                words = [w for w in run if len(w) >= 3]
                if (len(run) >= 2 or it.style in ("LOCATION", "ENTITY_NAME")) and words and not any(w.lower().replace("’", "'") in sp.tokens for w in words):
                    issues.append(self._mismatch(ctx, cfg, it, " ".join(run), "absent", review, "name"))
                    checked += 1
        counter = it.data.get("counter")
        if isinstance(counter, dict) and counter.get("to") is not None:
            spec = counter_spec(it.content)
            try:
                final = float(counter["to"])
            except (TypeError, ValueError):
                final = math.nan
            if spec is not None and not (math.isfinite(final) and _same(final, float(spec["to"]))):
                shown = it.content.strip()
                issues.append(self.issue(
                    "text.value_mismatch", FACT, Severity.WARNING, "Possible inconsistency: on-screen value differs from the script",
                    description=f"The counter on {it.label} ends on {counter['to']}, while the text content shows {shown}. Potential inconsistency detected between the two on-screen values. Review recommended.",
                    scene_id=it.scene_id, clip=it.clip, track=it.track, ctx=ctx, affected=[f"Number card {it.clip.id}"],
                    why="A counter that settles on a different figure than the text it belongs to shows the viewer two values.", current=f"counter ends on {counter['to']}", recommended=f"the figure in the text ({shown})",
                    suggested_fix="Review the counter and the text content against the script.", fix=review, confidence=cfg.confidence, viewer_impact=0.6, signature=f"counter:{counter['to']}:{shown}",
                    metrics={"kind": "counter", "counter_to": counter["to"], "content": shown}))
                checked += 1
        return checked

    def _mismatch(self, ctx: QCContext, cfg: _Cfg, it: _Item, value: str, verdict: str, review: QCFixSpec, kind: str) -> QCIssue:
        differs = verdict == "differs"
        return self.issue(
            "text.value_mismatch", FACT, Severity.WARNING if kind == "value" else Severity.NOTICE, "Possible inconsistency: on-screen value differs from the script",
            description=f"The on-screen {'value' if kind == 'value' else 'name'} {value} does not appear in the narration of this scene. Review recommended."
            + (" Other values are spoken in this scene." if differs and kind == "value" else ""),
            scene_id=it.scene_id, clip=it.clip, track=it.track, ctx=ctx, affected=[f"Text {it.label}"],
            why="On-screen figures and names should follow the script: a difference may be a typing slip, a changed script, or a deliberate addition.", current=f"on screen: {value}",
            recommended="a value that appears in the scene's narration or script, or a deliberate addition", suggested_fix="Review the on-screen text against the script and the narration.", fix=review,
            confidence=min(cfg.confidence, 72.0 if differs else 65.0) if kind == "value" else 55.0, viewer_impact=0.6 if kind == "value" else 0.25,
            signature=f"{kind}:{_norm(value)}", metrics={"kind": kind, "value": value, "verdict": verdict})

    def _evidence(self, ctx: QCContext, scene: Any, gfx: list[_Item], issues: list[QCIssue]) -> None:
        if not gfx:
            return
        intent = ctx.project.visual_intents.get(scene.id)
        itype = getattr(getattr(intent, "type", None), "value", "") if intent is not None else ""
        supported = any(c.requires_evidence for c in scene.claims) or bool(scene.numbers) or itype in EVIDENCE_INTENTS
        if supported:
            return
        for g in gfx:
            issues.append(self._issue(
                ctx, g, "text.evidence_unsupported", Severity.NOTICE, "Evidence treatment may not support the spoken claim", category=FACT,
                description="An evidence highlight points at a region of the picture, but the narration of this scene contains no claim, figure or date that needs evidence. Review recommended.",
                why="A highlight tells the viewer 'look here, this proves it'; without a claim to prove it can mislead or distract.", current="an evidence highlight with no evidence claim in the scene",
                recommended="a highlight only where the narration makes a claim that the picture supports", suggested_fix="Check that the highlighted region supports what is being said, or remove the highlight.",
                fix=fix_catalog.navigate("graphics.change", "Review the evidence graphic", clip_id=g.clip.id), affected=[f"Evidence highlight {g.clip.id}"], confidence=60.0, viewer_impact=0.3,
                signature="no-claim", metrics={"visual_intent": itype, "claims": len(scene.claims)}))


def text_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Project totals from the per-scene text metrics."""
    t = sum(int(v) for k, v in metrics.items() if k.endswith(".texts_checked"))
    g = sum(int(v) for k, v in metrics.items() if k.endswith(".graphics_checked"))
    return {"texts_checked": t, "graphics_checked": g}
