"""AI editorial review (spec 26, 46, 47): a judgement ACROSS the deterministic findings, behind a provider abstraction.

The deterministic checkers find specific, measurable problems. An editorial reviewer asks the questions an editor asks about the whole video - does the picture support the
narration, is the pace right, are the important statements emphasised, is anything repetitive, distracting or confusing - and notices what only shows up when findings are
read together (a scene with several small problems is a weak section; a number shown with the wrong picture and unclear captions is likely to confuse).

Nothing here depends on a particular AI service: ``AIEditorialQCProvider`` is the contract and a reviewer is chosen by name.

* ``LocalAIEditorialQCProvider`` - the default: a deterministic, rule-based reviewer working from the request only. No network, no API, fully reproducible. It is labelled
  honestly as a rule-based review, not as an AI model.
* ``APIEditorialQCProvider`` - builds a prompt from the request and asks a client callable for JSON; the answer is validated strictly (unknown or malformed parts are dropped,
  confidence and severity are capped) before anything reaches the user.
* ``FutureEditorialQCProvider`` - documented placeholder that reports itself unavailable.

A reviewer returns concise decision factors (one sentence), never its reasoning. Findings are worded as possibilities ("Potential ... Review recommended"), carry a confidence, are
never allowed above WARNING, and QC does not judge whether any statement is true.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from app.core.exceptions import AppError
from app.core.textutil import content_terms
from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCIssue, SCORE_GROUPS
from app.qc.media_facts import extras
from app.qc.severity import Severity, cap_for_confidence, parse_severity

QUESTIONS: dict[str, str] = {
    "q1": "Does the video visually support the narration?", "q2": "Does the pacing feel appropriate?", "q3": "Are important statements emphasized?",
    "q4": "Are visuals unnecessarily repetitive?", "q5": "Are there distracting effects?", "q6": "Are scene transitions logical?", "q7": "Are captions readable?",
    "q8": "Does the video maintain visual continuity?", "q9": "Are there obvious weak sections?", "q10": "Is anything likely to confuse the viewer?"}
VERDICTS = ("yes", "mostly", "partly", "no")
# the questions asked "is there a problem?" answer yes when there is NOT one: q4, q5, q9 and q10 read "no problem" as "no"; the verdict below always means "this aspect is fine".
SEVERITY_WEIGHT = {"CRITICAL": 4.0, "ERROR": 3.0, "WARNING": 1.5, "NOTICE": 0.5, "INFO": 0.0}
RELATED = {  # question -> issue code prefixes that bear on it
    "q1": ("visual.mismatch", "visual.subject_mismatch", "visual.weak", "visual.generic", "visual.evidence", "scene.coverage", "scene.visual", "sync.visual"),
    "q2": ("pacing.too_fast", "pacing.too_slow", "pacing.uneven", "pacing.over_edited", "cut.micro_cuts", "cut.awkward", "cut.unnecessary"),
    "q3": ("pacing.under_emphasized", "scene.support"),
    "q4": ("visual.repetition",),
    "q5": ("motion.", "pacing.over_edited", "transition.distracting", "transition.excessive"),
    "q6": ("transition.", "continuity.subject_jump", "continuity.location_jump"),
    "q7": ("caption.", "text."),
    "q8": ("continuity.", "style."),
}
MAX_FINDINGS = 12
MAX_REQUEST_CHARS = 24000
MAX_REASON_CHARS = 240
ALLOWED_SEVERITIES = ("INFO", "NOTICE", "WARNING")  # an AI judgement is never an error or a critical problem
CODE_RE = re.compile(r"^[a-z][a-z0-9_.]{2,60}$")
NEW_TOPIC_LAG = 1.5  # s: the picture of the previous subject stays this long after the narration moved on
GROUP_OF_PREFIX = (("visual.", "visual_accuracy"), ("scene.", "timeline"), ("timeline.", "timeline"), ("sync.", "sync"), ("pacing.", "pacing"), ("cut.", "pacing"), ("caption.", "captions"),
                   ("text.", "captions"), ("audio.", "audio"), ("silence.", "audio"), ("continuity.", "continuity"), ("motion.", "continuity"), ("transition.", "continuity"),
                   ("frames.", "technical"), ("asset.", "technical"), ("media.", "technical"), ("render.", "technical"), ("preflight.", "technical"), ("style.", "pacing"))


class ProviderUnavailable(AppError):
    """The chosen reviewer cannot run (no client configured, planned for later ...)."""


class ProviderError(AppError):
    """The reviewer answered, but not with something usable."""


def group_of(code: str) -> str:
    return next((g for p, g in GROUP_OF_PREFIX if code.startswith(p)), "continuity")


# ---------------------------------------------------------------------------------------------- the provider contract
@dataclass
class EditorialRequest:
    """Everything a reviewer may look at, as plain data: no project objects, no file paths, nothing that identifies the user."""

    project_name: str = ""
    duration: float = 0.0
    canvas: list[int] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)
    timeline: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)  # the deterministic checkers' results
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def size(self) -> int:
        return len(json.dumps(self.to_dict(), default=str))


@dataclass
class EditorialFinding:
    code: str
    title: str
    reason: str
    severity: str = "NOTICE"
    confidence: float = 60.0
    scene_id: str | None = None
    start: float | None = None
    end: float | None = None
    suggested_fix: str = ""
    group_hint: str = "continuity"


@dataclass
class EditorialReview:
    provider: str
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)  # q1..q10 -> {"verdict", "confidence", "reason"}
    findings: list[EditorialFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AIEditorialQCProvider(ABC):
    name: str = "abstract"
    label: str = "Abstract editorial reviewer"

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(usable, why not)."""

    @abstractmethod
    def review(self, request: EditorialRequest) -> EditorialReview:
        """Answer the ten questions and add findings. May raise ``ProviderUnavailable`` / ``ProviderError``; the local provider never raises on a normal project."""


# ---------------------------------------------------------------------------------------------- local, deterministic reviewer
class LocalAIEditorialQCProvider(AIEditorialQCProvider):
    name = "local"
    label = "Local rule-based editorial review"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def review(self, request: EditorialRequest) -> EditorialReview:
        rev = EditorialReview(self.name, notes=["Rule-based review of the checkers' findings and metrics (no AI model, no network)."])
        findings = [f for f in request.findings if f.get("severity") != "INFO"]
        metrics = request.metrics
        present = {f.get("checker") for f in request.findings} | set(metrics)
        weak = self._weak_scenes(request, findings)
        confusing = self._confusing(request, findings)
        distracting = self._distracting(request)
        for q in QUESTIONS:
            pen, n, why = self._penalty(q, findings, metrics, request, weak, confusing, distracting)
            conf = min(90.0, 55.0 + 7.0 * len({"visual", "pacing", "continuity", "caption", "motion", "transition", "sync"} & present))
            rev.answers[q] = {"verdict": self._verdict(pen), "confidence": round(conf, 0), "reason": why}
        rev.findings = (self._topic_lag(request, findings) + [self._weak_finding(s) for s in weak[:4]] + [self._confusing_finding(s) for s in confusing[:3]]
                        + [self._distracting_finding(s) for s in distracting[:3]])[:MAX_FINDINGS]
        return rev

    # ---- scoring
    @staticmethod
    def _verdict(pen: float) -> str:
        return "yes" if pen < 0.5 else "mostly" if pen < 1.5 else "partly" if pen < 4.0 else "no"

    @staticmethod
    def _weight(f: dict) -> float:
        return SEVERITY_WEIGHT.get(f.get("severity", "NOTICE"), 0.5) * float(f.get("confidence", 100.0)) / 100.0

    def _penalty(self, q, findings, metrics, request, weak, confusing, distracting) -> tuple[float, int, str]:
        if q == "q9":
            pen = 1.5 * len(weak)
            return pen, len(weak), (f"{len(weak)} scene(s) have several open findings together ({', '.join(w['label'] for w in weak[:4])})." if weak else "No scene collects several open findings.")
        if q == "q10":
            pen = 2.0 * len(confusing)
            return pen, len(confusing), (f"{len(confusing)} scene(s) with a claim or number have conflicting signals ({', '.join(c['label'] for c in confusing[:4])})." if confusing
                                         else "No claim or number scene has conflicting signals.")
        rel = [f for f in findings if f["code"].startswith(RELATED[q])]
        pen = sum(self._weight(f) for f in rel)
        extra = ""
        if q == "q1":
            mean = (metrics.get("visual") or {}).get("mean_current_score")
            if isinstance(mean, (int, float)):
                pen += 3.0 if mean < 60 else 1.0 if mean < 75 else 0.0
                extra = f" Rechecked visual fit averages {mean:.0f}/100."
        if q == "q4":
            rep = (metrics.get("continuity") or {}).get("repetition_score")
            if isinstance(rep, (int, float)):
                pen += rep / 25.0
                extra = f" Repetition score {rep:.0f}/100."
        if q == "q5":
            pen += 1.5 * len(distracting)
        top = ", ".join(sorted({f["code"] for f in rel})[:3])
        why = (f"{len(rel)} related finding(s) ({top}).{extra}" if rel else f"No related findings.{extra}" if extra else "No related findings and the measured values are within the limits.")
        return pen, len(rel), why

    # ---- cross-finding judgements
    def _by_scene(self, request: EditorialRequest, findings: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for f in findings:
            if f.get("scene_id"):
                out.setdefault(f["scene_id"], []).append(f)
        return out

    def _weak_scenes(self, request: EditorialRequest, findings: list[dict]) -> list[dict]:
        out = []
        label = {s["id"]: s["label"] for s in request.scenes}
        for sid, fs in self._by_scene(request, findings).items():
            pen = sum(self._weight(f) for f in fs)
            groups = {group_of(f["code"]) for f in fs}
            if pen >= 3.0 and len(groups) >= 2 and len(fs) >= 3:
                out.append({"scene_id": sid, "label": label.get(sid, sid), "penalty": round(pen, 1), "findings": fs, "groups": sorted(groups)})
        return sorted(out, key=lambda s: -s["penalty"])

    def _confusing(self, request: EditorialRequest, findings: list[dict]) -> list[dict]:
        out = []
        by = self._by_scene(request, findings)
        for s in request.scenes:
            if not (s.get("claims") or s.get("numbers")):
                continue
            fs = by.get(s["id"], [])
            groups = {group_of(f["code"]) for f in fs if f["severity"] in ("WARNING", "ERROR", "CRITICAL") or f["code"].startswith(("visual.", "caption.", "text.", "sync."))}
            if len(groups & {"visual_accuracy", "sync", "captions"}) >= 2:
                out.append({"scene_id": s["id"], "label": s["label"], "groups": sorted(groups), "findings": fs, "number": bool(s.get("numbers"))})
        return out

    def _distracting(self, request: EditorialRequest) -> list[dict]:
        out = []
        for s in request.scenes:
            signals = [s.get("moving", 0) >= 1, s.get("transitions", 0) >= 1, s.get("graphics", 0) >= 2, s.get("shots", 0) >= 3 and s["end"] > s["start"] and s["shots"] / (s["end"] - s["start"]) > 0.33]
            if sum(signals) >= 3:
                out.append({"scene_id": s["id"], "label": s["label"], "start": s["start"], "end": s["end"], "signals": sum(signals)})
        return out

    def _topic_lag(self, request: EditorialRequest, findings: list[dict]) -> list[EditorialFinding]:
        skip = {f["scene_id"] for f in findings if f["code"].startswith(("sync.visual", "scene.visual", "visual.mismatch", "visual.subject_mismatch"))}  # already reported
        out = []
        for s in request.scenes:
            lag = s.get("lag")
            if lag is None or lag < NEW_TOPIC_LAG or s["id"] in skip or s.get("fit_scene", 1.0) >= 0.2 or s.get("fit_prev", 0.0) < 0.3:
                continue
            out.append(EditorialFinding(
                "topic_lag", "The picture stays on the previous subject", f"Potential delay detected: the narration of scene {s['label']} introduces a new topic, but the visual remains on the previous subject for {lag:.1f} seconds. "
                "Review recommended.", "NOTICE", 70.0, s["id"], s["start"], s["start"] + lag, "Change the picture closer to where the narration moves on.", "sync"))
        return out[:4]

    def _group_for(self, scene: dict) -> str:
        counts: dict[str, float] = {}
        for f in scene["findings"]:
            counts[group_of(f["code"])] = counts.get(group_of(f["code"]), 0.0) + self._weight(f)
        return max(counts, key=counts.get) if counts else "continuity"  # type: ignore[arg-type]

    def _weak_finding(self, s: dict) -> EditorialFinding:
        codes = ", ".join(sorted({f["code"] for f in s["findings"]})[:4])
        return EditorialFinding("weak_section", "Potential weak section", f"Potential weak section detected: scene {s['label']} collects {len(s['findings'])} open findings across {', '.join(s['groups'])} ({codes}). "
                                "Review recommended.", "WARNING" if s["penalty"] >= 5 else "NOTICE", min(85.0, 55.0 + 5.0 * len(s["findings"])), s["scene_id"], None, None,
                                "Open this scene first and resolve the findings together.", self._group_for(s))

    def _confusing_finding(self, s: dict) -> EditorialFinding:
        what = "a number" if s["number"] else "a claim"
        return EditorialFinding("confusing", "Viewers may be confused here", f"Potential confusion detected: scene {s['label']} presents {what} while {', '.join(s['groups'])} give conflicting signals. Review recommended.",
                                "NOTICE", 62.0, s["scene_id"], None, None, "Check that the picture, the captions and the timing all support the statement.", "visual_accuracy")

    def _distracting_finding(self, s: dict) -> EditorialFinding:
        return EditorialFinding("distracting", "Potential visual overload", f"Potential distraction detected: scene {s['label']} combines movement, transitions, several graphics and quick cuts. Review recommended.",
                                "NOTICE", 60.0, s["scene_id"], s["start"], s["end"], "Keep the effect that carries the message and drop the rest.", "continuity")


# ---------------------------------------------------------------------------------------------- API reviewer (any client; validated strictly)
class APIEditorialQCProvider(AIEditorialQCProvider):
    """``client(prompt) -> str`` returns the model's JSON answer. QC never talks to the network itself: the client is supplied by the application."""

    name = "api"
    label = "API editorial review"

    def __init__(self, client: Callable[[str], str] | None = None, name: str | None = None) -> None:
        self.client = client
        if name:
            self.name = name

    def is_available(self) -> tuple[bool, str]:
        return (True, "") if self.client is not None else (False, "No AI client is configured for the API editorial review.")

    def build_prompt(self, request: EditorialRequest) -> str:
        body = json.dumps(request.to_dict(), default=str, separators=(",", ":"))
        if len(body) > MAX_REQUEST_CHARS:  # keep the structure, shorten the bulk
            slim = request.to_dict()
            for s in slim["scenes"]:
                s["narration"] = s.get("narration", "")[:80]
            slim["findings"] = slim["findings"][:60]
            body = json.dumps(slim, default=str, separators=(",", ":"))[:MAX_REQUEST_CHARS]
        qs = "\n".join(f"{k}: {v}" for k, v in QUESTIONS.items())
        return ("You are a careful video editor reviewing a finished edit. Answer from the data only; do not judge whether any statement is true.\n"
                f"Questions:\n{qs}\n"
                'Reply with JSON only: {"answers": {"q1": {"verdict": "yes|mostly|partly|no", "confidence": 0-100, "reason": "one short sentence"}, ...}, '
                '"findings": [{"code": "snake_case", "scene_id": "scene id or null", "severity": "INFO|NOTICE|WARNING", "confidence": 0-100, "title": "...", "reason": "one short sentence", '
                '"suggested_fix": "...", "group": "visual_accuracy|sync|pacing|captions|audio|continuity|timeline|technical"}]}\n'
                "Give decision factors, not your reasoning. Word findings as possibilities.\n"
                f"DATA: {body}")

    def review(self, request: EditorialRequest) -> EditorialReview:
        ok, why = self.is_available()
        if not ok:
            raise ProviderUnavailable(why)
        try:
            text = self.client(self.build_prompt(request))  # type: ignore[misc]
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - the client is arbitrary code
            raise ProviderUnavailable("The AI service could not be reached.", details=str(exc)[:200]) from exc
        return parse_review(self.name, text, request)

    # nothing else: the answer is never trusted beyond what parse_review lets through


def parse_review(provider: str, text: str, request: EditorialRequest) -> EditorialReview:
    """Validate an AI answer strictly. Malformed or unknown parts are dropped (and counted in the notes); an answer that is not JSON at all is an error."""
    m = re.search(r"\{.*\}", text, re.S) if isinstance(text, str) else None  # a client may hand back anything
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise ProviderError("The AI reviewer did not answer in the expected format.")
    rev = EditorialReview(provider)
    dropped = 0
    answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
    for q in QUESTIONS:
        a = answers.get(q)
        if not isinstance(a, dict) or a.get("verdict") not in VERDICTS or not _num(a.get("confidence")):
            if q in answers:
                dropped += 1
            continue
        rev.answers[q] = {"verdict": a["verdict"], "confidence": _clamp(a["confidence"]), "reason": _sentence(a.get("reason"))}
    scenes = {s["id"] for s in request.scenes}
    raw = data.get("findings") if isinstance(data.get("findings"), list) else []
    for f in raw[:MAX_FINDINGS * 3]:
        if len(rev.findings) >= MAX_FINDINGS:
            dropped += 1
            continue
        if not (isinstance(f, dict) and isinstance(f.get("title"), str) and f["title"].strip() and isinstance(f.get("reason"), str) and CODE_RE.match(str(f.get("code", ""))) and _num(f.get("confidence"))):
            dropped += 1
            continue
        sid = f.get("scene_id")
        if sid is not None and (not isinstance(sid, str) or sid not in scenes):
            dropped += 1  # a finding about a scene that does not exist (or names one with something that is not an id) cannot be placed
            continue
        sev = str(f.get("severity", "NOTICE")).upper()
        sev = sev if sev in ALLOWED_SEVERITIES else "NOTICE"
        grp = str(f.get("group", "")).strip()
        rev.findings.append(EditorialFinding(f["code"], f["title"].strip()[:90], _sentence(f["reason"]), sev, _clamp(f["confidence"]), sid, None, None, _sentence(f.get("suggested_fix")),
                                             grp if grp in SCORE_GROUPS else "continuity"))
    if dropped:
        rev.notes.append(f"{dropped} malformed or unknown part(s) of the AI answer were ignored.")
    return rev


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)  # NaN and infinity are not a confidence


def _clamp(v: Any) -> float:
    return float(max(0.0, min(100.0, v)))


def _sentence(v: Any) -> str:
    """One concise sentence: the first one, capped. (Anything longer is reasoning, which QC neither asks for nor stores.)"""
    s = " ".join(str(v or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    return (m.group(1) if m else s)[:MAX_REASON_CHARS]


class FutureEditorialQCProvider(AIEditorialQCProvider):
    """Placeholder for a later, richer reviewer (for example one that also looks at frames). Reports itself unavailable instead of pretending."""

    name = "future"
    label = "Future editorial review (not available yet)"

    def is_available(self) -> tuple[bool, str]:
        return False, "This editorial reviewer is planned for a later version."

    def review(self, request: EditorialRequest) -> EditorialReview:
        raise ProviderUnavailable(self.is_available()[1])


def provider_for(settings, ctx: QCContext | None = None) -> tuple[AIEditorialQCProvider, str]:
    """The reviewer to use and a note when a fallback was needed: the one the service set, else the one named in the settings; unknown or unavailable -> the local reviewer."""
    chosen = getattr(ctx, "ai_provider", None) if ctx is not None else None
    note = ""
    if chosen is None:
        name = str(settings.ai_provider or "local").lower()
        chosen = {"local": LocalAIEditorialQCProvider, "api": APIEditorialQCProvider, "future": FutureEditorialQCProvider}.get(name)
        if chosen is None:
            return LocalAIEditorialQCProvider(), f"The AI reviewer “{name}” is unknown; the local rule-based review was used."
        chosen = chosen()
    ok, why = chosen.is_available()
    if ok:
        return chosen, note
    return LocalAIEditorialQCProvider(), f"{chosen.label} is unavailable ({why}); the local rule-based review was used."


# ---------------------------------------------------------------------------------------------- the checker
class EditorialChecker(BaseChecker):
    id = "editorial"
    label = "AI editorial review"
    categories = (QCCategory.EDITORIAL,)
    domains = ("timeline", "scenes", "transcript", "captions", "audio", "visual", "reference")
    settings_sections = ("ai_review_enabled", "ai_provider", "ai_confidence_caps", "min_confidence_to_report")
    scene_local = False
    expensive = True
    uses_shared = True
    version = "1"

    def input_hash(self, ctx: QCContext) -> str:
        chosen = getattr(ctx.ai_provider, "name", "") if ctx.ai_provider is not None else ""
        return sha(super().input_hash(ctx), chosen, extras(ctx, "facts", "assets"))  # the request carries claim / number counts and the picture names

    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        s = ctx.settings
        if not s.ai_review_enabled:
            out.notes.append("AI editorial review is switched off in the QC settings")
            out.metrics = {"provider": None, "enabled": False}
            return out
        provider, note = provider_for(s, ctx)
        if note:
            out.notes.append(note)
        report(0.1, "Preparing the review")
        request = build_request(ctx)
        report(0.3, f"Reviewing with {provider.label}")
        review = provider.review(request)  # a failure is the checker's failure: it is not swallowed
        ctx.check_cancel()
        out.notes += review.notes
        caps = tuple((float(a), str(b)) for a, b in s.ai_confidence_caps)
        covered = self._covered(ctx)
        dropped_dups = 0
        for f in review.findings:
            if f.confidence < s.min_confidence_to_report:
                continue
            if f.scene_id and (f.scene_id, f.group_hint) in covered and f.severity != "INFO" and not f.code.startswith(("weak_section", "confusing")):
                dropped_dups += 1  # the deterministic checkers already say this about this scene
                continue
            out.issues.append(self._issue(ctx, provider, f, caps))
        if dropped_dups:
            out.notes.append(f"{dropped_dups} editorial finding(s) repeat what the checkers already report")
        weak = [i.scene_id for i in out.issues if i.code == "editorial.weak_section"]
        out.metrics = {"provider": provider.name, "provider_label": provider.label, "answers": review.answers, "weak_scenes": weak, "request_size": request.size()}
        report(1.0, "Editorial review complete")
        return out

    @staticmethod
    def _covered(ctx: QCContext) -> set[tuple[str, str]]:
        out = set()
        for cid, o in ctx.shared.items():
            for i in getattr(o, "issues", []):
                if i.scene_id and i.severity in (Severity.CRITICAL, Severity.ERROR, Severity.WARNING) and i.confidence >= 70 and i.status.value == "OPEN":
                    out.add((i.scene_id, i.score_group))
        return out

    def _issue(self, ctx: QCContext, provider: AIEditorialQCProvider, f: EditorialFinding, caps) -> QCIssue:
        sev = cap_for_confidence(parse_severity(f.severity), f.confidence, caps)
        if sev in (Severity.ERROR, Severity.CRITICAL):  # defence in depth: nothing from an AI reviewer is ever above WARNING
            sev = Severity.WARNING
        code = f.code if f.code.startswith("editorial.") else f"editorial.{f.code}"
        iss = self.issue(
            code, QCCategory.EDITORIAL, sev, f.title, description=f.reason, scene_id=f.scene_id, start=f.start, end=f.end, source=f"ai:{provider.name}", confidence=f.confidence,
            why="An editor's overall impression of the scene, not a measurement.", current="judgement", recommended="review", suggested_fix=f.suggested_fix or "Review the scene.",
            fix=fx.navigate("open.scene", "Open the scene", scene_id=f.scene_id) if f.scene_id else None, viewer_impact=0.35, signature=sha(provider.name, code, f.scene_id), group_hint=f.group_hint,
            metrics={"provider": provider.name, "signature": sha(provider.name, code, f.scene_id)}, ctx=ctx)
        return iss


# ---------------------------------------------------------------------------------------------- the request
def build_request(ctx: QCContext) -> EditorialRequest:
    """A compact, provider-independent summary of the project and of what the other checkers found. Built on the worker from the detached snapshot."""
    p = ctx.project
    shared = ctx.shared
    vm = shared.get("visual")
    vis_scores = ((vm.metrics or {}).get("per_scene") or {}) if vm is not None else {}
    scenes = []
    clips = ctx.visual_clips()
    for idx, sc in enumerate(ctx.scenes):
        inside = [(t, c) for t, c in clips if c.timeline_start < sc.end - 1e-6 and c.timeline_end > sc.start + 1e-6]
        main = max(inside, key=lambda r: min(r[1].timeline_end, sc.end) - max(r[1].timeline_start, sc.start), default=None)
        asset = ctx.asset(main[1].asset_id) if main else None
        intent = p.visual_intents.get(sc.id)
        sterms = {w for w in content_terms(" ".join([sc.topic, sc.narration])) if len(w) > 2}
        prev_terms = {w for w in content_terms(" ".join([ctx.scenes[idx - 1].topic, ctx.scenes[idx - 1].narration])) if len(w) > 2} if idx else set()
        # the picture on screen when the narration of this scene begins: if it was already there before, how long does it stay?
        at_start = next((c for _t, c in inside if c.timeline_start <= sc.start + 0.05 < c.timeline_end), None)
        lag, vterms = None, set()
        if at_start is not None and at_start.timeline_start < sc.start - 0.05 and ctx.asset(at_start.asset_id) is not None:
            old = ctx.asset(at_start.asset_id)
            vterms = {w for w in content_terms(old.name.rsplit(".", 1)[0].replace("_", " ")) if len(w) > 2}  # type: ignore[union-attr]
            lag = round(min(at_start.timeline_end, sc.end) - sc.start, 2)
        starts = [c for _t, c in inside if sc.start - 1e-6 <= c.timeline_start < sc.end]
        scenes.append({
            "id": sc.id, "label": sc.label, "start": round(sc.start, 2), "end": round(sc.end, 2), "narration": sc.narration[:300], "importance": round(sc.importance, 2),
            "visual_type": intent.type.value if intent else "LITERAL", "claims": len(sc.claims), "numbers": len(sc.numbers),
            "visual": {"asset": asset.name if asset else None, "source": asset.source_type.value if asset else None, **{k: v for k, v in (vis_scores.get(sc.id) or {}).items() if k in ("original", "current")}},
            "shots": len(starts), "moving": sum(1 for c in starts if any(k.property in ("scale", "position_x", "position_y") for k in c.keyframes)),
            "transitions": sum(1 for c in starts if c.transition and str(c.transition.get("type", "CUT")).upper() != "CUT"),
            "graphics": sum(1 for _t, c in ctx.clips_in(sc.start, sc.end, kinds=("text", "graphic"))), "captions": sum(1 for _t, c in ctx.clips_in(sc.start, sc.end, kinds=("caption",))),
            "lag": lag, "fit_scene": _share(vterms, sterms), "fit_prev": _share(vterms, prev_terms)})
    findings, metrics = [], {}
    for cid, o in shared.items():
        for i in getattr(o, "issues", []):
            if i.status.value == "OPEN":
                findings.append({"checker": cid, "code": i.code, "scene_id": i.scene_id, "severity": i.severity.value, "confidence": i.confidence, "title": i.title[:90], "start": i.start_time})
        m = getattr(o, "metrics", None)
        if m:
            metrics[cid] = _lean(m)
    ref = None
    if p.reference_settings.enabled and p.reference_style_profile is not None:
        ref = {"dimensions": p.reference_settings.dimensions(), "scores": p.reference_style_profile.scores.as_dict()}
    tl = {"visual_clips": len(clips), "tracks": len(ctx.timeline.tracks), "captions": len(ctx.caption_clips()), "text": len(ctx.text_clips()), "graphics": len(ctx.graphic_clips())}
    return EditorialRequest(p.project_name, round(ctx.duration, 2), list(ctx.canvas), scenes, tl, ref, findings[:400], metrics)


def _share(a: set[str], b: set[str]) -> float:
    return round(len(a & b) / len(a), 3) if a else 1.0


def _lean(m: dict, limit: int = 1500) -> dict:
    """Keep a checker's metrics small and JSON-safe (curves and per-item tables are trimmed)."""
    out: dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > limit:
            continue
        out[k] = v
    return json.loads(json.dumps(out, default=str))
