"""SyncChecker: is everything the viewer sees on time with what the narrator says?

The voice-over is the master clock. For every scene the checker compares the *spoken* timing (the scene's transcript words and sentences) with the timing of what is
on screen: the first / last picture of the scene, captions, number / date / emphasis graphics, evidence highlights and the moments the picture changes. It measures
drift in milliseconds and classifies it with the project's sync tolerances only (``QCSettings.sync``).

What counts as "in sync" is deliberately forgiving where the edit engine and a human editor are forgiving:

* a picture may appear any time inside the pause *before* its statement (the engine cuts mid-pause); it is "early" only when it covers words of the previous
  statement, and "late" when it arrives after the statement has started;
* a number / date graphic may lead its word by the engine's 0.15 s;
* a mid-scene cut is only judged when it lands inside a spoken key figure or key claim (a cut between two sentences, in a pause, is always fine).

Everything scene-local: only the scene's own words, clips, assignment and its immediate neighbours' boundary words are read, so a result is reusable per scene.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.qc import fix_catalog
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import VISUAL_TRACK_KINDS, ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory, QCFixSpec, QCIssue
from app.qc.media_facts import extras
from app.qc.settings import SyncThresholds
from app.qc.severity import Severity
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.track import Track

if TYPE_CHECKING:  # pragma: no cover
    from app.analysis.models import Scene
    from app.editing.context import SceneContext
    from app.transcription.models import Word

EPS_MS = 0.01  # float noise on millisecond comparisons (word times are rounded to 1 ms)
MATCH_MIN_RATIO = 0.6  # share of a caption's / graphic's words that must agree with the transcript before the text counts as "matched"
OWN_WORDS_CONFIDENCE = 55.0  # a caption compared only with its own stored word timing (no transcript counterpart found)
FOLLOW_FRAMES = 1.5  # a picture that starts within this many frames of another's end is the same cut
EVIDENCE_INTENTS = ("EVIDENCE", "DATA")
_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_DIGITS = re.compile(r"\d[\d,]*\.?\d*")


# ---------------------------------------------------------------------------------------------- small helpers
def _norm(text: str) -> str:
    """Comparison form of a spoken / displayed token: lower case, no punctuation, no ordinal suffix ("$1,250," -> "1250", "15th" -> "15")."""
    t = re.sub(r"[\W_]+", "", str(text).lower())
    m = _ORDINAL.match(t)
    return m.group(1) if m else t


def _toks(text: str) -> list[str]:
    return [n for n in (_norm(t) for t in str(text).split()) if n]


def _value(text: str) -> float | None:
    m = _DIGITS.search(str(text))
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def _band(ms: float, cfg: SyncThresholds) -> Severity | None:
    """Drift class from the project's tolerances: [minor, moderate) NOTICE, [moderate, major) WARNING, major and above ERROR, below minor nothing."""
    a = abs(ms) + EPS_MS
    if a < cfg.minor_ms:
        return None
    if a < cfg.moderate_ms:
        return Severity.NOTICE
    if a < cfg.major_ms:
        return Severity.WARNING
    return Severity.ERROR


def _reaches(ms: float, specific_ms: float, cfg: SyncThresholds) -> bool:
    """A drift counts for a specific rule (caption early, visual late, emphasis miss) when it reaches both that rule's tolerance and the general minor tolerance."""
    return abs(ms) + EPS_MS >= max(cfg.minor_ms, specific_ms)


def _impact(ms: float, base: float) -> float:
    return min(1.0, base + abs(ms) / 3000.0)


def _confidence(ratio: float, tokens: int) -> float:
    c = 60.0 + 40.0 * ratio
    return min(c, 90.0) if tokens <= 2 else c  # one or two words can match in many places


def _fmt(ms: float) -> str:
    return f"{abs(ms):.0f} ms"


def _short(text: str, n: int = 48) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------------------------- matching spoken words
@dataclass
class _Run:
    """Where a displayed phrase sits in the narration."""

    first: "Word"  # the spoken word aligned with the phrase's first word
    last: "Word"
    ratio: float  # share of the phrase's words that equal the aligned transcript words
    aligned: list["Word | None"]  # per phrase word: its transcript word when equal, else None


def _match_run(tokens: list[str], pairs: list[tuple[str, "Word"]], hint: float) -> _Run | None:
    """Best in-order alignment of ``tokens`` with consecutive transcript words: most equal words first, then the occurrence nearest to ``hint`` (seconds)."""
    n, m = len(tokens), len(pairs)
    if n == 0 or m == 0:
        return None
    best: tuple[tuple[int, float], int, int] | None = None
    for off in range(max(1, m - n + 1)):
        k = min(n, m - off)
        score = sum(1 for i in range(k) if tokens[i] == pairs[off + i][0])
        key = (score, -abs(pairs[off][1].start - hint))
        if best is None or key > best[0]:
            best = (key, off, score)
    assert best is not None
    _key, off, score = best
    ratio = score / n
    if ratio < MATCH_MIN_RATIO:
        return None
    k = min(n, m - off)
    aligned = [pairs[off + i][1] if i < k and tokens[i] == pairs[off + i][0] else None for i in range(n)]
    return _Run(pairs[off][1], pairs[off + k - 1][1], ratio, aligned)


@dataclass
class _Stmt:
    """A spoken statement a graphic belongs to."""

    start: float
    end: float
    confidence: float
    label: str


@dataclass
class _Phrase:
    label: str
    start: float
    end: float
    kind: str  # "number" (any cut inside the run) | "claim" (a cut inside a spoken word of the claim)
    words: list["Word"]
    confidence: float


# ---------------------------------------------------------------------------------------------- the timeline index
@dataclass
class _Bucket:
    visuals: list[tuple[Track, Clip]] = field(default_factory=list)
    captions: list[tuple[Track, Clip]] = field(default_factory=list)
    texts: list[tuple[Track, Clip]] = field(default_factory=list)
    graphics: list[tuple[Track, Clip]] = field(default_factory=list)


@dataclass
class _Index:
    scenes: dict[str, _Bucket] = field(default_factory=dict)
    visuals: list[tuple[Clip, str | None]] = field(default_factory=list)  # every visible picture clip, with the scene it was assigned to


def _build_index(ctx: QCContext) -> _Index:
    """Assign every visible clip to a scene once. A clip belongs to its own scene id while it overlaps that scene; otherwise (user-added clip, or a stale id: scene
    commands never touch the timeline) to the scene at its middle. Hidden tracks are not on screen, so they are not part of what the viewer sees."""
    scenes = ctx.scenes
    by_id = {s.id: s for s in scenes}
    starts = [s.start for s in scenes]

    def scene_of(c: Clip) -> str | None:
        own = by_id.get(c.scene_id) if c.scene_id else None
        if own is not None and c.timeline_end > own.start + 1e-6 and c.timeline_start < own.end - 1e-6:
            return own.id
        mid = c.timeline_start + c.duration / 2
        i = bisect_right(starts, mid + 1e-6) - 1
        return scenes[i].id if i >= 0 and mid < scenes[i].end + 1e-6 else None

    idx = _Index()
    for track in ctx.timeline.tracks:
        if track.hidden:
            continue
        for clip in sorted(track.clips, key=lambda c: (c.timeline_start, c.id)):
            if not (math.isfinite(clip.timeline_start) and math.isfinite(clip.duration)) or clip.duration <= 1e-6:
                continue  # a non-finite time is the timeline checker's CRITICAL finding, and cannot be placed
            if clip.kind == KIND_MEDIA and track.kind in VISUAL_TRACK_KINDS:
                slot = "visuals"
            elif clip.kind == KIND_CAPTION:
                slot = "captions"
            elif clip.kind == KIND_TEXT:
                slot = "texts"
            elif clip.kind == KIND_GRAPHIC:
                slot = "graphics"
            else:
                continue
            sid = scene_of(clip)
            if slot == "visuals":
                idx.visuals.append((clip, sid))
            if sid is not None:
                getattr(idx.scenes.setdefault(sid, _Bucket()), slot).append((track, clip))
    return idx


@dataclass
class _S:
    """Everything one scene's analysis reads."""

    scene: "Scene"
    sc: "SceneContext | None"
    words: list["Word"]
    pairs: list[tuple[str, "Word"]]  # (normalised text, word) of the words that have any letters / digits
    wmap: dict[str, "Word"]
    bucket: _Bucket


def sync_summary(metrics: dict[str, Any]) -> dict[str, float]:
    """Project totals recomputed from the per-scene metric keys (the engine merges re-analysed scenes into the old metrics, so stored totals can be partial)."""
    n = 0.0
    worst = 0.0
    total = 0.0
    for k, v in metrics.items():
        if k.endswith(".captions_checked"):
            sid = k[: -len(".captions_checked")]
            c = float(v)
            n += c
            total += c * float(metrics.get(f"{sid}.mean_caption_drift_ms", 0.0))
            worst = max(worst, float(metrics.get(f"{sid}.max_caption_drift_ms", 0.0)))
    return {"captions_checked": n, "max_caption_drift_ms": worst, "mean_caption_drift_ms": round(total / n, 1) if n else 0.0}


class SyncChecker(BaseChecker):
    id = "sync"
    label = "Narration Sync"
    categories = (QCCategory.SYNC,)
    domains = ("timeline", "scenes", "transcript", "captions")
    # the cache key must cover every setting that can change an answer: the tolerances, the "small shift" limit and fix permissions (the fix's safe flag), declared gaps
    settings_sections = ("sync", "max_caption_shift_seconds", "fix_permissions", "intentional_gaps")
    scene_local = True
    expensive = False
    version = "1"

    # ------------------------------------------------------------------ cache keys: the numbers / claims a cut or graphic is judged against, and the boundary words of the neighbouring scenes
    def input_hash(self, ctx: QCContext) -> str:
        return sha(super().input_hash(ctx), extras(ctx, "facts"))

    def scene_input_hash(self, ctx: QCContext, scene_id: str) -> str:
        return sha(super().scene_input_hash(ctx, scene_id), extras(ctx, "facts", "neighbours", scene_id=scene_id))

    # ------------------------------------------------------------------ the run
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        scenes = ctx.target_scenes()
        tr = ctx.project.transcription.transcript
        if tr is None or not tr.words:
            out.notes.append("No transcript: picture / text timing could not be compared with the narration (captions are compared with their own word timing)")
        idx: _Index = ctx.memo("sync.index", lambda: _build_index(ctx))
        metrics: dict[str, Any] = {}
        all_drifts: list[float] = []
        unmeasured = 0
        for n, scene in enumerate(scenes):
            ctx.check_cancel()
            report(n / max(1, len(scenes)), ctx.scene_label(scene.id))
            st = self._state(ctx, scene, idx)
            drifts, skipped = self._captions(ctx, st, out.issues)
            all_drifts += drifts
            unmeasured += skipped
            pre = f"scene.{scene.id}."
            metrics[pre + "captions_checked"] = len(drifts)
            metrics[pre + "max_caption_drift_ms"] = round(max((abs(d) for d in drifts), default=0.0), 1)
            metrics[pre + "mean_caption_drift_ms"] = round(sum(abs(d) for d in drifts) / len(drifts), 1) if drifts else 0.0
            metrics[pre + "mean_signed_caption_drift_ms"] = round(sum(drifts) / len(drifts), 1) if drifts else 0.0
            if st.words:
                metrics.update({pre + k: v for k, v in self._visuals(ctx, st, idx, out.issues).items()})
                self._texts(ctx, st, out.issues)
                self._cuts(ctx, st, idx, out.issues)
        metrics["captions_checked"] = len(all_drifts)
        metrics["max_caption_drift_ms"] = round(max((abs(d) for d in all_drifts), default=0.0), 1)
        metrics["mean_caption_drift_ms"] = round(sum(abs(d) for d in all_drifts) / len(all_drifts), 1) if all_drifts else 0.0
        metrics["captions_unmeasured"] = unmeasured
        out.metrics = metrics
        out.notes.append(f"{len(all_drifts)} caption(s) compared with the narration in {len(scenes)} scene(s)" + (f"; {unmeasured} could not be matched to any spoken words" if unmeasured else ""))
        report(1.0, "")
        return out

    def _state(self, ctx: QCContext, scene: "Scene", idx: _Index) -> _S:
        sc = ctx.scene_ctx(scene.id)
        words = list(sc.words) if sc is not None else ctx.words_between(scene.start, scene.end)
        pairs = [(n, w) for w in words if (n := _norm(w.text))]
        return _S(scene, sc, words, pairs, {w.word_id: w for w in words}, idx.scenes.get(scene.id) or _Bucket())

    # ------------------------------------------------------------------ issue helper
    def _mk(self, ctx: QCContext, st: _S, code: str, sev: Severity, title: str, *, clip: Clip | None = None, track: Track | None = None, start: float | None = None,
            end: float | None = None, conf: float = 100.0, desc: str = "", why: str = "", current: str = "", rec: str = "", fix_text: str = "", fix: QCFixSpec | None = None,
            impact: float = 0.5, sig: str = "", metrics: dict[str, Any] | None = None, affected: str = "") -> QCIssue:
        return self.issue(code, QCCategory.SYNC, sev, title, description=desc, scene_id=st.scene.id, clip=clip, track=track, start=start, end=end, confidence=conf, why=why,
                          current=current, recommended=rec, suggested_fix=fix_text, fix=fix, viewer_impact=impact, signature=sig, metrics=metrics,
                          affected=[affected] if affected else None, ctx=ctx)

    @staticmethod
    def _open(clip: Clip, text: str = "Check the timing on the timeline") -> QCFixSpec:
        return fix_catalog.navigate("open.timeline", text, clip_id=clip.id)

    # ------------------------------------------------------------------ captions: drift, early, emphasis
    def _captions(self, ctx: QCContext, st: _S, issues: list[QCIssue]) -> tuple[list[float], int]:
        cfg = ctx.settings.sync
        drifts: list[float] = []
        unmeasured = 0
        for track, clip in st.bucket.captions:
            text = clip.text or {}
            cw = [w for w in (text.get("words") or []) if isinstance(w, dict)]
            toks = [(n, i) for i, w in enumerate(cw) if (n := _norm(w.get("text", "")))] if cw else [(n, i) for i, n in enumerate(_toks(text.get("text", "")))]
            if not toks:
                unmeasured += 1
                continue
            own_start = _num(cw[0].get("start")) if cw else None
            run = _match_run([n for n, _i in toks], st.pairs, own_start if own_start is not None else clip.timeline_start)
            if run is not None:
                ref, ref_end, conf, how = run.first.start, run.last.end, _confidence(run.ratio, len(toks)), "transcript"
            elif own_start is not None:
                ref, ref_end, conf, how = own_start, _num(cw[-1].get("end")) or own_start, OWN_WORDS_CONFIDENCE, "own"
            else:
                unmeasured += 1
                continue
            delta_ms = (clip.timeline_start - ref) * 1000.0
            drifts.append(delta_ms)
            flagged = self._caption_drift(ctx, st, track, clip, ref, ref_end, delta_ms, conf, how, run, cfg, issues)
            if cw and run is not None and not flagged:
                self._caption_emphasis(ctx, st, track, clip, cw, toks, run, cfg, issues)
        return drifts, unmeasured

    def _caption_drift(self, ctx: QCContext, st: _S, track: Track, clip: Clip, ref: float, ref_end: float, delta_ms: float, conf: float, how: str, run: _Run | None,
                       cfg: SyncThresholds, issues: list[QCIssue]) -> bool:
        late = delta_ms > 0
        if late:
            if delta_ms + EPS_MS < cfg.minor_ms:
                return False
        elif not _reaches(delta_ms, cfg.caption_early_ms, cfg):
            return False
        sev = _band(delta_ms, cfg)
        if sev is None:
            return False
        shift = ref - clip.timeline_start  # signed seconds to move the window (negative = earlier)
        new_end = clip.timeline_end + shift  # the window moves as a whole; moving it earlier never cuts the last word off
        if late:
            new_end = max(new_end, ref_end)
        fix = fix_catalog.caption_retime(clip.id, ref, new_end, shift, ctx.settings)
        if conf < 90.0 and fix.safe:  # an uncertain match is never applied without a click
            fix.safe, fix.needs_confirmation = False, True
        basis = "the transcript" if how == "transcript" else "the caption's own stored word timing (its text could not be matched to the transcript)"
        if late:
            code, title = "sync.caption_drift", f"Caption appears {_fmt(delta_ms)} after the voice"
            desc = (f"The caption “{_short(_cap_text(clip))}” appears at {clip.timeline_start:.2f} s; its first word is spoken at {ref:.2f} s according to {basis}: "
                    f"{_fmt(delta_ms)} late.")
            why = "A caption that trails the voice makes the viewer read words they have already heard and weakens the link between speech and text."
        else:
            code, title = "sync.caption_early", f"Caption appears {_fmt(delta_ms)} before the voice"
            desc = (f"The caption “{_short(_cap_text(clip))}” appears at {clip.timeline_start:.2f} s; its first word is not spoken until {ref:.2f} s according to {basis}: "
                    f"{_fmt(delta_ms)} early.")
            why = "A caption that shows before its words are spoken gives the content away and makes the narration feel out of step."
        issues.append(self._mk(
            ctx, st, code, sev, title, clip=clip, track=track, start=min(clip.timeline_start, ref), end=max(clip.timeline_start, ref) + 0.001, conf=conf, desc=desc, why=why,
            current=f"{'+' if late else '-'}{abs(delta_ms):.0f} ms against the first spoken word", rec=f"Caption starts within {(cfg.minor_ms if late else max(cfg.minor_ms, cfg.caption_early_ms)):.0f} ms of the first spoken word ({ref:.2f} s)",
            fix_text=fix.summary, fix=fix, impact=_impact(delta_ms, 0.3), sig=f"{clip.id}:{'lag' if late else 'early'}:{round(delta_ms / 200)}", affected=f"Caption {clip.slot or clip.id}",
            metrics={"drift_ms": round(delta_ms, 1), "spoken_start": round(ref, 3), "match": how, "match_ratio": round(run.ratio, 2) if run else None}))
        return True

    def _caption_emphasis(self, ctx: QCContext, st: _S, track: Track, clip: Clip, cw: list[dict], toks: list[tuple[str, int]], run: _Run, cfg: SyncThresholds,
                          issues: list[QCIssue]) -> None:
        """Emphasised words are highlighted from their stored word timing: it must agree with when the word is spoken, and the caption must still be on screen then."""
        pos_of = {ci: p for p, (_n, ci) in enumerate(toks)}
        for mark in (clip.text or {}).get("emphasis") or []:
            if not isinstance(mark, dict):
                continue
            ci = mark.get("word_index")
            p = pos_of.get(ci) if isinstance(ci, int) else None
            spoken = run.aligned[p] if p is not None and p < len(run.aligned) else None
            if spoken is None:
                continue
            stored = _num(cw[ci].get("start"))
            if stored is None:
                continue
            word_ms = (stored - spoken.start) * 1000.0
            gone_ms = (spoken.start - clip.timeline_end) * 1000.0  # the caption has left the screen before the word is spoken
            if _reaches(word_ms, cfg.emphasis_miss_ms, cfg):
                miss_ms, why_text = word_ms, f"the caption's highlight for “{cw[ci].get('text', '')}” is timed {stored:.2f} s but the word is spoken at {spoken.start:.2f} s"
            elif gone_ms > 0 and _reaches(gone_ms, cfg.emphasis_miss_ms, cfg):
                miss_ms, why_text = gone_ms, f"the caption leaves the screen at {clip.timeline_end:.2f} s, before “{cw[ci].get('text', '')}” is spoken at {spoken.start:.2f} s"
            else:
                continue
            sev = _band(miss_ms, cfg)
            if sev is None:
                continue
            issues.append(self._mk(
                ctx, st, "sync.emphasis_miss", sev, f"Emphasised word is {_fmt(miss_ms)} off the voice", clip=clip, track=track, conf=_confidence(run.ratio, len(toks)) * 0.9,
                desc=f"In the caption “{_short(_cap_text(clip))}” {why_text}: {_fmt(miss_ms)} apart.", why="Emphasis only works when it lands on the word being said; off the beat it distracts instead of stressing.",
                current=f"{_fmt(miss_ms)} from the spoken word", rec=f"Emphasis within {cfg.emphasis_miss_ms:.0f} ms of the spoken word", fix_text="Check the caption timing on the timeline.", fix=self._open(clip),
                impact=_impact(miss_ms, 0.3), sig=f"{clip.id}:cap-emph:{ci}:{round(miss_ms / 200)}", affected=f"Caption {clip.slot or clip.id}",
                metrics={"miss_ms": round(miss_ms, 1), "word": cw[ci].get("text", ""), "spoken_start": round(spoken.start, 3)}))

    # ------------------------------------------------------------------ pictures: late / early / ends early / overstays
    def _visuals(self, ctx: QCContext, st: _S, idx: _Index, issues: list[QCIssue]) -> dict[str, Any]:
        cfg = ctx.settings.sync
        vis = sorted(st.bucket.visuals, key=lambda tc: (tc[1].timeline_start, tc[1].id))
        if not vis:
            return {}
        scene, sc = st.scene, st.sc
        w0, wl = st.words[0], st.words[-1]
        track, first = vis[0]
        lag_ms = (first.timeline_start - w0.start) * 1000.0
        metrics: dict[str, Any] = {"first_visual_offset_ms": round(lag_ms, 1)}
        label = ctx.scene_label(scene.id)
        covering = any(c.timeline_start <= w0.start + 1e-6 and c.timeline_end >= w0.start + cfg.minor_ms / 1000.0 and sid != scene.id for c, sid in idx.visuals)

        # -- the first picture arrives after its statement began
        sev = _band(lag_ms, cfg)
        if (lag_ms > 0 and sev is not None and _reaches(lag_ms, cfg.visual_late_ms, cfg) and not covering and not ctx.in_intentional_gap(w0.start, first.timeline_start)):
            intent = st.sc.intent.type.value if st.sc is not None and st.sc.intent is not None else ""
            important = intent in EVIDENCE_INTENTS
            issues.append(self._mk(
                ctx, st, "sync.important_visual_late" if important else "sync.visual_late", sev,
                f"{'Evidence visual' if important else 'Visual'} starts {_fmt(lag_ms)} after its statement",
                clip=first, track=track, start=w0.start, end=first.timeline_start, conf=90.0,
                desc=(f"The first visual of {label} starts at {first.timeline_start:.2f} s, {_fmt(lag_ms)} after the narration begins (“{_short(w0.text, 20)}” at {w0.start:.2f} s). "
                      f"For that time the viewer hears the statement without its picture."),
                why="The picture should arrive with the statement it illustrates; late pictures leave the viewer looking at something unrelated." if not important
                else "Evidence and data visuals carry the claim being made; showing them after it was spoken weakens the proof.",
                current=f"+{lag_ms:.0f} ms after the first spoken word", rec=f"Start within {max(cfg.minor_ms, cfg.visual_late_ms):.0f} ms of the first spoken word ({w0.start:.2f} s)",
                fix_text="Review the visual's start on the timeline; moving AI visuals is not done automatically.", fix=self._open(first), impact=_impact(lag_ms, 0.45),
                sig=f"{first.id}:late:{round(lag_ms / 250)}", affected=f"Visual {first.slot or first.id}", metrics={"offset_ms": round(lag_ms, 1), "first_word": w0.text}))

        # -- the first picture arrives while the previous statement is still being spoken
        if sc is not None and sc.prev is not None:
            early_ms = (sc.prev.last_word_end - first.timeline_start) * 1000.0
            sev = _band(early_ms, cfg)
            if early_ms > 0 and sev is not None:
                issues.append(self._mk(
                    ctx, st, "sync.visual_early", sev, f"Visual starts {_fmt(early_ms)} before the previous statement ends", clip=first, track=track,
                    start=first.timeline_start, end=sc.prev.last_word_end, conf=90.0,
                    desc=(f"The first visual of {label} starts at {first.timeline_start:.2f} s but the previous scene's narration continues until {sc.prev.last_word_end:.2f} s: "
                          f"the picture changes {_fmt(early_ms)} too soon."),
                    why="A picture that changes during the previous sentence tells the viewer about something the narrator has not reached yet.",
                    current=f"-{early_ms:.0f} ms before the previous statement ended", rec=f"Start at or after {sc.prev.last_word_end:.2f} s (inside the pause before the statement)",
                    fix_text="Review the visual's start on the timeline.", fix=self._open(first), impact=_impact(early_ms, 0.4), sig=f"{first.id}:early:{round(early_ms / 250)}",
                    affected=f"Visual {first.slot or first.id}", metrics={"offset_ms": round(-early_ms, 1), "edge": "start"}))

        # -- the last picture leaves before its statement is finished
        last_track, last = max(vis, key=lambda tc: (tc[1].timeline_end, tc[1].id))
        stmt_end = min(wl.end, scene.end)
        short_ms = (stmt_end - last.timeline_end) * 1000.0
        followed = any(c is not last and abs(c.timeline_start - last.timeline_end) <= FOLLOW_FRAMES * ctx.frame for c, _sid in idx.visuals)
        sev = _band(short_ms, cfg)
        if short_ms > 0 and sev is not None and not followed and not ctx.in_intentional_gap(last.timeline_end, stmt_end):
            issues.append(self._mk(
                ctx, st, "sync.visual_early", sev, f"Visual ends {_fmt(short_ms)} before its statement is finished", clip=last, track=last_track, start=last.timeline_end, end=stmt_end,
                conf=85.0,
                desc=(f"The last visual of {label} ends at {last.timeline_end:.2f} s, while the narration of the scene continues until {stmt_end:.2f} s ({_fmt(short_ms)} with no picture "
                      f"for the statement)."),
                why="The picture should stay until its statement is finished, otherwise the end of the sentence is spoken over nothing or over unrelated footage.",
                current=f"ends {short_ms:.0f} ms before the last spoken word", rec=f"Keep the visual until {stmt_end:.2f} s", fix_text="Extend the visual or check what follows it on the timeline.",
                fix=self._open(last), impact=_impact(short_ms, 0.4), sig=f"{last.id}:ends-early:{round(short_ms / 250)}", affected=f"Visual {last.slot or last.id}",
                metrics={"offset_ms": round(-short_ms, 1), "edge": "end"}))

        # -- a picture stays on screen over the next statement
        if sc is not None and sc.next is not None:
            nxt = ctx.scene(sc.next.scene_id)
            nw: list[Word] | None = None
            for tr_, c in vis:
                if c.timeline_end <= scene.end + cfg.minor_ms / 1000.0 - 1e-6:
                    continue  # does not even reach into the next scene by the tolerance
                if nw is None:
                    nw = ctx.words_between(nxt.start, nxt.end) if nxt is not None else []
                if not nw:
                    break
                over_ms = (c.timeline_end - nw[0].start) * 1000.0
                sev = _band(over_ms, cfg)
                if over_ms > 0 and sev is not None:
                    issues.append(self._mk(
                        ctx, st, "sync.visual_overstay", sev, f"Visual stays {_fmt(over_ms)} into the next statement", clip=c, track=tr_, start=nw[0].start, end=c.timeline_end, conf=85.0,
                        desc=(f"A visual of {label} stays on screen until {c.timeline_end:.2f} s, but the next scene's narration starts at {nw[0].start:.2f} s: the old picture is "
                              f"still showing for {_fmt(over_ms)} of the new statement."),
                        why="A picture that outlives its statement keeps illustrating something the narrator has moved on from.",
                        current=f"{over_ms:.0f} ms past the start of the next statement", rec=f"End at or before {nw[0].start:.2f} s",
                        fix_text="Trim the visual on the timeline.", fix=self._open(c), impact=_impact(over_ms, 0.4), sig=f"{c.id}:overstay:{round(over_ms / 250)}",
                        affected=f"Visual {c.slot or c.id}", metrics={"overstay_ms": round(over_ms, 1)}))
        return metrics

    # ------------------------------------------------------------------ text graphics, evidence highlights
    def _texts(self, ctx: QCContext, st: _S, issues: list[QCIssue]) -> None:
        cfg = ctx.settings.sync
        for track, clip in st.bucket.texts:
            t = clip.text or {}
            content = str(t.get("content") or "").strip()
            if not content or t.get("derived"):
                continue  # a derived headline is not taken from the narration: there is no spoken counterpart
            variant = _variant(t)
            counter = bool(t.get("counter")) or _preset(clip) == "counter"
            onset = _onset(clip)
            stmt = self._text_statement(content, variant, st, onset)
            if stmt is None:
                continue
            important = variant in ("NUMBER", "DATE")
            lag_ms = (onset - stmt.start) * 1000.0
            if important and not counter and lag_ms > 0 and _reaches(lag_ms, cfg.visual_late_ms, cfg):
                sev = _band(lag_ms, cfg)
                if sev is not None:
                    kind = "date" if variant == "DATE" else "figure"
                    issues.append(self._mk(
                        ctx, st, "sync.important_visual_late", sev, f"{kind.capitalize()} graphic appears {_fmt(lag_ms)} after it is spoken", clip=clip, track=track,
                        start=stmt.start, end=onset, conf=stmt.confidence,
                        desc=(f"The graphic “{_short(content, 30)}” appears at {onset:.2f} s, {_fmt(lag_ms)} after the narrator says {stmt.label} ({stmt.start:.2f} s)."),
                        why="A number or date is easiest to take in when it is seen as it is said; after the moment it only repeats what was already heard.",
                        current=f"+{lag_ms:.0f} ms after the spoken {kind}", rec=f"Appear within {max(cfg.minor_ms, cfg.visual_late_ms):.0f} ms of the spoken {kind} ({stmt.start:.2f} s)",
                        fix_text="Review the graphic's start on the timeline.", fix=self._open(clip), impact=_impact(lag_ms, 0.5), sig=f"{clip.id}:late:{round(lag_ms / 250)}",
                        affected=f"Graphic {clip.slot or clip.id}", metrics={"offset_ms": round(lag_ms, 1), "spoken_start": round(stmt.start, 3), "variant": variant}))
                    continue
            if not _has_emphasis(clip):
                continue
            early_ms = (stmt.start - onset) * 1000.0
            late_ms = (onset - stmt.end) * 1000.0
            miss_ms = early_ms if early_ms > 0 else late_ms
            if miss_ms <= 0 or not _reaches(miss_ms, cfg.emphasis_miss_ms, cfg):
                continue
            sev = _band(miss_ms, cfg)
            if sev is None:
                continue
            side = "before" if early_ms > 0 else "after"
            issues.append(self._mk(
                ctx, st, "sync.emphasis_miss", sev, f"Emphasis lands {_fmt(miss_ms)} {side} the spoken word", clip=clip, track=track, start=min(onset, stmt.start), end=max(onset, stmt.end),
                conf=stmt.confidence * 0.9,
                desc=(f"The {'counter' if counter else 'emphasis'} animation of “{_short(content, 30)}” starts at {onset:.2f} s, but {stmt.label} is spoken between {stmt.start:.2f} s and "
                      f"{stmt.end:.2f} s: it lands {_fmt(miss_ms)} {side} the word."),
                why="Emphasis only works on the beat of the word it stresses; off the beat it distracts instead of stressing.",
                current=f"{'-' if early_ms > 0 else '+'}{miss_ms:.0f} ms from the spoken word", rec=f"Emphasis within {cfg.emphasis_miss_ms:.0f} ms of the spoken word",
                fix_text="Review the graphic's timing on the timeline.", fix=self._open(clip), impact=_impact(miss_ms, 0.35), sig=f"{clip.id}:emph:{round(miss_ms / 200)}",
                affected=f"Graphic {clip.slot or clip.id}", metrics={"miss_ms": round(miss_ms, 1), "spoken_start": round(stmt.start, 3), "spoken_end": round(stmt.end, 3)}))
        for track, clip in st.bucket.graphics:
            if "highlight" not in (clip.effects or {}):
                continue
            self._evidence(ctx, st, track, clip, cfg, issues)

    def _text_statement(self, content: str, variant: str, st: _S, hint: float) -> _Stmt | None:
        """The spoken words a text graphic is about: a figure by its numeric mention, anything else by its words in order."""
        if variant in ("NUMBER", "DATE"):
            cv, ct = _value(content), _toks(content)
            best: tuple[float, list[Word], Any] | None = None
            for m in st.scene.numbers:
                ws = [st.wmap[i] for i in m.word_ids if i in st.wmap]
                if not ws:
                    continue
                if (m.value is not None and cv is not None and abs(m.value - cv) < 1e-9) or (ct and ct == _toks(m.text)):
                    d = abs(ws[0].start - hint)
                    if best is None or d < best[0]:
                        best = (d, ws, m)
            if best is not None:
                ws = best[1]
                return _Stmt(ws[0].start, ws[-1].end, 92.0, f"“{_short(best[2].text, 24)}”")
        run = _match_run(_toks(content), st.pairs, hint)
        if run is None:
            return None
        return _Stmt(run.first.start, run.last.end, _confidence(run.ratio, len(_toks(content))), f"“{_short(content, 24)}”")

    def _evidence(self, ctx: QCContext, st: _S, track: Track, clip: Clip, cfg: SyncThresholds, issues: list[QCIssue]) -> None:
        """An evidence highlight is late when it only appears after the statement it points at has been completely said (the engine's own zoom-in delay is inside the sentence)."""
        sc = st.sc
        ends: list[float] = []
        if sc is not None:
            by_id = {s.sentence_id: s for s in sc.sentences}
            ends = [by_id[c.sentence_id].end for c in st.scene.claims if c.requires_evidence and c.sentence_id in by_id]
        conf = 80.0
        if ends:
            end = max(ends)
        else:
            end, conf = st.words[-1].end, 60.0  # no claim to point at: the whole narration of the scene is the statement
        onset = _onset(clip)
        lag_ms = (onset - end) * 1000.0
        if lag_ms <= 0 or not _reaches(lag_ms, cfg.visual_late_ms, cfg):
            return
        sev = _band(lag_ms, cfg)
        if sev is None:
            return
        issues.append(self._mk(
            ctx, st, "sync.important_visual_late", sev, f"Evidence highlight appears {_fmt(lag_ms)} after its statement", clip=clip, track=track, start=end, end=onset, conf=conf,
            desc=(f"The evidence highlight starts at {onset:.2f} s; the statement it supports ends at {end:.2f} s ({_fmt(lag_ms)} earlier)."),
            why="A highlight is meant to show where the evidence is while the claim is being made; after the statement it can no longer support it.",
            current=f"+{lag_ms:.0f} ms after the statement ended", rec="Highlight while the claim is being spoken", fix_text="Review the highlight's start on the timeline.", fix=self._open(clip),
            impact=_impact(lag_ms, 0.5), sig=f"{clip.id}:evidence-late:{round(lag_ms / 250)}", affected=f"Highlight {clip.slot or clip.id}",
            metrics={"offset_ms": round(lag_ms, 1), "statement_end": round(end, 3)}))

    # ------------------------------------------------------------------ cuts and transitions inside key phrases
    def _phrases(self, st: _S) -> list[_Phrase]:
        out: list[_Phrase] = []
        for m in st.scene.numbers:
            ws = [st.wmap[i] for i in m.word_ids if i in st.wmap]
            if ws:
                out.append(_Phrase(f"figure “{_short(m.text, 24)}”", ws[0].start, ws[-1].end, "number", ws, 90.0))
        if st.sc is not None:
            by_id = {s.sentence_id: s for s in st.sc.sentences}
            seen: set[str] = set()
            for c in st.scene.claims:
                s = by_id.get(c.sentence_id)
                if not c.requires_evidence or s is None or c.sentence_id in seen:
                    continue
                seen.add(c.sentence_id)
                ws = [st.wmap[i] for i in s.word_ids if i in st.wmap]
                if ws:
                    out.append(_Phrase(f"claim “{_short(c.text, 40)}”", ws[0].start, ws[-1].end, "claim", ws, 75.0))
        return out

    def _cuts(self, ctx: QCContext, st: _S, idx: _Index, issues: list[QCIssue]) -> None:
        cfg = ctx.settings.sync
        phrases = self._phrases(st)
        if not phrases or not st.bucket.visuals:
            return
        margin = cfg.minor_ms / 1000.0
        cuts: dict[int, tuple[float, Clip, Track, str, str]] = {}
        for track, c in st.bucket.visuals:
            tr = c.transition or {}
            kind_name = str(tr.get("type", "CUT")).upper()
            dur = _num(tr.get("duration")) or 0.0
            if kind_name != "CUT" and dur > 0:
                t, kind = c.timeline_start + dur / 2.0, f"{kind_name.lower()} transition"
            else:
                t, kind = c.timeline_start, "cut"
            cuts.setdefault(round(t * 1000), (t, c, track, kind, "in"))
            leaves_to_picture = any(o is not c and abs(o.timeline_start - c.timeline_end) <= FOLLOW_FRAMES * ctx.frame for o, _sid in idx.visuals)
            if not leaves_to_picture and c.timeline_end < ctx.duration - 1e-3:
                cuts.setdefault(round(c.timeline_end * 1000), (c.timeline_end, c, track, "cut", "out"))
        for t, c, track, kind, _side in sorted(cuts.values(), key=lambda x: x[0]):
            hit: tuple[_Phrase, float] | None = None
            for p in sorted(phrases, key=lambda p: p.kind != "number"):  # a figure is more specific than the claim around it
                if p.kind == "number":
                    depth = min(t - p.start, p.end - t)
                else:  # inside a spoken word of the claim
                    depth = max((min(t - w.start, w.end - t) for w in p.words if w.start < t < w.end), default=-1.0)
                if depth > margin:
                    hit = (p, depth * 1000.0)
                    break
            if hit is None:
                continue
            p, depth_ms = hit
            sev = _band(depth_ms, cfg)
            if sev is None:
                continue
            issues.append(self._mk(
                ctx, st, "sync.cut_in_phrase", sev, f"{kind.capitalize()} falls inside a spoken {p.label.split(' ')[0]}", clip=c, track=track, start=p.start, end=p.end, conf=p.confidence,
                desc=(f"A {kind} at {t:.2f} s lands {_fmt(depth_ms)} inside the spoken {p.label} ({p.start:.2f} to {p.end:.2f} s)."),
                why="Changing the picture in the middle of a key figure or claim pulls attention away just as it is being said.",
                current=f"{kind} at {t:.2f} s, inside {p.start:.2f}-{p.end:.2f} s", rec=f"Change the picture before {p.start:.2f} s or after {p.end:.2f} s",
                fix_text="Move the cut into the pause before or after the phrase on the timeline.", fix=self._open(c), impact=_impact(depth_ms, 0.45),
                sig=f"{c.id}:cut:{round(t, 1)}:{p.label}", affected=f"Visual {c.slot or c.id}", metrics={"cut_time": round(t, 3), "depth_ms": round(depth_ms, 1), "kind": kind}))


# ---------------------------------------------------------------------------------------------- clip readers
def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cap_text(clip: Clip) -> str:
    return str((clip.text or {}).get("text") or " ".join(str(w.get("text", "")) for w in (clip.text or {}).get("words") or [] if isinstance(w, dict)))


def _preset(clip: Clip) -> str:
    spec = (clip.animation or {}).get("in")
    return str(spec.get("preset", "")) if isinstance(spec, dict) else ""


def _onset(clip: Clip) -> float:
    """When the clip's emphasis animation begins: the clip start plus the in-animation's delay."""
    spec = (clip.animation or {}).get("in")
    delay = _num(spec.get("delay")) if isinstance(spec, dict) else None
    return clip.timeline_start + max(0.0, delay or 0.0)


def _variant(t: dict[str, Any]) -> str:
    v = str(t.get("variant") or "").upper()
    if v:
        return v
    return {"NUMBER_CARD": "NUMBER", "DATE": "DATE", "LOWER_THIRD": "LOWER_THIRD", "ENTITY_NAME": "LOWER_THIRD", "HEADLINE": "HEADLINE", "WARNING": "WARNING"}.get(str(t.get("style") or "").upper(), "TEXT")


def _has_emphasis(clip: Clip) -> bool:
    t = clip.text or {}
    return bool(t.get("emphasis")) or bool(t.get("counter")) or _preset(clip) in ("pop", "counter", "highlight")
