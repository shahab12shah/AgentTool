"""QCContext: the frozen, read-only view of a project that every checker analyses.

The context holds a *detached deep copy* of the project (built on the caller's thread), so a QC job on a worker thread never reads objects the UI thread is
editing, and nothing a checker does can change the live project. Fixes are never executed from here: they go through QCFixEngine, on the UI thread, as commands.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from app.qc.settings import QCSettings
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT, Clip
from app.timeline.track import Track, TrackKind

if TYPE_CHECKING:  # pragma: no cover
    from app.analysis.models import Scene
    from app.editing.context import EditingContext, SceneContext
    from app.media.asset import Asset
    from app.project.project import Project
    from app.rendering.ffmpeg_service import FFmpegService
    from app.rendering.probe import MediaProbeService
    from app.transcription.models import Sentence, Word

VISUAL_TRACK_KINDS = (TrackKind.VIDEO, TrackKind.IMAGE)
QC_SECTIONS = ("qc_issues", "qc_runs", "qc_history", "qc_fixes", "qc_ignored_issues", "qc_cache", "qc_scores", "render_qc_results")  # not needed (and not copied) inside the snapshot

ProgressFn = Callable[[float, str], None]


class QCCancelled(Exception):
    """The user cancelled the QC run."""


def snapshot_project(project: "Project") -> "Project":
    """A detached copy of ``project`` (same media root) that is safe to read from another thread."""
    from app.project.project import Project  # noqa: PLC0415

    doc = copy.deepcopy(project.to_document())
    for k in QC_SECTIONS:
        doc[k] = {} if isinstance(doc.get(k), dict) else []
    clone = Project.from_document(doc, root=project.root)
    clone.qc_settings = copy.deepcopy(project.qc_settings)
    return clone


def _canon(v: Any) -> Any:
    """Canonical form for hashing: 15 and 15.0 are the same number (a saved-and-reopened project must hash like the live one), containers are walked, everything else is str()'d."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_canon(x) for x in v]
    return str(v)


def sha(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(_canon(list(parts)), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:14]


@dataclass
class QCContext:
    project: "Project"  # detached copy: read-only by convention
    settings: QCSettings
    ffmpeg: "FFmpegService | None" = None
    probe: "MediaProbeService | None" = None
    cancel: threading.Event = field(default_factory=threading.Event)
    scene_filter: set[str] | None = None  # when set, scene-local checkers analyse only these scenes
    rendered_file: Path | None = None  # a render to inspect (frame checks); None = timeline-only QC
    now: float = 0.0
    ai_provider: Any = None  # an AIEditorialQCProvider chosen by the service (None = built from the settings)
    shared: dict[str, Any] = field(default_factory=dict)  # checker id -> CheckerOutput of the checkers that ran before (the editorial review reads them)
    _edit: "EditingContext | None" = field(default=None, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ construction / control
    @classmethod
    def build(cls, project: "Project", settings: QCSettings | None = None, *, ffmpeg: "FFmpegService | None" = None, probe: "MediaProbeService | None" = None,
              scene_filter: Iterable[str] | None = None, rendered_file: Path | None = None, cancel: threading.Event | None = None, detach: bool = True) -> "QCContext":
        snap = snapshot_project(project) if detach else project
        return cls(snap, copy.deepcopy(settings or project.qc_settings), ffmpeg, probe, cancel or threading.Event(), set(scene_filter) if scene_filter else None, rendered_file)

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise QCCancelled()

    # ------------------------------------------------------------------ basics
    @property
    def root(self) -> Path | None:
        return self.project.root

    @property
    def fps(self) -> int:
        return max(1, int(self.project.settings.fps or 30))

    @property
    def frame(self) -> float:
        return 1.0 / self.fps

    @property
    def canvas(self) -> tuple[int, int]:
        return int(self.project.settings.width), int(self.project.settings.height)

    @property
    def timeline(self):
        return self.project.timeline

    @property
    def voice_duration(self) -> float | None:
        return self.project.voice_over.duration

    @property
    def duration(self) -> float:
        """The video's length: the voice-over (the master clock) when there is one, else the end of the last scene, else the timeline."""
        if self.project.voice_over.duration:
            return float(self.project.voice_over.duration)
        if self.project.scenes:
            return float(max(s.end for s in self.project.scenes))
        return float(self.timeline.duration)

    # ------------------------------------------------------------------ scenes / narration
    @property
    def scenes(self) -> "list[Scene]":
        return sorted(self.project.scenes, key=lambda s: s.start)

    def scene(self, scene_id: str | None) -> "Scene | None":
        return next((s for s in self.project.scenes if s.id == scene_id), None) if scene_id else None

    def scene_at(self, t: float) -> "Scene | None":
        return next((s for s in self.scenes if s.start - 1e-6 <= t < s.end + 1e-6), None)

    def target_scenes(self) -> "list[Scene]":
        """The scenes a scene-local checker must analyse in this run."""
        return [s for s in self.scenes if self.scene_filter is None or s.id in self.scene_filter]

    def scene_label(self, scene_id: str | None) -> str:
        s = self.scene(scene_id)
        return f"Scene {s.label}" if s else (scene_id or "project")

    @property
    def edit_context(self) -> "EditingContext":
        """The Phase 4 per-scene bundle (words, sentences, intent, assignment, asset info, neighbours): built once, from the detached copy."""
        if self._edit is None:
            from app.editing.context import build_context  # noqa: PLC0415

            self._edit = build_context(self.project)
        return self._edit

    def scene_ctx(self, scene_id: str) -> "SceneContext | None":
        return next((c for c in self.edit_context.scenes if c.scene.id == scene_id), None)

    @property
    def words(self) -> "list[Word]":
        tr = self.project.transcription.transcript
        return list(tr.words) if tr else []

    @property
    def sentences(self) -> "list[Sentence]":
        tr = self.project.transcription.transcript
        return list(tr.sentences) if tr else []

    def words_between(self, start: float, end: float) -> "list[Word]":
        tr = self.project.transcription.transcript
        return tr.words_between(start, end) if tr else []

    # ------------------------------------------------------------------ timeline access
    def tracks(self, kind: TrackKind | None = None) -> list[Track]:
        return [t for t in self.timeline.tracks if kind is None or t.kind is kind]

    def clips(self, *, kind: str | None = None, track_kinds: tuple[TrackKind, ...] | None = None, scene_id: str | None = None) -> list[tuple[Track, Clip]]:
        out = []
        for t in self.timeline.tracks:
            if track_kinds is not None and t.kind not in track_kinds:
                continue
            for c in sorted(t.clips, key=lambda c: (c.timeline_start, c.id)):
                if kind is not None and c.kind != kind:
                    continue
                if scene_id is not None and c.scene_id != scene_id:
                    continue
                out.append((t, c))
        return out

    def visual_clips(self) -> list[tuple[Track, Clip]]:
        """What the viewer sees as picture: media clips on video / image tracks, in track order then time."""
        return self.clips(kind=KIND_MEDIA, track_kinds=VISUAL_TRACK_KINDS)

    def caption_clips(self) -> list[tuple[Track, Clip]]:
        return self.clips(kind=KIND_CAPTION)

    def text_clips(self) -> list[tuple[Track, Clip]]:
        return self.clips(kind=KIND_TEXT)

    def graphic_clips(self) -> list[tuple[Track, Clip]]:
        return self.clips(kind=KIND_GRAPHIC)

    def audio_clips(self, role: str | None = None) -> list[tuple[Track, Clip]]:
        out = []
        for t, c in self.clips(kind=KIND_MEDIA, track_kinds=(TrackKind.AUDIO,)):
            if role is None or str(c.audio.get("role", "")).upper() == role.upper():
                out.append((t, c))
        return out

    def clip_scene_id(self, clip: Clip) -> str | None:
        """The scene a clip belongs to: its own ``scene_id``, else the scene it mostly overlaps."""
        if clip.scene_id:
            return clip.scene_id
        mid = clip.timeline_start + clip.duration / 2
        s = self.scene_at(mid)
        return s.id if s else None

    def clips_in(self, start: float, end: float, *, kinds: Iterable[str] | None = None) -> list[tuple[Track, Clip]]:
        ks = set(kinds) if kinds else None
        return [(t, c) for t, c in self.clips() if c.timeline_end > start + 1e-6 and c.timeline_start < end - 1e-6 and (ks is None or c.kind in ks)]

    def track_audible(self, track: Track) -> bool:
        if track.hidden or track.muted:
            return False
        solo = any(t.solo for t in self.timeline.tracks if t.kind is TrackKind.AUDIO)
        return not solo or track.solo

    # ------------------------------------------------------------------ assets
    def asset(self, asset_id: str | None) -> "Asset | None":
        return self.project.assets.get(asset_id) if asset_id and asset_id in self.project.assets else None

    def asset_path(self, asset: "Asset") -> Path:
        return self.project.asset_path(asset)

    # ------------------------------------------------------------------ ownership / locks (QC reports on locked things but never fixes them)
    def locked_scene_ids(self) -> set[str]:
        p = self.project
        return set(p.timeline_generation.locked_scenes) | set(p.presentation_generation.locked_scenes)

    def is_protected(self, track: Track | None, clip: Clip | None) -> tuple[bool, str]:
        """(protected, reason). A protected element is never modified by QC: user-created, locked, on a locked track, or in a locked scene."""
        if clip is None:
            return False, ""
        if track is not None and track.locked:
            return True, f"Track {track.name} is locked"
        if clip.locked:
            return True, "This element is locked by you"
        if str(clip.created_by).upper() == "USER":
            return True, "This element was edited or created by you"
        if clip.scene_id and clip.scene_id in self.locked_scene_ids():
            return True, "The scene is locked"
        dec = self.project.editing_decisions.get(clip.ai_decision_id) or self.project.presentation_decisions.get(clip.ai_decision_id)
        if dec is not None and getattr(dec, "locked", False):
            return True, "The decision behind this element is locked"
        return False, ""

    def in_intentional_gap(self, start: float, end: float) -> bool:
        return any(g[0] - 1e-6 <= start and end <= g[1] + 1e-6 for g in self.settings.intentional_gaps)

    def in_intentional_black(self, start: float, end: float) -> bool:
        return any(g[0] - 1e-6 <= start and end <= g[1] + 1e-6 for g in self.settings.frames.intentional_black)

    # ------------------------------------------------------------------ cache fingerprints (what a checker's result depends on)
    def memo(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def timeline_hash(self) -> str:
        return self.memo("h.timeline", lambda: sha([[t.id, t.kind.value, t.hidden, t.muted, t.locked, t.solo, round(t.volume, 3), [c.to_dict() for c in sorted(t.clips, key=lambda c: (c.timeline_start, c.id))]]
                                                      for t in self.timeline.tracks], self.canvas, self.fps))

    def scenes_hash(self) -> str:
        return self.memo("h.scenes", lambda: sha([[s.id, s.label, round(s.start, 3), round(s.end, 3), s.narration, round(s.importance, 3), s.topic] for s in self.scenes],
                                                    {k: v.type.value for k, v in self.project.visual_intents.items()}))

    def transcript_hash(self) -> str:
        return self.memo("h.transcript", lambda: sha([[w.word_id, round(w.start, 3), round(w.end, 3), w.text] for w in self.words]))

    def assets_hash(self) -> str:
        def calc() -> str:
            rows = []
            for a in sorted(self.project.assets.all(), key=lambda a: a.id):
                try:
                    st = self.asset_path(a).stat()
                    fs = [st.st_size, int(st.st_mtime)]
                except OSError:
                    fs = None
                rows.append([a.id, a.content_hash, a.duration, a.width, a.height, fs])
            return sha(rows)

        return self.memo("h.assets", calc)

    def audio_hash(self) -> str:
        p = self.project
        return self.memo("h.audio", lambda: sha(p.audio_settings, p.audio_processing, [d for d in p.ducking_events], p.voice_over.asset_id, p.voice_over.duration,
                                                  p.audio_analysis.audio_hash if p.audio_analysis else "", p.render_settings.audio_sample_rate))

    def captions_hash(self) -> str:
        p = self.project
        return self.memo("h.captions", lambda: sha(p.caption_settings, p.caption_styles, [c.to_dict() for _t, c in self.caption_clips()]))

    def visual_hash(self) -> str:
        p = self.project
        return self.memo("h.visual", lambda: sha({k: (v.asset_id, v.approved, v.skipped, v.accuracy_score, v.candidate_id) for k, v in p.visual_assignments.items()},
                                                   {k: getattr(v, "total", None) for k, v in p.candidate_scores.items()}))

    def reference_hash(self) -> str:
        p = self.project
        return self.memo("h.reference", lambda: sha(p.reference_settings, p.reference_style_overrides, p.reference_style_profile.signature() if p.reference_style_profile else ""))

    def render_hash(self) -> str:
        p = self.project
        return self.memo("h.render", lambda: sha(p.render_settings, p.settings, p.proxies, p.timeline_generation.version if hasattr(p.timeline_generation, "version") else 0))

    def basis_hash(self) -> str:
        """Global facts that most checkers read without a domain of their own, so every checker's cache key and every scene's key covers them: the narration (the master clock: its
        asset and length), which scenes / decisions are locked (they decide what may be fixed), the editing brief per scene, the editing style, the export resolution and how findings are filtered / capped by confidence."""
        def calc() -> str:
            p = self.project
            briefs = {sid: [bool(getattr(b, "keep_static", False)), bool(getattr(b, "evidence_treatment_needed", False))]
                      for sid, b in (getattr(p.editing_strategy, "briefs", None) or {}).items()} if p.editing_strategy else {}
            return sha(p.voice_over.asset_id, p.voice_over.duration, self.duration, sorted(self.locked_scene_ids()),
                       sorted(d.decision_id for d in p.editing_decisions.values() if getattr(d, "locked", False)),
                       sorted(d.decision_id for d in p.presentation_decisions.values() if getattr(d, "locked", False)),
                       briefs, str(getattr(p.editing_settings, "style", "")), p.render_settings.resolution,
                       self.settings.min_confidence_to_report, self.settings.ai_confidence_caps)  # the last two filter / cap what a checker returns: cached findings were cut with the old values

        return self.memo("h.basis", calc)

    def global_signature(self, domains: Iterable[str]) -> str:
        """The part of a checker's inputs that belongs to NO single scene (the frame size and rate, the caption settings and styles, the visual preferences ...). A scene's own key
        must cover it too, otherwise a changed global value leaves every scene 'unchanged' and the old findings are re-used."""
        ds = set(domains)
        p = self.project

        def calc() -> str:
            parts: list[Any] = [self.canvas, self.fps]
            if "captions" in ds:
                parts += [p.caption_settings, p.caption_styles]
            if "visual" in ds:
                parts.append(p.visual_preferences)
            if "audio" in ds:
                parts.append(self.audio_hash())
            if "reference" in ds:
                parts.append(self.reference_hash())
            if "render" in ds:
                parts.append(self.render_hash())
            return sha(*parts)

        return self.memo("h.global." + ",".join(sorted(ds)), calc)

    def domain_hash(self, domain: str) -> str:
        fn = {"timeline": self.timeline_hash, "scenes": self.scenes_hash, "transcript": self.transcript_hash, "assets": self.assets_hash, "audio": self.audio_hash,
              "captions": self.captions_hash, "visual": self.visual_hash, "reference": self.reference_hash, "render": self.render_hash}.get(domain)
        return fn() if fn else ""

    def shared_signature(self) -> str:
        """Fingerprint of the other checkers' findings (what the editorial review is based on)."""
        rows = []
        for cid in sorted(self.shared):
            out = self.shared[cid]
            rows.append([cid, sorted(i.fingerprint for i in getattr(out, "issues", [])), json.dumps(getattr(out, "metrics", {}), sort_keys=True, default=str)[:4000]])
        return sha(rows)

    def scene_signature(self, scene_id: str) -> str:
        """Everything about one scene that can change a scene-local checker's answer: its content and words, the clips that overlap it (any track), its visual assignment, and its neighbours' subject."""
        def calc() -> str:
            s = self.scene(scene_id)
            if s is None:
                return ""
            sc = self.scene_ctx(scene_id)
            words = [[w.word_id, round(w.start, 3), round(w.end, 3), w.text] for w in self.words_between(s.start, s.end)]
            near = self.clips_in(s.start - 0.5, s.end + 0.5)
            clips = [c.to_dict() for _t, c in near]
            flags = sorted({(t.id, t.hidden, t.muted, t.locked, t.solo, round(t.volume, 3)) for t, _c in near})  # a muted / locked / hidden track changes what the clips mean
            a = self.project.visual_assignments.get(scene_id)
            neigh = [(n.scene_id, n.topic, n.asset_id) for n in ((sc.prev, sc.next) if sc else ()) if n is not None]
            intent = self.project.visual_intents.get(scene_id)
            used = sorted({c.asset_id for _t, c in self.clips_in(s.start, s.end) if c.asset_id})
            facts = []
            for aid in used:  # the files behind this scene's clips (a replaced / missing / changed file changes the answer)
                ast = self.asset(aid)
                if ast is None:
                    facts.append([aid, None])
                    continue
                try:
                    st = self.asset_path(ast).stat()
                    facts.append([aid, ast.content_hash, st.st_size, int(st.st_mtime), ast.width, ast.height, ast.duration])
                except OSError:
                    facts.append([aid, ast.content_hash, None])
            return sha(facts, [s.id, s.label, round(s.start, 3), round(s.end, 3), s.narration, round(s.importance, 3), s.topic, [c.text for c in s.claims], [n.text for n in s.numbers]], words, clips, flags,
                       (a.asset_id, a.approved, a.skipped, a.accuracy_score, a.selected_by, a.candidate_id) if a else None, neigh, (intent.type.value, intent.primary_subject) if intent else None,
                       scene_id in self.locked_scene_ids())

        return self.memo(f"h.scene.{scene_id}", calc)
