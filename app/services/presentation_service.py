"""Application service for the professional presentation layer (Phase 5).

    UI -> PresentationService -> AudioEngine / CaptionEngine / MotionGraphicsEngine / PresentationAssembler -> Timeline -> Project

Voice-over is the master clock. Everything produced here is a normal editable timeline object (captions on V6, graphics on V4/V5, music
on A2, SFX on A3) plus a presentation decision; nothing is rendered or burned in.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from app.audio.analysis import AudioAnalysisService
from app.audio.backend import AudioError
from app.audio.ducking import AudioDuckingService, Important, speech_from_silence, speech_segments
from app.audio.engine import AudioEngine
from app.audio.mix import AudioMixService
from app.audio.processing import problems as processing_problems
from app.audio.sfx import SfxMoment, SfxPlanner
from app.captions.engine import CaptionEngine, break_lines, layout_for, retime_words
from app.captions.keywords import KeywordService
from app.captions.styles import PRESETS, effective_style, style_for
from app.core.commands import Command, CommandStack, CompositeCommand
from app.core.events import EventBus, Topics
from app.core.exceptions import AppError
from app.editing.context import EditingContext, SceneContext, build_context
from app.editing.models import Creator, DecisionType, TextGraphic, now_iso
from app.editing.strategy import RuleBasedProvider
from app.editing.validator import ValidationIssue
from app.jobs.job import Job
from app.jobs.job_manager import JobManager
from app.logging.logger import get_logger, log_event
from app.media.asset import AssetType
from app.presentation import animation as anim
from app.presentation.assembly import PresState, PresentationAssembler, own_clip, own_decision, owned, settings_hash
from app.presentation.graphics import EVIDENCE_TOOLS, GraphicsPlan, GraphicsPlanner
from app.presentation.models import (
    AudioSettings,
    CaptionSegment,
    CaptionSettings,
    CaptionStyle,
    CaptionWord,
    HighlightMode,
    Part,
    PresentationSession,
    PresentationType,
    PreviewMode,
    SfxCategory,
    VoiceAnalysis,
    VoiceProcessingSettings,
)
from app.presentation.validator import PresentationValidator
from app.project.phase5_commands import (
    ApplyPresentationCommand,
    MarkPresentationEditCommand,
    RecordPresentationDeleteCommand,
    SetAssetExtraCommand,
    SetSettingCommand,
)
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT
from app.timeline.keyframes import Keyframe
from app.timeline.track import TrackKind
from app.transcription.status import current_audio_hash

_log = get_logger(__name__)
Apply = Callable[[Command], None]
LOCK_ASPECTS = ("CAPTION", "GRAPHIC", "MUSIC", "SFX", "MIX", "SCENE")
POSITIONS = ("bottom", "center", "top", "custom")
ROLES = ("music", "sfx")


class PresentationError(AppError):
    """The presentation layer could not be created or changed."""


@dataclass
class GenOutcome:
    caption_plans: dict = field(default_factory=dict)  # scene id -> (segments, keywords)
    graphic_plans: dict = field(default_factory=dict)  # scene id -> GraphicsPlan
    failure: tuple[str, str, str] | None = None  # (part, scene id, message)
    done: dict = field(default_factory=lambda: {"CAPTIONS": [], "GRAPHICS": []})


class PresentationService:
    def __init__(self, projects: ProjectManager, commands: CommandStack, jobs: JobManager, bus: EventBus, apply: Apply, ffmpeg_path: Callable[[], str],
                 media, checkpoint: Callable[[Project, str], str]) -> None:
        self._projects, self._commands, self._jobs, self._bus, self._apply = projects, commands, jobs, bus, apply
        self._media, self._checkpoint = media, checkpoint
        self.audio = AudioEngine(ffmpeg_path, self._cache_root)
        self.keywords = KeywordService()
        self._running = False
        self.progress: dict = {"state": "IDLE", "part": "", "scene": 0, "total": 0, "operation": ""}
        self.music = MusicService(self)
        self.sfx = SFXService(self)

    # ------------------------------------------------------------------ basics
    def _project(self) -> Project:
        if self._projects.current is None or self._projects.current.root is None:
            raise PresentationError("Open or create a project first.")
        return self._projects.current

    def _cache_root(self) -> Path | None:
        p = self._projects.current
        return (p.root / "cache") if p is not None and p.root is not None else None

    @property
    def running(self) -> bool:
        return self._running

    def caption_styles(self) -> dict[str, CaptionStyle]:
        p = self._project()
        return {**PRESETS, **p.caption_styles}

    # ------------------------------------------------------------------ settings (all undoable, all persistent)
    def update_caption_settings(self, **changes) -> CaptionSettings:
        p = self._project()
        new = replace(p.caption_settings, **changes)
        if new.position not in POSITIONS:
            raise PresentationError(f"Unknown caption position “{new.position}”.")
        if new.style_id not in self.caption_styles():
            raise PresentationError(f"Unknown caption style “{new.style_id}”.")
        if new.highlight_mode not in {m.value for m in HighlightMode}:
            raise PresentationError(f"Unknown highlight mode “{new.highlight_mode}”.")
        if new.max_lines not in (1, 2):
            raise PresentationError("Captions can have 1 or 2 lines.")
        for k in ("safe_margin_left", "safe_margin_right", "safe_margin_top", "safe_margin_bottom"):
            if not (0.0 <= getattr(new, k) <= 0.4):
                raise PresentationError("Safe margins must be between 0% and 40%.")
        if new.safe_margin_left + new.safe_margin_right >= 0.8:
            raise PresentationError("The safe margins leave no room for text.")
        if not (0.3 <= new.reading_speed <= 2.0):
            raise PresentationError("Reading speed must be between 0.3 and 2.0.")
        self._commands.execute(SetSettingCommand(p, "caption_settings", new, "Change caption settings"))
        return new

    def update_audio_settings(self, **changes) -> AudioSettings:
        p = self._project()
        new = replace(p.audio_settings, **changes)
        for k in ("music_level", "important_level", "pause_level", "intro_level", "sfx_level", "voice_level"):
            if not (0.0 <= getattr(new, k) <= 1.5):
                raise PresentationError("Levels must be between 0% and 150%.")
        if new.important_level > new.music_level + 1e-9 or new.music_level > new.pause_level + 1e-9 and new.pause_level > 0:
            raise PresentationError("Levels must satisfy: important ≤ normal ≤ pause rise.")
        cmds: list[Command] = [SetSettingCommand(p, "audio_settings", new, "Change audio settings")]
        if "voice_enhancement" in changes and changes["voice_enhancement"] != p.audio_settings.voice_enhancement:
            proc = replace(p.audio_processing, enabled=bool(changes["voice_enhancement"]))
            if proc.enabled and p.audio_processing == VoiceProcessingSettings():
                proc = replace(proc, highpass_hz=80.0, compression=True, limiter=True)  # sensible starting chain, every value stays editable
            cmds.append(SetSettingCommand(p, "audio_processing", proc, "Voice enhancement"))
        self._commands.execute(cmds[0] if len(cmds) == 1 else CompositeCommand("Change audio settings", cmds, scope="editing"))
        return new

    def update_voice_processing(self, **changes) -> VoiceProcessingSettings:
        p = self._project()
        new = replace(p.audio_processing, **changes)
        bad = processing_problems(new)
        if bad:
            raise PresentationError(bad[0], details="; ".join(bad))
        self._commands.execute(SetSettingCommand(p, "audio_processing", new, "Change voice processing"))
        return new

    def save_caption_style(self, style: CaptionStyle) -> None:
        """Create or edit a style (presets can be edited too: the project copy wins)."""
        p = self._project()
        if not style.style_id.strip():
            raise PresentationError("A style needs an id.")
        styles = dict(p.caption_styles)
        styles[style.style_id] = style
        self._commands.execute(SetSettingCommand(p, "caption_styles", styles, "Edit caption style"))

    def reset_caption_style(self, style_id: str) -> None:
        p = self._project()
        if style_id in p.caption_styles:
            styles = {k: v for k, v in p.caption_styles.items() if k != style_id}
            self._commands.execute(SetSettingCommand(p, "caption_styles", styles, "Reset caption style"))

    # ------------------------------------------------------------------ voice dependency (master clock)
    def staleness(self) -> dict:
        """Which parts depend on a voice-over that has changed since they were generated."""
        p = self._project()
        cs, g = p.caption_settings, p.presentation_generation
        voice_hash = current_audio_hash(p)
        tr = p.transcription.transcript
        out = {"voice_hash": voice_hash, "captions_outdated": False, "analysis_outdated": False, "transcript_outdated": False, "message": "", "acknowledged": False,
               "settings_changed": False}
        if g.captions.status == "COMPLETE":
            changed = bool(voice_hash and cs.generated_audio_hash and cs.generated_audio_hash != voice_hash)
            tr_changed = bool(tr is not None and cs.generated_transcript_id and cs.generated_transcript_id != tr.transcript_id)
            out["captions_outdated"] = changed or tr_changed
            out["settings_changed"] = g.captions.input_hash != settings_hash(cs, (p.settings.width, p.settings.height)) and not out["captions_outdated"]
            out["acknowledged"] = bool(out["captions_outdated"] and cs.stale_acknowledged_hash == voice_hash)
        if p.audio_analysis is not None and voice_hash and p.audio_analysis.audio_hash != voice_hash:
            out["analysis_outdated"] = True
        if tr is not None and voice_hash and tr.audio.content_hash != voice_hash:
            out["transcript_outdated"] = True
        if out["captions_outdated"]:
            out["message"] = "Voice-over changed. Captions need regeneration." + (" (You chose to keep the existing captions.)" if out["acknowledged"] else "")
        return out

    def acknowledge_stale(self) -> None:
        """Keep Existing: remember that the user decided to keep the current captions for this voice-over."""
        p = self._project()
        self._commands.execute(SetSettingCommand(p, "caption_settings", replace(p.caption_settings, stale_acknowledged_hash=current_audio_hash(p) or ""), "Keep existing captions"))

    def caption_review(self) -> dict:
        """Review Changes: what a regeneration would do (nothing is changed)."""
        p = self._project()
        tr = p.transcription.transcript
        asm_clips = [c for t in p.timeline.tracks for c in t.clips if c.kind == KIND_CAPTION]
        kept = [c for c in asm_clips if owned(c)]
        removable = [c for c in asm_clips if not owned(c)]
        new_n = 0
        if tr is not None:
            ctx = build_context(p)
            for sc in ctx.scenes:
                segs, _k = self._plan_captions(sc, p, ctx)
                new_n += len(segs)
        return {"current": len(asm_clips), "preserved_user_or_locked": len(kept), "removed_ai": len(removable), "new_ai_estimate": new_n,
                "transcript_words": len(tr.words) if tr else 0, "stale": self.staleness()}

    def sync_voice_clip(self) -> bool:
        """Point the A1 voice clip at the current voice-over asset (after a replacement). Returns True when it changed anything."""
        p = self._project()
        vo = p.assets.get(p.voice_over.asset_id) if p.voice_over.asset_id else None
        if vo is None or not vo.duration:
            raise PresentationError("There is no voice-over to place.")
        asm = PresentationAssembler(p)
        asm.ensure_tracks()
        a1 = asm.state.timeline.get_track("track_a1")
        voice = next((c for c in a1.clips if c.slot == "voice" or c.audio.get("role") == "VOICE"), None)
        if voice is not None and voice.asset_id == vo.id and abs(voice.duration - vo.duration) < 0.02:
            return False
        if voice is None:
            from app.timeline.clip import Clip
            from app.timeline.timeline import new_clip_id

            a1.clips.append(Clip(new_clip_id(), "track_a1", vo.id, 0.0, vo.duration, 0.0, vo.duration, created_by="SYSTEM", slot="voice", audio={"role": "VOICE", "volume": 1.0}))
        else:
            voice.asset_id, voice.source_in, voice.source_out, voice.duration, voice.timeline_start = vo.id, 0.0, vo.duration, vo.duration, 0.0
        self._commands.execute(ApplyPresentationCommand(p, asm.state, "Update voice-over clip"))
        return True

    # ------------------------------------------------------------------ audio analysis, waveforms
    def voice_analysis(self) -> VoiceAnalysis | None:
        return self._project().audio_analysis

    def analyze_voice(self, force: bool = False) -> Job | None:
        """AudioAnalysisService in a background job (cached by the voice-over's content hash)."""
        p = self._project()
        vo = p.assets.get(p.voice_over.asset_id) if p.voice_over.asset_id else None
        if vo is None:
            raise PresentationError("Import the voice-over first.")
        h = current_audio_hash(p) or vo.content_hash or ""
        if p.audio_analysis is not None and p.audio_analysis.audio_hash == h and not force and (p.audio_analysis.speaking_rate_wps is not None or p.transcription.transcript is None):
            return None
        path, tr = p.asset_path(vo), p.transcription.transcript if p.transcription.transcript and p.transcription.transcript.audio.content_hash == h else None

        def work(ctx) -> VoiceAnalysis:
            return self.audio.analysis.analyze_voice(path, vo.id, h, vo.duration, tr, lambda pct, msg: ctx.report(pct, msg))

        def done(job: Job) -> None:
            if self._projects.current is p:
                self._apply(SetSettingCommand(p, "audio_analysis", job.result, "Voice analysis"))
                self._bus.publish(Topics.STATUS, message="Voice-over analysis complete.")

        def failed(job: Job) -> None:
            self._bus.publish(Topics.ERROR, message=job.error or "The voice-over could not be analysed.", title="Audio analysis")

        return self._jobs.submit("audio_analysis", work, title="Analysing the voice-over", on_complete=done, on_error=failed)

    def waveform(self, asset_id: str, request: bool = True):
        """A cached waveform, or None. With ``request`` a background job is started to create a missing one (never call that from a paint event)."""
        p = self._projects.current
        a = p.assets.get(asset_id) if p else None
        if a is None or a.type is not AssetType.AUDIO:
            return None
        key = a.content_hash or a.id
        wf = self.audio.waveforms.cached(key)
        if wf is None and request:
            self._request_waveform(p, a, key)
        return wf

    def _request_waveform(self, p: Project, a, key: str) -> None:
        pending = getattr(self, "_wf_pending", set())
        self._wf_pending = pending
        failed_keys = getattr(self, "_wf_failed", set())
        self._wf_failed = failed_keys
        if key in pending or key in failed_keys:  # a broken file is tried once; the audio report explains why it has no waveform
            return
        pending.add(key)
        path = p.asset_path(a)

        def work(ctx):
            return self.audio.waveforms.compute(path, key, lambda pct, msg: ctx.report(pct, msg))

        def done(job: Job) -> None:
            pending.discard(key)
            self._bus.publish(Topics.PROJECT_CHANGED, scope="waveform", command=None, action="do")

        def failed(job: Job) -> None:
            pending.discard(key)
            failed_keys.add(key)

        try:
            self._jobs.submit("waveform", work, title=f"Waveform: {a.name}", on_complete=done, on_error=failed, on_cancel=failed)
        except Exception:
            pending.discard(key)

    def audio_report(self) -> dict:
        p = self._project()
        a = p.audio_analysis
        return {"analysis": a, "issues": list(a.issues) if a else [], "outdated": self.staleness()["analysis_outdated"],
                "processing": p.audio_processing, "masking": [m.message for m in self.masking()]}

    def masking(self):
        """Where music or SFX would mask the voice (VoicePriorityController)."""
        p = self._project()
        plan = AudioMixService(None).build_plan(p, PreviewMode.FULL)  # type: ignore[arg-type]
        speech = self._speech(p)
        ctrl = self.audio.priority(p.audio_settings)
        out = []
        for role in ("MUSIC", "SFX"):
            out += ctrl.masking(role, lambda t, r=role: plan.gain_at(r, t), speech, plan.duration)
        return out

    # ------------------------------------------------------------------ library (music + SFX assets)
    def import_audio(self, path: Path, role: str, category: str | None = None) -> list[Job]:
        if role not in ROLES:
            raise PresentationError("Audio is imported as music or as a sound effect.")
        if role == "sfx" and category and category.upper() not in {c.value for c in SfxCategory}:
            raise PresentationError(f"Unknown SFX category “{category}”.")
        p = self._project()

        def tagged(asset) -> None:
            if asset.type is not AssetType.AUDIO:
                self._bus.publish(Topics.ERROR, message=f"{asset.name} is not an audio file.", title="Import audio")
                return
            extra = {"role": role}
            if role == "sfx":
                extra["category"] = (category or self._guess_category(asset.name)).upper()
            self._apply(SetAssetExtraCommand(p, asset.id, **extra))

        return self._media.import_files([Path(path)], on_asset=tagged)

    @staticmethod
    def _guess_category(name: str) -> str:
        low = name.lower()
        for cat in SfxCategory:
            if cat.value.lower() in low:
                return cat.value
        return SfxCategory.IMPACT.value

    def tag_asset(self, asset_id: str, role: str, category: str | None = None) -> None:
        p = self._project()
        a = p.assets.get(asset_id)
        if a is None or a.type is not AssetType.AUDIO:
            raise PresentationError("Choose an audio asset.")
        if role not in ROLES:
            raise PresentationError("The role must be music or sfx.")
        extra = {"role": role}
        if role == "sfx":
            extra["category"] = (category or a.extra.get("category") or SfxCategory.IMPACT.value).upper()
            if extra["category"] not in {c.value for c in SfxCategory}:
                raise PresentationError(f"Unknown SFX category “{category}”.")
        self._commands.execute(SetAssetExtraCommand(p, asset_id, **extra))

    def library(self, role: str) -> list:
        p = self._project()
        vo = p.voice_over.asset_id
        out = []
        for a in p.assets.all():
            if a.type is not AssetType.AUDIO or a.id == vo:
                continue
            r = a.extra.get("role") or ("music" if (a.duration or 0) > 20 else "sfx")
            if r == role:
                out.append(a)
        return out

    def sfx_library(self) -> dict[str, list]:
        lib: dict[str, list] = {}
        for a in self.library("sfx"):
            lib.setdefault(str(a.extra.get("category", SfxCategory.IMPACT.value)).upper(), []).append((a.id, a.duration))
        return lib

    # ------------------------------------------------------------------ shared inputs
    @staticmethod
    def _words(p: Project):
        tr = p.transcription.transcript
        return tr.words if tr else []

    def _speech(self, p: Project) -> list[tuple[float, float]]:
        words = self._words(p)
        if words:
            return speech_segments(words)
        a = p.audio_analysis
        vo = p.assets.get(p.voice_over.asset_id) if p.voice_over.asset_id else None
        if a is not None and vo is not None and vo.duration:
            return speech_from_silence(vo.duration, a.silence_regions)
        return []

    @staticmethod
    def _total_duration(p: Project) -> float:
        vo = p.assets.get(p.voice_over.asset_id) if p.voice_over.asset_id else None
        return max([vo.duration or 0.0 if vo else 0.0] + [s.end for s in p.scenes] + [0.0])

    # ------------------------------------------------------------------ plans (pure, run in the worker)
    def _caption_cache_file(self, sc: SceneContext, p: Project, ctx: EditingContext) -> Path | None:
        import hashlib
        import json as _json

        if p.root is None:
            return None
        cs = p.caption_settings
        audio = sorted(p.audio_analysis.emphasis_candidates) if p.audio_analysis else []
        styles = {k: v for k, v in p.caption_styles.items()}
        key = hashlib.sha1(_json.dumps([sc.scene.id, [(w.word_id, w.text, round(w.start, 3), round(w.end, 3)) for w in sc.words], settings_hash(cs, ctx.canvas), audio,
                                        [(n.text, n.kind.value) for n in sc.scene.numbers], [(e.text, e.type.value) for e in sc.scene.entities],
                                        {k: v.__dict__ for k, v in styles.items()}], default=str).encode()).hexdigest()[:20]
        return p.root / "cache" / "presentation" / "captions" / f"{key}.json"

    def _plan_captions(self, sc: SceneContext, p: Project, ctx: EditingContext):
        """Caption segments + keywords of one scene; cached on disk by the words, the settings and the analysis (unchanged scenes are not recomputed)."""
        import json as _json

        from app.captions.keywords import Keyword

        f = self._caption_cache_file(sc, p, ctx)
        try:
            if f is not None and f.is_file():
                d = _json.loads(f.read_text(encoding="utf-8"))
                return [CaptionSegment.from_dict(x) for x in d["segments"]], [Keyword(**k) for k in d["keywords"]]
        except Exception:
            _log.warning("Ignoring unreadable caption plan cache %s", f)
        segs, kws = self._plan_captions_fresh(sc, p, ctx)
        try:
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(_json.dumps({"segments": [s.to_dict() for s in segs], "keywords": [k.__dict__ for k in kws]}), encoding="utf-8")
        except OSError:
            _log.debug("Could not write the caption plan cache", exc_info=True)
        return segs, kws

    def _plan_captions_fresh(self, sc: SceneContext, p: Project, ctx: EditingContext):
        cs = p.caption_settings
        eng = CaptionEngine(cs, self.caption_styles(), ctx.canvas)
        words = [CaptionWord(w.word_id, w.text, w.start, w.end) for w in sc.words]
        if not words:
            return [], []
        in_scene = {w.word_id for w in words}
        groups = []
        for sent in sc.sentences:
            ws = [w for w in words if w.word_id in set(sent.word_ids) & in_scene]
            if ws:
                groups.append(ws)
        covered = {w.word_id for g in groups for w in g}
        leftover = [w for w in words if w.word_id not in covered]
        if leftover:
            groups.append(leftover)
            groups.sort(key=lambda g: g[0].start)
        audio_emph = set(p.audio_analysis.emphasis_candidates) if p.audio_analysis else set()
        kws = self.keywords.detect(sc.scene, sc.words, audio_emph) if (cs.keyword_highlight or cs.number_emphasis) else []
        segs = eng.segment(sc.scene.id, groups, kws, scene_end=sc.scene.end)
        return segs, kws

    def _plan_graphics(self, sc: SceneContext, ctx: EditingContext, provider: RuleBasedProvider, profile, planner: GraphicsPlanner) -> GraphicsPlan:
        brief = provider.build_brief(sc, ctx, profile)
        gp = planner.plan_scene(sc, brief)
        if brief.evidence_treatment_needed and ctx.settings.evidence_treatment:
            gp.evidence = planner.evidence_tool(sc)
        return gp

    # ------------------------------------------------------------------ generation
    def generate(self, parts: list[str] | None = None, scene_ids: list[str] | None = None, scope: str | None = None) -> Job | None:
        """Generate captions / graphics / audio for the whole video or some scenes. One background job, one undoable commit."""
        p = self._project()
        if self._running:
            raise PresentationError("The presentation is already being created. Wait for it to finish.")
        if not p.scenes:
            raise PresentationError("There are no scenes yet. Analyse the voice-over into scenes first.")
        wanted = [x for x in (parts or [pt.value for pt in Part])]
        bad = [x for x in wanted if x not in {pt.value for pt in Part}]
        if bad:
            raise PresentationError(f"Unknown part “{bad[0]}”.")
        if (Part.CAPTIONS.value in wanted or Part.GRAPHICS.value in wanted) and not self._words(p):
            raise PresentationError("Transcribe the voice-over first: captions and graphics follow the word timestamps.")
        if Part.CAPTIONS.value in wanted and not p.caption_settings.enabled:
            wanted.remove(Part.CAPTIONS.value)
        if not wanted:
            raise PresentationError("Captions are switched off and nothing else was requested.")
        order = [s.id for s in p.scenes]
        ids = [i for i in order if scene_ids is None or i in scene_ids]
        if not ids:
            raise PresentationError("None of the requested scenes exist.")
        ok, why = self.audio.available()
        if Part.AUDIO.value in wanted and not ok and p.audio_settings.music_enabled and self.library("music"):
            raise PresentationError(f"Audio tools are unavailable: {why}")
        ctx = build_context(p)
        todo = [i for i in ids if i not in p.presentation_generation.locked_scenes]
        scope = scope or ("ALL" if scene_ids is None else "SCENE" if len(ids) == 1 else "SELECTED")
        session = PresentationSession(f"pres_{len(p.presentation_sessions) + 1:04d}", wanted, scope, list(todo), "RUNNING")
        p.presentation_sessions.append(session)
        self._running = True
        self.progress = {"state": "RUNNING", "part": "", "scene": 0, "total": len(todo), "operation": "Starting"}
        if len(todo) < len(ids):
            session.log.append(f"{len(ids) - len(todo)} locked scene(s) skipped.")
        provider = RuleBasedProvider()
        profile = provider.analyze_video(ctx)
        planner = GraphicsPlanner(ctx, p.caption_settings.reduced_motion)
        steps = [Part.CAPTIONS.value, Part.GRAPHICS.value]

        def work(jc) -> GenOutcome:
            out = GenOutcome()
            for n, sid in enumerate(todo, start=1):
                if jc.is_cancelled():
                    break
                sc = ctx.by_id(sid)
                for part in steps:
                    if part not in wanted:
                        continue
                    self.progress.update(part=part, scene=n, operation=f"{part.title()}: scene {sc.scene.label}")
                    jc.report(100.0 * (n - 1) / max(1, len(todo)), f"Scene {n} / {len(todo)} — {part.title()}")
                    try:
                        if part == Part.CAPTIONS.value:
                            out.caption_plans[sid] = self._plan_captions(sc, p, ctx)
                        else:
                            out.graphic_plans[sid] = self._plan_graphics(sc, ctx, provider, profile, planner)
                        out.done[part].append(sid)
                    except Exception as exc:
                        _log.exception("Presentation %s failed for scene %s", part, sid)
                        out.failure = (part, sid, getattr(exc, "user_message", None) or str(exc) or type(exc).__name__)
                        session.log.append(f"Scene {sc.scene.label}: {part} FAILED — {out.failure[2]}")
                        return out
            return out

        def done(job: Job) -> None:
            self._running = False
            if self._projects.current is not p:
                return
            self._commit(p, ctx, session, wanted, todo, job.result)

        def failed(job: Job) -> None:
            self._running = False
            session.status, session.error, session.finished_at = "FAILED", job.error or "The presentation failed.", now_iso()
            self.progress["state"] = "FAILED"
            self._bus.publish(Topics.ERROR, message=session.error, title="Presentation")

        def cancelled(job: Job) -> None:
            self._running = False
            session.status, session.finished_at = "CANCELED", now_iso()
            session.log.append("Canceled: nothing was changed.")
            self.progress["state"] = "CANCELED"
            self._bus.publish(Topics.STATUS, message="Presentation canceled — the timeline was not changed.")

        log_event(_log, "presentation.started", parts=wanted, scenes=len(todo))
        return self._jobs.submit("presentation", work, title=f"Presentation ({', '.join(x.title() for x in wanted)})", on_complete=done, on_error=failed, on_cancel=cancelled)

    def regenerate_captions(self, scene_ids: list[str] | None = None) -> Job | None:
        return self.generate([Part.CAPTIONS.value], scene_ids)

    def regenerate_graphics(self, scene_ids: list[str] | None = None) -> Job | None:
        return self.generate([Part.GRAPHICS.value], scene_ids)

    def regenerate_audio(self, scene_ids: list[str] | None = None) -> Job | None:
        return self.generate([Part.AUDIO.value], scene_ids)

    def regenerate_scene(self, scene_id: str) -> Job | None:
        return self.generate(None, [scene_id], "SCENE")

    def regenerate_all(self) -> Job | None:
        return self.generate(None, None, "ALL")

    def retry_failed(self) -> Job | None:
        """Resume at the scene/part that failed; scenes already done are not redone."""
        p = self._project()
        st = p.presentation_generation.scene_status
        order = [s.id for s in p.scenes]
        for part in (Part.CAPTIONS.value, Part.GRAPHICS.value):
            ids = [i for i in order if st.get(part, {}).get(i) in ("FAILED", "PENDING")]
            if ids:
                return self.generate([part], ids, "RETRY")
        raise PresentationError("There are no failed or pending scenes to retry.")

    def cancel(self) -> None:
        for j in self._jobs.active_jobs():
            if j.type == "presentation":
                self._jobs.cancel(j.id)

    # ------------------------------------------------------------------ commit (UI thread)
    def _commit(self, p: Project, ctx: EditingContext, session: PresentationSession, wanted: list[str], todo: list[str], out: GenOutcome) -> None:
        try:
            asm = PresentationAssembler(p)
            asm.ensure_tracks()
            before_state = PresState.capture(p)
            before = {(i.code, i.clip_id, i.message) for i in PresentationValidator(p, before_state).validate() if i.severity == "error"}
            placed: dict[str, list[str]] = {pt: [] for pt in wanted}
            for sid, (segs, kws) in out.caption_plans.items():
                asm.captions(sid, segs, kws)
                placed[Part.CAPTIONS.value].append(sid)
            for sid, gp in out.graphic_plans.items():
                asm.graphics(gp)
                placed[Part.GRAPHICS.value].append(sid)
            audio_ids: set[str] = set()
            if Part.AUDIO.value in wanted:
                audio_ids = set(todo)
                self._commit_audio(asm, p, ctx, audio_ids)
                placed[Part.AUDIO.value] = sorted(audio_ids)
            g = asm.state.generation
            cs_hash = settings_hash(p.caption_settings, ctx.canvas)
            tr = p.transcription.transcript
            for part in wanted:
                done_ids = set(placed.get(part, []))
                if part == Part.AUDIO.value and not out.failure:
                    done_ids = set(todo)
                status = g.scene_status.setdefault(part, {})
                for sid in todo:
                    if sid in done_ids:
                        status[sid] = "COMPLETE"
                if out.failure and out.failure[0] == part:
                    fi = todo.index(out.failure[1]) if out.failure[1] in todo else None
                    for i, sid in enumerate(todo):
                        if fi is not None and i == fi:
                            status[sid] = "FAILED"
                        elif fi is not None and i > fi and sid not in done_ids:
                            status[sid] = "PENDING"
                if part in (Part.CAPTIONS.value, Part.GRAPHICS.value) and out.failure and out.failure[0] != part and not placed.get(part):
                    pass
                if not (out.failure and out.failure[0] == part and not done_ids):
                    asm.finish_part(part.lower(), cs_hash if part == Part.CAPTIONS.value else str(len(todo)))
                else:
                    setattr(g, part.lower(), type(getattr(g, part.lower()))(getattr(g, part.lower()).version, "FAILED", now_iso(), "", out.failure[2]))
            if Part.CAPTIONS.value in wanted and placed.get(Part.CAPTIONS.value):
                asm.state.caption_settings.generated_transcript_id = tr.transcript_id if tr else ""
                asm.state.caption_settings.generated_audio_hash = current_audio_hash(p) or ""
                asm.state.caption_settings.stale_acknowledged_hash = ""
            errors = self._new_errors(p, asm.state, before)
            session.completed = sorted({sid for ids in placed.values() for sid in ids}, key=[s.id for s in p.scenes].index)
            if errors:
                session.status, session.validation_errors, session.finished_at = "FAILED", [str(e) for e in errors[:20]], now_iso()
                session.error = "The generated presentation did not pass validation, so the current timeline was kept."
                self.progress["state"] = "FAILED"
                self._bus.publish(Topics.ERROR, message=session.error, title="Presentation", details="; ".join(session.validation_errors[:5]))
                return
            session.checkpoint = self._checkpoint(p, "before_presentation")
            self._commands.execute(ApplyPresentationCommand(p, asm.state, "Generate presentation"))
            session.failed_part, session.failed_scene = (out.failure[0], out.failure[1]) if out.failure else ("", "")
            session.error = out.failure[2] if out.failure else ""
            session.status = "FAILED" if out.failure else "COMPLETED"
            session.finished_at = now_iso()
            session.log.append(f"Committed {len(session.completed)} scene(s) as one undo step.")
            self.progress["state"] = session.status
            self._bus.publish(Topics.STATUS, message=("Presentation stopped at a failed scene; earlier scenes were kept." if out.failure else "Presentation complete."))
        except Exception as exc:  # nothing was installed: the assembler works on copies
            _log.exception("Presentation commit failed")
            session.status, session.error, session.finished_at = "FAILED", getattr(exc, "user_message", None) or str(exc), now_iso()
            self.progress["state"] = "FAILED"
            self._bus.publish(Topics.ERROR, message=f"The presentation could not be applied; the timeline was not changed. {session.error}", title="Presentation")

    def _new_errors(self, p: Project, state: PresState, baseline: set) -> list[ValidationIssue]:
        return [i for i in PresentationValidator(p, state).validate() if i.severity == "error" and (i.code, i.clip_id, i.message) not in baseline]

    # ------------------------------------------------------------------ audio commit: SFX -> ducking -> music
    def _sfx_moments(self, asm: PresentationAssembler, ctx: EditingContext, scene_ids: set[str]) -> list[SfxMoment]:
        out: list[SfxMoment] = []
        order = {s.scene.id: s for s in ctx.scenes}
        for t in asm.state.timeline.tracks:
            for c in t.clips:
                sid = c.scene_id
                if sid not in scene_ids or sid not in order:
                    continue
                imp = order[sid].scene.importance
                if c.kind == KIND_TEXT and c.text:
                    style, emph = c.text.get("style"), c.text.get("emphasis")
                    if style == "NUMBER_CARD":
                        out.append(SfxMoment(sid, c.timeline_start, "NUMBER", 0.55 + 0.35 * imp, c.text.get("content", "")))
                    elif style == "WARNING":
                        out.append(SfxMoment(sid, c.timeline_start, "WARNING", 0.9, c.text.get("content", "")))
                    elif style == "HEADLINE" and emph == "PUNCH_TEXT":
                        out.append(SfxMoment(sid, c.timeline_start, "PUNCH", 0.75, c.text.get("content", "")))
                    elif style == "HEADLINE":
                        out.append(SfxMoment(sid, c.timeline_start, "HEADLINE", 0.6, c.text.get("content", "")))
                elif c.kind == KIND_GRAPHIC and ("evidence" in c.effects or "highlight" in c.effects):
                    out.append(SfxMoment(sid, c.timeline_start, "EVIDENCE", 0.65, "evidence"))
                elif c.kind == KIND_MEDIA and c.transition and c.transition.get("type") in ("DISSOLVE", "WIPE", "SLIDE", "FADE") and t.kind is not TrackKind.AUDIO:
                    out.append(SfxMoment(sid, c.timeline_start, "TRANSITION", 0.6 + (0.2 if order[sid].starts_section else 0.0), c.transition["type"].lower()))
        return out

    def _commit_audio(self, asm: PresentationAssembler, p: Project, ctx: EditingContext, scene_ids: set[str]) -> None:
        aset = p.audio_settings
        total = self._total_duration(p)
        placed_sfx: list[tuple[float, float]] = []
        if aset.sfx_enabled:
            events = SfxPlanner(aset, dynamic=ctx.settings.style == "dynamic").plan(self._sfx_moments(asm, ctx, scene_ids), self.sfx_library(), total)
        else:
            events = []
        made = asm.place_sfx(scene_ids, events, p.assets)
        for t in asm.state.timeline.tracks:
            for c in t.clips:
                if c.id in made:
                    placed_sfx.append((c.timeline_start, c.timeline_end))
        # AI-placed music follows the video structure again (user-placed/locked/edited music is left alone)
        self._replace_ai_music(asm, p, ctx)
        if aset.auto_ducking:
            speech = self._speech(p)
            plan = self._duck_plan(asm, p, ctx, speech, total, placed_sfx)
            kfs = AudioDuckingService(aset).keyframes(plan)
            asm.apply_ducking(plan, kfs, aset)
        else:
            asm.state.ducking_events[:] = [e for e in asm.state.ducking_events if e.created_by == "USER"]
        asm.finish_part("audio", "audio")
        for sid in scene_ids:
            plan_ = asm.plan_for(sid)
            plan_.ducking_events = [e.event_id for e in asm.state.ducking_events if e.scene_id == sid]
            plan_.music_state = {"level": aset.music_level, "important_level": aset.important_level, "auto_ducking": aset.auto_ducking}

    def _duck_plan(self, asm: PresentationAssembler, p: Project, ctx: EditingContext, speech, total: float, sfx: list[tuple[float, float]]):
        aset = p.audio_settings
        svc = AudioDuckingService(aset)
        important: list[Important] = []
        for sc in ctx.scenes:
            brief = p.editing_strategy.briefs.get(sc.scene.id)
            imp = brief.importance if brief else sc.scene.importance
            if imp >= aset.important_threshold:
                important.append(Important(sc.scene.start, sc.scene.end, f"Important narration (scene {sc.scene.label})", sc.scene.id))
        for t in asm.state.timeline.tracks:
            for c in t.clips:
                if c.kind == KIND_TEXT and c.text and c.text.get("style") in ("NUMBER_CARD", "DATE", "WARNING") and c.scene_id:
                    important.append(Important(max(0.0, c.timeline_start - 0.3), c.timeline_start + 1.4, f"Important figure “{c.text.get('content', '')}”", c.scene_id))
        intensity = [[0.0, 1.0]]
        for sc in ctx.scenes:
            if sc.starts_section:
                intensity += [[sc.scene.start, 1.25], [sc.scene.start + 6.0, 1.0]]
        return svc.plan(speech, total, important, sfx, sorted(intensity))

    def recommend_music(self) -> dict:
        """AI recommendation for placing music on top of the video structure. Music never changes the narration timing."""
        p = self._project()
        a = p.audio_settings
        voice_end = max([s.end for s in p.scenes] + [0.0])
        vo = p.assets.get(p.voice_over.asset_id) if p.voice_over.asset_id else None
        voice_end = max(voice_end, vo.duration if vo and vo.duration else 0.0)
        sections = [s.start for s in p.scenes[1:] if any(s.narration.lower().lstrip().startswith(x) for x in ("now let's", "moving on", "next,", "meanwhile", "in conclusion", "let's talk"))]
        return {"start": 0.0, "end": round(voice_end + a.music_fade_out, 3), "fade_in": a.music_fade_in, "fade_out": a.music_fade_out, "loop": a.loop_music,
                "section_intensity": [[round(t, 2), 1.25] for t in sections], "reason": "Music starts with the video, ducks under the narration, changes intensity at new sections and fades out after the voice."}

    def _replace_ai_music(self, asm: PresentationAssembler, p: Project, ctx: EditingContext) -> None:
        tl = asm.state.timeline
        if not any(t.id == "track_a2" for t in tl.tracks):
            return
        groups: dict[str, list] = {}
        for c in tl.get_track("track_a2").clips:
            groups.setdefault(c.metadata.get("assignment_id", c.id), []).append(c)
        rec = self.recommend_music()
        for aid, pieces in groups.items():
            if any(owned(c) for c in pieces) or any(c.created_by != Creator.AI.value for c in pieces):
                continue
            if any(d.slot in (f"music:{aid}", f"duck:{aid}") and (d.created_by is Creator.USER or d.locked) for d in asm.state.decisions.values()):
                continue  # the user edited or locked this placement's mix: keep it exactly
            asset = p.assets.get(pieces[0].asset_id)
            if asset is None:
                continue
            vol = float(pieces[0].audio.get("volume", 1.0))
            ducking = bool(pieces[0].audio.get("ducking", True))
            for c in pieces:
                tl.get_track("track_a2").clips.remove(c)
            for d in [d for d in asm.state.decisions.values() if d.slot in (f"music:{aid}", f"duck:{aid}") and d.created_by is Creator.AI and not d.locked]:
                del asm.state.decisions[d.decision_id]
            asm.place_music(aid, asset, rec["start"], rec["end"], vol, rec["fade_in"], rec["fade_out"], rec["loop"], Creator.AI, rec["reason"], 85.0, ducking)

    # ================================================================== editing API (every call is one undoable step)
    def _edit(self, description: str, mutate: Callable[[PresentationAssembler], object]):
        """Apply ``mutate`` to a copy of the presentation state, validate (new errors only), then install it as one undo step."""
        p = self._project()
        asm = PresentationAssembler(p)
        baseline = {(i.code, i.clip_id, i.message) for i in PresentationValidator(p, PresState.capture(p)).validate() if i.severity == "error"}
        result = mutate(asm)
        errors = self._new_errors(p, asm.state, baseline)
        if errors:
            raise PresentationError(errors[0].message, details="; ".join(str(e) for e in errors[:5]))
        self._commands.execute(ApplyPresentationCommand(p, asm.state, description))
        return result

    @staticmethod
    def _clip(asm: PresentationAssembler, clip_id: str, kinds: tuple[str, ...]):
        c = asm.state.timeline.get_clip(clip_id)
        if c is None or c.kind not in kinds:
            raise PresentationError("That object no longer exists or is not the right kind of object.")
        return c

    def _decision_of(self, asm: PresentationAssembler, clip, types):
        return next((d for d in asm.state.decisions.values() if d.target_id == clip.id and d.type in types), None)

    # ------------------------------------------------------------------ captions
    def update_caption(self, clip_id: str, *, text: str | None = None, style_id: str | None = None, style_overrides: dict | None = None, position: str | None = None,
                       position_xy: list[float] | None = None, highlight_mode: str | None = None, animation: dict | None = None, emphasis: list | None = None,
                       start: float | None = None, duration: float | None = None) -> None:
        """Edit a caption. It becomes USER-owned; regeneration keeps it."""
        p = self._project()
        styles = self.caption_styles()
        cs = p.caption_settings

        def go(asm: PresentationAssembler) -> None:
            c = self._clip(asm, clip_id, (KIND_CAPTION,))
            seg = CaptionSegment.from_dict(c.text)
            if text is not None:
                if not text.strip():
                    raise PresentationError("A caption cannot be empty.")
                seg.words = retime_words(seg.words, text)
                seg.text = " ".join(w.text for w in seg.words)
                eff = effective_style(style_for(styles, seg.style_id), cs, seg.style_overrides)
                seg.lines = break_lines(seg.words, layout_for(cs, eff, asm.canvas))
                seg.emphasis = [m for m in seg.emphasis if m.word_index < len(seg.words)]
            if style_id is not None:
                if style_id not in styles:
                    raise PresentationError(f"Unknown caption style “{style_id}”.")
                seg.style_id = style_id
            if style_overrides is not None:
                seg.style_overrides = {**seg.style_overrides, **style_overrides}
            if position is not None:
                if position not in POSITIONS:
                    raise PresentationError(f"Unknown caption position “{position}”.")
                seg.position = position
                if position != "custom":
                    seg.position_xy = []
            if position_xy is not None:
                seg.position, seg.position_xy = "custom", [float(position_xy[0]), float(position_xy[1])]
            if highlight_mode is not None:
                if highlight_mode not in {m.value for m in HighlightMode}:
                    raise PresentationError(f"Unknown highlight mode “{highlight_mode}”.")
                seg.highlight_mode = highlight_mode
            if animation is not None:
                seg.animation = animation
                c.animation = deepcopy(animation)
            if emphasis is not None:
                from app.presentation.models import EmphasisMark

                seg.emphasis = [EmphasisMark(int(i), str(cat), str(sty), "Set by the user.") for i, cat, sty in emphasis if 0 <= int(i) < len(seg.words)]
            if start is not None or duration is not None:
                ns = float(start if start is not None else c.timeline_start)
                nd = float(duration if duration is not None else c.duration)
                if nd <= 0.05:
                    raise PresentationError("A caption needs a positive duration.")
                if not asm.track_free(c.track_id, ns, ns + nd, {c.id}):
                    raise PresentationError("That timing would overlap another caption.")
                c.timeline_start, c.duration = round(ns, 4), round(nd, 4)
            seg.start, seg.end = c.timeline_start, c.timeline_end
            seg.reading_cps = round(len(seg.text) / max(c.duration, 0.25), 2)
            c.text = seg.to_dict()
            for d in own_clip(asm, c):
                if d.type is PresentationType.CAPTION:
                    d.parameters.update(style_id=seg.style_id, lines=len(seg.lines), reading_cps=seg.reading_cps, emphasis_words=seg.emphasis_words, highlight_mode=seg.highlight_mode)
                    d.start, d.duration, d.confidence, d.reason = c.timeline_start, c.duration, 100.0, "Edited by the user."

        self._edit("Edit caption", go)

    def apply_caption_style(self, style_id: str, scene_ids: list[str] | None = None) -> int:
        """Switch AI-owned captions to ``style_id`` without regenerating them (user-edited captions keep their own look)."""
        if style_id not in self.caption_styles():
            raise PresentationError(f"Unknown caption style “{style_id}”.")
        n = [0]

        def go(asm: PresentationAssembler) -> None:
            for t in asm.state.timeline.tracks:
                for c in t.clips:
                    if c.kind == KIND_CAPTION and (scene_ids is None or c.scene_id in scene_ids) and not owned(c):
                        c.text["style_id"] = style_id
                        n[0] += 1

        self._edit("Apply caption style", go)
        return n[0]

    # ------------------------------------------------------------------ graphics
    def add_graphic(self, kind: str, content: str, start: float, duration: float, *, subtitle: str = "", scene_id: str = "", position: tuple[float, float] | None = None,
                    size: int | None = None, animation: dict | None = None) -> str:
        """A user-made text graphic (headline, lower third, number, date, warning, label, definition, comparison, location, entity). Returns its clip id."""
        from app.timeline.clip import Clip
        from app.timeline.timeline import new_clip_id

        kinds = {"HEADLINE": "HEADLINE", "LOWER_THIRD": "LOWER_THIRD", "NUMBER": "NUMBER_CARD", "DATE": "DATE", "WARNING": "WARNING", "LABEL": "LABEL",
                 "DEFINITION": "DEFINITION", "COMPARISON": "COMPARISON", "LOCATION": "LOCATION", "ENTITY": "ENTITY_NAME"}
        k = kind.upper()
        if k not in kinds:
            raise PresentationError(f"Unknown graphic type “{kind}”.")
        if not content.strip():
            raise PresentationError("The text cannot be empty.")
        if duration <= 0.05 or start < 0:
            raise PresentationError("Give the graphic a start of 0 or more and a positive duration.")
        p = self._project()
        level = anim.intensity_level(p.editing_settings.motion_intensity)
        variant = {"NUMBER_CARD": "NUMBER", "ENTITY_NAME": "LOWER_THIRD"}.get(kinds[k], kinds[k])
        positions = {"NUMBER_CARD": (0.5, 0.42), "LOWER_THIRD": (0.08, 0.80), "ENTITY_NAME": (0.08, 0.80), "HEADLINE": (0.5, 0.14), "WARNING": (0.5, 0.20), "DATE": (0.5, 0.40),
                     "LOCATION": (0.08, 0.12)}

        def go(asm: PresentationAssembler) -> str:
            asm.ensure_tracks()
            if not asm.track_free("track_v5", start, start + duration):
                raise PresentationError("There is already a text graphic at that time. Move or shorten it first.")
            sid = scene_id or next((s.id for s in p.scenes if s.start <= start < s.end), "")
            g = TextGraphic(f"user_{len(asm.state.timeline.all_clips())}", content.strip(), start, duration, tuple(position or positions.get(kinds[k], (0.5, 0.8))), kinds[k], "", "fade", 0.5, sid,
                            "user", size=size or (88 if kinds[k] == "NUMBER_CARD" else 56), alignment="left" if variant == "LOWER_THIRD" else "center",
                            background="box" if variant in ("LOWER_THIRD", "WARNING") else "none")
            clip = Clip(new_clip_id(), "track_v5", "", start, duration, kind=KIND_TEXT, scene_id=sid, slot=f"user:{variant.lower()}:{start:.2f}", created_by=Creator.USER.value,
                        text={**g.to_dict(), "variant": variant, "title": content.strip(), "subtitle": subtitle, "counter": None, "derived": False},
                        animation=animation or anim.default_animation(variant, level, p.caption_settings.reduced_motion), metadata={"phase": 5})
            asm.state.timeline.get_track("track_v5").clips.append(clip)
            asm.state.timeline.get_track("track_v5").sort()
            ptype = {"NUMBER": PresentationType.NUMBER_GRAPHIC, "DATE": PresentationType.DATE_GRAPHIC, "LOWER_THIRD": PresentationType.LOWER_THIRD,
                     "HEADLINE": PresentationType.HEADLINE}.get(variant, PresentationType.TEXT_GRAPHIC)
            d = asm.decision(sid, ptype, clip.slot, clip.id, start, duration, {"content": content.strip(), "variant": variant}, "Added by the user.", 100.0, Creator.USER)
            clip.ai_decision_id = d.decision_id
            return clip.id

        return self._edit("Add graphic", go)

    def update_graphic(self, clip_id: str, **fields) -> None:
        """Edit a text graphic: content, subtitle, style, position (x, y), size, alignment, opacity, background, start, duration, counter."""
        allowed = {"content", "subtitle", "style", "position", "size", "alignment", "opacity", "background", "font", "start", "duration", "title", "emphasis"}
        unknown = set(fields) - allowed
        if unknown:
            raise PresentationError(f"Unsupported graphic properties: {sorted(unknown)}")

        def go(asm: PresentationAssembler) -> None:
            c = self._clip(asm, clip_id, (KIND_TEXT,))
            t = dict(c.text or {})
            if "content" in fields:
                if not str(fields["content"]).strip():
                    raise PresentationError("The text cannot be empty.")
                t["content"] = str(fields["content"]).strip()
                t["title"] = t["content"]
                if t.get("counter"):
                    from app.presentation.graphics import counter_spec

                    t["counter"] = counter_spec(t["content"])
            for k in ("subtitle", "style", "size", "alignment", "opacity", "background", "font", "title", "emphasis"):
                if k in fields:
                    t[k] = fields[k]
            if "position" in fields:
                x, y = fields["position"]
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise PresentationError("The position must stay inside the frame (0–1).")
                t["position"] = [float(x), float(y)]
            if int(t.get("size", 1)) <= 0:
                raise PresentationError("The size must be positive.")
            if "start" in fields or "duration" in fields:
                ns = float(fields.get("start", c.timeline_start))
                nd = float(fields.get("duration", c.duration))
                if nd <= 0.05 or ns < 0:
                    raise PresentationError("The graphic needs a positive duration.")
                if not asm.track_free(c.track_id, ns, ns + nd, {c.id}):
                    raise PresentationError("That timing would overlap another graphic.")
                c.timeline_start, c.duration = round(ns, 4), round(nd, 4)
            t["start"], t["duration"] = c.timeline_start, c.duration
            c.text = t
            for d in own_clip(asm, c):
                d.parameters.update({k: t.get(k) for k in ("content", "style", "size", "position") if k in t})
                d.start, d.duration, d.confidence, d.reason = c.timeline_start, c.duration, 100.0, "Edited by the user."

        self._edit("Edit graphic", go)

    def set_graphic_animation(self, clip_id: str, side: str, preset: str, *, duration: float | None = None, easing: str | None = None, delay: float | None = None,
                              start_value: float | None = None, end_value: float | None = None) -> None:
        """Change the in/out animation of a caption or graphic (Fade, Slide, Scale, Pop, Reveal, Type-on, Counter, Highlight...)."""
        if side not in ("in", "out"):
            raise PresentationError("The animation side must be “in” or “out”.")
        if preset != "none" and preset not in anim.PRESET_NAMES:
            raise PresentationError(f"Unknown animation preset “{preset}”.")
        if easing is not None and easing not in anim.EASINGS:
            raise PresentationError(f"Unknown easing “{easing}”.")

        def go(asm: PresentationAssembler) -> None:
            c = self._clip(asm, clip_id, (KIND_TEXT, KIND_GRAPHIC, KIND_CAPTION))
            a = anim.normalize(c.animation)
            if preset == "none":
                a.pop(side, None)
            else:
                s = anim.spec(preset)
                for k, v in (("duration", duration), ("easing", easing), ("delay", delay), ("start_value", start_value), ("end_value", end_value)):
                    if v is not None:
                        s[k] = v
                a[side] = s
            bad = anim.problems(a, c.duration)
            if bad:
                raise PresentationError(bad[0])
            c.animation = a
            if c.kind == KIND_CAPTION and c.text:
                c.text["animation"] = deepcopy(a)
            for d in own_clip(asm, c):
                d.parameters["animation"] = deepcopy(a)
                d.confidence, d.reason = 100.0, "Edited by the user."

        self._edit("Change animation", go)

    def move_graphic(self, clip_id: str, x: float, y: float) -> None:
        self.update_graphic(clip_id, position=(x, y))

    def set_evidence_tool(self, clip_id: str, tool: str, *, region: list[float] | None = None, dim: bool | None = None) -> None:
        """Evidence graphics for documents/charts/screenshots: focus box, highlight, underline, pointer, magnify, crop, dim. The source is never altered."""
        if tool not in EVIDENCE_TOOLS:
            raise PresentationError(f"Unknown evidence tool “{tool}”.")

        def go(asm: PresentationAssembler) -> None:
            c = self._clip(asm, clip_id, (KIND_GRAPHIC,))
            ev = dict(c.effects.get("evidence") or {})
            hl = dict(c.effects.get("highlight") or {})
            ev["tool"] = tool
            if region is not None:
                if (len(region) != 4 or not all(0.0 <= v <= 1.0 for v in region) or region[2] <= 0 or region[3] <= 0
                        or region[0] + region[2] > 1.0001 or region[1] + region[3] > 1.0001):
                    raise PresentationError("The region must be four values (x, y, width, height) inside the frame.")
                ev["region"], hl["region"] = list(region), list(region)
            if dim is not None:
                ev["dim"], hl["darken_surround"] = dim, dim
            c.effects["evidence"], c.effects["highlight"] = ev, hl
            for d in own_clip(asm, c):
                d.parameters.update({"tool": tool, **({"region": ev.get("region")} if region is not None else {})})
                d.confidence, d.reason = 100.0, "Edited by the user."

        self._edit("Change evidence tool", go)

    # ------------------------------------------------------------------ music
    def add_music(self, asset_id: str, *, start: float | None = None, end: float | None = None, volume: float | None = None, loop: bool | None = None,
                  fade_in: float | None = None, fade_out: float | None = None, ai: bool = True) -> str:
        """Put a music asset on A2. Without explicit timing the AI placement recommendation is used; ducking is generated automatically."""
        p = self._project()
        a = p.assets.get(asset_id)
        if a is None or a.type is not AssetType.AUDIO or not a.duration:
            raise PresentationError("Choose an audio file from the project.")
        if not p.audio_settings.music_enabled:
            raise PresentationError("Music is switched off in the audio settings.")
        rec = self.recommend_music()
        s0 = rec["start"] if start is None else start
        e0 = rec["end"] if end is None else end
        if e0 - s0 <= 0.1 or s0 < 0:
            raise PresentationError("The music needs a start of 0 or more and a positive length.")
        fi = rec["fade_in"] if fade_in is None else fade_in
        fo = rec["fade_out"] if fade_out is None else fade_out
        lp = rec["loop"] if loop is None else loop
        by = Creator.AI if (ai and start is None and end is None) else Creator.USER
        aid = f"music_{len(p.timeline.get_track('track_a2').clips) + 1:03d}_{asset_id}"
        reason = rec["reason"] if by is Creator.AI else "Placed by the user."

        def go(asm: PresentationAssembler) -> str:
            asm.ensure_tracks()
            try:
                asm.place_music(aid, a, s0, e0, 1.0 if volume is None else volume, fi, fo, lp, by, reason, 85.0 if by is Creator.AI else 100.0)
            except ValueError as exc:
                raise PresentationError(str(exc)) from exc
            if p.audio_settings.auto_ducking:
                self._duck_in(asm, p)
            return aid

        return self._edit("Add music", go)

    def _duck_in(self, asm: PresentationAssembler, p: Project) -> None:
        ctx = build_context(p)
        total = self._total_duration(p)
        sfx = [(c.timeline_start, c.timeline_end) for t in asm.state.timeline.tracks if t.kind is TrackKind.AUDIO for c in t.clips if c.audio.get("role") == "SFX"]
        plan = self._duck_plan(asm, p, ctx, self._speech(p), total, sfx)
        asm.apply_ducking(plan, AudioDuckingService(p.audio_settings).keyframes(plan), p.audio_settings)

    def _music_pieces(self, asm: PresentationAssembler, assignment_id: str) -> list:
        pieces = [c for c in asm.state.timeline.get_track("track_a2").clips if c.metadata.get("assignment_id") == assignment_id]
        if not pieces:
            raise PresentationError("That music no longer exists.")
        return sorted(pieces, key=lambda c: c.timeline_start)

    def set_music(self, assignment_id: str, **changes) -> None:
        """Volume, fades, loop, start, end of a music placement. Becomes USER-owned."""
        allowed = {"volume", "fade_in", "fade_out", "loop", "start", "end", "ducking"}
        unknown = set(changes) - allowed
        if unknown:
            raise PresentationError(f"Unsupported music properties: {sorted(unknown)}")
        p = self._project()

        def go(asm: PresentationAssembler) -> None:
            pieces = self._music_pieces(asm, assignment_id)
            first, last = pieces[0], pieces[-1]
            asset = p.assets.get(first.asset_id)
            vol = float(changes.get("volume", first.audio.get("volume", 1.0)))
            if not (0.0 <= vol <= 4.0):
                raise PresentationError("The volume must be between 0 and 400%.")
            if {"start", "end", "loop"} & set(changes):  # re-place the pieces
                s0, e0 = float(changes.get("start", first.timeline_start)), float(changes.get("end", last.timeline_end))
                if e0 - s0 <= 0.1 or s0 < 0:
                    raise PresentationError("The music needs a positive length.")
                keep_kf = [k for c in pieces for k in c.keyframes]
                for c in pieces:
                    asm.state.timeline.get_track("track_a2").clips.remove(c)
                old_d = [d for d in asm.state.decisions.values() if d.slot == f"music:{assignment_id}"]
                for d in old_d:
                    del asm.state.decisions[d.decision_id]
                try:
                    new = asm.place_music(assignment_id, asset, s0, e0, vol, float(changes.get("fade_in", first.audio.get("fade_in", 0))),
                                          float(changes.get("fade_out", last.audio.get("fade_out", 0))), bool(changes.get("loop", first.audio.get("loop", True))), Creator.USER,
                                          "Edited by the user.", 100.0, bool(changes.get("ducking", first.audio.get("ducking", True))))
                except ValueError as exc:
                    raise PresentationError(str(exc)) from exc
                if p.audio_settings.auto_ducking and new:
                    self._duck_in(asm, p)
                _ = keep_kf
                return
            for c in pieces:
                c.audio["volume"] = vol
                if "ducking" in changes:
                    c.audio["ducking"] = bool(changes["ducking"])
            if "fade_in" in changes:
                first.audio["fade_in"] = max(0.0, min(float(changes["fade_in"]), first.duration))
            if "fade_out" in changes:
                last.audio["fade_out"] = max(0.0, min(float(changes["fade_out"]), last.duration))
            for c in pieces:
                for d in own_clip(asm, c):
                    d.parameters.update({k: v for k, v in changes.items() if k in ("volume", "fade_in", "fade_out", "loop", "ducking")})
                    d.confidence, d.reason = 100.0, "Edited by the user."

        self._edit("Edit music", go)

    def replace_music(self, assignment_id: str, asset_id: str) -> None:
        p = self._project()
        a = p.assets.get(asset_id)
        if a is None or a.type is not AssetType.AUDIO or not a.duration:
            raise PresentationError("Choose an audio file from the project.")

        def go(asm: PresentationAssembler) -> None:
            pieces = self._music_pieces(asm, assignment_id)
            first, last = pieces[0], pieces[-1]
            vol, fi, fo = float(first.audio.get("volume", 1.0)), float(first.audio.get("fade_in", 0)), float(last.audio.get("fade_out", 0))
            loop, duck = bool(first.audio.get("loop", True)), bool(first.audio.get("ducking", True))
            s0, e0 = first.timeline_start, last.timeline_end
            for c in pieces:
                asm.state.timeline.get_track("track_a2").clips.remove(c)
            for d in [d for d in asm.state.decisions.values() if d.slot == f"music:{assignment_id}"]:
                del asm.state.decisions[d.decision_id]
            asm.place_music(assignment_id, a, s0, e0, vol, fi, fo, loop, Creator.USER, f"Replaced by the user with {a.name}.", 100.0, duck)
            if p.audio_settings.auto_ducking:
                self._duck_in(asm, p)

        self._edit("Replace music", go)

    def delete_music(self, assignment_id: str) -> None:
        def go(asm: PresentationAssembler) -> None:
            pieces = self._music_pieces(asm, assignment_id)
            for c in pieces:
                asm.state.timeline.get_track("track_a2").clips.remove(c)
            for d in [d for d in asm.state.decisions.values() if d.slot in (f"music:{assignment_id}", f"duck:{assignment_id}")]:
                del asm.state.decisions[d.decision_id]

        self._edit("Delete music", go)

    # ------------------------------------------------------------------ ducking keyframes (editable)
    def ducking_keyframes(self, assignment_id: str) -> list[tuple[float, float]]:
        p = self._project()
        pieces = sorted((c for c in p.timeline.get_track("track_a2").clips if c.metadata.get("assignment_id") == assignment_id), key=lambda c: c.timeline_start)
        return [(round(c.timeline_start + k.time, 3), k.value) for c in pieces for k in sorted(c.keyframes, key=lambda x: x.time) if k.property == "volume"]

    def set_ducking_keyframes(self, assignment_id: str, keyframes: list[tuple[float, float]]) -> None:
        """Replace the volume keyframes of a music placement (timeline seconds, linear level). The ducking decision becomes USER-owned."""
        pts = sorted((float(t), float(v)) for t, v in keyframes)
        if any(not (0.0 <= v <= 4.0) or t < 0 for t, v in pts) or any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
            raise PresentationError("Keyframes need increasing times and volumes between 0 and 400%.")

        def go(asm: PresentationAssembler) -> None:
            pieces = self._music_pieces(asm, assignment_id)
            slot = f"duck:{assignment_id}"
            d = next((d for d in asm.state.decisions.values() if d.slot == slot), None)
            if d is None:
                d = asm.decision("", PresentationType.DUCKING, slot, pieces[0].id, pts[0][0] if pts else 0.0, 0.0, {"assignment_id": assignment_id}, "Edited by the user.", 100.0)
            d = own_decision(asm, d)
            d.reason, d.confidence = "Volume keyframes edited by the user.", 100.0
            d.parameters.update(keyframe_count=len(pts), edited=True)
            src = [Keyframe("volume", t, v) for t, v in pts]
            from app.timeline.keyframes import value_at

            for c in pieces:
                c.keyframes = [k for k in c.keyframes if k.property != "volume"]
                inside = [(t, v) for t, v in pts if c.timeline_start - 1e-6 <= t <= c.timeline_end + 1e-6]
                edge = [(c.timeline_start, value_at(src, "volume", c.timeline_start)), (c.timeline_end, value_at(src, "volume", c.timeline_end))] if pts else []
                allp = sorted({round(t, 3): v for t, v in edge + inside}.items())
                c.keyframes += [Keyframe("volume", round(t - c.timeline_start, 3), round(v, 4), "linear", d.decision_id) for t, v in allp]

        self._edit("Edit ducking keyframes", go)

    # ------------------------------------------------------------------ SFX
    def add_sfx(self, asset_id: str, time: float, *, volume: float | None = None, scene_id: str = "", duration: float | None = None, category: str | None = None) -> str:
        """A sound effect placed by the user (layered on an extra SFX track when it overlaps another)."""
        p = self._project()
        a = p.assets.get(asset_id)
        if a is None or a.type is not AssetType.AUDIO or not a.duration:
            raise PresentationError("Choose an audio file from the project.")
        if not p.audio_settings.sfx_enabled:
            raise PresentationError("Sound effects are switched off in the audio settings.")
        if time < 0:
            raise PresentationError("A sound effect cannot start before 0:00.")
        dur = min(float(duration or a.duration), a.duration)
        cat = (category or a.extra.get("category") or SfxCategory.IMPACT.value).upper()
        if cat not in {c.value for c in SfxCategory}:
            raise PresentationError(f"Unknown SFX category “{category}”.")
        sid = scene_id or next((s.id for s in p.scenes if s.start <= time < s.end), "")

        def go(asm: PresentationAssembler) -> str:
            from app.timeline.clip import Clip
            from app.timeline.timeline import new_clip_id

            asm.ensure_tracks()
            tid = asm._free_sfx_track(time, time + dur)
            slot = f"sfx:user:{time:.2f}"
            clip = Clip(new_clip_id(), tid, a.id, round(time, 4), round(dur, 4), 0.0, round(dur, 4), kind=KIND_MEDIA, scene_id=sid, slot=slot, created_by=Creator.USER.value,
                        audio={"role": "SFX", "volume": p.audio_settings.sfx_level if volume is None else volume, "fade_in": 0.0, "fade_out": min(0.08, dur / 3), "category": cat},
                        metadata={"phase": 5, "sfx_id": f"sfx_user_{time:.2f}", "trigger": "USER"})
            d = asm.decision(sid, PresentationType.SFX, slot, clip.id, time, dur, {"asset_id": a.id, "category": cat, "timestamp": time, "trigger": "USER"},
                             "Placed by the user.", 100.0, Creator.USER)
            clip.ai_decision_id = d.decision_id
            asm.state.timeline.get_track(tid).clips.append(clip)
            asm.state.timeline.get_track(tid).sort()
            return clip.id

        return self._edit("Add sound effect", go)

    def set_clip_audio(self, clip_id: str, **changes) -> None:
        """Volume, fades (and category) of any audio clip: voice-over, music or SFX. Becomes USER-owned."""
        allowed = {"volume", "fade_in", "fade_out", "category"}
        unknown = set(changes) - allowed
        if unknown:
            raise PresentationError(f"Unsupported audio properties: {sorted(unknown)}")

        def go(asm: PresentationAssembler) -> None:
            c = self._clip(asm, clip_id, (KIND_MEDIA,))
            if not any(t.id == c.track_id and t.kind is TrackKind.AUDIO for t in asm.state.timeline.tracks):
                raise PresentationError("That clip is not an audio clip.")
            if "volume" in changes:
                v = float(changes["volume"])
                if not (0.0 <= v <= 4.0):
                    raise PresentationError("The volume must be between 0 and 400%.")
                c.audio["volume"] = v
            for k in ("fade_in", "fade_out"):
                if k in changes:
                    v = float(changes[k])
                    if v < 0:
                        raise PresentationError("A fade cannot be negative.")
                    c.audio[k] = v
            if float(c.audio.get("fade_in", 0)) + float(c.audio.get("fade_out", 0)) > c.duration + 0.02:
                raise PresentationError("The fades are longer than the clip.")
            if "category" in changes:
                cat = str(changes["category"]).upper()
                if cat not in {x.value for x in SfxCategory}:
                    raise PresentationError(f"Unknown SFX category “{changes['category']}”.")
                c.audio["category"] = cat
            for d in own_clip(asm, c):
                d.parameters.update({k: v for k, v in changes.items()})
                d.confidence, d.reason = 100.0, "Edited by the user."

        self._edit("Edit audio clip", go)

    def replace_sfx(self, clip_id: str, asset_id: str) -> None:
        p = self._project()
        a = p.assets.get(asset_id)
        if a is None or a.type is not AssetType.AUDIO or not a.duration:
            raise PresentationError("Choose an audio file from the project.")

        def go(asm: PresentationAssembler) -> None:
            c = self._clip(asm, clip_id, (KIND_MEDIA,))
            dur = min(c.duration, a.duration)
            c.asset_id, c.source_in, c.source_out, c.duration = a.id, 0.0, dur, dur
            for d in own_clip(asm, c):
                d.parameters["asset_id"] = a.id
                d.reason, d.confidence = f"Replaced by the user with {a.name}.", 100.0

        self._edit("Replace sound effect", go)

    # ------------------------------------------------------------------ locks
    def set_lock(self, scene_id: str, aspect: str, locked: bool = True) -> None:
        aspect = aspect.upper()
        if aspect not in LOCK_ASPECTS:
            raise PresentationError(f"Unknown lock “{aspect}”.")

        def go(asm: PresentationAssembler) -> None:
            st = asm.state
            if aspect == "SCENE":
                ids = st.generation.locked_scenes
                if locked and scene_id not in ids:
                    ids.append(scene_id)
                if not locked and scene_id in ids:
                    ids.remove(scene_id)
                return
            for t in st.timeline.tracks:
                for c in t.clips:
                    if c.scene_id != scene_id:
                        continue
                    kind_ok = ((aspect == "CAPTION" and c.kind == KIND_CAPTION) or (aspect == "GRAPHIC" and c.kind in (KIND_TEXT, KIND_GRAPHIC))
                               or (aspect == "MUSIC" and c.audio.get("role") == "MUSIC") or (aspect == "SFX" and c.audio.get("role") == "SFX"))
                    if kind_ok:
                        c.locked = locked
            if aspect == "MIX":  # lock the ducking/mix decisions (they are global to the music, so every music placement is locked)
                for d in st.decisions.values():
                    if d.type is PresentationType.DUCKING:
                        d.locked = locked
                for c in st.timeline.get_track("track_a2").clips if any(t.id == "track_a2" for t in st.timeline.tracks) else []:
                    c.locked = locked

        self._edit(f"{'Lock' if locked else 'Unlock'} {aspect.lower()}", go)

    def lock_clip(self, clip_id: str, locked: bool = True) -> None:
        def go(asm: PresentationAssembler) -> None:
            c = asm.state.timeline.get_clip(clip_id)
            if c is None:
                raise PresentationError("That clip no longer exists.")
            c.locked = locked
            for d in asm.state.decisions.values():
                if d.target_id == clip_id:
                    d.locked = locked

        self._edit(f"{'Lock' if locked else 'Unlock'} clip", go)

    # ------------------------------------------------------------------ previews (background)
    def mix_plan(self, mode: PreviewMode = PreviewMode.FULL):
        return self.audio.mix.build_plan(self._project(), mode)

    def preview_mix(self, mode: PreviewMode = PreviewMode.FULL, start: float = 0.0, end: float | None = None, on_ready: Callable[[Path | None], None] | None = None) -> Job:
        """Render a *preview* of Voice / Music / SFX / Voice+Music / Voice+SFX / Full mix to the cache in a background job."""
        p = self._project()
        cache = p.root / "cache"
        proc = p.audio_processing
        h = current_audio_hash(p) or ""
        vo = p.assets.get(p.voice_over.asset_id) if p.voice_over.asset_id else None

        def work(ctx) -> Path | None:
            ctx.report(10, "Preparing the voice")
            voice_path = None
            if vo is not None and proc.enabled and p.audio_settings.voice_enhancement:
                voice_path = self.audio.processing.render_preview(p.asset_path(vo), cache, h, proc, vo.duration)
            plan = self.audio.mix.build_plan(p, mode, voice_path)
            return self.audio.mix.render(plan, cache, start, end, lambda pct, msg: ctx.report(pct, msg))

        def done(job: Job) -> None:
            if on_ready:
                on_ready(job.result)

        def failed(job: Job) -> None:
            self._bus.publish(Topics.ERROR, message=job.error or "The audio preview could not be created.", title="Audio preview")
            if on_ready:
                on_ready(None)

        return self._jobs.submit("audio_preview", work, title=f"Audio preview: {mode.value.replace('_', ' ').title()}", on_complete=done, on_error=failed)

    # ------------------------------------------------------------------ validation / inspection
    def validate(self) -> list[ValidationIssue]:
        p = self._project()
        return PresentationValidator(p, PresState.capture(p)).validate()

    def describe(self, clip_id: str) -> dict | None:
        """Everything the timeline inspector shows for a caption / graphic / music / SFX clip (None for other clips)."""
        p = self._projects.current
        clip = p.timeline.get_clip(clip_id) if p else None
        if clip is None:
            return None
        decs = [d for d in p.presentation_decisions.values() if d.target_id == clip_id and d.type in (PresentationType.CAPTION, PresentationType.NUMBER_GRAPHIC,
                PresentationType.DATE_GRAPHIC, PresentationType.LOWER_THIRD, PresentationType.HEADLINE, PresentationType.TEXT_GRAPHIC, PresentationType.EVIDENCE_GRAPHIC,
                PresentationType.MOTION_GRAPHIC, PresentationType.MUSIC, PresentationType.SFX)]
        d = decs[0] if decs else None
        base = {"decision": d, "created_by": clip.created_by, "locked": clip.locked, "scene_id": clip.scene_id, "reason": d.reason if d else "", "confidence": d.confidence if d else None}
        if clip.kind == KIND_CAPTION and clip.text:
            t = clip.text
            st = effective_style(style_for(self.caption_styles(), t.get("style_id", p.caption_settings.style_id)), p.caption_settings, t.get("style_overrides"))
            return {**base, "kind": "CAPTION", "font": st.font, "size_pct": round(st.size_rel * 100, 1), "weight": st.weight, "text": t.get("text", ""), "start": clip.timeline_start,
                    "end": clip.timeline_end, "style_id": t.get("style_id"),
                    "position": t.get("position"), "animation": clip.animation, "highlight_mode": t.get("highlight_mode"), "emphasis": t.get("emphasis", []),
                    "lines": t.get("lines", [])}
        if clip.kind in (KIND_TEXT, KIND_GRAPHIC):
            return {**base, "kind": "GRAPHIC" if clip.kind == KIND_TEXT else "EVIDENCE", "text": clip.text, "effects": clip.effects, "animation": clip.animation,
                    "start": clip.timeline_start, "end": clip.timeline_end}
        role = clip.audio.get("role")
        if role in ("MUSIC", "SFX"):
            a = p.assets.get(clip.asset_id)
            return {**base, "kind": role, "asset": a.name if a else clip.asset_id, "start": clip.timeline_start, "end": clip.timeline_end, "volume": clip.audio.get("volume", 1.0),
                    "fade_in": clip.audio.get("fade_in", 0.0), "fade_out": clip.audio.get("fade_out", 0.0), "ducking": clip.audio.get("ducking"),
                    "keyframes": [(k.time, k.value) for k in clip.keyframes if k.property == "volume"], "category": clip.audio.get("category"),
                    "assignment_id": clip.metadata.get("assignment_id")}
        return None

    def override_command(self, clip_id: str, action: str) -> Command | None:
        p = self._projects.current
        clip = p.timeline.get_clip(clip_id) if p else None
        if p is None or clip is None or not (clip.metadata.get("phase") == 5 or clip.metadata.get("phase5")
                                              or any(d.target_id == clip_id for d in p.presentation_decisions.values())):
            return None
        return RecordPresentationDeleteCommand(p, clip_id) if action == "delete" else MarkPresentationEditCommand(p, clip_id)

    def scene_rows(self) -> list[dict]:
        p = self._project()
        g = p.presentation_generation
        rows = []
        for s in p.scenes:
            def n(pred):
                return sum(1 for t in p.timeline.tracks for c in t.clips if c.scene_id == s.id and pred(c))
            rows.append({"scene_id": s.id, "label": s.label, "locked": s.id in g.locked_scenes,
                         "captions": n(lambda c: c.kind == KIND_CAPTION), "graphics": n(lambda c: c.kind in (KIND_TEXT, KIND_GRAPHIC)),
                         "sfx": n(lambda c: c.audio.get("role") == "SFX"),
                         "caption_status": g.scene_status.get("CAPTIONS", {}).get(s.id, "PENDING"), "graphics_status": g.scene_status.get("GRAPHICS", {}).get(s.id, "PENDING"),
                         "audio_status": g.scene_status.get("AUDIO", {}).get(s.id, "PENDING")})
        return rows


_ = (re, DecisionType, AudioAnalysisService, AudioError)


class MusicService:
    """Music on A2: import, preview, add, trim/loop/fade/volume (``set``), replace, delete, ducking. A thin facade over ``PresentationService``."""

    def __init__(self, svc: PresentationService) -> None:
        self._s = svc

    def import_music(self, path: Path) -> list[Job]:
        return self._s.import_audio(path, "music")

    def library(self) -> list:
        return self._s.library("music")

    def recommend(self) -> dict:
        return self._s.recommend_music()

    def add(self, asset_id: str, **kw) -> str:
        return self._s.add_music(asset_id, **kw)

    def set(self, assignment_id: str, **changes) -> None:
        self._s.set_music(assignment_id, **changes)

    def replace(self, assignment_id: str, asset_id: str) -> None:
        self._s.replace_music(assignment_id, asset_id)

    def delete(self, assignment_id: str) -> None:
        self._s.delete_music(assignment_id)

    def preview(self, **kw) -> Job:
        return self._s.preview_mix(PreviewMode.MUSIC, **kw)


class SFXService:
    """Sound effects on A3 (layered on extra tracks): import, place, trim/move via the timeline, volume/fade, replace, delete. A thin facade."""

    def __init__(self, svc: PresentationService) -> None:
        self._s = svc

    def import_sfx(self, path: Path, category: str | None = None) -> list[Job]:
        return self._s.import_audio(path, "sfx", category)

    def library(self) -> dict[str, list]:
        return self._s.sfx_library()

    def add(self, asset_id: str, time: float, **kw) -> str:
        return self._s.add_sfx(asset_id, time, **kw)

    def set(self, clip_id: str, **changes) -> None:
        self._s.set_clip_audio(clip_id, **changes)

    def replace(self, clip_id: str, asset_id: str) -> None:
        self._s.replace_sfx(clip_id, asset_id)

    def preview(self, **kw) -> Job:
        return self._s.preview_mix(PreviewMode.SFX, **kw)
