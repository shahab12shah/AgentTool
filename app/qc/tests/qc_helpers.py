"""Shared builders for QC tests: a synthetic project (scenes, narration words, assets, timeline) without ffmpeg or the AI pipeline, so checker tests are fast and exact.

    p = new_project(tmp_path, seconds=30)
    a = add_asset(p, "city.mp4", "video", duration=12)
    s1 = add_scene(p, 0, 10, "Silver prices rose sharply last week.", importance=0.8)
    narrate(p)                                 # transcript words for every scene's narration (deterministic timing)
    add_clip(p, "track_v1", a, 0, 10, scene=s1)
    ctx = qc_ctx(p)
    out = TimelineChecker().run(ctx, lambda f, m: None)
"""

from __future__ import annotations

from pathlib import Path

from app.analysis.models import Origin, Scene, SceneStatus
from app.media.asset import Asset, AssetType, SourceType
from app.project.project import Project
from app.project.project_schema import ProjectSettings
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import QCContext
from app.qc.settings import QCSettings
from app.storage.paths import ProjectPaths
from app.timeline.clip import Clip
from app.timeline.timeline import new_clip_id
from app.transcription.models import AudioInfo, ProviderInfo, Sentence, Transcript, TranscriptionState, Word


def new_project(root: Path, *, seconds: float = 30.0, fps: int = 30, size: tuple[int, int] = (1920, 1080), voice: bool = True) -> Project:
    """A project with the default 9 tracks, a voice-over asset (a real tiny WAV so file checks pass) and an empty transcript."""
    p = Project.new("QC test", ProjectSettings(size[0], size[1], fps, "16:9"))
    p.root = Path(root)
    ProjectPaths(p.root).create_structure()
    if voice:
        a = add_asset(p, "voice.wav", "audio", duration=seconds, w=None, h=None, write=True)
        p.voice_over.asset_id, p.voice_over.filename, p.voice_over.duration = a.id, a.name, seconds
    return p


def add_asset(p: Project, name: str, kind: str = "video", *, duration: float | None = 10.0, w: int | None = 1920, h: int | None = 1080, write: bool = True, content_hash: str | None = None,
              has_audio: bool = False, sample_rate: int | None = None) -> Asset:
    """Register an asset. ``write`` creates a placeholder file (QC's existence checks need a file; decode checks need real media, which tests create with ffmpeg helpers)."""
    atype = AssetType(kind)
    aid = p.assets.new_id()
    rel = f"media/{'audio' if kind == 'audio' else 'video' if kind == 'video' else 'images'}/{aid}_{name}"
    path = p.root / rel
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 64)
    asset = Asset(aid, atype, SourceType.USER_MEDIA, rel, name, duration if kind != "image" else None, w, h, 30.0 if kind == "video" else None, None, has_audio, None,
                  sample_rate or (48000 if kind == "audio" else None), 2 if kind == "audio" else None, 64 if write else 0, content_hash or f"hash_{aid}")
    p.assets.add(asset)
    return asset


def add_scene(p: Project, start: float, end: float, text: str = "", *, importance: float = 0.5, topic: str = "", label: str | None = None) -> Scene:
    sid = f"scene_{len(p.scenes) + 1:03d}"
    s = Scene(sid, label or str(len(p.scenes) + 1), start, end, text, [], topic or text[:30], "", importance, 0.9, SceneStatus.READY, Origin.AI)
    p.scenes.append(s)
    p.scenes.sort(key=lambda x: x.start)
    return s


def narrate(p: Project, wps: float = 2.6) -> Transcript:
    """Spread each scene's ``narration`` text over the scene's time span as words (one sentence per scene, evenly timed) and install it as the project's transcript."""
    words, sents = [], []
    n = 0
    for s in p.scenes:
        toks = s.narration.split()
        if not toks:
            continue
        span = max(0.2, (s.end - s.start) - 0.2)
        step = span / len(toks)
        ids = []
        for i, tok in enumerate(toks):
            w = Word(f"w_{n:06d}", tok, round(s.start + 0.1 + i * step, 3), round(s.start + 0.1 + (i + 0.85) * step, 3), 0.95)
            words.append(w)
            ids.append(w.word_id)
            n += 1
        sid = f"sent_{len(sents):04d}"
        sents.append(Sentence(sid, s.narration, words[-len(toks)].start, words[-1].end, ids, 0.95))
        s.sentence_ids = [sid]
    dur = p.voice_over.duration or max([s.end for s in p.scenes] + [0.0])
    tr = Transcript("tr_test", AudioInfo(p.voice_over.asset_id or "", dur, 48000, 2, None, "voice.wav"), words, sents, ProviderInfo("test"), "punctuation")
    p.transcription = TranscriptionState(tr)
    _ = wps
    return tr


def add_clip(p: Project, track_id: str, asset: Asset | None, start: float, duration: float, *, scene: Scene | None = None, kind: str = "media", **kw) -> Clip:
    """Insert a clip directly (no commands, no overlap checks): QC tests *want* to be able to build broken timelines."""
    t = p.timeline.get_track(track_id)
    c = Clip(kw.pop("id", new_clip_id()), track_id, asset.id if asset else "", start, duration, source_in=kw.pop("source_in", 0.0),
             source_out=kw.pop("source_out", kw.pop("source_in", 0.0) + duration * kw.get("speed", 1.0)), kind=kind, scene_id=scene.id if scene else kw.pop("scene_id", ""), **kw)
    t.clips.append(c)
    t.sort()
    return c


def qc_ctx(p: Project, settings: QCSettings | None = None, **kw) -> QCContext:
    """A context over the project itself (no deep copy needed in tests)."""
    if settings is not None:
        p.qc_settings = settings
    return QCContext.build(p, p.qc_settings, detach=False, **kw)


def run_checker(checker: BaseChecker, ctx: QCContext) -> CheckerOutput:
    return checker.run(ctx, lambda f, m: None)


def codes(out: CheckerOutput) -> list[str]:
    return sorted(i.code for i in out.issues)


def find(out: CheckerOutput, code: str):
    return [i for i in out.issues if i.code == code]
