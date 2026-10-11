"""Synthetic projects for repeatable performance benchmarks and large-project tests: no real media, no network, no AI.

``build_project(scenes=500, ...)`` creates a real ``Project`` (scenes, transcript words, a library of placeholder assets, a multi-track timeline with
captions, text, graphics, music/SFX and dense keyframes). Media files are tiny placeholders unless ``real_files`` is requested, so a 1,000-scene project
builds in a second or two. Everything is deterministic for a given ``seed``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from app.analysis.models import Origin, Scene, SceneStatus
from app.media.asset import Asset, AssetType, SourceType
from app.project.project import Project
from app.project.project_schema import ProjectSettings
from app.storage.paths import ProjectPaths
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT, Clip
from app.timeline.keyframes import Keyframe
from app.timeline.timeline import new_clip_id
from app.transcription.models import AudioInfo, ProviderInfo, Sentence, Transcript, TranscriptionState, Word

SIZES = {"small": 20, "medium": 100, "large": 500, "stress": 1000}
WORDS = ("silver prices rose sharply last week as investors moved away from risky assets while central banks signalled caution about inflation and growth in several "
         "major economies across the region").split()


@dataclass(frozen=True)
class SyntheticSpec:
    scenes: int = 100
    scene_seconds: float = 6.0
    assets_per_scene: float = 1.5  # library size relative to the scene count (hundreds to thousands of imported assets)
    fraction_4k: float = 0.25  # share of video assets declared 3840x2160 (the rest 1920x1080)
    fraction_images: float = 0.35
    captions_per_scene: int = 3
    keyframes_per_visual: int = 8  # dense animation
    extra_tracks: int = 0  # additional video tracks beyond the nine defaults (multi-track stress)
    seed: int = 7


@dataclass
class SyntheticProject:
    project: Project
    spec: SyntheticSpec
    clips: int
    assets: int
    words: int
    duration: float

    def summary(self) -> dict:
        return {"scenes": self.spec.scenes, "clips": self.clips, "assets": self.assets, "words": self.words, "duration_s": round(self.duration, 1), "tracks": len(self.project.timeline.tracks)}


def build_project(root: Path, spec: SyntheticSpec | int = 100, *, real_files: bool = False, name: str = "Synthetic", media_pool: dict[str, Path] | None = None) -> SyntheticProject:
    """Create the project in memory with its folder structure under ``root`` (not saved; call ``ProjectManager.save``/``Project.to_document`` as needed).

    ``media_pool`` maps "image" / "video" / "audio" to a small REAL file; every asset of that kind then gets its own copy of it (so thumbnails, probing and proxies work for
    real) while the declared resolution / duration stay those of the synthetic library (e.g. 4K) — the declared metadata, not the bytes, is what the project model sees."""
    spec = SyntheticSpec(scenes=spec) if isinstance(spec, int) else spec
    rng = random.Random(spec.seed)
    p = Project.new(name, ProjectSettings(1920, 1080, 30, "16:9"))
    p.root = Path(root)
    ProjectPaths(p.root).create_structure()
    total = spec.scenes * spec.scene_seconds

    def asset(kind: str, w, h, dur) -> Asset:
        aid = p.assets.new_id()
        folder = {"audio": "audio", "video": "video", "image": "images"}[kind]
        rel = f"media/{folder}/{aid}.{'wav' if kind == 'audio' else 'mp4' if kind == 'video' else 'png'}"
        size = 64
        if media_pool and kind in media_pool:
            import shutil

            path = p.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(media_pool[kind], path)
            size = path.stat().st_size
        elif real_files:
            path = p.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\0" * 64)
        a = Asset(aid, AssetType(kind), SourceType.USER_MEDIA, rel, f"{kind}_{aid}", dur, w, h, 30.0 if kind == "video" else None, None, False, None,
                  48000 if kind == "audio" else None, 2 if kind == "audio" else None, size, f"hash_{aid}")
        p.assets.add(a)
        return a

    voice = asset("audio", None, None, total)
    p.voice_over.asset_id, p.voice_over.filename, p.voice_over.duration = voice.id, voice.name, total
    music, sfx = asset("audio", None, None, total), asset("audio", None, None, 2.0)

    n_assets = max(3, int(spec.scenes * spec.assets_per_scene))
    lib: list[Asset] = []
    for _ in range(n_assets):
        if rng.random() < spec.fraction_images:
            lib.append(asset("image", 1920, 1080, None))
        else:
            big = rng.random() < spec.fraction_4k
            lib.append(asset("video", 3840 if big else 1920, 2160 if big else 1080, rng.choice((6.0, 12.0, 45.0, 180.0))))

    for i in range(spec.extra_tracks):
        from app.timeline.track import Track, TrackKind

        p.timeline.insert_track(Track(id=f"track_x{i + 1}", name=f"X{i + 1} Extra", kind=TrackKind.VIDEO))

    words, sentences, clips = [], [], 0
    wn = 0
    for i in range(spec.scenes):
        t0, t1 = i * spec.scene_seconds, (i + 1) * spec.scene_seconds
        toks = [WORDS[(i * 3 + k) % len(WORDS)] for k in range(14)]
        text = " ".join(toks).capitalize() + "."
        sid = f"scene_{i + 1:04d}"
        s = Scene(sid, str(i + 1), t0, t1, text, [], toks[0], "", 0.5, 0.9, SceneStatus.READY, Origin.AI)
        p.scenes.append(s)
        ids = []
        step = (spec.scene_seconds - 0.4) / len(toks)
        for k, tok in enumerate(toks):
            w = Word(f"w_{wn:07d}", tok, round(t0 + 0.1 + k * step, 3), round(t0 + 0.1 + (k + 0.85) * step, 3), 0.95)
            words.append(w)
            ids.append(w.word_id)
            wn += 1
        sent = Sentence(f"sent_{i:05d}", text, words[-len(toks)].start, words[-1].end, ids, 0.95)
        sentences.append(sent)
        s.sentence_ids = [sent.sentence_id]

        a = lib[i % len(lib)]
        track = "track_v3" if a.type is AssetType.IMAGE else ("track_v1" if i % 2 == 0 else "track_v2")
        vis = Clip(new_clip_id(), track, a.id, t0, spec.scene_seconds, source_in=0.0, source_out=spec.scene_seconds, scene_id=sid, created_by="AI")
        for k in range(spec.keyframes_per_visual):
            vis.keyframes.append(Keyframe("scale", spec.scene_seconds * k / max(1, spec.keyframes_per_visual - 1), 1.0 + 0.02 * k))
        p.timeline.get_track(track).clips.append(vis)
        clips += 1
        # captions: a few per scene on the caption track
        cw = spec.scene_seconds / max(1, spec.captions_per_scene)
        for c in range(spec.captions_per_scene):
            cap = Clip(new_clip_id(), "track_v6", "", t0 + c * cw, cw, kind=KIND_CAPTION, scene_id=sid, created_by="AI", text={"text": " ".join(toks[c * 4:(c + 1) * 4]) or tok})
            p.timeline.get_track("track_v6").clips.append(cap)
            clips += 1
        if i % 4 == 0:
            p.timeline.get_track("track_v5").clips.append(Clip(new_clip_id(), "track_v5", "", t0 + 0.5, 2.5, kind=KIND_TEXT, scene_id=sid, created_by="AI", text={"text": f"Title {i}"}))
            clips += 1
        if i % 5 == 0:
            p.timeline.get_track("track_v4").clips.append(Clip(new_clip_id(), "track_v4", "", t0 + 1.0, 3.0, kind=KIND_GRAPHIC, scene_id=sid, created_by="AI", effects={"kind": "lower_third"}))
            clips += 1
        if i % 6 == 0:
            p.timeline.get_track("track_a3").clips.append(Clip(new_clip_id(), "track_a3", sfx.id, t0 + 0.2, 2.0, source_out=2.0, created_by="AI"))
            clips += 1
        for k in range(spec.extra_tracks):
            if i % (k + 2) == 0:
                x = lib[(i + k) % len(lib)]
                if x.type is AssetType.VIDEO:
                    p.timeline.get_track(f"track_x{k + 1}").clips.append(Clip(new_clip_id(), f"track_x{k + 1}", x.id, t0, spec.scene_seconds, source_out=spec.scene_seconds))
                    clips += 1
    p.timeline.get_track("track_a1").clips.append(Clip(new_clip_id(), "track_a1", voice.id, 0.0, total, source_out=total, audio={"role": "VOICE"}))
    p.timeline.get_track("track_a2").clips.append(Clip(new_clip_id(), "track_a2", music.id, 0.0, total, source_out=total, audio={"role": "MUSIC", "volume": 0.2}))
    clips += 2
    for t in p.timeline.tracks:
        t.sort()
    tr = Transcript("tr_syn", AudioInfo(voice.id, total, 48000, 2, None, "voice.wav"), words, sentences, ProviderInfo("synthetic"), "punctuation")
    p.transcription = TranscriptionState(tr)
    return SyntheticProject(p, spec, clips, len(p.assets.all()), len(words), total)
