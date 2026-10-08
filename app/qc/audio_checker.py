"""Audio quality control (spec 19-21): the voice-over first, then music and sound effects against it, then silence.

Priority is VOICE > SFX > MUSIC: the narration must stay clear and dominant, music supports, effects accent. There is no universal loudness target; every threshold is
in ``QCSettings.audio``. Measurements come from the real audio (decoded on the QC worker through the Phase 5 backend) combined with the mix plan the renderer uses
(clip volume x track volume x volume keyframes x fades), so "how loud is the music under this sentence" is answered with the same gains the export will apply.
Without FFmpeg the checker falls back to the stored voice analysis and says so (lower confidence); it never crashes.

Silence is never removed automatically: a pause the transcript supports (a sentence boundary, a gap between words) is intentional; only silence where the script has
words, or an unexplained long one, is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.audio.backend import ANALYSIS_SR, AudioError, FFmpegAudioBackend
from app.audio.ducking import speech_from_silence, speech_segments
from app.audio.loudness import SILENCE_DB, WIN, clip_runs, db, envelope, silence_regions
from app.audio.mix import AudioMixService, MixItem
from app.audio.priority import VoicePriorityController
from app.presentation.exports import music_assignments
from app.presentation.models import PreviewMode
from app.qc import fix_catalog as fx
from app.qc.checker_base import BaseChecker, CheckerOutput
from app.qc.context import ProgressFn, QCContext, sha
from app.qc.issue_model import QCCategory
from app.qc.media_facts import ffmpeg_ready
from app.qc.severity import Severity
from app.timeline.clip import KIND_MEDIA, Clip
from app.timeline.track import Track, TrackKind

STEP = 0.1  # s: resolution of the gain curves
MIN_SEGMENT = 0.8  # speech stretches shorter than this are too short to compare in level
MAX_JUMP_ISSUES = 6
SEVERE_CLIP_RUNS = 10
STEREO_SR = 8000
ABRUPT_SECONDS = 0.15  # a level change faster than this is a click / pump, not a duck


def _lin_db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-6))


@dataclass
class _Voice:
    clip: Clip | None = None
    track: Track | None = None
    samples: np.ndarray | None = None  # mono at ANALYSIS_SR
    offset: float = 0.0  # timeline time of file time 0
    gain_db: float = 0.0  # clip x track x processing gain
    peak_db: float | None = None
    rms_db: float | None = None  # over the spoken stretches, before gain
    env: np.ndarray | None = None
    fallback: bool = False  # measurements come from the stored analysis, not from decoded audio
    notes: list[str] = field(default_factory=list)


@dataclass
class _Group:
    """The pieces of one music assignment / one sound effect, with their level relative to the voice."""

    key: str
    role: str
    items: list[MixItem]
    clip: Clip | None
    track: Track | None
    rms_db: float
    measured: bool


class AudioChecker(BaseChecker):
    id = "audio"
    label = "Audio"
    categories = (QCCategory.AUDIO, QCCategory.SILENCE)
    domains = ("audio", "timeline", "transcript", "assets")
    settings_sections = ("audio",)
    scene_local = False
    expensive = True
    version = "1"

    # ------------------------------------------------------------------ the work
    def run(self, ctx: QCContext, report: ProgressFn) -> CheckerOutput:
        out = CheckerOutput()
        cfg = ctx.settings.audio
        plan = AudioMixService(None).build_plan(ctx.project, PreviewMode.FULL)  # type: ignore[arg-type] - planning needs no backend
        report(0.05, "Reading the voice-over")
        voice = self._load_voice(ctx, out)
        speech = self._speech(ctx, voice)
        out.metrics["speech_seconds"] = round(sum(b - a for a, b in speech), 2)
        out.metrics["speech_ratio"] = round(out.metrics["speech_seconds"] / ctx.duration, 3) if ctx.duration > 0 else 0.0
        ctx.check_cancel()
        report(0.25, "Checking the voice")
        self._voice(ctx, out, cfg, voice, speech)
        ctx.check_cancel()
        report(0.5, "Checking silence")
        self._silence(ctx, out, cfg, voice, speech)
        ctx.check_cancel()
        report(0.65, "Checking music and effects against the voice")
        voice_gain = lambda t: plan.gain_at("VOICE", t)  # noqa: E731
        self._music(ctx, out, cfg, plan.items, voice, speech, voice_gain)
        ctx.check_cancel()
        self._sfx(ctx, out, cfg, plan.items, voice, speech, voice_gain)
        report(1.0, "Audio check complete")
        out.metrics.update(peak_db=_r(voice.peak_db), voice_rms_db=_r(voice.rms_db), lufs=_r(ctx.project.audio_analysis.lufs) if ctx.project.audio_analysis else None)
        out.notes.append("voice measured from decoded audio" if voice.samples is not None else "voice measured from the stored analysis (audio not decoded)" if voice.fallback else "voice not measured")
        return out

    # ------------------------------------------------------------------ voice: load / measure
    def _voice_clips(self, ctx: QCContext) -> list[tuple[Track, Clip]]:
        vid = ctx.project.voice_over.asset_id
        return [(t, c) for t, c in ctx.clips(kind=KIND_MEDIA, track_kinds=(TrackKind.AUDIO,))
                if str(c.audio.get("role", "")).upper() == "VOICE" or c.slot == "voice" or (vid and c.asset_id == vid)]

    def _load_voice(self, ctx: QCContext, out: CheckerOutput) -> _Voice:
        v = _Voice()
        clips = self._voice_clips(ctx)
        if clips:
            v.track, v.clip = clips[0]
            v.offset = v.clip.timeline_start - v.clip.source_in
            proc = ctx.project.audio_processing
            v.gain_db = _lin_db(float(v.clip.audio.get("volume", 1.0)) * float(v.track.volume)) + (float(proc.gain_db) if getattr(proc, "enabled", False) else 0.0)  # type: ignore[union-attr]
        asset = ctx.asset(ctx.project.voice_over.asset_id)
        if asset is not None and ffmpeg_ready(ctx):
            path = ctx.asset_path(asset)
            if path.is_file():
                try:
                    v.samples = ctx.memo("audio.voice", lambda: self._decode(ctx, path))
                except AudioError as exc:
                    out.notes.append(f"voice not decoded: {exc.user_message}")
        if v.samples is not None and len(v.samples):
            v.peak_db = db(float(np.max(np.abs(v.samples))))
            v.env = envelope(v.samples, ANALYSIS_SR, WIN)
        else:
            an = ctx.project.audio_analysis
            if an is not None and asset is not None and an.asset_id == asset.id:
                v.fallback, v.peak_db, v.rms_db = True, an.peak_db, an.rms_db
                out.notes.append("FFmpeg is not available: using the stored voice analysis (lower confidence)")
        return v

    @staticmethod
    def _decode(ctx: QCContext, path: Path) -> np.ndarray:
        return FFmpegAudioBackend(lambda: ctx.ffmpeg.ffmpeg()).decode_mono(path)  # type: ignore[union-attr]

    def _speech(self, ctx: QCContext, voice: _Voice) -> list[tuple[float, float]]:
        """When the voice is speaking, on the timeline: the transcript's words when there are any, else the audio's non-silent stretches."""
        words = ctx.words
        if words:
            return speech_segments(words)
        if voice.env is not None:
            sil = silence_regions(voice.env, WIN, SILENCE_DB, 0.3)
            dur = len(voice.env) * WIN
            return [(a + voice.offset, b + voice.offset) for a, b in speech_from_silence(dur, sil)]
        return []

    def _slice(self, voice: _Voice, a: float, b: float) -> np.ndarray:
        """Voice samples between two timeline times."""
        if voice.samples is None:
            return np.zeros(0, dtype=np.float32)
        i, j = int((a - voice.offset) * ANALYSIS_SR), int((b - voice.offset) * ANALYSIS_SR)
        return voice.samples[max(0, i):max(0, j)]

    # ------------------------------------------------------------------ voice checks
    def _voice(self, ctx: QCContext, out: CheckerOutput, cfg, v: _Voice, speech: list[tuple[float, float]]) -> None:
        p = ctx.project
        if v.clip is not None and v.track is not None and not self._audible(ctx, v):
            why = "its track is muted or hidden" if not ctx.track_audible(v.track) else "its volume is zero"
            out.issues.append(self.issue(
                "audio.voice_missing", QCCategory.AUDIO, Severity.CRITICAL, "The voice-over cannot be heard", description=f"The voice-over is on the timeline but {why}.", clip=v.clip, track=v.track,
                why="A video without audible narration loses its message.", current="inaudible", recommended="audible narration", suggested_fix="Unmute the voice track or raise its volume.", viewer_impact=1.0,
                signature="inaudible", ctx=ctx))
        if v.clip is not None and p.voice_over.duration and v.clip.duration < p.voice_over.duration - cfg.voice_duration_tolerance:
            cut = p.voice_over.duration - v.clip.duration
            out.issues.append(self.issue(
                "audio.voice_duration", QCCategory.AUDIO, Severity.ERROR, "The voice-over is cut short", description=f"The voice-over is {p.voice_over.duration:.1f} s long but only {v.clip.duration:.1f} s of it is on the timeline "
                f"({cut:.1f} s missing; tolerance {cfg.voice_duration_tolerance:.1f} s).", clip=v.clip, track=v.track, start=v.clip.timeline_end, end=v.clip.timeline_end + cut,
                why="Narration is cut off before it ends.", current=f"{v.clip.duration:.1f} s", recommended=f"{p.voice_over.duration:.1f} s", suggested_fix="Extend the voice clip to the full recording.",
                viewer_impact=0.9, signature=sha(round(cut, 1)), ctx=ctx))
        if v.samples is None and not v.fallback:
            return
        measured = v.samples is not None
        conf = 100.0 if measured else 60.0
        # ---- clipping
        if measured:
            level = 10 ** (cfg.clip_dbfs / 20.0)
            hot, runs, times = clip_runs(v.samples, level=level, run=max(2, cfg.clip_samples))  # type: ignore[arg-type]
            if runs:
                ratio = hot / max(1, len(v.samples))  # type: ignore[arg-type]
                sev = Severity.ERROR if runs >= SEVERE_CLIP_RUNS or ratio > 1e-4 else Severity.WARNING
                at = times[0] + v.offset if times else None
                out.issues.append(self.issue(
                    "audio.voice_clipping", QCCategory.AUDIO, sev, "The voice-over clips", description=f"{runs} stretch(es) of flat-topped samples at or above {cfg.clip_dbfs:.1f} dBFS ({hot} samples, first near {_t(at)}).",
                    start=at, end=(at + 0.5) if at is not None else None, affected=["voice-over"], why="Clipped speech sounds harsh and distorted; it cannot be repaired afterwards.", current=f"peak {v.peak_db:.1f} dBFS, {runs} runs",
                    recommended="a clean recording or lower gain", suggested_fix="Re-record or re-export the narration at a lower level; enable the limiter as a stop-gap.", viewer_impact=0.7 if sev is Severity.ERROR else 0.4,
                    signature=sha(runs // 5), metrics={"runs": runs, "hot_samples": hot, "peak_db": _r(v.peak_db)}, ctx=ctx))
        elif v.peak_db is not None and (v.peak_db + v.gain_db) >= cfg.clip_dbfs and v.fallback and (ctx.project.audio_analysis and ctx.project.audio_analysis.clipped_samples > cfg.clip_samples):
            out.issues.append(self.issue(
                "audio.voice_clipping", QCCategory.AUDIO, Severity.WARNING, "The voice-over clips", description=f"The stored analysis counted {ctx.project.audio_analysis.clipped_samples} clipped samples.",
                affected=["voice-over"], why="Clipped speech sounds harsh and distorted.", current=f"peak {v.peak_db:.1f} dBFS", recommended="a clean recording", suggested_fix="Re-record or lower the gain.",
                confidence=conf, viewer_impact=0.4, signature="analysis-clip", ctx=ctx))
        if v.peak_db is not None and v.gain_db > 0 and v.peak_db + v.gain_db >= 0.0 and not getattr(p.audio_processing, "limiter", False) and measured:
            out.issues.append(self.issue(
                "audio.voice_clipping", QCCategory.AUDIO, Severity.WARNING, "The voice gain pushes the narration into clipping",
                description=f"The recording peaks at {v.peak_db:.1f} dBFS and the voice gain is +{v.gain_db:.1f} dB, so the mix exceeds full scale.", affected=["voice-over"], why="The export would distort on the loudest words.",
                current=f"{v.peak_db + v.gain_db:+.1f} dBFS", recommended="below 0 dBFS", suggested_fix="Lower the voice volume or turn the limiter on.", viewer_impact=0.5, signature="gain-clip", ctx=ctx))
        # ---- level over the spoken stretches
        rms = self._speech_rms(v, speech)
        if rms is not None:
            v.rms_db = rms
        if v.rms_db is not None:
            eff = v.rms_db + v.gain_db
            if eff < cfg.voice_min_rms_dbfs:
                sev = Severity.ERROR if eff < cfg.voice_min_rms_dbfs - 10 else Severity.WARNING
                out.issues.append(self.issue(
                    "audio.voice_quiet", QCCategory.AUDIO, sev, "The voice-over is very quiet", description=f"Spoken parts average {eff:.1f} dBFS (threshold {cfg.voice_min_rms_dbfs:.0f} dBFS).",
                    affected=["voice-over"], why="Viewers will turn the volume up and then be hit by louder music and effects.", current=f"{eff:.1f} dBFS RMS", recommended=f"above {cfg.voice_min_rms_dbfs:.0f} dBFS",
                    suggested_fix="Raise the voice volume or enable voice normalisation.", confidence=conf, viewer_impact=0.6, signature=sha(round(eff / 3)), ctx=ctx))
        if not measured:
            return
        # ---- sudden level changes between spoken stretches
        segs = [(a, b, self._rms(self._slice(v, a, b))) for a, b in speech if b - a >= MIN_SEGMENT]
        jumps = [(segs[i - 1], segs[i]) for i in range(1, len(segs)) if segs[i][2] is not None and segs[i - 1][2] is not None and abs(segs[i][2] - segs[i - 1][2]) > cfg.voice_jump_db]
        for (a0, b0, r0), (a1, b1, r1) in jumps[:MAX_JUMP_ISSUES]:
            out.issues.append(self.issue(
                "audio.voice_jump", QCCategory.AUDIO, Severity.WARNING, "Sudden volume change in the narration", description=f"The level changes by {abs(r1 - r0):.1f} dB between {_t(a0)}-{_t(b0)} and {_t(a1)}-{_t(b1)} "
                f"(limit {cfg.voice_jump_db:.0f} dB).", start=b0, end=a1 if a1 > b0 else b0 + 0.5, scene_id=(ctx.scene_at(a1).id if ctx.scene_at(a1) else None), why="Listeners have to keep adjusting; it also makes ducking inconsistent.",
                current=f"{r0:.1f} -> {r1:.1f} dBFS", recommended=f"within {cfg.voice_jump_db:.0f} dB", suggested_fix="Normalise or compress the voice, or match the level of the two takes.", viewer_impact=0.4,
                signature=sha(round(a1), round(abs(r1 - r0))), ctx=ctx))
        if len(jumps) > MAX_JUMP_ISSUES:
            out.notes.append(f"{len(jumps) - MAX_JUMP_ISSUES} more voice level jumps not listed")
        # ---- noise between the words
        gap_env = self._gap_env(v, speech)
        if gap_env is not None and len(gap_env) >= 5:
            noise = float(np.percentile(gap_env, 50))
            if noise > cfg.noise_floor_dbfs:
                out.issues.append(self.issue(
                    "audio.voice_noise", QCCategory.AUDIO, Severity.WARNING if noise > cfg.noise_floor_dbfs + 10 else Severity.NOTICE, "Background noise in the voice recording",
                    description=f"Between the words the recording sits at {noise:.1f} dBFS (threshold {cfg.noise_floor_dbfs:.0f} dBFS).", affected=["voice-over"], why="Noise is amplified by loudness normalisation and distracts from the words.",
                    current=f"{noise:.1f} dBFS", recommended=f"below {cfg.noise_floor_dbfs:.0f} dBFS", suggested_fix="Enable noise reduction on the voice or re-record in a quieter place.",
                    confidence=80.0, viewer_impact=0.3, signature=sha(round(noise / 4)), metrics={"noise_floor_db": round(noise, 1)}, ctx=ctx))
        self._channels(ctx, out, cfg)

    @staticmethod
    def _audible(ctx: QCContext, v: _Voice) -> bool:
        return bool(v.track and v.clip and ctx.track_audible(v.track) and float(v.clip.audio.get("volume", 1.0)) > 1e-4 and float(v.track.volume) > 1e-4)

    @staticmethod
    def _rms(x: np.ndarray) -> float | None:
        return db(float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))) if len(x) else None

    def _speech_rms(self, v: _Voice, speech: list[tuple[float, float]]) -> float | None:
        """Power-mean level over the spoken stretches (the pauses must not drag the average down)."""
        if v.samples is None:
            return None
        parts = [self._slice(v, a, b) for a, b in speech]
        parts = [x for x in parts if len(x)]
        return self._rms(np.concatenate(parts)) if parts else self._rms(v.samples)

    def _gap_env(self, v: _Voice, speech: list[tuple[float, float]]) -> np.ndarray | None:
        """Envelope frames (dBFS) from the gaps between speech, ignoring the first/last 0.2 s of each gap (word tails)."""
        if v.env is None or not speech:
            return None
        frames: list[np.ndarray] = []
        edges = [(0.0, speech[0][0])] + [(speech[i][1], speech[i + 1][0]) for i in range(len(speech) - 1)] + [(speech[-1][1], len(v.env) * WIN + v.offset)]
        for a, b in edges:
            a, b = a + 0.2, b - 0.2
            if b - a >= 0.2:
                i, j = int(max(0.0, a - v.offset) / WIN), int(max(0.0, b - v.offset) / WIN)
                if j > i:
                    frames.append(v.env[i:j])
        return np.concatenate(frames) if frames else None

    def _channels(self, ctx: QCContext, out: CheckerOutput, cfg) -> None:
        asset = ctx.asset(ctx.project.voice_over.asset_id)
        if asset is None or (asset.channels or 0) < 2:
            return
        path = ctx.asset_path(asset)
        try:
            st = ctx.memo("audio.voice_stereo", lambda: self._decode_stereo(ctx, path))
        except AudioError:
            return
        if st is None or st.shape[1] < STEREO_SR:
            return
        l_db, r_db = self._rms(st[0]), self._rms(st[1])
        corr = float(np.corrcoef(st[0][::4], st[1][::4])[0, 1]) if l_db is not None and r_db is not None and np.std(st[0]) > 1e-6 and np.std(st[1]) > 1e-6 else 1.0
        if l_db is None or r_db is None:
            return
        if abs(l_db - r_db) > 30:
            quiet = "left" if l_db < r_db else "right"
            out.issues.append(self.issue(
                "audio.channel", QCCategory.AUDIO, Severity.WARNING, "One channel of the voice-over is silent", description=f"The {quiet} channel is {abs(l_db - r_db):.0f} dB quieter than the other.",
                affected=["voice-over"], why="The narration plays from one ear only on stereo headphones.", current=f"L {l_db:.0f} / R {r_db:.0f} dBFS", recommended="both channels at a similar level",
                suggested_fix="Convert the voice-over to mono or re-export it with both channels.", viewer_impact=0.5, signature=f"silent-{quiet}", ctx=ctx))
        elif corr < -0.5:
            out.issues.append(self.issue(
                "audio.channel", QCCategory.AUDIO, Severity.WARNING, "The voice-over channels are out of phase", description=f"Left and right are inversely correlated ({corr:.2f}); on a mono speaker the voice will cancel out.",
                affected=["voice-over"], why="Mono playback (phones, smart speakers) would lose the narration.", current=f"correlation {corr:.2f}", recommended="in-phase channels", suggested_fix="Invert one channel or convert to mono.",
                viewer_impact=0.7, signature="phase", ctx=ctx))

    @staticmethod
    def _decode_stereo(ctx: QCContext, path: Path) -> np.ndarray | None:
        import subprocess  # noqa: PLC0415

        try:
            r = subprocess.run([ctx.ffmpeg.ffmpeg(), "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(STEREO_SR), "-f", "f32le", "-"],  # type: ignore[union-attr]
                               capture_output=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            raise AudioError("The voice-over could not be decoded for the channel check.") from exc
        if r.returncode != 0 or not r.stdout:
            return None
        x = np.frombuffer(r.stdout, dtype="<f4")
        x = x[: (len(x) // 2) * 2].reshape(-1, 2)
        return x.T.copy()

    # ------------------------------------------------------------------ silence
    def _silence(self, ctx: QCContext, out: CheckerOutput, cfg, v: _Voice, speech: list[tuple[float, float]]) -> None:
        regions: list[tuple[float, float]]
        conf = 100.0
        if v.env is not None:
            regions = [(a + v.offset, b + v.offset) for a, b in silence_regions(v.env, WIN, SILENCE_DB, min(cfg.accidental_gap_seconds, 0.5))]
        elif v.fallback and ctx.project.audio_analysis is not None:
            regions = [(float(a), float(b)) for a, b in ctx.project.audio_analysis.silence_regions]
            conf = 60.0
        else:
            return
        words = ctx.words
        intentional: list[list[float]] = []
        total = 0.0
        first_word, last_word = (words[0].start, words[-1].end) if words else (0.0, ctx.duration)
        for a, b in regions:
            ctx.check_cancel()
            length = b - a
            total += length
            if ctx.in_intentional_gap(a, b):
                continue
            spoken = [w for w in words if a - 1e-6 <= (w.start + w.end) / 2 <= b + 1e-6]
            sc = ctx.scene_at((a + b) / 2)
            if spoken and length >= cfg.accidental_gap_seconds:
                out.issues.append(self.issue(
                    "silence.accidental", QCCategory.SILENCE, Severity.ERROR, "Silence where the script has words",
                    description=f"The audio is silent for {length:.1f} s around {_t(a)}, but the transcript has {len(spoken)} word(s) there (\"{' '.join(w.text for w in spoken[:6])}\").",
                    start=a, end=b, scene_id=sc.id if sc else None, why="Part of the narration is missing or muted.", current=f"{length:.1f} s of silence", recommended="the spoken words audible",
                    suggested_fix="Check the voice-over file or re-record this passage (QC never removes or fills silence by itself).", confidence=conf, viewer_impact=0.9,
                    signature=sha(round(a, 0), len(spoken)), metrics={"words": len(spoken)}, ctx=ctx))
                continue
            if spoken:
                continue  # a short dropout under words is covered by the accidental-silence rule above
            if a < first_word - 1e-6 or b > last_word + 1e-6:
                if length >= cfg.unnatural_silence_seconds * 2:
                    out.issues.append(self.issue(
                        "silence.excessive", QCCategory.SILENCE, Severity.NOTICE, "Long silence before or after the narration", description=f"{length:.1f} s of silence at {_t(a)}-{_t(b)} outside the spoken part.",
                        start=a, end=b, why="Dead air at the start or end of a video loses viewers.", current=f"{length:.1f} s", recommended=f"under {cfg.unnatural_silence_seconds * 2:.1f} s",
                        suggested_fix="Trim the recording or cover it with music (QC never removes silence automatically).", confidence=min(conf, 85.0), viewer_impact=0.2, signature=sha(round(a, 0)), ctx=ctx))
                continue
            if length < cfg.unnatural_silence_seconds:
                continue
            prev = next((w for w in reversed(words) if w.end <= a + 0.3), None)
            at_boundary = prev is not None and prev.text.rstrip().endswith((".", "!", "?", "…", ":"))
            if at_boundary and length < cfg.unnatural_silence_seconds * 2:
                intentional.append([round(a, 2), round(b, 2)])
            elif at_boundary:
                out.issues.append(self.issue(
                    "silence.excessive", QCCategory.SILENCE, Severity.WARNING, "Long silence in the narration", description=f"{length:.1f} s without any sound after a finished sentence ({_t(a)}-{_t(b)}).",
                    start=a, end=b, scene_id=sc.id if sc else None, why="Long dead air makes viewers think the video stalled.", current=f"{length:.1f} s", recommended=f"under {cfg.unnatural_silence_seconds * 2:.1f} s or intentional",
                    suggested_fix="Shorten the pause in the voice-over, or declare it intentional (QC never removes silence automatically).", confidence=min(conf, 85.0), viewer_impact=0.4,
                    signature=sha(round(a, 0)), ctx=ctx))
            else:
                out.issues.append(self.issue(
                    "silence.unnatural", QCCategory.SILENCE, Severity.WARNING, "Unexplained silence inside a sentence", description=f"{length:.1f} s of silence at {_t(a)} with no pause in the script around it.",
                    start=a, end=b, scene_id=sc.id if sc else None, why="A gap in mid-sentence usually means a dropout or a bad edit of the recording.", current=f"{length:.1f} s", recommended="continuous speech",
                    suggested_fix="Listen to this point in the voice-over; fix the recording if it is a dropout.", confidence=min(conf, 80.0), viewer_impact=0.5, signature=sha(round(a, 0)), ctx=ctx))
        if intentional:
            out.issues.append(self.issue(
                "silence.intentional", QCCategory.SILENCE, Severity.INFO, f"{len(intentional)} deliberate pause(s) recognised",
                description="Pauses after finished sentences: " + ", ".join(f"{_t(a)}-{_t(b)}" for a, b in intentional[:6]) + ". Left as they are.", why="Pauses give the narration room to breathe; QC never removes them.",
                current=f"{len(intentional)} pause(s)", recommended="keep", suggested_fix="Nothing to do.", viewer_impact=0.0, signature="pauses", metrics={"pauses": intentional[:20]}, ctx=ctx))
        out.metrics.update(silence_seconds=round(total, 2), silence_regions=len(regions), intentional_pauses=len(intentional))

    # ------------------------------------------------------------------ music / sfx
    def _groups(self, ctx: QCContext, items: list[MixItem], role: str) -> list[_Group]:
        by: dict[str, list[MixItem]] = {}
        clip_of = {c.id: (t, c) for t, c in ctx.clips(kind=KIND_MEDIA, track_kinds=(TrackKind.AUDIO,))}
        for it in items:
            if it.role != role or it.clip_id not in clip_of:
                continue
            t, c = clip_of[it.clip_id]
            by.setdefault(str(c.metadata.get("assignment_id" if role == "MUSIC" else "sfx_id", c.id)), []).append(it)
        groups = []
        for key, its in by.items():
            its.sort(key=lambda i: i.start)
            t, c = clip_of[its[0].clip_id]
            rms, measured = self._source_rms(ctx, its[0].path)
            groups.append(_Group(key, role, its, c, t, rms, measured))
        return sorted(groups, key=lambda g: g.items[0].start)

    def _source_rms(self, ctx: QCContext, path: Path) -> tuple[float, bool]:
        """Average level (dBFS) of the source's active frames; a nominal -20 dBFS when it cannot be decoded."""
        if not ffmpeg_ready(ctx) or not path.is_file():
            return -20.0, False

        def calc() -> tuple[float, bool]:
            try:
                x = self._decode(ctx, path)
            except AudioError:
                return -20.0, False
            env = envelope(x, ANALYSIS_SR, WIN)
            active = env[env > -50.0]
            return (float(10 * np.log10(np.mean(10 ** (active / 10.0)))) if len(active) else -20.0), True

        return ctx.memo(f"audio.rms.{path}", calc)

    def _curve(self, items: list[MixItem], duration: float) -> np.ndarray:
        n = int(duration / STEP) + 1
        ts = np.arange(n) * STEP
        return np.array([sum(i.gain_at(float(t)) for i in items) for t in ts], dtype=np.float64)

    def _music(self, ctx: QCContext, out: CheckerOutput, cfg, items: list[MixItem], voice: _Voice, speech, voice_gain) -> None:
        groups = self._groups(ctx, items, "MUSIC")
        if not groups or not speech:
            return
        dur = ctx.duration
        n = int(dur / STEP) + 1
        speaking = np.zeros(n, dtype=bool)
        for a, b in speech:
            speaking[int(a / STEP):int(min(b, dur) / STEP) + 1] = True
        vgain = np.array([voice_gain(float(t)) for t in np.arange(n) * STEP])
        v_rms = (voice.rms_db if voice.rms_db is not None else -20.0)
        controller = VoicePriorityController(ctx.project.audio_settings)
        assignments = {a.assignment_id: a for a in music_assignments(ctx.project)}
        depths: list[float] = []
        for g in groups:
            ctx.check_cancel()
            curve = self._curve(g.items, dur)
            active = curve > 1e-4
            under = active & speaking & (vgain > 1e-4)
            if under.sum() * STEP < 1.0:
                continue
            gaps = active & ~speaking
            rel = (g.rms_db + 20 * np.log10(np.maximum(curve, 1e-6))) - (v_rms + 20 * np.log10(np.maximum(vgain, 1e-6)))  # music level relative to the voice, dB
            rel_speech = float(np.median(rel[under]))
            gain_speech = float(np.median(curve[under]))
            gap_gain = float(np.median(curve[gaps])) if gaps.sum() * STEP >= 1.0 else None
            depth = _lin_db(gap_gain) - _lin_db(gain_speech) if gap_gain else None
            if depth is not None:
                depths.append(depth)
            masked = controller.masking("MUSIC", lambda t, c=curve: float(c[min(len(c) - 1, int(t / STEP))]), speech, dur)
            masked_s = sum(m.end - m.start + STEP for m in masked)
            loud_s = float((rel[under] > cfg.music_over_voice_db).sum() * STEP)
            sc_id = self._scene_for(ctx, g.items[0].start)
            where = dict(start=g.items[0].start, end=g.items[-1].end, scene_id=sc_id, clip=g.clip, track=g.track, ctx=ctx)
            conf = 100.0 if (g.measured and not voice.fallback) else 75.0
            m = {"relative_db": round(rel_speech, 1), "duck_depth_db": _r(depth), "masked_seconds": round(masked_s, 1), "loud_seconds": round(loud_s, 1)}
            if rel_speech > 0.0 and loud_s >= 1.0:
                out.issues.append(self.issue(
                    "audio.music_masks_speech", QCCategory.AUDIO, Severity.ERROR, "Music covers the narration",
                    description=f"While the voice speaks the music is {rel_speech:+.1f} dB relative to it (louder than the voice) for {loud_s:.1f} s.", why="Voice must stay dominant; the narration is the message.",
                    current=f"{rel_speech:+.1f} dB vs voice", recommended=f"{cfg.music_over_voice_db:.0f} dB or lower", suggested_fix="Lower the music under speech (ducking) or reduce its level.", confidence=conf,
                    viewer_impact=0.9, signature=sha(g.key), metrics=m, fix=self._duck_fix(ctx, cfg, g, speech, gain_speech, rel_speech, assignments), **where))
            elif (depth is not None and depth < cfg.min_duck_db and (masked_s >= 1.0 or loud_s >= 1.0)) or (depth is None and loud_s >= 1.0 and masked_s >= 1.0):
                has_depth = f" The music only falls {max(depth, 0.0):.1f} dB under speech (needs {cfg.min_duck_db:.0f} dB)." if depth is not None else ""
                out.issues.append(self.issue(
                    "audio.insufficient_ducking", QCCategory.AUDIO, Severity.WARNING, "Music is not lowered enough under the voice",
                    description=f"Under speech the music sits {rel_speech:+.1f} dB relative to the voice for {max(loud_s, masked_s):.1f} s.{has_depth}", why="Competing music makes the narration harder to follow.",
                    current=f"{max(depth or 0.0, 0.0):.1f} dB of ducking", recommended=f"at least {cfg.min_duck_db:.0f} dB", suggested_fix="Add ducking keyframes so the music dips under the voice and returns in pauses.",
                    confidence=conf, viewer_impact=0.6, signature=sha(g.key), metrics=m, fix=self._duck_fix(ctx, cfg, g, speech, gain_speech, rel_speech, assignments), **where))
            elif loud_s >= 1.0:
                out.issues.append(self.issue(
                    "audio.music_loud", QCCategory.AUDIO, Severity.WARNING, "Music is too loud next to the voice",
                    description=f"Even when ducked the music averages {rel_speech:+.1f} dB relative to the voice (limit {cfg.music_over_voice_db:.0f} dB) for {loud_s:.1f} s.", why="Music should support, not compete.",
                    current=f"{rel_speech:+.1f} dB vs voice", recommended=f"{cfg.music_over_voice_db:.0f} dB or lower", suggested_fix="Lower the music volume.", confidence=conf, viewer_impact=0.5,
                    signature=sha(g.key), metrics=m, fix=self._level_fix(ctx, g, rel_speech - cfg.music_over_voice_db), **where))
            self._music_jumps(ctx, out, cfg, g, curve)
        if depths:
            out.metrics["duck_depth_db"] = {"min": round(min(depths), 1), "median": round(float(np.median(depths)), 1), "max": round(max(depths), 1)}

    def _duck_fix(self, ctx: QCContext, cfg, g: _Group, speech, gain_speech: float, rel_speech: float, assignments):
        need = max(cfg.min_duck_db, rel_speech - cfg.music_over_voice_db + 2.0)  # at least the depth the user asked for, and enough to get under the loudness limit
        target = min(gain_speech * 10 ** (-need / 20.0), ctx.project.audio_settings.music_level)
        key = g.key if g.key in assignments else str(g.clip.metadata.get("assignment_id", g.clip.id)) if g.clip else g.key
        spans = [[a, min(b, g.items[-1].end)] for a, b in speech if b > g.items[0].start and a < g.items[-1].end]
        return fx.audio_duck("MUSIC", key, spans, target, ctx.settings) if spans else None

    def _level_fix(self, ctx: QCContext, g: _Group, excess_db: float):
        if g.clip is None or excess_db <= 0:
            return None
        cur = float(g.clip.audio.get("volume", 1.0))
        return fx.audio_level(g.clip.id, cur * 10 ** (-(excess_db + 2.0) / 20.0), f"Lower the volume by {excess_db + 2.0:.0f} dB", ctx.settings)

    def _music_jumps(self, ctx: QCContext, out: CheckerOutput, cfg, g: _Group, curve: np.ndarray) -> None:
        """Abrupt level changes inside one piece (keyframes closer than a ramp) and between abutting pieces."""
        bad: list[tuple[float, float]] = []
        for it in g.items:
            kf = sorted(it.keyframes, key=lambda k: k.time)
            for k0, k1 in zip(kf, kf[1:]):
                span = k1.time - k0.time
                if span < ABRUPT_SECONDS and min(k0.value, k1.value) > 0:
                    d = abs(_lin_db(k1.value) - _lin_db(k0.value))
                    if d > cfg.voice_jump_db:
                        bad.append((it.start + k0.time, d))
        for a, b in zip(g.items, g.items[1:]):
            if 0 <= b.start - a.end < ABRUPT_SECONDS:
                ga, gb = a.gain_at(a.end - 1e-3), b.gain_at(b.start + 1e-3)
                if min(ga, gb) > 1e-4 and abs(_lin_db(gb) - _lin_db(ga)) > cfg.voice_jump_db:
                    bad.append((b.start, abs(_lin_db(gb) - _lin_db(ga))))
        if bad:
            t, d = bad[0]
            out.issues.append(self.issue(
                "audio.music_jump", QCCategory.AUDIO, Severity.WARNING, "Music level jumps suddenly", description=f"The music changes by {d:.0f} dB in under {ABRUPT_SECONDS:.2f} s at {_t(t)} ({len(bad)} place(s)).",
                start=t, end=t + 0.5, scene_id=self._scene_for(ctx, t), clip=g.clip, track=g.track, why="An instant level change is heard as a click or a pump.", current=f"{d:.0f} dB step",
                recommended=f"a ramp of at least {ABRUPT_SECONDS:.2f} s", suggested_fix="Lengthen the fade or the ducking ramp at this point.", viewer_impact=0.3, signature=sha(g.key, round(t)), ctx=ctx))

    def _sfx(self, ctx: QCContext, out: CheckerOutput, cfg, items: list[MixItem], voice: _Voice, speech, voice_gain) -> None:
        groups = self._groups(ctx, items, "SFX")
        if not groups:
            return
        v_rms = (voice.rms_db if voice.rms_db is not None else -20.0)
        controller = VoicePriorityController(ctx.project.audio_settings)
        flagged_clips: set[str] = set()
        for g in groups:
            for it in g.items:
                ctx.check_cancel()
                t_end = min(it.end, it.start + 3.0)
                overlap = self._overlap(speech, it.start, t_end)
                if overlap < 0.2:
                    continue
                mid = it.start + min(it.duration, 3.0) / 2
                gain = max(it.gain_at(it.start + 0.05), it.gain_at(mid))
                vg = max(voice_gain(mid), 1e-3)
                rel = (g.rms_db + _lin_db(gain)) - (v_rms + _lin_db(vg))
                clip = next((c for _t, c in ctx.clips(kind=KIND_MEDIA) if c.id == it.clip_id), None)
                track = next((t for t, c in ctx.clips(kind=KIND_MEDIA) if c.id == it.clip_id), None)
                common = dict(start=it.start, end=it.end, scene_id=self._scene_for(ctx, it.start), clip=clip, track=track, ctx=ctx)
                if rel > cfg.sfx_over_voice_db:
                    flagged_clips.add(it.clip_id)
                    cur = float(clip.audio.get("volume", 1.0)) if clip else 1.0
                    out.issues.append(self.issue(
                        "audio.sfx_loud", QCCategory.AUDIO, Severity.WARNING, "Sound effect too loud over the narration",
                        description=f"The effect at {_t(it.start)} plays {rel:+.1f} dB relative to the voice while speech is active (limit {cfg.sfx_over_voice_db:.0f} dB).", why="Effects are accents; they must not cover words.",
                        current=f"{rel:+.1f} dB vs voice", recommended=f"{cfg.sfx_over_voice_db:.0f} dB or lower", suggested_fix="Lower the effect's volume.", viewer_impact=0.5, signature=sha(it.clip_id),
                        metrics={"relative_db": round(rel, 1)}, fix=(fx.audio_level(it.clip_id, cur * 10 ** (-(rel - cfg.sfx_over_voice_db + 2.0) / 20.0), f"Lower the effect by {rel - cfg.sfx_over_voice_db + 2.0:.0f} dB", ctx.settings) if clip else None),
                        **common))
                elif gain > controller.sfx_ceiling + 1e-6 and overlap >= 0.3:
                    out.issues.append(self.issue(
                        "audio.sfx_over_speech", QCCategory.AUDIO, Severity.NOTICE, "Sound effect plays over speech at a high setting",
                        description=f"The effect at {_t(it.start)} overlaps {overlap:.1f} s of speech at {gain:.0%} gain (ceiling {controller.sfx_ceiling:.0%}).", why="Loud effects over words reduce clarity.",
                        current=f"{gain:.0%}", recommended=f"{controller.sfx_ceiling:.0%} or lower", suggested_fix="Lower the effect's volume or move it into a pause.", viewer_impact=0.3,
                        signature=sha(it.clip_id), confidence=80.0, **common))
        # ---- how often
        starts = sorted(i.start for g in groups for i in g.items)
        worst, at = 0, 0.0
        for i, s0 in enumerate(starts):
            n = sum(1 for s1 in starts[i:] if s1 < s0 + 60.0)
            if n > worst:
                worst, at = n, s0
        if worst > cfg.sfx_repeat_per_minute:
            out.issues.append(self.issue(
                "audio.sfx_repeat", QCCategory.AUDIO, Severity.WARNING, "Too many sound effects", description=f"{worst} effects within one minute from {_t(at)} (limit {cfg.sfx_repeat_per_minute:.0f}).",
                start=at, end=at + 60.0, why="Constant effects feel busy and wear out their impact.", current=f"{worst} per minute", recommended=f"at most {cfg.sfx_repeat_per_minute:.0f}",
                suggested_fix="Keep the effects that carry meaning and remove the rest.", viewer_impact=0.3, signature=sha(round(at / 30)), ctx=ctx))
        gap = ctx.project.audio_settings.min_sfx_gap
        seen_paths: dict[Path, list[float]] = {}
        for g in groups:
            for i in g.items:
                seen_paths.setdefault(i.path, []).append(i.start)
        for path, ts in seen_paths.items():
            ts.sort()
            close = [(a, b) for a, b in zip(ts, ts[1:]) if b - a < gap]
            if close:
                out.issues.append(self.issue(
                    "audio.sfx_repeat", QCCategory.AUDIO, Severity.NOTICE, "The same sound effect repeats quickly", description=f"“{path.stem}” plays again within {gap:.0f} s at {_t(close[0][1])} ({len(close)} time(s)).",
                    start=close[0][0], end=close[0][1], why="Hearing the identical effect twice in a few seconds draws attention to it.", current=f"{close[0][1] - close[0][0]:.1f} s apart",
                    recommended=f"at least {gap:.0f} s apart", suggested_fix="Use a different effect or remove one.", viewer_impact=0.2, signature=sha(str(path.name), round(close[0][0])), confidence=85.0, ctx=ctx))

    @staticmethod
    def _overlap(speech: list[tuple[float, float]], a: float, b: float) -> float:
        return sum(max(0.0, min(b, y) - max(a, x)) for x, y in speech)

    @staticmethod
    def _scene_for(ctx: QCContext, t: float) -> str | None:
        s = ctx.scene_at(t)
        return s.id if s else None


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 1)


def _t(t: float | None) -> str:
    if t is None:
        return "?"
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"

