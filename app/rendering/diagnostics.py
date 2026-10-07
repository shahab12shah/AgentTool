"""RenderDiagnostics: the preflight check. Finds problems *before* anything is rendered and says what to do about each."""

from __future__ import annotations

from pathlib import Path

from app.core.constants import MIN_CLIP_DURATION, TIME_EPSILON
from app.core.exceptions import AppError
from app.presentation.animation import problems as animation_problems
from app.rendering import presets as P
from app.rendering.errors import EncoderUnavailableError, RenderError
from app.rendering.ffmpeg_service import FFmpegService
from app.rendering.fonts import FontResolver
from app.rendering.models import PreflightItem, PreflightReport, RenderSnapshot, disk_free, has_audio_clips
from app.rendering.output import OutputManager
from app.rendering.planner import EncoderSelector, FINAL_CHUNK_SECONDS, make_plan, plan_chunks
from app.rendering.probe import MediaProbeService
from app.rendering.sources import proxy_usable
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_MEDIA, KIND_TEXT
from app.timeline.track import TrackKind

VIDEO_MBPS_1080 = {"draft": 1.5, "standard": 6.0, "high": 12.0, "maximum": 24.0, "custom": 12.0}


def validate_timeline(snap: RenderSnapshot) -> tuple[list[str], list[str]]:
    """Structural problems a render cannot work around (errors) and oddities it can (warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    for t in snap.tracks:
        last_end, clips = -1.0, sorted(t.clips, key=lambda c: c.timeline_start)
        for c in clips:
            tag = f"{t.name} · {c.kind} {c.id[-6:]}"
            if c.duration < MIN_CLIP_DURATION - TIME_EPSILON:
                errors.append(f"{tag}: the clip is too short ({c.duration:.3f}s).")
            if c.timeline_start < -TIME_EPSILON:
                errors.append(f"{tag}: the clip starts before 0:00.")
            if c.speed <= 0:
                errors.append(f"{tag}: the speed must be greater than zero.")
            if c.scale <= 0:
                errors.append(f"{tag}: the scale must be greater than zero.")
            if not 0.0 <= c.opacity <= 1.0:
                errors.append(f"{tag}: the opacity must be between 0 and 1.")
            if t.kind is not TrackKind.AUDIO and c.timeline_start < last_end - TIME_EPSILON and c.kind == KIND_MEDIA:
                warnings.append(f"{tag}: overlaps the previous clip on the same track; the later clip is drawn on top.")
            last_end = max(last_end, c.timeline_end)
            for k in c.keyframes:
                for p in k.problems(c.duration):
                    errors.append(f"{tag}: invalid keyframe — {p}.")
            for p in animation_problems(c.animation, c.duration) if c.animation else []:
                errors.append(f"{tag}: invalid animation — {p}.")
            if c.kind == KIND_MEDIA and c.asset_id not in snap.assets:
                errors.append(f"{tag}: refers to an asset that is not in the project.")
            if c.transition and float(c.transition.get("duration", 0)) > c.duration + 1e-6:
                warnings.append(f"{tag}: the transition is longer than the clip and will be shortened.")
    return errors, warnings


class RenderDiagnostics:
    def __init__(self, ffmpeg: FFmpegService, probe: MediaProbeService, fonts: FontResolver, selector: EncoderSelector, outputs: OutputManager) -> None:
        self.ff, self.probe, self.fonts, self.selector, self.outputs = ffmpeg, probe, fonts, selector, outputs

    def run(self, snap: RenderSnapshot, output_dir: Path | None = None, explicit_output: Path | None = None, cache_dir: Path | None = None,
            allow_proxy_assets: set[str] | None = None, check_hardware: bool = True) -> PreflightReport:
        rep = PreflightReport()
        s = snap.settings
        allow = allow_proxy_assets or set()

        def add(i: str, label: str, status: str, message: str = "", fix: str = "", assets: list[str] | None = None, details: list[str] | None = None) -> PreflightItem:
            it = PreflightItem(i, label, status, message or label, fix, assets or [], details or [])
            rep.items.append(it)
            return it

        # ---- FFmpeg
        ok, msg = self.ff.detect()
        if not ok:
            add("ffmpeg", "FFmpeg", "error", msg, "Install FFmpeg or set its location in Settings.")
            return rep
        v = self.ff.version()
        add("ffmpeg", "FFmpeg", "ok", f"FFmpeg {v.major}.{v.minor} ready" if v.parsed else "FFmpeg ready")
        lacking = self.ff.capabilities().has_filters("ass", "overlay", "geq", "alphamerge", "amix", "alimiter")
        if lacking:
            add("filters", "FFmpeg features", "error", "FFmpeg lacks required features: " + ", ".join(lacking), "Install a full FFmpeg build (with libass).")
        # ---- timeline
        errs, warns = validate_timeline(snap)
        if snap.duration <= 0:
            errs.insert(0, "The timeline is empty — add at least one clip.")
        if errs:
            add("timeline", "Timeline", "error", errs[0] if len(errs) == 1 else f"Timeline has {len(errs)} problems", "Open the timeline and fix the listed clips.", details=errs)
        elif warns:
            add("timeline", "Timeline", "warning", f"Timeline valid ({len(warns)} note(s))", details=warns)
        else:
            add("timeline", "Timeline", "ok", "Timeline valid")
        # ---- voice-over
        vo = snap.assets.get(snap.voice_asset_id) if snap.voice_asset_id else None
        if vo is None:
            add("voice", "Voice-over", "warning", "No voice-over in this project — the video will have no narration", "Import a voice-over on the Voice page.")
        elif not Path(vo.path).is_file():
            add("voice", "Voice-over", "error", f"Voice-over file is missing ({vo.name})", "Locate the voice-over file.", [vo.id])
        else:
            add("voice", "Voice-over", "ok", "Voice-over found")
        # ---- media
        visual_ids = sorted({c.asset_id for t, c in snap.clips() if t.kind is not TrackKind.AUDIO and c.kind == KIND_MEDIA and c.asset_id})
        audio_ids = sorted({c.asset_id for t, c in snap.clips() if t.kind is TrackKind.AUDIO and c.asset_id})
        missing_visual = [a for a in visual_ids if a in snap.assets and not Path(snap.assets[a].path).is_file()]
        missing_audio = [a for a in audio_ids if a in snap.assets and not Path(snap.assets[a].path).is_file() and a != snap.voice_asset_id]
        proxy_only = [a for a in missing_visual if proxy_usable(snap.proxies.get(a)) and a not in allow]
        rep.missing_assets = missing_visual + missing_audio
        rep.proxy_only_assets = proxy_only
        usable_via_proxy = [a for a in missing_visual if a in allow and proxy_usable(snap.proxies.get(a))]
        hard_missing = [a for a in missing_visual if a not in usable_via_proxy]
        found = len(visual_ids) - len(hard_missing)
        if hard_missing:
            names = [f"{snap.assets[a].name}" + ("  (proxy available)" if a in proxy_only else "") for a in hard_missing]
            add("visuals", "Visuals", "error", f"{len(hard_missing)} asset(s) missing — {found}/{len(visual_ids)} visuals found",
                "Locate the missing files (Fix Issues) or choose to use the proxy.", hard_missing, names)
        else:
            add("visuals", "Visuals", "ok", f"{found}/{len(visual_ids)} visuals found" + (f" ({len(usable_via_proxy)} via proxy)" if usable_via_proxy else ""))
        if usable_via_proxy:
            add("proxy_final", "Proxy media", "warning", f"{len(usable_via_proxy)} asset(s) will be exported from proxy media because you chose to", assets=usable_via_proxy)
        elif s.use_proxies:
            add("proxy_final", "Proxy media", "warning", "This export uses proxy media (lower quality) because you chose to")
        # ---- unreadable / unsupported media
        bad: list[str] = []
        for a in visual_ids + audio_ids:
            ref = snap.assets.get(a)
            if ref is None or a in missing_visual or a in missing_audio:
                continue
            info, err = self.probe.try_probe(Path(ref.path))
            if info is None:
                bad.append(f"{ref.name}: {err}")
                rep.missing_assets.append(a)
            else:
                ref.width, ref.height, ref.fps, ref.has_alpha, ref.rotation = (info.width, info.height, info.fps, info.has_alpha, info.rotation) if ref.type != "audio" else (ref.width, ref.height, ref.fps, ref.has_alpha, ref.rotation)
        if bad:
            add("media", "Media readable", "error", f"{len(bad)} file(s) cannot be read or are not supported", "Replace or re-encode these files.", details=bad)
        # ---- captions
        caps = [c for t, c in snap.clips() if c.kind == KIND_CAPTION]
        if caps:
            broken = [c for c in caps if not (c.text or {}).get("words") and not (c.text or {}).get("text")]
            if broken:
                add("captions", "Captions", "error", f"{len(broken)} caption(s) have no text", "Regenerate or edit those captions.")
            else:
                add("captions", "Captions", "ok", f"Captions valid ({len(caps)})" if snap.caption_settings.enabled else f"{len(caps)} captions (turned off — not exported)")
        else:
            add("captions", "Captions", "ok", "No captions in this timeline")
        # ---- fonts
        subs = self._fonts(snap)
        if subs:
            add("fonts", "Fonts", "warning", f"{len(subs)} font(s) are not installed; a fallback will be used", "Install the font or pick another one.", details=subs)
        else:
            add("fonts", "Fonts", "ok", "Fonts available")
        # ---- audio
        if missing_audio:
            add("audio", "Audio", "error", f"{len(missing_audio)} audio file(s) missing", "Locate the music / sound-effect files.", missing_audio, [snap.assets[a].name for a in missing_audio])
        elif has_audio_clips(snap):
            add("audio", "Audio", "ok", "Audio valid")
        else:
            add("audio", "Audio", "warning", "No audible audio tracks — the video will be silent")
        # ---- output settings
        resolved = None
        try:
            problems = P.compatibility_problems(s)
            if problems:
                raise RenderError(problems[0], kind="invalid_settings")
            resolved = self.selector.resolve(s, snap) if check_hardware else self.selector.resolve(type(s)(**{**s.to_dict(), "hardware_acceleration": "cpu"}), snap)
            details = [f"{resolved.width}×{resolved.height} @ {resolved.fps} fps", f"{P.CODEC_LABELS[resolved.video_codec]} ({resolved.encoder}{', hardware' if resolved.hardware else ''})",
                       f"{P.AUDIO_LABELS[resolved.audio_codec]} {resolved.audio_bitrate_kbps} kbps · {resolved.sample_rate} Hz", f"quality: {resolved.quality}"] + resolved.notes
            add("settings", "Output settings", "ok", "Output settings valid", details=details)
        except EncoderUnavailableError as exc:
            add("settings", "Output settings", "error", exc.user_message, "Choose a supported codec: " + (", ".join(exc.alternatives) or "H.264"), details=[exc.possible_issue])
        except RenderError as exc:
            fix = "Use Auto or CPU instead (the CPU encoder is available)." if exc.can_fallback_cpu else (exc.possible_issue or "Change the export settings.")
            add("settings", "Output settings", "error", exc.user_message, fix, details=[exc.possible_issue] if exc.possible_issue else [])
        # ---- output path + disk
        target_dir = explicit_output.parent if explicit_output else (output_dir or self.outputs.default_dir)
        problem = self.outputs.validate_target(target_dir)
        if explicit_output is not None and explicit_output.is_dir():
            problem = f"“{explicit_output}” is a folder, not a file."
        if problem:
            add("output", "Output location", "error", problem, "Choose another output location.")
        else:
            add("output", "Output location", "ok", "Output location valid")
        if resolved is not None:
            rep.required_bytes = self._estimate_bytes(snap, resolved)
            rep.available_bytes = disk_free(target_dir)
            cache_free = disk_free(cache_dir) if cache_dir else -1
            tight = (0 <= rep.available_bytes < rep.required_bytes) or (0 <= cache_free < rep.required_bytes * 0.6)
            if tight:
                add("disk", "Disk space", "error", f"Not enough disk space. Required: approximately {_gb(rep.required_bytes)}. Available: approximately {_gb(min(x for x in (rep.available_bytes, cache_free) if x >= 0))}.",
                    "Free some space or choose another drive.")
            else:
                add("disk", "Disk space", "ok", f"Disk space sufficient (about {_gb(rep.required_bytes)} needed)")
            chunks = plan_chunks(snap, resolved.fps, FINAL_CHUNK_SECONDS)
            rep.plan = make_plan(snap, resolved, chunks)
            rep.plan.uses_proxy_assets = usable_via_proxy
        return rep

    # ------------------------------------------------------------------ helpers
    def _fonts(self, snap: RenderSnapshot) -> list[str]:
        used: set[tuple[str, bool]] = set()
        from app.captions.styles import effective_style, style_for

        for t, c in snap.clips():
            if c.kind == KIND_TEXT and c.text:
                used.add((str(c.text.get("font", "Sans")), c.text.get("emphasis") in ("BOLD_TEXT", "PUNCH_TEXT", "NUMBER_CARD", "WARNING_TEXT")))
            elif c.kind == KIND_CAPTION and c.text:
                st = effective_style(style_for(snap.caption_styles, c.text.get("style_id", snap.caption_settings.style_id)), snap.caption_settings, c.text.get("style_overrides"))
                used.add((st.font, st.weight == "bold"))
        out = []
        for fam, bold in sorted(used):
            m = self.fonts.resolve(fam, bold)
            if m.substituted:
                out.append(m.reason or f"“{fam}” is not installed.")
        return sorted(set(out))

    @staticmethod
    def _estimate_bytes(snap: RenderSnapshot, r: P.ResolvedOutput) -> int:
        """A rough size: video at the quality's typical rate for the pixel count, audio, and room for the cached sections and the mix."""
        px = (r.width * r.height) / (1920.0 * 1080.0)
        mbps = (r.bitrate_kbps / 1000.0) if r.bitrate_kbps else VIDEO_MBPS_1080.get(r.quality, 12.0) * px * (r.fps / 30.0) ** 0.5
        video = mbps * 1e6 / 8 * snap.duration
        audio = r.audio_bitrate_kbps * 1000 / 8 * snap.duration
        mix = snap.duration * r.sample_rate * 2 * 4 * 0.5 if has_audio_clips(snap) else 0
        return int((video * 2.1 + audio + mix) * 1.1)


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 else f"{max(1, n // 1024 ** 2)} MB"


_ = (AppError, KIND_GRAPHIC)
