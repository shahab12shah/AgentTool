"""FFmpegCommandBuilder: compiled timeline -> FFmpeg argument lists and filter-graph scripts.

* Commands are ``list[str]`` (never a shell string). Inputs are absolute paths passed as separate arguments, so spaces, parentheses,
  apostrophes and Unicode need no quoting on any OS.
* Filter graphs are written to a script file (``-filter_complex_script``) in the render's working directory, and FFmpeg runs *in that
  directory*: subtitle and font paths are relative names, so no path ever needs filtergraph escaping (Windows ``C:\\...`` included).
* Nothing here executes anything.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath

from app.audio.mix import AudioMixService
from app.audio.processing import build_chain
from app.rendering.compiler import AudioPlan, CompiledChunk, TimelineCompiler, VideoLayer
from app.rendering.expressions import fnum
from app.rendering.models import RenderSnapshot
from app.rendering.presets import ResolvedOutput

PI = math.pi
GRAPH_NAME = "graph.txt"


@dataclass
class BuiltCommand:
    args: list[str]
    graph: str = ""
    files: dict[str, str] = field(default_factory=dict)  # relative file name -> text written into the working directory before running
    expected_seconds: float = 0.0  # length of the output (progress denominator)
    cache_key: str = ""
    output: Path | None = None
    inputs: list[str] = field(default_factory=list)  # input files (for fingerprints / existence checks)
    description: str = ""

    def printable(self) -> str:
        from app.rendering.ffmpeg_service import redact

        return " ".join(_q(redact(a)) for a in self.args)


def _q(a: str) -> str:
    return f'"{a}"' if re.search(r"\s|[()'\"]", a) else a


def to_geq(expr: str) -> str:
    """The same expression for the ``geq`` filter, whose time variable is ``T`` (a standalone ``t`` only: ``lt(`` and ``clip(`` stay)."""
    return re.sub(r"(?<![A-Za-z0-9_])t(?![A-Za-z0-9_(])", "T", expr)


def even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


class FFmpegCommandBuilder:
    def __init__(self, ffmpeg: str, out: ResolvedOutput, snapshot: RenderSnapshot, compiler: TimelineCompiler, fps_mode_flag: str = "-fps_mode") -> None:
        self.ff, self.out, self.s, self.c = ffmpeg, out, snapshot, compiler
        self.fps = out.fps
        self.k = out.width / float(snapshot.canvas_w)  # canvas pixels -> output pixels
        self.fps_mode_flag = fps_mode_flag

    # ------------------------------------------------------------------ video chunk
    def video_chunk(self, cc: CompiledChunk, out_path: Path, encode_args: list[str] | None = None) -> BuiltCommand:
        ch = cc.chunk
        dur = ch.frames / self.fps
        args: list[str] = [self.ff, "-hide_banner", "-nostats", "-v", "warning", "-progress", "pipe:1", "-stats_period", "0.25"]
        graph: list[str] = []
        inputs: list[str] = []
        files: dict[str, str] = {}
        graph.append(f"color=c=black:s={self.out.width}x{self.out.height}:r={self.fps}:d={fnum(dur + 1.0 / self.fps)},format=yuv420p,setpts=PTS-STARTPTS+{fnum(ch.start)}/TB[base0]")
        cur, n_in, ass_n = "base0", 0, 0
        for step in cc.steps:
            if step.kind == "layer":
                L = step.layer
                assert L is not None
                in_args, chain, in_label = self._layer(L, n_in, graph)
                args += in_args
                inputs.append(L.path)
                label = f"L{n_in}"
                graph.append(f"[{in_label}]{chain}[{label}]")
                nxt = f"c{n_in}"
                x = f"(main_w-overlay_w)/2+({L.x.expr})*{fnum(self.k)}"
                y = f"(main_h-overlay_h)/2+({L.y.expr})*{fnum(self.k)}"
                en = f"between(t,{fnum(L.t0 - 1e-4)},{fnum(L.t1 - 0.5 / self.fps)})"
                graph.append(f"[{cur}][{label}]overlay=x='{x}':y='{y}':enable='{en}':eof_action=pass:format=auto[{nxt}]")
                cur = nxt
                n_in += 1
            else:
                name = f"overlay_{ass_n}.ass"
                files[name] = self.c.ass.render(ch.start, ch.end)
                nxt = f"a{ass_n}"
                graph.append(f"[{cur}]ass=filename={name}:fontsdir=fonts[{nxt}]")
                cur = nxt
                ass_n += 1
        graph.append(f"[{cur}]setpts=PTS-STARTPTS,format=yuv420p[vout]")
        text = ";\n".join(graph) + "\n"
        files[GRAPH_NAME] = text
        args += ["-filter_complex_script", GRAPH_NAME, "-map", "[vout]", "-an", "-frames:v", str(ch.frames), self.fps_mode_flag, "cfr", "-r", str(self.fps)]
        args += list(encode_args if encode_args is not None else self.out.video_args())
        args += [*self.out.segment_args(), "-y", str(out_path)]
        cmd = BuiltCommand(args, text, files, dur, "", out_path, inputs, f"video chunk {ch.index + 1} ({ch.start:.2f}s–{ch.end:.2f}s)")
        cmd.cache_key = self._key(cmd, files)
        return cmd

    def _key(self, cmd: BuiltCommand, files: dict[str, str]) -> str:
        """Everything that decides the pixels of a chunk: graph, overlay scripts, encoder arguments, and the identity of every input file."""
        h = hashlib.sha1()
        h.update(repr([a for a in cmd.args if a != str(cmd.output)]).encode())
        for name in sorted(files):
            h.update(name.encode())
            h.update(files[name].encode())
        for p in cmd.inputs:
            try:
                st = Path(p).stat()
                h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
            except OSError:
                h.update(f"{p}|missing".encode())
        for m in self.c.ass.fonts:
            for f in m.files:
                try:
                    st = Path(f).stat()
                    h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode())
                except OSError:
                    pass
        return h.hexdigest()[:24]

    # ------------------------------------------------------------------ one layer
    def _layer(self, L: VideoLayer, idx: int, graph: list[str]) -> tuple[list[str], str, str]:
        """(input arguments, filter chain, input label). Mask streams are appended to ``graph`` as separate segments."""
        fps = self.fps
        n_frames = max(1, int(round((L.t1 - L.t0) * fps)))
        dur = n_frames / fps
        chain: list[str] = []
        if L.kind == "image":
            in_args = ["-loop", "1", "-framerate", str(fps), "-t", fnum(dur + 2.0 / fps), "-i", L.path]
            chain.append("setpts=PTS-STARTPTS")
        elif L.kind == "hold":
            in_args = ["-ss", fnum(L.seek), "-t", "0.5", "-i", L.path]
            chain += ["trim=end_frame=1", "setpts=PTS-STARTPTS", f"tpad=stop_mode=clone:stop_duration={fnum(dur + 0.2)}", f"fps=fps={fps}:round=near"]
        else:
            in_args = ["-ss", fnum(L.seek), "-t", fnum(L.src_span + L.tpad), "-i", L.path]
            chain.append("setpts=PTS-STARTPTS" if abs(L.speed - 1.0) < 1e-9 else f"setpts=(PTS-STARTPTS)/{fnum(L.speed)}")
            chain.append(f"fps=fps={fps}:round=near")
            if L.tpad > 0:
                chain.append(f"tpad=stop_mode=clone:stop_duration={fnum(L.tpad)}")
        chain.append(f"trim=end_frame={n_frames}")
        chain.append(f"setpts=PTS-STARTPTS+{fnum(L.t0)}/TB")
        if L.crop:
            cx, cy, cw, ch = L.crop
            chain.append(f"crop=w='trunc(iw*{fnum(cw)}/2)*2':h='trunc(ih*{fnum(ch)}/2)*2':x='trunc(iw*{fnum(cx)}/2)*2':y='trunc(ih*{fnum(cy)}/2)*2'")
        if L.blur > 0:
            chain.append(f"gblur=sigma={fnum(L.blur * self.k)}")
        const_op = L.opacity.value if L.opacity.is_const else None
        needs_alpha = L.needs_mask or L.rotated or L.has_alpha or L.fade_in or L.fade_out or (const_op is not None and const_op < 0.999)
        if needs_alpha:
            chain.append("format=yuva420p")
        in_label = f"{idx}:v"
        if L.needs_mask:
            mw, mh = (480, 270) if not L.reveal.is_const else (16, 16)
            terms = [f"255*clip({to_geq(L.opacity.expr)},0,1)"]
            if not L.reveal.is_const:
                terms.append(f"lt(X,W*clip({to_geq(L.reveal.expr)},0,1))")
            sw, sh = even(L.stream_w), even(L.stream_h)
            graph.append(f"color=c=white:s={mw}x{mh}:r={fps}:d={fnum(dur + 1.0 / fps)},setpts=PTS-STARTPTS+{fnum(L.t0)}/TB,format=gray,"
                         f"geq=lum='{'*'.join(terms)}',scale={sw}:{sh}:flags=bilinear,format=gray[M{idx}]")
            graph.append(f"[{in_label}]{','.join(chain)}[S{idx}]")
            if L.has_alpha:
                graph.append(f"[S{idx}]split[Sa{idx}][Sb{idx}];[Sb{idx}]alphaextract[Ax{idx}];[Ax{idx}][M{idx}]blend=all_mode=multiply[Am{idx}];[Sa{idx}][Am{idx}]alphamerge[Q{idx}]")
            else:
                graph.append(f"[S{idx}][M{idx}]alphamerge[Q{idx}]")
            in_label, chain = f"Q{idx}", []
        # rotation (fixed square canvas so the size never changes mid-stream)
        sw_c, sh_c = float(L.stream_w), float(L.stream_h)
        rd = even(math.ceil(math.hypot(sw_c, sh_c)))
        if L.rotated:
            a = f"{fnum((L.rot.value or 0.0) * PI / 180)}" if L.rot.is_const else f"(({L.rot.expr})*PI/180)"
            chain.append(f"rotate=a='{a}':c=none:ow={rd}:oh={rd}")
        # geometry: canvas px per source px = base * scale(t) ; output px = canvas px * k
        ratio = L.box_w / max(1.0, sw_c)  # original display size per stream pixel (1 unless a proxy is smaller)
        f_expr = f"{fnum(L.base * self.k)}*({L.scale.expr})" if not L.scale.is_const else None
        if L.rotated:
            w_expr = (f"{rd}*{fnum(ratio)}*{f_expr}" if f_expr else fnum(rd * ratio * L.base * self.k * (L.scale.value or 1.0)))
            h_expr = w_expr
            static_w = static_h = even(rd * ratio * L.base * self.k * (L.scale.value or 1.0)) if not f_expr else 0
        else:
            w_expr = f"{fnum(L.box_w)}*{f_expr}" if f_expr else ""
            h_expr = f"{fnum(L.box_h)}*{f_expr}" if f_expr else ""
            static_w = even(L.box_w * L.base * self.k * (L.scale.value or 1.0)) if not f_expr else 0
            static_h = even(L.box_h * L.base * self.k * (L.scale.value or 1.0)) if not f_expr else 0
        if f_expr:
            chain.append(f"scale=w='trunc(max(2,min(16384,{w_expr}))/2)*2':h='trunc(max(2,min(16384,{h_expr}))/2)*2':eval=frame:flags=bicubic")
        else:
            chain.append(f"scale=w={min(static_w, 16384)}:h={min(static_h, 16384)}:flags=bicubic")
        if L.fade_in:
            chain.append(f"fade=t=in:st={fnum(L.fade_in[0])}:d={fnum(L.fade_in[1])}:alpha=1")
        if L.kind == "hold" and L.fade_out:
            chain.append(f"fade=t=out:st={fnum(L.fade_out[0])}:d={fnum(L.fade_out[1])}:alpha=1")
        if const_op is not None and const_op < 0.999 and not L.needs_mask:
            chain.append(f"lutyuv=a='val*{fnum(max(0.0, const_op))}'")
        return in_args, ",".join(chain) if chain else "null", in_label

    # ------------------------------------------------------------------ audio
    def audio_mix(self, plan: AudioPlan, out_path: Path) -> BuiltCommand | None:
        """One filter graph for the whole timeline: gain, keyframed volume (ducking), fades, pan, delay, mix, limiter. Returns ``None`` if nothing is audible."""
        if not plan.items:
            return None
        sr = self.out.sample_rate
        args = [self.ff, "-hide_banner", "-nostats", "-v", "warning", "-progress", "pipe:1", "-stats_period", "0.25"]
        graph, labels, inputs = [], [], []
        voice_chain = build_chain(self.s.audio_processing, None)  # "" unless the user enabled voice processing (applied to the voice-over only, never to the file)
        for i, it in enumerate(plan.items):
            args += ["-ss", fnum(it.source_in), "-t", fnum(it.duration * it.speed + 0.1), "-i", it.path]
            inputs.append(it.path)
            parts = [f"[{i}:a]asetpts=PTS-STARTPTS", f"aresample={sr}", "aformat=sample_fmts=fltp:channel_layouts=stereo"]
            if voice_chain and it.is_voice:
                parts.append(voice_chain)
            parts += _atempo(it.speed)
            parts.append(f"asetpts=N/SR/TB")  # exact timestamps: the volume expression uses ``t`` and must never see an unknown time
            parts.append(f"volume='{AudioMixService.volume_expr(it.gain, it.keyframes, 0.0)}':eval=frame")
            if it.fade_in > 0:
                parts.append(f"afade=t=in:st=0:d={fnum(max(0.01, it.fade_in))}")
            if it.fade_out > 0 and it.duration > it.fade_out:
                parts.append(f"afade=t=out:st={fnum(it.duration - it.fade_out)}:d={fnum(it.fade_out)}")
            elif it.fade_out > 0:
                parts.append(f"afade=t=out:st=0:d={fnum(it.duration)}")
            if abs(it.pan) > 1e-6:
                left, right = min(1.0, 1.0 - it.pan), min(1.0, 1.0 + it.pan)
                parts.append(f"pan=stereo|c0={fnum(left)}*c0|c1={fnum(right)}*c1")
            parts.append(f"atrim=end={fnum(it.duration)}")
            ms = int(round(it.start * 1000))
            parts.append(f"adelay={ms}|{ms}")
            graph.append(",".join(parts) + f"[m{i}]")
            labels.append(f"[m{i}]")
        total = max(plan.duration, 0.1)
        graph.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0:duration=longest,alimiter=limit=0.97:level=0,"
                     f"apad=whole_dur={fnum(total)},atrim=end={fnum(total)},aformat=sample_fmts=s32:sample_rates={sr}:channel_layouts=stereo[aout]")
        text = ";\n".join(graph) + "\n"
        args += ["-filter_complex_script", GRAPH_NAME, "-map", "[aout]", "-c:a", "flac", "-bits_per_raw_sample", "24", "-ar", str(sr), "-y", str(out_path)]
        cmd = BuiltCommand(args, text, {GRAPH_NAME: text}, total, "", out_path, inputs, "audio mix")
        cmd.cache_key = hashlib.sha1((text + repr([(p, _stat(p)) for p in inputs])).encode()).hexdigest()[:24]
        return cmd

    # ------------------------------------------------------------------ concat / mux
    def concat(self, chunk_paths: list[Path], list_name: str, out_path: Path, total_seconds: float) -> BuiltCommand:
        lines = "".join("file '" + PurePath(p).as_posix().replace("'", "'\\''") + "'\n" for p in chunk_paths)  # forward slashes: a backslash is an escape in the concat list
        args = [self.ff, "-hide_banner", "-nostats", "-v", "warning", "-progress", "pipe:1", "-stats_period", "0.25", "-f", "concat", "-safe", "0", "-i", list_name,
                "-c", "copy", *([ "-f", "mpegts"] if self.out.segment_ext == "ts" else []), "-y", str(out_path)]
        return BuiltCommand(args, "", {list_name: lines}, total_seconds, "", out_path, [str(p) for p in chunk_paths], "join video chunks")

    def mux(self, video: Path, audio: Path | None, out_path: Path, total_seconds: float) -> BuiltCommand:
        args = [self.ff, "-hide_banner", "-nostats", "-v", "warning", "-progress", "pipe:1", "-stats_period", "0.25", "-i", str(video)]
        if audio is not None:
            args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *self.out.audio_args()]
        else:
            args += ["-map", "0:v:0", "-c:v", "copy", "-an"]
        if self.out.video_codec == "h265" and self.out.container == "mp4":
            args += ["-tag:v", "hvc1"]
        args += [*self.out.container_args(), "-t", fnum(total_seconds), "-y", str(out_path)]
        return BuiltCommand(args, "", {}, total_seconds, "", out_path, [str(video)] + ([str(audio)] if audio else []), "encode audio and write the final file")


def _stat(p: str) -> str:
    try:
        st = Path(p).stat()
        return f"{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return "missing"


def _atempo(speed: float) -> list[str]:
    """``atempo`` only accepts 0.5–2.0 per instance; chain instances for other speeds."""
    if abs(speed - 1.0) < 1e-6:
        return []
    out, s = [], speed
    while s > 2.0 + 1e-9:
        out.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        out.append("atempo=0.5")
        s /= 0.5
    out.append(f"atempo={fnum(s)}")
    return out
