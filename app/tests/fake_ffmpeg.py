#!/usr/bin/env python3
"""A TEST DOUBLE for ffmpeg: it runs the real ffmpeg, but can be told (through a control file) to fail part way.

    FAKE_FFMPEG_REAL   path of the real ffmpeg
    FAKE_FFMPEG_MODE   path of a file whose text is the current mode:
                         ok                 behave exactly like ffmpeg
                         fail_chunk:0.72    kill the video-section render at 72% of its output and exit 1
                         fail_audio         fail the audio mix immediately
                         fail_mux           fail the final encode/mux step immediately
                         hang_chunk         produce no progress at all (for stall handling)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def main() -> int:
    real = os.environ["FAKE_FFMPEG_REAL"]
    args = sys.argv[1:]
    try:
        mode = open(os.environ["FAKE_FFMPEG_MODE"], encoding="utf-8").read().strip()
    except OSError:
        mode = "ok"
    is_chunk = "-filter_complex_script" in args and "-frames:v" in args
    is_audio = "-c:a" in args and "flac" in args and "-filter_complex_script" in args
    is_mux = "-c:v" in args and "copy" in args and args.count("-i") >= 1 and "-filter_complex_script" not in args and "concat" not in args
    if is_audio and mode == "fail_audio" or is_mux and mode == "fail_mux":
        sys.stderr.write("[fake] Conversion failed! (injected failure)\n")
        return 1
    if is_chunk and mode.startswith("fail_chunk"):
        frac = float(mode.split(":")[1])
        frames = int(args[args.index("-frames:v") + 1])
        fps = float(args[args.index("-r") + 1])
        total = frames / fps
        p = subprocess.Popen([real, *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("out_time_us="):
                try:
                    if int(line.split("=")[1]) / 1e6 >= frac * total:
                        p.kill()
                        sys.stderr.write("[fake] Error while encoding: Conversion failed! (injected failure)\n")
                        return 1
                except ValueError:
                    pass
        return p.wait()
    if is_chunk and mode == "hang_chunk":
        time.sleep(3600)
        return 1
    os.execv(real, [real, *args])
    return 0


if __name__ == "__main__":
    sys.exit(main())
