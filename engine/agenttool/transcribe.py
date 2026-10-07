"""Word-level timing for the voiceover.

Uses faster-whisper when installed (best). Otherwise falls back to spreading the
script words across the detected speech region of the audio.
"""
import re

from .util import ffprobe_duration, run, progress


def split_sentences(script):
    text = re.sub(r"\s+", " ", script.strip())
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def script_words(script):
    return re.findall(r"\S+", script)


def _whisper_words(audio, model_size="base"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    model = WhisperModel(model_size, compute_type="int8")
    segments, _ = model.transcribe(audio, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    return words or None


def _speech_region(audio, total):
    """Find leading/trailing silence with ffmpeg silencedetect."""
    p = run(["ffmpeg", "-hide_banner", "-i", audio, "-af",
             "silencedetect=noise=-38dB:d=0.35", "-f", "null", "-"], check=False)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    s, e = 0.0, total
    if ends and starts and starts[0] <= 0.05:
        s = ends[0]
    if starts and (not ends or starts[-1] > (ends[-1] if ends else 0)) and starts[-1] > total * 0.5:
        e = starts[-1]
    return s, max(e, s + 1.0)


def _even_words(words, s, e):
    weights = [max(len(w), 2) for w in words]
    tot = float(sum(weights))
    t, out = s, []
    for w, wt in zip(words, weights):
        d = (e - s) * wt / tot
        out.append({"w": w, "start": t, "end": t + d})
        t += d
    return out


def timed_script_words(script, audio):
    """Return (words, total_duration, method) where words carry the SCRIPT spelling."""
    total = ffprobe_duration(audio)
    sw = script_words(script)
    if not sw:
        raise ValueError("script is empty")
    progress("transcribe", 5, "Listening to the voiceover")
    ww = None
    try:
        ww = _whisper_words(audio)
    except Exception as exc:  # model download failure etc.
        progress("transcribe", 8, f"Whisper unavailable ({exc}); using even timing")
    if ww:
        n = len(ww)
        out = []
        for i, w in enumerate(sw):
            j = round(i * (n - 1) / max(len(sw) - 1, 1))
            src = ww[j]
            out.append({"w": w, "start": src["start"], "end": src["end"]})
        # make timings monotonic and non-overlapping
        for a, b in zip(out, out[1:]):
            if b["start"] < a["start"]:
                b["start"] = a["start"]
            if a["end"] > b["start"] and b["start"] > a["start"]:
                a["end"] = b["start"]
        return out, total, "whisper"
    s, e = _speech_region(audio, total)
    return _even_words(sw, s, e), total, "even"
