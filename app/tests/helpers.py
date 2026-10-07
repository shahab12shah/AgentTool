"""Test doubles for Phase 2. ScriptedProvider is a TEST DOUBLE: it is not a speech recogniser."""

from __future__ import annotations

from pathlib import Path

from app.transcription.provider import RawTranscription, RawWord, TranscriptionProvider


def timed_words(text: str, wps: float = 2.6, sentence_pause: float = 0.7, clause_pause: float = 0.15,
                start: float = 0.0, confidence: float = 0.95, punctuation: bool = True) -> list[RawWord]:
    """Deterministic word timings: duration grows with word length, pauses after punctuation."""
    out, t = [], start
    for tok in text.split():
        bare = tok.strip(".,?!;:")
        dur = max(0.18, len(bare) * 0.06 + 0.1) * (2.6 / wps)
        out.append(RawWord(tok if punctuation else bare.lower(), t, t + dur, confidence))
        t += dur
        if tok.endswith((".", "?", "!")):
            t += sentence_pause
        elif tok.endswith((",", ";", ":")):
            t += clause_pause
    return out


class ScriptedProvider(TranscriptionProvider):
    """Returns pre-defined words; optionally fails on demand. Used only by tests."""

    name = "scripted-test"
    kind = "local"

    def __init__(self, words: list[RawWord] | str, fail: str | None = None, punctuation: bool = True) -> None:
        self.words = timed_words(words, punctuation=punctuation) if isinstance(words, str) else words
        self.fail = fail
        self.calls = 0

    def is_available(self):
        return True, ""

    def config_key(self) -> str:
        return "scripted"

    def transcribe(self, audio_path: Path, language=None, progress=None, should_cancel=None) -> RawTranscription:
        from app.core.exceptions import TranscriptionError

        self.calls += 1
        if progress:
            progress(0.5, "scripted")
        if self.fail:
            raise TranscriptionError(self.fail)
        return RawTranscription(list(self.words), "en", "scripted")


BLOCKS = [
    "Silver demand has changed dramatically. Solar manufacturers are now consuming more of the metal.",
    "Now let's talk about the IRS. The IRS sent a notice to affected taxpayers in the United States.",
    "The notice says the deadline is April 15th. Missing it can trigger a penalty of 5% per month.",
    "Meanwhile, Tesla announced a new battery factory in Texas. Elon Musk said production will start in 2027.",
    "How does a solar panel actually work? Sunlight hits the silicon cells and the silver wiring collects the current.",
    "Gold behaved very differently last year. The price of gold climbed 20% while bonds stayed flat.",
    "Moving on to housing. Mortgage rates in Canada are higher than in Germany.",
    "Researchers at NASA published a study about lunar dust. The report describes how fine particles damage equipment.",
    "Let's talk about retirement. Retirees aged 65+ must take required distributions from their IRA.",
    "In Tokyo, a new robotics company opened a factory. Robots assemble phones and computers there.",
    "Bitcoin crashed last month. Investors panicked and sold billions of dollars in a single day.",
    "Next, consider inflation. Inflation hit 3.5% in 2027 according to the Federal Reserve.",
    "The Supreme Court ruled on a major tax law. The ruling changes how deductions work for families.",
    "Now let's talk about batteries. Lithium batteries store energy for electric vehicles and solar power.",
    "Amazon expanded its warehouses across Europe. Delivery drones now carry packages to customers.",
    "In conclusion, diversification matters. Remember that nobody can predict the future.",
]
NARRATION = " ".join(BLOCKS)


def make_audio(path: Path, seconds: float) -> Path:
    """A real WAV file of the given length (content is irrelevant to the scripted provider)."""
    import subprocess

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}", str(path)], check=True)
    return path


def run_transcription(ws, provider: "ScriptedProvider", force: bool = False):
    """Transcribe the open project's voice-over with a test provider and wait for the job."""
    ws.transcripts.registry.register(provider)
    job = ws.transcripts.transcribe(force=force, provider_name=provider.name)
    assert ws.jobs.wait_idle(60)
    return job


def run_scenes(ws, **kw):
    job = ws.scenes.analyze(**kw)
    assert ws.jobs.wait_idle(60)
    return job
