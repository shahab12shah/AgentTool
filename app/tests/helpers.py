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


# ====================================================================== Phase 3 research test doubles
import json as _json
import threading as _threading
import time as _time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.core.exceptions import ProviderError
from app.media.asset import SourceType
from app.research.models import Acquisition, Candidate, ResearchQuery
from app.research.providers.base import SearchContext, SourceProvider, blank_candidate


class FakeProvider(SourceProvider):
    """Deterministic provider for tests (NOT a real source). Serves a fixed catalogue and can fail on demand."""

    name = "fake"
    label = "Fake provider (tests)"

    def __init__(self, items=None, source_types=(SourceType.STOCK_VIDEO, SourceType.STOCK_IMAGE), name="fake",
                 fail: str | None = None, fail_if=None, available=(True, ""), delay: float = 0.0, **kw):
        super().__init__(**kw)
        self.name = name
        self.source_types = tuple(source_types)
        self.items = list(items or [])
        self.fail, self.fail_if = fail, fail_if
        self.available = available
        self.delay = delay
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = _threading.Lock()

    def is_available(self):
        return self.available

    def config_key(self):
        return f"fake:{self.name}:{len(self.items)}"

    def search(self, query: ResearchQuery, source_type: SourceType, limit: int, ctx: SearchContext):
        with self._lock:
            self.calls.append(query.text)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                _time.sleep(self.delay)
            if self.fail or (self.fail_if and self.fail_if(query)):
                raise ProviderError(self.fail or "provider failed for this query")
            out = []
            for spec in self.items:
                if spec.get("source_type", self.source_types[0]) != source_type:
                    continue
                c = blank_candidate(query, source_type, spec.get("kind", "VIDEO" if source_type in (SourceType.STOCK_VIDEO, SourceType.YOUTUBE, SourceType.WEB_VIDEO) else "IMAGE"), self.name)
                c.title, c.description = spec["title"], spec.get("description", "")
                c.tags = list(spec.get("tags", []))
                c.duration, c.width, c.height = spec.get("duration", 8.0 if c.kind == "VIDEO" else None), spec.get("width", 1920), spec.get("height", 1080)
                c.provider_id = spec.get("id", spec["title"])
                c.source_reference = spec.get("url", f"https://example.test/{c.provider_id.replace(' ', '-')}")
                c.media_url = spec.get("media_url", "")
                c.local_path = spec.get("local_path", "")
                c.thumbnail_path = spec.get("thumbnail_path", "")
                c.acquisition = spec.get("acquisition", Acquisition.REFERENCE_ONLY)
                if "evidence_kind" in spec:
                    c.evidence_kind = spec["evidence_kind"]
                out.append(c)
            return out[:limit]
        finally:
            with self._lock:
                self.active -= 1

    def fetch_thumbnail(self, candidate, dest, http):
        return False

    def acquire(self, candidate, dest_dir, ctx):
        from app.core.exceptions import AcquisitionError

        if candidate.local_path and Path(candidate.local_path).is_file():
            return Path(candidate.local_path)
        raise AcquisitionError("fake provider has no file")


class MockWeb:
    """A local HTTP server standing in for remote APIs. Routes map a path prefix to a callable(handler, body)->(status, type, bytes)."""

    def __init__(self):
        self.routes: dict[str, object] = {}
        self.requests: list[dict] = []
        handler = self._make_handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        _threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _make_handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def _serve(self, body=b""):
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "method": self.command, "body": body})
                for prefix, fn in sorted(outer.routes.items(), key=lambda kv: -len(kv[0])):
                    if self.path.split("?")[0].startswith(prefix):
                        status, ctype, payload = fn(self, body)
                        self.send_response(status)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                self._serve()

            def do_POST(self):
                self._serve(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            def log_message(self, *a):
                pass

        return H

    def json(self, prefix: str, payload, status: int = 200):
        data = _json.dumps(payload).encode()
        self.routes[prefix] = lambda h, b: (status, "application/json", data)

    def file(self, prefix: str, path: Path, ctype: str = "application/octet-stream"):
        data = Path(path).read_bytes()
        self.routes[prefix] = lambda h, b: (200, ctype, data)

    def html(self, prefix: str, html: str):
        self.routes[prefix] = lambda h, b: (200, "text/html", html.encode())

    def count(self, prefix: str) -> int:
        return sum(1 for r in self.requests if r["path"].startswith(prefix))

    def close(self):
        self.server.shutdown()


def allow_local_http(ws) -> None:
    """Providers refuse private hosts by default; tests talk to 127.0.0.1 so opt in explicitly."""
    for p in ws.research.registry.all():
        p.http.allow_private = True
        if hasattr(p, "allow_private"):
            p.allow_private = True


def make_image(path: Path, kind: str = "testsrc", size: str = "320x180") -> Path:
    import subprocess

    src = {"testsrc": "testsrc=size={s}", "testsrc2": "testsrc2=size={s}", "red": "color=c=red:size={s}", "mandel": "mandelbrot=size={s}",
           "gradient": "gradients=size={s}:seed=7"}[kind].format(s=size)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src, "-frames:v", "1", str(path)], check=True)
    return path


def make_video(path: Path, seconds: float = 4.0, kind: str = "testsrc") -> Path:
    import subprocess

    src = {"testsrc": "testsrc=size=640x360:rate=24", "testsrc2": "testsrc2=size=640x360:rate=24", "mandel": "mandelbrot=size=640x360:rate=24"}[kind]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"{src}:duration={seconds}", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def solar_brief(**kw):
    from app.research.models import EvidenceLevel, ResearchBrief

    base = dict(scene_id="scene_014", topic="Solar silver demand", primary_subject="solar installations", secondary_subject="silver",
                action="manufacturing", context="solar energy industry", visual_type="PROCESS", narration="Silver demand from solar installations continues to rise.",
                entities=["silver", "solar panels"], entity_types={"silver": "FINANCIAL_INSTRUMENT", "solar panels": "TECHNOLOGY"},
                claims=["Solar installations are increasing silver demand."], claim_types=["FACT"], evidence_level=EvidenceLevel.POSSIBLE,
                preferred_sources=["STOCK_VIDEO", "WEB_IMAGE", "SCREENSHOT", "AI_GENERATED"], avoid=["Generic silver coins unrelated to solar"],
                avoid_terms=["coin", "coins", "bullion", "jewelry"], video_topic="Silver", scene_start=74.2, scene_end=80.4)
    base.update(kw)
    return ResearchBrief(**base)


def cand(title, description="", source_type=SourceType.STOCK_VIDEO, kind=None, tags=(), duration=8.0, width=1920, height=1080, scene_id="scene_014",
         cid=None, **kw):
    c = Candidate(candidate_id=cid or f"candidate_{abs(hash((title, source_type.value))) % 99999:05d}", scene_id=scene_id, source_type=source_type,
                  kind=kind or ("VIDEO" if source_type in (SourceType.STOCK_VIDEO, SourceType.YOUTUBE, SourceType.WEB_VIDEO) else "IMAGE"),
                  title=title, description=description, tags=list(tags), width=width, height=height, provider="fake",
                  duration=duration if (kind or "VIDEO") == "VIDEO" and source_type in (SourceType.STOCK_VIDEO, SourceType.YOUTUBE, SourceType.WEB_VIDEO) else None,
                  provider_id=title)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ====================================================================== Phase 5 audio helpers
def write_speech_wav(path, words, total: float, sr: int = 16000, amp: float = 0.4, freq: float = 220.0, noise: float = 0.0, loud: dict | None = None):
    """A WAV whose energy follows word timings: tone bursts during words, silence (or faint noise) between them. ``loud`` maps word index -> gain."""
    import wave

    import numpy as np

    n = int(total * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(1)
    if noise:
        sig += (rng.standard_normal(n) * noise).astype(np.float32)
    for i, w in enumerate(words):
        a, b = int(w.start * sr), min(n, int(w.end * sr))
        g = (loud or {}).get(i, 1.0)
        env = np.ones(b - a, dtype=np.float32)
        ramp = min(200, (b - a) // 4)
        if ramp > 0:
            env[:ramp], env[-ramp:] = np.linspace(0, 1, ramp), np.linspace(1, 0, ramp)
        sig[a:b] += (amp * g * np.sin(2 * np.pi * freq * t[a:b]) * env).astype(np.float32)
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path


def write_tone_wav(path, seconds: float, amp: float = 0.3, freq: float = 440.0, sr: int = 16000):
    import wave

    import numpy as np

    t = np.arange(int(seconds * sr)) / sr
    pcm = (np.clip(amp * np.sin(2 * np.pi * freq * t), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path
