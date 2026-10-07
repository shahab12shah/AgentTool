from __future__ import annotations

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from app.core.exceptions import InvalidTranscriptError, JobCancelled, TranscriptionError
from app.media.asset import Asset, AssetType, SourceType
from app.tests.conftest import import_and_wait, needs_ffmpeg
from app.tests.helpers import NARRATION, ScriptedProvider, make_audio, run_transcription, timed_words
from app.transcription.models import AudioInfo, ProviderInfo, Sentence, Transcript, Word
from app.transcription.provider import RawTranscription, RawWord
from app.transcription.providers.api_provider import OpenAICompatibleProvider
from app.transcription.providers.whisper_provider import FasterWhisperProvider
from app.transcription.sentences import by_pauses, by_punctuation, build_sentences
from app.transcription.service import TranscriptionEngine, normalize_words
from app.transcription.status import TranscriptStatus, transcript_status

AUDIO = Asset("media_00001", AssetType.AUDIO, SourceType.USER_MEDIA, "a.wav", "a.wav", duration=30.0, sample_rate=44100, channels=1, content_hash="abc")


def make_transcript(text="Silver demand has changed. It rose fast.", **kw) -> Transcript:
    return TranscriptionEngine().run(ScriptedProvider(text), Path("a.wav"), AUDIO, "en", **kw).transcript


# ------------------------------------------------------------------ word / sentence model
def test_word_timestamps_are_never_rounded_and_survive_serialisation():
    raw = [RawWord("Silver", 4.2137, 4.6921, 0.98), RawWord("demand", 4.7003, 5.1119, 0.91)]
    words, repaired, dropped = normalize_words(raw)
    assert (words[0].start, words[0].end, words[0].confidence) == (4.2137, 4.6921, 0.98) and not repaired and not dropped
    tr = Transcript("t", AudioInfo("a", 6.0, 44100, 1), words, [Sentence("s0", "Silver demand", 4.2137, 5.1119, ["w_000000", "w_000001"], 0.945)], ProviderInfo("x"))
    again = Transcript.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert again == tr and again.words[0].start == 4.2137


def test_transcript_contains_required_audio_metadata_words_and_sentences():
    tr = make_transcript()
    assert (tr.audio.file_id, tr.audio.duration, tr.audio.sample_rate, tr.audio.channels) == ("media_00001", 30.0, 44100, 1)
    w = tr.words[0]
    assert w.word_id and w.text == "Silver" and w.start < w.end and w.confidence == 0.95
    assert len(tr.sentences) == 2
    s = tr.sentences[0]
    assert s.sentence_id and s.text == "Silver demand has changed." and s.word_ids == [x.word_id for x in tr.words[:4]]
    assert s.start == tr.words[0].start and s.end == tr.words[3].end and s.confidence == pytest.approx(0.95)
    tr.validate()


def test_sentence_lookup_by_time():
    tr = make_transcript()
    mid = (tr.words[1].start + tr.words[1].end) / 2
    assert tr.sentence_at(mid) is tr.sentences[0] and tr.word_at(mid) is tr.words[1]
    assert tr.sentence_at(-1) is None
    assert tr.sentence_at(tr.sentences[1].start + 0.01) is tr.sentences[1]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: setattr(t.words[0], "end", t.words[0].start - 1),          # end before start
        lambda t: setattr(t.words[0], "start", -0.5),                          # negative
        lambda t: setattr(t.words[2], "start", t.words[1].start - 0.5),        # out of order
        lambda t: setattr(t.words[1], "word_id", t.words[0].word_id),          # duplicate id
        lambda t: setattr(t.words[0], "confidence", 1.5),                      # confidence out of range
        lambda t: setattr(t.words[0], "start", float("nan")),                  # non-finite
        lambda t: t.sentences[0].word_ids.append("nope"),                      # unknown word reference
        lambda t: setattr(t.sentences[1], "start", t.sentences[0].end - 1.0),  # overlapping sentences
        lambda t: setattr(t.words[0], "text", "  "),                           # empty word
    ],
)
def test_invalid_transcripts_are_rejected(mutate):
    tr = make_transcript()
    mutate(tr)
    with pytest.raises(InvalidTranscriptError):
        tr.validate()


def test_provider_quirks_are_repaired_or_dropped_not_trusted():
    raw = [RawWord("b", 2.0, 1.0, 0.5), RawWord("a", 0.5, 0.9, 7.0), RawWord("", 1.0, 2.0), RawWord("x", float("inf"), 3.0), RawWord("c", -1.0, 0.2)]
    words, repaired, dropped = normalize_words(raw)
    assert dropped == 2 and repaired >= 3
    assert sorted(w.text for w in words) == ["a", "b", "c"]  # repaired words kept, empty/non-finite dropped
    assert all(w.end >= w.start >= 0 for w in words) and all(w.confidence is None or 0 <= w.confidence <= 1 for w in words)
    assert [w.start for w in words] == sorted(w.start for w in words)


def test_engine_rejects_empty_speech_and_unavailable_provider():
    class Silent(ScriptedProvider):
        def transcribe(self, *a, **k):
            return RawTranscription([], "en", "x")

    with pytest.raises(TranscriptionError, match="No speech"):
        TranscriptionEngine().run(Silent(""), Path("a.wav"), AUDIO)

    class Broken(ScriptedProvider):
        def is_available(self):
            return False, "model missing"

    with pytest.raises(TranscriptionError, match="model missing"):
        TranscriptionEngine().run(Broken("x y z"), Path("a.wav"), AUDIO)


# ------------------------------------------------------------------ sentence strategies
def test_sentence_strategies():
    words, _, _ = normalize_words(timed_words("One two three four. Five six seven eight nine."))
    assert [len(s.word_ids) for s in by_punctuation(words)] == [4, 5]
    assert build_sentences(words)[1] == "punctuation"
    bare, _, _ = normalize_words(timed_words("One two three four. Five six seven eight nine.", punctuation=False))
    assert [len(s.word_ids) for s in by_pauses(bare)] == [4, 5] and build_sentences(bare)[1] == "pause"
    # script-guided: boundary follows the aligned script sentence index
    guide = {w.word_id: (0 if i < 3 else 1) for i, w in enumerate(bare)}
    sents, strategy = build_sentences(bare, guide)
    assert strategy == "script-guided" and [len(s.word_ids) for s in sents] == [3, 6]


def test_long_unpunctuated_run_is_capped():
    words, _, _ = normalize_words([RawWord(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(130)])
    assert all(len(s.word_ids) <= 41 for s in by_pauses(words))


# ------------------------------------------------------------------ providers
def test_faster_whisper_provider_maps_words_with_injected_model():
    class Seg:
        def __init__(self, words, end):
            self.words, self.end = words, end

    class W:
        def __init__(self, word, start, end, p):
            self.word, self.start, self.end, self.probability = word, start, end, p

    class Info:
        duration, language = 4.0, "en"

    class FakeModel:
        def transcribe(self, path, **kw):
            assert kw["word_timestamps"] is True
            return iter([Seg([W(" Silver", 0.1, 0.5, 0.99), W(" demand", 0.55, 1.0, 0.8)], 2.0), Seg([W(" rose.", 2.1, 2.6, 0.7)], 4.0)]), Info()

    p = FasterWhisperProvider(model=lambda: "tiny", model_factory=lambda name: FakeModel())
    seen = []
    out = p.transcribe(Path("a.wav"), "en", progress=lambda f, m: seen.append(f))
    assert [(w.text, w.start, w.end, w.confidence) for w in out.words] == [("Silver", 0.1, 0.5, 0.99), ("demand", 0.55, 1.0, 0.8), ("rose.", 2.1, 2.6, 0.7)]
    assert seen and seen[-1] <= 1.0
    with pytest.raises(JobCancelled):
        p.transcribe(Path("a.wav"), "en", should_cancel=lambda: True)
    ok, why = FasterWhisperProvider(model=lambda: "", model_factory=lambda n: None).is_available()
    assert not ok and "model" in why.lower()


def test_faster_whisper_load_failure_is_a_friendly_error():
    def boom(name):
        raise OSError("no such model")

    p = FasterWhisperProvider(model=lambda: "nope", model_factory=boom)
    with pytest.raises(TranscriptionError, match="could not be loaded"):
        p.transcribe(Path("a.wav"))


class _Api(BaseHTTPRequestHandler):
    status = 200
    seen: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _Api.seen = {"auth": self.headers.get("Authorization"), "path": self.path, "body": body}
        payload = json.dumps({"language": "en", "text": "hi there", "words": [
            {"word": "hi", "start": 0.12, "end": 0.4}, {"word": "there", "start": 0.45, "end": 0.9}]}).encode()
        self.send_response(_Api.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload if _Api.status == 200 else b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture
def api_server():
    server = HTTPServer(("127.0.0.1", 0), _Api)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    _Api.status = 200
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@needs_ffmpeg
def test_api_provider_against_a_local_mock_server(api_server, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_STT_KEY", "sk-secret-123")
    audio = make_audio(tmp_path / "a.wav", 1.0)
    p = OpenAICompatibleProvider(base_url=lambda: api_server, model=lambda: "whisper-1", key_env=lambda: "TEST_STT_KEY")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    out = p.transcribe(audio, "en")
    assert [(w.text, w.start, w.end) for w in out.words] == [("hi", 0.12, 0.4), ("there", 0.45, 0.9)]
    assert _Api.seen["auth"] == "Bearer sk-secret-123" and _Api.seen["path"].endswith("/audio/transcriptions")
    assert b"timestamp_granularities[]" in _Api.seen["body"] and b"verbose_json" in _Api.seen["body"]
    _Api.status = 401
    with pytest.raises(TranscriptionError, match="rejected the key"):
        p.transcribe(audio, "en")
    monkeypatch.delenv("TEST_STT_KEY")
    ok, why = p.is_available()
    assert not ok and "TEST_STT_KEY" in why and "sk-" not in why


def test_api_provider_never_logs_the_key(api_server, tmp_path, monkeypatch):
    import logging

    from app.logging.logger import setup_logging

    logs = tmp_path / "logs"
    setup_logging(logs, level=logging.DEBUG, console=False)
    monkeypatch.setenv("TEST_STT_KEY", "sk-very-secret")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....")
    _Api.status = 401  # force the error path, which is where secrets tend to leak
    with pytest.raises(TranscriptionError):
        OpenAICompatibleProvider(base_url=lambda: api_server, key_env=lambda: "TEST_STT_KEY").transcribe(audio)
    for h in logging.getLogger("agenttool").handlers:
        h.flush()
    text = "".join(f.read_text() for f in logs.glob("*.log")) + "".join(
        f.read_text() for f in logs.glob("*.log.*"))
    assert "sk-very-secret" not in text


@needs_ffmpeg
def test_pocketsphinx_provider_runs_and_returns_well_formed_timings(tmp_path):
    """Real offline engine on real audio. Asserts structure/timing validity only — NOT recognition accuracy."""
    pytest.importorskip("pocketsphinx")
    import subprocess

    wav = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=3:c=pink:a=0.2", str(wav)], check=True)
    from app.transcription.providers.pocketsphinx_provider import PocketSphinxProvider

    p = PocketSphinxProvider()
    assert p.is_available()[0] and p.accuracy_note
    out = p.transcribe(wav, "en")  # noise may legitimately produce zero or a few words
    words, _, _ = normalize_words(out.words)
    assert all(0 <= w.start <= w.end <= 3.2 for w in words)
    with pytest.raises(TranscriptionError, match="English"):
        p.transcribe(wav, "de")


# ------------------------------------------------------------------ service level: status, cache, change detection
@needs_ffmpeg
def test_transcription_job_status_cache_and_voiceover_replacement(voice_ws, tmp_path):
    ws = voice_ws
    p = ws.provider
    assert ws.transcripts.status() is TranscriptStatus.NOT_STARTED
    run_transcription(ws, p)
    tr = ws.project.transcription.transcript
    assert ws.transcripts.status() is TranscriptStatus.COMPLETE and p.calls == 1
    assert tr.audio.file_id == ws.project.voice_over.asset_id and tr.audio.content_hash
    assert tr.sentence_strategy == "script-guided" and len(tr.sentences) == 32
    assert ws.project.script_alignment is not None

    # up to date: no job, no provider call
    assert run_transcription(ws, p) is None and p.calls == 1

    # script edit alone: transcript stays COMPLETE, alignment becomes outdated (cheap re-align, no re-transcription)
    from app.transcription.status import AlignmentStatus, alignment_status

    ws.set_script(NARRATION + " One more sentence.")
    assert ws.transcripts.status() is TranscriptStatus.COMPLETE and alignment_status(ws.project) is AlignmentStatus.OUTDATED
    ws.transcripts.realign()
    assert ws.jobs.wait_idle(30) and alignment_status(ws.project) is AlignmentStatus.CURRENT and p.calls == 1
    assert ws.project.script.text.endswith("One more sentence.")  # the script is never rewritten by alignment

    # replacing the voice-over marks the transcript OUTDATED but keeps it (never silently deleted)
    other = make_audio(tmp_path / "other.wav", 40.0)
    ws.media.import_voice_over(other)
    assert ws.jobs.wait_idle(30)
    assert ws.transcripts.status() is TranscriptStatus.OUTDATED
    assert ws.project.transcription.transcript is tr
    run_transcription(ws, ScriptedProvider("A brand new narration about copper mining. It happens underground."))
    assert ws.transcripts.status() is TranscriptStatus.COMPLETE
    assert ws.project.transcription.transcript.words[0].text == "A"


@needs_ffmpeg
def test_transcript_cache_is_reused_for_the_same_audio_and_provider(voice_ws):
    ws = voice_ws
    p = ws.provider
    run_transcription(ws, p)
    cache_files = list((ws.project.root / "cache" / "transcripts").glob("*.json"))
    assert len(cache_files) == 1
    old = ws.project.transcription.transcript
    run_transcription(ws, p, force=True)  # forced re-run still honours nothing but the provider
    assert p.calls == 2
    # now simulate "reopen": cached file satisfies a fresh run without calling the provider
    ws.project.transcription = type(ws.project.transcription)()  # forget the in-project transcript
    before = p.calls
    job = ws.transcripts.transcribe(provider_name=p.name)
    assert ws.jobs.wait_idle(30)
    assert p.calls == before and ws.project.transcription.transcript is not None
    assert job.message.startswith("Using cached") or job.progress == 100


@needs_ffmpeg
def test_failed_transcription_is_reported_and_retry_works(voice_ws):
    ws = voice_ws
    failing = ScriptedProvider(NARRATION, fail="Provider exploded.")
    run_transcription(ws, failing)
    assert ws.transcripts.status() is TranscriptStatus.FAILED
    assert ws.project.transcription.last_error == "Provider exploded."
    run_transcription(ws, ScriptedProvider(NARRATION))  # retry with a working provider
    assert ws.transcripts.status() is TranscriptStatus.COMPLETE and ws.project.transcription.last_error is None


@needs_ffmpeg
def test_transcribe_requires_voiceover_and_provider(project_ws):
    ws = project_ws
    with pytest.raises(TranscriptionError, match="voice-over"):
        ws.transcripts.transcribe()
    with pytest.raises(TranscriptionError, match="Unknown"):
        ws.transcripts.resolve_provider("nonexistent")


@needs_ffmpeg
def test_transcript_survives_save_and_reopen(voice_ws):
    ws = voice_ws
    run_transcription(ws, ws.provider)
    before = ws.project.transcription.transcript.to_dict()
    root = ws.project.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert ws.project.transcription.transcript.to_dict() == before
    assert ws.transcripts.status() is TranscriptStatus.COMPLETE
    assert ws.project.script_alignment is not None
