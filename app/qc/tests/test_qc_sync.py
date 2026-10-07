"""SyncChecker: caption drift / early, visual late / early / overstay, important visuals, emphasis misses, cuts inside key phrases, tolerances, locks, scene-local reuse.

The synthetic project has two scenes and a transcript with exact word times (``SENTENCES``), so every drift in these tests is an exact number of milliseconds.
"""

from __future__ import annotations

from app.analysis.models import Claim, ClaimType, NumberKind, NumericMention, VisualIntent, VisualType
from app.qc.issue_model import CheckerState
from app.qc.qc_engine import PreviousState, QCEngine
from app.qc.settings import QCSettings
from app.qc.severity import Severity
from app.qc.sync_checker import SyncChecker, sync_summary
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, new_project, qc_ctx, run_checker
from app.timeline.clip import Clip
from app.transcription.models import AudioInfo, ProviderInfo, Sentence, Transcript, TranscriptionState, Word

# (scene index, [(text, start, end)]): one sentence each
SENTENCES = [
    (0, [("Silver", 0.5, 0.9), ("prices", 0.9, 1.3), ("rose", 1.3, 1.6), ("42%", 1.6, 2.3), ("last", 2.4, 2.7), ("week.", 2.7, 3.1)]),
    (0, [("Penalties", 4.0, 4.6), ("apply", 4.6, 5.0), ("immediately.", 5.0, 5.8)]),
    (0, [("Experts", 6.5, 7.0), ("expect", 7.0, 7.5), ("further", 7.5, 8.0), ("gains", 8.0, 8.5), ("soon.", 8.5, 9.2)]),
    (1, [("Demand", 10.5, 11.0), ("keeps", 11.0, 11.4), ("growing", 11.4, 12.0), ("across", 12.0, 12.5), ("Asia.", 12.5, 13.2)]),
    (1, [("Prices", 14.0, 14.5), ("may", 14.5, 14.8), ("fall", 14.8, 15.3), ("later.", 15.3, 16.0)]),
    (1, [("Prices", 17.0, 17.4), ("may", 17.4, 17.7), ("rise", 17.7, 18.2), ("soon.", 18.2, 18.8)]),
]
W_NUMBER = 3  # "42%"
W_PENALTIES = 6
W_DEMAND = 14
W_PRICES_2 = 19  # first "Prices may fall later."
W_PRICES_3 = 23  # second "Prices may ..." (the same words again, in the same scene)


class World:
    def __init__(self, tmp_path, *, transcript: bool = True):
        self.p = p = new_project(tmp_path, seconds=20)
        self.a = add_asset(p, "city.mp4", "video", duration=30)
        self.s1 = add_scene(p, 0, 10, " ".join(t for sc, ws in SENTENCES if sc == 0 for t, _a, _b in ws), importance=0.8)
        self.s2 = add_scene(p, 10, 20, " ".join(t for sc, ws in SENTENCES if sc == 1 for t, _a, _b in ws))
        self.words: list[Word] = []
        self.sentences: list[Sentence] = []
        for k, (_sc, ws) in enumerate(SENTENCES):
            ids = []
            for t, a, b in ws:
                w = Word(f"w_{len(self.words):06d}", t, a, b, 0.95)
                self.words.append(w)
                ids.append(w.word_id)
            self.sentences.append(Sentence(f"sent_{k:04d}", " ".join(x[0] for x in ws), ws[0][1], ws[-1][2], ids, 0.95))
        self.s1.sentence_ids = [s.sentence_id for s in self.sentences[:3]]
        self.s2.sentence_ids = [s.sentence_id for s in self.sentences[3:]]
        self.s1.numbers = [NumericMention("42%", NumberKind.PERCENTAGE, 42.0, "", False, [self.words[W_NUMBER].word_id])]
        self.s1.claims = [Claim("sent_0000_c0", "Silver prices rose 42% last week.", ClaimType.NUMBER, "sent_0000", True)]
        if transcript:
            dur = p.voice_over.duration
            p.transcription = TranscriptionState(Transcript("tr_sync", AudioInfo(p.voice_over.asset_id or "", dur, 48000, 2, None, "voice.wav"), self.words, self.sentences,
                                                            ProviderInfo("test"), "punctuation"))

    # ---- builders
    def visuals(self, edges=((0.0, 10.0), (10.0, 20.0)), by="AI"):
        """One V1 picture per (start, end) pair, the first pair on scene 1, the rest on scene 2."""
        out = []
        for i, (a, b) in enumerate(edges):
            out.append(add_clip(self.p, "track_v1", self.a, a, b - a, scene=self.s1 if (a + b) / 2 < 10 else self.s2, created_by=by, slot=f"visual:{i}"))
        return out

    def caption(self, first: int, last: int, shift: float = 0.0, *, by: str = "AI", text: str | None = None, emphasis=None, own_shift: float = 0.0, hold: float = 0.12) -> Clip:
        ws = self.words[first:last + 1]
        scene = self.s1 if ws[0].start < 10 else self.s2
        words = [{"word_id": x.word_id, "text": x.text, "start": round(x.start + own_shift, 3), "end": round(x.end + own_shift, 3)} for x in ws]
        start, end = ws[0].start + shift, ws[-1].end + hold + shift
        seg = {"caption_id": f"{scene.id}_c0", "scene_id": scene.id, "start": start, "end": end, "text": text or " ".join(x.text for x in ws), "lines": [text or " ".join(x.text for x in ws)],
               "words": words, "emphasis": emphasis or []}
        return add_clip(self.p, "track_v6", None, round(start, 3), round(end - start, 3), scene=scene, kind="caption", created_by=by, slot="caption:0", text=seg)

    def text(self, start: float, content: str, variant: str, *, emphasis: str = "BOLD_TEXT", counter=None, preset: str = "pop", dur: float = 2.0, delay: float = 0.0, by: str = "AI",
             scene=None, derived: bool = False) -> Clip:
        body = {"text_id": "t", "content": content, "start": start, "duration": dur, "variant": variant, "emphasis": emphasis, "counter": counter, "derived": derived}
        return add_clip(self.p, "track_v5", None, start, dur, scene=scene or self.s1, kind="text", created_by=by, slot="text:0", text=body,
                        animation={"in": {"preset": preset, "duration": 0.3, "delay": delay}})

    def highlight(self, start: float, dur: float = 2.0, by: str = "AI") -> Clip:
        return add_clip(self.p, "track_v4", None, start, dur, scene=self.s1, kind="graphic", created_by=by, slot="evidence:0:highlight",
                        effects={"highlight": {"region": [0.1, 0.2, 0.5, 0.3], "style": "box", "darken_surround": True}})


def run(w: World, settings: QCSettings | None = None, **kw):
    return run_checker(SyncChecker(), qc_ctx(w.p, settings, **kw))


# ------------------------------------------------------------------ correct synchronization produces nothing
def test_correct_synchronization_produces_no_issues(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5)
    w.caption(6, 8)
    w.caption(14, 18)
    w.text(1.45, "42%", "NUMBER", emphasis="NUMBER_CARD")  # the engine's 0.15 s lead on the spoken figure
    w.text(3.85, "PENALTIES", "WARNING", emphasis="WARNING_TEXT")
    w.highlight(1.5)  # while the claim is being spoken
    out = run(w)
    assert out.issues == [] and out.metrics["captions_checked"] == 3 and out.metrics["max_caption_drift_ms"] == 0.0
    assert out.metrics[f"scene.{w.s1.id}.first_visual_offset_ms"] == -500.0 and out.notes and "3 caption(s)" in out.notes[-1]


def test_checker_declares_its_cache_key_accurately(tmp_path):
    chk = SyncChecker()
    assert (chk.id, chk.scene_local, chk.expensive) == ("sync", True, False) and set(chk.domains) == {"timeline", "scenes", "transcript", "captions"}
    w = World(tmp_path)
    base = chk.input_hash(qc_ctx(w.p))
    s = QCSettings()
    s.caption.max_cps = 30  # not read by the checker
    assert chk.input_hash(qc_ctx(w.p, s)) == base
    s2 = QCSettings()
    s2.max_caption_shift_seconds = 0.2  # decides whether a retime is "safe": it is part of the answer
    assert chk.input_hash(qc_ctx(w.p, s2)) != base
    s3 = QCSettings()
    s3.sync.moderate_ms = 300
    assert chk.input_hash(qc_ctx(w.p, s3)) != base


# ------------------------------------------------------------------ caption drift (lag) and early
def test_caption_lagging_420_ms_is_a_warning_with_a_safe_retime(tmp_path):
    w = World(tmp_path)
    w.visuals()
    cap = w.caption(0, 5, 0.42)
    out = run(w)
    [i] = find(out, "sync.caption_drift")
    assert codes(out) == ["sync.caption_drift"] and i.severity is Severity.WARNING and i.scene_id == w.s1.id and i.timeline_item_id == cap.id and i.track_id == "track_v6"
    assert i.fix.kind == "caption.retime" and i.fix.params["new_start"] == 0.5 and i.fix.params["delta"] == -0.42 and i.fix.params["new_end"] == round(cap.timeline_end - 0.42, 3)
    assert i.fix.safe and i.auto_fix_available and i.auto_fix_safe and i.confidence == 100.0  # |delta| <= max_caption_shift_seconds (0.6) and the permission is "auto"
    assert "420 ms" in i.title and "420 ms" in i.description and i.current_value.startswith("+420 ms") and i.why_it_matters and i.recommended_value and i.suggested_fix
    assert out.metrics[f"scene.{w.s1.id}.max_caption_drift_ms"] == 420.0 and out.metrics[f"scene.{w.s1.id}.captions_checked"] == 1


def test_caption_lagging_900_ms_is_an_error_that_needs_confirmation(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, 0.9)
    [i] = run(w).issues
    assert i.code == "sync.caption_drift" and i.severity is Severity.ERROR
    assert not i.fix.safe and i.fix.needs_confirmation and i.auto_fix_available and not i.auto_fix_safe  # beyond the "small shift" limit: a click is needed


def test_drift_bands_follow_the_tolerances(tmp_path):
    expected = {0.05: None, 0.10: Severity.NOTICE, 0.24: Severity.NOTICE, 0.25: Severity.WARNING, 0.49: Severity.WARNING, 0.50: Severity.ERROR, 1.5: Severity.ERROR}
    for shift, sev in expected.items():
        w = World(tmp_path / f"d{int(shift * 100)}")
        w.visuals()
        w.caption(0, 5, shift)
        got = run(w).issues
        assert (got[0].severity if got else None) is sev, shift


def test_caption_appearing_before_the_voice_is_caption_early(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, -0.3)
    [i] = run(w).issues
    assert i.code == "sync.caption_early" and i.severity is Severity.WARNING and "300 ms" in i.title and i.fix.params["delta"] == 0.3 and i.fix.safe and i.current_value.startswith("-300 ms")
    assert i.fix.params["new_start"] == 0.5


def test_caption_slightly_early_is_within_the_early_tolerance(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, -0.12)  # more than minor (100) but under caption_early_ms (150)
    assert run(w).issues == []
    w2 = World(tmp_path / "b")
    w2.visuals()
    w2.caption(0, 5, -0.2)
    assert codes(run(w2)) == ["sync.caption_early"] and run(w2).issues[0].severity is Severity.NOTICE


def test_custom_tolerances_are_respected(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, 0.42)
    loose = QCSettings()
    loose.sync.minor_ms, loose.sync.moderate_ms, loose.sync.major_ms = 500.0, 700.0, 900.0
    assert run(w, loose).issues == []
    strict = QCSettings()
    strict.sync.minor_ms, strict.sync.moderate_ms, strict.sync.major_ms = 50.0, 100.0, 300.0
    assert run(w, strict).issues[0].severity is Severity.ERROR
    w2 = World(tmp_path / "e")
    w2.visuals()
    w2.caption(0, 5, -0.2)
    tight = QCSettings()
    tight.sync.caption_early_ms = 400.0
    assert run(w2, tight).issues == []  # the early tolerance is its own setting


def test_the_retime_moves_the_window_and_never_the_words(tmp_path):
    w = World(tmp_path)
    w.visuals()
    cap = w.caption(0, 5, 0.42)
    before = cap.snapshot()
    run(w)
    assert cap.to_dict() == before.to_dict()  # the checker never mutates the project
    # a lagging caption whose window is also shorter than the speech keeps the last word on screen after the shift
    w2 = World(tmp_path / "b")
    w2.visuals()
    c2 = w2.caption(0, 5, 0.42, hold=-0.2)
    [i] = run(w2).issues
    assert i.fix.params["new_end"] >= w2.words[5].end and c2.timeline_end - 0.42 < w2.words[5].end


# ------------------------------------------------------------------ matching the caption to the transcript
def test_a_repeated_phrase_is_matched_to_the_nearest_occurrence(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(W_PRICES_2, W_PRICES_2 + 3)  # "Prices may fall later." at 14.0
    w.caption(W_PRICES_3, W_PRICES_3 + 3, 0.42)  # "Prices may rise soon." at 17.0 (the same first words), 420 ms late
    out = run(w)
    [i] = out.issues
    assert i.code == "sync.caption_drift" and i.fix.params["new_start"] == 17.0 and out.metrics["captions_checked"] == 2


def test_an_edited_caption_text_falls_back_to_its_own_word_timing_with_less_confidence(tmp_path):
    w = World(tmp_path)
    w.visuals()
    # the user rewrote the text: it no longer matches the transcript; the stored words say it was spoken at 0.5 s but the window starts 0.7 s later
    w.caption(0, 5, 0.7, text="Silver gained forty two percent")
    cap = w.p.timeline.get_track("track_v6").clips[0]
    cap.text["words"] = [{"word_id": f"edit_{k}", "text": t, "start": 0.5 + k * 0.2, "end": 0.7 + k * 0.2} for k, t in enumerate(cap.text["text"].split())]
    [i] = run(w).issues
    assert i.code == "sync.caption_drift" and i.confidence == 55.0 and i.metrics["match"] == "own" and "could not be matched" in i.description
    assert i.fix is not None and not i.fix.safe and i.fix.needs_confirmation  # never applied silently on an uncertain match
    cap.timeline_start = 0.5  # window back on the stored word timing: clean
    assert run(w).issues == []


def test_caption_without_a_transcript_is_compared_with_its_own_words(tmp_path):
    w = World(tmp_path, transcript=False)
    w.visuals()
    w.caption(0, 5, 0.42)  # the words were generated at the spoken time; the window drifted
    out = run(w)
    assert out.notes[0].startswith("No transcript") and codes(out) == ["sync.caption_drift"] and out.issues[0].confidence == 55.0


def test_a_caption_with_nothing_to_compare_is_counted_not_flagged(tmp_path):
    w = World(tmp_path, transcript=False)
    w.visuals()
    add_clip(w.p, "track_v6", None, 1.0, 1.0, scene=w.s1, kind="caption", created_by="AI", text={"caption_id": "x", "text": "", "words": []})
    out = run(w)
    assert out.issues == [] and out.metrics["captions_unmeasured"] == 1


# ------------------------------------------------------------------ ownership: a user's caption is reported, never auto-fixed
def test_user_owned_caption_is_reported_with_auto_fix_disabled(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, 0.42, by="USER")
    [i] = run(w).issues
    assert i.code == "sync.caption_drift" and i.locked and not i.auto_fix_available and not i.auto_fix_safe and "auto-fix is disabled" in i.fix_blocked_reason and i.fix is not None


def test_locked_clip_locked_track_and_locked_scene_disable_the_fix(tmp_path):
    for how in ("clip", "track", "scene"):
        w = World(tmp_path / how)
        w.visuals()
        cap = w.caption(0, 5, 0.42)
        if how == "clip":
            cap.locked = True
        elif how == "track":
            w.p.timeline.get_track("track_v6").locked = True
        else:
            w.p.presentation_generation.locked_scenes = [w.s1.id]
        [i] = run(w).issues
        assert i.locked and not i.auto_fix_available, how


def test_fix_permission_never_disables_the_fix(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, 0.42)
    s = QCSettings()
    s.fix_permissions["caption.retime"] = "never"
    [i] = run(w, s).issues
    assert not i.auto_fix_available and i.fix_blocked_reason == "Disabled in QC settings"
    s.fix_permissions["caption.retime"] = "confirm"
    [j] = run(w, s).issues
    assert j.auto_fix_available and not j.auto_fix_safe


# ------------------------------------------------------------------ visuals
def test_visual_starting_1200_ms_after_its_statement_is_visual_late(tmp_path):
    w = World(tmp_path)
    [v1, _v2] = w.visuals(((1.7, 10.0), (10.0, 20.0)))
    [i] = run(w).issues
    assert i.code == "sync.visual_late" and i.severity is Severity.ERROR and i.scene_id == w.s1.id and i.timeline_item_id == v1.id and "1200 ms" in i.title
    assert i.fix.kind == "open.timeline" and i.fix.route.value == "NAVIGATE" and not i.auto_fix_safe  # moving AI visuals is never an automatic fix
    assert i.start_time == 0.5 and i.end_time == 1.7 and 0 < i.confidence < 100 and i.metrics["offset_ms"] == 1200.0


def test_visual_arriving_within_the_late_tolerance_is_fine(tmp_path):
    w = World(tmp_path)
    w.visuals(((0.8, 10.0), (10.0, 20.0)))  # 300 ms after the first word: under visual_late_ms (400)
    assert run(w).issues == []
    w2 = World(tmp_path / "b")
    w2.visuals(((0.95, 10.0), (10.0, 20.0)))  # 450 ms: reaches it, below major
    [i] = run(w2).issues
    assert i.code == "sync.visual_late" and i.severity is Severity.WARNING


def test_a_visual_on_an_evidence_scene_is_an_important_visual(tmp_path):
    w = World(tmp_path)
    w.visuals(((1.7, 10.0), (10.0, 20.0)))
    w.p.visual_intents[w.s1.id] = VisualIntent(w.s1.id, VisualType.EVIDENCE)
    assert codes(run(w)) == ["sync.important_visual_late"]


def test_visual_late_is_not_reported_for_a_declared_gap_or_a_hidden_track(tmp_path):
    w = World(tmp_path)
    w.visuals(((1.7, 10.0), (10.0, 20.0)))
    s = QCSettings()
    s.intentional_gaps = [[0.0, 2.0]]
    assert run(w, s).issues == []
    w2 = World(tmp_path / "h")
    w2.visuals(((1.7, 10.0), (10.0, 20.0)))
    w2.p.timeline.get_track("track_v1").hidden = True  # not on screen: nothing to be late
    assert run(w2).issues == []


def test_visual_starting_during_the_previous_statement_is_visual_early(tmp_path):
    w = World(tmp_path)
    [_v1, v2] = w.visuals(((0.0, 8.9), (8.9, 20.0)))  # the previous scene is spoken until 9.2 s
    out = run(w)
    [i] = out.issues
    assert i.code == "sync.visual_early" and i.severity is Severity.WARNING and i.scene_id == w.s2.id and i.timeline_item_id == v2.id and "300 ms" in i.title and i.metrics["edge"] == "start"
    assert not any(x.code == "sync.visual_early" and x.scene_id == w.s1.id for x in out.issues)  # the same cut is reported once, by the picture that starts it


def test_a_cut_inside_the_pause_before_the_statement_is_not_early(tmp_path):
    w = World(tmp_path)
    w.visuals(((0.0, 9.3), (9.3, 20.0)))  # after the last word (9.2 s), well before the next one (10.5 s)
    assert run(w).issues == []


def test_visual_leaving_before_the_statement_is_finished(tmp_path):
    w = World(tmp_path)
    [v1, _v2] = w.visuals(((0.0, 8.0), (10.0, 20.0)))  # nothing on screen for the last second of the scene's speech
    [i] = run(w).issues
    assert i.code == "sync.visual_early" and i.metrics["edge"] == "end" and i.severity is Severity.ERROR and i.timeline_item_id == v1.id and "1200 ms" in i.title


def test_visual_staying_over_the_next_statement_is_an_overstay_reported_once(tmp_path):
    w = World(tmp_path)
    [v1, _v2] = w.visuals(((0.0, 11.0), (11.0, 20.0)))  # scene 2's narration starts at 10.5 s
    out = run(w)
    assert codes(out) == ["sync.visual_overstay"]  # scene 2's picture is late only because of it: not reported again
    [i] = out.issues
    assert i.scene_id == w.s1.id and i.timeline_item_id == v1.id and i.severity is Severity.ERROR and i.metrics["overstay_ms"] == 500.0 and i.start_time == 10.5


def test_visual_ending_in_the_pause_after_its_statement_is_not_an_overstay(tmp_path):
    w = World(tmp_path)
    w.visuals(((0.0, 10.3), (10.3, 20.0)))  # 200 ms into scene 2 but still before its first word (10.5 s)
    assert run(w).issues == []


def test_user_owned_visual_is_reported_with_the_fix_disabled(tmp_path):
    w = World(tmp_path)
    w.visuals(((1.7, 10.0), (10.0, 20.0)), by="USER")
    [i] = run(w).issues
    assert i.code == "sync.visual_late" and i.locked and not i.auto_fix_available


# ------------------------------------------------------------------ number / date graphics and evidence highlights
def test_number_graphic_after_the_spoken_figure_is_an_important_visual_late(tmp_path):
    w = World(tmp_path)
    w.visuals()
    g = w.text(3.0, "42%", "NUMBER", emphasis="NUMBER_CARD")  # the figure is said at 1.6 s
    [i] = run(w).issues
    assert i.code == "sync.important_visual_late" and i.severity is Severity.ERROR and i.timeline_item_id == g.id and "1400 ms" in i.title and i.track_id == "track_v5"
    assert i.metrics["spoken_start"] == 1.6 and i.confidence == 92.0 and i.fix.kind == "open.timeline"


def test_number_graphic_inside_the_tolerance_or_leading_the_word_is_fine(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.text(1.85, "42%", "NUMBER", emphasis="NUMBER_CARD", preset="fade_in")  # 250 ms after the word starts: inside visual_late_ms (400)
    assert run(w).issues == []
    w2 = World(tmp_path / "b")
    w2.visuals()
    w2.text(0.9, "42%", "NUMBER", emphasis="NUMBER_CARD", preset="fade_in")  # early graphics are not "late"; the emphasis rule below is about the beat
    assert [i.code for i in run(w2).issues] == ["sync.emphasis_miss"]


def test_a_figure_spoken_in_words_is_found_through_its_mention(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.s1.numbers = [NumericMention("forty two percent", NumberKind.PERCENTAGE, 42.0, "", True, [w.words[W_NUMBER].word_id, w.words[W_NUMBER + 1].word_id])]
    w.text(3.0, "42%", "NUMBER", emphasis="NUMBER_CARD", preset="fade_in")
    [i] = run(w).issues
    assert i.code == "sync.important_visual_late" and i.metrics["spoken_start"] == 1.6


def test_evidence_highlight_after_its_claim_was_said_is_late(tmp_path):
    w = World(tmp_path)
    w.visuals()
    h = w.highlight(4.0)  # the claim sentence ended at 3.1 s
    [i] = run(w).issues
    assert i.code == "sync.important_visual_late" and i.timeline_item_id == h.id and i.severity is Severity.ERROR and i.metrics["statement_end"] == 3.1 and "Evidence highlight" in i.title
    w2 = World(tmp_path / "b")
    w2.visuals()
    w2.highlight(3.3)  # 200 ms after the statement: under visual_late_ms
    assert run(w2).issues == []


def test_evidence_highlight_without_a_claim_uses_the_whole_narration_with_less_confidence(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.s1.claims = []
    w.highlight(10.0 - 0.01)
    assert run(w).issues[0].confidence == 60.0 if run(w).issues else True
    w2 = World(tmp_path / "b")
    w2.visuals()
    w2.s1.claims = []
    w2.highlight(9.9)
    out = run(w2)
    assert [i.confidence for i in out.issues if i.code == "sync.important_visual_late"] == [60.0]


# ------------------------------------------------------------------ emphasis
def test_emphasis_animation_missing_its_word_is_an_emphasis_miss(tmp_path):
    w = World(tmp_path)
    w.visuals()
    g = w.text(5.2, "PENALTIES", "WARNING", emphasis="WARNING_TEXT")  # "Penalties" is said 4.0-4.6 s
    [i] = run(w).issues
    assert i.code == "sync.emphasis_miss" and i.severity is Severity.ERROR and i.timeline_item_id == g.id and i.metrics["miss_ms"] == 600.0 and "after" in i.title


def test_emphasis_on_the_beat_is_clean(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.text(3.85, "PENALTIES", "WARNING", emphasis="WARNING_TEXT")
    w.text(4.7, "apply immediately", "TEXT", emphasis="BOLD_TEXT", scene=w.s1, dur=1.0)  # still inside the spoken phrase
    assert run(w).issues == []


def test_a_counter_that_starts_long_after_the_number_is_an_emphasis_miss_not_a_late_graphic(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.text(4.0, "42%", "NUMBER", emphasis="NUMBER_CARD", preset="counter", counter={"from": 0, "to": 42, "decimals": 0, "prefix": "", "suffix": "%"})
    [i] = run(w).issues
    assert i.code == "sync.emphasis_miss" and "counter" in i.description and i.metrics["miss_ms"] == 1700.0


def test_animation_delay_moves_the_emphasis_moment(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.text(4.0, "PENALTIES", "WARNING", emphasis="WARNING_TEXT", delay=0.0)
    assert codes(run(w)) == []  # on the word
    w2 = World(tmp_path / "b")
    w2.visuals()
    w2.text(3.0, "PENALTIES", "WARNING", emphasis="WARNING_TEXT", delay=2.2)  # the clip starts early, the animation lands at 5.2 s
    assert codes(run(w2)) == ["sync.emphasis_miss"]


def test_text_that_is_not_in_the_narration_cannot_be_compared(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.text(7.0, "A Section Headline", "HEADLINE", derived=True)
    w.text(7.0, "Completely unrelated words", "TEXT")
    assert run(w).issues == []


def test_caption_highlight_timed_off_the_spoken_word(tmp_path):
    w = World(tmp_path)
    w.visuals()
    # the emphasised caption word (index 3, "42%") carries timing 0.6 s later than the transcript: the highlight lands after the word
    w.caption(0, 5, emphasis=[{"word_index": 3, "category": "NUMBER", "style": "COLOR_CHANGE", "reason": "figure"}], own_shift=0.0)
    cap = w.p.timeline.get_track("track_v6").clips[0]
    assert run(w).issues == []
    cap.text["words"][3]["start"] = round(1.6 + 0.6, 3)
    cap.text["words"][3]["end"] = 2.9
    [i] = run(w).issues
    assert i.code == "sync.emphasis_miss" and i.metrics["miss_ms"] == 600.0 and i.metrics["word"] == "42%"


def test_caption_leaving_before_its_emphasised_word_is_spoken(tmp_path):
    w = World(tmp_path)
    w.visuals()
    cap = w.caption(0, 5, emphasis=[{"word_index": 5, "category": "KEYWORD", "style": "BOLD", "reason": ""}])
    cap.duration = 1.2  # the window was trimmed: it is gone at 1.7 s, "week." is said at 2.7 s
    [i] = run(w).issues
    assert i.code == "sync.emphasis_miss" and "leaves the screen" in i.description and i.metrics["miss_ms"] == 1000.0


# ------------------------------------------------------------------ cuts / transitions inside key phrases
def test_cut_inside_a_spoken_figure(tmp_path):
    w = World(tmp_path)
    [_a, b, _c] = w.visuals(((0.0, 2.0), (2.0, 10.0), (10.0, 20.0)))  # "42%" is said 1.6-2.3 s
    [i] = run(w).issues
    assert i.code == "sync.cut_in_phrase" and i.severity is Severity.WARNING and i.timeline_item_id == b.id and i.metrics["depth_ms"] == 300.0 and "figure" in i.title
    assert i.start_time == 1.6 and i.end_time == 2.3 and i.fix.route.value == "NAVIGATE"


def test_cut_in_the_pause_between_sentences_is_fine(tmp_path):
    w = World(tmp_path)
    w.visuals(((0.0, 3.5), (3.5, 6.0), (6.0, 10.0), (10.0, 20.0)))
    assert run(w).issues == []


def test_cut_right_at_the_edge_of_a_figure_is_natural(tmp_path):
    w = World(tmp_path)
    w.visuals(((0.0, 1.55), (1.55, 2.35), (2.35, 10.0), (10.0, 20.0)))  # within 100 ms of both edges
    assert run(w).issues == []


def test_transition_running_through_a_figure(tmp_path):
    w = World(tmp_path)
    [_a, b, _c] = w.visuals(((0.0, 1.8), (1.8, 10.0), (10.0, 20.0)))
    b.transition = {"type": "FADE", "duration": 0.6, "decision_id": ""}  # half-way at 2.1 s, inside the figure (1.6-2.3 s)
    [i] = find(run(w), "sync.cut_in_phrase")
    assert "fade transition" in i.title.lower() and i.metrics["kind"] == "fade transition" and i.timeline_item_id == b.id and i.metrics["cut_time"] == 2.1
    w2 = World(tmp_path / "after")
    [_a, b2, _c] = w2.visuals(((0.0, 2.5), (2.5, 10.0), (10.0, 20.0)))
    b2.transition = {"type": "FADE", "duration": 0.6, "decision_id": ""}  # half-way at 2.8 s: after the figure
    assert find(run(w2), "sync.cut_in_phrase") == []


def test_cut_during_a_word_of_a_key_claim_but_not_between_its_words(tmp_path):
    w = World(tmp_path)
    [_a, b, _c] = w.visuals(((0.0, 0.7), (0.7, 10.0), (10.0, 20.0)))  # inside "Silver" (0.5-0.9 s)
    [i] = run(w).issues
    assert i.code == "sync.cut_in_phrase" and i.timeline_item_id == b.id and "claim" in i.title and i.confidence == 75.0 and i.severity is Severity.NOTICE
    w2 = World(tmp_path / "b")
    w2.visuals(((0.0, 1.3), (1.3, 10.0), (10.0, 20.0)))  # exactly between "prices" and "rose"
    assert run(w2).issues == []


def test_cut_in_a_phrase_that_is_not_a_key_phrase_is_left_to_the_pacing_checker(tmp_path):
    w = World(tmp_path)
    w.visuals(((0.0, 7.2), (7.2, 10.0), (10.0, 20.0)))  # inside "Experts expect ...": no figure, no claim
    assert run(w).issues == []


# ------------------------------------------------------------------ metrics, text, robustness
def test_metrics_report_max_mean_and_count_per_scene(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, 0.42)
    w.caption(6, 8, 0.06)
    w.caption(14, 18, -0.0)
    out = run(w)
    k = f"scene.{w.s1.id}."
    assert out.metrics[k + "captions_checked"] == 2 and out.metrics[k + "max_caption_drift_ms"] == 420.0 and out.metrics[k + "mean_caption_drift_ms"] == 240.0
    assert out.metrics[k + "mean_signed_caption_drift_ms"] == 240.0 and out.metrics[f"scene.{w.s2.id}.captions_checked"] == 1
    assert out.metrics["captions_checked"] == 3 and out.metrics["max_caption_drift_ms"] == 420.0 and out.metrics["mean_caption_drift_ms"] == 160.0
    assert sync_summary(out.metrics) == {"captions_checked": 3.0, "max_caption_drift_ms": 420.0, "mean_caption_drift_ms": 160.0}


def test_hidden_caption_track_is_not_what_the_viewer_sees(tmp_path):
    w = World(tmp_path)
    w.visuals()
    w.caption(0, 5, 0.9)
    w.p.timeline.get_track("track_v6").hidden = True
    out = run(w)
    assert out.issues == [] and out.metrics["captions_checked"] == 0


def test_scene_without_narration_or_visuals_is_quietly_skipped(tmp_path):
    p = new_project(tmp_path, seconds=10)
    add_scene(p, 0, 10, "")
    a = add_asset(p, "a.mp4", "video", duration=20)
    add_clip(p, "track_v1", a, 3, 4, created_by="AI")
    out = run_checker(SyncChecker(), qc_ctx(p))
    assert out.issues == [] and out.notes[0].startswith("No transcript")


def test_a_stale_scene_id_still_lands_in_the_scene_at_its_time(tmp_path):
    w = World(tmp_path)
    [v1, v2] = w.visuals(((1.7, 10.0), (10.0, 20.0)))
    v1.scene_id = w.s2.id  # scene commands never rewrite clip scene ids: the clip is positioned in scene 1
    [i] = run(w).issues
    assert i.code == "sync.visual_late" and i.scene_id == w.s1.id
    assert v2.scene_id == w.s2.id


def test_fingerprint_survives_small_changes_and_changes_with_substantial_ones(tmp_path):
    fps = []
    for shift in (0.42, 0.44, 0.9):
        w = World(tmp_path / f"f{int(shift * 100)}")
        w.visuals()
        cap = w.caption(0, 5, shift)
        cap.id = "clip_fixed"
        fps.append(run(w).issues[0].fingerprint)
    assert fps[0] == fps[1] and fps[0] != fps[2]


def test_progress_is_reported_and_cancel_is_honoured(tmp_path):
    import pytest

    from app.qc.context import QCCancelled

    w = World(tmp_path)
    w.visuals()
    seen = []
    SyncChecker().run(qc_ctx(w.p), lambda f, m: seen.append((f, m)))
    assert seen and seen[-1][0] == 1.0 and 0.0 in [f for f, _m in seen]
    ctx = qc_ctx(w.p)
    ctx.cancel.set()
    with pytest.raises(QCCancelled):
        SyncChecker().run(ctx, lambda f, m: None)


# ------------------------------------------------------------------ scene-local contract
def _two_scene_problems(tmp_path):
    w = World(tmp_path)
    w.visuals(((1.7, 5.0), (5.0, 10.0), (10.0, 20.0)))  # scene 1: first visual late (two pictures, so fixing the first leaves the clip next to scene 2 untouched)
    w.caption(W_DEMAND, W_DEMAND + 4, 0.42)  # scene 2: caption drift
    return w


def test_every_issue_is_scene_scoped_and_a_scene_filter_analyses_only_that_scene(tmp_path):
    w = _two_scene_problems(tmp_path)
    full = run(w)
    assert sorted((i.scene_id, i.code) for i in full.issues) == [(w.s1.id, "sync.visual_late"), (w.s2.id, "sync.caption_drift")] and all(i.scene_id for i in full.issues)
    only2 = run(w, scene_filter=[w.s2.id])
    assert [(i.scene_id, i.code) for i in only2.issues] == [(w.s2.id, "sync.caption_drift")]
    assert all(f"scene.{w.s1.id}." not in k for k in only2.metrics) and f"scene.{w.s2.id}.captions_checked" in only2.metrics


def test_scene_signature_isolates_scenes_so_only_the_changed_one_is_reanalysed(tmp_path):
    w = _two_scene_problems(tmp_path)
    chk = SyncChecker()
    eng = QCEngine([chk])
    first = eng.run(qc_ctx(w.p))
    assert first.run.checkers["sync"].state is CheckerState.DONE and len(first.run.issues) == 2
    h1 = chk.scene_input_hash(qc_ctx(w.p), w.s1.id)
    w.caption(W_PRICES_3, W_PRICES_3 + 3, 0.9)  # a new problem in scene 2 only
    assert chk.scene_input_hash(qc_ctx(w.p), w.s1.id) == h1
    second = eng.run(qc_ctx(w.p), previous=PreviousState(first.run.issues, first.cache))
    st = second.run.checkers["sync"]
    assert st.reused_scenes == 1 and st.analyzed_scenes == 1
    assert sorted((i.scene_id, i.code) for i in second.run.issues) == [(w.s1.id, "sync.visual_late"), (w.s2.id, "sync.caption_drift"), (w.s2.id, "sync.caption_drift")]
    # fixing scene 1's visual re-analyses scene 1 and keeps scene 2's findings (a clip that overlaps the neighbour's +-0.5 s window would re-analyse that scene too: the signature is conservative)
    first_clip = w.p.timeline.get_track("track_v1").clips[0]
    first_clip.timeline_start, first_clip.duration = 0.0, 5.0
    third = eng.run(qc_ctx(w.p), previous=PreviousState(second.run.issues, second.cache))
    assert third.run.checkers["sync"].analyzed_scenes == 1 and sorted(i.code for i in third.run.issues) == ["sync.caption_drift", "sync.caption_drift"]
    assert sync_summary(third.run.metrics["sync"])["captions_checked"] == 2


def test_the_checker_never_mutates_the_project(tmp_path):
    w = _two_scene_problems(tmp_path)
    w.text(5.2, "PENALTIES", "WARNING", emphasis="WARNING_TEXT")
    w.highlight(4.0)
    before = w.p.to_document()
    out = run(w)
    assert len(out.issues) >= 4 and w.p.to_document() == before
