"""Phase 7 structure analysis: sections / hook / pacing curve / section transitions / energy arc, from known synthetic observation lists.

The observations are built by hand (cut times, text events, motion and audio series, silences, transitions), so every structural fact the analyzer should find
(where the pacing changes, how intense the opening is, what happens at a section boundary) is known by construction. No media and no FFmpeg are needed.
"""

from __future__ import annotations

import pytest

from app.core.serialization import from_plain, to_plain
from app.reference.shot_detector import compute_shot_stats
from app.reference.structure_analyzer import StructureAnalyzer
from app.reference.style_model import SECTION_KINDS, HookProfile, ReferenceEvents, Shot, TextEvent


# ---------------------------------------------------------------------------------------------- builders
def cuts_every(gap: float, start: float, end: float) -> list[float]:
    out, t = [], start + gap
    while t < end - 1e-6:
        out.append(round(t, 3))
        t += gap
    return out


def audio_series(duration: float, pieces: list[tuple[float, float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """pieces = (start, end, loudness, voice, music); two samples per second."""
    out, t = [], 0.0
    while t < duration:
        row = next(((p[2], p[3], p[4]) for p in pieces if p[0] <= t < p[1]), (0.0, 0.0, 0.0))
        out.append((round(t, 3), *row))
        t += 0.5
    return out


def motion_series(duration: float, pieces: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    return [(float(t), next((p[2] for p in pieces if p[0] <= t < p[1]), 0.0)) for t in range(int(duration))]


def text(start: float, end: float, kind: str = "HEADLINE", position: str = "center", height: float = 0.09) -> TextEvent:
    return TextEvent(start, end, kind, position, height, 0.5, 1, False, False, False, 0.8)


def shots_from_cuts(cuts: list[float], duration: float) -> list[Shot]:
    edges = [0.0, *cuts, duration]
    return [Shot(f"shot_{i:03d}", edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def three_part(duration: float = 90.0) -> tuple[ReferenceEvents, list[float]]:
    """0-15 s: very fast, loud, busy opening with title cards. 15-75 s: calm, slow, narrated. 75-90 s: end card (text, music, no voice)."""
    cuts = cuts_every(1.5, 0, 15) + cuts_every(6.0, 15, 75) + cuts_every(5.0, 75, duration)
    ev = ReferenceEvents(cut_times=cuts)
    ev.text_events = [text(1, 3), text(6, 8, "NUMBER_CARD"), text(11, 13)] + [text(76, 80), text(81, 85, "TEXT", "center", 0.07), text(86, 89)]
    ev.motion_series = motion_series(duration, [(0, 15, 0.6), (15, 75, 0.1), (75, duration, 0.2)])
    ev.audio_series = audio_series(duration, [(0, 15, 0.8, 0.8, 0.3), (15, 75, 0.35, 0.7, 0.1), (75, duration, 0.5, 0.0, 0.7)])
    ev.silences = [(14.2, 15.4)]
    return ev, cuts


def uniform(duration: float = 60.0, gap: float = 4.0) -> tuple[ReferenceEvents, list[float]]:
    cuts = cuts_every(gap, 0, duration)
    ev = ReferenceEvents(cut_times=cuts)
    ev.motion_series = motion_series(duration, [(0, duration, 0.2)])
    ev.audio_series = audio_series(duration, [(0, duration, 0.4, 0.7, 0.2)])
    return ev, cuts


# ---------------------------------------------------------------------------------------------- sections
def test_boundaries_are_found_where_the_video_actually_changes():
    ev, cuts = three_part()
    res = StructureAnalyzer().analyze(ev, 90.0, shots_from_cuts(cuts, 90.0))
    assert len(res.boundaries) == 2
    assert abs(res.boundaries[0] - 15.0) <= 3.0 and abs(res.boundaries[1] - 75.0) <= 3.0
    s = res.sections
    assert len(s) == 3 and s[0].start == 0.0 and s[-1].end == pytest.approx(90.0)
    assert all(a.end == pytest.approx(b.start) for a, b in zip(s, s[1:]))  # the sections tile the video
    assert all(x.kind in SECTION_KINDS for x in s)
    assert s[0].kind == "HOOK" and s[-1].kind in ("CTA", "CONCLUSION")
    assert s[0].cuts_per_minute > 3 * s[1].cuts_per_minute and s[0].average_shot_duration < s[1].average_shot_duration
    assert res.intro_end == pytest.approx(s[0].end) and res.outro_start == pytest.approx(s[-1].start)


def test_a_short_closing_end_card_with_text_and_music_but_no_voice_is_a_call_to_action():
    ev, cuts = three_part()
    res = StructureAnalyzer().analyze(ev, 90.0, shots_from_cuts(cuts, 90.0))
    assert res.sections[-1].kind == "CTA" and res.cta_start == pytest.approx(res.sections[-1].start)
    # the same ending without the signals of an end card is a conclusion
    ev2, cuts2 = three_part()
    ev2.text_events = ev2.text_events[:3]
    ev2.audio_series = audio_series(90.0, [(0, 15, 0.8, 0.8, 0.3), (15, 90, 0.35, 0.7, 0.1)])
    res2 = StructureAnalyzer().analyze(ev2, 90.0)
    assert res2.cta_start is None and res2.sections[-1].kind == "CONCLUSION"


def test_an_unchanging_video_is_one_section_and_cues_alone_do_not_create_boundaries():
    ev, _ = uniform()
    res = StructureAnalyzer().analyze(ev, 60.0)
    assert res.boundaries == [] and len(res.sections) == 1 and res.sections[0].kind == "EXPLANATION"
    assert res.confidence <= 0.4
    ev.silences = [(29.0, 31.0)]
    ev.music_change_times = [30.0]
    ev.transitions = [(30.0, "DISSOLVE", 0.8)]
    ev.text_events = [text(30.5, 33)]
    assert StructureAnalyzer().analyze(ev, 60.0).boundaries == []


def test_a_modest_change_is_found_and_cues_pull_it_to_the_right_second():
    cuts = cuts_every(4.0, 0, 40) + cuts_every(1.6, 40, 80)
    ev = ReferenceEvents(cut_times=cuts)
    ev.motion_series = motion_series(80.0, [(0, 40, 0.15), (40, 80, 0.45)])
    ev.transitions = [(40.0, "DISSOLVE", 0.8)]
    res = StructureAnalyzer().analyze(ev, 80.0)
    assert len(res.boundaries) == 1 and abs(res.boundaries[0] - 40.0) <= 2.0
    # a mere doubling of the cut rate with nothing else changing is below the bar: one section
    plain = ReferenceEvents(cut_times=cuts_every(4.0, 0, 40) + cuts_every(2.0, 40, 80), motion_series=motion_series(80.0, [(0, 80, 0.15)]))
    assert StructureAnalyzer().analyze(plain, 80.0).boundaries == []


def test_boundaries_snap_to_a_nearby_shot_boundary():
    ev, cuts = three_part()
    res = StructureAnalyzer().analyze(ev, 90.0)
    for b in res.boundaries:
        near = min(abs(b - c) for c in cuts)
        assert near < 1e-6 or all(abs(b - c) > 1.5 for c in cuts)  # on a cut whenever one is within 1.5 s


def test_section_count_is_bounded_and_sections_keep_a_minimum_length():
    cuts, pieces_m, pieces_a = [], [], []
    t = 0.0
    for i in range(12):  # alternate busy / calm every 20 s for 240 s
        gap = 1.0 if i % 2 == 0 else 8.0
        cuts += cuts_every(gap, t, t + 20)
        pieces_m.append((t, t + 20, 0.7 if i % 2 == 0 else 0.05))
        pieces_a.append((t, t + 20, 0.9 if i % 2 == 0 else 0.2, 0.5, 0.2))
        t += 20
    ev = ReferenceEvents(cut_times=cuts, motion_series=motion_series(240.0, pieces_m), audio_series=audio_series(240.0, pieces_a))
    res = StructureAnalyzer().analyze(ev, 240.0)
    assert 4 <= len(res.sections) <= 9
    assert all(s.duration >= 6.0 for s in res.sections)
    assert all(b - a >= 6.0 for a, b in zip(res.boundaries, res.boundaries[1:]))


def test_interior_sections_get_distinct_kinds_from_their_energy_and_text():
    # hook, context, a slow quiet text-heavy part (evidence), a loud busy peak (reveal), a calm explanation, conclusion
    cuts = cuts_every(2.0, 0, 20) + cuts_every(5.0, 20, 60) + cuts_every(12.0, 60, 110) + cuts_every(1.2, 110, 130) + cuts_every(6.0, 130, 180) + cuts_every(6.0, 180, 200)
    ev = ReferenceEvents(cut_times=cuts)
    ev.motion_series = motion_series(200.0, [(0, 20, 0.5), (20, 60, 0.15), (60, 110, 0.03), (110, 130, 0.85), (130, 200, 0.12)])
    ev.audio_series = audio_series(200.0, [(0, 20, 0.7, 0.8, 0.2), (20, 60, 0.45, 0.8, 0.1), (60, 110, 0.2, 0.8, 0.0), (110, 130, 0.95, 0.8, 0.5), (130, 200, 0.4, 0.8, 0.1)])
    ev.text_events = [text(62 + 5 * i, 66 + 5 * i, "TEXT", "center", 0.05) for i in range(9)]
    res = StructureAnalyzer().analyze(ev, 200.0)
    kinds = [s.kind for s in res.sections]
    assert kinds[0] == "HOOK" and kinds[-1] in ("CONCLUSION", "CTA")
    assert "REVEAL" in kinds and "EVIDENCE" in kinds and kinds.count("REVEAL") == 1
    reveal = res.sections[kinds.index("REVEAL")]
    assert 105 <= reveal.start <= 115 and 125 <= reveal.end <= 135
    evidence = res.sections[kinds.index("EVIDENCE")]
    assert evidence.cuts_per_minute < reveal.cuts_per_minute / 4 and evidence.text_per_minute > 5
    assert all(0.2 <= s.confidence <= 0.75 for s in res.sections)  # names are guesses: never high confidence


# ---------------------------------------------------------------------------------------------- pacing curve
def test_pacing_curve_follows_the_changes_in_cut_rate_and_tiles_the_video():
    cuts = cuts_every(2.0, 0, 30) + cuts_every(12.0, 30, 90) + cuts_every(2.0, 90, 120)
    res = StructureAnalyzer().analyze(ReferenceEvents(cut_times=cuts), 120.0, shots_from_cuts(cuts, 120.0))
    curve = res.pacing_curve
    assert curve[0].start == 0.0 and curve[-1].end == pytest.approx(120.0)
    assert all(a.end == pytest.approx(b.start) for a, b in zip(curve, curve[1:]))
    assert [c.label for c in curve] == ["Fast", "Slow", "Fast"] or [c.label for c in curve] == ["Very Fast", "Slow", "Very Fast"]
    assert abs(curve[0].end - 30) <= 7 and abs(curve[1].end - 90) <= 7
    assert curve[0].cuts_per_minute == pytest.approx(30, abs=5) and curve[1].cuts_per_minute == pytest.approx(5, abs=2)
    assert curve[0].average_shot_duration == pytest.approx(2.0, abs=0.4) and curve[1].average_shot_duration > 6
    assert all(a.label != b.label for a, b in zip(curve, curve[1:]))  # neighbours with the same class are one segment


def test_a_steady_video_has_a_single_pacing_segment_and_short_blips_are_absorbed():
    ev, cuts = uniform(90.0, 4.0)
    assert len(StructureAnalyzer().analyze(ev, 90.0).pacing_curve) == 1
    blip = sorted(cuts + cuts_every(0.8, 40, 43))  # three seconds of rapid cutting inside a steady video
    res = StructureAnalyzer().analyze(ReferenceEvents(cut_times=blip), 90.0)
    assert all(s.end - s.start >= 6.0 for s in res.pacing_curve[:-1]) or len(res.pacing_curve) <= 2


def test_pacing_curve_carries_motion_and_text_rates_per_segment():
    cuts = cuts_every(2.0, 0, 40) + cuts_every(10.0, 40, 80)
    ev = ReferenceEvents(cut_times=cuts, motion_series=motion_series(80.0, [(0, 40, 0.6), (40, 80, 0.1)]))
    ev.text_events = [text(2 + 3 * i, 4 + 3 * i, "HEADLINE") for i in range(12)]  # all in the first 40 s
    curve = StructureAnalyzer().analyze(ev, 80.0).pacing_curve
    assert curve[0].motion > curve[-1].motion + 0.3 and curve[0].text_per_minute > 10 and curve[-1].text_per_minute == pytest.approx(0, abs=1.0)


# ---------------------------------------------------------------------------------------------- hook
def test_hook_windows_are_measured_and_the_fast_loud_opening_stands_out():
    ev, _ = three_part()
    hook = StructureAnalyzer().analyze(ev, 90.0).hook
    assert [w.seconds for w in hook.windows] == [5, 10, 15, 30]
    w10 = hook.windows[1]
    assert w10.shot_rate == pytest.approx(40, abs=8) and w10.relative_pacing > 2.0
    assert w10.text_density >= 6 and w10.number_emphasis >= 6 and w10.motion == pytest.approx(0.6, abs=0.05) and w10.audio_intensity > 0.8
    assert hook.intensity > 0.5
    assert "Fast visual switching" in hook.traits and "Strong text" in hook.traits and "High motion" in hook.traits


def test_hook_windows_only_include_those_that_fit_the_video():
    ev, _ = uniform(24.0, 3.0)
    assert [w.seconds for w in StructureAnalyzer().analyze(ev, 24.0).hook.windows] == [5, 10, 15]
    ev, _ = uniform(12.0, 3.0)
    assert [w.seconds for w in StructureAnalyzer().analyze(ev, 12.0).hook.windows] == [5]
    ev, _ = uniform(6.0, 2.0)
    assert [w.seconds for w in StructureAnalyzer().analyze(ev, 6.0).hook.windows] == [3]


def test_an_opening_paced_like_the_rest_has_no_hook_intensity():
    ev, _ = uniform(90.0, 4.0)
    hook = StructureAnalyzer().analyze(ev, 90.0).hook
    assert hook.intensity < 0.1 and "Opening paced like the rest of the video" in hook.traits
    assert all(w.relative_pacing == pytest.approx(1.0, abs=0.3) for w in hook.windows)


def test_a_calmer_opening_than_the_rest_is_not_called_intense():
    cuts = cuts_every(8.0, 0, 30) + cuts_every(2.0, 30, 90)
    ev = ReferenceEvents(cut_times=cuts, motion_series=motion_series(90.0, [(0, 30, 0.05), (30, 90, 0.5)]))
    assert StructureAnalyzer().analyze(ev, 90.0).hook.intensity == 0.0


# ---------------------------------------------------------------------------------------------- section transitions
def test_section_transition_style_reads_what_happens_at_the_boundaries():
    ev, _ = three_part()
    ev.transitions = [(13.5, "DISSOLVE", 0.8), (75.0, "FADE", 0.6)]
    ev.silences = [(13.0, 14.8), (74.4, 75.6)]
    ev.music_change_times = [14.0, 74.8]
    ev.text_events += [text(14.5, 17.5, "HEADLINE")]
    ev.text_events += [text(75.5, 77.5, "NUMBER_CARD")]
    res = StructureAnalyzer().analyze(ev, 90.0)
    assert len(res.boundaries) == 2
    sts = res.section_transition_style
    assert set(sts.transitions_between_sections) == {"DISSOLVE", "FADE"} and sum(sts.transitions_between_sections.values()) == pytest.approx(1.0)
    assert sts.pause_before_section >= 1.0 and sts.text_card_rate == 1.0 and sts.music_change_rate == 1.0 and sts.pacing_change_rate == 0.5  # only the first change is also a change of pace
    assert "Fades or dissolves between sections" in sts.traits and "A pause before a new section" in sts.traits and "Title or text cards introduce sections" in sts.traits


def test_plain_cuts_between_sections_are_reported_as_hard_cuts():
    ev, cuts = three_part()
    ev.silences, ev.text_events, ev.transitions, ev.music_change_times = [], [], [], []
    sts = StructureAnalyzer().analyze(ev, 90.0).section_transition_style
    assert sts.transitions_between_sections.get("CUT") == 1.0 and "Hard cuts between sections" in sts.traits and sts.text_card_rate == 0.0


def test_no_boundaries_means_an_empty_transition_style():
    ev, _ = uniform()
    sts = StructureAnalyzer().analyze(ev, 60.0).section_transition_style
    assert sts.transitions_between_sections == {} and sts.traits == [] and sts.pause_before_section == 0.0


# ---------------------------------------------------------------------------------------------- energy arc, confidence, robustness
def test_energy_arc_shapes():
    def arc(pieces_cuts, pieces_m):
        cuts = [c for a, b, g in pieces_cuts for c in cuts_every(g, a, b)]
        ev = ReferenceEvents(cut_times=cuts, motion_series=motion_series(90.0, pieces_m))
        return StructureAnalyzer().analyze(ev, 90.0)

    front = arc([(0, 30, 1.5), (30, 90, 8.0)], [(0, 30, 0.7), (30, 90, 0.05)])
    build = arc([(0, 60, 8.0), (60, 90, 1.5)], [(0, 60, 0.05), (60, 90, 0.7)])
    mid = arc([(0, 30, 8.0), (30, 60, 1.5), (60, 90, 8.0)], [(0, 30, 0.05), (30, 60, 0.7), (60, 90, 0.05)])
    flat = arc([(0, 90, 4.0)], [(0, 90, 0.2)])
    assert front.arc_shape == "front-loaded" and build.arc_shape == "building" and mid.arc_shape == "peak in the middle" and flat.arc_shape == "flat"
    assert 3 <= len(front.energy_arc) <= 24 and all(0.0 <= e <= 1.0 for _, e in front.energy_arc)
    assert front.energy_arc[0][1] > front.energy_arc[-1][1] and build.energy_arc[0][1] < build.energy_arc[-1][1]


def test_confidence_is_never_high_and_grows_with_evidence():
    ev, cuts = three_part()
    rich = StructureAnalyzer().analyze(ev, 90.0)
    bare = StructureAnalyzer().analyze(ReferenceEvents(cut_times=cuts), 90.0)
    short = StructureAnalyzer().analyze(uniform(10.0, 2.0)[0], 10.0)
    assert 0.5 < rich.confidence <= 0.75 and bare.confidence < rich.confidence and short.confidence <= 0.4
    assert any("estimates" in n for n in rich.notes)


def test_it_is_deterministic():
    ev, cuts = three_part()
    a = StructureAnalyzer().analyze(ev, 90.0, shots_from_cuts(cuts, 90.0))
    b = StructureAnalyzer().analyze(ev, 90.0, shots_from_cuts(cuts, 90.0))
    assert to_plain(a) == to_plain(b)


def test_empty_and_degenerate_inputs_do_not_fail():
    empty = StructureAnalyzer().analyze(ReferenceEvents(), 30.0)
    assert len(empty.sections) == 1 and empty.sections[0].start == 0.0 and empty.sections[0].end == pytest.approx(30.0) and empty.pacing_curve
    assert empty.hook.windows and empty.hook.intensity == 0.0
    zero = StructureAnalyzer().analyze(ReferenceEvents(), 0.0)
    assert zero.sections == [] and zero.pacing_curve == [] and zero.confidence == 0.0
    tiny = StructureAnalyzer().analyze(ReferenceEvents(cut_times=[1.0]), 3.0)
    assert len(tiny.sections) == 1 and tiny.hook.windows[0].seconds == 1
    out_of_range = StructureAnalyzer().analyze(ReferenceEvents(cut_times=[-5.0, 500.0], text_events=[text(900, 905)]), 20.0)
    assert out_of_range.sections


def test_shots_are_optional_and_agree_with_the_cut_list():
    ev, cuts = three_part()
    with_shots = StructureAnalyzer().analyze(ev, 90.0, shots_from_cuts(cuts, 90.0))
    without = StructureAnalyzer().analyze(ev, 90.0)
    assert [round(b) for b in with_shots.boundaries] == [round(b) for b in without.boundaries]
    assert with_shots.sections[0].average_shot_duration == pytest.approx(without.sections[0].average_shot_duration, abs=0.4)
    only_shots = ReferenceEvents()  # no cut list: the cuts come from the shots
    res = StructureAnalyzer().analyze(only_shots, 90.0, shots_from_cuts(cuts, 90.0))
    assert res.sections[0].cuts_per_minute > res.sections[1].cuts_per_minute


def test_sections_feed_the_shot_statistics_per_section():
    ev, cuts = three_part()
    shots = shots_from_cuts(cuts, 90.0)
    res = StructureAnalyzer().analyze(ev, 90.0, shots)
    stats = compute_shot_stats(shots, 90.0, res.sections, 0)
    assert len(stats.cuts_per_section) == len(res.sections) and sum(stats.cuts_per_section) == len(shots) - 1
    assert stats.cuts_per_section[0] > stats.cuts_per_section[1] / 2  # the 15 s opening has about as many cuts as the 60 s middle has /2


def test_result_pieces_round_trip_through_the_project_serialisation():
    ev, cuts = three_part()
    res = StructureAnalyzer().analyze(ev, 90.0)
    hook = from_plain(HookProfile, to_plain(res.hook))
    assert to_plain(hook) == to_plain(res.hook)
    from app.reference.style_model import Section

    assert [to_plain(from_plain(Section, to_plain(s))) for s in res.sections] == [to_plain(s) for s in res.sections]


def test_it_works_inside_the_feature_extractor_contract():
    """The extractor calls ``analyze(events, duration, shots_or_None)`` and reads exactly these attributes."""
    res = StructureAnalyzer().analyze(ReferenceEvents(cut_times=[3.0, 6.0]), 9.0, None)
    for attr in ("sections", "pacing_curve", "hook", "section_transition_style", "confidence", "notes"):
        assert hasattr(res, attr)
    assert isinstance(res.notes, list) and 0.0 <= res.confidence <= 1.0
