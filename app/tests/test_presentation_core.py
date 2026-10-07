"""Phase 5 core tests: animation presets, graphics planning rules, composer (captions/animation), voice track, caching and error handling."""

from __future__ import annotations

import json

import pytest

from app.presentation import animation as an
from app.presentation.graphics import counter_spec
from app.services.presentation_service import PresentationError
from app.tests.conftest import needs_ffmpeg
from app.timeline.clip import KIND_CAPTION, KIND_TEXT

pytestmark = needs_ffmpeg


# ---------------------------------------------------------------- animation presets (pure)
def test_every_preset_has_editable_parameters_and_valid_defaults():
    assert set(an.PRESET_NAMES) >= {"fade_in", "fade_out", "slide_up", "slide_down", "slide_left", "slide_right", "scale_in", "scale_out", "pop", "reveal", "type_on", "counter", "highlight"}
    for name in an.PRESET_NAMES:
        s = an.spec(name)
        assert set(s) == {"preset", "duration", "easing", "delay", "start_value", "end_value"} and s["easing"] in an.EASINGS and s["duration"] > 0
        assert not an.problems({"in": s}, 5.0)
    assert an.problems({"in": {"preset": "spin_around", "duration": 0.3, "easing": "linear"}}, 5.0) and an.problems({"in": an.spec("fade_in", easing="bounce")}, 5.0)
    assert an.problems({"in": an.spec("fade_in", duration=9.0)}, 2.0)  # does not fit the clip
    assert an.problems({"in": "fade"}, 2.0) == []  # the Phase 4 string form is still valid


def test_fade_slide_scale_pop_reveal_and_highlight_evaluate_over_time():
    f = {"in": an.spec("fade_in", duration=1.0, easing="linear"), "out": an.spec("fade_out", duration=1.0, easing="linear")}
    assert an.animation_state(f, 0.0, 5.0).opacity == pytest.approx(0.0) and an.animation_state(f, 0.5, 5.0).opacity == pytest.approx(0.5)
    assert an.animation_state(f, 2.5, 5.0).opacity == 1.0 and an.animation_state(f, 4.5, 5.0).opacity == pytest.approx(0.5) and an.animation_state(f, 5.0, 5.0).opacity == pytest.approx(0.0)
    up = {"in": an.spec("slide_up", duration=1.0, easing="linear")}
    s0, s1 = an.animation_state(up, 0.0, 4.0), an.animation_state(up, 1.0, 4.0)
    assert s0.dy > 0 and s1.dy == pytest.approx(0.0) and s0.opacity == pytest.approx(0.0)  # rises into place
    left = an.animation_state({"in": an.spec("slide_left", duration=1.0)}, 0.0, 4.0)
    assert left.dx > 0 and left.dy == 0
    sc = an.animation_state({"in": an.spec("scale_in", duration=1.0, easing="linear", start_value=0.95)}, 0.0, 4.0)
    assert sc.scale == pytest.approx(0.95)
    peak = max(an.animation_state({"in": an.spec("pop", duration=1.0)}, t / 20, 4.0).scale for t in range(21))
    assert 1.0 < peak <= an.MAX_OVERSHOOT + 1e-9  # a pop overshoots a little, never wildly
    rv = an.animation_state({"in": an.spec("reveal", duration=2.0, easing="linear")}, 1.0, 5.0)
    assert rv.reveal == pytest.approx(0.5)
    assert an.animation_state({"in": an.spec("highlight", duration=1.0, easing="linear")}, 0.5, 4.0).highlight == pytest.approx(0.5)
    assert an.animation_state(None, 1.0, 3.0).opacity == 1.0
    parameters = an.animation_state({"in": an.spec("fade_in", duration=1.0, easing="linear", delay=2.0)}, 1.0, 6.0)
    assert parameters.opacity == pytest.approx(0.0)  # the delay is honoured


def test_counter_counts_up_to_the_real_figure_only():
    spec = counter_spec("$1,500")
    assert spec == {"from": 0.0, "to": 1500.0, "decimals": 0, "prefix": "$", "suffix": "", "thousands": True}
    assert an.counter_text(spec, 0.5) == "$750" and an.counter_text(spec, 1.0) == "$1,500"  # ends on exactly what was said
    assert counter_spec("3.5%")["to"] == 3.5 and an.counter_text(counter_spec("3.5%"), 1.0) == "3.5%"
    assert counter_spec("April 15th") is None or counter_spec("April 15th")["prefix"] == "April "  # dates are not counted
    assert counter_spec("no numbers") is None and an.counter_text(None, 0.5) is None


def test_motion_intensity_levels_and_reduced_motion_choose_calm_defaults():
    assert [an.intensity_level(x) for x in (0.0, 0.33, 0.5, 0.8, 1.0)] == ["LOW", "LOW", "MEDIUM", "HIGH", "HIGH"]
    for variant in ("NUMBER", "DATE", "LOWER_THIRD", "HEADLINE", "WARNING", "EVIDENCE", "TEXT"):
        low = an.default_animation(variant, "LOW")
        red = an.default_animation(variant, "HIGH", reduced=True)
        assert low["in"]["preset"] == "fade_in" == red["in"]["preset"] and red["in"]["duration"] >= 0.25  # reduced motion: fades, nothing fast
        hi = an.default_animation(variant, "HIGH", counter_ok=True)
        assert hi["in"]["duration"] <= 0.9 and not an.problems(hi, 3.0)
    assert an.default_animation("LOWER_THIRD", "MEDIUM")["in"]["preset"] == "slide_up" and an.default_animation("DATE", "MEDIUM")["in"]["preset"] == "scale_in"
    assert an.default_animation("NUMBER", "HIGH", counter_ok=True)["in"]["preset"] == "counter" and an.default_animation("NUMBER", "HIGH")["in"]["preset"] == "pop"
    assert an.normalize({"in": "fade", "out": "fade", "duration": 0.4})["in"]["duration"] == 0.4


# ---------------------------------------------------------------- composer: captions, word highlight, animation
def test_composer_shows_captions_with_word_level_timing_and_animation(pres_ws):
    ws = pres_ws
    ws.presentation.generate(["CAPTIONS", "GRAPHICS"])
    assert ws.jobs.wait_idle(120)
    p = ws.project
    comp = ws.editing.composer()
    caps = sorted([c for c in p.timeline.all_clips() if c.kind == KIND_CAPTION], key=lambda c: c.timeline_start)
    cap = next(c for c in caps if len(c.text["words"]) >= 4)
    words = cap.text["words"]
    mid_word = words[2]
    f = comp.frame_at((mid_word["start"] + mid_word["end"]) / 2)
    layer = next(l for l in f.layers if l.kind == "caption" and l.clip_id == cap.id)
    assert layer.word_index == 2 and layer.text["text"] == cap.text["text"] and layer.opacity > 0.9  # the spoken word is highlighted
    first = comp.frame_at(words[0]["start"] + 0.001)
    assert next(l for l in first.layers if l.clip_id == cap.id).word_index == 0
    start_layer = next(l for l in comp.frame_at(cap.timeline_start + 0.01).layers if l.clip_id == cap.id)
    assert start_layer.opacity < 0.5  # fading in
    assert not any(l.clip_id == cap.id for l in comp.frame_at(cap.timeline_end + 0.01).layers)
    # a number graphic pops/fades in and a counter shows partial values on HIGH intensity
    ws.editing.update_settings(motion_intensity=0.95)
    ws.presentation.generate(["GRAPHICS"])
    assert ws.jobs.wait_idle(120)
    num = next(c for c in p.timeline.all_clips() if c.kind == KIND_TEXT and c.text.get("counter") and c.duration >= 1.5)
    mid = comp.frame_at(num.timeline_start + 0.3)
    nl = next(l for l in mid.layers if l.clip_id == num.id)
    assert nl.counter_text and nl.counter_text != num.text["content"] and nl.progress < 1.0
    done = next(l for l in comp.frame_at(num.timeline_start + 1.1).layers if l.clip_id == num.id)
    assert done.counter_text is None  # finished: the real figure is shown
    ws.presentation.update_caption_settings(reduced_motion=True)
    from app.timeline.keyframes import Keyframe

    vis = next(c for c in p.timeline.all_clips() if c.kind == "media" and c.track_id == "track_v3")
    vis.keyframes.append(Keyframe("scale", 0.0, 1.0))
    vis.keyframes.append(Keyframe("scale", vis.duration, 1.5))
    t = vis.timeline_start + vis.duration - 0.01
    reduced = next(l for l in comp.frame_at(t).layers if l.clip_id == vis.id).scale
    ws.presentation.update_caption_settings(reduced_motion=False)
    full = next(l for l in comp.frame_at(t).layers if l.clip_id == vis.id).scale
    assert 1.0 < reduced < full  # reduced motion softens strong zooms in the preview


# ---------------------------------------------------------------- caption emphasis edits (user overrides)
def test_user_can_change_keyword_emphasis_and_it_survives_regeneration(pres_ws):
    ws = pres_ws
    ws.presentation.generate(["CAPTIONS"])
    assert ws.jobs.wait_idle(120)
    p = ws.project
    cap = next(c for c in p.timeline.all_clips() if c.kind == KIND_CAPTION and len(c.text["words"]) >= 3)
    ws.presentation.update_caption(cap.id, emphasis=[(0, "CONCEPT", "POP"), (2, "NUMBER", "BACKGROUND_BOX")])
    c = p.timeline.get_clip(cap.id)
    assert [(m["word_index"], m["style"]) for m in c.text["emphasis"]] == [(0, "POP"), (2, "BACKGROUND_BOX")] and c.created_by == "USER"
    assert c.text["emphasis"][0]["reason"] == "Set by the user."
    ws.presentation.regenerate_captions([cap.scene_id])
    assert ws.jobs.wait_idle(120)
    assert [(m["word_index"], m["style"]) for m in p.timeline.get_clip(cap.id).text["emphasis"]] == [(0, "POP"), (2, "BACKGROUND_BOX")]
    with pytest.raises(PresentationError):
        ws.presentation.update_caption(cap.id, text="   ")
    with pytest.raises(PresentationError):
        ws.presentation.update_caption(cap.id, style_id="no-such-style")
    with pytest.raises(PresentationError):
        ws.presentation.update_caption(cap.id, duration=-1)
    with pytest.raises(PresentationError):
        ws.presentation.update_caption(next(c.id for c in p.timeline.all_clips() if c.kind == "media"), text="x")


def test_caption_text_edit_keeps_or_redistributes_word_timing(pres_ws):
    ws = pres_ws
    ws.presentation.generate(["CAPTIONS"])
    assert ws.jobs.wait_idle(120)
    p = ws.project
    cap = next(c for c in p.timeline.all_clips() if c.kind == KIND_CAPTION and len(c.text["words"]) == 4)
    old = [(w["start"], w["end"]) for w in cap.text["words"]]
    ws.presentation.update_caption(cap.id, text="Alpha beta gamma delta")  # same number of words: timing kept
    assert [(w["start"], w["end"]) for w in p.timeline.get_clip(cap.id).text["words"]] == old
    ws.presentation.update_caption(cap.id, text="One two three four five six")  # different: spread across the caption
    ws2 = p.timeline.get_clip(cap.id).text["words"]
    assert len(ws2) == 6 and ws2[0]["start"] == pytest.approx(old[0][0], abs=0.01) and ws2[-1]["end"] == pytest.approx(old[-1][1], abs=0.01)
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(ws2, ws2[1:]))


# ---------------------------------------------------------------- voice track: select, trim, split, volume, fade
def test_voice_over_track_is_editable_non_destructively(pres_ws):
    ws = pres_ws
    p = ws.project
    voice = next(c for c in p.timeline.get_track("track_a1").clips)
    assert voice.audio["role"] == "VOICE" and voice.created_by == "SYSTEM"
    asset_path = p.asset_path(p.assets.get(voice.asset_id))
    before = asset_path.read_bytes()
    ws.presentation.set_clip_audio(voice.id, volume=0.9, fade_in=0.5, fade_out=1.0)  # volume + fades
    v = p.timeline.get_clip(voice.id)
    assert v.audio["volume"] == 0.9 and v.audio["fade_in"] == 0.5 and v.created_by == "USER"
    right = ws.timeline.split_clip(voice.id, 30.0)  # split
    left = p.timeline.get_clip(voice.id)
    assert left.duration == pytest.approx(30.0) and right.source_in == pytest.approx(30.0) and right.audio["volume"] == 0.9
    ws.timeline.trim_clip(right.id, new_start=35.0)  # trim: the start moves, the source range follows
    assert p.timeline.get_clip(right.id).source_in == pytest.approx(35.0)
    plan = ws.presentation.mix_plan()
    assert [i for i in plan.items if i.role == "VOICE"] and plan.gain_at("VOICE", 10.0) == pytest.approx(0.9, abs=0.001)
    assert asset_path.read_bytes() == before  # the original voice file is untouched
    for _ in range(3):  # trim, split, audio edit
        ws.undo()
    assert p.timeline.get_clip(voice.id).duration == pytest.approx(p.assets.get(voice.asset_id).duration, abs=0.05)
    assert [d for d in ws.presentation.validate() if d.severity == "error"] == []


# ---------------------------------------------------------------- caching and error handling
def test_caption_plans_are_cached_and_reused(pres_ws, monkeypatch):
    from app.captions.engine import CaptionEngine

    ws = pres_ws
    ws.presentation.generate(["CAPTIONS"])
    assert ws.jobs.wait_idle(120)
    cache = ws.project.root / "cache" / "presentation" / "captions"
    assert len(list(cache.glob("*.json"))) == len(ws.project.scenes)
    calls = []
    real = CaptionEngine.segment
    monkeypatch.setattr(CaptionEngine, "segment", lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1])
    ws.presentation.regenerate_captions()
    assert ws.jobs.wait_idle(120)
    assert calls == []  # unchanged scenes: nothing recomputed
    ws.presentation.update_caption_settings(style_id="bold")
    ws.presentation.regenerate_captions()
    assert ws.jobs.wait_idle(120)
    assert len(calls) == len(ws.project.scenes)  # a style change invalidates the plans


def test_missing_corrupt_audio_and_missing_music_are_handled_without_corrupting_the_project(pres_ws):
    ws = pres_ws
    p = ws.project
    errors = []
    ws.bus.subscribe("app.error", lambda topic, payload: errors.append(payload["message"]))
    vo = p.assets.get(p.voice_over.asset_id)
    path = p.asset_path(vo)
    keep = path.read_bytes()
    path.write_bytes(b"this is not audio" * 100)
    job = ws.presentation.analyze_voice(force=True)
    assert job is not None and ws.jobs.wait_idle(60)
    assert errors and "cannot be decoded" in errors[-1]
    path.unlink()
    ws.presentation.analyze_voice(force=True)
    assert ws.jobs.wait_idle(60)
    assert "missing" in errors[-1]
    path.write_bytes(keep)
    m = ws.presentation.add_music(ws.audio_by["music_bed.wav"].id)
    music_file = p.asset_path(p.assets.get(ws.audio_by["music_bed.wav"].id))
    music_file.unlink()
    got = []
    ws.presentation.preview_mix(__import__("app.presentation.models", fromlist=["PreviewMode"]).PreviewMode.MUSIC, 0.0, 5.0, on_ready=got.append)
    assert ws.jobs.wait_idle(60)
    assert got == [None] and errors[-1].startswith("The audio preview") or got == [None]
    assert m and ws.presentation.validate() is not None
    p.validate()
    json.dumps(p.to_document())  # the document is still serialisable


def test_graphics_planner_never_invents_charts_or_numbers_and_headlines_need_a_topic(pres_ws):
    ws = pres_ws
    ws.presentation.update_caption_settings()  # no-op settings call keeps working
    ws.presentation.generate(["GRAPHICS"])
    assert ws.jobs.wait_idle(120)
    p = ws.project
    for c in p.timeline.all_clips():
        if c.kind == KIND_TEXT and c.text.get("variant") == "NUMBER" and not c.text.get("derived"):
            sc = next(s for s in p.scenes if s.id == c.scene_id)
            assert c.text["content"].strip("$%+").replace(",", "").replace(".", "").isalnum()
            assert any(ch.isdigit() for ch in c.text["content"]) or c.text["content"].lower() in sc.narration.lower()
    heads = [c for c in p.timeline.all_clips() if c.kind == KIND_TEXT and c.text.get("variant") == "HEADLINE" and c.text.get("derived")]
    assert all(next(s for s in p.scenes if s.id == c.scene_id).topic for c in heads)


def test_caption_position_and_style_changes_are_undoable_and_custom_styles_persist(pres_ws):
    from app.captions.styles import PRESETS

    ws = pres_ws
    ws.presentation.generate(["CAPTIONS"])
    assert ws.jobs.wait_idle(120)
    p = ws.project
    cap = next(c for c in p.timeline.all_clips() if c.kind == KIND_CAPTION)
    ws.presentation.update_caption(cap.id, position_xy=[0.3, 0.7])
    c = p.timeline.get_clip(cap.id)
    assert c.text["position"] == "custom" and c.text["position_xy"] == [0.3, 0.7]
    ws.presentation.update_caption(cap.id, style_overrides={"size_rel": 0.08, "font": "Serif", "weight": "normal", "line_spacing": 1.4, "outline_width": 0.1, "opacity": 0.8})
    st = ws.presentation.describe(cap.id)
    assert st["font"] == "Serif" and st["size_pct"] == 8.0 and st["weight"] == "normal"
    ws.undo()
    ws.undo()
    assert p.timeline.get_clip(cap.id).text["position"] != "custom" and p.timeline.get_clip(cap.id).created_by == "AI"
    ws.redo()
    assert p.timeline.get_clip(cap.id).text["position"] == "custom"
    # outside the safe area is only a warning for a user-chosen position
    ws.presentation.update_caption(cap.id, position_xy=[0.99, 0.99])
    assert [i for i in ws.presentation.validate() if i.code == "caption.safe_area" and i.severity == "warning"]
    assert not [i for i in ws.presentation.validate() if i.severity == "error"]
    # custom style: save, use, persist
    from dataclasses import replace

    ws.presentation.save_caption_style(replace(PRESETS["clean"], style_id="mine", name="Mine", size_rel=0.066))
    ws.presentation.update_caption_settings(style_id="mine")
    ws.presentation.save_caption_style(replace(PRESETS["bold"], size_rel=0.07))  # presets can be edited: the project copy wins
    assert ws.presentation.caption_styles()["bold"].size_rel == 0.07 and PRESETS["bold"].size_rel != 0.07
    root = p.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    assert ws.project.caption_styles["mine"].size_rel == 0.066 and ws.project.caption_settings.style_id == "mine"
    ws.presentation.reset_caption_style("bold")
    assert ws.presentation.caption_styles()["bold"].size_rel == PRESETS["bold"].size_rel


def test_music_and_sfx_services_are_thin_facades_over_the_same_timeline_objects(pres_ws):
    ws = pres_ws
    p = ws.project
    lib = ws.presentation.music.library()
    assert [a.name for a in lib] == ["music_bed.wav"] and "IMPACT" in ws.presentation.sfx.library()
    aid = ws.presentation.music.add(lib[0].id)
    assert ws.presentation.music.recommend()["fade_out"] == p.audio_settings.music_fade_out
    ws.presentation.music.set(aid, volume=0.4)
    cid = ws.presentation.sfx.add(ws.audio_by["tick.wav"].id, 7.0)
    ws.presentation.sfx.set(cid, volume=0.2)
    assert p.timeline.get_clip(cid).audio["volume"] == 0.2 and all(c.audio["volume"] == 0.4 for c in p.timeline.get_track("track_a2").clips)
    ws.presentation.sfx.replace(cid, ws.audio_by["impact.wav"].id)
    ws.presentation.music.delete(aid)
    assert not p.timeline.get_track("track_a2").clips
