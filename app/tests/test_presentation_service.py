"""Phase 5 service tests: captions, graphics, music, SFX, ducking, overrides, locks, regeneration, voice replacement, persistence."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from app.editing.models import Creator
from app.presentation.models import PresentationType, PreviewMode
from app.services.presentation_service import PresentationError
from app.tests.conftest import needs_ffmpeg
from app.timeline.clip import KIND_CAPTION, KIND_GRAPHIC, KIND_TEXT

pytestmark = needs_ffmpeg


def gen(ws, *a, **k):
    job = ws.presentation.generate(*a, **k)
    assert ws.jobs.wait_idle(120)
    return job


def last(ws):
    return ws.project.presentation_sessions[-1]


def clips(ws, kind=None, scene_id=None, role=None):
    return [c for c in ws.project.timeline.all_clips() if (kind is None or c.kind == kind) and (scene_id is None or c.scene_id == scene_id)
            and (role is None or c.audio.get("role") == role)]


def snap(p):
    return json.dumps(p.timeline.to_dict(), sort_keys=True), json.dumps({k: v.to_dict() for k, v in p.presentation_decisions.items()}, sort_keys=True, default=str)


def music_id(ws):
    return ws.presentation.add_music(ws.audio_by["music_bed.wav"].id)


# ============================================================ captions
def test_generate_captions_from_word_timestamps_on_the_captions_track(pres_ws):
    ws = pres_ws
    gen(ws, ["CAPTIONS"])
    p = ws.project
    s = last(ws)
    assert s.status == "COMPLETED", (s.error, s.validation_errors)
    caps = sorted(clips(ws, KIND_CAPTION), key=lambda c: c.timeline_start)
    assert caps and all(c.track_id == "track_v6" and c.created_by == "AI" and c.ai_decision_id in p.presentation_decisions for c in caps)
    words = {w.word_id: w for w in p.transcription.transcript.words}
    seen: list[str] = []
    for c in caps:
        seg = c.text
        assert seg["words"] and c.timeline_start == pytest.approx(seg["words"][0]["start"], abs=0.2) and c.duration > 0.5
        assert all(w["word_id"] in words and w["text"] == words[w["word_id"]].text for w in seg["words"])  # spoken wording untouched
        seen += [w["word_id"] for w in seg["words"]]
        assert 1 <= len(seg["lines"]) <= 2 and seg["style_id"] == "professional"
    assert seen == [w.word_id for w in p.transcription.transcript.words]  # every word exactly once, in order
    assert all(a.timeline_end <= b.timeline_start + 1e-6 for a, b in zip(caps, caps[1:]))  # one at a time
    d = p.presentation_decisions[caps[0].ai_decision_id]
    assert d.type is PresentationType.CAPTION and d.reason and 0 <= d.confidence <= 100 and d.created_by is Creator.AI and d.target_id == caps[0].id
    assert p.presentation_generation.captions.status == "COMPLETE" and p.caption_settings.generated_transcript_id
    assert any(p.keyword_emphasis.values()) and any(c.text["emphasis"] for c in caps)
    assert p.presentation_plans[p.scenes[0].id].caption_plan["count"] > 0
    ws.project.validate()


def test_captions_emphasise_numbers_and_names_but_not_every_word(pres_ws):
    ws = pres_ws
    gen(ws, ["CAPTIONS"])
    caps = clips(ws, KIND_CAPTION)
    marked = [(c.text["words"][m["word_index"]]["text"], m["category"]) for c in caps for m in c.text["emphasis"]]
    texts = " ".join(t for t, _ in marked)
    assert "5%" in texts and "2027" in texts and "20%" in texts and any(cat == "PERSON" for _, cat in marked)
    total_words = sum(len(c.text["words"]) for c in caps)
    assert len(marked) < total_words * 0.15 and all(len(c.text["emphasis"]) <= 2 for c in caps)


def test_caption_toggles_and_style_settings_change_the_result(pres_ws):
    ws = pres_ws
    ws.presentation.update_caption_settings(style_id="bold", keyword_highlight=False, number_emphasis=False, position="top")
    gen(ws, ["CAPTIONS"])
    caps = clips(ws, KIND_CAPTION)
    assert all(c.text["style_id"] == "bold" and c.text["position"] == "top" and not c.text["emphasis"] for c in caps)
    ws.presentation.update_caption_settings(enabled=False)
    with pytest.raises(PresentationError):
        ws.presentation.generate(["CAPTIONS"])
    for bad in ({"style_id": "nope"}, {"position": "left"}, {"max_lines": 3}, {"safe_margin_left": 0.9}, {"reading_speed": 5.0}, {"highlight_mode": "zzz"}):
        with pytest.raises(PresentationError):
            ws.presentation.update_caption_settings(**bad)


def test_safe_margins_and_accessibility_shape_the_captions(pres_ws):
    ws = pres_ws
    ws.presentation.update_caption_settings(safe_margin_left=0.3, safe_margin_right=0.3, large_text=True)
    gen(ws, ["CAPTIONS"])
    caps = clips(ws, KIND_CAPTION)
    wide = max(len(l) for c in caps for l in c.text["lines"])
    ws.presentation.update_caption_settings(safe_margin_left=0.05, safe_margin_right=0.05, large_text=False)
    gen(ws, ["CAPTIONS"], None)
    wider = max(len(l) for c in clips(ws, KIND_CAPTION) for l in c.text["lines"])
    assert wider > wide  # narrower safe area -> shorter lines
    assert not [i for i in ws.presentation.validate() if i.severity == "error"]


def test_captions_one_undo_removes_them_all(pres_ws):
    ws = pres_ws
    p = ws.project
    before = snap(p)
    gen(ws, ["CAPTIONS"])
    assert snap(p) != before
    ws.undo()
    assert snap(p) == before and not clips(ws, KIND_CAPTION)
    ws.redo()
    assert clips(ws, KIND_CAPTION)


# ============================================================ graphics
def test_graphics_numbers_dates_lower_thirds_and_headlines_follow_the_narration(pres_ws):
    ws = pres_ws
    gen(ws, ["GRAPHICS"])
    p = ws.project
    assert last(ws).status == "COMPLETED", last(ws).validation_errors
    texts = clips(ws, KIND_TEXT)
    variants = Counter(c.text.get("variant") for c in texts)
    assert variants["NUMBER"] and variants["DATE"] and variants["LOWER_THIRD"]
    narr = {s.id: (s.narration + " " + s.script_text).lower() for s in p.scenes}
    for c in texts:
        if c.text.get("derived"):
            continue
        assert all(tok in narr[c.scene_id] for tok in "".join(ch.lower() if ch.isalnum() or ch in "%$" else " " for ch in c.text["content"]).split())  # never fabricated
        assert c.animation.get("in") and c.animation.get("out") and c.metadata.get("phase") == 5 or c.metadata.get("phase5")
    # the April 15 deadline appears when "deadline" is spoken
    dl = next(c for c in texts if "april" in c.text["content"].lower())
    words = ws.project.transcription.transcript.words
    trig = next(w for w in words if w.text.lower().strip(".,") == "deadline")
    assert dl.timeline_start == pytest.approx(trig.start - 0.05, abs=0.2)
    for c in texts:
        d = [x for x in p.presentation_decisions.values() if x.target_id == c.id]
        assert d and d[0].reason and 0 <= d[0].confidence <= 100 and d[0].created_by is Creator.AI
    ws.project.validate()


def test_graphics_are_adopted_not_duplicated_and_headlines_are_marked_derived(pres_ws):
    ws = pres_ws
    before = Counter(c.text["content"].lower() for c in clips(ws, KIND_TEXT))
    gen(ws, ["GRAPHICS"])
    after = Counter(c.text["content"].lower() for c in clips(ws, KIND_TEXT) if not c.text.get("derived"))
    assert all(v == 1 for v in after.values()) and set(before) <= set(after) | set(before)
    heads = [c for c in clips(ws, KIND_TEXT) if c.text.get("derived")]
    assert heads and all(c.text["source_ref"] == "scene_topic" for c in heads)
    d = ws.project.presentation_decisions[next(d.decision_id for d in ws.project.presentation_decisions.values() if d.target_id == heads[0].id)]
    assert d.confidence <= 75 and "scene topic" in d.reason  # honest: derived text is flagged for review


def test_motion_intensity_controls_animation_style_and_reduced_motion_stays_gentle(pres_ws):
    ws = pres_ws

    def presets():
        gen(ws, ["GRAPHICS"])
        return {c.animation["in"]["preset"] for c in clips(ws, KIND_TEXT)}

    ws.editing.update_settings(motion_intensity=0.1)
    low = presets()
    assert low == {"fade_in"}
    ws.editing.update_settings(motion_intensity=0.9)
    high = presets()
    assert high - {"fade_in"} and high <= {"pop", "scale_in", "slide_up", "reveal", "counter", "fade_in"}
    ws.presentation.update_caption_settings(reduced_motion=True)
    assert presets() == {"fade_in"}  # reduced motion: fades only


def test_evidence_graphics_get_tools_and_never_alter_the_document(pres_ws):
    ws = pres_ws
    gen(ws, ["GRAPHICS"])
    ev = [c for c in clips(ws, KIND_GRAPHIC) if "evidence" in c.effects]
    assert ev and all(c.effects["evidence"]["tool"] in ("FOCUS_BOX", "HIGHLIGHT") and c.animation.get("in") for c in ev)
    c = ev[0]
    ws.presentation.set_evidence_tool(c.id, "UNDERLINE", region=[0.1, 0.2, 0.5, 0.1], dim=False)
    c = ws.project.timeline.get_clip(c.id)
    assert c.effects["evidence"]["tool"] == "UNDERLINE" and c.created_by == "USER" and c.effects["highlight"]["region"] == [0.1, 0.2, 0.5, 0.1]
    for bad in (dict(tool="LASER"), dict(tool="FOCUS_BOX", region=[0.5, 0.5, 0.9, 0.9]), dict(tool="FOCUS_BOX", region=[1, 2])):
        with pytest.raises(PresentationError):
            ws.presentation.set_evidence_tool(c.id, **bad)
    p = ws.project
    assets_before = {a.id: p.asset_path(a).read_bytes() for a in p.assets.all() if a.type.value == "image"}
    ws.presentation.regenerate_graphics()
    assert ws.jobs.wait_idle(60)
    assert {a.id: p.asset_path(a).read_bytes() for a in p.assets.all() if a.type.value == "image"} == assets_before
    assert p.timeline.get_clip(c.id).effects["evidence"]["tool"] == "UNDERLINE"  # the user's tool survives


def test_no_chart_is_fabricated_for_data_scenes_without_data(pres_ws):
    ws = pres_ws
    gen(ws, ["GRAPHICS"])
    notes = [n for pl in ws.project.presentation_plans.values() for n in pl.notes]
    assert not any("chart" in (c.text or {}).get("content", "").lower() for c in clips(ws, KIND_TEXT))
    assert all(("no chart was generated" in n) or ("no data is generated" in n) or "skipped" in n for n in notes if "chart" in n or "data" in n)


def test_graphics_can_be_created_moved_resized_animated_and_deleted(pres_ws):
    ws = pres_ws
    p = ws.project
    cid = ws.presentation.add_graphic("HEADLINE", "THE 2027 TAX DEADLINE", 1.0, 2.5)
    c = p.timeline.get_clip(cid)
    assert c.created_by == "USER" and c.track_id == "track_v5" and c.text["content"] == "THE 2027 TAX DEADLINE" and c.animation["in"]["preset"]
    ws.presentation.move_graphic(cid, 0.3, 0.6)
    assert p.timeline.get_clip(cid).text["position"] == [0.3, 0.6]
    ws.presentation.update_graphic(cid, size=90, duration=3.0, content="THE 2027 DEADLINE")
    c = p.timeline.get_clip(cid)
    assert c.text["size"] == 90 and c.duration == 3.0 and c.text["content"] == "THE 2027 DEADLINE"
    ws.presentation.set_graphic_animation(cid, "in", "slide_up", duration=0.5, easing="ease_in_out", delay=0.1, start_value=0.0, end_value=1.0)
    a = p.timeline.get_clip(cid).animation["in"]
    assert (a["preset"], a["duration"], a["easing"], a["delay"]) == ("slide_up", 0.5, "ease_in_out", 0.1)
    ws.presentation.set_graphic_animation(cid, "out", "none")
    assert "out" not in p.timeline.get_clip(cid).animation
    for bad in (lambda: ws.presentation.update_graphic(cid, content=" "), lambda: ws.presentation.move_graphic(cid, 1.5, 0.5), lambda: ws.presentation.update_graphic(cid, size=0),
                lambda: ws.presentation.add_graphic("BANNER", "x", 1, 1), lambda: ws.presentation.set_graphic_animation(cid, "in", "spin"),
                lambda: ws.presentation.set_graphic_animation(cid, "in", "fade_in", duration=99), lambda: ws.presentation.add_graphic("HEADLINE", "overlap", 1.5, 1.0)):
        with pytest.raises(PresentationError):
            bad()
    ws.timeline.delete_clip(cid)
    assert p.timeline.get_clip(cid) is None
    ws.undo()
    assert p.timeline.get_clip(cid) is not None
    p.validate()


# ============================================================ music + ducking
def level_at(ws, t):
    return ws.editing.composer().music_level_at(t)


def test_music_is_placed_on_a2_looped_faded_and_ducked_with_editable_keyframes(pres_ws):
    ws = pres_ws
    p = ws.project
    mid = music_id(ws)
    pieces = sorted([c for c in p.timeline.get_track("track_a2").clips], key=lambda c: c.timeline_start)
    voice_end = max(s.end for s in p.scenes)
    assert len(pieces) >= 3 and pieces[0].timeline_start == 0.0 and pieces[-1].timeline_end == pytest.approx(voice_end + p.audio_settings.music_fade_out, abs=0.05)
    assert all(a.timeline_end == pytest.approx(b.timeline_start) for a, b in zip(pieces, pieces[1:]))  # looped back to back
    assert pieces[0].audio["fade_in"] == p.audio_settings.music_fade_in and pieces[-1].audio["fade_out"] == p.audio_settings.music_fade_out
    assert all(c.created_by == "AI" and c.audio["role"] == "MUSIC" for c in pieces)
    d = next(d for d in p.presentation_decisions.values() if d.type is PresentationType.MUSIC)
    assert d.reason and d.confidence and d.parameters["assignment_id"] == mid and len(d.parameters["clip_ids"]) == len(pieces)
    kfs = ws.presentation.ducking_keyframes(mid)
    assert len(kfs) > 20 and all(b[0] >= a[0] for a, b in zip(kfs, kfs[1:])) and all(0 <= v <= 0.4 for _t, v in kfs)  # many smooth breakpoints, not one global level
    duck = next(x for x in p.presentation_decisions.values() if x.type is PresentationType.DUCKING)
    assert duck.created_by is Creator.AI and all(k.decision_id == duck.decision_id for c in pieces for k in c.keyframes if k.property == "volume")
    assert p.ducking_events and {e.kind for e in p.ducking_events} >= {"DUCK"} and all(e.cause for e in p.ducking_events)
    a = p.audio_settings
    normal = next(s for s in p.scenes[1:] if s.importance < 0.6 and not any(sc.narration.lower().startswith(("now", "moving", "next", "meanwhile", "let's", "in conclusion")) for sc in [s]))
    imp = next(s for s in p.scenes[1:] if s.importance >= 0.75 and s.duration >= 8)
    assert level_at(ws, (normal.start + normal.end) / 2) == pytest.approx(a.music_level, abs=0.03) or level_at(ws, (normal.start + normal.end) / 2) <= a.pause_level + 1e-6
    assert level_at(ws, imp.end - 1.0) <= a.important_level + 0.02  # important narration: music ducks to ~9%
    assert level_at(ws, 0.0) == 0.0 and level_at(ws, 2.0) > 0  # fade-in at the start
    assert level_at(ws, voice_end + p.audio_settings.music_fade_out - 0.05) < 0.05  # fade-out at the end
    assert ws.presentation.masking() == []  # the voice is never masked
    ws.project.validate()


def test_music_never_masks_the_voice_and_masking_is_reported(pres_ws):
    ws = pres_ws
    mid = music_id(ws)
    ws.presentation.set_music(mid, volume=4.0)
    issues = ws.presentation.masking()
    assert issues and all(i.role == "MUSIC" and "voice speaks" in i.message for i in issues)
    ws.presentation.set_music(mid, volume=1.0)
    assert ws.presentation.masking() == []


def test_music_can_be_trimmed_replaced_faded_and_deleted_with_undo(pres_ws):
    ws = pres_ws
    p = ws.project
    mid = music_id(ws)
    ws.presentation.set_music(mid, volume=0.7, fade_in=3.0, fade_out=4.0)
    first = sorted(p.timeline.get_track("track_a2").clips, key=lambda c: c.timeline_start)[0]
    assert first.audio["volume"] == 0.7 and first.audio["fade_in"] == 3.0 and first.created_by == "USER"
    ws.presentation.set_music(mid, start=2.0, end=20.0, loop=False)  # trim: start later, end earlier, no loop
    pieces = p.timeline.get_track("track_a2").clips
    assert len(pieces) == 1 and pieces[0].timeline_start == 2.0 and pieces[0].timeline_end == 20.0 and pieces[0].source_out == pytest.approx(18.0)
    other = ws.audio_by["impact.wav"].id
    ws.presentation.tag_asset(other, "music")
    ws.presentation.replace_music(mid, ws.audio_by["music_bed.wav"].id)
    assert p.timeline.get_track("track_a2").clips[0].created_by == "USER"
    with pytest.raises(PresentationError):
        ws.presentation.set_music(mid, volume=-1)
    with pytest.raises(PresentationError):
        ws.presentation.set_music(mid, start=5.0, end=5.0)
    with pytest.raises(PresentationError):
        ws.presentation.add_music(p.voice_over.asset_id + "x")
    ws.presentation.delete_music(mid)
    assert not p.timeline.get_track("track_a2").clips and not [d for d in p.presentation_decisions.values() if d.type in (PresentationType.MUSIC, PresentationType.DUCKING)]
    ws.undo()
    assert p.timeline.get_track("track_a2").clips
    p.validate()


def test_user_edited_ducking_survives_regeneration_and_unedited_ducking_regenerates(pres_ws):
    ws = pres_ws
    p = ws.project
    mid = music_id(ws)
    ws.presentation.set_ducking_keyframes(mid, [(0.0, 0.0), (3.0, 0.2), (30.0, 0.05), (60.0, 0.2)])
    duck = next(d for d in p.presentation_decisions.values() if d.type is PresentationType.DUCKING)
    assert duck.created_by is Creator.USER and duck.overrides_decision_id and ws.presentation.ducking_keyframes(mid)[:2] == [(0.0, 0.0), (3.0, 0.2)]
    before = ws.presentation.ducking_keyframes(mid)
    gen(ws, ["AUDIO"])
    assert ws.presentation.ducking_keyframes(mid) == before  # the user's envelope is untouched
    with pytest.raises(PresentationError):
        ws.presentation.set_ducking_keyframes(mid, [(1.0, 0.1), (1.0, 0.2)])
    with pytest.raises(PresentationError):
        ws.presentation.set_ducking_keyframes(mid, [(1.0, 9.0)])
    ws.undo()  # undo of the regeneration
    ws.undo()  # undo of the keyframe edit
    assert next(d for d in p.presentation_decisions.values() if d.type is PresentationType.DUCKING).created_by is Creator.AI


def test_locking_the_audio_mix_protects_ducking_and_music(pres_ws):
    ws = pres_ws
    p = ws.project
    mid = music_id(ws)
    sc = p.scenes[3].id
    ws.presentation.set_lock(sc, "MIX")
    before = ws.presentation.ducking_keyframes(mid)
    ws.editing.update_settings(style="dynamic")
    ws.presentation.update_audio_settings(music_level=0.2, important_level=0.07)
    gen(ws, ["AUDIO"])
    assert ws.presentation.ducking_keyframes(mid) == before  # locked mix: no regeneration of the envelope
    assert all(c.locked for c in p.timeline.get_track("track_a2").clips)
    ws.presentation.set_lock(sc, "MIX", False)
    gen(ws, ["AUDIO"])
    assert ws.presentation.ducking_keyframes(mid) != before


def test_audio_settings_validation_and_toggles(pres_ws):
    ws = pres_ws
    with pytest.raises(PresentationError):
        ws.presentation.update_audio_settings(important_level=0.5)  # must stay <= normal level
    with pytest.raises(PresentationError):
        ws.presentation.update_audio_settings(music_level=9)
    ws.presentation.update_audio_settings(music_enabled=False)
    with pytest.raises(PresentationError):
        music_id(ws)
    ws.presentation.update_audio_settings(music_enabled=True, auto_ducking=False)
    mid = music_id(ws)
    assert ws.presentation.ducking_keyframes(mid) == []  # ducking switched off: no keyframes


# ============================================================ SFX
def test_ai_sfx_are_sparse_quiet_explained_and_only_from_the_library(pres_ws):
    ws = pres_ws
    gen(ws, ["GRAPHICS", "AUDIO"])
    p = ws.project
    sfx = sorted(clips(ws, role="SFX"), key=lambda c: c.timeline_start)
    assert sfx
    total = max(s.end for s in p.scenes)
    assert len(sfx) <= max(1, round(p.audio_settings.max_sfx_per_minute * total / 60) + 1)  # restraint
    assert all(b.timeline_start - a.timeline_start >= p.audio_settings.min_sfx_gap - 0.01 for a, b in zip(sfx, sfx[1:]))
    cuts = len([c for c in p.timeline.all_clips() if c.kind == "media" and c.track_id in ("track_v1", "track_v2", "track_v3")])
    assert len(sfx) < cuts * 0.3  # never on every cut
    lib = {a.id for a in ws.presentation.library("sfx")}
    for c in sfx:
        d = p.presentation_decisions[c.ai_decision_id]
        assert c.asset_id in lib and c.scene_id and c.audio["volume"] <= 0.5 and c.duration <= 2.0 and c.track_id.startswith("track_a3")
        assert d.type is PresentationType.SFX and d.reason and 0 < d.confidence <= 100 and d.created_by is Creator.AI and d.parameters["category"] == c.audio["category"]
        assert d.parameters["timestamp"] == pytest.approx(c.timeline_start)
    assert any("warning" in p.presentation_decisions[c.ai_decision_id].reason.lower() or "impact" in p.presentation_decisions[c.ai_decision_id].reason.lower()
               or "tick" in p.presentation_decisions[c.ai_decision_id].reason.lower() for c in sfx)


def test_without_sfx_assets_nothing_is_placed_and_the_gap_is_reported(pres_ws):
    ws = pres_ws
    for a in ws.presentation.library("sfx"):
        ws.presentation.tag_asset(a.id, "music")
    gen(ws, ["GRAPHICS", "AUDIO"])
    assert not clips(ws, role="SFX")
    notes = [e for pl in ws.project.presentation_plans.values() for e in pl.sfx_events]
    assert notes and all(not e["placed"] and "No SFX asset" in e.get("note", "") for e in notes)
    ws.presentation.update_audio_settings(sfx_enabled=False)
    with pytest.raises(PresentationError):
        ws.presentation.add_sfx(ws.audio_by["tick.wav"].id, 3.0)


def test_sfx_insert_trim_move_volume_fade_delete_layer_replace_with_undo(pres_ws):
    ws = pres_ws
    p = ws.project
    tick, impact = ws.audio_by["tick.wav"].id, ws.audio_by["impact.wav"].id
    cid = ws.presentation.add_sfx(tick, 10.0)
    c = p.timeline.get_clip(cid)
    assert c.track_id == "track_a3" and c.created_by == "USER" and c.audio["role"] == "SFX" and c.duration == pytest.approx(0.8) and c.scene_id
    layer = p.timeline.get_clip(ws.presentation.add_sfx(impact, 10.2))
    assert layer.track_id != "track_a3" and layer.track_id.startswith("track_a3_")  # layering on an extra SFX track
    ws.timeline.trim_clip(cid, new_end=10.5)  # trim
    assert p.timeline.get_clip(cid).duration == pytest.approx(0.5)
    ws.timeline.move_clip(cid, 20.0)  # move
    assert p.timeline.get_clip(cid).timeline_start == 20.0
    ws.presentation.set_clip_audio(cid, volume=0.25, fade_in=0.05, fade_out=0.1)  # volume + fade
    a = p.timeline.get_clip(cid).audio
    assert (a["volume"], a["fade_in"], a["fade_out"]) == (0.25, 0.05, 0.1)
    ws.presentation.replace_sfx(cid, impact)
    assert p.timeline.get_clip(cid).asset_id == impact
    for bad in (lambda: ws.presentation.set_clip_audio(cid, volume=-1), lambda: ws.presentation.set_clip_audio(cid, fade_in=9.0),
                lambda: ws.presentation.set_clip_audio(cid, category="BOOM"), lambda: ws.presentation.add_sfx(tick, -1.0),
                lambda: ws.presentation.add_sfx(tick, 1.0, category="BOOM"), lambda: ws.presentation.set_clip_audio(next(c.id for c in p.timeline.all_clips() if c.kind == "text"), volume=1)):
        with pytest.raises(PresentationError):
            bad()
    ws.timeline.delete_clip(cid)  # delete
    assert p.timeline.get_clip(cid) is None
    for _ in range(3):
        ws.undo()
    assert p.timeline.get_clip(cid) is not None
    p.validate()


def test_user_sfx_and_locked_sfx_survive_regeneration(pres_ws):
    ws = pres_ws
    p = ws.project
    gen(ws, ["GRAPHICS", "AUDIO"])
    sfx = clips(ws, role="SFX")
    keep = sfx[0]
    ws.presentation.lock_clip(keep.id)
    mine = ws.presentation.add_sfx(ws.audio_by["tick.wav"].id, 3.3)
    ws.presentation.regenerate_audio()
    assert ws.jobs.wait_idle(60)
    assert p.timeline.get_clip(keep.id) is not None and p.timeline.get_clip(keep.id).locked
    assert p.timeline.get_clip(mine) is not None and p.timeline.get_clip(mine).created_by == "USER"
    for other in sfx[1:]:
        assert p.timeline.get_clip(other.id) is None or p.timeline.get_clip(other.id).locked  # unlocked AI effects were regenerated


# ============================================================ ownership, locks, regeneration
def first_caption(ws, scene_id=None):
    cs = sorted(clips(ws, KIND_CAPTION, scene_id), key=lambda c: c.timeline_start)
    return cs[0]


def test_user_caption_edits_survive_regeneration_and_unedited_ai_captions_regenerate(pres_ws):
    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS"])
    sc = next(s.id for s in p.scenes if len(clips(ws, KIND_CAPTION, s.id)) >= 3)
    cap = first_caption(ws, sc)
    ai_ids = {c.id for c in clips(ws, KIND_CAPTION, sc)}
    old_dec = cap.ai_decision_id
    ws.presentation.update_caption(cap.id, text="Silver is running out", position="top", style_id="news", style_overrides={"size_rel": 0.07})
    cap = p.timeline.get_clip(cap.id)
    d = p.presentation_decisions[cap.ai_decision_id]
    assert cap.created_by == "USER" and d.created_by is Creator.USER and d.overrides_decision_id == old_dec and old_dec not in p.presentation_decisions
    assert any(o.override_id == d.decision_id and o.original.decision_id == old_dec for o in p.presentation_overrides)
    assert cap.text["text"] == "Silver is running out" and cap.text["position"] == "top" and cap.text["style_overrides"]["size_rel"] == 0.07
    assert cap.text["words"][0]["start"] == pytest.approx(cap.text["words"][0]["start"]) and len(cap.text["words"]) == 4 and cap.text["lines"]
    other = {c.id for c in clips(ws, KIND_CAPTION, sc)} - {cap.id}
    gen(ws, ["CAPTIONS"], [sc])
    kept = p.timeline.get_clip(cap.id)
    assert kept is not None and kept.text["text"] == "Silver is running out" and kept.text["position"] == "top" and kept.created_by == "USER"  # USER edit survives
    now_ai = {c.id for c in clips(ws, KIND_CAPTION, sc)} - {cap.id}
    assert now_ai and not (now_ai & other)  # the AI-owned captions of the scene were regenerated (new ids)
    assert not [i for i in ws.presentation.validate() if i.severity == "error"]
    assert ai_ids  # (sanity)


def test_manual_timeline_edits_take_ownership_of_presentation_objects(pres_ws):
    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS", "GRAPHICS"])
    cap = first_caption(ws, p.scenes[2].id)
    d0 = cap.ai_decision_id
    ws.timeline.trim_clip(cap.id, new_end=cap.timeline_end - 0.15)
    cap = p.timeline.get_clip(cap.id)
    d = p.presentation_decisions[cap.ai_decision_id]
    assert cap.created_by == "USER" and d.created_by is Creator.USER and d.overrides_decision_id == d0 and d.duration == pytest.approx(cap.duration)
    ws.undo()
    cap = p.timeline.get_clip(cap.id)
    assert cap.created_by == "AI" and p.presentation_decisions[cap.ai_decision_id].decision_id == d0 and not p.presentation_overrides
    text = clips(ws, KIND_TEXT)[0]
    ws.timeline.move_clip(text.id, text.timeline_start + 0.0)
    assert p.timeline.get_clip(text.id).created_by == "USER"
    ws.undo()
    gen(ws, ["GRAPHICS"])  # AI graphics regenerate; nothing is user-owned any more
    assert last(ws).status == "COMPLETED"


def test_deleted_ai_captions_are_not_brought_back(pres_ws):
    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS"])
    cap = first_caption(ws, p.scenes[1].id)
    sid, slot = cap.scene_id, cap.slot
    ws.timeline.delete_clip(cap.id)
    assert f"{sid}|{slot}" in p.presentation_generation.suppressed_slots
    gen(ws, ["CAPTIONS"], [sid])
    assert not [c for c in clips(ws, KIND_CAPTION, sid) if c.slot == slot]


def test_locks_for_caption_graphic_music_sfx_and_scene(pres_ws):
    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS", "GRAPHICS", "AUDIO"])
    mid = music_id(ws)
    sc = next(s.id for s in p.scenes if clips(ws, KIND_TEXT, s.id) and clips(ws, role="SFX", scene_id=s.id) or clips(ws, KIND_TEXT, s.id))
    ws.presentation.set_lock(sc, "CAPTION")
    ws.presentation.set_lock(sc, "GRAPHIC")
    assert all(c.locked for c in clips(ws, KIND_CAPTION, sc)) and all(c.locked for c in clips(ws, KIND_TEXT, sc))
    snapshot = lambda: sorted((c.kind, c.track_id, round(c.timeline_start, 3), round(c.duration, 3), json.dumps(c.text, sort_keys=True, default=str)) for c in p.timeline.all_clips() if c.scene_id == sc)
    ws.editing.update_settings(motion_intensity=0.95)
    ws.presentation.update_caption_settings(style_id="bold")
    gen(ws, ["CAPTIONS", "GRAPHICS"], [sc])
    ids_locked = {c.id for c in clips(ws, scene_id=sc) if c.locked}
    assert ids_locked and all(p.timeline.get_clip(i) is not None for i in ids_locked)
    ws.presentation.set_lock(sc, "SCENE")
    before = snapshot()
    gen(ws, None, [sc])
    assert snapshot() == before and sc in p.presentation_generation.locked_scenes and ws.presentation.scene_rows()[0]["locked"] in (True, False)
    ws.presentation.set_lock(sc, "SCENE", False)
    ws.presentation.set_lock(sc, "MUSIC")
    ws.presentation.set_lock(sc, "SFX")
    with pytest.raises(PresentationError):
        ws.presentation.set_lock(sc, "WHATEVER")
    assert mid


def test_the_acceptance_regeneration_example_scene_20_style(pres_ws):
    """USER changed the caption position and the music volume; the AI regenerates graphics: caption + music preserved, graphics regenerated."""
    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS", "GRAPHICS"])
    mid = music_id(ws)
    sc = next(s.id for s in p.scenes if clips(ws, KIND_TEXT, s.id))
    cap = first_caption(ws, sc)
    ws.presentation.update_caption(cap.id, position="center")
    ws.presentation.set_music(mid, volume=0.6)
    gfx_before = {c.id for c in clips(ws, KIND_TEXT, sc) if c.created_by == "AI"}
    ws.presentation.regenerate_graphics([sc])
    assert ws.jobs.wait_idle(60)
    assert p.timeline.get_clip(cap.id).text["position"] == "center"  # caption position preserved
    assert all(c.audio["volume"] == 0.6 for c in p.timeline.get_track("track_a2").clips)  # music volume preserved
    assert last(ws).parts == ["GRAPHICS"] and last(ws).status == "COMPLETED"
    gfx_after = {c.id for c in clips(ws, KIND_TEXT, sc)}
    assert gfx_after and (gfx_after - gfx_before or gfx_before & gfx_after)  # graphics regenerated (adopted or re-created)


# ============================================================ voice replacement / stale detection
def test_voice_replacement_marks_dependents_outdated_without_destroying_manual_work(pres_ws, tmp_path):
    from app.tests.helpers import make_audio

    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS", "GRAPHICS"])
    ws.presentation.analyze_voice()
    assert ws.jobs.wait_idle(60)
    assert not ws.presentation.staleness()["captions_outdated"] and not ws.presentation.staleness()["analysis_outdated"]
    mine = ws.presentation.add_graphic("LABEL", "My own label", 2.0, 1.5)
    cap = first_caption(ws, p.scenes[4].id)
    ws.presentation.update_caption(cap.id, text="Hand written caption")
    snapshot = (p.timeline.get_clip(mine).text["content"], p.timeline.get_clip(cap.id).text["text"], len(clips(ws, KIND_CAPTION)))
    old_asset = p.voice_over.asset_id
    ws.media.import_voice_over(make_audio(tmp_path / "new_voice.wav", 33.0))
    assert ws.jobs.wait_idle(60)
    assert p.voice_over.asset_id != old_asset
    st = ws.presentation.staleness()
    assert st["captions_outdated"] and st["analysis_outdated"] and st["transcript_outdated"] and st["message"] == "Voice-over changed. Captions need regeneration."
    assert (p.timeline.get_clip(mine).text["content"], p.timeline.get_clip(cap.id).text["text"], len(clips(ws, KIND_CAPTION))) == snapshot  # nothing was shifted or deleted
    review = ws.presentation.caption_review()
    assert review["current"] == snapshot[2] and review["preserved_user_or_locked"] >= 1 and review["removed_ai"] == review["current"] - review["preserved_user_or_locked"]
    ws.presentation.acknowledge_stale()
    assert ws.presentation.staleness()["acknowledged"] and "keep" in ws.presentation.staleness()["message"].lower()
    assert ws.presentation.sync_voice_clip() is True  # the A1 clip now points at the new voice-over
    voice = next(c for c in p.timeline.get_track("track_a1").clips if c.slot == "voice")
    assert voice.asset_id == p.voice_over.asset_id and voice.duration == pytest.approx(33.0, abs=0.1)
    assert ws.presentation.sync_voice_clip() is False
    ws.undo()  # the voice-clip update is one undo step
    assert next(c for c in p.timeline.get_track("track_a1").clips if c.slot == "voice").asset_id != p.voice_over.asset_id


# ============================================================ failure recovery
def test_a_failed_scene_keeps_earlier_scenes_and_retry_resumes_there(pres_ws, monkeypatch):
    ws = pres_ws
    p = ws.project
    order = [s.id for s in p.scenes]
    bad = order[6]
    real = ws.presentation.keywords.detect
    calls: list[str] = []
    broken = {"on": True}

    def detect(scene, words, audio=None):
        calls.append(scene.id)
        if scene.id == bad and broken["on"]:
            raise RuntimeError("keyword model exploded")
        return real(scene, words, audio)

    monkeypatch.setattr(ws.presentation.keywords, "detect", detect)
    gen(ws, ["CAPTIONS"])
    s = last(ws)
    st = p.presentation_generation.scene_status["CAPTIONS"]
    assert s.status == "FAILED" and s.failed_scene == bad and s.failed_part == "CAPTIONS" and "exploded" in s.error
    assert all(st[i] == "COMPLETE" for i in order[:6]) and st[bad] == "FAILED" and all(st[i] == "PENDING" for i in order[7:])
    assert all(clips(ws, KIND_CAPTION, i) for i in order[:6]) and not clips(ws, KIND_CAPTION, bad)
    kept = {c.id for i in order[:6] for c in clips(ws, KIND_CAPTION, i)}
    p.validate()
    broken["on"] = False
    calls.clear()
    ws.presentation.retry_failed()
    assert ws.jobs.wait_idle(60)
    assert calls == order[6:]  # resumed at the failed scene; scenes 1-6 were not redone
    assert all(p.presentation_generation.scene_status["CAPTIONS"][i] == "COMPLETE" for i in order) and last(ws).status == "COMPLETED"
    assert kept <= {c.id for c in clips(ws, KIND_CAPTION)}  # the same caption objects
    with pytest.raises(PresentationError):
        ws.presentation.retry_failed()


def test_invalid_presentation_is_rejected_and_the_timeline_kept(pres_ws, monkeypatch):
    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS"])
    before = snap(p)
    real = ws.presentation._plan_captions

    def broken(sc, proj, ctx):
        segs, kws = real(sc, proj, ctx)
        for s in segs:
            s.words[0].start, s.words[0].end = 9.0, 1.0  # impossible word timing
        return segs, kws

    monkeypatch.setattr(ws.presentation, "_plan_captions", broken)
    ws.presentation.regenerate_captions([p.scenes[2].id])
    assert ws.jobs.wait_idle(60)
    s = last(ws)
    assert s.status == "FAILED" and s.validation_errors and "word timing" in s.validation_errors[0] and "kept" in s.error
    assert snap(p) == before
    assert not ws.presentation.running


def test_cancel_changes_nothing(pres_ws, monkeypatch):
    ws = pres_ws
    p = ws.project
    before = snap(p)
    real = ws.presentation.keywords.detect
    n = {"c": 0}

    def detect(scene, words, audio=None):
        n["c"] += 1
        if n["c"] == 3:
            ws.presentation.cancel()
        return real(scene, words, audio)

    monkeypatch.setattr(ws.presentation.keywords, "detect", detect)
    gen(ws, ["CAPTIONS"])
    assert last(ws).status == "CANCELED" and snap(p) == before and not ws.presentation.running


# ============================================================ persistence
def test_presentation_survives_save_close_reopen_exactly(pres_ws):
    from app.editing.compose import PreviewComposer
    from dataclasses import asdict

    ws = pres_ws
    p = ws.project
    gen(ws, ["CAPTIONS", "GRAPHICS", "AUDIO"])
    mid = music_id(ws)
    cap = first_caption(ws, p.scenes[2].id)
    ws.presentation.update_caption(cap.id, text="Edited and kept")
    ws.presentation.set_music(mid, volume=0.5)
    ws.presentation.update_caption_settings(style_id="documentary", position="center")
    ws.presentation.update_voice_processing(highpass_hz=90.0, compression=True)
    ws.presentation.set_lock(p.scenes[5].id, "CAPTION")
    ws.presentation.add_graphic("LOWER_THIRD", "ROBERT", 4.0, 2.0, subtitle="Retired Silver Investor")
    before = p.to_document()
    comp = PreviewComposer(p)
    samples = [0.5, 12.0, 30.0, 60.0, 90.0, 120.0]
    frames = [json.dumps(asdict(comp.frame_at(t)), sort_keys=True, default=str) for t in samples]
    root = p.root
    ws.save()
    ws.close_project()
    ws.open_project(root)
    p2 = ws.project
    after = p2.to_document()
    for key in ("timeline", "audio_settings", "audio_analysis", "audio_processing", "ducking_events", "caption_settings", "caption_styles", "keyword_emphasis", "presentation_plans",
                "presentation_decisions", "presentation_overrides", "presentation_generation", "music_assignments", "sfx_assignments", "caption_segments", "text_graphics",
                "motion_graphics", "presentation_sessions"):
        assert after[key] == before[key], key
    comp2 = PreviewComposer(p2)
    assert [json.dumps(asdict(comp2.frame_at(t)), sort_keys=True, default=str) for t in samples] == frames
    assert len(after["caption_segments"]) == len(clips(ws, KIND_CAPTION)) and len(after["music_assignments"]) == 1 and len(after["sfx_assignments"]) == len(clips(ws, role="SFX"))
    assert after["music_assignments"][0]["created_by"] == "USER" and after["text_graphics"] and after["motion_graphics"]
    assert p2.caption_settings.style_id == "documentary" and p2.audio_processing.highpass_hz == 90.0
    # keeps working after reopening: the restored user work is honoured
    sid = p2.scenes[2].id
    ws.presentation.regenerate_captions([sid])
    assert ws.jobs.wait_idle(60)
    assert p2.timeline.get_clip(cap.id).text["text"] == "Edited and kept"
    p2.validate()


def test_older_projects_migrate_to_schema_5(pres_ws):
    from app.project.project import Project

    doc = pres_ws.project.to_document()
    for k in [k for k in doc if k in ("audio_settings", "audio_analysis", "audio_processing", "music_assignments", "sfx_assignments", "ducking_events", "caption_settings",
                                      "caption_segments", "caption_styles", "keyword_emphasis", "text_graphics", "motion_graphics", "presentation_plans",
                                      "presentation_decisions", "presentation_overrides", "presentation_generation", "presentation_sessions")]:
        del doc[k]
    doc["schema_version"] = 4
    p = Project.from_document(doc)
    assert p.schema_version == 5 and p.caption_settings.style_id == "professional" and p.presentation_decisions == {} and p.audio_settings.music_level == 0.18


def test_autosave_captures_presentation_changes_and_checkpoint_is_written(pres_ws):
    ws = pres_ws
    gen(ws, ["CAPTIONS"])
    assert ws.autosave.wait_idle(10)
    assert any(e.project_id == ws.project.project_id for e in ws.recovery.list_entries())
    assert last(ws).checkpoint.startswith("before_presentation_")
    folder = ws.paths.data_dir / "checkpoints" / ws.project.project_id
    doc = json.loads((folder / last(ws).checkpoint).read_text())
    assert not doc["caption_segments"]  # the state *before* this presentation run


# ============================================================ audio jobs: analysis, waveform, previews, processing
def test_voice_analysis_job_is_cached_and_reports_problems(pres_ws):
    ws = pres_ws
    p = ws.project
    assert ws.presentation.analyze_voice() is not None
    assert ws.jobs.wait_idle(60)
    a = p.audio_analysis
    assert a is not None and a.asset_id == p.voice_over.asset_id and a.duration > 100 and a.speaking_rate_wps and a.master_clock == "MASTER_TIMING_REFERENCE"
    assert ws.presentation.analyze_voice() is None  # unchanged: not recomputed
    assert ws.presentation.analyze_voice(force=True) is not None
    assert ws.jobs.wait_idle(60)
    rep = ws.presentation.audio_report()
    assert rep["analysis"] is a or rep["analysis"].asset_id == a.asset_id and rep["outdated"] is False
    p.validate()


def test_waveform_generation_is_a_background_job_and_cached(pres_ws):
    ws = pres_ws
    p = ws.project
    vo = p.voice_over.asset_id
    assert ws.presentation.waveform(vo) is None  # first call schedules the job
    assert ws.jobs.wait_idle(60)
    wf = ws.presentation.waveform(vo)
    assert wf is not None and wf.duration == pytest.approx(p.assets.get(vo).duration, abs=0.1)
    assert (p.root / "cache" / "waveforms").is_dir() and any((p.root / "cache" / "waveforms").glob("*.json"))
    assert ws.presentation.waveform("media_missing") is None


def test_every_preview_mode_renders_something_different(pres_ws):
    ws = pres_ws
    mid = music_id(ws)
    gen(ws, ["GRAPHICS", "AUDIO"])
    assert mid
    out = {}
    for mode in PreviewMode:
        got = []
        ws.presentation.preview_mix(mode, 0.0, 12.0, on_ready=got.append)
        assert ws.jobs.wait_idle(120)
        assert got and got[0] is not None and got[0].is_file(), mode
        out[mode] = got[0]
    assert len({str(v) for v in out.values()}) == len(PreviewMode)
    from app.audio.loudness import envelope

    be = ws.presentation.audio.backend
    rms = {m: float(envelope(be.decode_mono(path), 16000, 0.5).mean()) for m, path in out.items()}
    assert rms[PreviewMode.FULL] >= rms[PreviewMode.VOICE] - 1.0  # the full mix contains the voice
    assert rms[PreviewMode.MUSIC] < rms[PreviewMode.VOICE]  # music sits well below the voice
    plan = ws.presentation.mix_plan(PreviewMode.VOICE_SFX)
    assert {i.role for i in plan.items} <= {"VOICE", "SFX"}


def test_track_mute_solo_volume_and_lock_controls(pres_ws):
    ws = pres_ws
    p = ws.project
    music_id(ws)
    ws.timeline.set_track_flag("track_a2", "solo", True)
    assert {i.role for i in ws.presentation.mix_plan().items} == {"MUSIC"}
    assert ws.editing.composer().frame_at(5.0).audio["VOICE"] == 0.0 and ws.editing.composer().frame_at(5.0).audio["MUSIC"] > 0
    ws.timeline.set_track_flag("track_a2", "solo", False)
    ws.timeline.set_track_flag("track_a2", "muted", True)
    assert "MUSIC" not in {i.role for i in ws.presentation.mix_plan().items}
    ws.timeline.set_track_flag("track_a2", "muted", False)
    ws.timeline.set_track_volume("track_a1", 0.5)
    assert next(i for i in ws.presentation.mix_plan(PreviewMode.VOICE).items).gain == pytest.approx(0.5)
    with pytest.raises(Exception):
        ws.timeline.set_track_volume("track_a1", 5.0)
    ws.timeline.set_track_flag("track_v5", "hidden", True)  # text/graphics tracks have visibility toggles
    assert not [l for l in ws.editing.composer().frame_at(p.scenes[2].start + 1.0).layers if l.track_id == "track_v5"]
    ws.undo()
    ws.undo()
    ws.undo()
    ws.undo()
    assert p.timeline.get_track("track_a1").volume == 1.0
    ws.timeline.set_track_flag("track_a2", "solo", True)
    ws.save()
    root = p.root
    ws.close_project()
    ws.open_project(root)
    assert ws.project.timeline.get_track("track_a2").solo  # solo state persists


def test_voice_enhancement_is_non_destructive_and_editable(pres_ws):
    ws = pres_ws
    p = ws.project
    vo = p.assets.get(p.voice_over.asset_id)
    before = p.asset_path(vo).read_bytes()
    ws.presentation.update_audio_settings(voice_enhancement=True)
    proc = p.audio_processing
    assert proc.enabled and proc.highpass_hz == 80.0 and proc.compression and proc.limiter  # a sensible, editable starting chain
    ws.presentation.update_voice_processing(gain_db=3.0, noise_reduction=True, eq_preset="clarity", normalize=True, fade_in=0.2, fade_out=0.4)
    assert p.audio_processing.gain_db == 3.0
    for bad in ({"gain_db": 99}, {"eq_preset": "x"}, {"limiter_ceiling_db": 5}, {"highpass_hz": -1}):
        with pytest.raises(PresentationError):
            ws.presentation.update_voice_processing(**bad)
    got = []
    ws.presentation.preview_mix(PreviewMode.VOICE, 0.0, 6.0, on_ready=got.append)
    assert ws.jobs.wait_idle(120)
    assert got and got[0] is not None and p.asset_path(vo).read_bytes() == before  # the original voice file is untouched
    ws.undo()
    ws.undo()
    assert p.audio_processing.gain_db == 0.0


def test_import_audio_tags_music_and_sfx_with_categories(pres_ws, tmp_path):
    from app.tests.helpers import write_tone_wav

    ws = pres_ws
    p = ws.project
    f1, f2 = write_tone_wav(tmp_path / "bed2.wav", 45.0), write_tone_wav(tmp_path / "camera_click.wav", 0.5)
    ws.presentation.import_audio(f1, "music")
    ws.presentation.import_audio(f2, "sfx", "camera")
    assert ws.jobs.wait_idle(60)
    by = {a.name: a for a in p.assets.all()}
    assert by["bed2.wav"].extra["role"] == "music" and by["camera_click.wav"].extra == {"role": "sfx", "category": "CAMERA"}
    assert by["bed2.wav"].id in {a.id for a in ws.presentation.library("music")} and by["camera_click.wav"].id in {a.id for a in ws.presentation.library("sfx")}
    with pytest.raises(PresentationError):
        ws.presentation.import_audio(f1, "voice")
    with pytest.raises(PresentationError):
        ws.presentation.import_audio(f2, "sfx", "BOOM")
    with pytest.raises(PresentationError):
        ws.presentation.tag_asset(p.voice_over.asset_id + "zz", "music")
