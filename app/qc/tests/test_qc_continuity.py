"""Visual repetition and continuity."""

from __future__ import annotations

from PIL import Image

from app.analysis.models import Entity, EntityType, VisualIntent, VisualType
from app.media.asset import SourceType
from app.qc.continuity_checker import ContinuityChecker
from app.qc.severity import Severity
from app.qc.tests.qc_helpers import add_asset, add_clip, add_scene, codes, find, narrate, new_project, qc_ctx, run_checker
from app.timeline.keyframes import Keyframe

SECONDS = 10.0


def sequence(tmp_path, rows, *, kind="video", by="AI", intents=None, entities=None):
    """``rows``: (narration, asset key, asset name). One scene of 10 s per row; rows with the same key share one asset."""
    p = new_project(tmp_path, seconds=SECONDS * len(rows))
    assets: dict[str, object] = {}
    scenes, clips = [], []
    for i, (text, key, name) in enumerate(rows):
        s = add_scene(p, i * SECONDS, (i + 1) * SECONDS, text, topic=text[:40])
        scenes.append(s)
        if key not in assets:
            a = add_asset(p, f"{name}.{'mp4' if kind == 'video' else 'png'}", kind, duration=60 if kind == "video" else None)
            assets[key] = a
    narrate(p)
    for i, s in enumerate(scenes):
        p.visual_intents[s.id] = VisualIntent(s.id, (intents or {}).get(i, VisualType.LITERAL), primary_subject=" ".join(rows[i][0].split()[:3]))  # the subject comes from the narration, not from the picture
        if entities and i in entities:
            s.entities = [Entity(e, t) for e, t in entities[i]]
        clips.append(add_clip(p, "track_v1", assets[rows[i][1]], i * SECONDS, SECONDS, scene=s, source_in=5.0, created_by=by))
    return p, scenes, clips, assets


def run(p, **kw):
    return run_checker(ContinuityChecker(), qc_ctx(p, **kw))


GOOD = [("The Federal Reserve raised interest rates again this week.", "a", "federal reserve building"),
        ("Higher rates push investors toward silver and other metals.", "b", "silver bars investors metals"),
        ("Industrial demand for silver keeps rising at the refinery level.", "c", "silver refinery industrial demand"),
        ("Solar panel makers now use silver in every cell they produce.", "d", "solar panel silver cell factory")]


def filler(n, prefix="x"):
    return [(f"Topic {i} is about subject {prefix}{i} and its details.", f"{prefix}{i}", f"subject {prefix}{i} details") for i in range(n)]


# ---------------------------------------------------------------- repetition
def test_clean_sequence_has_no_issue_and_reports_metrics(tmp_path):
    p, *_ = sequence(tmp_path, GOOD)
    out = run(p)
    assert out.issues == []
    assert out.metrics["repetition_score"] == 0.0 and out.metrics["scenes_checked"] == 4 and "continuity_score" in out.metrics and out.metrics["per_asset"] == {}


def test_same_image_in_adjacent_scenes(tmp_path):
    rows = filler(1, "p") + [("Silver supply is falling.", "s", "silver mine pit"), ("Silver supply keeps falling.", "s", "silver mine pit")] + filler(1, "q")
    p, scenes, clips, assets = sequence(tmp_path, rows, kind="image")
    out = run(p)
    i = find(out, "visual.repetition_adjacent")
    assert len(i) == 1 and i[0].severity in (Severity.WARNING, Severity.NOTICE) and i[0].confidence <= 90 and "silver mine pit" in i[0].description
    a = assets["s"]
    assert out.metrics["per_asset"][a.id]["uses"] == 2 and out.metrics["per_asset"][a.id]["score"] > 0 and out.metrics["repetition_score"] > 0
    assert i[0].fix.kind == "visual.search_again" and not i[0].auto_fix_safe


def test_heavy_reuse_far_apart_is_flagged_with_a_score(tmp_path):
    rows = []
    for k in range(5):
        rows.append(("Silver supply is falling.", "s", "silver mine pit"))
        rows += filler(3, f"g{k}")
    p, scenes, clips, assets = sequence(tmp_path, rows, kind="image")
    out = run(p)
    i = find(out, "visual.repetition")
    assert len(i) == 1 and i[0].severity is Severity.WARNING and "5 times" in i[0].description and "Repetition score" in i[0].description
    assert out.metrics["per_asset"][assets["s"].id]["uses"] == 5


def test_different_parts_of_one_video_are_not_a_repeat(tmp_path):
    rows = []
    for k in range(5):
        rows.append((f"Segment {k} of the story.", "s", "silver mine footage"))
        rows += filler(3, f"g{k}")
    p, scenes, clips, assets = sequence(tmp_path, rows)
    for n, c in enumerate([c for c in clips if c.asset_id == assets["s"].id]):
        c.source_in, c.source_out = n * 100.0, n * 100.0 + 10.0
    assert not [c for c in codes(run(p)) if c.startswith("visual.repetition")]
    for c in [c for c in clips if c.asset_id == assets["s"].id]:
        c.source_in, c.source_out = 5.0, 15.0  # the same stretch every time
    assert find(run(p), "visual.repetition")


def test_threshold_follows_sensitivity(tmp_path):
    rows = []
    for k in range(3):
        rows.append(("Silver supply is falling.", "s", "silver mine pit"))
        rows += filler(3, f"g{k}")
    p, *_ = sequence(tmp_path, rows, kind="image")
    assert find(run(p), "visual.repetition") == []  # three uses are within the default limit
    p.qc_settings.repetition.sensitivity = 1.0
    assert len(find(run(p), "visual.repetition")) == 1


def test_intentional_reuse_is_not_flagged(tmp_path):
    def rows():
        return filler(1, "p") + [("Silver supply is falling.", "s", "company logo"), ("Silver supply keeps falling.", "s", "company logo")] + filler(1, "q")

    # declared by the user in the settings
    p, scenes, clips, assets = sequence(tmp_path / "a", rows(), kind="image")
    p.qc_settings.repetition.intentional_assets = [assets["s"].id]
    out = run(p)
    assert not [c for c in codes(out) if c.startswith("visual.repetition")] and out.metrics["per_asset"][assets["s"].id]["intentional"]
    # the editing decisions give a reason
    p, scenes, clips, assets = sequence(tmp_path / "b", rows(), kind="image")
    clips[2].metadata["reuse_reason"] = "callback to the opening"
    assert not [c for c in codes(run(p)) if c.startswith("visual.repetition")]
    # the user put it there
    p, scenes, clips, assets = sequence(tmp_path / "c", rows(), kind="image")
    clips[2].created_by = "USER"
    assert not [c for c in codes(run(p)) if c.startswith("visual.repetition")]
    # a recurring chart for the same company
    p, scenes, clips, assets = sequence(tmp_path / "d", rows(), kind="image", intents={1: VisualType.DATA, 2: VisualType.DATA},
                                        entities={1: [("Acme Mining", EntityType.COMPANY)], 2: [("Acme Mining", EntityType.COMPANY)]})
    assert not [c for c in codes(run(p)) if c.startswith("visual.repetition")]
    # a callback in the narration
    r = rows()
    r[2] = ("As we saw earlier, silver supply keeps falling.", "s", "company logo")
    p, *_ = sequence(tmp_path / "e", r, kind="image")
    assert not [c for c in codes(run(p)) if c.startswith("visual.repetition")]


def test_same_framing_again_is_a_notice(tmp_path):
    rows = [("Silver supply is falling.", "s", "silver mine pit")] + filler(3, "m") + [("Silver supply keeps dropping.", "s", "silver mine pit")] + filler(1, "n")
    p, *_ = sequence(tmp_path, rows, kind="image")
    p.qc_settings.repetition.min_gap_seconds = 5.0
    p.qc_settings.repetition.adjacent_scenes = 1
    i = find(run(p), "visual.repetition_composition")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE


def test_lookalike_ai_images(tmp_path):
    names = ["futuristic city skyline neon", "futuristic city skyline neon night", "futuristic city skyline neon glow"]
    rows = [(f"Scene {k} narration about markets.", f"k{k}", n) for k, n in enumerate(names)] + filler(1, "z")
    p, scenes, clips, assets = sequence(tmp_path, rows, kind="image")
    for a in assets.values():
        a.source_type = SourceType.AI_GENERATED
    i = find(run(p), "visual.repetition_generated")
    assert len(i) == 1 and "3 generated images" in i[0].description
    for a in list(assets.values())[:1]:
        a.source_type = SourceType.USER_MEDIA
    assert find(run(p), "visual.repetition_generated") == []


# ---------------------------------------------------------------- continuity
WILD = [("The Federal Reserve raised interest rates again this week.", "a", "federal reserve building"),
        ("That pushed prices higher across the whole metals market.", "w", "wildlife lions savanna"),
        ("The refinery processes silver ore into bars.", "c", "silver refinery smelter")]


def test_the_wildlife_footage_is_the_outlier(tmp_path):
    p, scenes, *_ = sequence(tmp_path, WILD)
    out = run(p)
    i = find(out, "continuity.unrelated")
    assert len(i) == 1 and i[0].scene_id == scenes[1].id and i[0].severity in (Severity.WARNING, Severity.NOTICE)
    assert i[0].confidence < 100 and "metadata" in i[0].description and i[0].fix.kind == "visual.replace" and not i[0].auto_fix_safe


def test_a_narrated_cutaway_is_not_flagged(tmp_path):
    rows = list(WILD)
    rows[1] = ("Imagine a herd of lions on the savanna: that is how the market reacted to higher prices.", "w", "wildlife lions savanna")
    p, *_ = sequence(tmp_path, rows)
    assert find(run(p), "continuity.unrelated") == []


def test_already_flagged_by_the_visual_checker_is_not_repeated(tmp_path):
    from app.qc.checker_base import CheckerOutput

    p, scenes, *_ = sequence(tmp_path, WILD)
    ctx = qc_ctx(p)
    vis = ContinuityChecker().issue("visual.mismatch", __import__("app.qc.issue_model", fromlist=["QCCategory"]).QCCategory.VISUAL_ACCURACY, Severity.WARNING, "x", scene_id=scenes[1].id)
    ctx.shared["visual"] = CheckerOutput(issues=[vis])
    out = run_checker(ContinuityChecker(), ctx)
    assert find(out, "continuity.unrelated") == [] and any("already flagged" in n for n in out.notes)


def test_abrupt_change_of_subject_without_a_signpost(tmp_path):
    rows = [("The central bank raised interest rates on Tuesday morning.", "a", "central bank building"), ("Penguins huddle together to survive Antarctic winters.", "b", "penguins antarctic ice"),
            ("Penguins also lay eggs on the Antarctic ice.", "c", "penguins eggs antarctic")]
    p, scenes, *_ = sequence(tmp_path, rows)
    i = find(run(p), "continuity.subject_jump")
    assert len(i) == 1 and i[0].scene_id == scenes[1].id and i[0].severity is Severity.NOTICE
    rows[1] = ("Meanwhile, penguins huddle together to survive Antarctic winters.", "b", "penguins antarctic ice")
    p2, *_ = sequence(tmp_path / "signposted", rows)
    assert find(run(p2), "continuity.subject_jump") == []


def test_cartoon_next_to_real_footage_clashes(tmp_path):
    rows = [("Silver supply from mines is falling.", "a", "silver mine footage"), ("Demand is rising in factories.", "b", "cartoon factory illustration"), ("Prices react in markets.", "c", "silver price chart footage")]
    p, scenes, clips, assets = sequence(tmp_path, rows)
    i = find(run(p), "continuity.style_clash")
    assert len(i) == 1 and i[0].severity is Severity.NOTICE
    rows2 = [(f"Scene {k} about silver markets.", f"k{k}", ("cartoon silver illustration" if k % 2 else "silver market footage")) for k in range(6)]
    p2, *_ = sequence(tmp_path / "many", rows2)
    assert find(run(p2), "continuity.style_clash")[0].severity is Severity.WARNING


def test_fast_zoom_straight_after_a_still_shot(tmp_path):
    p, scenes, clips, assets = sequence(tmp_path, GOOD)
    clips[1].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 0.5, 1.9)]
    i = find(run(p), "continuity.extreme_motion_jump")
    assert len(i) == 1 and i[0].scene_id == scenes[1].id
    clips[0].keyframes = [Keyframe("scale", 0.0, 1.0), Keyframe("scale", 5.0, 1.1)]  # the shot before it already moves
    assert find(run(p), "continuity.extreme_motion_jump") == []


def test_big_jump_in_scale_between_pictures(tmp_path):
    p, scenes, clips, assets = sequence(tmp_path, GOOD)
    clips[2].scale = 2.6
    i = find(run(p), "continuity.scale_jump")
    assert len(i) >= 1 and "limit 1.8x" in i[0].description


def test_sudden_brightness_jump_uses_thumbnails_when_present(tmp_path):
    p, scenes, clips, assets = sequence(tmp_path, GOOD)
    out = run(p)
    assert find(out, "continuity.color_mismatch") == [] and any("no thumbnails" in n for n in out.notes)
    from app.media.thumbnails import ThumbnailService

    for k, a in enumerate(assets.values()):
        t = ThumbnailService.thumbnail_path(p.root, a)
        t.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (250, 250, 250) if k == 1 else (10, 10, 10)).save(t)
    i = find(run(p), "continuity.color_mismatch")
    assert len(i) >= 1 and i[0].severity is Severity.NOTICE


def test_contract_and_no_mutation(tmp_path):
    p, *_ = sequence(tmp_path, WILD)
    c = ContinuityChecker()
    assert not c.scene_local and c.expensive and c.uses_shared and set(c.settings_sections) >= {"repetition", "continuity"}
    before = p.to_document()
    run(p)
    assert p.to_document() == before


def test_one_misplaced_picture_is_one_finding_not_also_a_subject_jump(tmp_path):
    """The wildlife footage between two finance scenes was reported twice: as a picture that belongs nowhere and as an abrupt change of subject into it."""
    p, scenes, *_ = sequence(tmp_path, WILD)
    out = run(p)
    assert [i.scene_id for i in find(out, "continuity.unrelated")] == [scenes[1].id]
    assert [i for i in find(out, "continuity.subject_jump") if i.scene_id == scenes[1].id] == []
