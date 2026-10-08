"""Phase 7 OriginalityGuard: requests to copy become abstract style requests; reference media stays out of the media library and the timeline; what leaves the analysis is abstract."""

from __future__ import annotations

import json

import pytest

from app.editing.overrides import EditingStrategyOverrides
from app.media.asset import Asset, AssetType, SourceType
from app.project.project import Project
from app.reference.application import ReferenceAsset
from app.reference.originality import KINDS, OriginalityGuard
from app.reference.style_model import StyleFeatures, build_profile_from_features
from app.timeline.clip import Clip

guard = OriginalityGuard()


# ---------------------------------------------------------------------------------------------- requests
def test_the_specs_own_example():
    r = guard.convert_request("Copy the competitor's exact opening.")
    assert r.flagged and r.kinds == ["OPENING"]
    assert r.abstract_instruction == "Use a fast, high-information opening with strong text emphasis."
    assert r.neutralised and "not followed" in r.neutralised[0]


@pytest.mark.parametrize("text,kinds", [
    ("Use their footage and script", {"FOOTAGE", "TEXT"}),
    ("Copy the shot order frame by frame", {"SEQUENCE"}),
    ("Reuse their logo and intro music", {"BRANDING", "MUSIC"}),
    ("Replicate their lower thirds and animations exactly", {"GRAPHICS"}),
    ("Make the captions look the same as theirs", {"TYPOGRAPHY"}),
    ("Steal the thumbnail", {"THUMBNAIL"}),
    ("Use the same camera angles and composition", {"COMPOSITION"}),
    ("I want the same music", {"MUSIC"}),
    ("Copy the whole thing", {"FOOTAGE"}),
    ("Recreate the exact transitions in the same order", {"TRANSITIONS"}),
    ("Copy the ad copy", {"TEXT"}),
    ("Download the video from https://example.com/watch?v=abc and use it", {"FOOTAGE"}),
])
def test_requests_to_reproduce_specific_content_are_neutralised(text, kinds):
    r = guard.convert_request(text)
    assert r.flagged and set(r.kinds) == kinds, r.kinds
    assert all(KINDS[k][1] in r.abstract_instruction for k in kinds)
    assert r.neutralised and len(r.neutralised) == len(kinds) and r.findings[0].replacement


@pytest.mark.parametrize("text", [
    "Faster pacing with subtle zooms",
    "Make it like the reference",
    "Make the pacing exactly like the reference",
    "Make my video exactly as fast as theirs",
    "Use a similar look and feel with bold captions",
    "Match their cut rhythm, caption density and music ducking",
    "Quicker cuts and more keyword highlighting",
    "Copy their structure and pacing",
])
def test_abstract_wishes_pass_through_unchanged(text):
    r = guard.convert_request(text)
    assert not r.flagged and r.abstract_instruction == text and not r.findings and not r.neutralised


def test_a_quoted_passage_is_never_repeated():
    r = guard.convert_request('Use the line "Welcome back to the channel everybody" exactly as they say it')
    assert r.flagged and r.kinds == ["TEXT"]
    assert "Welcome" not in r.abstract_instruction and "everybody" not in r.abstract_instruction
    assert all("Welcome" not in n for n in r.neutralised + r.notes)


def test_only_the_copying_part_of_a_mixed_request_is_replaced():
    r = guard.convert_request("Make the pacing faster and the zooms subtle. Copy their exact graphics. Add more captions.")
    assert r.flagged and r.kinds == ["GRAPHICS"]
    assert "Make the pacing faster" in r.abstract_instruction and "Add more captions" in r.abstract_instruction
    assert KINDS["GRAPHICS"][1] in r.abstract_instruction and "graphics" not in r.kept[0].lower()
    assert set(r.hints) >= {"pacing", "motion", "captions"}


def test_links_are_never_followed_or_kept():
    r = guard.convert_request("Make it like https://www.example.com/watch?v=123 with faster cuts")
    assert not r.flagged and "example.com" not in r.abstract_instruction
    assert any("Links are never opened" in n for n in r.notes)


def test_possessive_material_of_the_user_is_not_a_reference():
    assert not guard.convert_request("Re-use my footage from last week with the same pacing").flagged
    assert not guard.convert_request("Reuse my own intro music").flagged
    assert guard.convert_request("Re-use their footage from last week").flagged


def test_empty_and_whitespace_requests():
    for t in ("", "   ", None):
        r = guard.convert_request(t)  # type: ignore[arg-type]
        assert not r.flagged and r.abstract_instruction == "" and not r.findings


def test_the_abstract_instruction_never_contains_a_request_to_copy():
    import re

    for text in ("Copy their exact opening and logo", "Reproduce the script word for word", "Rip the footage", "Steal the thumbnail and the graphics"):
        out = guard.convert_request(text).abstract_instruction.lower()
        assert out and not re.search(r"\b(copy|reproduce|rip|steal|exact|exactly|word for word)\b", out), out


# ---------------------------------------------------------------------------------------------- isolation of the reference media
def _project(tmp_path) -> Project:
    p = Project.new("Iso")
    p.root = tmp_path / "proj"
    (p.root / "references" / "ref_a").mkdir(parents=True)
    (p.root / "media").mkdir()
    (p.root / "references" / "ref_a" / "reference_video.mp4").write_bytes(b"x" * 2000)
    (p.root / "media" / "mine.mp4").write_bytes(b"y" * 1000)
    p.reference_assets["ref_a"] = ReferenceAsset("ref_a", "competitor", "references/ref_a/reference_video.mp4", "copy", "abc", 2000)
    return p


def _asset(pid: str, path: str, size: int = 1000) -> Asset:
    return Asset(pid, AssetType.VIDEO, SourceType.USER_MEDIA, path, pid + ".mp4", 10.0, 320, 180, 24.0, size_bytes=size)


def test_a_clean_project_passes_the_isolation_audit(tmp_path):
    p = _project(tmp_path)
    p.assets.add(_asset("a1", "media/mine.mp4"))
    p.timeline.get_track("track_v1").clips.append(Clip("c1", "track_v1", "a1", 0.0, 5.0))
    audit = guard.audit_project(p)
    assert audit.ok and not audit.errors and not audit.warnings


def test_a_reference_in_the_media_library_or_on_the_timeline_is_an_error(tmp_path):
    p = _project(tmp_path)
    p.assets.add(_asset("bad", "references/ref_a/reference_video.mp4", 2000))
    p.timeline.get_track("track_v1").clips.append(Clip("c_bad", "track_v1", "bad", 0.0, 5.0))
    audit = guard.audit_project(p)
    assert not audit.ok and any("reference video" in e for e in audit.errors) and any("c_bad" in e for e in audit.errors)


def test_a_linked_reference_file_is_found_by_its_path(tmp_path):
    p = _project(tmp_path)
    outside = tmp_path / "elsewhere" / "theirs.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"z" * 3000)
    p.reference_assets["ref_b"] = ReferenceAsset("ref_b", "linked", str(outside), "reference", "def", 3000)
    p.assets.add(_asset("lnk", str(outside), 3000))
    assert not guard.audit_project(p).ok
    assert guard.is_reference_path(p, outside) and guard.is_reference_path(p, p.root / "references" / "ref_a" / "reference_video.mp4")
    assert not guard.is_reference_path(p, p.root / "media" / "mine.mp4")


def test_a_same_sized_asset_is_only_a_warning(tmp_path):
    p = _project(tmp_path)
    (p.root / "media" / "maybe.mp4").write_bytes(b"w" * 2000)
    p.assets.add(_asset("maybe", "media/maybe.mp4", 2000))
    audit = guard.audit_project(p)
    assert audit.ok and audit.warnings and "same size" in audit.warnings[0]


# ---------------------------------------------------------------------------------------------- what leaves the analysis is abstract
def test_overrides_must_hold_numbers_and_known_ids_only():
    ok = EditingStrategyOverrides(target_shot_duration=3.0, caption_style="bold", caption_position="center", caption_max_words=5)
    assert guard.audit_overrides(ok) == []
    bad = EditingStrategyOverrides(caption_style="Welcome back to the channel")
    assert guard.audit_overrides(bad)
    quoted = EditingStrategyOverrides(target_shot_duration=3.0, notes=['They say "Welcome back"'])
    assert guard.audit_overrides(quoted)


def test_a_style_profile_holds_aggregates_only():
    prof = build_profile_from_features(StyleFeatures(duration=60.0, cuts_per_minute=20.0, average_shot_duration=3.0))
    assert guard.audit_profile(prof) == []
    blob = json.dumps(prof.to_dict())
    assert "frame" not in blob.lower() and "transcript" not in blob.lower()
    from app.reference.style_model import ReferenceStyleProfile

    bad = prof.to_dict()
    bad["warnings"] = ["x" * 300]
    assert guard.audit_profile(ReferenceStyleProfile.from_dict(bad))  # a long free-text value is reported
    smuggled = prof.to_dict()
    smuggled["caption_style"]["text"] = "Welcome back"
    assert guard.audit_profile(ReferenceStyleProfile.from_dict(smuggled)) == []  # unknown keys are dropped when the profile is rebuilt: no text field can ride along
