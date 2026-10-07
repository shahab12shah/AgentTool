from __future__ import annotations

import json

import pytest

from app.core.exceptions import InvalidProjectError, ProjectError
from app.project.project import Project
from app.project.project_manager import ProjectManager
from app.project.project_schema import migrate_document, validate_document
from app.core.events import EventBus
from app.core.constants import PROJECT_SUBDIRS, resolve_dimensions


@pytest.fixture
def pm(tmp_path):
    return ProjectManager(EventBus(), tmp_path / "recent.json")


def test_create_builds_folder_structure_and_valid_json(pm, tmp_path):
    p = pm.create("My Finance Video", tmp_path)
    root = tmp_path / "My Finance Video"
    assert (root / "project.json").is_file()
    for sub in PROJECT_SUBDIRS:
        assert (root / sub).is_dir()
    doc = json.loads((root / "project.json").read_text())
    validate_document(doc)
    assert doc["schema_version"] == 4
    assert doc["project"]["name"] == "My Finance Video"
    assert doc["settings"] == {"width": 1920, "height": 1080, "fps": 30, "aspect_ratio": "16:9"}
    assert len(doc["timeline"]["tracks"]) == 9
    assert p.project_id.startswith("proj_")


def test_create_rejects_bad_names_and_non_empty_folder(pm, tmp_path):
    with pytest.raises(ProjectError):
        pm.create("   ", tmp_path)
    with pytest.raises(ProjectError):
        pm.create("a/b", tmp_path)
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file.txt").write_text("x")
    with pytest.raises(ProjectError):
        pm.create("taken", tmp_path)


def test_save_updates_timestamp_and_roundtrips(pm, tmp_path):
    p = pm.create("P", tmp_path)
    p.script.text = "Hello world"
    p.updated_at = "2000-01-01T00:00:00+00:00"
    pm.save()
    assert p.updated_at != "2000-01-01T00:00:00+00:00"
    pm2 = ProjectManager(EventBus())
    loaded = pm2.open(tmp_path / "P")
    assert loaded.script.text == "Hello world"
    assert loaded.project_id == p.project_id
    assert not (tmp_path / "P" / "project.json.tmp").exists()


def test_save_is_atomic_when_validation_fails(pm, tmp_path, monkeypatch):
    p = pm.create("P", tmp_path)
    before = (tmp_path / "P" / "project.json").read_text()
    # Make the temp file fail verification: the existing file must remain untouched.
    import app.project.project_manager as mod

    def boom(doc):
        raise InvalidProjectError("nope")

    monkeypatch.setattr(mod, "validate_document", boom)
    p.script.text = "changed"
    with pytest.raises(InvalidProjectError):
        pm.save()
    assert (tmp_path / "P" / "project.json").read_text() == before
    assert not (tmp_path / "P" / "project.json.tmp").exists()


def test_save_refuses_inconsistent_project(pm, tmp_path):
    p = pm.create("P", tmp_path)
    p.voice_over.asset_id = "media_99999"
    with pytest.raises(InvalidProjectError) as e:
        pm.save()
    assert e.value.problems


def test_open_corrupt_json_gives_friendly_error_and_backup_works(pm, tmp_path):
    pm.create("P", tmp_path)
    pm.save()  # creates .bak from the first good file
    (tmp_path / "P" / "project.json").write_text("{ not json")
    with pytest.raises(InvalidProjectError) as e:
        ProjectManager(EventBus()).open(tmp_path / "P")
    assert "corrupt" in e.value.user_message.lower()
    assert "backup" in e.value.user_message.lower()
    restored = ProjectManager(EventBus()).open(tmp_path / "P", from_backup=True)
    assert restored.project_name == "P"


def test_open_missing_folder_or_file(pm, tmp_path):
    with pytest.raises(ProjectError):
        pm.open(tmp_path / "nothing")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("timeline"),
        lambda d: d.update(schema_version="1"),
        lambda d: d["settings"].update(fps=0),
        lambda d: d["assets"].append({"id": "x"}),
        lambda d: d["project"].pop("id"),
        lambda d: d.update(scenes={}),
    ],
)
def test_schema_validation_rejects_bad_documents(pm, tmp_path, mutate):
    doc = pm.create("P", tmp_path).to_document()
    mutate(doc)
    with pytest.raises(InvalidProjectError):
        validate_document(doc)
    with pytest.raises(InvalidProjectError):
        Project.from_document(doc)


def test_schema_rejects_non_object_and_newer_version(pm, tmp_path):
    with pytest.raises(InvalidProjectError):
        validate_document([])
    doc = pm.create("P", tmp_path).to_document()
    doc["schema_version"] = 99
    with pytest.raises(InvalidProjectError) as e:
        migrate_document(doc)
    assert "newer" in e.value.user_message


def test_recent_projects_tracks_opened_projects(pm, tmp_path):
    pm.create("A", tmp_path)
    pm.close()
    pm.open(tmp_path / "A")
    assert [r["name"] for r in pm.recent_projects()] == ["A"]


def test_resolution_and_aspect_ratio_mapping():
    assert resolve_dimensions("1920×1080", "16:9") == (1920, 1080)
    assert resolve_dimensions("1920×1080", "9:16") == (1080, 1920)
    assert resolve_dimensions("3840×2160", "1:1") == (2160, 2160)


def test_save_as_copies_project(pm, tmp_path):
    p = pm.create("Orig", tmp_path)
    (p.root / "media" / "audio" / "x.wav").write_bytes(b"data")
    copy = pm.save_as(tmp_path / "elsewhere", "Copy", p)
    assert (tmp_path / "elsewhere" / "Copy" / "media" / "audio" / "x.wav").read_bytes() == b"data"
    assert copy.project_name == "Copy" and copy.project_id == p.project_id
    with pytest.raises(ProjectError):
        pm.save_as(tmp_path / "elsewhere", "Copy", p)  # target exists and is not empty
