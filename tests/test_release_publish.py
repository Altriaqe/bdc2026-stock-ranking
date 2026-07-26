from __future__ import annotations

from pathlib import Path

import pytest

import release_publish
from release_publish import PublishItem, ensure_inside_workspace, publish_with_rollback, restore_backup


def test_publish_replaces_files_and_can_restore(tmp_path: Path):
    old = tmp_path / "old.txt"
    candidate = tmp_path / "candidate.txt"
    destination = tmp_path / "published.txt"
    old.write_text("old", encoding="utf-8")
    candidate.write_text("new", encoding="utf-8")
    destination.write_text("before", encoding="utf-8")
    backup = tmp_path / "backup"

    publish_with_rollback([PublishItem(candidate, destination)], tmp_path, backup)
    assert destination.read_text(encoding="utf-8") == "new"
    restore_backup(backup, [PublishItem(candidate, destination)])
    assert destination.read_text(encoding="utf-8") == "before"


def test_publish_rolls_back_when_second_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    candidate_a = tmp_path / "a.new"
    candidate_b = tmp_path / "b.new"
    destination_a = tmp_path / "a.txt"
    destination_b = tmp_path / "b.txt"
    candidate_a.write_text("new-a", encoding="utf-8")
    candidate_b.write_text("new-b", encoding="utf-8")
    destination_a.write_text("old-a", encoding="utf-8")
    destination_b.write_text("old-b", encoding="utf-8")
    original_replace = release_publish.os.replace
    calls = {"count": 0}

    def fail_second(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(release_publish.os, "replace", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        publish_with_rollback(
            [PublishItem(candidate_a, destination_a), PublishItem(candidate_b, destination_b)],
            tmp_path,
            tmp_path / "backup",
        )
    assert destination_a.read_text(encoding="utf-8") == "old-a"
    assert destination_b.read_text(encoding="utf-8") == "old-b"


def test_paths_outside_workspace_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="工作区"):
        ensure_inside_workspace(tmp_path, tmp_path.parent / "outside.txt")
