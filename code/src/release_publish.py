from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PublishItem:
    candidate: Path
    destination: Path


def ensure_inside_workspace(workspace: Path, path: Path) -> Path:
    workspace_resolved = workspace.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError as error:
        raise ValueError(f"路径必须位于工作区内：{path}") from error
    return resolved


def _remove_exact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy_item(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _backup_item(source: Path, destination: Path, backup_dir: Path) -> dict[str, object]:
    record: dict[str, object] = {"destination": str(destination), "existed": source.exists()}
    if source.exists():
        backup_path = backup_dir / f"{uuid.uuid4().hex}-{destination.name}"
        _copy_item(source, backup_path)
        record["backup"] = str(backup_path)
    return record


def _restore_records(records: list[dict[str, object]]) -> None:
    for record in reversed(records):
        destination = Path(str(record["destination"]))
        _remove_exact(destination)
        if bool(record["existed"]):
            backup = Path(str(record["backup"]))
            _copy_item(backup, destination)


def restore_backup(backup_dir: Path, items: Sequence[PublishItem]) -> None:
    manifest_path = backup_dir / "backup_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"备份清单不存在：{manifest_path}")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(records) != len(items):
        raise ValueError("备份清单与发布项目数量不一致。")
    _restore_records(records)


def publish_with_rollback(
    items: Sequence[PublishItem],
    workspace: Path,
    backup_dir: Path,
) -> None:
    workspace_resolved = workspace.resolve()
    backup_dir_resolved = ensure_inside_workspace(workspace_resolved, backup_dir)
    backup_dir_resolved.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for item in items:
        candidate = ensure_inside_workspace(workspace_resolved, item.candidate)
        destination = ensure_inside_workspace(workspace_resolved, item.destination)
        if not candidate.exists():
            raise FileNotFoundError(f"候选发布项不存在：{candidate}")
        records.append(_backup_item(destination, destination, backup_dir_resolved))
        records[-1]["destination"] = str(destination)
        records[-1]["candidate"] = str(candidate)
    manifest_path = backup_dir_resolved / "backup_manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        for record in records:
            candidate = Path(str(record["candidate"]))
            destination = Path(str(record["destination"]))
            temp_destination = destination.parent / f".{destination.name}.release-{uuid.uuid4().hex}"
            _copy_item(candidate, temp_destination)
            _remove_exact(destination)
            os.replace(temp_destination, destination)
    except Exception:
        _restore_records(records)
        raise
