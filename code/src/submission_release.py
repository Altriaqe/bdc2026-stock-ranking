from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from release_publish import PublishItem, publish_with_rollback
from release_validation import (
    MarketDataReport,
    SubmissionReport,
    TarReport,
    assert_backtest_gates,
    sha256_file,
    validate_market_data,
    validate_result,
    validate_tar,
)


@dataclass(frozen=True)
class ReleaseConfig:
    workspace: Path
    cutoff_date: date
    experiment_name: str = "xgb_ranker_v3"
    docker_image: str = "bdc2026:latest"
    tar_name: str = "霹雳.tar"
    push: bool = False
    keep_run_dir: bool = False


@dataclass(frozen=True)
class ReleaseResult:
    run_id: str
    manifest_path: Path
    result_path: Path
    result_sha256: str
    tar_path: Path
    tar_sha256: str
    commit_sha: str | None
    remote_sha: str | None


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        result = CommandResult(tuple(args), completed.returncode, time.perf_counter() - started, stdout, stderr)
    except subprocess.TimeoutExpired as error:
        stdout = str(error.stdout or "")
        stderr = str(error.stderr or "") + f"\nTIMEOUT after {timeout_seconds}s"
        result = CommandResult(tuple(args), 124, time.perf_counter() - started, stdout, stderr)
    log_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"命令失败（{result.returncode}）：{' '.join(args)}\n{stderr[-2000:]}")
    return result


def _git_output(workspace: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Git 命令失败：{' '.join(args)}\n{completed.stderr}")
    return completed.stdout.strip()


def _docker_image_id(workspace: Path, image: str) -> str | None:
    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else None


def _restore_docker_tag(workspace: Path, image: str, previous_id: str | None) -> None:
    if previous_id:
        subprocess.run(["docker", "tag", previous_id, image], cwd=workspace, capture_output=True, check=False)
    else:
        subprocess.run(["docker", "rmi", image], cwd=workspace, capture_output=True, check=False)


def _require_clean_git(workspace: Path) -> None:
    if _git_output(workspace, ["branch", "--show-current"]) != "main":
        raise RuntimeError("发布必须在 main 分支执行。")
    if not _git_output(workspace, ["remote", "get-url", "origin"]):
        raise RuntimeError("Git 远端 origin 不存在。")
    if _git_output(workspace, ["status", "--porcelain", "--untracked-files=all"]):
        raise RuntimeError("发布开始前工作树必须干净。")


def _make_run_dir(workspace: Path) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = workspace / "temp" / "submission-release" / run_id
    for name in ("candidate/data", "candidate/model/xgb_ranker_v3", "candidate/output", "candidate/docs/validation", "docker-context", "docker-artifacts", "backup", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _copy_candidate_data(run_dir: Path, workspace: Path) -> None:
    source_test = workspace / "data" / "test.csv"
    if source_test.is_file():
        shutil.copy2(source_test, run_dir / "candidate" / "data" / "test.csv")
    else:
        (run_dir / "candidate" / "data" / "test.csv").write_text("stock_id,weight\n", encoding="utf-8")


def stage_market_data(config: ReleaseConfig, run_id: str, run_dir: Path) -> MarketDataReport:
    candidate_data = run_dir / "candidate" / "data"
    downloader = config.workspace / "get_stock_data.py"
    _run_command(
        [
            sys.executable,
            str(downloader),
            "--start-date",
            "2024-01-01",
            "--end-date",
            config.cutoff_date.isoformat(),
            "--adjustflag",
            "1",
            "--output",
            str(candidate_data / "stock_data.csv"),
            "--stock-list-output",
            str(candidate_data / "hs300_stock_list.csv"),
        ],
        cwd=config.workspace,
        log_path=run_dir / "logs" / "download.json",
        timeout_seconds=30 * 60,
    )
    report = validate_market_data(candidate_data / "stock_data.csv", config.cutoff_date)
    shutil.copy2(candidate_data / "stock_data.csv", candidate_data / "train.csv")
    _copy_candidate_data(run_dir, config.workspace)
    return report


def _candidate_experiment(config: ReleaseConfig, run_id: str) -> str:
    return f"release_{run_id}"


def run_local_candidate(config: ReleaseConfig, run_id: str, run_dir: Path, data_report: MarketDataReport) -> dict[str, Any]:
    candidate_train = run_dir / "candidate" / "data" / "train.csv"
    experiment = _candidate_experiment(config, run_id)
    _run_command(
        [
            sys.executable,
            str(config.workspace / "code" / "src" / "train.py"),
            "--train-data",
            str(candidate_train),
            "--experiment-name",
            experiment,
            "--feature-preset",
            "alpha_v1",
            "--production-models",
            "xgb_ranker,lgb_ranker,hgb_regressor",
            "--production-portfolio-size",
            "5",
        ],
        cwd=config.workspace,
        log_path=run_dir / "logs" / "local_train.json",
        timeout_seconds=8 * 60 * 60,
    )
    experiment_dir = config.workspace / "model" / experiment
    report_path = experiment_dir / "backtest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert_backtest_gates(report)
    local_result = run_dir / "candidate" / "output" / "local_result.csv"
    local_reports = run_dir / "candidate" / "temp-local"
    _run_command(
        [
            sys.executable,
            str(config.workspace / "code" / "src" / "test.py"),
            "--inference-data",
            str(candidate_train),
            "--experiment-name",
            experiment,
            "--output-path",
            str(local_result),
            "--report-dir",
            str(local_reports),
        ],
        cwd=config.workspace,
        log_path=run_dir / "logs" / "local_inference.json",
        timeout_seconds=5 * 60,
    )
    return {"experiment": experiment, "backtest": report, "local_result": validate_result(local_result, set(data_report.stock_ids)).to_dict()}


def _copy_context(root: Path, context: Path, candidate_data: Path) -> None:
    ignored_names = {".git", ".venv", "temp", "model", "output"}

    def ignore(directory: str, names: list[str]) -> set[str]:
        result = {name for name in names if name in ignored_names or name.endswith((".tar", ".pyc"))}
        if Path(directory).name == "data":
            result.update(name for name in names if name.endswith(".csv"))
        return result

    shutil.copytree(root, context, ignore=ignore)
    context_data = context / "data"
    context_data.mkdir(parents=True, exist_ok=True)
    for source in candidate_data.iterdir():
        if source.is_file():
            shutil.copy2(source, context_data / source.name)


def run_docker_candidate(config: ReleaseConfig, run_id: str, run_dir: Path, data_report: MarketDataReport) -> dict[str, Any]:
    context = run_dir / "docker-context"
    _copy_context(config.workspace, context, run_dir / "candidate" / "data")
    base_image = f"bdc2026-release:{run_id}"
    trained_image = f"bdc2026-trained:{run_id}"
    full_container = f"bdc2026-full-{run_id}"
    inference_container = f"bdc2026-infer-{run_id}"
    previous_image_id = _docker_image_id(config.workspace, config.docker_image)
    _run_command(
        ["docker", "build", "--progress=plain", "-t", base_image, str(context)],
        cwd=config.workspace,
        log_path=run_dir / "logs" / "docker_build.json",
        timeout_seconds=60 * 60,
    )
    try:
        _run_command(
            ["docker", "run", "--name", full_container, "--network", "none", base_image],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_full_run.json",
            timeout_seconds=8 * 60 * 60,
        )
        _run_command(
            ["docker", "commit", full_container, trained_image],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_commit.json",
            timeout_seconds=10 * 60,
        )
        full_result = run_dir / "docker-artifacts" / "full_result.csv"
        _run_command(
            ["docker", "cp", f"{full_container}:/app/output/result.csv", str(full_result)],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_copy_full_result.json",
            timeout_seconds=5 * 60,
        )
        _run_command(
            ["docker", "run", "--name", inference_container, "--network", "none", "-e", "REPRODUCE_FROM_TRAIN=0", trained_image],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_inference_run.json",
            timeout_seconds=5 * 60,
        )
        final_result = run_dir / "candidate" / "output" / "result.csv"
        _run_command(
            ["docker", "cp", f"{inference_container}:/app/output/result.csv", str(final_result)],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_copy_result.json",
            timeout_seconds=5 * 60,
        )
        full_report = validate_result(full_result, set(data_report.stock_ids))
        final_report = validate_result(final_result, set(data_report.stock_ids))
        if full_report.sha256 != final_report.sha256:
            raise ValueError("Docker 完整运行与第二次推理结果哈希不一致。")
        artifacts_model = run_dir / "candidate" / "model" / "xgb_ranker_v3"
        _run_command(
            ["docker", "cp", f"{inference_container}:/app/model/xgb_ranker_v3/.", str(artifacts_model)],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_copy_model.json",
            timeout_seconds=10 * 60,
        )
        portfolio_report = run_dir / "candidate" / "output" / "portfolio_report.json"
        _run_command(
            ["docker", "cp", f"{inference_container}:/app/temp/portfolio_report.json", str(portfolio_report)],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_copy_portfolio.json",
            timeout_seconds=5 * 60,
        )
        docker_report = json.loads((artifacts_model / "backtest_report.json").read_text(encoding="utf-8"))
        assert_backtest_gates(docker_report)
        tar_path = run_dir / "candidate" / config.tar_name
        _run_command(
            ["docker", "save", "-o", str(tar_path), trained_image],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_save.json",
            timeout_seconds=30 * 60,
        )
        tar_report = validate_tar(tar_path)
        _run_command(
            ["docker", "tag", trained_image, config.docker_image],
            cwd=config.workspace,
            log_path=run_dir / "logs" / "docker_tag_final.json",
            timeout_seconds=5 * 60,
        )
        return {
            "image": config.docker_image,
            "previous_image_id": previous_image_id,
            "full_result": full_report.to_dict(),
            "result": final_report.to_dict(),
            "tar": tar_report.to_dict(),
            "backtest": docker_report,
        }
    finally:
        for container in (inference_container, full_container):
            subprocess.run(["docker", "rm", "-f", container], cwd=config.workspace, capture_output=True, check=False)
        subprocess.run(["docker", "rmi", base_image], cwd=config.workspace, capture_output=True, check=False)


def _write_candidate_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "candidate" / "docs" / "validation" / "latest-submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_validation_markdown(run_dir: Path, config: ReleaseConfig, payload: dict[str, Any]) -> Path:
    path = run_dir / "candidate" / "docs" / "validation" / f"{config.cutoff_date.isoformat()}-b-stage-validation.md"
    result = payload["docker"]["result"]
    lines = [
        f"# B 阶段发布验证 {config.cutoff_date.isoformat()}",
        "",
        f"- 数据哈希：`{payload['market_data']['sha256']}`",
        f"- 结果哈希：`{result['sha256']}`",
        f"- tar 哈希：`{payload['docker']['tar']['sha256']}`",
        f"- 股票：{', '.join(result['stock_ids'])}",
        "- 权重：每只 0.2，共 5 只",
        "- Docker：network none，完整训练与第二次推理哈希一致",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _publish(config: ReleaseConfig, run_dir: Path, manifest: Path, validation_doc: Path) -> None:
    candidate = run_dir / "candidate"
    items = [
        PublishItem(candidate / "data" / "stock_data.csv", config.workspace / "data" / "stock_data.csv"),
        PublishItem(candidate / "data" / "train.csv", config.workspace / "data" / "train.csv"),
        PublishItem(candidate / "data" / "test.csv", config.workspace / "data" / "test.csv"),
        PublishItem(candidate / "data" / "hs300_stock_list.csv", config.workspace / "data" / "hs300_stock_list.csv"),
        PublishItem(candidate / "model" / "xgb_ranker_v3", config.workspace / "model" / "xgb_ranker_v3"),
        PublishItem(candidate / "output" / "result.csv", config.workspace / "output" / "result.csv"),
        PublishItem(candidate / config.tar_name, config.workspace / config.tar_name),
        PublishItem(manifest, config.workspace / "docs" / "validation" / "latest-submission.json"),
        PublishItem(validation_doc, config.workspace / "docs" / "validation" / validation_doc.name),
    ]
    publish_with_rollback(items, config.workspace, run_dir / "backup")


def _commit_and_push(config: ReleaseConfig, run_id: str, manifest_path: Path) -> tuple[str | None, str | None]:
    if not config.push:
        return None, None
    tracked = [
        "output/result.csv",
        "docs/validation/latest-submission.json",
        f"docs/validation/{config.cutoff_date.isoformat()}-b-stage-validation.md",
    ]
    _run_command(["git", "add", "--", *tracked], cwd=config.workspace, log_path=config.workspace / "temp" / "submission-release" / run_id / "logs" / "git_add.json", timeout_seconds=60)
    _run_command(["git", "diff", "--cached", "--check"], cwd=config.workspace, log_path=config.workspace / "temp" / "submission-release" / run_id / "logs" / "git_check.json", timeout_seconds=60)
    _run_command(["git", "commit", "-m", f"chore: publish B-stage result {config.cutoff_date.isoformat()}"], cwd=config.workspace, log_path=config.workspace / "temp" / "submission-release" / run_id / "logs" / "git_commit.json", timeout_seconds=120)
    commit_sha = _git_output(config.workspace, ["rev-parse", "HEAD"])
    _run_command(["git", "push", "origin", "main"], cwd=config.workspace, log_path=config.workspace / "temp" / "submission-release" / run_id / "logs" / "git_push.json", timeout_seconds=120)
    remote_line = _git_output(config.workspace, ["ls-remote", "origin", "refs/heads/main"])
    remote_sha = remote_line.split()[0]
    if remote_sha != commit_sha:
        raise RuntimeError(f"远端 SHA 与本地不一致：{remote_sha} != {commit_sha}")
    return commit_sha, remote_sha


def run_release(config: ReleaseConfig) -> ReleaseResult:
    workspace = config.workspace.resolve()
    if config.cutoff_date > date.today():
        raise ValueError("cutoff-date 不能晚于运行日期。")
    if Path(config.tar_name).name != config.tar_name:
        raise ValueError("tar-name 必须是工作区根目录下的单一文件名。")
    _require_clean_git(workspace)
    run_id, run_dir = _make_run_dir(workspace)
    data_report = stage_market_data(config, run_id, run_dir)
    local = run_local_candidate(config, run_id, run_dir, data_report)
    docker = run_docker_candidate(config, run_id, run_dir, data_report)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "validated",
        "run_id": run_id,
        "created_at": _now_utc(),
        "cutoff_date": config.cutoff_date.isoformat(),
        "market_data": data_report.to_dict(),
        "local": local,
        "docker": docker,
    }
    manifest = _write_candidate_manifest(run_dir, payload)
    validation_doc = _write_validation_markdown(run_dir, config, payload)
    try:
        _publish(config, run_dir, manifest, validation_doc)
    except Exception:
        _restore_docker_tag(config.workspace, config.docker_image, docker.get("previous_image_id"))
        raise
    if config.push:
        commit_sha, remote_sha = _commit_and_push(config, run_id, manifest)
        payload["status"] = "pushed"
        payload["commit_sha"] = commit_sha
        payload["remote_sha"] = remote_sha
        formal_manifest = config.workspace / "docs" / "validation" / "latest-submission.json"
        formal_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _run_command(["git", "add", "--", str(formal_manifest.relative_to(workspace))], cwd=workspace, log_path=run_dir / "logs" / "git_status_update_add.json", timeout_seconds=60)
        _run_command(["git", "commit", "-m", f"chore: record pushed B-stage manifest {config.cutoff_date.isoformat()}"], cwd=workspace, log_path=run_dir / "logs" / "git_status_update_commit.json", timeout_seconds=120)
        _run_command(["git", "push", "origin", "main"], cwd=workspace, log_path=run_dir / "logs" / "git_status_update_push.json", timeout_seconds=120)
        payload["commit_sha"] = _git_output(workspace, ["rev-parse", "HEAD"])
        payload["remote_sha"] = _git_output(workspace, ["ls-remote", "origin", "refs/heads/main"]).split()[0]
        formal_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ReleaseResult(
        run_id=run_id,
        manifest_path=config.workspace / "docs" / "validation" / "latest-submission.json",
        result_path=config.workspace / "output" / "result.csv",
        result_sha256=sha256_file(config.workspace / "output" / "result.csv"),
        tar_path=config.workspace / config.tar_name,
        tar_sha256=sha256_file(config.workspace / config.tar_name),
        commit_sha=payload.get("commit_sha"),
        remote_sha=payload.get("remote_sha"),
    )
