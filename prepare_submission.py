from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "src"))

from submission_release import ReleaseConfig, run_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="事务式刷新、验证、Docker 复现和提交 B 阶段结果。")
    parser.add_argument("--cutoff-date", required=True, type=date.fromisoformat)
    parser.add_argument("--experiment-name", default="xgb_ranker_v3")
    parser.add_argument("--docker-image", default="bdc2026:latest")
    parser.add_argument("--tar-name", default="霹雳.tar")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--keep-run-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_release(
        ReleaseConfig(
            workspace=Path(__file__).resolve().parent,
            cutoff_date=args.cutoff_date,
            experiment_name=args.experiment_name,
            docker_image=args.docker_image,
            tar_name=args.tar_name,
            push=args.push,
            keep_run_dir=args.keep_run_dir,
        )
    )
    print(f"release run_id={result.run_id}")
    print(f"result={result.result_path} sha256={result.result_sha256}")
    print(f"tar={result.tar_path} sha256={result.tar_sha256}")
    print(f"commit={result.commit_sha} remote={result.remote_sha}")


if __name__ == "__main__":
    main()
