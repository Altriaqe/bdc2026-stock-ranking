from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inference_can_write_only_to_isolated_paths(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    formal_result = root / "output" / "result.csv"
    formal_before = sha256(formal_result) if formal_result.exists() else None
    isolated_result = tmp_path / "candidate" / "result.csv"
    isolated_reports = tmp_path / "reports"
    command = [
        sys.executable,
        str(root / "code" / "src" / "test.py"),
        "--experiment-name",
        "xgb_ranker_v3",
        "--output-path",
        str(isolated_result),
        "--report-dir",
        str(isolated_reports),
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = pd.read_csv(isolated_result, dtype={"stock_id": str})
    assert list(result.columns) == ["stock_id", "weight"]
    assert len(result) == 5
    assert result["weight"].tolist() == [0.2] * 5
    assert (isolated_reports / "submission_check.json").is_file()
    assert (isolated_reports / "portfolio_report.json").is_file()
    if formal_before is not None:
        assert sha256(formal_result) == formal_before
