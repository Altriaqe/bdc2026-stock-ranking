from __future__ import annotations

import io
import tarfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from release_validation import (
    assert_backtest_gates,
    validate_market_data,
    validate_result,
    validate_tar,
)


def make_market(rows_per_stock: int = 60, *, invalid: bool = False) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-01", periods=rows_per_stock)
    for index in range(2):
        for day in dates:
            close = 10.0 + index
            rows.append(
                {
                    "stock_id": f"{index + 1:06d}",
                    "trade_date": day.strftime("%Y-%m-%d"),
                    "open": close,
                    "close": close,
                    "high": close + (0.5 if not invalid else -1.0),
                    "low": close - 0.5,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "turnover_rate": 1.0,
                    "pct_chg": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_market_data_report_accepts_valid_data(tmp_path: Path):
    path = tmp_path / "market.csv"
    make_market().to_csv(path, index=False, lineterminator="\n")
    report = validate_market_data(path, date(2026, 3, 25), minimum_stock_count=2)
    assert report.stock_count == 2
    assert report.latest_day_stock_count == 2
    assert report.short_history_stock_count == 0


def test_market_data_rejects_duplicate_and_invalid_ohlc(tmp_path: Path):
    frame = make_market(invalid=True)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "market.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="重复主键"):
        validate_market_data(path, date(2026, 3, 25), minimum_stock_count=2)

    path2 = tmp_path / "invalid.csv"
    make_market(invalid=True).to_csv(path2, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="非法 OHLC"):
        validate_market_data(path2, date(2026, 3, 25), minimum_stock_count=2)


def test_market_data_rejects_cutoff_and_short_history(tmp_path: Path):
    path = tmp_path / "market.csv"
    make_market(rows_per_stock=59).to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="截止日"):
        validate_market_data(path, date(2026, 3, 25), minimum_stock_count=2)
    with pytest.raises(ValueError, match="历史不足"):
        validate_market_data(path, date(2026, 3, 24), minimum_stock_count=2)


def test_result_requires_exact_five_equal_weight_rows(tmp_path: Path):
    path = tmp_path / "result.csv"
    pd.DataFrame({"stock_id": [f"{i:06d}" for i in range(5)], "weight": [0.2] * 5}).to_csv(
        path, index=False, lineterminator="\n"
    )
    report = validate_result(path, {f"{i:06d}" for i in range(5)})
    assert report.row_count == 5
    assert report.weight_sum == 1.0

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"stock_id": ["000000"] * 5, "weight": [0.2] * 5}).to_csv(
        bad, index=False, lineterminator="\n"
    )
    with pytest.raises(ValueError, match="重复"):
        validate_result(bad)


def test_result_rejects_crlf(tmp_path: Path):
    path = tmp_path / "result.csv"
    path.write_bytes(b"stock_id,weight\r\n000000,0.2\r\n000001,0.2\r\n000002,0.2\r\n000003,0.2\r\n000004,0.2\r\n")
    with pytest.raises(ValueError, match="LF"):
        validate_result(path)


def test_tar_validation(tmp_path: Path):
    path = tmp_path / "bundle.tar"
    with tarfile.open(path, "w") as archive:
        payload = b"ok"
        info = tarfile.TarInfo("result.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    report = validate_tar(path)
    assert report.member_count == 1
    with pytest.raises(ValueError, match="上限"):
        validate_tar(path, max_bytes=1)


def _strategy(mean: float, worst: float) -> dict[str, float]:
    return {"pred_top_k_return_mean": mean, "pred_top_k_worst": worst}


def test_backtest_gates_accept_improved_positive_strategy():
    folds = [
        {
            "purge_group_count": 5,
            "strategy_metrics": {
                "equal_rank_ensemble_top5": _strategy(0.01, -0.05),
                "current_overlay_top1": _strategy(-0.01, -0.10),
            },
        }
        for _ in range(4)
    ]
    assert_backtest_gates({"purged_outer_folds": folds}) is None


def test_backtest_gates_reject_nonpositive_latest():
    folds = [
        {
            "purge_group_count": 5,
            "strategy_metrics": {
                "equal_rank_ensemble_top5": _strategy(0.01 if i < 3 else -0.01, -0.05),
                "current_overlay_top1": _strategy(-0.01, -0.10),
            },
        }
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="收益门槛"):
        assert_backtest_gates({"purged_outer_folds": folds})
