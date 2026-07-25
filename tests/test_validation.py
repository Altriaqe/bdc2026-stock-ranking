import numpy as np
import pandas as pd

from featurework import (
    build_purged_walk_forward_splits,
    calculate_top_k_return_metrics,
    select_rebalance_dates,
)


def make_frame(days: int = 40, stocks: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=days)
    rows = []
    for date in dates:
        for stock_index in range(stocks):
            rows.append(
                {
                    "trade_date": date,
                    "stock_id": f"{stock_index:06d}",
                    "future_return": (stock_index - 2) / 100.0,
                    "pred_score": float(stock_index),
                }
            )
    return pd.DataFrame(rows)


def test_purged_split_removes_five_label_days():
    frame = make_frame()
    splits = build_purged_walk_forward_splits(
        frame,
        fold_count=2,
        validation_ratio=0.2,
        min_train_groups=10,
        purge_groups=5,
    )
    all_dates = sorted(frame["trade_date"].unique())
    for train, valid in splits:
        train_last = all_dates.index(train["trade_date"].max())
        valid_first = all_dates.index(valid["trade_date"].min())
        assert valid_first - train_last > 5


def test_rebalance_dates_are_non_overlapping():
    dates = select_rebalance_dates(make_frame(), stride=5)
    assert dates == list(pd.bdate_range("2026-01-01", periods=40)[::5])


def test_top5_metric_matches_official_equal_weight_return():
    metrics = calculate_top_k_return_metrics(make_frame(days=10), k=5, rebalance_stride=5)
    expected = np.mean([0.01, 0.02, 0.03, 0.00, -0.01])
    assert metrics["pred_top_k_return_mean"] == round(float(expected), 8)
    assert metrics["evaluated_week_count"] == 2


def test_invalid_split_parameters_fail_fast():
    frame = make_frame()
    for kwargs in (
        {"fold_count": 0, "validation_ratio": 0.2, "min_train_groups": 10, "purge_groups": 5},
        {"fold_count": 2, "validation_ratio": 1.0, "min_train_groups": 10, "purge_groups": 5},
        {"fold_count": 2, "validation_ratio": 0.2, "min_train_groups": 100, "purge_groups": 5},
    ):
        try:
            build_purged_walk_forward_splits(frame, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
