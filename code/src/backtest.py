from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from featurework import select_rebalance_dates
from portfolio import PortfolioSelection, select_top5_portfolio


def calculate_weekly_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("weekly returns must be finite and non-empty")
    tail_count = max(1, int(np.ceil(array.size * 0.1)))
    return {
        "week_count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "positive_rate": float((array > 0).mean()),
        "q10": float(np.quantile(array, 0.1)),
        "cvar10": float(np.sort(array)[:tail_count].mean()),
        "worst": float(array.min()),
    }


def select_risk_configuration(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("risk configuration rows are empty")
    evaluated = []
    for row in rows:
        candidate = dict(row)
        candidate["robust_score"] = float(candidate["mean"]) + float(candidate["cvar_penalty"]) * min(
            float(candidate["cvar10"]), 0.0
        )
        evaluated.append(candidate)
    evaluated.sort(
        key=lambda item: (
            -round(float(item["robust_score"]), 12),
            -float(item["mean"]),
            -float(item["cvar10"]),
            float(item["variance_penalty"])
            + float(item["correlation_penalty"])
            + float(item["cvar_penalty"]),
            float(item["variance_penalty"]),
            float(item["correlation_penalty"]),
            float(item["cvar_penalty"]),
        )
    )
    return evaluated[0]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def evaluate_portfolio_series(
    scored_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    *,
    candidate_pool_size: int,
    covariance_window: int,
    variance_penalty: float,
    correlation_penalty: float,
    rebalance_stride: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {"trade_date", "stock_id", "pred_score", "future_return"}
    if not required.issubset(scored_frame.columns):
        raise ValueError(f"scored frame missing columns: {sorted(required - set(scored_frame.columns))}")
    scored = scored_frame.copy()
    scored["trade_date"] = pd.to_datetime(scored["trade_date"], errors="coerce")
    scored["stock_id"] = scored["stock_id"].astype(str).str.zfill(6)
    scored["future_return"] = pd.to_numeric(scored["future_return"], errors="coerce")
    scored = scored.dropna(subset=["trade_date", "stock_id", "pred_score", "future_return"])
    rebalance_dates = select_rebalance_dates(scored, stride=rebalance_stride)

    weekly_rows: list[dict[str, object]] = []
    previous_selection: set[str] | None = None
    for rebalance_date in rebalance_dates:
        one_day = scored[scored["trade_date"] == rebalance_date].copy()
        selection: PortfolioSelection = select_top5_portfolio(
            one_day,
            market_frame,
            candidate_pool_size,
            covariance_window,
            variance_penalty,
            correlation_penalty,
        )
        selected_set = set(selection.selected_stock_ids)
        selected_returns = one_day[one_day["stock_id"].isin(selected_set)]["future_return"]
        if len(selected_returns) != 5:
            raise ValueError(f"missing future returns for selected portfolio on {rebalance_date}")
        portfolio_return = float(selected_returns.mean())
        baseline_return = float(one_day["future_return"].mean())
        weekly_rows.append(
            {
                "trade_date": pd.Timestamp(rebalance_date),
                "stock_ids": selection.selected_stock_ids,
                "portfolio_return": portfolio_return,
                "baseline_return": baseline_return,
                "excess_return": portfolio_return - baseline_return,
                "portfolio_variance": selection.portfolio_variance,
                "mean_correlation": selection.mean_correlation,
                "selection_score": selection.selection_score,
                "jaccard_vs_previous": (
                    _jaccard(previous_selection, selected_set) if previous_selection is not None else 1.0
                ),
                "degraded_reason": selection.degraded_reason,
            }
        )
        previous_selection = selected_set

    details = pd.DataFrame(weekly_rows)
    if details.empty:
        raise ValueError("no weekly portfolios were evaluated")
    statistics = calculate_weekly_statistics(details["portfolio_return"].astype(float).tolist())
    baseline_statistics = calculate_weekly_statistics(details["baseline_return"].astype(float).tolist())
    statistics.update(
        {
            "baseline_mean": baseline_statistics["mean"],
            "mean_excess_return": float(details["excess_return"].mean()),
            "baseline_outperformance_rate": float((details["excess_return"] > 0).mean()),
            "mean_jaccard": float(details["jaccard_vs_previous"].mean()),
            "degraded_week_count": int(details["degraded_reason"].notna().sum()),
        }
    )
    return details, statistics


def evaluate_parameter_grid(
    scored_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    *,
    candidate_pool_size: int,
    covariance_window: int,
    variance_penalties: tuple[float, ...],
    correlation_penalties: tuple[float, ...],
    cvar_penalties: tuple[float, ...],
    rebalance_stride: int,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, pd.DataFrame]]:
    leaderboard: list[dict[str, object]] = []
    details_by_key: dict[str, pd.DataFrame] = {}
    for variance_penalty, correlation_penalty in product(variance_penalties, correlation_penalties):
        details, statistics = evaluate_portfolio_series(
            scored_frame,
            market_frame,
            candidate_pool_size=candidate_pool_size,
            covariance_window=covariance_window,
            variance_penalty=float(variance_penalty),
            correlation_penalty=float(correlation_penalty),
            rebalance_stride=rebalance_stride,
        )
        details_key = f"variance={variance_penalty}|correlation={correlation_penalty}"
        details_by_key[details_key] = details
        for cvar_penalty in cvar_penalties:
            row: dict[str, object] = {
                "variance_penalty": float(variance_penalty),
                "correlation_penalty": float(correlation_penalty),
                "cvar_penalty": float(cvar_penalty),
                **statistics,
                "details_key": details_key,
            }
            row["robust_score"] = float(statistics["mean"]) + float(cvar_penalty) * min(
                float(statistics["cvar10"]), 0.0
            )
            leaderboard.append(row)
    selected = select_risk_configuration(
        [
            {
                "variance_penalty": float(row["variance_penalty"]),
                "correlation_penalty": float(row["correlation_penalty"]),
                "cvar_penalty": float(row["cvar_penalty"]),
                "mean": float(row["mean"]),
                "cvar10": float(row["cvar10"]),
            }
            for row in leaderboard
        ]
    )
    leaderboard.sort(
        key=lambda row: (
            -float(row["robust_score"]),
            -float(row["mean"]),
            -float(row["cvar10"]),
            float(row["variance_penalty"])
            + float(row["correlation_penalty"])
            + float(row["cvar_penalty"]),
        )
    )
    return selected, leaderboard, details_by_key
