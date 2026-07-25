from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class PortfolioSelection:
    submission: pd.DataFrame
    candidates: list[dict[str, object]]
    selected_stock_ids: list[str]
    portfolio_variance: float
    mean_correlation: float
    selection_score: float
    degraded_reason: str | None


def shrink_model_weights(
    robust_scores: dict[str, float],
    shrinkage: float,
    cap: float,
) -> dict[str, float]:
    names = list(robust_scores)
    if not names:
        raise ValueError("no model scores")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between zero and one")
    if not 0.0 < cap <= 1.0 or cap * len(names) < 1.0 - 1e-12:
        raise ValueError("cap cannot support a unit-sum weight vector")

    equal = np.full(len(names), 1.0 / len(names), dtype=float)
    positive_values = []
    for name in names:
        try:
            score = float(robust_scores[name])
        except (TypeError, ValueError):
            score = 0.0
        positive_values.append(max(0.0, score) if np.isfinite(score) else 0.0)
    positive = np.asarray(positive_values, dtype=float)
    validation = positive / positive.sum() if positive.sum() > 0 else equal.copy()
    weights = shrinkage * equal + (1.0 - shrinkage) * validation

    for _ in range(len(names) * 3):
        over = weights > cap + 1e-12
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        receivers = weights < cap - 1e-12
        if not receivers.any():
            break
        capacity = cap - weights[receivers]
        base = weights[receivers]
        shares = base / base.sum() if base.sum() > 1e-12 else capacity / capacity.sum()
        addition = np.minimum(excess * shares, capacity)
        weights[receivers] += addition
        remaining = excess - float(addition.sum())
        if remaining > 1e-12:
            weights[receivers] += remaining * capacity / capacity.sum()

    weights = np.minimum(weights, cap)
    weights /= weights.sum()
    if np.max(weights) > cap + 1e-10:
        raise RuntimeError("failed to enforce model weight cap")
    return dict(zip(names, weights.tolist()))


def combine_rank_scores(
    score_map: dict[str, np.ndarray],
    model_weights: dict[str, float],
    stock_ids: pd.Series | np.ndarray | list[str],
) -> np.ndarray:
    if not score_map:
        raise ValueError("no model scores")
    stock_array = np.asarray(stock_ids, dtype=str)
    expected_length = len(stock_array)
    names = list(score_map)
    if set(names) != set(model_weights):
        raise ValueError("model score names and model weight names differ")
    weights = np.asarray([float(model_weights[name]) for name in names], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("model weights must be finite, non-negative, and non-zero")
    weights /= weights.sum()

    ranked_scores = []
    for name in names:
        scores = np.asarray(score_map[name], dtype=float)
        if scores.shape != (expected_length,):
            raise ValueError(f"model score length mismatch for {name}")
        clean_scores = np.where(np.isfinite(scores), scores, -np.inf)
        ranked_scores.append(rankdata(clean_scores, method="average") / max(expected_length, 1))
    return np.average(np.vstack(ranked_scores), axis=0, weights=weights)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    if np.allclose(array, array[0]):
        return np.full_like(array, 0.5, dtype=float)
    return rankdata(array, method="average") / len(array)


def _prepare_candidate_returns(
    market_frame: pd.DataFrame,
    stock_ids: list[str],
    target_date: pd.Timestamp,
    covariance_window: int,
) -> pd.DataFrame:
    history = market_frame.loc[
        (market_frame["trade_date"] <= target_date) & market_frame["stock_id"].isin(stock_ids),
        ["trade_date", "stock_id", "close"],
    ].copy()
    pivot = history.pivot_table(index="trade_date", columns="stock_id", values="close", aggfunc="last")
    pivot = pivot.sort_index().reindex(columns=stock_ids).ffill().tail(covariance_window + 1)
    returns = pivot.pct_change(fill_method=None).iloc[1:].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(returns) < covariance_window:
        raise ValueError("insufficient common return history for covariance estimation")
    return returns


def select_top5_portfolio(
    scored_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    candidate_pool_size: int,
    covariance_window: int,
    variance_penalty: float,
    correlation_penalty: float,
) -> PortfolioSelection:
    required_scored = {"stock_id", "trade_date", "pred_score"}
    required_market = {"stock_id", "trade_date", "close"}
    if not required_scored.issubset(scored_frame.columns):
        raise ValueError(f"scored frame missing columns: {sorted(required_scored - set(scored_frame.columns))}")
    if not required_market.issubset(market_frame.columns):
        raise ValueError(f"market frame missing columns: {sorted(required_market - set(market_frame.columns))}")
    if candidate_pool_size < 5 or covariance_window < 2:
        raise ValueError("candidate pool must be at least five and covariance window at least two")

    scores = scored_frame.copy()
    scores["stock_id"] = scores["stock_id"].astype(str).str.zfill(6)
    scores["trade_date"] = pd.to_datetime(scores["trade_date"], errors="coerce")
    scores["pred_score"] = pd.to_numeric(scores["pred_score"], errors="coerce")
    scores = scores.dropna(subset=["stock_id", "trade_date", "pred_score"])
    if scores.empty:
        raise ValueError("no valid scored stocks")
    target_date = scores["trade_date"].max()
    scores = scores[scores["trade_date"] == target_date].copy()

    market = market_frame.copy()
    market["stock_id"] = market["stock_id"].astype(str).str.zfill(6)
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.dropna(subset=["stock_id", "trade_date", "close"])
    market = market[(market["trade_date"] <= target_date) & (market["close"] > 0)].copy()

    stock_quality = market.groupby("stock_id", sort=False).agg(
        latest_date=("trade_date", "max"),
        close_count=("close", "count"),
    )
    eligible_ids = stock_quality.index[
        (stock_quality["latest_date"] == target_date)
        & (stock_quality["close_count"] >= covariance_window + 1)
    ]
    eligible = scores[scores["stock_id"].isin(eligible_ids)].copy()
    eligible = eligible.sort_values(["pred_score", "stock_id"], ascending=[False, True])
    candidates = eligible.head(candidate_pool_size).reset_index(drop=True)
    if len(candidates) < 5:
        raise ValueError(f"fewer than five eligible stocks remain: {len(candidates)}")

    candidate_ids = candidates["stock_id"].tolist()
    candidate_report = candidates[["stock_id", "pred_score"]].to_dict(orient="records")
    degraded_reason: str | None = None
    try:
        returns = _prepare_candidate_returns(market, candidate_ids, target_date, covariance_window)
        covariance = LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_
        diagonal = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
        correlation = covariance / np.outer(diagonal, diagonal)
        np.fill_diagonal(correlation, 1.0)
    except Exception as error:
        degraded_reason = f"covariance_fallback:{type(error).__name__}:{error}"
        selected_ids = candidate_ids[:5]
        submission = pd.DataFrame({"stock_id": sorted(selected_ids), "weight": [0.2] * 5})
        return PortfolioSelection(
            submission=submission,
            candidates=candidate_report,
            selected_stock_ids=sorted(selected_ids),
            portfolio_variance=0.0,
            mean_correlation=0.0,
            selection_score=float(candidates.head(5)["pred_score"].mean()),
            degraded_reason=degraded_reason,
        )

    subsets = list(combinations(range(len(candidate_ids)), 5))
    prediction_values = np.empty(len(subsets), dtype=float)
    variance_values = np.empty(len(subsets), dtype=float)
    correlation_values = np.empty(len(subsets), dtype=float)
    equal_weights = np.full(5, 0.2, dtype=float)
    predictions = candidates["pred_score"].to_numpy(dtype=float)
    upper_indices = np.triu_indices(5, k=1)
    for subset_index, subset in enumerate(subsets):
        indices = np.asarray(subset, dtype=int)
        subset_covariance = covariance[np.ix_(indices, indices)]
        subset_correlation = correlation[np.ix_(indices, indices)]
        prediction_values[subset_index] = float(predictions[indices].mean())
        variance_values[subset_index] = float(equal_weights @ subset_covariance @ equal_weights)
        correlation_values[subset_index] = float(subset_correlation[upper_indices].mean())

    variance_ranks = _percentile_rank(variance_values)
    correlation_ranks = _percentile_rank(correlation_values)
    objective_values = (
        prediction_values
        - float(variance_penalty) * variance_ranks
        - float(correlation_penalty) * correlation_ranks
    )
    ordered_indices = sorted(
        range(len(subsets)),
        key=lambda index: (
            -float(objective_values[index]),
            tuple(candidate_ids[position] for position in subsets[index]),
        ),
    )
    best_index = ordered_indices[0]
    best_subset = subsets[best_index]
    selected_ids = sorted(candidate_ids[position] for position in best_subset)
    submission = pd.DataFrame({"stock_id": selected_ids, "weight": [0.2] * 5})
    return PortfolioSelection(
        submission=submission,
        candidates=candidate_report,
        selected_stock_ids=selected_ids,
        portfolio_variance=float(variance_values[best_index]),
        mean_correlation=float(correlation_values[best_index]),
        selection_score=float(objective_values[best_index]),
        degraded_reason=degraded_reason,
    )
