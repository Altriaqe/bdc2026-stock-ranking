import numpy as np
import pandas as pd
import pytest

from portfolio import combine_rank_scores, select_top5_portfolio, shrink_model_weights


def test_shrunk_weights_are_capped_and_normalized():
    weights = shrink_model_weights(
        {"xgb_ranker": 10.0, "lgb_ranker": 1.0, "hgb_regressor": 0.0},
        shrinkage=0.5,
        cap=0.5,
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert max(weights.values()) <= 0.5 + 1e-12


def test_invalid_scores_fall_back_to_equal_weights():
    weights = shrink_model_weights({"xgb": -1.0, "lgb": float("nan"), "hgb": 0.0}, 0.5, 0.5)
    assert weights == pytest.approx({"xgb": 1 / 3, "lgb": 1 / 3, "hgb": 1 / 3})


def test_rank_scores_use_named_weights():
    combined = combine_rank_scores(
        {"xgb": np.array([3.0, 2.0, 1.0]), "lgb": np.array([1.0, 2.0, 3.0])},
        {"xgb": 0.75, "lgb": 0.25},
        ["000001", "000002", "000003"],
    )
    assert combined[0] > combined[2]


def test_portfolio_is_five_stock_equal_weight_and_deterministic(market_frame, scored_frame):
    first = select_top5_portfolio(scored_frame, market_frame, 15, 60, 0.5, 0.25)
    second = select_top5_portfolio(scored_frame, market_frame, 15, 60, 0.5, 0.25)
    assert first.submission.equals(second.submission)
    assert first.submission.shape == (5, 2)
    assert first.submission["stock_id"].nunique() == 5
    assert first.submission["weight"].tolist() == [0.2] * 5
    assert first.degraded_reason is None


def test_portfolio_rejects_fewer_than_five_eligible_stocks(market_frame, scored_frame):
    with pytest.raises(ValueError, match="fewer than five"):
        select_top5_portfolio(scored_frame.head(4), market_frame, 15, 60, 0.5, 0.25)


def test_covariance_failure_has_deterministic_prediction_fallback(monkeypatch, market_frame, scored_frame):
    class BrokenLedoitWolf:
        def fit(self, values):
            raise ArithmeticError("synthetic covariance failure")

    monkeypatch.setattr("portfolio.LedoitWolf", BrokenLedoitWolf)
    result = select_top5_portfolio(scored_frame, market_frame, 15, 60, 0.5, 0.25)
    assert result.submission["stock_id"].tolist() == ["000000", "000001", "000002", "000003", "000004"]
    assert result.degraded_reason is not None
