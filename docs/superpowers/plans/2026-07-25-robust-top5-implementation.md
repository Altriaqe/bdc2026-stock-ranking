# BDC2026 Robust Top5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overfit Top1 production path with a deterministic, purged-walk-forward-validated, three-model Top5 equal-weight portfolio that improves historical downside stability without sacrificing expected weekly return.

**Architecture:** Keep the official Baostock-compatible acquisition, feature engineering, and T+1/T+5 label path. Add purged weekly validation, rank-based model blending, and a focused portfolio module that enumerates five-stock subsets from the top 15 candidates using prediction, Ledoit-Wolf variance, and correlation. Training writes every selected parameter into metadata; inference reads only metadata and visible market history.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, pytest, Docker, uv.

## Global Constraints

- Output must be UTF-8 `output/result.csv` with header `stock_id,weight`.
- Output must contain exactly 5 distinct stocks, each with weight `0.2`, total weight `1.0`.
- Return label and scoring must use `open(T+5) / open(T+1) - 1`.
- Training must remain under 8 hours; inference must remain under 5 minutes; image must remain under 10GB.
- Training and inference must run without network access and must reproduce byte-identical results with random seed `42`.
- Data acquisition remains the official-baseline-compatible Baostock daily, HS300, post-adjusted path; no unreported external model or data is introduced.
- `data/test.csv` and B-stage future returns must never participate in training or parameter selection.
- Production score overlay `0.7 * zscore(volume_ratio_20)` must be removed; volume features remain ordinary model inputs.

---

## File Structure

- Create `tests/test_validation.py`: purging, weekly sampling, and official-weight metric tests.
- Create `tests/test_config.py`: exact production-default tests.
- Create `tests/test_portfolio.py`: ensemble shrinkage, covariance selection, equal-weight output, and determinism tests.
- Create `tests/test_inference.py`: metadata-driven Top5 integration test with lightweight fake models.
- Create `code/src/portfolio.py`: model-score blending and five-stock subset selection only.
- Create `code/src/backtest.py`: non-overlapping weekly portfolio evaluation and robust parameter-grid selection only.
- Modify `code/src/config.py`: all fixed validation, ensemble, and portfolio parameters.
- Modify `code/src/featurework.py`: purged time splits, rebalance dates, and official-weight metrics.
- Modify `code/src/train.py`: outer-fold predictions, robust model weights and risk-grid choice, metadata/report persistence.
- Modify `code/src/test.py`: load three models, remove overlay, call portfolio selection, write deterministic Top5 output.
- Modify `code/src/compliance.py`: require exact production position count and exact configured weights when requested.
- Modify `pyproject.toml` and `uv.lock`: declare pytest as a development dependency only.
- Modify `readme.md`, `docs/project-guide.md`, and `docs/session-operation-summary.md`: describe the validated production path and final evidence.

---

### Task 1: Test Harness and Robust Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `code/src/config.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `ProjectConfig` fields consumed by validation, portfolio, backtest, training, and inference.
- Produces: test import path that exposes `code/src` without packaging the application.

- [ ] **Step 1: Add the test import path and deterministic portfolio fixtures**

Create `tests/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture
def market_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2025-09-01", periods=80)
    rows = []
    for stock_index in range(20):
        for day_index, date in enumerate(dates):
            deterministic_wave = np.sin(day_index / (3.0 + stock_index / 10.0))
            close = 100.0 + stock_index + 0.05 * day_index + deterministic_wave
            rows.append({
                "stock_id": f"{stock_index:06d}",
                "trade_date": date,
                "open": close - 0.1,
                "close": close,
                "high": close + 0.2,
                "low": close - 0.2,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def scored_frame(market_frame: pd.DataFrame) -> pd.DataFrame:
    latest = market_frame["trade_date"].max()
    stocks = sorted(market_frame["stock_id"].unique())
    return pd.DataFrame({
        "stock_id": stocks,
        "trade_date": latest,
        "pred_score": np.linspace(1.0, 0.0, len(stocks)),
    })
```

- [ ] **Step 2: Add the failing configuration test**

```python
from config import ProjectConfig


def test_robust_production_defaults():
    config = ProjectConfig()
    assert config.portfolio_size == 5
    assert config.production_model_names == ("xgb_ranker", "lgb_ranker", "hgb_regressor")
    assert config.production_score_overlay_enabled is False
    assert config.purge_group_count == 5
    assert config.rebalance_stride == 5
    assert config.portfolio_candidate_pool_size == 15
    assert config.portfolio_covariance_window == 60
    assert config.portfolio_weight == 0.2
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_config.py::test_robust_production_defaults -q`

Expected: FAIL because the robust configuration fields do not exist and `portfolio_size` is still `1`.

- [ ] **Step 4: Add exact configuration values**

Add these frozen dataclass fields to `ProjectConfig`:

```python
portfolio_size: int = 5
production_model_names: tuple[str, ...] = ("xgb_ranker", "lgb_ranker", "hgb_regressor")
production_score_overlay_enabled: bool = False
purge_group_count: int = 5
rebalance_stride: int = 5
outer_walk_forward_fold_count: int = 4
inner_walk_forward_fold_count: int = 3
min_train_groups: int = 252
portfolio_candidate_pool_size: int = 15
portfolio_covariance_window: int = 60
portfolio_weight: float = 0.2
model_weight_shrinkage: float = 0.5
model_weight_cap: float = 0.5
variance_penalty_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
correlation_penalty_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
cvar_penalty_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
backtest_report_filename: str = "backtest_report.json"
portfolio_report_filename: str = "portfolio_report.json"
```

Add `build_backtest_report_path()` and `build_portfolio_report_path()` methods parallel to the existing metadata path methods.

- [ ] **Step 5: Add pytest as a development dependency and lock it**

Add:

```toml
[dependency-groups]
dev = ["pytest>=8.2,<9.0"]
```

Run: `uv lock`

Expected: `uv.lock` updates without changing production dependency constraints.

- [ ] **Step 6: Run the test**

Run: `pytest tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml uv.lock code/src/config.py tests/conftest.py tests/test_config.py
git commit -m "test: define robust top5 production defaults"
```

---

### Task 2: Purged Walk-Forward and Official Weekly Metrics

**Files:**
- Modify: `code/src/featurework.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Produces: `build_purged_walk_forward_splits(frame, fold_count, validation_ratio, min_train_groups, purge_groups) -> list[tuple[pd.DataFrame, pd.DataFrame]]`.
- Produces: `select_rebalance_dates(frame, stride) -> list[pd.Timestamp]`.
- Produces: `calculate_top_k_return_metrics(scored_frame, k, rebalance_stride=5) -> dict[str, float]` using official equal weights.
- Consumes: `future_return`, `trade_date`, `stock_id`, and `pred_score` columns.

- [ ] **Step 1: Write failing purging and metric tests**

```python
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
    for day_index, date in enumerate(dates):
        for stock_index in range(stocks):
            rows.append({
                "trade_date": date,
                "stock_id": f"{stock_index:06d}",
                "future_return": (stock_index - 2) / 100.0,
                "pred_score": float(stock_index),
            })
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
    for train, valid in splits:
        all_dates = sorted(frame["trade_date"].unique())
        train_last = all_dates.index(train["trade_date"].max())
        valid_first = all_dates.index(valid["trade_date"].min())
        assert valid_first - train_last > 5


def test_rebalance_dates_are_non_overlapping():
    dates = select_rebalance_dates(make_frame(), stride=5)
    assert dates == list(pd.bdate_range("2026-01-01", periods=40)[::5])


def test_top5_metric_matches_official_equal_weight_return():
    metrics = calculate_top_k_return_metrics(make_frame(days=10), k=5, rebalance_stride=5)
    expected = np.mean([0.01, 0.02, 0.03, 0.00, -0.01])
    assert metrics["pred_top_k_return_mean"] == round(expected, 8)
    assert metrics["evaluated_week_count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation.py -q`

Expected: FAIL because the purged split and rebalance functions do not exist and the old metric sums TopK returns.

- [ ] **Step 3: Implement purged splits and rebalance dates**

```python
def select_rebalance_dates(frame: pd.DataFrame, stride: int) -> list[pd.Timestamp]:
    if stride < 1:
        raise ValueError("rebalance stride must be positive")
    dates = sorted(pd.to_datetime(frame["trade_date"].drop_duplicates()).tolist())
    return dates[::stride]


def build_purged_walk_forward_splits(
    ranking_frame: pd.DataFrame,
    *,
    fold_count: int,
    validation_ratio: float,
    min_train_groups: int,
    purge_groups: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    unique_dates = sorted(ranking_frame["trade_date"].drop_duplicates().tolist())
    validation_size = max(1, int(round(len(unique_dates) * validation_ratio)))
    available = len(unique_dates) - min_train_groups - purge_groups
    actual_folds = min(fold_count, max(0, available // validation_size))
    if actual_folds < 1:
        raise ValueError("insufficient dates for purged walk-forward")
    splits = []
    for fold_index in range(actual_folds):
        validation_end = len(unique_dates) - (actual_folds - fold_index - 1) * validation_size
        validation_start = validation_end - validation_size
        train_end = validation_start - purge_groups
        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[validation_start:validation_end]
        if len(train_dates) < min_train_groups:
            continue
        splits.append((
            ranking_frame[ranking_frame["trade_date"].isin(train_dates)].copy(),
            ranking_frame[ranking_frame["trade_date"].isin(valid_dates)].copy(),
        ))
    if not splits:
        raise ValueError("no valid purged walk-forward split")
    return splits
```

- [ ] **Step 4: Replace summed TopK metrics with official weighted weekly metrics**

For each selected rebalance date, compute predicted and oracle TopK returns with `mean()`, cross-sectional equal-weight baseline with `mean()`, then return these exact fields:

```python
{
    "pred_top_k_return_mean": round(float(np.mean(predicted)), 8),
    "pred_top_k_return_median": round(float(np.median(predicted)), 8),
    "pred_top_k_return_std": round(float(np.std(predicted)), 8),
    "pred_top_k_positive_rate": round(float(np.mean(np.asarray(predicted) > 0)), 8),
    "pred_top_k_q10": round(float(np.quantile(predicted, 0.1)), 8),
    "pred_top_k_cvar10": round(float(np.mean(np.sort(predicted)[:max(1, int(np.ceil(len(predicted) * 0.1)))])), 8),
    "pred_top_k_worst": round(float(np.min(predicted)), 8),
    "oracle_top_k_return_mean": round(float(np.mean(oracle)), 8),
    "baseline_top_k_return_mean": round(float(np.mean(baseline)), 8),
    "top_k_relative_score": round(normalized_score, 8),
    "evaluated_week_count": len(predicted),
}
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_validation.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/src/featurework.py tests/test_validation.py
git commit -m "feat: add purged weekly portfolio validation"
```

---

### Task 3: Robust Ensemble and Top5 Portfolio Module

**Files:**
- Create: `code/src/portfolio.py`
- Create: `tests/test_portfolio.py`

**Interfaces:**
- Produces: `shrink_model_weights(robust_scores, shrinkage, cap) -> dict[str, float]`.
- Produces: `combine_rank_scores(score_map, model_weights, stock_ids) -> np.ndarray`.
- Produces: `select_top5_portfolio(scored_frame, market_frame, candidate_pool_size, covariance_window, variance_penalty, correlation_penalty) -> PortfolioSelection`.
- Produces: `PortfolioSelection(submission, candidates, variance, mean_correlation, degraded_reason)`.

- [ ] **Step 1: Write failing weight and portfolio tests**

```python
import numpy as np
import pandas as pd

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
    assert weights == {"xgb": 1 / 3, "lgb": 1 / 3, "hgb": 1 / 3}


def test_portfolio_is_five_stock_equal_weight_and_deterministic(market_frame, scored_frame):
    first = select_top5_portfolio(scored_frame, market_frame, 15, 60, 0.5, 0.25)
    second = select_top5_portfolio(scored_frame, market_frame, 15, 60, 0.5, 0.25)
    assert first.submission.equals(second.submission)
    assert first.submission.shape == (5, 2)
    assert first.submission["stock_id"].nunique() == 5
    assert first.submission["weight"].tolist() == [0.2] * 5
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_portfolio.py -q`

Expected: FAIL because `portfolio.py` does not exist.

- [ ] **Step 3: Implement deterministic model-weight shrinkage**

```python
def shrink_model_weights(
    robust_scores: dict[str, float],
    shrinkage: float,
    cap: float,
) -> dict[str, float]:
    names = list(robust_scores)
    if not names:
        raise ValueError("no model scores")
    equal = np.full(len(names), 1.0 / len(names))
    positive = np.array([
        max(0.0, float(robust_scores[name])) if np.isfinite(robust_scores[name]) else 0.0
        for name in names
    ])
    validation = positive / positive.sum() if positive.sum() > 0 else equal.copy()
    weights = shrinkage * equal + (1.0 - shrinkage) * validation
    for _ in range(len(names) + 1):
        excess = float(np.maximum(weights - cap, 0.0).sum())
        weights = np.minimum(weights, cap)
        if excess <= 1e-12:
            break
        receivers = weights < cap - 1e-12
        if not receivers.any():
            break
        base = weights[receivers]
        shares = base / base.sum() if base.sum() > 0 else np.full(receivers.sum(), 1 / receivers.sum())
        weights[receivers] += excess * shares
    weights /= weights.sum()
    return dict(zip(names, weights.tolist()))
```

- [ ] **Step 4: Implement rank blending and subset enumeration**

Use `scipy.stats.rankdata(method="average")` to convert each model score to `[0,1]` percentile rank. For portfolio selection:

1. require all candidates to have the global latest market date and 60 valid close returns;
2. take top 15 by blended score, tie-breaking by `stock_id` ascending;
3. estimate the 15x15 covariance with `sklearn.covariance.LedoitWolf`;
4. enumerate `itertools.combinations(range(candidate_count), 5)`;
5. compute equal-weight variance and average off-diagonal correlation per subset;
6. percentile-rank variance and correlation over all subsets;
7. maximize mean blended score minus configured penalties;
8. tie-break lexicographically by the five stock codes;
9. return sorted selected stocks with `weight=0.2`.

The implementation must raise `ValueError` when fewer than five eligible stocks remain and use prediction-only Top5 with a recorded `degraded_reason` if Ledoit-Wolf estimation fails.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_portfolio.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/src/portfolio.py tests/test_portfolio.py
git commit -m "feat: select robust equal-weight top5 portfolios"
```

---

### Task 4: Weekly Backtest and Risk-Grid Selection

**Files:**
- Create: `code/src/backtest.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: validation frames containing `trade_date`, `stock_id`, `future_return`, model scores, and visible market history.
- Produces: `evaluate_portfolio_series(...) -> tuple[pd.DataFrame, dict[str, float]]`.
- Produces: `select_risk_configuration(...) -> dict[str, float]`.
- Produces: JSON-serializable outer-fold details and parameter leaderboard.

- [ ] **Step 1: Write failing CVaR and selection tests**

```python
from backtest import calculate_weekly_statistics, select_risk_configuration


def test_cvar10_uses_worst_ceil_ten_percent():
    stats = calculate_weekly_statistics([-0.10, -0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    assert stats["cvar10"] == -0.075


def test_risk_selection_uses_robust_score_then_simpler_penalty():
    rows = [
        {"variance_penalty": 0.5, "correlation_penalty": 0.25, "cvar_penalty": 0.5, "mean": 0.03, "cvar10": -0.01},
        {"variance_penalty": 0.0, "correlation_penalty": 0.0, "cvar_penalty": 0.0, "mean": 0.025, "cvar10": -0.03},
    ]
    selected = select_risk_configuration(rows)
    assert selected["variance_penalty"] == 0.5
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_backtest.py -q`

Expected: FAIL because `backtest.py` does not exist.

- [ ] **Step 3: Implement exact weekly statistics**

```python
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
```

- [ ] **Step 4: Implement parameter-grid evaluation**

Evaluate the Cartesian product of variance, correlation, and CVaR penalty grids. For each risk pair, build a portfolio on every non-overlapping validation date using only market rows dated at or before that date. Compute official equal-weight future return and the statistics above. For each CVaR penalty, compute:

```python
robust_score = statistics["mean"] + cvar_penalty * min(statistics["cvar10"], 0.0)
```

Sort by robust score descending, mean descending, CVaR descending, then total penalty ascending. Persist every parameter row and every weekly selected stock set.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_backtest.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/src/backtest.py tests/test_backtest.py
git commit -m "feat: evaluate robust weekly portfolio grids"
```

---

### Task 5: Training Integration and Production Metadata

**Files:**
- Modify: `code/src/train.py`
- Modify: `code/src/featurework.py`
- Create: `tests/test_training_metadata.py`

**Interfaces:**
- Consumes: purged outer folds, three trained models, weekly metrics, and risk-grid selector.
- Produces: all three production model files.
- Produces: `model_metadata.json` with `selected_models`, `ensemble_method="shrunk_rank_average"`, `ensemble_weights`, and `portfolio` configuration.
- Produces: `backtest_report.json` with outer-fold weekly details and baselines.

- [ ] **Step 1: Write failing metadata-schema test**

```python
from train import build_production_metadata


def test_metadata_contains_reproducible_portfolio_configuration():
    payload = build_production_metadata(
        feature_columns=["close_return_5"],
        feature_windows=(3, 5, 10, 20),
        model_weights={"xgb_ranker": 0.4, "lgb_ranker": 0.35, "hgb_regressor": 0.25},
        variance_penalty=0.5,
        correlation_penalty=0.25,
    )
    assert payload["selected_models"] == ["xgb_ranker", "lgb_ranker", "hgb_regressor"]
    assert payload["ensemble_method"] == "shrunk_rank_average"
    assert payload["portfolio"] == {
        "size": 5,
        "candidate_pool_size": 15,
        "covariance_window": 60,
        "variance_penalty": 0.5,
        "correlation_penalty": 0.25,
        "weight": 0.2,
    }
    assert "score_overlay" not in payload
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_training_metadata.py -q`

Expected: FAIL because the metadata builder does not exist and overlay is still serialized.

- [ ] **Step 3: Refactor training around purged outer folds**

Replace `build_walk_forward_splits` calls with `build_purged_walk_forward_splits`. For each outer fold:

1. fit XGB, LGB, and HGB on the purged train frame;
2. generate all three score columns on validation;
3. use preceding folds for model robust scores when available, equal weights in the first fold;
4. evaluate current Top1, XGB Top5, equal rank ensemble Top5, and shrunk-rank risk-controlled Top5;
5. append weekly details and fold metrics to `backtest_report.json`.

After outer validation, derive final model robust scores from the median fold `robust_score`, shrink/cap them, select the risk configuration by median outer-fold rank, and train all three models on the full labeled frame.

- [ ] **Step 4: Serialize exact production metadata**

Add `build_production_metadata` and ensure the written JSON contains no overlay branch. Store fold counts, purge groups, rebalance stride, risk-grid winner, model weights, feature columns, random seed, and result columns. Store training-data maximum date and SHA-256 so the inference report identifies the exact input snapshot.

- [ ] **Step 5: Run focused and full tests**

Run: `pytest tests/test_training_metadata.py tests/test_validation.py tests/test_backtest.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/src/train.py code/src/featurework.py tests/test_training_metadata.py
git commit -m "feat: train robust three-model weekly ensemble"
```

---

### Task 6: Metadata-Driven Inference and Compliance

**Files:**
- Modify: `code/src/test.py`
- Modify: `code/src/compliance.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: three model files, metadata ensemble weights, latest visible market data, and portfolio configuration.
- Produces: exact five-row `output/result.csv` and `temp/portfolio_report.json`.
- Produces: compliance report requiring five positions at 0.2 each.

- [ ] **Step 1: Write failing compliance and inference tests**

```python
import pandas as pd
import pytest

from compliance import validate_submission_frame


def test_production_submission_requires_exact_equal_weight_top5():
    valid = pd.DataFrame({
        "stock_id": ["000001", "000002", "000003", "000004", "000005"],
        "weight": [0.2] * 5,
    })
    result = validate_submission_frame(valid, max_positions=5, exact_positions=5, exact_weight=0.2)
    assert result.weight_sum == 1.0

    with pytest.raises(ValueError):
        validate_submission_frame(valid.iloc[:4], max_positions=5, exact_positions=5, exact_weight=0.2)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_inference.py -q`

Expected: FAIL because exact position and weight checks do not exist.

- [ ] **Step 3: Remove overlay and integrate rank ensemble**

Delete `build_default_score_overlay` and `apply_score_overlay`. Load `ensemble_weights` from metadata, call `combine_rank_scores`, and pass the scored inference frame plus standardized visible market history to `select_top5_portfolio` using metadata parameters.

Write `portfolio_report.json` with the top-15 candidates, selected five, predicted scores, variance, mean correlation, penalties, latest visible date, degradation reason, and result SHA-256.

- [ ] **Step 4: Add strict production compliance**

Extend `validate_submission_frame` with optional `exact_positions` and `exact_weight`. Require exact five rows and `np.allclose(weight, 0.2, atol=1e-12)` from `test.py`; retain generic official maximum checks for other callers.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_inference.py tests/test_portfolio.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add code/src/test.py code/src/compliance.py tests/test_inference.py
git commit -m "feat: generate deterministic top5 production results"
```

---

### Task 7: Backtest, Select, and Train on Current Visible Data

**Files:**
- Modify: `model/xgb_ranker_v3/*` through the training command; model files remain ignored by Git.
- Modify: `output/result.csv`
- Modify: `temp/*.json` through validation commands; temp files remain ignored by Git.
- Modify: `readme.md`
- Modify: `docs/project-guide.md`
- Modify: `docs/session-operation-summary.md`

**Interfaces:**
- Produces: empirical comparison of current Top1, simple Top5, equal ensemble Top5, and risk-controlled Top5.
- Produces: current visible-data production model and exact result.

- [ ] **Step 1: Run all tests before expensive training**

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 2: Run full training with timing**

Run:

```powershell
Measure-Command { python code/src/train.py --experiment-name xgb_ranker_v3 --feature-preset alpha_v1 --production-models xgb_ranker,lgb_ranker,hgb_regressor --production-portfolio-size 5 }
```

Expected: exit code 0, elapsed time under 8 hours, three model files plus metadata and backtest report written.

- [ ] **Step 3: Audit backtest evidence**

Run a read-only report script that prints each strategy's outer-fold mean, median, positive rate, CVaR10, worst week, and fold standard deviation. Confirm the selected strategy is determined by the documented ordering, not manually overwritten. If risk-controlled Top5 fails the design fallback gate, switch production to the validated simpler Top5 alternative and record the reason.

- [ ] **Step 4: Generate and validate result**

Run:

```powershell
python code/src/test.py --experiment-name xgb_ranker_v3
python code/src/score.py
```

Expected: `result.csv` has five unique stocks at 0.2 each. `score.py` is diagnostic only on the old March scoring file and must not drive B-stage parameter changes.

- [ ] **Step 5: Update documentation with measured values**

Replace old Top1 and overlay descriptions with the actual selected strategy, backtest table, training time, inference time, selected stocks, metadata paths, and the statement that B-stage final data must be refreshed after the 2026-07-31 close.

- [ ] **Step 6: Commit source, result, and documentation**

```powershell
git add code/src tests pyproject.toml uv.lock output/result.csv readme.md docs/project-guide.md docs/session-operation-summary.md
git commit -m "feat: deliver robust top5 B-stage strategy"
```

---

### Task 8: Same-Source Refresh, Determinism, Docker, and Remote Delivery

**Files:**
- Modify: `data/*.csv` through the official-compatible downloader; data stays ignored by Git and is included in Docker build context.
- Modify: `output/result.csv`
- Modify: `readme.md` and reports with final measured evidence.
- Create: final team tar outside Git tracking.

**Interfaces:**
- Produces: B-stage model trained through the latest visible 2026-07-31 market close.
- Produces: byte-identical local and Docker results, final image, commit, and pushed `origin/main`.

- [ ] **Step 1: Refresh the official-compatible data snapshot**

After the 2026-07-31 close is available, run:

```powershell
python get_stock_data.py --start-date 2024-01-01 --end-date 2026-07-31 --output data/stock_data.csv
```

Normalize the downloaded file into `data/train.csv` without creating a future-label `data/test.csv` split. Validate 300 stocks where available, no duplicate `(stock_id, trade_date)`, maximum date `2026-07-31`, required 12 columns, and finite positive OHLC values.

- [ ] **Step 2: Run final training and inference twice**

For each run, remove only generated model/output/temp artifacts inside their exact project subdirectories, then execute `python code/src/train.py` and `python code/src/test.py`. Record SHA-256 of `output/result.csv` after each run.

Expected: both hashes match and both outputs contain the same five ordered stocks at 0.2.

- [ ] **Step 3: Build and run Docker without network**

Run:

```powershell
docker build -t bdc2026:latest .
docker compose up --force-recreate
docker run --rm --network none bdc2026:latest /bin/sh -lc "/app/run.sh"
docker images bdc2026:latest --format "{{.Repository}}:{{.Tag}} {{.Size}}"
```

Expected: both Compose and explicit `--network none` runs succeed; inference is under five minutes; image size is under 10GB; container result hash matches the local final hash.

- [ ] **Step 4: Export the submission image**

Run: `docker save -o 霹雳-B阶段.tar bdc2026:latest`

Expected: uncompressed tar exists, loads with `docker load -i`, and is under 10GB.

- [ ] **Step 5: Final completion audit**

Verify every official constraint, every design acceptance item, full pytest output, backtest detail coverage, current data maximum date, training/inference timing, five-row result, two local hashes, Docker hash, image size, tar size, Git status, and documentation accuracy. Treat any missing evidence as incomplete.

- [ ] **Step 6: Commit and push final evidence**

```powershell
git add output/result.csv readme.md docs/project-guide.md docs/session-operation-summary.md
git commit -m "chore: finalize B-stage reproducible submission"
git push origin main
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: clean worktree and remote `main` hash equal to local `HEAD`.
