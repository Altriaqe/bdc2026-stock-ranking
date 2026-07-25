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
            rows.append(
                {
                    "stock_id": f"{stock_index:06d}",
                    "trade_date": date,
                    "open": close - 0.1,
                    "close": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def scored_frame(market_frame: pd.DataFrame) -> pd.DataFrame:
    latest = market_frame["trade_date"].max()
    stocks = sorted(market_frame["stock_id"].unique())
    return pd.DataFrame(
        {
            "stock_id": stocks,
            "trade_date": latest,
            "pred_score": np.linspace(1.0, 0.0, len(stocks)),
        }
    )
