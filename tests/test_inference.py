import pandas as pd
import pytest

from compliance import validate_submission_frame


def test_production_submission_requires_exact_equal_weight_top5():
    valid = pd.DataFrame(
        {
            "stock_id": ["000001", "000002", "000003", "000004", "000005"],
            "weight": [0.2] * 5,
        }
    )
    result = validate_submission_frame(
        valid,
        max_positions=5,
        required_columns=("stock_id", "weight"),
        weight_upper_bound=1.0,
        exact_positions=5,
        exact_weight=0.2,
    )
    assert result.weight_sum == 1.0

    with pytest.raises(ValueError, match="必须包含 5 只"):
        validate_submission_frame(
            valid.iloc[:4],
            max_positions=5,
            required_columns=("stock_id", "weight"),
            weight_upper_bound=1.0,
            exact_positions=5,
            exact_weight=0.2,
        )


def test_production_submission_rejects_unequal_weights():
    invalid = pd.DataFrame(
        {
            "stock_id": ["000001", "000002", "000003", "000004", "000005"],
            "weight": [0.3, 0.2, 0.2, 0.2, 0.1],
        }
    )
    with pytest.raises(ValueError, match="每只股票权重必须"):
        validate_submission_frame(
            invalid,
            max_positions=5,
            required_columns=("stock_id", "weight"),
            weight_upper_bound=1.0,
            exact_positions=5,
            exact_weight=0.2,
        )
