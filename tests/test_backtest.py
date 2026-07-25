import pytest

from backtest import calculate_weekly_statistics, select_risk_configuration


def test_cvar10_uses_worst_ceil_ten_percent():
    stats = calculate_weekly_statistics([-0.10, -0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    assert stats["cvar10"] == pytest.approx(-0.075)


def test_risk_selection_uses_robust_score_then_simpler_penalty():
    rows = [
        {
            "variance_penalty": 0.5,
            "correlation_penalty": 0.25,
            "cvar_penalty": 0.5,
            "mean": 0.03,
            "cvar10": -0.01,
        },
        {
            "variance_penalty": 0.0,
            "correlation_penalty": 0.0,
            "cvar_penalty": 0.0,
            "mean": 0.025,
            "cvar10": -0.03,
        },
    ]
    selected = select_risk_configuration(rows)
    assert selected["variance_penalty"] == 0.5


def test_empty_weekly_returns_fail():
    with pytest.raises(ValueError):
        calculate_weekly_statistics([])
