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
    assert payload["ensemble_method"] == "equal_rank_average"
    assert payload["portfolio"] == {
        "size": 5,
        "candidate_pool_size": 15,
        "covariance_window": 60,
        "variance_penalty": 0.5,
        "correlation_penalty": 0.25,
        "weight": 0.2,
    }
    assert "score_overlay" not in payload
