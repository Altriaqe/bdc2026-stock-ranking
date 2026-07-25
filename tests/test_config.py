from config import ProjectConfig


def test_robust_production_defaults():
    config = ProjectConfig()
    assert config.portfolio_size == 5
    assert config.production_model_names == ("xgb_ranker", "lgb_ranker", "hgb_regressor")
    assert config.production_score_overlay_enabled is False
    assert config.purge_group_count == 5
    assert config.rebalance_stride == 5
    assert config.validation_ratio == 0.1
    assert config.inner_min_train_groups == 126
    assert config.portfolio_candidate_pool_size == 15
    assert config.portfolio_covariance_window == 60
    assert config.portfolio_weight == 0.2
