from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "model"
    output_dir: Path = PROJECT_ROOT / "output"
    temp_dir: Path = PROJECT_ROOT / "temp"
    docs_dir: Path = PROJECT_ROOT / "docs"

    train_filename: str = "train.csv"
    test_filename: str = "test.csv"
    full_data_filename: str = "stock_data.csv"
    inference_filename: str = "train.csv"
    result_filename: str = "result.csv"

    training_summary_filename: str = "training_summary.json"
    project_config_filename: str = "project_config.json"
    training_report_filename: str = "training_report.json"
    model_metadata_filename: str = "model_metadata.json"
    feature_importance_filename: str = "feature_importance.csv"
    ranker_model_filename: str = "xgb_ranker.json"
    ranker_model_lgb_filename: str = "lgb_ranker.txt"
    ranker_model_hgb_filename: str = "hgb_regressor.pkl"
    submission_check_filename: str = "submission_check.json"
    self_score_filename: str = "self_score.json"

    experiment_name: str = "xgb_ranker_v3"
    docker_image_name: str = "bdc2026"
    submission_tar_placeholder: str = "your_team_name.tar"
    result_columns: tuple[str, str] = ("stock_id", "weight")
    max_portfolio_size: int = 5
    portfolio_size: int = 5
    production_model_names: tuple[str, ...] = ("xgb_ranker", "lgb_ranker", "hgb_regressor")
    production_score_overlay_enabled: bool = False
    production_score_overlay_feature: str = "volume_ratio_20"
    production_score_overlay_weight: float = 0.7
    production_score_overlay_method: str = "additive_zscore"
    weight_upper_bound: float = 1.0
    top_k_metric: int = 5

    feature_windows: tuple[int, ...] = (3, 5, 10, 20)
    feature_preset: str = "alpha_v1"
    future_buy_offset: int = 1
    future_sell_offset: int = 5
    validation_ratio: float = 0.1
    min_train_groups: int = 252
    inner_min_train_groups: int = 126
    walk_forward_fold_count: int = 4
    purge_group_count: int = 5
    rebalance_stride: int = 5
    outer_walk_forward_fold_count: int = 4
    inner_walk_forward_fold_count: int = 3
    label_bucket_count: int = 10
    label_return_clip_quantile: float = 0.0
    head_sample_weight_quantile: float = 0.0
    head_sample_weight_value: float = 1.0
    augmentation_noise_std: float = 0.01
    augmentation_noise_fraction: float = 0.3
    continuous_target_clip_quantile: float = 0.02
    random_seed: int = 42

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

    xgb_objective: str = "rank:ndcg"
    xgb_eval_metric: str = "ndcg@5"
    xgb_n_estimators: int = 600
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 5
    xgb_min_child_weight: float = 8.0
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_alpha: float = 0.05
    xgb_reg_lambda: float = 1.2
    xgb_n_jobs: int = 1
    xgb_tree_method: str = "hist"

    lgb_objective: str = "lambdarank"
    lgb_metric: str = "ndcg"
    lgb_n_estimators: int = 800
    lgb_learning_rate: float = 0.03
    lgb_max_depth: int = 6
    lgb_num_leaves: int = 63
    lgb_min_child_samples: int = 20
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.05
    lgb_reg_lambda: float = 1.0
    lgb_n_jobs: int = 1
    lgb_ndcg_eval_at: int = 5

    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 300
    hgb_max_depth: int = 6
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 30
    hgb_l2_regularization: float = 0.1

    @property
    def train_data_path(self) -> Path:
        return self.data_dir / self.train_filename

    @property
    def test_data_path(self) -> Path:
        return self.data_dir / self.test_filename

    @property
    def inference_data_path(self) -> Path:
        return self.data_dir / self.inference_filename

    @property
    def full_data_path(self) -> Path:
        return self.data_dir / self.full_data_filename

    @property
    def result_path(self) -> Path:
        return self.output_dir / self.result_filename

    @property
    def submission_check_path(self) -> Path:
        return self.temp_dir / self.submission_check_filename

    @property
    def self_score_path(self) -> Path:
        return self.temp_dir / self.self_score_filename

    def build_run_dir(self, experiment_name: str | None = None) -> Path:
        return self.model_dir / (experiment_name or self.experiment_name)

    def build_project_config_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.project_config_filename

    def build_training_summary_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.training_summary_filename

    def build_training_report_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.training_report_filename

    def build_model_metadata_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.model_metadata_filename

    def build_feature_importance_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.feature_importance_filename

    def build_backtest_report_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.backtest_report_filename

    def build_portfolio_report_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.portfolio_report_filename

    def build_ranker_model_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.ranker_model_filename

    def build_ranker_model_lgb_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.ranker_model_lgb_filename

    def build_ranker_model_hgb_path(self, experiment_name: str | None = None) -> Path:
        return self.build_run_dir(experiment_name) / self.ranker_model_hgb_filename

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload.items()
        }
