from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ProjectConfig
from backtest import evaluate_parameter_grid, evaluate_portfolio_series
from featurework import (
    build_purged_walk_forward_splits,
    build_training_bundle,
    calculate_top_k_return_metrics,
    combine_model_scores,
    derive_validation_weights,
    load_dataframe,
    safe_zscore,
    standardize_market_dataframe,
    summarize_dataframe,
)
from portfolio import combine_rank_scores, shrink_model_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模型股票排序集成方案。")
    parser.add_argument("--train-data", type=str, default=None, help="训练数据路径，默认使用 data/train.csv")
    parser.add_argument("--experiment-name", type=str, default=None, help="实验目录名称")
    parser.add_argument("--feature-preset", type=str, default=None, help="baseline_v1 / alpha_v1 / cross_v1 / full_v1")
    parser.add_argument("--production-models", type=str, default=None, help="生产推理使用的模型列表，逗号分隔")
    parser.add_argument("--production-portfolio-size", type=int, default=None, help="生产推理输出的持仓数量，1~5")
    parser.add_argument("--label-bucket-count", type=int, default=None, help="标签分桶数，默认使用配置值")
    parser.add_argument("--label-return-clip-quantile", type=float, default=None, help="按交易日裁剪 future_return 的分位数")
    parser.add_argument("--head-sample-weight-quantile", type=float, default=None, help="按交易日给头部样本加权的分位数阈值")
    parser.add_argument("--head-sample-weight-value", type=float, default=None, help="头部样本加权权重值")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_xgboost_module():
    try:
        import xgboost as xgb
    except ImportError as error:
        raise ImportError("未安装 xgboost。") from error
    return xgb


def load_lightgbm_module():
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ImportError("未安装 lightgbm。") from error
    return lgb


def load_hist_gradient_boosting_regressor():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as error:
        raise ImportError("未安装支持 HistGradientBoostingRegressor 的 scikit-learn。") from error
    return HistGradientBoostingRegressor


def augment_features(x_df: pd.DataFrame, noise_std: float, fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x_aug = x_df.copy()
    cols = x_df.columns.tolist()
    mask = rng.random(size=len(cols)) < fraction
    for i, col in enumerate(cols):
        if mask[i]:
            x_aug[col] = x_aug[col] + rng.normal(0, noise_std * (x_df[col].std() + 1e-12), size=len(x_df))
    return x_aug


def clip_continuous_target(values: pd.Series, quantile: float) -> pd.Series:
    lower = float(values.quantile(quantile))
    upper = float(values.quantile(1.0 - quantile))
    return values.clip(lower=lower, upper=upper)


def train_xgb_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    qid_train: pd.Series,
    sample_weight: pd.Series | None,
    config: ProjectConfig,
    seed: int,
):
    xgb = load_xgboost_module()
    model = xgb.XGBRanker(
        objective=config.xgb_objective,
        eval_metric=config.xgb_eval_metric,
        n_estimators=config.xgb_n_estimators,
        learning_rate=config.xgb_learning_rate,
        max_depth=config.xgb_max_depth,
        min_child_weight=config.xgb_min_child_weight,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        reg_alpha=config.xgb_reg_alpha,
        reg_lambda=config.xgb_reg_lambda,
        n_jobs=config.xgb_n_jobs,
        random_state=seed,
        tree_method=config.xgb_tree_method,
    )
    x_train_aug = augment_features(x_train, config.augmentation_noise_std, config.augmentation_noise_fraction, seed)
    fit_kwargs: dict[str, object] = {"qid": qid_train, "verbose": False}
    if sample_weight is not None:
        group_weight = (
            pd.DataFrame({"qid": qid_train, "sample_weight": sample_weight})
            .groupby("qid", sort=False)["sample_weight"]
            .mean()
            .to_numpy(dtype=float)
        )
        fit_kwargs["sample_weight"] = group_weight
    model.fit(x_train_aug, y_train, **fit_kwargs)
    return model


def train_lgb_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    qid_train: pd.Series,
    sample_weight: pd.Series | None,
    config: ProjectConfig,
    seed: int,
):
    lgb = load_lightgbm_module()
    train_dataset = lgb.Dataset(
        x_train,
        y_train,
        group=pd.Series(qid_train).groupby(qid_train, sort=False).size().values,
        weight=None if sample_weight is None else np.asarray(sample_weight, dtype=float),
    )
    model = lgb.train(
        params={
            "objective": config.lgb_objective,
            "metric": config.lgb_metric,
            "num_leaves": config.lgb_num_leaves,
            "max_depth": config.lgb_max_depth,
            "learning_rate": config.lgb_learning_rate,
            "min_data_in_leaf": config.lgb_min_child_samples,
            "feature_fraction": config.lgb_colsample_bytree,
            "bagging_fraction": config.lgb_subsample,
            "lambda_l1": config.lgb_reg_alpha,
            "lambda_l2": config.lgb_reg_lambda,
            "num_threads": config.lgb_n_jobs,
            "ndcg_eval_at": config.lgb_ndcg_eval_at,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set=train_dataset,
        num_boost_round=config.lgb_n_estimators,
    )
    return model


def train_hgb_regressor(x_train: pd.DataFrame, y_train_continuous: pd.Series, config: ProjectConfig, seed: int):
    estimator_cls = load_hist_gradient_boosting_regressor()
    x_train_aug = augment_features(x_train, config.augmentation_noise_std, config.augmentation_noise_fraction, seed)
    model = estimator_cls(
        learning_rate=config.hgb_learning_rate,
        max_iter=config.hgb_max_iter,
        max_depth=config.hgb_max_depth,
        max_leaf_nodes=config.hgb_max_leaf_nodes,
        min_samples_leaf=config.hgb_min_samples_leaf,
        l2_regularization=config.hgb_l2_regularization,
        random_state=seed,
    )
    model.fit(x_train_aug, y_train_continuous)
    return model


def predict_model(model_name: str, model, features: pd.DataFrame) -> np.ndarray:
    if model_name == "xgb_ranker":
        return np.asarray(model.predict(features), dtype=float)
    if model_name == "lgb_ranker":
        return np.asarray(model.predict(features), dtype=float)
    if model_name == "hgb_regressor":
        return np.asarray(model.predict(features), dtype=float)
    raise ValueError(f"不支持的模型名称：{model_name}")


def evaluate_models_on_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    trained_models: dict[str, object],
    ensemble_methods: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    x_eval = frame[feature_columns]
    score_map = {model_name: predict_model(model_name, model, x_eval) for model_name, model in trained_models.items()}

    model_metrics: dict[str, dict[str, float]] = {}
    for model_name, scores in score_map.items():
        model_metrics[model_name] = calculate_top_k_return_metrics(frame.assign(pred_score=scores), k=5)

    validation_weights = derive_validation_weights(model_metrics, "top_k_relative_score")
    ensemble_metrics: dict[str, dict[str, float]] = {}
    for method in ensemble_methods:
        scores = combine_model_scores(score_map, method=method, weights=validation_weights)
        ensemble_metrics[method] = calculate_top_k_return_metrics(frame.assign(pred_score=scores), k=5)

    return model_metrics, ensemble_metrics, score_map


def fit_all_models(train_frame: pd.DataFrame, feature_columns: list[str], config: ProjectConfig) -> dict[str, object]:
    x_train = train_frame[feature_columns]
    y_train = train_frame["relevance_label"]
    qid_train = train_frame["qid"]
    sample_weight = train_frame["sample_weight"] if "sample_weight" in train_frame.columns else None
    continuous_target = clip_continuous_target(
        train_frame["future_return"],
        quantile=config.continuous_target_clip_quantile,
    )
    return {
        "xgb_ranker": train_xgb_ranker(
            x_train,
            y_train,
            qid_train,
            sample_weight,
            config,
            seed=config.random_seed,
        ),
        "lgb_ranker": train_lgb_ranker(
            augment_features(x_train, config.augmentation_noise_std, config.augmentation_noise_fraction, config.random_seed + 1),
            y_train,
            qid_train,
            sample_weight,
            config,
            seed=config.random_seed,
        ),
        "hgb_regressor": train_hgb_regressor(x_train, continuous_target, config, seed=config.random_seed + 2),
    }


def summarize_fold_metrics(metric_records: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not metric_records:
        return {}
    keys = list(metric_records[0].keys())
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        metric_names = metric_records[0][key].keys()
        summary[key] = {
            metric_name: round(float(np.mean([fold[key][metric_name] for fold in metric_records])), 8)
            for metric_name in metric_names
        }
    return summary


def derive_robust_model_scores(
    metric_records: list[dict[str, dict[str, float]]],
) -> dict[str, float]:
    if not metric_records:
        raise ValueError("model metric records are empty")
    model_names = list(metric_records[0])
    robust_scores: dict[str, float] = {}
    for model_name in model_names:
        relative_values = np.asarray(
            [fold[model_name]["top_k_relative_score"] for fold in metric_records],
            dtype=float,
        )
        return_values = np.asarray(
            [fold[model_name]["pred_top_k_return_mean"] for fold in metric_records],
            dtype=float,
        )
        finite_relative = relative_values[np.isfinite(relative_values)]
        finite_returns = return_values[np.isfinite(return_values)]
        if finite_relative.size == 0 or finite_returns.size == 0:
            robust_scores[model_name] = 0.0
            continue
        score = float(np.median(finite_relative))
        if float(np.median(finite_returns)) <= 0.0:
            score = min(score, 0.0)
        robust_scores[model_name] = score
    return robust_scores


def build_rank_scored_frame(
    frame: pd.DataFrame,
    score_map: dict[str, np.ndarray],
    model_weights: dict[str, float],
) -> pd.DataFrame:
    selected_scores = {name: score_map[name] for name in model_weights}
    combined = combine_rank_scores(
        selected_scores,
        model_weights,
        frame["stock_id"],
    )
    result = frame.copy()
    result["pred_score"] = combined
    return result


def frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    serializable = frame.copy()
    if "trade_date" in serializable.columns:
        serializable["trade_date"] = pd.to_datetime(serializable["trade_date"]).dt.strftime("%Y-%m-%d")
    return serializable.to_dict(orient="records")


def build_production_metadata(
    *,
    feature_columns: list[str],
    feature_windows: tuple[int, ...],
    model_weights: dict[str, float],
    variance_penalty: float,
    correlation_penalty: float,
    config: ProjectConfig | None = None,
) -> dict[str, object]:
    resolved = config or ProjectConfig()
    return {
        "model_type": "xgboost_lightgbm_hgb_robust_rank_ensemble",
        "selected_models": list(model_weights),
        "ensemble_method": "shrunk_rank_average",
        "ensemble_weights": {name: float(weight) for name, weight in model_weights.items()},
        "feature_columns": feature_columns,
        "feature_windows": list(feature_windows),
        "portfolio": {
            "size": 5,
            "candidate_pool_size": int(resolved.portfolio_candidate_pool_size),
            "covariance_window": int(resolved.portfolio_covariance_window),
            "variance_penalty": float(variance_penalty),
            "correlation_penalty": float(correlation_penalty),
            "weight": float(resolved.portfolio_weight),
        },
    }


def select_final_risk_configuration(
    outer_leaderboards: list[list[dict[str, object]]],
) -> dict[str, float]:
    if not outer_leaderboards:
        raise ValueError("outer risk leaderboards are empty")
    rank_map: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    robust_map: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    for leaderboard in outer_leaderboards:
        for rank, row in enumerate(leaderboard, start=1):
            key = (
                float(row["variance_penalty"]),
                float(row["correlation_penalty"]),
                float(row["cvar_penalty"]),
            )
            rank_map[key].append(rank)
            robust_map[key].append(float(row["robust_score"]))
    complete_keys = [key for key, ranks in rank_map.items() if len(ranks) == len(outer_leaderboards)]
    if not complete_keys:
        raise ValueError("no risk configuration appears in every outer fold")
    best_key = min(
        complete_keys,
        key=lambda key: (
            float(np.median(rank_map[key])),
            -float(np.mean(robust_map[key])),
            sum(key),
            key,
        ),
    )
    return {
        "variance_penalty": best_key[0],
        "correlation_penalty": best_key[1],
        "cvar_penalty": best_key[2],
        "median_outer_rank": float(np.median(rank_map[best_key])),
        "mean_outer_robust_score": float(np.mean(robust_map[best_key])),
    }


def run_inner_selection(
    outer_train_frame: pd.DataFrame,
    feature_columns: list[str],
    market_frame: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    inner_splits = build_purged_walk_forward_splits(
        outer_train_frame,
        fold_count=config.inner_walk_forward_fold_count,
        validation_ratio=config.validation_ratio,
        min_train_groups=config.inner_min_train_groups,
        purge_groups=config.purge_group_count,
    )
    inner_metric_records: list[dict[str, dict[str, float]]] = []
    inner_outputs: list[tuple[pd.DataFrame, dict[str, np.ndarray]]] = []
    for inner_index, (inner_train, inner_validation) in enumerate(inner_splits, start=1):
        print(
            f"[训练]   Inner {inner_index}/{len(inner_splits)}："
            f"train_dates={inner_train['trade_date'].nunique()} valid_dates={inner_validation['trade_date'].nunique()}"
        )
        models = fit_all_models(inner_train, feature_columns, config)
        metrics, _, score_map = evaluate_models_on_frame(
            inner_validation,
            feature_columns,
            models,
            ["rank_average_equal"],
        )
        inner_metric_records.append(metrics)
        inner_outputs.append((inner_validation, score_map))

    robust_scores = derive_robust_model_scores(inner_metric_records)
    model_weights = shrink_model_weights(
        robust_scores,
        shrinkage=config.model_weight_shrinkage,
        cap=config.model_weight_cap,
    )
    inner_scored = pd.concat(
        [build_rank_scored_frame(frame, score_map, model_weights) for frame, score_map in inner_outputs],
        ignore_index=True,
    )
    selected_risk, leaderboard, details_by_key = evaluate_parameter_grid(
        inner_scored,
        market_frame,
        candidate_pool_size=config.portfolio_candidate_pool_size,
        covariance_window=config.portfolio_covariance_window,
        variance_penalties=config.variance_penalty_grid,
        correlation_penalties=config.correlation_penalty_grid,
        cvar_penalties=config.cvar_penalty_grid,
        rebalance_stride=config.rebalance_stride,
    )
    details_key = (
        f"variance={selected_risk['variance_penalty']}|"
        f"correlation={selected_risk['correlation_penalty']}"
    )
    report = {
        "split_count": len(inner_splits),
        "robust_model_scores": robust_scores,
        "model_weights": model_weights,
        "selected_risk": selected_risk,
        "leaderboard": leaderboard,
        "selected_weekly_details": frame_records(details_by_key[details_key]),
    }
    return model_weights, selected_risk, report


def main() -> None:
    args = parse_args()
    base_config = ProjectConfig()
    production_portfolio_size = (
        args.production_portfolio_size
        if args.production_portfolio_size is not None
        else base_config.portfolio_size
    )
    if production_portfolio_size < 1 or production_portfolio_size > base_config.max_portfolio_size:
        raise ValueError(
            f"production_portfolio_size 必须位于 1 到 {base_config.max_portfolio_size} 之间，当前为 {production_portfolio_size}"
        )
    config = ProjectConfig(
        feature_preset=args.feature_preset if args.feature_preset is not None else base_config.feature_preset,
        portfolio_size=production_portfolio_size,
        label_bucket_count=args.label_bucket_count if args.label_bucket_count is not None else base_config.label_bucket_count,
        label_return_clip_quantile=(
            args.label_return_clip_quantile
            if args.label_return_clip_quantile is not None
            else base_config.label_return_clip_quantile
        ),
        head_sample_weight_quantile=(
            args.head_sample_weight_quantile
            if args.head_sample_weight_quantile is not None
            else base_config.head_sample_weight_quantile
        ),
        head_sample_weight_value=(
            args.head_sample_weight_value
            if args.head_sample_weight_value is not None
            else base_config.head_sample_weight_value
        ),
    )
    set_seed(config.random_seed)

    experiment_name = args.experiment_name or config.experiment_name
    train_data_path = Path(args.train_data) if args.train_data else config.train_data_path
    run_dir = config.build_run_dir(experiment_name)
    ranker_model_xgb_path = config.build_ranker_model_path(experiment_name)
    ranker_model_lgb_path = config.build_ranker_model_lgb_path(experiment_name)
    ranker_model_hgb_path = config.build_ranker_model_hgb_path(experiment_name)
    model_metadata_path = config.build_model_metadata_path(experiment_name)
    feature_importance_xgb_path = config.build_feature_importance_path(experiment_name)
    feature_importance_lgb_path = Path(str(feature_importance_xgb_path).replace(".csv", "_lgb.csv"))
    training_summary_path = config.build_training_summary_path(experiment_name)
    training_report_path = config.build_training_report_path(experiment_name)
    project_config_path = config.build_project_config_path(experiment_name)

    print("[训练] 开始执行多模型股票排序训练。")
    print(f"[训练] 训练数据：{train_data_path}")
    print(f"[训练] 实验目录：{run_dir}")

    raw_dataframe = load_dataframe(train_data_path)
    raw_summary = summarize_dataframe(raw_dataframe)
    bundle = build_training_bundle(
        raw_dataframe,
        windows=config.feature_windows,
        feature_preset=config.feature_preset,
        future_buy_offset=config.future_buy_offset,
        future_sell_offset=config.future_sell_offset,
        validation_ratio=config.validation_ratio,
        min_train_groups=config.min_train_groups,
        label_bucket_count=config.label_bucket_count,
        label_clip_quantile=config.label_return_clip_quantile,
        head_weight_quantile=config.head_sample_weight_quantile,
        head_weight_value=config.head_sample_weight_value,
    )
    ranking_frame = bundle.ranking_frame.copy()
    feature_columns = bundle.feature_columns
    market_frame = standardize_market_dataframe(raw_dataframe)
    walk_forward_splits = build_purged_walk_forward_splits(
        ranking_frame,
        fold_count=config.outer_walk_forward_fold_count,
        validation_ratio=config.validation_ratio,
        min_train_groups=config.min_train_groups,
        purge_groups=config.purge_group_count,
    )

    fold_model_metrics: list[dict[str, dict[str, float]]] = []
    fold_reports: list[dict[str, object]] = []
    outer_leaderboards: list[list[dict[str, object]]] = []

    for fold_index, (train_frame, validation_frame) in enumerate(walk_forward_splits, start=1):
        print(
            f"[训练] Outer {fold_index}/{len(walk_forward_splits)}："
            f" train_dates={train_frame['trade_date'].nunique()} valid_dates={validation_frame['trade_date'].nunique()}"
        )
        inner_weights, inner_risk, inner_report = run_inner_selection(
            train_frame,
            feature_columns,
            market_frame,
            config,
        )
        trained_models = fit_all_models(train_frame, feature_columns, config)
        model_metrics, _, score_map = evaluate_models_on_frame(
            validation_frame,
            feature_columns,
            trained_models,
            ["rank_average_equal"],
        )
        fold_model_metrics.append(model_metrics)
        robust_scored = build_rank_scored_frame(validation_frame, score_map, inner_weights)
        robust_details, robust_statistics = evaluate_portfolio_series(
            robust_scored,
            market_frame,
            candidate_pool_size=config.portfolio_candidate_pool_size,
            covariance_window=config.portfolio_covariance_window,
            variance_penalty=float(inner_risk["variance_penalty"]),
            correlation_penalty=float(inner_risk["correlation_penalty"]),
            rebalance_stride=config.rebalance_stride,
        )
        _, outer_leaderboard, _ = evaluate_parameter_grid(
            robust_scored,
            market_frame,
            candidate_pool_size=config.portfolio_candidate_pool_size,
            covariance_window=config.portfolio_covariance_window,
            variance_penalties=config.variance_penalty_grid,
            correlation_penalties=config.correlation_penalty_grid,
            cvar_penalties=config.cvar_penalty_grid,
            rebalance_stride=config.rebalance_stride,
        )
        outer_leaderboards.append(outer_leaderboard)

        xgb_scores = score_map["xgb_ranker"]
        current_overlay_scores = safe_zscore(xgb_scores)
        if "volume_ratio_20" in validation_frame.columns:
            current_overlay_scores = current_overlay_scores + 0.7 * safe_zscore(
                validation_frame["volume_ratio_20"].to_numpy(dtype=float)
            )
        current_top1_metrics = calculate_top_k_return_metrics(
            validation_frame.assign(pred_score=current_overlay_scores),
            k=1,
            rebalance_stride=config.rebalance_stride,
        )
        xgb_top5_metrics = calculate_top_k_return_metrics(
            validation_frame.assign(pred_score=xgb_scores),
            k=5,
            rebalance_stride=config.rebalance_stride,
        )
        equal_model_weights = {name: 1.0 / len(score_map) for name in score_map}
        equal_rank_scored = build_rank_scored_frame(validation_frame, score_map, equal_model_weights)
        equal_rank_top5_metrics = calculate_top_k_return_metrics(
            equal_rank_scored,
            k=5,
            rebalance_stride=config.rebalance_stride,
        )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "train_group_count": int(train_frame["trade_date"].nunique()),
                "validation_group_count": int(validation_frame["trade_date"].nunique()),
                "purge_group_count": int(config.purge_group_count),
                "inner_selection": inner_report,
                "model_metrics": model_metrics,
                "strategy_metrics": {
                    "current_overlay_top1": current_top1_metrics,
                    "xgb_top5_equal_weight": xgb_top5_metrics,
                    "equal_rank_ensemble_top5": equal_rank_top5_metrics,
                    "robust_risk_controlled_top5": robust_statistics,
                },
                "robust_weekly_details": frame_records(robust_details),
                "outer_parameter_leaderboard": outer_leaderboard,
            }
        )

    aggregated_model_metrics = summarize_fold_metrics(fold_model_metrics)
    final_robust_model_scores = derive_robust_model_scores(fold_model_metrics)
    final_model_weights = shrink_model_weights(
        final_robust_model_scores,
        shrinkage=config.model_weight_shrinkage,
        cap=config.model_weight_cap,
    )
    final_risk = select_final_risk_configuration(outer_leaderboards)
    print(f"[训练] 稳健模型权重：{final_model_weights}")
    print(f"[训练] 稳健组合参数：{final_risk}")

    if args.production_models:
        production_selected_models = [item.strip() for item in args.production_models.split(",") if item.strip()]
    else:
        production_selected_models = list(base_config.production_model_names)
    required_models = {"xgb_ranker", "lgb_ranker", "hgb_regressor"}
    if set(production_selected_models) != required_models:
        raise ValueError("稳健生产路径必须同时使用 xgb_ranker、lgb_ranker 和 hgb_regressor。")
    production_selected_models = ["xgb_ranker", "lgb_ranker", "hgb_regressor"]
    production_ensemble_method = "shrunk_rank_average"
    production_ensemble_weights = {name: final_model_weights[name] for name in production_selected_models}

    production_models = fit_all_models(ranking_frame.copy(), feature_columns, config)

    run_dir.mkdir(parents=True, exist_ok=True)
    production_models["xgb_ranker"].save_model(str(ranker_model_xgb_path))
    production_models["lgb_ranker"].save_model(str(ranker_model_lgb_path))
    joblib.dump(production_models["hgb_regressor"], ranker_model_hgb_path)

    imp_xgb = pd.DataFrame({"feature": feature_columns, "importance": production_models["xgb_ranker"].feature_importances_})
    imp_xgb = imp_xgb.sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort")
    imp_xgb.to_csv(feature_importance_xgb_path, index=False, encoding="utf-8")

    imp_lgb = pd.DataFrame({"feature": feature_columns, "importance": production_models["lgb_ranker"].feature_importance()})
    imp_lgb = imp_lgb.sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort")
    imp_lgb.to_csv(feature_importance_lgb_path, index=False, encoding="utf-8")

    config_payload = config.to_dict()
    config_payload["train_data_path"] = str(train_data_path)
    config_payload["experiment_name"] = experiment_name
    write_json(project_config_path, config_payload)

    training_summary_payload = {
        "message": "稳健 Top5 多模型股票排序训练完成。",
        "raw_data_summary": raw_summary,
        "ranking_rows": int(len(ranking_frame)),
        "production_train_rows": int(len(ranking_frame)),
        "production_train_groups": int(ranking_frame["trade_date"].nunique()),
        "walk_forward_fold_count": int(len(walk_forward_splits)),
        "purge_group_count": int(config.purge_group_count),
        "rebalance_stride": int(config.rebalance_stride),
        "label_bucket_count": int(config.label_bucket_count),
        "label_return_clip_quantile": float(config.label_return_clip_quantile),
        "head_sample_weight_quantile": float(config.head_sample_weight_quantile),
        "head_sample_weight_value": float(config.head_sample_weight_value),
        "feature_count": int(len(feature_columns)),
        "feature_preset": bundle.feature_preset,
        "feature_group_sizes": {group_name: len(columns) for group_name, columns in bundle.feature_groups.items()},
        "production_selected_models": production_selected_models,
        "production_ensemble_weights": production_ensemble_weights,
        "production_risk_configuration": final_risk,
        "portfolio_size": int(config.portfolio_size),
        "feature_columns": feature_columns,
    }
    write_json(training_summary_path, training_summary_payload)

    training_report_payload = {
        "purged_outer_folds": fold_reports,
        "aggregated_model_metrics": aggregated_model_metrics,
        "robust_model_scores": final_robust_model_scores,
        "production_selected_models": production_selected_models,
        "production_ensemble_method": production_ensemble_method,
        "production_ensemble_weights": production_ensemble_weights,
        "production_risk_configuration": final_risk,
        "top_features_xgb": imp_xgb.head(20).to_dict(orient="records"),
        "top_features_lgb": imp_lgb.head(20).to_dict(orient="records"),
    }
    write_json(training_report_path, training_report_payload)
    write_json(config.build_backtest_report_path(experiment_name), training_report_payload)

    metadata_payload = build_production_metadata(
        feature_columns=feature_columns,
        feature_windows=config.feature_windows,
        model_weights=production_ensemble_weights,
        variance_penalty=float(final_risk["variance_penalty"]),
        correlation_penalty=float(final_risk["correlation_penalty"]),
        config=config,
    )
    metadata_payload.update({
        "experiment_name": experiment_name,
        "feature_preset": bundle.feature_preset,
        "feature_group_sizes": {group_name: len(columns) for group_name, columns in bundle.feature_groups.items()},
        "future_buy_offset": config.future_buy_offset,
        "future_sell_offset": config.future_sell_offset,
        "label_bucket_count": config.label_bucket_count,
        "label_return_clip_quantile": config.label_return_clip_quantile,
        "head_sample_weight_quantile": config.head_sample_weight_quantile,
        "head_sample_weight_value": config.head_sample_weight_value,
        "walk_forward_fold_count": len(walk_forward_splits),
        "purge_group_count": config.purge_group_count,
        "rebalance_stride": config.rebalance_stride,
        "random_seed": config.random_seed,
        "result_columns": list(config.result_columns),
        "max_portfolio_size": config.max_portfolio_size,
        "portfolio_size": config.portfolio_size,
        "training_data": {
            "path": str(train_data_path),
            "sha256": file_sha256(train_data_path),
            "max_trade_date": pd.to_datetime(market_frame["trade_date"]).max().strftime("%Y-%m-%d"),
            "row_count": int(len(market_frame)),
            "stock_count": int(market_frame["stock_id"].nunique()),
        },
        "xgb_params": {
            "objective": config.xgb_objective,
            "n_estimators": config.xgb_n_estimators,
            "learning_rate": config.xgb_learning_rate,
            "max_depth": config.xgb_max_depth,
        },
        "lgb_params": {
            "objective": config.lgb_objective,
            "n_estimators": config.lgb_n_estimators,
            "learning_rate": config.lgb_learning_rate,
            "max_depth": config.lgb_max_depth,
            "num_leaves": config.lgb_num_leaves,
        },
        "hgb_params": {
            "learning_rate": config.hgb_learning_rate,
            "max_iter": config.hgb_max_iter,
            "max_depth": config.hgb_max_depth,
            "max_leaf_nodes": config.hgb_max_leaf_nodes,
            "min_samples_leaf": config.hgb_min_samples_leaf,
            "l2_regularization": config.hgb_l2_regularization,
        },
    })
    write_json(model_metadata_path, metadata_payload)

    print("[训练] 多模型训练完成。")
    print(f"[训练] 特征数量：{len(feature_columns)}")
    print(f"[训练] 生产集成：{production_ensemble_method}")
    print(f"[训练] 生产模型权重：{production_ensemble_weights}")
    print(f"[训练] 生产组合参数：{final_risk}")
    print(f"[训练] XGBoost 模型：{ranker_model_xgb_path}")
    print(f"[训练] LightGBM 模型：{ranker_model_lgb_path}")
    print(f"[训练] HGB 模型：{ranker_model_hgb_path}")


if __name__ == "__main__":
    main()
