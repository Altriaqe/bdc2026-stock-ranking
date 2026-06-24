from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ProjectConfig
from featurework import (
    build_training_bundle,
    build_walk_forward_splits,
    calculate_top_k_return_metrics,
    combine_model_scores,
    derive_validation_weights,
    load_dataframe,
    summarize_dataframe,
)


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    walk_forward_splits = build_walk_forward_splits(
        ranking_frame,
        fold_count=config.walk_forward_fold_count,
        validation_ratio=config.validation_ratio,
        min_train_groups=config.min_train_groups,
    )

    ensemble_methods = [
        "zscore_average_equal",
        "rank_average_equal",
        "validation_weighted_zscore_average",
    ]
    fold_model_metrics: list[dict[str, dict[str, float]]] = []
    fold_ensemble_metrics: list[dict[str, dict[str, float]]] = []
    fold_reports: list[dict[str, object]] = []

    for fold_index, (train_frame, validation_frame) in enumerate(walk_forward_splits, start=1):
        print(
            f"[训练] Fold {fold_index}/{len(walk_forward_splits)}："
            f" train_dates={train_frame['trade_date'].nunique()} valid_dates={validation_frame['trade_date'].nunique()}"
        )
        trained_models = fit_all_models(train_frame, feature_columns, config)
        model_metrics, ensemble_metrics, _ = evaluate_models_on_frame(
            validation_frame,
            feature_columns,
            trained_models,
            ensemble_methods,
        )
        fold_model_metrics.append(model_metrics)
        fold_ensemble_metrics.append(ensemble_metrics)
        fold_reports.append(
            {
                "fold_index": fold_index,
                "train_group_count": int(train_frame["trade_date"].nunique()),
                "validation_group_count": int(validation_frame["trade_date"].nunique()),
                "model_metrics": model_metrics,
                "ensemble_metrics": ensemble_metrics,
            }
        )

    aggregated_model_metrics = summarize_fold_metrics(fold_model_metrics)
    aggregated_ensemble_metrics = summarize_fold_metrics(fold_ensemble_metrics)
    validation_weights = derive_validation_weights(aggregated_model_metrics, "top_k_relative_score")

    diagnostic_best_ensemble_method = max(
        aggregated_ensemble_metrics.items(),
        key=lambda item: item[1].get("top_k_relative_score", float("-inf")),
    )[0]
    print(f"[训练] 诊断最优集成方式：{diagnostic_best_ensemble_method}")
    print(f"[训练] 诊断模型权重：{validation_weights}")

    if args.production_models:
        production_selected_models = [item.strip() for item in args.production_models.split(",") if item.strip()]
    else:
        production_selected_models = list(base_config.production_model_names)
    if not production_selected_models:
        raise ValueError("production_models 不能为空。")
    unsupported_models = [name for name in production_selected_models if name not in {"xgb_ranker", "lgb_ranker", "hgb_regressor"}]
    if unsupported_models:
        raise ValueError(f"存在不支持的 production_models：{unsupported_models}")
    production_ensemble_method = "zscore_average_equal"
    equal_weight = 1.0 / len(production_selected_models)
    production_ensemble_weights = {model_name: equal_weight for model_name in production_selected_models}

    production_models = fit_all_models(bundle.train_frame.copy(), feature_columns, config)
    validation_model_metrics, validation_ensemble_metrics, validation_score_map = evaluate_models_on_frame(
        bundle.validation_frame,
        feature_columns,
        production_models,
        ensemble_methods,
    )
    production_ensemble_scores = combine_model_scores(
        {name: validation_score_map[name] for name in production_selected_models},
        method=production_ensemble_method,
        weights=production_ensemble_weights,
    )
    production_ensemble_metrics = calculate_top_k_return_metrics(
        bundle.validation_frame.assign(pred_score=production_ensemble_scores),
        k=config.top_k_metric,
    )

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
        "message": "多模型股票排序集成训练完成。",
        "raw_data_summary": raw_summary,
        "ranking_rows": int(len(ranking_frame)),
        "train_rows": int(len(bundle.train_frame)),
        "validation_rows": int(len(bundle.validation_frame)),
        "train_groups": int(bundle.train_frame["trade_date"].nunique()),
        "validation_groups": int(bundle.validation_frame["trade_date"].nunique()),
        "walk_forward_fold_count": int(len(walk_forward_splits)),
        "label_bucket_count": int(config.label_bucket_count),
        "label_return_clip_quantile": float(config.label_return_clip_quantile),
        "head_sample_weight_quantile": float(config.head_sample_weight_quantile),
        "head_sample_weight_value": float(config.head_sample_weight_value),
        "feature_count": int(len(feature_columns)),
        "feature_preset": bundle.feature_preset,
        "feature_group_sizes": {group_name: len(columns) for group_name, columns in bundle.feature_groups.items()},
        "production_selected_models": production_selected_models,
        "portfolio_size": int(config.portfolio_size),
        "feature_columns": feature_columns,
    }
    write_json(training_summary_path, training_summary_payload)

    training_report_payload = {
        "walk_forward_folds": fold_reports,
        "aggregated_model_metrics": aggregated_model_metrics,
        "aggregated_ensemble_metrics": aggregated_ensemble_metrics,
        "holdout_model_metrics": validation_model_metrics,
        "holdout_ensemble_metrics": validation_ensemble_metrics,
        "diagnostic_best_ensemble_method": diagnostic_best_ensemble_method,
        "diagnostic_model_weights": validation_weights,
        "production_selected_models": production_selected_models,
        "production_ensemble_method": production_ensemble_method,
        "production_ensemble_weights": production_ensemble_weights,
        "production_holdout_metrics": production_ensemble_metrics,
        "top_features_xgb": imp_xgb.head(20).to_dict(orient="records"),
        "top_features_lgb": imp_lgb.head(20).to_dict(orient="records"),
    }
    write_json(training_report_path, training_report_payload)

    metadata_payload = {
        "model_type": "xgboost_lightgbm_hgb_ensemble",
        "selected_models": production_selected_models,
        "ensemble_method": production_ensemble_method,
        "ensemble_weights": production_ensemble_weights,
        "diagnostic_selected_models": ["xgb_ranker", "lgb_ranker", "hgb_regressor"],
        "diagnostic_ensemble_method": diagnostic_best_ensemble_method,
        "diagnostic_ensemble_weights": validation_weights,
        "experiment_name": experiment_name,
        "feature_columns": feature_columns,
        "feature_preset": bundle.feature_preset,
        "feature_group_sizes": {group_name: len(columns) for group_name, columns in bundle.feature_groups.items()},
        "feature_windows": list(config.feature_windows),
        "future_buy_offset": config.future_buy_offset,
        "future_sell_offset": config.future_sell_offset,
        "label_bucket_count": config.label_bucket_count,
        "label_return_clip_quantile": config.label_return_clip_quantile,
        "head_sample_weight_quantile": config.head_sample_weight_quantile,
        "head_sample_weight_value": config.head_sample_weight_value,
        "walk_forward_fold_count": config.walk_forward_fold_count,
        "result_columns": list(config.result_columns),
        "max_portfolio_size": config.max_portfolio_size,
        "portfolio_size": config.portfolio_size,
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
    }
    write_json(model_metadata_path, metadata_payload)

    print("[训练] 多模型训练完成。")
    print(f"[训练] 特征数量：{len(feature_columns)}")
    print(f"[训练] 生产集成：{production_ensemble_method}")
    print(f"[训练] Holdout 生产得分：{production_ensemble_metrics['top_k_relative_score']:.6f}")
    print(f"[训练] XGBoost 模型：{ranker_model_xgb_path}")
    print(f"[训练] LightGBM 模型：{ranker_model_lgb_path}")
    print(f"[训练] HGB 模型：{ranker_model_hgb_path}")


if __name__ == "__main__":
    main()
