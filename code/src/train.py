from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from config import ProjectConfig
from featurework import (
    build_training_bundle,
    calculate_top_k_return_metrics,
    load_dataframe,
    summarize_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 XGBoost + LightGBM 双模型集成。")
    parser.add_argument("--train-data", type=str, default=None, help="训练数据路径，默认使用 data/train.csv")
    parser.add_argument("--experiment-name", type=str, default=None, help="实验目录名称")
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


def augment_features(x_df: pd.DataFrame, noise_std: float, fraction: float, seed: int) -> pd.DataFrame:
    """对部分特征加轻微高斯噪声，增加训练样本多样性。"""
    rng = np.random.RandomState(seed)
    x_aug = x_df.copy()
    cols = x_df.columns.tolist()
    mask = rng.random(size=len(cols)) < fraction
    for i, col in enumerate(cols):
        if mask[i]:
            x_aug[col] = x_aug[col] + rng.normal(0, noise_std * (x_df[col].std() + 1e-12), size=len(x_df))
    return x_aug


def validate_top_k_ensemble(
    frame: pd.DataFrame, feature_columns: list[str], xgb_ranker, lgb_ranker, k: int
) -> dict[str, float]:
    x_valid = frame[feature_columns]
    xgb_scores = xgb_ranker.predict(x_valid)
    lgb_scores = lgb_ranker.predict(x_valid)

    xgb_z = zscore(xgb_scores)
    lgb_z = zscore(lgb_scores)
    ensemble_scores = (xgb_z + lgb_z) / 2.0

    scored = frame.copy()
    scored["pred_score"] = ensemble_scores
    return calculate_top_k_return_metrics(scored, k=k)


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    xgb = load_xgboost_module()
    lgb = load_lightgbm_module()
    set_seed(config.random_seed)

    experiment_name = args.experiment_name or config.experiment_name
    train_data_path = Path(args.train_data) if args.train_data else config.train_data_path
    run_dir = config.build_run_dir(experiment_name)
    ranker_model_xgb_path = config.build_ranker_model_path(experiment_name)
    ranker_model_lgb_path = config.build_ranker_model_lgb_path(experiment_name)
    model_metadata_path = config.build_model_metadata_path(experiment_name)
    feature_importance_xgb_path = config.build_feature_importance_path(experiment_name)
    feature_importance_lgb_path = Path(str(feature_importance_xgb_path).replace(".csv", "_lgb.csv"))
    training_summary_path = config.build_training_summary_path(experiment_name)
    training_report_path = config.build_training_report_path(experiment_name)
    project_config_path = config.build_project_config_path(experiment_name)

    print("[训练] 开始执行 XGBoost + LightGBM 双模型训练。")
    print(f"[训练] 训练数据：{train_data_path}")
    print(f"[训练] 实验目录：{run_dir}")

    raw_dataframe = load_dataframe(train_data_path)
    raw_summary = summarize_dataframe(raw_dataframe)

    bundle = build_training_bundle(
        raw_dataframe,
        windows=config.feature_windows,
        future_buy_offset=config.future_buy_offset,
        future_sell_offset=config.future_sell_offset,
        validation_ratio=config.validation_ratio,
        min_train_groups=config.min_train_groups,
        label_bucket_count=config.label_bucket_count,
    )

    train_frame = bundle.train_frame.copy()
    validation_frame = bundle.validation_frame.copy()
    feature_columns = bundle.feature_columns

    x_train = train_frame[feature_columns]
    y_train = train_frame["relevance_label"]
    qid_train = train_frame["qid"]

    x_validation = validation_frame[feature_columns]
    y_validation = validation_frame["relevance_label"]
    qid_validation = validation_frame["qid"]

    # ----- XGBoost -----
    print("[训练] 正在训练 XGBoost Ranker...")
    x_train_aug = augment_features(
        x_train, config.augmentation_noise_std, config.augmentation_noise_fraction, config.random_seed,
    )
    xgb_ranker = xgb.XGBRanker(
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
        random_state=config.random_seed,
        tree_method=config.xgb_tree_method,
    )
    xgb_ranker.fit(
        x_train_aug, y_train,
        qid=qid_train,
        eval_set=[(x_validation, y_validation)],
        eval_qid=[qid_validation],
        verbose=False,
    )

    validation_frame["xgb_score"] = xgb_ranker.predict(x_validation)
    xgb_metrics = calculate_top_k_return_metrics(
        validation_frame.assign(pred_score=validation_frame["xgb_score"]),
        k=config.top_k_metric,
    )
    print(f"[训练] XGBoost 验证集 Top-{config.top_k_metric} 相对得分：{xgb_metrics['top_k_relative_score']:.6f}")

    # ----- LightGBM -----
    print("[训练] 正在训练 LightGBM Ranker...")
    x_train_aug_lgb = augment_features(
        x_train, config.augmentation_noise_std, config.augmentation_noise_fraction, config.random_seed + 1,
    )
    lgb_train = lgb.Dataset(
        x_train_aug_lgb, y_train,
        group=train_frame.groupby("qid", sort=False).size().values,
    )
    lgb_valid = lgb.Dataset(
        x_validation, y_validation,
        group=validation_frame.groupby("qid", sort=False).size().values,
        reference=lgb_train,
    )

    lgb_ranker = lgb.train(
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
            "seed": config.random_seed,
            "feature_fraction_seed": config.random_seed,
            "bagging_seed": config.random_seed,
            "data_random_seed": config.random_seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set=lgb_train,
        num_boost_round=config.lgb_n_estimators,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
    )

    validation_frame["lgb_score"] = lgb_ranker.predict(x_validation)
    lgb_metrics = calculate_top_k_return_metrics(
        validation_frame.assign(pred_score=validation_frame["lgb_score"]),
        k=config.top_k_metric,
    )
    print(f"[训练] LightGBM 验证集 Top-{config.top_k_metric} 相对得分：{lgb_metrics['top_k_relative_score']:.6f}")

    # ----- Ensemble -----
    ensemble_metrics = validate_top_k_ensemble(
        validation_frame, feature_columns, xgb_ranker, lgb_ranker, k=config.top_k_metric,
    )
    print(f"[训练] 集成模型验证集 Top-{config.top_k_metric} 相对得分：{ensemble_metrics['top_k_relative_score']:.6f}")

    # ----- Save -----
    run_dir.mkdir(parents=True, exist_ok=True)
    xgb_ranker.save_model(str(ranker_model_xgb_path))
    lgb_ranker.save_model(str(ranker_model_lgb_path))

    imp_xgb = pd.DataFrame({"feature": feature_columns, "importance": xgb_ranker.feature_importances_})
    imp_xgb = imp_xgb.sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort")
    imp_xgb.to_csv(feature_importance_xgb_path, index=False, encoding="utf-8")

    imp_lgb = pd.DataFrame({"feature": feature_columns, "importance": lgb_ranker.feature_importance()})
    imp_lgb = imp_lgb.sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort")
    imp_lgb.to_csv(feature_importance_lgb_path, index=False, encoding="utf-8")

    config_payload = config.to_dict()
    config_payload["train_data_path"] = str(train_data_path)
    config_payload["experiment_name"] = experiment_name
    write_json(project_config_path, config_payload)

    training_summary_payload = {
        "message": "XGBoost + LightGBM 双模型集成训练。",
        "raw_data_summary": raw_summary,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "train_groups": int(train_frame["trade_date"].nunique()),
        "validation_groups": int(validation_frame["trade_date"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    write_json(training_summary_path, training_summary_payload)

    training_report_payload = {
        "xgb_metrics": xgb_metrics,
        "lgb_metrics": lgb_metrics,
        "ensemble_metrics": ensemble_metrics,
        "top_features_xgb": imp_xgb.head(20).to_dict(orient="records"),
        "top_features_lgb": imp_lgb.head(20).to_dict(orient="records"),
    }
    write_json(training_report_path, training_report_payload)

    metadata_payload = {
        "model_type": "xgboost_lightgbm_ensemble",
        "ensemble_method": "zscore_average",
        "experiment_name": experiment_name,
        "feature_columns": feature_columns,
        "feature_windows": list(config.feature_windows),
        "future_buy_offset": config.future_buy_offset,
        "future_sell_offset": config.future_sell_offset,
        "label_bucket_count": config.label_bucket_count,
        "result_columns": list(config.result_columns),
        "max_portfolio_size": config.max_portfolio_size,
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
    }
    write_json(model_metadata_path, metadata_payload)

    print("[训练] 双模型训练完成。")
    print(f"[训练] 特征数量：{len(feature_columns)}")
    print(f"[训练] XGBoost  得分：{xgb_metrics['top_k_relative_score']:.6f}")
    print(f"[训练] LightGBM 得分：{lgb_metrics['top_k_relative_score']:.6f}")
    print(f"[训练] 集成     得分：{ensemble_metrics['top_k_relative_score']:.6f}")
    print(f"[训练] XGBoost 模型：{ranker_model_xgb_path}")
    print(f"[训练] LightGBM 模型：{ranker_model_lgb_path}")


if __name__ == "__main__":
    main()
