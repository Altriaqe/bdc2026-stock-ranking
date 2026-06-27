from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from compliance import SubmissionCheckResult, validate_submission_frame, write_submission_report
from config import ProjectConfig
from featurework import (
    combine_model_scores,
    create_submission_from_scores,
    load_dataframe,
    prepare_inference_frame,
    safe_zscore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission result file.")
    parser.add_argument("--inference-data", type=str, default=None, help="Inference input path, default data/train.csv")
    parser.add_argument("--test-data", type=str, default=None, help="Legacy alias for --inference-data")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment directory name")
    return parser.parse_args()


def load_xgboost_module():
    import xgboost as xgb

    return xgb


def load_lightgbm_module():
    import lightgbm as lgb

    return lgb


def load_selected_models(config: ProjectConfig, experiment_name: str, selected_models: list[str]) -> dict[str, object]:
    models: dict[str, object] = {}
    if "xgb_ranker" in selected_models:
        xgb = load_xgboost_module()
        ranker_xgb_path = config.build_ranker_model_path(experiment_name)
        if not ranker_xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {ranker_xgb_path}")
        xgb_ranker = xgb.XGBRanker()
        xgb_ranker.load_model(str(ranker_xgb_path))
        models["xgb_ranker"] = xgb_ranker

    if "lgb_ranker" in selected_models:
        lgb = load_lightgbm_module()
        ranker_lgb_path = config.build_ranker_model_lgb_path(experiment_name)
        if not ranker_lgb_path.exists():
            raise FileNotFoundError(f"LightGBM model not found: {ranker_lgb_path}")
        models["lgb_ranker"] = lgb.Booster(model_file=str(ranker_lgb_path))

    if "hgb_regressor" in selected_models:
        ranker_hgb_path = config.build_ranker_model_hgb_path(experiment_name)
        if not ranker_hgb_path.exists():
            raise FileNotFoundError(f"HGB model not found: {ranker_hgb_path}")
        models["hgb_regressor"] = joblib.load(ranker_hgb_path)

    if not models:
        raise ValueError("No loadable models found in metadata.")
    return models


def predict_model(model_name: str, model, features) -> np.ndarray:
    if model_name == "xgb_ranker":
        return np.asarray(model.predict(features), dtype=float)
    if model_name == "lgb_ranker":
        return np.asarray(model.predict(features), dtype=float)
    if model_name == "hgb_regressor":
        return np.asarray(model.predict(features), dtype=float)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_default_score_overlay(config: ProjectConfig) -> dict[str, object]:
    return {
        "enabled": bool(config.production_score_overlay_enabled),
        "method": config.production_score_overlay_method,
        "feature": config.production_score_overlay_feature,
        "weight": float(config.production_score_overlay_weight),
    }


def apply_score_overlay(
    inference_frame,
    base_scores: np.ndarray,
    overlay_config: dict[str, object],
) -> np.ndarray:
    if not bool(overlay_config.get("enabled", False)):
        return np.asarray(base_scores, dtype=float)

    method = str(overlay_config.get("method", "additive_zscore"))
    feature_name = str(overlay_config.get("feature", "")).strip()
    weight = float(overlay_config.get("weight", 0.0))

    if method != "additive_zscore":
        raise ValueError(f"Unsupported score overlay method: {method}")
    if not feature_name:
        raise ValueError("Score overlay feature is empty.")
    if feature_name not in inference_frame.columns:
        raise ValueError(f"Score overlay feature not found: {feature_name}")

    feature_values = (
        inference_frame[feature_name]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return safe_zscore(base_scores) + weight * safe_zscore(feature_values)


def main() -> None:
    args = parse_args()
    config = ProjectConfig()

    experiment_name = args.experiment_name or config.experiment_name
    inference_data_arg = args.inference_data or args.test_data
    inference_data_path = Path(inference_data_arg) if inference_data_arg else config.inference_data_path
    metadata_path = config.build_model_metadata_path(experiment_name)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    print("[inference] start model inference")
    print(f"[inference] input: {inference_data_path}")
    print(f"[inference] experiment: {config.build_run_dir(experiment_name)}")
    if args.test_data and not args.inference_data:
        print("[inference] --test-data is deprecated, prefer --inference-data")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns: list[str] = metadata["feature_columns"]
    feature_windows = tuple(metadata["feature_windows"])
    selected_models: list[str] = metadata.get("selected_models", ["xgb_ranker", "lgb_ranker"])
    ensemble_method = metadata.get("ensemble_method", "zscore_average_equal")
    ensemble_weights: dict[str, float] = metadata.get("ensemble_weights", {})
    score_overlay: dict[str, object] = metadata.get("score_overlay", build_default_score_overlay(config))
    portfolio_size: int = int(metadata.get("portfolio_size", config.portfolio_size))
    portfolio_size = max(1, min(portfolio_size, config.max_portfolio_size))

    raw_dataframe = load_dataframe(inference_data_path)
    inference_frame = prepare_inference_frame(
        raw_dataframe,
        windows=feature_windows,
        feature_columns=feature_columns,
    )
    models = load_selected_models(config, experiment_name, selected_models)

    x = inference_frame[feature_columns]
    score_map = {model_name: predict_model(model_name, model, x) for model_name, model in models.items()}
    inference_frame["base_pred_score"] = combine_model_scores(
        score_map,
        method=ensemble_method,
        weights=ensemble_weights,
    )
    inference_frame["pred_score"] = apply_score_overlay(
        inference_frame,
        inference_frame["base_pred_score"].to_numpy(dtype=float),
        score_overlay,
    )
    submission = create_submission_from_scores(
        inference_frame,
        max_positions=portfolio_size,
    )
    model_names = list(models.keys())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.result_path, index=False, encoding="utf-8")

    check_result = validate_submission_frame(
        submission,
        max_positions=config.max_portfolio_size,
        required_columns=config.result_columns,
        weight_upper_bound=config.weight_upper_bound,
    )
    final_result = SubmissionCheckResult(
        row_count=check_result.row_count,
        unique_stock_count=check_result.unique_stock_count,
        weight_sum=check_result.weight_sum,
        stock_ids=check_result.stock_ids,
        result_path=str(config.result_path),
    )
    write_submission_report(config.submission_check_path, final_result)

    print("[inference] model inference completed")
    print(f"[inference] method: {ensemble_method}")
    print(f"[inference] models: {model_names}")
    print(f"[inference] score_overlay: {score_overlay}")
    print(f"[inference] result: {config.result_path}")
    print(f"[inference] row_count: {final_result.row_count}")
    print(f"[inference] stock_ids: {final_result.stock_ids}")
    print(f"[inference] weight_sum: {final_result.weight_sum:.6f}")
    print(f"[inference] check_report: {config.submission_check_path}")


if __name__ == "__main__":
    main()
