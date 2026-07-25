from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compliance import SubmissionCheckResult, validate_submission_frame, write_submission_report
from config import ProjectConfig
from featurework import (
    load_dataframe,
    prepare_inference_frame,
    standardize_market_dataframe,
)
from portfolio import combine_rank_scores, select_top5_portfolio


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    ensemble_method = metadata.get("ensemble_method", "")
    ensemble_weights: dict[str, float] = metadata.get("ensemble_weights", {})
    portfolio_config: dict[str, object] = metadata.get("portfolio", {})
    if ensemble_method != "shrunk_rank_average":
        raise ValueError(f"Unsupported production ensemble method: {ensemble_method}")
    if set(selected_models) != {"xgb_ranker", "lgb_ranker", "hgb_regressor"}:
        raise ValueError("Robust production inference requires all three models.")
    if int(portfolio_config.get("size", 0)) != 5:
        raise ValueError("Robust production metadata must request exactly five stocks.")

    raw_dataframe = load_dataframe(inference_data_path)
    market_frame = standardize_market_dataframe(raw_dataframe)
    inference_frame = prepare_inference_frame(
        raw_dataframe,
        windows=feature_windows,
        feature_columns=feature_columns,
    )
    models = load_selected_models(config, experiment_name, selected_models)

    x = inference_frame[feature_columns]
    score_map = {model_name: predict_model(model_name, model, x) for model_name, model in models.items()}
    inference_frame["pred_score"] = combine_rank_scores(
        score_map,
        ensemble_weights,
        inference_frame["stock_id"],
    )
    selection = select_top5_portfolio(
        inference_frame,
        market_frame,
        candidate_pool_size=int(portfolio_config["candidate_pool_size"]),
        covariance_window=int(portfolio_config["covariance_window"]),
        variance_penalty=float(portfolio_config["variance_penalty"]),
        correlation_penalty=float(portfolio_config["correlation_penalty"]),
    )
    submission = selection.submission
    model_names = list(models.keys())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.result_path, index=False, encoding="utf-8")

    check_result = validate_submission_frame(
        submission,
        max_positions=config.max_portfolio_size,
        required_columns=config.result_columns,
        weight_upper_bound=config.weight_upper_bound,
        exact_positions=5,
        exact_weight=float(portfolio_config["weight"]),
    )
    final_result = SubmissionCheckResult(
        row_count=check_result.row_count,
        unique_stock_count=check_result.unique_stock_count,
        weight_sum=check_result.weight_sum,
        stock_ids=check_result.stock_ids,
        result_path=str(config.result_path),
    )
    write_submission_report(config.submission_check_path, final_result)
    portfolio_report_path = config.temp_dir / config.portfolio_report_filename
    portfolio_report_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_report_path.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "inference_data_path": str(inference_data_path),
                "latest_visible_date": pd.to_datetime(inference_frame["trade_date"]).max().strftime("%Y-%m-%d"),
                "ensemble_method": ensemble_method,
                "ensemble_weights": ensemble_weights,
                "portfolio_config": portfolio_config,
                "candidates": selection.candidates,
                "selected_stock_ids": selection.selected_stock_ids,
                "portfolio_variance": selection.portfolio_variance,
                "mean_correlation": selection.mean_correlation,
                "selection_score": selection.selection_score,
                "degraded_reason": selection.degraded_reason,
                "result_sha256": file_sha256(config.result_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("[inference] model inference completed")
    print(f"[inference] method: {ensemble_method}")
    print(f"[inference] models: {model_names}")
    print(f"[inference] ensemble_weights: {ensemble_weights}")
    print(f"[inference] portfolio: {portfolio_config}")
    print(f"[inference] result: {config.result_path}")
    print(f"[inference] row_count: {final_result.row_count}")
    print(f"[inference] stock_ids: {final_result.stock_ids}")
    print(f"[inference] weight_sum: {final_result.weight_sum:.6f}")
    print(f"[inference] check_report: {config.submission_check_path}")
    print(f"[inference] portfolio_report: {portfolio_report_path}")


if __name__ == "__main__":
    main()
