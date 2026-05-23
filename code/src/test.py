from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import zscore

from compliance import SubmissionCheckResult, validate_submission_frame, write_submission_report
from config import ProjectConfig
from featurework import (
    create_submission_from_scores,
    load_dataframe,
    prepare_inference_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的双模型集成生成结果文件。")
    parser.add_argument("--inference-data", type=str, default=None, help="推理输入数据路径，默认使用 data/train.csv")
    parser.add_argument("--test-data", type=str, default=None, help="兼容旧参数")
    parser.add_argument("--experiment-name", type=str, default=None, help="实验目录名称")
    return parser.parse_args()


def load_xgboost_module():
    import xgboost as xgb
    return xgb


def load_lightgbm_module():
    import lightgbm as lgb
    return lgb


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    xgb = load_xgboost_module()
    lgb = load_lightgbm_module()

    experiment_name = args.experiment_name or config.experiment_name
    inference_data_arg = args.inference_data or args.test_data
    inference_data_path = Path(inference_data_arg) if inference_data_arg else config.inference_data_path
    metadata_path = config.build_model_metadata_path(experiment_name)
    ranker_xgb_path = config.build_ranker_model_path(experiment_name)
    ranker_lgb_path = config.build_ranker_model_lgb_path(experiment_name)

    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到模型元数据文件：{metadata_path}")
    if not ranker_xgb_path.exists():
        raise FileNotFoundError(f"未找到 XGBoost 模型：{ranker_xgb_path}")
    if not ranker_lgb_path.exists():
        raise FileNotFoundError(f"未找到 LightGBM 模型：{ranker_lgb_path}")

    print("[推理] 开始执行 XGBoost + LightGBM 双模型集成推理。")
    print(f"[推理] 推理输入：{inference_data_path}")
    print(f"[推理] 实验目录：{config.build_run_dir(experiment_name)}")
    if args.test_data and not args.inference_data:
        print("[推理] 提示：你使用了兼容旧参数 --test-data，建议改用 --inference-data。")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns: list[str] = metadata["feature_columns"]
    feature_windows = tuple(metadata["feature_windows"])

    raw_dataframe = load_dataframe(inference_data_path)
    inference_frame = prepare_inference_frame(
        raw_dataframe,
        windows=feature_windows,
        feature_columns=feature_columns,
    )

    # Load both models
    xgb_ranker = xgb.XGBRanker()
    xgb_ranker.load_model(str(ranker_xgb_path))
    lgb_ranker = lgb.Booster(model_file=str(ranker_lgb_path))

    # Predict and ensemble
    x = inference_frame[feature_columns]
    xgb_scores = xgb_ranker.predict(x)
    lgb_scores = lgb_ranker.predict(x)

    xgb_z = zscore(np.clip(xgb_scores, -1e10, 1e10))
    lgb_z = zscore(np.clip(lgb_scores, -1e10, 1e10))
    inference_frame["pred_score"] = (xgb_z + lgb_z) / 2.0

    submission = create_submission_from_scores(
        inference_frame,
        max_positions=config.max_portfolio_size,
    )

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

    print("[推理] 双模型集成推理完成。")
    print(f"[推理] 结果文件：{config.result_path}")
    print(f"[推理] 股票数量：{final_result.row_count}")
    print(f"[推理] 股票代码：{final_result.stock_ids}")
    print(f"[推理] 权重和：{final_result.weight_sum:.6f}")
    print(f"[推理] 校验报告：{config.submission_check_path}")


if __name__ == "__main__":
    main()
