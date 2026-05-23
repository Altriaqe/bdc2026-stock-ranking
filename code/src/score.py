from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按官方 score_self 口径计算本地自测得分。")
    parser.add_argument("--submission", type=str, default="output/result.csv", help="待评分结果文件路径。")
    parser.add_argument("--score-data", type=str, default="data/test.csv", help="真实未来一周数据路径。")
    parser.add_argument("--baseline-result", type=str, default=None, help="可选：官方基准结果文件路径。")
    parser.add_argument("--output-json", type=str, default="temp/self_score.json", help="评分报告输出路径。")
    return parser.parse_args()


def normalize_submission_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "stock_id" in df.columns:
        rename_map["stock_id"] = "股票代码"
    if "weight" in df.columns:
        rename_map["weight"] = "权重"
    normalized = df.rename(columns=rename_map).copy()

    required_columns = {"股票代码", "权重"}
    if not required_columns.issubset(normalized.columns):
        raise ValueError("结果文件缺少必要列，必须包含 stock_id/股票代码 与 weight/权重。")

    normalized["股票代码"] = normalized["股票代码"].astype(str).str.strip().str.zfill(6)
    normalized["权重"] = pd.to_numeric(normalized["权重"], errors="coerce")
    if normalized["权重"].isna().any():
        raise ValueError("结果文件中的权重列存在无法解析的值。")
    return normalized[["股票代码", "权重"]]


def validate_submission(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("结果文件不能为空。")
    if len(df) > 5:
        raise ValueError("结果文件最多只能包含 5 只股票。")
    if df["股票代码"].duplicated().any():
        raise ValueError("结果文件中存在重复股票代码。")

    weight_sum = float(df["权重"].sum())
    if weight_sum < 0 or weight_sum > 1.0 + 1e-8:
        raise ValueError(f"结果文件权重和必须位于 0 到 1 之间，当前为 {weight_sum:.6f}。")


def normalize_score_data(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(
        columns={
            "stock_id": "股票代码",
            "trade_date": "日期",
            "open": "开盘",
            "close": "收盘",
        }
    ).copy()
    required_columns = {"股票代码", "日期", "开盘", "收盘"}
    if not required_columns.issubset(normalized.columns):
        raise ValueError("评分数据缺少必要字段，至少需要包含 股票代码/日期/开盘/收盘。")

    normalized["股票代码"] = normalized["股票代码"].astype(str).str.strip().str.zfill(6)
    normalized["日期"] = pd.to_datetime(normalized["日期"], errors="coerce")
    normalized["开盘"] = pd.to_numeric(normalized["开盘"], errors="coerce")
    normalized["收盘"] = pd.to_numeric(normalized["收盘"], errors="coerce")
    normalized = normalized.dropna(subset=["股票代码", "日期", "开盘", "收盘"]).copy()
    return normalized[["股票代码", "日期", "开盘", "收盘"]]


def calculate_return(group: pd.DataFrame) -> float:
    ordered = group.sort_values("日期")
    start = ordered.iloc[0]
    end = ordered.iloc[-1]
    return float((end["开盘"] - start["开盘"]) / (start["开盘"] + 1e-12))


def calculate_score(submission: pd.DataFrame, score_data: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    filtered = score_data[score_data["股票代码"].isin(submission["股票代码"])].copy()
    filtered = filtered.groupby("股票代码", group_keys=False).tail(5).copy()
    if filtered.empty:
        raise ValueError("评分数据中没有匹配到结果文件中的股票代码。")

    details = []
    for stock_code, one_stock in filtered.groupby("股票代码", sort=True):
        details.append({"股票代码": stock_code, "收益率": calculate_return(one_stock)})
    detail = pd.DataFrame(details).merge(submission, on="股票代码", how="left")
    final_score = float((detail["收益率"] * detail["权重"]).sum())
    return final_score, detail


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    score_data_path = Path(args.score_data)
    output_json_path = Path(args.output_json)

    submission = normalize_submission_columns(pd.read_csv(submission_path))
    validate_submission(submission)
    score_data = normalize_score_data(pd.read_csv(score_data_path))

    current_score, detail = calculate_score(submission, score_data)
    payload: dict[str, object] = {
        "submission_path": str(submission_path),
        "score_data_path": str(score_data_path),
        "current_score": current_score,
        "detail": detail.to_dict(orient="records"),
    }

    if args.baseline_result:
        baseline_submission = normalize_submission_columns(pd.read_csv(args.baseline_result))
        validate_submission(baseline_submission)
        baseline_score, baseline_detail = calculate_score(baseline_submission, score_data)
        payload["baseline_result_path"] = str(Path(args.baseline_result))
        payload["baseline_score"] = baseline_score
        payload["delta_vs_baseline"] = current_score - baseline_score
        payload["rank_eligible"] = current_score > baseline_score
        payload["baseline_detail"] = baseline_detail.to_dict(orient="records")

    write_json(output_json_path, payload)

    print("[评分] 本地自测完成。")
    print(f"[评分] 结果文件：{submission_path}")
    print(f"[评分] 评分数据：{score_data_path}")
    print(f"[评分] 当前得分：{current_score:.12f}")
    if args.baseline_result:
        print(f"[评分] 基准得分：{payload['baseline_score']:.12f}")
        print(f"[评分] 与基准差值：{payload['delta_vs_baseline']:.12f}")
        print(f"[评分] 是否超过基准：{payload['rank_eligible']}")
    print(f"[评分] 评分报告：{output_json_path}")


if __name__ == "__main__":
    main()
