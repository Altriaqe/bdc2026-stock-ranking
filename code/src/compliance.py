from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SubmissionCheckResult:
    row_count: int
    unique_stock_count: int
    weight_sum: float
    stock_ids: list[str]
    result_path: str


def validate_submission_frame(
    df: pd.DataFrame,
    *,
    max_positions: int,
    required_columns: tuple[str, str],
    weight_upper_bound: float,
) -> SubmissionCheckResult:
    expected_columns = list(required_columns)
    current_columns = list(df.columns)
    if current_columns != expected_columns:
        raise ValueError(
            "结果文件表头不符合要求。"
            f" 当前表头：{current_columns}，要求表头：{expected_columns}"
        )

    if df.empty:
        raise ValueError("结果文件不能为空，至少需要输出 1 只股票。")

    if len(df) > max_positions:
        raise ValueError(f"结果文件最多只能包含 {max_positions} 只股票。")

    stock_column, weight_column = required_columns
    normalized_stock_ids = df[stock_column].astype(str).str.strip().tolist()
    if any(not stock_id for stock_id in normalized_stock_ids):
        raise ValueError("结果文件中存在空的股票代码，请检查输出。")

    if len(set(normalized_stock_ids)) != len(normalized_stock_ids):
        raise ValueError("结果文件中存在重复股票代码，请确保每只股票只出现一次。")

    weights = pd.to_numeric(df[weight_column], errors="coerce")
    if weights.isna().any():
        raise ValueError("结果文件中的 weight 列存在无法解析的值。")

    if (weights < 0).any():
        raise ValueError("结果文件中的 weight 不能为负数。")

    weight_sum = float(weights.sum())
    if weight_sum > weight_upper_bound + 1e-8:
        raise ValueError(
            f"结果文件中的权重和不能超过 {weight_upper_bound}，当前为 {weight_sum:.6f}。"
        )

    return SubmissionCheckResult(
        row_count=int(len(df)),
        unique_stock_count=int(len(set(normalized_stock_ids))),
        weight_sum=round(weight_sum, 6),
        stock_ids=normalized_stock_ids,
        result_path="",
    )


def write_submission_report(path: Path, result: SubmissionCheckResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
