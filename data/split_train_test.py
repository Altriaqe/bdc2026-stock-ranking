from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "stock_id": ("stock_id", "股票代码"),
    "trade_date": ("trade_date", "日期"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按日期区间切分训练集和测试集。")
    parser.add_argument("--input", type=str, default="data/stock_data.csv", help="原始数据路径。")
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录。")
    parser.add_argument("--train-start", type=str, default="2024-01-02", help="训练集开始日期。")
    parser.add_argument("--train-end", type=str, default="2026-03-06", help="训练集结束日期。")
    parser.add_argument("--test-start", type=str, default="2026-03-09", help="测试集开始日期。")
    parser.add_argument("--test-end", type=str, default="2026-03-13", help="测试集结束日期。")
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    stripped_columns = {str(column).strip(): str(column) for column in df.columns}
    for alias in aliases:
        if alias in stripped_columns:
            return stripped_columns[alias]
    return None


def normalize_date(date_str: str, arg_name: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"{arg_name} 日期格式无效：{date_str}")
    return timestamp.normalize()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[切分] 开始按日期切分训练集和测试集。")
    print(f"[切分] 原始数据：{input_path}")

    data_frame = pd.read_csv(input_path)
    stock_column = resolve_column(data_frame, COLUMN_ALIASES["stock_id"])
    date_column = resolve_column(data_frame, COLUMN_ALIASES["trade_date"])

    if stock_column is None or date_column is None:
        if {"updateDate", "code", "code_name", "stock_id"}.issubset(set(map(str, data_frame.columns))):
            raise ValueError(
                "你传入的文件看起来是 hs300_stock_list.csv。"
                " 这个文件只有约 300 行，只是成分股名单，不能直接切分训练集。"
                " 请先使用 get_stock_data.py 生成包含历史行情的 data/stock_data.csv。"
            )
        raise ValueError("原始数据缺少股票代码列或日期列，无法完成切分。")

    print(f"[切分] 原始数据行数：{len(data_frame)}")
    print(f"[切分] 原始数据股票数：{data_frame[stock_column].nunique()}")

    data_frame[date_column] = pd.to_datetime(data_frame[date_column], errors="coerce")
    if data_frame[date_column].isna().any():
        bad_rows = int(data_frame[date_column].isna().sum())
        raise ValueError(f"原始数据中存在无法解析的日期，共 {bad_rows} 行。")

    train_start = normalize_date(args.train_start, "--train-start")
    train_end = normalize_date(args.train_end, "--train-end")
    test_start = normalize_date(args.test_start, "--test-start")
    test_end = normalize_date(args.test_end, "--test-end")

    if train_start > train_end:
        raise ValueError("训练集开始日期晚于结束日期。")
    if test_start > test_end:
        raise ValueError("测试集开始日期晚于结束日期。")

    train_frame = data_frame[(data_frame[date_column] >= train_start) & (data_frame[date_column] <= train_end)].copy()
    test_frame = data_frame[(data_frame[date_column] >= test_start) & (data_frame[date_column] <= test_end)].copy()

    train_frame = train_frame.sort_values([stock_column, date_column]).reset_index(drop=True)
    test_frame = test_frame.sort_values([stock_column, date_column]).reset_index(drop=True)

    if "trade_date" in train_frame.columns:
        train_frame["trade_date"] = train_frame["trade_date"].dt.strftime("%Y-%m-%d")
        test_frame["trade_date"] = test_frame["trade_date"].dt.strftime("%Y-%m-%d")
    elif "日期" in train_frame.columns:
        train_frame["日期"] = train_frame["日期"].dt.strftime("%Y-%m-%d")
        test_frame["日期"] = test_frame["日期"].dt.strftime("%Y-%m-%d")

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_frame.to_csv(train_path, index=False, encoding="utf-8")
    test_frame.to_csv(test_path, index=False, encoding="utf-8")

    print(f"[切分] 训练集已写入：{train_path}，共 {len(train_frame)} 行，股票数 {train_frame[stock_column].nunique()}")
    print(f"[切分] 测试集已写入：{test_path}，共 {len(test_frame)} 行，股票数 {test_frame[stock_column].nunique()}")
    print(f"[切分] 训练区间：{train_start.date()} ~ {train_end.date()}")
    print(f"[切分] 测试区间：{test_start.date()} ~ {test_end.date()}")


if __name__ == "__main__":
    main()
