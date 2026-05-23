#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import baostock as bs
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通过 baostock 直接下载沪深300股票历史数据。")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2026-03-15", help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument(
        "--output",
        type=str,
        default="data/stock_data.csv",
        help="历史数据输出文件，默认 data/stock_data.csv",
    )
    parser.add_argument(
        "--stock-list-output",
        type=str,
        default="data/hs300_stock_list.csv",
        help="沪深300成分股列表输出文件，默认 data/hs300_stock_list.csv",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.02,
        help="每只股票请求后的等待时间，避免请求过快。",
    )
    parser.add_argument(
        "--adjustflag",
        type=str,
        default="1",
        help="复权方式，1 表示后复权。",
    )
    return parser.parse_args()


def login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock 登录失败：{result.error_msg}")
    print("[数据] baostock 登录成功。")


def logout() -> None:
    bs.logout()
    print("[数据] baostock 已退出。")


def query_hs300_stocks() -> pd.DataFrame:
    print("[数据] 正在获取沪深300成分股列表...")
    result = bs.query_hs300_stocks()
    if result.error_code != "0":
        raise RuntimeError(f"获取沪深300成分股失败：{result.error_msg}")

    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    stock_frame = pd.DataFrame(rows, columns=result.fields)
    if stock_frame.empty:
        raise RuntimeError("未获取到沪深300成分股列表。")

    stock_frame["stock_id"] = (
        stock_frame["code"]
        .astype(str)
        .str.replace("sh.", "", regex=False)
        .str.replace("sz.", "", regex=False)
        .str.zfill(6)
    )
    print(f"[数据] 成功获取 {len(stock_frame)} 只沪深300成分股。")
    return stock_frame


def query_stock_history(
    baostock_code: str,
    *,
    start_date: str,
    end_date: str,
    adjustflag: str,
) -> pd.DataFrame | None:
    result = bs.query_history_k_data_plus(
        baostock_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )
    if result.error_code != "0":
        raise RuntimeError(f"{baostock_code} 查询失败：{result.error_msg}")

    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    if not rows:
        return None

    history_frame = pd.DataFrame(rows, columns=result.fields)
    numeric_columns = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for column_name in numeric_columns:
        history_frame[column_name] = pd.to_numeric(history_frame[column_name], errors="coerce")

    history_frame["amplitude"] = (
        (history_frame["high"] - history_frame["low"]) / (history_frame["preclose"] + 1e-12) * 100.0
    )
    history_frame["change_amount"] = history_frame["close"] - history_frame["preclose"]
    history_frame["trade_date"] = pd.to_datetime(history_frame["date"], errors="coerce")
    history_frame["stock_id"] = (
        history_frame["code"]
        .astype(str)
        .str.replace("sh.", "", regex=False)
        .str.replace("sz.", "", regex=False)
        .str.zfill(6)
    )

    output_columns = [
        "stock_id",
        "trade_date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "amplitude",
        "change_amount",
    ]
    history_frame = history_frame[output_columns].rename(
        columns={
            "turn": "turnover_rate",
            "pctChg": "pct_chg",
        }
    )
    history_frame["trade_date"] = history_frame["trade_date"].dt.strftime("%Y-%m-%d")
    return history_frame


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    stock_list_output_path = Path(args.stock_list_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stock_list_output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[数据] 开始执行股票历史数据下载。")
    print(f"[数据] 时间范围：{args.start_date} ~ {args.end_date}")
    print(f"[数据] 历史数据输出：{output_path}")

    login()
    try:
        stock_list_frame = query_hs300_stocks()
        stock_list_frame.to_csv(stock_list_output_path, index=False, encoding="utf-8")
        print(f"[数据] 成分股列表已写入：{stock_list_output_path}")
        print("[数据] 提示：该文件通常只有 300 行，因为它只是当前沪深300成分股名单，不是训练用历史行情数据。")

        history_frames: list[pd.DataFrame] = []
        failed_codes: list[str] = []

        for index, row in stock_list_frame.iterrows():
            stock_id = row["stock_id"]
            baostock_code = row["code"]
            print(f"[数据] 正在下载 {index + 1}/{len(stock_list_frame)}：{stock_id}")

            try:
                history_frame = query_stock_history(
                    baostock_code,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adjustflag=args.adjustflag,
                )
                if history_frame is not None and not history_frame.empty:
                    history_frames.append(history_frame)
                else:
                    failed_codes.append(stock_id)
                    print(f"[数据] {stock_id} 未返回历史数据。")
            except Exception as error:
                failed_codes.append(stock_id)
                print(f"[数据] {stock_id} 下载失败：{error}")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        if not history_frames:
            raise RuntimeError("所有股票历史数据下载均失败，未生成数据文件。")

        full_history = pd.concat(history_frames, ignore_index=True)
        full_history = full_history.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
        full_history.to_csv(output_path, index=False, encoding="utf-8")

        print("[数据] 历史数据下载完成。")
        print(f"[数据] 成功写入 {len(full_history)} 行。")
        print(f"[数据] 文件路径：{output_path}")
        print("[数据] 下一步请使用 data/split_train_test.py 将 stock_data.csv 切分为 train.csv 和 test.csv。")
        if failed_codes:
            print(f"[数据] 有 {len(failed_codes)} 只股票下载失败或无数据，例如：{failed_codes[:10]}")
    finally:
        logout()


if __name__ == "__main__":
    main()
