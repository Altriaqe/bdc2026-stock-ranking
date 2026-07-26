from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compliance import validate_submission_frame


@dataclass(frozen=True)
class MarketDataReport:
    row_count: int
    stock_count: int
    date_min: str
    date_max: str
    latest_day_stock_count: int
    duplicate_key_count: int
    invalid_ohlc_count: int
    critical_missing_count: int
    suspended_metric_missing_count: int
    short_history_stock_count: int
    stock_ids: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubmissionReport:
    row_count: int
    unique_stock_count: int
    weight_sum: float
    stock_ids: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TarReport:
    size_bytes: int
    member_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_stock_ids(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    if values.str.fullmatch(r"\d{6}").all():
        return values
    padded = values.str.zfill(6)
    if not padded.str.fullmatch(r"\d{6}").all():
        raise ValueError("stock_id 必须是 6 位数字。")
    return padded


def validate_market_data(
    path: Path,
    cutoff_date: date,
    *,
    minimum_stock_count: int = 300,
    minimum_history_days: int = 60,
) -> MarketDataReport:
    if not path.is_file():
        raise FileNotFoundError(f"行情文件不存在：{path}")
    frame = pd.read_csv(path, dtype={"stock_id": str})
    required = {"stock_id", "trade_date", "open", "close", "high", "low"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"行情文件缺少字段：{sorted(missing_columns)}")

    frame["stock_id"] = _normalize_stock_ids(frame["stock_id"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if frame["trade_date"].isna().any():
        raise ValueError("行情文件存在无法解析的 trade_date。")
    numeric_columns = ["open", "close", "high", "low"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    critical_missing_count = int(frame[numeric_columns].isna().any(axis=1).sum())
    if critical_missing_count:
        raise ValueError(f"行情文件关键价格缺失行数：{critical_missing_count}")

    duplicate_key_count = int(frame.duplicated(["stock_id", "trade_date"]).sum())
    if duplicate_key_count:
        raise ValueError(f"行情文件存在重复主键：{duplicate_key_count}")
    invalid_ohlc = (
        (frame["open"] <= 0)
        | (frame["close"] <= 0)
        | (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    invalid_ohlc_count = int(invalid_ohlc.sum())
    if invalid_ohlc_count:
        raise ValueError(f"行情文件存在非法 OHLC 行：{invalid_ohlc_count}")

    stock_ids = tuple(sorted(frame["stock_id"].unique()))
    if len(stock_ids) < minimum_stock_count:
        raise ValueError(f"行情股票数不足：{len(stock_ids)} < {minimum_stock_count}")
    max_date = frame["trade_date"].max().date()
    if max_date != cutoff_date:
        raise ValueError(f"行情最大日期为 {max_date}，不等于要求截止日 {cutoff_date}")
    latest_day_stock_count = int(frame.loc[frame["trade_date"].dt.date == cutoff_date, "stock_id"].nunique())
    if latest_day_stock_count < minimum_stock_count:
        raise ValueError(f"截止日股票覆盖不足：{latest_day_stock_count} < {minimum_stock_count}")
    short_history_stock_count = int((frame.groupby("stock_id").size() < minimum_history_days).sum())
    if short_history_stock_count:
        raise ValueError(f"历史不足 {minimum_history_days} 日的股票数：{short_history_stock_count}")

    metric_columns = [column for column in ("volume", "amount", "turnover_rate", "pct_chg") if column in frame]
    suspended_metric_missing_count = int(frame[metric_columns].isna().any(axis=1).sum()) if metric_columns else 0
    return MarketDataReport(
        row_count=int(len(frame)),
        stock_count=len(stock_ids),
        date_min=frame["trade_date"].min().date().isoformat(),
        date_max=max_date.isoformat(),
        latest_day_stock_count=latest_day_stock_count,
        duplicate_key_count=duplicate_key_count,
        invalid_ohlc_count=invalid_ohlc_count,
        critical_missing_count=critical_missing_count,
        suspended_metric_missing_count=suspended_metric_missing_count,
        short_history_stock_count=short_history_stock_count,
        stock_ids=stock_ids,
        sha256=sha256_file(path),
    )


def validate_result(path: Path, allowed_stock_ids: set[str] | None = None) -> SubmissionReport:
    raw = path.read_bytes()
    if b"\r\n" in raw or b"\r" in raw:
        raise ValueError("结果文件必须使用 LF 行尾。")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("结果文件必须是 UTF-8 编码。") from error
    frame = pd.read_csv(path, dtype={"stock_id": str})
    check = validate_submission_frame(
        frame,
        max_positions=5,
        required_columns=("stock_id", "weight"),
        weight_upper_bound=1.0,
        exact_positions=5,
        exact_weight=0.2,
    )
    stock_ids = tuple(_normalize_stock_ids(frame["stock_id"]).tolist())
    if allowed_stock_ids is not None and not set(stock_ids).issubset(allowed_stock_ids):
        raise ValueError("结果包含不在候选股票池中的股票。")
    return SubmissionReport(
        row_count=check.row_count,
        unique_stock_count=check.unique_stock_count,
        weight_sum=check.weight_sum,
        stock_ids=stock_ids,
        sha256=sha256_file(path),
    )


def validate_tar(path: Path, max_bytes: int = 10_000_000_000) -> TarReport:
    size_bytes = path.stat().st_size
    if size_bytes > max_bytes:
        raise ValueError(f"tar 文件超过大小上限：{size_bytes} > {max_bytes}")
    with tarfile.open(path, mode="r") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"tar 包含不安全路径：{member.name}")
    return TarReport(size_bytes=size_bytes, member_count=len(members), sha256=sha256_file(path))


def _metric(strategy: dict[str, Any], key: str) -> float:
    value = strategy.get(key)
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"回测指标缺失或无效：{key}")
    return float(value)


def assert_backtest_gates(report: dict[str, Any]) -> None:
    folds = report.get("purged_outer_folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ValueError("回测必须包含 4 个 purged outer folds。")
    fixed_values: list[dict[str, float]] = []
    old_values: list[dict[str, float]] = []
    for fold in folds:
        if int(fold.get("purge_group_count", -1)) != 5:
            raise ValueError("回测每个 outer fold 必须 purge 5 个交易日。")
        strategies = fold.get("strategy_metrics", {})
        fixed = strategies.get("equal_rank_ensemble_top5")
        old = strategies.get("current_overlay_top1")
        if not isinstance(fixed, dict) or not isinstance(old, dict):
            raise ValueError("回测缺少生产策略或旧基线策略。")
        fixed_values.append({key: _metric(fixed, key) for key in ("pred_top_k_return_mean", "pred_top_k_worst")})
        old_values.append({key: _metric(old, key) for key in ("pred_top_k_return_mean", "pred_top_k_worst")})
    fixed_mean = float(np.mean([item["pred_top_k_return_mean"] for item in fixed_values]))
    latest_mean = fixed_values[-1]["pred_top_k_return_mean"]
    fixed_worst = min(item["pred_top_k_worst"] for item in fixed_values)
    old_worst = min(item["pred_top_k_worst"] for item in old_values)
    if fixed_mean <= 0 or latest_mean <= 0:
        raise ValueError(f"固定等权 Top5 收益门槛未通过：mean={fixed_mean}, latest={latest_mean}")
    if fixed_worst <= old_worst:
        raise ValueError(f"固定等权 Top5 最差周未优于旧基线：{fixed_worst} <= {old_worst}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
