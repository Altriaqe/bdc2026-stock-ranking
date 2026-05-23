from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "stock_id": ("stock_id", "StockID", "ticker", "symbol", "股票代码"),
    "trade_date": ("trade_date", "date", "日期"),
    "open": ("open", "开盘"),
    "close": ("close", "收盘"),
    "high": ("high", "最高"),
    "low": ("low", "最低"),
    "volume": ("volume", "成交量"),
    "amount": ("amount", "成交额"),
    "amplitude": ("amplitude", "振幅"),
    "change_amount": ("change_amount", "涨跌额"),
    "pct_chg": ("pct_chg", "涨跌幅"),
    "turnover_rate": ("turnover_rate", "turnover", "换手率"),
}
REQUIRED_CANONICAL_COLUMNS = ("stock_id", "trade_date", "open", "close", "high", "low", "volume")


@dataclass(frozen=True)
class RankingDatasetBundle:
    ranking_frame: pd.DataFrame
    feature_columns: list[str]
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到必需的数据文件：{csv_path}")

    if csv_path.stat().st_size == 0:
        raise ValueError(f"数据文件为空：{csv_path}")

    return pd.read_csv(csv_path)


def _find_matching_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    stripped_columns = {str(column).strip(): str(column) for column in df.columns}
    for alias in aliases:
        if alias in stripped_columns:
            return stripped_columns[alias]

    for column in df.columns:
        normalized = str(column).strip().lower()
        for alias in aliases:
            if alias.lower() == normalized:
                return str(column)
    return None


def resolve_stock_id_column(df: pd.DataFrame) -> str:
    column_name = _find_matching_column(df, CANONICAL_COLUMN_ALIASES["stock_id"])
    if column_name is None:
        raise ValueError(
            "未识别到股票代码列。当前支持的列名包括："
            f"{', '.join(CANONICAL_COLUMN_ALIASES['stock_id'])}"
        )
    return column_name


def resolve_date_column(df: pd.DataFrame) -> str | None:
    return _find_matching_column(df, CANONICAL_COLUMN_ALIASES["trade_date"])


def normalize_stock_id(raw_value: object) -> str:
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return ""

    value = value.replace("sh.", "").replace("sz.", "").replace("SH.", "").replace("SZ.", "")
    if "." in value:
        value = value.split(".", maxsplit=1)[0]

    return value.zfill(6) if value.isdigit() else value


def summarize_dataframe(df: pd.DataFrame) -> dict[str, int | str]:
    summary: dict[str, int | str] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }

    try:
        stock_id_column = resolve_stock_id_column(df)
        summary["stock_id_column"] = stock_id_column
        summary["unique_stock_count"] = int(df[stock_id_column].nunique(dropna=True))
    except ValueError:
        summary["stock_id_column"] = ""
        summary["unique_stock_count"] = 0

    date_column = resolve_date_column(df)
    summary["date_column"] = date_column or ""
    return summary


def standardize_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    missing_required: list[str] = []

    for canonical_name, aliases in CANONICAL_COLUMN_ALIASES.items():
        matched = _find_matching_column(df, aliases)
        if matched is not None:
            rename_map[matched] = canonical_name
        elif canonical_name in REQUIRED_CANONICAL_COLUMNS:
            missing_required.append(canonical_name)

    if missing_required:
        raise ValueError(f"输入数据缺少必需字段：{missing_required}")

    standardized = df.rename(columns=rename_map).copy()
    selected_columns = [
        column_name
        for column_name in CANONICAL_COLUMN_ALIASES
        if column_name in standardized.columns
    ]
    standardized = standardized[selected_columns]

    standardized["stock_id"] = standardized["stock_id"].map(normalize_stock_id)
    standardized["trade_date"] = pd.to_datetime(standardized["trade_date"], errors="coerce")

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "change_amount",
        "pct_chg",
        "turnover_rate",
    ]
    for column_name in numeric_columns:
        if column_name in standardized.columns:
            standardized[column_name] = pd.to_numeric(standardized[column_name], errors="coerce")

    standardized = standardized.dropna(subset=list(REQUIRED_CANONICAL_COLUMNS)).copy()
    standardized = standardized.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    return standardized


def engineer_features(df: pd.DataFrame, windows: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    feature_frame = df.copy()
    grouped = feature_frame.groupby("stock_id", sort=False, group_keys=False)
    prev_close = grouped["close"].shift(1)

    feature_frame["intraday_return"] = feature_frame["close"] / (feature_frame["open"] + 1e-12) - 1.0
    feature_frame["high_low_range"] = (feature_frame["high"] - feature_frame["low"]) / (feature_frame["open"] + 1e-12)
    feature_frame["close_to_high"] = feature_frame["close"] / (feature_frame["high"] + 1e-12) - 1.0
    feature_frame["close_to_low"] = feature_frame["close"] / (feature_frame["low"] + 1e-12) - 1.0
    feature_frame["price_position"] = (
        (feature_frame["close"] - feature_frame["low"])
        / (feature_frame["high"] - feature_frame["low"] + 1e-12)
    )
    feature_frame["upper_shadow"] = (
        feature_frame["high"] - np.maximum(feature_frame["open"], feature_frame["close"])
    ) / (feature_frame["open"] + 1e-12)
    feature_frame["lower_shadow"] = (
        np.minimum(feature_frame["open"], feature_frame["close"]) - feature_frame["low"]
    ) / (feature_frame["open"] + 1e-12)
    feature_frame["open_gap_1"] = grouped["open"].pct_change(1).fillna(0.0)
    feature_frame["close_return_1"] = grouped["close"].pct_change(1).fillna(0.0)
    feature_frame["open_to_prev_close"] = feature_frame["open"] / (prev_close + 1e-12) - 1.0
    feature_frame["volume_return_1"] = grouped["volume"].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame["log_volume"] = np.log1p(feature_frame["volume"])

    feature_columns: list[str] = [
        "intraday_return",
        "high_low_range",
        "close_to_high",
        "close_to_low",
        "price_position",
        "upper_shadow",
        "lower_shadow",
        "open_gap_1",
        "close_return_1",
        "open_to_prev_close",
        "volume_return_1",
        "log_volume",
    ]

    for window in windows:
        rolling_close_mean = grouped["close"].transform(lambda series: series.rolling(window, min_periods=1).mean())
        rolling_close_std = grouped["close_return_1"].transform(
            lambda series: series.rolling(window, min_periods=2).std()
        )
        rolling_close_mean_return = grouped["close_return_1"].transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        rolling_volume_mean = grouped["volume"].transform(lambda series: series.rolling(window, min_periods=1).mean())
        rolling_high_max = grouped["high"].transform(lambda series: series.rolling(window, min_periods=1).max())
        rolling_low_min = grouped["low"].transform(lambda series: series.rolling(window, min_periods=1).min())

        feature_frame[f"close_return_{window}"] = grouped["close"].pct_change(window).replace([np.inf, -np.inf], np.nan)
        feature_frame[f"volume_return_{window}"] = grouped["volume"].pct_change(window).replace([np.inf, -np.inf], np.nan)
        feature_frame[f"return_mean_{window}"] = rolling_close_mean_return
        feature_frame[f"price_to_ma_{window}"] = feature_frame["close"] / (rolling_close_mean + 1e-12) - 1.0
        feature_frame[f"volatility_{window}"] = rolling_close_std
        feature_frame[f"volume_ratio_{window}"] = feature_frame["volume"] / (rolling_volume_mean + 1e-12)
        feature_frame[f"high_breakout_{window}"] = feature_frame["close"] / (rolling_high_max + 1e-12) - 1.0
        feature_frame[f"low_rebound_{window}"] = feature_frame["close"] / (rolling_low_min + 1e-12) - 1.0
        feature_frame[f"range_position_{window}"] = (
            (feature_frame["close"] - rolling_low_min) / (rolling_high_max - rolling_low_min + 1e-12)
        )

        feature_columns.extend(
            [
                f"close_return_{window}",
                f"volume_return_{window}",
                f"return_mean_{window}",
                f"price_to_ma_{window}",
                f"volatility_{window}",
                f"volume_ratio_{window}",
                f"high_breakout_{window}",
                f"low_rebound_{window}",
                f"range_position_{window}",
            ]
        )

    cross_section_columns = [
        "close_return_1",
        "intraday_return",
        "price_to_ma_5",
        "price_to_ma_20",
        "volume_ratio_5",
        "volatility_10",
    ]
    for column_name in cross_section_columns:
        if column_name in feature_frame.columns:
            rank_column = f"{column_name}_cs_rank"
            feature_frame[rank_column] = feature_frame.groupby("trade_date")[column_name].rank(pct=True)
            feature_columns.append(rank_column)

    if "turnover_rate" in feature_frame.columns:
        feature_frame["turnover_rate"] = feature_frame["turnover_rate"].fillna(0.0)
        feature_columns.append("turnover_rate")
        for window in windows:
            turnover_mean = grouped["turnover_rate"].transform(lambda series: series.rolling(window, min_periods=1).mean())
            ratio_column = f"turnover_ratio_{window}"
            feature_frame[ratio_column] = feature_frame["turnover_rate"] / (turnover_mean + 1e-12)
            feature_columns.append(ratio_column)
        feature_frame["turnover_rate_cs_rank"] = feature_frame.groupby("trade_date")["turnover_rate"].rank(pct=True)
        feature_columns.append("turnover_rate_cs_rank")

    if "amount" in feature_frame.columns:
        feature_frame["log_amount"] = np.log1p(feature_frame["amount"].clip(lower=0.0))
        feature_columns.append("log_amount")
        for window in windows:
            amount_mean = grouped["amount"].transform(lambda series: series.rolling(window, min_periods=1).mean())
            ratio_column = f"amount_ratio_{window}"
            feature_frame[ratio_column] = feature_frame["amount"] / (amount_mean + 1e-12)
            feature_columns.append(ratio_column)

    if "amplitude" in feature_frame.columns:
        feature_frame["amplitude"] = feature_frame["amplitude"].fillna(0.0)
        feature_columns.append("amplitude")
        feature_frame["amplitude_cs_rank"] = feature_frame.groupby("trade_date")["amplitude"].rank(pct=True)
        feature_columns.append("amplitude_cs_rank")

    if "change_amount" in feature_frame.columns:
        feature_frame["change_amount"] = feature_frame["change_amount"].fillna(0.0)
        feature_columns.append("change_amount")

    if "pct_chg" in feature_frame.columns:
        feature_frame["pct_chg"] = feature_frame["pct_chg"].fillna(0.0) / 100.0
        feature_columns.append("pct_chg")
        feature_frame["pct_chg_cs_rank"] = feature_frame.groupby("trade_date")["pct_chg"].rank(pct=True)
        feature_columns.append("pct_chg_cs_rank")

    feature_frame[feature_columns] = (
        feature_frame[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return feature_frame, feature_columns


def _build_relevance_labels(df: pd.DataFrame, *, bucket_count: int) -> pd.Series:
    percentile = df.groupby("trade_date")["future_return"].rank(method="first", ascending=False, pct=True)
    labels = ((1.0 - percentile) * float(bucket_count)).clip(
        lower=0.0,
        upper=float(bucket_count) - 1e-6,
    ).astype(int)
    return labels


def split_train_validation_by_date(
    ranking_frame: pd.DataFrame,
    *,
    validation_ratio: float,
    min_train_groups: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(ranking_frame["trade_date"].drop_duplicates().tolist())
    if len(unique_dates) < min_train_groups + 1:
        raise ValueError(
            "可用于训练的交易日过少，无法切分训练集和验证集。"
            f" 当前只有 {len(unique_dates)} 个交易日，至少需要 {min_train_groups + 1} 个。"
        )

    validation_group_count = max(1, int(round(len(unique_dates) * validation_ratio)))
    validation_group_count = min(validation_group_count, len(unique_dates) - min_train_groups)
    validation_dates = set(unique_dates[-validation_group_count:])

    train_frame = ranking_frame[~ranking_frame["trade_date"].isin(validation_dates)].copy()
    validation_frame = ranking_frame[ranking_frame["trade_date"].isin(validation_dates)].copy()

    if train_frame["trade_date"].nunique() < min_train_groups:
        raise ValueError("训练集交易日数量不足，无法完成排序模型训练。")
    if validation_frame["trade_date"].nunique() < 1:
        raise ValueError("验证集交易日数量不足，无法完成验证。")

    return train_frame, validation_frame


def build_training_bundle(
    raw_df: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    future_buy_offset: int,
    future_sell_offset: int,
    validation_ratio: float,
    min_train_groups: int,
    label_bucket_count: int,
) -> RankingDatasetBundle:
    market_df = standardize_market_dataframe(raw_df)
    min_rows_per_stock = int(market_df.groupby("stock_id").size().min()) if not market_df.empty else 0
    unique_dates = int(market_df["trade_date"].nunique()) if not market_df.empty else 0
    feature_frame, feature_columns = engineer_features(market_df, windows=windows)

    grouped = feature_frame.groupby("stock_id", sort=False, group_keys=False)
    feature_frame["open_t_buy"] = grouped["open"].shift(-future_buy_offset)
    feature_frame["open_t_sell"] = grouped["open"].shift(-future_sell_offset)
    feature_frame["future_return"] = (
        feature_frame["open_t_sell"] / (feature_frame["open_t_buy"] + 1e-12) - 1.0
    )

    ranking_frame = feature_frame.dropna(subset=["future_return"]).copy()
    ranking_frame = ranking_frame[ranking_frame["open_t_buy"] > 0].copy()
    ranking_frame["relevance_label"] = _build_relevance_labels(
        ranking_frame,
        bucket_count=label_bucket_count,
    )
    ranking_frame = ranking_frame.sort_values(["trade_date", "stock_id"]).reset_index(drop=True)

    if ranking_frame.empty:
        raise ValueError(
            "训练数据在按 T+1 开盘买入、T+5 开盘卖出构造标签后为空。"
            f" 当前每只股票最少只有 {min_rows_per_stock} 条记录，当前交易日数量为 {unique_dates}。"
            f" 至少需要每只股票有 {future_sell_offset + 1} 条以上连续记录，且需要足够多的交易日用于训练/验证切分。"
        )

    ranking_frame["qid"] = pd.factorize(ranking_frame["trade_date"].astype(str))[0].astype(int)
    train_frame, validation_frame = split_train_validation_by_date(
        ranking_frame,
        validation_ratio=validation_ratio,
        min_train_groups=min_train_groups,
    )
    return RankingDatasetBundle(
        ranking_frame=ranking_frame,
        feature_columns=feature_columns,
        train_frame=train_frame,
        validation_frame=validation_frame,
    )


def prepare_inference_frame(
    raw_df: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    feature_columns: list[str],
) -> pd.DataFrame:
    market_df = standardize_market_dataframe(raw_df)
    feature_frame, _ = engineer_features(market_df, windows=windows)

    latest_rows = (
        feature_frame.sort_values(["stock_id", "trade_date"])
        .groupby("stock_id", group_keys=False)
        .tail(1)
        .copy()
    )

    for column_name in feature_columns:
        if column_name not in latest_rows.columns:
            latest_rows[column_name] = 0.0

    latest_rows[feature_columns] = (
        latest_rows[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    latest_rows = latest_rows.sort_values(["trade_date", "stock_id"]).reset_index(drop=True)
    return latest_rows


def create_submission_from_scores(
    scored_frame: pd.DataFrame,
    *,
    max_positions: int,
) -> pd.DataFrame:
    if scored_frame.empty:
        raise ValueError("没有可用于生成结果的股票样本。")

    ranked = scored_frame.sort_values("pred_score", ascending=False).copy()
    top_count = min(max_positions, len(ranked))
    top_stocks = ranked.head(top_count)["stock_id"].tolist()

    equal_weight = round(1.0 / top_count, 6)
    weights = [equal_weight for _ in range(top_count)]
    weights[-1] = round(1.0 - sum(weights[:-1]), 6)
    return pd.DataFrame({"stock_id": top_stocks, "weight": weights})


def calculate_top_k_return_metrics(
    scored_frame: pd.DataFrame,
    *,
    k: int,
) -> dict[str, float]:
    evaluation_frame = scored_frame.copy()
    evaluation_frame = evaluation_frame.sort_values(["trade_date", "stock_id"]).reset_index(drop=True)

    predicted_values: list[float] = []
    oracle_values: list[float] = []
    baseline_values: list[float] = []

    for _, one_day_frame in evaluation_frame.groupby("trade_date", sort=True):
        predicted_values.append(float(one_day_frame.nlargest(k, "pred_score")["future_return"].sum()))
        oracle_values.append(float(one_day_frame.nlargest(k, "future_return")["future_return"].sum()))
        baseline_values.append(float(one_day_frame["future_return"].mean() * min(k, len(one_day_frame))))

    pred_mean = float(np.mean(predicted_values)) if predicted_values else 0.0
    oracle_mean = float(np.mean(oracle_values)) if oracle_values else 0.0
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else 0.0

    denominator = oracle_mean - baseline_mean
    normalized_score = 0.0
    if abs(denominator) > 1e-12:
        normalized_score = (pred_mean - baseline_mean) / denominator

    return {
        "pred_top_k_return_mean": round(pred_mean, 8),
        "oracle_top_k_return_mean": round(oracle_mean, 8),
        "baseline_top_k_return_mean": round(baseline_mean, 8),
        "top_k_relative_score": round(normalized_score, 8),
    }
