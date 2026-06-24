from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


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
    feature_groups: dict[str, list[str]]
    feature_preset: str
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame


FEATURE_PRESET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "baseline_v1": ("baseline_v1",),
    "alpha_v1": ("baseline_v1", "price_volume_alpha_v1"),
    "market_v1": ("baseline_v1", "market_cross_v1"),
    "path_v1": ("baseline_v1", "path_shape_v1"),
    "path_plus_v2": ("baseline_v1", "path_shape_v1", "path_shape_v2"),
    "cross_v1": ("baseline_v1", "market_cross_v1", "path_shape_v1"),
    "full_v1": ("baseline_v1", "price_volume_alpha_v1", "market_cross_v1", "path_shape_v1", "path_shape_v2"),
}
FEATURE_PRESET_ALIASES: dict[str, str] = {
    "base": "baseline_v1",
    "baseline": "baseline_v1",
    "baseline_v1": "baseline_v1",
    "alpha": "alpha_v1",
    "alpha_v1": "alpha_v1",
    "market": "market_v1",
    "market_v1": "market_v1",
    "path": "path_v1",
    "path_v1": "path_v1",
    "path2": "path_plus_v2",
    "path_plus_v2": "path_plus_v2",
    "cross": "cross_v1",
    "cross_v1": "cross_v1",
    "full": "full_v1",
    "full_v1": "full_v1",
}
LEGACY_ALPHA_V1_ORDER: tuple[str, ...] = (
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
    "close_to_vwap",
    "close_return_3",
    "volume_return_3",
    "return_mean_3",
    "price_to_ma_3",
    "volatility_3",
    "volume_ratio_3",
    "high_breakout_3",
    "low_rebound_3",
    "range_position_3",
    "close_return_5",
    "volume_return_5",
    "return_mean_5",
    "price_to_ma_5",
    "volatility_5",
    "volume_ratio_5",
    "high_breakout_5",
    "low_rebound_5",
    "range_position_5",
    "close_return_10",
    "volume_return_10",
    "return_mean_10",
    "price_to_ma_10",
    "volatility_10",
    "volume_ratio_10",
    "high_breakout_10",
    "low_rebound_10",
    "range_position_10",
    "close_return_20",
    "volume_return_20",
    "return_mean_20",
    "price_to_ma_20",
    "volatility_20",
    "volume_ratio_20",
    "high_breakout_20",
    "low_rebound_20",
    "range_position_20",
    "open_volume_corr_10",
    "close_volume_corr_10",
    "vwap_volume_corr_10",
    "open_volume_corr_20",
    "close_volume_corr_20",
    "vwap_volume_corr_20",
    "delta_price_to_ma_10_3",
    "close_return_1_cs_rank",
    "intraday_return_cs_rank",
    "price_to_ma_5_cs_rank",
    "price_to_ma_20_cs_rank",
    "volume_ratio_5_cs_rank",
    "volatility_10_cs_rank",
    "close_to_vwap_cs_rank",
    "delta_price_to_ma_10_3_cs_rank",
    "open_volume_corr_20_cs_rank",
    "turnover_rate",
    "turnover_ratio_3",
    "turnover_ratio_5",
    "turnover_ratio_10",
    "turnover_ratio_20",
    "turnover_rate_cs_rank",
    "log_amount",
    "amount_ratio_3",
    "amount_ratio_5",
    "amount_ratio_10",
    "amount_ratio_20",
    "amplitude",
    "amplitude_cs_rank",
    "change_amount",
    "pct_chg",
    "pct_chg_cs_rank",
)


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


def _grouped_rolling_corr(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    window: int,
) -> pd.Series:
    return df.groupby("stock_id", sort=False, group_keys=False)[[left_column, right_column]].apply(
        lambda frame: frame[left_column].rolling(window, min_periods=2).corr(frame[right_column])
    ).reset_index(level=0, drop=True)


def resolve_feature_preset_name(feature_preset: str) -> str:
    normalized = FEATURE_PRESET_ALIASES.get(feature_preset.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(FEATURE_PRESET_COMPONENTS))
        raise ValueError(f"不支持的特征预设：{feature_preset}。可选值：{supported}")
    return normalized


def select_feature_columns(feature_groups: dict[str, list[str]], feature_preset: str) -> tuple[list[str], str]:
    resolved_preset = resolve_feature_preset_name(feature_preset)
    if resolved_preset == "alpha_v1":
        available_columns = set(feature_groups.get("baseline_v1", [])) | set(feature_groups.get("price_volume_alpha_v1", []))
        selected_columns = [column_name for column_name in LEGACY_ALPHA_V1_ORDER if column_name in available_columns]
        if len(selected_columns) == len(LEGACY_ALPHA_V1_ORDER):
            return selected_columns, resolved_preset
    selected_columns: list[str] = []
    for group_name in FEATURE_PRESET_COMPONENTS[resolved_preset]:
        for column_name in feature_groups.get(group_name, []):
            if column_name not in selected_columns:
                selected_columns.append(column_name)
    return selected_columns, resolved_preset


def engineer_features(
    df: pd.DataFrame,
    windows: tuple[int, ...],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    feature_frame = df.copy()
    grouped = feature_frame.groupby("stock_id", sort=False, group_keys=False)
    prev_close = grouped["close"].shift(1)
    typical_price = (
        feature_frame["open"] + feature_frame["high"] + feature_frame["low"] + feature_frame["close"]
    ) / 4.0
    alpha_windows = tuple(window for window in windows if window in {10, 20})
    feature_columns: list[str] = []
    feature_groups: dict[str, list[str]] = {
        "baseline_v1": [],
        "price_volume_alpha_v1": [],
        "market_cross_v1": [],
        "path_shape_v1": [],
        "path_shape_v2": [],
    }

    def register(columns: list[str], family: str) -> None:
        for column_name in columns:
            if column_name not in feature_columns:
                feature_columns.append(column_name)
            if column_name not in feature_groups[family]:
                feature_groups[family].append(column_name)

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
    if "amount" in feature_frame.columns:
        feature_frame["vwap_proxy"] = feature_frame["amount"] / (feature_frame["volume"] + 1e-12)
    else:
        feature_frame["vwap_proxy"] = typical_price
    feature_frame["close_to_vwap"] = feature_frame["close"] / (feature_frame["vwap_proxy"] + 1e-12) - 1.0

    register([
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
    ], "baseline_v1")
    register(["close_to_vwap"], "price_volume_alpha_v1")

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
        rolling_abs_return_sum = grouped["close_return_1"].transform(
            lambda series: series.abs().rolling(window, min_periods=1).sum()
        )
        rolling_positive_ratio = grouped["close_return_1"].transform(
            lambda series: series.gt(0.0).rolling(window, min_periods=1).mean()
        )
        rolling_positive_sum = grouped["close_return_1"].transform(
            lambda series: series.clip(lower=0.0).rolling(window, min_periods=1).sum()
        )
        rolling_negative_sum = grouped["close_return_1"].transform(
            lambda series: (-series.clip(upper=0.0)).rolling(window, min_periods=1).sum()
        )
        rolling_amount_mean = None
        if "amount" in feature_frame.columns:
            rolling_amount_mean = grouped["amount"].transform(lambda series: series.rolling(window, min_periods=1).mean())

        feature_frame[f"close_return_{window}"] = grouped["close"].pct_change(window).replace([np.inf, -np.inf], np.nan)
        feature_frame[f"volume_return_{window}"] = grouped["volume"].pct_change(window).replace([np.inf, -np.inf], np.nan)
        feature_frame[f"return_mean_{window}"] = rolling_close_mean_return
        feature_frame[f"price_to_ma_{window}"] = feature_frame["close"] / (rolling_close_mean + 1e-12) - 1.0
        feature_frame[f"volatility_{window}"] = rolling_close_std
        feature_frame[f"volume_ratio_{window}"] = feature_frame["volume"] / (rolling_volume_mean + 1e-12)
        if rolling_amount_mean is not None:
            feature_frame[f"amount_ratio_{window}"] = feature_frame["amount"] / (rolling_amount_mean + 1e-12)
        feature_frame[f"high_breakout_{window}"] = feature_frame["close"] / (rolling_high_max + 1e-12) - 1.0
        feature_frame[f"low_rebound_{window}"] = feature_frame["close"] / (rolling_low_min + 1e-12) - 1.0
        feature_frame[f"range_position_{window}"] = (
            (feature_frame["close"] - rolling_low_min) / (rolling_high_max - rolling_low_min + 1e-12)
        )
        feature_frame[f"path_efficiency_{window}"] = (
            feature_frame[f"close_return_{window}"] / (rolling_abs_return_sum + 1e-12)
        )
        feature_frame[f"path_consistency_{window}"] = (
            feature_frame[f"return_mean_{window}"] / (feature_frame[f"volatility_{window}"] + 1e-12)
        )
        feature_frame[f"positive_day_ratio_{window}"] = rolling_positive_ratio
        feature_frame[f"win_loss_strength_{window}"] = rolling_positive_sum / (rolling_negative_sum + 1e-12)

        register(
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
            ],
            "baseline_v1",
        )
        register(
            [
                f"path_efficiency_{window}",
                f"path_consistency_{window}",
            ],
            "path_shape_v1",
        )
        if window in {5, 10}:
            register(
                [
                    f"positive_day_ratio_{window}",
                    f"win_loss_strength_{window}",
                ],
                "path_shape_v2",
            )

    for window in alpha_windows:
        feature_frame[f"open_volume_corr_{window}"] = _grouped_rolling_corr(feature_frame, "open", "volume", window)
        feature_frame[f"close_volume_corr_{window}"] = _grouped_rolling_corr(feature_frame, "close", "volume", window)
        feature_frame[f"vwap_volume_corr_{window}"] = _grouped_rolling_corr(feature_frame, "vwap_proxy", "volume", window)

    if 10 in windows:
        feature_frame["delta_price_to_ma_10_3"] = grouped["price_to_ma_10"].shift(0) - grouped["price_to_ma_10"].shift(3)
        feature_frame["squeeze_breakout_10"] = feature_frame["price_to_ma_10"] / (feature_frame["volatility_10"] + 1e-12)
        register(["squeeze_breakout_10"], "path_shape_v2")

    if 20 in windows:
        feature_frame["breakout_stability_20"] = feature_frame["high_breakout_20"] * feature_frame["path_efficiency_10"]
        feature_frame["recovery_quality_20"] = feature_frame["low_rebound_20"] * feature_frame["positive_day_ratio_5"]
        register(["breakout_stability_20", "recovery_quality_20"], "path_shape_v2")

    alpha_feature_columns: list[str] = []
    for window in alpha_windows:
        alpha_feature_columns.extend(
            [
                f"open_volume_corr_{window}",
                f"close_volume_corr_{window}",
                f"vwap_volume_corr_{window}",
            ]
        )
    if 10 in windows:
        alpha_feature_columns.append("delta_price_to_ma_10_3")
    register(alpha_feature_columns, "price_volume_alpha_v1")

    market_state = (
        feature_frame.groupby("trade_date", sort=False)
        .agg(
            market_return_mean_1=("close_return_1", "mean"),
            market_return_std_1=("close_return_1", "std"),
            market_intraday_mean=("intraday_return", "mean"),
            market_price_to_ma_20_mean=("price_to_ma_20", "mean"),
            market_volume_ratio_5_mean=("volume_ratio_5", "mean"),
            market_amount_ratio_5_mean=("amount_ratio_5", "mean") if "amount" in feature_frame.columns else ("volume_ratio_5", "mean"),
        )
        .reset_index()
    )
    market_state["market_up_ratio_1"] = feature_frame.groupby("trade_date", sort=False)["close_return_1"].apply(
        lambda series: float((series > 0.0).mean())
    ).values
    market_state["market_top_decile_return_1"] = feature_frame.groupby("trade_date", sort=False)["close_return_1"].apply(
        lambda series: float(series.nlargest(max(1, int(np.ceil(len(series) * 0.1)))).mean())
    ).values
    market_state["market_amount_top_share_1"] = feature_frame.groupby("trade_date", sort=False)["amount"].apply(
        lambda series: float(
            series.nlargest(max(1, int(np.ceil(len(series) * 0.1)))).sum() / (series.sum() + 1e-12)
        )
    ).values if "amount" in feature_frame.columns else 0.0
    market_state = market_state.sort_values("trade_date").reset_index(drop=True)
    market_state["market_breadth_thrust_5"] = market_state["market_up_ratio_1"].rolling(5, min_periods=1).mean()
    market_state["market_dispersion_trend_5"] = market_state["market_return_std_1"].rolling(5, min_periods=1).mean()
    feature_frame = feature_frame.merge(market_state, on="trade_date", how="left", sort=False)

    feature_frame["excess_return_1"] = feature_frame["close_return_1"] - feature_frame["market_return_mean_1"]
    feature_frame["excess_intraday_return"] = feature_frame["intraday_return"] - feature_frame["market_intraday_mean"]
    feature_frame["excess_price_to_ma_20"] = feature_frame["price_to_ma_20"] - feature_frame["market_price_to_ma_20_mean"]
    feature_frame["excess_volume_ratio_5"] = feature_frame["volume_ratio_5"] - feature_frame["market_volume_ratio_5_mean"]
    feature_frame["attention_gap_5"] = feature_frame["amount_ratio_5"] - feature_frame["market_amount_ratio_5_mean"]
    feature_frame["breadth_adjusted_return_5"] = feature_frame["close_return_5"] * feature_frame["market_breadth_thrust_5"]
    feature_frame["dispersion_adjusted_breakout_20"] = feature_frame["high_breakout_20"] / (
        feature_frame["market_dispersion_trend_5"] + 1e-12
    )
    feature_frame["leader_follow_through_5"] = feature_frame["close_return_5"] - feature_frame["market_top_decile_return_1"]
    feature_frame["amount_crowding_pressure_1"] = (
        feature_frame["amount_ratio_5"] * feature_frame["market_amount_top_share_1"]
    )
    register(
        [
            "market_return_mean_1",
            "market_return_std_1",
            "market_up_ratio_1",
            "market_top_decile_return_1",
            "market_breadth_thrust_5",
            "market_dispersion_trend_5",
            "market_amount_top_share_1",
            "excess_return_1",
            "excess_intraday_return",
            "excess_price_to_ma_20",
            "excess_volume_ratio_5",
            "attention_gap_5",
            "breadth_adjusted_return_5",
            "dispersion_adjusted_breakout_20",
            "leader_follow_through_5",
            "amount_crowding_pressure_1",
        ],
        "market_cross_v1",
    )

    cross_section_columns = [
        "close_return_1",
        "intraday_return",
        "price_to_ma_5",
        "price_to_ma_20",
        "volume_ratio_5",
        "volatility_10",
        "close_to_vwap",
        "delta_price_to_ma_10_3",
        "open_volume_corr_20",
        "path_efficiency_10",
        "path_consistency_10",
        "excess_return_1",
        "excess_price_to_ma_20",
        "excess_volume_ratio_5",
        "breadth_adjusted_return_5",
        "dispersion_adjusted_breakout_20",
        "attention_gap_5",
        "positive_day_ratio_10",
        "win_loss_strength_10",
        "squeeze_breakout_10",
        "breakout_stability_20",
        "recovery_quality_20",
    ]
    for column_name in cross_section_columns:
        if column_name in feature_frame.columns:
            rank_column = f"{column_name}_cs_rank"
            feature_frame[rank_column] = feature_frame.groupby("trade_date")[column_name].rank(pct=True)
            if column_name in feature_groups["baseline_v1"]:
                register([rank_column], "baseline_v1")
            elif column_name in feature_groups["price_volume_alpha_v1"]:
                register([rank_column], "price_volume_alpha_v1")
            elif column_name in feature_groups["path_shape_v1"]:
                register([rank_column], "path_shape_v1")
            elif column_name in feature_groups["path_shape_v2"]:
                register([rank_column], "path_shape_v2")
            else:
                register([rank_column], "market_cross_v1")

    if "turnover_rate" in feature_frame.columns:
        feature_frame["turnover_rate"] = feature_frame["turnover_rate"].fillna(0.0)
        register(["turnover_rate"], "baseline_v1")
        for window in windows:
            turnover_mean = grouped["turnover_rate"].transform(lambda series: series.rolling(window, min_periods=1).mean())
            ratio_column = f"turnover_ratio_{window}"
            feature_frame[ratio_column] = feature_frame["turnover_rate"] / (turnover_mean + 1e-12)
            register([ratio_column], "baseline_v1")
        feature_frame["turnover_rate_cs_rank"] = feature_frame.groupby("trade_date")["turnover_rate"].rank(pct=True)
        register(["turnover_rate_cs_rank"], "baseline_v1")

    if "amount" in feature_frame.columns:
        feature_frame["log_amount"] = np.log1p(feature_frame["amount"].clip(lower=0.0))
        register(["log_amount"], "baseline_v1")
        for window in windows:
            register([f"amount_ratio_{window}"], "baseline_v1")

    if "amplitude" in feature_frame.columns:
        feature_frame["amplitude"] = feature_frame["amplitude"].fillna(0.0)
        register(["amplitude"], "baseline_v1")
        feature_frame["amplitude_cs_rank"] = feature_frame.groupby("trade_date")["amplitude"].rank(pct=True)
        register(["amplitude_cs_rank"], "baseline_v1")

    if "change_amount" in feature_frame.columns:
        feature_frame["change_amount"] = feature_frame["change_amount"].fillna(0.0)
        register(["change_amount"], "baseline_v1")

    if "pct_chg" in feature_frame.columns:
        feature_frame["pct_chg"] = feature_frame["pct_chg"].fillna(0.0) / 100.0
        register(["pct_chg"], "baseline_v1")
        feature_frame["pct_chg_cs_rank"] = feature_frame.groupby("trade_date")["pct_chg"].rank(pct=True)
        register(["pct_chg_cs_rank"], "baseline_v1")

    feature_frame[feature_columns] = (
        feature_frame[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return feature_frame, feature_columns, feature_groups


def _clip_future_return_by_date(df: pd.DataFrame, clip_quantile: float) -> pd.Series:
    if clip_quantile <= 0.0:
        return df["future_return"]
    clipped = df.groupby("trade_date", group_keys=False)["future_return"].transform(
        lambda series: series.clip(
            lower=series.quantile(clip_quantile),
            upper=series.quantile(1.0 - clip_quantile),
        )
    )
    return clipped


def _build_relevance_labels(df: pd.DataFrame, *, bucket_count: int, clip_quantile: float = 0.0) -> pd.Series:
    target = _clip_future_return_by_date(df, clip_quantile=clip_quantile)
    percentile = target.groupby(df["trade_date"]).rank(method="first", ascending=False, pct=True)
    labels = ((1.0 - percentile) * float(bucket_count)).clip(
        lower=0.0,
        upper=float(bucket_count) - 1e-6,
    ).astype(int)
    return labels


def build_sample_weights(
    df: pd.DataFrame,
    *,
    head_quantile: float,
    head_weight: float,
) -> pd.Series:
    weights = pd.Series(1.0, index=df.index, dtype=float)
    if head_quantile <= 0.0 or head_weight <= 1.0:
        return weights
    ranks = df.groupby("trade_date")["future_return"].rank(method="first", ascending=False, pct=True)
    weights.loc[ranks <= head_quantile] = float(head_weight)
    return weights


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


def build_walk_forward_splits(
    ranking_frame: pd.DataFrame,
    *,
    fold_count: int,
    validation_ratio: float,
    min_train_groups: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    unique_dates = sorted(ranking_frame["trade_date"].drop_duplicates().tolist())
    total_groups = len(unique_dates)
    if total_groups < min_train_groups + 1:
        raise ValueError(
            "可用于 walk-forward 的交易日数量不足，无法构建多折验证。"
            f" 当前仅有 {total_groups} 个交易日，至少需要 {min_train_groups + 1} 个。"
        )

    validation_group_count = max(1, int(round(total_groups * validation_ratio)))
    max_fold_count = max(1, (total_groups - min_train_groups) // validation_group_count)
    actual_fold_count = min(fold_count, max_fold_count)

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_idx in range(actual_fold_count):
        validation_end = total_groups - (actual_fold_count - fold_idx - 1) * validation_group_count
        validation_start = max(min_train_groups, validation_end - validation_group_count)
        validation_dates = set(unique_dates[validation_start:validation_end])
        train_dates = unique_dates[:validation_start]
        if len(train_dates) < min_train_groups or not validation_dates:
            continue

        train_frame = ranking_frame[ranking_frame["trade_date"].isin(train_dates)].copy()
        validation_frame = ranking_frame[ranking_frame["trade_date"].isin(validation_dates)].copy()
        if train_frame.empty or validation_frame.empty:
            continue
        splits.append((train_frame, validation_frame))

    if not splits:
        raise ValueError("未能构建有效的 walk-forward 切分，请检查交易日数量与参数设置。")
    return splits


def build_training_bundle(
    raw_df: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    feature_preset: str,
    future_buy_offset: int,
    future_sell_offset: int,
    validation_ratio: float,
    min_train_groups: int,
    label_bucket_count: int,
    label_clip_quantile: float = 0.0,
    head_weight_quantile: float = 0.0,
    head_weight_value: float = 1.0,
) -> RankingDatasetBundle:
    market_df = standardize_market_dataframe(raw_df)
    min_rows_per_stock = int(market_df.groupby("stock_id").size().min()) if not market_df.empty else 0
    unique_dates = int(market_df["trade_date"].nunique()) if not market_df.empty else 0
    feature_frame, all_feature_columns, feature_groups = engineer_features(market_df, windows=windows)
    feature_columns, resolved_feature_preset = select_feature_columns(feature_groups, feature_preset)

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
        clip_quantile=label_clip_quantile,
    )
    ranking_frame["sample_weight"] = build_sample_weights(
        ranking_frame,
        head_quantile=head_weight_quantile,
        head_weight=head_weight_value,
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
        feature_groups=feature_groups,
        feature_preset=resolved_feature_preset,
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
    feature_frame, _, _ = engineer_features(market_df, windows=windows)

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


def safe_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    std = float(np.nanstd(array))
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(array, dtype=float)
    mean = float(np.nanmean(array))
    normalized = (array - mean) / std
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def rank_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    if np.allclose(array, array[0], equal_nan=True):
        return np.zeros_like(array, dtype=float)
    ranked = rankdata(array, method="average")
    centered = ranked / max(len(array), 1)
    centered = centered - float(np.mean(centered))
    centered[~np.isfinite(centered)] = 0.0
    return centered


def combine_model_scores(score_map: dict[str, np.ndarray], method: str, weights: dict[str, float] | None = None) -> np.ndarray:
    if not score_map:
        raise ValueError("没有可用于集成的模型分数。")

    ordered_items = [(name, np.asarray(scores, dtype=float)) for name, scores in score_map.items()]
    lengths = {scores.shape[0] for _, scores in ordered_items}
    if len(lengths) != 1:
        raise ValueError("集成模型的分数长度不一致，无法组合。")

    if method not in {"zscore_average_equal", "rank_average_equal", "validation_weighted_zscore_average"}:
        raise ValueError(f"不支持的集成方法：{method}")

    if method == "rank_average_equal":
        transformed = [rank_normalize(scores) for _, scores in ordered_items]
        return np.mean(np.vstack(transformed), axis=0)

    transformed_map = {name: safe_zscore(scores) for name, scores in ordered_items}
    if method == "zscore_average_equal":
        return np.mean(np.vstack(list(transformed_map.values())), axis=0)

    if weights is None:
        raise ValueError("validation_weighted_zscore_average 需要提供权重。")
    valid_weights = np.array([float(weights.get(name, 0.0)) for name, _ in ordered_items], dtype=float)
    valid_weights = np.clip(valid_weights, a_min=0.0, a_max=None)
    if not np.isfinite(valid_weights).all() or float(valid_weights.sum()) <= 1e-12:
        valid_weights = np.full(len(ordered_items), 1.0 / len(ordered_items), dtype=float)
    else:
        valid_weights = valid_weights / valid_weights.sum()
    stacked = np.vstack([transformed_map[name] for name, _ in ordered_items])
    return np.average(stacked, axis=0, weights=valid_weights)


def derive_validation_weights(model_metrics: dict[str, dict[str, float]], metric_name: str) -> dict[str, float]:
    raw_weights: dict[str, float] = {}
    for model_name, metrics in model_metrics.items():
        metric_value = float(metrics.get(metric_name, 0.0))
        raw_weights[model_name] = max(0.0, metric_value)

    total = sum(raw_weights.values())
    if total <= 1e-12:
        equal_weight = 1.0 / max(len(raw_weights), 1)
        return {model_name: equal_weight for model_name in raw_weights}
    return {model_name: weight / total for model_name, weight in raw_weights.items()}


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
