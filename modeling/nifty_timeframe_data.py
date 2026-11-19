"""Causal stock/timeframe features and training-period factor selection."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol
from features.intraday_features import add_intraday_features
from features.swing_features import add_swing_features
from features.technical_indicators import add_price_features, add_technical_indicators
from modeling.nifty_preparation import _aggregate_sessions, available_candles
from modeling.nifty_price_training import PriceDataset, _array_hash
from modeling.nifty_same_candle_correlations import (
    _coefficients, _daily_external, _global_candles, _read_candles,
)
from modeling.nifty_price_workflow import _read_global_part
from modeling.nifty_universe import COVERAGE_END, Membership
from strategies import box_breakout, mean_reversion, momentum, trend_following


class UnusableTimeframe(ValueError):
    """A stock/timeframe has insufficient eligible history."""


HISTORY_START = date(2022, 1, 1)


def training_universe(membership: tuple[Membership, ...]) -> list[str]:
    """Union of members overlapping the requested window within verified coverage."""
    return sorted({item.symbol for item in membership if item.valid_from <= COVERAGE_END
                   and (item.valid_to is None or item.valid_to > HISTORY_START)})


def training_history(symbol: str, membership: tuple[Membership, ...]) -> tuple[Membership, ...]:
    if symbol not in training_universe(membership):
        raise ValueError(f"{symbol} was not a verified Nifty member during 2022-2026")
    # This is an explicit research-universe policy, not point-in-time index membership.
    return (Membership(symbol, HISTORY_START, None, "research_universe_full_history", ""),)


def select_factors(rows: list[dict], *, minimum: int = 60) -> dict:
    if minimum < 3:
        raise ValueError("At least three paired returns are required")
    eligible = [row for row in rows if row["return_status"] == "scored"
                and row["return_pairs"] >= minimum]
    for row in eligible:
        values = [row["return_pearson"], row["return_spearman"]]
        if not all(np.isfinite(value) and abs(value) <= 1.00000001 for value in values):
            raise ValueError("Invalid correlation coefficients")
    for step in range(31):
        pearson = (70 - step) / 100
        spearman = max(40, 70 - 5 * step) / 100
        chosen = [row for row in eligible
                  if abs(row["return_pearson"]) + 1e-12 >= pearson
                  and abs(row["return_spearman"]) + 1e-12 >= spearman]
        if chosen:
            return {"pearson_threshold": pearson, "spearman_threshold": spearman,
                    "selected": sorted(chosen, key=lambda row: (row["source"], row["factor"]))}
    return {"pearson_threshold": 0.4, "spearman_threshold": 0.4, "selected": []}


def read_stock(root: Path, symbol: str, timeframe: str, membership=None) -> pd.DataFrame | None:
    group = "minute" if timeframe in INTRADAY_TIMEFRAMES else "day"
    folder = root / group / _safe_symbol(symbol)
    path = folder / f"{timeframe}.csv.gz"
    daily = folder / "1d.csv.gz"
    if not path.is_file() or (timeframe in {"2d", "trading_week"} and not daily.is_file()):
        return None
    raw = pd.read_csv(path)
    groups = _aggregate_sessions(daily, timeframe) if timeframe in {"2d", "trading_week"} else None
    return available_candles(raw, timeframe, groups, membership, daily_source=timeframe == "1d")


def strategy_features(candles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Use the existing indicators and four executable strategy families."""
    data = candles.loc[~candles["is_partial"]].reset_index(drop=True).copy()
    frame = pl.from_pandas(data[["timestamp", "open", "high", "low", "close", "volume"]]).lazy()
    frame = frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Asia/Kolkata").dt.date().alias("_session_date"))
    frame = add_technical_indicators(add_price_features(frame))
    intraday = timeframe in INTRADAY_TIMEFRAMES
    frame = add_intraday_features(frame) if intraday else add_swing_features(frame)
    style = "intraday" if intraday else "swing"
    for name, strategy in (("box_breakout", box_breakout), ("momentum", momentum),
                           ("trend_following", trend_following), ("mean_reversion", mean_reversion)):
        long, short, _ = strategy.conditions(style)
        frame = frame.with_columns(
            (long.fill_null(False).cast(pl.Int8) - short.fill_null(False).cast(pl.Int8))
            .alias("strategy_" + name))
    result = frame.collect().to_pandas()
    names = ["pct_change", "ret_5", "ret_20", "candle_body_pct", "candle_range_pct",
             "close_location", "gap_pct", "rsi_7", "rsi_14", "adx_14", "atr_pct",
             "bb_width", "bb_pct_b", "stoch_k_14", "stoch_d_3", "relative_volume",
             "box_width_atr", "box_breakout_distance_up_atr", "box_breakout_distance_down_atr",
             "strategy_box_breakout", "strategy_momentum", "strategy_trend_following",
             "strategy_mean_reversion"]
    for period in (9, 20, 50, 200):
        name = f"ema_{period}_distance"
        result[name] = result["close"] / result[f"ema_{period}"] - 1
        names.append(name)
    for name in ("macd", "macd_hist"):
        result[name + "_pct"] = result[name] / result["close"]
        names.append(name + "_pct")
    output = data[["timestamp", "available_at", "close"]].copy()
    output[names] = result[names].replace([np.inf, -np.inf], np.nan)
    return output


def attach_factor(features: pd.DataFrame, factor: pd.DataFrame, name: str,
                  timeframe: str) -> pd.DataFrame:
    if factor.empty:
        raise ValueError(f"Selected factor has no observations: {name}")
    if factor["available_at"].duplicated().any():
        raise ValueError(f"Duplicate factor availability: {name}")
    tolerance = (timedelta(minutes=INTRADAY_TIMEFRAMES[timeframe]) if timeframe in INTRADAY_TIMEFRAMES
                 else timedelta(days={"1d": 7, "2d": 10, "trading_week": 14}[timeframe]))
    right = factor.rename(columns={"available_at": "factor_available_at", "factor_return": name})
    result = pd.merge_asof(features.sort_values("available_at"), right.sort_values("factor_available_at"),
                           left_on="available_at", right_on="factor_available_at",
                           direction="backward", tolerance=tolerance)
    if (result["factor_available_at"] > result["available_at"]).any():
        raise ValueError("A factor observation is available after its decision")
    result[name + ":missing"] = result[name].isna().astype(float)
    return result.drop(columns="factor_available_at")


def training_observations(observations: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    if observations.empty:
        return observations
    if pd.api.types.is_integer_dtype(observations["end"]):
        boundary = pd.Timestamp(cutoff, tz="Asia/Kolkata").tz_convert("UTC").value
        valid = observations["end"] < boundary
    else:
        valid = observations["end"] < cutoff
    if "available_at" in observations:
        boundary_time = pd.Timestamp(cutoff, tz="Asia/Kolkata").tz_convert("UTC")
        valid &= observations["available_at"] < boundary_time
    return observations.loc[valid].copy()


def add_availability(observations: pd.DataFrame, candles: pd.DataFrame, timeframe: str,
                     timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    timestamps = pd.to_datetime(candles["timestamp"], utc=True)
    starts = (timestamps.dt.as_unit("ns").astype("int64") if timeframe in INTRADAY_TIMEFRAMES
              else timestamps.dt.tz_convert(timezone).dt.date)
    availability = pd.DataFrame({"start": starts,
                                 "available_at": pd.to_datetime(candles["available_at"], utc=True)})
    return observations.merge(availability, on="start", validate="one_to_one")


class FactorStore:
    def __init__(self, config: dict, reference: list[date]) -> None:
        self.stocks = Path(config["stocks"])
        self.reference = reference
        self.global_roots: dict[tuple[str, str], Path] = {}
        self.external, _ = _daily_external(
            Path(config["india"]), Path(config["treasury"]), Path(config["prices"]))
        self.local = sorted(name for name in config["local_factors"]
                            if (self.stocks / "day" / name / "1d.csv.gz").is_file())
        for directory in config["global_roots"]:
            root = Path(directory)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            if summary["status"] != "complete":
                raise ValueError(f"Incomplete global collection: {root}")
            for item in summary["factors"]:
                key = ("global:" + summary["source"], item["symbol"])
                if key in self.global_roots:
                    old = self.global_roots[key]
                    for timeframe in config["timeframes"]:
                        if (old / "prepared" / key[1] / f"{timeframe}.parquet").read_bytes() != (
                            root / "prepared" / key[1] / f"{timeframe}.parquet").read_bytes():
                            raise ValueError(f"Conflicting duplicate global factor: {key}")
                else:
                    self.global_roots[key] = root

    def keys(self) -> list[tuple[str, str]]:
        return sorted([*(("local", name) for name in self.local),
                       *self.global_roots, *self.external])

    def load(self, key: tuple[str, str], timeframe: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        source, symbol = key
        if source == "local":
            observations = _read_candles(self.stocks, symbol, timeframe, self.reference)
            candles = read_stock(self.stocks, symbol, timeframe)
            timezone = "Asia/Kolkata"
        elif key in self.global_roots:
            root = self.global_roots[key]
            observations = _global_candles(root, symbol, timeframe)
            if observations is None:
                return None, None
            candles = _read_global_part(root, symbol, timeframe)
            timezone = "America/New_York"
        else:
            if timeframe != "1d":
                return None, None
            observations = self.external[key].copy()
            # Source release timestamps are not verified; impose a next-day publication lag.
            observations["available_at"] = (
                pd.to_datetime(observations["end"].astype(str)).dt.tz_localize("America/New_York")
                + timedelta(days=2)).dt.tz_convert("UTC")
            return observations, observations[["available_at", "return"]].rename(
                columns={"return": "factor_return"}).dropna()
        if observations is None or candles is None:
            return None, None
        joined = add_availability(observations, candles, timeframe, timezone)
        return joined, joined[["available_at", "return"]].rename(
            columns={"return": "factor_return"}).dropna()


def chronological_boundaries(examples: pd.DataFrame) -> tuple[date, date]:
    days = sorted(set(examples["session"]))
    if len(days) < 10:
        raise UnusableTimeframe("Fewer than ten eligible sessions for chronological splits")
    return days[int(len(days) * 0.70)], days[int(len(days) * 0.85)]


def make_dataset(frame: pd.DataFrame, names: list[str], symbol: str, timeframe: str,
                 validation: date, holdout: date, sequence_length: int, selection: dict) -> PriceDataset:
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive")
    count = len(frame)
    features = frame[names].to_numpy(dtype=np.float32)
    targets = frame["target_log_return"].to_numpy(dtype=np.float32)
    if np.isinf(features).any() or not np.isfinite(targets).all():
        raise ValueError("Invalid features or targets")
    sequences = np.full((count, sequence_length), -1, dtype=np.int64)
    contiguous = frame["timestamp"].eq(frame["target_timestamp"].shift()).to_numpy()
    # Reset recurrent history at missing/partial candles and at intraday session boundaries.
    run_start = 0
    splits: list[list[int]] = [[], [], []]
    validation_time = pd.Timestamp(validation, tz="Asia/Kolkata").tz_convert("UTC")
    holdout_time = pd.Timestamp(holdout, tz="Asia/Kolkata").tz_convert("UTC")
    for index in range(count):
        if index == 0 or not contiguous[index]:
            run_start = index
        if index - run_start + 1 < sequence_length:
            continue
        sequences[index] = np.arange(index - sequence_length + 1, index + 1)
        decision = frame["available_at"].iloc[index]
        label = frame["label_available_at"].iloc[index]
        if decision < validation_time and label < validation_time:
            splits[0].append(index)
        elif validation_time <= decision < holdout_time and label < holdout_time:
            splits[1].append(index)
        elif decision >= holdout_time:
            splits[2].append(index)
    if any(not part for part in splits):
        raise UnusableTimeframe("Train, validation and holdout each require complete sequences")
    anchors = frame["anchor_close"].to_numpy(dtype=np.float64)
    closes = frame["target_close"].to_numpy(dtype=np.float64)
    sessions = np.asarray(frame["session"].tolist(), dtype="datetime64[D]")
    target_sessions = np.asarray(frame["target_session"].tolist(), dtype="datetime64[D]")
    codes = np.zeros(count, dtype=np.int32)
    indices = [np.asarray(part, dtype=np.int64) for part in splits]
    content = _array_hash((features, targets, sequences, anchors, closes, sessions, target_sessions,
                           codes, *indices))
    provenance = json.dumps({"symbol": symbol, "timeframe": timeframe, "selection": selection,
                             "validation_start": validation.isoformat(), "holdout_start": holdout.isoformat(),
                             "trend_definition": "sign of predicted next-candle log return"},
                            sort_keys=True, allow_nan=False)
    digest = hashlib.sha256((content + provenance + json.dumps(names)).encode()).hexdigest()
    return PriceDataset(features, targets, sequences, anchors, closes, sessions, target_sessions, codes,
                        *indices, tuple(names), (symbol,), digest, content, provenance)
