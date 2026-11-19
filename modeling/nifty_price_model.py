"""Causal daily-price examples and untrained hybrid regression components."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.nifty_preparation import TIMEFRAMES, join_available
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, Membership


class NoPriceExamples(ValueError):
    """No eligible next-session examples exist for this stock and window."""


def load_prepared_candles(root: Path, symbol: str, timeframe: str, *, max_rows: int) -> pd.DataFrame | None:
    """Load one bounded, checksum-verified stock/timeframe preparation job."""
    if timeframe not in TIMEFRAMES or max_rows < 1:
        raise ValueError("Invalid timeframe or maximum row count")
    summary = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    jobs = [job for job in summary["jobs"] if job["kind"] == "stock"
            and job["symbol"] == symbol and job["timeframe"] == timeframe]
    if len(jobs) != 1:
        raise ValueError(f"Expected exactly one preparation job for {symbol} {timeframe}")
    job = jobs[0]
    if job["status"] in {"missing_input", "no_eligible_rows"}:
        return None
    if job["status"] != "prepared" or job["rows"] > max_rows:
        raise ValueError(f"Unavailable or oversized preparation job for {symbol} {timeframe}")
    folder = root / "stock" / _safe_symbol(symbol) / timeframe
    parts = []
    total = 0
    for index in range(job["chunks"]):
        target = folder / f"part-{index:06d}.parquet"
        checkpoint = target.with_suffix(".json")
        meta = json.loads(checkpoint.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != meta["sha256"]:
            raise ValueError(f"Corrupt prepared candle checkpoint: {target}")
        parquet = pq.ParquetFile(target)
        if parquet.metadata.num_rows != meta["rows"]:
            raise ValueError(f"Corrupt prepared candle checkpoint: {target}")
        total += meta["rows"]
        if total > max_rows:
            raise ValueError(f"Prepared candle rows exceed the limit for {symbol} {timeframe}")
        parts.append(pd.read_parquet(target))
    if total != job["rows"] or not parts:
        raise ValueError(f"Prepared candle row count mismatch for {symbol} {timeframe}")
    return pd.concat(parts, ignore_index=True)


def _validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "available_at", "open", "high", "low", "close", "volume", "is_partial"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing candle columns: {sorted(required - set(frame.columns))}")
    result = frame.loc[:, sorted(required)].copy().reset_index(drop=True)
    for column in ("timestamp", "available_at"):
        if not isinstance(result[column].dtype, pd.DatetimeTZDtype) or result[column].isna().any():
            raise ValueError(f"Candle {column} must be timezone-aware and non-null")
        result[column] = result[column].dt.tz_convert("UTC")
    if (result["timestamp"].duplicated().any() or result["available_at"].duplicated().any()
            or not result["timestamp"].is_monotonic_increasing
            or not result["available_at"].is_monotonic_increasing
            or (result["available_at"] <= result["timestamp"]).any()):
        raise ValueError("Candle timestamps must be ordered, unique and causally available")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    prices = result[["open", "high", "low", "close"]]
    if (not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
            or not np.isfinite(result["volume"]).all() or (result["volume"] < 0).any()
            or (result["high"] < prices[["open", "low", "close"]].max(axis=1)).any()
            or (result["low"] > prices[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Invalid candle OHLCV values")
    if result["is_partial"].isna().any() or not result["is_partial"].isin([True, False]).all():
        raise ValueError("Invalid candle partial flags")
    return result


def selected_factors(selection: dict, symbol: str, first_session: date) -> list[tuple[str, str]]:
    """Require a selection fixed strictly before every example in this batch."""
    if selection.get("status") != "research_only" or selection.get("threshold_absolute") != 0.7:
        raise ValueError("Unsupported factor selection")
    cutoff = date.fromisoformat(selection["train_end_exclusive"])
    if cutoff > first_session:
        raise ValueError("Factor selection cutoff is later than the first example session")
    stocks = [row for row in selection["stocks"] if row["symbol"] == symbol]
    if len(stocks) != 1:
        raise ValueError(f"Selection must contain exactly one entry for {symbol}")
    stock = stocks[0]
    if stock["status"] in {"insufficient_observations", "missing_stock_data"}:
        raise ValueError(f"Cannot use factor selection for {symbol}: {stock['status']}")
    if stock["status"] not in {"selected", "no_qualifying_factor"}:
        raise ValueError(f"Invalid factor selection status for {symbol}")
    selected = []
    for item in stock["selected"]:
        correlation = float(item["correlation"])
        if not np.isfinite(correlation) or not 0.7 <= abs(correlation) <= 1.000000001:
            raise ValueError("Selected factor does not meet the absolute correlation threshold")
        selected.append((item["source"], item["factor"]))
    if len(selected) != len(set(selected)) or bool(selected) != (stock["status"] == "selected"):
        raise ValueError("Inconsistent selected factors")
    selected.sort()
    return selected


def build_examples(
    symbol: str,
    daily: pd.DataFrame,
    candles: dict[str, pd.DataFrame | None],
    factor_returns: dict[tuple[str, str], pd.DataFrame],
    selection: dict,
    reference_sessions: list[date],
    membership: tuple[Membership, ...],
    *,
    start: date,
    end: date,
    max_candle_age: dict[str, timedelta],
    max_factor_age: timedelta,
    allow_no_factors: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build next-reference-session log-return labels and strictly as-of features."""
    if not COVERAGE_START <= start < end <= COVERAGE_END:
        raise ValueError("Example window is outside verified membership coverage")
    if (len(reference_sessions) != len(set(reference_sessions))
            or reference_sessions != sorted(reference_sessions)):
        raise ValueError("Reference sessions must be sorted and unique")
    if any(entry.symbol != symbol for entry in membership):
        raise ValueError("Membership intervals do not match the stock")
    daily = _validate_candles(daily)
    days = daily["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    if days.duplicated().any() or not days.is_monotonic_increasing:
        raise ValueError("Daily target sessions must be sorted and unique")
    daily = daily.assign(session=days).set_index("session", drop=False)
    rows = []
    for session, target in zip(reference_sessions, reference_sessions[1:]):
        if not start <= session < end or target >= end or session not in daily.index or target not in daily.index:
            continue
        if not any(item.contains(session) and item.contains(target) for item in membership):
            continue
        current, future = daily.loc[session], daily.loc[target]
        if current["is_partial"] or future["is_partial"]:
            continue
        decision = pd.Timestamp(session).tz_localize("Asia/Kolkata") + timedelta(days=1)
        decision = decision.tz_convert("UTC")
        if current["available_at"] > decision or future["available_at"] <= decision:
            raise ValueError("Daily close availability is incompatible with decision and target")
        rows.append((session, target, decision, future["available_at"], float(current["close"]),
                     float(future["close"]), float(np.log(future["close"] / current["close"]))))
    result = pd.DataFrame(rows, columns=["session", "target_session", "available_at",
                                         "label_available_at", "anchor_close", "target_close", "target_log_return"])
    if result.empty:
        raise NoPriceExamples(f"No eligible consecutive daily examples for {symbol}")
    return _assemble_features(symbol, result, candles, factor_returns, selection, membership,
                              max_candle_age=max_candle_age, max_factor_age=max_factor_age,
                              allow_no_factors=allow_no_factors)


def build_forecast_features(
    symbol: str, daily: pd.DataFrame, candles: dict[str, pd.DataFrame | None],
    factor_returns: dict[tuple[str, str], pd.DataFrame], selection: dict,
    membership: tuple[Membership, ...], *, session: date, target_session: date,
    max_candle_age: dict[str, timedelta], max_factor_age: timedelta,
    allow_no_factors: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build features without knowing the target close; caller supplies the next exchange session."""
    if not COVERAGE_START <= session < target_session <= COVERAGE_END:
        raise ValueError("Forecast dates are outside verified membership coverage")
    if any(item.symbol != symbol for item in membership):
        raise ValueError("Membership intervals do not match the stock")
    if not any(item.contains(session) and item.contains(target_session) for item in membership):
        raise ValueError("Forecast session and target must share a membership interval")
    daily = _validate_candles(daily)
    days = daily["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    if days.duplicated().any() or not days.is_monotonic_increasing:
        raise ValueError("Daily target sessions must be sorted and unique")
    matches = daily.loc[days == session]
    if len(matches) != 1 or bool(matches.iloc[0]["is_partial"]):
        raise ValueError("Forecast requires one complete daily candle for the decision session")
    current = matches.iloc[0]
    decision = (pd.Timestamp(session).tz_localize("Asia/Kolkata") + timedelta(days=1)).tz_convert("UTC")
    if current["available_at"] > decision:
        raise ValueError("Daily close is not yet available at the decision time")
    result = pd.DataFrame([(session, target_session, decision, float(current["close"]))],
                          columns=["session", "target_session", "available_at", "anchor_close"])
    return _assemble_features(symbol, result, candles, factor_returns, selection, membership,
                              max_candle_age=max_candle_age, max_factor_age=max_factor_age,
                              allow_no_factors=allow_no_factors)


def _assemble_features(
    symbol: str, result: pd.DataFrame, candles: dict[str, pd.DataFrame | None],
    factor_returns: dict[tuple[str, str], pd.DataFrame], selection: dict,
    membership: tuple[Membership, ...], *, max_candle_age: dict[str, timedelta],
    max_factor_age: timedelta, allow_no_factors: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if set(candles) != set(TIMEFRAMES) or set(max_candle_age) != set(TIMEFRAMES):
        raise ValueError("All nine candle timeframes and their maximum ages are required")
    if any(age <= timedelta(0) for age in max_candle_age.values()) or max_factor_age <= timedelta(0):
        raise ValueError("Maximum feature ages must be positive")
    chosen = selected_factors(selection, symbol, min(result["session"]))
    if not chosen and not allow_no_factors:
        raise ValueError(f"No qualifying factors for {symbol}; stock-only research requires explicit opt-in")
    if set(factor_returns) != set(chosen):
        raise ValueError("Factor inputs must exactly match the point-in-time selection")
    features = []
    missing = []
    for timeframe in TIMEFRAMES:
        names = [f"{timeframe}_{suffix}" for suffix in ("close_ratio", "body", "range", "partial")]
        features.extend(names)
        frame = candles[timeframe]
        if frame is None:
            missing.append(timeframe)
            for name in names:
                result[name] = np.nan
            continue
        frame = _validate_candles(frame)
        derived = frame[["available_at", "timestamp"]].rename(columns={"timestamp": "_candle_start"}).copy()
        derived[names[0]] = frame["close"]
        derived[names[1]] = frame["close"] / frame["open"] - 1
        derived[names[2]] = (frame["high"] - frame["low"]) / frame["close"]
        derived[names[3]] = frame["is_partial"].astype(float)
        result = join_available(result, derived, tolerance=max_candle_age[timeframe])
        candle_days = result["_candle_start"].dt.tz_convert("Asia/Kolkata").dt.date
        eligible = pd.Series(False, index=result.index)
        for item in membership:
            eligible |= (candle_days >= item.valid_from) & (
                True if item.valid_to is None else result["session"] < item.valid_to
            )
        result.loc[~eligible, names] = np.nan
        result[names[0]] = result[names[0]] / result["anchor_close"] - 1
        result = result.drop(columns=["factor_available_at", "_candle_start"])
    for index, key in enumerate(chosen):
        frame = factor_returns[key]
        if set(frame.columns) != {"factor_available_at", "factor_return"}:
            raise ValueError(f"Invalid factor-return columns for {key}")
        frame = frame.rename(columns={"factor_available_at": "available_at"}).copy()
        values = pd.to_numeric(frame["factor_return"], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid factor returns for {key}")
        frame["factor_return"] = values
        feature = f"factor_{index}_return"
        frame = frame.rename(columns={"factor_return": feature})
        result = join_available(result, frame, tolerance=max_factor_age)
        if key[0] == "local":
            result.loc[result["factor_available_at"] != result["available_at"], feature] = np.nan
        result = result.drop(columns="factor_available_at")
        features.append(feature)
    return result, features, missing


def build_price_tree(*, device: str = "cpu") -> Any:
    """Build an untrained XGBoost log-return regressor."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("XGBoost device must be cpu or cuda")
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("XGBoost is required; install requirements-training.txt") from exc
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", device=device)


def build_price_lstm(torch: Any, feature_count: int) -> Any:
    """Build an untrained next-session log-return regressor."""
    if feature_count < 1:
        raise ValueError("At least one feature is required")

    class PriceLSTM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = torch.nn.LSTM(feature_count, 32, num_layers=2,
                                           batch_first=True, dropout=0.2)
            self.head = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                            torch.nn.Dropout(0.2), torch.nn.Linear(32, 1))

        def forward(self, values: Any) -> Any:
            output, _ = self.recurrent(values)
            return self.head(output[:, -1, :]).squeeze(-1)

    return PriceLSTM()


def combine_price_predictions(
    anchor_close: np.ndarray, tree_log_returns: np.ndarray, lstm_log_returns: np.ndarray, *,
    tree_weight: float,
) -> np.ndarray:
    """Combine fitted regressors' log-return outputs into positive price forecasts."""
    anchor = np.asarray(anchor_close, dtype=np.float64)
    tree = np.asarray(tree_log_returns, dtype=np.float64)
    lstm = np.asarray(lstm_log_returns, dtype=np.float64)
    if (not 0 <= tree_weight <= 1 or anchor.ndim != 1 or tree.shape != anchor.shape
            or lstm.shape != anchor.shape or not np.isfinite(anchor).all()
            or (anchor <= 0).any() or not np.isfinite(tree).all() or not np.isfinite(lstm).all()):
        raise ValueError("Invalid hybrid price prediction inputs")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        predicted = anchor * np.exp(tree_weight * tree + (1 - tree_weight) * lstm)
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Hybrid price forecasts must be finite and positive")
    return predicted


def predict_hybrid_prices(
    tree: Any, network: Any, torch: Any, tabular: np.ndarray, sequences: np.ndarray,
    anchor_close: np.ndarray, *, tree_weight: float, device: str = "cpu",
) -> np.ndarray:
    """Predict from fitted tree and normalized LSTM sequence batches."""
    tabular = np.asarray(tabular, dtype=np.float32)
    sequences = np.asarray(sequences, dtype=np.float32)
    if (tabular.ndim != 2 or sequences.ndim != 3 or len(tabular) == 0
            or len(tabular) != len(sequences) or tabular.shape[1] != sequences.shape[2]
            or sequences.shape[1] == 0 or np.isinf(tabular).any()
            or not np.isfinite(sequences).all()):
        raise ValueError("Invalid tabular or normalized sequence batch")
    tree_returns = np.asarray(tree.predict(tabular))
    was_training = network.training
    try:
        network.eval()
        with torch.inference_mode():
            lstm_returns = network(torch.as_tensor(sequences, device=device)).cpu().numpy()
    finally:
        network.train(was_training)
    return combine_price_predictions(anchor_close, tree_returns, lstm_returns, tree_weight=tree_weight)
