from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from backtesting.execution_simulator import ExecutionCosts, resolve_exit, session_date
from features.feature_engineering import build_features
from strategies import generate_strategy_signals

ModelStyle = Literal["intraday", "swing"]
OutcomeLabel = Literal[0, 1, 2]
_NON_FEATURE_COLUMNS = {
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
    "source_bars",
    "trading_days",
}


@dataclass(frozen=True)
class TrainingDataset:
    features: np.ndarray
    sequences: np.ndarray
    labels: np.ndarray
    net_returns_r: np.ndarray
    timestamps: tuple[datetime, ...]
    label_end_timestamps: tuple[datetime, ...]
    feature_names: tuple[str, ...]
    style: ModelStyle
    strategy: str
    horizon: int
    sequence_length: int


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_start: datetime
    test_start: datetime


def impute_and_normalize_sequences(
    sequences: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Impute missing historical features with training means, then scale."""
    broadcast_mean = mean[None, None, :]
    finite_sequences = np.where(np.isfinite(sequences), sequences, broadcast_mean)
    safe_scale = np.where(scale < 1e-8, 1.0, scale)
    return ((finite_sequences - broadcast_mean) / safe_scale[None, None, :]).astype(
        np.float32
    )


def _numeric_feature_names(frame: pl.DataFrame) -> tuple[str, ...]:
    names = tuple(
        name
        for name, dtype in frame.schema.items()
        if name not in _NON_FEATURE_COLUMNS
        and not name.startswith("_")
        and dtype.is_numeric()
    )
    if not names:
        raise ValueError("No numeric engineered features are available for training")
    return names


def _outcome_label(net_pnl: float, risk_per_unit: float) -> OutcomeLabel:
    net_r = net_pnl / risk_per_unit
    if net_r <= -0.1:
        return 0
    if net_r >= 0.1:
        return 2
    return 1


def _label_candidate(
    *,
    bars: list[dict[str, object]],
    signal_index: int,
    signal: dict[str, object],
    style: ModelStyle,
    horizon: int,
    costs: ExecutionCosts,
) -> tuple[OutcomeLabel, datetime, float] | None:
    entry_index = signal_index + 1
    if entry_index >= len(bars):
        return None
    signal_timestamp = signal["timestamp"]
    entry_bar = bars[entry_index]
    entry_timestamp = entry_bar["timestamp"]
    if style == "intraday" and session_date(entry_timestamp) != session_date(signal_timestamp):
        return None

    direction = 1 if signal["direction"] == "long" else -1
    entry_raw = float(entry_bar["open"])
    entry_price = costs.entry_fill(entry_raw, direction)
    stop_distance = abs(float(signal["entry_price"]) - float(signal["stop_price"]))
    if stop_distance <= 0:
        return None
    stop_price = entry_price - direction * stop_distance
    target_price = entry_price + direction * stop_distance * float(signal["reward_risk"])
    end_index = min(len(bars), entry_index + horizon)

    exit_raw = None
    exit_timestamp = None
    last_eligible_index = entry_index
    for index in range(entry_index + 1, end_index):
        bar = bars[index]
        if style == "intraday" and session_date(bar["timestamp"]) != session_date(entry_timestamp):
            break
        last_eligible_index = index
        exit_fill = resolve_exit(
            direction=direction,
            open_price=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            stop_price=stop_price,
            target_price=target_price,
        )
        if exit_fill is not None:
            exit_raw = exit_fill.price
            exit_timestamp = bar["timestamp"]
            break
    if exit_raw is None:
        exit_raw = float(bars[last_eligible_index]["close"])
        exit_timestamp = bars[last_eligible_index]["timestamp"]

    exit_price = costs.exit_fill(float(exit_raw), direction)
    gross_pnl = direction * (float(exit_raw) - entry_raw)
    slippage_cost = direction * (
        (entry_price - entry_raw) + (float(exit_raw) - exit_price)
    )
    fees = costs.fees(entry_price, exit_price)
    net_pnl = gross_pnl - slippage_cost - fees
    net_return_r = net_pnl / stop_distance
    return _outcome_label(net_pnl, stop_distance), exit_timestamp, net_return_r


def build_training_dataset(
    input_paths: list[str | Path],
    *,
    style: ModelStyle,
    strategy: str,
    horizon: int,
    sequence_length: int = 60,
    costs: ExecutionCosts | None = None,
) -> TrainingDataset:
    """Build causal strategy-event examples with cost-aware forward outcome labels."""
    if not input_paths:
        raise ValueError("At least one --input file is required")
    if horizon < 1:
        raise ValueError("horizon must be at least one bar")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least one bar")
    if style not in ("intraday", "swing"):
        raise ValueError("style must be 'intraday' or 'swing'")
    costs = costs or ExecutionCosts()

    feature_names: tuple[str, ...] | None = None
    feature_rows: list[np.ndarray] = []
    sequence_rows: list[np.ndarray] = []
    labels: list[int] = []
    net_returns_r: list[float] = []
    timestamps: list[datetime] = []
    label_end_timestamps: list[datetime] = []

    for input_path in input_paths:
        path = Path(input_path)
        source_name = path.name[:-7] if path.name.lower().endswith(".csv.gz") else path.stem
        symbol = path.parent.name
        if symbol.lower() in {"minute", "day", "intraday", "swing", "raw"}:
            symbol = source_name
        frame = build_features(path, style=style).collect()
        if "symbol" not in frame.columns:
            frame = frame.with_columns(pl.lit(symbol).alias("symbol"))
        current_features = _numeric_feature_names(frame)
        if feature_names is None:
            feature_names = current_features
        elif current_features != feature_names:
            raise ValueError(
                f"Feature columns differ between input files; {path} is incompatible"
            )

        signals = generate_strategy_signals(
            frame,
            strategy=strategy,
            style=style,
            symbol=symbol,
        )
        if signals.is_empty():
            continue
        values = frame.select(feature_names).to_numpy().astype(np.float32, copy=False)
        bars = frame.select(
            "timestamp", "open", "high", "low", "close"
        ).iter_rows(named=True)
        bar_rows = list(bars)
        row_indices = {
            timestamp: index
            for index, timestamp in enumerate(frame["timestamp"].to_list())
        }
        for signal in signals.iter_rows(named=True):
            signal_index = row_indices[signal["timestamp"]]
            start_index = signal_index - sequence_length + 1
            if start_index < 0:
                continue
            feature_vector = values[signal_index]
            sequence = values[start_index : signal_index + 1]
            if not np.isfinite(feature_vector).all():
                continue
            outcome = _label_candidate(
                bars=bar_rows,
                signal_index=signal_index,
                signal=signal,
                style=style,
                horizon=horizon,
                costs=costs,
            )
            if outcome is None:
                continue
            label, label_end, net_return_r = outcome
            feature_rows.append(feature_vector.copy())
            sequence_rows.append(sequence.copy())
            labels.append(label)
            net_returns_r.append(net_return_r)
            timestamps.append(signal["timestamp"])
            label_end_timestamps.append(label_end)

    if not feature_rows or feature_names is None:
        raise ValueError(
            "No usable labeled strategy events were found. Check the input data, "
            "strategy, history length, and horizon."
        )
    order = sorted(range(len(timestamps)), key=lambda index: timestamps[index])
    return TrainingDataset(
        features=np.stack([feature_rows[index] for index in order]),
        sequences=np.stack([sequence_rows[index] for index in order]),
        labels=np.asarray([labels[index] for index in order], dtype=np.int64),
        net_returns_r=np.asarray(
            [net_returns_r[index] for index in order],
            dtype=np.float32,
        ),
        timestamps=tuple(timestamps[index] for index in order),
        label_end_timestamps=tuple(label_end_timestamps[index] for index in order),
        feature_names=feature_names,
        style=style,
        strategy=strategy,
        horizon=horizon,
        sequence_length=sequence_length,
    )


def chronological_split(
    dataset: TrainingDataset,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetSplit:
    """Split by timestamp and purge examples whose forward labels cross boundaries."""
    if not 0 < validation_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 0.5")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation and test fractions leave too little training data")
    count = len(dataset.timestamps)
    if count < 20:
        raise ValueError(f"At least 20 labeled events are required; found {count}")

    timestamps = dataset.timestamps
    unique_timestamps = sorted(set(timestamps))
    if len(unique_timestamps) < 3:
        raise ValueError("At least three distinct signal timestamps are required")
    validation_position = int(
        len(unique_timestamps) * (1.0 - validation_fraction - test_fraction)
    )
    test_position = int(len(unique_timestamps) * (1.0 - test_fraction))
    validation_position = min(max(validation_position, 1), len(unique_timestamps) - 2)
    test_position = min(
        max(test_position, validation_position + 1),
        len(unique_timestamps) - 1,
    )
    validation_start = unique_timestamps[validation_position]
    test_start = unique_timestamps[test_position]

    train = np.asarray(
        [
            index
            for index, timestamp in enumerate(timestamps)
            if timestamp < validation_start
            and dataset.label_end_timestamps[index] < validation_start
        ],
        dtype=np.int64,
    )
    validation = np.asarray(
        [
            index for index, timestamp in enumerate(timestamps)
            if validation_start <= timestamp < test_start
            and dataset.label_end_timestamps[index] < test_start
        ],
        dtype=np.int64,
    )
    test = np.asarray(
        [
            index
            for index, timestamp in enumerate(timestamps)
            if timestamp >= test_start
        ],
        dtype=np.int64,
    )
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Chronological split is empty after purging overlapping forward labels; "
            "add more data or shorten the label horizon."
        )
    if len(np.unique(dataset.labels[train])) < 2:
        raise ValueError("The training period must contain at least two outcome classes")
    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        validation_start=validation_start,
        test_start=test_start,
    )
