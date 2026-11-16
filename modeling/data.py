from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
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
_CACHE_COPY_CHUNK_BYTES = 8 * 1024**2


def _write_npy_from_raw(
    raw_path: Path,
    output_path: Path,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    array_name: str,
) -> None:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Raw {array_name} cache has {actual_bytes:,} bytes; "
            f"expected {expected_bytes:,} for shape {shape} and dtype {dtype}"
        )

    header = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    copied = 0
    started = time.monotonic()
    last_report = started
    with raw_path.open("rb") as source, output_path.open("wb") as target:
        np.lib.format.write_array_header_2_0(target, header)
        while chunk := source.read(_CACHE_COPY_CHUNK_BYTES):
            target.write(chunk)
            copied += len(chunk)
            now = time.monotonic()
            if now - last_report >= 15:
                elapsed = now - started
                rate = copied / max(elapsed, 1e-6)
                remaining = (expected_bytes - copied) / max(rate, 1e-6)
                print(
                    f"Finalizing {array_name}: {copied / 1024**3:.2f}/"
                    f"{expected_bytes / 1024**3:.2f} GiB "
                    f"({100 * copied / max(expected_bytes, 1):.1f}%), "
                    f"about {remaining / 60:.1f} min remaining",
                    flush=True,
                )
                last_report = now
        target.flush()

    if copied != expected_bytes:
        output_path.unlink(missing_ok=True)
        raise OSError(
            f"Raw {array_name} cache ended after {copied:,} bytes; "
            f"expected {expected_bytes:,}"
        )
    print(
        f"Finalized {array_name}: {expected_bytes / 1024**3:.2f} GiB "
        f"in {(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )


@dataclass(frozen=True)
class TrainingDataset:
    features: np.ndarray
    sequences: np.ndarray
    labels: np.ndarray
    net_returns_r: np.ndarray
    timestamps: tuple[datetime, ...] | np.ndarray
    label_end_timestamps: tuple[datetime, ...] | np.ndarray
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
    observed: np.ndarray | None = None,
) -> np.ndarray:
    """Impute missing historical features with training means, then scale."""
    broadcast_mean = mean[None, None, :]
    finite_sequences = np.where(np.isfinite(sequences), sequences, broadcast_mean)
    safe_scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (finite_sequences - broadcast_mean) / safe_scale[None, None, :]
    if observed is not None:
        normalized = np.where(observed[None, None, :], normalized, 0.0)
    return normalized.astype(np.float32)


def training_feature_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit scaling using finite training values only; unobserved columns stay disabled."""
    finite = np.isfinite(values)
    if finite.all():
        mean = values.mean(axis=0).astype(np.float32)
        scale = values.std(axis=0).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return mean, scale, np.ones(values.shape[1], dtype=bool)
    counts = finite.sum(axis=0)
    safe_values = np.where(finite, values, 0.0)
    mean = (safe_values.sum(axis=0, dtype=np.float64) / np.maximum(counts, 1)).astype(
        np.float32
    )
    differences = np.where(finite, values - mean, 0.0)
    scale = np.sqrt(
        np.square(differences, dtype=np.float64).sum(axis=0) / np.maximum(counts, 1)
    ).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return mean, scale, counts > 0


def _usable_feature_vector(
    values: np.ndarray, feature_names: tuple[str, ...], path: Path
) -> bool:
    finite = np.isfinite(values)
    if path.name.lower() in {"trading_week.csv", "trading_week.csv.gz"}:
        if "ema_200" in feature_names:
            finite[feature_names.index("ema_200")] = True
    return bool(finite.all())


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


def _dataset_cache_manifest(
    input_paths: list[str | Path],
    *,
    style: ModelStyle,
    strategy: str,
    horizon: int,
    sequence_length: int,
    costs: ExecutionCosts,
    max_events_per_file: int | None = None,
    event_stride: int = 1,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "format_version": 4,
        "sequence_dtype": "float32",
        "max_events_per_file": max_events_per_file,
        "style": style,
        "strategy": strategy,
        "horizon": horizon,
        "sequence_length": sequence_length,
        "costs": {
            "commission_bps": costs.commission_bps,
            "other_fees_bps": costs.other_fees_bps,
            "slippage_bps": costs.slippage_bps,
        },
        "inputs": [
            {
                "path": str(Path(input_path).resolve()),
                "size": Path(input_path).stat().st_size,
                "mtime_ns": Path(input_path).stat().st_mtime_ns,
            }
            for input_path in input_paths
        ],
    }
    if event_stride != 1:
        manifest["event_stride"] = event_stride
    return manifest


def _load_cached_dataset(
    cache_dir: Path,
    manifest: dict[str, object],
) -> TrainingDataset | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved.get("configuration") != manifest:
        raise ValueError(f"Dataset cache does not match current inputs: {cache_dir}")
    required = (
        "features.npy",
        "sequences.npy",
        "labels.npy",
        "net_returns_r.npy",
        "timestamps_ns.npy",
        "label_end_timestamps_ns.npy",
    )
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Dataset cache is incomplete under {cache_dir}: {', '.join(missing)}"
        )
    arrays = {
        name: np.load(cache_dir / name, mmap_mode="r")
        for name in required
    }
    return TrainingDataset(
        features=arrays["features.npy"],
        sequences=arrays["sequences.npy"],
        labels=arrays["labels.npy"],
        net_returns_r=arrays["net_returns_r.npy"],
        timestamps=arrays["timestamps_ns.npy"],
        label_end_timestamps=arrays["label_end_timestamps_ns.npy"],
        feature_names=tuple(saved["feature_names"]),
        style=saved["style"],
        strategy=saved["strategy"],
        horizon=saved["horizon"],
        sequence_length=saved["sequence_length"],
    )


def _datetime_to_ns(value: datetime) -> np.int64:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(normalized, "ns").astype(np.int64)


def _build_cached_training_dataset(
    input_paths: list[str | Path],
    *,
    style: ModelStyle,
    strategy: str,
    horizon: int,
    sequence_length: int,
    costs: ExecutionCosts,
    cache_dir: str | Path,
    max_events_per_file: int | None = None,
    event_stride: int = 1,
) -> TrainingDataset:
    output = Path(cache_dir)
    manifest = _dataset_cache_manifest(
        input_paths,
        style=style,
        strategy=strategy,
        horizon=horizon,
        sequence_length=sequence_length,
        costs=costs,
        max_events_per_file=max_events_per_file,
        event_stride=event_stride,
    )
    if output.exists():
        cached = _load_cached_dataset(output, manifest)
        if cached is None:
            raise FileExistsError(
                f"Incomplete dataset cache exists: {output}. Remove it before retrying."
            )
        return cached

    building = output.with_name(f"{output.name}.building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    raw_paths = {
        "features": building / "features.bin",
        "sequences": building / "sequences.bin",
        "labels": building / "labels.bin",
        "returns": building / "returns.bin",
        "timestamps": building / "timestamps.bin",
        "label_ends": building / "label_ends.bin",
    }
    feature_names: tuple[str, ...] | None = None
    row_count = 0
    raw_bytes_written = 0
    chunk_size = 512
    feature_buffer: list[np.ndarray] = []
    sequence_buffer: list[np.ndarray] = []
    labels_buffer: list[int] = []
    returns_buffer: list[float] = []
    timestamps_buffer: list[int] = []
    label_ends_buffer: list[int] = []

    def flush_chunk() -> None:
        nonlocal row_count, raw_bytes_written
        if not feature_buffer:
            return
        arrays = {
            "features": np.asarray(feature_buffer, dtype=np.float32),
            "sequences": np.asarray(sequence_buffer, dtype=np.float32),
            "labels": np.asarray(labels_buffer, dtype=np.int64),
            "returns": np.asarray(returns_buffer, dtype=np.float32),
            "timestamps": np.asarray(timestamps_buffer, dtype=np.int64),
            "label_ends": np.asarray(label_ends_buffer, dtype=np.int64),
        }
        expected_sequences = (
            len(feature_buffer),
            sequence_length,
            arrays["features"].shape[1],
        )
        if arrays["sequences"].shape != expected_sequences:
            raise ValueError(
                "Pooled sequence chunk has an unexpected shape: "
                f"{arrays['sequences'].shape}; expected {expected_sequences}"
            )
        chunk_bytes = sum(array.nbytes for array in arrays.values())
        free_bytes = shutil.disk_usage(building).free
        if free_bytes < raw_bytes_written + chunk_bytes + 2 * 1024**3:
            raise OSError(
                "Insufficient disk space for the disk-backed pooled dataset cache; "
                "free space or reduce the symbol filter before retrying."
            )
        for name, array in arrays.items():
            with raw_paths[name].open("ab") as target:
                array.tofile(target)
        raw_bytes_written += chunk_bytes
        row_count += len(feature_buffer)
        feature_buffer.clear()
        sequence_buffer.clear()
        labels_buffer.clear()
        returns_buffer.clear()
        timestamps_buffer.clear()
        label_ends_buffer.clear()

    try:
        for file_index, input_path in enumerate(input_paths, start=1):
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
            if event_stride > 1:
                signals = signals.gather_every(event_stride)
            if signals.is_empty():
                if file_index % 100 == 0:
                    print(
                        f"Prepared {file_index:,}/{len(input_paths):,} input files "
                        f"for {strategy} {path.parent.name} {path.name} (no signals)",
                        flush=True,
                    )
                continue
            if max_events_per_file and signals.height > max_events_per_file:
                picks = np.linspace(
                    0, signals.height - 1, max_events_per_file
                ).astype(np.int64)
                signals = signals[picks.tolist()]
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
                if not _usable_feature_vector(feature_vector, feature_names, path):
                    continue
                feature_vector = np.where(np.isfinite(feature_vector), feature_vector, np.nan)
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
                feature_buffer.append(feature_vector.copy())
                sequence_buffer.append(
                    values[start_index : signal_index + 1].copy()
                )
                labels_buffer.append(int(label))
                returns_buffer.append(net_return_r)
                timestamps_buffer.append(_datetime_to_ns(signal["timestamp"]))
                label_ends_buffer.append(_datetime_to_ns(label_end))
                if len(feature_buffer) >= chunk_size:
                    flush_chunk()
            if file_index % 100 == 0:
                print(
                    f"Prepared {file_index:,}/{len(input_paths):,} input files "
                    f"for {strategy} {path.parent.name} {path.name}",
                    flush=True,
                )
        flush_chunk()
        if not row_count or feature_names is None:
            raise ValueError(
                "No usable labeled strategy events were found. Check the input "
                "data, strategy, history length, and horizon."
            )
        print(f"Collected {row_count:,} usable labeled events for {strategy}.", flush=True)

        shapes = {
            "features": (row_count, len(feature_names)),
            "sequences": (row_count, sequence_length, len(feature_names)),
            "labels": (row_count,),
            "returns": (row_count,),
            "timestamps": (row_count,),
            "label_ends": (row_count,),
        }
        dtypes = {
            "features": np.float32,
            "sequences": np.float32,
            "labels": np.int64,
            "returns": np.float32,
            "timestamps": np.int64,
            "label_ends": np.int64,
        }
        output_names = {
            "features": "features.npy",
            "sequences": "sequences.npy",
            "labels": "labels.npy",
            "returns": "net_returns_r.npy",
            "timestamps": "timestamps_ns.npy",
            "label_ends": "label_end_timestamps_ns.npy",
        }
        for name, shape in shapes.items():
            _write_npy_from_raw(
                raw_paths[name],
                building / output_names[name],
                dtype=np.dtype(dtypes[name]),
                shape=shape,
                array_name=name,
            )
            raw_paths[name].unlink(missing_ok=True)

        (building / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration": manifest,
                    "feature_names": list(feature_names),
                    "row_count": row_count,
                    "style": style,
                    "strategy": strategy,
                    "horizon": horizon,
                    "sequence_length": sequence_length,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        building.replace(output)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise

    cached = _load_cached_dataset(output, manifest)
    if cached is None:
        raise RuntimeError(f"Failed to finalize dataset cache: {output}")
    return cached


def build_training_dataset(
    input_paths: list[str | Path],
    *,
    style: ModelStyle,
    strategy: str,
    horizon: int,
    sequence_length: int = 60,
    costs: ExecutionCosts | None = None,
    cache_dir: str | Path | None = None,
    max_events_per_file: int | None = None,
    event_stride: int = 1,
) -> TrainingDataset:
    """Build causal strategy-event examples with cost-aware forward outcome labels."""
    if not input_paths:
        raise ValueError("At least one --input file is required")
    if horizon < 1:
        raise ValueError("horizon must be at least one bar")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least one bar")
    if event_stride < 1:
        raise ValueError("event_stride must be at least one")
    if event_stride != 1 and max_events_per_file is not None:
        raise ValueError("event_stride cannot be combined with max_events_per_file")
    if style not in ("intraday", "swing"):
        raise ValueError("style must be 'intraday' or 'swing'")
    costs = costs or ExecutionCosts()
    if cache_dir is not None:
        return _build_cached_training_dataset(
            input_paths,
            style=style,
            strategy=strategy,
            horizon=horizon,
            sequence_length=sequence_length,
            costs=costs,
            cache_dir=cache_dir,
            max_events_per_file=max_events_per_file,
            event_stride=event_stride,
        )

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
        if event_stride > 1:
            signals = signals.gather_every(event_stride)
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
            if not _usable_feature_vector(feature_vector, feature_names, path):
                continue
            feature_vector = np.where(np.isfinite(feature_vector), feature_vector, np.nan)
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
    cached_timestamps = isinstance(timestamps, np.ndarray)
    if cached_timestamps:
        unique_timestamps = np.unique(timestamps)
    else:
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
    validation_boundary = unique_timestamps[validation_position]
    test_boundary = unique_timestamps[test_position]
    if cached_timestamps:
        validation_start = datetime.fromtimestamp(
            int(validation_boundary) / 1_000_000_000,
            tz=timezone.utc,
        )
        test_start = datetime.fromtimestamp(
            int(test_boundary) / 1_000_000_000,
            tz=timezone.utc,
        )
        label_ends = dataset.label_end_timestamps
        train = np.flatnonzero(
            (timestamps < validation_boundary)
            & (label_ends < validation_boundary)
        )
        validation = np.flatnonzero(
            (timestamps >= validation_boundary)
            & (timestamps < test_boundary)
            & (label_ends < test_boundary)
        )
        test = np.flatnonzero(timestamps >= test_boundary)
    else:
        validation_start = validation_boundary
        test_start = test_boundary
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
