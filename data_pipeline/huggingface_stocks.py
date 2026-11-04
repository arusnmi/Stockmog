from __future__ import annotations

import csv
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import polars as pl

from config.settings import DATA_DIR
from features._shared import safe_divide
from features.technical_indicators import add_price_features, add_technical_indicators

HUGGINGFACE_REPO_ID = "Saintforest/indian-stock-market-minute-data"
HUGGINGFACE_CACHE_DIR = DATA_DIR / "downloads" / "huggingface"
STOCK_DATA_DIR = DATA_DIR / "stocks"
NSE_HOLIDAYS_CSV = DATA_DIR / "calendar" / "nse_holidays.csv"
INTRADAY_TIMEFRAMES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
TIMEFRAME_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
    "source_bars",
    "trading_days",
    "is_partial",
]
OUTPUT_SUFFIX = ".csv.gz"

DatasetSplit = Literal["minute", "day"]


def load_nse_holidays(file_path: str | Path) -> set[date]:
    """Load NSE market-closure dates from a CSV with a YYYY-MM-DD `date` column."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"NSE holiday calendar not found: {path}. Create it with a `date` "
            "column containing official NSE market-closure dates."
        )

    with path.open(encoding="utf-8-sig", newline="") as calendar_file:
        reader = csv.DictReader(calendar_file)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            raise ValueError(f"NSE holiday calendar must contain a `date` column: {path}")
        holidays: set[date] = set()
        for line_number, row in enumerate(reader, start=2):
            raw_date = (row.get("date") or "").strip()
            if not raw_date:
                continue
            try:
                holidays.add(date.fromisoformat(raw_date))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid holiday date on line {line_number} in {path}: {raw_date!r}; "
                    "expected YYYY-MM-DD."
                ) from exc
    if not holidays:
        raise ValueError(
            f"NSE holiday calendar contains no dates: {path}. Add official NSE "
            "market-closure dates in YYYY-MM-DD format before preparing data."
        )
    return holidays


def _filter_trading_days(frame: pl.DataFrame, holidays: set[date]) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Asia/Kolkata"),
        pl.col("timestamp").dt.convert_time_zone("Asia/Kolkata").dt.date().alias("_trading_date"),
    ).filter(pl.col("_trading_date").dt.weekday() <= 5)
    if holidays:
        frame = frame.filter(~pl.col("_trading_date").is_in(sorted(holidays)))
    return frame


def _safe_symbol(symbol: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol.strip())
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Invalid stock symbol: {symbol!r}")
    return safe


def _prepare_stock_frame(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Parquet shard is missing columns: {', '.join(sorted(missing))}")

    timestamp_type = frame.schema["timestamp"]
    if not isinstance(timestamp_type, pl.Datetime):
        raise ValueError(
            f"Expected a datetime timestamp column, got {timestamp_type!r}."
        )
    if timestamp_type.time_zone is None:
        raise ValueError("Source timestamps must include the UTC timezone.")

    expressions = [
        pl.col("timestamp")
        .dt.convert_time_zone("Asia/Kolkata")
        .alias("timestamp"),
        *[
            pl.col(name).cast(pl.Float64, strict=True).alias(name)
            for name in ("open", "high", "low", "close")
        ],
        pl.col("volume").cast(pl.Float64, strict=False).fill_null(0.0).alias("volume"),
    ]
    if "oi" in frame.columns:
        expressions.append(pl.col("oi").cast(pl.Float64, strict=False).alias("oi"))
    else:
        expressions.append(pl.lit(None, dtype=pl.Float64).alias("oi"))

    frame = frame.select(expressions).sort("timestamp")
    frame = frame.unique(subset=["timestamp"], keep="last", maintain_order=True)
    invalid_rows = frame.filter(
        pl.any_horizontal(
            pl.any_horizontal(
                *[
                    pl.col(name).is_null() | ~pl.col(name).is_finite()
                    for name in ("open", "high", "low", "close")
                ]
            ),
            pl.col("open") <= 0,
            pl.col("high") < pl.max_horizontal("open", "low", "close"),
            pl.col("low") > pl.min_horizontal("open", "high", "close"),
        )
    ).height
    if invalid_rows:
        raise ValueError(f"Found {invalid_rows} invalid OHLC rows.")
    return frame


def _aggregate_ohlcv(
    frame: pl.DataFrame,
    *,
    group_columns: list[str],
    symbol: str,
    expected_bars: int | None,
) -> pl.DataFrame:
    price_volume = (
        pl.col("_price_volume_sum")
        if "_price_volume_sum" in frame.columns
        else ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
        * pl.col("volume")
    )
    source_bar_count = (
        pl.col("source_bars").sum()
        if "source_bars" in frame.columns
        else pl.len()
    )
    aggregated = (
        frame.group_by(group_columns, maintain_order=True)
        .agg(
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("oi").last().alias("oi"),
            source_bar_count.alias("source_bars"),
            pl.col("_trading_date").n_unique().alias("trading_days"),
            pl.len().alias("_group_rows"),
            price_volume.sum().alias("_price_volume_sum"),
        )
        .sort("timestamp")
    )
    return aggregated.with_columns(
        pl.lit(symbol).alias("symbol"),
        (
            pl.col("_group_rows") < expected_bars
            if expected_bars is not None
            else pl.lit(False)
        ).alias("is_partial"),
    ).select(
        *TIMEFRAME_COLUMNS,
        "_price_volume_sum",
    )


def resample_intraday(
    frame: pl.DataFrame,
    *,
    minutes: int,
    symbol: str,
    holidays: set[date] | None = None,
) -> pl.DataFrame:
    """Resample UTC minute rows into IST session-anchored candles."""
    if minutes not in INTRADAY_TIMEFRAMES.values():
        raise ValueError(f"Unsupported intraday candle size: {minutes} minutes")

    frame = _filter_trading_days(frame, holidays or set())
    session_open_minute = 9 * 60 + 15
    session_close_minute = 15 * 60 + 30
    frame = frame.with_columns(
        pl.col("_trading_date").alias("_session_date"),
        (
            pl.col("timestamp").dt.hour().cast(pl.Int64) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int64)
        ).alias("_minute_of_day"),
    ).filter(
        (pl.col("_minute_of_day") >= session_open_minute)
        & (pl.col("_minute_of_day") < session_close_minute)
    )
    frame = frame.with_columns(
        (
            (pl.col("_minute_of_day") - session_open_minute) // minutes
        ).alias("_bucket_index")
    ).with_columns(
        (
            session_open_minute + pl.col("_bucket_index") * minutes
        ).alias("_bucket_start_minute")
    ).with_columns(
        pl.datetime(
            pl.col("_session_date").dt.year(),
            pl.col("_session_date").dt.month(),
            pl.col("_session_date").dt.day(),
            hour=pl.col("_bucket_start_minute") // 60,
            minute=pl.col("_bucket_start_minute") % 60,
            time_unit="us",
        )
        .dt.replace_time_zone("Asia/Kolkata")
        .alias("timestamp")
    )

    return _aggregate_ohlcv(
        frame,
        group_columns=["_session_date", "_bucket_index"],
        symbol=symbol,
        expected_bars=minutes,
    )


def resample_daily(
    frame: pl.DataFrame,
    *,
    symbol: str,
    holidays: set[date] | None = None,
) -> pl.DataFrame:
    """Aggregate minute rows into one candle per actual market session."""
    frame = _filter_trading_days(frame, holidays or set())
    frame = frame.with_columns(
        (
            pl.col("timestamp").dt.hour().cast(pl.Int64) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int64)
        ).alias("_minute_of_day")
    ).filter(
        (pl.col("_minute_of_day") >= 9 * 60 + 15)
        & (pl.col("_minute_of_day") < 15 * 60 + 30)
    )
    return _aggregate_ohlcv(
        frame,
        group_columns=["_trading_date"],
        symbol=symbol,
        expected_bars=375,
    )


def _prepare_daily_candles(
    frame: pl.DataFrame,
    *,
    holidays: set[date] | None = None,
) -> pl.DataFrame:
    frame = _filter_trading_days(frame, holidays or set())
    expressions = []
    if "source_bars" not in frame.columns:
        expressions.append(pl.lit(1, dtype=pl.UInt32).alias("source_bars"))
    if "trading_days" not in frame.columns:
        expressions.append(pl.lit(1, dtype=pl.UInt32).alias("trading_days"))
    if "is_partial" not in frame.columns:
        expressions.append(pl.lit(False).alias("is_partial"))
    if "_price_volume_sum" not in frame.columns:
        expressions.append(
            (
                (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
                * pl.col("volume")
            ).alias("_price_volume_sum")
        )
    return frame.with_columns(expressions)


def resample_bi_daily(
    frame: pl.DataFrame,
    *,
    symbol: str,
    holidays: set[date] | None = None,
) -> pl.DataFrame:
    """Aggregate consecutive market sessions in non-overlapping pairs."""
    frame = _prepare_daily_candles(frame, holidays=holidays)
    frame = frame.sort("timestamp").with_row_index("_session_index")
    frame = frame.with_columns(
        (pl.col("_session_index") // 2).alias("_two_session_group")
    )
    return _aggregate_ohlcv(
        frame,
        group_columns=["_two_session_group"],
        symbol=symbol,
        expected_bars=2,
    )


def resample_trading_week(
    frame: pl.DataFrame,
    *,
    symbol: str,
    holidays: set[date] | None = None,
) -> pl.DataFrame:
    """Aggregate observed NSE sessions into a Monday-labelled trading week."""
    frame = _prepare_daily_candles(frame, holidays=holidays)
    frame = frame.with_columns(
        pl.col("timestamp").dt.truncate("1w").alias("_week_start")
    ).with_columns(
        pl.col("_week_start").alias("timestamp")
    )
    weekly = _aggregate_ohlcv(
        frame,
        group_columns=["_week_start"],
        symbol=symbol,
        expected_bars=None,
    )
    expected_sessions = []
    for week_start in weekly["timestamp"].to_list():
        week_date = week_start.date()
        expected_sessions.append(
            sum(
                (week_date + timedelta(days=offset)).weekday() < 5
                and (week_date + timedelta(days=offset)) not in (holidays or set())
                for offset in range(5)
            )
        )
    return weekly.with_columns(
        pl.Series("expected_trading_days", expected_sessions, dtype=pl.UInt8)
    ).with_columns(
        (pl.col("trading_days") < pl.col("expected_trading_days")).alias("is_partial")
    )


def add_stock_indicators(
    candles: pl.DataFrame,
    *,
    timeframe: str,
) -> pl.LazyFrame:
    """Append deterministic technical indicators to one stock's candles."""
    if timeframe not in {*INTRADAY_TIMEFRAMES, "1d", "2d", "trading_week"}:
        raise ValueError(f"Unsupported output timeframe: {timeframe}")

    frame = candles.lazy().sort("timestamp")
    frame = add_price_features(frame)
    frame = add_technical_indicators(frame)
    frame = frame.with_columns(
        pl.col("close").rolling_mean(window_size=20, min_samples=20).alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50, min_samples=50).alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200, min_samples=200).alias("sma_200"),
        pl.col("timestamp").dt.date().alias("_vwap_period"),
        (
            (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        ).alias("_typical_price"),
    ).with_columns(
        (
            pl.col("_price_volume_sum")
            .cum_sum()
            .over("_vwap_period")
        ).alias("_cumulative_price_volume"),
        pl.col("volume").cum_sum().over("_vwap_period").alias("_cumulative_volume"),
    ).with_columns(
        safe_divide(
            pl.col("_cumulative_price_volume"),
            pl.col("_cumulative_volume"),
        ).alias("vwap"),
        (
            pl.col("bb_upper") - pl.col("bb_lower")
        ).alias("bb_width"),
        safe_divide(
            pl.col("bb_upper") - pl.col("bb_lower"),
            pl.col("bb_mid"),
        ).alias("bb_width_pct"),
    )
    return frame.drop(
        "_price_volume_sum",
        "_vwap_period",
        "_typical_price",
        "_cumulative_price_volume",
        "_cumulative_volume",
    )


def _write_feature_csv(
    candles: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    output_root: Path,
) -> Path:
    safe_symbol = _safe_symbol(symbol)
    output = output_root / safe_symbol / f"{timeframe}{OUTPUT_SUFFIX}"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.gz")
    try:
        add_stock_indicators(candles, timeframe=timeframe).sink_csv(
            temporary_output,
            compression="gzip",
            null_value="",
        )
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    return output


def process_stock_frame(
    frame: pl.DataFrame,
    *,
    symbol: str,
    split: DatasetSplit,
    output_root: str | Path,
    holidays: set[date] | None = None,
    date_range: tuple[date, date] | None = None,
) -> list[Path]:
    """Create all requested per-stock timeframes and indicators from one symbol."""
    if split not in ("minute", "day"):
        raise ValueError("split must be 'minute' or 'day'")
    if not symbol.strip():
        raise ValueError("symbol must not be empty")

    frame = _prepare_stock_frame(frame)
    output_directory = Path(output_root) / split
    outputs: list[Path] = []
    if split == "minute":
        for timeframe, minutes in INTRADAY_TIMEFRAMES.items():
            candles = resample_intraday(
                frame,
                minutes=minutes,
                symbol=symbol,
                holidays=holidays,
            )
            if candles.height:
                outputs.append(
                    _write_feature_csv(
                        candles,
                        symbol=symbol,
                        timeframe=timeframe,
                        output_root=output_directory,
                    )
                )
        daily = resample_daily(frame, symbol=symbol, holidays=holidays)
    else:
        daily = _prepare_daily_candles(frame, holidays=holidays)
        if date_range is not None:
            start_date, end_date = date_range
            if start_date > end_date:
                raise ValueError("date_range start must not be after end")
            daily = daily.filter(
                pl.col("timestamp").dt.date().is_between(
                    start_date,
                    end_date,
                    closed="both",
                )
            )

    daily = _prepare_daily_candles(daily, holidays=holidays)
    if daily.height:
        outputs.append(
            _write_feature_csv(
                daily,
                symbol=symbol,
                timeframe="1d",
                output_root=output_directory,
            )
        )
        for timeframe, candles in (
            ("2d", resample_bi_daily(daily, symbol=symbol, holidays=holidays)),
            (
                "trading_week",
                resample_trading_week(daily, symbol=symbol, holidays=holidays),
            ),
        ):
            if candles.height:
                outputs.append(
                    _write_feature_csv(
                        candles,
                        symbol=symbol,
                        timeframe=timeframe,
                        output_root=output_directory,
                    )
                )
    return outputs


def _process_parquet_shard(
    parquet_path: str | Path,
    *,
    split: DatasetSplit,
    output_root: Path,
    selected_symbols: set[str] | None,
    seen_symbols: set[str],
    batch_size: int,
    holidays: set[date],
    date_range: tuple[date, date] | None = None,
) -> tuple[int, int, date | None, date | None]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading Hugging Face Parquet shards requires PyArrow. "
            "Install the project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    parquet_file = pq.ParquetFile(parquet_path)
    symbol_column = parquet_file.schema_arrow.get_field_index("symbol")
    timestamp_column = parquet_file.schema_arrow.get_field_index("timestamp")
    if symbol_column < 0:
        raise ValueError(f"Parquet shard has no symbol column: {parquet_path}")
    if timestamp_column < 0:
        raise ValueError(f"Parquet shard has no timestamp column: {parquet_path}")

    selected_rows = 0
    processed_symbols = 0
    minimum_date: date | None = None
    maximum_date: date | None = None
    active_symbol: str | None = None
    symbol_batches: list[Any] = []

    def flush_symbol() -> None:
        nonlocal selected_rows, processed_symbols, symbol_batches
        if active_symbol is None or not symbol_batches:
            return
        if active_symbol in seen_symbols:
            raise ValueError(
                f"Symbol {active_symbol} appears in multiple Parquet shards. "
                "The source ordering differs from the dataset documentation."
            )
        seen_symbols.add(active_symbol)
        stock_table = pa.Table.from_batches(symbol_batches)
        stock_frame = pl.from_arrow(stock_table)
        paths = process_stock_frame(
            stock_frame,
            symbol=active_symbol,
            split=split,
            output_root=output_root,
            holidays=holidays,
            date_range=date_range,
        )
        selected_rows += stock_frame.height
        processed_symbols += 1
        print(
            f"{split}: {active_symbol}: {stock_frame.height} rows -> "
            f"{len(paths)} CSVs",
            flush=True,
        )
        symbol_batches = []

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        timestamps = batch.column(timestamp_column)
        if not pa.types.is_timestamp(timestamps.type) or timestamps.type.tz is None:
            raise ValueError(
                f"Parquet timestamps must be timezone-aware: {parquet_path}"
            )
        if len(timestamps):
            extrema = pc.min_max(timestamps)
            batch_min = extrema["min"].as_py()
            batch_max = extrema["max"].as_py()
            local_min_date = _ist_date(batch_min)
            local_max_date = _ist_date(batch_max)
            minimum_date = (
                local_min_date
                if minimum_date is None
                else min(minimum_date, local_min_date)
            )
            maximum_date = (
                local_max_date
                if maximum_date is None
                else max(maximum_date, local_max_date)
            )
        symbols = batch.column(symbol_column).to_numpy(zero_copy_only=False)
        if len(symbols) == 0:
            continue
        boundaries = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
        starts = np.concatenate((np.array([0]), boundaries))
        ends = np.concatenate((boundaries, np.array([len(symbols)])))

        for start, end in zip(starts, ends):
            batch_symbol_value = symbols[int(start)]
            if batch_symbol_value is None:
                raise ValueError(f"Null symbol in Parquet shard: {parquet_path}")
            batch_symbol = str(batch_symbol_value).strip().upper()
            if not batch_symbol:
                raise ValueError(f"Empty symbol in Parquet shard: {parquet_path}")
            if active_symbol != batch_symbol:
                flush_symbol()
                active_symbol = batch_symbol
            if selected_symbols is None or batch_symbol in selected_symbols:
                symbol_batches.append(batch.slice(int(start), int(end - start)))

    flush_symbol()
    return processed_symbols, selected_rows, minimum_date, maximum_date


def _ist_date(timestamp: datetime) -> date:
    """Return the Asia/Kolkata calendar date for a timezone-aware timestamp."""
    if timestamp.tzinfo is None:
        raise ValueError("Dataset timestamps must include a timezone.")
    return (timestamp.astimezone(timezone.utc) + timedelta(hours=5, minutes=30)).date()


def _scan_minute_date_range(
    parquet_path: str | Path,
    *,
    batch_size: int,
) -> tuple[date, date]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading Hugging Face Parquet shards requires PyArrow. "
            "Install the project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    parquet_file = pq.ParquetFile(parquet_path)
    timestamp_column = parquet_file.schema_arrow.get_field_index("timestamp")
    if timestamp_column < 0:
        raise ValueError(f"Parquet shard has no timestamp column: {parquet_path}")
    timestamp_type = parquet_file.schema_arrow.field(timestamp_column).type
    if not pa.types.is_timestamp(timestamp_type) or timestamp_type.tz is None:
        raise ValueError(f"Parquet timestamps must be timezone-aware: {parquet_path}")

    minimum_date: date | None = None
    maximum_date: date | None = None
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["timestamp"],
    ):
        timestamps = batch.column(0)
        if not len(timestamps):
            continue
        extrema = pc.min_max(timestamps)
        batch_min = _ist_date(extrema["min"].as_py())
        batch_max = _ist_date(extrema["max"].as_py())
        minimum_date = batch_min if minimum_date is None else min(minimum_date, batch_min)
        maximum_date = batch_max if maximum_date is None else max(maximum_date, batch_max)
    if minimum_date is None or maximum_date is None:
        raise ValueError(f"Minute Parquet shard contains no timestamps: {parquet_path}")
    return minimum_date, maximum_date


def prepare_huggingface_dataset(
    *,
    repo_id: str = HUGGINGFACE_REPO_ID,
    split: Literal["minute", "day", "both"] = "both",
    symbols: Iterable[str] | None = None,
    output_root: str | Path = STOCK_DATA_DIR,
    download_dir: str | Path = HUGGINGFACE_CACHE_DIR,
    keep_shards: bool = False,
    batch_size: int = 65_536,
    holidays_csv: str | Path = NSE_HOLIDAYS_CSV,
) -> dict[str, int | str]:
    """Download and process HF Parquet shards incrementally, one stock at a time."""
    if split not in ("minute", "day", "both"):
        raise ValueError("split must be 'minute', 'day', or 'both'")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    requested_symbols = (
        {str(symbol).strip().upper() for symbol in symbols}
        if symbols is not None
        else None
    )
    if requested_symbols is not None:
        if not requested_symbols or "" in requested_symbols:
            raise ValueError("Provide at least one non-empty requested symbol.")
    holidays = load_nse_holidays(holidays_csv)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install the project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    selected_splits = ("minute", "day") if split in ("both", "day") else ("minute",)
    shard_paths = {
        current_split: sorted(
            path
            for path in repo_files
            if path.startswith(f"{current_split}/") and path.endswith(".parquet")
        )
        for current_split in selected_splits
    }
    empty_splits = [name for name, paths in shard_paths.items() if not paths]
    if empty_splits:
        raise ValueError(
            f"Repository contains no Parquet shards for split(s): {', '.join(empty_splits)}"
        )

    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_root)
    processed_symbols = 0
    source_rows = 0
    matched_symbols: set[str] = set()
    minute_date_min: date | None = None
    minute_date_max: date | None = None

    if split == "day":
        for shard_path in shard_paths["minute"]:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=shard_path,
                    repo_type="dataset",
                    local_dir=download_path,
                )
            )
            try:
                shard_min, shard_max = _scan_minute_date_range(
                    local_path,
                    batch_size=batch_size,
                )
                minute_date_min = (
                    shard_min
                    if minute_date_min is None
                    else min(minute_date_min, shard_min)
                )
                minute_date_max = (
                    shard_max
                    if minute_date_max is None
                    else max(minute_date_max, shard_max)
                )
            finally:
                if not keep_shards:
                    local_path.unlink(missing_ok=True)

    processing_splits = ("minute", "day") if split == "both" else (split,)
    for current_split in processing_splits:
        if current_split == "day" and (
            minute_date_min is None or minute_date_max is None
        ):
            raise ValueError("Could not determine the minute dataset's date coverage.")
        seen_symbols: set[str] = set()
        for shard_path in shard_paths[current_split]:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=shard_path,
                    repo_type="dataset",
                    local_dir=download_path,
                )
            )
            try:
                count, rows, shard_min, shard_max = _process_parquet_shard(
                    local_path,
                    split=current_split,
                    output_root=output_path,
                    selected_symbols=requested_symbols,
                    seen_symbols=seen_symbols,
                    batch_size=batch_size,
                    holidays=holidays,
                    date_range=(
                        (minute_date_min, minute_date_max)
                        if current_split == "day"
                        else None
                    ),
                )
                processed_symbols += count
                source_rows += rows
                matched_symbols.update(seen_symbols)
                if current_split == "minute" and shard_min is not None:
                    minute_date_min = (
                        shard_min if minute_date_min is None else min(minute_date_min, shard_min)
                    )
                if current_split == "minute" and shard_max is not None:
                    minute_date_max = (
                        shard_max if minute_date_max is None else max(minute_date_max, shard_max)
                    )
            finally:
                if not keep_shards:
                    local_path.unlink(missing_ok=True)

    if split in ("both", "day") and (
        minute_date_min is None or minute_date_max is None
    ):
        raise ValueError("Could not determine the minute dataset's date coverage.")

    if requested_symbols is not None:
        missing_symbols = requested_symbols - matched_symbols
        if missing_symbols:
            raise ValueError(
                "Requested symbols were not found in the selected split(s): "
                + ", ".join(sorted(missing_symbols))
            )
    result: dict[str, int | str] = {
        "processed_symbols": processed_symbols,
        "source_rows": source_rows,
        "shards": sum(len(paths) for paths in shard_paths.values()),
    }
    if minute_date_min is not None and minute_date_max is not None:
        result["minute_start_date"] = minute_date_min.isoformat()
        result["minute_end_date"] = minute_date_max.isoformat()
    return result
