from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

_REQUIRED_TRAINING_COLUMNS = {"timestamp", "open", "high", "low", "close"}


def scan_training_csv(file_path: str | Path) -> pl.LazyFrame:
    """Return a low-memory lazy scan of a normalized OHLCV CSV for training."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    if not (
        path.suffix.lower() == ".csv"
        or path.name.lower().endswith(".csv.gz")
    ):
        raise ValueError(f"Training data must be a CSV file: {path}")

    frame = pl.scan_csv(
        path,
        try_parse_dates=True,
        low_memory=True,
    )
    columns = set(frame.collect_schema().names())
    missing_columns = _REQUIRED_TRAINING_COLUMNS - columns
    if missing_columns:
        raise ValueError(
            "Training CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    return frame


def inspect_training_csv(file_path: str | Path) -> dict[str, Any]:
    """Collect only summary statistics from a lazy CSV scan."""
    frame = scan_training_csv(file_path)
    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("timestamp").min().alias("first_timestamp"),
        pl.col("timestamp").max().alias("last_timestamp"),
    ).collect()
    rows = summary["rows"][0]
    return {
        "file": str(Path(file_path)),
        "rows": rows,
        "first_timestamp": str(summary["first_timestamp"][0]) if rows else None,
        "last_timestamp": str(summary["last_timestamp"][0]) if rows else None,
        "columns": frame.collect_schema().names(),
    }
