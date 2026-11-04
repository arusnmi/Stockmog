from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from config import settings
from data_pipeline.csv_dataset import scan_training_csv

from ._shared import add_market_context
from .intraday_features import add_intraday_features
from .swing_features import add_swing_features
from .technical_indicators import add_price_features, add_technical_indicators

FeatureStyle = Literal["intraday", "swing"]


def build_features(
    input_path: str | Path,
    *,
    style: FeatureStyle,
    box_window: int | None = None,
    relative_volume_window: int = 20,
    opening_range_bars: int = 3,
    market_index_path: str | Path | None = None,
    volatility_index_path: str | Path | None = None,
) -> pl.LazyFrame:
    """Construct a deterministic, lazy feature frame from one OHLCV CSV."""
    if style not in ("intraday", "swing"):
        raise ValueError("style must be 'intraday' or 'swing'")

    source = scan_training_csv(input_path)
    columns = set(source.collect_schema().names())
    frame = source.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"), strict=True),
        *(
            [pl.col("volume").cast(pl.Float64, strict=False)]
            if "volume" in columns
            else [pl.lit(None, dtype=pl.Float64).alias("volume")]
        ),
        *[
            pl.col(name).cast(pl.Float64, strict=False)
            for name in ("open", "high", "low", "close")
        ],
    ).sort("timestamp")
    frame = frame.with_columns(pl.col("timestamp").dt.date().alias("_session_date"))
    frame = add_price_features(frame)
    frame = add_technical_indicators(frame)

    if style == "intraday":
        frame = add_intraday_features(
            frame,
            box_window=box_window if box_window is not None else 12,
            relative_volume_window=relative_volume_window,
            opening_range_bars=opening_range_bars,
        )
    else:
        frame = add_swing_features(
            frame,
            box_window=box_window if box_window is not None else 10,
            relative_volume_window=relative_volume_window,
        )

    frame = add_market_context(
        frame,
        market_index_path=str(market_index_path) if market_index_path is not None else None,
        volatility_index_path=(
            str(volatility_index_path) if volatility_index_path is not None else None
        ),
    )
    return frame.drop("_session_date")


def write_features(
    input_path: str | Path,
    *,
    style: FeatureStyle,
    output_path: str | Path | None = None,
    box_window: int | None = None,
    relative_volume_window: int = 20,
    opening_range_bars: int = 3,
    market_index_path: str | Path | None = None,
    volatility_index_path: str | Path | None = None,
) -> Path:
    """Stream engineered features to a CSV file without materializing the full dataset."""
    source_path = Path(input_path)
    if output_path is None:
        output = settings.FEATURES_DIR / style / f"{source_path.stem}_features.csv"
    else:
        output = Path(output_path)
    if output.suffix.lower() != ".csv":
        raise ValueError(f"Feature output must be a CSV file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.name}.tmp")

    try:
        build_features(
            source_path,
            style=style,
            box_window=box_window,
            relative_volume_window=relative_volume_window,
            opening_range_bars=opening_range_bars,
            market_index_path=market_index_path,
            volatility_index_path=volatility_index_path,
        ).sink_csv(temporary_output, null_value="")
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    return output
