from __future__ import annotations

import polars as pl

from ._shared import (
    add_box_features,
    add_previous_day_levels,
    add_relative_volume,
    add_volume_weighted_features,
    safe_divide,
)


def add_intraday_features(
    frame: pl.LazyFrame,
    *,
    box_window: int = 12,
    relative_volume_window: int = 20,
    opening_range_bars: int = 3,
) -> pl.LazyFrame:
    """Add intraday session-aware features; opening range appears only after it closes."""
    if opening_range_bars < 1:
        raise ValueError("opening range must contain at least one candle")

    frame = add_previous_day_levels(frame)
    frame = add_box_features(frame, window=box_window, by_session=True)
    frame = add_relative_volume(
        frame,
        window=relative_volume_window,
        by_session=True,
    )
    frame = add_volume_weighted_features(frame)

    frame = frame.with_columns(
        pl.col("timestamp").cum_count().over("_session_date").alias("_session_bar")
    )
    opening_high = (
        pl.when(pl.col("_session_bar") <= opening_range_bars)
        .then(pl.col("high"))
        .otherwise(None)
        .max()
        .over("_session_date")
    )
    opening_low = (
        pl.when(pl.col("_session_bar") <= opening_range_bars)
        .then(pl.col("low"))
        .otherwise(None)
        .min()
        .over("_session_date")
    )
    frame = frame.with_columns(
        pl.when(pl.col("_session_bar") >= opening_range_bars)
        .then(opening_high)
        .otherwise(None)
        .alias("opening_range_high"),
        pl.when(pl.col("_session_bar") >= opening_range_bars)
        .then(opening_low)
        .otherwise(None)
        .alias("opening_range_low"),
    ).with_columns(
        safe_divide(
            pl.col("close") - pl.col("opening_range_high"),
            pl.col("atr_14"),
        ).alias("opening_range_distance_up_atr"),
        safe_divide(
            pl.col("opening_range_low") - pl.col("close"),
            pl.col("atr_14"),
        ).alias("opening_range_distance_down_atr"),
        safe_divide(
            pl.col("close") - pl.col("previous_day_high"),
            pl.col("atr_14"),
        ).alias("previous_day_high_distance_atr"),
        safe_divide(
            pl.col("previous_day_low") - pl.col("close"),
            pl.col("atr_14"),
        ).alias("previous_day_low_distance_atr"),
    )
    return frame.drop("_session_bar")
