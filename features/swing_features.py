from __future__ import annotations

import polars as pl

from ._shared import (
    add_box_features,
    add_previous_day_levels,
    add_relative_volume,
    safe_divide,
)


def add_swing_features(
    frame: pl.LazyFrame,
    *,
    box_window: int = 10,
    relative_volume_window: int = 20,
) -> pl.LazyFrame:
    """Add daily swing features using prior sessions for boxes and volume baselines."""
    frame = add_previous_day_levels(frame)
    frame = add_box_features(frame, window=box_window, by_session=False)
    frame = add_relative_volume(
        frame,
        window=relative_volume_window,
        by_session=False,
    )
    return frame.with_columns(
        safe_divide(
            pl.col("close") - pl.col("previous_day_high"),
            pl.col("atr_14"),
        ).alias("previous_day_high_distance_atr"),
        safe_divide(
            pl.col("previous_day_low") - pl.col("close"),
            pl.col("atr_14"),
        ).alias("previous_day_low_distance_atr"),
    )
