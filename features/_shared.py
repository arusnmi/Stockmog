from __future__ import annotations

import polars as pl

from data_pipeline.csv_dataset import scan_training_csv


def safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(
            denominator.is_not_null()
            & denominator.is_finite()
            & (denominator != 0)
        )
        .then(numerator / denominator)
        .otherwise(None)
    )


def add_previous_day_levels(frame: pl.LazyFrame) -> pl.LazyFrame:
    daily = (
        frame.group_by("_session_date")
        .agg(
            pl.col("high").max().alias("_day_high"),
            pl.col("low").min().alias("_day_low"),
        )
        .sort("_session_date")
        .with_columns(
            pl.col("_day_high").shift(1).alias("previous_day_high"),
            pl.col("_day_low").shift(1).alias("previous_day_low"),
        )
        .select("_session_date", "previous_day_high", "previous_day_low")
    )
    return frame.join(daily, on="_session_date", how="left")


def add_box_features(
    frame: pl.LazyFrame,
    *,
    window: int,
    by_session: bool,
) -> pl.LazyFrame:
    if window < 1:
        raise ValueError("box window must be at least one candle")

    high = pl.col("high").rolling_max(window_size=window, min_samples=window).shift(1)
    low = pl.col("low").rolling_min(window_size=window, min_samples=window).shift(1)
    if by_session:
        high = high.over("_session_date")
        low = low.over("_session_date")

    frame = frame.with_columns(
        high.alias("box_high"),
        low.alias("box_low"),
    ).with_columns(
        (pl.col("box_high") - pl.col("box_low")).alias("box_width"),
    )
    return frame.with_columns(
        safe_divide(pl.col("box_width"), pl.col("atr_14")).alias("box_width_atr"),
        safe_divide(pl.col("close") - pl.col("box_high"), pl.col("atr_14")).alias(
            "box_breakout_distance_up_atr"
        ),
        safe_divide(pl.col("box_low") - pl.col("close"), pl.col("atr_14")).alias(
            "box_breakout_distance_down_atr"
        ),
    )


def add_relative_volume(
    frame: pl.LazyFrame,
    *,
    window: int,
    by_session: bool,
) -> pl.LazyFrame:
    if window < 1:
        raise ValueError("relative-volume window must be at least one candle")

    baseline = (
        pl.col("volume")
        .rolling_mean(window_size=window, min_samples=window)
        .shift(1)
    )
    if by_session:
        baseline = baseline.over("_session_date")

    return frame.with_columns(
        safe_divide(pl.col("volume"), baseline).alias("relative_volume")
    )


def add_volume_weighted_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    frame = frame.with_columns(
        pl.col("volume").fill_null(0.0).alias("_safe_volume"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(
            "_typical_price"
        ),
    ).with_columns(
        (pl.col("_typical_price") * pl.col("_safe_volume"))
        .cum_sum()
        .over("_session_date")
        .alias("_session_price_volume"),
        pl.col("_safe_volume").cum_sum().over("_session_date").alias("_session_volume"),
    ).with_columns(
        pl.when(pl.col("_session_volume") > 0)
        .then(pl.col("_session_price_volume") / pl.col("_session_volume"))
        .otherwise(None)
        .alias("vwap")
    ).with_columns(
        (safe_divide(pl.col("close"), pl.col("vwap")) - 1.0).alias(
            "distance_from_vwap_pct"
        )
    )
    return frame.drop(
        "_safe_volume",
        "_typical_price",
        "_session_price_volume",
        "_session_volume",
    )


def add_market_context(
    frame: pl.LazyFrame,
    *,
    market_index_path: str | None,
    volatility_index_path: str | None,
) -> pl.LazyFrame:
    for path, prefix in (
        (market_index_path, "market_index"),
        (volatility_index_path, "volatility_index"),
    ):
        if path is None:
            continue

        context = (
            scan_training_csv(path)
            .select(
                pl.col("timestamp").cast(pl.Datetime("us"), strict=True),
                pl.col("close").cast(pl.Float64, strict=False),
            )
            .sort("timestamp")
            .with_columns(
                pl.col("close").alias(f"{prefix}_close"),
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias(
                    f"{prefix}_return_1"
                ),
            )
            .select(
                "timestamp",
                f"{prefix}_close",
                f"{prefix}_return_1",
            )
        )
        frame = frame.sort("timestamp").join_asof(
            context,
            on="timestamp",
            strategy="backward",
            allow_exact_matches=False,
        )
    return frame
