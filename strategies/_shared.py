from __future__ import annotations

import polars as pl

from . import _types


def build_signals(
    frame: pl.DataFrame,
    *,
    strategy: _types.StrategyName,
    style: _types.StrategyStyle,
    long_condition: pl.Expr,
    short_condition: pl.Expr,
    reason: str,
    symbol: str | None,
    stop_atr_multiple: float = 1.0,
    reward_risk: float = 2.0,
) -> pl.DataFrame:
    if stop_atr_multiple <= 0 or reward_risk <= 0:
        raise ValueError("stop_atr_multiple and reward_risk must be greater than zero")

    required = {"timestamp", "close", "atr_14"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Strategy input is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if not isinstance(frame.schema["timestamp"], pl.Datetime):
        raise ValueError("Strategy timestamps must be datetimes.")

    frame = frame.sort("timestamp").with_columns(
        long_condition.fill_null(False).alias("_long_setup"),
        short_condition.fill_null(False).alias("_short_setup"),
    )
    if style == "intraday":
        session = pl.col("timestamp").dt.date()
        previous_long = pl.col("_long_setup").shift(1).over(session).fill_null(False)
        previous_short = pl.col("_short_setup").shift(1).over(session).fill_null(False)
    else:
        previous_long = pl.col("_long_setup").shift(1).fill_null(False)
        previous_short = pl.col("_short_setup").shift(1).fill_null(False)

    frame = frame.with_columns(
        (pl.col("_long_setup") & ~previous_long).alias("_long_signal"),
        (pl.col("_short_setup") & ~previous_short).alias("_short_signal"),
    ).with_columns(
        pl.when(pl.col("_long_signal"))
        .then(pl.lit("long"))
        .when(pl.col("_short_signal"))
        .then(pl.lit("short"))
        .otherwise(None)
        .alias("direction"),
        pl.col("close").cast(pl.Float64, strict=False).alias("_entry_price"),
        pl.col("atr_14").cast(pl.Float64, strict=False).alias("_atr"),
    )
    valid_atr = (
        pl.col("_atr").is_not_null()
        & pl.col("_atr").is_finite()
        & (pl.col("_atr") > 0)
        & pl.col("_entry_price").is_not_null()
        & pl.col("_entry_price").is_finite()
    )
    valid_signal = pl.col("direction").is_not_null() & valid_atr
    risk = pl.col("_atr") * stop_atr_multiple
    frame = frame.with_columns(
        pl.when(pl.col("direction") == "long")
        .then(pl.col("_entry_price") - risk)
        .when(pl.col("direction") == "short")
        .then(pl.col("_entry_price") + risk)
        .otherwise(None)
        .alias("stop_price"),
        pl.when(pl.col("direction") == "long")
        .then(pl.col("_entry_price") + risk * reward_risk)
        .when(pl.col("direction") == "short")
        .then(pl.col("_entry_price") - risk * reward_risk)
        .otherwise(None)
        .alias("target_price"),
    )
    symbol_expression = (
        pl.col("symbol").cast(pl.String)
        if "symbol" in frame.columns
        else pl.lit(symbol or "UNKNOWN")
    )
    return (
        frame.filter(valid_signal)
        .select(
            symbol_expression.alias("symbol"),
            "timestamp",
            pl.lit(strategy).alias("strategy"),
            pl.lit(style).alias("style"),
            "direction",
            pl.col("_entry_price").alias("entry_price"),
            "stop_price",
            "target_price",
            (risk).alias("risk_per_unit"),
            pl.lit(reward_risk).alias("reward_risk"),
            pl.lit(reason).alias("reason"),
        )
        .sort("timestamp")
    )
