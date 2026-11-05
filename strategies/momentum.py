from __future__ import annotations

import polars as pl

from ._types import StrategyStyle


def conditions(style: StrategyStyle) -> tuple[pl.Expr, pl.Expr, str]:
    relative_volume_threshold = 1.5 if style == "intraday" else 1.3
    long_condition = (
        (pl.col("rsi_14") >= 60)
        & (pl.col("macd_hist") > 0)
        & (pl.col("relative_volume") >= relative_volume_threshold)
    )
    short_condition = (
        (pl.col("rsi_14") <= 40)
        & (pl.col("macd_hist") < 0)
        & (pl.col("relative_volume") >= relative_volume_threshold)
    )
    reason = (
        "RSI confirms directional momentum, MACD histogram agrees, and "
        f"relative volume is >= {relative_volume_threshold:g}."
    )
    return long_condition, short_condition, reason
