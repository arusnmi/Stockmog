from __future__ import annotations

import polars as pl

from ._types import StrategyStyle


def conditions(style: StrategyStyle) -> tuple[pl.Expr, pl.Expr, str]:
    long_condition = (
        (pl.col("ema_9") > pl.col("ema_20"))
        & (pl.col("ema_20") > pl.col("ema_50"))
        & (pl.col("close") > pl.col("ema_20"))
        & (pl.col("adx_14") >= 20)
    )
    short_condition = (
        (pl.col("ema_9") < pl.col("ema_20"))
        & (pl.col("ema_20") < pl.col("ema_50"))
        & (pl.col("close") < pl.col("ema_20"))
        & (pl.col("adx_14") >= 20)
    )
    if style == "intraday":
        long_condition &= pl.col("close") > pl.col("vwap")
        short_condition &= pl.col("close") < pl.col("vwap")
        reason = "EMA alignment and ADX >= 20, confirmed by price relative to VWAP."
    else:
        reason = "EMA alignment, price relative to EMA 20, and ADX >= 20."
    return long_condition, short_condition, reason
