from __future__ import annotations

import polars as pl

from ._types import StrategyStyle


def conditions(style: StrategyStyle) -> tuple[pl.Expr, pl.Expr, str]:
    del style
    long_condition = (
        (pl.col("close") <= pl.col("bb_lower"))
        & (pl.col("rsi_14") <= 30)
        & (pl.col("adx_14") < 25)
    )
    short_condition = (
        (pl.col("close") >= pl.col("bb_upper"))
        & (pl.col("rsi_14") >= 70)
        & (pl.col("adx_14") < 25)
    )
    reason = (
        "Close is outside the Bollinger Band with an extreme RSI in a "
        "non-trending (ADX < 25) market; experimental mean-reversion setup."
    )
    return long_condition, short_condition, reason
