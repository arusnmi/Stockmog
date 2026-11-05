from __future__ import annotations

import polars as pl

from ._types import StrategyStyle


def conditions(style: StrategyStyle) -> tuple[pl.Expr, pl.Expr, str]:
    relative_volume_threshold = 1.5 if style == "intraday" else 1.3
    long_condition = (
        (pl.col("close") > pl.col("box_high") + 0.1 * pl.col("atr_14"))
        & (pl.col("relative_volume") >= relative_volume_threshold)
        & (pl.col("box_width") > 0)
    )
    short_condition = (
        (pl.col("close") < pl.col("box_low") - 0.1 * pl.col("atr_14"))
        & (pl.col("relative_volume") >= relative_volume_threshold)
        & (pl.col("box_width") > 0)
    )
    reason = (
        "Close broke the prior range by at least 0.1 ATR with relative volume "
        f">= {relative_volume_threshold:g}."
    )
    return long_condition, short_condition, reason
