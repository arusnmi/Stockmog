from __future__ import annotations

import polars as pl

from ._shared import safe_divide


def add_price_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Add candle-shape, return and gap features using only current and prior bars."""
    frame = frame.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("_close_change"),
        pl.col("close").shift(1).alias("_previous_close"),
    )
    return frame.with_columns(
        (safe_divide(pl.col("close"), pl.col("_previous_close")) - 1.0).alias(
            "pct_change"
        ),
        (pl.col("close") - pl.col("open")).alias("candle_body"),
        safe_divide(
            pl.col("close") - pl.col("open"),
            pl.col("open"),
        ).alias("candle_body_pct"),
        (pl.col("high") - pl.max_horizontal("open", "close")).alias("upper_wick"),
        (pl.min_horizontal("open", "close") - pl.col("low")).alias("lower_wick"),
        (pl.col("high") - pl.col("low")).alias("candle_range"),
        safe_divide(pl.col("high") - pl.col("low"), pl.col("close")).alias(
            "candle_range_pct"
        ),
        (safe_divide(pl.col("open"), pl.col("_previous_close")) - 1.0).alias(
            "gap_pct"
        ),
    )


def _wilder_mean(column: str, period: int) -> pl.Expr:
    return pl.col(column).ewm_mean(
        alpha=1.0 / period,
        adjust=False,
        min_samples=period,
    )


def add_technical_indicators(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Add EMA, RSI, MACD, ATR, Bollinger and ADX features without future values."""
    for period in (9, 20, 50, 200):
        frame = frame.with_columns(
            pl.col("close")
            .ewm_mean(span=period, adjust=False, min_samples=period)
            .alias(f"ema_{period}")
        )

    frame = frame.with_columns(
        pl.when(pl.col("_close_change") > 0)
        .then(pl.col("_close_change"))
        .otherwise(0.0)
        .alias("_gain"),
        pl.when(pl.col("_close_change") < 0)
        .then(-pl.col("_close_change"))
        .otherwise(0.0)
        .alias("_loss"),
    )
    for period in (7, 14):
        frame = frame.with_columns(
            _wilder_mean("_gain", period).alias(f"_avg_gain_{period}"),
            _wilder_mean("_loss", period).alias(f"_avg_loss_{period}"),
        )
        relative_strength = pl.col(f"_avg_gain_{period}") / pl.col(f"_avg_loss_{period}")
        frame = frame.with_columns(
            pl.when((pl.col(f"_avg_gain_{period}") == 0) & (pl.col(f"_avg_loss_{period}") == 0))
            .then(50.0)
            .when(pl.col(f"_avg_loss_{period}") == 0)
            .then(100.0)
            .otherwise(100.0 - 100.0 / (1.0 + relative_strength))
            .alias(f"rsi_{period}")
        )

    frame = frame.with_columns(
        (
            pl.col("close").ewm_mean(span=12, adjust=False, min_samples=12)
            - pl.col("close").ewm_mean(span=26, adjust=False, min_samples=26)
        ).alias("macd"),
        (pl.col("high") - pl.col("low")).alias("_true_range_hl"),
        (pl.col("high") - pl.col("_previous_close")).abs().alias("_true_range_high_gap"),
        (pl.col("low") - pl.col("_previous_close")).abs().alias("_true_range_low_gap"),
    )
    frame = frame.with_columns(
        pl.col("macd")
        .ewm_mean(span=9, adjust=False, min_samples=9)
        .alias("macd_signal"),
        pl.max_horizontal(
            "_true_range_hl",
            "_true_range_high_gap",
            "_true_range_low_gap",
        ).alias("_true_range"),
    ).with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
        _wilder_mean("_true_range", 14).alias("atr_14"),
    )

    frame = frame.with_columns(
        safe_divide(pl.col("atr_14"), pl.col("close")).alias("atr_pct"),
        pl.col("close").rolling_mean(window_size=20, min_samples=20).alias("bb_mid"),
        pl.col("close")
        .rolling_std(window_size=20, min_samples=20, ddof=0)
        .alias("_bb_std"),
    ).with_columns(
        (pl.col("bb_mid") + 2.0 * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("bb_mid") - 2.0 * pl.col("_bb_std")).alias("bb_lower"),
    ).with_columns(
        safe_divide(
            pl.col("bb_upper") - pl.col("bb_lower"),
            pl.col("bb_mid"),
        ).alias("bb_width"),
        safe_divide(
            pl.col("close") - pl.col("bb_lower"),
            pl.col("bb_upper") - pl.col("bb_lower"),
        ).alias("bb_pct_b"),
    )

    frame = frame.with_columns(
        (pl.col("high") - pl.col("high").shift(1)).alias("_high_change"),
        (pl.col("low").shift(1) - pl.col("low")).alias("_low_change"),
    ).with_columns(
        pl.when(
            (pl.col("_high_change") > pl.col("_low_change"))
            & (pl.col("_high_change") > 0)
        )
        .then(pl.col("_high_change"))
        .otherwise(0.0)
        .alias("_plus_dm"),
        pl.when(
            (pl.col("_low_change") > pl.col("_high_change"))
            & (pl.col("_low_change") > 0)
        )
        .then(pl.col("_low_change"))
        .otherwise(0.0)
        .alias("_minus_dm"),
    ).with_columns(
        (
            100.0
            * safe_divide(
                _wilder_mean("_plus_dm", 14),
                _wilder_mean("_true_range", 14),
            )
        ).alias("_plus_di"),
        (
            100.0
            * safe_divide(
                _wilder_mean("_minus_dm", 14),
                _wilder_mean("_true_range", 14),
            )
        ).alias("_minus_di"),
    ).with_columns(
        (
            100.0
            * safe_divide(
                (pl.col("_plus_di") - pl.col("_minus_di")).abs(),
                pl.col("_plus_di") + pl.col("_minus_di"),
            )
        )
        .alias("_dx")
    ).with_columns(
        _wilder_mean("_dx", 14).alias("adx_14"),
    )

    return frame.drop(
        "_close_change",
        "_previous_close",
        "_gain",
        "_loss",
        "_avg_gain_7",
        "_avg_loss_7",
        "_avg_gain_14",
        "_avg_loss_14",
        "_true_range_hl",
        "_true_range_high_gap",
        "_true_range_low_gap",
        "_true_range",
        "_bb_std",
        "_high_change",
        "_low_change",
        "_plus_dm",
        "_minus_dm",
        "_plus_di",
        "_minus_di",
        "_dx",
    )
