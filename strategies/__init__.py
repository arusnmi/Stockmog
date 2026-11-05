from __future__ import annotations

import polars as pl

from . import box_breakout, mean_reversion, momentum, trend_following
from ._shared import build_signals
from ._types import StrategyName, StrategyStyle
from .catalog import (
    CATALOG,
    CATALOG_STRATEGIES,
    catalog_conditions,
    catalog_required_features,
    strategies_for_style,
)

STRATEGIES: tuple[StrategyName, ...] = (
    "box_breakout",
    "momentum",
    "trend_following",
    "mean_reversion",
)
_STRATEGY_FEATURES: dict[StrategyName, set[str]] = {
    "box_breakout": {"box_high", "box_low", "box_width", "relative_volume"},
    "momentum": {"rsi_14", "macd_hist", "relative_volume"},
    "trend_following": {"ema_9", "ema_20", "ema_50", "adx_14"},
    "mean_reversion": {"bb_lower", "bb_upper", "rsi_14", "adx_14"},
}


def generate_strategy_signals(
    frame: pl.DataFrame,
    *,
    strategy: StrategyName,
    style: StrategyStyle,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Generate close-of-bar signals with explicit entry, stop, and target prices."""
    if style not in ("intraday", "swing"):
        raise ValueError("style must be 'intraday' or 'swing'")

    if strategy in CATALOG:
        required_features = catalog_required_features(strategy)
        missing_features = required_features - set(frame.columns)
        if missing_features:
            raise ValueError(
                f"{strategy} requires engineered feature columns: "
                + ", ".join(sorted(missing_features))
            )
        long_condition, short_condition, reason = catalog_conditions(
            strategy,
            style=style,
        )
        return build_signals(
            frame,
            strategy=strategy,
            style=style,
            long_condition=long_condition,
            short_condition=short_condition,
            reason=reason,
            symbol=symbol,
        )

    strategy_rules = {
        "box_breakout": box_breakout.conditions,
        "momentum": momentum.conditions,
        "trend_following": trend_following.conditions,
        "mean_reversion": mean_reversion.conditions,
    }
    try:
        long_condition, short_condition, reason = strategy_rules[strategy](style)
    except KeyError as exc:
        raise ValueError(f"Unsupported strategy: {strategy!r}") from exc

    required_features = _STRATEGY_FEATURES[strategy] | {"timestamp", "close", "atr_14"}
    if strategy == "trend_following" and style == "intraday":
        required_features.add("vwap")
    missing_features = required_features - set(frame.columns)
    if missing_features:
        raise ValueError(
            f"{strategy} requires engineered feature columns: "
            + ", ".join(sorted(missing_features))
        )

    return build_signals(
        frame,
        strategy=strategy,
        style=style,
        long_condition=long_condition,
        short_condition=short_condition,
        reason=reason,
        symbol=symbol,
    )


def generate_all_signals(
    frame: pl.DataFrame,
    *,
    style: StrategyStyle,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Generate signals independently for all four strategy families."""
    signals = [
        generate_strategy_signals(
            frame,
            strategy=strategy,
            style=style,
            symbol=symbol,
        )
        for strategy in STRATEGIES
    ]
    return pl.concat(signals).sort(["timestamp", "strategy", "direction"])


def generate_all_catalog_signals(
    frame: pl.DataFrame,
    *,
    style: StrategyStyle,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Generate signals for catalog setups supported by the selected bar style."""
    signals = [
        generate_strategy_signals(
            frame,
            strategy=strategy,
            style=style,
            symbol=symbol,
        )
        for strategy in strategies_for_style(style)
    ]
    return pl.concat(signals).sort(["timestamp", "strategy", "direction"])


__all__ = [
    "CATALOG",
    "CATALOG_STRATEGIES",
    "STRATEGIES",
    "StrategyName",
    "StrategyStyle",
    "generate_all_catalog_signals",
    "generate_all_signals",
    "generate_strategy_signals",
    "strategies_for_style",
]
