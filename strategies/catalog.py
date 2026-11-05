from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict, cast

import polars as pl


class CatalogSetup(TypedDict):
    id: str
    name: str
    entry_hypothesis: str
    timeframes: str
    features: str
    family: str
    references: str
    exit_template: str


_CATALOG_PATH = Path(__file__).resolve().parents[1] / "docs" / "strategy-catalog.csv"
_EXPECTED_FIELDS = (
    "id",
    "name",
    "entry_hypothesis",
    "timeframes",
    "features",
    "family",
    "references",
    "exit_template",
)


def _load_catalog() -> dict[str, CatalogSetup]:
    with _CATALOG_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _EXPECTED_FIELDS:
            raise ValueError(f"Unexpected strategy catalog columns in {_CATALOG_PATH}")
        source_rows = list(reader)

    if len(source_rows) != 200:
        raise ValueError(f"Expected 200 setups in {_CATALOG_PATH}")
    rows: dict[str, CatalogSetup] = {}
    for expected_id, source_row in enumerate(source_rows, start=1):
        if any(source_row.get(field) is None for field in _EXPECTED_FIELDS):
            raise ValueError(f"Incomplete strategy catalog row {expected_id} in {_CATALOG_PATH}")
        setup = cast(
            CatalogSetup,
            {field: source_row[field] for field in _EXPECTED_FIELDS},
        )
        if setup["id"] != f"{expected_id:03d}":
            raise ValueError(
                f"Expected setup ID {expected_id:03d} at row {expected_id} in "
                f"{_CATALOG_PATH}, found {setup['id']!r}"
            )
        rows[f"catalog_{setup['id']}"] = setup
    return rows


CATALOG: dict[str, CatalogSetup] = _load_catalog()
CATALOG_STRATEGIES: tuple[str, ...] = tuple(CATALOG)


def strategies_for_style(style: str) -> tuple[str, ...]:
    if style not in ("intraday", "swing"):
        raise ValueError("style must be 'intraday' or 'swing'")
    timeframe_flag = "I" if style == "intraday" else "S"
    return tuple(
        strategy
        for strategy in CATALOG_STRATEGIES
        if timeframe_flag in CATALOG[strategy]["timeframes"]
    )


def _rule_family(setup: CatalogSetup) -> str:
    family = setup["family"].lower()
    name = setup["name"].lower()
    if "candlestick" in family or "candle" in name:
        return "candlestick"
    if "calendar" in family:
        return "calendar"
    if "intraday" in family:
        return "intraday"
    if "gap" in family or "gap" in name:
        return "gap"
    if "volume" in family:
        return "volume"
    if any(
        token in family
        for token in ("failed", "reversal", "reversion", "divergence", "exhaustion")
    ):
        return "reversal"
    if any(
        token in family
        for token in ("breakout", "breakdown", "range", "level", "support", "resistance")
    ):
        return "breakout"
    if "volatility" in family or "compression" in family:
        return "volatility"
    if any(token in family for token in ("trend", "momentum")):
        return "trend"
    return "trend"


def _direction_scope(setup: CatalogSetup) -> tuple[bool, bool]:
    description = f"{setup['name']} {setup['entry_hypothesis']}".lower()
    long_terms = ("buy", "long", "bullish", "uptrend", "positive")
    short_terms = ("short", "bearish", "downtrend", "downside", "negative", "sell")
    wants_long = any(term in description for term in long_terms)
    wants_short = any(term in description for term in short_terms)
    if wants_long == wants_short:
        return True, True
    return wants_long, wants_short


def catalog_conditions(
    strategy: str,
    *,
    style: str,
) -> tuple[pl.Expr, pl.Expr, str]:
    try:
        setup = CATALOG[strategy]
    except KeyError as exc:
        raise ValueError(f"Unsupported catalog strategy: {strategy!r}") from exc

    timeframe_flag = "I" if style == "intraday" else "S"
    if timeframe_flag not in setup["timeframes"]:
        raise ValueError(
            f"{strategy} ({setup['name']}) does not support {style} bars; "
            f"catalog timeframes are {setup['timeframes']}"
        )

    close = pl.col("close")
    previous_close = close.shift(1)
    atr = pl.col("atr_14")
    ema_fast = pl.col("ema_9")
    ema_slow = pl.col("ema_20")
    rsi = pl.col("rsi_14")
    box_high = pl.col("box_high")
    box_low = pl.col("box_low")
    relative_volume = pl.col("relative_volume")
    family = _rule_family(setup)

    if family == "candlestick":
        long_condition = (
            (close > pl.col("open"))
            & (close > pl.col("high").shift(1))
            & (pl.col("lower_wick") >= pl.col("candle_body").abs())
        )
        short_condition = (
            (close < pl.col("open"))
            & (close < pl.col("low").shift(1))
            & (pl.col("upper_wick") >= pl.col("candle_body").abs())
        )
    elif family == "calendar":
        # Calendar labels are adapted into a causal trend filter; no future calendar
        # return is used as a signal.
        long_condition = (close > ema_slow) & (pl.col("ret_20") > 0)
        short_condition = (close < ema_slow) & (pl.col("ret_20") < 0)
    elif family == "intraday":
        long_condition = (
            (close > pl.col("opening_range_high"))
            & (pl.col("ret_5") > 0)
            & (close > pl.col("vwap"))
        )
        short_condition = (
            (close < pl.col("opening_range_low"))
            & (pl.col("ret_5") < 0)
            & (close < pl.col("vwap"))
        )
    elif family == "gap":
        long_condition = (pl.col("gap_pct") >= 0.005) & (close > pl.col("open"))
        short_condition = (pl.col("gap_pct") <= -0.005) & (close < pl.col("open"))
    elif family == "volume":
        long_condition = (
            (relative_volume >= 1.5)
            & (close > pl.col("open"))
            & (pl.col("close_location") >= 0.7)
        )
        short_condition = (
            (relative_volume >= 1.5)
            & (close < pl.col("open"))
            & (pl.col("close_location") <= 0.3)
        )
    elif family == "reversal":
        long_condition = (
            (rsi <= 35)
            & (close > previous_close)
            & (close > pl.col("bb_lower"))
        )
        short_condition = (
            (rsi >= 65)
            & (close < previous_close)
            & (close < pl.col("bb_upper"))
        )
    elif family == "breakout":
        long_condition = (close > box_high + 0.1 * atr) & (relative_volume >= 1.2)
        short_condition = (close < box_low - 0.1 * atr) & (relative_volume >= 1.2)
    elif family == "volatility":
        long_condition = (
            (pl.col("bb_width") > pl.col("bb_width").shift(1))
            & (close > box_high)
            & (pl.col("ret_5") > 0)
        )
        short_condition = (
            (pl.col("bb_width") > pl.col("bb_width").shift(1))
            & (close < box_low)
            & (pl.col("ret_5") < 0)
        )
    else:
        long_condition = (
            (close > ema_fast)
            & (ema_fast > ema_slow)
            & (pl.col("macd_hist") > 0)
        )
        short_condition = (
            (close < ema_fast)
            & (ema_fast < ema_slow)
            & (pl.col("macd_hist") < 0)
        )

    allow_long, allow_short = _direction_scope(setup)
    if not allow_long:
        long_condition = pl.lit(False)
    if not allow_short:
        short_condition = pl.lit(False)
    return long_condition, short_condition, (
        f"Catalog setup {setup['id']}: {setup['name']} "
        f"(family template: {family}; research hypothesis, not a validated edge)"
    )


def catalog_required_features(strategy: str) -> set[str]:
    try:
        family = _rule_family(CATALOG[strategy])
    except KeyError as exc:
        raise ValueError(f"Unsupported catalog strategy: {strategy!r}") from exc

    requirements = {
        "candlestick": {"open", "high", "low", "close", "candle_body", "upper_wick", "lower_wick"},
        "calendar": {"close", "ema_20", "ret_20"},
        "intraday": {"close", "opening_range_high", "opening_range_low", "ret_5", "vwap"},
        "gap": {"open", "close", "gap_pct"},
        "volume": {"open", "close", "relative_volume", "close_location"},
        "reversal": {"close", "rsi_14", "bb_lower", "bb_upper"},
        "breakout": {"close", "atr_14", "box_high", "box_low", "relative_volume"},
        "volatility": {"close", "box_high", "box_low", "bb_width", "ret_5"},
        "trend": {"close", "ema_9", "ema_20", "macd_hist"},
    }
    return requirements[family] | {"timestamp", "atr_14"}
