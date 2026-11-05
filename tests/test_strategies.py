import unittest
from datetime import datetime, timedelta

import polars as pl

from strategies import (
    CATALOG,
    CATALOG_STRATEGIES,
    STRATEGIES,
    generate_all_catalog_signals,
    generate_all_signals,
    generate_strategy_signals,
    strategies_for_style,
)


def _features(*, rows: int = 4) -> pl.DataFrame:
    timestamps = [datetime(2024, 1, 2, 9, 15) + timedelta(minutes=5 * i) for i in range(rows)]
    close = [100.0, 103.0, 104.0, 95.0][:rows]
    return pl.DataFrame(
        {
            "symbol": ["TEST"] * rows,
            "timestamp": timestamps,
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "atr_14": [2.0] * rows,
            "box_high": [100.0] * rows,
            "box_low": [90.0, 90.0, 90.0, 96.0][:rows],
            "box_width": [10.0] * rows,
            "relative_volume": [2.0] * rows,
            "rsi_14": [50.0, 75.0, 65.0, 25.0][:rows],
            "macd_hist": [0.0, 1.0, 1.0, -1.0][:rows],
            "ema_9": [103.0, 104.0, 105.0, 90.0][:rows],
            "ema_20": [102.0, 102.0, 103.0, 96.0][:rows],
            "ema_50": [101.0, 101.0, 102.0, 100.0][:rows],
            "adx_14": [20.0] * rows,
            "vwap": [99.0, 102.0, 103.0, 96.0][:rows],
            "bb_lower": [95.0, 95.0, 95.0, 96.0][:rows],
            "bb_upper": [105.0, 102.0, 105.0, 105.0][:rows],
        }
    )


class TradingStrategyTests(unittest.TestCase):
    def test_box_breakout_emits_long_and_short_once_per_setup(self):
        signals = generate_strategy_signals(
            _features(),
            strategy="box_breakout",
            style="intraday",
        )

        self.assertEqual(signals["direction"].to_list(), ["long", "short"])
        self.assertEqual(signals["entry_price"].to_list(), [103.0, 95.0])
        self.assertEqual(signals["stop_price"].to_list(), [101.0, 97.0])
        self.assertEqual(signals["target_price"].to_list(), [107.0, 91.0])
        self.assertEqual(signals["risk_per_unit"].to_list(), [2.0, 2.0])

    def test_momentum_and_trend_following_use_separate_rules(self):
        frame = _features()
        momentum = generate_strategy_signals(
            frame,
            strategy="momentum",
            style="intraday",
        )
        trend = generate_strategy_signals(
            frame,
            strategy="trend_following",
            style="intraday",
        )

        self.assertEqual(momentum["direction"].to_list(), ["long", "short"])
        self.assertEqual(trend["direction"].to_list(), ["long", "short"])
        self.assertEqual(momentum["timestamp"].to_list(), trend["timestamp"].to_list())

    def test_mean_reversion_is_an_independent_experimental_rule(self):
        frame = _features().with_columns(
            pl.Series("rsi_14", [50.0, 75.0, 65.0, 25.0]),
            pl.Series("adx_14", [20.0] * 4),
        )
        signals = generate_strategy_signals(
            frame,
            strategy="mean_reversion",
            style="swing",
        )

        self.assertEqual(signals["direction"].to_list(), ["short", "long"])
        self.assertTrue(all("experimental" in reason for reason in signals["reason"]))

    def test_intraday_signal_setup_can_trigger_again_on_a_new_session(self):
        frame = pl.concat(
            [
                _features(rows=2),
                _features(rows=2).with_columns(
                    (pl.col("timestamp") + pl.duration(days=1)).alias("timestamp")
                ),
            ]
        )
        signals = generate_strategy_signals(
            frame,
            strategy="trend_following",
            style="intraday",
        )

        self.assertEqual(signals["direction"].to_list(), ["long", "long"])

    def test_all_strategies_have_reviewable_signal_columns(self):
        signals = generate_all_signals(_features(), style="intraday")

        self.assertEqual(set(signals["strategy"].unique()), set(STRATEGIES))
        self.assertEqual(
            signals.columns,
            [
                "symbol",
                "timestamp",
                "strategy",
                "style",
                "direction",
                "entry_price",
                "stop_price",
                "target_price",
                "risk_per_unit",
                "reward_risk",
                "reason",
            ],
        )

    def test_missing_strategy_features_raise_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "box_high"):
            generate_strategy_signals(
                _features().drop("box_high"),
                strategy="box_breakout",
                style="intraday",
            )

    def test_catalog_contains_200_setups_and_respects_timeframe_eligibility(self):
        self.assertEqual(len(CATALOG), 200)
        self.assertEqual(len(set(CATALOG_STRATEGIES)), 200)
        self.assertEqual(len(strategies_for_style("intraday")), 180)
        self.assertEqual(len(strategies_for_style("swing")), 144)
        self.assertEqual(
            set(strategies_for_style("intraday"))
            | set(strategies_for_style("swing")),
            set(CATALOG_STRATEGIES),
        )

    def test_catalog_strategy_emits_signals_for_existing_model_pipeline(self):
        frame = _features().with_columns(
            pl.Series("ema_9", [99.0, 101.0, 102.0, 90.0]),
            pl.Series("ema_20", [98.0, 100.0, 101.0, 91.0]),
            pl.Series("macd_hist", [0.0, 1.0, 1.0, -1.0]),
        )
        signals = generate_strategy_signals(
            frame,
            strategy="catalog_003",
            style="intraday",
        )

        self.assertEqual(signals["strategy"].to_list(), ["catalog_003"])
        self.assertEqual(signals["direction"].to_list(), ["long"])
        self.assertIn("Moving-average state cross", signals["reason"][0])

    def test_catalog_all_generates_only_timeframe_compatible_strategies(self):
        frame = _features().with_columns(
            pl.lit(105.0).alias("opening_range_high"),
            pl.lit(90.0).alias("opening_range_low"),
            pl.lit(0.01).alias("ret_5"),
            pl.lit(1.0).alias("ret_20"),
            pl.lit(0.8).alias("close_location"),
            pl.lit(0.1).alias("gap_pct"),
            pl.lit(0.02).alias("bb_width"),
            (pl.col("close") - pl.col("open")).alias("candle_body"),
            (pl.col("high") - pl.max_horizontal("open", "close")).alias(
                "upper_wick"
            ),
            (pl.min_horizontal("open", "close") - pl.col("low")).alias(
                "lower_wick"
            ),
        )
        signals = generate_all_catalog_signals(frame, style="intraday")
        self.assertTrue(
            set(signals["strategy"].unique()).issubset(
                set(strategies_for_style("intraday"))
            )
        )
        self.assertNotIn("catalog_001", signals["strategy"].to_list())

    def test_catalog_setup_rejects_unsupported_style(self):
        with self.assertRaisesRegex(ValueError, "does not support intraday"):
            generate_strategy_signals(
                _features(),
                strategy="catalog_001",
                style="intraday",
            )

    def test_signal_generation_rejects_nonpositive_atr_and_invalid_risk_settings(self):
        frame = _features().with_columns(pl.lit(0.0).alias("atr_14"))
        signals = generate_strategy_signals(
            frame,
            strategy="box_breakout",
            style="intraday",
        )
        self.assertTrue(signals.is_empty())

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            from strategies._shared import build_signals

            build_signals(
                _features(),
                strategy="box_breakout",
                style="intraday",
                long_condition=pl.lit(True),
                short_condition=pl.lit(False),
                reason="test",
                symbol=None,
                stop_atr_multiple=0,
            )


if __name__ == "__main__":
    unittest.main()
