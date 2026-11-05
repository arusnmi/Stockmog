import unittest
from datetime import datetime, timedelta, timezone

import polars as pl

from backtesting import BacktestConfig, run_backtest
from backtesting.execution_simulator import (
    ExecutionCosts,
    calculate_quantity,
    resolve_exit,
    session_date,
)


def _bars(
    *,
    highs: list[float],
    lows: list[float],
    opens: list[float] | None = None,
    closes: list[float] | None = None,
    dates: list[datetime] | None = None,
) -> pl.DataFrame:
    count = len(highs)
    timestamps = dates or [
        datetime(2024, 1, 2, 9, 15) + timedelta(minutes=5 * index)
        for index in range(count)
    ]
    return pl.DataFrame(
        {
            "symbol": ["TEST"] * count,
            "timestamp": timestamps,
            "open": opens or [100.0] * count,
            "high": highs,
            "low": lows,
            "close": closes or [100.0] * count,
        },
        schema_overrides={
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
    )


def _signals(*, rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["TEST"] * len(rows),
            "timestamp": [row["timestamp"] for row in rows],
            "strategy": ["box_breakout"] * len(rows),
            "direction": [row.get("direction", "long") for row in rows],
            "entry_price": [row.get("entry_price", 100.0) for row in rows],
            "stop_price": [row.get("stop_price", 99.0) for row in rows],
            "target_price": [row.get("target_price", 102.0) for row in rows],
            "reward_risk": [row.get("reward_risk", 2.0) for row in rows],
        }
    )


class BacktestingTests(unittest.TestCase):
    def test_long_enters_next_bar_and_exits_at_target_after_costs(self):
        timestamps = [
            datetime(2024, 1, 2, 9, 15) + timedelta(minutes=5 * index)
            for index in range(3)
        ]
        bars = _bars(
            highs=[100.5, 101.0, 103.0],
            lows=[99.5, 99.5, 100.0],
            closes=[100.0, 100.5, 102.0],
            dates=timestamps,
        )
        signal = _signals(rows=[{"timestamp": timestamps[0]}])
        config = BacktestConfig(
            initial_capital=10_000,
            risk_per_trade=0.01,
            max_open_risk=0.02,
            max_daily_loss=0.1,
            max_drawdown=0.2,
            costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
        )

        result = run_backtest(bars, signal, style="swing", config=config)

        self.assertEqual(result.trades.height, 1)
        trade = result.trades.row(0, named=True)
        self.assertEqual(trade["entry_timestamp"], timestamps[1])
        self.assertEqual(trade["exit_timestamp"], timestamps[2])
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertEqual(trade["quantity"], 100)
        self.assertEqual(trade["gross_pnl"], 200.0)
        self.assertEqual(trade["net_pnl"], 200.0)
        self.assertEqual(result.summary["ending_equity"], 10_200.0)

    def test_stop_wins_when_both_barriers_are_touched(self):
        exit_fill = resolve_exit(
            direction=1,
            open_price=100,
            high=103,
            low=98,
            stop_price=99,
            target_price=102,
        )
        self.assertIsNotNone(exit_fill)
        self.assertEqual(exit_fill.reason, "stop_loss")
        self.assertEqual(exit_fill.price, 99)

    def test_timezone_aware_bar_timestamps_use_indian_trading_date(self):
        self.assertEqual(
            session_date(datetime(2024, 1, 1, 18, 30, tzinfo=timezone.utc)),
            datetime(2024, 1, 2).date(),
        )

    def test_yearly_report_uses_indian_date_for_utc_daily_timestamps(self):
        timestamps = [
            datetime(2023, 12, 31, 18, 25, tzinfo=timezone.utc),
            datetime(2023, 12, 31, 18, 30, tzinfo=timezone.utc),
        ]
        bars = _bars(
            highs=[100.5, 103],
            lows=[99.5, 99.5],
            dates=timestamps,
        )
        signal = _signals(rows=[{"timestamp": timestamps[0]}])
        result = run_backtest(
            bars,
            signal,
            style="swing",
            config=BacktestConfig(
                initial_capital=10_000,
                risk_per_trade=0.005,
                max_open_risk=0.01,
                max_daily_loss=0.1,
                max_drawdown=0.2,
                costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
            ),
        )
        self.assertEqual(result.yearly["year"].to_list(), [2023, 2024])
        self.assertEqual(result.yearly["trade_count"].to_list(), [0, 1])

    def test_gap_through_stop_fills_at_worse_open(self):
        exit_fill = resolve_exit(
            direction=1,
            open_price=97,
            high=100,
            low=96,
            stop_price=99,
            target_price=102,
        )
        self.assertEqual(exit_fill.reason, "stop_loss")
        self.assertEqual(exit_fill.price, 97)

    def test_short_trade_applies_adverse_slippage_on_both_sides(self):
        timestamps = [
            datetime(2024, 1, 2, 9, 15) + timedelta(minutes=5 * index)
            for index in range(3)
        ]
        bars = _bars(
            opens=[100, 100, 97],
            highs=[100.5, 100.5, 100],
            lows=[99.5, 99.5, 96],
            closes=[100, 100, 97],
            dates=timestamps,
        )
        signal = _signals(
            rows=[
                {
                    "timestamp": timestamps[0],
                    "direction": "short",
                    "stop_price": 101.0,
                    "target_price": 98.0,
                }
            ]
        )
        config = BacktestConfig(
            initial_capital=10_000,
            risk_per_trade=0.01,
            max_open_risk=0.02,
            max_daily_loss=0.1,
            max_drawdown=0.2,
            costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=10),
        )

        result = run_backtest(bars, signal, style="swing", config=config)

        trade = result.trades.row(0, named=True)
        self.assertEqual(trade["direction"], "short")
        self.assertLess(trade["entry_price"], 100)
        self.assertGreater(trade["exit_price"], 97)
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertGreater(trade["slippage_cost"], 0)
        self.assertEqual(
            trade["transaction_cost"],
            trade["fees"] + trade["slippage_cost"],
        )
        self.assertAlmostEqual(
            trade["net_pnl"],
            trade["gross_pnl"] - trade["transaction_cost"],
        )
        self.assertAlmostEqual(
            result.summary["transaction_costs"],
            result.summary["fees"] + result.summary["slippage_costs"],
        )

    def test_intraday_positions_are_closed_at_session_end(self):
        dates = [datetime(2024, 1, 2, 15, 20), datetime(2024, 1, 2, 15, 25)]
        bars = _bars(
            highs=[100.5, 101],
            lows=[99.5, 99.5],
            closes=[100, 100.5],
            dates=dates,
        )
        result = run_backtest(
            bars,
            _signals(rows=[{"timestamp": dates[0]}]),
            style="intraday",
            config=BacktestConfig(
                initial_capital=10_000,
                risk_per_trade=0.01,
                max_open_risk=0.02,
                costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
            ),
        )
        self.assertEqual(result.trades["exit_reason"].to_list(), ["session_end"])
        self.assertEqual(result.trades["exit_timestamp"][0], dates[1])

    def test_intraday_signal_at_session_close_is_not_carried_to_next_day(self):
        dates = [datetime(2024, 1, 2, 15, 25), datetime(2024, 1, 3, 9, 15)]
        bars = _bars(highs=[100.5, 101], lows=[99.5, 99.5], dates=dates)
        result = run_backtest(
            bars,
            _signals(rows=[{"timestamp": dates[0]}]),
            style="intraday",
        )
        self.assertTrue(result.trades.is_empty())

    def test_portfolio_risk_limit_rejects_an_additional_position(self):
        timestamps = [
            datetime(2024, 1, 2, 9, 15),
            datetime(2024, 1, 2, 9, 20),
        ]
        bars = pl.concat(
            [
                _bars(highs=[100.5, 100.5], lows=[99.5, 99.5], dates=timestamps),
                _bars(highs=[100.5, 100.5], lows=[99.5, 99.5], dates=timestamps).with_columns(
                    pl.lit("OTHER").alias("symbol")
                ),
            ]
        )
        signals = pl.concat(
            [
                _signals(rows=[{"timestamp": timestamps[0]}]),
                _signals(rows=[{"timestamp": timestamps[0]}]).with_columns(
                    pl.lit("OTHER").alias("symbol")
                ),
            ]
        )
        result = run_backtest(
            bars,
            signals,
            style="swing",
            config=BacktestConfig(
                initial_capital=10_000,
                risk_per_trade=0.005,
                max_open_risk=0.005,
                max_daily_loss=0.1,
                max_drawdown=0.2,
                max_positions=3,
                costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
            ),
        )
        self.assertEqual(result.trades.height, 1)

    def test_daily_loss_limit_halts_new_entries_after_a_stopped_trade(self):
        timestamps = [
            datetime(2024, 1, 2, 9, 15),
            datetime(2024, 1, 2, 9, 20),
            datetime(2024, 1, 2, 9, 25),
        ]
        first_symbol = _bars(
            highs=[100.5, 100.5, 100.5],
            lows=[99.5, 98.0, 99.5],
            dates=timestamps,
        )
        second_symbol = first_symbol.with_columns(pl.lit("OTHER").alias("symbol"))
        bars = pl.concat([first_symbol, second_symbol])
        first_signal = _signals(rows=[{"timestamp": timestamps[0]}])
        second_signal = _signals(rows=[{"timestamp": timestamps[1]}]).with_columns(
            pl.lit("OTHER").alias("symbol")
        )
        result = run_backtest(
            bars,
            pl.concat([first_signal, second_signal]),
            style="swing",
            config=BacktestConfig(
                initial_capital=10_000,
                risk_per_trade=0.005,
                max_open_risk=0.01,
                max_daily_loss=0.005,
                max_drawdown=0.2,
                max_positions=3,
                costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
            ),
        )
        self.assertEqual(result.trades.height, 1)
        self.assertEqual(result.trades["exit_reason"][0], "stop_loss")

    def test_costs_and_risk_configuration_are_validated(self):
        with self.assertRaisesRegex(ValueError, "not negative"):
            ExecutionCosts(commission_bps=-1)
        with self.assertRaisesRegex(ValueError, "less than 10,000"):
            ExecutionCosts(slippage_bps=10_000)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            BacktestConfig(risk_per_trade=0.02, max_open_risk=0.01)
        quantity = calculate_quantity(
            equity=10_000,
            risk_fraction=0.01,
            entry_price=100,
            stop_distance=2,
            costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
        )
        self.assertEqual(quantity, 50)

    def test_invalid_signal_and_bar_rows_are_rejected(self):
        bars = _bars(highs=[100.5, 100.5], lows=[99.5, 99.5])
        signal = _signals(rows=[{"timestamp": bars["timestamp"][0]}])
        invalid_signal = signal.with_columns(pl.lit("up").alias("direction"))
        with self.assertRaisesRegex(ValueError, "invalid row"):
            run_backtest(bars, invalid_signal, style="swing")

        invalid_bars = bars.with_columns(
            pl.when(pl.col("timestamp") == bars["timestamp"][0])
            .then(0.0)
            .otherwise(pl.col("open"))
            .alias("open")
        )
        with self.assertRaisesRegex(ValueError, "invalid OHLC"):
            run_backtest(bars=invalid_bars, signals=signal, style="swing")

    def test_yearly_report_groups_closed_trades_by_exit_year(self):
        dates = [
            datetime(2023, 12, 29, 15, 20),
            datetime(2023, 12, 29, 15, 25),
            datetime(2024, 1, 2, 9, 15),
            datetime(2024, 1, 2, 9, 20),
        ]
        bars = _bars(
            highs=[100.5, 103, 100.5, 103],
            lows=[99.5, 99.5, 99.5, 99.5],
            dates=dates,
        )
        signals = _signals(rows=[{"timestamp": dates[0]}, {"timestamp": dates[2]}])
        result = run_backtest(
            bars,
            signals,
            style="swing",
            config=BacktestConfig(
                initial_capital=10_000,
                risk_per_trade=0.005,
                max_open_risk=0.02,
                max_daily_loss=0.1,
                max_drawdown=0.2,
                costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
            ),
        )
        self.assertEqual(result.yearly["year"].to_list(), [2023, 2024])
        self.assertEqual(result.yearly["trade_count"].to_list(), [1, 1])


if __name__ == "__main__":
    unittest.main()
