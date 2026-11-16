import unittest
from datetime import datetime, timedelta
from importlib.util import find_spec

import pandas as pd

from backtesting.execution_simulator import ExecutionCosts

if find_spec("backtrader") is not None:
    from backtesting.rl_execution import ExecutionConfig, simulate_episode


def bars(rows):
    return pd.DataFrame(
        [row[1:] for row in rows],
        index=pd.DatetimeIndex([datetime.fromisoformat(row[0]) for row in rows]),
        columns=["open", "high", "low", "close", "volume"],
    )


@unittest.skipUnless(find_spec("backtrader") is not None, "Backtrader is not installed")
class BacktraderExecutionTests(unittest.TestCase):
    def test_orders_fill_next_open_and_results_include_signed_costs(self):
        frame = bars([
            ("2024-01-02T09:15", 100, 101, 99, 100, 1000),
            ("2024-01-02T09:16", 110, 111, 104, 105, 1000),
            ("2024-01-02T09:17", 90, 91, 89, 90, 1000),
        ])
        actions = iter(["open_long", "close_long", "wait"])
        result = simulate_episode(
            frame,
            config=ExecutionConfig(style="intraday", interval="1m"),
            policy=lambda _: next(actions),
        )
        self.assertEqual(len(result.executions), 2)
        self.assertEqual(result.executions[0].submitted_at, frame.index[0])
        self.assertEqual(result.executions[0].executed_at, frame.index[1])
        self.assertAlmostEqual(result.executions[0].price, 110.055)
        self.assertAlmostEqual(result.executions[1].price, 89.955)
        self.assertAlmostEqual(result.final_equity, 48984.9995)
        self.assertEqual(result.formatted_summary()["profit_loss"], "-1015.00")
        self.assertEqual(result.formatted_summary()["profit_loss_percent"], "-2.03")

    def test_intraday_short_forced_to_close_at_same_session_next_open(self):
        frame = bars([
            ("2024-01-02T14:45", 100, 101, 99, 100, 1000),
            ("2024-01-02T15:00", 100, 101, 99, 100, 1000),
            ("2024-01-02T15:15", 95, 96, 94, 95, 1000),
        ])
        result = simulate_episode(
            frame,
            config=ExecutionConfig(style="intraday", interval="15m"),
            policy=lambda obs: "open_short" if obs.timestamp == frame.index[0] else "wait",
        )
        self.assertEqual([entry.action for entry in result.executions], ["open_short", "cover_short"])
        self.assertEqual(result.executions[1].submitted_at, frame.index[1])
        self.assertEqual(result.executions[1].executed_at, frame.index[2])
        self.assertEqual(result.final_position, 0)

    def test_missing_session_end_does_not_simulate_an_overnight_cover(self):
        frame = bars([
            ("2024-01-02T14:45", 100, 101, 99, 100, 1000),
            ("2024-01-02T15:00", 100, 101, 99, 100, 1000),
            ("2024-01-03T09:15", 100, 101, 99, 100, 1000),
        ])
        with self.assertRaisesRegex(RuntimeError, "later session"):
            simulate_episode(
                frame,
                config=ExecutionConfig(style="intraday", interval="15m"),
                policy=lambda _: "open_short",
            )

    def test_short_liquidation_uses_last_session_open_for_each_interval(self):
        for interval, minutes in (
            ("1m", 1), ("5m", 5), ("15m", 15),
            ("20m", 20), ("30m", 30), ("1h", 60),
        ):
            with self.subTest(interval=interval):
                last = datetime(2024, 1, 2, 9, 15) + timedelta(
                    minutes=(374 // minutes) * minutes
                )
                timestamps = [
                    last - timedelta(minutes=2 * minutes),
                    last - timedelta(minutes=minutes),
                    last,
                ]
                frame = pd.DataFrame(
                    [[100, 101, 99, 100, 1000]] * 3,
                    index=pd.DatetimeIndex(timestamps),
                    columns=["open", "high", "low", "close", "volume"],
                )
                result = simulate_episode(
                    frame,
                    config=ExecutionConfig(style="intraday", interval=interval),
                    policy=lambda obs: (
                        "open_short" if obs.timestamp == timestamps[0] else "wait"
                    ),
                )
                self.assertEqual(result.executions[-1].executed_at, timestamps[-1])
                self.assertEqual(result.final_position, 0)

    def test_drawdown_liquidates_at_next_open_and_stops(self):
        frame = bars([
            ("2024-01-02T09:15", 100, 100, 100, 100, 1),
            ("2024-01-02T09:16", 100, 200, 100, 200, 1),
            ("2024-01-02T09:17", 200, 200, 80, 80, 1),
            ("2024-01-02T09:18", 70, 70, 70, 70, 1),
            ("2024-01-02T09:19", 500, 500, 500, 500, 1),
        ])
        actions = iter(["open_long", "hold"])
        result = simulate_episode(
            frame,
            config=ExecutionConfig(
                style="intraday", interval="1m", initial_cash=1000,
                exposure_fraction=1.0,
                costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
            ),
            policy=lambda _: next(actions),
        )
        self.assertTrue(result.drawdown_triggered)
        self.assertEqual(result.peak_equity, 2000)
        self.assertEqual(result.executions[-1].executed_at, frame.index[3])
        self.assertEqual(result.final_equity, 700)
        self.assertEqual(result.final_position, 0)

    def test_invalid_actions_and_unfilled_orders_fail_explicitly(self):
        frame = bars([
            ("2024-01-02T09:15", 100, 101, 99, 100, 1),
            ("2024-01-02T09:16", 100, 101, 99, 100, 1),
        ])
        config = ExecutionConfig(style="intraday", interval="1m")
        with self.assertRaisesRegex(ValueError, "invalid for position"):
            simulate_episode(
                frame, config=config,
                policy=lambda obs: "close_long" if obs.timestamp == frame.index[0] else "wait",
            )
        with self.assertRaisesRegex(RuntimeError, "before a submitted market order"):
            simulate_episode(
                frame, config=config,
                policy=lambda obs: "wait" if obs.timestamp == frame.index[0] else "open_long",
            )
        with self.assertRaisesRegex(ValueError, "only for intraday"):
            simulate_episode(
                bars([
                    ("2024-01-02T00:00", 100, 101, 99, 100, 1),
                    ("2024-01-03T00:00", 100, 101, 99, 100, 1),
                ]),
                config=ExecutionConfig(style="swing", interval="1d"),
                policy=lambda obs: (
                    "open_short" if obs.timestamp.date().day == 2 else "wait"
                ),
            )

    def test_rejects_misaligned_intraday_bars(self):
        frame = bars([
            ("2024-01-02T09:16", 100, 101, 99, 100, 1),
            ("2024-01-02T09:31", 100, 101, 99, 100, 1),
        ])
        with self.assertRaisesRegex(ValueError, "must align"):
            simulate_episode(
                frame,
                config=ExecutionConfig(style="intraday", interval="15m"),
                policy=lambda _: "wait",
            )


if __name__ == "__main__":
    unittest.main()
