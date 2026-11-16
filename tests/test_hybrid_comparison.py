import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl

from backtesting.execution_simulator import ExecutionCosts

if find_spec("backtrader") is not None:
    from backtesting.rl_execution import ExecutionConfig, simulate_episode
    from modeling.hybrid_comparison import (
        _make_policy,
        _evaluate_symbol,
        _summarize_result,
        compare_hybrid_models,
    )


def bars(times):
    return pd.DataFrame(
        [[100, 101, 99, 100, 1000], [110, 111, 104, 105, 1000],
         [90, 91, 89, 90, 1000]][:len(times)],
        index=pd.DatetimeIndex(times),
        columns=["open", "high", "low", "close", "volume"],
    )


def candidate(local_time, probability, expected_return=-1, uncertainty=0.99):
    return {
        "timestamp": (pd.Timestamp(local_time).tz_localize("Asia/Kolkata")
                      .tz_convert("UTC").tz_localize(None).to_pydatetime()),
        "direction": "long",
        "win_probability": probability,
        "expected_return_r": expected_return,
        "uncertainty": uncertainty,
    }


@unittest.skipUnless(find_spec("backtrader") is not None, "Backtrader is not installed")
class HybridComparisonTests(unittest.TestCase):
    def test_filter_and_next_open_with_signed_pnl(self):
        frame = bars(["2024-01-02 09:15", "2024-01-02 09:16", "2024-01-02 09:17"])
        config = ExecutionConfig(
            style="intraday", interval="1m",
            costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
        )
        candidates = [candidate(frame.index[0], 0.30)]
        for mode, expected_orders in (("no_trade", 0), ("legacy", 0), ("new", 2)):
            with self.subTest(mode=mode):
                policy, stats = _make_policy(
                    bars=frame, candidates=candidates, config=config, horizon=1, mode=mode,
                )
                result = simulate_episode(frame, config=config, policy=policy)
                summary = _summarize_result(result, stats, intraday=True)
                self.assertEqual(len(result.executions), expected_orders)
                if mode == "new":
                    self.assertEqual(result.executions[0].executed_at, frame.index[1])
                    self.assertEqual(result.executions[1].executed_at, frame.index[2])
                    self.assertEqual(summary["profit_loss"], "-1000.00")
                    self.assertEqual(summary["max_drawdown_percent"], "-2.00")
                    self.assertEqual(summary["losing_trades"], 1)

    def test_legacy_requires_all_gates_and_ignores_unscorable(self):
        frame = bars(["2024-01-02 09:15", "2024-01-02 09:16", "2024-01-02 09:17"])
        config = ExecutionConfig(style="intraday", interval="1m")
        for row, expected in (
            (candidate(frame.index[0], None), 0),
            (candidate(frame.index[0], 0.50, -1, 0.5), 0),
            (candidate(frame.index[0], 0.50, 1, 0.96), 0),
            (candidate(frame.index[0], 0.45, 1, 0.95), 2),
        ):
            with self.subTest(row=row):
                policy, _ = _make_policy(
                    bars=frame, candidates=[row], config=config, horizon=1, mode="legacy",
                )
                result = simulate_episode(frame, config=config, policy=policy)
                self.assertEqual(len(result.executions), expected)

    def test_final_position_is_marked_to_market_without_unfillable_exit(self):
        frame = bars(["2024-01-02 09:15", "2024-01-02 09:16", "2024-01-02 09:17"])
        config = ExecutionConfig(style="intraday", interval="1m")
        policy, stats = _make_policy(
            bars=frame, candidates=[candidate(frame.index[0], 0.3)],
            config=config, horizon=12, mode="new",
        )
        result = simulate_episode(frame, config=config, policy=policy)
        summary = _summarize_result(result, stats, intraday=True)
        self.assertNotEqual(result.final_position, 0)
        self.assertEqual(summary["closed_trades"], 0)
        self.assertEqual(summary["order_count"], 1)

    def test_cross_session_long_rejected_intraday_but_allowed_for_swing(self):
        frame = bars(["2024-01-02 15:00", "2024-01-03 09:15", "2024-01-03 09:30"])
        for style, interval in (("intraday", "15m"), ("swing", "1d")):
            config = ExecutionConfig(style=style, interval=interval)
            _, stats = _make_policy(
                bars=frame, candidates=[candidate(frame.index[0], 0.3)],
                config=config, horizon=1, mode="new",
            )
            # Use explicit actions so the case exercises the execution check.
            actions = iter(["open_long", "close_long", "wait"])
            result = simulate_episode(frame, config=config, policy=lambda _: next(actions))
            if style == "intraday":
                with self.assertRaisesRegex(ValueError, "crossed an exchange session"):
                    _summarize_result(result, stats, intraday=True)
            else:
                self.assertEqual(_summarize_result(result, stats, intraday=False)["closed_trades"], 1)

    def test_insufficient_test_bars_are_explicit(self):
        frame = pl.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-02").to_pydatetime()],
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.0], "volume": [1.0],
        })
        signals = frame.select("timestamp").head(0)
        job = {
            "style": "swing", "strategy": "momentum",
            "test_start": "2024-01-01T00:00:00+00:00", "model_dir": "model",
        }
        with patch("modeling.hybrid_comparison.build_features") as build, \
                patch("modeling.hybrid_comparison.generate_strategy_signals", return_value=signals):
            build.return_value.collect.return_value = frame
            result = _evaluate_symbol(job=job, input_path=Path("AAA") / "1d.csv.gz",
                                      score_batch_size=10)
        self.assertEqual(result["status"], "insufficient_test_bars")
        self.assertEqual(result["test_bar_count"], 1)

    def test_resume_and_input_and_model_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            model = batch / "15m" / "box_breakout"
            model.mkdir(parents=True)
            data = root / "stocks"
            for symbol in ("AAA", "BBB"):
                path = data / "minute" / symbol / "15m.csv.gz"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"input")
            def write(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
            write(batch / "batch-state.json", {
                "status": "completed", "completed_jobs_by_stage": {"hybrid_calibration": 1},
            })
            write(batch / "batch-manifest.json", {"configuration": {
                "commission_bps": 3, "other_fees_bps": 7, "slippage_bps": 5,
            }})
            write(model / "hybrid-batch-metrics.json", {})
            write(model / "batch-job.json", {
                "interval": "15m", "strategy": "box_breakout", "style": "intraday",
                "horizon": 12, "input_fingerprints": [
                    {"path": rf"D:\old\minute\{symbol}\15m.csv.gz", "size": 5}
                    for symbol in ("AAA", "BBB")
                ],
            })
            write(model / "metadata.json", {
                "style": "intraday", "strategy": "box_breakout", "horizon": 12,
                "test_start": "2024-01-01T00:00:00+00:00",
            })
            write(model / "hybrid.json", {"model_weights": {"lstm": 1.0}})
            (model / "lstm.pt").write_bytes(b"model")
            scenario = {
                "initial_capital": "50000.00", "profit_loss": "-1.25",
                "closed_trades": 1, "winning_trades": 0, "losing_trades": 1,
            }
            evaluated = {"status": "evaluated", "scenarios": {
                mode: scenario for mode in ("no_trade", "legacy", "new")
            }}
            output = root / "output"
            args = dict(model_roots=[batch], stock_data_root=data, output_root=output)
            missing = data / "minute" / "BBB" / "15m.csv.gz"
            missing.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "training input is missing"):
                compare_hybrid_models(**args)
            missing.write_bytes(b"input")
            with patch("modeling.hybrid_comparison._evaluate_symbol",
                       side_effect=[evaluated, RuntimeError("interrupted")]) as evaluate:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    compare_hybrid_models(**args)
                self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(json.loads((output / "comparison-run.json").read_text())["status"],
                             "in_progress")
            with patch("modeling.hybrid_comparison._evaluate_symbol", return_value=evaluated) as evaluate:
                result = compare_hybrid_models(**args, resume=True)
                self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(result["totals"]["new"]["profit_loss"], "-2.50")
            self.assertEqual(result["totals"]["new"]["initial_capital"], "100000.00")
            self.assertEqual(result["by_timeframe_strategy"][0]["evaluated_count"], 2)
            input_path = data / "minute" / "AAA" / "15m.csv.gz"
            input_path.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "Input changed after comparison"):
                compare_hybrid_models(**args, resume=True)
            (model / "lstm.pt").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "model artifacts changed"):
                compare_hybrid_models(**args, resume=True)
            (model / "hybrid-batch-metrics.json").unlink()
            with self.assertRaisesRegex(ValueError, "Completed model count does not match"):
                compare_hybrid_models(**args, resume=True)


if __name__ == "__main__":
    unittest.main()
