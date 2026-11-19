import hashlib
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from data_pipeline.global_factor_data import make_timeframe, prepare_factor, regular_bars
from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped
from modeling.nifty_preparation import join_available


def source_frame():
    return pd.DataFrame({
        "ts": pd.to_datetime(["2022-03-11T14:29Z", "2022-03-11T14:30Z",
                              "2022-03-11T14:31Z", "2022-03-14T13:30Z"], utc=True),
        "symbol": "TEST", "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 5.0,
    })


class GlobalFactorTests(unittest.TestCase):
    def test_sessions_and_aggregate_availability_respect_new_york_dst(self):
        bars, report = regular_bars(source_frame(), "TEST")
        self.assertEqual(len(bars), 3)
        self.assertEqual(report["excluded_extended_hours_rows"], 1)
        five = make_timeframe(bars, "5m")
        self.assertEqual(five["available_at"][0], pd.Timestamp("2022-03-11T14:35Z"))
        self.assertTrue(five["is_partial"][0])
        daily = make_timeframe(bars, "1d")
        self.assertEqual(daily["available_at"].tolist(), [
            pd.Timestamp("2022-03-12T05:00Z"), pd.Timestamp("2022-03-15T04:00Z")])
        two = make_timeframe(bars, "2d")
        self.assertEqual(two["available_at"][0], pd.Timestamp("2022-03-15T04:00Z"))
        week = make_timeframe(bars, "trading_week")
        self.assertEqual(week["available_at"][0], pd.Timestamp("2022-03-14T04:00Z"))
        self.assertTrue(week["is_partial"].all())

    def test_join_does_not_use_same_day_us_close_at_nse_decision_time(self):
        bars, _ = regular_bars(source_frame(), "TEST")
        daily = make_timeframe(bars, "1d")
        stock = pd.DataFrame({"available_at": pd.to_datetime(["2022-03-14T10:00Z"], utc=True)})
        joined = join_available(stock, daily, tolerance=timedelta(days=4))
        self.assertEqual(joined["factor_available_at"][0], pd.Timestamp("2022-03-12T05:00Z"))

    def test_stop_and_resume_reuses_completed_factor_checkpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            source_frame().to_parquet(source, index=False)
            item = {"symbol": "TEST", "group": "test", "path": "test.parquet",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            output = root / "out"
            control = TrainingControl(root)

            def stop_after_first(bars, timeframe):
                result = make_timeframe(bars, timeframe)
                if timeframe == "1m":
                    control.request("stop")
                return result

            with patch("data_pipeline.global_factor_data.make_timeframe", side_effect=stop_after_first):
                with self.assertRaises(PreparationStopped):
                    prepare_factor(item, source, output, control)
            first = output / "TEST" / "1m.parquet"
            before = first.stat().st_mtime_ns
            control.request("continue")
            report = prepare_factor(item, source, output, control)
            self.assertEqual(len(report["timeframes"]), 9)
            self.assertEqual(first.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
