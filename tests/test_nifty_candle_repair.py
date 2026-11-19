import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_pipeline.nifty_candle_repair import (
    derived_candidates, expected_labels, read_candles, recoverable_rows,
    recover_from_intervals, save_repair,
)


def candles(timestamps):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "open": 100., "high": 102., "low": 99., "close": 101., "volume": 10.,
        "source_bars": 1, "trading_days": 1, "is_partial": False,
    })


class CandleRepairTests(unittest.TestCase):
    def test_only_complete_finer_candles_can_rebuild_missing_intraday(self):
        source = candles(pd.date_range("2022-01-03T03:45Z", periods=10, freq="min"))
        full = derived_candidates(source, None, [date(2022, 1, 3)], "5m", "AAA")
        self.assertEqual(len(full), 2)
        self.assertEqual(full["volume"].tolist(), [50., 50.])
        gap = source.drop(index=2)
        result = derived_candidates(gap, None, [date(2022, 1, 3)], "5m", "AAA")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["timestamp"].iloc[0], source["timestamp"].iloc[5])
        self.assertTrue(derived_candidates(None, source, [], "1m", "AAA").empty)
        self.assertTrue(derived_candidates(None, source, [], "5m", "AAA").empty)

    def test_daily_can_supply_multiday_but_not_intraday_and_skips_reference_gaps(self):
        days = pd.bdate_range("2022-01-03", periods=10, tz="Asia/Kolkata")
        daily = candles(days)
        reference = days.date.tolist()
        pairs = derived_candidates(None, daily, reference, "2d", "AAA")
        weeks = derived_candidates(None, daily, reference, "trading_week", "AAA")
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len(weeks), 2)
        self.assertEqual(weeks["volume"].tolist(), [50., 50.])
        missing_day = daily.drop(index=1).reset_index(drop=True)
        result = derived_candidates(None, missing_day, reference, "2d", "AAA")
        self.assertNotIn(daily["timestamp"].iloc[0], result["timestamp"].tolist())
        result = derived_candidates(None, missing_day, reference, "trading_week", "AAA")
        self.assertEqual(len(result), 1)

    def test_repair_does_not_replace_complete_rows_and_validates_roundtrip(self):
        source = candles(pd.date_range("2022-01-03T03:45Z", periods=10, freq="min"))
        derived = derived_candidates(source, None, [], "5m", "AAA")
        existing = derived.iloc[:1].copy()
        existing["close"] = 100.5
        recovered = recoverable_rows(existing, derived)
        self.assertEqual(len(recovered), 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "5m.csv.gz"
            existing.to_csv(path, index=False, compression="gzip")
            save_repair(path, recovered, "AAA")
            result = read_candles(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result["close"].iloc[0], 100.5)
            self.assertTrue(recoverable_rows(result, derived).empty)
        existing["is_partial"] = True
        self.assertEqual(len(recoverable_rows(existing, derived)), 2)

    def test_expected_intraday_labels_are_session_anchored(self):
        days = [date(2022, 1, 3), date(2022, 1, 4)]
        labels = expected_labels(days, days, "20m")
        self.assertEqual(len(labels), 38)
        self.assertEqual(labels[0], pd.Timestamp("2022-01-03T03:45Z"))
        self.assertEqual(labels[18], pd.Timestamp("2022-01-03T09:45Z"))

    def test_mixed_intervals_cover_exactly_and_never_split_coarse_bars(self):
        source = candles(pd.date_range("2022-01-03T03:45Z", periods=15, freq="min"))
        fives = derived_candidates(source, None, [], "5m", "AAA")
        # Taking the first minute greedily would fail; use the complete 5m instead.
        sources = {"1m": source.drop(index=2), "5m": fives}
        label = pd.DatetimeIndex([source["timestamp"].iloc[0]])
        result = recover_from_intervals(sources, label, "15m")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["volume"].iloc[0], 150.)
        self.assertTrue(recover_from_intervals({"5m": fives}, label, "1m").empty)
        self.assertTrue(recover_from_intervals(
            {"1m": source.drop(index=2), "5m": fives.iloc[1:]}, label, "15m").empty)

    def test_daily_can_be_rebuilt_from_complete_five_minute_bars(self):
        source = candles(pd.date_range("2022-01-03T03:45Z", periods=375, freq="min"))
        fives = derived_candidates(source, None, [], "5m", "AAA")
        labels = pd.DatetimeIndex(["2022-01-02T18:30Z"])
        result = recover_from_intervals({"5m": fives}, labels, "1d")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["volume"].iloc[0], 3750.)
        self.assertTrue(recover_from_intervals({"5m": fives.iloc[:-1]}, labels, "1d").empty)


if __name__ == "__main__":
    unittest.main()
