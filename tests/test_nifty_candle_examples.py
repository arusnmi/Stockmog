import unittest
from datetime import date

import numpy as np
import pandas as pd

from modeling.nifty_candle_examples import build_candle_examples
from modeling.nifty_preparation import available_candles, daily_source_aggregates
from modeling.nifty_universe import Membership
from tests.test_nifty_preparation import candles


class NextCandleExamplesTests(unittest.TestCase):
    def setUp(self):
        self.days = [day.date() for day in pd.bdate_range("2022-01-03", periods=15)]
        self.members = (Membership("AAA", self.days[0], None, "test", ""),)
        self.start, self.end = date(2022, 1, 3), date(2022, 2, 1)

    def make(self, timeframe, frame, **kwargs):
        return build_candle_examples("AAA", timeframe, frame, self.days, self.members,
                                     start=self.start, end=self.end, **kwargs)

    def test_every_intraday_timeframe_uses_adjacent_candle_close(self):
        for timeframe, minute in (("1m", 1), ("5m", 5), ("15m", 15),
                                  ("20m", 20), ("30m", 30), ("1h", 60)):
            with self.subTest(timeframe=timeframe):
                stamps = pd.date_range("2022-01-03 09:15", periods=4, freq=f"{minute}min",
                                       tz="Asia/Kolkata")
                raw = candles([stamp.isoformat() for stamp in stamps])
                raw["close"] = [100, 101, 102, 103]
                raw["high"] = [101, 102, 103, 104]
                prepared = available_candles(raw, timeframe)
                result = self.make(timeframe, prepared)
                self.assertEqual(len(result), 3)
                np.testing.assert_array_equal(result["target_close"], [101, 102, 103])
                self.assertTrue((result["label_available_at"] > result["available_at"]).all())
                missing = prepared.drop(index=1).reset_index(drop=True)
                self.assertEqual(self.make(timeframe, missing)["target_close"].tolist(), [103])

    def test_intraday_never_bridges_overnight_or_partial(self):
        raw = candles(["2022-01-03T15:28:00+0530", "2022-01-03T15:29:00+0530",
                       "2022-01-04T09:15:00+0530", "2022-01-04T09:16:00+0530"])
        raw.loc[1, "is_partial"] = True
        prepared = available_candles(raw, "1m")
        result = self.make("1m", prepared)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["session"].iloc[0], self.days[1])

    def test_daily_and_multisession_pairs_require_reference_continuity(self):
        raw = candles([f"{day}T00:00:00+0530" for day in self.days])
        raw["close"] = np.arange(100, 115)
        raw["open"] = raw["close"]
        raw["high"] = raw["close"] + 1
        raw["low"] = raw["close"] - 1
        for timeframe in ("1d", "2d", "trading_week"):
            with self.subTest(timeframe=timeframe):
                prepared = (available_candles(raw, "1d", daily_source=True)
                            if timeframe == "1d" else daily_source_aggregates(raw, timeframe))
                result = self.make(timeframe, prepared)
                self.assertGreater(len(result), 0)
                self.assertTrue((result["label_available_at"] > result["available_at"]).all())
                self.assertTrue((result["target_close"] > result["anchor_close"]).all())
                if len(prepared) > 3:
                    gap = prepared.drop(index=1).reset_index(drop=True)
                    after = self.make(timeframe, gap)
                    self.assertLess(len(after), len(result))

    def test_membership_and_date_window_apply_to_target(self):
        raw = candles([f"{day}T00:00:00+0530" for day in self.days[:5]])
        prepared = available_candles(raw, "1d", daily_source=True)
        result = build_candle_examples("AAA", "1d", prepared, self.days,
                                       (Membership("AAA", self.days[0], self.days[2], "test", ""),),
                                       start=self.start, end=self.end)
        self.assertEqual(result["target_session"].tolist(), [self.days[1]])
        result = build_candle_examples("AAA", "1d", prepared, self.days, self.members,
                                       start=self.start, end=self.days[2])
        self.assertEqual(result["target_session"].tolist(), [self.days[1]])

    def test_decision_cannot_use_an_anchor_available_after_next_candle_starts(self):
        raw = candles(["2022-01-03T09:15:00+0530", "2022-01-03T09:16:00+0530",
                       "2022-01-03T09:17:00+0530"])
        prepared = available_candles(raw, "1m")
        prepared.loc[0, "available_at"] = pd.Timestamp("2022-01-03T03:46:30Z")
        result = self.make("1m", prepared)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["timestamp"].iloc[0], prepared["timestamp"].iloc[1])


if __name__ == "__main__":
    unittest.main()
