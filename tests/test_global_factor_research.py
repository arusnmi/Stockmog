import unittest
from datetime import date, timedelta

import pandas as pd

from modeling.global_factor_research import correlate, global_returns, next_session_targets
from modeling.nifty_universe import Membership


class GlobalFactorResearchTests(unittest.TestCase):
    def test_factor_returns_exclude_incomplete_and_missing_sessions(self):
        sessions = [date(2022, 1, day) for day in (3, 4, 5, 6, 7)]
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-03T14:30Z", "2022-01-04T14:30Z",
                                         "2022-01-05T14:30Z", "2022-01-07T14:30Z"], utc=True),
            "available_at": pd.to_datetime(["2022-01-04T05:00Z", "2022-01-05T05:00Z",
                                            "2022-01-06T05:00Z", "2022-01-08T05:00Z"], utc=True),
            "close": [100.0, 110.0, 120.0, 130.0], "is_partial": [False, False, True, False],
        })
        result = global_returns(frame, sessions)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["factor_return"][0], 0.1)
        self.assertEqual(result["factor_available_at"][0], pd.Timestamp("2022-01-05T05:00Z"))

    def test_targets_do_not_cross_missing_sessions_membership_or_cutoff(self):
        days = [date(2022, 1, day) for day in (3, 4, 5, 6, 7)]
        closes = pd.Series([100.0, 101.0, 103.0, 104.0], index=[days[0], days[1], days[3], days[4]])
        member = (Membership("AAA", days[0], days[4], "test", ""),)
        result = next_session_targets(closes, days, member, days[4])
        self.assertEqual(result["session"].tolist(), [days[0]])
        self.assertEqual(result["target_session"].tolist(), [days[1]])
        self.assertEqual(result["available_at"][0], pd.Timestamp("2022-01-03T18:30Z"))

    def test_partial_session_requires_actual_fresh_matching_close(self):
        sessions = [date(2022, 1, day) for day in (3, 4, 5, 6)]
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-03T14:30Z", "2022-01-04T14:30Z",
                                         "2022-01-05T14:30Z", "2022-01-06T14:30Z"], utc=True),
            "available_at": pd.to_datetime(["2022-01-04T05:00Z", "2022-01-05T05:00Z",
                                            "2022-01-06T05:00Z", "2022-01-07T05:00Z"], utc=True),
            "close": [100.0, 110.0, 120.0, 130.0], "is_partial": [True] * 4,
        })
        minutes = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-03T20:59Z", "2022-01-04T20:55Z",
                                         "2022-01-05T20:50Z", "2022-01-06T20:59Z"], utc=True),
            "available_at": pd.to_datetime(["2022-01-03T21:00Z", "2022-01-04T20:56Z",
                                            "2022-01-05T20:51Z", "2022-01-06T21:00Z"], utc=True),
            "close": [100.0, 110.0, 120.0, 130.0],
        })
        result = global_returns(frame, sessions, minutes=minutes, max_close_age_minutes=5)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["factor_return"][0], 0.1)
        minutes.loc[1, "close"] = 111.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            global_returns(frame, sessions, minutes=minutes, max_close_age_minutes=5)

    def test_correlation_uses_only_available_factors(self):
        targets = pd.DataFrame({
            "available_at": pd.to_datetime(["2022-01-04T10:00Z", "2022-01-05T10:00Z",
                                           "2022-01-06T10:00Z"], utc=True),
            "target_return": [0.1, 0.2, 0.3],
        })
        factors = pd.DataFrame({
            "factor_available_at": pd.to_datetime(["2022-01-04T05:00Z", "2022-01-05T05:00Z",
                                                   "2022-01-06T05:00Z", "2022-01-06T11:00Z"], utc=True),
            "factor_return": [0.1, 0.2, 0.3, -100.0],
        })
        count, value = correlate(targets, factors, max_age=timedelta(days=2), minimum=3)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value, 1.0)
        self.assertIsNone(correlate(targets, factors, max_age=timedelta(hours=1), minimum=3))


if __name__ == "__main__":
    unittest.main()
