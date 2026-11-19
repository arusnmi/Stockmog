import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_pipeline.nifty_prediction_series import (
    merge_daily_sources,
    normalize_provider_rows,
    rolling_daily_predictions,
)
from modeling.nifty_preparation import available_candles


IST = ZoneInfo("Asia/Kolkata")


def sample_daily(count=6):
    days = pd.bdate_range("2026-01-05", periods=count, tz="Asia/Kolkata")
    close = 100 + np.arange(count, dtype=float)
    raw = pd.DataFrame({
        "timestamp": days.tz_convert("UTC"),
        "open": close - .5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.full(count, 1000.),
        "source_bars": 1,
        "trading_days": 1,
        "is_partial": False,
    })
    candles = available_candles(raw, "1d", daily_source=True)
    features = candles[["timestamp", "available_at", "close"]].copy()
    features["input"] = np.arange(count, dtype=float)
    return raw, candles, features, days


class EchoPredictor:
    def __init__(self):
        self.windows = []

    def predict(self, tabular, sequences, anchors, *, feature_names):
        self.windows.append(sequences.copy())
        return anchors + 1


class PredictionSeriesTests(unittest.TestCase):
    def test_incomplete_current_session_is_excluded_before_close(self):
        now = datetime(2026, 1, 6, 15, 44, tzinfo=IST)
        rows = [
            {"timestamp": "2026-01-05", "open": 10, "high": 12, "low": 9,
             "close": 11, "volume": 100},
            {"timestamp": "2026-01-06", "open": 11, "high": 13, "low": 10,
             "close": 12, "volume": 100},
        ]
        result = normalize_provider_rows(rows, now=now)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["timestamp"].iloc[0].tz_convert(IST).date().isoformat(),
                         "2026-01-05")

    def test_latest_session_is_included_after_close(self):
        now = datetime(2026, 1, 6, 15, 45, tzinfo=IST)
        rows = [
            {"timestamp": "2026-01-05", "open": 10, "high": 12, "low": 9,
             "close": 11, "volume": 100},
            {"timestamp": "2026-01-06", "open": 11, "high": 13, "low": 10,
             "close": 12, "volume": 100},
        ]
        result = normalize_provider_rows(rows, now=now)
        self.assertEqual(len(result), 2)

    def test_rolling_predictions_use_causal_adjacent_windows_through_final_session(self):
        _, candles, features, days = sample_daily()
        predictor = EchoPredictor()
        output = rolling_daily_predictions(
            candles, features, [item.date() for item in days], predictor, ["input"], 2,
            days[2].date())
        self.assertEqual(output["target_session"].tolist(),
                         [day.date().isoformat() for day in days[2:]])
        self.assertEqual(output["decision_session"].iloc[0], days[1].date().isoformat())
        self.assertEqual(output["actual_close"].iloc[-1], candles["close"].iloc[-1])
        self.assertTrue(all(window.shape == (1, 2, 1) for window in predictor.windows))
        np.testing.assert_array_equal(predictor.windows[0][0, :, 0], [0, 1])

    def test_missing_session_breaks_both_window_and_target_adjacency(self):
        _, candles, features, days = sample_daily()
        keep = [0, 1, 2, 4, 5]
        candles = candles.iloc[keep].reset_index(drop=True)
        features = features.iloc[keep].reset_index(drop=True)
        predictor = EchoPredictor()
        output = rolling_daily_predictions(
            candles, features, [item.date() for item in days], predictor, ["input"], 2,
            days[2].date())
        self.assertEqual(output["target_session"].tolist(), [days[2].date().isoformat()])

    def test_saved_history_is_preserved_and_only_later_yahoo_rows_are_appended(self):
        raw, _, _, _ = sample_daily(6)
        local = raw.iloc[:4].copy()
        provider = raw.iloc[2:].copy()
        provider.loc[provider.index[:2], "close"] = 42
        merged = merge_daily_sources(local, provider, "TEST")
        self.assertEqual(merged["close"].tolist(), raw["close"].tolist())


if __name__ == "__main__":
    unittest.main()
