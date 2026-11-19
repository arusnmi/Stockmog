import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from data_pipeline.nifty_factor_updates import extend_factor
from modeling.nifty_public_factor_research import REPOSITORY


IST = ZoneInfo("Asia/Kolkata")


class Provider:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def fetch_history(self, symbol, start, end, interval="1d"):
        self.calls.append((symbol, start, end))
        return self.rows


def row(day, price):
    return {"timestamp": day, "open": price, "high": price, "low": price,
            "close": price, "volume": 100}


class FactorUpdateTests(unittest.TestCase):
    def test_external_returns_append_after_saved_history_with_existing_lag(self):
        key = ("india:" + REPOSITORY, "NIFTY_50:Close")
        old = pd.DataFrame({"end": [pd.Timestamp("2026-01-05").date()], "close": [100.]})
        store = SimpleNamespace(external={key: old})
        values = pd.DataFrame({"available_at": [pd.Timestamp("2026-01-07", tz="America/New_York")
                                               .tz_convert("UTC")],
                               "factor_return": [0.01]})
        provider = Provider([row("2026-01-06", 110), row("2026-01-07", 121)])
        result, info = extend_factor(
            key, values, store, {}, Path("."), [], provider,
            datetime(2026, 1, 7, 16, tzinfo=IST))
        self.assertEqual(provider.calls[0][0:2], ("^NSEI", pd.Timestamp("2026-01-06").date()))
        self.assertEqual(info["status"], "extended")
        self.assertEqual(info["rows_added"], 2)
        self.assertAlmostEqual(result["factor_return"].iloc[-1], 0.1)
        self.assertEqual(result["available_at"].iloc[-1],
                         pd.Timestamp("2026-01-09", tz="America/New_York").tz_convert("UTC"))

    def test_local_factor_does_not_make_return_across_missing_reference_session(self):
        days = pd.bdate_range("2026-01-05", periods=4)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "day" / "TEST"
            path.mkdir(parents=True)
            data = pd.DataFrame([row("2026-01-05T00:00:00+05:30", 100)])
            data = data.assign(source_bars=1, trading_days=1, is_partial=False)
            data.to_csv(path / "1d.csv.gz", index=False, compression="gzip")
            provider = Provider([row("2026-01-07", 110), row("2026-01-08", 121)])
            values = pd.DataFrame({"available_at": pd.Series([], dtype="datetime64[ns, UTC]"),
                                   "factor_return": pd.Series([], dtype=float)})
            with patch("data_pipeline.nifty_factor_updates._local_ticker", return_value="TEST.NS"):
                result, info = extend_factor(
                    ("local", "TEST"), values, SimpleNamespace(), {}, root,
                    [day.date() for day in days], provider, datetime(2026, 1, 8, 16, tzinfo=IST))
            self.assertEqual(info["rows_added"], 1)
            self.assertAlmostEqual(result["factor_return"].iloc[0], 0.1)

    def test_unavailable_factor_is_reported_without_replacing_saved_values(self):
        key = ("india:" + REPOSITORY, "NIFTY_50:Close")
        store = SimpleNamespace(external={key: pd.DataFrame(
            {"end": [pd.Timestamp("2026-01-05").date()], "close": [100.]})})
        values = pd.DataFrame({"available_at": [pd.Timestamp("2026-01-07", tz="UTC")],
                               "factor_return": [0.01]})
        provider = Provider([])
        with patch("data_pipeline.nifty_factor_updates.sys.stderr"):
            result, info = extend_factor(
                key, values, store, {}, Path("."), [], provider,
                datetime(2026, 1, 7, 16, tzinfo=IST))
        self.assertEqual(info["status"], "unavailable")
        self.assertIn("returned no daily candles", info["error"])
        pd.testing.assert_frame_equal(result, values)


if __name__ == "__main__":
    unittest.main()
