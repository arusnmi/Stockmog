import csv
import contextlib
import io
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped
from modeling.nifty_same_candle_correlations import _coefficients, _observations, _read_candles, run
from modeling.nifty_universe import Membership


def _write_candles(path, timestamps, closes, partial=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "timestamp": timestamps, "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": 1.0, "source_bars": 1, "trading_days": 1,
        "is_partial": partial if partial is not None else [False] * len(closes),
    })
    frame.to_csv(path, index=False, compression="gzip")


class SameCandleCorrelationTests(unittest.TestCase):
    def test_spearman_uses_average_tied_ranks_and_returns_require_same_previous_interval(self):
        days = [date(2022, 1, i) for i in range(3, 8)]
        stock = pd.DataFrame({
            "start": days, "end": days, "sessions": list(map(str, days)),
            "previous_start": [None, *days[:-1]], "previous_end": [None, *days[:-1]],
            "previous_sessions": [None, *map(str, days[:-1])],
            "close": [10., 20., 30., 40., 50.], "return": [np.nan, 1., 0.5, 1 / 3, 0.25],
        })
        factor = stock.copy()
        factor["close"] = [5., 8., 8., 20., 25.]
        factor["return"] = [np.nan, 0.6, 0., 1.5, 0.25]
        factor.loc[2, "previous_start"] = date(2021, 12, 31)
        actual = _coefficients(stock, factor, 3)
        self.assertEqual(actual["price_pairs"], 5)
        self.assertEqual(actual["return_pairs"], 3)
        self.assertEqual(actual["price_first"], "2022-01-03")
        self.assertEqual(actual["price_last"], "2022-01-07")
        self.assertAlmostEqual(actual["price_pearson"], np.corrcoef(stock["close"], factor["close"])[0, 1])
        self.assertAlmostEqual(
            actual["price_spearman"],
            np.corrcoef([1, 2, 3, 4, 5], [1, 2.5, 2.5, 4, 5])[0, 1],
        )
        self.assertAlmostEqual(
            actual["return_pearson"],
            np.corrcoef(stock.loc[[1, 3, 4], "return"], factor.loc[[1, 3, 4], "return"])[0, 1],
        )

    def test_membership_gaps_and_resume_integrity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stocks = root / "stocks"
            daily_timestamps = [f"2022-01-0{day}T00:00:00+0530" for day in (3, 4, 5)]
            for name in ("AAA", "BBB", "RELIANCE"):
                _write_candles(stocks / "day" / name / "1d.csv.gz", daily_timestamps, [10, 11, 12])
            timestamps = [f"2022-01-0{day}T09:{minute:02d}:00+0530"
                          for day in (3, 4, 5) for minute in (15, 16, 17)]
            _write_candles(stocks / "minute" / "AAA" / "1m.csv.gz", timestamps, list(range(10, 19)))
            factor_timestamps = [stamp for stamp in timestamps if stamp != "2022-01-04T09:16:00+0530"]
            _write_candles(stocks / "minute" / "BBB" / "1m.csv.gz", factor_timestamps, list(range(20, 28)))
            member = Membership("AAA", date(2022, 1, 4), None, "test", "")
            reference = [date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)]
            stock = _read_candles(stocks, "AAA", "1m", reference, (member,))
            factor = _read_candles(stocks, "BBB", "1m", reference)
            self.assertEqual(len(stock), 6)
            result = _coefficients(stock, factor, 3)
            self.assertEqual(result["price_pairs"], 5)
            self.assertEqual(result["return_pairs"], 2)
            self.assertEqual(result["return_status"], "insufficient_pairs")
            self.assertIsNone(result["return_pearson"])
            membership = root / "members.csv"
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-04,,test,\n", encoding="utf-8"
            )
            output = root / "output"
            args = dict(membership=membership, stocks=stocks, output=output, symbols=["AAA"],
                        factors=["BBB"], timeframes=["1m"])
            run(**args)
            with (output / "stocks" / "AAA.csv").open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["price_pairs"], "5")
            run(**args, resume=True)
            with (output / "stocks" / "AAA.csv").open("a", encoding="utf-8") as file:
                file.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "output changed"):
                run(**args, resume=True)
            stopped = root / "stopped"
            stopped_args = {**args, "output": stopped}

            def request_stop(*values, **kwargs):
                result = _read_candles(*values, **kwargs)
                TrainingControl(stopped).request("stop")
                return result

            with patch("modeling.nifty_same_candle_correlations._read_candles", side_effect=request_stop):
                with self.assertRaises(PreparationStopped):
                    run(**stopped_args)
            run(**stopped_args, resume=True)
            self.assertTrue((stopped / "stocks" / "AAA.csv").is_file())

            _write_candles(stocks / "minute" / "BBB" / "1m.csv.gz", factor_timestamps, [-1] * 8)
            invalid = root / "invalid"
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                run(**{**args, "output": invalid})
            with (invalid / "stocks" / "AAA.csv").open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["price_status"], "invalid_factor_candles")
            self.assertIn("Invalid candle close", row["error"])
            self.assertIn("Invalid factor candles", errors.getvalue())

    def test_rate_changes_are_absolute_and_missing_values_not_filled(self):
        data = _observations(pd.Series(["2022-01-03", "2022-01-04", "2022-01-05"]),
                             pd.Series([-0.5, None, 0.5]), rate=True)
        self.assertEqual(data["start"].tolist(), [date(2022, 1, 3), date(2022, 1, 5)])
        self.assertEqual(data["return"].iloc[1], 1.)
        self.assertEqual(data["previous_start"].iloc[1], date(2022, 1, 3))

    def test_multisession_candles_with_different_internal_sessions_do_not_match(self):
        days = pd.Series(["2022-01-03", "2022-01-10", "2022-01-17"])
        stock = _observations(days, pd.Series([10., 11., 12.]))
        factor = stock.copy()
        stock["sessions"] = ["Mon,Tue,Thu,Fri"] * 3
        factor["sessions"] = ["Mon,Wed,Thu,Fri"] * 3
        self.assertEqual(_coefficients(stock, factor, 3)["price_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
