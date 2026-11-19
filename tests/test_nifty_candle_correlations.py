import csv
import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped
from modeling.nifty_candle_correlations import _correlation, _events, _factor, _groups, run
from modeling.nifty_preparation import TIMEFRAMES


class CandleCorrelationTests(unittest.TestCase):
    def test_grouped_correlation_matches_direct_asof_including_expiry_boundary(self):
        factor = _factor("test", "factor", pd.to_datetime(["2022-01-03T00:00:00Z",
                                                            "2022-01-06T00:00:00Z",
                                                            "2022-01-09T00:00:00Z"]),
                         [1, 3, 5], 2)
        examples = pd.DataFrame({
            "available_at": pd.to_datetime(["2022-01-03T00:00:00Z", "2022-01-04T00:00:00Z",
                                             "2022-01-05T00:00:00Z", "2022-01-05T00:00:00.000000001Z",
                                             "2022-01-06T00:00:00Z", "2022-01-07T00:00:00Z",
                                             "2022-01-09T00:00:00Z", "2022-01-10T00:00:00Z"],
                                            format="ISO8601"),
            "target_log_return": [1., 2., 1., 100., 2., 4., 4., 5.],
        })
        n, value, status = _correlation(factor, _groups(examples, _events([factor])), 3)
        matched = pd.merge_asof(
            examples, pd.DataFrame({"available_at": pd.to_datetime(factor.times, utc=True),
                                    "x": factor.values}),
            on="available_at", direction="backward", tolerance=pd.Timedelta(days=2),
        ).dropna()
        self.assertEqual(status, "scored")
        self.assertEqual(n, len(matched))
        self.assertAlmostEqual(value, matched["x"].corr(matched["target_log_return"]), places=12)

    def test_constant_and_insufficient_values_are_reported(self):
        times = pd.to_datetime(["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z",
                                "2022-01-03T00:00:00Z"])
        factor = _factor("test", "constant", times, [0.1, 0.1, 0.1], 1)
        examples = pd.DataFrame({"available_at": times, "target_log_return": [1., 2., 3.]})
        groups = _groups(examples, _events([factor]))
        self.assertEqual(_correlation(factor, groups, 3), (3, None, "constant_values"))
        self.assertEqual(_correlation(factor, groups, 4), (3, None, "insufficient_pairs"))

    def test_each_stock_and_timeframe_is_separate_and_resume_reuses_outputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = root / "members.csv"
            membership.write_text("index_name,symbol,valid_from,valid_to,source,source_url\n"
                                  "Nifty 50,AAA,2022-01-01,,test,\n"
                                  "Nifty 50,BBB,2022-01-01,,test,\n", encoding="utf-8")
            stocks = root / "stocks"
            reference = stocks / "day" / "RELIANCE" / "1d.csv.gz"
            reference.parent.mkdir(parents=True)
            pd.DataFrame({"timestamp": [f"2022-01-0{x}T00:00:00+0530" for x in range(3, 8)],
                          "close": [100, 101, 102, 103, 104]}).to_csv(reference, index=False,
                                                                       compression="gzip")
            roots = [root / name for name in ("intraday", "daily", "multisession")]
            for directory in roots:
                directory.mkdir()
                jobs = [{"kind": "stock", "symbol": symbol, "timeframe": tf,
                         "status": "prepared", "chunks": 1, "rows": 3}
                        for symbol in ("AAA", "BBB") for tf in TIMEFRAMES]
                (directory / "coverage.json").write_text(json.dumps({"jobs": jobs}))
            india, treasury, prices = [root / name for name in ("india", "treasury", "prices")]
            for directory in (india, treasury, prices):
                directory.mkdir()
                (directory / "summary.json").write_text(json.dumps({"train_end_exclusive": "2022-02-01"}))
            times = pd.to_datetime(["2022-01-03T00:00:00Z", "2022-01-04T00:00:00Z",
                                    "2022-01-05T00:00:00Z"])
            factors = [_factor("external", "X", times, [1, 2, 4], 5),
                       _factor("local_daily", "AAA", times, [1, 2, 4], 5, "AAA")]
            output = root / "out"
            args = dict(membership=membership, stocks=stocks, intraday=roots[0], daily=roots[1],
                        multisession=roots[2], global_roots=[], india=india, treasury=treasury,
                        prices=prices, output=output, start=date(2022, 1, 1), end=date(2022, 2, 1),
                        minimum=3)

            def examples(symbol, *unused, **kwargs):
                values = np.array([1., 2., 4.])
                if symbol == "BBB":
                    values = -values
                return pd.DataFrame({"available_at": times, "target_log_return": values})

            with patch("modeling.nifty_candle_correlations._read_sources", return_value=factors), \
                 patch("modeling.nifty_candle_correlations.load_prepared_candles", return_value=pd.DataFrame()), \
                 patch("modeling.nifty_candle_correlations.build_candle_examples", side_effect=examples):
                summary = run(**args)
                self.assertEqual(summary["factor_count"], 2)
                for symbol, expected in (("AAA", 1.0), ("BBB", -1.0)):
                    with (output / "stocks" / f"{symbol}.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), len(TIMEFRAMES) * len(factors))
                    self.assertEqual({row["timeframe"] for row in rows}, set(TIMEFRAMES))
                    self.assertTrue(all(abs(float(row["correlation"]) - expected) < 1e-12
                                        for row in rows if row["factor"] == "X"))
                    if symbol == "AAA":
                        self.assertTrue(all(row["status"] == "target_stock_excluded"
                                            for row in rows if row["factor"] == "AAA"))
                with patch("modeling.nifty_candle_correlations.build_candle_examples",
                           side_effect=AssertionError("Completed stock was rebuilt")):
                    run(**args, resume=True)
                stopped = root / "stopped"
                stopped_args = {**args, "output": stopped, "symbols": ["AAA"]}

                def stop_after_examples(symbol, *unused, **kwargs):
                    result = examples(symbol, *unused, **kwargs)
                    TrainingControl(stopped).request("stop")
                    return result

                with patch("modeling.nifty_candle_correlations.build_candle_examples",
                           side_effect=stop_after_examples):
                    with self.assertRaises(PreparationStopped):
                        run(**stopped_args)
                self.assertFalse((stopped / "stocks" / "AAA.csv").exists())
                run(**stopped_args, resume=True)
                self.assertTrue((stopped / "stocks" / "AAA.csv").is_file())
                with membership.open("a", encoding="utf-8") as file:
                    file.write("\n")
                with self.assertRaisesRegex(ValueError, "Inputs or settings changed"):
                    run(**args, resume=True)


if __name__ == "__main__":
    unittest.main()
