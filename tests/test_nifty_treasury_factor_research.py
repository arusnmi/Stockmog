import io
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped
from modeling.global_factor_research import correlate
from modeling.nifty_treasury_factor_research import rate_factors, run


class TreasuryFactorResearchTests(unittest.TestCase):
    def test_rates_use_new_york_next_midnight_and_do_not_bridge_missing_values(self):
        frame = pd.DataFrame({"Date": ["03/08/2024", "03/11/2024", "03/12/2024", "03/13/2024"],
                              "10 Yr": [-0.5, 0.1, None, 0.3]})
        factors = rate_factors([frame], "real")
        changes = factors["real:10 Yr:change"]
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes["factor_return"].iloc[0], 0.6)
        self.assertEqual(changes["factor_available_at"].iloc[0], pd.Timestamp("2024-03-12T04:00:00Z"))
        targets = pd.DataFrame({"available_at": pd.to_datetime(["2024-03-12T03:59:00Z",
                                                                 "2024-03-12T04:00:00Z",
                                                                 "2024-03-12T04:01:00Z"]),
                                "target_return": [1.0, 2.0, 3.0]})
        self.assertIsNone(correlate(targets, changes, max_age=timedelta(days=5), minimum=3))

    def test_stop_resume_and_reject_corrupt_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = pd.date_range("2022-01-03", periods=12)
            prices = [100, 103, 101, 105, 99, 102, 110, 108, 107, 111, 109, 112]
            stocks = root / "stocks"
            for name in ("AAA", "RELIANCE"):
                directory = stocks / "day" / name
                directory.mkdir(parents=True)
                pd.DataFrame({"timestamp": [f"{day.date()}T00:00:00+0530" for day in dates],
                              "close": prices}).to_csv(directory / "1d.csv.gz", index=False,
                                                        compression="gzip")
            membership = root / "membership.csv"
            membership.write_text("index_name,symbol,valid_from,valid_to,source,source_url\n"
                                  "Nifty 50,AAA,2022-01-01,,test,\n", encoding="utf-8")
            content = pd.DataFrame({"Date": dates.strftime("%m/%d/%Y"),
                                    "Rate": prices}).to_csv(index=False).encode()
            output = root / "output"
            calls = []

            def fetch(*args, **kwargs):
                calls.append(args[0])
                if len(calls) == 1:
                    TrainingControl(output).request("stop")
                return io.BytesIO(content)

            with patch("modeling.nifty_treasury_factor_research.urlopen", side_effect=fetch):
                with self.assertRaises(PreparationStopped):
                    run(membership, stocks, output, train_end=date(2022, 1, 15),
                        minimum_observations=3)
                self.assertEqual(len(calls), 1)
                result = run(membership, stocks, output, train_end=date(2022, 1, 15),
                             minimum_observations=3, resume=True)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(result["factor_candidates"]), 6)
                self.assertEqual(result["stocks"]["AAA"]["status"], "scored")
                (output / "source" / "nominal_2022.csv").write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    run(membership, stocks, output, train_end=date(2022, 1, 15),
                        minimum_observations=3, resume=True)


if __name__ == "__main__":
    unittest.main()
