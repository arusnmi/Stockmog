import io
import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped
from modeling.nifty_public_factor_research import _blob_hash, _read_returns, run


class PublicFactorResearchTests(unittest.TestCase):
    def test_returns_reject_duplicates_and_break_on_missing_reference_days(self):
        days = [date(2022, 1, day) for day in range(3, 8)]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "rates.csv"
            pd.DataFrame({"Date": [days[0], days[1], days[3], days[4]],
                          "rate": [100, 110, 120, 132]}).to_csv(path, index=False)
            result = _read_returns(path, ("rate",), days)["rates:rate"]
            self.assertAlmostEqual(result.loc[days[1]], 0.1)
            self.assertTrue(pd.isna(result.loc[days[3]]))
            self.assertAlmostEqual(result.loc[days[4]], 0.1)
            with path.open("a", encoding="utf-8") as file:
                file.write("2022-01-07,130\n")
            with self.assertRaisesRegex(ValueError, "dated and unique"):
                _read_returns(path, ("rate",), days)

    def test_stop_resume_and_detect_corrupt_pinned_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = pd.date_range("2022-01-03", periods=10)
            prices = [100, 103, 101, 105, 99, 102, 110, 108, 107, 111]
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
            output = root / "output"
            content = pd.DataFrame({"Date": dates.date, "rate": prices}).to_csv(index=False).encode()
            files = {"MACRO/a.csv": ("rate",), "MACRO/b.csv": ("rate",)}
            catalog = {name: {"bytes": len(content), "git_blob_sha1": _blob_hash(content)}
                       for name in files}
            actual_read = _read_returns
            calls = []

            def stop_after_read(path, columns, reference):
                calls.append(path.name)
                result = actual_read(path, columns, reference)
                if path.name == "a.csv":
                    TrainingControl(output).request("stop")
                return result

            with patch("modeling.nifty_public_factor_research.FILES", files), \
                 patch("modeling.nifty_public_factor_research._catalog", return_value=catalog), \
                 patch("modeling.nifty_public_factor_research.urlopen", side_effect=lambda *a, **k: io.BytesIO(content)):
                with patch("modeling.nifty_public_factor_research._read_returns", side_effect=stop_after_read):
                    with self.assertRaises(PreparationStopped):
                        run(membership, stocks, output, train_end=date(2022, 1, 13),
                            minimum_observations=3)
                self.assertEqual(calls, ["a.csv"])
                self.assertFalse((output / "summary.json").exists())
                summary = run(membership, stocks, output, train_end=date(2022, 1, 13),
                              minimum_observations=3, resume=True)
                self.assertEqual(summary["stocks"]["AAA"]["status"], "scored")
                self.assertEqual(len(json.loads((output / "stock_checkpoints" / "AAA.json").read_text())), 2)
                (output / "source" / "a.csv").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "has changed"):
                    run(membership, stocks, output, train_end=date(2022, 1, 13),
                        minimum_observations=3, resume=True)


if __name__ == "__main__":
    unittest.main()
