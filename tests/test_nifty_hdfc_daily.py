import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from data_pipeline.nifty_hdfc_daily import MEMBER, collect, normalize
from modeling.factor_research import _write_json


class HdfcDailyTests(unittest.TestCase):
    def test_identical_duplicates_are_removed_and_missing_dates_reported(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-04"]),
            "open": [100.0, 100.0, 101.0], "high": [102.0, 102.0, 103.0],
            "low": [99.0, 99.0, 100.0], "close": [101.0, 101.0, 102.0],
            "volume": [20, 20, 30],
        })
        result, report = normalize(frame, {date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)})
        self.assertEqual(len(result), 2)
        self.assertEqual(report["identical_rows_removed"], 1)
        self.assertEqual(report["missing_reference_sessions"], ["2022-01-05"])
        self.assertEqual(result["timestamp"][0], "2022-01-03T00:00:00+0530")

    def test_conflicting_duplicates_are_rejected(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "open": [100.0, 100.0], "high": [102.0, 102.0], "low": [99.0, 99.0],
            "close": [101.0, 102.0], "volume": [20, 20],
        })
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            normalize(frame, {date(2022, 1, 3)})

    def test_interrupted_publication_recovers_without_overwriting(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2022-01-03", "2022-01-04"]),
                "open": [100.0, 101.0], "high": [102.0, 103.0], "low": [99.0, 100.0],
                "close": [101.0, 102.0], "volume": [20, 30],
            }).to_parquet(source)
            reference = root / "reference.csv.gz"
            pd.DataFrame({"timestamp": ["2022-01-03T00:00:00+0530", "2022-01-04T00:00:00+0530"]}).to_csv(reference, index=False)
            output, destination = root / "run", root / "day" / "HDFC" / "1d.csv.gz"

            def fake_download(path, control):
                with ZipFile(path, "w") as archive:
                    archive.write(source, MEMBER)

            def crash_before_summary(path, data):
                if path.name == "summary.json":
                    raise OSError("Simulated interruption")
                _write_json(path, data)

            with patch("data_pipeline.nifty_hdfc_daily.download_archive", side_effect=fake_download):
                with patch("data_pipeline.nifty_hdfc_daily._write_json", side_effect=crash_before_summary):
                    with self.assertRaisesRegex(OSError, "Simulated interruption"):
                        collect(reference, output, destination)
                self.assertTrue(destination.is_file())
                original = destination.read_bytes()
                result = collect(reference, output, destination, resume=True)
            self.assertEqual(result["sessions"], 2)
            self.assertEqual(destination.read_bytes(), original)
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
