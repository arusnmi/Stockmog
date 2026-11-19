import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd

from data_pipeline.nifty_hdfc_archive import collect


class HdfcArchiveTests(unittest.TestCase):
    def test_quarantines_wrong_date_and_reports_unavailable_months(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "1d.csv.gz"
            pd.DataFrame({"timestamp": [
                "2022-01-03T00:00:00+0530", "2022-01-04T00:00:00+0530",
                "2023-03-01T00:00:00+0530",
            ]}).to_csv(reference, index=False)
            valid = b"HDFC,20220103,09:15,100,101,99,100,10,0\n"
            wrong_date = b"HDFC,20220103,09:15,100,101,99,100,10,0\n"
            responses = []
            for content in (valid, wrong_date):
                response = MagicMock()
                response.__enter__.return_value.read.return_value = content
                responses.append(response)
            output = root / "archive"
            with patch("data_pipeline.nifty_hdfc_archive.urlopen", side_effect=responses):
                summary = collect(reference, output)
            self.assertEqual([entry["status"] for entry in summary["sessions"]],
                             ["downloaded", "invalid_source", "not_in_archive"])
            self.assertTrue((output / "raw" / "2022-01-04.invalid.txt").is_file())
            self.assertFalse((output / "raw" / "2022-01-04.txt").exists())
            with patch("data_pipeline.nifty_hdfc_archive.urlopen", side_effect=AssertionError("unexpected download")):
                resumed = collect(reference, output, resume=True)
            self.assertEqual([entry["status"] for entry in resumed["sessions"]],
                             ["downloaded", "invalid_source", "not_in_archive"])
            reference.write_bytes(reference.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                collect(reference, output, resume=True)


if __name__ == "__main__":
    unittest.main()
