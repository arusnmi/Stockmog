import csv
import tempfile
import unittest
from pathlib import Path

from data_pipeline.csv_import import import_csv, infer_symbol_and_interval
from data_pipeline.csv_dataset import inspect_training_csv, scan_training_csv
from data_pipeline.storage import CSVStore


class CsvImportTests(unittest.TestCase):
    def test_kaggle_filename_infers_symbol_and_interval(self):
        symbol, interval = infer_symbol_and_interval("NIFTY 50_5minute.csv")

        self.assertEqual(symbol, "NIFTY 50")
        self.assertEqual(interval, "5m")

    def test_import_writes_normalized_training_csv_and_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "NIFTY 50_5minute.csv"
            output_path = root / "normalized" / "NIFTY 50_5m.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Date Time", "Open", "High", "Low", "Close"])
                writer.writerow(["2024-01-02 09:15:00", "100", "102", "99", "101"])
                writer.writerow(["2024-01-02 09:20:00", "101", "103", "100", "102"])
                writer.writerow(["2024-01-02 09:20:00", "101", "103", "100", "102"])
                writer.writerow(["2024-01-02 09:30:00", "102", "104", "101", "103"])
                writer.writerow(["2024-01-02 09:35:00", "103", "102", "104", "101"])
                writer.writerow(["2024-01-03 09:15:00", "103", "105", "102", "104"])

            result = import_csv(csv_path, output_path=output_path)

            self.assertEqual(result["rows_read"], 6)
            self.assertEqual(result["rows_written"], 4)
            self.assertEqual(result["duplicate_rows"], 1)
            self.assertEqual(result["invalid_rows"], 1)
            self.assertEqual(result["missing_candles"], 1)
            self.assertEqual(Path(result["output"]), output_path)
            self.assertTrue(Path(result["errors_file"]).is_file())

            stored_rows = scan_training_csv(output_path).collect()
            self.assertEqual(stored_rows.height, 4)
            self.assertTrue(stored_rows["volume"].is_null().all())

            summary = inspect_training_csv(output_path)
            self.assertEqual(summary["rows"], 4)
            self.assertEqual(summary["first_timestamp"], "2024-01-02 09:15:00")
            self.assertEqual(summary["last_timestamp"], "2024-01-03 09:15:00")

    def test_import_accepts_separate_date_and_time_columns_with_overrides(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "custom.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Date", "Time", "Open", "High", "Low", "Close", "Volume"])
                writer.writerow(["2024-01-02", "09:15:00", "100", "102", "99", "101", "500"])

            output_path = root / "normalized.csv"
            result = import_csv(
                csv_path,
                symbol="NIFTY 50",
                interval="1m",
                output_path=output_path,
            )

            self.assertEqual(result["rows_written"], 1)
            stored_row = scan_training_csv(output_path).collect().row(0, named=True)
            self.assertEqual(str(stored_row["timestamp"]), "2024-01-02 09:15:00")
            self.assertEqual(stored_row["volume"], 500.0)

    def test_csv_store_writes_direct_training_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = CSVStore(temporary_directory)
            path = store.save_rows(
                "NIFTY 50",
                "5m",
                [
                    {
                        "timestamp": "2024-01-02 09:15:00",
                        "open": 100,
                        "high": 102,
                        "low": 99,
                        "close": 101,
                        "volume": None,
                    }
                ],
            )

            frame = scan_training_csv(path).collect()

            self.assertEqual(path.suffix, ".csv")
            self.assertEqual(frame.height, 1)
            self.assertEqual(frame["close"][0], 101)


if __name__ == "__main__":
    unittest.main()
