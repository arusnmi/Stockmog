import unittest

from data_pipeline.validation import validate_ohlcv


class ValidateOhlcvTests(unittest.TestCase):
    def test_duplicate_and_missing_candles_are_detected(self):
        rows = [
            {"timestamp": "2024-01-01", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100},
            {"timestamp": "2024-01-02", "open": 11, "high": 13, "low": 10, "close": 12, "volume": 110},
            {"timestamp": "2024-01-04", "open": 12, "high": 14, "low": 11, "close": 13, "volume": 120},
            {"timestamp": "2024-01-04", "open": 12, "high": 14, "low": 11, "close": 13, "volume": 120},
        ]

        summary = validate_ohlcv(rows, interval="1d")

        self.assertEqual(summary["total_rows"], 4)
        self.assertEqual(summary["duplicate_count"], 1)
        self.assertGreaterEqual(summary["missing_count"], 1)
        self.assertGreater(len(summary["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
