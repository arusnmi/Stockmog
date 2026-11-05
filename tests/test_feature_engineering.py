import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from config import settings
from features.feature_engineering import build_features, write_features


def _write_intraday_csv(path: Path, *, days: int = 3, bars_per_day: int = 4) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        start = datetime(2024, 1, 2, 9, 15)
        for day in range(days):
            session_date = start.date() + timedelta(days=day)
            for bar in range(bars_per_day):
                price = 100.0 + day * 10 + bar
                writer.writerow(
                    {
                        "timestamp": (
                            datetime.combine(session_date, start.time())
                            + timedelta(minutes=5 * bar)
                        ).isoformat(sep=" "),
                        "open": price,
                        "high": price + 2,
                        "low": price - 1,
                        "close": price + 1,
                        "volume": (bar + 1) * 100,
                    }
                )


def _write_daily_csv(path: Path, *, days: int = 30) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        start = datetime(2024, 1, 2)
        for day in range(days):
            price = 100.0 + day
            writer.writerow(
                {
                    "timestamp": (start + timedelta(days=day)).isoformat(sep=" "),
                    "open": price,
                    "high": price + 2,
                    "low": price - 1,
                    "close": price + 1,
                    "volume": (day + 1) * 100,
                }
            )


class FeatureEngineeringTests(unittest.TestCase):
    def test_intraday_features_are_session_aware_and_lagged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "NIFTY 50_5m.csv"
            _write_intraday_csv(input_path)

            frame = build_features(
                input_path,
                style="intraday",
                box_window=2,
                relative_volume_window=2,
                opening_range_bars=2,
            ).collect()

            self.assertEqual(frame.height, 12)
            self.assertIn("rsi_7", frame.columns)
            self.assertIn("ema_200", frame.columns)
            self.assertIn("macd_hist", frame.columns)
            self.assertIn("atr_14", frame.columns)
            self.assertIn("bb_pct_b", frame.columns)
            self.assertIn("adx_14", frame.columns)
            for column in (
                "ret_5",
                "ret_20",
                "close_location",
                "stoch_k_14",
                "stoch_d_3",
                "obv",
                "obv_sma_20",
                "plus_di_14",
                "minus_di_14",
            ):
                self.assertIn(column, frame.columns)
            self.assertTrue(all(value is None for value in frame["ret_5"][:5]))
            self.assertIsNotNone(frame["obv"][1])

            self.assertIsNone(frame["box_high"][0])
            self.assertEqual(frame["box_high"][2], 103.0)
            self.assertIsNone(frame["opening_range_high"][0])
            self.assertEqual(frame["opening_range_high"][1], 103.0)
            self.assertEqual(frame["opening_range_high"][2], 103.0)
            self.assertIsNone(frame["relative_volume"][0])
            self.assertIsNone(frame["relative_volume"][1])
            self.assertEqual(frame["relative_volume"][2], 2.0)
            self.assertEqual(frame["previous_day_high"][4], 105.0)
            self.assertAlmostEqual(frame["vwap"][0], 100.6666666667)
            self.assertAlmostEqual(frame["vwap"][1], 101.3333333333)
            self.assertIsNone(frame["box_high"][4])

    def test_market_context_uses_only_strictly_earlier_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "symbol.csv"
            market_path = root / "market.csv"
            _write_intraday_csv(input_path, days=1, bars_per_day=3)
            _write_intraday_csv(market_path, days=1, bars_per_day=3)

            market_rows = pl.read_csv(market_path).with_columns(
                (pl.col("close") + 100).alias("close")
            )
            market_rows.write_csv(market_path)
            frame = build_features(
                input_path,
                style="intraday",
                market_index_path=market_path,
            ).collect()

            self.assertIsNone(frame["market_index_close"][0])
            self.assertEqual(frame["market_index_close"][1], 101.0 + 100)

    def test_feature_output_is_csv_and_uses_configured_feature_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "NIFTY_1d.csv"
            _write_daily_csv(input_path)
            output_root = Path(temporary_directory) / "features"
            original_features_dir = settings.FEATURES_DIR
            settings.FEATURES_DIR = output_root
            try:
                output_path = write_features(input_path, style="swing", box_window=2)
            finally:
                settings.FEATURES_DIR = original_features_dir

            self.assertEqual(
                output_path,
                output_root / "swing" / "NIFTY_1d_features.csv",
            )
            self.assertEqual(output_path.suffix, ".csv")
            self.assertTrue(output_path.is_file())
            self.assertEqual(pl.scan_csv(output_path).collect().height, 30)

    def test_swing_features_use_prior_daily_bars(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "NIFTY_1d.csv"
            _write_daily_csv(input_path)

            frame = build_features(
                input_path,
                style="swing",
                box_window=10,
                relative_volume_window=5,
            ).collect()

            self.assertEqual(frame["previous_day_high"][1], 102.0)
            self.assertIsNone(frame["box_high"][9])
            self.assertEqual(frame["box_high"][10], 111.0)
            self.assertEqual(frame["relative_volume"][5], 2.0)
            self.assertNotIn("vwap", frame.columns)
            self.assertNotIn("opening_range_high", frame.columns)

    def test_changing_a_future_bar_does_not_change_earlier_features(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_path = root / "original.csv"
            changed_path = root / "changed.csv"
            _write_daily_csv(original_path)

            with original_path.open("r", newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[-1].update({"open": "900", "high": "1000", "low": "800", "close": "950"})
            with changed_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            original = build_features(original_path, style="swing").collect()
            changed = build_features(changed_path, style="swing").collect()
            for column in (
                "ema_20",
                "rsi_14",
                "macd",
                "atr_14",
                "bb_mid",
                "adx_14",
                "box_high",
                "relative_volume",
                "previous_day_high",
            ):
                self.assertEqual(original[column][20], changed[column][20], column)

    def test_flat_prices_and_zero_atr_produce_null_ratios_not_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "flat.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
                )
                writer.writeheader()
                for day in range(30):
                    writer.writerow(
                        {
                            "timestamp": (datetime(2024, 1, 1) + timedelta(days=day)).isoformat(),
                            "open": 100,
                            "high": 100,
                            "low": 100,
                            "close": 100,
                            "volume": 0,
                        }
                    )

            frame = build_features(input_path, style="swing", box_window=2).collect()

            self.assertTrue(frame["bb_pct_b"].is_null().all())
            self.assertTrue(frame["box_width_atr"].is_null().all())
            for name, dtype in frame.schema.items():
                if dtype in (pl.Float32, pl.Float64):
                    self.assertTrue(frame[name].drop_nulls().is_finite().all(), name)

    def test_database_path_is_not_part_of_project_configuration(self):
        self.assertFalse(hasattr(settings, "DB_PATH"))
        self.assertFalse(hasattr(settings, "DB_DIR"))

    def test_project_setup_does_not_create_database_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_paths = (
                settings.DATA_DIR,
                settings.RAW_DIR,
                settings.PROCESSED_DIR,
                settings.FEATURES_DIR,
            )
            settings.DATA_DIR = root / "data"
            settings.RAW_DIR = settings.DATA_DIR / "raw"
            settings.PROCESSED_DIR = settings.DATA_DIR / "processed"
            settings.FEATURES_DIR = settings.DATA_DIR / "features"
            try:
                settings.ensure_project_structure()
            finally:
                (
                    settings.DATA_DIR,
                    settings.RAW_DIR,
                    settings.PROCESSED_DIR,
                    settings.FEATURES_DIR,
                ) = original_paths

            self.assertFalse((root / "database").exists())


if __name__ == "__main__":
    unittest.main()
