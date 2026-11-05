from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from data_pipeline.csv_dataset import scan_training_csv
from data_pipeline.huggingface_stocks import (
    _process_parquet_shard,
    _prepare_stock_frame,
    _scan_minute_date_range,
    add_stock_indicators,
    load_nse_holidays,
    prepare_huggingface_dataset,
    process_stock_frame,
    resample_bi_daily,
    resample_intraday,
    resample_trading_week,
)


def _minute_frame(*, rows: int = 65) -> pl.DataFrame:
    start = datetime(2024, 1, 2, 3, 45)
    timestamps = pl.datetime_range(
        start,
        start + timedelta(minutes=rows - 1),
        interval="1m",
        time_zone="UTC",
        eager=True,
    )
    prices = [100.0 + index / 10 for index in range(rows)]
    return pl.DataFrame(
        {
            "symbol": ["TEST"] * rows,
            "timestamp": timestamps,
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": [price + 0.5 for price in prices],
            "volume": [100] * rows,
            "oi": [0] * rows,
        }
    )


class HuggingFaceStockDataTests(unittest.TestCase):
    def test_intraday_resampling_uses_ist_session_buckets_and_volume(self):
        source = _minute_frame(rows=65)

        one_minute = resample_intraday(source, minutes=1, symbol="TEST")
        five_minute = resample_intraday(source, minutes=5, symbol="TEST")
        fifteen_minute = resample_intraday(source, minutes=15, symbol="TEST")
        twenty_minute = resample_intraday(source, minutes=20, symbol="TEST")
        thirty_minute = resample_intraday(source, minutes=30, symbol="TEST")
        hourly = resample_intraday(source, minutes=60, symbol="TEST")

        self.assertEqual(one_minute.height, 65)
        self.assertEqual(one_minute["source_bars"][0], 1)
        self.assertEqual(five_minute.height, 13)
        self.assertEqual(five_minute["timestamp"][0].hour, 9)
        self.assertEqual(five_minute["timestamp"][0].minute, 15)
        self.assertEqual(five_minute["volume"][0], 500)
        self.assertEqual(five_minute["source_bars"][0], 5)
        self.assertFalse(five_minute["is_partial"][0])

        self.assertEqual(fifteen_minute.height, 5)
        self.assertFalse(fifteen_minute["is_partial"][0])
        self.assertTrue(fifteen_minute["is_partial"][-1])

        self.assertEqual(twenty_minute.height, 4)
        self.assertEqual(twenty_minute["source_bars"].to_list(), [20, 20, 20, 5])
        self.assertTrue(twenty_minute["is_partial"][-1])

        self.assertEqual(thirty_minute.height, 3)
        self.assertTrue(thirty_minute["is_partial"][-1])

        self.assertEqual(hourly.height, 2)
        self.assertEqual(hourly["source_bars"][-1], 5)
        self.assertTrue(hourly["is_partial"][-1])

    def test_trading_week_skips_weekends_and_nse_holiday(self):
        daily = pl.DataFrame(
            {
                "symbol": ["TEST"] * 7,
                "timestamp": pl.datetime_range(
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 7),
                    interval="1d",
                    time_zone="UTC",
                    eager=True,
                ),
                "open": [100.0 + i for i in range(7)],
                "high": [110.0 + i for i in range(7)],
                "low": [90.0 + i for i in range(7)],
                "close": [105.0 + i for i in range(7)],
                "volume": [1000] * 7,
                "oi": [0] * 7,
            }
        )

        weekly = resample_trading_week(
            daily,
            symbol="TEST",
            holidays={date(2024, 1, 1)},
        )

        self.assertEqual(weekly.height, 1)
        self.assertEqual(weekly["open"][0], 101.0)
        self.assertEqual(weekly["high"][0], 114.0)
        self.assertEqual(weekly["low"][0], 91.0)
        self.assertEqual(weekly["close"][0], 109.0)
        self.assertEqual(weekly["volume"][0], 4000)
        self.assertEqual(weekly["source_bars"][0], 4)
        self.assertEqual(weekly["trading_days"][0], 4)
        self.assertEqual(weekly["expected_trading_days"][0], 4)
        self.assertFalse(weekly["is_partial"][0])

    def test_bi_daily_groups_consecutive_trading_sessions_around_holiday(self):
        daily = pl.DataFrame(
            {
                "symbol": ["TEST"] * 5,
                "timestamp": pl.datetime_range(
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 5),
                    interval="1d",
                    time_zone="UTC",
                    eager=True,
                ),
                "open": [100.0 + i for i in range(5)],
                "high": [110.0 + i for i in range(5)],
                "low": [90.0 + i for i in range(5)],
                "close": [105.0 + i for i in range(5)],
                "volume": [1000] * 5,
                "oi": [0] * 5,
            }
        )

        two_day = resample_bi_daily(
            daily,
            symbol="TEST",
            holidays={date(2024, 1, 2)},
        )

        self.assertEqual(two_day.height, 2)
        self.assertEqual(two_day["trading_days"].to_list(), [2, 2])
        self.assertEqual(two_day["volume"].to_list(), [2000, 2000])
        self.assertEqual(two_day["is_partial"].to_list(), [False, False])

    def test_indicators_include_requested_calculations_and_compressed_csv_is_readable(self):
        candles = resample_intraday(_minute_frame(rows=220), minutes=5, symbol="TEST")
        indicators = add_stock_indicators(candles, timeframe="5m").collect()

        for column in (
            "rsi_14",
            "macd",
            "macd_signal",
            "ema_20",
            "sma_20",
            "atr_14",
            "adx_14",
            "vwap",
            "bb_upper",
            "bb_mid",
            "bb_lower",
        ):
            self.assertIn(column, indicators.columns)
        self.assertTrue(indicators["vwap"].is_not_null().all())

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = process_stock_frame(
                _minute_frame(rows=65),
                symbol="TEST",
                split="minute",
                output_root=temporary_directory,
            )
            self.assertEqual(len(paths), 9)
            self.assertTrue(all(path.name.endswith(".csv.gz") for path in paths))
            scanned = scan_training_csv(paths[0]).collect()
            self.assertGreater(scanned.height, 0)
            self.assertIn("rsi_14", scanned.columns)
            self.assertTrue((Path(temporary_directory) / "minute" / "TEST" / "1d.csv.gz").is_file())
            self.assertTrue(
                (Path(temporary_directory) / "minute" / "TEST" / "2d.csv.gz").is_file()
            )
            self.assertTrue(
                (Path(temporary_directory) / "minute" / "TEST" / "trading_week.csv.gz").is_file()
            )

    def test_invalid_ohlc_rows_are_dropped_with_a_warning(self):
        source = _minute_frame(rows=3).with_row_index("_row").with_columns(
            pl.when(pl.col("_row") == 0)
            .then(pl.col("open") + 0.01)
            .otherwise(pl.col("low"))
            .alias("low")
        ).drop("_row")

        warning = StringIO()
        with redirect_stdout(warning):
            cleaned = _prepare_stock_frame(source)

        self.assertEqual(cleaned.height, 2)
        self.assertIn("dropped 1 invalid OHLC row(s) for TEST", warning.getvalue())

    def test_parquet_shard_streams_symbols_across_batch_boundaries(self):
        timestamps = pl.datetime_range(
            datetime(2024, 1, 2, 3, 45),
            datetime(2024, 1, 2, 3, 49),
            interval="1m",
            time_zone="UTC",
            eager=True,
        ).to_list()
        timestamps = timestamps + timestamps
        prices = [100.0 + index for index in range(10)]
        table = pa.table(
            {
                "symbol": ["TEST"] * 5 + ["OTHER"] * 5,
                "timestamp": pa.array(
                    timestamps,
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "open": prices,
                "high": [price + 1 for price in prices],
                "low": [price - 1 for price in prices],
                "close": [price + 0.5 for price in prices],
                "volume": [100] * 10,
                "oi": [0] * 10,
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            parquet_path = Path(temporary_directory) / "tiny.parquet"
            output_root = Path(temporary_directory) / "output"
            pq.write_table(table, parquet_path, row_group_size=4)
            seen_symbols = set()

            processed, source_rows, minute_start, minute_end, pending = _process_parquet_shard(
                parquet_path,
                split="minute",
                output_root=output_root,
                selected_symbols={"TEST", "OTHER"},
                seen_symbols=seen_symbols,
                batch_size=3,
                holidays=set(),
            )

            self.assertEqual(processed, 2)
            self.assertEqual(source_rows, 10)
            self.assertEqual(minute_start, date(2024, 1, 2))
            self.assertEqual(minute_end, date(2024, 1, 2))
            self.assertIsNone(pending)
            self.assertEqual(seen_symbols, {"TEST", "OTHER"})
            self.assertEqual(len(list(output_root.rglob("*.csv.gz"))), 18)
            self.assertEqual(
                _scan_minute_date_range(parquet_path, batch_size=3),
                (date(2024, 1, 2), date(2024, 1, 2)),
            )

    def test_parquet_shard_boundary_merges_continuous_symbol_rows(self):
        table = _minute_frame(rows=10).to_arrow()

        with tempfile.TemporaryDirectory() as temporary_directory:
            first_shard = Path(temporary_directory) / "first.parquet"
            second_shard = Path(temporary_directory) / "second.parquet"
            output_root = Path(temporary_directory) / "output"
            pq.write_table(table.slice(0, 5), first_shard, row_group_size=3)
            pq.write_table(table.slice(5, 5), second_shard, row_group_size=3)
            seen_symbols = set()

            first_result = _process_parquet_shard(
                first_shard,
                split="minute",
                output_root=output_root,
                selected_symbols={"TEST"},
                seen_symbols=seen_symbols,
                batch_size=3,
                holidays=set(),
                defer_last_symbol=True,
            )
            processed, source_rows, _, _, pending = _process_parquet_shard(
                second_shard,
                split="minute",
                output_root=output_root,
                selected_symbols={"TEST"},
                seen_symbols=seen_symbols,
                batch_size=3,
                holidays=set(),
                pending_symbol=first_result[4],
            )

            self.assertEqual(first_result[:2], (0, 0))
            self.assertEqual(processed, 1)
            self.assertEqual(source_rows, 10)
            self.assertIsNone(pending)
            self.assertEqual(seen_symbols, {"TEST"})
            five_minute = scan_training_csv(
                output_root / "minute" / "TEST" / "5m.csv.gz"
            ).collect()
            self.assertEqual(five_minute.height, 2)
            self.assertEqual(five_minute["source_bars"].to_list(), [5, 5])

    def test_daily_output_is_clipped_to_dataset_wide_minute_date_range(self):
        daily = pl.DataFrame(
            {
                "symbol": ["TEST"] * 5,
                "timestamp": pl.datetime_range(
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 5),
                    interval="1d",
                    time_zone="UTC",
                    eager=True,
                ),
                "open": [100.0 + i for i in range(5)],
                "high": [110.0 + i for i in range(5)],
                "low": [90.0 + i for i in range(5)],
                "close": [105.0 + i for i in range(5)],
                "volume": [1000] * 5,
                "oi": [0] * 5,
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            process_stock_frame(
                daily,
                symbol="TEST",
                split="day",
                output_root=temporary_directory,
                holidays=set(),
                date_range=(date(2024, 1, 2), date(2024, 1, 4)),
            )
            one_day = scan_training_csv(
                Path(temporary_directory) / "day" / "TEST" / "1d.csv.gz"
            ).collect()

            self.assertEqual(
                one_day["timestamp"].dt.date().to_list(),
                [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            )

    def test_empty_symbol_filter_is_rejected_before_downloading(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            prepare_huggingface_dataset(symbols=[])

    def test_holiday_csv_requires_valid_dates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calendar_path = Path(temporary_directory) / "holidays.csv"
            calendar_path.write_text(
                "date,name\n2024-01-26,Republic Day\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_nse_holidays(calendar_path),
                {date(2024, 1, 26)},
            )

            calendar_path.write_text("date,name\nnot-a-date,Holiday\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_nse_holidays(calendar_path)

    def test_parquet_file_is_unlocked_after_schema_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            parquet_path = Path(temporary_directory) / "missing_symbol.parquet"
            pq.write_table(
                pa.table({"value": [1]}),
                parquet_path,
            )
            with self.assertRaisesRegex(ValueError, "no symbol column"):
                _process_parquet_shard(
                    parquet_path,
                    split="minute",
                    output_root=Path(temporary_directory) / "output",
                    selected_symbols=None,
                    seen_symbols=set(),
                    batch_size=10,
                    holidays=set(),
                )
            parquet_path.unlink()
            self.assertFalse(parquet_path.exists())


if __name__ == "__main__":
    unittest.main()
