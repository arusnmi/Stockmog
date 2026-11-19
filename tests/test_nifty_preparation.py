import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch

import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped
from modeling.nifty_preparation import available_candles, build_catalog, daily_source_aggregates, join_available, prepare, report_coverage
from modeling.nifty_universe import Membership


def candles(timestamps, trading_days=1, partial=False):
    return pd.DataFrame({
        "timestamp": timestamps, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 5.0, "source_bars": trading_days,
        "trading_days": trading_days, "is_partial": partial,
    })


class NiftyPreparationTests(unittest.TestCase):
    def test_catalog_distinguishes_minute_and_daily_source_coverage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stocks = root / "stocks"
            for group, symbol in [("minute", "RELIANCE"), ("day", "RELIANCE"), ("day", "HDFC")]:
                directory = stocks / group / symbol
                directory.mkdir(parents=True)
                candles(["2022-01-03T00:00:00+0530"]).to_csv(
                    directory / "1d.csv.gz", index=False, compression="gzip")
            membership = root / "membership.csv"
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,HDFC,2022-01-03,2022-01-04,test,\n", encoding="utf-8")
            result = build_catalog(membership, stocks, root / "catalog")
            hdfc = next(item for item in result["records"] if item["symbol"] == "HDFC")
            self.assertIsNone(hdfc["daily_first"])
            self.assertEqual(hdfc["daily_source"]["sessions"], 1)
            self.assertEqual(hdfc["daily_source"]["missing_reference_sessions"], [])

    def test_availability_uses_candle_end_and_daily_session_close(self):
        minute = available_candles(candles(["2022-01-03T15:15:00+0530"]), "1h")
        self.assertEqual(minute["available_at"][0], pd.Timestamp("2022-01-03T10:00:00Z"))
        daily = available_candles(candles(["2022-01-03T00:00:00+0530"]), "1d")
        self.assertEqual(daily["available_at"][0], pd.Timestamp("2022-01-03T10:00:00Z"))
        derived = available_candles(candles(["2022-01-03T09:15:00+0530"]), "1d")
        self.assertEqual(derived["available_at"][0], pd.Timestamp("2022-01-03T10:00:00Z"))
        special = available_candles(candles(["2022-10-24T00:00:00+0530"]), "1d", daily_source=True)
        self.assertEqual(special["available_at"][0], pd.Timestamp("2022-10-24T18:30:00Z"))

    def test_multiday_availability_and_membership_do_not_look_forward(self):
        start, final = date(2022, 1, 3), date(2022, 1, 4)
        frame = candles(["2022-01-03T00:00:00+0530"], trading_days=2)
        two = available_candles(frame, "2d", {start: (final, 2)})
        self.assertEqual(two["available_at"][0], pd.Timestamp("2022-01-04T10:00:00Z"))
        weekly = available_candles(frame, "trading_week", {start: (final, 2)})
        self.assertEqual(weekly["available_at"][0], pd.Timestamp("2022-01-09T18:30:00Z"))
        membership = (Membership("AAA", start, final, "test", ""),)
        self.assertTrue(available_candles(frame, "2d", {start: (final, 2)}, membership).empty)
        partial = candles(["2022-01-03T00:00:00+0530"], partial=True)
        self.assertTrue(available_candles(partial, "2d", {start: (start, 1)}).empty)
        with self.assertRaisesRegex(ValueError, "counts"):
            available_candles(frame, "2d", {start: (final, 1)})

    def test_asof_join_excludes_future_and_stale_factor_values(self):
        stock = pd.DataFrame({"available_at": pd.to_datetime(
            ["2022-01-03T10:00Z", "2022-01-03T11:00Z"], utc=True)})
        factor = pd.DataFrame({"available_at": pd.to_datetime(
            ["2022-01-03T10:30Z"], utc=True), "close": [7]})
        joined = join_available(stock, factor, tolerance=timedelta(minutes=40))
        self.assertTrue(pd.isna(joined["close"][0]))
        self.assertEqual(joined["close"][1], 7)
        stale = join_available(stock, factor, tolerance=timedelta(minutes=10))
        self.assertTrue(stale["close"].isna().all())
        stock["available_at"] = stock["available_at"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "explicit timezones"):
            join_available(stock, factor, tolerance=timedelta(minutes=40))

    def test_daily_source_aggregates_use_real_sessions_and_conservative_availability(self):
        frame = candles(["2022-10-24T00:00:00+0530", "2022-10-25T00:00:00+0530",
                         "2022-10-27T00:00:00+0530"])
        two = daily_source_aggregates(frame, "2d")
        self.assertEqual(len(two), 1)
        self.assertEqual(two["available_at"][0], pd.Timestamp("2022-10-25T18:30Z"))
        week = daily_source_aggregates(frame, "trading_week")
        self.assertEqual(week["available_at"][0], pd.Timestamp("2022-10-30T18:30Z"))
        self.assertTrue(week["is_partial"][0])
        member = (Membership("AAA", date(2022, 10, 24), date(2022, 10, 25), "test", ""),)
        self.assertTrue(daily_source_aggregates(frame, "2d", member).empty)

    def test_pause_stop_resume_and_changed_input_rejection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stocks = root / "stocks"
            (stocks / "day").mkdir(parents=True)
            directory = stocks / "minute" / "AAA"
            directory.mkdir(parents=True)
            source = directory / "1m.csv.gz"
            candles(["2022-01-03T09:15:00+0530", "2022-01-03T09:16:00+0530",
                     "2022-01-03T09:17:00+0530"]).to_csv(source, index=False)
            membership = root / "membership.csv"
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-03,2022-01-04,test,\n", encoding="utf-8")
            output = root / "prepared"
            started, release = Event(), Event()
            errors = []
            options = dict(stock_symbols=["AAA"], factor_symbols=[], timeframes=["1m"], chunk_rows=1)

            def wait_for_pause(*args, **kwargs):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release first chunk")
                return available_candles(*args, **kwargs)

            def run():
                try:
                    prepare(membership, stocks, output, **options)
                except PreparationStopped:
                    pass
                except Exception as exc:
                    errors.append(exc)

            with patch("modeling.nifty_preparation.available_candles", side_effect=wait_for_pause):
                thread = Thread(target=run)
                thread.start()
                try:
                    self.assertTrue(started.wait(5))
                    control = TrainingControl(output, poll_seconds=0.01)
                    control.request("pause")
                    release.set()
                    deadline = monotonic() + 5
                    while control.action() != "paused" and monotonic() < deadline:
                        sleep(0.01)
                    self.assertEqual(control.action(), "paused")
                    control.request("stop")
                    thread.join(10)
                finally:
                    release.set()
                    if thread.is_alive():
                        TrainingControl(output).request("stop")
                        thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            first = output / "stock" / "AAA" / "1m" / "part-000000.parquet"
            self.assertTrue(first.is_file())
            before = first.stat().st_mtime_ns
            result = prepare(membership, stocks, output, resume=True, **options)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["jobs"][0]["rows"], 3)
            self.assertEqual(first.stat().st_mtime_ns, before)
            self.assertEqual(len(list(first.parent.glob("*.parquet"))), 3)
            self.assertEqual(report_coverage(output)["jobs"][0]["rows"], 3)
            membership.write_text(membership.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inputs or settings changed"):
                prepare(membership, stocks, output, resume=True, **options)
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-04,2022-01-05,test,\n", encoding="utf-8")
            empty = prepare(membership, stocks, root / "empty", **options)
            self.assertEqual(empty["jobs"][0]["status"], "no_eligible_rows")
            self.assertEqual(empty["status"], "complete_with_gaps")


if __name__ == "__main__":
    unittest.main()
