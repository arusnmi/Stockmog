import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import forecast_stock_prices as forecast


class StockForecastTests(unittest.TestCase):
    def test_next_friday_uses_a_future_friday(self):
        self.assertEqual(forecast.next_friday(date(2026, 11, 14)), date(2026, 11, 20))
        self.assertEqual(forecast.next_friday(date(2026, 11, 20)), date(2026, 11, 27))

    def test_calendar_includes_special_sessions_and_excludes_holidays(self):
        days = dict(forecast.calendar_days(
            date(2026, 1, 30), date(2026, 2, 3), {date(2026, 2, 3): "test closure"},
        ))
        self.assertEqual(days[date(2026, 1, 31)], "weekend")
        self.assertEqual(days[date(2026, 2, 1)], "special: Union Budget")
        self.assertEqual(days[date(2026, 2, 2)], "trading")
        self.assertEqual(days[date(2026, 2, 3)], "holiday: test closure")

    def test_trend_uses_cutoff_prices_and_trading_sessions(self):
        sessions = [date(2026, 1, 30), date(2026, 2, 1)]
        predictions = forecast.make_forecasts([100.0] * 20 + [110.0], sessions)
        self.assertAlmostEqual(predictions[sessions[0]], 110 * 1.1 ** (1 / 20))
        self.assertAlmostEqual(predictions[sessions[1]], 110 * 1.1 ** (2 / 20))
        with self.assertRaisesRegex(ValueError, "21 valid"):
            forecast.make_forecasts([100.0] * 20, sessions)

    def test_csv_includes_only_trading_sessions(self):
        cutoff = forecast.CUTOFF
        prices = {cutoff - timedelta(days=i): 100.0 for i in range(40)}
        prices[date(2026, 1, 15)] = 1.0

        def fake_history(symbol, start, end):
            if end == cutoff:
                return prices
            return {date(2026, 1, 30): 101.0, date(2026, 2, 2): 102.0}

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(forecast, "market_clock", return_value=datetime(2026, 2, 2, 17)), \
                patch.object(forecast, "fetch_holidays",
                             return_value={date(2026, 1, 15): "Election",
                                           date(2026, 2, 3): "test closure"}), \
                patch.object(forecast, "history", side_effect=fake_history), \
                patch.object(forecast, "special_closes",
                             return_value={symbol: 99.0 for symbol in forecast.SYMBOLS}), \
                patch.object(forecast, "write_chart"):
            output = forecast.run(date(2026, 2, 4), Path(temp))
            with (output / "daily_prices.csv").open(encoding="utf-8", newline="") as stream:
                rows = {(row["symbol"], row["date"]): row for row in csv.DictReader(stream)}
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertNotIn(("SAKAR", "2026-01-31"), rows)
        self.assertNotIn(("SAKAR", "2026-02-03"), rows)
        self.assertEqual(rows["SAKAR", "2026-02-01"]["market_status"], "special: Union Budget")
        self.assertEqual(rows["SAKAR", "2026-02-01"]["actual_close"], "99.00")
        self.assertEqual(rows["SAKAR", "2026-02-04"]["actual_close"], "")
        self.assertEqual(rows["SAKAR", "2026-02-04"]["forecast"], "100.00")
        self.assertEqual(metadata["compared_stock_days"], 15)
        expected = (1 / 101 + 1 / 99 + 2 / 102) / 3 * 100
        self.assertEqual(metadata["mean_absolute_percentage_error"], f"{expected:.2f}")

    def test_error_unavailable_without_actual_closes(self):
        prices = {forecast.CUTOFF - timedelta(days=i): 100.0 for i in range(40)}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(forecast, "market_clock", return_value=datetime(2026, 1, 30, 10)), \
                patch.object(forecast, "fetch_holidays", return_value={}), \
                patch.object(forecast, "history", return_value=prices), \
                patch.object(forecast, "write_chart"):
            output = forecast.run(date(2026, 1, 30), Path(temp))
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["compared_stock_days"], 0)
        self.assertIsNone(metadata["mean_absolute_percentage_error"])

    def test_latest_uses_completed_session_and_includes_special_history(self):
        now = datetime(2026, 11, 15, 14)
        history_dates = {
            now.date() - timedelta(days=i): 100.0 for i in range(60)
            if now.date() - timedelta(days=i) != date(2026, 11, 8)
        }
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(forecast, "market_clock", return_value=now), \
                patch.object(forecast, "fetch_holidays",
                             return_value={date(2026, 11, 10): "Diwali"}), \
                patch.object(forecast, "history", return_value=history_dates) as history, \
                patch.object(forecast, "special_closes",
                             return_value={symbol: 105.0 for symbol in forecast.SYMBOLS}) as special, \
                patch.object(forecast, "write_chart") as chart:
            output = forecast.run(None, Path(temp), latest=True)
            with (output / "daily_prices.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertTrue(output.name.startswith("forecast_latest_2026-11-13_2026-11-20_"))
        self.assertEqual(metadata["cutoff"], "2026-11-13")
        self.assertEqual(metadata["compared_stock_days"], 0)
        self.assertEqual({row["date"] for row in rows},
                         {f"2026-11-{day}" for day in ("16", "17", "18", "19", "20")})
        self.assertTrue(all(call.args[2] == date(2026, 11, 13) for call in history.call_args_list))
        special.assert_called_once_with(date(2026, 11, 8), forecast.SYMBOLS)
        self.assertEqual(chart.call_args.args[3], date(2026, 11, 13))


if __name__ == "__main__":
    unittest.main()
