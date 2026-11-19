import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch

import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped, consecutive_returns, load_closes, run_audit, screen_factors
from modeling.nifty_universe import Membership


class FactorResearchTests(unittest.TestCase):
    def test_screens_next_day_returns_without_using_cutoff_or_later_labels(self):
        days = [date(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        prices = pd.Series([100, 103, 101, 105, 99, 102, 110, 108, 107, 111], index=days)
        next_returns = prices.shift(-1).div(prices).sub(1)
        factors = pd.DataFrame({"positive": next_returns, "negative": -next_returns})
        membership = (Membership("TEST", days[0], None, "test", ""),)
        result = screen_factors(
            prices, factors, membership, train_end=days[7], minimum_observations=3, reference_sessions=days
        )
        self.assertEqual([(name, count) for name, count, _ in result], [
            ("positive", 6), ("negative", 6)
        ])
        self.assertAlmostEqual(result[0][2], 1)
        self.assertAlmostEqual(result[1][2], -1)
        changed = prices.copy()
        changed.loc[days[7]:] *= 10
        self.assertEqual(
            result,
            screen_factors(changed, factors, membership, train_end=days[7], minimum_observations=3, reference_sessions=days),
        )

    def test_excludes_dates_outside_membership(self):
        days = [date(2022, 1, 1) + timedelta(days=i) for i in range(10)]
        prices = pd.Series([100, 103, 101, 105, 99, 102, 110, 108, 107, 111], index=days)
        factors = pd.DataFrame({"factor": prices.shift(-1).div(prices).sub(1)})
        membership = (Membership("TEST", days[2], days[6], "test", ""),)
        result = screen_factors(
            prices, factors, membership, train_end=days[8], minimum_observations=3, reference_sessions=days
        )
        self.assertEqual(result[0][1], 3)

    def test_missing_reference_session_breaks_stock_and_factor_returns(self):
        days = [date(2022, 1, 3) + timedelta(days=i) for i in range(7)]
        prices = pd.Series([100, 103, 105, 99, 102, 110], index=days[:2] + days[3:])
        factors = pd.DataFrame({"factor": prices.shift(-1).div(prices).sub(1)})
        member = (Membership("TEST", days[0], None, "test", ""),)
        result = screen_factors(prices, factors, member, train_end=days[6],
                                minimum_observations=3, reference_sessions=days)
        self.assertEqual(result[0][1], 3)
        returns = consecutive_returns(prices, days)
        self.assertTrue(pd.isna(returns.loc[days[3]]))
        self.assertAlmostEqual(returns.loc[days[4]], 99 / 105 - 1)

    def _fixture(self, root):
        stocks = root / "stocks"
        dates = pd.date_range("2022-01-03", periods=12)
        for symbol in ("NIFTY_FACTOR", "AAA", "RELIANCE"):
            directory = stocks / "day" / symbol
            directory.mkdir(parents=True)
            pd.DataFrame({
                "timestamp": [f"{d.date()}T00:00:00+0530" for d in dates],
                "close": [100, 103, 101, 105, 99, 102, 110, 108, 107, 111, 109, 112],
            }).to_csv(directory / "1d.csv.gz", index=False, compression="gzip")
        membership = root / "membership.csv"
        membership.write_text(
            "index_name,symbol,valid_from,valid_to,source,source_url\n"
            "Nifty 50,AAA,2022-01-01,,test,\n",
            encoding="utf-8",
        )
        return membership, stocks, root / "audit"

    def test_stop_and_resume_reuses_completed_factor_and_checks_inputs(self):
        with TemporaryDirectory() as temporary:
            membership, stocks, output = self._fixture(Path(temporary))
            calls = []

            def stop_after_first_factor(path):
                calls.append(path.parent.name)
                result = load_closes(path)
                if path.parent.name == "NIFTY_FACTOR":
                    TrainingControl(output).request("stop")
                return result

            with patch("modeling.factor_research.load_closes", side_effect=stop_after_first_factor):
                with self.assertRaises(PreparationStopped):
                    run_audit(membership, stocks, output, train_end=date(2022, 1, 14),
                              minimum_observations=3)
            self.assertTrue((output / "factor_checkpoints" / "NIFTY_FACTOR.csv").is_file())
            self.assertFalse((output / "summary.json").exists())
            calls.clear()

            def record(path):
                calls.append(path.parent.name)
                return load_closes(path)

            with patch("modeling.factor_research.load_closes", side_effect=record):
                run_audit(membership, stocks, output, train_end=date(2022, 1, 14),
                          minimum_observations=3, resume=True)
            self.assertEqual(calls, ["RELIANCE", "AAA"])
            self.assertTrue((output / "summary.json").is_file())
            with membership.open("a", encoding="utf-8") as file:
                file.write("\n")
            with self.assertRaisesRegex(ValueError, "Inputs or settings changed"):
                run_audit(membership, stocks, output, train_end=date(2022, 1, 14),
                          minimum_observations=3, resume=True)

    def test_peer_candidates_exclude_target_stock_and_keep_existing_policy(self):
        import json

        with TemporaryDirectory() as temporary:
            membership, stocks, output = self._fixture(Path(temporary))
            run_audit(membership, stocks, output, train_end=date(2022, 1, 14),
                      minimum_observations=3, include_peers=True)
            summary = json.loads((output / "summary.json").read_text())
            results = json.loads((output / "stock_checkpoints" / "AAA.json").read_text())
            self.assertEqual(summary["factor_candidates"], 3)
            self.assertNotIn("AAA", {row[0] for row in results})
            self.assertIn("RELIANCE", {row[0] for row in results})
            manifest = json.loads((output / "preparation-manifest.json").read_text())
            self.assertEqual(manifest["format_version"], 4)
            with self.assertRaisesRegex(ValueError, "Inputs or settings changed"):
                run_audit(membership, stocks, output, train_end=date(2022, 1, 14),
                          minimum_observations=3, resume=True)

    def test_pause_and_resume_during_factor_preparation(self):
        with TemporaryDirectory() as temporary:
            membership, stocks, output = self._fixture(Path(temporary))
            started, release = Event(), Event()
            errors = []

            def wait_during_read(path):
                if path.parent.name == "NIFTY_FACTOR":
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release the data reader")
                return load_closes(path)

            def run():
                try:
                    run_audit(membership, stocks, output, train_end=date(2022, 1, 14),
                              minimum_observations=3)
                except Exception as exc:
                    errors.append(exc)

            with patch("modeling.factor_research.load_closes", side_effect=wait_during_read):
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
                    self.assertTrue(thread.is_alive())
                finally:
                    release.set()
                    TrainingControl(output).request("continue")
                    thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
