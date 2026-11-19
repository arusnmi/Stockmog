import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from modeling.factor_research import PreparationStopped, run_audit
from modeling.nifty_factor_selection import _sha, load_audit, run, stock_rows


class NiftyFactorSelectionTests(unittest.TestCase):
    def test_exact_signed_threshold_deduplication_and_resume_identity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = root / "membership.csv"
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-01,,test,\n"
                "Nifty 50,BBB,2022-01-01,,test,\n"
                "Nifty 50,CCC,2022-01-01,,test,\n", encoding="utf-8")
            audits = []
            for name, source, candidates, rows in [
                ("local", "local", {"POS", "NEG", "BELOW"}, {
                    "AAA": [["POS", 60, 0.7], ["NEG", 60, -0.7], ["BELOW", 60, 0.6999999999]],
                    "BBB": [["POS", 60, 0.2]], "CCC": []}),
                ("global1", "same-source", {"SPY"}, {
                    "AAA": [["SPY", 60, 0.8]], "BBB": [], "CCC": []}),
                ("global2", "same-source", {"SPY"}, {
                    "AAA": [["SPY", 60, 0.8]], "BBB": [], "CCC": []}),
            ]:
                directory = root / name
                (directory / "stock_checkpoints").mkdir(parents=True)
                manifest = directory / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                summary = {"historical_symbols": 3, "missing_daily_files": [],
                           "no_scorable_factor": [symbol for symbol, values in rows.items() if not values],
                           "no_qualifying_factor": [symbol for symbol, values in rows.items()
                                                    if values and all(abs(row[2]) < 0.7 for row in values)]}
                (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                for symbol, values in rows.items():
                    (directory / "stock_checkpoints" / f"{symbol}.json").write_text(
                        json.dumps(values), encoding="utf-8")
                audits.append({"directory": directory, "source": source, "candidates": candidates,
                               "summary": summary, "minimum": 60, "manifest": manifest,
                               "settings": None if name == "local" else {"max_age_days": 5}})
            output = root / "selection"
            with patch("modeling.nifty_factor_selection.load_audit", side_effect=audits):
                with patch("modeling.nifty_factor_selection.TrainingControl.checkpoint_boundary", return_value=False):
                    with self.assertRaises(PreparationStopped):
                        run(membership, root / "local", [root / "global1", root / "global2"],
                            output, train_end=date(2025, 1, 1))
            self.assertFalse((output / "selection.json").exists())
            with patch("modeling.nifty_factor_selection.load_audit", side_effect=audits):
                result = run(membership, root / "local", [root / "global1", root / "global2"],
                             output, train_end=date(2025, 1, 1), resume=True)
            aaa, bbb, ccc = result["stocks"]
            self.assertEqual([x["factor"] for x in aaa["selected"]], ["SPY", "NEG", "POS"])
            self.assertEqual(aaa["scored_factors"], 4)
            self.assertEqual(bbb["status"], "no_qualifying_factor")
            self.assertEqual(ccc["status"], "insufficient_observations")
            (root / "local" / "stock_checkpoints" / "BBB.json").write_text("[]", encoding="utf-8")
            with patch("modeling.nifty_factor_selection.load_audit", side_effect=audits):
                with self.assertRaisesRegex(ValueError, "Selection inputs changed"):
                    run(membership, root / "local", [root / "global1", root / "global2"],
                        output, train_end=date(2025, 1, 1), resume=True)

    def test_rejects_mismatched_checkpoint_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stock_checkpoints").mkdir()
            (root / "stock_checkpoints" / "AAA.json").write_text('[["FACTOR",60,0.8]]', encoding="utf-8")
            audit = {"directory": root, "candidates": {"FACTOR"}, "minimum": 60,
                     "summary": {"missing_daily_files": [], "no_scorable_factor": [],
                                 "no_qualifying_factor": ["AAA"]}}
            with self.assertRaisesRegex(ValueError, "summary does not match"):
                stock_rows(audit, "AAA")

    def test_rejects_audit_when_a_source_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = root / "membership.csv"
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-01,,test,\n", encoding="utf-8")
            stocks = root / "stocks"
            dates = pd.date_range("2022-01-03", periods=6)
            for symbol in ("AAA", "RELIANCE", "NIFTY_FACTOR"):
                directory = stocks / "day" / symbol
                directory.mkdir(parents=True)
                pd.DataFrame({"timestamp": [f"{d.date()}T00:00:00+0530" for d in dates],
                              "close": [100, 103, 101, 105, 99, 102]}).to_csv(
                                  directory / "1d.csv.gz", index=False, compression="gzip")
            output = root / "audit"
            run_audit(membership, stocks, output, train_end=date(2022, 1, 10), minimum_observations=3)
            load_audit(output, "local", _sha(membership), date(2022, 1, 10))
            factor = stocks / "day" / "NIFTY_FACTOR" / "1d.csv.gz"
            with factor.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Audit input changed"):
                load_audit(output, "local", _sha(membership), date(2022, 1, 10))


if __name__ == "__main__":
    unittest.main()
