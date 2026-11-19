import json
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped, run_audit
from modeling.nifty_factor_selection import run as select_factors
from modeling.nifty_price_workflow import _factor_inputs, _validate_selection, load_run, prepare_run
from modeling.nifty_universe import Membership
from tests.test_nifty_price_training import fixture


class PriceWorkflowTests(unittest.TestCase):
    def test_training_command_requires_explicit_research_opt_in(self):
        with TemporaryDirectory() as temporary:
            command = [sys.executable, "-m", "modeling.nifty_price_workflow", "train",
                       "--run-dir", str(Path(temporary) / "run"), "--tree-rounds", "1",
                       "--epochs", "1", "--batch-size", "2", "--learning-rate", "0.001", "--device", "cpu"]
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("--research-only", process.stderr)
            self.assertFalse((Path(temporary) / "run").exists())

    def test_selection_reconstruction_rejects_modified_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = root / "membership.csv"
            membership.write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-01,,test,\n", encoding="utf-8")
            stocks = root / "stocks"
            for symbol in ("AAA", "RELIANCE", "NIFTY_FACTOR"):
                folder = stocks / "day" / symbol
                folder.mkdir(parents=True)
                pd.DataFrame({"timestamp": [f"2022-01-0{day}T00:00:00+0530" for day in range(3, 9)],
                              "close": [100, 103, 101, 105, 99, 102]}).to_csv(
                                  folder / "1d.csv.gz", index=False, compression="gzip")
            audit, output = root / "audit", root / "selection"
            run_audit(membership, stocks, audit, train_end=date(2022, 1, 10), minimum_observations=3)
            result = select_factors(membership, audit, [], output, train_end=date(2022, 1, 10))
            self.assertEqual(_validate_selection(output / "selection.json", membership), result)
            result["stocks"][0]["status"] = "no_qualifying_factor"
            result["stocks"][0]["selected"] = []
            (output / "selection.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_selection(output / "selection.json", membership)

    def test_resolves_local_and_global_return_methods_and_checks_global_age(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            global_audit = root / "global"
            factor_root = root / "collection"
            local.mkdir()
            global_audit.mkdir()
            factor_root.mkdir()
            local_factor = root / "NIFTY_FACTOR" / "1d.csv.gz"
            local_factor.parent.mkdir()
            pd.DataFrame({"timestamp": [f"2022-01-0{day}T00:00:00+0530" for day in range(3, 6)],
                          "close": [100, 110, 121]}).to_csv(local_factor, index=False, compression="gzip")
            (local / "preparation-manifest.json").write_text(json.dumps({
                "factor_inputs": [{"path": str(local_factor)}]}), encoding="utf-8")
            (factor_root / "summary.json").write_text(json.dumps({
                "factors": [{"symbol": "SPY"}, {"symbol": "ETF"}]}), encoding="utf-8")
            (global_audit / "identity.json").write_text(json.dumps({
                "inputs": [{}, {}, {"path": str(factor_root / "summary.json")}]}), encoding="utf-8")
            selection = {"audits": [{"directory": str(local), "source": "local"},
                                     {"directory": str(global_audit), "source": "global",
                                      "settings": {"max_age_days": 5, "max_close_age_minutes": None}}]}
            days = [date(2022, 1, day) for day in range(3, 6)]
            us = pd.DataFrame({
                "timestamp": pd.to_datetime(["2022-01-03T14:30Z", "2022-01-04T14:30Z",
                                               "2022-01-05T14:30Z"], utc=True),
                "available_at": pd.to_datetime(["2022-01-04T05:00Z", "2022-01-05T05:00Z",
                                                  "2022-01-06T05:00Z"], utc=True),
                "close": [100, 110, 121], "is_partial": [False, False, False]})
            chosen = {("local", "NIFTY_FACTOR"), ("global", "ETF")}
            with patch("modeling.nifty_price_workflow._read_global_part", return_value=us):
                result = _factor_inputs(selection, chosen, days, 5, TrainingControl(root / "control"))
            self.assertEqual(len(result[("local", "NIFTY_FACTOR")]), 2)
            self.assertAlmostEqual(result[("global", "ETF")]["factor_return"].iloc[0], 0.1)
            self.assertEqual(result[("local", "NIFTY_FACTOR")]["factor_available_at"].iloc[0],
                             pd.Timestamp("2022-01-04T18:30Z"))
            with self.assertRaisesRegex(ValueError, "maximum age"):
                _factor_inputs(selection, chosen, days, 6, TrainingControl(root / "control"))

    def test_preparation_stop_resume_and_reconstruct_without_training(self):
        days, frame, selection, members = fixture()
        selection["stocks"].append({"symbol": "BBB", "status": "no_qualifying_factor", "selected": []})
        membership = members + (Membership("BBB", days[0], None, "test", ""),)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit"
            audit.mkdir()
            reference = root / "reference.csv"
            reference.write_text("test", encoding="utf-8")
            (audit / "preparation-manifest.json").write_text(json.dumps({
                "reference_calendar": {"path": str(reference)}}), encoding="utf-8")
            selection["audits"] = [{"directory": str(audit), "source": "local"}]
            config = {"membership": str(root / "membership"), "selection": str(root / "selection"),
                      "reference": str(reference), "intraday": str(root), "daily": str(root),
                      "multisession": str(root), "symbols": ["AAA", "BBB"],
                      "start": str(days[0]), "validation_start": str(days[7]),
                      "holdout_start": str(days[10]), "end": "2022-01-15",
                      "intraday_age_hours": 72, "daily_age_days": 7, "multiday_age_days": 14,
                      "factor_age_days": 5, "sequence_length": 2, "max_rows": 100,
                      "allow_no_factors": True, "allow_missing_timeframes": False, "skip_unusable": False}
            Path(config["membership"]).write_text("fixture", encoding="utf-8")
            output = root / "run"
            built = []

            def build(symbol, *args, **kwargs):
                built.append(symbol)
                if symbol == "AAA" and len(built) == 1:
                    TrainingControl(output).request("stop")
                return frame.copy(), [], []

            with patch("modeling.nifty_price_workflow._validate_selection", return_value=selection), \
                 patch("modeling.nifty_price_workflow._input_identity",
                       side_effect=lambda value: {"version": 1, "config": value}), \
                 patch("modeling.nifty_price_workflow.parse_nifty_membership", return_value=membership), \
                 patch("modeling.nifty_price_workflow._daily_dates", return_value=days), \
                 patch("modeling.nifty_price_workflow._factor_inputs", return_value={}), \
                 patch("modeling.nifty_price_workflow.load_prepared_candles", return_value=pd.DataFrame()), \
                 patch("modeling.nifty_price_workflow.build_examples", side_effect=build):
                with self.assertRaises(PreparationStopped):
                    prepare_run(config, output)
                self.assertEqual(built, ["AAA"])
                self.assertTrue((output / "examples" / "AAA.parquet").exists())
                summary = prepare_run(config, output, resume=True)
                self.assertEqual(built, ["AAA", "BBB"])
                self.assertGreater(summary["split_counts"]["holdout"], 0)
                dataset, _ = load_run(output)
                self.assertEqual(dataset.symbols, ("AAA", "BBB"))
                with (output / "examples" / "AAA.parquet").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    load_run(output)

    def test_requires_matching_reference_and_explicit_stock_only_opt_in(self):
        days, _, selection, members = fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit"
            audit.mkdir()
            first, second = root / "first", root / "second"
            first.write_text("different", encoding="utf-8")
            second.write_text("reference", encoding="utf-8")
            (audit / "preparation-manifest.json").write_text(json.dumps({
                "reference_calendar": {"path": str(second)}}), encoding="utf-8")
            selection["audits"] = [{"directory": str(audit), "source": "local"}]
            config = {"membership": str(first), "selection": str(first), "reference": str(first),
                      "symbols": ["AAA"], "start": str(days[0]), "validation_start": str(days[7]),
                      "holdout_start": str(days[10]), "end": "2022-01-15",
                      "allow_no_factors": False, "skip_unusable": False}
            with patch("modeling.nifty_price_workflow._validate_selection", return_value=selection), \
                 patch("modeling.nifty_price_workflow._input_identity", return_value={"config": config}), \
                 patch("modeling.nifty_price_workflow.parse_nifty_membership", return_value=members), \
                 patch("modeling.nifty_price_workflow._daily_dates", return_value=days):
                with self.assertRaisesRegex(ValueError, "Reference calendar differs"):
                    prepare_run(config, root / "run")
                config["reference"] = str(second)
                with self.assertRaisesRegex(ValueError, "allow-no-factors"):
                    prepare_run(config, root / "run")


if __name__ == "__main__":
    unittest.main()
