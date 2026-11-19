import contextlib
import io
import json
import re
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from modeling import nifty_price_evaluation, nifty_price_report, nifty_price_workflow
from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped, run_audit
from modeling.nifty_factor_selection import run as select_factors
from modeling.nifty_preparation import TIMEFRAMES, prepare, report_coverage
from modeling.nifty_price_workflow import load_run
from tests.test_nifty_price_training import fake_xgboost


def _write_candles(path, dates, prices, *, intraday=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [f"{day.date()}T{'09:15' if intraday else '00:00'}:00+0530" for day in dates]
    pd.DataFrame({
        "timestamp": timestamps, "open": prices, "high": np.asarray(prices) + 1,
        "low": np.asarray(prices) - 1, "close": prices, "volume": 100,
        "source_bars": 1, "trading_days": 1, "is_partial": False,
    }).to_csv(path, index=False, compression="gzip")


def _inputs(root):
    dates = pd.bdate_range("2022-01-03", periods=45)
    changes = 0.008 * np.sin(np.arange(len(dates)) * 0.9)
    factor = 100 * np.cumprod(1 + changes)
    stock_a = 100 * np.cumprod(1 + np.r_[0, changes[:-1]])
    stock_b = 200 * np.cumprod(1 + 0.006 * np.cos(np.arange(len(dates)) * 0.43))
    stocks = root / "stocks"
    for symbol, values in (("RELIANCE", stock_a), ("AAA", stock_a),
                           ("BBB", stock_b), ("NIFTY_FACTOR", factor)):
        _write_candles(stocks / "day" / symbol / "1d.csv.gz", dates, values)
    for symbol, values in (("AAA", stock_a), ("BBB", stock_b)):
        for timeframe in TIMEFRAMES[:6]:
            _write_candles(stocks / "minute" / symbol / f"{timeframe}.csv.gz",
                           dates, values, intraday=True)
    membership = root / "membership.csv"
    membership.write_text(
        "index_name,symbol,valid_from,valid_to,source,source_url\n"
        "Nifty 50,AAA,2022-01-03,,test,\n"
        "Nifty 50,BBB,2022-01-03,,test,\n"
        "Nifty 50,CCC,2022-01-03,,test,\n", encoding="utf-8")
    audit = root / "audit"
    run_audit(membership, stocks, audit, train_end=dates[13].date(), minimum_observations=5)
    selection = root / "selection"
    selected = select_factors(membership, audit, [], selection, train_end=dates[13].date())
    for output, timeframes, source in (
        (root / "intraday", list(TIMEFRAMES[:6]), "minute"),
        (root / "daily", ["1d"], "day"),
        (root / "multiday", ["2d", "trading_week"], "day"),
    ):
        prepare(membership, stocks, output, stock_symbols=["AAA", "BBB", "CCC"],
                factor_symbols=[], timeframes=timeframes, chunk_rows=100, candle_source=source)
        report_coverage(output)
    return dates, membership, stocks, selection, selected, {"AAA": stock_a, "BBB": stock_b}


class NiftyPriceWorkflowIntegrationTests(unittest.TestCase):
    def test_synthetic_pipeline_stops_resumes_and_preserves_holdout(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates, membership, stocks, selection, selected, prices = _inputs(root)
            statuses = {stock["symbol"]: stock["status"] for stock in selected["stocks"]}
            self.assertEqual(statuses, {"AAA": "selected", "BBB": "no_qualifying_factor",
                                        "CCC": "missing_stock_data"})
            run = root / "run"
            prepare_args = [
                "prepare", "--run-dir", str(run), "--membership", str(membership),
                "--selection", str(selection / "selection.json"),
                "--reference", str(stocks / "day" / "RELIANCE" / "1d.csv.gz"),
                "--intraday", str(root / "intraday"), "--daily", str(root / "daily"),
                "--multisession", str(root / "multiday"),
                "--start", str(dates[13].date()), "--validation-start", str(dates[25].date()),
                "--holdout-start", str(dates[34].date()),
                "--end", str(dates[-1].date() + timedelta(days=1)),
                "--symbols", "AAA", "BBB", "CCC", "--intraday-age-hours", "96",
                "--daily-age-days", "7", "--multiday-age-days", "14", "--factor-age-days", "5",
                "--sequence-length", "2", "--max-rows", "100", "--allow-no-factors", "--skip-unusable",
            ]

            def invoke(module, args):
                with patch.object(sys, "argv", ["nifty-command", *args]), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    module.main()

            def stopped(module, args):
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["nifty-command", *args]), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), \
                     self.assertRaises(SystemExit) as caught:
                    module.main()
                self.assertNotEqual(caught.exception.code, 0)
                self.assertIn("stopped", stderr.getvalue().lower())

            write = nifty_price_workflow._write_json
            loader = nifty_price_workflow.load_prepared_candles
            entered, release = Event(), Event()
            outcomes = []
            stderr = io.StringIO()

            def wait_for_pause(*args, **kwargs):
                if not entered.is_set():
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("Test did not release candle loading")
                return loader(*args, **kwargs)

            def stop_preparation(path, value):
                write(path, value)
                if path.parent.name == "examples":
                    TrainingControl(run).request("stop")

            def prepare_worker():
                try:
                    nifty_price_workflow.main()
                except BaseException as exc:
                    outcomes.append(exc)

            with patch.object(sys, "argv", ["nifty-command", *prepare_args]), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), \
                 patch.object(nifty_price_workflow, "_write_json", stop_preparation), \
                 patch.object(nifty_price_workflow, "load_prepared_candles", wait_for_pause):
                thread = Thread(target=prepare_worker)
                thread.start()
                try:
                    self.assertTrue(entered.wait(10))
                    control = TrainingControl(run)
                    control.request("pause")
                    release.set()
                    deadline = monotonic() + 10
                    while control.action() != "paused" and monotonic() < deadline:
                        sleep(0.01)
                    self.assertEqual(control.action(), "paused")
                    self.assertTrue(thread.is_alive())
                    control.request("continue")
                    thread.join(30)
                finally:
                    release.set()
                    if thread.is_alive():
                        TrainingControl(run).request("stop")
                        thread.join(30)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], SystemExit)
            self.assertEqual(outcomes[0].code, 2)
            self.assertIn("stopped", stderr.getvalue().lower())
            first = run / "examples" / "AAA.parquet"
            original = first.read_bytes()
            invoke(nifty_price_workflow, [*prepare_args, "--resume"])
            self.assertEqual(first.read_bytes(), original)
            prepared = json.loads((run / "preparation-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(prepared["skipped"], [{"symbol": "CCC", "reason": "missing_stock_data"}])
            self.assertEqual([item["selected_factors"] for item in prepared["stocks"]], [1, 0])
            self.assertTrue(all(not item["missing_timeframes"] for item in prepared["stocks"]))
            dataset, _ = load_run(run)
            factor_column = dataset.feature_names.index('factor:["local","NIFTY_FACTOR"]')
            self.assertTrue(np.isfinite(dataset.features[dataset.symbol_codes == 0, factor_column]).any())
            self.assertTrue(np.isnan(dataset.features[dataset.symbol_codes == 1, factor_column]).all())
            self.assertFalse(np.intersect1d(dataset.holdout, dataset.train).size)
            self.assertFalse(np.intersect1d(dataset.holdout, dataset.validation).size)

            train_args = ["train", "--run-dir", str(run), "--research-only", "--tree-rounds", "2",
                          "--epochs", "1", "--batch-size", "8", "--learning-rate", "0.001", "--device", "cpu"]
            eval_args = ["evaluate", "--run-dir", str(run), "--batch-size", "3", "--device", "cpu"]

            def stop_first_tree(round_number):
                if round_number == 1:
                    TrainingControl(run / "model").request("stop")

            fake, matrices, rounds = fake_xgboost(stop_first_tree)
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), \
                 patch.object(torch.optim.AdamW, "step"):
                stopped(nifty_price_workflow, train_args)
                self.assertEqual(rounds, [1])
                self.assertFalse((run / "model" / "training-summary.json").exists())
                invoke(nifty_price_workflow, [*train_args, "--resume"])
                self.assertEqual(rounds, [1, 2])
                self.assertEqual([len(matrix.values) for matrix in matrices[:2]],
                                 [len(dataset.train), len(dataset.validation)])
                write_eval = nifty_price_evaluation._write_json

                def stop_evaluation(path, value):
                    write_eval(path, value)
                    if path.name == "evaluation-state.json" and value["status"] == "running":
                        TrainingControl(run / "evaluation").request("stop")

                with patch.object(nifty_price_evaluation, "_write_json", stop_evaluation):
                    stopped(nifty_price_workflow, eval_args)
                first_eval = run / "evaluation" / "parts" / "part-000000.parquet"
                original_eval = first_eval.read_bytes()
                invoke(nifty_price_workflow, [*eval_args, "--resume"])
                self.assertEqual(first_eval.read_bytes(), original_eval)

            prediction = pd.read_csv(run / "evaluation" / "holdout-predictions.csv",
                                     float_precision="round_trip")
            self.assertEqual(len(prediction), len(dataset.holdout))
            self.assertEqual(set(prediction["symbol"]), {"AAA", "BBB"})
            source_closes = {(symbol, str(day.date())): value for symbol, values in prices.items()
                             for day, value in zip(dates, values)}
            for row in prediction.itertuples():
                self.assertAlmostEqual(row.actual_close, source_closes[(row.symbol, row.target_session)])
            np.testing.assert_array_equal(prediction["actual_close"].to_numpy(),
                                          dataset.target_closes[dataset.holdout])

            report_args = ["--evaluation", str(run / "evaluation"), "--output", str(run / "report")]
            write_report = nifty_price_report._write_json

            def stop_report(path, value):
                write_report(path, value)
                if path.parent.name == "graphs":
                    TrainingControl(run / "report").request("stop")

            with patch.object(nifty_price_report, "_write_json", stop_report):
                stopped(nifty_price_report, report_args)
            graph = next((run / "report" / "graphs").glob("*.png"))
            original_graph = graph.read_bytes()
            invoke(nifty_price_report, [*report_args, "--resume"])
            self.assertEqual(graph.read_bytes(), original_graph)
            report = json.loads((run / "report" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual({item["symbol"] for item in report["stocks"]}, {"AAA", "BBB"})
            pdf = (run / "report" / "stock-graphs.pdf").read_bytes()
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", pdf)), 2)
            TrainingControl(run).request("stop")
            with self.assertRaisesRegex(PreparationStopped, "dataset assembly"):
                load_run(run)
            TrainingControl(run).request("continue")
            with (stocks / "day" / "BBB" / "1d.csv.gz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                load_run(run)


if __name__ == "__main__":
    unittest.main()
