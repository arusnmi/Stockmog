import json
import io
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import date
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock, Thread
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling.control import TrainingControl
from modeling.nifty_candle_examples import build_candle_examples
from modeling.nifty_preparation import available_candles
from modeling.nifty_price_training import PriceTrainingConfig
from modeling.nifty_same_candle_correlations import _read_candles
from modeling.nifty_timeframe_data import (
    add_availability, attach_factor, chronological_boundaries, make_dataset,
    select_factors, strategy_features, training_history, training_observations, training_universe,
)
from modeling.nifty_timeframe_workflow import (
    _dataset, _prepare_job, _print_training_progress, _scan_config, _sequence_length, _train_job,
    _validate_training_device, evaluate, main, predict, prepare, run_training, train,
)
from modeling.nifty_universe import Membership
from modeling.factor_research import PreparationStopped


def thread_executor(**kwargs):
    return ThreadPoolExecutor(max_workers=kwargs["max_workers"])


def coefficient(pearson, spearman, count=100):
    return {"source": "local", "factor": "BBB", "return_status": "scored",
            "return_pairs": count, "return_pearson": pearson, "return_spearman": spearman}


def daily_candles(count=180):
    days = pd.bdate_range("2022-01-03", periods=count, tz="Asia/Kolkata")
    close = 100 * np.exp(np.cumsum(np.sin(np.arange(count) / 3) * .01 + .001))
    return pd.DataFrame({
        "timestamp": days, "open": close * .998, "high": close * 1.01, "low": close * .99,
        "close": close, "volume": np.arange(count) * 100 + 1000.,
        "source_bars": 1, "trading_days": 1, "is_partial": False,
    })


class TimeframeDataTests(unittest.TestCase):
    def test_universe_includes_joiners_and_leavers_but_not_pre_2022_members(self):
        members = (
            Membership("OLD", date(2014, 1, 1), date(2018, 4, 2), "test", ""),
            Membership("BOUNDARY", date(2020, 1, 1), date(2022, 1, 1), "test", ""),
            Membership("LEFT", date(2020, 1, 1), date(2023, 1, 1), "test", ""),
            Membership("JOINED", date(2025, 1, 1), None, "test", ""),
        )
        self.assertEqual(training_universe(members), ["JOINED", "LEFT"])
        self.assertTrue(training_history("LEFT", members)[0].contains(date(2026, 1, 1)))
        self.assertTrue(training_history("JOINED", members)[0].contains(date(2022, 1, 3)))
        with self.assertRaisesRegex(ValueError, "not a verified Nifty member"):
            training_history("OLD", members)

    def test_dual_thresholds_stop_at_first_success_and_floor(self):
        strong = select_factors([coefficient(.7, .7), {**coefficient(.69, .65), "factor": "CCC"}])
        self.assertEqual([row["factor"] for row in strong["selected"]], ["BBB"])
        relaxed = select_factors([coefficient(-.69, -.65)])
        self.assertEqual((relaxed["pearson_threshold"], relaxed["spearman_threshold"]), (.69, .65))
        relaxed_spearman = select_factors([coefficient(.65, -.45)])
        self.assertEqual((relaxed_spearman["pearson_threshold"],
                          relaxed_spearman["spearman_threshold"]), (.65, .45))
        floor = select_factors([coefficient(.4, -.4)])
        self.assertEqual(len(floor["selected"]), 1)
        self.assertEqual((floor["pearson_threshold"], floor["spearman_threshold"]), (.4, .4))
        for row in (coefficient(.399, .9), coefficient(.9, .399), coefficient(.9, .9, 59)):
            self.assertEqual(select_factors([row])["selected"], [])
        self.assertEqual(select_factors([]),
                         {"pearson_threshold": .4, "spearman_threshold": .4, "selected": []})

    def test_training_filter_checks_final_interval_and_actual_availability(self):
        observations = pd.DataFrame({
            "end": [date(2022, 1, 4), date(2022, 1, 5), date(2022, 1, 6)],
            "available_at": pd.to_datetime(["2022-01-04T18:30Z", "2022-01-10T18:30Z",
                                            "2022-01-06T18:30Z"]),
        })
        filtered = training_observations(observations, date(2022, 1, 6))
        self.assertEqual(filtered.index.tolist(), [0])

    def test_factor_join_never_uses_future_values_or_stale_intraday_values(self):
        features = pd.DataFrame({"available_at": pd.to_datetime([
            "2022-01-03T03:46Z", "2022-01-03T03:47Z", "2022-01-03T03:50Z"])})
        factor = pd.DataFrame({
            "available_at": pd.to_datetime(["2022-01-03T03:47Z", "2022-01-03T03:51Z"]),
            "factor_return": [1., 999.],
        })
        result = attach_factor(features, factor, "factor:test", "1m")
        self.assertTrue(np.isnan(result["factor:test"].iloc[0]))
        self.assertEqual(result["factor:test"].iloc[1], 1.)
        self.assertTrue(np.isnan(result["factor:test"].iloc[2]))
        self.assertEqual(result["factor:test:missing"].tolist(), [1., 0., 1.])

    def test_strategy_features_are_shared_and_causal(self):
        candles = available_candles(daily_candles(), "1d", daily_source=True)
        before = strategy_features(candles, "1d")
        changed = candles.copy()
        changed.loc[130:, ["close", "high", "low", "open"]] *= 3
        after = strategy_features(changed, "1d")
        pd.testing.assert_frame_equal(before.iloc[:130], after.iloc[:130])
        self.assertTrue({"strategy_box_breakout", "strategy_momentum", "strategy_trend_following",
                         "strategy_mean_reversion"}.issubset(before.columns))

    def test_dataset_purges_labels_and_does_not_bridge_missing_candles(self):
        raw = daily_candles()
        candles = available_candles(raw, "1d", daily_source=True)
        reference = raw["timestamp"].dt.date.tolist()
        members = (Membership("AAA", reference[0], None, "test", ""),)
        examples = build_candle_examples("AAA", "1d", candles, reference, members,
                                         start=reference[0], end=date(2023, 1, 1))
        validation, holdout = chronological_boundaries(examples)
        examples["feature"] = np.arange(len(examples), dtype=float)
        dataset = make_dataset(examples, ["feature"], "AAA", "1d", validation, holdout, 3, {})
        boundary = pd.Timestamp(validation, tz="Asia/Kolkata").tz_convert("UTC")
        self.assertTrue((examples.iloc[dataset.train]["label_available_at"] < boundary).all())
        self.assertFalse(np.intersect1d(dataset.train, dataset.holdout).size)
        gap = examples.drop(index=5).reset_index(drop=True)
        other = make_dataset(gap, ["feature"], "AAA", "1d", validation, holdout, 3, {})
        self.assertNotIn(5, other.train)
        self.assertNotIn(6, other.train)
        self.assertIn(7, other.train)
        self.assertEqual(_sequence_length("1h", 20), 5)
        self.assertEqual(_sequence_length("trading_week", 20), 3)


class TimeframeWorkflowTests(unittest.TestCase):
    def test_new_preparation_fingerprints_current_sources_not_old_scan_snapshot(self):
        from modeling.factor_research import _fingerprint
        from modeling.nifty_price_evaluation import _sha

        with TemporaryDirectory() as directory:
            root = Path(directory)
            scan = root / "scan"
            (scan / "stocks").mkdir(parents=True)
            external = root / "data" / "external"
            external.mkdir(parents=True)
            (external / "nse_index_membership_0786cd915974.csv").write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2022-01-03,,test,\n", encoding="utf-8")
            source = root / "prices.csv"
            source.write_text("before repair", encoding="utf-8")
            old = _fingerprint(source)
            source.write_text("after verified repair", encoding="utf-8")
            (scan / "summary.json").write_text('{"status":"complete"}', encoding="utf-8")
            (scan / "identity.json").write_text(
                json.dumps({"symbols": ["AAA"], "inputs": [old]}), encoding="utf-8")
            csv = scan / "stocks" / "AAA.csv"
            csv.write_text("historical scan output", encoding="utf-8")
            csv.with_suffix(".json").write_text(json.dumps({"sha256": _sha(csv)}), encoding="utf-8")
            with patch("modeling.nifty_timeframe_workflow.PROJECT", root):
                _, fingerprints = _scan_config(scan, 3, None, ["1d"], 60)
                self.assertIn(_fingerprint(source), fingerprints)
                self.assertNotIn(old, fingerprints)
                csv.write_text("corrupted", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Corrupt correlation output"):
                    _scan_config(scan, 3, None, ["1d"], 60)

    def fixture(self, root):
        raw = daily_candles()
        for symbol in ("AAA", "BBB"):
            folder = root / "stocks" / "day" / symbol
            folder.mkdir(parents=True)
            raw.to_csv(folder / "1d.csv.gz", index=False, compression="gzip")
        membership = root / "membership.csv"
        membership.write_text("index_name,symbol,valid_from,valid_to,source,source_url\n"
                              "Nifty 50,AAA,2022-01-03,,test,\n", encoding="utf-8")
        config = {"stocks": str(root / "stocks"), "membership": str(membership),
                  "symbols": ["AAA"], "timeframes": ["1d"], "minimum_pairs": 60,
                  "sequence_length": 3}
        reference = raw["timestamp"].dt.date.tolist()
        observations = _read_candles(root / "stocks", "BBB", "1d", reference)
        candles = available_candles(raw, "1d", daily_source=True)
        observations = add_availability(observations, candles, "1d")

        class Factors:
            def __init__(self):
                self.reference = reference

            def keys(self):
                return [("local", "AAA"), ("local", "BBB"), ("economic", "missing")]

            def load(self, key, timeframe):
                if key[0] == "economic":
                    return None, None
                return observations, observations[["available_at", "return"]].rename(
                    columns={"return": "factor_return"}).dropna()

        return config, Factors()

    def test_prepares_resumes_and_propagates_controls_without_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, factors = self.fixture(root)
            output = root / "run"
            with patch("modeling.nifty_timeframe_workflow._scan_config", return_value=(config, [])), \
                 patch("modeling.nifty_timeframe_workflow.FactorStore", return_value=factors), \
                 patch("modeling.nifty_timeframe_workflow.load_closes",
                       return_value=pd.Series(index=factors.reference, dtype=float)), \
                 patch("modeling.nifty_timeframe_workflow._code_hashes", return_value={}), \
                 patch("modeling.nifty_timeframe_workflow.ProcessPoolExecutor", side_effect=thread_executor), \
                 patch("modeling.nifty_timeframe_workflow.train_pooled_price_model") as trainer:
                summary = prepare(output, root / "scan")
                trainer.assert_not_called()
                self.assertEqual(summary["jobs"][0]["factors"], 1)
                folder = output / "jobs" / "AAA" / "1d"
                meta = json.loads((folder / "prepared.json").read_text())
                last = meta["selection"]["selected"][0]["return_last"]
                self.assertLess(last, meta["validation_start"])
                with patch("modeling.nifty_timeframe_workflow._prepare_job") as rebuild:
                    prepare(output, root / "scan", resume=True)
                    rebuild.assert_not_called()
                train(output, PriceTrainingConfig(sequence_length=3, tree_rounds=1, epochs=1))
                self.assertEqual(trainer.call_count, 1)
                self.assertEqual(trainer.call_args.kwargs["control"].run_dir, output)
                self.assertFalse(trainer.call_args.kwargs["control"].acknowledge)
                self.assertEqual(trainer.call_args.kwargs["config"].sequence_length, 3)
                with (folder / "examples.parquet").open("ab") as stream:
                    stream.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "Corrupt prepared file"):
                    _dataset(folder)

    def test_stop_during_preparation_and_no_factor_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, factors = self.fixture(root)
            control = TrainingControl(root / "run")
            control.request("stop")
            with self.assertRaises(PreparationStopped):
                _prepare_job(config, "AAA", "1d", root / "job", factors, control)
            control.request("continue")
            factors.keys = lambda: [("local", "AAA"), ("economic", "missing")]
            meta = _prepare_job(config, "AAA", "1d", root / "job", factors, control)
            self.assertEqual(meta["selection"]["selected"], [])
            self.assertIn("strategy_box_breakout", meta["feature_names"])
            self.assertFalse(any(name.startswith("factor:") for name in meta["feature_names"]))

    def test_prepares_full_history_after_leaving_index(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, factors = self.fixture(root)
            Path(config["membership"]).write_text(
                "index_name,symbol,valid_from,valid_to,source,source_url\n"
                "Nifty 50,AAA,2020-01-01,2022-02-01,test,\n", encoding="utf-8")
            control = TrainingControl(root / "run")
            meta = _prepare_job(config, "AAA", "1d", root / "job", factors, control)
            examples = pd.read_parquet(root / "job" / "examples.parquet")
            self.assertGreater(examples["session"].max(), date(2022, 2, 1))
            self.assertGreater(meta["split_counts"]["holdout"], 0)

    def test_evaluation_exports_price_and_trend_without_fitting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, factors = self.fixture(root)
            output = root / "run"
            folder = output / "jobs" / "AAA" / "1d"
            meta = _prepare_job(config, "AAA", "1d", folder, factors, TrainingControl(output))
            dataset = _dataset(folder)
            model = folder / "model"
            model.mkdir()
            (model / "training-summary.json").write_text(json.dumps({"dataset_sha256": dataset.identity}))
            summary = {"jobs": [{"symbol": "AAA", "timeframe": "1d"}]}

            class Predictor:
                def predict(self, tabular, sequences, anchors, *, feature_names):
                    return anchors * 1.01

            with patch("modeling.nifty_timeframe_workflow._run_summary", return_value=summary), \
                 patch("modeling.nifty_timeframe_workflow.load_price_predictor", return_value=Predictor()), \
                 patch("modeling.nifty_timeframe_workflow.train_pooled_price_model") as trainer:
                report = evaluate(output)
                trainer.assert_not_called()
                self.assertEqual(report["status"], "complete")
                predictions = pd.read_csv(folder / "holdout-predictions.csv")
                self.assertEqual(set(predictions.predicted_trend), {"up"})
                self.assertEqual(len(predictions), meta["split_counts"]["holdout"])
                with patch("modeling.nifty_timeframe_workflow.read_stock",
                           return_value=available_candles(daily_candles(), "1d", daily_source=True)), \
                     patch("modeling.nifty_timeframe_workflow.parse_nifty_membership",
                           return_value=(Membership("AAA", factors.reference[0], None, "test", ""),)), \
                     patch("modeling.nifty_timeframe_workflow.load_closes",
                           return_value=pd.Series(index=factors.reference, dtype=float)):
                    forecast = predict(output, "AAA", "1d")
                self.assertEqual(forecast["predicted_trend"], "up")
                self.assertAlmostEqual(forecast["forecast_close"], forecast["anchor_close"] * 1.01)
                self.assertIn("as_of", forecast)


class ParallelTimeframeTests(unittest.TestCase):
    def test_runs_two_jobs_concurrently_with_bounded_queue(self):
        jobs = [{"symbol": str(index), "timeframe": "1d"} for index in range(5)]
        barrier = Barrier(2)
        lock = Lock()
        active = 0
        peak = 0
        visited = []

        def worker(output, job, recorded, resume):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                visited.append(job["symbol"])
            if int(job["symbol"]) < 2:
                barrier.wait(timeout=10)
            sleep(.02)
            with lock:
                active -= 1
            return {**job, "elapsed_seconds": .02}

        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("modeling.nifty_timeframe_workflow._run_summary", return_value={"jobs": jobs}), \
                 patch("modeling.nifty_timeframe_workflow.ProcessPoolExecutor",
                       side_effect=thread_executor) as executor, \
                 patch("modeling.nifty_timeframe_workflow._train_job", side_effect=worker):
                result = train(output, PriceTrainingConfig(), workers=2)
            self.assertEqual(executor.call_args.kwargs["mp_context"].get_start_method(), "spawn")
            self.assertEqual(peak, 2)
            self.assertEqual(set(visited), {"0", "1", "2", "3", "4"})
            self.assertEqual(result["jobs"], 5)
            self.assertEqual(result["workers"], 2)
            self.assertEqual(len(result["job_timings"]), 5)

    def test_worker_failure_stops_other_workers_and_does_not_dispatch_more(self):
        jobs = [{"symbol": str(index), "timeframe": "1m"} for index in range(5)]
        barrier = Barrier(2)
        visited = []
        saved = Event()

        def worker(output, job, recorded, resume):
            visited.append(job["symbol"])
            barrier.wait(timeout=10)
            if job["symbol"] == "0":
                raise RuntimeError("CUDA out of memory")
            control = TrainingControl(output, poll_seconds=.01, acknowledge=False)
            deadline = monotonic() + 10
            while monotonic() < deadline:
                if not control.checkpoint_boundary(saved.set):
                    raise PreparationStopped("worker stopped")
                sleep(.01)
            raise AssertionError("Coordinator did not stop its worker")

        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("modeling.nifty_timeframe_workflow._run_summary", return_value={"jobs": jobs}), \
                 patch("modeling.nifty_timeframe_workflow.ProcessPoolExecutor", side_effect=thread_executor), \
                 patch("modeling.nifty_timeframe_workflow._train_job", side_effect=worker):
                with self.assertRaisesRegex(RuntimeError, "0 1m.*out of memory"):
                    train(output, PriceTrainingConfig(), workers=2)
            self.assertEqual(set(visited), {"0", "1"})
            self.assertTrue(saved.is_set())
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["active_jobs"], [])
            self.assertFalse((output / "training-summary.json").exists())
            with patch("modeling.nifty_timeframe_workflow._run_summary", return_value={"jobs": jobs}), \
                 patch("modeling.nifty_timeframe_workflow.ProcessPoolExecutor",
                       side_effect=thread_executor), \
                 patch("modeling.nifty_timeframe_workflow._train_job",
                       side_effect=lambda output, job, recorded, resume: {
                           **job, "elapsed_seconds": .01, "resumed": resume}) as restarted:
                result = train(output, PriceTrainingConfig(), workers=1, resume=True)
            self.assertEqual(restarted.call_count, 5)
            self.assertTrue(all(job["resumed"] for job in result["job_timings"]))
            self.assertEqual(result["workers"], 1)

    def test_acknowledged_stop_and_pause_are_observed_by_every_worker(self):
        with TemporaryDirectory() as directory:
            coordinator = TrainingControl(directory, poll_seconds=.01)
            worker = TrainingControl(directory, poll_seconds=.01, acknowledge=False)
            coordinator.request("stop")
            self.assertFalse(coordinator.checkpoint_boundary(lambda: None))
            saved = []
            self.assertFalse(worker.checkpoint_boundary(lambda: saved.append("stop")))
            self.assertEqual(saved, ["stop"])
            coordinator.request("pause")
            coordinator._write_status("paused")
            checkpoint = Event()
            finished = Event()

            def paused_worker():
                worker.checkpoint_boundary(checkpoint.set)
                finished.set()

            thread = Thread(target=paused_worker)
            thread.start()
            try:
                self.assertTrue(checkpoint.wait(timeout=5))
                self.assertFalse(finished.is_set())
                coordinator.request("continue")
                self.assertTrue(finished.wait(timeout=5))
            finally:
                coordinator.request("stop")
                thread.join(timeout=5)

    def test_windows_spawn_propagates_stop_without_loading_or_training_models(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            coordinator = TrainingControl(output)
            coordinator.request("stop")
            coordinator.checkpoint_boundary(lambda: None)
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
                futures = [executor.submit(
                    _train_job, output, {"symbol": str(index), "timeframe": "1d"},
                    asdict(PriceTrainingConfig()), False) for index in range(2)]
                for future in futures:
                    with self.assertRaises(PreparationStopped):
                        future.result(timeout=60)
            self.assertFalse((output / "jobs").exists())

    def test_cuda_default_and_workers_reach_training_command(self):
        with patch("sys.argv", ["nifty_timeframe_workflow", "train", "--research-only"]), \
             patch("modeling.nifty_timeframe_workflow.run_training",
                   return_value={"status": "test"}) as trainer:
            main()
        self.assertEqual(trainer.call_args.args[2].device, "cuda")
        self.assertEqual(trainer.call_args.kwargs["workers"], 2)

    def test_combined_command_prepares_before_training_and_resumes_both(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            calls = []
            with patch("modeling.nifty_timeframe_workflow._validate_training_device"), \
                 patch("modeling.nifty_timeframe_workflow.prepare",
                       side_effect=lambda *args, **kwargs: calls.append(("prepare", kwargs))), \
                 patch("modeling.nifty_timeframe_workflow.train",
                       side_effect=lambda *args, **kwargs: calls.append(("train", kwargs))):
                run_training(output, output / "scan", PriceTrainingConfig(), resume=True,
                             symbols=["INFY"], timeframes=["1d"])
            self.assertEqual([item[0] for item in calls], ["prepare", "train"])
            self.assertTrue(all(item[1]["resume"] for item in calls))
            self.assertFalse(calls[1][1]["reset_control"])
            self.assertEqual(calls[0][1]["symbols"], ["INFY"])

    def test_stop_at_stage_transition_prevents_training(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)

            def finish_preparing(*args, **kwargs):
                TrainingControl(output).request("stop")

            with patch("modeling.nifty_timeframe_workflow._validate_training_device"), \
                 patch("modeling.nifty_timeframe_workflow.prepare", side_effect=finish_preparing), \
                 patch("modeling.nifty_timeframe_workflow.train") as trainer:
                with self.assertRaises(PreparationStopped):
                    run_training(output, output / "scan", PriceTrainingConfig(), resume=True)
                trainer.assert_not_called()

    def test_pause_between_preparation_and_training_blocks_both(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            paused = Event()
            trained = Event()
            errors = []

            def finish_preparing(*args, **kwargs):
                TrainingControl(output).request("pause")
                paused.set()

            def run():
                try:
                    run_training(output, output / "scan", PriceTrainingConfig())
                except BaseException as exc:
                    errors.append(exc)

            with patch("modeling.nifty_timeframe_workflow._validate_training_device"), \
                 patch("modeling.nifty_timeframe_workflow.prepare", side_effect=finish_preparing), \
                 patch("modeling.nifty_timeframe_workflow.train", side_effect=lambda *a, **k: trained.set()):
                thread = Thread(target=run)
                thread.start()
                try:
                    self.assertTrue(paused.wait(timeout=5))
                    self.assertFalse(trained.is_set())
                    deadline = monotonic() + 5
                    while TrainingControl(output).action() != "paused" and monotonic() < deadline:
                        sleep(.01)
                    self.assertEqual(TrainingControl(output).action(), "paused")
                    TrainingControl(output).request("continue")
                    self.assertTrue(trained.wait(timeout=5))
                finally:
                    TrainingControl(output).request("stop")
                    thread.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertFalse(thread.is_alive())

    def test_console_progress_includes_job_and_checkpoint_without_status_command(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / "jobs" / "INFY" / "1d" / "model"
            folder.mkdir(parents=True)
            (folder / "training-state.json").write_text(json.dumps({
                "stage": "lstm", "tree_round": 100, "tree_rounds": 100,
                "epoch": 3, "epochs": 20, "batch_offset": 256}))
            text = io.StringIO()
            with redirect_stdout(text):
                _print_training_progress(output, [{"symbol": "INFY", "timeframe": "1d"}], 2, 10)
            self.assertIn("2/10 jobs complete", text.getvalue())
            self.assertIn("INFY 1d - lstm", text.getvalue())
            self.assertIn("epoch 3/20", text.getvalue())
            with patch("sys.argv", ["workflow", "status"]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)

    def test_console_progress_handles_unavailable_state_and_reports_other_jobs(self):
        busy = {"symbol": "BRITANNIA", "timeframe": "1m"}
        ready = {"symbol": "INFY", "timeframe": "1d"}
        snapshot = json.dumps({"stage": "lstm", "tree_round": 100, "tree_rounds": 100,
                               "epoch": 3, "epochs": 20, "batch_offset": 256})
        for error, message in (
            (PermissionError("sharing violation"), "progress snapshot unavailable"),
            (FileNotFoundError(), "loading dataset / initializing model"),
        ):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as directory:
                output = Path(directory)
                for job in (busy, ready):
                    folder = output / "jobs" / job["symbol"] / job["timeframe"] / "model"
                    folder.mkdir(parents=True)
                    (folder / "training-state.json").write_text(snapshot, encoding="utf-8")
                with patch.object(Path, "read_text", side_effect=[error, snapshot]), \
                     redirect_stdout(io.StringIO()) as text:
                    _print_training_progress(output, [busy, ready], 100, 543)
                self.assertIn(f"BRITANNIA 1m - {message}", text.getvalue())
                self.assertIn("INFY 1d - lstm", text.getvalue())

    def test_cuda_unavailable_and_invalid_workers_fail_explicitly(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA training requires"):
                _validate_training_device("cuda")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("xgboost.build_info", return_value={"USE_CUDA": False}):
            with self.assertRaisesRegex(RuntimeError, "XGBoost lacks CUDA"):
                _validate_training_device("cuda")
        with self.assertRaisesRegex(ValueError, "Worker count"):
            train(Path("unused"), PriceTrainingConfig(), workers=0)


if __name__ == "__main__":
    unittest.main()
