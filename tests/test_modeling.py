import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backtesting.execution_simulator import ExecutionCosts
from main import build_parser
from modeling.control import TrainingControl, _replace_with_retry
from modeling.batch_plan import (
    build_training_batch_plan,
    iter_planned_jobs,
    iter_planned_steps,
    write_training_batch_plan,
)
from modeling.data import (
    DatasetSplit,
    TrainingDataset,
    _label_candidate,
    _outcome_label,
    chronological_split,
    impute_and_normalize_sequences,
)
from modeling.batch_trainer import run_training_batch
from modeling.batch_trainer import _recommended_worker_count
from modeling.data import DatasetSplit, TrainingDataset
from modeling.training import (
    _xgboost_device,
    calibrate_hybrid,
    classification_metrics,
)


class ModelingTests(unittest.TestCase):
    def test_xgboost_device_uses_cuda_when_build_supports_it(self):
        class XGBoostCudaBuild:
            @staticmethod
            def build_info():
                return {"USE_CUDA": True}

        class XGBoostCpuBuild:
            @staticmethod
            def build_info():
                return {"USE_CUDA": False}

        self.assertEqual(_xgboost_device(XGBoostCudaBuild), "cuda")
        self.assertEqual(_xgboost_device(XGBoostCpuBuild), "cpu")

    def test_batch_worker_count_respects_cpu_and_memory_budgets(self):
        gib = 1024**3
        self.assertEqual(
            _recommended_worker_count(
                cpu_count=12,
                total_memory_bytes=16 * gib,
                available_memory_bytes=12 * gib,
            ),
            2,
        )
        self.assertEqual(
            _recommended_worker_count(
                cpu_count=12,
                total_memory_bytes=16 * gib,
                available_memory_bytes=5 * gib,
            ),
            1,
        )
        self.assertEqual(
            _recommended_worker_count(
                cpu_count=2,
                total_memory_bytes=64 * gib,
                available_memory_bytes=64 * gib,
            ),
            1,
        )

    def test_atomic_replace_retries_transient_windows_permission_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary) / "state.json.tmp"
            destination = Path(temporary) / "state.json"
            temporary_path.write_text("checkpoint", encoding="utf-8")
            original_replace = Path.replace
            attempts = 0

            def replace_with_transient_errors(path, target):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporary file lock")
                return original_replace(path, target)

            with patch.object(Path, "replace", replace_with_transient_errors):
                _replace_with_retry(temporary_path, destination)

            self.assertEqual(attempts, 3)
            self.assertEqual(destination.read_text(encoding="utf-8"), "checkpoint")

    def test_outcome_label_accounts_for_net_r_thresholds(self):
        self.assertEqual(_outcome_label(-0.2, 1.0), 0)
        self.assertEqual(_outcome_label(0.0, 1.0), 1)
        self.assertEqual(_outcome_label(0.2, 1.0), 2)

    def test_sequence_imputation_uses_training_mean_before_normalizing(self):
        result = impute_and_normalize_sequences(
            np.asarray([[[np.nan, 4.0]]], dtype=np.float32),
            mean=np.asarray([1.0, 2.0], dtype=np.float32),
            scale=np.asarray([2.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(result, [[[0.0, 1.0]]])

    def test_candidate_label_simulates_next_bar_and_target(self):
        bars = [
            {
                "timestamp": datetime(2024, 1, 2, 9, 15),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
            },
            {
                "timestamp": datetime(2024, 1, 2, 9, 20),
                "open": 100.0,
                "high": 103.0,
                "low": 99.5,
                "close": 100.0,
            },
            {
                "timestamp": datetime(2024, 1, 2, 9, 25),
                "open": 100.0,
                "high": 103.0,
                "low": 99.5,
                "close": 102.0,
            },
        ]
        result = _label_candidate(
            bars=bars,
            signal_index=0,
            signal={
                "timestamp": bars[0]["timestamp"],
                "direction": "long",
                "entry_price": 100.0,
                "stop_price": 99.0,
                "reward_risk": 2.0,
            },
            style="swing",
            horizon=5,
            costs=ExecutionCosts(
                commission_bps=0,
                other_fees_bps=0,
                slippage_bps=0,
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], bars[2]["timestamp"])
        self.assertAlmostEqual(result[2], 2.0)

    def test_chronological_split_purges_labels_crossing_boundaries(self):
        timestamps = tuple(
            datetime(2024, 1, 1) + timedelta(days=index // 2)
            for index in range(40)
        )
        dataset = TrainingDataset(
            features=np.zeros((40, 2), dtype=np.float32),
            sequences=np.zeros((40, 3, 2), dtype=np.float32),
            labels=np.asarray([index % 3 for index in range(40)]),
            net_returns_r=np.asarray([index / 10 for index in range(40)]),
            timestamps=timestamps,
            label_end_timestamps=tuple(
                timestamp + timedelta(hours=1) for timestamp in timestamps
            ),
            feature_names=("feature_a", "feature_b"),
            style="swing",
            strategy="box_breakout",
            horizon=1,
            sequence_length=3,
        )
        split = chronological_split(dataset)
        self.assertTrue(
            all(
                dataset.label_end_timestamps[index] < split.validation_start
                for index in split.train
            )
        )
        self.assertTrue(
            all(
                dataset.label_end_timestamps[index] < split.test_start
                for index in split.validation
            )
        )
        self.assertLess(max(split.train), min(split.validation))
        self.assertLess(max(split.validation), min(split.test))
        self.assertFalse(
            set(timestamps[index] for index in split.train)
            & set(timestamps[index] for index in split.validation)
        )
        self.assertFalse(
            set(timestamps[index] for index in split.validation)
            & set(timestamps[index] for index in split.test)
        )

    def test_training_pause_saves_checkpoint_and_waits_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = TrainingControl(temporary, poll_seconds=0.005)
            control.request("pause")
            saved = []
            result = []
            worker = threading.Thread(
                target=lambda: result.append(
                    control.checkpoint_boundary(lambda: saved.append(True))
                )
            )
            worker.start()
            deadline = time.monotonic() + 2
            while control.action() != "paused" and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(control.action(), "paused")
            self.assertEqual(saved, [True])
            control.request("continue")
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, [True])

    def test_training_stop_saves_and_exits_at_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = TrainingControl(temporary)
            control.request("stop")
            saved = []
            self.assertFalse(control.checkpoint_boundary(lambda: saved.append(True)))
            self.assertEqual(saved, [True])
            self.assertEqual(control.action(), "stopped")

    def test_hybrid_calibration_uses_validation_probabilities(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2, 0, 2])
        probabilities = np.full((len(labels), 3), 0.1)
        probabilities[np.arange(len(labels)), labels] = 0.8
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_hybrid(
                labels=labels,
                model_probabilities={
                    "xgboost": probabilities,
                    "lstm": probabilities,
                },
                run_dir=temporary,
                regime_thresholds={"adx_feature": "adx_14"},
            )
            saved = json.loads((Path(temporary) / "hybrid.json").read_text())
        import torch

        self.assertAlmostEqual(sum(result["model_weights"].values()), 1.0)
        self.assertEqual(set(result["model_weights"]), {"xgboost", "lstm"})
        self.assertEqual(
            result["calibration_device"],
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.assertEqual(result, saved)
        self.assertEqual(result["regime_thresholds"]["adx_feature"], "adx_14")

    def test_training_and_hybrid_commands_are_available_without_training(self):
        parser = build_parser()
        train_args = parser.parse_args(
            [
                "train-models",
                "--input",
                "prices.csv",
                "--style",
                "swing",
                "--strategy",
                "box_breakout",
            ]
        )
        control_args = parser.parse_args(
            ["training-control", "pause", "--run-dir", "run"]
        )
        hybrid_args = parser.parse_args(
            [
                "hybrid-signals",
                "--input",
                "prices.csv",
                "--style",
                "swing",
                "--strategy",
                "box_breakout",
                "--model-dir",
                "run",
            ]
        )
        self.assertEqual(train_args.models, "all")
        self.assertEqual(control_args.action, "pause")
        self.assertEqual(hybrid_args.minimum_win_probability, 0.45)

        catalog_args = parser.parse_args(
            [
                "train-models",
                "--input",
                "prices.csv",
                "--style",
                "swing",
                "--strategy",
                "catalog_001",
            ]
        )
        self.assertEqual(catalog_args.strategy, "catalog_001")

    def test_batch_plan_is_compact_stage_ordered_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stock_data = root / "stocks"
            for split, symbol, interval in (
                ("minute", "AAA", "1m"),
                ("minute", "AAA", "20m"),
                ("day", "AAA", "1d"),
                ("minute", "BBB", "20m"),
            ):
                symbol_directory = stock_data / split / symbol
                symbol_directory.mkdir(parents=True, exist_ok=True)
                (symbol_directory / f"{interval}.csv.gz").write_bytes(b"fixture")

            plan = build_training_batch_plan(
                stock_data_dir=stock_data,
                run_root=root / "runs",
            )
            jobs = list(iter_planned_jobs(plan))
            steps = list(iter_planned_steps(plan))
            output = write_training_batch_plan(
                plan,
                output_path=root / "plan.json",
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(plan["status"], "planned")
        self.assertFalse(plan["training_started"])
        self.assertEqual(plan["symbols"], ["AAA", "BBB"])
        self.assertEqual(plan["model_stages"], ["xgboost", "lstm", "hybrid_calibration"])
        self.assertEqual(plan["execution_order"].split(":")[0], "stage-major")
        self.assertEqual(plan["job_count"], len(jobs))
        self.assertEqual(plan["total_steps"], len(steps))
        self.assertEqual(
            [step["stage"] for step in steps[:len(jobs)]],
            ["xgboost"] * len(jobs),
        )
        self.assertEqual(
            [step["stage"] for step in steps[len(jobs):2 * len(jobs)]],
            ["lstm"] * len(jobs),
        )
        self.assertEqual(
            [step["stage"] for step in steps[2 * len(jobs):]],
            ["hybrid_calibration"] * len(jobs),
        )
        self.assertEqual(saved["job_count"], len(jobs))

    def test_batch_plan_refuses_to_overwrite_existing_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stocks = root / "stocks"
            (stocks / "minute" / "AAA").mkdir(parents=True)
            (stocks / "minute" / "AAA" / "20m.csv.gz").write_bytes(b"fixture")
            plan = build_training_batch_plan(
                stock_data_dir=stocks,
                run_root=root / "runs",
            )
            output = root / "plan.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_training_batch_plan(plan, output_path=output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_batch_trainer_runs_multiple_jobs_concurrently(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for symbol in ("AAA", "BBB"):
                input_path = root / "stocks" / "minute" / symbol / "20m.csv.gz"
                input_path.parent.mkdir(parents=True)
                input_path.write_bytes(b"fixture")
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "training_started": False,
                        "model_stages": [
                            "xgboost",
                            "lstm",
                            "hybrid_calibration",
                        ],
                        "stock_data_dir": str(root / "stocks"),
                        "run_root": str(root / "runs"),
                        "job_count": 2,
                        "groups": [
                            {
                                "style": "intraday",
                                "interval": "20m",
                                "strategies": ["box_breakout"],
                                "symbols": ["AAA", "BBB"],
                                "job_count": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            timestamps = tuple(
                datetime(2024, 1, 1) + timedelta(days=index)
                for index in range(60)
            )
            dataset = TrainingDataset(
                features=np.ones((60, 2), dtype=np.float32),
                sequences=np.ones((60, 3, 2), dtype=np.float32),
                labels=np.asarray([index % 3 for index in range(60)]),
                net_returns_r=np.zeros(60, dtype=np.float32),
                timestamps=timestamps,
                label_end_timestamps=timestamps,
                feature_names=("adx_14", "atr_pct"),
                style="intraday",
                strategy="box_breakout",
                horizon=12,
                sequence_length=3,
            )
            split = DatasetSplit(
                train=np.arange(0, 40),
                validation=np.arange(40, 50),
                test=np.arange(50, 60),
                validation_start=timestamps[40],
                test_start=timestamps[50],
            )
            probabilities = np.full((10, 3), 1.0 / 3.0)
            active = 0
            peak_active = 0
            active_lock = threading.Lock()

            def tracked_training(operation):
                nonlocal active, peak_active
                with active_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                try:
                    time.sleep(0.03)
                    return operation()
                finally:
                    with active_lock:
                        active -= 1

            def fake_xgboost(*_args, run_dir, **_kwargs):
                def train():
                    (Path(run_dir) / "xgboost.state.json").write_text(
                        json.dumps({"completed": True}),
                        encoding="utf-8",
                    )
                    return {
                        "validation_probabilities": probabilities,
                        "test_probabilities": probabilities,
                    }

                return tracked_training(train)

            def fake_lstm(*_args, run_dir, **_kwargs):
                def train():
                    (Path(run_dir) / "lstm.checkpoint.pt").write_bytes(b"checkpoint")
                    return {
                        "validation_probabilities": probabilities,
                        "test_probabilities": probabilities,
                    }

                return tracked_training(train)

            with (
                patch(
                    "modeling.batch_trainer._build_dataset_for_job",
                    return_value=(dataset, split),
                ),
                patch("modeling.batch_trainer._available_worker_count", return_value=2),
                patch("modeling.batch_trainer._xgboost_uses_cuda", return_value=False),
                patch("modeling.batch_trainer.train_xgboost", side_effect=fake_xgboost),
                patch("modeling.batch_trainer.train_lstm", side_effect=fake_lstm),
                patch("modeling.batch_trainer.torch_lstm_completed", return_value=True),
                patch("modeling.batch_trainer._calibrate_job_hybrid"),
            ):
                result = run_training_batch(
                    plan_path=plan_path,
                    run_root=root / "runs",
                    sequence_length=3,
                    xgboost_rounds=1,
                    lstm_epochs=1,
                )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(peak_active, 2)

    def test_batch_planning_cli_is_available_without_running_training(self):
        args = build_parser().parse_args(["plan-training-batch"])
        self.assertEqual(args.command, "plan-training-batch")
        self.assertEqual(args.output, "data\\models\\batch-plans\\all-symbols-all-intervals.json")

    def test_batch_trainer_runs_xgboost_then_lstm_then_hybrid(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for symbol in ("AAA", "BBB"):
                input_path = root / "stocks" / "minute" / symbol / "20m.csv.gz"
                input_path.parent.mkdir(parents=True)
                input_path.write_bytes(b"fixture")
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "training_started": False,
                        "model_stages": [
                            "xgboost",
                            "lstm",
                            "hybrid_calibration",
                        ],
                        "stock_data_dir": str(root / "stocks"),
                        "run_root": str(root / "runs"),
                        "job_count": 2,
                        "groups": [
                            {
                                "style": "intraday",
                                "interval": "20m",
                                "strategies": ["box_breakout"],
                                "symbols": ["AAA", "BBB"],
                                "job_count": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            timestamps = tuple(
                datetime(2024, 1, 1) + timedelta(days=index)
                for index in range(60)
            )
            dataset = TrainingDataset(
                features=np.ones((60, 2), dtype=np.float32),
                sequences=np.ones((60, 3, 2), dtype=np.float32),
                labels=np.asarray([index % 3 for index in range(60)]),
                net_returns_r=np.zeros(60, dtype=np.float32),
                timestamps=timestamps,
                label_end_timestamps=timestamps,
                feature_names=("adx_14", "atr_pct"),
                style="intraday",
                strategy="box_breakout",
                horizon=12,
                sequence_length=3,
            )
            split = DatasetSplit(
                train=np.arange(0, 40),
                validation=np.arange(40, 50),
                test=np.arange(50, 60),
                validation_start=timestamps[40],
                test_start=timestamps[50],
            )
            probabilities = np.full((10, 3), 1.0 / 3.0)
            calls = []

            def fake_xgboost(*_args, run_dir, **_kwargs):
                calls.append("xgboost")
                (Path(run_dir) / "xgboost.state.json").write_text(
                    json.dumps({"completed": True}),
                    encoding="utf-8",
                )
                return {
                    "validation_probabilities": probabilities,
                    "test_probabilities": probabilities,
                }

            def fake_lstm(*_args, run_dir, **_kwargs):
                calls.append("lstm")
                (Path(run_dir) / "lstm.checkpoint.pt").write_bytes(b"checkpoint")
                return {
                    "validation_probabilities": probabilities,
                    "test_probabilities": probabilities,
                }

            with (
                patch(
                    "modeling.batch_trainer._build_dataset_for_job",
                    return_value=(dataset, split),
                ),
                patch("modeling.batch_trainer.train_xgboost", side_effect=fake_xgboost),
                patch("modeling.batch_trainer.train_lstm", side_effect=fake_lstm),
                patch("modeling.batch_trainer.torch_lstm_completed", return_value=True),
            ):
                first = run_training_batch(
                    plan_path=plan_path,
                    run_root=root / "runs",
                    max_jobs_per_stage=1,
                    sequence_length=3,
                    xgboost_rounds=1,
                    lstm_epochs=1,
                )
                self.assertEqual(first["status"], "paused_at_job_limit")
                self.assertEqual(first["stage_index"], 0)
                self.assertEqual(first["job_index"], 1)

                second = run_training_batch(
                    plan_path=plan_path,
                    run_root=root / "runs",
                    resume=True,
                    max_jobs_per_stage=1,
                    sequence_length=3,
                    xgboost_rounds=1,
                    lstm_epochs=1,
                )
                self.assertEqual(second["status"], "paused_at_job_limit")
                self.assertEqual(second["stage_index"], 1)
                self.assertEqual(second["job_index"], 1)

                third = run_training_batch(
                    plan_path=plan_path,
                    run_root=root / "runs",
                    resume=True,
                    max_jobs_per_stage=1,
                    sequence_length=3,
                    xgboost_rounds=1,
                    lstm_epochs=1,
                )
                self.assertEqual(third["status"], "paused_at_job_limit")
                self.assertEqual(third["stage_index"], 2)
                self.assertEqual(third["job_index"], 1)

                result = run_training_batch(
                    plan_path=plan_path,
                    run_root=root / "runs",
                    resume=True,
                    max_jobs_per_stage=1,
                    sequence_length=3,
                    xgboost_rounds=1,
                    lstm_epochs=1,
                )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                calls,
                [
                    "xgboost",
                    "xgboost",
                    "lstm",
                    "lstm",
                    "xgboost",
                    "lstm",
                    "xgboost",
                    "lstm",
                ],
            )
            for symbol in ("AAA", "BBB"):
                job_dir = root / "runs" / "20m" / "box_breakout" / symbol
                self.assertTrue((job_dir / "hybrid.json").is_file())
                self.assertTrue((job_dir / "hybrid-batch-metrics.json").is_file())

    def test_batch_trainer_requires_explicit_large_matrix_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "training_started": False,
                        "model_stages": [
                            "xgboost",
                            "lstm",
                            "hybrid_calibration",
                        ],
                        "job_count": 10001,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "allow-large-batch"):
                run_training_batch(
                    plan_path=plan_path,
                    run_root=root / "runs",
                )

    def test_metrics_report_profitable_event_precision_and_recall(self):
        metrics = classification_metrics(
            np.asarray([0, 1, 2, 2]),
            np.asarray(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.6, 0.2, 0.2],
                ]
            ),
        )
        self.assertEqual(metrics["accuracy_pct"], 75.0)
        self.assertEqual(metrics["profitable_outcome_recall_pct"], 50.0)
        self.assertEqual(metrics["profitable_outcome_precision_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
