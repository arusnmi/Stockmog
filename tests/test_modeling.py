import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from backtesting.execution_simulator import ExecutionCosts
from main import build_parser
from modeling.control import TrainingControl
from modeling.data import (
    DatasetSplit,
    TrainingDataset,
    _label_candidate,
    _outcome_label,
    chronological_split,
    impute_and_normalize_sequences,
)
from modeling.training import calibrate_hybrid, classification_metrics


class ModelingTests(unittest.TestCase):
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
        self.assertAlmostEqual(sum(result["model_weights"].values()), 1.0)
        self.assertEqual(set(result["model_weights"]), {"xgboost", "lstm"})
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
