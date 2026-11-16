import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl

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
    _write_npy_from_raw,
    build_training_dataset,
    chronological_split,
    impute_and_normalize_sequences,
    training_feature_statistics,
)
from modeling.batch_trainer import run_training_batch
from modeling.batch_trainer import _recommended_worker_count
from modeling.data import DatasetSplit, TrainingDataset
from modeling.hybrid import predict_hybrid_signals
from modeling.walk_forward import walk_forward_splits
from modeling.training import (
    _xgboost_device,
    calibrate_hybrid,
    classification_metrics,
)


class ModelingTests(unittest.TestCase):
    def test_duration_format_uses_readable_units(self):
        from modeling.eta import format_duration

        self.assertEqual(format_duration(42), "42s")
        self.assertEqual(format_duration(125), "2m 05s")
        self.assertEqual(format_duration(3 * 3600 + 120), "3h 02m")
        self.assertEqual(format_duration(2 * 86400 + 3600), "2d 1h 00m")

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

    def test_missing_training_features_are_imputed_without_using_future_values(self):
        mean, scale, observed = training_feature_statistics(
            np.asarray([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
        )
        np.testing.assert_array_equal(observed, [True, False])
        np.testing.assert_allclose(mean, [2.0, 0.0])
        np.testing.assert_allclose(scale, [1.0, 1.0])
        normalized = impute_and_normalize_sequences(
            np.asarray([[[np.nan, 500.0], [3.0, np.nan]]], dtype=np.float32),
            mean=mean,
            scale=scale,
            observed=observed,
        )
        np.testing.assert_allclose(normalized, [[[0.0, 0.0], [1.0, 0.0]]])

    def test_weekly_events_allow_missing_ema_200_but_not_other_missing_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "trading_week.csv.gz"
            input_path.write_bytes(b"fixture")
            timestamps = [datetime(2024, 1, 1) + timedelta(days=7 * i) for i in range(8)]
            frame = pl.DataFrame({
                "timestamp": timestamps,
                "open": [100.0] * 8,
                "high": [102.0] * 8,
                "low": [98.0] * 8,
                "close": [100.5] * 8,
                "ema_200": [None] * 8,
                "other": [1.0, 1.0, 1.0, None, 1.0, 1.0, 1.0, 1.0],
            }).with_columns(pl.col("ema_200").cast(pl.Float64))
            signals = pl.DataFrame({
                "timestamp": [timestamps[2], timestamps[3], timestamps[4]],
                "direction": ["long"] * 3,
                "entry_price": [100.0] * 3,
                "stop_price": [99.0] * 3,
                "reward_risk": [1.0] * 3,
            })
            with (
                patch("modeling.data.build_features", return_value=SimpleNamespace(collect=lambda: frame)),
                patch("modeling.data.generate_strategy_signals", return_value=signals),
            ):
                memory = build_training_dataset(
                    [input_path], style="swing", strategy="box_breakout",
                    horizon=1, sequence_length=2,
                )
                cached = build_training_dataset(
                    [input_path], style="swing", strategy="box_breakout",
                    horizon=1, sequence_length=2, cache_dir=root / "cache",
                )
            self.assertEqual(len(memory.labels), 2)
            self.assertEqual(len(cached.labels), 2)
            self.assertTrue(np.isnan(memory.features[:, 0]).all())
            np.testing.assert_allclose(cached.features, memory.features, equal_nan=True)
            del cached, memory

    def test_weekly_inference_scores_missing_ema_with_training_mask(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        timestamps = [datetime(2024, 1, 1) + timedelta(days=7 * i) for i in range(3)]
        frame = pl.DataFrame({
            "timestamp": timestamps,
            "ema_200": [None, None, 1000.0],
            "other": [1.0, 2.0, None],
        }).with_columns(pl.col("ema_200").cast(pl.Float64))
        signals = pl.DataFrame({
            "timestamp": timestamps[1:],
            "style": ["swing", "swing"],
            "strategy": ["momentum", "momentum"],
        })
        inputs = []

        class FakeNetwork:
            def to(self, device):
                return self

            def load_state_dict(self, state):
                pass

            def eval(self):
                pass

            def __call__(self, tensor):
                inputs.append(tensor.cpu().numpy())
                return torch.tensor(
                    [[0.0, 0.0, 2.0, 1.0, -1.0]], device=tensor.device
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(json.dumps({
                "style": "swing", "strategy": "momentum",
                "feature_names": ["ema_200", "other"],
                "mean": [0.0, 1.0], "scale": [1.0, 1.0],
                "sequence_length": 2, "optional_missing_features": ["ema_200"],
                "observed_features": [False, True],
            }), encoding="utf-8")
            (root / "hybrid.json").write_text(json.dumps({
                "format_version": 1, "model_weights": {"lstm": 1.0},
                "temperature": 1.0,
            }), encoding="utf-8")
            checkpoint = {
                "feature_names": ["ema_200", "other"], "style": "swing",
                "strategy": "momentum", "mean": [0.0, 1.0],
                "scale": [1.0, 1.0], "observed_features": [False, True],
                "model_state": {},
            }
            with (
                patch("torch.load", return_value=checkpoint),
                patch("modeling.hybrid._build_lstm", return_value=FakeNetwork()),
            ):
                result = predict_hybrid_signals(frame, signals, model_dir=root)
                rejected = predict_hybrid_signals(
                    frame, signals, model_dir=root, minimum_win_probability=1.0,
                )
                all_candidates = predict_hybrid_signals(
                    frame, signals, model_dir=root, minimum_win_probability=1.0,
                    include_all_candidates=True,
                )
                unscorable = predict_hybrid_signals(
                    frame, signals.tail(1), model_dir=root,
                    include_all_candidates=True,
                )
                zero_weight_lstm = None
                if find_spec("xgboost") is not None:
                    (root / "hybrid.json").write_text(json.dumps({
                        "format_version": 1,
                        "model_weights": {"xgboost": 1.0, "lstm": 0.0},
                        "temperature": 1.0,
                    }), encoding="utf-8")
                    booster = SimpleNamespace(
                        load_model=lambda _: None,
                        set_param=lambda _: None,
                        predict=lambda _: np.asarray([[0.1, 0.2, 0.7]]),
                    )
                    with (
                        patch("xgboost.Booster", return_value=booster),
                        patch("xgboost.DMatrix", return_value=object()),
                    ):
                        zero_weight_lstm = predict_hybrid_signals(
                            frame, signals.head(1), model_dir=root,
                            include_all_candidates=True,
                        )
            (root / "hybrid.json").write_text(json.dumps({
                "format_version": 1, "model_weights": {"lstm": 1.0},
                "temperature": 1.0,
            }), encoding="utf-8")

            class UncertainLossNetwork(FakeNetwork):
                def __call__(self, tensor):
                    return torch.tensor(
                        [[0.0, 0.0, 0.0, -1.0, -1.0]], device=tensor.device
                    )

            with (
                patch("torch.load", return_value=checkpoint),
                patch("modeling.hybrid._build_lstm", return_value=UncertainLossNetwork()),
            ):
                experimental = predict_hybrid_signals(
                    frame, signals, model_dir=root, include_all_candidates=True,
                )
                legacy = predict_hybrid_signals(
                    frame, signals, model_dir=root, include_all_candidates=True,
                    minimum_win_probability=0.45,
                    require_return_and_low_uncertainty=True,
                )
                stricter = predict_hybrid_signals(
                    frame, signals, model_dir=root, include_all_candidates=True,
                    minimum_win_probability=0.40,
                )
        self.assertEqual(result.height, 1)
        self.assertEqual(result["timestamp"][0], timestamps[1])
        np.testing.assert_array_equal(inputs[0][0, :, 0], [0.0, 0.0])
        self.assertEqual(rejected.height, 0)
        self.assertEqual(all_candidates.height, 2)
        self.assertEqual(all_candidates["accepted"].to_list(), [False, None])
        self.assertIsNotNone(all_candidates["win_probability"][0])
        self.assertIsNone(all_candidates["win_probability"][1])
        self.assertEqual(unscorable["accepted"].to_list(), [None])
        self.assertIsNone(unscorable["win_probability"][0])
        if zero_weight_lstm is not None:
            self.assertAlmostEqual(zero_weight_lstm["win_probability"][0], 0.7)
            self.assertAlmostEqual(zero_weight_lstm["expected_return_r"][0], 1.0)
        self.assertEqual(experimental["accepted"].to_list(), [True, None])
        self.assertLess(experimental["expected_return_r"][0], 0)
        self.assertGreater(experimental["uncertainty"][0], 0.95)
        self.assertEqual(legacy["accepted"].to_list(), [False, None])
        self.assertEqual(stricter["accepted"].to_list(), [False, None])

    def test_event_stride_is_causal_and_cached(self):
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
        frame = pl.DataFrame({
            "timestamp": timestamps, "open": [100.0] * 12,
            "high": [102.0] * 12, "low": [98.0] * 12,
            "close": [100.0] * 12, "feature": [1.0] * 12,
        })
        signals = pl.DataFrame({"timestamp": timestamps[1:10]})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "1d.csv.gz"
            path.write_bytes(b"fixture")
            with (
                patch("modeling.data.build_features", return_value=SimpleNamespace(collect=lambda: frame)),
                patch("modeling.data.generate_strategy_signals", return_value=signals),
                patch("modeling.data._label_candidate", side_effect=lambda **kw: (
                    0, kw["signal"]["timestamp"] + timedelta(hours=1), 0.0
                )),
            ):
                memory = build_training_dataset(
                    [path], style="swing", strategy="momentum", horizon=1,
                    sequence_length=1, event_stride=3,
                )
                with patch("modeling.data.generate_strategy_signals", return_value=signals.head(6)):
                    prefix = build_training_dataset(
                        [path], style="swing", strategy="momentum", horizon=1,
                        sequence_length=1, event_stride=3,
                    )
                cached = build_training_dataset(
                    [path], style="swing", strategy="momentum", horizon=1,
                    sequence_length=1, event_stride=3,
                    cache_dir=Path(temporary) / "cache",
                )
                self.assertEqual(memory.timestamps, tuple(timestamps[1:10:3]))
                self.assertEqual(prefix.timestamps, memory.timestamps[:2])
                self.assertEqual(len(cached.labels), 3)
                np.testing.assert_array_equal(cached.features, memory.features)
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    build_training_dataset(
                        [path], style="swing", strategy="momentum", horizon=1,
                        event_stride=3, max_events_per_file=2,
                    )
                del cached

    def test_pooled_dataset_cache_uses_disk_backed_arrays_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "stocks" / "minute" / "AAA" / "15m.csv.gz"
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(b"fixture")
            timestamps = tuple(
                datetime(2024, 1, 1) + timedelta(days=index)
                for index in range(12)
            )
            frame = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100.0] * 12,
                    "high": [102.0] * 12,
                    "low": [98.0] * 12,
                    "close": [100.5] * 12,
                    "feature_x": [float(index * 10_000) for index in range(12)],
                }
            )
            signals = pl.DataFrame(
                {
                    "timestamp": [timestamps[index] for index in (3, 5, 7)],
                    "direction": ["long", "short", "long"],
                    "entry_price": [100.0] * 3,
                    "stop_price": [99.0] * 3,
                    "reward_risk": [1.0] * 3,
                }
            )
            with (
                patch(
                    "modeling.data.build_features",
                    return_value=SimpleNamespace(collect=lambda: frame),
                ),
                patch("modeling.data.generate_strategy_signals", return_value=signals),
            ):
                dataset = build_training_dataset(
                    [input_path],
                    style="swing",
                    strategy="box_breakout",
                    horizon=1,
                    sequence_length=2,
                    cache_dir=root / "cache",
                )
            self.assertIsInstance(dataset.features, np.memmap)
            self.assertIsInstance(dataset.sequences, np.memmap)
            self.assertEqual(dataset.features.shape, (3, 1))
            self.assertEqual(dataset.sequences.shape, (3, 2, 1))
            self.assertEqual(dataset.sequences.dtype, np.float32)
            self.assertEqual(dataset.timestamps.dtype, np.int64)
            np.testing.assert_array_equal(
                dataset.sequences,
                [
                    [[20_000.0], [30_000.0]],
                    [[40_000.0], [50_000.0]],
                    [[60_000.0], [70_000.0]],
                ],
            )
            with patch("modeling.data.build_features", side_effect=AssertionError):
                cached = build_training_dataset(
                    [input_path],
                    style="swing",
                    strategy="box_breakout",
                    horizon=1,
                    sequence_length=2,
                    cache_dir=root / "cache",
                )
            self.assertIsInstance(cached.sequences, np.memmap)
            np.testing.assert_array_equal(cached.labels, dataset.labels)
            del cached, dataset

    def test_raw_cache_finalization_streams_into_valid_npy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "sequences.bin"
            output_path = root / "sequences.npy"
            source = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
            source.tofile(raw_path)

            _write_npy_from_raw(
                raw_path,
                output_path,
                dtype=np.dtype(np.float32),
                shape=source.shape,
                array_name="test sequences",
            )

            np.testing.assert_array_equal(np.load(output_path), source)
            self.assertEqual(raw_path.stat().st_size, source.nbytes)

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

    def test_walk_forward_purges_labels_and_reserves_policy_holdout(self):
        timestamps = tuple(
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=index // 2)
            for index in range(120)
        )
        ends = tuple(timestamp + timedelta(days=2) for timestamp in timestamps)
        dataset = TrainingDataset(
            features=np.zeros((120, 1), dtype=np.float32),
            sequences=np.zeros((120, 1, 1), dtype=np.float32),
            labels=np.arange(120) % 3,
            net_returns_r=np.zeros(120),
            timestamps=timestamps,
            label_end_timestamps=ends,
            feature_names=("feature",),
            style="swing",
            strategy="momentum",
            horizon=2,
            sequence_length=1,
        )
        folds = walk_forward_splits(dataset, folds=3)
        self.assertEqual(len(folds), 4)
        self.assertEqual([fold.policy_holdout for fold in folds], [False] * 3 + [True])
        seen = set()
        for fold in folds:
            split = fold.split
            self.assertTrue(all(ends[index] < split.validation_start for index in split.train))
            self.assertTrue(all(ends[index] < split.test_start for index in split.validation))
            self.assertTrue(all(timestamps[index] >= split.test_start for index in split.test))
            if fold.prediction_end is not None:
                self.assertTrue(
                    all(timestamps[index] < fold.prediction_end for index in split.test)
                )
            indices = set(split.test.tolist())
            self.assertFalse(seen & indices)
            seen.update(indices)
        self.assertEqual(
            folds[-1].split.test_start,
            datetime(2024, 2, 21, tzinfo=timezone.utc),
        )

        cached = TrainingDataset(
            **{
                **dataset.__dict__,
                "timestamps": np.asarray(
                    [int(timestamp.timestamp()) * 1_000_000_000 for timestamp in timestamps],
                    dtype=np.int64,
                ),
                "label_end_timestamps": np.asarray(
                    [int(end.timestamp()) * 1_000_000_000 for end in ends],
                    dtype=np.int64,
                ),
            }
        )
        for memory_fold, cached_fold in zip(folds, walk_forward_splits(cached, folds=3)):
            np.testing.assert_array_equal(memory_fold.split.train, cached_fold.split.train)
            np.testing.assert_array_equal(memory_fold.split.validation, cached_fold.split.validation)
            np.testing.assert_array_equal(memory_fold.split.test, cached_fold.split.test)

    def test_walk_forward_rejects_empty_purged_periods(self):
        timestamps = tuple(datetime(2024, 1, 1) + timedelta(days=i) for i in range(40))
        dataset = TrainingDataset(
            features=np.zeros((40, 1), dtype=np.float32),
            sequences=np.zeros((40, 1, 1), dtype=np.float32),
            labels=np.arange(40) % 3,
            net_returns_r=np.zeros(40),
            timestamps=timestamps,
            label_end_timestamps=tuple(t + timedelta(days=100) for t in timestamps),
            feature_names=("feature",),
            style="swing",
            strategy="momentum",
            horizon=100,
            sequence_length=1,
        )
        with self.assertRaisesRegex(ValueError, "empty after purging"):
            walk_forward_splits(dataset)

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
        self.assertEqual(hybrid_args.minimum_win_probability, 0.22)
        self.assertFalse(hybrid_args.legacy_hybrid_filter)
        backtest_args = parser.parse_args([
            "backtest", "--input", "prices.csv", "--style", "swing",
            "--strategy", "box_breakout", "--legacy-hybrid-filter",
            "--minimum-win-probability", "0.45",
        ])
        self.assertTrue(backtest_args.legacy_hybrid_filter)
        self.assertEqual(backtest_args.minimum_win_probability, 0.45)

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

    def test_batch_plan_pools_symbols_across_all_timeframes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stock_data = root / "stocks"
            for split, symbol, interval in (
                ("minute", "AAA", "1m"),
                ("minute", "BBB", "1m"),
                ("minute", "AAA", "5m"),
                ("minute", "AAA", "15m"),
                ("minute", "AAA", "20m"),
                ("minute", "AAA", "30m"),
                ("minute", "AAA", "1h"),
                ("day", "AAA", "1d"),
                ("day", "AAA", "2d"),
                ("day", "AAA", "trading_week"),
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
            selected = build_training_batch_plan(
                stock_data_dir=stock_data,
                run_root=root / "retry",
                intervals=["20m", "trading_week"],
            )

        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["format_version"], 3)
        self.assertFalse(plan["training_started"])
        self.assertNotIn("symbols", plan)
        self.assertEqual(plan["input_file_count"], 10)
        self.assertEqual(plan["strategy_count"], 4)
        self.assertEqual(plan["interval_count"], 9)
        self.assertEqual(plan["job_count"], 36)
        self.assertTrue(all(job["universe"] == "pooled" for job in jobs))
        self.assertTrue(
            all("symbol" not in job and "symbols" not in job for job in jobs)
        )
        self.assertEqual(len(jobs[0]["inputs"]), 2)
        self.assertEqual(
            {group["interval"] for group in plan["groups"]},
            {"1m", "5m", "15m", "20m", "30m", "1h", "1d", "2d", "trading_week"},
        )
        self.assertTrue(
            all(
                "input_files" in group and "symbols" not in group
                for group in plan["groups"]
            )
        )
        self.assertIn("feature-engineered in memory", plan["execution_warning"])
        self.assertEqual(len({job["run_dir"] for job in jobs}), len(jobs))
        self.assertEqual(plan["model_stages"], ["xgboost", "lstm", "hybrid_calibration"])
        self.assertEqual(plan["execution_order"].split(":")[0], "job-major")
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
        self.assertEqual(selected["job_count"], 8)
        self.assertEqual(
            {group["interval"] for group in selected["groups"]},
            {"20m", "trading_week"},
        )

    def test_batch_plan_refuses_to_overwrite_existing_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stocks = root / "stocks"
            (stocks / "minute" / "AAA").mkdir(parents=True)
            (stocks / "minute" / "AAA" / "15m.csv.gz").write_bytes(b"fixture")
            plan = build_training_batch_plan(
                stock_data_dir=stocks,
                run_root=root / "runs",
                symbols=["AAA"],
            )
            output = root / "plan.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_training_batch_plan(plan, output_path=output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_batch_trainer_processes_pooled_jobs_sequentially(self):
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
                        "format_version": 3,
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
                                "strategies": ["box_breakout", "momentum"],
                                "input_files": [
                                    str(root / "stocks" / "minute" / symbol / "20m.csv.gz")
                                    for symbol in ("AAA", "BBB")
                                ],
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
            self.assertEqual(peak_active, 1)

    def test_batch_planning_cli_is_available_without_running_training(self):
        args = build_parser().parse_args(["plan-training-batch"])
        self.assertEqual(args.command, "plan-training-batch")
        self.assertEqual(
            args.output,
            "data\\models\\batch-plans\\all-timeframes-pooled.json",
        )
        self.assertIsNone(args.symbols)

    def test_batch_reports_skipped_jobs_as_completed_with_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps({
                "format_version": 3, "training_started": False,
                "model_stages": ["xgboost", "lstm", "hybrid_calibration"],
                "run_root": str(root / "run"), "job_count": 1,
                "groups": [{
                    "style": "intraday", "interval": "1h",
                    "strategies": ["momentum"], "input_files": ["fixture.csv"],
                    "job_count": 1,
                }],
            }), encoding="utf-8")
            with patch("modeling.batch_trainer._process_batch_job", return_value="skipped") as process:
                state = run_training_batch(plan_path=plan_path, run_root=root / "run")
                self.assertEqual(state["status"], "completed_with_failures")
                self.assertEqual(state["failed_jobs"], 1)
                self.assertEqual(process.call_count, 3)
                resumed = run_training_batch(
                    plan_path=plan_path, run_root=root / "run", resume=True
                )
                self.assertEqual(resumed["status"], "completed_with_failures")
                self.assertEqual(process.call_count, 3)

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
                        "format_version": 3,
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
                                "strategies": ["box_breakout", "momentum"],
                                "input_files": [
                                    str(root / "stocks" / "minute" / symbol / "20m.csv.gz")
                                    for symbol in ("AAA", "BBB")
                                ],
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
                self.assertEqual(second["status"], "completed")
                self.assertEqual(second["stage_index"], 0)
                self.assertEqual(second["job_index"], 2)
                result = second

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                calls,
                [
                    "xgboost",
                    "lstm",
                    "xgboost",
                    "lstm",
                    "xgboost",
                    "lstm",
                    "xgboost",
                    "lstm",
                ],
            )
            for strategy in ("box_breakout", "momentum"):
                job_dir = root / "runs" / "20m" / strategy / "pooled"
                self.assertTrue((job_dir / "hybrid.json").is_file())
                self.assertTrue((job_dir / "hybrid-batch-metrics.json").is_file())

    def test_batch_trainer_requires_explicit_large_matrix_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "format_version": 3,
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
