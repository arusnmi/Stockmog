import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl

from modeling.data import DatasetSplit, TrainingDataset
from modeling.walk_forward import WalkForwardFold
from modeling.walk_forward_scores import (
    WalkForwardScoreConfig,
    _input_fingerprint,
    _verify_input,
    _write_symbol_scores,
    create_walk_forward_scores,
)


def fold(start, end, holdout=False):
    return WalkForwardFold(
        split=DatasetSplit(
            train=np.asarray([0]),
            validation=np.asarray([1]),
            test=np.asarray([2]),
            validation_start=start - timedelta(days=1),
            test_start=start,
        ),
        prediction_end=end,
        policy_holdout=holdout,
    )


class WalkForwardScoreTests(unittest.TestCase):
    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bars.csv.gz"
            path.write_bytes(b"before")
            fingerprint = _input_fingerprint(path)
            path.write_bytes(b"different length")
            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                _verify_input(path, fingerprint)

    def test_scoring_with_real_hybrid_inference_is_causal(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        from modeling.training import _build_lstm

        days = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(8)]
        frame = pl.DataFrame({
            "timestamp": days, "symbol": ["AAA"] * 8,
            "close": [100.0] * 8, "feature": [float(i) for i in range(8)],
        })
        signals = pl.DataFrame({
            "timestamp": [days[3], days[6]],
            "symbol": ["AAA"] * 2, "style": ["swing"] * 2,
            "strategy": ["momentum"] * 2,
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "fold"
            model_dir.mkdir()
            (model_dir / "metadata.json").write_text(json.dumps({
                "style": "swing", "strategy": "momentum",
                "feature_names": ["feature"], "mean": [0.0], "scale": [1.0],
                "sequence_length": 2,
            }), encoding="utf-8")
            (model_dir / "hybrid.json").write_text(json.dumps({
                "format_version": 1, "model_weights": {"lstm": 1.0},
                "temperature": 1.0,
            }), encoding="utf-8")
            torch.save({
                "feature_names": ["feature"], "style": "swing", "strategy": "momentum",
                "mean": [0.0], "scale": [1.0],
                "model_state": _build_lstm(torch, 1).state_dict(),
            }, model_dir / "lstm.pt")
            job = {
                "universe": "pooled", "style": "swing", "interval": "1d",
                "strategy": "momentum", "inputs": [], "run_dir": str(root),
            }
            folds = (fold(days[2].replace(tzinfo=timezone.utc), None, True),)

            def score_frame(input_frame, name, signal_data=signals):
                with (
                    patch("modeling.walk_forward_scores.build_features", return_value=SimpleNamespace(collect=lambda: input_frame)),
                    patch("modeling.walk_forward_scores.generate_strategy_signals", return_value=signal_data),
                ):
                    path = root / f"{name}.parquet"
                    _write_symbol_scores(
                        input_path=root / "AAA" / "1d.csv.gz", job=job, folds=folds,
                        fold_dirs=[model_dir], destination=path,
                        config=WalkForwardScoreConfig(sequence_length=2),
                    )
                    return pl.read_parquet(path)

            original = score_frame(frame, "original")
            altered = score_frame(
                frame.with_columns(
                    pl.when(pl.col("timestamp") > days[3])
                    .then(pl.col("feature") + 1000)
                    .otherwise(pl.col("feature"))
                    .alias("feature")
                ),
                "altered",
            )
            empty = score_frame(frame, "empty", signals.head(0))
        self.assertEqual(original.height, 2)
        self.assertEqual(empty.height, 0)
        self.assertIn("policy_holdout", empty.columns)
        self.assertIn("win_probability", empty.columns)
        self.assertAlmostEqual(original["win_probability"][0], altered["win_probability"][0])
        self.assertAlmostEqual(
            sum(original[name][0] for name in (
                "loss_probability", "neutral_probability", "win_probability"
            )),
            1.0,
            places=6,
        )

    def test_scoring_assigns_fold_and_holdout_without_losing_candidates(self):
        days = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
        utc = [date.replace(tzinfo=timezone.utc) for date in days]
        frame = pl.DataFrame({
            "timestamp": days, "symbol": ["M&M"] * 12,
            "close": [100.0] * 12, "feature": [1.0] * 12,
        })
        signals = pl.DataFrame({
            "timestamp": [days[i] for i in (1, 3, 5, 6, 8, 9, 10)],
            "symbol": ["M&M"] * 7,
            "style": ["swing"] * 7,
            "strategy": ["momentum"] * 7,
        })
        folds = (fold(utc[3], utc[6]), fold(utc[6], utc[9]), fold(utc[9], None, True))
        observed = []

        def fake_predict(features, candidates, *, model_dir, include_all_candidates):
            self.assertTrue(include_all_candidates)
            self.assertLessEqual(features["timestamp"][-1], candidates["timestamp"][-1])
            self.assertLessEqual(features.height, 3)
            observed.append((model_dir.name, candidates["timestamp"][0]))
            is_unscorable = candidates["timestamp"][0] == days[8]
            return candidates.with_columns(
                pl.lit(None if is_unscorable else 0.6, dtype=pl.Float64).alias("win_probability"),
                pl.lit(None if is_unscorable else False, dtype=pl.Boolean).alias("accepted"),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "M_M" / "1d.csv.gz"
            destination = root / "scores" / "00000.parquet"
            job = {
                "universe": "pooled", "style": "swing", "interval": "1d",
                "strategy": "momentum", "inputs": [str(path)], "run_dir": str(root),
            }
            with (
                patch("modeling.walk_forward_scores.build_features", return_value=SimpleNamespace(collect=lambda: frame)),
                patch("modeling.walk_forward_scores.generate_strategy_signals", return_value=signals),
                patch("modeling.walk_forward_scores.predict_hybrid_signals", side_effect=fake_predict),
            ):
                symbol, count = _write_symbol_scores(
                    input_path=path, job=job, folds=folds,
                    fold_dirs=[root / str(i) for i in range(3)],
                    destination=destination,
                    config=WalkForwardScoreConfig(sequence_length=3, score_batch_size=1),
                )
            result = pl.read_parquet(destination)
        self.assertEqual(count, 6)
        self.assertEqual(symbol, "M&M")
        self.assertEqual(result["timestamp"].to_list(), [utc[i] for i in (3, 5, 6, 8, 9, 10)])
        self.assertEqual(result["walk_forward_fold"].to_list(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(result["policy_holdout"].to_list(), [False] * 4 + [True] * 2)
        self.assertIsNone(result["win_probability"][3])
        self.assertEqual([name for name, _ in observed], ["0", "0", "1", "1", "2", "2"])

    def test_plan_orchestration_fits_each_fold_and_scores_all_symbols(self):
        timestamps = tuple(datetime(2024, 1, 1) + timedelta(days=i) for i in range(60))
        dataset = TrainingDataset(
            features=np.zeros((60, 1), dtype=np.float32),
            sequences=np.zeros((60, 1, 1), dtype=np.float32),
            labels=np.arange(60) % 3,
            net_returns_r=np.zeros(60),
            timestamps=timestamps,
            label_end_timestamps=tuple(t + timedelta(hours=1) for t in timestamps),
            feature_names=("feature",), style="swing", strategy="momentum",
            horizon=1, sequence_length=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / symbol / "1d.csv.gz" for symbol in ("AAA", "BBB")]
            for path in paths:
                path.parent.mkdir()
                path.write_bytes(b"fixture")
            job = {
                "universe": "pooled", "style": "swing", "interval": "1d",
                "strategy": "momentum", "inputs": [str(path) for path in paths],
                "run_dir": str(root / "old-models"),
            }
            output = root / "new-output"
            calls = []

            def fake_dataset(inputs, **kwargs):
                self.assertEqual(inputs, job["inputs"])
                self.assertEqual(kwargs["event_stride"], 2)
                kwargs["cache_dir"].mkdir()
                return dataset

            def fake_train(data, split, **kwargs):
                calls.append(split)
                return {}

            def fake_score(**kwargs):
                self.assertEqual(len(kwargs["folds"]), 3)
                kwargs["destination"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["destination"].write_bytes(b"fixture")
                return kwargs["input_path"].parent.name, 1

            with (
                patch("modeling.walk_forward_scores._read_plan", return_value={}),
                patch("modeling.walk_forward_scores.iter_planned_jobs", return_value=iter([job])),
                patch("modeling.walk_forward_scores.build_training_dataset", side_effect=fake_dataset),
                patch("modeling.walk_forward_scores._save_job_metadata"),
                patch("modeling.walk_forward_scores.train_xgboost", side_effect=fake_train),
                patch("modeling.walk_forward_scores.train_lstm", side_effect=fake_train),
                patch("modeling.walk_forward_scores._calibrate_job_hybrid"),
                patch("modeling.walk_forward_scores._write_symbol_scores", side_effect=fake_score),
            ):
                create_walk_forward_scores(
                    plan_path=root / "plan.json", output_root=output,
                    config=WalkForwardScoreConfig(folds=2, event_stride=2),
                )
            manifest = json.loads((output / "score-run.json").read_text())
            score_manifest = json.loads((output / "1d" / "momentum" / "scores.json").read_text())
            fold_manifest = json.loads((output / "1d" / "momentum" / "folds.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual([item["symbol"] for item in score_manifest["inputs"]], ["AAA", "BBB"])
            self.assertEqual(score_manifest["inputs"][0]["input_fingerprint"]["size"], 7)
            self.assertEqual([item["policy_holdout"] for item in fold_manifest["folds"]], [False, False, True])
            self.assertEqual(len(calls), 6)
            self.assertFalse((output / "1d" / "momentum" / "dataset-cache").exists())
            with self.assertRaises(FileExistsError):
                with (
                    patch("modeling.walk_forward_scores._read_plan", return_value={}),
                    patch("modeling.walk_forward_scores.iter_planned_jobs", return_value=iter([job])),
                ):
                    create_walk_forward_scores(plan_path=root / "plan.json", output_root=output)


if __name__ == "__main__":
    unittest.main()
