import json
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from modeling.control import TrainingControl
from modeling.nifty_preparation import TIMEFRAMES
from modeling.nifty_price_training import (
    PriceTrainingConfig, _price_lock, load_price_predictor, prepare_price_dataset, train_pooled_price_model,
)
from modeling.nifty_universe import Membership
from modeling.training import TrainingStopped


def fixture():
    days = [date(2022, 1, 1) + timedelta(days=index) for index in range(14)]
    frame = pd.DataFrame({
        "session": days[:-1], "target_session": days[1:],
        "available_at": pd.to_datetime([str(day) + "T18:30Z" for day in days[:-1]], utc=True),
        "label_available_at": pd.to_datetime([str(day) + "T18:30Z" for day in days[1:]], utc=True),
        "anchor_close": np.arange(100, 113, dtype=float),
        "target_log_return": np.linspace(-0.02, 0.02, 13),
    })
    for tf in TIMEFRAMES:
        for suffix in ("close_ratio", "body", "range", "partial"):
            frame[f"{tf}_{suffix}"] = np.linspace(0, 0.1, 13)
    frame["target_close"] = frame["anchor_close"] * np.exp(frame["target_log_return"])
    selection = {"status": "research_only", "threshold_absolute": 0.7,
                 "train_end_exclusive": "2022-01-01",
                 "stocks": [{"symbol": "AAA", "status": "no_qualifying_factor", "selected": []}]}
    member = (Membership("AAA", days[0], None, "test", ""),)
    return days, frame, selection, member


def make_dataset():
    days, frame, selection, membership = fixture()
    return prepare_price_dataset({"AAA": frame}, selection, membership, days,
                                 validation_start=days[7], holdout_start=days[10],
                                 sequence_length=2, allow_no_factors=True)


def fake_xgboost(on_train=None):
    matrices = []
    calls = []

    class Matrix:
        def __init__(self, values, label=None):
            self.values = np.asarray(values)
            self.label = label
            matrices.append(self)

    class Booster:
        def __init__(self):
            self.rounds = 0

        def load_model(self, raw):
            self.rounds = int(bytes(raw if isinstance(raw, bytearray) else Path(raw).read_bytes()).decode())

        def set_param(self, value):
            pass

        def save_raw(self, raw_format):
            return str(self.rounds).encode()

        def predict(self, matrix):
            return np.full(len(matrix.values), self.rounds * 0.001, dtype=np.float32)

    def train(params, matrix, *, num_boost_round, xgb_model, verbose_eval):
        booster = xgb_model if xgb_model is not None else Booster()
        booster.rounds += 1
        calls.append(booster.rounds)
        if on_train:
            on_train(booster.rounds)
        return booster

    module = SimpleNamespace(__version__="test", DMatrix=Matrix, Booster=Booster, train=train)
    return module, matrices, calls


class PriceTrainingTests(unittest.TestCase):
    def test_parent_control_stops_before_fitting_and_keeps_checkpoint(self):
        dataset = make_dataset()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = TrainingControl(root / "parent")
            parent.request("stop")
            fake, _, calls = fake_xgboost()
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__):
                with self.assertRaises(TrainingStopped):
                    train_pooled_price_model(dataset, root / "model", config=PriceTrainingConfig(
                        sequence_length=2, tree_rounds=1, epochs=1), research_only=True, control=parent)
            self.assertEqual(calls, [])
            self.assertTrue((root / "model" / "checkpoint.pt").is_file())
            self.assertEqual(parent.action(), "stopped")

    def test_purges_boundaries_and_never_uses_missing_sessions_in_sequences(self):
        days, frame, selection, membership = fixture()
        frame.loc[4, "label_available_at"] = pd.Timestamp("2022-01-12T18:30Z")
        dataset = prepare_price_dataset({"AAA": frame}, selection, membership, days,
                                        validation_start=days[7], holdout_start=days[10],
                                        sequence_length=2, allow_no_factors=True)
        self.assertNotIn(4, dataset.train)
        self.assertNotIn(6, dataset.train)
        self.assertNotIn(9, dataset.validation)
        self.assertTrue(all(index >= 10 for index in dataset.holdout))
        missing = frame.drop(index=3).reset_index(drop=True)
        other = prepare_price_dataset({"AAA": missing}, selection, membership, days,
                                      validation_start=days[7], holdout_start=days[10],
                                      sequence_length=2, allow_no_factors=True)
        self.assertNotIn(3, other.train)
        with self.assertRaisesRegex(ValueError, "stock-only"):
            prepare_price_dataset({"AAA": frame}, selection, membership, days,
                                  validation_start=days[7], holdout_start=days[10], sequence_length=2)
        selection["train_end_exclusive"] = "2022-01-05"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            prepare_price_dataset({"AAA": frame}, selection, membership, days,
                                  validation_start=days[7], holdout_start=days[10],
                                  sequence_length=2, allow_no_factors=True)

    def test_actual_close_must_match_the_return_label(self):
        days, frame, selection, membership = fixture()
        frame.loc[2, "target_close"] *= 1.1
        with self.assertRaisesRegex(ValueError, "target closes"):
            prepare_price_dataset({"AAA": frame}, selection, membership, days,
                                  validation_start=days[7], holdout_start=days[10],
                                  sequence_length=2, allow_no_factors=True)

    def test_stops_and_resumes_tree_without_using_holdout_or_fitting(self):
        dataset = make_dataset()
        config = PriceTrainingConfig(sequence_length=2, tree_rounds=2, epochs=1, batch_size=3)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"

            def stop_after_first(round_number):
                if round_number == 1:
                    TrainingControl(output).request("stop")

            fake, matrices, calls = fake_xgboost(stop_after_first)
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), patch.object(torch.optim.AdamW, "step"):
                with self.assertRaises(TrainingStopped):
                    train_pooled_price_model(dataset, output, config=config, research_only=True)
            self.assertEqual(calls, [1])
            self.assertTrue((output / "checkpoint.pt").is_file())
            self.assertFalse((output / "training-summary.json").exists())
            fake, matrices, calls = fake_xgboost()
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), patch.object(torch.optim.AdamW, "step"):
                result = train_pooled_price_model(dataset, output, config=config, research_only=True, resume=True)
                self.assertEqual(calls, [2])
                self.assertEqual([len(matrix.values) for matrix in matrices[:2]],
                                 [len(dataset.train), len(dataset.validation)])
                predictor = load_price_predictor(output)
                row = dataset.validation[:1]
                prices = predictor.predict(dataset.features[row], dataset.features[dataset.sequences[row]],
                                           np.array([100.0]), feature_names=list(dataset.feature_names))
                self.assertEqual(prices.shape, (1,))
                self.assertGreater(prices[0], 0)
                with self.assertRaisesRegex(ValueError, "feature order"):
                    predictor.predict(dataset.features[row], dataset.features[dataset.sequences[row]],
                                      np.array([100.0]), feature_names=["wrong"])
            self.assertEqual(result["reserved_holdout_examples"], len(dataset.holdout))
            with (output / "price-tree.ubj").open("ab") as stream:
                stream.write(b"bad")
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__):
                with self.assertRaisesRegex(ValueError, "Corrupt"):
                    load_price_predictor(output)
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__):
                with self.assertRaisesRegex(ValueError, "changed"):
                    train_pooled_price_model(dataset, output, config=PriceTrainingConfig(
                        sequence_length=2, tree_rounds=3, epochs=1), research_only=True, resume=True)

    def test_stops_mid_lstm_and_resumes_next_batch(self):
        dataset = make_dataset()
        config = PriceTrainingConfig(sequence_length=2, tree_rounds=1, epochs=1,
                                     batch_size=2, checkpoint_batches=10)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            fake, _, _ = fake_xgboost()
            calls = []

            def stop_once(optimizer, *args, **kwargs):
                calls.append(1)
                TrainingControl(output).request("stop")

            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), patch.object(torch.optim.AdamW, "step", stop_once):
                with self.assertRaises(TrainingStopped):
                    train_pooled_price_model(dataset, output, config=config, research_only=True)
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["stage"], "lstm")
            self.assertEqual(checkpoint["batch_offset"], 2)
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), patch.object(torch.optim.AdamW, "step") as step:
                train_pooled_price_model(dataset, output, config=config, research_only=True, resume=True)
            remaining = (len(dataset.train) + 1) // 2 - 1
            self.assertEqual(step.call_count, remaining)
            self.assertEqual(len(calls), 1)

    def test_research_acknowledgement_is_required(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "research_only"):
                train_pooled_price_model(make_dataset(), Path(temporary) / "run",
                                         config=PriceTrainingConfig(sequence_length=2),
                                         research_only=False)

    def test_preparation_can_stop_and_live_lock_is_not_removed(self):
        days, frame, selection, membership = fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = TrainingControl(root)
            control.request("stop")
            with self.assertRaises(TrainingStopped):
                prepare_price_dataset({"AAA": frame}, selection, membership, days,
                                      validation_start=days[7], holdout_start=days[10],
                                      sequence_length=2, allow_no_factors=True, control=control)
            self.assertEqual(control.action(), "stopped")
            with _price_lock(root):
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    with _price_lock(root):
                        pass
                self.assertTrue((root / "price-training.lock").is_file())
            self.assertFalse((root / "price-training.lock").exists())

    def test_pooling_factors_preserves_identity_and_ignores_holdout_statistics(self):
        days, frame, selection, membership = fixture()
        other = frame.copy()
        other["factor_0_return"] = np.linspace(0.1, 0.2, len(other))
        selection["stocks"].append({"symbol": "BBB", "status": "selected", "selected": [
            {"source": "local", "factor": "F", "correlation": 0.8}]})
        membership += (Membership("BBB", days[0], None, "test", ""),)
        dataset = prepare_price_dataset({"AAA": frame, "BBB": other}, selection, membership, days,
                                        validation_start=days[7], holdout_start=days[10],
                                        sequence_length=2, allow_no_factors=True)
        factor_column = dataset.feature_names.index('factor:["local","F"]')
        self.assertTrue(np.isnan(dataset.features[:len(frame), factor_column]).all())
        self.assertTrue(np.isfinite(dataset.features[len(frame):, factor_column]).all())
        with TemporaryDirectory() as temporary:
            dataset.features[0, 0] = 123
            with self.assertRaisesRegex(ValueError, "changed"):
                train_pooled_price_model(dataset, Path(temporary) / "run",
                                         config=PriceTrainingConfig(sequence_length=2), research_only=True)

    def test_pause_and_continue_during_tree_stage(self):
        dataset = make_dataset()
        config = PriceTrainingConfig(sequence_length=2, tree_rounds=1, epochs=1, batch_size=4)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            requested = Event()
            errors = []

            def request_pause(round_number):
                TrainingControl(output).request("pause")
                requested.set()

            fake, _, _ = fake_xgboost(request_pause)

            def worker():
                try:
                    train_pooled_price_model(dataset, output, config=config, research_only=True)
                except Exception as exc:
                    errors.append(exc)

            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), patch.object(torch.optim.AdamW, "step"):
                thread = Thread(target=worker)
                thread.start()
                try:
                    self.assertTrue(requested.wait(10))
                    control = TrainingControl(output)
                    deadline = monotonic() + 10
                    while control.action() != "paused" and monotonic() < deadline:
                        sleep(0.01)
                    self.assertEqual(control.action(), "paused")
                    self.assertTrue((output / "checkpoint.pt").is_file())
                    self.assertTrue(thread.is_alive())
                finally:
                    TrainingControl(output).request("continue")
                    thread.join(30)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue((output / "training-summary.json").is_file())

    def test_real_xgboost_artifact_format_without_fitting(self):
        import xgboost as xgb
        dataset = make_dataset()
        config = PriceTrainingConfig(sequence_length=2, tree_rounds=1, epochs=1, batch_size=4)

        def unfitted(params, matrix, **kwargs):
            return xgb.Booster(params, [matrix])

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch("xgboost.train", side_effect=unfitted), patch.object(torch.optim.AdamW, "step"):
                train_pooled_price_model(dataset, output, config=config, research_only=True)
                predictor = load_price_predictor(output)
                row = dataset.validation[:1]
                prices = predictor.predict(dataset.features[row], dataset.features[dataset.sequences[row]],
                                           np.array([100.0]), feature_names=list(dataset.feature_names))
            self.assertTrue(np.isfinite(prices).all())


if __name__ == "__main__":
    unittest.main()
