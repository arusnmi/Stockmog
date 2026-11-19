import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from modeling.control import TrainingControl
from modeling.nifty_price_evaluation import EvaluationStopped, evaluate_holdout
from modeling.nifty_price_training import PriceTrainingConfig, train_pooled_price_model
from tests.test_nifty_price_training import fake_xgboost, make_dataset


def fake_model(root, dataset):
    root.mkdir()
    (root / "price-tree.ubj").write_bytes(b"mock tree")
    (root / "price-lstm.pt").write_bytes(b"mock network")
    (root / "training-summary.json").write_text(json.dumps({
        "status": "complete", "research_only": True, "dataset_sha256": dataset.identity,
        "feature_names": list(dataset.feature_names), "reserved_holdout_examples": len(dataset.holdout),
        "provenance": json.loads(dataset.provenance_json),
    }), encoding="utf-8")


class NiftyPriceEvaluationTests(unittest.TestCase):
    def test_only_holdout_compares_actual_prices_and_baseline(self):
        dataset = make_dataset()
        calls = []

        class Predictor:
            def predict(self, tabular, sequences, anchor_close, *, feature_names):
                calls.extend(tabular[:, 0].tolist())
                return anchor_close + 2

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, output = root / "model", root / "report"
            fake_model(model, dataset)
            with patch("modeling.nifty_price_evaluation.load_price_predictor", return_value=Predictor()):
                result = evaluate_holdout(dataset, model, output, batch_size=2)
            predictions = pd.read_csv(output / "holdout-predictions.csv", float_precision="round_trip")
            holdout = dataset.holdout
            self.assertEqual(len(predictions), len(holdout))
            self.assertEqual(calls, dataset.features[holdout, 0].tolist())
            np.testing.assert_array_equal(predictions["actual_close"].to_numpy(), dataset.target_closes[holdout])
            np.testing.assert_array_equal(predictions["baseline_close"].to_numpy(), dataset.anchor_closes[holdout])
            np.testing.assert_array_equal(predictions["forecast_close"].to_numpy(), dataset.anchor_closes[holdout] + 2)
            expected_mae = np.mean(np.abs(dataset.anchor_closes[holdout] + 2 - dataset.target_closes[holdout]))
            self.assertAlmostEqual(result["metrics"]["model"]["mae"], expected_mae)
            self.assertEqual(result["metrics"]["model"]["count"], len(holdout))
            self.assertEqual(result["stocks"]["AAA"]["last_close_baseline"]["count"], len(holdout))
            self.assertEqual(predictions["target_session"].tolist(),
                             dataset.target_sessions[holdout].astype(str).tolist())

    def test_stop_resume_checkpoint_and_corruption_checks(self):
        dataset = make_dataset()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, output = root / "model", root / "report"
            fake_model(model, dataset)
            calls = []

            class Predictor:
                def __init__(self, stop=False):
                    self.stop = stop

                def predict(self, tabular, sequences, anchor_close, *, feature_names):
                    calls.append(float(anchor_close[0]))
                    if self.stop:
                        TrainingControl(output).request("stop")
                    return anchor_close + 1

            with patch("modeling.nifty_price_evaluation.load_price_predictor", return_value=Predictor(stop=True)):
                with self.assertRaises(EvaluationStopped):
                    evaluate_holdout(dataset, model, output, batch_size=1)
            self.assertEqual(len(calls), 1)
            self.assertFalse((output / "holdout-summary.json").exists())
            with patch("modeling.nifty_price_evaluation.load_price_predictor", return_value=Predictor()):
                report = evaluate_holdout(dataset, model, output, batch_size=1, resume=True)
            self.assertEqual(len(calls), len(dataset.holdout))
            self.assertEqual(report["status"], "complete")
            part = output / "parts" / "part-000000.parquet"
            with part.open("ab") as stream:
                stream.write(b"changed")
            with patch("modeling.nifty_price_evaluation.load_price_predictor", return_value=Predictor()):
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    evaluate_holdout(dataset, model, output, batch_size=1, resume=True)

    def test_model_and_dataset_mismatch_and_overlapping_output_fail(self):
        dataset = make_dataset()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            fake_model(model, dataset)
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_holdout(dataset, model, model / "report")
            dataset.anchor_closes[dataset.holdout[0]] += 1
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate_holdout(dataset, model, root / "report")

    def test_full_artifact_loading_uses_mock_updates_only(self):
        dataset = make_dataset()
        config = PriceTrainingConfig(sequence_length=2, tree_rounds=1, epochs=1, batch_size=4)
        fake, _, _ = fake_xgboost()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, output = root / "model", root / "report"
            with patch.multiple("xgboost", DMatrix=fake.DMatrix, Booster=fake.Booster,
                                train=fake.train, __version__=fake.__version__), patch.object(torch.optim.AdamW, "step"):
                train_pooled_price_model(dataset, model, config=config, research_only=True)
                report = evaluate_holdout(dataset, model, output, batch_size=2)
            self.assertEqual(report["metrics"]["model"]["count"], len(dataset.holdout))
            self.assertTrue((output / "holdout-predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
