import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtesting.execution_simulator import ExecutionCosts
from modeling.rl_training import RLTrainingConfig
from modeling.rl_evaluation import evaluate_rl_policies


class FakePPO:
    loads = []
    predictions = []

    @classmethod
    def load(cls, path, *, device):
        cls.loads.append((path, device))
        return cls()

    def predict(self, observation, *, deterministic, action_masks):
        self.predictions.append((deterministic, action_masks.copy()))
        return 0, None


class FakeEnv:
    def __init__(self):
        self.bars = pd.DataFrame(index=range(3))
        self.closed = False
        self.steps = 0

    def reset(self):
        return np.zeros(2, dtype=np.float32), {}

    def action_masks(self):
        return np.asarray([True, False, False, False, False, False])

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            return np.zeros(2, dtype=np.float32), 0.0, False, False, {"equity": 50_000}
        return np.zeros(2, dtype=np.float32), -0.002, True, False, {
            "initial_capital": "50000.00",
            "final_equity": "49900.00",
            "profit_loss": "-100.00",
            "profit_loss_percent": "-0.20",
            "peak_equity": "50000.00",
            "open_position_shares": "0.00",
            "drawdown_triggered": False,
        }

    def close(self):
        self.closed = True


class RLEvaluationTests(unittest.TestCase):
    def setUp(self):
        FakePPO.loads.clear()
        FakePPO.predictions.clear()

    @staticmethod
    def _fingerprint(path):
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _make_completed_run(self, root):
        scores_root = root / "scores"
        job_dir = scores_root / "15m" / "momentum"
        bars_path = root / "stock-data" / "AAA" / "15m.csv.gz"
        score_path = job_dir / "scores" / "00000.parquet"
        policy_root = root / "policies"
        policy_job_dir = policy_root / "15m" / "momentum"
        for path in (job_dir, bars_path.parent, score_path.parent, policy_job_dir):
            path.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "symbol": ["AAA", "AAA", "AAA"],
            "timestamp": [
                "2024-01-03T09:15:00+0530",
                "2024-01-03T09:30:00+0530",
                "2024-01-03T09:45:00+0530",
            ],
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [100, 110, 120],
        }).to_csv(bars_path, index=False, compression="gzip")
        pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-03T03:45:00Z"], utc=True),
            "direction": ["long"],
            "loss_probability": [0.2],
            "neutral_probability": [0.3],
            "win_probability": [0.5],
            "expected_return_r": [0.1],
            "uncertainty": [0.2],
            "walk_forward_fold": [1],
            "policy_holdout": [True],
        }).to_parquet(score_path, index=False)
        input_entry = {
            "symbol": "AAA",
            "symbol_directory": "AAA",
            "input": str(bars_path.resolve()),
            "input_fingerprint": self._fingerprint(bars_path),
            "scores": "scores/00000.parquet",
            "rows": 1,
        }
        folds = [
            {
                "index": 0,
                "validation_start_utc": "2024-01-02T00:00:00+00:00",
                "prediction_start_utc": "2024-01-02T01:00:00+00:00",
                "prediction_end_utc": "2024-01-03T00:00:00+00:00",
                "policy_holdout": False,
                "training_events": 10,
                "validation_events": 2,
                "prediction_events": 3,
            },
            {
                "index": 1,
                "validation_start_utc": "2024-01-02T20:00:00+00:00",
                "prediction_start_utc": "2024-01-03T00:00:00+00:00",
                "prediction_end_utc": None,
                "policy_holdout": True,
                "training_events": 12,
                "validation_events": 2,
                "prediction_events": 2,
            },
        ]
        (job_dir / "folds.json").write_text(
            json.dumps({"format_version": 1, "folds": folds}),
            encoding="utf-8",
        )
        (job_dir / "scores.json").write_text(
            json.dumps({
                "format_version": 1,
                "timestamp_timezone": "UTC",
                "inputs": [input_entry],
            }),
            encoding="utf-8",
        )
        score_path_fingerprint = self._fingerprint(score_path)
        inputs = [{
            "symbol": "AAA",
            "symbol_directory": "AAA",
            "bars": str(bars_path.resolve()),
            "scores": str(score_path.resolve()),
            "input_fingerprint": input_entry["input_fingerprint"],
            "score_fingerprint": score_path_fingerprint,
        }]
        training_config = RLTrainingConfig()
        costs = ExecutionCosts()
        (policy_job_dir / "agent.zip").write_bytes(b"test checkpoint")
        (policy_job_dir / "progress.json").write_text(
            json.dumps({
                "format_version": 1,
                "status": "complete",
                "identity": {
                    "job": str(job_dir.resolve()),
                    "configuration": asdict(training_config),
                    "costs": asdict(costs),
                    "folds": folds,
                    "episodes": ["0:AAA"],
                    "input_files": inputs,
                },
                "completed_episodes": ["0:AAA"],
                "trained_timesteps": 2048,
            }),
            encoding="utf-8",
        )
        score_manifest_path = scores_root / "score-run.json"
        score_manifest_path.write_text(
            json.dumps({"format_version": 1, "status": "complete"}),
            encoding="utf-8",
        )
        score_stat = self._fingerprint(score_manifest_path)
        (policy_root / "rl-run.json").write_text(
            json.dumps({
                "format_version": 1,
                "status": "complete",
                "identity": {
                    "format_version": 1,
                    "scores_root": str(scores_root.resolve()),
                    "score_manifest_fingerprint": score_stat,
                    "configuration": asdict(training_config),
                    "costs": asdict(costs),
                },
                "completed_jobs": ["15m/momentum"],
            }),
            encoding="utf-8",
        )
        return scores_root, policy_root

    def test_evaluates_reserved_holdout_and_writes_signed_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores_root, policy_root = self._make_completed_run(root)
            output_root = root / "evaluation"
            envs = []

            def make_env(**kwargs):
                self.assertTrue(kwargs["policy_holdout"])
                self.assertEqual(kwargs["fold_index"], 1)
                env = FakeEnv()
                envs.append(env)
                return env

            with (
                patch(
                    "modeling.rl_evaluation._require_rl_dependencies",
                    return_value=(FakePPO, lambda _: None),
                ),
                patch("modeling.rl_evaluation._episode_data", side_effect=make_env),
            ):
                summary = evaluate_rl_policies(
                    scores_root=scores_root,
                    policies_root=policy_root,
                    output_root=output_root,
                )

            self.assertEqual(summary["overall"]["profit_loss"], "-100.00")
            self.assertEqual(summary["overall"]["profit_loss_percent"], "-0.20")
            self.assertEqual(summary["overall"]["loss_episodes"], 1)
            self.assertEqual(len(FakePPO.loads), 1)
            self.assertTrue(FakePPO.predictions)
            self.assertTrue(all(deterministic for deterministic, _ in FakePPO.predictions))
            self.assertTrue(all(env.closed for env in envs))
            self.assertTrue((output_root / "holdout-results.csv").is_file())
            self.assertTrue((output_root / "holdout-summary.json").is_file())

    def test_resume_reuses_completed_episode_result_without_reloading_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores_root, policy_root = self._make_completed_run(root)
            output_root = root / "evaluation"

            def make_env(**kwargs):
                return FakeEnv()

            with (
                patch(
                    "modeling.rl_evaluation._require_rl_dependencies",
                    return_value=(FakePPO, lambda _: None),
                ),
                patch("modeling.rl_evaluation._episode_data", side_effect=make_env),
            ):
                evaluate_rl_policies(
                    scores_root=scores_root,
                    policies_root=policy_root,
                    output_root=output_root,
                )
                summary = evaluate_rl_policies(
                    scores_root=scores_root,
                    policies_root=policy_root,
                    output_root=output_root,
                    resume=True,
                )
            self.assertEqual(len(FakePPO.loads), 1)
            self.assertEqual(summary["overall"]["episode_count"], 1)


if __name__ == "__main__":
    unittest.main()
