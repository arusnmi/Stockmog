import json
import tempfile
import unittest
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from backtesting.execution_simulator import ExecutionCosts
from backtesting.rl_execution import ExecutionConfig
from modeling.control import TrainingControl
from modeling.rl_training import (
    RLTrainingConfig,
    _read_bars,
    _train_job,
    _train_rl_policies_locked,
)


class FakeEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakePPO:
    instances = []
    loads = []

    def __init__(self, policy, env, **kwargs):
        self.env = env
        self.num_timesteps = 0
        self.learn_calls = []
        self.saved = False
        self.instances.append(self)

    @classmethod
    def load(cls, path, *, env=None, device=None):
        cls.loads.append((path, env, device))
        instance = cls.__new__(cls)
        instance.env = env
        instance.num_timesteps = 10
        instance.learn_calls = []
        instance.saved = False
        return instance

    def set_env(self, env):
        self.env = env

    def learn(self, *, total_timesteps, reset_num_timesteps, progress_bar):
        self.learn_calls.append(total_timesteps)
        self.num_timesteps += total_timesteps

    def save(self, path):
        Path(path).write_bytes(b"checkpoint")
        self.saved = True


class RLTrainingTests(unittest.TestCase):
    def setUp(self):
        FakePPO.instances.clear()
        FakePPO.loads.clear()

    @staticmethod
    def _training_input(root, symbol):
        bars_path = root / f"{symbol}.csv.gz"
        scores_path = root / f"{symbol}.parquet"
        pd.DataFrame({
            "symbol": [symbol, symbol],
            "timestamp": [
                "2024-01-02T09:15:00+0530",
                "2024-01-02T09:16:00+0530",
            ],
            "open": [100, 101], "high": [102, 103],
            "low": [99, 100], "close": [101, 102], "volume": [1, 2],
        }).to_csv(bars_path, index=False, compression="gzip")
        pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}).to_parquet(
            scores_path,
        )

        def fingerprint(path):
            stat = path.stat()
            return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

        return {
            "symbol": symbol,
            "symbol_directory": symbol,
            "bars": str(bars_path),
            "scores": str(scores_path),
            "input_fingerprint": fingerprint(bars_path),
            "score_fingerprint": fingerprint(scores_path),
        }

    def test_training_uses_non_holdout_symbol_folds_and_writes_boundary_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "scores" / "15m" / "momentum"
            job_output = root / "policies" / "15m" / "momentum"
            job_dir.mkdir(parents=True)
            folds = [
                {"policy_holdout": False},
                {"policy_holdout": False},
                {"policy_holdout": True},
            ]
            inputs = [
                self._training_input(root, symbol) for symbol in ("AAA", "BBB")
            ]
            envs = []

            def make_env(**kwargs):
                env = FakeEnv()
                envs.append((kwargs["item"]["symbol"], kwargs["fold_index"], env))
                return env

            with patch("modeling.rl_training._episode_data", side_effect=make_env):
                status = _train_job(
                    job_dir=job_dir,
                    job_output=job_output,
                    intervals="15m",
                    strategy="momentum",
                    folds_payload={"folds": folds},
                    inputs=inputs,
                    config=RLTrainingConfig(
                        timesteps_per_episode=8, rollout_steps=4, batch_size=2,
                    ),
                    costs=ExecutionCosts(),
                    control=TrainingControl(root / "policies"),
                    resume=False,
                    seed_random=lambda _: None,
                    ppo_type=FakePPO,
                )
            self.assertEqual(status, "complete")
            self.assertEqual(
                [(symbol, fold) for symbol, fold, _ in envs],
                [("AAA", 0), ("AAA", 1), ("BBB", 0), ("BBB", 1)],
            )
            self.assertTrue(all(env.closed for _, _, env in envs))
            state = json.loads((job_output / "progress.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["completed_episodes"]), 4)
            self.assertEqual(state["trained_timesteps"], 32)
            self.assertTrue((job_output / "agent.zip").is_file())
            self.assertEqual(sum(bool(fold["policy_holdout"]) for fold in folds), 1)

    def test_resume_repeats_only_uncheckpointed_episode_and_loads_agent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "scores" / "1d" / "momentum"
            job_output = root / "policies" / "1d" / "momentum"
            job_dir.mkdir(parents=True)
            inputs = [self._training_input(root, "AAA")]
            folds = [
                {"policy_holdout": False},
                {"policy_holdout": False},
                {"policy_holdout": True},
            ]
            config = RLTrainingConfig(
                timesteps_per_episode=8, rollout_steps=4, batch_size=2,
            )
            stop_after_checkpoint = SimpleNamespace(
                checkpoint_boundary=lambda _: False,
            )
            with patch("modeling.rl_training._episode_data", return_value=FakeEnv()):
                status = _train_job(
                    job_dir=job_dir, job_output=job_output, intervals="1d",
                    strategy="momentum", folds_payload={"folds": folds},
                    inputs=inputs, config=config, costs=ExecutionCosts(),
                    control=stop_after_checkpoint, resume=False,
                    seed_random=lambda _: None, ppo_type=FakePPO,
                )
            self.assertEqual(status, "stopped")
            first = json.loads((job_output / "progress.json").read_text())
            self.assertEqual(first["completed_episodes"], ["0:AAA"])
            self.assertTrue((job_output / "agent.zip").is_file())

            resume_control = SimpleNamespace(checkpoint_boundary=lambda _: True)
            with patch("modeling.rl_training._episode_data", return_value=FakeEnv()):
                status = _train_job(
                    job_dir=job_dir, job_output=job_output, intervals="1d",
                    strategy="momentum", folds_payload={"folds": folds},
                    inputs=inputs, config=config, costs=ExecutionCosts(),
                    control=resume_control, resume=True,
                    seed_random=lambda _: None, ppo_type=FakePPO,
                )
            self.assertEqual(status, "complete")
            self.assertEqual(len(FakePPO.loads), 1)
            state = json.loads((job_output / "progress.json").read_text())
            self.assertEqual(state["completed_episodes"], ["0:AAA", "1:AAA"])

    def test_resume_resets_stopped_control_before_continuing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores_root = root / "scores"
            job_dir = scores_root / "15m" / "momentum"
            output_root = root / "policies"
            job_dir.mkdir(parents=True)
            output_root.mkdir()
            (scores_root / "score-run.json").write_text(
                json.dumps({"format_version": 1, "status": "complete"}),
                encoding="utf-8",
            )
            patches = (
                patch("modeling.rl_training._require_rl_dependencies",
                      return_value=(FakePPO, lambda _: None)),
                patch("modeling.rl_training._score_job",
                      return_value=(job_dir, {"folds": []}, [])),
                patch("modeling.rl_training._train_job", return_value="complete"),
            )
            with patches[0], patches[1], patches[2]:
                _train_rl_policies_locked(
                    scores_root=scores_root,
                    output_root=output_root,
                )

            control_path = output_root / "control.json"
            control_path.write_text(json.dumps({"action": "stopped"}), encoding="utf-8")
            with patches[0], patches[1], patches[2] as train_job:
                _train_rl_policies_locked(
                    scores_root=scores_root,
                    output_root=output_root,
                    resume=True,
                )
            self.assertEqual(json.loads(control_path.read_text())["action"], "continue")
            train_job.assert_not_called()

    def test_read_bars_validates_symbol_and_parses_exchange_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bars.csv.gz"
            pd.DataFrame({
                "symbol": ["M&M", "M&M"],
                "timestamp": [
                    "2024-01-02T09:15:00+0530",
                    "2024-01-02T09:16:00+0530",
                ],
                "open": [100, 101], "high": [102, 103],
                "low": [99, 100], "close": [101, 102], "volume": [1, 2],
            }).to_csv(path, index=False, compression="gzip")
            result = _read_bars(path, "M_M")
        self.assertEqual(result.index[0], pd.Timestamp("2024-01-02T03:45:00Z"))
        self.assertEqual(result.iloc[0]["close"], 101)

    def test_config_rejects_unreasonable_ppo_intervals(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            RLTrainingConfig(timesteps_per_episode=1000, rollout_steps=256)
        with self.assertRaisesRegex(ValueError, "exceed"):
            RLTrainingConfig(timesteps_per_episode=256, rollout_steps=256, batch_size=512)

    @unittest.skipUnless(
        find_spec("gymnasium") and find_spec("backtrader") and find_spec("sb3_contrib"),
        "Optional PPO dependencies are not installed",
    )
    def test_maskable_ppo_accepts_environment_action_masks_without_training(self):
        import numpy as np
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks

        from backtesting.rl_env import BacktraderTradingEnv

        index = pd.date_range("2024-01-02 09:15", periods=5, freq="min")
        bars = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [102.0] * 5, "low": [98.0] * 5,
                "close": [100.0] * 5, "volume": [1000.0] * 5,
            },
            index=index,
        )
        scores = pd.DataFrame(columns=[
            "timestamp", "direction", "loss_probability", "neutral_probability",
            "win_probability", "expected_return_r", "uncertainty",
        ])
        env = BacktraderTradingEnv(
            bars, scores, config=ExecutionConfig(style="intraday", interval="1m"),
        )
        try:
            model = MaskablePPO(
                "MlpPolicy", env, n_steps=4, batch_size=2, seed=3,
                device="cpu", verbose=0,
            )
            observation, _ = env.reset(seed=3)
            masks = get_action_masks(model.get_env())
            self.assertEqual(masks.shape, (1, env.action_space.n))
            action, _ = model.predict(
                observation, action_masks=env.action_masks(), deterministic=True,
            )
            self.assertTrue(env.action_masks()[int(action)])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
