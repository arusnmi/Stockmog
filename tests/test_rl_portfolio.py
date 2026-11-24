from __future__ import annotations

import unittest
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline.rl_colab_training import PPOConfig
from data_pipeline.rl_portfolio_training import (
    PORTFOLIO_SYMBOLS, TIMEFRAMES, SharedPortfolioEnv, prepare_portfolio, train_portfolio,
)


def portfolio_bars():
    starts = pd.date_range("2025-01-01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame({"timestamp": starts, "available_at": starts + pd.Timedelta(minutes=1)})
    for symbol, price in zip(PORTFOLIO_SYMBOLS, (100, 200, 500, 1000)):
        frame[f"{symbol}:open"] = price
        frame[f"{symbol}:close"] = price
    return frame


def action(env, choices):
    return int(np.flatnonzero(np.all(env.actions == choices, axis=1))[0])


class SharedPortfolioTests(unittest.TestCase):
    def setUp(self):
        config = PPOConfig(initial_cash=50_000, fee=0, slippage=0, lookback=2)
        self.env = SharedPortfolioEnv(portfolio_bars(), np.zeros((5, 2)), config)

    def test_equal_buys_share_one_account_and_sells_fund_new_position(self):
        self.env.step(action(self.env, [1, 1, 0, 0]))
        self.assertEqual(self.env.shares.tolist(), [250, 125, 0, 0])
        self.assertEqual(self.env.cash, 0)
        self.assertEqual(self.env.equity, 50_000)
        swap = action(self.env, [2, 0, 0, 1])
        self.assertTrue(self.env.action_masks()[swap])
        self.env.step(swap)
        self.assertEqual(self.env.shares.tolist(), [0, 125, 0, 25])
        self.assertEqual(self.env.cash, 0)
        self.assertEqual(self.env.equity, 50_000)

    def test_invalid_orders_cannot_double_spend_or_short(self):
        self.env.step(action(self.env, [1, 0, 0, 0]))
        invalid = action(self.env, [1, 0, 2, 0])
        self.assertFalse(self.env.action_masks()[invalid])
        self.env.step(invalid)
        self.assertEqual(self.env.shares.tolist(), [500, 0, 0, 0])
        self.assertEqual(self.env.cash, 0)
        self.assertEqual(self.env.rejections, 1)

    def test_fee_and_slippage_are_paid_from_shared_cash(self):
        config = PPOConfig(initial_cash=50_000, fee=0.01, slippage=0, lookback=2)
        env = SharedPortfolioEnv(portfolio_bars(), np.zeros((5, 2)), config)
        env.step(action(env, [1, 1, 0, 0]))
        self.assertEqual(env.shares.tolist(), [247, 123, 0, 0])
        self.assertAlmostEqual(env.cash, 207)
        self.assertAlmostEqual(env.equity, 49_507)

    def test_mask_requires_each_equal_budget_to_afford_a_share(self):
        config = PPOConfig(initial_cash=1000, fee=0, slippage=0)
        env = SharedPortfolioEnv(portfolio_bars(), np.zeros((5, 2)), config)
        self.assertTrue(env.action_masks()[action(env, [0, 0, 0, 1])])
        self.assertFalse(env.action_masks()[action(env, [1, 0, 0, 1])])
        self.assertTrue(env.action_masks()[action(env, [1, 1, 0, 0])])
        env.cash = 99
        self.assertFalse(env.action_masks()[action(env, [1, 0, 0, 0])])
        self.assertTrue(env.action_masks()[action(env, [0, 0, 0, 0])])

    def test_mask_uses_known_close_not_future_open_and_rejections_are_penalized(self):
        bars = portfolio_bars()
        bars.loc[1, "BSE:open"] = 110
        config = PPOConfig(initial_cash=100, fee=0, slippage=0)
        env = SharedPortfolioEnv(bars, np.zeros((5, 2)), config)
        buy = action(env, [1, 0, 0, 0])
        self.assertTrue(env.action_masks()[buy])
        _, reward, _, _, info = env.step(buy)
        self.assertEqual(info["trades"], 0)
        self.assertEqual(info["rejections"], 1)
        self.assertAlmostEqual(reward, -0.001)
        self.assertEqual(info["cash"], 100)

    def test_estimated_sale_proceeds_can_fund_a_purchase(self):
        config = PPOConfig(initial_cash=1000, fee=0, slippage=0)
        env = SharedPortfolioEnv(portfolio_bars(), np.zeros((5, 2)), config)
        env.step(action(env, [1, 0, 0, 0]))
        swap = action(env, [2, 0, 0, 1])
        self.assertTrue(env.action_masks()[swap])
        env.step(swap)
        self.assertEqual(env.shares.tolist(), [0, 0, 0, 1])
        self.assertEqual(env.cash, 0)

    def test_next_open_and_no_fill_outside_split(self):
        bars = portfolio_bars()
        bars.loc[0, "available_at"] += pd.Timedelta(seconds=1)
        bars.loc[1, "available_at"] += pd.Timedelta(seconds=1)
        bars.loc[0, "BSE:close"] = 100
        bars.loc[1, "BSE:open"] = 999
        bars.loc[2, "BSE:open"] = 200
        env = SharedPortfolioEnv(bars, np.zeros((5, 2)), PPOConfig(initial_cash=1000, fee=0, slippage=0))
        env.step(action(env, [1, 0, 0, 0]))
        self.assertEqual(env.index, 2)
        self.assertEqual(env.shares[0], 5)

        bars["available_at"] += pd.Timedelta(days=1)
        env = SharedPortfolioEnv(bars, np.zeros((5, 2)), PPOConfig())
        _, _, done, _, info = env.step(action(env, [1, 0, 0, 0]))
        self.assertTrue(done)
        self.assertEqual(info["trades"], 0)

    def test_portfolio_policy_trains_with_action_masks(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from data_pipeline.rl_news_policy import LowPriorityNewsPolicy

        config = PPOConfig(lookback=2, rollout=8, batch=8, envs=1)
        inputs = np.column_stack([np.zeros(5), np.arange(5)]).astype(np.float32)
        vec = DummyVecEnv([lambda: SharedPortfolioEnv(portfolio_bars(), inputs, config)])
        model = MaskablePPO(
            LowPriorityNewsPolicy, vec, n_steps=8, batch_size=8, n_epochs=1, seed=1,
            policy_kwargs={"news_indices": [1, 3], "net_arch": {"pi": [16], "vf": [16]}})
        model.learn(total_timesteps=16)
        self.assertEqual(model.num_timesteps, 16)
        vec.close()

    def test_timeframe_predictions_are_asof_and_scaling_uses_training_only(self):
        starts = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
        decisions = starts + pd.Timedelta(minutes=1)
        history = pd.DataFrame({"available_at": pd.to_datetime(
            ["2025-01-03T00:01Z", "2025-01-10T00:01Z"])})

        def fake_prepare(bundle, manifest, job, **kwargs):
            frame = pd.DataFrame({"timestamp": starts, "available_at": decisions,
                                  "open": 100.0, "close": 100.0})
            features = pd.DataFrame({"market": np.arange(20, dtype=float), "news_count": 0.0})
            return frame, features

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, files = [], {}
            for symbol in PORTFOLIO_SYMBOLS:
                for tf in TIMEFRAMES:
                    folder = f"jobs/{symbol}/{tf}"
                    jobs.append({"symbol": symbol, "timeframe": tf, "folder": folder})
                    if tf != "1m":
                        path = root / folder
                        path.mkdir(parents=True)
                        (path / "prepared.json").write_text("{}", encoding="utf-8")
                        for name in ("history.parquet", "prepared.json", "model/price-lstm.pt",
                                     "model/price-tree.ubj", "model/training-summary.json"):
                            files[folder + "/" + name] = "verified"
            with patch("data_pipeline.rl_portfolio_training.prepare_job_features", side_effect=fake_prepare), \
                 patch("data_pipeline.rl_portfolio_training.sha256", return_value="verified"), \
                 patch("data_pipeline.rl_portfolio_training.pd.read_parquet", return_value=history), \
                 patch("data_pipeline.rl_portfolio_training.predictor_stream",
                       return_value=np.array([0.1, 0.9], dtype=np.float32)):
                bars, values, splits, normalizer = prepare_portfolio(root, {"jobs": jobs, "files": files})
        names = normalizer["features"]
        i = names.index("BSE:5m:model_log_return")
        n = len(names)
        self.assertEqual(values[0, n + i], 0)
        self.assertEqual(values[4, n + i], 1)
        restored = values[4, i] * normalizer["scale"][i] + normalizer["mean"][i]
        self.assertAlmostEqual(restored, 0.1, places=5)
        market = names.index("BSE:market")
        self.assertAlmostEqual(normalizer["mean"][market], 5.5)
        self.assertEqual(len(bars), 20)
        self.assertEqual(len(splits["train"]), 12)

    def test_training_checkpoint_resumes_and_evaluates_one_portfolio(self):
        frame = portfolio_bars()
        values = np.zeros((5, 6), dtype=np.float32)
        normalizer = {"features": ["BSE:market", "BSE:news_count", "BSE:research_covered"],
                      "mean": [0] * 3, "scale": [1] * 3}
        splits = {name: np.arange(5) for name in ("train", "validation", "test")}
        config = PPOConfig(steps=16, lookback=2, rollout=8, batch=8, envs=1,
                           epochs=1, hidden=(16,), fee=0, slippage=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, output = root / "bundle", root / "output"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(json.dumps({
                "status": "complete", "news_complete": False,
                "files": {"research.parquet": "digest"}}), encoding="utf-8")
            with patch("data_pipeline.rl_portfolio_training.prepare_portfolio",
                       return_value=(frame, values, splits, normalizer)):
                trained = train_portfolio(bundle, output, config, allow_partial_news=True,
                                          allow_partial_research=True)
                tested = train_portfolio(bundle, output, config, allow_partial_news=True,
                                         allow_partial_research=True, evaluate_test=True)
            self.assertEqual(trained["timesteps"], 16)
            self.assertEqual(tested["split"], "test")
            self.assertEqual(tested["cash"]["equity"], 50_000)
            self.assertEqual(set(tested["ppo"]["open_shares"]), set(PORTFOLIO_SYMBOLS))
            self.assertTrue((output / "policy.zip").is_file())
            self.assertTrue((output / "test-summary.json").is_file())

    def test_portfolio_notebook_cells_compile(self):
        path = Path(__file__).resolve().parents[1] / "google_colab" / "phase2_ppo_portfolio_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_notebook_legacy_fix_is_repeatable_and_penalizes_failed_gap_buy(self):
        import data_pipeline.rl_portfolio_training as portfolio

        class LegacyPortfolioEnv(SharedPortfolioEnv):
            def step(self, selected):
                valid = bool(self.action_masks()[int(selected)])
                before = self.rejections
                observation, reward, terminated, truncated, info = super().step(selected)
                if valid and self.rejections > before:
                    reward += 0.001
                return observation, reward, terminated, truncated, info

        path = Path(__file__).resolve().parents[1] / "google_colab" / "phase2_ppo_portfolio_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "affordability-fix")
        code = compile("".join(cell["source"]), "affordability-fix", "exec")
        scope = {"BUNDLE": Path("bundle"),
                 "file_hash": lambda _: "380532d39a170fb351a41affb4c2d06499ca70ec0ac22bb71f1baa3c8299cfa4"}
        with patch.object(portfolio, "SharedPortfolioEnv", LegacyPortfolioEnv):
            exec(code, scope)
            exec(code, scope)
            bars = portfolio_bars()
            bars.loc[1, "BSE:open"] = 110
            env = LegacyPortfolioEnv(bars, np.zeros((5, 2)),
                                     PPOConfig(initial_cash=100, fee=0, slippage=0))
            self.assertFalse(env.action_masks()[action(env, [0, 0, 0, 1])])
            _, reward, _, _, info = env.step(action(env, [1, 0, 0, 0]))
            self.assertAlmostEqual(reward, -0.001)
            self.assertEqual(info["rejections"], 1)

    def test_portfolio_notebook_benchmark_uses_one_temporary_policy(self):
        path = Path(__file__).resolve().parents[1] / "google_colab" / "phase2_ppo_portfolio_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "benchmark")
        code = compile("".join(cell["source"]), "portfolio-benchmark", "exec")
        paths = []
        def fake_train(bundle, output, config, **kwargs):
            self.assertTrue(output.is_dir())
            paths.append(output)
            return {"timesteps": config.steps}
        with tempfile.TemporaryDirectory() as work:
            scope = {"WORK": Path(work), "BUNDLE": Path(work), "Path": Path, "PPOConfig": PPOConfig,
                     "manifest": {"news_complete": False, "research_complete": False},
                     "ALLOW_PARTIAL_NEWS": True, "ALLOW_PARTIAL_RESEARCH": True,
                     "NEWS_LOGIT_LIMIT": 0.05, "RESEARCH_LOGIT_LIMIT": 0.15,
                     "STEPS": 262_144, "train_portfolio": fake_train,
                     "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))}
            output = io.StringIO()
            with patch("time.perf_counter", side_effect=[0, 20]), redirect_stdout(output):
                exec(code, scope)
            self.assertIn("Projected 262,144 training steps", output.getvalue())
            self.assertEqual(len(paths), 1)
            self.assertFalse(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
