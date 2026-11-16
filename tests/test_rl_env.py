import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from backtesting.execution_simulator import ExecutionCosts
from backtesting.rl_execution import ExecutionConfig

if find_spec("gymnasium") is not None and find_spec("backtrader") is not None:
    from backtesting.rl_env import (
        ACTION_NAMES,
        BacktraderTradingEnv,
        prepare_fold_episode,
    )


def bars(index):
    return pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [102.0] * len(index),
            "low": [98.0] * len(index),
            "close": [100.0] * len(index),
            "volume": [1000.0] * len(index),
        },
        index=index,
    )


def score_row(timestamp, direction="long", *, holdout=False, fold=0):
    return {
        "timestamp": timestamp,
        "direction": direction,
        "loss_probability": 0.2,
        "neutral_probability": 0.3,
        "win_probability": 0.5,
        "expected_return_r": 1.0,
        "uncertainty": 0.25,
        "walk_forward_fold": fold,
        "policy_holdout": holdout,
    }


@unittest.skipUnless(
    find_spec("gymnasium") is not None and find_spec("backtrader") is not None,
    "Gymnasium or Backtrader is not installed",
)
class BacktraderGymEnvironmentTests(unittest.TestCase):
    def test_environment_passes_gymnasium_checker_and_can_reset(self):
        index = pd.date_range("2024-01-02 09:15", periods=5, freq="min")
        market = bars(index)
        scores = pd.DataFrame(columns=[
            "timestamp", "direction", "loss_probability", "neutral_probability",
            "win_probability", "expected_return_r", "uncertainty",
        ])
        env = BacktraderTradingEnv(
            market, scores,
            config=ExecutionConfig(style="intraday", interval="1m"),
            lookback=2,
        )
        try:
            check_env(env, skip_render_check=True)
            env.reset(seed=9)
            env.step(0)
        finally:
            env.close()

    def test_fold_preparation_filters_scores_and_bars_at_exact_utc_bounds(self):
        index = pd.date_range("2024-01-02 09:15", periods=5, freq="min")
        market = bars(index)
        scores = pd.DataFrame([
            score_row(pd.Timestamp("2024-01-02T03:45:00Z")),
            score_row(pd.Timestamp("2024-01-02T03:46:00Z"), fold=1),
            score_row(pd.Timestamp("2024-01-02T03:47:00Z"), holdout=True),
            score_row(pd.Timestamp("2024-01-02T03:48:00Z"), fold=1),
        ])
        fold = {
            "prediction_start_utc": "2024-01-02T03:46:00+00:00",
            "prediction_end_utc": "2024-01-02T03:49:00+00:00",
            "policy_holdout": False,
        }
        selected_bars, selected_scores = prepare_fold_episode(
            market, scores, fold=fold, fold_index=1, policy_holdout=False,
        )
        self.assertEqual(selected_bars.index.to_list(), index[1:4].to_list())
        self.assertEqual(selected_scores.index.to_list(), index[[1, 3]].to_list())
        self.assertEqual(selected_scores.iloc[0]["walk_forward_fold"], 1)

    def test_real_backtrader_step_reward_and_invalid_action_penalty(self):
        index = pd.date_range("2024-01-02 09:15", periods=3, freq="min")
        market = bars(index)
        market.iloc[1, market.columns.get_loc("open")] = 110.0
        market.iloc[1, market.columns.get_loc("high")] = 120.0
        market.iloc[1, market.columns.get_loc("low")] = 109.0
        market.iloc[1, market.columns.get_loc("close")] = 120.0
        market.iloc[2, market.columns.get_loc("open")] = 120.0
        market.iloc[2, market.columns.get_loc("high")] = 120.0
        market.iloc[2, market.columns.get_loc("low")] = 120.0
        market.iloc[2, market.columns.get_loc("close")] = 120.0
        scores = pd.DataFrame([score_row(pd.Timestamp(index[0], tz="Asia/Kolkata").tz_convert("UTC"))])
        config = ExecutionConfig(
            style="intraday",
            interval="1m",
            initial_cash=2000.0,
            exposure_fraction=0.8,
            costs=ExecutionCosts(commission_bps=0, other_fees_bps=0, slippage_bps=0),
        )
        env = BacktraderTradingEnv(market, scores, config=config, lookback=2)
        try:
            initial, info = env.reset(seed=17)
            self.assertEqual(initial.shape, env.observation_space.shape)
            self.assertTrue(env.observation_space.contains(initial))
            self.assertEqual(info["action_mask"].tolist(), [True, False, True, False, True, False])
            np.testing.assert_allclose(initial[10:17], [1, 1, 0.2, 0.3, 0.5, 0.2, 0.25])

            next_state, reward, terminated, truncated, info = env.step(2)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["position"], 16)
            self.assertAlmostEqual(reward, 0.08)
            self.assertEqual(info["timestamp"], index[1].to_pydatetime())

            _, penalized_reward, terminated, truncated, info = env.step(2)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["invalid_action"])
            self.assertAlmostEqual(penalized_reward, -0.001)
            self.assertTrue(env.action_masks()[1])
            _, final_reward, terminated, truncated, info = env.step(1)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["profit_loss"], "160.00")
            self.assertEqual(info["profit_loss_percent"], "8.00")
            self.assertAlmostEqual(final_reward, 0.0)
        finally:
            env.close()

    def test_short_actions_are_masked_on_swing_data(self):
        index = pd.date_range("2024-01-02", periods=3, freq="D")
        market = bars(index)
        scores = pd.DataFrame(columns=[
            "timestamp", "direction", "loss_probability", "neutral_probability",
            "win_probability", "expected_return_r", "uncertainty",
        ])
        env = BacktraderTradingEnv(
            market, scores,
            config=ExecutionConfig(style="swing", interval="1d"),
        )
        try:
            _, _ = env.reset()
            mask = env.action_masks()
            self.assertFalse(mask[4])
            self.assertTrue(mask[2])
            _, _, _, _, info = env.step(4)
            self.assertTrue(info["invalid_action"])
        finally:
            env.close()

    def test_rejects_mixed_fold_scores_and_duplicate_candidates(self):
        index = pd.date_range("2024-01-02 09:15", periods=3, freq="min")
        market = bars(index)
        scores = pd.DataFrame([
            score_row(pd.Timestamp(index[0], tz="Asia/Kolkata").tz_convert("UTC"), fold=0),
            score_row(pd.Timestamp(index[1], tz="Asia/Kolkata").tz_convert("UTC"), fold=1),
        ])
        with self.assertRaisesRegex(ValueError, "Only one scored candidate"):
            BacktraderTradingEnv(
                market,
                pd.concat([scores.iloc[[0]], scores.iloc[[0]]], ignore_index=True),
                config=ExecutionConfig(style="intraday", interval="1m"),
            )
        with self.assertRaisesRegex(ValueError, "cannot mix score folds"):
            BacktraderTradingEnv(
                market,
                scores,
                config=ExecutionConfig(style="intraday", interval="1m"),
            )
        self.assertEqual(len(ACTION_NAMES), 6)


if __name__ == "__main__":
    unittest.main()
