from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from data_pipeline.rl_colab_training import PPOConfig, PriceActionEnv, evaluate, temporal_splits, train_jobs
from data_pipeline.rl_news_stream import EVENT_COLUMNS, news_features, normalize_events, parse_gkg, save_json


def bars(opens=None, closes=None):
    starts = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame({"timestamp": starts, "available_at": starts + pd.Timedelta(days=1),
                         "open": opens if opens is not None else [100] * 5,
                         "close": closes if closes is not None else [100] * 5})


class ColabEnvironmentTests(unittest.TestCase):
    def test_next_open_costs_hold_and_sell(self):
        config = PPOConfig(initial_cash=1000, exposure=1, fee=0.01, slippage=0)
        env = PriceActionEnv(bars([100, 100, 110, 120, 120], [100, 105, 110, 120, 120]),
                             np.zeros((5, 2)), config)
        _, _, _, _, first = env.step(1)
        self.assertEqual(env.shares, 9)
        self.assertAlmostEqual(first["equity"], 1036)
        self.assertEqual(env.action_masks().tolist(), [True, False, True])
        env.step(0)
        _, _, _, _, final = env.step(2)
        self.assertEqual(env.shares, 0)
        self.assertAlmostEqual(final["equity"], 1160.2)

    def test_wait_preserves_cash_and_no_shorting(self):
        env = PriceActionEnv(bars(), np.zeros((5, 2)), PPOConfig())
        _, reward, _, _, info = env.step(2)
        self.assertLess(reward, 0)
        self.assertEqual(info["rejections"], 1)
        self.assertEqual(env.shares, 0)
        self.assertEqual(evaluate(env)["profit"], 0)

    def test_delayed_candle_availability_skips_early_open(self):
        frame = bars([100, 999, 110, 110, 110])
        frame.loc[0, "available_at"] += pd.Timedelta(hours=1)
        env = PriceActionEnv(frame, np.zeros((5, 1)), PPOConfig(exposure=1, fee=0, slippage=0))
        env.step(1)
        self.assertEqual(env.index, 2)
        self.assertEqual(env.shares, 454)

    def test_no_fill_outside_split(self):
        frame = bars()
        frame["available_at"] += pd.Timedelta(days=10)
        env = PriceActionEnv(frame, np.zeros((5, 1)), PPOConfig())
        _, _, done, _, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(info["trades"], 0)

    def test_reward_charges_incremental_drawdown(self):
        frame = bars(closes=[100, 90, 90, 90, 90])
        config = PPOConfig(exposure=1, fee=0, slippage=0, drawdown_penalty=0.1, downside_penalty=0)
        env = PriceActionEnv(frame, np.zeros((5, 1)), config)
        _, reward, _, _, info = env.step(1)
        self.assertAlmostEqual(info["equity"], 45000)
        self.assertAlmostEqual(reward, np.log(0.9) - 0.01)

    def test_splits_keep_sessions_separate(self):
        frame = pd.DataFrame({"available_at": pd.date_range(
            "2025-01-01", periods=20 * 4, freq="6h", tz="UTC")})
        split = temporal_splits(frame)
        self.assertLess(max(split["train"]), min(split["validation"]))
        self.assertLess(max(split["validation"]), min(split["test"]))

    def test_ineligible_history_is_explicit_not_a_completed_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "manifest.json", {
                "status": "complete", "news_complete": False, "jobs": [
                    {"symbol": "HDFC", "timeframe": "trading_week", "rl_eligible": False,
                     "rl_ineligible_reason": "Fewer than 15 post-predictor decision sessions"}]})
            result = train_jobs(root, root / "output", PPOConfig(), allow_partial_news=True)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["jobs"], [])
            self.assertEqual(result["unavailable_jobs"][0]["symbol"], "HDFC")

    def test_maskable_ppo_smoke(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        config = PPOConfig(hidden=(16,), lookback=2, rollout=8, batch=8, envs=1)
        vec = DummyVecEnv([lambda: PriceActionEnv(bars(), np.zeros((5, 2)), config)])
        model = MaskablePPO("MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1,
                            policy_kwargs={"net_arch": [16]}, device="cpu", seed=1)
        model.learn(total_timesteps=16)
        result = evaluate(PriceActionEnv(bars(), np.zeros((5, 2)), config), model)
        self.assertTrue(np.isfinite(result["equity"]))
        vec.close()


class NewsStreamTests(unittest.TestCase):
    def test_archive_source_imports_without_repository_files(self):
        from data_pipeline.rl_colab_data import package_archive

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            save_json(bundle / "export-state.json", {"status": "incomplete"})
            archive = package_archive(bundle, repo=repo, jobs=[], skipped=[],
                                      coverage={"status": "partial"})
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(archive) as packed:
                packed.extractall(extracted, filter="data")
            source = extracted / "stockmog-ppo" / "source"
            manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("source/docs/strategy-catalog.csv", manifest["files"])
            env = os.environ.copy()
            env["PYTHONPATH"] = str(source)
            subprocess.run(
                [sys.executable, "-c",
                 "from data_pipeline.rl_colab_training import PPOConfig, train_jobs; "
                 "from data_pipeline.rl_portfolio_training import train_portfolio; "
                 "from modeling.nifty_price_training import load_price_predictor"],
                cwd=root, env=env, check=True, capture_output=True, text=True,
            )

    def test_future_news_and_later_tone_never_leak(self):
        events = pd.DataFrame([
            ["ABC", "2025-01-01T10:00:00Z", "url1", "gkg", np.nan],
            ["ABC", "2025-01-01T11:00:00Z", "url1", "doc", 100.0],
            ["ABC", "2025-01-02T10:00:00Z", "url2", "gkg", 100.0],
        ], columns=EVENT_COLUMNS)
        decisions = pd.Series(pd.to_datetime(["2025-01-01T09:00:00Z", "2025-01-01T12:00:00Z"]))
        output = news_features(decisions, events, "ABC")
        self.assertEqual(output["news_count_24h"].iloc[0], 0)
        self.assertAlmostEqual(output["news_count_24h"].iloc[1], np.log(2))
        self.assertEqual(output["news_tone_missing_24h"].iloc[1], 1)
        self.assertEqual(output["news_tone_24h"].iloc[1], 0)

    def test_microsecond_timestamps_have_same_asof_result(self):
        events = pd.DataFrame([["ABC", "2025-01-01T10:00:00Z", "u", "gkg", 10]],
                              columns=EVENT_COLUMNS)
        decisions = pd.Series(pd.DatetimeIndex(["2025-01-01T11:00:00Z"]).as_unit("us"))
        self.assertAlmostEqual(news_features(decisions, events, "ABC")["news_count_24h"].iloc[0],
                               np.log(2))

    def test_timezone_required(self):
        frame = pd.DataFrame([["ABC", "2025-01-01", "u", "gkg", 0]], columns=EVENT_COLUMNS)
        with self.assertRaisesRegex(ValueError, "timezone"):
            normalize_events(frame)

    def test_gkg_keeps_only_metadata_and_delays_availability(self):
        row = ["id", "20250102090000", "1", "source", "https://example.com/a"] + [""] * 11
        row[13], row[15] = "RELIANCE INDUSTRIES", "12,0,0"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("gkg.csv", "\t".join(row))
        result = parse_gkg(stream.getvalue(), pd.Timestamp("2025-01-02T09:00Z"))
        self.assertEqual(result["symbol"].tolist(), ["RELIANCE"])
        self.assertEqual(result["available_at"].iloc[0], pd.Timestamp("2025-01-02T09:15Z"))
        self.assertEqual(list(result), EVENT_COLUMNS)

class NewsPriorityPolicyTests(unittest.TestCase):
    def test_extreme_news_cannot_reverse_clear_market_preference(self):
        import torch
        from data_pipeline.rl_news_policy import BoundedNewsActionNet

        network = BoundedNewsActionNet(2, 1, 3, 0.05)
        with torch.no_grad():
            network.primary.weight.zero_()
            network.primary.bias.copy_(torch.tensor([0.3, 0.0, 0.0]))
            network.news.weight.copy_(torch.tensor([[-1e6], [1e6], [1e6]]))
        inputs = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1e6], [0.0, 0.0, -1e6]])
        scores = network(inputs)
        self.assertEqual(scores.argmax(dim=1).tolist(), [0, 0, 0])
        self.assertLessEqual(float((scores - scores[0]).abs().max().detach()), 0.050001)
        network.limit = 0
        self.assertTrue(torch.equal(network(inputs), scores[0].expand_as(scores)))

    def test_news_does_not_enter_critic_or_bypass_masks(self):
        import torch
        from gymnasium import spaces
        from data_pipeline.rl_news_policy import LowPriorityNewsPolicy

        policy = LowPriorityNewsPolicy(
            spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32), spaces.Discrete(3),
            lambda _: 3e-4, news_indices=[1, 3], news_logit_limit=0.05,
            net_arch={"pi": [16], "vf": [16]})
        inputs = torch.tensor([[1.0, 0.0, 2.0, 0.0], [1.0, 1e6, 2.0, -1e6]])
        with torch.no_grad():
            policy.action_net.news.weight.fill_(1e6)
            values = policy.predict_values(inputs)
            self.assertTrue(torch.equal(values[0], values[1]))
            actions, _, log_prob = policy(inputs, deterministic=True,
                                          action_masks=np.array([[True, False, False]] * 2))
            self.assertEqual(actions.tolist(), [0, 0])
            _, checked_log_prob, _ = policy.evaluate_actions(
                inputs, actions, action_masks=torch.tensor([[True, False, False]] * 2))
            self.assertTrue(torch.allclose(log_prob, checked_log_prob))

    def test_low_priority_ppo_trains_and_reloads(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from data_pipeline.rl_news_policy import LowPriorityNewsPolicy

        config = PPOConfig(lookback=2)
        inputs = np.column_stack([np.zeros(5), np.arange(5)]).astype(np.float32)
        env = DummyVecEnv([lambda: PriceActionEnv(bars(), inputs, config)])
        model = MaskablePPO(
            LowPriorityNewsPolicy, env, n_steps=8, batch_size=8, n_epochs=1, seed=1,
            policy_kwargs={"news_indices": [1, 3], "net_arch": {"pi": [16], "vf": [16]}})
        model.learn(total_timesteps=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.zip"
            model.save(path)
            loaded = MaskablePPO.load(path, env=env)
            self.assertEqual(loaded.policy.news_logit_limit, 0.05)
            self.assertEqual(loaded.policy.action_net.limit, 0.05)
            result = evaluate(PriceActionEnv(bars(), inputs, config), loaded)
            self.assertTrue(np.isfinite(result["equity"]))
        env.close()

    def test_research_branch_is_bounded_and_excluded_from_critic(self):
        import torch
        from gymnasium import spaces
        from data_pipeline.rl_news_policy import LowPriorityNewsPolicy

        policy = LowPriorityNewsPolicy(
            spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32), spaces.Discrete(3),
            lambda _: 3e-4, news_indices=[1], research_indices=[3, 4],
            research_logit_limit=0.15, net_arch={"pi": [16], "vf": [16]})
        self.assertEqual(policy.mlp_extractor.primary_indices.tolist(), [0, 2])
        inputs = torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 2.0, 1e6, -1e6]])
        with torch.no_grad():
            self.assertTrue(torch.equal(policy.action_net.research.weight, torch.zeros(3, 2)))
            policy.action_net.research.weight.copy_(torch.tensor([[1e6, 0.0], [-1e6, 0.0], [0.0, 1e6]]))
            values = policy.predict_values(inputs)
            self.assertTrue(torch.equal(values[0], values[1]))
            logits = policy.action_net(policy.mlp_extractor.forward_actor(inputs))
        shift = (logits[1] - logits[0]).abs()
        self.assertLessEqual(float(shift.max()), 0.150001)
        self.assertGreater(float(shift.max()), 0.14)
        with self.assertRaises(ValueError):
            LowPriorityNewsPolicy(
                spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32), spaces.Discrete(3),
                lambda _: 3e-4, news_indices=[1], research_indices=[1],
                net_arch={"pi": [16], "vf": [16]})


class ResearchStreamTests(unittest.TestCase):
    def event(self, when, kind, score=np.nan, value=np.nan, source="nse_filing", symbol="M&M"):
        return {"symbol": symbol, "available_at": pd.Timestamp(when, tz="UTC"), "source": source,
                "kind": kind, "event_id": f"{kind}-{when}", "score": score, "value": value}

    def test_features_only_use_events_available_by_decision(self):
        from data_pipeline.rl_research_stream import research_features

        events = pd.DataFrame([
            self.event("2025-01-01 10:00", "credit_rating", score=1.0),
            self.event("2025-01-02 10:00", "earnings", 0.4, 0.2, source="nse_results"),
            self.event("2025-01-03 10:00", "analyst_report", 0.5, 120.0, source="user_report"),
            self.event("2025-01-05 10:00", "earnings", -0.9, -0.9, source="nse_results"),
        ])
        decisions = pd.Series(pd.to_datetime(
            ["2025-01-01 09:59", "2025-01-03 10:00", "2025-01-05 09:59"], utc=True))
        features = research_features(decisions, pd.Series([100.0, 100.0, 150.0]), events, "M&M")
        self.assertTrue(features.iloc[0][["research_credit_score", "research_profit_yoy",
                                          "research_target_upside"]].isna().all())
        self.assertEqual(features.iloc[1]["research_credit_score"], 1.0)
        self.assertAlmostEqual(features.iloc[1]["research_target_upside"], 0.2)
        self.assertEqual(features.iloc[1]["research_analyst_rating_90d"], 0.5)
        self.assertEqual(features.iloc[2]["research_profit_yoy"], 0.4)
        self.assertAlmostEqual(features.iloc[2]["research_target_upside"], -0.2)
        other = research_features(decisions, pd.Series([1.0] * 3), events, "INFY")
        self.assertTrue((other["research_covered"] == 0).all())
        self.assertTrue(other.drop(columns="research_covered").isna().all().all())

    def test_user_reports_are_lagged_and_validated(self):
        from data_pipeline.rl_research_stream import report_events

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports.csv"
            path.write_text("symbol,published_at,rating,target_price\n"
                            "ONGC,2025-03-10,Buy,300\n"
                            "ONGC,2025-03-11T08:00:00+05:30,Sell,\n"
                            "INFY,2025-03-10,Buy,2000\n", encoding="utf-8")
            events = report_events(path, ("ONGC",))
            self.assertEqual(events["available_at"].tolist(), [
                pd.Timestamp("2025-03-11 09:00", tz="Asia/Kolkata"),
                pd.Timestamp("2025-03-11 08:15", tz="Asia/Kolkata")])
            self.assertEqual(events["score"].tolist(), [1.0, -1.0])
            self.assertEqual(events["value"].iloc[0], 300.0)
            for row in ("ONGC,2025-03-11 08:00,Buy,300", "ONGC,2025-03-10,Moon,300"):
                path.write_text("symbol,published_at,rating,target_price\n" + row + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    report_events(path, ("ONGC",))

    def test_earnings_yoy_prefers_consolidated_quarter_facts(self):
        from data_pipeline.rl_research_stream import earnings_events

        def xbrl(start, end, profit, revenue):
            return (f'<xbrli:xbrl xmlns:xbrli="x" xmlns:in="i">'
                    f'<xbrli:context id="Q"><xbrli:period><xbrli:startDate>{start}</xbrli:startDate>'
                    f'<xbrli:endDate>{end}</xbrli:endDate></xbrli:period></xbrli:context>'
                    f'<xbrli:context id="S"><xbrli:segment/><xbrli:period><xbrli:startDate>{start}'
                    f'</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate></xbrli:period></xbrli:context>'
                    f'<in:ProfitLossForPeriod contextRef="S">999</in:ProfitLossForPeriod>'
                    f'<in:ProfitLossForPeriod contextRef="Q">{profit}</in:ProfitLossForPeriod>'
                    f'<in:RevenueFromOperations contextRef="Q">{revenue}</in:RevenueFromOperations>'
                    f'</xbrli:xbrl>').encode()

        files = {"old.xml": xbrl("2024-07-01", "2024-09-30", 100, 1000),
                 "standalone.xml": xbrl("2024-07-01", "2024-09-30", 1, 1),
                 "new.xml": xbrl("2025-07-01", "2025-09-30", 150, 1000)}

        class FakeClient:
            def fetch(self, url, params=None, *, binary=False, use_cache=True):
                if binary:
                    return files[url]
                if "integrated" in url:
                    return {"data": [{"qe_Date": "30-SEP-2025", "broadcast_Date": "10-Nov-2025 18:00:00",
                                      "consolidated": "Consolidated", "xbrl": "new.xml",
                                      "type_Sub": "Original"}], "totalCount": 1}
                return [{"toDate": "30-Sep-2024", "exchdisstime": "08-Nov-2024 17:00:00",
                         "consolidated": "Non-Consolidated", "xbrl": "standalone.xml"},
                        {"toDate": "30-Sep-2024", "exchdisstime": "08-Nov-2024 17:05:00",
                         "consolidated": "Consolidated", "xbrl": "old.xml"}]

        events, warnings = earnings_events("M&M", FakeClient())
        self.assertEqual(warnings, [])
        latest = events.loc[events["event_id"] == "2025-09-30"].iloc[0]
        self.assertAlmostEqual(latest["score"], np.tanh(0.5))
        self.assertAlmostEqual(latest["value"], 0.0)
        self.assertEqual(latest["available_at"], pd.Timestamp("2025-11-10 18:15", tz="Asia/Kolkata"))


if __name__ == "__main__":
    unittest.main()
