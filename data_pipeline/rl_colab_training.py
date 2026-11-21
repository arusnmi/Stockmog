"""Causal buy/sell/stay PPO experiments for independent stock/timeframe jobs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from data_pipeline.rl_colab_data import sha256
from data_pipeline.rl_news_stream import news_features, save_json
from data_pipeline.rl_research_stream import research_features


@dataclass(frozen=True)
class PPOConfig:
    steps: int = 262_144
    rollout: int = 1024
    batch: int = 1024
    envs: int = 4
    epochs: int = 5
    hidden: tuple[int, ...] = (512, 512, 256)
    learning_rate: float = 3e-4
    lookback: int = 20
    episode_bars: int = 2048
    initial_cash: float = 50_000
    exposure: float = 0.1
    fee: float = 0.001
    slippage: float = 0.0005
    drawdown_penalty: float = 0.1
    downside_penalty: float = 0.01
    max_drawdown: float = 0.5
    news_logit_limit: float = 0.05
    research_logit_limit: float = 0.15
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.steps, self.rollout, self.batch, self.envs, self.epochs,
               self.lookback, self.episode_bars, *self.hidden) < 1:
            raise ValueError("PPO sizes must be positive")
        if not self.hidden or self.batch > self.rollout * self.envs:
            raise ValueError("Invalid network or minibatch size")
        if self.rollout * self.envs % self.batch:
            raise ValueError("Minibatch size must divide the rollout buffer")
        if not 0 < self.exposure <= 1 or not 0 < self.max_drawdown < 1:
            raise ValueError("Invalid exposure or drawdown limit")
        if (not math.isfinite(self.initial_cash) or self.initial_cash <= 0
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("Capital and learning rate must be finite and positive")
        if not all(math.isfinite(x) and 0 <= x < 1 for x in
                   (self.fee, self.slippage, self.drawdown_penalty, self.downside_penalty)):
            raise ValueError("Invalid execution cost or reward penalty")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Choose cpu or cuda")
        if not math.isfinite(self.news_logit_limit) or not 0 <= self.news_logit_limit <= 0.1:
            raise ValueError("News logit limit must be between zero and 0.1")
        if not math.isfinite(self.research_logit_limit) or not 0 <= self.research_logit_limit <= 0.3:
            raise ValueError("Research logit limit must be between zero and 0.3")


class PriceActionEnv(gym.Env):
    """NumPy execution avoids per-bar Backtrader threads during PPO rollouts."""

    metadata = {"render_modes": []}

    def __init__(self, bars: pd.DataFrame, observations: np.ndarray, config: PPOConfig,
                 *, training: bool = False):
        super().__init__()
        if len(bars) < 3 or len(bars) != len(observations):
            raise ValueError("At least three aligned bars and observations are required")
        prices = bars[["open", "close"]].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(observations).all():
            raise ValueError("Prices and observations must be finite; prices must be positive")
        starts = pd.DatetimeIndex(bars["timestamp"]).as_unit("ns")
        available = pd.DatetimeIndex(bars["available_at"]).as_unit("ns")
        if (starts.tz is None or available.tz is None or starts.hasnans or available.hasnans
                or starts.has_duplicates
                or not starts.is_monotonic_increasing or not available.is_monotonic_increasing
                or np.any(available.asi8 <= starts.asi8)):
            raise ValueError("Bar starts/availability must be ordered and causally timestamped")
        self.config, self.training = config, training
        self.opens, self.closes = prices[:, 0], prices[:, 1]
        self.inputs = np.asarray(observations, dtype=np.float32)
        self.next_bar = np.maximum(np.arange(len(bars)) + 1,
                                   np.searchsorted(starts.asi8, available.asi8, side="left"))
        self.action_space = spaces.Discrete(3)  # 0 stay, 1 buy, 2 sell
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(config.lookback * self.inputs.shape[1] + 5,), dtype=np.float32)
        self.reset()

    def _observation(self) -> np.ndarray:
        window = np.zeros((self.config.lookback, self.inputs.shape[1]), dtype=np.float32)
        start = max(self.start, self.index - self.config.lookback + 1)
        data = self.inputs[start:self.index + 1]
        window[-len(data):] = data
        state = [self.shares * self.closes[self.index] / self.equity,
                 self.cash / self.equity, np.log(self.equity / self.config.initial_cash),
                 1 - self.equity / self.peak, (self.end - self.index) / (self.end - self.start)]
        return np.concatenate([window.ravel(), np.asarray(state, dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        last_start = max(0, len(self.inputs) - min(self.config.episode_bars, len(self.inputs)))
        self.start = int(self.np_random.integers(last_start + 1)) if self.training else 0
        self.end = (min(len(self.inputs), self.start + self.config.episode_bars)
                    if self.training else len(self.inputs))
        self.index = self.start
        self.cash = self.equity = self.peak = float(self.config.initial_cash)
        self.shares = self.trades = self.rejections = 0
        self.done = False
        return self._observation(), {}

    def action_masks(self) -> np.ndarray:
        return np.asarray([True, self.shares == 0, self.shares > 0], dtype=bool)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset a terminated episode before stepping")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0=stay, 1=buy, 2=sell")
        invalid = not self.action_masks()[action]
        previous, prior_dd = self.equity, 1 - self.equity / self.peak
        target = int(self.next_bar[self.index])
        if target >= self.end:
            self.done = True
            return self._observation(), 0.0, True, False, self.summary()
        price = self.opens[target]
        if invalid:
            self.rejections += 1
        elif action == 1:
            cost = price * (1 + self.config.slippage) * (1 + self.config.fee)
            quantity = int(min(self.cash, self.equity * self.config.exposure) // cost)
            if quantity:
                self.cash -= quantity * cost
                self.shares = quantity
                self.trades += 1
            else:
                self.rejections += 1
        elif action == 2:
            self.cash += self.shares * price * (1 - self.config.slippage) * (1 - self.config.fee)
            self.shares = 0
            self.trades += 1
        self.index = target
        self.equity = self.cash + self.shares * self.closes[target]
        self.peak = max(self.peak, self.equity)
        dd = 1 - self.equity / self.peak
        log_return = np.log(self.equity / previous)
        reward = (log_return - self.config.drawdown_penalty * max(0, dd - prior_dd)
                  - self.config.downside_penalty * min(0, log_return) ** 2
                  - (0.001 if invalid else 0))
        self.done = target >= self.end - 1 or dd >= self.config.max_drawdown
        return self._observation(), float(reward), self.done, False, self.summary()

    def summary(self) -> dict:
        return {"equity": self.equity, "profit": self.equity - self.config.initial_cash,
                "return_percent": (self.equity / self.config.initial_cash - 1) * 100,
                "drawdown_percent": (1 - self.equity / self.peak) * 100,
                "trades": self.trades, "rejections": self.rejections, "open_shares": self.shares,
                "ending_valuation": "marked to market; open shares are not assumed sold"}


def temporal_splits(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    days = pd.to_datetime(frame["available_at"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    unique = sorted(set(days))
    if len(unique) < 15:
        raise ValueError("Fewer than 15 post-predictor decision sessions; cannot form useful RL splits")
    validation, test = unique[int(len(unique) * 0.6)], unique[int(len(unique) * 0.8)]
    return {"train": np.flatnonzero(days < validation),
            "validation": np.flatnonzero((days >= validation) & (days < test)),
            "test": np.flatnonzero(days >= test)}


def predictor_stream(frame: pd.DataFrame, meta: dict, model_path: Path) -> np.ndarray:
    from modeling.nifty_price_training import load_price_predictor

    predictor = load_price_predictor(model_path, device="cpu")
    names = meta["feature_names"]
    matrix = frame[names].to_numpy(dtype=np.float32)
    length = meta["sequence_length"]
    available = pd.to_datetime(frame["available_at"], utc=True)
    starts = frame["sequence_start"].to_numpy()
    indices = np.flatnonzero((available >= pd.Timestamp(meta["predictor_frozen_at"]))
                             & (np.arange(len(frame)) - starts + 1 >= length))
    values = np.full(len(frame), np.nan, dtype=np.float32)
    for offset in range(0, len(indices), 512):
        batch = indices[offset:offset + 512]
        sequences = matrix[batch[:, None] - np.arange(length - 1, -1, -1)]
        anchors = frame["close"].to_numpy(dtype=float)[batch]
        prices = np.asarray(predictor.predict(matrix[batch], sequences, anchors, feature_names=names))
        if prices.shape != anchors.shape or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Frozen price predictor returned invalid prices")
        values[batch] = np.log(prices / anchors)
    return values


def prepare_job(bundle: Path, manifest: dict, job: dict, *, allow_partial_news: bool = False,
                allow_partial_research: bool = False):
    if not manifest["news_complete"] and not allow_partial_news:
        raise ValueError("News coverage is incomplete. Explicitly acknowledge with --allow-partial-news.")
    has_research = "research.parquet" in manifest["files"]
    if has_research and not manifest.get("research_complete", False) and not allow_partial_research:
        raise ValueError("Research coverage is partial. Explicitly acknowledge with --allow-partial-research.")
    folder = bundle / job["folder"]
    relevant = [entry for entry in manifest["jobs"] if entry["symbol"] == job["symbol"]]
    required = ["news.parquet"] + (["research.parquet"] if has_research else [])
    for entry in relevant:
        required.append(entry["folder"] + "/history.parquet")
        required.append(entry["folder"] + "/prepared.json")
    required += [job["folder"] + "/" + name for name in
                 ("prepared.json", "model/price-lstm.pt", "model/price-tree.ubj",
                  "model/training-summary.json")]
    for name in required:
        if sha256(bundle / name) != manifest["files"][name]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    meta = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(folder / "history.parquet")
    features = frame[["available_at", *meta["feature_names"]]].copy()
    for entry in relevant:
        if entry["timeframe"] == job["timeframe"]:
            continue
        other_meta = json.loads((bundle / entry["folder"] / "prepared.json").read_text(encoding="utf-8"))
        columns = ["ret_5", "ret_20", "rsi_14", *[
            name for name in other_meta["feature_names"] if name.startswith("factor:")]]
        columns = list(dict.fromkeys(columns))
        other = pd.read_parquet(bundle / entry["folder"] / "history.parquet",
                                columns=["available_at", *columns])
        cutoff = pd.Timestamp(other_meta["selection"]["train_end_exclusive"], tz="Asia/Kolkata")
        other = other.loc[other["available_at"] >= cutoff]
        other = other.rename(columns={column: f"{entry['timeframe']}:{column}" for column in columns})
        features = pd.merge_asof(features, other, on="available_at", direction="backward",
                                 tolerance=pd.Timedelta(days=14))
    features["model_log_return"] = predictor_stream(frame, meta, folder / "model")
    news = news_features(frame["available_at"], pd.read_parquet(bundle / "news.parquet"), job["symbol"])
    parts = [features.drop(columns="available_at"), news]
    if has_research:
        parts.append(research_features(frame["available_at"], frame["close"],
                                       pd.read_parquet(bundle / "research.parquet"), job["symbol"]))
    features = pd.concat(parts, axis=1)
    eligible = (frame["available_at"] >= pd.Timestamp(meta["predictor_frozen_at"]))
    frame, features = frame.loc[eligible].reset_index(drop=True), features.loc[eligible].reset_index(drop=True)
    splits = temporal_splits(frame)
    values = features.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Infinite input feature")
    train_values = pd.DataFrame(values[splits["train"]])
    mean = train_values.mean().fillna(0).to_numpy(dtype=np.float32)
    scale = train_values.std(ddof=0).fillna(1).to_numpy(dtype=np.float32)
    scale[scale < 1e-6] = 1
    observed = np.isfinite(values).astype(np.float32)
    values = np.clip(np.nan_to_num((values - mean) / scale), -10, 10)
    values = np.concatenate([values, observed], axis=1)
    return frame, values, splits, {"features": list(features), "mean": mean.tolist(), "scale": scale.tolist()}


def evaluate(env: PriceActionEnv, model=None, *, buy_hold: bool = False) -> dict:
    observation, _ = env.reset()
    done, peak_dd, returns = False, 0.0, []
    while not done:
        if model is not None:
            action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
        else:
            action = 1 if buy_hold and env.shares == 0 else 0
        before = env.equity
        observation, _, done, _, info = env.step(action)
        returns.append(np.log(env.equity / before))
        peak_dd = max(peak_dd, info["drawdown_percent"])
    info.update(max_drawdown_percent=peak_dd, decisions=len(returns),
                downside_rms=float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2))))
    return info


def train_jobs(bundle: Path, output: Path, config: PPOConfig, *, symbols: list[str] | None = None,
               timeframes: list[str] | None = None, allow_partial_news: bool = False,
               allow_partial_research: bool = False, evaluate_test: bool = False) -> dict:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from data_pipeline.rl_news_policy import LowPriorityNewsPolicy

    torch.set_num_threads(2)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Dataset export is incomplete")
    jobs = [job for job in manifest["jobs"]
            if (not symbols or job["symbol"] in symbols)
            and (not timeframes or job["timeframe"] in timeframes)]
    if not jobs:
        raise ValueError("No matching policy jobs")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"dataset_sha256": sha256(bundle / "manifest.json"),
                "config": json.loads(json.dumps(asdict(config))),
                "allow_partial_news": allow_partial_news,
                "allow_partial_research": allow_partial_research}
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Training settings or data changed; use a new output directory")
    else:
        save_json(identity_path, identity)
    results, unavailable = [], []
    total_started = time.monotonic()
    for index, job in enumerate(jobs, 1):
        if job.get("rl_eligible") is False:
            record = {"symbol": job["symbol"], "timeframe": job["timeframe"],
                      "reason": job["rl_ineligible_reason"]}
            unavailable.append(record)
            print("RL JOB UNAVAILABLE", json.dumps(record), flush=True)
            continue
        folder = output / job["folder"]
        report_path = folder / ("test.json" if evaluate_test else "validation.json")
        if report_path.exists():
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            if sha256(folder / "policy.zip") != saved["policy_sha256"]:
                raise ValueError(f"Policy changed after evaluation: {folder}")
            results.append(saved)
            continue
        started = time.monotonic()
        bars, values, splits, normalizer = prepare_job(
            bundle, manifest, job, allow_partial_news=allow_partial_news,
            allow_partial_research=allow_partial_research)
        folder.mkdir(parents=True, exist_ok=True)
        train_rows = splits["train"]
        if min(len(part) for part in splits.values()) < 3:
            raise ValueError(f"Too few rows in RL split: {job}")
        def make_env():
            return PriceActionEnv(bars.iloc[train_rows], values[train_rows], config, training=True)
        env = DummyVecEnv([make_env for _ in range(config.envs)])
        policy_path = folder / "policy.zip"
        progress = folder / "progress.json"
        if policy_path.exists():
            if not progress.exists() or json.loads(progress.read_text())["sha256"] != sha256(policy_path):
                raise ValueError(f"Incomplete or corrupt policy checkpoint: {folder}")
            model = MaskablePPO.load(policy_path, env=env, device=config.device)
        else:
            if evaluate_test:
                raise FileNotFoundError(f"No trained policy: {policy_path}")
            feature_count = len(normalizer["features"])
            news_columns = [index for index, name in enumerate(normalizer["features"])
                            if name.startswith("news_")]
            research_columns = [index for index, name in enumerate(normalizer["features"])
                                if name.startswith("research_")]

            def window_indices(columns: list[int]) -> list[int]:
                return [row * values.shape[1] + column + mask_offset
                        for row in range(config.lookback) for column in columns
                        for mask_offset in (0, feature_count)]
            news_indices = window_indices(news_columns)
            research_indices = window_indices(research_columns)
            model = MaskablePPO(LowPriorityNewsPolicy, env, device=config.device, seed=config.seed,
                                n_steps=config.rollout, batch_size=config.batch, n_epochs=config.epochs,
                                learning_rate=config.learning_rate, gamma=0.99, gae_lambda=0.95,
                                clip_range=0.2, ent_coef=0.005, target_kl=0.03,
                                policy_kwargs={"net_arch": {"pi": list(config.hidden), "vf": list(config.hidden)},
                                               "news_indices": news_indices,
                                               "news_logit_limit": config.news_logit_limit,
                                               "research_indices": research_indices,
                                               "research_logit_limit": config.research_logit_limit,
                                               "ortho_init": False})
        if evaluate_test and model.num_timesteps < config.steps:
            raise ValueError("Complete training before evaluating the test period")
        if not evaluate_test:
            while model.num_timesteps < config.steps:
                steps = min(65_536, config.steps - model.num_timesteps)
                model.learn(total_timesteps=steps, reset_num_timesteps=False)
                temporary = folder / "policy.tmp.zip"
                model.save(temporary)
                temporary.replace(policy_path)
                save_json(progress, {"timesteps": model.num_timesteps, "sha256": sha256(policy_path)})
                print(f"{job['symbol']} {job['timeframe']}: {model.num_timesteps}/{config.steps}", flush=True)
            save_json(folder / "normalization.json", normalizer)
        selected = splits["test" if evaluate_test else "validation"]
        def eval_env():
            return PriceActionEnv(bars.iloc[selected], values[selected], config)
        result = {"symbol": job["symbol"], "timeframe": job["timeframe"],
                  "split": "test" if evaluate_test else "validation",
                  "ppo": evaluate(eval_env(), model), "buy_hold": evaluate(eval_env(), buy_hold=True),
                  "cash": evaluate(eval_env()), "elapsed_seconds": time.monotonic() - started,
                  "timesteps": model.num_timesteps, "policy_sha256": sha256(policy_path),
                  "split_rows": {name: len(rows) for name, rows in splits.items()},
                  "news_complete": manifest["news_complete"],
                  "research_included": "research.parquet" in manifest["files"]}
        save_json(report_path, result)
        results.append(result)
        env.close()
        print(f"Finished {index}/{len(jobs)}: {job['symbol']} {job['timeframe']}", flush=True)
    summary = {"status": "partial" if unavailable else "complete", "research_only": True,
               "jobs": results, "unavailable_jobs": unavailable,
               "elapsed_seconds": time.monotonic() - total_started,
               "warning": "Independent stock/timeframe accounts are not a capital-constrained portfolio."}
    save_json(output / ("test-summary.json" if evaluate_test else "validation-summary.json"), summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--timeframes", nargs="+")
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--allow-partial-news", action="store_true")
    parser.add_argument("--allow-partial-research", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    train_jobs(args.bundle, args.output, PPOConfig(steps=args.steps, device=args.device),
               symbols=args.symbols, timeframes=args.timeframes,
               allow_partial_news=args.allow_partial_news,
               allow_partial_research=args.allow_partial_research, evaluate_test=args.evaluate_test)
