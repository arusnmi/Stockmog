"""Train one shared-cash PPO portfolio using all stock timeframes."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from data_pipeline.rl_colab_data import sha256
from data_pipeline.rl_colab_training import PPOConfig, predictor_stream, prepare_job_features, temporal_splits
from data_pipeline.rl_news_stream import save_json


PORTFOLIO_SYMBOLS = ("BSE", "M&M", "MAXHEALTH", "ONGC")
TIMEFRAMES = ("1m", "5m", "15m", "20m", "30m", "1h", "1d", "2d", "trading_week")


def prepare_portfolio(bundle: Path, manifest: dict, *, allow_partial_news: bool = False,
                      allow_partial_research: bool = False):
    """Align stocks at common minute decisions; fit scaling only on portfolio training sessions."""
    jobs = {(j["symbol"], j["timeframe"]): j for j in manifest["jobs"]}
    missing = [(symbol, tf) for symbol in PORTFOLIO_SYMBOLS for tf in TIMEFRAMES
               if (symbol, tf) not in jobs]
    if missing:
        raise ValueError(f"Portfolio requires all stock/timeframe jobs: {missing}")
    frames, inputs = [], []
    for symbol in PORTFOLIO_SYMBOLS:
        print(f"Preparing portfolio inputs for {symbol}", flush=True)
        frame, features = prepare_job_features(
            bundle, manifest, jobs[symbol, "1m"], allow_partial_news=allow_partial_news,
            allow_partial_research=allow_partial_research)
        for tf in TIMEFRAMES[1:]:
            job = jobs[symbol, tf]
            folder = bundle / job["folder"]
            for name in ("history.parquet", "prepared.json", "model/price-lstm.pt",
                         "model/price-tree.ubj", "model/training-summary.json"):
                key = job["folder"] + "/" + name
                if sha256(bundle / key) != manifest["files"][key]:
                    raise ValueError(f"Dataset checksum mismatch: {key}")
            meta = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
            history = pd.read_parquet(folder / "history.parquet")
            predictions = pd.DataFrame({
                "available_at": history["available_at"],
                f"{tf}:model_log_return": predictor_stream(history, meta, folder / "model"),
            })
            joined = pd.merge_asof(frame[["available_at"]], predictions, on="available_at",
                                   direction="backward", tolerance=pd.Timedelta(days=14))
            features[f"{tf}:model_log_return"] = joined[f"{tf}:model_log_return"].to_numpy()
        frame = frame[["timestamp", "available_at", "open", "close"]].copy()
        if frame["available_at"].duplicated().any():
            raise ValueError(f"Duplicate decision times for {symbol}")
        frame = frame.set_index("available_at")
        features.index = frame.index
        frames.append(frame)
        inputs.append(features.add_prefix(symbol + ":"))

    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index, sort=False)
    common = common.sort_values()
    if len(common) < 3:
        raise ValueError("Insufficient common minute data for the portfolio")
    frames = [frame.loc[common] for frame in frames]
    timestamps = frames[0]["timestamp"]
    if any(not timestamps.equals(frame["timestamp"]) for frame in frames[1:]):
        raise ValueError("Stock decision times do not share the same candle starts")
    bars = pd.DataFrame({"timestamp": timestamps.to_numpy(), "available_at": common})
    for symbol, frame in zip(PORTFOLIO_SYMBOLS, frames):
        bars[f"{symbol}:open"] = frame["open"].to_numpy()
        bars[f"{symbol}:close"] = frame["close"].to_numpy()
    features = pd.concat([frame.loc[common].reset_index(drop=True) for frame in inputs], axis=1)
    splits = temporal_splits(bars)
    values = features.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Infinite portfolio input feature")
    train_values = pd.DataFrame(values[splits["train"]])
    mean = train_values.mean().fillna(0).to_numpy(dtype=np.float32)
    scale = train_values.std(ddof=0).fillna(1).to_numpy(dtype=np.float32)
    scale[scale < 1e-6] = 1
    observed = np.isfinite(values).astype(np.float32)
    values = np.clip(np.nan_to_num((values - mean) / scale), -10, 10)
    values = np.concatenate([values, observed], axis=1)
    normalizer = {"features": list(features), "mean": mean.tolist(), "scale": scale.tolist()}
    return bars, values, splits, normalizer


class SharedPortfolioEnv(gym.Env):
    """Single cash account; each action specifies stay/buy/sell for each stock."""

    metadata = {"render_modes": []}

    def __init__(self, bars: pd.DataFrame, observations: np.ndarray, config: PPOConfig,
                 *, training: bool = False):
        super().__init__()
        if len(bars) < 3 or len(bars) != len(observations):
            raise ValueError("At least three aligned portfolio bars are required")
        self.symbols = PORTFOLIO_SYMBOLS
        self.config, self.training = config, training
        self.opens = bars[[f"{s}:open" for s in self.symbols]].to_numpy(dtype=float)
        self.closes = bars[[f"{s}:close" for s in self.symbols]].to_numpy(dtype=float)
        self.inputs = np.asarray(observations, dtype=np.float32)
        starts = pd.DatetimeIndex(bars["timestamp"]).as_unit("ns")
        available = pd.DatetimeIndex(bars["available_at"]).as_unit("ns")
        if (starts.tz is None or available.tz is None or starts.hasnans or available.hasnans
                or starts.has_duplicates or not starts.is_monotonic_increasing
                or not available.is_monotonic_increasing or np.any(available.asi8 <= starts.asi8)):
            raise ValueError("Portfolio bar times must be ordered and causal")
        if (not np.isfinite(self.opens).all() or not np.isfinite(self.closes).all()
                or np.any(self.opens <= 0) or np.any(self.closes <= 0)
                or not np.isfinite(self.inputs).all()):
            raise ValueError("Portfolio prices and inputs must be finite and prices positive")
        self.next_bar = np.maximum(np.arange(len(bars)) + 1,
                                   np.searchsorted(starts.asi8, available.asi8, side="left"))
        self.actions = np.asarray(list(product(range(3), repeat=len(self.symbols))), dtype=np.int8)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(config.lookback * self.inputs.shape[1] + len(self.symbols) + 4,),
            dtype=np.float32)
        self.reset()

    def _observation(self) -> np.ndarray:
        window = np.zeros((self.config.lookback, self.inputs.shape[1]), dtype=np.float32)
        start = max(self.start, self.index - self.config.lookback + 1)
        data = self.inputs[start:self.index + 1]
        window[-len(data):] = data
        positions = self.shares * self.closes[self.index] / self.equity
        state = [self.cash / self.equity, np.log(self.equity / self.config.initial_cash),
                 1 - self.equity / self.peak, (self.end - self.index) / (self.end - self.start)]
        return np.concatenate([window.ravel(), positions.astype(np.float32),
                               np.asarray(state, dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        last_start = max(0, len(self.inputs) - min(self.config.episode_bars, len(self.inputs)))
        self.start = int(self.np_random.integers(last_start + 1)) if self.training else 0
        self.end = min(len(self.inputs), self.start + self.config.episode_bars) if self.training else len(self.inputs)
        self.index = self.start
        self.cash = self.equity = self.peak = float(self.config.initial_cash)
        self.shares = np.zeros(len(self.symbols), dtype=np.int64)
        self.trades = self.rejections = 0
        self.done = False
        return self._observation(), {}

    def action_masks(self) -> np.ndarray:
        buys = self.actions == 1
        sells = self.actions == 2
        valid_sell = (self.actions != 2) | (self.shares > 0)
        prices = self.closes[self.index]
        estimated_proceeds = (sells * self.shares * prices).sum(axis=1) * (
            1 - self.config.slippage) * (1 - self.config.fee)
        budget = (self.cash + estimated_proceeds) / np.maximum(buys.sum(axis=1), 1)
        estimated_cost = prices * (1 + self.config.slippage) * (1 + self.config.fee)
        valid_buy = ~buys | ((self.shares == 0) & (budget[:, None] >= estimated_cost))
        return np.all(valid_buy & valid_sell, axis=1)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset a terminated episode before stepping")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Invalid portfolio action")
        actions = self.actions[action]
        invalid = not self.action_masks()[action]
        prior_rejections = self.rejections
        previous, prior_dd = self.equity, 1 - self.equity / self.peak
        target = int(self.next_bar[self.index])
        if target >= self.end:
            self.done = True
            return self._observation(), 0.0, True, False, self.summary()
        prices = self.opens[target]
        if invalid:
            self.rejections += 1
        else:
            selling = np.flatnonzero(actions == 2)
            for index in selling:
                self.cash += self.shares[index] * prices[index] * (1 - self.config.slippage) * (1 - self.config.fee)
                self.shares[index] = 0
                self.trades += 1
            buying = np.flatnonzero(actions == 1)
            budget = self.cash / len(buying) if len(buying) else 0
            for index in buying:
                unit_cost = prices[index] * (1 + self.config.slippage) * (1 + self.config.fee)
                quantity = int(budget // unit_cost)
                if quantity:
                    self.cash -= quantity * unit_cost
                    self.shares[index] = quantity
                    self.trades += 1
                else:
                    self.rejections += 1
        self.index = target
        self.equity = float(self.cash + np.dot(self.shares, self.closes[target]))
        self.peak = max(self.peak, self.equity)
        dd = 1 - self.equity / self.peak
        log_return = np.log(self.equity / previous)
        reward = (log_return - self.config.drawdown_penalty * max(0, dd - prior_dd)
                  - self.config.downside_penalty * min(0, log_return) ** 2
                  - (0.001 if self.rejections > prior_rejections else 0))
        self.done = target >= self.end - 1 or dd >= self.config.max_drawdown
        return self._observation(), float(reward), self.done, False, self.summary()

    def summary(self) -> dict:
        return {"equity": self.equity, "cash": self.cash,
                "profit": self.equity - self.config.initial_cash,
                "return_percent": (self.equity / self.config.initial_cash - 1) * 100,
                "drawdown_percent": (1 - self.equity / self.peak) * 100,
                "trades": self.trades, "rejections": self.rejections,
                "open_shares": dict(zip(self.symbols, self.shares.tolist())),
                "ending_valuation": "marked to market; open shares are not assumed sold"}


def evaluate_portfolio(env: SharedPortfolioEnv, model=None, *, buy_hold: bool = False) -> dict:
    observation, _ = env.reset()
    done, peak_dd, returns = False, 0.0, []
    while not done:
        if model is not None:
            action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
        else:
            desired = np.where(env.shares == 0, 1, 0) if buy_hold and env.trades == 0 else np.zeros(len(env.symbols))
            action = int(np.flatnonzero(np.all(env.actions == desired, axis=1))[0])
        before = env.equity
        observation, _, done, _, info = env.step(action)
        returns.append(np.log(env.equity / before))
        peak_dd = max(peak_dd, info["drawdown_percent"])
    info.update(max_drawdown_percent=peak_dd, decisions=len(returns),
                downside_rms=float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2))))
    return info


def train_portfolio(bundle: Path, output: Path, config: PPOConfig, *,
                    allow_partial_news: bool = False, allow_partial_research: bool = False,
                    evaluate_test: bool = False) -> dict:
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
    identity = {"dataset_sha256": sha256(bundle / "manifest.json"),
                "config": json.loads(json.dumps(asdict(config))),
                "symbols": list(PORTFOLIO_SYMBOLS), "timeframes": list(TIMEFRAMES),
                "allow_partial_news": allow_partial_news, "allow_partial_research": allow_partial_research}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Training settings or data changed; use a new output directory")
    else:
        save_json(identity_path, identity)
    policy_path, progress = output / "policy.zip", output / "progress.json"
    if evaluate_test and not policy_path.exists():
        raise FileNotFoundError(f"No trained policy: {policy_path}")
    if policy_path.exists() and (not progress.exists()
                                  or json.loads(progress.read_text())["sha256"] != sha256(policy_path)):
        raise ValueError(f"Incomplete or corrupt policy checkpoint: {output}")
    started = time.monotonic()
    bars, values, splits, normalizer = prepare_portfolio(
        bundle, manifest, allow_partial_news=allow_partial_news, allow_partial_research=allow_partial_research)
    if min(len(part) for part in splits.values()) < 3:
        raise ValueError("Too few rows in portfolio split")
    def make_env():
        return SharedPortfolioEnv(bars.iloc[splits["train"]], values[splits["train"]], config, training=True)
    env = DummyVecEnv([make_env for _ in range(config.envs)])
    if policy_path.exists():
        model = MaskablePPO.load(policy_path, env=env, device=config.device)
    else:
        feature_count = len(normalizer["features"])
        news = [i for i, name in enumerate(normalizer["features"]) if name.split(":", 1)[1].startswith("news_")]
        research = [i for i, name in enumerate(normalizer["features"])
                    if name.split(":", 1)[1].startswith("research_")]
        def window_indices(columns):
            return [row * values.shape[1] + column + mask_offset
                    for row in range(config.lookback) for column in columns for mask_offset in (0, feature_count)]
        model = MaskablePPO(
            LowPriorityNewsPolicy, env, device=config.device, seed=config.seed,
            n_steps=config.rollout, batch_size=config.batch, n_epochs=config.epochs,
            learning_rate=config.learning_rate, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005, target_kl=0.03,
            policy_kwargs={"net_arch": {"pi": list(config.hidden), "vf": list(config.hidden)},
                           "news_indices": window_indices(news), "news_logit_limit": config.news_logit_limit,
                           "research_indices": window_indices(research),
                           "research_logit_limit": config.research_logit_limit, "ortho_init": False})
    if evaluate_test and model.num_timesteps < config.steps:
        raise ValueError("Complete training before evaluating the test period")
    if not evaluate_test:
        while model.num_timesteps < config.steps:
            steps = min(65_536, config.steps - model.num_timesteps)
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
            temporary = output / "policy.tmp.zip"
            model.save(temporary)
            temporary.replace(policy_path)
            save_json(progress, {"timesteps": model.num_timesteps, "sha256": sha256(policy_path)})
            print(f"Portfolio: {model.num_timesteps}/{config.steps}", flush=True)
        save_json(output / "normalization.json", normalizer)
    selected = splits["test" if evaluate_test else "validation"]
    def eval_env():
        return SharedPortfolioEnv(bars.iloc[selected], values[selected], config)
    result = {"status": "complete", "split": "test" if evaluate_test else "validation",
              "symbols": list(PORTFOLIO_SYMBOLS), "ppo": evaluate_portfolio(eval_env(), model),
              "buy_hold": evaluate_portfolio(eval_env(), buy_hold=True),
              "cash": evaluate_portfolio(eval_env()), "elapsed_seconds": time.monotonic() - started,
              "timesteps": model.num_timesteps, "policy_sha256": sha256(policy_path),
              "split_rows": {name: len(rows) for name, rows in splits.items()},
              "news_complete": manifest["news_complete"],
              "research_included": "research.parquet" in manifest["files"]}
    save_json(output / ("test-summary.json" if evaluate_test else "validation-summary.json"), result)
    env.close()
    return result
