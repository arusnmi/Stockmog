from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .rl_execution import (
    Action,
    ExecutionConfig,
    EpisodeResult,
    Observation,
    _short_exit_cutoff,
    simulate_episode,
)

ACTION_NAMES: tuple[str, ...] = (
    "wait",
    "hold",
    "buy",
    "sell",
    "open_short",
    "cover_short",
)
_EXECUTION_ACTIONS: dict[int, Action] = {
    0: "wait",
    1: "hold",
    2: "open_long",
    3: "close_long",
    4: "open_short",
    5: "cover_short",
}
_SCORE_COLUMNS = (
    "loss_probability",
    "neutral_probability",
    "win_probability",
    "expected_return_r",
    "uncertainty",
)


def _local_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index
    return index.tz_convert("Asia/Kolkata").tz_localize(None)


def prepare_fold_episode(
    bars: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    fold: dict[str, Any],
    fold_index: int,
    policy_holdout: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one exact score fold and the matching bars, preserving its holdout flag."""
    if "timestamp" not in scores.columns:
        raise ValueError("Score rows must contain timestamps")
    if "walk_forward_fold" not in scores.columns or "policy_holdout" not in scores.columns:
        raise ValueError("Score rows must identify their fold and policy holdout status")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.empty:
        raise ValueError("Bars must have a nonempty DatetimeIndex")
    start_text = fold.get("prediction_start_utc")
    end_text = fold.get("prediction_end_utc")
    if not isinstance(start_text, str):
        raise ValueError("Fold metadata is missing prediction_start_utc")
    start = pd.Timestamp(start_text)
    if start.tzinfo is None:
        raise ValueError("Fold boundaries must be timezone-aware UTC timestamps")
    start = start.tz_convert("Asia/Kolkata").tz_localize(None)
    end = None
    if end_text is not None:
        if not isinstance(end_text, str):
            raise ValueError("prediction_end_utc must be a timestamp or null")
        end = pd.Timestamp(end_text)
        if end.tzinfo is None:
            raise ValueError("Fold boundaries must be timezone-aware UTC timestamps")
        end = end.tz_convert("Asia/Kolkata").tz_localize(None)
        if end <= start:
            raise ValueError("Fold prediction end must be after its start")

    local_bars = bars.copy()
    local_bars.index = _local_index(local_bars.index)
    if local_bars.index.has_duplicates or not local_bars.index.is_monotonic_increasing:
        raise ValueError("Bar timestamps must be unique and strictly increasing")

    fold_scores = scores.loc[
        (scores["walk_forward_fold"] == fold_index)
        & (scores["policy_holdout"].astype(bool) == policy_holdout)
    ].copy()
    if not fold_scores.empty:
        if fold_scores["timestamp"].isna().any():
            raise ValueError("Score timestamps cannot be null")
        score_times = pd.DatetimeIndex(
            pd.to_datetime(fold_scores["timestamp"], utc=True)
        ).tz_convert("Asia/Kolkata").tz_localize(None)
        fold_scores.index = score_times
        if ((fold_scores.index < start).any()
                or (end is not None and (fold_scores.index >= end).any())):
            raise ValueError("Scores cross the selected fold's prediction boundaries")
        if fold_scores.index.has_duplicates:
            raise ValueError("Only one scored candidate per timestamp is supported")
    else:
        fold_scores.index = pd.DatetimeIndex([])

    if bool(fold.get("policy_holdout")) != policy_holdout:
        raise ValueError("Requested holdout mode does not match fold metadata")
    mask = local_bars.index >= start
    if end is not None:
        mask &= local_bars.index < end
    episode_bars = local_bars.loc[mask]
    if len(episode_bars) < 2:
        raise ValueError("Selected fold has fewer than two market bars")
    return episode_bars, fold_scores


class BacktraderTradingEnv(gym.Env[np.ndarray, int]):
    """Gymnasium adapter that blocks on policy actions while Backtrader advances bars."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        scores: pd.DataFrame,
        *,
        config: ExecutionConfig,
        lookback: int = 20,
        invalid_action_penalty: float = 0.001,
    ) -> None:
        super().__init__()
        if lookback < 1:
            raise ValueError("lookback must be at least one")
        if not np.isfinite(invalid_action_penalty) or invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be finite and nonnegative")
        required_bars = {"open", "high", "low", "close", "volume"}
        missing_bars = required_bars - set(bars.columns)
        if missing_bars:
            raise ValueError(f"Bars are missing columns: {', '.join(sorted(missing_bars))}")
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.empty:
            raise ValueError("Bars must have a nonempty DatetimeIndex")
        if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
            raise ValueError("Bar timestamps must be unique and strictly increasing")
        missing_scores = set(("timestamp", "direction", *_SCORE_COLUMNS)) - set(scores.columns)
        if missing_scores:
            raise ValueError(f"Scores are missing columns: {', '.join(sorted(missing_scores))}")
        if scores["timestamp"].isna().any():
            raise ValueError("Score timestamps cannot be null")
        score_times = pd.DatetimeIndex(
            pd.to_datetime(scores["timestamp"], utc=True)
        ).tz_convert("Asia/Kolkata").tz_localize(None)
        score_frame = scores.copy()
        score_frame.index = score_times
        if score_frame.index.has_duplicates:
            raise ValueError("Only one scored candidate per timestamp is supported")
        if {"walk_forward_fold", "policy_holdout"}.issubset(score_frame.columns):
            if (
                score_frame["walk_forward_fold"].nunique(dropna=False) != 1
                or score_frame["policy_holdout"].nunique(dropna=False) != 1
            ):
                raise ValueError("An RL episode cannot mix score folds or holdout status")
        if not score_frame["direction"].isin(("long", "short")).all():
            raise ValueError("Candidate directions must be long or short")
        self.bars = bars.copy()
        self.bars.index = _local_index(self.bars.index)
        if self.bars.index.has_duplicates or not self.bars.index.is_monotonic_increasing:
            raise ValueError("Bar timestamps must be unique and strictly increasing")
        self.scores = score_frame
        self.config = config
        self.lookback = lookback
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self._bar_features = self._make_bar_features()
        self._score_features = self._make_score_features()
        self._observation_size = lookback * 5 + 14 + 5
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self._observation_size,),
            dtype=np.float32,
        )
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._actions: queue.Queue[int | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._index = 0
        self._previous_equity = config.initial_cash
        self._last_observation: Observation | None = None
        self._last_result: EpisodeResult | None = None

    def _make_bar_features(self) -> np.ndarray:
        values = self.bars[["open", "high", "low", "close", "volume"]].to_numpy(
            dtype=np.float64
        )
        if (
            not np.isfinite(values).all()
            or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
        ):
            raise ValueError("Bars contain non-finite or invalid prices or volume")
        closes = values[:, 3]
        volume_median = pd.Series(values[:, 4]).rolling(
            20, min_periods=1
        ).median().to_numpy()
        previous_close = np.concatenate(([closes[0]], closes[:-1]))
        return np.column_stack(
            (
                values[:, 0] / closes - 1.0,
                values[:, 1] / closes - 1.0,
                values[:, 2] / closes - 1.0,
                closes / previous_close - 1.0,
                np.log1p(values[:, 4] / np.maximum(volume_median, 1.0)),
            )
        ).astype(np.float32)

    def _make_score_features(self) -> np.ndarray:
        result = np.zeros((len(self.bars), 14), dtype=np.float32)
        bar_positions = {timestamp: index for index, timestamp in enumerate(self.bars.index)}
        for timestamp, row in self.scores.iterrows():
            position = bar_positions.get(timestamp)
            if position is None:
                raise ValueError(f"Score timestamp {timestamp} has no matching market bar")
            direction_index = 0 if row["direction"] == "long" else 1
            offset = direction_index * 7
            result[position, offset] = 1.0
            raw = np.asarray([row[name] for name in _SCORE_COLUMNS], dtype=np.float64)
            if not np.isfinite(raw).all():
                continue
            probabilities = raw[:3]
            if (
                (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-4)
                or raw[4] < 0
            ):
                raise ValueError(f"Invalid model scores at {timestamp}")
            result[position, offset + 1] = 1.0
            result[position, offset + 2 : offset + 5] = probabilities
            result[position, offset + 5] = np.clip(raw[3] / 5.0, -1.0, 1.0)
            result[position, offset + 6] = np.clip(raw[4], 0.0, 1.0)
        return result

    def _observation_vector(self, observation: Observation, index: int) -> np.ndarray:
        start = max(0, index - self.lookback + 1)
        history = self._bar_features[start : index + 1]
        if len(history) < self.lookback:
            history = np.pad(history, ((self.lookback - len(history), 0), (0, 0)))
        score_features = self._score_features[index]
        equity_ratio = observation.equity / self.config.initial_cash
        drawdown = max(
            0.0,
            (observation.peak_equity - observation.equity)
            / max(observation.peak_equity, 1e-12),
        )
        local_time = observation.timestamp.hour * 60 + observation.timestamp.minute
        angle = 2.0 * np.pi * (local_time - 9 * 60 - 15) / 375.0
        position_exposure = (
            observation.position * observation.close / max(observation.equity, 1e-12)
        )
        state = np.asarray(
            [
                np.clip(position_exposure, -1.0, 1.0),
                np.clip(equity_ratio - 1.0, -1.0, 5.0),
                np.clip(drawdown, 0.0, 1.0),
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )
        vector = np.concatenate((history.ravel(), score_features, state)).astype(
            np.float32
        )
        if vector.shape != (self._observation_size,) or not np.isfinite(vector).all():
            raise RuntimeError("The RL observation has an invalid shape or value")
        return np.clip(vector, self.observation_space.low, self.observation_space.high)

    def _action_mask(self, observation: Observation, *, last_bar: bool = False) -> np.ndarray:
        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        if observation.position == 0:
            mask[0] = True
            if not last_bar:
                mask[2] = True
                if (
                    self.config.style == "intraday"
                    and self._index < len(self.bars) - 2
                    and observation.timestamp.time() < _short_exit_cutoff(self.config.interval)
                ):
                    mask[4] = True
        elif observation.position > 0:
            mask[1] = True
            mask[3] = not last_bar
        else:
            mask[1] = True
            mask[5] = not last_bar
        return mask

    def action_masks(self) -> np.ndarray:
        if self._last_observation is None:
            raise RuntimeError("Call reset before requesting an action mask")
        return self._action_mask(
            self._last_observation,
            last_bar=self._index >= len(self.bars) - 1,
        )

    def _policy(self, observation: Observation) -> Action:
        if self._closed:
            raise RuntimeError("Environment was closed during an episode")
        self._events.put(("observation", observation))
        action = self._actions.get()
        if action is None:
            raise RuntimeError("Environment episode was cancelled")
        last_bar = self._index >= len(self.bars) - 1
        mask = self._action_mask(observation, last_bar=last_bar)
        if mask[action]:
            return _EXECUTION_ACTIONS[action]
        return "hold" if observation.position else "wait"

    def _run_episode(self) -> None:
        try:
            result = simulate_episode(
                self.bars,
                config=self.config,
                policy=self._policy,
            )
        except Exception as exc:
            self._events.put(("error", exc))
        else:
            self._events.put(("finished", result))

    def _next_event(self) -> tuple[str, Any]:
        kind, payload = self._events.get()
        if kind == "error":
            raise RuntimeError("Backtrader RL episode failed") from payload
        return kind, payload

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._worker is not None and self._worker.is_alive():
            self._actions.put(None)
            self._worker.join(timeout=10)
            if self._worker.is_alive():
                raise RuntimeError("The previous Backtrader episode did not stop for reset")
        if len(self.bars) < 2:
            raise ValueError("An episode requires at least two market bars")
        self._events = queue.Queue()
        self._actions = queue.Queue()
        self._worker = None
        self._started = True
        self._closed = False
        self._index = 0
        self._previous_equity = self.config.initial_cash
        self._last_observation = None
        self._last_result = None
        self._worker = threading.Thread(target=self._run_episode, daemon=True)
        self._worker.start()
        kind, payload = self._next_event()
        if kind != "observation":
            raise RuntimeError("Backtrader ended before its first market observation")
        self._last_observation = payload
        self._previous_equity = payload.equity
        return self._observation_vector(payload, 0), {
            "timestamp": payload.timestamp,
            "equity": payload.equity,
            "position": payload.position,
            "action_mask": self.action_masks(),
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._started or self._last_observation is None:
            raise RuntimeError("Call reset before step")
        if self._worker is None or not self._worker.is_alive():
            raise RuntimeError("The episode has already finished")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in [0, {len(ACTION_NAMES) - 1}]")
        observation = self._last_observation
        mask = self.action_masks()
        invalid = not bool(mask[action])
        self._actions.put(int(action))
        kind, payload = self._next_event()
        if kind == "observation":
            self._index += 1
            next_observation: Observation = payload
            reward = (
                next_observation.equity - self._previous_equity
            ) / self.config.initial_cash
            self._previous_equity = next_observation.equity
            self._last_observation = next_observation
            done = False
            vector = self._observation_vector(next_observation, self._index)
            info = {
                "timestamp": next_observation.timestamp,
                "equity": next_observation.equity,
                "position": next_observation.position,
                "action_mask": self.action_masks(),
                "requested_action": ACTION_NAMES[action],
                "invalid_action": invalid,
            }
        elif kind == "finished":
            self._last_result = payload
            reward = (payload.final_equity - self._previous_equity) / self.config.initial_cash
            done = True
            vector = self._observation_vector(observation, self._index)
            info = {
                **payload.formatted_summary(),
                "drawdown_triggered": payload.drawdown_triggered,
                "requested_action": ACTION_NAMES[action],
                "invalid_action": invalid,
            }
        else:
            raise RuntimeError(f"Unexpected Backtrader episode event: {kind}")
        if invalid:
            reward -= self.invalid_action_penalty
        return vector, float(reward), done, False, info

    def close(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._closed = True
            return
        self._closed = True
        self._actions.put(None)
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            raise RuntimeError("Backtrader worker did not stop after environment close")
