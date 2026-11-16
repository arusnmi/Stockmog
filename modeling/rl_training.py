from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator
from typing import Any

import pandas as pd

from backtesting.execution_simulator import ExecutionCosts
from backtesting.rl_env import BacktraderTradingEnv, prepare_fold_episode
from backtesting.rl_execution import ExecutionConfig
from .batch_trainer import _write_atomic
from .control import TrainingControl, _replace_with_retry


@dataclass(frozen=True)
class RLTrainingConfig:
    timesteps_per_episode: int = 2048
    rollout_steps: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    seed: int = 42
    lookback: int = 20
    initial_cash: float = 50_000.0
    exposure_fraction: float = 0.1
    max_drawdown: float = 0.5
    invalid_action_penalty: float = 0.001
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "timesteps_per_episode", "rollout_steps", "batch_size", "lookback",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.timesteps_per_episode % self.rollout_steps:
            raise ValueError("timesteps_per_episode must be a multiple of rollout_steps")
        if self.batch_size > self.rollout_steps:
            raise ValueError("batch_size cannot exceed rollout_steps")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if (
            not isfinite(self.exposure_fraction)
            or not 0 < self.exposure_fraction <= 1
        ):
            raise ValueError("exposure_fraction must be between zero and one")
        if not isfinite(self.max_drawdown) or not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be between zero and one")
        if (
            not isfinite(self.invalid_action_penalty)
            or self.invalid_action_penalty < 0
        ):
            raise ValueError("invalid_action_penalty must be finite and nonnegative")
        if not self.device:
            raise ValueError("device cannot be empty")


def _require_rl_dependencies() -> tuple[Any, Any]:
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise RuntimeError(
            "RL training requires the optional packages in requirements-rl.txt."
        ) from exc
    return MaskablePPO, set_random_seed


@contextmanager
def _training_lock(root: Path, *, lock_name: str = "trainer.lock") -> Iterator[None]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("RL training requires psutil from requirements-training.txt") from exc
    lock_path = root / lock_name
    process = psutil.Process()
    lock_data = {
        "pid": os.getpid(),
        "create_time": process.create_time(),
    }
    for _ in range(2):
        try:
            with lock_path.open("x", encoding="utf-8") as lock_file:
                json.dump(lock_data, lock_file)
                lock_file.write("\n")
            break
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                active = psutil.Process(int(existing["pid"]))
                is_active = abs(active.create_time() - float(existing["create_time"])) < 1e-3
            except psutil.NoSuchProcess:
                is_active = False
            except psutil.AccessDenied:
                is_active = True
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot determine whether RL lock is active: {lock_path}") from exc
            if is_active:
                raise RuntimeError(f"Another process is active: {lock_path}")
            try:
                latest = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"RL lock changed while being inspected: {lock_path}") from exc
            if latest != existing:
                raise RuntimeError(f"RL lock changed while being inspected: {lock_path}")
            lock_path.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"Could not acquire RL training lock: {lock_path}")
    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == lock_data:
            lock_path.unlink(missing_ok=True)


def _read_bars(path: Path, expected_symbol_directory: str) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    header = pd.read_csv(path, compression="infer", nrows=0)
    if not required.issubset(header.columns):
        raise ValueError(f"Training CSV is missing required columns: {path}")
    usecols = sorted(required | ({"symbol"} if "symbol" in header.columns else set()))
    frame = pd.read_csv(path, compression="infer", usecols=usecols)
    if "symbol" in frame.columns:
        symbols = frame["symbol"].dropna().astype(str).unique().tolist()
        if len(symbols) != 1:
            raise ValueError(f"Expected a single symbol in {path}")
        from data_pipeline.huggingface_stocks import _safe_symbol

        if _safe_symbol(symbols[0]).casefold() != expected_symbol_directory.casefold():
            raise ValueError(f"Symbol in {path} does not match its directory")
        frame.drop(columns=["symbol"], inplace=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame.set_index("timestamp", inplace=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Bar timestamps must be unique and increasing: {path}")
    return frame


def _score_job(root: Path, interval: str, strategy: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    job_dir = root / interval / strategy
    fold_path = job_dir / "folds.json"
    scores_path = job_dir / "scores.json"
    if not fold_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(f"Walk-forward fold or score manifest is missing under {job_dir}")
    folds_payload = json.loads(fold_path.read_text(encoding="utf-8"))
    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    if folds_payload.get("format_version") != 1 or scores_payload.get("format_version") != 1:
        raise ValueError(f"Unsupported score manifest under {job_dir}")
    folds = folds_payload.get("folds")
    inputs = scores_payload.get("inputs")
    if not isinstance(folds, list) or not folds or not isinstance(inputs, list) or not inputs:
        raise ValueError(f"Score manifests are empty under {job_dir}")
    if sum(bool(fold.get("policy_holdout")) for fold in folds) != 1:
        raise ValueError(f"Expected exactly one policy holdout fold under {job_dir}")
    input_entries = []
    seen_symbols = set()
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid score input entry under {job_dir}")
        symbol = item.get("symbol")
        symbol_directory = item.get("symbol_directory")
        bar_path = Path(str(item.get("input", ""))).resolve()
        scores_path_item = (job_dir / str(item.get("scores", ""))).resolve()
        try:
            scores_path_item.relative_to(job_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Score file path escapes its job directory: {scores_path_item}") from exc
        if (
            not isinstance(symbol, str)
            or not isinstance(symbol_directory, str)
            or not bar_path.is_file()
            or not scores_path_item.is_file()
            or not isinstance(item.get("input_fingerprint"), dict)
        ):
            raise ValueError(f"Invalid or missing symbol score files under {job_dir}")
        if symbol.casefold() in seen_symbols:
            raise ValueError(f"Duplicate symbol score entry: {symbol}")
        seen_symbols.add(symbol.casefold())
        input_entries.append({
            "symbol": symbol,
            "symbol_directory": symbol_directory,
            "bars": str(bar_path),
            "scores": str(scores_path_item),
            "input_fingerprint": item["input_fingerprint"],
            "score_fingerprint": _file_fingerprint(scores_path_item),
        })
    return job_dir, folds_payload, input_entries


def _episode_key(fold_index: int, symbol: str) -> str:
    return f"{fold_index}:{symbol}"


def _write_model_checkpoint(model: Any, model_path: Path) -> None:
    temporary = model_path.with_name(f"{model_path.stem}.tmp.zip")
    try:
        model.save(str(temporary))
        if not temporary.is_file():
            raise RuntimeError(f"RL library did not write checkpoint: {temporary}")
        _replace_with_retry(temporary, model_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _episode_data(
    *,
    item: dict[str, Any],
    folds: list[dict[str, Any]],
    fold_index: int,
    interval: str,
    strategy: str,
    config: RLTrainingConfig,
    costs: ExecutionCosts,
    bars: pd.DataFrame,
    score_rows: pd.DataFrame,
    policy_holdout: bool = False,
) -> BacktraderTradingEnv:
    bar_path = Path(item["bars"])
    fingerprint = item["input_fingerprint"]
    stat = bar_path.stat()
    if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != fingerprint:
        raise RuntimeError(f"Training bars changed since score generation: {bar_path}")
    scores_path = Path(item["scores"])
    if _file_fingerprint(scores_path) != item["score_fingerprint"]:
        raise RuntimeError(f"Walk-forward scores changed during RL training: {scores_path}")
    if "walk_forward_fold" not in score_rows or "policy_holdout" not in score_rows:
        raise ValueError(f"Score file has no fold provenance: {scores_path}")
    fold = folds[fold_index]
    if bool(fold.get("policy_holdout")) != policy_holdout:
        raise ValueError("Fold holdout status does not match the requested episode mode")
    episode_bars, episode_scores = prepare_fold_episode(
        bars,
        score_rows,
        fold=fold,
        fold_index=fold_index,
        policy_holdout=policy_holdout,
    )
    style = "intraday" if interval in {"1m", "5m", "15m", "20m", "30m", "1h"} else "swing"
    return BacktraderTradingEnv(
        episode_bars,
        episode_scores,
        config=ExecutionConfig(
            style=style,
            interval=interval,
            initial_cash=config.initial_cash,
            exposure_fraction=config.exposure_fraction,
            max_drawdown=config.max_drawdown,
            costs=costs,
        ),
        lookback=config.lookback,
        invalid_action_penalty=config.invalid_action_penalty,
    )


def _file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _train_job(
    *,
    job_dir: Path,
    job_output: Path,
    intervals: str,
    strategy: str,
    folds_payload: dict[str, Any],
    inputs: list[dict[str, Any]],
    config: RLTrainingConfig,
    costs: ExecutionCosts,
    control: TrainingControl,
    resume: bool,
    seed_random: Any,
    ppo_type: Any,
) -> str:
    folds = folds_payload["folds"]
    episodes = [
        _episode_key(index, item["symbol"])
        for item in inputs
        for index, fold in enumerate(folds)
        if not fold.get("policy_holdout")
    ]
    if not episodes:
        raise ValueError(f"No non-holdout episodes available for {job_dir}")
    job_output.mkdir(parents=True, exist_ok=True)
    model_path = job_output / "agent.zip"
    state_path = job_output / "progress.json"
    identity = {
        "job": str(job_dir.resolve()),
        "configuration": asdict(config),
        "costs": asdict(costs),
        "folds": folds,
        "episodes": episodes,
        "input_files": inputs,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"RL progress exists; use --resume: {state_path}")
        if state.get("identity") != identity:
            raise ValueError(f"RL job inputs or configuration changed: {job_output}")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume without RL progress: {state_path}")
        state = {
            "format_version": 1,
            "status": "in_progress",
            "identity": identity,
            "completed_episodes": [],
            "trained_timesteps": 0,
        }
        _write_atomic(state_path, state)

    completed = set(state["completed_episodes"])
    unexpected = completed - set(episodes)
    if unexpected:
        raise ValueError(f"RL progress contains unknown episodes: {sorted(unexpected)}")
    if completed and not model_path.is_file():
        raise FileNotFoundError(f"RL checkpoint is missing: {model_path}")

    model = None
    active_env = None
    try:
        for item in inputs:
            bar_path = Path(item["bars"])
            if _file_fingerprint(bar_path) != item["input_fingerprint"]:
                raise RuntimeError(f"Training bars changed since score generation: {bar_path}")
            bars = _read_bars(bar_path, item["symbol_directory"])
            score_path = Path(item["scores"])
            if _file_fingerprint(score_path) != item["score_fingerprint"]:
                raise RuntimeError(f"Walk-forward scores changed: {score_path}")
            score_rows = pd.read_parquet(score_path)
            for fold_index, fold in enumerate(folds):
                if fold.get("policy_holdout"):
                    continue
                episode = _episode_key(fold_index, item["symbol"])
                if episode in completed:
                    continue
                next_env = _episode_data(
                    item=item,
                    folds=folds,
                    fold_index=fold_index,
                    interval=intervals,
                    strategy=strategy,
                    config=config,
                    costs=costs,
                    bars=bars,
                    score_rows=score_rows,
                )
                if active_env is not None:
                    active_env.close()
                seed_random(config.seed + len(completed))
                if model is None:
                    if completed:
                        model = ppo_type.load(
                            str(model_path), env=next_env, device=config.device
                        )
                    else:
                        model = ppo_type(
                            "MlpPolicy",
                            next_env,
                            n_steps=config.rollout_steps,
                            batch_size=config.batch_size,
                            learning_rate=config.learning_rate,
                            gamma=config.gamma,
                            seed=config.seed,
                            device=config.device,
                            verbose=1,
                            policy_kwargs={"net_arch": [128, 128]},
                        )
                else:
                    model.set_env(next_env)
                active_env = next_env
                model.learn(
                    total_timesteps=config.timesteps_per_episode,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                _write_model_checkpoint(model, model_path)
                completed.add(episode)
                state["completed_episodes"] = [
                    key for key in episodes if key in completed
                ]
                state["trained_timesteps"] = int(model.num_timesteps)
                _write_atomic(state_path, state)
                print(
                    f"Trained {strategy} {intervals}: {len(completed)}/{len(episodes)} "
                    f"symbol-fold episodes ({item['symbol']}, fold {fold_index})",
                    flush=True,
                )
                if not control.checkpoint_boundary(
                    lambda: _write_model_checkpoint(model, model_path)
                ):
                    state["status"] = "stopped"
                    _write_atomic(state_path, state)
                    return "stopped"
    finally:
        if active_env is not None:
            active_env.close()
    if model is None:
        model = ppo_type.load(str(model_path), device=config.device)
    state["status"] = "complete"
    _write_atomic(state_path, state)
    return "complete"


def _train_rl_policies_locked(
    *,
    scores_root: str | Path,
    output_root: str | Path,
    config: RLTrainingConfig | None = None,
    costs: ExecutionCosts | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train masked PPO policies on all non-holdout symbol/fold episodes."""
    config = config or RLTrainingConfig()
    costs = costs or ExecutionCosts()
    ppo_type, seed_random = _require_rl_dependencies()
    score_root = Path(scores_root).resolve()
    score_manifest_path = score_root / "score-run.json"
    if not score_manifest_path.is_file():
        raise FileNotFoundError(f"Walk-forward score run not found: {score_manifest_path}")
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("format_version") != 1 or score_manifest.get("status") != "complete":
        raise ValueError("Walk-forward scores must come from a complete score run")
    score_fingerprint = {
        "size": score_manifest_path.stat().st_size,
        "mtime_ns": score_manifest_path.stat().st_mtime_ns,
    }
    output = Path(output_root).resolve()
    if not output.exists():
        if resume:
            raise FileNotFoundError(f"Cannot resume missing RL output directory: {output}")
        output.mkdir(parents=True)
    elif not output.is_dir():
        raise NotADirectoryError(f"RL output path is not a directory: {output}")
    run_path = output / "rl-run.json"
    run_identity = {
        "format_version": 1,
        "scores_root": str(score_root),
        "score_manifest_fingerprint": score_fingerprint,
        "configuration": asdict(config),
        "costs": asdict(costs),
    }
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"RL run exists; pass --resume: {run_path}")
        if run.get("identity") != run_identity:
            raise ValueError("RL run settings or score manifest changed")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume without run manifest: {run_path}")
        run = {
            "format_version": 1,
            "status": "in_progress",
            "identity": run_identity,
            "completed_jobs": [],
        }
        _write_atomic(run_path, run)

    jobs = []
    for interval_dir in sorted(score_root.iterdir()):
        if not interval_dir.is_dir() or interval_dir.name.startswith("."):
            continue
        for strategy_dir in sorted(interval_dir.iterdir()):
            if strategy_dir.is_dir():
                jobs.append((interval_dir.name, strategy_dir.name))
    if not jobs:
        raise ValueError(f"No scored policy jobs found under {score_root}")
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    completed_jobs = set(run["completed_jobs"])
    for interval, strategy in jobs:
        job_key = f"{interval}/{strategy}"
        if job_key in completed_jobs:
            continue
        job_dir, folds_payload, inputs = _score_job(score_root, interval, strategy)
        status = _train_job(
            job_dir=job_dir,
            job_output=output / interval / strategy,
            intervals=interval,
            strategy=strategy,
            folds_payload=folds_payload,
            inputs=inputs,
            config=config,
            costs=costs,
            control=control,
            resume=resume,
            seed_random=seed_random,
            ppo_type=ppo_type,
        )
        if status == "stopped":
            run["status"] = "stopped"
            _write_atomic(run_path, run)
            return run
        completed_jobs.add(job_key)
        run["completed_jobs"] = sorted(completed_jobs)
        _write_atomic(run_path, run)
    run["status"] = "complete"
    _write_atomic(run_path, run)
    return run


def train_rl_policies(
    *,
    scores_root: str | Path,
    output_root: str | Path,
    config: RLTrainingConfig | None = None,
    costs: ExecutionCosts | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if not output.exists():
        if resume:
            raise FileNotFoundError(
                f"Cannot resume missing RL output directory: {output}"
            )
        output.mkdir(parents=True)
    if not output.is_dir():
        raise NotADirectoryError(f"RL output path is not a directory: {output}")
    with _training_lock(output):
        return _train_rl_policies_locked(
            scores_root=scores_root,
            output_root=output,
            config=config,
            costs=costs,
            resume=resume,
        )
