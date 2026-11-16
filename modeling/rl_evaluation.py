from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.execution_simulator import ExecutionCosts
from .batch_trainer import _write_atomic
from .control import _replace_with_retry
from .rl_training import (
    RLTrainingConfig,
    _episode_data,
    _file_fingerprint,
    _require_rl_dependencies,
    _read_bars,
    _score_job,
    _training_lock,
)

_RESULT_COLUMNS = (
    "interval",
    "strategy",
    "symbol",
    "holdout_fold",
    "holdout_start_utc",
    "holdout_end_utc",
    "initial_capital",
    "final_equity",
    "profit_loss",
    "profit_loss_percent",
    "peak_equity",
    "open_position_shares",
    "drawdown_triggered",
)


def _paths_overlap(first: Path, second: Path) -> bool:
    for parent, child in ((first, second), (second, first)):
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            pass
    return False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_evaluation_context(
    *,
    scores_root: Path,
    policies_root: Path,
) -> tuple[dict[str, Any], RLTrainingConfig, ExecutionCosts, list[dict[str, Any]]]:
    score_manifest_path = scores_root / "score-run.json"
    policy_manifest_path = policies_root / "rl-run.json"
    if not score_manifest_path.is_file():
        raise FileNotFoundError(f"Walk-forward score run not found: {score_manifest_path}")
    if not policy_manifest_path.is_file():
        raise FileNotFoundError(f"RL policy run not found: {policy_manifest_path}")

    score_manifest = _read_json(score_manifest_path)
    if (
        score_manifest.get("format_version") != 1
        or score_manifest.get("status") != "complete"
    ):
        raise ValueError("Evaluation requires a complete walk-forward score run")
    policy_run = _read_json(policy_manifest_path)
    if policy_run.get("format_version") != 1 or policy_run.get("status") != "complete":
        raise ValueError("Evaluation requires a complete RL policy run")
    run_identity = policy_run.get("identity")
    if not isinstance(run_identity, dict):
        raise ValueError(f"RL policy run has no valid identity: {policy_manifest_path}")
    if Path(str(run_identity.get("scores_root", ""))).resolve() != scores_root:
        raise ValueError("The policy run was trained from a different scores directory")

    score_fingerprint = _file_fingerprint(score_manifest_path)
    if run_identity.get("score_manifest_fingerprint") != score_fingerprint:
        raise RuntimeError("The walk-forward score manifest changed after policy training")
    config_data = run_identity.get("configuration")
    costs_data = run_identity.get("costs")
    if not isinstance(config_data, dict) or not isinstance(costs_data, dict):
        raise ValueError("The policy run has invalid training configuration or costs")
    config = RLTrainingConfig(**config_data)
    costs = ExecutionCosts(**costs_data)

    jobs = []
    for interval_dir in sorted(scores_root.iterdir()):
        if not interval_dir.is_dir() or interval_dir.name.startswith("."):
            continue
        for strategy_dir in sorted(interval_dir.iterdir()):
            if strategy_dir.is_dir():
                jobs.append((interval_dir.name, strategy_dir.name))
    expected_job_keys = {f"{interval}/{strategy}" for interval, strategy in jobs}
    completed_job_keys = policy_run.get("completed_jobs")
    if (
        not isinstance(completed_job_keys, list)
        or any(not isinstance(key, str) for key in completed_job_keys)
        or len(set(completed_job_keys)) != len(completed_job_keys)
        or set(completed_job_keys) != expected_job_keys
    ):
        raise ValueError("The completed policy jobs do not match the score jobs")
    if not jobs:
        raise ValueError(f"No scored policy jobs found under {scores_root}")

    evaluation_jobs = []
    job_fingerprints = []
    for interval, strategy in jobs:
        job_key = f"{interval}/{strategy}"
        job_dir, folds_payload, inputs = _score_job(scores_root, interval, strategy)
        folds = folds_payload["folds"]
        holdout_indexes = [
            index for index, fold in enumerate(folds) if fold.get("policy_holdout")
        ]
        if len(holdout_indexes) != 1:
            raise ValueError(f"Expected exactly one policy holdout fold in {job_dir}")

        policy_job_dir = policies_root / interval / strategy
        model_path = policy_job_dir / "agent.zip"
        progress_path = policy_job_dir / "progress.json"
        if not model_path.is_file() or not progress_path.is_file():
            raise FileNotFoundError(f"Completed policy artifacts are missing under {policy_job_dir}")
        progress = _read_json(progress_path)
        progress_identity = progress.get("identity")
        if (
            progress.get("format_version") != 1
            or not isinstance(progress_identity, dict)
            or progress.get("status") != "complete"
        ):
            raise ValueError(f"Policy job is not complete: {policy_job_dir}")

        expected_training_episodes = [
            f"{index}:{item['symbol']}"
            for item in inputs
            for index, fold in enumerate(folds)
            if not fold.get("policy_holdout")
        ]
        if (
            progress_identity.get("job") != str(job_dir.resolve())
            or progress_identity.get("configuration") != config_data
            or progress_identity.get("costs") != costs_data
            or progress_identity.get("folds") != folds
            or progress_identity.get("input_files") != inputs
            or progress_identity.get("episodes") != expected_training_episodes
            or progress.get("completed_episodes") != expected_training_episodes
        ):
            raise ValueError(f"Policy progress does not match score inputs: {policy_job_dir}")

        entry = {
            "interval": interval,
            "strategy": strategy,
            "job_key": job_key,
            "job_dir": job_dir,
            "policy_job_dir": policy_job_dir,
            "model_path": model_path,
            "folds_payload": folds_payload,
            "folds": folds,
            "holdout_index": holdout_indexes[0],
            "inputs": inputs,
        }
        evaluation_jobs.append(entry)
        job_fingerprints.append({
            "job_key": job_key,
            "folds": _file_fingerprint(job_dir / "folds.json"),
            "scores": _file_fingerprint(job_dir / "scores.json"),
            "agent": _file_fingerprint(model_path),
            "progress": _file_fingerprint(progress_path),
            "inputs": [
                {
                    "symbol": item["symbol"],
                    "bars": item["bars"],
                    "input_fingerprint": item["input_fingerprint"],
                    "scores": item["scores"],
                    "score_fingerprint": item["score_fingerprint"],
                }
                for item in inputs
            ],
        })

    identity = {
        "format_version": 1,
        "scores_root": str(scores_root),
        "policies_root": str(policies_root),
        "score_manifest_fingerprint": score_fingerprint,
        "policy_manifest_fingerprint": _file_fingerprint(policy_manifest_path),
        "configuration": config_data,
        "costs": costs_data,
        "jobs": job_fingerprints,
    }
    return identity, config, costs, evaluation_jobs


def _episode_identity(
    *,
    job: dict[str, Any],
    input_index: int,
    item: dict[str, Any],
    config: RLTrainingConfig,
    costs: ExecutionCosts,
) -> dict[str, Any]:
    fold_index = job["holdout_index"]
    return {
        "job_key": job["job_key"],
        "symbol": item["symbol"],
        "input_index": input_index,
        "fold_index": fold_index,
        "fold": job["folds"][fold_index],
        "input_fingerprint": item["input_fingerprint"],
        "score_fingerprint": item["score_fingerprint"],
        "agent_fingerprint": _file_fingerprint(job["model_path"]),
        "configuration": config.__dict__,
        "costs": costs.__dict__,
    }


def _validate_episode_result(result: Any, path: Path) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError(f"Invalid holdout result in {path}")
    for field in (
        "initial_capital",
        "final_equity",
        "profit_loss",
        "profit_loss_percent",
        "peak_equity",
        "open_position_shares",
    ):
        value = result.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field} in holdout result {path}") from exc
        if not isfinite(number):
            raise ValueError(f"Non-finite {field} in holdout result {path}")
        if not isinstance(value, str) or f"{number:.2f}" != value:
            raise ValueError(f"{field} must be a two-decimal string in {path}")
    if not isinstance(result.get("drawdown_triggered"), bool):
        raise ValueError(f"Invalid drawdown status in holdout result {path}")
    initial = float(result["initial_capital"])
    profit = float(result["profit_loss"])
    expected_profit = float(result["final_equity"]) - initial
    expected_percent = 100 * expected_profit / initial
    if initial <= 0 or abs(profit - expected_profit) > 0.011:
        raise ValueError(f"Inconsistent capital and P/L values in {path}")
    if abs(float(result["profit_loss_percent"]) - expected_percent) > 0.011:
        raise ValueError(f"Inconsistent P/L percentage in {path}")
    return result


def _run_episode(model: Any, env: Any, *, symbol: str) -> dict[str, Any]:
    observation, _ = env.reset()
    final_info: dict[str, Any] | None = None
    for _ in range(len(env.bars) + 1):
        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise RuntimeError(f"Policy returned a non-scalar action for {symbol}")
        observation, _, terminated, truncated, info = env.step(int(action_array.item()))
        if terminated or truncated:
            final_info = info
            break
    if final_info is None:
        raise RuntimeError(f"Policy did not finish the holdout episode for {symbol}")
    result = _validate_episode_result(final_info, Path(symbol))
    result["drawdown_triggered"] = final_info["drawdown_triggered"]
    return result


def _result_path(output_root: Path, job: dict[str, Any], input_index: int) -> Path:
    return (
        output_root
        / "episodes"
        / job["interval"]
        / job["strategy"]
        / f"{input_index:05d}.json"
    )


def _evaluate_job(
    *,
    job: dict[str, Any],
    output_root: Path,
    config: RLTrainingConfig,
    costs: ExecutionCosts,
    ppo_type: Any,
    resume: bool,
) -> None:
    result_dir = output_root / "episodes" / job["interval"] / job["strategy"]
    expected_names = {f"{index:05d}.json" for index in range(len(job["inputs"]))}
    if result_dir.exists():
        unexpected = {
            path.name for path in result_dir.glob("*.json")
        } - expected_names
        if unexpected:
            raise ValueError(f"Unexpected holdout result files under {result_dir}: {sorted(unexpected)}")

    pending = []
    for input_index, item in enumerate(job["inputs"]):
        bars_path = Path(item["bars"])
        scores_path = Path(item["scores"])
        if _file_fingerprint(bars_path) != item["input_fingerprint"]:
            raise RuntimeError(f"Training bars changed since score generation: {bars_path}")
        if _file_fingerprint(scores_path) != item["score_fingerprint"]:
            raise RuntimeError(f"Walk-forward scores changed: {scores_path}")
        result_path = _result_path(output_root, job, input_index)
        episode_identity = _episode_identity(
            job=job,
            input_index=input_index,
            item=item,
            config=config,
            costs=costs,
        )
        if result_path.exists():
            if not resume:
                raise FileExistsError(f"Holdout result already exists: {result_path}")
            saved = _read_json(result_path)
            if saved.get("identity") != episode_identity:
                raise ValueError(f"Saved holdout result no longer matches its inputs: {result_path}")
            _validate_episode_result(saved.get("result"), result_path)
        else:
            pending.append((input_index, item, episode_identity, result_path))

    if not pending:
        return

    model = ppo_type.load(str(job["model_path"]), device=config.device)
    for input_index, item, episode_identity, result_path in pending:
        bars = _read_bars(Path(item["bars"]), item["symbol_directory"])
        score_rows = pd.read_parquet(item["scores"])
        env = _episode_data(
            item=item,
            folds=job["folds"],
            fold_index=job["holdout_index"],
            interval=job["interval"],
            strategy=job["strategy"],
            config=config,
            costs=costs,
            bars=bars,
            score_rows=score_rows,
            policy_holdout=True,
        )
        try:
            result = _run_episode(model, env, symbol=item["symbol"])
            if float(result["initial_capital"]) != config.initial_cash:
                raise RuntimeError(f"Unexpected initial capital in holdout episode: {item['symbol']}")
            row = {
                "interval": job["interval"],
                "strategy": job["strategy"],
                "symbol": item["symbol"],
                "holdout_fold": job["holdout_index"],
                "holdout_start_utc": job["folds"][job["holdout_index"]][
                    "prediction_start_utc"
                ],
                "holdout_end_utc": job["folds"][job["holdout_index"]][
                    "prediction_end_utc"
                ],
                **result,
            }
        finally:
            env.close()
        _write_atomic(result_path, {
            "format_version": 1,
            "identity": episode_identity,
            "result": row,
        })
        print(
            f"Evaluated {job['interval']} {job['strategy']} "
            f"{item['symbol']} ({input_index + 1}/{len(job['inputs'])})",
            flush=True,
        )


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No holdout episodes were evaluated")
    initial = sum(float(row["initial_capital"]) for row in rows)
    final = sum(float(row["final_equity"]) for row in rows)
    profit = sum(float(row["profit_loss"]) for row in rows)

    def summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        group_initial = sum(float(row["initial_capital"]) for row in group)
        group_final = sum(float(row["final_equity"]) for row in group)
        group_profit = sum(float(row["profit_loss"]) for row in group)
        return {
            "episode_count": len(group),
            "initial_capital": f"{group_initial:.2f}",
            "final_equity": f"{group_final:.2f}",
            "profit_loss": f"{group_profit:.2f}",
            "profit_loss_percent": (
                f"{100 * group_profit / group_initial:.2f}"
                if group_initial
                else "0.00"
            ),
            "profitable_episodes": sum(float(row["profit_loss"]) > 0 for row in group),
            "loss_episodes": sum(float(row["profit_loss"]) < 0 for row in group),
            "flat_episodes": sum(float(row["profit_loss"]) == 0 for row in group),
            "drawdown_liquidations": sum(row["drawdown_triggered"] for row in group),
            "open_position_episodes": sum(
                float(row["open_position_shares"]) != 0 for row in group
            ),
        }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["interval"], row["strategy"]), []).append(row)
    by_job = [
        {"interval": interval, "strategy": strategy, **summary(group)}
        for (interval, strategy), group in sorted(grouped.items())
    ]
    overall = {
        **summary(rows),
        "initial_capital": f"{initial:.2f}",
        "final_equity": f"{final:.2f}",
        "profit_loss": f"{profit:.2f}",
        "profit_loss_percent": f"{100 * profit / initial:.2f}" if initial else "0.00",
    }
    return overall, by_job


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        pd.DataFrame(rows, columns=_RESULT_COLUMNS).to_csv(temporary, index=False)
        _replace_with_retry(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _evaluate_rl_policies_locked(
    *,
    scores_root: str | Path,
    policies_root: str | Path,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    score_root = Path(scores_root).resolve()
    policy_root = Path(policies_root).resolve()
    if _paths_overlap(output_root, score_root) or _paths_overlap(output_root, policy_root):
        raise ValueError("Evaluation output must be separate from score and policy directories")
    identity, config, costs, jobs = _load_evaluation_context(
        scores_root=score_root,
        policies_root=policy_root,
    )
    run_path = output_root / "evaluation-run.json"
    if run_path.exists():
        if not resume:
            raise FileExistsError(f"Evaluation run exists; pass --resume: {run_path}")
        run = _read_json(run_path)
        if run.get("identity") != identity:
            raise ValueError("Evaluation inputs or policy artifacts changed")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume without evaluation manifest: {run_path}")
        existing = [path for path in output_root.iterdir() if path.name != "evaluation.lock"]
        if existing:
            raise FileExistsError(f"Evaluation output directory is not empty: {output_root}")
        run = {
            "format_version": 1,
            "status": "in_progress",
            "identity": identity,
            "completed_jobs": [],
        }
        _write_atomic(run_path, run)

    ppo_type, _ = _require_rl_dependencies()
    completed_jobs = set(run.get("completed_jobs", []))
    expected_jobs = {job["job_key"] for job in jobs}
    if completed_jobs - expected_jobs:
        raise ValueError("Evaluation progress contains unknown jobs")
    for job in jobs:
        _evaluate_job(
            job=job,
            output_root=output_root,
            config=config,
            costs=costs,
            ppo_type=ppo_type,
            resume=resume,
        )
        completed_jobs.add(job["job_key"])
        run["completed_jobs"] = sorted(completed_jobs)
        _write_atomic(run_path, run)

    rows = []
    for job in jobs:
        for input_index, _ in enumerate(job["inputs"]):
            result_path = _result_path(output_root, job, input_index)
            saved = _read_json(result_path)
            expected_identity = _episode_identity(
                job=job,
                input_index=input_index,
                item=job["inputs"][input_index],
                config=config,
                costs=costs,
            )
            if saved.get("identity") != expected_identity:
                raise ValueError(f"Saved holdout result no longer matches its inputs: {result_path}")
            result = _validate_episode_result(saved.get("result"), result_path)
            rows.append(saved["result"])
            if result["drawdown_triggered"] != saved["result"]["drawdown_triggered"]:
                raise ValueError(f"Holdout result changed while being read: {result_path}")

    overall, by_job = _aggregate(rows)
    summary = {
        "format_version": 1,
        "status": "complete",
        "scores_root": str(score_root),
        "policies_root": str(policy_root),
        "episode_count": len(rows),
        "aggregation_note": (
            "Totals sum independent simulations, each using the configured initial "
            "capital per symbol; they are not a capital-constrained portfolio backtest. "
            "Equity is marked to the final bar; any remaining long position is reported "
            "rather than closed without a subsequent bar."
        ),
        "overall": overall,
        "by_timeframe_strategy": by_job,
        "results_csv": str(output_root / "holdout-results.csv"),
    }
    _write_atomic(output_root / "holdout-summary.json", summary)
    _write_csv(output_root / "holdout-results.csv", rows)
    run["status"] = "complete"
    _write_atomic(run_path, run)
    return summary


def evaluate_rl_policies(
    *,
    scores_root: str | Path,
    policies_root: str | Path,
    output_root: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    score_root = Path(scores_root).resolve()
    policy_root = Path(policies_root).resolve()
    if _paths_overlap(output, score_root) or _paths_overlap(output, policy_root):
        raise ValueError("Evaluation output must be separate from score and policy directories")
    if not output.exists():
        if resume:
            raise FileNotFoundError(f"Cannot resume missing evaluation directory: {output}")
        output.mkdir(parents=True)
    if not output.is_dir():
        raise NotADirectoryError(f"Evaluation output path is not a directory: {output}")
    with _training_lock(output, lock_name="evaluation.lock"):
        return _evaluate_rl_policies_locked(
            scores_root=scores_root,
            policies_root=policies_root,
            output_root=output,
            resume=resume,
        )
