from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd
import polars as pl

from backtesting.execution_simulator import ExecutionCosts
from backtesting.rl_execution import (
    Action,
    ExecutionConfig,
    Observation,
    _short_exit_cutoff,
    simulate_episode,
)
from features.feature_engineering import build_features
from strategies import generate_strategy_signals
from .batch_trainer import _write_atomic
from .hybrid import DEFAULT_MINIMUM_WIN_PROBABILITY, predict_hybrid_signals
from .rl_training import _training_lock


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _implementation_identity() -> dict[str, Any]:
    project = Path(__file__).resolve().parent.parent
    sources = {}
    for package in ("backtesting", "features", "modeling", "strategies"):
        for path in sorted((project / package).rglob("*.py")):
            sources[path.relative_to(project).as_posix()] = _sha256(path)
    packages = {}
    for name in ("backtrader", "numpy", "pandas", "polars", "torch", "xgboost",
                 "scikit-learn", "joblib", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"sources": sources, "packages": packages}


def _discover_jobs(model_roots: list[Path]) -> list[dict[str, Any]]:
    jobs = []
    seen = set()
    for root in model_roots:
        manifest_path = root / "batch-manifest.json"
        state_path = root / "batch-state.json"
        if not manifest_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"Batch manifest or state missing in {root}")
        state = _read_json(state_path)
        if state.get("status") not in {"completed", "completed_with_failures"}:
            raise ValueError(f"Batch run is not complete: {root}")
        manifest = _read_json(manifest_path)
        settings = manifest["configuration"]
        costs = ExecutionCosts(
            commission_bps=settings["commission_bps"],
            other_fees_bps=settings["other_fees_bps"],
            slippage_bps=settings["slippage_bps"],
        )
        metrics_files = sorted(root.rglob("hybrid-batch-metrics.json"))
        expected_models = state["completed_jobs_by_stage"]["hybrid_calibration"]
        if len(metrics_files) != expected_models:
            raise ValueError(
                f"Completed model count does not match batch state in {root}: "
                f"found {len(metrics_files)}, expected {expected_models}"
            )
        for metrics_path in metrics_files:
            model_dir = metrics_path.parent
            batch_job = _read_json(model_dir / "batch-job.json")
            metadata = _read_json(model_dir / "metadata.json")
            interval = batch_job["interval"]
            strategy = batch_job["strategy"]
            key = (interval, strategy)
            if key in seen:
                raise ValueError(f"Duplicate completed model for {interval} {strategy}")
            seen.add(key)
            if (
                batch_job["style"] != metadata["style"]
                or strategy != metadata["strategy"]
                or batch_job["horizon"] != metadata["horizon"]
            ):
                raise ValueError(f"Model metadata does not match batch job: {model_dir}")
            raw_paths = batch_job["input_fingerprints"]
            inputs = []
            seen_symbols = set()
            for entry in raw_paths:
                original = entry["path"]
                source = PureWindowsPath(original) if "\\" in original else Path(original)
                symbol = source.parent.name
                if (
                    not symbol
                    or symbol in {".", ".."}
                    or source.name.lower() != f"{interval}.csv.gz".lower()
                    or symbol.casefold() in seen_symbols
                ):
                    raise ValueError(f"Invalid or duplicate training input: {original}")
                seen_symbols.add(symbol.casefold())
                inputs.append({"symbol": symbol, "size": entry["size"]})
            if not inputs:
                raise ValueError(f"No recorded model inputs in {model_dir}")
            model_files = ["batch-job.json", "metadata.json", "hybrid.json", "lstm.pt"]
            weights = _read_json(model_dir / "hybrid.json")["model_weights"]
            if weights.get("xgboost", 0) > 0:
                model_files.append("xgboost.ubj")
            for name in weights:
                if name not in {"xgboost", "lstm"} and weights[name] > 0:
                    model_files.append(f"{name}.joblib")
            jobs.append({
                "interval": interval,
                "strategy": strategy,
                "style": metadata["style"],
                "horizon": metadata["horizon"],
                "test_start": metadata["test_start"],
                "model_dir": model_dir,
                "model_hashes": {name: _sha256(model_dir / name) for name in model_files},
                "costs": costs,
                "inputs": inputs,
            })
    if not jobs:
        raise ValueError("No completed hybrid models were found")
    return sorted(jobs, key=lambda job: (job["interval"], job["strategy"]))


def _make_policy(
    *,
    bars: pd.DataFrame,
    candidates: list[dict[str, Any]],
    config: ExecutionConfig,
    horizon: int,
    mode: str,
) -> tuple[Any, dict[str, float]]:
    choices: dict[datetime, str] = {}
    for row in candidates:
        probability = row["win_probability"]
        if probability is None:
            continue
        if mode == "new":
            accepted = probability >= DEFAULT_MINIMUM_WIN_PROBABILITY
        elif mode == "legacy":
            accepted = (
                probability >= 0.45
                and row["expected_return_r"] > 0
                and row["uncertainty"] <= 0.95
            )
        elif mode == "no_trade":
            accepted = False
        else:
            raise ValueError(f"Unsupported comparison mode: {mode}")
        if accepted:
            stamp = (
                pd.Timestamp(row["timestamp"]).tz_localize("UTC")
                .tz_convert("Asia/Kolkata").tz_localize(None).to_pydatetime()
            )
            if stamp in choices:
                raise ValueError(f"Multiple accepted signals at {stamp}")
            choices[stamp] = row["direction"]
    indices = {stamp.to_pydatetime(): i for i, stamp in enumerate(bars.index)}
    entered_at: int | None = None
    stats = {"peak": config.initial_cash, "max_drawdown": 0.0}

    def policy(observation: Observation) -> Action:
        nonlocal entered_at
        stats["peak"] = max(stats["peak"], observation.equity)
        stats["max_drawdown"] = max(
            stats["max_drawdown"],
            (stats["peak"] - observation.equity) / stats["peak"],
        )
        index = indices[observation.timestamp]
        if observation.position:
            if index == len(bars) - 1:
                return "hold"
            session_end = (
                config.style == "intraday"
                and observation.timestamp.time() >= _short_exit_cutoff(config.interval)
            )
            if session_end or (entered_at is not None and index - entered_at >= horizon):
                return "close_long" if observation.position > 0 else "cover_short"
            return "hold"
        direction = choices.get(observation.timestamp)
        if direction is None or index >= len(bars) - 2:
            return "wait"
        if config.style == "intraday":
            if observation.timestamp.time() >= _short_exit_cutoff(config.interval):
                return "wait"
        elif direction == "short":
            return "wait"
        if observation.equity * config.exposure_fraction < observation.close:
            return "wait"
        entered_at = index
        return "open_long" if direction == "long" else "open_short"

    return policy, stats


def _summarize_result(
    result: Any, stats: dict[str, float], *, intraday: bool,
) -> dict[str, Any]:
    pending = None
    trade_pnls = []
    for execution in result.executions:
        if intraday and execution.submitted_at.date() != execution.executed_at.date():
            raise ValueError("An order crossed an exchange session; the simulation is invalid")
        if execution.action in {"open_long", "open_short"}:
            if pending is not None:
                raise ValueError("Multiple open executions in one simulated position")
            pending = execution
        else:
            if pending is None or execution.size != -pending.size:
                raise ValueError("Mismatched position executions in simulator output")
            trade_pnls.append(
                (execution.price - pending.price) * pending.size
                - pending.commission - execution.commission
            )
            pending = None
    if (pending is None) != (result.final_position == 0):
        raise ValueError("Open position does not match simulator executions")
    stats["peak"] = max(stats["peak"], result.peak_equity)
    stats["max_drawdown"] = max(
        stats["max_drawdown"],
        (stats["peak"] - result.final_equity) / stats["peak"],
    )
    max_drawdown_percent = -100 * stats["max_drawdown"]
    return {
        **result.formatted_summary(),
        "max_drawdown_percent": (
            f"{max_drawdown_percent:.2f}" if max_drawdown_percent else "0.00"
        ),
        "closed_trades": len(trade_pnls),
        "winning_trades": sum(value > 0 for value in trade_pnls),
        "losing_trades": sum(value < 0 for value in trade_pnls),
        "flat_trades": sum(value == 0 for value in trade_pnls),
        "order_count": len(result.executions),
        "drawdown_triggered": result.drawdown_triggered,
    }


def _evaluate_symbol(
    *,
    job: dict[str, Any],
    input_path: Path,
    score_batch_size: int,
) -> dict[str, Any]:
    from typing import cast
    from modeling.data import ModelStyle
    from strategies._types import StrategyName, StrategyStyle

    frame = build_features(input_path, style=cast(ModelStyle, job["style"])).collect()
    signals = generate_strategy_signals(
        frame,
        strategy=cast(StrategyName, job["strategy"]),
        style=cast(StrategyStyle, job["style"]),
        symbol=input_path.parent.name,
    )
    boundary = datetime.fromisoformat(job["test_start"])
    if boundary.tzinfo is None:
        raise ValueError(f"Model test boundary has no timezone: {job['model_dir']}")
    boundary_utc = pd.Timestamp(boundary).tz_convert("UTC").tz_localize(None).to_pydatetime()
    signals = signals.filter(pl.col("timestamp") >= boundary_utc)
    candidates: list[dict[str, Any]] = []
    for offset in range(0, signals.height, score_batch_size):
        scores = predict_hybrid_signals(
            frame,
            signals.slice(offset, score_batch_size),
            model_dir=job["model_dir"],
            include_all_candidates=True,
        )
        candidates.extend(scores.iter_rows(named=True))
    bar_frame = (
        frame.filter(pl.col("timestamp") >= boundary_utc)
        .select("timestamp", "open", "high", "low", "close", "volume")
        .to_pandas()
    )
    if len(bar_frame) < 2:
        return {
            "status": "insufficient_test_bars",
            "test_bar_count": len(bar_frame),
            "candidate_count": len(candidates),
        }
    bar_frame["timestamp"] = (
        pd.to_datetime(bar_frame["timestamp"], utc=True)
        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    )
    bars = bar_frame.set_index("timestamp")
    config = ExecutionConfig(
        style=cast(ModelStyle, job["style"]),
        interval=job["interval"],
        initial_cash=50_000.0,
        exposure_fraction=0.1,
        max_drawdown=0.5,
        costs=job["costs"],
    )
    scenarios = {}
    for mode in ("no_trade", "legacy", "new"):
        policy, stats = _make_policy(
            bars=bars,
            candidates=candidates,
            config=config,
            horizon=job["horizon"],
            mode=mode,
        )
        result = simulate_episode(bars, config=config, policy=policy)
        scenarios[mode] = _summarize_result(
            result, stats, intraday=config.style == "intraday",
        )
    return {
        "status": "evaluated",
        "test_bar_count": len(bars),
        "candidate_count": len(candidates),
        "unscorable_candidates": sum(row["win_probability"] is None for row in candidates),
        "scenarios": scenarios,
    }


def compare_hybrid_models(
    *,
    model_roots: list[str | Path],
    stock_data_root: str | Path,
    output_root: str | Path,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    score_batch_size: int = 1024,
    resume: bool = False,
) -> dict[str, Any]:
    if score_batch_size < 1:
        raise ValueError("score_batch_size must be positive")
    roots = [Path(root).resolve() for root in model_roots]
    data_root = Path(stock_data_root).resolve()
    output = Path(output_root).resolve()
    if not roots or any(not root.is_dir() for root in roots) or not data_root.is_dir():
        raise FileNotFoundError("All model roots and the stock data root must exist")
    if any(_paths_overlap(output, path) for path in [*roots, data_root]):
        raise ValueError("Output must be separate from model and stock data directories")
    selected_symbols = {value.upper() for value in symbols} if symbols else None
    selected_intervals = set(intervals) if intervals else None
    jobs = [
        job for job in _discover_jobs(roots)
        if selected_intervals is None or job["interval"] in selected_intervals
    ]
    if not jobs:
        raise ValueError("No completed models match the selected intervals")
    entries = []
    for job in jobs:
        split = "minute" if job["style"] == "intraday" else "day"
        for item in job["inputs"]:
            if selected_symbols is not None and item["symbol"].upper() not in selected_symbols:
                continue
            input_path = data_root / split / item["symbol"] / f"{job['interval']}.csv.gz"
            if not input_path.is_file():
                raise FileNotFoundError(f"Recorded training input is missing: {input_path}")
            if input_path.stat().st_size != item["size"]:
                raise ValueError(f"Input size differs from the training record: {input_path}")
            entries.append((job, item["symbol"], input_path))
    if not entries:
        raise ValueError("No model inputs match the selected symbols")
    if not output.exists():
        if resume:
            raise FileNotFoundError(f"Cannot resume missing output directory: {output}")
        output.mkdir(parents=True)
    if not output.is_dir():
        raise NotADirectoryError(output)

    with _training_lock(output, lock_name="comparison.lock"):
        run_path = output / "comparison-run.json"
        identity = {
            "format_version": 1,
            "model_roots": [str(root) for root in roots],
            "data_root": str(data_root),
            "symbols": sorted(selected_symbols) if selected_symbols else None,
            "intervals": sorted(selected_intervals) if selected_intervals else None,
            "score_batch_size": score_batch_size,
            "implementation": _implementation_identity(),
            "jobs": [
                {
                    "interval": job["interval"],
                    "strategy": job["strategy"],
                    "model_dir": str(job["model_dir"]),
                    "model_hashes": job["model_hashes"],
                }
                for job in jobs
            ],
        }
        if run_path.exists():
            if not resume:
                raise FileExistsError(f"Comparison already exists; use --resume: {run_path}")
            run = _read_json(run_path)
            if run.get("identity") != identity:
                raise ValueError("Comparison configuration, implementation or model artifacts changed")
            run["status"] = "in_progress"
            _write_atomic(run_path, run)
            _write_atomic(output / "comparison-summary.json", {"status": "in_progress"})
        else:
            if resume:
                raise FileNotFoundError(f"Comparison manifest missing: {run_path}")
            if any(path.name != "comparison.lock" for path in output.iterdir()):
                raise FileExistsError(f"Comparison output directory is not empty: {output}")
            run = {"format_version": 1, "status": "in_progress", "identity": identity}
            _write_atomic(run_path, run)

        rows = []
        for index, (job, symbol, input_path) in enumerate(entries, start=1):
            result_path = output / "symbols" / job["interval"] / job["strategy"] / f"{symbol}.json"
            input_identity = _file_identity(input_path)
            if result_path.exists():
                saved = _read_json(result_path)
                if saved.get("input_identity") != input_identity:
                    raise ValueError(f"Input changed after comparison: {input_path}")
                result = saved["result"]
            else:
                result = _evaluate_symbol(
                    job=job, input_path=input_path, score_batch_size=score_batch_size,
                )
                if _file_identity(input_path) != input_identity:
                    raise RuntimeError(f"Input changed during comparison: {input_path}")
                _write_atomic(result_path, {
                    "format_version": 1, "input_identity": input_identity, "result": result,
                })
            rows.append({
                "interval": job["interval"],
                "strategy": job["strategy"],
                "symbol": symbol,
                **result,
            })
            if index % 100 == 0 or index == len(entries):
                print(f"Compared {index:,}/{len(entries):,} symbol-model episodes", flush=True)

        evaluated = [row for row in rows if row["status"] == "evaluated"]

        def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
            totals = {}
            for mode in ("no_trade", "legacy", "new"):
                pnl = sum(float(row["scenarios"][mode]["profit_loss"]) for row in group)
                capital = sum(
                    float(row["scenarios"][mode]["initial_capital"]) for row in group
                )
                closed = sum(row["scenarios"][mode]["closed_trades"] for row in group)
                wins = sum(row["scenarios"][mode]["winning_trades"] for row in group)
                totals[mode] = {
                    "profit_loss": f"{pnl:.2f}",
                    "profit_loss_percent": f"{100 * pnl / capital:.2f}" if capital else None,
                    "initial_capital": f"{capital:.2f}",
                    "closed_trades": closed,
                    "winning_trades": wins,
                    "losing_trades": sum(
                        row["scenarios"][mode]["losing_trades"] for row in group
                    ),
                    "win_rate_percent": f"{100 * wins / closed:.2f}" if closed else None,
                }
            return totals

        totals = summarize(evaluated)
        by_job = []
        for job in jobs:
            group = [
                row for row in evaluated
                if row["interval"] == job["interval"] and row["strategy"] == job["strategy"]
            ]
            by_job.append({
                "interval": job["interval"],
                "strategy": job["strategy"],
                "evaluated_count": len(group),
                "totals": summarize(group),
            })
        summary = {
            "format_version": 1,
            "status": "complete",
            "episode_count": len(rows),
            "evaluated_count": len(evaluated),
            "insufficient_test_bars": len(rows) - len(evaluated),
            "totals": totals,
            "by_timeframe_strategy": by_job,
            "limitations": [
                "Retrospective test period has already been inspected; this is not a fresh forward test.",
                "Totals sum independent symbol simulations, not a shared-capital portfolio.",
                "Input size was checked against training metadata, but size alone does not establish file identity.",
                "Signals use next-open orders and a horizon/session exit, not the original stop/target label simulator.",
                "Shorts are only simulated intraday; open positions are marked to final equity.",
            ],
        }
        _write_atomic(output / "comparison-summary.json", summary)
        run["status"] = "complete"
        _write_atomic(run_path, run)
        return summary
