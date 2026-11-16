from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypedDict

from .control import _replace_with_retry
from strategies import STRATEGIES


_INTRADAY_INTERVALS = ("1m", "5m", "15m", "20m", "30m", "1h")
_SWING_INTERVALS = ("1d", "2d", "trading_week")
_MODEL_STAGES = ("xgboost", "lstm", "hybrid_calibration")


class BatchGroup(TypedDict):
    style: str
    interval: str
    strategies: list[str]
    input_files: list[str]
    job_count: int


class TrainingJob(TypedDict):
    universe: str
    style: str
    interval: str
    strategy: str
    inputs: list[str]
    run_dir: str


class TrainingBatchPlan(TypedDict):
    format_version: int
    created_at: str
    status: str
    training_started: bool
    stock_data_dir: str
    run_root: str
    strategy_catalog_note: str
    symbol_pooling: str
    execution_warning: str
    model_stages: list[str]
    execution_order: str
    groups: list[BatchGroup]
    input_file_count: int
    strategy_count: int
    interval_count: int
    job_count: int
    model_fit_count: int
    hybrid_calibration_count: int
    total_steps: int


def build_training_batch_plan(
    *,
    stock_data_dir: str | Path,
    run_root: str | Path,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: list[str] | tuple[str, ...] | None = None,
) -> TrainingBatchPlan:
    data_root = Path(stock_data_dir).resolve()
    model_root = Path(run_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Stock data directory does not exist: {data_root}")

    requested_symbols = (
        {symbol.strip().upper() for symbol in symbols if symbol.strip()}
        if symbols is not None
        else None
    )
    if requested_symbols is not None and not requested_symbols:
        raise ValueError("At least one training symbol must be selected")
    requested_intervals = set(intervals) if intervals is not None else None
    supported_intervals = set(_INTRADAY_INTERVALS + _SWING_INTERVALS)
    if requested_intervals is not None and (
        not requested_intervals or requested_intervals - supported_intervals
    ):
        raise ValueError("Select at least one supported training interval")

    groups: list[BatchGroup] = []
    for style, split, intervals in (
        ("intraday", "minute", _INTRADAY_INTERVALS),
        ("swing", "day", _SWING_INTERVALS),
    ):
        strategies = list(STRATEGIES)
        for interval in intervals:
            if requested_intervals is not None and interval not in requested_intervals:
                continue
            split_root = data_root / split
            if not split_root.is_dir():
                continue
            input_paths = sorted(
                (symbol_dir / f"{interval}.csv.gz").resolve()
                for symbol_dir in split_root.iterdir()
                if symbol_dir.is_dir()
                and (symbol_dir / f"{interval}.csv.gz").is_file()
                if requested_symbols is None
                or symbol_dir.name.upper() in requested_symbols
            )
            if not input_paths:
                continue
            groups.append(
                {
                    "style": style,
                    "interval": interval,
                    "strategies": strategies.copy(),
                    "input_files": [str(path) for path in input_paths],
                    "job_count": len(strategies),
                }
            )

    job_count = sum(group["job_count"] for group in groups)
    if job_count == 0:
        raise ValueError(
            "No supported training CSV files found for the selected "
            f"symbols under {data_root}"
        )
    return {
        "format_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "training_started": False,
        "stock_data_dir": str(data_root),
        "run_root": str(model_root),
        "strategy_catalog_note": (
            "Catalog setups are omitted because they repeat shared templates. "
            "Only the four distinct implemented strategy families are included; "
            "they are candidates for out-of-sample evaluation, not proven winners."
        ),
        "symbol_pooling": (
            "All matching symbol inputs are pooled into one model per strategy "
            "and interval; symbols are not represented as a model dimension."
        ),
        "execution_warning": (
            "Pooled event arrays are cached on disk and LSTM batches are loaded "
            "on demand. A single input CSV is still feature-engineered in memory; "
            "check available disk space and run a representative pilot first."
        ),
        "input_file_count": sum(len(group["input_files"]) for group in groups),
        "model_stages": list(_MODEL_STAGES),
        "execution_order": (
            "job-major: build/cache one pooled dataset, train XGBoost and LSTM, "
            "calibrate the hybrid, then remove that job's dataset cache"
        ),
        "groups": groups,
        "strategy_count": len(STRATEGIES),
        "interval_count": len({group["interval"] for group in groups}),
        "job_count": job_count,
        "model_fit_count": job_count * 2,
        "hybrid_calibration_count": job_count,
        "total_steps": job_count * len(_MODEL_STAGES),
    }


def iter_planned_jobs(
    plan: TrainingBatchPlan,
    *,
    start_index: int = 0,
) -> Iterator[TrainingJob]:
    if start_index < 0 or start_index > plan["job_count"]:
        raise ValueError("start_index must be within the planned job range")
    if (
        sum(group["job_count"] for group in plan["groups"]) != plan["job_count"]
        or any(
            group["job_count"] != len(group["strategies"])
            for group in plan["groups"]
        )
    ):
        raise ValueError("Training batch groups do not match the declared job count")
    run_root = Path(plan["run_root"])
    remaining = start_index
    for group in plan["groups"]:
        if remaining >= group["job_count"]:
            remaining -= group["job_count"]
            continue
        if not group["strategies"]:
            raise ValueError("A training batch group must have strategies")
        inputs = group["input_files"]
        for strategy in group["strategies"][remaining:]:
            yield {
                "universe": "pooled",
                "style": group["style"],
                "interval": group["interval"],
                "strategy": strategy,
                "inputs": inputs.copy(),
                "run_dir": str(
                    run_root
                    / group["interval"]
                    / strategy
                    / "pooled"
                ),
            }
        remaining = 0


def iter_planned_steps(
    plan: TrainingBatchPlan,
) -> Iterator[dict[str, Any]]:
    for stage in plan["model_stages"]:
        for job in iter_planned_jobs(plan):
            yield {"stage": stage, **job}


def write_training_batch_plan(
    plan: TrainingBatchPlan,
    *,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(
            f"Training batch plan already exists: {output}. Choose a new path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(plan, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output
