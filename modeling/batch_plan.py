from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TypedDict

from .control import _replace_with_retry
from strategies import CATALOG_STRATEGIES, STRATEGIES, strategies_for_style


_INTRADAY_INTERVALS = ("1m", "5m", "15m", "20m", "30m", "1h")
_SWING_INTERVALS = ("1d", "2d", "trading_week")
_MODEL_STAGES = ("xgboost", "lstm", "hybrid_calibration")


class BatchGroup(TypedDict):
    style: str
    interval: str
    strategies: list[str]
    symbols: list[str]
    job_count: int


class TrainingBatchPlan(TypedDict):
    format_version: int
    created_at: str
    status: str
    training_started: bool
    stock_data_dir: str
    run_root: str
    strategy_catalog_note: str
    model_stages: list[str]
    execution_order: str
    groups: list[BatchGroup]
    symbols: list[str]
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
) -> TrainingBatchPlan:
    data_root = Path(stock_data_dir).resolve()
    model_root = Path(run_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Stock data directory does not exist: {data_root}")

    symbol_names = sorted(
        {
            directory.name
            for split in ("minute", "day")
            for directory in (data_root / split).glob("*")
            if directory.is_dir()
        }
    )
    if not symbol_names:
        raise ValueError(f"No per-symbol stock data directories found under {data_root}")

    groups: list[BatchGroup] = []
    strategies_by_style = {
        "intraday": list(STRATEGIES) + list(strategies_for_style("intraday")),
        "swing": list(STRATEGIES) + list(strategies_for_style("swing")),
    }
    for style, split, intervals in (
        ("intraday", "minute", _INTRADAY_INTERVALS),
        ("swing", "day", _SWING_INTERVALS),
    ):
        strategies = strategies_by_style[style]
        for interval in intervals:
            symbols = sorted(
                symbol
                for symbol in symbol_names
                if (data_root / split / symbol / f"{interval}.csv.gz").is_file()
            )
            if not symbols:
                continue
            groups.append(
                {
                    "style": style,
                    "interval": interval,
                    "strategies": strategies.copy(),
                    "symbols": symbols,
                    "job_count": len(symbols) * len(strategies),
                }
            )

    job_count = sum(group["job_count"] for group in groups)
    if job_count == 0:
        raise ValueError(f"No supported training CSV files found under {data_root}")
    return {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "training_started": False,
        "stock_data_dir": str(data_root),
        "run_root": str(model_root),
        "strategy_catalog_note": (
            "Catalog setups use documented shared family templates; these are "
            "research hypotheses, not independently validated implementations."
        ),
        "model_stages": list(_MODEL_STAGES),
        "execution_order": (
            "stage-major: finish every XGBoost job, then every LSTM job, then "
            "calibrate each XGBoost/LSTM hybrid"
        ),
        "groups": groups,
        "symbols": symbol_names,
        "strategy_count": len(STRATEGIES) + len(CATALOG_STRATEGIES),
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
) -> Iterator[dict[str, str]]:
    if start_index < 0 or start_index > plan["job_count"]:
        raise ValueError("start_index must be within the planned job range")
    if sum(group["job_count"] for group in plan["groups"]) != plan["job_count"]:
        raise ValueError("Training batch groups do not match the declared job count")
    data_root = Path(plan["stock_data_dir"])
    run_root = Path(plan["run_root"])
    remaining = start_index
    for group in plan["groups"]:
        if remaining >= group["job_count"]:
            remaining -= group["job_count"]
            continue
        split = "minute" if group["style"] == "intraday" else "day"
        strategy_count = len(group["strategies"])
        if strategy_count == 0:
            raise ValueError("A training batch group must have strategies")
        symbol_index, strategy_index = divmod(remaining, strategy_count)
        remaining = 0
        for symbol_offset, symbol in enumerate(
            group["symbols"][symbol_index:],
            start=symbol_index,
        ):
            input_path = data_root / split / symbol / f"{group['interval']}.csv.gz"
            first_strategy = strategy_index if symbol_offset == symbol_index else 0
            for strategy in group["strategies"][first_strategy:]:
                yield {
                    "symbol": symbol,
                    "style": group["style"],
                    "interval": group["interval"],
                    "strategy": strategy,
                    "input": str(input_path),
                    "run_dir": str(
                        run_root
                        / group["interval"]
                        / strategy
                        / symbol
                    ),
                }


def iter_planned_steps(
    plan: TrainingBatchPlan,
) -> Iterator[dict[str, str]]:
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
