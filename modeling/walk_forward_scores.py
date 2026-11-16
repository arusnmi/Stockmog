from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from backtesting.execution_simulator import ExecutionCosts
from data_pipeline.huggingface_stocks import _safe_symbol
from features.feature_engineering import build_features
from strategies import STRATEGIES, generate_strategy_signals
from strategies._types import StrategyName, StrategyStyle

from .batch_plan import TrainingJob, iter_planned_jobs
from .batch_trainer import _calibrate_job_hybrid, _read_plan, _save_job_metadata, _write_atomic
from .control import TrainingControl, _replace_with_retry
from .data import ModelStyle, build_training_dataset
from .hybrid import DEFAULT_MINIMUM_WIN_PROBABILITY, predict_hybrid_signals
from .training import train_lstm, train_xgboost
from .walk_forward import WalkForwardFold, walk_forward_splits

_INTRADAY = {"1m", "5m", "15m", "20m", "30m", "1h"}
_SWING = {"1d", "2d", "trading_week"}


@dataclass(frozen=True)
class WalkForwardScoreConfig:
    folds: int = 3
    warmup_fraction: float = 0.5
    holdout_fraction: float = 0.15
    validation_fraction: float = 0.2
    sequence_length: int = 60
    event_stride: int = 1
    xgboost_rounds: int = 500
    xgboost_patience: int = 30
    lstm_epochs: int = 40
    lstm_patience: int = 8
    batch_size: int = 32
    seed: int = 42
    score_batch_size: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "folds", "sequence_length", "event_stride", "xgboost_rounds",
            "xgboost_patience", "lstm_epochs", "lstm_patience", "batch_size",
            "score_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if not 0 < self.warmup_fraction < 1 or not 0 < self.holdout_fraction < 1:
            raise ValueError("warmup and holdout fractions must be between zero and one")
        if self.warmup_fraction + self.holdout_fraction >= 1:
            raise ValueError("warmup and holdout fractions must leave a prediction period")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")


def _validate_job(job: TrainingJob) -> None:
    if job["strategy"] not in STRATEGIES:
        raise ValueError(f"Unsupported strategy in plan: {job['strategy']!r}")
    if (job["style"] == "intraday" and job["interval"] not in _INTRADAY) or (
        job["style"] == "swing" and job["interval"] not in _SWING
    ) or job["style"] not in ("intraday", "swing"):
        raise ValueError(f"Invalid style or interval in plan: {job['style']}, {job['interval']}")
    inputs = [str(Path(path).resolve()) for path in job["inputs"]]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("A scoring job must contain distinct input files")
    symbols = [Path(path).parent.name.upper() for path in inputs]
    if len(set(symbols)) != len(symbols):
        raise ValueError("A scoring job cannot contain duplicate symbol directories")


def _fold_manifest(fold: WalkForwardFold, index: int) -> dict[str, object]:
    return {
        "index": index,
        "validation_start_utc": fold.split.validation_start.isoformat(),
        "prediction_start_utc": fold.split.test_start.isoformat(),
        "prediction_end_utc": (
            fold.prediction_end.isoformat() if fold.prediction_end else None
        ),
        "policy_holdout": fold.policy_holdout,
        "training_events": len(fold.split.train),
        "validation_events": len(fold.split.validation),
        "prediction_events": len(fold.split.test),
    }


def _input_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _verify_input(path: Path, expected: dict[str, int]) -> None:
    if _input_fingerprint(path) != expected:
        raise RuntimeError(f"Input changed during walk-forward scoring: {path}")


def _write_symbol_scores(
    *,
    input_path: Path,
    job: TrainingJob,
    folds: tuple[WalkForwardFold, ...],
    fold_dirs: list[Path],
    destination: Path,
    config: WalkForwardScoreConfig,
) -> tuple[str, int]:
    if not folds or len(folds) != len(fold_dirs):
        raise ValueError("Each walk-forward fold must have exactly one model directory")
    frame = build_features(input_path, style=cast(ModelStyle, job["style"])).collect()
    if frame.schema.get("timestamp") != pl.Datetime("us"):
        raise ValueError("Feature timestamps must be naive UTC datetimes with microsecond precision")
    symbol_directory = input_path.parent.name
    symbol = symbol_directory
    if "symbol" in frame.columns:
        symbols = frame["symbol"].unique().to_list()
        if (
            len(symbols) != 1
            or symbols[0] is None
            or _safe_symbol(symbols[0]).casefold() != symbol_directory.casefold()
        ):
            raise ValueError(
                f"Input {input_path} does not contain one symbol matching its directory"
            )
        symbol = symbols[0]
    signals = generate_strategy_signals(
        frame,
        strategy=cast(StrategyName, job["strategy"]),
        style=cast(StrategyStyle, job["style"]),
        symbol=symbol,
    )
    timestamps = frame["timestamp"].cast(pl.Int64).to_numpy()
    if len(timestamps) > 1 and np.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"Input {input_path} has duplicate or unordered timestamps")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    writer = None
    row_count = 0
    try:
        for index, (fold, model_dir) in enumerate(zip(folds, fold_dirs)):
            predicate = pl.col("timestamp") >= fold.split.test_start.replace(tzinfo=None)
            if fold.prediction_end is not None:
                predicate &= pl.col("timestamp") < fold.prediction_end.replace(tzinfo=None)
            selected = signals.filter(predicate)
            for offset in range(0, selected.height, config.score_batch_size):
                candidates = selected.slice(offset, config.score_batch_size)
                candidate_times = candidates["timestamp"].cast(pl.Int64).to_numpy()
                positions = np.searchsorted(timestamps, candidate_times)
                if (
                    np.any(positions >= len(timestamps))
                    or not np.array_equal(timestamps[positions], candidate_times)
                ):
                    raise ValueError(f"Candidate timestamps do not match {input_path}")
                first = max(0, int(positions[0]) - config.sequence_length + 1)
                last = int(positions[-1])
                scored = predict_hybrid_signals(
                    frame.slice(first, last - first + 1),
                    candidates,
                    model_dir=model_dir,
                    include_all_candidates=True,
                ).with_columns(
                    pl.col("timestamp").dt.replace_time_zone("UTC"),
                    pl.lit(index, dtype=pl.Int32).alias("walk_forward_fold"),
                    pl.lit(fold.policy_holdout).alias("policy_holdout"),
                )
                if scored.height != candidates.height:
                    raise RuntimeError(f"Hybrid scoring omitted candidates for {input_path}")
                table = scored.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                row_count += len(scored)
        if writer is None:
            scored = predict_hybrid_signals(
                frame, signals.head(0), model_dir=fold_dirs[0],
                include_all_candidates=True,
            ).with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC"),
                pl.lit(0, dtype=pl.Int32).alias("walk_forward_fold"),
                pl.lit(False).alias("policy_holdout"),
            )
            table = scored.to_arrow()
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
        writer.close()
        writer = None
        _replace_with_retry(temporary, destination)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return symbol, row_count


def create_walk_forward_scores(
    *,
    plan_path: str | Path,
    output_root: str | Path,
    config: WalkForwardScoreConfig | None = None,
    costs: ExecutionCosts | None = None,
) -> None:
    """Fit chronological fold models and score every planned symbol's candidates."""
    config = config or WalkForwardScoreConfig()
    costs = costs or ExecutionCosts()
    plan_path = Path(plan_path).resolve()
    plan = _read_plan(plan_path)
    jobs = list(iter_planned_jobs(plan))
    if not jobs:
        raise ValueError("The training plan contains no jobs")
    for job in jobs:
        _validate_job(job)
    keys = [(job["interval"], job["strategy"]) for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Training plan contains duplicate timeframe/strategy jobs")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    control = TrainingControl(root)
    manifest_path = root / "score-run.json"
    manifest = {
        "format_version": 1,
        "status": "in_progress",
        "plan_path": str(plan_path),
        "configuration": vars(config),
        "costs": vars(costs),
        "timestamp_timezone": "UTC",
        "candidate_acceptance": {
            "minimum_win_probability": DEFAULT_MINIMUM_WIN_PROBABILITY,
            "require_return_and_low_uncertainty": False,
        },
        "job_count": len(jobs),
        "completed_jobs": 0,
    }
    _write_atomic(manifest_path, manifest)
    for job_index, planned_job in enumerate(jobs):
        job_dir = root / planned_job["interval"] / planned_job["strategy"]
        job = cast(TrainingJob, {**planned_job, "run_dir": str(job_dir)})
        job_dir.mkdir(parents=True)
        input_paths = [Path(path).resolve() for path in job["inputs"]]
        fingerprints = [_input_fingerprint(path) for path in input_paths]
        print(
            f"Walk-forward job {job_index + 1}/{len(jobs)}: "
            f"{job['interval']} {job['strategy']} ({len(job['inputs'])} symbols)",
            flush=True,
        )
        dataset = build_training_dataset(
            job["inputs"],
            style=cast(ModelStyle, job["style"]),
            strategy=job["strategy"],
            horizon=12 if job["style"] == "intraday" else 10,
            sequence_length=config.sequence_length,
            costs=costs,
            cache_dir=job_dir / "dataset-cache",
            event_stride=config.event_stride,
        )
        for path, fingerprint in zip(input_paths, fingerprints):
            _verify_input(path, fingerprint)
        folds = walk_forward_splits(
            dataset,
            folds=config.folds,
            warmup_fraction=config.warmup_fraction,
            holdout_fraction=config.holdout_fraction,
            validation_fraction=config.validation_fraction,
        )
        fold_dirs = []
        for index, fold in enumerate(folds):
            fold_dir = job_dir / "folds" / f"{index:02d}"
            fold_dir.mkdir(parents=True)
            fold_dirs.append(fold_dir)
            _save_job_metadata(run_dir=fold_dir, job=job, dataset=dataset, split=fold.split)
            print(f"Fitting fold {index + 1}/{len(folds)}", flush=True)
            xgboost_result = train_xgboost(
                dataset, fold.split, run_dir=fold_dir, control=control,
                max_rounds=config.xgboost_rounds, patience=config.xgboost_patience,
                seed=config.seed,
            )
            lstm_result = train_lstm(
                dataset, fold.split, run_dir=fold_dir, control=control,
                epochs=config.lstm_epochs, patience=config.lstm_patience,
                batch_size=config.batch_size, seed=config.seed,
            )
            _calibrate_job_hybrid(
                job=job, run_dir=fold_dir, dataset=dataset, split=fold.split,
                xgboost_result=xgboost_result, lstm_result=lstm_result,
            )
            del xgboost_result, lstm_result
        _write_atomic(job_dir / "folds.json", {
            "format_version": 1,
            "folds": [_fold_manifest(fold, index) for index, fold in enumerate(folds)],
        })
        del dataset
        gc.collect()
        shutil.rmtree(job_dir / "dataset-cache")
        inputs = []
        for index, path_string in enumerate(job["inputs"]):
            input_path = Path(path_string).resolve()
            _verify_input(input_path, fingerprints[index])
            score_path = job_dir / "scores" / f"{index:05d}.parquet"
            symbol, rows = _write_symbol_scores(
                input_path=input_path, job=job, folds=folds, fold_dirs=fold_dirs,
                destination=score_path, config=config,
            )
            _verify_input(input_path, fingerprints[index])
            inputs.append({
                "symbol": symbol,
                "symbol_directory": input_path.parent.name,
                "input": str(input_path.resolve()),
                "input_fingerprint": fingerprints[index],
                "scores": str(score_path.relative_to(job_dir)),
                "rows": rows,
            })
            if (index + 1) % 100 == 0:
                print(f"Scored {index + 1}/{len(job['inputs'])} symbols", flush=True)
        _write_atomic(job_dir / "scores.json", {
            "format_version": 1, "timestamp_timezone": "UTC", "inputs": inputs,
        })
        manifest["completed_jobs"] = job_index + 1
        _write_atomic(manifest_path, manifest)
    manifest["status"] = "complete"
    _write_atomic(manifest_path, manifest)
