from __future__ import annotations

import gc
import hashlib
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np

from backtesting.execution_simulator import ExecutionCosts
from .batch_plan import TrainingBatchPlan, iter_planned_jobs
from .control import TrainingControl, _replace_with_retry
from .data import (
    DatasetSplit,
    ModelStyle,
    TrainingDataset,
    build_training_dataset,
    chronological_split,
)
from .training import (
    TrainingStopped,
    _probabilities_from_logits,
    _xgboost_device,
    calibrate_hybrid,
    classification_metrics,
    train_lstm,
    train_xgboost,
)

_BATCH_FORMAT_VERSION = 1
_BYTES_PER_GIB = 1024**3
_WORKER_MEMORY_GIB = 4
_RESERVED_MEMORY_GIB = 2
_LSTM_TRAINING_LOCK = Lock()


def _recommended_worker_count(
    *,
    cpu_count: int,
    total_memory_bytes: int,
    available_memory_bytes: int,
) -> int:
    cpu_limit = max(1, cpu_count // 2)
    total_memory_limit = max(
        1,
        (total_memory_bytes - _RESERVED_MEMORY_GIB * _BYTES_PER_GIB)
        // (_WORKER_MEMORY_GIB * _BYTES_PER_GIB),
    )
    available_memory_limit = max(
        1,
        (available_memory_bytes - _RESERVED_MEMORY_GIB * _BYTES_PER_GIB)
        // (_WORKER_MEMORY_GIB * _BYTES_PER_GIB),
    )
    return int(min(cpu_limit, total_memory_limit, available_memory_limit))


def _available_worker_count() -> int:
    import psutil

    memory = psutil.virtual_memory()
    return _recommended_worker_count(
        cpu_count=os.cpu_count() or 1,
        total_memory_bytes=memory.total,
        available_memory_bytes=memory.available,
    )


def _xgboost_uses_cuda() -> bool:
    import xgboost

    return _xgboost_device(xgboost) == "cuda"


def _train_lstm_serialized(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _LSTM_TRAINING_LOCK:
        return train_lstm(*args, **kwargs)


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_plan(path: Path) -> TrainingBatchPlan:
    plan = cast(TrainingBatchPlan, json.loads(path.read_text(encoding="utf-8")))
    if plan.get("format_version") != 1 or plan.get("model_stages") != [
        "xgboost",
        "lstm",
        "hybrid_calibration",
    ]:
        raise ValueError(f"Unsupported or invalid training batch plan: {path}")
    if plan.get("training_started") is not False:
        raise ValueError(f"Training plan has an unexpected execution state: {path}")
    return plan


def _plan_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as lock_file:
                lock_file.write(json.dumps({"pid": os.getpid()}) + "\n")
            return os.getpid()
        except FileExistsError:
            try:
                lock_data = json.loads(path.read_text(encoding="utf-8"))
                process_id = int(lock_data["pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot determine whether batch lock is active: {path}"
                ) from exc
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                path.unlink()
                continue
            except PermissionError:
                pass
            raise RuntimeError(
                f"Another batch trainer is active with PID {process_id}; "
                f"lock file: {path}"
            )
    raise RuntimeError(f"Could not acquire batch trainer lock: {path}")


def _release_lock(path: Path, process_id: int) -> None:
    if not path.exists():
        return
    try:
        lock_data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if lock_data.get("pid") == process_id:
        path.unlink(missing_ok=True)


def _build_dataset_for_job(
    job: dict[str, str],
    *,
    sequence_length: int,
    validation_fraction: float,
    test_fraction: float,
    costs: ExecutionCosts,
) -> tuple[TrainingDataset, DatasetSplit]:
    horizon = 12 if job["style"] == "intraday" else 10
    dataset = build_training_dataset(
        [job["input"]],
        style=cast(ModelStyle, job["style"]),
        strategy=job["strategy"],
        horizon=horizon,
        sequence_length=sequence_length,
        costs=costs,
    )
    split = chronological_split(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    return dataset, split


def _save_job_metadata(
    *,
    run_dir: Path,
    job: dict[str, str],
    dataset: TrainingDataset,
    split: DatasetSplit,
) -> None:
    manifest_path = run_dir / "batch-job.json"
    stat = Path(job["input"]).stat()
    manifest = {
        "format_version": _BATCH_FORMAT_VERSION,
        **job,
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "sequence_length": dataset.sequence_length,
        "horizon": dataset.horizon,
        "feature_names": list(dataset.feature_names),
        "validation_fraction": len(split.validation) / len(dataset.labels),
        "test_fraction": len(split.test) / len(dataset.labels),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"Input or dataset metadata changed since this job was started: "
                f"{job['symbol']} {job['interval']} {job['strategy']}"
            )
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata is missing: {metadata_path}")
        return

    training_features = dataset.features[split.train]
    mean = np.nanmean(training_features, axis=0).astype(np.float32)
    scale = np.nanstd(training_features, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    metadata = {
        "format_version": 1,
        "style": dataset.style,
        "strategy": dataset.strategy,
        "feature_names": list(dataset.feature_names),
        "mean": np.nan_to_num(mean, nan=0.0).tolist(),
        "scale": np.nan_to_num(scale, nan=1.0).tolist(),
        "sequence_length": dataset.sequence_length,
        "horizon": dataset.horizon,
        "class_names": {
            "0": "loss",
            "1": "near_break_even",
            "2": "profitable",
        },
        "training_examples": len(split.train),
        "validation_examples": len(split.validation),
        "test_examples": len(split.test),
        "validation_start": split.validation_start.isoformat(),
        "test_start": split.test_start.isoformat(),
        "input_fingerprint": {
            "path": str(Path(job["input"]).resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
    }
    _write_atomic(metadata_path := run_dir / "metadata.json", metadata)
    _write_atomic(manifest_path, manifest)


def _calibrate_job_hybrid(
    *,
    job: dict[str, str],
    run_dir: Path,
    dataset: TrainingDataset,
    split: DatasetSplit,
    xgboost_result: dict[str, Any],
    lstm_result: dict[str, Any],
) -> dict[str, Any]:
    adx_index = (
        dataset.feature_names.index("adx_14")
        if "adx_14" in dataset.feature_names
        else None
    )
    atr_index = (
        dataset.feature_names.index("atr_pct")
        if "atr_pct" in dataset.feature_names
        else None
    )
    training_features = dataset.features[split.train]
    regime_thresholds = {
        "adx_trend_threshold": 25.0,
        "atr_pct_low_threshold": (
            float(np.nanpercentile(training_features[:, atr_index], 25))
            if atr_index is not None
            else None
        ),
        "atr_pct_high_threshold": (
            float(np.nanpercentile(training_features[:, atr_index], 75))
            if atr_index is not None
            else None
        ),
        "adx_feature": "adx_14" if adx_index is not None else None,
        "volatility_feature": "atr_pct" if atr_index is not None else None,
    }
    calibration = calibrate_hybrid(
        labels=dataset.labels[split.validation],
        model_probabilities={
            "xgboost": xgboost_result["validation_probabilities"],
            "lstm": lstm_result["validation_probabilities"],
        },
        run_dir=run_dir,
        regime_thresholds=regime_thresholds,
    )
    blended_test = sum(
        weight
        * {
            "xgboost": xgboost_result["test_probabilities"],
            "lstm": lstm_result["test_probabilities"],
        }[model_name]
        for model_name, weight in calibration["model_weights"].items()
    )
    hybrid_test = _probabilities_from_logits(
        np.log(np.clip(blended_test, 1e-12, 1.0)),
        calibration["temperature"],
    )
    result = {
        "style": job["style"],
        "interval": job["interval"],
        "strategy": job["strategy"],
        "symbol": job["symbol"],
        "validation": calibration["validation_metrics"],
        "test": classification_metrics(
            dataset.labels[split.test],
            hybrid_test,
        ),
        "model_weights": calibration["model_weights"],
        "temperature": calibration["temperature"],
        "validation_log_loss": calibration["validation_log_loss"],
    }
    _write_atomic(run_dir / "hybrid-batch-metrics.json", result)
    return result


def _job_run_directory(root: Path, job: dict[str, str]) -> Path:
    return root / job["interval"] / job["strategy"] / job["symbol"]


def _prepare_for_stage(
    job: dict[str, str],
    *,
    run_dir: Path,
    sequence_length: int,
    validation_fraction: float,
    test_fraction: float,
    costs: ExecutionCosts,
) -> tuple[TrainingDataset | None, DatasetSplit | None]:
    job_status_path = run_dir / "batch-job-status.json"
    if job_status_path.is_file():
        job_status = json.loads(job_status_path.read_text(encoding="utf-8"))
        if job_status.get("status") == "failed":
            print(
                f"Skipping {job['symbol']} {job['interval']} {job['strategy']}: "
                f"{job_status['error']}"
            )
            return None, None
    try:
        dataset, split = _build_dataset_for_job(
            job,
            sequence_length=sequence_length,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            costs=costs,
        )
    except (ValueError, FileNotFoundError) as exc:
        _write_atomic(
            job_status_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "job": job,
            },
        )
        print(
            f"Skipping {job['symbol']} {job['interval']} {job['strategy']}: {exc}"
        )
        return None, None
    _save_job_metadata(
        run_dir=run_dir,
        job=job,
        dataset=dataset,
        split=split,
    )
    return dataset, split


def _train_stage(
    stage: str,
    *,
    job: dict[str, str],
    run_dir: Path,
    dataset: TrainingDataset,
    split: DatasetSplit,
    control: TrainingControl,
    config: dict[str, Any],
) -> None:
    if stage == "xgboost":
        train_xgboost(
            dataset,
            split,
            run_dir=run_dir,
            control=control,
            max_rounds=config["xgboost_rounds"],
            patience=config["xgboost_patience"],
            seed=config["seed"],
        )
    elif stage == "lstm":
        _train_lstm_serialized(
            dataset,
            split,
            run_dir=run_dir,
            control=control,
            epochs=config["lstm_epochs"],
            patience=config["lstm_patience"],
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
    else:
        for model_name in ("xgboost", "lstm"):
            state_path = run_dir / f"{model_name}.state.json" if model_name == "xgboost" else run_dir / "lstm.checkpoint.pt"
            if model_name == "xgboost":
                if not state_path.is_file() or not json.loads(
                    state_path.read_text(encoding="utf-8")
                ).get("completed"):
                    raise RuntimeError(
                        f"Hybrid stage requires a completed XGBoost model for {job}"
                    )
            elif not state_path.is_file() or not torch_lstm_completed(state_path):
                raise RuntimeError(f"Hybrid stage requires a completed LSTM model for {job}")

        xgboost_result = train_xgboost(
            dataset,
            split,
            run_dir=run_dir,
            control=control,
            max_rounds=config["xgboost_rounds"],
            patience=config["xgboost_patience"],
            seed=config["seed"],
        )
        lstm_result = _train_lstm_serialized(
            dataset,
            split,
            run_dir=run_dir,
            control=control,
            epochs=config["lstm_epochs"],
            patience=config["lstm_patience"],
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
        _calibrate_job_hybrid(
            job=job,
            run_dir=run_dir,
            dataset=dataset,
            split=split,
            xgboost_result=xgboost_result,
            lstm_result=lstm_result,
        )
        del xgboost_result, lstm_result
    _write_atomic(
        run_dir / "batch-job-status.json",
        {"status": f"{stage}_complete", "job": job},
    )


def _process_batch_job(
    stage: str,
    *,
    job: dict[str, str],
    run_root: Path,
    sequence_length: int,
    validation_fraction: float,
    test_fraction: float,
    costs: ExecutionCosts,
    control: TrainingControl,
    config: dict[str, Any],
) -> str:
    run_dir = _job_run_directory(run_root, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    job_status_path = run_dir / "batch-job-status.json"
    if job_status_path.is_file():
        job_status = json.loads(job_status_path.read_text(encoding="utf-8"))
        if job_status.get("status") == f"{stage}_complete":
            return "completed"

    dataset, split = _prepare_for_stage(
        job,
        run_dir=run_dir,
        sequence_length=sequence_length,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        costs=costs,
    )
    if dataset is None or split is None:
        return "skipped"
    try:
        _train_stage(
            stage,
            job=job,
            run_dir=run_dir,
            dataset=dataset,
            split=split,
            control=control,
            config=config,
        )
    except TrainingStopped:
        return "stopped"
    finally:
        del dataset, split
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return "completed"


def torch_lstm_completed(checkpoint_path: Path) -> bool:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to inspect the LSTM checkpoint."
        ) from exc
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    return bool(checkpoint.get("completed"))


def run_training_batch(
    *,
    plan_path: str | Path,
    run_root: str | Path,
    resume: bool = False,
    max_jobs_per_stage: int | None = None,
    allow_large_batch: bool = False,
    sequence_length: int = 60,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    commission_bps: float = 3.0,
    other_fees_bps: float = 7.0,
    slippage_bps: float = 5.0,
    seed: int = 42,
    xgboost_rounds: int = 500,
    xgboost_patience: int = 30,
    lstm_epochs: int = 40,
    lstm_patience: int = 8,
    batch_size: int = 32,
) -> dict[str, Any]:
    plan_file = Path(plan_path).resolve()
    if not plan_file.is_file():
        raise FileNotFoundError(f"Training batch plan not found: {plan_file}")
    plan = _read_plan(plan_file)
    if max_jobs_per_stage is not None and max_jobs_per_stage < 1:
        raise ValueError("max_jobs_per_stage must be at least one")
    if plan["job_count"] > 10_000 and max_jobs_per_stage is None and not allow_large_batch:
        raise ValueError(
            f"This batch contains {plan['job_count']:,} jobs. Add --max-jobs-per-stage "
            "to run a bounded slice, or --allow-large-batch to execute the full queue."
        )
    if sequence_length < 1 or min(
        xgboost_rounds,
        xgboost_patience,
        lstm_epochs,
        lstm_patience,
        batch_size,
    ) < 1:
        raise ValueError("Sequence length, model limits, and batch size must be positive")

    required_packages = ("xgboost", "torch", "psutil")
    from importlib.util import find_spec

    missing_packages = [
        package for package in required_packages if find_spec(package) is None
    ]
    if missing_packages:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing_packages)
            + ". Install requirements.txt and requirements-training.txt first."
        )
    import torch

    import xgboost

    xgboost_device = _xgboost_device(xgboost)
    lstm_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Training devices: XGBoost={xgboost_device}, LSTM={lstm_device}"
        + (
            " (installed PyTorch build has no CUDA support)"
            if lstm_device == "cpu"
            else ""
        ),
        flush=True,
    )

    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "batch.lock"
    process_id = _acquire_lock(lock_path)
    try:
        config = {
            "sequence_length": sequence_length,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "commission_bps": commission_bps,
            "other_fees_bps": other_fees_bps,
            "slippage_bps": slippage_bps,
            "seed": seed,
            "xgboost_rounds": xgboost_rounds,
            "xgboost_patience": xgboost_patience,
            "lstm_epochs": lstm_epochs,
            "lstm_patience": lstm_patience,
            "batch_size": batch_size,
        }
        manifest_path = root / "batch-manifest.json"
        manifest = {
            "format_version": _BATCH_FORMAT_VERSION,
            "plan_path": str(plan_file),
            "plan_sha256": _plan_fingerprint(plan_file),
            "run_root": str(root),
            "configuration": config,
            "model_stages": plan["model_stages"],
        }
        state_path = root / "batch-state.json"
        if manifest_path.is_file() or state_path.is_file():
            if not resume:
                raise FileExistsError(
                    f"Batch run exists under {root}; pass --resume to continue."
                )
            if not manifest_path.is_file() or not state_path.is_file():
                raise ValueError(f"Incomplete batch run metadata under {root}")
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError(
                    "Plan or training settings differ from this batch run. "
                    "Use the original plan/settings or a new --run-root."
                )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "completed":
                print("This batch run is already complete.")
                return state
        else:
            if resume:
                raise FileNotFoundError(f"No resumable batch state found under {root}")
            _write_atomic(manifest_path, manifest)
            state = {
                "format_version": _BATCH_FORMAT_VERSION,
                "status": "running",
                "stage_index": 0,
                "job_index": 0,
                "completed_jobs_by_stage": {
                    stage: 0 for stage in plan["model_stages"]
                },
                "failed_jobs": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_atomic(state_path, state)

        control = TrainingControl(root)
        if resume:
            control.request("continue")
        costs = ExecutionCosts(
            commission_bps=commission_bps,
            other_fees_bps=other_fees_bps,
            slippage_bps=slippage_bps,
        )

        def save_state() -> None:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_atomic(state_path, state)

        for stage_index in range(int(state["stage_index"]), len(plan["model_stages"])):
            stage = plan["model_stages"][stage_index]
            state["status"] = "running"
            save_state()
            start_job = int(state["job_index"]) if stage_index == int(state["stage_index"]) else 0
            jobs_processed = 0
            job_index = start_job
            while job_index < plan["job_count"]:
                workers = _available_worker_count()
                if stage in {"xgboost", "hybrid_calibration"} and _xgboost_uses_cuda():
                    workers = 1
                remaining_cap = (
                    max_jobs_per_stage - jobs_processed
                    if max_jobs_per_stage is not None
                    else workers
                )
                window_size = min(
                    workers,
                    remaining_cap,
                    plan["job_count"] - job_index,
                )
                jobs = list(
                    itertools.islice(
                        iter_planned_jobs(plan, start_index=job_index),
                        window_size,
                    )
                )
                print(
                    f"{stage}: running {len(jobs)} job(s) concurrently "
                    f"(resource limit: {workers} worker(s))",
                    flush=True,
                )
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            _process_batch_job,
                            stage,
                            job=job,
                            run_root=root,
                            sequence_length=sequence_length,
                            validation_fraction=validation_fraction,
                            test_fraction=test_fraction,
                            costs=costs,
                            control=control,
                            config=config,
                        )
                        for job in jobs
                    ]
                    results = [future.result() for future in futures]

                completed_count = results.count("completed")
                state["completed_jobs_by_stage"][stage] += completed_count
                if stage_index == 0:
                    state["failed_jobs"] += results.count("skipped")
                jobs_processed += len(jobs)

                stopped_offset = next(
                    (
                        offset
                        for offset, result in enumerate(results)
                        if result == "stopped"
                    ),
                    None,
                )
                if stopped_offset is not None:
                    state["stage_index"] = stage_index
                    state["job_index"] = job_index + stopped_offset
                    state["status"] = "stopped"
                    save_state()
                    print(
                        "Batch stopped at model checkpoints. Rerun the same "
                        "command with --resume."
                    )
                    return state

                job_index += len(jobs)
                if job_index >= plan["job_count"]:
                    state["stage_index"] = stage_index + 1
                    state["job_index"] = 0
                else:
                    state["stage_index"] = stage_index
                    state["job_index"] = job_index
                save_state()
                if not control.checkpoint_boundary(save_state):
                    state["status"] = "stopped"
                    save_state()
                    print("Batch stopped. Rerun the same command with --resume.")
                    return state
                if (
                    max_jobs_per_stage is not None
                    and jobs_processed >= max_jobs_per_stage
                    and state["stage_index"] == stage_index
                ):
                    state["status"] = "paused_at_job_limit"
                    save_state()
                    print(
                        f"Reached --max-jobs-per-stage={max_jobs_per_stage} in "
                        f"{stage}; rerun the same command with --resume."
                    )
                    return state
                print(
                    f"{stage}: {job_index:,}/{plan['job_count']:,} jobs; "
                    f"status={state['status']}",
                    flush=True,
                )
            state["stage_index"] = stage_index + 1
            state["job_index"] = 0
            save_state()

        state["status"] = "completed"
        save_state()
        print(
            f"Batch complete: {sum(state['completed_jobs_by_stage'].values()):,} "
            f"successful stage-jobs; {state['failed_jobs']:,} jobs skipped for "
            f"unusable or missing data."
        )
        return state
    finally:
        _release_lock(lock_path, process_id)
