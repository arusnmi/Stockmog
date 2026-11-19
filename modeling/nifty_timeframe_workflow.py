"""Prepare, train, evaluate and forecast separate research stock/timeframe models."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json, load_closes
from modeling.nifty_candle_examples import NoCandleExamples, build_candle_examples
from modeling.nifty_factor_selection import _validate_fingerprints
from modeling.nifty_preparation import TIMEFRAMES
from modeling.nifty_price_evaluation import _metrics, _sha
from modeling.nifty_price_training import (
    PriceTrainingConfig, _price_lock, load_price_predictor, train_pooled_price_model,
)
from modeling.nifty_same_candle_correlations import _coefficients, _read_candles
from modeling.nifty_timeframe_data import (
    HISTORY_START, FactorStore, UnusableTimeframe, add_availability, attach_factor,
    chronological_boundaries, make_dataset, read_stock, select_factors, strategy_features,
    training_history, training_observations, training_universe,
)
from modeling.nifty_universe import (
    COVERAGE_END, local_symbol, parse_nifty_membership,
)
from modeling.training import TrainingStopped


PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = Path("data") / "models" / "nifty-timeframe"
DEFAULT_SCAN = Path("data") / "external" / "nifty_same_candle_correlations"


def _boundary(control: TrainingControl) -> None:
    paused = control.acknowledge and control.action() in {"pause", "paused"}
    if paused:
        print("Paused. Waiting for resume or stop...", flush=True)
    if not control.checkpoint_boundary(lambda: None):
        if control.acknowledge:
            print("Stop requested; leaving this stage at its checkpoint.", flush=True)
        raise PreparationStopped("Stopped at a checkpoint; repeat this command with --resume")
    if paused:
        print("Resuming...", flush=True)


def _parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    _replace_with_retry(temporary, path)


def _job_dir(output: Path, symbol: str, timeframe: str) -> Path:
    return output / "jobs" / _safe_symbol(symbol) / timeframe


def _sequence_length(timeframe: str, requested: int) -> int:
    if timeframe in INTRADAY_TIMEFRAMES:
        return min(requested, max(1, 300 // INTRADAY_TIMEFRAMES[timeframe]))
    return min(requested, 3 if timeframe == "trading_week" else 10 if timeframe == "2d" else requested)


def _scan_config(scan: Path, sequence_length: int, symbols: list[str] | None,
                 timeframes: list[str] | None, minimum: int) -> tuple[dict, list[dict]]:
    summary = json.loads((scan / "summary.json").read_text(encoding="utf-8"))
    identity_path = scan / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("The same-candle scan must be complete")
    data = PROJECT / "data"
    membership_path = data / "external" / "nse_index_membership_0786cd915974.csv"
    membership = parse_nifty_membership(membership_path.read_text(encoding="utf-8-sig"))
    all_symbols = training_universe(membership)
    if not set(all_symbols) <= set(identity["symbols"]):
        raise ValueError("The correlation scan does not cover the requested 2022–2026 universe")
    chosen = all_symbols if symbols is None else symbols
    frames = list(TIMEFRAMES if timeframes is None else timeframes)
    if (not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(all_symbols)
            or not frames or len(set(frames)) != len(frames) or not set(frames) <= set(TIMEFRAMES)
            or sequence_length < 1 or minimum < 3):
        raise ValueError("Invalid symbols, timeframes, sequence length or minimum pairs")
    # Historical scan outputs stay immutable; selection is recomputed from current inputs.
    inputs = []
    for symbol in all_symbols:
        csv_path = scan / "stocks" / f"{_safe_symbol(symbol)}.csv"
        checkpoint = csv_path.with_suffix(".json")
        if _sha(csv_path) != json.loads(checkpoint.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError(f"Corrupt correlation output: {csv_path}")
        inputs.extend((csv_path, checkpoint))
    for record in identity["inputs"]:
        inputs.append(Path(record["path"]))
    config = {
        "stocks": str(data / "stocks"),
        "membership": str(data / "external" / "nse_index_membership_0786cd915974.csv"),
        "global_roots": [str(data / "external" / name) for name in
                         ("global_factor_research", "nifty_extra_sector_factors")],
        "india": str(data / "external" / "nifty_all_factors_india_2025"),
        "treasury": str(data / "external" / "nifty_all_factors_treasury_2025"),
        "prices": str(data / "external" / "nifty_all_factors_prices_2025"),
        "symbols": sorted(chosen), "timeframes": frames, "minimum_pairs": minimum,
        "sequence_length": sequence_length, "correlation_scan": str(scan.resolve()),
        "history_policy": "2022_2026_union_full_available_history",
        "local_factors": sorted({_safe_symbol(local_symbol(symbol, membership)) for symbol in all_symbols}),
    }
    inputs.extend((identity_path, scan / "summary.json", membership_path))
    return config, [_fingerprint(path) for path in sorted(set(inputs))]


def _code_hashes() -> dict:
    paths = [*sorted((PROJECT / "modeling").glob("*.py")),
             *sorted((PROJECT / "features").glob("*.py")), *sorted((PROJECT / "strategies").glob("*.py")),
             PROJECT / "data_pipeline" / "huggingface_stocks.py"]
    return {str(path.relative_to(PROJECT)): _sha(path) for path in paths}


def _load_job(folder: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    for name in ("examples", "forecast_features"):
        path = folder / f"{name}.parquet"
        if _sha(path) != meta[name + "_sha256"]:
            raise ValueError(f"Corrupt prepared file: {path}")
    return meta, pd.read_parquet(folder / "examples.parquet"), pd.read_parquet(folder / "forecast_features.parquet")


def _dataset(folder: Path):
    meta, examples, _ = _load_job(folder)
    return make_dataset(
        examples, meta["feature_names"], meta["symbol"], meta["timeframe"],
        date.fromisoformat(meta["validation_start"]), date.fromisoformat(meta["holdout_start"]),
        meta["sequence_length"], meta["selection"])


def _prepare_job(config: dict, symbol: str, timeframe: str, folder: Path,
                 factors: FactorStore, control: TrainingControl) -> dict:
    members = parse_nifty_membership(Path(config["membership"]).read_text(encoding="utf-8-sig"))
    relevant = training_history(symbol, members)
    local = local_symbol(symbol, members)
    candles = read_stock(Path(config["stocks"]), local, timeframe, relevant)
    if candles is None:
        raise UnusableTimeframe("missing_stock_candles")
    examples = build_candle_examples(symbol, timeframe, candles, factors.reference, relevant,
                                     start=HISTORY_START, end=COVERAGE_END)
    validation, holdout = chronological_boundaries(examples)
    target = _read_candles(Path(config["stocks"]), local, timeframe, factors.reference, relevant)
    if target is None:
        raise UnusableTimeframe("missing_stock_candles")
    target = training_observations(add_availability(target, candles, timeframe), validation)
    rows = []
    for index, key in enumerate(factors.keys()):
        _boundary(control)
        if key == ("local", _safe_symbol(local)):
            continue
        observations, _ = factors.load(key, timeframe)
        if observations is None:
            row = {"return_status": "missing_factor_timeframe", "return_pairs": 0}
        else:
            observations = training_observations(observations, validation)
            row = _coefficients(target, observations, config["minimum_pairs"])
            row = {name: value for name, value in row.items() if name.startswith("return_")}
        rows.append({"source": key[0], "factor": key[1], **row})
        if index % 16 == 0:
            print(f"Preparing {symbol} {timeframe}: factors {index + 1}/{len(factors.keys())} "
                  f"({key[1]})", flush=True)
            _write_json(control.run_dir / "state.json", {
                "status": "preparing", "stock": symbol, "timeframe": timeframe,
                "factor": key[1], "completed_factors": index + 1, "total_factors": len(factors.keys())})
    selection = select_factors(rows, minimum=config["minimum_pairs"])
    selection.update(train_end_exclusive=validation.isoformat(), minimum_pairs=config["minimum_pairs"],
                     metric="absolute same-candle return Pearson AND Spearman", candidates=rows)
    features = strategy_features(candles, timeframe)
    for item in selection["selected"]:
        _boundary(control)
        key = (item["source"], item["factor"])
        _, values = factors.load(key, timeframe)
        if values is None:
            raise ValueError(f"Selected factor is unavailable: {key}")
        features = attach_factor(features, values, "factor:" + json.dumps(key), timeframe)
    names = [name for name in features if name not in {"timestamp", "available_at", "close"}]
    sequence = _sequence_length(timeframe, config["sequence_length"])
    examples = examples.merge(features.drop(columns=["available_at", "close"]), on="timestamp",
                              validate="one_to_one", how="left")
    dataset = make_dataset(examples, names, symbol, timeframe, validation, holdout, sequence, selection)
    folder.mkdir(parents=True, exist_ok=True)
    _parquet(examples, folder / "examples.parquet")
    _parquet(features, folder / "forecast_features.parquet")
    meta = {"symbol": symbol, "timeframe": timeframe, "sequence_length": sequence,
            "selection": selection, "feature_names": names, "validation_start": validation.isoformat(),
            "holdout_start": holdout.isoformat(), "examples_sha256": _sha(folder / "examples.parquet"),
            "forecast_features_sha256": _sha(folder / "forecast_features.parquet"),
            "split_counts": {"train": len(dataset.train), "validation": len(dataset.validation),
                             "holdout": len(dataset.holdout)}, "dataset_sha256": dataset.identity}
    _write_json(folder / "prepared.json", meta)
    return meta


def prepare(output: Path, scan: Path, *, resume: bool = False, sequence_length: int = 20,
            symbols: list[str] | None = None, timeframes: list[str] | None = None,
            minimum: int = 60) -> dict:
    config, fingerprints = _scan_config(scan, sequence_length, symbols, timeframes, minimum)
    identity = {"version": 1, "config": config, "inputs": fingerprints, "code": _code_hashes()}
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists: {output}; repeat with --resume")
    if not output.exists() and resume:
        raise FileNotFoundError(f"No run to resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with _price_lock(output):
        manifest = output / "identity.json"
        if resume:
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Preparation inputs or settings changed; use a new run directory")
        else:
            _write_json(manifest, identity)
        control = TrainingControl(output)
        if resume:
            control.request("continue")
        print("Preparation: loading factor collections...", flush=True)
        _boundary(control)
        reference = list(load_closes(Path(config["stocks"]) / "day" / "RELIANCE" / "1d.csv.gz").index)
        factors = FactorStore(config, reference)
        jobs, skipped = [], []
        for symbol in config["symbols"]:
            for timeframe in config["timeframes"]:
                _boundary(control)
                folder = _job_dir(output, symbol, timeframe)
                if (folder / "prepared.json").is_file():
                    meta, _, _ = _load_job(folder)
                else:
                    try:
                        meta = _prepare_job(config, symbol, timeframe, folder, factors, control)
                    except (UnusableTimeframe, NoCandleExamples) as exc:
                        record = {"symbol": symbol, "timeframe": timeframe, "reason": str(exc)}
                        skipped.append(record)
                        print(f"Skipped {symbol} {timeframe}: {exc}", flush=True)
                        continue
                jobs.append({"symbol": symbol, "timeframe": timeframe,
                             "factors": len(meta["selection"]["selected"]),
                             "prepared_sha256": _sha(folder / "prepared.json")})
                _write_json(output / "state.json", {"status": "preparing", "completed_jobs": len(jobs),
                                                    "skipped_jobs": len(skipped), "stock": symbol,
                                                    "timeframe": timeframe})
                print(f"Prepared {symbol} {timeframe}: {len(meta['selection']['selected'])} factors", flush=True)
        _validate_fingerprints(fingerprints)
        if not jobs:
            raise ValueError("No usable stock/timeframe jobs; inspect the reported missing history")
        result = {"status": "complete", "research_only": True, "jobs": jobs, "skipped": skipped,
                  "limitations": [
                      "Retrospective union of 2022-2026 members; history outside membership is included.",
                      "This universe is not a point-in-time Nifty strategy and has selection/survivorship bias.",
                      "Factor thresholds are selected on training returns only and frozen for validation/test.",
                      "Contemporaneous association does not establish next-candle predictability.",
                      "Economic release times are unverified; availability assumes end of following New York day.",
                      "Source rights, adjustments and exchange calendars remain unverified.",
                  ]}
        _write_json(output / "preparation-summary.json", result)
        _write_json(output / "state.json", {"status": "prepared", "completed_jobs": len(jobs),
                                            "skipped_jobs": len(skipped)})
        return result


def _run_summary(output: Path) -> dict:
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    if identity["code"] != _code_hashes():
        raise ValueError("Workflow code changed after preparation; use a new run directory")
    _validate_fingerprints(identity["inputs"])
    summary = json.loads((output / "preparation-summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Preparation is not complete")
    for job in summary["jobs"]:
        path = _job_dir(output, job["symbol"], job["timeframe"]) / "prepared.json"
        if _sha(path) != job["prepared_sha256"]:
            raise ValueError(f"Prepared job metadata changed: {path}")
    return summary


def _validate_training_device(device: str) -> None:
    if device == "cuda":
        import torch
        import xgboost

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requires CUDA-enabled PyTorch and a working NVIDIA GPU; "
                               "use --device cpu explicitly for CPU training")
        if not xgboost.build_info().get("USE_CUDA"):
            raise RuntimeError("Installed XGBoost lacks CUDA support; use a CUDA-enabled build or --device cpu")


def _train_job(output: Path, job: dict, recorded: dict, resume: bool) -> dict:
    import torch

    started = time.monotonic()
    torch.set_num_threads(2)
    control = TrainingControl(output, acknowledge=False)
    _boundary(control)
    folder = _job_dir(output, job["symbol"], job["timeframe"])
    meta, _, _ = _load_job(folder)
    dataset = _dataset(folder)
    if dataset.identity != meta["dataset_sha256"]:
        raise ValueError(f"Prepared dataset changed: {folder}")
    model = folder / "model"
    training_config = PriceTrainingConfig(**{**recorded, "sequence_length": meta["sequence_length"]})
    try:
        train_pooled_price_model(dataset, model, config=training_config, research_only=True,
                                 resume=resume and model.exists(), control=control)
    finally:
        if training_config.device == "cuda" and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return {"symbol": job["symbol"], "timeframe": job["timeframe"],
            "elapsed_seconds": time.monotonic() - started}


def _print_training_progress(output: Path, active: list[dict], completed: int, total: int) -> None:
    print(f"Training: {completed}/{total} jobs complete; {len(active)} active", flush=True)
    for job in active:
        path = _job_dir(output, job["symbol"], job["timeframe"]) / "model" / "training-state.json"
        try:
            snapshot = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            detail = "loading dataset / initializing model"
        except PermissionError:
            detail = "progress snapshot unavailable; will retry on next update"
        else:
            saved = json.loads(snapshot)
            detail = (f"{saved['stage']}: tree {saved['tree_round']}/{saved['tree_rounds']}, "
                      f"epoch {saved['epoch']}/{saved['epochs']}, batch offset {saved['batch_offset']}")
        print(f"  {job['symbol']} {job['timeframe']} - {detail}", flush=True)


def train(output: Path, config: PriceTrainingConfig, *, resume: bool = False, workers: int = 2,
          reset_control: bool = True) -> dict:
    if workers < 1:
        raise ValueError("Worker count must be positive")
    _validate_training_device(config.device)
    summary = _run_summary(output)
    with _price_lock(output):
        control = TrainingControl(output)
        if resume and reset_control:
            control.request("continue")
        _boundary(control)
        settings = output / "training-config.json"
        recorded = asdict(config)
        if settings.exists():
            if not resume or json.loads(settings.read_text(encoding="utf-8")) != recorded:
                raise ValueError("Training already exists or settings changed; use --resume with identical settings")
        else:
            _write_json(settings, recorded)
        completed: list[dict] = []
        total = len(summary["jobs"])
        if not total:
            raise ValueError("No prepared training jobs")
        worker_count = min(workers, total)
        jobs = iter(summary["jobs"])
        pending = {}
        failure: BaseException | None = None
        started = time.monotonic()
        last_progress = started - 15

        def state(status: str, **extra) -> None:
            _write_json(output / "state.json", {
                "status": status, "workers": worker_count, "device": config.device,
                "completed_jobs": len(completed), "total_jobs": total,
                "active_jobs": list(pending.values()), **extra})

        # Spawn isolates CUDA contexts, random seeds and model checkpoints on Windows.
        with ProcessPoolExecutor(max_workers=worker_count,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            try:
                while len(completed) < total:
                    _boundary(control)
                    while len(pending) < worker_count:
                        job = next(jobs, None)
                        if job is None:
                            break
                        pending[executor.submit(_train_job, output, job, recorded, resume)] = job
                    state("training")
                    now = time.monotonic()
                    if now - last_progress >= 15:
                        _print_training_progress(output, list(pending.values()), len(completed), total)
                        last_progress = now
                    finished, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        job = pending.pop(future)
                        try:
                            result = future.result()
                        except (PreparationStopped, TrainingStopped):
                            raise
                        except Exception as exc:
                            raise RuntimeError(
                                f"Training failed for {job['symbol']} {job['timeframe']}: {exc}. "
                                "For CUDA out-of-memory, retry with --workers 1 --resume."
                            ) from exc
                        completed.append(result)
                        print(f"Trained {job['symbol']} {job['timeframe']} "
                              f"({len(completed)}/{total}, {result['elapsed_seconds']:.1f}s)", flush=True)
            except BaseException as exc:
                control.request("stop")
                for future in pending:
                    future.cancel()
                state("stopping", error=str(exc))
                failure = exc
                # The executor waits for active workers to save before releasing the run lock.
        if failure is not None:
            control.checkpoint_boundary(lambda: None)
            state("stopped" if isinstance(failure, (PreparationStopped, TrainingStopped, KeyboardInterrupt))
                  else "failed", error=str(failure), active_jobs=[])
            raise failure
        result = {"status": "complete", "jobs": len(completed), "research_only": True,
                  "device": config.device, "workers": worker_count,
                  "elapsed_seconds": time.monotonic() - started, "job_timings": completed}
        _write_json(output / "training-summary.json", result)
        state("trained")
        return result


def run_training(output: Path, scan: Path, config: PriceTrainingConfig, *, resume: bool = False,
                 workers: int = 2, sequence_length: int = 20, symbols: list[str] | None = None,
                 timeframes: list[str] | None = None, minimum: int = 60) -> dict:
    if workers < 1:
        raise ValueError("Worker count must be positive")
    _validate_training_device(config.device)
    prepare(output, scan, resume=resume, sequence_length=sequence_length,
            symbols=symbols, timeframes=timeframes, minimum=minimum)
    _boundary(TrainingControl(output))
    print(f"Preparation complete. Starting training: {config.device}, {workers} workers.", flush=True)
    return train(output, config, resume=resume, workers=workers, reset_control=False)


def _predict(predictor, dataset, indices: np.ndarray, batch_size: int, control: TrainingControl) -> np.ndarray:
    values = []
    for start in range(0, len(indices), batch_size):
        _boundary(control)
        batch = indices[start:start + batch_size]
        values.append(predictor.predict(dataset.features[batch], dataset.features[dataset.sequences[batch]],
                                        dataset.anchor_closes[batch], feature_names=list(dataset.feature_names)))
    return np.concatenate(values)


def evaluate(output: Path, *, device: str = "cpu", batch_size: int = 256,
             resume: bool = False) -> dict:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    summary = _run_summary(output)
    with _price_lock(output):
        control = TrainingControl(output)
        if resume:
            control.request("continue")
        reports = []
        for job in summary["jobs"]:
            _boundary(control)
            folder = _job_dir(output, job["symbol"], job["timeframe"])
            dataset = _dataset(folder)
            model = folder / "model"
            model_summary = json.loads((model / "training-summary.json").read_text(encoding="utf-8"))
            if model_summary["dataset_sha256"] != dataset.identity:
                raise ValueError("Model dataset differs from prepared data")
            predictor = load_price_predictor(model, device=device)
            indices = dataset.holdout
            prices = _predict(predictor, dataset, indices, batch_size, control)
            _, examples, _ = _load_job(folder)
            frame = examples.iloc[indices][["timestamp", "available_at", "target_timestamp",
                                            "label_available_at"]].reset_index(drop=True)
            frame["anchor_close"] = dataset.anchor_closes[indices]
            frame["actual_close"] = dataset.target_closes[indices]
            frame["forecast_close"] = prices
            frame["baseline_close"] = frame["anchor_close"]
            frame["predicted_trend"] = np.where(prices > frame["anchor_close"], "up",
                                                np.where(prices < frame["anchor_close"], "down", "flat"))
            path = folder / "holdout-predictions.csv"
            temporary = path.with_suffix(".tmp")
            frame.to_csv(temporary, index=False)
            _replace_with_retry(temporary, path)
            report = {"symbol": job["symbol"], "timeframe": job["timeframe"],
                      "model": _metrics(frame, "forecast_close"),
                      "last_close_baseline": _metrics(frame, "baseline_close")}
            reports.append(report)
            _write_json(folder / "holdout-summary.json", report)
        result = {"status": "complete", "research_only": True, "jobs": reports,
                  "warning": "Do not repeatedly tune models against this holdout."}
        _write_json(output / "holdout-summary.json", result)
        return result


def predict(output: Path, symbol: str, timeframe: str, *, device: str = "cpu") -> dict:
    summary = _run_summary(output)
    if not any(job["symbol"] == symbol and job["timeframe"] == timeframe for job in summary["jobs"]):
        raise ValueError("No prepared model job for that stock/timeframe")
    folder = _job_dir(output, symbol, timeframe)
    meta, _, features = _load_job(folder)
    sequence_length = meta["sequence_length"]
    # Reuse the label builder to verify adjacency without using any future prices as inputs.
    members = parse_nifty_membership((PROJECT / "data" / "external" /
                                     "nse_index_membership_0786cd915974.csv").read_text(encoding="utf-8-sig"))
    relevant = training_history(symbol, members)
    candles = read_stock(PROJECT / "data" / "stocks", local_symbol(symbol, members), timeframe, relevant)
    if candles is None:
        raise ValueError("Stock candles are missing")
    reference = list(load_closes(PROJECT / "data" / "stocks" / "day" / "RELIANCE" / "1d.csv.gz").index)
    pairs = build_candle_examples(symbol, timeframe, candles, reference,
                                  relevant, start=HISTORY_START, end=COVERAGE_END)
    if timeframe in INTRADAY_TIMEFRAMES:
        # A session's final bar cannot forecast an overnight return this model never learned.
        features = features.loc[features["timestamp"] <= pairs["timestamp"].max()]
    tail = features.tail(sequence_length)
    if len(tail) < sequence_length:
        raise ValueError("Insufficient latest history for prediction")
    links = set(zip(pairs["timestamp"], pairs["target_timestamp"]))
    if any((left, right) not in links for left, right in zip(tail["timestamp"], tail["timestamp"].iloc[1:])):
        raise ValueError("Latest prediction sequence crosses a missing/partial candle or session")
    model_summary = json.loads((folder / "model" / "training-summary.json").read_text(encoding="utf-8"))
    if model_summary["dataset_sha256"] != meta["dataset_sha256"]:
        raise ValueError("Model dataset differs from prepared data")
    predictor = load_price_predictor(folder / "model", device=device)
    values = tail[meta["feature_names"]].to_numpy(dtype=np.float32)
    anchor = float(tail["close"].iloc[-1])
    price = float(predictor.predict(values[-1:], values[None, :, :], np.array([anchor]),
                                     feature_names=meta["feature_names"])[0])
    result = {"symbol": symbol, "timeframe": timeframe, "as_of": tail["available_at"].iloc[-1].isoformat(),
              "anchor_close": anchor, "forecast_close": price,
              "predicted_trend": "up" if price > anchor else "down" if price < anchor else "flat",
              "horizon": "next consecutive complete candle",
              "selected_factors": len(meta["selection"]["selected"]),
              "missing_factor_features": [name for name in meta["feature_names"]
                                          if name.startswith("factor:") and not name.endswith(":missing")
                                          and pd.isna(tail[name].iloc[-1])],
              "warning": "Historical snapshot forecast, not a live quote or trading recommendation."}
    _write_json(folder / "forecast.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "evaluate", "predict", "pause", "resume", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        if name in {"prepare", "train", "evaluate"}:
            command.add_argument("--resume", action="store_true")
        if name in {"prepare", "train"}:
            command.add_argument("--correlations", type=Path, default=DEFAULT_SCAN)
            command.add_argument("--symbols", nargs="+")
            command.add_argument("--timeframes", nargs="+", choices=TIMEFRAMES)
            command.add_argument("--sequence-length", type=int, default=20)
            command.add_argument("--minimum-pairs", type=int, default=60)
        if name in {"train", "evaluate", "predict"}:
            command.add_argument("--device", choices=("cpu", "cuda"),
                                 default="cuda" if name == "train" else "cpu")
        if name in {"train", "evaluate"}:
            command.add_argument("--batch-size", type=int, default=128)
        if name == "train":
            command.add_argument("--research-only", action="store_true", required=True)
            command.add_argument("--tree-rounds", type=int, default=100)
            command.add_argument("--epochs", type=int, default=20)
            command.add_argument("--workers", type=int, default=2)
        if name == "predict":
            command.add_argument("--symbol", required=True)
            command.add_argument("--timeframe", choices=TIMEFRAMES, required=True)
    args = parser.parse_args()
    try:
        output = args.run_dir.resolve()
        if args.command == "prepare":
            result = prepare(output, args.correlations.resolve(), resume=args.resume,
                             sequence_length=args.sequence_length, symbols=args.symbols,
                             timeframes=args.timeframes, minimum=args.minimum_pairs)
        elif args.command == "train":
            result = run_training(
                output, args.correlations.resolve(),
                PriceTrainingConfig(tree_rounds=args.tree_rounds, epochs=args.epochs,
                                    batch_size=args.batch_size, device=args.device),
                resume=args.resume, workers=args.workers, sequence_length=args.sequence_length,
                symbols=args.symbols, timeframes=args.timeframes, minimum=args.minimum_pairs)
        elif args.command == "evaluate":
            result = evaluate(output, device=args.device, batch_size=args.batch_size, resume=args.resume)
        elif args.command == "predict":
            result = predict(output, args.symbol, args.timeframe, device=args.device)
        else:
            if not (output / "identity.json").is_file():
                raise FileNotFoundError(f"No prepared run at {output}")
            action = "continue" if args.command == "resume" else args.command
            TrainingControl(output).request(action)
            result = {"requested": action}
        print(json.dumps(result, indent=2, default=str))
    except (PreparationStopped, TrainingStopped) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
