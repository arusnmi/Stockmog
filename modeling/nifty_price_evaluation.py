"""Evaluate a fitted research price model against untouched daily holdout data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import _write_json
from modeling.nifty_price_training import (
    PriceDataset, _price_lock, load_price_predictor, validate_dataset_content,
)


class EvaluationStopped(RuntimeError):
    """Evaluation stopped after a completed batch checkpoint."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlap(first: Path, second: Path) -> bool:
    for parent, child in ((first, second), (second, first)):
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            continue
    return False


def _metrics(frame: pd.DataFrame, column: str) -> dict:
    actual = frame["actual_close"].to_numpy(dtype=np.float64)
    predicted = frame[column].to_numpy(dtype=np.float64)
    anchor = frame["anchor_close"].to_numpy(dtype=np.float64)
    if (len(actual) == 0 or not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.isfinite(predicted).all() or (predicted <= 0).any()
            or not np.isfinite(anchor).all() or (anchor <= 0).any()):
        raise ValueError("Invalid prices in holdout results")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        error = predicted - actual
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        mape = float(np.mean(np.abs(error) / actual) * 100)
    directional_accuracy = float(np.mean(np.sign(predicted - anchor) == np.sign(actual - anchor)) * 100)
    return {"count": len(actual), "mae": mae, "rmse": rmse, "mape_percent": mape,
            "directional_accuracy_percent": directional_accuracy}


def evaluate_holdout(
    dataset: PriceDataset, model_dir: Path, output: Path, *, batch_size: int = 256,
    device: str = "cpu", resume: bool = False,
) -> dict:
    """Score only the model's reserved holdout; no fitting or calibration is performed."""
    if batch_size < 1 or device not in {"cpu", "cuda"}:
        raise ValueError("Invalid evaluation batch size or device")
    model_dir, output = model_dir.resolve(), output.resolve()
    if _overlap(model_dir, output):
        raise ValueError("Evaluation and model directories must not overlap")
    validate_dataset_content(dataset)
    if (len(dataset.holdout) == 0 or np.intersect1d(dataset.holdout, dataset.train).size
            or np.intersect1d(dataset.holdout, dataset.validation).size):
        raise ValueError("Holdout must be nonempty and disjoint from training and validation")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model_paths = [model_dir / name for name in ("training-summary.json", "price-tree.ubj", "price-lstm.pt")]
    with _price_lock(model_dir):
        summary = json.loads(model_paths[0].read_text(encoding="utf-8"))
        if (summary.get("status") != "complete" or summary.get("research_only") is not True
                or summary.get("dataset_sha256") != dataset.identity
                or summary.get("feature_names") != list(dataset.feature_names)
                or summary.get("reserved_holdout_examples") != len(dataset.holdout)):
            raise ValueError("Training summary does not match the reserved holdout dataset")
        model_hashes = [_sha(path) for path in model_paths]
        identity = {"version": 1, "dataset_sha256": dataset.identity, "content_sha256": dataset.content_hash,
                    "model_dir": str(model_dir), "model_sha256": model_hashes,
                    "device": device, "batch_size": batch_size}
        if output.exists():
            if not resume:
                raise FileExistsError(f"Evaluation output already exists: {output}")
        else:
            if resume:
                raise FileNotFoundError(f"Evaluation output does not exist: {output}")
            output.mkdir(parents=True)
        with _price_lock(output):
            manifest = output / "identity.json"
            if resume:
                if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
                    raise ValueError("Evaluation inputs or settings changed; use a new directory")
            else:
                _write_json(manifest, identity)
            predictor = load_price_predictor(model_dir, device=device)
            control = TrainingControl(output)
            if resume:
                control.request("continue")
            parts_dir = output / "parts"
            parts_dir.mkdir(exist_ok=True)

            def boundary() -> None:
                if not control.checkpoint_boundary(lambda: None):
                    raise EvaluationStopped("Holdout evaluation stopped; restart with resume=True")

            parts = []
            total = len(dataset.holdout)
            for number, start in enumerate(range(0, total, batch_size)):
                boundary()
                indices = dataset.holdout[start:start + batch_size]
                part = parts_dir / f"part-{number:06d}.parquet"
                checkpoint = part.with_suffix(".json")
                if checkpoint.is_file():
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if not part.is_file() or _sha(part) != saved["sha256"] or saved["rows"] != len(indices):
                        raise ValueError(f"Evaluation checkpoint is missing or corrupt: {part}")
                    frame = pd.read_parquet(part)
                    if len(frame) != len(indices):
                        raise ValueError(f"Evaluation checkpoint row count is invalid: {part}")
                else:
                    forecasts = predictor.predict(
                        dataset.features[indices], dataset.features[dataset.sequences[indices]],
                        dataset.anchor_closes[indices], feature_names=list(dataset.feature_names),
                    )
                    if (np.asarray(forecasts).shape != (len(indices),)
                            or not np.isfinite(forecasts).all() or (forecasts <= 0).any()):
                        raise ValueError("Model produced invalid holdout forecasts")
                    codes = dataset.symbol_codes[indices]
                    if (codes < 0).any() or (codes >= len(dataset.symbols)).any():
                        raise ValueError("Invalid stock identity in holdout dataset")
                    frame = pd.DataFrame({
                        "symbol": [dataset.symbols[code] for code in codes],
                        "decision_session": np.datetime_as_string(dataset.sessions[indices], unit="D"),
                        "target_session": np.datetime_as_string(dataset.target_sessions[indices], unit="D"),
                        "anchor_close": dataset.anchor_closes[indices],
                        "actual_close": dataset.target_closes[indices],
                        "forecast_close": forecasts,
                        "baseline_close": dataset.anchor_closes[indices],
                    })
                    _metrics(frame, "forecast_close")
                    frame["forecast_error"] = frame["forecast_close"] - frame["actual_close"]
                    frame["baseline_error"] = frame["baseline_close"] - frame["actual_close"]
                    temporary = part.with_suffix(".tmp")
                    frame.to_parquet(temporary, index=False)
                    _replace_with_retry(temporary, part)
                    _write_json(checkpoint, {"rows": len(frame), "sha256": _sha(part)})
                parts.append(frame)
                _write_json(output / "evaluation-state.json",
                            {"status": "running", "completed": min(start + len(indices), total), "total": total})
                boundary()
            results = pd.concat(parts, ignore_index=True)
            if len(results) != total:
                raise ValueError("Holdout result count does not match dataset")
            metrics = {
                "model": _metrics(results, "forecast_close"),
                "last_close_baseline": _metrics(results, "baseline_close"),
            }
            by_stock = {}
            for symbol, group in results.groupby("symbol", sort=True):
                by_stock[symbol] = {"model": _metrics(group, "forecast_close"),
                                    "last_close_baseline": _metrics(group, "baseline_close")}
            validate_dataset_content(dataset)
            if [_sha(path) for path in model_paths] != model_hashes:
                raise ValueError("Model artifacts changed during evaluation")
            csv_path = output / "holdout-predictions.csv"
            temporary_csv = output / "holdout-predictions.csv.tmp"
            results.to_csv(temporary_csv, index=False)
            _replace_with_retry(temporary_csv, csv_path)
            report = {
                "status": "complete", "research_only": True, "dataset_sha256": dataset.identity,
                "model_dir": str(model_dir), "holdout_start": summary["provenance"]["holdout_start"],
                "baseline": "Forecast the next session's close as the decision-session close.",
                "metrics": metrics, "stocks": by_stock,
                "stocks_without_holdout": sorted(set(dataset.symbols) - set(by_stock)),
                "predictions_sha256": _sha(csv_path),
                "limitations": [
                    "Sample-weighted price errors combine rupee prices across different stocks.",
                    "Directional accuracy counts an unchanged forecast as correct only for an unchanged close.",
                    "Evaluation uses proxy session calendars and research data with unverified rights, adjustments and timing.",
                    "Repeated model decisions based on this holdout can invalidate its use as an untouched test.",
                ],
            }
            _write_json(output / "evaluation-state.json", {"status": "complete", "completed": total, "total": total})
            _write_json(output / "holdout-summary.json", report)
            return report
