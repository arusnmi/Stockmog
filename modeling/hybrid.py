from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .data import _numeric_feature_names, impute_and_normalize_sequences
from .training import (
    _build_lstm,
    _fixed_class_probabilities,
    _probabilities_from_logits,
)


def _regime(row: dict[str, Any], thresholds: dict[str, Any]) -> str:
    volatility_feature = thresholds.get("volatility_feature")
    volatility = row.get(volatility_feature) if volatility_feature else None
    low = thresholds.get("atr_pct_low_threshold")
    high = thresholds.get("atr_pct_high_threshold")
    if volatility is not None and high is not None and volatility >= high:
        return "high_volatility"
    if volatility is not None and low is not None and volatility <= low:
        return "low_volatility"
    trend_feature = thresholds.get("adx_feature")
    trend_value = row.get(trend_feature) if trend_feature else None
    if trend_value is not None and trend_value >= thresholds.get(
        "adx_trend_threshold", 25.0
    ):
        return "trend"
    return "range"


def predict_hybrid_signals(
    features: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    model_dir: str | Path,
    minimum_win_probability: float = 0.45,
) -> pl.DataFrame:
    """Score strategy candidates with the calibrated XGBoost/LSTM hybrid."""
    if not 0.0 <= minimum_win_probability <= 1.0:
        raise ValueError("minimum_win_probability must be between zero and one")
    model_path = Path(model_dir)
    metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_path / "hybrid.json").read_text(encoding="utf-8"))
    if calibration.get("format_version") != 1:
        raise ValueError(f"Unsupported hybrid model format in {model_path / 'hybrid.json'}")
    model_weights = calibration.get("model_weights")
    if not isinstance(model_weights, dict) or not model_weights:
        raise ValueError(f"Hybrid calibration in {model_path} has no model weights")
    if any(float(weight) > 0 for weight in model_weights.values()) and (
        "lstm" not in model_weights
    ):
        raise ValueError("Hybrid model lacks the LSTM return/uncertainty output")
    if signals.height and (
        metadata.get("style") not in signals["style"].unique().to_list()
        or metadata.get("strategy") not in signals["strategy"].unique().to_list()
    ):
        raise ValueError(
            "The trained hybrid style/strategy does not match the candidate signals"
        )
    feature_names = tuple(metadata["feature_names"])
    available = _numeric_feature_names(features)
    missing = set(feature_names) - set(available)
    if missing:
        raise ValueError(
            "Current feature set is missing model columns: "
            + ", ".join(sorted(missing))
        )
    if signals.is_empty():
        return signals.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("loss_probability"),
            pl.lit(None, dtype=pl.Float64).alias("neutral_probability"),
            pl.lit(None, dtype=pl.Float64).alias("win_probability"),
            pl.lit(None, dtype=pl.Float64).alias("expected_return_r"),
            pl.lit(None, dtype=pl.Float64).alias("return_uncertainty_r"),
            pl.lit(None, dtype=pl.Float64).alias("uncertainty"),
            pl.lit(None, dtype=pl.String).alias("market_regime"),
            pl.lit(False).alias("accepted"),
        )

    feature_matrix = features.select(feature_names).to_numpy().astype(
        np.float32,
        copy=False,
    )
    feature_rows = features.select("timestamp", *feature_names).iter_rows(named=True)
    rows_by_timestamp = {row["timestamp"]: row for row in feature_rows}
    timestamp_indices = {
        timestamp: index for index, timestamp in enumerate(features["timestamp"])
    }
    mean = np.asarray(metadata["mean"], dtype=np.float32)
    scale = np.asarray(metadata["scale"], dtype=np.float32)
    scale[scale < 1e-8] = 1.0
    normalized = np.where(
        np.isfinite(feature_matrix),
        (feature_matrix - mean[None, :]) / scale[None, :],
        0.0,
    )

    sequence_length = int(metadata["sequence_length"])
    candidate_rows = list(signals.iter_rows(named=True))
    usable = []
    sequences = []
    for candidate_index, candidate in enumerate(candidate_rows):
        index = timestamp_indices.get(candidate["timestamp"])
        if index is None or index + 1 < sequence_length:
            continue
        if not np.isfinite(feature_matrix[index]).all():
            continue
        sequence = feature_matrix[index + 1 - sequence_length : index + 1]
        usable.append(candidate_index)
        sequences.append(sequence)
    if not usable:
        return signals.head(0).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("loss_probability"),
            pl.lit(None, dtype=pl.Float64).alias("neutral_probability"),
            pl.lit(None, dtype=pl.Float64).alias("win_probability"),
            pl.lit(None, dtype=pl.Float64).alias("expected_return_r"),
            pl.lit(None, dtype=pl.Float64).alias("return_uncertainty_r"),
            pl.lit(None, dtype=pl.Float64).alias("uncertainty"),
            pl.lit(None, dtype=pl.String).alias("market_regime"),
            pl.lit(False).alias("accepted"),
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for hybrid prediction. Install requirements-training.txt."
        ) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_features = feature_matrix[
        [timestamp_indices[candidate_rows[index]["timestamp"]] for index in usable]
    ]
    predicted: dict[str, np.ndarray] = {}
    for model_name, weight in model_weights.items():
        if float(weight) <= 0 or model_name == "lstm":
            continue
        if model_name == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError(
                    "XGBoost is required for hybrid prediction. Install "
                    "requirements-training.txt."
                ) from exc
            booster = xgb.Booster()
            booster.load_model(model_path / "xgboost.ubj")
            matrix = xgb.DMatrix(
                candidate_features,
                feature_names=list(feature_names),
            )
            predicted[model_name] = booster.predict(matrix).reshape(-1, 3)
        else:
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError(
                    "Scikit-learn/joblib is required for hybrid prediction. "
                    "Install requirements-training.txt."
                ) from exc
            estimator = joblib.load(model_path / f"{model_name}.joblib")
            if model_name == "logistic_regression":
                predicted[model_name] = _fixed_class_probabilities(
                    estimator[-1],
                    estimator[:-1].transform(candidate_features),
                )
            else:
                predicted[model_name] = _fixed_class_probabilities(
                    estimator,
                    candidate_features,
                )

    if float(model_weights.get("lstm", 0.0)) > 0:
        checkpoint = torch.load(
            model_path / "lstm.pt",
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("feature_names") != list(feature_names):
            raise ValueError("LSTM feature order does not match model metadata")
        if checkpoint.get("style") != metadata.get("style") or checkpoint.get(
            "strategy"
        ) != metadata.get("strategy"):
            raise ValueError("LSTM style/strategy does not match model metadata")
        if not np.array_equal(
            np.asarray(checkpoint.get("mean"), dtype=np.float32),
            mean,
        ) or not np.array_equal(
            np.asarray(checkpoint.get("scale"), dtype=np.float32),
            np.asarray(metadata["scale"], dtype=np.float32),
        ):
            raise ValueError("LSTM scaling parameters do not match model metadata")
        network = _build_lstm(torch, len(feature_names)).to(device)
        network.load_state_dict(checkpoint["model_state"])
        network.eval()
        with torch.no_grad():
            normalized_sequences = impute_and_normalize_sequences(
                np.stack(sequences),
                mean=mean,
                scale=scale,
            )
            lstm_outputs = network(
                torch.from_numpy(normalized_sequences).to(device)
            ).cpu().numpy()
        predicted["lstm"] = _probabilities_from_logits(lstm_outputs[:, :3])
        expected_returns_r = np.clip(lstm_outputs[:, 3], -5.0, 5.0)
        return_uncertainty_r = np.exp(
            0.5 * np.clip(lstm_outputs[:, 4], -5.0, 3.0)
        )
    else:
        raise ValueError("Hybrid prediction requires an LSTM model for return estimates")

    blended = sum(
        float(weight) * predicted[model_name]
        for model_name, weight in model_weights.items()
        if model_name in predicted
    )
    logits = np.log(np.clip(blended, 1e-12, 1.0))
    probabilities = _probabilities_from_logits(
        logits,
        float(calibration["temperature"]),
    )
    scored: list[dict[str, Any] | None] = [None] * len(candidate_rows)
    thresholds = calibration.get("regime_thresholds", {})
    for row_index, candidate_index in enumerate(usable):
        row = dict(candidate_rows[candidate_index])
        row_probabilities = probabilities[row_index]
        entropy = -float(
            np.sum(row_probabilities * np.log(np.clip(row_probabilities, 1e-12, 1.0)))
            / np.log(3.0)
        )
        return_uncertainty = float(return_uncertainty_r[row_index])
        combined_uncertainty = max(
            entropy,
            return_uncertainty / (1.0 + return_uncertainty),
        )
        row.update(
            {
                "loss_probability": float(row_probabilities[0]),
                "neutral_probability": float(row_probabilities[1]),
                "win_probability": float(row_probabilities[2]),
                "expected_return_r": float(expected_returns_r[row_index]),
                "return_uncertainty_r": return_uncertainty,
                "uncertainty": combined_uncertainty,
                "market_regime": _regime(
                    rows_by_timestamp[row["timestamp"]],
                    thresholds,
                ),
                "accepted": bool(
                    row_probabilities[2] >= minimum_win_probability
                    and expected_returns_r[row_index] > 0
                    and combined_uncertainty <= 0.95
                ),
            }
        )
        scored[candidate_index] = row

    output_rows = [
        row
        for row in scored
        if row is not None and row["win_probability"] >= minimum_win_probability
        and row["expected_return_r"] > 0
        and row["uncertainty"] <= 0.95
    ]
    schema = {
        **signals.schema,
        "loss_probability": pl.Float64,
        "neutral_probability": pl.Float64,
        "win_probability": pl.Float64,
        "expected_return_r": pl.Float64,
        "return_uncertainty_r": pl.Float64,
        "uncertainty": pl.Float64,
        "market_regime": pl.String,
        "accepted": pl.Boolean,
    }
    return pl.DataFrame(output_rows, schema=schema) if output_rows else pl.DataFrame(schema=schema)
