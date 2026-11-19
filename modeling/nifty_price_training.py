"""Pooled research price training with chronological splits and resumable controls."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from modeling.control import TrainingControl, _replace_with_retry
from modeling.data import impute_and_normalize_sequences, training_feature_statistics
from modeling.factor_research import _write_json
from modeling.nifty_preparation import TIMEFRAMES
from modeling.nifty_price_model import build_price_lstm, predict_hybrid_prices, selected_factors
from modeling.nifty_universe import Membership
from modeling.training import TrainingStopped


@dataclass(frozen=True)
class PriceTrainingConfig:
    sequence_length: int = 20
    tree_rounds: int = 100
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"
    checkpoint_batches: int = 10

    def __post_init__(self) -> None:
        if min(self.sequence_length, self.tree_rounds, self.epochs, self.batch_size,
               self.checkpoint_batches) < 1:
            raise ValueError("Sequence length, rounds, epochs and batch sizes must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if self.seed < 0 or self.seed + max(self.tree_rounds, self.epochs) > 2**31 - 1:
            raise ValueError("Seed and training length must fit the supported random-seed range")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Device must be cpu or cuda")


@dataclass(frozen=True)
class PriceDataset:
    features: np.ndarray
    targets: np.ndarray
    sequences: np.ndarray
    anchor_closes: np.ndarray
    target_closes: np.ndarray
    sessions: np.ndarray
    target_sessions: np.ndarray
    symbol_codes: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    holdout: np.ndarray
    feature_names: tuple[str, ...]
    symbols: tuple[str, ...]
    identity: str
    content_hash: str
    provenance_json: str


def _factor_name(key: tuple[str, str]) -> str:
    return "factor:" + json.dumps(key, ensure_ascii=True, separators=(",", ":"))


def align_price_features(frame: pd.DataFrame, keys: list[tuple[str, str]],
                         feature_names: tuple[str, ...]) -> np.ndarray:
    """Align a stock's selected factor columns to the pooled model's feature order."""
    expected = {f"factor_{index}_return" for index in range(len(keys))}
    actual = {name for name in frame.columns if name.startswith("factor_")}
    if expected != actual:
        raise ValueError("Factor columns do not match selected factors")
    renamed = frame.rename(columns={f"factor_{index}_return": _factor_name(key)
                                    for index, key in enumerate(keys)})
    if not set(_factor_name(key) for key in keys).issubset(feature_names):
        raise ValueError("Selected factors are not present in the pooled model")
    base = {f"{tf}_{suffix}" for tf in TIMEFRAMES for suffix in ("close_ratio", "body", "range", "partial")}
    if not base.issubset(renamed.columns) or not base.issubset(feature_names):
        raise ValueError("Pooled candle features are missing")
    values = renamed.reindex(columns=feature_names).apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Pooled features contain infinite values")
    return values


def _array_hash(arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_dataset_content(dataset: PriceDataset) -> None:
    arrays = (dataset.features, dataset.targets, dataset.sequences, dataset.anchor_closes,
              dataset.target_closes, dataset.sessions, dataset.target_sessions, dataset.symbol_codes,
              dataset.train, dataset.validation, dataset.holdout)
    if _array_hash(arrays) != dataset.content_hash:
        raise ValueError("Prepared dataset arrays changed after validation")


def prepare_price_dataset(
    examples: dict[str, pd.DataFrame], selection: dict,
    membership: tuple[Membership, ...], reference_sessions: list[date], *,
    validation_start: date, holdout_start: date, sequence_length: int,
    allow_no_factors: bool = False, control: TrainingControl | None = None,
) -> PriceDataset:
    """Pool per-stock causal examples, with purged validation and holdout boundaries."""
    if not examples or not validation_start < holdout_start or sequence_length < 1:
        raise ValueError("Invalid symbols, split dates or sequence length")
    if reference_sessions != sorted(set(reference_sessions)):
        raise ValueError("Reference sessions must be sorted and unique")
    base = tuple(f"{tf}_{suffix}" for tf in TIMEFRAMES
                 for suffix in ("close_ratio", "body", "range", "partial"))
    factor_keys: dict[str, list[tuple[str, str]]] = {}
    frames: dict[str, pd.DataFrame] = {}
    all_factors: set[tuple[str, str]] = set()
    session_indices = {day: index for index, day in enumerate(reference_sessions)}
    for symbol in sorted(examples):
        if control is not None:
            _boundary(control, lambda: None)
        frame = examples[symbol].copy().reset_index(drop=True)
        required = {"session", "target_session", "available_at", "label_available_at",
                    "anchor_close", "target_close", "target_log_return", *base}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{symbol} has missing or empty training examples")
        if (frame["session"].duplicated().any() or not frame["session"].is_monotonic_increasing
                or not all(isinstance(day, date) for day in frame["session"])
                or not all(isinstance(day, date) for day in frame["target_session"])):
            raise ValueError(f"{symbol} sessions must be sorted, unique dates")
        for column in ("available_at", "label_available_at"):
            if not isinstance(frame[column].dtype, pd.DatetimeTZDtype) or frame[column].isna().any():
                raise ValueError(f"{symbol} timestamps must be timezone-aware")
        if (frame["label_available_at"] <= frame["available_at"]).any():
            raise ValueError(f"{symbol} has labels available before the decision")
        decisions = (pd.to_datetime(frame["session"].astype(str)).dt.tz_localize("Asia/Kolkata")
                     + timedelta(days=1)).dt.tz_convert("UTC")
        if not (frame["available_at"].dt.tz_convert("UTC") == decisions).all():
            raise ValueError(f"{symbol} decision timestamps do not match their sessions")
        earliest_labels = (pd.to_datetime(frame["target_session"].astype(str)).dt.tz_localize("Asia/Kolkata")
                           + timedelta(days=1)).dt.tz_convert("UTC")
        if (frame["label_available_at"].dt.tz_convert("UTC") < earliest_labels).any():
            raise ValueError(f"{symbol} label availability precedes the target-session close")
        anchors = pd.to_numeric(frame["anchor_close"], errors="raise")
        if not np.isfinite(anchors).all() or (anchors <= 0).any():
            raise ValueError(f"{symbol} contains invalid anchor prices")
        closes = pd.to_numeric(frame["target_close"], errors="raise")
        returns = pd.to_numeric(frame["target_log_return"], errors="raise")
        if (not np.isfinite(closes).all() or (closes <= 0).any() or not np.isfinite(returns).all()
                or not np.allclose(np.log(closes / anchors), returns, rtol=1e-6, atol=1e-8)):
            raise ValueError(f"{symbol} target closes do not match log-return labels")
        intervals = [item for item in membership if item.symbol == symbol]
        for day, target in zip(frame["session"], frame["target_session"]):
            position = session_indices.get(day)
            if (position is None or position + 1 >= len(reference_sessions)
                    or reference_sessions[position + 1] != target
                    or not any(item.contains(day) and item.contains(target) for item in intervals)):
                raise ValueError(f"{symbol} has an invalid target session or membership")
        keys = selected_factors(selection, symbol, frame["session"].iloc[0])
        if not keys and not allow_no_factors:
            raise ValueError(f"{symbol} has no qualifying factors; stock-only research requires explicit opt-in")
        expected = {f"factor_{index}_return" for index in range(len(keys))}
        actual = {name for name in frame.columns if name.startswith("factor_")}
        if actual != expected:
            raise ValueError(f"{symbol} factors do not match its selected factors")
        factor_keys[symbol] = keys
        all_factors.update(keys)
        frames[symbol] = frame
    names = base + tuple(_factor_name(key) for key in sorted(all_factors))
    feature_frames = []
    targets = []
    anchor_closes, target_closes, session_dates, target_dates, symbol_codes = [], [], [], [], []
    sequence_rows = []
    train, validation, holdout = [], [], []
    offset = 0
    validation_boundary = (pd.Timestamp(validation_start).tz_localize("Asia/Kolkata")
                           + timedelta(days=1)).tz_convert("UTC")
    holdout_boundary = (pd.Timestamp(holdout_start).tz_localize("Asia/Kolkata")
                        + timedelta(days=1)).tz_convert("UTC")
    for symbol_code, (symbol, frame) in enumerate(frames.items()):
        if control is not None:
            _boundary(control, lambda: None)
        numeric = align_price_features(frame, factor_keys[symbol], names)
        labels = pd.to_numeric(frame["target_log_return"], errors="raise").to_numpy(dtype=np.float32)
        if np.isinf(numeric).any() or not np.isfinite(labels).all():
            raise ValueError(f"{symbol} has invalid features or labels")
        feature_frames.append(numeric)
        targets.append(labels)
        anchor_closes.append(frame["anchor_close"].to_numpy(dtype=np.float64))
        target_closes.append(frame["target_close"].to_numpy(dtype=np.float64))
        session_dates.append(np.asarray(frame["session"].tolist(), dtype="datetime64[D]"))
        target_dates.append(np.asarray(frame["target_session"].tolist(), dtype="datetime64[D]"))
        symbol_codes.append(np.full(len(frame), symbol_code, dtype=np.int32))
        intervals = [item for item in membership if item.symbol == symbol]
        sessions = frame["session"].tolist()
        target_days = frame["target_session"].tolist()
        for index in range(sequence_length - 1, len(frame)):
            first = index - sequence_length + 1
            days = sessions[first:index + 1]
            positions = [session_indices[day] for day in days]
            if positions != list(range(positions[0], positions[0] + sequence_length)):
                continue
            if not any(item.contains(days[0]) and item.contains(target_days[index]) for item in intervals):
                continue
            row = offset + index
            sequence_rows.append((row, list(range(offset + first, row + 1))))
            label_available = frame["label_available_at"].iloc[index]
            if (sessions[index] < validation_start and target_days[index] < validation_start
                    and label_available < validation_boundary):
                train.append(row)
            elif (validation_start <= sessions[index] < holdout_start and target_days[index] < holdout_start
                  and label_available < holdout_boundary):
                validation.append(row)
            elif sessions[index] >= holdout_start:
                holdout.append(row)
        offset += len(frame)
    if not train or not validation or not holdout:
        raise ValueError("Train, validation and untouched holdout must each contain complete sequences")
    values = np.concatenate(feature_frames)
    labels = np.concatenate(targets)
    anchors = np.concatenate(anchor_closes)
    closes = np.concatenate(target_closes)
    sessions_array = np.concatenate(session_dates)
    targets_array = np.concatenate(target_dates)
    codes = np.concatenate(symbol_codes)
    sequence_lookup = np.full((len(values), sequence_length), -1, dtype=np.int64)
    for row, indices in sequence_rows:
        sequence_lookup[row] = indices
    splits = [np.asarray(part, dtype=np.int64) for part in (train, validation, holdout)]
    content_hash = _array_hash((values, labels, sequence_lookup, anchors, closes,
                                sessions_array, targets_array, codes, *splits))
    digest = hashlib.sha256(content_hash.encode())
    provenance_json = json.dumps({"selection": selection, "validation_start": validation_start.isoformat(),
                                  "holdout_start": holdout_start.isoformat(), "allow_no_factors": allow_no_factors},
                                 sort_keys=True, allow_nan=False)
    digest.update(json.dumps({"names": names, "symbols": sorted(examples),
                              "provenance": provenance_json}, sort_keys=True).encode())
    for symbol, frame in frames.items():
        digest.update(symbol.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(
            frame[["session", "target_session", "available_at", "label_available_at", "anchor_close", "target_close"]],
            index=False).to_numpy().tobytes())
    digest.update(json.dumps([{"symbol": entry.symbol, "start": entry.valid_from.isoformat(),
                               "end": entry.valid_to.isoformat() if entry.valid_to else None}
                              for entry in membership], sort_keys=True).encode())
    digest.update(json.dumps([day.isoformat() for day in reference_sessions]).encode())
    return PriceDataset(values, labels, sequence_lookup, anchors, closes, sessions_array, targets_array,
                        codes, *splits, names, tuple(sorted(examples)),
                        digest.hexdigest(), content_hash, provenance_json)


def _atomic_torch_save(torch: Any, payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    _replace_with_retry(temporary, path)


def _boundary(control: TrainingControl, save: Any) -> None:
    if not control.checkpoint_boundary(save):
        raise TrainingStopped("Price training stopped after saving its checkpoint")


@contextmanager
def _price_lock(output: Path) -> Iterator[None]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("Price training requires psutil from requirements-training.txt") from exc
    path = output / "price-training.lock"
    owner = {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(owner, stream)
            break
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                process = psutil.Process(int(existing["pid"]))
                active = abs(process.create_time() - float(existing["create_time"])) < 1e-3
            except psutil.NoSuchProcess:
                active = False
            except psutil.AccessDenied:
                active = True
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot determine whether training lock is active: {path}") from exc
            if active:
                raise RuntimeError(f"Another process is using the price training run: {path}")
            if json.loads(path.read_text(encoding="utf-8")) != existing:
                raise RuntimeError(f"Training lock changed while being inspected: {path}")
            path.unlink()
    else:
        raise RuntimeError(f"Could not acquire price training lock: {path}")
    try:
        yield
    finally:
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) == owner:
            path.unlink()


def train_pooled_price_model(
    dataset: PriceDataset, output: Path, *, config: PriceTrainingConfig,
    research_only: bool, resume: bool = False, control: TrainingControl | None = None,
) -> dict:
    """Train only when explicitly called; no holdout labels are used for fitting or calibration."""
    if not research_only:
        raise ValueError("Data rights and source timing are unverified; explicit research_only=True is required")
    if dataset.sequences.shape[1] != config.sequence_length:
        raise ValueError("Dataset and training sequence lengths differ")
    validate_dataset_content(dataset)
    try:
        import torch
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("Price training requires requirements-training.txt") from exc
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use it")
    identity = {"version": 2, "dataset_sha256": dataset.identity, "config": asdict(config),
                "torch_version": str(torch.__version__), "xgboost_version": str(xgb.__version__),
                "numpy_version": np.__version__}
    if output.exists():
        if not resume:
            raise FileExistsError(f"Training output already exists: {output}")
    else:
        if resume:
            raise FileNotFoundError(f"Training output does not exist: {output}")
        output.mkdir(parents=True)
    with _price_lock(output):
        manifest = output / "identity.json"
        if resume:
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Training dataset, settings or dependency versions changed")
        else:
            _write_json(manifest, identity)
        cudnn_enabled = torch.backends.cudnn.enabled
        if config.device == "cuda" and os.name == "nt":
            torch.backends.cudnn.enabled = False
        try:
            return _train_locked(dataset, output, config, torch, xgb, resume=resume, control=control)
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled


def _train_locked(dataset: PriceDataset, output: Path, config: PriceTrainingConfig,
                  torch: Any, xgb: Any, *, resume: bool, control: TrainingControl | None = None) -> dict:
    own_control = control is None
    control = TrainingControl(output) if control is None else control
    if resume and own_control:
        control.request("continue")
    path = output / "checkpoint.pt"
    mean, scale, observed = training_feature_statistics(dataset.features[dataset.train])
    values = dataset.features.copy()
    values[:, ~observed] = np.nan
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = build_price_lstm(torch, len(dataset.feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    state: dict[str, Any] = {
        "stage": "tree", "round": 0, "tree": None, "best_tree": None,
        "best_tree_loss": None, "epoch": 0, "batch_offset": 0, "best_lstm": None,
        "best_lstm_loss": None,
    }

    def save() -> None:
        state["model"] = model.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["torch_rng"] = torch.get_rng_state()
        state["cuda_rng"] = torch.cuda.get_rng_state_all() if config.device == "cuda" else []
        _atomic_torch_save(torch, state, path)
        _write_json(output / "training-state.json", {
            "stage": state["stage"], "tree_round": state["round"], "tree_rounds": config.tree_rounds,
            "epoch": state["epoch"], "epochs": config.epochs, "batch_offset": state["batch_offset"],
        })

    if path.exists():
        if not resume:
            raise FileExistsError(f"Unexpected existing training checkpoint: {path}")
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if config.device == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
    else:
        save()
    _boundary(control, save)

    train_matrix = xgb.DMatrix(values[dataset.train], label=dataset.targets[dataset.train])
    val_matrix = xgb.DMatrix(values[dataset.validation], label=dataset.targets[dataset.validation])
    params = {"objective": "reg:squarederror", "tree_method": "hist", "device": config.device,
              "max_depth": 4, "eta": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "nthread": 2}
    booster = None
    if state["tree"] is not None:
        booster = xgb.Booster()
        booster.load_model(bytearray(state["tree"]))
    if state["stage"] == "tree":
        for iteration in range(state["round"], config.tree_rounds):
            booster = xgb.train({**params, "seed": config.seed + iteration}, train_matrix,
                                num_boost_round=1, xgb_model=booster, verbose_eval=False)
            prediction = booster.predict(val_matrix)
            loss = float(np.mean(np.square(prediction - dataset.targets[dataset.validation])))
            if not math.isfinite(loss):
                raise ValueError("XGBoost validation loss is not finite")
            state["tree"] = bytes(booster.save_raw(raw_format="ubj"))
            state["round"] = iteration + 1
            if state["best_tree_loss"] is None or loss < state["best_tree_loss"]:
                state["best_tree_loss"] = loss
                state["best_tree"] = state["tree"]
            save()
            _boundary(control, save)
        state["stage"] = "lstm"
        save()
    if state["best_tree"] is None:
        raise ValueError("Missing best XGBoost checkpoint")

    def sequence_batch(indices: np.ndarray) -> Any:
        raw = values[dataset.sequences[indices]]
        normalized = impute_and_normalize_sequences(raw, mean=mean, scale=scale, observed=observed)
        return torch.from_numpy(normalized).to(device)

    def lstm_predictions(indices: np.ndarray) -> np.ndarray:
        model.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(indices), config.batch_size):
                predictions.append(model(sequence_batch(indices[start:start + config.batch_size])).cpu().numpy())
        result = np.concatenate(predictions)
        if not np.isfinite(result).all():
            raise ValueError("LSTM predictions are not finite")
        return result

    if state["stage"] == "lstm":
        for epoch in range(state["epoch"], config.epochs):
            model.train()
            permutation = np.random.default_rng(config.seed + epoch).permutation(dataset.train)
            for offset in range(state["batch_offset"], len(permutation), config.batch_size):
                indices = permutation[offset:offset + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(sequence_batch(indices))
                target = torch.from_numpy(dataset.targets[indices]).to(device)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("LSTM training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                state["batch_offset"] = offset + len(indices)
                if (offset // config.batch_size + 1) % config.checkpoint_batches == 0:
                    save()
                _boundary(control, save)
            prediction = lstm_predictions(dataset.validation)
            loss = float(np.mean(np.square(prediction - dataset.targets[dataset.validation])))
            if not math.isfinite(loss):
                raise ValueError("LSTM validation loss is not finite")
            if state["best_lstm_loss"] is None or loss < state["best_lstm_loss"]:
                state["best_lstm_loss"] = loss
                state["best_lstm"] = {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()}
            state["epoch"] = epoch + 1
            state["batch_offset"] = 0
            save()
            _boundary(control, save)
        state["stage"] = "finalize"
        save()
    if state["best_lstm"] is None:
        raise ValueError("Missing best LSTM checkpoint")
    model.load_state_dict(state["best_lstm"])
    booster = xgb.Booster()
    booster.load_model(bytearray(state["best_tree"]))
    tree_pred = booster.predict(val_matrix)
    lstm_pred = lstm_predictions(dataset.validation)
    delta = tree_pred - lstm_pred
    denominator = float(np.dot(delta, delta))
    weight = 0.5 if denominator == 0 else float(np.clip(
        np.dot(delta, dataset.targets[dataset.validation] - lstm_pred) / denominator, 0, 1))
    if not math.isfinite(weight):
        raise ValueError("Hybrid calibration is not finite")
    _boundary(control, save)
    tree_path = output / "price-tree.ubj"
    temp_tree = output / "price-tree.tmp.ubj"
    temp_tree.write_bytes(state["best_tree"])
    _replace_with_retry(temp_tree, tree_path)
    _atomic_torch_save(torch, {"model": state["best_lstm"], "feature_names": list(dataset.feature_names),
                               "mean": mean.tolist(), "scale": scale.tolist(), "observed": observed.tolist(),
                               "sequence_length": config.sequence_length}, output / "price-lstm.pt")
    lstm_path = output / "price-lstm.pt"
    summary = {"status": "complete", "research_only": True, "dataset_sha256": dataset.identity,
               "symbols": list(dataset.symbols), "feature_names": list(dataset.feature_names),
               "provenance": json.loads(dataset.provenance_json),
               "training_examples": len(dataset.train), "validation_examples": len(dataset.validation),
               "reserved_holdout_examples": len(dataset.holdout), "tree_weight": weight,
               "tree_validation_mse": state["best_tree_loss"], "lstm_validation_mse": state["best_lstm_loss"],
               "tree_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest(),
               "lstm_sha256": hashlib.sha256(lstm_path.read_bytes()).hexdigest(),
               "warning": "Not evaluated on holdout; source rights, price adjustments and timing remain unverified."}
    _write_json(output / "training-summary.json", summary)
    state["stage"] = "complete"
    save()
    return summary


def load_price_predictor(output: Path, *, device: str = "cpu") -> Any:
    """Load verified, trusted local artifacts; return a feature-order-checked predictor."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    import torch
    import xgboost as xgb
    summary = json.loads((output / "training-summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("research_only") is not True:
        raise ValueError("Price training is not complete")
    tree_path, lstm_path = output / "price-tree.ubj", output / "price-lstm.pt"
    for path, key in ((tree_path, "tree_sha256"), (lstm_path, "lstm_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key]:
            raise ValueError(f"Corrupt price model artifact: {path}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use it")
    saved = torch.load(lstm_path, map_location=device, weights_only=True)
    if saved["feature_names"] != summary["feature_names"]:
        raise ValueError("LSTM feature order does not match training summary")
    mean = np.asarray(saved["mean"], dtype=np.float32)
    scale = np.asarray(saved["scale"], dtype=np.float32)
    observed = np.asarray(saved["observed"], dtype=bool)
    if (mean.shape != (len(saved["feature_names"]),) or scale.shape != mean.shape
            or observed.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid LSTM normalization statistics")
    network = build_price_lstm(torch, len(mean)).to(device)
    network.load_state_dict(saved["model"])
    booster = xgb.Booster()
    booster.load_model(tree_path)
    booster.set_param({"device": device})

    class Predictor:
        def predict(self, tabular: np.ndarray, sequences: np.ndarray, anchor_close: np.ndarray,
                    *, feature_names: list[str]) -> np.ndarray:
            if feature_names != saved["feature_names"]:
                raise ValueError("Prediction feature order differs from training")
            tabular = np.asarray(tabular, dtype=np.float32).copy()
            sequences = np.asarray(sequences, dtype=np.float32)
            if (tabular.ndim != 2 or tabular.shape[1] != len(mean) or sequences.ndim != 3
                    or sequences.shape[1:] != (saved["sequence_length"], len(mean))
                    or np.isinf(tabular).any() or np.isinf(sequences).any()):
                raise ValueError("Prediction features or sequences have an invalid shape")
            tabular[:, ~observed] = np.nan
            normalized = impute_and_normalize_sequences(sequences, mean=mean, scale=scale, observed=observed)

            class TreeAdapter:
                def predict(self, values: np.ndarray) -> np.ndarray:
                    return booster.predict(xgb.DMatrix(values))

            return predict_hybrid_prices(TreeAdapter(), network, torch, tabular, normalized,
                                         anchor_close, tree_weight=float(summary["tree_weight"]), device=device)

    return Predictor()
